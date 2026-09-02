"""Tests for the Phase-4.3 binary CVD evaluation harness.

WHY THESE ARE MOSTLY NEGATIVE CONTROLS
--------------------------------------
This project has shipped silently-wrong evaluation code twice (read the
docstring of `eval_binary_comparison/embedding_io.py`: a `startswith("e0")`
filter that quietly threw away 3,096 of 4,096 dimensions, twice). Both times a
happy-path check passed, because a happy-path check cannot tell a correct
pipeline from one that is wrong in a direction that looks plausible.

So the suite is built around the question "what would still pass if this code
were broken?":

  * a PERFECTLY SEPARATING stub must give AUC 1.0            (it is wired up)
  * a PURE NOISE stub must give AUC ~ 0.5                    (no free signal)
  * an INVERTED stub must give AUC ~ 0.0, not 1.0            (not abs()-ed,
    not symmetric, label polarity is not silently flipped)
  * feeding a deliberately contaminated population must RAISE (the leak guard
    is load-bearing, not decorative)
  * feeding a single-class series must RAISE                 (the standing
    batch-signature guard is load-bearing)
  * multi-token log-prob summation must equal a value computed by hand, from
    a different expression, at the right sequence positions
  * a stub whose score is a pure function of series_id — ZERO per-sample
    signal — must show a HIGH pooled AUC and a within-series AUC of exactly
    0.5. This is the proof that the batch-shortcut diagnostic detects the
    shortcut Phase 0 found is available (no Stage-2 training series contains
    both classes, so series_id predicts the training label perfectly). It has
    already earned its keep: it caught two real bugs in that diagnostic —
    float-noise amplification in z-score centering, and the prevalence bias of
    rank-centered pooling.

Run:
    python3 eval_binary_comparison/tests/test_binary_cvd_eval.py -v
    python3 -m pytest eval_binary_comparison/tests/ -v      # if pytest exists
"""

from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from eval_binary_comparison.run_binary_cvd_eval import (  # noqa: E402
    PROMPT_VARIANTS,
    AnswerScorer,
    AnswerScores,
    LeakageError,
    PromptVariant,
    SingleClassSeriesError,
    TinyLlavaScorer,
    aggregate_per_series,
    answer_logprobs_from_logits,
    assert_no_stage2_overlap,
    assert_series_have_both_classes,
    binary_metrics,
    encoder_linear_probe_baseline,
    series_centered_pooled_auc,
    series_centered_scores,
    stratified_pairwise_auc,
    series_shortcut_diagnostics,
    within_series_auc,
    forced_choice_probabilities,
    load_holdout_population,
    main,
    per_series_metrics,
    prior_only_baseline,
    split_answer_token_ids,
    tissue_only_baseline,
)

ENCODED_DIM = 515


# --------------------------------------------------------------------------
# Stub scorers. Each returns answer log-probs with a KNOWN relationship to the
# label, so the metric that comes out the far end has a known correct value.
# --------------------------------------------------------------------------

class _StubScorer(AnswerScorer):
    """Base: knows the labels, in geo_accession order, and fabricates log-probs.

    n_tokens is (1, 1) so the `sum` and `mean` scorings coincide and a test can
    assert on both without the length normalization changing the answer.
    """

    is_debug = True

    def __init__(self, order: list, y: np.ndarray, seed: int = 0):
        self.index = {g: i for i, g in enumerate(order)}
        self.y = np.asarray(y).astype(int)
        self.rng = np.random.default_rng(seed)
        self.calls = 0

    def _margin(self, n: int) -> np.ndarray:
        raise NotImplementedError

    def score(self, embeddings, question, answers):
        self.calls += 1
        n = len(embeddings)
        margin = self._margin(n)
        lp = np.zeros((n, len(answers)), dtype=np.float64)
        lp[:, 0] = margin          # affirmative
        lp[:, 1] = 0.0             # negative
        return AnswerScores(lp, np.ones(len(answers), dtype=int))


class PerfectScorer(_StubScorer):
    name = "stub:perfect"

    def _margin(self, n):
        return np.where(self.y[:n] == 1, 3.0, -3.0)


class NoiseScorer(_StubScorer):
    name = "stub:noise"

    def _margin(self, n):
        return self.rng.normal(0.0, 1.0, size=n)


class InvertedScorer(_StubScorer):
    name = "stub:inverted"

    def _margin(self, n):
        return np.where(self.y[:n] == 1, -3.0, 3.0)


class LengthBiasedScorer(_StubScorer):
    """Both answers equally likely PER TOKEN, but the affirmative is 3 tokens
    and the negative 1. The `sum` scoring must then prefer the negative for
    every sample while the `mean` scoring must be exactly tied."""

    name = "stub:length-biased"

    def score(self, embeddings, question, answers):
        n = len(embeddings)
        lp = np.zeros((n, 2), dtype=np.float64)
        lp[:, 0] = -3.0            # 3 tokens at -1.0 each
        lp[:, 1] = -1.0            # 1 token  at -1.0
        return AnswerScores(lp, np.array([3, 1]))


# --------------------------------------------------------------------------
# Synthetic fixture: a miniature holdout that satisfies every real invariant.
# --------------------------------------------------------------------------

def build_fixture(tmp: Path, n_series: int = 24, per_series: int = 40,
                  contaminate: int = 0, single_class_series: int = 0) -> dict:
    """Write labels parquet + holdout json + stage2 json + encoded .npy dir.

    `contaminate` puts that many holdout accessions into stage2_train.json.
    `single_class_series` makes that many series all-positive.
    """
    import pandas as pd

    rows = []
    k = 0
    for s in range(n_series):
        sid = f"GSE{90000 + s}"
        all_pos = s < single_class_series
        for j in range(per_series):
            pos = True if all_pos else (j % 2 == 0)
            # sample_index is deliberately out of the ARCHS4 H5's range. In a
            # real run it indexes that file's sample columns, and the tissue
            # baseline reads it; a fabricated in-range index would make the
            # baseline silently "succeed" on synthetic data (and would make
            # this test pass or fail depending on whether the 40 GB H5 happens
            # to be on the machine).
            rows.append({"sample_index": 10 ** 9 + k,
                         "geo_accession": f"GSM{500000 + k}",
                         "series_id": sid,
                         "cvd_subtype": "hypertension" if pos else None,
                         "is_positive": pos,
                         "is_neg_hard": not pos,
                         "is_neg_whole_corpus": False})
            k += 1
    df = pd.DataFrame(rows)

    labels = tmp / "labels.parquet"
    df.to_parquet(labels)

    holdout = tmp / "holdout.json"
    holdout.write_text(json.dumps({
        "n_series": n_series,
        "holdout_series": sorted(df["series_id"].unique().tolist())}))

    # Stage-2 training corpus: disjoint by construction, then deliberately
    # polluted with `contaminate` real holdout accessions.
    train_images = [f"GSM{900000 + i}.npy" for i in range(50)]
    train_images += [f"{g}.npy" for g in df["geo_accession"][:contaminate]]
    stage2 = tmp / "stage2_train.json"
    stage2.write_text(json.dumps(
        [{"image": im, "conversations": [{"from": "human", "value": "<image>\nq"},
                                         {"from": "gpt", "value": "a"}]}
         for im in train_images]))

    enc = tmp / "encoded"
    enc.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    for g in df["geo_accession"]:
        np.save(enc / f"{g}.npy", rng.standard_normal(ENCODED_DIM).astype(np.float32))

    return {"labels": labels, "holdout": holdout, "stage2": stage2,
            "encoded": enc, "frame": df}


def argv_for(fx: dict, tmp: Path, tag: str, extra=()) -> list:
    return [
        "--labels", str(fx["labels"]),
        "--holdout", str(fx["holdout"]),
        "--stage2-json", str(fx["stage2"]),
        "--encoded-dir", str(fx["encoded"]),
        "--encoder-baseline-json", str(tmp / "no_such_probe.json"),
        "--tables-dir", str(tmp / "tables"),
        "--plots-dir", str(tmp / "plots"),
        "--out-tag", tag,
    ] + list(extra)


class _FixtureCase(unittest.TestCase):
    """Shares one fixture across the class — building it costs ~1k file writes."""

    n_series = 24
    per_series = 40
    contaminate = 0
    single_class_series = 0

    @classmethod
    def setUpClass(cls):
        import tempfile
        cls._tmpdir = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls._tmpdir.name)
        cls.fx = build_fixture(cls.tmp, cls.n_series, cls.per_series,
                               cls.contaminate, cls.single_class_series)
        cls.df = cls.fx["frame"]
        cls.y = cls.df["is_positive"].astype(int).to_numpy()
        cls.order = cls.df["geo_accession"].tolist()

    @classmethod
    def tearDownClass(cls):
        cls._tmpdir.cleanup()

    def run_main(self, scorer, tag, extra=()):
        rc = main(argv_for(self.fx, self.tmp, tag, extra), scorer=scorer)
        self.assertEqual(rc, 0)
        return json.loads((self.tmp / "tables" / f"{tag}.json").read_text())


# --------------------------------------------------------------------------
# (a) (b) (c) — end-to-end through main(), with known-answer stubs
# --------------------------------------------------------------------------

class TestEndToEndKnownAnswer(_FixtureCase):

    def _all_aucs(self, payload, scoring="sum"):
        return [r["scorings"][scoring]["pooled"]["roc_auc"]
                for r in payload["runs"]]

    def test_a_perfect_scorer_gives_auc_one(self):
        p = self.run_main(PerfectScorer(self.order, self.y), "perfect")
        for scoring in ("sum", "mean"):
            for auc in self._all_aucs(p, scoring):
                self.assertEqual(auc, 1.0)
        for r in p["runs"]:
            pooled = r["scorings"]["sum"]["pooled"]
            self.assertEqual(pooled["accuracy"], 1.0)
            self.assertEqual(pooled["pr_auc"], 1.0)
            # Per-series must be perfect too, not just the pooled number.
            per = r["scorings"]["sum"]["per_series_summary"]
            self.assertEqual(per["n_series"], self.n_series)
            self.assertEqual(per["roc_auc_series_mean"], 1.0)

    def test_b_noise_scorer_gives_auc_half(self):
        p = self.run_main(NoiseScorer(self.order, self.y, seed=7), "noise")
        n = len(self.y)
        # SE of AUC under H0 is ~ sqrt(1/(12 * n_pos * n_neg) * (n+1)); for a
        # 960-sample balanced set that is ~0.019, so 5 SE is ~0.10.
        for auc in self._all_aucs(p):
            self.assertAlmostEqual(auc, 0.5, delta=0.10)
        self.assertGreater(n, 500, "noise tolerance assumes a large fixture")

    def test_c_inverted_scorer_gives_auc_zero(self):
        """The one that proves the metric is not abs()-ed or polarity-agnostic.

        A pipeline that ranked by |score|, or that recovered the label from the
        wrong column, or that sorted descending where it should sort ascending,
        would return 1.0 here and look like a triumph.
        """
        p = self.run_main(InvertedScorer(self.order, self.y), "inverted")
        for scoring in ("sum", "mean"):
            for auc in self._all_aucs(p, scoring):
                self.assertEqual(auc, 0.0)
        for r in p["runs"]:
            self.assertEqual(r["scorings"]["sum"]["pooled"]["accuracy"], 0.0)

    def test_outputs_are_written_and_self_describing(self):
        p = self.run_main(PerfectScorer(self.order, self.y), "outputs")
        tables = self.tmp / "tables"
        self.assertTrue((tables / "outputs.json").exists())
        self.assertTrue((tables / "outputs.csv").exists())
        self.assertTrue((tables / "outputs_predictions.csv").exists())
        self.assertTrue((self.tmp / "plots" / "outputs.png").exists())

        # A stub run must never be mistakable for a result.
        self.assertTrue(p["debug_scorer_not_a_real_result"])

        self.assertEqual(p["population"]["n"], len(self.y))
        self.assertEqual(p["population"]["n_positive"], int(self.y.sum()))
        self.assertEqual(p["population"]["n_series"], self.n_series)
        self.assertEqual(p["guards"]["stage2_overlap"]["overlap"], 0)
        self.assertTrue(
            p["guards"]["series_class_balance"]["every_series_has_both_classes"])

        header = (tables / "outputs.csv").read_text().splitlines()
        self.assertIn("roc_auc", header[0])
        self.assertTrue(any(",prior_only," in ln for ln in header))

    def test_every_variant_and_both_orders_are_run(self):
        p = self.run_main(PerfectScorer(self.order, self.y), "variants")
        got = {(r["variant"], r["order"]) for r in p["runs"]}
        expected = {(v.name, o) for v in PROMPT_VARIANTS.values()
                    for o in v.orders()}
        self.assertEqual(got, expected)
        self.assertGreaterEqual(len({v for v, _ in got}), 2,
                                "at least two phrasings are required")
        # Order control present for every variant that lists its options.
        for v in PROMPT_VARIANTS.values():
            if v.presents_options:
                self.assertIn(v.name, p["order_bias"]["sum"]["per_variant"])
                bias = p["order_bias"]["sum"]["per_variant"][v.name]
                self.assertIn("roc_auc_gap", bias)
                self.assertIn("mean_p_yes_shift", bias)

    def test_phrasing_sensitivity_is_reported(self):
        p = self.run_main(PerfectScorer(self.order, self.y), "phrasing")
        ps = p["phrasing_sensitivity"]["sum"]
        self.assertEqual(ps["spread"], 0.0)      # identical stub -> no spread
        self.assertEqual(len(ps["per_run_roc_auc"]), len(p["runs"]))

    def test_baselines_are_reported_alongside(self):
        p = self.run_main(PerfectScorer(self.order, self.y), "baselines")
        b = p["baselines"]
        self.assertAlmostEqual(b["prior_only"]["accuracy"], 0.5, places=6)
        self.assertEqual(b["prior_only"]["roc_auc"], 0.5)
        # Encoder baseline file deliberately absent -> reported unavailable,
        # never silently substituted.
        self.assertFalse(b["encoder_linear_probe"]["available"])
        self.assertIn("reason", b["encoder_linear_probe"])
        # No tissue label is resolvable for these fabricated sample indices, so
        # the baseline must say so rather than inventing one.
        self.assertFalse(b["tissue_only"]["available"])
        self.assertIn("UNMEASURED", b["tissue_only"]["note"])


# --------------------------------------------------------------------------
# (d) leak assertion, (e) single-class assertion
# --------------------------------------------------------------------------

class TestLeakGuard(_FixtureCase):
    n_series = 6
    per_series = 10
    contaminate = 3

    def test_d_leak_assertion_fires_on_contaminated_population(self):
        with self.assertRaises(LeakageError) as ctx:
            load_holdout_population(self.fx["labels"], self.fx["holdout"],
                                    self.fx["stage2"])
        msg = str(ctx.exception)
        self.assertIn("3 of", msg)
        self.assertIn("contaminated", msg)

    def test_d_leak_assertion_fires_through_main(self):
        """The guard must be reachable from the entry point, not only from the
        helper — a guard that main() bypasses is not a guard."""
        with self.assertRaises(LeakageError):
            main(argv_for(self.fx, self.tmp, "leak"),
                 scorer=PerfectScorer(self.order, self.y))

    def test_leak_guard_matches_on_the_image_filename(self):
        with self.assertRaises(LeakageError):
            assert_no_stage2_overlap(["GSMdeadbeef"], self._one_image_json())
        # And passes when the accession is genuinely absent.
        stats = assert_no_stage2_overlap(["GSMnotinthere"], self._one_image_json())
        self.assertEqual(stats["overlap"], 0)

    def _one_image_json(self):
        p = self.tmp / "one_image.json"
        p.write_text(json.dumps([{"image": "GSMdeadbeef.npy"}]))
        return p


class TestSingleClassSeriesGuard(_FixtureCase):
    n_series = 6
    per_series = 10
    single_class_series = 2

    def test_e_single_class_series_assertion_fires(self):
        with self.assertRaises(SingleClassSeriesError) as ctx:
            load_holdout_population(self.fx["labels"], self.fx["holdout"],
                                    self.fx["stage2"])
        msg = str(ctx.exception)
        self.assertIn("2 of 6", msg)
        self.assertIn("batch-", msg)

    def test_e_single_class_series_assertion_fires_through_main(self):
        with self.assertRaises(SingleClassSeriesError):
            main(argv_for(self.fx, self.tmp, "singleclass"),
                 scorer=PerfectScorer(self.order, self.y))

    def test_e_per_series_breakdown_refuses_single_class_series(self):
        """The breakdown has its own guard: even if a population somehow got
        past the load-time check, the per-series table must not quietly emit a
        row for a series it cannot score."""
        series = np.array(["A"] * 4 + ["B"] * 4)
        y = np.array([0, 1, 0, 1, 1, 1, 1, 1])       # B is all-positive
        p = np.linspace(0, 1, 8)
        with self.assertRaises(SingleClassSeriesError):
            per_series_metrics(y, p, series)

    def test_guard_passes_on_a_properly_mixed_population(self):
        series = np.array(["A", "A", "B", "B"])
        y = np.array([0, 1, 1, 0])
        out = assert_series_have_both_classes(series, y)
        self.assertTrue(out["every_series_has_both_classes"])
        self.assertEqual(out["n_series"], 2)


# --------------------------------------------------------------------------
# (f) multi-token log-prob summation, against a hand-computed value
# --------------------------------------------------------------------------

class TestMultiTokenLogProb(unittest.TestCase):

    def test_f_sum_matches_hand_computation(self):
        """Two-token answer, hand-computed from a different expression.

        Sequence length 5, vocab 4, answer = [2, 0] occupying positions 3 and 4.
        The logits that PREDICT them therefore sit at positions 2 and 3.
        Positions 0, 1 and 4 are filled with values that would produce a very
        different total if the alignment were off by one in either direction.
        """
        import torch

        logits = torch.zeros(1, 5, 4)
        logits[0, 0] = torch.tensor([9.0, -9.0, -9.0, -9.0])   # decoy
        logits[0, 1] = torch.tensor([-9.0, 9.0, -9.0, -9.0])   # decoy
        logits[0, 2] = torch.tensor([0.0, 1.0, 2.0, 3.0])      # predicts tok 2
        logits[0, 3] = torch.tensor([1.0, 0.5, 0.25, 0.125])   # predicts tok 0
        logits[0, 4] = torch.tensor([-9.0, -9.0, 9.0, -9.0])   # decoy

        answer_ids = [2, 0]

        def hand_logsoftmax(row, idx):
            z = math.log(sum(math.exp(v) for v in row))
            return row[idx] - z

        expected_1 = hand_logsoftmax([0.0, 1.0, 2.0, 3.0], 2)
        expected_2 = hand_logsoftmax([1.0, 0.5, 0.25, 0.125], 0)
        expected = expected_1 + expected_2

        total, per_token = answer_logprobs_from_logits(logits, answer_ids)
        self.assertAlmostEqual(float(total[0]), expected, places=5)
        self.assertAlmostEqual(float(per_token[0, 0]), expected_1, places=5)
        self.assertAlmostEqual(float(per_token[0, 1]), expected_2, places=5)
        # Sanity on the hand value itself, so a wrong `expected` cannot make a
        # wrong implementation pass: both terms are log-probs, hence negative,
        # and the first is log(e^2 / (e^0+e^1+e^2+e^3)).
        self.assertLess(expected_1, 0.0)
        self.assertLess(expected_2, 0.0)
        self.assertAlmostEqual(
            expected_1,
            math.log(math.exp(2.0) / (math.exp(0.0) + math.exp(1.0)
                                      + math.exp(2.0) + math.exp(3.0))),
            places=10)

    def test_f_alignment_is_off_by_one_sensitive(self):
        """Prove the decoys would in fact have caught a misalignment."""
        import torch

        logits = torch.zeros(1, 5, 4)
        logits[0, 0] = torch.tensor([9.0, -9.0, -9.0, -9.0])
        logits[0, 1] = torch.tensor([-9.0, 9.0, -9.0, -9.0])
        logits[0, 2] = torch.tensor([0.0, 1.0, 2.0, 3.0])
        logits[0, 3] = torch.tensor([1.0, 0.5, 0.25, 0.125])
        logits[0, 4] = torch.tensor([-9.0, -9.0, 9.0, -9.0])

        correct, _ = answer_logprobs_from_logits(logits, [2, 0])
        shifted = torch.log_softmax(logits[0, 3:5].float(), dim=-1)
        wrong = float(shifted[0, 2] + shifted[1, 0])
        self.assertGreater(abs(float(correct[0]) - wrong), 5.0)

    def test_f_single_token_answer_is_the_same_code_path(self):
        import torch
        logits = torch.zeros(1, 3, 4)
        logits[0, 1] = torch.tensor([0.0, 1.0, 2.0, 3.0])
        total, per_token = answer_logprobs_from_logits(logits, [3])
        z = math.log(sum(math.exp(v) for v in [0.0, 1.0, 2.0, 3.0]))
        self.assertAlmostEqual(float(total[0]), 3.0 - z, places=5)
        self.assertEqual(tuple(per_token.shape), (1, 1))

    def test_f_batch_is_scored_independently(self):
        import torch
        logits = torch.zeros(2, 4, 3)
        logits[0, 2] = torch.tensor([5.0, 0.0, 0.0])
        logits[1, 2] = torch.tensor([0.0, 0.0, 5.0])
        total, _ = answer_logprobs_from_logits(logits, [0])
        self.assertGreater(float(total[0]), float(total[1]))

    def test_f_rejects_a_sequence_too_short_to_hold_the_answer(self):
        import torch
        with self.assertRaises(ValueError):
            answer_logprobs_from_logits(torch.zeros(1, 2, 4), [1, 2])
        with self.assertRaises(ValueError):
            answer_logprobs_from_logits(torch.zeros(1, 5, 4), [])


# --------------------------------------------------------------------------
# Answer-span recovery (the differencing trick) and prompt construction
# --------------------------------------------------------------------------

class _FakeSpaceMergingTokenizer:
    """Mimics the one property of Llama's SentencePiece that matters here: a
    leading space is absorbed into the following piece, so encoding an answer
    in isolation yields DIFFERENT ids from the ones it has in context."""

    def __init__(self):
        self.vocab = {}

    def _id(self, piece):
        return self.vocab.setdefault(piece, len(self.vocab) + 10)

    def _pieces(self, text):
        out, i = [], 0
        while i < len(text):
            if text[i] == " ":
                j = i + 1
                while j < len(text) and text[j] != " ":
                    j += 1
                out.append(text[i:j])           # "▁word": space merged in
                i = j
            else:
                j = i
                while j < len(text) and text[j] != " ":
                    j += 1
                out.append(text[i:j])
                i = j
        return [p for p in out if p]

    def __call__(self, text):
        class _R:
            pass
        r = _R()
        r.input_ids = [self._id(p) for p in self._pieces(text)]
        return r

    def encode(self, text, add_special_tokens=False):
        return [self._id(p) for p in self._pieces(text)]

    def decode(self, ids):
        rev = {v: k for k, v in self.vocab.items()}
        return "".join(rev.get(i, "?") for i in ids)


class _FakeLlamaTemplate:
    """Same shape as tinyllava's LlamaTemplate: system + USER: … + ASSISTANT: …

    `transformers` is not installed in every environment that runs these tests,
    so the real template cannot be imported here. This stands in for it, and
    reproduces the two properties the scorer depends on: the answer follows
    'ASSISTANT: ' with exactly one space, and tokenization merges that space
    into the first answer piece.
    """

    SYSTEM = ("A chat between a curious user and an artificial intelligence "
              "assistant. ")

    def prompt(self, questions, answers):
        msg = self.SYSTEM
        for q, a in zip(questions, answers):
            q = q.replace("<image>", "").strip()
            msg += f"USER: <image>\n{q} ASSISTANT: {a}</s>"
        return msg

    @staticmethod
    def tokenizer_image_token(prompt, tokenizer, return_tensors=None):
        ids = tokenizer(prompt).input_ids
        if return_tensors == "pt":
            import torch
            return torch.tensor(ids, dtype=torch.long)
        return ids


class TestAnswerSpanRecovery(unittest.TestCase):

    def test_split_returns_the_added_tokens(self):
        self.assertEqual(split_answer_token_ids([1, 2, 3], [1, 2, 3, 4, 5]),
                         [4, 5])

    def test_split_raises_when_prefix_is_not_a_prefix(self):
        with self.assertRaises(ValueError) as ctx:
            split_answer_token_ids([1, 2, 3], [1, 9, 3, 4])
        self.assertIn("not a prefix", str(ctx.exception))

    def test_split_raises_when_the_answer_adds_nothing(self):
        with self.assertRaises(ValueError):
            split_answer_token_ids([1, 2, 3], [1, 2, 3])

    def test_prefix_never_ends_in_a_space(self):
        """The whole differencing trick collapses if the prefix carries the
        trailing space, because SentencePiece merges it forward."""
        s = TinyLlavaScorer(model=None, tokenizer=_FakeSpaceMergingTokenizer(),
                            template=_FakeLlamaTemplate())
        prefix, full = s.build_texts("Does this sample show CVD?", "Yes")
        self.assertTrue(prefix.endswith("ASSISTANT:"), prefix[-30:])
        self.assertFalse(prefix.endswith(" "))
        self.assertEqual(full, prefix + " Yes")
        self.assertNotIn("</s>", full)        # EOS is not part of the scored span

    def test_multi_token_answer_recovers_every_token(self):
        tok = _FakeSpaceMergingTokenizer()
        s = TinyLlavaScorer(model=None, tokenizer=tok,
                            template=_FakeLlamaTemplate())
        q = "Does this sample show CVD?"
        one = s.answer_token_ids(q, "Yes")
        many = s.answer_token_ids(q, "Yes, this sample shows evidence.")
        self.assertEqual(len(one), 1)
        self.assertEqual(len(many), 5)
        # First token differs: "▁Yes" vs "▁Yes," — the exact reason the answer
        # cannot be tokenized in isolation.
        self.assertNotEqual(one[0], many[0])

    def test_isolated_encoding_would_have_been_wrong(self):
        """Documents the bug this differencing avoids: encoding the answer on
        its own gives ids that never appear in the real sequence."""
        tok = _FakeSpaceMergingTokenizer()
        s = TinyLlavaScorer(model=None, tokenizer=tok,
                            template=_FakeLlamaTemplate())
        in_context = s.answer_token_ids("Q?", "Yes")
        isolated = tok.encode("Yes")
        self.assertNotEqual(in_context, isolated)

    def test_build_texts_raises_if_the_template_drops_the_answer(self):
        class _Dropping(_FakeLlamaTemplate):
            def prompt(self, questions, answers):
                return "USER: q ASSISTANT:"
        s = TinyLlavaScorer(model=None, tokenizer=_FakeSpaceMergingTokenizer(),
                            template=_Dropping())
        with self.assertRaises(RuntimeError):
            s.build_texts("q", "Yes")


# --------------------------------------------------------------------------
# Prompt variants, forced choice, metrics, baselines
# --------------------------------------------------------------------------

class TestPromptVariants(unittest.TestCase):

    def test_at_least_two_phrasings_exist(self):
        self.assertGreaterEqual(len(PROMPT_VARIANTS), 2)

    def test_at_least_one_variant_carries_an_order_control(self):
        self.assertTrue(any(v.presents_options for v in PROMPT_VARIANTS.values()))

    def test_option_order_actually_swaps_the_rendered_text(self):
        v = PROMPT_VARIANTS["v2_options"]
        a = v.render("positive_first")
        b = v.render("negative_first")
        self.assertNotEqual(a, b)
        self.assertLess(a.index(v.positive_option), a.index(v.negative_option))
        self.assertLess(b.index(v.negative_option), b.index(v.positive_option))
        # The question itself is unchanged; only the option order moves.
        self.assertTrue(a.startswith(v.question_template.split("{options}")[0]))

    def test_variant_without_options_has_one_order(self):
        v = PROMPT_VARIANTS["v1_direct"]
        self.assertEqual(v.orders(), ("n/a",))
        with self.assertRaises(ValueError):
            v.render("positive_first")

    def test_no_placeholder_survives_rendering(self):
        for v in PROMPT_VARIANTS.values():
            for o in v.orders():
                self.assertNotIn("{options}", v.render(o))
                self.assertNotIn("{first}", v.render(o))


class TestForcedChoice(unittest.TestCase):

    def test_equal_logprobs_give_one_half(self):
        s = AnswerScores(np.array([[-2.0, -2.0]]), np.array([1, 1]))
        p = forced_choice_probabilities(s)
        self.assertAlmostEqual(p["p_yes_sum"][0], 0.5, places=12)
        self.assertAlmostEqual(p["p_yes_mean"][0], 0.5, places=12)

    def test_matches_a_hand_computed_two_way_softmax(self):
        s = AnswerScores(np.array([[-1.0, -3.0]]), np.array([1, 1]))
        expected = math.exp(-1.0) / (math.exp(-1.0) + math.exp(-3.0))
        self.assertAlmostEqual(forced_choice_probabilities(s)["p_yes_sum"][0],
                               expected, places=12)

    def test_is_numerically_stable_at_extremes(self):
        s = AnswerScores(np.array([[-1e4, -1.0], [-1.0, -1e4]]), np.array([1, 1]))
        p = forced_choice_probabilities(s)["p_yes_sum"]
        self.assertTrue(np.all(np.isfinite(p)))
        self.assertAlmostEqual(p[0], 0.0, places=12)
        self.assertAlmostEqual(p[1], 1.0, places=12)

    def test_length_normalization_changes_the_answer(self):
        """The reason both scorings are reported. Per token the two answers are
        identical; only the sum prefers the shorter one."""
        s = LengthBiasedScorer([], np.array([])).score(np.zeros((4, 3)), "q",
                                                       ["long one", "no"])
        p = forced_choice_probabilities(s)
        self.assertTrue(np.all(p["p_yes_sum"] < 0.5))          # sum: prefers "no"
        self.assertTrue(np.allclose(p["p_yes_mean"], 0.5))     # mean: tied

    def test_rejects_anything_other_than_two_answers(self):
        with self.assertRaises(ValueError):
            forced_choice_probabilities(
                AnswerScores(np.zeros((3, 3)), np.array([1, 1, 1])))

    def test_answer_scores_rejects_a_zero_token_answer(self):
        with self.assertRaises(ValueError):
            AnswerScores(np.zeros((3, 2)), np.array([1, 0]))


class TestMetricsAndBaselines(unittest.TestCase):

    def test_binary_metrics_refuses_a_single_class_population(self):
        with self.assertRaises(SingleClassSeriesError):
            binary_metrics(np.ones(10, int), np.linspace(0, 1, 10))

    def test_accuracy_uses_the_natural_half_threshold(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([0.49, 0.51, 0.49, 0.51])
        self.assertAlmostEqual(binary_metrics(y, p)["accuracy"], 0.5)

    def test_prior_only_baseline_is_the_majority_rate(self):
        b = prior_only_baseline(np.array([1] * 70 + [0] * 30))
        self.assertAlmostEqual(b["accuracy"], 0.7)
        self.assertAlmostEqual(b["pr_auc"], 0.7)
        self.assertEqual(b["roc_auc"], 0.5)
        self.assertEqual(b["majority_class"], 1)

    def test_encoder_baseline_degrades_when_absent(self):
        out = encoder_linear_probe_baseline(Path("/nonexistent/probe.json"))
        self.assertFalse(out["available"])
        self.assertIn("reason", out)

    def test_encoder_baseline_reads_the_three_way_probe_file(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "probe_three_way.json"
            p.write_text(json.dumps({"populations": {"holdout_clean": {
                "BulkFormer-93M": {"linear": {"roc_auc_mean": 0.668,
                                              "roc_auc_std": 0.05,
                                              "pr_auc_mean": 0.632,
                                              "pr_auc_std": 0.04}}}}}))
            out = encoder_linear_probe_baseline(p)
        self.assertTrue(out["available"])
        self.assertAlmostEqual(out["roc_auc_mean"], 0.668)

    def test_tissue_baseline_degrades_and_says_so(self):
        import pandas as pd
        df = pd.DataFrame({"geo_accession": ["a", "b"], "series_id": ["s", "s"]})
        out = tissue_only_baseline(df, np.array([0, 1]), tissue_path=None)
        self.assertFalse(out["available"])
        self.assertIn("note", out)

    def test_tissue_baseline_runs_when_a_label_file_is_supplied(self):
        import tempfile
        import pandas as pd
        n_series, per = 10, 20
        rows, tis = [], []
        rng = np.random.default_rng(3)
        for s in range(n_series):
            for j in range(per):
                g = f"GSM{s}_{j}"
                pos = j % 2 == 0
                rows.append({"geo_accession": g, "series_id": f"S{s}"})
                # Tissue is deliberately informative, so a working baseline
                # must land well above chance.
                tis.append({"geo_accession": g,
                            "tissue": ("heart" if pos else "blood")
                            if rng.random() < 0.9
                            else ("blood" if pos else "heart")})
        df = pd.DataFrame(rows)
        y = np.array([1 if j % 2 == 0 else 0
                      for _ in range(n_series) for j in range(per)])
        with tempfile.TemporaryDirectory() as d:
            tp = Path(d) / "tissue.csv"
            pd.DataFrame(tis).to_csv(tp, index=False)
            out = tissue_only_baseline(df, y, tissue_path=tp)
        self.assertTrue(out["available"], out)
        self.assertGreater(out["roc_auc_mean"], 0.7)

    def test_per_series_aggregate_is_a_mean_over_series_not_samples(self):
        rows = [{"series_id": "A", "roc_auc": 1.0, "pr_auc": 1.0, "accuracy": 1.0},
                {"series_id": "B", "roc_auc": 0.0, "pr_auc": 0.0, "accuracy": 0.0}]
        agg = aggregate_per_series(rows)
        self.assertEqual(agg["n_series"], 2)
        self.assertAlmostEqual(agg["roc_auc_series_mean"], 0.5)
        self.assertAlmostEqual(agg["roc_auc_series_median"], 0.5)


class SeriesSignatureScorer(_StubScorer):
    """ZERO per-sample signal. The score is a pure function of series_id.

    This is the model Phase 0 warns about: the Stage-2 training population has
    no mixed-class series, so a model can learn "which study is this" and get
    the training label right every time without reading any biology. Within a
    single holdout series it has nothing to say, and its scores there are
    literally constant.
    """

    name = "stub:series-signature"

    def __init__(self, series: np.ndarray, series_score: dict):
        self.series = np.asarray(series)
        self.series_score = series_score

    def score(self, embeddings, question, answers):
        n = len(embeddings)
        lp = np.zeros((n, 2), dtype=np.float64)
        # The pipeline turns (lp_pos - lp_neg) into a probability, so feed the
        # log-odds that reproduce the intended per-series probability.
        q = np.array([self.series_score[s] for s in self.series[:n]])
        lp[:, 0] = np.log(q / (1.0 - q))
        return AnswerScores(lp, np.array([1, 1]))


def build_skewed_fixture(tmp: Path, n_series: int = 24, per_series: int = 20):
    """Series that are MIXED (so every guard passes) but class-SKEWED.

    Half the series are 90% positive, half 10% positive. A score that depends
    only on series_id therefore ranks well pooled, while carrying no
    information at all inside any single series — which is exactly the
    situation the within-series diagnostic exists to detect. The 92 real
    holdout series are skewed in just this way; a 50/50 fixture would make the
    shortcut undetectable by construction and the test vacuous.
    """
    import pandas as pd

    rows, series_score = [], {}
    k = 0
    for s in range(n_series):
        sid = f"GSE{80000 + s}"
        pos_heavy = s % 2 == 0
        n_pos = int(per_series * (0.9 if pos_heavy else 0.1))
        # Probabilities, not log-odds: binary_metrics scores accuracy at the
        # natural 0.5 threshold and a Brier score, both of which need [0, 1].
        series_score[sid] = 0.9 if pos_heavy else 0.1
        for j in range(per_series):
            pos = j < n_pos
            rows.append({"sample_index": 10 ** 9 + k,
                         "geo_accession": f"GSM{700000 + k}",
                         "series_id": sid, "cvd_subtype": None,
                         "is_positive": pos, "is_neg_hard": not pos,
                         "is_neg_whole_corpus": False})
            k += 1
    df = pd.DataFrame(rows)

    labels = tmp / "skew_labels.parquet"
    df.to_parquet(labels)
    holdout = tmp / "skew_holdout.json"
    holdout.write_text(json.dumps(
        {"holdout_series": sorted(df["series_id"].unique().tolist())}))
    stage2 = tmp / "skew_stage2.json"
    stage2.write_text(json.dumps([{"image": "GSMunrelated.npy"}]))
    enc = tmp / "skew_encoded"
    enc.mkdir(exist_ok=True)
    rng = np.random.default_rng(1)
    for g in df["geo_accession"]:
        np.save(enc / f"{g}.npy", rng.standard_normal(ENCODED_DIM).astype(np.float32))
    return {"labels": labels, "holdout": holdout, "stage2": stage2,
            "encoded": enc, "frame": df, "series_score": series_score}


class TestSeriesShortcutDetection(unittest.TestCase):
    """The coordinator's requirement (4): prove the diagnostic detects the
    shortcut, not merely that it runs."""

    @classmethod
    def setUpClass(cls):
        import tempfile
        cls._tmpdir = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls._tmpdir.name)
        cls.fx = build_skewed_fixture(cls.tmp)
        cls.df = cls.fx["frame"]
        cls.y = cls.df["is_positive"].astype(int).to_numpy()
        cls.series = cls.df["series_id"].to_numpy()
        cls.p_series_only = np.array(
            [cls.fx["series_score"][s] for s in cls.series], dtype=float)

    @classmethod
    def tearDownClass(cls):
        cls._tmpdir.cleanup()

    def test_series_only_score_is_high_pooled_and_chance_within_series(self):
        """THE test. Pooled AUC is high, within-series AUC is exactly 0.5.

        The pooled value is not approximate: with 12 series at 18 pos / 2 neg
        and 12 at 2 pos / 18 neg, and a score constant within each group, the
        concordant pairs are 216*216 and the tied pairs 2*216*24, giving
        (46656 + 5184) / 57600 = 0.900 exactly.
        """
        d = series_shortcut_diagnostics(self.y, self.p_series_only, self.series)
        self.assertAlmostEqual(d["pooled_roc_auc"], 0.9, places=10)
        self.assertAlmostEqual(d["within_series"]["weighted_mean"], 0.5, places=10)
        self.assertAlmostEqual(d["within_series"]["unweighted_mean"], 0.5, places=10)
        # The series-free pooled variants must also collapse to chance.
        self.assertAlmostEqual(
            d["series_centered_pooled"]["stratified_pairwise"]["roc_auc"],
            0.5, places=10)
        # zscore too. This one is the reason the fixture uses probabilities
        # like 0.9: twenty copies of 0.9 have a floating-point mean that is not
        # 0.9, so a naive (v - mean) / std amplifies 1e-16 of rounding noise
        # into +/-1 and hands the series signature straight back. It returned
        # 0.9 here before that was fixed.
        self.assertAlmostEqual(
            d["series_centered_pooled"]["zscore"]["roc_auc"], 0.5, places=10)
        # And the gap must be reported, large and positive.
        self.assertAlmostEqual(d["pooled_minus_within_series"], 0.4, places=10)

    def test_real_per_sample_signal_survives_both(self):
        """The complement: without it, a diagnostic that always says 0.5 would
        pass the test above."""
        p = 0.25 + 0.5 * self.y.astype(float) + 1e-9 * np.arange(len(self.y))
        d = series_shortcut_diagnostics(self.y, p, self.series)
        self.assertAlmostEqual(d["pooled_roc_auc"], 1.0, places=10)
        self.assertAlmostEqual(d["within_series"]["weighted_mean"], 1.0, places=10)
        self.assertAlmostEqual(
            d["series_centered_pooled"]["stratified_pairwise"]["roc_auc"],
            1.0, places=10)
        self.assertAlmostEqual(
            d["series_centered_pooled"]["zscore"]["roc_auc"], 1.0, places=10)
        self.assertAlmostEqual(d["pooled_minus_within_series"], 0.0, places=10)

    def test_detection_survives_the_full_pipeline(self):
        """End to end through main(), because a diagnostic that only works when
        called directly is not wired up."""
        rc = main(argv_for(self.fx, self.tmp, "shortcut",
                           ["--prompt-variant", "v1_direct",
                            "--encoder-embeddings", str(self.tmp / "none.parquet")]),
                  scorer=SeriesSignatureScorer(self.series, self.fx["series_score"]))
        self.assertEqual(rc, 0)
        p = json.loads((self.tmp / "tables" / "shortcut.json").read_text())
        d = p["runs"][0]["scorings"]["sum"]["series_shortcut_diagnostics"]
        self.assertAlmostEqual(d["pooled_roc_auc"], 0.9, places=6)
        self.assertAlmostEqual(d["within_series"]["weighted_mean"], 0.5, places=6)
        self.assertAlmostEqual(p["headline"]["llm_pooled_roc_auc"], 0.9, places=6)
        self.assertAlmostEqual(
            p["headline"]["llm_within_series_roc_auc"], 0.5, places=6)
        # The CSV must carry both, side by side, or a reader will quote one.
        self.assertAlmostEqual(
            d["series_centered_pooled"]["stratified_pairwise"]["roc_auc"],
            0.5, places=6)
        head = (self.tmp / "tables" / "shortcut.csv").read_text().splitlines()[0]
        self.assertIn("within_series_auc_weighted", head)
        self.assertLess(head.index("roc_auc"),
                        head.index("within_series_auc_weighted"))

    def test_eligibility_is_reported_not_silently_applied(self):
        y = np.array([1, 0, 1, 0] + [1, 0] + [1, 1, 1, 0, 0, 0])
        series = np.array(["small"] * 4 + ["tiny"] * 2 + ["ok"] * 6)
        p = np.arange(len(y), dtype=float)
        w = within_series_auc(y, p, series, min_n=6)
        self.assertEqual(w["n_series_total"], 3)
        self.assertEqual(w["n_series_eligible"], 1)
        self.assertEqual(w["n_series_skipped"], 2)
        reasons = {r["series_id"]: r["reason"] for r in w["skipped"]}
        self.assertIn("fewer than 6", reasons["small"])
        self.assertIn("fewer than 6", reasons["tiny"])

    def test_single_class_series_is_skipped_with_a_stated_reason(self):
        y = np.array([1] * 8 + [1, 0, 1, 0, 1, 0])
        series = np.array(["allpos"] * 8 + ["mixed"] * 6)
        p = np.arange(len(y), dtype=float)
        w = within_series_auc(y, p, series)
        self.assertEqual(w["n_series_eligible"], 1)
        self.assertEqual(w["skipped"][0]["reason"], "single class")

    def test_weighted_and_unweighted_means_differ_when_sizes_differ(self):
        """One huge series must not silently become the whole number."""
        y = np.array([1, 0] * 40 + [1, 0] * 3)
        series = np.array(["big"] * 80 + ["small"] * 6)
        p = np.concatenate([np.tile([1.0, 0.0], 40),      # big: perfect
                            np.tile([0.0, 1.0], 3)])      # small: inverted
        w = within_series_auc(y, p, series)
        self.assertEqual(w["n_series_eligible"], 2)
        self.assertAlmostEqual(w["unweighted_mean"], 0.5, places=10)
        self.assertGreater(w["weighted_mean"], 0.9)

    def test_rank_centered_pooling_is_prevalence_biased_hence_not_reported(self):
        """Documents WHY the reported series-free statistic is the stratified
        pairwise concordance and not rank-centered pooling.

        With a perfect per-sample signal but unequal class balance across
        series, rank-centered pooling returns ~0.68 while the truth is 1.0: a
        sample's normalized rank depends on how many of the OTHER class share
        its series. This test exists so nobody reinstates the simpler-looking
        statistic.
        """
        y = self.df["is_positive"].astype(int).to_numpy()
        p = 0.25 + 0.5 * y.astype(float) + 1e-9 * np.arange(len(y))
        rank_based = series_centered_pooled_auc(y, p, self.series, "rank")
        strat = stratified_pairwise_auc(y, p, self.series)
        self.assertLess(rank_based["roc_auc"], 0.8)
        self.assertAlmostEqual(strat["roc_auc"], 1.0, places=10)

    def test_zscore_centering_ignores_floating_point_only_variance(self):
        """The bug the shortcut test caught: a series whose scores are all the
        same value has std ~1e-16, not 0, and dividing by it resurrects the
        series signature at full strength."""
        series = np.array(["a"] * 20 + ["b"] * 20)
        p = np.array([0.9] * 20 + [0.1] * 20)
        self.assertGreater(p[:20].std(), 0.0)        # not exactly zero
        c = series_centered_scores(p, series, "zscore")
        self.assertTrue(np.allclose(c, 0.0))

    def test_rank_centering_is_scale_free(self):
        """A series whose scores are on a huge scale must not dominate the
        centered pooled AUC — that is why rank centering is reported."""
        series = np.array(["a"] * 4 + ["b"] * 4)
        p = np.array([0.1, 0.2, 0.3, 0.4, 1000.0, 2000.0, 3000.0, 4000.0])
        c = series_centered_scores(p, series, "rank")
        self.assertTrue(np.allclose(np.sort(c[:4]), np.sort(c[4:])))
        self.assertAlmostEqual(float(c.min()), -0.5)
        self.assertAlmostEqual(float(c.max()), 0.5)

    def test_centering_handles_singletons_and_zero_variance(self):
        series = np.array(["a", "b", "b"])
        p = np.array([5.0, 2.0, 2.0])
        for method in ("rank", "zscore"):
            c = series_centered_scores(p, series, method)
            self.assertEqual(float(c[0]), 0.0)          # singleton
            if method == "zscore":
                self.assertTrue(np.allclose(c[1:], 0.0))  # zero variance
        with self.assertRaises(ValueError):
            series_centered_scores(p, series, "nonsense")

    def test_centered_auc_uses_the_small_series_within_auc_discards(self):
        """The stated reason the two diagnostics complement each other."""
        series = np.array(["s%d" % (i // 2) for i in range(20)])
        y = np.array([1, 0] * 10)
        p = np.array([1.0, 0.0] * 10)                  # perfect inside each pair
        w = within_series_auc(y, p, series, min_n=6)
        self.assertEqual(w["n_series_eligible"], 0)    # every series has n=2
        self.assertIsNone(w["weighted_mean"])
        c = stratified_pairwise_auc(y, p, series)
        self.assertAlmostEqual(c["roc_auc"], 1.0, places=10)
        self.assertEqual(c["n_series_contributing"], 10)
        self.assertEqual(c["n_pairs"], 10)


class TestEncoderWithinSeriesBaseline(_FixtureCase):
    """The encoder must get the SAME diagnostic, or the comparison is not
    like-for-like — and when it cannot, that must be said out loud."""

    n_series = 8
    per_series = 20

    def test_encoder_diagnostic_reports_unavailable_when_it_cannot_run(self):
        p = self.run_main(PerfectScorer(self.order, self.y), "encnone",
                          ["--encoder-embeddings", str(self.tmp / "absent.parquet")])
        d = p["baselines"]["encoder_linear_probe"]["series_shortcut_diagnostics"]
        self.assertFalse(d["available"])
        self.assertIn("UNMEASURED", d["reason"])

    def test_encoder_diagnostic_runs_on_a_real_embedding_parquet(self):
        """Builds a small 515-d parquet whose signal is per-sample, so the
        encoder's within-series AUC must come back well above chance."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        rng = np.random.default_rng(11)
        n = len(self.y)
        X = rng.standard_normal((n, ENCODED_DIM)).astype(np.float32)
        # Spread the signal over several dimensions and make it strong: a
        # logistic probe on 515 dimensions from 160 samples is otherwise
        # fighting its own variance, and this test is about the diagnostic
        # being wired to the encoder, not about probe sample efficiency.
        X[:, :20] += 6.0 * self.y[:, None]          # per-sample, not per-series
        cols = {"sample_index": pa.array(
            self.df["sample_index"].to_numpy(), type=pa.int64())}
        for j in range(ENCODED_DIM):
            cols[f"e{j:04d}"] = pa.array(X[:, j])
        path = self.tmp / "enc.parquet"
        pq.write_table(pa.table(cols), path)

        p = self.run_main(PerfectScorer(self.order, self.y), "encreal",
                          ["--encoder-embeddings", str(path)])
        e = p["baselines"]["encoder_linear_probe"]
        d = e["series_shortcut_diagnostics"]
        self.assertNotEqual(d.get("available"), False)
        self.assertGreater(d["pooled_roc_auc"], 0.9)
        self.assertGreater(d["within_series"]["weighted_mean"], 0.9)
        self.assertGreater(
            d["series_centered_pooled"]["stratified_pairwise"]["roc_auc"], 0.9)
        self.assertEqual(p["headline"]["encoder_within_series_roc_auc"],
                         d["within_series"]["weighted_mean"])
        # And it must appear in the CSV as a baseline row with both numbers.
        lines = (self.tmp / "tables" / "encreal.csv").read_text().splitlines()
        row = [ln for ln in lines if "BulkFormer-93M linear probe" in ln]
        self.assertEqual(len(row), 1)


class TestMisuseGuards(unittest.TestCase):

    def test_allow_missing_embeddings_requires_a_debug_scorer(self):
        with self.assertRaises(SystemExit):
            main(["--allow-missing-embeddings", "--lora-ckpt", "x"])

    def test_a_real_run_requires_a_checkpoint(self):
        with self.assertRaises(SystemExit):
            main([])


if __name__ == "__main__":
    unittest.main(verbosity=2)
