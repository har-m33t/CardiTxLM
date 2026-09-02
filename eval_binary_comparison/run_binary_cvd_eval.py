"""run_binary_cvd_eval.py — Phase 4.3: the held-out binary CVD-vs-control
evaluation, scored by forced-choice answer log-probability.

WHAT THIS IS
------------
`.claude/stage_2_revisions.md` Phase 4.3: now that Stage 2 was trained on a
genuine discriminative task, ask the trained model that same question on the
CLEAN holdout and score it directly. Not generation, not string matching —
forced choice over the log-probability of the affirmative vs the negative
answer continuation.

WHAT IT INHERITS FROM `matched_binary_eval.py`
----------------------------------------------
That script built the first version of this idea, before the discriminative
retrain existed. Reused here, unchanged in spirit:

  * the forced-choice framing itself (one forward pass, read the answer
    logits, softmax over just the two options — no `.generate()`)
  * `linear_probe.probe._fold_metrics` as the metric implementation, imported
    rather than rewritten, so these numbers are computed by the same code as
    every probe number in this project
  * the model-loading path (`evaluate_and_compare.load_model`) and the
    conversation construction via `TemplateFactory(conv_version)` with
    `conv_version="llama"` — this repo's name for Vicuna v1.5's format
  * the pre-encoded 515-d passthrough: `BulkFormerVisionTower.forward` detects
    width 515 and does not re-run the encoder

WHAT IS DIFFERENT, AND WHY
--------------------------
1. POPULATION. `matched_binary_eval.py` scored the whole 31k probe population,
   98% of whose positives were in Stage-2 training, and reported a
   "contaminated" and a "held-out" number side by side. This scores ONLY the 92
   mixed holdout series (1,341 pos + 1,266 neg_hard). Overlap with
   `stage2_train.json` is a HARD ASSERTION here, not a reported statistic.

2. MULTI-TOKEN ANSWERS. `matched_binary_eval.py` read the logits for the FIRST
   token of "Yes"/"No" at the final prompt position. That silently assumes the
   answer is one token, which stops being true the moment a phrasing is longer
   than one word. This scores the full answer continuation by the chain rule:

       logP(answer | prompt) = sum_j logP(t_j | prompt, t_<j)

   and reports BOTH the summed and the length-normalized (mean per token)
   variant, because the sum structurally favours the shorter answer and the
   mean structurally favours the answer whose tokens are individually more
   probable. If the two disagree, the ranking was a length artifact.

3. PHRASING AND ORDER CONTROLS. A result that flips when the question is
   reworded is not a result. Every variant is run; for variants that present
   the two options explicitly, the options are ALSO presented in the opposite
   order and the gap is reported as an order-bias measure.

4. BASELINES. The number is uninterpretable alone, so three are reported
   alongside: always-predict-majority, tissue-only (if a tissue label is
   available), and the frozen BulkFormer-93M linear probe on this same holdout
   split — the encoder baseline this whole project is measured against.

5. WITHIN-SERIES AUC — the number this evaluation actually turns on. Phase 0
   found that the Stage-2 TRAINING population contains ZERO mixed-class series
   (388 positive-only, 1,174 negative-only). That is structural: the holdout
   was defined as every mixed series, so training received the complement.
   `series_id` is therefore a perfect predictor of the training label, and a
   model can ace the discriminative task on batch signature alone.

   Pooled holdout AUC cannot tell those apart. AUC computed WITHIN each of the
   92 mixed holdout series can, because it compares samples that share a batch,
   platform, lab and largely a tissue:

       shortcut-taking model  ->  pooled HIGH, within-series ~0.5
       real per-sample signal ->  both above chance

   The frozen encoder gets the identical treatment (its probe is re-run here
   for out-of-fold per-sample scores), so its within-series AUC is the bar the
   LLM has to clear. Pooled and within-series are always reported together.

THE STANDING GUARD
------------------
Any metric computed over a series containing only one class is a batch-
signature artifact, not a disease result (see `comparison_report.md` §3: a
0.9343 that turned out to be 15 single-class series). The 92 holdout series are
mixed by construction — so this ASSERTS it rather than trusting it.

Run (GPU):
    python -m eval_binary_comparison.run_binary_cvd_eval \\
        --lora-ckpt checkpoints/stage2-lora-bulkformer-93M-hypB \\
        --batch-size 32

Run (CPU pipeline smoke test, produces NO usable number):
    python -m eval_binary_comparison.run_binary_cvd_eval \\
        --debug-scorer random --allow-missing-embeddings --out-tag smoke
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO = Path(__file__).resolve().parent.parent

HOLDOUT_JSON = REPO / "data/cvd_transcriptome/holdout_series.json"
PROBE_LABELS = REPO / "linear_probe/probe_sample_labels.parquet"
STAGE2_JSON = REPO / "data/cvd_transcriptome/text_files/stage2_train.json"
ENCODED_DIR = REPO / "data/cvd_transcriptome/embeddings_encoded"
ENCODER_BASELINE_JSON = REPO / "stage2_regen_report/tables/probe_three_way.json"
ENCODER_EMBEDDINGS = (REPO
                      / "linear_probe/embeddings/embeddings_BulkFormer-93M.parquet")

OUT_TABLES = REPO / "stage2_regen_report/tables"
OUT_PLOTS = REPO / "stage2_regen_report/plots"

#: Same seed as every other probe in this project. Only used by the tissue-only
#: baseline's grouped CV; the LLM scores need no folds (nothing is fitted).
SEED = 20260707
K_FOLDS = 5

#: The encoder passes 515-d vectors through untouched. Anything else means the
#: wrong cache was pointed at.
ENCODED_DIM = 515


# --------------------------------------------------------------------------
# Prompt variants
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class PromptVariant:
    """One phrasing of the discriminative question.

    `positive_answer` / `negative_answer` are the literal answer continuations
    whose log-probability is compared. They are NOT assumed to be one token —
    see `TinyLlavaScorer` for how the answer span is recovered.

    A variant that lists the two options inside the question has
    `presents_options=True`; it is then run twice, once with the affirmative
    option first and once with the negative option first, and the difference is
    reported as position bias. A variant that does not list options has no
    order to control and is run once.
    """

    name: str
    question_template: str          # contains "{options}" iff presents_options
    positive_answer: str
    negative_answer: str
    presents_options: bool = False
    positive_option: str = ""
    negative_option: str = ""
    options_frame: str = "Answer either '{first}' or '{second}'."

    def orders(self) -> Tuple[str, ...]:
        return ("positive_first", "negative_first") if self.presents_options \
            else ("n/a",)

    def render(self, order: str = "n/a") -> str:
        if not self.presents_options:
            if order != "n/a":
                raise ValueError(
                    f"variant {self.name!r} presents no options, so order "
                    f"{order!r} is meaningless")
            return self.question_template
        if order == "positive_first":
            first, second = self.positive_option, self.negative_option
        elif order == "negative_first":
            first, second = self.negative_option, self.positive_option
        else:
            raise ValueError(f"unknown order {order!r}")
        opts = self.options_frame.format(first=first, second=second)
        return self.question_template.format(options=opts)


#: `v1_direct` is verbatim the question used by `matched_binary_eval.py` and by
#: the Phase-2 discriminative training items, so it is the in-distribution one.
#: The other two exist to prove the result is not a property of that wording.
PROMPT_VARIANTS: Dict[str, PromptVariant] = {
    "v1_direct": PromptVariant(
        name="v1_direct",
        question_template=(
            "Based on this transcriptomic profile, does this sample show "
            "evidence of cardiovascular disease?"),
        positive_answer="Yes",
        negative_answer="No",
    ),
    "v2_options": PromptVariant(
        name="v2_options",
        question_template=(
            "Based on this transcriptomic profile, does this sample show "
            "evidence of cardiovascular disease? {options}"),
        positive_answer="Yes, this sample shows evidence of cardiovascular disease.",
        negative_answer="No, this sample shows no evidence of cardiovascular disease.",
        presents_options=True,
        positive_option="Yes, this sample shows evidence of cardiovascular disease.",
        negative_option="No, this sample shows no evidence of cardiovascular disease.",
    ),
    "v3_clinical": PromptVariant(
        name="v3_clinical",
        question_template=(
            "A bulk transcriptomic profile from a human tissue sample is "
            "provided. Is the donor of this sample affected by cardiovascular "
            "disease? {options}"),
        positive_answer="Affected",
        negative_answer="Unaffected",
        presents_options=True,
        positive_option="affected",
        negative_option="unaffected",
    ),
}


# --------------------------------------------------------------------------
# Population + the two hard guards
# --------------------------------------------------------------------------

class LeakageError(AssertionError):
    """Holdout sample found in the Stage-2 training corpus."""


class SingleClassSeriesError(AssertionError):
    """A contributing series carries only one class.

    Separating such a series is separating studies, not disease. See
    comparison_report.md §3.
    """


def assert_no_stage2_overlap(geo_accessions: Sequence[str],
                             stage2_json: Path) -> dict:
    """HARD assertion: no evaluated sample appears in Stage-2 training.

    Matched on the `image` filename, which is `<GEO_ACCESSION>.npy` — the same
    key the training loader uses, so this compares the identifier the model
    actually consumed rather than a re-derived one.
    """
    records = json.loads(Path(stage2_json).read_text())
    trained = {r["image"] for r in records if "image" in r}
    ours = {f"{g}.npy" for g in geo_accessions}
    overlap = sorted(ours & trained)
    if overlap:
        raise LeakageError(
            f"{len(overlap)} of {len(ours)} holdout samples appear in "
            f"{Path(stage2_json).name} — the evaluation is contaminated and "
            f"its numbers would be meaningless. First offenders: "
            f"{overlap[:10]}")
    return {"n_evaluated": len(ours),
            "n_stage2_training_images": len(trained),
            "overlap": 0,
            "checked_on": "image filename (<GEO_ACCESSION>.npy)"}


def assert_series_have_both_classes(series_ids: Sequence[str],
                                    y: np.ndarray) -> dict:
    """HARD assertion: every contributing series carries both classes.

    The project's standing guard. The 92 holdout series are mixed *by
    construction* (holdout_series.json's own definition), which is exactly why
    this is asserted and not assumed — a construction can be re-run wrong.
    """
    y = np.asarray(y)
    series_ids = np.asarray(series_ids)
    bad = []
    for s in sorted(set(series_ids.tolist())):
        m = series_ids == s
        classes = set(y[m].tolist())
        if len(classes) < 2:
            bad.append((s, int(m.sum()), sorted(classes)))
    if bad:
        raise SingleClassSeriesError(
            f"{len(bad)} of {len(set(series_ids.tolist()))} contributing "
            f"series contain only one class; any metric over them is a batch-"
            f"signature artifact, not a disease result (comparison_report.md "
            f"§3). First offenders (series, n, classes): {bad[:10]}")
    return {"n_series": len(set(series_ids.tolist())),
            "n_series_missing_a_class": 0,
            "every_series_has_both_classes": True}


def load_holdout_population(labels_path: Path = PROBE_LABELS,
                            holdout_path: Path = HOLDOUT_JSON,
                            stage2_json: Path = STAGE2_JSON) -> tuple:
    """Return (frame, guards). Both hard assertions fire here, before anything
    expensive is loaded."""
    import pandas as pd

    held = set(json.loads(Path(holdout_path).read_text())["holdout_series"])
    labels = pd.read_parquet(labels_path)
    keep = (labels["series_id"].isin(held)
            & (labels["is_positive"] | labels["is_neg_hard"]))
    cols = ["geo_accession", "series_id", "is_positive", "is_neg_hard"]
    if "sample_index" in labels.columns:      # needed by the tissue baseline
        cols = ["sample_index"] + cols
    df = labels.loc[keep, cols].copy()
    df = df.reset_index(drop=True)
    df["y"] = df["is_positive"].astype(int)

    guards = {
        "stage2_overlap": assert_no_stage2_overlap(df["geo_accession"], stage2_json),
        "series_class_balance": assert_series_have_both_classes(
            df["series_id"].to_numpy(), df["y"].to_numpy()),
    }
    return df, guards


def load_encoded_embeddings(geo_accessions: Sequence[str],
                            encoded_dir: Path = ENCODED_DIR,
                            allow_missing: bool = False) -> tuple:
    """Load the pre-encoded 515-d vectors. Fails at RUN time, not import time.

    The negative half of this cache is being materialized concurrently, so a
    missing file is a plausible, expected failure and gets a message that says
    which samples and how many rather than a bare FileNotFoundError.
    """
    encoded_dir = Path(encoded_dir)
    missing = [g for g in geo_accessions if not (encoded_dir / f"{g}.npy").exists()]
    if missing and not allow_missing:
        raise FileNotFoundError(
            f"{len(missing)} of {len(geo_accessions)} holdout samples have no "
            f"pre-encoded vector under {encoded_dir}. The negative half of this "
            f"cache is built separately; re-run this once it covers both "
            f"classes. Missing e.g.: {missing[:10]}")

    X = np.zeros((len(geo_accessions), ENCODED_DIM), dtype=np.float32)
    for i, g in enumerate(geo_accessions):
        p = encoded_dir / f"{g}.npy"
        if not p.exists():
            continue
        v = np.load(p).astype(np.float32).reshape(-1)
        if v.shape[0] != ENCODED_DIM:
            raise ValueError(
                f"{p} has width {v.shape[0]}, expected {ENCODED_DIM}. The "
                f"vision tower's pre-encoded passthrough keys off width 515; "
                f"anything else would be re-encoded or rejected.")
        X[i] = v
    return X, {"n": len(geo_accessions), "n_missing": len(missing),
               "missing_filled_with_zeros": bool(missing and allow_missing)}


# --------------------------------------------------------------------------
# The scoring interface — the only place the real model is touched
# --------------------------------------------------------------------------

@dataclass
class AnswerScores:
    """Log-probabilities of each candidate answer continuation.

    sum_logprob : [n_samples, n_answers]  sum_j logP(t_j | prompt, t_<j)
    n_tokens    : [n_answers]             how many tokens each answer spans
    """
    sum_logprob: np.ndarray
    n_tokens: np.ndarray

    def __post_init__(self):
        self.sum_logprob = np.asarray(self.sum_logprob, dtype=np.float64)
        self.n_tokens = np.asarray(self.n_tokens, dtype=np.int64)
        if self.sum_logprob.ndim != 2:
            raise ValueError(f"sum_logprob must be 2-d, got {self.sum_logprob.shape}")
        if self.sum_logprob.shape[1] != self.n_tokens.shape[0]:
            raise ValueError(
                f"sum_logprob has {self.sum_logprob.shape[1]} answers but "
                f"n_tokens has {self.n_tokens.shape[0]}")
        if (self.n_tokens <= 0).any():
            raise ValueError("an answer spans zero tokens; nothing to score")


class AnswerScorer:
    """Everything downstream of this is model-free and CPU-testable.

    Implementations return the total log-probability of each answer string as a
    continuation of the prompt, for each sample's expression profile.
    """

    #: Set by implementations that are not the real model, so the report can
    #: refuse to be mistaken for a result.
    is_debug: bool = False
    name: str = "abstract"

    def score(self, embeddings: np.ndarray, question: str,
              answers: Sequence[str]) -> AnswerScores:
        raise NotImplementedError


def split_answer_token_ids(prefix_ids: Sequence[int],
                           full_ids: Sequence[int]) -> List[int]:
    """The tokens the answer occupies, by differencing two tokenizations.

    Why not just `tokenizer.encode(answer)`: Llama's SentencePiece tokenizer
    merges the leading space into the first answer piece ("▁Yes"), so encoding
    the answer in isolation produces different ids from the ones that actually
    appear in the full sequence. Differencing is the only way to get the ids
    the model will really be conditioned on.

    Requires the prefix tokenization to be a genuine prefix of the full one. It
    normally is, because the prefix ends at "ASSISTANT:" — a non-space
    boundary. If it is ever not, that is a silent scoring corruption, so it
    raises.
    """
    prefix_ids = list(prefix_ids)
    full_ids = list(full_ids)
    if len(full_ids) <= len(prefix_ids):
        raise ValueError(
            f"answer added no tokens: prefix={len(prefix_ids)} "
            f"full={len(full_ids)}")
    if full_ids[:len(prefix_ids)] != prefix_ids:
        n = next((i for i in range(len(prefix_ids))
                  if full_ids[i] != prefix_ids[i]), len(prefix_ids))
        raise ValueError(
            f"prompt tokenization is not a prefix of prompt+answer "
            f"(diverges at token {n}: prefix has {prefix_ids[n:n+4]}, full has "
            f"{full_ids[n:n+4]}). Differencing would score the wrong span.")
    return full_ids[len(prefix_ids):]


def answer_logprobs_from_logits(logits, answer_ids: Sequence[int]):
    """Per-token log-probs of the answer, aligned from the RIGHT.

    `logits` is [B, S, V] over the full prompt+answer sequence. The answer
    occupies the last A positions, so logits at S-A-1 .. S-2 are the ones that
    predict it. Aligning from the right rather than the left is deliberate: the
    multimodal path replaces the single `<image>` token with the connector's
    output, and if that ever emits more than one token the left-hand offsets
    all shift while the right-hand ones do not.

    Returns (sum_logprob [B], per_token_logprob [B, A]) as torch tensors.
    """
    import torch

    a = len(answer_ids)
    if a == 0:
        raise ValueError("empty answer")
    if logits.shape[1] < a + 1:
        raise ValueError(
            f"sequence length {logits.shape[1]} cannot hold a {a}-token answer "
            f"plus at least one conditioning position")
    window = logits[:, logits.shape[1] - a - 1: logits.shape[1] - 1, :].float()
    lp = torch.log_softmax(window, dim=-1)                       # [B, A, V]
    idx = torch.as_tensor(list(answer_ids), dtype=torch.long,
                          device=lp.device).view(1, a, 1).expand(lp.shape[0], a, 1)
    per_token = lp.gather(-1, idx).squeeze(-1)                   # [B, A]
    return per_token.sum(dim=-1), per_token


class TinyLlavaScorer(AnswerScorer):
    """The real thing: Vicuna-7B + Stage-1 connector + Stage-2 LoRA.

    Conversation construction follows `extract_llm_latents.py` exactly —
    `TemplateFactory(conv_version)`, `conv_version="llama"` being this repo's
    name for the Vicuna v1.5 format — and the prompt text is taken from the
    template's own `prompt()`, so the eval string is byte-identical in shape to
    the training string.
    """

    name = "tinyllava"

    def __init__(self, model, tokenizer, conv_version: str = "llama",
                 batch_size: int = 32, template=None):
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        if template is None:
            # Imported lazily: `transformers` is not installed on every machine
            # that needs to import this module (e.g. to run the tests), and a
            # top-level import would make the whole file unloadable there.
            from tinyllava.data.template import TemplateFactory
            template = TemplateFactory(conv_version)()
        self.template = template
        self._answer_token_cache: Dict[Tuple[str, str], List[int]] = {}

    _SENTINEL = "<<<ANSWER>>>"

    def build_texts(self, question: str, answer: str) -> Tuple[str, str]:
        """(prefix_text, full_text) sharing an exact string prefix.

        The template renders 'system USER: <image>\\nQ ASSISTANT: A</s>'. The
        answer is located by rendering with a sentinel, so this does not
        hard-code the separator and survives a template change.
        """
        q = "<image>\n" + question
        rendered = self.template.prompt([q], [self._SENTINEL])
        if self._SENTINEL not in rendered:
            raise RuntimeError(
                "template.prompt() dropped the answer sentinel; cannot locate "
                "the answer span. Template: "
                f"{type(self.template).__name__}")
        head = rendered.split(self._SENTINEL)[0]
        # head ends with 'ASSISTANT: ' — the trailing space must live on the
        # FULL side, never on the prefix side, or SentencePiece will merge it
        # into the first answer piece and break the prefix relation.
        return head.rstrip(), head + answer

    def answer_token_ids(self, question: str, answer: str) -> List[int]:
        key = (question, answer)
        if key not in self._answer_token_cache:
            prefix_text, full_text = self.build_texts(question, answer)
            prefix_ids = self.template.tokenizer_image_token(
                prefix_text, self.tokenizer)
            full_ids = self.template.tokenizer_image_token(
                full_text, self.tokenizer)
            self._answer_token_cache[key] = split_answer_token_ids(
                prefix_ids, full_ids)
        return self._answer_token_cache[key]

    def score(self, embeddings: np.ndarray, question: str,
              answers: Sequence[str]) -> AnswerScores:
        import torch

        n = len(embeddings)
        out = np.zeros((n, len(answers)), dtype=np.float64)
        n_tokens = []
        self.model.eval()

        for j, answer in enumerate(answers):
            _, full_text = self.build_texts(question, answer)
            full_ids = self.template.tokenizer_image_token(
                full_text, self.tokenizer, return_tensors="pt")
            ans_ids = self.answer_token_ids(question, answer)
            n_tokens.append(len(ans_ids))
            print(f"    answer {j} {answer[:40]!r}: {len(ans_ids)} tokens "
                  f"{ans_ids} -> {[self.tokenizer.decode([i]) for i in ans_ids]}")

            device = next(self.model.parameters()).device
            dtype = next(self.model.parameters()).dtype
            ids = full_ids.to(device)
            for i in range(0, n, self.batch_size):
                chunk = embeddings[i:i + self.batch_size]
                b = len(chunk)
                images = torch.from_numpy(np.asarray(chunk)).to(
                    device=device, dtype=dtype)
                input_ids = ids.unsqueeze(0).expand(b, -1).contiguous()
                with torch.inference_mode():
                    logits = self.model(input_ids=input_ids, images=images).logits
                # The multimodal path substitutes the <image> token with the
                # connector's output; if that ever emits more than one token the
                # sequence grows. Right-alignment is immune to growth BEFORE the
                # answer, but not to the sequence being truncated to shorter than
                # the answer, so state the delta once and refuse a shrink.
                delta = int(logits.shape[1]) - int(input_ids.shape[1])
                if i == 0:
                    print(f"      seq: input={input_ids.shape[1]} "
                          f"model={logits.shape[1]} (image expansion {delta:+d}); "
                          f"answer occupies the last {len(ans_ids)} positions")
                if delta < 0:
                    raise RuntimeError(
                        f"model returned a SHORTER sequence than the input "
                        f"({logits.shape[1]} < {input_ids.shape[1]}); the answer "
                        f"span can no longer be located by right-alignment")
                s, _ = answer_logprobs_from_logits(logits, ans_ids)
                out[i:i + b, j] = s.detach().cpu().numpy()
                if (i // self.batch_size) % 20 == 0:
                    print(f"      {min(i + b, n)}/{n}", flush=True)
        return AnswerScores(out, np.asarray(n_tokens))


class _DebugScorer(AnswerScorer):
    """NOT A RESULT. Exists so the whole pipeline can be exercised on a CPU.

    Every output written while one of these is in use carries
    `debug_scorer_not_a_real_result: true`.
    """

    is_debug = True

    def __init__(self, mode: str, y: Optional[np.ndarray] = None, seed: int = 0):
        if mode not in ("random", "leaky_oracle"):
            raise ValueError(f"unknown debug scorer {mode!r}")
        self.mode = mode
        self.name = f"debug:{mode}"
        self.y = None if y is None else np.asarray(y)
        self.rng = np.random.default_rng(seed)

    def score(self, embeddings, question, answers):
        n, k = len(embeddings), len(answers)
        lp = -self.rng.random((n, k)) * 5.0
        if self.mode == "leaky_oracle":
            if self.y is None:
                raise ValueError("leaky_oracle needs labels")
            lp[:, 0] += self.y * 4.0
        return AnswerScores(lp, np.arange(1, k + 1))


# --------------------------------------------------------------------------
# Forced choice, metrics, per-series breakdown
# --------------------------------------------------------------------------

def forced_choice_probabilities(scores: AnswerScores) -> Dict[str, np.ndarray]:
    """P(affirmative) under the two-way softmax, in both scoring conventions.

    Answer 0 is the affirmative, answer 1 the negative.

    `sum`  — softmax over the total sequence log-likelihoods. This is the
             literal forced choice, and is what a model actually "prefers", but
             it structurally favours whichever answer is shorter.
    `mean` — softmax over the per-token means. Removes the length preference,
             but rewards an answer whose individual tokens are bland.

    Both are reported. Neither is privileged. Disagreement between them means
    the ranking was driven by answer length, not by the profile.
    """
    if scores.sum_logprob.shape[1] != 2:
        raise ValueError("forced choice needs exactly 2 answers")
    lp = scores.sum_logprob
    nt = scores.n_tokens.astype(np.float64)
    out = {}
    for tag, margin in (("sum", lp[:, 0] - lp[:, 1]),
                        ("mean", lp[:, 0] / nt[0] - lp[:, 1] / nt[1])):
        # sigmoid(margin) == softmax over the pair. Computed branch-wise rather
        # than with np.where, which would evaluate BOTH expressions and overflow
        # on the large-|margin| side before discarding it. Log-prob differences
        # of several thousand are ordinary here (a 10-token answer at -400
        # nats/token is not exotic), so this is a real case, not a theoretical
        # one — and an inf/nan would silently poison the AUC.
        m = np.asarray(margin, dtype=np.float64)
        p = np.empty_like(m)
        hi = m >= 0
        p[hi] = 1.0 / (1.0 + np.exp(-m[hi]))
        e = np.exp(m[~hi])
        p[~hi] = e / (1.0 + e)
        out[f"p_yes_{tag}"] = p
    return out


def binary_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    """ROC-AUC, PR-AUC, accuracy at the natural threshold (0.5), and the rest.

    Delegates to `linear_probe.probe._fold_metrics` so these are the same
    numbers, from the same code, as every probe result in this project. 0.5 is
    the *natural* threshold here because p is a two-way softmax: 0.5 is exactly
    the point where the model prefers one answer over the other.
    """
    from linear_probe.probe import _fold_metrics
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=np.float64)
    # accuracy-at-0.5 and the Brier score both presuppose a calibrated-scale
    # probability, not an arbitrary monotone score. Say so here rather than
    # letting sklearn raise three frames down.
    if p.size and (p.min() < 0.0 or p.max() > 1.0):
        raise ValueError(
            f"scores must be probabilities in [0, 1] (got [{p.min():.4g}, "
            f"{p.max():.4g}]); the natural-threshold accuracy and the Brier "
            f"score are undefined otherwise. Pass p_yes, not a log-odds margin.")
    if len(set(y.tolist())) < 2:
        raise SingleClassSeriesError(
            "binary_metrics called on a single-class population; ROC-AUC is "
            "undefined and any number here would be an artifact")
    m = _fold_metrics(y, p)
    m["n"] = int(len(y))
    m["n_positive"] = int(y.sum())
    m["n_negative"] = int((1 - y).sum())
    return m


def per_series_metrics(y: np.ndarray, p: np.ndarray,
                       series: Sequence[str]) -> List[dict]:
    """One row per series. Asserts first — a single-class series contributing
    to a breakdown is the exact failure mode §3 of comparison_report.md
    documents."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=np.float64)
    series = np.asarray(series)
    assert_series_have_both_classes(series, y)
    rows = []
    for s in sorted(set(series.tolist())):
        m = series == s
        row = {"series_id": s}
        row.update(binary_metrics(y[m], p[m]))
        rows.append(row)
    return rows


#: A series needs at least this many samples before its own AUC means anything.
#: Below it a single sample flips the number between 0 and 1.
MIN_SERIES_N_FOR_AUC = 6


def within_series_auc(y: np.ndarray, p: np.ndarray, series: Sequence[str],
                      min_n: int = MIN_SERIES_N_FOR_AUC) -> dict:
    """AUC computed INSIDE each series, then pooled. The batch-shortcut test.

    WHY THIS IS THE LOAD-BEARING NUMBER
    -----------------------------------
    Phase 0 established that in the Stage-2 TRAINING population no GEO series
    contains both classes: 388 positive-only series, 1,174 negative-only, zero
    mixed. That is structural, not chance — the holdout was *defined* as every
    mixed-class series, so training got the complement. `series_id` is
    therefore a PERFECT predictor of the training label, and a model can score
    well on the discriminative task by recognising batch signature while
    learning no disease biology whatsoever.

    Pooled holdout AUC cannot separate those two stories. Within-series AUC
    can: comparing samples inside one series holds batch, platform, lab, and
    (largely) tissue fixed, so only genuine per-sample biology can order them.

        shortcut-taking model  ->  pooled HIGH, within-series ~0.5
        real per-sample signal ->  both above chance

    Reported as a weighted mean (by the number of positive-negative pairs each
    series contributes, which is what a pooled AUC would weight by) AND an
    unweighted mean (one vote per series, so one enormous study cannot carry
    the number). Both are given because they answer different questions and a
    gap between them is itself informative.

    Never report this without the pooled number beside it, or vice versa.
    """
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=np.float64)
    series = np.asarray(series)

    rows, skipped = [], []
    for s in sorted(set(series.tolist())):
        m = series == s
        n, npos, nneg = int(m.sum()), int(y[m].sum()), int((1 - y[m]).sum())
        if npos == 0 or nneg == 0:
            skipped.append({"series_id": s, "n": n, "n_positive": npos,
                            "n_negative": nneg, "reason": "single class"})
            continue
        if n < min_n:
            skipped.append({"series_id": s, "n": n, "n_positive": npos,
                            "n_negative": nneg,
                            "reason": f"fewer than {min_n} samples"})
            continue
        rows.append({"series_id": s, "n": n, "n_positive": npos,
                     "n_negative": nneg, "n_pairs": npos * nneg,
                     "roc_auc": float(roc_auc_score(y[m], p[m]))})

    aucs = np.array([r["roc_auc"] for r in rows], dtype=np.float64)
    pairs = np.array([r["n_pairs"] for r in rows], dtype=np.float64)
    out = {
        "min_series_n": min_n,
        "n_series_total": int(len(set(series.tolist()))),
        "n_series_eligible": int(len(rows)),
        "n_series_skipped": int(len(skipped)),
        "skipped": skipped,
        "weighted_mean": float(np.sum(aucs * pairs) / pairs.sum()) if len(rows) else None,
        "unweighted_mean": float(aucs.mean()) if len(rows) else None,
        "unweighted_std": float(aucs.std()) if len(rows) else None,
        "median": float(np.median(aucs)) if len(rows) else None,
        "n_series_above_half": int((aucs > 0.5).sum()) if len(rows) else 0,
        "per_series": rows,
        "interpretation": ("~0.5 here with a high pooled AUC means the model "
                           "is reading series/batch signature, not disease; "
                           "Phase 0 showed the Stage-2 training population "
                           "makes exactly that shortcut available"),
    }
    return out


def stratified_pairwise_auc(y: np.ndarray, p: np.ndarray,
                            series: Sequence[str]) -> dict:
    """One global, rank-based, series-free AUC: the Mann-Whitney concordance
    over every positive-negative pair that lives in the SAME series.

    This is the complement to `within_series_auc`, not a duplicate of it. That
    function needs a per-series sample floor before a per-series AUC means
    anything, and discards everything below it. This one never forms a
    cross-series pair either, so it is equally free of batch signature, but it
    has no floor: a series with one positive and one negative contributes its
    single pair. All the small series within-series AUC has to throw away are
    used here.

    With no size floor it coincides exactly with `within_series_auc`'s
    pair-weighted mean over the series both admit — by construction, since the
    pair-weighted mean of per-series AUCs IS this ratio.

    WHY NOT RANK-CENTER AND POOL. The obvious alternative — normalize each
    series' scores to ranks in [-0.5, 0.5] and take one global AUC — is WRONG
    when series differ in class balance, and the holdout series do. A sample's
    normalized rank depends on how many samples of the other class sit in its
    series, so in a 90%-positive series the positives are pushed toward the
    middle while in a 10%-positive series the negatives are. Pooling those puts
    negatives above positives across series and drags the number down: on a
    stub with a *perfect* per-sample signal it returns 0.68, not 1.0. This
    statistic has no such artifact.
    """
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=np.float64)
    series = np.asarray(series)

    conc, pairs, used, excluded = 0.0, 0, 0, 0
    for s in set(series.tolist()):
        m = series == s
        npos, nneg = int(y[m].sum()), int((1 - y[m]).sum())
        if npos == 0 or nneg == 0:
            excluded += 1
            continue
        # roc_auc_score is the concordance RATE over this series' pairs, so
        # multiplying by the pair count recovers the concordance COUNT; ties
        # already contribute 0.5 inside it.
        conc += float(roc_auc_score(y[m], p[m])) * npos * nneg
        pairs += npos * nneg
        used += 1
    if pairs == 0:
        raise SingleClassSeriesError(
            "no series contains both classes; no within-series pair exists")
    return {"method": "stratified_pairwise",
            "roc_auc": conc / pairs,
            "n_pairs": int(pairs),
            "n_series_contributing": used,
            "n_series_excluded_single_class": excluded,
            "note": "no minimum series size; every same-series pair counts once"}


def series_centered_scores(p: np.ndarray, series: Sequence[str],
                           method: str = "rank") -> np.ndarray:
    """Remove each series' own score offset.

    `zscore` — (p - mean) / std within series. Keeps magnitude information, and
               unlike rank normalization is not distorted by a series' class
               balance, so it is the centering actually reported.
    `rank`   — average rank within series mapped to [-0.5, 0.5]. Scale free,
               but prevalence-sensitive (see `stratified_pairwise_auc`), so it
               is NOT reported as a headline statistic. Kept because it is the
               right tool when only the ordering within a series matters.
    """
    p = np.asarray(p, dtype=np.float64)
    series = np.asarray(series)
    out = np.zeros_like(p)
    for s in set(series.tolist()):
        m = series == s
        v = p[m]
        n = v.size
        if n < 2:
            out[m] = 0.0
            continue
        if method == "rank":
            from scipy.stats import rankdata
            r = rankdata(v)                       # average ranks for ties
            out[m] = (r - 1.0) / (n - 1.0) - 0.5
        elif method == "zscore":
            mean = float(v.mean())
            sd = float(v.std())
            # A series whose scores are all equal has sd 0 in exact arithmetic
            # but ~1e-16 in floating point, because the mean of twenty copies
            # of 0.9 is not exactly 0.9. Dividing by that amplifies pure
            # rounding noise to +/-1 and hands the series signature straight
            # back — this exact bug made a series-only stub score 0.9 instead
            # of the 0.5 it must score. Compare against the data's own scale.
            if not np.isfinite(sd) or sd <= 1e-12 * max(1.0, abs(mean)):
                out[m] = 0.0
            else:
                out[m] = (v - mean) / sd
        else:
            raise ValueError(f"unknown centering method {method!r}")
    return out


def series_centered_pooled_auc(y: np.ndarray, p: np.ndarray,
                               series: Sequence[str], method: str = "zscore") -> dict:
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y).astype(int)
    c = series_centered_scores(p, series, method)
    if len(set(y.tolist())) < 2:
        raise SingleClassSeriesError("single-class population")
    return {"method": method, "roc_auc": float(roc_auc_score(y, c)),
            "n": int(len(y)),
            "n_series": int(len(set(np.asarray(series).tolist())))}


def series_shortcut_diagnostics(y: np.ndarray, p: np.ndarray,
                                series: Sequence[str]) -> dict:
    """Pooled, within-series, and series-centered — always together.

    The contrast between the first and the rest is the whole diagnostic; any
    one of them alone is misleading, so they are produced by one call and
    stored in one block.
    """
    pooled = binary_metrics(y, p)
    within = within_series_auc(y, p, series)
    centered = {"stratified_pairwise": stratified_pairwise_auc(y, p, series),
                "zscore": series_centered_pooled_auc(y, p, series, "zscore")}
    gap = None
    if within["weighted_mean"] is not None:
        gap = pooled["roc_auc"] - within["weighted_mean"]
    return {
        "pooled_roc_auc": pooled["roc_auc"],
        "pooled_pr_auc": pooled["pr_auc"],
        "pooled_accuracy": pooled["accuracy"],
        "within_series": within,
        "series_centered_pooled": centered,
        "pooled_minus_within_series": gap,
        "reading": ("a large positive pooled_minus_within_series is the "
                    "signature of series/batch shortcut; near zero means the "
                    "pooled number survives holding batch fixed"),
    }


def aggregate_per_series(rows: List[dict]) -> dict:
    out = {"n_series": len(rows)}
    for key in ("roc_auc", "pr_auc", "accuracy"):
        vals = [r[key] for r in rows
                if r.get(key) is not None and not math.isnan(r[key])]
        out[f"{key}_series_mean"] = float(np.mean(vals)) if vals else None
        out[f"{key}_series_std"] = float(np.std(vals)) if vals else None
        out[f"{key}_series_median"] = float(np.median(vals)) if vals else None
    return out


# --------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------

def prior_only_baseline(y: np.ndarray) -> dict:
    """Always predict the majority class. The floor any real result must clear.

    ROC-AUC of a constant score is 0.5 by definition (no ranking information);
    PR-AUC of a constant score is the positive prevalence.
    """
    y = np.asarray(y).astype(int)
    prev = float(y.mean())
    majority = 1 if prev >= 0.5 else 0
    return {
        "name": "prior_only (always predict majority)",
        "majority_class": majority,
        "prevalence_positive": prev,
        "accuracy": float(max(prev, 1.0 - prev)),
        "roc_auc": 0.5,
        "pr_auc": prev,
        "n": int(len(y)),
    }


def _tissue_via_phase0_chain(df, phase0_analysis) -> Optional[dict]:
    """geo_accession -> coarse tissue, via Phase 0's own normalization chain.

    `read_source_names` + `normalize_tissue` come from
    `qa_generation/build_per_sample_de.py` (imported, never reimplemented — the
    bucket key has to be the same one the DE pipeline used or the numbers do
    not line up), and `coarse_tissue` is Phase 0's reporting grouping on top.
    Returns None on any failure; a missing H5 or a renamed helper degrades the
    baseline, it does not break the evaluation.
    """
    try:
        import logging
        from qa_generation.build_per_sample_de import (  # type: ignore
            normalize_tissue, read_source_names)
        coarse = getattr(phase0_analysis, "coarse_tissue")
        idx = np.asarray(df["sample_index"], dtype=np.int64)
        raw = read_source_names(idx, logging.getLogger("binary_cvd_eval.tissue"))
        tis = [coarse(normalize_tissue(t)) for t in raw]
    except Exception:
        return None
    if not tis or len(tis) != len(df):
        return None
    return {str(g): str(t) for g, t in zip(df["geo_accession"], tis)}


def _load_tissue_labels(df, tissue_path: Optional[Path]) -> Optional[dict]:
    """Tissue per geo_accession, from an explicit file or from Phase 0's
    analysis module if it happens to exist yet.

    `scripts/hypothesis_b/phase0_analysis.py` is being written concurrently, so
    every path here is best-effort: absence degrades to "not available", never
    to a crash and never to a fabricated tissue label.
    """
    import pandas as pd

    if tissue_path is not None:
        p = Path(tissue_path)
        t = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
        cols = {c.lower(): c for c in t.columns}
        acc = cols.get("geo_accession")
        tis = next((cols[c] for c in cols if "tissue" in c), None)
        if acc is None or tis is None:
            return None
        return dict(zip(t[acc].astype(str), t[tis].astype(str)))

    # Phase 0 does not cache a per-sample tissue table — it computes tissue for
    # the TRAINING-eligible pool only, and this evaluation needs the holdout.
    # So reuse its normalization chain (which is itself the DE pipeline's, not
    # a new one) and apply it to the holdout accessions. Every step is
    # best-effort: this module must import and run with none of it present.
    try:
        from scripts.hypothesis_b import phase0_analysis  # type: ignore
    except Exception:
        phase0_analysis = None                                  # type: ignore

    if phase0_analysis is not None and "sample_index" in getattr(df, "columns", []):
        got = _tissue_via_phase0_chain(df, phase0_analysis)
        if got:
            return got

    if phase0_analysis is None:
        return None
    for attr in ("load_tissue_labels", "tissue_labels", "sample_tissue_map",
                 "load_tissue_map"):
        fn = getattr(phase0_analysis, attr, None)
        if fn is None:
            continue
        try:
            got = fn() if callable(fn) else fn
        except Exception:
            continue
        if isinstance(got, dict) and got:
            return {str(k): str(v) for k, v in got.items()}
        if hasattr(got, "columns"):
            cols = {c.lower(): c for c in got.columns}
            acc = cols.get("geo_accession")
            tis = next((cols[c] for c in cols if "tissue" in c), None)
            if acc and tis:
                return dict(zip(got[acc].astype(str), got[tis].astype(str)))
    return None


def tissue_only_baseline(df, y: np.ndarray, tissue_path: Optional[Path] = None) -> dict:
    """How much of the label is predictable from tissue type ALONE.

    This is the confound check flagged in the linear-probe work: if `neg_hard`
    skews to a different tissue mix than the positives, a model can score well
    by learning "tissue" and never touching disease. A high number here does
    not invalidate the LLM result on its own, but it caps how much of it can be
    attributed to disease signal.

    One-hot tissue -> logistic regression, grouped 5-fold by series so tissue
    cannot be memorized per study.
    """
    mapping = _load_tissue_labels(df, tissue_path)
    if not mapping:
        return {"available": False,
                "reason": ("no tissue labels: scripts/hypothesis_b/"
                           "phase0_analysis.py exposes none yet and no "
                           "--tissue-labels file was given"),
                "note": ("without this, the tissue-vs-disease confound is "
                         "UNMEASURED, not ruled out")}

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.preprocessing import OneHotEncoder
    from linear_probe.probe import _fold_metrics

    tis = np.array([mapping.get(str(g), "__unknown__")
                    for g in df["geo_accession"]], dtype=object)
    covered = float((tis != "__unknown__").mean())
    if covered < 0.5:
        return {"available": False,
                "reason": f"tissue labels cover only {covered:.1%} of the holdout"}

    y = np.asarray(y).astype(int)
    groups = df["series_id"].to_numpy()
    try:
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:                      # sklearn < 1.2
        enc = OneHotEncoder(handle_unknown="ignore", sparse=False)

    skf = StratifiedGroupKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)
    rows = []
    for tr, va in skf.split(tis.reshape(-1, 1), y, groups):
        if len(set(y[va].tolist())) < 2:
            continue
        Xtr = enc.fit_transform(tis[tr].reshape(-1, 1))
        Xva = enc.transform(tis[va].reshape(-1, 1))
        clf = LogisticRegression(max_iter=2000, class_weight="balanced",
                                 random_state=SEED)
        clf.fit(Xtr, y[tr])
        rows.append(_fold_metrics(y[va], clf.predict_proba(Xva)[:, 1]))
    if not rows:
        return {"available": False, "reason": "no usable folds"}
    out = {"available": True, "n_folds": len(rows),
           "n_tissues": int(len(set(tis.tolist()))),
           "label_coverage": covered}
    for key in ("roc_auc", "pr_auc", "accuracy"):
        vals = [r[key] for r in rows if r.get(key) is not None]
        out[f"{key}_mean"] = float(np.mean(vals))
        out[f"{key}_std"] = float(np.std(vals))
    return out


def encoder_oof_predictions(embeddings_path: Path, df) -> Optional[np.ndarray]:
    """Out-of-fold P(positive) from the frozen encoder's linear probe.

    The three-way probe file stores fold-level metrics, not per-sample scores,
    and the within-series diagnostic needs per-sample scores. So the probe is
    re-run here on the same population, folds, seed and estimator as
    `run_regen_eval.py`, and only the out-of-fold predictions are kept — no
    sample is ever scored by a model that saw it.

    Features come through `embedding_io.load_embeddings`, which asserts no
    dimension was dropped. Do NOT replace it with a `startswith("e0")` filter.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from eval_binary_comparison.embedding_io import load_embeddings

    import pandas as pd

    path = Path(embeddings_path)
    if not path.exists() or "sample_index" not in getattr(df, "columns", []):
        return None
    try:
        X, ids = load_embeddings(path)
        order = pd.Series(np.arange(ids.size), index=ids)
        sel = order.reindex(np.asarray(df["sample_index"], dtype=np.int64))
        if sel.isna().any():
            return None
        X = X[sel.to_numpy(dtype=np.int64)]
    except Exception:
        return None

    y = df["y"].to_numpy().astype(int)
    groups = df["series_id"].astype(str).to_numpy()
    oof = np.full(len(y), np.nan, dtype=np.float64)
    skf = StratifiedGroupKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)
    for tr, va in skf.split(X, y, groups):
        est = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced",
                               random_state=SEED))
        est.fit(X[tr], y[tr])
        oof[va] = est.predict_proba(X[va])[:, 1]
    if np.isnan(oof).any():
        return None
    return oof


def encoder_linear_probe_baseline(path: Path = ENCODER_BASELINE_JSON) -> dict:
    """The frozen BulkFormer-93M linear probe on THIS holdout split.

    Read from the three-way probe's own output rather than recomputed, so it is
    literally the same number the rest of the report quotes. If that file is
    absent the baseline is reported as unavailable — it is never silently
    replaced with a number computed a different way.
    """
    p = Path(path)
    if not p.exists():
        return {"available": False, "reason": f"{p} not found — run "
                                              f"run_regen_eval.py first"}
    try:
        d = json.loads(p.read_text())
        e = d["populations"]["holdout_clean"]["BulkFormer-93M"]["linear"]
    except Exception as exc:
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
    try:
        source = str(p.relative_to(REPO))
    except ValueError:
        source = str(p)
    return {"available": True,
            "source": source,
            "per_fold_note": ("roc_auc_mean is the mean of per-fold AUCs, the "
                              "form every probe in this project reports"),
            "roc_auc_mean": e.get("roc_auc_mean"),
            "roc_auc_std": e.get("roc_auc_std"),
            "pr_auc_mean": e.get("pr_auc_mean"),
            "pr_auc_std": e.get("pr_auc_std"),
            "note": ("grouped 5-fold CV over the same 92 holdout series; the "
                     "LLM number is a zero-shot forced choice with nothing "
                     "fitted, so the probe has a fitting advantage")}


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def _fmt(v) -> str:
    return "n/a" if v is None else f"{v:.4f}"


def run_all_variants(scorer: AnswerScorer, X: np.ndarray, df,
                     variants: Sequence[PromptVariant]) -> List[dict]:
    y = df["y"].to_numpy().astype(int)
    series = df["series_id"].to_numpy()
    runs = []
    for v in variants:
        for order in v.orders():
            question = v.render(order)
            print(f"\n  [{v.name} / {order}] {question}")
            scores = scorer.score(X, question, [v.positive_answer,
                                                v.negative_answer])
            probs = forced_choice_probabilities(scores)
            run = {
                "variant": v.name,
                "order": order,
                "question": question,
                "positive_answer": v.positive_answer,
                "negative_answer": v.negative_answer,
                "answer_token_counts": {
                    "positive": int(scores.n_tokens[0]),
                    "negative": int(scores.n_tokens[1])},
                "scorings": {},
                "_p": {},
            }
            for tag in ("sum", "mean"):
                p = probs[f"p_yes_{tag}"]
                run["_p"][tag] = p
                rows = per_series_metrics(y, p, series)
                diag = series_shortcut_diagnostics(y, p, series)
                run["scorings"][tag] = {
                    "pooled": binary_metrics(y, p),
                    "series_shortcut_diagnostics": diag,
                    "per_series_summary": aggregate_per_series(rows),
                    "per_series": rows,
                }
                m = run["scorings"][tag]["pooled"]
                w = diag["within_series"]
                print(f"    scoring={tag:<4} POOLED ROC-AUC={m['roc_auc']:.4f} "
                      f"PR-AUC={m['pr_auc']:.4f} acc={m['accuracy']:.4f}")
                print(f"                 WITHIN-SERIES ROC-AUC="
                      f"{_fmt(w['weighted_mean'])} weighted / "
                      f"{_fmt(w['unweighted_mean'])} unweighted "
                      f"({w['n_series_eligible']}/{w['n_series_total']} series) "
                      f"| stratified-pairwise="
                      f"{_fmt(diag['series_centered_pooled']['stratified_pairwise']['roc_auc'])}")
            runs.append(run)
    return runs


def phrasing_sensitivity(runs: List[dict], scoring: str = "sum") -> dict:
    """The spread of ROC-AUC across phrasings and orders.

    Surfaced as a first-class number because a result that moves with the
    wording is a property of the wording.
    """
    vals = {f"{r['variant']}/{r['order']}":
            r["scorings"][scoring]["pooled"]["roc_auc"] for r in runs}
    if not vals:
        return {}
    lo, hi = min(vals.values()), max(vals.values())
    return {"scoring": scoring, "per_run_roc_auc": vals,
            "min": lo, "max": hi, "spread": hi - lo,
            "interpretation": ("a spread comparable to the margin over chance "
                               "means the phrasing, not the profile, is doing "
                               "the work")}


def order_bias(runs: List[dict], scoring: str = "sum") -> dict:
    """Position bias: same question, options swapped.

    Reported per variant as the ROC-AUC gap and the mean shift in P(yes). A
    model insensitive to disease but sensitive to option order will show a
    large shift here and it must not be read as signal.
    """
    out = {}
    by_variant: Dict[str, Dict[str, dict]] = {}
    for r in runs:
        by_variant.setdefault(r["variant"], {})[r["order"]] = r
    for variant, orders in by_variant.items():
        if "positive_first" not in orders or "negative_first" not in orders:
            continue
        a, b = orders["positive_first"], orders["negative_first"]
        ma = a["scorings"][scoring]["pooled"]
        mb = b["scorings"][scoring]["pooled"]
        out[variant] = {
            "roc_auc_positive_first": ma["roc_auc"],
            "roc_auc_negative_first": mb["roc_auc"],
            "roc_auc_gap": ma["roc_auc"] - mb["roc_auc"],
            "mean_p_yes_positive_first": float(np.mean(a["_p"][scoring])),
            "mean_p_yes_negative_first": float(np.mean(b["_p"][scoring])),
            "mean_p_yes_shift": float(np.mean(a["_p"][scoring])
                                      - np.mean(b["_p"][scoring])),
        }
    return {"scoring": scoring, "per_variant": out}


def write_outputs(payload: dict, runs: List[dict], df,
                  json_path: Path, csv_path: Path,
                  predictions_path: Optional[Path] = None) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, default=_jsonable) + "\n")

    # within_series_* sit immediately beside roc_auc so the two are never read
    # apart: a pooled AUC without its within-series companion is exactly the
    # number that cannot distinguish disease signal from batch signature.
    fields = ["row_type", "variant", "order", "scoring", "n", "n_positive",
              "n_negative", "roc_auc", "within_series_auc_weighted",
              "within_series_auc_unweighted", "stratified_pairwise_auc",
              "series_centered_zscore_auc", "pooled_minus_within_series",
              "n_series_eligible", "n_series_skipped",
              "pr_auc", "accuracy", "sensitivity",
              "specificity", "f1", "brier", "question"]

    def diag_cols(d: dict) -> dict:
        if not d or d.get("available") is False:
            return {}
        w_ = d["within_series"]
        return {
            "within_series_auc_weighted": w_["weighted_mean"],
            "within_series_auc_unweighted": w_["unweighted_mean"],
            "stratified_pairwise_auc":
                d["series_centered_pooled"]["stratified_pairwise"]["roc_auc"],
            "series_centered_zscore_auc": d["series_centered_pooled"]["zscore"]["roc_auc"],
            "pooled_minus_within_series": d["pooled_minus_within_series"],
            "n_series_eligible": w_["n_series_eligible"],
            "n_series_skipped": w_["n_series_skipped"],
        }

    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in runs:
            for tag, s in r["scorings"].items():
                w.writerow({"row_type": "llm", "variant": r["variant"],
                            "order": r["order"], "scoring": tag,
                            "question": r["question"], **s["pooled"],
                            **diag_cols(s.get("series_shortcut_diagnostics"))})
        b = payload["baselines"]
        w.writerow({"row_type": "baseline", "variant": "prior_only",
                    "order": "n/a", "scoring": "n/a", **b["prior_only"]})
        if b["encoder_linear_probe"].get("available"):
            e = b["encoder_linear_probe"]
            ed = e.get("series_shortcut_diagnostics") or {}
            row = {"row_type": "baseline",
                   "variant": "BulkFormer-93M linear probe",
                   "order": "n/a", "scoring": "n/a",
                   "roc_auc": ed.get("pooled_roc_auc", e.get("roc_auc_mean")),
                   "pr_auc": ed.get("pooled_pr_auc", e.get("pr_auc_mean"))}
            row.update(diag_cols(ed))
            w.writerow(row)
        if b["tissue_only"].get("available"):
            t = b["tissue_only"]
            w.writerow({"row_type": "baseline", "variant": "tissue_only",
                        "order": "n/a", "scoring": "n/a",
                        "roc_auc": t["roc_auc_mean"], "pr_auc": t["pr_auc_mean"],
                        "accuracy": t["accuracy_mean"]})

    if predictions_path is not None:
        with open(predictions_path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["geo_accession", "series_id", "true_label", "variant",
                        "order", "scoring", "p_yes"])
            for r in runs:
                for tag, p in r["_p"].items():
                    for g, s, yy, pp in zip(df["geo_accession"], df["series_id"],
                                            df["y"], p):
                        w.writerow([g, s, int(yy), r["variant"], r["order"],
                                    tag, f"{pp:.6f}"])


def _jsonable(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(f"not JSON serializable: {type(o)}")


def make_plot(runs: List[dict], df, payload: dict, path: Path,
              scoring: str = "sum") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    y = df["y"].to_numpy().astype(int)
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 5.0))

    for r in runs:
        p = r["_p"][scoring]
        fpr, tpr, _ = roc_curve(y, p)
        auc = r["scorings"][scoring]["pooled"]["roc_auc"]
        ax[0].plot(fpr, tpr, lw=1.4,
                   label=f"{r['variant']}/{r['order']} (AUC={auc:.3f})")
    ax[0].plot([0, 1], [0, 1], "k--", lw=0.8, label="chance (0.500)")
    enc = payload["baselines"]["encoder_linear_probe"]
    title = f"ROC — holdout binary CVD, scoring={scoring}"
    if enc.get("available"):
        title += f"\nfrozen BulkFormer-93M linear probe = {enc['roc_auc_mean']:.3f}"
    ax[0].set_title(title, fontsize=10)
    ax[0].set_xlabel("FPR")
    ax[0].set_ylabel("TPR")
    ax[0].legend(fontsize=7, loc="lower right")

    # Panel 2: pooled vs WITHIN-SERIES, side by side. The gap between the two
    # bars of a pair is the batch-shortcut measurement, and it is the point of
    # the figure — so the two are never plotted apart.
    labels, pooled_v, within_v = [], [], []
    for r in runs:
        labels.append(f"{r['variant']}\n{r['order']}")
        s = r["scorings"][scoring]
        pooled_v.append(s["pooled"]["roc_auc"])
        d = s.get("series_shortcut_diagnostics") or {}
        wm = (d.get("within_series") or {}).get("weighted_mean")
        within_v.append(np.nan if wm is None else wm)

    xs = np.arange(len(labels))
    ax[1].bar(xs - 0.19, pooled_v, width=0.36, color="#4878A8", label="pooled")
    ax[1].bar(xs + 0.19, within_v, width=0.36, color="#6ACC64",
              label="within-series (weighted)")
    ax[1].axhline(0.5, color="k", ls="--", lw=0.8, label="chance")

    ed = (enc.get("series_shortcut_diagnostics") or {}) if enc.get("available") else {}
    if ed and ed.get("available") is not False:
        ax[1].axhline(ed["pooled_roc_auc"], color="#C44E52", ls="-", lw=1.3,
                      label=f"encoder pooled ({ed['pooled_roc_auc']:.3f})")
        ew = ed["within_series"]["weighted_mean"]
        if ew is not None:
            ax[1].axhline(ew, color="#C44E52", ls="-.", lw=1.6,
                          label=f"ENCODER WITHIN-SERIES ({ew:.3f}) — the bar")
    elif enc.get("available") and enc.get("roc_auc_mean") is not None:
        ax[1].axhline(enc["roc_auc_mean"], color="#C44E52", ls="-", lw=1.2,
                      label=f"encoder probe ({enc['roc_auc_mean']:.3f})")
    tb = payload["baselines"]["tissue_only"]
    if tb.get("available"):
        ax[1].axhline(tb["roc_auc_mean"], color="#DD8452", ls=":", lw=1.2,
                      label=f"tissue-only ({tb['roc_auc_mean']:.3f})")
    ax[1].set_xticks(xs)
    ax[1].set_xticklabels(labels, fontsize=7, rotation=30, ha="right")
    ax[1].set_ylim(0.0, 1.0)
    ax[1].set_ylabel("ROC-AUC")
    ax[1].set_title("Pooled vs within-series ROC-AUC\n"
                    "(pooled high + within ~0.5 = series/batch shortcut)",
                    fontsize=10)
    ax[1].legend(fontsize=6.5, loc="upper right")

    if payload.get("debug_scorer_not_a_real_result"):
        fig.suptitle("DEBUG SCORER — NOT A REAL RESULT", color="red",
                     fontsize=13, y=0.995)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _headline(runs: List[dict], encoder: dict, baselines: dict,
              scoring: str = "sum") -> dict:
    """The four numbers that have to be read together, hoisted to the top.

    Anyone quoting one of these without the others is quoting a number that
    cannot mean what they think it means.
    """
    best = None
    for r in runs:
        d = r["scorings"][scoring].get("series_shortcut_diagnostics") or {}
        if not d:
            continue
        if best is None or d["pooled_roc_auc"] > best[1]["pooled_roc_auc"]:
            best = (f"{r['variant']}/{r['order']}", d)
    ed = (encoder.get("series_shortcut_diagnostics") or {}) \
        if encoder.get("available") else {}
    out = {
        "scoring": scoring,
        "prior_only_accuracy": baselines["prior_only"]["accuracy"],
        "tissue_only_roc_auc": (baselines["tissue_only"].get("roc_auc_mean")
                                if baselines["tissue_only"].get("available")
                                else None),
        "encoder_pooled_roc_auc": ed.get("pooled_roc_auc"),
        "encoder_within_series_roc_auc": (
            (ed.get("within_series") or {}).get("weighted_mean")),
        "note": ("within-series AUC is the batch-shortcut-proof number: the "
                 "Stage-2 training population has ZERO mixed-class series "
                 "(388 positive-only, 1,174 negative-only), so series_id is a "
                 "perfect training-label predictor and pooled AUC alone cannot "
                 "rule out that the model learned batch signature"),
    }
    if best is not None:
        out["best_llm_run_by_pooled"] = best[0]
        out["llm_pooled_roc_auc"] = best[1]["pooled_roc_auc"]
        out["llm_within_series_roc_auc"] = best[1]["within_series"]["weighted_mean"]
        out["llm_pooled_minus_within_series"] = best[1]["pooled_minus_within_series"]
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    # Checkpoint args mirror extract_llm_latents.py / matched_binary_eval.py.
    ap.add_argument("--lora-ckpt",
                    help="Stage-2 LoRA checkpoint dir; it carries the base "
                         "Vicuna-7B path and the Stage-1 connector in its "
                         "config, exactly as extract_llm_latents.py assumes")
    ap.add_argument("--conv-version", default="llama",
                    help="this repo's name for the Vicuna v1.5 format")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cuda")

    ap.add_argument("--labels", type=Path, default=PROBE_LABELS)
    ap.add_argument("--holdout", type=Path, default=HOLDOUT_JSON)
    ap.add_argument("--stage2-json", type=Path, default=STAGE2_JSON)
    ap.add_argument("--encoded-dir", type=Path, default=ENCODED_DIR)
    ap.add_argument("--encoder-baseline-json", type=Path,
                    default=ENCODER_BASELINE_JSON)
    ap.add_argument("--encoder-embeddings", type=Path,
                    default=ENCODER_EMBEDDINGS,
                    help="515-d frozen-encoder parquet; re-probed out-of-fold "
                         "so the encoder gets the same within-series diagnostic")
    ap.add_argument("--tissue-labels", type=Path, default=None,
                    help="optional parquet/csv with geo_accession + tissue; "
                         "otherwise scripts/hypothesis_b/phase0_analysis.py is "
                         "probed for one and the baseline degrades if absent")

    ap.add_argument("--prompt-variant", action="append", default=None,
                    choices=sorted(PROMPT_VARIANTS) + ["all"],
                    help="repeatable; default all")
    ap.add_argument("--limit", type=int, default=0, help="debug: first N samples")

    ap.add_argument("--tables-dir", type=Path, default=OUT_TABLES)
    ap.add_argument("--plots-dir", type=Path, default=OUT_PLOTS)
    ap.add_argument("--out-tag", default="binary_cvd_eval")

    ap.add_argument("--debug-scorer", choices=["random", "leaky_oracle"],
                    default=None,
                    help="CPU pipeline smoke test. Produces NO usable number "
                         "and stamps the output as such.")
    ap.add_argument("--allow-missing-embeddings", action="store_true",
                    help="only permitted with --debug-scorer")
    return ap


def main(argv=None, scorer: Optional[AnswerScorer] = None) -> int:
    """`scorer` is the seam the test suite uses: pass an AnswerScorer and the
    entire pipeline runs with no model, no GPU and no transformers install."""
    args = build_parser().parse_args(argv)

    injected = scorer is not None
    if args.allow_missing_embeddings and not (args.debug_scorer or injected):
        raise SystemExit("--allow-missing-embeddings is only permitted with "
                         "--debug-scorer; a real run must score real vectors")
    if not args.debug_scorer and not injected and not args.lora_ckpt:
        raise SystemExit("--lora-ckpt is required (or use --debug-scorer)")

    names = args.prompt_variant or ["all"]
    if "all" in names:
        names = sorted(PROMPT_VARIANTS)
    variants = [PROMPT_VARIANTS[n] for n in names]

    print("=" * 78)
    print("PHASE 4.3 — HELD-OUT BINARY CVD-vs-CONTROL, FORCED-CHOICE LOG-PROB")
    print("=" * 78)

    df, guards = load_holdout_population(args.labels, args.holdout,
                                         args.stage2_json)
    if args.limit:
        df = df.iloc[:args.limit].reset_index(drop=True)
        guards["series_class_balance"] = assert_series_have_both_classes(
            df["series_id"].to_numpy(), df["y"].to_numpy())
    y = df["y"].to_numpy().astype(int)
    print(f"holdout population: n={len(df)} pos={int(y.sum())} "
          f"neg={int((1 - y).sum())} series={df['series_id'].nunique()}")
    print(f"guard: zero overlap with {Path(args.stage2_json).name} — asserted")
    print(f"guard: every contributing series has both classes — asserted")

    X, emb_stats = load_encoded_embeddings(df["geo_accession"], args.encoded_dir,
                                           allow_missing=args.allow_missing_embeddings)
    print(f"embeddings: {X.shape} (missing {emb_stats['n_missing']})")

    if injected:
        pass
    elif args.debug_scorer:
        print("\n*** DEBUG SCORER — THE NUMBERS BELOW ARE NOT A RESULT ***\n")
        scorer = _DebugScorer(args.debug_scorer, y=y)
    else:
        from evaluate_and_compare import load_model
        model, tokenizer = load_model(args.lora_ckpt, device=args.device)
        scorer = TinyLlavaScorer(model, tokenizer, args.conv_version,
                                 args.batch_size)

    runs = run_all_variants(scorer, X, df, variants)

    # The encoder gets the SAME diagnostic, on the same samples and the same
    # series, or the comparison is not like-for-like. Its within-series AUC is
    # the bar the LLM has to clear.
    series = df["series_id"].to_numpy()
    encoder = encoder_linear_probe_baseline(args.encoder_baseline_json)
    enc_oof = encoder_oof_predictions(args.encoder_embeddings, df)
    if enc_oof is not None:
        encoder["available"] = True
        encoder["oof_source"] = str(args.encoder_embeddings)
        encoder["series_shortcut_diagnostics"] = series_shortcut_diagnostics(
            y, enc_oof, series)
        encoder["oof_note"] = (
            "computed here from out-of-fold predictions of the same probe "
            "(same population, folds, seed, estimator as run_regen_eval.py); "
            "pooled_roc_auc is over pooled OOF scores and so differs slightly "
            "from the mean-of-folds roc_auc_mean above")
    else:
        encoder["series_shortcut_diagnostics"] = {
            "available": False,
            "reason": ("encoder embeddings not loadable, so the encoder's "
                       "within-series AUC — the bar the LLM must clear — is "
                       "UNMEASURED")}

    payload = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "phase": "stage_2_revisions.md Phase 4.3",
        "scorer": scorer.name,
        "debug_scorer_not_a_real_result": bool(scorer.is_debug),
        "lora_ckpt": args.lora_ckpt,
        "conv_version": args.conv_version,
        "scoring_method": {
            "forced_choice": ("softmax over the total log-probability of the "
                              "affirmative vs the negative answer continuation; "
                              "no generation, no string matching"),
            "multi_token": ("logP(answer|prompt) = sum_j logP(t_j|prompt,t_<j), "
                            "by the chain rule over every answer token, "
                            "recovered by differencing the prompt and "
                            "prompt+answer tokenizations"),
            "variants_reported": ["sum (raw sequence log-likelihood)",
                                  "mean (length-normalized, per token)"],
            "natural_threshold": 0.5,
        },
        "population": {
            "n": int(len(df)),
            "n_positive": int(y.sum()),
            "n_negative": int((1 - y).sum()),
            "n_series": int(df["series_id"].nunique()),
            "embeddings": emb_stats,
        },
        "guards": guards,
        "baselines": {
            "prior_only": prior_only_baseline(y),
            "tissue_only": tissue_only_baseline(df, y, args.tissue_labels),
            "encoder_linear_probe": encoder,
        },
        "phrasing_sensitivity": {s: phrasing_sensitivity(runs, s)
                                 for s in ("sum", "mean")},
        "order_bias": {s: order_bias(runs, s) for s in ("sum", "mean")},
        "runs": [{k: v for k, v in r.items() if k != "_p"} for r in runs],
    }
    payload["headline"] = _headline(runs, encoder, payload["baselines"])

    tables = Path(args.tables_dir)
    plots = Path(args.plots_dir)
    tables.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)
    json_path = tables / f"{args.out_tag}.json"
    csv_path = tables / f"{args.out_tag}.csv"
    pred_path = tables / f"{args.out_tag}_predictions.csv"
    write_outputs(payload, runs, df, json_path, csv_path, pred_path)
    print(f"\nwrote {json_path}")
    print(f"wrote {csv_path}")
    print(f"wrote {pred_path}")

    try:
        plot_path = plots / f"{args.out_tag}.png"
        make_plot(runs, df, payload, plot_path)
        print(f"wrote {plot_path}")
    except Exception as exc:                       # a plot is never load-bearing
        print(f"[warn] plot skipped: {type(exc).__name__}: {exc}")

    b = payload["baselines"]
    print("\n--- baselines ---")
    print(f"  prior-only accuracy : {b['prior_only']['accuracy']:.4f} "
          f"(ROC-AUC 0.500 by construction)")
    if b["encoder_linear_probe"].get("available"):
        e = b["encoder_linear_probe"]
        print(f"  frozen encoder probe: ROC-AUC {_fmt(e.get('roc_auc_mean'))} "
              f"(mean of folds)")
        ed = e.get("series_shortcut_diagnostics") or {}
        if ed.get("available") is not False:
            print(f"      pooled OOF        : {_fmt(ed['pooled_roc_auc'])}")
            print(f"      WITHIN-SERIES     : "
                  f"{_fmt(ed['within_series']['weighted_mean'])} weighted / "
                  f"{_fmt(ed['within_series']['unweighted_mean'])} unweighted "
                  f"<-- THE BAR THE LLM MUST CLEAR")
            print(f"      series-free pooled: "
                  f"stratified-pairwise "
                  f"{_fmt(ed['series_centered_pooled']['stratified_pairwise']['roc_auc'])} / "
                  f"z {_fmt(ed['series_centered_pooled']['zscore']['roc_auc'])}")
            print(f"      pooled - within   : "
                  f"{_fmt(ed['pooled_minus_within_series'])}")
        else:
            print(f"      WITHIN-SERIES     : UNAVAILABLE — {ed.get('reason')}")
    else:
        print(f"  frozen encoder probe: UNAVAILABLE — "
              f"{b['encoder_linear_probe']['reason']}")
    if b["tissue_only"].get("available"):
        print(f"  tissue-only         : ROC-AUC {b['tissue_only']['roc_auc_mean']:.4f}")
    else:
        print(f"  tissue-only         : UNAVAILABLE — {b['tissue_only']['reason']}")
    ps = payload["phrasing_sensitivity"]["sum"]
    print(f"\nphrasing spread (sum scoring): {ps.get('spread', float('nan')):.4f} "
          f"[{ps.get('min', float('nan')):.4f}, {ps.get('max', float('nan')):.4f}]")
    if scorer.is_debug:
        print("\n*** DEBUG SCORER — DISCARD THESE NUMBERS ***")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
