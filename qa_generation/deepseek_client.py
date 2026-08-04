"""Async DeepSeek V4-Flash client for Step 4 of the QA generation pipeline.

Takes the `(filled_question, deterministic_answer)` pairs produced by Step 3 and
asks DeepSeek to paraphrase them, preserving every fact verbatim. The model is a
rewriter here, never a source of new information.

Design notes (see `deepseek_api_notes.md` for the sourced API findings):

* **Concurrency, not RPM/TPM.** DeepSeek publishes no requests-per-minute or
  tokens-per-minute limit; it publishes a *concurrency* limit (2500 in-flight
  requests per account for ``deepseek-v4-flash``) and returns HTTP 429 above it.
  So the throttle here is a semaphore, with an optional RPM cap left available
  but off by default.
* **Canonical prompts.** The generation prompts live in
  ``prompts/stage{1,2}_generation_prompt.txt`` and are loaded verbatim. Those
  files are authoritative — this module must never carry its own copy.
* **Cacheable prefix.** Context caching is on by default and keys off an exact
  prefix match. Each prompt is fixed above its ``Given:`` block, and all
  per-sample values are substituted inside that block at the very end, so the
  long shared prefix caches. Nothing per-sample may move above it.
* **SKIP is terminal.** A ``SKIP:`` response means the ground truth was missing.
  It is recorded as-is and never retried — retrying it is exactly how a
  fabricated answer would enter the corpus.

Usage::

    export DEEPSEEK_API_KEY=...        # never hardcoded, never logged
    python -m qa_generation.deepseek_client \
        --input  qa_generation/filled_pairs_stage1.jsonl \
        --output qa_generation/generated_pairs_stage1.jsonl \
        --stage 1

Input JSONL, one object per line::

    {"id": "GSM1234567__direct_abundance_query__0007",
     "stage": 1,
     "category": "direct_abundance_query",
     "templated_question": "Report the expression level of {gene}.",
     "filled_question": "Report the expression level of NPPA.",
     "gt_answer": "The expression level of NPPA in this sample is 8.42 (log2-CPM).",
     "image": "GSM1234567.npy"}

``templated_question`` and ``category`` fill the canonical prompt's ``Given:``
block; if ``templated_question`` is absent the filled question is used instead.

``gt_answer`` may be ``insufficient_data`` or ``(none provided)``, in which case
the prompt instructs the model to return ``SKIP:`` and the client refuses to
accept anything else.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

import httpx

LOGGER = logging.getLogger("deepseek_client")

# --------------------------------------------------------------------------- #
# API constants
# --------------------------------------------------------------------------- #

API_BASE_URL = "https://api.deepseek.com"
CHAT_COMPLETIONS_PATH = "/chat/completions"
DEFAULT_MODEL = "deepseek-v4-flash"
API_KEY_ENV_VAR = "DEEPSEEK_API_KEY"

#: Published concurrency ceiling for ``deepseek-v4-flash``, account-wide across
#: all API keys. We stay well under it by default — see ``--concurrency``.
PUBLISHED_CONCURRENCY_LIMIT = 2500

#: USD per token. Source: https://api-docs.deepseek.com/quick_start/pricing
#: Checked 2026-08-03. DeepSeek reserves the right to change these; a peak/
#: off-peak policy (2x during 09:00-12:00 and 14:00-18:00 Beijing time) is
#: announced but not yet in effect. Re-check before a full-scale run.
PRICING_CHECKED = "2026-08-03"
PRICE_PER_TOKEN = {
    "deepseek-v4-flash": {
        "cache_hit": 0.0028 / 1_000_000,
        "cache_miss": 0.14 / 1_000_000,
        "output": 0.28 / 1_000_000,
    },
    "deepseek-v4-pro": {
        "cache_hit": 0.003625 / 1_000_000,
        "cache_miss": 0.435 / 1_000_000,
        "output": 0.87 / 1_000_000,
    },
}

#: Errors worth retrying. 429 = concurrency ceiling, 500/503 = server-side.
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
#: Errors that will never succeed on retry — fail the item immediately.
TERMINAL_STATUS = frozenset({400, 401, 402, 422})

# --------------------------------------------------------------------------- #
# The generation prompts — the cacheable prefix.
#
# `prompts/stage{1,2}_generation_prompt.txt` are the authoritative source and are
# loaded verbatim; do not paraphrase, reformat, or re-inline them here.
#
# Everything above each prompt's `Given:` block must stay byte-identical across
# every call in a run. Editing a prompt file mid-run silently invalidates the
# cache for every subsequent request (and multiplies input cost by 50x), so treat
# them as frozen once a run starts.
# --------------------------------------------------------------------------- #

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
STAGE_PROMPT_FILES = {
    1: PROMPTS_DIR / "stage1_generation_prompt.txt",
    2: PROMPTS_DIR / "stage2_generation_prompt.txt",
}


def _load_prompt(stage: int) -> str:
    path = STAGE_PROMPT_FILES[stage]
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(
            "canonical Stage {} prompt is missing or unreadable at {}. It is the "
            "authoritative source for the generation prompt and is not duplicated "
            "in this module — restore the file rather than inlining the text."
            .format(stage, path)
        ) from exc


#: Verbatim contents of the canonical prompt files. Loaded rather than inlined so
#: the files remain the single source of truth and the constants cannot drift
#: from them again.
STAGE1_INSTRUCTIONS = _load_prompt(1)
STAGE2_INSTRUCTIONS = _load_prompt(2)
STAGE_INSTRUCTIONS = {1: STAGE1_INSTRUCTIONS, 2: STAGE2_INSTRUCTIONS}

#: The `Given:` block's placeholders — the only ones substituted per call. The
#: prompts also contain literal `{gene}`, `{gene_A}`, `{gene_B}`,
#: `{comparison_group}` and `{condition}` inside their worked examples and
#: restriction text, which must survive into the sent prompt untouched. That is
#: why substitution is targeted replacement and NOT str.format(), which would
#: raise KeyError on those.
STAGE_ANSWER_PLACEHOLDER = {1: "{deterministic_answer}", 2: "{gold_answer}"}

INSUFFICIENT_DATA = "insufficient_data"
SKIP_PREFIX = "SKIP:"
#: Ground-truth values that mean "no answer was computed". The canonical Stage 1
#: prompt writes a missing answer as `(none provided)`; Stage 2 uses
#: `insufficient_data`. Either must produce a skip, never a paraphrase.
MISSING_GT_VALUES = frozenset({"", INSUFFICIENT_DATA, "(none provided)", "none provided"})


#: Per-category output-token caps, overriding `max_tokens` for that category only.
#:
#: `ranking_ordering_query` answers restate a ranked gene list that can run to 290
#: genes (Step 2 widened those windows deliberately, to break the mitochondrial
#: dominance measured in its Task 5 — narrowing them again would undo that fix, so
#: the token limit moves instead). Its longest question+answer is 4,918 characters;
#: at a deliberately pessimistic 2.0 chars/token that is 2,459 output tokens, so
#: 3,000 leaves ~22% headroom. Truncating a ranked list turns a correct answer into
#: a wrong one, which is the same failure class as the threshold_query bug.
#:
#: Every other category's longest question+answer is 653 characters (~326 tokens at
#: the same pessimistic ratio), so they stay on the tighter default cap.
MAX_TOKENS_BY_CATEGORY = {
    # Raised 3000 -> 4096 after the full run: 17 of 39,954 responses (0.04%)
    # hit the 3000 cap and were truncated mid-list. The pilot's observed max was
    # 2,705 tokens, but the full-corpus tail runs higher — the longest ground
    # truths here are ~4,780 characters (290-gene percentile bands) and the model
    # expands them to "GENE at VALUE" prose, which costs more tokens than the
    # source. 4096 clears the longest observed completion with headroom.
    "ranking_ordering_query": 4096,
}


def is_missing_gt(gt_answer: str) -> bool:
    return (gt_answer or "").strip().lower() in MISSING_GT_VALUES


def build_messages(
    stage: int,
    filled_question: str,
    gt_answer: str,
    category: str = "",
    templated_question: str = "",
) -> List[Dict[str, str]]:
    """Fill the canonical prompt's ``Given:`` block and send it as one message.

    The canonical prompts are self-contained: role framing, restrictions, worked
    examples, the ``Given:`` block and the output format are one document, so it
    is sent as a single user message rather than being split into system/user
    parts. Everything above ``Given:`` is byte-identical across calls, so the
    prefix cache still hits — verified against the live API.
    """
    try:
        prompt = STAGE_INSTRUCTIONS[stage]
        answer_placeholder = STAGE_ANSWER_PLACEHOLDER[stage]
    except KeyError:
        raise ValueError("stage must be 1 or 2, got {!r}".format(stage))

    filled = prompt
    for placeholder, value in (
        ("{category}", category),
        ("{templated_question}", templated_question),
        ("{filled_question}", filled_question),
        (answer_placeholder, gt_answer),
    ):
        filled = filled.replace(placeholder, value)
    return [{"role": "user", "content": filled}]


# --------------------------------------------------------------------------- #
# Request / result records
# --------------------------------------------------------------------------- #


@dataclass
class GenerationRequest:
    """One (question, ground-truth answer) pair awaiting paraphrase."""

    id: str
    stage: int
    filled_question: str
    gt_answer: str
    category: Optional[str] = None
    #: The unfilled template, e.g. "Report the expression level of {gene}." The
    #: canonical prompts' `Given:` block asks for this alongside the filled
    #: question; Step 3 emits it as part of its `(templated_question,
    #: filled_question, deterministic_answer)` triples.
    templated_question: Optional[str] = None
    image: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, obj: Dict[str, Any]) -> "GenerationRequest":
        known = {"id", "stage", "filled_question", "gt_answer", "category",
                 "templated_question", "image"}
        missing = [k for k in ("id", "stage", "filled_question", "gt_answer") if k not in obj]
        if missing:
            raise ValueError("input record missing required field(s): {}".format(missing))
        return cls(
            id=str(obj["id"]),
            stage=int(obj["stage"]),
            filled_question=obj["filled_question"],
            gt_answer=obj["gt_answer"],
            category=obj.get("category"),
            templated_question=obj.get("templated_question"),
            image=obj.get("image"),
            extra={k: v for k, v in obj.items() if k not in known},
        )


@dataclass
class Usage:
    """Token accounting for one call, straight from the API's usage object."""

    cache_hit_tokens: int = 0
    cache_miss_tokens: int = 0
    output_tokens: int = 0

    @classmethod
    def from_api(cls, usage: Optional[Dict[str, Any]]) -> "Usage":
        usage = usage or {}
        hit = int(usage.get("prompt_cache_hit_tokens", 0) or 0)
        # Older/compat responses may omit the cache split; fall back to treating
        # the whole prompt as a miss so cost is never under-reported.
        if "prompt_cache_miss_tokens" in usage:
            miss = int(usage.get("prompt_cache_miss_tokens") or 0)
        else:
            miss = max(int(usage.get("prompt_tokens", 0) or 0) - hit, 0)
        return cls(
            cache_hit_tokens=hit,
            cache_miss_tokens=miss,
            output_tokens=int(usage.get("completion_tokens", 0) or 0),
        )

    def cost_usd(self, model: str) -> float:
        rates = PRICE_PER_TOKEN.get(model)
        if rates is None:
            return 0.0
        return (
            self.cache_hit_tokens * rates["cache_hit"]
            + self.cache_miss_tokens * rates["cache_miss"]
            + self.output_tokens * rates["output"]
        )

    def as_dict(self) -> Dict[str, int]:
        return {
            "cache_hit_tokens": self.cache_hit_tokens,
            "cache_miss_tokens": self.cache_miss_tokens,
            "output_tokens": self.output_tokens,
        }


@dataclass
class GenerationResult:
    """What gets appended to the checkpoint file, one JSON object per line."""

    id: str
    status: str  # "ok" | "skip" | "error"
    stage: int
    category: Optional[str]
    image: Optional[str]
    filled_question: str
    gt_answer: str
    paraphrased_question: Optional[str] = None
    paraphrased_answer: Optional[str] = None
    skip_reason: Optional[str] = None
    parse_anomaly: Optional[str] = None
    error: Optional[str] = None
    raw_response: Optional[str] = None
    usage: Usage = field(default_factory=Usage)
    attempts: int = 0
    latency_s: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        out = {
            "id": self.id,
            "status": self.status,
            "stage": self.stage,
            "category": self.category,
            "image": self.image,
            "filled_question": self.filled_question,
            "gt_answer": self.gt_answer,
            "paraphrased_question": self.paraphrased_question,
            "paraphrased_answer": self.paraphrased_answer,
            "skip_reason": self.skip_reason,
            "parse_anomaly": self.parse_anomaly,
            "error": self.error,
            "usage": self.usage.as_dict(),
            "attempts": self.attempts,
            "latency_s": round(self.latency_s, 3),
        }
        if self.status != "ok":
            out["raw_response"] = self.raw_response
        return out


# --------------------------------------------------------------------------- #
# Response parsing
# --------------------------------------------------------------------------- #


def _strip_label_markup(line: str) -> str:
    """Remove Markdown emphasis/heading decoration around a Question:/Answer: label.

    "**Question:**  text" -> "Question: text". Only the leading label is touched;
    emphasis inside the answer body is left alone.
    """
    stripped = line.lstrip()
    stripped = re.sub(r"^#{1,6}\s*", "", stripped)
    m = re.match(r"^(\*{1,3}|_{1,3})?\s*(QUESTION|ANSWER)\s*\1?\s*:\s*\1?\s*",
                 stripped, flags=re.IGNORECASE)
    if not m:
        return line
    label = m.group(2)
    return "{}: {}".format(label.capitalize(), stripped[m.end():].lstrip())


def parse_completion(text: str, gt_answer: str = "") -> Tuple[str, Dict[str, Optional[str]]]:
    """Parse the model's two-line output.

    Returns ``(status, fields)`` where status is ``"ok"``, ``"skip"``, or
    ``"error"``. A ``SKIP:`` response is recognised before anything else and is
    never treated as a malformed completion.

    ``gt_answer`` is used as a guard, not a hint: when the ground truth was
    ``insufficient_data`` the only acceptable outcome is a skip. Observed in the
    connectivity test, the model sometimes wraps the skip inside the normal
    two-line format (``ANSWER: SKIP: ...``); without this guard that would be
    accepted as a genuine QA pair and land in the corpus.
    """
    stripped = (text or "").strip()
    gt_missing = is_missing_gt(gt_answer)

    if not stripped:
        return "error", {"error": "empty completion"}

    if stripped.upper().startswith(SKIP_PREFIX):
        return "skip", {"skip_reason": stripped[len(SKIP_PREFIX):].strip() or None}

    question = answer = None
    current = None
    for raw_line in stripped.splitlines():
        # The model intermittently decorates the labels with Markdown emphasis
        # or a heading ("**Question:**", "### Answer:"). Measured at 0.07% of
        # calls in the checkpoint-1 slice, always with correct content behind
        # it. Strip the decoration before matching rather than discard a good
        # completion over formatting.
        line = _strip_label_markup(raw_line)
        upper = line.upper()
        if upper.startswith("QUESTION:"):
            current = "q"
            question = line.split(":", 1)[1].strip()
        elif upper.startswith("ANSWER:"):
            current = "a"
            answer = line.split(":", 1)[1].strip()
        elif current == "q" and question is not None:
            question = (question + " " + line.strip()).strip()
        elif current == "a" and answer is not None:
            answer = (answer + " " + line.strip()).strip()

    if question and answer:
        # A skip smuggled inside the two-line format is still a skip.
        if answer.upper().startswith(SKIP_PREFIX):
            return "skip", {
                "skip_reason": answer[len(SKIP_PREFIX):].strip() or None,
                "parse_anomaly": "SKIP returned inside an Answer: line",
            }
        if gt_missing:
            return "skip", {
                "skip_reason": INSUFFICIENT_DATA,
                "parse_anomaly": (
                    "ground truth was missing but the model returned a "
                    "paraphrase; discarded rather than retried"
                ),
            }
        return "ok", {"paraphrased_question": question, "paraphrased_answer": answer}

    if gt_missing:
        return "skip", {"skip_reason": INSUFFICIENT_DATA,
                        "parse_anomaly": "malformed completion on missing-ground-truth input"}
    return "error", {"error": "malformed completion: expected Question:/Answer: lines"}


# --------------------------------------------------------------------------- #
# Running cost / usage accounting
# --------------------------------------------------------------------------- #


class UsageTracker:
    """Cumulative token and cost totals, logged periodically."""

    def __init__(self, model: str, log_every: int = 100) -> None:
        self.model = model
        self.log_every = log_every
        self.calls = 0
        self.ok = 0
        self.skipped = 0
        self.errors = 0
        self.cache_hit_tokens = 0
        self.cache_miss_tokens = 0
        self.output_tokens = 0
        self.cost_usd = 0.0
        self.started = time.monotonic()

    def record(self, result: GenerationResult) -> None:
        self.calls += 1
        if result.status == "ok":
            self.ok += 1
        elif result.status == "skip":
            self.skipped += 1
        else:
            self.errors += 1

        u = result.usage
        self.cache_hit_tokens += u.cache_hit_tokens
        self.cache_miss_tokens += u.cache_miss_tokens
        self.output_tokens += u.output_tokens
        call_cost = u.cost_usd(self.model)
        self.cost_usd += call_cost

        LOGGER.debug(
            "%s status=%s in=%d(hit=%d) out=%d cost=$%.6f attempts=%d %.2fs",
            result.id,
            result.status,
            u.cache_hit_tokens + u.cache_miss_tokens,
            u.cache_hit_tokens,
            u.output_tokens,
            call_cost,
            result.attempts,
            result.latency_s,
        )
        if self.log_every and self.calls % self.log_every == 0:
            LOGGER.info("progress: %s", self.summary_line())

    @property
    def cache_hit_rate(self) -> float:
        total_in = self.cache_hit_tokens + self.cache_miss_tokens
        return (self.cache_hit_tokens / total_in) if total_in else 0.0

    def summary_line(self) -> str:
        elapsed = max(time.monotonic() - self.started, 1e-9)
        return (
            "{calls} calls ({ok} ok / {skip} skip / {err} err) | "
            "in {inp:,} tok (cache hit {rate:.1%}) | out {out:,} tok | "
            "${cost:.4f} running | {rps:.1f} calls/s"
        ).format(
            calls=self.calls,
            ok=self.ok,
            skip=self.skipped,
            err=self.errors,
            inp=self.cache_hit_tokens + self.cache_miss_tokens,
            rate=self.cache_hit_rate,
            out=self.output_tokens,
            cost=self.cost_usd,
            rps=self.calls / elapsed,
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "calls": self.calls,
            "ok": self.ok,
            "skipped": self.skipped,
            "errors": self.errors,
            "cache_hit_tokens": self.cache_hit_tokens,
            "cache_miss_tokens": self.cache_miss_tokens,
            "output_tokens": self.output_tokens,
            "cache_hit_rate": round(self.cache_hit_rate, 4),
            "cost_usd": round(self.cost_usd, 6),
            "pricing_checked": PRICING_CHECKED,
            "elapsed_s": round(time.monotonic() - self.started, 1),
        }


# --------------------------------------------------------------------------- #
# Incremental checkpointing
# --------------------------------------------------------------------------- #


class JsonlCheckpoint:
    """Append-only JSONL sink, flushed per record so a crash loses nothing.

    Also indexes the ids already present so an interrupted run resumes instead of
    re-paying for completed calls.
    """

    def __init__(self, path: Path, fsync_every: int = 25) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fsync_every = fsync_every
        self._since_sync = 0
        self._lock = asyncio.Lock()
        self.completed_ids: Set[str] = self._scan_completed()
        self._fh = self.path.open("a", encoding="utf-8")

    def _scan_completed(self) -> Set[str]:
        """Ids already written with a terminal status (``ok`` or ``skip``).

        Errors are deliberately not counted as done, so a resume retries them.
        """
        done: Set[str] = set()
        if not self.path.exists():
            return done
        with self.path.open("r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    LOGGER.warning("checkpoint %s line %d is not valid JSON; ignoring",
                                   self.path, lineno)
                    continue
                if rec.get("status") in ("ok", "skip") and "id" in rec:
                    done.add(str(rec["id"]))
        return done

    async def write(self, result: GenerationResult) -> None:
        line = json.dumps(result.as_dict(), ensure_ascii=False)
        async with self._lock:
            self._fh.write(line + "\n")
            self._fh.flush()
            self._since_sync += 1
            if self.fsync_every and self._since_sync >= self.fsync_every:
                os.fsync(self._fh.fileno())
                self._since_sync = 0

    def close(self) -> None:
        try:
            self._fh.flush()
            os.fsync(self._fh.fileno())
        finally:
            self._fh.close()


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #


class RateLimitError(RuntimeError):
    pass


class TerminalAPIError(RuntimeError):
    """A 4xx that retrying cannot fix (bad key, no balance, bad request)."""


def get_api_key(env_var: str = API_KEY_ENV_VAR) -> str:
    """Read the key from the environment. Never logged, never defaulted."""
    key = os.environ.get(env_var, "").strip()
    if not key:
        raise RuntimeError(
            "{} is not set. Export it (or source a .env that sets it) before "
            "running; the key is never read from a file in this repo.".format(env_var)
        )
    return key


def redact(text: str, key: Optional[str] = None) -> str:
    """Belt-and-braces scrub so a key can never reach a log line."""
    key = key or os.environ.get(API_KEY_ENV_VAR, "")
    if key and key in text:
        text = text.replace(key, "<redacted>")
    return text


class DeepSeekClient:
    """Bounded-concurrency async client with retry, checkpointing and costing."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        concurrency: int = 32,
        max_retries: int = 5,
        base_backoff_s: float = 1.0,
        max_backoff_s: float = 60.0,
        timeout_s: float = 180.0,
        max_tokens: int = 512,
        max_tokens_by_category: Optional[Dict[str, int]] = None,
        temperature: float = 0.7,
        thinking: bool = False,
        requests_per_minute: Optional[int] = None,
        api_key: Optional[str] = None,
        base_url: str = API_BASE_URL,
        log_every: int = 100,
    ) -> None:
        if concurrency < 1:
            raise ValueError("concurrency must be >= 1")
        if concurrency > PUBLISHED_CONCURRENCY_LIMIT:
            LOGGER.warning(
                "concurrency=%d exceeds the published account limit of %d for %s; "
                "expect sustained 429s", concurrency, PUBLISHED_CONCURRENCY_LIMIT, model,
            )
        self.model = model
        self.concurrency = concurrency
        self.max_retries = max_retries
        self.base_backoff_s = base_backoff_s
        self.max_backoff_s = max_backoff_s
        self.timeout_s = timeout_s
        self.max_tokens = max_tokens
        self.max_tokens_by_category = (
            MAX_TOKENS_BY_CATEGORY
            if max_tokens_by_category is None
            else max_tokens_by_category
        )
        self.temperature = temperature
        self.thinking = thinking
        self.requests_per_minute = requests_per_minute
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key or get_api_key()
        self.tracker = UsageTracker(model=model, log_every=log_every)
        self._sem: Optional[asyncio.Semaphore] = None
        self._rpm_lock: Optional[asyncio.Lock] = None
        self._next_slot = 0.0

    # -- internals ---------------------------------------------------------- #

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": "Bearer {}".format(self._api_key),
            "Content-Type": "application/json",
        }

    def max_tokens_for(self, category: Optional[str]) -> int:
        """Output-token cap for a category, falling back to the global default."""
        if not category:
            return self.max_tokens
        return self.max_tokens_by_category.get(category, self.max_tokens)

    def _payload(self, req: GenerationRequest) -> Dict[str, Any]:
        return {
            "model": self.model,
            "messages": build_messages(
                req.stage,
                req.filled_question,
                req.gt_answer,
                category=req.category or "",
                templated_question=req.templated_question or req.filled_question,
            ),
            "max_tokens": self.max_tokens_for(req.category),
            "temperature": self.temperature,
            "stream": False,
            # Thinking is enabled by default at "high" effort on v4-flash, and
            # reasoning tokens bill as output. Measured on this exact prompt it
            # was ~62% of output tokens for zero fidelity gain — this is a
            # verbatim-preserving rewrite, not a reasoning task. See
            # connectivity_test_result.md.
            "thinking": {"type": "enabled" if self.thinking else "disabled"},
        }

    async def _throttle(self) -> None:
        """Optional RPM pacing. Off by default — DeepSeek publishes no RPM cap."""
        if not self.requests_per_minute:
            return
        interval = 60.0 / float(self.requests_per_minute)
        assert self._rpm_lock is not None
        async with self._rpm_lock:
            now = time.monotonic()
            wait = self._next_slot - now
            self._next_slot = max(now, self._next_slot) + interval
        if wait > 0:
            await asyncio.sleep(wait)

    def _backoff_delay(self, attempt: int, retry_after: Optional[float]) -> float:
        if retry_after is not None:
            return min(retry_after, self.max_backoff_s)
        delay = min(self.base_backoff_s * (2 ** (attempt - 1)), self.max_backoff_s)
        return delay * (0.5 + random.random())  # full-ish jitter

    async def _call_once(
        self, http: httpx.AsyncClient, req: GenerationRequest
    ) -> Tuple[str, Usage]:
        resp = await http.post(
            CHAT_COMPLETIONS_PATH, headers=self._headers(), json=self._payload(req)
        )
        if resp.status_code in TERMINAL_STATUS:
            raise TerminalAPIError(
                "HTTP {}: {}".format(resp.status_code, redact(resp.text[:500], self._api_key))
            )
        if resp.status_code in RETRYABLE_STATUS:
            retry_after = resp.headers.get("retry-after")
            raise RateLimitError(
                json.dumps(
                    {
                        "status": resp.status_code,
                        "retry_after": float(retry_after) if retry_after else None,
                        "body": redact(resp.text[:300], self._api_key),
                    }
                )
            )
        resp.raise_for_status()
        body = resp.json()
        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise RuntimeError(
                "unexpected response shape: {}".format(redact(json.dumps(body)[:300], self._api_key))
            )
        return content, Usage.from_api(body.get("usage"))

    async def _process(
        self,
        http: httpx.AsyncClient,
        req: GenerationRequest,
        checkpoint: Optional[JsonlCheckpoint],
    ) -> GenerationResult:
        assert self._sem is not None
        started = time.monotonic()
        result = GenerationResult(
            id=req.id,
            status="error",
            stage=req.stage,
            category=req.category,
            image=req.image,
            filled_question=req.filled_question,
            gt_answer=req.gt_answer,
        )

        async with self._sem:
            last_error = "no attempt made"
            for attempt in range(1, self.max_retries + 1):
                result.attempts = attempt
                await self._throttle()
                try:
                    content, usage = await self._call_once(http, req)
                except TerminalAPIError as exc:
                    last_error = str(exc)
                    LOGGER.error("%s terminal API error, not retrying: %s", req.id, last_error)
                    break
                except RateLimitError as exc:
                    last_error = str(exc)
                    retry_after = None
                    try:
                        retry_after = json.loads(str(exc)).get("retry_after")
                    except (json.JSONDecodeError, AttributeError):
                        pass
                except (httpx.TimeoutException, httpx.TransportError) as exc:
                    last_error = "{}: {}".format(type(exc).__name__, exc)
                    retry_after = None
                except (httpx.HTTPStatusError, RuntimeError, ValueError) as exc:
                    last_error = "{}: {}".format(type(exc).__name__, redact(str(exc), self._api_key))
                    retry_after = None
                else:
                    result.usage = usage
                    result.raw_response = content
                    status, fields = parse_completion(content, req.gt_answer)
                    result.status = status
                    result.paraphrased_question = fields.get("paraphrased_question")
                    result.paraphrased_answer = fields.get("paraphrased_answer")
                    result.skip_reason = fields.get("skip_reason")
                    result.parse_anomaly = fields.get("parse_anomaly")
                    result.error = fields.get("error")
                    if status == "skip":
                        # Terminal by design: a SKIP is the model correctly
                        # refusing to invent an answer. Never retried.
                        LOGGER.info("%s SKIP: %s%s", req.id, result.skip_reason,
                                    " [{}]".format(result.parse_anomaly)
                                    if result.parse_anomaly else "")
                    break

                if attempt < self.max_retries:
                    delay = self._backoff_delay(attempt, retry_after)
                    LOGGER.warning(
                        "%s attempt %d/%d failed (%s); retrying in %.1fs",
                        req.id, attempt, self.max_retries, last_error[:200], delay,
                    )
                    await asyncio.sleep(delay)
            else:
                LOGGER.error("%s exhausted %d attempts", req.id, self.max_retries)

            if result.status == "error" and result.error is None:
                result.error = last_error

        result.latency_s = time.monotonic() - started
        self.tracker.record(result)
        if checkpoint is not None:
            await checkpoint.write(result)
        return result

    # -- public ------------------------------------------------------------- #

    async def run(
        self,
        requests: Sequence[GenerationRequest],
        checkpoint: Optional[JsonlCheckpoint] = None,
    ) -> List[GenerationResult]:
        """Process every request with bounded concurrency.

        Results are checkpointed as each one lands, not at the end.
        """
        self._sem = asyncio.Semaphore(self.concurrency)
        self._rpm_lock = asyncio.Lock()
        self._next_slot = time.monotonic()

        limits = httpx.Limits(
            max_connections=self.concurrency, max_keepalive_connections=self.concurrency
        )
        timeout = httpx.Timeout(self.timeout_s, connect=30.0)
        async with httpx.AsyncClient(
            base_url=self.base_url, timeout=timeout, limits=limits
        ) as http:
            tasks = [
                asyncio.ensure_future(self._process(http, req, checkpoint)) for req in requests
            ]
            results = await asyncio.gather(*tasks)
        return list(results)


# --------------------------------------------------------------------------- #
# I/O helpers
# --------------------------------------------------------------------------- #


def read_requests(path: Path) -> Iterator[GenerationRequest]:
    with Path(path).open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield GenerationRequest.from_json(json.loads(line))
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError("{}:{}: {}".format(path, lineno, exc))


def filter_pending(
    requests: Iterable[GenerationRequest], completed: Set[str]
) -> List[GenerationRequest]:
    pending = [r for r in requests if r.id not in completed]
    if completed:
        LOGGER.info("resume: %d already complete, %d pending", len(completed), len(pending))
    return pending


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--input", type=Path, help="JSONL of filled pairs from Step 3")
    p.add_argument("--output", type=Path, help="JSONL checkpoint file (appended)")
    p.add_argument("--stage", type=int, choices=(1, 2),
                   help="override the stage on every record")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--max-retries", type=int, default=5)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--thinking", action="store_true",
                   help="enable thinking mode (off by default: ~62%% more output "
                        "tokens on this prompt with no measured fidelity gain)")
    p.add_argument("--rpm", type=int, default=None,
                   help="optional client-side requests/min cap (off by default)")
    p.add_argument("--limit", type=int, default=None, help="process at most N records")
    p.add_argument("--usage-summary", type=Path, default=None,
                   help="write the run's cumulative usage/cost JSON here")
    p.add_argument("--log-level", default="INFO")
    return p


async def _main_async(args: argparse.Namespace) -> int:
    if not args.input or not args.output:
        LOGGER.error("--input and --output are both required")
        return 2

    requests = list(read_requests(args.input))
    if args.stage is not None:
        for r in requests:
            r.stage = args.stage
    if args.limit is not None:
        requests = requests[: args.limit]

    checkpoint = JsonlCheckpoint(args.output)
    pending = filter_pending(requests, checkpoint.completed_ids)
    if not pending:
        LOGGER.info("nothing to do — all %d records already complete", len(requests))
        checkpoint.close()
        return 0

    client = DeepSeekClient(
        model=args.model,
        concurrency=args.concurrency,
        max_retries=args.max_retries,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        thinking=args.thinking,
        requests_per_minute=args.rpm,
    )
    LOGGER.info(
        "starting %d calls | model=%s concurrency=%d | pricing checked %s",
        len(pending), args.model, args.concurrency, PRICING_CHECKED,
    )
    try:
        await client.run(pending, checkpoint=checkpoint)
    finally:
        checkpoint.close()

    LOGGER.info("done: %s", client.tracker.summary_line())
    if args.usage_summary:
        Path(args.usage_summary).write_text(
            json.dumps(client.tracker.as_dict(), indent=2) + "\n", encoding="utf-8"
        )
    return 0 if client.tracker.errors == 0 else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    sys.exit(main())
