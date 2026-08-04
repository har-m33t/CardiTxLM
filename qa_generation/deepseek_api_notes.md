# DeepSeek API — rate limits, caching, and batch options

**Checked:** 2026-08-03. All primary claims are from `api-docs.deepseek.com`;
where a claim comes from a third party or from our own measurement, it says so.
Nothing here is carried over from prior assumptions — the earlier notes in
`qa_generation_pipeline_todo.md` were re-verified from source, and one of them
turned out to be incomplete (the cache-hit price tier).

---

## 1. Rate limits

**DeepSeek does not publish RPM or TPM limits. It publishes a concurrency
limit.** This is the finding that most changes the batching design — the usual
"N requests per minute" throttle is not the right shape.

| Model | Max concurrent requests |
|---|---|
| `deepseek-v4-pro` | 500 |
| `deepseek-v4-flash` | **2500** |

Details from <https://api-docs.deepseek.com/quick_start/rate_limit>:

* **Account-level, not key-level.** "Concurrency limits are calculated at the
  account level, regardless of which API Key is used." Adding keys does not add
  throughput.
* **Exceeding it returns HTTP 429.**
* **Optional per-user sub-quotas.** A `user_id` parameter partitions concurrency,
  content-safety handling, KVCache storage, and scheduling per end user, so one
  user can't monopolize the account quota. Not relevant for a single batch job —
  we deliberately want all calls sharing one KVCache namespace so the prefix
  cache hits.
* **Long requests are held open, not dropped.** While a request is queued the
  server sends keep-alives: empty lines for non-streaming requests, `: keep-alive`
  SSE comments for streaming ones. So a client timeout must be generous — a slow
  response is not a failed one. `deepseek_client.py` uses a 180 s read timeout.

Note the docs' own 429 advice is unusually blunt: "Please pace your requests
reasonably" *or consider switching to alternative LLM providers.* There is no
published quota to appeal to, and the third-party consensus
([requesty.ai](https://www.requesty.ai/blog/rate-limits-for-llm-providers-openai-anthropic-and-deepseek),
[chat-deep.ai](https://chat-deep.ai/docs/api-rate-limits/)) is that the effective
ceiling is adjusted dynamically based on account traffic and current server load.
**Treat 2500 as a ceiling that can move, not a guarantee** — which is exactly why
the client retries 429 with exponential backoff instead of assuming a fixed rate
is safe.

### Error codes and what's retryable

From <https://api-docs.deepseek.com/quick_start/error_codes>:

| Code | Meaning | Retry? |
|---|---|---|
| 400 | Invalid request body format | **No** — fix the request |
| 401 | Wrong API key | **No** |
| 402 | Insufficient balance | **No** — top up |
| 422 | Invalid parameters | **No** |
| 429 | Rate limit reached | **Yes**, with backoff |
| 500 | Server error | **Yes** — "retry after a brief wait" |
| 503 | Server overloaded | **Yes** — "retry after a brief wait" |

The client encodes this split directly (`TERMINAL_STATUS` vs `RETRYABLE_STATUS`):
a 402 mid-run fails the item immediately and loudly rather than burning five
backoff cycles per remaining call.

### Practical throughput for a 250K-call job

At the measured ~2.5 s mean latency (thinking disabled):

| Concurrency | Throughput | 250K calls in |
|---|---|---|
| 16 | ~6 calls/s | ~10.9 h |
| 32 | ~13 calls/s | ~5.4 h |
| 64 | ~26 calls/s | ~2.7 h |
| 128 | ~51 calls/s | ~1.4 h |

The client defaults to **concurrency 32** — 1.3% of the published ceiling,
finishing in a single working day. Raise it if that's too slow; there is a lot of
headroom, but ramp rather than jumping to 2500, since the effective limit is
dynamic and a 429 storm at full fan-out wastes more time than it saves.

---

## 2. Prompt caching

**Supported, on by default, and worth a 50x discount on the cached portion.**

From <https://api-docs.deepseek.com/guides/kv_cache>:

* **Zero configuration.** "The DeepSeek API Context Caching on Disk Technology is
  enabled by default for all users, without needing to modify their code." There
  is no `cache_control` marker to place, unlike some other providers.
* **Exact-prefix matching.** "A subsequent request can only hit the cache if it
  **fully matches** a **cache prefix unit**." Cache units form at request
  boundaries, at common-prefix detection points, and at fixed token intervals
  within long inputs.
* **Best-effort.** No guaranteed hit rate.
* **Expiry:** "Once the cache is no longer in use, it will be automatically
  cleared, usually within a few hours to a few days." Fine for a run that
  completes in hours; a run split across days pays one cold miss per resumption.
* **Reported per call** in the `usage` object as `prompt_cache_hit_tokens` and
  `prompt_cache_miss_tokens` (also mirrored at
  `prompt_tokens_details.cached_tokens`).

### Cache-hit pricing vs cache-miss

From the pricing page — **this is the number missing from the project's earlier
notes**:

| | Per 1M input tokens | Ratio |
|---|---|---|
| Cache **miss** | $0.14 | 1x |
| Cache **hit** | **$0.0028** | **1/50** |

Cached input is effectively free. This is a 98% reduction, not the 90% that
several third-party summaries quote — the docs' own pricing table is the
authority here.

### What we measured

The docs don't state a cache granularity, so it was measured. Sending repeated
calls that share the canonical prompt prefix:

* **Warm-cache hits land on 64-token boundaries** — 768 tokens of an ~871-token
  Stage 1 prompt (88.1%), 1024 of an ~1134-token Stage 2 prompt (90.3%). Every
  observed hit value was a multiple of 64, consistent with 64-token cache units.
* First call against a given prefix is always a full miss (cold cache).
* Warming is not instant: a first pass of 7 calls at concurrency 2 hit only 13.2%
  cumulative, while an immediate second pass of the same 7 hit **90.7%**. A
  250K-call run amortizes the cold start to nothing, but a small pilot batch will
  understate the cache benefit — don't extrapolate cost from the first 50 calls.
* Hit sizes vary within a run (640 vs 768 observed on Stage 1) — it is
  best-effort. The estimate uses the warm values above as steady state.

**Design consequence:** the prompt prefix must be byte-identical across calls and
nothing sample-specific may precede it. The canonical prompts put all per-call
values in a `Given:` block at the very end, so `build_messages()` substitutes
only there and sends the result as a single user message; everything above
`Given:` is constant and caches. The module docstring warns that editing a prompt
file mid-run invalidates the cache for every later call — which would silently
multiply input cost by 50x. Sorting the work so all Stage 1 calls precede all
Stage 2 calls also avoids alternating between two different prefixes.

### Value of caching on this job

| | Cost |
|---|---|
| With caching (measured 768/1024-token prefix hits) | $8.98 |
| Without caching | $37.07 |
| **Saved** | **$28.09 (76%)** |

Caching is now the dominant cost lever — the canonical prompts are long and
almost entirely cacheable, so ~93% of all input tokens bill at the 1/50 rate.
Watch the cumulative cache-hit rate in the run log; if it collapses, so does the
economics of the run.

---

## 3. Batch API

**There is no DeepSeek batch API.** The documentation navigation
(<https://api-docs.deepseek.com/>) lists Quick Start, Thinking Mode, Multi-round
Conversation, Chat Prefix Completion (Beta), FIM Completion (Beta), JSON Output,
Tool Calls, Context Caching, the Responses API, and the Anthropic-compatible API
— **no batch, asynchronous-batch, or queued/offline inference section, and no
batch pricing tier on the pricing page.**

Several third-party pages assert a "50% off asynchronous batch" tier for DeepSeek
([deploybase.ai](https://deploybase.ai/articles/deepseek-api-pricing),
[nxcode.io](https://www.nxcode.io/resources/news/deepseek-api-pricing-complete-guide-2026)).
**I could not confirm this from DeepSeek's own docs and it should be treated as
wrong** — it looks like generalization from providers that do offer one (OpenAI,
Anthropic, Together). Related: the historical 50%-off off-peak discount applied
to the V3/R1 generation and ended when the `deepseek-chat` / `deepseek-reasoner`
aliases were retired on 2026-07-24; the V4 replacement is the *2x peak surcharge*
described in the cost estimate, which is a penalty, not a discount.

**Conclusion: standard `/chat/completions` with client-side async fan-out is the
only option, and it is the right one here.** With no batch discount available,
the cost levers that remain are prompt caching (76% saved), disabling thinking
mode (52% saved), and — if the peak policy takes effect — scheduling outside
09:00–12:00 and 14:00–18:00 Beijing time.

---

## 4. Thinking mode — the finding that wasn't on the checklist

Not something the task asked to check, but it dominates output cost, so it
belongs here.

From <https://api-docs.deepseek.com/guides/thinking_mode>: thinking mode is
**"enabled by default, with the default effort being high"** on `deepseek-v4-flash`.
Chain-of-thought comes back in `reasoning_content` alongside `content`, and
measurement confirms **reasoning tokens are counted inside `completion_tokens`**
— i.e. billed at the $0.28/1M output rate. A response carrying 35 tokens of
visible text reported 104 completion tokens, of which
`completion_tokens_details.reasoning_tokens` was 69.

Disable with `"thinking": {"type": "disabled"}` (OpenAI-format endpoint) or
`"reasoning": {"effort": "none"}` (Anthropic-format endpoint). For a
verbatim-preserving paraphrase task the reasoning buys nothing — fidelity was
identical with it off — and it costs 2.63x the output tokens plus roughly double
the latency. The client disables it by default.

---

## 5. Other API facts worth having on record

| | |
|---|---|
| Base URL (OpenAI format) | `https://api.deepseek.com` |
| Base URL (Anthropic format) | `https://api.deepseek.com/anthropic` |
| Models | `deepseek-v4-flash`, `deepseek-v4-pro` |
| Context length | 1M tokens |
| Max output | 384K tokens |
| Offline tokenizer | `deepseek_tokenizer.zip` from the docs |
| Rough token estimate | 1 English character ≈ 0.3 token |

Context length is irrelevant at ~600 tokens per call, and the 0.3 tokens/char
heuristic proved good — it predicted input within 5–10% of the measured values.
It badly underestimates *output*, but only because of the hidden reasoning
tokens, not because the ratio is wrong.

---

## Sources

- [Models & Pricing](https://api-docs.deepseek.com/quick_start/pricing)
- [Rate Limit & Isolation](https://api-docs.deepseek.com/quick_start/rate_limit)
- [Context Caching](https://api-docs.deepseek.com/guides/kv_cache)
- [Thinking Mode](https://api-docs.deepseek.com/guides/thinking_mode)
- [Error Codes](https://api-docs.deepseek.com/quick_start/error_codes)
- [Token & Token Usage](https://api-docs.deepseek.com/quick_start/token_usage)
- [API docs index (checked for a batch section)](https://api-docs.deepseek.com/)
- Third-party, used only where flagged as unconfirmed: [Requesty](https://www.requesty.ai/blog/rate-limits-for-llm-providers-openai-anthropic-and-deepseek), [chat-deep.ai](https://chat-deep.ai/docs/api-rate-limits/), [DeployBase](https://deploybase.ai/articles/deepseek-api-pricing), [NxCode](https://www.nxcode.io/resources/news/deepseek-api-pricing-complete-guide-2026)
