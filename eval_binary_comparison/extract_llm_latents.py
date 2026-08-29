"""extract_llm_latents.py — cache the trained multimodal LLM's latent
representation of each expression profile, so a linear probe can be run on it.

This is `linear_probe/extract.py` one level further down the stack. That script
caches the FROZEN ENCODER's pooled embedding per sample. This one pushes those
same embeddings through the trained connector + LoRA'd Vicuna and caches the
LLM's hidden state per sample. Feed the output to the same probe, same folds,
same metric, and the two rows are directly comparable — which is exactly the
comparison the results slide makes ("Linear Probe against Trained Multi-modal
LLM", "aligning foundational model embeddings into LLM latent space provides a
signal rich representation for downstream probing").

Input is the probe's own 515-d parquet, so the encoder is never re-run: those
vectors ARE the tower's pooled output (extract.py mean-pools over genes exactly
as BulkFormerVisionTower.forward does) and the tower passes them through
untouched by width.

Two poolings are cached, because the slide does not say which it used:
  * `imgtok` — final-layer hidden state AT the expression-token position. The
    causal mask means this position attends only to the system preamble, so it
    is close to a prompt-independent view of the projected embedding.
  * `meanpool` — final-layer hidden state mean-pooled over all positions.

Run:
    python -m eval_binary_comparison.extract_llm_latents \
        --lora-ckpt checkpoints/stage2-lora-bulkformer-93M \
        --embeddings linear_probe/embeddings/embeddings_BulkFormer-93M.parquet \
        --out-prefix linear_probe/embeddings/embeddings_LLM-latent \
        --batch-size 32
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

PROMPT = ("Based on this transcriptomic profile, does this sample show evidence "
          "of cardiovascular disease?")


def load_population(parquet_path):
    import pyarrow.parquet as pq
    t = pq.read_table(parquet_path).to_pydict()
    ecols = sorted(c for c in t if c.startswith("e0"))
    n = len(t["geo_accession"])
    X = np.empty((n, len(ecols)), dtype=np.float32)
    for j, c in enumerate(ecols):
        X[:, j] = np.asarray(t[c], dtype=np.float32)
    meta = {k: list(t[k]) for k in
            ("geo_accession", "series_id", "cvd_subtype", "is_positive",
             "is_neg_hard", "is_neg_whole_corpus", "pool", "sample_index")}
    return X, meta


def build_prompt_ids(tokenizer, conv_version="llama"):
    from tinyllava.data.template import TemplateFactory
    template = TemplateFactory(conv_version)()
    q = "<image>\n" + PROMPT
    msg = [{"from": "human", "value": q}, {"from": "gpt", "value": ""}]
    prompt = None
    try:
        enc = template.encode(msg, tokenizer, mode="eval")
        if isinstance(enc, dict) and "prompt" in enc:
            prompt = enc["prompt"]
    except Exception:
        pass
    if prompt is None:
        prompt = ("A chat between a curious user and an artificial intelligence "
                  "assistant. The assistant gives helpful, detailed, and polite "
                  f"answers to the user's questions. USER: {q} ASSISTANT:")
    return template.tokenizer_image_token(prompt, tokenizer, return_tensors="pt")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora-ckpt", required=True)
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="debug: only N samples")
    args = ap.parse_args(argv)

    from evaluate_and_compare import load_model
    from tinyllava.utils.constants import IMAGE_TOKEN_INDEX

    X, meta = load_population(args.embeddings)
    if args.limit:
        X = X[:args.limit]
        meta = {k: v[:args.limit] for k, v in meta.items()}
    print(f"population: {len(X)} samples, encoder dim {X.shape[1]}")

    model, tokenizer = load_model(args.lora_ckpt)
    ids = build_prompt_ids(tokenizer)
    img_pos = int((ids == IMAGE_TOKEN_INDEX).nonzero()[0].item())
    print(f"prompt_len={ids.shape[0]}  image-token index={img_pos}")
    ids = ids.to(model.device)

    H = model.config.text_config.hidden_size if hasattr(model.config, "text_config") \
        else model.config.hidden_size
    out_img = np.zeros((len(X), H), dtype=np.float32)
    out_mean = np.zeros((len(X), H), dtype=np.float32)

    for i in range(0, len(X), args.batch_size):
        chunk = X[i:i + args.batch_size]
        b = len(chunk)
        images = torch.from_numpy(chunk).to(device=model.device, dtype=model.dtype)
        input_ids = ids.unsqueeze(0).expand(b, -1).contiguous()
        with torch.inference_mode():
            o = model(input_ids=input_ids, images=images, output_hidden_states=True)
        hs = o.hidden_states[-1].float()          # [B, S, H] final layer
        if i == 0:
            assert hs.shape[1] == input_ids.shape[1], (
                f"sequence length changed ({hs.shape[1]} vs {input_ids.shape[1]}); "
                "the image-token index would no longer be valid")
            print(f"hidden states {tuple(hs.shape)} — seq length preserved, index valid")
        out_img[i:i + b] = hs[:, img_pos, :].cpu().numpy()
        out_mean[i:i + b] = hs.mean(dim=1).cpu().numpy()
        if (i // args.batch_size) % 50 == 0:
            print(f"  {min(i+b, len(X))}/{len(X)}", flush=True)

    import pyarrow as pa, pyarrow.parquet as pq
    for tag, arr in (("imgtok", out_img), ("meanpool", out_mean)):
        cols = {k: pa.array(meta[k]) for k in meta}
        for j in range(arr.shape[1]):
            cols[f"e{j:04d}"] = pa.array(arr[:, j].astype(np.float32))
        path = f"{args.out_prefix}-{tag}.parquet"
        pq.write_table(pa.table(cols), path, compression="zstd")
        print(f"wrote {path}  ({arr.shape[0]}x{arr.shape[1]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
