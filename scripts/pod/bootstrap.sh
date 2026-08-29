#!/bin/bash
# Pod bootstrap for the Stage-2 regeneration retrain.
#
# Brings a fresh Runpod GPU pod from bare image to "ready to run
# scripts/train/train_stage2_lora.sh". Idempotent: every step checks for its own
# output first, so re-running after a failure resumes rather than restarting.
#
# Usage (on the pod):
#   bash scripts/pod/bootstrap.sh 2>&1 | tee /workspace/bootstrap.log
#
# ---------------------------------------------------------------------------
# WHAT THIS PULLS AND WHY IT ISN'T ALL IN GIT
# ---------------------------------------------------------------------------
# Three large inputs are deliberately gitignored and must be fetched here:
#
#   1. BulkFormer-93M.pt (304 MB) + support files (123 MB).
#      `.gitignore:108` excludes bulkencoders/checkpoints/. Fetched by the
#      repo's own bulkencoders/download.py (Google Drive + Zenodo), which is
#      idempotent and size-checks what it already has.
#      NOTE: this is needed EVEN THOUGH training feeds pre-encoded 515-d
#      vectors, because BulkFormerVisionTower.__init__ constructs and loads the
#      encoder unconditionally. Only its forward() is skipped by the passthrough.
#
#   2. lmsys/vicuna-7b-v1.5 (~13.5 GB) from HuggingFace.
#
#   3. The Stage-2 training bundle, which ships in-repo as a zip
#      (data/cvd_transcriptome/text_files/stage2_train.zip) plus the pre-encoded
#      515-d vectors. Those are only ~18 MB because the tower is bypassed —
#      the 668 MB raw [20010] vectors are NOT needed on the pod at all.
# ---------------------------------------------------------------------------

set -euo pipefail

REPO="${REPO:-/workspace/CardioLLM}"
VENV="${VENV:-/workspace/venv}"
PY="${VENV}/bin/python"
PIP="${VENV}/bin/pip"

step() { echo; echo "=== [$(date -u +%H:%M:%S)] $* ==="; }

# ---------------------------------------------------------------------------
step "system packages"
# git-lfs is not used by this repo but pulls in nothing harmful; ninja speeds up
# any source build that does happen (flash-attn's, if the wheel path misses).
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq git curl unzip build-essential ninja-build >/dev/null

# ---------------------------------------------------------------------------
step "python venv"
if [ ! -x "$PY" ]; then
    python3 -m venv --system-site-packages "$VENV"
fi
"$PIP" install -q --upgrade pip setuptools wheel

# torch comes from the base image via --system-site-packages; do NOT reinstall it
# (a pip torch would not match the image's CUDA build).
"$PY" - <<'EOF'
import torch
print(f"torch {torch.__version__}  cuda={torch.version.cuda}  "
      f"available={torch.cuda.is_available()}  n_gpu={torch.cuda.device_count()}")
EOF

# ---------------------------------------------------------------------------
step "python packages"
# transformers is pinned BELOW 4.46: tinyllava/train/train.py passes
# --evaluation_strategy, which 4.46 removed. The 2026-08-13 run logged the
# deprecation warning for exactly this flag, so it was on a <4.46 build.
"$PIP" install -q \
    "transformers==4.44.2" "tokenizers>=0.19,<0.20" "accelerate==0.33.0" \
    "peft==0.12.0" "deepspeed==0.15.4" \
    "sentencepiece==0.2.0" "protobuf" "einops" "einops-exts" "timm" \
    "tensorboard" "tensorboardX" "shortuuid" "ninja" \
    "numpy<2" "scikit-learn" "pandas" "pyarrow" "matplotlib" "h5py" \
    "gdown" "requests" "huggingface_hub[cli]"

# torch_geometric supplies GCNConv, imported by
# bulkencoders/BulkFormer/utils/BulkFormer_block.py. Pure-python wheel; the
# compiled torch-scatter/torch-sparse companions are NOT required for GCNConv.
"$PIP" install -q "torch_geometric"

# flash-attn is what --attn_implementation flash_attention_2 needs. It is
# OPTIONAL here: if the build fails, training falls back to sdpa via ATTN_IMPL
# (see train_stage2_lora.sh) rather than blocking the run. Stage-2 sequences are
# short (mean 141, p95 190 tokens) so the attention kernel is not the bottleneck.
step "flash-attn (optional)"
if "$PY" -c "import flash_attn" 2>/dev/null; then
    echo "flash-attn already present"
elif "$PIP" install -q flash-attn --no-build-isolation; then
    echo "flash-attn installed"
else
    echo "WARNING: flash-attn unavailable — run training with ATTN_IMPL=sdpa"
fi

# ---------------------------------------------------------------------------
step "install repo (editable)"
cd "$REPO"
"$PIP" install -q -e . --no-deps

# ---------------------------------------------------------------------------
step "BulkFormer-93M checkpoint + support files"
# 93M only. The encoder scale is LOCKED at 93M project-wide (see the sweep note
# in stage2_regen_report/tables/); downloading the other four variants would
# cost ~1.1 GB for nothing.
if [ ! -s bulkencoders/checkpoints/bulkformer/models/BulkFormer-93M.pt ]; then
    "$PY" -m bulkencoders.download --models 93M
else
    echo "BulkFormer-93M.pt already present"
fi
ls -la bulkencoders/checkpoints/bulkformer/models/ bulkencoders/checkpoints/bulkformer/support/

# ---------------------------------------------------------------------------
step "Vicuna-7B v1.5"
if [ ! -d "${HF_HOME:-/workspace/hf}/hub/models--lmsys--vicuna-7b-v1.5" ]; then
    export HF_HOME="${HF_HOME:-/workspace/hf}"
    "$VENV/bin/hf" download lmsys/vicuna-7b-v1.5 \
        --exclude "*.bin" "*.h5" "*.msgpack" \
        || "$VENV/bin/huggingface-cli" download lmsys/vicuna-7b-v1.5
else
    echo "vicuna already cached"
fi

# ---------------------------------------------------------------------------
step "unpack the Stage-2 training bundle"
cd "$REPO/data/cvd_transcriptome/text_files"
[ -f stage2_train.json ] || unzip -o -q stage2_train.zip
cd "$REPO/data/cvd_transcriptome"
[ -d embeddings_encoded ] || unzip -o -q embeddings_encoded.zip
cd "$REPO"
"$PY" - <<'EOF'
import json, pathlib
p = pathlib.Path("data/cvd_transcriptome/text_files/stage2_train.json")
d = json.loads(p.read_text())
print(f"stage2_train.json: {len(d):,} items")
enc = pathlib.Path("data/cvd_transcriptome/embeddings_encoded")
print(f"encoded vectors:   {len(list(enc.glob('*.npy'))):,} files")
h = json.loads(pathlib.Path("data/cvd_transcriptome/holdout_series.json").read_text())
print(f"holdout:           {h['n_series']} series, "
      f"{h['n_holdout_positive']:,} pos / {h['n_holdout_neg_hard']:,} neg")
EOF

# ---------------------------------------------------------------------------
step "verify the pre-encoded 515-d cache against a live tower forward"
# The cache was DERIVED from linear_probe's embedding parquet rather than
# produced by a GPU encoder pass — the shortcut that removes an entire GPU stage
# from this retrain. It was verified bitwise-exact locally against a live CPU
# forward of the real tower (max_abs_diff 0.0, with a rolled-vector control); see
# data/cvd_transcriptome/encoded_cache_manifest.json. This re-checks it here, on
# different hardware and a freshly downloaded BulkFormer checkpoint, before
# hours of paid GPU time depend on it.
#
# A full re-encode is deliberately impossible on the pod: the raw [20010]
# vectors are 668 MB and not shipping them is the entire point. A 16-sample
# slice travels with the repo (1.0 MB) purely so this check can run.
cd "$REPO/data/cvd_transcriptome"
[ -d verify_sample_raw ] || unzip -o -q verify_sample_raw.zip
cd "$REPO"
"$PY" -m integration.verify_encoded_cache --device cuda

# ---------------------------------------------------------------------------
step "preflight assertions"
# Fail here, cheaply, rather than 4 minutes into a distributed launch.
"$PY" - <<'EOF'
import json, pathlib, sys
fail = []

cfg = json.loads(pathlib.Path("integration/bulkformer_hf_config/config.json").read_text())
if cfg.get("bulkformer_variant") != "BulkFormer-93M":
    fail.append(f"encoder scale not locked to 93M: {cfg.get('bulkformer_variant')}")
if cfg.get("hidden_size") != 515:
    fail.append(f"hidden_size should be 515 for 93M, got {cfg.get('hidden_size')}")

ck = pathlib.Path("checkpoints/stage1-connector-93M/connector/pytorch_model.bin")
if not ck.exists():
    fail.append("stage-1 connector checkpoint missing — stage 2 starts from it")

# Holdout must not leak into training. This is the single most important
# invariant of the whole regeneration; assert it on the pod too, because the
# bundle could in principle have been rebuilt after the split was written.
train = json.loads(pathlib.Path("data/cvd_transcriptome/text_files/stage2_train.json").read_text())
hold = json.loads(pathlib.Path("data/cvd_transcriptome/holdout_series.json").read_text())
held = {a + ".npy" for a in hold["holdout_geo_accession"]}
leaked = {it["image"] for it in train} & held
if leaked:
    fail.append(f"{len(leaked)} holdout samples present in stage2_train.json")

for f in fail:
    print("PREFLIGHT FAIL:", f)
sys.exit(1 if fail else 0)
EOF

step "ready"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo "run: IMAGE_PATH=data/cvd_transcriptome/embeddings_encoded SAVE_STRATEGY=no bash scripts/train/train_stage2_lora.sh"
