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
    "gdown==5.2.0" "requests" "huggingface_hub[cli]"

# BulkFormer's own dependency chain. All three are required to CONSTRUCT the
# tower, not merely to run it, because BulkFormerVisionTower.__init__ builds the
# interaction graph — so training needs them even though the pre-encoded
# passthrough means the encoder never actually runs a forward.
#   torch_geometric   GCNConv, imported by utils/BulkFormer_block.py
#   torch-sparse      SparseTensor, which torch_geometric refuses to construct
#                     without it ("'SparseTensor' requires 'torch-sparse'").
#                     Must match the exact torch+CUDA build, hence the PyG
#                     wheel index rather than plain PyPI (a source build needs
#                     the full CUDA toolkit and takes tens of minutes).
#   performer-pytorch BulkFormer's attention block
"$PIP" install -q "torch_geometric"
TORCH_V=$("$PY" -c "import torch;print(torch.__version__.split('+')[0])")
CU_V=$("$PY" -c "import torch;print('cu'+torch.version.cuda.replace('.',''))")
"$PIP" install -q torch-sparse torch-scatter \
    -f "https://data.pyg.org/whl/torch-${TORCH_V}+${CU_V}.html"
"$PIP" install -q "performer-pytorch"
"$PY" -c "import torch_sparse, performer_pytorch; print('torch_sparse', torch_sparse.__version__)"

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
step "BulkFormer source"
# bulkencoders/BulkFormer/ is the upstream authors' repo, vendored locally and
# gitignored (.gitignore:103), so a clone of THIS repo does not carry it.
# tinyllava/model/vision_tower/bulkformer.py puts that directory on sys.path and
# imports `utils.BulkFormer`, so it must be present before the tower can be
# constructed at all.
#
# CAVEAT: this pulls upstream HEAD. The local vendored copy is what every
# measurement in this project was made against; if upstream has moved, the
# encoder architecture could differ. The 93M checkpoint load is strict
# (load_state_dict(..., strict=True) in bulkformer.py), so an incompatible
# version fails loudly rather than silently — but if that happens, copy the
# vendored bulkencoders/BulkFormer/ across instead of debugging upstream.
if [ ! -f bulkencoders/BulkFormer/utils/BulkFormer.py ]; then
    git clone --depth 1 https://github.com/KangBoming/BulkFormer.git \
        bulkencoders/BulkFormer
else
    echo "BulkFormer source already present"
fi

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
# Unconditional overwrite, deliberately. A `[ -f ... ] ||` guard here would
# skip re-extraction after a `git pull` brought a NEW bundle, silently training
# on the previous corpus — the worst possible failure for a run whose entire
# point is that the corpus changed. Extraction costs seconds; a stale corpus
# costs the experiment.
cd "$REPO/data/cvd_transcriptome/text_files"
unzip -o -q stage2_train.zip
cd "$REPO/data/cvd_transcriptome"
unzip -o -q embeddings_encoded.zip
# NOTE: the in-repo zip holds only the 8,553 Stage-2 POSITIVES. Hypothesis B
# also trains on negatives and needs the full 31,032-entry cache, which is
# 66 MB and deliberately not in git — transfer it separately:
#   scp embeddings_encoded_v2.zip <pod>:/workspace/ && \
#     (cd data/cvd_transcriptome && unzip -o -q /workspace/embeddings_encoded_v2.zip)
unzip -o -q verify_sample_raw.zip
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
    fail.append(
        "stage-1 connector missing at checkpoints/stage1-connector-93M/connector/"
        "pytorch_model.bin. checkpoints/ is gitignored (.gitignore:59), so a "
        "clone does NOT carry it — copy it from the workstation:\n"
        "    scp -r checkpoints/stage1-connector-93M <pod>:$PWD/checkpoints/\n"
        "It is 4.2 MB and is the ONLY Stage-1 artifact carrying training; it "
        "must be copied, never regenerated.")
elif not pathlib.Path("checkpoints/stage1-connector-93M/language_model/config.json").exists():
    fail.append(
        "stage-1 connector present but language_model/ is not. "
        "training_recipe/base.py loads the LLM from <stage1>/language_model, "
        "and Stage 1 froze the LLM so those are the unmodified base weights. "
        "Rebuild them exactly (13.5 GB, from the HF cache already downloaded):\n"
        "    python -m integration.materialize_stage1_llm "
        "--ckpt ./checkpoints/stage1-connector-93M")

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

# ---------------------------------------------------------------------------
step "NCCL peer-to-peer transport check"
# On at least one L40S host this hangs: every rank blocks on the FIRST
# collective and the run dies 600 s later with a watchdog timeout, having
# logged nothing that names the cause. It presents as "training never starts",
# not as an error, so it is worth ten seconds to test up front.
#
# The fix is NCCL_P2P_DISABLE=1. The cost is a slower gradient all-reduce; with
# ZeRO-2 and only ~322 M trainable parameters that is affordable (measured 7
# steps/min against 35 on a healthy host).
cat > /tmp/nccl_probe.py <<'NCCLPROBE'
import torch, torch.distributed as dist
dist.init_process_group("nccl")
r = dist.get_rank(); torch.cuda.set_device(r)
t = torch.ones(1024, device=f"cuda:{r}") * r
dist.all_reduce(t)
if r == 0:
    print(f"all_reduce ok (sum={t[0].item()})")
dist.destroy_process_group()
NCCLPROBE
NGPU=$(nvidia-smi -L | wc -l)
if timeout 90 "$VENV/bin/torchrun" --nproc_per_node="$NGPU" --master_port=29599 \
       /tmp/nccl_probe.py 2>/dev/null | grep -q "all_reduce ok"; then
    echo "NCCL peer-to-peer OK — no flag needed"
elif NCCL_P2P_DISABLE=1 timeout 90 "$VENV/bin/torchrun" --nproc_per_node="$NGPU" \
       --master_port=29598 /tmp/nccl_probe.py 2>/dev/null | grep -q "all_reduce ok"; then
    echo "WARNING: default NCCL p2p HANGS on this host; it works with p2p off."
    echo "         Prefix training and eval with NCCL_P2P_DISABLE=1."
else
    echo "WARNING: NCCL all_reduce failed both with and without p2p — "
    echo "         multi-GPU training will not work on this host."
fi

step "ready"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo "run: IMAGE_PATH=data/cvd_transcriptome/embeddings_encoded SAVE_STRATEGY=no bash scripts/train/train_stage2_lora.sh"
