#!/usr/bin/env bash
# ============================================================
# cloud_comparison_bench_setup.sh — Ship code + models to rented GPU
# and run the 5-config comparison benchmark.
#
# Usage:
#   ./tools/cloud_comparison_bench_setup.sh <user@remote-ip> [model-name]
#
# Examples:
#   ./tools/cloud_comparison_bench_setup.sh root@209.38.1.123 qwen2.5-14b-instruct
#   ./tools/cloud_comparison_bench_setup.sh root@209.38.1.123 qwen2.5-7b-instruct
#   ./tools/cloud_comparison_bench_setup.sh root@209.38.1.123   # defaults to 14b
#
# What it does:
#   1. Validates local artifacts (cdnav3/ + cdnav3_k64/)
#   2. Ships helix-substrate code (no receipts, no .git)
#   3. Ships compressed models (both k=256 and k=64 CDNA)
#   4. Downloads dense + GPTQ + AWQ weights on remote
#   5. Installs Python deps
#   6. Runs cloud_comparison_bench.py
#   7. Pulls all receipts back to local
#
# Requirements:
#   - Remote: Ubuntu + CUDA + Python 3.10+
#   - RTX 4090 (24 GB) for 7B, A100/H100 (80 GB) for 14B
# ============================================================

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <user@remote-ip> [model-name]"
    echo "  model-name: qwen2.5-14b-instruct (default) or qwen2.5-7b-instruct"
    echo "Example: $0 root@209.38.1.123 qwen2.5-14b-instruct"
    exit 1
fi

REMOTE="$1"
MODEL_NAME="${2:-qwen2.5-14b-instruct}"
HELIX_DIR="$(cd "$(dirname "$0")/.." && pwd)"
WORK="/home/user/helix_bench"

MODEL_DIR="$HOME/models/$MODEL_NAME"

# Map model name to HuggingFace repos
case "$MODEL_NAME" in
    qwen2.5-14b-instruct)
        HF_DENSE="Qwen/Qwen2.5-14B-Instruct"
        HF_GPTQ="Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4"
        HF_AWQ="Qwen/Qwen2.5-14B-Instruct-AWQ"
        MIN_TENSORS=300  # 48 blocks × 7 = 336
        ;;
    qwen2.5-7b-instruct)
        HF_DENSE="Qwen/Qwen2.5-7B-Instruct"
        HF_GPTQ="Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"
        HF_AWQ="Qwen/Qwen2.5-7B-Instruct-AWQ"
        MIN_TENSORS=150  # 28 blocks × 7 = 196
        ;;
    *)
        echo "ERROR: Unknown model '$MODEL_NAME'"
        echo "Supported: qwen2.5-14b-instruct, qwen2.5-7b-instruct"
        exit 1
        ;;
esac

echo "============================================================"
echo " HELIX CLOUD 5-CONFIG COMPARISON SETUP"
echo "============================================================"
echo " Remote:    $REMOTE"
echo " Model:     $MODEL_NAME"
echo " Workspace: $WORK"
echo " Code:      $HELIX_DIR"
echo ""
echo " Artifacts to ship:"
echo "   k=256: $MODEL_DIR/cdnav3/"
echo "   k=64:  $MODEL_DIR/cdnav3_k64/"
echo " HF repos:"
echo "   Dense: $HF_DENSE"
echo "   GPTQ:  $HF_GPTQ"
echo "   AWQ:   $HF_AWQ"
echo "============================================================"

# ── Validate local artifacts ──
echo ""
echo "[1/7] Validating local artifacts..."

CDNA_256="$MODEL_DIR/cdnav3"
CDNA_K64="$MODEL_DIR/cdnav3_k64"

if [ ! -d "$CDNA_256" ]; then
    echo "ERROR: $CDNA_256 not found. Run compression first."
    exit 1
fi

N_256=$(ls -d "$CDNA_256"/*.cdnav3 2>/dev/null | wc -l)
echo "  k=256 CDNA: $N_256 tensors"
if [ "$N_256" -lt "$MIN_TENSORS" ]; then
    echo "ERROR: Only $N_256 tensors in k=256 CDNA. Expected $MIN_TENSORS+."
    exit 1
fi

if [ ! -d "$CDNA_K64" ]; then
    echo "ERROR: $CDNA_K64 not found. Run k=64 compression first."
    exit 1
fi

N_K64=$(ls -d "$CDNA_K64"/*.cdnav3 2>/dev/null | wc -l)
echo "  k=64 CDNA:  $N_K64 tensors"
if [ "$N_K64" -lt "$MIN_TENSORS" ]; then
    echo "ERROR: Only $N_K64 tensors in k=64 CDNA. Expected $MIN_TENSORS+."
    exit 1
fi

echo "  All artifacts present."

# ── Step 2: Create code tarball ──
echo ""
echo "[2/7] Packing helix-substrate code..."
CODE_TAR="/tmp/helix_substrate_comparison.tar.gz"
tar czf "$CODE_TAR" \
    -C "$(dirname "$HELIX_DIR")" \
    --exclude='*.pyc' \
    --exclude='__pycache__' \
    --exclude='.git' \
    --exclude='receipts' \
    --exclude='*.safetensors' \
    --exclude='*.bin' \
    --exclude='*.npy' \
    "$(basename "$HELIX_DIR")/helix_substrate" \
    "$(basename "$HELIX_DIR")/tools/cloud_comparison_bench.py" \
    "$(basename "$HELIX_DIR")/tools/cloud_comparison_bench_setup.sh"

CODE_SIZE=$(du -sh "$CODE_TAR" | cut -f1)
echo "  Code tarball: $CODE_SIZE"

# ── Step 3: Ship code ──
echo ""
echo "[3/7] Shipping code to $REMOTE..."
ssh "$REMOTE" "mkdir -p $WORK"
scp "$CODE_TAR" "$REMOTE:$WORK/helix_substrate_comparison.tar.gz"
ssh "$REMOTE" "cd $WORK && tar xzf helix_substrate_comparison.tar.gz"

# ── Step 4: Ship compressed models ──
echo ""
echo "[4/7] Shipping compressed models (both k=256 and k=64 CDNA + config/tokenizer)..."

REMOTE_MODEL="$WORK/models/$MODEL_NAME"
ssh "$REMOTE" "mkdir -p $REMOTE_MODEL"

# Ship k=256 CDNA
echo "  Shipping k=256 CDNA ($N_256 tensors)..."
rsync -az --progress \
    "$CDNA_256/" "$REMOTE:$REMOTE_MODEL/cdnav3/"

# Ship k=64 CDNA
echo "  Shipping k=64 CDNA ($N_K64 tensors)..."
rsync -az --progress \
    "$CDNA_K64/" "$REMOTE:$REMOTE_MODEL/cdnav3_k64/"

# Ship config + tokenizer files
echo "  Shipping config + tokenizer files..."
rsync -az \
    --include='config.json' \
    --include='tokenizer.json' \
    --include='tokenizer_config.json' \
    --include='merges.txt' \
    --include='vocab.json' \
    --include='generation_config.json' \
    --include='special_tokens_map.json' \
    --include='added_tokens.json' \
    --exclude='*' \
    "$MODEL_DIR/" "$REMOTE:$REMOTE_MODEL/"

# ── Step 5+6+7: Install deps, download weights, run benchmark ──
echo ""
echo "[5/7] Installing deps on $REMOTE..."
echo "[6/7] Downloading dense + GPTQ + AWQ weights..."
echo "[7/7] Running 5-config comparison benchmark..."

ssh "$REMOTE" bash -s <<REMOTE_SCRIPT
set -euo pipefail

WORK="$WORK"
cd "\$WORK"

echo "=== Installing Python deps ==="
pip3 install -q \
    'torch<2.11' --index-url https://download.pytorch.org/whl/cu126 2>&1 | tail -3
pip3 install -q \
    "transformers>=4.45" \
    safetensors scipy datasets numpy triton sentencepiece accelerate \
    huggingface_hub auto-gptq autoawq 2>&1 | tail -3

echo ""
echo "=== GPU Info ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || echo "no nvidia-smi"

echo ""
echo "=== Checking CUDA ==="
python3 -c "import torch; print(f'torch {torch.__version__}, cuda={torch.cuda.is_available()}')"

# ── Download dense weights ──
echo ""
echo "=== Downloading dense weights: $HF_DENSE ==="
MODEL_DIR="\$WORK/models/$MODEL_NAME"

HAS_WEIGHTS=0
if [ -f "\$MODEL_DIR/model.safetensors" ]; then HAS_WEIGHTS=1; fi
if [ -f "\$MODEL_DIR/model.safetensors.index.json" ]; then HAS_WEIGHTS=1; fi

if [ "\$HAS_WEIGHTS" -eq 0 ]; then
    echo "  Downloading $HF_DENSE -> \$MODEL_DIR ..."
    huggingface-cli download "$HF_DENSE" --local-dir "\$MODEL_DIR" \
        --include "*.safetensors" "*.safetensors.index.json"
else
    echo "  Dense weights already present"
fi

# ── Download GPTQ ──
echo ""
echo "=== Downloading GPTQ Int4: $HF_GPTQ ==="
GPTQ_DIR="\$WORK/models/$(echo "$HF_GPTQ" | tr '/' '_')"
if [ ! -d "\$GPTQ_DIR" ] || [ ! -f "\$GPTQ_DIR/config.json" ]; then
    huggingface-cli download "$HF_GPTQ" --local-dir "\$GPTQ_DIR"
else
    echo "  GPTQ already downloaded"
fi

# ── Download AWQ ──
echo ""
echo "=== Downloading AWQ: $HF_AWQ ==="
AWQ_DIR="\$WORK/models/$(echo "$HF_AWQ" | tr '/' '_')"
if [ ! -d "\$AWQ_DIR" ] || [ ! -f "\$AWQ_DIR/config.json" ]; then
    huggingface-cli download "$HF_AWQ" --local-dir "\$AWQ_DIR"
else
    echo "  AWQ already downloaded"
fi

# ── Verify ──
echo ""
echo "=== Verifying models ==="
echo "  Dense:     \$(ls \$MODEL_DIR/*.safetensors 2>/dev/null | wc -l) safetensors"
echo "  CDNA k256: \$(ls -d \$MODEL_DIR/cdnav3/*.cdnav3 2>/dev/null | wc -l) tensors"
echo "  CDNA k64:  \$(ls -d \$MODEL_DIR/cdnav3_k64/*.cdnav3 2>/dev/null | wc -l) tensors"
echo "  GPTQ:      \$(ls \$GPTQ_DIR/*.safetensors 2>/dev/null | wc -l) safetensors"
echo "  AWQ:       \$(ls \$AWQ_DIR/*.safetensors 2>/dev/null | wc -l) safetensors"

# ── Run benchmark ──
echo ""
echo "=== Running 5-Config Comparison Benchmark ==="
cd "\$WORK/helix-substrate"

python3 tools/cloud_comparison_bench.py \
    --model-dir "\$MODEL_DIR" \
    --gptq-model "\$GPTQ_DIR" \
    --awq-model "\$AWQ_DIR" \
    --output-dir "\$WORK/receipts"

echo ""
echo "=== DONE ==="
find "\$WORK/receipts" -name "*.json" -type f | sort
REMOTE_SCRIPT

# ── Pull receipts back ──
echo ""
echo "[DONE] Pulling receipts back..."
echo "============================================================"

mkdir -p "$HELIX_DIR/receipts/cloud_comparison"
rsync -az "$REMOTE:$WORK/receipts/" "$HELIX_DIR/receipts/"

echo ""
echo "Receipts pulled to: $HELIX_DIR/receipts/"
find "$HELIX_DIR/receipts/cloud_comparison" -name "*.json" -type f 2>/dev/null | sort
echo ""
echo "============================================================"
echo " 5-CONFIG COMPARISON BENCHMARK COMPLETE ($MODEL_NAME)"
echo " Don't forget to tear down the instance!"
echo "============================================================"
