#!/usr/bin/env bash
# ============================================================
# ship_to_cloud.sh — Pack and ship everything to a rented GPU
#
# Usage:
#   ./tools/ship_to_cloud.sh <user@remote-ip> [model_name]
#
# Example:
#   ./tools/ship_to_cloud.sh root@209.38.1.123
#   ./tools/ship_to_cloud.sh root@209.38.1.123 qwen2.5-7b-instruct
#
# What it does:
#   1. Tarballs helix-substrate code (no receipts, no .git)
#   2. Tarballs compressed model (CDNA only, no dense safetensors)
#   3. Ships both via rsync/scp
#   4. SSHs in and kicks off the benchmark
#
# The remote instance needs: Ubuntu + CUDA + Python 3.10+
# RunPod/TensorDock defaults are fine.
# ============================================================

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <user@remote-ip> [model_name]"
    echo "Example: $0 root@209.38.1.123 qwen2.5-7b-instruct"
    exit 1
fi

REMOTE="$1"
MODEL="${2:-qwen2.5-7b-instruct}"
MODEL_DIR="$HOME/models/$MODEL"
HELIX_DIR="$(cd "$(dirname "$0")/.." && pwd)"
WORK="/home/user/helix_bench"

echo "============================================================"
echo " Shipping to $REMOTE"
echo " Model: $MODEL"
echo " Code:  $HELIX_DIR"
echo "============================================================"

# ── Step 1: Check model is ready ──
CDNA_DIR="$MODEL_DIR/cdnav3"
if [ ! -d "$CDNA_DIR" ]; then
    echo "ERROR: $CDNA_DIR not found. Compression not complete?"
    exit 1
fi

N_TENSORS=$(ls -d "$CDNA_DIR"/*.cdnav3 2>/dev/null | wc -l)
echo "CDNA tensors: $N_TENSORS"

if [ "$N_TENSORS" -lt 10 ]; then
    echo "ERROR: Only $N_TENSORS tensors. Compression likely incomplete."
    exit 1
fi

# ── Step 2: Create code tarball (exclude heavy stuff) ──
echo ""
echo "[1/4] Packing helix-substrate code..."
CODE_TAR="/tmp/helix_substrate_code.tar.gz"
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
    "$(basename "$HELIX_DIR")/tools/cloud_bench_run.py" \
    "$(basename "$HELIX_DIR")/tools/cloud_bench_setup.sh"

CODE_SIZE=$(du -sh "$CODE_TAR" | cut -f1)
echo "  Code tarball: $CODE_SIZE"

# ── Step 3: Ship code ──
echo ""
echo "[2/4] Shipping code to $REMOTE..."
ssh "$REMOTE" "mkdir -p $WORK"
scp "$CODE_TAR" "$REMOTE:$WORK/helix_substrate_code.tar.gz"
ssh "$REMOTE" "cd $WORK && tar xzf helix_substrate_code.tar.gz"

# ── Step 4: Ship compressed model (CDNA only + tokenizer/config) ──
echo ""
echo "[3/4] Shipping compressed model (this may take a while)..."
ssh "$REMOTE" "mkdir -p $WORK/models/$MODEL"

# rsync CDNA dir + config files, exclude dense weights
rsync -az --progress \
    --include='cdnav3/***' \
    --include='config.json' \
    --include='tokenizer.json' \
    --include='tokenizer_config.json' \
    --include='merges.txt' \
    --include='vocab.json' \
    --include='generation_config.json' \
    --exclude='*.safetensors' \
    --exclude='.cache/***' \
    --exclude='model.safetensors.index.json' \
    "$MODEL_DIR/" "$REMOTE:$WORK/models/$MODEL/"

# ── Step 5: Install deps and run ──
echo ""
echo "[4/4] Installing deps and running benchmark on $REMOTE..."
ssh "$REMOTE" bash -s <<REMOTE_SCRIPT
set -euo pipefail

WORK="/home/user/helix_bench"
cd "\$WORK"

echo "=== Installing Python deps ==="
pip3 install -q \
    'torch<2.11' --index-url https://download.pytorch.org/whl/cu126 2>&1 | tail -3
pip3 install -q \
    "transformers>=4.45" \
    safetensors scipy datasets numpy triton sentencepiece accelerate huggingface_hub 2>&1 | tail -3

echo ""
echo "=== GPU Info ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || echo "no nvidia-smi"

echo ""
echo "=== Checking CUDA ==="
python3 -c "import torch; print(f'torch {torch.__version__}, cuda={torch.cuda.is_available()}')"

# Find CDNA dir shipped from local
CDNA_SRC=\$(ls -d \$WORK/models/*/cdnav3 2>/dev/null | head -1)
CDNA_MODEL=\$(dirname "\$CDNA_SRC")
MODEL_NAME=\$(basename "\$CDNA_MODEL")

if [ -z "\$CDNA_SRC" ]; then
    echo "ERROR: No CDNA found in \$WORK/models/"
    exit 1
fi

echo "CDNA source: \$CDNA_SRC"
echo "CDNA tensors: \$(ls -d \$CDNA_SRC/*.cdnav3 2>/dev/null | wc -l)"

# Download full model weights from HuggingFace (fast datacenter bandwidth)
# Need real weights for embed_tokens/lm_head (HelixLinear only swaps nn.Linear)
FULL_DIR="\$WORK/models/\${MODEL_NAME}-full"
if [ ! -f "\$FULL_DIR/model.safetensors" ] && [ ! -f "\$FULL_DIR/model.safetensors.index.json" ]; then
    echo ""
    echo "=== Downloading full model from HuggingFace ==="
    HF_NAME="${MODEL}"
    # Map common local names to HF repo names
    case "\$MODEL_NAME" in
        qwen2.5-7b-instruct) HF_NAME="Qwen/Qwen2.5-7B-Instruct" ;;
        qwen2.5-3b-instruct) HF_NAME="Qwen/Qwen2.5-3B-Instruct" ;;
        *) HF_NAME="\$MODEL_NAME" ;;
    esac
    echo "  HF repo: \$HF_NAME → \$FULL_DIR"
    huggingface-cli download "\$HF_NAME" --local-dir "\$FULL_DIR"
fi

# Symlink CDNA into the full model dir
if [ ! -d "\$FULL_DIR/cdnav3" ]; then
    ln -s "\$CDNA_SRC" "\$FULL_DIR/cdnav3"
    echo "Symlinked CDNA: \$CDNA_SRC → \$FULL_DIR/cdnav3"
fi

echo ""
echo "=== Running benchmark ==="
cd "\$WORK/helix-substrate"

python3 tools/cloud_bench_run.py \
    --model-dir "\$FULL_DIR" \
    --model-name "\$MODEL_NAME" \
    --output-dir "\$WORK/receipts"

echo ""
echo "=== DONE ==="
ls -la "\$WORK/receipts/"
REMOTE_SCRIPT

# ── Step 6: Pull receipts back ──
echo ""
echo "============================================================"
echo " Pulling receipts back..."
echo "============================================================"
RECEIPT_DIR="$HELIX_DIR/receipts/cloud_bench"
mkdir -p "$RECEIPT_DIR"
rsync -az "$REMOTE:$WORK/receipts/" "$RECEIPT_DIR/"
echo "Receipts at: $RECEIPT_DIR/"
ls -la "$RECEIPT_DIR/"

echo ""
echo "============================================================"
echo " DONE. Don't forget to tear down the instance."
echo "============================================================"
