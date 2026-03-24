#!/usr/bin/env bash
# ============================================================
# cloud_mega_bench_setup.sh — Ship code + models to rented GPU
# and run the full 5-test mega-benchmark.
#
# Usage:
#   ./tools/cloud_mega_bench_setup.sh <user@remote-ip>
#
# Example:
#   ./tools/cloud_mega_bench_setup.sh root@209.38.1.123
#
# What it does:
#   1. Ships helix-substrate code (no receipts, no .git)
#   2. Ships compressed models (Qwen 7B, Qwen 3B, Mamba) — CDNA only
#   3. Downloads dense weights from HuggingFace on the remote
#   4. Installs Python deps
#   5. Runs cloud_mega_bench.py
#   6. Pulls all receipts back to local
#
# Requirements:
#   - Remote: Ubuntu + CUDA + Python 3.10+
#   - RunPod / TensorDock / Lambda defaults are fine
#   - RTX 4090 (24 GB) recommended
# ============================================================

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <user@remote-ip>"
    echo "Example: $0 root@209.38.1.123"
    exit 1
fi

REMOTE="$1"
HELIX_DIR="$(cd "$(dirname "$0")/.." && pwd)"
WORK="/home/user/helix_bench"

# Model local paths
MODEL_7B_NAME="qwen2.5-7b-instruct"
MODEL_3B_NAME="qwen2.5-3b-instruct"
MAMBA_NAME="mamba-130m-hf"

MODEL_7B_DIR="$HOME/models/$MODEL_7B_NAME"
MODEL_3B_DIR="$HOME/models/$MODEL_3B_NAME"
MAMBA_DIR="$HOME/models/$MAMBA_NAME"

# HuggingFace repo names (for remote download)
HF_7B="Qwen/Qwen2.5-7B-Instruct"
HF_3B="Qwen/Qwen2.5-3B-Instruct"
HF_MAMBA="state-spaces/mamba-130m-hf"

echo "============================================================"
echo " HELIX CLOUD MEGA-BENCHMARK SETUP"
echo "============================================================"
echo " Remote:    $REMOTE"
echo " Workspace: $WORK"
echo " Code:      $HELIX_DIR"
echo ""
echo " Models to ship:"
echo "   Qwen 7B:  $MODEL_7B_DIR"
echo "   Qwen 3B:  $MODEL_3B_DIR"
echo "   Mamba:    $MAMBA_DIR"
echo "============================================================"

# ── Validate local models ──
for MODEL_NAME in "$MODEL_7B_NAME" "$MODEL_3B_NAME" "$MAMBA_NAME"; do
    LOCAL_DIR="$HOME/models/$MODEL_NAME"
    CDNA="$LOCAL_DIR/cdnav3"
    if [ ! -d "$CDNA" ]; then
        echo "ERROR: $CDNA not found. Compression not complete?"
        exit 1
    fi
    N_TENSORS=$(ls -d "$CDNA"/*.cdnav3 2>/dev/null | wc -l)
    echo "  $MODEL_NAME: $N_TENSORS CDNA tensors"
    if [ "$N_TENSORS" -lt 5 ]; then
        echo "ERROR: Only $N_TENSORS tensors for $MODEL_NAME. Compression likely incomplete."
        exit 1
    fi
done

# ── Step 1: Create code tarball ──
echo ""
echo "[1/6] Packing helix-substrate code..."
CODE_TAR="/tmp/helix_substrate_mega.tar.gz"
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
    "$(basename "$HELIX_DIR")/tools/cloud_mega_bench.py" \
    "$(basename "$HELIX_DIR")/tools/cloud_bench_run.py" \
    "$(basename "$HELIX_DIR")/tools/cloud_mega_bench_setup.sh"

CODE_SIZE=$(du -sh "$CODE_TAR" | cut -f1)
echo "  Code tarball: $CODE_SIZE"

# ── Step 2: Ship code ──
echo ""
echo "[2/6] Shipping code to $REMOTE..."
ssh "$REMOTE" "mkdir -p $WORK"
scp "$CODE_TAR" "$REMOTE:$WORK/helix_substrate_mega.tar.gz"
ssh "$REMOTE" "cd $WORK && tar xzf helix_substrate_mega.tar.gz"

# ── Step 3: Ship compressed models (CDNA only + config/tokenizer) ──
echo ""
echo "[3/6] Shipping compressed models (CDNA dirs + tokenizer configs)..."

ship_model() {
    local MODEL_NAME="$1"
    local LOCAL_DIR="$HOME/models/$MODEL_NAME"
    local REMOTE_DIR="$WORK/models/$MODEL_NAME"

    echo "  Shipping $MODEL_NAME..."
    ssh "$REMOTE" "mkdir -p $REMOTE_DIR"

    rsync -az --progress \
        --include='cdnav3/***' \
        --include='config.json' \
        --include='tokenizer.json' \
        --include='tokenizer_config.json' \
        --include='merges.txt' \
        --include='vocab.json' \
        --include='generation_config.json' \
        --include='special_tokens_map.json' \
        --include='added_tokens.json' \
        --exclude='*.safetensors' \
        --exclude='*.safetensors.index.json' \
        --exclude='*.bin' \
        --exclude='.cache/***' \
        --exclude='README.md' \
        --exclude='LICENSE' \
        "$LOCAL_DIR/" "$REMOTE:$REMOTE_DIR/"
}

ship_model "$MODEL_7B_NAME"
ship_model "$MODEL_3B_NAME"
ship_model "$MAMBA_NAME"

# ── Step 4 + 5: Install deps, download dense weights, run benchmark ──
echo ""
echo "[4/6] Installing deps + downloading dense weights on $REMOTE..."
echo "[5/6] Running mega-benchmark..."

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
    huggingface_hub mamba_ssm 2>&1 | tail -3

echo ""
echo "=== GPU Info ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || echo "no nvidia-smi"

echo ""
echo "=== Checking CUDA ==="
python3 -c "import torch; print(f'torch {torch.__version__}, cuda={torch.cuda.is_available()}')"

# ── Download dense weights from HuggingFace ──
echo ""
echo "=== Downloading dense weights from HuggingFace ==="

download_dense() {
    local MODEL_NAME="\$1"
    local HF_REPO="\$2"
    local MODEL_DIR="\$WORK/models/\$MODEL_NAME"

    HAS_WEIGHTS=0
    if [ -f "\$MODEL_DIR/model.safetensors" ]; then HAS_WEIGHTS=1; fi
    if [ -f "\$MODEL_DIR/model.safetensors.index.json" ]; then HAS_WEIGHTS=1; fi
    if [ -f "\$MODEL_DIR/pytorch_model.bin" ]; then HAS_WEIGHTS=1; fi

    if [ "\$HAS_WEIGHTS" -eq 0 ]; then
        echo "  Downloading \$HF_REPO -> \$MODEL_DIR ..."
        # Download only weight files + index (CDNA and config already shipped)
        huggingface-cli download "\$HF_REPO" --local-dir "\$MODEL_DIR" \
            --include "*.safetensors" "*.safetensors.index.json" "*.bin"
    else
        echo "  \$MODEL_NAME: dense weights already present"
    fi
}

download_dense "$MODEL_7B_NAME" "$HF_7B"
download_dense "$MODEL_3B_NAME" "$HF_3B"
download_dense "$MAMBA_NAME" "$HF_MAMBA"

# ── Verify all models are complete ──
echo ""
echo "=== Verifying models ==="
for M in "$MODEL_7B_NAME" "$MODEL_3B_NAME" "$MAMBA_NAME"; do
    D="\$WORK/models/\$M"
    CDNA_COUNT=\$(ls -d "\$D/cdnav3/"*.cdnav3 2>/dev/null | wc -l)
    HAS_W="no"
    if [ -f "\$D/model.safetensors" ] || [ -f "\$D/model.safetensors.index.json" ] || [ -f "\$D/pytorch_model.bin" ]; then
        HAS_W="yes"
    fi
    echo "  \$M: cdna_tensors=\$CDNA_COUNT, dense_weights=\$HAS_W"
done

# ── Run mega-benchmark ──
echo ""
echo "=== Running Mega-Benchmark ==="
cd "\$WORK/helix-substrate"

python3 tools/cloud_mega_bench.py \
    --model-dir "\$WORK/models/$MODEL_7B_NAME" \
    --model-dir-3b "\$WORK/models/$MODEL_3B_NAME" \
    --mamba-dir "\$WORK/models/$MAMBA_NAME" \
    --output-dir "\$WORK/receipts"

echo ""
echo "=== DONE ==="
find "\$WORK/receipts" -name "*.json" -type f | sort
REMOTE_SCRIPT

# ── Step 6: Pull receipts back ──
echo ""
echo "[6/6] Pulling receipts back..."
echo "============================================================"

# Create local receipt dirs
for SUBDIR in cloud_bench wide_r4 dual_model_load mamba_gpu; do
    mkdir -p "$HELIX_DIR/receipts/$SUBDIR"
done

rsync -az "$REMOTE:$WORK/receipts/" "$HELIX_DIR/receipts/"

echo ""
echo "Receipts pulled to: $HELIX_DIR/receipts/"
find "$HELIX_DIR/receipts" -name "*.json" -newer "$CODE_TAR" -type f | sort
echo ""
echo "============================================================"
echo " MEGA-BENCHMARK COMPLETE"
echo " Don't forget to tear down the instance!"
echo "============================================================"
