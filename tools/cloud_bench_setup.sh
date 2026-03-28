#!/usr/bin/env bash
# ============================================================
# cloud_bench_setup.sh — One-command cloud GPU benchmark setup
#
# Usage (on rented instance):
#   curl -sL <your-url>/cloud_bench_setup.sh | bash
#   # OR: scp this to the instance and run it
#
# What it does:
#   1. Installs Python deps
#   2. Pulls helix-substrate from your box (or git)
#   3. Copies compressed model from your box
#   4. Runs full benchmark suite
#   5. Outputs receipt JSON
#
# Prerequisites:
#   - Ubuntu 22.04+ with CUDA (RunPod/TensorDock default)
#   - SSH access back to your box (for rsync), OR pre-uploaded tarball
# ============================================================

set -euo pipefail

ECHO_BOX="${USER}@<YOUR_TAILSCALE_IP_OR_HOSTNAME>"
WORK_DIR="/workspace/helix_bench"
MODEL_NAME="qwen2.5-7b-instruct"

echo "============================================================"
echo " Helix Cloud Benchmark Setup"
echo " $(date -Iseconds)"
echo " GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'no nvidia-smi')"
echo "============================================================"

# ── Step 1: System deps ──
echo "[1/6] Installing system dependencies..."
apt-get update -qq && apt-get install -y -qq rsync python3-pip > /dev/null 2>&1 || true

# ── Step 2: Python deps ──
echo "[2/6] Installing Python packages..."
pip3 install -q \
    torch torchvision torchaudio \
    transformers==4.57.0 \
    safetensors==0.6.2 \
    scipy==1.15.3 \
    datasets==4.1.1 \
    numpy \
    triton \
    sentencepiece \
    accelerate

# ── Step 3: Pull code ──
echo "[3/6] Setting up workspace..."
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

# Option A: rsync from your box (uncomment and fill in)
# rsync -az --progress "${ECHO_BOX}:~/helix-substrate/" ./helix-substrate/

# Option B: If you pre-uploaded a tarball
# tar xzf /workspace/helix-substrate.tar.gz

# Option C: git clone (if you push to a repo)
# git clone https://github.com/youruser/helix-substrate.git

echo "  Using helix-substrate at: $(pwd)/helix-substrate"

# ── Step 4: Pull compressed model ──
echo "[4/6] Syncing compressed model..."
MODEL_DIR="$WORK_DIR/models/$MODEL_NAME"
mkdir -p "$MODEL_DIR"

# Sync CDNA v3 compressed dir + config/tokenizer (NOT the dense safetensors)
# rsync -az --progress \
#     --include='cdnav3/***' \
#     --include='config.json' \
#     --include='tokenizer.json' \
#     --include='tokenizer_config.json' \
#     --include='merges.txt' \
#     --include='vocab.json' \
#     --include='generation_config.json' \
#     --exclude='*.safetensors' \
#     --exclude='.cache/***' \
#     "${ECHO_BOX}:~/models/${MODEL_NAME}/" "$MODEL_DIR/"

echo "  Model at: $MODEL_DIR"
echo "  CDNA tensors: $(ls "$MODEL_DIR/cdnav3/" 2>/dev/null | wc -l)"

# ── Step 5: Run benchmark ──
echo "[5/6] Running benchmark..."
cd "$WORK_DIR/helix-substrate"

python3 tools/cloud_bench_run.py \
    --model-dir "$MODEL_DIR" \
    --model-name "$MODEL_NAME" \
    --output-dir "$WORK_DIR/receipts"

# ── Step 6: Summary ──
echo ""
echo "============================================================"
echo " DONE. Receipts at: $WORK_DIR/receipts/"
echo "============================================================"
echo ""
echo "To copy receipts back to your box:"
echo "  rsync -az $WORK_DIR/receipts/ ${ECHO_BOX}:~/helix-substrate/receipts/cloud_bench/"
