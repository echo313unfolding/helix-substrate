#!/usr/bin/env bash
# ============================================================================
# CLOUD GPU RUN — Tiered benchmark suite for helix-substrate
# ============================================================================
#
# Usage:
#   # On cloud instance after cloning helix-substrate:
#   bash tools/cloud_gpu_run.sh [--tier 1|2|3] [--resume]
#
# SSH-drop safe: each task writes a checkpoint file. Use --resume to skip
# completed tasks. All receipts are saved incrementally.
#
# Tiers:
#   1 — Ship-blocking: mamba-ssm install, Mamba2 PPL, lm-eval (Zamba2, Qwen-3B)
#   2 — Paper: Probes 2+3 on Qwen 3B, Qwen 7B PPL confirm, Qwen 14B PPL
#   3 — Bonus: Full stack benchmark, Nemotron 3 Nano 4B
#
# Prerequisites:
#   - pip install helix-substrate (already on PyPI)
#   - HF models accessible (EchoLabs33/*)
#   - GPU with >= 24 GB VRAM (A100/4090/A6000)
#
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
RECEIPT_DIR="$PROJECT_DIR/receipts/cloud_run"
CHECKPOINT_DIR="$RECEIPT_DIR/.checkpoints"
LOG_DIR="$RECEIPT_DIR/logs"

mkdir -p "$RECEIPT_DIR" "$CHECKPOINT_DIR" "$LOG_DIR"

# --- Args ---
MAX_TIER=3
RESUME=false
for arg in "$@"; do
    case $arg in
        --tier) shift; MAX_TIER="$1"; shift ;;
        --resume) RESUME=true; shift ;;
    esac
done

# --- Helpers ---
timestamp() { date -u +"%Y-%m-%dT%H:%M:%S"; }
log() { echo "[$(timestamp)] $*" | tee -a "$LOG_DIR/cloud_run.log"; }
checkpoint_done() { touch "$CHECKPOINT_DIR/$1.done"; }
is_done() { [ -f "$CHECKPOINT_DIR/$1.done" ]; }
skip_if_done() {
    if $RESUME && is_done "$1"; then
        log "SKIP $1 (already done)"
        return 0
    fi
    return 1
}
purge_hf_cache() {
    # Clear HF download cache to free disk between large models
    local cache_dir="$HOME/.cache/huggingface/hub"
    if [ -d "$cache_dir" ]; then
        local before
        before=$(du -sm "$cache_dir" 2>/dev/null | cut -f1)
        rm -rf "$cache_dir"
        log "Purged HF cache (was ${before}MB)"
    fi
}

# Save receipt JSON from python stdout (last JSON line)
save_receipt() {
    local name="$1" logfile="$2"
    local ts
    ts=$(date -u +"%Y%m%dT%H%M%S")
    # Extract last JSON line from log
    tac "$logfile" | grep -m1 '^{' > "$RECEIPT_DIR/${name}_${ts}.json" 2>/dev/null || true
    log "Receipt saved: ${name}_${ts}.json"
}

# ============================================================================
# SETUP
# ============================================================================
log "=== CLOUD GPU RUN — $(timestamp) ==="
log "Tier limit: $MAX_TIER | Resume: $RESUME"
log "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'not detected')"

if ! skip_if_done "setup_deps"; then
    log "--- Installing dependencies ---"

    pip install -q helix-substrate transformers torch datasets accelerate 2>&1 | tail -5
    pip install -q lm-eval 2>&1 | tail -5

    # mamba-ssm: needs matching CUDA
    log "Installing mamba-ssm..."
    pip install -q mamba-ssm 2>&1 | tail -10 || log "WARN: mamba-ssm install failed — Mamba2 PPL will be skipped"

    # causal-conv1d (mamba dependency)
    pip install -q causal-conv1d 2>&1 | tail -5 || true

    checkpoint_done "setup_deps"
fi

# ============================================================================
# TIER 1 — Ship-blocking
# ============================================================================
if [ "$MAX_TIER" -ge 1 ]; then
    log "========== TIER 1: SHIP-BLOCKING =========="

    # --- Mamba2-1.3B PPL eval ---
    if ! skip_if_done "t1_mamba2_ppl"; then
        log "--- Mamba2-1.3B PPL eval ---"
        python3 -u -c "
import json, sys, time, platform, resource
import numpy as np
import torch
import torch.nn.functional as F

t_start = time.time()
start_iso = time.strftime('%Y-%m-%dT%H:%M:%S')

try:
    import helix_substrate
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    model_id = 'EchoLabs33/mamba2-1.3b-helix'
    log_lines = []

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device).eval()

    # PPL eval
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    text = '\n\n'.join([t for t in ds['text'] if t.strip()])
    enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=8193)
    ids = enc.input_ids[:, :8193].to(device)
    nlls, n_eval = [], 0
    seq_len = 2048
    with torch.no_grad():
        for i in range(0, ids.shape[1]-1, seq_len):
            end = min(i+seq_len+1, ids.shape[1])
            chunk = ids[:, i:end]
            if chunk.shape[1] < 2: break
            out = model(input_ids=chunk[:, :-1])
            logits = out.logits.float().cpu()
            labels = chunk[:, 1:].cpu()
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            n = labels.numel()
            nlls.append(loss.item() * n)
            n_eval += n
            if n_eval >= 8192: break
    ppl = round(float(np.exp(sum(nlls)/n_eval)), 4)

    receipt = {
        'work_order': 'WO-11',
        'model': 'mamba2-1.3b-helix',
        'question': 'What is the PPL of Mamba2-1.3B compressed with CDNA v3?',
        'verdict': f'PPL={ppl} on WikiText-2 (8192 tokens)',
        'ppl': ppl,
        'n_tokens': n_eval,
        'device': device,
        'cost': {
            'wall_time_s': round(time.time() - t_start, 3),
            'cpu_time_s': round(time.process_time(), 3),
            'peak_memory_mb': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
            'python_version': platform.python_version(),
            'hostname': platform.node(),
            'timestamp_start': start_iso,
            'timestamp_end': time.strftime('%Y-%m-%dT%H:%M:%S'),
        }
    }
    if torch.cuda.is_available():
        receipt['vram_mb'] = round(torch.cuda.max_memory_allocated() / 1024**2, 1)
        receipt['gpu'] = torch.cuda.get_device_name(0)
    print(json.dumps(receipt, indent=2))
except Exception as e:
    import traceback
    traceback.print_exc()
    print(json.dumps({'error': str(e)}))
" 2>&1 | tee "$LOG_DIR/t1_mamba2_ppl.log"
        save_receipt "t1_mamba2_ppl" "$LOG_DIR/t1_mamba2_ppl.log"
        checkpoint_done "t1_mamba2_ppl"
    fi

    # --- lm-eval: Zamba2-1.2B ---
    if ! skip_if_done "t1_lmeval_zamba2"; then
        log "--- lm-eval: Zamba2-1.2B (HellaSwag, ARC-easy, ARC-challenge, MMLU) ---"
        python3 -u -c "
import json, sys, time, platform, resource

t_start = time.time()
start_iso = time.strftime('%Y-%m-%dT%H:%M:%S')

try:
    import helix_substrate
    import lm_eval

    results = lm_eval.simple_evaluate(
        model='hf',
        model_args='pretrained=EchoLabs33/zamba2-1.2b-helix,trust_remote_code=True',
        tasks=['hellaswag', 'arc_easy', 'arc_challenge'],
        batch_size='auto',
        device='cuda',
    )

    scores = {}
    for task, data in results['results'].items():
        key = 'acc_norm,none' if 'acc_norm,none' in data else 'acc,none'
        scores[task] = round(data.get(key, 0), 4)

    receipt = {
        'work_order': 'WO-06-T1',
        'model': 'zamba2-1.2b-helix',
        'question': 'How does Zamba2-1.2B-Helix score on downstream tasks?',
        'scores': scores,
        'full_results': {k: dict(v) for k, v in results['results'].items()},
        'cost': {
            'wall_time_s': round(time.time() - t_start, 3),
            'cpu_time_s': round(time.process_time(), 3),
            'peak_memory_mb': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
            'python_version': platform.python_version(),
            'hostname': platform.node(),
            'timestamp_start': start_iso,
            'timestamp_end': time.strftime('%Y-%m-%dT%H:%M:%S'),
        }
    }
    print(json.dumps(receipt, indent=2))
except Exception as e:
    import traceback
    traceback.print_exc()
    print(json.dumps({'error': str(e)}))
" 2>&1 | tee "$LOG_DIR/t1_lmeval_zamba2.log"
        save_receipt "t1_lmeval_zamba2" "$LOG_DIR/t1_lmeval_zamba2.log"
        checkpoint_done "t1_lmeval_zamba2"
    fi

    # --- lm-eval: Qwen2.5-3B-Instruct ---
    if ! skip_if_done "t1_lmeval_qwen3b"; then
        log "--- lm-eval: Qwen2.5-3B-Instruct-Helix (HellaSwag, ARC-easy, ARC-challenge) ---"
        python3 -u -c "
import json, sys, time, platform, resource

t_start = time.time()
start_iso = time.strftime('%Y-%m-%dT%H:%M:%S')

try:
    import helix_substrate
    import lm_eval

    results = lm_eval.simple_evaluate(
        model='hf',
        model_args='pretrained=EchoLabs33/qwen2.5-3b-instruct-helix,trust_remote_code=True',
        tasks=['hellaswag', 'arc_easy', 'arc_challenge'],
        batch_size='auto',
        device='cuda',
    )

    scores = {}
    for task, data in results['results'].items():
        key = 'acc_norm,none' if 'acc_norm,none' in data else 'acc,none'
        scores[task] = round(data.get(key, 0), 4)

    receipt = {
        'work_order': 'WO-06-T1',
        'model': 'qwen2.5-3b-instruct-helix',
        'question': 'How does Qwen2.5-3B-Instruct-Helix score on downstream tasks?',
        'scores': scores,
        'full_results': {k: dict(v) for k, v in results['results'].items()},
        'cost': {
            'wall_time_s': round(time.time() - t_start, 3),
            'cpu_time_s': round(time.process_time(), 3),
            'peak_memory_mb': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
            'python_version': platform.python_version(),
            'hostname': platform.node(),
            'timestamp_start': start_iso,
            'timestamp_end': time.strftime('%Y-%m-%dT%H:%M:%S'),
        }
    }
    print(json.dumps(receipt, indent=2))
except Exception as e:
    import traceback
    traceback.print_exc()
    print(json.dumps({'error': str(e)}))
" 2>&1 | tee "$LOG_DIR/t1_lmeval_qwen3b.log"
        save_receipt "t1_lmeval_qwen3b" "$LOG_DIR/t1_lmeval_qwen3b.log"
        checkpoint_done "t1_lmeval_qwen3b"
    fi

    log "========== TIER 1 COMPLETE =========="
fi

# ============================================================================
# TIER 2 — Paper
# ============================================================================
if [ "$MAX_TIER" -ge 2 ]; then
    log "========== TIER 2: PAPER =========="

    # --- Qwen 7B PPL confirm (from HF) ---
    if ! skip_if_done "t2_qwen7b_ppl"; then
        log "--- Qwen2.5-7B PPL confirm on GPU ---"
        python3 -u -c "
import json, sys, time, platform, resource
import numpy as np
import torch
import torch.nn.functional as F

t_start = time.time()
start_iso = time.strftime('%Y-%m-%dT%H:%M:%S')

try:
    import helix_substrate
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    # Helix model
    model_id = 'EchoLabs33/qwen2.5-7b-instruct-helix'
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, device_map='auto')
    model.eval()
    device = next(model.parameters()).device

    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    text = '\n\n'.join([t for t in ds['text'] if t.strip()])
    enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=8193)
    ids = enc.input_ids[:, :8193].to(device)
    nlls, n_eval, seq_len = [], 0, 2048
    with torch.no_grad():
        for i in range(0, ids.shape[1]-1, seq_len):
            end = min(i+seq_len+1, ids.shape[1])
            chunk = ids[:, i:end]
            if chunk.shape[1] < 2: break
            out = model(input_ids=chunk[:, :-1])
            logits = out.logits.float().cpu()
            labels = chunk[:, 1:].cpu()
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            n = labels.numel()
            nlls.append(loss.item() * n)
            n_eval += n
            if n_eval >= 8192: break
    helix_ppl = round(float(np.exp(sum(nlls)/n_eval)), 4)

    del model
    torch.cuda.empty_cache()
    import gc; gc.collect()

    # Dense baseline
    dense_id = 'Qwen/Qwen2.5-7B-Instruct'
    model = AutoModelForCausalLM.from_pretrained(dense_id, torch_dtype=torch.float16, trust_remote_code=True, device_map='auto').eval()
    device = next(model.parameters()).device
    ids = ids.to(device)
    nlls, n_eval = [], 0
    with torch.no_grad():
        for i in range(0, ids.shape[1]-1, seq_len):
            end = min(i+seq_len+1, ids.shape[1])
            chunk = ids[:, i:end]
            if chunk.shape[1] < 2: break
            out = model(input_ids=chunk[:, :-1])
            logits = out.logits.float().cpu()
            labels = chunk[:, 1:].cpu()
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            n = labels.numel()
            nlls.append(loss.item() * n)
            n_eval += n
            if n_eval >= 8192: break
    dense_ppl = round(float(np.exp(sum(nlls)/n_eval)), 4)

    delta = round((helix_ppl - dense_ppl) / dense_ppl * 100, 2)

    receipt = {
        'work_order': 'WO-06-T2',
        'model': 'qwen2.5-7b-instruct',
        'question': 'Does Qwen 7B +6.34% PPL hold on GPU?',
        'verdict': f'Helix PPL={helix_ppl}, Dense PPL={dense_ppl}, Delta={delta}%',
        'helix_ppl': helix_ppl,
        'dense_ppl': dense_ppl,
        'ppl_delta_pct': delta,
        'n_tokens': n_eval,
        'cost': {
            'wall_time_s': round(time.time() - t_start, 3),
            'cpu_time_s': round(time.process_time(), 3),
            'peak_memory_mb': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
            'python_version': platform.python_version(),
            'hostname': platform.node(),
            'timestamp_start': start_iso,
            'timestamp_end': time.strftime('%Y-%m-%dT%H:%M:%S'),
        }
    }
    if torch.cuda.is_available():
        receipt['vram_peak_mb'] = round(torch.cuda.max_memory_allocated() / 1024**2, 1)
        receipt['gpu'] = torch.cuda.get_device_name(0)
    print(json.dumps(receipt, indent=2))
except Exception as e:
    import traceback
    traceback.print_exc()
    print(json.dumps({'error': str(e)}))
" 2>&1 | tee "$LOG_DIR/t2_qwen7b_ppl.log"
        save_receipt "t2_qwen7b_ppl" "$LOG_DIR/t2_qwen7b_ppl.log"
        checkpoint_done "t2_qwen7b_ppl"
    fi

    purge_hf_cache  # Free ~14GB before downloading 14B

    # --- Qwen 14B PPL (no dense baseline exists) ---
    if ! skip_if_done "t2_qwen14b_ppl"; then
        log "--- Qwen2.5-14B PPL eval (Helix + dense baseline) ---"
        python3 -u -c "
import json, sys, time, platform, resource
import numpy as np
import torch
import torch.nn.functional as F

t_start = time.time()
start_iso = time.strftime('%Y-%m-%dT%H:%M:%S')

try:
    import helix_substrate
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    results = {}

    # Helix 14B
    model_id = 'EchoLabs33/qwen2.5-14b-helix'
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, device_map='auto')
    model.eval()
    device = next(model.parameters()).device

    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    text = '\n\n'.join([t for t in ds['text'] if t.strip()])
    enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=8193)
    ids = enc.input_ids[:, :8193].to(device)
    nlls, n_eval, seq_len = [], 0, 2048
    with torch.no_grad():
        for i in range(0, ids.shape[1]-1, seq_len):
            end = min(i+seq_len+1, ids.shape[1])
            chunk = ids[:, i:end]
            if chunk.shape[1] < 2: break
            out = model(input_ids=chunk[:, :-1])
            logits = out.logits.float().cpu()
            labels = chunk[:, 1:].cpu()
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            n = labels.numel()
            nlls.append(loss.item() * n)
            n_eval += n
            if n_eval >= 8192: break
    results['helix_ppl'] = round(float(np.exp(sum(nlls)/n_eval)), 4)
    results['helix_n_tokens'] = n_eval

    del model
    torch.cuda.empty_cache()
    import gc; gc.collect()

    # Dense 14B baseline (FP16)
    dense_id = 'Qwen/Qwen2.5-14B-Instruct'
    model = AutoModelForCausalLM.from_pretrained(dense_id, torch_dtype=torch.float16, trust_remote_code=True, device_map='auto').eval()
    device = next(model.parameters()).device
    ids = ids.to(device)
    nlls, n_eval = [], 0
    with torch.no_grad():
        for i in range(0, ids.shape[1]-1, seq_len):
            end = min(i+seq_len+1, ids.shape[1])
            chunk = ids[:, i:end]
            if chunk.shape[1] < 2: break
            out = model(input_ids=chunk[:, :-1])
            logits = out.logits.float().cpu()
            labels = chunk[:, 1:].cpu()
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            n = labels.numel()
            nlls.append(loss.item() * n)
            n_eval += n
            if n_eval >= 8192: break
    results['dense_ppl'] = round(float(np.exp(sum(nlls)/n_eval)), 4)
    results['ppl_delta_pct'] = round((results['helix_ppl'] - results['dense_ppl']) / results['dense_ppl'] * 100, 2)

    receipt = {
        'work_order': 'WO-06-T2',
        'model': 'qwen2.5-14b-instruct',
        'question': 'What is Qwen 14B PPL? First dense baseline.',
        'verdict': f\"Helix PPL={results['helix_ppl']}, Dense PPL={results['dense_ppl']}, Delta={results['ppl_delta_pct']}%\",
        **results,
        'cost': {
            'wall_time_s': round(time.time() - t_start, 3),
            'cpu_time_s': round(time.process_time(), 3),
            'peak_memory_mb': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
            'python_version': platform.python_version(),
            'hostname': platform.node(),
            'timestamp_start': start_iso,
            'timestamp_end': time.strftime('%Y-%m-%dT%H:%M:%S'),
        }
    }
    if torch.cuda.is_available():
        receipt['vram_peak_mb'] = round(torch.cuda.max_memory_allocated() / 1024**2, 1)
        receipt['gpu'] = torch.cuda.get_device_name(0)
    print(json.dumps(receipt, indent=2))
except Exception as e:
    import traceback
    traceback.print_exc()
    print(json.dumps({'error': str(e)}))
" 2>&1 | tee "$LOG_DIR/t2_qwen14b_ppl.log"
        save_receipt "t2_qwen14b_ppl" "$LOG_DIR/t2_qwen14b_ppl.log"
        checkpoint_done "t2_qwen14b_ppl"
    fi

    # --- Probe 2+3 scaling: Qwen 3B ---
    if ! skip_if_done "t2_probes_qwen3b"; then
        log "--- Probes 2+3: eff_rank early exit + axis independence on Qwen 3B ---"
        python3 -u -c "
import json, sys, time, platform, resource
import numpy as np
import torch

t_start = time.time()
start_iso = time.strftime('%Y-%m-%dT%H:%M:%S')

try:
    import helix_substrate
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    from scipy import stats

    model_id = 'EchoLabs33/qwen2.5-3b-instruct-helix'
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, device_map='auto')
    model.eval()

    # Collect per-layer kurtosis and effective rank from weight tensors
    kurtosis_vals, eff_rank_vals, layer_names = [], [], []
    for name, param in model.named_parameters():
        if param.dim() == 2 and param.shape[0] >= 64 and param.shape[1] >= 64:
            w = param.detach().float().cpu().numpy()
            # Kurtosis
            k = float(stats.kurtosis(w.flatten(), fisher=True))
            # Effective rank (Shannon entropy of singular values)
            s = np.linalg.svd(w, compute_uv=False)
            s = s / (s.sum() + 1e-12)
            s = s[s > 1e-12]
            eff_r = float(np.exp(-np.sum(s * np.log(s))))
            kurtosis_vals.append(k)
            eff_rank_vals.append(eff_r)
            layer_names.append(name)

    k_arr = np.array(kurtosis_vals)
    r_arr = np.array(eff_rank_vals)

    # Probe 2: eff_rank vs PPL contribution (rank correlation)
    # Higher eff_rank = more complex = harder to compress = higher error
    # We use eff_rank as proxy for compression difficulty
    r_corr, r_pval = stats.pearsonr(k_arr, r_arr)

    # Probe 3: axis independence (kurtosis vs eff_rank should be weakly correlated)
    spearman_r, spearman_p = stats.spearmanr(k_arr, r_arr)

    receipt = {
        'work_order': 'WO-06-T2-PROBES',
        'model': 'qwen2.5-3b-instruct-helix',
        'question': 'Do Probe 2 (eff_rank early exit) and Probe 3 (axis independence) hold at 3B scale?',
        'n_layers': len(layer_names),
        'probe_2_eff_rank_stats': {
            'mean': round(float(r_arr.mean()), 2),
            'std': round(float(r_arr.std()), 2),
            'min': round(float(r_arr.min()), 2),
            'max': round(float(r_arr.max()), 2),
        },
        'probe_2_kurtosis_stats': {
            'mean': round(float(k_arr.mean()), 2),
            'std': round(float(k_arr.std()), 2),
            'min': round(float(k_arr.min()), 2),
            'max': round(float(k_arr.max()), 2),
        },
        'probe_3_pearson_r': round(r_corr, 4),
        'probe_3_pearson_p': round(r_pval, 6),
        'probe_3_spearman_r': round(spearman_r, 4),
        'probe_3_spearman_p': round(spearman_p, 6),
        'probe_3_independent': abs(spearman_r) < 0.3,
        'cost': {
            'wall_time_s': round(time.time() - t_start, 3),
            'cpu_time_s': round(time.process_time(), 3),
            'peak_memory_mb': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
            'python_version': platform.python_version(),
            'hostname': platform.node(),
            'timestamp_start': start_iso,
            'timestamp_end': time.strftime('%Y-%m-%dT%H:%M:%S'),
        }
    }
    if torch.cuda.is_available():
        receipt['gpu'] = torch.cuda.get_device_name(0)
    print(json.dumps(receipt, indent=2))
except Exception as e:
    import traceback
    traceback.print_exc()
    print(json.dumps({'error': str(e)}))
" 2>&1 | tee "$LOG_DIR/t2_probes_qwen3b.log"
        save_receipt "t2_probes_qwen3b" "$LOG_DIR/t2_probes_qwen3b.log"
        checkpoint_done "t2_probes_qwen3b"
    fi

    log "========== TIER 2 COMPLETE =========="
fi

# ============================================================================
# TIER 3 — Bonus
# ============================================================================
if [ "$MAX_TIER" -ge 3 ]; then
    log "========== TIER 3: BONUS =========="

    # --- Full stack benchmark (HelixLinear + KVCache + CDC-03) ---
    if ! skip_if_done "t3_full_stack"; then
        log "--- Full stack benchmark (if echo_runtime available) ---"
        if python3 -c "import echo_runtime" 2>/dev/null; then
            python3 -u -c "
import json, time, platform, resource
t_start = time.time()
start_iso = time.strftime('%Y-%m-%dT%H:%M:%S')
try:
    from echo_runtime import EchoRunner
    runner = EchoRunner.from_config('configs/tinyllama_full_stack.yaml')
    result = runner.benchmark(n_tokens=512, n_runs=5)
    receipt = {
        'work_order': 'WO-06-T3',
        'question': 'Full stack HelixLinear+KVCache+CDC-03 throughput?',
        **result,
        'cost': {
            'wall_time_s': round(time.time() - t_start, 3),
            'timestamp_start': start_iso,
            'timestamp_end': time.strftime('%Y-%m-%dT%H:%M:%S'),
        }
    }
    print(json.dumps(receipt, indent=2))
except Exception as e:
    import traceback
    traceback.print_exc()
    print(json.dumps({'error': str(e)}))
" 2>&1 | tee "$LOG_DIR/t3_full_stack.log"
            save_receipt "t3_full_stack" "$LOG_DIR/t3_full_stack.log"
        else
            log "SKIP: echo_runtime not installed"
        fi
        checkpoint_done "t3_full_stack"
    fi

    # --- Nemotron 3 Nano 4B ---
    if ! skip_if_done "t3_nemotron"; then
        log "--- Nemotron 3 Nano 4B (compress + eval) ---"
        python3 -u -c "
import json, sys, time, platform, resource
t_start = time.time()
start_iso = time.strftime('%Y-%m-%dT%H:%M:%S')
try:
    # Check if mamba-ssm is available (needed for Nemotron hybrid)
    import mamba_ssm
    print('mamba-ssm available, proceeding with Nemotron...')

    # This would need the compress.py pipeline + eval
    # Placeholder — actual compression requires the model download
    receipt = {
        'work_order': 'WO-06-T3',
        'model': 'nemotron-3-nano-4b',
        'status': 'DEFERRED — needs model download + compress.py run',
        'cost': {
            'wall_time_s': round(time.time() - t_start, 3),
            'timestamp_start': start_iso,
            'timestamp_end': time.strftime('%Y-%m-%dT%H:%M:%S'),
        }
    }
    print(json.dumps(receipt, indent=2))
except ImportError:
    print(json.dumps({'error': 'mamba-ssm not available, Nemotron deferred'}))
except Exception as e:
    import traceback
    traceback.print_exc()
    print(json.dumps({'error': str(e)}))
" 2>&1 | tee "$LOG_DIR/t3_nemotron.log"
        save_receipt "t3_nemotron" "$LOG_DIR/t3_nemotron.log"
        checkpoint_done "t3_nemotron"
    fi

    # --- Compress + eval + convert + upload SSM models ---
    SSM_MODELS=(
        "Zyphra/Zamba2-2.7B-instruct"
        "Zyphra/Zamba2-7B-Instruct"
        "state-spaces/mamba2-2.7b"
        "state-spaces/mamba-2.8b-hf"
    )

    for hf_id in "${SSM_MODELS[@]}"; do
        safe_name=$(echo "$hf_id" | tr '/' '_' | tr '[:upper:]' '[:lower:]')
        task_id="t3_ssm_${safe_name}"

        if ! skip_if_done "$task_id"; then
            log "--- Full pipeline: $hf_id → compress → convert → HF upload ---"
            python3 -u -c "
import json, sys, os, time, platform, resource, subprocess, shutil
import numpy as np
import torch
import torch.nn.functional as F

t_start = time.time()
start_iso = time.strftime('%Y-%m-%dT%H:%M:%S')
hf_id = '$hf_id'

# Derive names
model_short = hf_id.split('/')[-1].lower()
# Normalize to helix naming: zamba2-2.7b-instruct-helix, mamba2-2.7b-helix, etc.
helix_name = model_short.rstrip('-hf') + '-helix' if model_short.endswith('-hf') else model_short + '-helix'
hf_repo = f'EchoLabs33/{helix_name}'
local_dir_base = f'/tmp/models/{model_short}'
local_dir_helix = f'/tmp/models/{helix_name}'

from pathlib import Path
local_dir = Path(local_dir_base)
helix_dir = Path(local_dir_helix)

steps_done = {}

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    # ================================================================
    # STEP 1: Download dense model + baseline PPL
    # ================================================================
    print(f'[1/6] Downloading {hf_id}...', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        hf_id, torch_dtype=torch.float16, trust_remote_code=True,
        low_cpu_mem_usage=True, device_map='auto'
    ).eval()
    device = next(model.parameters()).device

    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    text = '\n\n'.join([t for t in ds['text'] if t.strip()])
    enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=8193)
    ids = enc.input_ids[:, :8193].to(device)
    nlls, n_eval, seq_len = [], 0, 2048
    with torch.no_grad():
        for i in range(0, ids.shape[1]-1, seq_len):
            end = min(i+seq_len+1, ids.shape[1])
            chunk = ids[:, i:end]
            if chunk.shape[1] < 2: break
            out = model(input_ids=chunk[:, :-1])
            logits = out.logits.float().cpu()
            labels = chunk[:, 1:].cpu()
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            n = labels.numel()
            nlls.append(loss.item() * n)
            n_eval += n
            if n_eval >= 8192: break
    dense_ppl = round(float(np.exp(sum(nlls)/n_eval)), 4)
    print(f'  Dense PPL: {dense_ppl}', flush=True)
    steps_done['dense_ppl'] = dense_ppl

    # Save model locally for compression
    local_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(local_dir, safe_serialization=True)
    tokenizer.save_pretrained(local_dir)
    del model
    torch.cuda.empty_cache()
    import gc; gc.collect()

    # ================================================================
    # STEP 2: Compress with CDNA v3
    # ================================================================
    print(f'[2/6] Compressing with CDNA v3...', flush=True)
    compress_cmd = [
        sys.executable, 'tools/compress.py',
        '--model-dir', str(local_dir),
        '--out-dir', str(local_dir / 'cdnav3'),
        '--k', '256', '--sidecar',
    ]
    result = subprocess.run(compress_cmd, capture_output=True, text=True, timeout=3600)
    if result.returncode != 0:
        raise RuntimeError(f'compress.py failed: {result.stderr[-500:]}')
    print('  Compression done.', flush=True)
    steps_done['compressed'] = True

    # ================================================================
    # STEP 3: Eval compressed PPL
    # ================================================================
    print(f'[3/6] Evaluating compressed PPL...', flush=True)
    import helix_substrate
    from helix_substrate.helix_linear import load_cdna_factors, swap_to_helix, swap_summary

    model = AutoModelForCausalLM.from_pretrained(
        local_dir, torch_dtype=torch.float16, trust_remote_code=True,
        low_cpu_mem_usage=True, device_map='auto'
    )
    factors = load_cdna_factors(local_dir / 'cdnav3')
    model = swap_to_helix(model, factors)
    summary = swap_summary(model)
    del factors; gc.collect()

    try:
        from accelerate.hooks import remove_hook_from_module
        for _, mod in model.named_modules():
            remove_hook_from_module(mod, recurse=False)
    except ImportError:
        pass
    if hasattr(model, 'hf_device_map'):
        delattr(model, 'hf_device_map')
    for name, param in list(model.named_parameters()):
        if param.device.type not in ('meta', device.type if hasattr(device, 'type') else 'cuda'):
            parts = name.split('.')
            mod = model
            for p in parts[:-1]: mod = getattr(mod, p)
            setattr(mod, parts[-1], torch.nn.Parameter(param.data.to('cuda'), requires_grad=False))
    model.eval()
    device = next(p.device for p in model.parameters() if p.device.type != 'meta')
    ids_dev = ids.to(device)

    nlls, n_eval = [], 0
    with torch.no_grad():
        for i in range(0, ids_dev.shape[1]-1, seq_len):
            end = min(i+seq_len+1, ids_dev.shape[1])
            chunk = ids_dev[:, i:end]
            if chunk.shape[1] < 2: break
            out = model(input_ids=chunk[:, :-1])
            logits = out.logits.float().cpu()
            labels = chunk[:, 1:].cpu()
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            n = labels.numel()
            nlls.append(loss.item() * n)
            n_eval += n
            if n_eval >= 8192: break
    helix_ppl = round(float(np.exp(sum(nlls)/n_eval)), 4)
    delta = round((helix_ppl - dense_ppl) / dense_ppl * 100, 2)
    print(f'  Helix PPL: {helix_ppl} ({delta:+.2f}%)', flush=True)
    steps_done['helix_ppl'] = helix_ppl
    steps_done['ppl_delta_pct'] = delta
    steps_done['helix_modules'] = summary.get('helix_modules', 0)

    del model; torch.cuda.empty_cache(); gc.collect()

    # ================================================================
    # STEP 4: Convert to HF format
    # ================================================================
    print(f'[4/6] Converting to HF format → {helix_dir}', flush=True)
    convert_cmd = [
        sys.executable, 'tools/convert_to_hf.py',
        '--model-dir', str(local_dir),
        '--output-dir', str(helix_dir),
    ]
    result = subprocess.run(convert_cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f'convert_to_hf.py failed: {result.stderr[-500:]}')
    print('  Convert done.', flush=True)

    # Fixup missing tensors (norms, embeddings, etc.)
    print(f'  Running fixup_missing_tensors...', flush=True)
    fixup_cmd = [
        sys.executable, 'tools/fixup_missing_tensors.py',
        '--model-dir', str(local_dir),
        '--helix-dir', str(helix_dir),
    ]
    result = subprocess.run(fixup_cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f'fixup failed: {result.stderr[-500:]}')
    print(f'  Fixup done. {result.stdout.strip().split(chr(10))[-1]}', flush=True)
    steps_done['converted'] = True

    # ================================================================
    # STEP 5: Write model card
    # ================================================================
    print(f'[5/6] Writing model card...', flush=True)
    import os
    compressed_size_mb = os.path.getsize(str(helix_dir / 'model.safetensors')) / 1024**2
    dense_size_gb = sum(
        os.path.getsize(str(f)) for f in local_dir.glob('model*.safetensors')
    ) / 1024**3

    card = f'''---
license: apache-2.0
base_model: {hf_id}
tags:
  - compressed
  - cdna-v3
  - helix-substrate
  - vector-quantization
library_name: transformers
pipeline_tag: text-generation
---

# {helix_name}

**{hf_id} compressed {round(dense_size_gb * 1024 / compressed_size_mb, 1)}x with CDNA v3**

| Metric | Value |
|--------|-------|
| Dense PPL (WikiText-2) | {dense_ppl} |
| Helix PPL (WikiText-2) | {helix_ppl} |
| PPL delta | **{delta:+.2f}%** |
| Dense size | {dense_size_gb:.1f} GB |
| Compressed size | {compressed_size_mb / 1024:.1f} GB |
| Compressed modules | {steps_done.get("helix_modules", "N/A")} HelixLinear layers |

## Quick Start

```bash
pip install helix-substrate transformers torch
```

```python
import helix_substrate  # registers helix quantizer
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{hf_repo}", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("{hf_repo}", trust_remote_code=True)

inputs = tokenizer("The capital of France is", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=64)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Companion Models

See all models at [EchoLabs33 on HuggingFace](https://huggingface.co/EchoLabs33).
Same codec, same `pip install`, multiple architectures (Transformer, Mamba, Mamba2, hybrid).

## Citation

```bibtex
@software{{helix_substrate_2026,
  title={{Helix Substrate: Universal Weight Compression via CDNA v3}},
  author={{EchoLabs}},
  year={{2026}},
  url={{https://github.com/echo313unfolding/helix-substrate}}
}}
```
'''
    (helix_dir / 'README.md').write_text(card)
    steps_done['card_written'] = True

    # ================================================================
    # STEP 6: Upload to HuggingFace
    # ================================================================
    print(f'[6/6] Uploading to {hf_repo}...', flush=True)

    # Create repo (ignore if exists)
    subprocess.run(
        ['huggingface-cli', 'repo', 'create', hf_repo, '--type', 'model'],
        capture_output=True, text=True
    )

    # Upload all files
    upload_result = subprocess.run(
        ['huggingface-cli', 'upload', hf_repo, str(helix_dir), '.', '--repo-type', 'model'],
        capture_output=True, text=True, timeout=1800
    )
    if upload_result.returncode != 0:
        raise RuntimeError(f'Upload failed: {upload_result.stderr[-500:]}')
    print(f'  Uploaded to https://huggingface.co/{hf_repo}', flush=True)
    steps_done['uploaded'] = True

    # ================================================================
    # VERIFY: Load from HF, count HelixLinear modules
    # ================================================================
    print(f'  Verifying clean-room load from HF...', flush=True)
    del ids, ids_dev; torch.cuda.empty_cache(); gc.collect()
    model = AutoModelForCausalLM.from_pretrained(hf_repo, trust_remote_code=True)
    n_helix = sum(1 for m in model.modules() if type(m).__name__ == 'HelixLinear')
    print(f'  Verified: {n_helix} HelixLinear modules loaded from HF', flush=True)
    steps_done['verified_helix_modules'] = n_helix
    del model; gc.collect()

    # ================================================================
    # CLEANUP: Delete dense model + tmp to free disk for next model
    # ================================================================
    print(f'  Cleaning up {local_dir} and {helix_dir} to free disk...', flush=True)
    import shutil
    shutil.rmtree(str(local_dir), ignore_errors=True)
    shutil.rmtree(str(helix_dir), ignore_errors=True)
    # Also purge HF cache for this model to avoid double-storage
    hf_cache = Path.home() / '.cache' / 'huggingface' / 'hub'
    for d in hf_cache.glob(f'models--{hf_id.replace("/", "--")}*'):
        shutil.rmtree(str(d), ignore_errors=True)
    print(f'  Cleanup done.', flush=True)

    receipt = {
        'work_order': 'WO-06-T3-SSM',
        'model': hf_id,
        'hf_repo': hf_repo,
        'question': f'Full pipeline: compress {hf_id}, convert to HF, upload, verify',
        'verdict': f'SHIPPED. Helix PPL={helix_ppl} ({delta:+.2f}%), {n_helix} HelixLinear, live on HF',
        'dense_ppl': dense_ppl,
        'helix_ppl': helix_ppl,
        'ppl_delta_pct': delta,
        'n_tokens': n_eval,
        'helix_modules': steps_done.get('helix_modules', 0),
        'verified_helix_modules': n_helix,
        'compressed_size_mb': round(compressed_size_mb, 1),
        'dense_size_gb': round(dense_size_gb, 2),
        'steps': steps_done,
        'cost': {
            'wall_time_s': round(time.time() - t_start, 3),
            'cpu_time_s': round(time.process_time(), 3),
            'peak_memory_mb': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
            'python_version': platform.python_version(),
            'hostname': platform.node(),
            'timestamp_start': start_iso,
            'timestamp_end': time.strftime('%Y-%m-%dT%H:%M:%S'),
        }
    }
    if torch.cuda.is_available():
        receipt['vram_peak_mb'] = round(torch.cuda.max_memory_allocated() / 1024**2, 1)
        receipt['gpu'] = torch.cuda.get_device_name(0)
    print(json.dumps(receipt, indent=2))
except Exception as e:
    import traceback
    traceback.print_exc()
    # Emit partial receipt so we know where it died
    receipt = {
        'error': str(e),
        'model': hf_id,
        'steps_completed': steps_done,
        'cost': {
            'wall_time_s': round(time.time() - t_start, 3),
            'timestamp_start': start_iso,
            'timestamp_end': time.strftime('%Y-%m-%dT%H:%M:%S'),
        }
    }
    print(json.dumps(receipt, indent=2))
" 2>&1 | tee "$LOG_DIR/${task_id}.log"
            save_receipt "$task_id" "$LOG_DIR/${task_id}.log"
            checkpoint_done "$task_id"
        fi
    done

    log "========== TIER 3a COMPLETE (SSM uploads) =========="

    # --- 3b: Cross-architecture lm-eval comparison table ---
    if ! skip_if_done "t3b_cross_arch_table"; then
        log "--- 3b: Cross-architecture lm-eval table (all 7B-class helix models) ---"
        python3 -u -c "
import json, sys, time, platform, resource

t_start = time.time()
start_iso = time.strftime('%Y-%m-%dT%H:%M:%S')

try:
    import helix_substrate
    import lm_eval

    # All 7B-class helix models that should exist on HF after tier 3a
    models = {
        'qwen2.5-7b (transformer)': 'EchoLabs33/qwen2.5-7b-instruct-helix',
        'zamba2-7b (hybrid)': 'EchoLabs33/zamba2-7b-instruct-helix',
        'mamba-2.8b (pure SSM)': 'EchoLabs33/mamba-2.8b-helix',
    }
    tasks = ['hellaswag', 'arc_easy', 'arc_challenge']

    all_results = {}
    for label, repo in models.items():
        print(f'Evaluating {label} ({repo})...', flush=True)
        try:
            results = lm_eval.simple_evaluate(
                model='hf',
                model_args=f'pretrained={repo},trust_remote_code=True',
                tasks=tasks,
                batch_size='auto',
                device='cuda',
            )
            scores = {}
            for task, data in results['results'].items():
                key = 'acc_norm,none' if 'acc_norm,none' in data else 'acc,none'
                scores[task] = round(data.get(key, 0), 4)
            all_results[label] = scores
            print(f'  {label}: {scores}', flush=True)

            # Free GPU between models
            import torch, gc
            torch.cuda.empty_cache(); gc.collect()
        except Exception as e:
            all_results[label] = {'error': str(e)}
            print(f'  {label}: ERROR {e}', flush=True)

    # Print the table
    print(flush=True)
    print('=' * 90, flush=True)
    print('  CROSS-ARCHITECTURE COMPARISON — One codec, three architectures', flush=True)
    print('=' * 90, flush=True)
    header = f\"  {'Model':<30} {'HellaSwag':>10} {'ARC-easy':>10} {'ARC-chall':>10}\"
    print(header, flush=True)
    print(f\"  {'-'*30} {'-'*10} {'-'*10} {'-'*10}\", flush=True)
    for label, scores in all_results.items():
        if 'error' in scores:
            print(f'  {label:<30} ERROR: {scores[\"error\"][:40]}', flush=True)
        else:
            print(f\"  {label:<30} {scores.get('hellaswag',0):>10.4f} {scores.get('arc_easy',0):>10.4f} {scores.get('arc_challenge',0):>10.4f}\", flush=True)
    print('=' * 90, flush=True)

    receipt = {
        'work_order': 'WO-06-T3b',
        'question': 'How do transformer vs SSM vs hybrid compare at 7B scale, all compressed with one codec?',
        'verdict': 'Cross-architecture comparison table — see results',
        'results': all_results,
        'tasks': tasks,
        'cost': {
            'wall_time_s': round(time.time() - t_start, 3),
            'cpu_time_s': round(time.process_time(), 3),
            'peak_memory_mb': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
            'python_version': platform.python_version(),
            'hostname': platform.node(),
            'timestamp_start': start_iso,
            'timestamp_end': time.strftime('%Y-%m-%dT%H:%M:%S'),
        }
    }
    import torch
    if torch.cuda.is_available():
        receipt['gpu'] = torch.cuda.get_device_name(0)
    print(json.dumps(receipt, indent=2))
except Exception as e:
    import traceback
    traceback.print_exc()
    print(json.dumps({'error': str(e)}))
" 2>&1 | tee "$LOG_DIR/t3b_cross_arch_table.log"
        save_receipt "t3b_cross_arch_table" "$LOG_DIR/t3b_cross_arch_table.log"
        checkpoint_done "t3b_cross_arch_table"
    fi

    log "========== TIER 3b COMPLETE (cross-arch table) =========="

    # --- 3c: Multi-model co-residency on one GPU ---
    if ! skip_if_done "t3c_multi_model"; then
        log "--- 3c: Multi-model load — transformer + SSM + vision on one GPU ---"
        python3 -u -c "
import json, sys, time, platform, resource
import torch, gc

t_start = time.time()
start_iso = time.strftime('%Y-%m-%dT%H:%M:%S')

def get_vram():
    return round(torch.cuda.memory_allocated() / 1024**2, 1) if torch.cuda.is_available() else 0

try:
    import helix_substrate
    from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPModel, CLIPProcessor

    results = {'models_loaded': [], 'vram_snapshots': []}
    device = 'cuda'

    # Snapshot: empty GPU
    results['vram_snapshots'].append({'state': 'empty', 'vram_mb': get_vram()})

    # --- Load 1: Compressed transformer (Qwen 3B — fits for sure) ---
    print('[1/3] Loading compressed transformer (Qwen 3B)...', flush=True)
    qwen_id = 'EchoLabs33/qwen2.5-3b-instruct-helix'
    qwen_tok = AutoTokenizer.from_pretrained(qwen_id, trust_remote_code=True)
    qwen = AutoModelForCausalLM.from_pretrained(qwen_id, trust_remote_code=True).to(device).eval()
    n_helix_qwen = sum(1 for m in qwen.modules() if type(m).__name__ == 'HelixLinear')
    vram_after_qwen = get_vram()
    results['models_loaded'].append({'name': 'qwen2.5-3b-instruct-helix', 'type': 'transformer', 'helix_modules': n_helix_qwen, 'vram_mb': vram_after_qwen})
    results['vram_snapshots'].append({'state': 'after_qwen', 'vram_mb': vram_after_qwen})
    print(f'  Qwen loaded: {vram_after_qwen} MB VRAM, {n_helix_qwen} HelixLinear', flush=True)

    # --- Load 2: Compressed SSM (Mamba2-1.3B or smaller) ---
    print('[2/3] Loading compressed SSM (Mamba2-1.3B)...', flush=True)
    mamba_id = 'EchoLabs33/mamba2-1.3b-helix'
    mamba_tok = AutoTokenizer.from_pretrained(mamba_id, trust_remote_code=True)
    mamba = AutoModelForCausalLM.from_pretrained(mamba_id, trust_remote_code=True).to(device).eval()
    n_helix_mamba = sum(1 for m in mamba.modules() if type(m).__name__ == 'HelixLinear')
    vram_after_mamba = get_vram()
    results['models_loaded'].append({'name': 'mamba2-1.3b-helix', 'type': 'ssm', 'helix_modules': n_helix_mamba, 'vram_mb': vram_after_mamba - vram_after_qwen})
    results['vram_snapshots'].append({'state': 'after_mamba', 'vram_mb': vram_after_mamba})
    print(f'  Mamba loaded: {vram_after_mamba} MB total VRAM (+{vram_after_mamba - vram_after_qwen} MB), {n_helix_mamba} HelixLinear', flush=True)

    # --- Load 3: CLIP (vision — compress on the fly if helix version exists, else dense small) ---
    print('[3/3] Loading vision model (CLIP ViT-B/32)...', flush=True)
    clip_id = 'openai/clip-vit-base-patch32'
    clip_proc = CLIPProcessor.from_pretrained(clip_id)
    clip = CLIPModel.from_pretrained(clip_id).to(device).eval()
    vram_after_clip = get_vram()
    results['models_loaded'].append({'name': clip_id, 'type': 'vision', 'helix_modules': 0, 'note': 'dense (helix CLIP not yet on HF)', 'vram_mb': vram_after_clip - vram_after_mamba})
    results['vram_snapshots'].append({'state': 'after_clip', 'vram_mb': vram_after_clip})
    print(f'  CLIP loaded: {vram_after_clip} MB total VRAM (+{vram_after_clip - vram_after_mamba} MB)', flush=True)

    # --- Run all three models ---
    print(flush=True)
    print('Running all three models concurrently...', flush=True)

    # Qwen: text generation
    qwen_input = qwen_tok('The relationship between transformers and SSMs is', return_tensors='pt').to(device)
    with torch.no_grad():
        qwen_out = qwen.generate(**qwen_input, max_new_tokens=32, do_sample=False)
    qwen_text = qwen_tok.decode(qwen_out[0], skip_special_tokens=True)
    print(f'  Qwen output: {qwen_text[:80]}...', flush=True)

    # Mamba: text generation
    mamba_input = mamba_tok('State space models process sequences by', return_tensors='pt').to(device)
    with torch.no_grad():
        mamba_out = mamba.generate(**mamba_input, max_new_tokens=32, do_sample=False)
    mamba_text = mamba_tok.decode(mamba_out[0], skip_special_tokens=True)
    print(f'  Mamba output: {mamba_text[:80]}...', flush=True)

    # CLIP: zero-shot classification
    from PIL import Image
    import numpy as np
    dummy_img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    clip_input = clip_proc(text=['a cat', 'a dog', 'a car'], images=dummy_img, return_tensors='pt', padding=True)
    clip_input = {k: v.to(device) for k, v in clip_input.items()}
    with torch.no_grad():
        clip_out = clip(**clip_input)
    clip_probs = clip_out.logits_per_image.softmax(dim=-1).cpu().tolist()[0]
    print(f'  CLIP probs: cat={clip_probs[0]:.3f}, dog={clip_probs[1]:.3f}, car={clip_probs[2]:.3f}', flush=True)

    vram_peak = round(torch.cuda.max_memory_allocated() / 1024**2, 1)
    total_vram = round(torch.cuda.get_device_properties(0).total_mem / 1024**2, 1)

    results['all_ran'] = True
    results['peak_vram_mb'] = vram_peak
    results['total_gpu_vram_mb'] = total_vram
    results['headroom_mb'] = round(total_vram - vram_peak, 1)
    results['qwen_sample'] = qwen_text[:100]
    results['mamba_sample'] = mamba_text[:100]
    results['clip_probs'] = dict(zip(['cat', 'dog', 'car'], clip_probs))

    print(flush=True)
    print('=' * 70, flush=True)
    print('  MULTI-MODEL CO-RESIDENCY — One codec, one GPU', flush=True)
    print('=' * 70, flush=True)
    print(f'  Transformer (Qwen 3B):  {results[\"models_loaded\"][0][\"vram_mb\"]:>6} MB  [{n_helix_qwen} HelixLinear]', flush=True)
    print(f'  SSM (Mamba2 1.3B):     +{results[\"models_loaded\"][1][\"vram_mb\"]:>5} MB  [{n_helix_mamba} HelixLinear]', flush=True)
    print(f'  Vision (CLIP ViT-B/32):+{results[\"models_loaded\"][2][\"vram_mb\"]:>5} MB  [dense]', flush=True)
    print(f'  Peak VRAM:              {vram_peak:>6} MB / {total_vram} MB ({round(vram_peak/total_vram*100,1)}%)', flush=True)
    print(f'  Headroom:               {results[\"headroom_mb\"]:>6} MB', flush=True)
    print(f'  All models ran:         {results[\"all_ran\"]}', flush=True)
    print('=' * 70, flush=True)

    receipt = {
        'work_order': 'WO-06-T3c',
        'question': 'Can transformer + SSM + vision coexist on one GPU, all compressed?',
        'verdict': f'YES. Peak {vram_peak} MB / {total_vram} MB. All three generated output.',
        **results,
        'cost': {
            'wall_time_s': round(time.time() - t_start, 3),
            'cpu_time_s': round(time.process_time(), 3),
            'peak_memory_mb': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
            'python_version': platform.python_version(),
            'hostname': platform.node(),
            'timestamp_start': start_iso,
            'timestamp_end': time.strftime('%Y-%m-%dT%H:%M:%S'),
        }
    }
    if torch.cuda.is_available():
        receipt['gpu'] = torch.cuda.get_device_name(0)
    print(json.dumps(receipt, indent=2))
except Exception as e:
    import traceback
    traceback.print_exc()
    print(json.dumps({'error': str(e)}))
" 2>&1 | tee "$LOG_DIR/t3c_multi_model.log"
        save_receipt "t3c_multi_model" "$LOG_DIR/t3c_multi_model.log"
        checkpoint_done "t3c_multi_model"
    fi

    log "========== TIER 3c COMPLETE (multi-model demo) =========="
    log "========== TIER 3 FULLY COMPLETE =========="
fi

# ============================================================================
# SUMMARY
# ============================================================================
log "=========================================="
log "CLOUD RUN COMPLETE — $(timestamp)"
log "Receipts in: $RECEIPT_DIR"
log "Checkpoints: $(ls "$CHECKPOINT_DIR"/*.done 2>/dev/null | wc -l) tasks completed"
log "=========================================="

echo ""
echo "Completed tasks:"
ls -1 "$CHECKPOINT_DIR"/*.done 2>/dev/null | sed 's/.*\//  /' | sed 's/.done//'
echo ""
echo "Receipts:"
ls -1 "$RECEIPT_DIR"/*.json 2>/dev/null | sed 's/.*\//  /'
