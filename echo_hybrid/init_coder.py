"""
WO-ECHO-HYBRID-CODER-01: Initialize EchoHybridCoder from Qwen2.5-Coder-1.5B donor.

Steps:
    1. Build EchoHybridCoderModel (778M params)
    2. Load Qwen2.5-Coder-1.5B dense weights
    3. Transplant attention block weights at positions 5, 11, 17, 23
    4. Transplant embedding + lm_head
    5. Verify forward pass
    6. Compress with HXQ (grouped VQ d=2)
    7. Run 10-step born-compressed training with MorphSAT gate
    8. Emit receipt

Usage:
    python3 -m echo_hybrid.init_coder [--steps 10] [--gate]
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import torch
import torch.nn as nn

from echo_hybrid.coder_config import EchoHybridCoderConfig, EchoHybridCoderModel

RECEIPT_DIR = Path("receipts/echo_hybrid")
RECEIPT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Weight surgery from Qwen2.5-Coder-1.5B
# ---------------------------------------------------------------------------

def init_from_qwen(
    model: EchoHybridCoderModel,
    qwen_name: str = "Qwen/Qwen2.5-Coder-1.5B",
) -> dict:
    """Transplant attention + embedding weights from Qwen2.5-Coder-1.5B.

    Attention blocks at hybrid positions 5, 11, 17, 23 receive weights
    from Qwen layers 5, 11, 17, 23 (matching depth in the residual stream).

    Mamba blocks keep random initialization.

    Returns dict with surgery stats.
    """
    from transformers import AutoModelForCausalLM

    print(f"Loading donor: {qwen_name}...")
    donor = AutoModelForCausalLM.from_pretrained(
        qwen_name, torch_dtype=torch.float32, device_map="cpu"
    )
    donor_sd = donor.state_dict()
    print(f"  Donor loaded: {sum(p.numel() for p in donor.parameters()):,} params")

    stats = {"embed_loaded": 0, "attn_loaded": 0, "attn_skipped": [],
             "attn_errors": [], "mamba_random": 0}

    # --- Embedding ---
    embed_key = "model.embed_tokens.weight"
    if embed_key in donor_sd:
        src = donor_sd[embed_key]
        if src.shape == model.embed.weight.shape:
            model.embed.weight.data.copy_(src)
            stats["embed_loaded"] += 1
            print(f"  Embedding: {src.shape} ✓")
        else:
            stats["attn_errors"].append(
                f"embed shape mismatch: {src.shape} vs {model.embed.weight.shape}")

    # --- Attention blocks ---
    # Map hybrid attention positions to Qwen layer indices
    attn_positions = [i for i, b in enumerate(model.cfg.block_pattern) if b == "attn"]

    # Qwen2.5 layer key structure:
    #   model.layers.{i}.self_attn.{q,k,v,o}_proj.{weight,bias}
    #   model.layers.{i}.mlp.gate_proj.weight
    #   model.layers.{i}.mlp.up_proj.weight
    #   model.layers.{i}.mlp.down_proj.weight
    #   model.layers.{i}.input_layernorm.weight
    #   model.layers.{i}.post_attention_layernorm.weight

    attn_key_map = {
        # self_attn
        "q_proj.weight": "self_attn.q_proj.weight",
        "q_proj.bias": "self_attn.q_proj.bias",
        "k_proj.weight": "self_attn.k_proj.weight",
        "k_proj.bias": "self_attn.k_proj.bias",
        "v_proj.weight": "self_attn.v_proj.weight",
        "v_proj.bias": "self_attn.v_proj.bias",
        "o_proj.weight": "self_attn.o_proj.weight",
        # mlp
        "gate_proj.weight": "mlp.gate_proj.weight",
        "up_proj.weight": "mlp.up_proj.weight",
        "down_proj.weight": "mlp.down_proj.weight",
        # norms
        "input_layernorm.weight": "input_layernorm.weight",
        "post_attention_layernorm.weight": "post_attention_layernorm.weight",
    }

    for hybrid_pos in attn_positions:
        # Use the same layer index in Qwen (depth-matched)
        qwen_layer = hybrid_pos
        block = model.blocks[hybrid_pos].block  # CoderAttentionBlock

        loaded = []
        for our_suffix, qwen_suffix in attn_key_map.items():
            qwen_key = f"model.layers.{qwen_layer}.{qwen_suffix}"
            if qwen_key not in donor_sd:
                stats["attn_skipped"].append(f"pos={hybrid_pos}: missing {qwen_key}")
                continue

            src = donor_sd[qwen_key]

            # Navigate to our parameter
            parts = our_suffix.split(".")
            obj = block
            for p in parts[:-1]:
                obj = getattr(obj, p)
            param_name = parts[-1]

            if not hasattr(obj, param_name):
                stats["attn_skipped"].append(f"pos={hybrid_pos}: no {our_suffix}")
                continue

            dst = getattr(obj, param_name)
            if dst is None:
                stats["attn_skipped"].append(f"pos={hybrid_pos}: {our_suffix} is None")
                continue

            if src.shape == dst.shape:
                dst.data.copy_(src)
                loaded.append(our_suffix)
            else:
                stats["attn_errors"].append(
                    f"pos={hybrid_pos} {our_suffix}: {src.shape} vs {dst.shape}")

        stats["attn_loaded"] += len(loaded)
        print(f"  ATTN block {hybrid_pos} ← Qwen layer {qwen_layer}: "
              f"{len(loaded)}/{len(attn_key_map)} weights")

    # Count Mamba blocks (random init)
    stats["mamba_random"] = model.cfg.n_ssm

    # Free donor
    del donor, donor_sd
    import gc; gc.collect()

    print(f"\n  Surgery complete:")
    print(f"    Embedding: {'✓' if stats['embed_loaded'] else '✗'}")
    print(f"    Attention weights: {stats['attn_loaded']}")
    print(f"    Mamba blocks (random): {stats['mamba_random']}")
    if stats["attn_skipped"]:
        print(f"    Skipped: {len(stats['attn_skipped'])}")
    if stats["attn_errors"]:
        print(f"    Errors: {stats['attn_errors']}")

    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="WO-ECHO-HYBRID-CODER-01")
    parser.add_argument("--steps", type=int, default=10,
                        help="Sanity check training steps")
    parser.add_argument("--gate", action="store_true",
                        help="Enable MorphSAT training gate")
    parser.add_argument("--skip-donor", action="store_true",
                        help="Skip Qwen donor loading (random init only)")
    parser.add_argument("--skip-compress", action="store_true",
                        help="Skip HXQ compression test")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training sanity check")
    args = parser.parse_args()

    t_start = time.time()
    cpu_start = time.process_time()
    start_iso = time.strftime("%Y-%m-%dT%H:%M:%S")

    # Step 1: Build model
    print("=" * 60)
    print("STEP 1: Build EchoHybridCoderModel")
    print("=" * 60)
    cfg = EchoHybridCoderConfig()
    model = EchoHybridCoderModel(cfg)
    print(model)

    # Step 2: Weight surgery
    surgery_stats = None
    if not args.skip_donor:
        print("\n" + "=" * 60)
        print("STEP 2: Weight surgery from Qwen2.5-Coder-1.5B")
        print("=" * 60)
        surgery_stats = init_from_qwen(model)
    else:
        print("\n[skip-donor] Using random initialization only")

    # Step 3: Forward pass verification
    print("\n" + "=" * 60)
    print("STEP 3: Forward pass verification")
    print("=" * 60)
    dummy = torch.randint(0, cfg.vocab_size, (1, 64))
    with torch.no_grad():
        out = model(input_ids=dummy, labels=dummy)
    logit_shape = out["logits"].shape
    loss_val = out["loss"].item()
    print(f"  Input:  {dummy.shape}")
    print(f"  Logits: {logit_shape}")
    print(f"  Loss:   {loss_val:.4f}")
    fwd_pass = logit_shape == (1, 64, cfg.vocab_size)
    print(f"  Shape check: {'PASS' if fwd_pass else 'FAIL'}")

    # Step 4: HXQ compression
    compress_stats = None
    if not args.skip_compress:
        print("\n" + "=" * 60)
        print("STEP 4: Compress with VQ d=2")
        print("=" * 60)
        from echo_hybrid.train_phase1 import compress_all_linears
        t_compress = time.time()
        compressed = compress_all_linears(model, n_clusters=256, vector_dim=2)
        compress_time = time.time() - t_compress
        n_layers = len(compressed)
        utils = [f["codebook_utilization"] for f in compressed.values()]
        min_util = min(utils)
        avg_util = sum(utils) / len(utils)
        compress_stats = {
            "n_layers": n_layers,
            "vector_dim": 2,
            "n_clusters": 256,
            "min_utilization": round(min_util, 4),
            "avg_utilization": round(avg_util, 4),
            "compress_time_s": round(compress_time, 1),
        }
        print(f"  Compressed {n_layers} layers in {compress_time:.1f}s")
        print(f"  Codebook utilization: min={min_util:.3f} avg={avg_util:.3f}")
    else:
        print("\n[skip-compress] Skipping HXQ compression test")

    # Step 5: Training sanity check
    train_stats = None
    if not args.skip_train and not args.skip_compress:
        print("\n" + "=" * 60)
        print(f"STEP 5: {args.steps}-step training sanity check")
        print("=" * 60)
        from echo_hybrid.train_phase1 import Phase1Trainer, load_wikitext_chunks

        # Load minimal training data
        # Use Qwen tokenizer for data loading
        chunks = load_wikitext_chunks(seq_len=64, max_chunks=args.steps * 4)

        trainer = Phase1Trainer(
            model=model,
            lr=1e-4,
            compress_schedule=0,  # no recompression in sanity check
            n_clusters=256,
            vector_dim=2,
            device="cpu",
            vstep_mode="reassign",
            gate_enabled=args.gate,
        )

        losses = []
        for step in range(args.steps):
            batch = chunks[step % len(chunks)].unsqueeze(0)
            loss = trainer.train_step(batch, recent_losses=losses)
            losses.append(loss)
            print(f"  step {step+1:3d}/{args.steps}  loss={loss:.4f}")
            if trainer.halted:
                print(f"  [gate] HALTED: {trainer.halt_reason}")
                break

        train_stats = {
            "steps_completed": len(losses),
            "first_loss": round(losses[0], 4),
            "final_loss": round(losses[-1], 4),
            "halted": trainer.halted,
            "halt_reason": trainer.halt_reason,
            "gate_receipt": trainer.gate.to_receipt() if trainer.gate else None,
        }
        print(f"  Training: {losses[0]:.4f} → {losses[-1]:.4f}")
    elif args.skip_train:
        print("\n[skip-train] Skipping training sanity check")

    # Receipt
    wall_time = time.time() - t_start
    receipt = {
        "wo": "WO-ECHO-HYBRID-CODER-01",
        "timestamp": time.strftime("%Y-%m-%d"),
        "status": "PASS" if fwd_pass else "FAIL",
        "model": f"EchoHybridCoder-{cfg.n_blocks}block-{cfg.hidden_size}d",
        "block_pattern": cfg.block_pattern,
        "n_params_total": model.n_params(),
        "n_params_non_embed": model.n_params(exclude_embeddings=True),
        "attn_positions": [i for i, b in enumerate(cfg.block_pattern) if b == "attn"],
        "forward_pass": {
            "input_shape": list(dummy.shape),
            "output_shape": list(logit_shape),
            "loss": round(loss_val, 4),
            "passed": fwd_pass,
        },
        "surgery": surgery_stats,
        "compression": compress_stats,
        "training": train_stats,
        "cost": {
            "wall_time_s": round(wall_time, 3),
            "cpu_time_s": round(time.process_time() - cpu_start, 3),
            "peak_memory_mb": round(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
            "python_version": platform.python_version(),
            "hostname": platform.node(),
            "timestamp_start": start_iso,
            "timestamp_end": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    }

    out_path = RECEIPT_DIR / "wo_echo_hybrid_coder_01.json"
    with open(out_path, "w") as f:
        json.dump(receipt, f, indent=2)
    print(f"\nRECEIPT: {out_path}")
    print(f"STATUS: {receipt['status']}")

    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
