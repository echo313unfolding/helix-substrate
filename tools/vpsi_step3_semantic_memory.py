"""
Step 3: Build the Semantic Memory Table
========================================
SQLite table indexing all compressed modules with functional metadata:
- Architectural address (organism, phylum, chromosome, gene, position)
- Kurtosis tier (epigenetic mark)
- Compression tier
- Activation magnitude (from Step 1)
- Sensitivity (from Step 2)
- Codebook shape features
- File paths

Populates from existing CDNA v3 directories + Step 1/2 receipts.

WO-VPSI-01 Step 3

Usage:
    python3 tools/vpsi_step3_semantic_memory.py
"""
import sqlite3
import json
import numpy as np
import time
import platform
import resource
from pathlib import Path
from scipy.stats import kurtosis, skew

DB_PATH = "/home/voidstr3m33/helix-substrate/vpsi_semantic_memory.db"

MODELS = [
    {
        "model_id": "tinyllama-1.1b",
        "organism": "TinyLlama",
        "arch_class": "transformer",
        "cdna_dir": "/home/voidstr3m33/models/tinyllama_fp32/cdnav3/",
    },
    {
        "model_id": "qwen-1.5b-coder",
        "organism": "Qwen-1.5B",
        "arch_class": "transformer",
        "cdna_dir": "/home/voidstr3m33/models/qwen2.5-coder-1.5b-instruct/cdnav3/",
    },
    {
        "model_id": "qwen-3b-coder",
        "organism": "Qwen-3B",
        "arch_class": "transformer",
        "cdna_dir": "/home/voidstr3m33/models/qwen2.5-coder-3b-instruct/cdnav3/",
    },
    {
        "model_id": "mamba-130m",
        "organism": "Mamba-130m",
        "arch_class": "SSM",
        "cdna_dir": "/home/voidstr3m33/models/mamba-130m-hf/cdnav3/",
    },
    {
        "model_id": "mamba2-1.3b",
        "organism": "Mamba2-1.3B",
        "arch_class": "SSM",
        "cdna_dir": "/home/voidstr3m33/models/mamba2-1.3b/cdnav3/",
    },
    {
        "model_id": "qwen-3b-instruct",
        "organism": "Qwen-3B-Instruct",
        "arch_class": "transformer",
        "cdna_dir": "/home/voidstr3m33/models/qwen2.5-3b-instruct/cdnav3/",
    },
    {
        "model_id": "qwen-7b-instruct",
        "organism": "Qwen-7B-Instruct",
        "arch_class": "transformer",
        "cdna_dir": "/home/voidstr3m33/models/qwen2.5-7b-instruct/cdnav3/",
    },
    {
        "model_id": "qwen-14b-instruct",
        "organism": "Qwen-14B-Instruct",
        "arch_class": "transformer",
        "cdna_dir": "/home/voidstr3m33/models/qwen2.5-14b-instruct/cdnav3/",
    },
]

GENES = ["q_proj", "k_proj", "v_proj", "o_proj",
         "gate_proj", "up_proj", "down_proj",
         "in_proj", "out_proj", "x_proj", "dt_proj", "conv1d"]


def classify_tensor(tname):
    """Extract chromosome (module_type), gene (projection), layer_index from tensor name."""
    chromosome = "unknown"
    gene = "unknown"
    layer_index = -1

    if "self_attn" in tname or "attention" in tname:
        chromosome = "attention"
    elif "mlp" in tname or "feed_forward" in tname:
        chromosome = "mlp"
    elif "mixer" in tname or "ssm" in tname:
        chromosome = "state_space"
    elif "embed" in tname:
        chromosome = "embedding"
    elif "norm" in tname:
        chromosome = "norm"
    elif "conv" in tname:
        chromosome = "conv"
    elif "lm_head" in tname or "output" in tname:
        chromosome = "head"

    for g in GENES:
        if g in tname:
            gene = g
            break

    for part in tname.split("."):
        try:
            layer_index = int(part)
            break
        except ValueError:
            pass

    return chromosome, gene, layer_index


def load_step1_receipt():
    """Load latest Step 1 activation magnitude receipt."""
    receipt_dir = Path("/home/voidstr3m33/helix-substrate/receipts/vpsi/")
    files = sorted(receipt_dir.glob("step1_*.json"))
    if not files:
        return {}
    with open(files[-1]) as f:
        data = json.load(f)
    return data.get("module_stats", {})


def load_step2_receipt():
    """Load latest Step 2 sensitivity receipt."""
    receipt_dir = Path("/home/voidstr3m33/helix-substrate/receipts/vpsi/")
    files = sorted(receipt_dir.glob("step2_*.json"))
    if not files:
        return {}
    with open(files[-1]) as f:
        data = json.load(f)
    return data.get("sensitivity", {})


def main():
    t_start = time.time()
    cpu_start = time.process_time()
    start_iso = time.strftime('%Y-%m-%dT%H:%M:%S')

    # Load Step 1 and 2 data (may be empty if not yet run)
    act_mag = load_step1_receipt()
    sensitivity = load_step2_receipt()
    print("Step 1 data: %d modules" % len(act_mag))
    print("Step 2 data: %d modules" % len(sensitivity))

    # Create database
    db = sqlite3.connect(DB_PATH)
    cur = db.cursor()

    cur.execute("DROP TABLE IF EXISTS compressed_modules")
    cur.execute("""
        CREATE TABLE compressed_modules (
            module_id        TEXT PRIMARY KEY,
            model_id         TEXT NOT NULL,
            organism         TEXT NOT NULL,
            arch_class       TEXT NOT NULL,
            module_type      TEXT,
            projection       TEXT,
            layer_index      INTEGER,
            kurtosis         REAL,
            kurtosis_tier    TEXT,
            compression_tier TEXT,
            codebook_range   REAL,
            codebook_iqr     REAL,
            codebook_skew    REAL,
            codebook_mad     REAL,
            activation_mag   REAL,
            sensitivity_kl   REAL,
            cosine_fidelity  REAL,
            shape_rows       INTEGER,
            shape_cols       INTEGER,
            file_path        TEXT
        )
    """)

    # Populate from CDNA directories
    total_inserted = 0
    for model_info in MODELS:
        cdna_dir = Path(model_info["cdna_dir"])
        if not cdna_dir.exists():
            print("SKIP: %s (not found)" % cdna_dir)
            continue

        count = 0
        for p in sorted(cdna_dir.glob("*.cdnav3")):
            meta_path = p / "meta.json"
            cb_path = p / "codebook.npy"
            stats_path = p / "stats.json"
            if not meta_path.exists() or not cb_path.exists():
                continue

            meta = json.loads(meta_path.read_text())
            tname = meta["tensor_name"]
            shape = tuple(meta["shape"])
            cb = np.load(cb_path).astype(np.float64)

            # Classify
            chromosome, gene, layer_idx = classify_tensor(tname)

            # Codebook features
            cb_range = float(cb.max() - cb.min())
            cb_iqr = float(np.percentile(cb, 75) - np.percentile(cb, 25))
            cb_skew_val = float(skew(cb))
            cb_mad = float(np.mean(np.abs(cb - cb.mean())))
            cb_kurt = float(kurtosis(cb))

            # Kurtosis tier
            if cb_kurt > 20:
                kurt_tier = "high"
            elif cb_kurt > 5:
                kurt_tier = "medium"
            else:
                kurt_tier = "low"

            # Compression tier (check for SVD files)
            has_svd = (p / "svd_U.npy").exists()
            comp_tier = "vq_svd" if has_svd else "vq_only"

            # Cosine fidelity from stats.json
            cos_fid = None
            if stats_path.exists():
                try:
                    stats = json.loads(stats_path.read_text())
                    cos_fid = stats.get("cosine", stats.get("cosine_similarity"))
                except (json.JSONDecodeError, KeyError):
                    pass

            # Module path (strip .weight for matching with step1/2 data)
            module_path = tname
            if module_path.endswith(".weight"):
                module_path = module_path[:-7]

            # Activation magnitude (from Step 1, TinyLlama only)
            act_val = None
            if model_info["model_id"] == "tinyllama-1.1b" and module_path in act_mag:
                act_val = act_mag[module_path].get("mean_l2")

            # Sensitivity (from Step 2, TinyLlama only)
            sens_val = None
            if model_info["model_id"] == "tinyllama-1.1b" and module_path in sensitivity:
                sens_val = sensitivity[module_path].get("mean_kl")

            # Unique module_id
            module_id = "%s.%s" % (model_info["model_id"], module_path)

            cur.execute("""
                INSERT OR REPLACE INTO compressed_modules VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
            """, (
                module_id,
                model_info["model_id"],
                model_info["organism"],
                model_info["arch_class"],
                chromosome,
                gene,
                layer_idx,
                cb_kurt,
                kurt_tier,
                comp_tier,
                cb_range,
                cb_iqr,
                cb_skew_val,
                cb_mad,
                act_val,
                sens_val,
                cos_fid,
                shape[0] if len(shape) >= 2 else shape[0],
                shape[1] if len(shape) >= 2 else 1,
                str(p),
            ))
            count += 1

        print("%s: inserted %d modules" % (model_info["model_id"], count))
        total_inserted += count

    db.commit()

    # Summary queries
    print("\n" + "=" * 60)
    print("SEMANTIC MEMORY TABLE SUMMARY")
    print("=" * 60)

    cur.execute("SELECT COUNT(*) FROM compressed_modules")
    print("Total modules: %d" % cur.fetchone()[0])

    cur.execute("SELECT model_id, COUNT(*) FROM compressed_modules GROUP BY model_id")
    for row in cur.fetchall():
        print("  %s: %d" % row)

    cur.execute("SELECT arch_class, COUNT(*) FROM compressed_modules GROUP BY arch_class")
    print("\nBy architecture:")
    for row in cur.fetchall():
        print("  %s: %d" % row)

    cur.execute("SELECT module_type, COUNT(*) FROM compressed_modules GROUP BY module_type")
    print("\nBy module type:")
    for row in cur.fetchall():
        print("  %s: %d" % row)

    cur.execute("SELECT kurtosis_tier, COUNT(*) FROM compressed_modules GROUP BY kurtosis_tier")
    print("\nBy kurtosis tier:")
    for row in cur.fetchall():
        print("  %s: %d" % row)

    cur.execute("SELECT compression_tier, COUNT(*) FROM compressed_modules GROUP BY compression_tier")
    print("\nBy compression tier:")
    for row in cur.fetchall():
        print("  %s: %d" % row)

    # Example queries
    print("\n" + "=" * 60)
    print("EXAMPLE QUERIES")
    print("=" * 60)

    print("\n-- Attention modules from layers 8-15 with high sensitivity --")
    cur.execute("""
        SELECT module_id, layer_index, sensitivity_kl, kurtosis_tier
        FROM compressed_modules
        WHERE module_type = 'attention'
          AND layer_index BETWEEN 8 AND 15
          AND sensitivity_kl IS NOT NULL
        ORDER BY sensitivity_kl DESC
        LIMIT 10
    """)
    for row in cur.fetchall():
        print("  %s layer=%d kl=%.4f kurt=%s" % row)

    print("\n-- Most sensitive modules across all models --")
    cur.execute("""
        SELECT module_id, model_id, sensitivity_kl, activation_mag
        FROM compressed_modules
        WHERE sensitivity_kl IS NOT NULL
        ORDER BY sensitivity_kl DESC
        LIMIT 10
    """)
    for row in cur.fetchall():
        print("  %s (%s) kl=%.4f act=%.2f" % row)

    print("\n-- Candidates for lazy-loading (low sensitivity) --")
    cur.execute("""
        SELECT module_id, sensitivity_kl, activation_mag, kurtosis_tier
        FROM compressed_modules
        WHERE sensitivity_kl IS NOT NULL
        ORDER BY sensitivity_kl ASC
        LIMIT 10
    """)
    for row in cur.fetchall():
        print("  %s kl=%.4f act=%.2f kurt=%s" % row)

    print("\n-- Cross-architecture comparison: q_proj kurtosis by model --")
    cur.execute("""
        SELECT model_id, AVG(kurtosis), MIN(kurtosis), MAX(kurtosis)
        FROM compressed_modules
        WHERE projection = 'q_proj'
        GROUP BY model_id
    """)
    for row in cur.fetchall():
        print("  %s avg=%.2f range=[%.2f, %.2f]" % row)

    db.close()

    cost = {
        'wall_time_s': round(time.time() - t_start, 3),
        'cpu_time_s': round(time.process_time() - cpu_start, 3),
        'peak_memory_mb': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        'python_version': platform.python_version(),
        'hostname': platform.node(),
        'timestamp_start': start_iso,
        'timestamp_end': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }

    print("\nDatabase: %s" % DB_PATH)
    print("Cost: %s" % json.dumps(cost, indent=2))


if __name__ == "__main__":
    main()
