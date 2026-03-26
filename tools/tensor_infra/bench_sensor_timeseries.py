#!/usr/bin/env python3
"""Domain 10: Sensor Data / Time-Series Compression.
Compress scRNA-seq gene expression matrix and protein structure coordinates."""

import numpy as np
from pathlib import Path
from _common import *

def main():
    t_start, cpu_start, start_iso = start_cost()
    print("=" * 72)
    print("DOMAIN 10: Sensor Data / Time-Series Compression")
    print("=" * 72)

    # ── Part A: scRNA-seq ──
    print("\n  Part A: scRNA-seq (PBMC3K)")
    import anndata
    adata = anndata.read_h5ad(str(HELIX_ROOT / "data" / "pbmc3k_processed.h5ad"))

    # Extract expression matrix (may be sparse)
    if hasattr(adata.X, 'toarray'):
        expr_matrix = adata.X.toarray().astype(np.float32)
    else:
        expr_matrix = np.array(adata.X, dtype=np.float32)

    print(f"    Shape: {expr_matrix.shape}")
    print(f"    Size: {expr_matrix.nbytes / 1e6:.1f} MB")
    print(f"    Sparsity: {np.mean(expr_matrix == 0):.1%}")
    print(f"    Kurtosis: {kurtosis(expr_matrix):.2f}")

    out_dir = HELIX_ROOT / "tensor_infra_scratch" / "sensor"
    out_dir.mkdir(parents=True, exist_ok=True)

    scrna_results = {}
    for k_val in [256, 64]:
        policy = policy_vq(k=k_val, sidecar=True, max_corr=1024)
        stats, recon = compress_tensor(expr_matrix, f"pbmc3k_k{k_val}", out_dir, policy)

        cos = cosine_sim(expr_matrix, recon)

        # PCA comparison: explained variance
        from sklearn.decomposition import PCA
        pca_orig = PCA(n_components=10).fit(expr_matrix)
        pca_comp = PCA(n_components=10).fit(recon)
        var_diff = np.abs(pca_orig.explained_variance_ratio_ - pca_comp.explained_variance_ratio_)
        max_var_diff = float(np.max(var_diff))

        # Clustering comparison: k-means + ARI
        from sklearn.cluster import KMeans
        from sklearn.metrics import adjusted_rand_score
        km_orig = KMeans(n_clusters=5, random_state=42, n_init=3).fit(expr_matrix)
        km_comp = KMeans(n_clusters=5, random_state=42, n_init=3).fit(recon)
        ari = adjusted_rand_score(km_orig.labels_, km_comp.labels_)

        scrna_results[k_val] = {
            "cosine": round(cos, 6),
            "pca_max_var_diff": round(max_var_diff, 6),
            "ari": round(ari, 4),
            "compression_ratio": stats.get("compression_ratio", 1.0),
        }
        print(f"    k={k_val}: cos={cos:.6f}, PCA var_diff={max_var_diff:.6f}, "
              f"ARI={ari:.4f}, ratio={stats.get('compression_ratio', 1.0):.2f}x")

    # ── Part B: Protein coordinates ──
    print("\n  Part B: Protein PDB Coordinates")
    pdb_dir = HELIX_ROOT / "data" / "conformers"
    pdb_files = sorted(pdb_dir.glob("*.pdb"))[:10]  # first 10

    # Parse PDB files to extract CA atom coordinates
    protein_results = []
    all_coords = []  # Stack all proteins into one matrix for compression

    for pdb_path in pdb_files:
        coords = []
        with open(pdb_path) as f:
            for line in f:
                if line.startswith("ATOM") and line[12:16].strip() == "CA":
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append([x, y, z])
        if coords:
            all_coords.append(np.array(coords, dtype=np.float32))
            print(f"    {pdb_path.name}: {len(coords)} CA atoms")

    # Stack all coordinates
    if all_coords:
        max_len = max(c.shape[0] for c in all_coords)
        stacked = np.zeros((len(all_coords) * max_len, 3), dtype=np.float32)
        for i, c in enumerate(all_coords):
            stacked[i * max_len : i * max_len + c.shape[0]] = c

        # Reshape to make it wider: group 10 consecutive residues → (N/10, 30)
        n_rows = stacked.shape[0]
        group_size = 10
        usable = (n_rows // group_size) * group_size
        wide = stacked[:usable].reshape(-1, 3 * group_size)  # (N/10, 30)

        print(f"    Stacked: {stacked.shape} -> reshaped to {wide.shape} for VQ")

        policy = policy_vq(k=64, sidecar=True)  # k=64 for small matrices
        stats, recon_wide = compress_tensor(wide, "protein_coords", out_dir, policy)

        # Reshape back
        recon_3d = recon_wide.reshape(-1, 3)
        orig_3d = stacked[:usable]

        # RMSD
        rmsd = float(np.sqrt(np.mean((orig_3d - recon_3d) ** 2)))
        cos = cosine_sim(orig_3d, recon_3d)

        protein_result = {
            "n_proteins": len(all_coords),
            "total_atoms": stacked.shape[0],
            "cosine": round(cos, 6),
            "rmsd_angstrom": round(rmsd, 4),
            "compression_ratio": stats.get("compression_ratio", 1.0),
            "reshape_note": f"Grouped {group_size} residues -> (N/{group_size}, {3*group_size}) for VQ",
        }
        print(f"    Protein: cos={cos:.6f}, RMSD={rmsd:.4f} A, "
              f"ratio={stats.get('compression_ratio', 1.0):.2f}x")
    else:
        protein_result = {"error": "No PDB files parsed"}

    cost = finish_cost(t_start, cpu_start, start_iso)
    write_receipt("tensor_infra_domain_10", "sensor_timeseries", {
        "scrna_seq": {
            "dataset": "PBMC3K",
            "shape": list(expr_matrix.shape),
            "sparsity": round(float(np.mean(expr_matrix == 0)), 4),
            "per_k": {str(k): v for k, v in scrna_results.items()},
        },
        "protein": protein_result,
        "data_source": "REAL — PBMC3K scRNA-seq + PDB protein coordinates",
    }, cost)

if __name__ == "__main__":
    main()
