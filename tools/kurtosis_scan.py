#!/usr/bin/env python3
"""
Kurtosis preflight scanner for neural network weight compression.

Scans all 2D weight tensors in a model, computes Fisher kurtosis, and
predicts which tensors need SVD correction for VQ compression.

This is a zero-cost diagnostic — it reads weights and computes statistics
without actually compressing anything. Use it before compression to:

  1. Identify high-kurtosis tensors that will need SVD correction
  2. Predict expected compression difficulty per tensor
  3. Compare kurtosis profiles across architectures
  4. Estimate total SVD budget (how many tensors get expensive treatment)

Evidence (cross-architecture):
  TinyLlama  (transformer):     rho=0.7835, p=3.2e-33, n=154
  Mamba-130m (SSM):             rho=0.8534, p=1.2e-28, n=97
  Qwen2.5-7B (transformer+GQA): rho=0.5334, p=8.3e-16, n=196

Usage:
  python3 tools/kurtosis_scan.py --model ~/models/mamba-130m-hf
  python3 tools/kurtosis_scan.py --model ~/models/qwen2.5-7b-instruct --top 20
  python3 tools/kurtosis_scan.py --model ~/models/tinyllama_fp32 --json
"""

"""Thin wrapper — imports from helix_substrate.kurtosis_scan (the installable module).

Kept for backward compatibility with existing scripts that reference tools/kurtosis_scan.py.
"""
from helix_substrate.kurtosis_scan import main

if __name__ == "__main__":
    main()
