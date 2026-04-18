"""Comparator — detect model swaps by comparing fingerprint vs API probe.

Two detection layers:
  Layer 1 (timing): tokens/sec profile deviation (catches size-class swaps)
  Layer 2 (distribution): per-token logprob divergence (catches family swaps)

Optional Layer 3 (sidecar-weighted): diagnostic_weight from HXQ fingerprint
  focuses divergence measurement on tokens where the local model is CONFIDENT
  (low sidecar norm). Confident predictions are most diagnostic because:
  - Same model should agree on confident tokens
  - Different models disagree most on tokens they're each confident about

Usage:
    python3 -m api_polygraph.compare \
        --fingerprint fingerprint_llama3b.json \
        --probe api_probe_results.json
"""

import json
import math
import platform
import resource
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def compare(
    fingerprint_path: str,
    probe_path: str,
    output_path: Optional[str] = None,
    timing_sigma_threshold: float = 3.0,
    kl_threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compare a fingerprint against API probe results.

    Args:
        fingerprint_path: Path to fingerprint JSON (from fingerprint.py)
        probe_path: Path to API probe results JSON (from api_probe.py)
        output_path: Where to save comparison results
        timing_sigma_threshold: Sigma deviation for timing anomaly
        kl_threshold: KL divergence threshold for distribution anomaly

    Returns:
        Comparison results dict
    """
    t_start = time.time()
    ts_start = time.strftime("%Y-%m-%dT%H:%M:%S")

    with open(fingerprint_path) as f:
        fingerprint = json.load(f)
    with open(probe_path) as f:
        probe = json.load(f)

    fp_probes = {p["probe_id"]: p for p in fingerprint.get("probes", [])}
    api_probes = {p["probe_id"]: p for p in probe.get("probes", [])}

    # Match probes by ID
    matched = []
    for pid in fp_probes:
        if pid in api_probes:
            matched.append((fp_probes[pid], api_probes[pid]))

    if not matched:
        return {"error": "No matching probes found between fingerprint and API results"}

    print(f"Comparing {len(matched)} matched probes")
    print(f"  Fingerprint: {fingerprint.get('model_declared', '?')}")
    print(f"  API probe:   {probe.get('model_declared', '?')}")
    print("-" * 60)

    # Layer 1: Timing analysis
    timing_result = _analyze_timing(matched, fingerprint, probe, timing_sigma_threshold)

    # Layer 2: Distribution divergence (if logprobs available)
    distribution_result = _analyze_distribution(matched, kl_threshold)

    # Layer 3: Sidecar-weighted divergence (if HXQ fingerprint)
    sidecar_result = _analyze_sidecar_weighted(matched, kl_threshold)

    # Aggregate verdict
    verdict, confidence = _aggregate_verdict(
        timing_result, distribution_result, sidecar_result
    )

    results = {
        "comparison_version": "1.0",
        "fingerprint_model": fingerprint.get("model_declared", ""),
        "api_model_declared": probe.get("model_declared", ""),
        "api_model_reported": _get_reported_model(probe),
        "n_probes_matched": len(matched),
        "layer1_timing": timing_result,
        "layer2_distribution": distribution_result,
        "layer3_sidecar": sidecar_result,
        "verdict": verdict,
        "confidence": confidence,
        "cost": {
            "wall_time_s": round(time.time() - t_start, 3),
            "python_version": platform.python_version(),
            "hostname": platform.node(),
            "timestamp": ts_start,
        },
    }

    print(f"\n{'=' * 60}")
    print(f"VERDICT: {verdict} (confidence={confidence:.2f})")
    print(f"{'=' * 60}")

    if output_path is None:
        output_path = "polygraph_result.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {output_path}")

    return results


def _analyze_timing(
    matched: List[Tuple[dict, dict]],
    fingerprint: dict,
    probe: dict,
    sigma_threshold: float,
) -> dict:
    """Layer 1: Compare timing profiles."""
    fp_timing = fingerprint.get("profile", {}).get("mean_probe_ms", 0)
    api_timing = probe.get("timing_profile", {})

    # Per-probe timing comparison
    fp_times = []
    api_times = []
    for fp_p, api_p in matched:
        fp_t = fp_p.get("elapsed_ms", 0)
        api_t = api_p.get("elapsed_ms", 0)
        if fp_t > 0 and api_t > 0:
            fp_times.append(fp_t)
            api_times.append(api_t)

    if not fp_times:
        return {"status": "NO_DATA", "detail": "No timing data to compare"}

    # Compare distributions
    import statistics
    fp_mean = statistics.mean(fp_times)
    api_mean = statistics.mean(api_times)
    fp_std = statistics.stdev(fp_times) if len(fp_times) > 1 else fp_mean * 0.1

    # How many sigma is the API timing from the fingerprint timing?
    timing_ratio = api_mean / fp_mean if fp_mean > 0 else float("inf")
    sigma_deviation = abs(api_mean - fp_mean) / fp_std if fp_std > 0 else 0

    # A 3B model running 3x slower suggests a larger model
    anomaly = sigma_deviation > sigma_threshold or timing_ratio > 2.0 or timing_ratio < 0.5

    result = {
        "status": "ANOMALY" if anomaly else "CONSISTENT",
        "fingerprint_mean_ms": round(fp_mean, 1),
        "api_mean_ms": round(api_mean, 1),
        "timing_ratio": round(timing_ratio, 3),
        "sigma_deviation": round(sigma_deviation, 2),
        "threshold_sigma": sigma_threshold,
    }

    status_icon = "!!" if anomaly else "OK"
    print(f"  Layer 1 (timing): {status_icon} ratio={timing_ratio:.2f}x, sigma={sigma_deviation:.1f}")

    return result


def _analyze_distribution(
    matched: List[Tuple[dict, dict]],
    kl_threshold: float,
) -> dict:
    """Layer 2: Compare output probability distributions."""
    # Need logprobs on both sides
    pairs_with_logprobs = []
    for fp_p, api_p in matched:
        fp_lp = fp_p.get("logprobs", [])
        api_lp = api_p.get("logprobs", [])
        if fp_lp and api_lp:
            pairs_with_logprobs.append((fp_p, api_p))

    if not pairs_with_logprobs:
        return {
            "status": "NO_LOGPROBS",
            "detail": "API did not return logprobs — Layer 2 unavailable. "
                      "Request logprobs from provider or use timing-only detection.",
        }

    # Per-probe KL divergence on top-k logprobs
    per_probe_kl = []
    per_probe_token_match = []

    for fp_p, api_p in pairs_with_logprobs:
        kl, token_match_rate = _compute_probe_divergence(fp_p, api_p)
        per_probe_kl.append(kl)
        per_probe_token_match.append(token_match_rate)

    import statistics
    mean_kl = statistics.mean(per_probe_kl)
    mean_match = statistics.mean(per_probe_token_match)

    anomaly = mean_kl > kl_threshold or mean_match < 0.5

    result = {
        "status": "ANOMALY" if anomaly else "CONSISTENT",
        "n_probes_with_logprobs": len(pairs_with_logprobs),
        "mean_kl_divergence": round(mean_kl, 6),
        "mean_token_match_rate": round(mean_match, 4),
        "per_probe_kl": [round(k, 6) for k in per_probe_kl],
        "kl_threshold": kl_threshold,
    }

    status_icon = "!!" if anomaly else "OK"
    print(f"  Layer 2 (distribution): {status_icon} KL={mean_kl:.4f}, token_match={mean_match:.2%}")

    return result


def _analyze_sidecar_weighted(
    matched: List[Tuple[dict, dict]],
    kl_threshold: float,
) -> dict:
    """Layer 3: Sidecar-weighted divergence analysis.

    Uses diagnostic_weight from HXQ fingerprint to focus on tokens
    where the local model is CONFIDENT. These tokens are most diagnostic
    because same-model should agree on them, different-model won't.
    """
    # Need: logprobs on both sides AND sidecar weights on fingerprint
    weighted_pairs = []
    for fp_p, api_p in matched:
        fp_lp = fp_p.get("logprobs", [])
        api_lp = api_p.get("logprobs", [])
        weight = fp_p.get("diagnostic_weight")
        if fp_lp and api_lp and weight is not None:
            weighted_pairs.append((fp_p, api_p, weight))

    if not weighted_pairs:
        return {
            "status": "NO_SIDECAR",
            "detail": "Fingerprint has no sidecar weights (use --hxq mode) "
                      "or API has no logprobs.",
        }

    # Weighted KL: probes where local model is confident get more weight
    weighted_kl_sum = 0.0
    weight_sum = 0.0

    for fp_p, api_p, weight in weighted_pairs:
        kl, _ = _compute_probe_divergence(fp_p, api_p)
        weighted_kl_sum += kl * weight
        weight_sum += weight

    weighted_kl = weighted_kl_sum / weight_sum if weight_sum > 0 else 0.0

    # Compare to unweighted
    unweighted_kls = []
    for fp_p, api_p, _ in weighted_pairs:
        kl, _ = _compute_probe_divergence(fp_p, api_p)
        unweighted_kls.append(kl)

    import statistics
    unweighted_mean = statistics.mean(unweighted_kls) if unweighted_kls else 0.0

    # If weighted KL is higher than unweighted, it means confident tokens
    # diverge MORE than uncertain tokens — strong swap signal
    amplification = weighted_kl / unweighted_mean if unweighted_mean > 0 else 1.0
    anomaly = weighted_kl > kl_threshold

    result = {
        "status": "ANOMALY" if anomaly else "CONSISTENT",
        "n_probes_weighted": len(weighted_pairs),
        "weighted_kl": round(weighted_kl, 6),
        "unweighted_kl": round(unweighted_mean, 6),
        "amplification": round(amplification, 4),
        "interpretation": (
            "Confident tokens diverge MORE than uncertain tokens — strong swap signal"
            if amplification > 1.5
            else "Divergence is uniform across confidence levels — may be noise or subtle swap"
            if amplification > 0.8
            else "Divergence concentrated in uncertain tokens — likely benign variation"
        ),
    }

    status_icon = "!!" if anomaly else "OK"
    print(f"  Layer 3 (sidecar): {status_icon} weighted_KL={weighted_kl:.4f}, amplification={amplification:.2f}x")

    return result


def _compute_probe_divergence(fp_probe: dict, api_probe: dict) -> Tuple[float, float]:
    """Compute KL divergence and token match rate between two probe results.

    Returns (kl_divergence, token_match_rate)
    """
    fp_top_k = fp_probe.get("top_k", [])
    api_top_k = api_probe.get("top_k", [])

    if not fp_top_k or not api_top_k:
        # Fall back to raw logprob comparison
        fp_lp = fp_probe.get("logprobs", [])
        api_lp = api_probe.get("logprobs", [])
        min_len = min(len(fp_lp), len(api_lp))
        if min_len == 0:
            return 0.0, 0.0

        # Simple logprob difference as proxy for KL
        diffs = [abs(fp_lp[i] - api_lp[i]) for i in range(min_len)]
        mean_diff = sum(diffs) / len(diffs)

        # Token match
        fp_tokens = fp_probe.get("tokens", [])
        api_tokens = api_probe.get("tokens", [])
        min_tok = min(len(fp_tokens), len(api_tokens))
        matches = sum(1 for i in range(min_tok) if fp_tokens[i] == api_tokens[i])
        match_rate = matches / min_tok if min_tok > 0 else 0.0

        return mean_diff, match_rate

    # Full top-k KL divergence
    min_positions = min(len(fp_top_k), len(api_top_k))
    kl_per_position = []
    token_matches = 0

    for pos in range(min_positions):
        fp_dist = {t["token"]: math.exp(t["logprob"]) for t in fp_top_k[pos]}
        api_dist = {t["token"]: math.exp(t["logprob"]) for t in api_top_k[pos]}

        # Check if top-1 token matches
        fp_top1 = fp_top_k[pos][0]["token"] if fp_top_k[pos] else ""
        api_top1 = api_top_k[pos][0]["token"] if api_top_k[pos] else ""
        if fp_top1 == api_top1:
            token_matches += 1

        # KL(fp || api) over shared vocabulary
        all_tokens = set(fp_dist.keys()) | set(api_dist.keys())
        kl = 0.0
        for tok in all_tokens:
            p = fp_dist.get(tok, 1e-10)   # fingerprint distribution
            q = api_dist.get(tok, 1e-10)   # API distribution
            if p > 0:
                kl += p * math.log(p / q)
        kl_per_position.append(max(0.0, kl))

    mean_kl = sum(kl_per_position) / len(kl_per_position) if kl_per_position else 0.0
    match_rate = token_matches / min_positions if min_positions > 0 else 0.0

    return mean_kl, match_rate


def _aggregate_verdict(
    timing: dict,
    distribution: dict,
    sidecar: dict,
) -> Tuple[str, float]:
    """Combine all layers into a final verdict.

    Returns (verdict_string, confidence_float)
    """
    signals = []

    # Layer 1
    if timing.get("status") == "ANOMALY":
        signals.append(("timing", 0.6))  # timing alone is moderate evidence
    elif timing.get("status") == "CONSISTENT":
        signals.append(("timing_ok", -0.3))

    # Layer 2
    if distribution.get("status") == "ANOMALY":
        signals.append(("distribution", 0.8))  # distribution is strong evidence
    elif distribution.get("status") == "CONSISTENT":
        signals.append(("distribution_ok", -0.5))

    # Layer 3
    if sidecar.get("status") == "ANOMALY":
        amp = sidecar.get("amplification", 1.0)
        if amp > 1.5:
            signals.append(("sidecar_amplified", 0.9))  # very strong
        else:
            signals.append(("sidecar", 0.7))
    elif sidecar.get("status") == "CONSISTENT":
        signals.append(("sidecar_ok", -0.4))

    # Aggregate
    if not signals:
        return "INCONCLUSIVE", 0.0

    score = sum(s[1] for s in signals) / len(signals)

    if score > 0.3:
        return "SWAP_DETECTED", min(0.99, abs(score))
    elif score > 0.0:
        return "SUSPICIOUS", abs(score)
    else:
        return "CONSISTENT", min(0.99, abs(score))


def _get_reported_model(probe: dict) -> str:
    """Extract the model name the API actually reported in responses."""
    for p in probe.get("probes", []):
        reported = p.get("model_reported", "")
        if reported:
            return reported
    return ""


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare fingerprint vs API probe")
    parser.add_argument("--fingerprint", required=True, help="Path to fingerprint JSON")
    parser.add_argument("--probe", required=True, help="Path to API probe results JSON")
    parser.add_argument("--output", help="Output path")
    parser.add_argument("--timing-sigma", type=float, default=3.0, help="Timing anomaly threshold (sigma)")
    parser.add_argument("--kl-threshold", type=float, default=0.5, help="KL divergence anomaly threshold")
    args = parser.parse_args()

    compare(args.fingerprint, args.probe, args.output, args.timing_sigma, args.kl_threshold)
