"""API Probe — send standardized probes to a cloud API and record responses.

Supports OpenAI-compatible APIs (used by most providers).
Records: generated tokens, logprobs (if available), timing per request.

Usage:
    python3 -m api_polygraph.api_probe \
        --api-url https://api.provider.com/v1 \
        --api-key $API_KEY \
        --model llama-3.2-3b-instruct \
        --output api_probe_results.json
"""

import hashlib
import json
import os
import platform
import resource
import time
from typing import Any, Dict, List, Optional

from .probe_corpus import PROBES, CORPUS_VERSION


def probe_api(
    api_url: str,
    model: str,
    api_key: Optional[str] = None,
    output_path: Optional[str] = None,
    timeout: int = 30,
) -> Dict[str, Any]:
    """Send all probe prompts to an API and record responses.

    Args:
        api_url: Base URL (e.g., "https://api.provider.com/v1")
        model: Model name as the provider expects it
        api_key: API key (or set via environment)
        output_path: Where to save results
        timeout: Per-request timeout in seconds

    Returns:
        Probe results dict (also saved to disk)
    """
    import requests

    if api_key is None:
        api_key = os.environ.get("API_POLYGRAPH_KEY", "")

    t_start = time.time()
    cpu_start = time.process_time()
    ts_start = time.strftime("%Y-%m-%dT%H:%M:%S")

    chat_url = f"{api_url.rstrip('/')}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    print(f"Probing API: {api_url}")
    print(f"Model declared: {model}")
    print(f"Probes: {len(PROBES)}")
    print("-" * 50)

    probe_results = []
    for i, (probe_id, prompt, category, max_tokens) in enumerate(PROBES):
        print(f"  Probe {i+1}/{len(PROBES)}: {probe_id}...", end=" ", flush=True)

        result = _send_probe(
            chat_url, headers, model, prompt, max_tokens, timeout
        )
        result["probe_id"] = probe_id
        result["category"] = category
        result["prompt_hash"] = hashlib.sha256(prompt.encode()).hexdigest()[:16]

        probe_results.append(result)

        if result.get("error"):
            print(f"ERROR: {result['error']}")
        else:
            logprob_str = f", mean_lp={result.get('mean_logprob', 'N/A')}"
            print(f"{result['elapsed_ms']:.0f}ms, {len(result.get('tokens', []))} tokens{logprob_str}")

    # Build results
    results = {
        "probe_version": "1.0",
        "corpus_version": CORPUS_VERSION,
        "api_url": api_url,
        "model_declared": model,
        "n_probes": len(probe_results),
        "probes": probe_results,
        "timing_profile": _compute_timing_profile(probe_results),
        "cost": {
            "wall_time_s": round(time.time() - t_start, 3),
            "cpu_time_s": round(time.process_time() - cpu_start, 3),
            "peak_memory_mb": round(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1
            ),
            "python_version": platform.python_version(),
            "hostname": platform.node(),
            "timestamp_start": ts_start,
            "timestamp_end": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    }

    if output_path is None:
        output_path = f"api_probe_{model.replace('/', '_')}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {output_path}")

    return results


def _send_probe(
    chat_url: str,
    headers: dict,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: int,
) -> dict:
    """Send a single probe to the API."""
    import requests

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "logprobs": True,
        "top_logprobs": 5,
    }

    t_start = time.time()
    ttft = None

    try:
        resp = requests.post(
            chat_url,
            headers=headers,
            json=payload,
            timeout=timeout,
        )
        elapsed_ms = (time.time() - t_start) * 1000
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.Timeout:
        return {
            "prompt": prompt,
            "error": f"Timeout after {timeout}s",
            "elapsed_ms": (time.time() - t_start) * 1000,
        }
    except requests.exceptions.HTTPError as e:
        return {
            "prompt": prompt,
            "error": f"HTTP {resp.status_code}: {resp.text[:200]}",
            "elapsed_ms": (time.time() - t_start) * 1000,
        }
    except Exception as e:
        return {
            "prompt": prompt,
            "error": str(e),
            "elapsed_ms": (time.time() - t_start) * 1000,
        }

    # Parse response
    choice = data.get("choices", [{}])[0]
    message = choice.get("message", {})
    content = message.get("content", "")

    # Extract tokens and logprobs
    tokens = []
    logprobs = []
    top_k_per_position = []

    logprobs_data = choice.get("logprobs")
    if logprobs_data and "content" in logprobs_data:
        for token_info in logprobs_data["content"]:
            tokens.append(token_info.get("token", ""))
            logprobs.append(round(token_info.get("logprob", 0.0), 6))

            # Top alternatives
            top_alts = token_info.get("top_logprobs", [])
            top_k = [
                {"token": a.get("token", ""), "logprob": round(a.get("logprob", 0.0), 6)}
                for a in top_alts[:5]
            ]
            top_k_per_position.append(top_k)
    else:
        # API doesn't support logprobs — record tokens only
        tokens = list(content)  # character-level fallback

    # Usage stats
    usage = data.get("usage", {})

    result = {
        "prompt": prompt,
        "content": content,
        "tokens": tokens,
        "logprobs": logprobs,
        "top_k": top_k_per_position,
        "mean_logprob": round(sum(logprobs) / len(logprobs), 6) if logprobs else None,
        "has_logprobs": bool(logprobs),
        "elapsed_ms": round(elapsed_ms, 1),
        "usage": {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
        },
        "model_reported": data.get("model", ""),
        "error": None,
    }

    return result


def _compute_timing_profile(probe_results: List[Dict]) -> Dict:
    """Compute timing statistics across all probes."""
    times = [r.get("elapsed_ms", 0) for r in probe_results if not r.get("error")]
    if not times:
        return {"n": 0}

    import statistics
    return {
        "n": len(times),
        "mean_ms": round(statistics.mean(times), 1),
        "median_ms": round(statistics.median(times), 1),
        "stdev_ms": round(statistics.stdev(times), 1) if len(times) > 1 else 0,
        "min_ms": round(min(times), 1),
        "max_ms": round(max(times), 1),
        "p95_ms": round(sorted(times)[int(len(times) * 0.95)], 1) if len(times) >= 20 else round(max(times), 1),
    }


# Also support Ollama for local testing
def probe_ollama(
    model: str,
    output_path: Optional[str] = None,
    url: str = "http://localhost:11434",
) -> Dict[str, Any]:
    """Send probes to a local Ollama instance.

    Ollama doesn't support logprobs natively, so this captures
    timing and output text only (Layer 1 detection).
    """
    import requests

    t_start = time.time()
    ts_start = time.strftime("%Y-%m-%dT%H:%M:%S")

    print(f"Probing Ollama: {url}")
    print(f"Model: {model}")
    print("-" * 50)

    probe_results = []
    for i, (probe_id, prompt, category, max_tokens) in enumerate(PROBES):
        print(f"  Probe {i+1}/{len(PROBES)}: {probe_id}...", end=" ", flush=True)

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "temperature": 0,
                "num_predict": max_tokens,
            },
        }

        probe_start = time.time()
        try:
            resp = requests.post(f"{url}/api/chat", json=payload, timeout=60)
            elapsed_ms = (time.time() - probe_start) * 1000
            resp.raise_for_status()
            data = resp.json()

            content = data.get("message", {}).get("content", "")
            total_dur_ns = data.get("total_duration", 0)
            eval_dur_ns = data.get("eval_duration", 0)
            eval_count = data.get("eval_count", 0)

            tokens_per_sec = (eval_count / (eval_dur_ns / 1e9)) if eval_dur_ns > 0 else 0

            result = {
                "probe_id": probe_id,
                "category": category,
                "prompt": prompt,
                "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest()[:16],
                "content": content,
                "elapsed_ms": round(elapsed_ms, 1),
                "tokens_per_sec": round(tokens_per_sec, 2),
                "eval_count": eval_count,
                "total_duration_ms": round(total_dur_ns / 1e6, 1),
                "has_logprobs": False,
                "error": None,
            }
            print(f"{elapsed_ms:.0f}ms, {eval_count} tokens, {tokens_per_sec:.1f} tok/s")

        except Exception as e:
            result = {
                "probe_id": probe_id,
                "category": category,
                "prompt": prompt,
                "error": str(e),
                "elapsed_ms": round((time.time() - probe_start) * 1000, 1),
            }
            print(f"ERROR: {e}")

        probe_results.append(result)

    results = {
        "probe_version": "1.0",
        "corpus_version": CORPUS_VERSION,
        "api_url": url,
        "model_declared": model,
        "mode": "ollama",
        "n_probes": len(probe_results),
        "probes": probe_results,
        "timing_profile": _compute_timing_profile(probe_results),
        "cost": {
            "wall_time_s": round(time.time() - t_start, 3),
            "python_version": platform.python_version(),
            "hostname": platform.node(),
            "timestamp_start": ts_start,
            "timestamp_end": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    }

    if output_path is None:
        output_path = f"api_probe_ollama_{model.replace(':', '_')}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {output_path}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Probe an API for model fingerprinting")
    parser.add_argument("--api-url", help="API base URL")
    parser.add_argument("--api-key", help="API key")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--output", help="Output path")
    parser.add_argument("--ollama", action="store_true", help="Use Ollama instead of OpenAI API")
    parser.add_argument("--timeout", type=int, default=30, help="Per-request timeout")
    args = parser.parse_args()

    if args.ollama:
        probe_ollama(args.model, args.output, args.api_url or "http://localhost:11434")
    else:
        if not args.api_url:
            print("Error: --api-url required for OpenAI-compatible API")
            exit(1)
        probe_api(args.api_url, args.model, args.api_key, args.output, args.timeout)
