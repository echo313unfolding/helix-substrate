#!/usr/bin/env python3
"""
Adoption tracker for helix-substrate ecosystem.

Pulls metrics from HuggingFace, GitHub, and PyPI into a time-series JSONL log.
Run manually or on cron to track real adoption signals over time.

Usage:
  python3 tools/adoption_tracker.py                # snapshot + print summary
  python3 tools/adoption_tracker.py --json          # machine-readable output
  python3 tools/adoption_tracker.py --diff          # show deltas since last check
  python3 tools/adoption_tracker.py --history 7     # show last 7 snapshots
  python3 tools/adoption_tracker.py --watch 30      # re-check every 30 minutes

Data stored in: receipts/adoption/adoption_log.jsonl
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Where we store the time series
LOG_DIR = Path(__file__).resolve().parent.parent / "receipts" / "adoption"
LOG_FILE = LOG_DIR / "adoption_log.jsonl"

HF_ORG = "EchoLabs33"
GITHUB_ORG = "echo313unfolding"
GITHUB_REPOS = ["helix-substrate", "helix-online-kv"]
PYPI_PACKAGE = "helix-substrate"


def fetch_hf_stats():
    """Fetch download counts and likes from HuggingFace."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        models = list(api.list_models(author=HF_ORG))
        per_model = {}
        for m in models:
            per_model[m.id] = {
                "downloads": m.downloads or 0,
                "likes": m.likes or 0,
            }
        total_downloads = sum(v["downloads"] for v in per_model.values())
        total_likes = sum(v["likes"] for v in per_model.values())
        return {
            "total_downloads": total_downloads,
            "total_likes": total_likes,
            "n_models": len(models),
            "per_model": per_model,
        }
    except Exception as e:
        return {"error": str(e)}


def fetch_github_stats():
    """Fetch stars, forks, issues from GitHub via gh CLI."""
    repos = {}
    for repo_name in GITHUB_REPOS:
        try:
            result = subprocess.run(
                ["gh", "api", f"repos/{GITHUB_ORG}/{repo_name}",
                 "--jq", "{stars: .stargazers_count, forks: .forks_count, "
                         "watchers: .subscribers_count, open_issues: .open_issues_count}"],
                capture_output=True, text=True, timeout=15
            )
            if result.returncode == 0:
                repos[repo_name] = json.loads(result.stdout.strip())
            else:
                repos[repo_name] = {"error": result.stderr.strip()}
        except Exception as e:
            repos[repo_name] = {"error": str(e)}

    # Also check for recent issues/PRs/discussions
    for repo_name in GITHUB_REPOS:
        try:
            result = subprocess.run(
                ["gh", "api", f"repos/{GITHUB_ORG}/{repo_name}/issues",
                 "--jq", "length"],
                capture_output=True, text=True, timeout=15
            )
            if result.returncode == 0:
                repos[repo_name]["total_issues_ever"] = int(result.stdout.strip())
        except Exception:
            pass

    # Total stars across all repos
    total_stars = sum(r.get("stars", 0) for r in repos.values() if isinstance(r, dict))
    total_forks = sum(r.get("forks", 0) for r in repos.values() if isinstance(r, dict))
    return {
        "total_stars": total_stars,
        "total_forks": total_forks,
        "repos": repos,
    }


def fetch_pypi_stats():
    """Fetch recent download counts from PyPI stats API."""
    import urllib.request
    try:
        url = f"https://pypistats.org/api/packages/{PYPI_PACKAGE}/recent"
        req = urllib.request.Request(url, headers={"User-Agent": "adoption-tracker/1.0"})
        resp = urllib.request.urlopen(req, timeout=10)
        data = json.loads(resp.read().decode())
        return {
            "last_day": data.get("data", {}).get("last_day", 0),
            "last_week": data.get("data", {}).get("last_week", 0),
            "last_month": data.get("data", {}).get("last_month", 0),
        }
    except Exception as e:
        return {"error": str(e)}


def take_snapshot():
    """Take a full snapshot of all metrics."""
    ts = datetime.now(timezone.utc).isoformat()
    return {
        "timestamp": ts,
        "huggingface": fetch_hf_stats(),
        "github": fetch_github_stats(),
        "pypi": fetch_pypi_stats(),
    }


def save_snapshot(snapshot):
    """Append snapshot to JSONL log."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(snapshot) + "\n")


def load_history(n=None):
    """Load last N snapshots from log."""
    if not LOG_FILE.exists():
        return []
    lines = LOG_FILE.read_text(encoding="utf-8").strip().split("\n")
    entries = [json.loads(line) for line in lines if line.strip()]
    if n is not None:
        entries = entries[-n:]
    return entries


def compute_diff(current, previous):
    """Compute deltas between two snapshots."""
    diffs = {}

    # HF diffs
    hf_cur = current.get("huggingface", {})
    hf_prev = previous.get("huggingface", {})
    if "error" not in hf_cur and "error" not in hf_prev:
        diffs["hf_downloads"] = hf_cur.get("total_downloads", 0) - hf_prev.get("total_downloads", 0)
        diffs["hf_likes"] = hf_cur.get("total_likes", 0) - hf_prev.get("total_likes", 0)

        # Per-model diffs
        model_diffs = {}
        cur_models = hf_cur.get("per_model", {})
        prev_models = hf_prev.get("per_model", {})
        for model_id, cur_stats in cur_models.items():
            prev_stats = prev_models.get(model_id, {"downloads": 0, "likes": 0})
            dl_delta = cur_stats["downloads"] - prev_stats.get("downloads", 0)
            if dl_delta != 0:
                model_diffs[model_id] = dl_delta
        diffs["hf_per_model"] = model_diffs

    # GitHub diffs
    gh_cur = current.get("github", {})
    gh_prev = previous.get("github", {})
    diffs["gh_stars"] = gh_cur.get("total_stars", 0) - gh_prev.get("total_stars", 0)
    diffs["gh_forks"] = gh_cur.get("total_forks", 0) - gh_prev.get("total_forks", 0)

    # Time between snapshots
    try:
        t_cur = datetime.fromisoformat(current["timestamp"])
        t_prev = datetime.fromisoformat(previous["timestamp"])
        diffs["hours_elapsed"] = round((t_cur - t_prev).total_seconds() / 3600, 1)
    except Exception:
        diffs["hours_elapsed"] = None

    return diffs


def print_summary(snapshot, diff=None):
    """Print human-readable summary."""
    ts = snapshot["timestamp"][:19].replace("T", " ")
    print(f"{'=' * 60}")
    print(f"  Adoption Tracker  |  {ts} UTC")
    print(f"{'=' * 60}")

    # HuggingFace
    hf = snapshot.get("huggingface", {})
    if "error" in hf:
        print(f"\n  HuggingFace: ERROR - {hf['error']}")
    else:
        dl = hf.get("total_downloads", 0)
        likes = hf.get("total_likes", 0)
        n_models = hf.get("n_models", 0)
        dl_delta = f" (+{diff['hf_downloads']})" if diff and diff.get("hf_downloads", 0) > 0 else ""
        like_delta = f" (+{diff['hf_likes']})" if diff and diff.get("hf_likes", 0) > 0 else ""
        print(f"\n  HuggingFace ({n_models} models):")
        print(f"    Downloads: {dl:,}{dl_delta}")
        print(f"    Likes:     {likes}{like_delta}")

        # Per-model breakdown
        per_model = hf.get("per_model", {})
        if per_model:
            ranked = sorted(per_model.items(), key=lambda x: -x[1]["downloads"])
            print(f"\n    {'Model':<45s} {'DL':>6s}  {'Delta':>6s}")
            print(f"    {'-'*45} {'-'*6}  {'-'*6}")
            for model_id, stats in ranked:
                short = model_id.split("/")[-1]
                dl_d = ""
                if diff and diff.get("hf_per_model", {}).get(model_id, 0) > 0:
                    dl_d = f"+{diff['hf_per_model'][model_id]}"
                print(f"    {short:<45s} {stats['downloads']:6d}  {dl_d:>6s}")

    # GitHub
    gh = snapshot.get("github", {})
    total_stars = gh.get("total_stars", 0)
    total_forks = gh.get("total_forks", 0)
    star_delta = f" (+{diff['gh_stars']})" if diff and diff.get("gh_stars", 0) > 0 else ""
    fork_delta = f" (+{diff['gh_forks']})" if diff and diff.get("gh_forks", 0) > 0 else ""
    print(f"\n  GitHub:")
    print(f"    Stars:  {total_stars}{star_delta}")
    print(f"    Forks:  {total_forks}{fork_delta}")
    for repo_name, repo_stats in gh.get("repos", {}).items():
        if isinstance(repo_stats, dict) and "error" not in repo_stats:
            issues = repo_stats.get("open_issues", 0)
            watchers = repo_stats.get("watchers", 0)
            print(f"    {repo_name}: {repo_stats.get('stars', 0)} stars, "
                  f"{issues} open issues, {watchers} watchers")

    # PyPI
    pypi = snapshot.get("pypi", {})
    if "error" in pypi:
        print(f"\n  PyPI: unavailable ({pypi['error'][:60]})")
    else:
        print(f"\n  PyPI (helix-substrate):")
        print(f"    Last day:   {pypi.get('last_day', '?')}")
        print(f"    Last week:  {pypi.get('last_week', '?')}")
        print(f"    Last month: {pypi.get('last_month', '?')}")

    # Time delta
    if diff and diff.get("hours_elapsed"):
        print(f"\n  Time since last check: {diff['hours_elapsed']}h")

    # Signals summary
    print(f"\n  Real adoption signals:")
    real_signals = []
    if total_stars > 0:
        real_signals.append(f"{total_stars} GitHub stars")
    if total_forks > 0:
        real_signals.append(f"{total_forks} forks")
    if gh.get("repos", {}).get("helix-substrate", {}).get("open_issues", 0) > 0:
        real_signals.append(f"{gh['repos']['helix-substrate']['open_issues']} open issues (people are USING it)")
    hf_likes_total = hf.get("total_likes", 0) if "error" not in hf else 0
    if hf_likes_total > 0:
        real_signals.append(f"{hf_likes_total} HF likes")
    if not real_signals:
        real_signals.append("None yet - downloads alone don't confirm usage")
    for s in real_signals:
        print(f"    - {s}")

    print(f"\n{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description="Track adoption metrics for helix-substrate ecosystem")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of summary")
    parser.add_argument("--diff", action="store_true", help="Show deltas since last snapshot")
    parser.add_argument("--history", type=int, metavar="N", help="Show last N snapshots")
    parser.add_argument("--watch", type=int, metavar="MINS", help="Re-check every N minutes")
    parser.add_argument("--no-save", action="store_true", help="Don't save snapshot to log")
    args = parser.parse_args()

    if args.history:
        entries = load_history(args.history)
        if not entries:
            print("No history found.", file=sys.stderr)
            sys.exit(1)
        for i, entry in enumerate(entries):
            prev = entries[i - 1] if i > 0 else None
            diff = compute_diff(entry, prev) if prev else None
            if args.json:
                out = {"snapshot": entry}
                if diff:
                    out["diff"] = diff
                print(json.dumps(out, indent=2))
            else:
                print_summary(entry, diff)
                print()
        return

    while True:
        snapshot = take_snapshot()

        if not args.no_save:
            save_snapshot(snapshot)

        # Load previous for diff
        diff = None
        if args.diff:
            history = load_history(2)
            if len(history) >= 2:
                diff = compute_diff(history[-1], history[-2])

        if args.json:
            out = {"snapshot": snapshot}
            if diff:
                out["diff"] = diff
            print(json.dumps(out, indent=2))
        else:
            print_summary(snapshot, diff)

        if not args.watch:
            break

        print(f"\n  Next check in {args.watch} minutes...", file=sys.stderr)
        time.sleep(args.watch * 60)


if __name__ == "__main__":
    main()
