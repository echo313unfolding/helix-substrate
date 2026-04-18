"""Entry point for python3 -m api_polygraph."""

import sys


def main():
    if len(sys.argv) < 2:
        print("API Polygraph — detect when cloud providers swap your model")
        print()
        print("Usage: python3 -m api_polygraph <command> [args]")
        print()
        print("Commands:")
        print("  fingerprint  Generate local ground truth from open-weights model")
        print("  probe        Send standardized probes to a cloud API")
        print("  compare      Compare fingerprint vs API probe results")
        print()
        print("Workflow:")
        print("  1. python3 -m api_polygraph fingerprint --model meta-llama/Llama-3.2-3B-Instruct")
        print("  2. python3 -m api_polygraph probe --api-url https://api.provider.com/v1 --model llama-3b --api-key $KEY")
        print("  3. python3 -m api_polygraph compare --fingerprint fingerprint_*.json --probe api_probe_*.json")
        print()
        print("With HXQ sidecar weighting (Layer 3):")
        print("  1. python3 -m api_polygraph fingerprint --model /path/to/hxq/model --hxq")
        print("  2-3. Same as above")
        return

    cmd = sys.argv[1]
    # Remove the subcommand from argv so argparse in each module works
    sys.argv = [sys.argv[0]] + sys.argv[2:]

    if cmd == "fingerprint":
        from .fingerprint import main as fp_main
        # fingerprint.py uses __main__ block with argparse
        import argparse
        parser = argparse.ArgumentParser(description="Generate model fingerprint")
        parser.add_argument("--model", required=True)
        parser.add_argument("--hxq", action="store_true")
        parser.add_argument("--output", default=None)
        parser.add_argument("--device", default="auto")
        parser.add_argument("--dtype", default="float16")
        args = parser.parse_args()

        if args.hxq:
            from .fingerprint import generate_fingerprint_hxq
            generate_fingerprint_hxq(args.model, args.output, args.device)
        else:
            from .fingerprint import generate_fingerprint_direct
            generate_fingerprint_direct(args.model, args.output, args.device, args.dtype)

    elif cmd == "probe":
        from .api_probe import probe_api, probe_ollama
        import argparse
        parser = argparse.ArgumentParser(description="Probe an API")
        parser.add_argument("--api-url", default=None)
        parser.add_argument("--api-key", default=None)
        parser.add_argument("--model", required=True)
        parser.add_argument("--output", default=None)
        parser.add_argument("--ollama", action="store_true")
        parser.add_argument("--timeout", type=int, default=30)
        args = parser.parse_args()

        if args.ollama:
            probe_ollama(args.model, args.output, args.api_url or "http://localhost:11434")
        else:
            if not args.api_url:
                print("Error: --api-url required")
                sys.exit(1)
            probe_api(args.api_url, args.model, args.api_key, args.output, args.timeout)

    elif cmd == "compare":
        from .compare import compare
        import argparse
        parser = argparse.ArgumentParser(description="Compare fingerprint vs probe")
        parser.add_argument("--fingerprint", required=True)
        parser.add_argument("--probe", required=True)
        parser.add_argument("--output", default=None)
        parser.add_argument("--timing-sigma", type=float, default=3.0)
        parser.add_argument("--kl-threshold", type=float, default=0.5)
        args = parser.parse_args()

        compare(args.fingerprint, args.probe, args.output, args.timing_sigma, args.kl_threshold)

    else:
        print(f"Unknown command: {cmd}")
        print("Use: fingerprint, probe, or compare")
        sys.exit(1)


if __name__ == "__main__":
    main()
