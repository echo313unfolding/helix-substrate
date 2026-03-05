#!/bin/bash
# Benchmark receipt generator for helix-substrate
set -e

OUTPUT="receipts/benchmarks/16k_baseline.txt"

echo "=== helix-substrate Benchmark Receipt ===" > "$OUTPUT"
echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$OUTPUT"
echo "package: helix-substrate==$(python3 -c 'import helix_substrate; print(helix_substrate.__version__)')" >> "$OUTPUT"
echo "python: $(python3 --version)" >> "$OUTPUT"
echo "" >> "$OUTPUT"
echo "=== Hardware ===" >> "$OUTPUT"
lscpu | grep -E "^(Model name|CPU\(s\)|Thread|Core)" >> "$OUTPUT"
echo "memory: $(free -h | grep Mem | awk '{print $2}')" >> "$OUTPUT"
echo "" >> "$OUTPUT"
echo "=== Command ===" >> "$OUTPUT"
echo "python tools/bench_memory.py --rows 16384 --cols 16384 --block-rows 32" >> "$OUTPUT"
echo "" >> "$OUTPUT"
echo "=== Output ===" >> "$OUTPUT"
python3 tools/bench_memory.py --rows 16384 --cols 16384 --block-rows 32 2>&1 >> "$OUTPUT"
echo "" >> "$OUTPUT"
echo "=== SHA256 of this receipt ===" >> "$OUTPUT"
sha256sum "$OUTPUT" | cut -d' ' -f1

cat "$OUTPUT"
