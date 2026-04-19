#!/bin/bash
# Queue grouped VQ Phase 1 baseline — waits for V-step to finish, then launches.
# WO-HELIX-STE-01 follow-up: first born-compressed run on real HXQ codec (d=4).

VSTEP_PID=44461

echo "[queue] Waiting for V-step PID $VSTEP_PID to finish..."
while kill -0 $VSTEP_PID 2>/dev/null; do
    sleep 30
done
echo "[queue] V-step finished at $(date -Iseconds). Launching grouped VQ Phase 1..."

cd /home/voidstr3m33/helix-substrate

python3 -m echo_hybrid.train_phase1 \
  --steps 500 --batch-size 2 --seq-len 64 \
  --vector-dim 4 --compress-every 25 \
  --device cpu

echo "[queue] Grouped VQ Phase 1 complete at $(date -Iseconds)."
