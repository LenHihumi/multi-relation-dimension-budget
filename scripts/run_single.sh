#!/bin/bash
# single-width ablation: train the same objective at one width, evaluate on the full grid
# --- portable roots: derived from this file, overridable by environment ---
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${SB_ROOT:-$(cd "$HERE/.." && pwd)}"
OUT="${SB_OUT:-$ROOT/outputs/diag}"
CACHE="${SB_CACHE:-$(cd "$ROOT/.." && pwd)/.cache/sb}"
set -u
ROOT=$ROOT
for enc in clip_vitl14 dinov3_vitl16; do
  echo "=== $enc single@160 $(date +%H:%M:%S)"
  python3 -u "$ROOT/scripts/diag/proj_rank.py" --ds inshop --enc "$enc" --relation inst \
      --K 320 --steps 200 --queries 100 --threads 48 --only-prefix 160 \
      --out "$ROOT/outputs/diag/proj_single160__${enc}__inst.json" 2>&1 | tail -12
  echo "=== $enc single@48 $(date +%H:%M:%S)"
  python3 -u "$ROOT/scripts/diag/proj_rank.py" --ds inshop --enc "$enc" --relation inst \
      --K 320 --steps 200 --queries 100 --threads 48 --only-prefix 48 \
      --out "$ROOT/outputs/diag/proj_single48__${enc}__inst.json" 2>&1 | tail -12
done
echo "=== done $(date +%H:%M:%S)"
