#!/bin/bash
# Re-run the two ImageNet evaluations (the first pass died on int('animal')) and the end-to-end ANN check.
# --- portable roots: derived from this file, overridable by environment ---
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${SB_ROOT:-$(cd "$HERE/.." && pwd)}"
OUT="${SB_OUT:-$ROOT/outputs/diag}"
CACHE="${SB_CACHE:-$(cd "$ROOT/.." && pwd)/.cache/sb}"
set -u
ROOT=$ROOT
S=$ROOT/scripts/diag
O=$ROOT/outputs/diag
cd /tmp || exit 1
for rel in inst cat; do
  echo "=== imagenet eval $rel $(date +%H:%M:%S)"
  python3 -u "$S/proj_eval.py" --ds imagenet --enc clip_vitl14 --relation "$rel" \
      --w "$O/W_imagenet_${rel}.npz" --out "$O/projeval__imagenet__${rel}.json" 2>&1 | tail -20
done
for rel in inst cat; do
  echo "=== ann $rel $(date +%H:%M:%S)"
  python3 -u "$S/proj_ann.py" --enc clip_vitl14 --ds inshop --rel "$rel" \
      --w "$O/W_clip_vitl14_${rel}.npz" --out "$O/projann__clip__${rel}.json" 2>&1 | tail -14
done
echo "=== done $(date +%H:%M:%S)"
