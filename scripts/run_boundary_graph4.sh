#!/bin/bash
# 4-bit traversal with block scales: the only 4-bit configuration that survived the exact-search sweep.
# --- portable roots: derived from this file, overridable by environment ---
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${SB_ROOT:-$(cd "$HERE/.." && pwd)}"
OUT="${SB_OUT:-$ROOT/outputs/diag}"
CACHE="${SB_CACHE:-$(cd "$ROOT/.." && pwd)/.cache/sb}"
set -u
ROOT=$ROOT
for e in clip_vitl14 dinov3_vitl16 siglip2_so400m wemm_2b qwen3vl_emb_8b; do
  echo "=== $e $(date +%H:%M:%S)"
  python3 -u "$ROOT/scripts/diag/quant_boundary.py" --ds inshop --enc "$e" --skip-exact \
      --graph-bits 4 --graph-modes block32 --queries 300 \
      --out "$ROOT/outputs/diag/quant_boundary_graph4b__${e}__inshop.json"
done
echo "=== imagenet $(date +%H:%M:%S)"
python3 -u "$ROOT/scripts/diag/quant_boundary.py" --ds imagenet --enc clip_vitl14 --skip-exact \
    --graph-bits 4 --graph-modes block32 --queries 300 \
    --out "$ROOT/outputs/diag/quant_boundary_graph4b__clip_vitl14__imagenet.json"
echo "=== done $(date +%H:%M:%S)"
