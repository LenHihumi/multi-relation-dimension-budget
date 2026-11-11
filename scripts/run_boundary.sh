#!/bin/bash
# quant_boundary across every cell.  One cell at a time; each is CPU-bound on exact search, so running
# them concurrently only splits the same 128 cores.
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
for enc in clip_vitl14 dinov3_vitl16 siglip2_so400m wemm_2b qwen3vl_emb_8b; do
  echo "=== $enc $(date +%H:%M:%S)"
  python3 -u "$S/quant_boundary.py" --ds inshop --enc "$enc" \
      --bits 8,6,4,3,2 --modes tensor,block32,dim --graph-bits 8,4 \
      --queries 300 --out "$O/quant_boundary__${enc}__inshop.json"
done
echo "=== imagenet $(date +%H:%M:%S)"
python3 -u "$S/quant_boundary.py" --ds imagenet --enc clip_vitl14 \
    --bits 8,4 --modes tensor,block32 --graph-bits 8,4 \
    --queries 300 --out "$O/quant_boundary__clip_vitl14__imagenet.json"
echo "=== done $(date +%H:%M:%S)"
