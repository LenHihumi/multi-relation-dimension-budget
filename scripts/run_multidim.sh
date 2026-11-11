#!/bin/bash
# multidim across every cell: build / append / p50-p99 / RSS on the four designs.
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
  python3 -u "$S/multidim.py" --ds inshop --enc "$enc" --queries 300 \
      --out "$O/multidim__${enc}__inshop.json"
done
echo "=== imagenet $(date +%H:%M:%S)"
python3 -u "$S/multidim.py" --ds imagenet --enc clip_vitl14 --queries 200 \
    --out "$O/multidim__clip_vitl14__imagenet.json"
echo "=== done $(date +%H:%M:%S)"
