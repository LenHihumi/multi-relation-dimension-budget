#!/bin/bash
# Order matters: the In-shop cost table is timed first, on a quiet box, because build/latency numbers are
# only comparable when nothing else is running. The long single-cell jobs come after.
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
  echo "=== axes $enc $(date +%H:%M:%S)"
  python3 -u "$S/multidim.py" --ds inshop --enc "$enc" --queries 300 \
      --out "$O/multidim__${enc}__inshop.json" 2>&1 | tail -8
done
echo "=== imagenet eval inst $(date +%H:%M:%S)"
python3 -u "$S/proj_eval.py" --ds imagenet --enc clip_vitl14 --relation inst \
    --w "$O/W_imagenet_inst.npz" --out "$O/projeval__imagenet__inst.json" 2>&1 | tail -18
echo "=== axes imagenet $(date +%H:%M:%S)"
python3 -u "$S/multidim.py" --ds imagenet --enc clip_vitl14 --queries 200 \
    --out "$O/multidim__clip_vitl14__imagenet.json" 2>&1 | tail -8
echo "=== done $(date +%H:%M:%S)"
