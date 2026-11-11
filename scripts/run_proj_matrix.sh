#!/bin/bash
# The projection matrix: five encoders x two relations at 52K, then ImageNet at 1.28M.
# Each cell trains W, then re-evaluates it with the *independent* instrument (proj_eval.py).
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

for enc in clip_vitl14 dinov3_vitl16 siglip2_so400m wemm_2b qwen3vl_emb_5b qwen3vl_emb_8b; do
  case "$enc" in
    qwen3vl_emb_5b) continue;;            # not a cell we hold; kept for shape of the loop
  esac
  for rel in inst cat; do
    echo "=== $enc $rel $(date +%H:%M:%S)"
    python3 -u "$S/proj_rank.py" --ds inshop --enc "$enc" --relation "$rel" --K 320 --steps 200 \
        --queries 100 --threads 48 --save-w "$O/W_${enc}_${rel}.npz" \
        --out "$O/proj__${enc}__${rel}.json" 2>&1 | tail -25
    python3 -u "$S/proj_eval.py" --ds inshop --enc "$enc" --relation "$rel" \
        --w "$O/W_${enc}_${rel}.npz" --out "$O/projeval__${enc}__${rel}.json" 2>&1 | tail -18
  done
done

for rel in inst cat; do
  echo "=== imagenet $rel $(date +%H:%M:%S)"
  python3 -u "$S/proj_rank.py" --ds imagenet --enc clip_vitl14 --relation "$rel" --K 320 \
      --steps 200 --queries 100 --threads 48 --save-w "$O/W_imagenet_${rel}.npz" \
      --out "$O/proj__imagenet__${rel}.json" 2>&1 | tail -25
  python3 -u "$S/proj_eval.py" --ds imagenet --enc clip_vitl14 --relation "$rel" \
      --w "$O/W_imagenet_${rel}.npz" --out "$O/projeval__imagenet__${rel}.json" 2>&1 | tail -18
done

echo "=== ann check $(date +%H:%M:%S)"
for rel in inst cat; do
  python3 -u "$S/proj_ann.py" --enc clip_vitl14 --ds inshop --rel "$rel" \
      --w "$O/W_clip_vitl14_${rel}.npz" --out "$O/projann__clip__${rel}.json" 2>&1 | tail -12
done
echo "=== done $(date +%H:%M:%S)"
