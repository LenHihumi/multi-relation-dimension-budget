#!/bin/bash
# --- portable roots: derived from this file, overridable by environment ---
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${SB_ROOT:-$(cd "$HERE/.." && pwd)}"
OUT="${SB_OUT:-$ROOT/outputs/diag}"
CACHE="${SB_CACHE:-$(cd "$ROOT/.." && pwd)/.cache/sb}"
OUT=$OUT
S=$HERE/shared_index.py
for enc in siglip2_so400m wemm_2b qwen3vl_emb_8b; do
  f=$OUT/shared_index__${enc}__inshop.json
  if [ -f "$f" ]; then echo "skip $enc"; continue; fi
  echo "=== $enc $(date +%H:%M:%S) ==="
  python3 $S --enc $enc --ds inshop --out $f 2>&1 | tail -6
done
echo ALLDONE $(date +%H:%M:%S)
