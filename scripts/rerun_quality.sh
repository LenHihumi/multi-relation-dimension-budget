#!/bin/bash
# Regenerate every quality number with self-matches removed from the result lists.
# The bug: an index returns the query as its nearest neighbour, relevance excludes it, so rank 1 was a
# guaranteed non-relevant hit and AP@10 was capped at 0.69 for a four-image group.  Ratios survived but
# absolute numbers did not, so everything quality-related is re-measured here.
# --- portable roots: derived from this file, overridable by environment ---
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${SB_ROOT:-$(cd "$HERE/.." && pwd)}"
OUT="${SB_OUT:-$ROOT/outputs/diag}"
CACHE="${SB_CACHE:-$(cd "$ROOT/.." && pwd)/.cache/sb}"
set -u
D=$HERE
O=$OUT

echo "[$(date +%H:%M:%S)] In-shop, five encoders"
for e in clip_vitl14 dinov3_vitl16 siglip2_so400m wemm_2b qwen3vl_emb_8b; do
  echo "  $e"
  docker exec mt-cpu python3 $D/shared_index.py --enc $e --ds inshop \
    --out $O/shared_index__${e}__inshop.json 2>&1 | tail -5
done

echo "[$(date +%H:%M:%S)] ImageNet 1.28M designs (300 queries)"
docker exec mt-cpu python3 $D/shared_index.py --enc clip_vitl14 --ds imagenet --q 300 \
  --r-item 160 --r-cat 128 --out $O/shared_index__clip_vitl14__imagenet1m.json 2>&1 | tail -6

echo "[$(date +%H:%M:%S)] ImageNet latency / RSS"
docker exec mt-cpu python3 $D/ann_latency.py --queries 300 \
  --out $O/ann_latency__imagenet1m.json 2>&1 | tail -5

echo "[$(date +%H:%M:%S)] candidate recall, CLIP and DINOv3"
docker exec mt-cpu python3 $D/candidate_recall.py --enc clip_vitl14 --r-cat 48 --r-item 160 \
  --out $O/cand_recall__clip_vitl14.json 2>&1 | tail -2
docker exec mt-cpu python3 $D/candidate_recall.py --enc dinov3_vitl16 --r-cat 40 --r-item 320 \
  --out $O/cand_recall__dinov3_vitl16.json 2>&1 | tail -2

echo "[$(date +%H:%M:%S)] collecting tables"
docker exec mt-cpu python3 $D/collect_results.py > /dev/null 2>&1
echo "[$(date +%H:%M:%S)] DONE"
