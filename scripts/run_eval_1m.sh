#!/bin/bash
# Finish what run_imagenet_pipeline.sh could not: merge the seven encode parts, then prep + budget + eval.
#
# The pipeline script was edited while it was running; bash re-reads a script from a byte offset, so the
# edit shifted the offset and it died on a line that parses fine on its own.  Lesson: edit a *copy* and
# restart, never the file a running bash is streaming.
# --- portable roots: derived from this file, overridable by environment ---
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${SB_ROOT:-$(cd "$HERE/.." && pwd)}"
OUT="${SB_OUT:-$ROOT/outputs/diag}"
CACHE="${SB_CACHE:-$(cd "$ROOT/.." && pwd)/.cache/sb}"
set -u
C=$CACHE
D=$HERE
O=$OUT

echo "[$(date +%H:%M:%S)] merging seven parts"
docker exec mt-cpu python3 $D/merge_emb.py --out $C/imagenet1m \
  --parts $C/imagenet_c0 $C/imagenet_c1 $C/imagenet_c2 $C/imagenet_c3 \
          $C/imagenet_p2 $C/imagenet_p3a $C/imagenet_p3b 2>&1 | tail -12

echo "[$(date +%H:%M:%S)] prep + measured budgets + shared-index comparison"
docker exec mt-cpu python3 $D/imagenet_shared_eval.py --src $C/imagenet1m \
  --tag imagenet1m --queries 500 --budget-queries 200 > $O/imagenet1m_eval.log 2>&1
tail -30 $O/imagenet1m_eval.log
echo "[$(date +%H:%M:%S)] DONE"
