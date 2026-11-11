#!/bin/bash
# Orchestrate the rest of the ImageNet encode + evaluation on 218, detached from this ssh session.
#
# Why a script and not one command per step: the container restarted once mid-run and took 71 shards of
# work with it.  Everything here is therefore chunked and idempotent -- each step checks for its output
# before doing anything, so re-running the script resumes instead of redoing.
# --- portable roots: derived from this file, overridable by environment ---
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${SB_ROOT:-$(cd "$HERE/.." && pwd)}"
OUT="${SB_OUT:-$ROOT/outputs/diag}"
CACHE="${SB_CACHE:-$(cd "$ROOT/.." && pwd)/.cache/sb}"
set -u
C=$CACHE
D=$HERE
O=$OUT
# Contract: a GPU container with torch + transformers has this directory mounted
# at $MOUNT.  Both the container name and the mount point are variables, so the
# script can be re-pointed without editing.
CONTAINER="${SB_CONTAINER:-imnet-enc-g4}"
MOUNT="${SB_MOUNT:-/work}"
ENC="docker exec -d -e SB_ROOT=$MOUNT -e PYTHONPATH=$MOUNT/.cache/pylibs $CONTAINER bash -lc"
# inside the container the project is at $MOUNT, not at the host path -- passing
# host paths here silently exits with "can't open file" and the step looks like
# it ran
D_IN="$MOUNT/scripts"

log() { echo "[$(date +%H:%M:%S)] $*"; }

wait_dir() {   # wait_dir <dir> <timeout_minutes> <label>
  local d=$1 t=$(( $2 * 3 )) i=0
  while [ ! -f "$d/emb.npy" ]; do
    i=$((i + 1))
    [ $i -gt $t ] && { log "TIMEOUT waiting for $3"; return 1; }
    sleep 20
  done
  log "$3 ready ($(python3 -c "import json;print(json.load(open('$d/meta.json'))['n'])" 2>/dev/null || echo '?') rows)"
}

log "start; waiting for the four chunks of shards 0-99"
for i in 0 1 2 3; do wait_dir "$C/imagenet_c$i" 40 "chunk $i" || exit 1; done

log "launching the last 60 shards (250-309) as two chunks"
$ENC "python3 $D_IN/imagenet_encode.py --shards 30 --shard-start 250 --batch 512 --decode-threads 24 --dtype float16 --out-dir $MOUNT/.cache/sb/imagenet_p3a > $MOUNT/.cache/sb/enc_p3a.log 2>&1"
$ENC "python3 $D_IN/imagenet_encode.py --shards 30 --shard-start 280 --batch 512 --decode-threads 24 --dtype float16 --out-dir $MOUNT/.cache/sb/imagenet_p3b > $MOUNT/.cache/sb/enc_p3b.log 2>&1"

wait_dir "$C/imagenet_p2" 60 "pass 2 (shards 100-249)" || exit 1
wait_dir "$C/imagenet_p3a" 40 "pass 3a" || exit 1
wait_dir "$C/imagenet_p3b" 40 "pass 3b" || exit 1

log "merging seven parts"
docker exec mt-cpu python3 $D/merge_emb.py --out $CACHE/imagenet1m \
  --parts $CACHE/imagenet_c0 $CACHE/imagenet_c1 \
          $CACHE/imagenet_c2 $CACHE/imagenet_c3 \
          $CACHE/imagenet_p2 $CACHE/imagenet_p3a \
          $CACHE/imagenet_p3b >> $CACHE/merge.log 2>&1
tail -3 $CACHE/merge.log

log "prep + measured budgets + shared-index comparison"
docker exec mt-cpu python3 $D/imagenet_shared_eval.py --src $CACHE/imagenet1m \
  --tag imagenet1m --queries 500 --budget-queries 250 > $O/imagenet1m_eval.log 2>&1
tail -25 $O/imagenet1m_eval.log
log "DONE"
