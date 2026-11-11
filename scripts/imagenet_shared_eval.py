"""One command: ImageNet cache -> measured budgets -> shared-index comparison.

Three steps that must run in this order and share paths, so they are driven from here instead of being
typed by hand three times:

  1. `imagenet_prep.py`   uniform cache layout (gallery.jsonl, queries.json, embedding sidecar)
  2. `imagenet_budget.py` the paper's protocol on this corpus -> r* for the fine and coarse partitions
  3. `shared_index.py`    uniform-max vs share-f32 vs share-i8 at those measured budgets

Budgets are measured here, not imported: a budget is a property of (spectrum, partition), which is the
paper's own thesis, so borrowing In-shop's 48/160 would make the ImageNet result circular.
"""
import argparse
import json
import os
import re
import subprocess
import sys

from sb_paths import SB_ROOT, SB_OUT, SB_CACHE, SB_MODELS, REPO  # paths resolve relative to the deposit
HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
ROOT = SB_ROOT
DIAG = os.path.join(ROOT, 'projects/ImageSearchAgent/scripts/diag')
DUMP = os.path.join(ROOT, 'projects/ImageSearchAgent/outputs/diag/spectral_budget')
CACHE = os.path.join(ROOT, '.cache/sb')


def run(cmd):
    print('$ ' + ' '.join(cmd), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(r.stdout[-4000:])
    if r.returncode:
        sys.stderr.write(r.stderr[-3000:])
        raise SystemExit('step failed: %s' % cmd[1])
    return r.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=os.path.join(ROOT, '.cache/sb/imagenet1m'))
    ap.add_argument('--queries', type=int, default=300)
    ap.add_argument('--n', type=int, default=0, help='0 = all rows')
    ap.add_argument('--budget-queries', type=int, default=200)
    ap.add_argument('--tag', default='imagenet1m')
    ap.add_argument('--skip-budget', action='store_true')
    ap.add_argument('--merge-parts', nargs='*', default=[],
                    help='encode pass directories to concatenate into --src first')
    a = ap.parse_args()

    if a.merge_parts:
        run([PY, os.path.join(DIAG, 'merge_emb.py'), '--out', a.src,
             '--parts'] + list(a.merge_parts))

    out_budget = os.path.join(ROOT, 'projects/ImageSearchAgent/outputs/diag',
                              'imagenet_budget__%s.json' % a.tag)
    run([PY, os.path.join(DIAG, 'imagenet_prep.py'), '--src', a.src, '--dump', DUMP,
         '--cache', CACHE, '--ds', 'imagenet', '--queries', str(a.queries)])

    if a.skip_budget and os.path.exists(out_budget):
        print('reusing', out_budget)
    else:
        run([PY, os.path.join(DIAG, 'imagenet_budget.py'), '--src', a.src, '--n', str(a.n),
             '--queries', str(a.budget_queries), '--out', out_budget])
    b = json.load(open(out_budget))
    r_item = int(b['relations']['inst']['r_star'])
    r_cat = int(b['relations']['cat']['r_star'])
    print('measured budgets: r_inst=%d  r_cat=%d' % (r_item, r_cat))

    out_shared = os.path.join(ROOT, 'projects/ImageSearchAgent/outputs/diag',
                              'shared_index__clip_vitl14__%s.json' % a.tag)
    run([PY, os.path.join(DIAG, 'shared_index.py'), '--enc', 'clip_vitl14', '--ds', 'imagenet',
         '--n', str(a.n), '--q', str(a.queries), '--r-item', str(r_item), '--r-cat', str(r_cat),
         '--out', out_shared])
    print('wrote', out_shared)


if __name__ == '__main__':
    main()
