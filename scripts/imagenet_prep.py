"""Prepare the ImageNet cache for `shared_index.py` (uniform layout, no changes to that script).

`imagenet_encode.py` writes `emb.npy / labels.npy / meta.json` into one directory.  `shared_index.py`
expects the layout the paper's caches already use:

    $DUMP/emb/<enc>__<ds>.npy      embeddings
    $DUMP/emb/<enc>__<ds>.json     {"n": ..., "dim": ...}
    $CACHE/<ds>/gallery.jsonl      one {"ent": ..., "cat": ...} per row
    $CACHE/<ds>/queries.json       {"inst": [row ids]}

Making ImageNet fit that layout (instead of adding a second code path) means the ImageNet numbers and
the In-shop numbers come out of the *same* code, which is the only way they are comparable.

Relations: `ent` = the ImageNet class (1000 groups, fine), `cat` = the keyword group from
`imagenet_encode.coarse_of` (coarse).  Both are label-only, no query annotation, so the same 2,000
query rows serve both relations and only the relevance set differs -- same construction as In-shop.

Read-only w.r.t. the embeddings; writes only the sidecar, the jsonl and the queries file.
"""
import argparse
import json
import os
import sys

import numpy as np


from sb_paths import SB_ROOT, SB_OUT, SB_CACHE, SB_MODELS, REPO  # paths resolve relative to the deposit
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=SB_CACHE + '/imagenet')
    ap.add_argument('--dump', default=REPO + '/outputs/diag/spectral_budget')
    ap.add_argument('--cache', default=SB_CACHE)
    ap.add_argument('--enc', default='clip_vitl14')
    ap.add_argument('--ds', default='imagenet')
    ap.add_argument('--queries', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    # WordNet-hierarchy groups (depth 6), built once by imagenet_coarse.py; keyword matching on class
    # names was abandoned because "office" contains "ice" and "seacoast" contains "sea"
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from imagenet_coarse import load_groups

    meta = json.load(open(os.path.join(a.src, 'meta.json')))
    lab = np.load(os.path.join(a.src, 'labels.npy'))
    classes = meta['classes']
    n = int(len(lab))
    print('rows %d, classes %d' % (n, len(classes)))

    cdir = os.path.join(a.cache, a.ds)
    os.makedirs(cdir, exist_ok=True)
    idx2group = load_groups()
    coarse_names = [idx2group[int(i)] for i in lab]
    with open(os.path.join(cdir, 'gallery.jsonl'), 'w') as f:
        for i in range(n):
            f.write(json.dumps({'ent': int(lab[i]), 'cat': coarse_names[i]}) + '\n')

    # one image per row, so a query needs at least one same-class partner to be scoreable
    counts = np.bincount(lab, minlength=len(classes))
    eligible = np.where(counts[lab] > 1)[0]
    rng = np.random.default_rng(a.seed)
    q = np.sort(rng.choice(eligible, size=min(a.queries, len(eligible)), replace=False))
    json.dump({'inst': [int(i) for i in q], 'cat': [int(i) for i in q]},
              open(os.path.join(cdir, 'queries.json'), 'w'))
    print('queries %d (eligible %d), coarse groups %d'
          % (len(q), len(eligible), len(set(coarse_names))))

    edir = os.path.join(a.dump, 'emb')
    os.makedirs(edir, exist_ok=True)
    base = os.path.join(edir, '%s__%s' % (a.enc, a.ds))
    src = os.path.join(a.src, 'emb.npy')
    if not os.path.exists(base + '.npy'):
        os.symlink(src, base + '.npy')
    json.dump({'n': n, 'dim': int(meta['dim'])}, open(base + '.json', 'w'))
    print('wrote', base + '.json', 'and linked', base + '.npy')

    top = sorted(set(coarse_names))
    print('coarse groups:', top[:10], '...' if len(top) > 10 else '')
    print('largest classes:', counts.argsort()[-5:][::-1].tolist())


if __name__ == '__main__':
    main()
