"""Independent re-evaluation of a learned projection, on the paper's own protocol.

`proj_rank.py` trains W and reports a curve with its own instrument.  A gain that large has to be checked
by a second implementation before it is believed, and the check should share as little code as possible:

  * the query list is the paper's own (`queries.json`), not the trainer's random draw;
  * PCA is refit here at r_max = 640, which is the protocol the budget table uses, so the PCA column of
    this script must reproduce the paper's r_item = 160 (In-shop / CLIP) before anything else is trusted;
  * the reference for the tolerance is mAP at r_max on this same protocol, so both curves are scored
    against one number;
  * queries are split by the trainer's own train/test mask and reported separately, because a projection
    that only helps on the entities it was fitted on is not a result.

Usage:
  python3 proj_eval.py --enc clip_vitl14 --ds inshop --relation inst \
      --w /path/proj_clip_inst.npz
"""
import argparse
import json
import os

import numpy as np

from sb_paths import SB_ROOT, SB_OUT, SB_CACHE, SB_MODELS, REPO  # paths resolve relative to the deposit
DUMP = os.environ.get('DUMP', SB_OUT + '/spectral_budget')
CACHE = os.environ.get('CACHE', SB_CACHE)
RMAX = 640
GRID = [8, 12, 16, 24, 32, 40, 48, 64, 80, 96, 128, 160, 224, 320]


def load(enc, ds):
    if ds in ('imagenet', 'imagenet1m'):
        X = np.asarray(np.load(os.path.join(CACHE, 'imagenet1m', 'emb.npy'), mmap_mode='r'),
                       dtype=np.float32)
        lab = np.load(os.path.join(CACHE, 'imagenet1m', 'labels.npy'))
        meta = json.load(open(os.path.join(CACHE, 'imagenet1m', 'meta.json')))
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from imagenet_coarse import load_groups
        g = load_groups()
        coarse = np.array([g[int(i)] for i in lab])
        rows = [{'ent': int(lab[i]), 'cat': coarse[i]} for i in range(len(lab))]
        return X, rows
    base = os.path.join(DUMP, 'emb', '%s__%s' % (enc, ds))
    meta = json.load(open(base + '.json'))
    if os.path.exists(base + '.npy'):
        emb = np.load(base + '.npy', mmap_mode='r')
    else:
        emb = np.memmap(base + '.bin', dtype=np.float32, mode='r').reshape(meta['n'], meta['dim'])
    rows = [json.loads(l) for l in open(os.path.join(CACHE, ds, 'gallery.jsonl'))]
    return np.asarray(emb, dtype=np.float32), rows


def map_at(Z, r, q, per_row, groups, k=10):
    Zr = Z[:, :r]
    Zr = Zr / np.maximum(np.linalg.norm(Zr, axis=1, keepdims=True), 1e-9)
    aps = []
    for qi in q:
        s = Zr @ Zr[qi]
        s[qi] = -np.inf
        top = np.argpartition(-s, k)[:k]
        top = top[np.argsort(-s[top])]
        rel = groups[per_row[qi]]
        h, ap = 0, 0.0
        for rank, j in enumerate(top, 1):
            if j in rel:
                h += 1
                ap += h / rank
        aps.append(ap / min(len(rel), k))
    return float(np.mean(aps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--enc', default='clip_vitl14')
    ap.add_argument('--ds', default='inshop')
    ap.add_argument('--relation', default='inst')
    ap.add_argument('--w', required=True)
    ap.add_argument('--eps', type=float, default=0.05)
    ap.add_argument('--max-queries', type=int, default=0)
    ap.add_argument('--out', default='')
    a = ap.parse_args()

    X, rows = load(a.enc, a.ds)
    key = 'ent' if a.relation == 'inst' else 'cat'
    per_row = np.array([r[key] for r in rows])
    codes, inv = np.unique(per_row, return_inverse=True)
    per_row = inv
    groups = {}
    for i, v in enumerate(per_row):
        groups.setdefault(v, set()).add(i)

    if a.ds in ('imagenet', 'imagenet1m'):
        # no curated query list for this corpus: draw rows whose class has another member, once, seeded
        cnt = np.bincount(per_row, minlength=int(per_row.max()) + 1)
        elig = np.where(cnt[per_row] > 1)[0]
        rng = np.random.default_rng(0)
        q = np.sort(rng.choice(elig, size=min(300, len(elig)), replace=False))
    else:
        qs = json.load(open(os.path.join(CACHE, a.ds, 'queries.json')))
        qkey = 'inst' if a.relation == 'inst' else a.relation
        q = np.array([int(i) for i in qs.get(qkey, qs.get('inst'))])
        q = q[q < len(rows)]
    if a.max_queries:
        q = q[:a.max_queries]

    nz = np.load(a.w, allow_pickle=True)
    W, V, mu = nz['W'], nz['V'], nz['mu']
    is_tr, is_te = nz['is_tr'], nz['is_te']
    print('%s/%s rel=%s  n=%d  W %s  queries %d' % (a.enc, a.ds, a.relation, len(X), W.shape, len(q)))

    # PCA at the paper's r_max, refit here so the PCA column is an independent reproduction
    Xc = X - mu
    # covariance from the centred matrix in blocks: 52k x 768 fits, 1.28M does not
    mu2 = X.mean(0)
    G = np.zeros((X.shape[1], X.shape[1]), dtype=np.float64)
    for s0 in range(0, len(X), 20000):
        B = X[s0:s0 + 20000] - mu2
        G += B.T.astype(np.float64) @ B
    G /= max(len(X) - 1, 1)
    w, Vp = np.linalg.eigh(G)
    Vp = np.ascontiguousarray(Vp[:, np.argsort(w)[::-1]][:, :RMAX])
    print('pca %d comps refit' % Vp.shape[1])

    res = {'enc': a.enc, 'ds': a.ds, 'relation': a.relation, 'rmax': RMAX,
           'queries': int(len(q)), 'pca': {}, 'w': {}}
    for tag, mask in (('all', np.ones(len(rows), bool)), ('test', is_te), ('train', is_tr)):
        qq = q[mask[q]]
        if len(qq) < 10:
            continue
        print('\n-- %s queries (%d)' % (tag, len(qq)))
        zp = (X - mu2) @ Vp
        ref = map_at(zp, RMAX, qq, per_row, groups)
        pca_cv = {r: map_at(zp, r, qq, per_row, groups) for r in GRID}
        zw = Xc @ W
        w_cv = {r: map_at(zw, r, qq, per_row, groups) for r in GRID if r <= W.shape[1]}
        def rs(cv):
            hit = [r for r in sorted(cv) if 1 - cv[r] / ref <= a.eps]
            return int(hit[0]) if hit else None
        res['pca'][tag] = {'map_rmax': ref, 'curve': pca_cv, 'rstar': rs(pca_cv)}
        res['w'][tag] = {'curve': w_cv, 'rstar': rs(w_cv)}
        print('  ref pca@%d = %.4f   pca r*=%s   W r*=%s' % (RMAX, ref, rs(pca_cv), rs(w_cv)))
        print('  pca ' + ' '.join('%d:%.3f' % (r, pca_cv[r]) for r in GRID))
        print('  W   ' + ' '.join('%d:%.3f' % (r, w_cv[r]) for r in sorted(w_cv)))

    if a.out:
        json.dump(res, open(a.out, 'w'), indent=1, default=str)
        print('wrote', a.out)


if __name__ == '__main__':
    main()
