"""Measure the ImageNet budgets instead of importing In-shop's.

The whole point of the paper is that a budget is a property of (spectrum, partition) and must be
measured, so the ImageNet half of the shared-index study cannot borrow 48/160 from CLIP/In-shop.  This
runs the paper's own protocol on the ImageNet embedding: nested PCA prefixes over a rank grid, mAP@10
per prefix, then the bounded log-logistic fit and the 5%-tolerance crossing.

Two relations, both from labels only:
  inst  the ImageNet class (1000 groups, fine)
  cat   the keyword group from `imagenet_encode.coarse_of` (coarse)

Cost is exact search, not ANN: one projection matrix for the whole corpus (the prefix family is nested,
so prefixes are slices), then `nq x n` dot products per rank.  At 1M rows and 500 queries that is a few
minutes on CPU, which is why the grid is 12 points rather than the paper's 23.
"""
import argparse
import json
import os
import time

import numpy as np

from sb_paths import SB_ROOT, SB_OUT, SB_CACHE, SB_MODELS, REPO  # paths resolve relative to the deposit
RANKS = [8, 12, 16, 24, 32, 48, 64, 96, 128, 160, 224, 320, 448, 640]


def pca_basis(X, k):
    mu = X.mean(0)
    Xc = X - mu
    C = (Xc.T @ Xc) / (len(Xc) - 1)
    w, V = np.linalg.eigh(C)
    o = np.argsort(w)[::-1]
    return mu, V[:, o][:, :k], w[o]


def map_at_10(Z, r, qidx, per_row, groups, k=10):
    """Exact top-k over the first r columns; AP@10 with the same relevance rule as the paper.

    Slicing inside keeps the prefix family nested (one projection, prefixes as views) instead of
    re-projecting the corpus per rank.
    """
    aps = []
    Zr = Z[:, :r]
    for qi in qidx:
        rel = groups[per_row[qi]]
        s = Zr @ Zr[qi]
        s[qi] = -np.inf
        top = np.argpartition(-s, k)[:k]
        top = top[np.argsort(-s[top])]
        h, ap = 0, 0.0
        for rank, j in enumerate(top, 1):
            if j in rel:
                h += 1
                ap += h / rank
        aps.append(ap / min(len(rel), k))
    return float(np.mean(aps))


def loglog(rho, r0, s):
    return 1.0 / (1.0 + (r0 / np.clip(rho, 1e-6, None)) ** s)


def fit_grid(rho, dlt):
    """Least-squares (r0, s) on a grid; returns (r0, s) and prints the residual."""
    best = (None, None, np.inf)
    for r0 in np.logspace(-3, 0.3, 120):
        for s in np.linspace(0.5, 15.0, 120):
            err = float(np.sum((loglog(rho, r0, s) - dlt) ** 2))
            if err < best[2]:
                best = (float(r0), float(s), err)
    r0, s, err = best
    r2 = 1.0 - err / max(float(np.sum((dlt - dlt.mean()) ** 2)), 1e-12)
    print('  fit r0=%.4f s=%.2f  R2=%.3f' % (r0, s, r2))
    return r0, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=SB_CACHE + '/imagenet1m')
    ap.add_argument('--n', type=int, default=0)
    ap.add_argument('--queries', type=int, default=400)
    ap.add_argument('--eps', type=float, default=0.05)
    ap.add_argument('--out', default='')
    a = ap.parse_args()

    emb = np.load(os.path.join(a.src, 'emb.npy'), mmap_mode='r')
    lab = np.load(os.path.join(a.src, 'labels.npy'))
    meta = json.load(open(os.path.join(a.src, 'meta.json')))
    n = min(len(emb), a.n) if a.n else len(emb)
    X = np.asarray(emb[:n], dtype=np.float32)
    lab = lab[:n]
    print('rows %d, dim %d' % (n, X.shape[1]))

    # coarse label per row, from the same rule the encoder used
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from imagenet_coarse import load_groups
    classes = meta['classes']
    idx2group = load_groups()
    coarse = np.array([idx2group[int(i)] for i in lab])
    rel = {'inst': lab, 'cat': coarse}

    t0 = time.time()
    k = max(RANKS)
    mu, V, w = pca_basis(X, k)
    print('pca %d comps in %.1fs, top-%d energy %.3f'
          % (k, time.time() - t0, RANKS[-1],
             float(w[:RANKS[-1]].sum() / w.sum())))
    Z = (X - mu) @ V
    Z /= np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9)
    del X

    rng = np.random.default_rng(0)
    counts = np.bincount(lab, minlength=len(classes))
    elig = np.where(counts[lab] > 1)[0]
    qidx = np.sort(rng.choice(elig, size=min(a.queries, len(elig)), replace=False))
    groups = {}
    for key in ('inst', 'cat'):
        g = {}
        for i, v in enumerate(rel[key]):
            g.setdefault(v, set()).add(i)
        groups[key] = g
    print('queries %d; groups inst=%d cat=%d'
          % (len(qidx), len(groups['inst']), len(groups['cat'])))

    out = {'n': n, 'queries': len(qidx), 'eps': a.eps, 'ranks': [], 'relations': {}}
    curves = {}
    for key in ('inst', 'cat'):
        rows = []
        for r in RANKS:
            t = time.time()
            m = map_at_10(Z, r, qidx, rel[key], groups[key])
            rho = float(w[:r].sum() / w.sum())          # kept energy fraction at this rank
            rows.append({'r': r, 'rho': rho, 'map10': m, 'sec': round(time.time() - t, 1)})
            print('  %s r=%3d  rho=%.3f  mAP@10=%.4f  (%.1fs)' % (key, r, rho, m, time.time() - t))
        full = [x['map10'] for x in rows if x['r'] == RANKS[-1]][0]
        for x in rows:
            x['delta'] = 1.0 - x['map10'] / full        # loss relative to the largest prefix
        out['relations'][key] = {'curve': rows, 'map_full': full}
        curves[key] = (np.array([x['rho'] for x in rows]), np.array([x['delta'] for x in rows]))

    # budget = first rank whose loss is within eps (measured curve), plus a two-parameter fit for
    # reference.  The fit is a coarse grid search rather than scipy: mt-cpu has no scipy, and with 14
    # points the objective is smooth enough that a grid is exact to well under a grid step.
    for key, (rho, dlt) in curves.items():
        r0, s = fit_grid(rho, dlt)
        hit = [x['r'] for x in out['relations'][key]['curve'] if x['delta'] <= a.eps]
        r_star = int(hit[0]) if hit else int(RANKS[-1])
        out['relations'][key].update({'r0': r0, 's': s, 'r_star': r_star})
        print('%s: r0=%.3f s=%.2f -> r*(eps=%.2f) = %d' % (key, r0, s, a.eps, r_star))

    if a.out:
        json.dump(out, open(a.out, 'w'), indent=1)
        print('wrote', a.out)


if __name__ == '__main__':
    main()
