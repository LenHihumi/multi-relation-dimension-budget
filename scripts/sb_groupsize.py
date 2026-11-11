"""How does the budget move when the partition changes shape, or is noisy?

Sec. VI-D tells a deployer to "check the group count before spending a labelling budget",
and the review asks for the curve behind that sentence: what happens to eta^2 and to the
5%-tolerance budget as the number of groups sweeps a realistic range, and what happens when
a fraction of the labels are wrong. Neither exists in the released data, because both need
re-deriving eta^2 for a partition that is not the corpus's own.

For one cell this script:
  1. rebuilds the PCA basis and the projected scores (same code as sb_quant_budget.py);
  2. computes eta^2 per direction for a partition, then omega = lambda * eta^2, then the
     rank that keeps the label loss under 5% by exact search;
  3. repeats for the real partition, for random partitions at several group counts whose
     sizes are drawn from the real group-size distribution, and for label noise at several
     rates.

Output: <out>  (JSON, one record per variant)

Run inside a container:
  docker run --rm --entrypoint python3 -e SB_CACHE=/work/.cache/sb -e SB_OUT=/work/outputs/diag \\
    -v "$PWD":/work -w /work vllm/vllm-openai:v0.31.0 sb_groupsize.py ...
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import sb_retrieval as R  # noqa: E402

GRID = (8, 12, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 258, 320,
        384, 448, 512, 640, 768, 896, 1075, 1280, 1536, 1792, 2048, 3072, 4096)
EPS = 0.05


def log(m):
    print('[%s] %s' % (time.strftime('%H:%M:%S'), m), flush=True)


def eta2(Z, labels):
    """Between-group share of each direction's variance."""
    uniq, inv = np.unique(labels, return_inverse=True)
    n = len(inv)
    gmean = np.zeros((len(uniq), Z.shape[1]))
    np.add.at(gmean, inv, Z)
    counts = np.bincount(inv, minlength=len(uniq)).astype(float)
    gmean /= counts[:, None]
    overall = Z.mean(0)
    between = (counts[:, None] * (gmean - overall) ** 2).sum(0)
    total = ((Z - overall) ** 2).sum(0)
    return between / np.maximum(total, 1e-12), counts


def random_partition(sizes, n, rng):
    """Assign n rows to groups whose sizes are drawn from the empirical distribution."""
    out = np.empty(n, int)
    i, g = 0, 0
    while i < n:
        s = int(min(sizes[rng.integers(len(sizes))], n - i))
        out[i:i + s] = g
        i += s
        g += 1
    return out


def noisy(labels, frac, rng):
    out = labels.copy()
    uniq = np.unique(labels)
    k = int(frac * len(out))
    idx = rng.choice(len(out), size=k, replace=False)
    out[idx] = uniq[rng.integers(len(uniq), size=k)]
    return out


def budget_for(Z, q, labels, ranks, order_mode='lambda'):
    lam_all = None
    eta, counts = eta2(Z, labels)
    # omega = lambda * eta^2, with lambda from the projected energy
    lam = (Z ** 2).sum(0)
    omega = lam * eta
    # The paper's budgets are PCA prefixes (lambda order); omega order is the
    # label-weighted variant. They give different ranks, so the mode is explicit.
    order = np.arange(Z.shape[1]) if order_mode == 'lambda' else np.argsort(-omega)
    per_row, grp = None, None
    # relevance sets from the partition
    g = {}
    for i, v in enumerate(labels):
        g.setdefault(int(v), []).append(i)
    relsets = [set(g[int(labels[qi])]) - {int(qi)} for qi in q]

    Zf = Z
    D = Zf[q] @ Zf.T
    np.put_along_axis(D, np.repeat(q[:, None], D.shape[1], 1), -np.inf, axis=1)
    topf = np.argsort(-D, axis=1)[:, :10]

    def map10(top):
        tot = 0.0
        for row, rs in zip(top, relsets):
            if not rs:
                continue
            hits, cum = 0, 0.0
            for j, dd in enumerate(row):
                if dd in rs:
                    hits += 1
                    cum += hits / (j + 1)
            tot += cum / min(len(rs), 10)
        return tot / max(len(relsets), 1)

    full = map10(topf)
    r_star = None
    for r in ranks:
        A = order[:r]
        Zr = R.normalise(np.ascontiguousarray(Z[:, A]))
        D2 = Zr[q] @ Zr.T
        np.put_along_axis(D2, np.repeat(q[:, None], D2.shape[1], 1), -np.inf, axis=1)
        m = map10(np.argsort(-D2, axis=1)[:, :10])
        if 1 - m / full <= EPS:
            r_star = r
            break
    return {'n_groups': int(len(counts)), 'r_star': r_star,
            'eta2_mean': float(eta.mean()), 'eta2_max': float(eta.max()),
            'group_size_median': float(np.median(counts)),
            'full_map10': float(full)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--enc', required=True)
    ap.add_argument('--ds', required=True)
    ap.add_argument('--rel', default='inst')
    ap.add_argument('--out', required=True)
    ap.add_argument('--nq', type=int, default=1000)
    ap.add_argument('--order', choices=['lambda', 'omega'], default='lambda')
    a = ap.parse_args()

    X = R.embeddings(a.enc, a.ds)
    n, d = X.shape
    rows = R.gallery(a.ds)
    key = 'ent' if a.rel == 'inst' else 'cat'
    raw = [r.get(key) for r in rows[:n]]
    keep = np.array([i for i, v in enumerate(raw) if v is not None], int)
    X = X[keep]
    _, labels = np.unique([str(raw[i]) for i in keep], return_inverse=True)
    n = len(labels)
    rng = np.random.default_rng(0)
    q = R.query_rows(a.ds, n)

    mu, V, w = R.pca_basis(X, min(d, 768))
    Z = np.ascontiguousarray((X - mu) @ V, dtype=np.float32)
    ranks = [r for r in GRID if r <= Z.shape[1]]
    sizes = np.bincount(np.unique(labels, return_inverse=True)[1])
    log('%s/%s n=%d dim=%d groups=%d median size %.1f'
        % (a.enc, a.ds, n, Z.shape[1], len(sizes), float(np.median(sizes))))

    out = {'enc': a.enc, 'ds': a.ds, 'rel': a.rel, 'n': n, 'order': a.order,
           'variants': []}
    variants = [('real', labels)]
    for ng in (8, 32, 128, 512, 2048):
        variants.append(('random-%d' % ng, None))
    for frac in (0.05, 0.10, 0.25):
        variants.append(('noise-%02d' % int(frac * 100), noisy(labels, frac, np.random.default_rng(1))))

    for name, lab in variants:
        if lab is None:
            ng = int(name.split('-')[1])
            lab = None
            # draw sizes so that the expected number of groups is ng
            target = max(1, n // ng)
            sz = np.maximum(1, np.round(sizes * (target / np.median(sizes))).astype(int))
            lab = random_partition(sz, n, np.random.default_rng(7))
        s = budget_for(Z, q, lab, ranks, a.order)
        s['variant'] = name
        out['variants'].append(s)
        log('  %-12s groups=%-5d r*=%-5s eta2_mean %.4f' %
            (name, s['n_groups'], s['r_star'], s['eta2_mean']))

    json.dump(out, open(a.out, 'w'), indent=1)
    log('wrote ' + a.out)


if __name__ == '__main__':
    main()
