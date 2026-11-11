"""Does the label-derived order do causal work, at a fixed rank?

The manuscript's defence against "this is supervised PCA under another name" has been that Fisher and
omega = lambda * eta^2 induce the same *ordering*.  That is a statement about the order, not about the
order's effect.  This is the isolation the reviewers asked for: hold the rank fixed, change only which r
directions are kept, and measure retrieval.

Arms, per (encoder, corpus, relation) cell and rank:

  lam     the r largest eigenvalues                (the traditional spectral rule: a PCA prefix)
  omega   the r largest lambda_i * eta_i^2         (task energy)
  rand    r uniform random directions              (is any subset as good?)
  rand-rho r random directions whose retained eigenvalue share is *matched* to omega's
          (is omega just picking low-variance directions?  this removes that confound)

The matched-random arm samples directions with probability proportional to lambda^{-alpha} and bisects
alpha until the retained share matches the omega set's share to within 1%; the draws use the
Gumbel-top-k trick (Efraimidis-Spirakis) so the sampling is exact and vectorised.

Everything is exact search on the same vectors, same queries, same metric as the published cells.

Usage (inside the container):
  python3 -u sb_order_ablation.py --ranks 32,48,64,96,128,160 --out /path/order_ablation.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sb_retrieval as R  # noqa: E402


def weighted_top(w, r, rng):
    """Weighted sampling without replacement (Gumbel-top-k)."""
    if r >= len(w):
        return np.arange(len(w))
    key = np.log(np.maximum(w, 1e-30)) + rng.gumbel(size=len(w))
    return np.argpartition(-key, r)[:r]


def matched_random(lam, target_retained, r, rng, draws=3, iters=22):
    """Random r-subsets whose retained eigenvalue share matches `target_retained`."""
    tot = lam.sum()
    lo, hi = 0.0, 6.0
    best = []
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        w = lam ** (-mid)
        shares = []
        for _ in range(draws):
            idx = weighted_top(w, r, rng)
            shares.append(lam[idx].sum() / tot)
        m = float(np.mean(shares))
        if m > target_retained:      # too much energy retained -> push index to smaller eigenvalues
            lo = mid
        else:
            hi = mid
    w = lam ** (-0.5 * (lo + hi))
    for _ in range(draws):
        best.append(weighted_top(w, r, rng))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ranks', default='32,48,64,96,128,160')
    ap.add_argument('--out', default=os.path.join(R.OUT, 'order_ablation.json'))
    ap.add_argument('--max-n', type=int, default=0)
    ap.add_argument('--resume', action='store_true')
    a = ap.parse_args()
    ranks = [int(x) for x in a.ranks.split(',')]

    pairs = sorted(glob.glob(os.path.join(R.OUT, 'theory__*.json')))
    out = {'ranks': ranks, 'cells': []}
    t0 = time.time()
    for i, p in enumerate(pairs):
        th = json.load(open(p))
        enc, ds = th['encoder'], th['dataset']
        try:
            X = R.embeddings(enc, ds)
        except Exception as e:
            print('skip %s/%s: %s' % (enc, ds, e))
            continue
        n = len(X)
        if a.max_n:
            n = min(n, a.max_n)
        rows = R.gallery(ds)[:n]
        per_inst, g_inst = R.groups(rows, 'ent')
        per_cat, g_cat = R.groups(rows, 'cat')
        rels = []
        if per_inst is not None:
            rels.append(('inst', per_inst, g_inst))
        if per_cat is not None:
            rels.append(('cat', per_cat, g_cat))
        if not rels:
            print('skip %s/%s: no relevance partition' % (enc, ds))
            continue
        q = R.query_rows(ds, n)
        lam = np.array(th['lambda'], float)
        d = min(len(lam), X.shape[1])
        lam = lam[:d]

        # selections are drawn from a pool much wider than the largest rank we test, otherwise
        # 'pick r of them' degenerates into 'pick everything' at the top of the grid
        width = min(d, max(4 * max(ranks), 512))
        mu, V, _ = R.pca_basis(X[:n], width)
        Z = np.asarray(X[:n], dtype=np.float32) - mu
        Z = np.ascontiguousarray(Z @ V, dtype=np.float32)   # NOT renormalised: prefixes are normalised
        lam = lam[:width]                                   # selections can only use these directions
        print('[%d/%d] %s/%s n=%d d=%d (%.0fs)'
              % (i + 1, len(pairs), enc, ds, n, d, time.time() - t0), flush=True)

        for rel, per_row, grp in rels:
            eta = np.array(th['tasks'][rel]['eta2'], float)[:len(lam)]
            om = lam * eta
            for r in ranks:
                if r > V.shape[1]:
                    continue
                cell = {'enc': enc, 'ds': ds, 'rel': rel, 'r': r, 'n': n}
                sel_lam = np.argsort(-lam)[:r]
                sel_om = np.argsort(-om)[:r]
                tgt = lam[sel_om].sum() / lam.sum()
                cell['retained_lam'] = float(lam[sel_lam].sum() / lam.sum())
                cell['retained_om'] = float(tgt)
                cell['overlap'] = int(len(set(sel_lam.tolist()) & set(sel_om.tolist())))

                def score(sel):
                    A = np.ascontiguousarray(Z[:, sel])
                    return R.map10(R.normalise(A), q, per_row, grp)

                cell['map_lam'] = score(sel_lam)
                cell['map_om'] = score(sel_om)
                rng = np.random.default_rng(0)
                cell['map_rand'] = float(np.mean([
                    score(np.sort(rng.choice(width, size=r, replace=False)))
                    for _ in range(3)]))
                sets = matched_random(lam, tgt, r, rng, draws=3)
                cell['map_rand_rho'] = float(np.mean([score(np.sort(s)) for s in sets]))
                cell['rand_rho_retained'] = float(np.mean(
                    [lam[np.sort(s)].sum() / lam.sum() for s in sets]))
                out['cells'].append(cell)
                print('   %-4s r=%-3d lam %.4f  om %.4f  rand %.4f  rand-rho %.4f (kept %.3f)'
                      % (rel, r, cell['map_lam'], cell['map_om'], cell['map_rand'],
                         cell['map_rand_rho'], cell['rand_rho_retained']), flush=True)

    json.dump(out, open(a.out, 'w'), indent=1)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
