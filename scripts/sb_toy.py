"""Synthetic control for the paper's central qualitative claim.

Claim: two relevance partitions over the SAME data and the SAME principal basis
can require budgets that differ by several times, so no spectrum-only rule can
return both.

This script builds such a corpus from scratch, with no third-party data and no
external files, so the claim can be checked even if every real-corpus derived
file is withheld from the artifact:

  * N images, d dimensions, generated from a fixed seed;
  * the COARSE partition (10 groups) is encoded in the leading directions;
  * the FINE partition (100 groups, nested inside the coarse ones) is encoded in
    the middle directions, with smaller amplitude;
  * both partitions see the same eigenvalues, because there is one covariance.

We then truncate the PCA basis at every rank in a grid, recompute mAP@10 against
the full-dimension mAP@10, and report the smallest rank at which the loss falls
below 5% for each partition. The output is the analogue of Fig. 1: same
spectrum, two budgets.

This is a control, not a result: it shows the mechanism is real and mechanically
checkable. It does not reproduce the paper's numbers, which come from frozen
encoders on real corpora.

Usage:
  python3 sb_toy.py --out ../data/results/toy_synthetic.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

SEED = 20261106
N_FINE, PER_FINE = 100, 20          # 2,000 images, 100 fine groups
COARSE_OF_FINE = 10                 # 10 coarse groups, nested
D = 64
D_CAT = 6                           # directions carrying coarse identity
D_ITEM = 20                         # directions carrying fine identity
AMP_CAT, AMP_ITEM, AMP_NOISE = 3.0, 1.5, 0.45
GRID = [2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64]
EPS = 0.05


def log(m):
    print(m, flush=True)


def build(rng):
    n = N_FINE * PER_FINE
    fine = np.repeat(np.arange(N_FINE), PER_FINE)
    coarse = fine // COARSE_OF_FINE

    cat_dir = rng.standard_normal((COARSE_OF_FINE, D_CAT))
    cat_dir /= np.linalg.norm(cat_dir, axis=1, keepdims=True)
    item_dir = rng.standard_normal((N_FINE, D_ITEM))
    item_dir /= np.linalg.norm(item_dir, axis=1, keepdims=True)

    X = AMP_NOISE * rng.standard_normal((n, D))
    X[:, :D_CAT] += AMP_CAT * cat_dir[coarse]
    X[:, D_CAT:D_CAT + D_ITEM] += AMP_ITEM * item_dir[fine]
    return X.astype(np.float64), fine, coarse


def pca(X):
    mu = X.mean(0)
    Xc = X - mu
    # economy SVD; components are rows, ordered by decreasing singular value
    _, sv, vt = np.linalg.svd(Xc, full_matrices=False)
    lam = (sv ** 2) / max(len(X) - 1, 1)
    return mu, vt, lam


def ap_at_10(dist, labels, q, k=10, exclude_self=True):
    """Average precision at 10 for query index q, relevance = same label."""
    order = np.argsort(dist[q], kind='stable')
    if exclude_self:
        order = order[order != q]
    top = order[:k]
    hit = (labels[top] == labels[q]).astype(float)
    if hit.sum() == 0:
        return 0.0
    ranks = np.arange(1, k + 1)
    prec = np.cumsum(hit) / ranks
    return float((prec * hit).sum() / min(hit.sum(), k))


def mean_ap(Z, labels, k=10):
    # squared euclidean via Gram matrix, one query at a time (n is small)
    sq = (Z ** 2).sum(1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * Z @ Z.T
    return float(np.mean([ap_at_10(d2, labels, q, k) for q in range(len(Z))]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='toy_synthetic.json')
    ap.add_argument('--eps', type=float, default=EPS)
    ap.add_argument('--seed', type=int, default=SEED)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    X, fine, coarse = build(rng)
    mu, vt, lam = pca(X)
    Xc = X - mu

    full = {'fine': mean_ap(Xc, fine), 'coarse': mean_ap(Xc, coarse)}
    log('full-dimension mAP@10: fine %.4f  coarse %.4f' % (full['fine'], full['coarse']))

    curves = {'fine': [], 'coarse': []}
    for r in GRID:
        Z = Xc @ vt[:r].T
        curves['fine'].append(mean_ap(Z, fine))
        curves['coarse'].append(mean_ap(Z, coarse))
        log('r=%3d  fine %.4f  coarse %.4f' % (r, curves['fine'][-1], curves['coarse'][-1]))

    out = {'config': {'seed': args.seed, 'n': int(len(X)), 'd': D,
                      'fine_groups': N_FINE, 'coarse_groups': N_FINE // COARSE_OF_FINE,
                      'd_cat': D_CAT, 'd_item': D_ITEM,
                      'amp_cat': AMP_CAT, 'amp_item': AMP_ITEM, 'amp_noise': AMP_NOISE,
                      'grid': GRID, 'eps': args.eps},
           'top_eigenvalues': [round(float(x), 4) for x in lam[:12]],
           'energy_share_top8': float(lam[:8].sum() / lam.sum()),
           'full_map10': {k: round(v, 4) for k, v in full.items()},
           'map10_by_rank': {k: [round(v, 4) for v in vals] for k, vals in curves.items()},
           'rstar': {}, 'ratio': None}

    for part in ('fine', 'coarse'):
        delta = [1.0 - v / full[part] for v in curves[part]]
        r_star = next((r for r, dl in zip(GRID, delta) if dl <= args.eps), GRID[-1])
        out['rstar'][part] = int(r_star)
        out['delta_by_rank'] = out.get('delta_by_rank', {})
        out['delta_by_rank'][part] = [round(dl, 4) for dl in delta]

    out['ratio'] = round(out['rstar']['fine'] / max(out['rstar']['coarse'], 1), 2)
    out['note'] = ('Synthetic control: both partitions share one covariance, hence one '
                   'spectrum; only the label partition differs. Reproduces the qualitative '
                   'Fig. 1 phenomenon, not its numbers.')

    with open(args.out, 'w') as f:
        json.dump(out, f, indent=1)

    log('\nr* at %.0f%% tolerance: fine=%d  coarse=%d  ratio=%.2fx'
        % (100 * args.eps, out['rstar']['fine'], out['rstar']['coarse'], out['ratio']))
    log('energy in top 8 of %d: %.3f' % (D, out['energy_share_top8']))


if __name__ == '__main__':
    main()
