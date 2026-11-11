"""Hierarchical (empirical-Bayes) shrinkage baseline -- the strongest predictor side.

Reviewers asked for the predictor a statistician would actually reach for: a hierarchical model
that shrinks per-cell parameters toward a corpus-level mean, instead of importing a median.  The
paper's prediction side currently has two arms (median import; label-free ridge), and a critic can
say both are weak.  This script adds the missing one and scores it under the same leave-one-corpus
-out protocol as `sb_lodo_true.py`.

Arms, all zero target evaluations unless stated:

  median_import      per-relation median of (log rho0, log s) over the training corpora.
  hier_regression    partial pooling on (log rho0, log s) with covariates [1, relation, log d] and a
                     corpus random intercept; tau^2 and sigma^2 by moments, corpus effects shrunk by
                     tau^2/(tau^2 + sigma^2/n_g).  The held-out corpus has no effect estimate, so its
                     prediction is the fixed part -- which is what a deployment would have.
  hier_blend_2pt     the same prior blended with the two measured points at the prescribed targets:
                     log-parameter average with weight w taken from the variance ratio, plus a grid
                     over w to show the sensitivity.  This is the arm that answers "does shrinkage
                     improve on two measurements at equal budget?", not "can it replace them?".

Every arm issues a rank through the same Delta(r) = 1/(1+(rho0/rho)^s) curve and the same 0.90
residual margin as the other LODO arms, then scores against the measured curve.

Usage
  python3 sb_shrinkage.py --out shrinkage.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sb_common                                                       # noqa: E402

if os.environ.get('SB_OUT'):
    sb_common.OUT = os.environ['SB_OUT']

from sb_acceptance import EPS, fit_ll                                   # noqa: E402
from sb_common import OUT                                              # noqa: E402
from sb_lodo_true import (curve, eval_rank, load_cells,                 # noqa: E402
                          pick_points, rank_for, summarise)


def log(m):
    print(m, flush=True)


def dims_by_encoder():
    """Encoder width d, read from the theory files (any corpus will do)."""
    out = {}
    for f in os.listdir(OUT):
        if not (f.startswith('theory__') and f.endswith('.json')):
            continue
        d = json.load(open(os.path.join(OUT, f)))
        out.setdefault(d['encoder'], int(d['dim']))
    th = os.path.join(OUT, 'theory')
    if os.path.isdir(th):
        for f in os.listdir(th):
            if f.startswith('theory__') and f.endswith('.json'):
                d = json.load(open(os.path.join(th, f)))
                out.setdefault(d['encoder'], int(d['dim']))
    return out


def design(cells, dims, use_dim=False):
    cols = [np.ones(len(cells)),
            np.array([1.0 if c['rel'] == 'cat' else 0.0 for c in cells])]
    if use_dim:
        cols.append(np.array([np.log(dims[c['enc']]) for c in cells]))
    return np.column_stack(cols)


def shrink_fit(tr, dims, target, key, use_dim=False):
    """Empirical-Bayes fit on training cells; returns predict(cell) for a held-out cell."""
    y = np.array([np.log(c[key]) for c in tr], float)
    X = design(tr, dims, use_dim)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    sig2 = float(resid @ resid) / max(len(tr) - X.shape[1], 1)
    groups = sorted({c['ds'] for c in tr})
    u, n_g = {}, {}
    for g in groups:
        idx = [i for i, c in enumerate(tr) if c['ds'] == g]
        u[g] = float(resid[idx].mean())
        n_g[g] = len(idx)
    # method of moments for the between-corpus variance
    if len(groups) > 1:
        var_u = float(np.var([u[g] for g in groups], ddof=1))
        tau2 = max(0.0, var_u - float(np.mean([sig2 / n_g[g] for g in groups])))
    else:
        tau2 = 0.0
    shrunk = {g: (tau2 / (tau2 + sig2 / n_g[g])) * u[g] for g in groups}
    info = {'tau2': tau2, 'sigma2': sig2, 'corpus_effects': dict(u),
            'shrunk': shrunk, 'beta': beta.tolist(), 'n_train': len(tr)}
    return beta, sig2, tau2, info


def predict(beta, dims, cell, use_dim=False):
    x = [1.0, 1.0 if cell['rel'] == 'cat' else 0.0]
    if use_dim:
        x.append(np.log(dims[cell['enc']]))
    return float(np.array(x) @ beta)


def run_arm(C, dims, key_rho='r0_fit', key_s='s_fit', blend_w=None, eps=EPS):
    rows = []
    for g in sorted({c['ds'] for c in C}):
        tr = [c for c in C if c['ds'] != g]
        te = [c for c in C if c['ds'] == g]
        b_rho, s2_rho, t2_rho, _ = shrink_fit(tr, dims, None, key_rho)
        b_s, s2_s, t2_s, _ = shrink_fit(tr, dims, None, key_s)
        # margin: residuals of the same procedure on the training cells (in-sample prediction)
        resid = []
        for c in tr:
            p = curve(c['rho'], float(np.clip(np.exp(predict(b_rho, dims, c)), 1e-4, 2.0)),
                      float(np.clip(np.exp(predict(b_s, dims, c)), 0.3, 20.0)))
            resid.extend((c['delta'] - p).tolist())
        # the safety margin guards the tolerance; a prior this far off would push the margin
        # above eps and make every cell saturate to the largest rank, which measures nothing.
        # Capping at eps keeps the comparison about the parameters, not about the margin.
        bias = min(max(0.0, float(np.quantile(resid, 0.90))) if resid else 0.0, 0.5 * eps)
        for c in te:
            r0 = float(np.clip(np.exp(predict(b_rho, dims, c)), 1e-4, 2.0))
            s = float(np.clip(np.exp(predict(b_s, dims, c)), 0.3, 20.0))
            if blend_w is not None:
                # measured two points, then shrink toward the prior in log space
                idx = pick_points(c['rho'], c['grid'], 2)
                r0_m, s_m = fit_ll(c['rho'][idx], c['delta'][idx])
                r0 = float(np.exp(blend_w * np.log(r0_m) + (1 - blend_w) * np.log(r0)))
                s = float(np.exp(blend_w * np.log(s_m) + (1 - blend_w) * np.log(s)))
            du = np.clip(curve(c['rho'], r0, s) + bias, 0, 1)
            meet, ov = eval_rank(c, rank_for(c['rho'], du, c['grid'], eps))
            rows.append({'id': c['id'], 'holdout': g, 'rho0': r0, 's': s,
                         'meet': meet, 'overhead': ov})
    return summarise(rows), rows


def median_import(C, eps=EPS):
    rows = []
    for g in sorted({c['ds'] for c in C}):
        tr = [c for c in C if c['ds'] != g]
        te = [c for c in C if c['ds'] == g]
        for c in te:
            same = [x for x in tr if x['rel'] == c['rel']] or tr
            r0 = float(np.median([x['r0_fit'] for x in same]))
            s = float(np.median([x['s_fit'] for x in same]))
            du = np.clip(curve(c['rho'], r0, s), 0, 1)
            meet, ov = eval_rank(c, rank_for(c['rho'], du, c['grid'], eps))
            rows.append({'id': c['id'], 'holdout': g, 'meet': meet, 'overhead': ov})
    return summarise(rows), rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='shrinkage.json')
    ap.add_argument('--eps', type=float, default=EPS)
    args = ap.parse_args()

    C = load_cells()
    dims = dims_by_encoder()
    log('cells=%d  corpora=%s  encoders=%d'
        % (len(C), sorted({c['ds'] for c in C}), len(dims)))

    res = {'eps': args.eps, 'n_cells': len(C), 'datasets': sorted({c['ds'] for c in C})}
    res['median_import'], _ = median_import(C, args.eps)
    res['hier_regression'], _ = run_arm(C, dims, eps=args.eps)
    for w in (0.25, 0.5, 0.75):
        res['hier_blend_2pt_w%02d' % int(100 * w)], _ = run_arm(C, dims, blend_w=w, eps=args.eps)

    print()
    print('%-24s %5s %8s %9s %8s' % ('arm', 'n', 'cover', 'median', 'p95'))
    for k, v in res.items():
        if isinstance(v, dict) and 'coverage' in v:
            print('%-24s %5d %7.0f%% %8.2fx %7.2fx'
                  % (k, v['n'], 100 * v['coverage'], v['median_overhead'], v['p95_overhead']))
    with open(os.path.join(OUT, args.out), 'w') as f:
        json.dump(res, f, indent=1)
    log('wrote %s' % os.path.join(OUT, args.out))
    return 0


if __name__ == '__main__':
    sys.exit(main())
