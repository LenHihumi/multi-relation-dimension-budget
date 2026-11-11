"""Can the steepness s be predicted, or only measured?

Reviewers asked for one of two things: a mechanism for the log-logistic shape, or a model that
predicts s across cells with R^2 >= 0.5 under leave-one-corpus-out.  The paper currently claims
neither ("we claim no mechanism").  This script tests the second option, which needs no new
retrieval: s per cell is already fitted, and the candidate features are all label-free or cheap.

Features (all from theory__*.json, no target queries):
  log d            encoder width
  relation         item vs category partition
  log lam1/lam_r   how fast the spectrum decays over the first 32 directions
  mean eta^2       label concentration of the retained directions
  log groups       partition granularity
  log d_eff(d)     participation ratio at full dimension

Model: ridge on standardised features, alpha by inner 3-fold CV on the training corpora, scored
leave-one-corpus-out.  Reports held-out R^2 for log s, and the same for log rho0 as a sanity
check (the paper's level is supposed to be explainable, so a model that cannot predict rho0
either would be measuring the absence of any signal).

Usage
  python3 sb_shape_predict.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sb_common                                                       # noqa: E402

if os.environ.get('SB_OUT'):
    sb_common.OUT = os.environ['SB_OUT']

from sb_common import OUT                                              # noqa: E402
from sb_lodo_true import load_cells                                    # noqa: E402


def theory():
    th = {}
    for pat in ('theory__*.json', os.path.join('theory', 'theory__*.json')):
        for p in _glob(os.path.join(OUT, pat)):
            d = json.load(open(p))
            th[(d['encoder'], d['dataset'])] = d
    return th


def _glob(p):
    import glob
    return glob.glob(p)


def feats(c, th):
    t = th[(c['enc'], c['ds'])]
    lam = np.asarray(t['lambda'], float)
    eta = np.asarray(t['tasks'][c['rel']]['eta2'], float)
    n = min(len(lam), len(eta), 32)
    w = lam[:n] * eta[:n] ** 2
    deff = (w.sum() ** 2) / max(float((w ** 2).sum()), 1e-30)
    return [np.log(t['dim']),
            1.0 if c['rel'] == 'cat' else 0.0,
            float(np.log(lam[0] / max(lam[min(31, len(lam) - 1)], 1e-12))),
            float(eta[:n].mean()),
            float(np.log(t['tasks'][c['rel']]['n_groups'])),
            float(np.log(max(deff, 1.0)))]


def ridge_fit(X, y, alpha):
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-9] = 1.0
    Z = (X - mu) / sd
    Zc = np.column_stack([np.ones(len(Z)), Z])
    A = Zc.T @ Zc + alpha * np.eye(Zc.shape[1])
    A[0, 0] -= alpha
    beta = np.linalg.solve(A, Zc.T @ y)
    return mu, sd, beta


def ridge_pred(m, X):
    mu, sd, beta = m
    Z = (X - mu) / sd
    return np.column_stack([np.ones(len(Z)), Z]) @ beta


def cv_alpha(X, y, grid=(0.01, 0.1, 1.0, 10.0)):
    best, best_sse = grid[0], np.inf
    for a in grid:
        sse = 0.0
        for i in range(len(y)):
            m = np.ones(len(y), bool)
            m[i] = False
            mm = ridge_fit(X[m], y[m], a)
            sse += float((y[i] - ridge_pred(mm, X[~m])[0]) ** 2)
        if sse < best_sse:
            best, best_sse = a, sse
    return best


def run(C, th, key):
    y = np.array([np.log(c[key]) for c in C], float)
    X = np.array([feats(c, th) for c in C], float)
    pred = np.zeros_like(y)
    folds = {}
    for g in sorted({c['ds'] for c in C}):
        tr = np.array([c['ds'] != g for c in C])
        te = ~tr
        a = cv_alpha(X[tr], y[tr])
        m = ridge_fit(X[tr], y[tr], a)
        pred[te] = ridge_pred(m, X[te])
        folds[g] = {'n': int(te.sum()), 'alpha': a,
                    'r2': float(1 - ((y[te] - pred[te]) ** 2).sum()
                                / max(((y[te] - y[tr].mean()) ** 2).sum(), 1e-12))}
    r2 = float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    rho = float(np.corrcoef(np.argsort(np.argsort(y)), np.argsort(np.argsort(pred)))[0, 1])
    return {'key': key, 'n': len(C), 'loo_r2': r2, 'spearman': rho, 'by_fold': folds}


def main():
    C = load_cells()
    th = theory()
    missing = [c['id'] for c in C if (c['enc'], c['ds']) not in th]
    if missing:
        print('missing theory for %d cells, e.g. %s' % (len(missing), missing[:2]))
    C = [c for c in C if (c['enc'], c['ds']) in th]
    print('cells %d, corpora %s' % (len(C), sorted({c['ds'] for c in C})))
    for key in ('s_fit', 'r0_fit'):
        r = run(C, th, key)
        print('%-8s LOO R2=%.3f  Spearman=%.3f  by fold %s'
              % (key, r['loo_r2'], r['spearman'],
                 {k: round(v['r2'], 2) for k, v in r['by_fold'].items()}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
