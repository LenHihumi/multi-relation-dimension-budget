"""True leave-one-dataset-out, and the prior-plus-k-points repair.

The paper's cross-domain numbers hold out a (dataset, relation) pair, so the
same dataset's other relation is still in the training fold.  This script does
three things the reviewers asked for:

  1. a genuine leave-one-DATASET-out split (3 folds, one per corpus) with no
     cell of the held-out corpus in training;
  2. the same for leave-one-encoder-out, as a proxy for a new backbone;
  3. the "spectral prior + k truncation points" repair from the conclusion:
     the prior supplies the shape s (median over training folds of the same
     relation type), and k in {1,2} measured points supply the position rho0.
     The safety margin is the 90th percentile of the same procedure's residuals
     computed on the training folds only.

Every arm is scored the same way as Sec. V: pick a rank, then check the MEASURED
curve at that rank.  A cell "meets" when the measured Delta at the chosen rank
is <= eps.

Usage: python3 sb_lodo_true.py --out lodo_true.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from sb_acceptance import EPS, fit_ll
from sb_common import OUT

PRIOR_RHO = {1: (0.08,), 2: (0.04, 0.12), 3: (0.03, 0.08, 0.14)}


def log(m):
    print(m, flush=True)


def curve(rho, r0, s):
    return 1.0 / (1.0 + (r0 / np.maximum(rho, 1e-12)) ** s)


def rank_for(rho, delta_upper, grid, eps=EPS):
    ok = np.where(delta_upper <= eps)[0]
    if len(ok) == 0:
        return int(grid[-1])
    i = int(ok[0])
    return int(grid[i])


def pick_points(rho, grid, k):
    """Measured ranks closest to the prescribed rho targets."""
    idx = []
    for t in PRIOR_RHO[k]:
        j = int(np.argmin(np.abs(np.log(np.maximum(rho, 1e-9)) - np.log(t))))
        if j not in idx:
            idx.append(j)
    return sorted(idx)


def load_cells():
    from sb_effdim import load_theory
    th = load_theory()
    out = []
    for p in sorted(glob.glob(os.path.join(OUT, 'metric_std_parts', '*.json'))):
        d = json.load(open(p))
        key = (d['encoder'], d['dataset'])
        if key not in th:
            continue
        grid = np.array([int(r) for r in d['grid']], int)
        for rel, cell in d.get('cells', {}).items():
            mm = cell.get('metrics', {}).get('map10')
            if not mm or 'curve' not in mm:
                continue
            full = float(cell.get('full', {}).get('map10', 0) or 0)
            if full <= 0:
                continue
            n = min(len(mm['curve']), len(cell.get('rho', [])), len(grid))
            rho = np.array(cell['rho'][:n], float)
            delta = 1.0 - np.array(mm['curve'][:n], float) / full
            g = grid[:n]
            m = (delta > 1e-3) & (delta < 1 - 1e-3) & (rho > 1e-5)
            if m.sum() < 5:
                continue
            r0, s = fit_ll(rho[m], delta[m])
            # measured 95% budget
            ok = np.where(delta <= EPS)[0]
            r_true = int(g[ok[0]]) if len(ok) else int(g[-1])
            out.append({'id': '%s/%s/%s' % (d['encoder'], d['dataset'], rel),
                        'enc': d['encoder'], 'ds': d['dataset'], 'rel': rel,
                        'grid': g, 'rho': rho, 'delta': delta,
                        'r0_fit': float(r0), 's_fit': float(s),
                        'r_true': r_true,
                        'feat': spectral_feat(th[key], d, rel),
                        'feat_lab': spectral_feat(th[key], d, rel) + [
                            float(np.interp(0.15, rho[::-1], delta[::-1])), 0.0]})
    return out


def spectral_feat(th, d, rel):
    lam = np.array(th['lambda'], float)
    p = lam / lam.sum()
    ent = float(-(p * np.log(np.maximum(p, 1e-30))).sum())
    pr = float(lam.sum() ** 2 / (lam ** 2).sum())
    return [float(np.log(th['dim'])), float(np.log(pr)), ent,
            float(np.log(lam[0] / max(lam[1], 1e-12))),
            1.0 if rel == 'inst' else 0.0,
            1.0 if 'wemm' in d['encoder'] else 0.0]


def ridge_fit(X, y, alpha=1.0):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    A = Z.T @ Z + alpha * np.eye(Z.shape[1])
    w = np.linalg.solve(A, Z.T @ (y - y.mean()))
    return {'mu': mu, 'sd': sd, 'w': w, 'b': float(y.mean())}


def ridge_pred(m, X):
    return (((X - m['mu']) / m['sd']) @ m['w']) + m['b']


def eval_rank(c, r_hat):
    j = int(np.argmin(np.abs(c['grid'] - r_hat)))
    return (1 if c['delta'][j] <= EPS else 0), float(c['grid'][j] / max(c['r_true'], 1))


def summarise(rows):
    meet = np.array([r['meet'] for r in rows], float)
    ov = np.array([r['overhead'] for r in rows], float)
    l2 = np.abs(np.log2(np.clip(ov, 1e-6, None)))
    return {'n': len(rows), 'coverage': float(meet.mean()),
            'median_overhead': float(np.median(ov)),
            'within_1_25x': float(np.mean(l2 <= np.log2(1.25))),
            'within_2x': float(np.mean(l2 <= 1.0)),
            'p95_overhead': float(np.quantile(ov, 0.95)),
            'max_overhead': float(ov.max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='lodo_true.json')
    ap.add_argument('--eps', type=float, default=EPS)
    args = ap.parse_args()

    C = load_cells()
    log('cells %d  datasets %s  encoders %d'
        % (len(C), sorted({c['ds'] for c in C}), len({c['enc'] for c in C})))
    for c in C:
        c['y'] = np.log(max(c['r_true'], 1))

    res = {'eps': args.eps, 'n_cells': len(C),
           'datasets': sorted({c['ds'] for c in C}),
           'encoders': sorted({c['enc'] for c in C})}

    def group_folds(key):
        return sorted({c[key] for c in C})

    # ---------- arm 1/2: predictor, true leave-one-dataset / leave-one-encoder ----------
    for tag, key, fkey in (('spec_lodo_dataset', 'ds', 'feat'),
                           ('spec_lodo_encoder', 'enc', 'feat'),
                           ('lab_lodo_dataset', 'ds', 'feat_lab')):
        rows = []
        for g in group_folds(key):
            tr = [c for c in C if c[key] != g]
            te = [c for c in C if c[key] == g]
            model = ridge_fit(np.array([c[fkey] for c in tr]), np.array([c['y'] for c in tr]))
            pr = ridge_pred(model, np.array([c[fkey] for c in te]))
            for c, p in zip(te, pr):
                meet, ov = eval_rank(c, float(np.exp(p)))
                rows.append({'id': c['id'], 'holdout': g, 'pred': float(np.exp(p)),
                             'meet': meet, 'overhead': ov})
        res[tag] = {'summary': summarise(rows), 'rows': rows}
        s = res[tag]['summary']
        log('%-20s n=%d cov=%.0f%% med=%.2fx w2=%.0f%%'
            % (tag, s['n'], 100 * s['coverage'], s['median_overhead'], 100 * s['within_2x']))

    # ---------- arm 3: prior-s + k points, folds by dataset ----------
    for k in (1, 2):
        rows = []
        for g in group_folds('ds'):
            tr = [c for c in C if c['ds'] != g]
            te = [c for c in C if c['ds'] == g]
            # shape prior: median s of the same relation type in training folds
            marg = {}
            for rel in ('inst', 'cat'):
                ss = [c['s_fit'] for c in tr if c['rel'] == rel]
                if not ss:
                    ss = [c['s_fit'] for c in tr]
                marg[rel] = float(np.median(ss))
            # safety margin: residuals of the SAME procedure on training folds
            resid = []
            for c in tr:
                idx = pick_points(c['rho'], c['grid'], k)
                sp = marg[c['rel']]
                lg = [np.log(c['rho'][i]) - np.log(np.clip(c['delta'][i], 1e-4, 1 - 1e-4)
                                                  / (1 - np.clip(c['delta'][i], 1e-4, 1 - 1e-4))) / sp
                      for i in idx]
                r0 = float(np.exp(np.mean(lg)))
                resid.extend((c['delta'][idx] - curve(c['rho'][idx], r0, sp)).tolist())
            bias = max(0.0, float(np.quantile(resid, 0.90)))
            for c in te:
                sp = marg[c['rel']]
                idx = pick_points(c['rho'], c['grid'], k)
                lg = [np.log(c['rho'][i]) - np.log(np.clip(c['delta'][i], 1e-4, 1 - 1e-4)
                                                  / (1 - np.clip(c['delta'][i], 1e-4, 1 - 1e-4))) / sp
                      for i in idx]
                r0 = float(np.exp(np.mean(lg)))
                du = np.clip(curve(c['rho'], r0, sp) + bias, 0, 1)
                r_hat = rank_for(c['rho'], du, c['grid'], args.eps)
                meet, ov = eval_rank(c, r_hat)
                rows.append({'id': c['id'], 'holdout': g, 's_prior': sp,
                             'rho0': r0, 'bias': bias, 'rank': r_hat,
                             'meet': meet, 'overhead': ov})
        res['prior_s_plus_%dpt' % k] = {'summary': summarise(rows), 'rows': rows}
        s = res['prior_s_plus_%dpt' % k]['summary']
        log('%-20s n=%d cov=%.0f%% med=%.2fx w2=%.0f%%'
            % ('prior_s_plus_%dpt' % k, s['n'], 100 * s['coverage'],
               s['median_overhead'], 100 * s['within_2x']))

    # ---------- arm 4: free two-parameter fit with k points (same folds) ----------
    for k in (2, 3):
        rows = []
        for g in group_folds('ds'):
            tr = [c for c in C if c['ds'] != g]
            te = [c for c in C if c['ds'] == g]
            resid = []
            for c in tr:
                idx = pick_points(c['rho'], c['grid'], k)
                r0, s = fit_ll(c['rho'][idx], c['delta'][idx])
                resid.extend((c['delta'][idx] - curve(c['rho'][idx], r0, s)).tolist())
            bias = max(0.0, float(np.quantile(resid, 0.90)))
            for c in te:
                idx = pick_points(c['rho'], c['grid'], k)
                r0, s = fit_ll(c['rho'][idx], c['delta'][idx])
                du = np.clip(curve(c['rho'], r0, s) + bias, 0, 1)
                r_hat = rank_for(c['rho'], du, c['grid'], args.eps)
                meet, ov = eval_rank(c, r_hat)
                rows.append({'id': c['id'], 'holdout': g, 'rank': r_hat,
                             'meet': meet, 'overhead': ov})
        res['free_%dpt' % k] = {'summary': summarise(rows), 'rows': rows}
        s = res['free_%dpt' % k]['summary']
        log('%-20s n=%d cov=%.0f%% med=%.2fx w2=%.0f%%'
            % ('free_%dpt' % k, s['n'], 100 * s['coverage'],
               s['median_overhead'], 100 * s['within_2x']))

    # per-fold detail for the best repair arm
    for tag in ('prior_s_plus_2pt', 'free_3pt'):
        agg = {}
        for r in res[tag]['rows']:
            agg.setdefault(r['holdout'], []).append(r)
        res[tag]['by_fold'] = {k: summarise(v) for k, v in agg.items()}

    json.dump(res, open(os.path.join(OUT, args.out), 'w'), indent=1)
    print()
    print('%-22s %5s %8s %9s %8s' % ('arm', 'n', 'cover', 'median', 'p95'))
    for k, v in res.items():
        if isinstance(v, dict) and 'summary' in v:
            s = v['summary']
            print('%-22s %5d %7.0f%% %8.2fx %8.2fx'
                  % (k, s['n'], 100 * s['coverage'], s['median_overhead'], s['p95_overhead']))
    print('wrote', os.path.join(OUT, args.out))


if __name__ == '__main__':
    main()
