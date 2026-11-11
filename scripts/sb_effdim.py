"""Mechanism tests for the log-logistic response law.

Test 1 (form, not fit): is logit(Delta) linear in log rho?  A log-logistic law
is equivalent to logit(Delta) being affine in log rho, so a systematic quadratic
term means the form is only an approximation.  Uses the dense 23-rank mAP@10
curves so the curvature test has residual degrees of freedom.

Test 2 (sufficiency): after matching rho, does the effective dimension
d_eff(r) = (sum_{i<=r} w_i)^2 / sum_{i<=r} w_i^2  explain the change in Delta
caused by the scaling exponent gamma?  If it does, Delta is a function of
(rho, d_eff) and log-logistic is the rho-projection of a two-variable law.
If it does not, the gamma effect is a genuine third axis and the "law" is a
one-dimensional fit inside a fixed coordinate convention.

Usage: python3 sb_effdim.py --out effdim.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sb_common import OUT  # noqa: E402

EPS = 0.05
GAMMAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
RHOS = [0.05, 0.10, 0.20]


def load_theory():
    th = {}
    for p in glob.glob(os.path.join(OUT, 'theory__*.json')):
        d = json.load(open(p))
        th[(d['encoder'], d['dataset'])] = d
    return th


def omega(th, rel, gamma):
    lam = np.array(th['lambda'], float)
    return (lam ** (1.0 - gamma)) * np.array(th['tasks'][rel]['eta2'], float)


def rho_of(om, ranks):
    cum = np.clip(np.cumsum(om) / om.sum(), 0.0, 1.0)
    return np.array([1.0 - cum[min(r, len(cum)) - 1] for r in ranks])


def d_eff_of(om, ranks):
    out = []
    for r in ranks:
        w = om[:min(r, len(om))]
        s1, s2 = float(w.sum()), float((w ** 2).sum())
        out.append((s1 ** 2) / s2 if s2 > 0 else 1.0)
    return np.array(out, float)


def ols(X, y):
    """Return coefficients, standard errors, R^2."""
    XtX = X.T @ X
    beta = np.linalg.lstsq(XtX, X.T @ y, rcond=None)[0]
    resid = y - X @ beta
    n, p = X.shape
    dof = max(n - p, 1)
    sigma2 = float(resid @ resid) / dof
    cov = np.linalg.pinv(XtX) * sigma2
    se = np.sqrt(np.clip(np.diag(cov), 1e-30, None))
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else 0.0
    return beta, se, r2


def test_form():
    """Quadratic curvature of logit(Delta) vs log rho on 23-point curves."""
    rows = []
    for p in sorted(glob.glob(os.path.join(OUT, 'metric_std_parts', '*.json'))):
        d = json.load(open(p))
        grid = np.array([int(r) for r in d['grid']], int)
        for rel, cell in d.get('cells', {}).items():
            mm = cell.get('metrics', {}).get('map10')
            if not mm or 'curve' not in mm:
                continue
            curve = np.array(mm['curve'], float)
            full = float(cell.get('full', {}).get('map10', 0) or 0)
            if full <= 0:
                continue
            n = min(len(curve), len(grid), len(cell.get('rho', [])))
            rho = np.array(cell['rho'][:n], float)
            delta = 1.0 - curve[:n] / full
            m = (delta > 1e-3) & (delta < 1 - 1e-3) & (rho > 1e-5)
            if m.sum() < 6:
                continue
            x = np.log(rho[m])
            dc = np.clip(delta[m], 1e-6, 1 - 1e-6)
            z = np.log(dc / (1 - dc))
            X1 = np.vstack([np.ones_like(x), x]).T
            X2 = np.vstack([np.ones_like(x), x, x ** 2]).T
            _, _, r2_lin = ols(X1, z)
            b2, se2, r2_quad = ols(X2, z)
            t_quad = float(b2[2] / se2[2]) if se2[2] > 0 else 0.0
            rows.append({'enc': d['encoder'], 'ds': d['dataset'], 'rel': rel,
                         'n': int(m.sum()), 'r2_lin': r2_lin, 'r2_quad': r2_quad,
                         'd_r2': r2_quad - r2_lin, 't_quad': t_quad,
                         'b_lin': float(b2[1]), 'b_quad': float(b2[2])})
    if not rows:
        return {'cells': [], 'summary': {}}
    t = np.array([r['t_quad'] for r in rows])
    dr = np.array([r['d_r2'] for r in rows])
    return {'cells': rows,
            'summary': {'n_cells': len(rows),
                        'median_r2_lin': float(np.median([r['r2_lin'] for r in rows])),
                        'median_r2_quad': float(np.median([r['r2_quad'] for r in rows])),
                        'median_d_r2': float(np.median(dr)),
                        'median_abs_t_quad': float(np.median(np.abs(t))),
                        'frac_abs_t_gt_2': float(np.mean(np.abs(t) > 2.0)),
                        'median_slope': float(np.median([r['b_lin'] for r in rows]))}}


def test_sufficiency(th):
    """Match rho across gamma; does d_eff explain the residual Delta change?"""
    d = json.load(open(os.path.join(OUT, 'spectemp_full.json')))
    ranks_all = [int(r) for r in d['ranks']]
    pairs = []
    for c in d['cells']:
        if 'curves' not in c:
            continue
        key = (c['encoder'], c['dataset'])
        if key not in th or c['relation'] not in th[key]['tasks']:
            continue
        d_full = int(c['dim'])
        curves = {}
        for g in GAMMAS:
            vals = {r: c['curves'][f'r{r}|g{g:.1f}']['map10']
                    for r in ranks_all if f'r{r}|g{g:.1f}' in c['curves']}
            kf = f'r{d_full}|g{g:.1f}'
            full = c['curves'][kf]['map10'] if kf in c['curves'] else c['reference']['map10']
            if not vals or full <= 0:
                continue
            om = omega(th[key], c['relation'], g)
            rr = np.array(sorted(vals))
            curves[g] = {'rho': rho_of(om, rr), 'deff': d_eff_of(om, rr),
                         'delta': np.array([1.0 - vals[r] / full for r in rr]),
                         'ranks': rr}
        if 0.0 not in curves or 0.5 not in curves:
            continue
        for rho_star in RHOS:
            rec = {}
            for g in (0.0, 0.5):
                cv = curves[g]
                o = np.argsort(cv['rho'])   # rho decreases with rank
                if not (cv['rho'][o][0] <= rho_star <= cv['rho'][o][-1]):
                    break
                rec[g] = (float(np.interp(rho_star, cv['rho'][o], cv['delta'][o])),
                          float(np.interp(rho_star, cv['rho'][o], cv['deff'][o])))
            if len(rec) < 2:
                continue
            d_delta = rec[0.5][0] - rec[0.0][0]
            d_logde = float(np.log(rec[0.5][1]) - np.log(rec[0.0][1]))
            if abs(d_delta) < 1e-6 or abs(d_logde) < 1e-9:
                continue
            pairs.append({'enc': c['encoder'], 'ds': c['dataset'],
                          'rel': c['relation'], 'rho': rho_star,
                          'd_delta': float(d_delta), 'd_log_deff': d_logde,
                          'deff0': rec[0.0][1], 'deff5': rec[0.5][1]})
    if len(pairs) < 6:
        return {'pairs': pairs, 'summary': {'n': len(pairs)}}
    y = np.array([p['d_delta'] for p in pairs])
    x = np.array([p['d_log_deff'] for p in pairs])
    beta, se, r2 = ols(np.vstack([np.ones_like(x), x]).T, y)
    r2_within = []
    for rho_star in RHOS:
        idx = [i for i, p in enumerate(pairs) if p['rho'] == rho_star]
        if len(idx) < 4:
            continue
        xx, yy = x[idx], y[idx]
        if np.std(xx) < 1e-9:
            continue
        _, _, r2w = ols(np.vstack([np.ones_like(xx), xx]).T, yy)
        r2_within.append(r2w)
    return {'pairs': pairs,
            'summary': {'n': len(pairs),
                        'slope': float(beta[1]), 'slope_se': float(se[1]),
                        't_slope': float(beta[1] / se[1]) if se[1] > 0 else 0.0,
                        'r2': r2,
                        'median_r2_within_rho': float(np.median(r2_within)) if r2_within else None,
                        'median_abs_d_delta': float(np.median(np.abs(y))),
                        'median_abs_d_log_deff': float(np.median(np.abs(x)))},
            'by_relation': {rel: {'n': int(sum(1 for p in pairs if p['rel'] == rel)),
                                  'mean_d_delta': float(np.mean([p['d_delta'] for p in pairs
                                                                 if p['rel'] == rel]))}
                            for rel in sorted(set(p['rel'] for p in pairs))}}


def test_partial(th):
    """All (cell, gamma, rank) points: does d_eff add anything beyond rho?

    Each cell is demeaned first, so only within-cell variation remains and the
    test is not driven by between-cell differences.  Compares
    logit(Delta) ~ log rho + (log rho)^2   against   + log d_eff.

    Also reports the within-cell collinearity of log rho and log d_eff, and the
    per-cell d_eff slope: if d_eff were merely another monotone function of the
    rank, its slope would not be stable across independently fitted cells.
    """
    d = json.load(open(os.path.join(OUT, 'spectemp_full.json')))
    ranks_all = [int(r) for r in d['ranks']]
    zs, xs, ls = [], [], []
    per_cell = []
    for c in d['cells']:
        if 'curves' not in c:
            continue
        key = (c['encoder'], c['dataset'])
        if key not in th or c['relation'] not in th[key]['tasks']:
            continue
        d_full = int(c['dim'])
        cz, cx, cl = [], [], []
        for g in GAMMAS:
            vals = {r: c['curves'][f'r{r}|g{g:.1f}']['map10']
                    for r in ranks_all if f'r{r}|g{g:.1f}' in c['curves']}
            kf = f'r{d_full}|g{g:.1f}'
            full = c['curves'][kf]['map10'] if kf in c['curves'] else c['reference']['map10']
            if not vals or full <= 0:
                continue
            rr = np.array(sorted(vals))
            om = omega(th[key], c['relation'], g)
            rho, de = rho_of(om, rr), d_eff_of(om, rr)
            dl = np.array([1.0 - vals[r] / full for r in rr])
            m = (dl > 1e-3) & (dl < 1 - 1e-3) & (rho > 1e-5)
            if m.sum() < 4:
                continue
            dc = np.clip(dl[m], 1e-6, 1 - 1e-6)
            z = np.log(dc / (1 - dc))
            x, l = np.log(rho[m]), np.log(de[m])
            cz.append(z - z.mean()); cx.append(x - x.mean()); cl.append(l - l.mean())
        if not cz:
            continue
        z = np.concatenate(cz); x = np.concatenate(cx); l = np.concatenate(cl)
        zs.append(z); xs.append(x); ls.append(l)
        if len(z) >= 6 and np.std(l) > 1e-9:
            b, _, _ = ols(np.vstack([x, x ** 2, l]).T, z)
            per_cell.append({'enc': c['encoder'], 'ds': c['dataset'],
                             'rel': c['relation'], 'b_deff': float(b[2]),
                             'corr_rho_deff': float(np.corrcoef(x, l)[0, 1])})
    if not zs:
        return {'summary': {}}
    z, x, l = np.concatenate(zs), np.concatenate(xs), np.concatenate(ls)
    b0, _, r20 = ols(np.vstack([x, x ** 2]).T, z)
    b1, se1, r21 = ols(np.vstack([x, x ** 2, l]).T, z)
    slopes = np.array([p['b_deff'] for p in per_cell])
    corrs = np.array([p['corr_rho_deff'] for p in per_cell])
    return {'summary': {'n': int(len(z)), 'r2_base': r20, 'r2_with_deff': r21,
                        'd_r2': r21 - r20, 'b_deff': float(b1[2]),
                        't_deff': float(b1[2] / se1[2]) if se1[2] > 0 else 0.0,
                        'b_rho': float(b1[0]), 'b_rho2': float(b1[1]),
                        'corr_rho_deff_pooled': float(np.corrcoef(x, l)[0, 1]),
                        'median_cell_corr': float(np.median(corrs)) if len(corrs) else None,
                        'n_cells_fitted': len(per_cell),
                        'median_cell_b_deff': float(np.median(slopes)) if len(slopes) else None,
                        'frac_cells_negative_b_deff': float(np.mean(slopes < 0)) if len(slopes) else None},
            'per_cell': per_cell}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='effdim.json')
    args = ap.parse_args()
    th = load_theory()
    res = {'eps': EPS, 'gammas': GAMMAS, 'rhos': RHOS,
           'test1_form': test_form(), 'test2_sufficiency': test_sufficiency(th),
           'test2b_partial': test_partial(th)}
    json.dump(res, open(os.path.join(OUT, args.out), 'w'), indent=1)
    t1 = res['test1_form']['summary']
    t2 = res['test2_sufficiency']['summary']
    print('T1 cells=%d r2_lin=%.3f r2_quad=%.3f dR2=%.4f |t|=%.2f frac|t|>2=%.0f%%'
          % (t1.get('n_cells', 0), t1.get('median_r2_lin', 0), t1.get('median_r2_quad', 0),
             t1.get('median_d_r2', 0), t1.get('median_abs_t_quad', 0),
             100 * t1.get('frac_abs_t_gt_2', 0)), flush=True)
    print('T2 n=%d slope=%.3f (se %.3f, t %.1f) r2=%.3f r2_within_rho=%.3f'
          % (t2.get('n', 0), t2.get('slope', 0), t2.get('slope_se', 0),
             t2.get('t_slope', 0), t2.get('r2', 0), t2.get('median_r2_within_rho') or -1),
          flush=True)
    t3 = res['test2b_partial']['summary']
    print('T2b n=%d r2_base=%.3f r2_deff=%.3f dR2=%.4f b_deff=%.3f t=%.1f'
          % (t3.get('n', 0), t3.get('r2_base', 0), t3.get('r2_with_deff', 0),
             t3.get('d_r2', 0), t3.get('b_deff', 0), t3.get('t_deff', 0)), flush=True)
    print('wrote', os.path.join(OUT, args.out), flush=True)


if __name__ == '__main__':
    main()
