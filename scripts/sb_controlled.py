"""Controlled contrast: at matched rho, does d_eff still move Delta?

The regression in sb_effdim.py controls rho with a quadratic, which a reviewer
can dismiss as "d_eff is just another monotone function of the rank".  This
script removes the functional-form assumption: it bins the MEASURED (not
interpolated) points by rho and fits Delta ~ log d_eff inside each bin, after
demeaning per cell.  If the slope is negative and of comparable size across
bins, the d_eff effect survives without any model for the rho dependence.

Acceptance (pre-registered): median within-bin slope must be negative and its
magnitude at least half of the pooled regression estimate (-2.28 -> |b| >= 1.1).

Usage: python3 sb_controlled.py --out controlled.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sb_common import OUT  # noqa: E402
from sb_effdim import GAMMAS, d_eff_of, load_theory, omega, rho_of  # noqa: E402

BINS = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64]
POOLED_B = -2.282          # from sb_effdim.test_partial()
ACCEPT = abs(POOLED_B) / 2  # 1.141


def ols_no_intercept(x, y):
    beta = float((x @ y) / (x @ x)) if x @ x > 0 else 0.0
    resid = y - beta * x
    ss = float(y @ y)
    r2 = 1.0 - float(resid @ resid) / ss if ss > 0 else 0.0
    return beta, r2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='controlled.json')
    args = ap.parse_args()
    th = load_theory()
    d = json.load(open(os.path.join(OUT, 'spectemp_full.json')))
    ranks_all = [int(r) for r in d['ranks']]

    pts = []   # (cell_id, rho, delta, deff)
    for c in d['cells']:
        if 'curves' not in c:
            continue
        key = (c['encoder'], c['dataset'])
        if key not in th or c['relation'] not in th[key]['tasks']:
            continue
        cid = '%s/%s/%s' % (c['encoder'], c['dataset'], c['relation'])
        d_full = int(c['dim'])
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
            for i, r in enumerate(rr):
                if 1e-3 < dl[i] < 1 - 1e-3 and rho[i] > 1e-5:
                    pts.append((cid, float(rho[i]), float(dl[i]), float(de[i])))
    print('measured points', len(pts), flush=True)

    rows = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        sub = [p for p in pts if lo <= p[1] < hi]
        if len(sub) < 12:
            rows.append({'bin': [lo, hi], 'n': len(sub), 'slope': None})
            continue
        # demean per cell so the comparison is within-cell
        xs, ys, rs = [], [], []
        for cid in sorted({p[0] for p in sub}):
            q = [p for p in sub if p[0] == cid]
            if len(q) < 3:
                continue
            # logit scale, so the within-bin slope is comparable to the
            # pooled estimate (-2.28); raw Delta would be on a different scale.
            x = np.log([p[3] for p in q])
            lr = np.log([p[1] for p in q])
            dv = np.clip(np.array([p[2] for p in q]), 1e-4, 1 - 1e-4)
            y = np.log(dv / (1 - dv))
            xs.append(x - x.mean()); ys.append(y - y.mean())
            rs.append(lr - lr.mean())
        if not xs:
            rows.append({'bin': [lo, hi], 'n': len(sub), 'slope': None})
            continue
        x, y = np.concatenate(xs), np.concatenate(ys)
        r = np.concatenate(rs)
        if np.std(x) < 1e-9:
            rows.append({'bin': [lo, hi], 'n': len(sub), 'slope': None})
            continue
        b, r2 = ols_no_intercept(x, y)
        # residual collinearity inside the bin: if this is still high, part of
        # the within-bin explanatory power could come from leftover rho drift.
        corr_res = float(np.corrcoef(r, x)[0, 1]) if np.std(r) > 1e-9 else None
        rho_vals = np.array([p[1] for p in sub])
        rho_iqr = float(np.percentile(rho_vals, 75) - np.percentile(rho_vals, 25))
        # extreme-quartile contrast inside the bin
        order = np.argsort(x)
        k = max(1, len(order) // 4)
        d_hi = float(y[order[-k:]].mean())
        d_lo = float(y[order[:k]].mean())
        rows.append({'bin': [lo, hi], 'n': int(len(x)), 'slope': float(b), 'r2': float(r2),
                     'logit_gap_hi_minus_lo': d_hi - d_lo,
                     'corr_rho_deff_residual': corr_res,
                     'rho_iqr': rho_iqr, 'rho_median': float(np.median(rho_vals)),
                     'frac_negative_slope': None})

    slopes = [r['slope'] for r in rows if r.get('slope') is not None]
    med = float(np.median(slopes)) if slopes else None
    out = {'bins': rows, 'pooled_slope_reference': POOLED_B,
           'accept_threshold': ACCEPT, 'n_points': len(pts),
           'summary': {'n_bins_fitted': len(slopes),
                       'median_slope': med,
                       'frac_negative': float(np.mean(np.array(slopes) < 0)) if slopes else None,
                       'median_abs_slope': float(np.median(np.abs(slopes))) if slopes else None,
                       'pass': bool(med is not None and med < 0 and abs(med) >= ACCEPT),
                       'median_r2': float(np.median([r['r2'] for r in rows if 'r2' in r])) if slopes else None,
                       'median_rho_iqr': float(np.median([r['rho_iqr'] for r in rows if 'rho_iqr' in r])) if slopes else None,
                       'median_abs_corr_residual': float(np.median([abs(r['corr_rho_deff_residual'])
                                                                    for r in rows
                                                                    if r.get('corr_rho_deff_residual') is not None])) if slopes else None}}
    json.dump(out, open(os.path.join(OUT, args.out), 'w'), indent=1)
    for r in rows:
        if r.get('slope') is None:
            print('bin %.2f-%.2f n=%d (too few)' % (r['bin'][0], r['bin'][1], r['n']), flush=True)
        else:
            print('bin %.2f-%.2f n=%d slope=%.2f r2=%.3f gap=%.2f rhoIQR=%.3f corr_res=%s'
                  % (r['bin'][0], r['bin'][1], r['n'], r['slope'], r['r2'],
                     r['logit_gap_hi_minus_lo'], r['rho_iqr'],
                     'n/a' if r['corr_rho_deff_residual'] is None
                     else '%.2f' % r['corr_rho_deff_residual']), flush=True)
    print('SUMMARY', out['summary'], flush=True)
    print('wrote', os.path.join(OUT, args.out), flush=True)


if __name__ == '__main__':
    main()
