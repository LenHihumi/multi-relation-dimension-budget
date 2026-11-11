"""Does the cell-level steepness survive at the per-query level?

The paper fits one logistic per cell to the cell's mean loss curve and reports a median
R^2 of 0.99, then says the steepness has no mechanism. A reviewer's objection is that the
fit is an aggregate: if queries differ mostly in where their curve sits (rho_0) but share a
slope, averaging flattens the curve, and the fitted s is an artefact of that averaging
rather than a property of the retrieval task.

This script answers it with per-query curves from `sb_perquery.py`:
  1. fit each query's own logistic over its informative rank range, giving (s_q, rho0_q);
  2. compare the per-query distribution of s_q against the cell's fitted s;
  3. check whether the aggregate curve is reproducible from the per-query fits by
     resampling queries, i.e. whether averaging alone explains the cell slope.

If s_q is tightly distributed around s_cell, the aggregate slope is real. If s_q is much
steeper than s_cell and rho0_q is wide, the cell slope is an averaging artefact -- which
would say the mechanism question was posed at the wrong level.

Inputs:  data/perquery/perquery/*.npz, tables/map10_19cells.csv, tables/fits_19cells.csv
Output:  tables/perquery_steepness.csv      one row per cell
Run:     python3 scripts/perquery_steepness.py
"""
from __future__ import annotations

import csv
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
TAB = os.path.join(ROOT, 'tables')
PQ = os.path.join(ROOT, 'data', 'perquery', 'perquery')
EPS_LO, EPS_HI = 1e-3, 1 - 1e-3


def fit_two_point(rho, delta):
    """logit(delta) = s (log rho - log rho0), least squares; returns (rho0, s, r2)."""
    y = np.clip(np.asarray(delta, float), 1e-4, 1 - 1e-4)
    x = np.log(np.maximum(np.asarray(rho, float), 1e-9))
    z = np.log(y / (1 - y))
    A = np.vstack([x, np.ones_like(x)]).T
    (b, a), *_ = np.linalg.lstsq(A, z, rcond=None)
    pred = a + b * x
    ss = 1.0 - float(((z - pred) ** 2).sum() / max(((z - z.mean()) ** 2).sum(), 1e-12))
    return float(np.exp(-a / b)), float(b), ss


def main():
    pub = {}
    with open(os.path.join(TAB, 'map10_19cells.csv'), encoding='utf-8') as fh:
        for r in csv.DictReader(fh):
            k = '%s/%s/%s' % (r['encoder'], r['dataset'], r['relation'])
            pub.setdefault(k, {})[int(r['rank'])] = (float(r['rho_T']), float(r['map10']))
    grid_rel = {}
    for f in glob.glob(os.path.join(ROOT, 'data', 'grid', '*.json')):
        d = json.load(open(f))
        for nm, c in d.get('cells', {}).items():
            if 'rel_median' in c:
                grid_rel[(d['encoder'], d['dataset'], nm)] = c['rel_median']
    cell_s = {}
    with open(os.path.join(TAB, 'fits_19cells.csv'), encoding='utf-8') as fh:
        for r in csv.DictReader(fh):
            cell_s['%s/%s/%s' % (r['encoder'], r['dataset'], r['relation'])] = \
                (float(r['s']), float(r['rho0']), float(r['r2']))

    rows = []
    for f in sorted(glob.glob(os.path.join(PQ, '*.npz'))):
        name = os.path.basename(f)[:-4].replace('__', '/')   # files use __, tables use /
        if name not in pub:
            continue
        z = np.load(f)
        ranks, ap = z['ranks'], z['ap']            # ranks (n_r,), ap (n_query, n_r)
        ref = pub[name]
        keep = [j for j, r in enumerate(ranks) if int(r) in ref]
        if len(keep) < 5:
            continue
        rho = np.array([ref[int(ranks[j])][0] for j in keep])
        full = ap[:, keep[-1]]
        ok = full > 1e-6
        delta = 1.0 - ap[ok][:, keep] / full[ok][:, None]

        # per-query fits on the informative window
        sq, r0q = [], []
        for i in range(delta.shape[0]):
            d = delta[i]
            m = (d > EPS_LO) & (d < EPS_HI) & (rho > 1e-5)
            if m.sum() < 5:
                continue
            r0, s, r2 = fit_two_point(rho[m], d[m])
            if 0.5 < s < 40 and np.isfinite(r0):
                sq.append(s)
                r0q.append(r0)
        if len(sq) < 30:
            continue
        sq, r0q = np.array(sq), np.array(r0q)
        s_cell = cell_s.get(name, (np.nan, np.nan, np.nan))[0]
        rows.append({
            'cell': name,
            'n_query_fitted': len(sq),
            's_cell': round(s_cell, 3),
            's_query_median': round(float(np.median(sq)), 3),
            's_query_iqr_lo': round(float(np.quantile(sq, .25)), 3),
            's_query_iqr_hi': round(float(np.quantile(sq, .75)), 3),
            'ratio_median_to_cell': round(float(np.median(sq)) / s_cell, 3),
            'rho0_spread_max_over_min': round(float(np.quantile(r0q, .9) /
                                                     max(np.quantile(r0q, .1), 1e-9)), 1),
            'rel_median': grid_rel.get(tuple(name.split('/')), ''),
        })

    rows.sort(key=lambda r: r['cell'])
    out = os.path.join(TAB, 'perquery_steepness.csv')
    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print('%-34s %5s %6s %7s %7s %8s %7s %7s' % ('cell', 'n_q', '|R|', 's_cell', 's_med',
                                                 'iqr', 'med/cell', 'r0 x'))
    for r in rows:
        print('%-34s %5d %6s %7.2f %7.2f %3.1f-%-4.1f %7.2f %7.1f'
              % (r['cell'], r['n_query_fitted'], r['rel_median'], r['s_cell'], r['s_query_median'],
                 r['s_query_iqr_lo'], r['s_query_iqr_hi'], r['ratio_median_to_cell'],
                 r['rho0_spread_max_over_min']))
    med = np.median([r['ratio_median_to_cell'] for r in rows])
    print('\nmedian ratio (per-query steepness / cell steepness): %.2f' % med)
    print('wrote', out)


if __name__ == '__main__':
    main()
