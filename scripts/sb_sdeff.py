"""Does the steepness s actually track effective dimension d_eff?

The claim "S-shape steepness is set by the effective dimension" needs its own
test: the previous result only showed that Delta depends on d_eff after
controlling rho, which is a statement about the LEVEL, not about the slope s.

Two datasets:
  (a) 19 mAP@10 cells at gamma=0 (23-point curves, reliable fits)
  (b) 12 cells x 6 gamma levels (5-point curves, same fit as gamma_law)

Usage: python3 sb_sdeff.py --out sdeff.json
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
from sb_effdim import GAMMAS, d_eff_of, load_theory, omega, rho_of  # noqa: E402
from sb_acceptance import fit_ll  # noqa: E402


def ols(x, y):
    X = np.vstack([np.ones_like(x), x]).T
    b = np.linalg.lstsq(X.T @ X, X.T @ y, rcond=None)[0]
    r = y - X @ b
    r2 = 1.0 - float(r @ r) / float(((y - y.mean()) ** 2).sum())
    return float(b[1]), r2


def deff_at(th, rel, gamma, rho_star=0.10):
    lam = np.array(th['lambda'], float)
    om = (lam ** (1.0 - gamma)) * np.array(th['tasks'][rel]['eta2'], float)
    rr = np.arange(1, len(om) + 1)
    rho = rho_of(om, rr)
    de = d_eff_of(om, rr)
    o = np.argsort(rho)
    if not (rho[o][0] <= rho_star <= rho[o][-1]):
        return None
    return float(np.interp(rho_star, rho[o], de[o]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='sdeff.json')
    args = ap.parse_args()
    th = load_theory()

    # (a) 19 cells at gamma = 0
    rows_a = []
    for p in sorted(glob.glob(os.path.join(OUT, 'metric_std_parts', '*.json'))):
        d = json.load(open(p))
        key = (d['encoder'], d['dataset'])
        if key not in th:
            continue
        for rel, cell in d.get('cells', {}).items():
            mm = cell.get('metrics', {}).get('map10')
            if not mm or 'curve' not in mm or rel not in th[key].get('tasks', {}):
                continue
            full = float(cell.get('full', {}).get('map10', 0) or 0)
            if full <= 0:
                continue
            n = min(len(mm['curve']), len(cell.get('rho', [])))
            rho = np.array(cell['rho'][:n], float)
            delta = 1.0 - np.array(mm['curve'][:n], float) / full
            m = (delta > 1e-3) & (delta < 1 - 1e-3) & (rho > 1e-5)
            if m.sum() < 5:
                continue
            r0, s = fit_ll(rho[m], delta[m])
            de = deff_at(th[key], rel, 0.0)
            if de is None:
                continue
            rows_a.append({'id': '%s/%s/%s' % (d['encoder'], d['dataset'], rel),
                           's': s, 'rho0': r0, 'deff': de})

    # (b) 12 cells x 6 gammas, s from the same 5-point fit as gamma_law
    dd = json.load(open(os.path.join(OUT, 'spectemp_full.json')))
    ranks_all = [int(r) for r in dd['ranks']]
    rows_b = []
    for c in dd['cells']:
        if 'curves' not in c:
            continue
        key = (c['encoder'], c['dataset'])
        if key not in th or c['relation'] not in th[key].get('tasks', {}):
            continue
        d_full = int(c['dim'])
        for g in GAMMAS:
            vals = {r: c['curves'][f'r{r}|g{g:.1f}']['map10']
                    for r in ranks_all if f'r{r}|g{g:.1f}' in c['curves']}
            kf = f'r{d_full}|g{g:.1f}'
            full = c['curves'][kf]['map10'] if kf in c['curves'] else c['reference']['map10']
            if len(vals) < 4 or full <= 0:
                continue
            rr = np.array(sorted(vals))
            om = omega(th[key], c['relation'], g)
            rho = rho_of(om, rr)
            delta = np.array([1.0 - vals[r] / full for r in rr])
            m = (delta > 1e-3) & (delta < 1 - 1e-3) & (rho > 1e-5)
            if m.sum() < 4:
                continue
            r0, s = fit_ll(rho[m], delta[m])
            de = deff_at(th[key], c['relation'], g)
            if de is None:
                continue
            rows_b.append({'id': '%s/%s/%s/g%.1f' % (c['encoder'], c['dataset'],
                                                    c['relation'], g),
                           's': s, 'deff': de, 'gamma': g})

    out = {'set_a_gamma0': {'rows': rows_a}, 'set_b_all_gamma': {'rows': rows_b}}
    for tag, rows in (('set_a_gamma0', rows_a), ('set_b_all_gamma', rows_b)):
        if len(rows) < 5:
            out[tag]['summary'] = {'n': len(rows)}
            continue
        x = np.log([r['deff'] for r in rows])
        y = np.log([r['s'] for r in rows])
        b, r2 = ols(x, y)
        sp = float(np.corrcoef(x, y)[0, 1])
        out[tag]['summary'] = {'n': len(rows), 'slope_log_s_on_log_deff': b,
                               'r2': r2, 'pearson_log': sp,
                               'median_s': float(np.median([r['s'] for r in rows])),
                               'median_deff': float(np.median([r['deff'] for r in rows]))}
        print('%s n=%d slope=%.3f R2=%.3f pearson=%.3f  (median s=%.2f, d_eff=%.0f)'
              % (tag, len(rows), b, r2, sp, out[tag]['summary']['median_s'],
                 out[tag]['summary']['median_deff']), flush=True)
    json.dump(out, open(os.path.join(OUT, args.out), 'w'), indent=1)
    print('wrote', os.path.join(OUT, args.out), flush=True)


if __name__ == '__main__':
    main()
