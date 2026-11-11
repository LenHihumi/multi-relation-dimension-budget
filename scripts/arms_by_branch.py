"""Per-branch reporting for the deployed rule, with intervals.

Table II reports one row per decision rule: coverage, median, p95, worst.  A reviewer
cannot see from that how the three branches divide the work, how the tail is spread
across them, or how much of the median is sampling noise at n = 19.  This script
answers all three.

It does not re-implement the rule.  It rebuilds the cell objects from the two released
tables (tables/map10_19cells.csv and tables/fits_19cells.csv) and calls
sb_probe.run_three_branch, the same function that produced the grid, so the reduction
and the rule cannot diverge.  No expected value is embedded here: the reduced table is
committed, so a drift shows up as a diff after the rebuild, and the caller that wants a
hard gate compares against the committed file rather than against a number in this
script.

Outputs (under tables/):
  arms_by_branch.csv   branch, n, coverage, median, p95, worst, mean_evals, cells
  arms_bootstrap.csv   statistic, point, lo, hi   (1000 resamples of the 19 cells)

Run from the artifact root:  python3 scripts/arms_by_branch.py
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import sb_probe as sp  # noqa: E402

ROOT = os.path.join(HERE, '..')
TAB = os.path.join(ROOT, 'tables')
EPS = 0.05


def cells_from_tables():
    """Rebuild the (grid, rho, delta, r_true) cell objects the rule needs."""
    per = {}
    with open(os.path.join(TAB, 'map10_19cells.csv'), encoding='utf-8') as fh:
        for r in csv.DictReader(fh):
            k = '%s/%s/%s' % (r['encoder'], r['dataset'], r['relation'])
            per.setdefault(k, []).append((int(r['rank']), float(r['rho_T']),
                                          float(r['delta'])))
    true = {}
    with open(os.path.join(TAB, 'fits_19cells.csv'), encoding='utf-8') as fh:
        for r in csv.DictReader(fh):
            k = '%s/%s/%s' % (r['encoder'], r['dataset'], r['relation'])
            true[k] = float(r['rstar_measured'])
    out = []
    for k, v in sorted(per.items()):
        v.sort()
        out.append({'id': k, 'ds': k.split('/')[1],
                    'grid': np.array([x[0] for x in v]),
                    'rho': np.array([x[1] for x in v]),
                    'delta': np.array([x[2] for x in v]),
                    'r_true': true[k]})
    return out


def agg(rows):
    ov = np.array([r['overhead'] for r in rows], float)
    meet = np.array([r['meet'] for r in rows], float)
    ev = np.array([r['probes'] for r in rows], float)
    return dict(n=len(rows), coverage=float(meet.mean()), median=float(np.median(ov)),
                p95=float(np.quantile(ov, .95)), worst=float(ov.max()),
                mean_evals=float(ev.mean()))


def boot(ov, stat, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    v = np.asarray(ov, float)
    draws = np.array([stat(rng.choice(v, size=v.size, replace=True))
                      for _ in range(n_boot)])
    return float(np.quantile(draws, .025)), float(np.quantile(draws, .975))


def main():
    C = cells_from_tables()
    res, rows = sp.run_three_branch(C, eps=EPS)
    got = {'median': res['median_overhead'], 'p95': res['p95_overhead'],
           'worst': res['max_overhead'], 'evals': res['mean_evals']}
    if not 0.0 < got['median'] <= got['p95'] <= got['worst']:
        raise SystemExit('branch reduction is not ordered (median %.2f <= p95 %.2f <= '
                         'worst %.2f) -- refusing to write a table that cannot be right'
                         % (got['median'], got['p95'], got['worst']))
    print('reduction over %d cells, coverage %.0f%%: median %.2f, p95 %.2f, worst %.2f, '
          'evals %.2f' % (len(C), 100 * res['coverage'], got['median'], got['p95'],
                          got['worst'], got['evals']))

    desc = {'A': 'straddle: fit the crossing, issue it with the margin',
            'B': 'both anchors under tolerance: issue the smaller measured rank',
            'C': 'both anchors over tolerance: ladder upward by measuring'}
    with open(os.path.join(TAB, 'arms_by_branch.csv'), 'w', newline='') as fh:
        w = csv.writer(fh, lineterminator='\n')
        w.writerow(['branch', 'description', 'n', 'coverage', 'median', 'p95', 'worst',
                    'mean_evals', 'cells'])
        for b in ('A', 'B', 'C'):
            sub = [r for r in rows if r['branch'] == b]
            s = agg(sub) if sub else None
            w.writerow([b, desc[b], len(sub),
                        round(s['coverage'], 4) if s else '',
                        round(s['median'], 3) if s else '',
                        round(s['p95'], 3) if s else '',
                        round(s['worst'], 3) if s else '',
                        round(s['mean_evals'], 2) if s else '',
                        '; '.join(sorted(r['id'] for r in sub))])
            if s:
                print('%-2s n=%-2d cov %.2f  median %.2f  p95 %.2f  worst %.2f  evals %.2f'
                      % (b, s['n'], s['coverage'], s['median'], s['p95'], s['worst'],
                         s['mean_evals']))
        s = agg(rows)
        w.writerow(['all', 'deployed rule', s['n'], round(s['coverage'], 4),
                    round(s['median'], 3), round(s['p95'], 3), round(s['worst'], 3),
                    round(s['mean_evals'], 2), ''])

    with open(os.path.join(TAB, 'arms_bootstrap.csv'), 'w', newline='') as fh:
        w = csv.writer(fh, lineterminator='\n')
        w.writerow(['scope', 'statistic', 'n', 'point', 'lo', 'hi', 'n_boot'])
        scopes = [('all', [r['overhead'] for r in rows])]
        for b in ('A', 'B', 'C'):
            sub = [r['overhead'] for r in rows if r['branch'] == b]
            if len(sub) >= 5:
                scopes.append((b, sub))
        for scope, ov in scopes:
            for name, stat, pt in (('median', np.median, np.median(ov)),
                                   ('mean', np.mean, np.mean(ov)),
                                   ('p95', lambda a: np.quantile(a, .95),
                                    np.quantile(ov, .95))):
                if name == 'p95' and len(ov) < 10:
                    continue          # a 95th percentile of 8 values is the max; no interval
                lo, hi = boot(ov, stat)
                w.writerow([scope, name, len(ov), round(float(pt), 3), round(lo, 3),
                            round(hi, 3), 1000])
                print('%-4s %-6s n=%-2d %5.2f  [%.2f, %.2f]'
                      % (scope, name, len(ov), pt, lo, hi))
    print('\nwrote tables/arms_by_branch.csv, tables/arms_bootstrap.csv')


if __name__ == '__main__':
    main()
