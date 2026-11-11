"""Is the "not met" branch of the protocol an empty set?

The protocol has two decided outcomes and one degenerate one: decided-met,
decided-not-met, and undecidable -> issue the top of the bracket.  Only the first
and third have instances in the ledger, because everything that was issued met the
tolerance.  The question is whether the second branch can be exercised at all
without new retrieval.

It can.  The 23-rank grid itself contains the under-provisioned region: every cell
has grid points whose measured Delta exceeds the tolerance -- 173 of them in total,
between 3 and 17 per cell, the worst losing 94% of full-dimension mAP at rank 8.
These are not synthetic: they are the same 1000-query measurements the fits are
built on.  What was never done is *reading* them through the same |z| test.

Method.  For every grid point, Delta = 1 - curve/full, and the per-query spread is
recovered from the stored 5th/95th band exactly as in sb_resolution.py:
sd = (q95-q05)/(2*1.645).  The full-dimension spread is taken at the largest grid
rank, where Delta is smallest.  SE is the delta method with corr = 0.81, n = 1000,
and z = (Delta - eps)/SE, positive on the under-provisioned side.  A point is
*decided* when |z| > 1.96.

Output is the resolution curve of the third branch: decided-not-met as a function of
how far past the tolerance the point sits.  If deep under-provisioning decides and
only the band around eps does not, the three-way protocol is verified on existing
data and the "unverified branch" caveat can be narrowed to "the band, not the
branch".

Usage: python sb_under.py --out under.json
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import statistics as st
from math import sqrt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
D = os.path.join(ROOT, 'data')

EPS = 0.05
Z95 = 1.645
ZCRIT = 1.96
CORR = 0.81
N = 1000


def se_delta(sd_f, sd_r, mean_f, mean_r, corr=CORR, n=N):
    v = (sd_r ** 2 / mean_f ** 2
         + (mean_r ** 2) * (sd_f ** 2) / mean_f ** 4
         - 2 * mean_r * corr * sd_r * sd_f / mean_f ** 3) / n
    return sqrt(max(v, 1e-14))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='under.json')
    args = ap.parse_args()

    rows, cells = [], []
    for p in sorted(glob.glob(os.path.join(D, 'metric_std_parts', '*.json'))):
        d = json.load(open(p))
        grid = d['grid']
        for rel, c in d['cells'].items():
            mm = c['metrics']['map10']
            full = float(c['full']['map10'])
            curve = mm['curve']
            lo, hi = mm['band']
            sds = [max((hi[j] - lo[j]) / (2 * Z95), 0.0) for j in range(len(grid))]
            jmax = len(grid) - 1
            sd_f = sds[jmax] if sds[jmax] > 0 else max(sds)
            cid = '%s/%s/%s' % (d['encoder'], d['dataset'], rel)
            n_over = n_dec = 0
            for j, r in enumerate(grid):
                delta = 1.0 - float(curve[j]) / full
                if delta <= EPS:
                    continue
                n_over += 1
                se = se_delta(sd_f, sds[j], full, float(curve[j]))
                z = (delta - EPS) / se
                dec = abs(z) > ZCRIT
                n_dec += int(dec)
                rows.append({'id': cid, 'rank': r, 'delta': delta,
                             'excess': delta - EPS, 'se': se, 'z': z,
                             'state': 'not_met' if dec else 'undecidable_under',
                             'ratio_to_rstar': None})
            cells.append({'id': cid, 'n_over': n_over, 'n_decided': n_dec,
                          'decided_frac': n_dec / n_over if n_over else None})

    # how the decision rate depends on distance past the tolerance, in units of SE
    bands = [(0, 1), (1, 2), (2, 3), (3, 5), (5, 10), (10, 1e9)]
    curve = []
    for a, b in bands:
        sel = [r for r in rows if a <= r['z'] < b]
        curve.append({'z_lo': a, 'z_hi': b, 'n': len(sel),
                      'decided': sum(1 for r in sel if r['state'] == 'not_met')})

    # robustness: the SE came from the same delta method that under-estimates it
    sens = {}
    for label, scale in (('se_x1.189', 1.189), ('se_x1.597', 1.597)):
        k = sum(1 for r in rows if (r['z'] / scale) > ZCRIT)
        sens[label] = {'n_decided': k, 'frac': k / len(rows)}

    # replay the proposed experiment: issue at 0.5x / 0.7x / 0.9x the oracle rank
    seq = json.load(open(os.path.join(D, 'results', 'sequential.json')))
    rstar = {c['id']: c['issued'] for c in seq['per_cell']['S_fit']}
    replay = []
    grids = {}
    for p2 in sorted(glob.glob(os.path.join(D, 'metric_std_parts', '*.json'))):
        g = json.load(open(p2))
        for rel in g['cells']:
            grids['%s/%s/%s' % (g['encoder'], g['dataset'], rel)] = g['grid']
    for cid, r in rstar.items():
        if cid not in grids or cid not in {x['id'] for x in cells}:
            continue
        g = grids[cid]
        for frac in (0.5, 0.7, 0.9):
            target = frac * r
            j = min(range(len(g)), key=lambda i: abs(g[i] - target))
            match = [x for x in rows if x['id'] == cid and x['rank'] == g[j]]
            if not match:
                replay.append({'id': cid, 'frac': frac, 'target': target,
                               'rank': g[j], 'state': 'met_or_grid_miss'})
                continue
            x = match[0]
            replay.append({'id': cid, 'frac': frac, 'target': target, 'rank': g[j],
                           'delta': x['delta'], 'excess': x['excess'], 'z': x['z'],
                           'state': x['state']})

    n = len(rows)
    n_dec = sum(1 for r in rows if r['state'] == 'not_met')
    out = {'eps': EPS, 'n_grid_points_over_eps': n, 'n_decided_not_met': n_dec,
           'decided_frac': n_dec / n if n else None,
           'sensitivity': sens, 'replay_0.5_0.7_0.9': replay,
           'n_cells_with_over': len(cells),
           'resolution_curve': curve,
           'cells': sorted(cells, key=lambda c: -c['n_over']),
           'rows': sorted(rows, key=lambda r: r['z'])}

    print('grid points with Delta > eps: %d across %d cells' % (n, len(cells)))
    print('decided "not met" at |z| > %.2f: %d (%.0f%%)'
          % (ZCRIT, n_dec, 100 * n_dec / n if n else 0))
    print('\nwhere the decision happens (z = excess past tolerance in SE units):')
    print('%-12s %6s %8s %8s' % ('z band', 'n', 'decided', 'rate'))
    for b in curve:
        hi = b['z_hi'] if b['z_hi'] < 1e8 else 'inf'
        print('%-12s %6d %8d %7.0f%%'
              % ('[%s,%s)' % (b['z_lo'], hi), b['n'], b['decided'],
                 100 * b['decided'] / b['n'] if b['n'] else 0))
    print('\nper cell:')
    print('%-32s %8s %9s %7s' % ('cell', 'over-eps', 'decided', 'rate'))
    for c in out['cells']:
        print('%-32s %8d %9d %6.0f%%'
              % (c['id'][:32], c['n_over'], c['n_decided'], 100 * (c['decided_frac'] or 0)))
    print('\nundecided but under-provisioned (the band around eps):')
    for r in [x for x in out['rows'] if x['state'] == 'undecidable_under'][:8]:
        print('  %-30s rank %3d  delta %.3f  excess %.4f  z %.2f'
              % (r['id'][:30], r['rank'], r['delta'], r['excess'], r['z']))

    print('\nrobustness to the SE calibration range:')
    for k, v in sens.items():
        print('  %-10s decided %3d of %d (%.0f%%)'
              % (k, v['n_decided'], n, 100 * v['frac']))

    print('\nreplay of the proposed experiment (issue at 0.5x / 0.7x / 0.9x r*, '
          'nearest grid point):')
    print('%-30s %5s %6s %6s %8s %8s %6s %s'
          % ('cell', 'frac', 'r*', 'rank', 'delta', 'excess', 'z', 'state'))
    for x in replay[:24]:
        print('%-30s %5.1f %6d %6d %8.3f %8.4f %6.2f %s'
              % (x['id'][:30], x['frac'], x['target'], x['rank'],
                 x.get('delta', 0), x.get('excess', 0), x.get('z', 0), x['state']))
    agg = collections.defaultdict(lambda: [0, 0])
    for x in replay:
        if 'state' not in x:
            continue
        agg[x['frac']][1] += 1
        agg[x['frac']][0] += int(x['state'] == 'not_met')
    print('\ndecided rate by issue depth:')
    for frac in (0.5, 0.7, 0.9):
        a, b = agg[frac]
        print('  %.1fx r*: %d of %d decided not-met' % (frac, a, b))

    with open(os.path.join(D, 'results', args.out), 'w') as f:
        json.dump(out, f, indent=1)
    print('\nwrote %s' % os.path.join(D, 'results', args.out))


if __name__ == '__main__':
    main()
