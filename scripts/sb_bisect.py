"""Q2: what does fitting a curve buy over plain bisection?

The natural assumption-free opponent of the two-point fit is a bisection on the
rank: probe a rank, read the measured loss, halve the interval, repeat. It
assumes no functional form, needs no reference cells and no safety margin -- and
it lands exactly on the measured crossing.

This script simulates that procedure on the measured 23-rank grids, counts the
evaluations it spends, and puts it next to the fitted arms on the same
cost/quality axes. No retrieval is run: an "evaluation" is one look-up of the
measured loss at a rank, which is what a real evaluation would return.

Two things are reported that the paper needs:
  * evaluations per cell (median, max) -- the price of giving up the model;
  * whether the grid's loss sequence is monotone in rank at all, because
    bisection is only well defined on a monotone predicate.

Usage:
  python3 sb_bisect.py --out bisect.json
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

from sb_acceptance import EPS                                          # noqa: E402
from sb_common import OUT                                              # noqa: E402
from sb_lodo_true import load_cells                                    # noqa: E402


def log(m):
    print(m, flush=True)


def bisect(delta, eps):
    """Smallest index with delta <= eps by halving; returns (index, probes, ok)."""
    n = len(delta)
    if delta[-1] > eps:
        return n - 1, int(np.ceil(np.log2(n))), False
    lo, hi, probes = 0, n - 1, 0
    while lo < hi:
        mid = (lo + hi) // 2
        probes += 1
        if delta[mid] <= eps:
            hi = mid
        else:
            lo = mid + 1
    return lo, probes, True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='bisect.json')
    ap.add_argument('--eps', type=float, default=EPS)
    args = ap.parse_args()

    C = load_cells()
    rows = []
    for c in C:
        i_hat, probes, ok = bisect(c['delta'], args.eps)
        r_hat = int(c['grid'][i_hat])
        # the measured crossing: first index whose measured loss is within tolerance
        ok_idx = np.where(c['delta'] <= args.eps)[0]
        i_true = int(ok_idx[0]) if len(ok_idx) else len(c['delta']) - 1
        met = bool(c['delta'][i_hat] <= args.eps)
        rows.append({'id': c['id'], 'rank': r_hat, 'r_true': int(c['r_true']),
                     'overhead': float(r_hat / max(c['r_true'], 1)),
                     'probes': int(probes), 'converged': bool(ok),
                     'exact_crossing': bool(i_hat == i_true),
                     'meet': met, 'n_ranks': int(len(c['delta'])),
                     'monotone': bool(np.all(np.diff(c['delta']) <= 1e-9))})

    ov = np.array([r['overhead'] for r in rows])
    pr = np.array([r['probes'] for r in rows])
    res = {'eps': args.eps, 'n_cells': len(rows),
           'summary': {'coverage': float(np.mean([r['meet'] for r in rows])),
                       'median_overhead': float(np.median(ov)),
                       'p95_overhead': float(np.quantile(ov, .95)),
                       'max_overhead': float(np.max(ov)),
                       'median_probes': float(np.median(pr)),
                       'max_probes': int(np.max(pr)),
                       'frac_exact_crossing': float(np.mean([r['exact_crossing'] for r in rows])),
                       'frac_monotone_grids': float(np.mean([r['monotone'] for r in rows]))},
           'rows': rows}
    json.dump(res, open(os.path.join(OUT, args.out), 'w'), indent=1)

    s = res['summary']
    log('bisection over %d cells: coverage %.0f%%, median overhead %.2fx, p95 %.2fx, '
        'max %.2fx, median probes %.0f (max %d), exact crossing %.0f%%, monotone grids %.0f%%'
        % (len(rows), 100 * s['coverage'], s['median_overhead'], s['p95_overhead'],
           s['max_overhead'], s['median_probes'], s['max_probes'],
           100 * s['frac_exact_crossing'], 100 * s['frac_monotone_grids']))
    bad = [r['id'] for r in rows if not r['exact_crossing']]
    if bad:
        log('cells where bisection did not land on the measured crossing: %s' % bad)


if __name__ == '__main__':
    main()
