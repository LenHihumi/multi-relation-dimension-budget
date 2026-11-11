"""How much does the 78.6% coverage number move if the delta-method SE bias is
rescaled?  A zero-cost check on data that already exists locally.

`sb_resolution.py` reports expected_coverage = 78.6% from a delta-method SE that
under-estimates the bootstrap SE by 1.19x-1.60x on the three cells where both are
available.  It states the direction of the bias but does not propagate it.  This
script does two things, neither of which needs the GPU box:

1. Rescale every cell's SE by each measured ratio (and the mean) and recompute
   expected coverage, so the 78.6% becomes an interval rather than a point.

2. Compare the delta-method P(flip) against the bootstrap P(flip) directly on the
   three measured cells.  P(flip) -- not SE -- is the quantity that enters the
   coverage number, and the map from SE to P(flip) saturates in the tails, so the
   SE bias need not translate one-for-one into a coverage error.  Reporting both
   shows whether rescaling actually improves the estimate or overshoots.

Usage: python sb_resolution_sens.py --out resolution_sens.json
"""
from __future__ import annotations

import argparse
import json
import os
from math import erf, exp, pi, sqrt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, 'data')


def Phi(x):
    return 0.5 * (1 + erf(x / sqrt(2)))


def phi(x):
    return exp(-x * x / 2) / sqrt(2 * pi)


def sensitivity(z):
    """d P(flip) / d ln(scale) = |z| * phi(z): peaks at |z|=1, zero at both ends."""
    return abs(z) * phi(z)


def coverage(rows, scale, eps):
    """Expected coverage if every SE is multiplied by `scale`."""
    p = [Phi(-r['margin'] / (r['se_delta'] * scale)) for r in rows]
    return 1.0 - sum(p) / len(p), p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='resolution_sens.json')
    args = ap.parse_args()

    res = json.load(open(os.path.join(DATA, 'results', 'resolution.json')))
    eps, rows = res['eps'], res['rows']
    cal = res['calibration']
    ratios = {k: v['ratio'] for k, v in cal.items()}

    lo, hi = min(ratios.values()), max(ratios.values())
    mean_r = sum(ratios.values()) / len(ratios)
    scales = [('no rescale (as published)', 1.0),
              ('min measured ratio', lo),
              ('mean measured ratio', mean_r),
              ('max measured ratio', hi)]

    out = {'eps': eps, 'n_cells': len(rows),
           'calibration_ratios': ratios,
           'ratio_spread': hi / lo,
           'scales': []}
    print('calibration ratios (bootstrap SE / delta SE):')
    for k, v in sorted(ratios.items(), key=lambda kv: kv[1]):
        print('  %-30s %.3f' % (k[:30], v))
    print('  spread %.3f -> %.3f  (max/min = %.2fx)\n'
          % (lo, hi, hi / lo))

    print('%-30s %8s %8s %8s' % ('rescale', 'coverage', 'flips', 'delta'))
    for name, s in scales:
        cov, p = coverage(rows, s, eps)
        out['scales'].append({'name': name, 'scale': s, 'coverage': cov,
                              'expected_flips': (1 - cov) * len(rows)})
        print('%-30s %7.1f%% %8.2f %+8.1f pp'
              % (name, 100 * cov, (1 - cov) * len(rows), 100 * (cov - res['expected_coverage'])))

    # Per-cell sensitivity: which cells actually drive the movement?
    base = {r['id']: r['p_flip_delta'] for r in rows}
    top = []
    for sid in base:
        row = next(r for r in rows if r['id'] == sid)
        _, p_hi = coverage([row], hi, eps)
        top.append((sid, base[sid], p_hi[0], p_hi[0] - base[sid],
                    row['margin'] / row['se_delta']))
    top.sort(key=lambda t: -t[3])
    out['per_cell'] = [{'id': t[0], 'p_flip': t[1], 'p_flip_at_max_ratio': t[2],
                        'shift': t[3], 'margin_over_se': t[4],
                        'sensitivity': sensitivity(t[4])} for t in top]
    print('\nmost SE-sensitive cells (at max measured ratio):')
    print('%-30s %8s %8s %8s %8s %8s'
          % ('cell', 'p_flip', 'rescaled', 'shift', 'margin/SE', '|z|phi(z)'))
    for t in top[:6]:
        print('%-30s %8.3f %8.3f %+8.3f %8.2f %8.3f'
              % (t[0][:30], t[1], t[2], t[3], t[4], sensitivity(t[4])))

    # Direct check: delta P(flip) vs bootstrap P(flip) at n=1000.
    fid = json.load(open(os.path.join(DATA, 'fidelity', 'fidelity.json')))
    boot = {}
    for k, v in fid['cells'].items():
        boot[k] = v['by_size']['1000']['p_flip']
    out['p_flip_check'] = []
    print('\nP(flip) at n=1000: delta vs bootstrap (the quantity that enters coverage):')
    print('%-30s %8s %10s %8s %8s %8s %8s'
          % ('cell', 'delta', 'bootstrap', 'err', 'min-r', 'max-r', '|z|phi(z)'))
    for k, b in boot.items():
        r = next(x for x in rows if x['id'] == k.replace('__', '/'))
        z = r['margin'] / r['se_delta']
        p_min = Phi(-z / lo)
        p_max = Phi(-z / hi)
        out['p_flip_check'].append({'id': k, 'delta': r['p_flip_delta'], 'bootstrap': b,
                                    'err': r['p_flip_delta'] - b,
                                    'at_min_ratio': p_min, 'at_max_ratio': p_max,
                                    'margin_over_se': z, 'sensitivity': sensitivity(z)})
        print('%-30s %8.3f %10.3f %+8.3f %8.3f %8.3f %8.3f'
              % (k[:30], r['p_flip_delta'], b, r['p_flip_delta'] - b, p_min, p_max,
                 sensitivity(z)))

    with open(os.path.join(DATA, 'results', args.out), 'w') as f:
        json.dump(out, f, indent=1)
    print('\nwrote %s' % os.path.join(DATA, 'results', args.out))


if __name__ == '__main__':
    main()
