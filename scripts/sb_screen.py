"""Q3: is the Sec. 7 screening rule operational, or only a heuristic?

Sec. 7 tells a practitioner to compare a new corpus's effective rank and spectral
entropy against the range spanned by the 19 released cells, and to recalibrate
outside it. That rule has no error rate attached. This script measures the one
error rate we can measure without new corpora: leave one cell out, build the span
from the other 18, and see whether the held-out cell falls inside.

If almost nothing falls outside, the rule is near-vacuous -- which is the honest
reading, and the reason the paper should call it a heuristic.

Usage:
  python3 sb_screen.py --out screen.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sb_common                                                       # noqa: E402

if os.environ.get('SB_OUT'):
    sb_common.OUT = os.environ['SB_OUT']

from sb_common import OUT                                              # noqa: E402

EPS = 0.05


def log(m):
    print(m, flush=True)


def d_eff(w):
    return float(w.sum() ** 2 / max((w ** 2).sum(), 1e-12))


def entropy(w):
    p = w / max(w.sum(), 1e-12)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='screen.json')
    args = ap.parse_args()

    theory = {}
    paths = (glob.glob(os.path.join(OUT, 'theory__*.json'))
             + glob.glob(os.path.join(OUT, 'theory', '*.json')))
    for p in paths:
        d = json.load(open(p))
        theory[(d['encoder'], d['dataset'])] = d

    cells = []
    mpaths = sorted(glob.glob(os.path.join(OUT, 'metric_std_parts', '*.json')))
    for p in mpaths:
        d = json.load(open(p))
        th = theory.get((d['encoder'], d['dataset']))
        if th is None:
            continue
        lam = np.array(th['lambda'], float)
        grid = np.array([int(r) for r in d['grid']], int)
        for rel, cell in d.get('cells', {}).items():
            mm = cell.get('metrics', {}).get('map10')
            if not mm or 'curve' not in mm or rel not in th.get('tasks', {}):
                continue
            n = min(len(mm['curve']), len(cell.get('rho', [])), len(grid))
            delta = 1.0 - np.array(mm['curve'][:n], float) / float(cell['full']['map10'])
            ok = np.where(delta <= EPS)[0]
            if not len(ok):
                continue
            r = int(grid[ok[0]])
            eta = np.array(th['tasks'][rel]['eta2'], float)
            w = (lam * eta)[:r]
            cells.append({'id': '%s/%s/%s' % (d['encoder'], d['dataset'], rel),
                          'log_deff': float(np.log(d_eff(w))),
                          'entropy': entropy(w),
                          'rank': r})
    log('cells with a target rank: %d' % len(cells))

    for c in cells:
        rest = [o for o in cells if o['id'] != c['id']]
        lo_d = min(o['log_deff'] for o in rest)
        hi_d = max(o['log_deff'] for o in rest)
        lo_e = min(o['entropy'] for o in rest)
        hi_e = max(o['entropy'] for o in rest)
        c['inside'] = bool(lo_d <= c['log_deff'] <= hi_d and lo_e <= c['entropy'] <= hi_e)

    outside = [c['id'] for c in cells if not c['inside']]
    res = {'n_cells': len(cells),
           'flag_rate': len(outside) / len(cells),
           'flagged': outside,
           'note': ('Leave-one-cell-out span of (log effective rank, spectral entropy) at the '
                    'measured budget. Cells outside the span are the ones the rule would send '
                    'back for recalibration.'),
           'cells': cells}
    json.dump(res, open(os.path.join(OUT, args.out), 'w'), indent=1)
    log('leave-one-out flag rate %.0f%% (%d/%d): %s'
        % (100 * res['flag_rate'], len(outside), len(cells), outside))


if __name__ == '__main__':
    main()
