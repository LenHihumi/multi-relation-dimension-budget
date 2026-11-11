"""Two review items that cost no retrieval, both computed from the shipped grid.

A. leave-one-cell-out placement.  The paper's 100% coverage at 1.20x comes from a pair of
   measured ranks (rho = 0.04, 0.12) chosen while looking at the 19 cells it is scored on.
   The fair test is: pick the pair using only the OTHER cells, then score it on the held-out
   one.  This enumerates the 253 rank pairs, scores every (pair, cell) with the same
   leave-one-dataset-out margin as `sb_2pt_sensitivity.py`, and then applies the selection
   rule out of fold:

       pair* = argmax coverage on the training cells, tie-break on p95 overhead

   Two references are reported beside it: the prescribed pair (in-sample), and the best pair
   in-sample (also in-sample, i.e. the optimistic end).

B. tolerance sweep.  Table 7 compares rules at eps = 0.05 only.  For each eps in 1..15% we
   re-derive the oracle rank from the measured curve and score the two label-free spectral
   rules against it.  This is where "spectral rules fail systematically" gets an interval:
   the eps at which energy-90 and 0.25d coverage come back to 100% is the edge of the claim.

Neither item is in the 310 numeric checks; both read `metric_std_parts` and `theory` only.

Usage
  python3 sb_review_checks.py placement [--folds 19]
  python3 sb_review_checks.py tolerance
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sb_common                                                       # noqa: E402

if os.environ.get('SB_OUT'):
    sb_common.OUT = os.environ['SB_OUT']

from sb_acceptance import EPS, fit_ll                                   # noqa: E402
from sb_common import OUT                                              # noqa: E402
from sb_lodo_true import curve, eval_rank, load_cells, rank_for        # noqa: E402

BOUND_R0 = (1e-4, 1e6)
BOUND_S = (0.1, 20.0)


def at_bound(r0, s):
    return ((r0 <= BOUND_R0[0] * (1 + 1e-9)) or (r0 >= BOUND_R0[1] * (1 - 1e-9))
            or (s <= BOUND_S[0] * (1 + 1e-9)) or (s >= BOUND_S[1] * (1 - 1e-9)))


# ---------------------------------------------------------------- A. placement ----
def placement(eps=EPS, verbose=True, three_branch=False):
    C = load_cells()
    common = set(int(r) for r in C[0]['grid'])
    for c in C[1:]:
        common &= set(int(r) for r in c['grid'])
    common = sorted(common)
    cells = [c['id'] for c in C]
    idx = {c['id']: c for c in C}

    def rank_index(c, r):
        return int(np.where(c['grid'] == r)[0][0])

    # M[(ra, rb)][cell] = (meet, overhead, straddle)
    M = {}
    for ra, rb in itertools.combinations(common, 2):
        row = {}
        for g in sorted({c['ds'] for c in C}):
            tr = [c for c in C if c['ds'] != g]
            te = [c for c in C if c['ds'] == g]
            resid = []
            for c in tr:
                i, j = rank_index(c, ra), rank_index(c, rb)
                r0, s = fit_ll(c['rho'][[i, j]], c['delta'][[i, j]])
                resid.extend((c['delta'][[i, j]] - curve(c['rho'][[i, j]], r0, s)).tolist())
            bias = max(0.0, float(np.quantile(resid, 0.90))) if resid else 0.0
            for c in te:
                i, j = rank_index(c, ra), rank_index(c, rb)
                if i == j:
                    row[c['id']] = (0, 1e3, 0)
                    continue
                r0, s = fit_ll(c['rho'][[i, j]], c['delta'][[i, j]])
                du = np.clip(curve(c['rho'], r0, s) + bias, 0, 1)
                da, db = float(c['delta'][i]), float(c['delta'][j])
                if three_branch and da <= eps and db <= eps:
                    # both anchors already under tolerance: the fit cannot locate a crossing,
                    # so issue a measured rank -- the larger of the two -- instead (Sec. 5.2).
                    r_hat = max(int(c['grid'][i]), int(c['grid'][j]))
                else:
                    r_hat = rank_for(c['rho'], du, c['grid'], eps)
                meet, ov = eval_rank(c, r_hat)
                row[c['id']] = (meet, ov, int(min(da, db) <= eps <= max(da, db)))
        M[(ra, rb)] = row

    def score(pair, subset):
        m = np.array([M[pair][c][0] for c in subset], float)
        o = np.array([M[pair][c][1] for c in subset], float)
        return float(m.mean()), float(np.quantile(o, .95)), float(np.median(o))

    # in-sample best, for the optimistic reference
    allscore = {p: score(p, cells) for p in M}
    best_in = max(allscore, key=lambda p: (allscore[p][0], -allscore[p][1], -allscore[p][2]))

    # leave-one-cell-out
    held = []
    for cid in cells:
        train = [c for c in cells if c != cid]
        best = max(M, key=lambda p: (score(p, train)[0], -score(p, train)[1],
                                     -score(p, train)[2]))
        meet, ov, _ = M[best][cid]
        held.append((cid, best, meet, ov))
    cov = float(np.mean([h[2] for h in held]))
    med = float(np.median([h[3] for h in held]))
    p95 = float(np.quantile([h[3] for h in held], .95))

    # prescribed pair, as a reference: nearest ranks to rho 0.04 / 0.12 in every cell
    pre = []
    for c in C:
        i = int(np.argmin(np.abs(np.log(np.maximum(c['rho'], 1e-9)) - np.log(0.04))))
        j = int(np.argmin(np.abs(np.log(np.maximum(c['rho'], 1e-9)) - np.log(0.12))))
        pre.append((int(c['grid'][i]), int(c['grid'][j])))
    pre_cov = float(np.mean([M[p][c['id']][0] if p in M else 0
                             for p, c in zip(pre, C)])) if all(p in M for p in pre) else float('nan')

    out = {'eps': eps, 'n_cells': len(C), 'n_pairs': len(M),
           'in_sample_best': {'pair': best_in, 'coverage': allscore[best_in][0],
                              'p95_overhead': allscore[best_in][1]},
           'loo_coverage': cov, 'loo_median_overhead': med, 'loo_p95_overhead': p95,
           'prescribed_pair_coverage': pre_cov,
           'pair_chosen_per_fold': {h[0]: list(h[1]) for h in held}}
    if verbose:
        print('cells %d   pairs %d   eps %.3f' % (len(C), len(M), eps))
        print('in-sample best pair %s: coverage %.1f%%, p95 %.2fx'
              % (list(best_in), 100 * allscore[best_in][0], allscore[best_in][1]))
        print('prescribed pair coverage (in-sample): %.1f%%' % (100 * pre_cov))
        print('leave-one-cell-out: coverage %.1f%%  median %.2fx  p95 %.2fx'
              % (100 * cov, med, p95))
        miss = [(h[0], list(h[1])) for h in held if not h[2]]
        print('holds that miss (%d): %s' % (len(miss), miss if miss else 'none'))
    return out


# ---------------------------------------------------------------- B. tolerance ----
def energy_rank(lam, share):
    tot, c = float(sum(lam)), 0.0
    for i, x in enumerate(lam, start=1):
        c += x
        if c / tot >= share:
            return i
    return len(lam)


def tolerance(eps_grid=None):
    C = load_cells()
    th = {}
    for pat in ('theory__*.json', os.path.join('theory', 'theory__*.json')):
        for p in glob.glob(os.path.join(OUT, pat)):
            d = json.load(open(p))
            th[(d['encoder'], d['dataset'])] = d
    if not th:
        print('no theory files under %s' % OUT)
        return {'error': 'no theory'}
    eps_grid = eps_grid or [round(x, 3) for x in np.arange(0.01, 0.151, 0.01)]

    rows = []
    for eps in eps_grid:
        cov90 = cov25 = cov_pt = n_oracle = 0
        ov90, ov25, ov_pt = [], [], []
        for c in C:
            # oracle rank at this tolerance, from the measured curve
            hit = np.where(c['delta'] <= eps)[0]
            if len(hit) == 0:
                continue                      # no rank reaches eps: no oracle to compare to
            n_oracle += 1
            r_or = float(c['grid'][hit[0]])
            t = th.get((c['enc'], c['ds']))
            if t is None:
                raise KeyError('no theory for cell %s' % c['id'])
            lam = t['lambda']
            r90 = energy_rank(lam, 0.90)
            r25 = 0.25 * t['dim']
            for r, acc, ovs in ((r90, 'a', ov90), (r25, 'b', ov25)):
                j = int(np.argmin(np.abs(c['grid'] - r)))
                ok = c['delta'][j] <= eps
                if acc == 'a':
                    cov90 += int(ok)
                else:
                    cov25 += int(ok)
                ovs.append(float(c['grid'][j] / max(r_or, 1)))
            # the measured arm at this tolerance: prescribed rho targets, 2 points, and the
            # branch that issues a measured rank when both anchors are already under eps
            rho = np.maximum(c['rho'], 1e-12)
            i = int(np.argmin(np.abs(np.log(rho) - np.log(0.04))))
            j = int(np.argmin(np.abs(np.log(rho) - np.log(0.12))))
            r0, s = fit_ll(rho[[i, j]], c['delta'][[i, j]])
            da, db = float(c['delta'][i]), float(c['delta'][j])
            if da <= eps and db <= eps:
                r_hat = max(int(c['grid'][i]), int(c['grid'][j]))
            else:
                r_hat = rank_for(rho, np.clip(curve(rho, r0, s), 0, 1), c['grid'], eps)
            jj = int(np.argmin(np.abs(c['grid'] - r_hat)))
            cov_pt += int(c['delta'][jj] <= eps)
            ov_pt.append(float(c['grid'][jj] / max(r_or, 1)))
        rows.append({'eps': eps, 'n_cells_with_oracle': n_oracle,
                     'coverage_energy90': cov90 / max(n_oracle, 1),
                     'median_overhead_energy90': float(np.median(ov90)) if ov90 else None,
                     'coverage_ff25': cov25 / max(n_oracle, 1),
                     'median_overhead_ff25': float(np.median(ov25)) if ov25 else None,
                     'coverage_measured': cov_pt / max(n_oracle, 1),
                     'median_overhead_measured': float(np.median(ov_pt)) if ov_pt else None})
    print('%-6s %6s %9s %8s %9s %8s %9s %8s'
          % ('eps', 'cells', 'e90 cov', 'e90 med', 'ff25 cov', 'ff25 med',
             'meas cov', 'meas med'))
    for r in rows:
        print('%-6.2f %6d %8.0f%% %7.2fx %8.0f%% %7.2fx %8.0f%% %7.2fx'
              % (r['eps'], r['n_cells_with_oracle'],
                 100 * r['coverage_energy90'], r['median_overhead_energy90'] or float('nan'),
                 100 * r['coverage_ff25'], r['median_overhead_ff25'] or float('nan'),
                 100 * r['coverage_measured'], r['median_overhead_measured'] or float('nan')))
    return {'eps_grid': eps_grid, 'rows': rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('what', choices=['placement', 'tolerance'])
    ap.add_argument('--eps', type=float, default=EPS)
    ap.add_argument('--three-branch', action='store_true')
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    res = (placement(a.eps, three_branch=a.three_branch) if a.what == 'placement'
           else tolerance())
    if a.out:
        with open(a.out, 'w') as f:
            json.dump(res, f, indent=1, default=float)
        print('wrote', a.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
