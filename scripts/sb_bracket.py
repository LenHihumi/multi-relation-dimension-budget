"""Adaptive bracketing: turn the two-point rule into an algorithm instead of a precondition.

Reviewers' objection to the deployed rule is structural: the two prescribed points only work if
they bracket the crossing, arbitrary placement reaches the 89% point for a quarter of pairs, and
the paper's answer is a three-way branch rather than a procedure.  This script tests the obvious
repair, using only the measured curves (no retrieval):

    probe at rho = 0.04 and 0.12
    while not bracketed and budget left:
        both under tolerance  -> the crossing is at a smaller rho: probe at rho/3
        both over  tolerance  -> the crossing is at a larger rho:  probe at rho*2.5
    fit the log-logistic to the probed points (0.90 residual margin from the training corpora)
    issue the smallest rank whose predicted Delta <= eps

Budget is capped at three evaluations, which is the number the paper already reports as the
median for its three-branch arm (2.2).  The comparison is against the fixed pair with the same
margin, so the only difference is the policy.

Reports coverage, median/p95 overhead, mean evaluations and the fraction of cells bracketed
within budget.

Scope warning: this harness reproduces the paper's *coverage* (19/19 at the fixed pair) but not
its p95 (it reports 5.4x where Table 6 reports 2.75x), because the margin and the branch are
re-implemented here rather than imported from the Table 6 arm.  Compare coverage and evaluation
count across the two policies; do not compare p95 against the paper.

Usage
  python3 sb_bracket.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sb_common                                                       # noqa: E402

if os.environ.get('SB_OUT'):
    sb_common.OUT = os.environ['SB_OUT']

from sb_acceptance import EPS, fit_ll                                   # noqa: E402
from sb_lodo_true import curve, eval_rank, load_cells, rank_for        # noqa: E402

TARGETS = (0.04, 0.12)
BUDGET = 3


def idx_for(cell, rho_target):
    return int(np.argmin(np.abs(np.log(np.maximum(cell['rho'], 1e-9)) - np.log(rho_target))))


def margin(tr, eps):
    resid = []
    for c in tr:
        idx = [idx_for(c, t) for t in TARGETS]
        if idx[0] == idx[1]:
            continue
        r0, s = fit_ll(c['rho'][idx], c['delta'][idx])
        resid.extend((c['delta'][idx] - curve(c['rho'][idx], r0, s)).tolist())
    return min(max(0.0, float(np.quantile(resid, 0.90))) if resid else 0.0, 0.5 * eps)


def policy(cell, eps, bias, adaptive=True):
    """Return (issued_rank, n_evals, bracketed)."""
    rho = cell['rho']
    probed = [TARGETS[0], TARGETS[1]]
    while True:
        idx = sorted({idx_for(cell, t) for t in probed})
        d = [float(cell['delta'][i]) for i in idx]
        under = [x <= eps for x in d]
        if len(idx) >= 2 and any(under) and not all(under):
            break                                        # bracketed
        if not adaptive or len(probed) >= BUDGET or len(idx) < 2:
            break
        lo, hi = min(probed), max(probed)
        if all(under):
            probed.append(lo / 3.0)                      # crossing at smaller rho
        else:
            probed.append(min(hi * 2.5, 0.95))           # crossing at larger rho
        if len({idx_for(cell, t) for t in probed}) == len(idx):
            break                                        # the new target snapped back
    idx = sorted({idx_for(cell, t) for t in probed})
    d = [float(cell['delta'][i]) for i in idx]
    if all(x <= eps for x in d):
        # both anchors already under tolerance: the fit cannot locate a crossing, so issue a
        # measured rank (the paper's third branch) instead of extrapolating
        r_hat = max(int(cell['grid'][i]) for i in idx)
        return r_hat, len(probed), 0
    r0, s = fit_ll(rho[idx], cell['delta'][idx])
    du = np.clip(curve(rho, r0, s) + bias, 0, 1)
    r_hat = rank_for(rho, du, cell['grid'], eps)
    return r_hat, len(probed), int(any(x <= eps for x in d) and not all(x <= eps for x in d))


def run(C, eps, adaptive):
    rows = []
    for g in sorted({c['ds'] for c in C}):
        tr = [c for c in C if c['ds'] != g]
        te = [c for c in C if c['ds'] == g]
        bias = margin(tr, eps)
        for c in te:
            r_hat, n_ev, br = policy(c, eps, bias, adaptive)
            meet, ov = eval_rank(c, r_hat)
            rows.append({'id': c['id'], 'meet': meet, 'overhead': ov,
                         'evals': n_ev, 'bracketed': br})
    meet = np.array([r['meet'] for r in rows], float)
    ov = np.array([r['overhead'] for r in rows], float)
    ev = np.array([r['evals'] for r in rows], float)
    return {'n': len(rows), 'coverage': float(meet.mean()),
            'median_overhead': float(np.median(ov)),
            'p95_overhead': float(np.quantile(ov, .95)),
            'mean_evals': float(ev.mean()),
            'frac_bracketed': float(np.mean([r['bracketed'] for r in rows])),
            'misses': [r['id'] for r in rows if not r['meet']]}


def main():
    C = load_cells()
    print('cells %d  eps %.3f  targets %s  budget %d'
          % (len(C), EPS, TARGETS, BUDGET))
    for name, adaptive in (('fixed pair', False), ('adaptive brackets', True)):
        s = run(C, EPS, adaptive)
        print('%-18s cover %3.0f%%  median %.2fx  p95 %.2f  evals %.1f  bracketed %3.0f%%  misses %s'
              % (name, 100 * s['coverage'], s['median_overhead'], s['p95_overhead'],
                 s['mean_evals'], 100 * s['frac_bracketed'], s['misses'] or 'none'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
