"""Q5: does ONE probe at the fitted crossing buy a measured budget?

The two-point rule issues a rank from a fitted curve plus a reference margin.
Two ways for it to be wrong: the fit is confident and too small (quality loss,
unbounded), or the fit is unidentifiable and the parameter bound dumps it at the
full dimension (storage, 28.9x at p95).  A reviewer proposed the cheap repair:
fit two points, probe once AT THE FITTED CROSSING, and if the measured loss
there exceeds the tolerance, ship the full dimension instead.

That is three evaluations, and it replaces "the bound fired" with "I looked".
This script scores it on the measured 23-rank grids, no new retrieval:

  m2    two points, bounded log-logistic fit + reference margin  (published)
  m2p   m2 + one probe at the issued rank, fallback to full dimension
  u2    two points, UNBOUNDED line + margin  (what a practitioner without our
        parameter bounds would get; it under-provisions on two cells)
  u2p   u2 + one probe  --  the arm the reviewer asked for

It also prices the assumption under the bisection row: bisection needs a bracket
that already contains the crossing.  We re-run it from brackets anchored at
d/2, d/4 and d/8 and report what that assumption costs, because the "5
evaluations" in the paper are conditional on the bracket, not unconditional.

Usage:
  python3 sb_probe.py --out probe.json
"""
from __future__ import annotations

import argparse
import csv
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
from sb_lodo_true import curve, load_cells, rank_for                   # noqa: E402

PRIOR = (0.04, 0.12)
B_S = (0.1, 20.0)
B_R0 = (1e-4, 1e6)


def log(m):
    print(m, flush=True)


def fit_line(rho, delta, clipped):
    """Two-point log-logistic fit; `clipped` applies the paper's parameter bounds."""
    y = np.clip(np.asarray(delta, float), 1e-4, 1 - 1e-4)
    x = np.log(np.maximum(rho, 1e-9))
    z = np.log(y / (1 - y))
    A = np.vstack([x, np.ones_like(x)]).T
    b, a = np.linalg.lstsq(A, z, rcond=None)[0]
    s = float(b)
    if clipped:
        s = float(np.clip(s, *B_S))
    log_r0 = float(-a / s)
    if clipped:
        log_r0 = float(np.clip(log_r0, np.log(B_R0[0]), np.log(B_R0[1])))
    return float(np.exp(log_r0)), s


def idx_for(c, ta, tb):
    ia = int(np.argmin(np.abs(np.log(np.maximum(c['rho'], 1e-9)) - np.log(ta))))
    ib = int(np.argmin(np.abs(np.log(np.maximum(c['rho'], 1e-9)) - np.log(tb))))
    return (ia, ib) if ia != ib else None


def score(c, r_issued):
    j = int(np.argmin(np.abs(c['grid'] - r_issued)))
    r = int(c['grid'][j])
    return (1 if c['delta'][j] <= EPS else 0), float(r / max(c['r_true'], 1)), r


def run(C, clipped, probe, eps=EPS):
    rows = []
    for g in sorted({c['ds'] for c in C}):
        tr = [c for c in C if c['ds'] != g]
        te = [c for c in C if c['ds'] == g]
        resid = []
        for c in tr:
            ij = idx_for(c, *PRIOR)
            if ij is None:
                continue
            i, j = ij
            r0, s = fit_line(c['rho'][[i, j]], c['delta'][[i, j]], clipped)
            resid.extend((c['delta'][[i, j]] - curve(c['rho'][[i, j]], r0, s)).tolist())
        bias = max(0.0, float(np.quantile(resid, 0.90))) if resid else 0.0
        for c in te:
            ij = idx_for(c, *PRIOR)
            if ij is None:
                rows.append({'id': c['id'], 'meet': 0, 'overhead': 1e3, 'issued': -1,
                             'fire': 0, 'd_probe': 1.0, 'r_hat': -1})
                continue
            i, j = ij
            r0, s = fit_line(c['rho'][[i, j]], c['delta'][[i, j]], clipped)
            du = np.clip(curve(c['rho'], r0, s) + bias, 0, 1)
            r_hat = rank_for(c['rho'], du, c['grid'], eps)
            fire = 0
            d_probe = float(c['delta'][int(np.argmin(np.abs(c['grid'] - r_hat)))])
            if probe and d_probe > eps:
                fire = 1
                r_hat = int(c['grid'][-1])
            meet, ov, issued = score(c, r_hat)
            rows.append({'id': c['id'], 'meet': meet, 'overhead': ov, 'issued': issued,
                         'fire': fire, 'd_probe': d_probe,
                         'r_hat': int(r_hat)})
    meet = np.array([r['meet'] for r in rows], float)
    ov = np.array([r['overhead'] for r in rows], float)
    return {'n': len(rows),
            'evals': 3 if probe else 2,
            'coverage': float(meet.mean()),
            'median_overhead': float(np.median(ov)),
            'p95_overhead': float(np.quantile(ov, .95)),
            'max_overhead': float(np.max(ov)),
            'n_fire': int(sum(r['fire'] for r in rows)),
            'fire_cells': [r['id'] for r in rows if r['fire']],
            'miss_cells': [r['id'] for r in rows if not r['meet']],
            'max_loss_at_issued': float(max(
                c['delta'][int(np.argmin(np.abs(c['grid'] - r['issued'])))]
                for c, r in zip([x for x in C], rows) if r['issued'] > 0))}, rows


def run_guard(C, eps=EPS):
    """Two points plus the free check: if BOTH anchors are already under
    tolerance the crossing lies below both, the shape is unconstrained, and the
    honest budget is the smaller measured rank -- not an extrapolation to the
    full dimension.  Costs no extra evaluation: the two measurements decide it.
    """
    rows = []
    for g in sorted({c['ds'] for c in C}):
        tr = [c for c in C if c['ds'] != g]
        te = [c for c in C if c['ds'] == g]
        resid, anchors = [], {}
        for c in tr:
            ij = idx_for(c, *PRIOR)
            if ij is None:
                continue
            i, j = ij
            r0, s = fit_line(c['rho'][[i, j]], c['delta'][[i, j]], True)
            resid.extend((c['delta'][[i, j]] - curve(c['rho'][[i, j]], r0, s)).tolist())
        bias = max(0.0, float(np.quantile(resid, 0.90))) if resid else 0.0
        for c in te:
            ij = idx_for(c, *PRIOR)
            if ij is None:
                rows.append({'id': c['id'], 'meet': 0, 'overhead': 1e3,
                             'issued': -1, 'guarded': 0})
                continue
            i, j = ij
            da, db = float(c['delta'][i]), float(c['delta'][j])
            if max(da, db) <= eps:                       # both converged
                r_hat, guard = int(min(c['grid'][i], c['grid'][j])), 1
            else:
                r0, s = fit_line(c['rho'][[i, j]], c['delta'][[i, j]], True)
                du = np.clip(curve(c['rho'], r0, s) + bias, 0, 1)
                r_hat, guard = rank_for(c['rho'], du, c['grid'], eps), 0
            meet, ov, issued = score(c, r_hat)
            rows.append({'id': c['id'], 'meet': meet, 'overhead': ov,
                         'issued': issued, 'guarded': guard})
    meet = np.array([r['meet'] for r in rows], float)
    ov = np.array([r['overhead'] for r in rows], float)
    return {'n': len(rows), 'evals': 2, 'coverage': float(meet.mean()),
            'median_overhead': float(np.median(ov)),
            'p95_overhead': float(np.quantile(ov, .95)),
            'max_overhead': float(np.max(ov)),
            'n_guard': int(sum(r['guarded'] for r in rows)),
            'guard_cells': [r['id'] for r in rows if r['guarded']],
            'miss_cells': [r['id'] for r in rows if not r['meet']]}, rows


def run_guard_bisect(C, eps=EPS):
    """Guard, then spend probes only where the guard fires.

    When both anchors are under tolerance we know the crossing is below the
    smaller one, so bisect inside [smallest rank, smaller anchor] instead of
    issuing the anchor itself.  Cells whose anchors straddle the crossing pay
    nothing extra.  This is the arm that prices "no tail at all".
    """
    rows = []
    for g in sorted({c['ds'] for c in C}):
        tr = [c for c in C if c['ds'] != g]
        te = [c for c in C if c['ds'] == g]
        resid = []
        for c in tr:
            ij = idx_for(c, *PRIOR)
            if ij is None:
                continue
            i, j = ij
            r0, s = fit_line(c['rho'][[i, j]], c['delta'][[i, j]], True)
            resid.extend((c['delta'][[i, j]] - curve(c['rho'][[i, j]], r0, s)).tolist())
        bias = max(0.0, float(np.quantile(resid, 0.90))) if resid else 0.0
        for c in te:
            ij = idx_for(c, *PRIOR)
            if ij is None:
                rows.append({'id': c['id'], 'meet': 0, 'overhead': 1e3,
                             'issued': -1, 'probes': 0})
                continue
            i, j = ij
            da, db = float(c['delta'][i]), float(c['delta'][j])
            if max(da, db) <= eps:
                hi = int(np.argmin(np.abs(c['grid'] - min(c['grid'][i], c['grid'][j]))))
                k, probes, _ok = bisect_from(c['delta'], eps, 0, hi)
                r_hat = int(c['grid'][k])
                used = 2 + probes
            else:
                r0, s = fit_line(c['rho'][[i, j]], c['delta'][[i, j]], True)
                du = np.clip(curve(c['rho'], r0, s) + bias, 0, 1)
                r_hat, used = rank_for(c['rho'], du, c['grid'], eps), 2
            meet, ov, issued = score(c, r_hat)
            rows.append({'id': c['id'], 'meet': meet, 'overhead': ov,
                         'issued': issued, 'probes': used})
    meet = np.array([r['meet'] for r in rows], float)
    ov = np.array([r['overhead'] for r in rows], float)
    ev = np.array([r['probes'] for r in rows], float)
    return {'n': len(rows), 'coverage': float(meet.mean()),
            'median_overhead': float(np.median(ov)),
            'p95_overhead': float(np.quantile(ov, .95)),
            'max_overhead': float(np.max(ov)),
            'median_evals': float(np.median(ev)), 'mean_evals': float(ev.mean()),
            'max_evals': int(ev.max()),
            'miss_cells': [r['id'] for r in rows if not r['meet']]}, rows


def classify_branches(C, eps=EPS):
    """Which of the three branches each cell reaches at the prescribed targets.

    The paper quotes the counts (8 under / 8 straddle / 3 over) and the rule's
    three branches are defined by them, so they are recorded here rather than
    left implicit in the per-cell CSV.
    """
    out = {'under': [], 'straddle': [], 'over': []}
    for c in C:
        ij = idx_for(c, *PRIOR)
        if ij is None:
            out['under'].append(c['id'])          # unidentifiable: no branch
            continue
        i, j = ij
        da, db = float(c['delta'][i]), float(c['delta'][j])
        if max(da, db) <= eps:
            out['under'].append(c['id'])
        elif min(da, db) > eps:
            out['over'].append(c['id'])
        else:
            out['straddle'].append(c['id'])
    return {k: v for k, v in out.items()}


def run_three_branch(C, eps=EPS, sharp=False, probe_c=False):
    """The deployed rule, end to end: A fit, B smaller measured anchor, C ladder.

    m2g above is the same rule with the fit kept in branch C, which is what the
    three cells that reach that branch happen to tolerate (short extrapolation).
    This arm replaces that extrapolation with measurement, so the table describes
    the rule the paper recommends rather than the lucky case of it.
    """
    rows = []
    for g in sorted({c['ds'] for c in C}):
        tr = [c for c in C if c['ds'] != g]
        te = [c for c in C if c['ds'] == g]
        resid = []
        for c in tr:
            ij = idx_for(c, *PRIOR)
            if ij is None:
                continue
            i, j = ij
            r0, s = fit_line(c['rho'][[i, j]], c['delta'][[i, j]], True)
            resid.extend((c['delta'][[i, j]] - curve(c['rho'][[i, j]], r0, s)).tolist())
        bias = max(0.0, float(np.quantile(resid, 0.90))) if resid else 0.0
        for c in te:
            ij = idx_for(c, *PRIOR)
            if ij is None:
                rows.append({'id': c['id'], 'meet': 0, 'overhead': 1e3,
                             'issued': -1, 'branch': 'none', 'probes': 0})
                continue
            i, j = ij
            da, db = float(c['delta'][i]), float(c['delta'][j])
            if max(da, db) <= eps:                       # B: both under
                if sharp:
                    hi = int(np.argmin(np.abs(c['grid'] - min(c['grid'][i], c['grid'][j]))))
                    k, pr, _ok = bisect_from(c['delta'], eps, 0, hi)
                    r_hat, used = int(c['grid'][k]), 2 + pr
                else:
                    r_hat, used = int(min(c['grid'][i], c['grid'][j])), 2
                branch = 'B'
            elif min(da, db) > eps:                      # C: both over
                n = len(c['grid'])
                k = max(i, j)
                used = 2
                if probe_c:
                    # fit as hypothesis: probe the fitted crossing once and let
                    # the measurement arbitrate; the ladder is the continuation.
                    r0, s = fit_line(c['rho'][[i, j]], c['delta'][[i, j]], True)
                    du = np.clip(curve(c['rho'], r0, s) + bias, 0, 1)
                    jf = int(np.argmin(np.abs(c['grid'] - rank_for(
                        c['rho'], du, c['grid'], eps))))
                    used += 1
                    if c['delta'][jf] <= eps:
                        r_hat, branch = int(c['grid'][jf]), 'C'
                        meet, ov, issued = score(c, r_hat)
                        rows.append({'id': c['id'], 'meet': meet, 'overhead': ov,
                                     'issued': issued, 'branch': branch,
                                     'probes': used})
                        continue
                while c['delta'][k] > eps and k < n - 1:
                    tgt = 2 * c['grid'][k]
                    k = min(int(np.searchsorted(c['grid'], tgt)), n - 1)
                    used += 1
                r_hat, branch = int(c['grid'][k]), 'C'
            else:                                        # A: straddle, fit
                r0, s = fit_line(c['rho'][[i, j]], c['delta'][[i, j]], True)
                du = np.clip(curve(c['rho'], r0, s) + bias, 0, 1)
                r_hat, branch, used = rank_for(c['rho'], du, c['grid'], eps), 'A', 2
            meet, ov, issued = score(c, r_hat)
            rows.append({'id': c['id'], 'meet': meet, 'overhead': ov,
                         'issued': issued, 'branch': branch, 'probes': used})
    meet = np.array([r['meet'] for r in rows], float)
    ov = np.array([r['overhead'] for r in rows], float)
    ev = np.array([r['probes'] for r in rows], float)
    return {'n': len(rows), 'coverage': float(meet.mean()),
            'median_overhead': float(np.median(ov)),
            'p95_overhead': float(np.quantile(ov, .95)),
            'max_overhead': float(np.max(ov)),
            'median_evals': float(np.median(ev)), 'mean_evals': float(ev.mean()),
            'max_evals': int(ev.max()),
            'by_branch': {b: {'n': sum(1 for r in rows if r['branch'] == b),
                              'median': float(np.median([r['overhead'] for r in rows
                                                         if r['branch'] == b])) if any(
                                  r['branch'] == b for r in rows) else None}
                          for b in ('A', 'B', 'C', 'none')},
            'miss_cells': [r['id'] for r in rows if not r['meet']]}, rows


def run_branch_c(C, eps=EPS, factor=2.0):
    """Third branch: BOTH anchors above tolerance, so the crossing is above both
    and every model answer is an extrapolation.

    Placement: the smallest grid rank and the largest rank that is still above
    tolerance -- a wide spread on the wrong side of the crossing.  This placement
    does not occur at the paper's two prescribed targets (every cell there has at
    least one anchor under tolerance), so the numbers below are a scenario, not a
    deployment observation.  They answer one question: when the fit has to
    extrapolate, is it better than measuring upward?

      fit+margin   bounded two-point fit, extrapolated, plus the reference margin
      ladder       probe 2r, 4r, ... from the larger anchor until the loss is
                   measured under tolerance, then issue that rank
      full dim     ship d
    """
    rows = []
    for g in sorted({c['ds'] for c in C}):
        tr = [c for c in C if c['ds'] != g]
        te = [c for c in C if c['ds'] == g]
        resid = []
        for c in tr:
            ij = idx_for(c, *PRIOR)
            if ij is None:
                continue
            i, j = ij
            r0, s = fit_line(c['rho'][[i, j]], c['delta'][[i, j]], True)
            resid.extend((c['delta'][[i, j]] - curve(c['rho'][[i, j]], r0, s)).tolist())
        bias = max(0.0, float(np.quantile(resid, 0.90))) if resid else 0.0
        for c in te:
            above = np.where(c['delta'] > eps)[0]
            if len(above) < 2:
                continue
            i = int(above[0])
            # the second anchor is the first measured rank at least `factor` times
            # the first that is still above tolerance; factor=None uses the widest
            # admissible pair, which puts one anchor adjacent to the crossing and
            # therefore lets the fit interpolate -- see the README trap.
            if factor is None:
                j = int(above[-1])
            else:
                cand = [k for k in above if c['grid'][k] >= factor * c['grid'][i]]
                j = int(cand[0]) if cand else int(above[-1])
            n = len(c['grid'])
            r0, s = fit_line(c['rho'][[i, j]], c['delta'][[i, j]], True)
            du = np.clip(curve(c['rho'], r0, s) + bias, 0, 1)
            r_fit = rank_for(c['rho'], du, c['grid'], eps)
            k, probes = j, 0
            while c['delta'][k] > eps and k < n - 1:
                tgt = 2 * c['grid'][k]
                k = int(np.searchsorted(c['grid'], tgt, side='left'))
                k = min(k, n - 1)
                probes += 1
            r_lad = int(c['grid'][k])
            rec = {'id': c['id'], 'anchor_a': int(c['grid'][i]),
                   'anchor_b': int(c['grid'][j]), 'probes': probes, 'r_true': c['r_true']}
            # fit as hypothesis, one probe as arbiter, ladder as continuation
            jf = int(np.argmin(np.abs(c['grid'] - r_fit)))
            if c['delta'][jf] <= eps:
                r_probe, pr_probe = int(c['grid'][jf]), 0
            else:
                k2, pr_probe = max(i, j), 1
                while c['delta'][k2] > eps and k2 < n - 1:
                    tgt = 2 * c['grid'][k2]
                    k2 = min(int(np.searchsorted(c['grid'], tgt)), n - 1)
                    pr_probe += 1
                r_probe = int(c['grid'][k2])
            rec['probes_probe'] = 1 + pr_probe
            for tag, r in (('fit', r_fit), ('ladder', r_lad), ('full', int(c['grid'][-1])),
                           ('fitprobe', r_probe)):
                m, ov, issued = score(c, r)
                rec[tag + '_meet'] = m
                rec[tag + '_ov'] = ov
                rec[tag + '_rank'] = issued
            rows.append(rec)

    def agg(tag):
        meet = np.array([r[tag + '_meet'] for r in rows], float)
        ov = np.array([r[tag + '_ov'] for r in rows], float)
        return {'coverage': float(meet.mean()),
                'median_overhead': float(np.median(ov)),
                'p95_overhead': float(np.quantile(ov, .95)),
                'max_overhead': float(np.max(ov)),
                'miss_cells': [r['id'] for r in rows if not r[tag + '_meet']]}

    pr = np.array([r['probes'] for r in rows], float)
    return {'n': len(rows), 'factor': factor,
            'anchor_spread_median': float(np.median(
                [np.log(r['anchor_b'] / r['anchor_a']) for r in rows])),
            'evals': {'fit': 2, 'ladder': 2 + float(np.median(pr)),
                      'ladder_max': 2 + int(pr.max())},
            'median_probes_ladder': float(np.median(pr)),
            'median_probes_fitprobe': float(np.median(
                [r['probes_probe'] for r in rows])),
            'arms': {'fit': agg('fit'), 'ladder': agg('ladder'),
                     'fitprobe': agg('fitprobe'), 'full': agg('full')}}, rows


def bisect_from(delta, eps, lo_idx=0, hi_idx=None):
    """Bisect on the grid restricted to [lo_idx, end]; bracket must contain the crossing."""
    n = len(delta)
    hi_idx = (n - 1) if hi_idx is None else int(hi_idx)
    sub = delta[lo_idx:hi_idx + 1]
    if len(sub) == 0 or sub[-1] > eps:
        return lo_idx + max(len(sub) - 1, 0), int(np.ceil(np.log2(max(len(sub), 1)))), False
    lo, hi, probes = 0, len(sub) - 1, 0
    while lo < hi:
        mid = (lo + hi) // 2
        probes += 1
        if sub[mid] <= eps:
            hi = mid
        else:
            lo = mid + 1
    return lo_idx + lo, probes, True


def bracket_scan(C, eps):
    out = {}
    for frac, name in ((0.0, 'full grid (rank 8..d)'), (0.5, 'd/2..d'),
                       (0.25, 'd/4..d'), (0.125, 'd/8..d')):
        rows = []
        for c in C:
            d = float(c['grid'][-1])
            lo_idx = int(np.sum(c['grid'] < frac * d)) if frac > 0 else 0
            i_hat, probes, _ok = bisect_from(c['delta'], eps, lo_idx)
            r = int(c['grid'][i_hat])
            rows.append({'id': c['id'], 'rank': r, 'probes': int(probes),
                         'meet': int(c['delta'][i_hat] <= eps),
                         'overhead': float(r / max(c['r_true'], 1))})
        meet = np.array([x['meet'] for x in rows], float)
        ov = np.array([x['overhead'] for x in rows], float)
        pr = np.array([x['probes'] for x in rows], float)
        out[name] = {'coverage': float(meet.mean()),
                     'median_overhead': float(np.median(ov)),
                     'p95_overhead': float(np.quantile(ov, .95)),
                     'max_overhead': float(np.max(ov)),
                     'median_probes': float(np.median(pr)),
                     'max_probes': int(pr.max())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='probe.json')
    ap.add_argument('--csv', default='probe_cells.csv')
    ap.add_argument('--eps', type=float, default=EPS)
    args = ap.parse_args()

    C = load_cells()
    log('cells=%d  r_true=%s' % (len(C), sorted({c['r_true'] for c in C})))

    res = {'eps': args.eps, 'n_cells': len(C), 'targets': list(PRIOR), 'arms': {}}
    per_cell = {}
    for name, clipped, probe in (('m2', True, False), ('m2p', True, True),
                                 ('u2', False, False), ('u2p', False, True)):
        s, rows = run(C, clipped, probe, args.eps)
        res['arms'][name] = s
        per_cell[name] = {r['id']: r for r in rows}
        log('%-4s evals=%d cov=%3.0f%% med=%.2fx p95=%.2fx max=%.2fx fire=%d %s'
            % (name, s['evals'], 100 * s['coverage'], s['median_overhead'],
               s['p95_overhead'], s['max_overhead'], s['n_fire'],
               s['miss_cells'] or ''))

    res['arms']['m2g'], g_rows = run_guard(C, args.eps)
    s = res['arms']['m2g']
    per_cell['m2g'] = {r['id']: r for r in g_rows}
    log('%-4s evals=%d cov=%3.0f%% med=%.2fx p95=%.2fx max=%.2fx guard=%d %s'
        % ('m2g', s['evals'], 100 * s['coverage'], s['median_overhead'],
           s['p95_overhead'], s['max_overhead'], s['n_guard'],
           s['guard_cells']))

    res['arms']['m2gb'], gb_rows = run_guard_bisect(C, args.eps)
    s = res['arms']['m2gb']
    log('%-4s evals=med%.0f/mean%.1f cov=%3.0f%% med=%.2fx p95=%.2fx max=%.2fx %s'
        % ('m2gb', s['median_evals'], s['mean_evals'], 100 * s['coverage'],
           s['median_overhead'], s['p95_overhead'], s['max_overhead'],
           s['miss_cells'] or ''))

    res['arms']['m3'], m3_rows = run_three_branch(C, args.eps, sharp=False)
    res['arms']['m3b'], _ = run_three_branch(C, args.eps, sharp=True)
    res['arms']['m3p'], _ = run_three_branch(C, args.eps, probe_c=True)
    sp = res['arms']['m3p']
    log('%-4s evals=med%.0f/mean%.2f cov=%3.0f%% med=%.2fx p95=%.2fx max=%.2fx'
        % ('m3p', sp['median_evals'], sp['mean_evals'], 100 * sp['coverage'],
           sp['median_overhead'], sp['p95_overhead'], sp['max_overhead']))
    sb_ = res['arms']['m3b']
    log('%-4s evals=med%.0f/mean%.1f cov=%3.0f%% med=%.2fx p95=%.2fx max=%.2fx'
        % ('m3b', sb_['median_evals'], sb_['mean_evals'], 100 * sb_['coverage'],
           sb_['median_overhead'], sb_['p95_overhead'], sb_['max_overhead']))
    s = res['arms']['m3']
    log('%-4s evals=med%.0f/mean%.1f cov=%3.0f%% med=%.2fx p95=%.2fx max=%.2fx %s'
        % ('m3', s['median_evals'], s['mean_evals'], 100 * s['coverage'],
           s['median_overhead'], s['p95_overhead'], s['max_overhead'],
           {k: (v['n'], v['median']) for k, v in s['by_branch'].items() if v['n']}))

    res['branches'] = classify_branches(C, args.eps)
    log('\nbranches at the prescribed targets: ' + ', '.join(
        '%s=%d' % (k, len(v)) for k, v in res['branches'].items()))

    res['branch_c'] = {}
    for tag, fac in (('2x', 2.0), ('widest', None)):
        bc, _rows = run_branch_c(C, args.eps, fac)
        res['branch_c'][tag] = bc
        log('\nbranch C [%s] both anchors above tolerance, n=%d, spread %.2f nats'
            % (tag, bc['n'], bc['anchor_spread_median']))
        for k in ('fit', 'ladder', 'fitprobe', 'full'):
            a = bc['arms'][k]
            log('  %-7s cov=%3.0f%% med=%.2fx p95=%.2fx max=%.2fx %s'
                % (k, 100 * a['coverage'], a['median_overhead'], a['p95_overhead'],
                   a['max_overhead'], a['miss_cells'] or ''))
        log('  ladder probes: median %.0f, max %d'
            % (bc['median_probes_ladder'], bc['evals']['ladder_max'] - 2))

    res['brackets'] = bracket_scan(C, args.eps)
    log('\nbisection bracket sensitivity (probes are lookups on the measured grid)')
    for k, v in res['brackets'].items():
        log('  %-22s cov=%3.0f%% med=%.2fx p95=%.1fx med_probes=%.0f'
            % (k, 100 * v['coverage'], v['median_overhead'],
               v['p95_overhead'], v['median_probes']))

    ids = sorted(per_cell['m2'])
    with open(os.path.join(OUT, args.csv), 'w', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['cell', 'r_true', 'm2_rank', 'm2_ov', 'm2_meet',
                    'm2p_rank', 'm2p_ov', 'm2p_fire',
                    'u2_rank', 'u2_ov', 'u2_meet', 'u2p_rank', 'u2p_ov', 'd_probe',
                    'm2g_rank', 'm2g_ov', 'm2g_guard'])
        for i in ids:
            c = next(x for x in C if x['id'] == i)
            a, b, u, v = (per_cell[k][i] for k in ('m2', 'm2p', 'u2', 'u2p'))
            q = per_cell['m2g'][i]
            w.writerow([i, c['r_true'], a['issued'], round(a['overhead'], 2), a['meet'],
                        b['issued'], round(b['overhead'], 2), b['fire'],
                        u['issued'], round(u['overhead'], 2), u['meet'],
                        v['issued'], round(v['overhead'], 2), round(v['d_probe'], 3),
                        q['issued'], round(q['overhead'], 2), q['guarded']])

    json.dump(res, open(os.path.join(OUT, args.out), 'w'), indent=1)
    log('\nwrote %s and %s' % (args.out, args.csv))


if __name__ == '__main__':
    main()
