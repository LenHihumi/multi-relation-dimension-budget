"""Sensitivity of the two-point calibration to WHICH two points are measured.

The paper reports two-point calibration at 100% empirical coverage and a median
rank overhead of 1.00x.  Both numbers come from two prescribed rho targets
(0.04, 0.12).  A reviewer is entitled to ask whether those two points are what
produces the result.

This script answers that without any new retrieval.  It refits every cell from
every admissible pair of the 23 measured ranks, under the same
leave-one-corpus-out protocol and the same safety-margin rule as the `free_2pt`
arm of sb_lodo_true.py, and reports the distribution of (coverage, median
overhead) across pairs.

Two axes are swept, because they answer different questions:

  rank  - all C(23,2)=253 pairs of measured ranks.  Exhaustive, no arbitrary
          target grid.  Asks: if the practitioner happened to measure at these
          two truncation ranks, would the protocol still work?
  rho   - all pairs of rho targets on a 0.01..0.30 grid, each snapped to the
          nearest measured rank per cell.  Closer to how the rule is stated and
          deployed, at the cost of a grid choice.

Only the selected points enter the fit.  The full measured curve is used solely
as ground truth for scoring, exactly as in sb_lodo_true.py.

Usage:
  python3 sb_2pt_sensitivity.py --out sensitivity_2pt.json
  python3 sb_2pt_sensitivity.py --out smoke.json --max-pairs 6
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sb_common                                                       # noqa: E402

# sb_common reads its output directory at import time; set SB_OUT before any
# module does `from sb_common import OUT`, so the override is what those modules
# bind.
if os.environ.get('SB_OUT'):
    sb_common.OUT = os.environ['SB_OUT']

from sb_acceptance import EPS, fit_ll                                   # noqa: E402
from sb_common import OUT                                              # noqa: E402
from sb_lodo_true import curve, eval_rank, load_cells, rank_for        # noqa: E402

BOUND_R0 = (1e-4, 1e6)
BOUND_S = (0.1, 20.0)


def log(m):
    print(m, flush=True)


def at_bound(r0, s):
    return ((r0 <= BOUND_R0[0] * (1 + 1e-9)) or (r0 >= BOUND_R0[1] * (1 - 1e-9))
            or (s <= BOUND_S[0] * (1 + 1e-9)) or (s >= BOUND_S[1] * (1 - 1e-9)))


def run_pair(C, idx_of, eps):
    """Score one pair selection under leave-one-corpus-out.

    idx_of(cell) -> (i, j) measured-rank indices for that cell, or None when the
    two targets collapse onto the same rank (the pair is then unidentifiable by
    construction and every cell counts as a miss).

    Each row also records the measured Delta at the two points and whether the
    pair STRADDLES the target (one point already converged, one not).  That flag
    is what turns this from a sensitivity number into a placement rule: the
    interesting question is not "which pair is best" but "does coverage depend
    on whether the two points bracket the crossing".
    """
    rows = []
    for g in sorted({c['ds'] for c in C}):
        tr = [c for c in C if c['ds'] != g]
        te = [c for c in C if c['ds'] == g]
        resid = []
        for c in tr:
            ij = idx_of(c)
            if ij is None:
                continue
            i, j = ij
            r0, s = fit_ll(c['rho'][[i, j]], c['delta'][[i, j]])
            resid.extend((c['delta'][[i, j]] - curve(c['rho'][[i, j]], r0, s)).tolist())
        bias = max(0.0, float(np.quantile(resid, 0.90))) if resid else 0.0
        for c in te:
            ij = idx_of(c)
            if ij is None:
                rows.append({'id': c['id'], 'meet': 0, 'overhead': 1e3, 'bound': 1,
                             'd_a': 1.0, 'd_b': 1.0, 'straddle': 0, 'lsp': 0.0})
                continue
            i, j = ij
            r0, s = fit_ll(c['rho'][[i, j]], c['delta'][[i, j]])
            du = np.clip(curve(c['rho'], r0, s) + bias, 0, 1)
            meet, ov = eval_rank(c, rank_for(c['rho'], du, c['grid'], eps))
            da, db = float(c['delta'][i]), float(c['delta'][j])
            rows.append({'id': c['id'], 'meet': meet, 'overhead': ov,
                         'bound': int(at_bound(r0, s)),
                         'd_a': da, 'd_b': db,
                         'straddle': int(min(da, db) <= eps <= max(da, db)),
                         'lsp': float(abs(np.log(max(c['rho'][i], 1e-9)
                                                / max(c['rho'][j], 1e-9))))})
    meet = np.array([r['meet'] for r in rows], float)
    ov = np.array([r['overhead'] for r in rows], float)
    st = np.array([r['straddle'] for r in rows], float)
    out = {'n': len(rows),
           'coverage': float(meet.mean()),
           'median_overhead': float(np.median(ov)),
           'p95_overhead': float(np.quantile(ov, .95)),
           'max_overhead': float(np.max(ov)),
           'frac_le_1_5': float(np.mean(ov <= 1.5)),
           'n_bound': int(sum(r['bound'] for r in rows)),
           'n_straddle': int(st.sum())}
    if st.sum():
        out['coverage_straddle'] = float(meet[st > 0].mean())
        out['median_overhead_straddle'] = float(np.median(ov[st > 0]))
    if (1 - st).sum():
        out['coverage_nonstraddle'] = float(meet[st == 0].mean())
        out['median_overhead_nonstraddle'] = float(np.median(ov[st == 0]))
    return out, rows


def summarise_pairs(pairs):
    cov = np.array([p['coverage'] for p in pairs])
    med = np.array([p['median_overhead'] for p in pairs])
    p95 = np.array([p['p95_overhead'] for p in pairs])
    good = (cov >= 0.89) & (med <= 1.5)
    return {'n_pairs': len(pairs),
            'coverage_min': float(cov.min()), 'coverage_median': float(np.median(cov)),
            'coverage_max': float(cov.max()),
            'median_overhead_min': float(med.min()),
            'median_overhead_median': float(np.median(med)),
            'median_overhead_max': float(med.max()),
            'p95_overhead_median': float(np.median(p95)),
            'p95_overhead_max': float(p95.max()),
            'frac_pairs_cov_ge_89_and_med_le_1_5': float(good.mean()),
            'frac_pairs_cov_100': float(np.mean(cov >= 0.999)),
            'frac_pairs_cov_0': float(np.mean(cov <= 0.001))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='sensitivity_2pt.json')
    ap.add_argument('--eps', type=float, default=EPS)
    ap.add_argument('--max-pairs', type=int, default=0,
                    help='smoke test: only the first N pairs of each axis')
    args = ap.parse_args()

    C = load_cells()
    # Cells differ in how many ranks survive the mAP@10 bookkeeping, so an index
    # pair is not the same rank pair in two cells.  Sweep rank VALUES, restricted
    # to the ranks every cell measured.
    common = sorted(set.intersection(*[set(int(r) for r in c['grid']) for c in C]))
    R = len(common)
    log('cells=%d  grid_len=%s  common_ranks=%d  datasets=%s'
        % (len(C), sorted({len(c['grid']) for c in C}), R,
           sorted({c['ds'] for c in C})))

    def rank_index(c, r):
        return int(np.where(c['grid'] == r)[0][0])

    OBS = []  # (delta_a, delta_b, meet, overhead) over every (pair, cell)

    res = {'eps': args.eps, 'n_cells': len(C), 'n_ranks': R,
           'common_ranks': [int(r) for r in common],
           'datasets': sorted({c['ds'] for c in C})}

    # ---------------- axis 1: rank value pairs (exhaustive) ----------------
    pairs = []
    for ra, rb in itertools.combinations(common, 2):
        if args.max_pairs and len(pairs) >= args.max_pairs:
            break
        s, rows = run_pair(C, lambda c, a=ra, b=rb: (rank_index(c, a), rank_index(c, b)),
                           args.eps)
        OBS.extend((r['d_a'], r['d_b'], r['meet'], r['overhead'], r['lsp']) for r in rows)
        pairs.append({'rank_a': int(ra), 'rank_b': int(rb),
                      'rho_a_median': float(np.median(
                          [c['rho'][rank_index(c, ra)] for c in C])),
                      'rho_b_median': float(np.median(
                          [c['rho'][rank_index(c, rb)] for c in C])),
                      **s})
    res['rank_axis'] = {'summary': summarise_pairs(pairs), 'pairs': pairs}

    # ---------------- axis 2: rho target pairs ----------------
    targets = np.round(np.arange(0.01, 0.301, 0.01), 3)

    def idx_for_targets(c, ta, tb):
        ia = int(np.argmin(np.abs(np.log(np.maximum(c['rho'], 1e-9)) - np.log(ta))))
        ib = int(np.argmin(np.abs(np.log(np.maximum(c['rho'], 1e-9)) - np.log(tb))))
        return None if ia == ib else (ia, ib)

    rpairs = []
    for ta, tb in itertools.combinations(targets, 2):
        if args.max_pairs and len(rpairs) >= args.max_pairs:
            break
        s, rows = run_pair(C, lambda c, a=ta, b=tb: idx_for_targets(c, a, b), args.eps)
        OBS.extend((r['d_a'], r['d_b'], r['meet'], r['overhead'], r['lsp']) for r in rows)
        rpairs.append({'rho_a': float(ta), 'rho_b': float(tb), **s})
    res['rho_axis'] = {'summary': summarise_pairs(rpairs), 'pairs': rpairs}

    # ---------------- placement map: coverage by where the two points sit ------
    # Every observation above is a (point A, point B) pair on one cell.  Bucket
    # both points by their measured Delta and report coverage per cell of the
    # resulting 5x5 map.  This is the part that survives as a rule: it says
    # where the two points have to be, not which pair we happened to use.
    edges = [0.0, 0.05, 0.15, 0.35, 0.70, 1.01]
    nb = len(edges) - 1

    def bucket(d):
        for b in range(nb):
            if edges[b] <= d < edges[b + 1]:
                return b
        return nb - 1

    acc = {}
    for d_a, d_b, meet, ov, _lsp in OBS:
        k = (bucket(d_a), bucket(d_b))
        acc.setdefault(k, []).append((meet, ov))
    res['placement_map'] = {
        'edges': edges,
        'n': [[len(acc.get((i, j), [])) for j in range(nb)] for i in range(nb)],
        'coverage': [[(float(np.mean([m for m, _ in acc[(i, j)]]))
                       if acc.get((i, j)) else None) for j in range(nb)]
                     for i in range(nb)],
        'median_overhead': [[(float(np.median([o for _, o in acc[(i, j)]]))
                              if acc.get((i, j)) else None) for j in range(nb)]
                            for i in range(nb)]}

    strad = np.array([1 if min(o[0], o[1]) <= args.eps <= max(o[0], o[1]) else 0
                      for o in OBS])
    lsp = np.array([o[4] for o in OBS], float)
    meetv = np.array([o[2] for o in OBS], float)
    ovv = np.array([o[3] for o in OBS], float)
    sbins = [0.0, 0.5, 1.0, 1.5, 2.0, 10.0]
    res['spread_bins'] = {'edges_nats': sbins, 'rows': []}
    for b in range(len(sbins) - 1):
        m = (lsp >= sbins[b]) & (lsp < sbins[b + 1])
        if m.sum() == 0:
            continue
        res['spread_bins']['rows'].append(
            {'lo': sbins[b], 'hi': sbins[b + 1], 'n': int(m.sum()),
             'coverage': float(meetv[m].mean()),
             'median_overhead': float(np.median(ovv[m]))})

    res['pooled'] = {
        'n_total': int(len(OBS)),
        'straddle': {'n': int(strad.sum()),
                     'coverage': float(meetv[strad > 0].mean()) if strad.sum() else None,
                     'median_overhead': float(np.median(ovv[strad > 0])) if strad.sum() else None},
        'nonstraddle': {'n': int((1 - strad).sum()),
                        'coverage': float(meetv[strad == 0].mean()) if (1 - strad).sum() else None,
                        'median_overhead': float(np.median(ovv[strad == 0])) if (1 - strad).sum() else None}}

    # ---------------- the prescribed rule, for reference ----------------
    PRIOR = (0.04, 0.12)
    s, rows = run_pair(C, lambda c: idx_for_targets(c, *PRIOR), args.eps)
    res['prescribed'] = {'targets': list(PRIOR), **s,
                         'failed_cells': [r['id'] for r in rows if not r['meet']],
                         'tail_cells': [[r['id'], round(r['overhead'], 1)]
                                        for r in rows if r['overhead'] > 3]}

    json.dump(res, open(os.path.join(OUT, args.out), 'w'), indent=1)

    print()
    print('%-10s %6s %8s %8s %8s %8s %8s' %
          ('axis', 'pairs', 'cov_min', 'cov_med', 'med_ov_med', 'med_ov_max', 'pairs_ok'))
    for ax in ('rank_axis', 'rho_axis'):
        q = res[ax]['summary']
        print('%-10s %6d %7.0f%% %7.0f%% %8.2fx %8.2fx %7.0f%%'
              % (ax.replace('_axis', ''), q['n_pairs'],
                 100 * q['coverage_min'], 100 * q['coverage_median'],
                 q['median_overhead_median'], q['median_overhead_max'],
                 100 * q['frac_pairs_cov_ge_89_and_med_le_1_5']))
    q = res['pooled']
    print('\npooled over %d (pair,cell) observations' % q['n_total'])
    for tag in ('straddle', 'nonstraddle'):
        d = q[tag]
        if d['n']:
            print('  %-12s n=%5d cov=%5.1f%% med=%5.2fx'
                  % (tag, d['n'], 100 * d['coverage'], d['median_overhead']))
    print('\ncoverage by separation of the two points (natural log of the rho ratio)')
    for r in res['spread_bins']['rows']:
        print('  [%.1f,%.1f) n=%5d cov=%5.1f%% med=%5.2fx'
              % (r['lo'], r['hi'], r['n'], 100 * r['coverage'], r['median_overhead']))
    pm = res['placement_map']
    print('\nplacement map -- coverage by Delta at each of the two points')
    print('  rows: Delta at point A, columns: Delta at point B (edges %s)' % pm['edges'])
    for i in range(len(pm['edges']) - 1):
        cells = []
        for j in range(len(pm['edges']) - 1):
            c = pm['coverage'][i][j]
            cells.append('   --' if c is None else '%4.0f%%' % (100 * c))
        print('  A[%.2f,%.2f) ' % (pm['edges'][i], pm['edges'][i + 1]), ' '.join(cells))

    p = res['prescribed']
    print('\nprescribed rho=(0.04,0.12): cov=%.0f%% med=%.2fx p95=%.2fx max=%.2fx'
          % (100 * p['coverage'], p['median_overhead'], p['p95_overhead'],
             p['max_overhead']))
    if p['failed_cells']:
        print('failed cells:', p['failed_cells'])
    if p['tail_cells']:
        print('tail cells:', p['tail_cells'])


if __name__ == '__main__':
    main()
