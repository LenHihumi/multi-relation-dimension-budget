"""Sequential measurement: the prior picks where to look, the measurement decides.

The paper's rule spends two *fixed* probes (rho = 0.04, 0.12) and then reads them
with a three-branch decision.  A reviewer's objection is that this is a passive
design: the probe locations are chosen before anything is measured, so on cells
whose crossing is far from those two ranks the rule pays for a third and fourth
probe, or extrapolates.

This script asks the question that objection implies, on the measured 23-rank
grids and with no new retrieval: if the prior -- a label-aware ridge, the arm
already in the artifact -- is allowed to choose *where to measure* and *when to
stop*, but is never allowed to issue a budget, what does the evaluation/overhead
frontier look like?

Arms
  F        the deployed rule: fixed two probes + three branches (baseline)
  S_fit    prior-seeded start, halve/double until the crossing is bracketed,
           then the two-point fit issues the crossing
  S_meas   the same crawl, but the *measurement* issues the budget: the smallest
           rank measured to meet the tolerance.  No fit at all.
  S_1p     S_meas with a one-probe early exit: if the seeded probe already meets
           the tolerance, issue it and stop (this is the only way to spend one
           evaluation, and it is a prior-guaranteed budget, not a measured one)
  O_meas   the same crawl started at the true crossing (oracle start): the floor
           that seed quality can buy, not a deployable rule
  B5       bisection from [8, d], the model-free reference (five probes)

`--budget k` caps the number of probes; sweeping k gives the frontier.

Usage: python3 sb_sequential.py --out sequential.json
"""
from __future__ import annotations

import argparse
import collections
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
from sb_learned import ridge_fit, ridge_pred                           # noqa: E402
from sb_lodo_true import eval_rank, load_cells, rank_for, curve        # noqa: E402
from sb_labelpred import label_features                                # noqa: E402
from sb_probe import PRIOR, fit_line, idx_for                          # noqa: E402

EPS = 0.05


# --------------------------------------------------------------------------- #
# the prior: same features, same ridge, same split as the artifact's arm
# --------------------------------------------------------------------------- #
def load_theory():
    th = {}
    for p in glob.glob(os.path.join(OUT, 'theory__*.json')):
        d = json.load(open(p))
        th[(d['encoder'], d['dataset'])] = d
    return th


def prior_start(C):
    """Leave-one-corpus-out log r* prediction, used only to seed a probe."""
    pred = {}
    for g in sorted({c['ds'] for c in C}):
        tr = [c for c in C if c['ds'] != g]
        te = [c for c in C if c['ds'] == g]
        X = np.array([c['feat_la'] for c in tr], float)
        y = np.array([np.log(max(c['r_true'], 1)) for c in tr])
        model = ridge_fit(X, y)
        for c in te:
            pred[c['id']] = float(np.exp(ridge_pred(model, np.array([c['feat_la']], float))[0]))
    return pred


def margin_for(C, g, eps=EPS, q=0.90):
    """The paper's reference margin: 90th percentile residual of the two-point fit,
    measured on the folds that are *not* the target corpus."""
    resid = []
    for c in C:
        if c['ds'] == g:
            continue
        ij = idx_for(c, *PRIOR)
        if ij is None:
            continue
        r0, s = fit_line(c['rho'][[ij[0], ij[1]]], c['delta'][[ij[0], ij[1]]], True)
        resid.extend((c['delta'][[ij[0], ij[1]]]
                      - curve(c['rho'][[ij[0], ij[1]]], r0, s)).tolist())
    return max(0.0, float(np.quantile(resid, q))) if resid else 0.0


# --------------------------------------------------------------------------- #
# one crawl
# --------------------------------------------------------------------------- #
def nearest_idx(grid, r):
    return int(np.argmin(np.abs(np.log(np.maximum(grid, 1)) - np.log(max(r, 1.0)))))


def crawl(c, start_idx, eps=EPS, budget=8, adjacent=False):
    """Probe outward from `start_idx` until the crossing sits between two probes.

    Halving and doubling happen in *rank*, which is the quantity a deployment can
    choose; the grid only decides which measured point stands in for it.  The
    crawl is therefore a sequential design on the rank axis, evaluated offline on
    the 23 measured ranks.
    """
    grid, delta = c['grid'], c['delta']
    n = len(grid)
    seen = {int(start_idx): float(delta[start_idx])}
    k = int(start_idx)
    while len(seen) < budget:
        above = [i for i in seen if delta[i] > eps]
        under = [i for i in seen if delta[i] <= eps]
        if above and under and max(above) < min(under):
            return seen, (max(above), min(under)), 'bracket'
        if adjacent:                       # walk one grid step: no fit can help
            if not above:
                j = min(seen) - 1
            elif not under:
                j = max(seen) + 1
            else:
                a, b = max(above), min(under)
                j = a + 1
        elif not above:                    # every probe meets: the crossing is lower
            k = min(seen)
            j = int(np.searchsorted(grid, grid[k] / 2.0, side='right')) - 1
        elif not under:                    # none meets: the crossing is higher
            k = max(seen)
            j = int(np.searchsorted(grid, 2.0 * grid[k], side='left'))
        else:                              # both sides seen but not adjacent yet
            a, b = max(above), min(under)
            j = (a + b) // 2
        j = int(max(0, min(n - 1, j)))
        if j == k or j in seen:
            break
        seen[j] = float(delta[j])
        k = j

    above = [i for i in seen if delta[i] > eps]
    under = [i for i in seen if delta[i] <= eps]
    if above and under and max(above) < min(under):
        return seen, (max(above), min(under)), 'bracket'
    return seen, None, 'exhausted'


def issue_fit(c, pair, eps=EPS):
    """Two-point fit on the two bracketing probes; issued rank is the crossing."""
    a, b = pair
    r0, s = fit_line(c['rho'][[a, b]], c['delta'][[a, b]], True)
    r = rank_for(c['rho'], np.clip(curve(c['rho'], r0, s), 0, 1), c['grid'], eps)
    # keep it inside what was measured: a crossing outside [grid[a], grid[b]] is
    # an extrapolation and the measurement already contains a safe answer
    return int(np.clip(r, c['grid'][a], c['grid'][b]))


def issue_meas(c, pair, eps=EPS):
    """The measured answer: the smallest rank whose loss was measured under eps."""
    a, b = pair
    return int(c['grid'][b])


def run_arm(C, start_of, issue, budget, early=False, eps=EPS, adjacent=False,
            scale=None, pad_steps=0):
    """One arm over all cells.  `scale` is a per-corpus multiplicative widening of
    the issued rank (split conformal, see conformal_scale)."""
    rows = []
    for c in C:
        s_idx = nearest_idx(c['grid'], start_of(c))
        seen, pair, how = crawl(c, s_idx, eps, budget, adjacent)
        probes = len(seen)
        width = float('nan')
        if early and float(c['delta'][s_idx]) <= eps:
            # one probe, and the budget is the prior's, not a measured crossing
            r_hat = float(c['grid'][s_idx])
            probes = 1
            how = 'early'
        elif pair is not None:
            r_hat = issue(c, pair, eps)
            width = float(np.log(c['grid'][pair[1]] / c['grid'][pair[0]]))
        else:
            # Never bracketed.  Two sub-cases with very different costs:
            #   every probe met tolerance -> the crossing is below all of them, so
            #     the *smallest* probe is already a measured-safe budget.  Issuing
            #     the largest one (what this code did before) is safe but over-
            #     provisions by up to the probe spacing.
            #   no probe met tolerance -> the crossing is above all of them and the
            #     only safe answer inside the budget is the full dimension.
            under = [i for i in seen if c['delta'][i] <= eps]
            r_hat = int(c['grid'][min(under)]) if under else int(c['grid'][-1])
            how = 'exhausted-under' if under else 'exhausted-over'
        issued_raw = float(r_hat)
        probed_ranks = {int(c['grid'][i]) for i in seen}
        if scale is not None:
            r_hat = float(r_hat) * float(scale.get(c['ds'], 1.0))
        if pad_steps:
            j = int(np.argmin(np.abs(c['grid'] - float(r_hat))))
            r_hat = float(c['grid'][min(j + pad_steps, len(c['grid']) - 1)])
        meet, ov = eval_rank(c, r_hat)
        rows.append({'id': c['id'], 'ds': c['ds'], 'issued': int(round(float(r_hat))),
                     'issued_raw': int(round(issued_raw)),
                     'issued_is_probed': int(round(float(r_hat))) in probed_ranks,
                     'meet': meet, 'overhead': ov, 'probes': probes, 'stop': how,
                     'bracket_width_nats': width, 'r_true': c['r_true']})
    return rows


def conformal_scale(rows, q=0.90):
    """Split-conformal widening per corpus, calibrated on the *other* corpora.

    The issued rank is multiplied by the q-th percentile of the overhead that the
    same rule achieved on the folds it was calibrated on.  Under exchangeability
    this bounds the miss rate at 1-q; the point of reporting it is that the bound
    costs far less than the raw tail suggests.
    """
    out = {}
    for g in sorted({r['ds'] for r in rows}):
        other = [r['overhead'] for r in rows if r['ds'] != g]
        out[g] = float(np.quantile(other, q)) if other else 1.0
    return out


def summarise(rows):
    meet = np.array([r['meet'] for r in rows], float)
    ov = np.array([r['overhead'] for r in rows], float)
    pr = np.array([r['probes'] for r in rows], float)
    by = {}
    for g in sorted({r['ds'] for r in rows}):
        sub = [r for r in rows if r['ds'] == g]
        m = np.array([r['meet'] for r in sub], float)
        o = np.array([r['overhead'] for r in sub], float)
        p = np.array([r['probes'] for r in sub], float)
        by[g] = {'n': len(sub), 'coverage': float(m.mean()),
                 'median_overhead': float(np.median(o)),
                 'p95_overhead': float(np.quantile(o, .95)),
                 'mean_evals': float(p.mean())}
    nb = sum(1 for r in rows if r.get('stop') == 'bracket')
    n_interp = sum(1 for r in rows if r.get('issued_is_probed') is False)
    evals = np.array([r['probes'] for r in rows], float)
    return {'n': len(rows), 'by_corpus': by,
            'n_bracketed': int(nb),
            'bracket_rate': float(nb / max(len(rows), 1)),
            'n_issued_interpolated': int(n_interp),
            'evals_with_verification': float((evals + (1 if False else 0)).mean()),
            'mean_evals_with_verification': float(
                evals.mean() + n_interp / max(len(rows), 1)),
            'coverage': float(meet.mean()),
            'median_overhead': float(np.median(ov)),
            'p95_overhead': float(np.quantile(ov, .95)),
            'max_overhead': float(np.max(ov)),
            'median_evals': float(np.median(pr)),
            'mean_evals': float(pr.mean()),
            'max_evals': int(pr.max()),
            'eval_hist': dict(sorted(collections.Counter(pr.astype(int).tolist()).items())),
            'miss_cells': [r['id'] for r in rows if not r['meet']]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='sequential.json')
    ap.add_argument('--eps', type=float, default=EPS)
    ap.add_argument('--budgets', default='2,3,4,5,6,8')
    args = ap.parse_args()

    th = load_theory()
    C = load_cells()
    for c in C:
        c['feat_la'] = label_features(th, c['enc'], c['ds'], c['rel'])
    start = prior_start(C)

    res = {'eps': args.eps, 'n_cells': len(C),
           'targets': list(PRIOR), 'split': 'leave-one-corpus-out',
           'arms': {}, 'per_cell': {}}

    # ---- baseline: the deployed fixed two-probe rule ---------------------- #
    def fixed_two(margins, budget=99):
        out, n_lad = [], 0
        for c in C:
            ij = idx_for(c, *PRIOR)
            da, db = float(c['delta'][ij[0]]), float(c['delta'][ij[1]])
            probes = 2
            if max(da, db) <= args.eps:                # both under: issue the smaller
                r_hat = int(min(c['grid'][ij[0]], c['grid'][ij[1]]))
            elif min(da, db) > args.eps:               # both over: doubling ladder
                n_lad += 1
                k, n = max(ij), len(c['grid'])
                while c['delta'][k] > args.eps and k < n - 1 and probes < budget:
                    k = min(int(np.searchsorted(c['grid'], 2 * c['grid'][k])), n - 1)
                    probes += 1
                r_hat = int(c['grid'][k])
            else:                                      # straddle: fit (+ margin)
                a, b = ij
                r0, ss = fit_line(c['rho'][[a, b]], c['delta'][[a, b]], True)
                bias = 0.0 if margins is None else margins[c['ds']]
                du = np.clip(curve(c['rho'], r0, ss) + bias, 0, 1)
                r_hat = rank_for(c['rho'], du, c['grid'], args.eps)
            meet, ov = eval_rank(c, r_hat)
            out.append({'id': c['id'], 'ds': c['ds'], 'issued': r_hat, 'meet': meet,
                        'overhead': ov, 'probes': probes, 'r_true': c['r_true'],
                        'bracket_width_nats': float('nan')})
        return out, n_lad

    marg = {g: margin_for(C, g, args.eps) for g in sorted({c['ds'] for c in C})}
    base, n_ladder = fixed_two(marg)
    res['arms']['F_fixed2'] = summarise(base)
    res['arms']['F_fixed2']['cells_on_ladder'] = n_ladder
    # The reference margin is the 90th percentile residual of the fit on the other
    # folds; on this corpus it is zero for all three, so the deployed rule here is
    # the fit without a margin.  Recorded rather than assumed, because a zero
    # margin silently turns two arms into one.
    res['reference_margin'] = dict(marg)

    # ---- bisection reference --------------------------------------------- #
    from sb_probe import bisect_from
    b5 = []
    for c in C:
        k, pr, _ = bisect_from(c['delta'], args.eps, 0, None)
        meet, ov = eval_rank(c, int(c['grid'][k]))
        b5.append({'id': c['id'], 'ds': c['ds'], 'issued': int(c['grid'][k]),
                   'meet': meet, 'overhead': ov, 'probes': pr,
                   'r_true': c['r_true'], 'bracket_width_nats': float('nan')})
    res['arms']['B5_bisect'] = summarise(b5)

    # ---- the frontier ----------------------------------------------------- #
    frontier = {}
    for b in [int(x) for x in args.budgets.split(',')]:
        frontier[str(b)] = {}
        for tag, issue, early in (('S_fit', issue_fit, False),
                                  ('S_meas', issue_meas, False),
                                  ('S_1p', issue_meas, True)):
            rows = run_arm(C, lambda c: start[c['id']], issue, b, early, args.eps)
            frontier[str(b)][tag] = summarise(rows)
        rows = run_arm(C, lambda c: c['r_true'], issue_meas, b, False, args.eps)
        frontier[str(b)]['O_meas'] = summarise(rows)
    res['frontier'] = frontier

    # the headline setting: budget 4 (allow up to four probes)
    for tag in ('S_fit', 'S_meas', 'S_1p', 'O_meas'):
        issue = issue_fit if tag == 'S_fit' else issue_meas
        rows = run_arm(C, lambda c: start[c['id']] if tag != 'O_meas' else c['r_true'],
                       issue, 4, tag == 'S_1p', args.eps)
        res['arms'][tag] = summarise(rows)
        if tag in ('S_meas', 'S_fit'):
            res['per_cell'][tag] = rows

    # ---- how much of S_fit is the prior, and how much is the crawl? ---- #
    # a blind seed (d/8, what a deployment without a group key would use) and a
    # random seed, both followed by the same crawl and the same interpolation.
    rows = run_arm(C, lambda c: c['grid'][-1] / 8.0, issue_fit, 8, False, args.eps)
    res['arms']['R_fit_d8'] = summarise(rows)
    res['per_cell']['R_fit_d8'] = rows

    rng = np.random.default_rng(7)
    reps = []
    for _ in range(20):
        pick = {c['id']: float(rng.choice(c['grid'])) for c in C}
        reps.append(summarise(run_arm(C, lambda c: pick[c['id']], issue_fit, 8,
                                      False, args.eps)))
    keys = ('coverage', 'median_overhead', 'p95_overhead', 'max_overhead', 'mean_evals')
    res['arms']['R_fit_rand'] = {k: float(np.median([r[k] for r in reps])) for k in keys}
    res['arms']['R_fit_rand']['n_draws'] = len(reps)
    res['arms']['R_fit_rand']['coverage_min'] = float(min(r['coverage'] for r in reps))
    res['arms']['R_fit_rand']['by_corpus'] = {
        g: {k: float(np.median([r['by_corpus'][g][k] for r in reps]))
            for k in ('coverage', 'median_overhead', 'p95_overhead', 'mean_evals')}
        for g in sorted({c['ds'] for c in C})}

    # ---- split-conformal widening of the issued rank ---- #
    sc = conformal_scale(res['per_cell']['S_fit'], 0.90)
    rows = run_arm(C, lambda c: start[c['id']], issue_fit, 4, False, args.eps, scale=sc)
    res['arms']['S_fit_conf'] = summarise(rows)
    res['conformal'] = {'q': 0.90, 'scale': sc}

    # the conformal factor is below one grid step on every corpus, so a bound
    # stated in rank units cannot move the answer: pad one grid step instead and
    # read the price of an actual guarantee
    for k in (1, 2):
        rows = run_arm(C, lambda c: start[c['id']], issue_fit, 4, False, args.eps,
                       pad_steps=k)
        res['arms']['S_fit_pad%d' % k] = summarise(rows)

    # ---- the same-budget comparison: what a two-probe budget buys ---- #
    capped, _ = fixed_two(marg, budget=2)
    res['arms']['F_cap2'] = summarise(capped)
    res['same_budget'] = {}
    for b in (2, 3, 4):
        res['same_budget'][str(b)] = {
            'S_fit': frontier[str(b)]['S_fit'] if str(b) in frontier else None,
            'F_fixed2': res['arms']['F_fixed2'],
            'F_cap2': res['arms']['F_cap2'] if b == 2 else None,
            'B5_bisect': res['arms']['B5_bisect'],
        }

    # ---- how wide is the final bracket?  (is 1.00x a grid artifact?) ---- #
    w = [r['bracket_width_nats'] for r in res['per_cell']['S_fit']
         if np.isfinite(r['bracket_width_nats'])]
    res['bracket_width'] = {
        'n': len(w),
        'median_nats': float(np.median(w)),
        'p90_nats': float(np.quantile(w, .90)),
        'max_nats': float(np.max(w)),
        'median_rank_ratio': float(np.exp(np.median(w))),
        'one_grid_step': float(np.median(np.diff(np.log(C[0]['grid']))))}

    # same crawl, but walking one grid step at a time and issuing the measured
    # side: this is the measurement-only arm at full resolution, so the gap to
    # S_fit is exactly what the two-point interpolation is worth in probes
    rows = run_arm(C, lambda c: start[c['id']], issue_meas, 12, False, args.eps, True)
    res['arms']['S_meas_adj'] = summarise(rows)
    res['per_cell']['S_meas_adj'] = rows

    # ---- the monotonicity assumption the crawl rests on ----
    # The paper reports that the predicate "loss falls as rank grows" is monotone
    # on only 3 of the 19 grids.  The crawl does not *require* monotonicity to run
    # -- it only requires that some probe met tolerance and some did not -- so the
    # honest report is: the process-level assumption fails on most grids, and the
    # result-level claim is checked separately, on the issued rank itself.
    mono = [c['id'] for c in C if np.all(np.diff(c['delta']) <= 1e-9)]
    res['monotone_grids'] = {'n': len(mono), 'ids': sorted(mono)}

    # ---- which cell is left unbracketed at each budget, and why ----
    detail = {}
    for b in (2, 3, 4):
        rows_b = run_arm(C, lambda c: start[c['id']], issue_fit, b, False, args.eps)
        detail[str(b)] = [{'id': r['id'], 'stop': r['stop'], 'probes': r['probes'],
                           'issued': r['issued'], 'overhead': r['overhead']}
                          for r in rows_b if r['stop'] != 'bracket']
    res['unbracketed'] = detail

    # ---- does the issued rank need one more measurement to be a guarantee? ----
    # 14 of 19 issued ranks are interpolations rather than probed points, so the
    # headline evaluation count is *without* the final verification.  Adding it
    # costs one probe on exactly those cells, and the same is true of the paper's
    # three-branch rule, which interpolates on its straddling cells.
    st = res['arms']['S_fit']
    n_straddle = len(json.load(open(os.path.join(OUT, 'probe.json')))
                      ['branches']['straddle']) if os.path.exists(
        os.path.join(OUT, 'probe.json')) else 8
    # did an interpolation ever land on a rank that failed the tolerance?
    ipc = [r for r in res['per_cell']['S_fit'] if not r['issued_is_probed']]
    res['interpolation_check'] = {
        'n': len(ipc), 'n_failed': sum(1 for r in ipc if not r['meet']),
        'failed_cells': [r['id'] for r in ipc if not r['meet']]}
    res['verification'] = {
        'S_fit_issued_interpolated': st['n_issued_interpolated'],
        'S_fit_mean_evals_with_verification': st['mean_evals_with_verification'],
        'three_branch_straddle_cells': n_straddle,
        'three_branch_mean_evals_with_verification':
            res['arms']['F_fixed2']['mean_evals'] + n_straddle / len(C),
        'note': 'the headline counts are without the final verification probe'}

    # ---- how close did the interpolation come to failing? ----
    # Two different questions: how accurate is the fitted curve at the issued rank,
    # and how much room was left before the tolerance line.  If the error is small
    # but the margin is just as small, then 0/14 is a near miss, not safety.
    ierr = []
    for r in res['per_cell']['S_fit']:
        if r['issued_is_probed']:
            continue
        c = next(x for x in C if x['id'] == r['id'])
        seen, pair, _ = crawl(c, nearest_idx(c['grid'], start[c['id']]), args.eps, 4)
        r0, ss = fit_line(c['rho'][[pair[0], pair[1]]],
                          c['delta'][[pair[0], pair[1]]], True)
        j = int(np.argmin(np.abs(c['grid'] - r['issued'])))
        pred, act = float(curve(c['rho'][j], r0, ss)), float(c['delta'][j])
        ierr.append({'id': r['id'], 'pred': pred, 'actual': act,
                     'error': pred - act, 'margin': float(args.eps - act)})
    e = np.array([x['error'] for x in ierr])
    m = np.array([x['margin'] for x in ierr])
    res['interpolation_error'] = {
        'n': len(ierr), 'rows': ierr,
        'median_signed': float(np.median(e)),
        'min_signed': float(e.min()), 'max_signed': float(e.max()),
        'median_abs': float(np.median(np.abs(e))),
        'max_abs': float(np.abs(e).max()),
        'n_positive': int((e > 0).sum()), 'n_negative': int((e < 0).sum()),
        'median_margin': float(np.median(m)),
        'min_margin': float(m.min()), 'max_margin': float(m.max()),
        'median_abs_error_over_margin': float(np.median(np.abs(e) / np.maximum(m, 1e-9)))}

    # ---- adaptive verification: pay the insurance only where it might be needed ----
    # The 14 interpolations all met tolerance, so on this corpus every trigger rule
    # has the same *benefit* (zero failures caught) and they differ only in cost.
    # They are therefore reported as a cost distribution, not as a recommendation:
    # a rule cannot be validated against a data set with no failures.
    interp = [r for r in res['per_cell']['S_fit'] if not r['issued_is_probed']]
    grid_step = float(np.median(np.diff(np.log(C[0]['grid']))))
    trig = []
    for r in interp:
        c = next(x for x in C if x['id'] == r['id'])
        seen, pair, _ = crawl(c, nearest_idx(c['grid'], start[c['id']]), args.eps, 4)
        lo, hi = c['grid'][pair[0]], c['grid'][pair[1]]
        frac = (np.log(r['issued']) - np.log(lo)) / (np.log(hi) - np.log(lo))
        trig.append({'id': r['id'], 'width_steps': float(np.log(hi / lo) / grid_step),
                     'frac': float(frac),
                     'mono': bool(np.all(np.diff(c['delta']) <= 1e-9)),
                     'meet': bool(r['meet'])})
    rules = {
        'wide_bracket_gt4': lambda x: x['width_steps'] > 4,
        'interp_in_outer_quartile': lambda x: x['frac'] < 0.25 or x['frac'] > 0.75,
        'nonmono_and_outer_quartile': lambda x: (not x['mono']) and (x['frac'] < 0.25
                                                                   or x['frac'] > 0.75),
        'verify_all': lambda x: True,
    }
    res['adaptive_verification'] = {
        'n_interpolated': len(trig),
        'n_failed': sum(1 for x in trig if not x['meet']),
        'rules': {k: {'triggers': sum(1 for x in trig if f(x)),
                      'cost': float(res['arms']['S_fit']['mean_evals']
                                    + sum(1 for x in trig if f(x)) / len(C)),
                      'failures_caught': sum(1 for x in trig if f(x) and not x['meet'])}
                  for k, f in rules.items()},
        'note': 'every rule catches zero failures here; the rules differ only in cost'}

    # ---- the two cells the paper names, reported separately and per budget ----
    # the aggregate hides them, which is exactly the failure mode Sec. 5.4 warns
    # about; a reader must be able to recompute what a budget cap does to them.
    deg = {}
    for c in C:
        if c['id'] not in ('dinov3_vitl16/instre/inst', 'dinov3_vitl16/instre/cat'):
            continue
        deg[c['id']] = {'r_true': int(c['r_true']), 'by_budget': {}}
        for b in (2, 3, 4, 5):
            r = run_arm([c], lambda cc: start[cc['id']], issue_fit, b, False, args.eps)[0]
            deg[c['id']]['by_budget'][str(b)] = {
                'probes': int(r['probes']), 'issued': int(r['issued']),
                'overhead': float(r['overhead']), 'stop': r['stop']}
    res['degraded_cells'] = deg

    # seed quality: how close is the prior start to the true crossing?
    ratio = [start[c['id']] / max(c['r_true'], 1) for c in C]
    res['seed'] = {'median_ratio': float(np.median(ratio)),
                   'p05': float(np.quantile(ratio, .05)),
                   'p95': float(np.quantile(ratio, .95)),
                   'within_2x': float(np.mean([0.5 <= r <= 2.0 for r in ratio]))}

    print('\nseed: prior start / r*  median %.2fx, 5-95%% [%.2f, %.2f], within 2x %.0f%%'
          % (res['seed']['median_ratio'], res['seed']['p05'],
             res['seed']['p95'], 100 * res['seed']['within_2x']))
    print('\n%-10s %6s %8s %8s %8s %7s %7s' %
          ('arm', 'evals', 'cover', 'median', 'p95', 'max', 'miss'))
    for tag in ('F_fixed2', 'F_cap2', 'B5_bisect', 'R_fit_d8', 'R_fit_rand', 'S_fit',
                'S_fit_conf', 'S_fit_pad1', 'S_fit_pad2', 'S_meas', 'S_meas_adj',
                'S_1p', 'O_meas'):
        a = res['arms'][tag]
        miss = (len(a['miss_cells']) if 'miss_cells' in a else
                int(round(19 * (1 - a['coverage']))))
        print('%-10s %6.2f %7.0f%% %7.2fx %7.2fx %6.2fx %7d' %
              (tag, a['mean_evals'], 100 * a['coverage'], a['median_overhead'],
               a['p95_overhead'], a['max_overhead'], miss))
    bw = res['bracket_width']
    print('bracket: median %.2f nats (%.2fx rank), p90 %.2fx, max %.2fx; one grid step %.3f nats'
          % (bw['median_nats'], bw['median_rank_ratio'],
             float(np.exp(bw['p90_nats'])), float(np.exp(bw['max_nats'])),
             bw['one_grid_step']))
    print('conformal scale: ' + ', '.join('%s=%.3f' % (k, v)
                                          for k, v in res['conformal']['scale'].items()))
    print('\nby corpus (evals / coverage / median / p95):')
    print('  %-8s %-22s %-22s %-22s' % ('corpus', 'prior seed', 'blind d/8', 'random seed'))
    for g in sorted({c['ds'] for c in C}):
        cells = []
        for arm in ('S_fit', 'R_fit_d8', 'R_fit_rand'):
            v = res['arms'][arm]['by_corpus'][g]
            cells.append('%.2f / %.0f%% / %.2f / %.2f'
                         % (v['mean_evals'], 100 * v['coverage'],
                            v['median_overhead'], v['p95_overhead']))
        print('  %-8s %-22s %-22s %-22s' % (g, cells[0], cells[1], cells[2]))
    print('\nthree-probe budget, by corpus (S_fit):')
    for g, v in res['frontier']['3']['S_fit']['by_corpus'].items():
        print('  %-8s n=%d cover %.0f%% median %.2fx p95 %.2fx evals %.2f'
              % (g, v['n'], 100 * v['coverage'], v['median_overhead'],
                 v['p95_overhead'], v['mean_evals']))
    vv = res['verification']
    print('\nverification: %d of %d issued ranks are interpolations (not probed)'
          % (vv['S_fit_issued_interpolated'], len(C)))
    print('  S_fit          %.2f evals as reported, %.2f with one verification probe'
          % (st['mean_evals'], vv['S_fit_mean_evals_with_verification']))
    print('  three-branch   %.2f evals as reported, %.2f with verification on its %d straddling cells'
          % (res['arms']['F_fixed2']['mean_evals'],
             vv['three_branch_mean_evals_with_verification'], vv['three_branch_straddle_cells']))
    ie = res['interpolation_error']
    print('\ninterpolation error at the issued rank: |err| median %.4f, max %.4f '
          '(%d positive, %d negative)' % (ie['median_abs'], ie['max_abs'],
                                          ie['n_positive'], ie['n_negative']))
    print('  margin to tolerance at the issued rank: median %.4f, min %.4f'
          % (ie['median_margin'], ie['min_margin']))
    print('  |error| / margin, median: %.2f' % ie['median_abs_error_over_margin'])
    av = res['adaptive_verification']
    print('\nadaptive verification (%d interpolations, %d failures):'
          % (av['n_interpolated'], av['n_failed']))
    for k, v in av['rules'].items():
        print('  %-30s triggers %2d  cost %.2f  catches %d'
              % (k, v['triggers'], v['cost'], v['failures_caught']))
    ic = res['interpolation_check']
    print('  interpolated issuances that failed tolerance: %d of %d'
          % (ic['n_failed'], ic['n']))
    print('\nmonotone grids: %d of %d (the crawl does not require it, but the '
          'process claim does)' % (res['monotone_grids']['n'], len(C)))
    for b, rows_b in res['unbracketed'].items():
        if rows_b:
            print('  budget %s left %d unbracketed: %s' % (b, len(rows_b), ', '.join(
                '%s (%s, %d probes, %.2fx)' % (x['id'], x['stop'], x['probes'],
                                               x['overhead']) for x in rows_b)))
        else:
            print('  budget %s: every cell bracketed' % b)
    print('\ndegraded cells, by budget cap (probes: issued / overhead):')
    for cid, v in res['degraded_cells'].items():
        cells = ['%s: %d / %.2fx' % (b, x['issued'], x['overhead'])
                 for b, x in v['by_budget'].items()]
        print('  %-28s r*=%-4d %s' % (cid, v['r_true'], ' | '.join(cells)))
    print('  S_fit probe histogram: ' + ', '.join(
        '%d probes x %d' % (k, v) for k, v in res['arms']['S_fit']['eval_hist'].items()))
    print('\nsame budget (2 probes): S_fit %.0f%% / p95 %.1fx | three-branch-capped %.0f%% / p95 %.2fx'
          % (100 * res['frontier']['2']['S_fit']['coverage'],
             res['frontier']['2']['S_fit']['p95_overhead'],
             100 * res['arms']['F_cap2']['coverage'],
             res['arms']['F_cap2']['p95_overhead']))
    print('\nfrontier (mean evals -> median overhead | p95 | coverage):')
    for b, arms in frontier.items():
        line = '  budget %-2s ' % b
        for tag in ('S_fit', 'S_meas', 'S_1p'):
            a = arms[tag]
            line += ' | %s %.2f->%.2fx/%.2fx %.0f%%' % (
                tag, a['mean_evals'], a['median_overhead'], a['p95_overhead'],
                100 * a['coverage'])
        print(line)

    with open(os.path.join(OUT, args.out), 'w') as f:
        json.dump(res, f, indent=1)
    print('\nwrote %s' % os.path.join(OUT, args.out))


if __name__ == '__main__':
    main()
