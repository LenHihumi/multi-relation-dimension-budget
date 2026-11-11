"""Does the response surface have to be log-logistic?

The reviewer's sharpest question: with the same measured points, a monotone
nonparametric interpolant might do as well, in which case Eq. (1) is a
convenience rather than a finding, and the paper's positive contribution shrinks
to "measure two points and interpolate".

This script answers it with no new retrieval.  Every arm selects the same points,
uses the same leave-one-corpus-out folds and the same safety-margin rule as the
fitted arms in sb_lodo_true.py, and differs only in how the curve between and
beyond the points is drawn:

  ll    - bounded log-logistic fit (the paper's Eq. 1), closed form in
          log(rho)/logit(Delta), inverted for the budget
  pchip - monotone piecewise-cubic Hermite (Fritsch-Carlson) through the same
          points in the same coordinates, linear extrapolation beyond the ends
  iso   - isotonic regression (pool-adjacent-violators, non-decreasing since
          Delta grows with rho), linearly interpolated between the fitted
          values and flat beyond the ends

At k=2 the ll and pchip arms are algebraically the same line, so their agreement
is a consistency check on the implementation, not a result.  The comparison that
matters is k=3 and above, where the interpolants have room to differ from the
two-parameter form.

Usage:
  python3 sb_nonparam.py --out nonparam.json
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

from sb_acceptance import EPS, fit_ll                                   # noqa: E402
from sb_common import OUT                                              # noqa: E402
from sb_lodo_true import curve, eval_rank, load_cells, rank_for        # noqa: E402

# rho targets per k. 2 and 3 are the paper's own; 4 and 5 extend the same
# geometric ladder so that point placement is not confounded with k.
TARGETS = {2: (0.04, 0.12),
           3: (0.03, 0.08, 0.14),
           4: (0.02, 0.05, 0.10, 0.16),
           5: (0.02, 0.04, 0.08, 0.13, 0.20)}

LO, HI = 1e-4, 1 - 1e-4


def log(m):
    print(m, flush=True)


def pick(rho, k):
    idx = []
    for t in TARGETS[k]:
        j = int(np.argmin(np.abs(np.log(np.maximum(rho, 1e-9)) - np.log(t))))
        if j not in idx:
            idx.append(j)
    return sorted(idx)


def logit(d):
    d = np.clip(d, LO, HI)
    return np.log(d / (1 - d))


def expit(z):
    return 1.0 / (1.0 + np.exp(-z))


def pchip_slopes(x, y):
    """Fritsch-Carlson monotone slopes for strictly increasing x."""
    n = len(x)
    h = np.diff(x)
    d = np.diff(y) / h
    m = np.zeros(n)
    if n == 2:
        m[:] = d[0]
        return m
    for i in range(1, n - 1):
        if d[i - 1] * d[i] <= 0:
            m[i] = 0.0
        else:
            w1 = 2 * h[i] + h[i - 1]
            w2 = h[i] + 2 * h[i - 1]
            m[i] = (w1 + w2) / (w1 / d[i - 1] + w2 / d[i])
    # endpoints: standard one-sided three-point rule, then clamp
    for i, (h0, h1, d0, d1) in ((0, (h[0], h[1], d[0], d[1])),
                                (n - 1, (h[-1], h[-2], d[-1], d[-2]))):
        s = ((2 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
        if s * d0 <= 0:
            s = 0.0
        elif d0 * d1 <= 0 and abs(s) > abs(3 * d0):
            s = 3 * d0
        m[i] = s
    return m


def pchip_eval(x, y, m, q):
    """Evaluate the Hermite interpolant at q, linear beyond the ends."""
    out = np.empty_like(q, dtype=float)
    n = len(x)
    left = q <= x[0]
    right = q >= x[-1]
    inner = ~(left | right)
    out[left] = y[0] + m[0] * (q[left] - x[0])
    out[right] = y[-1] + m[-1] * (q[right] - x[-1])
    if inner.any():
        qq = q[inner]
        j = np.clip(np.searchsorted(x, qq) - 1, 0, n - 2)
        h = x[j + 1] - x[j]
        t = (qq - x[j]) / h
        h00 = 2 * t ** 3 - 3 * t ** 2 + 1
        h10 = t ** 3 - 2 * t ** 2 + t
        h01 = -2 * t ** 3 + 3 * t ** 2
        h11 = t ** 3 - t ** 2
        out[inner] = (h00 * y[j] + h10 * h * m[j] + h01 * y[j + 1] + h11 * h * m[j + 1])
    return out


def pava_decreasing(y):
    """Pool adjacent violators, monotone non-increasing output."""
    vals, wts, cnt = [], [], []
    for v in y:
        vals.append(float(v))
        wts.append(1.0)
        cnt.append(1)
        while len(vals) > 1 and vals[-2] < vals[-1]:
            w = wts[-2] + wts[-1]
            v = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / w
            c = cnt[-2] + cnt[-1]
            vals[-2:] = [v]
            wts[-2:] = [w]
            cnt[-2:] = [c]
    out = []
    for v, c in zip(vals, cnt):
        out.extend([v] * c)
    return np.array(out)


def predict(arm, rho_pts, d_pts, rho_all):
    """Predicted Delta over the full rho grid from the measured points."""
    x = np.log(np.maximum(rho_pts, 1e-9))
    order = np.argsort(x)
    x, d = x[order], np.clip(d_pts[order], LO, HI)
    q = np.log(np.maximum(rho_all, 1e-9))
    if arm == 'll':
        r0, s = fit_ll(rho_pts[order], d)
        return curve(np.exp(q), r0, s)
    z = logit(d)
    if arm == 'pchip':
        m = pchip_slopes(x, z)
        return np.clip(expit(pchip_eval(x, z, m, q)), 0, 1)
    if arm == 'iso':
        # Delta increases with rho, so the isotonic fit must be non-decreasing:
        # PAVA on the negated values, negated back.
        zz = -pava_decreasing(-z)
        return np.clip(expit(np.interp(q, x, zz)), 0, 1)
    raise ValueError(arm)


ARMS = ('ll', 'pchip', 'iso')


def run(C, k, eps):
    rows = {a: [] for a in ARMS}
    for g in sorted({c['ds'] for c in C}):
        tr = [c for c in C if c['ds'] != g]
        te = [c for c in C if c['ds'] == g]
        bias = {}
        for a in ARMS:
            resid = []
            for c in tr:
                idx = pick(c['rho'], k)
                du = predict(a, c['rho'][idx], c['delta'][idx], c['rho'][idx])
                resid.extend((c['delta'][idx] - du).tolist())
            bias[a] = max(0.0, float(np.quantile(resid, 0.90))) if resid else 0.0
        for c in te:
            idx = pick(c['rho'], k)
            for a in ARMS:
                du = np.clip(predict(a, c['rho'][idx], c['delta'][idx], c['rho']) + bias[a], 0, 1)
                meet, ov = eval_rank(c, rank_for(c['rho'], du, c['grid'], eps))
                rows[a].append({'id': c['id'], 'meet': meet, 'overhead': ov})
    out = {}
    for a in ARMS:
        meet = np.array([r['meet'] for r in rows[a]], float)
        ov = np.array([r['overhead'] for r in rows[a]], float)
        out[a] = {'n': len(meet), 'coverage': float(meet.mean()),
                  'median_overhead': float(np.median(ov)),
                  'p95_overhead': float(np.quantile(ov, .95)),
                  'max_overhead': float(np.max(ov)),
                  'frac_le_1_5': float(np.mean(ov <= 1.5))}
    return out, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='nonparam.json')
    ap.add_argument('--eps', type=float, default=EPS)
    args = ap.parse_args()

    C = load_cells()
    log('cells=%d' % len(C))

    res = {'eps': args.eps, 'n_cells': len(C), 'targets': {str(k): list(v) for k, v in TARGETS.items()},
           'by_k': {}}
    print()
    print('%-4s %-7s %6s %8s %8s %8s' % ('k', 'arm', 'n', 'cover', 'median', 'p95'))
    for k in sorted(TARGETS):
        out, rows = run(C, k, args.eps)
        res['by_k'][str(k)] = out
        for a in ARMS:
            s = out[a]
            print('%-4d %-7s %6d %7.0f%% %7.2fx %7.2fx'
                  % (k, a, s['n'], 100 * s['coverage'], s['median_overhead'],
                     s['p95_overhead']))
        # At k=2 the log-logistic IS the straight line through the two points, so
        # ll and pchip can only disagree where the fit's parameter bounds bind or
        # where the search for the crossing leaves the measured range. Key the
        # comparison by cell id: rows are appended in fold order, not cell order.
        if k == 2:
            by_id = {a: {r['id']: r for r in rows[a]} for a in ARMS}
            det = []
            for c in C:
                idx = pick(c['rho'], 2)
                r0, s_ = fit_ll(c['rho'][idx], c['delta'][idx])
                bound = (r0 <= 1e-4 * (1 + 1e-9) or r0 >= 1e6 * (1 - 1e-9)
                         or s_ <= 0.1 * (1 + 1e-9) or s_ >= 20.0 * (1 - 1e-9))
                det.append({'id': c['id'], 'rho0': float(r0), 's': float(s_), 'bound': bool(bound)})
            res['k2_fits'] = det
            diffs = []
            for i, d_ in by_id['ll'].items():
                o = by_id['pchip'][i]['overhead']
                if abs(d_['overhead'] - o) > 1e-6:
                    row = next(x for x in det if x['id'] == i)
                    diffs.append({'id': i, 'll': round(d_['overhead'], 2),
                                  'pchip': round(o, 2), 'bound': row['bound'],
                                  's': round(row['s'], 3), 'rho0': round(row['rho0'], 4)})
            res['k2_cells_where_arms_differ'] = diffs
            print('     k=2 cells where ll and pchip differ: %d' % len(diffs))
            for r in diffs:
                print('       %-34s ll=%6.1fx pchip=%5.2fx  s=%.2f rho0=%.4f bound=%s'
                      % (r['id'], r['ll'], r['pchip'], r['s'], r['rho0'], r['bound']))
        print()

    json.dump(res, open(os.path.join(OUT, args.out), 'w'), indent=1)


if __name__ == '__main__':
    main()
