"""Control: is d_eff just a proxy for retained-subspace energy?

Reviewer 2 (#3) argues that "retained subspace energy" -- the share of centred-vector energy
kept by the truncated subspace -- is the quantity the PCA-truncation literature uses, and that
our d_eff (participation ratio of retained TASK energy) is a different thing, so rejecting
"dimension-related mechanisms" with d_eff is a straw man.

Both quantities are computable from data we already ship: eigenvalues and eta^2 per direction
(theory/*.json), the measured map@10 curve per cell (metric_std_parts/*.json), and the deleted
task-energy fraction rho per rank.  This runs the horse race under ONE specification -- cell
fixed effects, log rho and its square, then each candidate predictor -- and reports which one
carries the variation.

Scope, so the numbers are not mistaken for §4.3's: the panel here is every (cell, rank) point
with 0 < Delta < 1 on the 19-cell grid (387 points), not the 151-point within-cell subset the
paper fits, so the R^2 levels differ.  What transfers is the comparison: the two predictors
enter the same regression, and the one that survives controls the other.

Not part of the 310 numeric checks: those cover published numbers, this covers a control that
is reported in the rebuttal.  Read-only; exit 0 if d_eff dominates (retained energy's |t|
collapses when both are present), else 2.
"""
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')


def cell_panel(enc, ds):
    th = json.load(open(os.path.join(DATA, 'theory', 'theory__%s__%s.json' % (enc, ds))))
    parts = json.load(open(os.path.join(DATA, 'metric_std_parts', '%s__%s.json' % (enc, ds))))
    out = []
    for task in ('inst', 'cat'):
        if task not in parts.get('cells', {}) or task not in th.get('tasks', {}):
            continue
        lam = np.asarray(th['lambda'], float)
        eta = np.asarray(th['tasks'][task]['eta2'], float)
        n = min(len(lam), len(eta))
        lam, eta = lam[:n], eta[:n]
        w = lam * eta ** 2
        cw, cw2, cl, cl_all = np.cumsum(w), np.cumsum(w ** 2), np.cumsum(lam), lam.sum()
        rho = np.asarray(parts['cells'][task]['rho'], float)
        curve = np.asarray(parts['cells'][task]['metrics']['map10']['curve'], float)
        grid = np.asarray(parts['grid'], int)
        full = curve[-1]
        for r, r0, m in zip(grid, rho, curve):
            r = min(int(r), n)
            if m <= 0 or full <= 0:
                continue
            delta = 1.0 - m / full
            if not (0.0 < delta < 1.0):
                continue
            deff = (cw[r - 1] ** 2) / max(cw2[r - 1], 1e-30)
            out.append(dict(cell='%s/%s/%s' % (enc, ds, task), rho=r0, delta=delta,
                            logdeff=np.log(max(deff, 1e-9)),
                            loglshare=np.log(max(cl[r - 1] / cl_all, 1e-12))))
    return out


def ols(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    ss = float(resid @ resid)
    n, k = X.shape
    sigma2 = ss / (n - k)
    cov = sigma2 * np.linalg.inv(X.T @ X)
    t = beta / np.sqrt(np.diag(cov))
    r2 = 1.0 - ss / float(((y - y.mean()) ** 2).sum())
    return beta, t, r2


def fit(rows, predictors):
    cells = sorted({r['cell'] for r in rows})
    y = np.array([np.log(r['delta'] / (1 - r['delta'])) for r in rows])
    lp = np.array([np.log(max(r['rho'], 1e-9)) for r in rows])
    cols = [np.ones(len(rows)), lp, lp ** 2]
    names = ['const', 'log rho', 'log rho^2']
    for p in predictors:
        cols.append(np.array([r[p] for r in rows]))
        names.append(p)
    for c in cells[:-1]:          # cell fixed effects: one dummy less than cells
        cols.append(np.array([1.0 if r['cell'] == c else 0.0 for r in rows]))
        names.append('fe')
    beta, t, r2 = ols(np.column_stack(cols), y)
    keep = [i for i, nm in enumerate(names) if nm != 'fe']
    return {names[i]: (float(beta[i]), float(t[i])) for i in keep}, r2, len(rows)


def main():
    rows = []
    for f in sorted(glob.glob(os.path.join(DATA, 'theory', 'theory__*__*.json'))):
        base = os.path.basename(f)[len('theory__'):-len('.json')]
        if '__' not in base:
            continue
        enc, ds = base.split('__', 1)
        if ds not in ('inshop', 'instre', 'sop'):
            continue
        try:
            rows.extend(cell_panel(enc, ds))
        except FileNotFoundError:
            continue
    print('panel points:', len(rows), 'cells:', len({r['cell'] for r in rows}), '\n')

    m0, r20, n = fit(rows, [])
    print('M0  base only                     R2=%.3f  n=%d' % (r20, n))
    t_both = {}
    for name, preds in (('M1  + log d_eff', ['logdeff']),
                        ('M2  + log retained lambda', ['loglshare']),
                        ('M3  + both', ['logdeff', 'loglshare'])):
        coef, r2, n = fit(rows, preds)
        if name.startswith('M3'):
            t_both = {k: v[1] for k, v in coef.items()}
        extra = ' '.join('%s=%.3f (t=%.1f)' % (k, v[0], v[1]) for k, v in coef.items()
                         if k not in ('const', 'log rho', 'log rho^2'))
        print('%-30s R2=%.3f  dR2=%+.3f  %s' % (name, r2, r2 - r20, extra))

    dominated = abs(t_both.get('loglshare', 0.0)) < 2.0 and abs(t_both.get('logdeff', 0.0)) > 2.0
    print('\nretained-subspace energy survives d_eff: %s -> %s'
          % ('no' if dominated else 'yes',
             'd_eff is not a proxy for it, it subsumes it' if dominated
             else 'the two carry separate variation; report both'))
    return 0 if dominated else 2


if __name__ == '__main__':
    main()
