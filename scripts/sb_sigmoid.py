"""Is the response curve sigmoid, and do its parameters decompose better than the slope?

A single slope c cannot describe a curve that is flat at small rho, steep in the
middle and saturating at large rho. Fitting  Delta = 1/(1+(rho0/rho)^s) instead
gives two parameters with a reading: rho0 is where quality starts to move (the
margin floor expressed in task energy), s is how abruptly it then collapses.

If log rho0 and log s decompose with a much smaller domain share than log c, the
law should be stated on (rho0, s) and "the domain constant" disappears.

Writes outputs/diag/spectral_budget/sigmoid.json.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sb_common import OUT

RHO0_GRID = 10 ** np.linspace(-4, 0.2, 169)
S_GRID = np.linspace(0.4, 6.0, 113)
EPS = 0.05


def load_theory():
    th = {}
    for p in glob.glob(os.path.join(OUT, 'theory__*.json')):
        d = json.load(open(p))
        th[(d['encoder'], d['dataset'])] = d
    return th


def fit_sigmoid(rho, y):
    """Grid search then local refine of Delta = 1/(1+(rho0/rho)^s)."""
    best = (np.inf, None, None)
    for r0 in RHO0_GRID:
        pred = 1.0 / (1.0 + (r0 / np.maximum(rho, 1e-12)) ** S_GRID[:, None])
        err = ((pred - y) ** 2).sum(axis=1)
        i = int(err.argmin())
        if err[i] < best[0]:
            best = (err[i], r0, S_GRID[i])
    _, r0, s = best
    for _ in range(3):
        for cand in np.linspace(r0 * 0.8, r0 * 1.25, 41):
            pred = 1.0 / (1.0 + (cand / np.maximum(rho, 1e-12)) ** s)
            e = ((pred - y) ** 2).sum()
            if e < best[0]:
                best = (e, cand, s)
        _, r0, s = best
        for cand in np.linspace(max(s * 0.85, 0.2), s * 1.18, 41):
            pred = 1.0 / (1.0 + (r0 / np.maximum(rho, 1e-12)) ** cand)
            e = ((pred - y) ** 2).sum()
            if e < best[0]:
                best = (e, r0, cand)
        _, r0, s = best
    return best


def r2(sse, y):
    tot = ((y - y.mean()) ** 2).sum()
    return float(1.0 - sse / tot) if tot > 0 else float('nan')


def nested_share(vals, dom, lib, rel, kk):
    """Sequential sums of squares for the nesting domain > library > relation > k."""
    y = np.asarray(vals, float)
    m = y.mean()
    tot = ((y - m) ** 2).sum()
    out = {}
    resid = y - m
    for name, g in (('domain', dom), ('library', lib), ('relation', rel), ('k', kk)):
        ss = 0.0
        for u in set(g):
            idx = g == u
            if idx.sum() < 2:
                continue
            ss += idx.sum() * (resid[idx].mean()) ** 2
        out[name] = float(ss / tot)
        for u in set(g):
            idx = g == u
            resid[idx] = resid[idx] - resid[idx].mean()
    out['residual'] = float((resid ** 2).sum() / tot)
    return out


def main():
    th = load_theory()
    rows = []
    for p in sorted(glob.glob(os.path.join(OUT, 'grid', '*.json'))):
        g = json.load(open(p))
        t = th.get((g['encoder'], g['dataset']))
        if not t:
            continue
        lam = np.array(t['lambda'], float)
        d = len(lam)
        ranks = [int(r) for r in g['grid'] if int(r) <= d]
        for cname, cell in g['cells'].items():
            rel = cname.split('@')[0]
            if rel not in t['tasks'] or rel == 'cat1':
                continue
            om = lam * np.array(t['tasks'][rel]['eta2'], float)
            cum = np.clip(np.cumsum(om) / om.sum(), 0.0, 1.0)
            rho = np.array([1.0 - cum[r - 1] for r in ranks])
            for k in ('k1', 'k10'):
                e = cell.get('cen', {}).get(k)
                if not e:
                    continue
                rc = np.array(e['rec_curve'][:len(ranks)], float)
                if len(rc) != len(ranks) or rc[-1] <= 0:
                    continue
                y = 1.0 - rc / rc[-1]
                m = (y < 0.98) & (rho > 1e-4)
                if m.sum() < 8:
                    continue
                sse, r0, s = fit_sigmoid(rho[m], y[m])
                c_lin = float((rho[m] @ y[m]) / (rho[m] @ rho[m]))
                r2_lin = r2(float(((c_lin * rho[m] - y[m]) ** 2).sum()), y[m])
                # sigmoid budget: smallest r with Delta(rho_T(r)) <= EPS
                dlt = 1.0 / (1.0 + (r0 / np.maximum(rho, 1e-12)) ** s)
                ok = [r for r, v in zip(ranks, dlt) if v <= EPS]
                r_pred = ok[0] if ok else d
                rows.append(dict(
                    enc=g['encoder'], ds=g['dataset'], rel=rel, k=k,
                    rstar=int(e['rstar05']), r_pred_sig=int(r_pred),
                    rho0=float(r0), s=float(s), r2_sig=r2(sse, y[m]),
                    c_lin=c_lin, r2_lin=r2_lin, n=int(m.sum())))
    print(f'{len(rows)} cells')

    out = {'n_cells': len(rows), 'cells': rows, 'eps': EPS}
    dom = np.array([r['ds'] for r in rows])
    lib = np.array([r['enc'] + '|' + r['ds'] for r in rows])
    rel = np.array([r['rel'] for r in rows])
    kk = np.array([r['k'] for r in rows])

    print('\nfit quality (mean / median R2)')
    for key in ('r2_sig', 'r2_lin'):
        v = np.array([r[key] for r in rows])
        print(f'  {key:8s} {np.nanmean(v):.3f} / {np.nanmedian(v):.3f}')
    wins = np.mean([r['r2_sig'] > r['r2_lin'] for r in rows])
    print(f'  sigmoid better in {wins:.0%} of cells')

    print('\nvariance decomposition (sequential shares)')
    print(f'  {"quantity":8s} {"domain":>8s} {"library":>8s} {"relation":>9s} {"k":>7s} {"resid":>7s}')
    for name, key in (('log c', 'c_lin'), ('log rho0', 'rho0'), ('log s', 's')):
        vals = np.log([r[key] for r in rows])
        sh = nested_share(vals, dom, lib, rel, kk)
        out[f'share_{key}'] = sh
        print(f'  {name:8s} {sh["domain"]:8.1%} {sh["library"]:8.1%} '
              f'{sh["relation"]:9.1%} {sh["k"]:7.1%} {sh["residual"]:7.1%}')
        out[f'log_{key}'] = dict(mean=float(vals.mean()), sd=float(vals.std()),
                                 by_domain={d: float(np.median(vals[dom == d]))
                                            for d in sorted(set(dom))})

    # budget error, sigmoid vs published linear
    def budget_stats(pred):
        ratio = np.array([max(p, 1) / max(r['rstar'], 1)
                          for p, r in zip(pred, rows)])
        l2 = np.abs(np.log2(ratio))
        per = {}
        for d in sorted(set(dom)):
            m = dom == d
            per[d] = dict(n=int(m.sum()), median_factor=float(2 ** np.median(l2[m])),
                          within2x=float(np.mean(l2[m] <= 1)))
        return dict(n=len(rows), median_factor=float(2 ** np.median(l2)),
                    within2x=float(np.mean(l2 <= 1)), by_domain=per)

    lin_pred = []
    for r in rows:
        ok = [x['rstar'] for x in []]
        lin_pred.append(r['rstar'])
    # linear law with per-domain c, as the current reference
    ref = np.zeros(len(rows), int)
    for d in sorted(set(dom)):
        m = dom == d
        c = float(np.median([rows[i]['c_lin'] for i in np.where(m)[0]]))
        for i in np.where(m)[0]:
            rr = rows[i]
            g = json.load(open(os.path.join(OUT, 'grid', f"{rr['enc']}__{rr['ds']}.json")))
            t = th[(rr['enc'], rr['ds'])]
            lam = np.array(t['lambda'], float)
            dd = len(lam)
            ranks = [int(x) for x in g['grid'] if int(x) <= dd]
            om = lam * np.array(t['tasks'][rr['rel']]['eta2'], float)
            cum = np.clip(np.cumsum(om) / om.sum(), 0, 1)
            rho = np.array([1.0 - cum[x - 1] for x in ranks])
            ok = [x for x, v in zip(ranks, rho) if v <= EPS / c]
            ref[i] = ok[0] if ok else dd
    sig_dom = np.zeros(len(rows), int)
    for d in sorted(set(dom)):
        idx = np.where(dom == d)[0]
        r0 = float(np.exp(np.median(np.log([rows[i]['rho0'] for i in idx]))))
        ss = float(np.exp(np.median(np.log([rows[i]['s'] for i in idx]))))
        for i in idx:
            rr = rows[i]
            g = json.load(open(os.path.join(OUT, 'grid', f"{rr['enc']}__{rr['ds']}.json")))
            t = th[(rr['enc'], rr['ds'])]
            lam = np.array(t['lambda'], float)
            dd = len(lam)
            ranks = [int(x) for x in g['grid'] if int(x) <= dd]
            om = lam * np.array(t['tasks'][rr['rel']]['eta2'], float)
            cum = np.clip(np.cumsum(om) / om.sum(), 0, 1)
            rho = np.array([1.0 - cum[x - 1] for x in ranks])
            dlt = 1.0 / (1.0 + (r0 / np.maximum(rho, 1e-12)) ** ss)
            ok = [x for x, v in zip(ranks, dlt) if v <= EPS]
            sig_dom[i] = ok[0] if ok else dd
    r0g = float(np.exp(np.median(np.log([r['rho0'] for r in rows]))))
    sg = float(np.exp(np.median(np.log([r['s'] for r in rows]))))
    sig_glob = np.zeros(len(rows), int)
    for i, rr in enumerate(rows):
        g = json.load(open(os.path.join(OUT, 'grid', f"{rr['enc']}__{rr['ds']}.json")))
        t = th[(rr['enc'], rr['ds'])]
        lam = np.array(t['lambda'], float)
        dd = len(lam)
        ranks = [int(x) for x in g['grid'] if int(x) <= dd]
        om = lam * np.array(t['tasks'][rr['rel']]['eta2'], float)
        cum = np.clip(np.cumsum(om) / om.sum(), 0, 1)
        rho = np.array([1.0 - cum[x - 1] for x in ranks])
        dlt = 1.0 / (1.0 + (r0g / np.maximum(rho, 1e-12)) ** sg)
        ok = [x for x, v in zip(ranks, dlt) if v <= EPS]
        sig_glob[i] = ok[0] if ok else dd
    sig_fix = np.zeros(len(rows), int)
    for d in sorted(set(dom)):
        idx = np.where(dom == d)[0]
        ss_d = float(np.exp(np.median(np.log([rows[i]['s'] for i in idx]))))
        for i in idx:
            rr = rows[i]
            g = json.load(open(os.path.join(OUT, 'grid', f"{rr['enc']}__{rr['ds']}.json")))
            t = th[(rr['enc'], rr['ds'])]
            lam = np.array(t['lambda'], float)
            dd = len(lam)
            ranks = [int(x) for x in g['grid'] if int(x) <= dd]
            om = lam * np.array(t['tasks'][rr['rel']]['eta2'], float)
            cum = np.clip(np.cumsum(om) / om.sum(), 0, 1)
            rho = np.array([1.0 - cum[x - 1] for x in ranks])
            dlt = 1.0 / (1.0 + (r0g / np.maximum(rho, 1e-12)) ** ss_d)
            ok = [x for x, v in zip(ranks, dlt) if v <= EPS]
            sig_fix[i] = ok[0] if ok else dd
    out['params'] = dict(rho0_global=r0g, s_global=sg)
    out['budget'] = dict(
        sigmoid_per_cell=budget_stats([r['r_pred_sig'] for r in rows]),
        sigmoid_per_domain=budget_stats(list(sig_dom)),
        sigmoid_global=budget_stats(list(sig_glob)),
        sigmoid_rho0fixed_s_perdomain=budget_stats(list(sig_fix)),
        linear_per_domain=budget_stats(list(ref)))
    b = out['budget']
    print('\nbudget error:')
    for name in ('sigmoid_per_cell', 'sigmoid_per_domain',
                 'sigmoid_rho0fixed_s_perdomain', 'sigmoid_global', 'linear_per_domain'):
        v = b[name]
        per = ' '.join(f"{d}={x['median_factor']:.2f}x" for d, x in v['by_domain'].items())
        print(f'  {name:20s} {v["median_factor"]:.2f}x {v["within2x"]:5.0%}  {per}')

    json.dump(out, open(os.path.join(OUT, 'sigmoid.json'), 'w'), indent=1)
    print('\nwrote', os.path.join(OUT, 'sigmoid.json'))


if __name__ == '__main__':
    main()
