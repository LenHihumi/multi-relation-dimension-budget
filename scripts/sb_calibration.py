"""Calibration protocol: how many labelled truncation points are enough?

Reframes the contribution as a *calibratable, verifiable budget rule* rather than
a label-free predictor of the optimal rank.  For each measured cell we treat the
full quality-versus-rank curve as ground truth, then simulate an operator who
measures quality at k truncation points, fits the log-logistic response law, and
derives a budget.  We report how close that budget is to the oracle rank, and
whether a conservative (bootstrap upper-quantile) budget meets the quality
target.

Log-logistic fitting is done in closed form:

    logit(Delta) = s * log(rho) - s * log(rho0)

so ordinary least squares on (log rho, logit Delta) yields s and rho0 directly.
This is fast enough for 200 bootstrap replicates per (cell, k).

Also measures end-to-end score-computation latency at each rank on one corpus,
so the ROI (calibration labels versus saved storage/latency) is quantified.

Usage: python3 sb_calibration.py --out calibration.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sb_common import OUT, l2n, load_emb, pca_basis  # noqa: E402

EPS = 0.05
KS = [3, 5, 10]
B_BOOT = 200
SEED = 7


def load_theory():
    th = {}
    for p in glob.glob(os.path.join(OUT, 'theory__*.json')):
        d = json.load(open(p))
        th[(d['encoder'], d['dataset'])] = d
    return th


def rho_curve(th, rel, ranks):
    lam = np.array(th['lambda'], float)
    om = lam * np.array(th['tasks'][rel]['eta2'], float)
    cum = np.clip(np.cumsum(om) / om.sum(), 0.0, 1.0)
    return np.array([1.0 - cum[min(r, len(cum)) - 1] for r in ranks])


def fit_ll(rho, y):
    """Closed-form log-logistic fit. Returns (rho0, s)."""
    y = np.clip(np.asarray(y, float), 1e-4, 1 - 1e-4)
    x = np.log(np.maximum(rho, 1e-9))
    z = np.log(y / (1 - y))
    A = np.vstack([x, np.ones_like(x)]).T
    b, a = np.linalg.lstsq(A, z, rcond=None)[0]
    s = max(float(b), 0.05)
    return float(np.exp(-a / s)), s


def pred(rho, rho0, s):
    return 1.0 / (1.0 + (rho0 / np.maximum(rho, 1e-12)) ** s)


def rstar(rho, delta, eps=EPS):
    idx = np.where(delta <= eps)[0]
    return int(idx[0]) if len(idx) else len(rho) - 1


def pick_points(rho, delta, k, rng):
    """k truncation points, log-spaced inside the operating interval rho<=0.15."""
    lo = max(float(rho[rho > 0].min()), 1e-4)
    hi = min(float(rho.max()), 0.15)
    if hi <= lo:
        hi = min(float(rho.max()), lo * 8)
    targets = np.geomspace(lo, hi, k)
    out = []
    for t in targets:
        j = int(np.argmin(np.abs(rho - t)))
        if j not in out:
            out.append(j)
    return np.array(sorted(out), int)


def build_curves(theory):
    """(cell, rho array, delta array, ranks, metric) for all available curves."""
    out = []
    for p in sorted(glob.glob(os.path.join(OUT, 'grid', '*.json'))):
        g = json.load(open(p))
        th = theory.get((g['encoder'], g['dataset']))
        if not th:
            continue
        d = int(th['dim'])
        ranks = [int(r) for r in g['grid'] if int(r) <= d]
        rho = rho_curve(th, 'inst', ranks)  # replaced below per relation
        for cname, cell in g['cells'].items():
            rel = cname.split('@')[0]
            if rel not in th['tasks'] or rel == 'cat1':
                continue
            rhor = rho_curve(th, rel, ranks)
            for kk in ('k1', 'k10'):
                e = cell.get('cen', {}).get(kk)
                if not e:
                    continue
                q = np.array(e['rec_curve'][:len(ranks)], float)
                if len(q) != len(ranks) or q[-1] <= 0:
                    continue
                delta = 1.0 - q / q[-1]
                out.append({'enc': g['encoder'], 'ds': g['dataset'], 'rel': rel,
                            'k': kk, 'metric': 'recall',
                            'ranks': np.array(ranks), 'rho': rhor, 'delta': delta})
    # mAP@10 curves with 23 measured ranks, cached by the standard-metric run.
    # These are dense enough for k = 3/5/10 calibration-point ablations.
    for f in sorted(glob.glob(os.path.join(OUT, 'metric_std_parts', '*.json'))):
        d = json.load(open(f))
        grid = [int(r) for r in d['grid']]
        for rel, v in d.get('cells', {}).items():
            mm = v.get('metrics', {}).get('map10')
            if not mm or 'curve' not in mm:
                continue
            curve = np.array(mm['curve'], float)
            full = float(v.get('full', {}).get('map10', 0) or 0)
            if full <= 0 or len(curve) < 6:
                continue
            n = min(len(curve), len(grid), len(v.get('rho', grid)))
            ranks = np.array(grid[:n], int)
            rho = np.array(v['rho'][:n], float) if len(v.get('rho', [])) >= n else \
                rho_curve(theory[(d['encoder'], d['dataset'])], rel, ranks)
            delta = 1.0 - curve[:n] / full
            out.append({'enc': d['encoder'], 'ds': d['dataset'], 'rel': rel,
                        'k': 'k10', 'metric': 'map10', 'ranks': ranks,
                        'rho': rho, 'delta': delta})
    return out


def latency_bench(theory):
    """End-to-end score-computation time at each rank, CLIP / In-shop."""
    try:
        X = load_emb('clip_vitl14', 'inshop')
    except Exception:  # noqa: BLE001
        return {'error': 'embedding not available'}
    mu, U, ev = pca_basis(X, device='cpu')
    n_q = 1000
    Q = X[:n_q]
    out = {}
    for r in (24, 48, 96, 192, 384, 768):
        P = U[:, :r]
        Xr = np.ascontiguousarray(l2n((X - mu) @ P))
        Qr = np.ascontiguousarray(l2n((Q - mu) @ P))
        ts = []
        for _ in range(3):
            t0 = time.time()
            _ = Qr @ Xr.T
            ts.append(time.time() - t0)
        out[str(r)] = {'bytes_fp32': 4 * r,
                       'score_sec_median': float(np.median(ts)),
                       'flops_rel_to_768': round(r / 768, 3)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='calibration.json')
    ap.add_argument('--no-latency', action='store_true')
    args = ap.parse_args()

    th = load_theory()
    curves = build_curves(th)
    rng = np.random.default_rng(SEED)
    print(f'{len(curves)} curves', flush=True)

    rows, summary = [], {}
    for metric in ('recall', 'map10'):
        cs = [c for c in curves if c['metric'] == metric]
        if not cs:
            continue
        for k in KS:
            rec = []
            for c in cs:
                n = len(c['rho'])
                if n < k + 2:
                    continue
                r_true = rstar(c['rho'], c['delta'])
                idx = pick_points(c['rho'], c['delta'], k, rng)
                if len(idx) < 2:
                    continue
                r0, s = fit_ll(c['rho'][idx], c['delta'][idx])
                r_fit = rstar(c['rho'], pred(c['rho'], r0, s))
                # Protocol step 5: the conservative budget uses the UPPER
                # confidence bound of Delta at each rho, not the upper quantile
                # of r* itself.  Taking the quantile of r* shrinks as k grows
                # (sampling variance falls) and silently loses conservatism;
                # the upper bound of the predicted curve does not.
                preds = []
                for _ in range(B_BOOT):
                    j = rng.choice(idx, size=len(idx), replace=True)
                    try:
                        b0, bs = fit_ll(c['rho'][j], c['delta'][j])
                    except Exception:  # noqa: BLE001
                        continue
                    preds.append(pred(c['rho'], b0, bs))
                if not preds:
                    continue
                delta_upper = np.quantile(np.array(preds), 0.95, axis=0)
                # Bootstrap captures parameter (sampling) uncertainty only.
                # The log-logistic fit also has MODEL bias at the operating
                # point; add the upper quantile of the calibration residuals
                # so the bound stays conservative as k grows.
                resid = c['delta'][idx] - pred(c['rho'][idx], r0, s)
                bias = max(0.0, float(np.quantile(resid, 0.95)))
                delta_upper = np.clip(delta_upper + bias, 0.0, 1.0)
                r_cons = min(rstar(c['rho'], delta_upper), n - 1)
                rec.append({
                    'point_ratio': float(c['ranks'][r_fit] / max(c['ranks'][r_true], 1)),
                    'fit_ratio': float(c['ranks'][r_fit] / max(c['ranks'][r_true], 1)),
                    'within_125': float(0.8 <= c['ranks'][r_fit] / max(c['ranks'][r_true], 1) <= 1.25),
                    'within_2x': float(0.5 <= c['ranks'][r_fit] / max(c['ranks'][r_true], 1) <= 2),
                    'cons_meets': float(c['delta'][r_cons] <= EPS),
                    'cons_overhead': float(c['ranks'][r_cons] / max(c['ranks'][r_true], 1)),
                })
            if not rec:
                continue
            def agg(key):
                return float(np.mean([x[key] for x in rec]))
            summary[f'{metric}|k{k}'] = {
                'n': len(rec), 'calibration_points': k,
                'median_factor': float(2 ** np.median(np.abs(np.log2([x['fit_ratio'] for x in rec])))),
                'within_125': agg('within_125'), 'within_2x': agg('within_2x'),
                'conservative_meets_target': agg('cons_meets'),
                'conservative_overhead': float(np.median([x['cons_overhead'] for x in rec])),
            }
            v = summary[f'{metric}|k{k}']
            print(f'{metric:7s} k={k:2d} n={v["n"]:2d} med={v["median_factor"]:.2f}x '
                  f'<=1.25x={v["within_125"]:.0%} 2x={v["within_2x"]:.0%} '
                  f'cons_meet={v["conservative_meets_target"]:.0%} '
                  f'cons_over={v["conservative_overhead"]:.2f}x', flush=True)

    lat = {} if args.no_latency else latency_bench(th)
    if lat and 'error' not in lat:
        print('latency:', {r: round(v['score_sec_median'], 3) for r, v in lat.items()}, flush=True)

    json.dump({'eps': EPS, 'boot': B_BOOT, 'summary': summary, 'latency': lat},
              open(os.path.join(OUT, args.out), 'w'), indent=1)
    print('wrote', os.path.join(OUT, args.out), flush=True)


if __name__ == '__main__':
    main()
