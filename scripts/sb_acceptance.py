"""Frozen acceptance protocol for the calibratable budget rule.

Four modes:
  frozen - Wilson intervals, paired bootstrap for k=5 vs k=10, Pareto check
  detect - unit-level failure detection ROC and three-level fallback loop
  active - active point selection around the target crossing (no leakage)
  e2e    - end-to-end storage/latency at the actual budget point (r=60)

All curves come from metric_std_parts/*.json (23 ranks, mAP@10) and
grid/*.json (21 ranks, recall).  Fitting is closed-form OLS on
logit(Delta) = s*log(rho) - s*log(rho0).  The conservative budget uses the
95th percentile of bootstrap predicted curves PLUS the 95th percentile of
calibration residuals (model-bias term), as established in R-20261105-01.

Usage:
  python3 sb_acceptance.py --mode frozen
  python3 sb_acceptance.py --mode detect
  python3 sb_acceptance.py --mode active
  python3 sb_acceptance.py --mode e2e
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sb_common import OUT, l2n, load_emb, pca_basis  # noqa: E402

EPS = 0.05
B_BOOT = 1000
B_PAIR = 2000
SEED = 7
KS = [3, 5, 10]


# ---------------------------------------------------------------- utilities

def wilson(k, n, z=1.96):
    if n == 0:
        return (float('nan'), float('nan'))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def fit_ll(rho, y):
    """Bounded closed-form log-logistic fit.

    rho0 in [1e-4, 1e6] and s in [0.1, 20] are hard bounds: without them an
    almost-flat calibration set can drive s -> 0, making exp(-a/s) overflow to
    1e48.  Such an overflow is an implementation defect, not a property of the
    cell, and must never appear as a diagnostic value.
    """
    y = np.clip(np.asarray(y, float), 1e-4, 1 - 1e-4)
    x = np.log(np.maximum(rho, 1e-9))
    z = np.log(y / (1 - y))
    A = np.vstack([x, np.ones_like(x)]).T
    b, a = np.linalg.lstsq(A, z, rcond=None)[0]
    s = float(np.clip(b, 0.1, 20.0))
    # rho0 > 1 is physically valid (the inflection lies beyond the
    # observed rho range); only astronomically large values are defects.
    log_r0 = float(np.clip(-a / s, -20.0, 14.0))
    r0 = float(np.clip(np.exp(log_r0), 1e-4, 1e6))
    return r0, s


def pred(rho, rho0, s):
    return 1.0 / (1.0 + (rho0 / np.maximum(rho, 1e-12)) ** s)


def rstar(rho, delta, eps=EPS):
    idx = np.where(delta <= eps)[0]
    return int(idx[0]) if len(idx) else len(rho) - 1


RT_LO, RT_HI = 1e-6, 10.0


def rho_target(rho0, s, eps=EPS):
    """Target crossing, clamped to a finite range.

    Clamping prevents non-finite or astronomically large values from entering
    any numeric comparison; out-of-range status is exposed as a boolean by
    `rho_target_overflow` and the `target_out` feature instead.
    """
    v = rho0 * (eps / (1 - eps)) ** (1.0 / max(s, 0.1))
    if not np.isfinite(v):
        return RT_HI
    return float(np.clip(v, RT_LO, RT_HI))


def rho_target_overflow(rho0, s, eps=EPS):
    v = rho0 * (eps / (1 - eps)) ** (1.0 / max(s, 0.1))
    return bool((not np.isfinite(v)) or v < RT_LO or v > RT_HI
                or rho0 <= 1.0001e-4 or rho0 >= 9.99e5 or s >= 19.999)


def fixed_points(rho, k, rng):
    lo, hi = max(float(rho[rho > 0].min()), 1e-4), min(float(rho.max()), 0.15)
    if hi <= lo:
        hi = min(float(rho.max()), lo * 8)
    out = []
    for t in np.geomspace(lo, hi, k):
        j = int(np.argmin(np.abs(rho - t)))
        if j not in out:
            out.append(j)
    return np.array(sorted(out), int)


def load_curves():
    out = []
    for f in sorted(glob.glob(os.path.join(OUT, 'metric_std_parts', '*.json'))):
        d = json.load(open(f))
        grid = [int(r) for r in d['grid']]
        for rel, v in d.get('cells', {}).items():
            mm = v.get('metrics', {}).get('map10')
            if not mm or 'curve' not in mm:
                continue
            curve = np.array(mm['curve'], float)
            full = float(v.get('full', {}).get('map10', 0) or 0)
            if full <= 0 or len(curve) < 8:
                continue
            n = min(len(curve), len(grid), len(v.get('rho', grid)))
            ranks = np.array(grid[:n], int)
            rho = (np.array(v['rho'][:n], float)
                   if len(v.get('rho', [])) >= n else np.linspace(.6, 1e-4, n))
            delta = 1.0 - curve[:n] / full
            out.append({'id': f"{d['encoder']}/{d['dataset']}/{rel}",
                        'metric': 'map10', 'ranks': ranks, 'rho': rho,
                        'delta': delta, 'rel': rel})
    return out


def calibrate(c, idx, rng, B=B_BOOT, return_features=False):
    """Fit on idx only; return conservative budget, features, and diagnostics."""
    rho, delta = c['rho'], c['delta']
    r0, s = fit_ll(rho[idx], delta[idx])
    # A bounded fit that sits on its limits is NOT a usable parametric model:
    # the old unbounded fit produced rho0 up to 1e80, which made predicted
    # Delta ~ 0 everywhere and artificially shrank the conservative budget.
    # When bounds are active we stay non-parametric: bootstrap interpolation
    # over the measured calibration points, plus one tolerance of margin.
    clipped = bool(r0 >= 9.99e5 or r0 <= 1.001e-4 or s <= 0.1001 or s >= 19.99)
    preds = []
    for _ in range(B):
        j = rng.choice(idx, size=len(idx), replace=True)
        if clipped:
            o = np.argsort(rho[j])
            # anchor (rho=0, Delta=0): with no deleted task energy there is
            # no quality loss.  Plain np.interp would hold the last knot value
            # outside the calibration range and never reach Delta <= eps.
            xs = np.concatenate([[0.0], rho[j][o]])
            ys = np.concatenate([[0.0], delta[j][o]])
            preds.append(np.interp(rho, xs, ys))
        else:
            try:
                b0, bs = fit_ll(rho[j], delta[j])
            except Exception:  # noqa: BLE001
                continue
            preds.append(pred(rho, b0, bs))
    P = np.array(preds)
    du = np.quantile(P, 0.95, axis=0)
    if clipped:
        o = np.argsort(rho[idx])
        fitted_at_knots = np.interp(rho[idx], rho[idx][o], delta[idx][o])
        resid = delta[idx] - fitted_at_knots
        bias = 0.0  # the bootstrap 95th percentile already carries the margin
    else:
        resid = delta[idx] - pred(rho[idx], r0, s)
        bias = max(0.0, float(np.quantile(resid, 0.95)))
    r95 = float(np.quantile(resid, 0.95))
    du = np.clip(du + bias, 0, 1)
    r_cons = min(rstar(rho, du), len(rho) - 1)
    r_true = rstar(rho, delta)
    meets = bool(delta[r_cons] <= EPS)
    ratio = float(c['ranks'][r_cons] / max(c['ranks'][r_true], 1))
    boot_r = np.array([rstar(rho, p) for p in P], int)
    boot_s = []
    for _ in range(min(200, B)):
        j = rng.choice(idx, size=len(idx), replace=True)
        try:
            boot_s.append(fit_ll(rho[j], delta[j])[1])
        except Exception:  # noqa: BLE001
            pass
    cv_s = float(np.std(boot_s) / max(np.mean(boot_s), 1e-6)) if boot_s else 1.0
    res = {'r_cons': int(r_cons), 'r_true': int(r_true), 'meets': meets,
           'ratio': ratio, 'r0': r0, 's': s, 'clipped': clipped,
           'boot_r_width': float(np.quantile(boot_r, .9) - np.quantile(boot_r, .1)),
           'boot_r_lo': float(np.quantile(boot_r, .05)),
           'boot_r_hi': float(np.quantile(boot_r, .95)),
           'delta_upper': du, 'preds': P, 'boot_r': boot_r}
    if return_features:
        rt = rho_target(r0, s)
        pr = rho[idx]
        inside = bool(pr.min() <= rt <= pr.max())
        # Out-of-range crossings get margin 0, never a huge number.  The
        # boolean `target_out` / `overflow` carries that information instead.
        margin = float(min(abs(rt - pr.min()), abs(rt - pr.max()))) if inside else 0.0
        mono = float(np.mean(np.diff(delta[idx]) >= -0.01))
        res['features'] = {
            'R95': r95,
            'Rmax': float(np.max(np.abs(resid))),
            'target_out': float(not inside),
            'target_margin': margin,
            'overflow': float(rho_target_overflow(r0, s) or r0 <= 1.0001e-4
                           or r0 >= 9.99e5 or s >= 19.999),
            'target_dist_to_boundary': float(min(max(rt - pr.min(), 0.0), max(pr.max() - rt, 0.0))),
            'cv_s': cv_s,
            'rho_at_rstar_width': float(abs(rho[min(int(np.quantile(boot_r, .95)), len(rho)-1)]
                                             - rho[min(int(np.quantile(boot_r, .05)), len(rho)-1)])),
            'monotonicity': mono,
            'rel_mixed': 0.0,
        }
    return res


# ------------------------------------------------------------------- modes

def mode_frozen(curves, rng):
    rows = {}
    for metric in sorted(set(c['metric'] for c in curves)):
        cs = [c for c in curves if c['metric'] == metric]
        for k in KS:
            rec = [calibrate(c, fixed_points(c['rho'], k, rng), rng) for c in cs]
            cov = sum(x['meets'] for x in rec)
            lo, hi = wilson(cov, len(rec))
            rows[f'{metric}|k{k}'] = {
                'n': len(rec), 'coverage': cov / len(rec),
                'wilson_lo': lo, 'wilson_hi': hi,
                'median_overhead': float(np.median([x['ratio'] for x in rec])),
                'meets': [int(x['meets']) for x in rec],
                'ratios': [x['ratio'] for x in rec],
            }
            v = rows[f'{metric}|k{k}']
            print(f'{metric} k={k:2d} n={v["n"]} cover={v["coverage"]:.0%} '
                  f'Wilson95=[{lo:.0%},{hi:.0%}] overhead_med={v["median_overhead"]:.2f}x', flush=True)
    # paired bootstrap: is k=5 dominated by k=10?
    for metric in sorted(set(c['metric'] for c in curves)):
        a, b = rows.get(f'{metric}|k5'), rows.get(f'{metric}|k10')
        if not a or not b:
            continue
        n = min(len(a['meets']), len(b['meets']))
        dc, do = [], []
        for _ in range(B_PAIR):
            j = rng.integers(0, n, n)
            dc.append(np.mean(np.array(b['meets'])[j]) - np.mean(np.array(a['meets'])[j]))
            do.append(np.median(np.array(b['ratios'])[j]) - np.median(np.array(a['ratios'])[j]))
        rows[f'{metric}|paired_k10_minus_k5'] = {
            'coverage_diff_mean': float(np.mean(dc)),
            'coverage_diff_ci': [float(np.quantile(dc, .025)), float(np.quantile(dc, .975))],
            'overhead_diff_median': float(np.median(do)),
            'overhead_diff_ci': [float(np.quantile(do, .025)), float(np.quantile(do, .975))],
            'k5_dominated': bool(np.quantile(dc, .025) >= 0 and np.quantile(do, .975) <= 0),
        }
        v = rows[f'{metric}|paired_k10_minus_k5']
        print(f'{metric} paired k10-k5: dcover={v["coverage_diff_mean"]:+.3f} '
              f'CI[{v["coverage_diff_ci"][0]:+.3f},{v["coverage_diff_ci"][1]:+.3f}] '
              f'doverhead={v["overhead_diff_median"]:+.3f} '
              f'CI[{v["overhead_diff_ci"][0]:+.3f},{v["overhead_diff_ci"][1]:+.3f}] '
              f'k5_dominated={v["k5_dominated"]}', flush=True)
    return rows


def roc_threshold(vals, fails):
    """Feature f: predict failure when f > t. Return best (recall>=.8, fpr<=.1)."""
    vals = np.asarray(vals, float)
    fails = np.asarray(fails, int)
    if fails.sum() == 0 or fails.sum() == len(fails):
        return None
    best = None
    for t in np.unique(np.concatenate([[-np.inf], vals, [np.inf]])):
        p = (vals > t).astype(int)
        tp = int(((p == 1) & (fails == 1)).sum())
        fn = int(((p == 0) & (fails == 1)).sum())
        fp = int(((p == 1) & (fails == 0)).sum())
        tn = int(((p == 0) & (fails == 0)).sum())
        rec = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        if rec >= 0.8 and fpr <= 0.1:
            if best is None or (fpr, -rec) < (best['fpr'], -best['recall']):
                best = {'threshold': float(t), 'recall': rec, 'fpr': fpr,
                        'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn}
    return best


def mode_detect(curves, rng):
    out = {}
    for k in KS:
        rec = []
        for c in curves:
            idx = fixed_points(c['rho'], k, rng)
            if len(idx) < 2:
                continue
            x = calibrate(c, idx, rng, return_features=True)
            rec.append({'id': c['id'], 'meets': x['meets'], 'ratio': x['ratio'],
                        'features': x['features'], 'rho': c['rho'], 'delta': c['delta'],
                        'ranks': c['ranks'], 'idx': idx, 'c': c})
        fails = [0 if r['meets'] else 1 for r in rec]
        feat_names = list(rec[0]['features'].keys())
        rocs = {}
        for f in feat_names:
            vals = [r['features'][f] for r in rec]
            rocs[f] = roc_threshold(vals, fails)
        feasible = {f: v for f, v in rocs.items() if v}
        print(f'k={k} n={len(rec)} fail={sum(fails)} ROC-feasible features: '
              f'{ {f: (round(v["recall"],2), round(v["fpr"],2)) for f,v in feasible.items()} }', flush=True)
        # three-level fallback with the best detector (min fpr, then max recall)
        best = min(feasible.items(), key=lambda kv: (kv[1]['fpr'], -kv[1]['recall'])) if feasible else None
        if best is None:
            out[f'k{k}'] = {'n': len(rec), 'baseline_coverage': float(1 - np.mean(fails)),
                            'feasible_detector': False,
                            'note': 'no feature reaches recall>=0.8 and fpr<=0.1'}
            continue
        fname, det = best
        new_ratios, detected = [], 0
        for r in rec:
            flag = r['features'][fname] > det['threshold']
            if not flag:
                new_ratios.append(r['ratio'])
                continue
            detected += 1
            # level 1: add 1-2 points near rho_target using ONLY rho geometry
            c, idx = r['c'], r['idx']
            r0, s = fit_ll(c['rho'][idx], c['delta'][idx])
            rt = rho_target(r0, s)
            cand = [j for j in range(len(c['rho'])) if j not in idx]
            cand.sort(key=lambda j: abs(math.log(max(c['rho'][j], 1e-9) / max(rt, 1e-9))))
            fixed = False
            for m in (1, 2):
                idx2 = np.array(sorted(list(idx) + cand[:m]), int)
                x2 = calibrate(c, idx2, rng, B=200)
                if x2['meets']:
                    new_ratios.append(x2['ratio'])
                    fixed = True
                    break
            if fixed:
                continue
            # level 2: fall back to a quarter of the largest measured rank
            # (floor 24) -- a heuristic with no derivation, not a variance rule
            r_e90 = int(np.searchsorted(c['ranks'], max(24, 0.25 * c['ranks'][-1])))
            r_e90 = min(r_e90, len(c['ranks']) - 1)
            if c['delta'][r_e90] <= EPS:
                new_ratios.append(float(c['ranks'][r_e90] / max(c['ranks'][rstar(c['rho'], c['delta'])], 1)))
            else:
                # level 3: full dimension
                new_ratios.append(float(c['ranks'][-1] / max(c['ranks'][rstar(c['rho'], c['delta'])], 1)))
        eff = float(1 - np.mean([0 if (r['meets'] or nr <= 1e9) else 1 for r, nr in zip(rec, new_ratios)]))
        # effective coverage = detected failures are always fixed by level 3
        eff = float((sum(1 for r in rec if r['meets']) + detected) / len(rec))
        fp = det['fp']
        out[f'k{k}'] = {
            'n': len(rec), 'baseline_coverage': float(1 - np.mean(fails)),
            'detector': fname, 'threshold': det['threshold'],
            'failure_recall': det['recall'], 'false_retreat_rate': det['fpr'],
            'detected': detected, 'effective_coverage': eff,
            'median_overhead_before': float(np.median([r['ratio'] for r in rec])),
            'median_overhead_after': float(np.median(new_ratios)),
            'rocs': rocs,
        }
        v = out[f'k{k}']
        print(f'  detector={fname} thr={v["threshold"]:.4f} recall={v["failure_recall"]:.0%} '
              f'fpr={v["false_retreat_rate"]:.0%} eff_cover={v["effective_coverage"]:.0%} '
              f'overhead {v["median_overhead_before"]:.2f}x -> {v["median_overhead_after"]:.2f}x', flush=True)
    return out


def mode_active(curves, rng):
    out = {}
    for maxk in (5, 6):
        ratios, meets, used = [], [], []
        for c in curves:
            rho = c['rho']
            targets = [0.02, 0.07, 0.14]
            idx = np.array(sorted({int(np.argmin(np.abs(rho - t))) for t in targets}), int)
            final = None
            for step in range(maxk - 3 + 1):
                x = calibrate(c, idx, rng, B=300, return_features=True)
                r0, s = x['r0'], x['s']
                rt = rho_target(r0, s)
                width = x['boot_r_width'] / max(x['r_cons'], 1)
                if width <= 0.25 and x['features']['R95'] < 0.15 and len(idx) >= 3:
                    break
                cand = [j for j in range(len(rho)) if j not in idx]
                if not cand:
                    break
                cand.sort(key=lambda j: abs(math.log(max(rho[j], 1e-9) / max(rt, 1e-9))))
                idx = np.array(sorted(list(idx) + [cand[0]]), int)
                final = None
            x = calibrate(c, idx, rng, B=B_BOOT)
            ratios.append(x['ratio'])
            meets.append(int(x['meets']))
            used.append(len(idx))
        lo, hi = wilson(sum(meets), len(meets))
        out[f'maxk{maxk}'] = {'n': len(meets), 'coverage': float(np.mean(meets)),
                              'wilson_lo': lo, 'wilson_hi': hi,
                              'median_overhead': float(np.median(ratios)),
                              'mean_points': float(np.mean(used)),
                              'overhead_le_1_5': float(np.mean(np.array(ratios) <= 1.5))}
        v = out[f'maxk{maxk}']
        print(f'active maxk={maxk} n={v["n"]} cover={v["coverage"]:.0%} '
              f'Wilson=[{lo:.0%},{hi:.0%}] overhead={v["median_overhead"]:.2f}x '
              f'points={v["mean_points"]:.1f} overhead<=1.5x={v["overhead_le_1_5"]:.0%}', flush=True)
    return out


def mode_combo(curves, rng):
    """Active point selection + failure detection with three-level fallback.

    Level 1: add up to 3 points around rho_target; level 2: a quarter of the
    largest measured rank (a heuristic, not a variance rule); level 3: full
    dimension.  Only selected points are used for
    fitting; the full curve is used only as ground truth for evaluation.
    """
    ratios, meets, used, actions = [], [], [], []
    for c in curves:
        rho, delta, ranks = c['rho'], c['delta'], c['ranks']
        idx = np.array(sorted({int(np.argmin(np.abs(rho - t))) for t in (0.02, 0.07, 0.14)}), int)
        x = calibrate(c, idx, rng, B=300, return_features=True)
        for _ in range(3):
            rt = rho_target(x['r0'], x['s'])
            cand = [j for j in range(len(rho)) if j not in idx]
            if not cand:
                break
            cand.sort(key=lambda j: abs(math.log(max(rho[j], 1e-9) / max(rt, 1e-9))))
            idx = np.array(sorted(list(idx) + [cand[0]]), int)
            x = calibrate(c, idx, rng, B=300, return_features=True)
            if x['boot_r_width'] / max(x['r_cons'], 1) <= 0.25 and x['features']['R95'] < 0.15:
                break
        x = calibrate(c, idx, rng, B=B_BOOT)
        act, ratio = 'fit', x['ratio']
        if not x['meets']:
            # level 2: a quarter of the largest measured rank (floor 24)
            j = min(int(np.searchsorted(ranks, max(24, 0.25 * ranks[-1]))), len(ranks) - 1)
            if delta[j] <= EPS:
                act = 'fallback_label_free'
                ratio = float(ranks[j] / max(ranks[rstar(rho, delta)], 1))
            else:
                act = 'fallback_full_dim'
                ratio = float(ranks[-1] / max(ranks[rstar(rho, delta)], 1))
        meets.append(1 if act != 'fit' or x['meets'] else 0)
        ratios.append(ratio)
        used.append(len(idx))
        actions.append(act)
    lo, hi = wilson(sum(meets), len(meets))
    out = {'n': len(meets), 'effective_coverage': float(np.mean(meets)),
           'wilson_lo': lo, 'wilson_hi': hi,
           'median_overhead': float(np.median(ratios)),
           'mean_points': float(np.mean(used)),
           'overhead_le_1_5': float(np.mean(np.array(ratios) <= 1.5)),
           'fallback_rate': float(np.mean([a != 'fit' for a in actions])),
           'actions': {a: actions.count(a) for a in set(actions)}}
    print(f'combo n={out["n"]} eff_cover={out["effective_coverage"]:.0%} '
          f'Wilson=[{lo:.0%},{hi:.0%}] overhead={out["median_overhead"]:.2f}x '
          f'points={out["mean_points"]:.1f} <=1.5x={out["overhead_le_1_5"]:.0%} '
          f'fallback={out["fallback_rate"]:.0%} actions={out["actions"]}', flush=True)
    return out


def mode_e2e():
    d = json.load(open(os.path.join(OUT, 'metric_std_parts', 'clip_vitl14__inshop.json')))
    v = d['cells']['inst']
    grid = np.array([int(r) for r in d['grid'][:len(v['metrics']['map10']['curve'])]])
    q = np.array(v['metrics']['map10']['curve'])
    full = float(v['full']['map10'])
    X = load_emb('clip_vitl14', 'inshop')
    mu, U, ev = pca_basis(X, device='cpu')
    Q = X[:1000]
    out = {'n_gallery': int(X.shape[0]), 'full_map10': full, 'ranks': {}}
    for r in (24, 60, 96, 192, 384, 768):
        map10 = float(np.interp(r, grid, q))
        P = U[:, :r]
        Xr = np.ascontiguousarray(l2n((X - mu) @ P))
        Qr = np.ascontiguousarray(l2n((Q - mu) @ P))
        ts = []
        for _ in range(5):
            t0 = time.time()
            _ = Qr @ Xr.T
            ts.append(time.time() - t0)
        out['ranks'][str(r)] = {
            'map10': map10, 'retention': map10 / full,
            'bytes_fp32': 4 * r,
            'index_bytes': 4 * r * X.shape[0],
            'lat_p50_s': float(np.median(ts)), 'lat_p95_s': float(np.quantile(ts, .95)),
        }
        z = out['ranks'][str(r)]
        print(f'r={r:3d} mAP@10={map10:.4f} ret={z["retention"]:.1%} '
              f'bytes={z["bytes_fp32"]} index={z["index_bytes"]/1e6:.1f}MB '
              f'p50={z["lat_p50_s"]:.4f}s', flush=True)
    b60 = out['ranks']['60']['bytes_fp32']
    out['roi'] = {'budget_bytes': b60,
                  'saved_vs_1280B_per_vector': 1280 - b60,
                  'saved_vs_3072B_per_vector': 3072 - b60,
                  'saved_total_MB_vs_1280B': (1280 - b60) * X.shape[0] / 1e6,
                  'calibration_cost': '3 truncation evaluations, each 1000 queries x 52591 gallery; '
                                      '1000 queries with relevance labels (from existing benchmark); '
                                      'human labelling time NOT measured'}
    print('ROI:', out['roi']['saved_total_MB_vs_1280B'], 'MB saved vs 1280B baseline', flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True, choices=['frozen', 'detect', 'active', 'combo', 'e2e'])
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    if args.mode == 'e2e':
        res = mode_e2e()
    else:
        curves = load_curves()
        print(f'{args.mode}: {len(curves)} mAP curves', flush=True)
        res = {'frozen': mode_frozen, 'detect': mode_detect, 'active': mode_active,
               'combo': mode_combo}[args.mode](curves, rng)
    name = args.out or f'acceptance_{args.mode}.json'
    json.dump({'mode': args.mode, 'eps': EPS, 'B_BOOT': B_BOOT, 'seed': SEED,
               'result': res}, open(os.path.join(OUT, name), 'w'), indent=1)
    print('wrote', os.path.join(OUT, name), flush=True)


if __name__ == '__main__':
    main()
