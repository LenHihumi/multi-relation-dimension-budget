"""Is a learned zero-shot predictor a substitute for on-site calibration?

The method's claim is that (rho0, s) must be calibrated with a few truncation
evaluations.  A reviewer will ask: why not learn a predictor from corpus
features instead?  This script answers it on the same 19 mAP@10 cells:

  arm A  zero-shot ridge predictor: spectral features -> log r*, leave-one-cell-out
  arm B  naive predictor: training-fold median r*
  arm C  on-site calibration with 3 truncation points (numbers from
         acceptance_frozen.json, i.e. r_cons / r_true per cell)

Fairness notes recorded in the output: the predictor needs a FULL rank grid per
training cell (23 retrieval evaluations each), while calibration needs 3 points
per target cell and no training set at all.

Usage: python3 sb_learned.py --out learned.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sb_common import OUT  # noqa: E402

EPS = 0.05
ALPHA = 1.0
SEED = 7


def load_theory():
    th = {}
    for p in glob.glob(os.path.join(OUT, 'theory__*.json')):
        d = json.load(open(p))
        th[(d['encoder'], d['dataset'])] = d
    return th


def cells(th):
    out = []
    for p in sorted(glob.glob(os.path.join(OUT, 'metric_std_parts', '*.json'))):
        d = json.load(open(p))
        key = (d['encoder'], d['dataset'])
        if key not in th:
            continue
        t = th[key]
        lam = np.array(t['lambda'], float)
        grid = np.array([int(r) for r in d['grid']], int)
        for rel, cell in d.get('cells', {}).items():
            mm = cell.get('metrics', {}).get('map10')
            if not mm or 'curve' not in mm or rel not in t.get('tasks', {}):
                continue
            curve = np.array(mm['curve'], float)
            full = float(cell.get('full', {}).get('map10', 0) or 0)
            if full <= 0:
                continue
            n = min(len(curve), len(grid), len(cell.get('rho', [])))
            if n < 6:
                continue
            rk, cv = grid[:n], curve[:n]
            delta = 1.0 - cv / full
            ok = np.where(delta <= EPS)[0]
            r_true = int(rk[ok[0]]) if len(ok) else int(rk[-1])
            eta = np.array(t['tasks'][rel]['eta2'], float)
            p = lam / lam.sum()
            ent = float(-(p * np.log(np.clip(p, 1e-12, None))).sum() / np.log(len(p)))
            pr = float(lam.sum() ** 2 / (lam ** 2).sum())
            rho15 = float(cell['rho'][min(n - 1, int(np.argmin(np.abs(np.array(cell['rho'][:n]) - 0.15))))])
            feat = [np.log(t['dim']), np.log(pr), ent,
                    float(np.log(lam[0] / max(lam[1], 1e-12))),
                    rho15, float(np.std(eta)),
                    1.0 if rel == 'inst' else 0.0,
                    1.0 if 'wemm' in d['encoder'] else 0.0]
            # indices 4 and 5 (rho15, std eta) require the group labels; the
            # rest are label-free spectral or meta features.
            out.append({'id': '%s/%s/%s' % (d['encoder'], d['dataset'], rel),
                        'enc': d['encoder'], 'ds': d['dataset'], 'rel': rel,
                        'dim': int(t['dim']), 'r_true': r_true, 'grid': rk,
                        'delta': delta, 'feat': feat,
                        'feat_spec': [feat[i] for i in (0, 1, 2, 3, 6, 7)]})
    return out


def ridge_fit(X, y, alpha=ALPHA):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    Z1 = np.hstack([np.ones((len(Z), 1)), Z])
    pen = np.eye(Z1.shape[1]) * alpha
    pen[0, 0] = 0.0
    beta = np.linalg.solve(Z1.T @ Z1 + pen, Z1.T @ y)
    return mu, sd, beta


def ridge_pred(model, X):
    mu, sd, beta = model
    Z = (X - mu) / sd
    return np.hstack([np.ones((len(Z), 1)), Z]) @ beta


def err_stats(pred, true):
    m = np.isfinite(pred)
    r = np.log2(np.clip(pred[m], 1, None) / np.clip(true[m], 1, None))
    if not len(r):
        return {'median_factor': float('nan'), 'within_2x': float('nan'),
                'within_1_25x': float('nan'), 'n': 0}
    return {'median_factor': float(2 ** np.median(np.abs(r))),
            'within_2x': float(np.mean(np.abs(r) <= 1.0)),
            'within_1_25x': float(np.mean(np.abs(r) <= np.log2(1.25))),
            'n': int(len(r))}


def loocv(C, y, margin_q=None, key='feat'):
    """LOOCV predictions, with an optional NESTED conservative margin.

    margin_q=0.90 adds the 90th percentile of the *training-fold* residuals to
    the held-out prediction.  Estimating the margin inside each fold is what
    makes this comparable to calibration, whose r*_cons is an upper confidence
    bound rather than a point estimate.
    """
    preds = np.zeros(len(C))
    for i in range(len(C)):
        idx = [j for j in range(len(C)) if j != i]
        X = np.array([C[j][key] for j in idx], float)
        model = ridge_fit(X, y[idx])
        m = 0.0
        if margin_q is not None:
            m = float(np.quantile(y[idx] - ridge_pred(model, X), margin_q))
        preds[i] = ridge_pred(model, np.array([C[i][key]], float))[0] + m
    return preds


def group_cv(C, y, group_key, margin_q=None, key='feat'):
    """Leave-one-group-out: hold out a whole dataset or a whole encoder."""
    preds = np.full(len(C), np.nan)
    for g in sorted({c[group_key] for c in C}):
        te = [i for i, c in enumerate(C) if c[group_key] == g]
        tr = [i for i, c in enumerate(C) if c[group_key] != g]
        if len(tr) < 6:
            continue
        X = np.array([C[j][key] for j in tr], float)
        model = ridge_fit(X, y[tr])
        m = 0.0
        if margin_q is not None:
            m = float(np.quantile(y[tr] - ridge_pred(model, X), margin_q))
        preds[te] = ridge_pred(model, np.array([C[j][key] for j in te], float)) + m
    return preds


def learning_curve(C, y, rng):
    rows = []
    for n_train in (5, 8, 11, 14, 16):
        errs = []
        for _ in range(30):
            perm = rng.permutation(len(C))
            tr, te = perm[:n_train], perm[n_train:]
            X = np.array([C[j]['feat'] for j in tr], float)
            model = ridge_fit(X, y[tr])
            p = ridge_pred(model, np.array([C[j]['feat'] for j in te], float))
            errs.append(np.median(np.abs(np.log2(np.clip(p, 1, None) /
                                                np.clip(y[te], 1, None)))))
        rows.append({'n_train': n_train, 'median_log2_err': float(np.median(errs)),
                     'median_factor': float(2 ** np.median(errs))})
    xs = np.array([r['n_train'] for r in rows], float)
    ys = np.array([r['median_log2_err'] for r in rows], float)
    slope = float(np.polyfit(xs, ys, 1)[0]) if len(xs) > 1 else 0.0
    return {'rows': rows, 'slope_log2err_per_cell': slope}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='learned.json')
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    th = load_theory()
    C = cells(th)
    y = np.log(np.array([c['r_true'] for c in C], float))

    pred = loocv(C, y)
    pred_m = loocv(C, y, margin_q=0.90)     # nested conservative margin
    pred_ds = group_cv(C, y, 'ds')          # leave-one-dataset-out
    pred_enc = group_cv(C, y, 'enc')        # leave-one-encoder-out
    Xall = np.array([c['feat'] for c in C], float)
    model = ridge_fit(Xall, y)
    pred_in = ridge_pred(model, Xall)
    sse = float(((y - pred) ** 2).sum())
    sst = float(((y - y.mean()) ** 2).sum())
    r2_loo = 1.0 - sse / sst if sst > 0 else 0.0
    r2_in = 1.0 - float(((y - pred_in) ** 2).sum()) / sst if sst > 0 else 0.0

    # metrics on the rank scale
    true_r = np.exp(y)

    def meets_of(p):
        out = []
        for c, pr in zip(C, p):
            if not np.isfinite(pr):
                continue
            j = int(np.clip(np.searchsorted(c['grid'], pr), 0, len(c['grid']) - 1))
            out.append(1 if c['delta'][j] <= EPS else 0)
        return float(np.mean(out)) if out else float('nan')

    stats = err_stats(np.exp(pred), true_r)
    stats_m = err_stats(np.exp(pred_m), true_r)
    stats_ds = err_stats(np.exp(pred_ds), true_r)
    stats_enc = err_stats(np.exp(pred_enc), true_r)

    # ablation: drop the two label-dependent features (rho15, std eta) so the
    # predictor sees only label-free spectral/meta features.
    pred_s = loocv(C, y, key='feat_spec')
    pred_sm = loocv(C, y, margin_q=0.90, key='feat_spec')
    pred_sds = group_cv(C, y, 'ds', key='feat_spec')
    stats_s = err_stats(np.exp(pred_s), true_r)
    stats_sm = err_stats(np.exp(pred_sm), true_r)
    stats_sds = err_stats(np.exp(pred_sds), true_r)

    # naive baseline: training-fold median rank
    naive = np.zeros(len(C))
    for i in range(len(C)):
        idx = [j for j in range(len(C)) if j != i]
        naive[i] = np.exp(np.median(y[idx]))
    naive_stats = err_stats(naive, true_r)

    # on-site calibration arm, from the frozen acceptance run
    cal = {'source': 'acceptance_frozen.json', 'k': 3}
    try:
        acc = json.load(open(os.path.join(OUT, 'acceptance_frozen.json')))['result']['map10|k3']
        ratios = np.array(acc['ratios'], float)
        cal.update({'median_factor': float(2 ** np.median(np.abs(np.log2(ratios)))),
                    'within_2x': float(np.mean(np.abs(np.log2(ratios)) <= 1.0)),
                    'coverage': float(acc['coverage']),
                    'median_overhead': float(acc['median_overhead']),
                    'points_per_cell': 3})
    except Exception as exc:  # noqa: BLE001
        cal['error'] = repr(exc)

    # margin sweep: coverage is a design choice, so compare the two arms on
    # their coverage-overhead frontier rather than at a single margin.
    sweep = []
    for q in (0.50, 0.70, 0.80, 0.85, 0.90, 0.95, 0.99):
        pm = loocv(C, y, margin_q=q, key='feat_spec')
        st = err_stats(np.exp(pm), true_r)
        sweep.append({'margin_q': q, 'coverage': meets_of(np.exp(pm)),
                      'median_factor': st['median_factor']})

    lc = learning_curve(C, y, rng)
    res = {'n_cells': len(C), 'alpha': ALPHA, 'features': 8,
           'armA_zero_shot': {'r2_loo': r2_loo, 'r2_in_sample': r2_in, **stats,
                              'meets_eps': meets_of(np.exp(pred))},
           'armA_margin90': {**stats_m, 'meets_eps': meets_of(np.exp(pred_m)),
                             'note': 'margin estimated inside each training fold'},
           'armA_lodo_dataset': {**stats_ds, 'meets_eps': meets_of(np.exp(pred_ds))},
           'armA_loo_encoder': {**stats_enc, 'meets_eps': meets_of(np.exp(pred_enc))},
           'armA_spec_only': {**stats_s, 'meets_eps': meets_of(np.exp(pred_s)),
                              'note': 'label-free spectral/meta features only'},
           'armA_spec_only_margin90': {**stats_sm, 'meets_eps': meets_of(np.exp(pred_sm))},
           'armA_spec_only_lodo': {**stats_sds, 'meets_eps': meets_of(np.exp(pred_sds))},
           'armB_naive_median': {**naive_stats, 'meets_eps': None},
           'armC_calibration': cal,
           'learning_curve': lc, 'margin_sweep_spec_only': sweep,
           'cost': {'predictor_training_evals': len(C) * 23,
                    'calibration_evals_per_cell': 3,
                    'note': 'predictor needs a full rank grid per training cell; '
                            'calibration needs 3 points on the target cell only'},
           'cells': [{'id': c['id'], 'r_true': c['r_true'], 'pred': float(p)}
                     for c, p in zip(C, np.exp(pred))]}
    json.dump(res, open(os.path.join(OUT, args.out), 'w'), indent=1)
    print('A point: R2_loo=%.3f median=%.2fx within2x=%.0f%% meets=%.0f%%'
          % (r2_loo, stats['median_factor'], 100 * stats['within_2x'],
             100 * meets_of(np.exp(pred))), flush=True)
    print('A margin90: median=%.2fx within2x=%.0f%% meets=%.0f%%'
          % (stats_m['median_factor'], 100 * stats_m['within_2x'],
             100 * meets_of(np.exp(pred_m))), flush=True)
    print('A LODO-dataset: n=%d median=%.2fx within2x=%.0f%% meets=%.0f%%'
          % (stats_ds['n'], stats_ds['median_factor'], 100 * stats_ds['within_2x'],
             100 * meets_of(np.exp(pred_ds))), flush=True)
    print('A LOO-encoder: n=%d median=%.2fx within2x=%.0f%% meets=%.0f%%'
          % (stats_enc['n'], stats_enc['median_factor'], 100 * stats_enc['within_2x'],
             100 * meets_of(np.exp(pred_enc))), flush=True)
    print('A spec-only: median=%.2fx within2x=%.0f%% meets=%.0f%%'
          % (stats_s['median_factor'], 100 * stats_s['within_2x'],
             100 * meets_of(np.exp(pred_s))), flush=True)
    print('A spec-only+margin90: median=%.2fx meets=%.0f%% | LODO spec-only: median=%.2fx meets=%.0f%%'
          % (stats_sm['median_factor'], 100 * meets_of(np.exp(pred_sm)),
             stats_sds['median_factor'], 100 * meets_of(np.exp(pred_sds))), flush=True)
    print('B naive: median=%.2fx within2x=%.0f%%'
          % (naive_stats['median_factor'], 100 * naive_stats['within_2x']), flush=True)
    print('C calibrate: median=%.2fx within2x=%.0f%% coverage=%.0f%%'
          % (cal.get('median_factor', -1), 100 * cal.get('within_2x', 0),
             100 * cal.get('coverage', 0)), flush=True)
    print('margin sweep (spec-only): %s'
          % [(r['margin_q'], round(100 * r['coverage']), round(r['median_factor'], 2))
             for r in sweep], flush=True)
    print('LC slope=%.4f per cell; %s' % (lc['slope_log2err_per_cell'],
          [(r['n_train'], round(r['median_factor'], 2)) for r in lc['rows']]), flush=True)
    print('wrote', os.path.join(OUT, args.out), flush=True)


if __name__ == '__main__':
    main()
