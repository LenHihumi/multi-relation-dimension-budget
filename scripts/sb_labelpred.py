"""The predictor we never tried: label-aware, still zero evaluations.

The paper's predictor is deliberately label-free, so that it applies when no
group key exists.  A reviewer is entitled to say that this benchmarks the
hardest case and then calls the result "prediction does not work".  This script
runs the arm that argument asks for: features computed from the task-energy
curve -- the labels a deployment already has -- with the same ridge, the same
target (log r*), the same leave-one-corpus-out split and the same scoring call
(eval_rank on the measured grid) as the paper's 68% / 1.40x arm, so the numbers
are the same quantity rather than a look-alike.

Three things are reported:

  arms       label-free vs label-aware, with and without the reference margin
  misses     for every cell a rule misses, whether the issued rank was below the
             measured crossing (under-provision: the dangerous direction) or
             above it (grid snapping)
  subsets    all 1023 non-empty subsets of the ten label-aware features, so the
             quoted number can be shown not to be a selection artifact

Usage: python3 sb_labelpred.py --out labelpred.json
"""
from __future__ import annotations

import argparse
import glob
import itertools
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
from sb_lodo_true import eval_rank, load_cells, summarise              # noqa: E402

EPS = 0.05

LA_NAMES = ['log d', 'log n_groups', 'log group size', 'log PR(omega)',
            'entropy(omega)', 'disc mass top', 'log rank@rho=.1',
            'std(eta^2)', 'spearman(lambda, omega)', 'partition type']


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else 0.0


def load_theory():
    th = {}
    for p in glob.glob(os.path.join(OUT, 'theory__*.json')):
        d = json.load(open(p))
        th[(d['encoder'], d['dataset'])] = d
    return th


def label_features(th, enc, ds, rel):
    t = th[(enc, ds)]
    lam = np.array(t['lambda'], float)
    task = t['tasks'][rel]
    eta2 = np.array(task['eta2'], float)
    om = lam * eta2
    p = om / om.sum()
    ent = float(-(p * np.log(np.clip(p, 1e-12, None))).sum() / np.log(len(p)))
    pr = float(om.sum() ** 2 / (om ** 2).sum())
    r10 = float(task.get('rank', {}).get('0.1', np.nan))
    return [
        float(np.log(t['dim'])),
        float(np.log(max(task.get('n_groups', 1), 1))),
        float(np.log(max(task.get('group_size_median', 1.0), 1.0))),
        float(np.log(pr)),
        ent,
        float(task.get('disc_mass_top', np.nan)),
        float(np.log(max(r10, 1.0))) if np.isfinite(r10) else float(np.log(t['dim'])),
        float(np.std(eta2)),
        spearman(lam, om),
        1.0 if rel == 'inst' else 0.0,
    ]


def loco(C, key, margin_q=None):
    """Leave-one-corpus-out predictions of log r*, scored with the paper's eval_rank."""
    rows = []
    for g in sorted({c['ds'] for c in C}):
        tr = [c for c in C if c['ds'] != g]
        te = [c for c in C if c['ds'] == g]
        X = np.array([c[key] for c in tr], float)
        yy = np.array([np.log(max(c['r_true'], 1)) for c in tr])
        model = ridge_fit(X, yy)
        m = 0.0
        if margin_q is not None:
            m = float(np.quantile(yy - ridge_pred(model, X), margin_q))
        for c in te:
            pred = float(np.exp(ridge_pred(model, np.array([c[key]], float))[0] + m))
            meet, ov = eval_rank(c, pred)
            rows.append({'id': c['id'], 'pred': pred, 'meet': meet, 'overhead': ov,
                         'r_true': c['r_true'],
                         'direction': 'under' if pred < c['r_true'] else 'over'})
    return rows


def missed(rows):
    miss = [r for r in rows if not r['meet']]
    und = [r for r in miss if r['direction'] == 'under']
    return {'n_miss': len(miss), 'under': len(und),
            'over': len(miss) - len(und),
            'id_under': [r['id'] for r in und],
            'depth_under_median': (float(np.median([r['r_true'] / max(r['pred'], 1)
                                                    for r in und])) if und else None)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='labelpred.json')
    ap.add_argument('--no-subsets', dest='subsets', action='store_false')
    args = ap.parse_args()

    th = load_theory()
    C = load_cells()
    for c in C:
        c['feat_la'] = label_features(th, c['enc'], c['ds'], c['rel'])
        c['feat_both'] = list(c['feat_la']) + list(c['feat'])

    res = {'n_cells': len(C), 'target': 'log r*', 'split': 'leave-one-corpus-out',
           'scoring': 'eval_rank on the measured 23-rank grid (as in Sec. 6.1)',
           'features_label_aware': LA_NAMES, 'arms': {}}

    def run(tag, key, margin_q):
        rows = loco(C, key, margin_q)
        s = summarise(rows)
        res['arms'][tag] = {'coverage': s['coverage'],
                            'median_overhead': s['median_overhead'],
                            'p95_overhead': s['p95_overhead'],
                            'max_overhead': s['max_overhead'],
                            'within_2x': s['within_2x'],
                            'misses': missed(rows)}
        m = res['arms'][tag]['misses']
        print('%-44s cov=%5.1f%% med=%.2fx p95=%.2fx | miss %d (%d under, %d over)'
              % (tag, 100 * s['coverage'], s['median_overhead'], s['p95_overhead'],
                 m['n_miss'], m['under'], m['over']))

    run('label-free, no margin (paper arm)', 'feat', None)
    run('label-aware, no margin', 'feat_la', None)
    run('label-free + margin 0.90', 'feat', 0.90)
    run('label-aware + margin 0.90', 'feat_la', 0.90)
    run('both feature sets, no margin', 'feat_both', None)

    # ---- is the quoted number a selection artifact? all non-empty subsets ----
    if args.subsets:
        rows = []
        for k in range(1, len(LA_NAMES) + 1):
            for idx in itertools.combinations(range(len(LA_NAMES)), k):
                for c in C:
                    c['feat_sub'] = [c['feat_la'][i] for i in idx]
                s = summarise(loco(C, 'feat_sub', None))
                rows.append({'k': k, 'coverage': s['coverage'],
                             'median_overhead': s['median_overhead']})
        cov = np.array([r['coverage'] for r in rows])
        med = np.array([r['median_overhead'] for r in rows])
        full = next(r for r in rows if r['k'] == len(LA_NAMES))
        res['subsets'] = {
            'n': len(rows),
            'coverage_median': float(np.median(cov)),
            'coverage_best': float(cov.max()),
            'coverage_worst': float(cov.min()),
            'median_overhead_median': float(np.median(med)),
            'median_overhead_best': float(med.min()),
            'full_set_coverage': full['coverage'],
            'full_set_coverage_percentile': float((cov <= full['coverage']).mean()),
            'share_beating_full_coverage': float(np.mean(cov > full['coverage']))}
        print('subsets: %d; coverage median %.0f%% (best %.0f%%, worst %.0f%%); '
              'full set %.0f%% at percentile %.0f'
              % (len(rows), 100 * res['subsets']['coverage_median'],
                 100 * cov.max(), 100 * cov.min(), 100 * full['coverage'],
                 100 * res['subsets']['full_set_coverage_percentile']))

    json.dump(res, open(os.path.join(OUT, args.out), 'w'), indent=1)
    print('\nwrote', args.out)


if __name__ == '__main__':
    main()
