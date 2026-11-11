"""Is the failure of offline budget prediction specific to *label-free* predictors?

The manuscript shows that a spectral law transferred across corpora misses its coverage target, and that a
zero-shot ridge on spectral features does no better.  A reviewer can answer that in one line: "you only
tried unsupervised features".  This script closes that gap with the missing arm -- a *supervised* regressor
that is allowed the label-derived statistics the method itself uses.

Arms, all evaluated leave-one-corpus-out over {inshop, instre, sop}:

  spec-sl   ridge on spectral features only (label-free)
  spec-lab  ridge on spectral + label-derived features (task-energy participation ratio, gamma entropy,
            discriminative mass, group count, median group size)
  median    predict the training-fold median budget (the trivial arm)
  ours 2pt  two truncation probes on the target cell, no training set at all (from lodo_true29.json)

Metric, identical to the manuscript's cross-corpus arm: a prediction *meets* the target when it is at least
the measured budget; `overhead` is predicted/measured.  Coverage is the fraction of held-out cells that
meet, and the reported error is the median overhead over cells that meet plus the median miss factor over
cells that do not.

The point of the arm is symmetric evidence: if a supervised regressor with more features fails in the same
way the unsupervised one does, then the failure is a property of offline prediction under corpus shift, not
of the feature set.

Usage (from artifact/):
  python3 scripts/sb_supervised_reg.py
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, 'data')
ALPHAS = [0.1, 1.0, 10.0, 100.0]


def ridge(Xtr, ytr, Xte, alpha):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    A = (Xtr - mu) / sd
    B = (Xte - mu) / sd
    A1 = np.hstack([A, np.ones((len(A), 1))])
    B1 = np.hstack([B, np.ones((len(B), 1))])
    w = np.linalg.solve(A1.T @ A1 + alpha * np.eye(A1.shape[1]), A1.T @ ytr)
    return B1 @ w


def spectral_features(th):
    lam = np.array([x for x in th['lambda'] if x > 0], float)
    d = len(lam)
    p = lam / lam.sum()
    f = [np.log(d), d / 1000.0]
    cum = np.cumsum(p)
    for k in (8, 16, 32, 64, 128, 256):
        if k <= d:
            f.append(float(cum[k - 1]))
    f.append(float(lam.sum() ** 2 / (lam ** 2).sum()) / d)          # participation ratio / d
    f.append(float(np.exp(-(p * np.log(p)).sum())) / d)             # spectral entropy / d
    f.append(float(np.log(lam[0] / max(np.median(lam), 1e-30))))
    return f


def label_features(th, rel):
    t = th['tasks'][rel]
    eta = np.array(t['eta2'], float)
    lam = np.array([x for x in th['lambda'] if x > 0], float)
    om = lam * eta
    p = om / om.sum()
    f = [float(np.log(max(t['n_groups'], 1))), float(np.log(max(t['group_size_median'], 1.0)))]
    f.append(float(om.sum() ** 2 / (om ** 2).sum()) / len(lam))     # task-energy participation ratio / d
    f.append(float(np.exp(-(p * np.log(np.maximum(p, 1e-30))).sum())) / len(lam))
    f.append(float(t.get('disc_mass_top', 0.0)))
    f.append(float(eta.mean()))
    return f


def load_cells(data):
    th = {}
    for p in glob.glob(os.path.join(data, '**', 'theory__*.json'), recursive=True):
        d = json.load(open(p))
        th[(d['encoder'], d['dataset'])] = d
    cells = []
    for p in sorted(glob.glob(os.path.join(data, 'grid', '*.json'))):
        g = json.load(open(p))
        t = th.get((g['encoder'], g['dataset']))
        if not t:
            continue
        for cname, cell in g['cells'].items():
            rel = cname.split('@')[0]
            if rel not in ('inst', 'cat') or rel not in t['tasks']:
                continue
            e = cell.get('cen', {}).get('k10') or {}
            rstar = e.get('rstar05')
            if not rstar:
                continue
            cells.append(dict(enc=g['encoder'], corpus=g['dataset'], rel=rel,
                              rstar=float(rstar), d=t['dim'],
                              xs=spectral_features(t), xl=label_features(t, rel)))
    return cells


def report(name, rows):
    meet = np.array([r['meet'] for r in rows], float)
    over = np.array([r['overhead'] for r in rows], float)
    hit = over[meet > 0]
    miss = over[meet == 0]
    out = {'n': len(rows), 'coverage': float(meet.mean()),
           'median_overhead': float(np.median(hit)) if len(hit) else None,
           'median_miss': float(np.median(miss)) if len(miss) else None}
    print('%-22s coverage %.3f  median overhead %s  median miss %s'
          % (name, out['coverage'],
             ('%.2fx' % out['median_overhead']) if out['median_overhead'] else '-',
             ('%.2fx' % out['median_miss']) if out['median_miss'] else '-'))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=DATA)
    ap.add_argument('--results', default=os.path.join(DATA, 'results'))
    ap.add_argument('--out', default=os.path.join(ROOT, 'tables'))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    cells = load_cells(a.data)
    corpora = sorted({c['corpus'] for c in cells})
    print('cells %d, corpora %s' % (len(cells), corpora))

    res = {'cells': len(cells), 'arms': {}}
    for arm, key in (('spec-sl', 'xs'), ('spec-lab', 'xl')):
        feats = np.array([c['xs'] + (c['xl'] if key == 'xl' else []) for c in cells], float)
        y = np.log(np.array([c['rstar'] for c in cells], float))
        rows = []
        for hold in corpora:
            tr = [i for i, c in enumerate(cells) if c['corpus'] != hold]
            te = [i for i, c in enumerate(cells) if c['corpus'] == hold]
            # alpha chosen on the training folds only, by leave-one-corpus-out inside the training set
            inner = {}
            for al in ALPHAS:
                errs = []
                for h2 in sorted({cells[i]['corpus'] for i in tr}):
                    itr = [i for i in tr if cells[i]['corpus'] != h2]
                    ite = [i for i in tr if cells[i]['corpus'] == h2]
                    if len(ite) < 3:
                        continue
                    pr = ridge(feats[itr], y[itr], feats[ite], al)
                    errs.append(np.mean(np.abs(pr - y[ite])))
                inner[al] = float(np.mean(errs)) if errs else np.inf
            al_best = min(inner, key=lambda k: inner[k])
            pred = np.exp(ridge(feats[tr], y[tr], feats[te], al_best))
            for i, p in zip(te, pred):
                p = float(min(max(p, 1.0), cells[i]['d']))
                rows.append({'id': '%s/%s/%s' % (cells[i]['enc'], cells[i]['corpus'], cells[i]['rel']),
                             'pred': p, 'need': cells[i]['rstar'],
                             'meet': int(p >= cells[i]['rstar']),
                             'overhead': p / cells[i]['rstar']})
            print('  %s hold %-6s alpha %-5s n_test %d' % (arm, hold, al_best, len(te)))
        res['arms'][arm] = {'summary': report(arm, rows), 'rows': rows}

    rows = []
    for hold in corpora:
        tr = [c['rstar'] for c in cells if c['corpus'] != hold]
        med = float(np.median(tr))
        for c in cells:
            if c['corpus'] != hold:
                continue
            rows.append({'id': '%s/%s/%s' % (c['enc'], c['corpus'], c['rel']), 'pred': med,
                         'need': c['rstar'], 'meet': int(med >= c['rstar']),
                         'overhead': med / c['rstar']})
    res['arms']['median'] = {'summary': report('median', rows), 'rows': rows}

    lodo = json.load(open(os.path.join(a.results, 'lodo_true29.json')))
    need = {'%s/%s/%s' % (c['enc'], c['corpus'], c['rel']): c['rstar'] for c in cells}
    for arm in ('free_2pt', 'spec_lodo_dataset'):
        rows = []
        for r in lodo.get(arm, {}).get('rows', []):
            n = need.get(r['id'])
            if not n:
                continue
            p = float(r['pred'] if 'pred' in r else r['rank'])
            rows.append({'id': r['id'], 'pred': p, 'need': n, 'meet': int(p >= n),
                         'overhead': p / n})
        res['arms'][arm] = {'summary': report(arm, rows), 'rows': rows}

    json.dump(res, open(os.path.join(a.out, 'supervised_reg.json'), 'w'), indent=1)
    print('wrote', os.path.join(a.out, 'supervised_reg.json'))


if __name__ == '__main__':
    main()
