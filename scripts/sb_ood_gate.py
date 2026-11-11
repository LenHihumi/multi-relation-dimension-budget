"""Does a label-free OOD screen mark the cells where the transferable predictor fails?

Pre-registered, fixed rule, no threshold tuning.  Three leave-one-corpus-out folds; for
each fold the reference range of every feature is the 5th-95th percentile *of the other
two corpora*, and a held-out cell is flagged when any feature falls outside its range.
Both versions of the screen are reported: a label-free one (task-energy effective rank,
normalised spectral entropy) and a label-aware one that adds the group count, because
group counts are not always available at deployment.

The label is re-derived under the paired bootstrap, not under the band approximation:
the predictor's issued rank is snapped to the nearest grid rank, and the cell counts as a
failure when the measured loss there exceeds eps.  `point` uses the bootstrap centre
(comparable to the coverage numbers); `decided` additionally requires the excess to clear
1.96 standard errors.

Pre-registered continuation rule: the screen has support only if at least two of the three
folds reach recall >= 0.5 with flag rate <= 0.5.  Otherwise no hierarchical prior is built.

Usage: python sb_ood_gate.py --out ood_gate.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from math import log, sqrt

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
D = os.path.join(ROOT, 'data')
R = os.path.join(D, 'results')

EPS = 0.05
N_BOOT = 2000
SEED = 0
Z = 1.96
GAMMA = 0.0

# group counts measured on the gallery labels (cat / ent fields of gallery.jsonl)
GROUPS = {('inshop', 'cat'): 23, ('inshop', 'inst'): 12995,
          ('instre', 'cat'): 21, ('instre', 'inst'): 200,
          ('sop', 'inst'): 22634}


def load_theory():
    th = {}
    for p in glob.glob(os.path.join(D, 'theory', 'theory__*.json')):
        d = json.load(open(p))
        th[(d['encoder'], d['dataset'])] = d
    return th


def features(th, enc, ds, rel, rank):
    """Task-energy effective rank at `rank`, and the spectral entropy of the top `rank`."""
    t = th[(enc, ds)]
    lam = np.asarray(t['lambda'], float)
    eta2 = np.asarray(t['tasks'][rel]['eta2'], float)
    om = (lam ** (1.0 - GAMMA)) * eta2
    r = max(1, min(int(rank), len(om)))
    o = om[:r]
    deff = float(o.sum() ** 2 / (o ** 2).sum()) if o.sum() > 0 else 0.0
    p = lam[:r] / lam[:r].sum()
    ent = float(-(p * np.log(np.maximum(p, 1e-300))).sum() / log(r)) if r > 1 else 0.0
    return deff, ent


def boot_at(cell_npz, rank):
    """Measured loss at the grid rank nearest to `rank`, plus its paired-bootstrap SE."""
    z = np.load(cell_npz)
    grid = z['grid']
    j = int(np.argmin(np.abs(grid - rank)))
    mr = z['perq'][j].astype(np.float64)
    mf = z['full'].astype(np.float64)
    delta = 1.0 - mr.mean() / mf.mean()
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(mr), size=(N_BOOT, len(mr)))
    dr = 1.0 - (mr[idx].mean(1) / mf[idx].mean(1))
    se = float(np.std(dr, ddof=1))
    return {'grid_rank': int(grid[j]), 'rank_err': float(abs(grid[j] - rank) / rank),
            'delta': float(delta), 'se': se,
            'z': float((delta - EPS) / se) if se > 0 else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='ood_gate.json')
    args = ap.parse_args()

    lodo = json.load(open(os.path.join(R, 'lodo_true.json')))['spec_lodo_dataset']['rows']
    th = load_theory()

    rows = []
    for r in lodo:
        enc, ds, rel = r['id'].split('/')
        pred = min(float(r['pred']), 1e9)
        npz = os.path.join(D, 'mesh', '%s__%s__%s.npz' % (enc, ds, rel))
        b = boot_at(npz, pred)
        deff, ent = features(th, enc, ds, rel, b['grid_rank'])
        rows.append({'id': r['id'], 'fold': ds, 'pred': pred, 'grid_rank': b['grid_rank'],
                     'rank_err': b['rank_err'], 'delta': b['delta'], 'se': b['se'], 'z': b['z'],
                     'fail_point': int(b['delta'] > EPS),
                     'fail_decided': int(b['z'] > Z),
                     'meet_band': int(r['meet']),
                     'deff': deff, 'entropy': ent,
                     'group_count': GROUPS.get((ds, rel), 0)})

    feats = {'label-free': ('deff', 'entropy'),
             'label-aware': ('deff', 'entropy', 'group_count')}
    folds = {}
    for ver, fs in feats.items():
        folds[ver] = {}
        for hold in ('inshop', 'instre', 'sop'):
            ref = [x for x in rows if x['fold'] != hold]
            held = [x for x in rows if x['fold'] == hold]
            lo = {f: float(np.percentile([x[f] for x in ref], 5)) for f in fs}
            hi = {f: float(np.percentile([x[f] for x in ref], 95)) for f in fs}
            for x in held:
                x['flag_' + ver] = int(any(x[f] < lo[f] or x[f] > hi[f] for f in fs))
            fp = sum(1 for x in held if x['flag_' + ver] and x['fail_point'])
            nf = sum(1 for x in held if x['fail_point'])
            fl = sum(1 for x in held if x['flag_' + ver])
            folds[ver][hold] = {'n': len(held), 'n_fail': nf, 'n_flag': fl,
                                'flag_rate': fl / len(held),
                                'recall': fp / nf if nf else None,
                                'precision': fp / fl if fl else None,
                                'tp': fp, 'fp': fl - fp, 'fn': nf - fp,
                                'tn': len(held) - fl - (nf - fp),
                                'bounds': {'lo': lo, 'hi': hi},
                                'flagged': [x['id'] for x in held if x['flag_' + ver]],
                                'failed_unflagged': [x['id'] for x in held
                                                     if x['fail_point'] and not x['flag_' + ver]]}

    verdicts = {}
    for ver in feats:
        ok = [h for h, m in folds[ver].items()
              if m['recall'] is not None and m['recall'] >= 0.5 and m['flag_rate'] <= 0.5]
        verdicts[ver] = {'folds_meeting_rule': ok, 'n_meeting': len(ok),
                         'supported': len(ok) >= 2}

    cov_boot = 1.0 - sum(x['fail_point'] for x in rows) / len(rows)
    cov_band = 1.0 - sum(1 - x['meet_band'] for x in rows) / len(rows)
    out = {'eps': EPS, 'n_cells': len(rows), 'rule': '5-95 pct of the other two corpora',
           'coverage_point_boot': cov_boot, 'coverage_band': cov_band,
           'max_rank_snap_error': max(x['rank_err'] for x in rows),
           'folds': folds, 'verdicts': verdicts, 'rows': rows}

    print('cells %d   coverage: bootstrap centre %.3f, band %.3f   max snap error %.1f%%'
          % (len(rows), cov_boot, cov_band, 100 * max(x['rank_err'] for x in rows)))
    print('bindings: %d of 19 changed label vs the band-derived meet'
          % sum(1 for x in rows if x['fail_point'] != 1 - x['meet_band']))
    for ver in feats:
        print('\n%s screen' % ver)
        print('%-8s %3s %5s %6s %7s %7s %9s' % ('fold', 'n', 'fail', 'flag', 'rate', 'recall', 'precision'))
        for h, m in folds[ver].items():
            print('%-8s %3d %5d %6d %6.2f %7s %9s'
                  % (h, m['n'], m['n_fail'], m['n_flag'], m['flag_rate'],
                     '%.2f' % m['recall'] if m['recall'] is not None else '-',
                     '%.2f' % m['precision'] if m['precision'] is not None else '-'))
        v = verdicts[ver]
        print('  folds meeting the rule: %s -> %s'
              % (v['folds_meeting_rule'], 'SUPPORTED' if v['supported'] else 'NOT SUPPORTED'))

    with open(os.path.join(R, args.out), 'w') as f:
        json.dump(out, f, indent=1)
    print('\nwrote %s' % os.path.join(R, args.out))


if __name__ == '__main__':
    main()
