"""Does the budget protocol need labels?  And is task energy doing the work?

Two questions a reviewer is entitled to ask, both answerable from data already on disk:

  Q1  A deployment without a fine-grained group key cannot compute mAP@10 against a relevance
      partition.  But the `grid` files carry a label-free quality metric -- `fid`, the top-10
      agreement with the full-width ranking -- on the same rank grid.  If a protocol driven by
      `fid` issues a rank that also satisfies the *label* tolerance, then the protocol needs no
      annotations at all, only the corpus and a full-width index.

  Q2  The protocol is stated in the task-energy coordinate rho_T, and rho_T needs labels.  If the
      same protocol works in the variance coordinate rho_V (label-free, from lambda alone) and in
      the raw rank coordinate, then what makes it work is *measuring*, not the coordinate -- and
      the paper must say so.

Four variants, one protocol body, identical branch logic:

    coordinate x metric        x = rho_T (labels)   x = rho_V (label-free)   x = log r (no x)
    metric = mAP@10 (labels)          A                       B                      C
    metric = 1 - fid (no labels)      -                       D                      -

  A  reproduces the deployed rule and is the control.
  B  isolates the coordinate.
  C  removes the coordinate entirely.
  D  removes labels from both the coordinate and the metric -- the deployable label-free rule.

Every variant is scored by the *label* metric at the rank it issues, so the comparison is not
circular, and the margin factor is calibrated leave-one-corpus-out, as in the paper.

Usage:
    python3 sb_nolabel.py --art <dir with grid/ metric_std_parts/ theory/> --out <json>
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

EPS = 0.05
RHO_TARGETS = (0.12, 0.04)   # probe coordinates, far probe first (larger rho = smaller r)
F_GRID = np.round(np.arange(0.55, 1.001, 0.01), 3)
CORPORA = ('inshop', 'instre', 'sop', 'abo')


# ---------------------------------------------------------------- data

def load_cells(art):
    grids = {}
    for p in glob.glob(os.path.join(art, 'grid', '*.json')):
        d = json.load(open(p))
        grids[(d['encoder'], d['dataset'])] = d
    metrics = {}
    for p in glob.glob(os.path.join(art, 'metric_std_parts', '*.json')):
        d = json.load(open(p))
        metrics[(d['encoder'], d['dataset'])] = d
    theory = {}
    for p in glob.glob(os.path.join(art, 'theory', '*.json')):
        d = json.load(open(p))
        theory[(d['encoder'], d['dataset'])] = d

    cells = []
    for key, g in sorted(grids.items()):
        if key not in metrics or key not in theory:
            continue
        m, th = metrics[key], theory[key]
        lam = np.asarray(th['lambda'], dtype=np.float64)
        cum = np.concatenate([[0.0], np.cumsum(lam)])
        tot = cum[-1]
        for rel in ('inst', 'cat'):
            if rel not in g['cells'] or rel not in m['cells']:
                continue
            gc, mc = g['cells'][rel], m['cells'][rel]
            if 'k10' not in gc.get('sub', {}):
                continue
            fid_ranks = np.asarray(g['grid'], dtype=int)
            fid = np.asarray(gc['sub']['k10']['fid_curve'], dtype=np.float64)
            met_ranks = np.asarray(m['grid'], dtype=int)
            map10 = np.asarray(mc['metrics']['map10']['curve'], dtype=np.float64)
            full = float(mc['full']['map10'])

            eta = np.asarray(th['tasks'][rel]['eta2'], dtype=np.float64)
            w = lam * eta
            cw = np.concatenate([[0.0], np.cumsum(w)])

            # ranks both files measured; the label-free and label curves must be read at the same r
            common = np.array(sorted(set(fid_ranks.tolist()) & set(met_ranks.tolist())), dtype=int)
            if len(common) < 6:
                continue
            fi = [int(np.where(fid_ranks == r)[0][0]) for r in common]
            mi = [int(np.where(met_ranks == r)[0][0]) for r in common]
            d_lab = 1.0 - map10[mi] / max(full, 1e-12)
            d_fid = 1.0 - fid[fi]
            gT = 1.0 - cw[common] / cw[-1]
            gV = 1.0 - cum[common] / tot
            cells.append({'enc': key[0], 'ds': key[1], 'rel': rel, 'ranks': common,
                          'd_lab': d_lab, 'd_fid': d_fid, 'rho_t': np.clip(gT, 1e-6, None),
                          'rho_v': np.clip(gV, 1e-6, None), 'full_map': full})
    return cells


# ---------------------------------------------------------------- surface

def fit_surface(x, y):
    """Delta = 1/(1+(rho0/x)^s) by grid search.  Returns (rho0, s, r2) or (nan, nan, -inf)."""
    ok = np.isfinite(y) & (y > 1e-4) & (y < 1 - 1e-4) & (x > 0)
    if ok.sum() < 4:
        return np.nan, np.nan, -np.inf
    x, y = x[ok], y[ok]
    lx, L = np.log(x), np.log(1.0 / y - 1.0)
    best = (-np.inf, np.nan, np.nan)
    for s in np.linspace(0.4, 6.0, 141):
        # least squares in logit space for log rho0 at fixed s
        a = L + s * lx
        lr0 = a.mean() / s
        pred = 1.0 / (1.0 + np.exp(s * (lr0 - lx)))     # = 1/(1+(rho0/x)^s): rises with x
        ss = 1.0 - ((pred - y) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12)
        if ss > best[0]:
            best = (ss, float(np.exp(lr0)), float(s))
    return best[1], best[2], best[0]


# ---------------------------------------------------------------- protocol

def issue_rank(cell, coord, delta, f):
    """Three-branch two-probe rule, exactly the body of the paper with x = `coord`.

    `f` is the margin: the issued crossing is moved to `f * x_cross`, i.e. slightly more
    conservative (smaller rho, so more dimensions).
    """
    ranks, x, y = cell['ranks'], cell[coord], delta
    if len(ranks) < 4:
        return None, 0
    probes, used = [], 0
    for t in RHO_TARGETS:
        i = int(np.argmin(np.abs(np.log(x) - np.log(t))))
        if i not in probes:
            probes.append(i)
    probes = sorted(probes, key=lambda i: -x[i])      # far probe first = smaller rank
    if len(probes) < 2:
        return None, 0
    i1, i2 = probes[0], probes[1]
    used = 2
    y1, y2 = y[i1], y[i2]
    if y1 <= EPS and y2 <= EPS:                       # both anchors under tolerance
        return int(ranks[i2]), used                   # smaller measured rank
    if y1 > EPS and y2 > EPS:                         # both over: measure upward
        i = i2
        while i + 1 < len(ranks) and y[i] > EPS:
            i += 1
            used += 1
        return int(ranks[i]), used
    # straddle: two-point log-logistic fit, then move to f * crossing
    L1, L2 = np.log(1.0 / max(y1, 1e-9) - 1.0), np.log(1.0 / max(y2, 1e-9) - 1.0)
    lx1, lx2 = np.log(x[i1]), np.log(x[i2])
    if abs(lx2 - lx1) < 1e-9:
        return int(ranks[i2]), used
    s = -(L2 - L1) / (lx2 - lx1)
    if not np.isfinite(s) or s <= 0:
        return int(ranks[i2]), used
    lr0 = lx1 + L1 / s
    lcross = lr0 - np.log(1.0 / EPS - 1.0) / s
    target = np.exp(lcross) * f                       # margin: smaller rho -> more dimensions
    # smallest rank whose rho is at or below the target
    idx = np.where(x <= target)[0]
    if len(idx) == 0:
        return int(ranks[-1]), used
    return int(ranks[idx[0]]), used


def label_delta_at(cell, r):
    i = int(np.argmin(np.abs(cell['ranks'] - r)))
    return float(cell['d_lab'][i])


def label_budget(cell):
    i = np.where(cell['d_lab'] <= EPS)[0]
    return int(cell['ranks'][i[0]]) if len(i) else int(cell['ranks'][-1])


def nolabel_budget(cell):
    i = np.where(cell['d_fid'] <= EPS)[0]
    return int(cell['ranks'][i[0]]) if len(i) else int(cell['ranks'][-1])


# ---------------------------------------------------------------- main

VARIANTS = {
    'A_rhoT_map10': ('rho_t', 'd_lab'),
    'B_rhoV_map10': ('rho_v', 'd_lab'),
    'C_rank_map10': (None, 'd_lab'),
    'D_rhoV_fid': ('rho_v', 'd_fid'),
    'E_rhoT_fid': ('rho_t', 'd_fid'),
}


def rank_coord(cell):
    return cell['ranks'].astype(np.float64)


def run(cells, art_out):
    # attach a pseudo-coordinate for variant C: the rank itself, scaled into (0,1]
    for c in cells:
        c['r'] = rank_coord(c) / float(c['ranks'].max())
        c['d_none'] = c['d_lab']

    res = {'eps': EPS, 'folds': {}, 'cells': {}}
    for vname, (coord, dkey) in VARIANTS.items():
        key = 'r' if coord is None else coord
        res['cells'][vname] = {}
        for c in cells:
            name = '%s__%s__%s' % (c['enc'], c['ds'], c['rel'])
            r_lab = label_budget(c)
            entry = {'r_label': r_lab}
            for f in (1.0,):
                r_hat, used = issue_rank(c, key, c[dkey], f)
                entry['rank@f1.0'] = r_hat
                entry['evals@f1.0'] = used
                entry['d_lab@f1.0'] = label_delta_at(c, r_hat) if r_hat else None
            res['cells'][vname][name] = entry

    # leave-one-corpus-out margin calibration, then scored on the held-out corpus
    for vname, (coord, dkey) in VARIANTS.items():
        key = 'r' if coord is None else coord
        for hold in CORPORA:
            train = [c for c in cells if c['ds'] != hold]
            test = [c for c in cells if c['ds'] == hold]
            if not test or not train:
                continue
            best = None
            for f in F_GRID:
                cov, ovh = [], []
                for c in train:
                    r_hat, _ = issue_rank(c, key, c[dkey], f)
                    ok = label_delta_at(c, r_hat) <= EPS
                    cov.append(ok)
                    if ok:
                        ovh.append(r_hat / label_budget(c))
                if not cov:
                    continue
                co = float(np.mean(cov))
                med = float(np.median(ovh)) if ovh else 9.9
                k = (0 if co >= 0.999 else 1, med)
                if best is None or k < best[0]:
                    best = (k, f, co, med)
            if best is None:
                continue
            _, f, co_tr, med_tr = best
            rows = []
            for c in test:
                r_hat, used = issue_rank(c, key, c[dkey], f)
                dl = label_delta_at(c, r_hat)
                rows.append({'cell': '%s__%s__%s' % (c['enc'], c['ds'], c['rel']),
                             'r_hat': r_hat, 'r_label': label_budget(c),
                             'd_lab': dl, 'ok': bool(dl <= EPS), 'evals': used})
            cov = float(np.mean([r['ok'] for r in rows]))
            ovh = [r['r_hat'] / r['r_label'] for r in rows if r['ok']]
            res['folds'].setdefault(vname, {})[hold] = {
                'f': f, 'train_cov': co_tr, 'train_med': med_tr,
                'test_cov': cov, 'test_med': float(np.median(ovh)) if ovh else None,
                'rows': rows}
    json.dump(res, open(art_out, 'w'), indent=1)
    return res


def report(cells, res):
    print('\n=== label-free vs label budget (5%% tolerance), %d cells' % len(cells))
    print('%-30s %6s %6s %6s %7s' % ('cell', 'r_lab', 'r_nolab', 'ratio', 'd_lab@nl'))
    rat, dnl = [], []
    for c in cells:
        rl, rn = label_budget(c), nolabel_budget(c)
        dl = label_delta_at(c, rn)
        rat.append(rn / rl)
        dnl.append(dl)
        print('%-30s %6d %6d %6.2f %7.3f%s'
              % ('%s__%s__%s' % (c['enc'][:14], c['ds'], c['rel']), rl, rn, rn / rl, dl,
                 '' if dl <= EPS else '  <-- miss'))
    print('median ratio %.2f   misses %d/%d' % (float(np.median(rat)), sum(d > EPS for d in dnl),
                                                len(dnl)))
    # the decision variable is bytes, and the link list does not shrink with the rank
    by = [((4 * nolabel_budget(c) + 256) / (4 * label_budget(c) + 256), c) for c in cells]
    bb = sorted(b for b, _ in by)
    print('bytes ratio (4r+256, M=32): median %.2f  p10 %.2f  max %.2f'
          % (float(np.median(bb)), bb[max(len(bb) // 10 - 1, 0)], bb[-1]))
    print('  worst three: ' + ', '.join('%s %.2fx' % ('%s/%s' % (c['enc'][:6], c['ds']), b)
                                        for b, c in sorted(by, key=lambda t: -t[0])[:3]))

    print('\n=== protocol, leave-one-corpus-out margin, scored by the label metric')
    print('%-16s %-6s %5s %6s %6s %6s' % ('variant', 'hold', 'f', 'cov', 'med', 'eval'))
    for v, folds in res['folds'].items():
        for hold, d in folds.items():
            ev = np.mean([r['evals'] for r in d['rows']])
            print('%-16s %-6s %5.2f %6.2f %6s %6.1f'
                  % (v, hold, d['f'], d['test_cov'],
                     '%.2f' % d['test_med'] if d['test_med'] else '-', ev))
    print('\n%-16s %8s %8s' % ('variant', 'pooled', 'med ovh'))
    for v, folds in res['folds'].items():
        all_rows = [r for d in folds.values() for r in d['rows']]
        if not all_rows:
            continue
        cov = np.mean([r['ok'] for r in all_rows])
        ovh = [r['r_hat'] / r['r_label'] for r in all_rows if r['ok']]
        print('%-16s %8.2f %8s' % (v, cov, '%.2f' % np.median(ovh) if ovh else '-'))

    print('\n=== surface fit quality by coordinate (median R^2 over cells)')
    for coord, delta in (('rho_t', 'd_lab'), ('rho_v', 'd_lab'), ('rho_v', 'd_fid')):
        r2 = []
        for c in cells:
            _, _, v = fit_surface(c[coord], c[delta])
            if np.isfinite(v):
                r2.append(v)
        print('%-8s x %-8s n=%2d  median R^2 %.3f  min %.3f'
              % (coord, delta, len(r2), float(np.median(r2)), float(np.min(r2))))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--art', default='.')
    ap.add_argument('--out', default='nolabel.json')
    a = ap.parse_args()
    cells = load_cells(a.art)
    print('cells loaded: %d' % len(cells))
    for c in cells:
        print('  %-22s %-6s %-4s ranks %d-%d  n=%d  full_map %.3f'
              % (c['enc'][:20], c['ds'], c['rel'], c['ranks'][0], c['ranks'][-1],
                 len(c['ranks']), c['full_map']))
    res = run(cells, a.out)
    report(cells, res)
