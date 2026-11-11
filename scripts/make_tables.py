"""Regenerates every table in the paper from the released JSON.

Outputs (in the same directory as the JSON, under artifact/tables/):
  recall38_cells.csv   38 cells x 21 ranks of recall@10, with r* and its CI
  map10_19cells.csv    19 cells x 23 ranks of mAP@10, with the deleted energy rho
  fits_19cells.csv     per-cell fitted (rho0, s), R2, measured r* for each metric
  rules_table.csv      the budget-rule comparison of Table 6
  lodo_arms.csv        every arm of the cross-corpus experiment

Run:  python3 make_tables.py            (from artifact/)
"""
import csv
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, 'data')
OUT = os.path.join(ROOT, 'tables')
EPS = 0.05


def load(p):
    return json.load(open(p))


def rstar(grid, curve, eps=EPS):
    grid = np.asarray(grid, float)
    curve = np.asarray(curve, float)
    ok = np.where(curve >= 1.0 - eps)[0]
    return (float(grid[ok[0]]) if len(ok) else float(grid[-1]))


def recall_table():
    rows = []
    for p in sorted(glob.glob(os.path.join(DATA, 'grid', '*.json'))):
        d = load(p)
        grid = d['grid']
        for rel, cell in d['cells'].items():
            if rel not in ('inst', 'cat'):
                continue
            for k in ('k1', 'k10'):
                e = cell.get('cen', {}).get(k)
                if not e:
                    continue
                curve = e['rec_curve'][:len(grid)]
                for r, v in zip(grid, curve):
                    rows.append([d['encoder'], d['dataset'], rel, k[1:],
                                 d['dim'], int(r), round(float(v), 5),
                                 round(float(v) / float(curve[-1]), 5)])
        # per-cell summary appended to a separate file by fits_table()
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'recall38_cells.csv'), 'w', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['encoder', 'dataset', 'relation', 'k', 'dim', 'rank',
                    'recall_at_k', 'retention'])
        w.writerows(rows)
    print('recall38_cells.csv rows', len(rows))


def map10_table():
    rows = []
    for p in sorted(glob.glob(os.path.join(DATA, 'metric_std_parts', '*.json'))):
        d = load(p)
        for rel, cell in d.get('cells', {}).items():
            mm = cell.get('metrics', {}).get('map10')
            if not mm:
                continue
            full = float(cell['full']['map10'])
            n = min(len(mm['curve']), len(cell['rho']), len(d['grid']))
            for i in range(n):
                rows.append([d['encoder'], d['dataset'], rel, int(d['grid'][i]),
                             round(float(cell['rho'][i]), 8),
                             round(float(mm['curve'][i]), 6),
                             round(1.0 - float(mm['curve'][i]) / full, 6)])
    with open(os.path.join(OUT, 'map10_19cells.csv'), 'w', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['encoder', 'dataset', 'relation', 'rank', 'rho_T',
                    'map10', 'delta'])
        w.writerows(rows)
    print('map10_19cells.csv rows', len(rows))


def fits_table():
    """Table 2.  Fits must come from the mAP@10 curves, not from the recall grid.

    sigmoid.json holds the fits on the 38-cell recall grid; using it here would
    silently mix protocols and the per-corpus medians would not match Table 2.
    """
    ms = load(os.path.join(DATA, 'results', 'metric_std.json'))['per_metric']['map10']['cells']
    idx = {(c['enc'], c['ds'], c['rel']): c for c in ms}
    rows = []
    for p in sorted(glob.glob(os.path.join(DATA, 'metric_std_parts', '*.json'))):
        d = load(p)
        for rel, cell in d.get('cells', {}).items():
            m = cell.get('metrics', {})
            full = float(cell['full']['map10'])
            n = min(len(m['map10']['curve']), len(d['grid']))
            rs = rstar(d['grid'][:n], np.array(m['map10']['curve'][:n]) / full)
            fit = idx.get((d['encoder'], d['dataset'], rel), {})
            rows.append([d['encoder'], d['dataset'], rel, d['dim'],
                         cell['n_query'], round(full, 6),
                         round(float(fit.get('rho0', 'nan')), 6),
                         round(float(fit.get('s', 'nan')), 4),
                         round(float(fit.get('r2', 'nan')), 4),
                         int(rs), int(fit.get('rstar_meas', rs)),
                         round(float(cell['full']['recall10']), 6),
                         round(float(cell['full']['ndcg10']), 6)])
    with open(os.path.join(OUT, 'fits_19cells.csv'), 'w', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['encoder', 'dataset', 'relation', 'dim', 'n_query',
                    'full_map10', 'rho0', 's', 'r2', 'rstar_from_curve',
                    'rstar_measured', 'full_recall10', 'full_ndcg10'])
        w.writerows(rows)
    print('fits_19cells.csv rows', len(rows))


def rules_table():
    """Table 6.  Source is metric_std.json, i.e. the mAP@10 protocol.

    rules_compare.json is the recall@10 version of the same comparison and its
    byte counts differ (896 vs 1280 for 90% variance).  Mixing the two is the
    exact error a reviewer would find by diffing the table against the artifact,
    so both are emitted, clearly labelled.
    """
    p = os.path.join(DATA, 'results', 'metric_std.json')
    if not os.path.exists(p):
        return
    rc = load(p)['rule_compare']
    for metric, tag in (('map10', 'rules_table.csv'), ('recall10', 'rules_table_recall10.csv')):
        rows = []
        for grp in ('all', 'inst', 'cat'):
            for name, v in rc[metric]['summary'][grp].items():
                rows.append([metric, grp, name, v['n'], round(float(v['hit_rate']), 4),
                             v.get('median_bytes'), v.get('median_bytes_at_target'),
                             round(float(v.get('median_shortfall_pts', 0.0)), 3),
                             round(float(v.get('median_ratio_to_oracle', 0.0)), 3)])
        with open(os.path.join(OUT, tag), 'w', newline='') as f:
            w = csv.writer(f, lineterminator='\n')
            w.writerow(['metric', 'partition', 'rule', 'n_cells', 'coverage',
                        'median_bytes', 'median_bytes_at_target',
                        'median_shortfall_points', 'median_rank_over_oracle'])
            w.writerows(rows)
        print(tag, 'rows', len(rows))


def lodo_table():
    p = os.path.join(DATA, 'results', 'lodo_true.json')
    if not os.path.exists(p):
        return
    d = load(p)
    rows = []
    for arm, v in d.items():
        if isinstance(v, dict) and 'summary' in v:
            s = v['summary']
            rows.append([arm, s['n'], round(s['coverage'], 4),
                         round(s['median_overhead'], 4),
                         round(s['within_1_25x'], 4), round(s['within_2x'], 4),
                         round(s['p95_overhead'], 3), round(s['max_overhead'], 3)])
    with open(os.path.join(OUT, 'lodo_arms.csv'), 'w', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['arm', 'n', 'coverage', 'median_overhead', 'within_1.25x',
                    'within_2x', 'p95_overhead', 'max_overhead'])
        w.writerows(rows)
    print('lodo_arms.csv rows', len(rows))


def placement_table():
    """Table 10: where the two calibration points sit (sensitivity of 2-point rule).

    Emitted from data/results/sensitivity_2pt.json, which ships every
    (pair, cell) evaluation of the two-point protocol.
    """
    p = os.path.join(ROOT, 'data', 'results', 'sensitivity_2pt.json')
    if not os.path.exists(p):
        return
    d = load(p)
    rows = []
    for r in d['spread_bins']['rows']:
        rows.append(['separation %.1f-%.1f nats' % (r['lo'], r['hi']), r['n'],
                     round(r['coverage'], 4), round(r['median_overhead'], 4)])
    for tag in ('straddle', 'nonstraddle'):
        q = d['pooled'][tag]
        if q['n']:
            rows.append([tag, q['n'], round(q['coverage'], 4),
                         round(q['median_overhead'], 4)])
    rows.append(['prescribed rho=(0.04,0.12)', d['prescribed']['n'],
                 round(d['prescribed']['coverage'], 4),
                 round(d['prescribed']['median_overhead'], 4)])
    with open(os.path.join(OUT, 'two_point_placement.csv'), 'w', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['stratum', 'n_evaluations', 'coverage', 'median_overhead'])
        w.writerows(rows)
    print('two_point_placement.csv rows', len(rows))


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    recall_table()
    map10_table()
    fits_table()
    rules_table()
    lodo_table()
    placement_table()
