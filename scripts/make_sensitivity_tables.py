"""Fold the sensitivity runs into two tables a reader can actually use.

Two questions the review asked for and the paper answers only in prose:

  1. How far does the budget move when the stored vectors are quantised harder than
     8 bits? `data/sensitivity/quant_b*.json` holds one run per bit width, each with a
     per-cell record of the rank before and after.
  2. What happens to the budget when the partition is coarser, finer, or noisier? The
     `gs_*.json` / `gsl_*.json` runs sweep random partitions at several group counts and
     relabel a fraction of the rows.

Both are reductions of released JSON; nothing is re-measured here.

Outputs: tables/quant_bits.csv, tables/group_sensitivity.csv
Run:     python3 scripts/make_sensitivity_tables.py
"""
from __future__ import annotations

import csv
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
SENS = os.path.join(ROOT, 'data', 'sensitivity')
TAB = os.path.join(ROOT, 'tables')


def quant():
    rows = []
    for f in sorted(glob.glob(os.path.join(SENS, 'quant_b*.json'))) + \
            sorted(glob.glob(os.path.join(SENS, 'quant_budget.json'))):
        d = json.load(open(f, encoding='utf-8'))
        bits = d.get('bits')
        for c in d.get('cells', []):
            before = c.get('rstar_f32')
            after = c.get('rstar_i8')
            if before is None or after is None:
                continue
            rows.append({'bits': bits, 'encoder': c.get('enc'), 'dataset': c.get('ds'),
                         'relation': c.get('rel'),
                         'rank_fp32': int(before), 'rank_quantised': int(after),
                         'shift': round(after / float(before), 3),
                         'rho0_fp32': round(c.get('rho0_f32', 0), 4),
                         'rho0_quantised': round(c.get('rho0_i8', 0), 4),
                         'steepness_fp32': round(c.get('s_f32', 0), 3),
                         'steepness_quantised': round(c.get('s_i8', 0), 3)})
    if not rows:
        return []
    # one row per (bits, cell); the 8-bit run can appear twice if both files exist
    seen, uniq = set(), []
    for r in sorted(rows, key=lambda x: (x['bits'], x['encoder'], x['dataset'], x['relation'])):
        k = (r['bits'], r['encoder'], r['dataset'], r['relation'])
        if k in seen:
            continue
        seen.add(k)
        uniq.append(r)
    with open(os.path.join(TAB, 'quant_bits.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(uniq[0].keys()))
        w.writeheader()
        w.writerows(uniq)
    # report the run's own definition of "within one grid step", which counts grid
    # positions rather than an arbitrary ratio band
    sums = {}
    for f in sorted(glob.glob(os.path.join(SENS, 'quant_b*.json'))) + \
            sorted(glob.glob(os.path.join(SENS, 'quant_budget.json'))):
        d = json.load(open(f, encoding='utf-8'))
        sums[d.get('bits')] = d.get('summary', {})
    for b in sorted(sums):
        sm = sums[b]
        print('%-2d-bit: %s cells, median shift %.2f, %.0f%% within one grid step, '
              'range %.2f-%.2f'
              % (b, sm.get('n'), sm.get('median_shift', float('nan')),
                 100 * sm.get('frac_within_one_step', float('nan')),
                 sm.get('min_shift', float('nan')), sm.get('max_shift', float('nan'))))
    return uniq


def groups():
    rows = []
    for f in sorted(glob.glob(os.path.join(SENS, 'gsl_*.json'))) + \
            sorted(glob.glob(os.path.join(SENS, 'gs_*.json'))):
        if os.path.basename(f).startswith('gs_smoke'):
            continue
        d = json.load(open(f, encoding='utf-8'))
        base = '%s/%s/%s' % (d['enc'], d['ds'], d['rel'])
        for v in d.get('variants', []):
            rows.append({'cell': base, 'order': d.get('order', 'omega'),
                         'variant': v['variant'], 'n_groups': v['n_groups'],
                         'group_size_median': v['group_size_median'],
                         'rank_5pct': v['r_star'], 'eta2_mean': round(v['eta2_mean'], 4)})
    if not rows:
        return []
    # keep the lambda-order sweep (the paper's budgets are PCA prefixes); the omega run is
    # the label-weighted variant and stays in data/sensitivity/ for anyone who wants it
    lam = {r['cell'] for r in rows if r['order'] == 'lambda'}
    rows = [r for r in rows if (r['order'] == 'lambda') or (r['cell'] not in lam)]
    with open(os.path.join(TAB, 'group_sensitivity.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    for cell in sorted({r['cell'] for r in rows}):
        sub = sorted([r for r in rows if r['cell'] == cell], key=lambda r: r['n_groups'])
        real = [r for r in sub if r['variant'] == 'real']
        print('%-32s order=%-6s real %d groups -> r*=%s | sweep %s'
              % (cell, sub[0]['order'],
                 real[0]['n_groups'] if real else -1,
                 real[0]['rank_5pct'] if real else '?',
                 ' '.join('%d:%s' % (r['n_groups'], r['rank_5pct'])
                          for r in sub if r['variant'].startswith('random'))))
    return rows


if __name__ == '__main__':
    print('== quantisation')
    quant()
    print('\n== partition shape')
    groups()
