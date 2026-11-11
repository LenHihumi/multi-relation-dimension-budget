"""Two tables the review asked for that the paper only summarises.

1. crosscell_pairs.csv -- all 11 (encoder, corpus) pairs at a 5% tolerance: the rank that
   retains 90% of the variance, the per-relation budgets, and the bytes of the three
   designs.  The paper prints the median ratio 1.7x and the span 1.0--5.3x; a reader who
   wants to know which pair is the 5.3x cannot get it from the text.

2. coverage_bytes.csv -- the coverage/byte trade-off of each budget-rule family across its
   own tolerance settings, from tables/rules_table.csv.  Table III compares families at one
   target; this exposes what each family does as it is tightened, which is the curve the
   review asked for.

Both are pure reductions of released JSON/CSV: nothing is re-measured.

Run from the artifact root:  python3 scripts/review_tables.py
"""
from __future__ import annotations

import csv
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
TAB = os.path.join(ROOT, 'tables')


def crosscell_pairs():
    d = json.load(open(os.path.join(ROOT, 'data', 'results', 'crosscell.json'),
                       encoding='utf-8'))
    rows = []
    for p in d['pairs']:
        rows.append([p['enc'], p['ds'], '/'.join(p['rels']), p['r90'],
                     p['rstar'].get('inst', ''), p['rstar'].get('cat', ''),
                     p['bytes_var90'], p['bytes_unified'], p['bytes_pair'],
                     round(p['ratio_var90_unified'], 3),
                     1 if p.get('var90_meets') else 0])
    rows.sort(key=lambda r: -r[9])
    with open(os.path.join(TAB, 'crosscell_pairs.csv'), 'w', newline='') as fh:
        w = csv.writer(fh, lineterminator='\n')
        w.writerow(['encoder', 'corpus', 'relations', 'r90', 'budget_item',
                    'budget_cat', 'bytes_var90', 'bytes_unified', 'bytes_pair',
                    'ratio_var90_over_unified', 'var90_meets_tolerance'])
        w.writerows(rows)
    s = d['summary']
    print('11 pairs: median %.2fx, min %.2f, max %.2f, %d of %d above the %.1fx threshold'
          % (s['median_ratio'], s['min_ratio'], s['max_ratio'], s['n_ge_thr'],
             s['n_pairs'], s['threshold']))
    for r in rows:
        print('  %-20s %-7s %-9s r90=%-5d inst=%-4s cat=%-4s var90=%-6d unified=%-6d %.2fx%s'
              % (r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[9],
                 '' if r[10] else '  (var90 misses)'))


def coverage_bytes():
    """Coverage against bytes for each rule family, from the released rules table."""
    fam = {'energy': 'energy retention', 'scree': 'scree / knee',
           'pr': 'participation ratio', 'ff': 'fixed fraction', 'law': 'response law'}
    rows = []
    with open(os.path.join(TAB, 'rules_table.csv'), encoding='utf-8') as fh:
        for r in csv.DictReader(fh):
            if r['partition'] != 'all':
                continue
            base = r['rule'].split('_')[0].rstrip('0123456789')
            key = base if base in fam else r['rule']
            rows.append([fam.get(key, r['rule']), r['rule'], int(r['n_cells']),
                         round(float(r['coverage']), 3), int(float(r['median_bytes_at_target']))])
    order = list(fam.values()) + ['oracle']
    rows.sort(key=lambda r: (order.index(r[0]) if r[0] in order else 99, -r[3]))
    with open(os.path.join(TAB, 'coverage_bytes.csv'), 'w', newline='') as fh:
        w = csv.writer(fh, lineterminator='\n')
        w.writerow(['family', 'rule', 'n_cells', 'coverage', 'bytes'])
        w.writerows(rows)
    print('\ncoverage vs bytes (19 cells, 95%% mAP@10 target):')
    for r in rows:
        print('  %-22s %-10s cov %.2f  %5d B' % (r[0], r[1], r[3], r[4]))


if __name__ == '__main__':
    crosscell_pairs()
    coverage_bytes()
