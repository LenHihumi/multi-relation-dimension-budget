"""Is there a unified, label-only rule for "which budget rule to use"?

The paper's boundary says labels pay on item retrieval and cost on category retrieval, where a
label-free participation ratio covers every cell in 208 bytes against 320.  Readers reasonably ask
for one decision rule instead of a per-relation story.  This script asks the only question that
matters for it: can any rule that costs no extra retrieval reach the storage of choosing the
cheapest rule per cell?

Inputs are the Table 7 grid (`metric_std.json` -> `rule_compare['map10']`), one row per (cell,
rule) with bytes, `meets` and the oracle rank, so nothing here needs new retrieval.

Three things are reported:

  uniform        the single rule that meets eps on every cell (no selection at all)
  two-rule oracle  perfect choice between the participation ratio and that rule, per cell
                   (an upper bound on what any selector using only these two can do)
  per-cell best  cheapest meeting rule per cell, the target every selector is measured against

Usage
  python3 sb_unified_rule.py
"""
from __future__ import annotations

import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DATA = os.environ.get('SB_DATA') or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'data')
TARGET_RATIO = 1.10


def main():
    d = json.load(open(os.path.join(DATA, 'results', 'metric_std.json')))
    cells = d['rule_compare']['map10']['cells']
    by = collections.defaultdict(dict)
    for c in cells:
        by[(c['enc'], c['ds'], c['rel'])][c['rule']] = c

    # the uniform rule = the one that meets eps everywhere; report every candidate that does
    names = sorted({c['rule'] for c in cells if c['rule'] != 'oracle'})
    uniform = {}
    for r in names:
        rows = [v[r] for v in by.values()]
        uniform[r] = {'misses': sum(1 for x in rows if not x['meets']),
                      'bytes': sum(x['bytes'] for x in rows)}
    full = [r for r in names if uniform[r]['misses'] == 0]
    print('rules that meet eps on all %d cells: %s' % (len(by), full or 'none'))
    if not full:
        return 2
    uni = min(full, key=lambda r: uniform[r]['bytes'])

    best = pr_oracle = uni_total = 0.0
    rows = []
    for k, v in sorted(by.items()):
        meet = {r: c for r, c in v.items() if r != 'oracle' and c['meets']}
        b = min(c['bytes'] for c in meet.values())
        two = [meet[r]['bytes'] for r in ('pr', uni) if r in meet]
        pr = meet['pr']['bytes'] if 'pr' in meet else None
        best += b
        pr_oracle += min(two) if two else b
        uni_total += v[uni]['bytes']
        rows.append(('/'.join(k), b, pr, v[uni]['bytes'], min(two) if two else b))

    print('\n%-26s %6s %7s %8s %8s' % ('cell', 'best', 'pr', uni, 'oracle2'))
    for r in rows:
        print('%-26s %6d %7s %8d %8d'
              % (r[0], r[1], r[2] if r[2] is not None else '-', r[3], r[4]))

    print('\nuniform "%s": %d bytes = %.3fx per-cell best, 0 misses'
          % (uni, uni_total, uni_total / best))
    print('two-rule oracle {pr, %s}: %d bytes = %.3fx per-cell best'
          % (uni, pr_oracle, pr_oracle / best))
    print('target %.2fx -> %s'
          % (TARGET_RATIO, 'met' if pr_oracle / best <= TARGET_RATIO else 'NOT met'))
    print('\nReading: a selector cannot beat the two-rule oracle without extra evaluations, so if'
          ' that bound is above the target, no label-only decision rule reaches it.  The gap is'
          ' the price of not measuring which rule is cheaper on the cell in front of you.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
