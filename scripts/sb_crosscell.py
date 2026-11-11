"""Cross-cell byte margin of the measured budget over the 90%-variance rule.

No new evaluation: every rank comes from an already frozen result file.

  r90        grid rank retaining 90% of the variance (rule `energy90`)
  r*_rel     per-relation oracle rank meeting the 5% tolerance (rule `law`,
             field `r_oracle`), i.e. the rank the deployment issues before the
             margin is added
  bytes      one graph = 4 r + 2 M 4 (M = 32 links of 4 bytes each)

Deployments, per encoder x corpus pair:

  var90    one graph at r90
  unified  one graph at max_rel r*_rel
  pair     one graph per relation at its own r*_rel

The paper's two end-to-end cells anchor this table.  In-shop/CLIP comes out at
1.14x here against the 1.12x that Sec. VII-D prints, because the grid rank is
snapped to 192 while the paper quotes the un-snapped 90% crossing (187);
the text cell (r90 = 648, budget 448) reproduces 1.39x exactly.  Do not read
the two rows as disagreeing: same rule, one grid step apart.

Usage: python3 sb_crosscell.py [--json out.json]
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
M = 32
LINK = 2 * M * 4          # bytes of the link list, per graph


def load():
    path = os.path.join(DATA, 'results', 'rules_compare.json')
    d = json.load(open(path))
    by = collections.defaultdict(dict)
    for c in d['cells']:
        by[(c['enc'], c['ds'])][(c['rel'], c['rule'])] = c
    return d['meta']['eps'], by


def pairs(by):
    out = []
    for (enc, ds), m in sorted(by.items()):
        rels = sorted({k[0] for k in m if k[1] == 'law'})
        if not rels:
            continue
        r90 = m[(rels[0], 'energy90')]['r']
        rstar = {r: m[(r, 'law')]['r_oracle'] for r in rels}
        e90_meets = all(m[(r, 'energy90')].get('meets') for r in rels)
        e90_delta = max(m[(r, 'energy90')].get('delta', 0.0) for r in rels)
        raw_meets = all(m[(r, 'law')].get('meets') for r in rels)
        uni = 4 * max(rstar.values()) + LINK
        var = 4 * r90 + LINK
        pr = sum(4 * v + LINK for v in rstar.values())
        out.append(dict(
            enc=enc, ds=ds, rels=rels, r90=r90, rstar=rstar,
            bytes_var90=var, bytes_unified=uni, bytes_pair=pr,
            ratio_var90_unified=var / uni,
            ratio_var90_pair=var / pr,
            rank_ratio_r90_over_max=r90 / max(rstar.values()),
            var90_meets=e90_meets, var90_delta=e90_delta,
            budget_meets=raw_meets,
        ))
    return out


def report(rows, eps):
    print('eps=%.2f  M=%d  link=%d B/graph' % (eps, M, LINK))
    print('%-14s %-7s %-9s %4s %-14s %6s %6s %6s %6s %6s %5s'
          % ('encoder', 'corpus', 'relations', 'r90', 'r* per rel', 'var90',
             'unified', 'pair', 'v/u', 'v/p', 'v90ok'))
    for r in sorted(rows, key=lambda x: -x['ratio_var90_unified']):
        print('%-14s %-7s %-9s %4d %-14s %6d %6d %6d %6.2f %6.2f %5s'
              % (r['enc'], r['ds'], ','.join(r['rels']), r['r90'],
                 ', '.join('%s:%d' % (k, v) for k, v in sorted(r['rstar'].items())),
                 r['bytes_var90'], r['bytes_unified'], r['bytes_pair'],
                 r['ratio_var90_unified'], r['ratio_var90_pair'], r['var90_meets']))
    valid = [r for r in rows if r['var90_meets']]
    ru = [r['ratio_var90_unified'] for r in valid]
    print('\npairs %d, var90 meets tolerance on %d' % (len(rows), len(valid)))
    print('var90/unified bytes on the valid pairs: median %.2fx, range %.2f-%.2fx'
          % (st.median(ru), min(ru), max(ru)))
    # the pre-registered criterion: r90 >= 2.5 x the largest measured budget
    thr = 2.5
    hi = [r for r in valid if r['rank_ratio_r90_over_max'] >= thr]
    lo = [r for r in valid if r['rank_ratio_r90_over_max'] < thr]
    print('criterion r90/max r* >= %.1f: %d pairs, byte margin %.2f-%.2fx'
          % (thr, len(hi), min([r['ratio_var90_unified'] for r in hi]),
             max([r['ratio_var90_unified'] for r in hi])) if hi else 'none')
    print('criterion below %.1f: %d pairs, byte margin %.2f-%.2fx'
          % (thr, len(lo), min([r['ratio_var90_unified'] for r in lo]),
             max([r['ratio_var90_unified'] for r in lo])) if lo else 'none')
    above2 = [r for r in valid if r['ratio_var90_unified'] >= 2.0]
    print('byte margin >= 2x: %d of %d valid pairs' % (len(above2), len(valid)))
    return dict(n_pairs=len(rows), n_valid=len(valid),
                median_ratio=st.median(ru), min_ratio=min(ru), max_ratio=max(ru),
                threshold=thr,
                n_ge_thr=len(hi), n_lt_thr=len(lo),
                n_ge_2x=len(above2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', default=os.path.join(DATA, 'results', 'crosscell.json'))
    a = ap.parse_args()
    eps, by = load()
    rows = pairs(by)
    s = report(rows, eps)
    json.dump(dict(eps=eps, M=M, link_bytes=LINK, source='rules_compare.json',
                   pairs=rows, summary=s), open(a.json, 'w'), indent=1)
    print('\nwrote', os.path.abspath(a.json))


if __name__ == '__main__':
    main()
