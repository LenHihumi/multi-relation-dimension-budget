"""Cross-corpus protocol evaluation, in the three reporting conventions that were
pre-registered before the fourth corpus existed (NEXT_METHODS.md section 13).

Adding a corpus answers two different questions and they must not be mixed:

  (1) FULL SET -- carry the old parameters straight over to the new cells.  A low
      number here can mean either "the protocol does not generalise" or "this corpus
      sits outside the reference range", and those are different findings.

  (2) IN-REFERENCE SUBSET -- restrict to the new cells whose spectral statistics fall
      inside the 5-95 percentile band of the reference corpus set.  If the protocol
      works here but not on the full set, the conclusion is "generalises inside the
      range, needs recalibration outside it", not "does not generalise".

  (3) RECALIBRATED -- measure on the target corpus instead of transferring.  This arm
      exists already (`free_2pt`, `prior_s_plus_2pt` in the same lodo run): two probes
      on the cell itself, no cross-corpus parameters.

The reference band is computed per relation, because `inst` and `cat` have different
group counts by construction and pooling them would make every `inst` cell look out of
range.

Statistics used (label-free, computed from the eigenvalue spectrum only):
  * participation ratio  (sum lambda)^2 / sum lambda^2
  * spectral entropy     exp(-sum p log p),  p = lambda / sum lambda
  * group count          tasks[rel]['n_groups']  (label-aware, reported separately)
"""
from __future__ import annotations

import argparse
import json
import math
import os

from sb_paths import SB_ROOT, SB_OUT, SB_CACHE, SB_MODELS, REPO  # paths resolve relative to the deposit
OUT = SB_OUT + '/sb29'
REFERENCE = ('inshop', 'instre', 'sop')      # the 19-cell manuscript set
NEW = ('abo',)


def spectrum_stats(path):
    d = json.load(open(path))
    lam = [x for x in d['lambda'] if x > 0]
    s1 = sum(lam)
    s2 = sum(x * x for x in lam)
    pr = s1 * s1 / s2
    p = [x / s1 for x in lam]
    h = math.exp(-sum(pi * math.log(pi) for pi in p))
    return {'pr': pr, 'spec_entropy': h, 'dim': d['dim'], 'energy': d['energy'],
            'tasks': {k: {'n_groups': v['n_groups'],
                          'group_size_median': v['group_size_median']}
                      for k, v in d['tasks'].items()}}


def pct(vals, q):
    v = sorted(vals)
    if not v:
        return None
    i = (len(v) - 1) * q
    lo, hi = int(math.floor(i)), int(math.ceil(i))
    return v[lo] if lo == hi else v[lo] + (v[hi] - v[lo]) * (i - lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lodo', default='lodo_true29.json')
    ap.add_argument('--out', default='crosscorpus.json')
    args = ap.parse_args()

    whole = json.load(open(os.path.join(OUT, args.lodo)))
    lodo = whole['spec_lodo_dataset']        # cross-corpus transfer, the arm of interest
    rows = lodo['rows']

    cells = []
    for r in rows:
        enc, ds, rel = r['id'].split('/')
        th = os.path.join(OUT, 'theory__%s__%s.json' % (enc, ds))
        st = spectrum_stats(th)
        cells.append({'id': r['id'], 'encoder': enc, 'corpus': ds, 'rel': rel,
                      'meet': bool(r['meet']), 'overhead': r['overhead'],
                      'pred': r['pred'], 'pr': st['pr'], 'spec_entropy': st['spec_entropy'],
                      'dim': st['dim'],
                      'n_groups': st['tasks'].get(rel, {}).get('n_groups'),
                      'group_median': st['tasks'].get(rel, {}).get('group_size_median')})

    # per-relation reference band, from the manuscript's three corpora only
    band = {}
    for rel in sorted({c['rel'] for c in cells}):
        ref = [c for c in cells if c['corpus'] in REFERENCE and c['rel'] == rel]
        band[rel] = {k: {'lo': pct([c[k] for c in ref], 0.05),
                         'hi': pct([c[k] for c in ref], 0.95),
                         'n_ref': len(ref)}
                     for k in ('pr', 'spec_entropy', 'n_groups')}

    def in_ref(c, keys):
        b = band[c['rel']]
        return all(b[k]['lo'] is not None and b[k]['lo'] <= c[k] <= b[k]['hi'] for k in keys)

    for c in cells:
        c['in_ref_label_free'] = in_ref(c, ('pr', 'spec_entropy'))
        c['in_ref_label_aware'] = in_ref(c, ('pr', 'spec_entropy', 'n_groups'))

    def summarise(sub):
        if not sub:
            return {'n': 0}
        ov = sorted(c['overhead'] for c in sub)
        return {'n': len(sub), 'meet': sum(1 for c in sub if c['meet']),
                'coverage': sum(1 for c in sub if c['meet']) / len(sub),
                'median_overhead': ov[len(ov) // 2],
                'under': sum(1 for c in sub if c['overhead'] < 1)}

    new = [c for c in cells if c['corpus'] in NEW]
    ref = [c for c in cells if c['corpus'] in REFERENCE]

    out = {
        'n_cells': len(cells), 'reference': list(REFERENCE), 'new': list(NEW),
        'band': band,
        # every arm in the file, so the recalibrated ones sit next to the transfer one
        'arms': {k: v['summary'] for k, v in whole.items()
                 if isinstance(v, dict) and 'summary' in v},
        'table': {
            'full_set': summarise(cells),
            'reference_set': summarise(ref),
            'new_corpus': summarise(new),
            'new_in_ref_free': summarise([c for c in new if c['in_ref_label_free']]),
            'new_out_ref_free': summarise([c for c in new if not c['in_ref_label_free']]),
            'new_in_ref_aware': summarise([c for c in new if c['in_ref_label_aware']]),
        },
        'cells': cells,
    }

    print('reference band (per relation, 5-95 pct of the three manuscript corpora):')
    for rel, b in band.items():
        print('  %-5s pr [%.1f, %.1f]  entropy [%.1f, %.1f]  groups [%s, %s]  (n=%d)'
              % (rel, b['pr']['lo'], b['pr']['hi'], b['spec_entropy']['lo'], b['spec_entropy']['hi'],
                 _fmt(b['n_groups']['lo']), _fmt(b['n_groups']['hi']), b['pr']['n_ref']))

    t = out['table']
    print('\n%-26s %3s %6s %8s %8s %6s' % ('subset', 'n', 'meet', 'cover', 'median', 'under'))
    for name in ('full_set', 'reference_set', 'new_corpus', 'new_in_ref_free',
                 'new_out_ref_free', 'new_in_ref_aware'):
        s = t[name]
        if not s.get('n'):
            print('%-26s   0  (empty)' % name)
            continue
        print('%-26s %3d %6d %7.0f%% %7.2fx %6d'
              % (name, s['n'], s['meet'], 100 * s['coverage'], s['median_overhead'], s['under']))

    print('\nnew-corpus cells (ABO), label-free in-reference flag:')
    for c in new:
        print('  %-32s pred %6.1f  ov %.2f  %-5s %-6s pr %5.1f  H %5.1f  groups %s'
              % (c['id'], c['pred'], c['overhead'], 'meet' if c['meet'] else 'FAIL',
                 'in' if c['in_ref_label_free'] else 'OUT', c['pr'], c['spec_entropy'],
                 _fmt(c['n_groups'])))

    print('\nrecalibrated arms (same held-out corpus, probes on the cell):')
    for k in ('free_2pt', 'prior_s_plus_2pt', 'prior_s_plus_1pt', 'spec_lodo_dataset'):
        if k in out['arms']:
            a = out['arms'][k]
            print('  %-20s coverage %.0f%%  median %.2fx  p95 %.2fx'
                  % (k, 100 * a.get('coverage', 0), a.get('median_overhead', 0),
                     a.get('p95_overhead', 0)))

    with open(os.path.join(OUT, 'results' if os.path.isdir(os.path.join(OUT, 'results')) else '.',
                           args.out), 'w') as f:
        json.dump(out, f, indent=1)
    print('\nwrote %s' % args.out)


def _fmt(v):
    return '-' if v is None else ('%.0f' % v)


if __name__ == '__main__':
    main()
