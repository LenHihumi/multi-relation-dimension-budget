"""Cross-corpus contrast in the dimensionality budget: the same encoder, the same
tolerance, two corpora, two relations.

The manuscript measures one number -- on In-shop with CLIP, a category query needs
48 dimensions and an instance query 160, i.e. a 3.3x budget ratio -- and states that
the parameters do not transfer across corpora.  Until now that second claim rested on
a *fitted* surface (rho0, s) that was shown to drift, not on the budget ratio itself
being measured twice.

ABO gives the second corpus: it carries a listing (item) layer and a product-type
layer in one gallery, so `cat` and `inst` can both be built from it exactly as they
are on In-shop.  This script tabulates r* for every (encoder, corpus, relation) it can
find and reports:

  1. per-cell r* and bytes;
  2. the inst/cat ratio for each (encoder, corpus);
  3. whether the direction (inst costs more than cat) holds in every cell;
  4. how far the ratio moves between corpora for the same encoder -- which is the
     quantity the "3.3x is not a constant" claim is about.

Reading rules, fixed before the numbers are looked at:
  * a ratio is only reported when both relations were measured on the same gallery,
    the same encoder, the same eps and the same grid -- mixing any of those makes the
    ratio meaningless;
  * r* is the smallest MEASURED grid rank meeting eps, so it is grid-quantised; the
    grid is geometric with ~1.19 steps, so one step of difference is one step of
    resolution, not a measured 19% effect;
  * the direction claim is count-based (how many of N cells have inst > cat), not a
    test on the ratio values, because the ratios are quantised by the grid.
"""
from __future__ import annotations

import argparse
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data')

# The manuscript's 19 cells live in metric_std_parts; the second corpus is kept in its
# own directory so that adding it cannot change what `repro.sh` rebuilds for the paper.
PART_DIRS = [os.path.join(DATA, 'metric_std_parts'), os.path.join(DATA, 'abo_parts')]

EPS = 0.05
BYTES_PER_DIM = 4          # float32

# corpora that can carry both relations in one gallery
PAIRED = ('inshop', 'abo')


def r_star(cell, grid):
    """Smallest measured grid rank whose quality loss is within eps.

    `grid` lives at the top of the parts file, not inside the cell.
    """
    curve = cell['metrics']['map10']['curve']
    full = cell['full']['map10']
    hits = [r for r, m in zip(grid, curve) if 1.0 - m / full <= EPS]
    return hits[0] if hits else None


def load_pairs():
    """(encoder, corpus) -> {rel: {r_star, bytes, full, rho0, s, r2}}, one pair per file."""
    out = {}
    for parts in PART_DIRS:
        if not os.path.isdir(parts):
            continue
        for f in sorted(os.listdir(parts)):
            if not f.endswith('.json'):
                continue
            enc, corpus = f[:-len('.json')].split('__')
            if corpus not in PAIRED:
                continue
            d = json.load(open(os.path.join(parts, f)))
            if not {'cat', 'inst'} <= set(d['cells']):
                continue
            row = {'n_gallery': d['n_gallery'], 'dim': d['dim'], 'grid': d['grid'],
                   'source': os.path.basename(parts)}
            for rel in ('cat', 'inst'):
                c = d['cells'][rel]
                r = r_star(c, d['grid'])
                if r is None:
                    row = None
                    break
                fit = c['metrics']['map10']['fit']
                row[rel] = {'r_star': r, 'bytes': r * BYTES_PER_DIM, 'full_map10': c['full']['map10'],
                            'rho0': fit['rho0'], 's': fit['s'], 'r2': fit['r2'],
                            'n_query': c['n_query']}
            if row:
                out[(enc, corpus)] = row
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='abo_contrast.json')
    args = ap.parse_args()

    pairs = load_pairs()
    if not pairs:
        raise SystemExit('no (encoder, corpus) pair carries both cat and inst')

    encoders = sorted({e for e, _ in pairs})
    corpora = sorted({c for _, c in pairs})

    rows = []
    for (enc, corpus) in sorted(pairs):
        p = pairs[(enc, corpus)]
        ratio = p['inst']['bytes'] / p['cat']['bytes']
        rows.append({'id': '%s/%s' % (enc, corpus), 'encoder': enc, 'corpus': corpus,
                     'cat': p['cat'], 'inst': p['inst'], 'ratio': ratio,
                     'n_gallery': p['n_gallery'], 'dim': p['dim']})

    # direction: inst must cost more than cat, in every cell
    direction = [r for r in rows if r['inst']['r_star'] > r['cat']['r_star']]
    ties = [r for r in rows if r['inst']['r_star'] == r['cat']['r_star']]

    # within-encoder drift between corpora
    drift = []
    for enc in encoders:
        got = {r['corpus']: r for r in rows if r['encoder'] == enc}
        if len(got) < 2:
            continue
        cs = sorted(got)
        base = got[cs[0]]
        for c in cs[1:]:
            cur = got[c]
            drift.append({'encoder': enc, 'from': cs[0], 'to': c,
                          'ratio_from': base['ratio'], 'ratio_to': cur['ratio'],
                          'rel_change': cur['ratio'] / base['ratio'] - 1.0,
                          'cat_from': base['cat']['r_star'], 'cat_to': cur['cat']['r_star'],
                          'inst_from': base['inst']['r_star'], 'inst_to': cur['inst']['r_star']})

    # spread of the ratio across everything measured
    ratios = [r['ratio'] for r in rows]

    out = {
        'eps': EPS, 'bytes_per_dim': BYTES_PER_DIM,
        'encoders': encoders, 'corpora': corpora, 'n_cells': len(rows),
        'rows': rows,
        'direction': {'inst_gt_cat': len(direction), 'ties': len(ties), 'of': len(rows)},
        'ratio': {'min': min(ratios), 'max': max(ratios),
                  'spread': max(ratios) / min(ratios),
                  'median': sorted(ratios)[len(ratios) // 2]},
        'drift': drift,
        'checks': {
            # grid steps are ~1.19x, so a ratio is only resolvable to that grain
            'grid_step_ratio': (lambda g: g[1] / g[0] if len(g) > 1 else None)(
                next(iter(pairs.values()))['grid']),
            'single_corpus_ratios': sorted({round(r['ratio'], 3) for r in rows
                                            if r['corpus'] == corpora[0]}),
        },
    }

    print('%-22s %-8s %8s %8s %10s %10s %7s' %
          ('cell', 'corpus', 'cat r*', 'inst r*', 'cat bytes', 'inst bytes', 'ratio'))
    for r in rows:
        print('%-22s %-8s %8d %8d %10d %10d %6.2fx' %
              (r['encoder'], r['corpus'], r['cat']['r_star'], r['inst']['r_star'],
               r['cat']['bytes'], r['inst']['bytes'], r['ratio']))

    print('\ndirection: inst > cat in %d of %d cells (ties %d)'
          % (out['direction']['inst_gt_cat'], out['n_cells'], out['direction']['ties']))
    print('ratio across all cells: %.2fx - %.2fx (median %.2fx, spread %.2fx)'
          % (out['ratio']['min'], out['ratio']['max'], out['ratio']['median'],
             out['ratio']['spread']))

    print('\nsame encoder, two corpora:')
    for d in drift:
        print('  %-20s %s %.2fx -> %s %.2fx  (%+.0f%%; cat %d->%d, inst %d->%d)'
              % (d['encoder'], d['from'], d['ratio_from'], d['to'], d['ratio_to'],
                 100 * d['rel_change'], d['cat_from'], d['cat_to'],
                 d['inst_from'], d['inst_to']))

    with open(os.path.join(DATA, 'results', args.out), 'w') as f:
        json.dump(out, f, indent=1)
    print('\nwrote %s' % args.out)


if __name__ == '__main__':
    main()
