"""Counts the paper needs: mask attrition in the fits, and the size of the label partition."""
import glob
import json
import os
from collections import defaultdict

import numpy as np

from sb_paths import SB_OUT

OUT = os.path.join(SB_OUT, 'spectral_budget')


def main():
    tot = drop_lo = drop_hi = drop_rho = kept = 0
    per_cell = []
    part = defaultdict(lambda: {'groups': set(), 'images': 0, 'cells': 0})
    for p in sorted(glob.glob(os.path.join(OUT, 'metric_std_parts', '*.json'))):
        d = json.load(open(p))
        th = json.load(open(os.path.join(OUT, 'theory__%s__%s.json' % (d['encoder'], d['dataset'])))) \
            if os.path.exists(os.path.join(OUT, 'theory__%s__%s.json' % (d['encoder'], d['dataset']))) else None
        grid = d['grid']
        for rel, cell in d.get('cells', {}).items():
            mm = cell.get('metrics', {}).get('map10')
            if not mm or 'curve' not in mm:
                continue
            full = float(cell.get('full', {}).get('map10', 0) or 0)
            n = min(len(mm['curve']), len(cell.get('rho', [])))
            rho = np.array(cell['rho'][:n], float)
            delta = 1.0 - np.array(mm['curve'][:n], float) / full
            m_ok = (delta > 1e-3) & (delta < 1 - 1e-3) & (rho > 1e-5)
            tot += n
            drop_lo += int((delta <= 1e-3).sum())
            drop_hi += int((delta >= 1 - 1e-3).sum())
            drop_rho += int((rho <= 1e-5).sum())
            kept += int(m_ok.sum())
            per_cell.append(m_ok.sum())
            if th and rel in th.get('tasks', {}):
                # group count from the eta2 vector is not enough; report n_gallery only
                pass
    print('points measured %d | kept %d (%.0f%%)' % (tot, kept, 100 * kept / tot))
    print('dropped: Delta<=1e-3 %d | Delta>=1-1e-3 %d | rho<=1e-5 %d'
          % (drop_lo, drop_hi, drop_rho))
    print('kept per cell: min %d median %.0f max %d' %
          (min(per_cell), np.median(per_cell), max(per_cell)))

    # label partition sizes
    for ds, f in (('inshop', 'emb/../meta_inshop.json'),):
        pass
    print()
    print('partition sizes (from build_cells grouping):')
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from sb_common import load_gallery, load_queries
    from sb_grid import build_cells
    for ds in ('inshop', 'sop', 'instre'):
        meta = load_gallery(ds)
        q = load_queries(ds)
        cells = build_cells(ds, meta, q, np.arange(len(meta)))
        for rel, c in cells.items():
            ng = len({meta[i].get('ent') for i in range(len(meta)) if meta[i].get('ent')}) if rel == 'inst' else None
            print('  %-7s %-5s queries=%-6d median |R|=%.1f'
                  % (ds, rel, len(c['rows']), float(np.median([len(r) for r in c['rel']]))))
        # distinct group keys
        for key in ('cat', 'catf', 'ent'):
            vals = {meta[i][key] for i in range(len(meta)) if meta[i].get(key)}
            if vals:
                print('  %-7s groups[%s]=%d' % (ds, key, len(vals)))
        if ds in ('inshop', 'sop'):
            print('  %-7s copy_queries=%d' % (ds, len(q.get('copy', []))))


if __name__ == '__main__':
    main()
