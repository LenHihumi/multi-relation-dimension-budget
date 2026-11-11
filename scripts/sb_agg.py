"""Aggregate the ordering-ablation and quantisation outputs into the tables the paper prints."""
import json
import sys
from collections import defaultdict

import numpy as np


def order(path):
    d = json.load(open(path))
    cells = d['cells']
    print('cells %d, ranks %s' % (len(cells), d['ranks']))
    for rel in ('inst', 'cat'):
        acc = defaultdict(list)
        for c in cells:
            if c['rel'] != rel:
                continue
            acc[c['r']].append((c['map_lam'], c['map_om'], c['map_rand'], c['map_rand_rho']))
        print('  %-4s %5s %7s %7s %7s %7s %7s %7s' %
              ('rel', 'r', 'lam', 'omega', 'rand', 'rand-rho', 'om-lam', 'n'))
        for r in sorted(acc):
            a = np.array(acc[r])
            print('  %-4s %5d %7.4f %7.4f %7.4f %7.4f %+7.4f %7d'
                  % (rel, r, a[:, 0].mean(), a[:, 1].mean(), a[:, 2].mean(), a[:, 3].mean(),
                     (a[:, 1] - a[:, 0]).mean(), len(a)))
    diffs = [c['map_om'] - c['map_lam'] for c in cells]
    print('  omega - lam: mean %+.4f  median %+.4f  max |d| %.4f  share |d|<0.002: %.2f'
          % (np.mean(diffs), np.median(diffs), np.max(np.abs(diffs)),
             np.mean([abs(x) < 0.002 for x in diffs])))
    rel_drop = [(c['map_rand'] - c['map_lam']) / max(c['map_lam'], 1e-9) for c in cells]
    print('  rand  - lam: mean relative drop %.1f%%' % (100 * np.mean(rel_drop)))


def quant(path):
    d = json.load(open(path))
    print('quant cells %d, bits %d' % (len(d['cells']), d['bits']))
    for c in d['cells']:
        print('  %-26s %-4s r* %3d -> %3d  (%.2fx)  rho0 %.2f->%.2f  s %.2f->%.2f  max|dmAP| %.4f'
              % ('%s/%s' % (c['enc'][:18], c['ds']), c['rel'], c['rstar_f32'], c['rstar_i8'],
                 c['shift'], c['rho0_f32'], c['rho0_i8'], c['s_f32'], c['s_i8'], c['max_abs_drop']))
    print('  summary', json.dumps(d['summary']))


def scale(path):
    d = json.load(open(path))
    ef = max(d['efs'])
    print('scale: ef=%d' % ef)
    print('  %8s %4s %7s %8s %8s %8s %7s %6s' %
          ('n', 'r', 'build_s', 'p50_ms', 'p99_ms', 'qps_1t', 'mAP_i', 'links'))
    for row in d['rows']:
        if row['ef'] != ef:
            continue
        print('  %8d %4d %7.1f %8.3f %8.3f %8.0f %7.4f %6.2f'
              % (row['n'], row['r'], row['build_s'], row['p50_ms'], row['p99_ms'],
                 row['qps_1t'], row['map_inst'], row['links_share']))
    print('  speed-up QPS(r_min)/QPS(r_max):', d.get('speedup'))


if __name__ == '__main__':
    kind = sys.argv[1]
    {'order': order, 'quant': quant, 'scale': scale}[kind](sys.argv[2])
