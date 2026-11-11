"""Pretty-print the LODO / prior-update and HNSW results for inclusion in the paper."""
import json
import os
import sys

import numpy as np

from sb_paths import SB_OUT

OUT = os.path.join(SB_OUT, 'spectral_budget')


def load(name):
    p = os.path.join(OUT, name)
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    d = load('lodo_true.json')
    if d:
        arms = [k for k, v in d.items() if isinstance(v, dict) and 'summary' in v]
        print('=== true LODO / repair arms (n=%d, eps=%.2f) ===' % (d['n_cells'], d['eps']))
        print('%-22s %6s %8s %8s %8s %8s %8s' %
              ('arm', 'cover', 'median', 'w1.25', 'w2', 'p95', 'max'))
        for k in arms:
            s = d[k]['summary']
            print('%-22s %5.0f%% %7.2fx %7.0f%% %7.0f%% %7.2fx %7.1fx' % (
                k, 100 * s['coverage'], s['median_overhead'],
                100 * s['within_1_25x'], 100 * s['within_2x'],
                s['p95_overhead'], s['max_overhead']))
        print()
        for tag in arms:
            agg = {}
            for r in d[tag]['rows']:
                agg.setdefault(r['holdout'], []).append(r)
            parts = []
            for k, v in agg.items():
                cov = np.mean([x['meet'] for x in v])
                med = np.median([x['overhead'] for x in v])
                parts.append('%s:%.0f%%/%.2fx' % (k, 100 * cov, med))
            print('%-22s %s' % (tag, '  '.join(parts)))
        print()
        bad = [r for r in d['free_2pt']['rows'] if r['overhead'] > 3]
        print('free_2pt tail cells:', [(r['id'], round(r['overhead'], 1), r['meet']) for r in bad])
        miss = [r for r in d['prior_s_plus_2pt']['rows'] if r['meet'] == 0]
        print('prior_s_plus_2pt misses (%d):' % len(miss))
        for r in miss:
            print('   %-38s ov=%.2f s_prior=%.2f rho0=%.3g' %
                  (r['id'], r['overhead'], r['s_prior'], r['rho0']))
        # how far the relation-typed s prior is from the fitted s
        print()
        sp = sorted({r['s_prior'] for r in d['prior_s_plus_2pt']['rows']})
        print('s priors used:', [round(x, 2) for x in sp])

    a = load('ann_hnsw.json')
    if a:
        print()
        print('=== HNSW end-to-end (%s/%s/%s) ===' % (a['encoder'], a['dataset'], a['relation']))
        print('%5s %8s %10s %9s %12s %9s %8s %9s' %
              ('rank', 'denseB', 'indexB', 'exactMAP', 'annMAP@ef128', 'annR/exact',
               'qps_1t', 'qps_24t'))
        for r, v in a['ranks'].items():
            s = [x for x in v['sweep'] if x['efSearch'] == 128][0]
            print('%5s %8d %10.0f %9.4f %12.4f %9.3f %8.0f %9.0f' % (
                r, v['bytes_per_vector_dense'], v['index_bytes_per_vector'],
                v['exact_map10'], s['ann_map10'], s['ann_recall_vs_exact'],
                s['qps_single_thread'], v['qps_24thread_ef128']))
        print()
        for r, v in a['ranks'].items():
            print('rank %s sweep:' % r)
            for s in v['sweep']:
                print('   ef=%-4d qps=%7.0f annR/ex=%.3f annMAP=%.4f' %
                      (s['efSearch'], s['qps_single_thread'],
                       s['ann_recall_vs_exact'], s['ann_map10']))
        # matched-approximation comparison: interpolate qps at annR/ex = 0.95
        print()
        print('qps at ann_recall_vs_exact >= 0.95:')
        for r, v in a['ranks'].items():
            cand = [s for s in v['sweep'] if s['ann_recall_vs_exact'] >= 0.95]
            if cand:
                s = min(cand, key=lambda x: x['qps_single_thread'])
                print('   rank %-4s ef=%-4d qps=%7.0f annR/ex=%.3f annMAP=%.4f' %
                      (r, s['efSearch'], s['qps_single_thread'],
                       s['ann_recall_vs_exact'], s['ann_map10']))


if __name__ == '__main__':
    main()
