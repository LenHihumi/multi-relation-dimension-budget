"""Regenerate every table of the projection paper from the JSONs the runs wrote.

Reads only files, prints LaTeX rows in the order the tables use them, so a reviewer can check the paper
against the artifact without opening a notebook. Run inside the container (paths default to the shared
disk).

  python3 collect_proj.py            # all tables
  python3 collect_proj.py --table 1  # one table
"""
import argparse
import glob
import json
import os

from sb_paths import SB_ROOT, SB_OUT, SB_CACHE, SB_MODELS, REPO  # paths resolve relative to the deposit
O = SB_OUT
ENCS = ['clip_vitl14', 'dinov3_vitl16', 'siglip2_so400m', 'wemm_2b', 'qwen3vl_emb_8b']
NAME = {'clip_vitl14': 'CLIP ViT-L/14', 'dinov3_vitl16': 'DINOv3 ViT-L/16',
        'siglip2_so400m': 'SigLIP2 SO400M', 'wemm_2b': 'WeMM-2B', 'qwen3vl_emb_8b': 'Qwen3-VL-8B'}
DIM = {'clip_vitl14': 768, 'dinov3_vitl16': 1024, 'siglip2_so400m': 1152, 'wemm_2b': 2048,
       'qwen3vl_emb_8b': 4096}


def load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def table1():
    print('% table:main  -- fine relation, independent instrument, test split')
    for e in ENCS:
        d = load(os.path.join(O, 'projeval__%s__inst.json' % e))
        if not d:
            print('  missing %s' % e)
            continue
        p, w = d['pca']['test'], d['w']['test']
        pc = {int(k): v for k, v in p['curve'].items()}
        wc = {int(k): v for k, v in w['curve'].items()}
        print('%-22s & %4d & %3d & %2d & %5.1f$\\times$ & %.3f / %.3f & %.3f / %.3f \\\\'
              % (NAME[e], DIM[e], p['rstar'], w['rstar'], p['rstar'] / w['rstar'],
                 pc[12], wc[12], pc[160], wc[160]))
        full = wc[max(wc)]
        hit = [r for r in sorted(wc) if 1 - wc[r] / full <= 0.05]
        print('    self-referenced learned budget: %s (own full width %.3f)'
              % (hit[0] if hit else 'none', full))


def table2():
    print('% table:coarse  -- coarse relation')
    for e in ENCS:
        d = load(os.path.join(O, 'projeval__%s__cat.json' % e))
        if not d:
            print('  missing %s' % e)
            continue
        r = [d['pca'][t]['rstar'] if t in d['pca'] else None for t in ('test', 'all')]
        w = [d['w'][t]['rstar'] if t in d['w'] else None for t in ('test', 'all')]
        print('%-22s & %3s & %3s & %3s & %3s \\\\'
              % (NAME[e], r[0], '---' if w[0] is None else w[0], r[1], '---' if w[1] is None else w[1]))


def table3():
    print('% table:ablate  -- budget by basis (trainer instrument, same queries/grid)')
    print('%-22s & %3s & %3s & %3s & %3s & %3s & %3s \\\\' % ('encoder', 'pca', 'fisher', 'task', 'rot',
                                                             'ce', 'rank'))
    for e in ENCS:
        d = load(os.path.join(O, 'proj__%s__inst.json' % e))
        if not d:
            continue
        m = d['methods']

        def g(k):
            v = m.get(k, {}).get('rstar')
            return '---' if v is None else v
        print('%-22s & %3s & %3s & %3s & %3s & %3s & %3s \\\\'
              % (NAME[e], g('pca'), g('fisher'), g('taskenergy'), g('randrot'), g('nested-ce'),
                 g('nested')))


def impact():
    print('% impact table -- 52,591 rows, M=32')
    n, L = 52591, 2 * 32 * 4
    for label, b, r in (('PCA, float32', 4, 160), ('PCA, int8', 1, 160),
                        ('learned, int8', 1, 12), ('learned, 4-bit', 0.5, 12)):
        bv = b * r + L
        tot = bv * n / 1e6
        print('%-16s & %3d & %5.1f MB & %4.0f\\%% vectors \\\\' % (label, bv, tot, 100 * b * r / bv))


def ann():
    print('% proj_ann -- end to end HNSW')
    for f in sorted(glob.glob(os.path.join(O, 'projann__*.json'))):
        d = json.load(open(f))
        print(os.path.basename(f))
        for k, v in sorted(d['rows'].items()):
            print('   %-12s mAP %.4f  %4.0f B/vec  %6.1f MB  build %5.1fs  QPS %6.0f'
                  % (k, v['mAP'], v['bytes_per_vec'], v['total_mb'], v['build_s'], v['qps']))


TABLES = {'1': table1, '2': table2, '3': table3, 'impact': impact, 'ann': ann}

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--table', default='')
    a = ap.parse_args()
    for k, fn in TABLES.items():
        if a.table and a.table != k:
            continue
        fn()
        print()
