"""Does the learned width survive an actual HNSW index?

The projection result is measured by exact search, which is the right instrument for a budget question but
not the thing a deployment runs.  This builds the index in both spaces at a few widths and reports mAP,
bytes and QPS, so the claim "r* falls from 160 to 12-16" can be checked end to end rather than in a
dot-product oracle.

Usage:
  python3 proj_ann.py --enc clip_vitl14 --ds inshop --w .../W_clip_inst.npz
"""
import argparse
import json
import os
import time

import numpy as np

from sb_paths import SB_ROOT, SB_OUT, SB_CACHE, SB_MODELS, REPO  # paths resolve relative to the deposit
DUMP = os.environ.get('DUMP', SB_OUT + '/spectral_budget')
CACHE = os.environ.get('CACHE', SB_CACHE)
M, EFC, EF = 32, 200, 128


def load(enc, ds):
    base = os.path.join(DUMP, 'emb', '%s__%s' % (enc, ds))
    meta = json.load(open(base + '.json'))
    if os.path.exists(base + '.npy'):
        emb = np.load(base + '.npy', mmap_mode='r')
    else:
        emb = np.memmap(base + '.bin', dtype=np.float32, mode='r').reshape(meta['n'], meta['dim'])
    rows = [json.loads(l) for l in open(os.path.join(CACHE, ds, 'gallery.jsonl'))]
    return np.asarray(emb, dtype=np.float32), rows


def build_search(Z, q, k=11):
    import faiss
    faiss.omp_set_num_threads(48)
    idx = faiss.IndexHNSWFlat(Z.shape[1], M)
    idx.hnsw.efConstruction = EFC
    t0 = time.time()
    idx.add(np.ascontiguousarray(Z))
    tb = time.time() - t0
    idx.hnsw.efSearch = EF
    faiss.omp_set_num_threads(1)
    t0 = time.time()
    _, ids = idx.search(np.ascontiguousarray(Z[q]), k)
    tq = time.time() - t0
    return ids, tb, len(q) / max(tq, 1e-9)


def ap_ids(ids, q, per_row, groups, k=10):
    aps = []
    for i, qi in enumerate(q):
        rel = groups[per_row[qi]]
        h, s = 0, 0.0
        for rank, j in enumerate(ids[i][:k], 1):
            if int(j) in rel and int(j) != int(qi):
                h += 1
                s += h / rank
        aps.append(s / min(len(rel) - 1, k))
    return float(np.mean(aps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--enc', default='clip_vitl14')
    ap.add_argument('--ds', default='inshop')
    ap.add_argument('--w', required=True)
    ap.add_argument('--rel', default='inst')
    ap.add_argument('--widths', default='12,16,24,32,48')
    ap.add_argument('--pca-widths', default='48,96,160,320')
    ap.add_argument('--out', default='')
    a = ap.parse_args()

    X, rows = load(a.enc, a.ds)
    key = 'ent' if a.rel == 'inst' else 'cat'
    per_row = np.array([r[key] for r in rows])
    codes, inv = np.unique(per_row, return_inverse=True)
    per_row = inv
    groups = {}
    for i, v in enumerate(per_row):
        groups.setdefault(v, set()).add(i)

    nz = np.load(a.w, allow_pickle=True)
    W, V, mu, is_te = nz['W'], nz['V'], nz['mu'], nz['is_te']
    qs = json.load(open(os.path.join(CACHE, a.ds, 'queries.json')))
    q = np.array([int(i) for i in qs.get('inst')])
    q = q[q < len(rows)]
    q = q[is_te[q]]
    print('%s/%s rel=%s queries(test) %d' % (a.enc, a.ds, a.rel, len(q)))

    Xc = X - mu
    res = {'enc': a.enc, 'ds': a.ds, 'rel': a.rel, 'queries': int(len(q)), 'rows': {}}
    for space, B, widths in (('learned', W, a.widths), ('pca', V, a.pca_widths)):
        for r in [int(x) for x in widths.split(',')]:
            if r > B.shape[1]:
                continue
            Z = np.ascontiguousarray(Xc @ B[:, :r])
            Z /= np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9)
            ids, tb, qps = build_search(Z, q)
            m = ap_ids(ids, q, per_row, groups)
            b = 4 * r + 2 * M * 4
            res['rows']['%s@%d' % (space, r)] = {'mAP': m, 'bytes_per_vec': b, 'build_s': tb,
                                                 'qps': qps, 'total_mb': b * len(rows) / 1e6}
            print('%-8s r=%-3d mAP %.4f  %5.0f B/vec  %7.1f MB  build %5.1fs  QPS %6.0f'
                  % (space, r, m, b, b * len(rows) / 1e6, tb, qps))
    if a.out:
        json.dump(res, open(a.out, 'w'), indent=1)
        print('wrote', a.out)


if __name__ == '__main__':
    main()
