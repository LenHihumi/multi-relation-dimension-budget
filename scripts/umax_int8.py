"""Is the saving from sharing, or from quantising the shared store?

The byte law compares a float32 uniform index against a shared design whose rerank store is int8.  A
reviewer's objection is exact: if the uniform index can also be int8, it is `1*r_item` bytes, and the
shared design is `4*r_cat + 1*r_item` -- never cheaper.  So the honest question is not "is int8 good" but
"can the *graph* be traversed in int8".  This measures four designs on the same vectors:

  umax-f32   graph @ r_item, float32          4 r_i + links
  umax-i8    graph @ r_item, scalar-quantised 1 r_i + links
  share-f32  graph @ r_cat, float32 + int8 rerank store at r_i   4 r_c + 1 r_i + links
  share-i8   graph @ r_cat, int8   + int8 rerank store at r_i    1 r_c + 1 r_i + links

If umax-i8 keeps quality, sharing as a storage story collapses to the r_cat < 0.25 r_item region and the
paper has to say so.  If int8 traversal costs quality, the shared design's saving is real and the
precondition is "the uniform index must keep its graph in float32".
"""
import argparse
import json
import os
import time

import numpy as np

from sb_paths import SB_ROOT, SB_OUT, SB_CACHE, SB_MODELS, REPO  # paths resolve relative to the deposit
CACHE = os.environ.get('CACHE', SB_CACHE)
DUMP = os.environ.get('DUMP', SB_OUT + '/spectral_budget')
M, EFC, EF, TOPK = 32, 200, 128, 100
LINKS = 2 * M * 4


def load(enc, ds):
    base = os.path.join(DUMP, 'emb', '%s__%s' % (enc, ds))
    meta = json.load(open(base + '.json'))
    emb = np.load(base + '.npy', mmap_mode='r') if os.path.exists(base + '.npy') else \
        np.memmap(base + '.bin', dtype=np.float32, mode='r').reshape(meta['n'], meta['dim'])
    rows = [json.loads(l) for l in open(os.path.join(CACHE, ds, 'gallery.jsonl'))]
    return np.asarray(emb, dtype=np.float32), rows, os.path.join(DUMP, 'metric_std_parts',
                                                                '%s__%s.json' % (enc, ds))


def build_f32(Z, threads=48):
    import faiss
    faiss.omp_set_num_threads(threads)
    idx = faiss.IndexHNSWFlat(Z.shape[1], M)
    idx.hnsw.efConstruction = EFC
    idx.add(np.ascontiguousarray(Z))
    idx.hnsw.efSearch = EF
    faiss.omp_set_num_threads(1)
    return idx


def build_i8(Z, threads=48):
    import faiss
    faiss.omp_set_num_threads(threads)
    idx = faiss.IndexHNSWSQ(Z.shape[1], faiss.ScalarQuantizer.QT_8bit, M)
    idx.hnsw.efConstruction = EFC
    idx.train(np.ascontiguousarray(Z))
    idx.add(np.ascontiguousarray(Z))
    idx.hnsw.efSearch = EF
    faiss.omp_set_num_threads(1)
    return idx



def rerank_batch(Zc, Z8, q, idx, topk):
    """Batched rerank: score all candidates of all queries in one einsum.

    A per-query Python loop over 100 candidates costs more than the graph search it is supposed to
    complement, which would make the shared design look slow for an implementation reason.  This is the
    fair version: one int32 einsum over (queries x candidates x dims).
    """
    _, cand = idx.search(np.ascontiguousarray(Zc[q].astype(np.float32)), topk + 1)
    B = len(q)
    C = np.empty((B, topk), dtype=np.int64)
    for i, qi in enumerate(q):
        row = [int(j) for j in cand[i] if int(j) != int(qi)]
        C[i] = row[:topk]
    V = Z8[C].astype(np.int32)                       # B x topk x r
    Q = Z8[q].astype(np.int32)                       # B x r
    S = np.einsum('bkd,bd->bk', V, Q)
    top = np.argsort(-S, axis=1)[:, :10]
    return C[np.arange(B)[:, None], top]

def search(idx, Q, k):
    _, ids = idx.search(np.ascontiguousarray(Q.astype(np.float32)), k)
    return ids


def strip(ids, q, k):
    out = []
    for r, qi in enumerate(q):
        out.append([int(j) for j in ids[r] if int(j) != int(qi)][:k])
    return np.array(out)


def ap10(ids, q, per_row, groups):
    aps = []
    for i, qi in enumerate(q):
        rel = groups[per_row[qi]] - {qi}
        h, ap = 0, 0.0
        for rank, j in enumerate(ids[i][:10], 1):
            if j in rel:
                h += 1
                ap += h / rank
        aps.append(ap / min(len(rel), 10))
    return float(np.mean(aps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--enc', default='clip_vitl14')
    ap.add_argument('--ds', default='inshop')
    ap.add_argument('--r-item', type=int, default=0)
    ap.add_argument('--r-cat', type=int, default=0)
    ap.add_argument('--queries', type=int, default=500)
    ap.add_argument('--out', default='')
    a = ap.parse_args()

    X, rows, parts_path = load(a.enc, a.ds)
    if not a.r_item or not a.r_cat:
        parts = json.load(open(parts_path))
        a.r_item = a.r_item or int(parts['cells']['inst']['metrics']['map10']['fit']['rstar_meas'])
        a.r_cat = a.r_cat or int(parts['cells']['cat']['metrics']['map10']['fit']['rstar_meas'])

    mu = X.mean(0)
    Xc = X - mu
    C = (Xc.T @ Xc) / (len(Xc) - 1)
    w, V = np.linalg.eigh(C)
    V = V[:, np.argsort(w)[::-1]][:, :a.r_item]
    Zi = Xc @ V
    del Xc, X
    Zi /= np.maximum(np.linalg.norm(Zi, axis=1, keepdims=True), 1e-9)
    Zc = np.ascontiguousarray(Zi[:, :a.r_cat])

    lab = np.array([r['ent'] for r in rows])
    cat = np.array([r['cat'] for r in rows])
    gi, gc = {}, {}
    for i, v in enumerate(lab):
        gi.setdefault(v, set()).add(i)
    for i, v in enumerate(cat):
        gc.setdefault(v, set()).add(i)
    uniq, inv, cnt = np.unique(lab, return_inverse=True, return_counts=True)
    rng = np.random.default_rng(0)
    q = np.sort(rng.choice(np.where(cnt[inv] > 1)[0], size=min(a.queries, len(uniq)), replace=False))

    scale = float(np.abs(Zi).max()) / 127.0
    Z8 = np.round(Zi / max(scale, 1e-12)).astype(np.int8)

    n = len(rows)
    res = {'enc': a.enc, 'ds': a.ds, 'n': n, 'r_item': a.r_item, 'r_cat': a.r_cat,
           'queries': int(len(q)), 'designs': {}}

    def record(name, ids, bytes_vec, build_s, t_search):
        res['designs'][name] = {
            'mAP_inst': ap10(ids, q, lab, gi), 'mAP_cat': ap10(ids, q, cat, gc),
            'bytes_per_vec': bytes_vec, 'total_mb': bytes_vec * n / 1e6,
            'build_s': build_s, 'qps': len(q) / max(t_search, 1e-9)}
        v = res['designs'][name]
        print('%-10s mAP %.4f / %.4f   %4d B/vec  %6.1f MB  build %5.0fs  %6.0f qps'
              % (name, v['mAP_inst'], v['mAP_cat'], bytes_vec, v['total_mb'], build_s, v['qps']))

    # ---- uniform float32
    t0 = time.time()
    idx = build_f32(Zi)
    t_build = time.time() - t0
    t0 = time.time()
    ids = strip(search(idx, Zi[q], 11), q, 10)
    record('umax-f32', ids, 4 * a.r_item + LINKS, t_build, time.time() - t0)
    del idx

    # ---- uniform int8 graph
    t0 = time.time()
    idx = build_i8(Zi)
    t_build = time.time() - t0
    t0 = time.time()
    ids = strip(search(idx, Zi[q], 11), q, 10)
    record('umax-i8', ids, a.r_item + LINKS, t_build, time.time() - t0)
    del idx

    # ---- shared, f32 coarse graph, int8 rerank store
    t0 = time.time()
    idx = build_f32(Zc)
    t_build = time.time() - t0
    t0 = time.time()
    ids = rerank_batch(Zc, Z8, q, idx, TOPK)
    record('share-f32', ids, 4 * a.r_cat + a.r_item + LINKS, t_build, time.time() - t0)
    del idx

    # ---- shared, int8 coarse graph, int8 rerank store
    t0 = time.time()
    idx = build_i8(Zc)
    t_build = time.time() - t0
    t0 = time.time()
    ids = rerank_batch(Zc, Z8, q, idx, TOPK)
    record('share-i8', ids, a.r_cat + a.r_item + LINKS, t_build, time.time() - t0)

    base = res['designs']['umax-f32']
    print('\nrelative to umax-f32:')
    for k, v in res['designs'].items():
        print('  %-10s bytes %.3fx  item %+.1f%%  cat %+.1f%%'
              % (k, v['bytes_per_vec'] / base['bytes_per_vec'],
                 100 * (v['mAP_inst'] / base['mAP_inst'] - 1),
                 100 * (v['mAP_cat'] / base['mAP_cat'] - 1)))

    if a.out:
        json.dump(res, open(a.out, 'w'), indent=1)
        print('wrote', a.out)


if __name__ == '__main__':
    main()
