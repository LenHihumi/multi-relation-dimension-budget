"""Latency and resident memory for the shared index on a real corpus (not synthetic vectors).

The scaling study measured build/QPS/p99 on Gaussian vectors, which is enough for how cost grows with n
but says nothing about the distribution real embeddings produce.  This measures, on the 1.28M-row
ImageNet embedding:

  umax   one graph at r_item                -> per-query p50/p99, RSS
  share  graph at r_cat + int8 rerank@r_item -> per-query p50/p99, RSS
  umin   one graph at r_cat for both relations -> quality only (the baseline sharing must beat)

Latency is single-threaded per query (what a p99 means for a serving loop); the build is multi-threaded
because it dominates wall clock and is not what the number describes.  RSS is read after the corpus
arrays are released, so it is the index's footprint plus the interpreter, not the corpus.
"""
import argparse
import gc
import json
import os
import time

import numpy as np

from sb_paths import SB_ROOT, SB_OUT, SB_CACHE, SB_MODELS, REPO  # paths resolve relative to the deposit
M = 32
EFC = 200
EF = 128
TOPK = 100
CACHE = os.environ.get('CACHE', SB_CACHE)
DUMP = os.environ.get('DUMP', SB_OUT + '/spectral_budget')


def rss_mb():
    for line in open('/proc/self/status'):
        if line.startswith('VmRSS:'):
            return float(line.split()[1]) / 1024.0
    return float('nan')


def rss_delta(before_mb):
    return rss_mb() - before_mb


def build(Z, threads=48):
    import faiss
    faiss.omp_set_num_threads(threads)
    idx = faiss.IndexHNSWFlat(Z.shape[1], M)
    idx.hnsw.efConstruction = EFC
    t0 = time.time()
    idx.add(np.ascontiguousarray(Z.astype(np.float32)))
    idx.hnsw.efSearch = EF
    faiss.omp_set_num_threads(1)
    return idx, time.time() - t0


def latencies(idx, Q, qrows, k=10):
    """Per-query latency, single thread, one query at a time (this is what a p99 describes).

    Self is dropped from the returned list: relevance excludes the query row, and a self hit occupies
    rank 1 as a guaranteed non-relevant item (capping AP@10 at 0.69 for a four-image group).
    """
    out, ids = [], []
    for q, qi in zip(Q, qrows):
        t0 = time.time()
        _, cand = idx.search(np.ascontiguousarray(q[None].astype(np.float32)), k + 1)
        out.append((time.time() - t0) * 1e3)
        ids.append([int(j) for j in cand[0] if int(j) != int(qi)][:k])
    return np.array(out), np.array(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=os.path.join(CACHE, 'imagenet1m'))
    ap.add_argument('--ds', default='imagenet',
                    help='query set to reuse from $CACHE/<ds>/queries.json, so the latency table and '
                         'the quality table describe the same queries')
    ap.add_argument('--r-item', type=int, default=160)
    ap.add_argument('--r-cat', type=int, default=128)
    ap.add_argument('--queries', type=int, default=300)
    ap.add_argument('--out', default='')
    a = ap.parse_args()

    emb = np.load(os.path.join(a.src, 'emb.npy'), mmap_mode='r')
    lab = np.load(os.path.join(a.src, 'labels.npy'))
    meta = json.load(open(os.path.join(a.src, 'meta.json')))
    X = np.asarray(emb, dtype=np.float32)
    n = len(X)
    print('rows %d dim %d' % (n, X.shape[1]))

    mu = X.mean(0)
    Xc = X - mu
    C = (Xc.T @ Xc) / (n - 1)
    w, V = np.linalg.eigh(C)
    V = V[:, np.argsort(w)[::-1]][:, :a.r_item]
    Zi = Xc @ V
    del Xc, X
    gc.collect()
    Zi /= np.maximum(np.linalg.norm(Zi, axis=1, keepdims=True), 1e-9)
    Zc = np.ascontiguousarray(Zi[:, :a.r_cat])

    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from imagenet_coarse import load_groups
    idx2group = load_groups()
    coarse = np.array([idx2group[int(i)] for i in lab])

    qp = os.path.join(CACHE, a.ds, 'queries.json')
    if os.path.exists(qp):
        q = np.array(json.load(open(qp))['inst'][:a.queries], dtype=np.int64)
        print('reusing %d queries from %s' % (len(q), qp))
    else:
        rng = np.random.default_rng(0)
        uniq, inv, cnt = np.unique(lab, return_inverse=True, return_counts=True)
        q = np.sort(rng.choice(np.where(cnt[inv] > 1)[0], size=min(a.queries, n), replace=False))

    def ap10(ids, per_row, groups):
        aps = []
        for r, qi in enumerate(q):
            rel = groups[per_row[qi]] - {qi}
            h, ap = 0, 0.0
            for rank, j in enumerate(ids[r][:10], 1):
                if j in rel:
                    h += 1
                    ap += h / rank
            aps.append(ap / min(len(rel), 10))
        return float(np.mean(aps))

    g_inst, g_cat = {}, {}
    for i, v in enumerate(lab):
        g_inst.setdefault(v, set()).add(i)
    for i, v in enumerate(coarse):
        g_cat.setdefault(v, set()).add(i)

    res = {'n': n, 'r_item': a.r_item, 'r_cat': a.r_cat, 'queries': int(len(q))}

    # ---- umax
    rss0 = rss_mb()
    idx_i, b_i = build(Zi)
    d_i = rss_delta(rss0)
    lat_i, ids_i = latencies(idx_i, Zi[q], q)
    del idx_i
    gc.collect()
    res['umax'] = {'build_s': b_i, 'p50_ms': float(np.percentile(lat_i, 50)),
                   'p99_ms': float(np.percentile(lat_i, 99)),
                   'qps': float(1000.0 / lat_i.mean()), 'rss_delta_mb': d_i,
                   'bytes_mb': 4.0 * 160 * n / 1e6 + 2 * 32 * 4 * n / 1e6,
                   'mAP_inst': ap10(ids_i, lab, g_inst), 'mAP_cat': ap10(ids_i, coarse, g_cat)}
    print('umax   build %5.1fs  p50 %.2f ms  p99 %.2f ms  RSS +%.0f MB  mAP %.4f/%.4f'
          % (b_i, res['umax']['p50_ms'], res['umax']['p99_ms'], res['umax']['rss_delta_mb'],
             res['umax']['mAP_inst'], res['umax']['mAP_cat']))

    # ---- umin: the cheap baseline both relations at the coarse budget
    rss0 = rss_mb()
    idx_c, b_c = build(Zc)
    d_c = rss_delta(rss0)
    lat_c, ids_c = latencies(idx_c, Zc[q], q)
    res['umin'] = {'build_s': b_c, 'p50_ms': float(np.percentile(lat_c, 50)),
                   'p99_ms': float(np.percentile(lat_c, 99)),
                   'qps': float(1000.0 / lat_c.mean()), 'rss_delta_mb': d_c,
                   'bytes_mb': 4.0 * a.r_cat * n / 1e6 + 2 * 32 * 4 * n / 1e6,
                   'mAP_inst': ap10(ids_c, lab, g_inst), 'mAP_cat': ap10(ids_c, coarse, g_cat)}
    print('umin   build %5.1fs  p50 %.2f ms  p99 %.2f ms  RSS +%.0f MB  mAP %.4f/%.4f'
          % (b_c, res['umin']['p50_ms'], res['umin']['p99_ms'], res['umin']['rss_delta_mb'],
             res['umin']['mAP_inst'], res['umin']['mAP_cat']))
    del idx_c
    gc.collect()

    # ---- share: coarse graph + int8 rerank store, one scale for the whole tensor
    scale = float(np.abs(Zi).max()) / 127.0
    Z8 = np.round(Zi / max(scale, 1e-12)).astype(np.int8)
    idx_s, _ = build(Zc)
    lat_s, ids_s = [], []
    for qi in q:
        t0 = time.time()
        _, cand = idx_s.search(np.ascontiguousarray(Zc[qi][None]), TOPK + 1)
        c = np.array([int(j) for j in cand[0] if int(j) != int(qi)][:TOPK])
        sims = Z8[c].astype(np.int32) @ Z8[qi].astype(np.int32)
        top = c[np.argsort(-sims)[:10]]
        lat_s.append((time.time() - t0) * 1e3)
        ids_s.append(top)
    lat_s = np.array(lat_s)
    ids_s = np.array(ids_s)
    res['share'] = {'build_s': b_c, 'p50_ms': float(np.percentile(lat_s, 50)),
                    'p99_ms': float(np.percentile(lat_s, 99)),
                    'qps': float(1000.0 / lat_s.mean()), 'rss_delta_mb': d_c,
                    'bytes_mb': (4.0 * a.r_cat + a.r_item) * n / 1e6 + 2 * 32 * 4 * n / 1e6,
                    'mAP_inst': ap10(ids_s, lab, g_inst), 'mAP_cat': ap10(ids_s, coarse, g_cat)}
    print('share  build %5.1fs  p50 %.2f ms  p99 %.2f ms  RSS +%.0f MB  mAP %.4f/%.4f'
          % (b_c, res['share']['p50_ms'], res['share']['p99_ms'], res['share']['rss_delta_mb'],
             res['share']['mAP_inst'], res['share']['mAP_cat']))

    if a.out:
        json.dump(res, open(a.out, 'w'), indent=1)
        print('wrote', a.out)


if __name__ == '__main__':
    main()
