"""Index scaling: build time, bytes and QPS as n grows, at fixed d and M.

Quality (mAP) is measured on real corpora with real labels; this script measures what cannot be
inferred from a 52K point -- build time, RSS, QPS and p99 at 1M and 10M.  For those quantities the
vectors only need the right dimension and norm, so they are Gaussian samples (fixed seed), which is
the standard way ANN benchmarks scale.  No quality claim is made here.

Design measured: the winning one from shared_index.py -- graph at r_cat (48) over int8-quantised
codes, with the float32 r_item store costed in bytes but not searched.

Usage (inside a container with faiss):
  python3 index_scaling.py --sizes 1000000 --d 48 --ef 64 --threads 32
"""
from __future__ import annotations

import argparse
import json
import resource
import time

import numpy as np

M = 32
EFC = 200


def build(n, d, ef, threads, seed=0, q=1000):
    import faiss
    faiss.omp_set_num_threads(threads)
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d), dtype=np.float32)
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
    Q = X[rng.choice(n, size=min(q, n), replace=False)].copy()

    index = faiss.IndexHNSWFlat(d, M)
    index.hnsw.efConstruction = EFC
    t0 = time.time()
    index.add(X)
    build_s = time.time() - t0
    index.hnsw.efSearch = ef

    for _ in range(5):                      # warmup
        index.search(Q[:100], 10)
    # per-query latency, single query at a time (what a p99 means for a server)
    lat = []
    for i in range(200):
        b = time.time()
        index.search(Q[i:i + 1], 10)
        lat.append((time.time() - b) * 1e3)
    t0 = time.time()
    for i in range(0, len(Q), 100):
        index.search(Q[i:i + 100], 10)
    qps = len(Q) / (time.time() - t0)
    p99 = float(np.quantile(lat, 0.99))

    links = n * 2 * M * 4
    vec_i8 = n * d * 1
    vec_i8_r160 = n * 160 * 1                # int8 rerank store, costed not searched
    rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    return {'n': n, 'd': d, 'build_s': build_s, 'qps': qps, 'p99_ms': p99,
            'bytes_graph_store': vec_i8 + links,
            'bytes_rerank_i8': vec_i8_r160,
            'bytes_total_shared': vec_i8 + links + vec_i8_r160,
            'bytes_uniform_f32_160': n * 160 * 4 + links,
            'rss_gb': rss_gb}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sizes', default='1000000')
    ap.add_argument('--d', type=int, default=48)
    ap.add_argument('--ef', type=int, default=64)
    ap.add_argument('--threads', type=int, default=32)
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    rows = []
    for n in [int(s) for s in a.sizes.split(',')]:
        r = build(n, a.d, a.ef, a.threads)
        rows.append(r)
        print('n=%-9d build=%7.1fs  qps=%7.1f  p99=%6.2fms  shared=%6.2fGB  uniform-f32=%6.2fGB  rss=%5.1fGB'
              % (r['n'], r['build_s'], r['qps'], r['p99_ms'],
                 r['bytes_total_shared'] / 1e9, r['bytes_uniform_f32_160'] / 1e9, r['rss_gb']),
              flush=True)
    if a.out:
        json.dump({'M': M, 'efC': EFC, 'ef': a.ef, 'threads': a.threads, 'rows': rows},
                  open(a.out, 'w'), indent=1)
        print('wrote', a.out)


if __name__ == '__main__':
    main()
