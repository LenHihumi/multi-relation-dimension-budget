"""Does the truncation benefit survive scale, or is it a single operating point?

The manuscript reports one end-to-end HNSW cell (In-shop, 52K, CLIP).  A reviewer is right to read that as
"you won where you looked".  This sweeps the two axes that matter and reports the third:

  gallery size  n in {10k, 50k, 200k, 1M}   (random subsets of ImageNet-1.28M, fixed seed)
  rank          r in {96, 128, 160, 192, 256}  (around the measured ImageNet budgets, 128 / 160)
  beam          efSearch in {16, 32, 64, 128}  (one build per (n, r); ef is a search parameter)

Per point: build seconds, QPS at 32 threads, p50/p99 single-thread latency, mAP@10 for the class relation
and the WordNet-coarse relation, bytes per vector, and the share of those bytes that is the link list.

Two things this is meant to settle:

  1. The link cost is 2*M*4 = 256 B/vector *independent of n*, so the byte split between vectors and links
     is a function of the rank alone, not of scale.  Any claim that the fixed graph overhead "becomes
     relatively cheaper at scale" is arithmetically wrong; what changes with n is the number of distance
     evaluations per query (log n hops) and the build time (n log n).
  2. What *does* grow with n is the truncation speed-up, because a query pays for hops x r and the hop
     count grows while r does not.  If the speed-up ratio at 1M is larger than at 10k, truncation is worth
     more at scale; if it is flat, the benefit is a property of the rank alone.

Usage (inside the container):
  python3 -u sb_scale_curve.py --sizes 10000,50000,200000,1000000 \
      --ranks 96,128,160,192,256 --efs 16,32,64,128 --out scale_curve.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

from sb_paths import SB_ROOT, SB_OUT, SB_CACHE, SB_MODELS, REPO  # paths resolve relative to the deposit
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

M, EFC = 32, 200
SB = SB_CACHE
OUT = SB_OUT + '/spectral_budget'


def load_imagenet():
    X = np.load(os.path.join(SB, 'imagenet1m', 'emb.npy'), mmap_mode='r')
    lab = np.load(os.path.join(SB, 'imagenet1m', 'labels.npy'))
    meta = json.load(open(os.path.join(SB, 'imagenet1m', 'meta.json')))
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from imagenet_coarse import load_groups
    g = load_groups()
    coarse = np.array([g[int(i)] for i in lab])
    return X, lab, coarse, meta


def pca(X, k):
    mu = np.asarray(X).mean(0)
    d = X.shape[1]
    G = np.zeros((d, d))
    for s in range(0, len(X), 20000):
        B = np.asarray(X[s:s + 20000], dtype=np.float64) - mu
        G += B.T @ B
    G /= max(len(X) - 1, 1)
    w, V = np.linalg.eigh(G)
    o = np.argsort(w)[::-1]
    return mu, V[:, o][:, :k]


def groups_of(per_row, sel):
    g = {}
    for i in sel:
        g.setdefault(per_row[i], set()).add(int(i))
    return g


def map10(idx, Q, q, per_row, grp, k=11):
    _, ids = idx.search(np.ascontiguousarray(Q), k)
    aps = []
    for r, qi in enumerate(q):
        rel = grp[per_row[qi]] - {int(qi)}
        h, s = 0, 0.0
        for rank, j in enumerate(ids[r][:10], 1):
            if int(j) in rel:
                h += 1
                s += h / rank
        aps.append(s / min(len(rel), 10) if rel else 0.0)
    return float(np.mean(aps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sizes', default='10000,50000,200000,1000000')
    ap.add_argument('--ranks', default='96,128,160,192,256')
    ap.add_argument('--efs', default='16,32,64,128')
    ap.add_argument('--queries', type=int, default=200)
    ap.add_argument('--out', default=os.path.join(OUT, 'scale_curve.json'))
    a = ap.parse_args()
    sizes = [int(x) for x in a.sizes.split(',')]
    ranks = [int(x) for x in a.ranks.split(',')]
    efs = [int(x) for x in a.efs.split(',')]

    import faiss
    X, lab, coarse, meta = load_imagenet()
    nmax = min(len(X), max(sizes))
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(X))
    print('imagenet rows %d, using up to %d' % (len(X), nmax))

    mu, V = pca(X[perm[:nmax]], max(ranks))
    del X
    res = {'sizes': sizes, 'ranks': ranks, 'efs': efs, 'M': M, 'rows': []}
    t0 = time.time()
    for n in sizes:
        sel = np.sort(perm[:n])
        Z = (np.asarray(np.load(os.path.join(SB, 'imagenet1m', 'emb.npy'), mmap_mode='r')[sel],
                        dtype=np.float32) - mu) @ V
        Z /= np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9)
        per_inst = lab[sel]
        per_cat = coarse[sel]
        g_i = groups_of(per_inst, range(n))
        g_c = groups_of(per_cat, range(n))
        elig = np.array([i for i in range(n) if len(g_i[per_inst[i]]) > 1])
        q = np.sort(rng.choice(elig, size=min(a.queries, len(elig)), replace=False))
        for r in ranks:
            Zr = np.ascontiguousarray(Z[:, :r])
            faiss.omp_set_num_threads(48)
            idx = faiss.IndexHNSWFlat(r, M)
            idx.hnsw.efConstruction = EFC
            tb = time.time()
            idx.add(Zr)
            tb = time.time() - tb
            Q = np.ascontiguousarray(Zr[q])
            for ef in efs:
                idx.hnsw.efSearch = ef
                faiss.omp_set_num_threads(1)
                ts = []
                for i in range(len(Q)):
                    t = time.perf_counter()
                    idx.search(np.ascontiguousarray(Q[i:i + 1]), 11)
                    ts.append((time.perf_counter() - t) * 1e3)
                ts = np.array(ts)
                # throughput needs a warm-up and repeats: the first batch after a thread-pool switch
                # pays the pool start-up, and with a fast index that dominates the measurement
                faiss.omp_set_num_threads(32)
                idx.search(np.ascontiguousarray(Q), 11)
                best = float('inf')
                for _ in range(3):
                    t = time.time()
                    idx.search(np.ascontiguousarray(Q), 11)
                    best = min(best, time.time() - t)
                qps = len(Q) / max(best, 1e-9)
                faiss.omp_set_num_threads(1)
                p50 = float(np.median(ts))
                row = dict(n=int(n), r=int(r), ef=int(ef), build_s=round(tb, 1),
                           qps=round(qps, 0), qps_1t=round(1000.0 / max(p50, 1e-6), 0),
                           p50_ms=round(p50, 3),
                           p99_ms=round(float(np.percentile(ts, 99)), 3),
                           map_inst=round(map10(idx, Q, q, per_inst, g_i), 4),
                           map_cat=round(map10(idx, Q, q, per_cat, g_c), 4),
                           bytes_vec=int(4 * r + 2 * M * 4),
                           links_share=round(2 * M * 4 / (4 * r + 2 * M * 4), 3))
                res['rows'].append(row)
                print('n=%-8d r=%-3d ef=%-3d build %6.1fs  QPS %7.0f  p50 %.3f  mAP %.4f/%.4f'
                      % (n, r, ef, tb, qps, row['p50_ms'], row['map_inst'], row['map_cat']),
                      flush=True)
            del idx, Zr, Q
        del Z
        print('--- n=%d done at %.0fs' % (n, time.time() - t0), flush=True)

    # speed-up of truncation as a function of n, at the widest beam.  Uses the single-thread latency,
    # because multi-thread throughput on a shared machine measures the neighbours as much as the index.
    ef = max(efs)
    by = {}
    for row in res['rows']:
        if row['ef'] == ef:
            by.setdefault(row['n'], {})[row['r']] = row['qps_1t']
    res['speedup'] = {str(n): (by[n].get(min(ranks), 0) / by[n].get(max(ranks), 1)) for n in by}
    print('truncation speed-up QPS(r=%d)/QPS(r=%d) at ef=%d: %s'
          % (min(ranks), max(ranks), ef, res['speedup']))
    json.dump(res, open(a.out, 'w'), indent=1)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
