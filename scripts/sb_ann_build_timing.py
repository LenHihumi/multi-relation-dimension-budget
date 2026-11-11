"""R1: is the HNSW build time at r=96 real, or a measurement artefact?

The paper reports build times of 0.5 / 16.3 / 25.3 / 43.0 s for ranks 96 / 192 /
384 / 768 on the same 52,591-vector gallery. A 32x jump between the first two is
large enough that a reviewer asked whether it is a warm-up or thread-count
artefact rather than a property of the graph.

This script re-measures it under a controlled protocol:

  * the OpenMP thread count is set explicitly before every build, and both 24 and
    1 are reported, because the original run interleaved builds with a QPS sweep
    that set the thread count to 1;
  * each rank is built four times: twice in ascending rank order and twice in
    descending order, so an order effect shows up as a difference between the
    two passes rather than being averaged away;
  * one warm-up build is discarded before any measurement.

Usage:
  python3 sb_ann_build_timing.py --out ann_build_timing.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sb_common                                                       # noqa: E402

if os.environ.get('SB_OUT'):
    sb_common.OUT = os.environ['SB_OUT']
if os.environ.get('SB_CACHE'):
    sb_common.SB = os.environ['SB_CACHE']

from sb_common import load_emb, pca_basis                               # noqa: E402

RANKS = (96, 192, 384, 768)


def log(m):
    print(m, flush=True)


def build(faiss, G, r, threads, efC=200, M=32):
    faiss.omp_set_num_threads(threads)
    idx = faiss.IndexHNSWFlat(r, M, faiss.METRIC_INNER_PRODUCT)
    idx.hnsw.efConstruction = efC
    t0 = time.time()
    idx.add(G)
    dt = time.time() - t0
    return dt, idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='ann_build_timing.json')
    ap.add_argument('--enc', default='clip_vitl14')
    ap.add_argument('--ds', default='inshop')
    args = ap.parse_args()

    import faiss
    log('faiss %s' % faiss.__version__)

    X = load_emb(args.enc, args.ds)
    mu, U, _ = pca_basis(X, device='cpu')
    log('gallery %s' % (X.shape,))
    G = {r: np.ascontiguousarray((X - mu) @ U[:r].T, dtype=np.float32) for r in RANKS}

    res = {'encoder': args.enc, 'dataset': args.ds, 'n_gallery': int(len(X)),
           'M': 32, 'efConstruction': 200, 'faiss': faiss.__version__,
           'ranks': {}}

    # discarded warm-up: first touch of the index code and of the projected data
    build(faiss, G[192], 192, 24)
    log('warm-up done')

    for threads in (24, 1):
        for order in ('ascending', 'descending'):
            seq = RANKS if order == 'ascending' else tuple(reversed(RANKS))
            for r in seq:
                dt, idx = build(faiss, G[r], r, threads)
                res['ranks'].setdefault(str(r), {}).setdefault(str(threads), {}).setdefault(order, []).append(round(dt, 2))
                log('threads=%2d %-10s r=%3d  %.1fs' % (threads, order, r, dt))
                del idx

    json.dump(res, open(os.path.join(sb_common.OUT, args.out), 'w'), indent=1)

    print()
    print('%-5s %-8s %-10s %-10s' % ('r', 'threads', 'ascending', 'descending'))
    for r in RANKS:
        for threads in (24, 1):
            a = res['ranks'][str(r)][str(threads)]['ascending']
            d = res['ranks'][str(r)][str(threads)]['descending']
            print('%-5d %-8d %-10s %-10s' % (r, threads, ['%.1f' % x for x in a],
                                             ['%.1f' % x for x in d]))


if __name__ == '__main__':
    main()
