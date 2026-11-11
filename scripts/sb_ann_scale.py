"""Does graph approximation stay separable from truncation as the gallery grows?

Sec. V-E claims the HNSW top-10 is identical to exact search at efSearch>=64,
and explicitly bounds that claim to 52,591 vectors. This script measures how the
overlap decays with gallery size at a fixed efSearch, which is the quantity the
claim depends on.

Protocol: one fixed query set of 1,000 rows, and nested galleries of 13k, 26k and
52k rows (13k subset of 26k subset of 52k, fixed seed), all projected onto the
same PCA basis of the full corpus at rank 192. Ground truth is exact top-10 in
that same truncated space, self-match excluded. HNSW is M=32,
efConstruction=200, search at efSearch 16/32/64/128. No labels are involved: the
quantity is agreement with exact search, not retrieval quality.

Usage:
  python3 sb_ann_scale.py --out ann_scale.json
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
from sb_ann import score, topk_idx                                      # noqa: E402

SIZES = (13000, 26000, 52591)
EFS = (16, 32, 64, 128)
NQ = 1000
R = 192
SEED = 20261106


def log(m):
    print(m, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='ann_scale.json')
    ap.add_argument('--enc', default='clip_vitl14')
    ap.add_argument('--ds', default='inshop')
    args = ap.parse_args()

    import faiss
    log('faiss %s' % faiss.__version__)

    X = load_emb(args.enc, args.ds)
    mu, U, _ = pca_basis(X, device='cpu')
    Gp = np.ascontiguousarray((X - mu) @ U[:R].T, dtype=np.float32)
    log('gallery %s projected to r=%d' % (X.shape, R))

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(X))
    q_idx = perm[:NQ]
    pool = perm[NQ:]

    res = {'encoder': args.enc, 'dataset': args.ds, 'rank': R, 'n_query': NQ,
           'efs': list(EFS), 'M': 32, 'efConstruction': 200,
           'faiss': faiss.__version__, 'sizes': {}}

    prev = []
    for N in SIZES:
        if N > len(X):
            continue
        add = pool[:N - NQ] if N - NQ <= len(pool) else pool
        gal = np.concatenate([q_idx, add])
        pos = {g: i for i, g in enumerate(gal)}
        qpos = np.array([pos[q] for q in q_idx])

        G = Gp[gal]
        Q = Gp[q_idx]

        # exact top-10, self excluded
        t0 = time.time()
        exact = []
        CH = 256
        for i0 in range(0, len(Q), CH):
            S = score(Q[i0:i0 + CH], G)
            S[np.arange(S.shape[0]), qpos[i0:i0 + CH]] = -np.inf
            exact.append(topk_idx(S, 10))
            del S
        exact = np.vstack(exact)
        t_exact = time.time() - t0

        t0 = time.time()
        index = faiss.IndexHNSWFlat(R, 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 200
        index.add(G)
        t_build = time.time() - t0

        row = {'n_gallery': int(len(gal)), 'build_s': round(t_build, 1),
               'exact_s': round(t_exact, 1), 'ef': {}}
        for ef in EFS:
            index.hnsw.efSearch = ef
            faiss.omp_set_num_threads(1)
            # queries are gallery rows, so the index returns the query itself.
            # Ask for one extra neighbour and drop the self-match, exactly as the
            # exact arm does by masking its diagonal.
            D, I = index.search(Q, 11)
            rec = 0.0
            for i in range(len(Q)):
                got = [int(x) for x in I[i].tolist() if int(x) != int(qpos[i])][:10]
                rec += len(set(got) & set(exact[i].tolist())) / 10.0
            rec /= len(Q)
            row['ef'][str(ef)] = round(rec, 4)
        res['sizes'][str(len(gal))] = row
        log('N=%6d build %5.1fs  ' % (len(gal), t_build) +
            '  '.join('ef%d=%.4f' % (ef, row['ef'][str(ef)]) for ef in EFS))
        prev = gal

    json.dump(res, open(os.path.join(sb_common.OUT, args.out), 'w'), indent=1)


if __name__ == '__main__':
    main()
