"""Shared multi-relation index: is there a design that beats uniform-max-rank on bytes?

Question (see SHARED_INDEX_PLAN.md): the paper says item retrieval needs rank 160 and category
retrieval rank 48 on the same basis.  A deployment usually builds ONE index at the larger rank
("uniform max").  Can a shared design beat it on total bytes at <=5% mAP@10 loss?

Designs, all on the same PCA basis:

  indep      two truncations, two HNSW graphs        (the per-relation baseline)
  umax       one HNSW graph at r_max, both relations (the baseline to beat)
  share-f32  graph at r_cat, top-k rerank with f32 r_max vectors
  share-i8   same, rerank store quantised to int8 (symmetric per-dimension scale)

Bytes are counted from the actual components: vectors (n*d*bits), links (n*2*M*4 for faiss HNSW),
and the PCA basis (d*r*4).  mAP@10 uses binary relevance with the paper's normalisation
(AP cut at 10, divided by min(|R|,10)).

Usage (inside the container, with the project mounted at $SB_ROOT):
  python3 shared_index.py --enc clip_vitl14 --ds inshop --n 10000 --q 200
  python3 shared_index.py --enc clip_vitl14 --ds inshop            # full 52,591 / 1,000
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from sb_paths import SB_ROOT, SB_OUT, SB_CACHE, SB_MODELS, REPO  # paths resolve relative to the deposit
CACHE = SB_CACHE
DUMP = SB_OUT + '/spectral_budget'
M = 32
EFC = 200
EF = 128
TOPK = 100


def load_corpus(enc, ds):
    base = os.path.join(DUMP, 'emb', '%s__%s' % (enc, ds))
    meta = json.load(open(base + '.json'))
    if os.path.exists(base + '.npy'):
        emb = np.load(base + '.npy', mmap_mode='r')
    else:                                   # raw float32 dump, shape from the sidecar
        emb = np.memmap(base + '.bin', dtype=np.float32, mode='r').reshape(meta['n'], meta['dim'])
    rows = [json.loads(l) for l in open(os.path.join(CACHE, ds, 'gallery.jsonl'))]
    return emb, meta, rows


def labels(rows, rel):
    """Return (label per row, label -> set of rows) for the given relation key."""
    key = 'ent' if rel in ('inst', 'ent') else 'cat'
    per_row = [r[key] for r in rows]
    groups = {}
    for i, lab in enumerate(per_row):
        groups.setdefault(lab, set()).add(i)
    return per_row, groups


def drop_self(ids, q):
    """Remove the query row from its own result list.

    Every index returns the query as its nearest neighbour, and relevance for a query excludes it.  Left
    in, it occupies rank 1 as a guaranteed non-relevant hit, which caps AP@10 at sum_{r=2..|R|+1}(r-1)/r
    /|R| -- 0.69 for a four-image group.  All designs are affected equally, so ratios survive, but the
    absolute numbers would be wrong and the constraint would bind hardest on the smallest groups.
    """
    rows = []
    for r, qi in enumerate(q):
        row = [int(j) for j in ids[r] if j != qi]
        if len(row) < ids.shape[1]:                     # searched k+1, so one row loses exactly one
            row += [row[-1]] * (ids.shape[1] - len(row))
        rows.append(row[:ids.shape[1]])
    return np.array(rows, dtype=ids.dtype)


def pca(X, r):
    mu = X.mean(0)
    Xc = X - mu
    C = (Xc.T @ Xc) / max(len(Xc) - 1, 1)
    w, V = np.linalg.eigh(C)
    idx = np.argsort(w)[::-1][:r]
    return mu, V[:, idx]


def search_faiss(Xb, Xq, k, ef=EF, threads=1, build_threads=32):
    """Build with many threads, search with one.

    The build is O(n log n d) and dominates the wall clock at 10^6 rows; the search timing is what the
    QPS number means, and a single thread is what a per-query latency corresponds to.  Doing both with
    `threads` was the reason a 1.3M-row run spent half an hour in graph construction.
    """
    import faiss
    d = Xb.shape[1]
    index = faiss.IndexHNSWFlat(d, M)
    index.hnsw.efConstruction = EFC
    faiss.omp_set_num_threads(build_threads)
    t0 = time.time()
    index.add(np.ascontiguousarray(Xb.astype(np.float32)))
    build = time.time() - t0
    faiss.omp_set_num_threads(threads)
    index.hnsw.efSearch = ef
    t0 = time.time()
    _, idx = index.search(np.ascontiguousarray(Xq.astype(np.float32)), k)
    qps = len(Xq) / max(time.time() - t0, 1e-9)
    return idx, build, qps


def exact_topk(Xb, Xq, k, block=2000):
    out = np.empty((len(Xq), k), dtype=np.int32)
    B = Xb / np.maximum(np.linalg.norm(Xb, axis=1, keepdims=True), 1e-9)
    for s in range(0, len(Xq), block):
        Q = Xq[s:s + block]
        Q = Q / np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-9)
        S = Q @ B.T
        out[s:s + block] = np.argpartition(-S, k, axis=1)[:, :k]
    return out


def ap10(ids, qidx, per_row, groups, k=10):
    """binary-relevance AP@10, normalised by min(|R|,10), self excluded."""
    aps = []
    for row, qi in enumerate(qidx):
        R = groups.get(per_row[qi], set()) - {qi}
        if not R:
            continue
        hits = 0
        ap = 0.0
        for rank, j in enumerate(ids[row][:k], start=1):
            if j in R:
                hits += 1
                ap += hits / rank
        aps.append(ap / min(len(R), k))
    return float(np.mean(aps)) if aps else float('nan')


def bytes_of(n, dims, rerank_bytes_per_vec=0, n_graphs=1, d_full=None, r_full=None):
    vec = sum(n * d * 4 for d in dims)
    links = n_graphs * n * 2 * M * 4
    basis = (d_full * r_full * 4) if d_full else 0
    return {'vectors': vec, 'links': links, 'basis': basis,
            'rerank': n * rerank_bytes_per_vec,
            'total': vec + links + basis + n * rerank_bytes_per_vec}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--enc', default='clip_vitl14')
    ap.add_argument('--ds', default='inshop')
    ap.add_argument('--n', type=int, default=0)
    ap.add_argument('--q', type=int, default=0)
    ap.add_argument('--r-cat', type=int, default=0, help='0 = take the measured r* for this cell')
    ap.add_argument('--r-item', type=int, default=0)
    ap.add_argument('--out', default='')
    a = ap.parse_args()

    if not a.r_cat or not a.r_item:        # use the paper's own measured budgets
        parts = json.load(open(os.path.join(DUMP, 'metric_std_parts',
                                            '%s__%s.json' % (a.enc, a.ds))))
        a.r_item = a.r_item or int(parts['cells']['inst']['metrics']['map10']['fit']['rstar_meas'])
        a.r_cat = a.r_cat or int(parts['cells']['cat']['metrics']['map10']['fit']['rstar_meas'])
    emb, meta, rows = load_corpus(a.enc, a.ds)
    n = a.n or emb.shape[0]
    X = np.asarray(emb[:n], dtype=np.float32)
    rows = rows[:n]
    qs = json.load(open(os.path.join(CACHE, a.ds, 'queries.json')))
    # the same query rows serve both relations (category/cat1/catf/inst are label partitions
    # over one query set), so only the relevance differs
    q_inst = [int(i) for i in qs['inst'] if int(i) < n][: (a.q or None)]
    q_cat = list(q_inst)
    print('n=%d dim=%d  queries inst=%d cat=%d' % (n, X.shape[1], len(q_inst), len(q_cat)))

    lab_inst, rel_inst = labels(rows, 'inst')
    lab_cat, rel_cat = labels(rows, 'cat')

    mu, V = pca(X, max(a.r_item, a.r_cat))
    Xc = (X - mu) @ V                      # full-width projection, prefix = truncation
    ev = np.sort(np.linalg.eigvalsh(np.cov(X.T.astype(np.float64))))[::-1]
    print('pca done, energy in top-%d = %.4f of total'
          % (a.r_item, float(ev[:a.r_item].sum() / ev.sum())))

    res = {}
    t0 = time.time()

    # ---- umax: one graph at r_item, both relations
    # Normalise before indexing.  faiss IndexHNSWFlat defaults to L2, and on unnormalised projections L2
    # and cosine disagree: the comparison between designs would then be a comparison between metrics.
    Xi = Xc[:, :a.r_item]
    Xi = Xi / np.maximum(np.linalg.norm(Xi, axis=1, keepdims=True), 1e-9)
    idx_i, b_i, qps_i = search_faiss(Xi, Xi[q_inst], 11)
    idx_c_at_item, _, qps_c_i = search_faiss(Xi, Xi[q_cat], 11)
    idx_i = drop_self(idx_i, q_inst)[:, :10]
    idx_c_at_item = drop_self(idx_c_at_item, q_cat)[:, :10]
    res['umax'] = {'mAP_inst': ap10(idx_i, q_inst, lab_inst, rel_inst),
                   'mAP_cat': ap10(idx_c_at_item, q_cat, lab_cat, rel_cat),
                   'bytes': bytes_of(n, [a.r_item], 0, 1, X.shape[1], a.r_item),
                   'build_s': b_i, 'qps': qps_i, 'qps_e2e': qps_i}

    # ---- indep: two graphs
    Xc_cat = np.ascontiguousarray(Xi[:, :a.r_cat])      # prefix of the same normalised vectors
    idx_c, b_c, qps_c = search_faiss(Xc_cat, Xc_cat[q_cat], 11)
    idx_c = drop_self(idx_c, q_cat)[:, :10]
    res['indep'] = {'mAP_inst': res['umax']['mAP_inst'],
                    'mAP_cat': ap10(idx_c, q_cat, lab_cat, rel_cat),
                    'bytes': bytes_of(n, [a.r_item, a.r_cat], 0, 2, X.shape[1], a.r_item),
                    'build_s': b_i + b_c, 'qps': min(qps_i, qps_c),
                    'qps_e2e': min(qps_i, qps_c)}

    # ---- share: graph at r_cat, rerank top-k at r_item
    cand, _, _ = search_faiss(Xc_cat, Xc_cat[q_inst], TOPK + 1)
    cand = drop_self(cand, q_inst)                      # candidates are scored, so self must go
    Xi_n = Xi / np.maximum(np.linalg.norm(Xi, axis=1, keepdims=True), 1e-9)
    rer = np.empty((len(q_inst), 10), dtype=np.int32)
    t_r = time.time()
    for r, qi in enumerate(q_inst):
        c = cand[r]
        s = Xi_n[c] @ Xi_n[qi]
        rer[r] = c[np.argsort(-s)[:10]]
    t_r32 = (time.time() - t_r) / max(len(q_inst), 1)      # seconds per query, single thread
    res['share-f32'] = {'mAP_inst': ap10(rer, q_inst, lab_inst, rel_inst),
                        'mAP_cat': ap10(cand[:, :10], q_cat, lab_cat, rel_cat),
                        'bytes': bytes_of(n, [a.r_cat], 4 * a.r_item, 1, X.shape[1], a.r_cat),
                        'build_s': b_c, 'qps': qps_c,
                        'qps_e2e': 1.0 / (1.0 / qps_c + t_r32)}

    # ---- share with int8 rerank store
    # ONE scale for the whole tensor: per-dimension scales reweight the dot product (dimensions
    # with a small dynamic range get amplified), which turns "quantisation" into a different
    # metric.  A per-tensor scale keeps int8 a lossy copy of cosine, which is what we want to
    # measure.  (A per-dim scale was tried first and *raised* mAP by 4 points -- a metric change
    # masquerading as a compression result.)
    scale = float(np.abs(Xi_n).max()) / 127.0
    Xi_i8 = np.round(Xi_n / np.maximum(scale, 1e-12)).astype(np.int8)
    rer8 = np.empty((len(q_inst), 10), dtype=np.int32)
    t_r = time.time()
    for r, qi in enumerate(q_inst):
        c = cand[r]
        # int8 matmul accumulates in int8 and overflows; sum in int32 instead
        s = (Xi_i8[c].astype(np.int32) @ Xi_i8[qi].astype(np.int32)).astype(np.float32)
        rer8[r] = c[np.argsort(-s)[:10]]
    t_r8 = (time.time() - t_r) / max(len(q_inst), 1)
    res['share-i8'] = {'mAP_inst': ap10(rer8, q_inst, lab_inst, rel_inst),
                       'mAP_cat': res['share-f32']['mAP_cat'],
                       'bytes': bytes_of(n, [a.r_cat], a.r_item, 1, X.shape[1], a.r_cat),
                       'build_s': b_c, 'qps': qps_c,
                       'qps_e2e': 1.0 / (1.0 / qps_c + t_r8)}

    base = res['umax']['bytes']['total']
    print('\n%-10s %8s %8s %10s %8s %8s %10s %8s'
          % ('design', 'mAP_inst', 'mAP_cat', 'MB', 'vs umax', 'build_s', 'QPS e2e', 'vs umax'))
    q_base = res['umax']['qps_e2e']
    for k, v in res.items():
        print('%-10s %8.4f %8.4f %10.1f %8.3f %8.1f %10.0f %7.2fx'
              % (k, v['mAP_inst'], v['mAP_cat'], v['bytes']['total'] / 1e6,
                 v['bytes']['total'] / base, v['build_s'], v['qps_e2e'],
                 v['qps_e2e'] / q_base))
    res['meta'] = {'n': n, 'enc': a.enc, 'ds': a.ds, 'r_cat': a.r_cat, 'r_item': a.r_item,
                   'M': M, 'efC': EFC, 'ef': EF, 'topk_rerank': TOPK,
                   'wall_s': time.time() - t0}
    print('\nwall %.1fs' % res['meta']['wall_s'])
    if a.out:
        json.dump(res, open(a.out, 'w'), indent=1)
        print('wrote', a.out)


if __name__ == '__main__':
    main()
