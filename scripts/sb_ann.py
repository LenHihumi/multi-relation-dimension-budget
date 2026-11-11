"""Does the rank budget survive a real ANN index (HNSW)?

The paper reports brute-force latency only.  This script closes that gap on one
cell (CLIP ViT-L/14, In-shop, item retrieval): build an HNSW graph on the
centred-PCA prefix at rank r, then measure QPS, memory and end-to-end mAP@10
against the gold relevance lists.

Two kinds of "recall" are kept apart on purpose:

  ann_recall_vs_exact  overlap with exact search IN THE SAME truncated space.
                       This is the graph's own approximation quality.
  ann_map10            mAP@10 of the ANN result against the gold labels.  This
                       is what a user feels, and it also includes the truncation
                       loss, so it is the number that can be compared across
                       ranks.

Usage: python3 sb_ann.py --enc clip_vitl14 --ds inshop --rel inst
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from sb_common import (OUT, load_emb, load_gallery, load_queries, l2n,
                       pca_basis, score, topk_idx, transform)
from sb_grid import build_cells

RANKS = (96, 192, 384, 768)
EFS = (16, 32, 64, 128, 256)


def log(m):
    print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)


def ap_at_k(top, rel, k=10):
    """AP@10 with cutoff min(|R|, 10), averaged over queries."""
    vals = []
    for t, r in zip(top, rel):
        if not r:
            continue
        rset = set(r)
        denom = min(len(rset), k)
        hits, acc = 0, 0.0
        for j, idx in enumerate(t[:k]):
            if int(idx) in rset:
                hits += 1
                acc += hits / (j + 1)
        vals.append(acc / denom)
    return float(np.mean(vals)) if vals else float('nan')


def _search_no_self(index, Q, self_ids, k):
    """Top-k from HNSW after removing each query's own gallery id."""
    D, I = index.search(Q, k + 1)
    out = np.empty((Q.shape[0], k), dtype=np.int64)
    for i in range(Q.shape[0]):
        row = I[i][I[i] != self_ids[i]][:k]
        if len(row) < k:                     # pad if the self hit was outside
            extra = [x for x in I[i] if x not in set(row.tolist())][:k - len(row)]
            row = np.concatenate([row, np.array(extra, dtype=np.int64)])
        out[i] = row[:k]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--enc', default='clip_vitl14')
    ap.add_argument('--ds', default='inshop')
    ap.add_argument('--rel', default='inst')
    ap.add_argument('--nq', type=int, default=0, help='0 = all queries')
    ap.add_argument('--out', default='ann_hnsw.json')
    args = ap.parse_args()

    import faiss
    log(f'faiss {faiss.__version__}')

    meta = load_gallery(args.ds)
    queries = load_queries(args.ds)
    X = load_emb(args.enc, args.ds)
    keep = np.arange(len(meta))
    cells = build_cells(args.ds, meta, queries, keep)
    cell = cells[args.rel]
    rows = np.array(cell['rows'], int) if args.nq <= 0 else np.array(cell['rows'][:args.nq], int)
    rel = cell['rel'] if args.nq <= 0 else cell['rel'][:args.nq]
    Q = X[rows]
    log(f'{args.enc}/{args.ds}/{args.rel}: gallery {X.shape} queries {Q.shape}')

    mu, U, ev = pca_basis(X, device='cpu')
    full_map10 = None
    res = {'encoder': args.enc, 'dataset': args.ds, 'relation': args.rel,
           'n_gallery': int(X.shape[0]), 'n_query': int(Q.shape[0]),
           'dim': int(X.shape[1]), 'ranks': {}, 'faiss': faiss.__version__,
           'metric': 'inner product on L2-normalised centred PCA prefix',
           'M': 32, 'efConstruction': 200}

    for r in RANKS:
        if r > X.shape[1]:
            continue
        t0 = time.time()
        Gp = np.ascontiguousarray(transform(X, mu, U, r, mode='cen'), dtype=np.float32)
        Qp = np.ascontiguousarray(transform(Q, mu, U, r, mode='cen'), dtype=np.float32)
        t_proj = time.time() - t0

        # exact reference in the same truncated space
        # exact reference in chunks: the full query set is ~25k, so a single
        # n_q x 52k score matrix would be 5 GB.  Same protocol as sb_metric_std
        # (all queries, AP normalised by min(|R|,10)).
        ex_maps, ex_recs, top_ex = [], [], []
        CH = 1024
        for i0 in range(0, Qp.shape[0], CH):
            Sc = score(Qp[i0:i0 + CH], Gp)
            # queries are gallery rows, so the self-match would always be rank 1;
            # sb_metric_std masks it and so must we, or mAP is systematically low.
            Sc[np.arange(Sc.shape[0]), rows[i0:i0 + CH]] = -np.inf
            tc = topk_idx(Sc, 10)
            del Sc
            ex_maps.append(ap_at_k(tc, rel[i0:i0 + CH], 10))
            ex_recs.append(float(np.mean(
                [len(set(t.tolist()) & set(x)) / min(10, len(x))
                 for t, x in zip(tc, rel[i0:i0 + CH]) if x])))
            top_ex.append(tc)
        top_ex = np.vstack(top_ex)
        ex_map = float(np.mean(ex_maps))
        ex_rec = float(np.mean(ex_recs))
        if r == X.shape[1]:
            full_map10 = ex_map

        # ---- HNSW ----
        t0 = time.time()
        index = faiss.IndexHNSWFlat(r, 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 200
        index.add(Gp)
        t_build = time.time() - t0
        ipath = os.path.join(OUT, f'_tmp_hnsw_r{r}.faiss')
        faiss.write_index(index, ipath)
        index_bytes = os.path.getsize(ipath)
        os.remove(ipath)

        sweep = []
        for ef in EFS:
            index.hnsw.efSearch = ef
            faiss.omp_set_num_threads(1)
            _search_no_self(index, Qp[:32], rows[:32], 10)   # warmup
            t0 = time.time()
            I = np.vstack([_search_no_self(index, Qp[i0:i0 + 4096],
                                           rows[i0:i0 + 4096], 10)
                           for i0 in range(0, Qp.shape[0], 4096)])
            dt = time.time() - t0
            ann_map = ap_at_k(I, rel, 10)
            ann_rec = float(np.mean([len(set(t.tolist()) & set(x)) / min(10, len(x))
                                     for t, x in zip(I, rel) if x]))
            ov = float(np.mean([len(set(i.tolist()) & set(e.tolist())) / 10.0
                                for i, e in zip(I, top_ex)]))
            sweep.append({'efSearch': ef, 'qps_single_thread': float(len(Qp) / dt),
                          'ann_recall_vs_exact': ov,
                          'ann_map10': ann_map, 'ann_recall10': ann_rec})

        # throughput with the default thread pool
        faiss.omp_set_num_threads(24)
        index.hnsw.efSearch = 128
        index.search(Qp[:32], 10)
        t0 = time.time()
        index.search(Qp, 10)
        qps_mt = float(len(Qp) / (time.time() - t0))
        faiss.omp_set_num_threads(1)

        res['ranks'][str(r)] = {
            'retention_vs_full': None,   # filled after the full-dim rank is known
            'bytes_per_vector_dense': int(r * 4),
            'index_bytes_total': int(index_bytes),
            'index_bytes_per_vector': float(index_bytes / X.shape[0]),
            'build_s': float(t_build), 'project_s': float(t_proj),
            'exact_map10': ex_map, 'exact_recall10': ex_rec,
            'qps_24thread_ef128': qps_mt, 'sweep': sweep}
        log(f'r={r}: exact mAP@10 {ex_map:.4f} | build {t_build:.1f}s | '
            f'index {index_bytes / X.shape[0]:.0f} B/vec | qps(ef128,1t) '
            f'{sweep[3]["qps_single_thread"]:.1f} | ann mAP@10 {sweep[3]["ann_map10"]:.4f}')

    res['full_dim_map10'] = full_map10
    for v in res['ranks'].values():
        v['retention_vs_full'] = (None if not full_map10
                                  else v['exact_map10'] / full_map10)
    json.dump(res, open(os.path.join(OUT, args.out), 'w'), indent=1)

    print()
    print('rank  bytes  idx B/vec  exactMAP  retain  annMAP(ef128)  annR/ex  qps1t  qps24t')
    for r, v in res['ranks'].items():
        s128 = [s for s in v['sweep'] if s['efSearch'] == 128][0]
        ret = v.get('retention_vs_full')
        print('%4s %6d %10.0f %9.4f %7s %14.4f %8.3f %6.0f %6.0f'
              % (r, v['bytes_per_vector_dense'], v['index_bytes_per_vector'],
                 v['exact_map10'], '%.3f' % ret if ret else '  -  ',
                 s128['ann_map10'], s128['ann_recall_vs_exact'],
                 s128['qps_single_thread'], v['qps_24thread_ef128']))
    print('wrote', os.path.join(OUT, args.out))


if __name__ == '__main__':
    main()
