"""Candidate depth and beam width: the two knobs the design actually exposes.

The shared store returns whatever the coarse graph proposes, so its quality is controlled by how many
candidates are reranked (TOPK) and how hard the graph search works (efSearch).  This sweeps both on the
best and the worst In-shop cell, so the paper can say why TOPK=100 and ef=128 rather than asserting it.

Reports mAP@10 for both relations and the rerank cost per query, since TOPK is where quality and latency
trade against each other directly.
"""
import argparse
import json
import os
import time

import numpy as np

from sb_paths import SB_ROOT, SB_OUT, SB_CACHE, SB_MODELS, REPO  # paths resolve relative to the deposit
CACHE = os.environ.get('CACHE', SB_CACHE)
DUMP = os.environ.get('DUMP', SB_OUT + '/spectral_budget')
M, EFC = 32, 200


def load(enc, ds):
    base = os.path.join(DUMP, 'emb', '%s__%s' % (enc, ds))
    meta = json.load(open(base + '.json'))
    emb = np.load(base + '.npy', mmap_mode='r') if os.path.exists(base + '.npy') else \
        np.memmap(base + '.bin', dtype=np.float32, mode='r').reshape(meta['n'], meta['dim'])
    rows = [json.loads(l) for l in open(os.path.join(CACHE, ds, 'gallery.jsonl'))]
    return np.asarray(emb, dtype=np.float32), rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--enc', default='clip_vitl14')
    ap.add_argument('--ds', default='inshop')
    ap.add_argument('--r-item', type=int, default=160)
    ap.add_argument('--r-cat', type=int, default=48)
    ap.add_argument('--queries', type=int, default=300)
    ap.add_argument('--out', default='')
    a = ap.parse_args()

    X, rows = load(a.enc, a.ds)
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
    g = {}
    for key in (lab, cat):
        d = {}
        for i, v in enumerate(key):
            d.setdefault(v, set()).add(i)
        g[id(key)] = d
    g_inst, g_cat = g[id(lab)], g[id(cat)]

    uniq, inv, cnt = np.unique(lab, return_inverse=True, return_counts=True)
    rng = np.random.default_rng(0)
    q = np.sort(rng.choice(np.where(cnt[inv] > 1)[0], size=min(a.queries, len(uniq)), replace=False))

    scale = float(np.abs(Zi).max()) / 127.0
    Z8 = np.round(Zi / max(scale, 1e-12)).astype(np.int8)

    import faiss
    faiss.omp_set_num_threads(32)
    idx = faiss.IndexHNSWFlat(Zc.shape[1], M)
    idx.hnsw.efConstruction = EFC
    idx.add(np.ascontiguousarray(Zc))
    faiss.omp_set_num_threads(1)

    def ap10(ids, per_row, groups):
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

    rows_out = []
    for ef in (32, 64, 128, 256):
        idx.hnsw.efSearch = ef
        for topk in (20, 50, 100, 200):
            _, cand = idx.search(np.ascontiguousarray(Zc[q]), topk + 1)
            ids, t0 = [], time.time()
            for i, qi in enumerate(q):
                c = np.array([int(j) for j in cand[i] if int(j) != int(qi)][:topk])
                sims = Z8[c].astype(np.int32) @ Z8[qi].astype(np.int32)
                ids.append(c[np.argsort(-sims)[:10]])
            ms = (time.time() - t0) / len(q) * 1e3
            rows_out.append({'ef': ef, 'topk': topk,
                             'mAP_inst': ap10(ids, lab, g_inst),
                             'mAP_cat': ap10(ids, cat, g_cat), 'rerank_ms': ms})
            print('  ef=%3d topk=%3d  mAP item %.4f  cat %.4f  rerank %.3f ms'
                  % (ef, topk, rows_out[-1]['mAP_inst'], rows_out[-1]['mAP_cat'], ms))

    if a.out:
        json.dump({'enc': a.enc, 'ds': a.ds, 'r_cat': a.r_cat, 'r_item': a.r_item,
                   'rows': rows_out}, open(a.out, 'w'), indent=1)
        print('wrote', a.out)


if __name__ == '__main__':
    main()
