"""Where the quality goes: candidate recall of the coarse graph, and why some cells lose more.

The shared design can only return what the coarse graph proposes.  So the item-side loss should be
explained by one number per cell: of the exact top-10 neighbours at the fine budget, how many appear in
the 100 candidates the coarse graph returns?  If that is ~1.0 the rerank is exact and the design loses
nothing; if it drops, the loss is a recall loss and no amount of reranking recovers it.

This also measures the same quantity for the uniform index (graph at the fine budget, ef=128), which is
how we can tell that the shared design's occasional *gain* is a move of the approximation, not an effect
of truncation.

Read-only; prints one row per encoder.
"""
import argparse
import json
import os

import numpy as np

from sb_paths import SB_ROOT, SB_OUT, SB_CACHE, SB_MODELS, REPO  # paths resolve relative to the deposit
CACHE = os.environ.get('CACHE', SB_CACHE)
DUMP = os.environ.get('DUMP', SB_OUT + '/spectral_budget')
M, EFC, EF, TOPK = 32, 200, 128, 100


def load(enc, ds):
    base = os.path.join(DUMP, 'emb', '%s__%s' % (enc, ds))
    meta = json.load(open(base + '.json'))
    if os.path.exists(base + '.npy'):
        emb = np.load(base + '.npy', mmap_mode='r')
    else:
        emb = np.memmap(base + '.bin', dtype=np.float32, mode='r').reshape(meta['n'], meta['dim'])
    rows = [json.loads(l) for l in open(os.path.join(CACHE, ds, 'gallery.jsonl'))]
    return np.asarray(emb, dtype=np.float32), rows


def hnsw(Z, threads=32):
    import faiss
    faiss.omp_set_num_threads(threads)
    idx = faiss.IndexHNSWFlat(Z.shape[1], M)
    idx.hnsw.efConstruction = EFC
    idx.add(np.ascontiguousarray(Z.astype(np.float32)))
    idx.hnsw.efSearch = EF
    faiss.omp_set_num_threads(1)
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--enc', default='clip_vitl14')
    ap.add_argument('--ds', default='inshop')
    ap.add_argument('--r-item', type=int, default=160)
    ap.add_argument('--r-cat', type=int, default=48)
    ap.add_argument('--queries', type=int, default=500)
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
    groups = {}
    for i, v in enumerate(lab):
        groups.setdefault(v, set()).add(i)
    uniq, inv, cnt = np.unique(lab, return_inverse=True, return_counts=True)
    rng = np.random.default_rng(0)
    q = np.sort(rng.choice(np.where(cnt[inv] > 1)[0], size=min(a.queries, len(uniq)), replace=False))

    # exact top-10 at the fine budget: ground truth for both the recall and the mAP
    exact = np.empty((len(q), 10), dtype=np.int32)
    for i, qi in enumerate(q):
        s = Zi @ Zi[qi]
        s[qi] = -np.inf
        t = np.argpartition(-s, 10)[:10]
        exact[i] = t[np.argsort(-s[t])]

    def strip(ids, k):
        out = []
        for r, qi in enumerate(q):
            out.append([int(j) for j in ids[r] if int(j) != int(qi)][:k])
        return np.array(out)

    idx_c = hnsw(Zc)
    _, cand = idx_c.search(np.ascontiguousarray(Zc[q]), TOPK + 1)
    cand = strip(cand, TOPK)
    idx_i = hnsw(Zi)
    _, umax = idx_i.search(np.ascontiguousarray(Zi[q]), 11)
    umax = strip(umax, 10)

    def cand_recall(cands, k=10):
        return float(np.mean([len(set(exact[i]) & set(cands[i][:TOPK])) / k for i in range(len(q))]))

    def top10_recall(cands):
        return float(np.mean([len(set(exact[i]) & set(cands[i][:10])) / 10 for i in range(len(q))]))

    def ap10(ids):
        aps = []
        for i, qi in enumerate(q):
            rel = groups[lab[qi]] - {qi}
            h, ap = 0, 0.0
            for rank, j in enumerate(ids[i][:10], 1):
                if j in rel:
                    h += 1
                    ap += h / rank
            aps.append(ap / min(len(rel), 10))
        return float(np.mean(aps))

    res = {'enc': a.enc, 'ds': a.ds, 'r_item': a.r_item, 'r_cat': a.r_cat, 'queries': int(len(q)),
           'mAP_exact': ap10(exact), 'mAP_umax': ap10(umax),
           'cand_recall@100': cand_recall(cand), 'umax_recall@10': top10_recall(umax)}
    print('%-16s r=%3d/%3d  exact %.4f  umax %.4f (recall@10 %.3f)  coarse recall@100 %.3f'
          % (a.enc, a.r_cat, a.r_item, res['mAP_exact'], res['mAP_umax'],
             res['umax_recall@10'], res['cand_recall@100']))
    if a.out:
        json.dump(res, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
