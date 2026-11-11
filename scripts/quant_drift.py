"""Does the budget survive quantisation?  Rank drift under int8 and PQ, on the paper's cell.

The byte model in the shared-index study stores the rerank vectors as int8.  That is only defensible if
quantisation changes *rankings* little enough that the measured budget still applies.  This measures the
drift directly, on CLIP/In-shop at the paper's item budget:

  * top-10 overlap with the float32 ranking (mean over queries)
  * 95th-percentile displacement of the k-th true neighbour's rank (the reviewer's "95% rank drift")

Variants: int8 (per-tensor scale), PQ with 8 sub-quantisers (faiss), and fp8 if the runtime has it.
Read-only; prints a table.
"""
import argparse
import json
import os

import faiss
import numpy as np

from sb_paths import SB_ROOT, SB_OUT, SB_CACHE, SB_MODELS, REPO  # paths resolve relative to the deposit
DUMP = os.environ.get('DUMP', SB_OUT + '/spectral_budget')
CACHE = os.environ.get('CACHE', SB_CACHE)


def load(enc, ds):
    base = os.path.join(DUMP, 'emb', '%s__%s' % (enc, ds))
    meta = json.load(open(base + '.json'))
    emb = np.load(base + '.npy', mmap_mode='r')
    rows = [json.loads(l) for l in open(os.path.join(CACHE, ds, 'gallery.jsonl'))]
    return np.asarray(emb, dtype=np.float32), rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--enc', default='clip_vitl14')
    ap.add_argument('--ds', default='inshop')
    ap.add_argument('--rank', type=int, default=160)
    ap.add_argument('--queries', type=int, default=500)
    ap.add_argument('--out', default='')
    a = ap.parse_args()

    X, rows = load(a.enc, a.ds)
    mu = X.mean(0)
    Xc = X - mu
    C = (Xc.T @ Xc) / (len(Xc) - 1)
    w, V = np.linalg.eigh(C)
    V = V[:, np.argsort(w)[::-1]][:, :a.rank]
    Z = Xc @ V
    del Xc
    Z /= np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9)

    lab = np.array([r['ent'] for r in rows])
    groups = {}
    for i, v in enumerate(lab):
        groups.setdefault(v, set()).add(i)
    rng = np.random.default_rng(0)
    # labels are strings in the In-shop cache (entity ids), so bincount is not applicable
    uniq, inv, cnt = np.unique(lab, return_inverse=True, return_counts=True)
    elig = np.where(cnt[inv] > 1)[0]
    q = np.sort(rng.choice(elig, size=min(a.queries, len(elig)), replace=False))

    def topk(mat, k=10):
        out = np.empty((len(q), k), dtype=np.int32)
        for i, qi in enumerate(q):
            s = mat @ mat[qi]
            s[qi] = -np.inf
            t = np.argpartition(-s, k)[:k]
            out[i] = t[np.argsort(-s[t])]
        return out

    base = topk(Z)
    variants = {}

    # int8, one scale for the whole tensor (per-dimension scales would change the metric)
    scale = float(np.abs(Z).max()) / 127.0
    Z8 = np.round(Z / scale).astype(np.int8)
    variants['int8'] = topk(Z8.astype(np.float32) * scale)

    # PQ at two operating points: 64 and 128 bits per vector (10x and 5x smaller than int8)
    d = Z.shape[1]
    for m in (8, 16, 32):
        if d % m:
            print('skip pq%dx8: %d dims not divisible by %d' % (m, d, m))
            continue
        pq = faiss.IndexPQ(d, m, 8)
        pq.train(Z)
        pq.add(Z)
        _, ids = pq.search(Z[q], 10)
        variants['pq%dx8' % m] = ids.astype(np.int32)

    # fp8 (e4m3): what a float8 deployment would store, cast through torch so the rounding is the
    # real hardware rounding rather than a numpy approximation
    try:
        import torch
        zt = torch.from_numpy(Z)
        f8 = zt.to(torch.float8_e4m3fn).to(torch.float32).numpy()
        variants['fp8e4m3'] = topk(f8)
    except Exception as e:                     # older torch: report the gap instead of guessing
        print('skip fp8: %s' % type(e).__name__)

    def drift(cand):
        ov = np.mean([len(set(base[i]) & set(cand[i])) / 10.0 for i in range(len(q))])
        # displacement of each true neighbour: |rank_quant - rank_f32| for neighbours found by f32
        disp = []
        for i, qi in enumerate(q):
            pos = {int(j): r for r, j in enumerate(base[i])}
            for r, j in enumerate(cand[i]):
                if int(j) in pos:
                    disp.append(abs(r - pos[int(j)]))
        return ov, float(np.quantile(disp, 0.95)) if disp else float('nan')

    print('%-8s %-14s %s' % ('variant', 'top-10 overlap', 'p95 rank disp'))
    print('%-8s %-14s %s' % ('f32', '1.000', '0.0'))
    res = {'rank': a.rank, 'queries': int(len(q)), 'variants': {}}
    for name, cand in variants.items():
        ov, p95 = drift(cand)
        res['variants'][name] = {'overlap': ov, 'p95_disp': p95}
        print('%-8s %-14.3f %.1f' % (name, ov, p95))

    if a.out:
        json.dump(res, open(a.out, 'w'), indent=1)
        print('wrote', a.out)


if __name__ == '__main__':
    main()
