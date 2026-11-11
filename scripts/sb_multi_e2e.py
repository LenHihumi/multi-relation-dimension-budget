"""What the budget decision is worth in an index: four deployments of one corpus, two relations.

The paper's byte arguments are per relation. This script prices the *deployment*: one PCA basis, two
relevance relations, and four ways to store them.

  full      one graph at the ambient dimension, both relations served from it       (do nothing)
  var90     one graph at the rank that retains 90% of the variance, both served    (the heuristic)
  unified   one graph at max(r_item, r_cat), both served                           (one measured budget)
  pair      one graph per relation at its own measured budget                      (the protocol)

`unified` is the byte-optimal use of a single measured budget; `pair` spends one extra link list to
give each relation its own width.  Reported per deployment: bytes per vector, whole index, build
time, per-relation mAP@10 retention against exact full-dimension search, single-thread p50/p99, and
end-to-end throughput on a 50/50 mixed workload.

Usage:
    python3 sb_multi_e2e.py --cell inshop --out multi_e2e__inshop.json
    python3 sb_multi_e2e.py --cell instre --enc dinov3_vitl16 --out multi_e2e__instre_dino.json
    python3 sb_multi_e2e.py --cell text   --out multi_e2e__text.json
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from sb_paths import SB_ROOT, SB_OUT, SB_CACHE, SB_MODELS, REPO  # paths resolve relative to the deposit
SB = SB_CACHE
OUT = SB_OUT + '/spectral_budget'
TXT = SB_OUT + '/textbudget'
M = 32
EFC = 200
EF = 128
TOPK = 11


def log(m):
    print('[%s] %s' % (time.strftime('%H:%M:%S'), m), flush=True)


def load_inshop(enc='clip_vitl14'):
    rows = [json.loads(l) for l in open(os.path.join(SB, 'inshop', 'gallery.jsonl'), encoding='utf-8')]
    base = os.path.join(OUT, 'emb', enc + '__inshop')
    meta = json.load(open(base + '.json'))
    if os.path.exists(base + '.npy'):
        X = np.load(base + '.npy')
    else:
        X = np.memmap(base + '.bin', dtype=np.float32, mode='r').reshape(meta['n'], meta['dim'])
    X = np.asarray(X, dtype=np.float32)
    qs = json.load(open(os.path.join(SB, 'inshop', 'queries.json')))
    q = np.array(qs['inst'], dtype=int)
    rel = {'item': [r['ent'] for r in rows], 'cat': [r['cat'] for r in rows]}
    return X, rel, q


def load_instre(enc='clip_vitl14'):
    """INSTRE-8000: 200 instance labels, 21 category labels, 1,000 instance queries.

    The label file is gallery_218.jsonl, which differs from gallery.jsonl only in the path prefix
    of the shared mount; row order and labels are identical, and the embeddings were written in
    that order.
    """
    rows = [json.loads(l) for l in open(os.path.join(SB, 'instre', 'gallery_218.jsonl'),
                                       encoding='utf-8')]
    base = os.path.join(OUT, 'emb', enc + '__instre')
    meta = json.load(open(base + '.json'))
    if os.path.exists(base + '.npy'):
        X = np.load(base + '.npy')
    else:
        X = np.memmap(base + '.bin', dtype=np.float32, mode='r').reshape(meta['n'], meta['dim'])
    X = np.asarray(X, dtype=np.float32)
    if X.shape[0] != len(rows):
        raise SystemExit('row/embedding mismatch: %d vs %d' % (X.shape[0], len(rows)))
    qs = json.load(open(os.path.join(SB, 'instre', 'queries.json')))
    q = np.array(qs['inst'], dtype=int)
    rel = {'item': [r['ent'] for r in rows], 'cat': [r['cat'] for r in rows]}
    return X, rel, q


def load_text():
    rows = [json.loads(l) for l in open(os.path.join(TXT, 'corpus.jsonl'), encoding='utf-8')]
    X = np.load(os.path.join(TXT, 'emb_qwen8b.npy')).astype(np.float32)
    rel = {'item': [r['parent'] for r in rows], 'cat': [r['cat'] for r in rows]}
    rng = np.random.default_rng(0)
    cnt = {}
    for v in rel['item']:
        cnt[v] = cnt.get(v, 0) + 1
    elig = np.array([i for i, v in enumerate(rel['item']) if cnt[v] > 1])
    q = np.sort(rng.choice(elig, size=1000, replace=False))
    return X, rel, q


def groups(labels):
    g = {}
    for i, v in enumerate(labels):
        g.setdefault(v, []).append(i)
    return g


def map10(top, relsets):
    tot = 0.0
    for row, rs in zip(top, relsets):
        if not rs:
            continue
        hits, cum = 0, 0.0
        for j, d in enumerate(row):
            if d in rs:
                hits += 1
                cum += hits / (j + 1)
        tot += cum / min(len(rs), 10)
    return tot / max(len(relsets), 1)


def build(Z, name):
    import faiss
    faiss.omp_set_num_threads(48)
    idx = faiss.IndexHNSWFlat(Z.shape[1], M)
    idx.hnsw.efConstruction = EFC
    t0 = time.time()
    idx.add(Z)
    dt = time.time() - t0
    idx.hnsw.efSearch = EF
    return idx, dt


def search_top(idx, Q, k):
    D, I = idx.search(np.ascontiguousarray(Q), k)
    return I[:, 1:]                                     # drop self, which is the first hit


def latency(idx, Q, k=TOPK):
    import faiss
    faiss.omp_set_num_threads(1)
    ts = []
    for i in range(len(Q)):
        t0 = time.perf_counter()
        idx.search(np.ascontiguousarray(Q[i:i + 1]), k)
        ts.append((time.perf_counter() - t0) * 1e3)
    faiss.omp_set_num_threads(48)
    ts = np.array(ts)
    return float(np.percentile(ts, 50)), float(np.percentile(ts, 99))


def single_qps(idx, Q, k=TOPK, reps=3):
    """One batch of |Q| searches, best of `reps` after one untimed warm-up batch.

    The warm-up is not decoration: `latency` drops faiss to one thread and restores 48, and the
    first 48-thread batch after that pays the OpenMP pool start-up, which at n=8k is comparable to
    the search itself (it read 192k QPS where the steady state is 410k).
    """
    import faiss
    Qc = np.ascontiguousarray(Q)
    idx.search(Qc, k)
    best = float('inf')
    for _ in range(reps):
        t0 = time.perf_counter()
        idx.search(Qc, k)
        best = min(best, time.perf_counter() - t0)
    return len(Q) / max(best, 1e-9)


def mixed_qps(pairs, reps=3):
    """Queries arrive 50/50; each is searched in the graph its relation uses."""
    import faiss
    n = sum(len(Q) for _, Q, _ in pairs)
    flat = [(idx, np.ascontiguousarray(Q), k) for idx, Q, k in pairs]
    for idx, Q, k in flat:
        idx.search(Q, k)
    best = float('inf')
    for _ in range(reps):
        t0 = time.perf_counter()
        for idx, Q, k in flat:
            idx.search(Q, k)
        best = min(best, time.perf_counter() - t0)
    return n / max(best, 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell', choices=['inshop', 'instre', 'text'], required=True)
    ap.add_argument('--enc', default='clip_vitl14', help='encoder tag for the image cells')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    if a.cell == 'text':
        X, rel, q = load_text()
    elif a.cell == 'instre':
        X, rel, q = load_instre(a.enc)
    else:
        X, rel, q = load_inshop(a.enc)
    enc_tag = 'text' if a.cell == 'text' else a.enc
    n, d = X.shape
    log('%s/%s corpus %d x %d' % (a.cell, enc_tag, n, d))

    mu = X.mean(0)
    Xc = X - mu
    G = (Xc.T @ Xc) / np.float64(n - 1)
    w, V = np.linalg.eigh(G)
    o = np.argsort(w)[::-1]
    w, V = w[o], V[:, o]
    cum = np.cumsum(w) / w.sum()
    r_var90 = int(np.searchsorted(cum, 0.90) + 1)
    log('90%% variance at rank %d' % r_var90)

    g = {k: groups(v) for k, v in rel.items()}
    relsets = {k: [set(g[k][rel[k][qq]]) - {int(qq)} for qq in q] for k in g}

    def prefix(r):
        Z = Xc @ V[:, :r]
        Z /= np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9)
        return np.ascontiguousarray(Z, dtype=np.float32)

    # exact full-dimension reference for both relations
    Zf = prefix(d)
    D = Zf[q] @ Zf.T
    np.put_along_axis(D, np.repeat(q[:, None], D.shape[1], 1), -np.inf, axis=1)
    topf = np.argsort(-D, axis=1)[:, :10]
    full_map = {k: map10(topf, relsets[k]) for k in rel}
    log('full-width exact mAP@10  item %.4f  cat %.4f' % (full_map['item'], full_map['cat']))

    # measured budgets, from the same instrument as the paper (5% of each relation's own full width)
    budgets = {}
    for k in rel:
        for r in [x for x in (16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320,
                              384, 448, 512, 640, 768, 1024, 1536, 2048, 3072, 4096) if x <= d]:
            Z = prefix(r)
            D2 = Z[q] @ Z.T
            np.put_along_axis(D2, np.repeat(q[:, None], D2.shape[1], 1), -np.inf, axis=1)
            t = np.argsort(-D2, axis=1)[:, :10]
            m = map10(t, relsets[k])
            if 1 - m / full_map[k] <= 0.05:
                budgets[k] = r
                break
    log('measured budgets: %s' % budgets)

    cache = {}

    def graph(r):
        if r not in cache:
            Z = prefix(r)
            idx, tb = build(Z, r)
            cache[r] = (Z, idx, tb)
        return cache[r]

    designs = {
        'full': {'item': d, 'cat': d},
        'var90': {'item': r_var90, 'cat': r_var90},
        'unified': {'item': max(budgets.values()), 'cat': max(budgets.values())},
        'pair': {'item': budgets['item'], 'cat': budgets['cat']},
    }

    res = {'cell': a.cell, 'enc': enc_tag, 'n': n, 'dim': d, 'r_var90': r_var90, 'budgets': budgets,
           'full_exact': full_map, 'designs': {}}
    for name, spec in designs.items():
        t0 = time.time()
        rows, pairs = {}, []
        for k, r in spec.items():
            Z, idx, tb = graph(r)
            I = search_top(idx, Z[q], TOPK)
            m = map10(I, relsets[k])
            p50, p99 = latency(idx, Z[q])
            rows[k] = {'r': int(r), 'map': m, 'retention': m / full_map[k], 'p50_ms': p50,
                       'p99_ms': p99, 'build_s': tb, 'qps_alone': single_qps(idx, Z[q])}
            pairs.append((idx, Z[q], TOPK))
        qps = mixed_qps(pairs)
        graphs = sorted(set(spec.values()))
        bvec = sum(4 * r + 2 * M * 4 for r in graphs)       # one link list per graph
        # throughput at any workload mix: a query of relation k costs 1/qps_k, so the mix is the
        # harmonic combination of the two single-relation rates (verified against the measured 50/50)
        mixes = {}
        for w in (0.1, 0.3, 0.5, 0.7, 0.9):
            mixes[str(w)] = 1.0 / (w / rows['item']['qps_alone'] + (1 - w) / rows['cat']['qps_alone'])
        res['designs'][name] = {
            'ranks': {k: int(v) for k, v in spec.items()}, 'per_relation': rows,
            'qps_mix': mixes, 'qps_mix50_measured': qps,
            'bytes_per_vec': bvec, 'index_mb': bvec * n / 1e6,
            'build_s': sum(cache[r][2] for r in graphs),
            'graphs': len(graphs), 'mixed_qps': qps,
            'wall_s': time.time() - t0}
        log('%-8s ranks %-20s %5d B/vec %6.1f MB  build %4.1fs  QPS(it/cat) %7.0f/%7.0f  '
            'mix .7/.5/.3 %6.0f/%6.0f/%6.0f  ret %.3f/%.3f'
            % (name, str(spec), bvec, bvec * n / 1e6, res['designs'][name]['build_s'],
               rows['item']['qps_alone'], rows['cat']['qps_alone'],
               mixes['0.7'], mixes['0.5'], mixes['0.3'],
               rows['item']['retention'], rows['cat']['retention']))

    json.dump(res, open(a.out, 'w'), indent=1)
    log('wrote ' + a.out)


if __name__ == '__main__':
    main()
