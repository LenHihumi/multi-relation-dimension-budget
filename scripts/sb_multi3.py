"""Three relevance relations on one basis: does a single max budget waste more as relations accumulate?

Same corpus and encoder as the text cell. The third relation is a review-rating bucket (1-2, 3, 4-5),
which is an exogenous field the corpus ships, not a partition induced by the queries.

Designs: full (ambient), var90 (90% variance), unified (max of the three measured budgets, one graph),
pair (one graph per relation). Reports bytes, build, per-relation QPS and retention, and the mixed
throughput at equal query shares.

Usage:
    python3 sb_multi3.py --out multi3__text.json
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from sb_paths import SB_ROOT, SB_OUT, SB_CACHE, SB_MODELS, REPO  # paths resolve relative to the deposit
TXT = SB_OUT + '/textbudget'
M, EFC, EF, TOPK = 32, 200, 128, 11


def log(m):
    print('[%s] %s' % (time.strftime('%H:%M:%S'), m), flush=True)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--queries', type=int, default=1000)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(os.path.join(TXT, 'corpus.jsonl'), encoding='utf-8')]
    X = np.load(os.path.join(TXT, 'emb_qwen8b.npy')).astype(np.float32)
    n, d = X.shape

    def bucket(r):
        if r is None:
            return 'na'
        return 'low' if r <= 2 else ('mid' if r < 4 else 'high')

    rel = {'item': [r['parent'] for r in rows], 'cat': [r['cat'] for r in rows],
           'rate': [bucket(r.get('rating')) for r in rows]}
    g = {}
    for k, v in rel.items():
        gg = {}
        for i, lab in enumerate(v):
            gg.setdefault(lab, []).append(i)
        g[k] = gg
    log('relations: ' + ', '.join('%s=%d groups' % (k, len(v)) for k, v in g.items()))

    rng = np.random.default_rng(0)
    elig = np.array([i for i in range(n) if len(g['item'][rel['item'][i]]) > 1])
    q = np.sort(rng.choice(elig, size=min(a.queries, len(elig)), replace=False))
    relsets = {k: [set(g[k][rel[k][qq]]) - {int(qq)} for qq in q] for k in g}

    mu = X.mean(0)
    Xc = X - mu
    G = (Xc.T @ Xc) / np.float64(n - 1)
    w, V = np.linalg.eigh(G)
    o = np.argsort(w)[::-1]
    w, V = w[o], V[:, o]
    r90 = int(np.searchsorted(np.cumsum(w) / w.sum(), 0.90) + 1)
    log('90%% variance at rank %d' % r90)

    def prefix(r):
        Z = Xc @ V[:, :r]
        Z /= np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9)
        return np.ascontiguousarray(Z, dtype=np.float32)

    Zf = prefix(d)
    D = Zf[q] @ Zf.T
    np.put_along_axis(D, np.repeat(q[:, None], D.shape[1], 1), -np.inf, axis=1)
    topf = np.argsort(-D, axis=1)[:, :10]
    full_map = {k: map10(topf, relsets[k]) for k in rel}
    log('full-width exact mAP@10 ' + '  '.join('%s %.4f' % (k, v) for k, v in full_map.items()))

    budgets = {}
    for k in rel:
        for r in (8, 12, 16, 24, 32, 40, 48, 64, 80, 96, 128, 160, 224, 320, 448, 640, 1024):
            Z = prefix(r)
            D2 = Z[q] @ Z.T
            np.put_along_axis(D2, np.repeat(q[:, None], D2.shape[1], 1), -np.inf, axis=1)
            if 1 - map10(np.argsort(-D2, axis=1)[:, :10], relsets[k]) / full_map[k] <= 0.05:
                budgets[k] = r
                break
    log('budgets: %s  max %d' % (budgets, max(budgets.values())))

    import faiss
    cache = {}

    def graph(r):
        if r not in cache:
            Z = prefix(r)
            faiss.omp_set_num_threads(48)
            idx = faiss.IndexHNSWFlat(r, M)
            idx.hnsw.efConstruction = EFC
            t0 = time.time()
            idx.add(Z)
            tb = time.time() - t0
            idx.hnsw.efSearch = EF
            cache[r] = (Z, idx, tb)
        return cache[r]

    def qps(idx, Z):
        faiss.omp_set_num_threads(48)
        idx.search(np.ascontiguousarray(Z[q]), TOPK)
        best = float('inf')
        for _ in range(3):
            t0 = time.perf_counter()
            idx.search(np.ascontiguousarray(Z[q]), TOPK)
            best = min(best, time.perf_counter() - t0)
        return len(q) / max(best, 1e-9)

    def lat1(idx, Z):
        """Single-thread p50/p99 over |q| one-query searches: the paper's latency instrument."""
        faiss.omp_set_num_threads(1)
        ts = []
        for i in range(len(q)):
            t0 = time.perf_counter()
            idx.search(np.ascontiguousarray(Z[q[i:i + 1]]), TOPK)
            ts.append((time.perf_counter() - t0) * 1e3)
        faiss.omp_set_num_threads(48)
        ts = np.array(ts)
        return float(np.percentile(ts, 50)), float(np.percentile(ts, 99))

    rmax = max(budgets.values())
    designs = {'full': {k: d for k in rel}, 'var90': {k: r90 for k in rel},
               'unified': {k: rmax for k in rel}, 'pair': dict(budgets)}

    res = {'n': n, 'dim': d, 'r90': r90, 'budgets': budgets, 'groups': {k: len(v) for k, v in g.items()},
           'full_exact': full_map, 'designs': {}}
    for name, spec in designs.items():
        per, rates = {}, {}
        for k, r in spec.items():
            Z, idx, tb = graph(r)
            I = idx.search(np.ascontiguousarray(Z[q]), TOPK)[1][:, 1:]
            m = map10(I, relsets[k])
            q_alone = qps(idx, Z)
            p50, p99 = lat1(idx, Z)
            per[k] = {'r': int(r), 'retention': m / full_map[k], 'qps_alone': q_alone,
                      'p50_ms': p50, 'p99_ms': p99, 'build_s': tb}
            rates[k] = per[k]['qps_alone']
        bvec = sum(4 * r + 2 * M * 4 for r in sorted(set(spec.values())))
        eq = 1.0 / np.mean([1.0 / rates[k] for k in spec])      # equal query shares
        res['designs'][name] = {'ranks': {k: int(v) for k, v in spec.items()},
                                'bytes_per_vec': bvec, 'index_mb': bvec * n / 1e6,
                                'build_s': sum(cache[r][2] for r in set(spec.values())),
                                'graphs': len(set(spec.values())),
                                'per_relation': per, 'qps_equal_share': float(eq)}
        eq_lat = np.mean([per[k]['p50_ms'] for k in spec])      # equal shares, per-query time
        res['designs'][name]['p50_equal_share_ms'] = float(eq_lat)
        log('%-8s ranks %-26s %5d B/vec  p50 %6.1f/%6.1f/%6.1f us  mean %6.1f  ret %.3f/%.3f/%.3f'
            % (name, str(spec), bvec,
               per['item']['p50_ms'] * 1000, per['cat']['p50_ms'] * 1000, per['rate']['p50_ms'] * 1000,
               eq_lat * 1000,
               per['item']['retention'], per['cat']['retention'], per['rate']['retention']))

    json.dump(res, open(a.out, 'w'), indent=1)
    log('wrote ' + a.out)


if __name__ == '__main__':
    main()
