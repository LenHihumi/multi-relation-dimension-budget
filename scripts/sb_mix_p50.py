"""Per-query p50 at a stated workload mix, for the three-relation text design.

`sb_multi3.py` reports one mixed number: the equal-share p50, computed as the arithmetic
mean of the three per-relation p50s. A reviewer's fair question is what happens at a mix a
deployment would actually see (90% item, 10% category), because the pair design's advantage
is supposed to grow when the mix is skewed. The mean of p50s cannot answer that -- the p50
of a mixture is not the mean of the component p50s -- so this script measures it: build the
same graphs, then issue a query sequence in the stated proportion and take the median of
the individual query latencies, single-threaded, which is the paper's latency instrument.

Validation: the equal-share run must land close to the published 100 us for the pair design
and 198 us for unified (multi3__text_p50.json). If it does not, the graphs being built here
are not the graphs in the paper and the rest of the output is suppressed.

Run inside a container (faiss needed):
  docker run --rm --entrypoint python3 -e PYTHONPATH=/work/envs/faiss \\
    -v "$PWD":/work -w /work vllm/vllm-openai:v0.31.0 sb_mix_p50.py --out ...
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from sb_paths import SB_OUT

TXT = os.environ.get('TXT_DIR', os.path.join(SB_OUT, 'textbudget'))
M, EFC, EF, TOPK = 32, 200, 128, 11
MIXES = {'equal-3way': (1 / 3., 1 / 3., 1 / 3.),
         'item/cat 50-50': (0.5, 0.5, 0.0),
         'item/cat 90-10': (0.9, 0.1, 0.0)}


def log(m):
    print('[%s] %s' % (time.strftime('%H:%M:%S'), m), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--queries', type=int, default=1000)
    ap.add_argument('--reps', type=int, default=3)
    a = ap.parse_args()

    ref = json.load(open(os.path.join(TXT, 'multi3__text_p50.json'), encoding='utf-8'))
    budgets, r90 = ref['budgets'], ref['r90']
    log('budgets %s  r90 %d' % (budgets, r90))

    rows = [json.loads(l) for l in open(os.path.join(TXT, 'corpus.jsonl'), encoding='utf-8')]
    X = np.load(os.path.join(TXT, 'emb_qwen8b.npy')).astype(np.float32)
    n, d = X.shape

    def bucket(r):
        if r is None:
            return 'na'
        return 'low' if r <= 2 else ('mid' if r < 4 else 'high')

    lab = {'item': [r['parent'] for r in rows], 'cat': [r['cat'] for r in rows],
           'rate': [bucket(r.get('rating')) for r in rows]}
    rng = np.random.default_rng(0)
    cnt = {}
    for v in lab['item']:
        cnt[v] = cnt.get(v, 0) + 1
    elig = np.array([i for i, v in enumerate(lab['item']) if cnt[v] > 1])
    q = np.sort(rng.choice(elig, size=min(a.queries, len(elig)), replace=False))

    mu = X.mean(0)
    Xc = X - mu
    G = (Xc.T @ Xc) / np.float64(n - 1)
    w, V = np.linalg.eigh(G)
    o = np.argsort(w)[::-1]
    V = V[:, o]

    def prefix(r):
        Z = Xc @ V[:, :r]
        Z /= np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9)
        return np.ascontiguousarray(Z, dtype=np.float32)

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
            idx.hnsw.efSearch = EF
            cache[r] = (Z, idx, time.time() - t0)
            log('  graph r=%-4d build %.2fs' % (r, cache[r][2]))
        return cache[r]

    rmax = max(budgets.values())
    designs = {'unified': {k: rmax for k in budgets}, 'pair': dict(budgets)}
    out = {'n': n, 'dim': d, 'budgets': budgets, 'mixes': {}, 'published': {}}
    for name, spec in designs.items():
        for k, r in spec.items():
            graph(r)
        out['published'][name] = ref['designs'][name].get('p50_equal_share_ms')

    for name, spec in designs.items():
        out['mixes'][name] = {}
        for mname, w in MIXES.items():
            ns = {k: int(round(w[i] * len(q))) for i, k in enumerate(('item', 'cat', 'rate'))}
            seq = []
            for k, c in ns.items():
                Z, idx, _ = cache[spec[k]]
                seq.extend([(k, idx, Z[qq]) for qq in q[:c]])
            order = np.random.default_rng(1).permutation(len(seq))
            seq = [seq[i] for i in order]
            faiss.omp_set_num_threads(1)
            for k, idx, z in seq[:50]:                      # warm-up
                idx.search(np.ascontiguousarray(z[None, :]), TOPK)
            best = None
            for _ in range(a.reps):
                ts = []
                for k, idx, z in seq:
                    t0 = time.perf_counter()
                    idx.search(np.ascontiguousarray(z[None, :]), TOPK)
                    ts.append((time.perf_counter() - t0) * 1e3)
                med = float(np.median(ts))
                best = med if best is None else min(best, med)
            faiss.omp_set_num_threads(48)
            out['mixes'][name][mname] = {'n': len(seq), 'p50_ms': best,
                                         'shares': {k: ns[k] / float(len(seq)) for k in ns}}
            log('%-8s %-16s n=%-5d p50 %.4f ms' % (name, mname, len(seq), best))

    pub = {n_: ref['designs'][n_].get('p50_equal_share_ms') for n_ in designs}
    gate = abs(out['mixes']['pair']['equal-3way']['p50_ms'] - pub['pair']) / pub['pair']
    log('equal-3way vs published: pair %.4f vs %.4f (%.0f%%), unified %.4f vs %.4f'
        % (out['mixes']['pair']['equal-3way']['p50_ms'], pub['pair'], 100 * gate,
           out['mixes']['unified']['equal-3way']['p50_ms'], pub['unified']))
    if gate > 0.25:
        out['WARNING'] = ('equal-share pair p50 is %.0f%% away from the published value; '
                          'the graphs here are probably not the paper\'s' % (100 * gate))
        log('WARNING: ' + out['WARNING'])
    json.dump(out, open(a.out, 'w'), indent=1)
    log('wrote ' + a.out)


if __name__ == '__main__':
    main()
