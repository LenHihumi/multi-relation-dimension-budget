"""Per-query p50 at a stated item/category mix, for the image cells.

Sec. VII-B argues the two-graph design pays off "at 90% category", but that number is an
extrapolation from the single-relation rates, not a measurement of a mixed query stream,
and Sec. VII-D's 2.0x for the three-relation text case turned out to be a mix effect (it
collapses at 90/10 item-heavy).  This script closes the same gap for the image cells: build
the unified and pair graphs from the released budgets, then issue a query sequence at a
stated item/category proportion and take the median of the individual single-threaded query
latencies.

Cells: every data/textbudget/multi_e2e2__*.json that carries two relations (text is handled
by sb_mix_p50.py, which has three).

Validation: the 50/50 run must land within 25% of the published p50_equal_share_ms, which is
the paper's own model (mean of per-relation p50) -- a wider tolerance than the model's own
noise because the two quantities differ.  A cell that misses is reported, not silently kept.

Run inside a container (faiss needed):
  docker run --rm --entrypoint python3 -e PYTHONPATH=/work/envs/faiss \\
    -e SB_CACHE=/work/.cache/sb -e SB_OUT=/work/outputs/diag \\
    -v "$PWD":/work -w /work vllm/vllm-openai:v0.31.0 sb_mix_image.py --out ...
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import sb_multi_e2e as M  # noqa: E402
from sb_paths import SB_OUT  # noqa: E402

TXT = os.environ.get('TXT_DIR', os.path.join(SB_OUT, 'textbudget'))
MIXES = {'50-50': (0.5, 0.5), '90-10': (0.9, 0.1), '10-90': (0.1, 0.9)}


def log(m):
    print('[%s] %s' % (time.strftime('%H:%M:%S'), m), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--limit', type=int, default=0, help='stop after N cells (smoke test)')
    a = ap.parse_args()

    import faiss
    cells = []
    for f in sorted(glob.glob(os.path.join(TXT, 'multi_e2e2__*.json'))):
        d = json.load(open(f, encoding='utf-8'))
        if d.get('cell') == 'text' or 'item' not in d.get('budgets', {}):
            continue
        if any(c[1]['enc'] == d['enc'] and c[1]['cell'] == d['cell'] for c in cells):
            continue
        cells.append((f, d))
    if a.limit:
        cells = cells[:a.limit]

    res = {'cells': []}
    for f, ref in cells:
        enc, ds = ref['enc'], ref['cell']
        try:
            X, rel, q = (M.load_instre(enc) if ds == 'instre' else M.load_inshop(enc))
        except Exception as e:
            log('skip %s/%s: %s' % (enc, ds, e))
            continue
        n, dd = X.shape
        mu = X.mean(0)
        Xc = X - mu
        G = (Xc.T @ Xc) / np.float64(n - 1)
        w, V = np.linalg.eigh(G)
        V = V[:, np.argsort(w)[::-1]]
        budgets = {k: int(v) for k, v in ref['budgets'].items()}
        rmax = max(budgets.values())

        def prefix(r):
            Z = Xc @ V[:, :r]
            Z /= np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9)
            return np.ascontiguousarray(Z, dtype=np.float32)

        cache = {}

        def graph(r):
            if r not in cache:
                Z = prefix(r)
                faiss.omp_set_num_threads(48)
                idx = faiss.IndexHNSWFlat(r, M.M)
                idx.hnsw.efConstruction = M.EFC
                t0 = time.time()
                idx.add(Z)
                idx.hnsw.efSearch = M.EF
                cache[r] = (Z, idx, time.time() - t0)
            return cache[r]

        designs = {'unified': {k: rmax for k in budgets}, 'pair': dict(budgets)}
        for spec in designs.values():
            for r in spec.values():
                graph(r)
        log('%s/%s n=%d budgets %s rmax %d' % (enc, ds, n, budgets, rmax))

        row = {'enc': enc, 'ds': ds, 'n': n, 'budgets': budgets,
               'published_equal_share': {k: ref['designs'][k].get('p50_equal_share_ms')
                                         for k in designs}, 'mixes': {}}
        for name, spec in designs.items():
            row['mixes'][name] = {}
            for mname, (wi, wc) in MIXES.items():
                ni, nc = int(round(wi * len(q))), int(round(wc * len(q)))
                seq = []
                for k, c in (('item', ni), ('cat', nc)):
                    Z, idx, _ = cache[spec[k]]
                    seq.extend([(idx, Z[qq]) for qq in q[:c]])
                order = np.random.default_rng(1).permutation(len(seq))
                seq = [seq[i] for i in order]
                faiss.omp_set_num_threads(1)
                for idx, z in seq[:20]:
                    idx.search(np.ascontiguousarray(z[None, :]), M.TOPK)
                reps_ms = []
                for _ in range(a.reps):
                    ts = []
                    for idx, z in seq:
                        t0 = time.perf_counter()
                        idx.search(np.ascontiguousarray(z[None, :]), M.TOPK)
                        ts.append((time.perf_counter() - t0) * 1e3)
                    reps_ms.append(float(np.median(ts)))
                best = float(np.median(reps_ms))
                faiss.omp_set_num_threads(48)
                row['mixes'][name][mname] = {'n': len(seq), 'p50_ms': best,
                                             'reps_ms': [round(v, 4) for v in reps_ms],
                                             'spread': round(max(reps_ms) / min(reps_ms), 2)}
                log('  %-8s %-6s n=%-5d p50 %.4f ms' % (name, mname, len(seq), best))
        u, p = row['mixes']['unified']['50-50']['p50_ms'], row['mixes']['pair']['50-50']['p50_ms']
        row['ratio_50_50'] = u / p
        row['ratio_90_10'] = (row['mixes']['unified']['90-10']['p50_ms'] /
                              row['mixes']['pair']['90-10']['p50_ms'])
        pubp = row['published_equal_share'].get('pair')
        if pubp:
            row['gate_pair_50_50_vs_model'] = round(p / pubp, 3)
        log('  ratios: 50-50 %.2fx, 90-10 %.2fx' % (row['ratio_50_50'], row['ratio_90_10']))
        res['cells'].append(row)

    json.dump(res, open(a.out, 'w'), indent=1)
    log('wrote ' + a.out)


if __name__ == '__main__':
    main()
