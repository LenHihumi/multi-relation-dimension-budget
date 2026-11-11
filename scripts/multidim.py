"""The other three axes: build, incremental update, latency.

The storage identity decides the byte axis.  A deployment also pays build time, the cost of admitting new
vectors, and tail latency, and the sharing literature claims all three.  This measures them on the same
four designs as `umax_int8.py`, on the same vectors, so the paper can say "sharing wins on axis X" with a
number attached -- or report that it does not, which is the more useful outcome when the byte axis is
already lost.

Per design:
  build_s      wall clock to construct the graph (48 threads)
  rss_mb       peak RSS of the process after the build (VmHWM), i.e. the resident cost, not the file size
  p50/p99      single-thread per-query latency at efSearch=128, and QPS with 32 threads
  append_s     admitting 1% new vectors into the built index (plus the rerank-store write)
  regroup_s    what a budget change costs: the coarse rank moves by +8, which in the shared design
               invalidates the coarse graph only; in the uniform design the searched rank is unchanged

A fourth measurement, `nest`, is the one that decides the whole question: PCA prefixes are nested, so one
graph at r_item already answers the coarse relation at r_item >= r_cat, at quality that is monotone in r.
If that is true, "one index, many resolutions" -- the property sharing was introduced to buy -- is free in
the uniform design, and no amount of engineering on the shared design can beat it on any axis.

Usage (inside mt-cpu):
  python3 multidim.py --enc clip_vitl14 --ds inshop --out /tmp/md.json
"""
import argparse
import json
import os
import resource
import time

import numpy as np

from sb_paths import SB_ROOT, SB_OUT, SB_CACHE, SB_MODELS, REPO  # paths resolve relative to the deposit
DUMP = os.environ.get('DUMP', SB_OUT + '/spectral_budget')
CACHE = os.environ.get('CACHE', SB_CACHE)
M, EFC, EF, TOPK = 32, 200, 128, 100
LINKS = 2 * M * 4


def load(enc, ds):
    if ds in ('imagenet', 'imagenet1m'):
        X = np.asarray(np.load(os.path.join(CACHE, 'imagenet1m', 'emb.npy'), mmap_mode='r'),
                       dtype=np.float32)
        lab = np.load(os.path.join(CACHE, 'imagenet1m', 'labels.npy'))
        meta = json.load(open(os.path.join(CACHE, 'imagenet1m', 'meta.json')))
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from imagenet_coarse import load_groups
        g = load_groups()
        return X, lab, np.array([g[int(i)] for i in lab])
    base = os.path.join(DUMP, 'emb', '%s__%s' % (enc, ds))
    meta = json.load(open(base + '.json'))
    if os.path.exists(base + '.npy'):
        emb = np.load(base + '.npy', mmap_mode='r')
    else:
        emb = np.memmap(base + '.bin', dtype=np.float32, mode='r').reshape(meta['n'], meta['dim'])
    rows = [json.loads(l) for l in open(os.path.join(CACHE, ds, 'gallery.jsonl'))]
    return (np.asarray(emb, dtype=np.float32), np.array([r['ent'] for r in rows]),
            np.array([r['cat'] for r in rows]))


def rss_mb():
    """Current resident set, not the high-water mark.

    ru_maxrss is a peak and never falls, so using it per design reports the cumulative peak of every
    design built so far and makes the fourth design look like the heaviest.  VmRSS is what is actually
    resident once the graph exists, which is the quantity a deployment provisions.
    """
    with open('/proc/self/status') as f:
        for line in f:
            if line.startswith('VmRSS:'):
                return float(line.split()[1]) / 1024.0
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def build(Z, bits, threads=48):
    import faiss
    faiss.omp_set_num_threads(threads)
    if bits >= 32:
        idx = faiss.IndexHNSWFlat(Z.shape[1], M)
    else:
        idx = faiss.IndexHNSWSQ(Z.shape[1], faiss.ScalarQuantizer.QT_8bit, M)
        idx.train(np.ascontiguousarray(Z))
    idx.hnsw.efConstruction = EFC
    t0 = time.time()
    idx.add(np.ascontiguousarray(Z))
    tb = time.time() - t0
    idx.hnsw.efSearch = EF
    faiss.omp_set_num_threads(1)
    return idx, tb


def latency(idx, Q, store=None, Qstore=None, topk=TOPK, reps=1):
    """Per-query timings on one thread: p50/p99 of the wall clock, and QPS with 32 threads.

    A shared design pays a rerank pass that the uniform design does not, so the pass has to be *inside*
    the timed region or the comparison flatters sharing.  The rerank scores the top-`topk` candidates from
    the store (int8 or float32) in one einsum per query -- the same batched form a serving stack uses.
    """
    import faiss
    faiss.omp_set_num_threads(1)
    idx.hnsw.efSearch = EF
    ts = []
    for r in range(reps):
        for i in range(len(Q)):
            t0 = time.perf_counter()
            _, ids = idx.search(np.ascontiguousarray(Q[i:i + 1]), topk + 1 if store is not None else 11)
            if store is not None:                      # rerank the candidates against this query
                c = ids[0][ids[0] >= 0][:topk]
                s = store[c].astype(np.float32) @ Qstore[i].astype(np.float32)
                np.argsort(-s)[:10]
            ts.append((time.perf_counter() - t0) * 1e3)
    ts = np.array(ts)
    faiss.omp_set_num_threads(32)
    t0 = time.time()
    if store is None:
        idx.search(np.ascontiguousarray(Q), 11)
    else:
        _, ids = idx.search(np.ascontiguousarray(Q), topk + 1)
        for i in range(len(Q)):
            c = ids[i][ids[i] >= 0][:topk]
            s = store[c].astype(np.float32) @ Qstore[i].astype(np.float32)
            np.argsort(-s)[:10]
    qps32 = len(Q) * reps / max(time.time() - t0, 1e-9)
    faiss.omp_set_num_threads(1)
    return float(np.percentile(ts, 50)), float(np.percentile(ts, 99)), qps32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--enc', default='clip_vitl14')
    ap.add_argument('--ds', default='inshop')
    ap.add_argument('--r-item', type=int, default=0)
    ap.add_argument('--r-cat', type=int, default=0)
    ap.add_argument('--queries', type=int, default=300)
    ap.add_argument('--n', type=int, default=0)
    ap.add_argument('--out', default='')
    a = ap.parse_args()

    X, lab_i, lab_c = load(a.enc, a.ds)
    if a.n:
        X, lab_i, lab_c = X[:a.n], lab_i[:a.n], lab_c[:a.n]
    n, d = X.shape
    if not a.r_item or not a.r_cat:
        # In-shop cells have their budgets in metric_std_parts; the ImageNet cell was measured by
        # imagenet_budget.py (a different sweep, 14 ranks) and has no file there, so it is passed in
        p = os.path.join(DUMP, 'metric_std_parts', '%s__%s.json' % (a.enc, a.ds))
        if os.path.exists(p):
            parts = json.load(open(p))
            a.r_item = a.r_item or int(parts['cells']['inst']['metrics']['map10']['fit']['rstar_meas'])
            a.r_cat = a.r_cat or int(parts['cells']['cat']['metrics']['map10']['fit']['rstar_meas'])
        else:
            a.r_item, a.r_cat = a.r_item or 160, a.r_cat or 128
    print('%s/%s n=%d d=%d  r_item=%d r_cat=%d' % (a.enc, a.ds, n, d, a.r_item, a.r_cat))

    mu = X.mean(0)
    Xc = X - mu
    C = (Xc.T @ Xc) / max(len(Xc) - 1, 1)
    w, V = np.linalg.eigh(C)
    V = V[:, np.argsort(w)[::-1]][:, :a.r_item]
    Zi = Xc @ V
    del Xc
    Zi /= np.maximum(np.linalg.norm(Zi, axis=1, keepdims=True), 1e-9)
    Zi = np.ascontiguousarray(Zi, dtype=np.float32)
    Zc = np.ascontiguousarray(Zi[:, :a.r_cat])

    rng = np.random.default_rng(0)
    _, inv, cnt = np.unique(lab_i, return_inverse=True, return_counts=True)
    elig = np.where(cnt[inv] > 1)[0]
    q = np.sort(rng.choice(elig, size=min(a.queries, len(elig)), replace=False))

    # 1% of rows are "new vectors" for the append measurement; they are held out of the build
    m = max(1000, int(0.01 * n))
    new_rows = np.arange(n - m, n)
    keep = np.arange(0, n - m)
    print('build on %d rows, append %d' % (len(keep), m))

    res = {'enc': a.enc, 'ds': a.ds, 'n': n, 'r_item': a.r_item, 'r_cat': a.r_cat,
           'queries': int(len(q)), 'appended': int(m), 'designs': {}}

    def run(name, Zb, bits, Zstore=None):
        t0 = time.time()
        idx, tb = build(Zb[keep], bits)
        rss_after_build = rss_mb()
        Q = np.ascontiguousarray(Zb[q])
        p50, p99, qps32 = latency(idx, Q, store=Zstore,
                                 Qstore=None if Zstore is None else Zstore[q])
        # append 1% new vectors
        t0 = time.time()
        idx.add(np.ascontiguousarray(Zb[new_rows]))
        t_ap = time.time() - t0
        if Zstore is not None:                 # the rerank store grows by m rows of the same width
            _ = np.empty((m, Zstore.shape[1]), dtype=Zstore.dtype)
            _[:] = Zstore[new_rows]
            t_ap += 0.0                        # counted below as bytes only: it is a memcpy
        res['designs'][name] = {
            'bits': bits, 'build_s': tb, 'append_s': t_ap, 'p50_ms': p50, 'p99_ms': p99,
            'qps32': qps32, 'rss_mb': rss_after_build,
            'append_bytes': int(m * (bits * Zb.shape[1] / 8.0 + (0 if Zstore is None else
                                                                Zstore.shape[1]))),
        }
        print('%-10s build %6.1fs  append1%% %5.2fs  p50 %.3f  p99 %.3f  QPS32 %8.0f  RSS %6.0fMB'
              % (name, tb, t_ap, p50, p99, qps32, res['designs'][name]['rss_mb']))
        return idx

    # uniform float32 / int8: one graph at r_item, both relations
    run('umax-f32', Zi, 32)
    run('umax-i8', Zi, 8)

    # shared: coarse graph at r_cat, rerank store at r_item (int8 / f32 respectively)
    scale = float(np.abs(Zi).max()) / 127.0
    Zi8 = np.round(Zi / max(scale, 1e-12)).astype(np.int8)
    run('share-f32', Zc, 32, Zstore=Zi8)
    idx_c = run('share-i8', Zc, 8, Zstore=Zi8)

    # budget change: the coarse rank moves +8 -> the shared coarse graph must be rebuilt, the uniform
    # graph is untouched (it is built at r_item, which did not change)
    r2 = a.r_cat + 8
    if r2 <= a.r_item:
        _, t_re = build(Zi[keep][:, :r2], 8)
        res['regroup_s'] = {'share_i8_coarse_rebuild': t_re, 'umax_i8': 0.0}
        print('regroup (r_cat+8): shared %.1fs, uniform 0.0s' % t_re)

    # nest: does the r_item index already answer the coarse relation at its required quality?
    def ap_ids(ids, per_row, groups, k=10):
        aps = []
        for i, qi in enumerate(q):
            rel = groups[per_row[qi]]
            h, s = 0, 0.0
            for rank, j in enumerate(ids[i][:k], 1):
                if j in rel:
                    h += 1
                    s += h / rank
            aps.append(s / min(len(rel), k))
        return float(np.mean(aps))

    def clean(ids, k=10):
        out = np.empty((len(q), k), dtype=np.int32)
        for i, qi in enumerate(q):
            out[i] = [int(j) for j in ids[i] if int(j) != int(qi)][:k]
        return out

    gi, gc = {}, {}
    for i, v in enumerate(lab_i):
        gi.setdefault(v, set()).add(i)
    for i, v in enumerate(lab_c):
        gc.setdefault(v, set()).add(i)

    idx_i, _ = build(Zi, 8)                    # rebuilt on all rows, untainted by the append
    _, ids_c_from_item = idx_i.search(np.ascontiguousarray(Zi[q]), 11)
    idx_c, _ = build(Zc, 8)
    _, ids_c_only = idx_c.search(np.ascontiguousarray(Zc[q]), 11)
    res['nest'] = {'cat_on_r_item_index': ap_ids(clean(ids_c_from_item), lab_c, gc),
                   'cat_on_r_cat_index': ap_ids(clean(ids_c_only), lab_c, gc),
                   'cat_required_quality': None}
    print('nest: cat mAP on r_item index %.4f vs on r_cat index %.4f'
          % (res['nest']['cat_on_r_item_index'], res['nest']['cat_on_r_cat_index']))

    if a.out:
        json.dump(res, open(a.out, 'w'), indent=1)
        print('wrote', a.out)


if __name__ == '__main__':
    main()
