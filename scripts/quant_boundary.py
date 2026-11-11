"""Where is the quantisation boundary?  Bits x scale granularity x role.

The system paper now says "quantise first": the uniform index is stored (and traversed) in int8, which
costs <=0.4% mAP and is 7-31% smaller than any shared design.  A reviewer's next question is the obvious
one -- *why 8 bits?*  If 4-bit works, the storage argument gets stronger; if it fails, the paper gains a
boundary ("quantisation is usable down to b bits, and not below") which is a positive result rather than
a hedge.

Three roles are measured on the same vectors, because a bit budget is not one number:

  exact   quantise, dequantise, exact top-10.        Isolates the representation: no graph, no candidate
                                                     stage.  This is the controlled measurement.
  graph   HNSW built and searched on the dequantised copy.  Confirms the representation survives the
                                                     actual traversal path (the paper's claim).
  rerank  f32 top-100 candidates, re-scored from the b-bit store.  The rerank layer's tolerance, which
                                                     is *stricter* than traversal: a perturbation that
                                                     merely reorders ranks 40-60 is harmless, the same
                                                     perturbation inside the top-10 is not.

Scale granularity is a confound, not a detail.  Per-tensor scales keep the quantised dot product a lossy
copy of cosine; per-dimension scales reweight the metric (the paper already lost a claim to that bug).
Per-block scales (32 dims) are what a 4-bit deployment actually uses.  All three are reported so the
boundary is not an artefact of the scale choice.

Usage (inside mt-cpu):
  python3 quant_boundary.py --enc clip_vitl14 --ds inshop --out /tmp/qb.json
  python3 quant_boundary.py --ds imagenet --bits 16,8,4 --no-graph
"""
import argparse
import json
import os
import time

import numpy as np

from sb_paths import SB_ROOT, SB_OUT, SB_CACHE, SB_MODELS, REPO  # paths resolve relative to the deposit
DUMP = os.environ.get('DUMP', SB_OUT + '/spectral_budget')
CACHE = os.environ.get('CACHE', SB_CACHE)
M, EFC, EF = 32, 200, 128


# ----------------------------------------------------------------------------- data
def load(enc, ds):
    """Return (X, per_row_inst, per_row_cat).  Two corpora, two label sources."""
    if ds in ('imagenet', 'imagenet1m'):
        X = np.asarray(np.load(os.path.join(CACHE, 'imagenet1m', 'emb.npy'), mmap_mode='r'),
                       dtype=np.float32)
        lab = np.load(os.path.join(CACHE, 'imagenet1m', 'labels.npy'))
        meta = json.load(open(os.path.join(CACHE, 'imagenet1m', 'meta.json')))
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from imagenet_coarse import load_groups
        idx2group = load_groups()
        coarse = np.array([idx2group[int(i)] for i in lab])
        return X, lab, coarse
    base = os.path.join(DUMP, 'emb', '%s__%s' % (enc, ds))
    meta = json.load(open(base + '.json'))
    if os.path.exists(base + '.npy'):
        emb = np.load(base + '.npy', mmap_mode='r')
    else:
        emb = np.memmap(base + '.bin', dtype=np.float32, mode='r').reshape(meta['n'], meta['dim'])
    rows = [json.loads(l) for l in open(os.path.join(CACHE, ds, 'gallery.jsonl'))]
    X = np.asarray(emb, dtype=np.float32)
    return X, np.array([r['ent'] for r in rows]), np.array([r['cat'] for r in rows])


def pick_queries(lab, nq, seed=0):
    """Queries are rows whose group has at least one other member (a query with no neighbour has no AP)."""
    _, inv, cnt = np.unique(lab, return_inverse=True, return_counts=True)
    elig = np.where(cnt[inv] > 1)[0]
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(elig, size=min(nq, len(elig)), replace=False))


# ------------------------------------------------------------------------ quantisation
def qdq(Z, bits, mode):
    """Quantise-dequantise.  Symmetric mid-tread: levels are {-2^(b-1)+1 .. 2^(b-1)-1} * scale."""
    if bits >= 16:
        return Z.copy()
    qmax = float(2 ** (bits - 1) - 1)
    if mode == 'tensor':
        scale = np.abs(Z).max() / qmax
        return np.round(Z / max(scale, 1e-12)) * max(scale, 1e-12)
    if mode == 'dim':
        scale = np.abs(Z).max(0, keepdims=True) / qmax
        scale = np.maximum(scale, 1e-12)
        return np.round(Z / scale) * scale
    if mode.startswith('block'):
        bs = int(mode[5:])
        d = Z.shape[1]
        pad = (-d) % bs
        Zp = np.pad(Z, ((0, 0), (0, pad))) if pad else Z
        sh = Zp.reshape(len(Zp), -1, bs)
        scale = np.abs(sh).max(2, keepdims=True) / qmax
        scale = np.maximum(scale, 1e-12)
        out = np.round(sh / scale) * scale
        out = out.reshape(len(Z), -1)
        return np.ascontiguousarray(out[:, :d])
    raise ValueError(mode)


def bytes_per_vec(bits, dim, links=True):
    return bits * dim / 8.0 + (2 * M * 4 if links else 0)


# ------------------------------------------------------------------------ metrics
def map_at_10(scores_fn, q, per_row, groups, k=10):
    aps = []
    for qi in q:
        s = scores_fn(qi)
        s = np.array(s, dtype=np.float32, copy=True)
        s[qi] = -np.inf
        top = np.argpartition(-s, k)[:k]
        top = top[np.argsort(-s[top])]
        rel = groups[per_row[qi]]
        h, ap = 0, 0.0
        for rank, j in enumerate(top, 1):
            if j in rel:
                h += 1
                ap += h / rank
        aps.append(ap / min(len(rel), k))
    return float(np.mean(aps))


def exact_scores(Z):
    def f(qi):
        return Z @ Z[qi]
    return f


def topk_ranks(scores_fn, q, k):
    out = np.empty((len(q), k), dtype=np.int32)
    for i, qi in enumerate(q):
        s = scores_fn(qi)
        s = np.array(s, dtype=np.float32, copy=True)
        s[qi] = -np.inf
        t = np.argpartition(-s, k)[:k]
        out[i] = t[np.argsort(-s[t])]
    return out


def rerank_drift(Zq, q, base_cand, k=10):
    """Re-score f32 top-k candidates from the quantised store; report overlap and p95 displacement."""
    ov, ov_top = [], 0.0
    disp = []
    for i, qi in enumerate(q):
        c = base_cand[i]
        s = Zq[c] @ Zq[qi]
        order = c[np.argsort(-s)]
        ov.append(len(set(order[:k].tolist()) & set(c[:k].tolist())) / k)
    return float(np.mean(ov)), float('nan')


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--enc', default='clip_vitl14')
    ap.add_argument('--ds', default='inshop')
    ap.add_argument('--bits', default='8,6,4,3,2')
    ap.add_argument('--modes', default='tensor,block32,dim')
    ap.add_argument('--rank', type=int, default=0, help='0 = the cell measured r_item')
    ap.add_argument('--queries', type=int, default=300)
    ap.add_argument('--n', type=int, default=0)
    ap.add_argument('--no-graph', action='store_true')
    ap.add_argument('--graph-bits', default='8,4')
    ap.add_argument('--graph-modes', default='tensor', help='comma list, crossed with --graph-bits')
    ap.add_argument('--skip-exact', action='store_true')
    ap.add_argument('--out', default='')
    a = ap.parse_args()

    t_all = time.time()
    X, lab_i, lab_c = load(a.enc, a.ds)
    if a.n:
        X, lab_i, lab_c = X[:a.n], lab_i[:a.n], lab_c[:a.n]
    n, d = X.shape

    # rank: the cell's measured item budget, so the sweep sits on the paper's own operating point
    if a.rank:
        r = a.rank
    else:
        try:
            parts = json.load(open(os.path.join(DUMP, 'metric_std_parts',
                                               '%s__%s.json' % (a.enc, a.ds))))
            r = int(parts['cells']['inst']['metrics']['map10']['fit']['rstar_meas'])
        except Exception:
            r = 160
    print('%s/%s  n=%d d=%d  rank=%d' % (a.enc, a.ds, n, d, r))

    mu = X.mean(0)
    Xc = X - mu
    C = (Xc.T @ Xc) / max(len(Xc) - 1, 1)
    w, V = np.linalg.eigh(C)
    V = V[:, np.argsort(w)[::-1]][:, :r]
    Z = Xc @ V
    del Xc, X
    Z /= np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9)
    print('pca %d comps, %.1fs' % (r, time.time() - t_all))

    q = pick_queries(lab_i, a.queries)
    gi, gc = {}, {}
    for i, v in enumerate(lab_i):
        gi.setdefault(v, set()).add(i)
    for i, v in enumerate(lab_c):
        gc.setdefault(v, set()).add(i)
    print('queries %d' % len(q))

    res = {'enc': a.enc, 'ds': a.ds, 'n': n, 'dim_raw': d, 'rank': r,
           'queries': int(len(q)), 'M': M, 'cells': []}

    def cell(bits, mode):
        row = {'bits': bits, 'mode': mode,
               'bytes_graph': bytes_per_vec(bits, r), 'bytes_rerank': bits * r / 8.0}
        return row

    # ---- exact role: the controlled measurement (same ranking function, only the store changes)
    base_scores = exact_scores(Z)
    t0 = time.time()
    m_i = map_at_10(base_scores, q, lab_i, gi)
    m_c = map_at_10(base_scores, q, lab_c, gc)
    print('f32 exact      inst %.4f  cat %.4f  (%.0fs)' % (m_i, m_c, time.time() - t0))
    res['reference'] = {'bits': 32, 'map_inst': m_i, 'map_cat': m_c}
    base_cand = topk_ranks(base_scores, q, 100)

    for bits in ([] if a.skip_exact else [int(x) for x in a.bits.split(',')]):
        for mode in a.modes.split(','):
            if bits >= 8 and mode != 'tensor':
                continue          # at 8 bits the scale granularity changes the metric, not the size
            t0 = time.time()
            Zq = qdq(Z, bits, mode)
            m_i = map_at_10(exact_scores(Zq), q, lab_i, gi)
            m_c = map_at_10(exact_scores(Zq), q, lab_c, gc)
            ov, p95 = rerank_drift(Zq, q, base_cand)
            row = cell(bits, mode)
            row.update({'exact_inst': m_i, 'exact_cat': m_c,
                        'd_inst': 100 * (m_i / res['reference']['map_inst'] - 1),
                        'd_cat': 100 * (m_c / res['reference']['map_cat'] - 1),
                        'rerank_overlap': ov})
            print('b=%-2d %-8s exact inst %.4f (%+.1f%%)  cat %.4f (%+.1f%%)  rerank ov %.3f  (%.0fs)'
                  % (bits, mode, m_i, row['d_inst'], m_c, row['d_cat'], ov, time.time() - t0))
            res['cells'].append(row)

    # ---- graph role: does the representation survive HNSW, and what does it cost in QPS
    if not a.no_graph:
        import faiss
        combos = [(int(b), m) for b in a.graph_bits.split(',') for m in a.graph_modes.split(',')]
        for bits, gmode in combos:
            Zq = np.ascontiguousarray(qdq(Z, bits, gmode))
            faiss.omp_set_num_threads(48)
            idx = faiss.IndexHNSWFlat(r, M)
            idx.hnsw.efConstruction = EFC
            t0 = time.time()
            idx.add(Zq)
            tb = time.time() - t0
            faiss.omp_set_num_threads(1)
            idx.hnsw.efSearch = EF
            t0 = time.time()
            _, ids = idx.search(np.ascontiguousarray(Zq[q]), 11)
            tq = time.time() - t0
            # drop self (searched k+1, so one row loses exactly one hit)
            clean = np.empty((len(q), 10), dtype=np.int32)
            for i, qi in enumerate(q):
                clean[i] = [int(j) for j in ids[i] if int(j) != int(qi)][:10]
            # AP directly on the returned ids
            def ap_ids(ids_mat, per_row, groups):
                aps = []
                for i, qi in enumerate(q):
                    rel = groups[per_row[qi]]
                    h, m = 0, 0.0
                    for rank, j in enumerate(ids_mat[i][:10], 1):
                        if j in rel:
                            h += 1
                            m += h / rank
                    aps.append(m / min(len(rel), 10))
                return float(np.mean(aps))
            g_i = ap_ids(clean, lab_i, gi)
            g_c = ap_ids(clean, lab_c, gc)
            row = cell(bits, gmode + '-graph')
            row.update({'graph_inst': g_i, 'graph_cat': g_c, 'build_s': tb,
                        'qps': len(q) / max(tq, 1e-9),
                        'd_graph_inst': 100 * (g_i / res['reference']['map_inst'] - 1)})
            print('b=%-2d graph  inst %.4f (%+.1f%%)  cat %.4f  build %.1fs  QPS %.0f'
                  % (bits, g_i, row['d_graph_inst'], g_c, tb, row['qps']))
            res['cells'].append(row)
            del idx, Zq

    res['wall_s'] = time.time() - t_all
    if a.out:
        json.dump(res, open(a.out, 'w'), indent=1)
        print('wrote', a.out)
    print('wall %.0fs' % res['wall_s'])


if __name__ == '__main__':
    main()
