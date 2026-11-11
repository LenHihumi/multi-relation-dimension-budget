"""Main grid: top-k fidelity / recall vs truncation rank r.

One JSON per (encoder, dataset).  Cells are (relation x k x truncation mode) and
each cell carries the r-curve, r*(95%) and r*(99%), query-level bootstrap CIs,
and the score margins that the theory in the paper links to those budgets.

Two metrics are reported side by side because they answer different questions:

  fid  top-k agreement with the full-dimension ranking.  Label-free; governed by
       the top-k boundary margin.  This is the metric of the existing ILIAS curve.
  rec  top-k recall against the relation's relevant set.  Task-dependent; governed
       by the relevant / non-relevant margin.  This is the metric that can differ
       between category, instance and copy retrieval.

Truncation modes:

  sub  Z_r = X U_r          lead-r coordinates of the PCA basis, mean retained
  cen  Z_r = (X - mu) U_r   PCA scores, mean removed

Usage:
    python3 sb_grid.py --encoder clip_vitl14 --dataset inshop --gpu 0
    python3 sb_grid.py --encoder clip_vitl14 --dataset inshop --sweep 1000,5000,25000
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np

from sb_common import (OUT, R_GRID, boot_rstar, energy_stats, l2n, load_emb,
                       load_gallery, load_queries, pca_basis, r_star, score,
                       topk_idx, transform)

KS = (1, 3, 10)
MAIN_K = 10


def log(m):
    print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)


def grouping(meta, keep, key):
    g = defaultdict(list)
    for i in keep:
        v = meta[i].get(key)
        if v is not None:
            g[v].append(int(i))
    return g


def build_cells(ds, meta, queries, keep):
    """Per-relation (query embeddings' source, relevant lists, rows to exclude).

    The label relations of In-shop form a granularity ladder on one gallery, with
    a common query set, so only the equivalence relation changes between cells:

        cat1  gender                       (2 classes)
        cat   gender x garment type       (23 classes, median 1.3k images)
        catf  x colourway                 (finer, ~1e2 images)
        inst  one physical product        (median 4.6 images)
        copy  one physical exposure       (1 image; augmented query)
    """
    keep_set = set(int(i) for i in keep)
    out = {}
    if ds in ('inshop',):
        ladder = [('cat1', 'cat1'), ('cat', 'cat'), ('catf', 'catf'), ('inst', 'ent')]
    elif ds == 'instre':
        # real-world third domain: semantic super-class (12), super-class x subset
        # (21), and the physical object itself (200 groups of 40 photos)
        ladder = [('cat1', 'cat1'), ('cat', 'cat'), ('inst', 'ent')]
    elif ds == 'sop':
        ladder = [('inst', 'ent')]
    elif ds == 'abo':
        # one gallery, two granularities: category (product type) and listing (item);
        # these are the two cells the 3.3x contrast is measured on for In-shop
        ladder = [('cat', 'cat'), ('inst', 'ent')]
    else:
        ladder = []

    qall = [int(i) for i in queries['inst'] if int(i) in keep_set]
    for name, key in ladder:
        g = grouping(meta, keep, key)
        q = [i for i in qall if len(g.get(meta[i][key], [])) >= 2]
        out[name] = {'src': 'gallery', 'rows': q,
                     'rel': [[j for j in g[meta[i][key]] if j != i] for i in q],
                     'excl': q}

    if ds in ('inshop', 'sop'):
        sel = [(k, int(c['src'])) for k, c in enumerate(queries['copy'])
               if int(c['src']) in keep_set]
        out['copy'] = {'src': 'copy', 'rows': [k for k, _ in sel],
                       'rel': [[s] for _, s in sel], 'excl': None}
    elif ds == 'roxford':
        rows, rel = [], []
        for qid, members in queries['rel'].items():
            qi = int(qid)
            m = [int(x) for x in members if int(x) in keep_set and int(x) != qi]
            if qi in keep_set and m:
                rows.append(qi)
                rel.append(m)
        out['copy'] = {'src': 'gallery', 'rows': rows, 'rel': rel, 'excl': rows}
    return out


def cell_stats(fid_pq, rec_pq, n_boot=1000, seed=0):
    out = {}
    for k in KS:
        e = {}
        for tag, pq, rel_mode in (('fid', fid_pq[k], False), ('rec', rec_pq[k], True)):
            curve = pq.mean(0)
            e[f'{tag}_curve'] = [round(float(x), 5) for x in curve]
            lo, hi = np.percentile(pq, 5, axis=0), np.percentile(pq, 95, axis=0)
            e[f'{tag}_band'] = [[round(float(a), 4) for a in lo],
                                [round(float(b), 4) for b in hi]]
            if rel_mode:
                rs5, c5 = r_star(R_GRID, curve, 0.05, True)
                rs1, c1 = r_star(R_GRID, curve, 0.01, True)
                e['rstar05'], e['rstar01'] = rs5, rs1
                e['censored05'], e['censored01'] = bool(c5), bool(c1)
                e['rstar05_ci'] = list(map(float, boot_rstar(R_GRID, pq, 0.05, True, n_boot, seed)))
                e['rstar01_ci'] = list(map(float, boot_rstar(R_GRID, pq, 0.01, True, n_boot, seed)))
                e['rec_ceiling'] = round(float(curve[-1]), 4)
            else:
                rs5, c5 = r_star(R_GRID, curve, 0.05, False)
                rs1, c1 = r_star(R_GRID, curve, 0.01, False)
                e['rstarfid05'], e['rstarfid01'] = rs5, rs1
                e['censored05'], e['censored01'] = bool(c5), bool(c1)
                e['rstarfid05_ci'] = list(map(float, boot_rstar(R_GRID, pq, 0.05, False, n_boot, seed)))
                e['rstarfid01_ci'] = list(map(float, boot_rstar(R_GRID, pq, 0.01, False, n_boot, seed)))
        out[f'k{k}'] = e
    return out


def run(enc, ds, gpu=0, sweep=(), tag=''):
    import torch
    dev = f'cuda:{gpu}' if (gpu >= 0 and torch.cuda.is_available()) else 'cpu'
    meta = load_gallery(ds)
    queries = load_queries(ds)
    X = load_emb(enc, ds)
    assert X.shape[0] == len(meta), (X.shape, len(meta))
    C = load_emb(enc, ds, '__copy') if ds in ('inshop', 'sop') else None
    log(f'{enc}/{ds}: gallery {X.shape} copy {None if C is None else C.shape}')

    perm = np.random.default_rng(7).permutation(len(meta))
    sizes = [len(meta)] + [n for n in sweep if n < len(meta)]
    res = {'encoder': enc, 'dataset': ds, 'dim': int(X.shape[1]),
           'n_gallery_full': int(X.shape[0]), 'grid': list(R_GRID), 'cells': {},
           'basis': {}}

    for N in sizes:
        keep = perm[:N] if N < len(meta) else np.arange(len(meta))
        row_of = {int(g): i for i, g in enumerate(keep)}
        Xs = X[keep]
        mu, U, evals = pca_basis(Xs, device=dev)
        est = energy_stats(evals, Xs.shape[1])
        # Project once into the full PCA basis and slice: projecting per r costs
        # sum_r d*r instead of d*d, which dominated the runtime at d=2048/4096.
        pos = {r: r for r in R_GRID}
        T = {}
        rho = {}
        for mode in ('sub', 'cen'):
            Xg = Xs if mode == 'sub' else l2n(Xs - mu)
            Tf = Xg @ U
            T[mode] = Tf
            sq = (Tf ** 2).cumsum(1)
            total = float(sq[:, -1].mean())
            rho[mode] = [round(float(1.0 - sq[:, min(r, X.shape[1]) - 1].mean() / total), 8)
                         for r in R_GRID]
        res['basis'][f'N{N}'] = {'energy': est, 'evals_head': [float(x) for x in evals[:32]],
                                 'evals_sum': float(evals.sum()), 'rho': rho}
        log(f'  N={N}: rank90={est["rank90"]} rank99={est["rank99"]} '
            f'PR={est["participation_ratio"]:.2f} top1={est["top1_frac"]:.3f}')

        cells = build_cells(ds, meta, queries, keep)
        suffix = '' if N == len(meta) else f'@N{N}'
        prep = {}
        for rel_name, spec in cells.items():
            rows = np.asarray(spec['rows'])
            excl = None if spec['excl'] is None else np.asarray(spec['excl'])
            excl_pos = None if excl is None else np.array([row_of[int(i)] for i in excl])
            prep[rel_name] = {
                'rows': rows, 'excl_pos': excl_pos,
                'base': (C[rows] if spec['src'] == 'copy' else Xs[rows]),
                'rel': spec['rel'], 'relsets': [set(r) for r in spec['rel']],
                'cell': {'n_query': int(len(rows)),
                         'rel_median': float(np.median([len(r) for r in spec['rel']])),
                         'src': spec['src'], 'margins': {}},
            }

        for mode in ('sub', 'cen'):
            Xg = Xs if mode == 'sub' else l2n(Xs - mu)
            acc = {}
            for rel_name, p in prep.items():
                Qg = p['base'] if mode == 'sub' else l2n(p['base'] - mu)
                Sf = _excl_score(Qg, Xg, p['excl_pos'], copy=True)
                p['cell']['margins'][mode] = margins_from(Sf, p['rel'], MAIN_K)
                n = len(p['rows'])
                acc[rel_name] = {
                    'Qg': Qg, 'ref': topk_idx(Sf, MAIN_K),
                    'fid': {k: np.zeros((n, len(R_GRID)), np.float32) for k in KS},
                    'rec': {k: np.zeros((n, len(R_GRID)), np.float32) for k in KS},
                }
                del Sf
            # gallery/query projections come from the one full-basis matmul above
            Tf = T[mode]
            QT = {k: a['Qg'] @ U for k, a in acc.items()}
            for ri, r in enumerate(R_GRID):
                rr = min(r, X.shape[1])
                Zr = l2n(Tf[:, :rr])
                for rel_name, a in acc.items():
                    Qz = l2n(QT[rel_name][:, :rr])
                    Sr = _excl_score(Qz, Zr, prep[rel_name]['excl_pos'])
                    top = topk_idx(Sr, MAIN_K)
                    n = len(prep[rel_name]['rows'])
                    sets = prep[rel_name]['relsets']
                    for k in KS:
                        a['fid'][k][:, ri] = [
                            len(set(top[i, :k].tolist()) & set(a['ref'][i, :k].tolist())) / k
                            for i in range(n)]
                        a['rec'][k][:, ri] = [
                            len(set(top[i, :k].tolist()) & sets[i]) / min(k, len(sets[i]))
                            for i in range(n)]
                    del Sr, top
                del Zr
            for rel_name, a in acc.items():
                c = prep[rel_name]['cell']
                c[mode] = cell_stats(a['fid'], a['rec'])
                log(f'  {rel_name}{suffix}/{mode}: rec@10={c[mode]["k10"]["rec_curve"][-1]:.3f} '
                    f'r*05={c[mode]["k10"]["rstar05"]} fid@10={c[mode]["k10"]["fid_curve"][-1]:.3f} '
                    f'fid r*05={c[mode]["k10"]["rstarfid05"]}')
        for rel_name, p in prep.items():
            res['cells'][rel_name + suffix] = p['cell']

    os.makedirs(os.path.join(OUT, 'grid'), exist_ok=True)
    p = os.path.join(OUT, 'grid', f'{enc}__{ds}{tag}.json')
    json.dump(res, open(p, 'w'), indent=1)
    log(f'wrote {p}')


def _excl_score(Q, X, excl_pos, copy=False):
    """Query-gallery inner products with the query itself removed when needed."""
    S = score(Q, X)
    if excl_pos is not None:
        if copy:
            S = S.copy()
        S[np.arange(Q.shape[0]), excl_pos] = -np.inf
    return S


def margins_from(S, rel, k):
    """Distribution of the two margins that govern the two budgets."""
    n = S.shape[0]
    g_rel, g_top, sig = [], [], []
    for i in range(n):
        s = S[i].astype(np.float64)
        r = rel[i]
        if r:
            rel_s = s[list(r)]
            mask = np.isfinite(s)
            mask[list(r)] = False
            if mask.any():
                g_rel.append(float(rel_s.min() - s[mask].max()))
        order = np.sort(s)[::-1]
        if order.size > k:
            g_top.append(float(order[k - 1] - order[k]))
        sig.append(float(s[np.isfinite(s)].std()))
    def q(a, p):
        return float(np.percentile(a, p)) if a else None
    return {'gap_rel_mean': float(np.mean(g_rel)) if g_rel else None,
            'gap_rel_q05': q(g_rel, 5), 'gap_rel_q50': q(g_rel, 50), 'gap_rel_q01': q(g_rel, 1),
            'gap_topk_q05': q(g_top, 5), 'gap_topk_q50': q(g_top, 50), 'gap_topk_q01': q(g_top, 1),
            'sigma': float(np.mean(sig))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--encoder', required=True)
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--sweep', default='')
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    run(a.encoder, a.dataset, a.gpu,
        [int(x) for x in a.sweep.split(',') if x], a.tag)


if __name__ == '__main__':
    main()
