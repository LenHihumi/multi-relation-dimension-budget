"""Rank curves and budget for the text corpus: does the same spectrum give two budgets in text too?

Two relations over one embedding matrix: `parent` (the product a review is about -- the item
relation) and `cat` (the review category -- the coarse relation).  Everything else follows the
paper's protocol: centre, prefix of the principal basis, L2-renormalise inside the prefix, exact
search, mAP@10 with self excluded, 5% tolerance.

Also records `fid`, the top-10 agreement with the full-width ranking, so the label-free criterion of
Sec. VII-A can be evaluated on this corpus as well.

Usage:
    python3 sb_text_grid.py --corpus corpus.jsonl --emb emb --out text__wemm.json
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np


def log(m):
    print('[%s] %s' % (time.strftime('%H:%M:%S'), m), flush=True)


def groups_of(rows, key):
    g = {}
    for i, r in enumerate(rows):
        g.setdefault(r[key], []).append(i)
    labels = np.array([r[key] for r in rows])
    return g, labels


def top10(Z, q, Zall, exclude_self=True, k=11):
    """Exact top-k inner products for the query rows, self removed."""
    out = np.empty((len(q), k), dtype=np.int64)
    for s in range(0, len(q), 256):
        b = q[s:s + 256]
        sc = Z[b] @ Zall.T
        if exclude_self:
            sc[np.arange(len(b)), b] = -np.inf
        idx = np.argpartition(-sc, k, axis=1)[:, :k]
        v = np.take_along_axis(sc, idx, 1)
        out[s:s + 256] = np.take_along_axis(idx, np.argsort(-v, 1), 1)
    return out[:, :10]


def map10(top, relsets):
    """AP@10 with |R| capped at 10, i.e. the paper's convention."""
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
    ap.add_argument('--corpus', required=True)
    ap.add_argument('--emb', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--queries', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--protocol', action='store_true', help='also run the 2-probe rule in rho_V')
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.corpus, encoding='utf-8')]
    X = np.load(a.emb + '.npy').astype(np.float32)
    assert len(rows) == len(X), (len(rows), X.shape)
    n, d = X.shape
    log('corpus %d x %d' % (n, d))

    g_par, lab_par = groups_of(rows, 'parent')
    g_cat, lab_cat = groups_of(rows, 'cat')
    log('products %d (median size %d)  categories %d (median size %d)'
        % (len(g_par), int(np.median([len(v) for v in g_par.values()])),
           len(g_cat), int(np.median([len(v) for v in g_cat.values()]))))

    mu = X.mean(0)
    Xc = X - mu
    # per-direction eta^2 of the label partition: the paper's task-energy coordinate, needed because
    # the protocol's probe targets (rho_T = 0.12, 0.04) are stated on it, not on retained variance
    def eta2(Z, lab):
        """One-way ANOVA per direction, in closed form: SSB = sum_g S_g^2/n_g - n*gm^2."""
        _, inv, cnt = np.unique(lab, return_inverse=True, return_counts=True)
        n_ = Z.shape[0]
        gm = Z.mean(0)
        ssb = np.empty(Z.shape[1])
        for j in range(Z.shape[1]):
            Sg = np.bincount(inv, weights=Z[:, j], minlength=len(cnt))
            ssb[j] = (Sg ** 2 / cnt).sum() - n_ * gm[j] ** 2
        sst = ((Z - gm) ** 2).sum(0)
        return np.clip(ssb / np.maximum(sst, 1e-12), 0.0, 1.0)
    G = (Xc.T @ Xc) / np.float64(n - 1)
    w, V = np.linalg.eigh(G)
    o = np.argsort(w)[::-1]
    w, V = w[o], V[:, o]
    log('spectrum: top1 %.3f  explained90 at rank %d'
        % (w[0] / w.sum(), int(np.searchsorted(np.cumsum(w) / w.sum(), 0.90) + 1)))

    log('computing eta^2 per direction (two relations) ...')
    Zall_ = Xc @ V
    eta = {'parent': eta2(Zall_, lab_par), 'cat': eta2(Zall_, lab_cat)}
    rho_t = {}
    for k, e in eta.items():
        ww = w * e
        rho_t[k] = 1.0 - np.concatenate([[0.0], np.cumsum(ww)]) / ww.sum()

    rng = np.random.default_rng(a.seed)
    elig = np.array([i for i in range(n) if len(g_par[rows[i]['parent']]) > 1])
    q = np.sort(rng.choice(elig, size=min(a.queries, len(elig)), replace=False))
    log('queries %d' % len(q))

    # text embeddings concentrate far less than CLIP/DINOv3 ones (explained90 at rank ~650 on this
    # corpus), so the grid has to reach much wider than the image cells to see a crossing at all
    grid = [8, 12, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384,
            448, 512, 640, 768, 1024, 1536, 2048, 3072, 4096]
    grid = [r for r in grid if r <= d]
    rel_par = [set(g_par[rows[i]['parent']]) - {int(i)} for i in q]
    rel_cat = [set(g_cat[rows[i]['cat']]) - {int(i)} for i in q]

    # full-width reference, for fid
    Zf = Xc @ V
    Zf /= np.maximum(np.linalg.norm(Zf, axis=1, keepdims=True), 1e-9)
    Zf = np.ascontiguousarray(Zf, dtype=np.float32)
    top_full = top10(Zf, q, Zf)
    map_full = {'parent': map10(top_full, rel_par), 'cat': map10(top_full, rel_cat)}
    log('full width  mAP@10  item %.4f  cat %.4f' % (map_full['parent'], map_full['cat']))

    res = {'corpus': a.corpus, 'n': n, 'dim': d, 'n_query': int(len(q)),
           'groups': {'parent': len(g_par), 'cat': len(g_cat)},
           'full': map_full, 'grid': grid, 'curves': {}}
    # deleted variance after keeping r dimensions; index r, so pad for r=d
    rho_v = 1.0 - np.concatenate([[0.0], np.cumsum(w)]) / w.sum()

    for r in grid:
        Zr = Xc @ V[:, :r]
        Zr /= np.maximum(np.linalg.norm(Zr, axis=1, keepdims=True), 1e-9)
        Zr = np.ascontiguousarray(Zr, dtype=np.float32)
        top = top10(Zr, q, Zr)
        fid = np.mean([len(set(t) & set(f)) / 10.0 for t, f in zip(top, top_full)])
        row = {'r': r, 'rho_v': float(rho_v[r]), 'fid': float(fid),
               'rho_t_parent': float(rho_t['parent'][r]), 'rho_t_cat': float(rho_t['cat'][r])}
        for name, rel in (('parent', rel_par), ('cat', rel_cat)):
            row['map_' + name] = float(map10(top, rel))
        res['curves'].setdefault('rows', []).append(row)
        log('  r=%-4d fid %.3f  item %.4f  cat %.4f' % (r, fid, row['map_parent'], row['map_cat']))

    # budgets at 5% of each relation's own full width, and the fidelity budget at 5%
    cur = res['curves']['rows']
    for name in ('parent', 'cat'):
        vals = [1.0 - x['map_' + name] / map_full[name] for x in cur]
        hit = [x['r'] for x, v in zip(cur, vals) if v <= 0.05]
        res['rstar_' + name] = hit[0] if hit else None
        res['delta_' + name] = vals
    hit = [x['r'] for x in cur if 1.0 - x['fid'] <= 0.05]
    res['rstar_fid'] = hit[0] if hit else None
    ratio = (res['rstar_parent'] / res['rstar_cat']) if (res['rstar_parent'] and res['rstar_cat']) else None
    res['ratio'] = ratio
    log('r* item %s  cat %s  ratio %s  r*(fid) %s'
        % (res['rstar_parent'], res['rstar_cat'], ratio, res['rstar_fid']))
    if a.protocol:
        # the deployed rule, on a label-free coordinate (rho_V) with the label metric, so the number is
        # comparable with the image cells: two probes at rho_V near 0.12 and 0.04, then three branches
        for name in ('parent', 'cat'):
            rows_, dv = cur, res['delta_' + name]
            xi = np.array([max(x['rho_t_parent' if name == 'parent' else 'rho_t_cat'], 1e-6)
                           for x in rows_])
            i1 = int(np.argmin(np.abs(np.log(xi) - np.log(0.12))))
            i2 = int(np.argmin(np.abs(np.log(xi) - np.log(0.04))))
            if i1 > i2:
                i1, i2 = i2, i1
            y1, y2 = dv[i1], dv[i2]
            used = 2
            if y1 <= 0.05 and y2 <= 0.05:
                issued = rows_[i2]['r']
            elif y1 > 0.05 and y2 > 0.05:
                i = i2
                while i + 1 < len(rows_) and dv[i] > 0.05:
                    i += 1
                    used += 1
                issued = rows_[i]['r']
            else:
                L1, L2 = np.log(1 / max(y1, 1e-9) - 1), np.log(1 / max(y2, 1e-9) - 1)
                lx1, lx2 = np.log(xi[i1]), np.log(xi[i2])
                s_ = -(L2 - L1) / (lx2 - lx1)
                lr0 = lx1 + L1 / s_
                cross = np.exp(lr0 - np.log(1 / 0.05 - 1) / s_)
                cand = [x['r'] for x, xv in zip(rows_, xi) if xv <= cross]
                issued = cand[0] if cand else rows_[-1]['r']
            j = [k for k, x in enumerate(rows_) if x['r'] == issued][0]
            res.setdefault('protocol', {})[name] = {
                'probes': [rows_[i1]['r'], rows_[i2]['r']], 'issued': issued, 'evals': used,
                'delta_at_issued': float(dv[j]), 'ok': bool(dv[j] <= 0.05)}
            log('  protocol %-6s probes %s -> %d  (oracle %s)  loss %.3f  %s'
                % (name, (rows_[i1]['r'], rows_[i2]['r']), issued, res['rstar_' + name],
                   dv[j], 'ok' if dv[j] <= 0.05 else 'MISS'))
    json.dump(res, open(a.out, 'w'), indent=1)
    log('wrote ' + a.out)


if __name__ == '__main__':
    main()
