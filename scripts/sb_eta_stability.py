"""Is the task-energy coordinate stable when eta^2 is estimated from many small groups?

The reviewer's question: eta^2 is a one-way ANOVA effect size, and several partitions
here have thousands of small groups (In-shop item: 12,995 groups over 52,591 images,
~4 per group).  Small, unbalanced groups bias eta^2 upward, so does the budget move?

Three re-estimates of the same coordinate, no new retrieval measurements:

  full     eta^2 on every document                      (what the paper uses)
  half A   eta^2 on the documents of a random half of the groups
  half B   the other half
  shrunk   eta^2 corrected by (n-1)/(n-k), the unbiased-ish ANOVA correction,
           clipped at 0 -- the strongest regularisation a sceptic would ask for

For each variant the script rebuilds rho_T(r) on the shared rank grid, re-runs the
paper's decision rule against the *measured* Delta curve (the curve is empirical and
does not depend on eta^2; only the coordinate does), and reports how far the crossing
rank moves.  It also reports the Spearman correlation of omega between the two halves.

Usage
  python3 sb_eta_stability.py --cell inshop --enc clip_vitl14
  python3 sb_eta_stability.py --cell instre --enc clip_vitl14
  python3 sb_eta_stability.py --cell text
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

from sb_paths import SB_ROOT, SB_OUT, SB_CACHE, SB_MODELS, REPO  # paths resolve relative to the deposit
SB = SB_CACHE
OUT = SB_OUT + '/spectral_budget'
TXT = SB_OUT + '/textbudget'
EPS = 0.05
TARGETS = (0.12, 0.04)


def log(m):
    print('[%s] %s' % (time.strftime('%H:%M:%S'), m), flush=True)


def load(cell, enc):
    if cell == 'inshop':
        rows = [json.loads(l) for l in open(os.path.join(SB, 'inshop', 'gallery.jsonl'),
                                            encoding='utf-8')]
        base = os.path.join(OUT, 'emb', enc + '__inshop')
        rel = {'item': [r['ent'] for r in rows], 'cat': [r['cat'] for r in rows]}
    elif cell == 'instre':
        rows = [json.loads(l) for l in open(os.path.join(SB, 'instre', 'gallery_218.jsonl'),
                                            encoding='utf-8')]
        base = os.path.join(OUT, 'emb', enc + '__instre')
        rel = {'item': [r['ent'] for r in rows], 'cat': [r['cat'] for r in rows]}
    else:
        rows = [json.loads(l) for l in open(os.path.join(TXT, 'corpus.jsonl'), encoding='utf-8')]
        base = None
        rel = {'item': [r['parent'] for r in rows], 'cat': [r['cat'] for r in rows]}
    if base is None:
        X = np.load(os.path.join(TXT, 'emb_qwen8b.npy')).astype(np.float32)
    elif os.path.exists(base + '.npy'):
        X = np.load(base + '.npy').astype(np.float32)
    else:
        meta = json.load(open(base + '.json'))
        X = np.asarray(np.memmap(base + '.bin', dtype=np.float32, mode='r')
                       .reshape(meta['n'], meta['dim']), dtype=np.float32)
    return X, rel


def codes(labels):
    idx, uniq = {}, []
    for v in labels:
        if v not in idx:
            idx[v] = len(uniq)
            uniq.append(v)
    return np.array([idx[v] for v in labels], dtype=np.int64), len(uniq)


def eta2(Z, g, k):
    """One-way ANOVA eta^2 per column, computed with a sort + reduceat."""
    n = Z.shape[0]
    order = np.argsort(g, kind='stable')
    Zs = Z[order]
    gs = g[order]
    bounds = np.flatnonzero(np.diff(gs)) + 1
    starts = np.concatenate([[0], bounds])
    Sg = np.add.reduceat(Zs, starts, axis=0)
    cnt = np.diff(np.concatenate([starts, [n]]))
    gm = Z.mean(0)
    ssb = (Sg ** 2 / cnt[:, None]).sum(0) - n * gm ** 2
    sst = (Z ** 2).sum(0) - n * gm ** 2
    return np.clip(ssb / np.maximum(sst, 1e-12), 0.0, 1.0)


def rho_T(w, lam, ranks):
    tot = w.sum()
    cum = np.cumsum(w)
    return np.array([1.0 - cum[r - 1] / tot for r in ranks])


def curves_for(cell, enc):
    """Measured Delta curves per relation: metric_std_parts for images, textbudget for text."""
    if cell == 'text':
        d = json.load(open(os.path.join(TXT, 'text__qwen8b.json')))
        return {'item': (d['grid'], d['delta_parent']), 'cat': (d['grid'], d['delta_cat'])}
    d = json.load(open(os.path.join(OUT, 'metric_std_parts', '%s__%s.json' % (enc, cell))))
    out = {}
    for rel, key in (('item', 'inst'), ('cat', 'cat')):
        if key in d['cells']:
            full = d['cells'][key]['full']['map10']
            out[rel] = (d['grid'], [1 - m / full for m in d['cells'][key]['metrics']['map10']['curve']])
    return out


def spearman(a, b):
    """Rank correlation without scipy (the container has numpy and faiss only)."""
    def rk(v):
        o = np.argsort(v)
        r = np.empty(len(v), dtype=float)
        r[o] = np.arange(1, len(v) + 1)
        return r
    ra, rb = rk(a) - rk(a).mean(), rk(b) - rk(b).mean()
    return float((ra @ rb) / np.sqrt((ra @ ra) * (rb @ rb)))


def protocol(ranks, rho, delta):
    """The paper's rule, run in silico.  Returns (probe ranks, crossing rank)."""
    order = np.argsort(np.abs(rho - TARGETS[0]))
    r1 = ranks[int(order[0])]
    order = np.argsort(np.abs(rho - TARGETS[1]))
    r2 = ranks[int(order[0])]
    i1, i2 = ranks.index(r1), ranks.index(r2)
    y1, y2 = delta[i1], delta[i2]
    rho1, rho2 = rho[i1], rho[i2]
    if y1 <= EPS and y2 <= EPS:
        return (r1, r2), r2
    if y1 > EPS and y2 > EPS:
        return (r1, r2), None            # measure upward; not resolvable in silico
    # fit logit(Delta) = a + b log(rho) through the two probes
    l1, l2 = np.log(rho1), np.log(rho2)
    z1, z2 = np.log(y1 / (1 - y1)), np.log(y2 / (1 - y2))
    b = (z2 - z1) / (l2 - l1)
    a = z1 - b * l1
    rho_star = float(np.exp((np.log(EPS / (1 - EPS)) - a) / b)) if b > 0 else 0.0
    below = [r for r, x in zip(ranks, rho) if x <= rho_star]
    return (r1, r2), (min(below) if below else max(ranks))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell', required=True, choices=['inshop', 'instre', 'text'])
    ap.add_argument('--enc', default='clip_vitl14')
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    X, rel = load(a.cell, a.enc)
    n, d = X.shape
    log('%s/%s %d x %d' % (a.cell, a.enc, n, d))

    mu = X.mean(0)
    Xc = X - mu
    G = (Xc.T @ Xc) / np.float64(n - 1)
    w, V = np.linalg.eigh(G)
    o = np.argsort(w)[::-1]
    w, V = w[o], V[:, o]
    Z = Xc @ V
    log('pca done, top eigenvalue %.4g' % w[0])

    enc = 'qwen3vl_emb_8b' if a.cell == 'text' else a.enc
    measured = curves_for(a.cell, a.enc)
    ranks = list(next(iter(measured.values()))[0])

    res = {'cell': a.cell, 'enc': enc, 'n': n, 'dim': d, 'ranks': ranks, 'cells': {}}
    rng = np.random.default_rng(a.seed)

    for part, labels in rel.items():
        if part not in measured:
            continue
        delta = measured[part][1]
        g, k = codes(labels)

        # split the *groups* in half, so each half sees disjoint entities
        perm = rng.permutation(k)
        half = np.zeros(k, dtype=bool)
        half[perm[:k // 2]] = True
        mA = half[g]
        mB = ~mA

        variants = {}
        e_full = eta2(Z, g, k)
        variants['full'] = e_full
        gA, _ = codes([labels[i] for i in np.flatnonzero(mA)])
        gB, _ = codes([labels[i] for i in np.flatnonzero(mB)])
        wA = eta2(Z[mA], gA, int(gA.max()) + 1)
        wB = eta2(Z[mB], gB, int(gB.max()) + 1)
        variants['halfA'] = wA
        variants['halfB'] = wB
        variants['shrunk'] = np.clip(1.0 - (1.0 - e_full) * (n - 1) / max(n - k, 1), 0.0, 1.0)

        rho_s = spearman(wA, wB)
        outs = {}
        for name, e in variants.items():
            rho = rho_T(w[:len(e)] * e, w[:len(e)], ranks)
            probes, cross = protocol(ranks, rho, delta)
            outs[name] = dict(probes=[int(p) for p in probes],
                              crossing=(int(cross) if cross else None),
                              rho=[round(float(x), 4) for x in rho])
        ref = outs['full']['crossing']
        shift = {k2: (None if (v['crossing'] is None or ref is None)
                      else int(abs(v['crossing'] - ref))) for k2, v in outs.items()}
        res['cells'][part] = dict(kgroups=int(k), median_group=float(n / k),
                                  spearman_half=round(rho_s, 4),
                                  rank_shift=shift, variants=outs)
        log('%-5s k=%-6d spearman(halfA,halfB)=%.3f  crossing full=%s  shifts=%s'
            % (part, k, rho_s, ref, shift))

    json.dump(res, open(a.out, 'w'), indent=1)
    log('wrote ' + a.out)


if __name__ == '__main__':
    main()
