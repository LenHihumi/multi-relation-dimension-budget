"""Domain-standard retrieval metrics instead of recall@k.

Everything in the spectral-budget study so far is scored with recall@k, which
counts hits inside the top-k and is therefore blind to the *order* within the
top-k -- exactly the quantity truncation disturbs.  The benchmarks have their own
conventions (In-shop: mAP@10 + top-k accuracy; SOP: R@1/@10/@100; INSTRE: mAP +
P@k) and the embedding-compression literature reports nDCG@10 (BEIR) and mAP@10
(MS MARCO).  This script recomputes the quality-versus-rank curve under

    recall@10   |topk_r(10) & R_q| / min(10, |R_q|)           (the old metric)
    map10       AP with cutoff 10, normalised by min(|R_q|, 10)
    mapR        classic AP, cutoff |R_q|, normalised by |R_q|
    ndcg10      binary-relevance nDCG@10, log2 discount,
                IDCG = sum_{t<=min(|R_q|,10)} 1/log2(t+1)
    r1/r10/r100 top-k accuracy (hit at rank <= k)
    p10         precision@10 = hits in top 10 / 10

for the same centred (cen) protocol, gallery, rank grid and cells as grid/*.json,
so every number here is directly comparable to the recall-based study.

Delta is redefined per metric as

    Delta_m(r) = 1 - m(r) / m(d)

i.e. the quality deficit relative to *that metric's* full-dimension value, and
the law  Delta = 1/(1+(rho0/rho)^s)  is refitted per cell and per metric.  rho is
the deleted task energy (from theory__{enc}__{ds}.json, unchanged: it is a
property of the spectrum and the labels, not of the metric).

Usage
    python3 sb_metric_std.py --pair clip_vitl14:inshop --cells inst   # probe
    python3 sb_metric_std.py --pair clip_vitl14:inshop
    python3 sb_metric_std.py --report

Writes  outputs/diag/spectral_budget/metric_std_parts/{enc}__{ds}.json  (per pair)
        outputs/diag/spectral_budget/metric_std.json                  (--report)
        outputs/diag/spectral_budget/figs/F24_metric_std.png          (--report)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sb_common import OUT, R_GRID, load_emb, load_gallery, load_queries, pca_basis
from sb_grid import build_cells

FP32 = 4                    # bytes per retained float32 coordinate
EPS = 0.05                  # the law's tolerance on Delta
TARGET = 0.95               # retention target for the rule comparison
CUT = 10                    # the standard cutoff for mAP@10 / nDCG@10
KEEP = ('inst', 'cat')      # the two relations of the published comparison
METRICS = ('recall10', 'map10', 'mapR', 'ndcg10', 'r1', 'r10', 'r100', 'p10')
PARTS = os.path.join(OUT, 'metric_std_parts')
FIG = os.path.join(OUT, 'figs')


def log(m):
    print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)


# ---------------------------------------------------------------------------
# metric kernel
# ---------------------------------------------------------------------------
DISC = 1.0 / np.log2(np.arange(2, 202, dtype=np.float64))     # 1/log2(p+1), p>=1


def idcg_table(nmax=100):
    """IDCG@10 for a query with n relevant items: sum_{t<=min(n,10)} 1/log2(t+1)."""
    t = np.zeros(nmax + 1)
    for n in range(1, nmax + 1):
        k = min(n, CUT)
        t[n] = DISC[:k].sum()
    return t


IDCG = idcg_table()


def eval_scores(S, qgid, gid, nrel, K):
    """All metrics of one (cell, rank) from the score matrix.

    Relevance is *per query* (the query's relation group), so it is carried as a
    gallery-side group id compared against the query's own id -- not as a global
    "relevant for someone" mask, which would credit a query with another query's
    neighbours.

    Only the top-K entries per row are ever inspected: a relevant item ranked
    beyond K can contribute to neither AP@cutoff (cutoff <= |R_q| <= K) nor any
    of the @k metrics, so argpartition + a sort of K elements is exact here and
    avoids sorting the whole 52k gallery per query.
    """
    n = S.shape[0]
    K = int(min(K, S.shape[1]))
    part = np.argpartition(-S, K - 1, axis=1)[:, :K]
    rows = np.arange(n)[:, None]
    ps = S[rows, part]
    order = np.argsort(-ps, axis=1, kind='stable')
    idx = part[rows, order]
    hits = gid[idx] == qgid[:, None]                          # n x K bool
    p = np.arange(1, K + 1, dtype=np.float32)[None, :]
    cum = np.cumsum(hits, axis=1).astype(np.float32)

    nr = np.maximum(nrel, 1).astype(np.float32)
    in10 = hits & (p <= CUT)
    denom10 = np.minimum(nr, CUT)

    out = {}
    out['map10'] = np.where(in10, cum / p, 0.0).sum(1) / denom10
    inR = hits & (p <= nr[:, None])
    out['mapR'] = np.where(inR, cum / p, 0.0).sum(1) / nr
    dcg = np.where(in10, DISC[np.minimum(p.astype(int), 200) - 1], 0.0).sum(1)
    out['ndcg10'] = dcg / IDCG[np.minimum(nrel, 100)]
    out['recall10'] = hits[:, :CUT].sum(1).astype(np.float32) / denom10
    out['p10'] = hits[:, :CUT].sum(1).astype(np.float32) / CUT
    for k in (1, 10, 100):
        if k <= K:
            out[f'r{k}'] = hits[:, :k].any(1).astype(np.float32)
        else:
            out[f'r{k}'] = np.full(n, np.nan, np.float32)
    return out


# ---------------------------------------------------------------------------
# sigmoid fit (identical grid search to sb_sigmoid.py)
# ---------------------------------------------------------------------------
RHO0_GRID = 10 ** np.linspace(-4, 0.2, 169)
S_GRID = np.linspace(0.4, 6.0, 113)


def fit_sigmoid(rho, y):
    best = (np.inf, None, None)
    for r0 in RHO0_GRID:
        pred = 1.0 / (1.0 + (r0 / np.maximum(rho, 1e-12)) ** S_GRID[:, None])
        err = ((pred - y) ** 2).sum(axis=1)
        i = int(err.argmin())
        if err[i] < best[0]:
            best = (err[i], r0, S_GRID[i])
    _, r0, s = best
    for _ in range(3):
        for cand in np.linspace(r0 * 0.8, r0 * 1.25, 41):
            pred = 1.0 / (1.0 + (cand / np.maximum(rho, 1e-12)) ** s)
            e = ((pred - y) ** 2).sum()
            if e < best[0]:
                best = (e, cand, s)
        _, r0, s = best
        for cand in np.linspace(max(s * 0.85, 0.2), s * 1.18, 41):
            pred = 1.0 / (1.0 + (r0 / np.maximum(rho, 1e-12)) ** cand)
            e = ((pred - y) ** 2).sum()
            if e < best[0]:
                best = (e, r0, cand)
        _, r0, s = best
    return best


def r2_of(sse, y):
    tot = ((y - y.mean()) ** 2).sum()
    return float(1.0 - sse / tot) if tot > 0 else float('nan')


def fit_cell(rho, curve, full):
    """Fit Delta on (rho, 1 - m(r)/m(d)) and find the measured budget."""
    y = 1.0 - np.asarray(curve, float) / full
    yc = np.maximum(y, 0.0)                # truncation can help by a hair
    m = (yc < 0.98) & (rho > 1e-4)
    if m.sum() < 8:
        return None
    sse, r0, s = fit_sigmoid(rho[m], yc[m])
    ok = np.where(y <= EPS)[0]
    r_meas = int(R_GRID[ok[0]]) if len(ok) else int(R_GRID[-1])
    return dict(rho0=float(r0), s=float(s), r2=r2_of(sse, yc[m]),
                rstar_meas=r_meas, censored=bool(not len(ok)),
                n_neg=int((y < -1e-9).sum()), n_fit=int(m.sum()))


# ---------------------------------------------------------------------------
# one (encoder, dataset) pair
# ---------------------------------------------------------------------------
GROUP_KEY = {'inshop': {'cat1': 'cat1', 'cat': 'cat', 'catf': 'catf', 'inst': 'ent'},
             'instre': {'cat1': 'cat1', 'cat': 'cat', 'inst': 'ent'},
             'sop': {'inst': 'ent'}}


def group_ids(meta, rel_name, ds, rows):
    """Gallery-side and query-side relation-group ids, in build_cells' order.

    build_cells defines rel[i] as "every member of i's group except i", so the
    same partition reassembled from the metadata reproduces the relevance
    relation exactly (asserted by the caller).
    """
    key = GROUP_KEY[ds][rel_name]
    vals = [str(m.get(key)) for m in meta]
    _, gid = np.unique(vals, return_inverse=True)
    gid = gid.astype(np.int32)
    return gid, gid[rows]


def theory_path(enc, ds):
    return os.path.join(OUT, f'theory__{enc}__{ds}.json')


def run_pair(enc, ds, only=(), probe=False):
    t0 = time.time()
    meta = load_gallery(ds)
    queries = load_queries(ds)
    X = load_emb(enc, ds)
    assert X.shape[0] == len(meta), (X.shape, len(meta))
    th = json.load(open(theory_path(enc, ds)))
    lam = np.array(th['lambda'], float)
    d = X.shape[1]
    log(f'{enc}/{ds}: X{X.shape} d={d} theory d={len(lam)}')

    mu, U, evals = pca_basis(X, device='cpu')
    log(f'  pca done ({time.time()-t0:.1f}s)  lam0={evals[0]:.4g}')

    Xg = X - mu
    Tf = Xg @ U
    sq = (Tf ** 2).cumsum(1)                      # |(x-mu) . U_r|^2, in PCA space
    nx = np.sqrt(np.maximum(sq, 1e-24))

    spec = build_cells(ds, meta, queries, np.arange(len(meta)))
    cells = {k: v for k, v in spec.items() if k in KEEP and (not only or k in only)}
    ranks = [int(r) for r in R_GRID if int(r) <= d]
    res = dict(encoder=enc, dataset=ds, dim=int(d), n_gallery=int(X.shape[0]),
               grid=ranks, mode='cen', cut=CUT, cells={})

    for name, sp in cells.items():
        rows = np.asarray(sp['rows'])
        rel = sp['rel']
        nrel = np.array([len(r) for r in rel], np.int32)
        gid, qgid = group_ids(meta, name, ds, rows)
        # the group construction must reproduce build_cells' relation sets
        assert np.all(np.bincount(gid)[gid[rows]] == nrel + 1), 'group/rel mismatch'
        Qg = Xg[rows]
        QT = Qg @ U
        Qn = np.sqrt(np.maximum((QT ** 2).cumsum(1), 1e-24))   # in PCA space
        K = int(min(X.shape[0] - 1, max(int(nrel.max()), 100)))
        log(f'  cell {name}: n_q={len(rows)} nrel med={np.median(nrel):.0f} '
            f'max={nrel.max()} K={K} groups={len(set(gid.tolist()))}')

        curves = {k: np.zeros((len(rows), len(ranks)), np.float32) for k in METRICS}
        # rho for this (basis, relation) -- identical to sb_sigmoid.py
        om = lam * np.array(th['tasks'][name]['eta2'], float)
        cum = np.clip(np.cumsum(om) / om.sum(), 0.0, 1.0)
        rho = np.array([1.0 - cum[r - 1] for r in ranks])

        # full-dimension reference (no truncation at all; == r=d under cen)
        Zf = Tf / nx[:, -1:]
        Qz = QT / Qn[:, -1:]
        S = Qz @ Zf.T
        S[np.arange(len(rows)), rows] = -np.inf
        full_m = eval_scores(S, qgid, gid, nrel, K)
        del S, Zf
        full = {k: float(np.nanmean(v)) for k, v in full_m.items()}

        for ri, r in enumerate(ranks):
            Z = Tf[:, :r] / nx[:, r - 1:r]
            Qz = QT[:, :r] / Qn[:, r - 1:r]
            S = Qz @ Z.T
            S[np.arange(len(rows)), rows] = -np.inf
            m = eval_scores(S, qgid, gid, nrel, K)
            for k in METRICS:
                curves[k][:, ri] = m[k]
            del S
            if probe:
                log(f'    r={r:5d}  rec10={curves["recall10"][:, ri].mean():.4f} '
                    f'map10={curves["map10"][:, ri].mean():.4f} '
                    f'ndcg10={curves["ndcg10"][:, ri].mean():.4f} '
                    f'({time.time()-t0:.0f}s)')

        cell = dict(n_query=int(len(rows)), nrel_median=float(np.median(nrel)),
                    nrel_max=int(nrel.max()), full=full, rho=[round(x, 8) for x in rho],
                    metrics={})
        for k in METRICS:
            c = curves[k]
            mean = c.mean(0)
            lo, hi = np.percentile(c, 5, axis=0), np.percentile(c, 95, axis=0)
            e = dict(curve=[round(float(x), 6) for x in mean],
                     band=[[round(float(a), 5) for a in lo],
                           [round(float(b), 5) for b in hi]])
            if k in ('recall10', 'map10', 'mapR', 'ndcg10'):
                f = fit_cell(rho, mean, full[k])
                e['fit'] = f
                # r* of the *fitted* law, for the rule comparison
                if f:
                    dlt = 1.0 / (1.0 + (f['rho0'] / np.maximum(rho, 1e-12)) ** f['s'])
                    okf = np.where(dlt <= EPS)[0]
                    e['rstar_fit'] = int(ranks[okf[0]]) if len(okf) else int(ranks[-1])
            cell['metrics'][k] = e
            log(f'    {k:9s} full={full[k]:.4f} last={mean[-1]:.4f} '
                f'r*={e.get("fit", {}) and e["fit"]["rstar_meas"]} '
                f'r2={(e.get("fit") or {}).get("r2", float("nan")):.3f}')
        res['cells'][name] = cell

    res['wall_s'] = round(time.time() - t0, 1)
    os.makedirs(PARTS, exist_ok=True)
    p = os.path.join(PARTS, f'{enc}__{ds}.json')
    json.dump(res, open(p, 'w'), indent=1)
    log(f'wrote {p}  ({res["wall_s"]}s)')
    return res


# ---------------------------------------------------------------------------
# report: law refit, budget shift, rule comparison, standard numbers, figure
# ---------------------------------------------------------------------------
RULES = [
    ('law', 'Ours (law)', 'labels'),
    ('law_e01', 'Ours (law, 1% tol.)', 'labels'),
    ('energy90', 'Energy 90%', 'none'),
    ('energy95', 'Energy 95%', 'none'),
    ('energy99', 'Energy 99%', 'none'),
    ('scree', 'Scree knee', 'none'),
    ('pr', 'Participation ratio', 'none'),
    ('pr2', 'Participation ratio x2', 'none'),
    ('ff25', 'Fixed 0.25 d', 'none'),
    ('ff50', 'Fixed 0.5 d', 'none'),
    ('oracle', 'Oracle @95%', 'oracle'),
]
ORDER = [r[0] for r in RULES]
LABEL = {r[0]: r[1] for r in RULES}
NEEDS = {r[0]: r[2] for r in RULES}


def cum_energy(lam):
    return np.cumsum(lam) / lam.sum()


def energy_rank(lam, share):
    cum = cum_energy(lam)
    idx = int(np.searchsorted(cum, share, side='left'))
    return int(min(max(idx + 1, 1), len(lam)))


def pr_rank(lam):
    return float(lam.sum() ** 2 / (lam ** 2).sum())


def scree_rank(lam, smooth=5):
    d = len(lam)
    x = np.log(np.arange(1, d + 1, dtype=float))
    y = np.log(np.maximum(lam, 1e-30))
    if smooth and d > smooth:
        ker = np.ones(smooth) / smooth
        y = np.convolve(np.pad(y, (smooth // 2, smooth // 2), mode='edge'),
                        ker, mode='valid')[:d]
    dy = np.gradient(y, x)
    d2y = np.gradient(dy, x)
    kap = np.abs(d2y) / np.power(1.0 + dy ** 2, 1.5)
    return int(np.argmax(kap) + 1)


def rho_curve(lam, eta2):
    om = lam * np.array(eta2, float)
    return 1.0 - np.clip(np.cumsum(om) / om.sum(), 0.0, 1.0)


def law_rank(rho, rho0, s, eps=EPS, margin=1.0):
    thr = rho0 * (eps / (1.0 - eps)) ** (1.0 / s) / margin
    ok = np.where(rho <= thr)[0]
    return int(ok[0] + 1) if len(ok) else len(rho)


def snap_up(r, ranks, d):
    r = int(min(max(int(round(r)), 1), d))
    i = int(np.searchsorted(ranks, r, side='left'))
    return int(ranks[min(i, len(ranks) - 1)])


def load_parts(metric):
    th = {}
    for p in glob.glob(os.path.join(OUT, 'theory__*.json')):
        t = json.load(open(p))
        th[(t['encoder'], t['dataset'])] = t
    cells = []
    for p in sorted(glob.glob(os.path.join(PARTS, '*.json'))):
        g = json.load(open(p))
        t = th.get((g['encoder'], g['dataset']))
        if t is None:
            continue
        lam = np.array(t['lambda'], float)
        d = len(lam)
        ranks = np.array([int(r) for r in g['grid'] if int(r) <= d], int)
        for cname, c in g['cells'].items():
            e = c['metrics'].get(metric)
            if not e:
                continue
            cur = np.array(e['curve'][:len(ranks)], float)
            if len(cur) != len(ranks) or cur[-1] <= 0:
                continue
            cells.append(dict(
                enc=g['encoder'], ds=g['dataset'], rel=cname, d=d, ranks=ranks,
                curve=cur, full=float(cur[-1]), lam=lam,
                eta2=np.array(t['tasks'][cname]['eta2'], float),
                fit=e.get('fit'), rho=np.array(c['rho'], float),
                rstar_meas=(e.get('fit') or {}).get('rstar_meas'),
                rstar_fit=e.get('rstar_fit'),
                nrel_median=c['nrel_median']))
    return cells


def nested_share(vals, dom, lib, rel):
    """Sequential sums of squares for the nesting domain > library > relation."""
    y = np.asarray(vals, float)
    tot = ((y - y.mean()) ** 2).sum()
    out = {}
    resid = y - y.mean()
    for name, g in (('domain', dom), ('library', lib), ('relation', rel)):
        ss = 0.0
        for u in set(g.tolist()):
            idx = g == u
            if idx.sum() < 2:
                continue
            ss += idx.sum() * (resid[idx].mean()) ** 2
        out[name] = float(ss / tot) if tot > 0 else float('nan')
        for u in set(g.tolist()):
            idx = g == u
            resid[idx] = resid[idx] - resid[idx].mean()
    out['residual'] = float((resid ** 2).sum() / tot) if tot > 0 else float('nan')
    return out


def spearman(a, b):
    """Rank correlation, no scipy (average ranks for ties are not needed here)."""
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else float('nan')


def law_lookup(cells):
    """Geometric medians of (rho0, s) per (dataset, relation) for this metric."""
    acc = {}
    for c in cells:
        if c['fit']:
            acc.setdefault((c['ds'], c['rel']), []).append(c['fit'])
    look = {}
    for key, v in acc.items():
        look[key] = dict(
            rho0=float(np.exp(np.median(np.log([x['rho0'] for x in v])))),
            s=float(np.exp(np.median(np.log([x['s'] for x in v])))),
            n=len(v))
    return look


def build_rules(cells, look, metric):
    rows = []
    for c in cells:
        ranks, d, lam = c['ranks'], c['d'], c['lam']
        rho = rho_curve(lam, c['eta2'])
        full = c['full']
        keep = c['curve'] / full
        ok = np.where(keep >= TARGET)[0]
        r_or = int(ranks[ok[0]]) if len(ok) else d
        lp = look[(c['ds'], c['rel'])]
        native = {
            'law': law_rank(rho, lp['rho0'], lp['s']),
            'law_e01': law_rank(rho, lp['rho0'], lp['s'], eps=0.01),
            'energy90': energy_rank(lam, 0.90),
            'energy95': energy_rank(lam, 0.95),
            'energy99': energy_rank(lam, 0.99),
            'scree': scree_rank(lam),
            'pr': int(round(pr_rank(lam))),
            'pr2': int(round(2 * pr_rank(lam))),
            'ff25': int(round(0.25 * d)),
            'ff50': int(round(0.5 * d)),
            'oracle': r_or,
        }
        for name in ORDER:
            r = snap_up(native[name], ranks, d)
            i = min(int(np.searchsorted(ranks, r, side='left')), len(ranks) - 1)
            q = float(c['curve'][i])
            ret = q / full
            rows.append(dict(rule=name, enc=c['enc'], ds=c['ds'], rel=c['rel'],
                             metric=metric, d=d, r_native=int(native[name]),
                             r=int(ranks[i]), bytes=int(ranks[i] * FP32),
                             bytes_frac=float(ranks[i] / d), q=q,
                             retention=float(ret), meets=bool(ret >= TARGET),
                             shortfall_pts=float(max(0.0, (TARGET - ret) * 100.0)),
                             r_oracle=r_or,
                             ratio_to_oracle=float(ranks[i] / max(r_or, 1))))
    return rows


def agg(sub):
    if not sub:
        return None
    b = np.array([r['bytes'] for r in sub], float)
    meets = np.array([r['meets'] for r in sub], bool)
    miss = [r['shortfall_pts'] for r in sub if not r['meets']]
    hit = [r['bytes'] for r in sub if r['meets']]
    return dict(n=len(sub), hit_rate=float(meets.mean()), n_hit=int(meets.sum()),
                median_bytes=float(np.median(b)),
                median_bytes_frac=float(np.median([r['bytes_frac'] for r in sub])),
                median_bytes_at_target=float(np.median(hit)) if hit else None,
                median_shortfall_pts=float(np.median(miss)) if miss else 0.0,
                median_ratio_to_oracle=float(np.median([r['ratio_to_oracle'] for r in sub])),
                p90_ratio_to_oracle=float(np.percentile([r['ratio_to_oracle'] for r in sub], 90)))


def table(rows):
    return {name: agg([r for r in rows if r['rule'] == name]) for name in ORDER}


def report():
    os.makedirs(FIG, exist_ok=True)
    out = dict(eps=EPS, target=TARGET, fp32_bytes=FP32, cut=CUT,
               convention=dict(
                   map10='AP with cutoff 10, normalised by min(|R_q|,10)',
                   mapR='classic AP, cutoff |R_q|, normalised by |R_q|',
                   ndcg10='binary relevance, log2 discount, IDCG over min(|R_q|,10)',
                   delta='Delta_m(r) = 1 - m(r)/m(d), full dimension = no truncation'),
               per_metric={}, pairs=[], rule_compare={}, std_metrics={})

    # ---------------- 1. does the law survive? ------------------------------
    print('\n=== 1. law refit per metric ===')
    print(f'  {"metric":9s} {"n":>3s} {"meanR2":>7s} {"medR2":>6s} '
          f'{"geo rho0":>9s} {"geo s":>6s} {"r2>0.95":>8s}')
    cells_by_metric = {}
    for metric in ('recall10', 'map10', 'mapR', 'ndcg10'):
        cells = load_parts(metric)
        cells_by_metric[metric] = cells
        fits = [c for c in cells if c['fit']]
        r2 = np.array([c['fit']['r2'] for c in fits])
        r0 = float(np.exp(np.median(np.log([c['fit']['rho0'] for c in fits]))))
        ss = float(np.exp(np.median(np.log([c['fit']['s'] for c in fits]))))
        out['per_metric'][metric] = dict(
            n=len(cells), n_fit=len(fits),
            r2_mean=float(np.nanmean(r2)), r2_median=float(np.nanmedian(r2)),
            r2_min=float(np.nanmin(r2)), frac_r2_95=float(np.mean(r2 > 0.95)),
            geo_rho0=r0, geo_s=ss,
            cells=[dict(enc=c['enc'], ds=c['ds'], rel=c['rel'], d=c['d'],
                        rho0=c['fit']['rho0'], s=c['fit']['s'], r2=c['fit']['r2'],
                        rstar_meas=c['fit']['rstar_meas'],
                        rstar_fit=c['rstar_fit'], full=c['full'],
                        n_neg=c['fit']['n_neg'])
                   for c in fits])
        print(f'  {metric:9s} {len(cells):3d} {np.nanmean(r2):7.3f} '
              f'{np.nanmedian(r2):6.3f} {r0:9.4f} {ss:6.2f} '
              f'{np.mean(r2 > 0.95):8.0%}')

    # do (rho0, s) shift systematically between metrics? paired per cell
    key = {(c['enc'], c['ds'], c['rel']): c for c in cells_by_metric['recall10'] if c['fit']}
    shifts = {}
    for metric in ('map10', 'mapR', 'ndcg10'):
        dl0, ds_, dr = [], [], []
        for c in cells_by_metric[metric]:
            if not c['fit']:
                continue
            b = key.get((c['enc'], c['ds'], c['rel']))
            if not b or not b['fit']:
                continue
            dl0.append(np.log(c['fit']['rho0'] / b['fit']['rho0']))
            ds_.append(np.log(c['fit']['s'] / b['fit']['s']))
            dr.append(np.log(max(c['fit']['rstar_meas'], 1) / max(b['fit']['rstar_meas'], 1)))
        shifts[metric] = dict(
            n=len(dl0),
            log_rho0_median=float(np.median(dl0)) if dl0 else None,
            log_s_median=float(np.median(ds_)) if ds_ else None,
            log_rstar_median=float(np.median(dr)) if dr else None,
            frac_s_larger=float(np.mean(np.array(ds_) > 0)) if ds_ else None,
            frac_rho0_smaller=float(np.mean(np.array(dl0) < 0)) if dl0 else None)
        print(f'  vs recall: {metric:8s} med log rho0={shifts[metric]["log_rho0_median"]:+.3f} '
              f'med log s={shifts[metric]["log_s_median"]:+.3f} '
              f'med log r*={shifts[metric]["log_rstar_median"]:+.3f} '
              f's larger in {shifts[metric]["frac_s_larger"]:.0%}')
    out['shift_vs_recall'] = shifts

    # is the *structure* of (rho0, s) preserved?  Sequential sums of squares of
    # log rho0 / log s under the same nesting as the recall study, plus the rank
    # correlation of s between metrics (does the ordering of relations survive?).
    print('\n  decomposition of the fitted parameters (sequential shares)')
    print(f'  {"metric":9s} {"param":7s} {"domain":>7s} {"library":>8s} '
          f'{"relation":>9s} {"resid":>7s} {"spearman vs rec":>16s}')
    for metric in ('recall10', 'map10', 'mapR', 'ndcg10'):
        cells = [c for c in cells_by_metric[metric] if c['fit']]
        dom = np.array([c['ds'] for c in cells])
        lib = np.array([c['enc'] + '|' + c['ds'] for c in cells])
        rel = np.array([c['rel'] for c in cells])
        dec = {}
        for pname, key_ in (('log_rho0', 'rho0'), ('log_s', 's')):
            vals = np.log([c['fit'][key_] for c in cells])
            sh = nested_share(vals, dom, lib, rel)
            dec[pname] = sh
            rho_s = ''
            if metric != 'recall10':
                b = np.array([key[(c['enc'], c['ds'], c['rel'])]['fit'][key_] for c in cells])
                rho_s = f'{spearman(vals, np.log(b)):16.2f}'
            print(f'  {metric:9s} {pname:7s} {sh["domain"]:7.1%} {sh["library"]:8.1%} '
                  f'{sh["relation"]:9.1%} {sh["residual"]:7.1%} {rho_s}')
        out['per_metric'][metric]['decomposition'] = dec

    # ---------------- 2. does the budget shift? ----------------------------
    print('\n=== 2. budget: r*(5%) measured per metric ===')
    ratio = {}
    for a in ('map10', 'mapR', 'ndcg10'):
        ra, rb = [], []
        for c in cells_by_metric[a]:
            b = key.get((c['enc'], c['ds'], c['rel']))
            if not b:
                continue
            ra.append(c['fit']['rstar_meas'])
            rb.append(b['fit']['rstar_meas'])
        ra, rb = np.array(ra, float), np.array(rb, float)
        ratio[a] = dict(n=len(ra), median_ratio=float(np.median(ra / np.maximum(rb, 1))),
                        frac_smaller=float(np.mean(ra < rb)),
                        frac_larger=float(np.mean(ra > rb)),
                        frac_leq=float(np.mean(ra <= rb)),
                        median_rstar=float(np.median(ra)),
                        median_rstar_recall=float(np.median(rb)))
        print(f'  {a:8s} r*/r*_recall: median {ratio[a]["median_ratio"]:.2f}  '
              f'looser in {ratio[a]["frac_smaller"]:.0%} tighter in '
              f'{ratio[a]["frac_larger"]:.0%}  '
              f'(median r* {ratio[a]["median_rstar"]:.0f} vs {ratio[a]["median_rstar_recall"]:.0f})')
    out['budget_shift'] = ratio

    # ---------------- 3. rule comparison under mAP@10 ----------------------
    print('\n=== 3. rule comparison, target = mAP@10 retention 95% ===')
    out['rule_compare'] = {}
    for metric in ('map10', 'recall10'):
        cells = cells_by_metric[metric]
        look = law_lookup(cells)
        rows = build_rules(cells, look, metric)
        tab = dict(all=table(rows), inst=table([r for r in rows if r['rel'] == 'inst']),
                   cat=table([r for r in rows if r['rel'] == 'cat']))
        out['rule_compare'][metric] = dict(
            lookup={f'{k[0]}__{k[1]}': v for k, v in look.items()},
            summary=tab, cells=rows)
        for grp in ('all', 'inst', 'cat'):
            print(f'-- {metric} / {grp}')
            print(f'   {"rule":22s} {"hit":>6s} {"medB":>7s} {"B@hit":>7s} '
                  f'{"short":>6s} {"r/r_or":>6s}')
            for rn in ORDER:
                v = tab[grp][rn]
                bh = f'{v["median_bytes_at_target"]:7.0f}' if v['median_bytes_at_target'] else '      -'
                print(f'   {LABEL[rn]:22s} {v["hit_rate"]:6.0%} {v["median_bytes"]:7.0f} '
                      f'{bh} {v["median_shortfall_pts"]:6.2f} {v["median_ratio_to_oracle"]:6.2f}')

    # ---------------- 4. standard per-dataset metrics ----------------------
    print('\n=== 4. standard metrics at full dim and at the law budget (mAP@10) ===')
    cells_m = cells_by_metric['map10']
    look = out['rule_compare']['map10']['lookup']
    raw = {}
    for p in glob.glob(os.path.join(PARTS, '*.json')):
        g = json.load(open(p))
        d_ = g['dim']
        ranks_g = np.array([int(r) for r in g['grid'] if int(r) <= d_], int)
        for cname, c in g['cells'].items():
            raw[(g['encoder'], g['dataset'], cname)] = (ranks_g, c)
    std = {}
    for c in cells_m:
        rho = rho_curve(c['lam'], c['eta2'])
        lp = look[f'{c["ds"]}__{c["rel"]}']
        r_law = snap_up(law_rank(rho, lp['rho0'], lp['s']), c['ranks'], c['d'])
        ranks_g, cg = raw[(c['enc'], c['ds'], c['rel'])]
        i = int(min(np.searchsorted(ranks_g, r_law, side='left'), len(ranks_g) - 1))
        std.setdefault(c['ds'], []).append(dict(
            enc=c['enc'], rel=c['rel'], d=c['d'], r_law=int(r_law),
            bytes=int(r_law * FP32), bytes_frac=float(r_law / c['d']),
            full={m: float(cg['metrics'][m]['curve'][-1]) for m in METRICS},
            at_law={m: float(cg['metrics'][m]['curve'][i]) for m in METRICS},
            at_last={m: float(cg['metrics'][m]['curve'][-1]) for m in METRICS},
        ))
    out['std_metrics'] = std

    for ds in sorted(std):
        print(f'-- {ds}')
        for e in std[ds]:
            f, a = e['full'], e['at_law']
            if ds == 'inshop':
                print(f'   {e["enc"]:16s} {e["rel"]:5s} r*={e["r_law"]:5d} '
                      f'({e["bytes_frac"]:.0%} d): mAP@10 {f["map10"]:.3f}->{a["map10"]:.3f}  '
                      f'R@1 {f["r1"]:.3f}->{a["r1"]:.3f}  R@10 {f["r10"]:.3f}->{a["r10"]:.3f}')
            elif ds == 'sop':
                print(f'   {e["enc"]:16s} {e["rel"]:5s} r*={e["r_law"]:5d} '
                      f'({e["bytes_frac"]:.0%} d): R@1 {f["r1"]:.3f}->{a["r1"]:.3f}  '
                      f'R@10 {f["r10"]:.3f}->{a["r10"]:.3f}  R@100 {f["r100"]:.3f}->{a["r100"]:.3f}')
            else:
                print(f'   {e["enc"]:16s} {e["rel"]:5s} r*={e["r_law"]:5d} '
                      f'({e["bytes_frac"]:.0%} d): mAP {f["mapR"]:.3f}->{a["mapR"]:.3f}  '
                      f'P@10 {f["p10"]:.3f}->{a["p10"]:.3f}')

    pairs = {}
    for p in glob.glob(os.path.join(PARTS, '*.json')):
        g = json.load(open(p))
        pairs[f'{g["encoder"]}__{g["dataset"]}'] = g.get('wall_s')
    out['pairs'] = pairs
    out['wall_total_s'] = float(sum(v for v in pairs.values() if v))

    json.dump(out, open(os.path.join(OUT, 'metric_std.json'), 'w'), indent=1)
    print('\nwrote', os.path.join(OUT, 'metric_std.json'))
    make_fig(out, cells_by_metric)
    return out


def make_fig(out, cells_by_metric):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.size': 8.5, 'axes.linewidth': 0.8,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig = plt.figure(figsize=(13.8, 10.2))
    gs = fig.add_gridspec(3, 3, hspace=0.55, wspace=0.28,
                          height_ratios=[1.15, 1.0, 0.72],
                          left=0.065, right=0.99, top=0.945, bottom=0.045)

    # ---- A: Delta vs rho, three metrics, 4 representative cells -----------
    keys = [('clip_vitl14', 'inshop', 'inst'), ('clip_vitl14', 'inshop', 'cat'),
            ('siglip2_so400m', 'instre', 'inst'), ('dinov3_vitl16', 'instre', 'cat')]
    keys = [k for k in keys if any((c['enc'], c['ds'], c['rel']) == k
                                   for c in cells_by_metric['recall10'])]
    ncol = min(len(keys), 4)
    inner = gs[0, :].subgridspec(1, ncol, wspace=0.28)
    style = {'recall10': ('#1f4e79', 'o', 'recall@10'),
             'map10': ('#c53030', 's', 'mAP@10'),
             'ndcg10': ('#2f855a', '^', 'nDCG@10')}
    rhmin, rhmax = {}, {}
    for k in keys:
        c = next((x for x in cells_by_metric['map10']
                  if (x['enc'], x['ds'], x['rel']) == k), None)
        rp = np.array(c['rho']) if c else np.array([1e-4])
        rp = rp[rp > 0]
        rhmin[k] = float(rp.min()) if rp.size else 1e-4
        rhmax[k] = float(np.max(c['rho'])) if c else 1.0
    for j, k in enumerate(keys):
        ax = fig.add_subplot(inner[0, j])
        for m, (col, mk, lab) in style.items():
            c = next((x for x in cells_by_metric[m]
                      if (x['enc'], x['ds'], x['rel']) == k and x['fit']), None)
            if not c:
                continue
            y = np.maximum(1.0 - c['curve'] / c['full'], 0.0)
            ax.scatter(c['rho'], y, s=9, color=col, marker=mk, alpha=0.55,
                       linewidths=0, label=lab if j == 0 else None)
            rg = np.logspace(np.log10(max(c['rho'].min(), 1e-6)),
                             np.log10(c['rho'].max()), 200)
            dlt = 1.0 / (1.0 + (c['fit']['rho0'] / rg) ** c['fit']['s'])
            ax.plot(rg, dlt, color=col, lw=1.1, alpha=0.9)
        ax.axhline(EPS, color='k', ls=':', lw=0.9)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim(8e-3, 1.5)
        ax.set_xlim(rhmin[k] * 0.8, rhmax[k] * 1.25)
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.tick_params(which='minor', length=0)
        ax.set_xlabel(r'deleted task energy $\rho_T(r)$')
        if j == 0:
            ax.set_ylabel(r'$\Delta_m = 1 - m(r)/m(d)$')
            ax.legend(frameon=False, fontsize=6.6, loc='lower left')
        enc, ds, rel = k
        ax.set_title(f'{enc} / {ds} / {rel}', fontsize=7.4, loc='left')
        if j == 0:
            ax.text(-0.28, 1.10, 'A', transform=ax.transAxes,
                    fontweight='bold', fontsize=11)

    # ---- B: r*_mAP / r*_recall -------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    rs = []
    for a, lw, lab in (('map10', 1.4, 'mAP@10'), ('mapR', 1.0, 'mAP@R'),
                       ('ndcg10', 1.0, 'nDCG@10')):
        cellset = {(c['enc'], c['ds'], c['rel']): c for c in cells_by_metric[a] if c['fit']}
        rr = []
        for c in cells_by_metric['recall10']:
            if not c['fit']:
                continue
            o = cellset.get((c['enc'], c['ds'], c['rel']))
            if not o:
                continue
            rr.append(o['fit']['rstar_meas'] / max(c['fit']['rstar_meas'], 1))
        rs.append((a, np.array(rr)))
    bins = np.logspace(np.log10(0.2), np.log10(8), 22)
    for (a, arr), col, lab in zip(rs, ['#c53030', '#dd6b20', '#2f855a'],
                                  ['mAP@10', 'mAP@R', 'nDCG@10']):
        ax.hist(arr, bins=bins, color=col, alpha=0.55, label=lab)
    ax.axvline(1.0, color='k', lw=1.0)
    ax.set_xscale('log')
    ax.set_xticks([0.25, 0.5, 1, 2, 4, 8])
    ax.set_xticklabels(['0.25', '0.5', '1', '2', '4', '8'])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.tick_params(axis='x', which='minor', length=0)
    ax.set_xlim(0.2, 8)
    ax.set_xlabel(r'$r^*_{\rm metric}\,/\,r^*_{\rm recall@10}$')
    ax.set_ylabel('cells')
    ax.legend(frameon=False, fontsize=6.8)
    bs = out['budget_shift']['map10']
    ax.set_title(f'B  budget shift  (median {bs["median_ratio"]:.2f}; '
                 f'mAP@10 needs more dims in {bs["frac_larger"]:.0%})', loc='left',
                 fontweight='bold', fontsize=8.4)

    # ---- C: rule comparison under mAP@10 ---------------------------------
    ax = fig.add_subplot(gs[1, 1])
    tab = out['rule_compare']['map10']['summary']
    order = [n for n in ORDER if n != 'oracle']
    x = np.arange(len(order))
    w = 0.38
    std_m = out['std_metrics']
    ax.bar(x - w / 2, [tab['inst'][n]['hit_rate'] for n in order], w,
           color='#2b6cb0', label='instance')
    ax.bar(x + w / 2, [tab['cat'][n]['hit_rate'] for n in order], w,
           color='#dd6b20', label='category')
    ax.axhline(TARGET, color='k', ls=':', lw=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[n] for n in order], rotation=42, ha='right', fontsize=6.6)
    ax.set_ylabel('fraction of cells at 95% mAP@10')
    ax.set_ylim(0, 1.15)
    ax.legend(frameon=False, fontsize=6.6, loc='upper left', ncol=2)
    ax.set_title('C  rule accuracy under mAP@10', loc='left', fontweight='bold', fontsize=8.4)

    ax = fig.add_subplot(gs[1, 2])
    oracle = np.median([r['bytes'] for r in out['rule_compare']['map10']['cells']
                        if r['rule'] == 'oracle'])
    ax.bar(x - w / 2, [tab['inst'][n]['median_bytes'] for n in order], w,
           color='#2b6cb0', label='instance')
    ax.bar(x + w / 2, [tab['cat'][n]['median_bytes'] for n in order], w,
           color='#dd6b20', label='category')
    ax.axhline(oracle, color='k', ls='--', lw=1.0,
               label=f'oracle @95% ({oracle:.0f} B)')
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[n] for n in order], rotation=42, ha='right', fontsize=6.6)
    ax.set_ylabel('median bytes per vector (fp32)')
    ax.set_yticks([128, 256, 512, 1024, 2048, 4096])
    ax.set_yticklabels(['128', '256', '512', '1k', '2k', '4k'])
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.tick_params(axis='y', which='minor', length=0)
    ax.set_ylim(60, 6000)
    ax.legend(frameon=False, fontsize=6.2, loc='upper left', ncol=2)
    ax.set_title('D  cost under mAP@10', loc='left', fontweight='bold', fontsize=8.4)

    # ---- E: standard metrics, full vs law budget (text panel) ------------
    ax = fig.add_subplot(gs[2, :])
    ax.axis('off')
    DS3 = {'inshop': 'ish', 'instre': 'ins', 'sop': 'sop'}
    hdr = (f'{"encoder":15s} {"rel":8s} {"r*":>4s} {"%d":>4s}  '
           f'{"mAP10":>13s} {"R@1":>11s} {"R@10":>11s} {"R@100":>11s}')
    def block(ds_list, title):
        out = [title]
        for ds in ds_list:
            for e in std_m.get(ds, []):
                f, a = e['full'], e['at_law']
                out.append(
                    f'{e["enc"][:15]:15s} {DS3[ds] + "/" + e["rel"]:8s} '
                    f'{e["r_law"]:4d} {e["bytes_frac"]*100:3.0f}%  '
                    f'{f["map10"]:.3f}->{a["map10"]:.3f} '
                    f'{f["r1"]:.3f}->{a["r1"]:.3f} '
                    f'{f["r10"]:.3f}->{a["r10"]:.3f} '
                    f'{f["r100"]:.3f}->{a["r100"]:.3f}')
        return out
    left = ['E  standard metrics  (full dim -> at the law\'s budget', '   mAP@10 rule; r* in PCA dims)', '']
    left += block(['inshop'], hdr)
    right = ['', '', '']
    right += block(['instre', 'sop'], hdr)
    ax.text(0.0, 0.97, '\n'.join(left), va='top', ha='left', fontsize=6.4,
            family='DejaVu Sans Mono', transform=ax.transAxes)
    ax.text(0.53, 0.97, '\n'.join(right), va='top', ha='left', fontsize=6.4,
            family='DejaVu Sans Mono', transform=ax.transAxes)

    fig.text(0.065, 0.972, 'Does the spectral budget survive the standard metrics?',
             fontsize=11, fontweight='bold')
    p = os.path.join(FIG, 'F24_metric_std.png')
    fig.savefig(p, dpi=200)
    fig.savefig(p.replace('.png', '.pdf'))
    print('wrote', p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pair', default='')
    ap.add_argument('--cells', default='')
    ap.add_argument('--probe', action='store_true')
    ap.add_argument('--report', action='store_true')
    a = ap.parse_args()
    if a.report:
        report()
        return
    enc, ds = a.pair.split(':')
    only = tuple(x for x in a.cells.split(',') if x)
    run_pair(enc, ds, only, a.probe)


if __name__ == '__main__':
    main()
