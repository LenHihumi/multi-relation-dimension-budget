"""Real issuance: measure Delta at the integer ranks below r*, never at a grid point.

sb_under.py replayed the "issue at 0.5x / 0.7x / 0.9x r*" experiment by *reading
the nearest grid point* of an existing curve, and by recovering that point's
per-query spread from the stored 5/95 band through a delta method with a borrowed
correlation.  That is a replay of a measurement, not a measurement: the rank that
was issued is not the rank that was evaluated, and the SE came from the very
approximation the experiment is supposed to replace.

This file does the real thing.  For each cell it takes r* from sequential.json
(arm S_fit), picks the three fractions of r*, rounds to an integer rank, nudges
off the grid if the integer happens to be a measured point, and runs the retrieval
at exactly that rank.  Delta = 1 - mAP@10(r) / mAP@10(d) with the full-dimension
value from metric_std_parts.  The SE is a paired bootstrap over the 1000 queries
(the same queries are used for r and for d, so the resampling must be paired).

Protocol is unchanged from sb_metric_std.run_pair: basis from pca_basis on the
*centred* gallery, Z_r = (X - mu) U_r renormalised per rank, mAP@10 normalised by
min(|R_q|, 10), relevance = the query's own relation group.

Usage:
    python sb_issue.py --smoke            # one point: clip/inshop/inst @ 96
    python sb_issue.py                    # the 15-point table -> issue.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

from sb_paths import SB_ROOT, SB_OUT, SB_CACHE, SB_MODELS, REPO  # paths resolve relative to the deposit
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sb_common                                                       # noqa: E402

# sb_common resolves SB and OUT through sb_paths, i.e. from the environment when it is
# set and from this directory's location otherwise.  The assignment below is kept so the
# two modules cannot disagree if one of them is imported before the environment is read.
sb_common.SB = SB_CACHE
sb_common.OUT = os.path.join(SB_OUT, 'spectral_budget')

from sb_common import load_emb, load_gallery, load_queries, pca_basis   # noqa: E402
from sb_grid import build_cells                                        # noqa: E402
from sb_metric_std import eval_scores, group_ids                        # noqa: E402

EPS = 0.05
ZCRIT = 1.96
N_BOOT = 2000
SEED = 0
PART = os.path.join(sb_common.OUT, 'metric_std_parts')

# the five cells: (encoder, dataset, relation)
CELLS = [
    ('siglip2_so400m', 'instre', 'cat'),     # over-supplied, decided  (|z|=7.68)
    ('siglip2_so400m', 'instre', 'inst'),    # over-supplied, edge      (|z|=3.26)
    ('clip_vitl14', 'instre', 'inst'),       # over-supplied, undecided (|z|=0.89)
    ('dinov3_vitl16', 'inshop', 'inst'),     # low mAP, undecided       (|z|=0.23)
    ('dinov3_vitl16', 'inshop', 'cat'),      # worst margin             (|z|=0.08)
]
FRACS = (0.5, 0.7, 0.9)


def log(m):
    print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)


def rstars():
    """r* of arm S_fit, keyed by cell id.  sequential.json lives with the results."""
    cands = [os.path.join(sb_common.OUT, 'sequential.json'),
             os.path.join(sb_common.OUT, 'results', 'sequential.json')]
    for p in cands:
        if os.path.exists(p):
            seq = json.load(open(p))
            return {c['id']: int(c['issued']) for c in seq['per_cell']['S_fit']}
    raise SystemExit('sequential.json not found in ' + ' / '.join(cands))


def off_grid(r, grid, d):
    """Nudge an integer rank off the measured grid.

    Both neighbours are one step away, so the direction is a choice; +1 is taken
    first because a higher rank means a *smaller* Delta, i.e. the nudge can only
    make the "not met" verdict harder, never easier.  The choice is recorded.
    """
    grid = set(int(g) for g in grid)
    if r not in grid:
        return r, 0
    for step in (1, -1, 2, -2):
        c = r + step
        if 2 <= c <= d and c not in grid:
            return c, step
    raise ValueError('no non-grid rank near %d' % r)


def paired_boot(mr, mf, n_boot=N_BOOT, seed=SEED):
    """SE of Delta = 1 - mean(mr)/mean(mf), resampling queries in pairs."""
    rng = np.random.default_rng(seed)
    n = mr.shape[0]
    idx = rng.integers(0, n, size=(n_boot, n))
    dr = 1.0 - mr[idx].mean(1) / mf[idx].mean(1)
    return float(dr.std(ddof=1)), float(np.percentile(dr, 2.5)), float(np.percentile(dr, 97.5))


def measure(enc, ds, rel, ranks, full_ref=None):
    """Return per-rank mAP@10 mean, per-query vector, and the full-dimension pair."""
    t0 = time.time()
    meta = load_gallery(ds)
    queries = load_queries(ds)
    X = load_emb(enc, ds)
    assert X.shape[0] == len(meta), (X.shape, len(meta))
    d = X.shape[1]

    mu, U, _ = pca_basis(X, device='cpu')
    log(f'  {enc}/{ds}: X{X.shape}  pca {time.time()-t0:.0f}s')

    Xg = X - mu
    Tf = Xg @ U
    nx = np.sqrt(np.maximum((Tf ** 2).cumsum(1), 1e-24))
    del Xg

    spec = build_cells(ds, meta, queries, np.arange(len(meta)))[rel]
    rows = np.asarray(spec['rows'])
    nrel = np.array([len(r) for r in spec['rel']], np.int32)
    gid, qgid = group_ids(meta, rel, ds, rows)
    assert np.all(np.bincount(gid)[gid[rows]] == nrel + 1), 'group/rel mismatch'
    K = int(min(X.shape[0] - 1, max(int(nrel.max()), 100)))

    QT = Tf[rows]
    Qn = np.sqrt(np.maximum((QT ** 2).cumsum(1), 1e-24))

    # full dimension, exactly as run_pair: cen, no truncation
    S = (QT / Qn[:, -1:]) @ (Tf / nx[:, -1:]).T
    S[np.arange(len(rows)), rows] = -np.inf
    mf = eval_scores(S, qgid, gid, nrel, K)['map10'].astype(np.float64)
    del S
    mf_mean = float(mf.mean())
    log(f'  {rel}: n_q={len(rows)} K={K} full map10={mf_mean:.4f}'
        + ('' if full_ref is None else f' (parts {full_ref:.4f})'))

    out = {}
    for r in ranks:
        Z = Tf[:, :r] / nx[:, r - 1:r]
        Qz = QT[:, :r] / Qn[:, r - 1:r]
        S = Qz @ Z.T
        S[np.arange(len(rows)), rows] = -np.inf
        mr = eval_scores(S, qgid, gid, nrel, K)['map10'].astype(np.float64)
        del S, Z, Qz
        out[r] = (mr, float(mr.mean()))
        log(f'    r={r:4d}  map10={out[r][1]:.4f}  '
            f'delta={1 - out[r][1]/mf_mean:+.4f}  ({time.time()-t0:.0f}s)')

    del Tf, QT, Qn, U
    return mf, mf_mean, out, d


def run(smoke=False):
    if smoke:
        enc, ds, rel = 'clip_vitl14', 'inshop', 'inst'
        parts = json.load(open(os.path.join(PART, f'{enc}__{ds}.json')))
        full_ref = float(parts['cells'][rel]['full']['map10'])
        mf, mf_mean, res, d = measure(enc, ds, rel, [96], full_ref)
        m = res[96][1]
        print(f'\nsmoke: {enc}/{ds}/{rel} r=96 map10={m:.4f} '
              f'expected 0.254 (parts full {full_ref:.4f} x keep 0.899)')
        print(f'       full map10 recomputed {mf_mean:.4f} vs parts {full_ref:.4f} '
              f'(diff {mf_mean - full_ref:+.4f})')
        return

    rstar = rstars()
    rows = []
    for enc, ds, rel in CELLS:
        cid = f'{enc}/{ds}/{rel}'
        parts = json.load(open(os.path.join(PART, f'{enc}__{ds}.json')))
        cellp = parts['cells'][rel]
        grid = [int(g) for g in parts['grid']]
        full_ref = float(cellp['full']['map10'])
        rs = rstar[cid]
        log(f'{cid}: r*={rs} full.map10={full_ref:.4f}')

        picks = []
        for frac in FRACS:
            tgt = frac * rs
            base = int(round(tgt))
            rank, step = off_grid(base, grid, parts['dim'])
            picks.append((frac, tgt, base, rank, step))
        ranks = sorted({p[3] for p in picks})
        mf, mf_mean, res, d = measure(enc, ds, rel, ranks, full_ref)

        for frac, tgt, base, rank, step in picks:
            mr, m = res[rank]
            delta = 1.0 - m / mf_mean
            se, lo, hi = paired_boot(mr, mf)
            z = (delta - EPS) / se if se > 0 else float('inf')
            if abs(z) > ZCRIT:
                state = 'not_met' if delta > EPS else 'met'
            else:
                state = 'undecidable'
            rows.append({'id': cid, 'r_star': rs, 'frac': frac,
                         'target': round(tgt, 2), 'rank': int(rank),
                         'rank_rounded': int(base), 'grid_shift': int(step),
                         'map10': round(m, 4), 'map10_full': round(mf_mean, 4),
                         'delta': round(delta, 4), 'se': round(se, 4),
                         'delta_lo': round(lo, 4), 'delta_hi': round(hi, 4),
                         'z': round(z, 2), 'state': state})
            log(f'  -> frac {frac}: rank {rank} (target {tgt:.1f}, nudge {step:+d}) '
                f'delta {delta:+.4f} se {se:.4f} z {z:+.2f} {state}')

    summary = {}
    for frac in FRACS:
        sel = [r for r in rows if r['frac'] == frac]
        summary[f'{frac:.1f}'] = {
            'n': len(sel),
            'decided': sum(1 for r in sel if r['state'] == 'not_met'),
            'met': sum(1 for r in sel if r['state'] == 'met'),
            'undecidable': sum(1 for r in sel if r['state'] == 'undecidable'),
            'mean_delta': round(float(np.mean([r['delta'] for r in sel])), 4)}

    out = {'eps': EPS, 'n_cells': len(CELLS), 'n_rows': len(rows),
           'n_boot': N_BOOT, 'protocol': 'cen', 'cut': 10, 'seed': SEED,
           'summary': {'by_frac': summary}, 'rows': rows}
    p = os.path.join(sb_common.OUT, 'issue.json')
    json.dump(out, open(p, 'w'), indent=1)

    print('\n%-30s %5s %5s %5s %8s %8s %7s  %s'
          % ('cell', 'frac', 'r*', 'rank', 'delta', 'se', 'z', 'state'))
    for r in rows:
        print('%-30s %5.1f %5d %5d %8.4f %8.4f %+7.2f  %s'
              % (r['id'][:30], r['frac'], r['r_star'], r['rank'],
                 r['delta'], r['se'], r['z'], r['state']))
    print('\ndecided not-met by issue depth:')
    for k, v in summary.items():
        print('  %s x r*: %d of %d  (met %d, undecidable %d, mean delta %.4f)'
              % (k, v['decided'], v['n'], v['met'], v['undecidable'], v['mean_delta']))
    print(f'\nwrote {p}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    run(a.smoke)


if __name__ == '__main__':
    main()
