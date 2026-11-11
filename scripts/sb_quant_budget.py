"""Does the budget survive int8 after truncation?

The manuscript's boundary list says quantised scoring was not tested.  A deployment that truncates and then
stores one byte per coordinate is the common case, so the question is not academic: does the measured 5%
budget move, and do the response-surface parameters (rho0, s) drift?

Protocol, per (encoder, corpus, relation) cell:

  float32   the published path: (x-mu) V_r, L2 renormalised, exact search
  int8      the same prefix, symmetrically quantised to 8 bits with ONE scale for the whole tensor and
            dequantised for scoring (never per-dimension scales: those change the metric)

Reported per cell: the two curves, the two budgets at 5%, the two fitted (rho0, s) with their R^2, and the
multiplicative shift r*_int8 / r*_float32.  Aggregated: median shift, worst shift, and the fraction of cells
whose budget does not move by more than one grid step.

Usage (inside the container):
  python3 -u sb_quant_budget.py --out /path/quant_budget.json
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
import sb_retrieval as R  # noqa: E402

GRID = [8, 12, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 320, 384, 448, 512, 640]
EPS = 0.05


def budget(rho, dlt, r_grid):
    hit = [r for r, x in zip(r_grid, dlt) if x <= EPS]
    return int(hit[0]) if hit else int(r_grid[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(R.OUT, 'quant_budget.json'))
    ap.add_argument('--bits', type=int, default=8)
    ap.add_argument('--max-n', type=int, default=0)
    ap.add_argument('--resume', action='store_true')
    a = ap.parse_args()

    pairs = sorted(glob.glob(os.path.join(R.OUT, 'theory__*.json')))
    out = {'bits': a.bits, 'eps': EPS, 'grid': GRID, 'cells': []}
    t0 = time.time()
    for i, p in enumerate(pairs):
        th = json.load(open(p))
        enc, ds = th['encoder'], th['dataset']
        try:
            X = R.embeddings(enc, ds)
        except Exception as e:
            print('skip %s/%s: %s' % (enc, ds, e))
            continue
        n = len(X) if not a.max_n else min(len(X), a.max_n)
        rows = R.gallery(ds)[:n]
        per_inst, g_inst = R.groups(rows, 'ent')
        per_cat, g_cat = R.groups(rows, 'cat')
        rels = []
        if per_inst is not None:
            rels.append(('inst', per_inst, g_inst))
        if per_cat is not None:
            rels.append(('cat', per_cat, g_cat))
        if not rels:
            print('skip %s/%s: no relevance partition' % (enc, ds))
            continue
        q = R.query_rows(ds, n)
        lam_full = np.array(th['lambda'], float)
        k = min(len(lam_full), X.shape[1], max(GRID))
        mu, V, w = R.pca_basis(X[:n], k)
        lam = w / w.sum()
        Z = np.ascontiguousarray((np.asarray(X[:n], dtype=np.float32) - mu) @ V, dtype=np.float32)
        print('[%d/%d] %s/%s n=%d k=%d (%.0fs)'
              % (i + 1, len(pairs), enc, ds, n, k, time.time() - t0), flush=True)

        ranks = [r for r in GRID if r <= k]
        cum = np.cumsum(lam)
        rho = np.array([1.0 - cum[r - 1] for r in ranks])  # deleted energy at rank r

        for rel, per_row, grp in rels:
            curve = {'f32': [], 'i8': []}
            for r in ranks:
                A = np.ascontiguousarray(Z[:, :r])
                A = R.normalise(A)
                curve['f32'].append(R.map10(A, q, per_row, grp))
                curve['i8'].append(R.map10(R.quantise(A, a.bits), q, per_row, grp))
            full32, full8 = curve['f32'][-1], curve['i8'][-1]
            d32 = np.array([1.0 - x / full32 for x in curve['f32']])
            d8 = np.array([1.0 - x / full8 for x in curve['i8']])
            r32, r8 = budget(rho, d32, ranks), budget(rho, d8, ranks)
            p0a, sa, r2a = R.fit_loglog(rho, d32)
            p0b, sb, r2b = R.fit_loglog(rho, d8)
            cell = dict(enc=enc, ds=ds, rel=rel, n=n, ranks=ranks, rho=rho.tolist(),
                        curve_f32=curve['f32'], curve_i8=curve['i8'],
                        d_f32=d32.tolist(), d_i8=d8.tolist(),
                        rstar_f32=r32, rstar_i8=r8, shift=r8 / max(r32, 1),
                        rho0_f32=p0a, s_f32=sa, r2_f32=r2a,
                        rho0_i8=p0b, s_i8=sb, r2_i8=r2b,
                        max_abs_drop=float(np.max(np.abs(np.array(curve['f32']) - np.array(curve['i8'])))))
            out['cells'].append(cell)
            print('   %-4s r* %3d -> %3d (%.2fx)  rho0 %.2f->%.2f  s %.2f->%.2f  max|dmap| %.4f'
                  % (rel, r32, r8, cell['shift'], p0a, p0b, sa, sb, cell['max_abs_drop']), flush=True)

    shifts = [c['shift'] for c in out['cells']]
    out['summary'] = dict(n=len(shifts), median_shift=float(np.median(shifts)),
                          max_shift=float(np.max(shifts)), min_shift=float(np.min(shifts)),
                          frac_within_one_step=float(np.mean([abs(s - 1) < 0.25 for s in shifts])))
    print('summary', out['summary'])
    json.dump(out, open(a.out, 'w'), indent=1)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
