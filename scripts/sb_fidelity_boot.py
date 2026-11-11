"""Does a cheaper measurement flip the verdict?  Bootstrap over queries.

`sb_fidelity.py` exports the per-query mAP@10 at the full dimension and at the
issued rank.  This script resamples those queries and asks the only question that
matters for a multi-fidelity evaluation: on a cell whose measured loss sits just
under the tolerance, does the subsampled interval cross epsilon?

Deliberately *not* reported: an average bias over cells.  An average would be
dominated by the safe cells and would hide exactly the failure this experiment is
about.

For each subsample size n:
  bias       mean over bootstrap of Delta_n  minus  Delta at the full 1000 queries
  interval   2.5 / 97.5 percentiles of Delta_n
  crosses    whether the interval contains epsilon
  width      interval width, compared with the margin (epsilon - Delta)

Usage: python sb_fidelity_boot.py --dir <dir with .npz> --out fidelity.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

EPS = 0.05
SIZES = (50, 100, 200, 400, 800, 1000)
B = 2000


def boot(full_pq, r_pq, n, reps=B, seed=0):
    rng = np.random.default_rng(seed)
    m = np.isfinite(full_pq) & np.isfinite(r_pq)
    f, r = full_pq[m], r_pq[m]
    N = len(f)
    n = min(n, N)
    out = np.empty(reps)
    for b in range(reps):
        idx = rng.integers(0, N, n)
        out[b] = 1.0 - r[idx].mean() / f[idx].mean()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--out', default='fidelity.json')
    args = ap.parse_args()

    res = {'eps': EPS, 'bootstrap_reps': B, 'sizes': list(SIZES), 'cells': {}}
    for p in sorted(glob.glob(os.path.join(args.dir, '*.npz'))):
        d = np.load(p)
        name = os.path.basename(p)[:-4]
        full = d['full'].astype(float)
        ranks = [int(x) for x in d['ranks']]
        r = ranks[0]
        rp = d['r%d' % r].astype(float)

        delta_full = 1.0 - np.nanmean(rp) / np.nanmean(full)
        margin = EPS - delta_full
        rows = {}
        for n in SIZES:
            bs = boot(full, rp, n)
            lo, hi = float(np.quantile(bs, .025)), float(np.quantile(bs, .975))
            rows[str(n)] = {
                'bias': float(bs.mean() - delta_full),
                'lo': lo, 'hi': hi, 'width': hi - lo,
                'crosses_eps': bool(lo <= EPS <= hi),
                'p_flip': float((bs > EPS).mean()),
                'width_over_margin': float((hi - lo) / margin) if margin > 0 else None}
        # smallest n whose interval is under a third of the margin
        need = None
        for n in SIZES:
            w = rows[str(n)]['width']
            if margin > 0 and w <= margin / 3.0:
                need = n
                break
        ok = np.isfinite(full) & np.isfinite(rp) & (full > 0)
        res['cells'][name] = {
            'rank': r, 'n_query': int(len(full)),
            'delta_full': float(delta_full), 'margin': float(margin),
            'per_query_sd_full': float(full[ok].std()),
            'per_query_sd_trunc': float(rp[ok].std()),
            'per_query_corr': float(np.corrcoef(full[ok], rp[ok])[0, 1]),
            'se_at_1000': float(full[ok].std() / np.sqrt(ok.sum())),
            'min_n_for_third_of_margin': need, 'by_size': rows}
        print('%-34s r=%-4d delta=%.4f margin=%.4f  need n<=%s'
              % (name[:34], r, delta_full, margin, need))
        for n in SIZES:
            x = rows[str(n)]
            print('   n=%-5d bias %+.4f  [%.4f, %.4f]  width %.4f  crosses %s'
                  % (n, x['bias'], x['lo'], x['hi'], x['width'], x['crosses_eps']))

    with open(os.path.join(args.dir, args.out), 'w') as f:
        json.dump(res, f, indent=1)
    print('\nwrote %s' % os.path.join(args.dir, args.out))


if __name__ == '__main__':
    main()
