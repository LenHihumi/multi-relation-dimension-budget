"""How many cells would flip their verdict on a different query sample?

`fidelity.json` measures this properly, with a bootstrap, on three cells.  The
other eleven interpolated cells have no per-query export, but `metric_std_parts`
does carry the 5th/95th percentile band of the per-query metric at every rank, so
the per-query spread can be recovered without going back to the box.

Method: delta-method standard error of Delta = 1 - mean(r)/mean(full), using the
band-derived spread and a correlation borrowed from the three measured cells
(0.76-0.92, median 0.81).  Then P(flip) = Phi((Delta - eps)/SE).

Calibration: the same delta method is applied to the three cells that *do* have a
bootstrap, so the bias of the approximation is measured rather than assumed.  It
under-estimates the standard error by 16-37% on those three; the direction is
stated with the numbers rather than corrected away, because correcting would be a
second extrapolation on top of the first.

Usage: python sb_resolution.py --out resolution.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from math import erf, sqrt

import numpy as np

EPS = 0.05
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, 'data')
CORR = 0.81          # median of the three measured cells
Z95 = 1.645          # band is the 5th/95th percentile


def Phi(x):
    return 0.5 * (1 + erf(x / sqrt(2)))


def se_delta(sd_f, sd_r, corr, mean_f, mean_r, n):
    """Delta-method variance of 1 - mean(r)/mean(f)."""
    v = (sd_r ** 2 / mean_f ** 2
         + (mean_r ** 2) * (sd_f ** 2) / mean_f ** 4
         - 2 * mean_r * corr * sd_r * sd_f / mean_f ** 3) / n
    return sqrt(max(v, 1e-14))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='resolution.json')
    args = ap.parse_args()

    seq = json.load(open(os.path.join(DATA, 'results', 'sequential.json')))
    issued = {r['id']: r for r in seq['per_cell']['S_fit'] if not r['issued_is_probed']}
    err = {r['id']: r for r in seq['interpolation_error']['rows']}

    boot = {}
    for p in glob.glob(os.path.join(DATA, 'fidelity', '*.npz')):
        z = np.load(p)
        full = z['full'].astype(float)
        key = [k for k in z.files if k.startswith('r') and k != 'ranks'][0]
        r = z[key].astype(float)
        ok = np.isfinite(full) & np.isfinite(r) & (full > 0)
        name = os.path.basename(p)[:-4]
        boot[name] = {'sd_f': float(full[ok].std()), 'sd_r': float(r[ok].std()),
                      'corr': float(np.corrcoef(full[ok], r[ok])[0, 1]),
                      'n': int(ok.sum()), 'se': float(full[ok].std() / np.sqrt(ok.sum()))}

    rows = []
    for p in sorted(glob.glob(os.path.join(DATA, 'metric_std_parts', '*.json'))):
        d = json.load(open(p))
        for rel, c in d.get('cells', {}).items():
            cid = '%s/%s/%s' % (d['encoder'], d['dataset'], rel)
            if cid not in issued:
                continue
            key = '%s__%s__%s' % (d['encoder'], d['dataset'], rel)
            mm = c['metrics']['map10']
            grid = np.array(d['grid'])
            j = int(np.argmin(np.abs(grid - issued[cid]['issued'])))
            band_sd = (mm['band'][1][j] - mm['band'][0][j]) / (2 * Z95)
            full = float(c['full']['map10'])
            mr = float(mm['curve'][j])
            if key in boot:
                b = boot[key]
                se = se_delta(b['sd_f'], b['sd_r'], b['corr'], full, mr, b['n'])
                se_boot = b['se']
            else:
                se = se_delta(band_sd, band_sd, CORR, full, mr, 1000)
                se_boot = None
            rows.append({'id': cid, 'margin': err[cid]['margin'], 'full_map10': full,
                         'band_sd': band_sd, 'se_delta': se, 'se_boot': se_boot,
                         'p_flip_delta': float(Phi((err[cid]['actual'] - EPS) / se)),
                         'measured': key in boot})

    rows.sort(key=lambda x: -x['p_flip_delta'])
    expect = sum(r['p_flip_delta'] for r in rows)
    res = {'eps': EPS, 'n_cells': len(rows), 'assumed_corr': CORR,
           'rows': rows,
           'n_p_flip_gt_10': sum(1 for r in rows if r['p_flip_delta'] > 0.10),
           'n_p_flip_gt_25': sum(1 for r in rows if r['p_flip_delta'] > 0.25),
           'n_p_flip_gt_40': sum(1 for r in rows if r['p_flip_delta'] > 0.40),
           'expected_flips': float(expect),
           'expected_coverage': float((len(rows) - expect) / len(rows)),
           'calibration': {r['id']: {'se_delta': r['se_delta'], 'se_boot': r['se_boot'],
                                     'ratio': r['se_boot'] / r['se_delta']}
                           for r in rows if r['measured']}}

    print('%-30s %-8s %-7s %-8s %-9s %s'
          % ('cell', 'margin', 'full', 'SE', 'P(flip)', 'measured'))
    for r in rows:
        print('%-30s %-8.4f %-7.3f %-8.4f %-9.3f %s'
              % (r['id'][:30], r['margin'], r['full_map10'], r['se_delta'],
                 r['p_flip_delta'], 'bootstrap' if r['measured'] else 'band'))
    print('\nP(flip) > 10%%: %d of %d   > 25%%: %d   > 40%%: %d'
          % (res['n_p_flip_gt_10'], len(rows), res['n_p_flip_gt_25'],
             res['n_p_flip_gt_40']))
    print('expected flips %.2f of %d  ->  expected coverage %.1f%%'
          % (expect, len(rows), 100 * res['expected_coverage']))
    print('\ncalibration (delta-method SE / bootstrap SE, <1 means under-estimated):')
    for k, v in res['calibration'].items():
        print('  %-30s delta %.4f  boot %.4f  ratio %.2f'
              % (k[:30], v['se_delta'], v['se_boot'], v['ratio']))

    with open(os.path.join(DATA, 'results', args.out), 'w') as f:
        json.dump(res, f, indent=1)
    print('\nwrote %s' % os.path.join(DATA, 'results', args.out))


if __name__ == '__main__':
    main()
