"""What does a coverage guarantee actually cost, in queries and in bytes?

Two proposals motivate this file: output a budget *interval* with a stated
guarantee instead of a point, and pick probe locations by information gain rather
than by a fixed ladder.  Both are answerable on data that already exists, and the
answer bounds how much either can buy.

Part 1 (queries).  A verdict is "decided at level alpha" when the margin exceeds
z_{1-alpha/2} standard errors, i.e. when a fresh sample of the same size would not
flip it with probability 1-alpha/2.  resolution.json gives margin and SE at 1000
queries for the 14 interpolated cells, so the query count needed for a 95% verdict
is n* = 1000 * (1.96 * SE / margin)^2.  Cells whose n* exceeds any plausible budget
cannot be decided by adding queries at all.

Part 2 (bytes).  Where the verdict cannot be decided, the only way to keep the
guarantee "issued rank >= r*" is to issue the top of the bracket instead of the
fitted crossing.  sequential.json records the bracket width in nats for each cell,
so the over-provision factor is exp(width/2) -- the price of converting an
undecidable verdict into a guaranteed one.

Part 3 (probes).  Two probes is the information-theoretic floor: the crossing of a
monotone curve is fixed by two evaluated points.  S_fit spends 2.37, so any smarter
placement of the probes can save at most 0.37 evaluations; this is a bound on the
whole "active design" direction, not a property of one heuristic.

Usage: python sb_guarantee.py --out guarantee.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
from math import exp, sqrt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
R = os.path.join(ROOT, 'data', 'results')

Z = 1.96
N0 = 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='guarantee.json')
    args = ap.parse_args()

    res = json.load(open(os.path.join(R, 'resolution.json')))
    seq = json.load(open(os.path.join(R, 'sequential.json')))

    # standard error: prefer the paired bootstrap measured at the issued rank
    # (se_audit), fall back to the band-derived delta-method value.  The two differ
    # by a corpus-dependent factor, so the source is recorded in the output.
    boot = {}
    sap = os.path.join(R, 'se_audit.json')
    if os.path.exists(sap):
        for r_ in json.load(open(sap))['rows']:
            boot[r_['id']] = r_['se_boot']
    eps = res['eps']
    fit = {c['id']: c for c in seq['per_cell']['S_fit']}

    rows = []
    for r in res['rows']:
        se = boot.get(r['id'], r['se_delta'])
        z = r['margin'] / se                       # |z| at 1000 queries
        n_star = N0 * (Z / z) ** 2 if z > 0 else float('inf')
        c = fit[r['id']]
        width = c['bracket_width_nats']
        over = exp(width / 2)                       # issue the top of the bracket
        rows.append({'id': r['id'], 'margin': r['margin'], 'se': se,
                     'se_band': r['se_delta'], 'se_source': 'bootstrap' if r['id'] in boot else 'band',
                     'z': z, 'n_star_95': n_star, 'issued': c['issued'],
                     'issued_top': c['issued'] * over, 'over_provision': over,
                     'probes': c['probes'], 'stop': c['stop']})

    tiers = {'decided at 1000': 0, 'needs <=5k': 0, 'needs <=100k': 0, 'beyond 100k': 0}
    for r in rows:
        n = r['n_star_95']
        if n <= N0:
            tiers['decided at 1000'] += 1
        elif n <= 5000:
            tiers['needs <=5k'] += 1
        elif n <= 100000:
            tiers['needs <=100k'] += 1
        else:
            tiers['beyond 100k'] += 1

    over = [r['over_provision'] for r in rows]

    # probes: read the 19-cell arms, not the 14-cell subset, so the floor comparison
    # is like for like.  O_meas starts at the true crossing -- the best any placement
    # heuristic could do -- and is therefore the measured ceiling on "active design".
    arms = seq['arms']
    probes = [c['probes'] for c in seq['per_cell']['S_fit']]
    mean_probes = sum(probes) / len(probes)
    hist = {}
    for p_ in probes:
        hist[p_] = hist.get(p_, 0) + 1
    oracle = arms['O_meas']['mean_evals']
    out = {'alpha': 0.05, 'z': Z, 'n_cells': len(rows), 'tiers': tiers,
           'mean_probes_19': mean_probes, 'probe_hist_19': hist,
           'mean_probes_14': sum(r['probes'] for r in rows) / len(rows),
           'probe_floor': 2.0, 'oracle_start_evals': oracle,
           'probe_headroom': mean_probes - oracle,
           'probe_headroom_frac': (mean_probes - oracle) / mean_probes,
           'over_provision': {'median': st.median(over), 'max': max(over),
                              'p90': sorted(over)[int(0.9 * (len(over) - 1))]},
           'rows': rows}

    print('cells: %d   (alpha = 5%%, z = %.2f, all margins/SE measured at 1000 queries)'
          % (len(rows), Z))
    print('%-30s %8s %8s %10s %10s %8s'
          % ('cell', 'margin', 'SE', '|z|', 'n* for 95%', 'over-prov'))
    for r in sorted(rows, key=lambda x: x['n_star_95']):
        n = r['n_star_95']
        ns = ('%.0f' % n) if n < 1e6 else '>1e6'
        print('%-30s %8.4f %8.4f %10.2f %10s %7.2fx'
              % (r['id'][:30], r['margin'], r['se'], r['z'], ns, r['over_provision']))

    print('\nhow many queries a 95%% verdict needs:')
    for k, v in tiers.items():
        print('  %-16s %2d of %d' % (k, v, len(rows)))

    print('\nprice of a guarantee where the verdict is undecidable (issue top of bracket):')
    print('  median %.2fx  p90 %.2fx  worst %.2fx  bytes'
          % (st.median(over), out['over_provision']['p90'], max(over)))
    print('\nprobes (19 cells): S_fit %.2f, histogram %s' % (mean_probes, hist))
    print('  oracle start (O_meas) %.2f -- this is the best any placement rule can do'
          % oracle)
    print('  so the whole "choose where to measure" direction is worth at most '
          '%.2f evaluations (%.0f%%)' % (mean_probes - oracle,
                                         100 * (mean_probes - oracle) / mean_probes))
    print('  cells already at the two-probe floor: %d of %d'
          % (hist.get(2, 0), len(probes)))

    with open(os.path.join(R, args.out), 'w') as f:
        json.dump(out, f, indent=1)
    print('\nwrote %s' % os.path.join(R, args.out))


if __name__ == '__main__':
    main()
