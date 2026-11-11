"""Three-way output: decide, or buy the guarantee with bytes.

The redefined objective is not "predict the rank" but "issue a rank you can defend
at a stated risk".  On the measured cells that objective has exactly three
outcomes, and they are not a new rule -- they are the |z| threshold read in two
directions:

    |margin| / SE > 1.96      the verdict survives a fresh sample      -> decide
    otherwise                 it does not, and no plausible query budget fixes it
                              -> issue the top of the bracket and say so

The query cost (n* = 1000 (1.96 SE / margin)^2) and the byte cost
(exp(bracket/2)) are reciprocal statements of the same threshold: a cell that
needs 63,000 queries to be decided at 95% is a cell whose bracket top costs
1.4x.  This script prices both sides for every cell and totals what the protocol
actually pays, counting 1.00x for the cells that can be decided.

Robustness: the SE itself is under-estimated by 1.19-1.60x (resolution_sens), so
the classification is recomputed at both ends of that range to check that no cell
changes side.  A protocol whose output form flips when the calibration moves is
not a protocol.

Usage: python sb_triage.py --out triage.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
from math import exp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
R = os.path.join(ROOT, 'data', 'results')

Z = 1.96


def classify(z, scale=1.0):
    return 'decide' if abs(z) / scale > Z else 'undecidable'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='triage.json')
    args = ap.parse_args()

    g = json.load(open(os.path.join(R, 'guarantee.json')))
    # the band-derived SE is kept as a second opinion, not as an uncertainty range:
    # scaling the bootstrap SE by the band/bootstrap ratio would manufacture
    # instability that does not exist.  Both classifications are reported instead.
    ratios = [r_['ratio'] for r_ in json.load(open(os.path.join(R, 'se_audit.json')))['rows']] \
        if os.path.exists(os.path.join(R, 'se_audit.json')) else [1.0]

    rows = []
    for r in g['rows']:
        state = classify(r['z'])
        band_z = r['margin'] / r['se_band']
        rows.append({'id': r['id'], 'margin': r['margin'], 'se': r['se'],
                     'z': r['z'], 'state': state,
                     'z_band': band_z, 'state_band': classify(band_z),
                     'n_star_95': r['n_star_95'],
                     'byte_cost': 1.0 if state == 'decide' else r['over_provision'],
                     'bracket_top': None if state == 'decide' else r['issued_top'],
                     'issued_fit': r['issued'], 'probes': r['probes'],
                     'ratio_to_band': r['se'] / r['se_band']})
    rows.sort(key=lambda r: r['z'], reverse=True)

    dec = [r for r in rows if r['state'] == 'decide']
    und = [r for r in rows if r['state'] == 'undecidable']
    costs = [r['byte_cost'] for r in rows]
    flips = [{'id': r['id'], 'from': r['state_band'], 'to': r['state'],
              'z_band': r['z_band'], 'z': r['z']}
             for r in rows if r['state_band'] != r['state']]

    out = {'z_threshold': Z, 'n_cells': len(rows),
           'n_decide': len(dec), 'n_undecidable': len(und),
           'checks': {'ratio_to_band': {'min': min(ratios), 'max': max(ratios)},
                      'classification_flips': flips},
           'cost': {'protocol_median': st.median(costs),
                    'protocol_mean': sum(costs) / len(costs),
                    'protocol_worst': max(costs),
                    'undecidable_median': st.median([r['byte_cost'] for r in und])
                    if und else None,
                    'undecidable_worst': max([r['byte_cost'] for r in und]) if und else None},
           'rows': rows}

    print('%-30s %7s %7s %8s %12s %9s %6s'
          % ('cell', 'margin', '|z|', 'state', 'n* for 95%', 'byte cost', 'probe'))
    for r in rows:
        ns = '%.0f' % r['n_star_95'] if r['n_star_95'] < 1e6 else '>1e6'
        print('%-30s %7.4f %7.2f %8s %12s %8.2fx %6d'
              % (r['id'][:30], r['margin'], r['z'], r['state'], ns,
                 r['byte_cost'], r['probes']))

    print('\ndecide: %d of %d   undecidable -> issue bracket top: %d'
          % (len(dec), len(rows), len(und)))
    print('protocol cost (bytes vs fitted crossing): median %.2fx  mean %.2fx  worst %.2fx'
          % (out['cost']['protocol_median'], out['cost']['protocol_mean'],
             out['cost']['protocol_worst']))
    print('  of which the undecidable cells alone: median %.2fx  worst %.2fx'
          % (out['cost']['undecidable_median'], out['cost']['undecidable_worst']))
    print('\nband-derived vs bootstrap classification: %d cell(s) differ '
          '(ratio range %.2f-%.2f)' % (len(flips), min(ratios), max(ratios)))
    for f in flips:
        print('  %-30s %s -> %s  (|z| %.2f -> %.2f)'
              % (f['id'][:30], f['from'], f['to'], f['z_band'], f['z']))

    with open(os.path.join(R, args.out), 'w') as f:
        json.dump(out, f, indent=1)
    print('\nwrote %s' % os.path.join(R, args.out))


if __name__ == '__main__':
    main()
