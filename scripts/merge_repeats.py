"""Collapse three repeat runs of a deployment cell into the file the paper prints.

The deployment table reports per-query p50 latency, and on a shared machine a
single run of that statistic is not stable: repeating the In-shop cell moves the
96-dimension (full) row by up to a third, because the two large-rank designs are
dominated by traversal and memory behaviour that other tenants perturb.  Bytes
and retention are deterministic, so the noise is only in the timing.

This script takes repeat_<cell>_r{1,2,3}.json, checks that everything except the
timing is bit-identical across repeats, and writes the canonical file with

  designs.<d>.per_relation.<r>.p50_ms   = median of the three runs
  designs.<d>.per_relation.<r>.p50_repeats_ms = the three values
  designs.<d>.p50_spread_pct            = (max-min)/median, per design
  designs.<d>.per_relation.<r>.qps_alone = median of the three

so a reader can see both the number used and how much it moved.

Usage: python3 merge_repeats.py [--cell inshop|text|instre_clip|instre_dinov3|instre_siglip ...]
"""
from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
TB = os.path.join(os.path.dirname(HERE), 'data', 'textbudget')

CANON = {
    'inshop': 'multi_e2e2__inshop.json',
    'text': 'multi_e2e2__text.json',
    'instre_clip_vitl14': 'multi_e2e2__instre_clip_vitl14.json',
    'instre_dinov3_vitl16': 'multi_e2e2__instre_dinov3_vitl16.json',
    'instre_siglip2_so400m': 'multi_e2e2__instre_siglip2_so400m.json',
}
DETERMINISTIC = ('designs.%s.ranks', 'designs.%s.bytes_per_vec', 'designs.%s.index_mb',
                 'designs.%s.graphs', 'designs.%s.build_s',
                 'designs.%s.per_relation.%s.retention', 'n', 'dim', 'r_var90', 'budgets')


def get(d, path):
    cur = d
    for k in path.split('.'):
        cur = cur[k]
    return cur


def repeats(prefix):
    fs = sorted(glob.glob(os.path.join(TB, 'repeat_%s_r*.json' % prefix)))
    return [json.load(open(f, encoding='utf-8')) for f in fs], fs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell', nargs='*', default=sorted(CANON))
    a = ap.parse_args()
    for cell in a.cell:
        ds, fs = repeats(cell)
        if len(ds) < 3:
            print('%-24s SKIP (only %d repeats)' % (cell, len(ds)))
            continue
        bad = []
        for path in DETERMINISTIC:
            if '%s' in path and path.count('%s') == 2:
                for design in ds[0]['designs']:
                    for rel in ds[0]['designs'][design]['per_relation']:
                        vals = [get(d, path % (design, rel)) for d in ds]
                        if len(set(map(repr, vals))) != 1:
                            bad.append((path % (design, rel), vals))
            elif '%s' in path:
                for design in ds[0]['designs']:
                    vals = [get(d, path % design) for d in ds]
                    # build_s is timing, not deterministic; skip it here
                    if path.endswith('build_s'):
                        continue
                    if len(set(map(repr, vals))) != 1:
                        bad.append((path % design, vals))
            else:
                vals = [get(d, path) for d in ds]
                if len(set(map(repr, vals))) != 1:
                    bad.append((path, vals))
        if bad:
            print('%-24s FAIL: non-timing fields differ across repeats' % cell)
            for b in bad[:5]:
                print('   ', b[0], b[1])
            raise SystemExit(1)

        out = copy.deepcopy(ds[0])
        for design, blk in out['designs'].items():
            spreads = []
            for rel, pr in blk['per_relation'].items():
                p = [d['designs'][design]['per_relation'][rel]['p50_ms'] for d in ds]
                q = [d['designs'][design]['per_relation'][rel]['qps_alone'] for d in ds]
                pr['p50_repeats_ms'] = [round(x, 4) for x in p]
                pr['p50_ms'] = round(st.median(p), 4)
                pr['qps_alone'] = round(st.median(q), 1)
                spreads.append(100.0 * (max(p) - min(p)) / max(st.median(p), 1e-9))
            blk['p50_spread_pct'] = round(max(spreads), 1)
            blk['p50_repeats'] = len(ds)
        out['merged_from'] = [os.path.basename(f) for f in fs]
        out['merge_note'] = ('p50_ms and qps_alone are medians over the repeat runs; '
                             'p50_repeats_ms holds the individual values; bytes and '
                             'retention are deterministic and asserted identical.')
        dst = os.path.join(TB, CANON[cell])
        json.dump(out, open(dst, 'w'), indent=1)
        sp = {d: out['designs'][d]['p50_spread_pct'] for d in out['designs']}
        print('%-24s -> %-38s spread %s' % (cell, os.path.basename(dst),
                                            ' '.join('%s=%.0f%%' % (k, v) for k, v in sp.items())))


if __name__ == '__main__':
    main()
