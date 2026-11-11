"""Dump per-query AP@10 across the rank grid, so a per-query steepness is computable.

The paper reports one steepness per cell, fitted to the cell's mean loss curve. Whether
that number is an aggregate of a homogeneous per-query behaviour, or an average over a
wide distribution, cannot be answered from the released tables: they carry the mean only.
This script re-runs the exact-search grid with the same code path as
`sb_quant_budget.py` (same basis, same normalisation, same relevance sets) and writes the
per-query AP@10 for every (cell, rank).

Validation is part of the output, not an afterthought: for every cell and rank, the mean
over queries must reproduce the value printed in `tables/map10_19cells.csv` to 5e-4. The
tolerance is not tighter because the two means accumulate the same float32 dot products in
a different order (BLAS thread count), which moves the last digits by ~1e-4; a genuinely
different pipeline is off by percent, not by 1e-4.

Output: <out>/perquery/<enc>__<ds>__<rel>.npz  with
    ranks  (n_r,)            the rank grid actually used
    ap     (n_query, n_r)    per-query AP@10 at each rank
    rel    ()                relation name
and <out>/perquery/manifest.json with the validation table.

CPU only, deterministic. Run inside a container:
  docker run --rm --entrypoint python3 -v "$PWD":/work -w /work \\
    vllm/vllm-openai:v0.31.0 /work/scripts/sb_perquery.py --out /work/outputs/perquery
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import sb_retrieval as R  # noqa: E402
from sb_paths import ARTIFACT  # noqa: E402

EPS = 0.05
GRID = (8, 12, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 258, 320,
        384, 448, 512, 640, 768, 896, 1075, 1280, 1536, 1792, 2048, 3072, 4096)


def map10_pq(Z, q, per_row, grp, k=10, block=64):
    """Per-query AP@10; same arithmetic as R.map10, which returns the mean of this."""
    out = np.empty(len(q), dtype=np.float64)
    for s in range(0, len(q), block):
        qq = q[s:s + block]
        S = Z[qq] @ Z.T
        for i, qi in enumerate(qq):
            S[i, qi] = -np.inf
            top = np.argpartition(-S[i], k)[:k]
            top = top[np.argsort(-S[i][top])]
            rel = grp[per_row[qi]] - {int(qi)}
            h, ap = 0, 0.0
            for rank, j in enumerate(top, 1):
                if int(j) in rel:
                    h += 1
                    ap += h / rank
            out[s + i] = ap / min(len(rel), k) if rel else 0.0
    return out


def published(tables_dir):
    """The paper's grid: (enc, ds, rel, rank) -> map10."""
    path = os.path.join(tables_dir, 'map10_19cells.csv')
    d = {}
    with open(path, encoding='utf-8') as fh:
        for r in csv.DictReader(fh):
            d[(r['encoder'], r['dataset'], r['relation'], int(r['rank']))] = float(r['map10'])
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(R.OUT, 'perquery'))
    ap.add_argument('--tables', default=os.path.join(ARTIFACT, 'tables'))
    ap.add_argument('--max-n', type=int, default=0)
    ap.add_argument('--only', default='', help='substring filter on enc__ds, for smoke tests')
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, 'perquery'), exist_ok=True)
    ref = published(a.tables)

    pairs = sorted(glob.glob(os.path.join(R.OUT, 'theory__*.json')))
    manifest, bad = [], []
    t0 = time.time()
    for i, p in enumerate(pairs):
        th = json.load(open(p))
        enc, ds = th['encoder'], th['dataset']
        if a.only and a.only not in '%s__%s' % (enc, ds):
            continue
        try:
            X = R.embeddings(enc, ds)
        except Exception as e:
            print('skip %s/%s: %s' % (enc, ds, e), flush=True)
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
            continue
        q = R.query_rows(ds, n)
        lam_full = np.array(th['lambda'], float)
        k = min(len(lam_full), X.shape[1], max(GRID))
        mu, V, w = R.pca_basis(X[:n], k)
        lam = w / w.sum()
        Z = np.ascontiguousarray((np.asarray(X[:n], dtype=np.float32) - mu) @ V,
                                 dtype=np.float32)
        ranks = [r for r in GRID if r <= k]
        cum = np.cumsum(lam)
        print('[%d/%d] %s/%s n=%d k=%d (%.0fs)'
              % (i + 1, len(pairs), enc, ds, n, k, time.time() - t0), flush=True)

        for rel, per_row, grp in rels:
            apq = np.empty((len(q), len(ranks)), dtype=np.float32)
            for ri, r in enumerate(ranks):
                A = R.normalise(np.ascontiguousarray(Z[:, :r]))
                apq[:, ri] = map10_pq(A, q, per_row, grp).astype(np.float32)
            # validation against the published grid
            diffs = []
            for ri, r in enumerate(ranks):
                key = (enc, ds, rel, int(r))
                if key in ref:
                    diffs.append(abs(float(apq[:, ri].mean()) - ref[key]))
            worst = max(diffs) if diffs else float('nan')
            tag = 'ok' if (diffs and worst <= 5e-4) else 'MISMATCH'
            if diffs and worst > 5e-4:
                bad.append({'cell': '%s/%s/%s' % (enc, ds, rel), 'worst': worst})
            np.savez_compressed(os.path.join(a.out, 'perquery',
                                             '%s__%s__%s.npz' % (enc, ds, rel)),
                                ranks=np.array(ranks, int), ap=apq, rel=rel)
            manifest.append({'cell': '%s/%s/%s' % (enc, ds, rel), 'n_query': len(q),
                             'n_ranks': len(ranks), 'checked': len(diffs),
                             'worst_abs_diff': None if not diffs else worst,
                             'status': tag})
            print('   %-4s %-24s %s  checked %d ranks, worst |diff| %.2e'
                  % (rel, '%s/%s/%s' % (enc, ds, rel), tag, len(diffs),
                     worst if diffs else -1), flush=True)
            del apq
        del Z

    json.dump({'cells': manifest, 'mismatches': bad,
               'wall_s': round(time.time() - t0, 1)},
              open(os.path.join(a.out, 'perquery', 'manifest.json'), 'w'), indent=1)
    print('\n%d cells written, %d mismatched, %.0fs'
          % (len(manifest), len(bad), time.time() - t0))
    if bad:
        print('DO NOT USE: mismatched cells %s' % bad)


if __name__ == '__main__':
    main()
