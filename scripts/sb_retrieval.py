"""Shared loader + exact-search retrieval for the two new control experiments.

Paths resolve through `sb_paths`, i.e. from `SB_CACHE` / `SB_OUT` when they are set and from this
directory's location otherwise.  Everything here is CPU-only and deterministic.

Why a separate module rather than sb_common: this one adds exact-search retrieval (self-excluded
AP@10) on top of the loaders, which the grid scripts do not need.  It keeps the same conventions as
sb_common (cen truncation, L2 renormalisation, self-excluded AP@10) so the numbers are comparable
with the published cells.
"""
from __future__ import annotations

import json
import os

import numpy as np

from sb_paths import SB_ROOT, SB_OUT, SB_CACHE, SB_MODELS, REPO  # paths resolve relative to the deposit
SB = SB_CACHE
OUT = SB_OUT + '/spectral_budget'


def gallery(ds):
    return [json.loads(l) for l in open(os.path.join(SB, ds, 'gallery.jsonl'), encoding='utf-8')]


def queries(ds):
    return json.load(open(os.path.join(SB, ds, 'queries.json')))


def embeddings(enc, ds):
    base = os.path.join(OUT, 'emb', '%s__%s' % (enc, ds))
    if os.path.exists(base + '.npy'):
        return np.load(base + '.npy', mmap_mode='r')
    meta = json.load(open(base + '.json'))
    return np.memmap(base + '.bin', dtype=np.float32, mode='r').reshape(meta['n'], meta['dim'])


def theory(enc, ds):
    return json.load(open(os.path.join(OUT, 'theory__%s__%s.json' % (enc, ds))))


def pca_basis(X, k, block=20000):
    """Top-k eigenvectors of the centred covariance, computed in row blocks (1M rows x 768 fits)."""
    mu = X.mean(0)
    d = X.shape[1]
    G = np.zeros((d, d), dtype=np.float64)
    for s in range(0, len(X), block):
        B = np.asarray(X[s:s + block], dtype=np.float64) - mu
        G += B.T @ B
    G /= max(len(X) - 1, 1)
    w, V = np.linalg.eigh(G)
    o = np.argsort(w)[::-1]
    return mu, V[:, o][:, :k], w[o]


def groups(rows, key):
    """Integer-coded groups for one relation, or (None, None) if the corpus does not carry it.

    SOP ships a single relevance partition, so its rows have no `cat` field; passing None into np.unique
    compares Nones and raises.  Callers skip the relation instead.
    """
    labs = [r.get(key) for r in rows]
    if any(v is None for v in labs):
        return None, None
    per_row = np.array(labs)
    codes, inv = np.unique(per_row, return_inverse=True)
    g = {}
    for i, v in enumerate(inv):
        g.setdefault(int(v), set()).add(i)
    return inv, g


def query_rows(ds, n):
    q = [int(i) for i in queries(ds).get('inst', [])]
    return np.array([i for i in q if i < n], dtype=int)


def map10(Z, q, per_row, grp, k=10, block=64):
    """Exact mAP@10 on L2-normalised rows, self excluded, AP normalised by min(|R|,k)."""
    out = np.empty(len(q), dtype=np.float64)
    Zn = Z if Z.dtype != np.float32 else Z
    for s in range(0, len(q), block):
        qq = q[s:s + block]
        S = Zn[qq] @ Zn.T
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
    return float(out.mean())


def normalise(A):
    return A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-9)


def quantise(A, bits=8):
    """Symmetric per-tensor scalar quantisation, dequantised.  One scale for the tensor, never per
    dimension: per-dimension scales change the metric rather than approximate it."""
    qmax = float(2 ** (bits - 1) - 1)
    sc = float(np.abs(A).max()) / qmax
    return np.round(A / max(sc, 1e-12)) * max(sc, 1e-12)


def fit_loglog(rho, dlt):
    best = (None, None, np.inf)
    for r0 in np.logspace(-3, 0.3, 120):
        for s in np.linspace(0.5, 15.0, 120):
            err = float(np.sum((1.0 / (1.0 + (r0 / np.clip(rho, 1e-6, None)) ** s) - dlt) ** 2))
            if err < best[2]:
                best = (float(r0), float(s), err)
    r0, s, err = best
    r2 = 1.0 - err / max(float(np.sum((dlt - dlt.mean()) ** 2)), 1e-12)
    return r0, s, r2
