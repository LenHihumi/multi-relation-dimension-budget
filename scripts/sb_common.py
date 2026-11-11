"""Shared geometry / retrieval utilities for the spectral-budget study.

Definitions used everywhere (and in the paper):

  gallery  X in R^{N x d}, rows L2-normalised (cosine geometry).
  basis    (mu, U) = eigen-decomposition of the *centered* covariance of X.
           U's columns are ordered by decreasing eigenvalue.

  two truncation operators, because they are not the same operation:
    sub  : Z_r = X U_r                 keep the r leading coordinates, the mean
                                       component that lives in span(U_r) survives
    cen  : Z_r = (X - mu) U_r          PCA scores, the whole mean is removed

  Both Z_r are L2-renormalised before scoring, matching rank_budget.py.

  fidelity@k(r) = |topk_r ∩ topk_full| / k          (agreement with the full-dim
                                                     ranking; label-free)
  recall@k(r)   = |topk_r ∩ R_q| / min(k, |R_q|)    (true retrieval quality)

  r*(eps) is reported against the *ceiling* of the same configuration:
    relative=False : min r s.t. curve(r) >= 1 - eps
    relative=True  : min r s.t. curve(r) >= (1 - eps) * curve(full)
  Fidelity always uses relative=False (its ceiling is 1).  Recall of a hard task
  can have a ceiling well below 1, so its budget is only meaningful relative.
"""
from __future__ import annotations

import json
import os

import numpy as np

from sb_paths import SB_CACHE, SB_OUT  # noqa: E402  (same directory, same resolution order)

SB = SB_CACHE           # embedding / gallery cache: SB_CACHE
OUT = os.path.join(SB_OUT, 'spectral_budget')   # run directory: SB_OUT/spectral_budget

R_GRID = (8, 12, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224,
          258, 320, 384, 448, 512, 640, 768, 896, 1075, 1280, 1536, 1792, 2048,
          3072, 4096)


def load_gallery(ds):
    path = os.path.join(SB, ds, 'gallery.jsonl')
    rows = []
    with open(path, encoding='utf-8') as fh:
        for ln in fh:
            rows.append(json.loads(ln))
    return rows


def load_queries(ds):
    return json.load(open(os.path.join(SB, ds, 'queries.json')))


def emb_path(enc, ds, tag=''):
    return os.path.join(OUT, 'emb', f'{enc}__{ds}{tag}.npy')


def load_emb(enc, ds, tag='', n=None):
    """Load an embedding matrix written either as .npy or as raw .bin + sidecar.

    Runs occasionally died between writing the .bin and its .json sidecar, so when
    the sidecar is missing the shape is recovered from the file size and the known
    gallery size (n rows, dim = bytes / (4 n)).
    """
    p = emb_path(enc, ds, tag)
    if os.path.exists(p):
        return np.load(p).astype(np.float32)
    raw = p.replace('.npy', '.bin')
    js = p.replace('.npy', '.json')
    if os.path.exists(js):
        meta = json.load(open(js))
        shape = tuple(meta['shape'])
    else:
        n = n or len(load_gallery(ds))
        size = os.path.getsize(raw)
        dim = size // (4 * n)
        assert dim * 4 * n == size, f'{raw}: {size} bytes not n*dim*4 for n={n}'
        shape = (n, dim)
    return np.fromfile(raw, dtype=np.float32).reshape(shape).astype(np.float32)


def have_emb(enc, ds, tag=''):
    p = emb_path(enc, ds, tag)
    return os.path.exists(p) or os.path.exists(p.replace('.npy', '.bin'))


def l2n(a):
    a = np.asarray(a, dtype=np.float32)
    return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)


def pca_basis(X, k=None, device='cpu'):
    """Eigen-decomposition of the centred second-moment matrix.

    Runs on GPU when available (A800: ~10 s for 52k x 4096) but accumulates in
    float32, which is enough for a spectrum that spans 6 orders of magnitude.
    """
    mu = X.mean(0, dtype=np.float64).astype(np.float32)
    d = X.shape[1]
    k = d if k is None else min(k, d)
    try:
        import torch
        if device.startswith('cuda') and torch.cuda.is_available():
            Xt = torch.from_numpy(np.ascontiguousarray(X)).to(device)
            mut = torch.from_numpy(mu).to(device)
            C = torch.zeros((d, d), dtype=torch.float32, device=device)
            for s in range(0, Xt.shape[0], 16384):
                Z = Xt[s:s + 16384] - mut
                C += Z.T @ Z
            C /= X.shape[0]
            C = (C + C.T) * 0.5
            ev, U = torch.linalg.eigh(C)
            ev = torch.flip(ev, (0,))[:k].cpu().numpy()
            U = torch.flip(U, (1,))[:, :k].cpu().numpy()
            del C, Xt
            torch.cuda.empty_cache()
            return mu, np.ascontiguousarray(U.astype(np.float32)), ev
    except Exception:  # noqa: BLE001  (fall back to numpy silently)
        pass
    C = np.zeros((d, d), dtype=np.float64)
    for s in range(0, X.shape[0], 8192):
        Z = (X[s:s + 8192] - mu).astype(np.float64)
        C += Z.T @ Z
    C /= X.shape[0]
    C = (C + C.T) * 0.5
    ev, U = np.linalg.eigh(C)
    order = np.argsort(ev)[::-1][:k]
    return mu, np.ascontiguousarray(U[:, order].astype(np.float32)), ev[order]


def transform(A, mu, U, r, mode='sub'):
    """Project A to rank r and L2-renormalise.  r=None keeps the full dimension."""
    if r is None:
        return A if mode == 'sub' else l2n(A - mu)
    P = U[:, :r]
    Z = A @ P if mode == 'sub' else (A - mu) @ P
    return l2n(Z)


def topk_idx(S, k, exclude=None):
    """Top-k indices per row.  `exclude[i]` (int) removes the query's own row."""
    n, N = S.shape
    k = min(k, N - (1 if exclude is not None else 0))
    if exclude is not None:
        S = S.copy()
        S[np.arange(n), exclude] = -np.inf
    part = np.argpartition(-S, k - 1, axis=1)[:, :k]
    rows = np.arange(n)[:, None]
    order = np.argsort(-S[rows, part], axis=1)
    return part[rows, order]


def score(Q, X, chunk=16384, Q_chunk=256):
    out = np.empty((Q.shape[0], X.shape[0]), dtype=np.float32)
    for i in range(0, Q.shape[0], Q_chunk):
        q = Q[i:i + Q_chunk]
        acc = []
        for s in range(0, X.shape[0], chunk):
            acc.append(q @ X[s:s + chunk].T)
        out[i:i + Q_chunk] = np.concatenate(acc, axis=1)
    return out


def recall_pq(top, rel, k):
    """Per-query recall@k, normalised by min(k, |R_q|)."""
    out = np.zeros(len(top), dtype=np.float32)
    for i, (t, r) in enumerate(zip(top, rel)):
        if not r:
            out[i] = np.nan
            continue
        out[i] = len(set(t.tolist()) & r) / min(k, len(r))
    return out


def fidelity_pq(top, ref, k):
    return np.array([len(set(t.tolist()) & set(r.tolist())) / k
                     for t, r in zip(top, ref)], dtype=np.float32)


def r_star(rgrid, curve, eps=0.05, relative=True):
    """Smallest r whose curve value reaches the tolerance band around the ceiling."""
    c = np.asarray(curve, dtype=np.float64)
    ref = c[-1]
    target = (1.0 - eps) * ref if relative else 1.0 - eps
    for r, v in zip(rgrid, c):
        if v >= target:
            return int(r), False
    return int(rgrid[-1]), True          # censored: never reached


def boot_rstar(rgrid, pq_mat, eps, relative, n_boot=1000, seed=0):
    """Bootstrap over queries; pq_mat is (n_query x n_r) of per-query values."""
    rng = np.random.default_rng(seed)
    n = pq_mat.shape[0]
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        vals.append(r_star(rgrid, pq_mat[idx].mean(0), eps, relative)[0])
    vals = np.array(vals)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def energy_stats(evals, total_dim):
    e = np.asarray(evals, dtype=np.float64)
    c = np.cumsum(e) / e.sum()
    out = {'top1_frac': float(e[0] / e.sum()),
           'participation_ratio': float(e.sum() ** 2 / (e ** 2).sum()),
           'rank90': int(np.searchsorted(c, 0.90) + 1),
           'rank99': int(np.searchsorted(c, 0.99) + 1),
           'rank999': int(np.searchsorted(c, 0.999) + 1)}
    return out


def margins(S, rel, k, exclude=None):
    """Score margins that govern the two budgets.

    gap_rel : mean over queries of (lowest relevant score - highest non-relevant)
              -> governs recall@k
    gap_topk: mean over queries of (k-th score - (k+1)-th score) -> governs
              fidelity@k
    sigma   : std of all scores (the natural scale of the problem)
    """
    n, N = S.shape
    g_rel, g_top, sig = [], [], []
    for i in range(n):
        s = S[i].astype(np.float64)
        if exclude is not None:
            s[exclude[i]] = -np.inf
        r = rel[i]
        if r:
            rel_scores = s[list(r)]
            mask = np.ones(N, bool)
            mask[list(r)] = False
            if exclude is not None:
                mask[exclude[i]] = False
            non = s[mask]
            if non.size:
                g_rel.append(rel_scores.min() - non.max())
        ss = np.sort(s)[::-1]
        if ss.size > k:
            g_top.append(ss[k - 1] - ss[k])
        sig.append(s[np.isfinite(s)].std())
    return {'gap_rel': float(np.mean(g_rel)) if g_rel else None,
            'gap_topk': float(np.mean(g_top)) if g_top else None,
            'sigma': float(np.mean(sig)),
            'n_query': int(n)}
