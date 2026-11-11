"""Weighting control: does the choice of label-derived weight change the deletion order?

Reviewers asked for the Fisher (between/within) weighting of the same deletion-order scheme,
i.e. omega_i = lambda_i * eta_i^2 / (1 - eta_i^2) instead of lambda_i * eta_i^2.  Running that
as a full arm would need a fresh retrieval pass per cell, because a different order means a
different prefix subspace and mAP cannot be reused.  It is not needed: the two weightings are
monotone transforms of each other in eta^2, so the question "does the order change" is
answered by the two arrays we already store (eigenvalues and eta^2 per direction).

What this reports, per (encoder, relation) cell on In-shop:

  rho      Spearman rank correlation of the two weightings over all d directions
  overlap  size of the intersection of the top-160 directions under each weighting

160 is the item budget on CLIP/In-shop and the largest budget in the paper, so an overlap of
159-160 out of 160 means the deletion order is the same up to the last seat of the table.

Exit 0 if every cell has Spearman >= 0.99 and overlap >= 158; exit 2 otherwise, because a
regression here would mean the paper's defence against the "supervised PCA under another
name" objection no longer holds.

Not part of the 310 numeric checks: those cover published numbers, this covers a control that
is reported only in the rebuttal.  Needs `artifact/data/theory/` and nothing else.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get('SB_DATA', os.path.join(HERE, '..', 'data'))
TOP = 160


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    files = sorted(glob.glob(os.path.join(DATA, 'theory', 'theory__*__inshop.json')))
    if not files:
        print('no In-shop theory files under %s' % os.path.join(DATA, 'theory'))
        return 2
    worst = (1.0, TOP)
    print('%-18s %-5s %8s %8s' % ('encoder', 'task', 'rho', 'overlap'))
    for f in files:
        d = json.load(open(f))
        lam = np.asarray(d['lambda'], dtype=float)
        for task in ('inst', 'cat'):
            eta = np.asarray(d['tasks'][task]['eta2'], dtype=float)
            n = min(len(lam), len(eta))
            w_task = lam[:n] * eta[:n] ** 2
            w_fda = w_task / np.clip(1.0 - eta[:n] ** 2, 1e-9, None)
            rho = spearman(w_task, w_fda)
            ov = len(set(np.argsort(-w_task)[:TOP]) & set(np.argsort(-w_fda)[:TOP]))
            print('%-18s %-5s %8.4f %8d' % (d['encoder'], task, rho, ov))
            worst = (min(worst[0], rho), min(worst[1], ov))

    ok = worst[0] >= 0.99 and worst[1] >= TOP - 2
    print('\nworst: rho %.4f, overlap %d/%d -> %s' % (worst[0], worst[1], TOP,
                                                     'order preserved' if ok else 'ORDER CHANGES'))
    return 0 if ok else 2


if __name__ == '__main__':
    sys.exit(main())
