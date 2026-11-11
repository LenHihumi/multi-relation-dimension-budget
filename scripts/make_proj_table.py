#!/usr/bin/env python3
"""Reduce the projection re-evaluation into the one table the paper quotes.

The claim in Sec. VI-B is that the budget belongs to the basis, not to the
corpus, and one number carries it: with a projection trained on the relation,
the same 5% tolerance needs 12 dimensions on every In-shop encoder, where the
PCA prefix needs 96-320.

The numbers come from `projeval__<enc>__inst.json`, which is a *second*
implementation: it refits PCA at r_max=640 on the paper's own protocol, uses the
paper's 1000 queries, and scores both spaces against the same reference.  Its
header requires the PCA column to reproduce the paper's r* (160 for In-shop /
CLIP) before anything else is believed, and it does.

r* is the smallest grid rank whose mAP reaches 95% of the PCA prefix's
full-width mAP -- one reference for both curves, so the comparison is not
against the learned basis's own ceiling.

Outputs: tables/proj_budget.csv
Run:     python3 scripts/make_proj_table.py
"""
from __future__ import annotations

import csv
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
SRC = os.path.join(ROOT, "data", "proj")
TAB = os.path.join(ROOT, "tables")
TOL = 0.95


def rstar(curve, ref):
    """Smallest grid rank reaching TOL of the shared reference."""
    ok = [int(r) for r, v in curve.items() if v >= TOL * ref]
    return min(ok) if ok else None


def main():
    rows = []
    for f in sorted(glob.glob(os.path.join(SRC, "projeval__*__inst.json"))):
        d = json.load(open(f, encoding="utf-8"))
        ref = d["pca"]["all"]["map_rmax"]
        pca_curve = d["pca"]["all"]["curve"]
        w_curve = d["w"]["all"]["curve"]
        rows.append({
            "encoder": d["enc"],
            "corpus": d["ds"],
            "relation": d["relation"],
            "dim": d.get("dim") or "",
            "queries": d["queries"],
            "pca_rstar": rstar(pca_curve, ref),
            "learned_rstar": rstar(w_curve, ref),
            "pca_map_at_12": round(pca_curve.get("12", 0.0), 4),
            "learned_map_at_12": round(w_curve.get("12", 0.0), 4),
            "pca_map_rmax": round(ref, 4),
        })
    if not rows:
        print("no projeval files in data/proj")
        return 2
    with open(os.path.join(TAB, "proj_budget.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), lineterminator='\n')
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print("%-18s PCA r*=%3s  learned r*=%3s  mAP@12 %.3f -> %.3f"
              % (r["encoder"], r["pca_rstar"], r["learned_rstar"],
                 r["pca_map_at_12"], r["learned_map_at_12"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
