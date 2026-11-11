#!/usr/bin/env python3
"""Figure 1, sized for the column it is printed in rather than for a screen.

The previous version was drawn at 6.2 x 3.9 inches and inserted at 0.74 of a
3.5-inch column, so every label was scaled to 42% of its nominal size: a 7.2 pt
footnote came out at 3 pt.  Two things were wrong and both are fixed here.

  1. The figure carried a title and a bottom line of text that repeated the
     caption ("one corpus, one encoder, one basis").  Removed -- the caption
     says it, and at 0.8 column width any extra line of 7 pt text is unreadable
     anyway.
  2. The canvas is now close to its printed size (3.0 in), so the nominal font
     sizes are the printed ones.  Nothing is scaled down by half.

Run:  python3 scripts/fig_motivation.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
SRC = os.path.join(ROOT, "data", "metric_std_parts", "clip_vitl14__inshop.json")
OUT = os.path.join(ROOT, "figs", "FIG1_motivation.png")

d = json.load(open(SRC, encoding="utf-8"))
grid = np.array([int(r) for r in d["grid"]], int)

# 3.0 in at 0.8 of a 3.5-in column is a 0.93 scale factor, so 8 pt prints as 7.4 pt.
fig, ax = plt.subplots(figsize=(3.1, 1.95))
for rel, col, lab in (("inst", "#d62728", "item retrieval"),
                      ("cat", "#1f77b4", "category retrieval")):
    c = d["cells"][rel]
    curve = np.array(c["metrics"]["map10"]["curve"], float)
    full = float(c["full"]["map10"])
    n = min(len(curve), len(grid))
    y = curve[:n] / full * 100
    ax.plot(grid[:n], y, "o-", ms=2.6, lw=1.2, color=col, label=lab)
    ok = np.where(y >= 95.0)[0]
    if len(ok):
        r95 = int(grid[ok[0]])
        ax.axvline(r95, color=col, ls=":", lw=0.9)
        ax.annotate("r*=%d" % r95, (r95, 86), color=col, fontsize=7.5,
                    ha="center", bbox=dict(fc="white", ec="none", alpha=0.85))

# the 95% line is explained by the caption; labelling it inside the axes only
# collides with the category curve in a 3-inch canvas
ax.axhline(95, color="grey", ls="--", lw=0.7)
ax.set_xscale("log")
ax.set_xlabel("retained dimensions (rank)", fontsize=8)
ax.set_ylabel("mAP@10 (% of full width)", fontsize=8)
ax.tick_params(labelsize=7.5)
ax.set_ylim(40, 104)
ax.grid(alpha=0.25, lw=0.5)
ax.legend(fontsize=7.2, loc="lower right", framealpha=0.9, handlelength=1.6)
fig.tight_layout(pad=0.2)
fig.savefig(OUT, dpi=300, bbox_inches="tight", pad_inches=0.02)
print("wrote", OUT, "at", fig.get_size_inches())
