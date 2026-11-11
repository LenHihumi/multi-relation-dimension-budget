"""Fig. 2 of the ICDE submission: the response surface, one calibration per cell.

This replaces the figure an external review asked for ("19 cells and one fitted curve"). A single
pooled curve is not what the paper claims and not what the data shows: fitting logit(Delta) on
log(rho_T) across all 19 cells at once gives R^2 = 0.68, while fitting each cell separately gives a
median of 0.99 (the number the paper quotes). So the honest picture is the family: nineteen
sigmoids of the same functional form, each with its own position rho0 and steepness s, drawn here
from the released per-cell fits rather than refitted, plus the masked measurements they pass through.

Inputs (both are released tables, no private paths):
  tables/fits_19cells.csv    encoder,dataset,relation,rho0,s,r2        19 rows
  tables/map10_19cells.csv   encoder,dataset,relation,rank,rho_T,delta 486 rows

Output:
  FIG2_response_law.png      single-column figure, 3.4in wide, 600 dpi
  FIG2_response_law.json     every plotted number, so the figure can be audited without matplotlib

Usage:  python fig_response_law.py [--tables DIR] [--out PREFIX]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os

import numpy as np

CORPUS = {                      # colour per corpus, colour-blind safe (Okabe-Ito subset)
    'inshop': ('#0072B2', 'In-shop'),
    'sop':    ('#D55E00', 'SOP'),
    'instre': ('#009E73', 'INSTRE'),
    'abo':    ('#CC79A7', 'Amazon'),
}


def curve(rho0, s, rho):
    """Eq. (law): logit(Delta) = s log(rho/rho0), i.e. a two-parameter log-logistic."""
    z = s * (np.log(rho) - math.log(rho0))
    return 1.0 / (1.0 + np.exp(-z))


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument('--tables', default=os.path.join(here, '..', 'tables'))
    ap.add_argument('--out', default=os.path.join(here, '..', 'tables', 'FIG2_response_law'))
    a = ap.parse_args()

    fits = list(csv.DictReader(open(os.path.join(a.tables, 'fits_19cells.csv'))))
    rows = list(csv.DictReader(open(os.path.join(a.tables, 'map10_19cells.csv'))))
    assert len(fits) == 19, len(fits)

    pts = []
    for r in rows:
        d, rho = float(r['delta']), float(r['rho_T'])
        if not (1e-3 < d < 1 - 1e-3) or rho <= 1e-5:
            continue
        pts.append({'enc': r['encoder'], 'ds': r['dataset'], 'rel': r['relation'],
                    'rank': int(r['rank']), 'rho': rho, 'delta': d})

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'STIXGeneral', 'DejaVu Serif'],
        'font.size': 7,
        'axes.linewidth': 0.6,
        'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
        'xtick.major.size': 2.2, 'ytick.major.size': 2.2,
        'legend.frameon': False,
        'legend.handlelength': 1.4,
        'savefig.dpi': 600,
    })

    fig, ax = plt.subplots(figsize=(3.4, 1.72))
    grid = np.logspace(-3.0, 0, 400)
    seen = set()
    for f in fits:
        ds = f['dataset']
        col, lab = CORPUS.get(ds, ('#555555', ds))
        ax.plot(grid, curve(float(f['rho0']), float(f['s']), grid),
                color=col, lw=0.7, alpha=0.85, zorder=2,
                label=lab if ds not in seen else None)
        seen.add(ds)
    MARK = {'inst': ('o', 6.0), 'cat': ('^', 8.0)}
    for ds in CORPUS:
        for rel, (mk, sz) in MARK.items():
            sel = [p for p in pts if p['ds'] == ds and p['rel'].startswith(rel)]
            if not sel:
                continue
            ax.scatter([p['rho'] for p in sel], [p['delta'] for p in sel], marker=mk,
                       s=sz, facecolors=CORPUS[ds][0], edgecolors='none',
                       alpha=0.55, zorder=3)

    ax.set_xscale('log')
    ax.set_xlim(1e-3, 1.0)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(r'deleted task energy $\rho_T$', labelpad=2)
    ax.set_ylabel(r'quality loss $\Delta$', labelpad=2)
    ax.set_xticks([1e-3, 1e-2, 1e-1, 1e0])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])

    med = float(np.median([float(f['r2']) for f in fits]))
    ax.text(0.03, 0.62, '19 cells, median $R^2$ = %.3f' % med,
            transform=ax.transAxes, fontsize=6.2, va='center', ha='left')
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=CORPUS[d][0], lw=1.0, label=CORPUS[d][1])
               for d in ('inshop', 'sop', 'instre')]
    handles += [Line2D([], [], color='0.35', lw=0, marker='o', ms=3.0, label='item'),
                Line2D([], [], color='0.35', lw=0, marker='^', ms=3.2, label='category')]
    ax.legend(handles=handles, loc='upper left', fontsize=6.2, ncol=2,
              handletextpad=0.4, columnspacing=0.8, borderaxespad=0.2, labelspacing=0.22)
    fig.tight_layout(pad=0.25)
    figdir = os.path.join(here, '..', 'figs')
    os.makedirs(figdir, exist_ok=True)
    stem = os.path.join(figdir, os.path.basename(a.out))
    fig.savefig(stem + '.png', bbox_inches='tight', pad_inches=0.01)
    # CreationDate would make the PDF differ on every run; the file is
    # committed, so the rebuild has to be byte-identical or the tree is
    # dirty after every `bash repro.sh`.
    fig.savefig(stem + '.pdf', bbox_inches='tight', pad_inches=0.01,
                metadata={'CreationDate': None})  # vector, for the camera-ready
    print('wrote', stem + '.png and .pdf')

    json.dump({
        'description': 'delta against deleted task energy, one fitted curve per cell',
        'n_cells': len(fits),
        'n_points_plotted': len(pts),
        'median_per_cell_r2': med,
        'curves': [{'encoder': f['encoder'], 'dataset': f['dataset'],
                    'relation': f['relation'], 'rho0': float(f['rho0']),
                    's': float(f['s']), 'r2': float(f['r2'])} for f in fits],
        'points': pts,
        'grid': {'xmin': 1e-3, 'xmax': 1.0, 'n': 400},
        'source_tables': ['tables/fits_19cells.csv', 'tables/map10_19cells.csv'],
    }, open(a.out + '.json', 'w', newline='\n'), indent=1)
    print('wrote', a.out + '.json')


if __name__ == '__main__':
    main()
