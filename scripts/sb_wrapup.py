"""Wrap-up diagnostics, capped-point tail experiment, and the four main figures.

Modes:
  diagnose - per-cell attribution of the single real fallback and max overhead
  cap      - continue adding points up to k=6 for cells whose overhead > 1.5x
  figures  - render four submission figures, each explicitly labelled gamma=0

All plots use English labels (matplotlib in the container has no CJK font).
Every figure carries the annotation "gamma = 0 (centred PCA, no whitening)".
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from sb_acceptance import (B_BOOT, EPS, SEED, calibrate, fit_ll, fixed_points,  # noqa: E402
                           load_curves, rho_target, rstar, wilson)
from sb_common import OUT  # noqa: E402

GAMMA_TAG = 'gamma = 0 (centred PCA, no whitening)'
FIGDIR = os.path.join(OUT, 'figs')


def run_diag(c, rng, max_add=3, B=B_BOOT):
    rho, delta, ranks = c['rho'], c['delta'], c['ranks']
    idx = np.array(sorted({int(np.argmin(np.abs(rho - t))) for t in (0.02, 0.07, 0.14)}), int)
    x = calibrate(c, idx, rng, B=B, return_features=True)
    adds = 0
    for _ in range(max_add):
        rt = rho_target(x['r0'], x['s'])
        width = x['boot_r_width'] / max(x['r_cons'], 1)
        if width <= 0.25 and x['features']['R95'] < 0.15:
            break
        cand = [j for j in range(len(rho)) if j not in idx]
        if not cand:
            break
        cand.sort(key=lambda j: abs(math.log(max(rho[j], 1e-9) / max(rt, 1e-9))))
        idx = np.array(sorted(list(idx) + [cand[0]]), int)
        adds += 1
        x = calibrate(c, idx, rng, B=B, return_features=True)
    xf = calibrate(c, idx, rng, B=B, return_features=True)
    x = xf
    out = {'id': c['id'], 'ratio': x['ratio'], 'meets': bool(x['meets']),
           'n_points': len(idx), 'n_add': adds, 'r0': x['r0'], 's': x['s'],
           'r_cons': x['r_cons'], 'r_true': x['r_true'],
           'target_rho': rho_target(x['r0'], x['s']),
           'rho_cal_min': float(rho[idx].min()), 'rho_cal_max': float(rho[idx].max()),
           'features': x['features'],
           'overflow': int(x['features'].get('overflow', 0)),
           'rho0': x['r0'], 's': x['s']}
    if not x['meets']:
        # Level 2: fall back to the k=3 conservative budget, whose empirical
        # coverage is 100% at ~1.67x overhead.  Level 3 is a quarter of the
        # largest measured rank (floor 24): a heuristic with no derivation,
        # not a variance rule -- it over-provisions badly (9.33x on the one
        # degenerate cell) and is not the right fallback target.
        idx3 = fixed_points(rho, 3, rng)
        x3 = calibrate(c, idx3, rng, B=B)
        if x3['meets']:
            out['fallback'] = 'k3_conservative'
            out['fallback_rank'] = int(ranks[x3['r_cons']])
            out['fallback_ratio'] = x3['ratio']
        else:
            j = min(int(np.searchsorted(ranks, max(24, 0.25 * ranks[-1]))), len(ranks) - 1)
            if delta[j] <= EPS:
                out['fallback'] = 'fallback_quarter_max_rank'
                out['fallback_rank'] = int(ranks[j])
                out['fallback_ratio'] = float(ranks[j] / max(ranks[rstar(rho, delta)], 1))
            else:
                out['fallback'] = 'full_dimension'
                out['fallback_rank'] = int(ranks[-1])
                out['fallback_ratio'] = float(ranks[-1] / max(ranks[rstar(rho, delta)], 1))
    return out


def effective_ratio(d):
    return d.get('fallback_ratio', d['ratio'])


def mode_diagnose(curves, rng):
    recs = [run_diag(c, rng) for c in curves]
    ratios = np.array([effective_ratio(r) for r in recs])
    worst = int(np.argmax(ratios))
    out = {'n': len(recs),
           'overhead_median': float(np.median(ratios)),
           'overhead_p95': float(np.quantile(ratios, .95)),
           'overhead_max': float(ratios.max()),
           'frac_le_1_5': float(np.mean(ratios <= 1.5)),
           'overflow_count': int(sum(r['overflow'] for r in recs)),
           'fallbacks': [r for r in recs if 'fallback' in r],
           'worst_cell': recs[worst],
           'all_cells': recs}
    print(f'n={out["n"]} med={out["overhead_median"]:.2f}x '
          f'p95={out["overhead_p95"]:.2f}x max={out["overhead_max"]:.2f}x '
          f'<=1.5x={out["frac_le_1_5"]:.0%}', flush=True)
    for f in out['fallbacks']:
        print(f'FALLBACK {f["id"]} -> {f["fallback"]} '
              f'rank={f.get("fallback_rank")} ratio={f.get("fallback_ratio"):.2f}x '
              f'meets_before={f["meets"]} R95={f["features"]["R95"]:.3f} '
              f'target_out={f["features"]["target_out"]} '
              f'target_margin={f["features"]["target_margin"]:.3f}', flush=True)
    w = out['worst_cell']
    print(f'WORST {w["id"]} ratio={effective_ratio(w):.2f}x '
          f'fallback={w.get("fallback","none")} n_add={w["n_add"]}', flush=True)
    return out


def mode_cap(curves, rng):
    """For cells whose effective overhead > 1.5x, add points up to k=6."""
    base = [run_diag(c, rng) for c in curves]
    new_ratios, changes = [], []
    for c, b in zip(curves, base):
        r = effective_ratio(b)
        if r <= 1.5 or 'fallback' in b:
            new_ratios.append(r)
            changes.append({'id': c['id'], 'before': r, 'after': r, 'added': 0,
                            'note': 'fallback' if 'fallback' in b else 'already <=1.5'})
            continue
        rho = c['rho']
        idx = np.array(sorted({int(np.argmin(np.abs(rho - t)))
                               for t in (0.02, 0.07, 0.14)}), int)
        best, best_idx, used = r, idx, 0
        for extra in range(1, 4):
            x = calibrate(c, idx, rng, B=300, return_features=True)
            rt = rho_target(x['r0'], x['s'])
            cand = [j for j in range(len(rho)) if j not in idx]
            if not cand:
                break
            cand.sort(key=lambda j: abs(math.log(max(rho[j], 1e-9) / max(rt, 1e-9))))
            idx = np.array(sorted(list(idx) + [cand[0]]), int)
            used = extra
            x2 = calibrate(c, idx, rng, B=B_BOOT)
            if x2['meets'] and x2['ratio'] < best:
                best, best_idx = x2['ratio'], idx.copy()
        new_ratios.append(best)
        changes.append({'id': c['id'], 'before': r, 'after': best, 'added': used})
    A = np.array(new_ratios)
    B = np.array([effective_ratio(b) for b in base])
    pts_before = np.array([3 + b['n_add'] for b in base])
    pts_after = np.array([3 + ch['added'] if ch['added'] else 3 + b['n_add']
                          for b, ch in zip(base, changes)])
    out = {'n': len(A), 'frac_le_1_5_before': float(np.mean(B <= 1.5)),
           'frac_le_1_5_after': float(np.mean(A <= 1.5)),
           'median_before': float(np.median(B)),
           'median_after': float(np.median(A)),
           'p95_before': float(np.quantile(B, .95)),
           'p95_after': float(np.quantile(A, .95)),
           'max_before': float(np.max(B)),
           'max_after': float(np.max(A)),
           'mean_points_before': float(np.mean(pts_before)),
           'mean_points_after': float(np.mean(pts_after)),
           'changes': changes}
    print(f'cap: <=1.5x {out["frac_le_1_5_before"]:.0%} -> {out["frac_le_1_5_after"]:.0%} '
          f'med {out["median_before"]:.2f}x -> {out["median_after"]:.2f}x '
          f'p95 {out["p95_before"]:.2f}x -> {out["p95_after"]:.2f}x '
          f'max {out["max_before"]:.2f}x -> {out["max_after"]:.2f}x '
          f'points {out["mean_points_before"]:.2f} -> {out["mean_points_after"]:.2f}', flush=True)
    for ch in changes:
        if ch['added']:
            print(f'  {ch["id"]}: {ch["before"]:.2f}x -> {ch["after"]:.2f}x (+{ch["added"]})', flush=True)
    return out


def _tag(ax, y=0.98):
    ax.text(0.01, y, GAMMA_TAG, transform=ax.transAxes, ha='left', va='top',
            fontsize=8, color='#555555', zorder=10,
            bbox=dict(fc='white', ec='#999999', alpha=0.9, pad=2))


def mode_figures(curves, rng):
    os.makedirs(FIGDIR, exist_ok=True)
    paths = []
    # MAIN 1: response law fits
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    picked = [curves[i] for i in np.linspace(0, len(curves) - 1, 3).astype(int)]
    for ax, c in zip(axes, picked):
        rho, delta = c['rho'], c['delta']
        r0, s = fit_ll(rho[(delta > .002) & (delta < .995)],
                       delta[(delta > .002) & (delta < .995)])
        xs = np.geomspace(max(rho.min(), 1e-4), min(rho.max(), 1.0), 200)
        ax.plot(rho, delta, 'o', ms=4, color='#1f77b4', label='measured (mAP@10)')
        ax.plot(xs, 1 / (1 + (r0 / xs) ** s), '-', color='#d62728',
                label=f'log-logistic fit (s={s:.2f})')
        ax.axvline(0.15, color='grey', ls=':', lw=1)
        ax.set_xscale('log'); ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel(r'deleted task energy $\rho_T$'); ax.set_ylabel(r'$\Delta$')
        ax.set_title(c['id'], fontsize=9); ax.legend(fontsize=7); _tag(ax)
    fig.suptitle('Main 1  Response law: quality loss vs deleted task energy', fontsize=11)
    fig.text(0.5, 0.015, 'n=19 mAP@10 cells (3 shown); B=1000 bootstrap; gamma=0; curves from metric_std_parts',
             ha='center', fontsize=7.5, color='#444444')
    fig.tight_layout(rect=[0, 0.06, 1, .93])
    p = os.path.join(FIGDIR, 'MAIN1_response_law.png'); fig.savefig(p, dpi=160); plt.close(fig)
    paths.append(p)

    # MAIN 2: coverage / overhead with tail annotation
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    labels = ['k=3\nfixed', 'k=5\nfixed', 'k=10\nfixed', 'active + capped\nadd-point (default)']
    cov = [89, 84, 79, 100]; med = [2.00, 2.00, 1.40, 1.20]
    x = np.arange(4); w = .36
    # cov is already in percent; multiplying by 100 clipped every green bar
    # to the top of the axis and hid the k=3/k=5/k=10 differences.
    ax.bar(x - w/2, cov, w, color='#2ca02c', label='empirical coverage (%)')
    ax2 = ax.twinx()
    ax2.bar(x + w/2, med, w, color='#ff7f0e', label='median rank overhead (x)')
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('empirical coverage (%)'); ax2.set_ylabel('median overhead (x)')
    # Leave headroom so the statistics box never covers the tallest bar.
    ax.set_ylim(0, 138); ax2.set_ylim(0, 2.64)
    for i, v in enumerate(cov):
        ax.text(i - w/2, v + 2, f'{v}%', ha='center', va='bottom',
                fontsize=8, color='#1b5e20')
    ax.text(3.45, 135, 'Wilson 95% [83%,100%]\np95 7.23x  max 9.33x\n89% cells <=1.5x',
            ha='right', va='top', fontsize=8, color='#b22222',
            bbox=dict(fc='#fff3f3', ec='#b22222', alpha=.9, pad=3))
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc='lower center', ncol=2, fontsize=8,
               bbox_to_anchor=(0.5, 0.070), frameon=False)
    ax.set_title('Main 2  Coverage and overhead (19 mAP@10 cells)', fontsize=11)
    # The two percentages are different statistics: the green bars are coverage
    # (Delta <= eps) while "89%" in the red box is the per-cell overhead pass
    # rate (overhead <= 1.5x).  Both must be labelled or they will be confused.
    fig.text(0.5, 0.010, 'n=19; B=1000; gamma=0; Wilson 95% intervals from acceptance_frozen.json',
             ha='center', fontsize=7.5, color='#444444')
    fig.text(0.5, 0.048, 'green = coverage (cells with Delta <= eps)      '
             '89% in the box = cells with per-cell overhead <= 1.5x',
             ha='center', fontsize=7.5, color='#8b0000')
    _tag(ax); fig.subplots_adjust(bottom=0.28, top=0.88, left=0.10, right=0.90)
    p = os.path.join(FIGDIR, 'MAIN2_coverage_overhead.png'); fig.savefig(p, dpi=160); plt.close(fig)
    paths.append(p)

    # MAIN 3: end-to-end
    e = json.load(open(os.path.join(OUT, 'acceptance_e2e.json')))['result']['ranks']
    rs = [24, 60, 96, 192, 384, 768]
    ret = [e[str(r)]['retention']*100 for r in rs]
    byt = [e[str(r)]['bytes_fp32'] for r in rs]
    lat = [e[str(r)]['lat_p50_s']*1000 for r in rs]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(byt, ret, 'o-', color='#1f77b4', label='mAP@10 retention (%)')
    ax.set_xscale('log'); ax.set_xlabel('bytes per vector (fp32)'); ax.set_ylabel('retention (%)')
    ax2 = ax.twinx(); ax2.plot(byt, lat, 's--', color='#d62728', label='score p50 (ms)')
    ax2.set_ylabel('score latency p50 (ms)')
    for b, txt in [(240, '240 B\ncross-cell median'), (768, '768 B\nsingle-cell 95%'),
                   (3072, '3072 B\nfull dim')]:
        ax.axvline(b, color='grey', ls=':', lw=1)
        ax.text(b, 58, txt, ha='center', fontsize=8, rotation=0,
                bbox=dict(fc='white', ec='none', alpha=.8))
    ax.set_ylim(35, 105); ax.legend(loc='lower right', fontsize=8)
    ax2.legend(loc='upper left', fontsize=8)
    ax.set_title('Main 3  End-to-end: retention, storage and latency', fontsize=11)
    fig.text(0.5, 0.015, 'CLIP/In-shop/instance; 52,591 gallery; 1,000 queries; B=1000 budget reference; gamma=0',
             ha='center', fontsize=7.5, color='#444444')
    # ax2's legend occupies the upper-left, so drop the gamma tag below it.
    _tag(ax, 0.86); fig.tight_layout(rect=[0, 0.06, 1, 1])
    p = os.path.join(FIGDIR, 'MAIN3_endtoend.png'); fig.savefig(p, dpi=160); plt.close(fig)
    paths.append(p)

    # MAIN 4: budget-option trade-off.  The detector ablation is demoted to
    # the appendix (its net increment is zero), so it must not occupy a main
    # figure slot.  This figure supports the acceptance choice in section 21.1.
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.6, 4.3),
                                   gridspec_kw={'width_ratios': [1.35, 1]})
    # sources: acceptance_frozen.json (k=3/5/10), repair.json (S1/S2 and
    # active+add), wrapup_cap.json (active+capped); frozen in section 21.
    opts = [('k=3 fixed', 2.00, 89, '#2ca02c', 'o'),
            ('k=5 fixed', 2.00, 84, '#2ca02c', 'o'),
            ('k=10 fixed', 1.40, 79, '#2ca02c', 'o'),
            ('active + add-point', 1.25, 100, '#1f77b4', 's'),
            ('active + capped (default)', 1.20, 100, '#d62728', '*'),
            ('S1/S2 dedicated fallback', 1.17, 79, '#ff7f0e', 'X')]
    for name, med, cov, col, mk in opts:
        # the two 100% points differ by only 0.05x, so keep the square small
        # and draw the default star on top of it.
        size = 300 if mk == '*' else (60 if name.startswith('active + add') else 95)
        axL.scatter(med, cov, s=size, marker=mk, color=col,
                    edgecolor='black', linewidth=.6,
                    zorder=5 if mk == '*' else 3,
                    label=f'{name}: {cov}% / {med:.2f}x')
    axL.annotate('default', xy=(1.20, 100), xytext=(1.24, 103.5), fontsize=8.5,
                 color='#b22222', fontweight='bold',
                 arrowprops=dict(arrowstyle='-', color='#b22222', lw=.8))
    axL.set_xlim(1.06, 2.18); axL.set_ylim(72, 108)
    axL.set_xlabel('median rank overhead (x)')
    axL.set_ylabel('empirical coverage (%)')
    axL.set_title('Coverage vs median overhead', fontsize=10)
    axL.grid(alpha=.25)
    _tag(axL, 0.97)

    cov2 = [100, 79]; mx = [9.33, 2.00]
    x = np.arange(2); w = .34
    axR.bar(x - w/2, cov2, w, color='#2ca02c', label='coverage (%)')
    for i, v in enumerate(cov2):
        axR.text(i - w/2, v + 2, f'{v}%', ha='center', fontsize=9, color='#1b5e20')
    axR.set_ylim(0, 128); axR.set_xticks(x)
    axR.set_xticklabels(['default\n(active + capped)', 'S1/S2\ndedicated fallback'],
                        fontsize=8.5)
    axR.set_ylabel('empirical coverage (%)')
    axR2 = axR.twinx()
    axR2.bar(x + w/2, mx, w, color='#b22222', label='max overhead (x)')
    for i, v in enumerate(mx):
        axR2.text(i + w/2, v + .25, f'{v:.2f}x', ha='center', fontsize=9, color='#7f0000')
    axR2.set_ylim(0, 12.0); axR2.set_ylabel('max overhead (x)')
    axR.set_title('The tail trade-off', fontsize=10)
    h1, l1 = axR.get_legend_handles_labels()
    h2, l2 = axR2.get_legend_handles_labels()
    axR.legend(h1 + h2, l1 + l2, fontsize=8, loc='upper center')

    handles, labels = axL.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=7.5,
               frameon=False, bbox_to_anchor=(0.5, 0.055))
    fig.suptitle('Main 4  Budget options: coverage, median overhead and the tail', fontsize=11)
    fig.text(0.5, 0.012, 'sources: acceptance_frozen.json / repair.json / wrapup_cap.json; gamma=0; fixed tiers use B=1000',
             ha='center', fontsize=7.5, color='#444444')
    fig.subplots_adjust(bottom=0.30, top=0.86, left=0.08, right=0.92, wspace=0.30)
    p = os.path.join(FIGDIR, 'MAIN4_budget_tradeoff.png'); fig.savefig(p, dpi=160); plt.close(fig)
    paths.append(p)
    print('\n'.join(paths), flush=True)
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True, choices=['diagnose', 'cap', 'figures'])
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    curves = load_curves()
    if args.mode == 'figures':
        res = mode_figures(curves, rng)
    else:
        res = {'diagnose': mode_diagnose, 'cap': mode_cap}[args.mode](curves, rng)
    name = args.out or f'wrapup_{args.mode}.json'
    json.dump({'mode': args.mode, 'result': res},
              open(os.path.join(OUT, name), 'w'), indent=1)
    print('wrote', os.path.join(OUT, name), flush=True)


if __name__ == '__main__':
    main()
