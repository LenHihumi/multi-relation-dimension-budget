"""Head-to-head of budget-selection decision rules on identical cells.

Cross-paper numbers are not comparable here (different corpora, different
backbones, different metrics).  What *is* comparable is the decision rule:
given a corpus spectrum -- and, for our law, the label structure -- which rank
does the method choose, does that choice reach a quality target, and how many
bytes does it spend.

Rules compared
--------------
law            r = min{r : Delta(rho_T(r)) <= 0.05},
               Delta = 1/(1+(rho0/rho)^s), (rho0, s) from the per-(domain,
               relation) lookup -- geometric medians over the cells of that
               group, never refitted per cell.
energy90/95/99 smallest r with cumulative lambda share >= 90/95/99 %.
scree          argmax curvature of log lambda against log rank.
pr             (sum lambda)^2 / sum lambda^2, and 2x that.
ff25 / ff50    fixed fraction 0.25 d / 0.5 d.
oracle         smallest r whose measured recall is >= 95 % of the cell's
               full-dim recall.  Upper bound on efficiency, not deployable.

Discretisation
--------------
Rules return an integer rank.  Budgets can only be bought at the measured grid
points, so every rule is snapped *up* to the smallest grid rank >= its native
answer (capped at d).  Snapping only adds bytes; it never lets a rule pass the
quality target on a rank it did not ask for.

Retention
---------
retention(r) = rec(r) / rec(d), i.e. the fraction of full-dimensional recall
kept.  The target is retention >= 0.95.  "shortfall" is the number of points
(100 x retention deficit) by which a cell misses.

Writes outputs/diag/spectral_budget/rules_compare.json
       outputs/diag/spectral_budget/figs/F23_rules_compare.png
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sb_paths import SB_OUT

OUT = os.path.join(SB_OUT, 'spectral_budget')
FIG = os.path.join(OUT, 'figs')

EPS = 0.05           # the law's tolerance on Delta
TARGET = 0.95        # quality target on retention
FP32 = 4             # bytes per retained coordinate

# ----------------------------------------------------------------------------
# rule registry: name -> (short label, label requirement, family)
# ----------------------------------------------------------------------------
RULES = [
    ('law',     'Ours (law)',             'labels (group split)', 'spectral+labels'),
    ('law_e01', 'Ours (law, 1% tol.)',    'labels (group split)', 'spectral+labels'),
    ('energy90', 'Energy 90%',            'none', 'label-free'),
    ('energy95', 'Energy 95%',            'none', 'label-free'),
    ('energy99', 'Energy 99%',            'none', 'label-free'),
    ('scree',   'Scree knee',             'none', 'label-free'),
    ('pr',      'Participation ratio',    'none', 'label-free'),
    ('pr2',     'Participation ratio x2', 'none', 'label-free'),
    ('ff25',    'Fixed 0.25 d',           'none', 'label-free'),
    ('ff50',    'Fixed 0.5 d',            'none', 'label-free'),
    ('oracle',  'Oracle @95%',            'labels (recall)', 'oracle'),
]
RULE_ORDER = [r[0] for r in RULES]
RULE_LABEL = {r[0]: r[1] for r in RULES}
RULE_NEEDS = {r[0]: r[2] for r in RULES}
RULE_FAMILY = {r[0]: r[3] for r in RULES}


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
def load_theory():
    th = {}
    for p in glob.glob(os.path.join(OUT, 'theory__*.json')):
        d = json.load(open(p))
        th[(d['encoder'], d['dataset'])] = d
    return th


def sigmoid_params():
    """Geometric medians of (rho0, s) per (domain, relation) from sigmoid.json."""
    s = json.load(open(os.path.join(OUT, 'sigmoid.json')))
    acc = {}
    for c in s['cells']:
        acc.setdefault((c['ds'], c['rel']), []).append((c['rho0'], c['s'], c['k']))
    look = {}
    for key, v in acc.items():
        r0 = float(np.exp(np.median(np.log([x[0] for x in v]))))
        ss = float(np.exp(np.median(np.log([x[1] for x in v]))))
        look[key] = dict(rho0=r0, s=ss, n_cells=len(v),
                         members=sorted({f'{k}' for _, _, k in v}))
    return look


def build_cells(th):
    cells = []
    for p in sorted(glob.glob(os.path.join(OUT, 'grid', '*.json'))):
        g = json.load(open(p))
        t = th.get((g['encoder'], g['dataset']))
        if not t:
            continue
        lam = np.array(t['lambda'], float)
        d = len(lam)
        ranks = np.array(sorted({int(r) for r in g['grid'] if int(r) <= d}), int)
        for cname, cell in g['cells'].items():
            rel = cname.split('@')[0]
            if rel not in ('inst', 'cat') or rel == 'cat1':
                continue
            if rel not in t['tasks']:
                continue
            for k in ('k1', 'k10'):
                e = cell.get('cen', {}).get(k)
                if not e:
                    continue
                rc = np.array((e.get('rec_curve') or [])[:len(ranks)], float)
                if len(rc) != len(ranks) or rc[-1] <= 0:
                    continue
                cells.append(dict(
                    enc=g['encoder'], ds=g['dataset'], rel=rel, k=k, d=d,
                    ranks=ranks, rc=rc, full=float(rc[-1]),
                    lam=lam,
                    eta2=np.array(t['tasks'][rel]['eta2'], float),
                    rstar_reported=e.get('rstar05'),
                ))
    return cells


# ----------------------------------------------------------------------------
# the label-free spectrum quantities
# ----------------------------------------------------------------------------
def cum_energy(lam):
    return np.cumsum(lam) / lam.sum()


def energy_rank(lam, share):
    cum = cum_energy(lam)
    idx = int(np.searchsorted(cum, share, side='left'))
    return int(min(max(idx + 1, 1), len(lam)))


def pr_rank(lam):
    return float(lam.sum() ** 2 / (lam ** 2).sum())


def scree_rank(lam, smooth=5):
    """Rank of maximum curvature of log lambda against log rank.

    Plain second differences on a raw eigenspectrum are dominated by sampling
    noise, so the log-spectrum is box-smoothed on a 5-point window first; the
    curvature is then  |y''| / (1+y'^2)^{3/2}  with y = log lambda, x = log r.
    """
    d = len(lam)
    x = np.log(np.arange(1, d + 1, dtype=float))
    y = np.log(np.maximum(lam, 1e-30))
    if smooth and smooth > 1 and d > smooth:
        ker = np.ones(smooth) / smooth
        y = np.convolve(np.pad(y, (smooth // 2, smooth // 2), mode='edge'),
                        ker, mode='valid')[:d]
    dy = np.gradient(y, x)
    d2y = np.gradient(dy, x)
    kap = np.abs(d2y) / np.power(1.0 + dy ** 2, 1.5)
    return int(np.argmax(kap) + 1)


def rho_curve(lam, eta2):
    om = lam * eta2
    cum = np.clip(np.cumsum(om) / om.sum(), 0.0, 1.0)
    rho = 1.0 - cum                       # rho[r-1] is the deleted task energy
    return rho


def law_rank(rho, rho0, s, eps=EPS, margin=1.0):
    """Smallest r with Delta(rho_T(r)) <= eps.

        Delta = 1/(1+(rho0/rho)^s) <= eps
        <=>  (rho0/rho)^s >= (1-eps)/eps
        <=>  rho <= rho0 * (eps/(1-eps))^(1/s)  =: thr

    rho_T falls monotonically in r, so this is the first index at which the
    deleted task energy drops below thr; if it never does, the whole space is
    needed (r = d) and the cell is reported as such.  `margin` divides the
    threshold, which lowers the deleted energy the rule is willing to accept
    and therefore buys extra retained dimensions -- used only for the safety
    curve that asks what it costs to turn a median-calibrated rule into a
    rule that reliably meets the target.
    """
    thr = rho0 * (eps / (1.0 - eps)) ** (1.0 / s) / margin
    ok = np.where(rho <= thr)[0]
    return int(ok[0] + 1) if len(ok) else len(rho)


# ----------------------------------------------------------------------------
def snap_up(r, ranks, d):
    r = int(min(max(int(round(r)), 1), d))
    i = int(np.searchsorted(ranks, r, side='left'))
    return int(ranks[min(i, len(ranks) - 1)])


def eval_at(ranks, rc, r):
    i = int(np.searchsorted(ranks, r, side='left'))
    i = min(i, len(ranks) - 1)
    return float(rc[i]), int(ranks[i])


# ----------------------------------------------------------------------------
def run():
    th = load_theory()
    look = sigmoid_params()
    cells = build_cells(th)
    print(f'{len(cells)} cells')

    recs = []
    margin_ctx = []
    for c in cells:
        d, lam, ranks = c['d'], c['lam'], c['ranks']
        rho = rho_curve(lam, c['eta2'])
        full = c['full']

        # oracle: smallest grid rank whose measured retention reaches TARGET
        keep = c['rc'] / full
        ok = np.where(keep >= TARGET)[0]
        r_or = int(ranks[ok[0]]) if len(ok) else d

        lp = look.get((c['ds'], c['rel']))
        native = {
            'law': law_rank(rho, lp['rho0'], lp['s']) if lp else d,
            'law_e01': law_rank(rho, lp['rho0'], lp['s'], eps=0.01) if lp else d,
            'energy90': energy_rank(lam, 0.90),
            'energy95': energy_rank(lam, 0.95),
            'energy99': energy_rank(lam, 0.99),
            'scree': scree_rank(lam),
            'pr': int(round(pr_rank(lam))),
            'pr2': int(round(2 * pr_rank(lam))),
            'ff25': int(round(0.25 * d)),
            'ff50': int(round(0.5 * d)),
            'oracle': r_or,
        }
        margin_ctx.append((c, rho, lp, r_or))
        for name in RULE_ORDER:
            r_nat = int(native[name])
            r = snap_up(r_nat, ranks, d)
            rec, r_used = eval_at(ranks, c['rc'], r)
            ret = rec / full
            recs.append(dict(
                rule=name, enc=c['enc'], ds=c['ds'], rel=c['rel'], k=c['k'],
                d=d, r_native=r_nat, r=r_used, bytes=int(r * FP32),
                bytes_frac=float(r / d),
                rec=rec, retention=float(ret), delta=float(1.0 - ret),
                meets=bool(ret >= TARGET),
                shortfall_pts=float(max(0.0, (TARGET - ret) * 100.0)),
                r_oracle=int(r_or),
                oracle_meets=bool(len(ok) > 0),
                ratio_to_oracle=float(r / max(r_or, 1)),
                rho0=lp['rho0'] if lp else None, s=lp['s'] if lp else None,
            ))

    # ---- aggregation -------------------------------------------------------
    def agg(sub):
        if not sub:
            return None
        b = np.array([r['bytes'] for r in sub], float)
        ratio = np.array([r['ratio_to_oracle'] for r in sub], float)
        meets = np.array([r['meets'] for r in sub], bool)
        miss = [r['shortfall_pts'] for r in sub if not r['meets']]
        hit = [r['bytes'] for r in sub if r['meets']]
        return dict(
            n=len(sub),
            hit_rate=float(meets.mean()),
            n_hit=int(meets.sum()), n_miss=int((~meets).sum()),
            median_bytes=float(np.median(b)),
            median_bytes_frac=float(np.median([r['bytes_frac'] for r in sub])),
            median_bytes_at_target=float(np.median(hit)) if hit else None,
            median_shortfall_pts=float(np.median(miss)) if miss else 0.0,
            median_ratio_to_oracle=float(np.median(ratio)),
            p90_ratio_to_oracle=float(np.percentile(ratio, 90)),
            overshoot_frac=float(np.mean(ratio > 1.0 + 1e-9)),
        )

    def table(subset):
        out = {}
        for name in RULE_ORDER:
            out[name] = agg([r for r in subset if r['rule'] == name])
        return out

    summary = dict(
        all=table(recs),
        inst=table([r for r in recs if r['rel'] == 'inst']),
        cat=table([r for r in recs if r['rel'] == 'cat']),
        k1=table([r for r in recs if r['k'] == 'k1']),
        k10=table([r for r in recs if r['k'] == 'k10']),
        by_relation_k={f'{a}__{b}': table([r for r in recs if r['rel'] == a and r['k'] == b])
                       for a in ('inst', 'cat') for b in ('k1', 'k10')},
        by_dataset={ds: table([r for r in recs if r['ds'] == ds])
                    for ds in sorted({r['ds'] for r in recs})},
    )

    # oracle reference row (the cheapest possible budget that still meets 95%)
    oracle_bytes = np.array([summary['all']['oracle']['median_bytes']], float)

    # ---- how much margin does the law need before it reliably hits 95%? ----
    # A median-calibrated rule meets a hard target about half the time by
    # construction.  This curve buys coverage by inflating the law's own
    # threshold -- no new parameter, no refit -- and prices each level of
    # coverage, so "our advantage survives on bytes" can be answered at equal
    # coverage instead of at equal nominal tolerance.
    margin_curve = {}
    for grp in ('all', 'inst', 'cat'):
        ctx = [x for x in margin_ctx if grp == 'all' or x[0]['rel'] == grp]
        curve = []
        for k in (1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 6.0):
            hits, bs = [], []
            for c, rho, lp, r_or in ctx:
                r_nat = law_rank(rho, lp['rho0'], lp['s'], margin=k) if lp else c['d']
                r = snap_up(r_nat, c['ranks'], c['d'])
                rec, r_used = eval_at(c['ranks'], c['rc'], r)
                hits.append(rec / c['full'] >= TARGET)
                bs.append(r_used * FP32)
            curve.append(dict(margin=k, hit_rate=float(np.mean(hits)),
                              median_bytes=float(np.median(bs))))
        margin_curve[grp] = curve
        print(f'  law margin sweep [{grp}] ' + '  '.join(
            f'x{c["margin"]:g}:{c["hit_rate"]:.0%}/{c["median_bytes"]:.0f}B'
            for c in curve))

    meta = dict(
        eps=EPS, target=TARGET, fp32_bytes=FP32, n_cells=len(cells),
        n_unique_cells=len({(r['enc'], r['ds'], r['rel'], r['k']) for r in recs}),
        rules=[dict(name=n, label=RULE_LABEL[n], needs=RULE_NEEDS[n],
                    family=RULE_FAMILY[n]) for n in RULE_ORDER],
        law_lookup={f'{k[0]}__{k[1]}': v for k, v in look.items()},
        snap='rules snapped up to the smallest measured grid rank >= native r',
    )

    # ---- report -----------------------------------------------------------
    def show(name, title):
        print(f'\n{title}')
        print(f'  {"rule":22s} {"n":>3s} {"hit":>6s} {"med B":>8s} '
              f'{"B@hit":>8s} {"short":>6s} {"r/r_or":>7s} {"needs"}')
        for rn in RULE_ORDER:
            v = summary[name][rn]
            if not v:
                continue
            b_hit = f'{v["median_bytes_at_target"]:8.0f}' if v['median_bytes_at_target'] else '       -'
            print(f'  {RULE_LABEL[rn]:22s} {v["n"]:3d} {v["hit_rate"]:6.0%} '
                  f'{v["median_bytes"]:8.0f} {b_hit} '
                  f'{v["median_shortfall_pts"]:6.2f} {v["median_ratio_to_oracle"]:7.2f} '
                  f'{RULE_NEEDS[rn]}')

    show('all', 'ALL CELLS (n=38)')
    show('inst', 'INSTANCE CELLS')
    show('cat', 'CATEGORY CELLS')

    res = dict(meta=meta, summary=summary, law_margin_curve=margin_curve, cells=recs)
    json.dump(res, open(os.path.join(OUT, 'rules_compare.json'), 'w'), indent=1)
    print('\nwrote', os.path.join(OUT, 'rules_compare.json'))

    make_fig(summary, recs, oracle_bytes[0])
    return res


# ----------------------------------------------------------------------------
def make_fig(summary, recs, oracle_bytes):
    os.makedirs(FIG, exist_ok=True)
    order = [n for n in RULE_ORDER if n != 'oracle']
    x = np.arange(len(order))
    w = 0.38

    plt.rcParams.update({'font.size': 8.5, 'axes.linewidth': 0.8,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(1, 3, figsize=(13.6, 4.6))

    # ---- A: hit rate, split inst / cat ------------------------------------
    a = ax[0]
    hi = [summary['inst'][n]['hit_rate'] for n in order]
    hc = [summary['cat'][n]['hit_rate'] for n in order]
    a.bar(x - w / 2, hi, w, color='#2b6cb0', label='instance (n=22)')
    a.bar(x + w / 2, hc, w, color='#dd6b20', label='category (n=16)')
    a.axhline(TARGET, color='k', ls=':', lw=1.0, label='95% target')
    a.set_xticks(x)
    a.set_xticklabels([RULE_LABEL[n] for n in order], rotation=42, ha='right',
                      fontsize=7.2)
    a.set_ylabel('fraction of cells at 95% retention', labelpad=2)
    a.set_ylim(0, 1.22)
    a.legend(frameon=False, fontsize=6.8, loc='upper left', ncol=3,
             handlelength=1.2, columnspacing=0.9)
    a.set_title('A  accuracy of the budget', loc='left', fontweight='bold')

    # ---- B: median bytes ---------------------------------------------------
    b = ax[1]
    bi = [summary['inst'][n]['median_bytes'] for n in order]
    bc = [summary['cat'][n]['median_bytes'] for n in order]
    b.bar(x - w / 2, bi, w, color='#2b6cb0', label='instance')
    b.bar(x + w / 2, bc, w, color='#dd6b20', label='category')
    b.axhline(oracle_bytes, color='k', ls='--', lw=1.1,
              label=f'oracle @95% ({oracle_bytes:.0f} B)')
    b.set_yscale('log')
    b.set_xticks(x)
    b.set_xticklabels([RULE_LABEL[n] for n in order], rotation=42, ha='right',
                      fontsize=7.2)
    b.set_ylabel('median bytes per vector (fp32, log)')
    b.set_ylim(1e2, 6e3)
    b.legend(frameon=False, fontsize=7.2, loc='upper left')
    b.set_title('B  cost of the budget', loc='left', fontweight='bold')

    # ---- C: chosen r vs oracle r ------------------------------------------
    c = ax[2]
    marks = {'law': ('o', '#c53030'), 'energy95': ('s', '#2f855a'),
             'pr': ('^', '#6b46c1')}
    for name in ('law', 'energy95', 'pr'):
        sub = [r for r in recs if r['rule'] == name]
        c.scatter([r['r_oracle'] for r in sub], [r['r'] for r in sub], s=26,
                  marker=marks[name][0], alpha=0.75, facecolors='none',
                  edgecolors=marks[name][1], linewidths=1.1,
                  label=RULE_LABEL[name])
    lim = np.array([6, 600])
    c.plot(lim, lim, 'k-', lw=1.0, label='r = r$_{oracle}$')
    c.plot(lim, 2 * lim, 'k:', lw=0.9, label='2x oracle')
    c.plot(lim, 0.5 * lim, 'k:', lw=0.9)
    c.set_xscale('log')
    c.set_yscale('log')
    c.set_xlim(*lim)
    c.set_ylim(*lim)
    c.set_xlabel('oracle rank (smallest r at 95% retention)')
    c.set_ylabel('rank chosen by the rule')
    c.legend(frameon=False, fontsize=6.8, loc='lower right')
    c.set_title('C  per-cell budget choice', loc='left', fontweight='bold')

    fig.subplots_adjust(left=0.072, right=0.99, top=0.90, bottom=0.27, wspace=0.30)
    p = os.path.join(FIG, 'F23_rules_compare.png')
    fig.savefig(p, dpi=200)
    fig.savefig(p.replace('.png', '.pdf'))
    print('wrote', p)


if __name__ == '__main__':
    run()
