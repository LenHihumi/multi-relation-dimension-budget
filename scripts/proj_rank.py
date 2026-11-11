"""Can a task-conditioned projection buy rank that PCA cannot?

The analysis paper's budget r* is measured on a PCA basis, so the only lever it has on storage is
truncation: keep the top r components, drop the rest.  A supervised projection changes the question --
learn W so that a *short prefix* of XW is already sufficient for this relation's ranking, then r* can
fall and every byte the deployment spends on vectors falls with it.  This is the feasibility check for
that idea, and it is designed so that it can fail cheaply.

Protocol (no leakage, one split per relation):
  rows are split 70/15/15 by *group* (entity for `inst`, category for `cat`), so test queries belong to
  groups the projection never saw; the PCA basis, the Fisher ratios and W are all fitted on train rows.
  mAP@10 uses the paper's relevance rule (binary, AP cut at 10, normalised by min(|R|,10), self excluded)
  and exact search, so every method is measured by the same instrument.

Methods:
  pca        PCA prefix (the paper's baseline)
  fisher     PCA components reordered by between/within ratio (the supervised baseline a reviewer will
             name; the paper already showed it is rho=1.000 with task energy)
  taskenergy PCA components reordered by lambda * eta^2 (the paper's own statistic, used as an order)
  randrot    a random orthogonal rotation of the PCA space (control: "any rotation" must not help)
  nested     ours: W initialised AT the PCA basis and optimised so that every prefix in P is a good
             retrieval space (multi-positive InfoNCE summed over prefixes; Matryoshka-style)
  nested-ce  the same nesting with a classification loss (answers "is this just LDA/Fisher again?")

Decision rule fixed before the run: if `nested` moves the item budget from 160 to <=128 (>=20% fewer
bytes at the same tolerance), the method is worth a paper; 150-155 is not, because a reviewer will ask
what it adds over reordering PCA, and the answer would be "nothing measurable".

Usage (inside mt-cpu):
  python3 proj_rank.py --enc clip_vitl14 --ds inshop --relation inst --steps 200
"""
import argparse
import json
import os
import time

import numpy as np

from sb_paths import SB_ROOT, SB_OUT, SB_CACHE, SB_MODELS, REPO  # paths resolve relative to the deposit
DUMP = os.environ.get('DUMP', SB_OUT + '/spectral_budget')
CACHE = os.environ.get('CACHE', SB_CACHE)


def load(enc, ds):
    if ds in ('imagenet', 'imagenet1m'):
        X = np.asarray(np.load(os.path.join(CACHE, 'imagenet1m', 'emb.npy'), mmap_mode='r'),
                       dtype=np.float32)
        lab = np.load(os.path.join(CACHE, 'imagenet1m', 'labels.npy'))
        meta = json.load(open(os.path.join(CACHE, 'imagenet1m', 'meta.json')))
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from imagenet_coarse import load_groups
        g = load_groups()
        return X, lab, np.array([g[int(i)] for i in lab])
    base = os.path.join(DUMP, 'emb', '%s__%s' % (enc, ds))
    meta = json.load(open(base + '.json'))
    if os.path.exists(base + '.npy'):
        emb = np.load(base + '.npy', mmap_mode='r')
    else:
        emb = np.memmap(base + '.bin', dtype=np.float32, mode='r').reshape(meta['n'], meta['dim'])
    rows = [json.loads(l) for l in open(os.path.join(CACHE, ds, 'gallery.jsonl'))]
    return (np.asarray(emb, dtype=np.float32), np.array([r['ent'] for r in rows]),
            np.array([r['cat'] for r in rows]))


def norm(Z):
    return Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9)


def map_at_10(Z, r, q, per_row, groups, k=10):
    # normalise *inside* the prefix: a row that is unit-norm at width K is not unit-norm at width r, and
    # the resulting per-row scale would silently replace cosine with a weighted dot product
    Zr = Z[:, :r]
    Zr = Zr / np.maximum(np.linalg.norm(Zr, axis=1, keepdims=True), 1e-9)
    aps = []
    for qi in q:
        s = Zr @ Zr[qi]
        s[qi] = -np.inf
        top = np.argpartition(-s, k)[:k]
        top = top[np.argsort(-s[top])]
        rel = groups[per_row[qi]]
        h, ap = 0, 0.0
        for rank, j in enumerate(top, 1):
            if j in rel:
                h += 1
                ap += h / rank
        aps.append(ap / min(len(rel), k))
    return float(np.mean(aps))


def curve(Z, prefixes, q, per_row, groups, ref=None):
    out = []
    for r in prefixes:
        m = map_at_10(Z, r, q, per_row, groups)
        out.append({'r': int(r), 'map10': m})
    ref = ref if ref is not None else max(x['map10'] for x in out)
    for x in out:
        x['delta'] = 1.0 - x['map10'] / ref
    return out, ref


def rstar(cv, eps):
    hit = [x['r'] for x in cv if x['delta'] <= eps]
    return int(hit[0]) if hit else int(cv[-1]['r'])


def fisher_order(Xtr, gtr, mu, V, w):
    """Between/within variance per PCA component, on centred projections."""
    Z = (Xtr - mu) @ V
    lab, inv = np.unique(gtr, return_inverse=True)
    Gm = np.zeros((len(lab), Z.shape[1]))
    np.add.at(Gm, inv, Z)
    cnt = np.bincount(inv, minlength=len(lab)).astype(np.float64)
    Gm /= np.maximum(cnt[:, None], 1)
    mean = Z.mean(0)
    Sb = np.sum(cnt[:, None] * (Gm - mean) ** 2, 0) / max(len(Z) - 1, 1)
    Sw = (np.sum(Z ** 2, 0) - np.sum(cnt[:, None] * Gm ** 2, 0)) / max(len(Z) - 1, 1)
    f = Sb / np.maximum(Sw, 1e-12)
    eta2 = Sb / np.maximum(Sb + Sw, 1e-12)
    return f, eta2


def train_nested(Xc, gtr, V0, prefixes, steps, lr, loss_type, temp, seed, threads, nkeys=8192,
                 head=0):
    """Optimise W so that every prefix of XcW is a good space for this relation's ranking.

    Each step projects a *sampled* key set (not the whole corpus): the projection is the dominant cost,
    and a 37k-row forward-backward through a 768x256 matrix made a 200-step run cost half an hour on a
    contended box.  The batch's own positives are forced into the key set, so the multi-positive term can
    never see an empty positive mask.
    """
    import torch
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    Xt = torch.from_numpy(np.ascontiguousarray(Xc, dtype=np.float32))
    if head and head < V0.shape[1]:
        # train the first `head` directions, leave the PCA tail untouched: a coarse relation lives in
        # the tail, and an unconstrained projection trained for a fine relation destroys it
        W = torch.nn.Parameter(torch.from_numpy(np.ascontiguousarray(V0[:, :head],
                                                                    dtype=np.float32)).clone())
        tail = torch.from_numpy(np.ascontiguousarray(V0[:, head:], dtype=np.float32))
    else:
        W = torch.nn.Parameter(torch.from_numpy(np.ascontiguousarray(V0, dtype=np.float32)).clone())
        tail = None
    opt = torch.optim.Adam([W], lr=lr)
    gl_np = gtr.astype(np.int64)
    gl = torch.from_numpy(gl_np)
    n = len(Xt)
    G = int(gl_np.max()) + 1
    B = 192
    # a query with no other member of its group has no positive at all, and the multi-positive logsumexp
    # over an empty mask is -inf, so sample only from groups with >= 2 members
    _, inv_tr, cnt_tr = np.unique(gl_np, return_inverse=True, return_counts=True)
    pool = np.where(cnt_tr[inv_tr] > 1)[0]
    by_group = {}
    for i in pool:
        by_group.setdefault(int(gl_np[i]), []).append(int(i))
    # a positive must be a *different* row of the group: sampling the query itself leaves the positive
    # mask empty once self is removed, and the multi-positive logsumexp over an empty mask is -inf
    buddy = {}
    for g, members in by_group.items():
        arr = np.array(members)
        for r in arr:
            others = arr[arr != r]
            buddy[int(r)] = int(others[0]) if len(others) else int(r)
    print('  train pool %d of %d rows, %d groups, %d keys/step' % (len(pool), n, len(by_group), nkeys))
    rng = np.random.default_rng(seed)
    for step in range(steps):
        idx = rng.choice(pool, size=min(B, len(pool)), replace=False)
        pos = np.array([buddy[int(q)] for q in idx])
        key = rng.choice(pool, size=min(nkeys, len(pool)), replace=False)
        rows = np.unique(np.concatenate([key, idx, pos]))
        Wf = W if tail is None else torch.cat([W, tail], dim=1)
        Z = Xt[torch.from_numpy(rows)] @ Wf
        qslot = np.searchsorted(rows, idx)
        gq = gl[torch.from_numpy(idx)]
        grow = gl[torch.from_numpy(rows)]
        loss = 0.0
        for p in prefixes:
            Zp = Z[:, :p]
            Zp = Zp / Zp.norm(dim=1, keepdim=True).clamp_min(1e-9)
            Q = Zp[torch.from_numpy(qslot)]
            logits = Q @ Zp.t() / temp
            pm = (gq[:, None] == grow[None, :])
            pm[torch.arange(len(idx)), torch.from_numpy(qslot)] = False
            if loss_type == 'rank':
                lse = torch.logsumexp(logits, dim=1)
                lse_pos = torch.logsumexp(logits.masked_fill(~pm, float('-inf')), dim=1)
                loss = loss - (lse_pos - lse).mean()
            else:                       # classification against per-prefix class means of the key set
                prot = torch.zeros(G, p)
                prot.index_add_(0, grow, Zp)
                cnt = torch.zeros(G).index_add_(0, grow, torch.ones(len(rows)))
                prot = prot / cnt[:, None].clamp_min(1e-9)
                prot = prot / prot.norm(dim=1, keepdim=True).clamp_min(1e-9)
                loss = loss + torch.nn.functional.cross_entropy((Q @ prot.t()) / temp, gq)
        loss = loss / len(prefixes)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([W], 1.0)
        opt.step()
        if step % 25 == 0 or step == steps - 1:
            print('  step %3d  loss %.4f' % (step, float(loss)), flush=True)
    return np.concatenate([W.detach().numpy(), tail.numpy()], axis=1) if tail is not None \
        else W.detach().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--enc', default='clip_vitl14')
    ap.add_argument('--ds', default='inshop')
    ap.add_argument('--relation', default='inst', choices=['inst', 'cat'])
    ap.add_argument('--K', type=int, default=320)
    ap.add_argument('--save-w', default='', help='npz path for W/V/mu, so an independent script can re-evaluate')
    ap.add_argument('--steps', type=int, default=200)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--temp', type=float, default=0.05)
    ap.add_argument('--eps', type=float, default=0.05)
    ap.add_argument('--queries', type=int, default=200)
    ap.add_argument('--threads', type=int, default=64)
    ap.add_argument('--nkeys', type=int, default=8192)
    ap.add_argument('--split', default='group', choices=['group', 'row'])
    ap.add_argument('--only-prefix', type=int, default=0,
                    help='train on a single prefix instead of the nested sum (ablation)')
    ap.add_argument('--head', type=int, default=0, help='0 = learn all columns; k = learn the first k only')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--prefixes', default='16,24,32,40,48,56,64,80,96,112,128,144,160,192,224,256,320')
    ap.add_argument('--methods', default='pca,fisher,taskenergy,randrot,nested,nested-ce')
    ap.add_argument('--out', default='')
    a = ap.parse_args()

    X, lab_i, lab_c = load(a.enc, a.ds)
    per_row = lab_i if a.relation == 'inst' else lab_c
    n, d = X.shape
    prefixes = [min(int(x), a.K) for x in a.prefixes.split(',')]
    prefixes = sorted(set(prefixes))

    # Two splits, because the two relations ask different questions of a learned basis.
    #   group  entire groups are held out: "does this work for entities that arrive later?"  This is the
    #          honest protocol for `inst`, where the indexed population grows.
    #   row    rows are held out, every group is represented in training: "does this work for new rows of
    #          the categories I already know?"  This is the honest protocol for `cat`, where the
    #          partition is fixed at build time and no deployment faces an unseen category.
    # labels are strings on the image corpora, so everything downstream works on dense integer codes
    uniq, inv = np.unique(per_row, return_inverse=True)
    per_row = inv.astype(np.int64)
    rng = np.random.default_rng(a.seed)
    if a.split == 'group':
        perm = rng.permutation(len(uniq))
        n_tr, n_va = int(0.7 * len(uniq)), int(0.15 * len(uniq))
        g_tr = set(perm[:n_tr].tolist())
        g_va = set(perm[n_tr:n_tr + n_va].tolist())
        is_tr = np.isin(per_row, list(g_tr))
        is_va = np.isin(per_row, list(g_va))
    else:
        perm = rng.permutation(len(per_row))
        n_tr, n_va = int(0.7 * len(per_row)), int(0.15 * len(per_row))
        is_tr = np.zeros(len(per_row), bool)
        is_va = np.zeros(len(per_row), bool)
        is_tr[perm[:n_tr]] = True
        is_va[perm[n_tr:n_tr + n_va]] = True
    is_te = ~(is_tr | is_va)
    print('%s/%s rel=%s  n=%d d=%d  groups=%d  train %d / val %d / test %d rows'
          % (a.enc, a.ds, a.relation, n, d, len(uniq), is_tr.sum(), is_va.sum(), is_te.sum()))

    # test queries: rows in test groups that have at least one other member
    groups = {}
    for i, v in enumerate(per_row):
        groups.setdefault(v, set()).add(i)
    elig = np.array([i for i in np.where(is_te)[0] if len(groups[per_row[i]]) > 1])
    q = np.sort(rng.choice(elig, size=min(a.queries, len(elig)), replace=False))
    print('test queries %d' % len(q))

    mu = X[is_tr].mean(0)
    Xtr = X[is_tr] - mu
    C = (Xtr.T @ Xtr) / max(len(Xtr) - 1, 1)
    w, V = np.linalg.eigh(C)
    o = np.argsort(w)[::-1]
    w, V = w[o][:a.K], V[:, o][:, :a.K]
    Xc = np.ascontiguousarray(X - mu)

    res = {'enc': a.enc, 'ds': a.ds, 'relation': a.relation, 'K': a.K, 'eps': a.eps,
           'n': n, 'queries': int(len(q)), 'prefixes': prefixes,
           'split': {'groups': int(len(uniq)), 'train_rows': int(is_tr.sum()),
                     'test_rows': int(is_te.sum())}, 'methods': {}}

    # common reference: the full-precision PCA operating point (K components), so no method can win by
    # shrinking its own denominator
    Zpca = norm(Xc @ V)
    ref_cv, ref = curve(Zpca, prefixes, q, per_row, groups)
    print('reference: PCA@%d mAP %.4f' % (a.K, ref))
    res['reference_map'] = ref
    res['methods']['pca'] = {'curve': ref_cv, 'rstar': rstar(ref_cv, a.eps)}

    f, eta2 = fisher_order(Xtr, per_row[is_tr], mu, V, w)
    orders = {'fisher': np.argsort(-f), 'taskenergy': np.argsort(-(w * eta2))}
    rr = np.random.default_rng(a.seed + 1)
    Qr, _ = np.linalg.qr(rr.normal(size=(a.K, a.K)))
    orders['randrot'] = np.arange(a.K)                 # applied as a rotation below, order unchanged

    for name, perm_idx in orders.items():
        Z = (Xc @ V)[:, perm_idx]
        if name == 'randrot':
            Z = Z @ Qr
        cv, _ = curve(norm(Z), prefixes, q, per_row, groups, ref=ref)
        res['methods'][name] = {'curve': cv, 'rstar': rstar(cv, a.eps)}
        print('%-12s r*=%3d  mAP@K %.4f' % (name, res['methods'][name]['rstar'], cv[-1]['map10']))

    # ---- ours: optimise W from the PCA initialisation
    # ablation: the same objective at ONE width.  The evaluation grid must stay the full one, so the
    # training set is a separate variable -- shrinking `prefixes` would also shrink the curve and make the
    # budget unmeasurable.
    train_prefixes = [a.only_prefix] if a.only_prefix else prefixes
    if a.only_prefix:
        print('  [ablation] training on a single prefix: %s' % train_prefixes)

    for mname, lt in (('nested', 'rank'), ('nested-ce', 'ce')):
        if mname not in a.methods.split(','):
            continue
        t0 = time.time()
        W = train_nested(Xc[is_tr], per_row[is_tr], V, train_prefixes, a.steps, a.lr, lt, a.temp,
                         a.seed, a.threads, a.nkeys, a.head)
        Z = norm(Xc @ W)
        cv, _ = curve(Z, prefixes, q, per_row, groups, ref=ref)
        res['methods'][mname] = {'curve': cv, 'rstar': rstar(cv, a.eps), 'train_s': time.time() - t0}
        if a.save_w and lt == 'rank':
            np.savez(a.save_w, W=W, V=V, mu=mu, relation=a.relation, enc=a.enc,
                     prefixes=np.array(prefixes), is_tr=is_tr, is_te=is_te)
        print('%-12s r*=%3d  mAP@K %.4f  (%.0fs)'
              % (mname, res['methods'][mname]['rstar'], cv[-1]['map10'], time.time() - t0))

    print('\n%-12s %6s %8s' % ('method', 'r*', 'mAP@K'))
    for k, v in res['methods'].items():
        print('%-12s %6d %8.4f' % (k, v['rstar'], v['curve'][-1]['map10']))
    base = res['methods']['pca']['rstar']
    for k, v in res['methods'].items():
        print('%s: r* %d = %.2fx of PCA' % (k, v['rstar'], v['rstar'] / max(base, 1)))

    if a.out:
        json.dump(res, open(a.out, 'w'), indent=1)
        print('wrote', a.out)


if __name__ == '__main__':
    main()
