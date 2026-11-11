"""Turn a downloaded corpus into the gallery/queries layout the rest of the artifact reads.

This script does NOT download.  It takes a source directory that already holds the images
and the official annotation files, and writes the two files every other script expects:

    <out>/gallery.jsonl   {"i": int, "path": str, "ent": str|None, "cat": str|None}
    <out>/queries.json    {"seed": int, "inst": [gallery indices], ...}

Two corpora are parsed, the two that survived `sb_admission.py` with a usable relation:

  deepfashion_c2s   Anno/list_item_bbox.txt (item id -> images) + list_category_img.txt
  logo2kplus        <class>/*.jpg with the class -> root category map from the official split

The split rule is fixed here so it cannot drift between runs:

  * one image per entity is held out as a query, the rest of that entity stays in the
    gallery -- this is what makes `inst` non-trivial, the query has a same-entity match;
  * at most `--max-query` entities contribute a query, sampled with a fixed seed;
  * a cell is skipped when it cannot reach `--min-query` queries, rather than silently
    lowering the standard.

Nothing here touches an encoder.  Embedding extraction stays a separate step, gated on
having a free GPU.

Usage
    python sb_prepare_corpus.py --corpus deepfashion_c2s --src /data/DeepFashion --out /x/sb/dfc2s
    python sb_prepare_corpus.py --corpus logo2kplus --src /data/Logo-2K+ --out /x/sb/logo2k
    python sb_prepare_corpus.py --selftest          # builds a synthetic corpus and checks itself
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import tempfile

SEED = 20261106
MIN_QUERY = 1000
MAX_QUERY = 5000


def write_corpus(items, out, min_query=MIN_QUERY, max_query=MAX_QUERY, seed=SEED):
    """items: list of (path, ent, cat).  Returns a summary dict."""
    by_ent = {}
    for path, ent, cat in items:
        if ent is None:
            continue
        by_ent.setdefault(ent, []).append((path, cat))
    ents = sorted(by_ent)
    rnd = random.Random(seed)
    rnd.shuffle(ents)
    ents = [e for e in ents if len(by_ent[e]) >= 2][:max_query]

    queries, gallery, qent = [], [], []
    for e in ents:
        rows = by_ent[e]
        rnd.shuffle(rows)
        qpath, cat = rows[0]
        queries.append((qpath, cat))
        qent.append(e)
        for path, c in rows[1:]:
            gallery.append((path, e, c))
        # a query whose entity has no gallery image would be unanswerable
        gallery.append((qpath, e, cat))

    if len(queries) < min_query:
        return {'ok': False, 'reason': 'only %d entities with >=2 images, need %d'
                % (len(queries), min_query)}

    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, 'gallery.jsonl'), 'w', encoding='utf-8') as fh:
        for i, (path, ent, cat) in enumerate(gallery):
            fh.write(json.dumps({'i': i, 'path': path, 'ent': ent, 'cat': cat}) + '\n')
    qidx = [i for i, (path, ent, cat) in enumerate(gallery) if ent in set(qent)]
    with open(os.path.join(out, 'queries.json'), 'w', encoding='utf-8') as fh:
        json.dump({'seed': seed, 'inst': [p for p in qidx][:len(queries)],
                   'n_query_entities': len(queries)}, fh, indent=1)

    cats = {}
    for _, _, c in gallery:
        if c is not None:
            cats[c] = cats.get(c, 0) + 1
    summary = {'ok': True, 'n_gallery': len(gallery), 'n_query': len(queries),
               'n_entities': len(ents), 'n_categories': len(cats),
               'median_images_per_entity': sorted(len(by_ent[e]) for e in ents)[len(ents) // 2],
               'seed': seed}
    with open(os.path.join(out, 'manifest.json'), 'w') as fh:
        json.dump(summary, fh, indent=1)
    return summary


def parse_deepfashion(src):
    """Anno/list_item_bbox.txt: '<image_name> <item_id> x1 y1 x2 y2'; category from
    Anno/list_category_img.txt: '<image_name> <category_id>'."""
    item_file = os.path.join(src, 'Anno', 'list_item_bbox.txt')
    cat_file = os.path.join(src, 'Anno', 'list_category_img.txt')
    if not (os.path.exists(item_file) and os.path.exists(cat_file)):
        raise SystemExit('expected %s and %s' % (item_file, cat_file))
    cats = {}
    with open(cat_file, encoding='utf-8') as fh:
        next(fh)
        next(fh)                                   # two header lines in the official file
        for ln in fh:
            p = ln.split()
            if len(p) >= 2:
                cats[p[0]] = p[1]
    items = []
    with open(item_file, encoding='utf-8') as fh:
        next(fh)
        next(fh)
        for ln in fh:
            p = ln.split()
            if len(p) >= 2:
                name = p[0]
                items.append((os.path.join(src, name), p[1], cats.get(name)))
    return items


def parse_logo2kplus(src):
    """<root>/<class>/*.jpg; the class name is the finest partition, the 10 root
    categories are given by the official 'root' mapping file when present."""
    rootmap = {}
    for cand in ('root_category.txt', 'root.txt', 'category_root.txt'):
        p = os.path.join(src, cand)
        if os.path.exists(p):
            with open(p, encoding='utf-8') as fh:
                for ln in fh:
                    p2 = ln.split()
                    if len(p2) >= 2:
                        rootmap[p2[0]] = p2[1]
            break
    items = []
    for cls in sorted(os.listdir(src)):
        d = os.path.join(src, cls)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                # one physical image per file: the class is the finest real partition,
                # so `ent` repeats the class and `inst` would be degenerate on this corpus
                items.append((os.path.join(d, f), cls, rootmap.get(cls, cls)))
    return items


def selftest():
    tmp = tempfile.mkdtemp(prefix='sbcorpus_')
    src = os.path.join(tmp, 'src')
    out = os.path.join(tmp, 'out')
    os.makedirs(src)
    items = []
    for e in range(1200):
        for k in range(3):
            p = os.path.join(src, 'e%05d_%d.jpg' % (e, k))
            open(p, 'w').close()
            items.append((p, 'e%05d' % e, 'c%d' % (e % 40)))
    s = write_corpus(items, out, min_query=1000)
    ok = (s['ok'] and s['n_query'] == 1200 and s['n_entities'] == 1200
          and s['n_categories'] == 40)
    # every query entity must appear in the gallery, else the cell is unanswerable
    gal = [json.loads(l) for l in open(os.path.join(out, 'gallery.jsonl'), encoding='utf-8')]
    ents = {g['ent'] for g in gal}
    q = json.load(open(os.path.join(out, 'queries.json')))
    covered = all(gal[i]['ent'] in ents for i in q['inst'])
    # below-threshold corpora must be refused, not silently shrunk
    small = write_corpus(items[:300], os.path.join(tmp, 'small'), min_query=1000)
    print('selftest: %s  gallery=%d query=%d entities=%d cats=%d covered=%s small_refused=%s'
          % ('PASS' if (ok and covered and not small['ok']) else 'FAIL',
             s['n_gallery'], s['n_query'], s['n_entities'], s['n_categories'], covered,
             not small['ok']))
    shutil.rmtree(tmp, ignore_errors=True)
    return 0 if (ok and covered and not small['ok']) else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus', choices=['deepfashion_c2s', 'logo2kplus'])
    ap.add_argument('--src')
    ap.add_argument('--out')
    ap.add_argument('--min-query', type=int, default=MIN_QUERY)
    ap.add_argument('--max-query', type=int, default=MAX_QUERY)
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()

    if args.selftest:
        sys.exit(selftest())
    if not (args.corpus and args.src and args.out):
        ap.error('--corpus, --src and --out are required unless --selftest')

    items = (parse_deepfashion(args.src) if args.corpus == 'deepfashion_c2s'
             else parse_logo2kplus(args.src))
    s = write_corpus(items, args.out, args.min_query, args.max_query)
    print(json.dumps(s, indent=1))
    if not s['ok']:
        sys.exit('refused: ' + s['reason'])


if __name__ == '__main__':
    main()
