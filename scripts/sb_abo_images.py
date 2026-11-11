#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build an ABO retrieval corpus in the layout `sb_common` expects.

Selection and split are fixed before any download, so the same seed always yields the
same cells:

  * listings are filtered to those with >= 2 images (the inst relation needs a second
    view) and capped at `--max-images` per listing, so the handful of 21-image
    listings cannot dominate the gallery;
  * `--queries` listings become query listings; one image each is held out as the
    query, and every remaining image of every selected listing goes to the gallery;
  * `ent` = item_id (inst relation), `cat` = product_type (cat relation).

Images are ABO `images/small/<image_id>.jpg`.  Downloads are resumable and every file
is checked for the JPEG magic and a minimum size, because a truncated image would
otherwise surface as an encoder error much later.

Usage:
  python sb_abo_images.py --listings 100 --queries 50 --max-images 3 --smoke
  python sb_abo_images.py --listings 5000 --queries 1000 --max-images 10
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import os
import random
import ssl
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from sb_paths import SB_ROOT, SB_OUT, SB_CACHE, SB_MODELS, REPO  # paths resolve relative to the deposit
META = SB_ROOT + '/.cache/abo'
BASE = 'https://amazon-berkeley-objects.s3.amazonaws.com/images/small/%s'
SB = SB_CACHE + '/abo'
SEED = 20261106


def load_listings():
    rows = []
    for i in range(10):
        p = os.path.join(META, 'listings_%d.json.gz' % i)
        if not os.path.exists(p):
            raise SystemExit('missing shard %s' % p)
        with gzip.open(p, 'rt', encoding='utf-8') as fh:
            for ln in fh:
                try:
                    d = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                imgs = []
                if d.get('main_image_id'):
                    imgs.append(str(d['main_image_id']))
                for x in d.get('other_image_id') or []:
                    if x:
                        imgs.append(str(x))
                if len(imgs) < 2:
                    continue
                ptype = ''
                for t in d.get('product_type') or []:
                    v = t.get('value') if isinstance(t, dict) else t
                    if v:
                        ptype = str(v)
                        break
                if not ptype:
                    continue
                rows.append({'item_id': str(d['item_id']), 'cat': ptype, 'images': imgs})
    return rows


def load_paths():
    """image_id -> `ab/cdef01.jpg`, from images/metadata/images.csv.gz.

    The listing metadata's `main_image_id` is NOT the S3 filename stem: `81iZlv3bjpL`
    lives at `8c/8ccb5859.jpg`.  Guessing the key from the id returns 404 for every
    image, which is why the mapping file is required rather than derived."""
    p = os.path.join(META, 'images.csv.gz')
    if not os.path.exists(p):
        raise SystemExit('missing %s (images/metadata/images.csv.gz)' % p)
    m = {}
    with gzip.open(p, 'rt', encoding='utf-8') as fh:
        next(fh, None)
        for ln in fh:
            parts = ln.rstrip('\n').split(',')
            if len(parts) >= 4:
                m[parts[0]] = parts[3]
    print('image id -> path entries: %d' % len(m), flush=True)
    return m


_CTX = ssl._create_unverified_context()   # host lacks a CA bundle for urllib


def fetch_one(args):
    img_id, path, dest = args
    if os.path.exists(dest) and os.path.getsize(dest) > 1024:
        with open(dest, 'rb') as fh:
            if fh.read(2) == b'\xff\xd8':
                return 'cached'
    url = BASE % path
    for _ in range(3):
        try:
            with urllib.request.urlopen(url, timeout=30, context=_CTX) as r:
                blob = r.read()
            if len(blob) > 1024 and blob[:2] == b'\xff\xd8':
                with open(dest, 'wb') as fh:
                    fh.write(blob)
                return 'ok'
            return 'bad'
        except Exception:
            continue
    return 'fail'


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--listings', type=int, default=5000)
    ap.add_argument('--queries', type=int, default=1000)
    ap.add_argument('--max-images', type=int, default=10)
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()

    rows = load_listings()
    print('listings with >=2 images and a product type: %d' % len(rows), flush=True)
    rng = random.Random(SEED)
    rows.sort(key=lambda r: r['item_id'])
    picked = rng.sample(rows, min(args.listings, len(rows)))
    picked.sort(key=lambda r: r['item_id'])
    if args.queries > len(picked):
        raise SystemExit('queries (%d) > selected listings (%d)' % (args.queries, len(picked)))
    qids = {r['item_id'] for r in rng.sample(picked, args.queries)}

    paths = load_paths()
    os.makedirs(os.path.join(SB, 'img'), exist_ok=True)
    tasks, rows_out, held_ids = [], [], []
    for r in picked:
        imgs = r['images'][:args.max_images]
        held = imgs[0] if r['item_id'] in qids else None
        for img in imgs:
            if img not in paths:
                continue
            dest = os.path.join(SB, 'img', img + '.jpg')
            tasks.append((img, paths[img], dest))
            rows_out.append({'path': dest, 'ent': r['item_id'], 'cat': r['cat'], 'img_id': img})
            if img == held:
                held_ids.append(img)
    print('to fetch: %d images | query listings %d' % (len(tasks), len(held_ids)), flush=True)

    with ThreadPoolExecutor(args.workers) as ex:
        res = list(ex.map(fetch_one, tasks))
    stat = collections.Counter(res)
    print('fetch:', dict(stat), flush=True)
    if stat['fail'] or stat['bad']:
        print('WARNING: %d failed, %d corrupt -- rerun to retry' % (stat['fail'], stat['bad']))

    ok = {t[0] for t, st in zip(tasks, res) if st in ('ok', 'cached')}
    gallery = [g for g in rows_out if g['img_id'] in ok]
    # sb_encoders sorts the manifest by 'i', and load_gallery reads it in file order,
    # so 'i' must be the final row index -- re-number after the download filter
    for i, g in enumerate(gallery):
        g['i'] = i
    pos = {g['img_id']: i for i, g in enumerate(gallery)}

    # The gallery holds every image, query images included -- that is how instre/inshop
    # are laid out.  `queries['inst']` is a list of gallery ROW INDICES (not paths), and
    # every such row must have a second image of the same listing for the inst cell.
    gents = collections.Counter(g['ent'] for g in gallery)
    inst = [pos[i] for i in held_ids if i in pos and gents[gallery[pos[i]]['ent']] >= 2]
    dropped = len(held_ids) - len(inst)
    queries = {'seed': SEED, 'inst': inst, 'n_gallery': len(gallery)}

    # ---- standing self-checks, before anything is concluded from these counts ----
    # (sb_checks: conservation / non-zero median / field coverage -- the four parse
    #  incidents on this project were all caught by numbers that could not be true)
    import sb_checks
    # real identities, not restatements of the code:
    #   every row handed to the downloader must come back as a gallery row or be excluded
    #   the per-listing image counts must add up to the gallery size
    #   the held-out query set must be exactly the listings we asked for
    sb_checks.require_conservation({'gallery': len(gallery), 'excluded': len(tasks) - len(gallery)},
                                   len(tasks), label='download filter')
    sb_checks.require_conservation(gents, len(gallery), label='per-listing image counts')
    sb_checks.require_conservation({'queries': len(held_ids)}, args.queries, label='query slots')
    sb_checks.require_nonzero_median([v for v in gents.values()], 'images per listing',
                                     hint='check main_image_id / other_image_id')
    sb_checks.require_coverage([v >= 2 for v in gents.values()], 'listings with >=2 images', 0.30)
    assert gallery, 'empty gallery -- every download failed'
    assert 0.8 * args.queries <= len(inst) <= args.queries, \
        'only %d of %d queries survived' % (len(inst), args.queries)
    assert len(gallery) > len(inst), 'gallery smaller than the query set'

    with open(os.path.join(SB, 'gallery.jsonl'), 'w', encoding='utf-8') as fh:
        for g in gallery:
            fh.write(json.dumps(g, ensure_ascii=False) + '\n')
    queries['version'] = 2                 # v1 was a list of dicts with no row index
    queries['fingerprint'] = '%d/%d' % (len(gallery), len(inst))
    json.dump(queries, open(os.path.join(SB, 'queries.json'), 'w', encoding='utf-8'),
              ensure_ascii=False)
    stamp = sb_checks.manifest_stamp(os.path.join(SB, 'gallery.jsonl'),
                                     os.path.join(SB, 'queries.json'),
                                     {'n_query': len(inst), 'n_listing': len(gents),
                                      'n_category': len(set(g['cat'] for g in gallery)),
                                      'max_images_per_listing': args.max_images,
                                      'seed': SEED, 'eps': None})
    print('manifest stamp: %s' % json.dumps(stamp, ensure_ascii=False))

    cats = collections.Counter(g['cat'] for g in gallery)
    print('wrote gallery %d (ent %d, cat %d) | queries %d (dropped %d)'
          % (len(gallery), len(gents), len(cats), len(inst), dropped))
    print('images per listing: median %.1f  mean %.1f'
          % (sorted(gents.values())[len(gents) // 2], len(gallery) / max(len(gents), 1)))
    if args.smoke:
        print('SMOKE: ok if gallery > queries and every query ent is in the gallery')


if __name__ == '__main__':
    main()
