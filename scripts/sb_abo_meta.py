"""Measure whether ABO can carry the inst relation -- metadata only, no images.

The corpus-extension branch turns on one question: is ABO's per-listing image count a
real multi-view of one physical product, or a bucket of colour variants?  The published
average (2.7 images per listing) cannot answer it, because a long tail of single-image
listings and a minority of variant buckets both produce the same mean.

So this script downloads the ten public listing shards (52 MB of gzipped JSON Lines, no
images, no credentials) and measures the four pre-registered criteria in
NEXT_METHODS.md section 13:

    A1  median images per listing >= 2 and single-image listings <= 50%
    A2  >= 80% of product types have >= 2 listings
    A3  the fixed split yields >= 1000 queries with every query entity in the gallery
    A4  listings whose images span multiple colours are <= 20% (they are variants,
        not views of one object, and would degrade inst into cat)

The decision is written to abo_meta.json together with the raw distributions, so the
verdict can be re-checked without re-downloading.

Usage
    python sb_abo_meta.py --out abo_meta.json            # downloads and measures
    python sb_abo_meta.py --cache /tmp/abo --out ...     # reuse an existing download
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import os
import random
import statistics as st
import sys
import urllib.request

BASE = 'https://amazon-berkeley-objects.s3.us-east-1.amazonaws.com/listings/metadata/'
SHARDS = ['listings_%d.json.gz' % i for i in range(10)]
SEED = 20261106
MAX_QUERY = 5000
MIN_QUERY = 1000


def fetch(cache):
    os.makedirs(cache, exist_ok=True)
    paths = []
    for name in SHARDS:
        p = os.path.join(cache, name)
        if not os.path.exists(p):
            url = BASE + name
            print('fetching %s' % url, flush=True)
            urllib.request.urlretrieve(url, p + '.part')
            os.replace(p + '.part', p)
        paths.append(p)
    return paths


def n_images(item):
    """ABO listings carry `main_image_id` (scalar) and `other_image_id` (list).
    There is no `images` field; reading one made every listing look single-image."""
    k = 1 if item.get('main_image_id') else 0
    others = item.get('other_image_id') or []
    return k + (len(others) if isinstance(others, list) else 0)


def colours_of(item):
    """Distinct colour VALUES within one language.

    ABO stores `color` as a list of {language_tag, value}; the same colour appears once
    per marketplace language, so a naive distinct-count over the list measures
    "how many languages describe this listing", not "how many colours it is sold in".
    Values are therefore grouped by language and the language with the most entries is
    used -- that is the best available proxy, since ABO has no per-image colour label.
    """
    by_lang = {}
    for c in item.get('color') or []:
        if isinstance(c, dict):
            v = c.get('value') or c.get('language_tagged_value')
            lang = str(c.get('language_tag') or '')
        else:
            v, lang = c, ''
        if v:
            by_lang.setdefault(lang, set()).add(str(v).strip().lower())
    if not by_lang:
        return set()
    return max(by_lang.values(), key=len)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default='/tmp/abo_cache')
    ap.add_argument('--out', default='abo_meta.json')
    args = ap.parse_args()

    paths = fetch(args.cache)

    n = 0
    img_counts = []
    per_type = collections.Counter()
    multi_colour = 0
    multi_colour_listable = 0          # only listings with >=2 images can be a variant bucket
    for p in paths:
        with gzip.open(p, 'rt', encoding='utf-8') as fh:
            for ln in fh:
                try:
                    it = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                n += 1
                k = n_images(it)
                img_counts.append(k)
                for t in it.get('product_type') or []:
                    v = t.get('value') if isinstance(t, dict) else t
                    if v:
                        per_type[str(v)] += 1
                        break
                if k >= 2 and len(colours_of(it)) > 1:
                    multi_colour += 1
                if k >= 2:
                    multi_colour_listable += 1

    # ---- standing self-checks: a parse failure must fail loudly, not become a verdict ----
    # Three incidents on this project produced plausible-but-wrong numbers from a wrong
    # field/level: ABO `it['images']` (0 images per listing), the strata bucket
    # `startswith('1')` (16+ folded into 1-3, totals 834 vs 668) and the content
    # fingerprint (385 of 668 unique).  Every one was caught by a number that could not
    # be true, never by reading the code -- so the impossibility checks live here.
    assert n > 0, 'no listings parsed'
    assert 50_000 <= n <= 200_000, 'listing count %d outside the plausible band' % n
    _med = st.median(img_counts)
    assert _med >= 1, ('median images per listing is %.1f -- every listing read as empty, '
                       'which is a parser failure, not a corpus property' % _med)
    _ge2 = sum(1 for k in img_counts if k >= 2) / n
    assert _ge2 > 0.3, 'only %.1f%% of listings have >=2 images -- check the image fields' % (100 * _ge2)
    print('self-check: %d listings, median %.0f images, %.1f%% with >=2  -> fields look right'
          % (n, _med, 100 * _ge2))

    med = st.median(img_counts)
    mean = sum(img_counts) / n
    single = sum(1 for k in img_counts if k == 1) / n
    ge2 = sum(1 for k in img_counts if k >= 2) / n

    # A2: product types with at least two listings
    types = len(per_type)
    types_ok = sum(1 for v in per_type.values() if v >= 2) / types if types else 0.0

    # A3: the fixed split -- one query per listing with >=2 images, capped, seeded
    listable = [i for i, k in enumerate(img_counts) if k >= 2]
    rnd = random.Random(SEED)
    rnd.shuffle(listable)
    n_query = min(len(listable), MAX_QUERY)
    # every query listing keeps one image in the gallery under the prepare rule, so the
    # gallery is non-empty by construction; what matters is that enough listable listings exist
    a3_ok = n_query >= MIN_QUERY

    a4_ok = (multi_colour / multi_colour_listable) <= 0.20 if multi_colour_listable else True

    crit = {'A1_median_ge2_and_single_le_half': bool(med >= 2 and single <= 0.50),
            'A2_types_with_ge2_listings_ge_80pct': bool(types_ok >= 0.80),
            'A3_split_yields_ge_1000_queries': bool(a3_ok),
            'A4_multi_colour_share_le_20pct': bool(a4_ok)}
    verdict = 'both' if all(crit.values()) else 'cat only'

    out = {'source': BASE, 'n_listings': n,
           'images_per_listing': {'median': med, 'mean': mean,
                                  'single_share': single, 'ge2_share': ge2,
                                  'p90': sorted(img_counts)[int(0.9 * (n - 1))],
                                  'max': max(img_counts)},
           'product_types': {'n_types': types, 'share_with_ge2_listings': types_ok,
                             'median_listings_per_type': st.median(list(per_type.values())) if types else 0},
           'split': {'listable_listings': len(listable), 'n_query': n_query,
                     'seed': SEED, 'max_query': MAX_QUERY},
           'colour_variants': {'listings_with_ge2_images': multi_colour_listable,
                               'multi_colour': multi_colour,
                               'share': (multi_colour / multi_colour_listable
                                         if multi_colour_listable else 0.0)},
           'criteria': crit, 'verdict': verdict}

    print('listings %d | images/listing: median %.0f mean %.2f  single %.1f%%  >=2 %.1f%%'
          % (n, med, mean, 100 * single, 100 * ge2))
    print('product types %d | %.1f%% have >=2 listings | median listings/type %.0f'
          % (types, 100 * types_ok, out['product_types']['median_listings_per_type']))
    print('listable (>=2 images) %d | fixed split yields %d queries' % (len(listable), n_query))
    print('multi-colour share among listable %.1f%% (%d of %d)'
          % (100 * out['colour_variants']['share'], multi_colour, multi_colour_listable))
    for k, v in crit.items():
        print('  %-42s %s' % (k, 'PASS' if v else 'FAIL'))
    print('\nverdict: %s' % verdict)

    with open(args.out, 'w') as fh:
        json.dump(out, fh, indent=1)
    print('wrote %s' % args.out)
    sys.exit(0 if verdict == 'both' else 2)


if __name__ == '__main__':
    main()
