"""Build a text corpus with two nested relevance partitions, for a cross-modality check.

Structure mirrors In-shop, which is why this corpus was chosen: a *document* plays the role of an
image, `parent_asin` (the product a review is about) plays the role of the item, and the review
category plays the role of the garment type.  Several reviews of one product are several photos of
one product; thousands of products sit inside one category.

Source: Amazon Reviews 2023 (McAuley-Lab), streamed from the mirror one category file at a time.
Reservoir sampling over the whole stream, so a file sorted by time does not bias the sample.

Usage:
    python3 sb_text_prep.py --per-cat 10000 --out /path/corpus.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys

import requests

# ModelScope's CDN is the reliable source from this cluster; hf-mirror answers 429 when several
# category files are pulled back to back.
MS = ('https://www.modelscope.cn/api/v1/datasets/McAuley-Lab/Amazon-Reviews-2023/repo'
      '?Revision=master&FilePath=raw/review_categories/%s.jsonl')
HF = ('https://hf-mirror.com/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/'
      'raw/review_categories/%s.jsonl')
BASE = MS

# small files only: the whole category is streamed and sampled, so nothing is truncated by position
CATS = ['Digital_Music', 'Gift_Cards', 'Magazine_Subscriptions', 'Subscription_Boxes',
        'Handmade_Products', 'All_Beauty']


def reservoir(url, want, seed):
    """Random `want` lines out of a whole JSONL stream (Algorithm R)."""
    rng = random.Random(seed)
    keep, n = [], 0
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        for line in r.iter_lines(chunk_size=1 << 20):
            if not line:
                continue
            n += 1
            if len(keep) < want:
                keep.append(line)
            else:
                j = rng.randrange(n)
                if j < want:
                    keep[j] = line
    return keep, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-cat', type=int, default=10000)
    ap.add_argument('--out', default='corpus.jsonl')
    ap.add_argument('--seed', type=int, default=20261109)
    ap.add_argument('--source', default='ms', choices=['ms', 'hf'])
    a = ap.parse_args()

    rows, stats = [], {}
    for i, cat in enumerate(CATS):
        url = (MS if a.source == 'ms' else HF) % cat
        try:
            lines, seen = reservoir(url, a.per_cat, a.seed + i)
        except Exception as e:                                   # a missing file must not kill the rest
            print('skip %-24s %s' % (cat, e), flush=True)
            continue
        kept = 0
        for ln in lines:
            try:
                d = json.loads(ln)
            except Exception:
                continue
            txt = ('%s. %s' % (d.get('title') or '', d.get('text') or '')).strip()
            par = d.get('parent_asin')
            if not par or len(txt) < 60:
                continue
            rows.append({'text': txt[:2000], 'parent': par, 'cat': cat,
                         'rating': d.get('rating')})
            kept += 1
        stats[cat] = {'streamed': seen, 'kept': kept}
        print('%-24s streamed %7d  kept %5d' % (cat, seen, kept), flush=True)

    # a fine group needs a partner: drop products with a single review in the sample
    from collections import Counter
    cnt = Counter(r['parent'] for r in rows)
    rows = [r for r in rows if cnt[r['parent']] > 1]
    for j, r in enumerate(rows):
        r['id'] = j
    print('docs %d  products %d  categories %d'
          % (len(rows), len({r['parent'] for r in rows}), len({r['cat'] for r in rows})))
    print('reviews/product: median %d  p90 %d  max %d'
          % (__import__('numpy').median([cnt[r['parent']] for r in rows]),
             __import__('numpy').percentile([cnt[r['parent']] for r in rows], 90),
             max(cnt[r['parent']] for r in rows)))
    with open(a.out, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    json.dump(stats, open(a.out + '.stats.json', 'w'), indent=1)
    print('wrote', a.out)


if __name__ == '__main__':
    sys.exit(main())
