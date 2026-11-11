"""Append more categories to an existing text corpus, with rate-limit backoff.

The mirror answers 429 when several large files are pulled back to back, so this runs one category at
a time, waits, and retries.  Idempotent: it re-filters single-review products and re-indexes ids.

Usage:
    python3 sb_text_add.py --corpus corpus.jsonl --cats Gift_Cards,All_Beauty --sleep 45
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter

import requests

BASE = ('https://hf-mirror.com/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/'
        'raw/review_categories/%s.jsonl')


def reservoir(url, want, seed, tries=5):
    rng = random.Random(seed)
    for t in range(tries):
        try:
            keep, n = [], 0
            with requests.get(url, stream=True, timeout=180) as r:
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
        except Exception as e:                                   # 429 is transient, everything else is not
            print('  attempt %d: %s' % (t + 1, e), flush=True)
            time.sleep(30 * (t + 1))
    raise RuntimeError('gave up on ' + url)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus', required=True)
    ap.add_argument('--cats', required=True)
    ap.add_argument('--per-cat', type=int, default=12000)
    ap.add_argument('--sleep', type=float, default=45)
    ap.add_argument('--seed', type=int, default=20261109)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.corpus, encoding='utf-8')]
    have = {r['cat'] for r in rows}
    print('existing docs %d  categories %s' % (len(rows), sorted(have)))

    for k, cat in enumerate([c.strip() for c in a.cats.split(',') if c.strip()]):
        if cat in have:
            print('have %s, skip' % cat)
            continue
        time.sleep(a.sleep)
        lines, seen = reservoir(BASE % cat, a.per_cat, a.seed + k)
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
            rows.append({'text': txt[:2000], 'parent': par, 'cat': cat, 'rating': d.get('rating')})
            kept += 1
        print('%-26s streamed %7d  kept %5d' % (cat, seen, kept), flush=True)

    cnt = Counter(r['parent'] for r in rows)
    rows = [r for r in rows if cnt[r['parent']] > 1]
    for j, r in enumerate(rows):
        r['id'] = j
    with open(a.corpus, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    sizes = sorted(len(v) for v in Counter(r['parent'] for r in rows).values())
    print('total docs %d  products %d  categories %d  size median %d p90 %d'
          % (len(rows), len(set(r['parent'] for r in rows)), len(set(r['cat'] for r in rows)),
             sizes[len(sizes) // 2], sizes[int(0.9 * len(sizes))]))


if __name__ == '__main__':
    sys.exit(main())
