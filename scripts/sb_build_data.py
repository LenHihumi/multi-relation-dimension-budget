"""Materialise the galleries used by the spectral-budget study.

Four galleries, each written as JPEG on local scratch plus a JSONL manifest:

  inshop   52,591 fashion photos, 11,318 items, hierarchical labels
           (category1 x category2 x category3).  -> instance AND category relations
  sop      22,634 Stanford-Online-Products photos, 11,318 items.  -> instance
  roxford  5,063 Oxford/Paris building photos with graded qrels.  -> copy (real)
  ilias    100,000 YFCC product photos (fidelity / gallery-size sweep).

Plus a *synthetic copy* relation on inshop and sop: deterministic near-duplicate
augmentation of a query image (resize-crop, rotate, photometric jitter, re-JPEG).
Copy detection is the only relation whose queries are not gallery members, so the
copy query set is generated here and stored on disk.

Outputs (all under $SB_GALLERY/<dataset>/):
  img/            gallery JPEGs (only for datasets shipped as parquet blobs)
  gallery.jsonl   {"i":int, "path":str, "ent":str|null, "cat":str|null}
  queries.json    {"inst":[gidx...], "copy":[{"src":gidx,"path":str}], "seed":int}

Run:
    python3 sb_build_data.py --datasets inshop,sop,roxford,ilias
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
import random
import statistics
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from sb_paths import SB_CACHE, SB_ROOT

ROOT = os.environ.get('SB_GALLERY', os.path.join(SB_CACHE, 'galleries'))
DS = os.environ.get('SB_DATASETS', os.path.join(SB_ROOT, 'datasets'))
SEED = 20261102
N_QUERY = 1000
N_COPY = 1000


def log(m):
    print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)


# --------------------------------------------------------------------------
# near-duplicate augmentation (copy relation)
# --------------------------------------------------------------------------
def aug_one(args):
    """Deterministic near-duplicate of one image: geometric + photometric + JPEG."""
    src, dst, seed = args
    from PIL import Image, ImageEnhance
    rng = random.Random(seed)
    with Image.open(src) as im:
        im = im.convert('RGB')
        w, h = im.size
        s = rng.uniform(0.60, 0.85)
        ratio = rng.uniform(0.85, 1.18)
        cw = max(16, int(w * s))
        ch = max(16, int(cw / ratio))
        ch = min(ch, h)
        x0 = rng.randint(0, max(0, w - cw))
        y0 = rng.randint(0, max(0, h - ch))
        im = im.crop((x0, y0, x0 + cw, y0 + ch))
        im = im.rotate(rng.uniform(-10, 10), resample=Image.BICUBIC,
                       expand=False, fillcolor=(127, 127, 127))
        im = ImageEnhance.Brightness(im).enhance(rng.uniform(0.82, 1.18))
        im = ImageEnhance.Contrast(im).enhance(rng.uniform(0.85, 1.15))
        im = ImageEnhance.Color(im).enhance(rng.uniform(0.9, 1.1))
        im.save(dst, 'JPEG', quality=rng.choice([42, 55, 70]))
    return dst


def aug_many(pairs, workers=32):
    with ProcessPoolExecutor(workers) as ex:
        list(ex.map(aug_one, pairs, chunksize=16))


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def write_jsonl(path, rows):
    with open(path, 'w', encoding='utf-8') as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + '\n')


def save_blob(args):
    """Write one parquet image blob to a JPEG file; returns row index."""
    idx, blob, out = args
    if os.path.exists(out):
        return idx
    from PIL import Image
    buf = blob
    if isinstance(blob, dict):          # {'bytes': ..., 'path': ...}
        buf = blob.get('bytes')
    if isinstance(buf, str):
        import base64
        buf = base64.b64decode(buf)
    with Image.open(io.BytesIO(buf)) as im:
        im.convert('RGB').save(out, 'JPEG', quality=95)
    return idx


# --------------------------------------------------------------------------
# builders
# --------------------------------------------------------------------------
def build_inshop():
    import pyarrow.parquet as pq
    d = os.path.join(ROOT, 'inshop')
    img_dir = os.path.join(d, 'img')
    os.makedirs(img_dir, exist_ok=True)
    t = pq.read_table(f'{DS}/inshop/data/data-00000-of-00001.parquet',
                      columns=['item_ID', 'category1', 'category2', 'category3',
                               'color', 'image'])
    rows = t.to_pylist()
    log(f'inshop rows {len(rows)}')
    jobs, meta = [], []
    for i, r in enumerate(rows):
        out = os.path.join(img_dir, f'{i:06d}.jpg')
        jobs.append((i, r['image'], out))
        item = r['item_ID']
        ent = item.rsplit('_', 2)[0]                     # MEN_Denim_id_00000080_01
        # granularity ladder, coarsest first:
        #   cat1  2 classes (men / women)
        #   cat   23 classes (garment type within gender)
        #   catf  garment type x colourway            <- finest non-instance level
        cat = f"{r['category1']}|{r['category2']}"
        catf = f"{cat}|{r['color']}"
        meta.append({'i': i, 'path': out, 'ent': ent, 'cat': cat,
                     'cat1': r['category1'], 'catf': catf})
    t0 = time.time()
    with ProcessPoolExecutor(32) as ex:
        list(ex.map(save_blob, jobs, chunksize=64))
    log(f'inshop images written in {time.time() - t0:.1f}s')

    ents = Counter(m['ent'] for m in meta)
    cats = Counter(m['cat'] for m in meta)
    catfs = Counter(m['catf'] for m in meta)
    log(f'inshop items={len(ents)} imgs/item median={statistics.median(ents.values())} '
        f'max={max(ents.values())}  singleton_items={sum(v == 1 for v in ents.values())}')
    log(f'inshop cat={len(cats)} median={statistics.median(cats.values())} '
        f'catf={len(catfs)} median={statistics.median(catfs.values())} '
        f'min={min(catfs.values())}')
    write_jsonl(os.path.join(d, 'gallery.jsonl'), meta)

    rng = np.random.default_rng(SEED)
    cand = np.array([i for i, m in enumerate(meta) if ents[m['ent']] >= 2])
    q = rng.choice(cand, N_QUERY, replace=False)
    copy_src = rng.choice(len(meta), N_COPY, replace=False)
    copy_dir = os.path.join(d, 'copy')
    os.makedirs(copy_dir, exist_ok=True)
    pairs = [(meta[int(s)]['path'], os.path.join(copy_dir, f'copy_{k:04d}.jpg'), SEED + k)
             for k, s in enumerate(copy_src)]
    aug_many(pairs)
    qjson = {'seed': SEED,
             'inst': sorted(int(x) for x in q),
             'copy': [{'src': int(s), 'path': os.path.join(copy_dir, f'copy_{k:04d}.jpg')}
                      for k, s in enumerate(copy_src)],
             'n_gallery': len(meta)}
    json.dump(qjson, open(os.path.join(d, 'queries.json'), 'w'), indent=1)
    log(f'inshop queries: inst={len(qjson["inst"])} copy={len(qjson["copy"])}')


def build_sop():
    """SOP ships 120,053 JPEGs (~10.6 per item).  Keep the first PER_ITEM images of
    each item so the gallery mixes a few coarse classes with many fine items but
    stays in the same cost bracket as In-shop."""
    d = os.path.join(ROOT, 'sop')
    os.makedirs(d, exist_ok=True)
    per_item = 4
    roots = sorted(glob.glob(f'{DS}/sop_hf/sop_images/*'))
    meta = []
    for item_dir in roots:
        ent = os.path.basename(item_dir)
        fs = sorted(glob.glob(os.path.join(item_dir, '**', '*.jpg'), recursive=True))[:per_item]
        for f in fs:
            meta.append({'i': len(meta), 'path': f, 'ent': ent, 'cat': None})
    ents = Counter(m['ent'] for m in meta)
    log(f'sop images={len(meta)} items={len(ents)} '
        f'median={statistics.median(ents.values())} singleton={sum(v == 1 for v in ents.values())}')
    write_jsonl(os.path.join(d, 'gallery.jsonl'), meta)

    rng = np.random.default_rng(SEED + 1)
    cand = np.array([i for i, m in enumerate(meta) if ents[m['ent']] >= 2])
    q = rng.choice(cand, min(N_QUERY, len(cand)), replace=False)
    copy_src = rng.choice(len(meta), N_COPY, replace=False)
    copy_dir = os.path.join(d, 'copy')
    os.makedirs(copy_dir, exist_ok=True)
    pairs = [(meta[int(s)]['path'], os.path.join(copy_dir, f'copy_{k:04d}.jpg'), SEED + 7000 + k)
             for k, s in enumerate(copy_src)]
    aug_many(pairs)
    qjson = {'seed': SEED + 1, 'inst': sorted(int(x) for x in q),
             'copy': [{'src': int(s), 'path': os.path.join(copy_dir, f'copy_{k:04d}.jpg')}
                      for k, s in enumerate(copy_src)],
             'n_gallery': len(meta)}
    json.dump(qjson, open(os.path.join(d, 'queries.json'), 'w'), indent=1)
    log(f'sop queries: inst={len(qjson["inst"])} copy={len(qjson["copy"])}')


def build_roxford():
    import pyarrow.parquet as pq
    d = os.path.join(ROOT, 'roxford')
    img_dir = os.path.join(d, 'img')
    os.makedirs(img_dir, exist_ok=True)
    t = pq.read_table(f'{DS}/roxford_hf/corpus-00000-of-00001.parquet')
    rows = t.to_pylist()
    id2i = {}
    jobs, meta = [], []
    for i, r in enumerate(rows):
        out = os.path.join(img_dir, f'{i:06d}.jpg')
        jobs.append((i, r['image'], out))
        id2i[r['id']] = i
        meta.append({'i': i, 'path': out, 'ent': r['id'], 'cat': None})
    with ProcessPoolExecutor(32) as ex:
        list(ex.map(save_blob, jobs, chunksize=64))
    write_jsonl(os.path.join(d, 'gallery.jsonl'), meta)
    rel = {}
    qs = pq.read_table(f'{DS}/roxford_hf/qrels-00000-of-00001.parquet').to_pylist()
    for r in qs:
        if int(r['score']) < 1:
            continue
        if r['query-id'] not in id2i or r['corpus-id'] not in id2i:
            continue
        rel.setdefault(r['query-id'], []).append(id2i[r['corpus-id']])
    qjson = {'seed': SEED, 'rel': {k: sorted(v) for k, v in rel.items()},
             'n_gallery': len(meta)}
    json.dump(qjson, open(os.path.join(d, 'queries.json'), 'w'), indent=1)
    sizes = [len(v) for v in rel.values()]
    log(f'roxford gallery={len(meta)} queries={len(rel)} rel/query median={statistics.median(sizes)} '
        f'min={min(sizes)} max={max(sizes)}')


def build_ilias():
    d = os.path.join(ROOT, 'ilias')
    os.makedirs(d, exist_ok=True)
    paths = sorted(glob.glob(f'{DS}/ilias/mini_ilias_yfcc100m/*.jpg'))
    meta = [{'i': i, 'path': p, 'ent': None, 'cat': None} for i, p in enumerate(paths)]
    write_jsonl(os.path.join(d, 'gallery.jsonl'), meta)
    rng = np.random.default_rng(SEED + 2)
    q = rng.choice(len(meta), 2000, replace=False)
    json.dump({'seed': SEED + 2, 'fidelity': sorted(int(x) for x in q),
               'n_gallery': len(meta)},
              open(os.path.join(d, 'queries.json'), 'w'), indent=1)
    log(f'ilias gallery={len(meta)} fidelity queries={len(q)}')


BUILDERS = {'inshop': build_inshop, 'sop': build_sop, 'roxford': build_roxford,
            'ilias': build_ilias}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', default='inshop,sop,roxford,ilias')
    a = ap.parse_args()
    os.makedirs(ROOT, exist_ok=True)
    for name in a.datasets.split(','):
        log(f'=== building {name}')
        BUILDERS[name]()
    log('done')


if __name__ == '__main__':
    main()
