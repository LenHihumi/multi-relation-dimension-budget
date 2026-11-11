"""Where does the ImageNet encode spend its time?  One shard, stage timings, no full run.

The throughput that matters is images/s per GPU, and the pipeline has four serial stages:
pyarrow batch extraction, JPEG decode, resize, and the forward pass.  Guessing which one dominates
is how a 2-hour job becomes a 6-hour job, so this measures them on one real shard.

Read-only; prints a table and exits.
"""
import io
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from sb_paths import SB_ROOT, SB_OUT, SB_CACHE, SB_MODELS, REPO  # paths resolve relative to the deposit
DATA = SB_ROOT + '/datasets/imagenet1k_parquet/data'
SHARD = os.path.join(DATA, sorted(os.listdir(DATA))[0])
BATCH = 512
N = 2048


def to_img(b):
    return np.asarray(Image.open(io.BytesIO(b)).convert('RGB').resize((224, 224), Image.BICUBIC),
                      dtype=np.float32) / 255.0


pf = pq.ParquetFile(SHARD)
pool = ThreadPoolExecutor(max_workers=32)
t = {'extract': 0.0, 'decode': 0.0, 'stack': 0.0}
n = 0
t0 = time.time()
for batch in pf.iter_batches(batch_size=BATCH, columns=['image']):
    a = time.time()
    raw = [x['bytes'] if isinstance(x, dict) else x for x in batch.column('image').to_pylist()]
    b = time.time()
    imgs = list(pool.map(to_img, raw))
    c = time.time()
    x = np.stack(imgs)
    d = time.time()
    t['extract'] += b - a
    t['decode'] += c - b
    t['stack'] += d - c
    n += len(raw)
    if n >= N:
        break
el = time.time() - t0
tot = sum(t.values())
print('rows %d in %.2fs -> %.1f img/s' % (n, el, n / el))
for k, v in t.items():
    print('  %-8s %6.2fs  %5.1f%%  (%.1f img/s)' % (k, v, 100 * v / tot, n / max(v, 1e-9)))
print('  rest     %6.2fs  %5.1f%%' % (el - tot, 100 * (el - tot) / el))
