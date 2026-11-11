"""How many ImageNet classes do the downloaded shards actually cover?

The shards are not class-balanced, so the number of distinct labels behind a prefix of the shard list is
a real constraint on the partitions (a "coarse" group with three classes in it is not coarse).  Reads
only the label column, so it costs seconds rather than minutes.
"""
import os
import sys

import numpy as np
import pyarrow.parquet as pq

from sb_paths import SB_ROOT, SB_OUT, SB_CACHE, SB_MODELS, REPO  # paths resolve relative to the deposit
DATA = (SB_ROOT
        + '/datasets/imagenet1k_parquet/data')


def main():
    want = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    files = sorted(os.listdir(DATA))
    files = [f for f in files if f.endswith('.parquet')]
    print('%d shards on disk' % len(files))
    cum, rows = [], 0
    labels = set()
    marks = {50, 100, 150, 200, 250}
    for i, f in enumerate(files, 1):
        col = pq.read_table(os.path.join(DATA, f), columns=['label']).column('label').to_numpy()
        labels.update(np.unique(col).tolist())
        rows += len(col)
        if i in marks or i == len(files):
            print('  first %3d shards: %7d rows, %4d classes' % (i, rows, len(labels)))
        if want and i >= want:
            break
    print('coverage of 1000 classes: %.1f%%' % (100.0 * len(labels) / 1000))


if __name__ == '__main__':
    main()
