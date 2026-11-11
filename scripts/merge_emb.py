"""Concatenate two encode passes into one cache.

The ImageNet encode is split by shard range (`--shard-start`), because one process over 242 shards is a
single point of failure and the second half can start as soon as the download finishes.  Merging is
row-order preserving: pass A holds the first N shards, pass B the rest, and nothing is sorted or
deduplicated, so `labels[i]` stays the class of `emb[i]` in the merged file.
"""
import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parts', nargs='+', required=True, help='directories, in shard order')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    embs, labs, dims = [], [], set()
    for d in a.parts:
        e = np.load(os.path.join(d, 'emb.npy'), mmap_mode='r')
        l = np.load(os.path.join(d, 'labels.npy'))
        assert len(e) == len(l), '%s: %d embeddings for %d labels' % (d, len(e), len(l))
        dims.add(e.shape[1])
        print('%s: %d rows, dim %d' % (d, len(e), e.shape[1]))
        embs.append(e)
        labs.append(l)
    assert len(dims) == 1, 'dimension mismatch across parts: %s' % sorted(dims)

    os.makedirs(a.out, exist_ok=True)
    n = int(sum(len(x) for x in embs))
    out = np.lib.format.open_memmap(os.path.join(a.out, 'emb.npy'), mode='w+',
                                    dtype=np.float32, shape=(n, dims.pop()))
    o = 0
    for e in embs:
        step = 200000
        for i in range(0, len(e), step):
            out[o:o + len(e[i:i + step])] = e[i:i + step]
            o += len(e[i:i + step])
    out.flush()
    labels = np.concatenate(labs)
    np.save(os.path.join(a.out, 'labels.npy'), labels)

    meta = json.load(open(os.path.join(a.parts[0], 'meta.json')))
    json.dump({'n': n, 'dim': int(out.shape[1]), 'classes': meta['classes'],
               'coarse_rule': meta.get('coarse_rule', ''), 'parts': list(a.parts)},
              open(os.path.join(a.out, 'meta.json'), 'w'))
    print('merged %d rows -> %s' % (n, a.out))


if __name__ == '__main__':
    main()
