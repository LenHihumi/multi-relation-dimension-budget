"""Embed the text corpus through an already-running vLLM pooling server.

Text only: WeMM-Embedding-2B is a multimodal encoder, and this call sends strings, which exercises
its text tower.  Nothing is loaded here, so the script cannot leak GPU memory.

Usage:
    python3 sb_text_embed.py --corpus corpus.jsonl --out emb --url http://host:30040
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import requests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus', required=True)
    ap.add_argument('--out', required=True, help='prefix; writes <out>.npy and <out>.json')
    ap.add_argument('--url', default=os.environ.get('SB_LLM_URL', 'http://localhost:8000/v1'),
                    help='OpenAI-compatible endpoint serving the text encoder')
    ap.add_argument('--model', default='WeMM-Embedding-2B')
    ap.add_argument('--batch', type=int, default=32)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.corpus, encoding='utf-8')]
    texts = [r['text'] for r in rows]
    print('docs %d' % len(texts))

    # the served model name must be the one we asked for, otherwise a stale server hijacks the port
    m = requests.get(a.url + '/v1/models', timeout=20).json()['data'][0]
    print('server model %s root %s' % (m['id'], m.get('root')))
    assert m['id'] == a.model, 'unexpected model on this port'

    out, dim = [], None
    t0 = time.time()
    for s in range(0, len(texts), a.batch):
        chunk = texts[s:s + a.batch]
        for attempt in range(4):
            try:
                r = requests.post(a.url + '/v1/embeddings',
                                  json={'model': a.model, 'input': chunk}, timeout=180)
                r.raise_for_status()
                break
            except Exception as e:
                if attempt == 3:
                    raise
                print('retry %d after %s' % (attempt, e), flush=True)
                time.sleep(2 + 2 * attempt)
        data = r.json()['data']
        data.sort(key=lambda d: d['index'])
        emb = np.asarray([d['embedding'] for d in data], dtype=np.float32)
        dim = emb.shape[1]
        out.append(emb)
        if (s // a.batch) % 200 == 0:
            el = time.time() - t0
            print('  %6d/%d  %.1f docs/s  eta %4.1f min'
                  % (s + len(chunk), len(texts), (s + len(chunk)) / max(el, 1e-9),
                     (len(texts) - s) / max((s + len(chunk)) / max(el, 1e-9), 1e-9) / 60), flush=True)

    E = np.concatenate(out, 0)
    np.save(a.out + '.npy', E)
    json.dump({'n': int(E.shape[0]), 'dim': int(E.shape[1]), 'model': a.model}, open(a.out + '.json', 'w'))
    print('wrote %s.npy  %s  in %.1f min' % (a.out, E.shape, (time.time() - t0) / 60))


if __name__ == '__main__':
    main()
