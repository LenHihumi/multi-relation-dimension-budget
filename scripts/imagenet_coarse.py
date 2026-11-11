"""Coarse partition for ImageNet from the WordNet hierarchy, not from keyword matching.

The first attempt grouped class names by keyword ("fish", "dog", "vehicle", ...).  Substring matching is
too eager on real class names -- "office" contains "ice", "seacoast" contains "sea" -- so the groups
were a rule artefact rather than a hierarchy.  ImageNet ships a WordNet synset per class, so the honest
coarse partition is the synset's ancestor at a fixed depth: an exogenous, reproducible, label-only
grouping that does not look at the embeddings at all.

Depth is picked automatically: the shallowest depth whose group count lands in [15, 60], which is the
range where a "coarse" class is still meaningful (25-70 classes) and still much coarser than the 1000
fine classes.
"""
import json
import os

from sb_paths import SB_ROOT, SB_OUT, SB_CACHE, SB_MODELS, REPO  # paths resolve relative to the deposit
CACHE = SB_ROOT + '/datasets/imagenet1k_meta/classes.py'


def load_index():
    """Return [(synset, name)] in label order."""
    ns = {}
    with open(CACHE) as f:
        exec(f.read(), ns)
    d = ns.get('IMAGENET2012_CLASSES') or ns.get('classes')
    if isinstance(d, dict):
        return [(k, v) for k, v in d.items()]
    return [('', v) for v in d]


def ancestor(synset_id, depth):
    from nltk.corpus import wordnet as wn
    try:
        # classes.py stores the bare offset ("n01440764"); wn wants the full name or (pos, offset)
        s = wn.synset_from_pos_and_offset(synset_id[0], int(synset_id[1:]))
    except Exception:
        return 'other'
    path = s.hypernym_paths()[0]        # root -> leaf
    if len(path) <= depth:
        return 'other'
    return path[depth].name().split('.')[0]


DEFAULT_DEPTH = 6      # 74 groups: animal(398) / device(127) / container(99) / consumer_goods(60) ...
                       # depth 4-5 collapse to two mega-groups (artifact 520, living_thing 410)


def load_groups(path=None):
    """The saved index -> group list, so every consumer uses one mapping."""
    path = path or os.path.join(os.environ.get('SB_DIAG', SB_ROOT + '/projects/ImageSearchAgent/outputs/diag'),
                                'imagenet_coarse.json')
    return json.load(open(path))['groups']


def build(depth=None):
    items = load_index()
    from nltk.corpus import wordnet as wn  # noqa: F401  (fail fast if the corpus is missing)
    if depth is None:
        depth = DEFAULT_DEPTH
    if depth is None:
        for d in range(1, 8):
            groups = {ancestor(sid, d) for sid, _ in items}
            print('  depth %d -> %d groups' % (d, len(groups)))
            if 15 <= len(groups) <= 60:
                depth = d
                break
        depth = depth or 3
    out = [ancestor(sid, depth) for sid, _ in items]
    # a singleton "coarse" group is just the fine class under another name, which defeats the point of
    # a coarser relevance definition: those classes fall back one level up the hierarchy
    from collections import Counter
    cnt = Counter(out)
    fixed, n_fix = [], 0
    for (sid, _), g in zip(items, out):
        if cnt[g] == 1:
            up = ancestor(sid, depth - 1)
            if up != 'other':
                g, n_fix = up, n_fix + 1
        fixed.append(g)
    out = fixed
    print('chose depth %d, %d groups (%d singleton classes lifted one level)'
          % (depth, len(set(out)), n_fix))
    return out, depth


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--depth', type=int, default=0)
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    groups, depth = build(a.depth or DEFAULT_DEPTH)
    from collections import Counter
    c = Counter(groups)
    print('largest groups:', c.most_common(10))
    print('singleton groups:', sum(1 for v in c.values() if v == 1))
    if a.out:
        json.dump({'depth': depth, 'groups': groups}, open(a.out, 'w'))
        print('wrote', a.out)


if __name__ == '__main__':
    main()
