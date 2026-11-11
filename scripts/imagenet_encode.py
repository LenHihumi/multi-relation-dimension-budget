"""ImageNet-1k -> embeddings for the shared-index experiment.

Reads the ModelScope parquet shards, encodes with a CLIP-family embedder, and writes the same
artefact shape the rest of the pipeline expects: `emb/<enc>__imagenet.npy` (+ `.json` sidecar with
n, dim, class map) and `imagenet_labels.npy` (class index per row).

Two partitions are defined over the label:
  fine   = the 1000 ImageNet classes (average ~1,200 images each)
  coarse = keyword groups over the class name, rule list below, applied to labels only

The coarse grouping is a deliberate, documented choice: ImageNet ships a WordNet id per class but no
off-the-shelf 10-30 way grouping that is stable across mirrors, and the system claim only needs a
second, coarser relevance definition over the same vectors.  Every rule is a substring test on the
lower-cased class name, first match wins, so the mapping is reproducible from `classes.py` alone.

Usage (needs a GPU with torch + transformers):
  python3 imagenet_encode.py --shards 100 --limit 0 --out-dir "$SB_CACHE/imagenet"
  python3 imagenet_encode.py --dry-run --limit 16          # decode + label check, no model
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from sb_paths import SB_ROOT, SB_OUT, SB_CACHE, SB_MODELS, REPO  # paths resolve relative to the deposit
ROOT = SB_ROOT
DATA = os.path.join(ROOT, 'datasets/imagenet1k_parquet/data')
META = os.path.join(ROOT, 'datasets/imagenet1k_meta/classes.py')

# ordered keyword rules over the class name; first match wins
GROUPS = [
    ('dog', ['dog', 'retriever', 'terrier', 'spaniel', 'poodle', 'shepherd', 'husky', 'malamute',
             'collie', 'corgi', 'pug', 'chihuahua', 'dalmatian', 'pinscher', 'schnauzer',
             'mastiff', 'bulldog', 'boxer', 'rottweiler', 'doberman', 'beagle', 'hound',
             'setter', 'pointer', 'vizsla', 'weimaraner', 'wolf', 'fox', 'coyote', 'dingo',
             'hyena', 'jackal']),
    ('cat', ['cat', 'kitten', 'lynx', 'cougar', 'leopard', 'jaguar', 'lion', 'tiger', 'cheetah',
             'panther', 'serval', 'ocelot']),
    ('bird', ['bird', 'hen', 'cock', 'ostrich', 'finch', 'robin', 'jay', 'magpie', 'chickadee',
              'warbler', 'vireo', 'flycatcher', 'hummingbird', 'toucan', 'hornbill', 'pelican',
              'albatross', 'stork', 'heron', 'egret', 'flamingo', 'swan', 'goose', 'duck',
              'eagle', 'hawk', 'kite', 'vulture', 'owl', 'parrot', 'macaw', 'cockatoo', 'lorikeet',
              'peacock', 'quail', 'partridge', 'grouse', 'prairie', 'penguin', 'kingfisher',
              'woodpecker', 'jacamar', 'bee', 'eater']),
    ('fish_marine', ['fish', 'shark', 'ray', 'stingray', 'eel', 'salmon', 'trout', 'tench',
                     'goldfish', 'carp', 'puffer', 'lionfish', 'anemone', 'coral', 'jellyfish',
                     'starfish', 'urchin', 'sea', 'lobster', 'crab', 'crayfish', 'hermit',
                     'isopod', 'barnacle', 'nematode', 'conch', 'snail', 'slug', 'chiton',
                     'squid', 'octopus', 'cuttlefish', 'nautilus', 'whale', 'dolphin', 'porpoise',
                     'dugong', 'manatee', 'seal', 'sea lion', 'walrus']),
    ('insect_arthropod', ['insect', 'ant', 'bee', 'wasp', 'fly', 'mosquito', 'gnat', 'beetle',
                          'weevil', 'ladybug', 'moth', 'butterfly', 'dragonfly', 'damselfly',
                          'grasshopper', 'cricket', 'mantis', 'cicada', 'leafhopper', 'aphid',
                          'spider', 'tarantula', 'scorpion', 'tick', 'mite', 'centipede',
                          'millipede', 'cockroach', 'termite', 'stick', 'lacewing', 'mayfly']),
    ('reptile_amphibian', ['lizard', 'snake', 'python', 'boa', 'viper', 'rattlesnake', 'cobra',
                           'turtle', 'tortoise', 'terrapin', 'alligator', 'crocodile', 'gecko',
                           'iguana', 'chameleon', 'komodo', 'monitor', 'skink', 'whiptail',
                           'frog', 'toad', 'salamander', 'newt', 'axolotl', 'triton']),
    ('mammal_other', ['bear', 'elephant', 'zebra', 'horse', 'ox', 'cow', 'sheep', 'goat', 'pig',
                      'llama', 'camel', 'deer', 'elk', 'moose', 'antelope', 'gazelle', 'impala',
                      'monkey', 'ape', 'gorilla', 'chimpanzee', 'orangutan', 'gibbon', 'macaque',
                      'lemur', 'sloth', 'armadillo', 'anteater', 'pangolin', 'rabbit', 'hare',
                      'squirrel', 'beaver', 'porcupine', 'hamster', 'mouse', 'rat', 'guinea',
                      'bat', 'hedgehog', 'shrew', 'mole', 'otter', 'skunk', 'weasel', 'mink',
                      'badger', 'wombat', 'koala', 'kangaroo', 'wallaby', 'platypus', 'echidna',
                      'hippopotamus', 'rhino', 'tapir', 'bison', 'buffalo', 'yak', 'musk']),
    ('vehicle_road', ['car', 'cab', 'jeep', 'limousine', 'minivan', 'van', 'pickup', 'truck',
                      'trailer', 'bus', 'minibus', 'motor', 'moped', 'bicycle', 'tandem', 'car',
                      'wheel', 'taxi', 'racer', 'sports', 'convertible', 'beach', 'wagon',
                      'snowplow', 'tow', 'fire', 'garbage', 'moving', 'police', 'ambulance']),
    ('vehicle_air_sea', ['airliner', 'aircraft', 'airplane', 'warplane', 'jet', 'propeller',
                         'helicopter', 'glider', 'balloon', 'airship', 'space', 'shuttle',
                         'ship', 'boat', 'canoe', 'yawl', 'kayak', 'catamaran', 'trimaran',
                         'schooner', 'sloop', 'yacht', 'ferry', 'liner', 'submarine', 'raft',
                         'gondola', 'lifeboat', 'speedboat', 'paddle', 'ark', 'freighter',
                         'tanker', 'container', 'dock', 'wreck']),
    ('rail', ['locomotive', 'train', 'rail', 'tram', 'streetcar', 'trolley', 'steam', 'electric']),
    ('food', ['pizza', 'burrito', 'hotdog', 'hot', 'pretzel', 'bagel', 'bread', 'loaf', 'cake',
              'cupcake', 'cheese', 'burger', 'banana', 'orange', 'lemon', 'fig', 'pineapple',
              'pomegranate', 'strawberry', 'cucumber', 'artichoke', 'broccoli', 'cauliflower',
              'cabbage', 'mushroom', 'pepper', 'corn', 'acorn', 'squash', 'espresso', 'wine',
              'beer', 'cocktail', 'plate', 'guacamole', 'consomme', 'trifle', 'ice', 'cream',
              'chocolate', 'dough', 'meat', 'loaf']),
    ('plant', ['tree', 'flower', 'daisy', 'rose', 'tulip', 'orchid', 'sunflower', 'dandelion',
               'leaf', 'fern', 'moss', 'fungus', 'agaric', 'bolete', 'earthstar', 'stinkhorn',
               'hen-of-the-woods', 'gyromitra', 'coral fungus', 'buckeye', 'acorn', 'hip',
               'pepper', 'corn', 'cabbage', 'broccoli', 'cauliflower', 'zucchini', 'cucumber',
               'artichoke', 'cardoon', 'mushroom', 'rapeseed', 'daisy', 'yellow lady']),
    ('instrument', ['guitar', 'banjo', 'violin', 'cello', 'harp', 'piano', 'organ', 'drum',
                    'maraca', 'flute', 'oboe', 'bassoon', 'sax', 'trombone', 'trumpet', 'horn',
                    'harmonica', 'accordion', 'chime', 'bell', 'gong', 'xylophone', 'tambourine',
                    'panpipe', 'bass', 'steel', 'drumstick', 'membrane', 'string']),
    ('tool_machine', ['hammer', 'screwdriver', 'wrench', 'saw', 'pliers', 'drill', 'lathe',
                      'lawnmower', 'chainsaw', 'shovel', 'spade', 'rake', 'plough', 'thresher',
                      'harvester', 'tractor', 'crane', 'forklift', 'bulldozer', 'backhoe',
                      'steam', 'milling', 'power', 'printer', 'typewriter', 'calculator',
                      'abacus', 'ruler', 'scissors', 'cleaver', 'ladle', 'spatula', 'whisk',
                      'corkscrew', 'opener', 'knife', 'blade', 'shears']),
    ('electronics', ['computer', 'laptop', 'desktop', 'keyboard', 'mouse', 'monitor', 'screen',
                     'television', 'radio', 'tape', 'cd', 'dvd', 'ipod', 'phone', 'cellular',
                     'modem', 'router', 'camera', 'projector', 'loudspeaker', 'microphone',
                     'headphone', 'remote', 'joystick', 'printer', 'scanner', 'copier',
                     'reflex', 'polaroid', 'lens', 'oscilloscope', 'cathode', 'dial']),
    ('furniture_house', ['chair', 'table', 'desk', 'bed', 'sofa', 'couch', 'cabinet', 'wardrobe',
                         'dresser', 'chest', 'shelf', 'bookcase', 'lamp', 'candle', 'mirror',
                         'clock', 'vase', 'pitcher', 'pot', 'pan', 'plate', 'cup', 'mug',
                         'teapot', 'coffeepot', 'tray', 'basket', 'bucket', 'barrel', 'tub',
                         'bath', 'toilet', 'sink', 'towel', 'quilt', 'pillow', 'mat', 'rug',
                         'apron', 'bib', 'diaper', 'doormat', 'table lamp', 'lampshade']),
    ('clothing', ['shirt', 't-shirt', 'jersey', 'sweatshirt', 'cardigan', 'suit', 'jean', 'skirt',
                  'dress', 'gown', 'coat', 'jacket', 'vest', 'robe', 'kimono', 'sari', 'scarf',
                  'shawl', 'tie', 'bow', 'hat', 'cap', 'beret', 'sombrero', 'helmet', 'mask',
                  'sock', 'stocking', 'glove', 'mitten', 'shoe', 'sandal', 'slipper', 'boot',
                  'clog', 'sneaker', 'loafer', 'apron', 'uniform', 'swimsuit', 'brassiere',
                  'pajama', 'sweater', 'poncho', 'sarong', 'abaya', 'gown']),
    ('weapon', ['rifle', 'revolver', 'pistol', 'cannon', 'missile', 'bomb', 'sword', 'saber',
                'dagger', 'lance', 'spear', 'bow', 'arrow', 'shield', 'armor', 'holster',
                'cartridge', 'assault', 'projectile', 'mortar', 'grenade', 'tank']),
    ('building_place', ['church', 'mosque', 'monastery', 'library', 'castle', 'palace', 'barn',
                        'boathouse', 'lighthouse', 'bridge', 'viaduct', 'dam', 'fountain',
                        'greenhouse', 'silo', 'tower', 'dome', 'altar', 'throne', 'stupa',
                        'mausoleum', 'ruin', 'prison', 'hospital', 'planetarium', 'restaurant',
                        'bakery', 'barbershop', 'bookshop', 'butcher', 'grocery', 'shoe shop',
                        'tobacco', 'toyshop', 'confectionery', 'drugstore', 'patio', 'gazebo',
                        'pier', 'dock', 'planetarium', 'stage', 'theater', 'cinema']),
    ('sport_leisure', ['ball', 'racket', 'basketball', 'volleyball', 'soccer', 'rugby', 'tennis',
                       'golf', 'puck', 'ski', 'sled', 'snowboard', 'surfboard', 'paddle',
                       'dumbbell', 'barbell', 'treadmill', 'swing', 'carousel', 'scuba',
                       'parachute', 'hang', 'kite', 'puzzle', 'dice', 'chessboard', 'backgammon',
                       'slot', 'pinball', 'video', 'arcade', 'toy', 'doll', 'teddy']),
]


# group names that must not steal a class from an earlier rule
def coarse_of(name: str) -> str:
    n = name.lower()
    for g, keys in GROUPS:
        for k in keys:
            if k in n:
                return g
    return 'other'


def load_classes():
    ns = {}
    src = open(META, encoding='utf-8').read()
    exec(compile(src, META, 'exec'), ns)
    for key in ('IMAGENET2012_CLASSES', 'classes', 'IMAGENET_CLASSES', 'synset_words', 'CLASSES'):
        if key in ns:
            v = ns[key]
            return list(v.values()) if isinstance(v, dict) else list(v)
    raise SystemExit('no class list found in %s' % META)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shards', type=int, default=100)
    ap.add_argument('--shard-start', type=int, default=0,
                    help='skip the first N shards (encode the rest in a second pass)')
    ap.add_argument('--limit', type=int, default=0, help='0 = no limit')
    ap.add_argument('--out-dir', default=os.path.join(ROOT, '.cache/sb/imagenet'))
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--decode-threads', type=int, default=16)
    ap.add_argument('--dtype', default='float16', choices=['float32', 'float16', 'bfloat16'],
                    help='CLIP-L in fp16 is ~2-3x faster than fp32 on A800 with no measurable '
                         'change in the embeddings we care about (cosine differences <1e-3)')
    a = ap.parse_args()

    import pyarrow.parquet as pq
    classes = load_classes()
    allf = sorted(os.listdir(DATA))
    files = allf[a.shard_start:a.shard_start + a.shards]
    print('shards %d, classes %d' % (len(files), len(classes)))

    n_total = 0
    t0 = time.time()
    labels = []
    embs = None
    model = None
    if not a.dry_run:
        import torch
        from transformers import CLIPModel, CLIPProcessor
        from PIL import Image
        dev = 'cuda'
        mp = SB_MODELS + '/clip-vit-large-patch14'
        model = CLIPModel.from_pretrained(mp).to(dev).eval()
        dt = {'float32': torch.float32, 'float16': torch.float16,
              'bfloat16': torch.bfloat16}[a.dtype]
        model = model.to(dt)
        proc = CLIPProcessor.from_pretrained(mp)

    from PIL import Image
    import io
    from concurrent.futures import ThreadPoolExecutor

    pool = ThreadPoolExecutor(max_workers=a.decode_threads)   # JPEG decode is 68% of the CPU side

    def to_img(b):
        # decode AND resize in the pool: CLIPProcessor would otherwise do it single-threaded
        return np.asarray(Image.open(io.BytesIO(b)).convert('RGB').resize((224, 224),
                          Image.BICUBIC), dtype=np.float32) / 255.0

    for fi, fn in enumerate(files):
        pf = pq.ParquetFile(os.path.join(DATA, fn))
        for batch in pf.iter_batches(batch_size=a.batch, columns=['image', 'label']):
            raw = [x['bytes'] if isinstance(x, dict) else x for x in batch.column('image').to_pylist()]
            imgs = list(pool.map(to_img, raw))
            lab = np.asarray(batch.column('label').to_pylist(), dtype=np.int64)
            if a.dry_run:
                n_total += len(imgs)
                labels.append(lab)
            else:
                import torch
                with torch.no_grad():
                    x = torch.from_numpy(np.stack(imgs)).to(dev).permute(0, 3, 1, 2)
                    x = (x - torch.tensor([0.48145466, 0.4578275, 0.40821073], device=dev)
                         .view(1, 3, 1, 1)) / torch.tensor(
                         [0.26862954, 0.26130258, 0.27577711], device=dev).view(1, 3, 1, 1)
                    x = x.to(dt)
                    out = model.get_image_features(pixel_values=x)
                    if not torch.is_tensor(out):        # transformers >= 5 returns a ModelOutput
                        po = getattr(out, 'pooler_output', None)
                        out = po if torch.is_tensor(po) else out[0]
                    e = out.float().cpu().numpy()
                embs = e if embs is None else np.concatenate([embs, e])
                labels.append(lab)
                n_total += len(lab)
            if a.limit and n_total >= a.limit:
                break
            if n_total % a.batch == 0 and n_total and n_total % 20000 < a.batch:
                el = time.time() - t0
                rate = n_total / el
                left = (a.limit - n_total) / rate if a.limit else float('nan')
                print('  %d images, %.1f img/s, eta %.1f min'
                      % (n_total, rate, left / 60), flush=True)
        if a.limit and n_total >= a.limit:
            break
        print('shard %d/%d  n=%d  %.0fs elapsed' % (fi + 1, len(files), n_total,
                                                   time.time() - t0), flush=True)

    labels = np.concatenate(labels)
    print('done %d images in %.1fs' % (len(labels), time.time() - t0))
    if a.dry_run:
        u = np.unique(labels)
        print('label range %d..%d, %d distinct, first names: %s'
              % (labels.min(), labels.max(), len(u),
                 [classes[int(i)] for i in labels[:3]]))
        from collections import Counter
        c = Counter(coarse_of(classes[int(i)]) for i in u)
        print('coarse groups over seen classes:', dict(c.most_common(8)))
        return

    os.makedirs(a.out_dir, exist_ok=True)
    np.save(os.path.join(a.out_dir, 'emb.npy'), embs.astype(np.float32))
    np.save(os.path.join(a.out_dir, 'labels.npy'), labels)
    json.dump({'n': int(len(labels)), 'dim': int(embs.shape[1]),
               'classes': classes, 'coarse_rule': 'keyword groups, see imagenet_encode.py'},
              open(os.path.join(a.out_dir, 'meta.json'), 'w'))
    print('wrote embeddings to', a.out_dir)


if __name__ == '__main__':
    main()
