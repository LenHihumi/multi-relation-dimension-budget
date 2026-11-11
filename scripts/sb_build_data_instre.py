"""Third (non-product) domain for the spectral-budget study: INSTRE.

INSTRE is an *instance retrieval* benchmark photographed in the wild (street
scenes, shop shelves, indoor clutter) as opposed to the studio packshots of
In-shop and SOP.  Two official subsets are used as gallery groups:

    INSTRE-S1  100 directories, one *physical object* each (a specific book,
               toy, snack packet) photographed in many scenes.
    INSTRE-S2  100 directories, one *brand / landmark* each (coca_cola,
               Eiffel, Mickey_Mouse), i.e. many physical instances of one
               identity.

Grouping ladder written into the manifest (coarsest -> finest):

    cat1  12 semantic super-categories  (hand taxonomy over the 200 dir names,
          listed verbatim in SUPCAT below -- this is the only hand label here)
    cat   cat1 x subset                (24 groups: semantic class within S1/S2)
    ent   the directory itself         (200 groups, the instance relation)

The hand taxonomy exists because INSTRE ships exactly one level of labels; the
rule is a fixed keyword table, not a per-image judgement, and it is reproduced
below so the split is auditable.

Gallery: first PER_OBJ images (sorted) of each of the 200 directories, so whole
groups are kept and every group has the same size.  200 x 40 = 8000 images,
which is the CPU-encoding cost bracket (A100s are occupied by other users).

Queries: for the instance and category relations the queries are gallery
members themselves (same convention as In-shop in sb_build_data.py): a query's
relevant set is the rest of its group, and the query row is excluded from
scoring.  No extra encoding is needed for this, which matters on CPU.

Outputs (same format as sb_build_data.py):
    $SB_INSTRE_GALLERY/gallery.jsonl   {"i","path","ent","cat","cat1","name","subset"}
    $SB_INSTRE_GALLERY/queries.json    {"inst":[...], "seed":int, "n_gallery":int}

Run:
    python3 sb_build_data_instre.py [--per-obj 40]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import time
from collections import Counter

import numpy as np

from sb_paths import SB_CACHE, SB_ROOT

ROOT = os.environ.get('SB_INSTRE_GALLERY', os.path.join(SB_CACHE, 'galleries', 'instre'))
DS = os.environ.get('SB_DATASETS', os.path.join(SB_ROOT, 'datasets'))
REL = os.environ.get('SB_INSTRE_RELEASE', os.path.join(DS, 'instre', 'INSTRE_release'))
SEED = 20261103
N_QUERY = 1000

# --------------------------------------------------------------------------
# hand taxonomy over the 200 INSTRE directory names (12 super-categories)
# --------------------------------------------------------------------------
SUPCAT = {
    # --- S1 (single-object) -------------------------------------------------
    'book_stationery': [
        '01a_canada_book', '01b_panda_book', '02a_parchis', '05b_notebook',
        '10b_prince_book', '11b_postcard', '12a_poker', '12b_mizimizi',
        '19a_correction_tape', '27a_china_unicom_card', '27b_china_mobile_card',
        '30a_Nutrient_Book', '31b_Blue_notebook', '32a_Glue_stick',
        '32b_bus_card', '34a_Caculator', '37a_crazybird_notes', '37b_zhangjunya',
        '38a_new_year_pic', '38b_stamp', '45a_5yuan', '45b_1yuan',
        '48b_pear_book', '49a_AstickMini',
        # S2
        'Renren_bookmark', 'Taiwan_EasyCard', 'panda_calendar',
    ],
    'toy_figure': [
        '03a_toy_oldman', '03b_toy_taobao', '04a_matryoshka', '04b_toy_youth',
        '05a_foxtoy', '06a_caterpillar', '07a_toy_phone', '07b_toy_cow',
        '09a_toy_deer', '09b_clay_figurine', '10a_hello_kitty', '13a_toy_car',
        '13b_toy_man', '15a_dwarf', '15b_porcelain_cat', '16a_toy_woman',
        '17b_toy_cat', '18a_Doraemon', '18b_toy_dog', '19b_dog_sharpener',
        '35a_toy_jeep', '35b_toy_tractor', '36a_disney_mat', '40a_toy_carman',
        '42a_Luffy', '42b_Chopper', '46a_god_of_fortune', '47a_toy_golden_fish',
        '47b_wierd_fish', '50a_toy_snowman', '49b_HelloPanda',
        # S2
        'toy_girl', 'toy_m_man', 'toy_snake', 'toy_tiger', 'toy_whale',
        'rubber_duck', 'mario', 'walle',
    ],
    'cartoon_character': [
        'Donald_Duck', 'Goofy', 'Mickey_Mouse', 'pooh', 'SpongeBob',
        'Kung_Fu_Panda', 'Tencent_Penguin', 'V4Vendetta_mask', 'glice_superman',
        'R2D2_mailbox', 'Moominvalley_Biscuit',
    ],
    'food_snack': [
        '11a_crunch', '14a_pocky', '14b_Nestle', '22a_Linden_honey',
        '22b_Oreo_cookies', '25a_Six_walnut', '26a_hersheys', '26b_run_hou_tang',
        '29a_Extra_Chewing_Gum',
        '29b_Pretz', '31a_Aodiao_Chocolate_Roll', '34b_Chips_Ahoy', '36b_Lycium_chinensis',
        '39a_CRISPY', '39b_yuji',
        # S2
        'Cheese_Peanuts', 'Crown_Biscuit', 'FUJIYA', 'Jose_Carioca',
        'Moominvalley_Biscuit', 'pringles', 'Smart_Milk', 'pangaoshou',
    ],
    'beverage': [
        '20a_coconut_juice', '24a_huiyuan_juice', '25b_nescafe', '41a_SevenUp',
        '41b_mrbrown_coffee', 'iced_black_tea', 'costa_coffee', 'starbucks',
        'Suntory_Caffe_Latte', 'coca_cola',
    ],
    'packaged_goods': [
        '02b_plate', '08a_crab', '17a_flowerpot', '20b_pechoin',
        '23a_Yunnan_Baiyao_Aerosol', '23b_Verbatim', '28a_Thermometer',
        '28b_TopStrong_Cup', '30b_Corn_Shape_Pothook', '33a_Alice_Guitar_String_Packaging',
        '33b_Pills_package', '40b_oldman_candle', '48a_502', '50b_sprike',
        # S2
        'nice_soap', 'iphone', 'BenQ_remote', 'Yeale', 'wangzai', 'darlie',
        'Doublemint', 'JDB', 'u_loveit', 'may_wind', 'mooncake_box',
    ],
    'brand_logo': [
        '06b_red_car', '44a_red_army', '21b_crest', '43a_opera_face',
        # S2
        'Adidas_Originals', 'Ali', 'Autoland', 'baidu', 'Burger_King',
        'Chick_Fil_A', 'Dunkin_Donuts', 'einstein_bros', 'FedEX', 'intel',
        'KFC', 'mastermind_JAPAN', 'Paul_Frank_Julius', 'seven_eleven',
        'subway_food', 'Taco_Bell', 'BarcelonaFC', 'Boston_Celtics', 'chelsea',
        'madrid_bear', 'worldCup2010_logo', 'Olympic_rings', 'zooland',
    ],
    'architecture': [
        '16b_taiwan101', '43b_garden_expo',
        # S2
        'akropoli', 'Colosseum', 'dancing_house', 'Eiffel', 'Elizabeth_Tower',
        'LingLong_Pagoda', 'National_Stadium', 'Notre_Dame_de_Paris',
        'Peking_Univ_Boya_Tower', 'Taj_Mahal', 'Temple_of_Heaven',
        'Torre_di_Pisa', 'Triumphal_Arch', 'Tsinghua_old_gate', 'Tiananmen',
    ],
    'monument_statue': [
        '08b_DDog', '21a_deli_penguin', '24b_fish_sticker', '44b_tiny_girl',
        '46b_smiling_boy', '49b_HelloPanda',
        # S2
        'Chairman_Mao_Statue', 'Che_Guevara', 'Manneken_Pis', 'Mount_Rushmore',
        'Putuo_Mountain_Avalokitesvara', 'Statue_of_Liberty', 'turinng_torso',
        'navvy', 'stop_sign', 'road_sign_school', 'clapperboard',
    ],
    'real_person': ['einstein_bros', 'Obama', 'Steven_Jobs'],
    'flag_emblem': [
        'Korean_flag', 'Malaysia_flag', 'UK_flag', 'worldCup2010_logo',
        'Olympic_rings',
    ],
    'plant_animal': [
        '01b_panda_book', 'Sunflower', 'panda_calendar', 'toy_snake', 'toy_tiger',
        'toy_whale',
    ],
}

# names that appear in more than one candidate list get exactly one home; the
# first match in this priority order wins, and duplicates are dropped silently
PRIORITY = ['architecture', 'monument_statue', 'cartoon_character', 'toy_figure',
            'real_person', 'flag_emblem', 'plant_animal', 'food_snack',
            'beverage', 'book_stationery', 'packaged_goods', 'brand_logo']


def log(m):
    print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)


def taxonomy():
    out = {}
    for cat in PRIORITY:
        for name in SUPCAT[cat]:
            out.setdefault(name, cat)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-obj', type=int, default=40)
    a = ap.parse_args()
    os.makedirs(ROOT, exist_ok=True)

    tax = taxonomy()
    meta = []
    for subset, sub in (('S1', 'INSTRE-S1'), ('S2', 'INSTRE-S2')):
        dirs = sorted(d for d in os.listdir(os.path.join(REL, sub))
                      if os.path.isdir(os.path.join(REL, sub, d)))
        log(f'{sub}: {len(dirs)} directories')
        missing = [d for d in dirs if d not in tax]
        if missing:
            raise SystemExit(f'unmapped directories: {missing}')
        for d in dirs:
            fs = sorted(glob.glob(os.path.join(REL, sub, d, '*.jpg')))
            fs = fs[:a.per_obj]
            sup = tax[d]
            for f in fs:
                meta.append({'i': len(meta), 'path': f,
                             'ent': f'{subset}/{d}',
                             'cat': f'{sup}|{subset}',
                             'cat1': sup,
                             'name': d, 'subset': subset})
            if not fs:
                log(f'  WARNING empty {d}')

    ents = Counter(m['ent'] for m in meta)
    cats = Counter(m['cat'] for m in meta)
    cat1 = Counter(m['cat1'] for m in meta)
    log(f'gallery={len(meta)} instances={len(ents)} '
        f'per-instance min={min(ents.values())} median={statistics.median(ents.values())}')
    log(f'cat={len(cats)} median={statistics.median(cats.values())} '
        f'cat1={len(cat1)} median={statistics.median(cat1.values())} '
        f'min={min(cat1.values())} max={max(cat1.values())}')
    for k, v in sorted(cat1.items(), key=lambda x: -x[1]):
        log(f'   {k:20s} {v:5d}')
    with open(os.path.join(ROOT, 'gallery.jsonl'), 'w', encoding='utf-8') as fh:
        for r in meta:
            fh.write(json.dumps(r, ensure_ascii=False) + '\n')

    rng = np.random.default_rng(SEED)
    cand = np.array([i for i, m in enumerate(meta) if ents[m['ent']] >= 2])
    q = rng.choice(cand, min(N_QUERY, len(cand)), replace=False)
    sub = Counter(meta[int(i)]['subset'] for i in q)
    json.dump({'seed': SEED, 'inst': sorted(int(x) for x in q),
               'n_gallery': len(meta)}, open(os.path.join(ROOT, 'queries.json'), 'w'))
    log(f'queries={len(q)} split={dict(sub)}')
    log('done')


if __name__ == '__main__':
    main()
