"""Admission check for corpora proposed to extend the study -- metadata only, no images.

The paper's claim rests on one contrast: the same spectrum needs 3.3x the rank for item
retrieval as for category retrieval.  A new corpus is only useful if it can carry at
least one of those two relations, and ideally both.  So the check is not "does it have
labels" but "which relations can be built from its annotation, and are the groups large
enough to hold a 1000-query evaluation".

Nothing is downloaded here.  `admission_candidates.json` carries the published counts with
their sources; every count is marked verified=false until it is re-derived from the actual
annotation files, which must happen before any embedding is extracted.  The point of
running this first is to drop candidates that cannot carry the relation at all, so the
download and the GPU pass are not spent on them.

Admission rules (fixed before looking at any run):
  A1  at least one relation with >= min_images_per_group images per group
  A2  gallery >= min_queries, so a 1000-query evaluation can be built without overlap
  A3  the set as a whole contains a fine cell (>1000 groups) and a coarse cell (<100 groups)
      -- otherwise the granularity contrast cannot be re-tested on new corpora
  A4  the entity layer must be real: different physical objects, not different photos of
      the same class.  A class of 50 crawled brand images is not 50 views of one object.

A corpus failing A4 can still enter, but only on the category side, and the extension then
cannot re-test the 3.3x contrast on it -- that must be stated in the pre-registration
rather than discovered afterwards.

Usage: python sb_admission.py --out admission.json
"""
from __future__ import annotations

import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
D = os.path.join(ROOT, 'data')
R = os.path.join(D, 'results')


def relations_of(c):
    """Relations that can actually be built, and why the others cannot."""
    out = {}
    for key, r in c['relations'].items():
        if r is None:
            out[key] = None
            continue
        ok = r['groups'] > 0
        out[key] = {'name': r['name'], 'groups': r['groups'], 'usable': ok}
    return out


def judge(c, req):
    rel = relations_of(c)
    has_inst = bool(rel.get('inst') and rel['inst']['usable'])
    has_cat = bool(rel.get('cat') and rel['cat']['usable'])
    has_cat1 = bool(rel.get('cat1') and rel['cat1']['usable'])
    reasons = []

    # A1 / A4: the entity layer must be real, not a class of crawled photos
    if c.get('entity_layer'):
        if not has_inst:
            reasons.append('entity_layer flagged but no item-level relation parsed')
        elif c.get('images_per_group_median', 0) < req['min_images_per_group']:
            # the median, not the minimum: a long tail of single-image listings does not
            # kill the relation as long as a solid half of the items carry two or more
            reasons.append('median images per item below %d' % req['min_images_per_group'])
    else:
        if has_inst:
            reasons.append('item relation present but entity_layer=false: counts disagree')

    # A2: can a 1000-query evaluation be built
    if c['images'] < req['min_queries']:
        reasons.append('fewer than %d images, cannot build a 1000-query evaluation'
                       % req['min_queries'])

    usable = [k for k, v in rel.items() if v and v['usable']]
    if not usable:
        reasons.append('no relation can be built')

    return {'relations': rel, 'usable_relations': usable,
            'can_retest_granularity_contrast': has_inst and has_cat and not reasons,
            'side': 'both' if (has_inst and has_cat) else
                    ('inst only' if has_inst else ('cat only' if (has_cat or has_cat1) else 'none')),
            'admitted': len(reasons) == 0, 'blocking': reasons}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='admission.json')
    args = ap.parse_args()

    spec = json.load(open(os.path.join(D, 'admission_candidates.json')))
    req = spec['requirements']
    rows = []
    for c in spec['corpora']:
        j = judge(c, req)
        rows.append({'name': c['name'], 'domain': c['domain'], 'images': c['images'],
                     'unverified_counts': not c.get('verified', False),
                     # every structural claim carries where it came from and whether we
                     # re-derived it; "structurally possible" must never read as "we checked"
                     'evidence': c.get('evidence') or {
                         'claim': 'relation structure as published',
                         'source': (c.get('sources') or [None])[0],
                         'kind': 'third_party_documentation',
                         'verified_by_us': bool(c.get('verified', False))},
                     **j})
    fine = [r for r in rows if any(v and v['usable'] and v['groups'] >= req['fine_group_target']
                                   for v in r['relations'].values())]
    coarse = [r for r in rows if any(v and v['usable'] and v['groups'] <= req['coarse_group_target']
                                     for v in r['relations'].values())]
    both = [r for r in rows if r['can_retest_granularity_contrast']]

    # a corpus whose counts are still unverified cannot be counted as a confirmed `both`,
    # however promising its published averages are: the pre-registered criterion is a
    # measured median, not an average from the dataset paper
    measured_both = [r for r in both if not r['unverified_counts']]
    out = {'requirements': req,
           'n_candidates': len(rows),
           'n_admitted': sum(1 for r in rows if r['admitted']),
           'set_level': {'fine_cells': [r['name'] for r in fine],
                         'coarse_cells': [r['name'] for r in coarse],
                         'can_retest_granularity_contrast': [r['name'] for r in both],
                         'A3_satisfied': bool(fine) and bool(coarse),
                         'contrast_on_new_corpora_possible': bool(both),
                         'contrast_structural': [r['name'] for r in both],
                         'contrast_measured': [r['name'] for r in measured_both]},
           'rows': rows}

    print('%-18s %8s %-10s %-28s %6s %s'
          % ('corpus', 'images', 'side', 'usable relations', 'admit', 'blocking'))
    for r in rows:
        ur = ', '.join('%s(%d)' % (r['relations'][k]['name'], r['relations'][k]['groups'])
                       for k in r['usable_relations'])
        print('%-18s %8d %-10s %-28s %6s %s'
              % (r['name'], r['images'], r['side'], ur,
                 'yes' if r['admitted'] else 'NO', '; '.join(r['blocking'])))

    sl = out['set_level']
    print('\nset level: fine %s | coarse %s | A3 %s'
          % (sl['fine_cells'], sl['coarse_cells'], sl['A3_satisfied']))
    print('can re-test the inst-vs-cat contrast, structurally: %s'
          % (sl['contrast_structural'] or 'NONE'))
    print('  measured from annotation files (counts verified): %s'
          % (sl['contrast_measured'] or 'NONE -- every candidate is still unverified'))
    print('all counts unverified until re-derived from annotation files: %s'
          % all(r['unverified_counts'] for r in rows))

    with open(os.path.join(R, args.out), 'w') as f:
        json.dump(out, f, indent=1)
    print('\nwrote %s' % os.path.join(R, args.out))


if __name__ == '__main__':
    main()
