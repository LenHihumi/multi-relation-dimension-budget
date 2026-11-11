"""Standing self-checks for parse layers.

Four incidents on this project produced plausible-but-wrong numbers from a wrong field,
a wrong level or a wrong join, and in every case the wrong result looked self-consistent:

  1. `sb_abl_strata.py` bucketed with `startswith('1')`, so `16+` also matched `1-3`;
     the four strata summed to 834 queries against a population of 668.
  2. A content fingerprint matched only 385 of 668 rows (multiple queries of one entity
     share the same top-3 pool), which silently collapsed the sample and pushed accuracy
     from 0.29 to 0.45.
  3. ABO listings were read as `item['images']`, a field that does not exist; every
     listing had 0 images, all four admission criteria failed, and the corpus was
     reported as `cat only`.
  4. ABO image ids were used as S3 filenames; 298 of 298 downloads 404'd and the builder
     logged `fetch: {'fail': 298}` without raising.

Every one was caught by a number that could not be true -- never by reading the code.
So the three checks below are the standing rule: after any parse, verify conservation,
non-zero medians and field coverage BEFORE looking at the conclusion.  A triggered check
means "inspect the schema", not "adjust the threshold".
"""
from __future__ import annotations

import hashlib
import json
import os


class ParseFailure(AssertionError):
    pass


def require_conservation(parts, total, label='rows'):
    """Sum of the partitions must equal the population -- catches bucketing/join bugs."""
    s = sum(parts.values()) if isinstance(parts, dict) else sum(parts)
    if s != total:
        raise ParseFailure('%s: partitions sum to %d, population is %d' % (label, s, total))
    return s


def require_nonzero_median(values, label, hint=''):
    """A median of 0 on a positive-by-construction quantity means the field was misread."""
    v = sorted(values)
    med = v[len(v) // 2] if v else 0
    if med <= 0:
        raise ParseFailure('%s: median is %s -- this is a parse failure, not a corpus '
                           'property. %s' % (label, med, hint))
    return med


def require_coverage(flags, label, minimum):
    """Fraction of rows with a non-empty key field must clear the floor."""
    n = len(flags)
    if n == 0:
        raise ParseFailure('%s: no rows survived parsing' % label)
    frac = sum(1 for f in flags if f) / n
    if frac < minimum:
        raise ParseFailure('%s: only %.1f%% of rows carry the field (floor %.0f%%)'
                           % (label, 100 * frac, 100 * minimum))
    return frac


def require_label_matches_impl(label, claimed, actual, what=''):
    """A label must describe what the code does, not what it was meant to do.

    `label_free_90pct_variance` was the label of a fallback that actually issued
    0.25 x (largest measured rank): 224 on a cell whose 90%-variance rank is 441.  The
    reported ratios (9.33x, 7.00x) were right and the *name* was wrong, which is the
    same failure shape as the four parse bugs -- a plausible-looking result whose stated
    meaning does not match the mechanism.  It is now `fallback_quarter_max_rank`, and the
    paper text says the same thing.

    Anywhere a rule carries a name with a number in it, assert that the number the name
    claims is the number the implementation produces.

    Where `claimed` comes from -- this matters, so it is explicit: it is supplied by the
    caller, translated into the SAME UNIT as `actual`, and never parsed out of the label
    string.  A label of `label_free_90pct_variance` claims "the rank at which 90% of the
    variance is retained", so its `claimed` is that rank (441 on the DINOv3/INSTRE cell),
    not the number 90.  Parsing "90" out of the string and comparing it to a rank would
    report a unit error dressed up as a naming error.

    `actual` is always computed from the implementation path.  A mismatch therefore reads
    "the name claims rank X, the code issues rank Y" -- one sentence, one meaning.

    `claimed` is per-cell, not a constant: the 90%-variance rank is 441 on DINOv3/INSTRE and
    something else on any other cell.  An audit that reuses a stored `claimed` after
    changing the cell set reports a false naming error, so the cell set is part of the
    contract, not an implementation detail.
    """
    if claimed != actual:
        raise ParseFailure('%s: label %r claims %s but the implementation produces %s%s'
                           % (label, label, claimed, actual,
                              (' (%s)' % what) if what else ''))
    return True


def manifest_stamp(gallery_path, queries_path, extra=None):
    """Write a sidecar that identifies which build produced a corpus manifest.

    An earlier ABO manifest was written by a superseded script (list-format queries, no
    row index); without a stamp the two are indistinguishable on disk.  The stamp holds
    the row count, a content hash of the row identifiers, and the schema version, so a
    stale manifest is detectable and a rebuild is auditable.
    """
    rows = [json.loads(ln) for ln in open(gallery_path, encoding='utf-8')]
    ids = '|'.join(str(r.get('img_id')) for r in rows)
    stamp = {
        'schema_version': 2,
        'gallery_rows': len(rows),
        'gallery_id_hash': hashlib.md5(ids.encode()).hexdigest()[:16],
        'queries_file': os.path.basename(queries_path),
        'queries_schema': 'dict with row indices' ,
    }
    if extra:
        stamp.update(extra)
    with open(os.path.join(os.path.dirname(gallery_path), 'MANIFEST.json'), 'w') as f:
        json.dump(stamp, f, indent=1)
    return stamp
