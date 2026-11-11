"""Print the four-axis cost table (bytes / build / append / latency / RSS) from multidim JSONs."""
import json
import os
import sys

from sb_paths import SB_ROOT, SB_OUT, SB_CACHE, SB_MODELS, REPO  # paths resolve relative to the deposit
O = SB_OUT
ORDER = ['umax-f32', 'umax-i8', 'share-f32', 'share-i8']


def main(cell):
    d = json.load(open(os.path.join(O, 'multidim__%s.json' % cell)))
    print('%% %s  n=%d r_item=%d r_cat=%d  appended=%d' % (cell, d['n'], d['r_item'], d['r_cat'],
                                                           d['appended']))
    L = 2 * 32 * 4
    ri, rc = d['r_item'], d['r_cat']
    # same convention as the byte model: `share` keeps a float32 coarse graph and an int8 store,
    # `share-i8` quantises the graph too
    BYTES = {'umax-f32': 4 * ri + L, 'umax-i8': ri + L,
             'share-f32': 4 * rc + ri + L, 'share-i8': rc + ri + L}
    for k in ORDER:
        v = d['designs'].get(k)
        if not v:
            continue
        print(r'%-14s & %4d & %6.1f & %5.2f & %6.3f & %6.3f & %6.0f & %6.0f \\' % (
            '\\texttt{%s}' % k, BYTES[k], v['build_s'], v['append_s'], v['p50_ms'], v['p99_ms'],
            v['qps32'], v['rss_mb']))
    if 'regroup_s' in d:
        print('%% regroup: ' + json.dumps(d['regroup_s']))
    if 'nest' in d:
        print('%% nest: ' + json.dumps(d['nest']))


if __name__ == '__main__':
    for c in (sys.argv[1:] or ['clip_vitl14__inshop']):
        main(c)
