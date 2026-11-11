"""Print the text cell's numbers in the form the paper needs."""
import json
import sys

d = json.load(open(sys.argv[1]))
rows = {x['r']: x for x in d['curves']['rows']}
ranks = [x['r'] for x in d['curves']['rows']]
print('n %d dim %d groups %s' % (d['n'], d['dim'], d['groups']))
print('full-width mAP@10  item %.4f  cat %.4f' % (d['full']['parent'], d['full']['cat']))
print('r* item %s  cat %s  ratio %s  r*(fid) %s'
      % (d['rstar_parent'], d['rstar_cat'], d['ratio'], d['rstar_fid']))
print('protocol', json.dumps(d.get('protocol'), indent=1))
print()
for r in (8, 16, 24, 32, 64, 128, 192, 256, 384, 448, 512, 768, 1024, 2048):
    x = rows.get(r)
    if not x:
        continue
    i = ranks.index(r)
    print('r=%-5d rho_t %.4f/%.4f  fid %.3f  item %.4f (d %.3f)  cat %.4f (d %.3f)'
          % (r, x['rho_t_parent'], x['rho_t_cat'], x['fid'], x['map_parent'], d['delta_parent'][i],
             x['map_cat'], d['delta_cat'][i]))
bytes_ratio = (4 * d['rstar_fid'] + 256) / (4 * d['rstar_parent'] + 256)
print('\nfid budget vs item budget in bytes: %.2fx' % bytes_ratio)
