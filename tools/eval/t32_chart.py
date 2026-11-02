import numpy as np, sys
TL, TR, BL, BR = np.array([240, 302.]), np.array([900, 290.]), np.array([250, 627.]), np.array([902, 620.])
NAMES = [["棕红","浅橙","灰褐","橄榄绿","浅粉","浅黄绿","橙","灰蓝","橙红"],
         ["紫","浅黄","橙黄","蓝","黄绿","红","黄","品红","浅黄绿2"],
         ["橙2","浅黄2","浅橙红","绿","米黄","浅肤","红橙","浅黄3","蓝紫"],
         ["红2","青绿","紫2","品红红","黄2","青灰","青","橙3","浅橙2"],
         ["灰1","灰2","灰3","灰4","灰5","灰6","灰7","灰8","灰9"]]
def centers():
    out = {}
    for r in range(5):
        for c in range(9):
            u, v = c / 8, r / 4
            p = (1-u)*(1-v)*TL + u*(1-v)*TR + (1-u)*v*BL + u*v*BR
            out[(r, c)] = p
    return out
def patches(z, keys=('inp','real','out','logH','dye','Dn'), h=15):
    res = {}
    for (r, c), p in centers().items():
        x, y = int(round(p[0])), int(round(p[1]))
        res[(r, c)] = {k: np.nanmedian(z[k][y-h:y+h+1, x-h:x+h+1].astype(float).reshape(-1, z[k].shape[-1]), 0) for k in keys}
        res[(r, c)]['core'] = float(z['core'][y-h:y+h+1, x-h:x+h+1].mean())
    return res
if __name__ == "__main__":
    z = np.load(sys.argv[1]); P = patches(z)
    for c in range(9):
        d = P[(4, c)]
        print(NAMES[4][c], 'inp', d['inp'].round(3), 'real', d['real'].round(3), 'out', d['out'].round(3), 'core', d['core'])
