"""T51b: 真实 (RPL Noritsu) 与模型 (t51_bw) 的黑白颗粒在同一输出码值上比较, 分解为 胶片侧密度颗粒 × 影调斜率。
真实胶片侧 σ48(D) = σ48(码值) / 真实斜率 (同帧扫描对底片翻拍的影调曲线); 模型同理 (σ48(码值) / 模型斜率)。
用法: python3 -I t51b_bw_compare.py t51/bw_level_real.json.gz t51/bw_level_model.json 输出.json"""
import gzip
import json
import sys
import numpy as np
R = json.load(gzip.open(sys.argv[1], 'rt') if sys.argv[1].endswith('.gz') else open(sys.argv[1]))
M = json.load(open(sys.argv[2]))
def model_at(m, code, key):
    a = sorted([(v['code'], v[key]) for v in m.values() if 15 < v['code'] < 245])
    c, y = np.array(a).T
    return float(np.interp(code, c, y)) if c[0] <= code <= c[-1] else np.nan
LV = [(40, 80), (80, 120), (120, 160), (160, 200), (200, 240)]
out = {}
for film, mkeys in (('6_KODAK400TX', ('kodak_trix_400', 'kodak_trix_400_xtol')), ('7_ILFORDHP5PLUS', ('ilford_hp5_plus',))):
    rows = []
    for k, r in R.items():
        if not k.startswith(film): continue
        tiles = np.array(r['tiles'])           # code, s_px, s_hp, s48
        tc, ts = np.array(r['tone_fit']), np.array(r['tone_slope'])
        for lo, hi in LV:
            sel = tiles[(tiles[:, 0] >= lo) & (tiles[:, 0] < hi)]
            if len(sel) < 10: continue
            c = float(np.median(sel[:, 0]))
            # 真实斜率: 该帧影调曲线在这一码值处 (曲线需单调且覆盖)
            o = np.argsort(tc)
            sl = float(np.interp(c, tc[o], ts[o])) if tc.min() <= c <= tc.max() else np.nan
            rows.append(dict(frame=k.split('/')[1].split('-')[0], lo=lo, n=len(sel), code=c, s_px=float(np.median(sel[:, 1])),
                             s_hp=float(np.median(sel[:, 2])), s48=float(np.median(sel[:, 3])), slope=sl))
    out[film] = rows
    print(film)
    for lo, hi in LV:
        g = [r for r in rows if r['lo'] == lo]
        if not g: continue
        c = np.median([r['code'] for r in g]); s48 = np.median([r['s48'] for r in g]); shp = np.median([r['s_hp'] for r in g]); spx = np.median([r['s_px'] for r in g])
        sl = np.nanmedian([r['slope'] for r in g]); slq = np.nanpercentile([r['slope'] for r in g], [25, 75])
        line = f'  code {lo}-{hi} frames {len(g)} code {c:.0f}  real s_px {spx:.2f} s_hp {shp:.2f} s48 {s48:.2f} slope {sl:.0f} [{slq[0]:.0f},{slq[1]:.0f}]  sD48_real {s48/sl*1e3:.1f}e-3'
        for mk in mkeys:
            m = M[mk]
            ms48, mhp, mpx, msl, mD = (model_at(m, c, q) for q in ('s48', 's_hp', 's_px', 'slope', 's48_Dn'))
            line += f'\n     {mk}: s_px {mpx:.2f} s_hp {mhp:.2f} s48 {ms48:.2f} slope {msl:.0f} s48_Dn {mD*1e3:.1f}e-3 (code/slope {ms48/msl*1e3:.1f}e-3) | ratio s_px {spx/mpx:.2f} s_hp {shp/mhp:.2f} s48 {s48/ms48:.2f} = film {(s48/sl)/(ms48/msl):.2f} x slope {sl/msl:.2f}'
        print(line)
json.dump(out, open(sys.argv[3], 'w'), indent=1)
