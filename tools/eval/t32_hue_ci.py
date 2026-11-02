"""每帧: 胶片中性像素 (模型三层 logH 极差 < TOL) → 单通道传递 F_c (真实 / 模型各自); 彩色像素按胶片曝光色相分组,
比较 实际输出 / 单通道预测。彩度比 = log 输出的通道极差 ÷ 单通道预测的通道极差 (<1 去饱和, >1 增饱和)。"""
import numpy as np, sys, warnings
warnings.filterwarnings('ignore')
from scipy import ndimage as ndi
G0 = np.array([-0.83960607, -0.84922118, -0.8927258])
TOL = 0.05
HUES = [("红", 0), ("黄", 60), ("绿", 120), ("青", 180), ("蓝", 240), ("品红", 300)]
def load(f):
    z = np.load(f)
    sm = lambda a: np.stack([ndi.uniform_filter(np.nan_to_num(a[..., c].astype(float), nan=-9), 5) for c in range(a.shape[-1])], -1)
    d = {k: sm(z[k]) for k in ('inp','real','out','logH','Dn','dye')}
    d['lh'] = d['logH'] - G0
    far = ndi.distance_transform_edt(~z['core'])*z['px_um'] > 500
    d['ok'] = ndi.binary_erosion((d['inp'].min(-1) > 0) & far & (d['real'].min(-1) > 0.002) & (d['out'].min(-1) > 0.002) & (d['real'].max(-1) < 0.97)
                                 & (z['halo_frac'].astype(float).max(-1) < 0.2), iterations=3)
    return d
def transfer(x, y, xs):
    edges = np.arange(-2.5, 2.5, 0.05); xc, yc = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (x >= a) & (x < b)
        if m.sum() >= 20: xc.append(np.median(x[m])); yc.append(np.median(np.log(y[m])))
    if len(xc) < 3: return np.full_like(xs, np.nan)
    xc, yc = np.array(xc), np.maximum.accumulate(np.array(yc))
    return np.where((xs >= xc[0]) & (xs <= xc[-1]), np.exp(np.interp(xs, xc, yc)), np.nan)
def run(f, label=None, show=True, mode=True):
    d = load(f)
    lh = d['lh']
    if mode:   # 本帧的"白" = 胶片曝光色度的众数 (灯箱帧: 墙面), 中性 / 彩度都相对它
        v = lh - lh.mean(-1, keepdims=True)
        bright = d['ok'] & (lh.mean(-1) > -1.0)
        a, b = v[bright][:, 0] - v[bright][:, 1], v[bright][:, 2] - v[bright][:, 1]
        hst, xe, ye = np.histogram2d(a, b, bins=60, range=[[-1.5, 1.5], [-1.5, 1.5]])
        i, j = np.unravel_index(np.argmax(hst), hst.shape)
        k = (np.abs(a - (xe[i] + xe[i+1]) / 2) < 0.05) & (np.abs(b - (ye[j] + ye[j+1]) / 2) < 0.05)
        v0 = np.median(v[bright][k], 0)
        lh = lh - v0
    neu = d['ok'] & (np.ptp(lh, -1) < TOL)
    chroma = np.ptp(lh, -1)
    hue = np.degrees(np.arctan2(np.sqrt(3) / 2 * (lh[..., 1] - lh[..., 2]), lh[..., 0] - 0.5 * (lh[..., 1] + lh[..., 2]))) % 360
    col = d['ok'] & (chroma > 0.25)
    pred = {w: np.stack([transfer(lh[neu][:, c] + (v0[c] if mode else 0), d[w][neu][:, c], lh[col][:, c] + (v0[c] if mode else 0)) for c in range(3)], -1) for w in ('real', 'out')}
    act = {w: d[w][col] for w in ('real', 'out')}
    v = np.isfinite(pred['real']).all(-1) & np.isfinite(pred['out']).all(-1)
    H, C = hue[col], chroma[col]
    out = []
    for name, h0 in HUES:
        for clo, chi in ((0.25, 0.5), (0.5, 0.8), (0.8, 3)):
            m = v & (((H - h0 + 180) % 360 - 180) ** 2 < 30 ** 2) & (C >= clo) & (C < chi)
            if m.sum() < 100: continue
            cr = lambda a, p: np.median(np.ptp(np.log10(a[m]), -1) / np.maximum(np.ptp(np.log10(p[m]), -1), 1e-3))
            rat = lambda a, p: np.median(a[m] / p[m], 0)
            gr = lambda a: np.median(a[m][:, 1] / a[m][:, 0])
            row = dict(hue=name, c=(clo, chi), n=int(m.sum()), cr_real=cr(act['real'], pred['real']), cr_out=cr(act['out'], pred['out']),
                       r_real=rat(act['real'], pred['real']), r_out=rat(act['out'], pred['out']), gr_real=gr(act['real']), gr_out=gr(act['out']),
                       gr_pr=gr(pred['real']), gr_po=gr(pred['out']))
            out.append(row)
            if show:
                print(f"{label or f} {name:3s} 彩度{clo:.2f}–{chi:.2f} n={m.sum():6d} | 彩度比 真实 {row['cr_real']:.2f} 模型 {row['cr_out']:.2f} | "
                      f"实际/预测 真实 {row['r_real'].round(2)} 模型 {row['r_out'].round(2)} | G/R 真实 {row['gr_real']:.2f} (预测 {row['gr_pr']:.2f}) 模型 {row['gr_out']:.2f} (预测 {row['gr_po']:.2f})")
    return out, int(neu.sum())
if __name__ == "__main__":
    for f in sys.argv[1:]:
        o, nn = run(f, f.split('/')[-1].replace('_t31_base', '').replace('.npz', ''))
        print(f"   (中性像素 {nn})")

