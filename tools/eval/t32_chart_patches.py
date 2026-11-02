import numpy as np, sys, colour, warnings
warnings.filterwarnings('ignore')
from t32_chart import patches, NAMES
G0 = np.array([-0.83960607, -0.84922118, -0.8927258])
f = sys.argv[1]
z = np.load(f); P = patches(z)
gray = [P[(4, c)] for c in range(8)]   # 灰9 = 不透光, 不用
def F(who, ch, x):
    xs = np.array([g['logH'][ch] for g in gray]); ys = np.log(np.maximum([g[who][ch] for g in gray], 1e-5))
    o = np.argsort(xs); return np.exp(np.interp(x, xs[o], ys[o], left=np.nan, right=np.nan))
def lab(rgb): return colour.XYZ_to_Lab(colour.sRGB_to_XYZ(np.clip(rgb, 0, 1), apply_cctf_decoding=False))
print(f"{f}\n灰阶: logH−中灰 / 真实 RGB / 模型 RGB")
for g, n in zip(gray, NAMES[4]):
    print(f"  {n}: {(g['logH']-G0).round(2)}  {g['real'].round(3)}  {g['out'].round(3)}")
print("\n色块 | RAW G/R B/R | 胶片曝光 G/R B/R (中灰=1) | 真实 G/R B/G | 模型 G/R B/G | 真实/单通道预测 R G B | 模型/单通道预测 R G B | 真实 C*ab h | 模型 C*ab h")
rows = []
for r in range(4):
    for c in range(9):
        d = P[(r, c)]
        e = 10 ** (d['logH'] - G0)
        pr = np.array([F('real', k, d['logH'][k]) for k in range(3)]); pm = np.array([F('out', k, d['logH'][k]) for k in range(3)])
        lr, lm = lab(d['real']), lab(d['out'])
        rows.append((NAMES[r][c], d, e, pr, pm, lr, lm))
        print(f"{NAMES[r][c]:5s} | {d['inp'][1]/d['inp'][0]:.2f} {d['inp'][2]/d['inp'][0]:.2f} | {e[1]/e[0]:.2f} {e[2]/e[0]:.2f} | "
              f"{d['real'][1]/d['real'][0]:.2f} {d['real'][2]/max(d['real'][1],1e-4):.2f} | {d['out'][1]/d['out'][0]:.2f} {d['out'][2]/max(d['out'][1],1e-4):.2f} | "
              f"{(d['real']/pr).round(2)} | {(d['out']/pm).round(2)} | {np.hypot(*lr[1:]):.0f} {np.degrees(np.arctan2(lr[2], lr[1]))%360:.0f} | {np.hypot(*lm[1:]):.0f} {np.degrees(np.arctan2(lm[2], lm[1]))%360:.0f}")
np.save(f.replace('.npz', '_patches.npy'), np.array(rows, dtype=object), allow_pickle=True)
