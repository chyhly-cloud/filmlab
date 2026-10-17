"""9 组对齐配对 (show_pairs 风格的 site/img) 按色相分区的彩度比与色相偏差; 格子按两边平均色相 / 彩度选取, 避免选择偏差。
用法: python3 tools/eval/pair_hue_stats.py 版本1,版本2"""
import numpy as np, colour, warnings, sys; warnings.filterwarnings("ignore")
from PIL import Image
D = "/tmp/claude-0/-home-user-filmlab/f67b7686-eba5-58fc-b23b-cf537590be15/scratchpad/versions/site/img/"
SC = ["cafe", "corridor", "building", "woodwall", "neon", "dusk", "skyline", "glass", "campus"]
VERS = sys.argv[1].split(",")
def cells(a, c=12):
    h, w = a.shape[:2]; a = a[: h // c * c, : w // c * c]
    return np.median(a.reshape(h // c, c, w // c, c, 3).transpose(0, 2, 1, 3, 4).reshape(-1, c * c, 3), 1)
lab = lambda x: colour.XYZ_to_Lab(colour.sRGB_to_XYZ(np.clip(x, 0, 1)))
sectors = [("红/品红", -30, 30), ("橙/肤", 30, 75), ("黄", 75, 110), ("绿/黄绿", 110, 180), ("青", 180, 240), ("蓝", 240, 300), ("紫/品", 300, 330)]
res = {v: [] for v in VERS}; R_all = []
for s in SC:
    R = lab(cells(np.asarray(Image.open(D + f"{s}_real.jpg"), float) / 255))
    valid = R[:, 0] > 8
    for v in VERS:
        S = lab(cells(np.asarray(Image.open(D + f"{s}_{v}.jpg"), float) / 255))
        ok = valid & (S[:, 0] > 3)
        res[v].append((S[ok], R[ok]))
print("色相区间 (按真实)   格子数  " + "  ".join(f"{v:>22s}" for v in VERS))
for name, h0, h1 in sectors:
    row = []
    for v in VERS:
        S = np.concatenate([a for a, b in res[v]]); Rr = np.concatenate([b for a, b in res[v]])
        hr = np.degrees(np.arctan2(Rr[:, 2], Rr[:, 1])) % 360; cr = np.hypot(Rr[:, 1], Rr[:, 2]); cs = np.hypot(S[:, 1], S[:, 2])
        lo, hi = h0 % 360, h1 % 360
        hm = np.degrees(np.arctan2(Rr[:, 2] + S[:, 2], Rr[:, 1] + S[:, 1])) % 360; m = ((hm >= lo) & (hm < hi) if lo < hi else ((hm >= lo) | (hm < hi))) & ((cr + cs) / 2 > 12)
        hs = np.degrees(np.arctan2(S[m, 2], S[m, 1])) % 360
        dh = (hs - hr[m] + 180) % 360 - 180
        row.append(f"彩度比 {np.median(cs[m]) / np.median(cr[m]):4.2f} 色相 {np.median(dh):+5.0f}°")
        n = m.sum()
    print(f"{name:10s} {n:7d}  " + "  ".join(f"{r:>22s}" for r in row))
for v in VERS:
    S = np.concatenate([a for a, b in res[v]]); Rr = np.concatenate([b for a, b in res[v]])
    print(f"{v:8s} 全部格子 ΔE76 中位 {np.median(np.linalg.norm(S - Rr, axis=1)):.2f}   ΔE00 中位 {np.median(colour.delta_E(S, Rr, method='CIE 2000')):.2f}")
