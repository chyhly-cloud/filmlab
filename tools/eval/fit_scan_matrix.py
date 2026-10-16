"""拟合扫描仪的饱和度校正矩阵: 真实 = 每帧通道增益 × (模拟 @ Aᵀ), A 行和 = 1, 在 Lab 上最小化 (soft-L1)。
输入为 show_pairs.py 生成、且模拟时 SATURATION_MATRIX = 单位矩阵的三联图。含留一验证。
用法: python3 tools/eval/fit_scan_matrix.py 三联图目录"""
import sys, numpy as np, colour, warnings; warnings.filterwarnings("ignore")
from PIL import Image
from scipy.optimize import least_squares
FILES = {i: f"{sys.argv[1]}/pair_{i}.jpg" for i in (1, 2, 3, 4, 5)}   # show_pairs.py 的输出
def cells(f, cell=16):
    a = np.asarray(Image.open(f)).astype(float) / 255; a = a[46:]
    w = (a.shape[1] - 20) // 3
    sim, real = a[:, w + 10:2 * w + 10], a[:, 2 * w + 20:3 * w + 20]
    S, R = [], []
    for y in range(0, a.shape[0] - cell, cell):
        for x in range(0, w - cell, cell):
            s, r = sim[y:y+cell, x:x+cell].reshape(-1, 3), real[y:y+cell, x:x+cell].reshape(-1, 3)
            if s.min() <= 0.0 or s.max() > 0.97 or r.max() > 0.97 or r.min() < 0.02: continue
            S.append(np.median(s, 0)); R.append(np.median(r, 0))
    lin = lambda v: colour.cctf_decoding(np.array(v))
    return lin(S), lin(R)
data = {i: cells(f) for i, f in FILES.items()}
def lab(x): return colour.XYZ_to_Lab(colour.sRGB_to_XYZ(np.clip(x, 1e-6, None), apply_cctf_decoding=False))
def model(p, ids):
    A = np.eye(3) + p[:9].reshape(3, 3) - np.diag(p[:9].reshape(3, 3).sum(1))   # 行和 = 1: 中性不变
    out = {}
    for k, i in enumerate(ids):
        g = np.exp(p[9 + 3 * k: 12 + 3 * k])
        out[i] = (data[i][0] @ A.T) * g
    return out, A
def resid(p, ids):
    out, _ = model(p, ids)
    return np.concatenate([(lab(out[i]) - lab(data[i][1])).ravel() for i in ids])
def fit(ids, fix_A=None):
    n = 9 + 3 * len(ids)
    if fix_A is None:
        r = least_squares(resid, np.zeros(n), args=(ids,), loss="soft_l1", f_scale=5)
        return r.x
    # 只拟合每帧增益, A 固定
    def rr(q): return resid(np.concatenate([fix_A, q]), ids)
    q = least_squares(rr, np.zeros(3 * len(ids)), loss="soft_l1", f_scale=5).x
    return np.concatenate([fix_A, q])
def de(p, ids):
    out, _ = model(p, ids)
    return {i: float(np.median(np.linalg.norm(lab(out[i]) - lab(data[i][1]), axis=1))) for i in ids}
ids = list(FILES)
p0 = fit(ids, fix_A=np.zeros(9)); print("仅每帧增益 (无矩阵):", {k: round(v, 1) for k, v in de(p0, ids).items()})
pa = fit(ids); _, A = model(pa, ids); print("全部拟合 A:\n", A.round(3)); print("  ", {k: round(v, 1) for k, v in de(pa, ids).items()})
print("留一验证 (A 用其余 4 对拟合, 被留出的一对只拟合增益):")
for i in ids:
    rest = [j for j in ids if j != i]
    pr = fit(rest); _, Ai = model(pr, rest)
    pi = fit([i], fix_A=pr[:9]); pz = fit([i], fix_A=np.zeros(9))
    print(f"  pair {i}: 无矩阵 {de(pz, [i])[i]:.1f} → 有矩阵 {de(pi, [i])[i]:.1f}   A 对角 {np.diag(Ai).round(2)}")
print("SATURATION_MATRIX =", repr(A.round(3)))
