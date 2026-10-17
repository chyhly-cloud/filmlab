"""P0: 5219 档案同源自洽 —— 带符号净谱 vs 截断, 能否重现档案自己的中性中间调光谱。"""
import sys, json, numpy as np
sys.path.insert(0, "/home/user/filmlab")
from filmlab.stocks import _clean, WAVELENGTHS as W
d = json.load(open("/home/user/filmlab/filmlab/data/profiles/kodak_vision3_500t.json"))["data"]
cd = _clean(d["channel_density"]); base = _clean(d["base_density"]); mid = np.array(d["midscale_neutral_density"], float)
midc = _clean(d["midscale_neutral_density"])
x = np.array(d["log_exposure"]); dc = np.nan_to_num(np.array(d["density_curves"], float))
valid = ~np.isnan(mid)
print("λ 有效范围 (mid)", W[valid].min(), W[valid].max(), " 中性光谱是否已含片基: mid@700", mid[64], " base@700", base[64])
def best(spec, mask, extra=""):
    out = []
    for i in range(len(x)):
        Dt = base + dc[i] @ spec.T
        r = (Dt - midc)[mask]
        out.append(np.sqrt((r ** 2).mean()))
    i = int(np.argmin(out)); Dt = base + dc[i] @ spec.T; r = Dt - midc
    j = np.argmax(np.abs(r * mask))
    return out[i], x[i], abs(r[j]), W[j], dc[i], r
for name, spec in [("带符号", cd), ("截断", np.clip(cd, 0, None))]:
    for mname, m in [("仅原始有效波段", valid), ("全部 81 (端点外推)", np.ones(81, bool))]:
        e, xi, mx, wl, Di, r = best(spec, m)
        print(f"{name:4s} {mname:14s} RMSE {e:.4f} D  最佳 x {xi:+.3f}  最大残差 {mx:.4f} D @ {wl:.0f} nm  层密度 {np.round(Di, 3)}")
# 自由三层浓度 (不受特性曲线约束) 的最小二乘: 区分"谱形状"与"曲线位置"
for name, spec in [("带符号", cd), ("截断", np.clip(cd, 0, None))]:
    c, *_ = np.linalg.lstsq(spec[valid], (midc - base)[valid], rcond=None)
    r = spec[valid] @ c - (midc - base)[valid]
    print(f"{name} 自由浓度拟合 RMSE {np.sqrt((r**2).mean()):.4f}  浓度 {np.round(c, 3)}")
# 物理边界: 三层浓度在 0 … Dmax 的所有组合, 总密度最小值
g = np.linspace(0, 1, 11)
dmax = dc.max(0)
mins = []
for a in g:
    for b in g:
        for cc in g:
            Dt = base + (np.array([a, b, cc]) * dmax) @ cd.T
            mins.append(Dt.min())
print("带符号谱: 11^3 浓度组合中总密度最小值", round(min(mins), 4), "(T ≤ 1 ⇔ 总密度 ≥ 0)")
# 仅成像净谱 (去片基) 的最小值 —— 即"相对片基的减少量"最大能到多少
print("净谱相对片基的最大减少 (Dmax 组合)", round(((dmax) @ cd.T).min(), 4), " 片基在该处", )
