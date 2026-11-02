import numpy as np, json
g = json.load(open(__import__("pathlib").Path(__file__).resolve().parent / "epfl_wedge_geometry.json"))
z = np.load("epfl/R1-17-18_t31_base.npz")
C = np.array(g["source_corners"]) * z["inp"].shape[1] / g["source_size"][0]
G0 = np.array([-0.83960607, -0.84922118, -0.8927258])
rows = []
print("条 | RAW RGB | logH−中灰 | Dn | 真实 RGB | 模型 RGB | 真实 G/R B/G | 模型 G/R B/G")
for i in range(14):
    u = (i + 0.5) / 14
    pts = []
    for v in (0.35, 0.5, 0.65):
        p = (1-u)*(1-v)*C[0] + u*(1-v)*C[1] + u*v*C[2] + (1-u)*v*C[3]
        pts.append(p)
    vals = {}
    for k in ("inp", "logH", "Dn", "real", "out"):
        a = z[k].astype(float)
        vals[k] = np.nanmedian(np.concatenate([a[int(p[1])-6:int(p[1])+7, int(p[0])-6:int(p[0])+7].reshape(-1, 3) for p in pts]), 0)
    rows.append(vals)
    r, o = vals["real"], vals["out"]
    print(f"{i+1:2d} | {vals['inp'].round(3)} | {(vals['logH']-G0).round(2)} | {vals['Dn'].round(3)} | {r.round(3)} | {o.round(3)} | {r[1]/r[0]:.2f} {r[2]/r[1]:.2f} | {o[1]/o[0]:.2f} {o[2]/o[1]:.2f}")
R = np.array([r["real"] for r in rows]); O = np.array([r["out"] for r in rows]); L = np.array([r["logH"] for r in rows])
print("真实每通道单调递增:", [bool(np.all(np.diff(R[:, c]) > -0.005)) for c in range(3)], " 模型:", [bool(np.all(np.diff(O[:, c]) > -0.005)) for c in range(3)])
# 通道反差 (log 输出 / logH, 中间 5–11 条)
s = slice(4, 11)
for nm, A in (("真实", R), ("模型", O)):
    print(nm, "log10 输出 / logH 斜率 R G B:", [round(float(np.polyfit(L[s, c], np.log10(A[s, c]), 1)[0]), 2) for c in range(3)])
