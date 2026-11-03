"""T43: 输运一致的片基光晕表 (filmlab/data/halo_transport_t43.json), docs/tasks/T43_HALATION_CHAIN.md。

原光晕链 (film.halation_response × halation_kernel) 把乳剂散射当作消光 (T_down、back、rho = T_down² 都按"散射即损失"),
再独立地乘 diffuse_fraction = 0.6 与 cos^m 角分布。本表用同一组已有物理量, 由逐波长单色 Monte Carlo
(tools/eval/photon_transport.py, MCML 式, 菲涅耳 / 全反射边界) 重新算光晕, 散射只改方向、不损失能量:
  · 叠层 (自上而下): 蓝感层 6 µm / 黄滤光层 1 µm / 绿感层 6 µm / 红感层 6 µm / 片基 130 µm, n 均 1.5, 顶面与片基背面为空气;
  · 每层吸收 = film.emulsion_stack 同一代理 (0.15·S/max S; 黄滤光层; 遮罩按层放置 "layers"), 感光组分与遮罩分开记账;
  · 散射消光密度 = film.TURBIDITY·(550/λ)^3 (每个感光层), Henyey–Greenstein g = 0.85 ——
    在该散射量下与 Kodak 5219 官方 MTF 10–30 c/mm 最一致的 g (R: 0.944/0.905/0.873 对 官方/1.03 ≈1.0/0.95/0.84);
  · 每层记录 direct(λ) = 未经片基背面反射的吸收, halo(λ) = 片基背面反射之后的吸收 (含片基–乳剂间的全部多次往返) 与其径向分布。
引擎 (修正 "halo_transport") 用 ratio_c(λ) = halo/direct 乘该层的光谱感光度作为光晕源, 用按 3200K × 感光度 × ratio 加权的径向分布作核。
用法: python3 t43_halo_transport.py 输出.json"""
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1])]
import photon_transport as pt

G = 0.85
TURB_SCALE = 1.0
TH = dict(B=6.0, Y=1.0, G=6.0, R=6.0)
BASE_UM = 130.0
EDGES = np.r_[np.arange(0, 3000, 10.0), np.arange(3000, 6001, 50.0)]
LAMS = list(range(400, 761, 10))
LN = np.log(10.0)


def stack():
    from filmlab import generation, film
    from filmlab.engine import Engine, Params
    from filmlab.recipes import recipe
    from filmlab.stocks import WAVELENGTHS as WL
    generation.use("t31_base")
    eng = Engine(Params(**{**recipe("cinestill_800t"), "roll": None, "frame": 0, "grain": 0}))
    st, v = eng.e.stock, eng.v
    S = 10.0 ** st.log_sensitivity
    A = np.asarray(v.layer_absorbance)[None, :] * S / S.max(0)
    Ay = v.yellow_filter / (1.0 + np.exp((WL - 505.0) / 9.0))
    mask = np.clip(st.base_density - st.base_density.min(), 0, None)
    wm = 1.0 / (1.0 + np.exp((WL - film.MASK_SPLIT_NM) / 10.0))
    return dict(st=st, v=v, S=S, A=A, Ay=Ay, Mm=mask * wm, Mc=mask * (1 - wm), WL=WL, film=film)


def layers(d, lam, s=TURB_SCALE, g=G):
    i = int(np.argmin(abs(d["WL"] - lam)))
    A, Ay, Mm, Mc = d["A"], d["Ay"], d["Mm"], d["Mc"]
    turb = s * d["film"].TURBIDITY * (550.0 / lam) ** 3
    L = [pt.Layer("B", TH["B"], A[i, 2] * LN / TH["B"], turb * LN / TH["B"], g),
         pt.Layer("Y", TH["Y"], Ay[i] * LN / TH["Y"], 0.0, 0.0),
         pt.Layer("G", TH["G"], (A[i, 1] + Mm[i]) * LN / TH["G"], turb * LN / TH["G"], g),
         pt.Layer("R", TH["R"], (A[i, 0] + Mc[i]) * LN / TH["R"], turb * LN / TH["R"], g),
         pt.Layer("base", BASE_UM, 1e-9, 0.0, 0.0)]
    act = {"R": A[i, 0] / max(A[i, 0] + Mc[i], 1e-12), "G": A[i, 1] / max(A[i, 1] + Mm[i], 1e-12), "B": 1.0}
    return L, act, i


def main(dst, photons=200_000):
    from filmlab import spectral
    d = stack()
    E = spectral.illuminant(d["st"].reference_illuminant)
    out = {"source": "tools/eval/t43_halo_transport.py", "lam": LAMS, "g": G, "turbidity_scale": TURB_SCALE,
           "thickness_um": TH, "base_um": BASE_UM, "edges_um": EDGES.tolist(),
           "ratio": {c: [] for c in "RGB"}, "direct": {c: [] for c in "RGB"}, "halo": {c: [] for c in "RGB"}}
    prof = {c: np.zeros(len(EDGES)) for c in "RGB"}            # 末格 = 超出 6000 µm
    for lam in LAMS:
        L, act, i = layers(d, lam)
        o = pt.simulate(L, photons, lam, n_top=1.0, n_bottom=1.0, radial_edges=EDGES, max_events=40000)
        assert o["ledger"]["event_residual"] < 1e-4, o["ledger"]
        for c, li, k in (("R", 3, 0), ("G", 2, 1), ("B", 0, 2)):
            dd = (o["absorbed"][li] - o["returned_absorbed"][li]) * act[c]
            h = o["returned_absorbed"][li] * act[c]
            r = h / dd if dd > 1e-6 else 0.0
            out["ratio"][c].append(r); out["direct"][c].append(dd); out["halo"][c].append(h)
            hh = np.r_[o["radial_returned"][li], o["returned_overflow"][li]] * act[c]
            if h > 0:
                prof[c] += E[i] * d["S"][i, k] * r * hh / h
        print(lam, " ".join(f"{c} {out['ratio'][c][-1]:.4f}" for c in "RGB"), flush=True)
    out["profile"] = {c: ((p / p.sum()) if p.sum() > 0 else p).tolist() for c, p in prof.items()}
    json.dump(out, open(dst, "w"))


if __name__ == "__main__":
    main(sys.argv[1])
