"""T31: 用独立的光子输运验证器 (tools/eval/photon_transport.py) 检查 "散射是消光还是漫射光的来源" 对光晕 G/R 的影响。
层 (自上而下): 蓝感层 6 µm | 黄滤光层 1 µm | 绿感层 6 µm (含 Mm) | 红感层 6 µm (含 Mc) | 片基 130 µm | 空气 (CineStill)。
各层吸收 = 模型同一密度 (感光吸收 A, 遮罩, 黄滤光), 散射光学厚度 = TURB·(550/λ)^3·ln10 每感光层, HG g。
光晕源 = 片基背面反射后、半径 > 100 µm 处被感光物质吸收的能量 (按 μ_AgX/μ_a 分给感光物质); 直接 = 未反射时感光吸收。
单位与 halation_response 相同: 光晕_c × (E·S_c)/(E·直接_c)。用法: python3 t31_mc.py g turb [photons]"""
import sys, numpy as np
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[2])]
import photon_transport as pt
from t31_halo_source import st, v, S, W, split_mask, LIGHTS, halo
g = float(sys.argv[1]); turb0 = float(sys.argv[2]); N = int(sys.argv[3]) if len(sys.argv) > 3 else 40000
place = sys.argv[4] if len(sys.argv) > 4 else "phys"
LN = np.log(10.0)
A = np.asarray(v.layer_absorbance)[None, :] * S / S.max(0)
Ay = v.yellow_filter / (1.0 + np.exp((W - 505.0) / 9.0))
Mc, Mm = split_mask()
TH = 6.0
lam = np.arange(400, 701, 10.0)
ret = np.zeros((len(lam), 3)); dirx = np.zeros((len(lam), 3))
for i, L in enumerate(lam):
    j = int(np.argmin(abs(W - L)))
    ts = turb0 * (550.0 / L) ** 3 * LN / TH
    if place == "phys":
        mG, mR = Mm[j], Mc[j]
    else:   # 遮罩全部放在红感层 (绿光往返各一次)
        mG, mR = 0.0, Mm[j] + Mc[j]
    aB, aG, aR = A[j, 2] * LN / TH, A[j, 1] * LN / TH, A[j, 0] * LN / TH
    mg, mr = mG * LN / TH, mR * LN / TH
    layers = [pt.Layer("B", TH, aB, ts, g), pt.Layer("Y", 1.0, Ay[j] * LN, 0.0, 0.0),
              pt.Layer("G", TH, aG + mg, ts, g), pt.Layer("R", TH, aR + mr, ts, g), pt.Layer("base", 130.0, 1e-6, 0.0, 0.0)]
    r = pt.simulate(layers, photons=N, seed=i, radial_edges=np.array([0, 100.0, 1e5]), n_top=1.0, n_bottom=1.0)
    sens_frac = np.array([aR / max(aR + mr, 1e-12), aG / max(aG + mg, 1e-12), 1.0])
    li = [3, 2, 0]                     # R G B 层在 layers 中的序号
    for c in range(3):
        ret[i, c] = r["radial_returned"][li[c], 1] * sens_frac[c]
        dirx[i, c] = (r["absorbed"][li[c]] - r["returned_absorbed"][li[c]]) * sens_frac[c]
np.savez(f"t31_mc_g{g}_t{turb0}_{place}.npz", lam=lam, ret=ret, dirx=dirx)
for nm, E81 in LIGHTS.items():
    E = np.interp(lam, W, E81)
    Sl = np.stack([np.interp(lam, W, S[:, c]) for c in range(3)], 1)
    unit = (E @ Sl) / np.maximum(E @ dirx, 1e-30)
    h = (E @ ret) * unit
    hm, _ = halo(E81, place="phys")
    print(f"g={g} turb={turb0} {place}  {nm:14s} 光晕源 G/R = {h[1]/h[0]:.3f}  B/G = {h[2]/max(h[1],1e-12):.3f}   R 强度 {h[0]:.4f}   (公式·物理放置 {hm[1]/hm[0]:.3f}, R {hm[0]:.4f})")
