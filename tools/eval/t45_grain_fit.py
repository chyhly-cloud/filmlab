"""T45: 5219 颗粒物理参数的拟合 (结果写在 filmlab/materials.py 的 GRAIN_5219)。
每层 10 个物理量: 3 个晶体亚群的中位 ECD 与涂布量、ECD 对数正态宽度 σ_ln、κ (每 µm³ 显影晶体的积分密度)、灰雾面积尺度 a_f、堆积结构因子 S0。
目标: (1) 仓库 kodak_vision3_500t 的原生特性曲线 (模型曝光轴, 净密度); (2) Kodak H-1-5219 的 48 µm 漫射 RMS 颗粒度,
按净密度对齐 (tools/eval/t45/k5219_gran.json)。光子数 Φ = η_ref × 3200K 各波段光子数, r = 4, T 颗粒厚 0.12 µm 固定。
用法: python3 tools/eval/t45_grain_fit.py"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from filmlab import grain
from filmlab.materials import emulsion, GRAIN_5219
import t45_grain_check as chk


def main():
    e = emulsion("kodak_vision3_500t")
    K, _ = chk.kodak("k5219_gran.json")
    x = e.log_exposure
    out = {}
    for c in range(3):
        net_k, s_k = K[c]
        tone = np.maximum.accumulate(e.S_nom[:, c] - e.S_nom[:, c].min())
        ok = (tone < 0.97 * tone.max())
        sel = net_k <= 0.95 * tone.max()

        def spec(q):
            ecd = [[2.0] * 3] * 3; amt = [[0.5] * 3] * 3
            ecd[c] = list(np.exp(q[0:3])); amt[c] = list(np.exp(q[3:6]))
            kap = [1.0] * 3; af = [100.0] * 3; s0 = [0.05] * 3
            kap[c] = np.exp(q[6]); af[c] = np.exp(q[7]); s0[c] = 1 / (1 + np.exp(-q[8]))
            sl = [0.3] * 3; sl[c] = 0.08 + 0.37 / (1 + np.exp(-q[9]))           # σ_ln ∈ (0.08, 0.45)
            return grain.GrainSpec(ecd_um=tuple(map(tuple, ecd)), amount=tuple(map(tuple, amt)), kappa=tuple(kap),
                                   fog_area_um2=tuple(af), packing_s0=tuple(s0), ei=500.0, illuminant="BB3200", sigma_ln=tuple(sl))

        def res(q):
            sp = spec(q)
            xs = grain.speed_points(e, sp)[c]
            H = 0.8 / 500.0 * 10.0 ** (x - xs)
            D, sg = grain.layer_curves(sp, c, H)
            i = np.argsort(D)
            m = np.interp(net_k[sel], D[i], sg[i])
            return np.concatenate([(D - tone)[ok] / 0.03, np.log(m / s_k[sel]) / 0.05 * np.sqrt(ok.sum() / sel.sum())])
        q0 = np.array([np.log(GRAIN_5219["ecd_um"][c][0]), np.log(GRAIN_5219["ecd_um"][c][1]), np.log(GRAIN_5219["ecd_um"][c][2]),
                       np.log(0.6), np.log(0.6), np.log(0.6), np.log(GRAIN_5219["kappa"][c]), np.log(GRAIN_5219["fog_area_um2"][c]), -2.9, 0.0])
        best = None
        for j in range(8):
            qq = q0 + (np.random.default_rng(j).normal(0, 0.4, 10) if j else 0)
            r = least_squares(res, qq, max_nfev=2000)
            if best is None or r.cost < best.cost:
                best = r
        q = best.x
        out["RGB"[c]] = dict(ecd=np.exp(q[0:3]).round(3).tolist(), amount=np.exp(q[3:6]).round(3).tolist(), kappa=round(float(np.exp(q[6])), 3),
                             a_f=round(float(np.exp(q[7])), 2), S0=round(float(1 / (1 + np.exp(-q[8]))), 4),
                             sigma_ln=round(float(0.08 + 0.37 / (1 + np.exp(-q[9]))), 3))
        print("RGB"[c], out["RGB"[c]], "cost", round(best.cost, 2))
    json.dump(out, open(HERE / "t45" / "fit_5219.json", "w"), indent=1)


if __name__ == "__main__":
    main()
