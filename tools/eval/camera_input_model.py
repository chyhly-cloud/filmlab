"""相机 RGB (RAW 解码, 日光白平衡) 与理想观察者 RGB 的关系: 矩阵误差还是加性底 (黑电平 / 杂散光)?"""
import sys, numpy as np, colour, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/user/filmlab"); sys.path.insert(0, "/home/user/filmlab/tools/eval")
import chart_500t as C
from scipy.optimize import least_squares
from filmlab import lights, spectral
from filmlab.rawio import load_raw
DATA = "/tmp/claude-0/-home-user-filmlab/f67b7686-eba5-58fc-b23b-cf537590be15/scratchpad/charts"
W = spectral.WAVELENGTHS; sh = colour.SpectralShape(W[0], W[-1], W[1] - W[0])
cc = colour.SDS_COLOURCHECKERS["ColorChecker N Ohta"]; R = np.array([cc[k].copy().align(sh).values for k in cc])
for scene, light in [("tungsten", "BB3200"), ("led5600", "D55")]:
    S = C.SCENES[scene]
    for dn in (True, False):
        raw = load_raw(f"{DATA}/ref/dlf_x100v/{S['raf']}.RAF", max_size=3000, denoise_raw=dn); raw = raw[0] if isinstance(raw, tuple) else raw
        P = C.patches(raw, S["grid_raw"]); I = lights.camera_rgb(R * spectral.illuminant(light)[None, :])
        I *= (P[21] @ [.2126, .7152, .0722]) / (I[21] @ [.2126, .7152, .0722])
        white = P[18] @ [.2126, .7152, .0722]
        def fit(kind):
            def f(p):
                if kind == "mat": pred = I @ p[:9].reshape(3, 3).T
                elif kind == "off": pred = I * p[:3] + p[3:6]
                else: pred = I @ p[:9].reshape(3, 3).T + p[9:12]
                return (np.log2(np.clip(pred, 1e-6, None)) - np.log2(np.clip(P, 1e-6, None)))[:23].ravel()
            x0 = {"mat": np.eye(3).ravel(), "off": np.r_[1, 1, 1, 0, 0, 0], "both": np.r_[np.eye(3).ravel(), 0, 0, 0]}[kind]
            r = least_squares(f, x0); return np.sqrt((r.fun ** 2).mean()), r.x
        e0 = np.sqrt(((np.log2(np.clip(I, 1e-6, None)) - np.log2(np.clip(P, 1e-6, None)))[:23] ** 2).mean())
        em, xm = fit("mat"); eo, xo = fit("off"); eb, xb = fit("both")
        print(f"{scene:8s} 去噪={dn!s:5s} 无模型 RMS {e0:.2f} 档 | 3×3 矩阵 {em:.2f} | 每通道增益+底 {eo:.2f} (底 = 白块的 {np.round(xo[3:6] / white * 100, 2)} %) | 矩阵+底 {eb:.2f} (底 {np.round(xb[9:12] / white * 100, 2)} %)")
