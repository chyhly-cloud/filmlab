"""光谱真值检验: Ohta 色卡实测反射光谱 × 光源 × 500T 感光度 = 各层真值曝光, 与 FilmLab 从相机 RGB 重建的曝光逐层比较 (档)。"""
import sys, numpy as np, colour, warnings; warnings.filterwarnings("ignore")
import os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from filmlab import lights, spectral, film
from filmlab.materials import emulsion
W = spectral.WAVELENGTHS
sds = colour.SDS_COLOURCHECKERS["ColorChecker N Ohta"]; names = list(sds)
R = np.stack([sds[k].copy().align(colour.SpectralShape(W[0], W[-1], W[1] - W[0])).values for k in names])
stock = emulsion("kodak_vision3_500t").stock; S = 10 ** stock.log_sensitivity
for light in ["D55", "BB3200"]:
    E = spectral.illuminant(light); rgb = lights.camera_rgb(R * E); Ht = (R * E) @ S
    for mode in ["spectral", "linear"]:
        Hp = film.exposure_response(stock, E, None, mode)(rgb); Hp = np.maximum(Hp * Ht[21, 1] / Hp[21, 1], 1e-12)
        st = np.log2(Hp / Ht); cast = st - st[:, 1:2]
        worst = np.argsort(-np.abs(st).max(1))[:4]
        print(f"{light:6s} {mode:8s} 各层 |误差| 中位 (档) {np.round(np.median(np.abs(st), 0), 3)}  最大 {np.round(np.abs(st).max(0), 2)}  "
              f"层间比 |R/G|,|B/G| 中位 {np.round(np.median(np.abs(cast[:, [0, 2]]), 0), 3)}")
        print("        最差: " + "; ".join(f"{names[i]} {np.round(st[i], 2)}" for i in worst))
