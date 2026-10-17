"""天空光谱 (高色温 CIE 日光 / 瑞利散射) 的真值曝光 vs FilmLab 重建: 层间比误差 (档)。"""
import sys, numpy as np, colour, warnings; warnings.filterwarnings("ignore")
import os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from filmlab import lights, spectral, film
from filmlab.materials import emulsion
W = spectral.WAVELENGTHS
stock = emulsion("kodak_vision3_500t").stock; S = 10 ** stock.log_sensitivity
D55 = spectral.illuminant("D55")
def daylight(T):
    xy = colour.temperature.CCT_to_xy_CIE_D(T); sd = colour.sd_CIE_illuminant_D_series(xy)
    return sd.copy().align(colour.SpectralShape(W[0], W[-1], W[1] - W[0])).values
skies = {"D100 (蓝天)": daylight(10000), "D150": daylight(15000), "D250 (深蓝天)": daylight(25000),
         "瑞利 D55×λ⁻⁴": D55 * (W / 550.0) ** -4, "D65": daylight(6504)}
# 天空是"光源"本身, 用日光类的 spectral 方法 (FilmLab 对日光区域的做法) 计算; 也比较 linear
for name, spec in skies.items():
    rad = spec / spec.max()
    rgb = lights.camera_rgb(rad[None, :])
    Ht = rad[None, :] @ S
    out = []
    for mode in ["spectral", "linear"]:
        Hp = film.exposure_response(stock, D55, None, mode)(rgb)
        r = np.log2(Hp / Ht) - np.log2(Hp / Ht)[:, 1:2]
        out.append(f"{mode}: R/G {r[0,0]:+.2f} 档  B/G {r[0,2]:+.2f} 档")
    print(f"{name:14s} 相机 RGB 比例 {np.round(rgb[0] / rgb[0][1], 2)}   " + "   ".join(out))
