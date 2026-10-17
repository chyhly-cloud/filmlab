"""暮光天空 (日光 × 瑞利 × 臭氧 Chappuis 吸收) 的真值层曝光 vs 重建。"""
import sys, numpy as np, colour, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/user/filmlab")
from filmlab import lights, spectral, film
from filmlab.materials import emulsion
W = spectral.WAVELENGTHS
o3 = np.loadtxt(sys.argv[1])
# 1 nm 平均后插值到 W
sig = np.interp(W, o3[:, 0], np.convolve(o3[:, 1], np.ones(100) / 100, "same"))
print("Chappuis 峰", W[np.argmax(sig * (W > 450))], "nm", sig.max())
N = 300 * 2.687e16      # 300 DU 垂直柱
stock = emulsion("kodak_vision3_500t").stock; S = 10 ** stock.log_sensitivity
D55 = spectral.illuminant("D55")
sun = spectral.illuminant("D55")
for m in [0, 5, 10, 20, 30]:
    rad = sun * (W / 550.0) ** -4 * np.exp(-sig * N * m)
    rad /= rad.max()
    rgb = lights.camera_rgb(rad[None, :])
    Ht = rad[None, :] @ S
    xy = colour.XYZ_to_xy(colour.sd_to_XYZ(colour.SpectralDistribution(dict(zip(W, rad)))))
    row = [f"臭氧气团 {m:2d}: τ600={sig[np.searchsorted(W,600)]*N*m:4.2f}  相机 log(R/G) {np.log10(rgb[0,0]/rgb[0,1]):+.2f} log(B/G) {np.log10(rgb[0,2]/rgb[0,1]):+.2f} | 真值胶片 log(R/G) {np.log10(Ht[0,0]/Ht[0,1]):+.2f} log(B/G) {np.log10(Ht[0,2]/Ht[0,1]):+.2f}"]
    for mode in ["spectral", "linear"]:
        Hp = film.exposure_response(stock, D55, None, mode)(rgb)
        e = np.log10(Hp / Ht)[0]; e = e - e[1]
        row.append(f"{mode} 误差 R {e[0]:+.2f} B {e[2]:+.2f}")
    print("  ".join(row))
