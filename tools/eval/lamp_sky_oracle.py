"""城市天光 = 灯光光谱 × 瑞利 λ^-4 (散射) 时, 相机色度、真值胶片层比与 D55 spectral 重建误差。"""
import sys, numpy as np, colour, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/user/filmlab")
from filmlab import lights, spectral, film
from filmlab.materials import emulsion
W = spectral.WAVELENGTHS
stock = emulsion("kodak_vision3_500t").stock; S = 10 ** stock.log_sensitivity
D55 = spectral.illuminant("D55")
for c in lights.classes():
    for n, s in c.members:
        for k, lab in [(0, "直射"), (4, "散射")]:
            rad = s * (W / 550.0) ** -k
            rgb = lights.camera_rgb(rad[None, :]); Ht = rad[None, :] @ S
            Hp = film.exposure_response(stock, D55, None, "spectral")(rgb)
            e = np.log10(Hp / Ht)[0]; e -= e[1]
            print(f"{c.key:12s} {n[:24]:24s} {lab} 相机 log R/G {np.log10(rgb[0,0]/rgb[0,1]):+.2f} B/G {np.log10(rgb[0,2]/rgb[0,1]):+.2f}  胶片 R/G {np.log10(Ht[0,0]/Ht[0,1]):+.2f} B/G {np.log10(Ht[0,2]/Ht[0,1]):+.2f}  日光重建误差 R {e[0]:+.2f} B {e[2]:+.2f}")
