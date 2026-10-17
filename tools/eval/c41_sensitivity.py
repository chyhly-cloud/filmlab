"""C-41 显影模型的有界敏感性: 理想光谱输入 (Ohta 色卡 × 3200K, 中性灰阶), 只看负片 + 扫描密度 (无逐帧校正)。"""
import sys, numpy as np, colour, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/user/filmlab"); sys.path.insert(0, "/home/user/filmlab/tools/eval")
import mask_variant as variant; variant.install(sys.argv[1] if len(sys.argv) > 1 else "clip")
from filmlab import chemistry, lights, spectral
from filmlab.engine import Engine, Params
from filmlab.materials import emulsion, without_remjet
W = spectral.WAVELENGTHS; sh = colour.SpectralShape(W[0], W[-1], W[1] - W[0])
cc = colour.SDS_COLOURCHECKERS["ColorChecker N Ohta"]
R = np.array([cc[k].copy().align(sh).values for k in cc])
E = spectral.illuminant("BB3200")
S = 10 ** emulsion("kodak_vision3_500t").stock.log_sensitivity
ideal = np.log10((R * E) @ S)
steps = np.arange(-3, 3.01, 0.25)
def vec(L):     # (r−g, b−g) 相对灰块 (第 19–24 块) 的对数分离, 灰轴按每通道反查
    s = np.empty_like(L[:18]); NEU = np.log10([0.90, 0.59, 0.36, 0.19, 0.09, 0.031])
    for c in range(3):
        o = np.argsort(L[18:, c]); s[:, c] = np.interp(L[:18, c], L[18:, c][o], NEU[o])
    return np.stack([s[:, 0] - s[:, 1], s[:, 2] - s[:, 1]], 1)
Vi = vec(ideal)
def run(label, process, act=None, shift=True):
    old = chemistry.DEVELOPERS["ECN-2 CD"]
    if act is not None:
        chemistry.DEVELOPERS["ECN-2 CD"] = old.__class__(**{**old.__dict__, "activity": act})
    f = chemistry.dye_spectra
    if not shift: chemistry.dye_spectra = lambda d, *a, **k: d
    film = without_remjet(emulsion("kodak_vision3_500t"), "800T")
    eng = Engine(Params(film=film, process=process, grain=0, roll=None, scene_lights=None, timing="gray", scanner="noritsu"))
    rgb_cc = lights.camera_rgb(R * E[None, :]); rgb_cc = rgb_cc * 0.184 / (rgb_cc[21] @ [0.2126, 0.7152, 0.0722])
    rgb_g = (0.184 * 2.0 ** steps)[:, None] * np.ones(3) * lights.camera_rgb(E[None, :])[0] / (lights.camera_rgb(E[None, :])[0] @ [0.2126, 0.7152, 0.0722])
    out = {}
    for k, rgb in (("cc", rgb_cc), ("g", rgb_g)):
        H = eng._expose_lin(rgb, eng.M); st = eng._develop(eng._log_exposure(H))
        out[k] = eng.scanner.normalized_density(eng._transmittance(st))
    chemistry.DEVELOPERS["ECN-2 CD"] = old; chemistry.dye_spectra = f
    g = out["g"]; mid = (steps >= -1) & (steps <= 1)
    gam = [np.polyfit(steps[mid] * np.log10(2), g[mid, c], 1)[0] for c in range(3)]
    xr = (g[:, 0] - g[:, 1]); xb = (g[:, 2] - g[:, 1])
    V = vec(-out["cc"]) if False else vec(out["cc"])
    m = np.hypot(*V.T) / np.hypot(*Vi.T)
    a = np.degrees(np.angle(np.exp(1j * (np.arctan2(V[:, 1], V[:, 0]) - np.arctan2(Vi[:, 1], Vi[:, 0])))))
    print(f"{label:28s} 扫描密度 gamma R {gam[0]:.3f} G {gam[1]:.3f} B {gam[2]:.3f} | 交叉 (−3…+3 档) R−G {np.ptp(xr):.3f} B−G {np.ptp(xb):.3f}"
          f" | 色卡分离比 {np.median(m):.2f} 方向 {np.median(np.abs(a)):3.0f}°")
run("ECN-2 原生", "ECN-2")
for t in (1.0, 1.1, 1 / 0.85, 1.3):
    run(f"C-41 τ={t:.3f}", "C-41", act=1 / t)
run("C-41 τ=1.176, 无染料平移", "C-41", act=0.85, shift=False)
