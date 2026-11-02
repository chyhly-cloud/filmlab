"""T32 第 3 步: 扫描级。染料光谱 (模型 = spektrafilm 截断谱; Kodak 原始带符号净谱) 在扫描通道上的串扰。"""
import os
import numpy as np, sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
from filmlab import generation; generation.use("t31_base")
from filmlab.engine import Engine, Params
from filmlab.recipes import recipe
from filmlab.stocks import WAVELENGTHS as W
from filmlab.scanning.optics import led_response, measure
from filmlab.scanner import SCANNERS
from filmlab.materials import emulsion
RAW = os.environ.get("KODAK5219_RAW", "raw5219")
def curve(n, zero=False):
    a = np.loadtxt(f"{RAW}/{n}.csv", delimiter=","); a = a[np.argsort(a[:, 0])]
    return np.interp(W, a[:, 0], a[:, 1], left=0 if zero else None, right=0 if zero else None)
F = np.stack([curve(f"statusM_{c}", True) for c in "rgb"], 1); F /= F.sum(0)
eng = Engine(Params(**{**recipe("cinestill_800t"), "roll": None, "grain": 0, "timing": "frame"}))
st = eng.e.stock
E_model = eng._dye_spectra("CD-4")          # C-41 显影剂下的染料谱 (含平移)
E_st = st.dye_density
k = np.load(__import__("filmlab.stocks").stocks.PROFILE_DIR / "kodak_vision3_500t_kodak_raw.npz")
E_k = k["E"]
base = st.base_density
m = SCANNERS["frontier"]
L = led_response(m)
print("扫描通道 峰值/σ:", m.led_peaks, m.led_widths)
for name, E in (("模型 (C-41 平移后)", E_model), ("模型档案 (ECN-2)", E_st), ("Kodak 原始带符号", E_k)):
    print(f"\n== {name}")
    for i, dn in enumerate("CMY"):
        e = E[:, i]
        pk = W[np.argmax(e)]
        # 单位染料 (光谱单位) 叠在片基上, 扫描通道密度增量 (小量线性) 与大量 (c=2) 的增量
        for c in (0.2, 2.0):
            T = 10.0 ** -(base + c * e)
            d = measure(T[None], m)[0] - measure(10.0 ** -base[None], m)[0]
            s = -np.log10((10.0 ** -(base + c * e)) @ F) + np.log10((10.0 ** -base) @ F)
            print(f"  {dn} 峰 {pk:.0f} nm, c={c}: 扫描 R/G/B 增量 {d.round(3)} 比 {(d / d.max()).round(3)} | Status M {(s / s.max()).round(3)}")
    i550 = np.argmin(abs(W - 550)); i650 = np.argmin(abs(W - 650))
    print(f"  青染料 550/650 nm: {E[i550, 0] / E[i650, 0]:.3f}; 品红 650/550: {E[i650, 1] / E[i550, 1]:.3f}")
# 扫描的 P (窄带 → 印放密度) 与 gain: 纯青密度在 Dn 上的样子
sc = eng.scanner
print("\nP =\n", sc.P.round(3), "\ngain", sc.gain.round(3), "gray0", sc.gray0.round(3))
for c in (0.5, 1.0, 1.5, 2.0):
    T = 10.0 ** -(base + c * E_model[:, 0])
    Dn = sc.normalized_density(T[None])[0]
    print(f"  纯青 c={c}: Dn {Dn.round(3)}  Dn_G/Dn_R {Dn[1] / Dn[0]:.3f}")
