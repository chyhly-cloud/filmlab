"""Vision3 500T ColorChecker 对比 (色调无关)。

数据: Die Lichtfänger "Kodak Vision3 500T im Test" (dielichtfaenger.com/blog/kodak-vision3-500t-im-test):
同一场景 (ColorChecker Passport, ARRI T1 3200K) 的 500T Noritsu HS-1800 平坦扫描 (Bild_4_NULL) 与
Fuji X100V RAW (X100V_02_01_Null.RAF)。

方法: 用每张图自己的灰阶 (第 4 行) 把每个通道反查成"等效场景对数反射率", 抵消扫描曲线、白平衡与曝光,
只剩"颜色相对灰阶的分离方向与大小" —— 对比真实扫描、理想光谱曝光 (Ohta 色卡光谱 × 3200K × 500T 感光度)
与本系统各阶段 (曝光 → 染料 → 扫描密度 → 输出)。

用法: python3 tools/eval/chart_500t.py 扫描.jpg X100V.RAF
"""
import sys
import numpy as np
import colour
from PIL import Image

sys.path.insert(0, __file__.rsplit("/tools/", 1)[0])
from filmlab.engine import Engine, Params
from filmlab.io import srgb_encode
from filmlab.materials import emulsion
from filmlab.rawio import load_raw
from filmlab.spectral import WAVELENGTHS, illuminant

# 色块中心网格 (x0, y0, 每列 dx, 每行 x 偏移, 每列 y 偏移, 每行 dy, 采样半宽), 全分辨率像素
GRID_SCAN = (3845, 2145, 106, -10, 7, 105, 20)          # 6774×4492 扫描
GRID_RAW = (1835, 1176.7, 37, -0.7, 0, 39, 8)            # load_raw(max_size=3000) 的 3123×2085
NEUTRAL = np.log10([0.90, 0.59, 0.36, 0.19, 0.09, 0.031])
NAMES = "深肤 浅肤 蓝天 树叶 蓝花 青绿 橙 紫蓝 中红 紫 黄绿 橙黄 蓝 绿 红 黄 品红 青".split()


def patches(img, g, scale=1.0):
    x0, y0, dx, dxr, dyc, dy, h = g
    hh = max(2, int(h * scale))
    out = []
    for r in range(4):
        for c in range(6):
            x, y = int((x0 + dx * c + dxr * r) * scale), int((y0 + dy * r + dyc * c) * scale)
            out.append(np.median(img[y - hh:y + hh, x - hh:x + hh].reshape(-1, 3), 0))
    return np.array(out, float)


def vec(P):
    """每通道按灰阶反查 → (r−g, b−g) 对数分离向量 (18 个彩色块)。P 只需与反射率单调相关。"""
    s = np.empty_like(P)
    for c in range(3):
        o = np.argsort(P[18:, c])
        s[:, c] = np.interp(P[:, c], P[18:, c][o], NEUTRAL[o])
    return np.stack([s[:18, 0] - s[:18, 1], s[:18, 2] - s[:18, 1]], 1)


def summary(name, V, ref, detail=False):
    m = np.hypot(*V.T) / np.hypot(*ref.T)
    a = np.degrees(np.angle(np.exp(1j * (np.arctan2(V[:, 1], V[:, 0]) - np.arctan2(ref[:, 1], ref[:, 0])))))
    print(f"{name:26s} 分离比 中位 {np.median(m):.2f}   |Δ方向| 中位 {np.median(np.abs(a)):3.0f}°   "
          + "  ".join(f"{NAMES[i]} {a[i]:+.0f}°" for i in (10, 15, 13, 16, 14)))
    if detail:
        for i in range(18):
            print(f"    {NAMES[i]:4s} 比 {m[i]:4.2f}  Δ {a[i]:+4.0f}°")


def ideal_exposure():
    sds = colour.SDS_COLOURCHECKERS["ColorChecker N Ohta"]
    W = WAVELENGTHS
    R = np.array([sds[k].copy().align(colour.SpectralShape(W[0], W[-1], W[1] - W[0])).values for k in sds])
    return (R * illuminant("BB3200")) @ 10.0 ** emulsion("kodak_vision3_500t").stock.log_sensitivity


def main(scan_path, raf_path):
    ideal = vec(np.log10(ideal_exposure()))
    summary("真实 Noritsu 扫描", vec(patches(np.asarray(Image.open(scan_path)), GRID_SCAN)), ideal)
    raw = load_raw(raf_path, max_size=3000)
    raw = raw[0] if isinstance(raw, tuple) else raw
    summary("X100V 相机 RGB (输入)", vec(np.log10(np.maximum(patches(raw, GRID_RAW), 1e-6))), ideal)
    gray = patches(raw, GRID_RAW)[21] @ np.array([0.2126, 0.7152, 0.0722])   # 线性亮度 (n5 色块)
    for scanner in ("noritsu", "frontier"):
        eng = Engine(Params(film="kodak_vision3_500t", process="ECN-2", scanner=scanner, timing="frame"))
        rec = {}
        dev, to_rgb = eng._develop, eng.scanner.to_rgb
        eng._develop = lambda x, **kw: rec.setdefault("logH", x) is not None and dev(x, **kw)
        eng.scanner.to_rgb = lambda Dn: (rec.__setitem__("Dn", Dn.copy()), to_rgb(Dn))[1]
        # 场记板: ISO 500 测光 → 色卡中灰 (n5, 反射率 ≈ 0.19) 落在胶片中灰
        out = eng.render(raw[::2, ::2] * 0.184 / gray)
        h, w = out.shape[:2]
        summary(f"{scanner}: 曝光 log H", vec(patches(rec["logH"], GRID_RAW, 0.5)), ideal)
        summary(f"{scanner}: 染料量", vec(patches(eng.state.dye, GRID_RAW, 0.5)), ideal)
        summary(f"{scanner}: 扫描密度", vec(-patches(rec["Dn"].reshape(h, w, 3), GRID_RAW, 0.5)), ideal)
        summary(f"{scanner}: 输出", vec(patches(np.round(srgb_encode(np.clip(out, 0, 1)) * 255), GRID_RAW, 0.5)),
                ideal)


if __name__ == "__main__":
    main(*sys.argv[1:3])
