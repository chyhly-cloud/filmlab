"""T33 A: 人造光源下蓝层 RGB → 曝光映射的系数与误差, 现状 (blue_xyz) 与三个候选方案 (docs/tasks/T33_BLUE_LAYER_DIR.md 第 2 节)。
方案: cur = XYZ 里 NNLS (现状 srgb_bxyz 的蓝层), srgb = sRGB NNLS (更早的), d1 = 只用 sRGB B (保白点),
      d2 = 只用 Z (保白点), d3x = XYZ NNLS 且 X 系数 = 0 (保白点), d3r = sRGB NNLS 且 R 系数 = 0 (保白点)。
用法: python3 t33_blue_map.py   (输出 4 张表: 系数、训练集误差、发光饱和色、T32 红像素与 T16 灯边)"""
import numpy as np, sys, json, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scipy.optimize import nnls
from filmlab import lights, generation, spectral
from filmlab.stocks import load_stock, WAVELENGTHS as W
generation.use("t31_base")
S = 10.0 ** load_stock("kodak_vision3_500t").log_sensitivity
TO_XYZ = np.linalg.inv(lights._daylight_wb()[0]).T          # 线性 sRGB → XYZ / Y_白 (行向量右乘)
LUMA = np.array([0.2126, 0.7152, 0.0722])


def fit(light, mode):
    """返回 (输入空间 'srgb'|'xyz', 蓝层系数 3 向量)。保白点 (白反射率下蓝层曝光精确)。"""
    R = lights.training_reflectances()
    rad = R * light[None, :]
    sc = 1.0 / (lights.camera_rgb(light) @ LUMA)
    tgt = (rad * sc) @ S[:, 2]
    rgb = lights.camera_rgb(rad) * sc
    xyz = rgb @ TO_XYZ
    w = 1.0 / np.maximum(((rad * sc) @ S).mean(1), 1e-6)       # 与 linear_response 相同的相对误差权重
    if mode in ("cur", "d3x", "d2"):
        sp, X = "xyz", xyz
    else:
        sp, X = "srgb", rgb
    if mode in ("cur", "srgb"):
        m = nnls(X * w[:, None], tgt * w)[0]
    elif mode == "d1":
        m = np.array([0, 0, 1.0])
    elif mode == "d2":
        m = np.array([0, 0, 1.0])
    elif mode == "d3x":
        m = np.r_[0.0, nnls(X[:, 1:] * w[:, None], tgt * w)[0]]
    elif mode == "d3r":
        m = np.r_[0.0, nnls(X[:, 1:] * w[:, None], tgt * w)[0]]
    wl = lights.camera_rgb(light) * sc
    wl = wl @ TO_XYZ if sp == "xyz" else wl
    m = m * ((light * sc) @ S[:, 2]) / max(wl @ m, 1e-12)
    return sp, m


def apply(sp, m, rgb):
    x = np.clip(rgb @ TO_XYZ, 0, None) if sp == "xyz" else rgb
    return x @ m


MODES = ["srgb", "cur", "d1", "d2", "d3x", "d3r"]


def gauss(c, fwhm):
    return np.exp(-0.5 * ((W - c) / (fwhm / 2.355)) ** 2)


def lines(ls, fwhm=5.0):
    return sum(a * gauss(c, fwhm) for c, a in ls)


# 发光饱和色 (窄带 / 实际光源光谱)。霓虹: Ne 主要谱线 (NIST 相对强度, 取可见区较强者); 绿霓虹: 汞氩管 + Zn2SiO4:Mn 绿荧光粉
NE = [(585.2, 20), (588.2, 10), (594.5, 10), (597.6, 5), (603.0, 4), (607.4, 6), (609.6, 6), (614.3, 10), (616.4, 4),
      (621.7, 5), (626.6, 8), (630.5, 4), (633.4, 10), (638.3, 10), (640.2, 20), (650.6, 15), (659.9, 10), (667.8, 5), (692.9, 10), (703.2, 8)]
EMIT = {
    "红 LED 630": gauss(630, 18),
    "红霓虹 (Ne)": lines(NE),
    "琥珀 LED 590": gauss(590, 15),
    "低压钠 589": lines([(589.0, 1.0)], 3),
    "高压钠 (HPS)": lights._sd("HPS"),
    "绿 LED 525": gauss(525, 30),
    "绿霓虹 (Hg+绿粉)": gauss(525, 40) + lines([(404.7, 0.05), (435.8, 0.15), (546.1, 0.3)]),
    "蓝 LED 465": gauss(465, 22),
}


def red_like(target):
    """与给定相机 RGB 一致的非负发光光谱 (红 630 + 绿 525 + 蓝 450 LED 组合; 以及荧光灯 × 红色亚克力长通 + 短波漏光)。"""
    out = {}
    B = np.stack([gauss(630, 18), gauss(525, 30), gauss(450, 20)], 1)
    a = np.linalg.solve(lights.camera_rgb(B.T).T, target)
    out["红630+绿525+蓝450 LED"] = (B @ np.clip(a, 0, None), a)
    lp = 1 / (1 + np.exp(-(W - 600) / 8))
    for n in ("Cool White FL", "Triphosphor FL"):
        fl = lights._sd(n)
        # 两个分量: 长通部分 × 荧光灯, 与 < 480nm 的漏光 × 荧光灯; 再加小量 525 绿 (亚克力的绿区透过) 让 G 匹配
        comp = np.stack([fl * lp, fl * (W < 480), fl * np.exp(-0.5 * ((W - 540) / 25) ** 2)], 1)
        a = np.linalg.solve(lights.camera_rgb(comp.T).T, target)
        out[f"{n} × 红亚克力 (长通 600 + 短波漏光)"] = (comp @ np.clip(a, 0, None), a)
    return out



def main():
    test_R = lights.training_reflectances.__wrapped__(2000, 2000)        # 独立随机种子 (同 T16)
    L32 = lights._blackbody(3200); s32 = 1.0 / (lights.camera_rgb(L32) @ LUMA)
    HgB = (L32 * s32) @ S[:, 2]                                           # 3200K 白 (Y=1) 的蓝层曝光
    zb = spectral.cmfs()[:, 2]
    k0 = (L32 @ S[:, 2]) / (L32 @ zb)
    print("== 1. 现状 blue_xyz 的蓝层系数: XYZ 里 [X Y Z] (以该灯白的蓝层曝光为 1) 与折算的 sRGB 系数; k = 每单位 Z 的蓝层曝光 / 3200K 黑体的")
    for c in lights.classes():
        if c.key == "daylight":
            continue
        for name, L in c.members:
            sc = 1.0 / (lights.camera_rgb(L) @ LUMA); Hw = (L * sc) @ S[:, 2]
            sp, m = fit(L, "cur")
            print(f"  {c.key:12s} {name[:30]:30s} XYZ {np.round(m / Hw, 3).tolist()}  sRGB {np.round(TO_XYZ @ m / Hw, 3).tolist()}"
                  f"  k/k0 {(L @ S[:, 2]) / (L @ zb) / k0:.2f}")
    print("\n== 2. 训练反射率 (独立 2000 条) 的蓝层误差: 中位 / 90% |Δlog10| [sRGB 蓝 < 0 的真实颜色的中位]; 类内各成员中位")
    for c in lights.classes():
        if c.key == "daylight":
            continue
        acc = {md: [] for md in MODES}
        for name, L in c.members:
            sc = 1.0 / (lights.camera_rgb(L) @ LUMA)
            rad = test_R * L[None, :] * sc
            rgb = lights.camera_rgb(rad); tgt = rad @ S[:, 2]; Hw = (L * sc) @ S[:, 2]
            ok = tgt > 1e-3 * Hw; neg = ok & (rgb[:, 2] < 0)
            for md in MODES:
                p = apply(*fit(L, md), rgb)
                e = np.abs(np.log10(np.maximum(p, 1e-6 * Hw) / tgt))
                acc[md].append((np.median(e[ok]), np.percentile(e[ok], 90), np.median(e[neg]) if neg.any() else np.nan))
        print(f"  {c.key:12s} " + " | ".join(f"{md} {np.median([v[0] for v in acc[md]]):.3f}/{np.median([v[1] for v in acc[md]]):.3f}"
                                          f" [{np.nanmedian([v[2] for v in acc[md]]):.2f}]" for md in MODES))
    print("\n== 3. 发光饱和色 (与中灰同亮度): 蓝层曝光 / 3200K 中灰的蓝层曝光; 真值 = 光谱直接积分; 各方案取全部人造光源成员的中位 [范围]")
    for n, E in EMIT.items():
        E = E / (lights.camera_rgb(E) @ LUMA)
        x = lights.camera_rgb(E); t = E @ S[:, 2] / HgB
        row = f"  {n:16s} sRGB(Y=1) {np.round(x, 3).tolist()} 真值 {t:.1e}"
        for md in ("cur", "d1", "d3r"):
            v = np.array([apply(*fit(Lm, md), x) / HgB for c in lights.classes() if c.key != "daylight" for _, Lm in c.members])
            row += f" | {md} {np.median(v):+.1e} [{v.min():+.1e}…{v.max():+.1e}]"
        print(row)
    print("\n== 4. EPFL 红像素 (T32 各档 RAW 中位, 线性 sRGB) 与 T16 灯边: 蓝层曝光 / 3200K 中灰 (Y=0.184) 的蓝层曝光; 类内成员中位")
    g32 = 0.184 * lights.light_rgb(L32)
    X = {"0…+1 档": [0.286, 0.034, 0.026], "+1…+2 档": [0.561, 0.098, 0.043], "+2…+3 档": [0.983, 0.069, 0.041],
         "T16 灯边": [1.29, 1.01, -0.09], "3200K 中灰": g32.tolist()}
    print(f"  3200K 中灰的线性 sRGB {np.round(g32, 4).tolist()} (日光中灰 0.184), Z {(g32 @ TO_XYZ)[2]:.4f} (日光中灰 {(np.full(3, .184) @ TO_XYZ)[2]:.4f})")
    for n, x in X.items():
        x = np.array(x); z = x @ TO_XYZ
        print(f"  {n}: Z/Z(3200K 中灰) {z[2] / (g32 @ TO_XYZ)[2]:.2f}, sRGB B/B(3200K 中灰) {x[2] / g32[2]:.2f}, Z 的份额 R G B {np.round(TO_XYZ[:, 2] * x / z[2], 2).tolist()}")
        for c in lights.classes():
            if c.key == "daylight":
                continue
            print(f"     {c.key:12s} " + " | ".join(
                f"{md} {np.median([apply(*fit(Lm, md), x) / (0.184 * HgB) for _, Lm in c.members]):+.2f}" for md in MODES))


if __name__ == "__main__":
    main()
