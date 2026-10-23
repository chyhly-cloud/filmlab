"""用 Kodak 原始测量重新解混 Vision3 5219 的分层曲线, 并用不参与解混的实测数据独立验证。

数据 (Kodak 5219 数据手册, 经第三方数字化; 在 spektrafilm 历史提交中): Status M 特性曲线 (R/G/B, 绝对密度含 D-min)、
三种染料的光谱密度 (峰值归一、已减 D-min, 品红在蓝区为负 = 有色成色剂被消耗的遮罩补偿)、最低密度光谱、
中性中间调的光谱密度、Status M 滤光响应。

解混: 每个曝光点求三层染料量 c ≥ 0, 使 Status M(D-min + Σ c_i ε_i) = Kodak 实测 → 按构造与 Status M 一致。
独立验证: Kodak 实测的"中性中间调光谱密度"不参与解混; 在同一 Status M 水平上, 合成光谱必须与它吻合。
对照: FilmLab 现行表示 (spektrafilm 解混的分层曲线 + 截断染料谱) 与 spektrafilm 带符号谱。

用法: python3 unmix_5219.py 原始数据目录 [输出.npz]
"""
import json, sys
import numpy as np
from scipy.optimize import least_squares
sys.path.insert(0, "/home/user/filmlab")
from filmlab.stocks import WAVELENGTHS as W, PROFILE_DIR, load_stock

RAW = sys.argv[1] if len(sys.argv) > 1 else "."
BAND = (W >= 400) & (W <= 700)          # Kodak 数据的测量范围


def curve(name, extrapolate="edge"):
    a = np.loadtxt(f"{RAW}/{name}.csv", delimiter=",")
    a = a[np.argsort(a[:, 0])]
    if extrapolate == "zero":
        return np.interp(W, a[:, 0], a[:, 1], left=0.0, right=0.0)
    return np.interp(W, a[:, 0], a[:, 1])


def status_m():
    F = np.stack([curve(f"statusM_{c}", "zero") for c in "rgb"], 1)
    return F / F.sum(0)


def sm(D, F):
    """光谱密度 (..., 81) → Status M 积分密度 (..., 3)。"""
    return -np.log10(np.clip(10.0 ** -D @ F, 1e-9, None))


def kodak():
    E = np.stack([curve(f"dye_density_{c}") for c in "cmy"], 1)
    base = curve("dye_density_min")
    mid = curve("dye_density_mid")
    hd = [np.loadtxt(f"{RAW}/density_curve_{c}.csv", delimiter=",") for c in "rgb"]
    x = hd[1][:, 0]
    D = np.stack([np.interp(x, h[:, 0], h[:, 1]) for h in hd], 1)
    return E, base, mid, x, D


def unmix(E, base, D, F):
    out = np.zeros((len(D), 3)); c = np.zeros(3)
    for i, t in enumerate(D):
        c = least_squares(lambda v: sm(base + E @ v, F) - t, np.maximum(c, 0.01), bounds=(0, 10)).x
        out[i] = c
    return out


def at_status(target, E, base, x, C, F):
    """在 x 上找 Status M 最接近 target 的曝光 (插值), 返回该处合成的光谱密度。"""
    xs = np.linspace(x.min(), x.max(), 4000)
    Cs = np.stack([np.interp(xs, x, C[:, k]) for k in range(3)], 1)
    S = sm(base + Cs @ E.T, F)
    j = np.argmin(((S - target) ** 2).sum(1))
    return base + Cs[j] @ E.T, S[j], xs[j]


def rms(a, b):
    return float(np.sqrt(np.mean((a - b)[BAND] ** 2)))


if __name__ == "__main__":
    F = status_m()
    E, base, mid, x, D = kodak()
    C = unmix(E, base, D, F)
    resid = sm(base + C @ E.T, F) - D
    print(f"Kodak 原始解混: Status M 残差 最大 {np.abs(resid).max():.4f} D")
    t_mid = sm(mid, F)
    print(f"Kodak 中性中间调的 Status M: {np.round(t_mid, 3)}")
    rows = {}
    sp_k, s_k, x_k = at_status(t_mid, E, base, x, C, F)
    rows["Kodak 原始谱 + 重新解混"] = (sp_k, s_k)
    st = load_stock("kodak_vision3_500t")
    d = json.loads((PROFILE_DIR / "kodak_vision3_500t.json").read_text())["data"]
    from filmlab.stocks import _clean
    signed = _clean(d["channel_density"])
    for name, Es in (("FilmLab 现行 (截断谱 + spektrafilm 分层曲线)", st.dye_density), ("spektrafilm 带符号谱 + 同一分层曲线", signed)):
        rows[name] = at_status(t_mid, Es, st.base_density, st.log_exposure, st.density_curves, F)[:2]
    print("独立验证: 与 Kodak 实测中性中间调光谱密度的差 (400–700 nm)")
    for name, (sp, s) in rows.items():
        dlt = (sp - mid)[BAND]
        print(f"  {name:36s} RMS {rms(sp, mid):.4f} D  最大 {np.abs(dlt).max():.4f} D  (Status M 差 {np.round(s - t_mid, 3)})")
    print("  片基 / D-min 光谱: Kodak 实测 vs FilmLab 现行  RMS", round(rms(base, st.base_density), 4), "D")
    # 中间调 gamma (Status M, 相对绿)
    def gam(Sx, Dm):
        m = (Dm[:, 1] - Dm[:, 1].min() > 0.3) & (Dm[:, 1] - Dm[:, 1].min() < 1.3)
        return [np.polyfit(Dm[m, 1], Dm[m, k], 1)[0] for k in (0, 2)]
    print("Kodak 实测 Status M 中间调反差 R/G, B/G:", np.round(gam(x, D), 3))
    if len(sys.argv) > 2:
        np.savez(sys.argv[2], E=E, base=base, x=x, C=C, D=D)
        print("已保存", sys.argv[2])
