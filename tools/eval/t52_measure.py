"""T52: 整张图上颗粒看不看得见 —— 原尺寸与手机尺寸 (长边 1200) 的颗粒 RMS, 真实 RPL 扫描作锚, 可见度判据
(docs/tasks/T52_GRAIN_VISIBILITY.md)。

量 (8 位亮度码值 0–255; 同一套代码用于模型与真实; 块与估计器同 T51b, tools/eval/t51b_bw_level.grain_tiles):
  · 均匀块 32 × 32 像素 (去一次平面; T45 判据: 2 像素盒以上不比白噪声多; 块平均码值 12–250, 截断像素 ≤ 5%);
    中间调 = 块平均码值 80–180。
  · σ_hp: 减去 σ = 1 像素高斯后的标准差 (高通); σ_px: 去平面后的标准差。8 位量化方差 1/12 已扣。
  · 手机尺寸: 编码后的整张 (码值, 浮点) 用 Lanczos 缩到长边 1200, 取整到 8 位, 同一套量; 块另要求它覆盖的原尺寸区域
    全是原尺寸的均匀块 (缩小后细纹理会像白噪声, 原尺寸的判据能把它排除)。
  · 可见度 (手机尺寸): 块的功率谱按倍频程分 4 带 (1/32–1/2 周/像素), 每带 RMS 码值 → 亮度对比度 (显示按 sRGB 解码,
    c = σ·dL/dcode / L) → 等效正弦调制 m = √2·c; 与 Barten (1999) 对比敏感度给出的阈值调制 1/S(f) 之比 R = m·S(f)。
    观看条件 (假设): 手机 460 ppi、1:1 显示、30 cm (94.9 像素/度), 白 300 cd/m², 视场 = 图像长边的视角。
    S 用 colour-science 0.4.7 的 contrast_sensitivity_function_Barten1999 (标准观察者; 瞳孔、视网膜照度按 Barten 1999)。
    这是"正弦光栅阈值"用于带限噪声的近似判据: R ≥ 1 大致为可见, R ≤ 0.5 为看不见; 不是噪声检测的精确 JND。
用法:
  python3 -I t52_measure.py scenes  帧目录  输出.json
  python3 -I t52_measure.py real    RPL目录 输出.json.gz
  python3 -I t52_measure.py uniform 输出.json [配方 ...]
"""
import gzip
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE))

from t51b_bw_level import T, boxmean, detrend  # noqa: E402

PHONE = 1200
MID = (80.0, 180.0)
VIEW = dict(ppi=460.0, dist_mm=300.0, white_cd=300.0)
BANDS = ((1 / 32, 1 / 16), (1 / 16, 1 / 8), (1 / 8, 1 / 4), (1 / 4, 1 / 2))   # 周/像素


def to_code(lin):
    from filmlab.scanning.spatial import encode
    return encode(np.clip(np.asarray(lin, float), 0, None)).mean(-1) * 255.0


def phone(Y, long=PHONE):
    """码值图 → 长边 long 的 Lanczos 缩小 (浮点), 取整到 8 位。"""
    from PIL import Image
    h, w = Y.shape
    k = long / max(h, w)
    im = Image.fromarray(np.asarray(Y, np.float32), "F").resize((round(w * k), round(h * k)), Image.LANCZOS)
    return np.clip(np.round(np.asarray(im, float)), 0, 255)


_hann = np.outer(np.hanning(T), np.hanning(T))
_fy, _fx = np.meshgrid(np.fft.fftfreq(T), np.fft.fftfreq(T), indexing="ij")
_fr = np.hypot(_fy, _fx)


def band_var(r):
    """去平面的块 → 每个倍频程带的方差 (码值²; 汉宁窗, 归一到总方差)。"""
    P = np.abs(np.fft.fft2((r - r.mean()) * _hann)) ** 2 / (_hann ** 2).sum() / r.size
    return [float(P[(_fr >= a) & (_fr < b)].sum()) for a, b in BANDS]


def tiles(Y, lo=12, hi=250, flat=1.6, at=None, bands=False):
    """均匀块 (y, x, 码值, σ_px, σ_hp[, 4 带方差])。at: 指定块位置 (不再判平坦; 用于颗粒关的同位置对照)。"""
    out = []
    H, W = Y.shape
    pos = at if at is not None else [(y, x) for y in range(0, H - T + 1, T) for x in range(0, W - T + 1, T)]
    for y, x in pos:
        t = Y[y:y + T, x:x + T]
        # 码值范围按块平均判; 只排除截断像素 (0 / 255) 超过 5% 的块 (粗颗粒在原尺寸上 σ 可达 30–45 码值,
        # 按块内最小 / 最大值判会系统地排除颗粒最强的块)
        if at is None and (not lo <= t.mean() <= hi or ((t <= 0) | (t >= 255)).mean() > 0.05):
            continue
        r = detrend(t)
        if at is None and boxmean(r, 4).var() > flat * boxmean(r, 2).var() / 4.0:
            continue
        s_px = np.sqrt(max(r.var() - 1 / 12, 0.0))
        s_hp = (r - gaussian_filter(r, 1.0, mode="reflect"))[2:-2, 2:-2].std()
        row = [int(y), int(x), float(t.mean()), float(s_px), float(s_hp)]
        if bands:
            row.append(band_var(r))
        out.append(row)
    return out


def phone_tiles(Yp, native_shape, nat):
    """手机尺寸的均匀块, 另要求它覆盖的原尺寸区域全部是原尺寸的均匀块 (排除缩小后才像白噪声的细纹理: 头发、织物)。"""
    H, W = native_shape
    k = max(H, W) / max(Yp.shape)
    ok = np.zeros((H // T + 1, W // T + 1), bool)
    for t in nat:
        ok[t[0] // T, t[1] // T] = True
    out = []
    for t in tiles(Yp, bands=True):
        y0, y1 = int(t[0] * k) // T, int(np.ceil((t[0] + T) * k)) // T
        x0, x1 = int(t[1] * k) // T, int(np.ceil((t[1] + T) * k)) // T
        if y1 * T + T > H or x1 * T + T > W:
            continue
        if ok[y0:y1 + 1, x0:x1 + 1].all():
            out.append(t)
    return out


def summarize(ts, rng=MID):
    s = [t for t in ts if rng[0] <= t[2] < rng[1]]
    if len(s) < 3:
        return dict(n=len(s))
    a = np.array([t[2:5] for t in s])
    d = dict(n=len(s), code=float(np.median(a[:, 0])), s_px=float(np.median(a[:, 1])), s_hp=float(np.median(a[:, 2])))
    if len(s[0]) > 5:
        bv = np.median(np.array([t[5] for t in s]), 0)
        d["band_rms"] = np.sqrt(bv).tolist()
    return d


def by_level(ts, edges=(12, 40, 80, 120, 160, 200, 240)):
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        d = summarize(ts, (a, b))
        if d["n"] >= 5:
            out.append(dict(range=[a, b], **d))
    return out


# ------------------------------------------------------------------ 可见度
def csf(u_cpd, L, field_deg):
    from colour.contrast import (contrast_sensitivity_function_Barten1999, pupil_diameter_Barten1999,
                                 retinal_illuminance_Barten1999, sigma_Barten1999)
    d = pupil_diameter_Barten1999(L, field_deg, field_deg)
    sig = sigma_Barten1999(0.5 / 60, 0.08 / 60, d)
    E = retinal_illuminance_Barten1999(L, d)
    return np.asarray(contrast_sensitivity_function_Barten1999(u=np.asarray(u_cpd, float), sigma=sig, E=E,
                                                               X_0=field_deg, Y_0=field_deg), float)


def visibility(band_rms, code, long_px=PHONE, view=VIEW):
    """band_rms: 4 带的 RMS 码值; code: 平均码值。返回每带 (中心 周/度, 对比度, 阈值调制, R) 与 max R。"""
    from filmlab.scanning.spatial import decode
    px_deg = view["dist_mm"] * np.tan(np.radians(1.0)) / (25.4 / view["ppi"])
    field = long_px / px_deg
    v = code / 255.0
    L = view["white_cd"] * float(decode(v))
    dLdc = view["white_cd"] * float(decode(v + 0.5 / 255) - decode(v - 0.5 / 255))
    rows = []
    for (a, b), s in zip(BANDS, band_rms):
        f = np.sqrt(a * b) * px_deg
        c = s * dLdc / L
        S = float(csf(f, L, field))
        rows.append(dict(f_cpd=float(f), contrast=float(c), m_thresh=1.0 / S, R=float(np.sqrt(2) * c * S)))
    return dict(px_per_deg=float(px_deg), field_deg=float(field), L_cd=L, bands=rows, R_max=max(r["R"] for r in rows))


# ------------------------------------------------------------------ 场景帧 (t52_render.py 的输出)
def scenes(fdir, dst):
    fdir = Path(fdir)
    out = {}
    for f in sorted(fdir.glob("*.npy")):
        if "_digital_" in f.name or f.name.endswith("_g0.npy"):
            continue
        meta = json.load(open(str(f).replace(".npy", "_meta.json")))
        Y = to_code(np.load(f).astype(np.float32))
        nat = tiles(Y)
        Yp = phone(Y)
        ph = phone_tiles(Yp, Y.shape, nat)
        r = dict(meta=meta, native=dict(mid=summarize(nat), levels=by_level(nat), n_tiles=len(nat)),
                 phone=dict(shape=list(Yp.shape), mid=summarize(ph), levels=by_level(ph), n_tiles=len(ph)))
        m = r["phone"]["mid"]
        if "band_rms" in m:
            r["phone"]["visibility"] = visibility(m["band_rms"], m["code"])
        # 原尺寸 100% 显示 (手机上放大到 1:1 看局部) 的同一判据
        natb = tiles(Y, bands=True)
        mn = summarize(natb)
        if "band_rms" in mn:
            r["native"]["visibility_100pct"] = visibility(mn["band_rms"], mn["code"], long_px=max(Y.shape))
        # 颗粒关的同一帧 (若有): 同位置的景物残余
        g0 = Path(str(f).replace(".npy", "_g0.npy"))
        if g0.exists():
            Y0 = to_code(np.load(g0).astype(np.float32))
            sel = [(t[0], t[1]) for t in nat if MID[0] <= t[2] < MID[1]]
            selp = [(t[0], t[1]) for t in ph if MID[0] <= t[2] < MID[1]]
            r["native"]["grain_off_same_tiles"] = summarize(tiles(Y0, at=sel))
            r["phone"]["grain_off_same_tiles"] = summarize(tiles(phone(Y0), at=selp))
        out[f.stem] = r
        print(f.stem, "native", r["native"]["mid"], "phone", {k: v for k, v in m.items() if k != "band_rms"},
              "R", round(r["phone"].get("visibility", {}).get("R_max", float("nan")), 2), flush=True)
    json.dump(out, open(dst, "w"), indent=1, ensure_ascii=False)


# ------------------------------------------------------------------ 真实 RPL (645, Noritsu, 约 20 µm/像素)
def real(root, dst):
    from PIL import Image
    root = Path(root)
    out = {}
    for film in ("6_KODAK400TX", "7_ILFORDHP5PLUS", "8_ILFORDDELTA3200PRO"):
        for f in sorted((root / film).glob("*.jpg")):
            Y = np.asarray(Image.open(f).convert("L"), float)
            nat = tiles(Y)
            Yp = phone(Y)
            ph = phone_tiles(Yp, Y.shape, nat)
            r = dict(shape=list(Y.shape), phone_shape=list(Yp.shape),
                     native=dict(levels=by_level(nat), mid=summarize(nat), n_tiles=len(nat)),
                     phone=dict(levels=by_level(ph), mid=summarize(ph), n_tiles=len(ph)))
            if "band_rms" in r["phone"]["mid"]:
                r["phone"]["visibility"] = visibility(r["phone"]["mid"]["band_rms"], r["phone"]["mid"]["code"])
            out[f"{film}/{f.name}"] = r
            print(film, f.name, r["native"]["mid"], {k: v for k, v in r["phone"]["mid"].items() if k != "band_rms"},
                  round(r["phone"].get("visibility", {}).get("R_max", float("nan")), 2), flush=True)
    with gzip.open(dst, "wt") as fh:
        json.dump(out, fh)


# ------------------------------------------------------------------ 模型均匀灰 (固定定光), 原尺寸 + 手机尺寸
CASES = {
    # 名: (配方, 像素间距 µm, 手机缩小倍数 = 整帧长边像素 / 1200)
    "645_trix_xtol": ("kodak_trix_400@xtol", 20.0, 2796 / PHONE),
    "645_hp5_d76": ("ilford_hp5_plus", 20.0, 2796 / PHONE),
    "35_trix_d76": ("kodak_trix_400", 36000 / 3032, 3032 / PHONE),
    "35_trix_rodinal": ("kodak_trix_400@rodinal", 36000 / 3032, 3032 / PHONE),
    "35_hp5_d76": ("ilford_hp5_plus", 36000 / 3032, 3032 / PHONE),
    "35_hp5_rodinal": ("ilford_hp5_plus@rodinal", 36000 / 3032, 3032 / PHONE),
    "half_trix_rodinal": ("kodak_trix_400@rodinal", 24000 / 2857, 2857 / PHONE),
    "110_trix_rodinal": ("kodak_trix_400@rodinal", 17000 / 2830, 2830 / PHONE),
}


def uniform(dst, names=None, n=960, evs=(-2, -1, 0, 1, 2)):
    import warnings
    warnings.filterwarnings("ignore")
    from dataclasses import replace
    from filmlab import generation, grain
    from filmlab.engine import Engine, Params
    from filmlab.recipes import recipe
    from filmlab.scanning import spatial
    generation.use("t51_bw")
    grain.SILVER_WORKERS = 3
    out = json.load(open(dst)) if Path(dst).exists() else {}
    for name in names or CASES:
        rec, px, k = CASES[name]
        # 冲印店孔径 = 该画幅的输出像素间距 (t52_render.py 同): 渲染像素已是孔径
        spatial.LABS["noritsu_rpl@t52"] = replace(spatial.LABS["noritsu_rpl"], pitch_um=px)
        for ev in evs:
            key = f"{name}/{ev:+d}"
            if key in out:
                continue
            eng = Engine(Params(**{**recipe(rec), "timing": "gray", "roll": None, "frame": 0, "grain": 1.0,
                                   "scene_lights": None, "frame_width_mm": n * px / 1000.0,
                                   "scan_spatial": "noritsu_rpl@t52"}))
            o = eng.render(np.full((n, n, 3), 0.18 * 2.0 ** ev))
            Y = np.round(to_code(o))
            Y = Y[8:-8, 8:-8]                                   # 银粒渲染的视场边界 (T50 局限 5)
            nat = tiles(Y, lo=0, hi=256)
            Yp = phone(Y, long=round(max(Y.shape) / k))
            ph = tiles(Yp, lo=0, hi=256, bands=True)
            mp = summarize(ph, (0, 256))
            r = dict(recipe=rec, px_um=px, phone_factor=k, ev=ev, native=summarize(nat, (0, 256)), phone=mp)
            if "band_rms" in mp:
                r["phone_visibility"] = visibility(mp["band_rms"], mp["code"])
            nb = summarize(tiles(Y, lo=0, hi=256, bands=True), (0, 256))
            if "band_rms" in nb:                                # 高光处 8 位截断时没有合格的块
                r["native_visibility_100pct"] = visibility(nb["band_rms"], nb["code"], long_px=round(PHONE * k))
            out[key] = r
            print(key, r["native"], {a: b for a, b in mp.items() if a != "band_rms"},
                  round(r.get("phone_visibility", {}).get("R_max", float("nan")), 2),
                  round(r.get("native_visibility_100pct", {}).get("R_max", float("nan")), 2), flush=True)
            json.dump(out, open(dst, "w"), indent=1)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "scenes":
        scenes(sys.argv[2], sys.argv[3])
    elif cmd == "real":
        real(sys.argv[2], sys.argv[3])
    else:
        uniform(sys.argv[2], sys.argv[3:] or None)
