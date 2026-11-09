"""T55f: 两个未解的差距 —— 分布尾巴, 与同码值下颗粒随底片密度的增长 —— 只用存下的测量 (data/realscan_rpl_filmside.json,
data/realscan_rpl_2d.json.gz) 与 silvergrain 自己的链。

1. 尾巴: 在原生网格 (120: 11.6 µm; 135: 5.34 µm) 上显式银粒渲染 vs closed_form (对数正态), 同一扫描链;
   比较每像素密度与输出码值的偏度 / 峰度 (显式渲染是否给出真实的尾巴)。
2. 胶片侧 σ_D(D): 真实 = 扫描 ↔ 底片翻拍配准后, σ_码值 / 本帧斜率 (见数据文件说明); 模型 = 显式银粒在 120 原生网格上 → 缩小 →
   锐化 (密度域, 线性) → 同一块估计器。若差距来自扫描仪的噪声底, 它只取决于底片密度 (同一台扫描仪), 两种片的超额方差应相同。
"""
import json
from pathlib import Path

import numpy as np

from . import scanner_fit as S

DATA = Path(__file__).resolve().parent / "data"
BLOCK = 32
FILMS = ("6_KODAK400TX", "7_ILFORDHP5PLUS")


def _boxmean(a, b):
    h, w = (a.shape[0] // b) * b, (a.shape[1] // b) * b
    return a[:h, :w].reshape(h // b, b, w // b, b).mean((1, 3))


def block_stats(Y, px_um=20.0):
    """32 × 32 块 (去一次平面): σ_px、σ_48 (2 × 2 盒 × 2·px/42.5)、每块自身归一的偏度 / 超额峰度的中位。"""
    s_px, s48, sk, ku = [], [], [], []
    for y in range(0, Y.shape[0] - BLOCK + 1, BLOCK):
        for x in range(0, Y.shape[1] - BLOCK + 1, BLOCK):
            r = S._detrend(Y[y:y + BLOCK, x:x + BLOCK])
            z = (r / r.std()).ravel()
            s_px.append(r.std()); s48.append(_boxmean(r, 2).std() * (2 * px_um / 42.5))
            sk.append(np.mean(z ** 3)); ku.append(np.mean(z ** 4) - 3)
    return dict(s_px=float(np.median(s_px)), s48=float(np.median(s48)), skew=float(np.median(sk)), kurt=float(np.median(ku)))


def model_film_side(film_key, evs=(-3, -2, -1, 0, 1, 2, 3, 4)):
    from ..image import _sharpen3, get_scanner
    sc = get_scanner("noritsu_rpl")
    rows = []
    for ev in evs:
        D, Dm, g = S.native_field(film_key, ev, method="explicit")
        st = block_stats(_sharpen3(D[..., 0], sc.sharpen_kx, sc.sharpen_ky) * 1000.0)
        rows.append(dict(ev=ev, D=float(Dm[0]), **st))
    return rows


def tails(film="kodak_trix_400", dev="d76", n=384):
    """显式 vs closed_form, 原生网格 (135: 输出 12 / 6 µm; 120: 输出 20 µm), full 模式固定定光, EV 0 / +2。"""
    import silvergrain as sg
    out = []
    for px, gauge in ((20.0, "120"), (12.0, "135"), (6.0, "135")):
        for ev in (0, 2):
            row = dict(px_um=px, gauge=gauge, ev=ev)
            for m in ("explicit", "closed_form"):
                r = sg.apply_to_image(np.full((n, n, 3), 0.18 * 2.0 ** ev), film, dev, format=n * px / 1000.0, film_gauge=gauge,
                                      mode="full", timing="fixed", seed=4, method=m, keep_render=True)
                st = block_stats(r.encoded[..., 1] * 255.0, px)
                d = np.asarray(r.render.density, float); z = (d - d.mean()) / d.std()
                row[m] = dict(code=float(r.encoded[..., 1].mean() * 255), code_skew=st["skew"], code_kurt=st["kurt"],
                              D_skew=float(np.mean(z ** 3)), D_kurt=float(np.mean(z ** 4) - 3), s=r.info["scan_grid"]["s"])
            out.append(row)
    return out


def run(log=print):
    ref = json.loads((DATA / "realscan_rpl_filmside.json").read_text(encoding="utf-8"))
    out = dict(film_side={}, tails=None)
    for f in FILMS:
        mod = model_film_side(f)
        Dm = [r["D"] for r in mod]
        rows = []
        for r in ref["film_side"][f]:
            m48 = float(np.interp(r["D"], Dm, [x["s48"] for x in mod])); mpx = float(np.interp(r["D"], Dm, [x["s_px"] for x in mod]))
            rows.append(dict(D=r["D"], real_s48=r["s48"], model_s48=m48, ratio_s48=r["s48"] / m48, real_s_px=r["s_px"], model_s_px=mpx,
                             ratio_s_px=r["s_px"] / mpx, excess_var_s48=r["s48"] ** 2 - m48 ** 2, frames=r["frames_min_max"]))
        out["film_side"][f] = dict(rows=rows, model=mod)
        log(f"  filmside {f}")
    out["tails"] = tails()
    log("  filmside tails")
    out["shape_by_code_real"] = ref["shape_by_code"]
    out["local_variance"] = ref["local_variance"]
    return out
