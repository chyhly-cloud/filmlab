"""T55j: view8 里发现的三个问题的检验。

1. grain 模式的黑点 (pepper): 颗粒传递的位置。合成画面 (大片亮中性 + 一条深红, Tri-X, 自动定光把深红印得比照片暗):
   比较 grain_density = "film" (T55j 默认: δ 在该片自己的 D̄, 传递在照片影调) / "lab_negative" (备选: 印成这张照片的底片 +
   该片的颜色响应, 传递在照片影调) / "film_print" (T55d: δ 与传递都在该片自己的印片位置) 的暗于一半像素比例与纯黑像素比例。
   一致性: 照片 = 该片无颗粒的印片时三者逐点相同 (测试 test_grain_mode_equals_full_mode_fluctuation)。
2. 彩色自动定光: RPL 31 帧的扫描 (data/rpl_colour_timing_t55j.json)。灰世界 (每通道把整帧平均印成中灰) 预言整帧平均颜色被
   消掉、中性墙带上补色; 中性平衡预言整帧平均保留场景的颜色、墙保持中性。灰世界分数 a: 墙的颜色 = −a × 场景平均颜色,
   即 −c_墙 = a (c_平均 − c_墙) (c = ln R/G、ln B/G, 线性光), 31 帧合并的最小二乘与标准误。
3. k = 2 的速度: 闭式解 / 显式银粒的像素 σ 比只取决于等效像素 px/k (场按 k 严格缩放); 在 2.9 µm 与原阈值 3.0 µm 比较
   (种子间散布作容差)。
"""
import json
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent / "data"


def pepper():
    import silvergrain as sg
    img = np.full((192, 192, 3), 0.8); img[:, 120:] = (0.30, 0.025, 0.02)
    out = {}
    for gd in ("film", "lab_negative", "film_print"):
        r = sg.apply_to_image(img, "kodak_trix_400", format=192 * 12 / 1000, seed=2, grain_density=gd)
        e = r.encoded[8:-8, 128:-8] * 255
        Y = e @ np.array([0.2126, 0.7152, 0.0722])
        out[gd] = dict(dark_half=float(np.mean(Y < 0.5 * Y.mean())), black=float(np.mean(e.max(-1) <= 2)), Y_mean=float(Y.mean()),
                       Y_sd=float(Y.std()))
    return out


def timing():
    d = json.loads((DATA / "rpl_colour_timing_t55j.json").read_text(encoding="utf-8"))

    def lin(c):
        c = np.asarray(c, float) / 255.0
        return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    C, Wl, per = [], [], {}
    for film, v in d["films"].items():
        for fr in v["frames"]:
            m = np.log(lin(fr["scan_mean_code"])); w = np.log(lin(fr["scan_wall_code"]))
            C.append([m[0] - m[1], m[2] - m[1]]); Wl.append([w[0] - w[1], w[2] - w[1]])
            per.setdefault(film, []).append(dict(tag=fr["tag"], mean_colour=C[-1], wall_colour=Wl[-1]))
    C, Wl = np.array(C), np.array(Wl)
    out = dict(n_frames=len(C), frame_mean_colour_median=np.median(C, 0).tolist(), wall_colour_median=np.median(Wl, 0).tolist(),
               frame_mean_colour_range=[C.min(0).tolist(), C.max(0).tolist()], per_film=per)
    for k, name in ((0, "R/G"), (1, "B/G")):
        x = C[:, k] - Wl[:, k]; y = -Wl[:, k]
        a = float((x * y).sum() / (x * x).sum()); res = y - a * x
        se = float(np.sqrt((res ** 2).sum() / (len(x) - 1) / (x * x).sum()))
        out[f"a_{name}"] = dict(a=a, se=se)
    return out


def kscale(n=160):
    from ..emulsion import derive
    from ..render import AUTO_MIN_PX_UM_MACRO, render
    from .fastpath import stats
    rows = []
    for dev in ("d76", "rodinal"):
        for k, px, seed in ((1.0, 3.0, 3), (1.0, 3.0, 4), (1.0, 2.9, 3), (2.0, 5.8, 3), (1.0, 2.67, 3)):
            em = derive("kodak_trix_400", dev, k)
            r = []
            for ev in (-2, 0, 2):
                H = np.full((n, n), 10.0 / em.film.ei * 2.0 ** ev)
                a = render(em, exposure=H, px_um=px, seed=seed, method="explicit")
                b = render(em, exposure=H, px_um=px, seed=seed, method="closed_form")
                r.append(stats(b.density)["sd"] / stats(a.density)["sd"])
            rows.append(dict(developer=dev, k=k, px_um=px, eq_um=px / k, seed=seed, sd_ratio=r))
    tol = {dev: min(min(x["sd_ratio"]) for x in rows if x["developer"] == dev and x["eq_um"] == 3.0) for dev in ("d76", "rodinal")}
    return dict(rows=rows, threshold_um=AUTO_MIN_PX_UM_MACRO, tolerance_min_ratio_at_3um=tol,
                k2_equals_k1_half_px=all(np.allclose(a["sd_ratio"], b["sd_ratio"]) for a in rows for b in rows
                                         if a["developer"] == b["developer"] and a["k"] == 2.0 and b["k"] == 1.0 and b["px_um"] == 2.9 and b["seed"] == a["seed"]))


def run(log=print):
    out = dict(pepper=pepper())
    log("  t55j pepper")
    out["timing"] = timing()
    out["kscale"] = kscale()
    log("  t55j kscale")
    return out
