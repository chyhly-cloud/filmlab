"""T53 颗粒粗细 k: 整条链 (t51_bw, Noritsu) 上的均匀灰与整帧的颗粒量 (docs/tasks/T53_GRAIN_COARSENESS.md)。

量与 T52 相同 (tools/eval/t52_measure.py 的同一套代码: 32 像素均匀块、σ_px / σ_hp、手机尺寸 = 长边 1200 的 Lanczos、
Barten 可见度判据)。均匀灰: 18% 灰 × 2^EV, timing = "gray" (固定定光), 960 × 960, 35 mm 像素间距 36000/3032 µm,
手机缩小倍数 3032/1200 (同 T52 的 35_trix_d76); 另记录平均码值与平均线性输出 (特性曲线经整条链是否不变)。
用法:
  python3 -I t53_measure.py uniform 输出.json [情形 ...]
  python3 -I t53_measure.py scenes  帧目录 输出.json
"""
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE))

import t52_measure as m52  # noqa: E402

PX35, F35 = 36000 / 3032, 3032 / m52.PHONE
PXH, FH = 24000 / 2857, 2857 / m52.PHONE
KS_MAIN = (0.5, 1.0, 1.5, 2.0, 3.0)
KS_FINE = (0.75, 1.25, 1.75, 2.5)


def _rec(base, k):
    return base if k == 1.0 else f"{base}+g{k:g}"


CASES = {}
for _k in KS_MAIN + KS_FINE:
    CASES[f"35_trix_d76_k{_k:g}"] = (_rec("kodak_trix_400", _k), PX35, F35, (-1, 0, 1) if _k in KS_MAIN else (0,))
CASES["35_trix_rodinal_k1"] = ("kodak_trix_400@rodinal", PX35, F35, (0,))
CASES["half_trix_d76_k1"] = ("kodak_trix_400", PXH, FH, (0,))
CASES["half_trix_rodinal_k1"] = ("kodak_trix_400@rodinal", PXH, FH, (0,))
for _k in (1.0, 2.0):
    CASES[f"35_portra400_k{_k:g}"] = (_rec("kodak_portra_400", _k), PX35, F35, (0,))


def uniform(dst, names=None, n=960):
    import warnings
    warnings.filterwarnings("ignore")
    from filmlab import generation, grain
    from filmlab.engine import Engine, Params
    from filmlab.recipes import recipe
    from filmlab.scanning import spatial
    generation.use("t51_bw")
    grain.SILVER_WORKERS = 3
    out = json.load(open(dst)) if Path(dst).exists() else {}
    for name in names or CASES:
        rec, px, k, evs = CASES[name]
        spatial.LABS["noritsu_rpl@t52"] = replace(spatial.LABS["noritsu_rpl"], pitch_um=px)   # 孔径 = 渲染像素 (同 T52)
        for ev in evs:
            key = f"{name}/{ev:+d}"
            if key in out:
                continue
            eng = Engine(Params(**{**recipe(rec), "timing": "gray", "roll": None, "frame": 0, "grain": 1.0,
                                   "scene_lights": None, "frame_width_mm": n * px / 1000.0,
                                   "scan_spatial": "noritsu_rpl@t52"}))
            o = eng.render(np.full((n, n, 3), 0.18 * 2.0 ** ev))[8:-8, 8:-8]   # 银粒渲染的视场边界 (T50 局限 5)
            Yf = m52.to_code(o)
            Y = np.round(Yf)
            nat = m52.tiles(Y, lo=0, hi=256)
            Yp = m52.phone(Y, long=round(max(Y.shape) / k))
            ph = m52.tiles(Yp, lo=0, hi=256, bands=True)
            mp = m52.summarize(ph, (0, 256))
            r = dict(recipe=rec, px_um=px, phone_factor=k, ev=ev, native=m52.summarize(nat, (0, 256)), phone=mp,
                     mean_code=float(Yf.mean()), mean_linear=np.asarray(o, float).mean((0, 1)).tolist(),
                     mean_code_phone=float(Yp.mean()))
            if "band_rms" in mp:
                r["phone_visibility"] = m52.visibility(mp["band_rms"], mp["code"])
            nb = m52.summarize(m52.tiles(Y, lo=0, hi=256, bands=True), (0, 256))
            if "band_rms" in nb:
                r["native_visibility_100pct"] = m52.visibility(nb["band_rms"], nb["code"], long_px=round(m52.PHONE * k))
            # 整场 (不分块) 的像素级标准差与相邻像素相关 (颗粒结构 ≪ 12 µm 像素时接近 0)
            d = Yf - Yf.mean()
            r["field"] = dict(sd=float(Yf.std()), rho_x=float((d[:, 1:] * d[:, :-1]).mean() / d.var()),
                              rho_y=float((d[1:] * d[:-1]).mean() / d.var()))
            out[key] = r
            print(key, r["native"], {a: b for a, b in mp.items() if a != "band_rms"}, "mean", round(r["mean_code"], 2),
                  "rho", round(r["field"]["rho_x"], 3), flush=True)
            json.dump(out, open(dst, "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "scenes":
        m52.scenes(sys.argv[2], sys.argv[3])
    else:
        uniform(sys.argv[2], sys.argv[3:] or None)
