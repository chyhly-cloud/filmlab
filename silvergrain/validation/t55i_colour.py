"""T55i: 彩色负片 (染料云路径) 的同一套检验 —— 每层 σ(D) 的形状、冲印店扫描链的彩色部分、暗部 (码值 50–130) 差距的归因。
只用存下的测量 (data/realscan_rpl_colour_t55i.json, data/kodak_gran/) 与 silvergrain 自己的链; RPL 的亮度胶片侧与同码值读数只作留出。

1. kodak_shape: 每层 48 µm 漫射 RMS σ(净密度), 模型 (产品口径: T46 通用推导) 对 Kodak 5219 / 5213 / 5207 / 5203 (独立数据)。
2. chroma: Noritsu 链的色度处理。真实: 平坦块的 Cb/Y、Cr/Y 功率比与 RGB 互相干 (按空间频率), 通道 σ 比与相关。
   模型: 均匀块 (full 模式, 固定定光) → 产品的 noritsu_rpl 链 → 8 位 → JPEG 4:2:0 (真实文件同一量化表; 4:2:0 是交付文件的格式,
   不在库里做); 对照: 色度平滑 σ = 0 (T55e 的链)。训练片 (Pro 400H、Portra 400 / 160) / 留出片 (Portra 800、Ektar 100)。
3. luma_filmside: 亮度的胶片侧。每帧每码值档 (T54 平坦块的中位): 翻拍净密度 D_c、本帧影调斜率 dCode_c/dD_c → 模型的三层独立染料颗粒
   (原生 120 网格, 缩小, 每层锐化 = 亮度锐化, 线性) 经本帧实测影调得到的亮度 σ_48 预测; 真实 / 预测 按码值、按 D_G、按帧,
   并拟合 log 比值 = a + b·(码值 − 128)/100 + c·(D_G − 0.5) (b: 跟输出码值走; c: 跟底片密度走)。色度处理不改变亮度, 不影响这一项。
4. dark_gap: 归因的其余证据 (存下的测量, 不重算): 景物结构的局部传递按码值 (冲印店局部反差)、相邻包围帧的平坦块相关 (景物纹理)、
   影调斜率估计法 (三次拟合 vs 局部线性)。
"""
import io
import json
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent / "data"
FILMS = {"1_FUJIPRO400H": "fujifilm_pro_400h", "2_KODAKPORTRA400": "kodak_portra_400", "4_KODAKPORTRA160": "kodak_portra_160",
         "3_KODAKPORTRA800": "kodak_portra_800", "5_KODAKEKTAR100": "kodak_ektar_100"}
TRAIN = ("1_FUJIPRO400H", "2_KODAKPORTRA400", "4_KODAKPORTRA160")
HOLD = ("3_KODAKPORTRA800", "5_KODAKEKTAR100")
YCC = np.array([[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5], [0.5, -0.418688, -0.081312]])
W601 = YCC[0]
N = 32
FB = np.array([0.0, 0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.4375, 0.5, 0.75])
FIT_BINS = slice(1, 5)                     # 0.0625–0.3125 周/像素 (档 0 含景物纹理)


def ref():
    return json.loads((DATA / "realscan_rpl_colour_t55i.json").read_text(encoding="utf-8"))


# ------------------------------------------------------------------ 1. Kodak 每层形状
def kodak_shape():
    from . import t46_official as T
    from ..emulsion import universal
    U0 = universal()
    R = {r["key"]: r for r in T.rows()}
    out = {}
    for key in ("k5219", "k5213", "k5207", "k5203"):
        r = R[key]
        e, d, sp, Hx, L = T.model_layers(r, U0["eta"], U0["beta"], U0)
        K = T.kodak(r["value"].split()[0])
        res = {}
        for c, (net, s) in K.items():
            D, sg = L[c]; D = D - D.min(); i = np.argsort(D)
            rows = []
            for dd in (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.3, 1.6):
                if dd > 0.95 * min(D.max(), net.max()):
                    continue
                m = float(np.interp(dd, D[i], sg[i])); k = float(np.interp(dd, net, s))
                rows.append(dict(D=dd, kodak=k * 1e3, model=m * 1e3, ratio=m / k))
            hi = [x for x in rows if x["D"] >= 1.0]
            pk = max(rows, key=lambda x: x["kodak"])
            res["RGB"[c]] = dict(rows=rows, fall_kodak=hi[-1]["kodak"] / pk["kodak"] if hi else None,
                                 fall_model=hi[-1]["model"] / max(x["model"] for x in rows) if hi else None,
                                 toe_ratio=float(np.mean([x["ratio"] for x in rows if x["D"] <= 0.2])),
                                 high_ratio=float(np.mean([x["ratio"] for x in hi])) if hi else None)
        out[key] = res
    return out


# ------------------------------------------------------------------ 2. 色度
def _detrend(t):
    h, w = t.shape[:2]
    yy, xx = np.mgrid[:h, :w]
    A = np.stack([np.ones(h * w), yy.ravel(), xx.ravel()], 1)
    c, *_ = np.linalg.lstsq(A, t.reshape(h * w, -1), rcond=None)
    return t - (A @ c).reshape(t.shape)


class _Acc:
    """RGB 交叉谱 (按 |f| 档)、YCbCr 功率谱、盒尺度通道协方差 (与真实同一估计器: 32 × 32 块, 去一次平面, Hann 窗)。"""
    _f = np.fft.fftfreq(N)
    BIN = np.digitize(np.hypot(_f[:, None], _f[None, :]), FB) - 1
    HANN = np.hanning(N)[:, None] * np.hanning(N)[None, :]

    def __init__(self):
        self.S = np.zeros((len(FB) - 1, 3, 3)); self.P = np.zeros((len(FB) - 1, 3)); self.C = {b: np.zeros((3, 3)) for b in (1, 2, 4, 8)}

    def add(self, t):
        r = _detrend(t.astype(float))
        F = np.fft.fft2(r * self.HANN[..., None], axes=(0, 1))
        X = np.einsum("hwc,hwd->hwcd", F, np.conj(F)).real
        Py = np.abs(np.fft.fft2(_detrend(t.astype(float) @ YCC.T) * self.HANN[..., None], axes=(0, 1))) ** 2
        for k in range(len(FB) - 1):
            m = self.BIN == k
            if m.any():
                self.S[k] += X[m].sum(0); self.P[k] += Py[m].sum(0)
        for b in self.C:
            bm = r.reshape(N // b, b, N // b, b, 3).mean((1, 3)).reshape(-1, 3); bm = bm - bm.mean(0)
            self.C[b] += bm.T @ bm / len(bm)

    def result(self):
        def rho(M):
            d = np.sqrt(np.diag(M)); return [float(M[0, 1] / d[0] / d[1]), float(M[1, 2] / d[1] / d[2]), float(M[0, 2] / d[0] / d[2])]
        return dict(chroma_over_luma=(self.P[:, 1:] / np.maximum(self.P[:, :1], 1e-30)).tolist(),
                    coh_RG_GB_RB=[rho(self.S[k]) for k in range(len(FB) - 1)],
                    rho_box={f"{b * 20}um": rho(self.C[b]) for b in self.C},
                    s_over_Y=None)


def _jpeg420(u8, qt):
    from PIL import Image
    b = io.BytesIO()
    Image.fromarray(u8).save(b, "JPEG", qtables={int(k): v for k, v in qt.items()}, subsampling=2)
    b.seek(0)
    return np.asarray(Image.open(b).convert("RGB"), float)


def chroma(n=448, evs=(-1.5, -0.5, 0.5, 1.5)):
    import dataclasses
    import silvergrain as sg
    from ..image import get_scanner
    R = ref()
    qt = R["jpeg"]["qtables_colour"]
    sc = get_scanner("noritsu_rpl")
    variants = {"product": sc, "no_chroma (T55e)": dataclasses.replace(sc, chroma_sigma_px=0.0)}
    out = dict(sigma_px=sc.chroma_sigma_px, films={})
    for film, key in FILMS.items():
        acc = {v: {"dark": _Acc(), "bright": _Acc()} for v in variants}
        sd = {v: {"dark": [], "bright": []} for v in variants}
        for ev in evs:
            for v, s in variants.items():
                r = sg.apply_to_image(np.full((n, n, 3), 0.18), key, None, format=n * 0.02, film_gauge="120", mode="full",
                                      timing="fixed", exposure_ev=ev, seed=int(60 + 2 * ev), scanner=s)
                e = np.clip(np.round(sg.image.srgb_encode(np.clip(r.linear, 0, 1)) * 255.0), 0, 255).astype(np.uint8)
                img = _jpeg420(e, qt)
                Ym = float((img @ W601).mean()); band = "dark" if Ym < 130 else "bright"
                for y in range(16, n - N - 15, N):
                    for x in range(16, n - N - 15, N):
                        t = img[y:y + N, x:x + N]
                        acc[v][band].add(t)
                        b2 = _detrend(t).reshape(N // 2, 2, N // 2, 2, 3).mean((1, 3)).reshape(-1, 3)
                        s = b2.std(0); sY = (b2 @ W601).std()
                        sd[v][band].append(s / sY)
        res = {}
        for v in variants:
            res[v] = {}
            for b in ("dark", "bright"):
                if not sd[v][b]:
                    continue
                m = acc[v][b].result(); m["s_over_Y"] = np.median(sd[v][b], 0).tolist()
                rr = np.array(R["chroma"][film][b]["chroma_over_luma"])[FIT_BINS]; mm = np.array(m["chroma_over_luma"])[FIT_BINS]
                e = np.log(np.maximum(mm, 1e-6) / np.maximum(rr, 1e-6))
                m["rms_log_chroma"] = float(np.sqrt(np.mean(e ** 2))); m["bias_log_chroma"] = float(np.mean(e))
                res[v][b] = m
        out["films"][film] = dict(model=res, real=dict(chroma=R["chroma"][film], channel_ratios=R["channel_ratios"][film]),
                                  split="train" if film in TRAIN else "hold")
    for v in variants:
        for split, fl in (("train", TRAIN), ("hold", HOLD)):
            out.setdefault("rms_log", {}).setdefault(v, {})[split] = float(np.sqrt(np.mean(
                [out["films"][f]["model"][v][b]["rms_log_chroma"] ** 2 for f in fl for b in out["films"][f]["model"][v]])))
    return out


# ------------------------------------------------------------------ 3. 亮度的胶片侧
def model_layers(film_key, evs=range(-6, 5), seed=21):
    """每层 σ_48 (×1e-3, 染料当量密度) 对净密度: 原生 120 网格 → 缩小到 20 µm → 每层可分离锐化 (亮度锐化是线性的)。"""
    from . import scanner_fit as S
    from .filmside import block_stats
    from ..image import _sharpen3, get_scanner
    sc = get_scanner("noritsu_rpl")
    fog = S.native_field(film_key, -10, seed=seed)[1]
    rows = []
    for ev in evs:
        D, Dm, _ = S.native_field(film_key, ev, seed=seed)
        rows.append(dict(ev=ev, D=(Dm - fog).tolist(),
                         s48=[block_stats(_sharpen3(D[..., c], sc.sharpen_kx, sc.sharpen_ky) * 1000.0)["s48"] for c in range(3)]))
    S._CACHE.clear()
    return rows


def luma_filmside():
    R = ref()
    U = json.loads((DATA.parent.parent / "data" / "lab_tone.json").read_text(encoding="utf-8"))["rpl_noritsu"]["colour"]["u_factors"]
    out = {}
    for film in FILMS:
        md = model_layers(film); u = np.array(U[film])
        Dm = np.array([r["D"] for r in md]); Sm = np.array([r["s48"] for r in md]) * 1e-3
        rows = []
        for ev, n, code, sY, *rest in R["luma_rows"][film]:
            D = np.array(rest[:3]); sl = np.array(rest[3:6])
            pred2 = 0.0
            for c in range(3):
                o = np.argsort(Dm[:, c]); s = np.interp(D[c] / u[c], Dm[o, c], Sm[o, c])
                pred2 += (W601[c] * sl[c] * u[c] * s) ** 2
            rows.append(dict(ev=ev, n=n, code=code, D_G=D[1] / u[1], real=sY, pred=float(np.sqrt(pred2)), ratio=float(sY / np.sqrt(pred2))))
        A = np.array([[r["code"], r["D_G"], r["ratio"], r["n"]] for r in rows])
        X = np.stack([np.ones(len(A)), (A[:, 0] - 128) / 100.0, A[:, 1] - 0.5], 1)
        wts = np.sqrt(A[:, 3])
        coef, *_ = np.linalg.lstsq(X * wts[:, None], np.log(A[:, 2]) * wts, rcond=None)

        def wmed(m):
            o = np.argsort(A[m, 2]); c = np.cumsum(A[m, 3][o]); return float(A[m, 2][o][np.searchsorted(c, c[-1] / 2)])
        by_code = [dict(code=lo + 15, ratio=wmed((A[:, 0] >= lo) & (A[:, 0] < lo + 30))) for lo in range(40, 230, 30)
                   if ((A[:, 0] >= lo) & (A[:, 0] < lo + 30)).sum() >= 2]
        dark = (A[:, 0] >= 50) & (A[:, 0] < 130); bright = (A[:, 0] >= 150) & (A[:, 0] < 220)
        out[film] = dict(model_layers=md, rows=rows, loglin_const_code_D=coef.tolist(), by_code=by_code,
                         dark_50_130=wmed(dark), bright_150_220=wmed(bright), split="train" if film in TRAIN else "hold")
    return out


# ------------------------------------------------------------------ 4. 暗部差距的其余证据
def dark_gap():
    R = ref()
    bg = R["band_gain"]; codes = bg["codes"]
    dark_idx = [i for i, c in enumerate(codes) if 50 <= c <= 130]
    band = {}
    for film, v in bg["films"].items():
        band[film] = {b: {ch: (float(np.nanmedian([x for x in (vals[i] for i in dark_idx) if x is not None]))
                               if any(vals[i] is not None for i in dark_idx) else None) for ch, vals in d.items()} for b, d in v.items()}
    tex = {f: dict(dark=float(np.median([x["rho"] for x in v["by_code"] if 50 <= x["code"] <= 130])),
                   bright=float(np.median([x["rho"] for x in v["by_code"] if x["code"] >= 145])),
                   null=float(np.median([x["rho"] for x in v["null"]]))) for f, v in R["texture"].items()}
    return dict(band_gain_dark_rel_bright=band, texture_rho=tex, texture_control=R["texture_control"],
                slope_estimator={f: {m: v["coef"] for m, v in d.items()} for f, d in R["slope_estimator"].items()},
                jpeg_sampling=R["jpeg"]["sampling"])


def run(log=print):
    out = dict(kodak_shape=kodak_shape())
    log("  t55i kodak_shape")
    out["chroma"] = chroma()
    log("  t55i chroma")
    out["luma_filmside"] = luma_filmside()
    log("  t55i luma_filmside")
    out["dark_gap"] = dark_gap()
    return out
