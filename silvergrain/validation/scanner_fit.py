"""T55e: Noritsu HS-1800 (RPL) 的扫描采样链 —— 在真实扫描的 2D 颗粒谱上拟合, 留出片与跨片种检验 (只用存下的测量)。

真实 (data/realscan_rpl_2d.json.gz, 53 帧, 测量方法见文件的 _note): 每帧的 2D 中位谱 (16 × 16 粗格)、每列 / 每行颗粒方差的
0.2747 周/像素调制、邻点相关、块内分布形状; data/edges_rpl_x.json.gz: T51 的输送方向直边。
模型 (silvergrain 自己的链, image.scan_grid / _to_output_grid / _sharpen3):
  均匀照片 (18% × 2^EV) → 该片的底片, 渲染在 HS-1800 的原生网格上 (120: 56 mm / 4824 = 11.6 µm; 像素孔径 = 原生像素)
  → 冲印店软件在处理后的图像上按 1.7253 倍缩小 (三角核按倍数拉宽) → 冲印店影调 (锚点使该块印到该帧的中位码值)
  → 输出像素上的可分离 3 抽头锐化 (kx 输送方向, ky CCD 方向) → 8 位。
拟合: 训练帧 = 彩色 Pro 400H、Portra 400、Portra 160 (与 T51 同一划分) 中有同 EV 模型场的帧; 只拟合 (kx, ky) —— 谱形
(log 中位谱, |f| > 0.08), 不拟合量级。对照结构: 不锐化; 再加扫描光学 σ_o = 3 µm; T55c 的链 (渲染在输出像素上, 网格高斯 0.45 像素
+ USM 1.03 / 0.68)。留出: 彩色 Portra 800 / Ektar 100; 跨片种: Tri-X (XTOL)、HP5+ (D-76) 只用训练出的 (kx, ky)。
直边: T51 的输送方向直边, 每条边的 σ_e 自由, 比较 (台阶 ⊗ σ_e ⊗ 本链 → kx) 与 T51 (台阶 ⊗ σ_e ⊗ 1 像素方盒 → USM) 的代价。
"""
import gzip
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

DATA = Path(__file__).resolve().parent / "data"
PX = 20.0
N_OUT = 448
TILE = 64
TRAIN = ("1_FUJIPRO400H", "2_KODAKPORTRA400", "4_KODAKPORTRA160")
HOLD = ("3_KODAKPORTRA800", "5_KODAKEKTAR100")
BW = ("6_KODAK400TX", "7_ILFORDHP5PLUS")
FILMS = {"1_FUJIPRO400H": ("fujifilm_pro_400h", None), "2_KODAKPORTRA400": ("kodak_portra_400", None),
         "3_KODAKPORTRA800": ("kodak_portra_800", None), "4_KODAKPORTRA160": ("kodak_portra_160", None),
         "5_KODAKEKTAR100": ("kodak_ektar_100", None), "6_KODAK400TX": ("kodak_trix_400", "xtol"),
         "7_ILFORDHP5PLUS": ("ilford_hp5_plus", "d76")}
MODEL_EVS = {"1_FUJIPRO400H": (-2, 0, 2), "2_KODAKPORTRA400": (-2, -1, 0, 1, 2, 3), "4_KODAKPORTRA160": (-2, 0, 2),
             "3_KODAKPORTRA800": (-2, 0, 2), "5_KODAKEKTAR100": (-2, 0, 2), "6_KODAK400TX": (-3, -2, -1, 0, 1, 2, 3),
             "7_ILFORDHP5PLUS": (-2, -1, 0, 1, 2)}
W601 = np.array([0.299, 0.587, 0.114])
_HANN = np.hanning(TILE)[:, None] * np.hanning(TILE)[None, :]
_HANN = _HANN / np.sqrt((_HANN ** 2).mean())


def load_real():
    return json.load(gzip.open(DATA / "realscan_rpl_2d.json.gz", "rt"))["frames"]


# ------------------------------------------------------------------ 测量 (与真实同一估计器) --
def _detrend(t):
    h, w = t.shape
    yy, xx = np.mgrid[:h, :w]
    A = np.stack([np.ones(h * w), yy.ravel(), xx.ravel()], 1)
    c, *_ = np.linalg.lstsq(A, t.ravel(), rcond=None)
    return t - (A @ c).reshape(h, w)


def _tiles(Y, border=8):
    out = []
    for y in range(border, Y.shape[0] - TILE - border + 1, TILE):
        for x in range(border, Y.shape[1] - TILE - border + 1, TILE):
            out.append((y, x, _detrend(Y[y:y + TILE, x:x + TILE])))
    return out


def quad16(S):
    S = np.fft.fftshift(np.asarray(S)); S = S / S.sum()
    Q = (S[32:, 32:] + S[32:0:-1, 32:0:-1][:32, :32] + S[32:, 32:0:-1][:, :32] + S[32:0:-1, 32:][:32]) / 4
    return Q.reshape(16, 2, 16, 2).mean((1, 3)) * 4096


_FQ = (np.arange(16) * 2 + 0.5) / 64
QMASK = np.hypot(_FQ[:, None], _FQ[None, :]) > 0.08


def stats(Y, full=True):
    """中位谱 (16 × 16)、邻点相关、行列方差调制、块内分布 (同 realscan_rpl_2d 的定义)。full=False 只算中位谱。"""
    ts = _tiles(Y)
    P = np.stack([np.abs(np.fft.fft2(r * _HANN) / TILE) ** 2 / r.var() for _, _, r in ts])
    if not full:
        return dict(quad16=quad16(np.median(P, 0)))
    rho, z = [], []
    H, W = Y.shape
    colv, colc, rowv, rowc = np.zeros(W), np.zeros(W), np.zeros(H), np.zeros(H)
    for y, x, r in ts:
        v = r.var()
        rho.append([np.mean(r[:, 1:] * r[:, :-1]) / v, np.mean(r[1:] * r[:-1]) / v, np.mean(r[1:, 1:] * r[:-1, :-1]) / v,
                    np.mean(r[:, 2:] * r[:, :-2]) / v, np.mean(r[2:] * r[:-2]) / v])
        z.append((r / np.sqrt(v)).ravel())
        colv[x:x + TILE] += (r ** 2).mean(0) / v; colc[x:x + TILE] += 1
        rowv[y:y + TILE] += (r ** 2).mean(1) / v; rowc[y:y + TILE] += 1

    def mod(v, c, f0=0.2747):
        ok = c > 0
        xx = np.nonzero(ok)[0]; vv = v[ok] / c[ok]; vv = vv - vv.mean()
        return max(2 * abs(np.sum(vv * np.exp(-2j * np.pi * f * xx))) / len(xx) for f in np.linspace(f0 - 0.004, f0 + 0.004, 81))
    z = np.concatenate(z); zc = np.clip(z, -4, 4)
    sub = {}
    for b in (64, 8):
        zz = np.concatenate([(_detrend(r[i:i + b, j:j + b] + 0.0) / max(_detrend(r[i:i + b, j:j + b] + 0.0).std(), 1e-9)).ravel()
                             for _, _, r in ts for i in range(0, TILE, b) for j in range(0, TILE, b)])
        sub[b] = [float(np.mean(zz ** 4) - 3), float(np.mean(zz ** 3))]
    return dict(quad16=quad16(np.median(P, 0)), rho=np.median(rho, 0), mod_x=mod(colv, colc), mod_y=mod(rowv, rowc),
                sd=float(np.median([r.std() for _, _, r in ts])), kurt_clip4=float(np.mean(zc ** 4) / np.mean(zc ** 2) ** 2 - 3),
                skew=float(np.mean(z ** 3)), kurt_subtile=sub, n=len(ts))


# ------------------------------------------------------------------ 模型 --
_CACHE = {}


def native_field(film_key, ev, seed=11, method=None):
    """均匀 18% × 2^EV 在 HS-1800 120 原生网格上的底片 → 输出网格 (N_OUT²) 的密度 (h, w, 层) 与该 EV 的平均密度。"""
    from ..image import GREY, expose, get_scanner, scan_grid, _to_output_grid
    from ..render import render
    key = (film_key, ev, seed, method)
    if key in _CACHE:
        return _CACHE[key]
    film, dev = FILMS[film_key]
    ex = expose(np.full((N_OUT, N_OUT, 3), GREY * 2.0 ** ev), film, dev, format=N_OUT * PX / 1000.0)
    g = scan_grid(ex.rgb.shape, ex.px_um, ex.format, get_scanner("noritsu_rpl"), gauge="120")
    shp = tuple(g["shape_native"])
    layers = 1 if ex.emulsion.film.is_bw else 3
    H = np.full(shp + ((3,) if layers == 3 else ()), 1.0) * (ex.H.reshape(-1, layers)[0] if layers == 3 else ex.H.ravel()[0])
    m = method or ("closed_form" if layers == 1 else "auto")
    rr = render(ex.emulsion, exposure=H, px_um=g["px_native"], seed=seed + ev, method=m)
    Dm = np.asarray(rr.mean, float).reshape(-1, layers).mean(0)
    out = _to_output_grid(rr, shp, (N_OUT, N_OUT), layers)
    D = np.asarray(out.density, float).reshape(N_OUT, N_OUT, layers)
    _CACHE[key] = (D, Dm, g)
    return _CACHE[key]


def t55c_field(film_key, ev, seed=11):
    """T55c 的链: 渲染在输出像素 (20 µm) 上, 输出网格上的高斯 0.45 像素 (透射率上)。"""
    from ..image import GREY, expose, _gauss
    from ..render import render
    film, dev = FILMS[film_key]
    ex = expose(np.full((N_OUT, N_OUT, 3), GREY * 2.0 ** ev), film, dev, format=N_OUT * PX / 1000.0)
    layers = 1 if ex.emulsion.film.is_bw else 3
    rr = render(ex.emulsion, exposure=ex.H, px_um=ex.px_um, seed=seed + ev, method="closed_form" if layers == 1 else "auto")
    D = np.asarray(rr.density, float).reshape(N_OUT, N_OUT, layers)
    D = np.stack([-np.log10(np.clip(_gauss(10.0 ** -D[..., c], 0.45), 1e-12, None)) for c in range(layers)], -1)
    return D, np.asarray(rr.mean, float).reshape(-1, layers).mean(0)


def to_luma(D, Dm, code):
    """密度 → 冲印店影调的码值 (锚点使平均密度印到 code; 彩色每层各自, 再取 BT.601 亮度)。"""
    from ..labtone import tone
    layers = D.shape[2]
    T = tone("rpl_noritsu", "bw" if layers == 1 else "colour")
    c = np.stack([T[k].code_of(D[..., k] - (Dm[k] - T[k].u_of(code))) for k in range(layers)], -1)
    return c[..., 0] if layers == 1 else c @ W601


def finish(Y, kx, ky, usm=None):
    from ..image import _sharpen3, _gauss
    Y = _sharpen3(Y, kx, ky, ccd_axis=0)
    if usm:
        Y = Y + usm[0] * (Y - _gauss(Y, usm[1]))
    return np.clip(np.round(Y), 0, 255)


def _frames(real, films):
    out = []
    for k, v in sorted(real.items()):
        if v["film"] in films and v["ev"] in MODEL_EVS.get(v["film"], ()) and v["n_tiles"] >= 30:
            out.append((k, v))
    return out


def fit_k(items, free=True, k0=(0.15, 0.3), usm=None, iso=False):
    """items: [(真实帧, 锐化之前的码值图)]; 拟合 (kx, ky) (iso: kx = ky) 或在 k0 上评价。返回 (kx, ky, rms)。"""
    RQ = [np.log(np.asarray(v["quad16"]))[QMASK] for v, _ in items]

    def res(q):
        kx, ky = (q[0], q[0]) if iso else (q[0], q[1])
        return np.concatenate([rq - np.log(stats(finish(Y, kx, ky, usm), full=False)["quad16"])[QMASK] for rq, (_, Y) in zip(RQ, items)])
    if free:
        x0 = [k0[0]] if iso else list(k0)
        s = least_squares(res, x0, diff_step=0.03, loss="soft_l1", f_scale=0.3, max_nfev=40,
                          bounds=([-0.2] * len(x0), [0.8] * len(x0)))
        q, r = (np.array([s.x[0], s.x[0]]) if iso else s.x), s.fun
    else:
        q, r = np.asarray(k0, float), res(k0)
    return float(q[0]), float(q[1]), float(np.sqrt(np.mean(r ** 2)))


# ------------------------------------------------------------------ 直边 --
_DX = 0.01
_G = np.arange(-12, 12 + _DX / 2, _DX)


def _cdf(kind, native_px=1 / 1.7253):
    def conv(a, b):
        return np.convolve(a, b, mode="same") * _DX
    if kind == "t51":
        k = (np.abs(_G) <= 0.5).astype(float)
    else:
        k = (np.abs(_G) <= native_px / 2).astype(float)
        k = conv(k / (k.sum() * _DX), np.clip(1 - np.abs(_G), 0, None))
    k = k / (k.sum() * _DX)
    return np.cumsum(k) * _DX


def _esf(d, d0, se, cdf, sharp):
    g = np.exp(-_G ** 2 / (2 * max(se, 0.05) ** 2)); g /= g.sum()
    n = len(g) // 2
    E = np.convolve(np.pad(cdf, n, mode="edge"), g, mode="valid")[: len(_G)]
    f = lambda x: np.interp(x - d0, _G, E)
    if sharp[0] == "sep3":
        k = sharp[1]
        return (1 + 2 * k) * f(d) - k * (f(d - 1) + f(d + 1))
    if sharp[0] == "usm":
        a, su = sharp[1], sharp[2]
        gs = np.exp(-_G ** 2 / (2 * su * su)); gs /= gs.sum()
        E2 = np.convolve(np.pad(E, len(gs) // 2, mode="edge"), gs, mode="valid")[: len(_G)]
        return (1 + a) * f(d) - a * np.interp(d - d0, _G, E2)
    return f(d)


def edge_costs(films, variants, hw=4.0):
    """每条边 (lo, 台阶, d0, σ_e, 两侧斜坡) 自由; 链与锐化固定。先按 T51 的方法筛边 (σ_e < 3 像素, 去掉残差最大的 20%)。"""
    ed = json.load(gzip.open(DATA / "edges_rpl_x.json.gz", "rt"))["films"]
    edges = [(np.asarray(e["d"], float), np.asarray(e["y"], float)) for f in films for e in ed.get(f, [])]

    def fit_one(d, y, cdf, sharp):
        ok = np.isfinite(y) & (np.abs(d) <= hw)
        dd, yy = d[ok], y[ok]
        n4 = max(4, ok.sum() // 8)
        lo, hi = yy[:n4].mean(), yy[-n4:].mean()

        def res(p):
            m = p[0] + p[1] * (_esf(dd, p[2], np.exp(p[3]), cdf, sharp) + p[4] * np.minimum(dd - p[2], 0) + p[5] * np.maximum(dd - p[2], 0))
            return (m - yy) / max(abs(p[1]), 1.0)
        best = None
        for se0 in (0.3, 1.0, 2.0):
            s = least_squares(res, [lo, hi - lo, 0.0, np.log(se0), 0.0, 0.0],
                              bounds=([-np.inf, -np.inf, -3, np.log(0.05), 0, 0], [np.inf, np.inf, 3, np.log(12), 0.05, 0.05]))
            if best is None or s.cost < best.cost:
                best = s
        return float(np.mean(best.fun ** 2)), float(np.exp(best.x[3]))
    c51 = _cdf("t51")
    first = [fit_one(d, y, c51, ("none",)) for d, y in edges]
    keep = [i for i in range(len(edges)) if first[i][1] < 3.0]
    keep = sorted(keep, key=lambda i: first[i][0])[: max(3, int(0.8 * len(keep)))]
    out = dict(n=len(keep))
    for name, (kind, sharp) in variants.items():
        cdf = _cdf(kind)
        r = [fit_one(*edges[i], cdf, sharp) for i in keep]
        out[name] = dict(cost=float(np.mean([a for a, _ in r])), sigma_e_median=float(np.median([b for _, b in r])))
    return out


# ------------------------------------------------------------------ 主程序 --
def run(log=print):
    from ..image import get_scanner
    sc = get_scanner("noritsu_rpl")
    real = load_real()
    out = dict(scanner=dict(sharpen_kx=sc.sharpen_kx, sharpen_ky=sc.sharpen_ky, native=sc.native, resample=sc.resample, optics_um=sc.optics_um))

    def items(films, optics_um=0.0):
        res = []
        for k, v in _frames(real, films):
            D, Dm, g = native_field(v["film"], v["ev"])
            if optics_um > 0:
                D = _blur_native(v["film"], v["ev"], optics_um)
            res.append((v, to_luma(D, Dm, v["code"])))
        return res
    tr = items(TRAIN); log(f"  scanner_fit 训练帧 {len(tr)}")
    kx, ky, rms = fit_k(tr)
    out["train"] = dict(frames=[v["film"] + f" EV{v['ev']:+d}" for v, _ in tr], kx=kx, ky=ky, rms=rms,
                        isotropic=dict(zip(("kx", "ky", "rms"), fit_k(tr, k0=(0.2, 0.2), iso=True))),
                        rms_no_sharpen=fit_k(tr, free=False, k0=(0.0, 0.0))[2],
                        rms_at_scanner_json=fit_k(tr, free=False, k0=(sc.sharpen_kx, sc.sharpen_ky))[2])
    tr3 = items(TRAIN, optics_um=3.0)
    out["train"]["optics_3um"] = dict(zip(("kx", "ky", "rms"), fit_k(tr3)))
    t55c = [(v, to_luma(*t55c_field(v["film"], v["ev"]), v["code"])) for v, _ in tr]
    out["train"]["t55c_chain_rms"] = fit_k(t55c, free=False, k0=(0.0, 0.0), usm=(1.032, 0.683))[2]
    log(f"  scanner_fit 训练 kx {kx:.3f} ky {ky:.3f} RMS {rms:.4f}")
    k = (sc.sharpen_kx, sc.sharpen_ky)
    for name, films in (("hold_colour", HOLD), ("bw_cross_type", BW)):
        it = items(films)
        r = dict(rms_at_train=fit_k(it, free=False, k0=k)[2], free=dict(zip(("kx", "ky", "rms"), fit_k(it))),
                 rms_no_sharpen=fit_k(it, free=False, k0=(0.0, 0.0))[2])
        r["frames"] = []
        for v, Y in it:
            st = stats(finish(Y, *k))
            r["frames"].append(dict(frame=f"{v['film']} EV{v['ev']:+d}", code=v["code"],
                                    rho_real=v["rho"][:2], rho_model=st["rho"][:2].tolist(),
                                    mod_real=[v["mod_x"], v["mod_y"]], mod_model=[st["mod_x"], st["mod_y"]],
                                    top_octave_real=_top(v["quad16"]), top_octave_model=_top(st["quad16"])))
        out[name] = r
        log(f"  scanner_fit {name}: RMS {r['rms_at_train']:.4f} (自由 {r['free']['rms']:.4f}, 不锐化 {r['rms_no_sharpen']:.4f})")
    out["edges"] = dict(train=edge_costs(TRAIN, {"none_t51_box": ("t51", ("none",)), "t51_usm": ("t51", ("usm", 1.032, 0.683)),
                                                 "chain_none": ("chain", ("none",)), "chain_sep3_kx": ("chain", ("sep3", sc.sharpen_kx))}),
                        bw=edge_costs(BW, {"none_t51_box": ("t51", ("none",)), "chain_sep3_kx": ("chain", ("sep3", sc.sharpen_kx))}))
    log("  scanner_fit 直边")
    # 分布形状 (显式银粒, 黑白): 模型的 8 / 64 像素子块峰度、偏度 对 真实
    shp = []
    for k_, v in _frames(real, BW):
        D, Dm, g = native_field(v["film"], v["ev"], method="explicit")
        st = stats(finish(to_luma(D, Dm, v["code"]), *k))
        shp.append(dict(frame=f"{v['film']} EV{v['ev']:+d}", code=v["code"], real_kurt_clip4=v["z"]["kurt_clip4"], model_kurt_clip4=st["kurt_clip4"],
                        real_skew=v["z"]["skew"], model_skew=st["skew"],
                        real_kurt_8px=v["kurt_subtile"]["8"][0] if v.get("kurt_subtile") else None, model_kurt_8px=st["kurt_subtile"][8][0],
                        real_sd=v["sd"], model_sd=st["sd"]))
    out["shape_bw"] = shp
    log("  scanner_fit 分布形状")
    return out


def _top(q):
    """顶部倍频程 (0.25–0.5 周/像素) 对 0.06–0.19 周/像素 的功率比, 按 x 轴 (fy < 0.06) 与 y 轴分开。"""
    q = np.asarray(q)
    return [float(q[:2, 8:].mean() / q[:2, 2:6].mean()), float(q[8:, :2].mean() / q[2:6, :2].mean())]


def _blur_native(film_key, ev, optics_um, seed=11):
    """原生网格上的扫描光学 (高斯 σ_o, 透射率上) 再缩小 —— 只用于 σ_o 的对照。"""
    from ..image import GREY, expose, get_scanner, scan_grid, resample2
    from ..render import render
    film, dev = FILMS[film_key]
    ex = expose(np.full((N_OUT, N_OUT, 3), GREY * 2.0 ** ev), film, dev, format=N_OUT * PX / 1000.0)
    g = scan_grid(ex.rgb.shape, ex.px_um, ex.format, get_scanner("noritsu_rpl"), gauge="120")
    shp = tuple(g["shape_native"])
    layers = 1 if ex.emulsion.film.is_bw else 3
    H = np.full(shp + ((3,) if layers == 3 else ()), 1.0) * (ex.H.reshape(-1, layers)[0] if layers == 3 else ex.H.ravel()[0])
    rr = render(ex.emulsion, exposure=H, px_um=g["px_native"], seed=seed + ev, method="closed_form" if layers == 1 else "auto")
    D = np.asarray(rr.density, float).reshape(shp + (layers,))
    s = optics_um / g["px_native"]
    fy = np.fft.fftfreq(shp[0])[:, None]; fx = np.fft.fftfreq(shp[1])[None, :]
    G = np.exp(-2 * np.pi ** 2 * s * s * (fx ** 2 + fy ** 2))
    D = np.stack([-np.log10(np.clip(np.fft.ifft2(np.fft.fft2(10.0 ** -D[..., c]) * G).real, 1e-12, None)) for c in range(layers)], -1)
    return resample2(D, (N_OUT, N_OUT)).reshape(N_OUT, N_OUT, layers)
