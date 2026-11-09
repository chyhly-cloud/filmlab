"""T51b: 黑白颗粒的绝对量级 —— RPL 真实 Noritsu 扫描 (Tri-X、HP5+) 与模型 (t51_bw, Noritsu 空间传递开) 在同一输出码值上比较,
并把比值分解为 胶片侧密度颗粒 × 影调斜率 dCode/dD_neg × 残差。

真实 (只读图像, 只存测量):
  · 颗粒: 每帧的均匀块 (32 × 32 像素 = 640 µm; T45 判据: 去一次平面, 2b 盒平均方差 ≤ 1.6 × b 盒方差 / 4, b = 2 像素;
    码值 12–250)。离焦背景、平滑的皮肤都自动入选。量 (8 位码值, 量化方差 1/12 已扣):
      σ_px  像素级 (去平面后的标准差), σ_hp  高通 (减去 σ = 1 像素高斯), σ_48 = 2 × 2 盒平均 (40 µm) 的标准差 × 40/42.5;
    按块平均码值分档。
  · 影调斜率: 同一帧的扫描与底片照片 (RPL 的 "Neg Sequence": 同一底片在灯箱上的数码翻拍, 片边未曝光处 = 片基) 配准,
    底片密度 D = −log10(T / T_片基) (翻拍按 sRGB 解码当作线性; 片边码值 213–224 各帧一致, 说明翻拍曝光固定、未截断)。
    逐帧在 D 上分档取扫描码值的中位 → 平滑的码值(D) → 斜率 dCode/dD。
模型: 均匀 18% 灰 × 2^EV, t51_bw, timing="gray" (固定定光), 20 µm/像素 (RPL 645 的像素间距), 颗粒开;
  同一套颗粒量; 扫描密度 Dn 的 48 µm 颗粒 (scan trace); 颗粒关、EV ± 0.1 的有限差分给 dCode/dDn。
用法: python3 -I t51b_bw_level.py real  RPL目录 输出.json
      python3 -I t51b_bw_level.py model 配方 输出.json [边长像素] [代际 (默认 t51_bw; T56 用 t56_bw)]
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, zoom

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))

PX_UM = 20.0
T = 32
EDGES = np.arange(0, 256, 16)


def srgb_lin(c):
    c = np.asarray(c, float) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def detrend(t):
    h, w = t.shape
    yy, xx = np.mgrid[:h, :w]
    A = np.stack([np.ones(h * w), yy.ravel(), xx.ravel()], 1)
    c, *_ = np.linalg.lstsq(A, t.ravel(), rcond=None)
    return t - (A @ c).reshape(h, w)


def boxmean(a, b):
    h, w = (a.shape[0] // b) * b, (a.shape[1] // b) * b
    return a[:h, :w].reshape(h // b, b, w // b, b).mean((1, 3))


def grain_tiles(Y, flat=1.6, lo=12, hi=250, synthetic=False):
    """Y: (H, W) 码值 (float)。返回每块 (y, x, 平均码值, σ_px, σ_hp, σ_48)。"""
    out = []
    H, W = Y.shape
    for y in range(0, H - T + 1, T):
        for x in range(0, W - T + 1, T):
            t = Y[y:y + T, x:x + T]
            if not synthetic and (t.min() < lo or t.max() > hi):
                continue
            r = detrend(t)
            if not synthetic and boxmean(r, 4).var() > flat * boxmean(r, 2).var() / 4.0:
                continue
            q = 0.0 if synthetic else 1.0 / 12.0
            s_px = np.sqrt(max(r.var() - q, 0.0))
            hp = (r - gaussian_filter(r, 1.0, mode="reflect"))[2:-2, 2:-2]
            s_hp = hp.std()
            s48 = np.sqrt(max(boxmean(r, 2).var() - q / 4, 0.0)) * (2 * PX_UM / 42.5)
            out.append((y, x, float(t.mean()), float(s_px), float(s_hp), float(s48)))
    return out


def by_level(tiles, key):
    """按平均码值分档的中位。key: 3 = σ_px, 4 = σ_hp, 5 = σ_48。"""
    a = np.array([[t[2], t[key]] for t in tiles])
    res = []
    for i in range(len(EDGES) - 1):
        s = a[(a[:, 0] >= EDGES[i]) & (a[:, 0] < EDGES[i + 1])]
        if len(s) >= 5:
            res.append((float(np.median(s[:, 0])), float(np.median(s[:, 1])), int(len(s))))
    return res


# ------------------------------------------------------------------ 真实: 扫描 ↔ 底片翻拍配准与影调斜率
def _edges(a):
    return np.hypot(gaussian_filter(a, 1.5, order=(0, 1)), gaussian_filter(a, 1.5, order=(1, 0)))


def register(scan, negD, scales=np.arange(0.30, 0.42, 0.004)):
    """scan (H, W) 码值, negD (h, w) 底片密度 (正像方向相同: 密度高 = 扫描亮)。在比例 s (底片像素 / 扫描像素) 上
    用边缘图的 FFT 互相关找平移, 取相关最大的 (s, dy, dx): 扫描 (y, x) ↔ 底片 (s·y + dy, s·x + dx)。"""
    best = None
    En = _edges(negD); En = (En - En.mean()) / En.std()
    for s in scales:
        sm = zoom(scan, s, order=1)
        Es = _edges(sm); Es = (Es - Es.mean()) / Es.std()
        H = En.shape[0] + Es.shape[0]; W = En.shape[1] + Es.shape[1]
        F = np.fft.rfft2(En, (H, W)) * np.conj(np.fft.rfft2(Es, (H, W)))
        c = np.fft.irfft2(F, (H, W)) / Es.size
        i = np.unravel_index(np.argmax(c), c.shape)
        if best is None or c[i] > best[0]:
            dy = i[0] if i[0] < H - Es.shape[0] else i[0] - H
            dx = i[1] if i[1] < W - Es.shape[1] else i[1] - W
            best = (float(c[i]), float(s), int(dy), int(dx))
    return best


def tone_curve(scan, negD, reg, step=8):
    """配准后逐 step 像素取 (D, 码值); 在 D 上按 0.05 分档取码值中位; 返回 (D 档, 码值, 斜率 dCode/dD)。"""
    _, s, dy, dx = reg
    sc = gaussian_filter(scan, 3.0)                     # 两图分辨率不同: 都平滑到底片翻拍的尺度
    nd = gaussian_filter(negD, 1.0)
    ys, xs = np.mgrid[0:scan.shape[0]:step, 0:scan.shape[1]:step]
    ny, nx = np.round(s * ys + dy).astype(int), np.round(s * xs + dx).astype(int)
    ok = (ny >= 0) & (ny < negD.shape[0]) & (nx >= 0) & (nx < negD.shape[1])
    D, C = nd[ny[ok], nx[ok]], sc[ys[ok], xs[ok]]
    bins = np.arange(np.percentile(D, 1), np.percentile(D, 99), 0.05)
    Dm, Cm = [], []
    for i in range(len(bins) - 1):
        m = (D >= bins[i]) & (D < bins[i + 1])
        if m.sum() >= 30:
            Dm.append(D[m].mean()); Cm.append(np.median(C[m]))
    Dm, Cm = np.array(Dm), np.array(Cm)
    # 单调平滑 (三次多项式) 后求导
    p = np.polyfit(Dm, Cm, 3)
    return Dm.tolist(), Cm.tolist(), np.polyval(np.polyder(p), Dm).tolist(), np.polyval(p, Dm).tolist()


def real(root, dst):
    from PIL import Image
    root = Path(root)
    out = {}
    for film, negdir in (("6_KODAK400TX", "TRIX-Neg Sequence"), ("7_ILFORDHP5PLUS", "Hp5-Neg Sequence")):
        for f in sorted((root / film).glob("*.jpg")):
            tag = f.name.split("-")[0]
            Y = np.asarray(Image.open(f).convert("L"), float)
            tiles = grain_tiles(Y)
            r = dict(n_tiles=len(tiles), tiles=[t[2:] for t in tiles],
                     s_px=by_level(tiles, 3), s_hp=by_level(tiles, 4), s48=by_level(tiles, 5))
            nf = root / film / negdir / f"{tag}.jpg"
            if nf.exists():
                N = np.asarray(Image.open(nf).convert("L"), float)
                base = np.median(N[5:15, N.shape[1] // 4: 3 * N.shape[1] // 4])
                negD = -np.log10(np.clip(srgb_lin(N), 1e-4, None) / srgb_lin(base))
                reg = register(Y, negD)
                Dm, Cm, slope, fitC = tone_curve(Y, negD, reg)
                r.update(neg_base_code=float(base), reg=reg, tone_D=Dm, tone_code=Cm, tone_fit=fitC, tone_slope=slope)
            out[f"{film}/{f.name}"] = r
            print(film, f.name, len(tiles), r.get("reg"), flush=True)
    json.dump(out, open(dst, "w"))


# ------------------------------------------------------------------ 模型
def model(rec, dst, n=384, evs=(-4, -3, -2, -1, 0, 1, 2, 3, 4), gen="t51_bw"):
    import warnings
    warnings.filterwarnings("ignore")
    from filmlab import generation, grain
    from filmlab.engine import Engine, Params
    from filmlab.recipes import recipe
    from filmlab.scanning.spatial import encode
    generation.use(gen)
    grain.SILVER_WORKERS = 1
    out = {}
    for ev in evs:
        res = {}
        for gr, size in ((1.0, n), (0.0, 32)):
            P = lambda e: Params(**{**recipe(rec), "timing": "gray", "roll": None, "frame": 0, "grain": gr,
                                    "scene_lights": None, "frame_width_mm": size * PX_UM / 1000.0})
            if gr > 0:
                eng = Engine(P(ev))
                neg = eng.develop_negative(np.full((size, size, 3), 0.18 * 2.0 ** ev))
                tr = {}
                o = eng.scan(neg, trace=tr)
                Y = encode(np.clip(o, 0, None)).mean(-1) * 255.0
                tiles = grain_tiles(Y, synthetic=True)
                Dn = tr["Dn"][..., 1]
                d48 = [np.sqrt(boxmean(detrend(Dn[y:y + T, x:x + T]), 2).var()) * (2 * PX_UM / 42.5)
                       for y in range(0, size - T + 1, T) for x in range(0, size - T + 1, T)]
                res.update(code=float(Y.mean()), Dn=float(Dn.mean()), s_px=float(np.median([t[3] for t in tiles])),
                           s_hp=float(np.median([t[4] for t in tiles])), s48=float(np.median([t[5] for t in tiles])),
                           s48_Dn=float(np.median(d48)))
            else:
                cs, ds = [], []
                for d in (-0.1, 0.1):
                    eng = Engine(P(ev))
                    neg = eng.develop_negative(np.full((size, size, 3), 0.18 * 2.0 ** (ev + d)))
                    tr = {}
                    o = eng.scan(neg, trace=tr)
                    cs.append(encode(np.clip(o, 0, None)).mean() * 255.0); ds.append(tr["Dn"][..., 1].mean())
                res.update(slope=float((cs[1] - cs[0]) / (ds[1] - ds[0])))
        out[f"{ev:+d}"] = res
        print(rec, ev, {k: round(v, 4) for k, v in res.items()}, flush=True)
    json.dump(out, open(dst, "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "real":
        real(sys.argv[2], sys.argv[3])
    else:
        model(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 384,
              gen=sys.argv[5] if len(sys.argv) > 5 else "t51_bw")
