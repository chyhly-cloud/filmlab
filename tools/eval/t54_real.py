"""T54: 真实冲印店扫描的彩色颗粒量级与影调斜率 (只读图像, 只存测量; docs/tasks/T54_COLOUR_GRAIN_LEVEL.md)。

颗粒 (grain_stats; 真实与模型同一代码): 8 位码值的 R、G、B 与 BT.601 亮度 Y (编码域)。均匀块 32 × 32 像素, 去一次平面,
  T45 平坦判据 (2b 盒平均方差 ≤ 1.6 × b 盒方差 / 4, b = 每 24 µm 的像素数, 至少 2) 对 Y 与三个通道都要通过; 码值 12–250。
  每块每通道: 平均码值, σ_px (去平面后的标准差), σ_hp (减去 σ = 1 像素高斯), σ_hp25 (减去 σ = 2.5 像素高斯; T51 的亮度量),
  σ_48 (约 48 µm 盒平均的标准差, 按面积律换到 42.5 µm 等面积方孔), 48 µm 盒级通道相关 (R–G、G–B、R–B)。8 位量化方差 1/12 已扣。
RPL (Noritsu, 645, 20 µm/像素; rpl 子命令):
  · 每帧颗粒。
  · 影调斜率 (T51b 的方法, 逐通道): 扫描与同一底片的灯箱翻拍 (Neg Sequence, Nikon D5300) 配准 (亮度边缘 FFT 互相关,
    比例网格); 翻拍按 sRGB 解码当作线性, 每通道底片密度 D_c = −log10(T_c / T_片基,c), 片基 (含橙色遮罩) 取片边未曝光带,
    逐通道 → 每帧每通道的码值(D_c) 与 dCode_c/dD_c。
  · 翻拍密度随曝光 (γ_c = dD_c/dEV): 相邻包围档的翻拍互相配准 (同一场景), 在 D_c 上分档取差的中位。
    于是 dCode_c/dEV = dCode_c/dD_c × dD_c/dEV, 翻拍密度的单位 (相机的光谱与影调) 在乘积里抵消。
EPFL (Frontier, 6.6 µm/像素; epfl 子命令): 每帧颗粒 (噪声卡、白场、楔、夜景); 影调斜率见 t54_epfl_tone.py。
用法: python3 -I t54_real.py rpl RPL目录 输出.json
      python3 -I t54_real.py epfl 输出.json 帧.tif ...
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

LUMA = np.array([0.299, 0.587, 0.114])
T = 32


def srgb_lin(c):
    c = np.asarray(c, float) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def detrend(t):
    h, w = t.shape[:2]
    yy, xx = np.mgrid[:h, :w]
    A = np.stack([np.ones(h * w), yy.ravel(), xx.ravel()], 1)
    c, *_ = np.linalg.lstsq(A, t.reshape(h * w, -1), rcond=None)
    return t - (A @ c).reshape(t.shape)


def boxmean(a, b):
    h, w = (a.shape[0] // b) * b, (a.shape[1] // b) * b
    return a[:h, :w].reshape(h // b, b, w // b, b, *a.shape[2:]).mean((1, 3))


def grain_stats(img, px_um, lo=12, hi=250, flat=1.6, synthetic=False, step=None):
    """img: (H, W, 3) 码值 (float 0–255)。返回每块一行:
    [y, x, 均值 R G B Y, σ_px ×4, σ_hp ×4, σ_hp25 ×4, σ_48 ×4, ρ48 RG GB RB] (32 列)。"""
    b = max(2, int(round(24.0 / px_um)))
    b48 = max(1, int(round(48.0 / px_um)))
    a48 = 42.5 / (b48 * px_um)
    q = 0.0 if synthetic else 1.0 / 12.0
    Yall = img @ LUMA
    X = np.concatenate([img, Yall[..., None]], -1)
    H, W = img.shape[:2]
    out = []
    st = step or T
    for y in range(0, H - T + 1, st):
        for x in range(0, W - T + 1, st):
            t = X[y:y + T, x:x + T]
            if not synthetic and (t[..., :3].min() < lo or t[..., :3].max() > hi):
                continue
            r = detrend(t)
            if not synthetic:
                v1 = boxmean(r, b).var((0, 1)); v2 = boxmean(r, 2 * b).var((0, 1))
                if np.any(v2 > flat * v1 / 4.0):
                    continue
            qv = np.array([q, q, q, q * (LUMA ** 2).sum()])
            s_px = np.sqrt(np.clip(r.var((0, 1)) - qv, 0, None))
            hp = (r - gaussian_filter(r, (1.0, 1.0, 0), mode="reflect"))[2:-2, 2:-2]
            hp25 = (r - gaussian_filter(r, (2.5, 2.5, 0), mode="reflect"))[4:-4, 4:-4]
            bm = boxmean(r, b48) if b48 > 1 else r
            s48 = np.sqrt(np.clip(bm.var((0, 1)) - qv / b48 ** 2, 0, None)) * a48
            C = np.cov(bm[..., :3].reshape(-1, 3).T)
            sd = np.sqrt(np.clip(np.diag(C) - q / b48 ** 2, 1e-12, None))
            rho = [C[0, 1] / sd[0] / sd[1], C[1, 2] / sd[1] / sd[2], C[0, 2] / sd[0] / sd[2]]
            out.append([y, x, *t.mean((0, 1)), *s_px, *hp.std((0, 1)), *hp25.std((0, 1)), *s48, *rho])
    return np.array(out)


COLS = {"mean": 2, "px": 6, "hp": 10, "hp25": 14, "s48": 18, "rho": 22}


def by_level(rows, key, ch, edges=np.arange(0, 256, 20), level_ch=None, min_n=5):
    """按 level_ch 通道 (默认同 ch) 的平均码值分档, 取 key 的中位。返回 [(档中位码值, 中位, 块数)]。"""
    lc = COLS["mean"] + (ch if level_ch is None else level_ch)
    k = COLS[key] + ch
    res = []
    for i in range(len(edges) - 1):
        s = rows[(rows[:, lc] >= edges[i]) & (rows[:, lc] < edges[i + 1])]
        if len(s) >= min_n:
            res.append((float(np.median(s[:, lc])), float(np.median(s[:, k])), int(len(s))))
    return res


# ------------------------------------------------------------------ RPL: 配准与影调
def _edges(a):
    return np.hypot(gaussian_filter(a, 1.5, order=(0, 1)), gaussian_filter(a, 1.5, order=(1, 0)))


def register(src, dst, scales):
    """src 映射到 dst: dst(s·y + dy, s·x + dx) ≈ src(y, x)。边缘图 FFT 互相关, 比例网格。返回 (相关, s, dy, dx)。"""
    from scipy.ndimage import zoom
    best = None
    En = _edges(dst); En = (En - En.mean()) / En.std()
    for s in scales:
        sm = zoom(src, s, order=1)
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


def neg_density(N):
    """翻拍 (h, w, 3) 码值 → 每通道底片密度 (片基 = 0) 与片基码值。片基: 上沿片边带 (行 3–20) 中较亮的一半的中位
    (片边印字是曝光的, 较暗)。"""
    band = N[3:20].reshape(-1, 3)
    Yb = band @ LUMA
    base = np.median(band[Yb >= np.median(Yb)], 0)
    D = -np.log10(np.clip(srgb_lin(N), 1e-4, None) / srgb_lin(base)[None, None, :])
    return D, base


def tone_curves(scan, negD, reg, step=8):
    """配准后每通道 (D_c, 码值_c) 分档 → 三次多项式; 返回每通道 dict(D, code, fit 系数)。"""
    _, s, dy, dx = reg
    ys, xs = np.mgrid[0:scan.shape[0]:step, 0:scan.shape[1]:step]
    ny, nx = np.round(s * ys + dy).astype(int), np.round(s * xs + dx).astype(int)
    ok = (ny >= 0) & (ny < negD.shape[0]) & (nx >= 0) & (nx < negD.shape[1])
    out = []
    for c in range(3):
        sc = gaussian_filter(scan[..., c], 3.0)
        nd = gaussian_filter(negD[..., c], 1.0)
        D, C = nd[ny[ok], nx[ok]], sc[ys[ok], xs[ok]]
        bins = np.arange(np.percentile(D, 1), np.percentile(D, 99), 0.04)
        Dm, Cm, n = [], [], []
        for i in range(len(bins) - 1):
            m = (D >= bins[i]) & (D < bins[i + 1])
            if m.sum() >= 30:
                Dm.append(float(D[m].mean())); Cm.append(float(np.median(C[m]))); n.append(int(m.sum()))
        p = np.polyfit(Dm, Cm, 3, w=np.sqrt(n))
        out.append(dict(D=Dm, code=Cm, n=n, poly=p.tolist()))
    return out


def gamma_pairs(negs, evs, step=4):
    """相邻包围档的翻拍互相配准 → 每通道 dD_c/dEV (按较低档的 D_c 分档的中位差 / 档差)。
    negs: [(D (h, w, 3))], evs: 每帧的 EV。返回 [(ev_lo, ev_hi, reg, 每通道 [(D, dD/dEV, n)])]。"""
    out = []
    order = np.argsort(evs)
    for i, j in zip(order[:-1], order[1:]):
        A, B = negs[i], negs[j]
        reg = register(A[..., 1], B[..., 1], np.arange(0.97, 1.031, 0.005))
        _, s, dy, dx = reg
        ys, xs = np.mgrid[20:A.shape[0] - 20:step, 20:A.shape[1] - 20:step]
        ny, nx = np.round(s * ys + dy).astype(int), np.round(s * xs + dx).astype(int)
        ok = (ny >= 20) & (ny < B.shape[0] - 20) & (nx >= 20) & (nx < B.shape[1] - 20)
        per = []
        for c in range(3):
            a = gaussian_filter(A[..., c], 2.0)[ys[ok], xs[ok]]
            bb = gaussian_filter(B[..., c], 2.0)[ny[ok], nx[ok]]
            m0 = a > 0.03                       # 只用画面 (片边与未曝光处 D ≈ 0)
            a, bb = a[m0], bb[m0]
            bins = np.arange(np.percentile(a, 2), np.percentile(a, 98), 0.05)
            rows = []
            for k in range(len(bins) - 1):
                m = (a >= bins[k]) & (a < bins[k + 1])
                if m.sum() >= 50:
                    rows.append((float(np.median(0.5 * (a[m] + bb[m]))), float(np.median(bb[m] - a[m]) / (evs[j] - evs[i])),
                                 int(m.sum())))
            per.append(rows)
        out.append((float(evs[i]), float(evs[j]), reg, per))
    return out


FILMS = {"2_KODAKPORTRA400": "Portra400", "1_FUJIPRO400H": "Fuji400", "3_KODAKPORTRA800": "Portra800",
         "4_KODAKPORTRA160": "Portra160", "5_KODAKEKTAR100": "Ektar100"}
EV = {"MINUS3": -3, "MINUS2": -2, "MINUS1": -1, "NORMAL": 0, "PLUS1": 1, "PLUS2": 2, "PLUS3": 3}


def rpl(root, dst, films=None):
    from PIL import Image
    root = Path(root)
    out = json.load(open(dst)) if Path(dst).exists() else {}
    for film, negname in FILMS.items():
        if films and film not in films:
            continue
        if film in out:
            continue
        fo = {"frames": {}}
        negs, evs = [], []
        for f in sorted((root / film).glob("*.jpg")):
            tag = f.name.split("-")[0]
            img = np.asarray(Image.open(f).convert("RGB"), float)
            rows = grain_stats(img, 20.0)
            r = dict(file=f.name, ev=EV[tag], n_tiles=len(rows), rows=np.round(rows, 4).tolist())
            nf = root / film / f"{negname}-Neg Sequence" / f"{tag}.jpg"
            if nf.exists():
                N = np.asarray(Image.open(nf).convert("RGB"), float)
                D, base = neg_density(N)
                reg = register(img @ LUMA, D @ LUMA, np.arange(0.30, 0.42, 0.004))
                r.update(neg_base_code=base.tolist(), reg=reg, tone=tone_curves(img, D, reg))
                negs.append(D); evs.append(EV[tag])
            fo["frames"][tag] = r
            print(film, tag, len(rows), r.get("reg"), flush=True)
        if len(negs) >= 2:
            fo["gamma"] = gamma_pairs(negs, np.array(evs, float))
            for g in fo["gamma"]:
                print(film, "γ", g[0], g[1], np.round(g[2], 3),
                      [np.round(np.median([x[1] for x in p]), 3) if p else None for p in g[3]], flush=True)
        out[film] = fo
        json.dump(out, open(dst, "w"))


def epfl(dst, frames):
    import tifffile
    out = json.load(open(dst)) if Path(dst).exists() else {}
    for f in frames:
        key = Path(f).stem.split("raphael-")[-1]
        if key in out:
            continue
        img = tifffile.imread(f).astype(float)[..., :3]
        rows = grain_stats(img, 6.6)
        out[key] = dict(file=Path(f).name, n_tiles=len(rows), rows=np.round(rows, 4).tolist())
        print(key, img.shape, len(rows), flush=True)
        json.dump(out, open(dst, "w"))


if __name__ == "__main__":
    if sys.argv[1] == "rpl":
        rpl(sys.argv[2], sys.argv[3], sys.argv[4:] or None)
    elif sys.argv[1] == "epfl":
        epfl(sys.argv[2], sys.argv[3:])
