"""T54: EPFL Frontier 扫描的影调斜率 dCode/dlogE (只读图像, 只存测量; docs/tasks/T54_COLOUR_GRAIN_LEVEL.md 第 4 节)。

每帧有同一场景的数码 RAW (Panasonic G81, 线性): RAW 每通道的线性值就是场景在该通道的相对曝光 (同一帧内, 相差一个常数)。
扫描与 RAW 配准 (缩小后 ORB 特征 + RANSAC 单应, skimage), 在扫描网格上取 (log10 RAW_c, 扫描码值_c), 都先平滑到约 100 µm
(两台相机的分辨率与镜头不同, 只取宏观影调); 按 log10 RAW_c 分档取码值中位 → 三次多项式 → dCode_c/dlogE。
同一帧内的扫描与 RAW 都是同一次拍摄 / 同一次冲印店定光, 所以这是该帧的影调曲线 (不受逐帧定光影响)。
RAW 截断 (≥ 0.98 白电平) 与暗噪声 (< 黑电平 + 4 σ) 处不用。
用法: python3 -I t54_epfl_tone.py 输出.json 扫描.tif RAW ...   (成对)
      python3 -I t54_epfl_tone.py tiles 输出.json 真实颗粒.json 扫描.tif RAW ...   (颗粒块的场景曝光)
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates, zoom

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def load_raw(path):
    import rawpy
    with rawpy.imread(str(path)) as r:
        img = r.postprocess(gamma=(1, 1), no_auto_bright=True, output_bps=16, half_size=True, use_camera_wb=False,
                            use_auto_wb=False, user_wb=[1.0, 1.0, 1.0, 1.0], output_color=rawpy.ColorSpace.raw,
                            user_flip=0)
        white = float(r.white_level); black = float(np.mean(r.black_level_per_channel))
    lin = img.astype(float) / 65535.0
    sat = img.max(-1) >= 0.98 * 65535 * (white - black) / white if white > 0 else img.max(-1) >= 65000
    return lin, sat


def _gray(a):
    g = a.mean(-1) if a.ndim == 3 else a
    g = np.log10(np.clip(g, 1e-5, None)) if g.max() <= 1.5 else g
    g = (g - np.percentile(g, 1)) / max(np.percentile(g, 99) - np.percentile(g, 1), 1e-6)
    return np.clip(g, 0, 1)


def register(scan_small, raw_small):
    """ORB + RANSAC 单应: raw_small 坐标 → scan_small 坐标。"""
    from skimage.feature import ORB, match_descriptors
    from skimage.measure import ransac
    from skimage.transform import ProjectiveTransform
    kp = []
    for im in (scan_small, raw_small):
        o = ORB(n_keypoints=3000, fast_threshold=0.03)
        o.detect_and_extract(im)
        kp.append((o.keypoints, o.descriptors))
    m = match_descriptors(kp[0][1], kp[1][1], cross_check=True, max_ratio=0.85)
    src = kp[1][0][m[:, 1]][:, ::-1]; dst = kp[0][0][m[:, 0]][:, ::-1]      # (x, y)
    model, inl = ransac((src, dst), ProjectiveTransform, min_samples=4, residual_threshold=2.5, max_trials=5000)
    return model, int(inl.sum()), int(len(m))


def tone(scan, raw, sat, f=8, step=6):
    """scan (H, W, 3) 码值, raw (h, w, 3) 线性。返回配准信息与每通道 (logE 档, 码值, 多项式)。"""
    s_small = _gray(zoom(scan.mean(-1), 1 / f, order=1))
    k = scan.shape[1] / f / raw.shape[1]
    r_small = _gray(zoom(raw.mean(-1), k, order=1))
    model, ninl, nm = register(s_small, r_small)
    # 扫描网格 (每 step 个小图像素) → RAW 坐标
    ys, xs = np.mgrid[0:s_small.shape[0]:1, 0:s_small.shape[1]:1]
    pts = np.stack([xs.ravel(), ys.ravel()], 1).astype(float)
    inv = model.inverse(pts)                       # 扫描小图 → RAW 小图 (x, y)
    rx, ry = inv[:, 0] / k, inv[:, 1] / k           # RAW 原坐标
    ok = (rx > 5) & (rx < raw.shape[1] - 5) & (ry > 5) & (ry < raw.shape[0] - 5)
    sm = gaussian_filter(sat.astype(float), 4.0)
    out = dict(inliers=ninl, matches=nm, H=model.params.tolist(), channels=[])
    for c in range(3):
        S = zoom(gaussian_filter(scan[..., c], f / 2.0), 1 / f, order=1)[:s_small.shape[0], :s_small.shape[1]].ravel()
        R = gaussian_filter(raw[..., c], f / 2.0 / (scan.shape[1] / raw.shape[1]))
        Rv = map_coordinates(R, [ry[ok], rx[ok]], order=1)
        Sv = S[ok]
        bad = map_coordinates(sm, [ry[ok], rx[ok]], order=1) > 0.01
        m = (~bad) & (Rv > 1e-4) & (Sv > 3) & (Sv < 252)
        L = np.log10(Rv[m]); C = Sv[m]
        bins = np.arange(np.percentile(L, 0.5), np.percentile(L, 99.5), 0.05)
        Lm, Cm, n = [], [], []
        for i in range(len(bins) - 1):
            mm = (L >= bins[i]) & (L < bins[i + 1])
            if mm.sum() >= 40:
                Lm.append(float(np.median(L[mm]))); Cm.append(float(np.median(C[mm]))); n.append(int(mm.sum()))
        p = np.polyfit(Lm, Cm, 3, w=np.sqrt(n))
        # 配准质量: 档内码值的 MAD
        out["channels"].append(dict(logE=Lm, code=Cm, n=n, poly=p.tolist()))
    return out


def tiles(dst, real_json, pairs):
    """颗粒块 (t54_real epfl 的每块 y, x) 在 RAW 上的场景曝光: RAW 用 rawio.load_raw (渲染所用的同一线性 sRGB, 1.0 = 截断,
    日光白平衡, 不重建高光、不降噪), 配准同上; 每块取其在 RAW 上覆盖区域的平均 → 胶片曝光档 (会话测光
    rawio.SESSION_LOG_SCALE: 胶片输入 = RAW × 10^0.613, 18% 灰 = 0 档; 与 t54_model uniform 的 0.18 × 2^EV 同一轴)。"""
    import tifffile
    from filmlab import rawio
    real = json.load(open(real_json))
    out = json.load(open(dst)) if Path(dst).exists() else {}
    for sp, rp in zip(pairs[0::2], pairs[1::2]):
        key = Path(sp).stem.split("raphael-")[-1]
        if key in out or key not in real:
            continue
        scan = tifffile.imread(sp).astype(float)[..., :3]
        rgb = rawio.load_raw(rp, max_size=2400, highlights="clip", denoise_raw=False)
        if (scan.shape[0] > scan.shape[1]) != (rgb.shape[0] > rgb.shape[1]):
            rgb = np.rot90(rgb)
        f = 8
        s_small = _gray(zoom(scan.mean(-1), 1 / f, order=1))
        k = scan.shape[1] / f / rgb.shape[1]
        r_small = _gray(zoom(rgb.mean(-1), k, order=1))
        try:
            model, ninl, nm = register(s_small, r_small)
        except ValueError as exc:
            out[key] = dict(error=str(exc)); json.dump(out, open(dst, "w")); continue
        R = np.array(real[key]["rows"])
        c = np.stack([(R[:, 1] + 16) / f, (R[:, 0] + 16) / f], 1)        # 扫描小图 (x, y)
        inv = model.inverse(c)
        rx, ry = inv[:, 0] / k, inv[:, 1] / k
        sm = gaussian_filter(rgb, (16 / f / k / 2, 16 / f / k / 2, 0))
        ok = (rx > 2) & (rx < rgb.shape[1] - 2) & (ry > 2) & (ry < rgb.shape[0] - 2)
        vals = np.full((len(R), 3), np.nan)
        for ch in range(3):
            vals[ok, ch] = map_coordinates(sm[..., ch], [ry[ok], rx[ok]], order=1)
        Y = vals @ np.array([0.2126, 0.7152, 0.0722])
        ev = np.log2(np.clip(Y, 1e-6, None) * 10.0 ** rawio.SESSION_LOG_SCALE / 0.18)
        out[key] = dict(inliers=ninl, matches=nm, raw_rgb=np.round(vals, 6).tolist(), ev=np.round(ev, 3).tolist())
        print(key, "inliers", ninl, "/", nm, "EV 分位 (5/50/95)", np.round(np.nanpercentile(ev, [5, 50, 95]), 2), flush=True)
        json.dump(out, open(dst, "w"))


def main():
    import tifffile
    if sys.argv[1] == "tiles":
        tiles(sys.argv[2], sys.argv[3], sys.argv[4:])
        return
    dst = sys.argv[1]
    out = json.load(open(dst)) if Path(dst).exists() else {}
    args = sys.argv[2:]
    for sp, rp in zip(args[0::2], args[1::2]):
        key = Path(sp).stem.split("raphael-")[-1]
        if key in out:
            continue
        scan = tifffile.imread(sp).astype(float)[..., :3]
        raw, sat = load_raw(rp)
        if (scan.shape[0] > scan.shape[1]) != (raw.shape[0] > raw.shape[1]):
            raw = np.rot90(raw); sat = np.rot90(sat)
        try:
            r = tone(scan, raw, sat)
        except ValueError as exc:                    # 无可配准的特征 (例: 均匀白场)
            print(key, "配准失败:", exc, flush=True)
            out[key] = dict(error=str(exc)); json.dump(out, open(dst, "w"))
            continue
        out[key] = r
        for c, ch in zip("RGB", r["channels"]):
            sl = np.polyval(np.polyder(np.array(ch["poly"])), np.array(ch["logE"]))
            print(key, c, "inliers", r["inliers"], "/", r["matches"], "code", np.round(ch["code"][::4], 0),
                  "slope", np.round(sl[::4], 0), flush=True)
        json.dump(out, open(dst, "w"))


if __name__ == "__main__":
    main()
