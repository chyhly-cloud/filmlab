"""T38: 在 HDRI 上渲染"假想 RAW"(透视投影, 截断按通道), 用 rawio 同一套块划分与 rawio._block_obs 计算观测量, 并记录真实 flux。

假想相机: 36×24 mm, 焦距 35 / 50 mm, 偏航 0, 45, …, 315°, 俯仰 0 / +20°; 成像 1600×1067 像素 (22.5 µm/像素, 同 t19_model),
  每像素 3×3 子样本 (最近邻取 HDRI 8k 像素)。
传感器截断: 线性 Rec.709 三通道各自在 S 处截断 (白平衡后白色同时饱和); 子样本任一通道 > S 即为截断子样本。
  clip_frac = 截断子样本比例; 观测图 = 截断后的子样本平均 (RAW 看到的值); 真实辐亮度只用于 flux。
S: 先取 S0 = 帧内真实亮度 Y 的第 90 百分位 × 2^3.71 (T37 的锚, 我们 RAW 的灯具块 base 中位 / P90), 再按本帧灯具块 base 中位 / S0
   修正一次, 使灯具块 base 中位 = P90 × 2^3.71 (与 RAW 中锚的定义一致)。
每块 (rawio 同一划分: clip_frac > 0.25 膨胀一圈连通; 同一分类; base = 块内观测 Y 的第 90 百分位):
  flux  = log2(Σ f·L / (Σ f·base)), L = 像素内截断子样本的真实 Y 平均 (= rawio STOPS_LOG 的 flux)
  peak  = log2(块内截断子样本真实 Y 最大值 / base)
  观测量 = rawio._block_obs(观测 Y, lab, k, sl, clip_frac, base): r, bg, skirt, slope, fmean, compact
  另记 (只作参考, RAW 里没有同一定义): nsat3 = 三通道都截断的截断子样本比例; 边缘颜色 log2(R/G)、log2(B/G) (块外 1–3 像素)。
用法: python3 -I t38_hdri_obs.py 名称 URL RGB文件 输出.tsv   (RGB 文件由 t37_hdr2y.c 的 rgb 模式生成)"""
import sys
from pathlib import Path
import numpy as np
from scipy import ndimage
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from filmlab import rawio

LUMA = rawio.LUMA
WI, HI, PITCH = 1600, 1067, 0.0225          # 像素, mm
SUB = 3
YAWS = np.arange(0, 360, 45)
PITCHES = (0.0, 20.0)
FOCALS = (35.0, 50.0)
A90 = 3.71


def render(img, yaw, pitch, focal):
    H, W, _ = img.shape
    ps, th = np.deg2rad(yaw), np.deg2rad(pitch)
    fwd = np.array([np.cos(th) * np.sin(ps), np.sin(th), np.cos(th) * np.cos(ps)])
    right = np.array([np.cos(ps), 0.0, -np.sin(ps)])
    up = np.cross(fwd, right) * -1.0
    if up[1] < 0:
        up = -up
    off = (np.arange(SUB) + 0.5) / SUB - 0.5
    xs = ((np.arange(WI)[:, None] + 0.5 + off[None, :]) - WI / 2).ravel() * PITCH          # (WI*SUB,)
    ys = (HI / 2 - (np.arange(HI)[:, None] + 0.5 + off[None, :])).ravel() * PITCH          # 行向下 → y 向上
    xs, ys = xs.astype(np.float32), ys.astype(np.float32)
    vx = xs[None, :] * np.float32(right[0]) + ys[:, None] * np.float32(up[0]) + np.float32(focal * fwd[0])   # (HI*SUB, WI*SUB)
    vy = ys[:, None] * np.float32(up[1]) + np.float32(focal * fwd[1]) + 0 * xs[None, :]
    vz = xs[None, :] * np.float32(right[2]) + ys[:, None] * np.float32(up[2]) + np.float32(focal * fwd[2])
    lat = np.arctan2(vy, np.hypot(vx, vz)); lon = np.arctan2(vx, vz)
    row = np.clip(((np.pi / 2 - lat) / np.pi * H).astype(np.int64), 0, H - 1)
    col = np.clip(((lon + np.pi) / (2 * np.pi) * W).astype(np.int64), 0, W - 1)
    return img[row, col]                                                                       # (HI*SUB, WI*SUB, 3)


def sensor(rs, S):
    """子样本 → 每像素: 观测 RGB、clip_frac、截断子样本真实 Y 之和、最大值、三通道全截断数。"""
    clipped = (rs > S).any(-1)
    obs = np.minimum(rs, S)
    Yt = rs @ LUMA
    blk = lambda a: a.reshape(HI, SUB, WI, SUB, *a.shape[2:])
    rgb_obs = blk(obs).mean((1, 3))
    f = blk(clipped.astype(np.float32)).mean((1, 3))
    Lsum = blk(np.where(clipped, Yt, 0.0)).sum((1, 3))
    Lmax = blk(np.where(clipped, Yt, 0.0)).max((1, 3))
    n3 = blk(((rs > S).all(-1)).astype(np.float32)).sum((1, 3))
    return rgb_obs, f, Lsum, Lmax, n3


def blocks(Y, cf):
    """rawio.reconstruct_highlights 同一块划分与分类; 逐块产出 (k, sl, 窗口, blob, area, cls, base)。"""
    mask = ndimage.binary_dilation(cf > 0.25)
    lab, n = ndimage.label(mask)
    total = mask.size
    for k, sl in enumerate(ndimage.find_objects(lab), start=1):
        pad = max(4, int(0.6 * np.sqrt((sl[0].stop - sl[0].start) * (sl[1].stop - sl[1].start))))
        y0, y1 = max(sl[0].start - pad, 0), min(sl[0].stop + pad, Y.shape[0])
        x0, x1 = max(sl[1].start - pad, 0), min(sl[1].stop + pad, Y.shape[1])
        blob = lab[y0:y1, x0:x1] == k
        area = int(blob.sum())
        base = float(np.percentile(Y[y0:y1, x0:x1][blob], 90))
        cls = "large" if area / total >= rawio.HIGHLIGHT_PRIOR[0][0] else "lamp"
        if rawio._line_like(cf[y0:y1, x0:x1] * blob > 0.25):
            cls = "line"
        yield lab, k, sl, (slice(y0, y1), slice(x0, x1)), blob, area, cls, base


def main(name, url, path, dst):
    H, W = np.fromfile(path, dtype=np.int32, count=2)
    img = np.fromfile(path, dtype=np.float32, offset=8).reshape(H, W, 3)
    fo = open(dst, "w")
    fo.write("hdri\turl\tfocal\tyaw\tpitch\tS_over_P90\tblob\tarea\tcls\tr_um\tbase_rel\tflux\tpeak\tbg\tskirt\tslope\tfmean\tcompact\t"
             "nsat3\tcol_rg\tcol_bg\tedge\n")
    for focal in FOCALS:
        for pitch in PITCHES:
            for yaw in YAWS:
                rs = render(img, yaw, pitch, focal)
                Yt_pix = (rs @ LUMA).reshape(HI, SUB, WI, SUB).mean((1, 3))
                S0 = float(np.percentile(Yt_pix, 90)) * 2.0 ** A90
                # 一次修正: 灯具块 base 中位 = P90 × 2^3.71
                rgb_obs, cf, _, _, _ = sensor(rs, S0)
                Y = rgb_obs @ LUMA
                bases = [b[-1] for b in blocks(Y, cf) if b[6] == "lamp"]
                m = float(np.clip(np.median(bases) / S0, 0.5, 1.0)) if bases else 1.0
                S = S0 / m
                rgb_obs, cf, Lsum, Lmax, n3 = sensor(rs, S)
                Y = rgb_obs @ LUMA
                nclip = cf * SUB * SUB
                for lab, k, sl, win, blob, area, cls, base in blocks(Y, cf):
                    fb = np.clip(cf[win][blob], 0.0, 1.0)
                    nc = nclip[win][blob]
                    L = np.where(nc > 0, Lsum[win][blob] / np.maximum(nc, 1), 0.0)
                    flux = float(np.log2(max((fb * L).sum(), 1e-30) / (fb.sum() * base)))
                    peak = float(np.log2(max(Lmax[win][blob].max(), 1e-30) / base))
                    ob = rawio._block_obs(Y, lab, k, sl, cf, base)
                    nsat3 = float(n3[win][blob].sum() / max(nc.sum(), 1))
                    d = ndimage.distance_transform_edt(~blob)
                    ring = (d >= 1) & (d <= 3) & (lab[win] == 0)
                    if ring.any():
                        c = rgb_obs[win][ring].mean(0)
                        crg, cbg = (float(np.log2(max(c[0], 1e-9) / max(c[1], 1e-9))), float(np.log2(max(c[2], 1e-9) / max(c[1], 1e-9))))
                    else:
                        crg = cbg = float("nan")
                    edge = int(sl[0].start == 0 or sl[1].start == 0 or sl[0].stop == HI or sl[1].stop == WI)
                    fo.write("\t".join(map(str, [name, url, int(focal), yaw, pitch, f"{S / (S0 * 2 ** -A90):.4g}", k, area, cls,
                             f"{ob['r'] * PITCH * 1000:.1f}", f"{base / S:.3f}", f"{flux:.3f}", f"{peak:.3f}", f"{ob['bg']:.3f}",
                             f"{ob['skirt']:.3f}", f"{ob['slope']:.3f}", f"{ob['fmean']:.3f}", f"{ob['compact']:.3f}", f"{nsat3:.3f}",
                             f"{crg:.3f}", f"{cbg:.3f}", edge])) + "\n")
                fo.flush()
    fo.close()


if __name__ == "__main__":
    main(*sys.argv[1:5])
