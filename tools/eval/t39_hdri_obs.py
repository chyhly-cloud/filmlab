"""T39: 同 t38_hdri_obs.py, 但传感器截断在相机原始通道 (白平衡前) 上判定, 并记录逐通道饱和的观测量 nsat3 / nch。

相机通道: cam = M_cam · XYZ, XYZ = M_709→XYZ · (HDRI 线性 Rec.709); M_cam 取我们夜景 RAW 的 LibRaw rgb_xyz_matrix
  (主: n29240 的 Nikon, D65 白在相机通道为 0.450 : 1 : 0.798; 对照: n33920 的 Sony 0.345 : 1 : 0.742、n30464 的 Fuji 0.436 : 1 : 0.620)。
  三个相机通道在同一满阱电平 S 处截断。观测图 = 截断后的相机通道换回 XYZ 再到 Rec.709 (LibRaw 的 XYZ 输出是同一条线性路径)。
  S 的锚: D65 白在 Y = P90 × 2^3.71 时恰好饱和 (最高通道), 再按本帧灯具块 base 中位修正一次 (同 T38)。
  对照矩阵: 块按主矩阵划分, 在同一批块上按对照矩阵 (同样以 D65 白定饱和电平) 重算 nsat3 / nch, 报告分档迁移。

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
用法: python3 -I t39_hdri_obs.py 名称 URL RGB文件 输出.tsv   (RGB 文件由 t37_hdr2y.c 的 rgb 模式生成)"""
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
M709 = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]])
CAMS = {   # LibRaw rgb_xyz_matrix (XYZ → 相机), 来自我们的 RAW
    "nikon_n29240": [[1.0405, -0.3755, -0.1270], [-0.5461, 1.3787, 0.1793], [-0.1040, 0.2015, 0.6785]],
    "sony_n33920": [[0.5991, -0.1456, -0.0455], [-0.4764, 1.2135, 0.2980], [-0.0707, 0.1425, 0.6701]],
    "fuji_n30464": [[1.1434, -0.4948, -0.1210], [-0.3746, 1.2042, 0.1903], [-0.0666, 0.1479, 0.5235]],
}
D65 = np.array([0.95047, 1.0, 1.08883])


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


def sensor(rs, S, cam):
    """子样本 (Rec.709) → 相机通道截断 → 每像素: 观测 Rec.709、clip_frac、截断子样本真实 Y 之和与最大值、
    三通道全饱和比例、逐通道饱和比例。"""
    M = np.asarray(cam) @ M709
    c = rs @ M.T.astype(np.float32)
    sat = c > S
    clipped = sat.any(-1)
    obs = np.minimum(c, S) @ np.linalg.inv(M).T.astype(np.float32)
    Yt = rs @ LUMA
    blk = lambda a: a.reshape(HI, SUB, WI, SUB, *a.shape[2:])
    rgb_obs = blk(obs).mean((1, 3))
    f = blk(clipped.astype(np.float32)).mean((1, 3))
    Lsum = blk(np.where(clipped, Yt, 0.0)).sum((1, 3))
    Lmax = blk(np.where(clipped, Yt, 0.0)).max((1, 3))
    f_all = blk(sat.all(-1).astype(np.float32)).mean((1, 3))
    f_ch = blk(sat.astype(np.float32)).mean((1, 3))
    return rgb_obs, f, Lsum, Lmax, f_all, f_ch


def white_sat(cam):
    """D65 白 (Y = 1) 在相机通道里的最高值: S = 这个值 × 欲饱和的 Y。"""
    return float((np.asarray(cam) @ D65).max())


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
    names = list(CAMS); main_cam = names[0]
    fo = open(dst, "w")
    fo.write("hdri\tfocal\tyaw\tpitch\tS_over_P90\tblob\tarea\tcls\tr_um\tbase_rel\tflux\tpeak\tbg\tskirt\tslope\tfmean\tcompact\t"
             "nsat3\tnch\t" + "\t".join(f"nsat3_{n}\tnch_{n}" for n in names[1:]) + "\tcol_rg\tcol_bg\tedge\n")
    for focal in FOCALS:
        for pitch in PITCHES:
            for yaw in YAWS:
                rs = render(img, yaw, pitch, focal)
                Yt_pix = (rs @ LUMA).reshape(HI, SUB, WI, SUB).mean((1, 3))
                P90 = float(np.percentile(Yt_pix, 90))
                S0 = P90 * 2.0 ** A90 * white_sat(CAMS[main_cam])
                rgb_obs, cf, *_ = sensor(rs, S0, CAMS[main_cam])
                Y = rgb_obs @ LUMA
                bases = [b[-1] for b in blocks(Y, cf) if b[6] == "lamp"]
                m = float(np.clip(np.median(bases) / (P90 * 2.0 ** A90), 0.25, 1.0)) if bases else 1.0
                S = S0 / m
                rgb_obs, cf, Lsum, Lmax, f_all, f_ch = sensor(rs, S, CAMS[main_cam])
                Y = rgb_obs @ LUMA
                alt = {}
                for n in names[1:]:
                    _, fa, _, _, fl, fc = sensor(rs, S / white_sat(CAMS[main_cam]) * white_sat(CAMS[n]), CAMS[n])
                    alt[n] = (fa, fl, fc)
                nclip = cf * SUB * SUB
                for lab, k, sl, win, blob, area, cls, base in blocks(Y, cf):
                    fb = np.clip(cf[win][blob], 0.0, 1.0)
                    nc = nclip[win][blob]
                    L = np.where(nc > 0, Lsum[win][blob] / np.maximum(nc, 1), 0.0)
                    flux = float(np.log2(max((fb * L).sum(), 1e-30) / (fb.sum() * base)))
                    peak = float(np.log2(max(Lmax[win][blob].max(), 1e-30) / base))
                    ob = rawio._block_obs(Y, lab, k, sl, cf, base, sat=(cf, f_all, f_ch.sum(-1)))
                    oa = {n: rawio._block_obs(Y, lab, k, sl, cf, base, sat=(a[0], a[1], a[2].sum(-1))) for n, a in alt.items()}
                    d = ndimage.distance_transform_edt(~blob)
                    ring = (d >= 1) & (d <= 3) & (lab[win] == 0)
                    if ring.any():
                        c = rgb_obs[win][ring].mean(0)
                        crg, cbg = (float(np.log2(max(c[0], 1e-9) / max(c[1], 1e-9))), float(np.log2(max(c[2], 1e-9) / max(c[1], 1e-9))))
                    else:
                        crg = cbg = float("nan")
                    edge = int(sl[0].start == 0 or sl[1].start == 0 or sl[0].stop == HI or sl[1].stop == WI)
                    fo.write("\t".join(map(str, [name, int(focal), yaw, pitch, f"{S / white_sat(CAMS[main_cam]) / P90:.4g}", k, area, cls,
                             f"{ob['r'] * PITCH * 1000:.1f}", f"{base / (S / white_sat(CAMS[main_cam])):.3f}", f"{flux:.3f}", f"{peak:.3f}",
                             f"{ob['bg']:.3f}", f"{ob['skirt']:.3f}", f"{ob['slope']:.3f}", f"{ob['fmean']:.3f}", f"{ob['compact']:.3f}",
                             f"{ob['nsat3']:.3f}", f"{ob['nch']:.3f}"] + [f"{oa[n][q]:.3f}" for n in names[1:] for q in ("nsat3", "nch")]
                             + [f"{crg:.3f}", f"{cbg:.3f}", edge])) + "\n")
                fo.flush()
    fo.close()


if __name__ == "__main__":
    main(*sys.argv[1:5])
