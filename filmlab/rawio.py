"""S0 计算摄影: RAW 解码与高光重建, 输出场景线性光 (线性 sRGB 原色, 允许超出色域的负值)。

只做解马赛克和色彩矩阵, 不做任何风格化。另外处理一个数字传感器特有的问题:
传感器在高光处饱和截断, 灯泡、太阳、镜面反光的真实亮度丢失了。胶片没有这个硬上限 ——
它的肩部会把高光渐渐压缩, 而且光晕的强度正比于光源的真实亮度。所以要先把截断的高光
按物理上合理的范围重建出来, 胶片模拟才能得到正确的高光与光晕。
"""
import zlib
from fractions import Fraction

import numpy as np
import rawpy
from scipy import ndimage

XYZ_TO_SRGB = np.array([[3.2404542, -1.5371385, -0.4985314],
                        [-0.9692660, 1.8760108, 0.0415560],
                        [0.0556434, -0.2040259, 1.0572252]])
LUMA = np.array([0.2126, 0.7152, 0.0722])


def _downsample(a, step):
    if step <= 1:
        return a
    h, w = (a.shape[0] // step) * step, (a.shape[1] // step) * step
    shp = (h // step, step, w // step, step) + a.shape[2:]
    return a[:h, :w].reshape(shp).mean((1, 3))     # 盒式平均 (线性光下, 物理正确)


def _exif(path):
    """读 EXIF; 部分格式 (Fuji RAF, Canon CR3) exifread 不认, 改读 RAW 内嵌的 JPEG 预览。"""
    import io
    import exifread
    try:
        with open(path, "rb") as fh:
            tags = exifread.process_file(fh, details=False)
        if "EXIF ExposureTime" in tags:
            return tags
    except Exception:
        pass
    try:
        with rawpy.imread(str(path)) as raw:
            th = raw.extract_thumb()
        if th.format == rawpy.ThumbFormat.JPEG:
            return exifread.process_file(io.BytesIO(th.data), details=False)
    except Exception:
        pass
    return {}


def clip_luminance(path):
    """由 EXIF 曝光参数估计传感器截断点对应的场景亮度 (cd/m²)。
    ISO 12232 饱和感光度: S_sat = 78 / H_sat; 焦平面照度 H = q·L·t / N² (q ≈ 0.65, 镜头透过率等)
    → L_clip = 78·N² / (q·t·S) ≈ 120·N² / (t·S)。读不到 EXIF 时返回 None。"""
    try:
        tags = _exif(path)
        val = lambda k: float(Fraction(str(tags[k].values[0])))
        t, n = val("EXIF ExposureTime"), val("EXIF FNumber")
        iso_key = "EXIF ISOSpeedRatings" if "EXIF ISOSpeedRatings" in tags else "EXIF PhotographicSensitivity"
        s = val(iso_key)
        if t > 0 and n > 0 and s > 0:
            return 120.0 * n * n / (t * s)
    except Exception:
        pass
    return None


# ------------------------------------------------------------ 降噪 --
def estimate_noise(img, block=8):
    """估计数码传感器噪声模型 方差 = a·信号 + b (a: 光子散粒噪声, b: 读出噪声), 每通道。
    在平坦小块里统计均值与方差, 取每个亮度区间方差的低分位数 (平坦区域) 做线性拟合。"""
    h, w = (img.shape[0] // block) * block, (img.shape[1] // block) * block
    blocks = img[:h, :w].reshape(h // block, block, w // block, block, -1)
    m = blocks.mean((1, 3)).reshape(-1, img.shape[-1])
    v = blocks.var((1, 3), ddof=1).reshape(-1, img.shape[-1])
    params = []
    for c in range(img.shape[-1]):
        mc, vc = m[:, c], v[:, c]
        ok = (mc > 0) & (mc < np.percentile(mc, 99))
        edges = np.quantile(mc[ok], np.linspace(0, 1, 21))
        xs, ys = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = ok & (mc >= lo) & (mc < hi)
            if sel.sum() > 20:
                xs.append(mc[sel].mean())
                ys.append(np.percentile(vc[sel], 10))
        a, b = np.polyfit(xs, ys, 1) if len(xs) >= 3 else (0.0, np.percentile(vc, 10))
        params.append((max(a, 1e-9), max(b, 1e-12)))
    return np.array(params)


def denoise(img, params, chroma_boost=2.5):
    """去除数码传感器噪声 (胶片上不该有; 胶片颗粒由胶片模型自己生成)。
    1. 广义 Anscombe 方差稳定变换: 让噪声在所有亮度下都是单位方差的高斯噪声
    2. 转到亮度 / 色差空间: 色差 (彩色噪点) 降得更狠, 亮度保留细节
    3. 小波 BayesShrink 软阈值, 再逆变换回线性光"""
    from skimage.restoration import denoise_wavelet
    a, b = params[:, 0], params[:, 1]
    gat = 2.0 / a * np.sqrt(np.maximum(a * img + 0.375 * a * a + b, 0))
    M = np.array([[1 / 3, 1 / 3, 1 / 3], [0.5, 0.0, -0.5], [-0.25, 0.5, -0.25]])
    opp = gat @ M.T
    out = np.empty_like(opp)
    out[..., 0] = denoise_wavelet(opp[..., 0], sigma=np.sqrt(1 / 3), wavelet="db2", mode="soft",
                                  method="BayesShrink", rescale_sigma=False)
    for c, sig in ((1, np.sqrt(0.5)), (2, np.sqrt(0.375))):
        out[..., c] = denoise_wavelet(opp[..., c], sigma=sig * chroma_boost, wavelet="db2", mode="soft",
                                      method="BayesShrink", rescale_sigma=False)
    g = out @ np.linalg.inv(M).T
    den = ((a * g / 2.0) ** 2 - 0.375 * a * a - b) / a
    # 去偏: 逆变换在低信号处有偏 (E[f(x)] ≠ f(E[x]))。传感器噪声在线性域均值为零,
    # 原图的局部平均是无偏的 → 用它校正降噪结果的局部均值 (每通道)
    from scipy.ndimage import gaussian_filter
    s = (6.0, 6.0, 0)
    ref, cur = gaussian_filter(img, s), gaussian_filter(den, s)
    return den + (ref - cur)              # 加性校正: 局部均值回到无偏的原图均值, 对暗部稳定



def load_raw(path, max_size=2400, white_balance="daylight", highlights="reconstruct", seed=0, denoise_raw=True):
    """white_balance:
      daylight  固定日光白平衡 (默认): 保留场景真实的光色, 由胶片自己决定钨丝灯 / 日光下的表现,
                就像胶片相机没有自动白平衡一样
      camera    相机记录的白平衡 (场景被"拉正")
      auto      自动白平衡
    highlights: reconstruct = 重建截断高光 (默认); clip = 保留传感器截断
    denoise_raw: 去除数码传感器噪声 (默认开)"""
    with rawpy.imread(str(path)) as raw:
        kw = dict(use_camera_wb=(white_balance == "camera"), use_auto_wb=(white_balance == "auto"))
        if white_balance == "daylight":
            wb = list(raw.daylight_whitebalance)
            if not np.all(np.isfinite(wb[:3])) or min(wb[:3]) <= 0:   # 部分 DNG 没有日光系数
                kw["use_camera_wb"] = True
            else:
                kw["user_wb"] = wb[:3] + [wb[1] if wb[3] <= 0 else wb[3]]
        common = dict(gamma=(1, 1), no_auto_bright=True, output_bps=16, half_size=True, user_flip=None)
        xyz = raw.postprocess(output_color=rawpy.ColorSpace.XYZ, **common, **kw).astype(np.float32) / 65535.0
        clip = None
        if highlights == "reconstruct":
            # 在相机原始 RGB (未白平衡) 里找饱和像素: 任一通道到达该通道的最大值即为截断
            cam = raw.postprocess(output_color=rawpy.ColorSpace.raw, user_wb=[1, 1, 1, 1], **common)
            mx = cam.reshape(-1, 3).max(0).astype(np.float32)
            clip = (cam >= 0.97 * mx).any(-1).astype(np.float32)
    step = max(1, int(round(max(xyz.shape[:2]) / max_size)))
    xyz = _downsample(xyz, step)
    if denoise_raw:      # 在线性的传感器信号上做 (下采样后噪声模型同样成立, 只是方差更小)
        xyz = denoise(xyz.astype(np.float64), estimate_noise(xyz))
    rgb = xyz @ XYZ_TO_SRGB.T
    if clip is not None and clip.any():
        frame_seed = zlib.crc32(str(path).encode()) ^ seed
        rgb = reconstruct_highlights(rgb, _downsample(clip, step), np.random.default_rng(frame_seed),
                                     clip_luminance(path))
    return rgb


# 截断高光的真实亮度先验。画面里看起来小不代表光源小 (远处的路灯), 所以不按面积区分灯具类型,
# 只把大面积高光 (天空、亮窗) 单独处理。有 EXIF 时用典型绝对亮度 (cd/m², 照明工程常用量级,
# 对数均匀取值):
#   灯具      带灯罩的路灯 / 霓虹灯管 / 荧光灯 / 灯箱 / 白炽灯泡   2e3 – 3e4
#   大面积    天空 / 云 / 亮窗                                   1e3 – 1e4 (不低于截断点)
# 夜景曝光加大, 截断点对应的亮度低, 同样的灯就比截断点亮得多 —— 这正是夜景光晕强的原因。
# 没有 EXIF 时退回"比截断点高出的档数"。上限 8 档。
HIGHLIGHT_PRIOR = [
    (5e-3, (2e3, 3e4), (2.0, 5.0)),
    (1.0, (1e3, 1e4), (0.0, 1.5)),
]
MAX_STOPS = 8.0


def reconstruct_highlights(rgb, clip_frac, rng, l_clip=None):
    """重建传感器截断的高光。

    每个截断区域:
      颜色  取自其周围一圈未截断的像素 (光源的颜色保留在它周围的光晕 / 漫射里)
      亮度  在光源典型绝对亮度范围内随机 (真实值已无法得知), 由 EXIF 换算成截断点以上的档数;
            不会低于截断点
      形状  从边缘 (= 截断亮度, 保证连续) 向内部逐渐变亮的穹顶, 模拟发光体中心最亮"""
    mask = ndimage.binary_dilation(clip_frac > 0.25)
    lab, n = ndimage.label(mask)
    if n == 0:
        return rgb
    out = rgb.copy()
    total = mask.size
    Y = rgb @ LUMA
    for k, sl in enumerate(ndimage.find_objects(lab), start=1):
        pad = max(4, int(0.6 * np.sqrt((sl[0].stop - sl[0].start) * (sl[1].stop - sl[1].start))))
        y0, y1 = max(sl[0].start - pad, 0), min(sl[0].stop + pad, rgb.shape[0])
        x0, x1 = max(sl[1].start - pad, 0), min(sl[1].stop + pad, rgb.shape[1])
        blob = lab[y0:y1, x0:x1] == k
        area = blob.sum()
        ring = ndimage.binary_dilation(blob, iterations=pad) & ~ndimage.binary_dilation(blob, iterations=2) \
            & (lab[y0:y1, x0:x1] == 0)
        sub = rgb[y0:y1, x0:x1]
        if ring.sum() >= 5:
            # 周围一圈里最亮的那部分最接近光源本身的颜色
            ry = Y[y0:y1, x0:x1][ring]
            sel = ring.copy()
            sel[ring] = ry >= np.percentile(ry, 70)
            col = np.median(sub[sel], 0)
        else:
            col = np.median(sub[blob], 0)
        col = np.clip(col, 1e-6, None)
        col = col / (col @ LUMA)
        base = np.percentile(Y[y0:y1, x0:x1][blob], 90)          # 截断亮度
        _, lum_range, stop_range = next(p for p in HIGHLIGHT_PRIOR if area / total < p[0])
        if l_clip is not None:
            L = np.exp(rng.uniform(*np.log(lum_range)))
            stops = float(np.clip(np.log2(L / l_clip), 0.05, MAX_STOPS))
        else:
            stops = rng.uniform(*stop_range)
        d = ndimage.distance_transform_edt(blob)
        dome = (d / max(d.max(), 1.0)) ** 0.7
        lum = base * 2.0 ** (stops * dome)
        region = out[y0:y1, x0:x1]
        region[blob] = (lum[blob][:, None] * col[None, :])
    return out


def exif_iso(path):
    try:
        tags = _exif(path)
        k = "EXIF ISOSpeedRatings" if "EXIF ISOSpeedRatings" in tags else "EXIF PhotographicSensitivity"
        return float(Fraction(str(tags[k].values[0])))
    except Exception:
        return None


def film_exposure(rgb, path=None, gray=0.184):
    """按"摄影师的曝光意图不变, 换用胶片按其 EI 重新测光"换算曝光 (保留夜景本来的明暗)。

    ISO 12232: 饱和感光度 H_sat = 78/S, 标准输出感光度的中灰曝光 H_m = 10/S,
    所以摄影师意图中的中灰落在传感器截断点的 10/78 ≈ 0.128 处 (与 ISO 无关)。
    load_raw 输出中 1.0 ≈ 截断, 于是把 0.128 对应到胶片的中灰 0.184。
    (若改用"相同快门和光圈", 高 ISO 夜景换到 EI 800 会欠曝好几档, 不符合胶片摄影师的实际做法。)"""
    return rgb * gray / (10.0 / 78.0)


def auto_expose(rgb, key=0.18):
    """平均测光: 让画面的几何平均亮度 = 中灰。"""
    Y = rgb @ LUMA
    gm = np.exp(np.mean(np.log(np.clip(Y, 1e-5, None))))
    return rgb * (key / gm)
