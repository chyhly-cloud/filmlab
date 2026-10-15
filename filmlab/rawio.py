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


def clip_luminance(path):
    """由 EXIF 曝光参数估计传感器截断点对应的场景亮度 (cd/m²)。
    ISO 12232 饱和感光度: S_sat = 78 / H_sat; 焦平面照度 H = q·L·t / N² (q ≈ 0.65, 镜头透过率等)
    → L_clip = 78·N² / (q·t·S) ≈ 120·N² / (t·S)。读不到 EXIF 时返回 None。"""
    try:
        import exifread
        with open(path, "rb") as fh:
            tags = exifread.process_file(fh, details=False)
        val = lambda k: float(Fraction(str(tags[k].values[0])))
        t, n = val("EXIF ExposureTime"), val("EXIF FNumber")
        iso_key = "EXIF ISOSpeedRatings" if "EXIF ISOSpeedRatings" in tags else "EXIF PhotographicSensitivity"
        s = val(iso_key)
        if t > 0 and n > 0 and s > 0:
            return 120.0 * n * n / (t * s)
    except Exception:
        pass
    return None


def load_raw(path, max_size=2400, white_balance="daylight", highlights="reconstruct", seed=0):
    """white_balance:
      daylight  固定日光白平衡 (默认): 保留场景真实的光色, 由胶片自己决定钨丝灯 / 日光下的表现,
                就像胶片相机没有自动白平衡一样
      camera    相机记录的白平衡 (场景被"拉正")
      auto      自动白平衡
    highlights: reconstruct = 重建截断高光 (默认); clip = 保留传感器截断"""
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
    rgb = _downsample(xyz, step) @ XYZ_TO_SRGB.T
    if clip is not None and clip.any():
        frame_seed = zlib.crc32(str(path).encode()) ^ seed
        rgb = reconstruct_highlights(rgb, _downsample(clip, step), np.random.default_rng(frame_seed),
                                     clip_luminance(path))
    return rgb


# 截断高光的真实亮度先验, 按高光面积占画面的比例分类。
# 有 EXIF 时用光源的典型绝对亮度 (cd/m², 照明工程常用量级, 对数均匀取值):
#   小而集中  LED 灯珠 / 灯丝 / 镜面反光     1e4 – 1e6
#   中等      磨砂灯罩 / 霓虹 / 荧光灯管 / 灯箱 3e3 – 5e4
#   大面积    天空 / 云 / 亮窗                2e3 – 2e4
# 夜景曝光加大, 截断点对应的亮度低, 同样的灯就比截断点亮得多 —— 这正是夜景光晕强的原因。
# 没有 EXIF 时退回"比截断点高出的档数"。
HIGHLIGHT_PRIOR = [
    (2e-4, (1e4, 1e6), (3.0, 7.0)),
    (5e-3, (3e3, 5e4), (1.5, 4.0)),
    (1.0, (2e3, 2e4), (0.0, 1.5)),
]


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
            stops = max(np.log2(L / l_clip), 0.05)
        else:
            stops = rng.uniform(*stop_range)
        d = ndimage.distance_transform_edt(blob)
        dome = (d / max(d.max(), 1.0)) ** 0.7
        lum = base * 2.0 ** (stops * dome)
        region = out[y0:y1, x0:x1]
        region[blob] = (lum[blob][:, None] * col[None, :])
    return out


def auto_expose(rgb, key=0.18):
    """平均测光: 让画面的几何平均亮度 = 中灰。"""
    Y = rgb @ LUMA
    gm = np.exp(np.mean(np.log(np.clip(Y, 1e-5, None))))
    return rgb * (key / gm)
