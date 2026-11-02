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


_EXPOSURE_TAGS = {
    "time_s": ("EXIF ExposureTime",),
    "f_number": ("EXIF FNumber",),
    "iso": ("EXIF ISOSpeedRatings", "EXIF PhotographicSensitivity"),
}


def _exposure_fields(tags):
    """Valid recorded exposure tags, retaining their original rational objects."""
    fields = {}
    for name, keys in _EXPOSURE_TAGS.items():
        for key in keys:
            if key not in tags:
                continue
            try:
                value = Fraction(str(tags[key].values[0]))
                if value > 0 and np.isfinite(float(value)):
                    fields[name] = (key, value)
                    break
            except (AttributeError, ValueError, TypeError, IndexError, ZeroDivisionError, OverflowError):
                continue
    return fields


def _exif_complete(path, *, trace=None):
    """Opt-in recorded-metadata completion, never inferred ISO or radiance.

    A partial RAW record is supplemented only by the same file's embedded JPEG.
    Common shutter/aperture values must agree to relative 1e-4 (JPEG rounding),
    recorded ISO must agree exactly, and common camera identity/time strings
    must agree. Existing valid RAW
    fields always win. Original tag objects/rationals are copied without fitting.
    ``trace`` optionally records provenance and rejected conflicts; the legacy
    ``_exif`` reader remains untouched for historical/default reproduction.
    """
    import io
    import exifread
    tags = {}
    try:
        with open(path, "rb") as fh:
            tags = exifread.process_file(fh, details=False)
    except Exception:
        pass
    fields = _exposure_fields(tags)
    info = {
        "mode": "complete_exif_candidate",
        "field_sources": {name: "RAW EXIF" for name in fields},
        "recorded_values": {name: str(tags[key].values[0]) for name, (key, _) in fields.items()},
        "conflicts": [],
        "status": "RAW complete" if len(fields) == 3 else "RAW incomplete",
    }

    def finish(result):
        if trace is not None:
            trace.update(info)
        return result

    if len(fields) == 3:
        return finish(tags)
    try:
        with rawpy.imread(str(path)) as raw:
            thumb = raw.extract_thumb()
        if thumb.format != rawpy.ThumbFormat.JPEG:
            info["status"] = "no JPEG exposure record"
            return finish(tags)
        other = exifread.process_file(io.BytesIO(thumb.data), details=False)
    except Exception as exc:
        info["status"] = "embedded exposure record unavailable"
        info["thumbnail_error"] = type(exc).__name__
        return finish(tags)
    other_fields = _exposure_fields(other)
    for name in _EXPOSURE_TAGS:
        if name not in fields or name not in other_fields:
            continue
        a, b = float(fields[name][1]), float(other_fields[name][1])
        same = fields[name][1] == other_fields[name][1] if name == "iso" else np.isclose(a, b, rtol=1e-4, atol=0)
        if not same:
            info["conflicts"].append(name)
    for key in ("Image Make", "Image Model", "EXIF DateTimeOriginal"):
        if key in tags and key in other:
            if " ".join(str(tags[key]).split()).casefold() != " ".join(str(other[key]).split()).casefold():
                info["conflicts"].append(key)
    if info["conflicts"]:
        info["status"] = "conflicting embedded record; RAW retained"
        return finish(tags)
    merged = tags.copy()
    for name, (key, _) in other_fields.items():
        if name not in fields:
            merged[key] = other[key]
            info["field_sources"][name] = "embedded JPEG EXIF"
            info["recorded_values"][name] = str(other[key].values[0])
    info["status"] = "completed from embedded JPEG" if len(_exposure_fields(merged)) == 3 else "still incomplete"
    return finish(merged)


def exif_exposure(path, *, complete_exif=False):
    """(曝光时间 s, 光圈 N, ISO); 读不到返回 None。"""
    try:
        tags = _exif_complete(path) if complete_exif else _exif(path)
        if complete_exif:
            fields = _exposure_fields(tags)
            return tuple(float(fields[name][1]) for name in ("time_s", "f_number", "iso"))
        val = lambda k: float(Fraction(str(tags[k].values[0])))
        k = "EXIF ISOSpeedRatings" if "EXIF ISOSpeedRatings" in tags else "EXIF PhotographicSensitivity"
        return val("EXIF ExposureTime"), val("EXIF FNumber"), val(k)
    except Exception:
        return None


# 光圈缺失 (手动镜头 FNumber = 0) 时的可能范围; None = 视为读不到 EXIF (原行为)。研究代际设为 (1.4, 8.0), 见 T17
APERTURE_RANGE = None


def clip_luminance(path, *, complete_exif=False):
    """由 EXIF 曝光参数计算截断亮度的名义代理 (cd/m²)。
    ISO 12232 饱和感光度: S_sat = 78 / H_sat; 焦平面照度 H = q·L·t / N² (q ≈ 0.65, 镜头透过率等)
    → L_clip = 78·N² / (q·t·S) ≈ 120·N² / (t·S)。读不到 EXIF 时返回 None。
    EXIF ISO 未必为实测 S_sat；补齐记录不会标定传感器、镜头或后续亮度先验。"""
    try:
        tags = _exif_complete(path) if complete_exif else _exif(path)
        if complete_exif:
            fields = _exposure_fields(tags)
            t, n, s = (float(fields[name][1]) for name in ("time_s", "f_number", "iso"))
            return 120.0 * n * n / (t * s)
        val = lambda k: float(Fraction(str(tags[k].values[0])))
        t = val("EXIF ExposureTime")
        n = val("EXIF FNumber") if "EXIF FNumber" in tags else 0.0
        iso_key = "EXIF ISOSpeedRatings" if "EXIF ISOSpeedRatings" in tags else "EXIF PhotographicSensitivity"
        s = val(iso_key)
        if n <= 0 and APERTURE_RANGE is not None:
            # 手动镜头 (无触点) 的 FNumber 记为 0 或缺失: 快门与 ISO 仍是实测值, 光圈取可能范围的几何中值 (T17)
            n = float(np.sqrt(APERTURE_RANGE[0] * APERTURE_RANGE[1]))
        if t > 0 and n > 0 and s > 0:
            return 120.0 * n * n / (t * s)
    except Exception:
        pass
    return None


# ------------------------------------------------------------ 降噪 --
def estimate_noise(img, bins=16):
    """估计数码传感器噪声模型 方差 = a·信号 + b (a: 光子散粒噪声, b: 读出噪声), 每通道。
    用最细尺度的对角 Haar 小波系数 HH = (x00 − x01 − x10 + x11)/2 —— 独立噪声下 var(HH) = var(x),
    而边缘、纹理几乎不产生对角高频, 所以不会被当成噪声 (Donoho: σ = median|HH| / 0.6745)。
    按 2×2 块的平均亮度分组, 每组用稳健的 MAD 估计方差, 再做非负线性拟合。"""
    h, w = (img.shape[0] // 2) * 2, (img.shape[1] // 2) * 2
    x = img[:h, :w]
    x00, x01, x10, x11 = x[0::2, 0::2], x[0::2, 1::2], x[1::2, 0::2], x[1::2, 1::2]
    hh = (x00 - x01 - x10 + x11) / 2.0
    m = (x00 + x01 + x10 + x11) / 4.0
    params = []
    for c in range(img.shape[-1]):
        mc, hc = m[..., c].ravel(), hh[..., c].ravel()
        ok = (mc > 0) & (mc < np.percentile(mc, 99))
        edges = np.quantile(mc[ok], np.linspace(0, 1, bins + 1))
        xs, ys = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = ok & (mc >= lo) & (mc < hi)
            if sel.sum() > 200:
                v = (np.median(np.abs(hc[sel])) / 0.6745) ** 2
                if v > 0:          # 极暗区量化后 |HH| 中位数可为 0: 无法估计, 且会让加权拟合除以零
                    xs.append(mc[sel].mean())
                    ys.append(v)
        if len(xs) >= 3:
            from scipy.optimize import nnls
            A = np.column_stack([xs, np.ones(len(xs))])
            a_, b_ = nnls(A / np.array(ys)[:, None] ** 0.5, np.array(ys) ** 0.5)[0]   # 相对误差加权
        else:
            a_, b_ = 0.0, float(np.median(ys)) if ys else 1e-8
        params.append((max(a_, 1e-9), max(b_, 1e-12)))
    return np.array(params)


def denoise(img, params, chroma_boost=2.5, out_matrix=None, clipped=None):
    """去除数码传感器噪声 (胶片上不该有; 胶片颗粒由胶片模型自己生成)。
    1. 广义 Anscombe 方差稳定变换: 让噪声在所有亮度下都是单位方差的高斯噪声
    2. 转到亮度 / 色差空间: 色差 (彩色噪点) 降得更狠, 亮度保留细节
    3. 小波 BayesShrink 软阈值, 再逆变换回线性光"""
    from skimage.restoration import denoise_wavelet
    a, b = params[:, 0], params[:, 1]
    gat = 2.0 / a * np.sqrt(np.maximum(a * img + 0.375 * a * a + b, 0))
    M = np.array([[1 / 3, 1 / 3, 1 / 3], [0.5, 0.0, -0.5], [-0.25, 0.5, -0.25]])
    opp = gat @ M.T
    if clipped is not None and np.any(clipped > 0):
        # 截断像素的颜色是假的 (只有部分通道饱和), 不是信号: 色差通道按缺失数据处理,
        # 用周围未截断像素的色差填补 (归一化卷积), 否则小波会把假色扩散成一片色斑
        from scipy.ndimage import gaussian_filter as _g
        # 下采样后像素里只要有截断的子像素, 平均颜色就被污染; 去马赛克还会把假色带到相邻像素
        w = 1.0 - ndimage.binary_dilation(clipped > 0).astype(float)
        for c in (1, 2):
            fill = _g(opp[..., c] * w, 3.0) / np.maximum(_g(w, 3.0), 1e-6)
            opp[..., c] = w * opp[..., c] + (1 - w) * fill
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
    # 加性校正: 局部均值回到无偏的原图均值。逆变换的偏差只在光子数少时存在 (信号 ≲ 几十个 a),
    # 亮处不做 —— 否则细亮线被平滑掉的能量会被加回到它周围, 形成光圈。
    w = 1.0 / (1.0 + np.clip(img, 0, None) / (20.0 * a))
    den = den + w * (ref - cur)
    # 与观测一致: 真实信号只可能落在观测值的噪声范围内 (±3σ, σ² = a·s + b)。
    # 亮处 σ 相对很小, 这一约束去掉了小波在细亮线 (铁轨反光、灯丝) 两侧的振铃与色差溢出;
    # 暗处 σ 大, 降噪不受影响。
    # 约束在之后要使用的颜色空间里做 (out_matrix: 例如 XYZ → sRGB): 矩阵有很大的负系数,
    # 各通道分别在范围内的小偏差, 换算后会放大成明显的色边。方差按矩阵传播 (各通道独立)。
    Mo = np.eye(3) if out_matrix is None else np.asarray(out_matrix)
    var = np.maximum(a * np.clip(img, 0, None) + b, 0)
    sd = 3.0 * np.sqrt(var @ (Mo ** 2).T)
    o_img, o_den = img @ Mo.T, den @ Mo.T
    return np.clip(o_den, o_img - sd, o_img + sd) @ np.linalg.inv(Mo).T



# 截断判定: "channel_max" = 每张图各通道最大值的 97% (历史做法; 没有任何饱和的画面也会把最亮的部分判为截断);
#           "sensor"      = 传感器数据在饱和值处的堆积 (见 sensor_saturation)。current / next 保持 channel_max。
CLIP_DETECT = "channel_max"
# 截断高光的重建: "prior" = 旧法 (整像素用周围颜色 × 亮度先验覆盖); "per_channel" = 保留未饱和通道的实测值 (实验, 只在 sensor 判定下可用);
#                 "skirt" = 同 prior, 但每块高出截断的档数由 RAW 裙边 / 同图近场 PSF 推算, 给不出稳定 PSF 时回退 prior (T28 候选)
#                 "stops" = 同 prior, 但档数与 EXIF 无关: 灯具类块在 STOPS_PRIOR 内均匀取值, 细线 / 大面积保持与灯具的相对偏移 (T30 候选)
HIGHLIGHT_RECON = "prior"


def sensor_saturation(raw, per_channel=False):
    """传感器实际饱和的半尺寸掩膜 (与 postprocess(half_size=True) 的输出同方向), 没有饱和时返回 None。
    必须在 postprocess 之前调用 (half_size 处理后 libraw 内部的原始数据已被缩小)。

    元数据里的白电平不可靠 (G81 标 4079, 数据堆积在 4095; 这台索尼标 15360, 堆积在 16372, 有损压缩还留下少量更高的码值),
    所以按数据判断: 在每个 CFA 通道最高 3% 量程内找计数最多的码值 (堆积), 若其附近 (容差 0.2% 量程) 的计数 ≥ 50
    且是下方相邻区间平均计数的 10 倍以上, 该通道在此饱和, 阈值取堆积的下沿。未饱和的通道不产生截断。"""
    data = raw.raw_image_visible
    col = raw.raw_colors_visible
    black = np.asarray(raw.black_level_per_channel, float)
    thr = np.full(4, np.inf)
    for c in range(4):
        v = data[col == c]
        if v.size == 0:
            continue
        mx = float(v.max())
        rng = max(mx - black[c], 1.0)
        vals, cnt = np.unique(v[v >= mx - 0.03 * rng], return_counts=True)
        vm = float(vals[np.argmax(cnt)])
        tol = max(1.0, 0.002 * rng)
        top = int((v >= vm - tol).sum())
        below = int(((v >= vm - 21 * tol) & (v < vm - tol)).sum()) / 20.0
        if top >= 50 and top > 10.0 * max(below, 1.0):
            thr[c] = vm - tol
    if not np.isfinite(thr).any():
        return None
    sat = data >= thr[col]
    h2, w2 = data.shape[0] // 2, data.shape[1] // 2
    if per_channel:
        # 半尺寸像素里每种颜色 (R, G, B) 的传感元是否饱和 (两个绿任一饱和即算), 顺序与 postprocess 输出一致
        desc = raw.color_desc.decode()
        blk = sat[:2 * h2, :2 * w2].reshape(h2, 2, w2, 2)
        cc = col[:2 * h2, :2 * w2].reshape(h2, 2, w2, 2)
        m = np.stack([(blk & np.isin(cc, [i for i, ch in enumerate(desc) if ch == name])).any((1, 3))
                      for name in "RGB"], -1)
    else:
        m = sat[:2 * h2, :2 * w2].reshape(h2, 2, w2, 2).any((1, 3))
    flip = raw.sizes.flip
    if flip == 3:
        m = m[::-1, ::-1]
    elif flip == 5:
        m = np.rot90(m, 1)
    elif flip == 6:
        m = np.rot90(m, -1)
    return m


def load_raw(path, max_size=2400, white_balance="daylight", highlights="reconstruct", seed=0, denoise_raw=True,
             clip_detect=None, complete_exif=False, source_stops=None):
    """white_balance:
      daylight  固定日光白平衡 (默认): 保留场景真实的光色, 由胶片自己决定钨丝灯 / 日光下的表现,
                就像胶片相机没有自动白平衡一样
      camera    相机记录的白平衡 (场景被"拉正")
      auto      自动白平衡
    highlights: reconstruct = 重建截断高光 (默认); clip = 保留传感器截断
    denoise_raw: 去除数码传感器噪声 (默认开)
    clip_detect: 截断判定方法, None = 模块默认 CLIP_DETECT
    complete_exif: 显式实验候选; 从匹配的内嵌JPEG补齐缺失曝光元数据。默认关闭。
                   只校正记录，不保证既有随机亮度先验准确。
    source_stops: 截断光源比亮度先验再亮 (+) / 暗 (−) 多少档, 同时决定灯芯与光晕强度 (T13)。
                  截断之上的真实亮度在单张 RAW 中不可知: EPFL 场景所需值从 −2 (黄昏) 到 +3 (夜间大面积亮窗) 不等。
                  None = 模块默认 SOURCE_STOPS (0 = 原先验, 逐位不变)。"""
    with rawpy.imread(str(path)) as raw:
        kw = dict(use_camera_wb=(white_balance == "camera"), use_auto_wb=(white_balance == "auto"))
        if white_balance == "daylight":
            wb = list(raw.daylight_whitebalance)
            if not np.all(np.isfinite(wb[:3])) or min(wb[:3]) <= 0:   # 部分 DNG 没有日光系数
                kw["use_camera_wb"] = True
            else:
                kw["user_wb"] = wb[:3] + [wb[1] if wb[3] <= 0 else wb[3]]
        common = dict(gamma=(1, 1), no_auto_bright=True, output_bps=16, half_size=True, user_flip=None)
        detect = clip_detect or CLIP_DETECT
        sensor = sensor_saturation(raw, per_channel=True) if detect == "sensor" and (highlights == "reconstruct" or denoise_raw) \
            else None
        xyz = raw.postprocess(output_color=rawpy.ColorSpace.XYZ, **common, **kw).astype(np.float32) / 65535.0
        clip = None
        if highlights == "reconstruct" or denoise_raw:
            if detect == "sensor":
                clip = np.zeros(xyz.shape[:2], np.float32)
                sat3 = np.zeros(xyz.shape[:2] + (3,), bool)
                if sensor is not None:
                    h, w = min(sensor.shape[0], clip.shape[0]), min(sensor.shape[1], clip.shape[1])
                    sat3[:h, :w] = sensor[:h, :w]
                    clip = sat3.any(-1).astype(np.float32)
                    if highlights == "reconstruct" and HIGHLIGHT_RECON == "per_channel":
                        cam = raw.postprocess(output_color=rawpy.ColorSpace.raw, user_wb=[1, 1, 1, 1], **common)
                        cam = cam.astype(np.float32) / 65535.0
            else:
                # 在相机原始 RGB (未白平衡) 里找饱和像素: 任一通道到达该通道的最大值即为截断
                cam = raw.postprocess(output_color=rawpy.ColorSpace.raw, user_wb=[1, 1, 1, 1], **common)
                mx = cam.reshape(-1, 3).max(0).astype(np.float32)
                clip = (cam >= 0.97 * mx).any(-1).astype(np.float32)
    step = max(1, int(round(max(xyz.shape[:2]) / max_size)))
    per_channel = detect == "sensor" and HIGHLIGHT_RECON == "per_channel" and highlights == "reconstruct" \
        and clip is not None and clip.any()
    if per_channel:
        # 相机原始通道 → XYZ 的线性关系 (同一张图、未饱和像素上拟合; libraw 在截断以下是线性的)
        ok = ~sat3.any(-1) & (cam.max(-1) > 0.01) & (cam.max(-1) < 0.9)
        A = np.linalg.lstsq(cam[ok].astype(np.float64), xyz[ok].astype(np.float64), rcond=None)[0]
        clip_lv = np.array([np.median(cam[..., c][sat3[..., c]]) if sat3[..., c].any() else np.inf for c in range(3)])
        f3 = _downsample(sat3.astype(np.float32), step)
    xyz = _downsample(xyz, step)
    # T28 "skirt": 裙边与 PSF 在降噪前的下采样线性值上测 (只在该模式下取, 其余模式不经过这里)
    skirt_lin = (xyz @ XYZ_TO_SRGB.T).astype(np.float64) if (HIGHLIGHT_RECON == "skirt" and highlights == "reconstruct"
                                                          and not per_channel) else None
    if denoise_raw:      # 在线性的传感器信号上做 (下采样后噪声模型同样成立, 只是方差更小)
        # 逐通道重建时, 截断像素里未饱和的通道是实测值, 不按缺失数据抹掉它们的颜色
        cl = None if (clip is None or per_channel) else _downsample(clip, step)
        xyz = denoise(xyz.astype(np.float64), estimate_noise(xyz), out_matrix=XYZ_TO_SRGB, clipped=cl)
    if per_channel:
        frame_seed = zlib.crc32(str(path).encode()) ^ seed
        l_clip = clip_luminance(path, complete_exif=True) if complete_exif else clip_luminance(path)
        cam_d = reconstruct_highlights_cam(xyz @ np.linalg.inv(A), f3, clip_lv, np.random.default_rng(frame_seed),
                                           l_clip, SOURCE_STOPS if source_stops is None else source_stops)
        return (cam_d @ A) @ XYZ_TO_SRGB.T
    rgb = xyz @ XYZ_TO_SRGB.T
    if highlights == "reconstruct" and clip is not None and clip.any():
        frame_seed = zlib.crc32(str(path).encode()) ^ seed
        l_clip = clip_luminance(path, complete_exif=True) if complete_exif else clip_luminance(path)
        kw = {} if skirt_lin is None else {"skirt_lin": skirt_lin}
        rgb = reconstruct_highlights(rgb, _downsample(clip, step), np.random.default_rng(frame_seed), l_clip,
                                     SOURCE_STOPS if source_stops is None else source_stops, **kw)
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
# 截断光源亮度的整体偏移 (档), load_raw(source_stops=None) 时使用。0 = 原先验。见 docs/tasks/T13_HALO_STRENGTH.md
SOURCE_STOPS = 0.0
# 截断区重建颜色的取样 (T16): "ring" = 外圈较宽环带里最亮的 30% (原行为);
# "edge" = 紧邻截断边界外 1–3 像素的实测像素 —— 离灯芯最近的同一光源的光, 保证边界两侧颜色连续
RECON_COLOUR = "ring"
# 细长的截断区 (宽 ≤ 2 像素, 长宽比 ≥ 4): 粗糙金属 / 湿地面上的镜面反光 (铁轨、车身边缘) 或霓虹灯管。
# 反光的亮度 ≈ 反射率 × 灯的亮度 × (灯的立体角 / 散射瓣的立体角), 被拉成一条线正说明散射瓣远大于灯,
# 所以比灯本身暗得多; 霓虹灯管本身约 1e3 cd/m²。估计范围 (对数均匀): 1e2 – 2e3。
LINE_PRIOR = ((1e2, 2e3), (0.5, 2.5))


# T30 候选 HIGHLIGHT_RECON = "stops" (docs/tasks/T30_UNIFORM_STOPS_PRIOR.md): 估计的物理量仍是"截断块高出截断电平多少档",
# 但不再由 EXIF 的截断亮度 L_clip 把绝对亮度 (cd/m²) 换算过来 —— 那样换算使各场景相差 4–6 档 (L_clip 3.24–1200 cd/m²,
# 加上 MAX_STOPS 截顶与缺光圈时的 2–5 档), 而 RAW 裙边在场景之间没有这种差别 (T27 第 2、3 节)。
# 灯具类: 档数在 STOPS_PRIOR 内均匀取值 (= 亮度对数均匀), 所有场景同一规则, 有无光圈 EXIF 都一样。
# 锚定在用户目视认可的长曝光组 (n29240 / n40825 的原先验档数分布), 不是独立测得的绝对亮度。
# 细线 / 大面积: 保持与灯具的相对关系 —— 各自 cd/m² 范围的下限 / 上限相对灯具范围下限 / 上限的档数差
#   (大面积 1e3–1e4: −1.00 / −1.58 档; 细线 1e2–2e3: −4.32 / −3.91 档), 见 _stops_range。
# 穹顶形状、颜色、截断判定、每块一次随机数抽取都与 prior 相同; 结果限制在 [0.05, MAX_STOPS]。
STOPS_PRIOR = (6.0, 8.0)
STOPS_LOG = None          # 诊断: 设为 list 时, 每块追加 {"area", "cls", "stops", "r_eq", "flux"} (所有模式; 不影响输出)


def _stops_range(lum_range):
    """stops 模式下某类块的档数范围: 灯具类为 STOPS_PRIOR, 其余类按 cd/m² 范围相对灯具范围平移。"""
    lamp = HIGHLIGHT_PRIOR[0][1]
    return (STOPS_PRIOR[0] + float(np.log2(lum_range[0] / lamp[0])),
            STOPS_PRIOR[1] + float(np.log2(lum_range[1] / lamp[1])))


def _line_like(blob):
    """按截断区的形状判断是否为细线: 主轴长度与平均宽度 (面积 / 长度)。"""
    ys, xs = np.nonzero(blob)
    if len(ys) < 3:
        return False
    ev = np.linalg.eigvalsh(np.cov(np.stack([ys, xs]).astype(float)) + 1e-9 * np.eye(2))
    length = np.sqrt(12.0 * ev[-1])
    width = len(ys) / max(length, 1.0)
    return width <= 2.0 and length >= 4.0 * width


def _shift_stops(stops, offset):
    """亮度先验整体平移 offset 档 (offset = 0 时原样返回, 逐位不变); 不低于 0.05 档, 上限随正偏移放宽。"""
    if not offset:
        return stops
    return float(np.clip(stops + offset, 0.05, MAX_STOPS + max(offset, 0.0)))


# T28 候选 HIGHLIGHT_RECON = "skirt" (docs/tasks/T28_SHAPE_BRIGHTNESS.md):
# 每块高出截断的档数 s 不再按 EXIF 的绝对亮度先验, 而由同一张 RAW 推算 ——
#   近场 PSF: 未截断的紧凑亮点 (峰值 0.1–0.9 截断电平, 离截断 > 20 px, 孤立, 背景 < 峰值 5%, 半峰面积 ≤ 9 px) 的径向剖面 / 峰值;
#   裙边: 截断核 (clip_frac > 0.25) 外 1–3 像素、未截断像素的线性亮度 − 背景 (12–16 px 环);
#   光源取半径 ρ 的均匀圆盘: 幅度由裙边定, ρ 由截断面积定 (_skirt_block_stops); s = log2(穹顶峰值处的预测值 / 截断电平)。
#   点源 (ρ = 0) 时即 s = log2(裙边 / (截断电平 × PSF(Δ)))。合成检验 (已知 PSF 的点源与 1–4 像素圆盘): 中位偏差 −0.3…+0.6 档。
# 假设: 截断源与未截断亮点共用同一镜头 PSF; 未截断亮点是近似点源 (若有一定大小, PSF 被估宽, s 偏低); PSF 径向对称、全画面相同。
# 回退到 prior (同一随机数, 结果与 prior 相同): 该图紧凑亮点 < SKIRT_PSF_MIN 个或 PSF 自助标准误 > SKIRT_PSF_MAX_SE 档;
#   单块: 大面积 (≥ 画面 0.5%)、背景 > 截断电平 25%、没有可用裙边像素。裙边不高于背景时取下限。
SKIRT_RANGE = (0.5, 8.0)
SKIRT_PSF_R = 12
SKIRT_PSF_MIN = 8
SKIRT_PSF_MAX_SE = 0.5
SKIRT_LOG = None          # 诊断: 设为 list 时, 每张图追加 {"psf": …, "blocks": […]}


def skirt_psf(lin, clip_frac):
    """同一张图里未截断紧凑亮点的近场 PSF 径向剖面 p(r)/p(0), r = 0..SKIRT_PSF_R (像素)。返回 (剖面或 None, 信息)。"""
    Y = lin @ LUMA
    info = {"n": 0}
    if not (clip_frac > 0.9).any():
        return None, info
    cl = float(np.median(Y[clip_frac > 0.9]))
    dc = ndimage.distance_transform_edt(~(clip_frac > 0))
    pk = (Y == ndimage.maximum_filter(Y, 7)) & (Y > 0.1 * cl) & (Y < 0.9 * cl) & (dc > 20)
    h, w = Y.shape
    W = SKIRT_PSF_R + 5
    yy, xx = np.mgrid[-W:W + 1, -W:W + 1]
    rr = np.hypot(yy, xx)
    rings = [np.abs(rr - r) <= 0.5 for r in range(SKIRT_PSF_R + 1)]
    bgr = (rr > SKIRT_PSF_R + 2) & (rr <= W)
    P = []
    for y, x in zip(*np.nonzero(pk)):
        if y < W or x < W or y >= h - W or x >= w - W:
            continue
        win = Y[y - W:y + W + 1, x - W:x + W + 1]
        bg = np.median(win[bgr])
        v = win - bg
        p0 = v[W, W]
        if bg > 0.05 * p0 or (v[rr <= 4] > 0.5 * p0).sum() > 9 or (win[rr <= SKIRT_PSF_R + 1] > win[W, W]).any():
            continue
        P.append([np.median(v[m]) / p0 for m in rings])
    P = np.array(P)
    info.update(n=len(P), clip=cl)
    if len(P) < SKIRT_PSF_MIN:
        info["why"] = "psf_points"
        return None, info
    prof = np.median(P, 0)
    br = np.random.default_rng(0)
    bs = np.array([np.median(P[br.integers(0, len(P), len(P))], 0) for _ in range(200)])
    se = np.std(np.log2(np.clip(bs[:, 1:4], 1e-6, None)), 0)
    info.update(prof=prof.tolist(), se=se.tolist())
    if not np.all(prof[1:4] > 0) or se.max() > SKIRT_PSF_MAX_SE:
        info["why"] = "psf_unstable"
        return None, info
    info["why"] = "ok"
    return np.clip(prof, 0.0, None), info


def _skirt_g(prof, pts, yy, xx):
    """Σ_j p(|y − x_j|) / n: 圆盘 (采样点 pts) 经 PSF 剖面 p (峰值 = 1) 后在 (yy, xx) 处的值, 点源时中心为 1。"""
    R = len(prof) - 1
    g = np.zeros(np.broadcast(yy, xx).shape)
    for py, px in pts:
        g += np.interp(np.hypot(yy - py, xx - px), np.arange(R + 1), prof, right=0.0)
    return g / len(pts)


def _skirt_block_stops(sk, k, sl, base):
    """一个截断块 (标号 k) 的档数估计 → (s 或 None, 信息, 原因)。sk 为 reconstruct_highlights 里准备的字典。
    光源 = 半径 ρ 的均匀圆盘 (中心 = 穹顶峰值像素, 与 p 的原点约定一致), 经同图 PSF 剖面 p:
      幅度 a 由边界外 1–3 像素的裙边定 (每个 Δ 一个值, 取几何平均);
      ρ 由截断面积定: 预测值 a·G ≥ 截断电平的像素数 = 实测 f ≥ 0.5 的像素数 (p 是像素尺度的剖面, 在像素中心比较)。点源 (ρ = 0) 时即 s = log2(裙边 / (截断电平 × p(Δ)))。
    s = log2(重建在穹顶峰值处应写入的光源部分 / (f·截断电平))。"""
    M = SKIRT_PSF_R + 6
    H, Wd = sk["Ylin"].shape
    y0, y1 = max(sl[0].start - M, 0), min(sl[0].stop + M, H)
    x0, x1 = max(sl[1].start - M, 0), min(sl[1].stop + M, Wd)
    win = (slice(y0, y1), slice(x0, x1))
    blob = sk["lab"][win] == k
    fr = np.clip(sk["clip"][win], 0.0, 1.0)
    free = (sk["own"][win] == k) & (fr == 0)
    dc = sk["dc"][win]
    Yl = sk["Ylin"][win]
    bgm = free & (dc > 12) & (dc <= 16)
    if bgm.sum() < 5:
        return None, None, "no_bg"
    bgv = float(np.median(Yl[bgm]))
    if bgv > 0.25 * base:
        return None, None, "bright_bg"
    fb = np.where(blob, fr, 0.0)
    a_obs = float(fb.sum())
    if a_obs <= 0:
        return None, None, "no_core"
    yy, xx = np.mgrid[0:blob.shape[0], 0:blob.shape[1]].astype(float)
    d = ndimage.distance_transform_edt(blob)
    py, px = np.unravel_index(np.argmax(d), d.shape)                 # 穹顶峰值像素 = 圆盘中心 (p 以峰值像素为原点)
    cy, cx = float(py), float(px)
    rr = np.hypot(yy - cy, xx - cx)
    prof = sk["prof"]
    rings = {D: free & (np.rint(dc) == D) & (rr <= len(prof) - 1) for D in (1, 2, 3)}
    rings = {D: m for D, m in rings.items() if m.sum() >= 3}
    if not rings:
        return None, None, "no_ring"
    if not any(np.median(Yl[m]) - bgv > 0 for m in rings.values()):
        return SKIRT_RANGE[0], {"rho": None}, "skirt<=bg"
    fc = float(fr[py, px])
    if fc <= 0:
        return None, None, "no_core"
    near = blob | (dc <= 4)
    ny, nx = np.nonzero(near)
    n_half = float((fb >= 0.5).sum())                               # 一半以上子像素截断的像素数
    best = None
    rmax = float(np.sqrt(a_obs / np.pi)) + 1.0
    grid = np.arange(0.0, min(rmax, 3.0) + 1e-9, 0.25)
    if rmax > 3.0:
        grid = np.concatenate([grid, np.geomspace(3.0, rmax, 10)[1:]])
    for rho in grid:
        if rho < 0.125:
            pts = [(cy, cx)]
        else:
            sp = max(0.25, rho / 6.0)                                   # 圆盘采样间距 (大圆盘放宽, 控制计算量)
            o = np.arange(-rho, rho + 1e-9, sp)
            pts = [(cy + u, cx + v) for u in o for v in o if u * u + v * v <= rho * rho] or [(cy, cx)]
        la = []
        for m in rings.values():
            g = _skirt_g(prof, pts, yy[m], xx[m])
            q = np.median((Yl[m] - bgv) / np.maximum(g, 1e-9))
            if q > 0:
                la.append(np.log(q))
        if not la:
            continue
        a = float(np.exp(np.mean(la)))
        gs = _skirt_g(prof, pts, ny.astype(float), nx.astype(float))
        a_pred = float((a * gs >= sk["cl"]).sum())
        err = abs(np.log(max(a_pred, 0.5) / max(n_half, 0.5)))
        if best is None or err < best[0]:
            best = (err, rho, a, pts, a_pred, (max(la) - min(la)) / np.log(2.0))
    if best is None:
        return SKIRT_RANGE[0], {"rho": None}, "skirt<=bg"
    err, rho, a, pts, a_pred, spread = best
    oc = a * float(_skirt_g(prof, pts, np.array([py]), np.array([px]))[0])
    lamp = oc - (1.0 - fc) * float(Yl[py, px])
    if lamp <= 0:
        return SKIRT_RANGE[0], {"rho": rho}, "skirt<=bg"
    return float(np.log2(lamp / (fc * base))), {"rho": float(rho), "n_half": n_half, "n_pred": a_pred,
                                                    "spread": float(spread)}, "ok"   # spread: 各 Δ 推得幅度之差 (档)


def reconstruct_highlights(rgb, clip_frac, rng, l_clip=None, source_stops=0.0, skirt_lin=None):
    """重建传感器截断的高光。

    每个截断区域:
      颜色  取自其周围一圈未截断的像素 (光源的颜色保留在它周围的光晕 / 漫射里)
      亮度  在光源典型绝对亮度范围内随机 (真实值已无法得知), 由 EXIF 换算成截断点以上的档数;
            不会低于截断点
      形状  从边缘 (= 截断亮度, 保证连续) 向内部逐渐变亮的穹顶, 模拟发光体中心最亮
      面积  只有像素里被截断的那部分子像素换成亮源 (clip_frac), 不放大细亮线的面积
    skirt_lin: 降噪前的线性 RGB; 给出时 (HIGHLIGHT_RECON = "skirt") 档数改由裙边 / 同图 PSF 推算, 见 skirt_psf"""
    mask = ndimage.binary_dilation(clip_frac > 0.25)
    lab, n = ndimage.label(mask)
    if n == 0:
        return rgb
    out = rgb.copy()
    total = mask.size
    Y = rgb @ LUMA
    sk = None
    if skirt_lin is not None:
        prof, pinfo = skirt_psf(skirt_lin, clip_frac)
        log = {"psf": pinfo, "blocks": []}
        if SKIRT_LOG is not None:
            SKIRT_LOG.append(log)
        if prof is not None:
            core = clip_frac > 0.25
            dc, (iy, ix) = ndimage.distance_transform_edt(~core, return_indices=True)
            sk = {"prof": prof, "cl": pinfo["clip"], "Ylin": skirt_lin @ LUMA, "lab": lab, "clip": clip_frac, "dc": dc,
                  "own": np.where(core, lab, 0)[iy, ix]}
    for k, sl in enumerate(ndimage.find_objects(lab), start=1):
        pad = max(4, int(0.6 * np.sqrt((sl[0].stop - sl[0].start) * (sl[1].stop - sl[1].start))))
        y0, y1 = max(sl[0].start - pad, 0), min(sl[0].stop + pad, rgb.shape[0])
        x0, x1 = max(sl[1].start - pad, 0), min(sl[1].stop + pad, rgb.shape[1])
        blob = lab[y0:y1, x0:x1] == k
        area = blob.sum()
        ring = ndimage.binary_dilation(blob, iterations=pad) & ~ndimage.binary_dilation(blob, iterations=2) \
            & (lab[y0:y1, x0:x1] == 0)
        sub = rgb[y0:y1, x0:x1]
        edge = ndimage.binary_dilation(blob, iterations=3) & ~blob & (lab[y0:y1, x0:x1] == 0)
        if RECON_COLOUR == "edge" and edge.sum() >= 5:
            col = np.median(rgb[y0:y1, x0:x1][edge], 0)
            col = col / (col @ LUMA) if col @ LUMA > 0 else np.ones(3)   # 不截负值: 与边界外像素完全同色
        elif ring.sum() >= 5:
            # 周围一圈里最亮的那部分最接近光源本身的颜色
            ry = Y[y0:y1, x0:x1][ring]
            sel = ring.copy()
            sel[ring] = ry >= np.percentile(ry, 70)
            col = np.median(sub[sel], 0)
        else:
            col = np.median(sub[blob], 0)
        if not (RECON_COLOUR == "edge" and edge.sum() >= 5):
            col = np.clip(col, 1e-6, None)
            col = col / (col @ LUMA)
        base = np.percentile(Y[y0:y1, x0:x1][blob], 90)          # 截断亮度
        _, lum_range, stop_range = next(p for p in HIGHLIGHT_PRIOR if area / total < p[0])
        if _line_like(clip_frac[y0:y1, x0:x1] * blob > 0.25):
            lum_range, stop_range = LINE_PRIOR
        if HIGHLIGHT_RECON == "stops":   # T30: 与 L_clip 无关, 同样每块抽一次随机数
            stops = float(np.clip(rng.uniform(*_stops_range(lum_range)), 0.05, MAX_STOPS))
        elif l_clip is not None:
            L = np.exp(rng.uniform(*np.log(lum_range)))
            stops = float(np.clip(np.log2(L / l_clip), 0.05, MAX_STOPS))
        else:
            stops = rng.uniform(*stop_range)
        if sk is not None:            # T28: 随机数照常抽取, 回退的块与 prior 逐位相同
            est, sD, why = (None, None, "large") if area / total >= HIGHLIGHT_PRIOR[0][0] else _skirt_block_stops(sk, k, sl, base)
            log["blocks"].append({"area": int(area), "prior": float(stops), "est": est, "info": sD, "why": why,
                                  "line": bool(lum_range == LINE_PRIOR[0])})
            if est is not None:
                stops = float(np.clip(est, *SKIRT_RANGE))
        stops = _shift_stops(stops, source_stops)
        d = ndimage.distance_transform_edt(blob)
        dome = (d / max(d.max(), 1.0)) ** 0.7
        lum = base * 2.0 ** (stops * dome)
        region = out[y0:y1, x0:x1]
        # 下采样像素是盒式平均: 只有被截断的那部分 (f) 是被藏起来的亮源, 其余保持观测值。
        # 细的反光线 (铁轨、金属边) 只占像素的一小部分, 不能把整个像素 (更不能把膨胀出的一圈) 都当成灯。
        f = np.clip(clip_frac[y0:y1, x0:x1][blob], 0.0, 1.0)[:, None]
        region[blob] = (1.0 - f) * region[blob] + f * (lum[blob][:, None] * col[None, :])
        if STOPS_LOG is not None:     # flux: 截断子像素里重建 / 截断电平 (档), 即 f 加权的 2^(stops·dome) 平均
            STOPS_LOG.append({"area": int(area), "stops": float(stops),
                              "cls": "line" if lum_range == LINE_PRIOR[0] else ("large" if area / total >= HIGHLIGHT_PRIOR[0][0] else "lamp"),
                              "r_eq": float(np.sqrt(f.sum() / np.pi)),
                              "flux": float(np.log2(max((f[:, 0] * lum[blob]).sum(), 1e-12) / max(f.sum() * base, 1e-12)))})
    return out


def reconstruct_highlights_cam(cam, sat_frac, clip_lv, rng, l_clip=None, source_stops=0.0):
    """逐通道重建截断高光 (相机原始通道, 传感器饱和判定时使用)。

    与 reconstruct_highlights 的区别: 截断像素里**未饱和的通道是实测值, 保留**, 并用来约束饱和通道:
      · 像素里还有未饱和的通道: 饱和通道 = 实测通道 × 光源颜色比例 (颜色取自周围一圈未截断像素, 同旧法), 不低于截断值。
        这样的光源最多只比截断点亮出颜色比例那么多, 不再套用"亮几档"的先验。
      · 三个通道全部饱和: 亮度不可辨识, 沿用旧法的亮度先验与穹顶形状 (HIGHLIGHT_PRIOR), 且不低于截断值与相邻实测推算值。
    下采样像素里饱和传感元的比例为 f_c, 观测值是截断值与未饱和部分的平均 → 只加回 f_c × (估计值 − 截断值)。"""
    any_sat = sat_frac.max(-1)
    mask = ndimage.binary_dilation(any_sat > 0.25)
    lab, n = ndimage.label(mask)
    if n == 0:
        return cam
    out = cam.copy()
    total = mask.size
    Y = cam.mean(-1)
    lv = np.where(np.isfinite(clip_lv), clip_lv, np.inf)
    for k, sl in enumerate(ndimage.find_objects(lab), start=1):
        pad = max(4, int(0.6 * np.sqrt((sl[0].stop - sl[0].start) * (sl[1].stop - sl[1].start))))
        y0, y1 = max(sl[0].start - pad, 0), min(sl[0].stop + pad, cam.shape[0])
        x0, x1 = max(sl[1].start - pad, 0), min(sl[1].stop + pad, cam.shape[1])
        blob = lab[y0:y1, x0:x1] == k
        f = sat_frac[y0:y1, x0:x1][blob]                       # (n, 3) 饱和传感元比例
        if not (f > 0).any():
            continue
        ring = ndimage.binary_dilation(blob, iterations=pad) & ~ndimage.binary_dilation(blob, iterations=2) \
            & (lab[y0:y1, x0:x1] == 0)
        sub = cam[y0:y1, x0:x1]
        if ring.sum() >= 5:
            ry = Y[y0:y1, x0:x1][ring]
            sel = ring.copy()
            sel[ring] = ry >= np.percentile(ry, 70)
            col = np.median(sub[sel], 0)
        else:
            col = np.median(sub[blob], 0)
        col = np.clip(col, 1e-6, None)
        obs = sub[blob]                                         # (n, 3)
        measured = f <= 0.0
        # 有实测通道的像素: 光源颜色标度 = 实测通道 / 颜色 (取最大, 偏保守地不低估)
        ratio = np.where(measured, obs / col[None, :], -np.inf)
        s_meas = ratio.max(1)
        est = s_meas[:, None] * col[None, :]
        full = ~measured.any(1)
        if full.any():
            area = blob.sum()
            _, lum_range, stop_range = next(p for p in HIGHLIGHT_PRIOR if area / total < p[0])
            if _line_like(any_sat[y0:y1, x0:x1] * blob > 0.25):
                lum_range, stop_range = LINE_PRIOR
            if l_clip is not None:
                L = np.exp(rng.uniform(*np.log(lum_range)))
                stops = float(np.clip(np.log2(L / l_clip), 0.05, MAX_STOPS))
            else:
                stops = rng.uniform(*stop_range)
            stops = _shift_stops(stops, source_stops)
            d = ndimage.distance_transform_edt(blob)
            dome = 2.0 ** (stops * (d / max(d.max(), 1.0)) ** 0.7)[blob]
            satc = np.isfinite(lv)
            ref = int(np.argmax(np.where(satc, col / np.where(satc, lv, 1.0), -np.inf)))
            base = lv[ref] / col[ref]                              # 颜色标度: 最受限的饱和通道刚好到截断
            floor = np.nanmax(np.where(np.isfinite(s_meas), s_meas, np.nan)) if np.isfinite(s_meas).any() else 0.0
            est[full] = np.maximum(dome[full] * base, floor)[:, None] * col[None, :]
        est = np.maximum(est, np.where(np.isfinite(lv), lv, 0.0)[None, :])
        excess = np.where(f > 0, f * np.maximum(est - np.where(np.isfinite(lv), lv, est)[None, :], 0.0), 0.0)
        region = out[y0:y1, x0:x1]
        region[blob] = region[blob] + np.nan_to_num(excess, posinf=0.0)
    return out


def exif_iso(path, *, complete_exif=False):
    try:
        tags = _exif_complete(path) if complete_exif else _exif(path)
        if complete_exif:
            return float(_exposure_fields(tags)["iso"][1])
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


# 摄影师实际给 CineStill 800T 的曝光 (相对 EI 800 的档数), 来自参考集来源页中摄影师对自己拍摄的陈述
# (正常显影; 不含推冲、只给建议的页面与存疑的页面): EI 320/400/500×4/"500或640"/640×2/800×4。
# 中位数 +0.5 档; 夜景 / 室内 (n=9) 中位 +0.68 档, 日光样本很少。tools/eval/ei_survey.tsv
EI_SURVEY_EV = (1.32, 1.0, 0.68, 0.68, 0.68, 0.68, 0.5, 0.32, 0.32, 0.0, 0.0, 0.0, 0.0)


# 胶片曝光的推算方式 (由 filmlab.generation.use 设置): "meter" = 按下述测光模型逐帧推算 (默认);
# "session" = 胶片与数码拿到同一份曝光 (RAW 线性 × 全程常数 10^SESSION_LOG_SCALE), 不逐帧测光。
# 常数 = EPFL 这次拍摄训练集 11 帧测光倍数的中位数 (docs/tasks/T06_LIGHT_COLOUR_DIAGNOSIS.md); 只适用于该次拍摄。
METERING = "meter"
SESSION_LOG_SCALE = 0.6129018861240094


def film_metering(rgb, l_clip=None, ei=800.0, rng=None, gray=0.184, t_original=None, bias_ev=None):
    """胶片摄影师用胶片相机拍同一场景: 按 EI 测光, 但受相机的物理极限约束。

    rgb     load_raw 的输出 (1.0 = 传感器截断), l_clip: 截断点的场景亮度 (cd/m², EXIF)。
    测光    中央重点的平均亮度 (线性光平均, 同测光表积分), 不计光源本身 (熟练的摄影师 / 评价测光
            不会让画面里的灯把整张压暗)。反射式测光方程: N²/t = L·S/K, K = 12.5。
    极限    手持时镜头最大光圈 N_min 与安全快门 t_max 决定能给的最大曝光; 场景比这更暗就只能欠曝,
            欠多少由绝对亮度决定 —— 夜景比室内更容易欠曝, 这是真实的物理约束。
            最小光圈 / 最高快门限制强光 (f/16, 1/1000)。
            原片用了比手持更长的曝光 (t_original, EXIF) 说明用了三脚架: 胶片摄影师同样可以。
    补偿    bias_ev: 摄影师在测光读数上多给 / 少给的曝光。默认按实测分布 EI_SURVEY_EV 每张随机抽取
            (摄影师自述的拍摄 EI); 同样受相机极限约束 —— 光线不够时补偿不了。
    随机    每张的镜头与快门在常见范围内取 (rng): N_min ∈ {1.4, 1.8, 2} (夜拍 800T 常用大光圈定焦),
            手持 t_max ∈ [1/60, 1/15] s。
    返回    胶片曝光 (中灰 = gray, 即胶片按 EI 正确曝光时中灰卡的位置) 与说明。"""
    if METERING == "session":
        return rgb * 10.0 ** SESSION_LOG_SCALE, "同一份曝光 (会话常数)"
    rng = np.random.default_rng(0) if rng is None else rng
    Y = np.clip(rgb @ LUMA, 0, None)
    h, w = Y.shape
    yy, xx = np.mgrid[0:h, 0:w]
    wgt = np.exp(-0.5 * (((yy - h / 2) / (0.35 * h)) ** 2 + ((xx - w / 2) / (0.35 * w)) ** 2))
    wgt = wgt * (Y < 0.9)                                   # 不计光源 / 截断的高光
    y_meter = float((Y * wgt).sum() / max(wgt.sum(), 1e-9))
    y_meter = max(y_meter, 1e-6)
    if bias_ev is None:
        bias_ev = float(rng.choice(EI_SURVEY_EV))
    scale, note = gray / y_meter * 2.0 ** bias_ev, f"测光 (补偿 {bias_ev:+.2f} 档)"
    if l_clip is not None:
        L = y_meter * l_clip                                # 测光读数 (cd/m²)
        need = L * ei / 12.5 / 2.0 ** bias_ev               # 需要的 N²/t (含负片 "宁过勿欠" 的曝光补偿)
        n_min = float(rng.choice([1.4, 1.8, 2.0]))
        t_max = float(np.exp(rng.uniform(np.log(1 / 60), np.log(1 / 15))))
        if t_original is not None and t_original > t_max:     # 三脚架
            t_max = float(t_original)
        lo, hi = n_min ** 2 / t_max, 16.0 ** 2 * 1000.0
        actual = float(np.clip(need, lo, hi))
        scale *= need / actual
        note = (f"测光 {L:.2g} cd/m², 需要 N²/t={need:.3g}, 相机 f/{n_min:g} · 1/{1 / t_max:.0f}s"
                f" → {np.log2(need / actual):+.1f} 档")
    return rgb * scale, note


def auto_expose(rgb, key=0.18):
    """平均测光: 让画面的几何平均亮度 = 中灰。"""
    Y = rgb @ LUMA
    gm = np.exp(np.mean(np.log(np.clip(Y, 1e-5, None))))
    return rgb * (key / gm)
