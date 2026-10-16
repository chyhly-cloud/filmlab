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


def exif_exposure(path):
    """(曝光时间 s, 光圈 N, ISO); 读不到返回 None。"""
    try:
        tags = _exif(path)
        val = lambda k: float(Fraction(str(tags[k].values[0])))
        k = "EXIF ISOSpeedRatings" if "EXIF ISOSpeedRatings" in tags else "EXIF PhotographicSensitivity"
        return val("EXIF ExposureTime"), val("EXIF FNumber"), val(k)
    except Exception:
        return None


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
                xs.append(mc[sel].mean())
                ys.append((np.median(np.abs(hc[sel])) / 0.6745) ** 2)
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
        if highlights == "reconstruct" or denoise_raw:
            # 在相机原始 RGB (未白平衡) 里找饱和像素: 任一通道到达该通道的最大值即为截断
            cam = raw.postprocess(output_color=rawpy.ColorSpace.raw, user_wb=[1, 1, 1, 1], **common)
            mx = cam.reshape(-1, 3).max(0).astype(np.float32)
            clip = (cam >= 0.97 * mx).any(-1).astype(np.float32)
    step = max(1, int(round(max(xyz.shape[:2]) / max_size)))
    xyz = _downsample(xyz, step)
    if denoise_raw:      # 在线性的传感器信号上做 (下采样后噪声模型同样成立, 只是方差更小)
        cl = None if clip is None else _downsample(clip, step)
        xyz = denoise(xyz.astype(np.float64), estimate_noise(xyz), out_matrix=XYZ_TO_SRGB, clipped=cl)
    rgb = xyz @ XYZ_TO_SRGB.T
    if highlights == "reconstruct" and clip is not None and clip.any():
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
# 细长的截断区 (宽 ≤ 2 像素, 长宽比 ≥ 4): 粗糙金属 / 湿地面上的镜面反光 (铁轨、车身边缘) 或霓虹灯管。
# 反光的亮度 ≈ 反射率 × 灯的亮度 × (灯的立体角 / 散射瓣的立体角), 被拉成一条线正说明散射瓣远大于灯,
# 所以比灯本身暗得多; 霓虹灯管本身约 1e3 cd/m²。估计范围 (对数均匀): 1e2 – 2e3。
LINE_PRIOR = ((1e2, 2e3), (0.5, 2.5))


def _line_like(blob):
    """按截断区的形状判断是否为细线: 主轴长度与平均宽度 (面积 / 长度)。"""
    ys, xs = np.nonzero(blob)
    if len(ys) < 3:
        return False
    ev = np.linalg.eigvalsh(np.cov(np.stack([ys, xs]).astype(float)) + 1e-9 * np.eye(2))
    length = np.sqrt(12.0 * ev[-1])
    width = len(ys) / max(length, 1.0)
    return width <= 2.0 and length >= 4.0 * width


def reconstruct_highlights(rgb, clip_frac, rng, l_clip=None):
    """重建传感器截断的高光。

    每个截断区域:
      颜色  取自其周围一圈未截断的像素 (光源的颜色保留在它周围的光晕 / 漫射里)
      亮度  在光源典型绝对亮度范围内随机 (真实值已无法得知), 由 EXIF 换算成截断点以上的档数;
            不会低于截断点
      形状  从边缘 (= 截断亮度, 保证连续) 向内部逐渐变亮的穹顶, 模拟发光体中心最亮
      面积  只有像素里被截断的那部分子像素换成亮源 (clip_frac), 不放大细亮线的面积"""
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
        if _line_like(clip_frac[y0:y1, x0:x1] * blob > 0.25):
            lum_range, stop_range = LINE_PRIOR
        if l_clip is not None:
            L = np.exp(rng.uniform(*np.log(lum_range)))
            stops = float(np.clip(np.log2(L / l_clip), 0.05, MAX_STOPS))
        else:
            stops = rng.uniform(*stop_range)
        d = ndimage.distance_transform_edt(blob)
        dome = (d / max(d.max(), 1.0)) ** 0.7
        lum = base * 2.0 ** (stops * dome)
        region = out[y0:y1, x0:x1]
        # 下采样像素是盒式平均: 只有被截断的那部分 (f) 是被藏起来的亮源, 其余保持观测值。
        # 细的反光线 (铁轨、金属边) 只占像素的一小部分, 不能把整个像素 (更不能把膨胀出的一圈) 都当成灯。
        f = np.clip(clip_frac[y0:y1, x0:x1][blob], 0.0, 1.0)[:, None]
        region[blob] = (1.0 - f) * region[blob] + f * (lum[blob][:, None] * col[None, :])
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


# 摄影师实际给 CineStill 800T 的曝光 (相对 EI 800 的档数), 来自参考集来源页中摄影师对自己拍摄的陈述
# (正常显影; 不含推冲、只给建议的页面与存疑的页面): EI 320/400/500×4/"500或640"/640×2/800×4。
# 中位数 +0.5 档; 夜景 / 室内 (n=9) 中位 +0.68 档, 日光样本很少。tools/eval/ei_survey.tsv
EI_SURVEY_EV = (1.32, 1.0, 0.68, 0.68, 0.68, 0.68, 0.5, 0.32, 0.32, 0.0, 0.0, 0.0, 0.0)


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
