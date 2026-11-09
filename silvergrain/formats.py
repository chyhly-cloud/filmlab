"""画幅 (T53): 名义画面尺寸 (长边, 短边) mm。图像放到胶片上时 "长边对长边": µm/像素 = 画幅长边 × 1000 / 图像长边像素数。
长宽比不同时, 图像对应画面里最大的同长宽比区域 (要整幅画面先按画幅长宽比裁切, 见 crop_to_format)。"""
import numpy as np

FILM_FORMATS = {
    "35mm": (36.0, 24.0), "half": (24.0, 18.0), "110": (17.0, 13.0), "126": (26.5, 26.5), "aps_c": (25.1, 16.7),
    "645": (56.0, 41.5), "6x6": (56.0, 56.0), "6x7": (69.5, 56.0), "6x9": (84.0, 56.0), "4x5": (121.0, 97.0),
}


def format_long_mm(fmt):
    """画幅名或长边 (mm) → 长边 mm。"""
    if isinstance(fmt, str):
        if fmt not in FILM_FORMATS:
            raise KeyError(f"未知画幅 {fmt!r}; 可选 {', '.join(FILM_FORMATS)}")
        return FILM_FORMATS[fmt][0]
    return float(fmt)


def px_um_for(shape, fmt="35mm"):
    """图像 (h, w[, c]) 放在画幅 fmt 上时每像素的胶片边长 (µm): 长边对长边。"""
    return format_long_mm(fmt) * 1000.0 / float(max(shape[0], shape[1]))


def crop_to_format(img, fmt):
    """把图像居中裁切到画幅 fmt (FILM_FORMATS) 的长宽比 (长边方向不变, 不重采样), 使它对应整幅画面。"""
    L, S = FILM_FORMATS[fmt]
    h, w = img.shape[:2]
    a = L / S
    if h >= w:
        hh, ww = (h, int(round(h / a))) if h / w <= a else (int(round(w * a)), w)
    else:
        ww, hh = (w, int(round(w / a))) if w / h <= a else (int(round(h * a)), h)
    y0, x0 = (h - hh) // 2, (w - ww) // 2
    return img[y0:y0 + hh, x0:x0 + ww]
