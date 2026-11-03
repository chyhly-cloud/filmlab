"""T19 光晕统计量: 对真实 CineStill 800T 扫描 (参考) 与模型成片用同一套代码测量点光源周围的径向颜色分布。

灯芯: 线性 RGB 三通道都 ≥ CORE (印成纯白) 的连通块, 不接触画面边缘。按等效半径分组 (胶片 µm)。
每个像素只归属最近的灯芯 (距离变换的最近点), 按到灯芯边缘的距离 (胶片 µm) 分环, 取线性 RGB 中位数。
输出每个灯芯组的: 色度 r = R/(R+G+B)、b = B/(R+G+B)、亮度 Y, 以及相对 1.5–2.5 mm 环 (局部背景) 的红色度增量。
画幅比例: 135 (36×24 mm): 宽高比 1.45–1.55 时长边 = 36 mm; 更方的裁切按短边 = 24 mm; 更宽的裁切比例不明, 跳过。
         6x7: 短边 = 56 mm。
"""
import numpy as np
from scipy import ndimage

BINS = np.array([0, 25, 50, 100, 150, 200, 300, 400, 500, 700, 1000, 1500, 2500])
SIZES = ((0, 60), (60, 150), (150, 400))       # 灯芯等效半径 (µm)
CORE = 0.9
DARK_BG = 0.02       # "暗背景灯" 判据: 1.5–2.5 mm 环背景三通道中位 < DARK_BG (t24/t27/t29 等)


def dark_bg_max(fname=""):
    """暗背景判据的逐通道阈值。默认 DARK_BG。T44: 冲印店黑位抬升 β 是逐通道仿射 (输出 = β + (1 − β)·原输出),
    环境变量 BG_LIFT="文件名子串:βR,βG,βB" 时, 文件名含该子串的图改用 β + (1 − β)·DARK_BG ——
    与未抬升时同一组灯 (中位与单调映射可交换), 只用于模型渲染; 参考扫描不受影响。"""
    import os
    spec = os.environ.get("BG_LIFT", "")
    if spec:
        key, _, b = spec.partition(":")
        if key and key in str(fname):
            b = np.array([float(x) for x in b.split(",")])
            return b + (1.0 - b) * DARK_BG
    return np.full(3, DARK_BG)


def px_um(shape, fmt):
    h, w = shape[:2]
    lo, hi = min(h, w), max(h, w)
    a = hi / lo
    if fmt in ("6x7", "6x6"):          # 中画幅: 短边 56 mm
        return 56000.0 / lo
    if 1.45 <= a <= 1.55:
        return 36000.0 / hi
    if a < 1.45:
        return 24000.0 / lo
    return None


def srgb_to_lin(x):
    x = np.asarray(x, float)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def profiles(lin, um):
    """lin: 线性 RGB (h,w,3); um: µm/像素。→ {size_class: [每灯芯 (len(BINS)-1, 3) 中位 RGB]}"""
    core = (lin >= CORE).all(-1)
    lab, n = ndimage.label(core)
    out = {s: [] for s in SIZES}
    if n == 0:
        return out
    h, w = core.shape
    d, (iy, ix) = ndimage.distance_transform_edt(~core, return_indices=True)
    d = d * um
    owner = lab[iy, ix]
    objs = ndimage.find_objects(lab)
    area = np.bincount(lab.ravel())
    reach = int(np.ceil(BINS[-1] / um)) + 2
    for k, sl in enumerate(objs, start=1):
        if sl[0].start == 0 or sl[1].start == 0 or sl[0].stop == h or sl[1].stop == w:
            continue
        r_eq = np.sqrt(area[k] / np.pi) * um
        cls = next((s for s in SIZES if s[0] <= r_eq < s[1]), None)
        if cls is None:
            continue
        y0, y1 = max(sl[0].start - reach, 0), min(sl[0].stop + reach, h)
        x0, x1 = max(sl[1].start - reach, 0), min(sl[1].stop + reach, w)
        mine = owner[y0:y1, x0:x1] == k
        dd = d[y0:y1, x0:x1]
        px = lin[y0:y1, x0:x1]
        prof = np.full((len(BINS) - 1, 3), np.nan)
        for i, (lo, hi) in enumerate(zip(BINS[:-1], BINS[1:])):
            m = mine & (dd > lo) & (dd <= hi)
            if m.sum() >= 6:
                prof[i] = np.median(px[m], 0)
        if np.isfinite(prof[-1]).all():            # 需要局部背景
            out[cls].append(prof)
    return out


def summarize(list_of_profiles):
    """多灯芯 → 中位色度 / 亮度 / 红色度增量 (相对 1.5–2.5 mm)。"""
    P = np.array(list_of_profiles)                  # (n, bins, 3)
    S = np.nansum(P, -1)
    r, b = P[..., 0] / S, P[..., 2] / S
    Y = P @ np.array([0.2126, 0.7152, 0.0722])
    dr = r - r[:, -1:]
    med = lambda a: np.nanmedian(a, 0)
    return {"n": len(P), "r": med(r), "b": med(b), "Y": med(Y), "dr": med(dr)}
