"""给任意照片加银盐颗粒: apply_to_image。每一步都是有物理定义的步骤 (见 README "apply_to_image 的物理定义")。

  0. 读入: sRGB 编码的 8 / 16 位 JPEG / PNG / TIFF, 或线性浮点数组; 彩色或单色; 任意尺寸与长宽比。
  1. 几何: 画幅 (默认 35 mm) 长边对图像长边 → 每像素的胶片边长 px_um = 画幅长边 × 1000 / 图像长边像素数
     (output_px 给定时, 先在线性光里按面积平均缩到该长边, 即以输出像素为扫描孔径)。
  2. 曝光: 线性值当作景物反射率 (中灰 0.18), 经 RGB → 光谱上采样、设计光源、可选 Wratten 滤镜, 按胶片每层自己的光谱感光度
     积分 (spectra.layer_response; 黑白 = 全色一条), 中性灰 0.18 落在 H = 10/EI (ISO 测光约定: 18% 灰卡的曝光;
     净密度 0.1 处 H = 0.8/EI, 所以中灰在速度点之上 log10(12.5) = 1.10), 再乘 2^EV (曝光补偿)。
  3. 底片: 该片 (+ 显影液, 颗粒粗细 k) 的晶体模型给出平均密度 (闭式特性曲线) 与逐像素颗粒密度 (render)。
  T55b 起, 两种模式都经冲印店的实测影调 (lab="rpl_noritsu", labtone.py; data/lab_tone.json): 输出码值 = T(D − D_锚),
  T 由 RPL Noritsu 扫描与同一底片翻拍的配准实测 (黑白 T51b 14 帧, 彩色 T54 31 帧每通道)。lab=None 为 T55 的旧做法 (finish)。
  4a. mode="full" (整卷胶片): 光谱曝光 → 该片的特性曲线 + 颗粒 → 底片密度 D → (可选扫描光学) → 冲印店定光 + 影调 → 码值。
        定光 timing="auto" (默认): 冲印店在无颗粒密度上自动定光 (同 filmlab T47 的预扫), 整帧平均密度 → 码值 118;
        彩色片每通道各自定光 (= 冲印店的自动色彩平衡)。timing="fixed": 测光中灰的密度 → 码值 118 (曝光补偿可见)。
        反转片: 正片透射率 10^−(D − D_min) (没有冲印店影调)。
  4b. mode="grain" (只加颗粒, 默认): 输出仍是这张照片 (影调、颜色), 只加物理颗粒。T55d 起的定义 (grain_density="film"):
        · 底片: 与 full 模式同一张底片 —— 该片自己的光谱曝光 H (步骤 2: RGB → 光谱 → 该片光谱感光度, 可选 Wratten 滤镜,
          18% 灰 = 10/EI × 2^EV) → 该片的特性曲线 → 平均密度 D̄ 与逐像素颗粒密度 D_px, δ = D_px − D̄。
        · 颗粒强度 = "这张底片在这家冲印店印出来, 这一点会显示的起伏": 冲印店定光 (timing, 与 full 模式同一约定: 默认整帧
          无颗粒平均密度 → 码值 118) 给出锚点 m, 这一点印成码值 T(D̄ − m); 颗粒让它变成 T(D̄ − m + δ)。
          相对亮度因子 f = lin(T(D̄ − m + δ)) / lin(T(D̄ − m)) (二阶平均偏移按同一 u 的系综扣除, 只剩 −log⟨T⟩)。
        · 输出 = 照片 × f: 黑白 = 中性因子 (T55c 的保色相方法: 三通道同一 f, 某通道会超过正像白时把色度向白收);
          彩色负片 = 每层自己的 f_c 乘到对应通道 (每层的曝光来自该层光谱感光度, 同一原理); 反转片 f = 10^−δ。
        · 与 full 模式一致: full 模式在这一点输出 lin(T(D̄ − m + δ)), 它对无颗粒值 lin(T(D̄ − m)) 的相对起伏就是 f; grain 模式把
          同一个 f 乘到照片上。所以两种模式同一像素的相对亮度起伏相同 (不加冲印店扫描时严格相等; 加 Noritsu 时 USM 的作用域
          不同, 差 ≤ 3%), 区别只在底下的图: full 是这张底片的印片, grain 是照片本身。
        · 照片的码值不再参与密度 (T55b/c 用照片亮度的码值经冲印店影调反推密度: grain_density="photo", 只为对照)。
          所以蓝色背景按全色片对蓝光的实际曝光决定颗粒, 加滤镜 (例如 Wratten 25) 会改变底片各处的密度与颗粒, 照片影调不变。
        照片按显示出来的样子处理: 线性值 > 1 (HDR 输入) 截到 1; 全黑的像素保持黑 (相对因子作用于 0)。
  5. 冲印店扫描 (scanner="auto" = 与 lab 同一家; None 关闭)。
     Noritsu HS-1800 (T55e, 线阵 CCD 采样链; scan_grid / _to_output_grid / _sharpen3):
       · 底片渲染在 CCD 的原生网格上 (原生间距 = 横跨片带的画面尺寸 / CCD 线数: 135 为 24 mm / 4492, 120 为 56 mm / 4824;
         像素孔径 = 原生像素, 扫描光学 σ = 0), 照片的曝光 H 先插值到原生网格;
       · 冲印店软件把原生扫描按这张照片的像素数缩小 (三角核按倍数拉宽 = 带抗混叠的双线性; 在密度域做), 照片更细时线性放大;
       · 输出像素上的可分离 3 抽头锐化 (输送方向 kx, CCD 方向 ky; 编码后的亮度上)。
       mode="grain" 时锐化作用于编码后的 (输出 − 照片) 差 (小信号线性化)。
     Frontier (T51): 光学高斯 σ (输出像素, 透射率上)、传感器孔径 (冲印店像素间距按画幅长边 / 36 换算, 比渲染像素大 1.2 倍
     以上时加方盒)、亮度 USM、色度降噪 (彩色)。
  6. 输出: 线性 sRGB (浮点) 与 sRGB 编码 8 / 16 位。不加输出端噪声, 不做调色。
"""
import time
from dataclasses import dataclass, field

import numpy as np

from . import database as db
from .emulsion import derive
from .films import film as _film
from .formats import format_long_mm
from .emulsion import silver_curve_table
from .render import render
from .spectra import layer_response

LN10 = np.log(10.0)
GREY = 0.18
_YCC = np.array([[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5], [0.5, -0.418688, -0.081312]])


# ---------------------------------------------------------------- 读写 --
def srgb_encode(lin):
    x = np.clip(np.asarray(lin, float), 0.0, None)
    if np.ndim(x) == 0:
        return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)
    lo = x <= 0.0031308
    out = np.power(x, 1 / 2.4)                   # 逐步原位计算 (同一公式; 24 MP 彩色的峰值内存)
    out *= 1.055
    out -= 0.055
    out[lo] = 12.92 * x[lo]
    return out


def srgb_decode(v):
    v = np.clip(np.asarray(v, float), 0.0, None)
    if np.ndim(v) == 0:
        return np.where(v <= 0.04045, v / 12.92, np.power((v + 0.055) / 1.055, 2.4))
    lo = v <= 0.04045
    out = v + 0.055
    out /= 1.055
    np.power(out, 2.4, out=out)
    out[lo] = v[lo] / 12.92
    return out


def load_image(src, encoding="auto"):
    """路径或数组 → (线性浮点 (h, w, 3) 或 (h, w), 说明)。整数数组 / 文件按 sRGB 编码解码; 浮点数组 encoding="auto" 时当作线性
    (encoding="srgb" 则先解码)。16 位 PNG / TIFF 经 imageio (可选依赖) 或 Pillow 读入。"""
    meta = {}
    if isinstance(src, (str, bytes)) or hasattr(src, "__fspath__"):
        arr = _read(src)
        meta["path"] = str(src)
    else:
        arr = np.asarray(src)
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[..., :3]
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[..., 0]
    if arr.dtype.kind in "ui":
        mx = 65535.0 if arr.dtype.itemsize >= 2 and arr.max() > 255 else 255.0
        meta.update(bits=16 if mx > 255 else 8, encoding="srgb")
        lin = srgb_decode(arr.astype(np.float64) / mx)
    else:
        a = arr.astype(np.float64)
        if encoding == "srgb":
            lin = srgb_decode(a); meta["encoding"] = "srgb"
        else:
            lin = a; meta["encoding"] = "linear"
    if lin.ndim == 3 and np.allclose(lin[..., 0], lin[..., 1]) and np.allclose(lin[..., 1], lin[..., 2]):
        meta["mono_content"] = True
    meta["mono"] = lin.ndim == 2
    return lin, meta


def _read(p):
    try:
        import imageio.v3 as iio
        return np.asarray(iio.imread(p))
    except ImportError:
        from PIL import Image
        return np.asarray(Image.open(p))


def save_image(path, lin, bits=8):
    """线性 → sRGB 编码写文件 (8 位 JPEG / PNG, 16 位 PNG / TIFF)。"""
    v = srgb_encode(np.clip(lin, 0, 1))
    if bits == 16:
        a = np.round(v * 65535).astype(np.uint16)
        try:
            import imageio.v3 as iio
            iio.imwrite(path, a)
            return
        except ImportError:
            pass
    from PIL import Image
    a = np.round(v * 255).astype(np.uint8)
    kw = dict(quality=95) if str(path).lower().endswith((".jpg", ".jpeg")) else {}
    Image.fromarray(a).save(path, **kw)


def resize_long(lin, long_px):
    """线性光里按面积平均 (方盒) 缩放到长边 long_px (输出像素 = 扫描孔径)。"""
    from PIL import Image
    h, w = lin.shape[:2]
    k = long_px / max(h, w)
    size = (max(1, round(w * k)), max(1, round(h * k)))
    def one(c):
        return np.asarray(Image.fromarray(np.asarray(c, np.float32), "F").resize(size, Image.BOX), np.float64)
    return one(lin) if lin.ndim == 2 else np.stack([one(lin[..., c]) for c in range(lin.shape[2])], -1)


def phone_size(enc_or_lin, long_px=1200, encoded=True):
    """手机尺寸预览 (T52 的约定): 编码后的整张用 Lanczos 缩到长边 1200。"""
    from PIL import Image
    v = enc_or_lin if encoded else srgb_encode(enc_or_lin)
    h, w = v.shape[:2]
    k = long_px / max(h, w)
    size = (round(w * k), round(h * k))
    def one(c):
        return np.asarray(Image.fromarray(np.asarray(c, np.float32), "F").resize(size, Image.LANCZOS), np.float64)
    return one(v) if v.ndim == 2 else np.stack([one(v[..., c]) for c in range(v.shape[2])], -1)


# --------------------------------------------------------- 曲线的斜率 --
def _curve_tables(em):
    """每层 (log10 H 表, 平均密度表, 斜率表, 速度点斜率)。"""
    out = []
    if em.path == "silver":
        lh, Dt = silver_curve_table(em.spec)
        layers = [(lh, Dt)]
    else:
        Hx = em.exposure_axis()
        D = em.curve.density_curves
        layers = [(np.log10(Hx[:, c]), D[:, c]) for c in range(3)]
    for lh, Dt in layers:
        g = np.gradient(Dt, lh)
        rev = Dt[0] > Dt[-1]
        net = (Dt.max() - Dt) if rev else (Dt - Dt.min())
        mono = np.maximum.accumulate(net[::-1])[::-1] if rev else np.maximum.accumulate(net)
        o = np.argsort(mono, kind="stable")
        lh_s = float(np.interp(0.1, mono[o], lh[o]))
        out.append((lh, Dt, g, float(np.interp(lh_s, lh, g))))
    return out


def _slope_at(tab, logH):
    lh, Dt, g, gs = tab
    gl = np.interp(logH, lh, g)
    return np.sign(gs) * np.maximum(np.abs(gl), abs(gs)), gl


def _avg_gradient(tab, logH_grey, stops=2.0):
    lh, Dt, _, _ = tab
    d = stops * np.log10(2.0)
    return float((np.interp(logH_grey + d, lh, Dt) - np.interp(logH_grey - d, lh, Dt)) / (2 * d))


EXP_BIAS_BIN = 0.05            # 系综按 log10 H 分档的宽度
EXP_BIAS_MIN = 256             # 每档至少的像素数 (不足时与相邻档合并)


def _exp_bias(x, logH):
    """x = δ/γ (输出的 log10 起伏) 在同一曝光下的系综: b(H) = log10 E[10^x | H] − E[x | H] (指数映射的 Jensen 项, ≥ 0)。
    v' = v·10^(x − b) 的期望 = v·10^E[x|H]: 只剩物理的 −log⟨T⟩ (E[x] = E[δ]/γ > 0), 与一阶线性传递的平均相同, 但恒为正。
    按 log10 H 分档 (0.05) 在整幅图上取同一曝光的全部像素 (不是空间邻域: 没有边缘光晕); 像素少的档与相邻档合并。"""
    k = np.floor(logH / EXP_BIAS_BIN).astype(np.int64)
    k0 = int(k.min())
    idx = (k - k0).ravel()
    n = int(idx.max()) + 1
    N = np.bincount(idx, minlength=n).astype(float)
    S0 = np.bincount(idx, weights=x.ravel(), minlength=n)
    m = float(np.max(x)) if x.size else 0.0
    S1 = np.bincount(idx, weights=np.exp(np.log(10.0) * (x.ravel() - m)), minlength=n)
    grp = np.zeros(n, np.int64)                           # 相邻的非空档依次合并成组, 每组 ≥ EXP_BIAS_MIN 像素
    g, acc = 0, 0.0
    nz = np.nonzero(N)[0]
    for j in nz:
        grp[j] = g
        acc += N[j]
        if acc >= EXP_BIAS_MIN:
            g, acc = g + 1, 0.0
    if acc > 0 and g > 0:                                 # 末尾不足的一组并入前一组
        grp[nz[grp[nz] == g]] = g - 1
    ng = int(grp[nz].max()) + 1 if len(nz) else 1
    GN = np.bincount(grp[nz], weights=N[nz], minlength=ng)
    G0 = np.bincount(grp[nz], weights=S0[nz], minlength=ng)
    G1 = np.bincount(grp[nz], weights=S1[nz], minlength=ng)
    bg = (np.log(G1 / np.maximum(GN, 1)) / np.log(10.0) + m) - G0 / np.maximum(GN, 1)
    b = np.zeros(n)
    b[nz] = bg[grp[nz]]
    return b[idx].reshape(x.shape)


# ---------------------------------------------------------- 冲印店扫描 --
@dataclass(frozen=True)
class Scanner:
    name: str
    pitch_um: float
    optics_px: float
    usm_sigma_px: float
    usm_amount: float
    usm_threshold: float
    chroma_sigma_px: float
    chroma_threshold: float
    source: str = ""
    optics_px_t51: float | None = None      # T51 原拟合值 (细像素连续高斯 + 方盒的口径), 只作记录
    # T55e: 线阵 CCD 扫描仪的采样链 (native 不为 None 时启用, 见 scan_grid / _to_output_grid / _sharpen3):
    native: dict | None = None              # 每种片幅 (135 / 120) 横跨片带方向的画面尺寸 (mm) 与该方向的 CCD 采样线数
    resample: str | None = None             # 原生采样 → 输出尺寸: "triangle" (按缩小倍数拉宽的三角核 = 带抗混叠的双线性)
    sharpen_kx: float = 0.0                 # 输出像素上的可分离 3 抽头锐化 [−k, 1 + 2k, −k]: x = 片带输送方向
    sharpen_ky: float = 0.0                 #                                                 y = CCD 线阵方向
    optics_um: float = 0.0                  # 扫描光学的高斯 σ (µm, 胶片面; T55e 拟合为 0)
    t51: dict | None = None                 # 被取代的 T51 / T55c 参数 (只作记录)


def get_scanner(key):
    if key is None or isinstance(key, Scanner):
        return key
    return Scanner(**db.scanners()["labs"][key])


GAUGE_120 = ("645", "6x6", "6x7", "6x9", "4x5")


def film_gauge(fmt):
    """画幅 → 片带: "120" (中画幅; 4x5 页片按 120 的扫描光学近似) / "135" (35 mm 片带上的画幅, 以及 110 / 126 / APS 的近似)。"""
    if isinstance(fmt, str):
        return "120" if fmt in GAUGE_120 else "135"
    return "120" if float(fmt) > 45.0 else "135"


SCAN_S_MAX = 4.0                # 原生网格最多比输出细 4 倍 (更粗的输出: 三角核宽 ≥ 8 个原生像素, 原生方盒的细节已被平均掉,
                                #   改在 输出间距 / 4 上渲染)
SCAN_MAX_PIXELS = 26e6          # 原生网格最多 2600 万像素 (内存: 24 MP 彩色 grain 模式在 9 GB 内); 超过时渲染间距略大于 CCD 间距


def scan_grid(shape, px_um, fmt, sc, gauge=None):
    """线阵 CCD 扫描仪 (sc.native) 的原生采样网格: 原生间距 = 横跨片带方向的画面尺寸 / CCD 线数 (HS-1800: 135 为 24 mm / 4492,
    120 为 56 mm / 4824; 两个方向同一间距)。冲印店输出 = 按这张照片的像素数订的扫描 (输出间距 = 渲染像素 px_um),
    缩小倍数 s = px_um / 原生间距 (s < 1: 照片比扫描仪能分辨的更细, 原生扫描再放大)。渲染在原生网格上 (像素孔径 = 原生像素)。
    s > SCAN_S_MAX 或像素过多时在 px_um / s_max 上渲染 (见 SCAN_S_MAX)。
    返回 dict(shape_native, px_native, s, native_pitch_um, ccd_axis, gauge) 或 None (该扫描仪没有原生采样模型)。
    ccd_axis: 图像里 CCD 线阵的方向 (0 = 竖直, 1 = 水平) —— 物理长度最接近横跨片带尺寸的那一边。"""
    if sc is None or not sc.native:
        return None
    gauge = gauge or film_gauge(fmt)
    g = sc.native[gauge]
    p_n = float(g["across_mm"]) * 1000.0 / float(g["lines"])
    h, w = shape[:2]
    s = min(px_um / p_n, SCAN_S_MAX, np.sqrt(SCAN_MAX_PIXELS / (h * w)))
    s = max(s, min(px_um / p_n, 1.0))
    if abs(s - 1.0) < 0.02:
        nh, nw, p = h, w, px_um
    else:
        nh, nw = max(int(round(h * s)), 2), max(int(round(w * s)), 2)
        p = px_um * h / nh
    ccd = 0 if abs(h * px_um / 1000.0 - g["across_mm"]) <= abs(w * px_um / 1000.0 - g["across_mm"]) else 1
    return dict(shape_native=(nh, nw), px_native=p, s=float(px_um / p), native_pitch_um=p_n, ccd_axis=ccd, gauge=gauge)


def _resample_matrix(n_in, n_out):
    """一维重采样矩阵 (n_out × n_in, 稀疏): 输出像素 j 的中心在输入坐标 (j + 0.5)·s − 0.5, s = n_in / n_out。
    缩小 (s > 1): 三角核按 s 拉宽 (半宽 s 个输入像素, = 带抗混叠的双线性, PIL 的 BILINEAR); 放大: 普通线性插值。
    边界按最近像素延拓。"""
    from scipy import sparse
    s = n_in / n_out
    w = max(s, 1.0)
    xc = (np.arange(n_out) + 0.5) * s - 0.5
    k = int(np.ceil(w)) + 1
    i0 = np.floor(xc - w).astype(np.int64) + 1
    idx = i0[:, None] + np.arange(2 * k + 1)[None, :]
    wt = np.clip(1.0 - np.abs(xc[:, None] - idx) / w, 0.0, None)
    idx = np.clip(idx, 0, n_in - 1)
    wt = wt / wt.sum(1, keepdims=True)
    rows = np.repeat(np.arange(n_out), idx.shape[1])
    return sparse.csr_matrix((wt.ravel(), (rows, idx.ravel())), shape=(n_out, n_in))


def resample2(a, shape_out):
    """(h, w[, c]) → shape_out 的可分离三角核重采样 (_resample_matrix)。"""
    h, w = a.shape[:2]
    H, W = shape_out
    if (h, w) == (H, W):
        return a
    Ay, Ax = _resample_matrix(h, H), _resample_matrix(w, W)
    flat = a.reshape(h, w, -1)
    out = np.empty((H, W, flat.shape[2]))
    for c in range(flat.shape[2]):
        out[..., c] = (Ax @ (Ay @ flat[..., c]).T).T
    return out.reshape((H, W) + a.shape[2:])


def _to_output_grid(rr, shape_native, shape_out, layers):
    """原生网格上的底片 (RenderResult) → 输出网格: 冲印店软件在处理后的 (对数 / 码值) 图像上缩放, 这里在密度域做
    (冲印店影调在小信号上对密度近乎线性; 与 T55e 的拟合同一口径)。平均密度、曝光 (取对数) 同样重采样。"""
    from .render import RenderResult
    shp = tuple(shape_native) + (layers,)
    D = resample2(np.asarray(rr.density, float).reshape(shp), shape_out)
    rr.density = None                                    # 原生数组用完即放 (24 MP 彩色的峰值内存)
    Dm = resample2(np.asarray(rr.mean, float).reshape(shp), shape_out)
    rr.mean = None
    lH = np.log(np.clip(np.asarray(rr.exposure, float).reshape(shp), 1e-30, None))
    rr.exposure = None
    lH = resample2(lH, shape_out)
    sq = (lambda a: a[..., 0]) if layers == 1 else (lambda a: a)
    return RenderResult(sq(D), sq(Dm), sq(np.exp(lH, out=lH)), rr.px_um, rr.method, rr.base_density)


GAUSS_IMPL = "continuous"      # "discrete" = scipy gaussian_filter (filmlab 引擎的做法; 只为对照, 见 validation.realscan)


def _gauss(a, s):
    """连续高斯点扩散 (σ = s 像素) 在像素网格上的作用: 频域乘 MTF exp(−2π² s² f²) (反射延拓 + FFT)。
    T51 的器件参数 (光学 σ_o、USM 半径) 是按连续传递函数 U(f) 定的; σ < 1 像素时离散采样的高斯核
    (scipy gaussian_filter) 几乎是恒等 (σ 0.35 像素时邻点权重 0.017), 与测得的传递不符, 所以这里在频域做。"""
    if s < 0.05:
        return a
    if GAUSS_IMPL == "discrete":
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(a, s, mode="reflect")
    from scipy import fft
    pad = int(np.ceil(4 * s)) + 2
    b = np.pad(a, pad, mode="reflect")
    fy = fft.fftfreq(b.shape[0])[:, None]; fx = fft.rfftfreq(b.shape[1])[None, :]
    G = np.exp(-2.0 * np.pi ** 2 * s * s * (fy * fy + fx * fx))
    out = fft.irfft2(fft.rfft2(b, workers=-1) * G, s=b.shape, workers=-1)
    return out[pad:pad + a.shape[0], pad:pad + a.shape[1]]


def _sharpen3(Y, kx, ky, ccd_axis=0):
    """可分离 3 抽头锐化 (输出像素): 输送方向 [−kx, 1 + 2kx, −kx], CCD 线阵方向 [−ky, 1 + 2ky, −ky] (直流增益 1)。
    ccd_axis: 图像里 CCD 线阵的方向 (0 = 竖直)。"""
    ka = {ccd_axis: ky, 1 - ccd_axis: kx}
    for ax in (0, 1):
        k = ka[ax]
        if k == 0:
            continue
        P = np.pad(Y, [(1, 1) if a == ax else (0, 0) for a in range(Y.ndim)], mode="edge")
        lo = P[:-2] if ax == 0 else P[:, :-2]
        hi = P[2:] if ax == 0 else P[:, 2:]
        Y = (1 + 2 * k) * Y - k * (lo + hi)
    return Y


def _usm(Y, sc, ccd_axis=0):
    if sc.sharpen_kx or sc.sharpen_ky:
        Y = _sharpen3(Y, sc.sharpen_kx, sc.sharpen_ky, ccd_axis)
    if sc.usm_amount <= 0 or sc.usm_sigma_px < 0.05:
        return Y
    d = Y - _gauss(Y, sc.usm_sigma_px)
    if sc.usm_threshold > 0:
        t = sc.usm_threshold / 255.0
        d = d * (1.0 - np.exp(-(d / t) ** 2))
    return Y + sc.usm_amount * d


def _box(a, k):
    from scipy.ndimage import uniform_filter
    return uniform_filter(a, k, mode="reflect") if k > 1.2 else a


def _software(enc, sc, mono, ccd_axis=0):
    """冲印店软件 (编码后的输出上; BT.601 亮度 / 色度): 亮度锐化 (USM 或 T55e 的可分离 3 抽头), 色度降噪 (彩色)。"""
    if mono or enc.ndim == 2:
        Y = enc if enc.ndim == 2 else enc.mean(-1)
        Y = _usm(Y, sc, ccd_axis)
        return Y if enc.ndim == 2 else np.repeat(Y[..., None], 3, -1)
    ycc = enc @ _YCC.T
    ycc[..., 0] = _usm(ycc[..., 0], sc, ccd_axis)
    if sc.chroma_sigma_px >= 0.05:
        for k in (1, 2):
            low = _gauss(ycc[..., k], sc.chroma_sigma_px)
            if sc.chroma_threshold > 0:
                d = ycc[..., k] - low
                ycc[..., k] = low + d * (1.0 - np.exp(-(d / (sc.chroma_threshold / 255.0)) ** 2))
            else:
                ycc[..., k] = low
    return ycc @ np.linalg.inv(_YCC).T


# ----------------------------------------------------------------- 主函数 --
@dataclass
class GrainImage:
    linear: np.ndarray          # 线性 sRGB 输出 (h, w, 3) 或 (h, w)
    info: dict = field(default_factory=dict)
    render: object = None       # render.RenderResult (底片面)

    @property
    def encoded(self):
        return srgb_encode(np.clip(self.linear, 0, 1))

    def save(self, path, bits=8):
        save_image(path, self.linear, bits)


@dataclass
class Exposure:
    """apply_to_image 的前半: 照片 → 胶片上的曝光。"""
    rgb: np.ndarray             # 线性照片 (h, w, 3)
    H: np.ndarray               # 每像素曝光 (lux·s): 黑白 (h, w), 彩色 (h, w, 3)
    px_um: float
    emulsion: object
    format: object
    mono_input: bool
    info: dict


def expose(image, film="kodak_trix_400", developer=None, grain_scale=1.0, format="35mm", output_px=None, exposure_ev=0.0,
           metering="fixed", filter=None, filter_compensate=True, encoding="auto"):
    """步骤 0–2: 读入、几何、曝光 (见模块说明)。"""
    lin, meta = load_image(image, encoding)
    if output_px is not None and output_px != max(lin.shape[:2]):
        lin = resize_long(lin, int(output_px))
    mono_in = lin.ndim == 2
    rgb = np.repeat(lin[..., None], 3, -1) if mono_in else lin
    h, w = rgb.shape[:2]
    px_um = format_long_mm(format) * 1000.0 / max(h, w)
    f = _film(film)
    em = derive(f, developer, grain_scale)
    resp, rinfo = layer_response(f, filter, filter_compensate)
    E = resp(rgb)                                         # (h, w, k) 相对曝光 (中灰 0.18)
    if metering == "average":
        key = float(np.exp(np.mean(np.log(np.clip(E.mean(-1), 1e-4, None)))))
    elif metering == "fixed":
        key = GREY
    else:
        raise ValueError("metering: 'fixed' / 'average'")
    H_grey = 10.0 / f.ei                                  # 18% 灰的曝光 (lux·s)
    scale = H_grey / key * 2.0 ** float(exposure_ev)
    H = (E[..., 0] if f.is_bw else E) * scale
    info = dict(film=f.key, film_name=f.name, developer=em.developer.key, grain_scale=em.grain_scale, format=format,
                px_um=px_um, shape=(h, w), exposure_ev=exposure_ev, metering=metering, metering_key=key, H_grey_lux_s=H_grey,
                filter=rinfo, emulsion=em.describe(), mono_input=mono_in, input=meta)
    return Exposure(rgb, H, px_um, em, format, mono_in, info)


def finish(ex, rr, mode="grain", scanner_key=None):
    """步骤 4–5: 底片 (render 的结果) → 输出 (mode "grain" / "full"; 可选冲印店扫描)。返回 (线性输出, 信息)。"""
    em, rgb = ex.emulsion, ex.rgb
    f = em.film
    h, w = rgb.shape[:2]
    tabs = _curve_tables(em)
    sc = get_scanner(scanner_key)
    grid = ex.info.get("scan_grid")
    ccd = grid["ccd_axis"] if grid else 0
    lab_px = None if (sc is None or grid or sc.native) else sc.pitch_um * format_long_mm(ex.format) / 36.0 / ex.px_um   # 冲印店像素间距 / 渲染像素
    layers = 1 if f.is_bw else 3
    D = rr.density.reshape(h, w, layers)
    Dm = rr.mean.reshape(h, w, layers)
    logH = np.log10(np.clip(rr.exposure.reshape(h, w, layers), 1e-30, None))
    lgrey = np.log10(ex.info["H_grey_lux_s"])
    info = dict(mode=mode, method=rr.method, scanner=None if sc is None else sc.name, lab_px=lab_px)
    if mode == "full":
        out = np.empty((h, w, layers))
        info["scan"] = []
        for c in range(layers):
            Dc = D[..., c]
            if lab_px is not None:                              # 扫描光学 (透射率上) + 孔径 (T51 的口径)
                T = _box(_gauss(10.0 ** -Dc, sc.optics_px), lab_px)
                Dc = -np.log10(np.clip(T, 1e-12, None))
            tab = tabs[c]
            if em.path == "dye" and em.curve.design == "reversal":
                dmin = float(tab[1].min())
                out[..., c] = 10.0 ** -(Dc - dmin)
                info["scan"].append(dict(kind="reversal_transmittance", D_min=dmin))
            else:
                Dg = float(np.interp(lgrey, tab[0], tab[1]))
                gbar = _avg_gradient(tab, lgrey)
                out[..., c] = GREY * 10.0 ** ((Dc - Dg) / gbar)
                info["scan"].append(dict(kind="negative_inversion", D_grey=Dg, gamma_bar=gbar))
        res = np.repeat(out, 3, -1) if layers == 1 else out
        if sc is not None:
            res = srgb_decode(_software(srgb_encode(res), sc, mono=f.is_bw, ccd_axis=ccd))
        if ex.mono_input and layers == 1:
            res = res[..., 0]
    elif mode == "grain":
        lg10 = np.empty((h, w, layers))                    # 输出的 log10 相对起伏 δ/γ_eff
        info["slopes"] = []
        for c in range(layers):
            g_eff, _ = _slope_at(tabs[c], logH[..., c])
            lg10[..., c] = (D[..., c] - Dm[..., c]) / g_eff
            info["slopes"].append(dict(gamma_speed_point=tabs[c][3], gamma_grey=float(np.interp(lgrey, tabs[c][0], tabs[c][2]))))
        if lab_px is not None:
            for c in range(layers):
                lg10[..., c] = _box(_gauss(lg10[..., c], sc.optics_px), lab_px)
        for c in range(layers):                            # 指数映射的二阶偏移按同一曝光的系综扣除 (见模块说明)
            lg10[..., c] -= _exp_bias(lg10[..., c], logH[..., c])
        fac = 10.0 ** lg10
        del lg10
        info["mean_rel_shift"] = [float(np.mean(fac[..., c]) - 1.0) for c in range(layers)]
        o = rgb * fac                                      # 黑白: 三通道同一相对起伏 (亮度); 彩色: 每层进对应通道
        del fac
        if sc is not None:
            ev = srgb_encode(rgb)
            o = srgb_decode(ev + _software(srgb_encode(o) - ev, sc, mono=f.is_bw, ccd_axis=ccd))
        res = o[..., 0] if (ex.mono_input and layers == 1) else o
    else:
        raise ValueError("mode: 'grain' / 'full'")
    return res, info


LAB_SCANNER = {"rpl_noritsu": "noritsu_rpl"}             # 冲印店影调 → 同一家冲印店的空间传递 (scanners.json)
LUMA_709 = np.array([0.2126, 0.7152, 0.0722])          # 线性 sRGB 的亮度 (黑白颗粒的密度由它定)
MEAN_FIX_BIN = 0.02            # 平均修正按 u (底片密度, 相对锚点) 分档的宽度
EPS = 1e-6


def lab_grain_exposure(ex, lab="rpl_noritsu"):
    """grain 模式 (T55b): 照片码值 → 冲印店影调的反函数 → 底片密度 D → 该片特性曲线的反函数 → 曝光 H。
    黑白: 照片的线性亮度 Y (BT.709) 的码值走黑白影调; 彩色负片: 每通道码值走该通道的影调; 反转片: 正片透射率 v = 10^−(D − D_min)。
    锚点: 照片码值 118 (sRGB 中灰) ↔ 该片在测光中灰 (H = 10/EI × 2^EV) 的平均密度。返回 (H, 辅助量 dict)。"""
    from .labtone import tone
    from .render import exposure_from_density
    em, rgb = ex.emulsion, ex.rgb
    f = em.film
    H_ref = ex.info["H_grey_lux_s"] * 2.0 ** float(ex.info["exposure_ev"])
    if f.is_bw:
        (T,) = tone(lab, "bw")
        Y = rgb @ LUMA_709
        code = 255.0 * srgb_encode(np.clip(Y, 0.0, 1.0))
        u = T.u_of(code)
        D_ref = float(em.mean_density(H_ref))
        H = exposure_from_density(em, D_ref + u)
        return H, dict(kind="bw", tones=(T,), u=u[..., None], base=np.clip(Y, 0, 1)[..., None], D_ref=[D_ref], Y=np.clip(Y, 0, 1))
    if em.curve.design == "reversal":
        Dmin = em.curve.density_curves.min(0)
        D = Dmin[None, None, :] - np.log10(np.clip(rgb, 1e-4, 1.0))
        return exposure_from_density(em, D), dict(kind="reversal", tones=None, D_min=Dmin.tolist())
    T3 = tone(lab, "colour")
    code = 255.0 * srgb_encode(np.clip(rgb, 0.0, 1.0))
    u = np.stack([T3[c].u_of(code[..., c]) for c in range(3)], -1)
    D_ref = em.mean_density(np.full(3, H_ref))
    H = exposure_from_density(em, D_ref[None, None, :] + u)
    return H, dict(kind="colour", tones=T3, u=u, base=np.clip(rgb, 0, 1), D_ref=np.asarray(D_ref).tolist())


def _mean_fix(y1, b, delta, g):
    """二阶 (影调曲率 / 指数映射) 的平均偏移按同一底片密度的系综扣除: 在 b (u, 或反转片的密度) 的每档 (0.02) 内,
    期望 E[y1 | b] 改成 g(b + E[δ | b]) —— 只剩物理的 −log⟨T⟩ (E[δ] > 0) 经影调的传递。与 _exp_bias 同一分档合并规则。"""
    k = np.floor(b / MEAN_FIX_BIN).astype(np.int64)
    idx = (k - k.min()).ravel()
    n = int(idx.max()) + 1
    N = np.bincount(idx, minlength=n).astype(float)
    grp = np.zeros(n, np.int64); gi, acc = 0, 0.0
    nz = np.nonzero(N)[0]
    for j in nz:
        grp[j] = gi; acc += N[j]
        if acc >= EXP_BIAS_MIN:
            gi, acc = gi + 1, 0.0
    if acc > 0 and gi > 0:
        grp[nz[grp[nz] == gi]] = gi - 1
    gidx = grp[idx]
    ng = int(gidx.max()) + 1
    GN = np.bincount(gidx, minlength=ng).astype(float)
    Ed = np.bincount(gidx, weights=delta.ravel(), minlength=ng) / GN
    t = g(b.ravel() + Ed[gidx])
    r = np.where(t > EPS, y1.ravel() / np.maximum(t, EPS), 1.0)
    R = np.bincount(gidx, weights=r, minlength=ng) / GN
    return y1 / np.where(R > 0, R, 1.0)[gidx].reshape(y1.shape)


def lab_anchor(ex, Dm, timing="auto"):
    """冲印店定光: 每层底片密度的锚点 m (该密度印成码值 118)。timing "auto": 无颗粒密度的整帧平均 (冲印店自动定光,
    同 filmlab T47 的预扫; 彩色每层各自, 即自动色彩平衡); "fixed": 测光中灰 (H = 10/EI) 的平均密度。full 与 grain 模式共用。"""
    em = ex.emulsion
    layers = Dm.shape[-1]
    if timing == "auto":
        return Dm.reshape(-1, layers).mean(0)
    if timing == "fixed":
        Hg = ex.info["H_grey_lux_s"]
        return np.atleast_1d(em.mean_density(Hg if em.film.is_bw else np.full(3, Hg)))
    raise ValueError("timing: 'auto' / 'fixed'")


def finish_lab(ex, rr, aux=None, mode="grain", lab="rpl_noritsu", timing="auto", scanner_key=None, release=False):
    """T55b 的步骤 4–5 (冲印店实测影调)。返回 (线性输出, 信息)。见模块说明。"""
    from .labtone import tone
    em, rgb = ex.emulsion, ex.rgb
    f = em.film
    h, w = rgb.shape[:2]
    sc = get_scanner(scanner_key)
    grid = ex.info.get("scan_grid")
    ccd = grid["ccd_axis"] if grid else 0
    lab_px = None if (sc is None or grid or sc.native) else sc.pitch_um * format_long_mm(ex.format) / 36.0 / ex.px_um
    layers = 1 if f.is_bw else 3
    D = rr.density.reshape(h, w, layers)
    Dm = rr.mean.reshape(h, w, layers)
    info = dict(mode=mode, lab=lab, method=rr.method, scanner=None if sc is None else sc.name, lab_px=lab_px)
    reversal = em.path == "dye" and em.curve.design == "reversal"
    dec = lambda c: srgb_decode(np.clip(c, 0.0, 255.0) / 255.0)
    if mode == "full":
        if lab_px is not None:                                  # 扫描光学 (透射率上) + 孔径 (T51 的口径)
            D = np.stack([-np.log10(np.clip(_box(_gauss(10.0 ** -D[..., c], sc.optics_px), lab_px), 1e-12, None))
                          for c in range(layers)], -1)
        if reversal:
            Dmin = em.curve.density_curves.min(0)
            out = 10.0 ** -(D - Dmin[None, None, :])
            info["scan"] = dict(kind="reversal_transmittance", D_min=Dmin.tolist())
        else:
            T = tone(lab, "bw" if f.is_bw else "colour")
            m = lab_anchor(ex, Dm, timing)
            code = np.stack([T[c].code_of(D[..., c] - m[c]) for c in range(layers)], -1)
            out = dec(code)
            info["scan"] = dict(kind="lab_tone", timing=timing, D_anchor=np.asarray(m).tolist())
        res = np.repeat(out, 3, -1) if layers == 1 else out
        if sc is not None:
            res = srgb_decode(_software(srgb_encode(res), sc, mono=f.is_bw, ccd_axis=ccd))
        if ex.mono_input and layers == 1:
            res = res[..., 0]
        return res, info
    if mode != "grain":
        raise ValueError("mode: 'grain' / 'full'")
    # grain 模式作用在照片显示出来的样子上: 线性值 > 1 (HDR 输入) 显示为白, 按 1 处理 (否则中性限制会把 HDR 高光压暗、改变影调)
    rgb = np.clip(rgb, 0.0, 1.0)
    delta = D - Dm
    if release:                                                          # 调用者不保留渲染结果: 逐像素密度用完即放
        D = None
        rr.density = None
        rr.exposure = None
    if lab_px is not None:
        delta = np.stack([_box(_gauss(delta[..., c], sc.optics_px), lab_px) for c in range(layers)], -1)
    film_density = aux is None                                           # T55d: 底片密度来自该片自己的光谱曝光 (expose)
    if film_density and not reversal:
        T = tone(lab, "bw" if f.is_bw else "colour")
        m = lab_anchor(ex, Dm, timing)
        aux = dict(kind="film", tones=T, u=Dm - m[None, None, :], base=np.clip(rgb, 0.0, 1.0) @ LUMA_709 if f.is_bw else None,
                   anchor=np.asarray(m).tolist())
        info["grain_density"] = "film"; info["anchor_D"] = aux["anchor"]; info["timing"] = timing
    elif not reversal:
        info["grain_density"] = "photo"
    if reversal:
        Dmin = em.curve.density_curves.min(0) if film_density else np.asarray(aux["D_min"])
        b = Dm                                                           # 正片: v = 10^−(D − D_min), 斜率 −1 (透射率本身)
        o = np.empty_like(rgb)
        for c in range(3):
            g = lambda x, c=c: 10.0 ** -(x - Dmin[c])
            y1 = _mean_fix(g(b[..., c] + delta[..., c]), b[..., c], delta[..., c], g)
            o[..., c] = rgb[..., c] * y1 / np.maximum(g(b[..., c]), EPS)
        info["mean_rel_shift"] = [float(np.mean(o[..., c]) / max(np.mean(rgb[..., c]), EPS) - 1) for c in range(3)]
    else:
        T = aux["tones"]
        fac = np.empty((h, w, layers))
        for c in range(layers):
            u = aux["u"][..., c]
            g = lambda x, c=c: dec(T[c].code_of(x))
            y1 = _mean_fix(g(u + delta[..., c]), u, delta[..., c], g)
            y0 = g(u)
            fac[..., c] = np.where(y0 > EPS, y1 / np.maximum(y0, EPS), 1.0)
            info.setdefault("slope_code_per_D", []).append(float(np.median(T[c].slope(u))))
        if layers == 1:
            # 黑白银粒 = 中性密度调制: 亮度 Y' = Y·k (k 对三通道相同, 色相不变)。亮的颗粒 = 该处底片密度低于平均
            # (显影晶体少于平均), 在正像上比平均亮; Y' 本身不超过冲印店影调的白 (≤ 1)。某个通道会超过正像的白 (1.0) 时,
            # 不截断亮度 (逐通道截断会改色相; 只把 k 压到 1/最亮通道会砍掉亮的颗粒、留下黑点: T55c 的分布检验),
            # 而是保持 Y' 与色相、把色度向白收 (o = Y' + t·(rgb·k − Y'), t 使最亮通道 = 1): 同相纸上接近纸白时颜色变淡。
            k = fac[..., 0]
            Yl = rgb @ LUMA_709                                          # 显示出来的照片 (已截到 0–1) 的亮度
            Yp = np.clip(Yl * k, 0.0, 1.0)
            if sc is not None:                                           # 冲印店软件 (USM) 作用于亮度的变化 (编码域), 仍是中性的
                e0 = srgb_encode(Yl)
                Yp = np.clip(srgb_decode(np.clip(e0 + _software(srgb_encode(Yp) - e0, sc, mono=True, ccd_axis=ccd), 0.0, 1.0)), 0.0, 1.0)
                k = np.where(Yl > EPS, Yp / np.maximum(Yl, EPS), 1.0)
            o = rgb * k[..., None]
            mx = o.max(-1)
            over = mx > 1.0
            if over.any():
                t = np.clip((1.0 - Yp[over]) / np.maximum(mx[over] - Yp[over], EPS), 0.0, 1.0)
                o[over] = Yp[over][:, None] + t[:, None] * (o[over] - Yp[over][:, None])
            dark = (aux["base"][..., 0] <= EPS) if aux["kind"] == "bw" else np.zeros(Yl.shape, bool)   # T55c: 全黑像素的中性颗粒
            if dark.any():
                yv = dec(T[0].code_of(aux["u"][..., 0] + delta[..., 0]))
                o[dark] = np.minimum(yv[dark], 1.0)[:, None]
            info["desaturated_fraction"] = float(np.mean(over))
        else:
            o = rgb * fac                                                # 彩色: 每层颗粒进对应通道, 码值不超出冲印店影调的 0–255
        info["mean_rel_shift"] = [float(np.mean(fac[..., c]) - 1.0) for c in range(layers)]
        del fac
    if sc is not None and not f.is_bw:                                   # 彩色: 冲印店软件作用于编码后的差 (亮度 USM / 色度降噪)
        ev = srgb_encode(rgb)
        o = srgb_decode(np.clip(ev + _software(srgb_encode(o) - ev, sc, mono=False, ccd_axis=ccd), 0.0, 1.0))
    res = o[..., 0] if (ex.mono_input and layers == 1) else o
    return res, info


def apply_to_image(image, film="kodak_trix_400", developer=None, grain_scale=1.0, format="35mm", output_px=None,
                   mode="grain", exposure_ev=0.0, metering="fixed", filter=None, filter_compensate=True, scanner="auto",
                   lab="rpl_noritsu", timing="auto", grain_density="film", seed=0, method="auto", encoding="auto", workers=1,
                   keep_render=False, film_gauge=None):
    """给照片加银盐颗粒。返回 GrainImage (linear, info)。参数:
    image: 路径或数组 (整数 = sRGB 编码; 浮点 = 线性, encoding="srgb" 时先解码)。
    film / developer: 胶片与显影液 (films.films() / films.developers(); None = 原生冲洗)。grain_scale: 颗粒粗细 k (0.5–3)。
    format: 画幅 ("35mm", "half", "110", "645", "6x7", … 或长边 mm)。output_px: 输出长边像素 (None = 原尺寸)。
    mode: "grain" (只加颗粒, 默认) / "full" (整卷胶片: 该片的影调 + 颗粒)。exposure_ev: 曝光补偿 (档)。
    lab: 冲印店影调 (实测, data/lab_tone.json): "rpl_noritsu" (默认); None = T55 的旧做法 (场景线性 + 直线反转, 只为对照)。
    grain_density: grain 模式的底片密度来源: "film" (默认, T55d: 该片自己的光谱曝光, 与 full 模式同一底片) /
                   "photo" (T55b/c: 由照片码值经冲印店影调反推, 只为对照)。
    timing: full 与 grain 模式的冲印店定光: "auto" (无颗粒密度的整帧平均 → 中灰码值) / "fixed" (测光中灰 → 中灰码值)。
    metering: "fixed" (线性值 0.18 = 中灰) / "average" (图像的对数平均亮度 = 中灰, 相机平均测光)。
    filter: Wratten 号 ("8", "11", "15", "25", "47", "58" …; 黑白常用; 只影响 full 模式的曝光); filter_compensate: 按中性灰补偿。
    scanner: "auto" (默认: 与 lab 同一家冲印店 —— rpl_noritsu → "noritsu_rpl" 的空间传递; lab=None 时不加) / None (不加) /
             "noritsu_rpl" / "frontier_epfl" (T51 实测的冲印店空间传递: 光学、孔径、USM、色度降噪)。
    method: 黑白颗粒的渲染法 "auto" / "explicit" / "closed_form" (见 render)。
    film_gauge: 片带 "135" / "120" (None = 由画幅推断; 线阵扫描仪的原生采样间距按片带, 见 scan_grid)。"""
    t0 = time.time()
    if scanner == "auto":                                        # 一家完整的冲印店: 影调与空间传递来自同一家
        scanner = LAB_SCANNER.get(lab)
    ex = expose(image, film, developer, grain_scale, format, output_px, exposure_ev, metering, filter, filter_compensate, encoding)
    aux = None
    H = ex.H
    if lab is not None and mode == "grain" and grain_density == "photo":     # T55b/c 的做法 (只为对照)
        H, aux = lab_grain_exposure(ex, lab)
    elif grain_density not in ("film", "photo"):
        raise ValueError("grain_density: 'film' / 'photo'")
    t1 = time.time()
    sc = get_scanner(scanner)
    grid = scan_grid(ex.rgb.shape, ex.px_um, format, sc, film_gauge)
    if grid is not None:                                         # T55e: 线阵 CCD 在原生间距上采样 (像素孔径 = 原生像素)
        ex.info["scan_grid"] = grid
        shp = tuple(grid["shape_native"])
        Hn = resample2(np.log(np.clip(H, 1e-30, None)), shp) if shp != ex.rgb.shape[:2] else np.log(np.clip(H, 1e-30, None))
        Hn = np.exp(Hn, out=Hn)
        rr = render(ex.emulsion, exposure=Hn, px_um=grid["px_native"], seed=seed, method=method, workers=workers)
        del Hn
        layers = 1 if ex.emulsion.film.is_bw else 3
        rr_native_method = rr.method
        if shp != ex.rgb.shape[:2]:
            rr = _to_output_grid(rr, shp, ex.rgb.shape[:2], layers)
        rr.method = rr_native_method
    else:
        rr = render(ex.emulsion, exposure=H, px_um=ex.px_um, seed=seed, method=method, workers=workers)
    t2 = time.time()
    if lab is None:
        res, info2 = finish(ex, rr, mode, scanner)
    else:
        res, info2 = finish_lab(ex, rr, aux, mode, lab, timing, scanner, release=not keep_render)
    info = dict(ex.info, **info2)
    info["seconds"] = dict(exposure=t1 - t0, grain=t2 - t1, finish=time.time() - t2, total=time.time() - t0)
    return GrainImage(res, info, rr if keep_render else None)
