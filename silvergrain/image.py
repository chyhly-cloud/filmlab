"""给任意照片加银盐颗粒: apply_to_image。每一步都是有物理定义的步骤 (见 README "apply_to_image 的物理定义")。

  0. 读入: sRGB 编码的 8 / 16 位 JPEG / PNG / TIFF, 或线性浮点数组; 彩色或单色; 任意尺寸与长宽比。
  1. 几何: 画幅 (默认 35 mm) 长边对图像长边 → 每像素的胶片边长 px_um = 画幅长边 × 1000 / 图像长边像素数
     (output_px 给定时, 先在线性光里按面积平均缩到该长边, 即以输出像素为扫描孔径)。
  2. 曝光: 线性值当作景物反射率 (中灰 0.18), 经 RGB → 光谱上采样、设计光源、可选 Wratten 滤镜, 按胶片每层自己的光谱感光度
     积分 (spectra.layer_response; 黑白 = 全色一条), 中性灰 0.18 落在 H = 10/EI (ISO 测光约定: 18% 灰卡的曝光;
     净密度 0.1 处 H = 0.8/EI, 所以中灰在速度点之上 log10(12.5) = 1.10), 再乘 2^EV (曝光补偿)。
  3. 底片: 该片 (+ 显影液, 颗粒粗细 k) 的晶体模型给出平均密度 (闭式特性曲线) 与逐像素颗粒密度 (render)。
  4a. mode="full" (整卷胶片): 底片密度 → 透射率 → 简单扫描成正像:
        负片 (黑白 / 彩色, 每层): P = 0.18 · 10^((D − D_g)/γ̄), D_g = 18% 灰的平均密度, γ̄ = 该片在中灰 ±2 档的平均斜率;
          即以胶片自己的平均反差作直线反转、中灰对齐 (不加冲印店的影调曲线、不做颜色校正; 彩色片每层只在自己的通道里读)。
        反转片: P = 10^−(D − D_min) (透射率本身, 白点 = 片基)。
      影调变成该片的影调 (趾部、肩部、反差)。
  4b. mode="grain" (只加颗粒, 默认): 保留照片自己的影调与颜色, 只加物理算出的颗粒:
        把每个像素看成 "该片在这一曝光下的底片, 经扫描与影调映射得到了照片的这个值", 映射 v ↔ D 就是该片特性曲线的反函数
        (log10 v 与 log10 H 只差常数)。颗粒使密度偏离 δ = D_px − D̄ (显式银粒 / 染料云在该密度处的起伏, 含空间结构),
        经反映射的局部斜率传到输出: δ log10 v = δ / γ(H), γ = dD̄/dlog10 H, 即 v' = v · 10^(δ/γ_eff − b(H))
        (斜率在起伏范围内取常数时反映射的精确形式, 恒为正; 一阶为 v·(1 + ln10·δ/γ))。b(H) = log10 E[10^(δ/γ)] − E[δ/γ]
        是指数映射自己的二阶偏移, 在同一曝光的全部像素上算出 (_exp_bias), 扣掉后输出的期望与一阶传递相同。
        γ_eff = max(γ, γ_s), γ_s = 速度点 (净 0.1) 的斜率 —— 速度点以下胶片不再记录可用的影像信息 (ISO 感光度的定义),
        扫描的黑位设在这里, 所以影调斜率不再变小 (否则 γ → 0 时颗粒发散)。
        输出平均 = 照片, 只差物理的 −log⟨T⟩: E[δ] = E[D_px] − D̄ > 0 (像素内平均透射率再取对数, T53 2.6 节),
        即 v·10^(E[δ]/γ); 见 info["mean_rel_shift"]。
        彩色片: 每层染料颗粒进对应通道 (R 层 → R 通道 …); 黑白片: 只有亮度 (三通道同一个相对起伏, 色相不变)。
  5. 冲印店扫描 (可选, scanner="noritsu_rpl" / "frontier_epfl"; T51 实测): 光学高斯 σ (输出像素, 透射率上)、传感器孔径
     (冲印店像素间距按画幅长边 / 36 换算, 比渲染像素大 1.2 倍以上时加方盒)、亮度 USM、色度降噪 (彩色)。
     mode="grain" 时只作用于颗粒部分 (照片本身已有自己的成像与锐化): 光学与孔径作用于 δ, USM / 色度降噪作用于编码后的
     (输出 − 照片) 差 (小信号线性化)。
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
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def srgb_decode(v):
    v = np.clip(np.asarray(v, float), 0.0, None)
    return np.where(v <= 0.04045, v / 12.92, np.power((v + 0.055) / 1.055, 2.4))


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


def get_scanner(key):
    if key is None or isinstance(key, Scanner):
        return key
    return Scanner(**db.scanners()["labs"][key])


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


def _usm(Y, sc):
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


def _software(enc, sc, mono):
    """冲印店软件 (编码后的输出上; BT.601 亮度 / 色度): 亮度 USM, 色度降噪 (彩色)。"""
    if mono or enc.ndim == 2:
        Y = enc if enc.ndim == 2 else enc.mean(-1)
        Y = _usm(Y, sc)
        return Y if enc.ndim == 2 else np.repeat(Y[..., None], 3, -1)
    ycc = enc @ _YCC.T
    ycc[..., 0] = _usm(ycc[..., 0], sc)
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
    lab_px = None if sc is None else sc.pitch_um * format_long_mm(ex.format) / 36.0 / ex.px_um   # 冲印店像素间距 / 渲染像素
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
            if sc is not None:                                           # 扫描光学 (透射率上) + 孔径
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
            res = srgb_decode(_software(srgb_encode(res), sc, mono=f.is_bw))
        if ex.mono_input and layers == 1:
            res = res[..., 0]
    elif mode == "grain":
        lg10 = np.empty((h, w, layers))                    # 输出的 log10 相对起伏 δ/γ_eff
        info["slopes"] = []
        for c in range(layers):
            g_eff, _ = _slope_at(tabs[c], logH[..., c])
            lg10[..., c] = (D[..., c] - Dm[..., c]) / g_eff
            info["slopes"].append(dict(gamma_speed_point=tabs[c][3], gamma_grey=float(np.interp(lgrey, tabs[c][0], tabs[c][2]))))
        if sc is not None:
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
            o = srgb_decode(ev + _software(srgb_encode(o) - ev, sc, mono=f.is_bw))
        res = o[..., 0] if (ex.mono_input and layers == 1) else o
    else:
        raise ValueError("mode: 'grain' / 'full'")
    return res, info


def apply_to_image(image, film="kodak_trix_400", developer=None, grain_scale=1.0, format="35mm", output_px=None,
                   mode="grain", exposure_ev=0.0, metering="fixed", filter=None, filter_compensate=True, scanner=None,
                   seed=0, method="auto", encoding="auto", workers=1, keep_render=False):
    """给照片加银盐颗粒。返回 GrainImage (linear, info)。参数:
    image: 路径或数组 (整数 = sRGB 编码; 浮点 = 线性, encoding="srgb" 时先解码)。
    film / developer: 胶片与显影液 (films.films() / films.developers(); None = 原生冲洗)。grain_scale: 颗粒粗细 k (0.5–3)。
    format: 画幅 ("35mm", "half", "110", "645", "6x7", … 或长边 mm)。output_px: 输出长边像素 (None = 原尺寸)。
    mode: "grain" (只加颗粒, 默认) / "full" (整卷胶片: 该片的影调 + 颗粒)。exposure_ev: 曝光补偿 (档)。
    metering: "fixed" (线性值 0.18 = 中灰) / "average" (图像的对数平均亮度 = 中灰, 相机平均测光)。
    filter: Wratten 号 ("8", "11", "15", "25", "47", "58" …; 黑白常用); filter_compensate: 按中性灰补偿滤镜系数。
    scanner: None / "noritsu_rpl" / "frontier_epfl" (T51 实测的冲印店空间传递)。
    method: 黑白颗粒的渲染法 "auto" / "explicit" / "closed_form" (见 render)。"""
    t0 = time.time()
    ex = expose(image, film, developer, grain_scale, format, output_px, exposure_ev, metering, filter, filter_compensate, encoding)
    t1 = time.time()
    rr = render(ex.emulsion, exposure=ex.H, px_um=ex.px_um, seed=seed, method=method, workers=workers)
    t2 = time.time()
    res, info2 = finish(ex, rr, mode, scanner)
    info = dict(ex.info, **info2)
    info["seconds"] = dict(exposure=t1 - t0, grain=t2 - t1, finish=time.time() - t2, total=time.time() - t0)
    return GrainImage(res, info, rr if keep_render else None)
