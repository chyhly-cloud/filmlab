"""色彩科学系统的验证。全部基于厂商数据与冲洗行业标准, 不需要实拍胶片。

    python -m pytest tests -q
"""
import numpy as np
import pytest

from filmlab import chemistry, spectral
from filmlab.densitometry import step_wedge
from filmlab.engine import Engine, Params
from filmlab.materials import bw_film, emulsion
from filmlab.recipes import RECIPES, recipe
from filmlab.stocks import list_stocks, load_stock

RAMP = np.repeat((0.184 * 2.0 ** np.arange(-4, 5, 1.0))[None, :, None], 3, 2)
FILMS = [k for k in list_stocks()["negative"] + list_stocks()["positive"] if not k.startswith("kodak_23")]


def chroma(lin):
    import colour
    lab = colour.XYZ_to_Lab(colour.sRGB_to_XYZ(np.clip(lin, 1e-6, 1), apply_cctf_decoding=False))
    return np.hypot(lab[..., 1], lab[..., 2])


# ----------------------------------------------------- 材料 / 化学 --
@pytest.mark.parametrize("key", FILMS)
def test_nominal_process_reproduces_datasheet(key):
    """原生工艺、标准时间冲洗的中性灰阶楔, 必须重现 datasheet 特性曲线 (含 DIR 反推的自洽性)。"""
    e = emulsion(key)
    x = e.log_exposure
    logH = np.repeat(x[40:220, None], 3, 1)
    st = chemistry.run(e, chemistry.PROCESSES[e.native_process], logH)
    expect = e.stock.density_curves[40:220]
    assert np.max(np.abs(st.dye - expect)) < 0.02


def test_push_kinetics_vs_kodak_portra_800():
    """显影动力学 vs Kodak Portra 800 推一档 / 推两档实测 (C-41 3:45 / 4:15)。
    数据横轴相对推冲后的标称感光度: 推 n 档的曲线 = 欠曝 n 档 + 推 n 档。"""
    e = emulsion("kodak_portra_800")
    x = e.log_exposure
    m = (x > -2.5) & (x < 2.2)
    for stops, key in [(1, "kodak_portra_800_push1"), (2, "kodak_portra_800_push2")]:
        proc = chemistry.PROCESSES["C-41"].push(stops)
        st = chemistry.run(e, proc, np.repeat(x[:, None] - 0.301 * stops, 3, 1))
        rms_model = np.sqrt(np.mean((st.dye - load_stock(key).density_curves)[m] ** 2))
        rms_none = np.sqrt(np.mean((e.S_nom - load_stock(key).density_curves)[m] ** 2))
        assert rms_model < 0.05 and rms_model < 0.45 * rms_none


def test_more_development_never_lowers_density():
    """同一曝光下, 显影更充分 (推冲 / 更活泼的显影液) 只会显影出更多的银和染料。"""
    e = emulsion("kodak_vision3_500t")
    x = np.repeat(e.log_exposure[:, None], 3, 1)
    base = chemistry.run(e, chemistry.PROCESSES["ECN-2"], x).dye
    more = chemistry.run(e, chemistry.PROCESSES["ECN-2"].push(1), x).dye
    assert np.all(more >= base - 1e-3)


def test_e6_push_gains_one_stop():
    """Kodak 定义: E-6 第一显影 +2 分钟 = 提升一档。欠曝一档 + 推一档, 中灰应回到原位。"""
    def gray_Y(**kw):
        out = Engine(Params(film="fujifilm_provia_100f", grain=0, **kw)).render(np.full((1, 1, 3), 0.184))
        return out[0, 0] @ np.array([0.2126, 0.7152, 0.0722])
    assert gray_Y(push=1, exposure_ev=-1) == pytest.approx(gray_Y(), rel=0.12)


def test_bw_film_in_c41_is_blank():
    """黑白胶片进 C-41: 没有成色剂 → 没有染料, 银被漂白 → 空白片 (现实中也是如此)。"""
    st = chemistry.run(bw_film(), chemistry.PROCESSES["C-41"], np.linspace(-2, 3, 20)[:, None] * np.ones(3))
    assert np.all(st.dye == 0) and np.allclose(st.silver, 0)


def test_bleach_bypass_retains_silver():
    proc = chemistry.PROCESSES["C-41"].with_bleach(0.0)
    e = emulsion("kodak_vision3_250d")
    logH = np.linspace(-2, 2, 20)[:, None] * np.ones(3)
    st = chemistry.run(e, proc, logH)
    assert np.all(np.diff(st.silver[:, 1]) >= 0) and st.silver.max() > 0.5


@pytest.mark.parametrize("film,process,polarity", [
    ("fujifilm_velvia_100", "C-41", "negative"),   # 正片负冲 → 负像
    ("kodak_portra_400", "E-6", "positive"),       # 负片正冲 → 正像
    ("kodak_portra_400", "C-41", "negative"),
    ("kodak_ektachrome_100", "E-6", "positive"),
])
def test_polarity_emerges_from_chemistry(film, process, polarity):
    assert Engine(Params(film=film, process=process, grain=0)).polarity == polarity


# ------------------------------------------------------------- 系统 --
@pytest.mark.parametrize("name", [r for r in RECIPES if not r.startswith(("xpro", "expired", "aerochrome", "standard"))])
def test_mid_gray_neutral(name):
    """负片靠冲印店定光 (相纸印放, 灰卡全校正), 反转片靠材料校准: 标准条件下中灰应为中性 18% 灰。
    (推冲反转片的偏色是真实现象, 允许稍大)"""
    out = Engine(Params(**{**recipe(name), "grain": 0, "output": "print", "roll": None})).render(RAMP)[0, 4]
    tol = 6.0 if name.startswith("push1_fujifilm") else 3.0
    assert chroma(out) < tol
    if not name.startswith("push1_fujifilm"):
        assert abs(out.mean() - 0.18) < 0.03


@pytest.mark.parametrize("name", RECIPES)
def test_tone_monotonic(name):
    out = Engine(Params(**{**recipe(name), "grain": 0})).render(RAMP)[0]
    Y = out @ np.array([0.2126, 0.7152, 0.0722])
    assert np.all(np.diff(Y) > -1e-3)


def test_exposure_compensation_changes_slides():
    """反转片没有冲印店补偿: 过曝一档必须更亮 (回归测试)。"""
    Y = [Engine(Params(film="fujifilm_velvia_100", grain=0, exposure_ev=ev)).render(np.full((1, 1, 3), 0.184))[0, 0, 1]
         for ev in (0, 1)]
    assert Y[1] > Y[0] * 1.5


def test_negative_latitude():
    """负片过曝 2 档, 冲印店定光后中灰仍接近标准 (负片宽容度)。"""
    a = Engine(Params(film="kodak_portra_400", grain=0)).render(RAMP)[0, 4]
    b = Engine(Params(film="kodak_portra_400", grain=0, exposure_ev=2)).render(RAMP)[0, 4]
    assert abs(a.mean() - b.mean()) < 0.03


def test_halation_red_without_remjet():
    img = np.full((200, 200, 3), 0.01)
    img[95:105, 95:105] = 200.0
    out = Engine(Params(**{**recipe("cinestill_800t"), "grain": 0, "frame_width_mm": 6.0})).render(img)
    ring = out[100, 115]
    assert ring[0] > ring[1] * 1.3 and ring[0] > ring[2] * 1.3


def test_status_m_sees_unwanted_absorption():
    """密度计读数应高于分析密度 (真实染料的多余吸收)。"""
    _, an, integ, _ = step_wedge(Params(film="kodak_portra_400", grain=0))
    assert np.all(integ[-5:, 1] > an[-5:, 1])


def test_tungsten_film_daylight_vs_tungsten_light():
    """灯光片 (500T): 日光下拍、冲印店不逐张校正 → 偏蓝; 钨丝灯下拍 → 接近中性。
    输入为日光白平衡的 RGB, 钨丝灯照明的灰卡在其中本身就是橙色的 (3200K 黑体相对 D55)。
    用胶片的设计光源 3200K (Kodak 5219 datasheet); 2856K 的 A 光源比设计更暖, 本身就留有偏色。"""
    import colour
    def gray_out(rgb):
        eng = Engine(Params(film="kodak_vision3_500t", grain=0, timing="standard"))
        return eng.render(np.asarray(rgb, float).reshape(1, 1, 3))[0, 0]
    xyz_a = colour.sd_to_XYZ(colour.sd_blackbody(3200)); xyz_d = colour.sd_to_XYZ(colour.SDS_ILLUMINANTS["D55"])
    M = colour.models.RGB_COLOURSPACE_sRGB.matrix_XYZ_to_RGB
    tung = (M @ xyz_a) / (M @ xyz_d)
    tung = 0.184 * tung / (tung @ np.array([0.2126, 0.7152, 0.0722]))
    day, tun = gray_out([0.184] * 3), gray_out(tung)
    assert day[2] > day[0] * 1.15          # 日光: 偏蓝
    assert chroma(tun) < chroma(day)       # 钨丝灯: 更接近中性


# ------------------------------------------------------ 光晕 / 随机性 --
def test_halation_ring_radius_from_base_optics():
    """光晕核的峰值半径应等于片基全反射的几何位置 2·t·tan(θc)。"""
    from filmlab.film import halation_kernel
    px = 10.0
    for t in (120.0, 140.0):
        k = halation_kernel(px, base_um=t)
        n = k.shape[0] // 2
        r_peak = np.argmax(k[n, n:]) * px
        expect = 2 * t * np.tan(np.arcsin(1 / 1.5))
        assert abs(r_peak - expect) < 2.5 * px


def test_halation_light_is_red_from_emulsion_stack():
    """到达片基并返回的光主要被红感层吸收 —— 来自乳剂叠层的光谱计算, 不是设定值。"""
    from filmlab import film, variation
    st = emulsion("kodak_vision3_500t").stock
    f = film.halation_response(st, "D55", None, variation.NOMINAL, 1.0)
    r, g, b = f(np.array([[1.0, 1.0, 1.0]]))[0]
    assert r > 3 * g and g > b


def test_remjet_blocks_halation():
    img = np.full((120, 120, 3), 0.01)
    img[55:65, 55:65] = 200.0
    a = Engine(Params(film="kodak_vision3_500t", process="ECN-2", grain=0, frame_width_mm=4, timing="standard",
                      output="print")).render(img)
    b = Engine(Params(**{**recipe("cinestill_800t"), "grain": 0, "frame_width_mm": 4, "timing": "standard",
                         "output": "print"})).render(img)
    ring = (slice(60, 61), slice(68, 72))
    assert b[ring][..., 0].mean() > 3 * a[ring][..., 0].mean()


def test_roll_variation_within_process_tolerance():
    """不同胶卷 / 冲洗的差异应在冲洗厂控制条的允许范围内 (中灰 ±0.1 档量级), 且可复现。"""
    from filmlab import variation
    ys = []
    for roll in range(8):
        out = Engine(Params(film="fujifilm_provia_100f", grain=0, roll=roll)).render(np.full((1, 1, 3), 0.184))
        ys.append(out[0, 0, 1])
    stops = np.log2(np.array(ys) / np.mean(ys))
    assert 0.005 < stops.std() < 0.15
    assert variation.sample(3) == variation.sample(3)


# ------------------------------------------------------------- 扫描 --
@pytest.mark.parametrize("name", ["kodak_portra_400", "kodak_gold_200", "fujifilm_pro_400h", "bw_pan"])
def test_scan_neutral_with_matching_profile(name):
    """扫描仪里有该胶片的档案时, 中灰扫描后应为中性 (输出目标严格中性的扫描仪; 冲印店的偏暖目标另计)。"""
    out = Engine(Params(**{**recipe(name), "grain": 0, "roll": None, "scanner": "frontier_neutral"})).render(RAMP)[0, 4]
    assert chroma(out) < 3.0 and abs(out.mean() - 0.18) < 0.03


def test_scan_without_profile_leaves_bounded_cast():
    """电影负片没有扫描档案, 用通用 Kodak 档案扫描 + 部分自动校正 → 留下有限的偏色。"""
    out = Engine(Params(film="kodak_vision3_250d", grain=0, roll=None)).render(RAMP)[0, 4]
    assert 1.0 < chroma(out) < 15.0


def test_dye_spectra_change_only_when_cross_processed():
    """染料由成色剂和显影剂共同决定: 只有遇到非设计的显影剂 (CD-3 胶片进 CD-4) 时光谱才改变,
    且展宽时积分吸收不变、峰值后移。"""
    from filmlab.stocks import WAVELENGTHS
    def spectra(**kw):
        eng = Engine(Params(grain=0, **kw))
        st = eng._develop(np.full((1, 3), 0.0))
        return eng._dye_spectra(st.dye_agent), eng.e.stock.dye_density
    s, d = spectra(film="kodak_vision3_500t", process="ECN-2")
    assert np.allclose(s, d)
    s, d = spectra(film="kodak_portra_400")
    assert np.allclose(s, d)
    s, d = spectra(**recipe("cinestill_800t"))
    assert not np.allclose(s, d)
    assert np.allclose(s.sum(0), d.sum(0))
    assert np.all(WAVELENGTHS[np.argmax(s, 0)] >= WAVELENGTHS[np.argmax(d, 0)])


def test_noise_estimate_survives_quantized_dark_frames():
    """极暗、量化严重的画面 (大部分 2×2 块完全相同) 不能让噪声拟合崩溃。"""
    from filmlab.rawio import estimate_noise
    rng = np.random.default_rng(0)
    img = np.round(rng.poisson(0.3, (256, 256, 3)) * 0.5) / 4000.0
    img[:128] = 0.002
    p = estimate_noise(img)
    assert np.all(np.isfinite(p)) and np.all(p > 0)


# ------------------------------------------------- 阶段契约 (docs/ARCHITECTURE.md) --
def test_negative_is_physical_endpoint():
    """Negative 携带三层曝光、染料量与光谱透射率; 扫描 Negative 与扫描 FilmState 结果相同。"""
    rgb = np.repeat(RAMP, 40, 0)
    eng = Engine(Params(**{**recipe("cinestill_800t"), "timing": "frame", "roll": 1}))
    neg = eng.develop_negative(rgb)
    assert neg.log_exposure.shape == rgb.shape and neg.dye.shape == rgb.shape
    T = neg.transmittance(np.arange(5))
    assert T.shape == (5, 81) and np.all((T > 0) & (T <= 1))
    sm = neg.status_m()
    assert np.all(np.diff(sm[0, :, 1]) > 0)                 # 曝光越多, 负片绿密度越高
    tr = {}
    a = eng.scan(neg, trace=tr)
    b = Engine(Params(**{**recipe("cinestill_800t"), "timing": "frame", "roll": 1})).render(rgb)
    assert np.array_equal(a, b) and tr["Dn"].shape == rgb.shape and tr["timing"].M is not None


def test_output_conditioning_is_separable():
    """输出调整 (经验风格) 关闭后, 定光与负片密度不变, 只有最终 RGB 改变。"""
    rgb = np.repeat(RAMP, 40, 0)
    out = {}
    for c in ("lab", None):
        eng = Engine(Params(**{**recipe("cinestill_800t"), "timing": "frame", "roll": 1, "output_conditioning": c}))
        tr = {}
        out[c] = (eng.scan(eng.develop_negative(rgb), trace=tr), tr)
    assert np.array_equal(out["lab"][1]["Dn"], out[None][1]["Dn"])
    assert np.array_equal(out["lab"][1]["timing"].K, out[None][1]["timing"].K)
    assert not np.allclose(out["lab"][0], out[None][0])
