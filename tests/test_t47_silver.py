"""T47: 显式银粒 (filmlab.grain.SilverSpec / render_silver; materials.silver_spec / trix_film; 代际 t47_bw = t46_base + grain_t47;
docs/tasks/T47_TRIX_SILVER_GRAIN.md)。闭式解、质量守恒、尺度不变、分块确定性、Tri-X 密度门槛、彩色片与现有代际不变。"""
import json
from pathlib import Path

import numpy as np
import pytest

from filmlab import generation, grain, materials
from filmlab.engine import Engine, Params
from filmlab.recipes import recipe

EVAL = Path(__file__).resolve().parents[1] / "tools" / "eval" / "t47"
SPEC = grain.SilverSpec(ecd_um=(0.6, 0.29, 0.133), sigma_ln=(0.186,) * 3, coat_g_m2=(1.5, 1.2, 1.2), ei=400.0,
                        eta=0.0248 * 3 * 8, beta=0.875, fog_area_um2=43.42, base_density=0.3, bins=5, shapes=3)


@pytest.fixture(autouse=True)
def _restore_generation():
    prev = generation.active()
    yield
    generation.use(prev)


def _csv(name):
    rows = [l for l in (EVAL / name).read_text().splitlines() if l and not l.startswith("#")]
    head = rows[0].split(",")
    return {h: np.array([float(r.split(",")[i]) for r in rows[1:]]) for i, h in enumerate(head)}


def test_generation_is_t46_base_plus_grain_t47():
    assert generation.GENERATIONS["t47_bw"] == ("D65", generation.GENERATIONS["t46_base"][1] + ("grain_t47",), {})
    for g, (_, corr, _) in generation.GENERATIONS.items():
        if g != "t47_bw":
            assert "grain_t47" not in corr, g


def test_silver_mass_and_footprint():
    """质量守恒: 每颗显影晶体的银 = 3.72e-12 g/µm³ × 晶体体积; 丝长 × 丝截面 = 银体积; 投影不透明面积 ≤ 丝长 × 丝径
    且 ≤ 丝团面积; 涂布银量 = Σ 晶体数 × 每晶体银。"""
    b = grain.silver_bins(SPEC)
    A = grain.silver_library(SPEC)["A_spec"]
    assert np.allclose((b["n"] * b["m"]).sum(1) * 1e12, SPEC.coat_g_m2)
    L = b["V"] / (np.pi * SPEC.filament_um ** 2 / 4)
    long = L / SPEC.filaments >= 10 * SPEC.filament_um              # 丝长 ≫ 丝径的档 (最小晶体的"丝"短到只剩端帽)
    assert long.sum() >= 8
    assert np.all(A[long] <= 1.1 * L[long] * SPEC.filament_um) and np.all(A <= SPEC.spread * b["a"] * 1.05)
    assert np.all(np.diff(A, axis=1) > 0)                         # 大晶体 → 大银粒
    bud = grain.silver_budget(SPEC)
    H = np.logspace(-4, 1, 50)
    assert np.all(grain.silver_density(SPEC, H) <= bud["D_max"] + 1e-9)
    assert np.allclose(grain.silver_developed_mass(SPEC, 1e6), bud["coat_g_m2"], rtol=1e-6)


def test_diffuse_is_specular_over_q():
    H = np.logspace(-4, 0, 20)
    assert np.allclose(grain.silver_density(SPEC, H, "specular"), SPEC.callier_q * grain.silver_density(SPEC, H, "diffuse"))


@pytest.mark.parametrize("delta", [0.125, 0.5431, 2.1725])
def test_stamps_conserve_single_grain_area(delta):
    """任意微元间距、任意亚格相位: 单颗银粒的 ∫(1 − T) 精确等于参考栅格上的不透明面积 (宏观密度与采样间距无关的前提)。"""
    A = grain.silver_library(SPEC)["A_spec"]
    st = grain.silver_stamps(SPEC, delta)
    for k in range(3):
        for q in range(SPEC.bins):
            cov = st[k][q][2]
            area = cov.sum((3, 4)) * delta * delta / 1.0
            assert np.allclose(area.mean(), A[k, q], rtol=1e-4)
            assert np.allclose(area.std(axis=(0, 1)), 0, atol=1e-6 * max(A[k, q], 1e-3) + 1e-9)


@pytest.mark.parametrize("px,sub,cs", [(8.69, 16, 4), (6.35, 8, 2), (2.0, 8, 8)])
def test_render_mean_density_equals_closed_form(px, sub, cs):
    """平均透射率 = 闭式解 exp(−Σ n p g A), 与像素尺寸、微元间距、粗格分级无关 (先在透射率域积分, 再取负对数)。"""
    for H0 in (0.004, 0.05):
        n = int(round(400 / px))
        T = grain.render_silver(SPEC, np.full((n, n), H0), px, 11, sub=sub, count_sub=cs)
        e = 2 * int(np.ceil(3 / px))                                  # 画幅边缘没有外侧晶体, 去掉
        D = -np.log10(T[e:-e, e:-e].mean())
        assert abs(D - grain.silver_density(SPEC, H0)) < 0.012, (px, H0, D)


def test_render_blocks_are_deterministic():
    """晶体按像素行用独立种子生成: 分块大小与进程数不改变结果 (跨块的银丝团两边一致)。"""
    H = np.full((40, 30), 0.02) * np.linspace(0.3, 3, 30)[None]
    a = grain.render_silver(SPEC, H, 8.69, 5, rows_per_task=7)
    b = grain.render_silver(SPEC, H, 8.69, 5, rows_per_task=16)
    c = grain.render_silver(SPEC, H, 8.69, 5, rows_per_task=16, workers=2)
    assert np.array_equal(a, b) and np.array_equal(b, c)


def test_closed_form_aperture_variance_matches_simulation():
    """显微视场 (0.25 µm 像素, 微元 0.125 µm, PSF 1 µm): 5 / 10 / 25 µm 孔径的透射率方差 = 闭式解 (泊松布尔模型协方差
    ⊛ 方孔 ⊛ PSF)。"""
    H0 = 0.02
    ET, C, h = grain.silver_covariance(SPEC, H0)
    sides = np.array([5.0, 10.0, 25.0])
    v = grain.window_variance(C, h, sides, psf_um=1.0, adjacent=False)
    sims = []
    for seed in range(3):
        T = grain.render_silver(SPEC, np.full((1600, 1600), H0), 0.25, 100 + seed, sub=2, count_sub=1, psf_um=1.0)[50:-50, 50:-50]
        sims.append([T[:1500 // c * c, :1500 // c * c].reshape(1500 // c, c, 1500 // c, c).mean((1, 3)).var() for c in (20, 40, 100)])
    sims = np.mean(sims, 0)
    assert np.all(np.abs(sims / v - 1) < np.array([0.08, 0.12, 0.25])), (sims, v)
    assert abs(-np.log10(ET) - grain.silver_density(SPEC, H0)) < 1e-12


def _disc_cov(tau, D, radius=1.0, h=0.02):
    """Codex 的单半径泊松圆盘闭式解: Cov_T(r) = E[T]² expm1(λ (1 − e^−τ)² A∩(r))。"""
    ET = 10 ** -D
    att = -np.expm1(-tau)
    lam = -np.log(ET) / (att * np.pi * radius ** 2)
    n = int(2 * 2 * radius / h) + 1
    x = (np.arange(n) - n // 2) * h
    r = np.hypot(x[:, None], x[None])
    q = np.clip(r / (2 * radius), 0, 1)
    ov = 2 * radius ** 2 * (np.arccos(q) - q * np.sqrt(1 - q * q))
    return ET * ET * np.expm1(lam * att ** 2 * ov), h


@pytest.mark.parametrize("tau,D,side,expected", [
    (0.1, 0.682, 5.0, 0.0006592085039966425), (0.1, 0.682, 10.0, 0.00018685858309939812),
    (1.0, 0.682, 5.0, 0.005485612398929873), (0.1, 0.682, 25.0, 3.2116186464694165e-05)])
def test_window_variance_reproduces_codex_disc_closed_form(tau, D, side, expected):
    """同一闭式解在 Codex 交接包的检查条件 (单半径 1 µm 圆盘, research/t47-trix/data/poisson-pointwise-single-60.csv) 上一致。"""
    C, h = _disc_cov(tau, D)
    v = grain.window_variance(C, h, [side], psf_um=0.0, adjacent=False)[0]
    assert abs(v / expected - 1) < 0.01


def test_trix_profile_density_gate():
    """Tri-X 400: 晶体模型 (漫射) 的 19 级净密度与 Schmidt D-76 曲线 |ΔD| ≤ 0.10, 银量上限高于最高一级 (1.576)。"""
    sd = _csv("schmidt_density_19.csv")
    e = materials.trix_film()
    sp = materials.silver_spec(e)
    H = 0.8 / 400 * 10 ** sd["aligned_log10_exposure"]
    net = grain.silver_density(sp, H) - grain.silver_density(sp, 0.0)
    assert np.max(np.abs(net - sd["net_D"])) <= 0.10
    bud = grain.silver_budget(sp)
    assert bud["D_max"] - grain.silver_density(sp, 0.0) > 1.576
    assert 3.0 <= bud["coat_g_m2"] <= 8.0 and 0.3 <= bud["covering_power"] <= 1.0
    # 胶片档案的特性曲线就是这条模型曲线 (加片基 = Schmidt 的片基 + 灰雾)
    x = e.log_exposure
    assert np.allclose(e.S_nom[:, 1], grain.silver_curve(sp, x))
    assert abs(sp.base_density + grain.silver_density(sp, 0.0) - materials.TRIX_GROSS_FOG) < 1e-9


def test_calibration_file_matches_materials():
    cal = json.loads((EVAL / "calib.json").read_text())
    assert cal["eta_scale"] == pytest.approx(materials.GRAIN_T47_ETA3D_SCALE, rel=1e-3)
    sp = materials.silver_spec(materials.trix_film())
    assert np.allclose(sp.coat_g_m2, cal["coat_g_m2"], rtol=1e-3)


def test_t47_engine_grain_is_mean_preserving_and_neutral():
    generation.use("t47_bw")
    img = np.full((64, 64, 3), 0.18)
    kw = {**recipe("kodak_trix_400"), "timing": "frame", "roll": None, "frame": 0, "frame_width_mm": 0.6}
    a = Engine(Params(**{**kw, "grain": 0})).develop_negative(img).state.silver
    b = Engine(Params(**{**kw, "grain": 1.0})).develop_negative(img).state.silver
    c = slice(8, -8)
    assert abs(a[c, c].mean() - b[c, c].mean()) < 0.01
    assert b[..., 1].std() > 0.01
    assert np.array_equal(b[..., 0], b[..., 1]) and np.array_equal(b[..., 1], b[..., 2])     # 单层中性银


def test_t47_leaves_colour_films_and_grain_off_unchanged():
    """彩色片在 t47_bw 与 t46_base 下逐位相同; 颗粒关闭时 Tri-X 在两代际也逐位相同。"""
    rng = np.random.default_rng(2)
    img = np.full((40, 60, 3), 0.05) + rng.uniform(0, 0.1, (40, 60, 3))
    for rec, gr in (("kodak_portra_400", 1.0), ("cinestill_800t", 1.0), ("kodak_trix_400", 0.0)):
        outs = []
        for g in ("t46_base", "t47_bw"):
            generation.use(g)
            e = Engine(Params(**{**recipe(rec), "timing": "frame", "roll": None, "frame": 0, "grain": gr, "frame_width_mm": 0.5}))
            outs.append(e.render(img))
        assert np.array_equal(outs[0], outs[1]), rec


def test_generic_bw_film_gets_silver_spec():
    """通用全色片 (bw_pan) 也走同一显式银粒路径, 档案只由它自己的特性曲线与通用常数生成。"""
    e = Engine(Params(**{**recipe("bw_pan"), "roll": None})).e
    sp = materials.silver_spec(e)
    assert np.all(np.asarray(sp.coat_g_m2) >= 0) and sum(sp.coat_g_m2) > 0
    generation.use("t47_bw")
    img = np.full((32, 32, 3), 0.18)
    s = Engine(Params(**{**recipe("bw_pan"), "timing": "frame", "roll": None, "frame": 0, "grain": 1.0,
                         "frame_width_mm": 0.3})).develop_negative(img).state.silver
    assert s[..., 1].std() > 0.005


def test_t47_frame_timing_ignores_grain():
    """冲印店按低分辨率预扫定光: 颗粒开 / 关时的自动定光相同 (影调相同, 颗粒只在局部)。"""
    generation.use("t47_bw")
    rng = np.random.default_rng(0)
    img = np.exp(rng.normal(np.log(0.1), 1.0, (48, 64, 3)))
    t = []
    for gr in (0.0, 1.0):
        e = Engine(Params(**{**recipe("kodak_trix_400"), "timing": "frame", "roll": None, "frame": 0, "grain": gr,
                             "frame_width_mm": 0.6}))
        e.render(img)
        t.append(e.scanner.timing)
    assert np.allclose(t[0].K, t[1].K) and np.allclose(t[0].s, t[1].s)


# ------------------------------------------------------------------ T47b --
def test_tgrain_volume_and_surface():
    """T 颗粒: 每晶体银 = 3.72e-12 × a·t; 有效量子效率的表面系数 1 (三维 3), T47 的 η 修正对两者相同。"""
    sp = SPEC.with_(thickness_um=0.12)
    b = grain.silver_bins(sp)
    assert np.allclose(b["m"], grain.AGBR_AG_G_UM3 * b["a"] * 0.12)
    U = materials.grain_universal()
    _, _, e3, _ = grain.silver_structure(400, U, 2.0)
    _, _, et, _ = grain.silver_structure(400, U, 2.0, thickness_um=0.12)
    _, _, et1, _ = grain.silver_structure(400, U, 2.0, thickness_um=0.12, scale_tgrain=False)
    assert np.isclose(e3, 3 * et) and np.isclose(et, 2 * et1)


@pytest.mark.parametrize("key", ["kodak_tmax_100", "kodak_tmax_400"])
def test_bw_silver_films_reproduce_schmidt_curves(key):
    """T-Max 100 / 400: 晶体模型 (漫射) 复现 Schmidt 的 D-76 曲线 |ΔD| ≤ 0.10, 银量上限高于最高一级。"""
    e = materials.bw_silver_film(key)
    sp = materials.silver_spec(e)
    f = materials.BW_SILVER_FILMS[key]
    x, net = map(np.asarray, getattr(materials, f["curve"]))
    H = 0.8 / f["ei"] * 10 ** x
    model = grain.silver_density(sp, H) - grain.silver_density(sp, 0.0)
    assert np.max(np.abs(model - net)) <= 0.10
    assert grain.silver_budget(sp)["D_max"] - grain.silver_density(sp, 0.0) > net.max()
    assert sp.thickness_um == materials.GRAIN_T_UM and np.allclose(e.S_nom[:, 1], grain.silver_curve(sp, e.log_exposure))


def test_t47b_loo_not_worse_than_t46():
    """T47b 留一 (tools/eval/t47b_bw_loo.py 的结果): 同口径 (D-76) 的官方值与 Schmidt 同仪器留出片, |log 误差| 中位不大于 T46。"""
    R = json.loads((EVAL / "t47b_loo.json").read_text())
    med = lambda v: float(np.median(np.abs(np.log(v))))
    J = [a for a in R["official"] if a["judged"]]
    assert {a["key"] for a in J} == {"tmax100", "tmax400"}
    assert med([a["t47_ratio"] for a in J]) <= med([a["t46_loo_ratio"] for a in J])
    H = [b for b in R["schmidt"] if b["role"] != "标定"]
    assert med([b["t47_ratio"] for b in H]) <= med([b["t46_ratio"] for b in H])


def test_tmax_engine_grain_is_mean_preserving():
    generation.use("t47_bw")
    img = np.full((48, 48, 3), 0.18)
    kw = {**recipe("kodak_tmax_400"), "timing": "frame", "roll": None, "frame": 0, "frame_width_mm": 0.45}
    a = Engine(Params(**{**kw, "grain": 0})).develop_negative(img).state.silver
    b = Engine(Params(**{**kw, "grain": 1.0})).develop_negative(img).state.silver
    c = slice(6, -6)
    assert abs(a[c, c].mean() - b[c, c].mean()) < 0.01 and b[..., 1].std() > 0.005
