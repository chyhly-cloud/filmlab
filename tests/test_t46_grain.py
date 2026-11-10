"""T46: 跨片种通用的银盐颗粒 (filmlab.grain.derive_spec / materials.grain_spec_t46; 代际 t46_base = t45_base + grain_t46;
docs/tasks/T46_GRAIN_GENERAL.md)。T45 的颗粒路径逐位不变; 新档案只由胶片的公开物理量 + 通用常数 + 冲洗工艺生成。"""
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from filmlab import generation, grain, materials
from filmlab.chemistry import FilmState
from filmlab.engine import Engine, Params
from filmlab.materials import bw_film, emulsion, without_remjet
from filmlab.recipes import recipe

EVAL = Path(__file__).resolve().parents[1] / "tools" / "eval"


@pytest.fixture(autouse=True)
def _restore_generation():
    prev = generation.active()
    yield
    generation.use(prev)


def _box_sigma48(a, um):
    n = 42.5 / um
    vs = []
    for b in {max(1, int(np.floor(n))), int(np.ceil(n))}:
        h, w = (a.shape[0] // b) * b, (a.shape[1] // b) * b
        m = a[:h, :w].reshape(h // b, b, w // b, b, -1).mean((1, 3))
        vs.append(m.var((0, 1)) * (b * um) ** 2)
    return np.sqrt(np.mean(vs, 0)) / 42.5


def _analytic(sp, c, d0):
    lh, Dt = grain._state_table(sp, c)
    i = np.argsort(Dt)
    return float(grain.layer_curves(sp, c, 10 ** np.interp(d0, Dt[i], lh[i]))[1])


def test_generation_is_t45_base_plus_grain_t46():
    assert generation.GENERATIONS["t46_base"] == ("D65", generation.GENERATIONS["t45_base"][1] + ("grain_t46",), {})
    for g, (_, corr, _) in generation.GENERATIONS.items():
        if g not in ("t46_base", "t47_bw", "t48_bw", "t49_bw", "t50_bw", "t51_base", "t51_bw", "t56_base", "t56_bw"):          # t47_bw = t46_base + grain_t47 (T47)
            assert "grain_t46" not in corr, g


def test_t45_specs_keep_t45_defaults():
    """T45 的档案不带 T46 的随机项 (走原来的代码路径, 逐位不变)。"""
    sp = materials.grain_spec(emulsion("kodak_vision3_500t"))
    assert sp.eta_beta == 0.0 and sp.yield_cv == 0.0 and sp.cloud_L_um is None


def test_derive_reproduces_t45_scaling_at_t45_constants():
    """β = 0、η = 0.05、无随机产量时, 通用推导给出 5219 的 T45 晶体尺寸与涂布量 (推导与 T45 的换算是同一关系)。"""
    e = emulsion("kodak_vision3_500t")
    base = materials.grain_spec(e)
    U = materials.grain_universal(eta=grain.ETA_REF, beta=0.0)
    sp = grain.derive_spec(e, 500, "BB3200", U)
    assert np.allclose(sp.ecd_um, base.ecd_um, rtol=1e-6)
    assert np.allclose(sp.amount, base.amount, rtol=1e-4, atol=1e-6)


def test_yield_randomness_is_compound_poisson():
    """每晶体产量随机 (伽马, CV): 均值不变, 方差中每晶体项乘以 1 + CV²。"""
    e = emulsion("kodak_vision3_500t")
    sp = materials.grain_spec_t46(e, "ECN-2")
    sp0 = sp.with_(yield_cv=0.0)
    H = 10 ** np.linspace(-4, 0, 40)
    D1, s1 = grain.layer_curves(sp, 1, H)
    D0, s0 = grain.layer_curves(sp0, 1, H)
    D2, s2 = grain.layer_curves(sp.with_(yield_cv=2 * sp.yield_cv), 1, H)
    assert np.allclose(D1, D0) and np.allclose(D2, D0)
    assert np.all(s1 > s0)
    assert np.allclose(s2 ** 2 - s0 ** 2, 4 * (s1 ** 2 - s0 ** 2))          # 增量 ∝ CV²


@pytest.mark.parametrize("um", [22.5, 6.6])
def test_t46_simulation_matches_analytic(um):
    """模拟 (晶体计数 → 尺寸档 → 二项显影 → 伽马产量 → 每亚群染料云) 的 48 µm RMS = 解析值; 均值不变。"""
    e = without_remjet(emulsion("kodak_vision3_500t"))
    sp = materials.grain_spec_t46(e, "C-41")
    d0 = 0.65
    N = int(3000.0 / um)
    st = FilmState(latent=None, halide=None, silver=np.zeros((N, N, 3)), dye=np.full((N, N, 3), d0), coupler=None)
    grain.apply(e, sp, st, um, np.random.default_rng(3))
    s = _box_sigma48(st.dye, um)
    for c in range(3):
        assert abs(st.dye[..., c].mean() - d0) < 0.005
        assert abs(s[c] / _analytic(sp, c, d0) - 1) < 0.15, (c, s[c], _analytic(sp, c, d0))


def test_cloud_size_grows_with_crystal_and_process_length():
    sp = materials.grain_spec_t46(emulsion("kodak_vision3_500t"), "C-41")
    for c in range(3):
        sig = grain.cloud_sigma_um(sp, c)
        assert sig[0] > sig[1] > sig[2] >= sp.cloud_L_um


def test_universal_physics_trends():
    """同一组通用常数: 感光度越高晶体越大、颗粒越粗; 三维晶体比同感光度的 T 颗粒粗; 黑白为单层中性银。"""
    U = materials.grain_universal()
    e = emulsion("kodak_portra_400")
    sig = []
    for ei in (100, 400, 1600):
        sp = grain.derive_spec(e, ei, "D55", U)
        sig.append(_analytic(sp, 1, 1.0))
        assert sp.eta_beta == materials.GRAIN_T46_BETA
    assert sig[0] < sig[1] < sig[2]
    eb = bw_film()
    sT = grain.derive_spec(eb, 400, "D55", U, kind="silver", thickness_um=materials.GRAIN_T_UM)
    s3 = grain.derive_spec(eb, 400, "D55", U, kind="silver", thickness_um=None)
    assert s3.single_layer and _analytic(s3, 1, 1.0) > _analytic(sT, 1, 1.0)


def test_process_parameters():
    assert materials.process_grain("C-41")["name"] == "C-41"
    assert materials.process_grain("ECN-2")["name"] == "ECN-2"
    assert materials.process_grain("E-6 push 1")["name"] == "E-6"
    assert materials.process_grain("D-76")["kappa_scale"] == materials.GRAIN_PROCESS["BW"]["kappa_scale"]
    assert materials.GRAIN_PROCESS["E-6"]["kappa_scale"] > 1.0       # E-6 彩色显影成色剂过量


def test_t46_specs_for_all_recipe_films():
    for rec in ("kodak_portra_400", "kodak_ektar_100", "fujifilm_velvia_100", "fujifilm_provia_100f", "kodak_vision3_50d",
                "bw_pan", "cinestill_800t", "kodak_gold_200"):
        eng = Engine(Params(**{**recipe(rec), "grain": 1, "roll": None}))
        sp = materials.grain_spec_t46(eng.e, eng.proc.name)
        a = np.asarray(sp.amount)
        assert np.all(a >= 0) and a.sum() > 0 and np.all(np.asarray(sp.ecd_um) > 0), rec


def test_t46_grain_on_engine_is_mean_preserving():
    generation.use("t46_base")
    img = np.full((96, 96, 3), 0.05)
    kw = {**recipe("cinestill_800t"), "timing": "frame", "roll": None, "frame": 0, "frame_width_mm": 2.0}
    a = Engine(Params(**{**kw, "grain": 0})).develop_negative(img).state.dye
    b = Engine(Params(**{**kw, "grain": 1.0})).develop_negative(img).state.dye
    assert np.allclose(a.mean((0, 1)), b.mean((0, 1)), atol=0.01)
    assert np.all(b.std((0, 1)) > 0)


def test_t46_without_grain_equals_t45():
    rng = np.random.default_rng(1)
    img = np.full((60, 90, 3), 0.02) + rng.uniform(0, 0.05, (60, 90, 3))
    outs = []
    for g in ("t45_base", "t46_base"):
        generation.use(g)
        eng = Engine(Params(**{**recipe("cinestill_800t"), "timing": "frame", "roll": None, "frame": 0, "grain": 0}))
        outs.append(eng.render(img))
    assert np.array_equal(outs[0], outs[1])


def test_film_table_and_leave_one_out():
    rows = list(csv.DictReader(open(EVAL / "t46" / "films.tsv"), delimiter="\t"))
    assert len(rows) >= 40 and all(r["source"].startswith("http") for r in rows)
    classes = {r["class"] for r in rows if r["metric"] != "none"}
    assert {"motion_neg", "neg", "reversal", "bw"} <= classes
    d = json.load(open(EVAL / "t46" / "loo.json"))
    assert d["beta"] == materials.GRAIN_T46_BETA and abs(d["eta"] - materials.GRAIN_T46_ETA) < 1e-4
    lo = np.array([v["rms_log"] for v in d["loo"].values()])
    t45 = np.array([v["rms_log"] for v in d["t45_pure"].values()])
    assert np.median(lo) < 0.25 and np.sqrt(np.mean(lo ** 2)) < np.sqrt(np.mean(t45 ** 2))
