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
    """显影动力学 vs Kodak Portra 800 推一档 / 推两档实测 (C-41 3:45 / 4:15)。"""
    e = emulsion("kodak_portra_800")
    x = e.log_exposure
    m = (x > -2.5) & (x < 2.5)
    for stops, key in [(1, "kodak_portra_800_push1"), (2, "kodak_portra_800_push2")]:
        proc = chemistry.PROCESSES["C-41"].push(stops)
        st = chemistry.run(e, proc, np.repeat(x[:, None], 3, 1))
        rms_model = np.sqrt(np.mean((st.dye - load_stock(key).density_curves)[m] ** 2))
        rms_none = np.sqrt(np.mean((e.S_nom - load_stock(key).density_curves)[m] ** 2))
        assert rms_model < 0.05 and rms_model < 0.45 * rms_none


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
@pytest.mark.parametrize("name", [r for r in RECIPES if not r.startswith(("xpro", "expired", "aerochrome", "tungsten"))])
def test_mid_gray_neutral(name):
    """负片靠冲印店定光号, 反转片靠材料校准: 标准条件下中灰应为中性 18% 灰。
    (推冲反转片的偏色是真实现象, 允许稍大)"""
    out = Engine(Params(**{**recipe(name), "grain": 0})).render(RAMP)[0, 4]
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
