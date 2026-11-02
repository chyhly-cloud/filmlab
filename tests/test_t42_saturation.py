"""T42: 数字相纸输出按通道独立 (paper_channel; 代际 t42_base = t31_base + paper_channel)。
现有代际不变; 中性灰阶与原色度换算逐点一致 (< 0.7%), 彩色只去掉跨通道项。"""
import numpy as np
import pytest

from filmlab import film, generation, rawio
from filmlab.engine import Engine, Params
from filmlab.recipes import recipe
from filmlab.scanning.paper import digital_paper

PAPER = "fujifilm_crystal_archive_typeii"


@pytest.fixture(autouse=True)
def _restore_generation():
    prev = generation.active()
    yield
    generation.use(prev)


def test_generation_is_t31_base_plus_paper_channel():
    assert generation.GENERATIONS["t42_base"] == ("D65", generation.GENERATIONS["t31_base"][1] + ("paper_channel",), {})
    generation.use("t42_base")
    assert (rawio.METERING, rawio.HIGHLIGHT_RECON, rawio.RECON_COLOUR, rawio.APERTURE_RANGE, film.MASK_PLACEMENT,
            rawio.STOPS_PRIOR) == ("session", "stops", "ring", None, "layers", (6.0, 8.0))
    for g, (_, corr, _) in generation.GENERATIONS.items():
        if g != "t42_base":
            assert "paper_channel" not in corr, g


def test_neutral_unchanged():
    a, b = digital_paper(PAPER), digital_paper(PAPER, per_channel=True)
    D = np.linspace(0.0, float(a.curve.max()), 40)
    Dp = np.repeat(D[:, None], 3, 1)
    oa, ob = a.reference_print(Dp), b.reference_print(Dp)
    assert np.allclose(ob, ob[:, :1])                              # 严格中性
    assert np.max(np.abs(ob - oa) / np.maximum(oa, 1e-3)) < 0.007   # 与色度换算的中性灰阶相差 < 0.7%


def _equiv(paper, out, Ds):
    """输出 → 中性灰阶上的等效相纸密度 (逐通道)。"""
    N = paper.reference_print(np.repeat(Ds[:, None], 3, 1))
    return np.array([np.interp(-np.log10(out[c]), -np.log10(N[:, c]), Ds) for c in range(3)])


def test_per_channel_has_no_cross_channel_term():
    a, b = digital_paper(PAPER), digital_paper(PAPER, per_channel=True)
    Ds = np.linspace(0.05, 2.0, 400)
    for paper, diag_only in ((a, False), (b, True)):
        J = np.zeros((3, 3))
        for j in range(3):
            d = np.full(3, 0.7); d[j] += 0.02
            J[j] = (_equiv(paper, paper.reference_print(d[None])[0], Ds)
                    - _equiv(paper, paper.reference_print(np.full((1, 3), 0.7))[0], Ds)) / 0.02
        off = J - np.diag(np.diag(J))
        if diag_only:
            np.testing.assert_allclose(J, np.eye(3), atol=0.01)
        else:   # 色度换算: R 通道放大约 10%, 有跨通道项 (T42 第 3 节)
            assert J[0, 0] > 1.05 and np.abs(off).max() > 0.05


def test_t42_only_changes_colour_not_grey():
    """同一负片密度: 设计光源中灰 (固定光号下印成灰) 两代际输出相同 (< 1%); 彩色负片密度的输出不同。"""
    rgb = np.array([[[0.40, 0.05, 0.03], [0.03, 0.10, 0.35]]]) * np.ones((8, 1, 1))
    out, grey = {}, {}
    for g in ("t31_base", "t42_base"):
        generation.use(g)
        eng = Engine(Params(**{**recipe("cinestill_800t"), "timing": "frame", "roll": None, "frame": 0, "grain": 0}))
        assert ("paper_channel" in eng.p.corrections) == (g == "t42_base")
        assert eng.scanner.digital_paper.per_channel == (g == "t42_base")
        grey[g] = eng.scanner.to_rgb(eng._lab_gray[None])[0]
        out[g] = eng.render(rgb)
    assert np.max(np.abs(grey["t42_base"] / grey["t31_base"] - 1)) < 0.01
    assert np.max(np.abs(np.log10(np.clip(out["t42_base"], 1e-3, None)) - np.log10(np.clip(out["t31_base"], 1e-3, None)))) > 0.02
