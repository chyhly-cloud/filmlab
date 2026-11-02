"""T32: 候选代际 t32_k5219 = t31_base + Vision3 5219 的 Kodak 原始数据修订 (染料光谱 + 按 Kodak Status M 重新解混的曲线)。
只有该代际启用修订; t31_base 与其他代际不变。"""
import numpy as np

from filmlab import film, generation, rawio, stocks


def test_t32_is_t31_base_plus_kodak_revision():
    wb, corr, rev = generation.GENERATIONS["t32_k5219"]
    wb0, corr0, rev0 = generation.GENERATIONS["t31_base"]
    assert (wb, corr) == (wb0, corr0) and rev0 == {} and rev == {"kodak_vision3_500t": "kodak_raw"}
    try:
        generation.use("t32_k5219")
        assert rawio.METERING == "session" and rawio.HIGHLIGHT_RECON == "stops" and film.MASK_PLACEMENT == "layers"
        k = np.load(stocks.PROFILE_DIR / "kodak_vision3_500t_kodak_raw.npz")
        np.testing.assert_array_equal(stocks.load_stock("kodak_vision3_500t").dye_density, k["E"])
        generation.use("t31_base")
        assert stocks.REVISIONS == {}
        assert not np.array_equal(stocks.load_stock("kodak_vision3_500t").dye_density, k["E"])
    finally:
        generation.use("current")


def test_revision_only_in_research_generations():
    for g, (_, _, rev) in generation.GENERATIONS.items():
        if g not in ("next_k5219", "t32_k5219"):
            assert rev == {}, g
