"""T33: (A) 蓝层 XYZ 映射已是只用 Z 的短波积分 (各灯光成员 X、Y 系数为 0); (B) DIR 层间抑制量按代际覆盖, 默认不变。"""
import numpy as np
import pytest

from filmlab import chemistry, generation, lights, materials
from filmlab.engine import Engine, Params
from filmlab.recipes import recipe
from filmlab.stocks import load_stock


@pytest.fixture(autouse=True)
def _restore():
    prev = generation.active()
    yield
    generation.use(prev)


def test_blue_xyz_mapping_is_pure_z_for_every_lamp_member():
    """blue_xyz (T16) 的非负拟合对所有人造光源成员都只给 Z 正系数: 蓝层曝光 = k·Z, 与 X、Y 无关。
    所以 T33 的候选 "只用 Z (保白点)" 与 "X 系数为 0" 与现状逐位相同, 没有另立代际。"""
    generation.use("t31_base")
    S = 10.0 ** load_stock("kodak_vision3_500t").log_sensitivity
    unit = np.eye(3) @ np.linalg.inv(np.linalg.inv(lights._daylight_wb()[0]).T)   # 单位 X、Y、Z 对应的线性 sRGB
    for c in lights.classes():
        if c.key == "daylight":
            continue
        for name, L in c.members:
            b = lights.linear_response(L, S, True, "xyz")(unit)[:, 2]
            assert b[2] > 0, name
            assert abs(b[0]) <= 1e-9 * b[2] and abs(b[1]) <= 1e-9 * b[2], (name, b)


def test_dir_amount_override_defaults_unchanged():
    for g in generation.GENERATIONS:
        generation.use(g)
        expect = generation.DIR_AMOUNT.get(g, {})
        assert materials.DIR_AMOUNT == expect, g
        if not expect:
            assert materials.emulsion("kodak_vision3_500t").dir_amount == 1.0
    generation.use("t31_base")
    assert materials.DIR_AMOUNT == {}
    e = Engine(Params(**{**recipe("cinestill_800t"), "roll": None})).e
    assert e.dir_amount == 1.0


def test_dir_amount_override_scales_interlayer_but_keeps_neutral():
    """覆盖只改该档案的 dir_amount; 中性灰阶 (三层同 logH) 的显影结果按构造不随 dir_amount 变 (_intrinsic_curve),
    彩色曝光的相邻层密度随 dir_amount 减小而升高。"""
    generation.use("t31_base")
    try:
        e1 = materials.emulsion("kodak_vision3_500t")
        materials.DIR_AMOUNT["kodak_vision3_500t"] = 0.5
        materials.emulsion.cache_clear()
        e5 = materials.emulsion("kodak_vision3_500t")
        assert e5.dir_amount == 0.5 and materials.emulsion("kodak_portra_400").dir_amount == 1.0
    finally:
        generation.use("t31_base")
    proc = chemistry.PROCESSES["C-41"]
    x = np.linspace(-2.0, 1.0, 7)
    neutral = np.repeat(x[:, None], 3, 1)
    red = np.stack([x + 0.6, x - 0.4, x - 0.5], 1)
    d = {a: (chemistry.run(e, proc, neutral).dye, chemistry.run(e, proc, red).dye) for a, e in ((1.0, e1), (0.5, e5))}
    np.testing.assert_allclose(d[0.5][0], d[1.0][0], atol=2e-3)
    assert np.all(d[0.5][1][:, 1] >= d[1.0][1][:, 1] - 1e-9) and np.any(d[0.5][1][:, 1] > d[1.0][1][:, 1] + 0.02)
