"""apply_to_image: 任意照片 (8 / 16 位 sRGB、线性浮点、单色), 两种模式, 物理定义的各步。"""
import numpy as np
import pytest

import silvergrain as sg
from silvergrain.image import srgb_encode


def _photo(h=96, w=144, mono=False):
    rng = np.random.default_rng(0)
    y = np.linspace(0.02, 0.9, w)[None, :].repeat(h, 0)
    img = np.stack([y, y * 0.8, y * 0.6], -1) * (1 + 0.0 * rng.random((h, w, 1)))
    return img[..., 1] if mono else img


def test_grain_mode_keeps_mean_and_hue():
    img = _photo()
    r = sg.apply_to_image(img, "kodak_trix_400", format="35mm", seed=1, method="explicit")
    o = r.linear
    assert o.shape == img.shape
    blocks = lambda a: a[:, :, 1].reshape(96, 12, 12).mean((0, 2))
    assert np.allclose(blocks(o) / blocks(img), 1.0, atol=0.06)
    # 黑白颗粒只在亮度上: 三通道同一相对起伏, 色比不变
    assert np.allclose(o[..., 0] / np.maximum(o[..., 1], 1e-9), img[..., 0] / img[..., 1], rtol=1e-6, atol=1e-6)
    assert r.info["px_um"] == pytest.approx(36000 / 144) and r.info["mode"] == "grain"


def test_colour_film_grain_per_layer():
    img = _photo()
    o = sg.apply_to_image(img, "kodak_portra_400", seed=2).linear
    rel = o / img - 1
    c = np.corrcoef(rel[..., 0].ravel(), rel[..., 2].ravel())[0, 1]
    assert abs(c) < 0.3                                          # 各层染料颗粒独立
    assert np.all(np.abs(rel.mean((0, 1))) < 0.03)


def test_inputs_8bit_16bit_mono():
    img = _photo()
    u8 = np.round(srgb_encode(img) * 255).astype(np.uint8)
    u16 = np.round(srgb_encode(img) * 65535).astype(np.uint16)
    a = sg.apply_to_image(u8, "kodak_trix_400", seed=3)
    b = sg.apply_to_image(u16, "kodak_trix_400", seed=3)
    assert a.info["input"]["bits"] == 8 and b.info["input"]["bits"] == 16
    assert np.abs(a.linear - b.linear).mean() < 0.01
    m = sg.apply_to_image(_photo(mono=True), "kodak_trix_400", seed=3)
    assert m.linear.ndim == 2


def test_full_mode_lab_tone_and_timing():
    grey = np.full((64, 64, 3), 0.18)
    r = sg.apply_to_image(grey, "kodak_trix_400", mode="full", seed=1)
    assert r.linear.mean() == pytest.approx(0.18, rel=0.06)        # 自动定光: 整帧平均密度 → 码值 118
    lo = sg.apply_to_image(grey, "kodak_trix_400", mode="full", timing="fixed", exposure_ev=-2, seed=1).linear.mean()
    hi = sg.apply_to_image(grey, "kodak_trix_400", mode="full", timing="fixed", exposure_ev=2, seed=1).linear.mean()
    assert lo < 0.18 / 3 and hi > 0.18 * 2                       # 固定定光: 影调跟着该片的曲线与冲印店影调
    au = sg.apply_to_image(grey, "kodak_trix_400", mode="full", exposure_ev=2, seed=1).linear.mean()
    assert au == pytest.approx(0.18, rel=0.06)                   # 自动定光补偿曝光


def test_legacy_lab_none_still_available():
    grey = np.full((64, 64, 3), 0.18)
    r = sg.apply_to_image(grey, "kodak_trix_400", mode="full", lab=None, seed=1)
    assert r.linear.mean() == pytest.approx(0.18, rel=0.06) and "scan" in r.info


def test_exposure_metering_and_filter():
    grey = np.full((32, 32, 3), 0.18)
    r = sg.expose(grey, "kodak_trix_400")
    assert np.allclose(r.H, 10 / 400)                            # 18% 灰 → H = 10/EI
    assert np.allclose(sg.expose(grey, "kodak_trix_400", exposure_ev=1).H, 20 / 400)
    f = sg.expose(grey, "kodak_trix_400", filter="25")           # 补偿后中性灰不变
    assert np.allclose(f.H, 10 / 400, rtol=1e-6)
    assert f.info["filter"]["filter_stops"][0] > 2               # Wratten 25 的系数约 3 档 (Kodak 公布 8×)
    red = np.zeros((8, 8, 3)); red[..., 0] = 0.3
    assert sg.expose(red, "kodak_trix_400", filter="25").H.mean() > sg.expose(red, "kodak_trix_400").H.mean()


def test_scanner_and_output_px_and_seed():
    img = _photo()
    a = sg.apply_to_image(img, "kodak_trix_400", scanner="noritsu_rpl", seed=5)
    b = sg.apply_to_image(img, "kodak_trix_400", scanner="noritsu_rpl", seed=5)
    assert np.array_equal(a.linear, b.linear)
    c = sg.apply_to_image(img, "kodak_trix_400", output_px=72, seed=5)
    assert c.linear.shape == (48, 72, 3) and c.info["px_um"] == pytest.approx(500.0)


def test_grain_mode_coarse_positive_and_mean():
    """k = 3: 输出恒为正, 不超过 1; 平均 = 照片经 −log⟨T⟩ 的物理偏移 g(u + E[δ]) / g(u) (冲印店影调传递), 别无其他偏移。"""
    from silvergrain.image import srgb_decode, srgb_encode
    from silvergrain.labtone import tone
    g = np.full((256, 384, 3), 0.18)
    r = sg.apply_to_image(g, "kodak_trix_400", grain_scale=3.0, format=384 * 12 / 1000, seed=2, method="closed_form",
                          keep_render=True, scanner=None)              # 只看颗粒与影调 (冲印店 USM 另会把少数像素推到 0)
    assert r.linear.min() > 0.0 and r.linear.max() <= 1.0
    (T,) = tone("rpl_noritsu", "bw")
    u = T.u_of(255 * srgb_encode(0.18))
    Ed = float((r.render.density - r.render.mean).mean())
    want = srgb_decode(T.code_of(u + Ed) / 255) / 0.18
    assert r.linear.mean() / 0.18 == pytest.approx(want, rel=0.01)


def test_bw_grain_on_colour_is_neutral_density():
    """黑白银粒在彩色照片上是中性密度: 色相 (色度方向) 严格不变, 不超过 1; 超过正像白的像素只把色度向白收 (亮度 = 颗粒的亮度),
    不截断亮的颗粒 (T55c)。"""
    rng = np.random.default_rng(0)
    img = np.stack([np.full((96, 128), 0.95), np.full((96, 128), 0.55), np.full((96, 128), 0.25)], -1) * rng.uniform(0.7, 1.0, (96, 128, 1))
    for sc in (None, "noritsu_rpl"):
        r = sg.apply_to_image(img, "kodak_trix_400", grain_scale=3.0, seed=4, scanner=sc)
        o = r.linear
        assert o.max() <= 1.0 + 1e-12
        L = np.array([0.2126, 0.7152, 0.0722])
        c, c0 = o - (o @ L)[..., None], img - (img @ L)[..., None]
        ok = np.linalg.norm(c, axis=-1) > 1e-3
        cos = (c * c0).sum(-1)[ok] / np.linalg.norm(c, axis=-1)[ok] / np.linalg.norm(c0, axis=-1)[ok]
        assert cos.min() > 1 - 1e-9
        assert o[..., 1].std() > 0.01 and 0.0 <= r.info["desaturated_fraction"] <= 1.0


def test_default_scanner_follows_lab():
    g = np.full((64, 64, 3), 0.18)
    assert sg.apply_to_image(g, "kodak_trix_400", seed=1).info["scanner"].startswith("Noritsu")
    assert sg.apply_to_image(g, "kodak_trix_400", seed=1, scanner=None).info["scanner"] is None
    assert sg.apply_to_image(g, "kodak_trix_400", seed=1, lab=None).info.get("scanner") is None


def test_lab_tone_data():
    from silvergrain.labtone import tone
    from silvergrain import database as db
    (T,) = tone("rpl_noritsu", "bw")
    assert np.all(np.diff(T.code) > 0) and T.code_of(0.0) == pytest.approx(118.0)
    assert 250 < float(T.slope(0.0)) < 360                       # 中灰处 dCode/dD (T51b RPL 实测 275–302 的量级)
    R, G, B = tone("rpl_noritsu", "colour")
    assert all(np.all(np.diff(c.code) > 0) for c in (R, G, B))
    d = db.load_json("lab_tone.json")["rpl_noritsu"]
    c = d["bw"]["curve"]
    assert d["bw"]["frames"] == 14 and d["zip"].startswith("https://") and len(c["slope_q25"]) == len(c["u"])
    i = int(np.argmin(np.abs(np.asarray(c["code"]) - 118)))
    assert c["slope_q25"][i] < c["slope_median"][i] < c["slope_q75"][i]
