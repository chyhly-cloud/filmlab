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
    o = sg.apply_to_image(img, "kodak_portra_400", seed=2, scanner=None).linear
    rel = o / img - 1
    c = np.corrcoef(rel[..., 0].ravel(), rel[..., 2].ravel())[0, 1]
    assert abs(c) < 0.3                                          # 各层染料颗粒独立 (冲印店的亮度锐化会让通道相关, 见 T55e)
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


def test_grain_mode_equals_full_mode_fluctuation():
    """T55d: grain 模式每个像素的相对亮度起伏 = full 模式同一像素的相对起伏 (同一底片、同一冲印店影调与定光; 不加冲印店扫描时严格)。"""
    from silvergrain.image import srgb_decode
    from silvergrain.labtone import tone
    img = np.zeros((96, 192, 3)); img[:, :64] = (0.45, 0.08, 0.06); img[:, 64:128] = (0.07, 0.10, 0.40); img[:, 128:] = 0.18
    g = sg.apply_to_image(img, "kodak_trix_400", seed=5, scanner=None)
    f = sg.apply_to_image(img, "kodak_trix_400", seed=5, scanner=None, mode="full", keep_render=True)
    (T,) = tone("rpl_noritsu", "bw")
    Dm = f.render.mean.reshape(96, 192)
    base = srgb_decode(T.code_of(Dm - Dm.mean()) / 255)
    L = np.array([0.2126, 0.7152, 0.0722])
    for sl in (np.s_[8:88, 8:56], np.s_[8:88, 72:120], np.s_[8:88, 136:184]):
        rg = np.log(np.clip(g.linear[sl] @ L, 1e-9, None) / (img[sl] @ L))
        rf = np.log(np.clip(f.linear[sl][..., 0], 1e-9, None) / base[sl])
        assert rg.std() == pytest.approx(rf.std(), rel=0.01)


def test_wratten_changes_bw_grain_density():
    """红滤镜 (Wratten 25): 蓝色区域的底片变薄 (颗粒随之), 红色区域变厚; 中性灰基本不变 (按中性灰补偿)。"""
    img = np.zeros((64, 192, 3)); img[:, :64] = (0.07, 0.10, 0.40); img[:, 64:128] = (0.45, 0.08, 0.06); img[:, 128:] = 0.18
    a = sg.apply_to_image(img, "kodak_trix_400", seed=1, keep_render=True).render.mean.reshape(64, 192)
    b = sg.apply_to_image(img, "kodak_trix_400", seed=1, filter="25", keep_render=True).render.mean.reshape(64, 192)
    assert b[:, :64].mean() < a[:, :64].mean() - 0.05
    assert b[:, 64:128].mean() > a[:, 64:128].mean() + 0.05
    assert abs(b[:, 128:].mean() - a[:, 128:].mean()) < 0.01


def test_neutral_grey_matches_t55c():
    """18% 中性灰: 该片的曝光密度 = 由照片码值反推的密度 (T55c), 颗粒强度相同 (不同的 H 给不同的随机实现, 比较统计量)。"""
    g = np.full((256, 256, 3), 0.18)
    a = sg.apply_to_image(g, "kodak_trix_400", format=256 * 12 / 1000, seed=3, grain_density="photo").linear[..., 1]
    b = sg.apply_to_image(g, "kodak_trix_400", format=256 * 12 / 1000, seed=3).linear[..., 1]
    assert np.log(b / 0.18).std() == pytest.approx(np.log(a / 0.18).std(), rel=0.03)
    assert b.mean() == pytest.approx(a.mean(), rel=0.01)


# ------------------------------------------------------------------ T55e: 线阵 CCD 扫描链 --
def test_scan_grid_geometry():
    from silvergrain.image import get_scanner, scan_grid
    sc = get_scanner("noritsu_rpl")
    g = scan_grid((2796, 2048), 20.0, "645", sc)                   # RPL 的 645 文件: 原生 4824 × 3533 缩小 1.7253 倍
    assert g["native_pitch_um"] == pytest.approx(56000 / 4824)
    assert g["s"] == pytest.approx(1.7253, rel=0.01) and g["ccd_axis"] == 0 and g["gauge"] == "120"
    g = scan_grid((1000, 1500), 8.0, "35mm", sc)                    # 135: 原生 24 mm / 4492; s = 输出间距 / 原生间距
    assert g["native_pitch_um"] == pytest.approx(24000 / 4492) and g["s"] == pytest.approx(8.0 / (24000 / 4492), rel=0.01)
    g = scan_grid((4000, 6000), 6.0, "35mm", sc)                    # 横幅整幅 35 mm: CCD 横跨 24 mm 的短边 (竖直);
    assert g["ccd_axis"] == 0                                       # 原生整幅 3000 万像素 > 上限 2600 万: 渲染间距略大
    assert g["shape_native"][0] * g["shape_native"][1] <= 26e6 * 1.001 and 1.0 < g["s"] < 6.0 / (24000 / 4492)
    g = scan_grid((64, 64), 562.5, "35mm", sc)                      # 很粗的输出: 原生网格最多细 4 倍
    assert g["shape_native"] == (256, 256)
    assert scan_grid((64, 64), 20.0, "35mm", get_scanner("frontier_epfl")) is None


def test_resample_triangle_fingerprint():
    """缩小 1.7253 倍的三角核: 每列方差以 1 − 0.7253 = 0.2747 周/像素调制 (真实 RPL 扫描的指纹, 深度约 0.04); 平均不变。"""
    from silvergrain.image import resample2
    rng = np.random.default_rng(0)
    a = rng.standard_normal((1400, 1400))
    o = resample2(a, (int(1400 / 1.7253), int(1400 / 1.7253)))
    v = (o ** 2).mean(0); v = v / v.mean() - 1
    x = np.arange(len(v))
    depth = max(2 * abs(np.sum(v * np.exp(-2j * np.pi * f * x))) / len(x) for f in np.linspace(0.27, 0.28, 41))
    assert 0.02 < depth < 0.07
    assert abs(resample2(np.full((100, 100), 3.0), (58, 58)).mean() - 3.0) < 1e-12


def test_noritsu_chain_crisp_and_anisotropic():
    """均匀灰经 Noritsu 链: 像素级颗粒是脆的 (CCD 方向邻点相关为负, 来自 3 抽头锐化; T55c 的网格高斯给的是 +0.3 左右),
    输送方向的锐化较弱 (邻点相关高于 CCD 方向) —— 与真实 RPL 扫描同号 (真实 Tri-X: y −0.06…−0.14, x −0.02…+0.18)。"""
    g = np.full((256, 256, 3), 0.18)
    o = sg.apply_to_image(g, "kodak_trix_400", "xtol", format=256 * 20 / 1000, film_gauge="120", mode="full", seed=3,
                          method="closed_form").encoded[..., 1] * 255
    r = o[8:-8, 8:-8] - o[8:-8, 8:-8].mean()
    v = r.var()
    rho_y = np.mean(r[1:] * r[:-1]) / v; rho_x = np.mean(r[:, 1:] * r[:, :-1]) / v
    assert rho_y < -0.05 and rho_x > rho_y + 0.05
