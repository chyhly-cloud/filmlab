"""T28: 截断块档数按 RAW 裙边 / 同图近场 PSF 推算 (rawio.HIGHLIGHT_RECON = "skirt", 代际 t28_skirt)。"""
import numpy as np
from scipy.signal import fftconvolve

from filmlab import generation, rawio


def _scene(disk_px=0.0, seed=1):
    """已知 PSF 的合成图: 流水线像素 = 2×2 个半尺寸像素 (在半尺寸像素上截断), 返回 (lin RGB, clip_frac, 未截断的真实值)。"""
    rng = np.random.default_rng(seed)
    H, W, U = 300, 400, 4
    yy, xx = np.mgrid[-40:41, -40:41] / (U / 2)
    psf = np.exp(-np.hypot(yy, xx) ** 2 / (2 * 0.8 ** 2)) + 0.002 * (1 + (np.hypot(yy, xx) / 1.5) ** 2) ** -1.5
    psf /= psf.sum()
    fine = np.zeros((H * U, W * U))
    dy, dx = np.mgrid[-40:41, -40:41]
    disk = (np.hypot(dy, dx) <= disk_px * U).astype(float)
    pts = []
    for i in range(50):
        y, x = rng.integers(40, H - 40) * U + rng.integers(0, U), rng.integers(40, W - 40) * U + rng.integers(0, U)
        pts.append((y, x, 2 ** rng.uniform(1.0, 6.0) if i < 15 else rng.uniform(0.15, 0.8), i < 15))   # 前 15 个截断
    for y, x, a, clipped in pts:
        if clipped and disk_px > 0:
            fine[y - 40:y + 41, x - 40:x + 41] += a * disk / disk.sum() * np.pi * (disk_px * U) ** 2
        else:
            fine[y, x] += a
    hs = fftconvolve(fine, psf, mode="same").reshape(H * 2, U // 2, W * 2, U // 2).mean((1, 3))
    k = np.median([hs[y // (U // 2), x // (U // 2)] / a for y, x, a, c in pts if not c])   # 未截断点: 峰值 ≈ 幅度 × 0.9
    hs = hs / k * 0.9 + 0.002
    full = hs.reshape(H, 2, W, 2).mean((1, 3))
    lin = np.minimum(hs, 1.0).reshape(H, 2, W, 2).mean((1, 3))
    cf = (hs >= 1.0).reshape(H, 2, W, 2).mean((1, 3))
    return np.repeat(lin[..., None], 3, -1), cf, full


def test_prior_path_unchanged_without_skirt_lin():
    rgb, cf, _ = _scene()
    a = rawio.reconstruct_highlights(rgb.copy(), cf, np.random.default_rng(0), None, 0.0)
    b = rawio.reconstruct_highlights(rgb.copy(), cf, np.random.default_rng(0), None, 0.0, skirt_lin=None)
    assert np.array_equal(a, b)


def test_fallback_without_psf_points_equals_prior():
    rgb, cf, _ = _scene()
    flat = rgb.copy()
    flat[(cf == 0)] = 0.002                      # 去掉所有未截断亮点 → 没有 PSF → 整图回退
    rawio.SKIRT_LOG = []
    try:
        a = rawio.reconstruct_highlights(flat.copy(), cf, np.random.default_rng(0), 5.0, 0.0)
        b = rawio.reconstruct_highlights(flat.copy(), cf, np.random.default_rng(0), 5.0, 0.0, skirt_lin=flat.copy())
        assert rawio.SKIRT_LOG[0]["psf"]["why"] == "psf_points"
    finally:
        rawio.SKIRT_LOG = None
    assert np.array_equal(a, b)


def test_skirt_recovers_known_brightness():
    """点源与半径 2 像素的圆盘: 估出的档数与真实值 (穹顶峰值处应写入的值 / 截断电平) 的中位差在 1 档以内。"""
    from scipy import ndimage
    for disk in (0.0, 2.0):
        rgb, cf, full = _scene(disk)
        rawio.SKIRT_LOG = []
        try:
            rawio.reconstruct_highlights(rgb.copy(), cf, np.random.default_rng(0), None, 0.0, skirt_lin=rgb.copy())
            lg = rawio.SKIRT_LOG[0]
        finally:
            rawio.SKIRT_LOG = None
        assert lg["psf"]["why"] == "ok"
        lab, n = ndimage.label(ndimage.binary_dilation(cf > 0.25))
        diff = []
        for k, (b, sl) in enumerate(zip(lg["blocks"], ndimage.find_objects(lab)), 1):
            if b["est"] is None or b["why"] != "ok":
                continue
            blob = lab[sl] == k
            Yo = rgb[sl][..., 0][blob]
            d = ndimage.distance_transform_edt(blob)
            i = np.argmax(d[blob])
            f = cf[sl][blob][i]
            true = np.log2((full[sl][blob][i] - Yo[i] + f * 1.0) / (f * np.percentile(Yo, 90)))
            diff.append(b["est"] - true)
        assert len(diff) >= 8
        assert abs(np.median(diff)) < 1.0, (disk, np.median(diff))


def test_generation_table():
    generation.use("t28_skirt")
    try:
        assert rawio.HIGHLIGHT_RECON == "skirt"
        assert "halo_forward" in generation.GENERATIONS["t28_skirt"][1]
    finally:
        generation.use("t26_fwd")
    assert rawio.HIGHLIGHT_RECON == "prior"
    generation.use("current")
