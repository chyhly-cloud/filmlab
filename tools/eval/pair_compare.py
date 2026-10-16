"""配对对比: 同一场景的数码原片 → 本系统 CineStill 800T 渲染, 与真实 800T 扫描逐区域比较。

1. 数码原片 (RAW 或 JPEG) → film_metering (固定补偿, 不随机) → cinestill_800t 渲染
2. 渲染结果按特征点 (ORB + RANSAC 单应性) 对齐到真实扫描
3. 在共同区域上按网格比较 Lab: 亮度曲线 (按真实 L* 分档)、暗 / 中 / 亮部的 a, b 偏差、彩度比

用法: python3 tools/eval/pair_compare.py 扫描.jpg 数码原片 [输出目录] [--bias EV]
"""
import sys
import os
import argparse
import numpy as np
import colour
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from filmlab.engine import Engine, Params
from filmlab.io import srgb_encode
from filmlab.recipes import recipe
from filmlab.rawio import load_raw, film_metering, exif_exposure

RAW_EXT = (".dng", ".cr2", ".cr3", ".nef", ".arw", ".raf", ".orf", ".rw2")


def load_digital(path, max_size=1600):
    if path.lower().endswith(RAW_EXT):
        rgb = load_raw(path, max_size=max_size)
        return (rgb[0] if isinstance(rgb, tuple) else rgb), True
    im = Image.open(path).convert("RGB")
    im.thumbnail((max_size, max_size))
    # JPEG: 相机已做白平衡与影调; 只能近似还原为线性 (sRGB 解码)。结论需打折扣。
    return colour.cctf_decoding(np.asarray(im) / 255.0), False


def render(rgb, bias_ev, seed=0, film="cinestill_800t", scanner=None):
    rgb, note = film_metering(rgb, rng=np.random.default_rng(seed), bias_ev=bias_ev)
    kw = recipe("cinestill_800t") if film == "cinestill_800t" else dict(film="kodak_vision3_500t", process="ECN-2")
    if scanner:
        kw["scanner"] = scanner
    eng = Engine(Params(**{**kw, "timing": "frame", "roll": 1, "frame": seed % 36}))
    return np.clip(eng.render(rgb), 0, 1), note


def _gray(a):
    return a @ np.array([0.2126, 0.7152, 0.0722]) if a.ndim == 3 else a


def align(moving, fixed):
    """moving → fixed 的单应性 (在 8 位 sRGB 灰度上做 ORB)。返回 warp 后的 moving 与有效区域掩码。"""
    from skimage.feature import ORB, match_descriptors
    from skimage.measure import ransac
    from skimage.transform import ProjectiveTransform, warp
    from skimage.exposure import equalize_adapthist
    g0, g1 = equalize_adapthist(_gray(moving)), equalize_adapthist(_gray(fixed))
    kp, desc = [], []
    for g in (g0, g1):
        orb = ORB(n_keypoints=3000, fast_threshold=0.03)
        orb.detect_and_extract(g)
        kp.append(orb.keypoints)
        desc.append(orb.descriptors)
    m = match_descriptors(desc[0], desc[1], cross_check=True, max_ratio=0.85)
    src, dst = kp[0][m[:, 0]][:, ::-1], kp[1][m[:, 1]][:, ::-1]      # (x, y)
    model, inl = ransac((src, dst), ProjectiveTransform, min_samples=4, residual_threshold=3, max_trials=5000)
    if model is None or inl.sum() < 25:
        raise RuntimeError(f"对齐失败 (内点 {0 if inl is None else inl.sum()})")
    out = warp(moving, model.inverse, output_shape=fixed.shape[:2], order=1, cval=np.nan)
    return out, np.all(np.isfinite(out), -1), int(inl.sum())


def lab(srgb):
    return colour.XYZ_to_Lab(colour.sRGB_to_XYZ(np.clip(srgb, 0, 1)))   # 输入为编码后的 sRGB


def compare(sim, real, mask, cell=24):
    """网格中位数 Lab, 只用两边都有效的格子。返回 (sim_lab, real_lab) 列表。"""
    h, w = real.shape[:2]
    S, R = [], []
    Ls, Lr = lab(np.where(mask[..., None], sim, 0)), lab(real)
    for y in range(0, h - cell + 1, cell):
        for x in range(0, w - cell + 1, cell):
            if mask[y:y + cell, x:x + cell].mean() < 0.95:
                continue
            S.append(np.median(Ls[y:y + cell, x:x + cell].reshape(-1, 3), 0))
            R.append(np.median(Lr[y:y + cell, x:x + cell].reshape(-1, 3), 0))
    return np.array(S), np.array(R)


def report(S, R, name=""):
    print(f"{name}  格子数 {len(S)}   ΔE76 中位 {np.median(np.linalg.norm(S - R, axis=1)):.1f}")
    bands = [("暗部 L<25", R[:, 0] < 25), ("中间 25–60", (R[:, 0] >= 25) & (R[:, 0] < 60)), ("亮部 ≥60", R[:, 0] >= 60)]
    for nm, b in bands:
        if b.sum() < 5:
            continue
        d = np.median(S[b] - R[b], 0)
        cs, cr = np.hypot(S[b, 1], S[b, 2]), np.hypot(R[b, 1], R[b, 2])
        print(f"   {nm:10s} n={b.sum():4d}  ΔL {d[0]:+5.1f}  Δa {d[1]:+5.1f}  Δb {d[2]:+5.1f}  "
              f"彩度 模拟/真实 {np.median(cs):4.1f}/{np.median(cr):4.1f}")
    # 影调曲线: 按真实 L* 分档, 模拟 L* 的中位数
    edges = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 101])
    row = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        b = (R[:, 0] >= lo) & (R[:, 0] < hi)
        row.append(f"{lo:>2d}+:{np.median(S[b, 0]):5.1f}" if b.sum() >= 3 else f"{lo:>2d}+:  -  ")
    print("   影调 (真实 L* 档 → 模拟 L*):", " ".join(row))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scan"); ap.add_argument("digital"); ap.add_argument("out", nargs="?")
    ap.add_argument("--bias", type=float, default=0.5, help="测光补偿 (档), 默认为拍摄者调查的中位数")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--film", default="cinestill_800t", help="cinestill_800t 或 vision3_500t_ecn2")
    ap.add_argument("--scanner", default=None)
    a = ap.parse_args()
    real = np.asarray(Image.open(a.scan).convert("RGB")).astype(float) / 255.0
    rgb, is_raw = load_digital(a.digital)
    sim_lin, note = render(rgb, a.bias, a.seed, a.film, a.scanner)
    sim = srgb_encode(sim_lin)
    # 对齐在扫描的分辨率上做 (扫描缩到长边 1600)
    s = 1600 / max(real.shape[:2])
    if s < 1:
        real = np.asarray(Image.fromarray((real * 255).astype(np.uint8)).resize(
            (int(real.shape[1] * s), int(real.shape[0] * s)), Image.LANCZOS)).astype(float) / 255.0
    warped, mask, n_in = align(sim, real)
    print(f"{os.path.basename(a.scan)}  ←  {os.path.basename(a.digital)} ({'RAW' if is_raw else 'JPEG'}; {note}; 内点 {n_in})")
    S, R = compare(warped, real, mask)
    report(S, R)
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        side = np.concatenate([np.where(mask[..., None], warped, 0), real], 1)
        stem = os.path.splitext(os.path.basename(a.scan))[0]
        Image.fromarray((np.clip(side, 0, 1) * 255).astype(np.uint8)).save(os.path.join(a.out, stem + "_sim_vs_real.jpg"), quality=90)
    return S, R


if __name__ == "__main__":
    main()
