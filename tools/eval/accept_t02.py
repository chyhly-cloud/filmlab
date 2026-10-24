"""任务 T02 (EPFL 冲印店扫描档案) 的验收 (规格见 docs/tasks/T02_LAB_PROFILE.md)。由验收方运行, 与拟合程序独立。

档案参数 (全部是扫描 / 冲印店过程的现有设置, 胶片参数不动):
  latd_normal       正常场景的平均密度偏移 (engine.LATD_NORMAL, 现值 −0.14)
  latd_limit        逐帧亮度校正上限 (Variation.latd_limit, 现值 0.12 D)
  highlight_aim     高光侧落在相纸上的目标反射率 (timing.HIGHLIGHT_AIM, 现值 0.65)
  shoulder_white    额外高光肩部的渐近白 (timing.SHOULDER_WHITE, 现值 0.93)
  color_strength    逐帧颜色校正力度 (Params.scan_color_strength, 现值 None = 随机 0.85–1.0; 标称取 0.925)
  conditioning      输出调整 "lab" (矩阵 + tint, 现值) 或 "none" (离散选择)
评估: next 代际、标称胶片变化 (roll=None)、grain=0、**实际自动定光** (timing="frame"), 对齐到真实扫描:
  全图 ΔE00 中位; 光源旁 (传感器饱和区外 0–300 µm) ΔE00 中位。
判定 (全部满足才通过): 参数在范围内且不贴边界; 3200K 灰卡物理参考彩度 < 0.5;
  EPFL 留出: 全图平均改善 ≥ 0.5、光源旁平均改善 ≥ 0.5、任何分组变差 ≤ 0.5、任何场景变差 ≤ 1.5。
  咖啡店 (他店) 只报告, 不作判据 (档案只代表 EPFL 这家店)。

用法: FILMLAB_DATA=… python3 accept_t02.py 参数.json 输出.json      (参数.json 写 {} 即为基准)
"""
import json, os, sys, zlib, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "tools/eval"), str(REPO)]

from accept_calibration import TRAIN, HOLDOUT       # 与 T01 相同的拆分
HOLDOUT_EPFL = {k: v for k, v in HOLDOUT.items() if k != "cafe"}
BOUNDS = {"latd_normal": (-0.40, 0.20), "latd_limit": (0.02, 0.50), "highlight_aim": (0.45, 0.90),
          "shoulder_white": (0.85, 0.995), "color_strength": (0.0, 1.0)}
CURRENT = {"latd_normal": -0.14, "latd_limit": 0.12, "highlight_aim": 0.65, "shoulder_white": 0.93,
           "color_strength": 0.925, "conditioning": "lab"}


def apply(params):
    """在本进程内设置档案 (每个任务一个新进程); 返回额外的 Params 关键字。"""
    from dataclasses import replace
    from filmlab import engine, variation, generation
    from filmlab.scanning import timing
    p = {**CURRENT, **params}
    engine.LATD_NORMAL = p["latd_normal"]
    timing.HIGHLIGHT_AIM = p["highlight_aim"]
    timing.SHOULDER_WHITE = p["shoulder_white"]
    s0 = variation.sample
    variation.sample = lambda *a, **kw: replace(s0(*a, **kw), latd_limit=p["latd_limit"])
    generation.use("next")
    return {"scan_color_strength": p["color_strength"],
            **({"output_conditioning": None} if p["conditioning"] == "none" else {})}


def scene_job(args):
    params, sid = args
    extra = apply(params)
    import colour, rawpy
    import oracle_timing as ot, pair_compare as pc
    from scipy.ndimage import distance_transform_edt, zoom
    from skimage.transform import warp, ProjectiveTransform
    from filmlab import rawio
    from filmlab.engine import Engine, Params
    from filmlab.recipes import recipe
    from filmlab.rawio import film_metering
    from filmlab.io import srgb_encode
    Hm, real = ot.geometry(sid)
    rawp, _, bias = ot.sources(sid)
    rgb = pc.load_digital(rawp, 1400)[0]
    seed = zlib.crc32(sid.encode()) % 1000
    m, _ = film_metering(rgb, rng=np.random.default_rng(seed), bias_ev=bias)
    eng = Engine(Params(**{**recipe("cinestill_800t"), "timing": "frame", "roll": None, "frame": seed % 36,
                           "grain": 0, **extra}))
    sim = srgb_encode(np.clip(eng.render(m), 0, 1))
    tf = ProjectiveTransform(Hm)
    w = warp(sim, tf.inverse, output_shape=real.shape[:2], order=1, cval=np.nan)
    ok = np.all(np.isfinite(w), -1)
    lab = lambda x: colour.XYZ_to_Lab(colour.sRGB_to_XYZ(np.clip(x, 0, 1)))
    de = colour.delta_E(lab(np.where(ok[..., None], w, 0)), lab(real), method="CIE 2000")
    with rawpy.imread(rawp) as r:
        sat = rawio.sensor_saturation(r)
    near_de = np.nan
    if sat is not None:
        s = zoom(sat.astype(float), (m.shape[0] / sat.shape[0], m.shape[1] / sat.shape[1]), order=1) > 0.25
        core = warp(s.astype(float), tf.inverse, output_shape=real.shape[:2], order=1, cval=0) > 0.5
        d = distance_transform_edt(~core) * 36000.0 / max(real.shape[:2])
        near = ok & (d > 0) & (d <= 300)
        if near.sum() > 100:
            near_de = float(np.median(de[near]))
    return sid, {"full": float(np.median(de[ok])), "near": near_de}


def gray_job(params):
    extra = apply(params)
    import colour
    from filmlab import lights, spectral
    from filmlab.engine import Engine, Params
    from filmlab.materials import emulsion
    from filmlab.recipes import recipe
    ref = emulsion("kodak_vision3_500t").stock.reference_illuminant
    rgb = np.tile(0.184 * lights.light_rgb(spectral.illuminant(ref)), (32, 32, 1))
    eng = Engine(Params(**{**recipe("cinestill_800t_reference"), "roll": None, "grain": 0, "scene_lights": None,
                           "scene_illuminant": ref, "scan_color_strength": 1.0}))
    lab = colour.XYZ_to_Lab(colour.sRGB_to_XYZ(np.clip(eng.render(rgb)[16, 16], 0, 1), apply_cctf_decoding=False))
    return float(np.hypot(lab[1], lab[2]))


def evaluate(params, scenes=None):
    import multiprocessing as mp
    scenes = scenes or (list(TRAIN) + list(HOLDOUT))
    with mp.get_context("spawn").Pool(int(os.environ.get("NPROC", "4")), maxtasksperchild=1) as p:
        r = dict(p.map(scene_job, [(params, s) for s in scenes]))
        gray = p.apply(gray_job, (params,))
    return {"params": {**CURRENT, **params}, "scenes": r, "gray_chroma": gray}


def verdict(res, base):
    p, msg, ok = res["params"], [], True
    for k, (lo, hi) in BOUNDS.items():
        if p[k] != CURRENT[k] and not (lo + 0.02 * (hi - lo) < p[k] < hi - 0.02 * (hi - lo)):
            ok = False; msg.append(f"{k}={p[k]:.3f} 越界或贴边界 [{lo}, {hi}]")
    if res["gray_chroma"] >= 0.5:
        ok = False; msg.append(f"灰卡彩度 {res['gray_chroma']:.2f}")
    S, B = res["scenes"], base["scenes"]
    dfull = {s: S[s]["full"] - B[s]["full"] for s in HOLDOUT_EPFL}
    dnear = {s: S[s]["near"] - B[s]["near"] for s in HOLDOUT_EPFL if np.isfinite(S[s]["near"]) and np.isfinite(B[s]["near"])}
    mf, mn = float(np.mean(list(dfull.values()))), float(np.mean(list(dnear.values())))
    if mf > -0.5:
        ok = False; msg.append(f"留出全图平均改善 {-mf:.2f} < 0.5")
    if mn > -0.5:
        ok = False; msg.append(f"留出光源旁平均改善 {-mn:.2f} < 0.5")
    groups = {g: float(np.mean([dfull[s] for s in HOLDOUT_EPFL if HOLDOUT_EPFL[s] == g])) for g in set(HOLDOUT_EPFL.values())}
    for g, v in groups.items():
        if v > 0.5:
            ok = False; msg.append(f"留出分组 {g} 变差 {v:.2f}")
    for s, v in dfull.items():
        if v > 1.5:
            ok = False; msg.append(f"留出场景 {s} 变差 {v:.2f}")
    summ = {"holdout_full_change": mf, "holdout_near_change": mn, "holdout_groups": groups,
            "train_full_change": float(np.mean([S[s]["full"] - B[s]["full"] for s in TRAIN])),
            "cafe_full_change (他店, 只报告)": S["cafe"]["full"] - B["cafe"]["full"] if "cafe" in S else None}
    return ok, msg, summ


if __name__ == "__main__":
    params = json.load(open(sys.argv[1]))
    res = evaluate(params)
    json.dump(res, open(sys.argv[2], "w"), indent=1, ensure_ascii=False)
    base_f = Path(sys.argv[2]).with_name("t02_baseline.json")
    if params and base_f.exists():
        ok, msg, summ = verdict(res, json.load(open(base_f)))
        print(json.dumps(summ, ensure_ascii=False, indent=1))
        print("通过" if ok else "不通过: " + "; ".join(msg))
    else:
        f = lambda keys, k: np.nanmean([res["scenes"][s][k] for s in keys])
        print("灰卡", round(res["gray_chroma"], 3), "| 训练 全图", round(f(TRAIN, "full"), 3), "光源旁", round(f(TRAIN, "near"), 3),
              "| 留出(EPFL) 全图", round(f(HOLDOUT_EPFL, "full"), 3), "光源旁", round(f(HOLDOUT_EPFL, "near"), 3))
