"""胶片参数校准的验收 (任务 T01, 规格见 docs/tasks/T01_FILM_CALIBRATION.md)。由验收方运行, 与拟合程序独立。

输入: 参数 JSON {"tau": C-41 显影程度, "dir": DIR 强度倍数, "diffuse_fraction": 光晕漫射比例}; 缺省 = 现行值。
评估: next 代际, 标称胶片变化 (roll=None), grain=0; 每个场景用定光预言机 O1 (只给三个印放光号) 后的 ΔE00 中位。
判定 (全部满足才通过):
  1. 参数在物理范围内, 且不贴边界 (距边界 > 范围的 2%);
  2. 3200K 灰卡在物理参考中彩度 < 0.5;
  3. ECN-2 原生工艺下中性灰阶的 Status M 与现行相差 ≤ 0.01 D (τ、DIR 不得改变 Kodak 中性曲线);
  4. 留出组: 平均 ΔE00 改善 ≥ 0.2; 任何留出分组变差 ≤ 0.3; 任何留出场景变差 ≤ 0.8。

用法: FILMLAB_DATA=数据目录 python3 accept_calibration.py 参数.json 输出.json      (参数.json 写 {} 即为基准)
"""
import json, os, sys, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "tools/eval"), str(REPO)]

# 事先固定的拆分 (按光源分组均衡; 拟合方只能使用 TRAIN)
TRAIN = {"R1-00-1": "夜景", "R1-02-3": "夜景", "R1-04-5": "夜景", "R1-06-7": "夜景", "R1-07-8": "黄昏",
         "R1-26-27": "钨丝灯箱", "R1-28-29": "钨丝灯箱", "R1-31-32": "白光灯管", "R1-33-34": "白光灯管",
         "R1-21-22": "品红灯管", "R1-25-26": "品红灯管"}
HOLDOUT = {"R1-01-2": "夜景", "R1-05-6": "夜景", "R1-03-4": "黄昏", "R1-27-28": "钨丝灯箱", "R1-30-31": "钨丝灯箱",
           "R1-32-33": "白光灯管", "R1-35-36": "白光灯管", "R1-23-24": "品红灯管", "R1-20-21": "色卡",
           "cafe": "他店日光"}
TAU0 = 1.0 / 0.85
BOUNDS = {"tau": (1.0, 1.3), "dir": (0.6, 1.4), "diffuse_fraction": (0.45, 0.75)}
CURRENT = {"tau": TAU0, "dir": 1.0, "diffuse_fraction": 0.6}


def apply(params):
    from dataclasses import replace
    from filmlab import chemistry, variation, generation
    p = {**CURRENT, **params}
    run0 = chemistry.run

    def run(e, process, logH, blur_px=0.0, tau_factor=1.0):
        if e.native_process == "ECN-2" and any(isinstance(s, chemistry.Develop) and s.developer == "C-41 CD"
                                               for s in process.steps):
            tau_factor = tau_factor * p["tau"] / TAU0
        return run0(e, process, logH, blur_px, tau_factor)
    chemistry.run = run
    di0 = chemistry.develop_image
    chemistry.develop_image = lambda e, logH, tau, inhib, blur_px, k="color": di0(e, logH, tau, inhib * p["dir"], blur_px, k)
    s0 = variation.sample
    variation.sample = lambda *a, **kw: replace(s0(*a, **kw), diffuse_fraction=p["diffuse_fraction"])
    generation.use("next")


def scene_job(args):
    params, sid = args
    apply(params)
    import colour, param_sensitivity as ps
    r = ps.evaluate_scene(sid, roll=None)
    k = r["keep"]
    return sid, float(np.median(colour.delta_E(r["S"][k], r["R"][k], method="CIE 2000")))


def constraints_job(params):
    apply(params)
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
    # ECN-2 原生: 中性灰阶 (−3…+3 档) 的 Status M
    ramp = 0.184 * 2.0 ** np.linspace(-3, 3, 25)
    img = (ramp[None, :, None] * lights.light_rgb(spectral.illuminant(ref))[None, None, :]).repeat(4, 0)
    e2 = Engine(Params(film="kodak_vision3_500t", process="ECN-2", roll=None, grain=0, scene_lights=None,
                       scene_illuminant=ref))
    sm = np.asarray(e2.develop_negative(img).status_m())[2]
    return float(np.hypot(lab[1], lab[2])), sm.tolist()


def evaluate(params):
    import multiprocessing as mp
    scenes = list(TRAIN) + list(HOLDOUT)
    with mp.get_context("spawn").Pool(int(os.environ.get("NPROC", "4")), maxtasksperchild=1) as p:
        de = dict(p.map(scene_job, [(params, s) for s in scenes]))
        chroma, sm = p.apply(constraints_job, (params,))
    return {"params": {**CURRENT, **params}, "dE": de, "gray_chroma": chroma, "statusM_ecn2": sm}


def verdict(res, base):
    p, msg, ok = res["params"], [], True
    for k, (lo, hi) in BOUNDS.items():
        margin = 0.02 * (hi - lo)
        if not (lo + margin < p[k] < hi - margin) and p[k] != CURRENT[k]:
            ok = False; msg.append(f"{k}={p[k]:.3f} 越界或贴边界 [{lo}, {hi}]")
    if res["gray_chroma"] >= 0.5:
        ok = False; msg.append(f"灰卡彩度 {res['gray_chroma']:.2f} ≥ 0.5")
    dsm = float(np.max(np.abs(np.array(res["statusM_ecn2"]) - np.array(base["statusM_ecn2"]))))
    if dsm > 0.01:
        ok = False; msg.append(f"ECN-2 中性 Status M 变化 {dsm:.3f} D > 0.01")
    d = {s: res["dE"][s] - base["dE"][s] for s in HOLDOUT}
    mean = float(np.mean(list(d.values())))
    groups = {g: float(np.mean([d[s] for s in HOLDOUT if HOLDOUT[s] == g])) for g in set(HOLDOUT.values())}
    if mean > -0.2:
        ok = False; msg.append(f"留出平均改善 {-mean:.2f} < 0.2")
    for g, v in groups.items():
        if v > 0.3:
            ok = False; msg.append(f"留出分组 {g} 变差 {v:.2f} > 0.3")
    for s, v in d.items():
        if v > 0.8:
            ok = False; msg.append(f"留出场景 {s} 变差 {v:.2f} > 0.8")
    train = float(np.mean([res["dE"][s] - base["dE"][s] for s in TRAIN]))
    return ok, msg, {"holdout_mean_change": mean, "holdout_groups": groups, "train_mean_change": train,
                     "statusM_max_change": dsm}


if __name__ == "__main__":
    params = json.load(open(sys.argv[1]))
    res = evaluate(params)
    json.dump(res, open(sys.argv[2], "w"), indent=1, ensure_ascii=False)
    base_f = Path(sys.argv[2]).with_name("baseline.json")
    if params and base_f.exists():
        ok, msg, summ = verdict(res, json.load(open(base_f)))
        print(json.dumps(summ, ensure_ascii=False, indent=1))
        print("通过" if ok else "不通过: " + "; ".join(msg))
    else:
        print(json.dumps({k: res[k] for k in ("gray_chroma",)}, ensure_ascii=False),
              "训练平均", round(np.mean([res["dE"][s] for s in TRAIN]), 3),
              "留出平均", round(np.mean([res["dE"][s] for s in HOLDOUT]), 3))
