"""任务 T03 (光晕斜程衰减) 的验收 (规格见 docs/tasks/T03_HALO_OBLIQUE.md)。由验收方运行, 与实现独立。

T03 没有自由参数 (衰减由乳剂已有透射率按深度平均算出), 不做拟合, 因此全部场景都是检验场景, 无训练 / 留出之分。
评估对象是一个代际 (候选 `next_oblique` 对比基准 `next`), 配置: 标称胶片 (roll=None)、grain=0、实际自动定光。
  1. 红带形状: 4 个有效窗口场景 (red_band.py), 150–1500 µm 红色度 r 相对真实扫描的 RMS 差;
  2. 全图 ΔE00 中位 (21 个 EPFL / 咖啡店场景) 与光源旁 (0–300 µm) ΔE00 中位;
  3. 3200K 灰卡在物理参考中的彩度。
判定 (全部满足才通过): 红带 RMS 至少 3/4 场景改善且平均改善 ≥ 0.03; 全图平均不变差 (≤ +0.05),
  任何分组变差 ≤ 0.3, 任何场景变差 ≤ 1.0; 光源旁平均不变差 (≤ +0.05); 灰卡彩度 < 0.5。
(另由验收方跑 current / legacy / next 严格逐位回归与全部测试, 并交用户目视。)

用法: FILMLAB_DATA=… python3 accept_t03.py 代际名 输出.json        (先用 next 生成 t03_baseline.json)
"""
import json, os, sys, zlib, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "tools/eval"), str(REPO)]

from accept_calibration import TRAIN, HOLDOUT
SCENES = {**TRAIN, **HOLDOUT}
BAND_SCENES = ["R1-00-1", "R1-03-4", "R1-04-5", "R1-07-8"]
BAND_BINS = [25, 50, 100, 150, 200, 300, 400, 500, 700, 1000, 1500]


def scene_job(args):
    gen, sid = args
    import accept_t02
    accept_t02.apply = lambda params, _g=gen: _use(_g)
    return accept_t02.scene_job(({}, sid))


def _use(gen):
    from filmlab import generation
    generation.use(gen)
    return {}


def band_job(args):
    gen, out, sid = args
    os.environ["GENERATION"] = gen
    os.environ["TAG"] = f"_{gen}"
    import red_band
    _, r = red_band.job((out, sid))
    rr = lambda v: v[0] / sum(v)
    e = [(rr(r["output"][i]) - rr(r["real"][i])) ** 2 for i, b in enumerate(BAND_BINS)
         if b >= 150 and r["real"][i] is not None and r["output"][i] is not None]
    return sid, float(np.sqrt(np.mean(e)))


def gray_job(gen):
    _use(gen)
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


def evaluate(gen, outdir):
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(int(os.environ.get("NPROC", "4")), maxtasksperchild=1) as p:
        sc = dict(p.map(scene_job, [(gen, s) for s in SCENES]))
        band = dict(p.map(band_job, [(gen, outdir, s) for s in BAND_SCENES]))
        gray = p.apply(gray_job, (gen,))
    return {"generation": gen, "scenes": sc, "band_rms": band, "gray_chroma": gray}


def verdict(res, base):
    ok, msg = True, []
    db = {s: res["band_rms"][s] - base["band_rms"][s] for s in BAND_SCENES}
    if sum(v < 0 for v in db.values()) < 3 or np.mean(list(db.values())) > -0.03:
        ok = False; msg.append(f"红带 RMS 改善场景 {sum(v < 0 for v in db.values())}/4, 平均变化 {np.mean(list(db.values())):+.3f}")
    S, B = res["scenes"], base["scenes"]
    df = {s: S[s]["full"] - B[s]["full"] for s in SCENES}
    mf = float(np.mean(list(df.values())))
    if mf > 0.05:
        ok = False; msg.append(f"全图平均变差 {mf:+.2f}")
    groups = {g: float(np.mean([df[s] for s in SCENES if SCENES[s] == g])) for g in set(SCENES.values())}
    for g, v in groups.items():
        if v > 0.3:
            ok = False; msg.append(f"分组 {g} 变差 {v:+.2f}")
    for s, v in df.items():
        if v > 1.0:
            ok = False; msg.append(f"场景 {s} 变差 {v:+.2f}")
    dn = [S[s]["near"] - B[s]["near"] for s in SCENES if np.isfinite(S[s]["near"]) and np.isfinite(B[s]["near"])]
    mn = float(np.mean(dn))
    if mn > 0.05:
        ok = False; msg.append(f"光源旁平均变差 {mn:+.2f}")
    if res["gray_chroma"] >= 0.5:
        ok = False; msg.append(f"灰卡彩度 {res['gray_chroma']:.2f}")
    return ok, msg, {"band_rms_change": db, "full_mean_change": mf, "groups": groups, "near_mean_change": mn,
                     "gray_chroma": res["gray_chroma"]}


if __name__ == "__main__":
    gen, out = sys.argv[1], sys.argv[2]
    outdir = str(Path(out).parent)
    res = evaluate(gen, outdir)
    json.dump(res, open(out, "w"), indent=1, ensure_ascii=False)
    base_f = Path(out).with_name("t03_baseline.json")
    if gen != "next" and base_f.exists():
        ok, msg, summ = verdict(res, json.load(open(base_f)))
        print(json.dumps(summ, ensure_ascii=False, indent=1))
        print("通过" if ok else "不通过: " + "; ".join(msg))
    else:
        print("红带 RMS", {k: round(v, 3) for k, v in res["band_rms"].items()},
              "| 全图平均", round(np.mean([v["full"] for v in res["scenes"].values()]), 3),
              "| 光源旁平均", round(np.nanmean([v["near"] for v in res["scenes"].values()]), 3), "| 灰卡", round(res["gray_chroma"], 3))
