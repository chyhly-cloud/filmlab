"""任务 T05 (冲印店固定光号偏差的纠正) 的验收 (规格见 docs/tasks/T05_LAB_BALANCE.md)。由验收方运行, 与实现独立。

Δ 由 T04 训练集 O1 偏移的中位数事先固定, 不在本任务拟合; 因此只按留出集判定 (EPFL 留出 9 帧), 咖啡店与训练集只报告。
配置与 T03 / T04 相同 (accept_t03.scene_job: 标称配置, roll=None, grain=0, 实际自动定光), 基准 tools/eval/t03_baseline.json (next)。
判定 (全部满足才通过): 代码施加的 Δ 与训练集中位数相差 < 1e-12 (绿通道与斜率不变); EPFL 留出全图平均改善 ≥ 0.5; 任何留出分组变差 ≤ 0.5、
任何留出场景变差 ≤ 1.5; EPFL 留出光源旁平均不变差 (≤ +0.05); 灰卡彩度 < 0.5。
(另由验收方跑 current / legacy / next 严格逐位回归与全部测试, 并交用户目视。)

用法: FILMLAB_DATA=… [REPO=被测仓库] python3 accept_t05.py 代际名 输出.json
"""
import json, os, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get("REPO", HERE.parents[1]))
sys.path[:0] = [str(REPO / "tools/eval"), str(REPO)]

from accept_calibration import TRAIN, HOLDOUT
HOLDOUT_EPFL = {k: v for k, v in HOLDOUT.items() if k != "cafe"}
SCENES = {**TRAIN, **HOLDOUT}


def expected_delta():
    fr = json.load(open(HERE / "t04_result.json"))["frames"]
    return np.median(np.array([fr[s]["x"] for s in TRAIN]), 0)


def code_delta(gen):
    """在被测代码中读出纠正量: 同一合成画面分别用 next 与候选代际做逐帧自动定光, 取 PrintTiming 的差 (K, 红, 蓝);
    同时返回其余量的最大变化 (三通道 K 的不一致、绿通道偏移、r、s、x_knee、x_white; 必须 < 1e-12)。"""
    from filmlab import generation
    from filmlab.engine import Engine, Params
    from filmlab.recipes import recipe
    rgb = np.random.default_rng(0).uniform(0.02, 0.6, (48, 64, 3))
    ts = []
    for g in ("next", gen):
        generation.use(g)
        eng = Engine(Params(**{**recipe("cinestill_800t"), "timing": "frame", "roll": None, "grain": 0}))
        eng.scan(eng.develop_negative(rgb))
        ts.append(eng.scanner.timing)
    a, b = ts
    oa, ra = a.M if a.M is not None else (np.zeros(3), np.ones(3))
    ob, rb = b.M if b.M is not None else (np.zeros(3), np.ones(3))
    dK = np.asarray(b.K, float) - np.asarray(a.K, float)
    other = [np.ptp(dK), abs(ob[1] - oa[1]), np.max(np.abs(np.asarray(rb) - ra)), np.max(np.abs(np.asarray(b.s) - a.s))]
    other += [abs((b.x_knee or 0) - (a.x_knee or 0)), abs((b.x_white or 0) - (a.x_white or 0))]
    return [float(dK.mean()), float(ob[0] - oa[0]), float(ob[2] - oa[2])], float(max(other))


def evaluate(gen):
    import multiprocessing as mp
    import accept_t03
    with mp.get_context("spawn").Pool(int(os.environ.get("NPROC", "3")), maxtasksperchild=1) as p:
        sc = dict(p.map(accept_t03.scene_job, [(gen, s) for s in SCENES]))
        gray = p.apply(accept_t03.gray_job, (gen,))
        delta, other = p.apply(code_delta, (gen,))
    return {"generation": gen, "scenes": sc, "gray_chroma": gray, "code_delta": delta, "code_other_change": other}


def verdict(res, base):
    ok, msg = True, []
    exp = expected_delta()
    if np.max(np.abs(np.asarray(res["code_delta"]) - exp)) >= 1e-12 or res["code_other_change"] >= 1e-12:
        ok = False; msg.append(f"代码 Δ {res['code_delta']} (其他变化 {res['code_other_change']}) ≠ 训练集中位数 {exp.tolist()}")
    S, B = res["scenes"], base["scenes"]
    df = {s: S[s]["full"] - B[s]["full"] for s in SCENES}
    mh = float(np.mean([df[s] for s in HOLDOUT_EPFL]))
    if mh > -0.5:
        ok = False; msg.append(f"EPFL 留出全图平均变化 {mh:+.3f} (需 ≤ −0.5)")
    groups = {g: float(np.mean([df[s] for s in HOLDOUT_EPFL if HOLDOUT_EPFL[s] == g])) for g in set(HOLDOUT_EPFL.values())}
    for g, v in groups.items():
        if v > 0.5:
            ok = False; msg.append(f"留出分组 {g} 变差 {v:+.2f}")
    for s in HOLDOUT_EPFL:                      # 咖啡店 (他店) 按规格只报告, 不判定
        if df[s] > 1.5:
            ok = False; msg.append(f"留出场景 {s} 变差 {df[s]:+.2f}")
    dn = [S[s]["near"] - B[s]["near"] for s in HOLDOUT_EPFL if np.isfinite(S[s]["near"]) and np.isfinite(B[s]["near"])]
    mn = float(np.mean(dn)) if dn else float("nan")
    if dn and mn > 0.05:
        ok = False; msg.append(f"EPFL 留出光源旁平均变差 {mn:+.3f}")
    if res["gray_chroma"] >= 0.5:
        ok = False; msg.append(f"灰卡彩度 {res['gray_chroma']:.2f}")
    report = {"holdout_epfl_full_change": mh, "holdout_groups": groups, "holdout_near_change": mn,
              "holdout_scenes": {s: round(df[s], 3) for s in HOLDOUT}, "cafe_full_change_report_only": round(df["cafe"], 3),
              "train_full_change_in_sample": float(np.mean([df[s] for s in TRAIN])), "gray_chroma": res["gray_chroma"]}
    return ok, msg, report


if __name__ == "__main__":
    gen, out = sys.argv[1], sys.argv[2]
    res = evaluate(gen)
    json.dump(res, open(out, "w"), indent=1, ensure_ascii=False)
    ok, msg, rep = verdict(res, json.load(open(HERE / "t03_baseline.json")))
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    print("通过" if ok else "不通过: " + "; ".join(msg))
    sys.exit(0 if ok else 1)
