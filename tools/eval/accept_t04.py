"""任务 T04 (冲印店逐帧校色力度的实测 + 误差账本) 的验收 (规格见 docs/tasks/T04_LAB_TIMING_MEASUREMENT.md)。由验收方运行, 与实现独立。

  1. filmlab/ 与基准提交逐字节一致 (BASE 环境变量, 默认 handoff 分支上规格提交);
  2. 账本 auto 阶段全图 ΔE 逐场景与 tools/eval/t03_baseline.json 闭合 (< 1e-6);
  3. 由 frames 的数值独立重算 k (点估计: 普通最小二乘; 区间: 对帧重抽样 2000 次的 2.5 / 97.5 百分位, 本文件自己的实现),
     与报告值比较 (点估计 < 1e-6, 区间端点 < 0.02), 并按四条规则独立得出结论, 必须与 verdict 一致;
  4. 随机抽 3 个场景在新进程里重跑 lab_timing.py, x 与账本数值完全一致 (RERUN=0 跳过)。
全部测试由验收方另行运行。

用法: FILMLAB_DATA=… [REPO=被测仓库] [BASE=提交] python3 accept_t04.py t04_result.json
"""
import json, os, subprocess, sys, tempfile
from pathlib import Path
import numpy as np

REPO = Path(os.environ.get("REPO", Path(__file__).resolve().parents[2]))
BASE = os.environ.get("BASE", "origin/handoff/recovered-research-2026-10-23")
BASELINE = Path(__file__).resolve().parent / "t03_baseline.json"
RULES = [((-0.25, 0.25), "不逐帧校色"), ((0.75, 1.25), "力度与模型相当"), ((0.25, 0.75), "部分校色")]


def ols_slope(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    xc = x - x.mean()
    return float((xc * (y - y.mean())).sum() / (xc * xc).sum())


def pooled_slope(cx, cy):
    """R、B 两分量各自去均值后合并成一个斜率 (各分量有自己的截距)。"""
    cx, cy = np.asarray(cx, float), np.asarray(cy, float)
    xc, yc = cx - cx.mean(0), cy - cy.mean(0)
    return float((xc * yc).sum() / (xc * xc).sum())


def boot(fn, n, seed=0, reps=2000):
    rng = np.random.default_rng(seed)
    v = []
    for _ in range(reps):
        i = rng.integers(0, n, n)
        try:
            s = fn(i)
        except ZeroDivisionError:
            continue
        if np.isfinite(s):
            v.append(s)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def rule(lo, hi):
    for (a, b), name in RULES:
        if a <= lo and hi <= b:
            return name
    return "数据不足以区分"


def check(res):
    ok, msg = True, []
    # 1. filmlab/ 不变
    d = subprocess.run(["git", "-C", str(REPO), "diff", "--stat", BASE, "--", "filmlab/"], capture_output=True, text=True)
    if d.returncode or d.stdout.strip():
        ok = False; msg.append(f"filmlab/ 与 {BASE} 不一致: {d.stdout.strip() or d.stderr.strip()}")
    # 2. auto 闭合
    base = json.load(open(BASELINE))["scenes"]
    for sid, b in base.items():
        led = res["ledger"].get(sid, {}).get("auto", {}).get("全图")
        if led is None or abs(led["dE"] - b["full"]) >= 1e-6:
            ok = False; msg.append(f"auto 未闭合 {sid}: {None if led is None else led['dE']} vs {b['full']}")
    # 3. k 独立重算 (咖啡店不进入回归)
    fr = {s: v for s, v in res["frames"].items() if s != "cafe"}
    sids = sorted(fr)
    ca = np.array([fr[s]["c_auto"] for s in sids]); cl = np.array([fr[s]["c_lab"] for s in sids])
    ka = np.array([fr[s]["K_auto"] for s in sids]); kl = np.array([fr[s]["K_auto"] + fr[s]["x"][0] for s in sids])
    for s in sids:
        exp = np.asarray(fr[s]["c_auto"]) + np.asarray(fr[s]["x"][1:])
        if np.max(np.abs(exp - fr[s]["c_lab"])) > 1e-9:
            ok = False; msg.append(f"c_lab ≠ c_auto + x: {s}")
    n = len(sids)
    mine = {"R": (ols_slope(ca[:, 0], cl[:, 0]), boot(lambda i: ols_slope(ca[i, 0], cl[i, 0]), n, 1)),
            "B": (ols_slope(ca[:, 1], cl[:, 1]), boot(lambda i: ols_slope(ca[i, 1], cl[i, 1]), n, 2)),
            "pooled": (pooled_slope(ca, cl), boot(lambda i: pooled_slope(ca[i], cl[i]), n, 3)),
            "K": (ols_slope(ka, kl), boot(lambda i: ols_slope(ka[i], kl[i]), n, 4))}
    for key, (pt, (lo, hi)) in mine.items():
        rp, rlo, rhi = res["k"][key]
        if abs(pt - rp) >= 1e-6:
            ok = False; msg.append(f"k[{key}] 点估计 {rp:.6f} vs 验收方 {pt:.6f}")
        if abs(lo - rlo) >= 0.02 or abs(hi - rhi) >= 0.02:
            ok = False; msg.append(f"k[{key}] 区间 [{rlo:.3f}, {rhi:.3f}] vs 验收方 [{lo:.3f}, {hi:.3f}]")
    v = rule(*mine["pooled"][1])
    if v not in res.get("verdict", ""):
        ok = False; msg.append(f"结论不一致: 报告 {res.get('verdict')!r}, 验收方 {v!r}")
    return ok, msg, {k: [round(p, 4), round(a, 4), round(b, 4)] for k, (p, (a, b)) in mine.items()}, v


def rerun(res, k=3, seed=0):
    """单场景输出与全量结果同结构 (只含该场景): {"frames": {sid: …} (咖啡店无), "ledger": {sid: …}}。"""
    bad = []
    for sid in np.random.default_rng(seed).choice(sorted(res["ledger"]), k, replace=False):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "r.json"
            subprocess.run([sys.executable, str(REPO / "tools/eval/lab_timing.py"), sid, str(out)], check=True,
                           cwd=str(REPO), capture_output=True)
            r = json.load(open(out))
        if r["ledger"][sid] != res["ledger"][sid]:
            bad.append(sid)
        if sid in res["frames"] and r["frames"][sid]["x"] != res["frames"][sid]["x"]:
            bad.append(sid)
    return sorted(set(bad))


if __name__ == "__main__":
    res = json.load(open(sys.argv[1]))
    ok, msg, mine, v = check(res)
    print("验收方重算 k:", json.dumps(mine, ensure_ascii=False), "| 结论:", v)
    if os.environ.get("RERUN", "1") != "0":
        bad = rerun(res)
        if bad:
            ok = False; msg.append(f"重跑不一致: {bad}")
    print("通过" if ok else "不通过: " + "; ".join(msg))
    sys.exit(0 if ok else 1)
