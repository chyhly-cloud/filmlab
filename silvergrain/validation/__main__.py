"""silvergrain 独立检验: python -m silvergrain.validation [--quick] [--full-t50] [--only t46,t47,...] [输出.json]

各项 (只用颗粒数据, 不用任何 800T / filmlab 管线):
  t46       38 种胶片官方 RMS 颗粒度的留一 (T46)               — 与 data/t46_loo_reference.json 比较 (应逐位相同)
  t47       Schmidt Tri-X / D-76 四孔径、19 级密度、Kodak 48 µm (T47) — 与 data/t47_reference.json
  t47b      黑白片官方 RMS (T-Max) 与 Schmidt 6 种片 (T47b)        — 与 data/t47b_reference.json
  t50       显影液 (T50): D-76 参照与数据库参数的预测 (--full-t50: 全网格留一, 约 30 分钟)
  t53       颗粒粗细 k 的标度律 (T53, 闭式)                      — 与 data/t53_reference.json
  fastpath  黑白快速路径 (closed_form) 对显式银粒
  realscan  真实扫描的记录值 (T51b / T52 / T54-RPL / T51 谱形): 同码值、胶片侧、谱形 (不下载)
结果写到 silvergrain/validation/results/report.json (或给定路径); 断点续跑: 已有的项不重算 (--redo 重算)。"""
import json
import sys
import time
from pathlib import Path

ORDER = ("t47", "t47b", "t53", "t50", "fastpath", "realscan", "t46")


def _default(o):
    import numpy as np
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    return str(o)


def main(argv):
    quick = "--quick" in argv
    dst = Path(next((a for a in argv if a.endswith(".json")), Path(__file__).resolve().parent / "results" / "report.json"))
    only = next((a.split("=", 1)[1].split(",") for a in argv if a.startswith("--only=")), ORDER)
    dst.parent.mkdir(parents=True, exist_ok=True)
    rep = json.loads(dst.read_text()) if dst.exists() and "--redo" not in argv else {}
    for name in ORDER:
        if name not in only or name in rep:
            continue
        t = time.time()
        print(f"[{name}] ...", flush=True)
        if name == "t46":
            from .t46_official import run
            r = run(full=not quick)
        elif name == "t47":
            from .t47_schmidt import schmidt_four_apertures
            r = schmidt_four_apertures()
        elif name == "t47b":
            from .t47_schmidt import t47b
            r = t47b()
        elif name == "t50":
            from .t50_developers import full, quick as q
            r = dict(quick=q())
            if "--full-t50" in argv:
                r["full"] = full()
        elif name == "t53":
            from .t53_scale import run
            r = run()
        elif name == "fastpath":
            from .fastpath import run
            r = run(n=96 if quick else 160)
        elif name == "realscan":
            from .realscan import run
            r = run()
        r = json.loads(json.dumps(r, default=_default))
        r["_seconds"] = round(time.time() - t, 1)
        rep[name] = r
        dst.write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=_default))
        print(f"[{name}] {r['_seconds']} s", flush=True)
    from .summary import summarize
    print(summarize(rep))


if __name__ == "__main__":
    main(sys.argv[1:])
