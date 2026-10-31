"""结构假设的留出验收 (T06 起): 在验收入口 (accept_t03.scene_job, next, roll=None, grain=0) 上施加一个预先命名的变体,
21 个场景对比 tools/eval/t03_baseline.json, 只按留出集判定 (判定与 T05 相同, 无拟合参数, 不需要代码 Δ 检查):
  EPFL 留出全图平均改善 ≥ 0.5; 任何留出分组变差 ≤ 0.5、任何留出场景变差 ≤ 1.5; EPFL 留出光源旁平均不变差 (≤ +0.05);
  灰卡彩度 < 0.5。训练集与咖啡店只报告。

变体 (在每个工作进程内施加, 不改仓库代码):
  no_slope   逐帧定光不做各通道斜率灰平衡 (balance_frame 的 slope_limit = 0; 偏移按 r = 1 一致地计算)
  fixed_exposure  胶片与数码拿到同一份曝光: 胶片曝光 = RAW 线性 × 全程常数, 不逐帧测光。常数 = 训练集 11 帧
             film_metering 倍数的中位数 (log10 = FIXED_LOG_SCALE, 由 tools/eval 的测光诊断在验收方事先算出, 不按 ΔE 选)
  no_satmat  输出调整不用 SATURATION_MATRIX (他店 SillyStill 拟合的经验矩阵, 实为降饱和), 保留 LAB_TINT
  plain      不做输出调整 (物理参考输出: 不用 SATURATION_MATRIX, 也不用 LAB_TINT)
  lab_colour 冲印店颜色处理的实测模型 (T07): 固定光号 (standard) + 逐帧颜色偏移 off = a − k·(底片整帧平均密度 − 档案中灰),
             R、B 各一组 (a, k), 取自 t07_result.json 训练集回归 (主估计量 mean), 照抄不调; 亮度不逐帧调整;
             只替换逐帧自动定光 (timing="frame"), 物理参考配方不受影响
  多个变体用逗号组合, 如 fixed_exposure,no_satmat

用法: FILMLAB_DATA=… python3 accept_variant.py 变体 输出.json
"""
import json, os, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1])]
from accept_calibration import TRAIN, HOLDOUT

VARIANTS = ("no_slope", "fixed_exposure", "no_satmat", "plain", "lab_colour")
# T07 训练集回归 (t07_result.json summary.mean.train_fit), 照抄
LAB_COLOUR = {"R": (-0.02300027899845071, 0.6458938636987954), "B": (0.007245716407750681, 0.3859439484936917)}
FIXED_LOG_SCALE = 0.6129018861240094


def install(variant):
    if "," in variant:
        for v in variant.split(","):
            install(v)
        return
    if variant == "no_satmat":
        from filmlab import scanner
        scanner.SATURATION_MATRIX = np.eye(3)
    elif variant == "plain":
        from filmlab import scanner
        scanner.Scanner._condition = lambda self, rgb: np.clip(rgb, 0.0, None)
    elif variant == "lab_colour":
        from dataclasses import replace
        from filmlab import engine, negative
        i0, m0 = engine.Engine.__init__, negative.Negative.measure

        def init(self, p, *a, **k):
            # 只替换逐帧自动定光 (冲印店的操作); 物理参考等其他定光方式不受影响 (与 T05 规格相同)
            frame = p.timing == "frame"
            i0(self, replace(p, timing="standard") if frame else p, *a, **k)
            sc = getattr(self, "scanner", None)
            if frame and sc is not None and not sc.mono:
                sc._t07_gray = sc.normalized_density(self._transmittance(self._gray_state(actual_light=False)))[0]
                sc._t07_base = sc.timing

        def measure(self, instrument, *a, **k):
            r = m0(self, instrument, *a, **k)
            sc = getattr(instrument, "__self__", None)
            if getattr(sc, "_t07_base", None) is not None and r.shape[-1] == 3:
                D = r.reshape(-1, 3)
                v = D[np.all(np.isfinite(D), -1)].mean(0) - sc._t07_gray
                t0 = sc._t07_base
                off0, r0 = t0.M if t0.M is not None else (np.zeros(3), np.ones(3))
                aR, kR = LAB_COLOUR["R"]; aB, kB = LAB_COLOUR["B"]
                d = np.array([aR - kR * (v[0] - v[1]), 0.0, aB - kB * (v[2] - v[1])])
                sc.timing = replace(t0, M=(np.asarray(off0) + d, r0))
            return r
        engine.Engine.__init__ = init
        negative.Negative.measure = measure
    elif variant == "no_slope":
        from filmlab import scanner
        b0 = scanner.Scanner.balance_frame

        def bal(self, Dn, nominal, mid_gray=0.18, density_limit=0.3, latd_normal=0.0, slope_limit=None, color_limit=None):
            kw = {} if color_limit is None else {"color_limit": color_limit}
            return b0(self, Dn, nominal, mid_gray, density_limit, latd_normal, slope_limit=0.0, **kw)
        scanner.Scanner.balance_frame = bal
    elif variant == "fixed_exposure":
        from filmlab import rawio
        rawio.film_metering = lambda rgb, *a, **k: (rgb * 10.0 ** FIXED_LOG_SCALE, "固定曝光")
    else:
        raise ValueError(variant)


def scene_job(args):
    variant, sid = args
    install(variant)
    import accept_t03
    return accept_t03.scene_job(("next", sid))


def gray_job(variant):
    install(variant)
    import accept_t03
    return accept_t03.gray_job("next")


if __name__ == "__main__":
    variant, out = sys.argv[1], sys.argv[2]
    import multiprocessing as mp
    import accept_t05
    S = {**TRAIN, **HOLDOUT}
    with mp.get_context("spawn").Pool(int(os.environ.get("NPROC", "3")), maxtasksperchild=1) as p:
        sc = dict(p.map(scene_job, [(variant, s) for s in S]))
        gray = p.apply(gray_job, (variant,))
    exp = accept_t05.expected_delta().tolist()
    res = {"variant": variant, "scenes": sc, "gray_chroma": gray, "code_delta": exp, "code_other_change": 0.0}
    json.dump(res, open(out, "w"), indent=1, ensure_ascii=False)
    ok, msg, rep = accept_t05.verdict(res, json.load(open(HERE / "t03_baseline.json")))
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    print("通过" if ok else "不通过: " + "; ".join(msg))
