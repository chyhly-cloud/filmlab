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
  多个变体用逗号组合, 如 fixed_exposure,no_satmat

用法: FILMLAB_DATA=… python3 accept_variant.py 变体 输出.json
"""
import json, os, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1])]
from accept_calibration import TRAIN, HOLDOUT

VARIANTS = ("no_slope", "fixed_exposure", "no_satmat", "plain")
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
