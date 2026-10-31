"""诊断: 已知光源类别 (one-hot) 与指定灯款时, 全图 ΔE00 中位 (与验收入口相同, 只替换光源识别)。"""
import sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
from pathlib import Path; sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[3])]


def job(args):
    sid, key, member = args
    from filmlab import lights
    if key is not None:
        cls = [c.key for c in lights.classes()]
        k = cls.index(key)
        c = lights.classes()[k]
        mem = dict(c.members)[member]

        def weights(ill, feas=None):
            W = np.zeros(ill.shape[:2] + (len(cls),)); W[..., k] = 1.0
            return W
        p0 = lights.pick_members
        lights.class_weights = weights
        lights.pick_members = lambda seed: {**p0(seed), key: (member, mem)}
    import accept_t03
    return sid, key, member, accept_t03.scene_job(("next", sid))[1]["full"]


if __name__ == "__main__":
    import multiprocessing as mp
    W = ["R1-31-32", "R1-33-34", "R1-32-33", "R1-35-36"]
    T = ["R1-26-27", "R1-28-29", "R1-27-28", "R1-30-31"]
    jobs = [(s, None, None) for s in W + T]
    jobs += [(s, "fluorescent", m) for s in W for m in ("Philips TL-84", "F32T8/TL841 (Triphosphor)", "Cool White FL", "FL11")]
    jobs += [(s, "tungsten", m) for s in T for m in ("2700K 白炽 / 卤素", "3000K 白炽 / 卤素")]
    with mp.get_context("spawn").Pool(3, maxtasksperchild=1) as p:
        for sid, key, mem, de in p.imap(job, jobs):
            print(f"{sid:9s} {str(key):12s} {str(mem):28s} 全图 ΔE00 {de:6.2f}", flush=True)
