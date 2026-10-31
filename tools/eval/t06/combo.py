"""诊断: 灯管帧上 "已知荧光灯光谱" × 基础变体 (命令行第一个参数, 默认 no_slope) 的组合, 全图 ΔE00。"""
import json, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
from pathlib import Path; sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[3])]


def job(args):
    sid, member, noslope = args
    from filmlab import lights
    import accept_variant
    if noslope:
        accept_variant.install(BASE)
    if member:
        cls = [c.key for c in lights.classes()]
        k = cls.index("fluorescent")
        mem = dict(lights.classes()[k].members)[member]
        lights.class_weights = lambda ill, feas=None: np.eye(len(cls))[k][None, None, :] * np.ones(ill.shape[:2] + (1,))
        p0 = lights.pick_members
        lights.pick_members = lambda seed: {**p0(seed), "fluorescent": (member, mem)}
    import accept_t03
    return sid, member, noslope, accept_t03.scene_job(("next", sid))[1]["full"]


BASE = sys.argv[1] if len(sys.argv) > 1 else "no_slope"

if __name__ == "__main__":
    import multiprocessing as mp
    S = ["R1-31-32", "R1-33-34", "R1-32-33", "R1-35-36", "R1-21-22", "R1-25-26", "R1-23-24"]
    jobs = [(s, m, ns) for s in S for m in (None, "Philips TL-84", "FL11", "Cool White FL") for ns in (False, True)]
    if BASE != "no_slope":
        jobs = [j for j in jobs if j[2]]
    with mp.get_context("spawn").Pool(3, maxtasksperchild=1) as p:
        for sid, m, ns, de in p.imap(job, jobs):
            print(f"{sid:9s} {str(m):14s} {BASE if ns else 'next':22s}  全图 ΔE00 {de:6.2f}", flush=True)
