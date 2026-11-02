"""T33 A (第 4 节) S1 反事实 (只诊断, 不进入渲染): 红物体 (胶片 logH_R − logH_G > 0.2…0.4 渐入) 的蓝层曝光不高于
"相机 sRGB B 相对 3200K 中灰的 B" (t3200, 胶片自己的参照), 对照 T32 s1b 的写法 (day: 相对日光中灰 0.184)。
在含 epfl/ (t31_base 轨迹) 的工作目录运行: FILMLAB_DATA=… python3 t33_blue_cf.py"""
import os, sys, numpy as np, warnings, json
warnings.filterwarnings("ignore")
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[2])]
NIGHT = ["R1-00-1","R1-01-2","R1-02-3","R1-03-4","R1-04-5","R1-05-6","R1-06-7","R1-07-8"]
G0 = np.array([-0.83960607, -0.84922118, -0.8927258])
def hook_for(mode):
    from filmlab import lights
    b3200 = 0.184 * lights.light_rgb(lights._blackbody(3200))[2]
    ref = {"day": 0.184, "t3200": b3200}[mode]
    def hook(lh, z):
        v = lh - G0
        w = np.clip((v[..., 0] - v[..., 1] - 0.2) / 0.2, 0, 1)
        cam = np.log10(np.clip(z["inp"][..., 2].astype(float), 1e-6, None) / ref) + G0[2]
        lh = lh.copy(); lh[..., 2] = np.where(w > 0, (1 - w) * lh[..., 2] + w * np.minimum(lh[..., 2], cam), lh[..., 2])
        return lh
    return hook
def job(a):
    mode, sid = a
    import t32_counterfactuals as cf, t32_rescan as rescan, t33_dir_sweep as sw
    def setup():
        from filmlab import generation; generation.use("t31_base")
        cf.LOGH_HOOK = hook_for(mode)
    out, tr, z = rescan.render(setup, sid, "_t31_base")
    full, near = rescan.de_scores(out, z, sid)
    return mode, sid, full, near, sw.red_stats(out, tr, z, sid)
if __name__ == "__main__":
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(4, maxtasksperchild=1) as p:
        res = p.map(job, [(m, s) for m in ("t3200", "day") for s in NIGHT], chunksize=1)
    for m in ("t3200", "day"):
        rr = [(s, f, n, r) for mm, s, f, n, r in res if mm == m]
        print(f"== {m}: 全图 {np.mean([f for s, f, n, r in rr]):.3f} 光源旁 {np.nanmean([n for s, f, n, r in rr]):.3f}")
        T = {k: np.concatenate([r[k] for s, f, n, r in rr if r is not None]) for k in ("inp", "real", "out", "o1")}
        ev = np.log2(T["inp"][:, 0] / 0.184)
        for lo in range(-2, 5):
            mk = (ev >= lo) & (ev < lo + 1)
            if mk.sum() < 30: continue
            q = lambda A, i, j: np.median(A[mk][:, i] / np.maximum(A[mk][:, j], 1e-9))
            print(f"  {lo:+d} n={mk.sum()} G/R 真实 {q(T['real'],1,0):.2f} | {q(T['out'],1,0):.2f} / {q(T['o1'],1,0):.2f} ; B/G 真实 {q(T['real'],2,1):.2f} | {q(T['out'],2,1):.2f} / {q(T['o1'],2,1):.2f}")
