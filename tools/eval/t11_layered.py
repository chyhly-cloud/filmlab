"""T11 候选 next_lab_layered = next_lab + halo_oblique, 斜程衰减改为按层深度 (docs/tasks/T11_HALO_LAYERED.md):
    A(θ) = w · 10^(−D·u) + (1 − w) · 均匀深度平均,  u = 1/cosθ − 1
w = 红感层之上散射的比例, 由 red_band.scatter_above_red 按 film.emulsion_stack 的同一设定算出 (不拟合)。
install() 在当前进程注册代际并替换 film.oblique_attenuation (画面与定光中灰共用)。

用法: FILMLAB_DATA=… python3 t11_layered.py 输出.json      (全图 / 光源旁 / 灰卡; 红带另用 red_band.py, GENERATION=next_lab_layered)
"""
import json, os, sys, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1])]
from accept_calibration import TRAIN, HOLDOUT
GEN = "next_lab_layered"


def install():
    from filmlab import film, generation, variation
    from filmlab.materials import emulsion
    import red_band
    w0, corr, rev = generation.GENERATIONS["next_lab"]
    generation.GENERATIONS[GEN] = (w0, corr + ("halo_oblique",), rev)
    generation.METERING[GEN] = generation.METERING["next_lab"]
    generation.use(GEN)
    w = red_band.scatter_above_red(emulsion("kodak_vision3_500t").stock, variation.sample(None, 0, None))

    def layered(theta, density):
        if not np.isfinite(density) or density < 0:
            raise ValueError("Oblique density must be finite and nonnegative")
        x = density * (1.0 / np.cos(theta) - 1.0) * np.log(10.0)
        uniform = np.divide(-np.expm1(-x), x, out=np.ones_like(x), where=x != 0)
        return w * np.exp(-x) + (1.0 - w) * uniform
    film.oblique_attenuation = layered
    return w


def scene_job(sid):
    install()
    import accept_t03
    return sid, accept_t03.scene_job((GEN, sid))[1]


def gray_job(_):
    w = install()
    import accept_t03
    return accept_t03.gray_job(GEN), w


if __name__ == "__main__":
    import multiprocessing as mp
    G = {**TRAIN, **HOLDOUT}
    with mp.get_context("spawn").Pool(int(os.environ.get("NPROC", "3")), maxtasksperchild=1) as p:
        sc = dict(p.map(scene_job, list(G)))
        gray, w = p.apply(gray_job, (0,))
    base = json.load(open(HERE / "t09_result.json"))["scenes"]          # next_lab (同一入口, 确定性)
    E = [s for s in G if s != "cafe"]; H = [s for s in HOLDOUT if s != "cafe"]
    d = {s: sc[s]["full"] - base[s]["next_lab"]["full"] for s in G}
    dn = {s: sc[s]["near"] - base[s]["next_lab"]["near"] for s in G}
    for s in sorted(G, key=lambda s: G[s]):
        print(f"{s:9s} {G[s]:5s} 全图 {base[s]['next_lab']['full']:6.2f} → {sc[s]['full']:6.2f} ({d[s]:+.2f})  光源旁 {dn[s]:+.2f}")
    mf = float(np.mean([d[s] for s in H])); mn = float(np.nanmean([dn[s] for s in H])); worst = max(E, key=d.get)
    print(f"\nw = {w:.4f}; 留出全图 {mf:+.3f}, 留出光源旁 {mn:+.3f}; 训练全图 {np.mean([d[s] for s in TRAIN]):+.3f}, "
          f"训练光源旁 {np.nanmean([dn[s] for s in TRAIN]):+.3f}; 最大变差 {worst} {d[worst]:+.2f}; 灰卡 {gray:.3f}")
    json.dump({"w": w, "scenes": sc, "gray_chroma": gray, "holdout_full": mf, "holdout_near": mn,
               "worst": [worst, d[worst]]}, open(sys.argv[1], "w"), indent=1)
