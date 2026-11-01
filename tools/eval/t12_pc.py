"""T12 候选 next_lab_halo_pc = next_lab_halo + 传感器饱和判定 + 逐通道高光重建 (docs/tasks/T12_HIGHLIGHT_PER_CHANNEL.md)。
install() 在当前进程注册代际并设置 rawio.HIGHLIGHT_RECON (CLIP_DETECT 由代际表设置)。

用法: FILMLAB_DATA=… python3 t12_pc.py 输出.json      (全图 / 光源旁 / 灰卡; 红带另用 red_band.py, GENERATION=next_lab_halo_pc)
"""
import json, os, sys, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1])]
from accept_calibration import TRAIN, HOLDOUT
GEN = "next_lab_halo_pc"


def install():
    from filmlab import generation, rawio
    generation.GENERATIONS[GEN] = generation.GENERATIONS["next_lab_halo"]
    generation.METERING[GEN] = generation.METERING["next_lab_halo"]
    generation.CLIP_DETECT[GEN] = "sensor"
    generation.HIGHLIGHT_RECON[GEN] = "per_channel"      # generation.use 按代际表设置 rawio.HIGHLIGHT_RECON
    generation.use(GEN)


def scene_job(sid):
    install()
    import accept_t03
    return sid, accept_t03.scene_job((GEN, sid))[1]


def gray_job(_):
    install()
    import accept_t03
    return accept_t03.gray_job(GEN)


if __name__ == "__main__":
    import multiprocessing as mp
    G = {**TRAIN, **HOLDOUT}
    with mp.get_context("spawn").Pool(int(os.environ.get("NPROC", "3")), maxtasksperchild=1) as p:
        sc = dict(p.map(scene_job, list(G)))
        gray = p.apply(gray_job, (0,))
    base = json.load(open(HERE / "t09_result.json"))["scenes"]          # next_lab (同一入口, 确定性)
    E = [s for s in G if s != "cafe"]; H = [s for s in HOLDOUT if s != "cafe"]
    d = {s: sc[s]["full"] - base[s]["next_lab"]["full"] for s in G}
    dn = {s: sc[s]["near"] - base[s]["next_lab"]["near"] for s in G}
    for s in sorted(G, key=lambda s: G[s]):
        print(f"{s:9s} {G[s]:5s} 全图 {base[s]['next_lab']['full']:6.2f} → {sc[s]['full']:6.2f} ({d[s]:+.2f})  光源旁 {dn[s]:+.2f}")
    mf = float(np.mean([d[s] for s in H])); mn = float(np.nanmean([dn[s] for s in H])); worst = max(E, key=d.get)
    print(f"\n留出全图 {mf:+.3f}, 留出光源旁 {mn:+.3f}; 训练全图 {np.mean([d[s] for s in TRAIN]):+.3f}, "
          f"训练光源旁 {np.nanmean([dn[s] for s in TRAIN]):+.3f}; 最大变差 {worst} {d[worst]:+.2f}; 灰卡 {gray:.3f}")
    json.dump({"scenes": sc, "gray_chroma": gray, "holdout_full": mf, "holdout_near": mn,
               "worst": [worst, d[worst]]}, open(sys.argv[1], "w"), indent=1)
