"""T25: 光晕部分 = 除碳 800T − 带 remjet 对照 (同一测量: 暗背景灯, 灯芯 60–150 / 150–400 µm, 超出背景的线性 R, 按距灯芯边缘距离)。
真实: 800T 参考集 − Vision3/Vision 500T 带 remjet 对照集 (不同场景, 灯的总体对总体)。
模型: 同一组 RAW 的除碳与带 remjet 两次渲染 (同一盏灯)。
用法: python3 t25_halo_part.py 800T参考目录 对照目录 模型缓存目录 后缀[,后缀…]   (后缀如 t22_bxyz_0 → 对照 t22_bxyz_0_remjet)"""
import subprocess, sys, re
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
BINS = [25, 50, 100, 150, 200, 300, 400, 500, 700, 1000, 1500]


def prof(d, glob=None):
    out = subprocess.run([sys.executable, str(HERE / "t24_dist_colour.py"), d] + ([glob] if glob else []),
                         capture_output=True, text=True).stdout
    res, cls = {}, None
    for line in out.splitlines():
        m = re.match(r"\[(.+?)\] 灯 (\d+)", line)
        if m:
            cls = m.group(1); res[cls] = {"n": int(m.group(2))}
        elif cls and "超出背景" in line:
            k = line.split()[0]
            res[cls][k] = np.array([float(x) for x in line.split(")")[1].split()])
    return res


ref, ctl, cache, sufs = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split(",")
R8, RC = prof(ref), prof(ctl)
print("距灯芯边缘 µm        " + " ".join(f"{b:6d}" for b in BINS))
for cls in ("60-150", "150-400"):
    print(f"\n[灯芯 {cls} µm]  真实 800T {R8[cls]['n']} 盏, 对照 {RC[cls]['n']} 盏")
    real = R8[cls]["R"] - RC[cls]["R"]
    print("  真实 光晕部分 R      " + " ".join(f"{x:6.3f}" for x in real))
    for s in sufs:
        M8, MC = prof(cache, f"*_{s}.png"), prof(cache, f"*_{s}_remjet.png")
        mod = M8[cls]["R"] - MC[cls]["R"]
        print(f"  模型 {s:16s}" + " ".join(f"{x:6.3f}" for x in mod) + f"   (灯 {M8[cls]['n']}/{MC[cls]['n']})")
        print(f"  模型 带碳 R         " + " ".join(f"{x:6.3f}" for x in MC[cls]["R"]) + "   (对照: 真实带碳 " +
              " ".join(f"{x:.3f}" for x in RC[cls]["R"][2:6]) + " @100–300)")
