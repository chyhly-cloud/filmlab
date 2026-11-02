"""T32 第 2 步: 胶片级。Kodak 5219 Status M (数据手册数字化) 与模型 500T 的对比; 红物体的层密度; DIR 方向。"""
import os
import numpy as np, sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
from filmlab import generation; generation.use("t31_base")
from filmlab.engine import Engine, Params, GRAY
from filmlab.recipes import recipe
from filmlab.stocks import WAVELENGTHS as W
from filmlab import chemistry
RAW = os.environ.get("KODAK5219_RAW", "raw5219")
def curve(n, zero=False):
    a = np.loadtxt(f"{RAW}/{n}.csv", delimiter=","); a = a[np.argsort(a[:, 0])]
    return np.interp(W, a[:, 0], a[:, 1], left=0 if zero else None, right=0 if zero else None)
F = np.stack([curve(f"statusM_{c}", True) for c in "rgb"], 1); F /= F.sum(0)
sm = lambda D: -np.log10(np.clip(10.0 ** -D @ F, 1e-9, None))
hd = [np.loadtxt(f"{RAW}/density_curve_{c}.csv", delimiter=",") for c in "rgb"]
hd = [h[np.argsort(h[:, 0])] for h in hd]
kx = hd[0][:, 0]
KD = np.stack([np.interp(kx, h[:, 0], h[:, 1]) for h in hd], 1)

for proc, label in (("ECN-2", "ECN-2 (Kodak 数据手册条件)"), ("C-41", "C-41 (CineStill 实际, τ=1.18)")):
    eng = Engine(Params(**{**recipe("cinestill_800t"), "process": proc, "roll": None, "grain": 0, "timing": "frame"}))
    e = eng.e
    x = np.linspace(-2.4, 4.0, 33)
    logH = np.repeat(x[:, None], 3, 1)
    st = chemistry.run(e, eng.proc, logH, 0.0, tau_factor=1.0)
    D = sm(np.log10(1 / eng._transmittance(st)))
    print(f"\n== {label}: 中性灰阶 Status M (R G B), 模型 vs Kodak 5219")
    for xi in (-2.0, -1.0, -0.84, 0.0, 0.5, 1.0, 1.5, 2.0, 2.41, 3.0, 3.5, 4.0):
        i = np.argmin(abs(x - xi))
        kd = np.array([np.interp(xi, kx, KD[:, c], right=np.nan) for c in range(3)])
        print(f"  logH {xi:+.2f}: 模型 {D[i].round(3)}  Kodak {kd.round(3)}")
    print("  染料量上限 (成色剂) ", e.coupler_cap.round(3), " S_nom 最大 ", e.S_nom.max(0).round(3))

# 红物体: 红层曝光 +k 档 (相对中灰), 绿 / 蓝层曝光 = 红层 × 比例; 层染料量与 DIR 的作用
eng = Engine(Params(**{**recipe("cinestill_800t"), "roll": None, "grain": 0, "timing": "frame"}))
e = eng.e
g0 = -0.84
print("\n== 红物体 (C-41): 红层 +k 档, 绿层 = 红 × 0.1 (曝光 G/R 0.1), 蓝 = 红 × 0.05; 染料量 (C M Y)")
for k in (0, 2, 3, 4, 6, 8):
    lr = g0 + k * np.log10(2)
    lh = np.array([[lr, lr + np.log10(0.1), lr + np.log10(0.05)]])
    full = chemistry.run(e, eng.proc, lh, 0.0, tau_factor=1.0).dye[0]
    # 没有层间作用: DIR 矩阵只留对角 (同层边缘效应), 内禀曲线按同一矩阵重解
    A0 = chemistry._dir_matrix
    chemistry._dir_matrix = lambda a, neg=True, _f=A0: np.diag(np.diag(_f(a, neg)))
    noint = chemistry.run(e, eng.proc, lh, 0.0, tau_factor=1.0).dye[0]
    chemistry._dir_matrix = A0
    # 中性对照: 绿层单独在同一曝光下的中性灰阶密度 (各层同一 logH = 绿层 logH)
    neut = chemistry.run(e, eng.proc, np.full((1, 3), lh[0, 1]), 0.0, tau_factor=1.0).dye[0]
    print(f"  +{k} 档: 有层间 {full.round(3)}  只同层 {noint.round(3)}  绿层若在中性灰阶 {neut[1]:.3f}")
