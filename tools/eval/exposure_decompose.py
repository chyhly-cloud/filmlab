"""RAW → 三层胶片曝光的误差分解 (Die Lichtfänger 色卡: X100V RAF + Ohta 反射谱 + 已知光源)。
总误差 = 输入 (相机 RGB vs 理想观察者 RGB) + 光源识别 (引擎的光源混合 vs 真实光源) + 光谱重建 (理想 RGB、正确光源 vs 真值)。
单位: 档 (log2)。每个场景先去掉灰块绿层的整体偏移。"""
import sys, numpy as np, colour, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/user/filmlab"); sys.path.insert(0, "/home/user/filmlab/tools/eval")
import chart_500t as C
from filmlab import lights, film, spectral
from filmlab.engine import Engine, Params
from filmlab.recipes import recipe
from filmlab.rawio import load_raw
import os
if os.environ.get("WB65"):
    import wb65_variant; wb65_variant.install()
FLARE = len(sys.argv) > 1 and sys.argv[1] == "flare"
DATA = "/tmp/claude-0/-home-user-filmlab/f67b7686-eba5-58fc-b23b-cf537590be15/scratchpad/charts"
W = spectral.WAVELENGTHS; sh = colour.SpectralShape(W[0], W[-1], W[1] - W[0])
cc = colour.SDS_COLOURCHECKERS["ColorChecker N Ohta"]; R = np.array([cc[k].copy().align(sh).values for k in cc])
GRAY = [19, 20, 21, 22]          # 白 n9.5 太亮, 黑 n2 太暗, 用中间四级
NAMES = C.NAMES + "白 n8 n6.5 n5 n3.5 黑".split()
def eng_H(img, G):
    eng = Engine(Params(**{**recipe("cinestill_800t"), "timing": "frame", "grain": 0, "roll": None}))
    rec = {}
    orig = eng._expose_mixed
    def hook(rgb):
        H, s = orig(rgb); rec["H"] = H; return H, s
    eng._expose_mixed = hook
    eng.render(img)
    mix = {k: (v[0], round(v[1], 2)) for k, v in eng.light_mix.items() if v[1] > 0.02}
    return C.patches(rec["H"], G, 0.5), mix, eng
def report(name, e, offset_ref):
    off = np.mean(offset_ref[GRAY, 1])
    e = e - off
    neu = e[GRAY].mean(0)
    ch = e[:18] - neu
    print(f"  {name:34s} 中性 (各层相对) R {neu[0]-neu[1]:+.2f} B {neu[2]-neu[1]:+.2f} 档 | 彩色块误差 RMS R {np.sqrt((ch[:,0]**2).mean()):.2f} G {np.sqrt((ch[:,1]**2).mean()):.2f} B {np.sqrt((ch[:,2]**2).mean()):.2f} 档"
          f"  最大 {np.abs(ch).max():.2f} ({NAMES[np.unravel_index(np.abs(ch).argmax(), ch.shape)[0]]} {'RGB'[np.unravel_index(np.abs(ch).argmax(), ch.shape)[1]]})")
    return ch
for scene, true_light, cls in [("tungsten", "BB3200", "linear"), ("led5600", "D55", "spectral"), ("led5600", "LED-B4", "linear")]:
    S = C.SCENES[scene]; G = S["grid_raw"]
    E = spectral.illuminant(true_light) if true_light != "LED-B4" else lights._sd("LED-B4")
    sens = 10.0 ** Engine(Params(**recipe("cinestill_800t"))).e.stock.log_sensitivity
    raw = load_raw(f"{DATA}/ref/dlf_x100v/{S['raf']}.RAF", max_size=3000); raw = raw[0] if isinstance(raw, tuple) else raw
    Praw = C.patches(raw, G)
    # 杂散光 (两台相机都会记录): 场景平均辐射 × k 加到每个色块; 拟合 k 与每通道增益 (白平衡残差)
    from scipy.optimize import least_squares
    rad = R * E; flare = rad.mean(0)
    def resid(p):
        I = lights.camera_rgb(rad + p[0] * flare[None, :]) * p[1:4]
        return (np.log2(np.clip(I, 1e-9, None)) - np.log2(np.clip(Praw, 1e-9, None)))[:23].ravel()
    fk = least_squares(resid, [0.05, 1, 1, 1], bounds=([0, 0, 0, 0], [1, 10, 10, 10])).x
    k = fk[0] if FLARE else 0.0
    rad = rad + k * flare[None, :]
    Hi = rad @ sens                                        # 真值层曝光 (含同样的杂散光)
    cam_ideal = lights.camera_rgb(rad)                     # 理想观察者 RGB (日光白平衡)
    print(f"   杂散光 k = {fk[0]:.3f} (场景平均辐射的比例), 白平衡残差增益 {np.round(fk[1:4] / fk[2], 3)}")
    k = Praw[21] @ [.2126, .7152, .0722] / (cam_ideal[21] @ [.2126, .7152, .0722])
    cam_ideal *= k
    img = raw[::2, ::2] * 0.184 / (Praw[21] @ [.2126, .7152, .0722])
    # 合成图: 色块区域替换为理想 RGB (其余像素不变 → 引擎的照明估计几乎相同)
    syn = img.copy(); x0, y0, dx, dxr, dyc, dy, h = G
    for r in range(4):
        for c in range(6):
            x, y = int((x0 + dx * c + dxr * r) * 0.5), int((y0 + dy * r + dyc * c) * 0.5)
            syn[y - 7:y + 8, x - 7:x + 8] = cam_ideal[r * 6 + c] * 0.184 / (Praw[21] @ [.2126, .7152, .0722])
    H_raw, mix, eng = eng_H(img, G)
    H_syn, mix2, _ = eng_H(syn, G)
    rsp = film.exposure_response(eng.e.stock, E, None, cls)
    H_rec = np.clip(rsp(cam_ideal), 1e-4 * Hi.max(), None)   # 线性法负值按极小正值计 (同引擎的截断量级)
    lg = lambda a: np.log2(np.clip(a, 1e-12, None))
    ideal_l = lg(Hi)
    print(f"== {scene} (真实光源按 {true_light} / {cls} 计)   引擎光源估计: {mix}")
    print(f"   相机 vs 理想观察者 RGB (中性, 去绿): R {np.log2(Praw[GRAY,0]/Praw[GRAY,1]).mean() - np.log2(cam_ideal[GRAY,0]/cam_ideal[GRAY,1]).mean():+.2f} B {np.log2(Praw[GRAY,2]/Praw[GRAY,1]).mean() - np.log2(cam_ideal[GRAY,2]/cam_ideal[GRAY,1]).mean():+.2f} 档")
    tot = report("总误差 (RAW, 引擎光源)", lg(H_raw) - ideal_l, lg(H_raw) - ideal_l)
    inp = report("  输入 (RAW − 理想RGB, 同引擎光源)", lg(H_raw) - lg(H_syn), lg(H_raw) - lg(H_syn))
    lid = report("  光源识别 (理想RGB: 引擎 − 正确光源)", lg(H_syn) - lg(H_rec), lg(H_syn) - lg(H_rec))
    rec = report("  光谱重建 (理想RGB、正确光源 − 真值)", lg(H_rec) - ideal_l, lg(H_rec) - ideal_l)
    worst = np.argsort(-np.abs(tot).max(1))[:5]
    print("   最大总误差的色块 (R,G,B 档; 输入 / 光源 / 重建 分量):")
    for i in worst:
        print(f"     {NAMES[i]:4s} 总 {np.round(tot[i],2)}  输入 {np.round(inp[i],2)}  光源 {np.round(lid[i],2)}  重建 {np.round(rec[i],2)}")
