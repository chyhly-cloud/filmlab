"""校准前的参数清单: 显影链中"靠估计填的物理量"各自的影响与可辨识性 (不拟合, 只算敏感度)。

每个参数取其物理范围的两端, 一次只改一个; 每个场景用定光预言机 O1 (只给冲印店三个印放光号) 消去整体偏色与亮度,
剩下的才是该参数造成、冲印店无法补偿的变化。按格子 (24 px) 记录 Lab 残差 (模拟 − 真实), 用于:
  · 各组 (夜景 / 黄昏 / 钨丝 / 白光 / 色卡 / 日光) 与区域 (光源旁 ≤ 500 µm、真实为绿色的格子) 的误差变化;
  · 参数效应之间的相似度 (余弦): 相似度高 = 数据里分不开, 不能同时拟合。
扫描参数 (LED、印放矩阵) 单列: 属于冲印店档案, 不进胶片参数。

用法: FILMLAB_DATA=数据目录 python3 param_sensitivity.py 输出目录 [变体…]      (先跑 base)
      python3 param_sensitivity.py 输出目录 --report
"""
import json, os, sys, zlib, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "tools/eval"), str(REPO)]

SCENES = {"R1-00-1": "夜景", "R1-02-3": "夜景", "R1-04-5": "夜景", "R1-06-7": "夜景", "R1-07-8": "黄昏",
          "R1-27-28": "钨丝灯箱", "R1-33-34": "白光灯管", "R1-20-21": "色卡", "cafe": "日光 (他店)"}
TAU0 = 1.0 / 0.85          # 现行 C-41 显影程度 (相对 ECN-2)
# 名称: (类别, 值, 物理范围与出处)
VARIANTS = {
    "base": (None, None, "next 代际, 第 1 卷抽样值"),
    "tau_lo": ("tau", 1.0, "C-41 中的显影程度 τ: 冻结区间 [1.0, 1.3] (CRITERIA τ 冻结)"),
    "tau_hi": ("tau", 1.3, ""),
    "dir_lo": ("dir", 0.6, "DIR 层间抑制强度倍数: 标定于 Kodak US 6,210,870 分色 gamma 比, 试探 ±40%"),
    "dir_hi": ("dir", 1.4, ""),
    "cyan_lo": ("v", ("dye_shift_nm", 7.0), "青染料在 CD-4 中的峰移: US 5,183,729 实测 7–15 nm"),
    "cyan_hi": ("v", ("dye_shift_nm", 15.0), ""),
    "mgy_lo": ("v", ("dye_shift_other_nm", 2.0), "品红 / 黄的峰移: 无公开数据, 估计 2–10 nm"),
    "mgy_hi": ("v", ("dye_shift_other_nm", 10.0), ""),
    "diffuse_lo": ("v", ("diffuse_fraction", 0.45), "光晕: 到达片基的光中漫射的比例, 估计 0.45–0.75 (与背面反射率等价, 合并)"),
    "diffuse_hi": ("v", ("diffuse_fraction", 0.75), ""),
    "absorb_lo": ("v", ("layer_absorbance", (0.08, 0.08, 0.08)), "光晕: 感光层对自身光的吸收密度, 估计 0.08–0.25"),
    "absorb_hi": ("v", ("layer_absorbance", (0.25, 0.25, 0.25)), ""),
    "forward_lo": ("v", ("forward", 1.0), "光晕: 漫射光前向程度 cos^m, 估计 1.0–2.5"),
    "forward_hi": ("v", ("forward", 2.5), ""),
    "turb_lo": ("turb", 0.05, "光晕: 未冲洗乳剂每层的散射消光密度 (550 nm), 现值 0.15, 无实测, 试探 0.05–0.45"),
    "turb_hi": ("turb", 0.45, ""),
    "scan_led": ("led", "noritsu", "扫描 (冲印店档案): Frontier LED 峰值改用 Noritsu 型估计"),
    "scan_pid": ("pid", None, "扫描 (冲印店档案): 印放密度矩阵 P = 单位矩阵"),
}


def apply(kind, val):
    """在本进程内打补丁 (每个任务一个新进程), 然后切代际以清空依赖缓存。"""
    from dataclasses import replace
    from filmlab import chemistry, film, variation, generation
    if kind == "tau":
        run0 = chemistry.run

        def run(e, process, logH, blur_px=0.0, tau_factor=1.0):
            if e.native_process == "ECN-2" and any(isinstance(s, chemistry.Develop) and s.developer == "C-41 CD"
                                                   for s in process.steps):
                tau_factor = tau_factor * val / TAU0
            return run0(e, process, logH, blur_px, tau_factor)
        chemistry.run = run
    elif kind == "dir":
        di0 = chemistry.develop_image
        chemistry.develop_image = lambda e, logH, tau, inhib, blur_px, k="color": di0(e, logH, tau, inhib * val, blur_px, k)
    elif kind == "v":
        s0, (field, x) = variation.sample, val
        variation.sample = lambda *a, **kw: replace(s0(*a, **kw), **{field: x})
    elif kind == "turb":
        es0 = film.emulsion_stack
        film.emulsion_stack = lambda stock, la, yf, turbidity=0.15, turbidity_power=3.0: es0(stock, la, yf, val, turbidity_power)
    elif kind == "led":
        from filmlab import scanner as scm
        f, n = scm.SCANNERS["frontier"], scm.SCANNERS["noritsu"]
        scm.SCANNERS["frontier"] = replace(f, led_peaks=n.led_peaks, led_widths=n.led_widths)
    elif kind == "pid":
        from filmlab.scanning import conversion
        conversion.printing_matrix = lambda *a, **k: np.eye(3)
    generation.use("next")


def job(args):
    out, name, sid = args
    kind, val, _ = VARIANTS[name]
    apply(kind, val)
    Path(out).mkdir(parents=True, exist_ok=True)
    np.savez(f"{out}/{name}__{sid}.npz", **evaluate_scene(sid, roll=1))
    return name, sid


def evaluate_scene(sid, roll=1):
    """在已打好补丁的进程里评估一个场景: 定光预言机 O1 后的逐格 Lab (S)、真实 (R)、有效格、到光源距离 (µm)、拟合光号。"""
    import colour, rawpy
    from dataclasses import replace
    from scipy.optimize import least_squares
    from scipy.ndimage import distance_transform_edt, zoom
    from skimage.transform import warp, ProjectiveTransform
    import oracle_timing as ot, pair_compare as pc
    from filmlab import rawio
    from filmlab.engine import Engine, Params
    from filmlab.recipes import recipe
    from filmlab.rawio import film_metering
    Hm, real = ot.geometry(sid)
    rawp, _, bias = ot.sources(sid)
    rgb = pc.load_digital(rawp, 1400)[0]
    seed = zlib.crc32(sid.encode()) % 1000
    m, _ = film_metering(rgb, rng=np.random.default_rng(seed), bias_ev=bias)
    eng = Engine(Params(**{**recipe("cinestill_800t"), "timing": "frame", "roll": roll, "frame": seed % 36, "grain": 0}))
    neg = eng.develop_negative(m)
    tr = {}
    eng.scan(neg, trace=tr)
    sc, Dn, t0 = eng.scanner, tr["Dn"], tr["timing"]
    # 光源 (传感器饱和) 距离, 用于"光源旁"区域
    with rawpy.imread(rawp) as r:
        sat = rawio.sensor_saturation(r)
    s = np.zeros(Dn.shape[:2]) if sat is None else zoom(sat.astype(float), (Dn.shape[0] / sat.shape[0], Dn.shape[1] / sat.shape[1]), order=1)
    tf = ProjectiveTransform(Hm)
    sw = warp(np.concatenate([Dn, s[..., None]], -1), tf.inverse, output_shape=real.shape[:2], order=1, cval=np.nan)
    ok = np.all(np.isfinite(sw), -1)
    um = 36000.0 / max(real.shape[:2])
    src = np.nan_to_num(sw[..., 3]) > 0.25
    dist = distance_transform_edt(~src) * um if src.any() else np.full(src.shape, 1e9)
    C, idx = ot.cells(np.concatenate([sw[..., :3], dist[..., None]], -1), ok)
    R = ot.lab_lin(colour.models.eotf_sRGB(ot.cells(real, ok)[0]))
    Dc, dc = C[:, :3], C[:, 3]
    keep = R[:, 0] > 5
    off0, r0 = t0.M if t0.M is not None else (np.zeros(3), np.ones(3))

    def render(x):
        sc.timing = replace(t0, K=t0.K + x[0], M=(off0 + np.array([x[1], 0.0, x[2]]), r0))
        return ot.lab_lin(sc.to_rgb(Dc))

    fit = least_squares(lambda x: np.nan_to_num((render(x) - R)[keep].ravel(), nan=50.0), [0, 0, 0],
                        bounds=([-1.5, -0.8, -0.8], [1.5, 0.8, 0.8]), loss="soft_l1", f_scale=5.0)
    return dict(S=render(fit.x), R=R, keep=keep, dist=dc, x=fit.x)


def gray_card_check(kind, val):
    """3200K 灰卡在物理参考中是否仍印成中性 (内部一致性; 每个变体都必须保持)。"""
    apply(kind, val)
    from filmlab import lights, spectral
    from filmlab.engine import Engine, Params
    from filmlab.materials import emulsion
    from filmlab.recipes import recipe
    ref = emulsion("kodak_vision3_500t").stock.reference_illuminant
    rgb = np.tile(0.184 * lights.light_rgb(spectral.illuminant(ref)), (32, 32, 1))
    eng = Engine(Params(**{**recipe("cinestill_800t_reference"), "roll": 1, "grain": 0, "scene_lights": None,
                           "scene_illuminant": ref, "scan_color_strength": 1.0}))
    import colour
    lab = colour.XYZ_to_Lab(colour.sRGB_to_XYZ(np.clip(eng.render(rgb)[16, 16], 0, 1), apply_cctf_decoding=False))
    return float(np.hypot(lab[1], lab[2]))


def _gc(name):
    return name, gray_card_check(*VARIANTS[name][:2])


def report(out):
    import colour
    load = lambda n, s: np.load(f"{out}/{n}__{s}.npz")
    names = [n for n in VARIANTS if all(os.path.exists(f"{out}/{n}__{s}.npz") for s in SCENES)]
    base = {s: load("base", s) for s in SCENES}
    groups = sorted(set(SCENES.values()), key=list(SCENES.values()).index)

    def metrics(n):
        res, row = [], {}
        for g in groups:
            row[g] = np.mean([np.median(colour.delta_E(load(n, s)["S"][base[s]["keep"]], base[s]["R"][base[s]["keep"]],
                                                       method="CIE 2000")) for s in SCENES if SCENES[s] == g])
        near, green = [], []
        for s in SCENES:
            d, b = load(n, s), base[s]
            k = b["keep"]
            res.append((d["S"] - b["S"])[k])
            R = b["R"][k]
            nr = d["dist"][k] <= 500
            if nr.sum():
                near.append((d["S"][k] - R)[nr])
            gr = (R[:, 1] < -8) & (R[:, 2] > 5) & (R[:, 0] > 20)
            if gr.sum():
                green.append((d["S"][k] - R)[gr])
        near, green = np.vstack(near), np.vstack(green)
        row["光源旁 ΔE"] = float(np.median(np.linalg.norm(near, axis=1)))
        row["光源旁 Δa/Δb"] = "%+.1f/%+.1f" % tuple(np.median(near[:, 1:], 0))
        row["绿色 Δa/Δb"] = "%+.1f/%+.1f" % tuple(np.median(green[:, 1:], 0))
        return row, np.concatenate(res).ravel()
    rows, vecs = {}, {}
    for n in names:
        rows[n], vecs[n] = metrics(n)
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(3, maxtasksperchild=1) as p:
        gc = dict(p.map(_gc, names))
    cols = groups + ["光源旁 ΔE", "光源旁 Δa/Δb", "绿色 Δa/Δb"]
    lines = ["| 变体 | " + " | ".join(cols) + " | 灰卡彩度 |", "|---" * (len(cols) + 2) + "|"]
    for n in names:
        r = rows[n]
        lines.append(f"| {n} | " + " | ".join(f"{r[c]:.2f}" if isinstance(r[c], float) else r[c] for c in cols)
                     + f" | {gc[n]:.2f} |")
    # 参数效应: 每个参数 = 上端 − 下端 (扫描参数 = 变体 − base); 相似度矩阵
    eff = {}
    for n in names:
        if n.endswith("_hi") and n[:-3] + "_lo" in vecs:
            eff[n[:-3]] = vecs[n] - vecs[n[:-3] + "_lo"]
        elif n.startswith("scan_"):
            eff[n] = vecs[n]
    keys = list(eff)
    size = {k: float(np.sqrt(np.mean(eff[k] ** 2))) for k in keys}
    lines += ["", "效应大小 (两端之差, 每格 Lab RMS): " + ", ".join(f"{k} {size[k]:.2f}" for k in keys), "",
              "效应相似度 |cos| (≥ 0.8 = 数据里分不开):", "", "| | " + " | ".join(keys) + " |", "|---" * (len(keys) + 1) + "|"]
    for a in keys:
        lines.append(f"| {a} | " + " | ".join(
            f"{abs(eff[a] @ eff[b]) / (np.linalg.norm(eff[a]) * np.linalg.norm(eff[b]) + 1e-12):.2f}" for b in keys) + " |")
    txt = "\n".join(lines)
    Path(f"{out}/report.md").write_text(txt)
    print(txt)


if __name__ == "__main__":
    out = sys.argv[1]
    if "--report" in sys.argv:
        report(out); sys.exit()
    names = sys.argv[2:] or list(VARIANTS)
    tasks = [(out, n, s) for n in names for s in SCENES if not os.path.exists(f"{out}/{n}__{s}.npz")]
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(int(os.environ.get("NPROC", "4")), maxtasksperchild=1) as p:
        for n, s in p.imap_unordered(job, tasks):
            print("done", n, s, flush=True)
