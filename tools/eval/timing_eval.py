"""自动定光策略的评估: 不读取目标扫描, 只由底片 (Dn) 和冲印规则求定光, 再与真实扫描比较。

两步:
  cache   每个配对渲染一次底片 → 缓存 Dn (半分辨率)、各种中性参考的 Dn、评估格子 (对齐到真实扫描)
  eval    对每个定光策略只重算 定光 + 数字相纸 (很快)

均匀画面 (空灯箱、白场) 没有特征点无法对齐, 改为比较画面中心区域的中位颜色。
同一冲印店: EPFL R1 (Frontier, 工单 A002382); 跨冲印店: SillyStill 咖啡店 (另一家)。

用法: timing_eval.py cache [条件名]   /   timing_eval.py eval [条件名]
"""
import csv, json, os, sys, zlib, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/user/filmlab/tools/eval"); sys.path.insert(0, "/home/user/filmlab")
SP = "/tmp/claude-0/-home-user-filmlab/f67b7686-eba5-58fc-b23b-cf537590be15/scratchpad"
EP = SP + "/epfl/data"
OUT = SP + "/timing"
MAN = {r["scene_id"]: r for r in csv.DictReader(open("/home/user/filmlab/tools/eval/epfl_800t_manifest.csv"))}
UNIFORM = {"R1-19-20", "R1-29-30", "R1-34-35", "R1-24-25"}           # 均匀画面: 比较中心区域
# 灰阶楔: 特征点太少, 按各级位置比较 (600 像素缩略图上的灰阶范围 x0, x1, y0, y1; 数码 / 胶片扫描各自的构图)
WEDGE = {"R1-17-18": ((114, 408, 135, 330, 576), (72, 413, 88, 328, 600))}


def wedge_steps(img, box, n=13):
    x0, x1, y0, y1, ref_w = box
    k = img.shape[1] / ref_w
    w = (x1 - x0) / n * k
    out = []
    for i in range(n):
        a, b = int(x0 * k + (i + 0.3) * w), int(x0 * k + (i + 0.7) * w)
        ya, yb = int((y0 + 0.3 * (y1 - y0)) * k), int((y0 + 0.7 * (y1 - y0)) * k)
        out.append(np.median(img[ya:yb, a:b].reshape(-1, img.shape[-1]), 0))
    return np.array(out)
SKIP = {"R1-18-19"}                                                  # 噪声卡 (无颜色信息)
import oracle_timing as _ot
SS = ["ss:" + n for n in _ot.SILLY_JPEG]
SCENES = [s for s in MAN if s not in SKIP] + ["cafe"] + SS
LAB = {s: "epfl_frontier" for s in MAN}
LAB["cafe"] = "sillystill"
LAB.update({s: "sillystill_jpeg" for s in SS})


def group(sid):
    if sid == "cafe":
        return "day"
    if sid.startswith("ss:"):
        return "jpeg"
    if "光晕测试" in MAN[sid]["description"]:
        return "halo_test"          # 数码有三组等效曝光, 与胶片曝光不对应; 不用于颜色评估
    c = MAN[sid]["scene_category"]
    return {"neon_led": "night", "blue_sky": "night", "other": "night" if sid == "R1-03-4" else "uniform",
            "colorchecker": "chart"}.get(c, c)


def sources(sid):
    return _ot.sources(sid)


def lab_lin(rgb):
    import colour
    return colour.XYZ_to_Lab(colour.sRGB_to_XYZ(np.clip(rgb, 0, 1), apply_cctf_decoding=False))


def engine_for(sid, cond_none=True):
    from filmlab.engine import Engine, Params
    from filmlab.recipes import recipe
    seed = zlib.crc32(sid.encode()) % 1000
    extra = {"output_conditioning": None} if cond_none else {}
    if os.environ.get("LIGHT_MEMBER"):
        extra["light_member"] = os.environ["LIGHT_MEMBER"]
    return Engine(Params(**{**recipe("cinestill_800t"), "timing": "frame", "roll": 1, "frame": seed % 36, **extra})), seed


def cache_one(args):
    sid, cond = args
    import ablation_flags as af; af.install()
    import colour
    from PIL import Image
    from skimage.transform import warp, ProjectiveTransform
    import pair_compare as pc
    import oracle_timing as ot
    from filmlab.rawio import film_metering
    f = f"{OUT}/{cond}/{sid.replace(':', '_')}.npz"
    if os.path.exists(f):
        return sid, "cached"
    rawp, scan, bias = sources(sid)
    rgb = pc.load_digital(rawp, 1400)[0]
    eng, seed = engine_for(sid)
    m, _ = film_metering(rgb, rng=np.random.default_rng(seed), bias_ev=bias)
    neg = eng.develop_negative(m)
    tr = {}
    eng.scan(neg, trace=tr)
    sc, Dn = eng.scanner, tr["Dn"]
    nd = lambda st: sc.normalized_density(eng._transmittance(st))[0]
    gray_ref = nd(eng._gray_state(actual_light=False))          # 胶片设计光源 (3200K) 下的中灰
    gray_day = nd(eng._gray_state(actual_light=True))           # 日光下的中灰
    Image.MAX_IMAGE_PIXELS = None
    if sid in WEDGE:
        real = Image.open(scan).convert("RGB"); real.thumbnail((1600, 1600)); real = np.asarray(real, float) / 255
        R = lab_lin(colour.models.eotf_sRGB(wedge_steps(real, WEDGE[sid][1])))
        C = wedge_steps(Dn, WEDGE[sid][0])
        mode = "cells"
    elif sid in UNIFORM:
        real = Image.open(scan).convert("RGB"); real.thumbnail((800, 800)); real = np.asarray(real, float) / 255
        h, w = real.shape[:2]
        R = lab_lin(colour.models.eotf_sRGB(np.median(real[h // 3:2 * h // 3, w // 3:2 * w // 3].reshape(-1, 3), 0)))[None]
        h, w = Dn.shape[:2]
        C = np.median(Dn[h // 3:2 * h // 3, w // 3:2 * w // 3].reshape(-1, 3), 0)[None]
        mode = "uniform"
    else:
        try:
            Hm, real = ot.geometry(sid)
        except Exception as e:
            return sid, f"对齐失败 {e}"
        sw = warp(Dn, ProjectiveTransform(Hm).inverse, output_shape=real.shape[:2], order=1, cval=np.nan)
        ok = np.all(np.isfinite(sw), -1)
        C, _ = ot.cells(sw, ok)
        R = lab_lin(colour.models.eotf_sRGB(ot.cells(real, ok)[0]))
        mode = "cells"
    os.makedirs(os.path.dirname(f), exist_ok=True)
    np.savez(f, Dn=Dn[::2, ::2].astype(np.float32), C=C, R=R, gray_ref=gray_ref, gray_day=gray_day,
             gray0=sc.gray0, strength=sc.strength, stretch_max=sc.stretch_max, latd_limit=eng.v.latd_limit, mode=mode)
    return sid, mode


# ------------------------------------------------------------------ 策略 --
def tm_limit():
    from filmlab.scanning import timing
    return timing.COLOR_LIMIT


def policies():
    """名称 → f(d, sc, tm) → PrintTiming。d: 缓存; 只用底片信息, 不用真实扫描。"""
    from filmlab.engine import LATD_NORMAL
    P = {}

    def frame(d, sc, tm, gray0, strength, density_limit=None, gb=None, thr=None):
        lim = float(d["latd_limit"]) if density_limit is None else density_limit
        return tm.frame_auto(sc.digital_paper, d["Dn"].reshape(-1, 3), d["gray_ref"], gray0, strength,
                             float(d["stretch_max"]), 0.18, lim, LATD_NORMAL, gray_balance=gb, chroma_thr=thr)
    g32 = lambda d: d["gray_ref"] - d["gray_ref"][1]
    P["auto (现行)"] = lambda d, sc, tm: frame(d, sc, tm, d["gray0"], float(d["strength"]))
    P["固定: 档案中灰 (Portra 800)"] = lambda d, sc, tm: frame(d, sc, tm, d["gray0"], 0.0)
    P["固定: 胶片 3200K 中灰"] = lambda d, sc, tm: frame(d, sc, tm, d["gray_ref"] - d["gray_ref"][1], 0.0)
    P["固定: 胶片日光中灰"] = lambda d, sc, tm: frame(d, sc, tm, d["gray_day"] - d["gray_day"][1], 0.0)
    P["固定: 只去片基"] = lambda d, sc, tm: frame(d, sc, tm, np.zeros(3), 0.0)
    # 亮度不逐帧校正 (按标准中灰的印放曝光), 与颜色分开检验
    P["亮度固定 + 颜色 auto"] = lambda d, sc, tm: frame(d, sc, tm, d["gray0"], float(d["strength"]), 0.0)
    P["全固定 (3200K)"] = lambda d, sc, tm: frame(d, sc, tm, g32(d), 0.0, 0.0)
    def highlight_white(d, sc, tm, pct=(95.0, 99.0), limit=tm_limit()):
        """以画面最亮的区域 (绿通道密度最高的一段) 为白: 每通道偏移使其中性, 偏移不超过 limit。亮度 / 肩部同现行。"""
        t = frame(d, sc, tm, d["gray0"], 0.0)
        D = d["Dn"].reshape(-1, 3); D = D[np.all(np.isfinite(D), -1)]
        lo, hi = np.percentile(D[:, 1], pct)
        h = D[(D[:, 1] >= lo) & (D[:, 1] <= hi)].mean(0)
        off = np.clip(-(h - h[1]), -limit, limit); off[1] = 0.0
        t.M = (off, np.ones(3))
        return t
    P["高光为白 (95–99%)"] = highlight_white
    P["高光为白 (99–99.9%)"] = lambda d, sc, tm: highlight_white(d, sc, tm, (99.0, 99.9))
    for k in (0.25, 0.5, 0.75):
        P[f"3200K 中灰, 力度 {k}"] = (lambda kk: lambda d, sc, tm: frame(d, sc, tm, g32(d), kk))(k)
    # Fuji US6160634 的灰平衡 (三种估计 0.2:0.5:0.3, 剔除高彩度像素; 系数为专利示例值)
    P["US6160634 (档案中灰)"] = lambda d, sc, tm: frame(d, sc, tm, d["gray0"], float(d["strength"]), gb="us6160634")
    for thr in (0.1, 0.2, 0.3):
        P[f"US 相对中位 {thr}"] = (lambda t: lambda d, sc, tm: frame(d, sc, tm, d["gray0"], 0.0, gb="us6160634", thr=t))(thr)
        P[f"US 相对灰轴 {thr}"] = (lambda t: lambda d, sc, tm: frame(d, sc, tm, d["gray0"], 0.0, gb="us6160634_axis", thr=t))(thr)
    P["US6160634 (3200K 中灰)"] = lambda d, sc, tm: frame(d, sc, tm, g32(d), float(d["strength"]), gb="us6160634")
    return P


KEY = ["auto (现行)", "US 相对中位 0.1", "US 相对灰轴 0.1", "US 相对中位 0.2", "US 相对灰轴 0.2", "US 相对中位 0.3", "US 相对灰轴 0.3"]


def eval_cond(cond, names=None):
    import colour
    from filmlab.scanning import timing as tm
    eng, _ = engine_for("R1-00-1")
    sc = eng.scanner
    P = policies()
    names = names or list(P)
    rows = {}
    for sid in SCENES:
        f = f"{OUT}/{cond}/{sid.replace(':', '_')}.npz"
        if not os.path.exists(f):
            continue
        d = dict(np.load(f))
        r = {}
        for n in names:
            sc.timing = P[n](d, sc, tm)
            S = lab_lin(sc.to_rgb(d["C"]))
            R = d["R"]
            if str(d["mode"]) == "uniform":
                r[n] = (float(colour.delta_E(S[0], R[0], method="CIE 2000")), *map(float, S[0] - R[0]))
            else:
                v = (R[:, 0] > 5) & (S[:, 0] > 2)
                r[n] = (float(np.median(colour.delta_E(S[v], R[v], method="CIE 2000"))), *map(float, np.median(S[v] - R[v], 0)))
        rows[sid] = r
    return rows, names


def report(rows, names):
    groups = {}
    for sid in rows:
        groups.setdefault((LAB[sid], group(sid)), []).append(sid)
    print(f"{'冲印店 / 组':28s}" + "".join(f"{n[:14]:>16s}" for n in names))
    for (lab, g), ids in sorted(groups.items()):
        print(f"{lab + ' / ' + g + f' ({len(ids)})':28s}" + "".join(f"{np.mean([rows[s][n][0] for s in ids]):16.2f}" for n in names))
    same = [s for s in rows if LAB[s] == "epfl_frontier" and group(s) != "halo_test"]
    print(f"{'同店全部 (' + str(len(same)) + ')':28s}" + "".join(f"{np.mean([rows[s][n][0] for s in same]):16.2f}" for n in names))


if __name__ == "__main__":
    cmd, cond = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "base")
    if cmd == "cache":
        import oracle_timing as ot
        for sid in SCENES:              # 单应性先串行求好
            if sid not in UNIFORM and sid not in WEDGE:
                try:
                    ot.geometry(sid)
                except Exception as e:
                    print(sid, "对齐失败", e, flush=True)
        import multiprocessing as mp
        run = os.environ["SCENES"].split(",") if os.environ.get("SCENES") else SCENES
        with mp.get_context("spawn").Pool(int(os.environ.get("NPROC", "4"))) as p:
            for sid, msg in p.imap_unordered(cache_one, [(s, cond) for s in run]):
                print(sid, msg, flush=True)
    else:
        rows, names = eval_cond(cond, KEY if os.environ.get("KEYONLY") else None)
        json.dump(rows, open(f"{OUT}/eval_{cond}.json", "w"), indent=1)
        report(rows, names)
