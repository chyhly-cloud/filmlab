"""T46: 38 种胶片官方 RMS 颗粒度的留一检验 (silvergrain 独立实现; 与 filmlab tools/eval/t46_grain_loo.py 同一算法、同一数据)。

每种胶片只用自己的公开物理量 (感光度、设计光源、晶体形态、彩色染料 / 黑白银、特性曲线、冲洗工艺) 与通用常数生成晶体群
(dye.derive_spec), 预测官方颗粒指标 (data/official_rms.tsv):
  curve_*      Kodak 电影负片每层 48 µm 漫射 RMS σ(净密度) (Status M; data/kodak_gran/)
  rms_net1     黑白片净漫射密度 1.0 处的 RMS (单层银; 曲线为 filmlab 的通用全色曲线, 同 T46)
  rms_visual   反转片漫射视觉密度 1.0 处的 RMS, 三层染料按视觉权重合成
  rms_fujineg  Fuji 彩色负片 (Fuji 自己的测法; 族常数 1 个, 由训练片定)
  PGI          Kodak Print Grain Index: PGI = C + (2/ln 1.06)·ln σ_印片视觉 (族常数 C 由训练片定)
通用常数中只有 η 与 β 由官方颗粒度定; 留一: 每次去掉一种, 在其余上定 (η, β) 与族常数, 再预测去掉的那种。
注意: 这里的黑白行是 T46 的 κ_Ag 银量模型 (dye.derive_spec kind="silver"), 显式银粒的黑白检验见 t47b.py。"""
import numpy as np
from scipy.ndimage import median_filter

from .. import database as db
from .. import dye
from ..emulsion import process_params, universal
from ..films import Curve

PGI_SLOPE = 2.0 / np.log(1.06)
FAMILY = ("PGI", "rms_fujineg", "rms_visual", "rms_net1")
LOGETA = np.linspace(np.log(0.002), np.log(0.4), 41)
BETAS = np.round(np.arange(0.0, 1.51, 0.125), 3)
T_UM = 0.12


def rows():
    return [r for r in db.official_rms() if r["metric"] not in ("none",) and r["key"] not in ("kodachrome64", "tmaxp3200")]


def _curve(key):
    d = db.colour_films()["films"]["bw_pan_generic" if key.startswith("bw_film") else key]
    return Curve(np.asarray(d["log_exposure"]), np.asarray(d["density_curves"]), d["design"]), d


def kodak(name):
    """Kodak 每层 (净密度, σ) (t45_grain_check.kodak 的同一处理: 净 = 累积最大, σ 对数中值滤波 5 点)。"""
    K = db.kodak_granularity(name)
    out = {}
    for i, c in enumerate("RGB"):
        D = np.array(K["density_" + c]); net = np.maximum.accumulate(np.clip(D - D[:3].mean(), 0, None))
        if "sigma_" + c in K:
            out[i] = (net, np.exp(median_filter(np.log(np.array(K["sigma_" + c])), 5, mode="nearest")))
    return out


def ill(r):
    return "BB3200" if r["design_light"].startswith("3200") else "D55"


def visual_weights(d, eps, base):
    cf = db.colour_films()
    V = np.asarray(cf["cie_y"]) * np.asarray(cf["illuminants"]["D55"])
    T = 10.0 ** -(np.asarray(d)[..., None, :] * eps).sum(-1) * 10.0 ** -base
    Dv = -np.log10((T * V).sum(-1) / V.sum())
    w = np.einsum("...l,lc->...c", T * V, eps) / (T * V).sum(-1)[..., None]
    return w, Dv


def model_layers(r, eta, beta, U0):
    e, d = _curve(r["curve"].split()[0])
    kind = "silver" if r["class"] == "bw" else "dye"
    t = None if r["model_morph"].startswith("3D") else T_UM
    proc = process_params("BW" if r["class"] == "bw" else r["process"])
    U = dict(U0, eta=eta, beta=beta)
    sp = dye.derive_spec(e, float(r["iso"]), ill(r), U, kind=kind, thickness_um=t, process=proc)
    Hx = dye.exposure_axis(e, float(r["iso"]))
    rev = e.design == "reversal"
    out = []
    for c in range(3):
        cc = 1 if sp.single_layer else c
        out.append(dye.layer_curves(sp, cc, Hx[:, c], reversal=rev))
    return e, d, sp, Hx, out


def predict(r, eta, beta, U0):
    e, d, sp, Hx, L = model_layers(r, eta, beta, U0)
    m = r["metric"]
    if m.startswith("curve"):
        K = kodak(r["value"].split()[0])
        err = []
        for c, (net, s) in K.items():
            D, sg = L[c]; D = D - D.min()
            i = np.argsort(D)
            for dd in (0.0, 0.2, 0.5, 1.0, 1.5):
                if dd > 0.95 * min(D.max(), net.max()):
                    continue
                err.append(np.log(np.interp(dd, D[i], sg[i]) / np.interp(dd, net, s)))
        return np.array(err), None, "curve"
    if m == "rms_net1":
        D, sg = L[1]; D = D - D.min(); i = np.argsort(D)
        return float(np.interp(1.0, D[i], sg[i])) * 1e3, float(r["value"]), m
    Dm = np.stack([l[0] for l in L], -1); Sg = np.stack([l[1] for l in L], -1)
    if m in ("rms_visual", "rms_fujineg"):
        w, Dv = visual_weights(np.clip(Dm, 0, None), np.asarray(d["dye_density"]), np.asarray(d["base_density"]))
        sv = np.sqrt(((w * Sg) ** 2).sum(-1))
        gross = "gross" in r["condition"] or "毛" in r["condition"]
        tgt = 1.0 if gross else Dv.min() + 1.0
        i = np.argsort(Dv)
        return float(np.interp(tgt, Dv[i], sv[i])) * 1e3, float(r["value"]), m
    if m == "PGI":
        cf = db.colour_films()
        V = np.asarray(cf["cie_y"]) * np.asarray(cf["illuminants"]["D50"])
        g = (np.asarray(cf["paper_endura_dye_density"]) * V[:, None]).sum(0) / V.sum()
        H = 10.0 / float(r["iso"])
        sc = np.array([np.interp(np.log10(H), np.log10(Hx[:, c]), Sg[:, c]) for c in range(3)])
        sv = np.sqrt(((g * sc) ** 2).sum())
        v = r["value"].split()[0]
        return float(sv), (None if v.startswith("<") else float(v)), m
    raise ValueError(m)


def residuals(R, P, ib, ie, train):
    fam = {}
    for r in R:
        if r["key"] in train and r["metric"] in FAMILY:
            p, o, m = P[r["key"]][ib][ie]
            if o is None:
                continue
            fam.setdefault(m, []).append((o / PGI_SLOPE - np.log(p)) if m == "PGI" else np.log(o / p))
    C = {k: float(np.mean(v)) for k, v in fam.items()}
    res = {}
    for r in R:
        p, o, m = P[r["key"]][ib][ie]
        if m == "curve":
            res[r["key"]] = (float(np.sqrt(np.mean(p ** 2))), float(np.mean(p)))
        elif m == "PGI":
            if o is None:
                pg = PGI_SLOPE * (np.log(p) + C.get("PGI", 0.0))
                res[r["key"]] = (max(0.0, (pg - 25.0) / PGI_SLOPE), max(0.0, (pg - 25.0) / PGI_SLOPE))
            else:
                e = np.log(p) + C.get("PGI", 0.0) - o / PGI_SLOPE
                res[r["key"]] = (abs(e), e)
        elif m in FAMILY:
            e = np.log(p) + C.get(m, 0.0) - np.log(o)
            res[r["key"]] = (abs(e), e)
        else:
            e = np.log(p / o)
            res[r["key"]] = (abs(e), e)
    return res, C


def fit(R, P, train):
    best = None
    for ib in range(len(BETAS)):
        for ie in range(len(LOGETA)):
            res, C = residuals(R, P, ib, ie, train)
            cost = sum(res[k][0] ** 2 for k in train)
            if best is None or cost < best[0]:
                best = (cost, ib, ie, C)
    return best


def run(full=True, log=print):
    """full=True: 全网格 (13 β × 41 η × 38 片) 的全体拟合与留一 (数分钟)。full=False: 只在参考结果 (loo.json) 记录的
    每片 (β, η) 上重算预测 (快速核对)。返回 dict (与 tools/eval/t46/loo.json 同结构) 与对照。"""
    from .refdata import load
    ref = load("t46_loo_reference.json")
    R = rows()
    U0 = universal()
    keys = [r["key"] for r in R]
    if not full:
        # 只在参考的全体拟合点 (β, η) 上算全部 38 片, 族常数与残差应与参考的 all_fit 逐位相同
        ib = int(np.argmin(np.abs(BETAS - ref["beta"]))); ie = int(np.argmin(np.abs(np.exp(LOGETA) - ref["eta"])))
        P = {}
        for r in R:
            row = [[None] * len(LOGETA) for _ in BETAS]
            row[ib][ie] = predict(r, np.exp(LOGETA[ie]), BETAS[ib], U0)
            P[r["key"]] = row
        res, C = residuals(R, P, ib, ie, set(keys))
        d = max(abs(res[k][0] - ref["all_fit"][k]["rms_log"]) + abs(res[k][1] - ref["all_fit"][k]["bias_log"]) for k in keys)
        return dict(mode="quick", beta=float(BETAS[ib]), eta=float(np.exp(LOGETA[ie])), family_const=C,
                    all_fit={k: dict(rms_log=v[0], bias_log=v[1]) for k, v in res.items()}, max_abs_diff_vs_reference=float(d))
    P = {}
    for r in R:
        P[r["key"]] = [[predict(r, np.exp(le), b, U0) for le in LOGETA] for b in BETAS]
        log(f"  T46 网格 {r['key']}")
    cost, ib, ie, C = fit(R, P, set(keys))
    allres, _ = residuals(R, P, ib, ie, set(keys))
    loo = {}
    for k in keys:
        _, jb, je, Ck = fit(R, P, set(keys) - {k})
        res, _ = residuals(R, P, jb, je, set(keys) - {k})
        loo[k] = dict(rms_log=res[k][0], bias_log=res[k][1], ratio=float(np.exp(res[k][1])), beta=float(BETAS[jb]),
                      eta=float(np.exp(LOGETA[je])))
    lo = np.array([v["rms_log"] for v in loo.values()])
    out = dict(beta=float(BETAS[ib]), eta=float(np.exp(LOGETA[ie])), family_const=C,
               all_fit={k: dict(rms_log=v[0], bias_log=v[1]) for k, v in allres.items()}, loo=loo,
               loo_median_abs_log=float(np.median(lo)), loo_rms_log=float(np.sqrt(np.mean(lo ** 2))))
    out["compare"] = compare(out, ref)
    return out


def compare(out, ref):
    d = [abs(out["loo"][k]["ratio"] - ref["loo"][k]["ratio"]) for k in ref["loo"]]
    return dict(beta=(out["beta"], ref["beta"]), eta=(out["eta"], ref["eta"]),
                max_abs_ratio_diff=float(max(d)), identical=bool(max(d) == 0.0 and out["beta"] == ref["beta"] and out["eta"] == ref["eta"]))
