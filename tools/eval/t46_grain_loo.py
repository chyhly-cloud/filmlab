"""T46: 跨片种通用银盐颗粒的留一检验 (官方颗粒度 → tools/eval/t46/films.tsv)。

每种胶片只用自己的公开物理量 (感光度、设计光源、晶体形态 T 颗粒 / 三维、彩色染料 / 黑白银、特性曲线、冲洗工艺)
与通用常数 (filmlab.materials.grain_universal) 生成卤化银晶体群 (filmlab.grain.derive_spec), 预测官方颗粒指标:
  curve_*      Kodak 电影负片每层 48 µm 漫射 RMS σ(净密度) (Status M)
  rms_net1     黑白片净漫射密度 1.0 处的 RMS (单层银)
  rms_visual   反转片漫射视觉密度 1.0 处 (Kodak: 毛密度; Fuji: 高于最小密度 1.0) 的 RMS, 三层染料按视觉权重合成
  rms_fujineg  Fuji 彩色负片 (Fuji 自己的测法, 与反转片不可比 → 族常数 1 个, 由训练片定)
  PGI          Kodak Print Grain Index (E-58): PGI = C + (2 / ln 1.06)·ln σ_印片视觉 (Bartleson 标度: 颗粒度变 6% = 2 单位),
               σ_印片视觉 = √Σ (g_c σ_c)² (负片灰卡处每层 σ, g_c = 相纸染料的视觉权重); C 为族常数, 由训练片定
通用常数中只有 η (a0 处的每面积量子效率) 与 β (量子效率随投影面积的指数) 由官方颗粒度定; 留一: 每次去掉一种胶片,
在其余胶片上定 (η, β) 与族常数, 再预测去掉的那种。
用法: python3 tools/eval/t46_grain_loo.py [输出.json]"""
import csv, json, sys, re
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from filmlab import grain, materials
from filmlab.materials import emulsion, bw_film
from filmlab.stocks import load_stock
import t45_grain_check as chk

PGI_SLOPE = 2.0 / np.log(1.06)
# 族常数 (每族 1 个 log 偏移, 只由训练片定): PGI 与 Fuji 负片是测量定义的常数; 反转片 (E-6) 与黑白 (银) 是冲洗工艺的
# 每 µm³ 显影晶体积分密度 κ 相对彩色负片的倍数 (σ ∝ √κ, 所以 κ_工艺 = κ × e^{2C})。
FAMILY = ("PGI", "rms_fujineg", "rms_visual", "rms_net1")
LOGETA = np.linspace(np.log(0.002), np.log(0.4), 41)
BETAS = np.round(np.arange(0.0, 1.51, 0.125), 3)


def rows():
    return [r for r in csv.DictReader(open(HERE / "t46" / "films.tsv"), delimiter="\t")
            if r["metric"] not in ("none",) and r["key"] not in ("kodachrome64", "tmaxp3200")]


def film_emulsion(r):
    key = r["curve"].split()[0]
    return bw_film() if key.startswith("bw_film") else emulsion(key)


def ill(r):
    return "BB3200" if r["design_light"].startswith("3200") else "D55"


def visual_weights(e, d):
    """(…,3) 每层染料量 d 处, 漫射视觉密度对每层染料的偏导与视觉密度本身 (CIE ȳ × D55)。"""
    from filmlab import spectral
    eps = e.stock.dye_density; base = e.stock.base_density
    V = spectral.cmfs()[:, 1] * spectral.illuminant("D55")
    T = 10.0 ** -(np.asarray(d)[..., None, :] * eps).sum(-1) * 10.0 ** -base
    Dv = -np.log10((T * V).sum(-1) / V.sum())
    w = np.einsum("...l,lc->...c", T * V, eps) / (T * V).sum(-1)[..., None]
    return w, Dv


def model_layers(r, eta, beta, U0):
    e = film_emulsion(r)
    kind = "silver" if r["class"] == "bw" else "dye"
    t = None if r["model_morph"].startswith("3D") else materials.GRAIN_T_UM
    proc = materials.process_grain("BW" if r["class"] == "bw" else r["process"])
    U = dict(U0, eta=eta, beta=beta)
    sp = grain.derive_spec(e, float(r["iso"]), ill(r), U, kind=kind, thickness_um=t, process=proc)
    Hx = grain.exposure_axis(e, float(r["iso"]))
    rev = e.design == "reversal"
    out = []
    for c in range(3):
        cc = 1 if sp.single_layer else c
        D, s = grain.layer_curves(sp, cc, Hx[:, c], reversal=rev)
        out.append((D, s))
    return e, sp, Hx, out


def predict(r, eta, beta, U0):
    """→ (预测值, 官方值, 类型)。curve 类返回逐点 log 误差数组。"""
    e, sp, Hx, L = model_layers(r, eta, beta, U0)
    m = r["metric"]
    if m.startswith("curve"):
        K, _ = chk.kodak("../t46/" + r["value"].split()[0]) if r["key"] == "k5207" else chk.kodak(r["value"].split()[0])
        err = []
        for c, (net, s) in K.items():
            D, sg = L[c]; D = D - D.min()
            i = np.argsort(D)
            for d in (0.0, 0.2, 0.5, 1.0, 1.5):
                if d > 0.95 * min(D.max(), net.max()):
                    continue
                err.append(np.log(np.interp(d, D[i], sg[i]) / np.interp(d, net, s)))
        return np.array(err), None, "curve"
    if m == "rms_net1":
        D, sg = L[1]; D = D - D.min(); i = np.argsort(D)
        return float(np.interp(1.0, D[i], sg[i])) * 1e3, float(r["value"]), m
    Dm = np.stack([l[0] for l in L], -1); Sg = np.stack([l[1] for l in L], -1)
    if m in ("rms_visual", "rms_fujineg"):
        w, Dv = visual_weights(e, np.clip(Dm, 0, None))
        sv = np.sqrt(((w * Sg) ** 2).sum(-1))
        gross = "gross" in r["condition"] or "毛" in r["condition"]
        tgt = 1.0 if gross else Dv.min() + 1.0
        i = np.argsort(Dv)
        return float(np.interp(tgt, Dv[i], sv[i])) * 1e3, float(r["value"]), m
    if m == "PGI":
        pap = load_stock("kodak_portra_endura")
        from filmlab import spectral
        V = spectral.cmfs()[:, 1] * spectral.illuminant("D50")
        g = (pap.dye_density * V[:, None]).sum(0) / V.sum()          # 相纸 C M Y 染料的视觉密度 (单位浓度)
        # 负片的 R / G / B 层分别控制相纸的 C / M / Y; 相纸反差对所有片相同 (只影响族常数)
        H = 10.0 / float(r["iso"])                                    # 18% 灰卡的曝光 (测光表定标 H = 10/S)
        sc = np.array([np.interp(np.log10(H), np.log10(Hx[:, c]), Sg[:, c]) for c in range(3)])
        sv = np.sqrt(((g * sc) ** 2).sum())
        v = r["value"].split()[0]
        return float(sv), (None if v.startswith("<") else float(v)), m
    raise ValueError(m)


def grid(R, U0):
    """每种胶片在 (β, η) 网格上的预测: P[f][ib, ie] = 预测 (curve 为逐点误差数组)。"""
    P = {}
    for r in R:
        P[r["key"]] = [[predict(r, np.exp(le), b, U0) for le in LOGETA] for b in BETAS]
        print(r["key"], flush=True)
    return P


def residuals(R, P, ib, ie, train):
    """在训练集上算族常数, 返回每片 log 误差 (curve: 逐点 RMS 与平均, 单值: log 比)。"""
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
            if o is None:     # "<25": 预测 ≥ 25 才算误差
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


def main(dst=None):
    R = rows()
    U0 = materials.grain_universal()
    P = grid(R, U0)
    keys = [r["key"] for r in R]
    cost, ib, ie, C = fit(R, P, set(keys))
    allres, _ = residuals(R, P, ib, ie, set(keys))
    loo = {}
    for k in keys:
        _, jb, je, Ck = fit(R, P, set(keys) - {k})
        res, _ = residuals(R, P, jb, je, set(keys) - {k})
        loo[k] = dict(rms_log=res[k][0], bias_log=res[k][1], ratio=float(np.exp(res[k][1])), beta=float(BETAS[jb]),
                      eta=float(np.exp(LOGETA[je])))
        print(f"留一 {k:12s} 预测/官方 {np.exp(loo[k]['bias_log']):.2f}  |log| {loo[k]['rms_log']:.3f}  (β {BETAS[jb]}, η {np.exp(LOGETA[je]):.4f})")
    # T45 的纯预测 (β = 0, η = 5219 的 0.05; 无族常数之外的自由量)
    ie45 = int(np.argmin(np.abs(LOGETA - np.log(grain.ETA_REF)))); ib0 = 0
    t45, _ = residuals(R, P, ib0, ie45, set(keys))
    out = dict(beta=float(BETAS[ib]), eta=float(np.exp(LOGETA[ie])), family_const=C,
               all_fit={k: dict(rms_log=v[0], bias_log=v[1]) for k, v in allres.items()}, loo=loo,
               t45_pure={k: dict(rms_log=v[0], bias_log=v[1]) for k, v in t45.items()},
               classes={r["key"]: r["class"] for r in R})
    print("全体拟合: β", out["beta"], "η", round(out["eta"], 4), "族常数", C)
    lo = np.array([v["rms_log"] for v in loo.values()]); t4 = np.array([t45[k][0] for k in keys])
    print(f"留一 |log 误差| 中位 {np.median(lo):.3f} (×{np.exp(np.median(lo)):.2f}), RMS {np.sqrt(np.mean(lo**2)):.3f};"
          f"  T45 纯预测 中位 {np.median(t4):.3f}, RMS {np.sqrt(np.mean(t4**2)):.3f}")
    if dst:
        json.dump(out, open(dst, "w"), indent=1, ensure_ascii=False)
    return out


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
