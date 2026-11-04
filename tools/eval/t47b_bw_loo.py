"""T47b: T47 的黑白片常数 (三维晶体有效量子效率 ×6.61、D-76 显影银形态: 丝团面积比、丝径) 是否对所有黑白片通用。

两组检验 (docs/tasks/T47_TRIX_SILVER_GRAIN.md "T47b"):
  A. 官方 RMS 颗粒度 (tools/eval/t46/films.tsv 的黑白片, 有 RMS 的; Ilford 不公布, 不纳入):
     Kodak 48 µm 漫射 RMS × 1000, 净漫射密度 1.0。每片按官方数据所用显影液判定: 只有 D-76 与 T47 的标定 (Schmidt D-76)
     同一口径 → 判定; HC-110 (Tri-X、Plus-X)、Microfine (Acros) 只报告。
     留一: 每片用 T46 留一检验里去掉该片后拟合的 (η, β) (tools/eval/t46/loo.json); η_3D ×6.61 来自 Schmidt Tri-X (不含任何
     Kodak 官方值), 银形态是先验。
  B. Schmidt 同一显微镜、同一估计器、同一显影液 (D-76 原液 20 °C) 的 6 种片 (tools/eval/t47/schmidt_bw_d76.csv):
     Tri-X (标定用) + T-Max 100 / 400、Delta 100 / 400 (平板)、Pan F+ (三维) 的 RMS_D (平台 → 48 µm 圆孔), 总 D ≈ 1。
每片的特性曲线: Schmidt 的 D-76 曲线 (同片种; Plus-X / Tri-X 320 借 Tri-X、Acros 借 T-Max 100 的曲线形状, 只报告);
晶体由该曲线对应的感光度推出 (T46 的约定)。同一输入下也算 T46 模型 (grain.derive_spec, κ_Ag) 的预测, 并列 T46 原留一 (通用曲线)。
用法: python3 tools/eval/t47b_bw_loo.py   → tools/eval/t47/t47b_loo.json"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from filmlab import grain, materials  # noqa: E402
from filmlab.materials import bw_film, from_stock  # noqa: E402

D = HERE / "t47"
TRIX_FOG = 0.318
# 官方 RMS 的显影液 (各数据册 "Image structure" 一节, 本任务核对): Tri-X F-4017 与 Plus-X F-8 为 HC-110 (B);
# T-Max 100 F-4016 与 T-Max 400 F-4043 为 D-76; Acros 为 Fujifilm Microfine
OFFICIAL_DEV = {"trix400": "HC-110 (B)", "trix320": "HC-110 (B)", "plusx125": "HC-110 (B)", "tmax100": "D-76",
                "tmax400": "D-76", "acros100": "Microfine"}
CURVE_OF = {"trix400": "trix", "trix320": "trix", "plusx125": "trix", "tmax100": "tmx", "tmax400": "tmy", "acros100": "tmx"}
SCHMIDT = {"trix": ("Tri-X 400", None), "tmx": ("T-Max 100 (TMX)", 0.12), "tmy": ("T-Max 400 (TMY)", 0.12),
           "d100": ("Delta 100", 0.12), "d400": ("Delta 400", 0.12), "panf": ("Pan F+", None)}


def schmidt_curves():
    """{key: (E, net, fog, EI_eff, RMS_D)}"""
    out = {}
    rows = [l for l in (D / "schmidt_density_19.csv").read_text().splitlines() if l and not l.startswith("#")]
    h = rows[0].split(",")
    t = {k: np.array([float(r.split(",")[i]) for r in rows[1:]]) for i, k in enumerate(h)}
    out["trix"] = (t["chart_relative_log10_exposure"], t["net_D"], TRIX_FOG, 400.0, 13.0)
    names = {v[0]: k for k, v in SCHMIDT.items()}
    rows = [r for r in csv.DictReader(l for l in (D / "schmidt_bw_d76.csv").read_text().splitlines() if not l.startswith("#"))]
    for name in dict.fromkeys(r["film"] for r in rows):
        rr = [r for r in rows if r["film"] == name]
        E = np.array([float(r["E"]) for r in rr]); g = np.array([float(r["gross_D"]) for r in rr])
        net = np.array([float(r["net_D"]) for r in rr])
        out[names[name]] = (E, net, float(g[0] - net[0]), float(rr[0]["EI_eff"]), float(rr[0]["RMS_D"]))
    return out


def H_of(E, net, ei):
    """相对曝光 → 绝对曝光: 净 D 0.10 处 H = 0.8/EI。"""
    m = np.maximum.accumulate(net) + 1e-9 * np.arange(len(net))
    E01 = float(np.interp(0.10, m, E))
    return 0.8 / ei * 10.0 ** (E - E01)


def t47_spec(E, net, fog, ei, thickness, eta, beta, scale_tgrain=True):
    U = dict(materials.grain_universal(eta=eta, beta=beta))
    return grain.silver_derive(ei, U, materials.GRAIN_T47_ETA3D_SCALE, H_of(E, net, ei), net, fog, materials.GRAIN_T47_MORPH,
                               thickness_um=thickness, scale_tgrain=scale_tgrain)


def H_at(spec, net):
    lh = np.linspace(-7, 3, 5001)
    Dn = grain.silver_curve(spec, lh) - grain.silver_curve(spec, [-14.0])[0]
    return float(10 ** np.interp(net, np.maximum.accumulate(Dn + 1e-12 * np.arange(len(Dn))), lh))


def t47_kodak48(spec):
    return 1000 * float(grain.silver_granularity(spec, H_at(spec, 1.0), [42.5], 0.0, "diffuse", adjacent=False)[0])


def t47_schmidt_rmsd(spec, net):
    """Schmidt 的 RMS_D: 大孔径平台 σ²A 换算到 48 µm 圆孔 (1000 √σ²A · 2 / (48 √π)); 漫射 (T47 的显微口径)。"""
    s = float(grain.silver_granularity(spec, H_at(spec, net), [200.0], 0.0, "diffuse", adjacent=False)[0]) * 200.0
    return 1000 * s * 2 / (48 * np.sqrt(np.pi))


def emulsion_from_curve(E, net, fog, ei):
    """T46 对照用: 以这条曲线 (净 D 对绝对 log H) 为特性曲线的黑白乳剂 (光谱等取通用全色片)。"""
    st = bw_film().stock
    x = np.linspace(-5.0, 2.0, 256)
    lh = np.log10(H_of(E, net, ei))
    slope = (net[-1] - net[-3]) / (lh[-1] - lh[-3])
    d = np.where(x <= lh[-1], np.interp(x, lh, net, left=0.0), net[-1] + slope * (x - lh[-1]))
    return from_stock(st.with_(log_exposure=x, density_curves=np.repeat((np.clip(d, 0, None) + 0.0)[:, None], 3, 1)))


def t46_pred(E, net, fog, ei, thickness, eta, beta, target="kodak", schmidt_net=None):
    """T46 模型 (grain.derive_spec, 显影银 κ_Ag, 工艺 BW) 在同一条曲线上的预测。"""
    e = emulsion_from_curve(E, net, fog, ei)
    U = materials.grain_universal(eta=eta, beta=beta)
    sp = grain.derive_spec(e, ei, "D55", U, kind="silver", thickness_um=thickness, process=materials.process_grain("BW"))
    Hx = grain.exposure_axis(e, ei)[:, 1]
    Dm, sg = grain.layer_curves(sp, 1, Hx)
    Dm = Dm - Dm.min(); i = np.argsort(Dm)
    return 1000 * float(np.interp(1.0 if target == "kodak" else schmidt_net, Dm[i], sg[i]))


def main():
    S = schmidt_curves()
    loo46 = json.loads((HERE / "t46" / "loo.json").read_text())
    rows = [r for r in csv.DictReader(open(HERE / "t46" / "films.tsv"), delimiter="\t")
            if r["class"] == "bw" and r["metric"] == "rms_net1" and r["key"] != "tmaxp3200"]
    A = []
    for r in rows:
        k = r["key"]
        E, net, fog, _, _ = S[CURVE_OF[k]]
        ei = float(r["iso"])
        th = None if r["model_morph"].startswith("3D") else materials.GRAIN_T_UM
        lo = loo46["loo"][k]
        sp = t47_spec(E, net, fog, ei, th, lo["eta"], lo["beta"])
        p47 = t47_kodak48(sp)
        p47v1 = t47_kodak48(t47_spec(E, net, fog, ei, th, lo["eta"], lo["beta"], scale_tgrain=False)) if th else p47
        p46 = t46_pred(E, net, fog, ei, th, lo["eta"], lo["beta"])
        A.append(dict(key=k, film=r["film"], iso=ei, morph="3D" if th is None else "T", developer=OFFICIAL_DEV[k],
                      judged=OFFICIAL_DEV[k] == "D-76", curve=CURVE_OF[k], official=float(r["value"]), t47=p47,
                      t47_ratio=p47 / float(r["value"]), t47_3d_only_ratio=p47v1 / float(r["value"]), t46_same_curve_ratio=p46 / float(r["value"]),
                      t46_loo_ratio=lo["ratio"], ecd_um=list(sp.ecd_um), coat_g_m2=float(sum(sp.coat_g_m2)),
                      covering_power=grain.silver_budget(sp)["covering_power"]))
        print(f"A {k:9s} {r['film']:22s} {OFFICIAL_DEV[k]:10s} 官方 {float(r['value']):5.1f}  T47 {p47:5.1f} ({p47/float(r['value']):.2f}; 只三维 {p47v1/float(r['value']):.2f})"
              f"  T46 同曲线 {p46/float(r['value']):.2f}  T46 留一 {lo['ratio']:.2f}", flush=True)
    B = []
    eta0, beta0 = materials.GRAIN_T46_ETA, materials.GRAIN_T46_BETA
    for k, (name, th) in SCHMIDT.items():
        E, net, fog, ei, rms = S[k]
        sp = t47_spec(E, net, fog, ei, th, eta0, beta0)
        nt = 1.0 - fog
        p47 = t47_schmidt_rmsd(sp, nt)
        p47v1 = t47_schmidt_rmsd(t47_spec(E, net, fog, ei, th, eta0, beta0, scale_tgrain=False), nt) if th else p47
        p46 = t46_pred(E, net, fog, ei, th, eta0, beta0, target="schmidt", schmidt_net=nt)
        dmax = float(np.max(np.abs(grain.silver_density(sp, H_of(E, net, ei)) - grain.silver_density(sp, 0.0) - net)))
        B.append(dict(key=k, film=name, morph="3D" if th is None else "T", ei_eff=ei, net_at_total1=nt, measured=rms,
                      t47=p47, t47_ratio=p47 / rms, t47_3d_only_ratio=p47v1 / rms, t46_ratio=p46 / rms, role="标定" if k == "trix" else "留出",
                      curve_max_abs=dmax, ecd_um=list(sp.ecd_um), coat_g_m2=float(sum(sp.coat_g_m2)),
                      covering_power=grain.silver_budget(sp)["covering_power"]))
        print(f"B {k:5s} {name:16s} EI {ei:5.0f} 净 {nt:.3f} 实测 {rms:5.1f}  T47 {p47:5.1f} ({p47/rms:.2f}; 只三维 {p47v1/rms:.2f})  T46 {p46/rms:.2f}"
              f"  曲线最大误差 {dmax:.3f}  ECD {np.round(sp.ecd_um, 2)}  银 {sum(sp.coat_g_m2):.2f}", flush=True)
    out = dict(official=A, schmidt=B)
    (D / "t47b_loo.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    return out


if __name__ == "__main__":
    main()
