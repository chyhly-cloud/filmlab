"""胶片与显影液数据库 (silvergrain/data/), 只含颗粒需要的量: 感光度、晶体形态、特性曲线、光谱感光度 (黑白: 全色响应用)、
官方 RMS 颗粒度与测量条件、来源 URL; 显影液: T50 的显影晶体成团 (μ, σ) 与每片显影阈值 r。

黑白片: silvergrain/data/bw_t48.json (T48) + bw_t49_spectra.json (T49b 光谱修正); Tri-X 的曲线沿用 T47 (Schmidt D-76)。
  = filmlab 代际 t49_bw / t50_bw / t51_bw 的黑白档案 (materials.bw_film_def(key, "t49"))。
彩色片: silvergrain/data/colour_films.json (由 filmlab 档案导出; spektrafilm 数据, CC BY-SA 4.0)。
"""
import re
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from . import database as db

BW_DEVELOPERS = ("d76", "xtol", "rodinal", "fx39")
COLOUR_PROCESSES = ("C-41", "ECN-2", "E-6")


@dataclass(frozen=True, eq=False)
class Curve:
    """特性曲线 (晶体模型的输入, 与 filmlab 的 Emulsion 同样的接口: log_exposure, S_nom, design, stock.density_curves)。
    log_exposure: (n,) 档案横轴 (log10 lux·s; 绝对曝光另由感光度定, 见 dye.exposure_axis); density_curves: (n, 3) 每层
    染料当量密度 (不含片基); design: "negative" | "reversal" | "bw"。S_nom: 原生冲洗下的成像银 (反转片 = Dmax − 染料)。"""
    log_exposure: np.ndarray
    density_curves: np.ndarray
    design: str

    @property
    def S_nom(self):
        D = self.density_curves
        return D.max(0) - D if self.design == "reversal" else D

    @property
    def stock(self):
        return self


@dataclass(frozen=True)
class Film:
    """一种胶片的公开物理量。kind: "bw" (显式银粒) | "negative" | "reversal" (染料云)。
    ei: 晶体推导与测光所用的感光度 (包装感光度; 净密度 0.1 处 H = 0.8/EI, 18% 灰 H = 10/EI)。
    thickness_um: T 颗粒 / 平板晶体厚度 (µm); None = 传统三维晶体。illuminant: 设计光源 (决定各层光子数)。
    process: 原生冲洗 (黑白 "d76"; 彩色 "C-41" / "ECN-2" / "E-6")。"""
    key: str
    name: str
    kind: str
    ei: float
    thickness_um: float | None
    crystal: str
    illuminant: str
    process: str
    curve_note: str = ""
    source: str = ""
    official: tuple = field(default=(), compare=False)

    @property
    def is_bw(self):
        return self.kind == "bw"

    def curve(self):
        """彩色: Curve (档案横轴); 黑白: dict(x, net, gross_fog) (以净 D 0.10 为零点的 log10 相对曝光)。"""
        return _colour_curve(self.key) if not self.is_bw else _bw_def(self.key)["curve_def"]

    def spectral(self):
        """光谱感光度: 黑白 (lam, logS) 一条; 彩色 (wavelengths, (81, 3) log10 每层)。"""
        if self.is_bw:
            return _bw_def(self.key)["spectral"]
        f = db.colour_films()
        return np.asarray(f["wavelengths"]), np.asarray(f["films"][self.key]["log_sensitivity"])


@dataclass(frozen=True)
class Developer:
    """显影液 / 冲洗工艺。黑白 (T50): 显影晶体成团 (cluster_mu 每团平均显影晶体数, cluster_um 团的高斯半径 µm; 0 = 泊松) 与
    每片的显影阈值光子数 r (由 Schmidt 该组合的曲线定; 没有数据的组合用 4 = D-76)。彩色: 工艺名 (T46 GRAIN_PROCESS)。"""
    key: str
    name: str
    kind: str                 # "bw" | "colour"
    cluster_mu: float = 0.0
    cluster_um: float = 0.0
    note: str = ""

    def photons_r(self, film_key):
        if self.kind != "bw":
            return 4.0
        return float(db.bw_t50_developers()["photons_r"].get(film_key, {}).get(self.key, 4.0))


# ------------------------------------------------------------------ 黑白 --
def _bw_film_def(key):
    """materials.bw_film_def(key, "t49") 的同一规则: T48 档案 + T49b 光谱; Tri-X 的曲线 = T47 (Schmidt D-76)。"""
    d = db.bw_t48()[key]
    c = d["curve"]
    if c is None:
        curve, fog, _ = db.trix_t47()
        src = "Tri-X 400: Schmidt D-76 曲线 + T47 标定"
        note = "Schmidt D-76 原液 20 °C 9 min (T47)"
    else:
        curve, fog, src, note = (c["x"], c["net"]), c["gross_fog"], f"T48: {d['name']}; 曲线 {c['process']}", c["process"]
    spectral = (d["spectral"]["lam"], d["spectral"]["logS"])
    sp49 = db.bw_t49_spectra()
    spec_src = src                       # 晶体档案的 source (= filmlab 曲线集 t48 的写法; 光谱不影响晶体)
    if key in sp49:
        spectral = (sp49[key]["lam"], sp49[key]["logS"])
        src = src + "; 光谱 T49b"
    return dict(name=d["name"], ei=float(d["ei"]), thickness_um=d["thickness"], curve=curve, gross_fog=fog, spectral=spectral,
                source=src, spec_source=spec_src, curve_note=note, crystal=d["crystal"])


@lru_cache(maxsize=None)
def _bw_def(key):
    f = _bw_film_def(key)
    x, net = map(np.asarray, f["curve"])
    f["curve_def"] = dict(x=x, net=net, gross_fog=float(f["gross_fog"]))
    f["spectral"] = tuple(map(np.asarray, f["spectral"]))
    return f


# ------------------------------------------------------------------ 彩色 --
@lru_cache(maxsize=None)
def _colour_curve(key):
    d = db.colour_films()["films"][key]
    return Curve(np.asarray(d["log_exposure"], float), np.asarray(d["density_curves"], float), d["design"])


def _official_rows():
    """官方颗粒度表 (T46) 按档案键分组 (curve 列的第一个词; 代用行标明)。"""
    out = {}
    for r in db.official_rms():
        k = r["curve"].split()[0] if r["curve"] else ""
        if k.startswith("bw_film"):
            k = {"trix400": "kodak_trix_400", "trix320": "kodak_trix_400", "plusx125": "kodak_plusx_125",
                 "tmax100": "kodak_tmax_100", "tmax400": "kodak_tmax_400", "acros100": "fuji_acros_100",
                 "tmaxp3200": ""}.get(r["key"], "")
        if r["metric"] == "none" or not k:
            if r["film"].startswith(("HP5", "FP4", "Pan F", "Delta 100", "Delta 400")):
                k = {"HP5": "ilford_hp5_plus", "FP4": "ilford_fp4_plus", "Pan F": "ilford_pan_f_plus",
                     "Delta 100": "ilford_delta_100", "Delta 400": "ilford_delta_400"}[re.match(r"(HP5|FP4|Pan F|Delta \d+)", r["film"]).group(1)]
            else:
                continue
        proxy = "代用" in r["curve"]
        if proxy:                       # 该行的官方值属于另一种片 (借用这条曲线), 不算这种片的官方值
            continue
        out.setdefault(k, []).append(dict(film=r["film"], metric=r["metric"], value=r["value"], condition=r["condition"],
                                          process=r["process"], iso=r["iso"], proxy_curve=proxy, source=r["source"],
                                          note=r["note"]))
    return out


@lru_cache(maxsize=None)
def films():
    """{档案键: Film}。"""
    off = _official_rows()
    out = {}
    for k in db.bw_t48():
        f = _bw_def(k)
        out[k] = Film(key=k, name=f["name"], kind="bw", ei=f["ei"], thickness_um=f["thickness_um"], crystal=f["crystal"],
                      illuminant="D55", process="d76", curve_note=f["curve_note"], source=f["source"],
                      official=tuple(off.get(k, ())))
    cf = db.colour_films()["films"]
    for k, d in cf.items():
        if k == "bw_pan_generic":
            continue
        crystal = "T 颗粒 (0.12 µm)" if d["thickness_um"] is not None else "传统三维"
        out[k] = Film(key=k, name=d["name"], kind=d["design"], ei=float(d["ei"]), thickness_um=d["thickness_um"],
                      crystal=crystal, illuminant=d["illuminant"], process=d["native_process"],
                      curve_note=f"数据册特性曲线 ({d['native_process']})", source="spektrafilm 档案 (CC BY-SA 4.0); 晶体: T46 通用推导",
                      official=tuple(off.get(k, ())))
    return out


@lru_cache(maxsize=None)
def developers():
    """{键: Developer}: 黑白 d76 / xtol / rodinal / fx39 (T50), 彩色 C-41 / ECN-2 / E-6 (T46)。推冲不可用 (T50 结论 3), 不列。"""
    d = db.bw_t50_developers()["developers"]
    names = {"d76": "Kodak D-76 (原液; 参照, 泊松)", "xtol": "Kodak XTOL", "rodinal": "Rodinal (Adox, 1+50)", "fx39": "Adox FX-39"}
    out = {k: Developer(k, names.get(k, k), "bw", float(v["cluster_mu"]), float(v["cluster_um"]), v.get("note", ""))
           for k, v in d.items()}
    for p in COLOUR_PROCESSES:
        out[p] = Developer(p, p, "colour", note="T46 冲洗工艺 (每晶体产量 CV、扩散长度 L、κ 倍数)")
    return out


def film(key):
    if isinstance(key, Film):
        return key
    try:
        return films()[key]
    except KeyError:
        raise KeyError(f"未知胶片 {key!r}; 可选: {', '.join(films())}") from None


def developer(key, film_=None):
    """显影液 (None = 胶片的原生冲洗)。"""
    if isinstance(key, Developer):
        return key
    if key is None:
        key = film(film_).process if film_ is not None else "d76"
    k = {"D-76": "d76", "D76": "d76", "XTOL": "xtol", "Rodinal": "rodinal", "FX39": "fx39", "FX-39": "fx39", "ID-11": "d76"}.get(key, key)
    k = k.lower() if k.lower() in BW_DEVELOPERS else k
    try:
        return developers()[k]
    except KeyError:
        raise KeyError(f"未知显影液 / 冲洗工艺 {key!r}; 可选: {', '.join(developers())}") from None
