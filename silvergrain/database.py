"""数据文件 (silvergrain/data/) 的读取。只读, 带缓存; 所有数值与来源都在 JSON / TSV 里, 这里不写任何数值。

  universal.json         跨片种通用常数 (T45 5219 结构、T46 η / β / κ_Ag / T 颗粒厚、冲洗工艺、T47 银丝形态与 η 修正、Tri-X 曲线)
  bw_t48.json            黑白片档案: 感光度、晶体形态、D-76 类特性曲线、光谱感光度、来源 URL (T48)
  bw_t49_spectra.json    T49b 修正后的光谱 (T-Max 100 与 Ilford 片)
  bw_t50_developers.json 显影液: 显影晶体成团 (μ, σ) 与每片显影阈值 r (T50); 推冲 (不可用, 只作记录)
  colour_films.json      彩色片的特性曲线、染料光谱、光谱感光度 (由 filmlab 档案导出, spektrafilm 数据, CC BY-SA 4.0)
  official_rms.tsv       46 种胶片的官方颗粒度与测量条件、晶体形态、来源 (T46 films.tsv)
  kodak_gran/*.json      Kodak 电影负片的漫射 RMS 颗粒度曲线数字化 (T45 / T46)
  wratten_filters.json   Kodak Wratten 滤镜透过率与公布的滤镜系数 (T49)
  scanners.json          冲印店扫描的空间传递 (T51 实测: Noritsu RPL、Frontier EPFL)
  upsampling_xyY.npz     Jakob–Hanika (2019) RGB → 光谱的查找表 (由 filmlab tools/build_upsampling.py 生成)
"""
import csv
import json
from functools import lru_cache
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"


def path(name):
    return DATA / name


@lru_cache(maxsize=None)
def load_json(name):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def universal():
    return load_json("universal.json")


def grain_5219():
    """T45 的 5219 标定 (GRAIN_5219 的形式: 元组)。"""
    g = universal()["t45_5219"]
    return dict(ecd_um=tuple(tuple(r) for r in g["ecd_um"]), kappa=tuple(g["kappa"]), fog_area_um2=tuple(g["fog_area_um2"]),
                packing_s0=tuple(g["packing_s0"]), ei=g["ei"], illuminant=g["illuminant"], thickness_um=g["thickness_um"],
                sigma_ln=tuple(g["sigma_ln"]))


def process_params():
    """冲洗工艺最后一步 (每晶体产量 CV、扩散长度 L、κ 倍数): {"C-41": {...}, "ECN-2", "E-6", "BW"}。"""
    return {k: dict(v) for k, v in universal()["process"].items()}


def t47_morph():
    return dict(universal()["t47"]["morph"])


def trix_t47():
    """Tri-X 400 的 T47 曲线 (Schmidt D-76) 与 F-4017 光谱: (curve (x, net), gross_fog, spectral (lam, logS))。"""
    t = universal()["trix_t47"]
    return tuple(tuple(r) for r in t["curve"]), t["gross_fog"], tuple(tuple(r) for r in t["spectral"])


def bw_t48():
    return load_json("bw_t48.json")


def bw_t49_spectra():
    return load_json("bw_t49_spectra.json")["films"]


def bw_t50_developers():
    return load_json("bw_t50_developers.json")


def colour_films():
    return load_json("colour_films.json")


def wratten():
    return load_json("wratten_filters.json")


def scanners():
    return load_json("scanners.json")


@lru_cache(maxsize=None)
def official_rms():
    """T46 的官方颗粒度表 (每行一个 dict; 列见 TSV 表头)。"""
    with open(DATA / "official_rms.tsv", encoding="utf-8") as f:
        return tuple(csv.DictReader(f, delimiter="\t"))


@lru_cache(maxsize=None)
def kodak_granularity(name):
    """Kodak 电影负片 RMS 颗粒度曲线数字化 (kodak_gran/<name>): 原始 JSON。"""
    return json.loads((DATA / "kodak_gran" / name).read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def upsampling_table():
    """Jakob–Hanika (x, y, 亮度) → 光谱 (81 波段, 380–780 nm / 5 nm) 的表; 不存在时 None。"""
    import numpy as np
    p = DATA / "upsampling_xyY.npz"
    if not p.exists():
        return None
    d = np.load(p)
    return d["x"], d["y"], d["lum"], d["spectra"].astype(np.float64)
