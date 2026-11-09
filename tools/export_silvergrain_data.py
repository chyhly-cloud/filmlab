"""T55: 把 silvergrain 需要、但只在 filmlab 胶片档案里的数据导出到 silvergrain/data/ (filmlab → silvergrain 的单向导出)。

导出 (silvergrain/data/colour_films.json):
  · 彩色片的特性曲线 (log_exposure 256 点, 每层 density_curves)、设计 (负片 / 反转片)、原生冲洗工艺、设计光源,
    染料光谱密度 (81 波段) 与片基光谱密度、每层光谱感光度 (log10, 已含 filmlab 对钨丝灯片的再平衡);
    T46 的公开物理量 (感光度、光源、晶体形态) 按 materials.GRAIN_PHYS 的同一规则解析;
  · 通用全色黑白曲线 (materials.bw_film(): 仓库 T46 黑白行所用的 "bw_film" 曲线);
  · 相纸 Kodak Portra Endura 的染料光谱密度 (T46 PGI 的视觉权重);
  · CIE 1931 2° ȳ 与光源 D50 / D55 / D65 / BB3200 / BB2850 的 81 波段光谱 (filmlab.spectral, colour-science)。
数据来源: filmlab/data/profiles (spektrafilm, Andrea Volpato, CC BY-SA 4.0; 厂商数据册数字化)。浮点按 repr 写出, 读回逐位相同。
用法: python3 tools/export_silvergrain_data.py      (仓库根目录)"""
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from filmlab import materials, spectral  # noqa: E402
from filmlab.stocks import WAVELENGTHS, load_stock  # noqa: E402

COLOUR = ["kodak_vision3_500t", "kodak_vision3_200t", "kodak_vision3_250d", "kodak_vision3_50d", "kodak_portra_160",
          "kodak_portra_400", "kodak_portra_800", "kodak_ektar_100", "kodak_gold_200", "kodak_ultramax_400",
          "fujifilm_pro_400h", "fujifilm_xtra_400", "fujifilm_c200", "fujifilm_provia_100f", "fujifilm_velvia_100",
          "kodak_ektachrome_100", "kodak_kodachrome_64"]


def arr(a):
    return np.asarray(a, float).tolist()


def phys(key, e):
    """materials._grain_spec_t46 的同一解析: GRAIN_PHYS, 否则名称里的感光度与档案光源。"""
    p = dict(materials.GRAIN_PHYS.get(key, {}))
    if "ei" not in p:
        m = re.search(r"(\d{2,4})", e.name)
        p["ei"] = float(m.group(1)) if m else 200.0
    if "illuminant" not in p:
        ri = str(e.stock.reference_illuminant)
        p["illuminant"] = "BB3200" if ri.startswith("BB") and float(re.sub(r"\D", "", ri) or 5500) < 4500 else "D55"
    t = p.get("thickness_um", materials.GRAIN_T_UM)
    return dict(ei=float(p["ei"]), illuminant=p["illuminant"], thickness_um=t)


def film_entry(key, e):
    st = e.stock
    return dict(name=e.name, design=e.design, native_process=e.native_process, reference_illuminant=str(st.reference_illuminant),
                log_exposure=arr(e.log_exposure), density_curves=arr(st.density_curves), dye_density=arr(st.dye_density),
                base_density=arr(st.base_density), log_sensitivity=arr(st.log_sensitivity), **phys(key, e))


def main():
    films = {k: film_entry(k, materials.emulsion(k)) for k in COLOUR}
    bw = materials.bw_film()
    films["bw_pan_generic"] = dict(film_entry("bw", bw), name="通用全色黑白 (filmlab bw_film: Portra 400 三层曲线均值 × 1.1)")
    illum = {n: arr(spectral.illuminant(n)) for n in ("D50", "D55", "D65", "BB3200", "BB2850")}
    out = dict(
        _source="filmlab/data/profiles (spektrafilm 项目, Andrea Volpato, CC BY-SA 4.0: 厂商数据册的光谱感光度、染料光谱密度、"
                "特性曲线数字化); 由 filmlab tools/export_silvergrain_data.py 导出 (T55)。本文件按 CC BY-SA 4.0 再分发。",
        _units="log_exposure: log10 lux·s (档案横轴); density_curves: 每层染料当量密度 (R G B 层); 光谱 380–780 nm / 5 nm (81)",
        wavelengths=arr(WAVELENGTHS), cie_y=arr(spectral.cmfs()[:, 1]), illuminants=illum,
        paper_endura_dye_density=arr(load_stock("kodak_portra_endura").dye_density), films=films)
    dst = ROOT / "silvergrain" / "data" / "colour_films.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    back = json.loads(dst.read_text(encoding="utf-8"))
    for k in COLOUR:
        e = materials.emulsion(k)
        assert np.array_equal(np.asarray(back["films"][k]["density_curves"]), e.stock.density_curves)
        assert np.array_equal(np.asarray(back["films"][k]["log_exposure"]), e.log_exposure)
    print("wrote", dst, dst.stat().st_size, "bytes;", len(films), "films")


if __name__ == "__main__":
    main()
