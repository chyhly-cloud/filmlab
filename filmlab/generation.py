"""色彩管线的代际 (见 tools/eval/CRITERIA.md "计算审计"、"下一代管线候选")。

  legacy   2026-10-18 之前的基线: RAW 日光白平衡按 D55 中性, 无计算修正 (分支 baseline-legacy)
  current  正式基线: legacy + 已核实且不引起退步的修正 (gray_halo, grain_aperture)
  next     下一代候选: current + 输入端的两处已核实计算修正 —— RAW 日光白平衡按 D65 中性 (LibRaw 约定)、
           非日光光源的线性曝光映射保白点 (linear_white); Vision3 5219 改用 Kodak 原始染料光谱并按 Kodak 实测
           Status M 曲线重新解混 (stocks._kodak_raw_5219)。扫描仪的经验常数 (饱和度矩阵、LAB_TINT、定光参数) 不变。

用法: from filmlab import generation; generation.use("next")   (整个进程生效; 默认为 current)
WB 参考与默认修正集是进程级的: 扫描仪档案等内部会再构造引擎、并缓存与相机 RGB 有关的结果,
切换时必须一起改并清空这些缓存, 否则子引擎会悄悄沿用旧设置。
"""
from . import engine, lights, materials, stocks
from .scanning import conversion, paper

GENERATIONS = {
    "legacy": ("D55", (), {}),
    "current": ("D55", ("gray_halo", "grain_aperture"), {}),
    "next": ("D65", ("gray_halo", "grain_aperture", "linear_white"), {"kodak_vision3_500t": "kodak_raw"}),
}
_active = "current"


def use(name: str):
    global _active
    wb, corr, revisions = GENERATIONS[name]
    lights.WB_REFERENCE = wb
    engine.DEFAULT_CORRECTIONS = corr
    stocks.REVISIONS.clear(); stocks.REVISIONS.update(revisions)
    stocks.load_stock.cache_clear()
    materials.emulsion.cache_clear()
    # 依赖相机 RGB 或默认修正的缓存
    lights.class_gamuts.cache_clear()
    engine._material_calibration.cache_clear()
    engine._reference_timing.cache_clear()
    conversion._PROFILES.clear()
    paper._PAPERS.clear()
    _active = name


def active() -> str:
    return _active
