"""色彩管线的代际 (见 tools/eval/CRITERIA.md "计算审计"、"下一代管线候选")。

  legacy   2026-10-18 之前的基线: RAW 日光白平衡按 D55 中性, 无计算修正 (分支 baseline-legacy)
  current  正式基线: legacy + 已核实且不引起退步的修正 (gray_halo, grain_aperture)
  next     下一代候选: current + 输入端的两处已核实计算修正 —— RAW 日光白平衡按 D65 中性 (LibRaw 约定)、
           非日光光源的线性曝光映射保白点 (linear_white)。扫描仪的经验常数 (饱和度矩阵、LAB_TINT、定光参数) 不变。
  next_k5219  实验: next + Vision3 5219 按 Kodak 原始测量重新解混 (stocks._kodak_raw_5219)
  next_halo   历史实验复现: next + 未标定光晕倍数（旧评估重复解码，原倍率证据已撤销）
  next_sensor 修正候选: next + RAW 截断按传感器饱和堆积判定 (rawio.sensor_saturation), 不再把未过曝画面中最亮的 3% 当作截断

用法: from filmlab import generation; generation.use("next")   (整个进程生效; 默认为 current)
WB 参考与默认修正集是进程级的: 扫描仪档案等内部会再构造引擎、并缓存与相机 RGB 有关的结果,
切换时必须一起改并清空这些缓存, 否则子引擎会悄悄沿用旧设置。
"""
from . import engine, lights, materials, rawio, stocks
from .scanning import conversion, paper

GENERATIONS = {
    "legacy": ("D55", (), {}),
    "current": ("D55", ("gray_halo", "grain_aperture"), {}),
    "next": ("D65", ("gray_halo", "grain_aperture", "linear_white"), {}),
    # T03 v2: optional uniform-depth oblique-path absorption candidate.
    "next_oblique": ("D65", ("gray_halo", "grain_aperture", "linear_white", "halo_oblique"), {}),
    # 实验: next + Vision3 5219 按 Kodak 原始测量重新解混。底片层面与 Kodak 实测一致 (独立验证), 但成片在全部评估上变差,
    # 机理未查明 (见 CRITERIA.md "Kodak 原始 5219"), 不并入 next。
    "next_k5219": ("D65", ("gray_halo", "grain_aperture", "linear_white"), {"kodak_vision3_500t": "kodak_raw"}),
    # 历史实验复现: next + 光晕强度 ×0.1。旧 halo_edge 重复解码，原 0.1/0.3 倍率证据撤销。
    # 保留数值以便历史复现；不是测得的物理强度，不并入 next。
    "next_halo": ("D65", ("gray_halo", "grain_aperture", "linear_white", "halation_measured"), {}),
}
# RAW 截断判定 (rawio.CLIP_DETECT); 未列出的代际用历史的 channel_max
CLIP_DETECT = {"next_sensor": "sensor"}
GENERATIONS["next_sensor"] = GENERATIONS["next"]
_active = "current"


def use(name: str):
    global _active
    wb, corr, revisions = GENERATIONS[name]
    lights.WB_REFERENCE = wb
    rawio.CLIP_DETECT = CLIP_DETECT.get(name, "channel_max")
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
