"""色彩管线的代际 (见 tools/eval/CRITERIA.md "计算审计"、"下一代管线候选")。

  legacy   2026-10-18 之前的基线: RAW 日光白平衡按 D55 中性, 无计算修正 (分支 baseline-legacy)
  current  正式基线: legacy + 已核实且不引起退步的修正 (gray_halo, grain_aperture)
  next     下一代候选: current + 输入端的两处已核实计算修正 —— RAW 日光白平衡按 D65 中性 (LibRaw 约定)、
           非日光光源的线性曝光映射保白点 (linear_white)。扫描仪的经验常数 (饱和度矩阵、LAB_TINT、定光参数) 不变。
  next_k5219  实验: next + Vision3 5219 按 Kodak 原始测量重新解混 (stocks._kodak_raw_5219)
  next_halo   历史实验复现: next + 未标定光晕倍数（旧评估重复解码，原倍率证据已撤销）
  next_lab    研究候选 (T07): next + 胶片与数码同一份曝光 + 实测的冲印店颜色处理 + 不套他店输出调整。
              通过预设留出条件, 但留出集多次使用、无独立数据验证; 只代表 EPFL 这次拍摄与这家冲印店。不并入 next。
  next_lab_halo 研究候选 (T11): next_lab + 光晕斜程衰减按层深度 (形状修正; 强度仍依赖截断光源的重建)。
              红带与全图通过, 光源旁未通过预设条件; 用户目视后同意作为研究代际。不并入 next_lab。
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
# T07 研究候选 (见 docs/tasks/T07_LAB_COLOUR.md): 只代表 EPFL 这次拍摄与这家冲印店, 不并入 next
GENERATIONS["next_lab"] = ("D65", ("gray_halo", "grain_aperture", "linear_white", "lab_measured"), {})
# T11 研究候选 (见 docs/tasks/T11_HALO_LAYERED.md)
GENERATIONS["next_lab_halo"] = ("D65", GENERATIONS["next_lab"][1] + ("halo_oblique", "halo_layered"), {})
# RAW 截断判定 (rawio.CLIP_DETECT); 未列出的代际用历史的 channel_max
CLIP_DETECT = {"next_sensor": "sensor"}
# 胶片曝光的推算方式 (rawio.METERING); 未列出的代际逐帧测光
# T15 研究候选: 光晕链的两项第一性原理补项 (docs/tasks/T15_HALO_PHYSICS.md)
#   next_lab_halo_ret   = next_lab_halo + 返回光斜光程吸收 (无自由参数)
#   next_lab_halo_plate = next_lab_halo_ret + 相机压片板反射 (反射率 engine.PLATE_REFLECTANCE)
GENERATIONS["next_lab_halo_ret"] = ("D65", GENERATIONS["next_lab_halo"][1] + ("halo_return",), {})
GENERATIONS["next_lab_halo_plate"] = ("D65", GENERATIONS["next_lab_halo_ret"][1] + ("halo_plate",), {})
METERING = {"next_lab": "session", "next_lab_halo": "session", "next_lab_halo_ret": "session",
            "next_lab_halo_plate": "session"}
GENERATIONS["next_sensor"] = GENERATIONS["next"]
_active = "current"


def use(name: str):
    global _active
    wb, corr, revisions = GENERATIONS[name]
    lights.WB_REFERENCE = wb
    rawio.CLIP_DETECT = CLIP_DETECT.get(name, "channel_max")
    rawio.METERING = METERING.get(name, "meter")
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
