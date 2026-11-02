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
CLIP_DETECT = {"next_sensor": "sensor", "next_lab_halo_meas": "sensor"}
# 截断高光重建方式 (rawio.HIGHLIGHT_RECON); 未列出的代际用 "prior"
HIGHLIGHT_RECON = {"next_lab_halo_meas": "per_channel"}
# 胶片曝光的推算方式 (rawio.METERING); 未列出的代际逐帧测光
# T15 研究候选: 光晕链的两项第一性原理补项 (docs/tasks/T15_HALO_PHYSICS.md)
#   next_lab_halo_ret   = next_lab_halo + 返回光斜光程吸收 (无自由参数)
#   next_lab_halo_plate = next_lab_halo_ret + 相机压片板反射 (反射率 engine.PLATE_REFLECTANCE)
GENERATIONS["next_lab_halo_ret"] = ("D65", GENERATIONS["next_lab_halo"][1] + ("halo_return",), {})
GENERATIONS["next_lab_halo_plate"] = ("D65", GENERATIONS["next_lab_halo_ret"][1] + ("halo_plate",), {})
# T16 诊断 next_lab_halo_meas = next_lab_halo_plate + 传感器饱和判定 + 逐通道重建 (未采用, 见 T16 文档)
GENERATIONS["next_lab_halo_meas"] = GENERATIONS["next_lab_halo_plate"]
# T16 研究候选 next_lab_halo_edge = next_lab_halo_plate + 蓝层在 XYZ 里非负映射 + 截断区颜色取紧邻边界的实测像素
GENERATIONS["next_lab_halo_edge"] = ("D65", GENERATIONS["next_lab_halo_plate"][1] + ("blue_xyz",), {})
# T19 研究候选 next_lab_halo_e2 = next_lab_halo + T16 (蓝层 XYZ、边界取色) + T17 (光圈缺失), 不含 T15 两项光晕补项
GENERATIONS["next_lab_halo_e2"] = ("D65", GENERATIONS["next_lab_halo"][1] + ("blue_xyz",), {})
# T20 研究候选 next_lab_halo_e2p = next_lab_halo_e2 + 相机压片板反射 (反射率 engine.PLATE_REFLECTANCE), 不含返回斜光程
GENERATIONS["next_lab_halo_e2p"] = ("D65", GENERATIONS["next_lab_halo_e2"][1] + ("halo_plate",), {})
# T20 外观选项 next_lab_halo_e2g = next_lab_halo_e2 + 拍摄端眩光 (engine.CAMERA_GLARE), 不属于胶片物理
GENERATIONS["next_lab_halo_e2g"] = ("D65", GENERATIONS["next_lab_halo_e2"][1] + ("camera_glare",), {})
RECON_COLOUR = {"next_lab_halo_edge": "edge", "next_lab_halo_e2": "edge", "next_lab_halo_e2p": "edge", "next_lab_halo_e2g": "edge"}
# T17: 手动镜头光圈缺失时, 由实测快门与 ISO + 光圈范围求截断亮度 (rawio.APERTURE_RANGE); 未列出的代际 None
APERTURE_RANGE = {"next_lab_halo_edge": (1.4, 8.0), "next_lab_halo_e2": (1.4, 8.0), "next_lab_halo_e2p": (1.4, 8.0),
                  "next_lab_halo_e2g": (1.4, 8.0)}
METERING = {"next_lab": "session", "next_lab_halo": "session", "next_lab_halo_ret": "session",
            "next_lab_halo_plate": "session", "next_lab_halo_meas": "session", "next_lab_halo_edge": "session",
            "next_lab_halo_e2": "session", "next_lab_halo_e2p": "session", "next_lab_halo_e2g": "session"}
# T22 交叉对照 (docs/tasks/T22_CROSS_AXES.md): 把"整体颜色/定光"和"灯的光晕"两条轴拆开看
#   A = next_lab: 研究线的固定冲印定光 + next 的光晕链 (无斜光程、环带取色)
#   B = next_x_e2ghalo: next 的逐帧定光 + e2g 的光晕链 (斜光程/分层、蓝层 XYZ、边界取色、光圈范围; 眩光由 CAMERA_GLARE 设定)
GENERATIONS["next_x_e2ghalo"] = ("D65", GENERATIONS["next"][1] + ("halo_oblique", "halo_layered", "blue_xyz", "camera_glare"), {})
RECON_COLOUR["next_x_e2ghalo"] = "edge"
APERTURE_RANGE["next_x_e2ghalo"] = (1.4, 8.0)
# T22 单项消融: 在 A (= next_lab) 上只加一项, 看各项对灯芯黄圈和灯的颜色的作用
_A = GENERATIONS["next_lab"][1]
GENERATIONS["t22_edge"] = ("D65", _A, {}); RECON_COLOUR["t22_edge"] = "edge"          # 截断区取边界颜色
GENERATIONS["t22_bxyz"] = ("D65", _A + ("blue_xyz",), {})                             # 蓝层 XYZ 映射
GENERATIONS["t22_ap"] = ("D65", _A, {}); APERTURE_RANGE["t22_ap"] = (1.4, 8.0)        # 光圈缺失时的截断亮度
GENERATIONS["t22_obl"] = ("D65", _A + ("halo_oblique", "halo_layered"), {})           # 光晕斜光程/分层
GENERATIONS["t22_glare"] = ("D65", _A + ("camera_glare",), {})                        # 拍摄端眩光
GENERATIONS["t22_ea"] = ("D65", _A, {}); RECON_COLOUR["t22_ea"] = "edge"; APERTURE_RANGE["t22_ea"] = (1.4, 8.0)  # 边界取色 + 光圈范围
GENERATIONS["t24_hk"] = ("D65", _A + ("blue_xyz", "halation_measured"), {})   # T24 诊断: 光晕强度 × engine.HALATION_MEASURED
GENERATIONS["t24_gl"] = ("D65", _A + ("blue_xyz", "camera_glare"), {})        # T24 诊断: 蓝层 XYZ + 拍摄端眩光
GENERATIONS["t24_hkgl"] = ("D65", _A + ("blue_xyz", "halation_measured", "camera_glare"), {})   # T24 诊断: 光晕强度 + 眩光
for _g in ("t22_edge", "t22_bxyz", "t22_ap", "t22_obl", "t22_glare", "t22_ea", "t24_hk", "t24_gl", "t24_hkgl"):
    METERING[_g] = "session"
# T26 研究候选 t26_fwd = t22_bxyz + 光晕核散射角分布指数 cos^m 改用 engine.HALO_FORWARD (docs/tasks/T26_HALO_ANGULAR.md)
GENERATIONS["t26_fwd"] = ("D65", GENERATIONS["t22_bxyz"][1] + ("halo_forward",), {})
METERING["t26_fwd"] = "session"
# T28 候选 t28_skirt = t26_fwd + 截断块档数由 RAW 裙边 / 同图近场 PSF 推算 (rawio.HIGHLIGHT_RECON = "skirt"; 回退时同 prior)
#   (docs/tasks/T28_SHAPE_BRIGHTNESS.md); 与 t26_fwd 一样须配 FWD (engine.HALO_FORWARD) 使用
GENERATIONS["t28_skirt"] = GENERATIONS["t26_fwd"]
METERING["t28_skirt"] = "session"
HIGHLIGHT_RECON["t28_skirt"] = "skirt"
# T30 候选: 截断块高出截断电平的档数不再由 EXIF 绝对亮度换算, 灯具类在 rawio.STOPS_PRIOR 内均匀取值 (rawio.HIGHLIGHT_RECON = "stops")
#   (docs/tasks/T30_UNIFORM_STOPS_PRIOR.md)。t30_stops = t22_bxyz + "stops"; t30_stops_f = t26_fwd + "stops", 须配 FWD (engine.HALO_FORWARD)
GENERATIONS["t30_stops"] = GENERATIONS["t22_bxyz"]
GENERATIONS["t30_stops_f"] = GENERATIONS["t26_fwd"]
for _g in ("t30_stops", "t30_stops_f"):
    METERING[_g] = "session"
    HIGHLIGHT_RECON[_g] = "stops"
# T30 研究基准 t30_base = t22_bxyz + 光晕核 cos^4 (engine.HALO_FORWARD) + 一致的截断灯亮度先验 (rawio "stops")
#   = t30_stops_f 在 HALO_FORWARD=4.0 下的别名; 用户目视采纳 (docs/tasks/T30_UNIFORM_STOPS_PRIOR.md)
GENERATIONS["t30_base"] = GENERATIONS["t30_stops_f"]
METERING["t30_base"] = "session"
HIGHLIGHT_RECON["t30_base"] = "stops"
RECON_COLOUR["t30_base"] = RECON_COLOUR.get("t30_stops_f", "ring")
APERTURE_RANGE["t30_base"] = APERTURE_RANGE.get("t30_stops_f")
GENERATIONS["next_sensor"] = GENERATIONS["next"]
_active = "current"


def use(name: str):
    global _active
    wb, corr, revisions = GENERATIONS[name]
    lights.WB_REFERENCE = wb
    rawio.CLIP_DETECT = CLIP_DETECT.get(name, "channel_max")
    rawio.HIGHLIGHT_RECON = HIGHLIGHT_RECON.get(name, "prior")
    rawio.RECON_COLOUR = RECON_COLOUR.get(name, "ring")
    rawio.APERTURE_RANGE = APERTURE_RANGE.get(name)
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
