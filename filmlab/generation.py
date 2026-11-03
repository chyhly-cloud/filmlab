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
from . import engine, film, lights, materials, rawio, stocks
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
# T31 候选 t31_mask = t30_base + 有色成色剂遮罩按层结构放置 (film.MASK_PLACEMENT = "layers"; docs/tasks/T31_HALO_COLOUR.md)
GENERATIONS["t31_mask"] = GENERATIONS["t30_base"]
METERING["t31_mask"] = "session"
HIGHLIGHT_RECON["t31_mask"] = "stops"
RECON_COLOUR["t31_mask"] = RECON_COLOUR["t30_base"]
APERTURE_RANGE["t31_mask"] = APERTURE_RANGE["t30_base"]
# 遮罩放置 (film.MASK_PLACEMENT); 未列出的代际 None (原放置)
MASK_PLACEMENT = {"t31_mask": "layers"}
# T31 研究基准 t31_base = t30_base + 遮罩按层结构放置 (物理放置, 光晕略红、略弱; docs/tasks/T31_HALO_COLOUR.md)
GENERATIONS["t31_base"] = GENERATIONS["t30_base"]
METERING["t31_base"] = "session"
HIGHLIGHT_RECON["t31_base"] = "stops"
MASK_PLACEMENT["t31_base"] = "layers"
# T32 候选 t32_k5219 = t31_base + Vision3 5219 按 Kodak 原始测量 (数据手册的三种染料光谱 + 按 Kodak Status M 特性曲线
#   重新解混的分层曲线; stocks._kodak_raw_5219, 与 next_k5219 同一修订)。默认关闭, 只代表"已有物理量取数据手册值"
#   (docs/tasks/T32_SATURATED_RED.md)
GENERATIONS["t32_k5219"] = ("D65", GENERATIONS["t31_base"][1], {"kodak_vision3_500t": "kodak_raw"})
METERING["t32_k5219"] = "session"
HIGHLIGHT_RECON["t32_k5219"] = "stops"
MASK_PLACEMENT["t32_k5219"] = "layers"
# T34 研究候选 t34_phys = t31_base + 光晕链的三个无自由参数物理项 (docs/tasks/T34_HALO_PHYSICS_REFIT.md):
#   halo_oblique (T03 下行斜程) + halo_layered (T11 按层深度) + halo_return (T15 返回斜光程); 不含 halo_plate (有参数)。
#   角分布指数 cos^m 仍由 engine.HALO_FORWARD 给出 (t19_model.py 的 FWD 可覆盖), 其余 (定光、灯亮度先验、遮罩、蓝层) 同 t31_base
GENERATIONS["t34_phys"] = ("D65", GENERATIONS["t31_base"][1] + ("halo_oblique", "halo_layered", "halo_return"), {})
METERING["t34_phys"] = "session"
HIGHLIGHT_RECON["t34_phys"] = "stops"
MASK_PLACEMENT["t34_phys"] = "layers"
RECON_COLOUR["t34_phys"] = RECON_COLOUR.get("t31_base", "ring")
APERTURE_RANGE["t34_phys"] = APERTURE_RANGE.get("t31_base")
# T36 研究候选 (docs/tasks/T36_GREEN_W_PLATE.md):
#   t36_phys = t34_phys + 按层深度的 w 逐层计算 (halo_layered_perlayer: 绿层 w 由绿层之上的散射算出, 蓝层 w = 0; 无自由参数)
#   t36_base = t36_phys + 哑光压片板的朗伯漫反射 (halo_plate + halo_plate_lambert, 反射率 engine.PLATE_REFLECTANCE_LAMBERT)
GENERATIONS["t36_phys"] = ("D65", GENERATIONS["t34_phys"][1] + ("halo_layered_perlayer",), {})
GENERATIONS["t36_base"] = ("D65", GENERATIONS["t36_phys"][1] + ("halo_plate", "halo_plate_lambert"), {})
for _g in ("t36_phys", "t36_base"):
    METERING[_g] = "session"
    HIGHLIGHT_RECON[_g] = "stops"
    MASK_PLACEMENT[_g] = "layers"
    RECON_COLOUR[_g] = RECON_COLOUR.get("t31_base", "ring")
    APERTURE_RANGE[_g] = APERTURE_RANGE.get("t31_base")
# T37 研究候选 t37_base = t31_base + 截断块亮度按夜景 HDRI 实测分布抽样 (rawio.HIGHLIGHT_RECON = "measured";
#   docs/tasks/T37_MEASURED_LAMP_STOPS.md)。其余 (定光、光晕链、遮罩、取色、光圈范围) 同 t31_base
GENERATIONS["t37_base"] = GENERATIONS["t31_base"]
METERING["t37_base"] = "session"
HIGHLIGHT_RECON["t37_base"] = "measured"
MASK_PLACEMENT["t37_base"] = "layers"
RECON_COLOUR["t37_base"] = RECON_COLOUR.get("t31_base", "ring")
APERTURE_RANGE["t37_base"] = APERTURE_RANGE.get("t31_base")
# T38 研究候选 t38_base = t31_base + 截断块亮度按观测量条件化的实测分布 (rawio.HIGHLIGHT_RECON = "measured_cond";
#   docs/tasks/T38_CONDITIONAL_LAMP_PRIOR.md)。其余同 t31_base
GENERATIONS["t38_base"] = GENERATIONS["t31_base"]
METERING["t38_base"] = "session"
HIGHLIGHT_RECON["t38_base"] = "measured_cond"
MASK_PLACEMENT["t38_base"] = "layers"
RECON_COLOUR["t38_base"] = RECON_COLOUR.get("t31_base", "ring")
APERTURE_RANGE["t38_base"] = APERTURE_RANGE.get("t31_base")
# T39 研究候选 t39_base = t31_base + 截断块亮度按 块外背景 × 传感器三通道饱和比例 条件化的实测分布
#   (rawio.HIGHLIGHT_RECON = "measured_cond2"; docs/tasks/T39_NSAT_CONDITIONAL_PRIOR.md)。截断判定仍为 channel_max, 其余同 t31_base
GENERATIONS["t39_base"] = GENERATIONS["t31_base"]
METERING["t39_base"] = "session"
HIGHLIGHT_RECON["t39_base"] = "measured_cond2"
MASK_PLACEMENT["t39_base"] = "layers"
RECON_COLOUR["t39_base"] = RECON_COLOUR.get("t31_base", "ring")
APERTURE_RANGE["t39_base"] = APERTURE_RANGE.get("t31_base")
# T39 诊断 t39_corr = t39_base + 同一张图同一条件格内的分位按实测帧内组内相关抽样 (rawio "measured_cond2_corr"); 不混进 t39_base
GENERATIONS["t39_corr"] = GENERATIONS["t31_base"]
METERING["t39_corr"] = "session"
HIGHLIGHT_RECON["t39_corr"] = "measured_cond2_corr"
MASK_PLACEMENT["t39_corr"] = "layers"
RECON_COLOUR["t39_corr"] = RECON_COLOUR.get("t31_base", "ring")
APERTURE_RANGE["t39_corr"] = APERTURE_RANGE.get("t31_base")
# T35 研究候选 t35_base = t31_base + 冲印店扫描影调的整体反差按 EPFL 灰阶楔实测 (lab_contrast: 固定光号的 s =
#   engine.LAB_SCAN_CONTRAST, 支点为设计光源中灰; docs/tasks/T35_SCAN_TONE.md)。其余 (光晕链、灯亮度先验、遮罩、定光颜色) 同 t31_base
GENERATIONS["t35_base"] = ("D65", GENERATIONS["t31_base"][1] + ("lab_contrast",), {})
METERING["t35_base"] = "session"
HIGHLIGHT_RECON["t35_base"] = "stops"
MASK_PLACEMENT["t35_base"] = "layers"
RECON_COLOUR["t35_base"] = RECON_COLOUR.get("t31_base", "ring")
APERTURE_RANGE["t35_base"] = APERTURE_RANGE.get("t31_base")
# T41 (docs/tasks/T41_NEAR_FIELD_GLOW.md): 灯芯旁近场 (正面扩散) 的逐项消融与修正, 均在 t31_base 上 (定光、灯亮度先验、遮罩、蓝层同 t31_base):
#   t41_emul  = t31_base + 乳剂正面散射按 Kodak 5219 官方 MTF (emulsion_mtf: 逐层指数 PSF, 取代高斯 σ = 4 µm)
#   t41_glare = t31_base + 拍摄端眩光 (camera_glare, 参数 engine.CAMERA_GLARE; 消融用, t19_model.py 的 GLARE 可覆盖)
#   t41_base  = t31_base + emulsion_mtf + 拍摄端镜头 PSF 近场 (camera_glare + camera_glare_t41, 参数 engine.CAMERA_GLARE_T41)
GENERATIONS["t41_emul"] = ("D65", GENERATIONS["t31_base"][1] + ("emulsion_mtf",), {})
GENERATIONS["t41_glare"] = ("D65", GENERATIONS["t31_base"][1] + ("camera_glare",), {})
GENERATIONS["t41_base"] = ("D65", GENERATIONS["t31_base"][1] + ("emulsion_mtf", "camera_glare", "camera_glare_t41"), {})
for _g in ("t41_emul", "t41_glare", "t41_base"):
    METERING[_g] = "session"
    HIGHLIGHT_RECON[_g] = "stops"
    MASK_PLACEMENT[_g] = "layers"
    RECON_COLOUR[_g] = RECON_COLOUR.get("t31_base", "ring")
    APERTURE_RANGE[_g] = APERTURE_RANGE.get("t31_base")
# T42 (docs/tasks/T42_SATURATION.md): 逐级定位过饱和。t42_base = t31_base + 数字相纸输出按通道独立 (paper_channel):
#   R 通道的跨通道放大来自 S4d 相纸染料 → D50 → sRGB 的色度换算 (中性灰阶不变); 其余 (定光、光晕链、灯亮度先验、遮罩、蓝层) 同 t31_base
GENERATIONS["t42_base"] = ("D65", GENERATIONS["t31_base"][1] + ("paper_channel",), {})
METERING["t42_base"] = "session"
HIGHLIGHT_RECON["t42_base"] = "stops"
MASK_PLACEMENT["t42_base"] = "layers"
RECON_COLOUR["t42_base"] = RECON_COLOUR.get("t31_base", "ring")
APERTURE_RANGE["t42_base"] = APERTURE_RANGE.get("t31_base")
# T43 (docs/tasks/T43_HALATION_CHAIN.md): 光晕链逐项核对。片基光晕改为输运一致的表 (halo_transport: 同一叠层的已有物理量,
#   散射不当作损失, HG g 由 Kodak 5219 MTF 定; 红层 ×1.10、绿层 ×1.31, 尾部更长)。
#   t43_base = t31_base + 实测灯亮度 (rawio "measured_cond2", T39) + halo_transport; 其余 (定光、遮罩、蓝层、cos^4 不再参与光晕核) 同 t31_base
#   t43_tr   = t31_base (有效亮度先验 U[6,8]) + halo_transport (消融: 只换光晕链)
GENERATIONS["t43_base"] = ("D65", GENERATIONS["t31_base"][1] + ("halo_transport",), {})
GENERATIONS["t43_tr"] = GENERATIONS["t43_base"]
for _g, _hr in (("t43_base", "measured_cond2"), ("t43_tr", "stops")):
    METERING[_g] = "session"
    HIGHLIGHT_RECON[_g] = _hr
    MASK_PLACEMENT[_g] = "layers"
    RECON_COLOUR[_g] = RECON_COLOUR.get("t31_base", "ring")
    APERTURE_RANGE[_g] = APERTURE_RANGE.get("t31_base")
# T44 研究基准 (2026-11-03 用户目视无差别, 负责人选定; docs/tasks/T44_LAB_SHADOW_TONE.md): 冲印店的暗部传递。t44_base = t31_base + 冲印店输出的逐通道黑位 (lab_shadow,
#   engine.LAB_SHADOW_BLACK: EPFL 18 帧实测, 中间调定光逐帧放开后暗部的共同黑位); 其余 (定光、光晕链、灯亮度先验、遮罩、蓝层) 同 t31_base
GENERATIONS["t44_base"] = ("D65", GENERATIONS["t31_base"][1] + ("lab_shadow",), {})
METERING["t44_base"] = "session"
HIGHLIGHT_RECON["t44_base"] = "stops"
MASK_PLACEMENT["t44_base"] = "layers"
RECON_COLOUR["t44_base"] = RECON_COLOUR.get("t31_base", "ring")
APERTURE_RANGE["t44_base"] = APERTURE_RANGE.get("t31_base")
# T45 研究候选 (docs/tasks/T45_GRAIN.md): 颗粒按银盐物理。t45_base = t44_base + 银盐颗粒 (grain_silver: 晶体尺寸分布 +
#   光子统计显影 + 染料云 / 银粒, 参数取 materials.grain_spec 的胶片档案; 5219 按 Kodak H-1-5219 的 RMS 颗粒度) +
#   扫描仪光学点扩散 (scan_psf, engine.SCAN_PSF_UM: EPFL Frontier 扫描颗粒自相关实测; 16 µm/像素以下才生效)。其余同 t44_base
GENERATIONS["t45_base"] = ("D65", GENERATIONS["t44_base"][1] + ("grain_silver", "scan_psf"), {})
METERING["t45_base"] = "session"
HIGHLIGHT_RECON["t45_base"] = "stops"
MASK_PLACEMENT["t45_base"] = "layers"
RECON_COLOUR["t45_base"] = RECON_COLOUR.get("t31_base", "ring")
APERTURE_RANGE["t45_base"] = APERTURE_RANGE.get("t31_base")
# T46 研究候选 (docs/tasks/T46_GRAIN_GENERAL.md): 颗粒档案改为跨片种通用推导 (grain_t46: 每种胶片只用自己的感光度、设计光源、
#   晶体形态、特性曲线与冲洗工艺 + 通用常数 η、β, 留一检验; 最后一步每晶体染料 / 银产量随机, 染料云大小随晶体与工艺)。其余同 t45_base
GENERATIONS["t46_base"] = ("D65", GENERATIONS["t45_base"][1] + ("grain_t46",), {})
METERING["t46_base"] = "session"
HIGHLIGHT_RECON["t46_base"] = "stops"
MASK_PLACEMENT["t46_base"] = "layers"
RECON_COLOUR["t46_base"] = RECON_COLOUR.get("t31_base", "ring")
APERTURE_RANGE["t46_base"] = APERTURE_RANGE.get("t31_base")
GENERATIONS["next_sensor"] = GENERATIONS["next"]
# DIR 层间抑制量按代际覆盖 (materials.DIR_AMOUNT: 档案键 → Emulsion.dir_amount); 未列出的代际不覆盖 (彩色 1.0)
DIR_AMOUNT = {}
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
    film.MASK_PLACEMENT = MASK_PLACEMENT.get(name)
    engine.DEFAULT_CORRECTIONS = corr
    stocks.REVISIONS.clear(); stocks.REVISIONS.update(revisions)
    materials.DIR_AMOUNT.clear(); materials.DIR_AMOUNT.update(DIR_AMOUNT.get(name, {}))
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
