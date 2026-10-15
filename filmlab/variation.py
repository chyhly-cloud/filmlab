"""物理随机性: 真实世界里没有两卷完全一样的胶片, 也没有两次完全一样的冲洗。

这里不是给图像"加随机效果", 而是让每个物理量在它真实的公差范围内取值:

  每卷胶片 (Roll)  片基厚度、乳剂各层的吸收强度、黄滤光层密度、乳剂散射特性
                    —— 制造公差 / 批次差异
  每次冲洗 (Lab)   显影时间、温度、药液活性 (补充液老化程度)
                    —— 冲洗机的工艺控制公差
  扫描 (Scan)      自动颜色校正的力度 —— 扫描仪设置与操作员习惯
  每张底片 (Frame) 颗粒 (卤化银晶体的随机分布)、搅拌不均造成的低频密度起伏
                    —— 在 engine 中按 frame 种子生成

范围来源: C-41 工艺控制要求显影温度 37.8 ± 0.15 °C, 时间 3:15 ± 几秒 (Kodak Z-131);
冲洗厂控制条允许的速度 / 反差偏差约 ± 0.05–0.10D。乳剂与片基参数为估计范围。
seed=None 表示"标称值", 用于测试与对比。
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Variation:
    # ---- 胶卷 (制造)
    base_um: float = 130.0                 # 片基厚度 (µm), 决定光晕环的半径
    layer_absorbance: tuple = (0.55, 0.55, 0.55)   # R G B 感光层对其敏感光的吸收密度
    yellow_filter: float = 1.5             # 黄滤光层在蓝光波段的密度
    diffuse_fraction: float = 0.6          # 穿过乳剂后被散射为漫射光的比例
    forward: float = 1.5                   # 漫射光的前向程度 (cos^m)
    # ---- 冲洗
    dev_time_factor: float = 1.0           # 实际显影时间 / 标称
    dev_temp_offset: float = 0.0           # 实际温度 − 标称 (°C)
    activity_factor: float = 1.0           # 药液活性 (补充液状态)
    # ---- 每张
    frame_seed: int = 0
    agitation_density: float = 0.0         # 搅拌不均造成的低频密度起伏 (D, 1σ)
    back_reflect: float = 1.0              # 由材料决定 (remjet / 防光晕层), 不随机
    # ---- 扫描
    scan_strength: float = 0.75            # 扫描仪自动颜色校正的力度 (设置 / 操作员习惯)

    @property
    def tau_factor(self):
        return self.dev_time_factor * self.activity_factor * 2.0 ** (self.dev_temp_offset / 6.0)


NOMINAL = Variation()


def sample(roll_seed: int | None, frame: int = 0, lab_seed: int | None = None) -> Variation:
    """roll_seed: 哪一卷胶片; lab_seed: 哪一次冲洗 (默认与胶卷相同); frame: 第几张。"""
    if roll_seed is None:
        return Variation(frame_seed=frame)
    r = np.random.default_rng([roll_seed, 1])
    lab = np.random.default_rng([roll_seed if lab_seed is None else lab_seed, 2])
    return Variation(
        base_um=float(r.normal(130.0, 4.0)),
        layer_absorbance=tuple(float(v) for v in r.uniform(0.4, 0.75, 3)),
        yellow_filter=float(r.uniform(1.1, 1.9)),
        diffuse_fraction=float(r.uniform(0.45, 0.75)),
        forward=float(r.uniform(1.0, 2.5)),
        dev_time_factor=float(lab.normal(1.0, 0.01)),        # 3:15 的 ±2 秒
        dev_temp_offset=float(lab.normal(0.0, 0.10)),        # ±0.15 °C 的控制范围
        activity_factor=float(lab.normal(1.0, 0.025)),       # 补充液状态
        frame_seed=int(np.random.default_rng([roll_seed, 3, frame]).integers(1 << 31)),
        agitation_density=float(lab.uniform(0.004, 0.012)),
        scan_strength=float(lab.uniform(0.6, 0.9)),
    )
