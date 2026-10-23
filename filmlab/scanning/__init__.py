"""S4 扫描: 按真实扫描仪的处理顺序拆分 (docs/ARCHITECTURE.md)。

  optics        S4a 光学测量   光源 × 传感器的通道光谱响应测量底片的 T(λ) → 扫描密度       (硬件)
  conversion    S4b 胶片类型换算 去片基、窄带 → 印放密度 (P)、档案反差 / 中灰 → Dn          (软件标准)
  timing        S4c 定光       密度域的每通道偏移 / 斜率、曝光、反差、高光肩部 (= 印放光号)   (操作员变量)
  paper         S4d 数字相纸   相纸特性曲线 + 相纸染料 → 输出参照的线性 RGB                  (物理 + 估计)
  conditioning  S4e 输出调整   饱和度矩阵、Lab tint (冲印店风格)                            (经验拟合)

`filmlab.scanner.Scanner` 把这些阶段组合起来 (保持原接口)。
"""
