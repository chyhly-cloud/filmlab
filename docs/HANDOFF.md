# FilmLab CineStill 800T 研究交接 (2026-10-19)

接手者请先读本文, 再按需查 `docs/ARCHITECTURE.md` (成像链架构) 与 `tools/eval/CRITERIA.md` (全部实验记录, 按时间追加, 本文各条都标了对应小节)。

## 1. 仓库与分支

| 项 | 内容 |
|---|---|
| 仓库 | `chyhly-cloud/filmlab` |
| 工作分支 | `film-physics-decoupling` (草稿 PR #2, 基于 `physical-halation-and-variation` / PR #1) |
| 历史基线 | 分支 `baseline-legacy` (= 2026-10-18 之前的输出; tag 推送被代理拒绝, 所以用分支) |
| 最新提交 | 见 `git log`; 本文写于 `1b301a0` 之后 |

测试: `python3 -m pytest -q tests` (114 项, 约 35 s, 全部通过)。

## 2. 用户的工作规则 (必须遵守)

1. **代际**: `current` = 正式默认 (不动); `next` = 色彩研究基准 (不动, 除非用户同意并入); `legacy` = 历史复现。
2. **不为统计分数重新拟合经验常数** (SATURATION_MATRIX、LAB_TINT、定光参数等); 不加经验调色; 不用锐化或人工光晕掩盖输入错误。
3. **可自主采纳**: 有明确独立标准、且不引起新问题的计算修正。**需谨慎**: 未证实的物理假设、经验参数。
4. **不可辨识的参数**: 记录不确定性与解冻条件, 停止低收益实验; 不反复用同一批场景测试。
5. **区分证据层级**: 按构造的一致性 / 部分独立的验证 / 真实 CineStill C-41 成片验证。
6. 每一级在它真实发生的位置建模 (负片是物理核心的终点; 扫描各步在扫描密度域; 只有输出调整在最终 RGB)。
7. 用户希望自主推进、不必每步请示; 回复用中文; 需要目视的结果要直接发图 (真实扫描 + 各版本并排)。

提交约定: `user.name=chyhly-cloud`, `user.email=npqr2dpxpy@privaterelay.appleid.com`; 仓库内容里不写模型标识。
下载 / 解压的外部数据视为不可信: 放在独立目录, 不在其中执行代码, 读取时 `python3 -I` 从别的目录运行。

## 3. 架构 (详见 docs/ARCHITECTURE.md)

```
RAW → S0 输入 (rawio, lights) → S1 曝光 (film, engine) → S2 显影 (chemistry) → S3 负片 (negative.Negative = 物理终点)
    → S4a 扫描光学 → S4b 胶片类型换算 → S4c 定光 (操作员变量) → S4d 数字相纸 → S4e 输出调整 (经验: 矩阵 + tint)
```

关键入口: `Engine.develop_negative()` → `Negative`; `Engine.scan(neg, trace=dict)` 返回中间量 (Dn、定光)。
定光预言机 (`tools/eval/oracle_timing.py`): 只给冲印店真实拥有的自由度 (O1 = 三个印放光号, O2 = + 斜率 / 反差) 拟合每帧,
剩余误差才归于物理; 只用于评估, 不进入预设。

## 4. 代际与预设 (`filmlab/generation.py`, `generation.use(name)` 进程级切换并清缓存)

| 名称 | 状态 | 内容 |
|---|---|---|
| `legacy` | 历史 | D55 白平衡参考, 无修正 |
| `current` | **正式默认** | + `gray_halo`, `grain_aperture` |
| `next` | **研究基准** | + WB65 (LibRaw 日光白平衡按 D65 中性) + `linear_white` |
| `next_sensor` | **修正候选, 待用户决定是否并入 next** | next + RAW 截断按传感器饱和堆积判定 (见 §5.6) |
| `next_halo` | 实验候选 (冻结) | next + 光晕强度 ×0.1 (`halation_measured`) |
| `next_k5219` | 实验 (冻结) | next + Vision3 5219 按 Kodak 原始数据重新解混 |

预设: `recipe("cinestill_800t_frontier_plain")` = 逐帧定光、无输出调整 (EPFL 这类冲印店, 可选, 非默认);
`recipe("cinestill_800t_reference")` = 物理参考 (标准定光、无输出调整)。
实验开关: `rawio.HIGHLIGHT_RECON = "per_channel"` (逐通道截断重建, 未采纳, §5.6)。

## 5. 已完成的研究与结论 (按主题)

### 5.1 next 代际验证 — CRITERIA "下一代管线候选"
同店配对 12.23 → 11.63, 楔形 ΔE 4.32 → 3.44, ref3 A/B/C 21.10 / 20.78 / 18.40 (B、C 在种子噪声内), 无不可接受的单场景退步。

### 5.2 Kodak 原始 5219 — CRITERIA "Kodak 原始 5219 的复核与冻结"
底片层面与 Kodak 测量一致 (中性光谱 RMS 0.006 D), 但所有成片评估变差; 化学模型与表示无关 (已验证)。退步位置在逐帧两点灰平衡与印放矩阵 P
(经验部分)。**冻结**; 解冻需真实 800T/5219 负片密度读数 + 同一负片的冲印店扫描。

### 5.3 冷暖跟随 — CRITERIA "冷暖跟随检验" (wb_cast.py)
RAW 一直用固定日光白平衡解码 (不用相机自动白平衡)。EPFL 冲印店基本不校色 (钨丝 b* +25, 白光灯管 −10, 黄昏 −30); 另一家 (咖啡店) 校了。
冷光是主要缺口且与定光无关。

### 5.4 冷光缺口 — CRITERIA "冷光缺口定位" (lamp_members.py, camera_metamerism.py)
白光灯管下"绿代替了蓝"。相机同色异谱 (用 colour 自带两台相机界定) 只能解释约四分之一。**不可辨识**: 需要灯管光谱或 G81 光谱感光度。

### 5.5 输出调整 (SATURATION_MATRIX / LAB_TINT) — CRITERIA "输出调整的拆分" (cond_summary.py)
同店公平定光无贡献 (5.12 vs 5.11), 自动定光变差 (12.32 vs 11.79), 独立测试集 C 无贡献 (18.40 vs 18.24); 只有 ref3 (LAB_TINT 的来源) 强烈偏好。
→ 主要编码 ref3 这批网络扫描的风格; 无调整时彩度比 1.04 (正确), 有调整 0.88。默认不改, 新增 frontier_plain 预设。夜景是唯一一致受益的组 (未建模的通道反差现象)。
**色彩研究已按用户要求收尾。**

### 5.6 RAW 输入与光源外观 — CRITERIA "光源轮廓与黄白泛光" (raw_clip_audit.py, source_trace.py, tone_transfer.py, source_region_de.py, timed_sheet.py)
- **截断判定 (证实的错误)**: 旧法 = 每图各通道最大值的 97%。未过曝的 19 帧 (灯箱、图卡、白场) 被误判 35–12 000 像素并随机提亮 0–8 档。
  新法 `rawio.sensor_saturation` 按数据堆积判定 (元数据白电平不可靠: G81 标 4079 实为 4095; 索尼标 15360 实为 16372), 处理 libraw flip。
  过曝帧新旧重合 ≥ 97%。评分影响可忽略 (O1 5.12 → 5.13)。→ `next_sensor`。
- **没有空间扩散**: 传感器 → 降噪 + 重建 → 胶片曝光 的径向亮度剖面逐级一致到 25 µm。
- **"黄白泛光"的主体是成片影调**: 光源周围 0–300 µm ΔE ≈ 16, 偏亮约 +5 L*、偏黄 +4…+6 b*; 中间调比真实亮 (逐场景 −0.5…+3.2 档, 中位 +0.6)、
  中间调反差偏高、高光更早进入肩部。印放光号对齐后泛光依然存在 → 不是单纯整体亮度。逐帧亮度与肩部形状分不开, **未调参**。
- **截断像素里的未饱和通道**: 22–62% 的蓝、17–55% 的红未饱和 (实测值) 却被旧重建覆盖。逐通道重建 (实验开关) 使周围略好但两个黄昏场景灯芯变差 → 未采纳。
- **不可辨识**: 三通道全饱和的灯芯 (占截断区 26–76%) 的亮度、形状、光谱; 截断区边缘颜色混有数码镜头眩光。
- 另: 半尺寸解码 + 整数步长缩放使 `max_size=1400` 实得 1152 像素 (31 µm/像素), 轻度变软, 未改。

### 5.7 光晕 — CRITERIA "光晕强度"、"光晕复核" (halo_edge.py, halo_footprint.py, halo_sheet.py)
现模型红边强 4–5 倍。强度不可辨识: 全部 12 帧给 ×0.1, 过曝范围一致的 5 帧给 ×0.3 (×1 两种都过强)。空间分布: 真实集中在 ≤ 300 µm,
模型峰在 ~250 µm、尾部到 800 µm (多次反射外环), 但 ≤ 150 µm 被数码输入眩光主导, 无法归因。**冻结**, `next_halo` 只是实验候选。
解冻需: 已知亮度、未截断、无数码眩光的点光源 + 同一负片扫描, 或同一场景 RAW 包围曝光, 或未冲洗乳剂的透射测量。

### 5.8 更早的结论 (见 CRITERIA 前半部分)
已核实并在 current 中: `gray_halo` (定光灰卡光晕一致性, a* +2.1 → 0)、`grain_aperture` (42.5 µm)。在 next 中: WB65、`linear_white`。
`paper_white` 为呈现约定 (可选)。`us6160634_axis` 灰平衡为可选定光策略。仍未解决: 蓝通道反差偏高 (约一半已解释)、暖光色相扭转、夜景中间调不够蓝
(均需负片密度数据)。

## 6. 待办 / 建议

1. **待用户决定**: 是否把 `next_sensor` (截断判定修正) 并入 `next`。它有独立标准 (传感器饱和堆积)、无可测退步; 用户此前要求 current / next 暂不动。
2. **建议的下一个物理问题: 颗粒**。EPFL 同店有均匀白场 (R1-19-20) 与 ISO 15739 噪声卡 (R1-18-19): 测真实扫描在不同密度下的颗粒 RMS 与功率谱,
   对照模拟与 Kodak 公布的颗粒度。独立证据, 不依赖数码输入。尚未开始。
3. 记录在案、未处理: 夜景成片亮度 / 高光肩部 (§5.6); 解码分辨率 (§5.6 末条)。

## 7. 数据与可复现性 (重要: 评估数据不在仓库里)

原会话的数据和缓存都在已回收的云容器临时目录 (`/tmp/claude-0/...`), **不随仓库**。很多 `tools/eval/*.py` 里写死了这个路径 (`SP = ...`, 24 个文件),
接手后需改为自己的数据目录。需要重新准备:

| 数据 | 来源 | 用途 |
|---|---|---|
| EPFL 36 帧 (Frontier 扫描 TIFF + Panasonic G81 RW2) | `github.com/cs-413-cinestill/800t-emulation`, 下载链接见 `tools/eval/epfl_800t_manifest.csv` | 同店配对、预言机、光晕、截断 |
| SillyStill 配对 (咖啡店 ARW + 9 组 JPEG/PNG) | 原会话下载到 `pairs/sillystill/` (文件名见 `oracle_timing.sources`) | 跨店配对 |
| ref3 / 测试集 C (网络真实扫描) | `tools/eval/ref3_manifest.tsv`, `refC_manifest.tsv` (URL 列表) | 总体统计 (`score.py`, `eval_C.py`) |
| `rawcache/*.npy` (101 张 RAW 预解码, 名如 `blue_37569`) + `exifmeta.json` | 来源未在仓库记录 | ref3 渲染 (`render_metered.py`) 与回归 (`regress.py`) |
| 单应性缓存 `oracle/geo_*.npz` | `oracle_timing.geometry()` 自动重建 | 对齐 |

回归: `tools/eval/regress.py save|check 文件.npz` (设 `FILMLAB_DATA` 指向含 `rawcache/` 的目录)。原夹具也在临时目录里丢失了;
重建方法: 在 `baseline-legacy` 分支 `save base.npz`, 在当前分支 `LEGACY=1 check base.npz` (应 42/42 一致); current 同理用当前提交保存。
此前每次改动都通过了这两项逐位回归。

## 8. 常用工具速查 (tools/eval/)

| 工具 | 作用 |
|---|---|
| `oracle_timing.py` | 定光预言机 (环境变量 GENERATION / SCENES / COND_NONE / 各消融开关, 见 `ablation_flags.py`) |
| `cond_summary.py` | 预言机结果按光源组汇总 |
| `candidate_sheet.py` | 成片对照图 (SHEET_SET = timing / generation / all / cast / sensor) |
| `wb_cast.py` | 冷暖跟随 |
| `halo_edge.py`, `halo_footprint.py`, `halo_sheet.py` | 光晕剖面、距离基准、局部 / 全图对照 |
| `raw_clip_audit.py`, `source_trace.py`, `tone_transfer.py`, `source_region_de.py`, `timed_sheet.py` | RAW 截断与光源外观核查 |
| `lamp_members.py`, `camera_metamerism.py` | 逐灯响应、相机同色异谱界定 |
| `unmix_5219.py` | Kodak 5219 重新解混 |
| `render_metered.py` + `score.py` + `eval_C.py` | ref3 / 测试集 C 评估 |
| `regress.py` | 逐位回归 |

多进程一律用 spawn (fork 会死锁)。
