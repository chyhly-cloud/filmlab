# T12：逐通道高光重建作为光晕强度的输入（Claude 执行；运行前写定）

T11 修正了光晕形状，但光源旁未通过：各场景所需的光晕强度相差 ×0 至 ×4，取决于截断光源在胶片上的过曝程度。

next_lab 的输入沿用默认路径（`CLIP_DETECT="channel_max"`、`HIGHLIGHT_RECON="prior"`），每个截断区按先验随机提亮若干档。
仓库中已有的逐通道重建（`rawio.reconstruct_highlights_cam`）的做法是：
- 截断像素中未饱和的通道作为实测值保留，并用光源颜色比例约束饱和通道；
- 只有三个通道全部饱和时才回退到亮度先验。

它需要 `CLIP_DETECT="sensor"` 与 `HIGHLIGHT_RECON="per_channel"`。

**候选** `next_lab_halo_pc` = next_lab_halo + 上述两个输入设置。不新增参数，不调先验。

**判定**（与 T11 相同，基准为 next_lab）：
1. 红带至少 3/4 改善，且平均 ≤ −0.03；
2. 全图 EPFL 留出 ≤ +0.05；任何 EPFL 场景变差 ≤ 1.0；
3. **光源旁 EPFL 留出 ≤ +0.05**；
4. 灰卡 < 0.5；
5. 光源局部交用户目视。

不通过则如实记录，不调参数。
