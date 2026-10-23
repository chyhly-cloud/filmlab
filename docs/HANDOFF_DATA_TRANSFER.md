# Claude Code外部缓存迁移

## 最新交付方式：GitHub优先

用户明确说明Claude Code运行于云端，不能依靠下载后手工上传。因此文字附件现已直接提交到[research-materials](research-materials/README.md)：研究提要PDF、三份研究Markdown及厨房样本LICENSE文字，带原件SHA。Claude更新交接分支即可读取，不需访问Codex主机的路径。

已检查主仓库`chyhly-cloud/filmlab`为**公开**仓库。用户希望继续使用同一仓库。现已创建仅供有推送权限的协作者访问的草稿Release，tag为`filmlab-cache-2026-10-23`，Release ID为`421987096`。**保持草稿，不发布含像素缓存的大包**；未来删除仓库不等于撤回已经公开的副本。明确限制再分发的厨房图像仍排除。

现有大包已切为10个≤1GiB分片，逐片及拼接SHA已核对，清单`tools/eval/handoff_release_parts.json`。**草稿目前没有任何资产，Claude尚不能下载缓存**。大包包含来自不同来源的像素缓存，不视为具有统一公开再分发许可的数据集。

网络已可访问GitHub API，代码推送和草稿创建成功，但实际`gh release upload`返回`uploads.github.com`的HTTP401（Bad credentials）。现有注入认证没有完成附件上传。已在环境配置草稿声明`FILMLAB_RELEASE_TOKEN`，目标域为`api.github.com`与`uploads.github.com`；需在环境设置安全填入仅限本仓库、Contents读写的凭据，不能把值发到聊天或提交Git。配置保存不等于凭据已提供或上传已成功。

Release实际发布且清单更新为`PUBLISHED`后，Claude可以：

```bash
python tools/eval/github_handoff_cache.py --repo OWNER/PRIVATE_DATA_REPO --directory "$HOME/filmlab-cache-parts"
```

脚本使用目标机正常`gh`认证，只获取缺失分片，核验每片及完整拼接SHA，不自动解包、不再组装一份10GB副本。当前未发布状态下会明确失败，避免无效重试。`--verify-only`可用于源机离线核对。

分片可按清单顺序拼接后解压：

```bash
mkdir -p "$HOME/filmlab-import-2026-10-23"
set -o pipefail
cat "$HOME"/filmlab-cache-parts/research-cache-private.tar.zst.part[0-9][0-9][0-9] | zstd -dc | tar -xf - -C "$HOME/filmlab-import-2026-10-23"
python tools/eval/verify_handoff_payload.py --manifest tools/eval/handoff_transfer_manifest.json --root "$HOME/filmlab-import-2026-10-23"
```

先通过下载脚本的SHA验证再解包，并保留原数据。后文小包/大包本地路径是备用导出，不是主要云端交付手段。

## 当前状态

代码/报告已在`handoff/recovered-research-2026-10-23`，但另一机器不能直接访问这里的`/workspace/...`。本次导出了迁移包，**未声称Claude已接收数据**，也没有将大包上传到GitHub/公共存储。主要传输方式为同仓库草稿Release；当前上传认证未完成。

原数据目录合计约13GB。迁移包排除两边已SHA确认一致的72个主要EPFL文件、受限Analog.Cafe图片和本地衍生区域，以及重复的DSC原件ZIP；保留1427个文件，原始载荷10.225GiB。其他补充数据可能与Claude已有数据重复，不因同名假设相同。

源机输出目录：`/workspace/filmlab-handoff/data-transfer/`。

## 先传小包

`briefs-and-records.zip`约6MB，包含：

- 564个文字/JSON研究记录的压缩包，不含图像/NPZ/NPY像素数组。
- 用户上传的2026研究提要PDF、三份10-20研究Markdown、LICENSE文本。
- 逐文件SHA256迁移清单、校验脚本及本说明。

它足以补上二手提要和详细记录，**不让依赖大数组的实验变成可重算**。原始2026实验摄影图/TIFF仍未取得。

DSC03906.ARW.zip原附件为33486819字节，可单独从原对话下载。大缓存包中也保留了`filmlab-data/user`下RAW，不重复放ZIP。

厨房TIFF及其预览/衍生图没有放入这些包。用户需在原许可允许的自身本地练习范围内使用其原附件，不公开上传或入Git，不用于训练。LICENSE文本和非图像验收事实在小包/仓库中，不代表额外使用许可。

## 大缓存包

`research-cache-private.tar.zst`包含研究NPZ/NPY、诊断输出、补充RAW、其他原有缓存及非受限附件。它是用户私有迁移包，不是公开数据集；没有把不同来源数据的许可统一改成可公开分发。

压缩完整性已通过`zstd -t`。归档大小和SHA见`tools/eval/handoff_transfer_receipt.json`，每个载荷文件SHA见`tools/eval/handoff_transfer_manifest.json`。清单中excluded列逐项说明排除理由。

同一台机器上的路径链接不是另一台主机的下载地址。只把路径贴给Claude不能完成传输；必须先实际下载/私有拷贝文件，确认目标收到后再验收。大包接近10GB，不一定适合聊天附件上传，可用用户自有私有存储或主机间传输；目前已选定同仓库草稿Release，但上传认证失败，尚无资产传输成功。

## 目标机安全导入与验证

先在新目录解开，避免覆盖Claude的原始数据/基准：

```bash
mkdir -p "$HOME/filmlab-import-2026-10-23"
set -o pipefail
zstd -dc research-cache-private.tar.zst | tar -xf - -C "$HOME/filmlab-import-2026-10-23"
python tools/eval/verify_handoff_payload.py --manifest tools/eval/handoff_transfer_manifest.json --root "$HOME/filmlab-import-2026-10-23"
```

校验应`expected=verified=1427`、errors为空、exit=0。缺失、同字节数损坏、路径越界、空清单均不能算成功。校验程序只读标准库，不执行压缩包中的外部源码。

随后按实际目标路径配置数据。**不要直接把FILMLAB_DATA切到这份补充目录**：它特意不带Claude已有的72个EPFL主文件。合并补充文件前核对同名文件SHA，保留原rawcache/配准/回归基准。同名不同内容不静默覆盖。

旧eval有些仍硬编码`/workspace/filmlab-audit`。导入到HOME并不会自动修改它们，应在有权限时保留路径映射，或显式移植eval数据根参数；不需改默认成像模型。RAW路径参与部分CRC随机种子，变路径可能改变随机结果，复现需沿用已记录种子/缓存及说明，不能只靠文件名相同宣称逐位重现。

## imagecodecs

本机安装的依赖不随Git或缓存迁移。Claude需在自己的Python虚拟环境安装与Python/NumPy兼容的`imagecodecs`以解码LZW RGB16，再运行`python -m pip check`及负片验收测试。本机历史验证为`imagecodecs==2026.10.10`，不要跨环境盲目固定不兼容版本或退回8位解码。

```bash
python -m pytest -q tests/test_negative_scan_intake.py tests/test_handoff_payload.py
```

此处依赖与工具通过不代表负片线性/光晕测量通过。current/next保持，不为搬运数据改变研究结论。
