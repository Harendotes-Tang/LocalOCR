# 原创 OCR 数据集与冻结比较

本工具生成原创合成训练材料和本地可分享回归包，保留精确 Unicode 标签，不进行 NFKC、宽度或空格归一化。私人截图仅在独立命令中评估；所有数据、字体、模型和逐项预测都写入忽略的 artifacts。实验候选 kaomoji-exp-20261011-004（历史标记 v4）已被拒绝接入、保持未启用，本工具仅将其作为冻结比较参照。实验编号不是软件或正式模型的 SemVer 发布版本。

## 开发准备

使用 Python 3.11，安装 requirements-datasets.txt。推理比较另外需要 requirements-service.txt 和 psutil；训练入口检查需要 Torch 2.10.0（CPU 可从官方 CPU wheel 索引安装）。这些安装和字体下载属于显式开发准备，生成和推理期间不联网。

```powershell
python -m pip install -r requirements-datasets.txt
# 真实比较另需固定服务依赖与资源统计库
python -m pip install -r requirements-service.txt psutil==7.2.2
# 仅训练入口检查或显式训练需要 Torch；按实际设备选择官方 wheel
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m tools.OcrDatasets.prepare_fonts --output artifacts/ocr-data-fonts-v1
```

font-sources.json 固定官方仓库提交、字体与 OFL 许可文件 SHA-256。字体来源为 Noto CJK、Noto Sans、Noto Sans Symbols2、Noto Sans Kannada 和 Noto Color Emoji。字体二进制不会进入回归分享包。自行提供字体清单时，必须满足许可、哈希与真实渲染检查。

Windows Pillow 未启用 RAQM；本实现使用 HarfBuzz 整形和 FreeType 栅格化。彩色字体的 CBDT 内嵌 PNG 可直接按已整形的字形 ID 和字形度量读取，以兼容缺少 PNG 支持的 FreeType 构建。emoji 需要单组合簇整形成一个字形且具有实际彩色像素。缺字、未组合、缺少颜色或裁切均排除并统计，不能把缺字方框作为正确样本。

## 生成、冻结与训练入口

创建本地排除清单，格式为 localocr.exclusions.v1 与 label_sha256 数组；每项为未经归一化的原始标签 UTF-8 SHA-256。纳入旧训练/开发/测试、用户保留文本及私人标签的指纹。该文件不导出。输出目录必须是本仓库 artifacts 下的新目录。

```powershell
python -m tools.OcrDatasets.generate --output artifacts/ocr-datasets-v1 --fonts artifacts/ocr-data-fonts-v1/fonts.json --exclusions artifacts/ocr-datasets-verification-20261011/exclusions.json --seed 731942 --groups 500
python -m tools.OcrDatasets.validate freeze --dataset artifacts/ocr-datasets-v1
python -m tools.OcrDatasets.validate validate --dataset artifacts/ocr-datasets-v1
python -m tools.OcrDatasets.report_coverage --dataset artifacts/ocr-datasets-v1 --output artifacts/coverage-audit-v1.json
python -m tools.OcrDatasets.training project --dataset artifacts/ocr-datasets-v1 --output artifacts/ocr-training-v1
python -m tools.OcrDatasets.training check --dataset artifacts/ocr-training-v1
```

五个用途分区为 train/dev/test/regression/acceptance；相同原始标签及其全部字体、明暗、尺寸、缩放、模糊和 JPEG 变体只属于一个分区。空白和原创几何负样本按原始配方分组。类别为 plain、punctuation、face、emoji、blank、graphic。无网页语料、游戏资产或遮挡文本训练标签。

训练投影仅包含 train/dev；字符表仅由这两个分区生成，不复制保留集。check 执行既有 CTC 识别网络的 CPU 前向与有限损失检查，不执行反向传播、优化器更新或写入权重。显式训练入口为 training run，参数沿用已有训练器；本轮未启动新模型训练。旧 KaomojiOcr 训练工具仍保留原有行为，发布原创数据应使用本入口。

## 本地分享与复现

```powershell
python -m tools.OcrDatasets.export --dataset artifacts/ocr-datasets-v1 --output artifacts/ocr-regression-v1
python -m tools.OcrDatasets.reproduce --package artifacts/ocr-regression-v1 --fonts artifacts/ocr-data-fonts-v1/fonts.json --output artifacts/ocr-regression-replayed-v1
```

原文、虚构记录、原创几何、生成图片和清单采用 CC0-1.0；代码采用仓库 GPL-3.0-or-later。分享包仅含回归图片、受限字段的标签/配方清单、来源和许可说明，不包含字体、模型、私人数据、私人路径或排除指纹。reproduce 直接重放公开记录，不依赖本地私人排除清单；在固定字体与已验证渲染环境中逐图核验字节哈希。跨平台图片字节一致性尚未验收。

## 冻结模型比较与使用身份

```powershell
python -m tools.OcrDatasets.evaluate --dataset artifacts/ocr-datasets-v1 --split test --candidate <冻结候选目录> --baseline-models artifacts/model-store/versions/baseline-1 --output artifacts/ocr-comparison-test-v1
```

acceptance 和 regression 可通过 --split 明确选择。可选 --private-manifest 仅接受 evaluation-only 清单，原图/裁片/标签/预测留在本地，汇总仅含匿名数量与指标。已用于此前判断的私人样本明确标记为回归证据，不能作为新的独立验收集。无法唯一确定 Unicode 的标签不计入精确成功、CER 或组合簇指标。

frozen-v4.json 固定此前批准的模型、字符表和训练报告三个权威哈希，并提供规范实验展示名。历史 schema_version=localocr.frozen-candidate.v1、identity=rejected-v4、三个哈希及旧证据目录名称不变；仅自带报告自洽的其他候选会被拒绝。新报告附带 candidate_display_name、candidate_identity_kind=experiment、candidate_enabled=false 和明确拒绝状态，不会启用或发布实验候选。private_source_count 为不同原图字节哈希数量，private_crop_count 为裁片条数。

比较执行两个真实 CPU 识别器，输入完全相同的人工指定裁片，绑定数据、模型、字符表和训练报告哈希。各类别报告整串精确数、码点 CER、符号 CER、组合簇指标、候选退化/误改、两个模型的负样本非空输出。组合簇统计仍保持码点序列，不归一化。指标不代表自动区域选择、真实截图端到端准确率或 APP 生命周期验收。

如果用 test/acceptance 结果选择规则或模型，必须记录：

```powershell
python -m tools.OcrDatasets.validate selection-use --dataset artifacts/ocr-datasets-v1 --split acceptance --reason '用于选择规则'
```

该操作保留冻结数据字节，记录绑定冻结哈希的使用状态；后续评估显示 development-regression、independent_acceptance=false。下一次独立质量声明必须另建新的验收清单。比较报告不会启用候选模型。

## 清单约定

dataset.json 使用 localocr.dataset.v1，记录种子、生成器版本/源码哈希、渲染器版本、字体提交/哈希/许可和分区文件哈希。每条记录含 id/group/kind、原始 label/label_sha256、图片路径/文件哈希、宽高、region、字体 ID、变换、原创许可、用途和实际字形证据。region 为 [x,y,width,height]，基于变换后栅格 alpha 的文字边界；JPEG 的压缩噪声不算文字字形。freeze.json 绑定清单、图片校验结果和 coverage.json；校验还检查解码像素跨分区重复。coverage.json 包含码点清单、类别数量、缺字/颜色排除及既有指纹排除数量。

像素无法总是区分规范等价的组合字符或不可见选择符；精确 Unicode 指标如实计错，不从视觉形状推断唯一标签。当前只交付 OCR 裁片数据，整卡字段定位和 APP 区域选择仍由 APP 工程负责。

本轮实际数据与匿名比较结果见 [冻结测评记录](RESULTS.md)。
