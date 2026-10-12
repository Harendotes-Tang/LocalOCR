# 预训练识别器微调研究

本工具使用固定官方预训练权重，分别微调 PP-OCRv6 Tiny 与 OpenOCR RepSVTR Mobile。训练采用 CUDA；导出及配对评测采用 ONNX Runtime CPU。研究模型的词表、预处理和训练架构单独声明，不作为 `localocr.bundle.v1` 安装或启用。当前执行状态与结果见[工程 README](../../README.md)。

## 输入与环境

| 模型 | 固定源码 | 原始训练权重 SHA-256 |
| --- | --- | --- |
| PP-OCRv6 Tiny | PaddleOCR `b03f46425e8ff4442b268ce449e3eef758146cd4` | `960cb4aa5276e3ac235b7f671fb8c9a7c1c1423617da0f96da66a33d0ed53f84` |
| RepSVTR Mobile | OpenOCR `1ccfc6ee6161f7133b192e16af3a9265273997ba`；权重发行 `develop0.0.1` | `fbf8fc54a1962c500cd07683408c72476c584c4c33e242e8a706fc9b59a9c0dc` |

已取得的四份官方归档/权重可用离线入口核验与准备。`inspect` 只读验证；`prepare` 仅接受尚不存在的本仓库 artifacts 目录，完成安全解压、完整源码/许可及固定摘要验证，不执行归档代码或下载。

```powershell
python -m tools.PretrainedOcr.inputs inspect `
  --openocr-archive <固定OpenOCR源码ZIP> --openocr-weight <固定RepSVTR权重> `
  --paddle-archive <固定PaddleOCR源码ZIP> --paddle-weight <固定Tiny权重> `
  --output artifacts/input-inspection
python -m tools.PretrainedOcr.inputs prepare `
  --openocr-archive <固定OpenOCR源码ZIP> --openocr-weight <固定RepSVTR权重> `
  --paddle-archive <固定PaddleOCR源码ZIP> --paddle-weight <固定Tiny权重> `
  --output artifacts/pretrained-round-20261011
```

已存在目录使用 `check-existing --openocr-receipt <研究目录/openocr-inputs.json> --paddle-receipt <研究目录/ppocr-inputs.json>`，不会覆盖。Tiny 当前研究入口固定本轮根目录；准备其他目录不会自动改变适配器或活动模型。正式源码输入已验证 418 个 OpenOCR 文件及 2,488 个 PaddleOCR 文件；实际大型复制未重复，小型准备产物已由原适配器真实验证器消费。

固定源码归档、解压文件、字典和权重均在本地核验后加载；修改输入清单不能覆盖固定归档哈希门禁。官方来源为 [Tiny 训练配置](https://github.com/PaddlePaddle/PaddleOCR/blob/b03f46425e8ff4442b268ce449e3eef758146cd4/configs/rec/PP-OCRv6/PP-OCRv6_tiny_rec.yml)、[OpenOCR 权重发行](https://github.com/Topdu/OpenOCR/releases/tag/develop0.0.1)及[微调说明](https://github.com/Topdu/OpenOCR/blob/1ccfc6ee6161f7133b192e16af3a9265273997ba/docs/finetune_rec.md)。输入获取属于开发准备；训练和评测不下载模型或读取远程图片。

| 环境 | 固定依赖 | 职责 |
| --- | --- | --- |
| OpenOCR | [requirements-pretrained-openocr.txt](../../requirements-pretrained-openocr.txt)；PyTorch 2.10.0 / CUDA 12.8 | 训练、原生 CPU 参考与 ONNX 导出 |
| Paddle CUDA | [requirements-pretrained-paddle.txt](../../requirements-pretrained-paddle.txt)；Paddle GPU 3.3.0 / CUDA 12.9 | 训练及训练后端的原生 CPU 参考 |
| Paddle CPU 导出 | [requirements-pretrained-paddle-export.txt](../../requirements-pretrained-paddle-export.txt)；Paddle CPU 3.1.1 / Paddle2ONNX 2.1.0 | 动态 ONNX 转换与数值验证 |
| 公共评测 | [requirements-service.txt](../../requirements-service.txt)及[数据依赖](../../requirements-datasets.txt) | 真实 CPU 推理、精确指标和资源计量 |

Paddle GPU 3.3.0 与 Paddle2ONNX 2.1.0 在本机 Windows 环境出现原生 DLL 导入失败，因此转换环境独立固定为 CPU 3.1.1。每个导出仍与训练后端的原生参考验证；依赖安装成功不能代替模型输出验证。各环境、缓存、源码、权重、图片和预测均保存在 `artifacts/`。源码 Apache 许可与模型资产再分发权限分别核验；研究产物不自动发布。

## 数据与识别语义

本轮 `data-v2` 包含 10,000 个正训练组、1,000 个正开发组、2,000 个正保留组，另加各分区 10% 的空白及几何负样本。正组分别生成 3 / 2 / 2 个变体，对应 31,000 / 2,100 / 4,200 张图片。标签、配方家族和跨分区像素互斥；字体、词汇原子和 emoji 原子可以共享，指标限于已声明的原创有限配方分布。

```powershell
python -m tools.PretrainedOcr.data validate --dataset artifacts/pretrained-round-20261011/data-v2
```

生成入口为 `data generate --dataset <新的独占目录> --fonts <已核验官方字体清单> --exclusions <旧标签摘要清单>`。生成器校验实际字符及组合渲染，不能可靠渲染的组整体排除。原创标签、图片和清单采用 CC0-1.0；生成器代码遵循仓库 GPL-3.0-or-later；字体遵循各自 OFL 许可并不随数据分发。

- 标签按原始 Unicode 码点处理；大小写、首尾空格、连续空格、组合字符及 ZWJ 均参与计分。
- 扩字只读取训练分区；原输出权重按字符身份重映射，CTC blank、空格及 NRTR 特殊 token 分开校验。保留集缺字按实际识别限制计错。
- CTC 长度要求为标签码点数加相邻重复次数。数据训练最大标签为 43 个码点，最大要求为 63 个时间步；Tiny 原训练固定 40 步无法覆盖，适配器改用官方推理池化，保留所有参数身份和旧权重。
- 正目标 CTC 按标签长度计量；空标签负样本按时间步计量。本机 Paddle CPU 空目标原生 CTC 探针复现进程退出；Tiny 适配器对空目标使用与 all-blank CTC 路径等价的闭式计算，NRTR 保留原 BOS/EOS 语义。此前未定位的 GPU 初探退出单独记录，不归因于该探针。
- 同一模型家族的微调前后采用相同预处理及完整图像画布。RepSVTR 使用 RGB/PIL，Tiny 使用 BGR/OpenCV；家族之间不强制混用预处理。

## 执行顺序与入口

首次研究必须依次完成输入校验、数据冻结、词表映射审计、真实正/负样本前后向探针、训练、动态导出、开发集选择冻结和新保留集首测。低层 CLI 不能替代该顺序。

```powershell
$round = 'artifacts/pretrained-round-20261011'
$openPython = "$round/env/openocr/Scripts/python.exe"
$paddlePython = "$round/env/paddle/Scripts/python.exe"
$exportPython = "$round/paddle/cpu-export-env/Scripts/python.exe"
$evalPython = 'artifacts/service-env/Scripts/python.exe'

& $openPython -m tools.PretrainedOcr.openocr_adapter --help
& $paddlePython -m tools.PretrainedOcr.paddle_adapter --help
& $evalPython -m tools.PretrainedOcr.evaluate --help
```

`openocr_adapter prepare` 接收 `--inputs-receipt`、`--dataset`、`--output` 和初始化种子；`paddle_adapter prepare` 接收本轮固定官方输入、`--dataset` 和 `--output`。准备物绑定训练数据、字典及初始化权重哈希。探针使用独占输出；探针更新后的权重不作为首轮训练初始化。

OpenOCR 训练入口：

```powershell
& $openPython -m tools.PretrainedOcr.openocr_adapter train `
  --prepared "$round/openocr/prepared" --budget "$round/gpu-budget.json" `
  --batch 16 --lr 0.0001 --seconds 2700 --steps 100000 --eval-every 500 --seed 20261011
```

Tiny 训练入口：

```powershell
& $paddlePython -m tools.PretrainedOcr.paddle_adapter train `
  --prepared "$round/paddle/prepared-v2-seeded" --dataset "$round/data-v2" `
  --output "$round/paddle/train-001" --batch-size 16 --learning-rate 0.0001 --seconds 2580
```

本轮账本已关闭，以下训练命令用于说明已验证的入口，不能再次执行本轮训练。新研究必须使用经确认的新预算、数据与输出身份，不能删除或重置本轮账本。

每次训练记录实际步骤、梯度、参数变化、负样本覆盖、开发指标、所选检查点和资源用量。初始化种子与训练随机种子分别记录。批量由真实最大训练形状的前后向探针决定；任何模拟测试均不作为 GPU 运行证明。

两框架共用一个 `gpu-budget.json`，累计租约不超过 7,200 秒，涵盖探针、GPU 训练、GPU 开发评估及保存清理。文件锁禁止并发 GPU 任务；原生退出留下未完成租约时，必须核实进程已结束并审计核销时间，不能删除账本或重置用量。磁盘余量低于 8 GiB 停止新训练写入。正常或受控异常退出在持有租约时释放 GPU 参数、优化器、临时张量和框架工作区。原生进程退出可能无法观测框架分配量，须核实实际原生 PID 已结束并按完整租约区间保守核销；不能把未观测清理写成分配量为零。本轮一项历史原生探针按 468.54 秒核销，超过其 120 秒请求值；该区间完整计入累计预算，不能宣称每个历史租约均按请求时限结束。

## CPU 导出与评测

OpenOCR `export` 绑定准备物、检查点、词表、预处理、固定源码及导出校验。Tiny 先用训练后端 `reference` 固定原生 CPU 输出，再用独立 CPU 环境 `export --native-reference <参考文件>` 转换。同一检查点的多宽度和批量输出须满足数值及最终 CTC 解码一致，并明确使用 `CPUExecutionProvider`。

```powershell
& $evalPython -m tools.PretrainedOcr.evaluate evaluate `
  --dataset "$round/data-v2" --split dev `
  --baseline-models artifacts/model-store/versions/baseline-1 `
  --candidate <原始Tiny/candidate.json> --candidate <微调Tiny/candidate.json> `
  --candidate <原始RepSVTR/candidate.json> --candidate <微调RepSVTR/candidate.json> `
  --output <新的本仓库artifacts评测目录>
```

开发选择必须绑定两个模型家族的微调前后、基线、数据、报告及完成后的预算收据。`freeze-selection` 生成不可覆盖的选择凭证；只有传入该凭证，`evaluate --split heldout --selection <凭证>` 才可首次打开新保留集。保留用量及每个模型的消费记录在推理前独占创建；失败重跑也不能伪装为首次独立评测。

冻结前复核真实子进程的成功退出、worker PID、预测文件 SHA、五项实际评测源码库存、模型与数据绑定、CPU provider/线程/资源，以及预测重算后的分类汇总、转移及门禁；仅提供元数据或修改汇总不能通过。保留首用凭据同时绑定选择 SHA 与独占 worker 消费槽，源或输入变化后拒绝执行。

旧原创合成数据仅消费已登记的 `regression` 分区，不读旧 `test` 或 `acceptance` 冒充新保留证据：

```powershell
& $evalPython -m tools.PretrainedOcr.regression evaluate `
  --dataset artifacts/ocr-datasets-v1 --selection "$round/evaluation/selection-final-v2.json" `
  --baseline-models artifacts/model-store/versions/baseline-1 `
  --candidate "$round/paddle/finetuned-export-001/candidate.json" `
  --candidate "$round/openocr/candidate-run1-best/candidate.json" `
  --output <新的本仓库artifacts回归目录>
```

公开汇总包括整串匹配、码点 CER、符号 CER、组合簇、字典覆盖、普通文字正确转错、错误改写、空白和图形非空输出、冷加载、CPU p50/p95 和内存。识别器负样本指标不等同于检测器或 APP 自动区域选择的误检率。旧已查看合成测试资料及私人真实裁片仅提供开发回归证据；不确定 Unicode 标签排除精确统计。新保留结果不得反馈到同一轮调参。

完整配对报告与训练收据由[匿名汇总入口](../Documentation/build_pretrained_summary.py)转换为白名单指标，再由[绘图入口](../Documentation/plot_pretrained_evaluation.py)生成图表。完整标签、样本预测、私人路径和权重不进入文档；质量门禁未通过时保留拒绝理由，不启用研究候选。

## 局部增强研究

本轮识别器同时作为自动候选区域的专项识别器评估，普通正文保留当前通用基线。独立整图数据和定位/融合工具位于 `tools/RegionKaomoji`，采用研究 schema，不是生产 `localocr.result.v1` 或模型包更新。候选发现覆盖、实际送模型误路由、裁片完整性及融合后的正文误改分别统计；单裁片成绩不证明这些能力。

首轮 RepSVTR 运行因 Windows 进度文件原子替换失败退出，已冻结失败状态、24,050 步日志及第 18,000 步最佳权重。后续暖启动使用独占目录与显式父 checkpoint SHA，验证父数据、词表、预处理及冻结证据，重新初始化 Adam 并记录新训练随机种子。均衡采样参数与开发选点口径分别记录；增加颜文字采样不等于更改为颜文字专用选点。

Tiny 暖启入口显式接收 `--warm-start <父best.pdparams>`、`--warm-start-receipt-sha256 <父training.json的SHA>`、`--freeze-backbone` 和 `--empty-ctc-weight 4`。父收据、源、数据及实际父 checkpoint 一并验证，优化器重新初始化；骨干参数与统计量都冻结。无优于父模型的开发结果时，`selection_outcome` 为 `parent-retained-no-development-improvement`，`best_checkpoint` 为空；最后权重不能作为已选最佳模型导出。

本轮实际四次更新为 Tiny 17,309 / 7,321 步、RepSVTR 24,050 / 13,448 步，累计 GPU 7,011.04 秒。最终两家族均保留首轮候选；新保留结果与每轮曲线见[工程 README](../../README.md)及[匿名汇总](../../docs/evaluation/pretrained-20261012/summary.json)。普通文字、负样本和颜文字质量门禁未通过，不改生产 bundle 或活动模型。
