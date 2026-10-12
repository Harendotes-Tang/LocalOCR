# LocalOCR 独立运行时 v1

软件版本及规范交付名称见 [版本命名](versioning.md)；当前软件为 `0.1.0-rc.2`，结果协议仍为 v1。

Windows x64、CPU 推理的独立目录包提供 `local-ai-ocr.exe` 和 `localocr-models.exe`。完整目录须一起复制；不要求客户端安装 Python。模型包与运行时分开管理。APP 负责进程启动、取消、请求目录清理和识别结果预览；安装器及 APP 真实流程由 APP 工程验收。

## 固定 JSON 入口

```powershell
local-ai-ocr.exe --input C:\Requests\request-1\input.png --output C:\Requests\request-1\result.json --models C:\LocalOCRModels\versions\text-ppocrv6-1.0.0 --manifest C:\LocalOCRModels\versions\text-ppocrv6-1.0.0\manifest.json --format json --model-version text-ppocrv6-1.0.0
```

路径必须为本机绝对路径，拒绝 UNC、映射网络驱动器和重解析点。模型 manifest 必须属于指定版本目录，版本标识必须匹配；客户端在请求开始时固定该版本路径。JSON 输出及同名 `.partial` 必须不存在，输入和输出须分开。推理输出先写入同目录独占的 `<输出名>.runtime-claim` 临时目录，成功时原子发布到指定新文件，JSON 发布不会覆盖后来出现的文件。客户端创建独占请求目录，完成或取消后清理自己的文件。

输出采用 [API 结果契约](api-v1.md) 的 `localocr.result.v1`：原图尺寸、区域四点坐标及轴对齐边界、真实区域文本、原始有限分数、分数语义和模型版本。保留精确 Unicode、空格和组合字符，不生成单字坐标或校准概率。`--format tsv` 保留旧 TSV 清理和舍入行为。

普通源图片仅 PNG/JPEG，最大 20 MiB、50,000,000 像素；输出最大 8 MiB。CPU 线程固定为 2，默认总期限 90 秒（包括加载），`--timeout` 允许设置更短期限且不能超过 90 秒。返回码 0 表示输出成功，1 表示运行失败，2 表示 CLI 参数错误，124 表示期限到达。失败信息写入 stderr，不将日志混入结果文件。

Windows broker 在启动推理子进程前建立 kill-on-close Job。外部终止 broker 后，其后代一起终止；正常失败或超时会等待子进程并清理已取得所有权的输出及 partial。强制终止时客户端负责清理自己的请求目录及其中遗留的 runtime-claim；运行时不会删除别的调用者创建的输出或 partial。运行时不监听 HTTP 或其他端口，不上传、不启用云端回退、不运行时下载。推理核心在模型校验和依赖导入前安装 Python 审计策略，拒绝网络操作及额外进程执行。依赖查询 Windows 版本时，shell 查询被拒绝后由标准库读取本机版本信息；不改写标准库实现。broker 与模型管理器仍可启动受 Job 监督的固定推理核心。Python 审计不等同于系统级防火墙或对任意原生代码的进程隔离。

## 独立模型更新

模型管理命令参数遵循 [模型包契约](model-bundle-v1.md)，把开发命令 `python -m localocr.bundle_cli` 替换为 `localocr-models.exe`。提供 `create-baseline`、`install`、`activate`、`list`。导入和激活在固定同目录 OCR EXE 中运行真实 CPU 加载及检测、方向、识别张量探针；兼容性标识、完整性与许可声明全部检查成功后才切换活动指针。失败保留旧活动版本。哈希证明完整性，不能代替来源可信度或质量验收。

运行时不内嵌模型权重、训练依赖、字体、数据集、私人样本或被拒绝的颜文字实验候选（历史标记 v4）。兼容模型版本更新不要求重新发布 APP；架构、预处理/后处理或结果 schema 的不兼容变化需要新契约和运行时验收。

## 可复现交付

构建入口为 `scripts/build-localocr-runtime.ps1`，使用固定 CPU 构建依赖和独立虚拟环境。运行包库存记录每个文件的大小、SHA256、依赖版本、完整许可文本、源码基准提交和未提交源码哈希，以及 PyInstaller 转换输入哈希。解压后须核验文件清单；具体本地 ZIP 路径、大小、SHA256 和真实冻结 EXE 验收记录由本次 OpenSpec verification 给出。源码工作区未提交不表示对应改动已经发布。

构建前须手动准备 FlatBuffers 25.12.19 官方完整许可，因为该版本 wheel 未附全文：固定来源 `https://raw.githubusercontent.com/google/flatbuffers/7e163021e59cca4f8e1e35a7c828b5c6b7915953/LICENSE`，SHA256 `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30`，保存为 `artifacts/runtime-build-v1/upstream-notices/flatbuffers-25.12.19-LICENSE.txt`。构建脚本仅读取并核验该文件，不自动下载。

## APP 所属无损 PNG 副本

APP 在核验原图不超过 20 MiB、合法图片头和 50,000,000 像素后，可在本机解码并生成所属无损 PNG，调用时额外传 `--owned-png`。同一原图转 PNG 后可能超过原始文件上限；该模式复用固定 worker 的上限 `50,000,000 * 4 + 1,048,576 = 201,048,576` 字节。普通输入不带标志，仍保持 20 MiB；不缩图、不损失文字，JSON v1 和模型兼容性不变。

该标志是调用方对源校验和 PNG 所有权的声明，运行时不能据此证明它来自合规原图。运行时在受监督的核心子进程中再次核验字节上限、实际 PNG 内容、单帧、原图像素与所有 PNG 块的 CRC（包括 IEND），并要求终止块后无尾随数据，随后仍完整解码像素。JPEG、伪 PNG、APNG、损坏或超限输入以及 self-test 组合均拒绝。上述验证计入期限，并沿用 Job 取消和所属临时目录清理。

源码归档包含 `tools/LocalAiOcr/build_worker.py` 等构建模块、固定 `provenance/source-snapshot.json` 和 prepared 许可。无 `.git` 的解压源码按该清单验证对应源文件大小/哈希并记录 archive-snapshot 来源，不要求私有 checkout，也不取外层 APP Git。完整构建仍需单独准备固定 Python/CPU 构建依赖；产物记录实际原生库版本及哈希。
