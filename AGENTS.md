# LocalOCR 工作边界

- 本工程负责 OCR 训练、测评、本机服务、模型包和独立运行时。APP 集成、UI、命名管道与发布由导随记录工程负责。
- 只修改本仓库。可读取原项目的冻结证据，但不得写入、回退、提交或切换导随记录的代码与分支。
- 在 main 继续；未经用户明确授权不得新建分支。保护既有改动。
- 新 API、模型更新与可见行为变更按 my-canonical-check -> grill-with-docs -> openspec-propose -> openspec-apply-change -> openspec-verify-change。复用已明确确认的决策，不重问离线优先、本机接口和手动模型包更新。
- add-local-ocr-api、add-reproducible-ocr-datasets、add-selfcontained-ocr-runtime 已完成本机实现与验收，历史证据见各 change 的 verification.md。当前软件版本为 0.1.0-rc.2，模型 text-ppocrv6-1.0.0、协议保持 v1；版本事实源为 localocr/version.py，交付索引为 artifacts/delivery-0.1.0-rc.2/localocr-delivery-0.1.0-rc.2.json，本机验证见同目录 verified-handoff-0.1.0-rc.2.json。standardize-ocr-version-identities 的 rc.1 命名与全新构建证据保留在该 change verification.md 和 artifacts/localocr-delivery-0.1.0-rc.1.json。旧 v1/v2 产物和共享活动模型保留。严格校验后实现，任务仅凭完整验证证据勾选；不提前同步主规格或归档。
- 推理只用本机 CPU，默认禁止上传、云端回退和运行时下载。API 仅监听 127.0.0.1，输入仅为 Base64 图片，不抓取远程 URL。
- API 与禁止联网的 worker 分进程；保留图片/输出上限、线程限制、超时、取消和临时文件清理。不能用未验证的模拟响应冒充真实推理或 SDK 兼容性。
- 私人原图、裁片、标签、网页 HTML/语料、预测、字体、模型权重和训练产物仅存 artifacts 等忽略目录；不得提交 GitHub 或放入发布包。
- 训练使用合成数据；私人真实截图仅独立验证。按开发集选择并冻结模型，再测保留测试集。精确 Unicode、空格和组合字符不作宽度归一化。
- 颜文字实验候选 kaomoji-exp-20261011-004（历史标记 v4）已拒绝接入且未启用。必须验证颜文字改善、普通文字和负样本无新增错误，以及实际区域选择、像素证据、CPU 资源和 APP 生命周期链路后，才向 APP 工程交付接入候选。
- CUDA 已在本地实测；MPS 尚未实机验收。不得将合成或手工裁片指标声明为真实截图端到端准确率。
- 职业图标分类和训练属于 APP 工程；不得在本工程引入 FFXIV 职业目录、游戏图标、分类器模型或训练脚本。
- README 和 API 文档采用专业、可复现措辞，注明当前状态。私人截图不使用图1/图2等外部读者无法核验的引用。
- GitHub 发布范围限于已审计的源码、测试、匿名评测文档和必要许可材料；模型和数据发布需要独立许可与质量检查。不要自动开放公网监听或启用云端。
