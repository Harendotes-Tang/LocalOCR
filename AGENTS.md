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
- 颜文字实验候选 kaomoji-exp-20261011-004（历史标记 v4）已拒绝接入且未启用。improve-kaomoji-grammar-routing-fusion（2026-10-13）已完成本机实现、双冻结与新保留集首测（裁片 1703/2400、混合整图 76/343、记录卡片整图 25/548，正文与负样本零新增），候选 ppocrv6-tiny-exp-20261013-001 保持 promote=false、enabled=false；真实截图仅作正文零丢失探针，已知输入截图未提供，真实颜文字验证记为阻塞；证据见该 change verification.md 与 docs/evaluation/kaomoji-grammar-20261013。improve-region-candidate-localization（2026-10-13，artifacts/region-candidates-20261013，零 GPU）只改候选定位规则：结构组件剔除、分块级括号对检测、整字形密度判据与按面积消费；卡片 dev 安全完整覆盖 11→44/269，真实截图像素分支候选 0→107，atoms-mixed-v2 保留 73/343 通过门禁，cards-v3 保留 20/507 因 1 处正文相交新增错误未过门禁；已知输入真实截图 14 张首测（2026-10-14）颜文字整串 4/50（基线 2/50）、正文零丢失，真实准确率远低于合成；证据见该 change verification.md 与 docs/evaluation/region-candidates-20261013。improve-real-rendering-bracket-evidence（2026-10-15，artifacts/real-rendering-20261015，零 GPU）只加两个可选配置键：形状检测核心墨迹阈值（冻结 80，并剔除致密字形内的伪括号笔画）与外括号宽度裁决（冻结 baseline-else-raw），缺省值逐项复现上轮；known-input-001 已被逐面诊断、降级为开发探针（最终配置探针 16/50、内嵌 12/35、正文零丢失，非独立）；新保留集首测 atoms-mixed-v3 35/343 通过门禁、cards-v4 9/549 因 1 处正文相交新增错误未过；宽度裁决在合成全角真值上整串下降（混合 v1 dev 35→19）如实记录；known-input-2 已于 2026-10-15 作废（未录入、未评测）：不得要求用户录入验证内容或指定输入方式、括号宽度，真实用户会键入也会从网页粘贴。add-browser-rendering-evaluation（2026-10-15，artifacts/browser-rendering-20261015，零 GPU，不改规则）以本机 Edge 与 Windows 系统字体渲染仿制记录卡片建立浏览器真实字体渲染集 browser-cards-v1（混合括号宽度、粘贴式颜文字、三档缩放、WebP 交付），开发集当前版本 4/255、上一版本 8/255、基线 0/255，WebP 与 100% 缩放几乎全错，两个配置都因正文新增错误未过门禁；既有守卫因此拒绝双冻结，保留集 120 张按用户决定继续封存、留给下一轮改进后首测（本轮为“开发集测量完成、保留集未首测”）；该集合不是真实截图，真实截图今后只取自日常使用并以复制的评论文字为真值；证据见该 change verification.md 与 docs/evaluation/browser-rendering-20261015。必须验证颜文字改善、普通文字和负样本无新增错误，以及实际区域选择、像素证据、CPU 资源和 APP 生命周期链路后，才向 APP 工程交付接入候选。
- CUDA 已在本地实测；MPS 尚未实机验收。不得将合成或手工裁片指标声明为真实截图端到端准确率。
- 职业图标分类和训练属于 APP 工程；不得在本工程引入 FFXIV 职业目录、游戏图标、分类器模型或训练脚本。
- README 和 API 文档采用专业、可复现措辞，注明当前状态。私人截图不使用图1/图2等外部读者无法核验的引用。
- GitHub 发布范围限于已审计的源码、测试、匿名评测文档和必要许可材料；模型和数据发布需要独立许可与质量检查。不要自动开放公网监听或启用云端。
