# 源码与依赖来源

本工程的初始基线来自 MentorRecorder。迁移时逐文件核对 SHA-256，原项目文件保持原样，记录见 source-snapshot.json。迁入内容包括文字 worker、构建/契约测试工具、颜文字工具及匿名测评报告。

源码许可为 GPL-3.0-or-later。docs/licenses/ocr 保留原基线的运行时及模型许可声明；其中 dependency-manifest.json 是原 Windows OCR 运行时的历史依赖清单，用于来源审计和补充许可证复制。它不代表 LocalOCR API 的新运行时已构建，不应作为新服务包的校验指纹。新服务需要生成自己的完整依赖与许可证清单。

模型、游戏图标、职业目录、字体、截图、网页语料、私人标签和训练产物未迁入源码仓库。职业图标分类及训练由原 APP 工程负责。训练依赖和固定打包依赖是不同环境，不应混用版本。

OpenAI 兼容子集参考官方 Chat Completions 和 Models 接口：

- https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
- https://developers.openai.com/api/reference/resources/models/methods/list

采用协议格式不构成 OpenAI 模型、SDK 全部功能或服务的背书。
