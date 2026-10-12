# LocalOCR 架构图验收记录

本图依据源码提交 `1f7e9569d0110e925960e91ec79595ee5ce6ad2b` 绘制，包含 7 个组件、6 条关系和 13 处源码引用。图源采用 Archify，按与导随记录器相同的流程生成离线 HTML，并通过其内置 SVG 导出功能取得 README 图片。

- README 图片：[SVG](localocr.svg)
- 离线交互成品：[HTML](localocr.html)
- 可编辑图源：[JSON](localocr.architecture.json)
- 图表查看器许可：[Archify MIT](ARCHIFY-LICENSE.txt)。公开前检查移除了 HTML 与 SVG 的全部内嵌字体载荷，采用系统字体回退；原导出器的 OFL 1.1 许可文本保留。
- 交付回执、浏览器回执和截图留在本地忽略目录；本文件记录可公开的摘要。

```text
diagram_type: architecture
specification_sha256: bfc1282b240068708c4296a6d6da4502bedcd07f351a8cad9679da5680b8bf05
html_sha256: ad1dfff77a0545095c0618a82a7c48d239f59aebbf64ed05cbc537b8888f8960
svg_sha256: 97ca90447e836229ddfa7177d926fdb7ea158f3b5934f59f3be04a38234242af
source_validation: 9/9 showcase, 0 errors, 0 warnings
repository_references: 13 verified
browser_evidence: passed
visual_review: passed
```

图源 3747 字节，公开 HTML 713201 字节，SVG 56036 字节。图源保持不变，2026-10-12 重新通过源码引用及 showcase 校验。公开处理仅删除两份成品各 6 个 `@font-face` 数据块；原交付回执保留为历史证据，不代表处理后成品的字节身份。处理后的上述 HTML/SVG 均重新取得绑定其 SHA-256 的浏览器证据。

2026-10-12 自动浏览器复验覆盖 1440×900、1600×1000、1920×1080、2048×1320 的默认阅读视图，各尺寸均无横向或纵向页面溢出。另逐张查看了 1440×900 和 2048×1320 的浅色、深色截图：节点、关系标签及说明卡片无裁切，连线未穿越无关节点，页面导航与图表内容分离。处理后的 SVG 以图片形式在浏览器中成功加载，并完成 1340×760 截图检查。字体显示依赖阅读设备的系统回退，未声明其他平台字体渲染一致。

自动回执的 `visualReview: pending` 保留原样，本文件单独记录图像复核结论。浏览器默认阅读视图、SVG 导出及图片加载有实际运行证据；搜索、聚焦和其余导出格式未逐项测试。

图中请求与响应合并为一条概览关系。图片解码和 OCR worker 由服务顺序监督；独立运行时 CLI 不包含 HTTP 服务。训练与评测工具在说明卡片中记录，不表示研究模型已经启用。图表验证不等同于软件运行、识别质量或应用集成验收；本次未重新训练、运行 OCR 或验收应用工程。
