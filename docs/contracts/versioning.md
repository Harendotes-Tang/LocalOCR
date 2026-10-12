# LocalOCR 版本与身份命名

## 独立的四类身份

| 类型 | 当前规范 | 含义 |
| --- | --- | --- |
| LocalOCR 软件 | `0.1.0-rc.2` | 新候选软件交付的 SemVer；不是旧ZIP修订编号 |
| 文字基线模型包 | `text-ppocrv6-1.0.0` | 模型族加该包版本；与软件升级独立，权重保持既有基线 |
| 协议 | `/v1`、`localocr.result.v1`、`localocr.bundle.v1` | 接口与数据结构兼容边界 |
| 颜文字实验 | `kaomoji-exp-20261011-004` | 历史标记v4的冻结实验，未启用、不是正式版本 |

软件版本的权威源为 localocr/version.py。主/次/修订表示不兼容变更、兼容新增、兼容修正；rc先行版本表示候选交付，不代表APP/安装器/跨机器已验收。已交付版本的字节不原地更改，新的源码或软件变更使用新的软件版本。规则依据 [SemVer 2.0.0](https://semver.org/lang/zh-CN/)。依赖版本、上游PP-OCRv6名称和实验编号均不充当产品版本。

## 交付文件

当前生成名称来自同一权威源。`0.1.0-rc.2` 修复推理依赖导入产生额外 shell 进程的问题；结果协议与模型权重保持原值。`0.1.0-rc.1` 交付字节保留。不同软件交付使用独立目录，模型 ZIP 的同名文件按完整哈希核验。

当前文件：

- localocr-runtime-0.1.0-rc.2-windows-x64.zip
- localocr-source-0.1.0-rc.2.zip
- localocr-model-text-ppocrv6-1.0.0.zip
- localocr-delivery-0.1.0-rc.2.json

运行时完整目录、模型包、源码分别交付。模型身份改变创建新不可变包并重新进行真实CPU探针；det/cls/rec字节哈希不变不表示模型质量获得新提升。旧baseline-1等已安装身份、旧包及旧v1/v2交付保留，不重写，也不自动改变活动模型。调用方查询GET /v1/models后使用当前id；schema_version与model_version用途不同，不能相互比较。

v4的旧冻结 identity=rejected-v4、模型/字符表/训练哈希和旧证据路径用于追溯，保持原值。新的实验显示名和明确的未启用状态可用于阅读，不能据此为该候选创建可接入包或宣称emoji/普通文字改善。

## 构建与验证

使用独立固定CPU依赖、完整许可和新输出目录重新冻结；仅重命名旧ZIP无法更新EXE内版本或证明当前源码。构建库存、源码provenance和交付索引一起核验。完整解压源码不需要私有.git，需另行准备pinned CPU构建环境和随源码的build-notices；实际原生库哈希必须匹配。库存追溯到源码字节，不表示已提交源码或公开发布。

本地交付入口 `python -m tools.LocalAiOcr.package_runtime` 要求绝对路径 `--runtime`、`--inventory`、`--model-package`、`--notices`、`--output-dir`。它校验软件版本/规范名称、所有运行时与源码字节、模型身份及既有权重，再生成新ZIP和delivery；已有输出一律拒绝，不安装或切换模型。
