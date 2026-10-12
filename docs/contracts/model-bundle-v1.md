# 离线模型包 v1

本地目录包，由 `python -m localocr.bundle_cli` 管理，不提供 HTTP 管理端点。仅导入信任来源的 ONNX 模型；包中 SHA-256 表示完整性，不能证明来源可信、数字签名或模型质量。模型及许可不随源码许可自动获得再分发权限。

## 创建、安装与启用

创建基线需要在本地合法取得既有三份精确字节模型。命令只复制指定模型和仓库保留的模型许可说明，不复制数据集、截图、字体或候选权重。

```powershell
python -m localocr.bundle_cli create-baseline --models <本地基线模型目录> --output artifacts/baseline-package --version text-ppocrv6-1.0.0
python -m localocr.bundle_cli install --source artifacts/baseline-package --store artifacts/model-store
python -m localocr.bundle_cli activate --store artifacts/model-store --version text-ppocrv6-1.0.0
python -m localocr.bundle_cli list --store artifacts/model-store
```

`install` 不隐式启用。安装与启用都要求真实 CPU 加载及张量执行探针成功，探针最多 90 秒；推理 worker 禁止网络和模型下载。显式回退使用同一个 activate 命令选择已安装旧版本，重新校验和探针后才切换。加载失败保留当前 active.json。

## Manifest

包根目录仅含 `manifest.json` 及声明的平铺 ONNX 和 UTF-8 `.txt`/`.md` 许可文件，不接受子目录、符号链接/Windows reparse point、路径穿越、绝对路径、执行代码和未声明文件。

| 字段 | v1 合约 |
| --- | --- |
| schema_version | `localocr.bundle.v1` |
| version | 1–64 位 ASCII 字母数字/点/横线/下划线；首字符字母数字 |
| runtime | rapidocr=`3.10.0`, onnxruntime=`1.31.0`，必须匹配本地安装版本 |
| compatibility.architecture | `ppocr-det-cls-rec-v1` |
| compatibility.dictionary | `onnx-embedded-v1` |
| compatibility.preprocessing | `rapidocr-3.10.0-default-v1` |
| compatibility.postprocessing | `localocr-symbol-reread-v1` |
| models | 恰好 det、cls、rec，每项含 file、size、sha256 |
| licenses | 1–16 项，每项含 file、size、sha256、license（SPDX 标识）、models（覆盖的 det/cls/rec 角色） |

字典来自识别 ONNX 的内嵌元数据；声明与真实加载都必须通过，不接受运行时下载的替代字典。预处理及后处理版本绑定当前代码，不允许包携带 Python 覆盖行为。单份模型不超过 512 MiB，模型及许可总量不超过 1 GiB，单份许可不超过 1 MiB。重复键、非有限 JSON 值、未知字段、Windows 保留名或重名一律拒绝。

## 原子性与本地状态

安装在所属模型库下 staging；全部内容和真实探针通过后原子重命名为 versions/<version>。已安装版本不可原地覆盖；不同模型字节使用新版本号。启用将新指针写入所属临时文件后原子替换 active.json。请求只读取一次活动快照，旧版本保留供在途请求和回退；不自动删除旧版本。

安装锁防止同时修改。进程异常后若残留锁，先确认没有所属安装进程，再人工处理；不自动猜测或删除其他进程锁。模型库应位于本用户可控目录，外部程序擅自修改 versions 不属于不可变 API 承诺。

模型、私人图像和训练数据位于本地忽略目录。不得因本地安装成功就把包上传 GitHub或放进发布包；发布须另做许可与质量检查。本次没有推广颜文字实验候选 kaomoji-exp-20261011-004（历史标记 v4） 或声明 emoji 已识别成功。

新基线身份 text-ppocrv6-1.0.0 与软件版本独立，保持已有权重。旧baseline-1等身份、安装目录和活动指针不自动迁移；更名需创建新不可变包并重新验证，不可改写原manifest。详见 [版本命名](versioning.md)。
