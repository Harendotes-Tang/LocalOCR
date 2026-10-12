# LocalOCR API v1

本机 OCR 接口，仅监听 `127.0.0.1`，默认端口 `8765`。API 与 CPU worker 分进程；无云端回退、图片上传、远程 URL 抓取或推理时下载。模型质量与接口兼容性是不同验收项；颜文字实验候选 `kaomoji-exp-20261011-004`（历史标记 v4）未启用。

## 启动与访问

使用独立 Python 3.11 环境安装 `requirements-service.txt`。开发依赖安装允许联网；服务运行不下载模型。先按 [离线模型包契约](model-bundle-v1.md) 安装并激活可信模型，再运行：

```powershell
python -m localocr --store artifacts/model-store --port 8765
```

本地令牌保存在模型库 `access-token`，不提交、不打包，不写入日志。两个端点均要求 `Authorization: Bearer <本地令牌>`。无令牌不能查询模型。仅接受本机 Host；拒绝浏览器跨域 Origin，不启用 CORS 或 HTTP 文档界面。模型库一次只允许一个服务实例；此控制不隔离已控制同一用户账户的其他程序。

## OpenAI 兼容子集

`GET /v1/models` 返回 `object=list` 和活动模型条目。模型 id 由活动模型包身份确定，客户端应查询后使用，不自行猜测。新基线身份为 `text-ppocrv6-1.0.0`；旧身份仍有效且不会自动切换。软件版本和模型身份独立，详见 [版本命名](versioning.md)。

`POST /v1/chat/completions` 仅支持 `model`、`messages`、`stream=false`（或省略 stream）。恰好一个 `role=user` 消息，content 是数组，恰好一个 `image_url` 数据 URI，可带一个说明文字部分。文字不控制识别策略；本接口不生成通用对话。示例 SDK 调用：

```python
import base64, json
from pathlib import Path
from openai import OpenAI, DefaultHttpxClient

client = OpenAI(
    base_url="http://127.0.0.1:8765/v1",
    api_key=Path("artifacts/model-store/access-token").read_text().strip(),
    http_client=DefaultHttpxClient(trust_env=False),
    max_retries=0, timeout=95,
)
model = client.models.list().data[0].id
uri = "data:image/png;base64," + base64.b64encode(Path("input.png").read_bytes()).decode("ascii")
reply = client.chat.completions.create(
    model=model, stream=False,
    messages=[{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": uri}}
    ]}],
)
result = json.loads(reply.choices[0].message.content)
client.close()
```

PNG/JPEG 的 MIME 声明必须匹配实际格式。拒绝外部 URL、动画/多帧、非法 Base64、重复 JSON 键、未知字段、stream=true、temperature/tools 等不支持选项。不支持 Responses、流式或其他完整 OpenAI 功能。SDK 支持范围以本仓库真实验收版本为准，不能推断所有 SDK 或客户端都兼容。[官方 Python SDK](https://github.com/openai/openai-python) 是兼容测试的来源。

## 客户端适用条件

软件宣称“OpenAI兼容”不能单独证明可接入本接口。它必须能把Chat Completions地址设为 `http://127.0.0.1:8765/v1`，配置本地Bearer令牌，发送一条user消息中的一张Base64 PNG/JPEG，并关闭流式及未支持选项。开启stream、自动添加system/历史消息、temperature/max_tokens/tools或image_url.detail都会被当前严格请求校验拒绝。使用Responses、远程图片地址或浏览器跨域请求也不适用。此处仅说明已有行为，不扩大参数接受范围。

返回assistant.content是OCR结果JSON字符串，客户端需解析regions；通用聊天界面可能直接显示JSON。当前已验收OpenAI Python SDK3.28.0及该测试请求形态，其他软件须实测其真实请求和显示流程。该OCR接口不执行文本提示驱动的通用问答。

## OCR 结果 `localocr.result.v1`

`assistant.content` 为 JSON 字符串，包含：

| 字段 | 含义 |
| --- | --- |
| schema_version | 固定 `localocr.result.v1` |
| model_version | 该请求固定的不可变模型包版本 |
| image.width / height | 输入图像原始像素尺寸；不擅自旋转或重新解释 EXIF 坐标 |
| regions[].text | 模型区域文本，保留空格、组合字符和精确 Unicode |
| regions[].box | 原图坐标四点区域；不代表单字坐标 |
| regions[].rect | 原图坐标的轴对齐边界 x/y/width/height |
| regions[].score | 实际识别分数；不虚构概率，不作 TSV 五位小数舍入 |
| score_semantics | 未校准分数及符号复读保守聚合的解释 |

空图合法返回空 regions。区域顺序来自 worker，不将视觉换行声明为用户手动换行。不生成虚构全文、逐字几何或校准概率。旧 TSV 入口保持原协议和文本清理行为，服务新增 JSON 模式保留真实区域文本。

## 上限与生命周期

| 项目 | 上限 |
| --- | --- |
| HTTP 编码请求体 | 28 MiB |
| Base64 解码后的源图 | 20 MiB |
| 图像像素 | 50,000,000 |
| worker JSON 输出 | 8 MiB |
| 每实例并发推理 | 1；忙时 429，不排队 |
| 请求处理截止 | 90 秒，含上传、解码和模型加载；终止/回收所属进程后清理 |

worker 使用 CPU，计算线程沿用基线限制。每次请求单独拥有图像临时目录、可终止的图像解码进程及 OCR 子进程。断连、超时或服务退出必须终止并回收 worker、删除临时图像和输出、释放槽位。错误不包含源图、令牌或私人路径。模型加载/解析失败显式失败，不能用模拟结果返回成功。

## 错误语义

错误返回 OpenAI 风格 error 对象（message/type/code），不返回伪造 OCR 成功结果。

| HTTP 状态 | 情况 |
| --- | --- |
| 400 | 非法 JSON/重复键、格式不匹配、远程 URL、动画、未知字段或 Host |
| 401 / 403 | Bearer 令牌无效 / 跨域 Origin |
| 404 | 请求 model 不是当前活动版本，或未提供的端点 |
| 413 / 415 | 图片/请求超过预算 / Content-Type 不是 application/json |
| 429 | 已有请求占用槽位，包括上传和解码阶段 |
| 499 | 请求断连；客户端通常已无法收到此响应 |
| 502 / 503 | 所属进程或输出失败 / 无有效活动模型 |
| 504 | 统一截止时间耗尽，停止并清理所属工作 |

## 证据边界

`tests/verify_live.py` 对合成文字图运行真实 SDK/CPU 请求，并检查真实 worker 的忙碌、断连、超时和退出回收。单元测试用于结构、错误输入和预算边界；受控失败注入不能作为 OCR 准确率或真实模型加载成功的证据。APP、私人截图端到端质量及新机器运行仍需独立验收；CPU自包含EXE另有本机冻结验收记录，其调用入口为CLI，不能直接当HTTP服务使用。本地 JSON/哈希校验线程采用有界输入；极端磁盘 I/O 可能延长截止后的清理时间。解码与推理均可终止。Python 网络审计钩子不是恶意原生代码的操作系统沙箱。
