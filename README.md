# GS-prompt-extend · 高尚 GS 提示词工作台

用 **3 个节点**打通「本地 GGUF 多模态模型」+「MiniMax H3 提示词工程」。

```text
[高尚 GS 模型加载器] ──┐
                       ├─► [高尚 GS 提示词工作台] ─► [高尚 GS 模型卸载器] ─► 继续接后面的节点
[@ 引用 H3 已连线的图] ─┘
```

## 安装（给新环境 / 别人部署）

**1. 放插件**：把整个 `GS-prompt-extend` 文件夹放进 `ComfyUI/custom_nodes/`，然后重启 ComfyUI。
缺依赖时插件不会让 ComfyUI 崩溃，3 个节点照常注册；装好依赖、放好模型之前，运行节点才会报中文提示。

**2. 装依赖**（Windows 便携版 ComfyUI，在 ComfyUI 根目录打开命令行）：

```bat
:: NVIDIA GPU 版（推荐，与作者环境一致）
python_embeded\python.exe -m pip install llama-cpp-python==0.3.40 --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
```

纯 CPU 机器：去掉上面命令里的 ` --extra-index-url ...` 一段再执行即可。

```bat
:: 验证是否装好
python_embeded\python.exe -c "from llama_cpp import Llama; print('ok')"
```

> 版本 **0.3.40 必须锁死**，不能装最新版——代码用到的 Qwen3.5/Gemma4 对话处理器对 llama.cpp 前端版本敏感。
> 判别装的是不是 GPU 版：看 `python_embeded\Lib\site-packages\llama_cpp\lib\` 里有没有 `ggml-cuda.dll`，有 = GPU 版。

**3. 放模型**到 `ComfyUI/models/LLM/`（目录不存在时插件会自动创建）。模型下载来源（HuggingFace）：

| 角色 | 主模型文件 | mmproj 文件 | 模型仓库 |
|---|---|---|---|
| 轻量试跑 | `Qwen3.5-4B-Q4_K_M.gguf` | `mmproj-BF16.gguf` | [unsloth/Qwen3.5-4B-GGUF](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF) |
| 推荐主力 | `Qwen3.5-9B-Q8_0.gguf` | `mmproj-BF16.gguf` | [unsloth/Qwen3.5-9B-GGUF](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF) |
| Gemma4 | `gemma-4-12b-it-Q8_0.gguf` | `mmproj-BF16.gguf` | [unsloth/gemma-4-12b-it-GGUF](https://huggingface.co/unsloth/gemma-4-12b-it-GGUF) |

要点：
- 这些仓库里的 mmproj 都叫 `mmproj-BF16.gguf`；**同时启用多个模型时，建议把 mmproj 改名加家族前缀**（如 `qwen3.5mmproj-BF16.gguf`、`gemma4-12b-mmproj-BF16.gguf`）以免撞名——加载器靠文件名里**含 `mmproj` 四字符**来识别，改名后仍能正常显示；
- 主模型与 mmproj 必须**成对**下载使用（9B 主模型不能配 4B 的 mmproj）；
- 加载器里「模型系列」要与文件匹配：`Qwen3.5-*` 选 `Qwen3.5-VL`，`gemma-4-*` 选 `Gemma4`。

## 节点（只有 3 个）

| 节点 | 作用 |
|---|---|
| **高尚 GS 模型加载器** | 一个节点里选 GGUF 主模型 + mmproj + 参数 |
| **高尚 GS 提示词工作台** | 一个节点框写提示词 + `@` 引用 + 一键扩写 + 看图 |
| **高尚 GS 模型卸载器** | 任意输入 → 任意输出透传；扩写完成后立刻卸载模型、释放显存/内存 |

> V3.4 变更：删掉「文本展示」（用你现有的展示节点即可）；新增「模型卸载器」——接在工作台后面，
> 提示词一生成就把 llama.cpp 放掉，不占显存内存。它还带 OUTPUT_NODE 标记，后面就算什么都不接也会执行。
> V3.1 已收敛：删掉了 V1/V2 的碎片化节点（视觉推理 / 音频推理 / 规则加载器 / 组装器 / 系统提示词 / 附加要求 / 负面要求 / 运行信息输出 / 图片输入接口）。

## 核心用法

**1. 放模型**

主模型和 mmproj 放到 `ComfyUI/models/LLM/`。

**2. `@` 引用 H3 已连线的图（重点，V3.1 已修好）**

在提示词框里输入 **`@`**，会弹出 **H3 节点上已连线（紫色）的参考素材**：

- `ref_images.ref_image_N` → `<Picture N>`
- `ref_videos.ref_video_N` → `<Video N>`
- `ref_audios.ref_audio_N` → `<Audio N>`
- `first_frame` → `<Picture 1>`
- `last_frame` → `<Picture 2>`

**灰色（没连线）的槽位不会显示**，只列出你真正启用的参考图。

选中即插入标记，**这些图会被自动从 `ComfyUI/input` 加载后喂给视觉模型** —— 不需要手动连 IMAGE 线。

**3. 一键扩写**

`模型` 输入接上加载器 → Queue Prompt，输出结构化 H3 提示词。不接模型时就是纯提示词框，原样输出。

## 任务模式

| 模式 | 说明 | 输出 |
|---|---|---|
| `full_ref` | 全参考（图+视频+音频） | 6 字段 |
| `T2VA` | 文生视频 | 4 段 |
| `I2VA` | 首帧图生视频 | 4 段 |
| `FL2VA` | 首尾帧视频 | 4 段 |
| `L2VA` | 尾帧图生视频 | 4 段 |

## 常用参数

| 参数 | 建议 |
|---|---|
| `最大边长` | 默认 1024，调小提速省显存 |
| `GPU层数` | `-1` 尽可能上 GPU，显存不够改小 |
| `KV缓存K/V类型` | 默认 F16，`q8_0` 省显存 |
| `输出think块` | 关（默认），开启保留思考过程 |

## 更新后要做的

- **改了 Python**：完全重启 ComfyUI
- **改了 `web/*.js`**：浏览器 `Ctrl+Shift+R` 硬刷新

## 备注

- 依赖 `llama-cpp-python`（当前环境 0.3.40）
- 参考素材通过 `@` 记录到节点上的 `参考素材清单` 字段，后端据此自动加载图片

## 联系方式

作者：**gs5217777**

安装、模型下载、报错或功能建议，欢迎直接联系作者，或在本仓库提 Issue。
