# -*- coding: utf-8 -*-
"""
GS-prompt-extend : LLM 引擎层

职责：
    用 llama-cpp-python 加载 GGUF 多模态模型（Qwen3-VL / Qwen3.5-VL / Qwen3.6-VL / Gemma4），
    并把「模型加载 / 视觉推理 / 音频推理 / 显存卸载」封装成 ComfyUI 节点。

对上层（H3 业务层）只暴露一种模型类型 GS_LLM_MODEL，业务节点无需关心底层是哪个系列。

模型放置目录： ComfyUI/models/LLM/
"""

import base64
import gc
import inspect
import io
import os
import re
import urllib.request
import wave
from collections.abc import Mapping
from dataclasses import dataclass
from functools import wraps

import numpy as np
from PIL import Image

import folder_paths
import comfy.model_management as mm

# ---------------------------------------------------------------- 依赖探测

try:
    from llama_cpp import Llama
except Exception:
    Llama = None

try:
    from llama_cpp import GGML_TYPE_Q8_0
except Exception:
    GGML_TYPE_Q8_0 = 8

try:
    from llama_cpp.llama_chat_format import Qwen3VLChatHandler
except Exception:
    Qwen3VLChatHandler = None

try:
    from llama_cpp.llama_chat_format import Qwen35ChatHandler
except Exception:
    Qwen35ChatHandler = None

try:
    from llama_cpp.llama_chat_format import Gemma4ChatHandler
except Exception:
    Gemma4ChatHandler = None

# ---------------------------------------------------------------- 常量

模型家族选项 = ["Qwen3-VL", "Qwen3.5-VL", "Qwen3.6-VL", "Gemma4"]

默认图片提示词 = ""
默认图片系统提示词 = "描述这张图,300字左右."
默认文本系统提示词 = "你是一个专业的视频提示词工程师。"
默认音频提示词 = "描述这段音频的内容。"
默认音频系统提示词 = "请分析这段音频并直接给出结果。"

默认KV缓存类型 = "默认(F16)"
Q8_0缓存类型 = "q8_0"
KV缓存类型选项 = [默认KV缓存类型, Q8_0缓存类型]

输入模式选项 = ["文本", "图片", "逐帧", "视频"]

# llama.cpp 只能加载 GGUF。放宽到 safetensors/bin/pth 会把同目录下的无关模型
# （例如 Florence-2 的 model.safetensors、pytorch_model.bin）也列进下拉框，
# 用户选中必然加载失败。因此这里严格只认 .gguf。
_MODEL_EXTS = [".gguf"]
_MMPROJ_EXTS = [".gguf"]

无模型占位 = "（请把模型放到 models/LLM）"


class AnyType(str):
    """ComfyUI 通配类型，用于「任意输入/输出」的透传节点。"""

    def __ne__(self, __value: object) -> bool:
        return False


any_type = AnyType("*")


def log(msg: str) -> None:
    print(f"[GS-prompt-extend] {msg}", flush=True)


# ---------------------------------------------------------------- 模型目录

def 确保llm目录已注册() -> None:
    """把 ComfyUI/models/LLM 注册进 folder_paths（若尚未注册），并确保目录真实存在。"""
    folder_name = "LLM"
    llm_dir = os.path.join(folder_paths.models_dir, folder_name)
    supported = set(getattr(folder_paths, "supported_pt_extensions", set()))
    llm_exts = supported | {".gguf"}

    # 目录不存在时自动创建，避免首次使用时不知道该往哪放模型
    try:
        os.makedirs(llm_dir, exist_ok=True)
    except Exception:
        pass

    try:
        if folder_name not in folder_paths.folder_names_and_paths:
            folder_paths.folder_names_and_paths[folder_name] = ([llm_dir], llm_exts)
            return

        paths, exts = folder_paths.folder_names_and_paths[folder_name]
        if llm_dir not in paths:
            paths.append(llm_dir)
        if isinstance(exts, set):
            exts.update(llm_exts)
        else:
            folder_paths.folder_names_and_paths[folder_name] = (paths, set(exts) | llm_exts)
    except Exception:
        return


def 列出llm文件() -> list[str]:
    确保llm目录已注册()
    try:
        return folder_paths.get_filename_list("LLM")
    except Exception:
        return []


def 列出主模型() -> list[str]:
    files = 列出llm文件()
    model_list = [
        f for f in files
        if "mmproj" not in f.lower() and os.path.splitext(f)[1].lower() in _MODEL_EXTS
    ]
    return model_list or [无模型占位]


def 列出mmproj() -> list[str]:
    files = 列出llm文件()
    return ["无"] + [
        f for f in files
        if "mmproj" in f.lower() and os.path.splitext(f)[1].lower() in _MMPROJ_EXTS
    ]


# ---------------------------------------------------------------- 图像工具

def 缩放图片到最大边(pil: Image.Image, 最大边长: int) -> Image.Image:
    if 最大边长 <= 0:
        return pil
    w, h = pil.size
    long_edge = max(w, h)
    if long_edge <= 最大边长:
        return pil
    scale = 最大边长 / float(long_edge)
    return pil.resize(
        (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
        resample=Image.BICUBIC,
    )


def 图片索引转base64(image_tensor, index: int, 最大边长: int) -> str:
    """把 ComfyUI IMAGE 的第 index 张转 JPEG base64。"""
    if image_tensor is None:
        return ""
    if index < 0 or index >= int(image_tensor.shape[0]):
        return ""

    img = image_tensor[index].cpu().numpy()
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    pil = Image.fromarray(img)
    pil = 缩放图片到最大边(pil, 最大边长)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def 本地图片转data_uri(image_path: str, 最大边长: int) -> str:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"找不到图片文件：{image_path}")
    with Image.open(image_path) as pil:
        if pil.mode != "RGB":
            pil = pil.convert("RGB")
        pil = 缩放图片到最大边(pil, 最大边长)
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=90)
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"


# ---------------------------------------------------------------- 文本清洗

def 清洗think块(text: str) -> str:
    """移除 <think>...</think> 思考块（Qwen 系）。"""
    if not isinstance(text, str) or not text:
        return "" if text is None else str(text)

    cleaned = text
    cleaned = re.sub(r"<think\b[^>]*>.*?</think>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if re.search(r"</think>", cleaned, flags=re.IGNORECASE):
        cleaned = re.sub(r"^.*?</think>\s*", "", cleaned, count=1, flags=re.DOTALL | re.IGNORECASE)
    cleaned = cleaned.replace("<think>", "").replace("</think>", "")
    return cleaned


def 清洗gemma4输出(text: str, 保留think块: bool) -> str:
    """清理 Gemma4 的通道控制标记，可选保留思考文本。"""
    if not isinstance(text, str) or not text:
        return "" if text is None else str(text)

    cleaned = text.replace("\r\n", "\n")

    if not 保留think块:
        cleaned = re.sub(
            r"<\|channel\>\s*(?:thought)?\s*\n?.*?<channel\|>",
            "", cleaned, flags=re.DOTALL | re.IGNORECASE,
        )
        cleaned = re.sub(r"<think\b[^>]*>.*?</think>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
        if re.search(r"</think>", cleaned, flags=re.IGNORECASE):
            cleaned = re.sub(r"^.*?</think>\s*", "", cleaned, count=1, flags=re.DOTALL | re.IGNORECASE)

    cleaned = re.sub(r"<\|channel\>\s*[\w-]*\s*\n?", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace("<channel|>", "").replace("<|think|>", "")
    cleaned = cleaned.replace("<think>", "").replace("</think>", "")
    return cleaned.strip()


def 清洗输出文本(text: str, family: str, 保留think块: bool) -> str:
    if str(family).startswith("Gemma4"):
        return 清洗gemma4输出(text, 保留think块)
    if not 保留think块:
        return 清洗think块(text)
    return text


# ---------------------------------------------------------------- 参数兼容

def llama构造参数是否可用(param_name: str) -> bool | None:
    if Llama is None:
        return None
    try:
        return param_name in inspect.signature(Llama.__init__).parameters
    except Exception:
        return None


def 解析kv缓存类型(value: str | None) -> int | None:
    if not value or value == 默认KV缓存类型:
        return None
    if value == Q8_0缓存类型:
        return GGML_TYPE_Q8_0
    raise ValueError(f"未知 KV 缓存类型：{value}")


def 规范化随机种子(seed_value):
    try:
        seed_value = int(seed_value)
    except Exception:
        return None
    return None if seed_value < 0 else seed_value


def 重置推理状态(llm) -> None:
    """清空 llama.cpp 的 KV cache / batch，避免上一轮残留影响本次推理。"""
    try:
        ctx = getattr(llm, "_ctx", None)
        if ctx is not None and hasattr(ctx, "memory_clear"):
            ctx.memory_clear(True)
    except Exception:
        pass

    try:
        mgr = getattr(llm, "_hybrid_cache_mgr", None)
        if mgr is not None and hasattr(mgr, "clear"):
            mgr.clear()
    except Exception:
        pass

    try:
        batch = getattr(llm, "_batch", None)
        if batch is not None and hasattr(batch, "reset"):
            batch.reset()
    except Exception:
        pass

    try:
        input_ids = getattr(llm, "input_ids", None)
        if input_ids is not None and hasattr(input_ids, "fill"):
            input_ids.fill(0)
    except Exception:
        pass

    try:
        reset = getattr(llm, "reset", None)
        if callable(reset):
            reset()
        elif hasattr(llm, "n_tokens"):
            llm.n_tokens = 0
    except Exception:
        pass


def 调用chat_completion(llm, *, messages, params: dict) -> dict:
    """
    兼容不同 llama-cpp-python 版本的参数名差异：
      - presence_penalty vs present_penalty
      - 老版本不认识 reasoning_budget 系列参数，需剔除
    """
    kwargs = dict(params or {})
    kwargs["messages"] = messages

    try:
        sig = inspect.signature(llm.create_chat_completion)
        allowed = sig.parameters
        has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in allowed.values())
    except Exception:
        sig = None
        allowed = {}
        has_var_kw = True

    if sig is not None:
        if "presence_penalty" in kwargs and "presence_penalty" not in allowed and "present_penalty" in allowed:
            kwargs["present_penalty"] = kwargs.pop("presence_penalty")
        if "present_penalty" in kwargs and "present_penalty" not in allowed and "presence_penalty" in allowed:
            kwargs["presence_penalty"] = kwargs.pop("present_penalty")

        reasoning_keys = {
            "reasoning_budget",
            "reasoning_start",
            "reasoning_end",
            "reasoning_budget_message",
            "reasoning_start_in_prompt",
            "reasoning_start_max_tokens",
        }
        if "reasoning_budget" not in allowed:
            kwargs = {k: v for k, v in kwargs.items() if k not in reasoning_keys}

        if not has_var_kw:
            kwargs = {k: v for k, v in kwargs.items() if k in allowed}

    return llm.create_chat_completion(**kwargs)


# ---------------------------------------------------------------- Chat Handler

def 创建qwen35聊天处理器(mmproj_path: str, *, enable_thinking: bool, preserve_thinking: bool):
    if Qwen35ChatHandler is None:
        raise RuntimeError("当前 llama-cpp-python 不支持 Qwen35ChatHandler，请更新 llama-cpp-python。")

    candidates = [
        {
            "clip_model_path": mmproj_path,
            "enable_thinking": enable_thinking,
            "add_vision_id": True,
            "preserve_thinking": preserve_thinking,
            "verbose": False,
        },
        {
            "clip_model_path": mmproj_path,
            "enable_thinking": enable_thinking,
            "preserve_thinking": preserve_thinking,
            "verbose": False,
        },
        {
            "clip_model_path": mmproj_path,
            "enable_thinking": enable_thinking,
            "add_vision_id": True,
            "verbose": False,
        },
        {
            "clip_model_path": mmproj_path,
            "enable_thinking": enable_thinking,
            "verbose": False,
        },
    ]

    last_error = None
    for kwargs in candidates:
        try:
            return Qwen35ChatHandler(**kwargs)
        except TypeError as exc:
            last_error = exc

    if last_error is not None:
        raise last_error
    raise RuntimeError("创建 Qwen35ChatHandler 失败。")


def 构建chat_handler(family: str, mmproj_path, think: bool, preserve_thinking: bool):
    if not mmproj_path:
        return None

    if family == "Qwen3-VL":
        if Qwen3VLChatHandler is None:
            raise RuntimeError("当前 llama-cpp-python 不支持 Qwen3VLChatHandler，请更新 llama-cpp-python。")
        # Qwen3 的 thinking 参数名在不同版本可能不同，逐级兜底
        try:
            return Qwen3VLChatHandler(clip_model_path=mmproj_path, force_reasoning=think, verbose=False)
        except Exception:
            try:
                return Qwen3VLChatHandler(clip_model_path=mmproj_path, use_think_prompt=think, verbose=False)
            except Exception:
                return Qwen3VLChatHandler(clip_model_path=mmproj_path, verbose=False)

    if family in ("Qwen3.5-VL", "Qwen3.6-VL"):
        return 创建qwen35聊天处理器(
            mmproj_path,
            enable_thinking=think,
            preserve_thinking=preserve_thinking,
        )

    if family == "Gemma4":
        if Gemma4ChatHandler is None:
            raise RuntimeError("当前 llama-cpp-python 不支持 Gemma4ChatHandler，请更新到带 Gemma4 支持的版本。")
        return Gemma4ChatHandler(clip_model_path=mmproj_path, enable_thinking=think, verbose=False)

    raise ValueError(f"未知模型系列：{family}")


# ---------------------------------------------------------------- 模型存储

@dataclass
class GSLLMModel:
    llm: object
    settings: dict
    chat_handler: object | None = None

    @property
    def family(self) -> str:
        return str(self.settings.get("family", ""))


class GSModelStorage:
    """单例模型槽。配置变化会自动重载；全局释放显存时会同步卸载。"""

    model: GSLLMModel | None = None

    @classmethod
    def unload(cls) -> None:
        try:
            if cls.model and getattr(cls.model.llm, "close", None):
                cls.model.llm.close()
        except Exception:
            pass
        cls.model = None
        gc.collect()
        try:
            mm.soft_empty_cache()
        except Exception:
            pass

    @classmethod
    def load(cls, config: dict) -> GSLLMModel:
        if Llama is None:
            raise RuntimeError("未检测到 llama-cpp-python（llama_cpp）。请先安装/更新该依赖。")

        if cls.model and cls.model.settings == config:
            return cls.model

        cls.unload()

        model_path = os.path.join(folder_paths.models_dir, "LLM", config["model"])
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到模型文件：{model_path}")

        mmproj = config.get("mmproj", "无")
        mmproj_path = None
        if mmproj and mmproj != "无":
            mmproj_path = os.path.join(folder_paths.models_dir, "LLM", mmproj)
            if not os.path.exists(mmproj_path):
                raise FileNotFoundError(f"找不到 mmproj 文件：{mmproj_path}")

        family = config["family"]
        think = bool(config.get("think", False))
        preserve_thinking = bool(config.get("preserve_thinking", False))

        chat_handler = 构建chat_handler(family, mmproj_path, think, preserve_thinking)

        n_ctx = int(config.get("n_ctx", 8192))
        n_gpu_layers = int(config.get("n_gpu_layers", -1))

        llama_kwargs = {
            "model_path": model_path,
            "chat_handler": chat_handler,
            "n_ctx": n_ctx,
            "n_gpu_layers": n_gpu_layers,
            "verbose": False,
        }

        if llama构造参数是否可用("ctx_checkpoints") is not False:
            llama_kwargs["ctx_checkpoints"] = 0

        # KV cache 量化
        type_k = 解析kv缓存类型(config.get("cache_type_k", 默认KV缓存类型))
        type_v = 解析kv缓存类型(config.get("cache_type_v", 默认KV缓存类型))
        wants_custom_kv = type_k is not None or type_v is not None
        supports_k = llama构造参数是否可用("type_k")
        supports_v = llama构造参数是否可用("type_v")

        if wants_custom_kv and (supports_k is False or supports_v is False):
            raise RuntimeError(
                "当前 llama-cpp-python 不支持 type_k/type_v（KV cache 量化），请更新该依赖后再使用 q8_0。"
            )
        if type_k is not None:
            llama_kwargs["type_k"] = type_k
        if type_v is not None:
            llama_kwargs["type_v"] = type_v

        # Qwen3.6 MoE 专家权重 CPU offload
        if family == "Qwen3.6-VL":
            cpu_moe = bool(config.get("cpu_moe", False))
            n_cpu_moe = int(config.get("n_cpu_moe", 0) or 0)
            supports_cpu_moe = llama构造参数是否可用("cpu_moe")
            supports_n_cpu_moe = llama构造参数是否可用("n_cpu_moe")
            wants_cpu_moe = cpu_moe
            wants_n_cpu_moe = n_cpu_moe > 0 and not cpu_moe

            if (wants_cpu_moe and supports_cpu_moe is False) or (wants_n_cpu_moe and supports_n_cpu_moe is False):
                raise RuntimeError(
                    "当前 llama-cpp-python 不支持 cpu_moe / n_cpu_moe，请更新到 0.3.37 或更高版本。"
                )
            if wants_cpu_moe:
                llama_kwargs["cpu_moe"] = True
            elif wants_n_cpu_moe:
                llama_kwargs["n_cpu_moe"] = n_cpu_moe

        log(f"正在加载模型：{config['model']}（{family}, n_ctx={n_ctx}, gpu_layers={n_gpu_layers}）")
        llm = Llama(**llama_kwargs)
        log("模型加载完成。")

        cls.model = GSLLMModel(llm=llm, settings=dict(config), chat_handler=chat_handler)
        return cls.model

    @classmethod
    def resolve(cls, model_obj) -> GSLLMModel:
        """把上游传入的模型对象解析成当前有效实例；被卸载过则自动重载。"""
        if not isinstance(model_obj, GSLLMModel):
            raise RuntimeError(
                "输入的模型对象无效。请先用「GS 模型加载器」加载模型，再把它的输出连过来。"
            )

        if cls.model is model_obj and getattr(cls.model, "llm", None) is not None:
            return cls.model

        if cls.model is not None and getattr(model_obj, "settings", None) == cls.model.settings:
            return cls.model

        if not hasattr(model_obj, "settings"):
            raise RuntimeError("输入的模型对象缺少配置信息，无法自动重载。请先运行「GS 模型加载器」。")

        log("检测到模型已被卸载或配置变化，正在自动重载……")
        return cls.load(model_obj.settings)


def 安装全局卸载挂钩() -> None:
    """让 ComfyUI 的全局「释放显存」同时 close 掉本插件的 llama.cpp 模型。"""
    try:
        if getattr(mm, "_gs_prompt_extend_unload_hook", False):
            return

        original = getattr(mm, "unload_all_models", None)
        if original is None or not callable(original):
            return

        @wraps(original)
        def wrapped_unload_all_models(*args, **kwargs):
            try:
                GSModelStorage.unload()
            except Exception:
                pass
            return original(*args, **kwargs)

        mm.unload_all_models = wrapped_unload_all_models
        mm._gs_prompt_extend_unload_hook = True
    except Exception:
        return


安装全局卸载挂钩()


# ---------------------------------------------------------------- 推理公共逻辑

def 构建帧索引(输入模式: str, 图片, 最多帧数: int) -> list[int]:
    total = int(图片.shape[0]) if 图片 is not None else 0

    if 输入模式 == "文本":
        return []
    if 输入模式 in ("图片", "逐帧", "视频") and total == 0:
        raise ValueError("未检测到图片输入。请接入 IMAGE，或把「输入模式」改为「文本」。")

    if 输入模式 == "图片":
        return [0]
    if 输入模式 == "逐帧":
        return list(range(total))
    if 输入模式 == "视频":
        if total == 1:
            return [0]
        count = min(max(int(最多帧数), 2), total)
        return np.linspace(0, total - 1, count, dtype=int).tolist()

    raise ValueError(f"未知输入模式：{输入模式}")


def 构造采样参数(
    最大生成token, 温度, top_p, top_k, seed,
    重复惩罚=1.0, 频率惩罚=0.0, 存在惩罚=0.0,
    思考预算token=-1,
) -> dict:
    params = {
        "max_tokens": int(最大生成token),
        "temperature": float(温度),
        "top_p": float(top_p),
        "top_k": int(top_k),
        "repeat_penalty": float(重复惩罚),
        "frequency_penalty": float(频率惩罚),
        "presence_penalty": float(存在惩罚),
        "seed": 规范化随机种子(seed),
        "stream": False,
        "stop": ["</s>"],
    }

    budget = int(思考预算token)
    if budget >= 0:
        params.update({
            "reasoning_budget": budget,
            "reasoning_start": "<|channel>",
            "reasoning_end": "<channel|>",
            "reasoning_start_max_tokens": None,
        })
    return params


def 收尾文本(text: str) -> str:
    return str(text).lstrip().removeprefix(": ").strip()


# ---------------------------------------------------------------- 节点：模型加载器

class GS_LLM_Loader:
    """加载 Qwen VL / Gemma4 的 GGUF 模型，输出统一的 GS_LLM_MODEL。"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "模型系列": (模型家族选项, {"default": "Qwen3.5-VL"}),
                "主模型": (列出主模型(), {"tooltip": "主模型文件（建议 .gguf），放到 ComfyUI/models/LLM/"}),
                "视觉投影mmproj": (列出mmproj(), {"default": "无", "tooltip": "多模态必须选 mmproj；纯文本可选「无」。"}),
                "上下文长度": ("INT", {"default": 8192, "min": 1024, "max": 327680, "step": 256, "tooltip": "对应 llama.cpp 的 n_ctx。提示词长就调大。"}),
                "GPU层数": ("INT", {"default": -1, "min": -1, "max": 9999, "step": 1, "tooltip": "-1=尽可能多上 GPU；0=纯 CPU。"}),
                "启用思考": ("BOOLEAN", {"default": False, "tooltip": "开启后模型会先输出思考过程再给结论，更慢但更准。"}),
            },
            "optional": {
                "保留历史think": ("BOOLEAN", {"default": False, "tooltip": "仅 Qwen3.5/3.6-VL 生效。保留历史轮次的 <think> 内容，会占用更多上下文。"}),
                "KV缓存K类型": (KV缓存类型选项, {"default": 默认KV缓存类型, "tooltip": "默认 F16；q8_0 可省显存，27B 以上模型可能提速。"}),
                "KV缓存V类型": (KV缓存类型选项, {"default": 默认KV缓存类型}),
                "MoE专家上CPU": ("BOOLEAN", {"default": False, "tooltip": "仅 Qwen3.6-VL 生效。显存不够时保命用，通常更慢。"}),
                "前N层专家上CPU": ("INT", {"default": 0, "min": 0, "max": 256, "step": 1, "tooltip": "仅 Qwen3.6-VL 生效。与「MoE专家上CPU」同时开时此项忽略。"}),
            },
        }

    RETURN_TYPES = ("GS_LLM_MODEL",)
    RETURN_NAMES = ("模型",)
    FUNCTION = "load"
    CATEGORY = "高尚GS/引擎"

    def load(
        self, 模型系列, 主模型, 视觉投影mmproj, 上下文长度, GPU层数, 启用思考,
        保留历史think=False, KV缓存K类型=默认KV缓存类型, KV缓存V类型=默认KV缓存类型,
        MoE专家上CPU=False, 前N层专家上CPU=0,
    ):
        if 主模型.startswith("（请把模型放到"):
            raise RuntimeError("未找到可用模型文件。请把 GGUF 模型放到 ComfyUI/models/LLM/ 后重启 ComfyUI。")

        config = {
            "family": 模型系列,
            "model": 主模型,
            "mmproj": 视觉投影mmproj,
            "think": bool(启用思考),
            "preserve_thinking": bool(保留历史think),
            "n_ctx": int(上下文长度),
            "n_gpu_layers": int(GPU层数),
            "cache_type_k": KV缓存K类型,
            "cache_type_v": KV缓存V类型,
            "cpu_moe": bool(MoE专家上CPU),
            "n_cpu_moe": int(前N层专家上CPU),
        }
        return (GSModelStorage.load(config),)


# ---------------------------------------------------------------- 节点：视觉 / 文本推理

class GS_LLM_Vision:
    """通用推理：文本对话、单图、逐帧、视频抽帧。"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "模型": ("GS_LLM_MODEL",),
                "输入模式": (输入模式选项, {"default": "文本", "tooltip": "文本=纯文字；图片=只读第1张；逐帧=一张张分别推理；视频=抽帧后一次性推理。"}),
                "提示词": ("STRING", {"default": 默认图片提示词, "multiline": True}),
                "系统提示词": ("STRING", {"default": 默认图片系统提示词, "multiline": True}),
            },
            "optional": {
                "图片": ("IMAGE",),
                "最多帧数": ("INT", {"default": 24, "min": 2, "max": 1024, "step": 1, "tooltip": "「视频」模式下均匀抽取的帧数。"}),
                "最大边长": ("INT", {"default": 1024, "min": 128, "max": 16384, "step": 64, "tooltip": "图片按最长边缩放，调小可提速。"}),
                "最大生成token": ("INT", {"default": 2048, "min": 20, "max": 32768, "step": 1}),
                "温度": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 20, "min": 0, "max": 200, "step": 1}),
                "重复惩罚": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "step": 1, "control_after_generate": True}),
                "输出think块": ("BOOLEAN", {"default": False, "tooltip": "关闭时只保留最终答案，自动清理思考块。"}),
                "思考预算token": ("INT", {"default": -1, "min": -1, "max": 8192, "step": 1, "tooltip": "-1=不限制；0=进入思考后立即结束；>0=限制思考 token 数。"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("文本",)
    FUNCTION = "run"
    CATEGORY = "高尚GS/引擎"

    def run(
        self, 模型, 输入模式, 提示词, 系统提示词,
        图片=None, 最多帧数=24, 最大边长=1024, 最大生成token=2048,
        温度=0.7, top_p=0.9, top_k=20, 重复惩罚=1.0, seed=0,
        输出think块=False, 思考预算token=-1,
    ):
        model = GSModelStorage.resolve(模型)
        llm = model.llm
        family = model.family

        if 输入模式 in ("图片", "逐帧", "视频") and model.chat_handler is None:
            raise RuntimeError(
                "当前模型未加载 mmproj，无法进行图像推理。请在「GS 模型加载器」里选择对应的 mmproj。"
            )

        messages = []
        system_text = (系统提示词 or "").strip()
        if 输入模式 == "文本":
            if not system_text or system_text == 默认图片系统提示词:
                system_text = 默认文本系统提示词
        elif 输入模式 == "视频" and system_text:
            system_text = "请将输入的图片序列当做视频而不是静态帧序列, " + system_text
        if system_text:
            messages.append({"role": "system", "content": system_text})

        frame_indices = 构建帧索引(输入模式, 图片, 最多帧数)

        params = 构造采样参数(
            最大生成token, 温度, top_p, top_k, seed,
            重复惩罚=重复惩罚,
            思考预算token=思考预算token,
        )

        prompt_text = (提示词 or "").strip()

        # ---- 文本模式
        if 输入模式 == "文本":
            if not prompt_text:
                raise ValueError("「文本」模式下，提示词不能为空。")
            messages.append({"role": "user", "content": prompt_text})
            重置推理状态(llm)
            out = 调用chat_completion(llm, messages=messages, params=params)
            try:
                text = out["choices"][0]["message"]["content"]
            except Exception:
                text = str(out)

        # ---- 逐帧模式
        elif 输入模式 == "逐帧":
            user_content = [{"type": "text", "text": prompt_text}, {"type": "image_url", "image_url": {"url": ""}}]
            messages.append({"role": "user", "content": user_content})

            out_parts = []
            for idx, frame_index in enumerate(frame_indices):
                if mm.processing_interrupted():
                    raise mm.InterruptProcessingException()
                img_b64 = 图片索引转base64(图片, frame_index, int(最大边长))
                if not img_b64:
                    continue
                user_content[1]["image_url"]["url"] = f"data:image/jpeg;base64,{img_b64}"
                重置推理状态(llm)
                out = 调用chat_completion(llm, messages=messages, params=params)
                try:
                    part = out["choices"][0]["message"]["content"]
                except Exception:
                    part = str(out)
                part = 清洗输出文本(part, family, bool(输出think块))
                out_parts.append(
                    f"====== 第{idx + 1}帧 ======\n{part}".strip() if len(frame_indices) > 1 else str(part).strip()
                )
            text = "\n\n".join([p for p in out_parts if p])

        # ---- 图片 / 视频模式
        else:
            user_content = [{"type": "text", "text": prompt_text}]
            for frame_index in frame_indices:
                img_b64 = 图片索引转base64(图片, frame_index, int(最大边长))
                if img_b64:
                    user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}})
            messages.append({"role": "user", "content": user_content})
            重置推理状态(llm)
            out = 调用chat_completion(llm, messages=messages, params=params)
            try:
                text = out["choices"][0]["message"]["content"]
            except Exception:
                text = str(out)

        text = 清洗输出文本(text, family, bool(输出think块))

        if mm.processing_interrupted():
            raise mm.InterruptProcessingException()

        return (收尾文本(text),)


# ---------------------------------------------------------------- 音频工具（Gemma4）

def 读取音频字段(audio_data, key: str, default=None):
    if audio_data is None:
        return default
    if isinstance(audio_data, Mapping):
        return audio_data.get(key, default)

    getter = getattr(audio_data, "get", None)
    if callable(getter):
        try:
            return getter(key, default)
        except TypeError:
            try:
                return getter(key)
            except Exception:
                pass
        except Exception:
            pass

    if hasattr(audio_data, key):
        return getattr(audio_data, key)
    try:
        return audio_data[key]
    except Exception:
        return default


def comfy音频转wav_base64(audio_data) -> str:
    waveform = 读取音频字段(audio_data, "waveform")
    sample_rate = 读取音频字段(audio_data, "sample_rate")
    if sample_rate is None:
        sample_rate = 读取音频字段(audio_data, "sampler_rate")
    if waveform is None or sample_rate is None:
        raise ValueError("ComfyUI 音频输入缺少 waveform 或 sample_rate。")

    if hasattr(waveform, "detach"):
        waveform = waveform.detach()
    if hasattr(waveform, "cpu"):
        waveform = waveform.cpu()
    wav_np = waveform.numpy() if hasattr(waveform, "numpy") else np.asarray(waveform)

    if wav_np.ndim == 3:
        wav_np = wav_np[0]
    elif wav_np.ndim == 1:
        wav_np = wav_np[np.newaxis, :]
    if wav_np.ndim != 2:
        raise ValueError(f"ComfyUI 音频 waveform 维度不受支持：{wav_np.shape}")

    wav_np = np.asarray(wav_np, dtype=np.float32)
    wav_np = np.nan_to_num(wav_np, nan=0.0, posinf=1.0, neginf=-1.0)
    wav_np = np.clip(wav_np, -1.0, 1.0)
    pcm16 = (wav_np.T * 32767.0).astype(np.int16, copy=False)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        wav_file.setnchannels(int(wav_np.shape[0]))
        wav_file.setsampwidth(2)
        wav_file.setframerate(int(sample_rate))
        wav_file.writeframes(pcm16.tobytes())
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def 构建音频输入项(audio_source: str = "", comfy_audio=None) -> dict:
    if comfy_audio is not None:
        return {"type": "input_audio", "input_audio": {"data": comfy音频转wav_base64(comfy_audio), "format": "wav"}}

    source = (audio_source or "").strip()
    if not source:
        raise ValueError("请提供音频路径/URL，或接入 ComfyUI 的 AUDIO 输入。")

    lower = source.lower()
    if lower.startswith("data:audio/"):
        header, sep, payload = source.partition(",")
        if not sep or not payload:
            raise ValueError("data URI 音频内容无效。")
        if "wav" in header:
            fmt = "wav"
        elif "mpeg" in header or "mp3" in header:
            fmt = "mp3"
        else:
            raise ValueError("音频节点仅支持 WAV 或 MP3 的 data URI。")
        return {"type": "input_audio", "input_audio": {"data": payload, "format": fmt}}

    if lower.startswith("http://") or lower.startswith("https://"):
        req = urllib.request.Request(source, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as response:
            audio_bytes = response.read()
            content_type = (response.headers.get("Content-Type", "") or "").lower()
        if not audio_bytes:
            raise ValueError(f"下载音频失败或内容为空：{source}")
        if "wav" in content_type or lower.endswith(".wav"):
            fmt = "wav"
        elif "mpeg" in content_type or "mp3" in content_type or lower.endswith(".mp3"):
            fmt = "mp3"
        else:
            raise ValueError("仅支持在线 WAV/MP3 音频。")
        return {"type": "input_audio", "input_audio": {"data": base64.b64encode(audio_bytes).decode("utf-8"), "format": fmt}}

    if not os.path.exists(source):
        raise FileNotFoundError(f"找不到音频文件：{source}")
    ext = os.path.splitext(source)[1].lower()
    if ext == ".wav":
        fmt = "wav"
    elif ext == ".mp3":
        fmt = "mp3"
    else:
        raise ValueError("本地音频仅支持 WAV/MP3；其他格式请先转码。")
    with open(source, "rb") as f:
        return {"type": "input_audio", "input_audio": {"data": base64.b64encode(f.read()).decode("utf-8"), "format": fmt}}


def 构建图片输入项(image_source: str = "", comfy_image=None, 最大边长: int = 1024) -> list[dict]:
    items: list[dict] = []

    if comfy_image is not None:
        total = int(comfy_image.shape[0]) if hasattr(comfy_image, "shape") else 0
        for index in range(total):
            img_b64 = 图片索引转base64(comfy_image, index, int(最大边长))
            if img_b64:
                items.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}})
        return items

    source = (image_source or "").strip()
    if not source:
        return items

    lower = source.lower()
    if lower.startswith("data:image/") or lower.startswith("http://") or lower.startswith("https://"):
        items.append({"type": "image_url", "image_url": {"url": source}})
        return items

    items.append({"type": "image_url", "image_url": {"url": 本地图片转data_uri(source, int(最大边长))}})
    return items


# ---------------------------------------------------------------- 节点：音频推理

class GS_LLM_Audio:
    """Gemma4 音频理解，支持 图片 + 音频 + 文本 联合输入。"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "模型": ("GS_LLM_MODEL",),
                "提示词": ("STRING", {"default": 默认音频提示词, "multiline": True}),
                "系统提示词": ("STRING", {"default": 默认音频系统提示词, "multiline": True}),
            },
            "optional": {
                "音频": ("AUDIO",),
                "音频路径或URL": ("STRING", {"default": "", "multiline": False, "tooltip": "本地 WAV/MP3 路径或 URL；接了 AUDIO 口时优先用 AUDIO。"}),
                "图片": ("IMAGE",),
                "图片路径或URL": ("STRING", {"default": "", "multiline": False, "tooltip": "可选。与音频一起做多模态联合输入。"}),
                "最大边长": ("INT", {"default": 1024, "min": 128, "max": 16384, "step": 64}),
                "最大生成token": ("INT", {"default": 1024, "min": 20, "max": 32768, "step": 1}),
                "温度": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 64, "min": 0, "max": 200, "step": 1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "step": 1, "control_after_generate": True}),
                "输出think块": ("BOOLEAN", {"default": False}),
                "思考预算token": ("INT", {"default": -1, "min": -1, "max": 8192, "step": 1}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("文本",)
    FUNCTION = "run"
    CATEGORY = "高尚GS/引擎"

    def run(
        self, 模型, 提示词, 系统提示词,
        音频=None, 音频路径或URL="", 图片=None, 图片路径或URL="",
        最大边长=1024, 最大生成token=1024, 温度=1.0, top_p=0.95, top_k=64,
        seed=0, 输出think块=False, 思考预算token=-1,
    ):
        model = GSModelStorage.resolve(模型)
        llm = model.llm
        family = model.family

        if model.chat_handler is None:
            raise RuntimeError("当前模型未加载 mmproj，无法进行音频推理。请在「GS 模型加载器」里选择对应的 mmproj。")

        messages = []
        system_text = (系统提示词 or "").strip()
        if system_text:
            messages.append({"role": "system", "content": system_text})

        prompt_text = (提示词 or "").strip()
        user_content = []
        if prompt_text:
            user_content.append({"type": "text", "text": prompt_text})
        user_content.extend(构建图片输入项(图片路径或URL, 图片, 最大边长=int(最大边长)))
        user_content.append(构建音频输入项(音频路径或URL, 音频))
        messages.append({"role": "user", "content": user_content})

        params = 构造采样参数(
            最大生成token, 温度, top_p, top_k, seed, 思考预算token=思考预算token,
        )

        重置推理状态(llm)
        out = 调用chat_completion(llm, messages=messages, params=params)
        try:
            text = out["choices"][0]["message"]["content"]
        except Exception:
            text = str(out)

        text = 清洗输出文本(text, family, bool(输出think块))

        if mm.processing_interrupted():
            raise mm.InterruptProcessingException()

        return (收尾文本(text),)


# ---------------------------------------------------------------- 节点：卸载

class GS_LLM_Unload:
    """透传任意输入，同时释放 llama.cpp 占用的显存/内存。"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "任意输入": (any_type, {
                    "tooltip": "接「高尚 GS 提示词工作台」的输出。链路跑到这里就会卸载模型、释放显存/内存，再把数据原样传给下游。",
                }),
            },
        }

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("任意输出",)
    FUNCTION = "run"
    CATEGORY = "高尚GS/引擎"
    # OUTPUT_NODE：即使后面什么都不接，这个节点也会被执行（否则 ComfyUI 会把
    # 「输出没人用」的节点剪掉，模型就永远卸不掉）。
    OUTPUT_NODE = True

    def run(self, 任意输入):
        GSModelStorage.unload()
        return (任意输入,)
