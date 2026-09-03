# -*- coding: utf-8 -*-
"""
GS-prompt-extend : 高尚 GS 提示词工作台

一个插件打通「本地 GGUF 多模态模型」+「MiniMax H3 提示词工程」，共 3 个节点：

    [高尚 GS 模型加载器] ──┐
                           ├─► [高尚 GS 提示词工作台] ─► [高尚 GS 模型卸载器] ─► 继续接后面的节点
    [@ 引用画布上的图] ────┘

  · 模型加载器：一个节点里选 GGUF + mmproj + 参数。
  · 提示词工作台：一个节点框写提示词 + @ 引用 + 扩写 + 看图。
  · 模型卸载器：任意输入 → 任意输出透传；跑完提示词后把 llama.cpp 的显存/内存全放掉。

模型放这里： ComfyUI/models/LLM/
"""

from .gs_engine import (
    GS_LLM_Loader,
    GS_LLM_Unload,
)
from .gs_nodes import (
    GS_H3_Workbench,
)

NODE_CLASS_MAPPINGS = {
    "GS_LLM_Loader": GS_LLM_Loader,
    "GS_H3_Workbench": GS_H3_Workbench,
    "GS_LLM_Unload": GS_LLM_Unload,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GS_LLM_Loader": "高尚 GS 模型加载器",
    "GS_H3_Workbench": "高尚 GS 提示词工作台",
    "GS_LLM_Unload": "高尚 GS 模型卸载器",
}

__version__ = "V3.4"

# 前端扩展目录：web/ 下的 .js 会被 ComfyUI 自动扫描并加载
WEB_DIRECTORY = "web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "__version__", "WEB_DIRECTORY"]

print(f"[GS-prompt-extend] 已注册 {len(NODE_CLASS_MAPPINGS)} 个节点（{__version__}）", flush=True)
