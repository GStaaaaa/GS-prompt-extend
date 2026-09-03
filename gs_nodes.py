# -*- coding: utf-8 -*-
"""
GS-prompt-extend : 高尚 GS 提示词工作台（业务层）V3.4

整合插件只保留 3 个节点，一条龙搞定「加载模型 → 写提示词/@引用/扩写/看图 → 卸载」：

    [高尚 GS 模型加载器] ──┐
                           ├─► [高尚 GS 提示词工作台] ─► [高尚 GS 模型卸载器] ─► 继续接后面
    [@ 引用 H3 已连线的图] ─┘

设计要点：
  - 模型加载器一个节点里选 GGUF + mmproj + 参数。
  - 工作台一个节点框写提示词；敲 @ 会弹出「H3 节点上已连线的参考素材」，
    选中即插入 <Picture N> / <Video N> / <Audio N> 标记。
  - 被 @ 引用的图会自动从 input 目录加载喂给视觉模型；不接模型时原样输出。
  - 卸载器（在 gs_engine.py）透传任意输入，同时把 llama.cpp 的显存/内存放掉，
    提示词扩写完不再占资源。
  - V3.4 护栏：引用了参考图却没 mmproj / 清单里找不到图 / 图加载失败，
    一律直接报错（参照 QwenTE/Gemma4TE），绝不静默降级让模型瞎编人物。
    （前端配套修复见 web/gs_workbench.js V3.4：清单 widget 不再移出 node.widgets，
    排队前按画布自动重建清单。）
"""

import json
import os
import re

import comfy.model_management as mm
import folder_paths

from . import gs_h3_rules as H3Rules
from .gs_engine import (
    GSModelStorage,
    构造采样参数,
    清洗输出文本,
    图片索引转base64,
    本地图片转data_uri,
    重置推理状态,
    收尾文本,
    调用chat_completion,
    log,
)

模式选项 = H3Rules.MODE_CHOICES


# ============================================================ 素材清单解析

def 解析素材清单(素材清单: str) -> list[dict]:
    """解析前端写入的参考素材清单（JSON 数组）。"""
    raw = (素材清单 or "").strip()
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except Exception:
        return [{"tag": line.strip(), "type": "text", "name": "", "subfolder": ""}
                for line in raw.splitlines() if line.strip()]

    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        return []

    结果 = []
    for item in data:
        if isinstance(item, dict):
            结果.append({
                "tag": str(item.get("tag", "")).strip(),
                "type": str(item.get("type", "image")).strip().lower(),
                "name": str(item.get("name", "")).strip(),
                "subfolder": str(item.get("subfolder", "")).strip(),
            })
        elif isinstance(item, str) and item.strip():
            结果.append({"tag": item.strip(), "type": "text", "name": "", "subfolder": ""})
    return 结果


def 解析素材文件路径(name: str, subfolder: str = "") -> str:
    """把素材名解析成磁盘绝对路径，找不到返回空串。"""
    if not name:
        return ""
    候选: list[str] = []
    try:
        p = folder_paths.get_annotated_filepath(name)
        if p:
            候选.append(p)
    except Exception:
        pass
    候选.append(name)
    try:
        input_dir = folder_paths.get_input_directory()
        if subfolder:
            候选.append(os.path.join(input_dir, subfolder, name))
        候选.append(os.path.join(input_dir, name))
    except Exception:
        pass
    for path in 候选:
        if path and os.path.isfile(path):
            return path
    return ""


标记正则 = re.compile(r"<\s*(picture|video|audio)\s*\d+\s*>", re.IGNORECASE)


def 提示词用到的标记(文本: str) -> set:
    """提取提示词里真正出现的 <Picture N> / <Video N> / <Audio N> 标记。"""
    return {m.group(0).replace(" ", "").lower() for m in 标记正则.finditer(文本 or "")}


def 清单转文字说明(素材: list[dict]) -> str:
    """把素材清单渲染成给模型看的文字。"""
    if not 素材:
        return ""
    行 = []
    for item in 素材:
        tag = item.get("tag", "")
        name = item.get("name", "")
        if item.get("type") == "text":
            行.append(tag)
            continue
        行.append(f"{tag} = {name}" if name else tag)
    return "\n".join(行)


# ============================================================ 节点：提示词工作台

class GS_H3_Workbench:
    """写提示词 + @ 引用 H3 已连线素材 + 一键扩写/看图，一个节点全搞定。"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "任务模式": (模式选项, {
                    "tooltip": "决定扩写规则。全参考=多图多视频多音频；T2VA=纯文字；I2VA=首帧；FL2VA=首尾帧；L2VA=尾帧。",
                }),
                "提示词": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "在这里写你的想法。输入 @ 插入 H3 上已连线的参考素材…",
                    "tooltip": "输入 @ 会弹出 H3 节点上已连线的参考素材（紫色=已启用）。",
                }),
            },
            "optional": {
                "模型": ("GS_LLM_MODEL", {
                    "tooltip": "接上「高尚 GS 模型加载器」才会扩写/看图；不接则原样输出提示词。",
                }),
                "参考素材清单": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "由前端 @ 选择自动写入，一般不用手动改。",
                }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "step": 1,
                                 "control_after_generate": True}),
                "最大边长": ("INT", {"default": 1024, "min": 128, "max": 16384, "step": 64,
                                    "tooltip": "图片按最长边缩放，调小可提速、省显存。"}),
                "时长秒": ("INT", {"default": 6, "min": 1, "max": 30, "step": 1,
                                 "tooltip": "提示词按几秒写（仅约束提示词时间轴，不改出片时长）。默认 6 = 沿用规则原样。"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("提示词",)
    FUNCTION = "run"
    CATEGORY = "高尚GS"

    def run(
        self, 任务模式, 提示词,
        模型=None, 参考素材清单="",
        seed=0, 最大边长=1024, 时长秒=6,
    ):
        mode_key = H3Rules.解析模式选项(任务模式)
        草稿 = (提示词 or "").strip()

        # -------------------------------------------------- 没接模型：纯提示词框
        if 模型 is None:
            return (收尾文本(草稿),)

        model = GSModelStorage.resolve(模型)
        llm = model.llm
        family = model.family

        # -------------------------------------------------- 组装 system
        system_text = H3Rules.取最终规则(mode_key)
        try:
            system_text = H3Rules.按秒数改写时长(system_text, 时长秒)
        except Exception as e:
            log(f"时长秒 改写跳过（不影响出词）：{e}")

        # -------------------------------------------------- 解析 @ 选中的参考素材
        素材 = 解析素材清单(参考素材清单)

        用到 = 提示词用到的标记(草稿)
        图片引用 = [t for t in 用到 if t.startswith("<picture")]
        图片引用原文 = [
            m.group(0) for m in 标记正则.finditer(草稿 or "")
            if m.group(1).lower() == "picture"
        ]

        use_vision = model.chat_handler is not None

        # ---- 护栏 1：引用了参考图但模型没装「眼睛」（mmproj=无）→ 直接报错，
        #      绝不静默降级瞎编（参照 QwenTE/Gemma4TE 的报错式做法）。
        #      V3.3 之前这里只打一行日志继续跑，结果人物描述全靠模型编。
        if 图片引用 and not use_vision:
            提示 = "、".join(图片引用原文) or "、".join(图片引用)
            raise RuntimeError(
                f"提示词里引用了参考图（{提示}），但「高尚 GS 模型加载器」的"
                "「视觉投影mmproj」选的是「无」——模型根本看不到这些图，"
                "写出来的人物必然对不上。请到加载器里把「视觉投影mmproj」"
                "改成 mmproj 文件后重跑。"
            )

        图片素材 = [
            m for m in 素材
            if m.get("type") == "image" and m.get("name")
            and (not 用到 or str(m.get("tag", "")).replace(" ", "").lower() in 用到)
        ]

        # ---- 护栏 2：引用了参考图，但清单里找不到对应的图 → 直接报错。
        if 图片引用 and not 图片素材:
            提示 = "、".join(图片引用原文) or "、".join(图片引用)
            raise RuntimeError(
                f"提示词里引用了参考图（{提示}），但「参考素材清单」里没有对应的图。"
                "请 F5 刷新页面后在提示词里重新 @ 选择参考素材"
                "（刷新后排队会自动按画布重建清单），并确认对应素材组没被旁路（紫色=已关闭）。"
            )

        # -------------------------------------------------- 组装 user 文本
        parts: list[str] = []
        if 草稿:
            parts.append(f"## 用户的原始想法\n{草稿}")
        else:
            parts.append(
                "## 用户的原始想法\n"
                "（用户未提供草稿，请你基于下方参考素材自行构思一个合理、有电影感的镜头。）"
            )

        if 素材:
            说明 = 清单转文字说明(素材)
            parts.append(
                "## 参考素材清单\n"
                "用户已在提示词中用 <Picture N> / <Video N> / <Audio N> 引用下列素材，"
                "请严格按这些标记来描述对应主体，不要张冠李戴：\n\n" + 说明
            )

        user_text = "\n\n".join(parts)

        # -------------------------------------------------- 收集图片（@ 引用的）
        messages = []
        if system_text:
            messages.append({"role": "system", "content": system_text})

        if not use_vision and 图片素材:
            log("当前模型未加载 mmproj，本次草稿没有 <Picture N> 引用，按纯文本扩写（清单里的图不会喂）。")

        图片项: list[dict] = []
        已加载: list[str] = []
        跳过: list[str] = []

        if use_vision:
            for item in 图片素材:
                path = 解析素材文件路径(item.get("name", ""), item.get("subfolder", ""))
                if not path:
                    跳过.append(item.get("name", "?"))
                    continue
                try:
                    图片项.append({
                        "type": "image_url",
                        "image_url": {"url": 本地图片转data_uri(path, int(最大边长))},
                    })
                    已加载.append(f"{item.get('tag', '')} -> {os.path.basename(path)}".strip())
                except Exception as exc:
                    跳过.append(f"{item.get('name', '?')}（{exc}）")

            # ---- 护栏 3：有想喂却加载失败的图 → 直接报错，绝不静默少喂。
            if 跳过:
                raise RuntimeError(
                    "有参考图找不到文件、无法喂给模型：" + "、".join(跳过) +
                    "。请检查图片是否还在 input 目录（或素材组是否被旁路），"
                    "F5 后重新 @ 选择再跑。"
                )

        if 图片项:
            user_content: list = [{"type": "text", "text": user_text}]
            user_content.extend(图片项)
            messages.append({"role": "user", "content": user_content})
        else:
            messages.append({"role": "user", "content": user_text})

        # -------------------------------------------------- 执行（固定采样参数，只留 seed/最大边长可调）
        params = 构造采样参数(
            最大生成token=3072, 温度=0.7, top_p=0.9, top_k=20, seed=seed,
            重复惩罚=1.05,
        )

        log(
            f"H3 工作台扩写（模式={mode_key}, 看图={len(图片项)}张, "
            f"max_tokens=3072）"
        )

        重置推理状态(llm)
        out = 调用chat_completion(llm, messages=messages, params=params)
        try:
            text = out["choices"][0]["message"]["content"]
        except Exception:
            text = str(out)

        text = 清洗输出文本(text, family, 保留think块=False)
        text = 收尾文本(text)

        if mm.processing_interrupted():
            raise mm.InterruptProcessingException()

        return (text,)
