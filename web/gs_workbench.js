/*
 * GS 提示词工作台 —— @ 提及功能（V3.3）
 *
 * 在提示词框里输入 @，弹出 H3 节点上「已连线」的参考素材列表：
 *   ref_images.ref_image_N  → <Picture N>
 *   ref_videos.ref_video_N  → <Video N>
 *   ref_audios.ref_audio_N  → <Audio N>
 *   first_frame             → <Picture 1>
 *   last_frame              → <Picture 2>
 *
 * 只列「当前开启（未被旁路）的参考图/音频」。判定依据：
 *   - 槽位必须已连线（input.link != null）
 *   - 向上追踪到素材节点，看它所在 group 是否被 rgthree Fast Groups
 *     Bypasser 旁路：组内节点 mode === 4（bypass）即「被关掉」，跳过；
 *     mode === 0（正常）才列入。
 *
 * 选中即插入标记，并把素材文件名写进「参考素材清单」，
 * 后端执行时自动从 input 目录加载这些图喂给视觉模型。
 *
 * 修复（V3.3）：
 *   - 用「隐藏 textarea + contenteditable 编辑器」包裹方案（参考 ComfyUI_MiniMaxH3_Director），
 *     textarea 保留在 DOM（CSS clip 隐藏），ComfyUI widget 系统不会崩，输入框一定正常显示。
 *   - @ 选中后插入带缩略图的 <Picture N> chip（内联 16px 缩略图 + 绿色圆角标签）。
 *   - 尊重 rgthree「参考图N/参考音频N」开关：被 bypass（mode=4）的素材不列出。
 *   - 面板定位做视口钳制，避免飘走。
 *
 * 修复（V3.4，看图=0 的根因）：
 *   - 旧版隐藏「参考素材清单」时把它从 node.widgets 里 splice 移除了。
 *     但 ComfyUI 按 node.widgets 的顺序序列化 widgets_values（保存和排队都是），
 *     widget 一被移走，后面的值整体错位：seed 的值落进清单槽（后端收到 "1"）、
 *     seed 槽收到 "randomize" 字符串（校验直接打回），素材 JSON 从来没到过后端
 *     —— 于是每次执行都是「看图=0张」，人物全靠模型编。
 *     （对照 TE_MAN 的 reference_labels：它是留在原位的普通 widget，所以人家没事。）
 *   - 现在只做视觉隐藏（高度压 0 + display:none），widget 留在数组里，序列化永远对齐。
 *   - 另加自愈：排队前按当前画布状态重建清单 —— 旧工作流里存的垃圾值、
 *     手敲的 <Picture N>、换了文件没重新 @ 的，都会被自动修正。
 */

import { app } from "../../../scripts/app.js";

const NODE_NAME = "GS_H3_Workbench";
const REF_WIDGET = "参考素材清单";
const PROMPT_WIDGET = "提示词";

// H3 节点上承载参考素材的输入槽 -> 标记前缀
const SLOT_RULES = [
    { re: /^ref_images\.ref_image_(\d+)$/, tag: (n) => `<Picture ${n}>`, type: "image" },
    { re: /^ref_videos\.ref_video_(\d+)$/, tag: (n) => `<Video ${n}>`, type: "video" },
    { re: /^ref_audios\.ref_audio_(\d+)$/, tag: (n) => `<Audio ${n}>`, type: "audio" },
    { re: /^ref_video_audios\.ref_video_audio_(\d+)$/, tag: (n) => `<Audio ${n}>`, type: "audio" },
    { re: /^first_frame$/, tag: () => `<Picture 1>`, type: "image" },
    { re: /^last_frame$/, tag: () => `<Picture 2>`, type: "image" },
];

/**
 * 判断一个节点是不是「参考素材宿主」（H3 类节点）。
 *
 * 判定依据是「有没有 ref_images.ref_image_N / ref_videos.ref_video_N /
 * ref_audios.ref_audio_N / first_frame / last_frame 这类输入槽」，而不是靠
 * 节点 type 里的 "h3"/"minimax" 字样。这样更准：
 *   - 不会因为 type 命名变了（如 MiniMaxH3xxx / MinimaxH3xxx）而漏掉
 *   - 不会把 GS_H3_Workbench 自己（type 里含 H3）误当成宿主
 *   - 支持各种 H3 变体节点（ReferenceToVideo / ImageToVideo / Director 等）
 */
function isRefHost(node) {
    if (!node || !node.inputs) return false;
    return node.inputs.some((input) =>
        SLOT_RULES.some((rule) => rule.re.test(String(input.name || "")))
    );
}

function getGraph() {
    return app.graph || (app.canvas && app.canvas.graph) || null;
}

function getLink(linkId) {
    const g = getGraph();
    if (!g || linkId == null) return null;
    return g.links ? g.links[linkId] : null;
}

function getNodeById(id) {
    const g = getGraph();
    if (!g || id == null) return null;
    return typeof g.getNodeById === "function" ? g.getNodeById(id) : null;
}

function splitFileRef(name) {
    const clean = String(name || "").replace(/\[.*?\]/g, "").trim();
    const idx = clean.lastIndexOf("/");
    if (idx >= 0) return [clean.slice(0, idx), clean.slice(idx + 1)];
    return ["", clean];
}

/** 从素材节点里找出文件名 widget 的值 */
function pickFileName(srcNode) {
    if (!srcNode) return "";
    const widgets = srcNode.widgets || [];
    const MEDIA_RE = /\.(png|jpe?g|webp|bmp|gif|mp4|mov|mkv|webm|avi|mp3|wav|flac|ogg|m4a)$/i;
    for (const w of widgets) {
        const v = w && w.value;
        if (typeof v === "string" && MEDIA_RE.test(v)) return v;
    }
    for (const w of widgets) {
        const v = w && w.value;
        if (typeof v === "string" && v.trim()) return v.trim();
    }
    return "";
}

/**
 * 判断一个节点是否被 rgthree Fast Groups Bypasser 旁路（忽略/关掉）。
 * bypass 的节点 mode === 4（ComfyUI 用 4 表示 bypass，LiteGraph.ALWAYS=0）。
 */
function isNodeBypassed(node) {
    if (!node) return false;
    return node.mode === 4 || node.mode === "4";
}

/**
 * 拿到一个节点的「名字」：KJNodes GetNode/SetNode 用 widgets[0].value
 * 存名字（如「图像1」「参考音频1」），用于两者互相匹配。
 *
 * 注意：前端运行时 LiteGraph 节点没有 widgets_values（那是序列化字段），
 * 只有 widgets[] 对象数组，每个 widget 有 .value / .name。
 */
function getNodeName(node) {
    if (!node) return "";
    const widgets = node.widgets || [];
    const first = widgets[0];
    if (first && first.value && typeof first.value === "string") return first.value.trim();
    // 兜底：序列化字段（刚加载时可能还没转成 widgets）
    const wv = node.widgets_values;
    if (Array.isArray(wv) && wv[0] && typeof wv[0] === "string") return wv[0].trim();
    return "";
}

/**
 * 沿一条素材链路的「源头节点」向上追，只要任一环节 mode=4（被旁路）
 * 就返回 true（这张参考图被关掉）。
 */
function chainDisabled(startNode) {
    let cursor = startNode;
    const visited = new Set();
    while (cursor && !visited.has(cursor.id)) {
        visited.add(cursor.id);
        if (isNodeBypassed(cursor)) return true;
        let next = null;
        for (const input of cursor.inputs || []) {
            if (input.link != null) {
                const link = getLink(input.link);
                if (link) {
                    const up = getNodeById(link.origin_id);
                    if (up) { next = up; break; }
                }
            }
        }
        cursor = next;
    }
    return false;
}

/**
 * 从 H3 槽位上游（通常是 GetNode）反查到真正的素材源（LoadImage/LoadAudio），
 * 并判断是否被 rgthree「参考图N/参考音频N」开关旁路。
 *
 * 机制：H3.ref_image_N 的上游是 GetNode「图像N」（无输入连线），
 * GetNode 通过名字匹配 SetNode「图像N」，SetNode 再向上连 LoadImage。
 * rgthree 旁路的是「参考图N」group 里的 LoadImage + Scale + SetNode，
 * 所以需要沿 SetNode 向上追，检查链路是否被 mode=4 关掉。
 */
function resolveRefSource(srcNode) {
    if (!srcNode) return null;
    // 情况 A：上游是 GetNode -> 按名字找 SetNode，再向上追 LoadImage
    const name = getNodeName(srcNode);
    if (name) {
        const g = getGraph();
        const allNodes = (g && g._nodes) || [];
        // 找同名 SetNode（KJNodes 的 SetNode，有 IMAGE/AUDIO 输入）
        for (const n of allNodes) {
            if (!n || n === srcNode) continue;
            if (!/setnode/i.test(n.type || "")) continue;
            if (getNodeName(n) !== name) continue;
            // 从 SetNode 向上追到 LoadImage，并检查 bypass
            let cursor = n;
            const visited = new Set();
            let disabled = false;
            let loader = null;
            while (cursor && !visited.has(cursor.id)) {
                visited.add(cursor.id);
                if (isNodeBypassed(cursor)) disabled = true;
                // 记录最上游的 LoadImage/LoadAudio
                if (/loadimage|loadaudio|loadvideo/i.test(cursor.type || "")) {
                    loader = cursor;
                }
                let next = null;
                for (const input of cursor.inputs || []) {
                    if (input.link != null) {
                        const link = getLink(input.link);
                        if (link) {
                            const up = getNodeById(link.origin_id);
                            if (up) { next = up; break; }
                        }
                    }
                }
                cursor = next;
            }
            if (loader) {
                return { loader, disabled };
            }
        }
    }
    // 情况 B：上游直接就是 LoadImage/LoadAudio/Scale（没有 Set/Get 中转）
    let cursor = srcNode;
    const visited = new Set();
    let disabled = false;
    let loader = null;
    while (cursor && !visited.has(cursor.id)) {
        visited.add(cursor.id);
        if (isNodeBypassed(cursor)) disabled = true;
        if (/loadimage|loadaudio|loadvideo/i.test(cursor.type || "")) loader = cursor;
        let next = null;
        for (const input of cursor.inputs || []) {
            if (input.link != null) {
                const link = getLink(input.link);
                if (link) { const up = getNodeById(link.origin_id); if (up) { next = up; break; } }
            }
        }
        cursor = next;
    }
    return loader ? { loader, disabled } : null;
}

/**
 * 收集 H3 节点上「已连线且未被旁路」的参考素材。
 * 先扫工作台下游的 H3 节点；如果没连下游，就扫全画布所有 H3 节点。
 */
function collectReferences(workbenchNode) {
    try {
        return _collectReferences(workbenchNode);
    } catch (e) {
        console.error("[GS] collectReferences 异常，降级返回空列表", e);
        return [];
    }
}

function _collectReferences(workbenchNode) {
    const out = [];
    const seen = new Set();

    // 1) 从工作台节点的输出出发，沿下游链路逐跳找「参考素材宿主」（H3 类节点）。
    //
    //    旧实现只走一跳、且不判断找到的是不是 H3，导致「工作台 → Text 中间节点 → H3」
    //    这种常见接法下，把 Text 当成目标后就跳过全画布扫描，结果一项都列不出来。
    //    现在改成：沿下游 BFS 一路找（上限 8 跳），只收集真正带 ref_* 槽位的节点。
    const h3Targets = new Set();
    const queue = [];
    for (const output of workbenchNode.outputs || []) {
        for (const linkId of output.links || []) {
            const link = getLink(linkId);
            if (!link) continue;
            const target = getNodeById(link.target_id);
            if (target) queue.push({ node: target, depth: 0 });
        }
    }
    const visitedDown = new Set([workbenchNode.id]);
    let head = 0;
    while (head < queue.length) {
        const { node, depth } = queue[head++];
        if (!node || visitedDown.has(node.id)) continue;
        visitedDown.add(node.id);
        if (isRefHost(node)) h3Targets.add(node);
        if (depth >= 8) continue; // 防环路 / 防止无意义深挖
        for (const out2 of node.outputs || []) {
            for (const linkId of out2.links || []) {
                const link = getLink(linkId);
                if (!link) continue;
                const next = getNodeById(link.target_id);
                if (next && !visitedDown.has(next.id)) queue.push({ node: next, depth: depth + 1 });
            }
        }
    }

    // 2) 下游链路里没找到任何宿主，就扫全画布
    if (h3Targets.size === 0) {
        const g = getGraph();
        for (const n of (g && g._nodes) || []) {
            if (!n || n === workbenchNode) continue;
            if (isRefHost(n)) h3Targets.add(n);
        }
    }

    // 3) 遍历每个 H3 节点的 inputs，只取已连线且未被旁路的槽位
    for (const node of h3Targets) {
        for (const input of node.inputs || []) {
            if (input.link == null) continue; // 没连线 -> 跳过
            for (const rule of SLOT_RULES) {
                const m = String(input.name || "").match(rule.re);
                if (!m) continue;

                const n = m[1] !== undefined ? parseInt(m[1], 10) + 1 : 1;
                const tag = rule.tag(n);
                if (seen.has(tag)) break;

                const upLink = getLink(input.link);
                const srcNode = upLink ? getNodeById(upLink.origin_id) : null;
                const resolved = resolveRefSource(srcNode);

                // 没找到素材源，或整条链路被 rgthree 开关旁路（关掉），则不列出
                if (!resolved || resolved.disabled) break;

                const rawName = pickFileName(resolved.loader);
                const [subfolder, filename] = splitFileRef(rawName);

                seen.add(tag);
                out.push({
                    tag,
                    type: rule.type,
                    name: filename,
                    subfolder,
                    label: filename || (resolved.loader ? resolved.loader.title || resolved.loader.type : "未连接"),
                    upstream: resolved.loader ? resolved.loader.type : "",
                });
                break;
            }
        }
    }

    return out;
}

// ---------------------------------------------------------------- 清单读写

/** 从可见 widgets 或隐藏的 _gs_hidden_widgets 里找「参考素材清单」widget */
function findManifestWidget(node) {
    let w = (node.widgets || []).find((x) => x && x.name === REF_WIDGET);
    if (!w && node._gs_hidden_widgets) {
        w = node._gs_hidden_widgets.find((x) => x && x.name === REF_WIDGET);
    }
    return w || null;
}

function readManifest(node) {
    const w = findManifestWidget(node);
    if (!w || !w.value) return [];
    try {
        const data = JSON.parse(w.value);
        return Array.isArray(data) ? data : [];
    } catch (e) {
        return [];
    }
}

function writeManifest(node, list) {
    const w = findManifestWidget(node);
    if (!w) return;
    w.value = JSON.stringify(list);
}

function upsertManifest(node, item) {
    const list = readManifest(node);
    const i = list.findIndex((x) => x && x.tag === item.tag);
    if (i >= 0) list[i] = item;
    else list.push(item);
    writeManifest(node, list);
}

// ---------------------------------------------------------------- 清单自愈（V3.4）

const normTag = (tag) => String(tag || "").replace(/\s+/g, "").toLowerCase();

/**
 * 按当前画布状态重建「参考素材清单」：
 * 扫提示词文本里出现的每个 <Picture/Video/Audio N> 标记，去画布上（H3 节点
 * 已连线且未旁路的素材）找到对应文件，重写进清单。
 * 排队前调用 —— 旧工作流里存的垃圾值（如错位产生的 "1"）会被自动修正。
 */
function rebuildManifestFromCanvas(node) {
    try {
        const promptW = (node.widgets || []).find((w) => w && w.name === PROMPT_WIDGET);
        if (!promptW) return 0;
        const text = String(promptW.value || "");

        const re = /<(picture|video|audio)\s+(\d+)\s*>/gi;
        const wanted = new Set(); // 规范化键集合，如 "<picture1>"
        let m;
        while ((m = re.exec(text))) wanted.add(normTag(m[0]));
        if (!wanted.size) return 0; // 文本里没有标记，不动清单

        const refs = collectReferences(node);
        const list = [];
        for (const item of refs) {
            if (!wanted.has(normTag(item.tag))) continue;
            list.push({
                tag: item.tag,
                type: item.type,
                name: item.name,
                subfolder: item.subfolder,
            });
        }
        writeManifest(node, list); // 画布是唯一事实源：找不到的标记就不喂（后端会报错提示）
        return list.length;
    } catch (e) {
        console.warn("[GS] 重建参考素材清单失败", e);
        return 0;
    }
}

/**
 * 修复旧工作流错位残留的数值控件：
 * 旧版序列化错位会把 "randomize" 之类的字符串灌进 seed / 最大边长，
 * 排队时被后端 INT 校验直接打回。排队前把非法值重置回默认。
 */
function sanitizeNumericWidgets(node) {
    const DEFAULTS = { seed: 0, "最大边长": 1024, "时长秒": 6 };
    for (const w of node.widgets || []) {
        if (!w || !(w.name in DEFAULTS)) continue;
        if (!Number.isFinite(Number(w.value))) {
            w.value = DEFAULTS[w.name];
            console.warn(`[GS] 检测到「${w.name}」值异常（旧工作流错位残留），已重置为 ${w.value}`);
        }
    }
}

function syncAllWorkbenchManifests() {
    const g = getGraph();
    if (!g) return;
    for (const n of g._nodes || []) {
        if (n && n.type === NODE_NAME) {
            sanitizeNumericWidgets(n);
            rebuildManifestFromCanvas(n);
        }
    }
}

// ---------------------------------------------------------------- 候选面板

let PANEL = null;

function closePanel() {
    if (PANEL && PANEL.parentNode) PANEL.parentNode.removeChild(PANEL);
    PANEL = null;
}

function thumbUrl(item) {
    if (item.type !== "image" || !item.name) return "";
    const sub = item.subfolder ? `&subfolder=${encodeURIComponent(item.subfolder)}` : "";
    return `/view?filename=${encodeURIComponent(item.name)}${sub}&type=input`;
}

function openPanel(node, textarea, items, query, onPick) {
    closePanel();

    const filtered = query
        ? items.filter((it) =>
              (it.label + " " + it.tag + " " + it.upstream).toLowerCase().includes(query.toLowerCase())
          )
        : items;

    // 即使没有候选也弹面板给出原因，避免「输入 @ 没反应、以为功能坏了」的困惑。
    if (!filtered.length) {
        const empty = document.createElement("div");
        empty.className = "gs-mention-panel gs-mention-empty";
        Object.assign(empty.style, {
            position: "fixed",
            zIndex: "9999",
            minWidth: "260px",
            maxWidth: "340px",
            padding: "10px 12px",
            borderRadius: "8px",
            background: "var(--comfy-input-bg, #1e1e1e)",
            border: "1px solid var(--border-color, #444)",
            boxShadow: "0 18px 45px rgba(0,0,0,0.35)",
            font: "12px system-ui, sans-serif",
            color: "var(--fg-color, #ddd)",
            lineHeight: "1.6",
            whiteSpace: "pre-line",
        });
        if (!items.length) {
            empty.textContent =
                "⚠️ 画布上没有已开启的参考素材\n\n去把参考图/视频/音频的开关打开（紫色圆点 = 已关闭），再重新输入 @。";
        } else {
            empty.textContent = `没有匹配「@${query}」的素材\n\n继续输入缩小范围，按 Esc 关闭。`;
        }
        document.body.appendChild(empty);
        PANEL = empty;
        try {
            const cr = getCaretRect(textarea);
            const ph = empty.offsetHeight || 120;
            // 优先在光标下方弹出；空间不够就翻到光标上方
            const belowOk = cr.bottom + 4 + ph < window.innerHeight - 12;
            const top = belowOk ? cr.bottom + 4 : Math.max(4, cr.top - ph - 4);
            empty.style.left = `${Math.max(4, Math.min(cr.left, window.innerWidth - empty.offsetWidth - 8))}px`;
            empty.style.top = `${top}px`;
        } catch (e) { /* 定位失败不影响使用 */ }
        return;
    }

    const panel = document.createElement("div");
    panel.className = "gs-mention-panel";
    Object.assign(panel.style, {
        position: "fixed",
        zIndex: "9999",
        maxHeight: "340px",
        overflowY: "auto",
        minWidth: "280px",
        padding: "6px",
        borderRadius: "8px",
        background: "var(--comfy-input-bg, #1e1e1e)",
        border: "1px solid var(--border-color, #444)",
        boxShadow: "0 18px 45px rgba(0,0,0,0.35)",
        font: "12px system-ui, sans-serif",
        color: "var(--fg-color, #ddd)",
    });

    // 彩蛋：标题栏显示素材计数和操作提示
    const head = document.createElement("div");
    head.textContent = `已开启 ${items.length} 个参考素材 · ↑↓ 选择 · Enter 确认 · Esc 关闭`;
    Object.assign(head.style, {
        padding: "4px 8px 8px",
        fontSize: "10px",
        color: "rgba(255,255,255,0.45)",
        userSelect: "none",
        borderBottom: "1px solid rgba(255,255,255,0.08)",
        marginBottom: "4px",
        position: "sticky",
        top: "-6px",
        background: "inherit",
    });
    panel.appendChild(head);

    filtered.forEach((item, idx) => {
        const row = document.createElement("div");
        row.className = "gs-mention-row";
        Object.assign(row.style, {
            display: "flex",
            alignItems: "center",
            gap: "8px",
            padding: "6px 8px",
            borderRadius: "6px",
            cursor: "pointer",
        });

        const url = thumbUrl(item);
        if (url) {
            const img = document.createElement("img");
            img.src = url;
            Object.assign(img.style, {
                width: "36px",
                height: "36px",
                objectFit: "cover",
                borderRadius: "4px",
                flex: "0 0 auto",
                background: "#000",
            });
            // 缩略图加载失败（文件不在 input 目录等）时换成类型图标，不留空白
            img.onerror = () => {
                const badge = document.createElement("span");
                badge.textContent = item.type === "audio" ? "♪" : item.type === "video" ? "▶" : "▣";
                Object.assign(badge.style, {
                    width: "36px",
                    height: "36px",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    borderRadius: "4px",
                    background: "rgba(128,128,128,0.2)",
                    flex: "0 0 auto",
                });
                img.replaceWith(badge);
            };
            row.appendChild(img);
        } else {
            const badge = document.createElement("span");
            badge.textContent = item.type === "audio" ? "♪" : item.type === "video" ? "▶" : "▣";
            Object.assign(badge.style, {
                width: "36px",
                height: "36px",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                borderRadius: "4px",
                background: "rgba(128,128,128,0.2)",
                flex: "0 0 auto",
            });
            row.appendChild(badge);
        }

        const col = document.createElement("div");
        col.style.minWidth = "0";
        const t1 = document.createElement("div");
        t1.textContent = item.tag;
        t1.style.fontWeight = "600";
        const t2 = document.createElement("div");
        t2.textContent = item.label;
        Object.assign(t2.style, {
            fontSize: "11px",
            opacity: "0.7",
            whiteSpace: "nowrap",
            overflow: "hidden",
            textOverflow: "ellipsis",
        });
        col.appendChild(t1);
        col.appendChild(t2);
        row.appendChild(col);

        row.addEventListener("mouseenter", () => setActive(idx));
        row.addEventListener("mousedown", (e) => {
            e.preventDefault();
            onPick(item);
        });
        panel.appendChild(row);
    });

    function setActive(idx) {
        panel.querySelectorAll(".gs-mention-row").forEach((c, i) => {
            c.style.background = i === idx ? "rgba(44,221,118,0.18)" : "transparent";
        });
    }
    setActive(0);

    document.body.appendChild(panel);
    PANEL = panel;

    try {
        const cr = getCaretRect(textarea);
        const ph = panel.offsetHeight || 340;
        // 优先在光标下方弹出；空间不够就翻到光标上方
        const belowOk = cr.bottom + 4 + ph < window.innerHeight - 12;
        const top = belowOk ? cr.bottom + 4 : Math.max(4, cr.top - ph - 4);
        const left = Math.max(4, Math.min(cr.left, window.innerWidth - panel.offsetWidth - 8));
        panel.style.left = `${left}px`;
        panel.style.top = `${top}px`;
    } catch (e) { /* 定位失败不影响使用 */ }
}

// ---------------------------------------------------------------- 扩展注册

app.registerExtension({
    name: "GS.PromptExtend.Workbench",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            try {
                this.size = [480, 280];
                // 隐藏「参考素材清单」widget（前端 @ 自动写入，不需要人工看/改）
                hideManifestWidget(this);
                attachMention(this);
            } catch (e) {
                console.warn("[GS] 工作台初始化失败", e);
            }
            return r;
        };
    },
});

/**
 * 隐藏「参考素材清单」widget，不让它占节点空间、不显示 JSON 代码。
 *
 * ⚠️ 只能视觉隐藏，绝对不能把它从 node.widgets 里移除！
 * ComfyUI 按 node.widgets 的顺序序列化 widgets_values（保存 & 排队都一样），
 * 移除一个 widget 会让后面所有值整体错位 —— V3.3 就是这么坏的：
 * 清单槽收到的是 seed 的值（"1"），素材 JSON 从没到过后端，看图永远是 0 张。
 * （TE_MAN 的 reference_labels 就是留在原位的普通 widget，所以人家一直没事。）
 */
function hideManifestWidget(node) {
    try {
        const w = (node.widgets || []).find((x) => x.name === REF_WIDGET);
        if (!w) return;
        w.computeSize = () => [0, -4]; // 负高度让布局引擎跳过它
        if (w.element) {
            w.element.style.display = "none";
            w.element.style.height = "0px";
        }
    } catch (e) {
        console.warn("[GS] 隐藏参考素材清单失败", e);
    }
}

// ============================================================ token 编辑器（contenteditable + 缩略图 chip）

const TAG_RE = /<(Picture|Video|Audio)\s+(\d+)\s*>/gi;
const TOKEN_CLASS = "gs-prompt-token";

let TOKEN_STYLES_INJECTED = false;
function injectTokenStyles() {
    if (TOKEN_STYLES_INJECTED) return;
    TOKEN_STYLES_INJECTED = true;
    const el = document.createElement("style");
    el.textContent = `
.gs-token-wrap{position:relative;display:flex;flex-direction:column;min-width:0;min-height:0;flex:1 1 auto;width:100%;box-sizing:border-box}
.gs-token-source{position:absolute!important;width:1px!important;height:1px!important;padding:0!important;margin:-1px!important;overflow:hidden!important;clip:rect(0,0,0,0)!important;border:0!important;opacity:0!important;pointer-events:none!important;resize:none!important}
.gs-token-editor{width:100%;min-height:140px;box-sizing:border-box;background:#181818;border:1px solid #333;border-radius:6px;color:#eee;padding:8px;font-size:12px;font-family:inherit;line-height:1.45;outline:none;overflow:auto;white-space:pre-wrap;word-break:break-word}
.gs-token-editor:focus{border-color:#4a7a5a}
.gs-token-editor:empty:before{content:attr(data-placeholder);color:#666;pointer-events:none}
.gs-token{display:inline-flex;align-items:center;gap:6px;max-width:100%;margin:0 2px;padding:2px 9px 2px 2px;border-radius:6px;vertical-align:middle;background:#2f6fd8;border:1px solid rgba(255,255,255,.22);color:#fff;font-size:12px;font-weight:600;line-height:1.3;user-select:none;cursor:default;white-space:nowrap;box-shadow:0 1px 2px rgba(0,0,0,.35)}
.gs-token[contenteditable="false"]{-webkit-user-modify:read-only}
.gs-token-video{background:#6d4fd0}
.gs-token-audio{background:#c98a1f}
.gs-token-thumb{width:20px;height:20px;border-radius:3px;object-fit:cover;flex-shrink:0;background:#0b0b0b}
.gs-token-glyph{width:20px;height:20px;border-radius:3px;flex-shrink:0;display:inline-flex;align-items:center;justify-content:center;font-size:11px;line-height:1;background:rgba(255,255,255,.16);color:#fff}
.gs-token-label{max-width:7em;overflow:hidden;text-overflow:ellipsis}
`;
    document.head.appendChild(el);
}

function kindOfTag(type) {
    const k = String(type || "").toLowerCase();
    if (k === "picture") return "image";
    if (k === "video") return "video";
    if (k === "audio") return "audio";
    return "image";
}

/** 根据 tag（如 <Picture 1>）从 manifest 里找到对应的素材项 */
function findRefByTag(list, tag) {
    return (list || []).find((x) => x && x.tag === tag) || null;
}

function makeTokenChip(tag, item) {
    const m = tag.match(/<(\w+)\s+(\d+)\s*>/i);
    const kind = m ? kindOfTag(m[1]) : "image";
    const chip = document.createElement("span");
    // gs-token 是样式类（见 injectTokenStyles），TOKEN_CLASS 是逻辑标记，两个都要有
    chip.className = `${TOKEN_CLASS} gs-token gs-token-${kind}`;
    chip.contentEditable = "false";
    chip.dataset.tag = tag;
    chip.title = tag;

    const url = item ? thumbUrl(item) : "";
    if (kind === "image" && url) {
        const img = document.createElement("img");
        img.className = "gs-token-thumb";
        img.src = url;
        img.onerror = () => { img.style.display = "none"; };
        chip.appendChild(img);
    } else {
        const glyph = document.createElement("span");
        glyph.className = "gs-token-glyph";
        glyph.textContent = kind === "video" ? "▶" : kind === "audio" ? "♪" : "▣";
        chip.appendChild(glyph);
    }
    const label = document.createElement("span");
    label.className = "gs-token-label";
    label.textContent = tag;
    chip.appendChild(label);

    chip.addEventListener("pointerdown", (event) => {
        event.preventDefault();
        const editor = chip.closest(".gs-token-editor");
        if (!editor) return;
        const range = document.createRange();
        const rect = chip.getBoundingClientRect();
        if (event.clientX < rect.left + rect.width / 2) range.setStartBefore(chip);
        else range.setStartAfter(chip);
        range.collapse(true);
        const sel = window.getSelection();
        sel.removeAllRanges();
        sel.addRange(range);
        editor.focus();
    });
    return chip;
}

/** 把 editor DOM 序列化成纯文本（chip → tag） */
function serializeTokenEditor(editor) {
    let out = "";
    const visit = (node) => {
        if (!node) return;
        if (node.nodeType === Node.TEXT_NODE) { out += node.textContent || ""; return; }
        if (node.nodeType !== Node.ELEMENT_NODE) return;
        if (node.classList && node.classList.contains(TOKEN_CLASS)) { out += node.dataset.tag || ""; return; }
        if (node.tagName === "BR") { out += "\n"; return; }
        for (const c of node.childNodes || []) visit(c);
        if (["DIV", "P"].includes(node.tagName)) out += "\n";
    };
    for (const c of editor.childNodes || []) visit(c);
    return out;
}

/** 用纯文本重建 editor DOM（tag → chip） */
function hydrateTokenEditor(editor, text, manifest) {
    editor.innerHTML = "";
    const source = String(text ?? "");
    let cursor = 0;
    TAG_RE.lastIndex = 0;
    let m;
    while ((m = TAG_RE.exec(source))) {
        if (m.index > cursor) {
            editor.appendChild(document.createTextNode(source.slice(cursor, m.index)));
        }
        const item = findRefByTag(manifest, m[0]);
        editor.appendChild(makeTokenChip(m[0], item));
        cursor = m.index + m[0].length;
    }
    if (cursor < source.length) editor.appendChild(document.createTextNode(source.slice(cursor)));
}

/**
 * 编辑器里是否存在「还没变成 chip 的纯文本 tag」。
 * 注意：chip 内部的 label 文本也长得像 tag，必须排除，否则每次按键都会误判成需要重建 DOM。
 */
function hasPlainTag(editor) {
    let node = null;
    try {
        const walker = document.createTreeWalker(editor, NodeFilter.SHOW_TEXT, null);
        node = walker.nextNode();
        while (node) {
            let host = node.parentElement;
            let insideChip = false;
            while (host && host !== editor) {
                if (host.classList && host.classList.contains(TOKEN_CLASS)) { insideChip = true; break; }
                host = host.parentElement;
            }
            if (!insideChip) {
                TAG_RE.lastIndex = 0;
                if (TAG_RE.test(node.textContent || "")) return true;
            }
            node = walker.nextNode();
        }
    } catch (e) { /* 忽略 */ }
    return false;
}

/** 编辑器内光标在序列化文本里的偏移 */
function serializedCaretOffset(editor) {
    const sel = window.getSelection();
    if (!sel || !sel.rangeCount || !editor.contains(sel.anchorNode)) return serializeTokenEditor(editor).length;
    const range = sel.getRangeAt(0);
    const pre = document.createRange();
    pre.selectNodeContents(editor);
    try { pre.setEnd(range.startContainer, range.startOffset); } catch (e) { return serializeTokenEditor(editor).length; }
    const tmp = document.createElement("div");
    tmp.appendChild(pre.cloneContents());
    return serializeTokenEditor(tmp).length;
}

/**
 * 获取编辑器内当前光标的视口坐标（用于 @ 面板精确定位到光标旁）。
 * 返回 { left, top, bottom } 或 null（光标不在编辑器内时）。
 */
function getCaretRect(editor) {
    const sel = window.getSelection();
    if (!sel || !sel.rangeCount) return null;
    const range = sel.getRangeAt(0).cloneRange();
    // 折叠到光标位置（range 可能跨节点，collapse 后只保留起点）
    range.collapse(false);
    let rect = range.getBoundingClientRect();
    // 有时候折叠后的 range 宽高为 0 且坐标在 (0,0)，说明取不到有效位置
    if (rect.width === 0 && rect.height === 0 && rect.top === 0 && rect.left === 0) {
        // 退回：用编辑器底部作为 fallback（旧行为）
        const er = editor.getBoundingClientRect();
        return { left: er.left, top: er.bottom, bottom: er.bottom + 4 };
    }
    return { left: rect.left, top: rect.top, bottom: rect.bottom };
}

function setCaretByOffset(editor, offset) {
    const target = Math.max(0, Number(offset) || 0);
    let seen = 0;
    const sel = window.getSelection();
    const range = document.createRange();
    const place = (node, at) => { range.setStart(node, at); range.collapse(true); sel.removeAllRanges(); sel.addRange(range); };
    const walk = (node) => {
        if (seen >= target) return true;
        if (node.nodeType === Node.TEXT_NODE) {
            const len = (node.textContent || "").length;
            if (seen + len >= target) { place(node, target - seen); seen = target; return true; }
            seen += len; return false;
        }
        if (node.nodeType !== Node.ELEMENT_NODE) return false;
        if (node.classList && node.classList.contains(TOKEN_CLASS)) {
            const tag = node.dataset.tag || "";
            if (seen + tag.length >= target) { range.setStartAfter(node); range.collapse(true); sel.removeAllRanges(); sel.addRange(range); seen = target; return true; }
            seen += tag.length; return false;
        }
        if (node.tagName === "BR") { seen += 1; return false; }
        for (const c of node.childNodes || []) { if (walk(c)) return true; }
        return false;
    };
    for (const c of editor.childNodes || []) { if (walk(c)) return; }
    range.selectNodeContents(editor); range.collapse(false); sel.removeAllRanges(); sel.addRange(range);
}

/** 光标前文本 */
function textBeforeCaret(editor) {
    const offset = serializedCaretOffset(editor);
    const full = serializeTokenEditor(editor);
    return { full, offset, before: full.slice(0, offset), after: full.slice(offset) };
}

/**
 * 核心：把 ComfyUI 的 textarea 包裹成「隐藏 textarea + contenteditable editor」。
 * textarea 保留在 DOM（只是视觉隐藏），ComfyUI widget 系统不会崩；
 * contenteditable 负责显示和编辑，textarea.value 始终是唯一数据源。
 */
function ensureTokenShell(textarea) {
    if (textarea.__gsTokenEditor) return textarea.__gsTokenEditor;
    injectTokenStyles();

    const wrap = document.createElement("div");
    wrap.className = "gs-token-wrap";
    const parent = textarea.parentNode;
    // 防御：onNodeCreated 阶段 textarea 可能还没挂到 DOM（parentNode 为 null），
    // 此时不能包裹，返回 null 让 attachMention 稍后重试。
    if (!parent) return null;
    parent.insertBefore(wrap, textarea);
    wrap.appendChild(textarea);
    textarea.classList.add("gs-token-source");
    textarea.setAttribute("tabindex", "-1");
    textarea.setAttribute("aria-hidden", "true");

    const editor = document.createElement("div");
    editor.className = "gs-token-editor";
    editor.contentEditable = "true";
    editor.dataset.placeholder = textarea.getAttribute("placeholder") || "在这里写你的想法。输入 @ 插入参考素材…";
    editor.setAttribute("role", "textbox");
    editor.setAttribute("aria-multiline", "true");
    editor.setAttribute("spellcheck", "false");
    wrap.appendChild(editor);

    textarea.__gsTokenEditor = editor;
    textarea.__gsTokenWrap = wrap;
    return editor;
}

function attachMention(node) {
    const widget = (node.widgets || []).find((w) => w.name === PROMPT_WIDGET);
    if (!widget) return;

    const ta = widget.element;
    if (!ta || ta.tagName !== "TEXTAREA") return;

    // onNodeCreated 阶段 textarea 的 element 往往还没挂到 DOM（parentNode 为 null），
    // 立即包裹会崩。这里做延迟重试，直到它真正进入文档树。
    if (!ta.parentNode) {
        if (ta.__gsAttachTries == null) ta.__gsAttachTries = 0;
        if (ta.__gsAttachTries > 40) return; // 最多等约 2 秒，避免死循环
        ta.__gsAttachTries += 1;
        requestAnimationFrame(() => attachMention(node));
        return;
    }

    ta.setAttribute("spellcheck", "false");
    const editor = ensureTokenShell(ta);
    if (!editor) {
        // 极端情况：仍然没挂载成功，稍后重试
        requestAnimationFrame(() => attachMention(node));
        return;
    }

    // 初始 hydrate
    const manifest = readManifest(node);
    hydrateTokenEditor(editor, widget.value || "", manifest);

    let activeItems = [];
    let activeIndex = 0;

    const syncToTextarea = () => {
        const text = serializeTokenEditor(editor);
        if (text === ta.value) return;
        ta.value = text;
        widget.value = text;
        ta.dispatchEvent(new Event("input", { bubbles: true }));
    };

    function commit(item) {
        const { full, before, offset } = textBeforeCaret(editor);
        const at = before.lastIndexOf("@");
        const start = at >= 0 ? at : offset;
        const next = full.slice(0, start) + item.tag + " " + full.slice(offset);
        const caret = start + item.tag.length + 1;

        upsertManifest(node, {
            tag: item.tag,
            type: item.type,
            name: item.name,
            subfolder: item.subfolder,
        });
        hydrateTokenEditor(editor, next, readManifest(node));
        setCaretByOffset(editor, caret);
        syncToTextarea();
        closePanel();
        editor.focus();
    }

    // ── 中文输入法（IME）保护 ────────────────────────────────────────────
    // 组合输入期间一旦重建编辑器 DOM，浏览器的拼音组合就会被打断，
    // 结果就是「按一个键出三个字母」（n你你hhaaooo 那种）。
    // 所以：组合期间只同步文本，不碰 DOM / 光标 / 面板，也不抢选字键。
    let composing = false;

    const refreshFromEditor = () => {
        try {
            // 只有出现「纯文本 <Picture N>」时才重建 DOM（粘贴 / 手敲 tag / 撤销）。
            // 已经是 chip 的 tag 不算，否则每敲一个字都重建，中文输入必炸。
            if (hasPlainTag(editor)) {
                const text = serializeTokenEditor(editor);
                const caret = serializedCaretOffset(editor);
                hydrateTokenEditor(editor, text, readManifest(node));
                setCaretByOffset(editor, caret);
            }
            syncToTextarea();

            const { before, offset } = textBeforeCaret(editor);
            const m = before.match(/@([^\s@]*)$/);
            if (!m) { closePanel(); return; }
            if (!PANEL) activeItems = collectReferences(node);
            openPanel(node, editor, activeItems, m[1] || "", commit);
        } catch (e) {
            console.warn("[GS] @ 面板刷新失败", e);
        }
    };

    editor.addEventListener("compositionstart", () => {
        composing = true;
        closePanel(); // 面板别挡住输入法候选词
    });
    editor.addEventListener("compositionupdate", () => { composing = true; });
    editor.addEventListener("compositionend", () => {
        composing = false;
        // 组合结束后的收尾 input 可能仍带 isComposing=true，
        // 延后一帧再处理，确保文字已经真正落进 DOM。
        requestAnimationFrame(() => { if (!composing) refreshFromEditor(); });
    });

    editor.addEventListener("input", (e) => {
        // 输入法组合中：只同步值到 textarea，绝不重建 DOM / 移动光标
        if (composing || e.isComposing) { syncToTextarea(); return; }
        refreshFromEditor();
    });

    editor.addEventListener("keydown", (e) => {
        // 阻止 ComfyUI 把 Ctrl+V 当成画布粘贴
        if ((e.ctrlKey || e.metaKey) && ["v", "c", "x"].includes(e.key.toLowerCase())) {
            e.stopPropagation();
        }
        // 组合中 / 输入法占位键（keyCode 229）：↑↓ 和 Enter 是选字键，必须放行
        if (composing || e.isComposing || e.keyCode === 229) return;
        if (!PANEL) return;
        const rows = [...PANEL.querySelectorAll(".gs-mention-row")];
        if (!rows.length) return;
        if (e.key === "ArrowDown" || e.key === "ArrowUp") {
            e.preventDefault();
            e.stopPropagation();
            activeIndex = (activeIndex + (e.key === "ArrowDown" ? 1 : -1) + rows.length) % rows.length;
            rows.forEach((c, i) => {
                c.style.background = i === activeIndex ? "rgba(44,221,118,0.18)" : "transparent";
            });
            rows[activeIndex].scrollIntoView({ block: "nearest" });
        } else if (e.key === "Enter" || e.key === "Tab") {
            e.preventDefault();
            e.stopPropagation();
            if (rows[activeIndex]) rows[activeIndex].dispatchEvent(new MouseEvent("mousedown"));
        } else if (e.key === "Escape") {
            e.preventDefault();
            e.stopPropagation();
            closePanel();
        }
    });

    editor.addEventListener("blur", () => {
        syncToTextarea();
        setTimeout(closePanel, 150);
    });
}

// ---------------------------------------------------------------- 排队前自愈（V3.4）
//
// 排队前按画布现状重建所有工作台的「参考素材清单」。
// 这样就算工作流文件里存的是垃圾值（旧版序列化错位产生的 "1"）、
// 素材换过文件没重新 @，也会在这一步被修正，后端才能真正把参考图喂给模型。

const _origQueuePrompt = app.queuePrompt;
if (typeof _origQueuePrompt === "function" && !app.__gsQueuePromptPatched) {
    app.__gsQueuePromptPatched = true;
    app.queuePrompt = async function (...args) {
        try {
            syncAllWorkbenchManifests();
        } catch (e) {
            console.warn("[GS] 排队前同步清单失败（不阻断排队）", e);
        }
        return _origQueuePrompt.apply(this, args);
    };
}

// 调试/测试钩子：控制台可手动触发一次清单重建
if (typeof window !== "undefined") {
    window.__gsSyncWorkbenchManifests = syncAllWorkbenchManifests;
}
