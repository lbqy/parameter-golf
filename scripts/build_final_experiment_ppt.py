#!/usr/bin/env python3
from __future__ import annotations

import html
import os
import zipfile
from datetime import datetime, timezone


OUT = "experiments/final_experiment_report.pptx"

SLIDE_W = 13_333_333
SLIDE_H = 7_500_000

BG = "F7F9FC"
INK = "152238"
MUTED = "56657A"
BLUE = "2563EB"
TEAL = "0F9F8F"
GREEN = "16A34A"
ORANGE = "EA580C"
RED = "DC2626"
PURPLE = "7C3AED"
LINE = "D7DEE8"
PANEL = "FFFFFF"


def esc(s: str) -> str:
    return html.escape(s, quote=True)


def emu(inch: float) -> int:
    return int(inch * 914400)


def tx_box(
    x: int,
    y: int,
    w: int,
    h: int,
    text: str,
    size: int = 22,
    color: str = INK,
    bold: bool = False,
    align: str = "l",
    font: str = "Microsoft YaHei",
    name: str = "Text",
) -> str:
    b = "<a:b/>" if bold else ""
    paras = []
    for line in text.split("\n"):
        paras.append(
            f"""
            <a:p>
              <a:pPr algn="{align}"/>
              <a:r>
                <a:rPr lang="zh-CN" sz="{size * 100}">{b}<a:solidFill><a:srgbClr val="{color}"/></a:solidFill><a:latin typeface="{font}"/><a:ea typeface="{font}"/></a:rPr>
                <a:t>{esc(line)}</a:t>
              </a:r>
            </a:p>"""
        )
    return f"""
    <p:sp>
      <p:nvSpPr><p:cNvPr id="{tx_box.next_id()}" name="{esc(name)}"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>
      <p:spPr><a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{w}" cy="{h}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/><a:ln><a:noFill/></a:ln></p:spPr>
      <p:txBody><a:bodyPr wrap="square" anchor="t"/><a:lstStyle/>{''.join(paras)}</p:txBody>
    </p:sp>"""


def _next_id():
    _next_id.i += 1
    return _next_id.i


_next_id.i = 1
tx_box.next_id = _next_id


def rect(
    x: int,
    y: int,
    w: int,
    h: int,
    fill: str = PANEL,
    line: str = LINE,
    radius: str = "roundRect",
    name: str = "Rect",
) -> str:
    return f"""
    <p:sp>
      <p:nvSpPr><p:cNvPr id="{tx_box.next_id()}" name="{esc(name)}"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>
      <p:spPr>
        <a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{w}" cy="{h}"/></a:xfrm>
        <a:prstGeom prst="{radius}"><a:avLst/></a:prstGeom>
        <a:solidFill><a:srgbClr val="{fill}"/></a:solidFill>
        <a:ln w="12000"><a:solidFill><a:srgbClr val="{line}"/></a:solidFill></a:ln>
      </p:spPr>
    </p:sp>"""


def line(x1: int, y1: int, x2: int, y2: int, color: str = LINE, width: int = 18000) -> str:
    return f"""
    <p:cxnSp>
      <p:nvCxnSpPr><p:cNvPr id="{tx_box.next_id()}" name="Line"/><p:cNvCxnSpPr/><p:nvPr/></p:nvCxnSpPr>
      <p:spPr>
        <a:xfrm><a:off x="{min(x1, x2)}" y="{min(y1, y2)}"/><a:ext cx="{abs(x2-x1)}" cy="{abs(y2-y1)}"/></a:xfrm>
        <a:prstGeom prst="line"><a:avLst/></a:prstGeom>
        <a:ln w="{width}"><a:solidFill><a:srgbClr val="{color}"/></a:solidFill></a:ln>
      </p:spPr>
    </p:cxnSp>"""


def bullet_list(
    x: int,
    y: int,
    w: int,
    h: int,
    bullets: list[str],
    size: int = 20,
    color: str = INK,
    gap: int = 1,
) -> str:
    paras = []
    for b in bullets:
        paras.append(
            f"""
            <a:p>
              <a:pPr marL="260000" indent="-180000"><a:buChar char="•"/></a:pPr>
              <a:r><a:rPr lang="zh-CN" sz="{size * 100}"><a:solidFill><a:srgbClr val="{color}"/></a:solidFill><a:latin typeface="Microsoft YaHei"/><a:ea typeface="Microsoft YaHei"/></a:rPr><a:t>{esc(b)}</a:t></a:r>
            </a:p>"""
        )
        for _ in range(gap - 1):
            paras.append("<a:p/>")
    return f"""
    <p:sp>
      <p:nvSpPr><p:cNvPr id="{tx_box.next_id()}" name="Bullets"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>
      <p:spPr><a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{w}" cy="{h}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/><a:ln><a:noFill/></a:ln></p:spPr>
      <p:txBody><a:bodyPr wrap="square" anchor="t"/><a:lstStyle/>{''.join(paras)}</p:txBody>
    </p:sp>"""


def header(title: str, subtitle: str = "") -> str:
    s = tx_box(emu(0.55), emu(0.28), emu(12.2), emu(0.45), title, size=24, bold=True, color=INK)
    if subtitle:
        s += tx_box(emu(0.57), emu(0.76), emu(11.8), emu(0.28), subtitle, size=11, color=MUTED)
    s += line(emu(0.55), emu(1.03), emu(12.75), emu(1.03), color=LINE, width=10000)
    return s


def slide_xml(shapes: str) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld>
    <p:bg><p:bgPr><a:solidFill><a:srgbClr val="{BG}"/></a:solidFill><a:effectLst/></p:bgPr></p:bg>
    <p:spTree>
      <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
      <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>
      {shapes}
    </p:spTree>
  </p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sld>"""


def table(x, y, col_ws, row_h, rows, header_fill=BLUE):
    shapes = ""
    for r, row in enumerate(rows):
        cx = x
        for c, val in enumerate(row):
            fill = header_fill if r == 0 else PANEL
            txt_color = "FFFFFF" if r == 0 else INK
            shapes += rect(cx, y + r * row_h, col_ws[c], row_h, fill=fill, line=LINE, radius="rect")
            shapes += tx_box(cx + emu(0.08), y + r * row_h + emu(0.05), col_ws[c] - emu(0.12), row_h - emu(0.08), str(val), size=12 if r else 12, color=txt_color, bold=(r == 0))
            cx += col_ws[c]
    return shapes


def phase_card(x, y, w, h, title, value, body, color):
    return (
        rect(x, y, w, h, fill=PANEL, line=color)
        + rect(x, y, emu(0.16), h, fill=color, line=color, radius="rect")
        + tx_box(x + emu(0.28), y + emu(0.15), w - emu(0.42), emu(0.25), title, size=13, bold=True, color=color)
        + tx_box(x + emu(0.28), y + emu(0.48), w - emu(0.42), emu(0.34), value, size=18, bold=True, color=INK)
        + tx_box(x + emu(0.28), y + emu(0.95), w - emu(0.42), h - emu(1.0), body, size=10, color=MUTED)
    )


def bar_chart(x, y, w, h, data):
    max_v = 1.24
    min_v = 1.06
    shapes = tx_box(x, y - emu(0.25), w, emu(0.2), "BPB 越低越好", size=11, color=MUTED)
    row_h = h // len(data)
    label_w = emu(2.7)
    chart_w = w - label_w - emu(0.5)
    for i, (name, val, color) in enumerate(data):
        yy = y + i * row_h
        shapes += tx_box(x, yy + emu(0.02), label_w, row_h, name, size=11, color=INK)
        # Lower BPB gets longer progress-to-target bar.
        frac = (max_v - val) / (max_v - min_v)
        bw = int(chart_w * max(0.02, min(1.0, frac)))
        shapes += rect(x + label_w, yy + emu(0.08), chart_w, emu(0.16), fill="E8EEF7", line="E8EEF7", radius="rect")
        shapes += rect(x + label_w, yy + emu(0.08), bw, emu(0.16), fill=color, line=color, radius="rect")
        shapes += tx_box(x + label_w + chart_w + emu(0.12), yy - emu(0.01), emu(0.9), row_h, f"{val:.4f}", size=11, bold=True, color=color)
    return shapes


slides: list[str] = []

# 1 title
slides.append(slide_xml(
    rect(emu(0.55), emu(0.55), emu(12.25), emu(6.35), fill=PANEL, line="E5EAF2")
    + tx_box(emu(0.95), emu(1.08), emu(11.4), emu(0.65), "Parameter Golf 实验汇报", size=34, bold=True, color=INK)
    + tx_box(emu(0.98), emu(1.85), emu(10.8), emu(0.45), "单卡 H100 / 1 小时 / 16MB 约束下的小模型压缩与优化探索", size=18, color=MUTED)
    + rect(emu(0.98), emu(2.75), emu(3.3), emu(1.25), fill="EFF6FF", line="BFDBFE")
    + tx_box(emu(1.2), emu(2.97), emu(2.8), emu(0.3), "最终 post-TTT BPB", size=13, color=MUTED)
    + tx_box(emu(1.2), emu(3.33), emu(2.8), emu(0.45), "1.08179851", size=24, bold=True, color=BLUE)
    + rect(emu(4.65), emu(2.75), emu(3.3), emu(1.25), fill="F0FDF4", line="BBF7D0")
    + tx_box(emu(4.87), emu(2.97), emu(2.8), emu(0.3), "相对第四阶段改善", size=13, color=MUTED)
    + tx_box(emu(4.87), emu(3.33), emu(2.8), emu(0.45), "−0.0834 BPB", size=24, bold=True, color=GREEN)
    + rect(emu(8.32), emu(2.75), emu(3.3), emu(1.25), fill="FFF7ED", line="FED7AA")
    + tx_box(emu(8.54), emu(2.97), emu(2.8), emu(0.3), "Artifact bytes", size=13, color=MUTED)
    + tx_box(emu(8.54), emu(3.33), emu(2.8), emu(0.45), "15,924,510", size=24, bold=True, color=ORANGE)
    + tx_box(emu(0.98), emu(5.95), emu(11.5), emu(0.3), "基于 experiments/final_experiment_report.md 自动生成", size=11, color=MUTED)
))

# 2 problem
slides.append(slide_xml(
    header("任务不是单纯训低 loss", "核心矛盾：训练质量、量化鲁棒性、提交容量必须同时成立")
    + rect(emu(0.75), emu(1.35), emu(3.8), emu(4.8), fill=PANEL, line="BFDBFE")
    + tx_box(emu(1.05), emu(1.65), emu(3.2), emu(0.35), "训练阶段", size=20, bold=True, color=BLUE)
    + bullet_list(emu(1.0), emu(2.15), emu(3.2), emu(2.0), ["pre-quant BPB", "模型本身是否学得好", "扩大模型常能改善这里"], size=15)
    + rect(emu(4.85), emu(1.35), emu(3.8), emu(4.8), fill=PANEL, line="BBF7D0")
    + tx_box(emu(5.15), emu(1.65), emu(3.2), emu(0.35), "导出阶段", size=20, bold=True, color=GREEN)
    + bullet_list(emu(5.1), emu(2.15), emu(3.2), emu(2.0), ["roundtrip BPB", "量化/压缩后质量还剩多少", "最终比较以这里为准"], size=15)
    + rect(emu(8.95), emu(1.35), emu(3.65), emu(4.8), fill=PANEL, line="FED7AA")
    + tx_box(emu(9.25), emu(1.65), emu(3.0), emu(0.35), "合规阶段", size=20, bold=True, color=ORANGE)
    + bullet_list(emu(9.2), emu(2.15), emu(3.0), emu(2.0), ["≤ 16MB", "压不下就不能提交", "压得下但变差也不行"], size=15)
    + tx_box(emu(1.0), emu(6.45), emu(11.3), emu(0.35), "结论：每个改动都要同时问三件事：学得好吗？压完还好吗？文件够小吗？", size=16, bold=True, color=INK, align="c")
))

# 3 road map
slides.append(slide_xml(
    header("方法演进：从局部优化到系统复现", "不是一开始知道答案，而是在大量负结果中逐步收缩搜索空间")
    + phase_card(emu(0.55), emu(1.35), emu(2.35), emu(4.75), "阶段一", "确定基座", "SP1024 → SP8192\nseq1024 → seq4096\n发现：质量强但超 16MB", BLUE)
    + phase_card(emu(3.05), emu(1.35), emu(2.35), emu(4.75), "阶段二", "压进 16MB", "RTN 兜底\nGPTQ fresh eval\nLQER 建立可信基线", TEAL)
    + phase_card(emu(5.55), emu(1.35), emu(2.35), emu(4.75), "阶段三", "训练动态", "QK gain\nembedding LR\nrecurrence + Muon", GREEN)
    + phase_card(emu(8.05), emu(1.35), emu(2.35), emu(4.75), "阶段四", "组件移植", "SparseGate / LeakyReLU² / PolarNS\n单项多负，组合小正", ORANGE)
    + phase_card(emu(10.55), emu(1.35), emu(2.25), emu(4.75), "阶段五", "records stack", "CaseOps + lrzip\nlegal TTT\nseed42 + warmdown 0.95", PURPLE)
))

# 4 Phase 1
slides.append(slide_xml(
    header("阶段一：先回答“训练基座应该长什么样”", "SP8192 + seq4096 后补确认，虽然后做，但逻辑上属于基座选择")
    + table(emu(0.7), emu(1.35), [emu(3.4), emu(2.0), emu(2.0), emu(5.0)], emu(0.58), [
        ["配置", "BPB", "状态", "判断"],
        ["SP1024 / seq1024", "1.2320", "合规", "baseline"],
        ["SP1024 / seq4096", "1.2113", "合规", "长上下文有效"],
        ["SP8192 / seq2048 / int8", "1.1815", "超 16MB", "质量跃迁，但装不下"],
        ["SP8192 / seq4096 / int8", "1.1807", "超 16MB", "更强训练基座"],
    ])
    + rect(emu(0.9), emu(5.0), emu(11.5), emu(1.2), fill="EFF6FF", line="BFDBFE")
    + tx_box(emu(1.15), emu(5.22), emu(11.0), emu(0.75), "关键转向：问题不再是“要不要用大词表”，而是“如何把 SP8192 + seq4096 的质量压进 16MB”。", size=19, bold=True, color=BLUE, align="c")
))

# 5 Phase 2
slides.append(slide_xml(
    header("阶段二：量化链路先踩坑，再变可信", "早期 GPTQ 异常、RTN 兜底、fresh roundtrip 修正，是后续实验可信的前提")
    + rect(emu(0.75), emu(1.28), emu(3.75), emu(4.9), fill=PANEL, line="FCA5A5")
    + tx_box(emu(1.0), emu(1.55), emu(3.2), emu(0.35), "1. 早期 GPTQ 反常", size=17, bold=True, color=RED)
    + bullet_list(emu(0.95), emu(2.05), emu(3.25), emu(2.7), ["matrix6/embed8 到 1.2888", "embed7 退化到 1.5161", "更高 bit 也异常", "定位：旧 checkpoint / eval 污染"], size=13)
    + rect(emu(4.85), emu(1.28), emu(3.75), emu(4.9), fill=PANEL, line="FED7AA")
    + tx_box(emu(5.1), emu(1.55), emu(3.2), emu(0.35), "2. RTN 兜底", size=17, bold=True, color=ORANGE)
    + bullet_list(emu(5.05), emu(2.05), emu(3.25), emu(2.7), ["packed RTN + brotli 可合规", "约 1.1892 BPB", "mixed bit 方向正确", "但质量上限不够"], size=13)
    + rect(emu(8.95), emu(1.28), emu(3.65), emu(4.9), fill=PANEL, line="BBF7D0")
    + tx_box(emu(9.2), emu(1.55), emu(3.1), emu(0.35), "3. fresh roundtrip", size=17, bold=True, color=GREEN)
    + bullet_list(emu(9.15), emu(2.05), emu(3.05), emu(2.7), ["量化后重建模型", "重新加载 artifact", "重新 compile 再 eval", "GPTQ + LQER 到 1.1766"], size=13)
    + tx_box(emu(0.9), emu(6.45), emu(11.6), emu(0.3), "经验：量化实验的第一目标不是刷分，而是让 roundtrip 评估可信。", size=16, bold=True, color=INK, align="c")
))

# 6 Phase 3
slides.append(slide_xml(
    header("阶段三：训练动态是小收益叠加", "从低比特基座出发，逐步确认哪些训练改动能被量化保留下来")
    + table(emu(0.65), emu(1.22), [emu(2.8), emu(1.55), emu(4.0), emu(4.0)], emu(0.5), [
        ["步骤", "BPB", "方法", "理解"],
        ["低比特基座", "1.1766", "GPTQ + LQER", "可比较起点"],
        ["QK gain", "1.1761", "QK_GAIN_INIT=5.0", "改善 attention 初始尺度"],
        ["hparam stack", "1.1748", "beta / clip / LR", "短训更稳"],
        ["tied embed LR", "1.1716", "TIED_EMBED_LR=0.04", "大词表 embedding 很敏感"],
        ["recurrence", "1.1701", "L3-L5 extra pass", "增加有效深度"],
        ["Muon + recurrence", "1.1684", "Muon momentum 0.97", "矩阵更新更好"],
        ["综合方案", "1.1674", "start 0.30 + LQER top4", "第三阶段最佳"],
    ], header_fill=TEAL)
    + rect(emu(0.85), emu(6.0), emu(11.65), emu(0.8), fill="F0FDF4", line="BBF7D0")
    + tx_box(emu(1.05), emu(6.16), emu(11.2), emu(0.35), "负结果同样有用：coprime loader、早期 warmdown/min_lr、lm-head-only TTT 都帮助排除低价值路线。", size=14, bold=True, color=GREEN, align="c")
))

# 7 Phase 4
slides.append(slide_xml(
    header("阶段四：records 组件不能机械拆开", "单项移植多为负，组合只小幅刷新，提示应转向系统 stack")
    + table(emu(0.7), emu(1.25), [emu(4.0), emu(2.0), emu(5.4)], emu(0.54), [
        ["探索方向", "结果", "判断"],
        ["LeakyReLU² 单项", "变差", "激活函数不能孤立替换"],
        ["partial RoPE", "变差", "当前训练栈不适配"],
        ["Polar NS 单项", "变差", "稳定性收益不足"],
        ["SmearGate 单项", "未超过基座", "局部平滑不够"],
        ["SparseGate 单项", "接近但不够", "有信号，不完整"],
        ["SparseGate + LeakyReLU² + PolarNS", "1.1652", "第四阶段最佳，但只改善 0.0021"],
    ], header_fill=ORANGE)
    + tx_box(emu(0.9), emu(6.25), emu(11.5), emu(0.35), "结论：records 的收益来自数据、结构、训练、压缩、TTT 的协同，而不是单个开关。", size=16, bold=True, color=ORANGE, align="c")
))

# 8 Search branches
slides.append(slide_xml(
    header("第五阶段前半：不是预先知道 records 更强", "并行验证根目录备份线、容量线、CaseOps 静态线后，才转向 records 主线")
    + table(emu(0.55), emu(1.15), [emu(3.7), emu(2.3), emu(6.2)], emu(0.48), [
        ["分支", "代表结果", "判断"],
        ["普通 SP8192 + 786k batch", "1.1647", "第一个小刷新，但仍是小收益"],
        ["普通 SP8192 + 917k batch", "1.1636", "根目录路线最佳小刷新"],
        ["seq8192", "1.1722", "上下文过长，step 损失大"],
        ["10 层模型", "1.1643", "合规但收益有限"],
        ["MLP3 加宽", "1.1532 / 超 16MB", "质量强，但容量不合规"],
        ["MLP3 容量修复", "1.1658", "embed6 损失太大"],
        ["CaseOps 静态训练", "1.1738 / 1.1744", "CaseOps 需与 records/TTT 配套"],
    ], header_fill=PURPLE)
))

# 9 records stack
slides.append(slide_xml(
    header("records stack：六层组合拳", "Phase 5 的跃迁来自系统组合，而不是单点技巧")
    + phase_card(emu(0.55), emu(1.18), emu(3.8), emu(1.55), "数据层", "CaseOps", "SP8192 tokenizer\nbyte sidecar / BOS boundary", BLUE)
    + phase_card(emu(4.75), emu(1.18), emu(3.8), emu(1.55), "结构层", "11L / MLP4 / lane", "LeakyReLU² / XSA\nparallel decoder / recurrence", TEAL)
    + phase_card(emu(8.95), emu(1.18), emu(3.8), emu(1.55), "信息流", "SparseGate / SmearGate", "head-output gate\nBOS leak fix", GREEN)
    + phase_card(emu(0.55), emu(3.25), emu(3.8), emu(1.55), "训练层", "PolarNS Muon", "QK gain / warmdown\nseed42 / grad clip / EMA", ORANGE)
    + phase_card(emu(4.75), emu(3.25), emu(3.8), emu(1.55), "压缩层", "GPTQ + LQER + lrzip", "int6 matrix / int7 embed\nper-group compression", PURPLE)
    + phase_card(emu(8.95), emu(3.25), emu(3.8), emu(1.55), "评估层", "legal phased TTT", "score-before-update\nLoRA adapters", RED)
    + tx_box(emu(0.9), emu(6.05), emu(11.6), emu(0.45), "本地运行是 fallback 版：缺 FA3 / TensorDescriptor，训练退 fixed sequence loader，因此剩余 gap 主要在吞吐和训练质量。", size=14, bold=True, color=INK, align="c")
))

# 10 TTT / training quality
slides.append(slide_xml(
    header("Phase 5 后半：TTT 确认平台，训练质量成为主杠杆", "先证明 TTT 有大收益，再证明 TTT 微扫不是继续下降的关键")
    + rect(emu(0.75), emu(1.25), emu(5.75), emu(4.7), fill=PANEL, line="BFDBFE")
    + tx_box(emu(1.05), emu(1.55), emu(5.2), emu(0.35), "TTT 微扫", size=19, bold=True, color=BLUE)
    + bullet_list(emu(1.0), emu(2.08), emu(5.1), emu(2.9), [
        "默认 TTT 已带来约 0.0149 BPB",
        "prefix docs / phase 数 / global lr 都扫过",
        "最佳邻域只改善约 5e-5 BPB",
        "迁移到更强 artifact 后几乎无收益",
    ], size=14)
    + rect(emu(6.85), emu(1.25), emu(5.75), emu(4.7), fill=PANEL, line="BBF7D0")
    + tx_box(emu(7.15), emu(1.55), emu(5.2), emu(0.35), "训练质量扫参", size=19, bold=True, color=GREEN)
    + bullet_list(emu(7.1), emu(2.08), emu(5.1), emu(2.9), [
        "seed42 明显优于 seed0/1234",
        "延后 loop 增加 step 但 BPB 变差",
        "batch917k 在 records stack 中负向",
        "warmdown 0.95 成为最终甜点",
    ], size=14)
    + tx_box(emu(0.9), emu(6.35), emu(11.6), emu(0.35), "结论：最终方案不是 TTT 微调出来的，而是更好的 no-TTT artifact + 默认 legal TTT。", size=15, bold=True, color=INK, align="c")
))

# 11 Results chart
slides.append(slide_xml(
    header("结果演进：从 1.23 到 1.08", "每次大跳跃都对应一次问题重定义")
    + bar_chart(emu(0.95), emu(1.35), emu(11.2), emu(4.65), [
        ("SP1024 baseline", 1.2320, RED),
        ("SP1024 seq4096", 1.2113, ORANGE),
        ("SP8192 int8 超限", 1.1807, ORANGE),
        ("GPTQ + LQER 基座", 1.1766, TEAL),
        ("训练动态综合", 1.1674, GREEN),
        ("records 组件组合", 1.1652, GREEN),
        ("records no-TTT", 1.0965, BLUE),
        ("records post-TTT", 1.0818, PURPLE),
    ])
    + rect(emu(0.95), emu(6.18), emu(11.2), emu(0.55), fill="F5F3FF", line="DDD6FE")
    + tx_box(emu(1.1), emu(6.31), emu(10.9), emu(0.25), "第四阶段到最终：1.16522320 → 1.08179851，改善 0.08342469 BPB", size=15, bold=True, color=PURPLE, align="c")
))

# 12 Lessons
slides.append(slide_xml(
    header("思考体会", "对小模型压缩挑战，最重要的是完整链路与负结果")
    + bullet_list(emu(0.9), emu(1.35), emu(11.6), emu(4.4), [
        "不能只看 pre-quant BPB：强模型如果压不进 16MB，就不能作为最终答案。",
        "负结果很有价值：它们帮助排除 loader、TTT 微参、导出微参、loop 延后等低收益方向。",
        "records 方法不能机械拆开：单个 gate/activation/optimizer 开关无法复现系统收益。",
        "工程环境也是算法的一部分：FA3、fused MLP、lrzip、byte sidecar 都会影响最终分数。",
        "局部微调有边界：真正跃迁来自数据、结构、训练、压缩、TTT 的联合设计。",
    ], size=20, gap=1)
))

# 13 Next
slides.append(slide_xml(
    header("后续工作", "当前差距主要来自单卡 fallback 训练质量与高吞吐环境")
    + rect(emu(0.8), emu(1.28), emu(11.75), emu(4.85), fill=PANEL, line="D8B4FE")
    + bullet_list(emu(1.05), emu(1.65), emu(11.1), emu(3.9), [
        "固化 warmdown 0.95 最终方案复现链路：CaseOps sidecar、records fallback、real lrzip、默认 TTT。",
        "优先解决 FA3 / fused MLP / TensorDescriptor 环境 blocker，恢复 records 高吞吐形态。",
        "环境恢复后先重跑 seed42 + warmdown 0.95，再做少量 seed / warmdown 邻域验证。",
        "在出现更强 no-TTT artifact 前，不继续投入大量 TTT 微扫或 export-only 微扫。",
        "若 kernel 短期不可解，考虑在根目录脚本中实现 doc-boundary 高吞吐路径。",
    ], size=17)
    + tx_box(emu(0.9), emu(6.42), emu(11.6), emu(0.3), "一句话结论：最终答案不是某个神奇参数，而是一条被大量淘汰实验筛出来的系统路线。", size=16, bold=True, color=PURPLE, align="c")
))


def content_types(n: int) -> str:
    slide_overrides = "\n".join(
        f'<Override PartName="/ppt/slides/slide{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>'
        for i in range(1, n + 1)
    )
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>
  <Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>
  <Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>
  <Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>
  {slide_overrides}
</Types>"""


def root_rels() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>"""


def presentation_xml(n: int) -> str:
    sld_ids = "\n".join(
        f'<p:sldId id="{255+i}" r:id="rId{i}"/>' for i in range(1, n + 1)
    )
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId{n+1}"/></p:sldMasterIdLst>
  <p:sldIdLst>{sld_ids}</p:sldIdLst>
  <p:sldSz cx="{SLIDE_W}" cy="{SLIDE_H}" type="wide"/>
  <p:notesSz cx="6858000" cy="9144000"/>
  <p:defaultTextStyle>
    <a:defPPr><a:defRPr lang="zh-CN"/></a:defPPr>
  </p:defaultTextStyle>
</p:presentation>"""


def presentation_rels(n: int) -> str:
    rels = [
        f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide{i}.xml"/>'
        for i in range(1, n + 1)
    ]
    rels.append(f'<Relationship Id="rId{n+1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/>')
    rels.append(f'<Relationship Id="rId{n+2}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="theme/theme1.xml"/>')
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  {' '.join(rels)}
</Relationships>"""


def slide_rels() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
</Relationships>"""


def master_xml() -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldMaster xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld><p:bg><p:bgPr><a:solidFill><a:srgbClr val="{BG}"/></a:solidFill><a:effectLst/></p:bgPr></p:bg><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld>
  <p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/>
  <p:sldLayoutIdLst><p:sldLayoutId id="1" r:id="rId1"/></p:sldLayoutIdLst>
  <p:txStyles><p:titleStyle/><p:bodyStyle/><p:otherStyle/></p:txStyles>
</p:sldMaster>"""


def master_rels() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/>
</Relationships>"""


def layout_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldLayout xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" type="blank" preserve="1">
  <p:cSld name="Blank"><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sldLayout>"""


def layout_rels() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="../slideMasters/slideMaster1.xml"/>
</Relationships>"""


def theme_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="ParameterGolf">
  <a:themeElements>
    <a:clrScheme name="Custom">
      <a:dk1><a:srgbClr val="152238"/></a:dk1><a:lt1><a:srgbClr val="FFFFFF"/></a:lt1>
      <a:dk2><a:srgbClr val="56657A"/></a:dk2><a:lt2><a:srgbClr val="F7F9FC"/></a:lt2>
      <a:accent1><a:srgbClr val="2563EB"/></a:accent1><a:accent2><a:srgbClr val="0F9F8F"/></a:accent2>
      <a:accent3><a:srgbClr val="16A34A"/></a:accent3><a:accent4><a:srgbClr val="EA580C"/></a:accent4>
      <a:accent5><a:srgbClr val="7C3AED"/></a:accent5><a:accent6><a:srgbClr val="DC2626"/></a:accent6>
      <a:hlink><a:srgbClr val="2563EB"/></a:hlink><a:folHlink><a:srgbClr val="7C3AED"/></a:folHlink>
    </a:clrScheme>
    <a:fontScheme name="Custom"><a:majorFont><a:latin typeface="Microsoft YaHei"/><a:ea typeface="Microsoft YaHei"/></a:majorFont><a:minorFont><a:latin typeface="Microsoft YaHei"/><a:ea typeface="Microsoft YaHei"/></a:minorFont></a:fontScheme>
    <a:fmtScheme name="Custom"><a:fillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:fillStyleLst><a:lnStyleLst><a:ln w="9525"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln></a:lnStyleLst><a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst><a:bgFillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:bgFillStyleLst></a:fmtScheme>
  </a:themeElements>
  <a:objectDefaults/><a:extraClrSchemeLst/>
</a:theme>"""


def core_xml() -> str:
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>Parameter Golf 单卡 H100 实验汇报</dc:title>
  <dc:creator>Codex</dc:creator>
  <cp:lastModifiedBy>Codex</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>
</cp:coreProperties>"""


def app_xml(n: int) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Codex</Application><PresentationFormat>Widescreen</PresentationFormat><Slides>{n}</Slides><Company></Company><AppVersion>1.0</AppVersion>
</Properties>"""


def build():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    n = len(slides)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types(n))
        z.writestr("_rels/.rels", root_rels())
        z.writestr("docProps/core.xml", core_xml())
        z.writestr("docProps/app.xml", app_xml(n))
        z.writestr("ppt/presentation.xml", presentation_xml(n))
        z.writestr("ppt/_rels/presentation.xml.rels", presentation_rels(n))
        z.writestr("ppt/slideMasters/slideMaster1.xml", master_xml())
        z.writestr("ppt/slideMasters/_rels/slideMaster1.xml.rels", master_rels())
        z.writestr("ppt/slideLayouts/slideLayout1.xml", layout_xml())
        z.writestr("ppt/slideLayouts/_rels/slideLayout1.xml.rels", layout_rels())
        z.writestr("ppt/theme/theme1.xml", theme_xml())
        for i, s in enumerate(slides, 1):
            z.writestr(f"ppt/slides/slide{i}.xml", s)
            z.writestr(f"ppt/slides/_rels/slide{i}.xml.rels", slide_rels())
    print(OUT)


if __name__ == "__main__":
    build()
