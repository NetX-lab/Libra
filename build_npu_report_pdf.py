from pathlib import Path
import html
import re

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate, Paragraph, PageBreak, Spacer,
    Table, TableStyle, Preformatted, KeepTogether
)


ROOT = Path("/Users/kevin/Documents/System Implementations/RL_Framework_npu")
SOURCE = ROOT / "Libra_NPU_Support_仿真工具报告.md"
OUTPUT = ROOT / "Libra_NPU_Support_仿真工具报告.pdf"
FONT_PATH = "/System/Library/Fonts/STHeiti Medium.ttc"
FENCE = chr(96) * 3


def register_fonts():
    pdfmetrics.registerFont(TTFont("HeitiSC", FONT_PATH, subfontIndex=1))


def inline_text(value):
    value = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", value)
    value = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", value)
    value = value.replace("**", "").replace("__", "")
    value = value.replace(chr(96), "")
    value = re.sub(r"<[^>]+>", "", value)
    return html.escape(value)


def boundary(line):
    s = line.strip()
    return (
        not s
        or s.startswith(("# ", "## ", "### ", FENCE, "|"))
        or re.match(r"^\s*(?:[-*] |\d+\. )", line) is not None
    )


def make_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ReportTitle", parent=styles["Title"], fontName="HeitiSC",
        fontSize=25, leading=34, alignment=TA_CENTER, textColor=colors.HexColor("#17365D"),
        spaceAfter=12,
    ))
    styles.add(ParagraphStyle(
        name="ReportSubtitle", parent=styles["Normal"], fontName="HeitiSC",
        fontSize=12, leading=20, alignment=TA_CENTER, textColor=colors.HexColor("#555555"),
        spaceAfter=20,
    ))
    styles.add(ParagraphStyle(
        name="ReportH1", parent=styles["Heading1"], fontName="HeitiSC",
        fontSize=16, leading=22, textColor=colors.HexColor("#17365D"),
        spaceBefore=13, spaceAfter=7, keepWithNext=True,
    ))
    styles.add(ParagraphStyle(
        name="ReportH2", parent=styles["Heading2"], fontName="HeitiSC",
        fontSize=12.5, leading=18, textColor=colors.HexColor("#1F4E79"),
        spaceBefore=9, spaceAfter=5, keepWithNext=True,
    ))
    styles.add(ParagraphStyle(
        name="ReportBody", parent=styles["BodyText"], fontName="HeitiSC",
        fontSize=9.6, leading=16, alignment=TA_LEFT, textColor=colors.HexColor("#222222"),
        spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name="ReportBullet", parent=styles["BodyText"], fontName="HeitiSC",
        fontSize=9.4, leading=15, leftIndent=14, firstLineIndent=-8,
        textColor=colors.HexColor("#222222"), spaceAfter=3,
    ))
    styles.add(ParagraphStyle(
        name="ReportSmall", parent=styles["BodyText"], fontName="HeitiSC",
        fontSize=8.5, leading=13, textColor=colors.HexColor("#555555"),
    ))
    styles.add(ParagraphStyle(
        name="ReportCode", parent=styles["Code"], fontName="HeitiSC",
        fontSize=7.2, leading=9, leftIndent=4, rightIndent=4,
    ))
    styles.add(ParagraphStyle(
        name="ReportTOC", parent=styles["BodyText"], fontName="HeitiSC",
        fontSize=10, leading=17, leftIndent=8, textColor=colors.HexColor("#333333"),
    ))
    return styles


def table_flowable(rows, styles):
    parsed = []
    for row in rows:
        cells = [c.strip() for c in row.strip().strip("|").split("|")]
        parsed.append(cells)
    if len(parsed) >= 2 and all(re.fullmatch(r"\s*:?-{2,}:?\s*", c) for c in parsed[1]):
        parsed.pop(1)
    if not parsed:
        return None
    width = len(parsed[0])
    normalized = []
    for row in parsed:
        row = row[:width] + [""] * max(0, width - len(row))
        normalized.append([Paragraph(inline_text(c), styles["ReportSmall"]) for c in row])
    if width == 2:
        col_widths = [55 * mm, 115 * mm]
    elif width == 3:
        col_widths = [42 * mm, 62 * mm, 66 * mm]
    elif width >= 4:
        col_widths = [34 * mm] + [141 * mm / (width - 1)] * (width - 1)
    else:
        col_widths = [170 * mm]
    table = Table(normalized, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D9EAF7")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#17365D")),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#A6A6A6")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F6F9FC")]),
    ]))
    return KeepTogether([table, Spacer(1, 4)])


def parse_markdown(path, styles):
    lines = path.read_text(encoding="utf-8").splitlines()
    story = []
    i = 0
    while i < len(lines):
        raw = lines[i]
        s = raw.strip()
        if not s:
            i += 1
            continue
        if s.startswith("# "):
            i += 1
            continue
        if s.startswith("## "):
            story.append(Paragraph(inline_text(s[3:]), styles["ReportH1"]))
            i += 1
            continue
        if s.startswith("### "):
            story.append(Paragraph(inline_text(s[4:]), styles["ReportH2"]))
            i += 1
            continue
        if s.startswith(FENCE):
            i += 1
            code = []
            while i < len(lines) and not lines[i].strip().startswith(FENCE):
                code.append(lines[i])
                i += 1
            i += 1
            box = Table([[Preformatted("\n".join(code), styles["ReportCode"])]], colWidths=[170 * mm])
            box.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F3F5F7")),
                ("BOX", (0, 0), (-1, -1), 0.4, colors.HexColor("#B7C3D0")),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]))
            story.extend([box, Spacer(1, 5)])
            continue
        if s.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(lines[i])
                i += 1
            tbl = table_flowable(rows, styles)
            if tbl:
                story.append(tbl)
            continue
        bullet = re.match(r"^\s*([-*]|\d+\.)\s+(.*)$", raw)
        if bullet:
            marker = bullet.group(1)
            story.append(Paragraph(f"{html.escape(marker)}  {inline_text(bullet.group(2))}", styles["ReportBullet"]))
            i += 1
            continue
        paragraph = [s]
        i += 1
        while i < len(lines) and not boundary(lines[i]):
            paragraph.append(lines[i].strip())
            i += 1
        story.append(Paragraph(inline_text(" ".join(paragraph)), styles["ReportBody"]))
    return story


def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D9E2F3"))
    canvas.line(18 * mm, 13 * mm, 192 * mm, 13 * mm)
    canvas.setFont("HeitiSC", 8)
    canvas.setFillColor(colors.HexColor("#777777"))
    canvas.drawString(18 * mm, 8 * mm, "Libra NPU_Support 仿真工具报告")
    canvas.drawRightString(192 * mm, 8 * mm, f"{doc.page}")
    canvas.restoreState()


def build():
    register_fonts()
    styles = make_styles()
    doc = BaseDocTemplate(
        str(OUTPUT), pagesize=A4,
        leftMargin=20 * mm, rightMargin=20 * mm,
        topMargin=18 * mm, bottomMargin=18 * mm,
        title="Libra NPU_Support 分支仿真工具报告",
        author="OpenAI Codex",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="normal")
    doc.addPageTemplates([PageTemplate(id="report", frames=frame, onPage=footer)])

    story = [Spacer(1, 35 * mm)]
    story.append(Paragraph("Libra NPU_Support 分支", styles["ReportTitle"]))
    story.append(Paragraph("仿真工具报告", styles["ReportTitle"]))
    story.append(Spacer(1, 12 * mm))
    story.append(Paragraph("面向 Agentic RL 后训练的资源规划、成本建模与动态重规划", styles["ReportSubtitle"]))
    meta = [
        ["分析对象", "NetX-lab/Libra · NPU_Support"],
        ["基线提交", "365f5e42d23b8641ac32e1adaa6feac6b5aa812b"],
        ["报告日期", "2026-09-09"],
        ["验证方式", "源码审查 + CPU 单元测试 + 全流程示例"],
    ]
    meta_table = Table([[Paragraph(inline_text(a), styles["ReportSmall"]),
                         Paragraph(inline_text(b), styles["ReportSmall"])] for a, b in meta],
                       colWidths=[32 * mm, 126 * mm])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#D9EAF7")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#A6A6A6")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 22 * mm))
    story.append(Paragraph(
        "结论：该分支的仿真工具以解析成本模型为主，兼容 Sailor/Vidur 外部模拟器，"
        "并通过 GRP 全局资源规划器将训练、Rollout、设备容量和运行时指标联动起来。"
        "CPU 验证链路已通过；真实 NPU 性能结论仍需在 Ascend 设备、CANN、torch-npu、"
        "HCCL 与 vLLM-Ascend 环境中实测校准。",
        styles["ReportBody"]))
    story.append(PageBreak())
    story.append(Paragraph("目录", styles["ReportH1"]))
    for entry in [
        "报告摘要", "1 项目定位与分析范围", "2 仿真工具总体架构",
        "3 仿真对象、输入与输出", "4 内置解析成本模型",
        "5 Sailor 与 Vidur 外部模拟器适配", "6 GRP 搜索与动态重规划",
        "7 NPU 运行链路与仿真关系", "8 安装与运行方法",
        "9 验证结果与复现实验", "10 输出结果与可观测性",
        "11 适用性、限制与风险", "12 改进与验收建议", "13 总结",
    ]:
        story.append(Paragraph("• " + inline_text(entry), styles["ReportTOC"]))
    story.append(PageBreak())
    story.extend(parse_markdown(SOURCE, styles))
    doc.build(story)
    print(OUTPUT)


if __name__ == "__main__":
    build()
