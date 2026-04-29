import csv
import io
from typing import Dict, List

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


def _safe_text(value: object) -> str:
    return str(value or "").strip()


def _build_entity_rows(entities: List[Dict[str, object]]) -> List[List[str]]:
    rows = [["Entity", "Type", "Confidence"]]
    for entity in entities:
        rows.append(
            [
                _safe_text(entity.get("text")),
                _safe_text(entity.get("label")),
                f"{float(entity.get('confidence', 0.0)):.2f}%",
            ]
        )
    if len(rows) == 1:
        rows.append(["No medical entities detected", "-", "-"])
    return rows


def _escape_pdf_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _wrap_text(text: str, max_chars: int = 95) -> List[str]:
    normalized = " ".join(_safe_text(text).split())
    if not normalized:
        return [""]

    words = normalized.split(" ")
    lines: List[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _build_pdf_lines(patient_data: Dict[str, object]) -> List[str]:
    lines = [
        f"Clinical Report - Patient {patient_data['patient_id']}",
        "",
        "1. Patient Information",
        f"Patient ID: {_safe_text(patient_data['patient_id'])}",
        "",
        "2. Clinical Text",
    ]
    lines.extend(_wrap_text(_safe_text(patient_data.get("raw_text")) or "No clinical text available."))
    lines.extend(["", "3. Extracted Entities", "Entity | Type | Confidence"])

    entities = patient_data.get("entities", [])
    if entities:
        for entity in entities:
            lines.extend(
                _wrap_text(
                    f"{_safe_text(entity.get('text'))} | {_safe_text(entity.get('label'))} | "
                    f"{float(entity.get('confidence', 0.0)):.2f}%"
                )
            )
    else:
        lines.append("No medical entities detected | - | -")

    risk = patient_data.get("risk", {})
    lines.extend(
        [
            "",
            "4. Risk Assessment",
            f"Risk Level: {_safe_text(risk.get('risk_level'))}",
        ]
    )
    lines.extend(_wrap_text(f"Explanation: {_safe_text(risk.get('explanation'))}"))
    lines.extend(["", "5. Insights"])

    insights = patient_data.get("insights", [])
    if insights:
        for insight in insights:
            lines.extend(_wrap_text(f"- {_safe_text(insight)}"))
    else:
        lines.append("No additional insights available.")

    lines.extend(["", "6. Summary"])
    lines.extend(_wrap_text(_safe_text(patient_data.get("summary")) or "No summary available."))
    return lines


def _build_simple_pdf(patient_data: Dict[str, object]) -> bytes:
    page_width = 595
    page_height = 842
    margin_left = 50
    start_y = 790
    line_height = 16
    max_lines_per_page = 44
    lines = _build_pdf_lines(patient_data)
    pages = [
        lines[index : index + max_lines_per_page]
        for index in range(0, len(lines), max_lines_per_page)
    ] or [[]]

    objects: List[bytes] = []

    def add_object(content: bytes) -> int:
        objects.append(content)
        return len(objects)

    font_id = add_object(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    page_ids = []

    for page_lines in pages:
        stream_parts = [b"BT\n/F1 11 Tf\n"]
        current_y = start_y
        for line in page_lines:
            safe_line = _escape_pdf_text(line)
            stream_parts.append(f"1 0 0 1 {margin_left} {current_y} Tm ({safe_line}) Tj\n".encode("latin-1", errors="replace"))
            current_y -= line_height
        stream_parts.append(b"ET")
        stream_bytes = b"".join(stream_parts)
        content_id = add_object(
            f"<< /Length {len(stream_bytes)} >>\nstream\n".encode("latin-1") + stream_bytes + b"\nendstream"
        )
        page_id = add_object(
            (
                f"<< /Type /Page /Parent {{PAGES_ID}} 0 R /MediaBox [0 0 {page_width} {page_height}] "
                f"/Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {content_id} 0 R >>"
            ).encode("latin-1")
        )
        page_ids.append(page_id)

    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    pages_id = add_object(f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode("latin-1"))

    for page_id in page_ids:
        objects[page_id - 1] = objects[page_id - 1].replace(b"{PAGES_ID}", str(pages_id).encode("latin-1"))

    catalog_id = add_object(f"<< /Type /Catalog /Pages {pages_id} 0 R >>".encode("latin-1"))

    buffer = io.BytesIO()
    buffer.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(buffer.tell())
        buffer.write(f"{index} 0 obj\n".encode("latin-1"))
        buffer.write(obj)
        buffer.write(b"\nendobj\n")

    xref_start = buffer.tell()
    buffer.write(f"xref\n0 {len(objects) + 1}\n".encode("latin-1"))
    buffer.write(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        buffer.write(f"{offset:010d} 00000 n \n".encode("latin-1"))
    buffer.write(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root {catalog_id} 0 R >>\n"
            f"startxref\n{xref_start}\n%%EOF"
        ).encode("latin-1")
    )
    return buffer.getvalue()


def generate_patient_pdf(patient_data: Dict[str, object]) -> bytes:
    if not REPORTLAB_AVAILABLE:
        return _build_simple_pdf(patient_data)

    buffer = io.BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=0.6 * inch,
        rightMargin=0.6 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
    )

    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    heading_style = styles["Heading2"]
    body_style = styles["BodyText"]
    body_style.spaceAfter = 8
    bullet_style = ParagraphStyle(
        "InsightBullet",
        parent=styles["BodyText"],
        leftIndent=14,
        bulletIndent=4,
        spaceAfter=4,
    )

    story = [
        Paragraph(f"Clinical Report - Patient {patient_data['patient_id']}", title_style),
        Spacer(1, 12),
        Paragraph("1. Patient Information", heading_style),
        Paragraph(f"Patient ID: {_safe_text(patient_data['patient_id'])}", body_style),
        Spacer(1, 6),
        Paragraph("2. Clinical Text", heading_style),
        Paragraph(_safe_text(patient_data.get("raw_text")) or "No clinical text available.", body_style),
        Spacer(1, 6),
        Paragraph("3. Extracted Entities", heading_style),
    ]

    entity_table = Table(_build_entity_rows(patient_data.get("entities", [])), colWidths=[3.2 * inch, 1.3 * inch, 1.2 * inch])
    entity_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D8E7FB")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.HexColor("#F7F9FC")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.extend([entity_table, Spacer(1, 10)])

    risk = patient_data.get("risk", {})
    story.extend(
        [
            Paragraph("4. Risk Assessment", heading_style),
            Paragraph(f"Risk Level: {_safe_text(risk.get('risk_level'))}", body_style),
            Paragraph(f"Explanation: {_safe_text(risk.get('explanation'))}", body_style),
            Spacer(1, 6),
            Paragraph("5. Insights", heading_style),
        ]
    )

    insights = patient_data.get("insights", [])
    if insights:
        for insight in insights:
            story.append(Paragraph(_safe_text(insight), bullet_style, bulletText="-"))
    else:
        story.append(Paragraph("No additional insights available.", body_style))

    story.extend(
        [
            Spacer(1, 6),
            Paragraph("6. Summary", heading_style),
            Paragraph(_safe_text(patient_data.get("summary")) or "No summary available.", body_style),
        ]
    )

    document.build(story)
    return buffer.getvalue()


def generate_patient_csv(patient_data: Dict[str, object]) -> bytes:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["patient_id", "entity", "type", "confidence", "risk_level", "summary"])

    entities = patient_data.get("entities", [])
    for entity in entities:
        writer.writerow(
            [
                _safe_text(patient_data.get("patient_id")),
                _safe_text(entity.get("text")),
                _safe_text(entity.get("label")),
                f"{float(entity.get('confidence', 0.0)):.2f}%",
                _safe_text(patient_data.get("risk", {}).get("risk_level")),
                _safe_text(patient_data.get("summary")),
            ]
        )

    return buffer.getvalue().encode("utf-8")
