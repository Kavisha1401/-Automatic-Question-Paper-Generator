import io
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt

# groups = [(section_heading_or_None, [(q_no, question_text, marks), ...]), ...]

def make_pdf(groups, title, subject, duration, total):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=ss["Title"], fontSize=16, spaceAfter=4)
    center = ParagraphStyle("c", parent=ss["Normal"], alignment=1, fontSize=11)
    sec = ParagraphStyle("sec", parent=ss["Heading3"], spaceBefore=12, spaceAfter=6)
    body = ParagraphStyle("b", parent=ss["Normal"], fontSize=11, leading=14)

    story = [Paragraph(escape(title), h1), Paragraph(escape(subject), center), Spacer(1, 8)]
    meta = Table([[f"Time: {duration}", f"Total Marks: {total}"]], colWidths=[8.5*cm, 8.5*cm])
    meta.setStyle(TableStyle([("ALIGN", (0, 0), (0, 0), "LEFT"), ("ALIGN", (1, 0), (1, 0), "RIGHT"),
                              ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold")]))
    story += [meta, HRFlowable(width="100%", thickness=1, color=colors.black, spaceBefore=4, spaceAfter=6)]

    for heading, items in groups:
        if heading:
            story.append(Paragraph(escape(heading), sec))
        rows = [[f"Q{n}.", Paragraph(escape(t), body), f"[{m}]"] for n, t, m in items]
        tbl = Table(rows, colWidths=[1.2*cm, 14.3*cm, 1.5*cm])
        tbl.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("ALIGN", (2, 0), (2, -1), "RIGHT"),
                                 ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
        story.append(tbl)
    doc.build(story)
    return buf.getvalue()


def make_docx(groups, title, subject, duration, total):
    d = Document()
    p = d.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(title); r.bold = True; r.font.size = Pt(16)
    p = d.add_paragraph(subject); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    d.add_paragraph(f"Time: {duration}\t\t\tTotal Marks: {total}")
    for heading, items in groups:
        if heading:
            d.add_heading(heading, level=3)
        for n, t, m in items:
            d.add_paragraph(f"Q{n}. {t}  [{m} marks]")
    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()