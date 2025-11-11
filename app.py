# app.py
# MediReport AI – PROFESSIONAL PDF REPORT (Fixed, Beautiful, Patient-Ready)
# Streamlit Cloud | Groq Free | ReportLab

import streamlit as st
import fitz
import easyocr
import requests
import json
import re
import csv
import io
from PIL import Image
import numpy as np
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
import html
import textwrap

# ================================
# 1. CONFIG
# ================================
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except KeyError:
    st.error("Set `GROQ_API_KEY` in **Streamlit Secrets**.")
    st.stop()

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
OCR_READER = easyocr.Reader(['en'], gpu=False)

# Models
EXTRACTION_MODEL = "llama-3.1-8b-instant"
SUMMARY_MODEL = "llama-3.3-70b-versatile"
FALLBACK_MODEL = "llama3-8b-8192"

# ================================
# 2. PROMPTS
# ================================
JSON_EXAMPLE = { "patient_name": "", "age": "", "gender": "", "report_date": "", "tests": [], "impression": "" }
JSON_EXAMPLE_STR = json.dumps(JSON_EXAMPLE, indent=2)

EXTRACT_PROMPT = f"Extract in JSON only. No extra text.\n\n{JSON_EXAMPLE_STR}\n\nText:\n{{TEXT}}"

SUMMARY_PROMPT = """
You are a senior physician. Write a **detailed, caring, patient-friendly** summary in **10+ bullet points**.

Include:
- Patient name, age, gender
- Key abnormalities with **bold values**
- Plain English explanation
- Possible causes
- Lifestyle tips
- Urgency level
- Next steps

Be empathetic and clear.

Report Text (first 3000 chars):
{{TEXT}}

Structured Data:
{{DATA}}
"""

# ================================
# 3. UTILS
# ================================
def extract_text_from_pdf(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    return "\n".join(page.get_text() for page in doc)

def ocr_image(pil_image):
    img_np = np.array(pil_image)
    results = OCR_READER.readtext(img_np, detail=0, paragraph=True)
    return "\n".join(results)

def call_groq(prompt, model=EXTRACTION_MODEL):
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = { "model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.3, "max_tokens": 1500, "n": 1 }
    try:
        resp = requests.post(GROQ_URL, json=payload, headers=headers, timeout=40)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return json.dumps({"error": str(e)})

def extract_structured(text):
    prompt = EXTRACT_PROMPT.replace("{{TEXT}}", text[:10000])
    result = call_groq(prompt, EXTRACTION_MODEL)
    try: data = json.loads(result)
    except: data = {"error": "Parse failed", "raw": result}
    for t in data.get("tests", []):
        try:
            val = float(re.search(r"[\d.]+", t["value"]).group())
            rng = re.findall(r"[\d.]+", t["range"] or "")
            if len(rng) >= 2:
                low, high = float(rng[0]), float(rng[-1])
                t["flag"] = "High" if val > high else "Low" if val < low else "Normal"
            else: t["flag"] = "Normal"
        except: t["flag"] = "Unknown"
    return data

def summarize_report(text, data):
    prompt = SUMMARY_PROMPT.replace("{{TEXT}}", text[:3000]).replace("{{DATA}}", json.dumps(data, indent=2))
    return call_groq(prompt, SUMMARY_MODEL)

def dict_to_csv(data):
    output = io.StringIO()
    writer = csv.writer(output)
    for k, v in data.items():
        if k not in ["tests", "error", "raw"]: writer.writerow([k.replace("_", " ").title(), v])
    writer.writerow([]); writer.writerow(["Test", "Value", "Unit", "Range", "Flag"])
    for t in data.get("tests", []): writer.writerow([t["name"], t["value"], t.get("unit",""), t.get("range",""), t.get("flag","")])
    return output.getvalue()

# ================================
# 4. PDF REPORT – PROFESSIONAL
# ================================
def generate_pdf_report(structured, summary):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.8*inch, bottomMargin=0.8*inch, leftMargin=0.8*inch, rightMargin=0.8*inch)
    styles = getSampleStyleSheet()

    # Styles
    title = ParagraphStyle('Title', parent=styles['Title'], fontSize=20, spaceAfter=15, textColor=colors.HexColor('#1E90FF'), alignment=1)
    heading = ParagraphStyle('Heading', parent=styles['Heading2'], fontSize=14, spaceAfter=8, textColor=colors.HexColor('#2E8B57'))
    normal = ParagraphStyle('Normal', parent=styles['Normal'], fontSize=11, spaceAfter=6, leading=14)
    small = ParagraphStyle('Small', parent=styles['Normal'], fontSize=9, textColor=colors.gray)

    story = []

    # === HEADER ===
    story.append(Paragraph("MediReport AI", title))
    story.append(Paragraph(f"Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", small))
    story.append(Spacer(1, 0.2*inch))

    # === PATIENT INFO ===
    story.append(Paragraph("Patient Summary Report", heading))
    p_name = structured.get('patient_name', 'N/A')
    p_age = structured.get('age', 'N/A')
    p_gender = structured.get('gender', 'N/A')
    info = f"<b>Name:</b> {p_name} &nbsp;&nbsp;&nbsp; <b>Age:</b> {p_age} &nbsp;&nbsp;&nbsp; <b>Gender:</b> {p_gender}"
    story.append(Paragraph(info, normal))
    story.append(Spacer(1, 0.3*inch))

    # === SUMMARY ===
    story.append(Paragraph("Your Health Insights", heading))
    clean_summary = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', summary)
    clean_summary = html.escape(clean_summary)
    clean_summary = clean_summary.replace('&lt;b&gt;', '<b>').replace('&lt;/b&gt;', '</b>')
    clean_summary = clean_summary.replace('•', '<br/>• ').replace('\n', '<br/>')
    story.append(Paragraph(f"<font name='Helvetica'>{clean_summary}</font>", normal))
    story.append(PageBreak())

    # === LAB TABLE ===
    story.append(Paragraph("Key Lab Results", heading))
    table_data = [["Test Name", "Value", "Unit", "Reference Range", "Status"]]
    for t in structured.get("tests", [])[:12]:
        status = t.get("flag", "Unknown")
        color = colors.green if status == "Normal" else colors.red if status == "High" else colors.orange
        status_cell = Paragraph(f"<font color='{color.name}'><b>{status}</b></font>", normal)
        table_data.append([t["name"], t["value"], t.get("unit",""), t.get("range",""), status_cell])

    table = Table(table_data, colWidths=[2.3*inch, 0.7*inch, 0.6*inch, 1.3*inch, 0.9*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E90FF')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 11),
        ('BOTTOMPADDING', (0,0), (-1,0), 12),
        ('BACKGROUND', (0,1), (-1,-1), colors.white),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('LEFTPADDING', (0,1), (-1,-1), 6),
        ('RIGHTPADDING', (0,1), (-1,-1), 6),
    ]))
    story.append(table)

    # === FOOTER ===
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph("<i>This is an AI-generated report for informational purposes only. Please consult your physician for medical advice.</i>", small))

    # Build
    try: doc.build(story)
    except: return None
    buffer.seek(0)
    return buffer

# ================================
# 5. UI
# ================================
st.set_page_config(page_title="MediReport AI", layout="centered", page_icon="medical")
st.title("MediReport AI")
st.markdown("### *Your Personal AI Pathologist*")
st.caption("Upload report → Get **detailed insights + professional PDF**")

uploaded = st.file_uploader("Upload Report", type=["pdf", "png", "jpg", "jpeg"])

if uploaded:
    with st.spinner("Reading..."):
        raw_text = extract_text_from_pdf(uploaded.read()) if uploaded.type == "application/pdf" else ocr_image(Image.open(uploaded))
    if not raw_text.strip(): st.error("No text."); st.stop()

    with st.spinner("Analyzing..."):
        structured = extract_structured(raw_text)
        summary = summarize_report(raw_text, structured)

    col1, col2 = st.columns([1.2, 1.8])
    with col1:
        st.subheader("Structured Data")
        st.json(structured, expanded=False)
    with col2:
        st.subheader("Your Health Summary")
        st.markdown(summary)

    col_a, col_b = st.columns(2)
    with col_a:
        st.download_button("Download CSV", dict_to_csv(structured), "report.csv", "text/csv")
    with col_b:
        pdf = generate_pdf_report(structured, summary)
        if pdf:
            name = re.sub(r'\W+', '_', structured.get("patient_name", "Patient"))
            st.download_button(
                "Download PDF Report",
                pdf,
                f"Health_Report_{name}_{datetime.now().strftime('%Y%m%d')}.pdf",
                "application/pdf"
            )
        else:
            st.error("PDF failed.")

st.markdown("---")
st.markdown("**Powered by Groq AI • ReportLab** | *Consult your doctor.* | November 11, 2025")
