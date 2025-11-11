# app.py
# MediReport AI – PDF Uses Dashboard Summary Only | No Errors | Professional

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
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
import html

# ================================
# 1. CONFIG
# ================================
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except KeyError:
    st.error("Set `GROQ_API_KEY` in Streamlit Secrets.")
    st.stop()

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
OCR_READER = easyocr.Reader(['en'], gpu=False)

EXTRACTION_MODEL = "llama-3.1-8b-instant"
SUMMARY_MODEL = "llama-3.3-70b-versatile"

# ================================
# 2. PROMPTS
# ================================
EXTRACT_PROMPT = """
Extract in JSON only. No extra text.

{
  "patient_name": "", "age": "", "gender": "", "report_date": "",
  "tests": [{"name": "", "value": "", "unit": "", "range": "", "flag": ""}],
  "impression": ""
}

Text:
{{TEXT}}
"""

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
    # SAFE: Ensure every test has required keys
    for t in data.get("tests", []):
        t.setdefault("name", "Unknown Test")
        t.setdefault("value", "N/A")
        t.setdefault("unit", "")
        t.setdefault("range", "")
        t.setdefault("flag", "Unknown")
        # Auto-flag if missing
        if t["flag"] == "":
            try:
                val = float(re.search(r"[\d.]+", t["value"]).group())
                rng = re.findall(r"[\d.]+", t["range"])
                if len(rng) >= 2:
                    low, high = float(rng[0]), float(rng[-1])
                    t["flag"] = "High" if val > high else "Low" if val < low else "Normal"
            except: t["flag"] = "Unknown"
    return data

def summarize_report(text, data):
    prompt = SUMMARY_PROMPT.replace("{{TEXT}}", text[:3000]).replace("{{DATA}}", json.dumps(data, indent=2))
    return call_groq(prompt, SUMMARY_MODEL)

def dict_to_csv(data):
    output = io.StringIO()
    writer = csv.writer(output)
    # Patient Info
    for k, v in data.items():
        if k not in ["tests", "error", "raw"]:
            writer.writerow([k.replace("_", " ").title(), v])
    writer.writerow([])
    writer.writerow(["Test", "Value", "Unit", "Range", "Flag"])
    # SAFE: Use .get() to avoid KeyError
    for t in data.get("tests", []):
        writer.writerow([
            t.get("name", "Unknown"),
            t.get("value", "N/A"),
            t.get("unit", ""),
            t.get("range", ""),
            t.get("flag", "Unknown")
        ])
    return output.getvalue()

# ================================
# 4. PDF: USE DASHBOARD SUMMARY ONLY
# ================================
def generate_pdf_report(patient_name, age, gender, summary_text):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.8*inch, bottomMargin=0.8*inch, leftMargin=0.8*inch, rightMargin=0.8*inch)
    styles = getSampleStyleSheet()

    title = ParagraphStyle('Title', parent=styles['Title'], fontSize=20, spaceAfter=15, textColor=colors.HexColor('#1E90FF'), alignment=1)
    heading = ParagraphStyle('Heading', parent=styles['Heading2'], fontSize=14, spaceAfter=8, textColor=colors.HexColor('#2E8B57'))
    normal = ParagraphStyle('Normal', parent=styles['Normal'], fontSize=11, spaceAfter=6, leading=14)
    small = ParagraphStyle('Small', parent=styles['Normal'], fontSize=9, textColor=colors.gray)

    story = []

    # Header
    story.append(Paragraph("MediReport AI", title))
    story.append(Paragraph(f"Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", small))
    story.append(Spacer(1, 0.2*inch))

    # Patient Info
    story.append(Paragraph("Patient Summary Report", heading))
    info = f"<b>Name:</b> {patient_name} &nbsp;&nbsp;&nbsp; <b>Age:</b> {age} &nbsp;&nbsp;&nbsp; <b>Gender:</b> {gender}"
    story.append(Paragraph(info, normal))
    story.append(Spacer(1, 0.3*inch))

    # Summary – FROM DASHBOARD ONLY
    story.append(Paragraph("Your Health Insights", heading))
    clean = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', summary_text)
    clean = html.escape(clean)
    clean = clean.replace('&lt;b&gt;', '<b>').replace('&lt;/b&gt;', '</b>')
    clean = clean.replace('•', '<br/>• ').replace('\n', '<br/>')
    story.append(Paragraph(f"<font name='Helvetica'>{clean}</font>", normal))

    # Footer
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph("<i>This is an AI-generated report for informational purposes only. Please consult your physician.</i>", small))

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
        summary = summarize_report(raw_text, structured)  # ← This is shown on dashboard

    # Extract patient info SAFELY
    p_name = structured.get("patient_name", "Patient")
    p_age = structured.get("age", "N/A")
    p_gender = structured.get("gender", "N/A")

    col1, col2 = st.columns([1.2, 1.8])
    with col1:
        st.subheader("Structured Data")
        st.json(structured, expanded=False)
    with col2:
        st.subheader("Your Health Summary")
        st.markdown(summary)  # ← Dashboard summary

    # === DOWNLOADS ===
    col_a, col_b = st.columns(2)
    with col_a:
        st.download_button("Download CSV", dict_to_csv(structured), "report.csv", "text/csv")
    with col_b:
        pdf = generate_pdf_report(p_name, p_age, p_gender, summary)  # ← Uses dashboard summary
        if pdf:
            safe_name = re.sub(r'\W+', '_', p_name)
            st.download_button(
                "Download PDF Report",
                pdf,
                f"Health_Report_{safe_name}_{datetime.now().strftime('%Y%m%d')}.pdf",
                "application/pdf"
            )
        else:
            st.error("PDF failed.")

st.markdown("---")
st.markdown("**Powered by Groq AI • ReportLab** | *Consult your doctor.* | November 11, 2025")
