# app.py
# MediReport AI – With Auto PDF Report Generator
# Streamlit Cloud Ready | Groq Free | PDF Export

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
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

# ================================
# 1. CONFIG: GROQ API
# ================================
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except KeyError:
    st.error("Set `GROQ_API_KEY` in **Streamlit Secrets**.")
    st.info("Example: `GROQ_API_KEY = \"gsk_your_key\"`")
    st.stop()

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
OCR_READER = easyocr.Reader(['en'], gpu=False)

# ================================
# 2. MODELS
# ================================
EXTRACTION_MODEL = "llama-3.1-8b-instant"
SUMMARY_MODEL = "llama-3.3-70b-versatile"
FALLBACK_MODEL = "llama3-8b-8192"

# ================================
# 3. PROMPTS
# ================================
JSON_EXAMPLE = {
    "patient_name": "", "age": "", "gender": "", "report_date": "",
    "tests": [{"name": "", "value": "", "unit": "", "range": "", "flag": ""}],
    "impression": ""
}
JSON_EXAMPLE_STR = json.dumps(JSON_EXAMPLE, indent=2)

EXTRACT_PROMPT = f"""
Extract in JSON only. No extra text.

{JSON_EXAMPLE_STR}

Text:
{{TEXT}}
"""

SUMMARY_PROMPT = """
You are a senior physician explaining lab results to a patient in simple, caring language.

Using the full report text and structured data below, write a **detailed, intuitive, and insightful** summary in **10+ bullet points**.

Include:
- Patient name, age, gender
- Key abnormalities with **bold values**
- What each means in plain English
- Possible causes (anemia, diabetes, etc.)
- Lifestyle tips
- When to see a doctor
- Urgency level (low/medium/high)
- Next steps (repeat tests, consult specialist)

Be empathetic, encouraging, and clear.

Report Text (first 3000 chars):
{{TEXT}}

Structured Data:
{{DATA}}
"""

# ================================
# 4. UTILS
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
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 1500,
        "n": 1
    }
    try:
        resp = requests.post(GROQ_URL, json=payload, headers=headers, timeout=40)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except requests.exceptions.HTTPError as e:
        err = e.response.text.lower()
        if "decommissioned" in err:
            st.warning(f"Model {model} unavailable. Using fallback.")
            return call_groq(prompt, FALLBACK_MODEL)
        return json.dumps({"error": f"API Error: {err}"})
    except Exception as e:
        return json.dumps({"error": str(e)})

def extract_structured(text):
    prompt = EXTRACT_PROMPT.replace("{{TEXT}}", text[:10000])
    result = call_groq(prompt, EXTRACTION_MODEL)
    try:
        data = json.loads(result)
    except:
        data = {"error": "Parse failed", "raw": result}

    for t in data.get("tests", []):
        try:
            val = float(re.search(r"[\d.]+", t["value"]).group())
            rng = re.findall(r"[\d.]+", t["range"] or "")
            if len(rng) >= 2:
                low, high = float(rng[0]), float(rng[-1])
                t["flag"] = "High" if val > high else "Low" if val < low else "Normal"
            else:
                t["flag"] = "Normal"
        except:
            t["flag"] = "Unknown"
    return data

def summarize_report(text, data):
    prompt = SUMMARY_PROMPT.replace("{{TEXT}}", text[:3000]) \
                          .replace("{{DATA}}", json.dumps(data, indent=2))
    return call_groq(prompt, SUMMARY_MODEL)

def dict_to_csv(data):
    output = io.StringIO()
    writer = csv.writer(output)
    for k, v in data.items():
        if k not in ["tests", "error", "raw"]:
            writer.writerow([k.replace("_", " ").title(), v])
    writer.writerow([])
    writer.writerow(["Test", "Value", "Unit", "Range", "Flag"])
    for t in data.get("tests", []):
        writer.writerow([t["name"], t["value"], t.get("unit",""), t.get("range",""), t.get("flag","")])
    return output.getvalue()

# ================================
# 5. PDF REPORT GENERATOR
# ================================
def generate_pdf_report(structured, summary, patient_name="Patient"):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=1*inch, bottomMargin=0.8*inch)
    styles = getSampleStyleSheet()
    story = []

    # Custom Styles
    title_style = ParagraphStyle('Title', parent=styles['Title'], fontSize=18, spaceAfter=20, textColor=colors.HexColor('#1E90FF'))
    heading_style = ParagraphStyle('Heading', parent=styles['Heading2'], fontSize=14, spaceAfter=10, textColor=colors.HexColor('#2E8B57'))
    normal_style = ParagraphStyle('Normal', parent=styles['Normal'], fontSize=11, spaceAfter=8, leading=14)

    # Header
    story.append(Paragraph("MediReport AI", title_style))
    story.append(Paragraph(f"<font size=12 color=gray>Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</font>", normal_style))
    story.append(Spacer(1, 0.3*inch))

    # Patient Info
    story.append(Paragraph("Patient Summary Report", heading_style))
    info = f"<b>Name:</b> {structured.get('patient_name', 'N/A')} &nbsp;&nbsp; <b>Age:</b> {structured.get('age', 'N/A')} &nbsp;&nbsp; <b>Gender:</b> {structured.get('gender', 'N/A')}"
    story.append(Paragraph(info, normal_style))
    story.append(Spacer(1, 0.2*inch))

    # Summary
    story.append(Paragraph("AI-Generated Health Insights", heading_style))
    summary_html = summary.replace("**", "<b>").replace("**", "</b>")
    story.append(Paragraph(summary_html, normal_style))
    story.append(Spacer(1, 0.4*inch))

    # Key Results Table
    story.append(Paragraph("Key Lab Results", heading_style))
    table_data = [["Test", "Value", "Unit", "Range", "Status"]]
    for t in structured.get("tests", [])[:10]:  # Top 10
        flag = t.get("flag", "")
        color = "Normal"
        if flag == "High": color = "<font color=red>High</font>"
        elif flag == "Low": color = "<font color=orange>Low</font>"
        else: color = "<font color=green>Normal</font>"
        table_data.append([t["name"], t["value"], t.get("unit",""), t.get("range",""), color])

    table = Table(table_data, colWidths=[2.2*inch, 0.8*inch, 0.6*inch, 1.2*inch, 0.9*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E90FF')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 11),
        ('BOTTOMPADDING', (0,0), (-1,0), 12),
        ('BACKGROUND', (0,1), (-1,-1), colors.beige),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(table)

    # Footer
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph("<i>This is an AI-generated report for informational purposes. Please consult your doctor.</i>", normal_style))

    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer

# ================================
# 6. UI
# ================================
st.set_page_config(page_title="MediReport AI", layout="centered", page_icon="medical")
st.title("MediReport AI")
st.markdown("### *Your Personal AI Pathologist*")
st.caption("Upload report → Get **detailed insights + printable PDF report**")

uploaded = st.file_uploader("Upload Report", type=["pdf", "png", "jpg", "jpeg"])

if uploaded:
    with st.spinner("Reading report..."):
        raw_text = extract_text_from_pdf(uploaded.read()) if uploaded.type == "application/pdf" else ocr_image(Image.open(uploaded))
    
    if not raw_text.strip():
        st.error("No text found. Try a clearer scan.")
        st.stop()

    with st.spinner("Analyzing with AI..."):
        structured = extract_structured(raw_text)
        summary = summarize_report(raw_text, structured)

    col1, col2 = st.columns([1.2, 1.8])
    with col1:
        st.subheader("Structured Data")
        st.json(structured, expanded=False)
    with col2:
        st.subheader("Your Health Summary")
        st.markdown(summary)

    # === DOWNLOAD BUTTONS ===
    col_a, col_b = st.columns(2)
    with col_a:
        st.download_button("Download CSV", dict_to_csv(structured), "report.csv", "text/csv")
    with col_b:
        patient_name = structured.get("patient_name", "Patient").replace(" ", "_")
        pdf_buffer = generate_pdf_report(structured, summary, patient_name)
        st.download_button(
            label="Download PDF Report",
            data=pdf_buffer,
            file_name=f"Health_Report_{patient_name}_{datetime.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf"
        )

# ================================
# 7. FOOTER
# ================================
st.markdown("---")
st.markdown(
    """
    **Powered by Groq AI** • **PDF Reports via ReportLab**  
    *No medical advice — consult your doctor.*  
    Made in India | November 11, 2025
    """
)
