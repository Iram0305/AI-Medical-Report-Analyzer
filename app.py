# app.py
# MediReport AI – PDF Input | Full Plots | Safe Extraction | Perfect PDF

import streamlit as st
import fitz
import easyocr
import requests
import json
import re
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import io
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
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

# ================================
# 2. PROMPTS
# ================================
EXTRACT_PROMPT = """
Extract in JSON only. No extra text.

{
  "patient_name": "",
  "age": "",
  "gender": "",
  "report_date": "",
  "tests": [
    {
      "name": "",
      "value": "",
      "unit": "",
      "range": "",
      "flag": "Normal/High/Low"
    }
  ]
}

Text:
{{TEXT}}
"""

SUMMARY_PROMPT = """
You are a senior physician. Write a **detailed, structured** summary in **sections**.

Use this structure:
## Patient Information
## Key Abnormal Findings
## Possible Causes
## Doctor to Consult
## Lifestyle Recommendations
## Urgency Level
## Next Steps

Include **bold values**, doctor types, urgency. **NO sign-off.**

Report Text:
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
    from PIL import Image
    import numpy as np
    img_np = np.array(pil_image)
    results = OCR_READER.readtext(img_np, detail=0, paragraph=True)
    return "\n".join(results)

def call_groq(prompt, model="llama-3.3-70b-versatile"):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 2000
    }
    try:
        resp = requests.post(GROQ_URL, json=payload, headers={"Authorization": f"Bearer {GROQ_API_KEY}"}, timeout=40)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except: return "Error generating response."

def extract_structured(text):
    prompt = EXTRACT_PROMPT.replace("{{TEXT}}", text[:15000])
    result = call_groq(prompt, "llama-3.1-8b-instant")
    try: data = json.loads(result)
    except: data = {"error": "Parse failed", "raw": result}
    
    # SAFE DEFAULTS
    tests = data.get("tests", [])
    for t in tests:
        t.setdefault("name", "Unknown Test")
        t.setdefault("value", "N/A")
        t.setdefault("unit", "")
        t.setdefault("range", "")
        t.setdefault("flag", "Unknown")  # ← THIS FIXES KeyError
    
    data["tests"] = tests
    return data

def summarize_report(text, data):
    prompt = SUMMARY_PROMPT.replace("{{TEXT}}", text[:4000]).replace("{{DATA}}", json.dumps(data, indent=2))
    return call_groq(prompt)

# ================================
# 4. VISUALIZATIONS
# ================================
def create_plots(df):
    # SAFE: Handle missing 'Flag'
    if 'flag' not in df.columns:
        df['flag'] = 'Unknown'
    df['flag'] = df['flag'].fillna('Unknown')
    
    # Bar Chart
    flag_counts = df['flag'].value_counts()
    fig_bar = px.bar(
        x=flag_counts.index, y=flag_counts.values,
        color=flag_counts.index,
        color_discrete_map={'Normal': '#2E8B57', 'High': '#DC143C', 'Low': '#FF8C00', 'Unknown': '#808080'},
        title="Test Results Overview",
        labels={'x': 'Status', 'y': 'Count'}
    )
    fig_bar.update_layout(showlegend=False, height=300)

    # Gauges
    gauges = []
    for _, row in df.iterrows():
        try:
            if 'Glucose' in row['name']:
                v, low, high = float(row['value']), *parse_range(row['range'])
                gauges.append(get_gauge(v, low, high, "Glucose", "mg/dL"))
            if 'HbA1c' in row['name']:
                v = float(row['value'])
                gauges.append(get_gauge(v, 0, 5.7, "HbA1c", "%"))
            if 'Cholesterol' in row['name']:
                v = float(row['value'])
                gauges.append(get_gauge(v, 0, 200, "Cholesterol", "mg/dL"))
        except: pass

    # Radar
    categories = ['Diabetes', 'Heart', 'Liver', 'Anemia']
    values = [0]*4
    for i, cat in enumerate(categories):
        matches = df[df['name'].str.contains(cat, case=False, na=False)]
        if not matches.empty and matches.iloc[0]['flag'] != 'Normal':
            values[i] = 3
        elif not matches.empty:
            values[i] = 1
    fig_radar = go.Figure(data=go.Scatterpolar(r=values, theta=categories, fill='toself'))
    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 3])), height=300)

    return fig_bar, gauges, fig_radar

def parse_range(range_str):
    nums = re.findall(r"[\d.]+", str(range_str))
    return [float(n) for n in nums] if nums else [0, 100]

def get_gauge(value, low, high, title, unit):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': f"{title} ({unit})"},
        gauge={
            'axis': {'range': [None, max(high*1.3, value*1.3)]},
            'bar': {'color': "red" if value > high else "orange" if value < low else "green"},
            'steps': [{'range': [0, low], 'color': "lightgray"}, {'range': [low, high], 'color': "yellow"}]
        }
    ))
    fig.update_layout(height=220, margin=dict(t=40, b=0))
    return fig

# ================================
# 5. PDF (NO HTML LEAKS)
# ================================
def add_plot_to_pdf(fig, story, width=5*inch, height=2.5*inch):
    img_data = fig.to_image(format="png")
    story.append(RLImage(io.BytesIO(img_data), width=width, height=height))
    story.append(Spacer(1, 0.2*inch))

def generate_pdf_report(p_name, p_age, p_gender, summary_text, fig_bar, gauges, fig_radar):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.8*inch)
    styles = getSampleStyleSheet()
    title = ParagraphStyle('Title', parent=styles['Title'], fontSize=20, spaceAfter=15, textColor=colors.HexColor('#1E90FF'), alignment=1)
    heading = ParagraphStyle('Heading', parent=styles['Heading2'], fontSize=14, spaceAfter=10)
    normal = ParagraphStyle('Normal', parent=styles['Normal'], fontSize=11, spaceAfter=8)
    small = ParagraphStyle('Small', parent=styles['Normal'], fontSize=9, textColor=colors.gray)

    story = []
    story.append(Paragraph("MediReport AI", title))
    story.append(Paragraph(f"Generated on {datetime.now().strftime('%B %d, %Y')}", small))
    story.append(Spacer(1, 0.3*inch))

    story.append(Paragraph("Patient Summary Report", heading))
    info = f"<b>Name:</b> {p_name} &nbsp;&nbsp; <b>Age:</b> {p_age} &nbsp;&nbsp; <b>Gender:</b> {p_gender}"
    story.append(Paragraph(info, normal))
    story.append(Spacer(1, 0.3*inch))

    add_plot_to_pdf(fig_bar, story)
    gauge_row = [RLImage(io.BytesIO(g.to_image(format="png")), width=1.8*inch, height=1.8*inch) for g in gauges[:3]]
    story.append(Table([gauge_row], colWidths=[1.9*inch]*3))
    story.append(Spacer(1, 0.3*inch))
    add_plot_to_pdf(fig_radar, story, width=4*inch, height=3*inch)

    sections = re.split(r'##\s+', summary_text)
    for sec in sections[1:]:
        lines = sec.strip().split('\n', 1)
        if len(lines) < 2: continue
        story.append(Paragraph(lines[0].strip(), heading))
        clean = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', lines[1])
        clean = html.escape(clean).replace('&lt;b&gt;', '<b>').replace('&lt;/b&gt;', '</b>')
        story.append(Paragraph(clean, normal))
        story.append(Spacer(1, 0.2*inch))

    story.append(Paragraph("<i>AI-generated report. Consult your doctor.</i>", small))
    doc.build(story)
    buffer.seek(0)
    return buffer

# ================================
# 6. UI
# ================================
st.set_page_config(page_title="MediReport AI", layout="wide", page_icon="medical")
st.title("MediReport AI")
st.markdown("### *Your AI Pathologist with Visual Insights*")

uploaded = st.file_uploader("Upload PDF Report", type=["pdf"])

if uploaded:
    with st.spinner("Reading PDF..."):
        raw_text = extract_text_from_pdf(uploaded.read())
    if not raw_text.strip():
        st.error("No text found.")
        st.stop()

    with st.spinner("Analyzing..."):
        structured = extract_structured(raw_text)
        summary = summarize_report(raw_text, structured)

    df = pd.DataFrame(structured.get("tests", []))
    if df.empty:
        st.error("No test data.")
        st.stop()

    p_name = structured.get("patient_name", "Patient")
    p_age = structured.get("age", "N/A")
    p_gender = structured.get("gender", "N/A")

    fig_bar, gauges, fig_radar = create_plots(df)

    col1, col2 = st.columns([1.4, 1])
    with col1:
        st.plotly_chart(fig_bar, use_container_width=True)
        for g in gauges:
            st.plotly_chart(g, use_container_width=True)
        st.plotly_chart(fig_radar, use_container_width=True)
    with col2:
        st.markdown("## Health Summary")
        sections = re.split(r'##\s+', summary)
        for sec in sections[1:]:
            lines = sec.strip().split('\n', 1)
            if len(lines) < 2: continue
            st.markdown(f"### {lines[0].strip()}")
            st.markdown(lines[1].strip())

    col_a, col_b = st.columns(2)
    with col_a:
        csv_data = df.to_csv(index=False).encode()
        st.download_button("Download CSV", csv_data, "report.csv", "text/csv")
    with col_b:
        pdf = generate_pdf_report(p_name, p_age, p_gender, summary, fig_bar, gauges, fig_radar)
        if pdf:
            safe_name = re.sub(r'\W+', '_', p_name)
            st.download_button("Download PDF Report", pdf, f"Health_Report_{safe_name}.pdf", "application/pdf")

st.markdown("---")
st.markdown("**Groq AI • Plotly • ReportLab** | *Consult your doctor.*")
