# app.py
# MediReport AI – CLEAN GAUGES | NO HTML | CORRECT NAME | NO EXTRA TESTS

import streamlit as st
import fitz
import requests
import json
import re
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import io
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

# ================================
# 1. CONFIG
# ================================
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except KeyError:
    st.error("Set `GROQ_API_KEY` in Streamlit Secrets.")
    st.stop()

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

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
    
    tests = data.get("tests", [])
    for t in tests:
        t.setdefault("name", "Unknown Test")
        t.setdefault("value", "N/A")
        t.setdefault("unit", "")
        t.setdefault("range", "")
        t.setdefault("flag", "Unknown")
    
    data["tests"] = tests
    return data

def summarize_report(text, data):
    prompt = SUMMARY_PROMPT.replace("{{TEXT}}", text[:4000]).replace("{{DATA}}", json.dumps(data, indent=2))
    return call_groq(prompt)

# ================================
# 4. VISUALIZATIONS – NO DELTA, CLEAN GAUGES
# ================================
def create_plots(df):
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
        labels={'x': 'Status', 'y': 'Number of Tests'}
    )
    fig_bar.update_layout(showlegend=False, height=400, margin=dict(t=80, b=60, l=60, r=60), title_x=0.5)

    # Gauges – NO DELTA
    gauges = []
    for _, row in df.iterrows():
        try:
            if 'Glucose' in row['name']:
                v, low, high = float(row['value']), *parse_range(row['range'])
                gauges.append(get_gauge(v, low, high, "Fasting Glucose", "mg/dL"))
            if 'HbA1c' in row['name']:
                v = float(row['value'])
                gauges.append(get_gauge(v, 0, 5.7, "HbA1c", "%"))
            if 'Cholesterol' in row['name']:
                v = float(row['value'])
                gauges.append(get_gauge(v, 0, 200, "Total Cholesterol", "mg/dL"))
        except: pass

    return fig_bar, gauges

def parse_range(range_str):
    nums = re.findall(r"[\d.]+", str(range_str))
    return [float(n) for n in nums] if nums else [0, 100]

def get_gauge(value, low, high, title, unit):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",  # REMOVED DELTA
        value=value,
        title={'text': f"<b>{title}</b><br><span style='font-size:0.8em'>{unit}</span>"},
        gauge={
            'axis': {'range': [None, max(high*1.3, value*1.3)]},
            'bar': {'color': "red" if value > high else "orange" if value < low else "green"},
            'steps': [{'range': [0, low], 'color': "lightgray"}, {'range': [low, high], 'color': "yellow"}]
        }
    ))
    fig.update_layout(height=300, margin=dict(t=80, b=20, l=40, r=40))
    return fig

# ================================
# 5. PDF – NO HTML, PLAIN TEXT STATUS
# ================================
def add_plot_to_pdf(fig, story, width=6*inch, height=3*inch):
    try:
        img_data = fig.to_image(format="png", engine="kaleido", width=800, height=400)
        story.append(RLImage(io.BytesIO(img_data), width=width, height=height))
        story.append(Spacer(1, 0.3*inch))
    except: 
        story.append(Paragraph("Plot could not be rendered.", getSampleStyleSheet()['Normal']))

def generate_pdf_report(p_name, p_age, p_gender, summary_text, fig_bar, gauges):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.8*inch)
    styles = getSampleStyleSheet()
    title = ParagraphStyle('Title', parent=styles['Title'], fontSize=20, spaceAfter=20, textColor=colors.HexColor('#1E90FF'), alignment=1)
    heading = ParagraphStyle('Heading', parent=styles['Heading2'], fontSize=14, spaceAfter=12)
    normal = ParagraphStyle('Normal', parent=styles['Normal'], fontSize=11, spaceAfter=8)
    small = ParagraphStyle('Small', parent=styles['Normal'], fontSize=9, textColor=colors.gray)

    story = []
    story.append(Paragraph("MediReport AI", title))
    story.append(Paragraph(f"Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", small))
    story.append(Spacer(1, 0.4*inch))

    story.append(Paragraph("Patient Summary Report", heading))
    info = f"<b>Name:</b> {p_name} &nbsp;&nbsp; <b>Age:</b> {p_age} &nbsp;&nbsp; <b>Gender:</b> {p_gender}"
    story.append(Paragraph(info, normal))
    story.append(Spacer(1, 0.4*inch))

    add_plot_to_pdf(fig_bar, story)

    for i in range(0, len(gauges), 2):
        row = gauges[i:i+2]
        row_imgs = []
        for g in row:
            try:
                img_data = g.to_image(format="png", engine="kaleido", width=500, height=300)
                row_imgs.append(RLImage(io.BytesIO(img_data), width=3*inch, height=1.8*inch))
            except: row_imgs.append(Paragraph("N/A", normal))
        story.append(Table([row_imgs], colWidths=[3.1*inch]*2))
        story.append(Spacer(1, 0.3*inch))

    # TABLE – PLAIN TEXT STATUS
    story.append(Paragraph("Key Lab Results", heading))
    table_data = [["Test", "Value", "Unit", "Range", "Status"]]
    for _, row in pd.DataFrame(structured.get("tests", [])).iterrows():
        status = row['flag']
        table_data.append([row['name'], row['value'], row.get('unit',''), row.get('range',''), status])  # PLAIN TEXT
    table = Table(table_data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E90FF')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('BACKGROUND', (0,1), (-1,-1), colors.white),
    ]))
    story.append(table)
    story.append(Spacer(1, 0.3*inch))

    # Summary
    sections = re.split(r'##\s+', summary_text)
    for sec in sections[1:]:
        lines = sec.strip().split('\n', 1)
        if len(lines) < 2: continue
        story.append(Paragraph(lines[0].strip(), heading))
        clean = re.sub(r'\*\*(.*?)\*\*', r'\1', lines[1])
        story.append(Paragraph(clean, normal))
        story.append(Spacer(1, 0.25*inch))

    story.append(Paragraph("AI-generated report. Consult your doctor.", small))
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

    fig_bar, gauges = create_plots(df)

    st.markdown("## Visual Insights")
    st.plotly_chart(fig_bar, use_container_width=True)
    for g in gauges:
        st.plotly_chart(g, use_container_width=True)

    st.markdown("---")
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
        pdf = generate_pdf_report(p_name, p_age, p_gender, summary, fig_bar, gauges)
        if pdf:
            safe_name = re.sub(r'\W+', '_', p_name)
            st.download_button("Download PDF Report", pdf, f"Health_Report_{safe_name}.pdf", "application/pdf")

st.markdown("---")
st.markdown("**Groq AI • Plotly • ReportLab** | *Consult your doctor.*")
