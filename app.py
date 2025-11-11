# app.py
# MediReport AI – FIXED PLOTS & FORMATTING

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
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import Image as RLImage  # FIXED: Added missing import

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
# 4. VISUALIZATIONS
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

    # Gauges
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
        mode="gauge+number",
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
# 5. PDF GENERATION WITH PROPER FORMATTING
# ================================
def create_section_divider():
    """Create a visual section divider"""
    return Spacer(1, 0.2*inch)

def add_plot_to_pdf(fig, story, width=6*inch, height=3*inch):
    """Add plotly figure to PDF"""
    try:
        img_data = fig.to_image(format="png", engine="kaleido", width=800, height=400)
        img = RLImage(io.BytesIO(img_data), width=width, height=height)
        story.append(img)
        story.append(Spacer(1, 0.3*inch))
    except Exception as e:
        story.append(Paragraph(f"Plot could not be rendered: {str(e)}", getSampleStyleSheet()['Normal']))
        story.append(Spacer(1, 0.2*inch))

def generate_pdf_report(p_name, p_age, p_gender, summary_text, fig_bar, gauges, df):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.8*inch, bottomMargin=0.8*inch)
    
    # Define styles
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontSize=24,
        textColor=colors.HexColor('#1E90FF'),
        spaceAfter=10,
        alignment=1,
        fontName='Helvetica-Bold'
    )
    
    subtitle_style = ParagraphStyle(
        'Subtitle',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.gray,
        spaceAfter=20,
        alignment=1
    )
    
    section_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading1'],
        fontSize=16,
        textColor=colors.HexColor('#1E90FF'),
        spaceAfter=12,
        spaceBefore=20,
        fontName='Helvetica-Bold',
        borderWidth=0,
        borderColor=colors.HexColor('#1E90FF'),
        borderPadding=5,
        backColor=colors.HexColor('#E6F2FF')
    )
    
    subsection_heading = ParagraphStyle(
        'SubsectionHeading',
        parent=styles['Heading2'],
        fontSize=13,
        textColor=colors.HexColor('#2C5AA0'),
        spaceAfter=8,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=6,
        leading=14
    )
    
    info_style = ParagraphStyle(
        'InfoStyle',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=8,
        textColor=colors.HexColor('#333333')
    )

    story = []
    
    # Header
    story.append(Paragraph("🏥 MediReport AI", title_style))
    story.append(Paragraph(
        f"Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}",
        subtitle_style
    ))
    story.append(Spacer(1, 0.3*inch))
    
    # Divider line
    story.append(Table([['']], colWidths=[6.5*inch], style=[
        ('LINEABOVE', (0,0), (-1,-1), 2, colors.HexColor('#1E90FF'))
    ]))
    story.append(Spacer(1, 0.4*inch))

    # Patient Information Section
    story.append(Paragraph("📋 Patient Information", section_heading))
    patient_data = [
        ['Name:', p_name],
        ['Age:', p_age],
        ['Gender:', p_gender],
        ['Report Date:', datetime.now().strftime('%B %d, %Y')]
    ]
    patient_table = Table(patient_data, colWidths=[1.5*inch, 5*inch])
    patient_table.setStyle(TableStyle([
        ('FONT', (0,0), (0,-1), 'Helvetica-Bold', 10),
        ('FONT', (1,0), (1,-1), 'Helvetica', 10),
        ('TEXTCOLOR', (0,0), (0,-1), colors.HexColor('#1E90FF')),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(patient_table)
    story.append(create_section_divider())
    
    # Visual Analysis Section
    story.append(Paragraph("📊 Visual Analysis", section_heading))
    story.append(Spacer(1, 0.2*inch))
    
    # Bar chart
    story.append(Paragraph("Test Results Distribution", subsection_heading))
    add_plot_to_pdf(fig_bar, story, width=6*inch, height=3*inch)
    
    # Gauges in rows of 2
    if gauges:
        story.append(Paragraph("Key Health Metrics", subsection_heading))
        for i in range(0, len(gauges), 2):
            row_gauges = gauges[i:i+2]
            row_imgs = []
            for g in row_gauges:
                try:
                    img_data = g.to_image(format="png", engine="kaleido", width=500, height=300)
                    row_imgs.append(RLImage(io.BytesIO(img_data), width=3*inch, height=1.8*inch))
                except:
                    row_imgs.append(Paragraph("Chart unavailable", normal_style))
            
            # Create table for side-by-side gauges
            gauge_table = Table([row_imgs], colWidths=[3.25*inch]*len(row_imgs))
            story.append(gauge_table)
            story.append(Spacer(1, 0.2*inch))
    
    story.append(create_section_divider())
    
    # Lab Results Table
    story.append(Paragraph("🔬 Detailed Lab Results", section_heading))
    story.append(Spacer(1, 0.2*inch))
    
    table_data = [["Test Name", "Value", "Unit", "Reference Range", "Status"]]
    for _, row in df.iterrows():
        status = row['flag']
        # Color code status
        if status == 'High':
            status_cell = Paragraph(f'<font color="red"><b>HIGH</b></font>', normal_style)
        elif status == 'Low':
            status_cell = Paragraph(f'<font color="orange"><b>LOW</b></font>', normal_style)
        elif status == 'Normal':
            status_cell = Paragraph(f'<font color="green"><b>NORMAL</b></font>', normal_style)
        else:
            status_cell = Paragraph(status, normal_style)
            
        table_data.append([
            Paragraph(row['name'], normal_style),
            row['value'],
            row.get('unit',''),
            row.get('range',''),
            status_cell
        ])
    
    results_table = Table(table_data, colWidths=[2*inch, 1*inch, 0.8*inch, 1.5*inch, 1.2*inch])
    results_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E90FF')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('ALIGN', (0,0), (-1,0), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 10),
        ('BOTTOMPADDING', (0,0), (-1,0), 12),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F5F5F5')]),
    ]))
    story.append(results_table)
    story.append(create_section_divider())
    
    # Medical Summary
    story.append(PageBreak())
    story.append(Paragraph("📝 Medical Summary & Recommendations", section_heading))
    story.append(Spacer(1, 0.2*inch))
    
    # Parse and format summary sections
    sections = re.split(r'##\s+', summary_text)
    for sec in sections[1:]:
        lines = sec.strip().split('\n', 1)
        if len(lines) < 2:
            continue
        
        # Section title
        story.append(Paragraph(f"• {lines[0].strip()}", subsection_heading))
        
        # Section content
        content = lines[1].strip()
        # Convert markdown bold to HTML
        content = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', content)
        
        # Split into paragraphs
        paragraphs = content.split('\n')
        for para in paragraphs:
            if para.strip():
                story.append(Paragraph(para.strip(), normal_style))
        
        story.append(Spacer(1, 0.15*inch))
    
    # Footer
    story.append(Spacer(1, 0.4*inch))
    story.append(Table([['']], colWidths=[6.5*inch], style=[
        ('LINEABOVE', (0,0), (-1,-1), 1, colors.grey)
    ]))
    story.append(Spacer(1, 0.1*inch))
    
    disclaimer = ParagraphStyle(
        'Disclaimer',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.grey,
        alignment=1
    )
    story.append(Paragraph(
        "⚠️ This is an AI-generated report for informational purposes only. "
        "Always consult with a qualified healthcare professional for medical advice.",
        disclaimer
    ))

    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer

# ================================
# 6. STREAMLIT UI WITH BETTER FORMATTING
# ================================
st.set_page_config(page_title="MediReport AI", layout="wide", page_icon="🏥")

# Custom CSS for better formatting
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1E90FF 0%, #00BFFF 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .section-divider {
        height: 3px;
        background: linear-gradient(90deg, #1E90FF 0%, transparent 100%);
        margin: 2rem 0;
    }
    .info-box {
        background-color: #E6F2FF;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1E90FF;
        margin: 1rem 0;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header"><h1 style="color: white; margin: 0;">🏥 MediReport AI</h1><p style="color: white; margin: 0;">Your AI-Powered Medical Report Analyzer</p></div>', unsafe_allow_html=True)

uploaded = st.file_uploader("📄 Upload Your Medical Report (PDF)", type=["pdf"])

if uploaded:
    with st.spinner("🔍 Reading PDF..."):
        raw_text = extract_text_from_pdf(uploaded.read())
    
    if not raw_text.strip():
        st.error("❌ No text found in PDF.")
        st.stop()

    with st.spinner("🧠 Analyzing with AI..."):
        structured = extract_structured(raw_text)
        summary = summarize_report(raw_text, structured)

    df = pd.DataFrame(structured.get("tests", []))
    if df.empty:
        st.error("❌ No test data found.")
        st.stop()

    p_name = structured.get("patient_name", "Patient")
    p_age = structured.get("age", "N/A")
    p_gender = structured.get("gender", "N/A")

    fig_bar, gauges = create_plots(df)

    # Patient Info Section
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("## 📋 Patient Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Name", p_name)
    with col2:
        st.metric("Age", p_age)
    with col3:
        st.metric("Gender", p_gender)

    # Visual Insights Section
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("## 📊 Visual Analysis")
    st.plotly_chart(fig_bar, use_container_width=True)
    
    if gauges:
        st.markdown("### Key Health Metrics")
        cols = st.columns(2)
        for idx, gauge in enumerate(gauges):
            with cols[idx % 2]:
                st.plotly_chart(gauge, use_container_width=True)

    # Health Summary Section
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("## 📝 Medical Summary & Recommendations")
    
    sections = re.split(r'##\s+', summary)
    for sec in sections[1:]:
        lines = sec.strip().split('\n', 1)
        if len(lines) < 2:
            continue
        
        with st.expander(f"📌 {lines[0].strip()}", expanded=True):
            st.markdown(lines[1].strip())

    # Download Section
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("## 💾 Download Reports")
    
    col_a, col_b = st.columns(2)
    with col_a:
        csv_data = df.to_csv(index=False).encode()
        st.download_button(
            "📊 Download CSV Data",
            csv_data,
            "medical_report_data.csv",
            "text/csv",
            use_container_width=True
        )
    with col_b:
        try:
            pdf = generate_pdf_report(p_name, p_age, p_gender, summary, fig_bar, gauges, df)
            safe_name = re.sub(r'\W+', '_', p_name)
            st.download_button(
                "📄 Download PDF Report",
                pdf,
                f"MediReport_{safe_name}_{datetime.now().strftime('%Y%m%d')}.pdf",
                "application/pdf",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"PDF generation failed: {str(e)}")

# Footer
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; color: grey; padding: 2rem;'>
    <p><strong>Powered by:</strong> Groq AI • Plotly • ReportLab</p>
    <p style='font-size: 0.9em;'>⚠️ This is an AI-generated analysis. Always consult your healthcare provider for medical advice.</p>
</div>
""", unsafe_allow_html=True)
