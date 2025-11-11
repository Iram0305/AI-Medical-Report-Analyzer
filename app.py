# app.py
# MediReport AI – WITH INTERACTIVE PLOTS, GAUGES, RADAR CHARTS
# Streamlit Cloud Ready | Groq Free | November 11, 2025

import streamlit as st
import fitz
import easyocr
import requests
import json
import re
import csv
import io
import pandas as pd
import numpy as np
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

EXTRACTION_MODEL = "llama-3.1-8b-instant"
SUMMARY_MODEL = "llama-3.3-70b-versatile"
FALLBACK_MODEL = "llama3-8b-8192"

# ================================
# 2. PROMPTS
# ================================
JSON_EXAMPLE_STR = json.dumps({
    "patient_name": "", "age": "", "gender": "", "report_date": "",
    "tests": [{"name": "", "value": "", "unit": "", "range": "", "flag": ""}],
    "impression": ""
}, indent=2)

EXTRACT_PROMPT = f"Extract in JSON only:\n{JSON_EXAMPLE_STR}\n\nText:\n{{TEXT}}"

SUMMARY_PROMPT = """
You are a compassionate senior doctor. Write a **detailed, empathetic, insightful** summary in **12+ bullet points**.

Include:
- Patient name, age, gender
- **Bold abnormal values**
- % deviation from normal
- Possible causes
- Lifestyle tips
- Urgency level
- Next steps

Use the full report and structured data.

Report:
{{TEXT}}

Data:
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
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
               "temperature": 0.3, "max_tokens": 1500, "n": 1}
    try:
        resp = requests.post(GROQ_URL, json=payload, headers=headers, timeout=40)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        if "decommissioned" in str(e).lower():
            return call_groq(prompt, FALLBACK_MODEL)
        return json.dumps({"error": str(e)})

def extract_structured(text):
    result = call_groq(EXTRACT_PROMPT.replace("{{TEXT}}", text[:10000]))
    try: data = json.loads(result)
    except: data = {"error": "Parse failed", "raw": result}

    for t in data.get("tests", []):
        try:
            val = float(re.search(r"[\d.]+", t["value"]).group())
            rng = re.findall(r"[\d.]+", t["range"] or "")
            if len(rng) >= 2:
                low, high = float(rng[0]), float(rng[-1])
                mid = (low + high) / 2
                dev = ((val - mid) / mid) * 100
                t["dev_pct"] = round(dev, 1)
                t["flag"] = "High" if val > high else "Low" if val < low else "Normal"
            else:
                t["dev_pct"], t["flag"] = 0, "Normal"
        except:
            t["dev_pct"], t["flag"] = 0, "Unknown"
    return data

def summarize_report(text, data):
    prompt = SUMMARY_PROMPT.replace("{{TEXT}}", text[:3000]).replace("{{DATA}}", json.dumps(data, indent=2))
    return call_groq(prompt, SUMMARY_MODEL)

def dict_to_csv(data):
    output = io.StringIO()
    writer = csv.writer(output)
    for k, v in data.items():
        if k not in ["tests", "error", "raw"]:
            writer.writerow([k.replace("_", " ").title(), v])
    writer.writerow([])
    writer.writerow(["Test", "Value", "Unit", "Range", "Flag", "% Dev"])
    for t in data.get("tests", []):
        writer.writerow([t["name"], t["value"], t.get("unit",""), t.get("range",""), t.get("flag",""), t.get("dev_pct","")])
    return output.getvalue()

# ================================
# 4. PLOTTING FUNCTIONS
# ================================
def plot_bar_chart(tests_df):
    fig = px.bar(
        tests_df, x='name', y='value_num',
        color='flag', color_discrete_map={'High': 'red', 'Low': 'blue', 'Normal': 'green'},
        title="Lab Values vs Normal Range",
        labels={'value_num': 'Value', 'name': 'Test'},
        hover_data={'low': True, 'high': True, 'dev_pct': True}
    )
    fig.add_scatter(x=tests_df['name'], y=tests_df['low'], mode='lines', name='Low', line=dict(dash='dot'))
    fig.add_scatter(x=tests_df['name'], y=tests_df['high'], mode='lines', name='High', line=dict(dash='dot'))
    fig.update_layout(height=500)
    return fig

def plot_gauges(tests_df):
    key_tests = ['Hemoglobin', 'Glucose', 'Cholesterol Total', 'Creatinine']
    figs = []
    for test in key_tests:
        row = tests_df[tests_df['name'].str.contains(test, case=False)]
        if row.empty: continue
        val = row.iloc[0]['value_num']
        low, high = row.iloc[0]['low'], row.iloc[0]['high']
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=val,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': row.iloc[0]['name']},
            delta={'reference': (low + high)/2},
            gauge={'axis': {'range': [None, high*1.2]},
                   'bar': {'color': "cyan"},
                   'steps': [{'range': [low, high], 'color': "lightgreen"}]}
        ))
        fig.update_layout(height=250)
        figs.append(fig)
    return figs

def plot_radar(tests_df):
    categories = ['Anemia Risk', 'Diabetes Risk', 'Heart Risk', 'Liver Stress', 'Kidney Health']
    values = [0]*5
    for i, cat in enumerate(categories):
        if 'anemia' in cat.lower() or 'hemoglobin' in str(tests_df).lower():
            values[i] = 80 if any(tests_df['flag']=='Low') else 20
        elif 'diabetes' in cat.lower():
            values[i] = 90 if any(tests_df['flag']=='High') and 'glucose' in str(tests_df).lower() else 10
        elif 'heart' in cat.lower():
            values[i] = 75 if any(tests_df['flag']=='High') and 'cholesterol' in str(tests_df).lower() else 15
        elif 'liver' in cat.lower():
            values[i] = 60 if any(tests_df['flag']=='High') and 'sgot' in str(tests_df).lower() else 10
        else:
            values[i] = 20
    values += values[:1]
    fig = go.Figure(data=go.Scatterpolar(r=values[:-1], theta=categories, fill='toself'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False, height=400)
    return fig

# ================================
# 5. UI
# ================================
st.set_page_config(page_title="MediReport AI", layout="wide", page_icon="chart")
st.title("MediReport AI")
st.markdown("### *Your AI Pathologist with Visual Insights*")

uploaded = st.file_uploader("Upload Report", type=["pdf", "png", "jpg", "jpeg"])

if uploaded:
    with st.spinner("Extracting text..."):
        raw_text = extract_text_from_pdf(uploaded.read()) if uploaded.type == "application/pdf" else ocr_image(Image.open(uploaded))
    
    if not raw_text.strip():
        st.error("No text found.")
        st.stop()

    with st.spinner("Analyzing..."):
        structured = extract_structured(raw_text)
        summary = summarize_report(raw_text, structured)

    # Prepare DataFrame
    tests = structured.get("tests", [])
    df_data = []
    for t in tests:
        try:
            low = float(re.search(r"[\d.]+", t["range"]).group()) if t["range"] else None
            high = float(re.findall(r"[\d.]+", t["range"])[-1]) if t["range"] and len(re.findall(r"[\d.]+", t["range"])) > 1 else low
            val = float(re.search(r"[\d.]+", t["value"]).group())
            df_data.append({
                'name': t["name"], 'value': t["value"], 'value_num': val,
                'unit': t.get("unit", ""), 'range': t["range"], 'flag': t["flag"],
                'low': low, 'high': high, 'dev_pct': t.get("dev_pct", 0)
            })
        except: pass
    tests_df = pd.DataFrame(df_data)

    # Layout
    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.subheader("Structured Data")
        st.json(structured, expanded=False)
        st.download_button("Download CSV", dict_to_csv(structured), "report.csv", "text/csv")

    with col2:
        st.subheader("Your Health Summary")
        st.markdown(summary)

    st.markdown("---")

    # PLOTS
    tab1, tab2, tab3 = st.tabs(["Bar Chart", "Health Gauges", "Risk Radar"])

    with tab1:
        if not tests_df.empty:
            fig = plot_bar_chart(tests_df)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No numeric tests to plot.")

    with tab2:
        gauges = plot_gauges(tests_df)
        cols = st.columns(len(gauges) if gauges else 1)
        for i, fig in enumerate(gauges):
            with cols[i]:
                st.plotly_chart(fig, use_container_width=True)

    with tab3:
        radar = plot_radar(tests_df)
        st.plotly_chart(radar, use_container_width=True)

# ================================
# 6. FOOTER
# ================================
st.markdown("---")
st.markdown(
    """
    **Powered by Groq AI • Plotly • Streamlit**  
    *Not medical advice. Consult your doctor.*  
    Updated: **November 11, 2025, 03:26 PM IST**
    """
)
