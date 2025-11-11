# app.py
# MediReport AI – Single File, Free, Streamlit Cloud Ready (FIXED: Escaped prompts for Groq)
# Uses Groq (free), EasyOCR (CPU), PyMuPDF
# Deploy with: requirements.txt + runtime.txt + secrets.toml

import streamlit as st
import fitz  # PyMuPDF
import easyocr
import requests
import json
import re
import csv
import io
from PIL import Image
import numpy as np

# ================================
# 1. CONFIG: GROQ API via st.secrets
# ================================
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except KeyError:
    st.error("Please set `GROQ_API_KEY` in **Streamlit Secrets** (Settings > Secrets).")
    st.info("Example: `GROQ_API_KEY = \"gsk_your_key_here\"`")
    st.stop()

if not GROQ_API_KEY.startswith("gsk_"):
    st.error("Invalid Groq API key format. Must start with `gsk_`.")
    st.stop()

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
OCR_READER = easyocr.Reader(['en'], gpu=False)

# ================================
# 2. PROMPTS (Embedded & Escaped for JSON Safety)
# ================================
# Escape JSON example to prevent payload malformation
JSON_EXAMPLE = {
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
    ],
    "impression": ""
}
JSON_EXAMPLE_STR = json.dumps(JSON_EXAMPLE, indent=2)

EXTRACT_PROMPT_TEMPLATE = """
Extract the following in JSON only. No extra text.

{json_example}

Text:
{{TEXT}}
"""

SUMMARY_PROMPT_TEMPLATE = """
Convert this medical report into a simple 4-5 bullet summary for a patient.
Use **bold** for abnormal values.

Report:
{{TEXT}}

Data:
{{DATA}}
"""

# ================================
# 3. UTILS: PDF, OCR, AI, CSV
# ================================
def extract_text_from_pdf(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    return "\n".join(page.get_text() for page in doc)

def ocr_image(pil_image):
    img_np = np.array(pil_image)
    results = OCR_READER.readtext(img_np, detail=0, paragraph=True)
    return "\n".join(results)

def call_groq(prompt, model="llama3-8b-8192"):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 1024,
        "n": 1  # Explicitly set to 1 (Groq requirement, prevents 400)
    }
    try:
        resp = requests.post(GROQ_URL, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 400:
            return json.dumps({"error": f"400 Bad Request: {e.response.text}"})
        return json.dumps({"error": str(e)})
    except Exception as e:
        return json.dumps({"error": str(e)})

def extract_structured(text):
    prompt = EXTRACT_PROMPT_TEMPLATE.format(
        json_example=JSON_EXAMPLE_STR,
        TEXT=text[:12000]
    )
    result = call_groq(prompt, model="llama3-8b-8192")
    try:
        data = json.loads(result)
    except:
        data = {"error": "Failed to parse AI response", "raw": result}

    # Auto-flag abnormal values
    for t in data.get("tests", []):
        try:
            val_match = re.search(r"[\d.]+", t["value"])
            if not val_match: continue
            val = float(val_match.group())
            range_match = re.findall(r"[\d.]+", t["range"] or "")
            if len(range_match) >= 2:
                low, high = float(range_match[0]), float(range_match[-1])
                t["flag"] = "High" if val > high else "Low" if val < low else "Normal"
            else:
                t["flag"] = "Normal"
        except:
            t["flag"] = "Unknown"
    return data

def summarize_report(text, data):
    prompt = SUMMARY_PROMPT_TEMPLATE.replace("{{TEXT}}", text[:6000]) \
                                   .replace("{{DATA}}", json.dumps(data, indent=2))
    return call_groq(prompt, model="llama3-70b-8192")

def dict_to_csv(data):
    output = io.StringIO()
    writer = csv.writer(output)
    for k, v in data.items():
        if k not in ["tests", "error", "raw"]:
            writer.writerow([k.replace("_", " ").title(), v])
    writer.writerow([])
    writer.writerow(["Test", "Value", "Unit", "Range", "Flag"])
    for t in data.get("tests", []):
        writer.writerow([
            t.get("name", ""),
            t.get("value", ""),
            t.get("unit", ""),
            t.get("range", ""),
            t.get("flag", "")
        ])
    return output.getvalue()

# ================================
# 4. STREAMLIT UI
# ================================
st.set_page_config(page_title="MediReport AI", layout="centered", page_icon="🩺")
st.title("🩺 MediReport AI")
st.caption("Upload medical report (PDF or Image) → Get AI-powered summary instantly")

uploaded = st.file_uploader(
    "Upload Report", 
    type=["pdf", "png", "jpg", "jpeg"],
    help="Supports scanned reports via OCR"
)

if uploaded:
    with st.spinner("Reading document..."):
        if uploaded.type == "application/pdf":
            raw_text = extract_text_from_pdf(uploaded.read())
        else:
            img = Image.open(uploaded)
            raw_text = ocr_image(img)

    if not raw_text.strip():
        st.error("No text found. Try a clearer image or PDF.")
        st.stop()

    st.info(f"Extracted text preview: {raw_text[:200]}...")  # Debug: Show snippet

    with st.spinner("Analyzing with AI (Groq)..."):
        structured = extract_structured(raw_text)
        summary = summarize_report(raw_text, structured)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Structured Data")
        st.json(structured, expanded=False)
    with col2:
        st.subheader("Patient-Friendly Summary")
        st.markdown(summary)

    csv_data = dict_to_csv(structured)
    st.download_button(
        "Download CSV",
        data=csv_data,
        file_name="medical_report_summary.csv",
        mime="text/csv"
    )

# ================================
# 5. FOOTER
# ================================
st.markdown("---")
st.markdown(
    """
    **Powered by:**  
    [Groq](https://groq.com) (Free Tier) • EasyOCR (CPU) • PyMuPDF • Streamlit  
    Made in India | November 11, 2025
    """
)
st.caption("For support: Check logs or reboot app if issues persist.")
