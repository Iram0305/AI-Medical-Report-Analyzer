# app.py
# MediReport AI – Enhanced Patient Summary (Detailed, Intuitive, Insightful)
# Streamlit Cloud Ready | Groq Free Tier | No Errors

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
# 2. MODELS (November 2025 – NON-DEPRECATED)
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

# ENHANCED SUMMARY PROMPT – DETAILED, INSIGHTFUL, EMPATHETIC
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
        if "decommissioned" in err or "invalid_request_error" in err:
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

    # Auto-flag
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
# 5. UI
# ================================
st.set_page_config(page_title="MediReport AI", layout="centered", page_icon="medical")
st.title("MediReport AI")
st.markdown("### *Your Personal AI Pathologist*")
st.caption("Upload report → Get **clear, caring, detailed insights** in seconds")

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

    st.download_button("Download CSV", dict_to_csv(structured), "report.csv", "text/csv")

# ================================
# 6. FOOTER
# ================================
st.markdown("---")
st.markdown(
    """
    **Powered by Groq AI** • Updated: **November 11, 2025**  
    *No medical advice — consult your doctor.*
    """
)
