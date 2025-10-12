import streamlit as st
import requests
import base64
import os
import streamlit.components.v1 as components

# -----------------------------
# 🧭 Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Zessta Doc Extractor",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -----------------------------
# 🎨 Elegant Custom Styling
# -----------------------------
st.markdown("""
<style>
    body { background-color: #f8f9fa; }
    .main { background-color: #ffffff; padding: 2rem; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); font-family: 'Inter', sans-serif; }
    h1 { color: #003366; text-align: center; font-weight: 700; font-size: 2.2rem; letter-spacing: 0.5px; margin-bottom: 1.2rem; }
    h2, h3 { color: #004080; font-weight: 600; }
    .stButton>button { background-color: #004080; color: white; border: none; border-radius: 6px; padding: 0.6em 1.2em; font-weight: 500; transition: background-color 0.3s ease; }
    .stButton>button:hover { background-color: #0059b3; }
    .stJson { border-radius: 8px; background-color: #f4f6f8 !important; padding: 10px !important; }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# 🧾 App Header
# -----------------------------
st.markdown("<h1>Zessta Doc Extractor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#555;'>Upload PDF and extract structured data</p>", unsafe_allow_html=True)
st.markdown("---")

# -----------------------------
# 📂 Upload PDF
# -----------------------------
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload PDF File")
    uploaded_file = st.file_uploader("Choose your PDF", type=["pdf"])

    if uploaded_file is not None:
        # Convert PDF to base64
        base64_pdf = base64.b64encode(uploaded_file.read()).decode('utf-8')
        pdf_display = f"""
        <iframe
            src="data:application/pdf;base64,{base64_pdf}"
            width="100%"
            height="900px"
            style="border: none;"
        ></iframe>
        """
        st.markdown(pdf_display, unsafe_allow_html=True)

    if uploaded_file is not None:
        save_dir = "/home/ubuntu/hyperthread/gemini_pipeline/upload_pdfs"
        os.makedirs(save_dir, exist_ok=True)
        pdf_path = os.path.join(save_dir, uploaded_file.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"✅ File saved as: `{uploaded_file.name}`")

        # # Convert PDF to base64 for embedding

        # with open(pdf_path, "rb") as f:
        #     base64_pdf = base64.b64encode(f.read()).decode("utf-8")

        # pdf_display = f"""
        # <div class="pdf-viewer">
        #     <iframe
        #         src="data:application/pdf;base64,{base64_pdf}"
        #         width="100%"
        #         height="900px"
        #         style="border:none;"
        #     ></iframe>
        # </div>
        # """
        # components.html(pdf_display, height=900)
# -----------------------------
# 🧠 Extraction Section
# -----------------------------
with col2:
    st.subheader("Extracted Data")

    if uploaded_file is not None:
        schema = {
            "FORMAT": "",
            "BILL_NO": "",
            "PATIENT_NAME": "",
            "IC_OR_PASSPORT_NO": "",
            "VISIT_TYPE": "",
            "ADMISSION_DATE_TIME": "",
            "DISCHARGE_DATE_TIME": "",
            "GL_REFERENCE_NO": "",
            "BILLING_CATEGORY": [
                {
                    "SERVICE_CODE": "",
                    "DESCRIPTION_OF_SERVICE": "",
                    "DATE": "",
                    "QTY": "",
                    "GROSS_AMOUNT": "",
                    "DISCOUNT": "",
                    "ALLOCATED_AMOUNT": ""
                }
            ]
        }

        total_schema = {
            "BILLING_SUBCATEGORY": {
                "ACCOMMODATION": "",
                "HOSPITAL_SUPPORT_FEES": "",
                "LABORATORY": "",
                "DRUGS_FORMULARY": "",
                "PROCEDURES": "",
                "CONSULTATION_FEES": "",
                "GRAND_TOTAL": ""
            }
        }

        st.markdown("<hr>", unsafe_allow_html=True)

        if st.button("🚀 Extract Data"):
            payload = {
                "pdf_path": pdf_path,
                "schema": schema,
                "total_schema": total_schema
            }
            with st.spinner("Extracting data... please wait ⏳"):
                try:
                    response = requests.post("http://127.0.0.1:8080/extract", json=payload)
                    if response.status_code == 200:
                        st.success("✅ Extraction successful!")
                        try:
                            result = response.json()
                            st.json(result)
                        except Exception:
                            st.write(response.text)
                    else:
                        st.error(f"❌ Error {response.status_code}")
                        st.write(response.text)
                except Exception as e:
                    st.error(f"⚠️ Failed to connect: {e}")
