import streamlit as st
import requests
import os
from pdf2image import convert_from_bytes

st.set_page_config(
    page_title="Zessta Doc Extractor",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("<h1 style='text-align:center;color:#003366'>Zessta Doc Extractor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#555;'>Upload PDF and extract structured data</p>", unsafe_allow_html=True)
st.markdown("---")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload PDF File")
    uploaded_file = st.file_uploader("Choose your PDF", type=["pdf"])

    if uploaded_file is not None:
        # Read PDF bytes once
        pdf_bytes = uploaded_file.read()

        # Save PDF
        save_dir = "/home/ubuntu/hyperthread/gemini_pipeline/upload_pdfs"
        os.makedirs(save_dir, exist_ok=True)
        pdf_path = os.path.join(save_dir, uploaded_file.name)
        with open(pdf_path, "wb") as f:
            f.write(pdf_bytes)
        st.success(f"✅ File saved as: `{uploaded_file.name}`")

        # Convert PDF to images
        with st.spinner("Rendering PDF pages..."):
            pages = convert_from_bytes(pdf_bytes)
            for i, page in enumerate(pages):
                st.image(page, caption=f"Page {i+1}", use_column_width=True)

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
