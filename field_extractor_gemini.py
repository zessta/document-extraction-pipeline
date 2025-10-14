import os
import json
import logging
from dotenv import load_dotenv
import google.generativeai as genai
import json_repair
from fastapi import HTTPException

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("pdf_extractor")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    logger.error("Missing GEMINI_API_KEY in environment variables.")
    raise ValueError("Missing GEMINI_API_KEY in environment variables.")
genai.configure(api_key=GEMINI_API_KEY)

def clean_newlines(obj):
    if isinstance(obj, dict):
        return {k: clean_newlines(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_newlines(v) for v in obj]
    if isinstance(obj, str):
        return obj.replace("\n", " ")
    return obj

def extract_from_pdf_direct(pdf_path: str, schema: dict, total_schema: dict = None, prompt: str = None):
    """
    Send PDF directly to Gemini (no manual text extraction). Gemini reads the PDF and returns structured JSON data.
    """
    try:
        logger.info(f"Uploading PDF to Gemini: {pdf_path}")
        pdf_file = genai.upload_file(pdf_path)
        logger.info(f"File uploaded to Gemini with name: {getattr(pdf_file, 'display_name', 'uploaded_file')}")
    except Exception as e:
        logger.error(f"PDF upload to Gemini failed: {e}")
        raise HTTPException(status_code=500, detail=f"PDF upload failed: {e}")

    if prompt is None:
        prompt = """
            Read the attached medical bill PDF and extract detailed information for all possible billing subcategories, including but not limited to:

            ACCOMMODATION, MEDICAL RECORD SERVICES, HOSPITAL SUPPORT FEES, GENERAL SUPPLIES, RADIOGRAPHY SUPPLIES, SURGICAL SUPPLIES, DRUGS FORMULARY, MEDICAL SUPPLIES, LABORATORY, DIAGNOSTIC SERVICES, NURSING SERVICES, EMERGENCY MEDICAL SERVICE, EQUIPMENT USAGE, MEDICAL GASES, OPERATING ROOM FEE, OPERATING THEATER FEES, OT-SUPPORT, OT SERVICES, OT SUPPLIES & CONSUMABLES, PACKAGE, PPE SUPPLIES, PROCEDURES, STERILE ITEMS AND SETS, PROCEDURE FEES, CONSULTATION FEES, REPORTING FEES.

            For each line item within these subcategories, extract:

                Service Code (string, or "")
                Description of Service (string, or "")
                Date (string, or "")
                Quantity (number, or 0)
                Gross Amount (number, or 0)
                Discount (number, or 0)
                Allocated Amount (number, or 0)

            Return a structured JSON with this format:

            {
            "FORMAT": "",
            "BILL_NO": "",
            "PATIENT_NAME": "",
            "IC_PASSPORT_NO": "",
            "VISIT_TYPE": "",
            "ADMISSION_DATE_TIME": "",
            "DISCHARGE_DATE_TIME": "",
            "GL_REFERENCE_NO": "",
            "BILLING_CATEGORY": [
                {
                "service_code": "",
                "description_of_service": "",
                "date": "",
                "qty": 0,
                "gross_amount": 0,
                "discount": 0,
                "allocated_amount": 0
                }
            ],
            "BILLING_SUBCATEGORY_DETAILS": {
                "ACCOMMODATION": [],
                "MEDICAL_RECORD_SERVICES": [],
                "HOSPITAL_SUPPORT_FEES": [],
                "GENERAL_SUPPLIES": [],
                "RADIOGRAPHY_SUPPLIES": [],
                "SURGICAL_SUPPLIES": [],
                "DRUGS_FORMULARY": [],
                "MEDICAL_SUPPLIES": [],
                "LABORATORY": [],
                "DIAGNOSTIC_SERVICES": [],
                "NURSING_SERVICES": [],
                "EMERGENCY_MEDICAL_SERVICE": [],
                "EQUIPMENT_USAGE": [],
                "MEDICAL_GASES": [],
                "OPERATING_ROOM_FEE": [],
                "OPERATING_THEATER_FEES": [],
                "OT_SUPPORT": [],
                "OT_SERVICES": [],
                "OT_SUPPLIES_CONSUMABLES": [],
                "PACKAGE": [],
                "PPE_SUPPLIES": [],
                "PROCEDURES": [],
                "STERILE_ITEMS_AND_SETS": [],
                "PROCEDURE_FEES": [],
                "CONSULTATION_FEES": [],
                "REPORTING_FEES": []
            },
            "TOTAL_ROOM_CHARGES": 0,
            "TOTAL_HOSPITAL_MEDICAL_SERVICES": 0,
            "TOTAL_HOSPITAL_CHARGES": 0,
            "TOTAL_CONSULTANT_FEES": 0,
            "GRAND_TOTAL": 0
            }

            Rules:
            - For any redacted fields, leave the value as an empty string ("").
            - Output ONLY the JSON. No explanations.
        """

    try:
        model = genai.GenerativeModel("gemini-2.5-flash-lite")
        response = model.generate_content([prompt, pdf_file])
        logger.info("Gemini API response received for PDF input")
    except Exception as e:
        logger.error(f"Gemini API call failed: {e}")
        raise HTTPException(status_code=502, detail="Gemini API request failed")

    if not response or not getattr(response, "text", None):
        logger.error("Empty or invalid response from Gemini API.")
        raise HTTPException(status_code=500, detail="Gemini API returned empty response")

    raw = response.text.strip()
    # Try parse JSON, repair if needed
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        try:
            fixed = json_repair.repair_json(raw)
            data = json.loads(fixed)
        except Exception as repair_error:
            logger.error(f"Failed to repair JSON: {repair_error}")
            raise HTTPException(status_code=500, detail="Gemini did not return valid JSON")

    return clean_newlines(data)
