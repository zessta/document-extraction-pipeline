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

def _strip_code_fences(text: str) -> str:
    """
    If the model returns a fenced code block (```json ... ```), extract the JSON inside.
    Otherwise return text unchanged.
    """
    if not isinstance(text, str):
        return text
    s = text.strip()
    if s.startswith("```"):
        # find first fence end of the opening line
        first_nl = s.find("\n")
        if first_nl == -1:
            return s
        body = s[first_nl + 1 :]
        end = body.rfind("```")
        if end != -1:
            return body[:end].strip()
    return s

def _stringify_values(obj):
    if isinstance(obj, dict):
        return {k: _stringify_values(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_stringify_values(v) for v in obj]
    if obj is None:
        return ""
    return str(obj)

_HOSPITAL_MED_SUBCATS = [
    "DIAGNOSTIC SERVICES","DRUGS FORMULARY","EMERGENCY MEDICAL SERVICE","EQUIPMENT USAGE",
    "GENERAL SUPPLIES","HOSPITAL SUPPORT FEES","LABORATORY","MEDICAL GASES","MEDICAL RECORD SERVICES",
    "MEDICAL SUPPLIES","NURSING SERVICES","OPERATING ROOM FEE","OPERATING THEATER FEES","OT SERVICES",
    "OT SUPPLIES & CONSUMABLES","OT-SUPPORT","PACKAGE","PPE SUPPLIES","PROCEDURES","RADIOGRAPHY SUPPLIES",
    "STERILE ITEMS AND SETS","SURGICAL SUPPLIES"
]
_CONSULT_SUBCATS = ["CONSULTATION FEES","PROCEDURE FEES","REPORTING FEES"]

def _normalize_output(data: dict) -> dict:
    if not isinstance(data, dict):
        return data
    data.setdefault("room_charges", {})
    if isinstance(data["room_charges"], dict):
        data["room_charges"].setdefault("ACCOMMODATION", [])

    data.setdefault("hospital_medical_services", {})
    if isinstance(data["hospital_medical_services"], dict):
        for k in _HOSPITAL_MED_SUBCATS:
            data["hospital_medical_services"].setdefault(k, [])

    data.setdefault("consultation_fees", {})
    if isinstance(data["consultation_fees"], dict):
        for k in _CONSULT_SUBCATS:
            data["consultation_fees"].setdefault(k, [])

    return data

def extract_from_pdf_direct(pdf_path: str, schema: dict, total_schema: dict = None, prompt: str = None):
    """
    Send PDF directly to Gemini (no manual text extraction). Gemini reads the PDF and returns structured JSON data.
    """
    # Load PDF as bytes and prepare inline content part (avoid ragStoreName requirement)
    try:
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        pdf_part = {"mime_type": "application/pdf", "data": pdf_bytes}
        logger.info(f"Loaded PDF file into memory: {len(pdf_bytes)} bytes")
    except Exception as e:
        logger.error(f"Failed to read PDF: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to read PDF: {e}")

    if prompt is None:
        prompt = """
            You are an intelligent data extraction assistant specializing in medical documents. Your task is to accurately extract all relevant information from the provided hospital bill text and format it into a single JSON object.

            Please adhere strictly to the following instructions:

            Follow the Schema: The final output must match the JSON structure provided below.
            Categorize Correctly: Place each line item into its correct category (room_charges, hospital_medical_services, or consultation_fees) and sub-category (e.g., DRUGS FORMULARY, PROCEDURE FEES).
            Include All Sub-categories: You must include every sub-category key listed in the schema. If no line items fall into a specific sub-category, represent it with an empty array []. Do not omit any keys.
            Extract All Fields: For each line item, extract all details: service_code, description, date, quantity, gross_amount, discount, allocated_amount.
            Maintain Data Type: All extracted values must be strings in the JSON.

            {
            "format": "",
            "bill_no": "",
            "provider_name": "",
            "patient_name": "",
            "ic/passport_no": "",
            "visit_type": "",
            "admission_date_time": "",
            "discharge_date_time": "",
            "gl_reference_no": "",
            "room_charges": {
                "ACCOMMODATION": []
            },
            "hospital_medical_services": {
                "DIAGNOSTIC SERVICES": [],
                "DRUGS FORMULARY": [],
                "EMERGENCY MEDICAL SERVICE": [],
                "EQUIPMENT USAGE": [],
                "GENERAL SUPPLIES": [],
                "HOSPITAL SUPPORT FEES": [],
                "LABORATORY": [],
                "MEDICAL GASES": [],
                "MEDICAL RECORD SERVICES": [],
                "MEDICAL SUPPLIES": [],
                "NURSING SERVICES": [],
                "OPERATING ROOM FEE": [],
                "OPERATING THEATER FEES": [],
                "OT SERVICES": [],
                "OT SUPPLIES & CONSUMABLES": [],
                "OT-SUPPORT": [],
                "PACKAGE": [],
                "PPE SUPPLIES": [],
                "PROCEDURES": [],
                "RADIOGRAPHY SUPPLIES": [],
                "STERILE ITEMS AND SETS": [],
                "SURGICAL SUPPLIES": []
            },
            "consultation_fees": {
                "CONSULTATION FEES": [],
                "PROCEDURE FEES": [],
                "REPORTING FEES": []
            },
            "total_room_charges": "",
            "total_hospital_medical_services": "",
            "total_consultant_fees": "",
            "grand_total": ""
            }

            Rules:
            - Output ONLY the JSON object. No explanations or markdown.
            - Include all sub-category keys exactly as shown, even if empty.
        """

    try:
        model = genai.GenerativeModel("gemini-2.5-flash-lite")
        # Use inline PDF part instead of upload_file (avoids ragStoreName requirement)
        response = model.generate_content([prompt, pdf_part])
        logger.info("Gemini API response received for PDF input")
    except Exception as e:
        logger.error(f"Gemini API call failed: {e}")
        raise HTTPException(status_code=502, detail="Gemini API request failed")

    if not response or not getattr(response, "text", None):
        logger.error("Empty or invalid response from Gemini API.")
        raise HTTPException(status_code=500, detail="Gemini API returned empty response")

    raw = response.text.strip()
    raw = _strip_code_fences(raw)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        try:
            fixed = json_repair.repair_json(raw)
            data = json.loads(fixed)
        except Exception as repair_error:
            logger.error(f"Failed to repair JSON: {repair_error}")
            raise HTTPException(status_code=500, detail="Gemini did not return valid JSON")

    # Enforce required subcategory keys and all-string leaf values
    data = _normalize_output(data)
    data = _stringify_values(data)
    return clean_newlines(data)
