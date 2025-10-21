# Minimal OpenAI PDF extraction for medical billing
import os
import json
from fastapi import HTTPException
import pdfplumber
import openai

OPENAI_API = 'test_key' #os.getenv("OPENAI_API_KEY")
if not OPENAI_API:
    raise HTTPException(status_code=500, detail="OPENAI_API_KEY environment variable not set.")
openai.api_key = OPENAI_API

def extract_text_from_pdf(pdf_path: str) -> str:
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        return text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF text extraction failed: {e}")

def extract_from_pdf_openai(pdf_path: str, schema: dict, model_name: str = "gpt-4.1-nano", prompt: str = None):
    pdf_text = extract_text_from_pdf(pdf_path)

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
    "ACCOMMODATION": [
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
    full_prompt = f"{prompt}\n\nHospital Bill Text:\n{pdf_text}\n"

    try:
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "Output only valid JSON. No explanations."},
                {"role": "user", "content": full_prompt},
            ],
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"OpenAI API request failed: {e}")

    if not response or not response.choices or not response.choices[0].message.content:
        raise HTTPException(status_code=500, detail="OpenAI API returned empty response")

    text = response.choices[0].message.content

    # Remove markdown code block markers if present
    stripped = text.strip()
    if stripped.startswith("```"):
        if stripped.startswith("```json"):
            stripped = stripped[len("```json"):].strip()
        else:
            stripped = stripped[len("```"):].strip()
        if stripped.endswith("```"):
            stripped = stripped[:-3].strip()
        text = stripped

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        try:
            import json_repair
            fixed_json = json_repair.repair_json(text)
            data = json.loads(fixed_json)
        except Exception:
            raise HTTPException(status_code=500, detail="OpenAI did not return valid JSON")

    return data
