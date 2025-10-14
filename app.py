from field_extractor_gpt import extract_from_pdf_openai
import os
import logging
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from models import ExtractRequest
from field_extractor_gemini import extract_from_pdf_direct
import google.generativeai as genai

# ---------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("pdf_extractor")

# ---------------------------------------------------------
# Gemini API Initialization (from environment)
# ---------------------------------------------------------
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_KEY = "AIzaSyCvCV7mOVOJo5ViN4unPvRJddfDVDUanTA"
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set. Please set it before running the app.")
genai.configure(api_key=GEMINI_API_KEY)

# ---------------------------------------------------------
# FastAPI App Configuration
# ---------------------------------------------------------
app = FastAPI(
    title="PDF to Structured JSON Extractor",
    description="Extracts structured fields from PDFs using Gemini or OpenAI models.",
    version="1.0.0"
)

# ---------------------------------------------------------
# Strict JSON Schema (matches requested format)
# ---------------------------------------------------------
STRICT_OUTPUT_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "Hospital Bill Extraction Schema",
    "type": "object",
    "definitions": {
        "line_item": {
            "type": "object",
            "properties": {
                "service_code": {"type": "string"},
                "description_of_service": {"type": "string"},
                "date": {"type": "string"},
                "qty": {"type": "number"},
                "gross_amount": {"type": "number"},
                "discount": {"type": "number"},
                "allocated_amount": {"type": "number"}
            },
            "required": [
                "service_code",
                "description_of_service",
                "date",
                "qty",
                "gross_amount",
                "discount",
                "allocated_amount"
            ],
            "additionalProperties": False
        }
    },
    "properties": {
        "FORMAT": {"type": "string"},
        "BILL_NO": {"type": "string"},
        "PATIENT_NAME": {"type": "string"},
        "IC_PASSPORT_NO": {"type": "string"},
        "VISIT_TYPE": {"type": "string"},
        "ADMISSION_DATE_TIME": {"type": "string"},
        "DISCHARGE_DATE_TIME": {"type": "string"},
        "GL_REFERENCE_NO": {"type": "string"},
        "BILLING_CATEGORY": {
            "type": "array",
            "items": {"$ref": "#/definitions/line_item"}
        },
        "BILLING_SUBCATEGORY_DETAILS": {
            "type": "object",
            "properties": {
                "ACCOMMODATION": {"type": "array", "items": {"$ref": "#/definitions/line_item"}},
                "MEDICAL_RECORD_SERVICES": {"type": "array", "items": {"$ref": "#/definitions/line_item"}},
                "HOSPITAL_SUPPORT_FEES": {"type": "array", "items": {"$ref": "#/definitions/line_item"}},
                "GENERAL_SUPPLIES": {"type": "array", "items": {"$ref": "#/definitions/line_item"}},
                "RADIOGRAPHY_SUPPLIES": {"type": "array", "items": {"$ref": "#/definitions/line_item"}},
                "SURGICAL_SUPPLIES": {"type": "array", "items": {"$ref": "#/definitions/line_item"}},
                "DRUGS_FORMULARY": {"type": "array", "items": {"$ref": "#/definitions/line_item"}},
                "MEDICAL_SUPPLIES": {"type": "array", "items": {"$ref": "#/definitions/line_item"}},
                "LABORATORY": {"type": "array", "items": {"$ref": "#/definitions/line_item"}},
                "DIAGNOSTIC_SERVICES": {"type": "array", "items": {"$ref": "#/definitions/line_item"}},
                "NURSING_SERVICES": {"type": "array", "items": {"$ref": "#/definitions/line_item"}},
                "EMERGENCY_MEDICAL_SERVICE": {"type": "array", "items": {"$ref": "#/definitions/line_item"}},
                "EQUIPMENT_USAGE": {"type": "array", "items": {"$ref": "#/definitions/line_item"}},
                "MEDICAL_GASES": {"type": "array", "items": {"$ref": "#/definitions/line_item"}},
                "OPERATING_ROOM_FEE": {"type": "array", "items": {"$ref": "#/definitions/line_item"}},
                "OPERATING_THEATER_FEES": {"type": "array", "items": {"$ref": "#/definitions/line_item"}},
                "OT_SUPPORT": {"type": "array", "items": {"$ref": "#/definitions/line_item"}},
                "OT_SERVICES": {"type": "array", "items": {"$ref": "#/definitions/line_item"}},
                "OT_SUPPLIES_CONSUMABLES": {"type": "array", "items": {"$ref": "#/definitions/line_item"}},
                "PACKAGE": {"type": "array", "items": {"$ref": "#/definitions/line_item"}},
                "PPE_SUPPLIES": {"type": "array", "items": {"$ref": "#/definitions/line_item"}},
                "PROCEDURES": {"type": "array", "items": {"$ref": "#/definitions/line_item"}},
                "STERILE_ITEMS_AND_SETS": {"type": "array", "items": {"$ref": "#/definitions/line_item"}},
                "PROCEDURE_FEES": {"type": "array", "items": {"$ref": "#/definitions/line_item"}},
                "CONSULTATION_FEES": {"type": "array", "items": {"$ref": "#/definitions/line_item"}},
                "REPORTING_FEES": {"type": "array", "items": {"$ref": "#/definitions/line_item"}}
            },
            "additionalProperties": {
                "type": "array",
                "items": {"$ref": "#/definitions/line_item"}
            }
        },
        "TOTAL_ROOM_CHARGES": {"type": "number"},
        "TOTAL_HOSPITAL_MEDICAL_SERVICES": {"type": "number"},
        "TOTAL_HOSPITAL_CHARGES": {"type": "number"},
        "TOTAL_CONSULTANT_FEES": {"type": "number"},
        "GRAND_TOTAL": {"type": "number"}
    },
    "required": [
        "FORMAT",
        "BILL_NO",
        "PATIENT_NAME",
        "IC_PASSPORT_NO",
        "VISIT_TYPE",
        "ADMISSION_DATE_TIME",
        "DISCHARGE_DATE_TIME",
        "GL_REFERENCE_NO",
        "BILLING_CATEGORY",
        "BILLING_SUBCATEGORY_DETAILS",
        "TOTAL_ROOM_CHARGES",
        "TOTAL_HOSPITAL_MEDICAL_SERVICES",
        "TOTAL_HOSPITAL_CHARGES",
        "TOTAL_CONSULTANT_FEES",
        "GRAND_TOTAL"
    ],
    "additionalProperties": False
}

# ---------------------------------------------------------
# Extraction Prompt (matches requested format)
# ---------------------------------------------------------
EXTRACTION_PROMPT = """
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
- Output ONLY the JSON. Do not include explanations or commentary.
"""

# ---------------------------------------------------------
# Health Check Endpoint
# ---------------------------------------------------------
@app.get("/health", tags=["Health"])
def health_check():
    status = {"app": "ok", "environment": {}, "gemini": {}}
    status["environment"]["GEMINI_API_KEY"] = "set" if GEMINI_API_KEY else "missing"

    try:
        model = genai.GenerativeModel("gemini-2.5-flash-lite")
        response = model.generate_content("ping")
        status["gemini"]["status"] = "reachable"
        status["gemini"]["model_used"] = "gemini-2.5-flash-lite"
        status["gemini"]["test_output"] = response.text.strip() if getattr(response, "text", None) else "no response"
    except Exception as e:
        status["gemini"]["status"] = "unreachable"
        status["gemini"]["error"] = str(e)

    return JSONResponse(content=status, status_code=200)

# ---------------------------------------------------------
# PDF Extraction Endpoint (Gemini)
# ---------------------------------------------------------
@app.post("/extract", tags=["PDF Extraction"])
def extract(request: ExtractRequest):
    logger.info(f"Received extract request for {request.pdf_path}")
    schema_to_use = request.schema if request.schema else STRICT_OUTPUT_SCHEMA
    try:
        result = extract_from_pdf_direct(
            request.pdf_path,
            schema_to_use,
            request.total_schema,
            prompt=EXTRACTION_PROMPT
        )
        return {"status": "success", "data": result}
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

# ---------------------------------------------------------
# OpenAI Extraction Endpoints
# ---------------------------------------------------------
@app.post("/extract-gpt4-1-nano", tags=["PDF Extraction"])
def extract_openai_gpt4_1_nano(request: ExtractRequest):
    logger.info(f"OpenAI GPT-4.1 nano extract for {request.pdf_path}")
    schema_to_use = request.schema if request.schema else STRICT_OUTPUT_SCHEMA
    try:
        result = extract_from_pdf_openai(
            request.pdf_path,
            schema_to_use,
            model_name="gpt-4.1-nano",
            prompt=EXTRACTION_PROMPT
        )
        return {"status": "success", "data": result}
    except Exception as e:
        logger.error(f"OpenAI GPT-4.1 nano extraction failed: {e}")
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.post("/extract-gpt5-nano", tags=["PDF Extraction"])
def extract_openai_gpt5_nano(request: ExtractRequest):
    logger.info(f"OpenAI GPT-5 nano extract for {request.pdf_path}")
    schema_to_use = request.schema if request.schema else STRICT_OUTPUT_SCHEMA
    try:
        result = extract_from_pdf_openai(
            request.pdf_path,
            schema_to_use,
            model_name="gpt-5-nano",
            prompt=EXTRACTION_PROMPT
        )
        return {"status": "success", "data": result}
    except Exception as e:
        logger.error(f"OpenAI GPT-5 nano extraction failed: {e}")
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})
