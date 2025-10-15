from field_extractor_gpt import extract_from_pdf_openai
import os
import logging
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from models import ExtractRequest
from field_extractor_gemini import extract_from_pdf_direct
import google.generativeai as genai
from dotenv import load_dotenv  # added

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
load_dotenv()  # added
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # changed to env
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
# Strict JSON Schema (updated to Grouped Hospital Bill Claim Schema)
# ---------------------------------------------------------
STRICT_OUTPUT_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "Grouped Hospital Bill Claim Schema",
    "description": "A schema for extracting fields from hospital bills, with billing information grouped by category and subcategory.",
    "type": "object",
    "definitions": {
        "line_item": {
            "type": "object",
            "properties": {
                "service_code": {"type": "string"},
                "description": {"type": "string"},
                "date": {"type": "string"},
                "quantity": {"type": "string"},
                "gross_amount": {"type": "string"},
                "discount": {"type": "string"},
                "allocated_amount": {"type": "string"}
            },
            "required": [
                "service_code",
                "description",
                "date",
                "quantity",
                "gross_amount",
                "discount",
                "allocated_amount"
            ]
        }
    },
    "properties": {
        "document_details": {
            "type": "object",
            "properties": {
                "format": {"type": "string"},
                "bill_no": {"type": "string"},
                "provider_name": {"type": "string"}
            },
            "required": ["format", "bill_no", "provider_name"]
        },
        "patient_information": {
            "type": "object",
            "properties": {
                "full_name": {"type": "string"},
                "identification_number": {"type": "string"},
                "policy_no": {"type": "string"}
            },
            "required": ["full_name", "identification_number", "policy_no"]
        },
        "claim_details": {
            "type": "object",
            "properties": {
                "visit_type": {"type": "string"},
                "admission_date_time": {"type": "string"},
                "discharge_date_time": {"type": "string"},
                "physician_name": {"type": "string"},
                "gl_reference_no": {"type": "string"}
            },
            "required": [
                "visit_type",
                "admission_date_time",
                "discharge_date_time",
                "physician_name",
                "gl_reference_no"
            ]
        },
        "billing_details": {
            "type": "object",
            "description": "A nested object where keys are the main categories, and their values are objects where keys are subcategories containing an array of line items.",
            "additionalProperties": {
                "type": "object",
                "additionalProperties": {
                    "type": "array",
                    "items": {"$ref": "#/definitions/line_item"}
                }
            }
        },
        "financial_information": {
            "type": "object",
            "properties": {
                "total_room_charges": {"type": "string"},
                "total_hospital_medical_services": {"type": "string"},
                "total_hospital_charges": {"type": "string"},
                "total_consultant_fees": {"type": "string"},
                "grand_total": {"type": "string"}
            },
            "required": [
                "total_room_charges",
                "total_hospital_medical_services",
                "total_hospital_charges",
                "total_consultant_fees",
                "grand_total"
            ]
        }
    },
    "required": [
        "document_details",
        "patient_information",
        "claim_details",
        "billing_details",
        "financial_information"
    ]
}

# ---------------------------------------------------------
# Extraction Prompt (strict, flat grouped output with all sub-categories)
# ---------------------------------------------------------
EXTRACTION_PROMPT = """
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
            prompt=EXTRACTION_PROMPT  # enforce strict prompt for grouped output
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
