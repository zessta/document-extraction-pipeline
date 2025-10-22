#Run this as python micare_openai.py <file_name> --model <model_name (etc gpt-5-mini)>

import os
import json
import base64
import tempfile
import logging
from typing import Optional, List
from contextlib import ExitStack

from dotenv import load_dotenv
from fastapi import HTTPException
import json_repair
import fitz  # PyMuPDF
from openai import OpenAI

# ---------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("pdf_extractor_imgonly")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def clean_newlines(obj):
    if isinstance(obj, dict):
        return {k: clean_newlines(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_newlines(v) for v in obj]
    if isinstance(obj, str):
        return obj.replace("\n", " ")
    return obj

def _strip_code_fences(text: str) -> str:
    if not isinstance(text, str):
        return text
    s = text.strip()
    if s.startswith("```"):
        first_nl = s.find("\n")
        if first_nl == -1:
            return s
        body = s[first_nl + 1:]
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
    "OT SUPPLIES & CONSUMABLES","OT-SUPPORT", "OT INSTRUMENTS", "PACKAGE","PPE SUPPLIES","PROCEDURES","RADIOGRAPHY SERVICES",
    "STERILE ITEMS AND SETS","SURGICAL SUPPLIES"
]
_CONSULT_SUBCATS = ["CONSULTATION FEES","PROCEDURE FEES","REPORTING FEES"]

def _normalize_output(data: dict) -> dict:
    """
    Normalize model output to match the declared schema:
      - consultation_fees: list of consultant dicts (each with 4 specific keys)
      - room_charges: dict with key "ACCOMMODATION": []
      - hospital_medical_services: dict with all expected subcategory keys -> []
    """
    if not isinstance(data, dict):
        return data

    # --- format/basic top-level fields present (leave as strings later stringified) ---
    # Not strictly necessary to prefill all scalars because _stringify_values handles them,
    # but we enforce container shapes below.

    # --- Room charges ---
    if not isinstance(data.get("room_charges"), dict):
        data["room_charges"] = {"ACCOMMODATION": []}
    else:
        data["room_charges"].setdefault("ACCOMMODATION", [])

    # --- Hospital medical services ---
    hms = data.get("hospital_medical_services")
    if not isinstance(hms, dict):
        hms = {}
    for k in _HOSPITAL_MED_SUBCATS:
        hms.setdefault(k, [])
    data["hospital_medical_services"] = hms

    # --- Consultation fees (LIST of consultant dicts) ---
    # Expected per-consultant object keys:
    CONSULT_KEYS = [
        "NAME OF CONSULTANT",
        "CONSULTATION FEES",
        "PROCEDURE FEES",
        "REPORTING FEES",
        "TOTAL FEE FOR CONSULTANT",
    ]

    def _blank_consultant():
        return {
            "NAME OF CONSULTANT": "",
            "CONSULTATION FEES": [],
            "PROCEDURE FEES": [],
            "REPORTING FEES": [],
            "TOTAL FEE FOR CONSULTANT": "",
        }

    cf = data.get("consultation_fees")

    # If model accidentally returned a dict, wrap it into a list.
    if isinstance(cf, dict):
        cf = [cf]
    # If it's missing or wrong type, start with a single blank consultant.
    if not isinstance(cf, list):
        cf = [_blank_consultant()]

    normalized_cf = []
    for item in cf:
        # If an entry isn't a dict, replace with a blank consultant block.
        if not isinstance(item, dict):
            normalized_cf.append(_blank_consultant())
            continue
        norm = _blank_consultant()
        # Copy known keys if present; ignore extras
        for k in CONSULT_KEYS:
            v = item.get(k, norm[k])
            # Ensure list-shaped keys are lists
            if k in ("CONSULTATION FEES", "PROCEDURE FEES", "REPORTING FEES"):
                v = v if isinstance(v, list) else []
            norm[k] = v
        normalized_cf.append(norm)

    data["consultation_fees"] = normalized_cf

    return data


# ---------------------------------------------------------------------
# Default prompt
# ---------------------------------------------------------------------

_DEFAULT_PROMPT = """
You are an intelligent data extraction assistant specializing in medical invoices. You are given ONLY images of the invoice pages (no extracted text). Read the images and produce a single JSON object that matches the schema below exactly.

Requirements:
- At the top of each page, there will be a field called FORMAT: The value of this field is either SUMMARY or DETAIL BREAK-UP. Only consider the pages where the value of the FORMAT field is SUMMARY. Simply ignore the other other pages. 
- In some pages, the value (SUMMARY / DETAIL BREAK-UP) may be absent. In this case, use the following logic. The document will always contain a set of SUMMARY pages followed by a set of DETAIL BREAK-UP pages. If pages 1, 3, 5 contain the value SUMMARY and page 6 contains the value DETAIL BREAK-UP, then pages 1-5 are the SUMMARY pages. In this case, process exactly these pages (1-5). In case pages 2, 3, 4 contain the value SUMMARY and page 5 onwards contain the keyword DETAIL BREAK-UP, then the summary pages are 1-4. In this case, once again, process exactly these pages. 
- The first page will ALWAYS be a summary page. Be sure to include this in your processing. 
- Follow the schema keys exactly; include every sub-category key even if empty (use []).
- When it comes to CONSULTANT(S) FEES, there might be multiple consultants who have attended to the patient. For each consultant, there will be three setions, REPORTING FEES, PROCEDURE FEES and CONSULTATION FEES. Add these up to get the total fee paid to the consultant (see the schema). Details for each consultant must be in the form of a dictionary (See the schema) and together they must be in the form of a list. Be sure not to clump all the charges for different consultants. 
- For each line item, extract: service_code, description, date, quantity, gross_amount, discount, allocated_amount.
- All values MUST be strings.
- If a value is not present, use an empty string "" (not null).
- Output ONLY the JSON (no prose, no markdown).

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
    "OT INSTRUMENTS": [],
    "PACKAGE": [],
    "PPE SUPPLIES": [],
    "PROCEDURES": [],
    "RADIOLOGY SERVICES": [],
    "STERILE ITEMS AND SETS": [],
    "SURGICAL SUPPLIES": []
},
"consultation_fees": [
    {
    "NAME OF CONSULTANT": "",
    "CONSULTATION FEES": [],
    "PROCEDURE FEES": [],
    "REPORTING FEES": [],
    "TOTAL FEE FOR CONSULTANT": ""
    }
],
"total_room_charges": "",
"total_hospital_medical_services": "",
"total_consultant_fees": "",
"grand_total": ""
}
"""

# ---------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------

def _pdf_to_png_tempfiles(pdf_path: str, dpi: int = 200, max_pages: Optional[int] = None) -> List[str]:
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to open PDF: {e}")

    png_paths: List[str] = []
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    for i, page in enumerate(doc):
        if max_pages and i >= max_pages:
            break
        pix = page.get_pixmap(matrix=mat, alpha=False)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f".p{i+1}.png")
        tmp.write(pix.tobytes("png"))
        tmp.flush()
        tmp.close()
        png_paths.append(tmp.name)
    doc.close()
    if not png_paths:
        raise HTTPException(status_code=400, detail="PDF has no renderable pages.")
    return png_paths

def _response_text_or_none(resp):
    if hasattr(resp, "output_text") and resp.output_text:
        return resp.output_text
    if hasattr(resp, "output") and resp.output:
        for o in resp.output:
            if hasattr(o, "content"):
                for c in o.content:
                    if getattr(c, "type", None) == "output_text":
                        return getattr(c, "text", None)
    return None

def extract_from_pdf_direct(
    pdf_path: str,
    schema: dict,
    total_schema: dict = None,
    prompt: str = None,
    model: str = "gpt-4o",
    dpi: int = 200,
    max_pages: Optional[int] = None
):
    # 1) Render PDF pages to temporary PNGs
    png_paths = _pdf_to_png_tempfiles(pdf_path, dpi=dpi, max_pages=max_pages)
    logger.info(f"Rendered {len(png_paths)} page(s) at ~{dpi} DPI")

    # 2) Base64 encode them as data URLs
    img_dataurls = []
    for p in png_paths:
        with open(p, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
            dataurl = f"data:image/png;base64,{b64}"
            img_dataurls.append(dataurl)
            logger.info(f"Encoded {p} ({len(b64)//1024} KB)")
    for p in png_paths:
        try:
            os.remove(p)
        except Exception:
            pass

    # 3) Build the message content
    user_prompt = prompt or _DEFAULT_PROMPT
    content = [{"type": "input_text", "text": user_prompt}]
    content += [{"type": "input_image", "image_url": url} for url in img_dataurls]
    messages = [{"role": "user", "content": content}]

    # 4) Send request to OpenAI

    try:
        resp = client.responses.create(
            model=model,
            input=messages#, When using gpt-5 of gpt-5-mini remove temperature parameter
            #temperature=0.0,
        )
        logger.info("OpenAI response received")
    except Exception as e:
        logger.error(f"OpenAI API request failed: {e}")
        raise HTTPException(status_code=502, detail=f"OpenAI API request failed: {e}")

    # 5) Parse the model output
    raw = _response_text_or_none(resp)
    if not raw or not isinstance(raw, str):
        raise HTTPException(status_code=500, detail="Empty response text from model")
    raw = _strip_code_fences(raw.strip())

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        try:
            fixed = json_repair.repair_json(raw)
            data = json.loads(fixed)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model returned invalid JSON: {e}")

    data = _normalize_output(data)
    data = _stringify_values(data)
    return clean_newlines(data)

# ---------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract invoice fields from PDF using GPT multimodal.")
    parser.add_argument("pdf", help="Path to invoice PDF")
    parser.add_argument("--model", default="gpt-4o", help="Model (e.g., gpt-4o, gpt-5 when available)")
    parser.add_argument("--dpi", type=int, default=200, help="Rendering DPI")
    parser.add_argument("--max-pages", type=int, default=None, help="Limit number of pages")
    args = parser.parse_args()

    try:
        result = extract_from_pdf_direct(
            pdf_path=args.pdf,
            schema={},
            model=args.model,
            dpi=args.dpi,
            max_pages=args.max_pages,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except HTTPException as e:
        logger.error(f"HTTP {e.status_code}: {e.detail}")
        raise
