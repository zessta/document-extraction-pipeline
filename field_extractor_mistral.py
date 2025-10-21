import os
import json
import logging
from fastapi import HTTPException
try:
    from mistralai import Mistral
except Exception:
    Mistral = None

logger = logging.getLogger("pdf_extractor_mistral")

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Mistral configuration
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "d2mZNOs9W2dgOaBJTW9uygxh67CmClDA")
MISTRAL_API_URL = os.getenv("MISTRAL_API_URL", "https://api.mistral.ai")
# NEW: OCR endpoint (configurable)
MISTRAL_OCR_URL = os.getenv("MISTRAL_OCR_URL", f"{MISTRAL_API_URL.rstrip('/')}/v1/ocr")

if not MISTRAL_API_KEY:
    logger.error("Missing MISTRAL_API_KEY in environment variables.")
    raise ValueError("Missing MISTRAL_API_KEY in environment variables.")

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

# NEW: lightweight PDF text extraction
try:
    import PyPDF2 as pypdf
except Exception:
    pypdf = None

def _extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from PDF using PyPDF2 (pure-python). Raises if lib missing.
    """
    if pypdf is None:
        raise HTTPException(
            status_code=500,
            detail="PyPDF2 not installed. Please `pip install PyPDF2` to enable Mistral text extraction."
        )
    try:
        text_parts = []
        with open(pdf_path, "rb") as f:
            reader = pypdf.PdfReader(f)
            for i, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text() or ""
                except Exception:
                    page_text = ""
                text_parts.append(f"\n\n--- Page {i+1} ---\n{page_text}")
        text = "".join(text_parts).strip()
        # Safety: truncate extremely long docs to keep within token limits
        return text[:300_000]
    except Exception as e:
        logger.error(f"Failed extracting text from PDF: {e}")
        raise HTTPException(status_code=400, detail=f"Failed extracting text from PDF: {e}")

# NEW: encode any file to base64
def _encode_file_to_base64(file_path: str) -> str:
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# NEW: call Mistral OCR API (PDF or image) and return extracted text (markdown/plain)
def _ocr_with_mistral(file_path: str) -> str:
    if not MISTRAL_API_KEY:
        raise HTTPException(status_code=500, detail="Missing MISTRAL_API_KEY")
    ext = os.path.splitext(file_path)[1].lower()
    field = "base64_pdf" if ext == ".pdf" else "base64_image"
    b64 = _encode_file_to_base64(file_path)
    payload = {field: b64, "output_format": "markdown"}
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }
    try:
        logger.info(f"Calling Mistral OCR at {MISTRAL_OCR_URL}")
        resp = requests.post(MISTRAL_OCR_URL, headers=headers, json=payload, timeout=300)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.error(f"Mistral OCR call failed: {e}")
        raise

    # Robust extraction of text field
    text = ""
    if isinstance(data, dict):
        # common keys: "text" / "output" / "result"
        text = data.get("text") or data.get("output") or data.get("result") or ""
        if not text and "data" in data and isinstance(data["data"], dict):
            text = data["data"].get("text", "")
    if not text:
        # fallback to entire JSON string
        text = json.dumps(data, ensure_ascii=False)
    # Truncate to stay within token limits
    return text[:300_000].strip()

def _mistral_client() -> "Mistral":
    """
    Ensure Mistral client is available and configured.
    """
    api_key = "d2mZNOs9W2dgOaBJTW9uygxh67CmClDA" #os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="Missing MISTRAL_API_KEY in environment variables.")
    if Mistral is None:
        raise HTTPException(status_code=500, detail="mistralai SDK not installed. Please `pip install mistralai`.")
    try:
        return Mistral(api_key=api_key)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to init Mistral client: {e}")

def _ocr_text_via_mistral(client: "Mistral", pdf_path: str) -> str:
    """
    Upload file and run OCR using 'mistral-ocr-latest'. Returns extracted text.
    """
    try:
        resp = client.files.upload(
            file={"file_name": os.path.basename(pdf_path), "content": open(pdf_path, "rb")},
            purpose="ocr",
        )
        file_id = resp.id
        logger.info(f"Mistral upload ok: file_id={file_id}")
    except Exception as e:
        logger.error(f"Mistral file upload failed: {e}")
        raise HTTPException(status_code=502, detail=f"Mistral file upload failed: {e}")

    try:
        ocr = client.ocr.process(
            model="mistral-ocr-latest",
            document={"type": "file", "file_id": file_id},
            include_image_base64=False,
        )
    except Exception as e:
        logger.error(f"Mistral OCR failed: {e}")
        raise HTTPException(status_code=502, detail=f"Mistral OCR request failed: {e}")

    text = ""
    try:
        if hasattr(ocr, "content") and ocr.content:
            text = ocr.content
        else:
            text = str(ocr)
    except Exception:
        text = str(ocr)
    text = (text or "").strip()
    if not text:
        raise HTTPException(status_code=500, detail="Mistral OCR returned empty content")
    return text[:300_000]  # keep within token limits

def extract_from_pdf_mistral(pdf_path: str, schema: dict, total_schema: dict = None, prompt: str = None, model: str = "mistral-small-latest"):
    """
    Use Mistral SDK pipeline:
      1) Upload PDF
      2) OCR with mistral-ocr-latest
      3) Chat completion to structure JSON per schema
    Returns normalized, stringified, newline-cleaned dict.
    """
    client = _mistral_client()
    logger.info(f"Processing PDF with Mistral: {pdf_path}")
    ocr_text = _ocr_text_via_mistral(client, pdf_path)
    logger.info(f"OCR text length: {len(ocr_text)}")

    schema_str = json.dumps(schema or {}, ensure_ascii=False, indent=2)
    system_prompt = (
        "You are an intelligent data extraction assistant specializing in medical documents. "
        "Return ONLY a single JSON object matching the schema. No explanations or markdown."
    )
    user_prompt = (prompt or "") + (
        "\n\nCRITICAL INSTRUCTIONS:\n"
        "1. Extract ALL line items.\n"
        "2. Categorize items into correct category/sub-category.\n"
        "3. For each line item, include: service_code, description, date, quantity, gross_amount, discount, allocated_amount.\n"
        "4. All values must be strings.\n"
        "5. Include ALL sub-category keys; use [] if empty.\n"
        "\nREQUIRED JSON SCHEMA:\n"
        f"{schema_str}\n\n"
        "Please extract and format the following OCR text from a hospital bill into the required JSON schema:\n\n"
        f"{ocr_text}"
    )

    try:
        chat = client.chat.complete(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
    except Exception as e:
        logger.error(f"Mistral chat.complete failed: {e}")
        raise HTTPException(status_code=502, detail=f"Mistral chat request failed: {e}")

    raw = ""
    try:
        if getattr(chat, "choices", None):
            raw = chat.choices[0].message.content or ""
        else:
            raw = str(chat)
    except Exception:
        raw = str(chat)
    if not raw:
        raise HTTPException(status_code=500, detail="Mistral chat completion returned empty content")

    raw = _strip_code_fences(raw.strip())
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        try:
            import json_repair
            data = json.loads(json_repair.repair_json(raw))
        except Exception as e:
            logger.error(f"JSON repair failed: {e}")
            raise HTTPException(status_code=500, detail="Mistral did not return valid JSON")

    data = _normalize_output(data)
    data = _stringify_values(data)
    return clean_newlines(data)
