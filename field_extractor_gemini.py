import google.generativeai as genai
import os, json, logging
from dotenv import load_dotenv
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

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

def clean_newlines(obj):
    """Recursively replace \n with space in all string values."""
    if isinstance(obj, dict):
        return {k: clean_newlines(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_newlines(v) for v in obj]
    elif isinstance(obj, str):
        return obj.replace("\n", " ")
    return obj


def extract_from_pdf_direct(pdf_path: str, schema: dict, total_schema: dict):
    """
    Send PDF directly to Gemini (no manual text extraction).
    Gemini reads the PDF and returns structured JSON data.
    """
    try:
        logger.info(f"Uploading PDF to Gemini: {pdf_path}")
        pdf_file = genai.upload_file(pdf_path)
        logger.info(f"File uploaded to Gemini with name: {pdf_file.display_name}")
    except Exception as e:
        logger.error(f"PDF upload to Gemini failed: {e}")
        raise HTTPException(status_code=500, detail=f"PDF upload failed: {e}")

    schema_str = json.dumps(schema, indent=2)
    total_str = json.dumps(total_schema, indent=2)

    prompt = (
        f"You are an expert data extractor. Read the attached PDF carefully and extract structured data "
        f"according to the following schema. Return only valid JSON.\n\n"
        f"Schema:\n{schema_str}\n\n"
        f"- Add one JSON object for {total_str}\n"
        f"- Use null for missing values.\n"
        f"- Do not include any explanations or text outside JSON.\n"
    )

    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content([prompt, pdf_file])
        logger.info("Gemini API response received for PDF input")
    except Exception as e:
        logger.error(f"Gemini API call failed: {e}")
        raise HTTPException(status_code=502, detail="Gemini API request failed")

    if not response or not response.text:
        logger.error("Empty or invalid response from Gemini API.")
        raise HTTPException(status_code=500, detail="Gemini API returned empty response")

    # Attempt to parse JSON
    try:
        data = json.loads(response.text)
    except json.JSONDecodeError:
        logger.warning("Invalid JSON returned by Gemini. Attempting repair...")
        try:
            fixed_json = json_repair.repair_json(response.text)
            data = json.loads(fixed_json)
            logger.info("Successfully repaired JSON")
        except Exception as repair_error:
            logger.error(f"Failed to repair JSON: {repair_error}")
            raise HTTPException(status_code=500, detail="Gemini did not return valid JSON")

    # Clean newlines
    return clean_newlines(data)
