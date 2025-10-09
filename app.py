import os
import logging
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from field_extractor_gemini import extract_from_pdf_direct
from models import ExtractRequest
import google.generativeai as genai
import uvicorn

# ---------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("pdf_extractor")

# ---------------------------------------------------------
# Gemini API Initialization
# ---------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("❌ GEMINI_API_KEY environment variable not set. Please export it before running the app.")

# Configure Gemini SDK
genai.configure(api_key=GEMINI_API_KEY)

# ---------------------------------------------------------
# FastAPI App Configuration
# ---------------------------------------------------------
app = FastAPI(
    title="PDF to Structured JSON Extractor",
    description="Extracts structured fields from PDFs using Gemini function calling.",
    version="1.0.0"
)

# ---------------------------------------------------------
# Health Check Endpoint
# ---------------------------------------------------------
@app.get("/health", tags=["Health"])
def health_check():
    """
    Health check endpoint.
    Returns the status of the application, environment variables, and Gemini API connectivity.
    """
    status = {"app": "ok", "environment": {}, "gemini": {}}

    # Check environment variable
    status["environment"]["GEMINI_API_KEY"] = "set" if GEMINI_API_KEY else "missing"

    # Check Gemini API connectivity
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content("ping")
        status["gemini"]["status"] = "reachable"
        status["gemini"]["model_used"] = "gemini-1.5-flash"
        status["gemini"]["test_output"] = response.text.strip() if response.text else "no response"
    except Exception as e:
        status["gemini"]["status"] = "unreachable"
        status["gemini"]["error"] = str(e)

    return JSONResponse(content=status, status_code=200)

# ---------------------------------------------------------
# PDF Extraction Endpoint
# ---------------------------------------------------------
@app.post("/extract", tags=["PDF Extraction"])
def extract(request: ExtractRequest):
    """
    Extract data from a PDF using Gemini API according to a provided schema.
    """
    logger.info(f"Received extract request for {request.pdf_path}")

    try:
        result = extract_from_pdf_direct(request.pdf_path, request.schema, request.total_schema)
        return {"status": "success", "data": result}
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

