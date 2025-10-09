
# PDF to Structured JSON Extractor (Gemini API + FastAPI)

This service extracts **structured fields** from PDF documents using **Google Gemini’s function-calling API**.  
It is designed for flexible schema-based extraction — ideal for invoices, hospital bills, and similar structured documents.

---

## 🚀 Features

- Extract structured data fields from PDFs using **Gemini API**
- Define your own schema for extraction
- REST API built with **FastAPI**
- Includes `/health` endpoint to check Gemini API connectivity
- Detailed logging for debugging and monitoring

---

## 🧱 Project Structure

```

.
├── field_extractor_gemini.py     # Logic for PDF text extraction + Gemini-based field mapping
├── models.py                     # Pydantic models (e.g., ExtractRequest)
├── main.py                       # FastAPI application entry point
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation

```

---

## ⚙️ Prerequisites

Before running this service, ensure you have:

- Python **3.9+**
- A valid **Google Gemini API key**
- The following environment variable set:
```bash
  export GEMINI_API_KEY="your_gemini_api_key_here"
```

## Installation

1. **Clone the repository**

```bash
   git clone https://github.com/zessta/document-extraction-pipeline.git
   cd document-extraction-pipeline
```

2. **Create and activate virtual environment**

   ```bash
   python3 -m venv gemini_env
   source gemini_env/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## ▶️ Running the FastAPI App

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Once started, the API will be available at:

* **Docs (Swagger UI):** [http://localhost:8000/docs](http://localhost:8000/docs)
* **Health Check:** [http://localhost:8000/health](http://localhost:8000/health)

---

## 🧠 Example Request

### Endpoint

`POST /extract`

### Request Body

```json
{
  "pdf_path": "/home/ubuntu/hyperthread/Data/pdf_data/sample_bill.pdf",
  "schema": {
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
  },
  "total_schema": {
    "BILLING_SUBCATEGORY": {
      "ACCOMMODATION": "",
      "SURGICAL_SUPPLIES": "",
      "LABORATORY": "",
      "CONSULTATION_FEES": "",
      "PROCEDURE_FEES": ""
    },
    "TOTAL_ROOM_CHARGES": "",
    "TOTAL_HOSPITAL_MEDICAL_SERVICES": "",
    "TOTAL_HOSPITAL_CHARGES": "",
    "TOTAL_CONSULTANT_FEES": "",
    "GRAND_TOTAL": ""
  }
}
```

### Response

```json
{
  "status": "success",
  "data": {
    "BILL_NO": "INV-2025-0910",
    "PATIENT_NAME": "REDACTED",
    "GRAND_TOTAL": "1578.00",
    ...
  }
}
```

---

## 🧩 Health Check Endpoint

**GET /health**

This verifies:

* App is running
* `GEMINI_API_KEY` is configured
* Gemini API is reachable

Example output:

```json
{
  "app": "ok",
  "environment": {
    "GEMINI_API_KEY": "set"
  },
  "gemini": {
    "status": "reachable",
    "model_used": "gemini-1.5-flash",
    "test_output": "pong"
  }
}
```

---

## 🧰 Logging

Logs are automatically written to stdout in the following format:

```
2025-10-09 12:45:10 | INFO | pdf_extractor | Received extract request for /path/to/file.pdf
```

---

## 🧪 Local Testing with cURL

```bash
curl -X POST "http://localhost:8000/extract" \
  -H "Content-Type: application/json" \
  -d @request.json
```

---
