import os
import glob
import json
import logging
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("save_gemini_responses")

# Configuration
ROOT_DIR = r"c:\\Users\\Ram\\Desktop\\workspace\\document-extraction-pipeline"
INPUT_FOLDERS = ["250625-set-b-clean", "250625-set-b-defect"]
OUT_DIR_CLEAN = os.path.join(ROOT_DIR, "gemini_clean")
OUT_DIR_DEFECT = os.path.join(ROOT_DIR, "gemini_defect")
EXTRACT_URL = "http://localhost:8001/extract"

# Minimal schema (matches API response shape)
SCHEMA = {
    "format": "",
    "bill_no": "",
    "provider_name": "",
    "patient_name": "",
    "ic/passport_no": "",
    "visit_type": "",
    "admission_date_time": "",
    "discharge_date_time": "",
    "gl_reference_no": "",
    "room_charges": {"ACCOMMODATION": []},
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
    "total_hospital_charges": "",
    "total_consultant_fees": "",
    "grand_total": ""
}

def post_extract(pdf_path: str) -> dict:
    payload = {"pdf_path": pdf_path, "schema": SCHEMA}
    headers = {"Content-Type": "application/json"}
    resp = requests.post(EXTRACT_URL, headers=headers, data=json.dumps(payload), timeout=300)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, dict) and data.get("status") == "success":
        return data.get("data", {})
    return data if isinstance(data, dict) else {}

def save_response(filepath_no_ext: str, resp_obj: dict, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{filepath_no_ext}_response.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(resp_obj, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved: {out_path}")

def process_folder(folder_name: str) -> int:
    input_dir = os.path.join(ROOT_DIR, folder_name)
    pdf_files = sorted(glob.glob(os.path.join(input_dir, "*.pdf")))
    if not pdf_files:
        logger.warning(f"No PDFs found in {input_dir}")
        return 0

    out_dir = OUT_DIR_CLEAN if "clean" in folder_name.lower() else OUT_DIR_DEFECT
    saved = 0
    for pdf in pdf_files:
        base = os.path.splitext(os.path.basename(pdf))[0]
        try:
            logger.info(f"[{folder_name}] Processing {base}.pdf")
            pred = post_extract(pdf)
            save_response(base, pred, out_dir)
            saved += 1
        except Exception as e:
            logger.error(f"Failed processing {base}.pdf: {e}")
    return saved

def main():
    total_saved = 0
    for folder in INPUT_FOLDERS:
        total_saved += process_folder(folder)
    print(json.dumps({"saved_files": total_saved, "clean_dir": OUT_DIR_CLEAN, "defect_dir": OUT_DIR_DEFECT}, ensure_ascii=False))

if __name__ == "__main__":
    main()
