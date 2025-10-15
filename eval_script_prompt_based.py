import os
import glob
import json
import csv
import requests
import logging
from typing import Dict, Any, Tuple, List

import openai

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("eval_script")

# Configuration
ROOT_DIR = r"c:\\Users\\Ram\\Desktop\\workspace\\document-extraction-pipeline"
INPUT_FOLDERS = ["250625-set-b-clean", "250625-set-b-defect"]
GROUND_TRUTH_DIR = os.path.join(ROOT_DIR, "ground_truth")  # fallback location for ground truth JSON
EXTRACT_URL = "http://localhost:8000/extract"  # API endpoint to call

OPENAI_API = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_EVAL_MODEL", "gpt-4o")
if not OPENAI_API:
    raise SystemExit("OPENAI_API_KEY is not set in environment.")
openai.api_key = OPENAI_API

# Target schema to send to the API
SCHEMA: Dict[str, Any] = {
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

def post_extract(pdf_path: str) -> Dict[str, Any]:
    payload = {"pdf_path": pdf_path, "schema": SCHEMA}
    headers = {"Content-Type": "application/json"}
    resp = requests.post(EXTRACT_URL, headers=headers, data=json.dumps(payload), timeout=300)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, dict) and data.get("status") == "success":
        return data.get("data", {})
    return data

def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def resolve_ground_truth(basename: str, input_dir: str) -> str:
    # 1) Prefer same-folder ground truth
    same_folder = os.path.join(input_dir, f"{basename}.json")
    if os.path.exists(same_folder):
        return same_folder
    # 2) Fallback: ROOT/ground_truth/<basename>.json
    gt_global = os.path.join(GROUND_TRUTH_DIR, f"{basename}.json")
    if os.path.exists(gt_global):
        return gt_global
    return ""

def gpt_compare(ground_truth: Dict[str, Any], prediction: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    system_prompt = (
        "You are a strict evaluator. Compare two hospital bill JSON objects: ground_truth and prediction. "
        "Compute an overall accuracy between 0 and 1.0 based on field-level similarity:\n"
        "- For scalar fields (strings/numbers), exact match after trimming whitespace (case-insensitive for strings). "
        "Numbers should match within ±0.01 tolerance.\n"
        "- For arrays of line items, ignore order. Align by best match on description_of_service, service_code, date, and amounts. "
        "Score is the average fraction of correctly matched fields across aligned items. Unmatched items are incorrect.\n"
        "- For nested objects, evaluate recursively and average over leaf fields.\n"
        "- If ground truth is empty string and prediction empty, consider correct. If ground truth empty and prediction non-empty, do not penalize.\n"
        "Return only valid JSON with:\n"
        " don't go for exact matching of column names, use intellisense & automatch columns & nested objects\n"
        "{ \"accuracy\": 0.0, \"summary\": \"one-line\", \"notes\": \"optional\" }"
    )
    user_content = {"ground_truth": ground_truth, "prediction": prediction}
    try:
        response = openai.ChatCompletion.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_content, ensure_ascii=False)}
            ],
            temperature=0,
            response_format={"type": "json_object"},
            max_tokens=800
        )
        content = response.choices[0].message.content
        result = json.loads(content)
        acc = float(result.get("accuracy", 0.0))
        return max(0.0, min(1.0, acc)), result
    except Exception as e:
        logger.error(f"OpenAI evaluation failed: {e}")
        return 0.0, {"accuracy": 0.0, "summary": "evaluation_failed", "notes": str(e)}

def find_missing_fields(gt: Any, pred: Any, path: str = "") -> List[str]:
    missing: List[str] = []
    if isinstance(gt, dict):
        if not isinstance(pred, dict):
            missing.extend([f"{path}.{k}".lstrip(".") for k in gt.keys()])
            return missing
        for k, v in gt.items():
            new_path = f"{path}.{k}" if path else k
            if k not in pred:
                missing.append(new_path)
            else:
                missing.extend(find_missing_fields(v, pred.get(k), new_path))
    elif isinstance(gt, list):
        if not isinstance(pred, list):
            missing.append(f"{path}[]")
            return missing
        if len(gt) > 0 and len(pred) == 0:
            missing.append(f"{path}[items_missing]")
            return missing
        if len(gt) > 0 and isinstance(gt[0], dict) and len(pred) > 0:
            template_keys = set(gt[0].keys())
            missing_keys = set()
            for item in pred:
                if isinstance(item, dict):
                    missing_keys |= (template_keys - set(item.keys()))
                else:
                    missing.append(f"{path}[*]")
            for mk in sorted(missing_keys):
                missing.append(f"{path}[].{mk}")
    return missing

def write_csv(rows: List[Dict[str, Any]], csv_path: str) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    fieldnames = ["file", "accuracy", "summary", "missing_count", "missing_fields"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

def evaluate_folder(folder_name: str) -> Tuple[str, float, int]:
    input_dir = os.path.join(ROOT_DIR, folder_name)
    pdf_files = sorted(glob.glob(os.path.join(input_dir, "*.pdf")))
    if not pdf_files:
        logger.warning(f"No PDFs found in {input_dir}")
    rows: List[Dict[str, Any]] = []
    accs: List[float] = []

    for pdf in pdf_files:
        fname = os.path.splitext(os.path.basename(pdf))[0]
        gt_path = resolve_ground_truth(fname, input_dir)
        if not gt_path:
            logger.warning(f"Ground truth missing for {fname}")
            continue

        logger.info(f"[{folder_name}] {fname}.pdf")
        try:
            pred = post_extract(pdf)
        except Exception as e:
            logger.error(f"Extraction failed for {fname}: {e}")
            continue

        try:
            gt = read_json(gt_path)
        except Exception as e:
            logger.error(f"Failed to read ground truth for {fname}: {e}")
            continue

        acc, details = gpt_compare(gt, pred)
        accs.append(acc)

        missing_fields = find_missing_fields(gt, pred)
        row = {
            "file": f"{fname}.pdf",
            "accuracy": round(acc, 4),
            "summary": details.get("summary", ""),
            "missing_count": len(missing_fields),
            "missing_fields": ";".join(missing_fields)
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))

    # Write CSV per folder
    csv_out = os.path.join(input_dir, f"eval_results_{folder_name}.csv")
    write_csv(rows, csv_out)
    logger.info(f"CSV written: {csv_out}")

    avg = round(sum(accs) / len(accs), 4) if accs else 0.0
    return csv_out, avg, len(accs)

def main():
    summary_rows = []
    for folder in INPUT_FOLDERS:
        csv_path, avg_acc, count = evaluate_folder(folder)
        summary_rows.append({"folder": folder, "files_evaluated": count, "average_accuracy": avg_acc, "csv": csv_path})

    # Write summary CSV at root
    summary_csv = os.path.join(ROOT_DIR, "eval_summary.csv")
    os.makedirs(os.path.dirname(summary_csv), exist_ok=True)
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["folder", "files_evaluated", "average_accuracy", "csv"])
        writer.writeheader()
        for r in summary_rows:
            writer.writerow(r)
    print(json.dumps({"summary_csv": summary_csv, "summary": summary_rows}, ensure_ascii=False))

if __name__ == "__main__":
    main()