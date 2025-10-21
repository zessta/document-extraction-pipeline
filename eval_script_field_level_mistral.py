import os
import glob
import json
import csv
import math
import logging
from typing import Any, Dict, List, Tuple
import requests
import time  # NEW: measure API response time

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("eval_mistral")

# Configuration
# ROOT_DIR = r"c:\\Users\\Ram\\Desktop\\workspace\\document-extraction-pipeline"
INPUT_FOLDERS = ["250625-set-b-clean", "250625-set-b-defect"]
BASE_GROUND_TRUTH_DIR = os.path.join(os.getcwd(), "ground_truth")
OUTPUT_DIR = os.path.join(os.getcwd(), "mistral_output")  # ensure separate folder
EXTRACT_URL = "http://localhost:8001/extract-mistral"  # use mistral endpoint

# Schema to send to the API (response shape)
SCHEMA: Dict[str, Any] = {
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
        "ACCOMMODATION": [
            { "service_code": "", "description": "", "date": "", "quantity": "", "gross_amount": "", "discount": "", "allocated_amount": "" }
        ]
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
    "total_hospital_charges": "",
    "total_consultant_fees": "",
    "grand_total": ""
}

# Whitelist of hospital subcategories (to align GT with response schema)
HOSPITAL_SUBCATS = [
    "DIAGNOSTIC SERVICES","DRUGS FORMULARY","EMERGENCY MEDICAL SERVICE","EQUIPMENT USAGE",
    "GENERAL SUPPLIES","HOSPITAL SUPPORT FEES","LABORATORY","MEDICAL GASES","MEDICAL RECORD SERVICES",
    "MEDICAL SUPPLIES","NURSING SERVICES","OPERATING ROOM FEE","OPERATING THEATER FEES","OT SERVICES",
    "OT SUPPLIES & CONSUMABLES","OT-SUPPORT","PACKAGE","PPE SUPPLIES","PROCEDURES","RADIOGRAPHY SUPPLIES",
    "STERILE ITEMS AND SETS","SURGICAL SUPPLIES"
]
CONSULT_SUBCATS = ["CONSULTATION FEES","PROCEDURE FEES","REPORTING FEES"]

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

def get_base_data(obj: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(obj, dict) and "data" in obj and isinstance(obj["data"], dict):
        return obj["data"]
    return obj

def resolve_base_truth(basename: str) -> str:
    path = os.path.join(BASE_GROUND_TRUTH_DIR, f"{basename}.json")
    return path if os.path.exists(path) else ""

def _try_parse_number(x: Any) -> Tuple[bool, float]:
    if x is None:
        return False, 0.0
    s = str(x).strip()
    neg = s.startswith("(") and s.endswith(")")
    s = s.strip("()").replace(",", "")
    try:
        v = float(s)
        return True, -v if neg else v
    except Exception:
        return False, 0.0

def is_number(x: Any) -> bool:
    ok, _ = _try_parse_number(x)
    return ok

def num_equal(a: Any, b: Any, tol: float = 0.01) -> bool:
    ok_a, va = _try_parse_number(a)
    ok_b, vb = _try_parse_number(b)
    if not (ok_a and ok_b):
        return False
    return math.isclose(va, vb, abs_tol=tol)

def normalize_str(s: Any) -> str:
    if s is None:
        return ""
    text = str(s).strip().lower()
    return " ".join(text.split())

def scalar_equal(gt: Any, pred: Any) -> bool:
    if isinstance(gt, str) and normalize_str(gt) == "":
        return True
    if (is_number(gt)) and (is_number(pred)):
        return num_equal(gt, pred)
    return normalize_str(gt) == normalize_str(pred)

def compare_items(gt_item: Dict[str, Any], pr_item: Dict[str, Any], path: str) -> Tuple[int, int, List[str], List[str]]:
    correct = 0
    total = 0
    mismatches: List[str] = []
    missing: List[str] = []
    for k, v in gt_item.items():
        total += 1
        p = path + f".{k}" if path else k
        if k not in pr_item:
            missing.append(p)
            continue
        if scalar_equal(v, pr_item[k]):
            correct += 1
        else:
            mismatches.append(p)
    return correct, total, mismatches, missing

def item_similarity(gt_item: Dict[str, Any], pr_item: Dict[str, Any]) -> float:
    if not isinstance(pr_item, dict):
        return 0.0
    if not gt_item:
        return 0.0
    matches = 0
    total = 0
    for k, v in gt_item.items():
        total += 1
        if k in pr_item and scalar_equal(v, pr_item[k]):
            matches += 1
    return matches / total if total else 0.0

def compare_lists_of_dicts(gt_list: List[Dict[str, Any]], pr_list: List[Dict[str, Any]], path: str) -> Tuple[int, int, List[str], List[str]]:
    correct = 0
    total = 0
    mismatches: List[str] = []
    missing: List[str] = []
    used = set()

    for i, gt_item in enumerate(gt_list):
        best_j = -1
        best_score = -1.0
        for j, pr_item in enumerate(pr_list):
            if j in used:
                continue
            score = item_similarity(gt_item, pr_item)
            if score > best_score:
                best_score = score
                best_j = j
        if best_j == -1:
            for k in gt_item.keys():
                total += 1
                missing.append(f"{path}[{i}].{k}")
            continue
        used.add(best_j)
        c, t, mm, ms = compare_items(gt_item, pr_list[best_j], f"{path}[{i}]")
        correct += c
        total += t
        mismatches.extend(mm)
        missing.extend(ms)

    return correct, total, mismatches, missing

def compare_json(gt: Any, pred: Any, path: str = "") -> Tuple[int, int, List[str], List[str]]:
    if isinstance(gt, dict):
        correct = 0
        total = 0
        mismatches: List[str] = []
        missing: List[str] = []
        if not isinstance(pred, dict):
            for k in gt.keys():
                missing.append(f"{path}.{k}".lstrip("."))
            return 0, len(gt), mismatches, missing
        for k, v in gt.items():
            p = f"{path}.{k}" if path else k
            if k not in pred:
                def count_leafs(x: Any) -> int:
                    if isinstance(x, dict):
                        return sum(count_leafs(vv) for vv in x.values())
                    if isinstance(x, list):
                        if not x:
                            return 1
                        if isinstance(x[0], dict):
                            return sum(len(item.keys()) for item in x)
                        return len(x)
                    return 1
                missing.append(p)
                total += count_leafs(v)
                continue
            c, t, mm, ms = compare_json(v, pred[k], p)
            correct += c
            total += t
            mismatches.extend(mm)
            missing.extend(ms)
        return correct, total, mismatches, missing

    if isinstance(gt, list):
        if not isinstance(pred, list):
            missing = [f"{path}[]"]
            if len(gt) > 0 and isinstance(gt[0], dict):
                total = sum(len(item.keys()) for item in gt)
            else:
                total = len(gt)
            return 0, total, [], missing

        if len(gt) == 0:
            return 0, 0, [], []

        if isinstance(gt[0], dict):
            return compare_lists_of_dicts(gt, pred, path)
        else:
            correct = 0
            total = len(gt)
            mismatches: List[str] = []
            missing: List[str] = []
            pr_counts: Dict[str, int] = {}
            for v in pred:
                key = json.dumps(v, sort_keys=True, ensure_ascii=False)
                pr_counts[key] = pr_counts.get(key, 0) + 1
            for i, v in enumerate(gt):
                key = json.dumps(v, sort_keys=True, ensure_ascii=False)
                if pr_counts.get(key, 0) > 0:
                    correct += 1
                    pr_counts[key] -= 1
                else:
                    mismatches.append(f"{path}[{i}]")
            return correct, total, mismatches, missing

    total = 1
    if scalar_equal(gt, pred):
        return 1, 1, [], []
    else:
        return 0, 1, [path], []

def _get_ci(d: Dict[str, Any], key: str) -> Any:
    if not isinstance(d, dict):
        return None
    for k, v in d.items():
        if str(k).strip().lower() == key.strip().lower():
            return v
    return None

def _ensure_line_items(items: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not isinstance(items, list):
        return out
    for li in items:
        if not isinstance(li, dict):
            continue
        out.append({
            "service_code": str(li.get("service_code", "")),
            "description": str(li.get("description", "")),
            "date": str(li.get("date", "")),
            "quantity": str(li.get("quantity", "")),
            "gross_amount": str(li.get("gross_amount", "")),
            "discount": str(li.get("discount", "")),
            "allocated_amount": str(li.get("allocated_amount", "")),
        })
    return out

def normalize_ground_truth_to_response(gt: Dict[str, Any]) -> Dict[str, Any]:
    # FIRST: Check if ground truth is already in response format
    if "room_charges" in gt and "hospital_medical_services" in gt:
        logger.info("Ground truth is already in response format, using as-is")
        gt_copy = gt.copy()
        
        # Ensure room_charges structure
        if "room_charges" not in gt_copy:
            gt_copy["room_charges"] = {"ACCOMMODATION": []}
        elif not isinstance(gt_copy["room_charges"], dict):
            gt_copy["room_charges"] = {"ACCOMMODATION": []}
        elif "ACCOMMODATION" not in gt_copy["room_charges"]:
            gt_copy["room_charges"]["ACCOMMODATION"] = []
        
        # Ensure hospital_medical_services structure
        if "hospital_medical_services" not in gt_copy:
            gt_copy["hospital_medical_services"] = {}
        if isinstance(gt_copy["hospital_medical_services"], dict):
            for sub in HOSPITAL_SUBCATS:
                if sub not in gt_copy["hospital_medical_services"]:
                    gt_copy["hospital_medical_services"][sub] = []
        
        # Ensure consultation_fees structure  
        if "consultation_fees" not in gt_copy:
            gt_copy["consultation_fees"] = {}
        if isinstance(gt_copy["consultation_fees"], dict):
            for sub in CONSULT_SUBCATS:
                if sub not in gt_copy["consultation_fees"]:
                    gt_copy["consultation_fees"][sub] = []
        
        # Ensure totals exist
        for field in ["total_room_charges", "total_hospital_medical_services", "total_hospital_charges", "total_consultant_fees", "grand_total"]:
            gt_copy.setdefault(field, "")
        
        return gt_copy

    # Legacy nested format normalization...
    doc = gt.get("document_details", {}) or {}
    pat = gt.get("patient_information", {}) or {}
    clm = gt.get("claim_details", {}) or {}
    bill = gt.get("billing_details", {}) or {}
    fin = gt.get("financial_information", {}) or {}

    # ...existing code for legacy format...

    normalized = {
        "format": str(doc.get("format", "")),
        "bill_no": str(doc.get("bill_no", "")),
        "provider_name": str(doc.get("provider_name", "")),
        "patient_name": str(pat.get("full_name", "")),
        "ic/passport_no": str(pat.get("identification_number", "")),
        "visit_type": str(clm.get("visit_type", "")),
        "admission_date_time": str(clm.get("admission_date_time", "")),
        "discharge_date_time": str(clm.get("discharge_date_time", "")),
        "gl_reference_no": str(clm.get("gl_reference_no", "")),
        "room_charges": {"ACCOMMODATION": []},
        "hospital_medical_services": {sub: [] for sub in HOSPITAL_SUBCATS},
        "consultation_fees": {sub: [] for sub in CONSULT_SUBCATS},
        "total_room_charges": str(fin.get("total_room_charges", "")),
        "total_hospital_medical_services": str(fin.get("total_hospital_medical_services", "")),
        "total_hospital_charges": str(fin.get("total_hospital_charges", "")),
        "total_consultant_fees": str(fin.get("total_consultant_fees", "")),
        "grand_total": str(fin.get("grand_total", "")),
    }
    return normalized

def save_individual_files(filename: str, gt: Dict[str, Any], pred: Dict[str, Any], folder_name: str) -> None:
    """Save ground truth and response as separate JSON files in mistral_output folder with folder suffix."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    folder_suffix = "clean" if "clean" in folder_name else "defect"
    gt_path = os.path.join(OUTPUT_DIR, f"{filename}_{folder_suffix}_groundtruth.json")
    response_path = os.path.join(OUTPUT_DIR, f"{filename}_{folder_suffix}_response.json")
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(gt, f, indent=2, ensure_ascii=False)
    with open(response_path, "w", encoding="utf-8") as f:
        json.dump(pred, f, indent=2, ensure_ascii=False)

def evaluate_one(pdf_path: str, base_json_path: str, folder_name: str) -> Dict[str, Any]:
    start_ts = time.perf_counter()  # NEW: start timer
    try:
        prediction = post_extract(pdf_path)
        response_time_ms = round((time.perf_counter() - start_ts) * 1000.0, 2)  # NEW
    except Exception as e:
        return {
            "file": os.path.basename(pdf_path),
            "schema_accuracy": 0.0,
            "values_accuracy": 0.0,
            "missing_count": 0,
            "mismatched_count": 0,
            "total_expected_fields": 0,
            "total_present_fields": 0,
            "response_time_ms": 0.0,  # NEW
        }

    try:
        base_obj = read_json(base_json_path)
    except Exception as e:
        return {
            "file": os.path.basename(pdf_path),
            "schema_accuracy": 0.0,
            "values_accuracy": 0.0,
            "missing_count": 0,
            "mismatched_count": 0,
            "total_expected_fields": 0,
            "total_present_fields": 0,
            "response_time_ms": response_time_ms,  # NEW
        }

    gt_raw = get_base_data(base_obj)
    gt = normalize_ground_truth_to_response(gt_raw)
    pred = prediction if isinstance(prediction, dict) else {}

    # Save individual files with folder suffix
    filename = os.path.splitext(os.path.basename(pdf_path))[0]
    save_individual_files(filename, gt, pred, folder_name)

    correct, total, mismatches, missing = compare_json(gt, pred)

    schema_total = total
    missing_count = len(missing)
    present_total = max(schema_total - missing_count, 0)

    schema_acc = (present_total / schema_total) if schema_total else 0.0
    values_acc = (correct / present_total) if present_total else 0.0

    return {
        "file": os.path.basename(pdf_path),
        "schema_accuracy": round(schema_acc, 4),
        "values_accuracy": round(values_acc, 4),
        "missing_count": missing_count,
        "mismatched_count": len(mismatches),
        "total_expected_fields": schema_total,
        "total_present_fields": present_total,
        "response_time_ms": response_time_ms,  # NEW
    }

def write_csv(rows: List[Dict[str, Any]], csv_path: str) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    fieldnames = [
        "file",
        "schema_accuracy",
        "values_accuracy",
        "missing_count",
        "mismatched_count",
        "total_expected_fields",
        "total_present_fields",
        "response_time_ms",  # NEW
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

def evaluate_folder(folder_name: str) -> Tuple[str, float, float, int]:
    input_dir = os.path.join(os.getcwd(), folder_name)
    pdf_files = sorted(glob.glob(os.path.join(input_dir, "*.pdf")))
    if not pdf_files:
        logger.warning(f"No PDFs found in {input_dir}")

    rows: List[Dict[str, Any]] = []
    accs_schema: List[float] = []
    accs_values: List[float] = []

    for pdf in pdf_files:
        fname = os.path.splitext(os.path.basename(pdf))[0]
        base_path = resolve_base_truth(fname)
        if not base_path:
            logger.warning(f"Base ground truth missing for {fname}")
            continue

        logger.info(f"[{folder_name}] {fname}.pdf")
        row = evaluate_one(pdf, base_path, folder_name)
        rows.append(row)
        accs_schema.append(row["schema_accuracy"])
        accs_values.append(row["values_accuracy"])

    csv_out = os.path.join(input_dir, f"eval_results_mistral_{folder_name}.csv")  # include 'mistral' in name
    write_csv(rows, csv_out)
    logger.info(f"CSV written: {csv_out}")

    avg_schema = round(sum(accs_schema) / len(accs_schema), 4) if accs_schema else 0.0
    avg_values = round(sum(accs_values) / len(accs_values), 4) if accs_values else 0.0
    return csv_out, avg_schema, avg_values, len(rows)

def main():
    summary_rows = []
    for folder in INPUT_FOLDERS:
        csv_path, avg_schema_acc, avg_values_acc, count = evaluate_folder(folder)
        summary_rows.append({
            "folder": folder,
            "files_evaluated": count,
            "average_schema_accuracy": avg_schema_acc,
            "average_values_accuracy": avg_values_acc,
            "csv": csv_path
        })

    summary_csv = os.path.join(os.getcwd(), "eval_summary_mistral.csv")  # consistent summary name
    os.makedirs(os.path.dirname(summary_csv), exist_ok=True)
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "folder",
                "files_evaluated",
                "average_schema_accuracy",
                "average_values_accuracy",
                "csv",
            ],
        )
        writer.writeheader()
        for r in summary_rows:
            writer.writerow(r)
    print(json.dumps({"summary_csv": summary_csv, "summary": summary_rows}, ensure_ascii=False))

if __name__ == "__main__":
    main()