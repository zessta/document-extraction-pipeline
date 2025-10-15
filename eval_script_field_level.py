import os
import glob
import json
import csv
import math
import logging
from typing import Any, Dict, List, Tuple
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("eval_converted")

# Configuration
ROOT_DIR = r"c:\\Users\\Ram\\Desktop\\workspace\\document-extraction-pipeline"
INPUT_FOLDERS = ["250625-set-b-clean", "250625-set-b-defect"]
BASE_GROUND_TRUTH_DIR = os.path.join(ROOT_DIR, "ground_truth")  # CHANGED: use ground_truth
EXTRACT_URL = "http://localhost:8000/extract"

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
    # Use 'data' if wrapped with {"status":"success","data":{...}}
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
    # collapse whitespace
    return " ".join(text.split())

def scalar_equal(gt: Any, pred: Any) -> bool:
    # If base is empty string, do not penalize any predicted value (consider correct).
    if isinstance(gt, str) and normalize_str(gt) == "":
        return True
    # Numbers (with commas/parentheses support)
    if (is_number(gt)) and (is_number(pred)):
        return num_equal(gt, pred)
    # Strings
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
    # similarity = fraction of gt_item fields that are equal in prediction
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
        # pick best match
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
            # nothing to compare; count all fields as missing
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
    # returns (correct, total, mismatches, missing_fields)
    if isinstance(gt, dict):
        correct = 0
        total = 0
        mismatches: List[str] = []
        missing: List[str] = []
        if not isinstance(pred, dict):
            # everything under gt considered missing
            for k in gt.keys():
                missing.append(f"{path}.{k}".lstrip("."))
            return 0, len(gt), mismatches, missing
        for k, v in gt.items():
            p = f"{path}.{k}" if path else k
            if k not in pred:
                # key missing entirely in prediction
                # count leafs under v
                def count_leafs(x: Any) -> int:
                    if isinstance(x, dict):
                        return sum(count_leafs(vv) for vv in x.values())
                    if isinstance(x, list):
                        if not x:
                            return 1
                        # approximate: count sum of fields in dict items or 1 per scalar
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
        # Only compare using gt as the template; extra pred items are ignored
        if not isinstance(pred, list):
            # all gt entries considered missing
            missing = [f"{path}[]"]
            # estimate total: if list of dicts use sum of fields else number of items
            if len(gt) > 0 and isinstance(gt[0], dict):
                total = sum(len(item.keys()) for item in gt)
            else:
                total = len(gt)
            return 0, total, [], missing

        if len(gt) == 0:
            # nothing to compare
            return 0, 0, [], []

        if isinstance(gt[0], dict):
            return compare_lists_of_dicts(gt, pred, path)
        else:
            # list of scalars - simple multiset compare
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

    # Scalars
    total = 1
    if scalar_equal(gt, pred):
        return 1, 1, [], []
    else:
        return 0, 1, [path], []

def _get_ci(d: Dict[str, Any], key: str) -> Any:
    """Case-insensitive get for dict keys."""
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
    """
    Map GT schema:
      document_details/patient_information/claim_details/billing_details/financial_information
    to response schema used by the API:
      format, bill_no, provider_name, patient_name, ic/passport_no, ..., room_charges, hospital_medical_services, consultation_fees
    """
    doc = gt.get("document_details", {}) or {}
    pat = gt.get("patient_information", {}) or {}
    clm = gt.get("claim_details", {}) or {}
    bill = gt.get("billing_details", {}) or {}
    # Financial totals intentionally ignored here to match sample response shape

    # Room charges (only ACCOMMODATION subcat)
    room_src = _get_ci(bill, "ROOM CHARGES") or {}
    room_out: Dict[str, List[Dict[str, Any]]] = {"ACCOMMODATION": []}
    if isinstance(room_src, dict):
        acc = _get_ci(room_src, "ACCOMMODATION")
        room_out["ACCOMMODATION"] = _ensure_line_items(acc)

    # Hospital medical services - include all whitelisted subcategories (empty if missing)
    hms_src = _get_ci(bill, "HOSPITAL MEDICAL SERVICES") or {}
    hms_out: Dict[str, List[Dict[str, Any]]] = {}
    if isinstance(hms_src, dict):
        for sub in HOSPITAL_SUBCATS:
            hms_out[sub] = _ensure_line_items(_get_ci(hms_src, sub))
    else:
        for sub in HOSPITAL_SUBCATS:
            hms_out[sub] = []

    # Consultation fees - flatten across consultants by subcategory
    consult_src = _get_ci(bill, "CONSULTANT(S) FEES") or {}
    consult_out: Dict[str, List[Dict[str, Any]]] = {k: [] for k in CONSULT_SUBCATS}
    if isinstance(consult_src, dict):
        for _consultant, buckets in consult_src.items():
            if not isinstance(buckets, dict):
                continue
            for sub in CONSULT_SUBCATS:
                consult_out[sub].extend(_ensure_line_items(_get_ci(buckets, sub)))

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
        "room_charges": room_out,
        "hospital_medical_services": hms_out,
        "consultation_fees": consult_out
    }
    return normalized

def evaluate_one(pdf_path: str, base_json_path: str) -> Dict[str, Any]:
    try:
        prediction = post_extract(pdf_path)
    except Exception as e:
        return {
            "file": os.path.basename(pdf_path),
            "accuracy": 0.0,
            "summary": f"extract_failed: {e}",
            "missing_count": 0,
            "missing_fields": "",
            "mismatched_count": 0,
            "mismatched_fields": ""
        }

    try:
        base_obj = read_json(base_json_path)
    except Exception as e:
        return {
            "file": os.path.basename(pdf_path),
            "accuracy": 0.0,
            "summary": f"base_read_failed: {e}",
            "missing_count": 0,
            "missing_fields": "",
            "mismatched_count": 0,
            "mismatched_fields": ""
        }

    gt_raw = get_base_data(base_obj)
    gt = normalize_ground_truth_to_response(gt_raw)  # CHANGED: normalize GT shape
    pred = prediction if isinstance(prediction, dict) else {}

    correct, total, mismatches, missing = compare_json(gt, pred)

    # New: split schema vs values accuracy
    schema_total = total
    missing_count = len(missing)
    present_total = max(schema_total - missing_count, 0)

    schema_acc = (present_total / schema_total) if schema_total else 0.0
    values_acc = (correct / present_total) if present_total else 0.0
    overall_acc = (correct / schema_total) if schema_total else 0.0

    return {
        "file": os.path.basename(pdf_path),
        "accuracy": round(overall_acc, 4),  # overall = values_correct over all leaf fields
        "schema_accuracy": round(schema_acc, 4),
        "values_accuracy": round(values_acc, 4),
        "summary": "ok",
        "missing_count": missing_count,
        "missing_fields": ";".join(missing),
        "mismatched_count": len(mismatches),
        "mismatched_fields": ";".join(mismatches)
    }

def write_csv(rows: List[Dict[str, Any]], csv_path: str) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    # New columns added: schema_accuracy, values_accuracy
    fieldnames = [
        "file",
        "accuracy",
        "schema_accuracy",
        "values_accuracy",
        "summary",
        "missing_count",
        "missing_fields",
        "mismatched_count",
        "mismatched_fields",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

def evaluate_folder(folder_name: str) -> Tuple[str, float, float, float, int]:
    input_dir = os.path.join(ROOT_DIR, folder_name)
    pdf_files = sorted(glob.glob(os.path.join(input_dir, "*.pdf")))
    if not pdf_files:
        logger.warning(f"No PDFs found in {input_dir}")

    rows: List[Dict[str, Any]] = []
    accs_overall: List[float] = []
    accs_schema: List[float] = []
    accs_values: List[float] = []

    for pdf in pdf_files:
        fname = os.path.splitext(os.path.basename(pdf))[0]
        base_path = resolve_base_truth(fname)
        if not base_path:
            logger.warning(f"Base ground truth missing for {fname}")
            continue

        logger.info(f"[{folder_name}] {fname}.pdf")
        row = evaluate_one(pdf, base_path)
        rows.append(row)
        accs_overall.append(row["accuracy"])
        accs_schema.append(row["schema_accuracy"])
        accs_values.append(row["values_accuracy"])
        # print(json.dumps(row, ensure_ascii=False))

    csv_out = os.path.join(input_dir, f"eval_results_converted_{folder_name}.csv")
    write_csv(rows, csv_out)
    logger.info(f"CSV written: {csv_out}")

    avg_overall = round(sum(accs_overall) / len(accs_overall), 4) if accs_overall else 0.0
    avg_schema = round(sum(accs_schema) / len(accs_schema), 4) if accs_schema else 0.0
    avg_values = round(sum(accs_values) / len(accs_values), 4) if accs_values else 0.0
    return csv_out, avg_overall, avg_schema, avg_values, len(accs_overall)

def main():
    summary_rows = []
    for folder in INPUT_FOLDERS:
        csv_path, avg_acc, avg_schema_acc, avg_values_acc, count = evaluate_folder(folder)
        summary_rows.append({
            "folder": folder,
            "files_evaluated": count,
            "average_accuracy": avg_acc,
            "average_schema_accuracy": avg_schema_acc,
            "average_values_accuracy": avg_values_acc,
            "csv": csv_path
        })

    summary_csv = os.path.join(ROOT_DIR, "eval_summary_converted.csv")
    os.makedirs(os.path.dirname(summary_csv), exist_ok=True)
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "folder",
                "files_evaluated",
                "average_accuracy",
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
