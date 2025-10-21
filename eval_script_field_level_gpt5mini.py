import os
import glob
import json
import csv
import math
import logging
from typing import Any, Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("eval_gpt5mini")

# Configuration
ROOT_DIR = r"c:\\Users\\Ram\\Desktop\\workspace\\document-extraction-pipeline"
GROUND_TRUTH_DIR = os.path.join(ROOT_DIR, "ground_truth")
PRED_DIR = os.path.join(ROOT_DIR, "gpt-5-mini-responses")  # folder with GPT-5-mini response JSONs
CSV_OUT = os.path.join(PRED_DIR, "eval_results_gpt5mini.csv")

# Whitelist of hospital subcategories (to align keys)
HOSPITAL_SUBCATS = [
    "DIAGNOSTIC SERVICES","DRUGS FORMULARY","EMERGENCY MEDICAL SERVICE","EQUIPMENT USAGE",
    "GENERAL SUPPLIES","HOSPITAL SUPPORT FEES","LABORATORY","MEDICAL GASES","MEDICAL RECORD SERVICES",
    "MEDICAL SUPPLIES","NURSING SERVICES","OPERATING ROOM FEE","OPERATING THEATER FEES","OT SERVICES",
    "OT SUPPLIES & CONSUMABLES","OT-SUPPORT","PACKAGE","PPE SUPPLIES","PROCEDURES","RADIOGRAPHY SUPPLIES",
    "STERILE ITEMS AND SETS","SURGICAL SUPPLIES"
]
CONSULT_SUBCATS = ["CONSULTATION FEES","PROCEDURE FEES","REPORTING FEES"]

def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

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
    # If already response-shaped, ensure keys and return
    if "room_charges" in gt and "hospital_medical_services" in gt:
        gt_copy = gt.copy()
        if "room_charges" not in gt_copy or not isinstance(gt_copy["room_charges"], dict):
            gt_copy["room_charges"] = {"ACCOMMODATION": []}
        else:
            gt_copy["room_charges"].setdefault("ACCOMMODATION", [])
        if "hospital_medical_services" not in gt_copy or not isinstance(gt_copy["hospital_medical_services"], dict):
            gt_copy["hospital_medical_services"] = {}
        if isinstance(gt_copy["hospital_medical_services"], dict):
            for sub in HOSPITAL_SUBCATS:
                gt_copy["hospital_medical_services"].setdefault(sub, [])
        if "consultation_fees" not in gt_copy or not isinstance(gt_copy["consultation_fees"], dict):
            gt_copy["consultation_fees"] = {}
        if isinstance(gt_copy["consultation_fees"], dict):
            for sub in CONSULT_SUBCATS:
                gt_copy["consultation_fees"].setdefault(sub, [])
        for field in ["total_room_charges", "total_hospital_medical_services", "total_hospital_charges", "total_consultant_fees", "grand_total"]:
            gt_copy.setdefault(field, "")
        return gt_copy

    # Legacy nested (preserve items if present)
    doc = gt.get("document_details", {}) or {}
    pat = gt.get("patient_information", {}) or {}
    clm = gt.get("claim_details", {}) or {}
    bill = gt.get("billing_details", {}) or {}
    fin = gt.get("financial_information", {}) or {}

    room_src = _get_ci(bill, "ROOM CHARGES") or {}
    room_out: Dict[str, List[Dict[str, Any]]] = {"ACCOMMODATION": []}
    if isinstance(room_src, dict):
        acc = _get_ci(room_src, "ACCOMMODATION")
        if acc:
            room_out["ACCOMMODATION"] = _ensure_line_items(acc)

    hms_src = _get_ci(bill, "HOSPITAL MEDICAL SERVICES") or {}
    hms_out: Dict[str, List[Dict[str, Any]]] = {}
    if isinstance(hms_src, dict):
        for sub in HOSPITAL_SUBCATS:
            items = _get_ci(hms_src, sub)
            hms_out[sub] = _ensure_line_items(items) if items else []
    else:
        for sub in HOSPITAL_SUBCATS:
            hms_out[sub] = []

    consult_src = _get_ci(bill, "CONSULTANT(S) FEES") or {}
    consult_out: Dict[str, List[Dict[str, Any]]] = {k: [] for k in CONSULT_SUBCATS}
    if isinstance(consult_src, dict):
        for _consultant, buckets in consult_src.items():
            if not isinstance(buckets, dict):
                continue
            for sub in CONSULT_SUBCATS:
                items = _get_ci(buckets, sub)
                if items:
                    consult_out[sub].extend(_ensure_line_items(items))

    return {
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
        "consultation_fees": consult_out,
        "total_room_charges": str(fin.get("total_room_charges", "")),
        "total_hospital_medical_services": str(fin.get("total_hospital_medical_services", "")),
        "total_hospital_charges": str(fin.get("total_hospital_charges", "")),
        "total_consultant_fees": str(fin.get("total_consultant_fees", "")),
        "grand_total": str(fin.get("grand_total", "")),
    }

def strip_pred_suffixes(name: str) -> str:
    """
    Convert a prediction filename into ground-truth basename by removing common suffixes.
    Examples:
      MH24114659_Redacted_clean_response.json -> MH24114659_Redacted
      MH24114659_Redacted_response.json -> MH24114659_Redacted
    """
    stem = os.path.splitext(name)[0]
    for suf in ["_clean_response", "_defect_response", "_response", "_clean", "_defect"]:
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
    return stem

def evaluate_all() -> Tuple[str, float, float, int]:
    os.makedirs(PRED_DIR, exist_ok=True)
    pred_files = sorted(glob.glob(os.path.join(PRED_DIR, "*.json")))
    if not pred_files:
        logger.warning(f"No prediction JSONs found in {PRED_DIR}")

    rows: List[Dict[str, Any]] = []
    accs_schema: List[float] = []
    accs_values: List[float] = []

    for pred_path in pred_files:
        pred_name = os.path.basename(pred_path)
        base = strip_pred_suffixes(pred_name)
        gt_path = os.path.join(GROUND_TRUTH_DIR, f"{base}.json")
        if not os.path.exists(gt_path):
            logger.warning(f"Ground truth missing for {pred_name} -> expected {gt_path}")
            continue

        try:
            gt_raw = read_json(gt_path)
            pred = read_json(pred_path)
        except Exception as e:
            logger.error(f"Failed reading files for {pred_name}: {e}")
            continue

        gt = normalize_ground_truth_to_response(gt_raw)

        correct, total, mismatches, missing = compare_json(gt, pred)
        schema_total = total
        missing_count = len(missing)
        present_total = max(schema_total - missing_count, 0)
        schema_acc = (present_total / schema_total) if schema_total else 0.0
        values_acc = (correct / present_total) if present_total else 0.0

        row = {
            "file": base + ".pdf",
            "schema_accuracy": round(schema_acc, 4),
            "values_accuracy": round(values_acc, 4),
            "missing_count": missing_count,
            "mismatched_count": len(mismatches),
            "total_expected_fields": schema_total,
            "total_present_fields": present_total,
        }
        rows.append(row)
        accs_schema.append(row["schema_accuracy"])
        accs_values.append(row["values_accuracy"])

    # Write CSV
    os.makedirs(os.path.dirname(CSV_OUT), exist_ok=True)
    with open(CSV_OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file",
                "schema_accuracy",
                "values_accuracy",
                "missing_count",
                "mismatched_count",
                "total_expected_fields",
                "total_present_fields",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    logger.info(f"CSV written: {CSV_OUT}")

    avg_schema = round(sum(accs_schema) / len(accs_schema), 4) if accs_schema else 0.0
    avg_values = round(sum(accs_values) / len(accs_values), 4) if accs_values else 0.0
    return CSV_OUT, avg_schema, avg_values, len(rows)

def main():
    csv_path, avg_schema_acc, avg_values_acc, count = evaluate_all()
    print(json.dumps({
        "csv": csv_path,
        "files_evaluated": count,
        "average_schema_accuracy": avg_schema_acc,
        "average_values_accuracy": avg_values_acc
    }, ensure_ascii=False))

if __name__ == "__main__":
    main()
