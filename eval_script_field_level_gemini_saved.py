import os
import glob
import json
import csv
import math
import logging
from typing import Any, Dict, List, Tuple
import re  # NEW
import time  # NEW: for timestamp fallback

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("eval_gemini_saved")

# Configuration
ROOT_DIR = r"c:\\Users\\Ram\\Desktop\\workspace\\document-extraction-pipeline"
GROUND_TRUTH_DIR = os.path.join(ROOT_DIR, "ground_truth")
PRED_DIRS = [
    os.path.join(ROOT_DIR, "gemini_clean"),
    # os.path.join(ROOT_DIR, "gemini_defect"),
]

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
    text = " ".join(text.split())
    # NEW: normalize hyphen/dash spacing to treat "a - b" == "a-b"
    text = re.sub(r"\s*[-–—]\s*", "-", text)
    return text

# NEW: case-insensitive key lookup
def _ci_lookup_key(d: Dict[str, Any], key: str) -> str | None:
    """Return the actual dict key in d matching 'key' ignoring case, or None."""
    if not isinstance(d, dict):
        return None
    key_l = str(key).lower()
    for k in d.keys():
        if str(k).lower() == key_l:
            return k
    return None

# NEW: recursively lowercase all dict keys
def _lower_keys(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k).lower(): _lower_keys(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_lower_keys(v) for v in obj]
    return obj

# NEW: strict leaf counter computed ONLY from ground truth
def _count_gt_leaves(x: Any) -> int:
    """
    Counts comparable leaf fields in GT:
    - scalar -> 1
    - list of dicts -> sum of number of keys per item
    - list of scalars -> len(list)
    - dict -> sum of children
    - empty lists/dicts -> 0
    """
    if x is None:
        return 0
    if isinstance(x, dict):
        if not x:
            return 0
        return sum(_count_gt_leaves(v) for v in x.values())
    if isinstance(x, list):
        if not x:
            return 0
        if isinstance(x[0], dict):
            return sum(len(item.keys()) for item in x)
        return len(x)
    # scalars (strings/numbers/etc.)
    return 1

# NEW: field-aware equality
def field_equal(key: str, gt: Any, pred: Any) -> bool:
    return scalar_equal(gt, pred)

# NEW: weight of a GT subtree when counting schema fields; empty values cost 0
def _leaf_weight(x: Any) -> int:
    if x is None:
        return 0
    if isinstance(x, str):
        return 0 if normalize_str(x) == "" else 1
    if isinstance(x, list):
        if not x:
            return 0
        if isinstance(x[0], dict):
            return sum(len(item.keys()) for item in x)
        return len(x)
    if isinstance(x, dict):
        if not x:
            return 0
        return sum(_leaf_weight(v) for v in x.values())
    # scalar (number/bool/etc.)
    return 1

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
        # CHANGED: use field_equal instead of scalar_equal
        if field_equal(k, v, pr_item[k]):
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
        if k in pr_item and field_equal(k, v, pr_item[k]):  # CHANGED
            matches += 1
    return matches / total if total else 0.0

# NEW: robust list matcher to reduce false mismatches
def _signature(d: Dict[str, Any]) -> str:
    if not isinstance(d, dict):
        return ""
    # Build a signature with key fields (normalized)
    parts = [
        f"code:{d.get('service_code','')}",
        f"date:{normalize_str(d.get('date',''))}",
        f"desc:{normalize_str(d.get('description',''))}",
        f"qty:{normalize_str(d.get('quantity',''))}",
        f"gross:{normalize_str(d.get('gross_amount',''))}",
        f"alloc:{normalize_str(d.get('allocated_amount',''))}",
    ]
    return "|".join(parts)

def compare_lists_of_dicts(gt_list: List[Dict[str, Any]], pr_list: List[Dict[str, Any]], path: str) -> Tuple[int, int, List[str], List[str]]:
    # Fast exits
    if not gt_list:
        return 0, 0, [], []
    if not pr_list:
        correct = 0
        total = sum(len(item.keys()) for item in gt_list)
        missing = []
        for i, gt_item in enumerate(gt_list):
            for k in gt_item.keys():
                missing.append(f"{path}[{i}].{k}")
        return 0, total, [], missing

    correct = 0
    total = 0
    mismatches: List[str] = []
    missing: List[str] = []

    # 1) Exact signature pairing
    gt_unused = list(range(len(gt_list)))
    pr_unused = list(range(len(pr_list)))
    sig_to_pr: Dict[str, List[int]] = {}
    for j in pr_unused:
        sig_to_pr.setdefault(_signature(pr_list[j]), []).append(j)

    pairs: List[Tuple[int, int]] = []
    for i in list(gt_unused):
        sig = _signature(gt_list[i])
        candidates = sig_to_pr.get(sig, [])
        if candidates:
            j = candidates.pop(0)
            pairs.append((i, j))
            gt_unused.remove(i)
            pr_unused.remove(j)

    # 2) Relaxed pairing by (service_code, date, description)
    if gt_unused and pr_unused:
        key_to_pr: Dict[Tuple[str, str, str], List[int]] = {}
        for j in pr_unused:
            key = (pr_list[j].get("service_code","")), normalize_str(pr_list[j].get("date","")),normalize_str(pr_list[j].get("description",""))
            key_to_pr.setdefault(key, []).append(j)
        for i in list(gt_unused):
            key = (gt_list[i].get("service_code","")),normalize_str(gt_list[i].get("date","")),normalize_str(gt_list[i].get("description",""))
            candidates = key_to_pr.get(key, [])
            if candidates:
                j = candidates.pop(0)
                pairs.append((i, j))
                gt_unused.remove(i)
                pr_unused.remove(j)

    # 3) Greedy best-score fallback for remaining
    used_pr = set(j for _, j in pairs)
    for i in gt_unused:
        best_j = -1
        best_score = -1.0
        for j in pr_unused:
            if j in used_pr:
                continue
            score = item_similarity(gt_list[i], pr_list[j])
            if score > best_score:
                best_score = score
                best_j = j
        if best_j != -1:
            pairs.append((i, best_j))
            used_pr.add(best_j)

    # Compare paired items
    for i, j in pairs:
        c, t, mm, ms = compare_items(gt_list[i], pr_list[j], f"{path}[{i}]")
        correct += c
        total += t
        mismatches.extend(mm)
        missing.extend(ms)

    # Any GT items left unmatched: count their fields as missing
    matched_gt = set(i for i, _ in pairs)
    for i in range(len(gt_list)):
        if i in matched_gt:
            continue
        for k in gt_list[i].keys():
            total += 1
            missing.append(f"{path}[{i}].{k}")

    return correct, total, mismatches, missing

def compare_json(gt: Any, pred: Any, path: str = "") -> Tuple[int, int, List[str], List[str]]:
    if isinstance(gt, dict):
        correct = 0
        total = 0
        mismatches: List[str] = []
        missing: List[str] = []
        if not isinstance(pred, dict):
            # everything under gt considered missing, but skip empty-value trees
            for k, v in gt.items():
                p = f"{path}.{k}".lstrip(".")
                w = _leaf_weight(v)
                if w > 0:
                    missing.append(p)
                    total += w
            return 0, total, mismatches, missing
        for k, v in gt.items():
            p = f"{path}.{k}" if path else k
            # CHANGED: find matching key in pred ignoring case
            pk = _ci_lookup_key(pred, k)
            if pk is None:
                # key missing entirely in prediction; skip penalty for empty values
                w = _leaf_weight(v)
                if w > 0:
                    missing.append(p)
                    total += w
                continue
            c, t, mm, ms = compare_json(v, pred[pk], p)
            correct += c
            total += t
            mismatches.extend(mm)
            missing.extend(ms)
        return correct, total, mismatches, missing

    if isinstance(gt, list):
        # If GT list is empty, no penalty regardless of pred type
        if len(gt) == 0:
            return 0, 0, [], []
        if not isinstance(pred, list):
            # all gt entries considered missing (non-empty GT list)
            missing = [f"{path}[]"]
            if isinstance(gt[0], dict):
                total = sum(len(item.keys()) for item in gt)
            else:
                total = len(gt)
            return 0, total, [], missing

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

    # Scalars
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
    # Response-shaped GT: ensure keys and return
    if "room_charges" in gt and "hospital_medical_services" in gt:
        gt_copy = gt.copy()
        if not isinstance(gt_copy.get("room_charges"), dict):
            gt_copy["room_charges"] = {"ACCOMMODATION": []}
        else:
            gt_copy["room_charges"].setdefault("ACCOMMODATION", [])
        if not isinstance(gt_copy.get("hospital_medical_services"), dict):
            gt_copy["hospital_medical_services"] = {}
        if isinstance(gt_copy["hospital_medical_services"], dict):
            for sub in HOSPITAL_SUBCATS:
                gt_copy["hospital_medical_services"].setdefault(sub, [])
        if not isinstance(gt_copy.get("consultation_fees"), dict):
            gt_copy["consultation_fees"] = {}
        if isinstance(gt_copy["consultation_fees"], dict):
            for sub in CONSULT_SUBCATS:
                gt_copy["consultation_fees"].setdefault(sub, [])
        for field in ["total_room_charges", "total_hospital_medical_services", "total_hospital_charges", "total_consultant_fees", "grand_total"]:
            gt_copy.setdefault(field, "")
        return gt_copy

    # Legacy nested structure (preserve items)
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
    stem = os.path.splitext(name)[0]
    for suf in ["_clean_response", "_defect_response", "_response"]:
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
    return stem

def evaluate_folder(pred_dir: str) -> Tuple[str, float, float, int]:
    os.makedirs(pred_dir, exist_ok=True)
    pred_files = sorted(glob.glob(os.path.join(pred_dir, "*.json")))
    if not pred_files:
        logger.warning(f"No prediction JSONs found in {pred_dir}")

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
        # NEW: lowercase keys for both GT and prediction before comparing
        gt = _lower_keys(gt)
        pred = _lower_keys(pred)

        # Compare to get correct and mismatches (present fields)
        correct, _, mismatches, missing = compare_json(gt, pred)

        # NEW: compute totals consistently
        schema_total = _count_gt_leaves(gt)  # from GT only
        present_total = correct + len(mismatches)  # fields we actually compared
        missing_count = max(schema_total - present_total, 0)

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

    # CSV path
    folder_tag = os.path.basename(pred_dir.rstrip("\\/"))
    csv_out = os.path.join(pred_dir, f"eval_results_{folder_tag}.csv")
    os.makedirs(os.path.dirname(csv_out), exist_ok=True)

    # NEW: safe CSV write with fallback on PermissionError
    def _write_csv_safe(target_path: str, rows: List[Dict[str, Any]]) -> str:
        try:
            with open(target_path, "w", newline="", encoding="utf-8") as f:
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
            return target_path
        except PermissionError:
            ts = int(time.time())
            fallback = os.path.join(
                os.path.dirname(target_path),
                f"eval_results_{folder_tag}_{ts}.csv"
            )
            logger.warning(f"Permission denied for {target_path}. Writing to fallback {fallback}")
            with open(fallback, "w", newline="", encoding="utf-8") as f:
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
            return fallback

    csv_out = _write_csv_safe(csv_out, rows)  # NEW: use safe writer
    logger.info(f"CSV written: {csv_out}")

    avg_schema = round(sum(accs_schema) / len(accs_schema), 4) if accs_schema else 0.0
    avg_values = round(sum(accs_values) / len(accs_values), 4) if accs_values else 0.0
    return csv_out, avg_schema, avg_values, len(rows)

def main():
    summary_rows = []
    for d in PRED_DIRS:
        csv_path, avg_schema_acc, avg_values_acc, count = evaluate_folder(d)
        summary_rows.append({
            "folder": os.path.basename(d),
            "files_evaluated": count,
            "average_schema_accuracy": avg_schema_acc,
            "average_values_accuracy": avg_values_acc,
            "csv": csv_path
        })

    summary_csv = os.path.join(ROOT_DIR, "eval_summary_gemini_saved.csv")
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
