import os
import json
import re
from typing import Any, Dict, List

BASE_DIR = r"c:\\Users\\Ram\\Desktop\\workspace\\document-extraction-pipeline"
INPUT_DIR = os.path.join(BASE_DIR, "ground_truth")
OUTPUT_DIR = os.path.join(BASE_DIR, "ground_truth_converted")

ALLOWED_SUBCATS = {
    "ACCOMMODATION",
    "MEDICAL_RECORD_SERVICES",
    "HOSPITAL_SUPPORT_FEES",
    "GENERAL_SUPPLIES",
    "RADIOGRAPHY_SUPPLIES",
    "SURGICAL_SUPPLIES",
    "DRUGS_FORMULARY",
    "MEDICAL_SUPPLIES",
    "LABORATORY",
    "DIAGNOSTIC_SERVICES",
    "NURSING_SERVICES",
    "EMERGENCY_MEDICAL_SERVICE",
    "EQUIPMENT_USAGE",
    "MEDICAL_GASES",
    "OPERATING_ROOM_FEE",
    "OPERATING_THEATER_FEES",
    "OT_SUPPORT",
    "OT_SERVICES",
    "OT_SUPPLIES_CONSUMABLES",
    "PACKAGE",
    "PPE_SUPPLIES",
    "PROCEDURES",
    "STERILE_ITEMS_AND_SETS",
    "PROCEDURE_FEES",
    "CONSULTATION_FEES",
    "REPORTING_FEES",
}

def normalize_subcat(name: str) -> str:
    if not name:
        return ""
    s = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").upper()
    s = re.sub(r"_+", "_", s)
    return s

def parse_number(x: Any) -> float:
    if isinstance(x, (int, float)):
        return float(x)
    if x is None:
        return 0.0
    s = str(x).strip()
    neg = s.startswith("(") and s.endswith(")")
    s = s.strip("()").replace(",", "")
    try:
        val = float(s) if s else 0.0
    except ValueError:
        val = 0.0
    return -val if neg else val

def parse_qty(x: Any) -> float | int:
    n = parse_number(x)
    return int(n) if float(n).is_integer() else float(n)

def convert_file(src_path: str) -> Dict[str, Any]:
    with open(src_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    doc = raw.get("document_details", {}) or {}
    patient = raw.get("patient_information", {}) or {}
    claim = raw.get("claim_details", {}) or {}
    bill = raw.get("billing_details", {}) or {}
    fin = raw.get("financial_information", {}) or {}

    # Prepare subcategory buckets
    subcat_details: Dict[str, List[Dict[str, Any]]] = {k: [] for k in ALLOWED_SUBCATS}
    flat_items: List[Dict[str, Any]] = []

    # Walk categories -> subcategories -> items
    if isinstance(bill, dict):
        for _cat_name, subcats in bill.items():
            if not isinstance(subcats, dict):
                continue
            for subcat_name, items in subcats.items():
                if not isinstance(items, list):
                    continue
                norm_key = normalize_subcat(subcat_name)
                for li in items:
                    item = {
                        "service_code": str(li.get("service_code", "")),
                        "description_of_service": str(li.get("description", "")),
                        "date": str(li.get("date", "")),
                        "qty": parse_qty(li.get("quantity", 0)),
                        "gross_amount": parse_number(li.get("gross_amount", 0)),
                        "discount": parse_number(li.get("discount", 0)),
                        "allocated_amount": parse_number(li.get("allocated_amount", 0)),
                    }
                    flat_items.append(item)
                    if norm_key in subcat_details:
                        subcat_details[norm_key].append(item)

    out = {
        "status": "success",
        "data": {
            "FORMAT": str(doc.get("format", "")),
            "BILL_NO": str(doc.get("bill_no", "")),
            "PATIENT_NAME": str(patient.get("full_name", "")),
            "IC_PASSPORT_NO": str(patient.get("identification_number", "")),
            "VISIT_TYPE": str(claim.get("visit_type", "")),
            "ADMISSION_DATE_TIME": str(claim.get("admission_date_time", "")),
            "DISCHARGE_DATE_TIME": str(claim.get("discharge_date_time", "")),
            "GL_REFERENCE_NO": str(claim.get("gl_reference_no", "")),
            "BILLING_CATEGORY": flat_items,
            "BILLING_SUBCATEGORY_DETAILS": subcat_details,
            "TOTAL_ROOM_CHARGES": parse_number(fin.get("total_room_charges", 0)),
            "TOTAL_HOSPITAL_MEDICAL_SERVICES": parse_number(fin.get("total_hospital_medical_services", 0)),
            "TOTAL_HOSPITAL_CHARGES": parse_number(fin.get("total_hospital_charges", 0)),
            "TOTAL_CONSULTANT_FEES": parse_number(fin.get("total_consultant_fees", 0)),
            "GRAND_TOTAL": parse_number(fin.get("grand_total", 0)),
        },
    }
    return out

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".json")]
    for fname in files:
        src = os.path.join(INPUT_DIR, fname)
        try:
            converted = convert_file(src)
            dst = os.path.join(OUTPUT_DIR, fname)
            with open(dst, "w", encoding="utf-8") as out_f:
                json.dump(converted, out_f, ensure_ascii=False, indent=2)
            print(f"Converted: {fname}")
        except Exception as e:
            print(f"Failed: {fname} -> {e}")

if __name__ == "__main__":
    main()
