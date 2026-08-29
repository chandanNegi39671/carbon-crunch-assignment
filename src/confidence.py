"""
Module 5: Confidence Scoring
Attach a 0-1 confidence score + reason to every extracted field.

Weighted formula:
  confidence = 0.5 * ocr_confidence
             + 0.3 * int(pattern_valid)
             + 0.2 * int(keyword_anchored)

If confidence < 0.7, set flag=True with a reason string.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Optional

# ─── Weights ─────────────────────────────────────────────────────────
_W_OCR = 0.5
_W_PATTERN = 0.3
_W_ANCHOR = 0.2
_FLAG_THRESHOLD = 0.7

# ─── Date format list for strptime validation ────────────────────────
_DATE_FORMATS = [
    # 4-digit year with time
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y %I:%M:%S %p",
    "%m/%d/%Y %H:%M:%S",
    "%m/%d/%Y %I:%M:%S %p",
    "%d/%m/%Y %H:%M",
    "%d/%m/%Y %I:%M %p",
    "%m/%d/%Y %H:%M",
    "%m/%d/%Y %I:%M %p",
    # 2-digit year with time
    "%d/%m/%y %H:%M:%S",
    "%d/%m/%y %I:%M:%S %p",
    "%m/%d/%y %H:%M:%S",
    "%m/%d/%y %I:%M:%S %p",
    "%d/%m/%y %H:%M",
    "%d/%m/%y %I:%M %p",
    "%m/%d/%y %H:%M",
    "%m/%d/%y %I:%M %p",
    # Date only
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%d/%m/%y",
    "%m/%d/%y",
    "%d-%m-%Y",
    "%m-%d-%Y",
    "%Y-%m-%d",
    "%d %b %Y",
    "%d %B %Y",
    "%b %d, %Y",
    "%B %d, %Y",
]

# ─── Currency cleaning regex ─────────────────────────────────────────
_CURRENCY_STRIP = re.compile(r"^(RM|Rs\.?|\$)\s*|[,\s]", re.IGNORECASE)


# ─── Pattern validation helpers ──────────────────────────────────────
def _is_valid_date(value: str) -> bool:
    """Does *value* parse with any known date format?"""
    if not value or not value.strip():
        return False
    v = value.strip()
    for fmt in _DATE_FORMATS:
        try:
            datetime.strptime(v, fmt)
            return True
        except ValueError:
            continue
    return False


def _is_valid_total(value: str) -> bool:
    """Does *value* look like a currency amount (e.g. 12.34)?"""
    if not value or not value.strip():
        return False
    cleaned = _CURRENCY_STRIP.sub("", value.strip())
    return bool(re.match(r"^\d+\.\d{1,2}$", cleaned))


def _is_valid_store_name(value: str) -> bool:
    """Is the store name non-empty and not purely numeric/symbolic?"""
    if not value or not value.strip():
        return False
    v = value.strip()
    # Reject if empty of letters
    if not re.search(r"[A-Za-z]", v):
        return False
    # Reject if very short (likely noise)
    if len(v) < 2:
        return False
    return True


def _pattern_valid(field_name: str, value: str) -> bool:
    """Route to the appropriate pattern validator."""
    if not value:
        return False
    if field_name == "date":
        return _is_valid_date(value)
    if field_name == "total_amount":
        return _is_valid_total(value)
    if field_name == "store_name":
        return _is_valid_store_name(value)
    # For items, always valid if non-empty
    return bool(value and value.strip())


# ─── Confidence reason determination ─────────────────────────────────
def _determine_reason(
    ocr_confidence: float,
    pattern_valid: bool,
    keyword_anchored: bool,
    value: str,
) -> Optional[str]:
    """Return the first applicable reason string for a flagged field."""
    if not value:
        return "missing_field"
    if ocr_confidence < 0.4:
        return "low_ocr_confidence"
    if not pattern_valid:
        return "unparsable_format"
    if not keyword_anchored:
        return "no_keyword_anchor"
    return None


# ─── Public API: score a single field ────────────────────────────────
def score_field(
    field_name: str,
    value: str,
    ocr_confidence: float,
    pattern_valid: bool,
    keyword_anchored: bool,
) -> dict:
    """Score a single extracted field.

    Parameters
    ----------
    field_name : str
        e.g. "store_name", "date", "total_amount"
    value : str
        Extracted text value.
    ocr_confidence : float
        Average OCR confidence for the words that produced this value (0-1).
    pattern_valid : bool
        Whether the value matches the expected pattern for its field type.
    keyword_anchored : bool
        Whether the field was found near its expected anchor keyword.

    Returns
    -------
    dict
        ``{"value": str, "confidence": float, "flag": bool, "reason": str | None}``
    """
    if not value:
        return {
            "value": "",
            "confidence": 0.0,
            "flag": True,
            "reason": "missing_field",
        }

    confidence = (
        _W_OCR * ocr_confidence
        + _W_PATTERN * int(pattern_valid)
        + _W_ANCHOR * int(keyword_anchored)
    )
    confidence = round(min(1.0, max(0.0, confidence)), 4)

    flag = confidence < _FLAG_THRESHOLD
    reason = None
    if flag:
        reason = _determine_reason(ocr_confidence, pattern_valid, keyword_anchored, value)

    return {
        "value": value,
        "confidence": confidence,
        "flag": flag,
        "reason": reason,
    }


# ─── Helper: find if a value appears near an anchor keyword ──────────
def _is_near_keyword(
    ocr_lines: list[dict],
    value: str,
    keywords: list[str],
    tolerance_y: float = 30.0,
) -> bool:
    """Check if any line containing *value* also contains one of the *keywords*,
    or if a keyword-line is within *tolerance_y* pixels vertically."""
    for line in ocr_lines:
        upper = line["text"].upper()
        # Direct: same line has keyword
        if value and value.upper() in upper and any(kw in upper for kw in keywords):
            return True
        if any(kw in upper for kw in keywords):
            # Check if a line with the value is nearby
            kw_y = (line["bbox"][0][1] + line["bbox"][1][1]) / 2.0
            for other in ocr_lines:
                if value and value in other["text"]:
                    other_y = (other["bbox"][0][1] + other["bbox"][1][1]) / 2.0
                    if abs(kw_y - other_y) <= tolerance_y:
                        return True
    return False


# ─── Public API: score an entire receipt ─────────────────────────────
def score_receipt(
    extracted: dict,
    ocr_lines: list[dict],
) -> dict:
    """Score all fields in an extracted receipt dict.

    Parameters
    ----------
    extracted : dict
        Output from ``extract_fields.extract_fields()``.
    ocr_lines : list[dict]
        Output from ``ocr_engine.run_ocr()`` — used to determine
        keyword anchoring and per-word OCR confidence.

    Returns
    -------
    dict
        Final receipt JSON with confidence-annotated fields.
    """
    # Compute average OCR confidence across all lines
    all_confs = [l["confidence"] for l in ocr_lines if l["text"].strip()]
    avg_ocr_conf = float(sum(all_confs) / len(all_confs)) if all_confs else 0.0

    # ── store_name ───────────────────────────────────────────────────
    store_val = extracted.get("store_name", "")
    store_anchored = _is_near_keyword(
        ocr_lines, store_val,
        ["STORE", "SHOP", "MALL", "CENTER", "SUPERMARKET", "MARKET"],
    )
    store_scored = score_field(
        "store_name", store_val, avg_ocr_conf,
        _pattern_valid("store_name", store_val),
        store_anchored,
    )

    # ── date ─────────────────────────────────────────────────────────
    date_val = extracted.get("date", "")
    date_anchored = _is_near_keyword(
        ocr_lines, date_val,
        ["DATE", "TIME", "TRANSACTION", "TRANS"],
    )
    date_scored = score_field(
        "date", date_val, avg_ocr_conf,
        _pattern_valid("date", date_val),
        date_anchored,
    )

    # ── total_amount ─────────────────────────────────────────────────
    total_val = extracted.get("total_amount", "")
    total_anchored = _is_near_keyword(
        ocr_lines, total_val,
        ["TOTAL", "AMOUNT DUE", "BALANCE DUE", "GRAND TOTAL", "ROUNDED TOTAL"],
    )
    total_scored = score_field(
        "total_amount", total_val, avg_ocr_conf,
        _pattern_valid("total_amount", total_val),
        total_anchored,
    )

    # ── items ────────────────────────────────────────────────────────
    scored_items: list[dict] = []
    for item in extracted.get("items", []):
        item_name = item.get("name", "")
        item_price = item.get("price", "")

        # Per-item OCR confidence: find the OCR line closest to this item
        item_ocr_conf = avg_ocr_conf
        for line in ocr_lines:
            if item_name and item_name in line["text"]:
                item_ocr_conf = line["confidence"]
                break

        price_valid = bool(re.match(r"^\d+\.\d{1,2}$", item_price))
        scored_items.append({
            "name": item_name,
            "price": item_price,
            "confidence": round(
                _W_OCR * item_ocr_conf + _W_PATTERN * int(price_valid) + _W_ANCHOR * 0.5,
                4,
            ),
        })

    return {
        "store_name": store_scored,
        "date": date_scored,
        "items": scored_items,
        "total_amount": total_scored,
    }


# ─── Standalone test runner ──────────────────────────────────────────
def main() -> None:
    """Smoke test: score a few sample receipts."""
    import json
    import sys
    from pathlib import Path

    import cv2

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    dataset_dir = project_root / "data" / "AI-OCR dataset"

    sys.path.insert(0, str(script_dir))
    from preprocess import preprocess
    from ocr_engine import run_ocr
    from extract_fields import extract_fields

    import csv
    triage_csv = project_root / "outputs" / "triage" / "dataset_triage.csv"
    if not triage_csv.exists():
        print("ERROR: Run dataset_triage.py first.", file=sys.stderr)
        sys.exit(1)

    buckets: dict[str, str] = {}
    with open(triage_csv, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            buckets[row["filename"]] = row["bucket"]

    samples: dict[str, str] = {}
    for fname, bucket in buckets.items():
        if bucket not in samples:
            samples[bucket] = fname

    print(f"Testing confidence scoring on {len(samples)} sample(s)...\n")
    for bucket, fname in sorted(samples.items()):
        img_path = dataset_dir / fname
        print(f"  {fname}  (bucket={bucket})")
        raw = cv2.imread(str(img_path))
        if raw is None:
            print("    SKIP")
            continue

        processed = preprocess(raw, bucket)
        ocr_lines = run_ocr(processed, bucket)
        extracted = extract_fields(ocr_lines)
        scored = score_receipt(extracted, ocr_lines)

        for key in ("store_name", "date", "total_amount"):
            s = scored[key]
            flag_str = " [FLAGGED]" if s["flag"] else ""
            print(f"    {key:15s}: {s['value']!r:30s}  conf={s['confidence']:.2f}  reason={s['reason']}{flag_str}")
        print(f"    items: {len(scored['items'])} scored")
        for it in scored["items"][:3]:
            print(f"      - {it['name']}: {it['price']}  (conf={it['confidence']:.2f})")
        print()


if __name__ == "__main__":
    main()
