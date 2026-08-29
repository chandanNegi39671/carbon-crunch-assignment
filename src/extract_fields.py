"""
Module 4: Field Extraction
Turn OCR line data into structured fields: store_name, date, items, total_amount.

Strategy:
- store_name: largest font-height text in the top 15% of image, else first line
- date: regex bank with priority ordering
- items: x-coordinate column clustering to find DESC / PRICE columns
- total_amount: keyword search with exclusion of TAX/GST/SUBTOTAL lines
"""

from __future__ import annotations

import re
from typing import Optional


# ─── Date regex bank (applied in priority order) ─────────────────────
_DATE_PATTERNS = [
    # Full timestamp: DD/MM/YYYY HH:MM:SS AM/PM
    # Uses \s* (not \s+) to handle OCR dropping the space between date and time
    re.compile(
        r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\s*\d{1,2}:\d{2}(:\d{2})?\s*(AM|PM)?)",
        re.IGNORECASE,
    ),
    # Date only: DD/MM/YYYY or MM-DD-YYYY
    re.compile(r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})"),
    # Named month: 12 Jan 2024 — requires real month abbreviation
    # to avoid false positives on product codes like "02 TUM 0812"
    re.compile(
        r"(\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|"
        r"Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|"
        r"Nov(?:ember)?|Dec(?:ember)?)\.?\s+\d{2,4})",
        re.IGNORECASE,
    ),
]

# ─── Total / keyword patterns ────────────────────────────────────────
_TOTAL_KEYWORDS = [
    "ROUNDED TOTAL",
    "GRAND TOTAL",
    "TOTAL",
    "AMOUNT DUE",
    "CASH TEND",
    "VISA TEND",
    "BALANCE DUE",
    "NET AMOUNT",
]

_EXCLUDE_KEYWORDS = ["TAX", "GST", "SUBTOTAL", "SUB TOTAL", "CHANGE", "CASH"]

# Reference / transaction line patterns — skip these entirely for total candidacy
_REFERENCE_PATTERNS = [
    re.compile(r"\bREF\s*#", re.IGNORECASE),
    re.compile(r"\bTERMINAL\b", re.IGNORECASE),
    re.compile(r"\bTRANS(?:\s+|ACTION)\s*ID\b", re.IGNORECASE),
    re.compile(r"\bAPPROVAL\s*#", re.IGNORECASE),
    re.compile(r"\bTC\s*#", re.IGNORECASE),
    re.compile(r"\bVALIDATION\b", re.IGNORECASE),
    # Long digit run: 8+ consecutive digits (phone, receipt ID, barcode, etc.)
    re.compile(r"\d{8,}"),
]

# Currency-like number: optional currency symbol + digits with optional decimals
_CURRENCY_RE = re.compile(r"[\$RMrm]?\s*(\d{1,6}(?:[,.]\d{1,2})?)")
_CURRENCY_STRICT_RE = re.compile(r"[\$RMrm]?\s*(\d{1,6}\.\d{2})")


# ─── Defensive bbox access ─────────────────────────────────────────
def _safe_bbox(line: dict) -> list | None:
    """Return the bbox from an OCR line if it is well-formed, else None."""
    bbox = line.get("bbox")
    if not bbox or len(bbox) != 2:
        return None
    return bbox


# ─── Store name extraction ───────────────────────────────────────────
def _extract_store_name(ocr_lines: list[dict]) -> str:
    """Largest average font-height text in the top 15% of image height,
    or first non-empty line as fallback."""
    if not ocr_lines:
        return ""

    # Filter to lines with valid bbox
    valid_lines = [l for l in ocr_lines if _safe_bbox(l) is not None]
    if not valid_lines:
        return ocr_lines[0]["text"].strip() if ocr_lines else ""

    # Determine image height from bounding boxes
    all_y_max = max(line["bbox"][1][1] for line in valid_lines)
    top_cutoff = all_y_max * 0.15

    # Filter lines in top 15%
    top_lines = [l for l in valid_lines if l["bbox"][0][1] <= top_cutoff]

    if not top_lines:
        # Fallback: first non-empty line with valid bbox
        for line in valid_lines:
            if line["text"].strip():
                return line["text"].strip()
        return ""

    # Among top lines, pick the one with the largest average font height
    def font_height(line: dict) -> float:
        return line["bbox"][1][1] - line["bbox"][0][1]

    best = max(top_lines, key=font_height)
    return best["text"].strip()


# ─── Date extraction ─────────────────────────────────────────────────
def _extract_date(ocr_lines: list[dict]) -> str:
    """Search all lines with the date regex bank in priority order."""
    for line in ocr_lines:
        text = line["text"]
        for pattern in _DATE_PATTERNS:
            match = pattern.search(text)
            if match:
                return match.group(1).strip()
    return ""


# ─── Column clustering for item extraction ───────────────────────────
def _detect_columns(ocr_lines: list[dict]) -> tuple[float, float]:
    """Detect the dominant x-positions for the description column and
    the price/amount column by clustering left-edge x-coordinates of
    price-like tokens across all lines.

    Returns (desc_x_threshold, price_x_median): tokens with x < threshold
    are in the description zone; price tokens cluster around the median.
    """
    # Filter to lines with valid bbox
    valid_lines = [l for l in ocr_lines if _safe_bbox(l) is not None]

    price_left_edges: list[float] = []
    desc_left_edges: list[float] = []

    for line in valid_lines:
        text = line["text"]
        # Check if this line contains a price-like number
        if _CURRENCY_STRICT_RE.search(text):
            # The price portion is likely the right-most token
            # Use the line's bounding box right edge as a proxy
            price_left_edges.append(line["bbox"][0][0])
        else:
            desc_left_edges.append(line["bbox"][0][0])

    if not price_left_edges:
        # Fallback: use a heuristic split at 60% of image width
        if valid_lines:
            max_x = max(l["bbox"][1][0] for l in valid_lines)
            return max_x * 0.6, max_x * 0.75
        return 300.0, 500.0

    price_x_median = float(sorted(price_left_edges)[len(price_left_edges) // 2])

    # Description column is typically the leftmost cluster
    if desc_left_edges:
        desc_x_threshold = float(sorted(desc_left_edges)[len(desc_left_edges) // 2]) + 50
    else:
        desc_x_threshold = price_x_median * 0.6

    return desc_x_threshold, price_x_median


def _extract_items(ocr_lines: list[dict]) -> list[dict[str, str]]:
    """Extract line items using column clustering.

    A valid item line has both a description (left zone) and a price
    (right zone / price-like number).
    """
    # Filter to lines with valid bbox
    valid_lines = [l for l in ocr_lines if _safe_bbox(l) is not None]

    desc_threshold, price_x = _detect_columns(valid_lines)
    items: list[dict[str, str]] = []

    for line in valid_lines:
        text = line["text"].strip()
        if not text:
            continue

        # Skip lines that look like headers, totals, or tax
        upper = text.upper()
        if any(kw in upper for kw in _TOTAL_KEYWORDS):
            continue
        if any(kw in upper for kw in _EXCLUDE_KEYWORDS):
            continue
        # Skip very short lines (likely noise)
        if len(text) < 3:
            continue

        # Try to find a price at the end of the line
        price_match = _CURRENCY_STRICT_RE.search(text)
        if not price_match:
            continue

        price_str = price_match.group(1)
        # Everything before the price is the description
        desc_part = text[: price_match.start()].strip()

        # Clean up description: remove stray qty/unit tokens at the end
        desc_part = _clean_item_description(desc_part)

        if desc_part and price_str:
            items.append({"name": desc_part, "price": price_str})

    return items


def _clean_item_description(desc: str) -> str:
    """Remove trailing quantity/unit markers from an item description,
    e.g. 'BANANAS 1 Lb' -> 'BANANAS', 'CLIF BAR x3' -> 'CLIF BAR'."""
    # Remove trailing patterns like "x3", "1 Lb", "2 Pk", etc.
    desc = re.sub(r"\s+x\d+$", "", desc, flags=re.IGNORECASE)
    desc = re.sub(r"\s+\d+\s*(?:lb|kg|oz|pk|ct|ea|pc|gm|g|l|ml)s?\s*$", "", desc, flags=re.IGNORECASE)
    # Remove trailing standalone digits (qty)
    desc = re.sub(r"\s+\d+\s*$", "", desc)
    # Remove trailing punctuation
    desc = desc.rstrip(" .,-/")
    return desc


# ─── Reference-line guard ────────────────────────────────────────────
def _is_reference_line(text: str) -> bool:
    """Return True if *text* looks like a reference/transaction/terminal
    number line that should never be treated as a total amount candidate."""
    return any(p.search(text) for p in _REFERENCE_PATTERNS)


# ─── Total amount extraction ─────────────────────────────────────────
def _extract_total(ocr_lines: list[dict]) -> str:
    """Search for total amount, prioritising ROUNDED TOTAL / GRAND TOTAL,
    excluding TAX / GST / SUBTOTAL lines unless no better candidate exists.
    """
    # Filter to lines with valid bbox
    valid_lines = [l for l in ocr_lines if _safe_bbox(l) is not None]

    best_total: Optional[str] = None
    best_priority = -1  # higher = better

    for line in valid_lines:
        text = line["text"].strip()
        upper = text.upper()

        # Skip lines that look like reference/transaction numbers entirely
        if _is_reference_line(text):
            continue

        # Determine priority of this line
        priority = -1
        if "ROUNDED TOTAL" in upper:
            priority = 5
        elif "GRAND TOTAL" in upper:
            priority = 4
        elif "TOTAL" in upper and not any(ex in upper for ex in ["SUBTOTAL", "SUB TOTAL"]):
            priority = 3
        elif "AMOUNT DUE" in upper or "BALANCE DUE" in upper or "NET AMOUNT" in upper:
            priority = 2
        elif "CASH TEND" in upper or "VISA TEND" in upper:
            priority = 1

        if priority < 0:
            continue

        # Check for exclusion keywords
        if priority <= 1 and any(ex in upper for ex in ["TAX", "GST", "SUBTOTAL", "SUB TOTAL"]):
            continue

        # Extract price-like number from this line
        # Prefer strict format (X.XX), then loose
        strict_match = _CURRENCY_STRICT_RE.search(text)
        price_match = strict_match if strict_match else _CURRENCY_RE.search(text)
        if not price_match:
            continue

        price_str = price_match.group(1).replace(",", "")

        # ── Sanity range checks ────────────────────────────────────
        try:
            numeric_val = float(price_str)
        except ValueError:
            continue

        is_strict_match = strict_match is not None
        is_strong_keyword = priority >= 3

        # General ceiling: > 5000 is implausible unless strict match on strong keyword
        if numeric_val > 5000:
            if not (is_strict_match and is_strong_keyword):
                continue

        # Lower ceiling for weak keywords (CASH TEND / VISA TEND) with loose matches
        if priority == 1 and not is_strict_match and numeric_val > 2000:
            continue

        if priority > best_priority:
            best_priority = priority
            best_total = price_str

    # Fallback: look for any line with "TOTAL" and grab the number
    if best_total is None:
        for line in valid_lines:
            text = line["text"].strip()
            upper = text.upper()
            if "TOTAL" in upper:
                # Skip reference lines even in fallback
                if _is_reference_line(text):
                    continue
                strict_match_fb = _CURRENCY_STRICT_RE.search(text)
                loose_match_fb = _CURRENCY_RE.search(text) if not strict_match_fb else None
                match = strict_match_fb or loose_match_fb
                if match:
                    price_str = match.group(1).replace(",", "")
                    try:
                        numeric_val = float(price_str)
                    except ValueError:
                        continue
                    # Same ceiling: > 5000 rejected unless strict match on strong keyword
                    is_strong = "ROUNDED TOTAL" in upper or "GRAND TOTAL" in upper or (
                        "TOTAL" in upper and not any(ex in upper for ex in ["SUBTOTAL", "SUB TOTAL"])
                    )
                    if numeric_val > 5000 and not (strict_match_fb is not None and is_strong):
                        continue
                    best_total = price_str
                    break

    return best_total or ""


# ─── Public API ──────────────────────────────────────────────────────
def extract_fields(ocr_lines: list[dict]) -> dict:
    """Extract structured fields from OCR line-grouped data.

    Parameters
    ----------
    ocr_lines : list[dict]
        Output from ``ocr_engine.run_ocr()`` — each dict has
        ``text``, ``bbox``, ``confidence``, ``line_id``.

    Returns
    -------
    dict
        ``{"store_name": str, "date": str,
          "items": [{"name": str, "price": str}],
          "total_amount": str}``
    """
    return {
        "store_name": _extract_store_name(ocr_lines),
        "date": _extract_date(ocr_lines),
        "items": _extract_items(ocr_lines),
        "total_amount": _extract_total(ocr_lines),
    }


# ─── Standalone test runner ──────────────────────────────────────────
def main() -> None:
    """Smoke test: run extraction on a few sample images end-to-end."""
    import sys
    from pathlib import Path

    import cv2

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    dataset_dir = project_root / "data" / "AI-OCR dataset"

    sys.path.insert(0, str(script_dir))
    from preprocess import preprocess
    from ocr_engine import run_ocr

    # Load triage CSV
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

    print(f"Testing field extraction on {len(samples)} sample(s)...\n")
    for bucket, fname in sorted(samples.items()):
        img_path = dataset_dir / fname
        print(f"  {fname}  (bucket={bucket})")
        raw = cv2.imread(str(img_path))
        if raw is None:
            print(f"    SKIP")
            continue

        processed = preprocess(raw, bucket)
        ocr_lines = run_ocr(processed, bucket)
        fields = extract_fields(ocr_lines)

        print(f"    store_name : {fields['store_name']}")
        print(f"    date       : {fields['date']}")
        print(f"    total      : {fields['total_amount']}")
        print(f"    items ({len(fields['items'])}):")
        for item in fields["items"][:5]:
            print(f"      - {item['name']}: {item['price']}")
        if len(fields["items"]) > 5:
            print(f"      ... and {len(fields['items']) - 5} more items")
        print()


if __name__ == "__main__":
    main()
