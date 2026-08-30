"""Turn OCR line data into structured fields: store_name, date, items, total_amount."""

from __future__ import annotations

import re
from typing import Optional


_DATE_PATTERNS = [
    # \s* instead of \s+ because OCR sometimes drops the space between date and time
    re.compile(
        r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\s*\d{1,2}:\d{2}(:\d{2})?\s*(AM|PM)?)",
        re.IGNORECASE,
    ),
    re.compile(r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})"),
    # Named month — only real month abbreviations, not product codes like "02 TUM 0812"
    re.compile(
        r"(\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|"
        r"Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|"
        r"Nov(?:ember)?|Dec(?:ember)?)\.?\s+\d{2,4})",
        re.IGNORECASE,
    ),
]

# Priority-3 keywords (strong, unambiguous "this is the total" signals)
_TOTAL_KEYWORDS = [
    "ROUNDED TOTAL",
    "GRAND TOTAL",
    "TOTAL",
    "AMOUNT DUE",
    "CASH TEND",
    "VISA TEND",
    "BALANCE DUE",
    "NET AMOUNT",
    "NET AMT",
    "BAL",
]

# Weak OCR-misread variants — last resort (priority 1), "TOTA" etc. risk false positives
_WEAK_TOTAL_KEYWORDS = ["TOTA", "TOTAI"]

_EXCLUDE_KEYWORDS = ["TAX", "GST", "SUBTOTAL", "SUB TOTAL", "CHANGE", "CASH"]

_REFERENCE_PATTERNS = [
    re.compile(r"\bREF\s*#", re.IGNORECASE),
    re.compile(r"\bTERMINAL\b", re.IGNORECASE),
    re.compile(r"\bTRANS(?:\s+|ACTION)\s*ID\b", re.IGNORECASE),
    re.compile(r"\bAPPROVAL\s*#", re.IGNORECASE),
    re.compile(r"\bTC\s*#", re.IGNORECASE),
    re.compile(r"\bVALIDATION\b", re.IGNORECASE),
    # 8+ consecutive digits — phone, receipt ID, barcode, etc.
    re.compile(r"\d{8,}"),
]

# Currency-like number — also accepts '-' as decimal (OCR misread) and stray spaces around separators
_CURRENCY_RE = re.compile(r"[\$RMrm]?\s*(\d{1,6}(?:\s?[,.\-]\s?\d{1,2})?)")
_CURRENCY_STRICT_RE = re.compile(r"[\$RMrm]?\s*(\d{1,6}\s?[.\-]\s?\d{2})")


def _normalize_amount_string(raw: str) -> str:
    """Normalize amount string to X.XX format (handles comma/dash as decimal OCR misreads)."""
    s = raw.strip()
    # Collapse stray spaces around decimal separator (OCR artifact)
    s = re.sub(r"\s*([.,\-])\s*", r"\1", s)

    if "-" in s and "." not in s and "," not in s:
        last_dash_idx = s.rfind("-")
        digits_after = s[last_dash_idx + 1:]
        if len(digits_after) == 2 and digits_after.isdigit():
            s = s[:last_dash_idx] + "." + digits_after
        else:
            s = s.replace("-", "")
        return s

    if "," in s and "." not in s:
        last_comma_idx = s.rfind(",")
        digits_after = s[last_comma_idx + 1:]
        if len(digits_after) == 2 and digits_after.isdigit():
            s = s[:last_comma_idx] + "." + digits_after
        else:
            s = s.replace(",", "")
        return s

    return s.replace(",", "")


def _best_match_on_line(pattern: re.Pattern, text: str) -> Optional[str]:
    """Return rightmost match — amounts are right-aligned, first number is usually a quantity."""
    matches = pattern.findall(text)
    if not matches:
        return None
    return matches[-1]


def _safe_bbox(line: dict) -> list | None:
    """Return bbox if well-formed, else None."""
    bbox = line.get("bbox")
    if not bbox or len(bbox) != 2:
        return None
    return bbox


def _extract_store_name(ocr_lines: list[dict]) -> str:
    """Largest font-height text in top 15% of image, or first line as fallback."""
    if not ocr_lines:
        return ""

    valid_lines = [l for l in ocr_lines if _safe_bbox(l) is not None]
    if not valid_lines:
        return ocr_lines[0]["text"].strip() if ocr_lines else ""

    all_y_max = max(line["bbox"][1][1] for line in valid_lines)
    top_cutoff = all_y_max * 0.15
    top_lines = [l for l in valid_lines if l["bbox"][0][1] <= top_cutoff]

    if not top_lines:
        for line in valid_lines:
            if line["text"].strip():
                return line["text"].strip()
        return ""

    # Pick the tallest font in the top band
    def font_height(line: dict) -> float:
        return line["bbox"][1][1] - line["bbox"][0][1]

    best = max(top_lines, key=font_height)
    return best["text"].strip()


def _extract_date(ocr_lines: list[dict]) -> str:
    """Search all lines with the date regex bank in priority order."""
    for line in ocr_lines:
        text = line["text"]
        for pattern in _DATE_PATTERNS:
            match = pattern.search(text)
            if match:
                return match.group(1).strip()
    return ""


def _detect_columns(ocr_lines: list[dict]) -> tuple[float, float]:
    """Detect x-positions for description and price columns via left-edge clustering."""
    valid_lines = [l for l in ocr_lines if _safe_bbox(l) is not None]

    price_left_edges: list[float] = []
    desc_left_edges: list[float] = []

    for line in valid_lines:
        text = line["text"]
        if _CURRENCY_STRICT_RE.search(text):
            # Use bbox left edge as a proxy for price column position
            price_left_edges.append(line["bbox"][0][0])
        else:
            desc_left_edges.append(line["bbox"][0][0])

    if not price_left_edges:
        # Fallback: heuristic split at 60% of image width
        if valid_lines:
            max_x = max(l["bbox"][1][0] for l in valid_lines)
            return max_x * 0.6, max_x * 0.75
        return 300.0, 500.0

    price_x_median = float(sorted(price_left_edges)[len(price_left_edges) // 2])

    if desc_left_edges:
        desc_x_threshold = float(sorted(desc_left_edges)[len(desc_left_edges) // 2]) + 50
    else:
        desc_x_threshold = price_x_median * 0.6

    return desc_x_threshold, price_x_median


def _extract_items(ocr_lines: list[dict]) -> list[dict[str, str]]:
    """Extract line items — needs both a description (left zone) and a price (right zone)."""
    valid_lines = [l for l in ocr_lines if _safe_bbox(l) is not None]

    desc_threshold, price_x = _detect_columns(valid_lines)
    items: list[dict[str, str]] = []

    for line in valid_lines:
        text = line["text"].strip()
        if not text:
            continue

        upper = text.upper()
        if any(kw in upper for kw in _TOTAL_KEYWORDS):
            continue
        if any(kw in upper for kw in _EXCLUDE_KEYWORDS):
            continue
        if len(text) < 3:
            continue

        price_match = _CURRENCY_STRICT_RE.search(text)
        if not price_match:
            continue

        price_str = price_match.group(1)
        desc_part = text[: price_match.start()].strip()
        desc_part = _clean_item_description(desc_part)

        if desc_part and price_str:
            items.append({"name": desc_part, "price": price_str})

    return items


def _clean_item_description(desc: str) -> str:
    """Strip trailing qty/unit markers, e.g. 'BANANAS 1 Lb' -> 'BANANAS'."""
    desc = re.sub(r"\s+x\d+$", "", desc, flags=re.IGNORECASE)
    desc = re.sub(r"\s+\d+\s*(?:lb|kg|oz|pk|ct|ea|pc|gm|g|l|ml)s?\s*$", "", desc, flags=re.IGNORECASE)
    desc = re.sub(r"\s+\d+\s*$", "", desc)
    desc = desc.rstrip(" .,-/")
    return desc


def _is_reference_line(text: str) -> bool:
    """True if text looks like a reference/transaction/terminal number line."""
    return any(p.search(text) for p in _REFERENCE_PATTERNS)


def _extract_total(ocr_lines: list[dict]) -> str:
    """Pick the best total-amount line, favoring ROUNDED/GRAND TOTAL over weaker signals."""
    valid_lines = [l for l in ocr_lines if _safe_bbox(l) is not None]

    best_total: Optional[str] = None
    best_priority = -1  # higher = better

    for line in valid_lines:
        text = line["text"].strip()
        upper = text.upper()

        if _is_reference_line(text):
            continue

        priority = -1
        if "ROUNDED TOTAL" in upper:
            priority = 5
        elif "GRAND TOTAL" in upper:
            priority = 4
        elif "TOTAL" in upper and not any(ex in upper for ex in ["SUBTOTAL", "SUB TOTAL"]):
            priority = 3
        elif any(kw in upper for kw in ("AMOUNT DUE", "BALANCE DUE", "NET AMOUNT", "NET AMT", "BAL")):
            priority = 2
        elif "CASH TEND" in upper or "VISA TEND" in upper:
            priority = 1
        elif any(kw in upper for kw in _WEAK_TOTAL_KEYWORDS):
            # weak OCR misreads ("TOTA", "TOTAI") — last resort only
            priority = 1

        if priority < 0:
            continue

        if priority <= 1 and any(ex in upper for ex in ["TAX", "GST", "SUBTOTAL", "SUB TOTAL"]):
            continue

        # Take rightmost number — amounts are right-aligned,
        # first number is usually a quantity
        strict_val = _best_match_on_line(_CURRENCY_STRICT_RE, text)
        loose_val = _best_match_on_line(_CURRENCY_RE, text) if strict_val is None else None
        raw_val = strict_val if strict_val is not None else loose_val
        if raw_val is None:
            continue

        price_str = _normalize_amount_string(raw_val)

        try:
            numeric_val = float(price_str)
        except ValueError:
            continue

        is_strict_match = strict_val is not None
        is_strong_keyword = priority >= 3

        # Reject >5000 unless strict match on a strong keyword
        if numeric_val > 5000:
            if not (is_strict_match and is_strong_keyword):
                continue

        # Tighter ceiling for weak keywords with loose matches
        if priority == 1 and not is_strict_match and numeric_val > 2000:
            continue

        if priority > best_priority:
            best_priority = priority
            best_total = price_str

    # Fallback: any line mentioning TOTAL
    if best_total is None:
        for line in valid_lines:
            text = line["text"].strip()
            upper = text.upper()
            has_total_kw = "TOTAL" in upper or any(kw in upper for kw in _WEAK_TOTAL_KEYWORDS)
            if has_total_kw:
                if _is_reference_line(text):
                    continue
                strict_val_fb = _best_match_on_line(_CURRENCY_STRICT_RE, text)
                loose_val_fb = _best_match_on_line(_CURRENCY_RE, text) if strict_val_fb is None else None
                raw_val_fb = strict_val_fb if strict_val_fb is not None else loose_val_fb
                if raw_val_fb is not None:
                    price_str = _normalize_amount_string(raw_val_fb)
                    try:
                        numeric_val = float(price_str)
                    except ValueError:
                        continue
                    is_strong = "ROUNDED TOTAL" in upper or "GRAND TOTAL" in upper or (
                        "TOTAL" in upper and not any(ex in upper for ex in ["SUBTOTAL", "SUB TOTAL"])
                    )
                    if numeric_val > 5000 and not (strict_val_fb is not None and is_strong):
                        continue
                    best_total = price_str
                    break

    return best_total or ""


def extract_fields(ocr_lines: list[dict]) -> dict:
    """Extract structured fields from OCR line-grouped data."""
    return {
        "store_name": _extract_store_name(ocr_lines),
        "date": _extract_date(ocr_lines),
        "items": _extract_items(ocr_lines),
        "total_amount": _extract_total(ocr_lines),
    }


def main() -> None:
    """Smoke test: run extraction on a few sample images."""
    import sys
    from pathlib import Path

    import cv2

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    dataset_dir = project_root / "data" / "AI-OCR dataset"

    sys.path.insert(0, str(script_dir))
    from preprocess import preprocess
    from ocr_engine import run_ocr

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