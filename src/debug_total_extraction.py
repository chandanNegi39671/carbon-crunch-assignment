"""
Debug script: Investigate why total_amount extraction returns empty
for specific receipts.

This is a read-only diagnostic tool — does NOT modify extract_fields.py.

NOTE: This script must NOT import preprocess/ocr_engine at module level
due to a known EasyOCR segfault when torch + cv2 are imported first.
All imports are deferred to main().
"""

from __future__ import annotations

import csv
import re
import sys
import json
from pathlib import Path


# ── Target filenames (stems) ─────────────────────────────────────────
TARGET_STEMS = [
    "9", "10", "11", "13",
    "X51005255805", "X51005288570", "X51005361923",
    "X51005433548", "X51005442341", "X51005442388", "X51005442397",
    "X51005444037", "X51005447848", "X51005587267", "X51005684949",
    "X51005712021", "X51005719874", "X51005719893", "X51005719904",
    "X51005742068", "X51005745190", "X51005745249", "X51005764161",
    "X51005806695",
]


def main() -> None:
    # ── Deferred imports (order matters for EasyOCR) ────────────────
    import cv2
    import numpy as np
    import easyocr

    _SCRIPT_DIR = Path(__file__).resolve().parent
    _PROJECT_ROOT = _SCRIPT_DIR.parent
    _DATASET_DIR = _PROJECT_ROOT / "data" / "AI-OCR dataset"
    _TRIAGE_CSV = _PROJECT_ROOT / "outputs" / "triage" / "dataset_triage.csv"
    _JSON_DIR = _PROJECT_ROOT / "outputs" / "json"

    # Init EasyOCR BEFORE importing preprocess/ocr_engine
    reader = easyocr.Reader(["en"], gpu=False)
    print("EasyOCR reader initialized\n")

    # Now safe to import project modules
    sys.path.insert(0, str(_SCRIPT_DIR))
    from preprocess import preprocess
    from ocr_engine import _group_into_lines
    from extract_fields import (
        _TOTAL_KEYWORDS,
        _CURRENCY_STRICT_RE,
        _CURRENCY_RE,
        _REFERENCE_PATTERNS,
        _is_reference_line,
        extract_fields,
    )

    # Load triage buckets
    buckets: dict[str, str] = {}
    if _TRIAGE_CSV.exists():
        with open(_TRIAGE_CSV, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                buckets[row["filename"]] = row["bucket"]

    results: dict[int, list[str]] = {0: [], 1: [], 2: [], 3: []}
    examples: dict[int, list] = {1: [], 2: [], 3: []}

    for stem in TARGET_STEMS:
        # Find image file
        img_path = None
        for ext in (".jpg", ".jpeg", ".png", ".webp", ".JPG"):
            candidate = _DATASET_DIR / f"{stem}{ext}"
            if candidate.exists():
                img_path = candidate
                break

        if img_path is None:
            print(f"\n{'='*70}")
            print(f"  {stem}: IMAGE NOT FOUND")
            print(f"{'='*70}")
            results[0].append(stem)
            continue

        bucket = buckets.get(img_path.name, "clean")

        print(f"\n{'='*70}")
        print(f"  {stem}  (file={img_path.name}, bucket={bucket})")
        print(f"{'='*70}")

        # Run OCR with error handling
        try:
            raw = cv2.imread(str(img_path))
            if raw is None:
                raise ValueError("cv2.imread returned None")

            # Resize to avoid EasyOCR segfault on large images
            h, w = raw.shape[:2]
            scale = min(1.0, 1600 / max(h, w))
            if scale < 1.0:
                raw = cv2.resize(raw, (int(w * scale), int(h, w)))

            processed = preprocess(raw, bucket)

            # EasyOCR needs BGR
            if len(processed.shape) == 2:
                img_bgr = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
            else:
                img_bgr = processed

            ocr_results = reader.readtext(img_bgr)

            # Convert to word dicts
            words = []
            for bbox, text, conf in ocr_results:
                pts = np.array(bbox, dtype=np.float32)
                x_min, y_min = pts.min(axis=0)
                x_max, y_max = pts.max(axis=0)
                words.append({
                    "text": text.strip(),
                    "bbox": [[float(x_min), float(y_min)], [float(x_max), float(y_max)]],
                    "confidence": float(conf),
                })

            ocr_lines = _group_into_lines(words)

        except Exception as exc:
            print(f"  OCR FAILED: {exc}")
            # Fall back to existing JSON
            existing_json = _JSON_DIR / f"{stem}.json"
            if existing_json.exists():
                with open(existing_json, encoding="utf-8") as fh:
                    data = json.load(fh)
                total_val = data.get("total_amount", {}).get("value", "")
                print(f"  Using existing JSON total_amount: {total_val!r}")
                if not total_val:
                    results[1].append(stem)
                    examples[1].append({"stem": stem, "reason": "OCR failed, JSON has empty total"})
            else:
                results[1].append(stem)
                examples[1].append({"stem": stem, "reason": "OCR failed, no JSON exists"})
            continue

        # Print all OCR lines with analysis
        print(f"\n  OCR lines ({len(ocr_lines)}):")
        analyses = []
        for i, line in enumerate(ocr_lines):
            text = line["text"].strip()
            if not text:
                continue

            upper = text.upper()
            has_total_kw = any(kw in upper for kw in _TOTAL_KEYWORDS)
            matched_kws = [kw for kw in _TOTAL_KEYWORDS if kw in upper]
            is_ref_line = _is_reference_line(text)
            ref_patterns_hit = [p.pattern for p in _REFERENCE_PATTERNS if p.search(text)]
            strict_match = _CURRENCY_STRICT_RE.search(text)
            loose_match = _CURRENCY_RE.search(text) if not strict_match else None

            analysis = {
                "has_total_keyword": has_total_kw,
                "matched_keywords": matched_kws,
                "is_reference_line": is_ref_line,
                "ref_patterns_hit": ref_patterns_hit,
                "strict_currency": strict_match.group(0) if strict_match else None,
                "strict_value": strict_match.group(1) if strict_match else None,
                "loose_currency": loose_match.group(0) if loose_match else None,
                "loose_value": loose_match.group(1) if loose_match else None,
            }
            analyses.append(analysis)

            # Build display string
            flags = []
            if has_total_kw:
                flags.append(f"KW:{matched_kws}")
            if is_ref_line:
                flags.append(f"REF:{ref_patterns_hit}")
            if strict_match:
                flags.append(f"STRICT={strict_match.group(1)}")
            elif loose_match:
                flags.append(f"LOOSE={loose_match.group(1)}")

            flag_str = "  ".join(flags) if flags else "(no match)"
            print(f"    [{i:3d}] {text!r:60s}  {flag_str}")

        # Run extract_fields
        fields = extract_fields(ocr_lines)
        total = fields["total_amount"]
        print(f"\n  extract_fields total_amount: {total!r}")

        # ── Categorise ──────────────────────────────────────────────
        kw_lines = [a for a in analyses if a["has_total_keyword"]]

        if not kw_lines:
            cat, explanation = 1, "No line contains any total keyword"
        else:
            kw_with_currency = [
                a for a in kw_lines
                if a["strict_currency"] or a["loose_currency"]
            ]

            if not kw_with_currency:
                kw_blocked = [a for a in kw_lines if a["is_reference_line"]]
                if kw_blocked:
                    cat, explanation = 3, (
                        f"Keyword line blocked by _is_reference_line(). "
                        f"Blocked: {[a['matched_keywords'] for a in kw_blocked]}"
                    )
                else:
                    cat, explanation = 2, (
                        f"Keyword line exists but no currency regex matched. "
                        f"Keywords: {[a['matched_keywords'] for a in kw_lines]}"
                    )
            else:
                # Check for reference rejection or ceiling
                found_reject = False
                for a in kw_with_currency:
                    if a["is_reference_line"]:
                        val = a["strict_value"] or a["loose_value"]
                        cat, explanation = 3, (
                            f"Keyword+currency rejected by _is_reference_line(). "
                            f"Line: {a['matched_keywords']}, val={val}"
                        )
                        found_reject = True
                        break
                    val_str = a["strict_value"] or a["loose_value"]
                    if val_str:
                        try:
                            numeric = float(val_str.replace(",", ""))
                            if numeric > 5000:
                                is_strict = a["strict_currency"] is not None
                                kw_upper = [k.upper() for k in a["matched_keywords"]]
                                is_strong = any(
                                    k in ("ROUNDED TOTAL", "GRAND TOTAL", "TOTAL")
                                    for k in kw_upper
                                )
                                if not (is_strict and is_strong):
                                    cat, explanation = 3, (
                                        f"Keyword+currency rejected by ceiling. "
                                        f"Line: {a['matched_keywords']}, val={numeric}, "
                                        f"strict={is_strict}, strong={is_strong}"
                                    )
                                    found_reject = True
                                    break
                        except ValueError:
                            pass

                if not found_reject:
                    cat, explanation = 2, (
                        "Keyword+currency found but total still empty "
                        "(unexpected — likely higher-priority line won)"
                    )

        print(f"  -> CATEGORY {cat}: {explanation}")
        results[cat].append(stem)
        if len(examples[cat]) < 3:
            examples[cat].append({
                "stem": stem,
                "reason": explanation,
                "kw_lines": [
                    (a["matched_keywords"], a["strict_value"] or a["loose_value"],
                     a["is_reference_line"], a["ref_patterns_hit"])
                    for a in kw_lines[:5]
                ],
            })

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  DIAGNOSTIC SUMMARY")
    print(f"{'='*70}")
    if results[0]:
        print(f"  Image not found:   {len(results[0])} — {results[0]}")
    for cat in (1, 2, 3):
        items = results.get(cat, [])
        print(f"  Category {cat}: {len(items)} — {items}")
    print(f"  Total targets: {len(TARGET_STEMS)}")

    for cat in (1, 2, 3):
        if examples[cat]:
            print(f"\n  --- Category {cat} examples ---")
            for ex in examples[cat]:
                print(f"    {ex['stem']}: {ex['reason']}")
                if ex.get("kw_lines"):
                    for kw, val, is_ref, ref_hits in ex["kw_lines"]:
                        print(f"      KW={kw}, value={val}, is_ref={is_ref}, ref_hits={ref_hits}")


if __name__ == "__main__":
    main()
