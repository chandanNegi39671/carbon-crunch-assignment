"""Helper: analyze a single receipt for the debug_total_extraction script."""
import sys
sys.path.insert(0, "src")

import cv2
import numpy as np
import easyocr
from pathlib import Path
from preprocess import preprocess
from ocr_engine import _group_into_lines
from extract_fields import (
    extract_fields, _TOTAL_KEYWORDS, _CURRENCY_STRICT_RE,
    _CURRENCY_RE, _is_reference_line, _REFERENCE_PATTERNS,
)

reader = easyocr.Reader(["en"], gpu=False)
dataset = Path("data/AI-OCR dataset")

stem = sys.argv[1]
for ext in (".jpg", ".jpeg", ".png", ".JPG"):
    p = dataset / f"{stem}{ext}"
    if p.exists():
        break
else:
    print(f"NOT_FOUND|{stem}")
    sys.exit(0)

raw = cv2.imread(str(p))
h, w = raw.shape[:2]
scale = min(1.0, 1600 / max(h, w))
if scale < 1.0:
    raw = cv2.resize(raw, (int(w * scale), int(h * scale)))
processed = preprocess(raw, "clean")
if len(processed.shape) == 2:
    img_bgr = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
else:
    img_bgr = processed

ocr_results = reader.readtext(img_bgr)
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
fields = extract_fields(ocr_lines)
total = fields["total_amount"]

print(f"RESULT|{stem}|{total}")
for i, line in enumerate(ocr_lines):
    text = line["text"].strip()
    if not text:
        continue
    upper = text.upper()
    flags = []
    matched_kws = [kw for kw in _TOTAL_KEYWORDS if kw in upper]
    if matched_kws:
        flags.append(f"KW:{matched_kws}")
    is_ref = _is_reference_line(text)
    if is_ref:
        ref_hits = [p.pattern for p in _REFERENCE_PATTERNS if p.search(text)]
        flags.append(f"REF:{ref_hits}")
    sm = _CURRENCY_STRICT_RE.search(text)
    lm = _CURRENCY_RE.search(text) if not sm else None
    if sm:
        flags.append(f"STRICT={sm.group(1)}")
    elif lm:
        flags.append(f"LOOSE={lm.group(1)}")
    flag_str = "  ".join(flags) if flags else ""
    print(f"LINE|{i}|{text}|{flag_str}")
