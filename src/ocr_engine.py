"""Run OCR and return text + bounding boxes + per-word confidence."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

_easyocr_reader = None
_tesseract_available: Optional[bool] = None  # None = not checked yet

LINE_Y_TOLERANCE = 15  # pixels — words within this vertical range = same line


def _get_easyocr_reader():
    """Singleton EasyOCR reader (English, CPU)."""
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr
        _easyocr_reader = easyocr.Reader(["en"], gpu=False)
    return _easyocr_reader


def _is_tesseract_available() -> bool:
    """Check whether tesseract binary is callable."""
    global _tesseract_available
    if _tesseract_available is None:
        try:
            import pytesseract
            # This will raise if tesseract binary is not found
            pytesseract.get_tesseract_version()
            _tesseract_available = True
        except Exception:  # noqa: BLE001
            _tesseract_available = False
    return _tesseract_available


def _run_easyocr(image: np.ndarray) -> list[dict]:
    """Run EasyOCR on image (BGR or grayscale). Returns word-level dicts."""
    reader = _get_easyocr_reader()
    if len(image.shape) == 2:
        img_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        img_bgr = image

    results = reader.readtext(img_bgr)
    words: list[dict] = []
    for bbox, text, conf in results:
        # Convert rotated rect bbox to [x_min, y_min, x_max, y_max]
        pts = np.array(bbox, dtype=np.float32)
        x_min, y_min = pts.min(axis=0)
        x_max, y_max = pts.max(axis=0)
        words.append({
            "text": text.strip(),
            "bbox": [[float(x_min), float(y_min)], [float(x_max), float(y_max)]],
            "confidence": float(conf),
        })
    return words


def _run_pytesseract(image: np.ndarray) -> list[dict]:
    """Run pytesseract on image (BGR or grayscale). Returns word-level dicts."""
    import pytesseract
    from pytesseract import Output

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    data = pytesseract.image_to_data(gray, output_type=Output.DICT)
    words: list[dict] = []
    n = len(data["text"])
    for i in range(n):
        text = data["text"][i].strip()
        conf = float(data["conf"][i]) / 100.0  # pytesseract returns 0-100
        if not text or conf <= 0:
            continue
        x = float(data["left"][i])
        y = float(data["top"][i])
        w = float(data["width"][i])
        h = float(data["height"][i])
        words.append({
            "text": text,
            "bbox": [[x, y], [x + w, y + h]],
            "confidence": max(0.0, min(1.0, conf)),
        })
    return words


def _group_into_lines(words: list[dict]) -> list[dict]:
    """Cluster word-level detections into lines by y-coordinate proximity."""
    if not words:
        return []

    for w in words:
        y_min = w["bbox"][0][1]
        y_max = w["bbox"][1][1]
        w["_y_center"] = (y_min + y_max) / 2.0

    sorted_words = sorted(words, key=lambda w: w["_y_center"])

    # Group: words within LINE_Y_TOLERANCE pixels vertically
    lines: list[list[dict]] = []
    current_line: list[dict] = [sorted_words[0]]

    for w in sorted_words[1:]:
        ref_y = current_line[-1]["_y_center"]
        if abs(w["_y_center"] - ref_y) <= LINE_Y_TOLERANCE:
            current_line.append(w)
        else:
            lines.append(current_line)
            current_line = [w]
    lines.append(current_line)

    result: list[dict] = []
    for line_id, line_words in enumerate(lines):
        line_words.sort(key=lambda w: w["bbox"][0][0])
        text = " ".join(w["text"] for w in line_words)

        x_min = min(w["bbox"][0][0] for w in line_words)
        y_min = min(w["bbox"][0][1] for w in line_words)
        x_max = max(w["bbox"][1][0] for w in line_words)
        y_max = max(w["bbox"][1][1] for w in line_words)

        avg_conf = float(np.mean([w["confidence"] for w in line_words]))

        result.append({
            "text": text,
            "bbox": [[x_min, y_min], [x_max, y_max]],
            "confidence": avg_conf,
            "line_id": line_id,
        })

    return result


def _merge_lines(
    easyocr_lines: list[dict],
    tesseract_lines: list[dict],
) -> list[dict]:
    """Merge lines from two OCR engines, keeping the higher-confidence version for duplicates."""
    if not tesseract_lines:
        return easyocr_lines
    if not easyocr_lines:
        return tesseract_lines

    merged: list[dict] = list(easyocr_lines)

    for t_line in tesseract_lines:
        t_y = (t_line["bbox"][0][1] + t_line["bbox"][1][1]) / 2.0
        best_match: Optional[int] = None
        best_dist = float("inf")

        for idx, e_line in enumerate(merged):
            e_y = (e_line["bbox"][0][1] + e_line["bbox"][1][1]) / 2.0
            dist = abs(t_y - e_y)
            if dist < LINE_Y_TOLERANCE and dist < best_dist:
                best_dist = dist
                best_match = idx

        if best_match is not None:
            if t_line["confidence"] > merged[best_match]["confidence"]:
                merged[best_match] = t_line
        else:
            merged.append(t_line)

    return merged


def run_ocr(image: np.ndarray, bucket: str) -> list[dict]:
    """Run OCR on image and return line-grouped results."""
    easyocr_words = _run_easyocr(image)
    easyocr_lines = _group_into_lines(easyocr_words)

    # Secondary vote from pytesseract on clean bucket only
    if bucket == "clean" and _is_tesseract_available():
        try:
            tess_words = _run_pytesseract(image)
            tess_lines = _group_into_lines(tess_words)
            return _merge_lines(easyocr_lines, tess_lines)
        except Exception as exc:  # noqa: BLE001
            print(f"  [OCR] pytesseract failed, using EasyOCR only: {exc}")

    return easyocr_lines


def main() -> None:
    """Smoke test: run OCR on a few sample images."""
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    dataset_dir = project_root / "data" / "AI-OCR dataset"

    if not dataset_dir.is_dir():
        print(f"ERROR: Dataset not found: {dataset_dir}", file=sys.stderr)
        sys.exit(1)

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

    sys.path.insert(0, str(script_dir))
    from preprocess import preprocess

    print(f"Running OCR on {len(samples)} sample(s)...\n")
    for bucket, fname in sorted(samples.items()):
        img_path = dataset_dir / fname
        print(f"  {fname}  (bucket={bucket})")
        raw = cv2.imread(str(img_path))
        if raw is None:
            print(f"    SKIP — could not read {img_path}")
            continue

        processed = preprocess(raw, bucket)
        lines = run_ocr(processed, bucket)

        print(f"    Detected {len(lines)} line(s):")
        for line in lines[:10]:  # show first 10
            print(
                f"      [L{line['line_id']:2d}] "
                f"conf={line['confidence']:.2f}  "
                f"\"{line['text']}\""
            )
        if len(lines) > 10:
            print(f"      ... and {len(lines) - 10} more lines")
        print()


if __name__ == "__main__":
    main()
