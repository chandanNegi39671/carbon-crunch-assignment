"""
Module 1: Dataset Triage
Classifies every image into a quality bucket before OCR, so preprocessing
can be routed differently per bucket.
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

# ─── Thresholds ──────────────────────────────────────────────────────
BLUR_THRESHOLD = 100.0
SKEW_THRESHOLD = 5.0
LOW_LIGHT_THRESHOLD = 80.0

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


# ─── Image I/O ───────────────────────────────────────────────────────
def read_image_grayscale(path: Path) -> Optional[np.ndarray]:
    """Read an image as grayscale. Falls back to PIL if OpenCV fails."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is not None:
        return img
    try:
        pil_img = Image.open(path).convert("L")
        return np.array(pil_img)
    except Exception:
        return None


def get_image_files(dataset_dir: Path) -> list[Path]:
    """Return sorted list of image files in *dataset_dir*."""
    files = [
        p for p in sorted(dataset_dir.iterdir())
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return files


# ─── Quality metrics ─────────────────────────────────────────────────
def calculate_blur_score(path: Path) -> float:
    """Laplacian variance — higher means sharper. NaN if unreadable."""
    gray = read_image_grayscale(path)
    if gray is None or gray.size == 0:
        return float("nan")
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(laplacian.var())


def calculate_skew_angle(path: Path) -> float:
    """Median line angle from Hough lines. 0.0 if no lines found."""
    gray = read_image_grayscale(path)
    if gray is None or gray.size == 0:
        return float("nan")

    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=math.pi / 180,
        threshold=100,
        minLineLength=100,
        maxLineGap=10,
    )
    if lines is None:
        return 0.0

    angles: list[float] = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        if abs(angle) < 45:
            angles.append(angle)

    if not angles:
        return 0.0
    return float(np.median(angles))


def calculate_brightness(path: Path) -> tuple[float, float]:
    """(mean, std) of grayscale pixel intensities. NaN if unreadable."""
    gray = read_image_grayscale(path)
    if gray is None or gray.size == 0:
        return (float("nan"), float("nan"))
    return (float(np.mean(gray)), float(np.std(gray)))


# ─── Classification ──────────────────────────────────────────────────
def classify_image(
    blur_score: float,
    skew_angle: float,
    brightness_mean: float,
) -> str:
    """Map metrics to a quality bucket string."""
    if math.isnan(blur_score) or math.isnan(skew_angle) or math.isnan(brightness_mean):
        return "unknown"
    if blur_score < BLUR_THRESHOLD:
        return "blurry"
    if abs(skew_angle) > SKEW_THRESHOLD:
        return "skewed"
    if brightness_mean < LOW_LIGHT_THRESHOLD:
        return "low_light"
    return "clean"


# ─── Per-image processing ────────────────────────────────────────────
def process_image(path: Path) -> Optional[dict]:
    """Compute triage metrics for a single image. None if fully unreadable."""
    gray = read_image_grayscale(path)
    if gray is None or gray.size == 0:
        return None

    blur_score = calculate_blur_score(path)
    skew_angle = calculate_skew_angle(path)
    brightness_mean, brightness_std = calculate_brightness(path)
    bucket = classify_image(blur_score, skew_angle, brightness_mean)

    return {
        "filename": path.name,
        "blur_score": blur_score,
        "skew_angle": skew_angle,
        "brightness_mean": brightness_mean,
        "brightness_std": brightness_std,
        "bucket": bucket,
    }


# ─── Main ────────────────────────────────────────────────────────────
def main() -> None:
    # Resolve paths relative to this script's location
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    dataset_dir = project_root / "data" / "AI-OCR dataset"
    output_dir = project_root / "outputs" / "triage"
    output_csv = output_dir / "dataset_triage.csv"

    if not dataset_dir.is_dir():
        print(f"ERROR: Dataset directory not found: {dataset_dir}", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = get_image_files(dataset_dir)
    print(f"Found {len(image_files)} image(s) in {dataset_dir}")

    rows: list[dict] = []
    failed: list[str] = []

    for idx, path in enumerate(image_files, start=1):
        try:
            result = process_image(path)
            if result is None:
                failed.append(path.name)
                print(f"  [{idx}/{len(image_files)}] SKIP (unreadable): {path.name}")
            else:
                rows.append(result)
                print(
                    f"  [{idx}/{len(image_files)}] {result['bucket']:>10s}  "
                    f"blur={result['blur_score']:.1f}  "
                    f"skew={result['skew_angle']:.2f}  "
                    f"bright={result['brightness_mean']:.1f}  "
                    f"{path.name}"
                )
        except Exception as exc:  # noqa: BLE001
            failed.append(path.name)
            print(f"  [{idx}/{len(image_files)}] ERROR: {path.name} -- {exc}", file=sys.stderr)

    # ── Write CSV ────────────────────────────────────────────────────
    fieldnames = [
        "filename",
        "blur_score",
        "skew_angle",
        "brightness_mean",
        "brightness_std",
        "bucket",
    ]
    with open(output_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved triage results -> {output_csv}")

    # ── Summary stats ────────────────────────────────────────────────
    if not rows:
        print("No images were successfully processed.")
        return

    print(f"\n{'='*60}")
    print(f"TRIAGE SUMMARY  --  {len(rows)} processed, {len(failed)} failed")
    print(f"{'='*60}")

    for metric in ("blur_score", "skew_angle", "brightness_mean", "brightness_std"):
        vals = [r[metric] for r in rows if not math.isnan(r[metric])]
        if not vals:
            continue
        arr = np.array(vals)
        print(
            f"  {metric:>20s}  "
            f"min={arr.min():.2f}  max={arr.max():.2f}  "
            f"mean={arr.mean():.2f}  median={float(np.median(arr)):.2f}"
        )

    bucket_counts: dict[str, int] = {}
    for r in rows:
        bucket_counts[r["bucket"]] = bucket_counts.get(r["bucket"], 0) + 1

    print(f"\n  Bucket distribution:")
    for bucket, count in sorted(bucket_counts.items()):
        print(f"    {bucket:>12s}: {count:4d}  ({100*count/len(rows):.1f}%)")

    if failed:
        print(f"\n  Failed images ({len(failed)}): {', '.join(failed)}")


if __name__ == "__main__":
    main()
