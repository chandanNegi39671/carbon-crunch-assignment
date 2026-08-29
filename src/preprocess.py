"""
Module 2: Preprocessing
Clean each image based on its triage bucket before OCR.
Each sub-step is its own testable function so individual steps can be
ablated/toggled independently.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Target long-edge dimension in pixels
TARGET_LONG_EDGE = 1600


# ─── Helper: read grayscale ──────────────────────────────────────────
def _read_gray(image: np.ndarray) -> np.ndarray:
    """Ensure *image* is single-channel grayscale."""
    if image is None or image.size == 0:
        raise ValueError("Empty image passed to preprocessor")
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image.copy()


# ─── Sub-step: resize ────────────────────────────────────────────────
def resize_long_edge(image: np.ndarray, target: int = TARGET_LONG_EDGE) -> np.ndarray:
    """Resize so the long edge equals *target* px, preserving aspect ratio."""
    h, w = image.shape[:2]
    long_edge = max(h, w)
    if long_edge == 0:
        return image
    scale = target / long_edge
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    return cv2.resize(image, (new_w, new_h), interpolation=interpolation)


# ─── Sub-step: denoise ──────────────────────────────────────────────
def denoise(image: np.ndarray, h: int = 10) -> np.ndarray:
    """Light non-local-means denoising."""
    return cv2.fastNlMeansDenoising(image, h=h)


# ─── Sub-step: deskew ───────────────────────────────────────────────
def deskew(image: np.ndarray) -> np.ndarray:
    """Rotate image to correct detected skew angle using border replication."""
    gray = _read_gray(image)
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
        return image

    angles: list[float] = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        if abs(angle) < 45:
            angles.append(angle)

    if not angles:
        return image

    median_angle = float(np.median(angles))
    if abs(median_angle) < 0.5:  # negligible — skip
        return image

    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    mat = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    return cv2.warpAffine(
        image, mat, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


# ─── Sub-step: contrast enhancement (CLAHE only) ────────────────────
def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """Apply CLAHE for local contrast enhancement.

    Returns CLAHE-enhanced grayscale (no binarization), since OCR engines
    like EasyOCR generally perform better on continuous-tone grayscale
    than hard-binarized images.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)


def binarize(image: np.ndarray) -> np.ndarray:
    """Adaptive thresholding for binarized output.

    Available for explicit use but not called automatically by the
    preprocessing pipeline.
    """
    return cv2.adaptiveThreshold(
        image, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,
        C=15,
    )


# ─── Sub-step: sharpen (unsharp mask) ───────────────────────────────
def sharpen(image: np.ndarray) -> np.ndarray:
    """Unsharp masking to partially recover edge sharpness.

    NOTE: this cannot fully undo blur, only mitigate it.
    """
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=3)
    # unsharp = original + (original - blurred) * amount
    sharpened = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
    return sharpened


# ─── Sub-step: perspective correction ────────────────────────────────
def correct_perspective(image: np.ndarray) -> np.ndarray:
    """Attempt to flatten a handheld photo by finding the largest
    4-point contour and applying a perspective warp.

    Falls back gracefully (returns original) if no reliable quadrilateral
    is found.
    """
    gray = _read_gray(image)
    h, w = gray.shape

    # Blur + Canny to find edges
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 200)

    # Dilate to close small gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image

    # Sort by area descending, try top candidates
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    image_area = h * w
    for contour in contours:
        area = cv2.contourArea(contour)
        # Require the contour to cover at least 15% of the image
        if area < 0.15 * image_area:
            break

        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype(np.float32)

            # Order points: top-left, top-right, bottom-right, bottom-left
            s = pts.sum(axis=1)
            d = np.diff(pts, axis=1).flatten()
            ordered = np.array([
                pts[np.argmin(s)],    # top-left
                pts[np.argmin(d)],    # top-right
                pts[np.argmax(s)],    # bottom-right
                pts[np.argmax(d)],    # bottom-left
            ], dtype=np.float32)

            # Target rectangle
            w_top = np.linalg.norm(ordered[1] - ordered[0])
            w_bot = np.linalg.norm(ordered[2] - ordered[3])
            max_w = int(max(w_top, w_bot))
            h_left = np.linalg.norm(ordered[3] - ordered[0])
            h_right = np.linalg.norm(ordered[2] - ordered[1])
            max_h = int(max(h_left, h_right))

            if max_w < 50 or max_h < 50:
                continue

            dst = np.array([
                [0, 0],
                [max_w - 1, 0],
                [max_w - 1, max_h - 1],
                [0, max_h - 1],
            ], dtype=np.float32)

            mat = cv2.getPerspectiveTransform(ordered, dst)
            warped = cv2.warpPerspective(image, mat, (max_w, max_h))
            return warped

    # No reliable 4-point contour found
    return image


# ─── Main preprocess entry point ─────────────────────────────────────
def preprocess(image: np.ndarray, bucket: str) -> np.ndarray:
    """Full preprocessing pipeline, routed by triage bucket.

    Parameters
    ----------
    image : np.ndarray
        Raw image (BGR or grayscale).
    bucket : str
        One of "clean", "blurry", "skewed", "low_light", "unknown".

    Returns
    -------
    np.ndarray
        Preprocessed image ready for OCR.
    """
    img = _read_gray(image)

    # 1. ALWAYS: resize long edge to TARGET_LONG_EDGE
    img = resize_long_edge(img)

    # 2. ALWAYS: light denoise
    img = denoise(img, h=10)

    # 3. Bucket-specific enhancements
    if bucket == "skewed":
        img = deskew(img)

    if bucket == "low_light":
        img = enhance_contrast(img)

    if bucket == "blurry":
        img = sharpen(img)

    # 4. Attempt perspective correction for handheld/photo-style images
    #    Detect via: aspect ratio far from typical scanned-receipt ratio
    #    (scanned receipts are usually ~0.4-0.7 width/height)
    img = _maybe_perspective_correct(img)

    return img


def _maybe_perspective_correct(image: np.ndarray) -> np.ndarray:
    """Heuristic: attempt perspective correction if the image looks like
    a handheld photo (aspect ratio outside typical scan range, or large
    non-white border region).

    This is intentionally conservative — skip gracefully if uncertain.
    """
    h, w = image.shape[:2]
    aspect = w / h if h > 0 else 1.0

    # Typical scanned receipts: aspect ~0.35-0.75 (portrait) or 1.3-2.8 (landscape)
    # Handheld photos tend to have wider aspect ratios and more background
    if 0.75 < aspect < 1.3:
        # Square-ish — likely a photo, attempt correction
        return correct_perspective(image)

    return image


# ─── Standalone test runner ──────────────────────────────────────────
def main() -> None:
    """Quick smoke test: preprocess a few sample images and save output."""
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    dataset_dir = project_root / "data" / "AI-OCR dataset"
    output_dir = project_root / "outputs" / "preprocess_test"

    if not dataset_dir.is_dir():
        print(f"ERROR: Dataset not found: {dataset_dir}", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load triage CSV to get bucket per image
    triage_csv = project_root / "outputs" / "triage" / "dataset_triage.csv"
    if not triage_csv.exists():
        print("ERROR: Run dataset_triage.py first to generate triage CSV.", file=sys.stderr)
        sys.exit(1)

    import csv
    buckets: dict[str, str] = {}
    with open(triage_csv, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            buckets[row["filename"]] = row["bucket"]

    # Pick one sample per bucket
    samples_by_bucket: dict[str, str] = {}
    for fname, bucket in buckets.items():
        if bucket not in samples_by_bucket:
            samples_by_bucket[bucket] = fname

    print(f"Testing preprocessing on {len(samples_by_bucket)} sample(s)...\n")

    for bucket, fname in sorted(samples_by_bucket.items()):
        img_path = dataset_dir / fname
        print(f"  {fname}  (bucket={bucket})")
        raw = cv2.imread(str(img_path))
        if raw is None:
            print(f"    SKIP — could not read {img_path}")
            continue

        processed = preprocess(raw, bucket)
        out_path = output_dir / f"preprocessed_{fname}"
        cv2.imwrite(str(out_path), processed)
        print(f"    raw={raw.shape} -> processed={processed.shape}  saved -> {out_path}")

    print(f"\nDone. Outputs in {output_dir}")


if __name__ == "__main__":
    main()
