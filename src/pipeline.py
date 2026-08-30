"""
Module 7: Pipeline Orchestrator
Single entry point that runs the full receipt processing pipeline:

  triage -> preprocess -> OCR -> extract fields -> confidence score -> summary

Run as: python src/pipeline.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2

# Ensure src/ is importable when running from project root
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from dataset_triage import get_image_files, process_image, SUPPORTED_EXTENSIONS
from preprocess import preprocess
from ocr_engine import run_ocr
from extract_fields import extract_fields
from confidence import score_receipt
from summary import generate_summary, save_summary


# ─── Per-image processing ────────────────────────────────────────────
def _process_one(
    img_path: Path,
    bucket: str,
) -> dict | None:
    """Full pipeline for a single image. Returns scored receipt dict or None."""
    raw = cv2.imread(str(img_path))
    if raw is None:
        return None

    processed = preprocess(raw, bucket)
    ocr_lines = run_ocr(processed, bucket)
    extracted = extract_fields(ocr_lines)
    scored = score_receipt(extracted, ocr_lines)
    return scored


# ─── Main ────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Receipt OCR pipeline")
    parser.add_argument(
        "--force", action="store_true",
        help="Ignore existing JSON outputs and reprocess every image from scratch",
    )
    args = parser.parse_args()
    force = args.force

    t_start = time.time()

    project_root = _SCRIPT_DIR.parent
    dataset_dir = project_root / "data" / "AI-OCR dataset"
    triage_csv = project_root / "outputs" / "triage" / "dataset_triage.csv"
    json_dir = project_root / "outputs" / "json"
    summary_path = project_root / "outputs" / "expense_summary.json"

    # ── Step 0: validate ─────────────────────────────────────────────
    if not dataset_dir.is_dir():
        print(f"ERROR: Dataset not found: {dataset_dir}", file=sys.stderr)
        sys.exit(1)

    json_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: triage ───────────────────────────────────────────────
    print("=" * 60)
    print("STEP 1: Dataset Triage")
    print("=" * 60)

    if triage_csv.exists():
        print(f"  Loading existing triage from {triage_csv}")
        buckets: dict[str, str] = {}
        with open(triage_csv, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                buckets[row["filename"]] = row["bucket"]
    else:
        print("  Running triage from scratch...")
        image_files = get_image_files(dataset_dir)
        buckets = {}
        triage_results: list[dict] = []  # store results for CSV writing
        for path in image_files:
            try:
                result = process_image(path)
                if result:
                    buckets[path.name] = result["bucket"]
                    triage_results.append(result)
            except Exception as exc:  # noqa: BLE001
                print(f"  WARN: triage failed for {path.name}: {exc}")
        # Save triage CSV using cached results (no recomputation)
        triage_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["filename", "blur_score", "skew_angle",
                       "brightness_mean", "brightness_std", "bucket"]
        with open(triage_csv, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for result in triage_results:
                writer.writerow(result)
        print(f"  Triage saved to {triage_csv}")

    print(f"  {len(buckets)} images bucketed\n")

    # ── Step 2-5: per-image pipeline ─────────────────────────────────
    print("=" * 60)
    print("STEP 2-5: Preprocess -> OCR -> Extract -> Score")
    print("=" * 60)

    image_files = get_image_files(dataset_dir)
    total = len(image_files)
    # Resume vs force mode
    existing_jsons = {p.stem for p in json_dir.glob("*.json")}
    if force:
        print("  Force mode: reprocessing all images (existing JSONs will be overwritten)")
        to_process = image_files
        skipped = 0
    else:
        to_process = [p for p in image_files if p.stem not in existing_jsons]
        skipped = total - len(to_process)
        if skipped > 0:
            print(f"  Resuming: {skipped} already processed, {len(to_process)} remaining")
    print()

    processed_count = skipped  # count existing as processed
    failed_count = 0
    failures: list[str] = []

    # Accumulators for stats
    conf_store: list[float] = []
    conf_date: list[float] = []
    conf_total: list[float] = []
    flagged_fields = 0

    # Load stats from existing JSONs — only in resume mode (not force).
    # In force mode, old JSONs are still on disk but will be overwritten
    # by Loop B, so loading them here would double-count stats.
    if not force:
        for jf in json_dir.glob("*.json"):
            try:
                with open(jf, encoding="utf-8") as fh:
                    data = json.load(fh)
                conf_store.append(data.get("store_name", {}).get("confidence", 0))
                conf_date.append(data.get("date", {}).get("confidence", 0))
                conf_total.append(data.get("total_amount", {}).get("confidence", 0))
                if data.get("store_name", {}).get("flag", False):
                    flagged_fields += 1
                if data.get("date", {}).get("flag", False):
                    flagged_fields += 1
                if data.get("total_amount", {}).get("flag", False):
                    flagged_fields += 1
                for it in data.get("items", []):
                    if it.get("confidence", 1.0) < 0.7:
                        flagged_fields += 1
            except Exception as exc:  # noqa: BLE001
                print(f"  WARN: failed to load {jf.name}: {exc}")

    for idx, img_path in enumerate(to_process, start=skipped + 1):
        fname = img_path.name
        bucket = buckets.get(fname, "clean")

        try:
            scored = _process_one(img_path, bucket)
            if scored is None:
                failed_count += 1
                failures.append(fname)
                print(f"  [{idx}/{total}] SKIP (unreadable): {fname}")
                continue

            # Save per-receipt JSON
            out_json = json_dir / f"{img_path.stem}.json"
            with open(out_json, "w", encoding="utf-8") as fh:
                json.dump(scored, fh, indent=2, ensure_ascii=False)

            processed_count += 1

            # Collect stats
            cs = scored["store_name"]["confidence"]
            cd = scored["date"]["confidence"]
            ct = scored["total_amount"]["confidence"]
            conf_store.append(cs)
            conf_date.append(cd)
            conf_total.append(ct)
            if scored["store_name"]["flag"]:
                flagged_fields += 1
            if scored["date"]["flag"]:
                flagged_fields += 1
            if scored["total_amount"]["flag"]:
                flagged_fields += 1
            for it in scored.get("items", []):
                if it.get("confidence", 1.0) < 0.7:
                    flagged_fields += 1

            n_items = len(scored.get("items", []))
            print(
                f"  [{idx}/{total}] {bucket:>10s}  "
                f"store={cs:.2f}  date={cd:.2f}  total={ct:.2f}  "
                f"items={n_items:2d}  {fname}"
            )

        except Exception as exc:  # noqa: BLE001
            failed_count += 1
            failures.append(fname)
            print(f"  [{idx}/{total}] ERROR: {fname} -- {exc}")

    # ── Step 6: summary ──────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("STEP 6: Financial Summary")
    print("=" * 60)

    summary = generate_summary(json_dir)
    save_summary(summary, summary_path)
    print(f"  Summary saved to {summary_path}")

    # ── Final stats ──────────────────────────────────────────────────
    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Total images     : {total}")
    print(f"  Processed        : {processed_count}")
    print(f"  Failed           : {failed_count}")
    print(f"  Time elapsed     : {elapsed:.1f}s ({elapsed/total:.1f}s per image)")

    if conf_store:
        import numpy as np
        print(f"\n  Average confidence per field type:")
        print(f"    store_name    : {np.mean(conf_store):.3f}")
        print(f"    date          : {np.mean(conf_date):.3f}")
        print(f"    total_amount  : {np.mean(conf_total):.3f}")
        print(f"    flagged fields: {flagged_fields}")

    print(f"\n  Financial summary:")
    print(f"    Total spend (all)         : {summary['total_spend_all']:.2f}")
    print(f"    Total spend (high-conf)   : {summary['total_spend_high_confidence']:.2f}")
    print(f"    Flagged receipts          : {summary['flagged_receipts_count']}")
    print(f"    Excluded outliers         : {summary['excluded_outlier_count']} (total={summary['excluded_outlier_total']:.2f})")
    print(f"    Unique stores             : {len(summary['spend_per_store'])}")

    if failures:
        print(f"\n  Failed images ({len(failures)}):")
        for f_name in failures[:20]:
            print(f"    - {f_name}")
        if len(failures) > 20:
            print(f"    ... and {len(failures) - 20} more")


if __name__ == "__main__":
    main()
