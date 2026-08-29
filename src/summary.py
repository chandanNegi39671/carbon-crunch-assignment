"""
Module 6: Financial Summary
Aggregate all per-receipt JSONs into one financial summary.

Features:
- Total spend (all vs high-confidence)
- Per-store breakdown with fuzzy name matching (threshold ~0.85)
- Flagged receipt count
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

# ─── Fuzzy matching threshold ────────────────────────────────────────
_SIMILARITY_THRESHOLD = 0.85


# ─── Helpers ─────────────────────────────────────────────────────────
def _normalise_name(name: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    name = name.lower().strip()
    name = re.sub(r"[^\w\s]", " ", name)
    name = re.sub(r"\s+", " ", name)
    return name.strip()


def _fuzzy_match(a: str, b: str) -> bool:
    """Return True if *a* and *b* are similar above the threshold."""
    na, nb = _normalise_name(a), _normalise_name(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    return SequenceMatcher(None, na, nb).ratio() >= _SIMILARITY_THRESHOLD


def _merge_store_name(existing: str, new: str) -> str:
    """Keep the longer (usually more complete) name when merging."""
    if len(new) > len(existing):
        return new
    return existing


# ─── Public API ──────────────────────────────────────────────────────
def generate_summary(json_dir: Path) -> dict:
    """Load every per-receipt JSON from *json_dir* and produce a summary.

    Parameters
    ----------
    json_dir : Path
        Directory containing ``<filename>.json`` files, each with the
        schema from ``confidence.score_receipt()``.

    Returns
    -------
    dict
        Summary with keys: ``total_spend_all``, ``total_spend_high_confidence``,
        ``transaction_count``, ``spend_per_store``, ``flagged_receipts_count``,
        ``receipts``.
    """
    json_files = sorted(json_dir.glob("*.json"))
    if not json_files:
        return {
            "total_spend_all": 0.0,
            "total_spend_high_confidence": 0.0,
            "transaction_count": 0,
            "spend_per_store": {},
            "flagged_receipts_count": 0,
            "receipts": [],
        }

    total_all = 0.0
    total_high_conf = 0.0
    flagged_count = 0
    receipts: list[dict] = []

    # Per-store accumulators: canonical_name -> {"total": float, "count": int, "original_names": set}
    store_data: dict[str, dict] = {}

    for jf in json_files:
        try:
            with open(jf, encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:  # noqa: BLE001
            print(f"  WARN: failed to load {jf.name}: {exc}")
            continue

        # Extract total amount
        total_field = data.get("total_amount", {})
        total_val_str = total_field.get("value", "")
        total_conf = total_field.get("confidence", 0.0)
        total_flag = total_field.get("flag", True)

        amount = 0.0
        try:
            cleaned = re.sub(r"^(RM|Rs\.?|\$)\s*|[,\s]", "", total_val_str, flags=re.IGNORECASE)
            amount = float(cleaned)
        except (ValueError, TypeError) as exc:  # noqa: BLE001
            print(f"  WARN: failed to parse amount from {jf.name}: {exc}")

        total_all += amount
        if total_conf >= 0.7:
            total_high_conf += amount
        if total_flag:
            flagged_count += 1

        # Store name
        store_name = data.get("store_name", {}).get("value", "UNKNOWN")

        # Fuzzy group into canonical store
        canonical = None
        for existing_key in store_data:
            if _fuzzy_match(existing_key, store_name):
                canonical = existing_key
                break

        if canonical is None:
            canonical = store_name
            store_data[canonical] = {"total": 0.0, "count": 0, "names": set()}
        store_data[canonical]["total"] += amount
        store_data[canonical]["count"] += 1
        store_data[canonical]["names"].add(store_name)

        # Keep the best (longest) name as display name
        store_data[canonical]["display"] = _merge_store_name(
            store_data[canonical].get("display", canonical), store_name
        )

        receipts.append({
            "file": jf.stem,
            "store_name": store_name,
            "date": data.get("date", {}).get("value", ""),
            "total": amount,
            "total_confidence": total_conf,
            "total_flagged": total_flag,
        })

    # Build spend_per_store output
    spend_per_store: dict[str, dict] = {}
    for canonical, info in store_data.items():
        display = info.get("display", canonical)
        spend_per_store[display] = {
            "total_spend": round(info["total"], 2),
            "transaction_count": info["count"],
            "known_variants": sorted(info["names"]),
        }

    return {
        "total_spend_all": round(total_all, 2),
        "total_spend_high_confidence": round(total_high_conf, 2),
        "transaction_count": len(json_files),
        "spend_per_store": spend_per_store,
        "flagged_receipts_count": flagged_count,
        "receipts": receipts,
    }


def save_summary(summary: dict, output_path: Path) -> None:
    """Write summary dict to *output_path* as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)


# ─── Standalone test runner ──────────────────────────────────────────
def main() -> None:
    """Smoke test: aggregate whatever JSONs exist in outputs/json/."""
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    json_dir = project_root / "outputs" / "json"
    output_path = project_root / "outputs" / "expense_summary.json"

    if not json_dir.exists() or not list(json_dir.glob("*.json")):
        print(f"No JSON files found in {json_dir}.")
        print("Run pipeline.py first to generate per-receipt JSONs.")
        sys.exit(1)

    summary = generate_summary(json_dir)
    save_summary(summary, output_path)

    print(f"Summary saved to {output_path}\n")
    print(f"  Transactions    : {summary['transaction_count']}")
    print(f"  Total spend     : {summary['total_spend_all']:.2f}")
    print(f"  High-conf spend : {summary['total_spend_high_confidence']:.2f}")
    print(f"  Flagged receipts: {summary['flagged_receipts_count']}")
    print(f"\n  Per-store breakdown:")
    for store, info in sorted(summary["spend_per_store"].items()):
        print(
            f"    {store:40s}  "
            f"total={info['total_spend']:>10.2f}  "
            f"txns={info['transaction_count']:3d}"
        )


if __name__ == "__main__":
    main()
