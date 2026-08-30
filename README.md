# Carbon Crunch — OCR-based Receipt Information Extraction

Extracts structured, confidence-scored information from 371 real-world receipt images and produces a financial summary.

## Setup

### Prerequisites

- Python 3.11
- `uv`
- Tesseract OCR

Tesseract must be installed separately and available on `PATH`.

Download: https://github.com/tesseract-ocr/tesseract

### Install dependencies

```bash
uv venv
uv pip install -r requirements.txt
````

On Windows, if you want to activate the virtual environment:

```bash
.venv\Scripts\activate
```

## Run

### Full pipeline (Modules 1–7)

```bash
uv run python src/pipeline.py
```

### Individual module

For example, to run dataset triage only:

```bash
uv run python src/dataset_triage.py
```

## Project Structure

```text
carbon-crunch-ocr/
├── data/
│   └── AI-OCR dataset/        # 371 receipt images (read-only)
├── src/
│   ├── dataset_triage.py      # Module 1 — quality triage
│   ├── preprocess.py          # Module 2 — image cleanup
│   ├── ocr_engine.py          # Module 3 — text detection/OCR
│   ├── extract_fields.py      # Module 4 — structured extraction
│   ├── confidence.py          # Module 5 — confidence scoring
│   ├── summary.py             # Module 6 — financial aggregation
│   └── pipeline.py            # Module 7 — end-to-end orchestrator
├── outputs/
│   ├── triage/
│   │   └── dataset_triage.csv
│   ├── json/
│   │   └── <receipt>.json
│   └── expense_summary.json
├── notebooks/
│   └── analysis.ipynb
├── docs/
│   └── documentation.pdf
├── README.md
└── requirements.txt
```

## Modules

| # | Module              | Purpose                                                                                     |
| - | ------------------- | ------------------------------------------------------------------------------------------- |
| 1 | `dataset_triage.py` | Classify images by quality: blurry / skewed / low_light / clean                             |
| 2 | `preprocess.py`     | Clean and normalize images per triage bucket before OCR                                     |
| 3 | `ocr_engine.py`     | Run EasyOCR, with Tesseract voting on clean images, and group words into lines              |
| 4 | `extract_fields.py` | Extract `store_name`, `date`, `items`, and `total_amount` using regex and column clustering |
| 5 | `confidence.py`     | Score each field from 0–1 using OCR confidence, pattern validity, and keyword anchoring     |
| 6 | `summary.py`        | Aggregate all receipts into a financial summary with fuzzy store-name matching              |
| 7 | `pipeline.py`       | Orchestrate Modules 1–6 end-to-end with resume and `--force` support                        |

## Dataset

* **371 receipt images (~143 MB)** with two main clusters:

  * ~349 SROIE-style clean scanned receipts
  * ~22 handheld phone photos with real-world noise including blur, skew, uneven lighting, and background clutter
* No ground-truth annotations were provided. Evaluation therefore relied on manual spot-checks and a targeted diagnostic script rather than automated precision/recall metrics.
* If the dataset is not committed to the repository (see `.gitignore`), place it at:

```text
data/AI-OCR dataset/
```

before running the pipeline.

## Results

*(As of the final pipeline run)*

* **371/371 images processed successfully**
* **0 failures**
* Average field confidence:

  * `store_name`: **0.66**
  * `date`: **0.65**
  * `total_amount`: **0.77**
* Financial summary:

  * Total spend across all receipts: **≈ RM 25,374**
  * High-confidence-only spend: **≈ RM 14,422**
  * Receipts flagged for review: **127**
  * Unique stores after fuzzy name matching: **281**
* See [`docs/documentation.pdf`](docs/documentation.pdf) for the full methodology, challenges, limitations, and future improvements.

## Output Format

Each receipt produces a JSON object similar to:

```json
{
  "store_name": {
    "value": "...",
    "confidence": 0.93,
    "flag": false,
    "reason": null
  },
  "date": {
    "value": "...",
    "confidence": 0.88,
    "flag": false,
    "reason": null
  },
  "items": [
    {
      "name": "...",
      "price": "...",
      "confidence": 0.81
    }
  ],
  "total_amount": {
    "value": "...",
    "confidence": 0.96,
    "flag": false,
    "reason": null
  }
}
```

Fields with confidence below **0.7** are flagged with a specific reason:

* `missing_field`
* `low_ocr_confidence`
* `unparsable_format`
* `no_keyword_anchor`

## Evaluation Notes

Because the provided dataset contains no ground-truth field annotations, automated precision/recall metrics cannot be computed reliably.

Evaluation therefore focuses on:

* OCR and extraction confidence
* Dataset-level diagnostic statistics
* Manual spot-checking
* Low-confidence field detection
* Robustness across clean, blurry, skewed, and low-light receipts
* Financial summary consistency
