# Carbon Crunch — OCR-based Receipt Information Extraction

Extracts structured, confidence-scored information from 371 real-world receipt images and produces a financial summary.

## Setup

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/Scripts/activate   # Windows (Git Bash)

# Install dependencies
pip install -r requirements.txt
```

> **Note:** Tesseract OCR must also be installed separately and available on `PATH`.
> Download: https://github.com/tesseract-ocr/tesseract

## Run

### Full pipeline (Modules 1-7)
```bash
python src/pipeline.py
```

### Individual module (e.g. triage only)
```bash
python src/dataset_triage.py
```

## Project Structure

```
carbon-crunch-mlops/
├── data/AI-OCR dataset/        # 371 receipt images (read-only)
├── src/
│   ├── dataset_triage.py       # Module 1 — quality triage
│   ├── preprocess.py           # Module 2 — image cleanup
│   ├── ocr_engine.py           # Module 3 — text detection/OCR
│   ├── extract_fields.py       # Module 4 — structured extraction
│   ├── confidence.py           # Module 5 — confidence scoring
│   ├── summary.py              # Module 6 — financial aggregation
│   └── pipeline.py             # Module 7 — end-to-end orchestrator
├── outputs/
│   ├── triage/dataset_triage.csv
│   ├── json/<receipt>.json
│   └── expense_summary.json
├── notebooks/analysis.ipynb
├── docs/documentation.pdf
└── requirements.txt
```

## Modules

| # | Module | Purpose |
|---|--------|---------|
| 1 | `dataset_triage.py` | Classify images by quality (blur, skew, lighting) |
| 2 | `preprocess.py` | Clean/normalise images per triage bucket |
| 3 | `ocr_engine.py` | EasyOCR + Tesseract fallback, line grouping |
| 4 | `extract_fields.py` | Regex + column-clustering field extraction |
| 5 | `confidence.py` | Weighted confidence scoring per field |
| 6 | `summary.py` | Aggregate receipts into expense summary |
| 7 | `pipeline.py` | End-to-end orchestrator |

## Dataset

- **371 images** (~143 MB), two visual clusters:
  - ~349 SROIE-style scanned receipts (clean, tabular)
  - ~22 handheld phone photos (variable quality)
- Resolution range: 433×543 to 2481×4032
