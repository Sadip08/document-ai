# Document AI

A visual document understanding pipeline that combines preprocessing, OCR, layout detection, reading-order reconstruction, table parsing, document classification, and rule-based information extraction — served behind a FastAPI endpoint.

Given a PDF or image, the pipeline predicts the document type (invoice, resume, paper, or form) and returns structured fields extracted from the OCR'd text.

> Built as part of the **Document AI Fellowship** roadmap (see `Document_AI_Fellowship_Complete_Development_Roadmap.pdf`), developed incrementally across 13 phases from preprocessing through deployment.

## Pipeline overview

```
PDF / Image
    ↓
Preprocessing        (grayscale → CLAHE → Otsu threshold)
    ↓
OCR                   (Tesseract: text, bounding boxes, confidence)
    ↓
Document Classification   (ResNet18, fine-tuned)
    ↓
Information Extraction    (regex rules, keyed by document type)
    ↓
{ document_type, extracted_fields, word_count }
```

A separate branch fuses OCR tokens with **YOLO-based layout detection** (11 region classes: Title, Text, Table, Picture, Caption, Footnote, Formula, List-item, Page-header/footer, Section-header) to reconstruct reading order and locate tables for structured parsing. This fusion logic lives in the notebooks and is not yet wired into the production API — see [Status](#status--known-gaps).

## Project structure

```
app/
  api.py                    FastAPI app: /health and /predict endpoints
src/
  preprocessing/pipeline.py Grayscale, denoise, CLAHE, Otsu, deskew, morphology
  ocr/ocr_engine.py         Tesseract wrapper → text + bboxes + confidence
  detection/train.py        YOLOv8 fine-tuning on the layout dataset
  classification/
    train_classifier.py     ResNet18 fine-tuning for document type classification
  extraction/
    extractor.py             Regex-based field extraction per document type
    table_parser.py          OpenCV/Tesseract-based table cell grouping → DataFrame
  orchestration/document_ai.py  Ties preprocessing → OCR → classification → extraction
  layout/, evaluation/, utils/  Reserved for future work (currently empty)
notebooks/                  Phase-by-phase experiments and write-ups (see below)
models/
  layout_detection_exp/     YOLO training run artifacts (weights, curves, metrics)
  doc_classifier_resnet18.pth  Trained classifier checkpoint
data/                       Raw PDFs, annotations, and train/valid splits (gitignored)
configs/, tests/            Currently empty — see Known Gaps
```

### Notebooks (development log)

| Notebook | Phase | Covers |
|---|---|---|
| `preprocessing.ipynb` | 1 | Grayscale, denoise, CLAHE, Otsu — visual comparison and observations |
| `ocr_baseline.ipynb` | 2 | Tesseract baseline, bounding-box visualization, failure modes |
| `ocr_layout_fusion.ipynb` | 5–6 | YOLO + OCR fusion via center-point containment; reading-order sort |
| `table_understanding.ipynb` | 7 | Table cropping, row/column grouping, cell OCR |
| `document_classification.ipynb` | 8 | ResNet18 fine-tuning experiments |
| `information_extraction.ipynb` | 9 | Classification + reading order → structured JSON |
| `robustness_experiments.ipynb` | 10 | Controlled blur/noise/contrast/rotation degradation vs. OCR/layout accuracy |
| `evaluation.ipynb` | 11 | CER / WER against a ground-truth benchmark |

## Setup

```bash
pip install -r requirements.txt
```

You'll also need [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed separately and available on your `PATH` (or update the path in `src/ocr/ocr_engine.py`).

## Running the API

```bash
uvicorn app.api:app --reload
```

- `GET /health` — reports whether the model pipeline loaded successfully.
- `POST /predict` — upload a PDF or image file; returns document type, extracted fields, and OCR word count.

## Model training

**Layout detector** (YOLOv8, fine-tuned on an 11-class document layout dataset — DocLayNet/RVL-CDIP derived, split via Roboflow):
```bash
python src/detection/train.py
```
Current result: **mAP50 ≈ 0.40** after 50 epochs (160 train / 40 valid images) — see `models/layout_detection_exp/results.csv`.

**Document classifier** (ResNet18, transfer learning, 4 classes: invoice/resume/paper/form, 50 images/class):
```bash
python src/classification/train_classifier.py
```
# Limitations & Future Improvements

This document tracks the current shortcomings of the Document AI pipeline and a prioritized set of improvements, based on a review of the codebase, training results, and notebooks as of the latest commit (`1c3e3e6`).

## Limitations

### Deployment & infrastructure
- **Hardcoded absolute Windows paths.** `app/api.py` (`YOLO_PATH`, `CLASSIFIER_PATH`) and `src/detection/train.py` (`yaml_path`) point at `E:\document-ai\...`. The API and training scripts will not run on any other machine or OS without manual edits.
- **CUDA is required unconditionally.** `Document_AI_Pipeline.__init__` and both training scripts call `torch.device("cuda")` directly. On a CPU-only machine, the pipeline fails to initialize (`/predict` then returns a 500 for every request) instead of degrading gracefully.
- **No containerization.** There's no Dockerfile, so reproducing the runtime environment (Tesseract binary, GPU drivers, Python deps) depends entirely on manual setup matching `requirements.txt`.
- **Tesseract path is hardcoded for Windows** in `src/ocr/ocr_engine.py` (`C:\Program Files\Tesseract-OCR\tesseract.exe`), left uncommented, which will break OCR on Linux/macOS unless edited.
- **CORS is wide open** (`allow_origins=["*"]`) with no auth on `/predict` — fine for local experimentation, not for anything public-facing.

### Pipeline completeness
- **Layout detection isn't in the API path.** The YOLO layout model, OCR/layout fusion (center-point containment), and reading-order reconstruction are demonstrated in `ocr_layout_fusion.ipynb` but `document_ai.py` never calls the YOLO model — `/predict` only does classification + regex extraction over raw OCR tokens.
- **Table parsing is disconnected too.** `table_parser.py` works on a manually cropped table image; there's no step in the pipeline that detects a table region (via YOLO) and automatically routes it to the parser.
- **Multi-page PDFs are ignored.** Both `api.py` and `pipeline.py`'s `render_pdf_page` only ever process page 0 (`doc[0]`). Any content beyond the first page is silently dropped.
- **No confidence/uncertainty in the output.** The API response doesn't surface OCR confidence, classification probability, or which fields failed to match — a caller can't distinguish "field not present" from "regex didn't match."

### Data & model quality
- **Small training sets.** Layout detection: 160 train / 40 valid images across 11 classes. Classification: 200 images total (50/class, 4 classes), with an **empty validation split** (`data/splits/doc_classification/valid` has no images) — so classifier accuracy has never actually been validated on held-out data.
- **Layout detector mAP50 ≈ 0.40** after 50 epochs (see `models/layout_detection_exp/results.csv`) — usable for demos, not reliable for production-grade region detection, especially for underrepresented classes (Formula, Footnote).
- **Two YOLO base checkpoints committed** (`yolov8n.pt` and `yolo26n.pt`), unclear which is canonical; only `yolov8n.pt` is referenced in `train.py`.
- **No cross-validation or held-out test set report** for the classifier — training accuracy is logged per epoch but there's no separate test evaluation in the repo.

### Extraction logic
- **Regex-only, narrow field coverage.** Invoice: number, date, total. Resume: email, phone. Paper: arXiv ID. Form: no fields defined at all (falls through to just `document_type`).
- **Brittle patterns.** E.g., the invoice-number regex assumes an `INV-`/`Invoice #` prefix; totals assume a `$`-prefixed decimal — real-world invoices in other currencies/formats or with OCR noise (misread characters) will silently fail to match with no fallback or fuzzy matching.
- **No use of spatial/layout information in extraction** — extraction runs over a flattened, unordered `" ".join(full_text)` string, not the reading-order-sorted, layout-aware tokens the fusion notebook already produces. Two nearby-but-unrelated numbers could be concatenated into one match.

### Testing & tooling
- **`configs/` and `tests/` are empty.** No automated tests for preprocessing, OCR, extraction, or the API endpoints; no config files for hyperparameters, class lists, or thresholds (values like `gap_threshold=50` in `table_parser.py` are hardcoded magic numbers).
- **No CI.** No linting, type-checking, or test workflow configured in the repo.
- **Notebooks aren't version-pinned or parameterized** — re-running them requires knowing which local paths/data were used at the time.

## Future Improvements

### Near-term (make it actually work end-to-end)
1. Replace hardcoded paths with environment variables or a `configs/settings.yaml`, resolved relative to the repo root.
2. Add a device-selection helper (`"cuda" if torch.cuda.is_available() else "cpu"`) used everywhere a device is set.
3. Wire the YOLO layout model into `Document_AI_Pipeline`: detect regions → associate OCR tokens per region → reading-order sort → pass structured, ordered text into `extract_information` instead of a flat token list.
4. Auto-route detected `Table` regions into `table_parser.py` rather than requiring a manual crop.
5. Support multi-page PDFs — process all pages (or accept a page range) instead of only page 0.
6. Populate `data/splits/doc_classification/valid` and report real held-out accuracy/F1 per class.

### Medium-term (quality & robustness)
7. Expand the classification label set beyond invoice/resume/paper/form (e.g., contracts, receipts, letters) and grow the dataset per class well past 50 images.
8. Replace/augment regex extraction with a more robust approach — key-value pair detection via spatial proximity, or a lightweight NER/LayoutLM-style model — while keeping regex as a fast fallback.
9. Add fuzzy matching or OCR-error-tolerant patterns (e.g., allow `l`/`1`/`I` confusion) for extraction fields.
10. Surface confidence scores in the API response for OCR, classification, and each extracted field so downstream consumers can flag low-confidence results.
11. Apply the mitigations already identified in `ocr_baseline.ipynb` (morphological opening, deskewing, adaptive thresholding, connected-component filtering) to the production preprocessing path — currently only grayscale → CLAHE → Otsu is used in `document_ai.py`, skipping denoise/deskew/morphology that exist in `pipeline.py` but aren't called.
12. Incorporate findings from `robustness_experiments.ipynb` into automated regression tests (e.g., assert OCR/CER stays within bounds under known degradations).

### Longer-term (production readiness)
13. Containerize the service (Dockerfile + docker-compose) with Tesseract and model weights baked in or fetched at startup.
14. Add authentication/rate-limiting and restrict CORS before any public deployment.
15. Build a proper test suite: unit tests for preprocessing/extraction functions, integration tests for `/predict` with fixture documents, and a CI workflow to run them on every PR.
16. Track experiments (data version, hyperparameters, metrics) with a lightweight tool (e.g., MLflow or DVC) instead of ad hoc notebook runs, to make results reproducible.
17. Add batch/async processing for large documents or high request volume, rather than synchronous single-page inference per request.
18. Consider replacing Tesseract with a more modern OCR engine (e.g., PaddleOCR, TrOCR) for a robustness/accuracy comparison, especially for degraded or handwritten documents.