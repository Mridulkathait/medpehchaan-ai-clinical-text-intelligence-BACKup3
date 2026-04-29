# MedPehchaan AI+ – Intelligent Clinical Text Intelligence System

MedPehchaan AI+ is a polished Streamlit app for **clinical text analysis**.  
It extracts key medical entities and presents safe, readable insights for demos, education, and research.

## Safety Notice

This project is an **educational/research prototype**.  
It does **not** provide medical diagnosis or treatment decisions.

## What This App Does

- Accepts typed clinical text
- Accepts uploaded `.txt`, `.pdf`, `.csv`, `.tsv`, `.xlsx`, and `.jsonl` files
- Lets user choose input mode:
  - typed text only
  - uploaded file only
  - combine both
- Cleans text with light preprocessing
- Runs biomedical/clinical token-classification NER
- Processes large tabular datasets in chunks to better handle 100k+ rows
- Extracts precise entities:
  - Disease
  - Symptom
  - Medication
  - Procedure
- Filters noisy spans and low-quality chunks
- Shows confidence and low-confidence flags
- Performs transparent risk classification
- Generates rule-based insights
- Generates concise entity-grounded summary
- Highlights entities in text with color coding

## Project Structure

```text
medpehchaan-ai-clinical-text-intelligence/
├── app.py
├── config.py
├── preprocessing.py
├── ner_engine.py
├── risk_engine.py
├── insight_engine.py
├── summary_engine.py
├── pdf_utils.py
├── text_utils.py
├── data/
│   └── sample_demo_input.txt
├── assets/
├── models/
├── requirements.txt
└── README.md
```

## NER Model Strategy

The app uses a **token-classification NER pipeline** from Hugging Face.

Fallback order:
1. `d4data/biomedical-ner-all` (primary biomedical NER model)
2. `samrawal/bert-base-uncased_clinical-ner` (fallback)
3. `emilyalsentzer/Bio_ClinicalBERT` (backbone fallback if available with token-classification head)

If a model candidate is unavailable, the app automatically tries the next candidate.

## Precision Improvements Implemented

- Label mapping into strict categories
- Subword aggregation (`aggregation_strategy="simple"`)
- Span cleaning and edge-stopword trimming
- Long phrase splitting and quality checks
- Per-label confidence thresholding
- Deduplication by label+entity
- Support dictionary layer for validation/completion (not primary extractor)
- Low-confidence flagging in UI

## Risk Rules (Explainable)

- **High**: chest pain, heart attack, stroke
- **Medium**: fever, infection
- **Low**: headache, cold

The app displays matched terms and explanation for the selected level.

## Setup

```bash
pip install -r requirements.txt
```

Optional for faster Hugging Face downloads and higher rate limits:

```bash
set HF_TOKEN=your_token_here
```

## Run

```bash
streamlit run app.py
```

## Demo Input

Use `data/sample_demo_input.txt` as a quick test sample.

## Notes

- This app is built to prefer **precision over recall**.
- If confidence is weak, entities are filtered or flagged instead of being forced.
