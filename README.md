# SEC 10-K Financial Risk Classification

## Overview

This project implements an end-to-end NLP pipeline for financial risk classification using SEC 10-K filings.

The system:

- Loads SEC filings from Hugging Face
- Extracts key financial sections
- Cleans and preprocesses text
- Generates TF-IDF and engineered features
- Trains XGBoost, AdaBoost, and CatBoost models
- Evaluates performance using multiple metrics
- Deploys the best model through FastAPI

Dataset:

https://huggingface.co/datasets/winterForestStump/10-K_sec_filings

---

## Problem Statement

Financial reports contain large amounts of information regarding a company's operations, financial condition, and potential risks.

The objective of this project is to classify SEC 10-K filings into:

- High Risk
- Medium Risk
- Low Risk

using information extracted from annual reports.

---

## Project Structure

project/

├── api/

│ └── app.py

├── src/

│ ├── preprocess.py

│ ├── features.py

│ ├── train.py

│ ├── evaluate.py

│ └── utils.py

├── models/

│ ├── xgboost_model.joblib

│ ├── adaboost_model.joblib

│ ├── catboost_model.joblib

│ ├── feature_builder.joblib

│ ├── label_encoder.joblib

│ ├── evaluation_results.json

│ └── best_model.json

├── README.md

├── MODEL_REPORT.md

├── requirements.txt

└── .gitignore

---

## Pipeline

### Stage 1 – Data Extraction & Preprocessing

- Stream SEC filings from Hugging Face
- Extract:
  - Risk Factors
  - Business Overview
  - Management Discussion & Analysis (MD&A)
  - Financial Statements

Cleaning includes:

- HTML removal
- URL removal
- SEC boilerplate removal
- Date removal
- Number removal
- Lowercasing
- Whitespace normalization

---

### Stage 2 – Feature Engineering

TF-IDF Features

| Section | Features |
|----------|----------:|
| Risk Factors | 7000 |
| MD&A | 5000 |
| Business | 1500 |
| Financial Statements | 300 |

Additional Features

- Document length
- Risk keyword density
- Low-risk keyword density
- Risk ratio
- Negation density
- Average word length
- Unique token ratio
- Section lengths

Total feature space: ~13,800 features

---

### Stage 3 – Model Training

Models trained:

1. XGBoost
2. AdaBoost
3. CatBoost

---

### Stage 4 – Evaluation

| Model | Accuracy | Precision | Recall | F1 Score |
|---------|---------:|---------:|---------:|---------:|
| XGBoost | 72.71% | 72.30% | 72.71% | 72.43% |
| AdaBoost | 64.79% | 68.87% | 64.79% | 65.58% |
| CatBoost | 72.71% | 72.78% | 72.71% | 72.74% |

Best Model: CatBoost

---

### Stage 5 – API Deployment

FastAPI endpoint:

POST /predict

Example Request

```json
{
  "text": "combined filing text",
  "risk_section": "...",
  "mda_section": "...",
  "business_section": "...",
  "financials_section": "..."
}
```

Example Response

```json
{
  "label": "high_risk",
  "confidence": 0.81,
  "all_probs": {
    "high_risk": 0.81,
    "medium_risk": 0.14,
    "low_risk": 0.05
  }
}
```

Swagger UI:

/docs

---

## Installation

pip install -r requirements.txt

---

## Running the Project

### Preprocess Data

python -m src.preprocess

### Train Models

python -m src.train

### Evaluate Models

python -m src.evaluate

### Run API

uvicorn api.app:app --reload

---

## Saved Models

- xgboost_model.joblib
- adaboost_model.joblib
- catboost_model.joblib
- feature_builder.joblib
- label_encoder.joblib

These artifacts are used directly during inference.

---

## Results

CatBoost achieved the best overall performance with a weighted F1 score of 72.74% and was selected as the final deployment model.

---

## Author

Hari Teja Meesala

B.Tech CSE

National Institute of Technology Tiruchirappalli