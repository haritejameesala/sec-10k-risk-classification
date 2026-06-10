# Model Report

## 1. Problem Statement

The objective of this project is to build a financial document intelligence system capable of classifying SEC 10-K filings into:

- High Risk
- Medium Risk
- Low Risk

using textual information extracted from annual financial reports.

---

## 2. Dataset

Source:

winterForestStump/10-K_sec_filings

The dataset contains annual SEC filings submitted to the U.S. Securities and Exchange Commission.

Key sections extracted:

- Risk Factors
- Management Discussion & Analysis (MD&A)
- Business Overview
- Financial Statements

A filtered dataset of 2400 filings was used for training and evaluation.

---

## 3. Label Creation Strategy

The dataset does not contain explicit risk labels.

Therefore a custom risk-scoring framework was developed.

### Risk Score Components

Risk Score =
0.4 × Keyword Score
+
0.6 × Sentiment Score

### Keyword Score

High-risk keywords:

- bankruptcy
- litigation
- fraud
- default
- impairment
- investigation
- restructuring

Low-risk keywords:

- strong growth
- profitability
- positive cash flow
- market leader
- strong balance sheet

Negation handling was implemented.

Examples:

- "no material weakness" → positive signal
- "not profitable" → negative signal

### Sentiment Score

Financial sentiment was calculated using the Loughran-McDonald financial dictionary through PySentiment2.

### Label Assignment

Risk scores were divided using quantiles:

- Bottom 33% → Low Risk
- Middle 33% → Medium Risk
- Top 33% → High Risk

This produced a balanced three-class classification problem.

---

## 4. Preprocessing

The preprocessing pipeline included:

- HTML removal
- URL removal
- SEC boilerplate removal
- Date removal
- Monetary value removal
- Number removal
- Lowercasing
- Whitespace normalization

Sections shorter than the minimum threshold were discarded to improve label quality.

---

## 5. Feature Engineering

### Section-Specific TF-IDF

| Section | Features |
|----------|----------:|
| Risk Factors | 7000 |
| MD&A | 5000 |
| Business | 1500 |
| Financial Statements | 300 |

### Engineered Features

Additional handcrafted features:

- Document length
- Risk keyword density
- Low-risk keyword density
- Risk ratio
- Negation density
- Average word length
- Unique token ratio
- Sentence count
- Section lengths

The final feature space combines sparse TF-IDF vectors with dense numerical features.

---

## 6. Models Trained

### XGBoost

Gradient boosting model with regularization designed for high-dimensional sparse feature spaces such as TF-IDF vectors.

### AdaBoost

Boosting model using shallow decision trees.

### CatBoost

Gradient boosting model optimized for robust generalization and class balance.

---

## 7. Model Comparison

| Model | Accuracy | Precision | Recall | F1 Score |
|---------|---------:|---------:|---------:|---------:|
| XGBoost | 72.71% | 72.30% | 72.71% | 72.43% |
| AdaBoost | 64.79% | 68.87% | 64.79% | 65.58% |
| CatBoost | 72.71% | 72.78% | 72.71% | 72.74% |

---

## 8. Confusion Matrix Analysis

### XGBoost

| Actual / Predicted | High | Medium | Low |
|------|------:|------:|------:|
| High | 126 | 8 | 29 |
| Medium | 3 | 132 | 24 |
| Low | 37 | 30 | 91 |

Observations:

- Strong Medium Risk detection.
- Low Risk documents are sometimes confused with High Risk.
- High and Medium classes are separated effectively.

---

### AdaBoost

| Actual / Predicted | High | Medium | Low |
|------|------:|------:|------:|
| High | 96 | 6 | 61 |
| Medium | 1 | 105 | 53 |
| Low | 24 | 24 | 110 |

Observations:

- Significant confusion between High Risk and Low Risk.
- Lower overall class separation capability.
- Weakest performance among all models.

---

### CatBoost

| Actual / Predicted | High | Medium | Low |
|------|------:|------:|------:|
| High | 124 | 8 | 31 |
| Medium | 3 | 126 | 30 |
| Low | 31 | 28 | 99 |

Observations:

- Most balanced confusion matrix.
- Better Low Risk handling than XGBoost.
- More consistent performance across all classes.

---

## 9. Class-Level Performance (CatBoost)

| Class | Precision | Recall | F1 |
|---------|---------:|---------:|---------:|
| High Risk | 78.48% | 76.07% | 77.26% |
| Medium Risk | 77.78% | 79.25% | 78.50% |
| Low Risk | 61.88% | 62.66% | 62.26% |

Medium Risk was the easiest class to identify.

Low Risk remained the most challenging due to overlap in language patterns with Medium Risk filings.

---

## 10. Best Model Selection

Selected Model: CatBoost

Reasons:

1. Highest weighted F1 Score (72.74%).
2. Best balance between precision and recall.
3. Most stable class-level performance.
4. Lower misclassification rate across classes.
5. Strong generalization on unseen filings.

CatBoost was automatically selected as the final deployment model.

---

## 11. Model Serialization

All trained models were serialized using Joblib.

Saved artifacts:

- xgboost_model.joblib
- adaboost_model.joblib
- catboost_model.joblib
- feature_builder.joblib
- label_encoder.joblib

These artifacts are used directly by the FastAPI inference pipeline.

---

## 12. Conclusion

An end-to-end NLP pipeline was developed for financial risk classification of SEC 10-K filings.

The system successfully performs:

- Document preprocessing
- Section extraction
- Feature engineering
- Multi-model training
- Performance evaluation
- FastAPI deployment

Among the evaluated models, CatBoost achieved the best overall performance and was selected for deployment.