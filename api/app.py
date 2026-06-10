import json

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

from src.features import FeatureBuilder
from src.utils import MODELS_DIR
from src.preprocess import clean_text

app = FastAPI(
    title="10-K Risk Classification API",
    version="1.0.0",
    description="Predict financial risk level from SEC 10-K filing text",
)


class PredictionRequest(BaseModel):

    text: str
    risk_section:       Optional[str] = None  
    mda_section:        Optional[str] = None   
    business_section:   Optional[str] = None   
    financials_section: Optional[str] = None  


with open(MODELS_DIR / "best_model.json") as f:
    best_model_data = json.load(f)

BEST_MODEL_NAME = best_model_data["best_model"]

model         = joblib.load(MODELS_DIR / f"{BEST_MODEL_NAME}_model.joblib")
feature_builder = FeatureBuilder.load()
label_encoder = joblib.load(MODELS_DIR / "label_encoder.joblib")

def _prepare_features(req: PredictionRequest) -> pd.DataFrame:

    structured = all([
        req.risk_section,
        req.mda_section,
        req.business_section,
        req.financials_section,
    ])

    if structured:
        clean_risk       = clean_text(req.risk_section)
        clean_mda        = clean_text(req.mda_section)
        clean_business   = clean_text(req.business_section)
        clean_financials = clean_text(req.financials_section)
        combined = " ".join([clean_risk, clean_mda, clean_business, clean_financials]).strip()
    else:
        combined         = clean_text(req.text)
  
        words     = combined.split()
        n         = len(words)
        half      = n // 2
        quarter   = n // 4

        clean_risk       = " ".join(words[:half])
        clean_mda        = " ".join(words[half:])
        clean_business   = " ".join(words[:quarter])
        clean_financials = " ".join(words[quarter:half])

    return pd.DataFrame([{
        "text":             combined,
        "clean_risk":       clean_risk,
        "clean_mda":        clean_mda,
        "clean_business":   clean_business,
        "clean_financials": clean_financials,
    }]), structured

@app.get("/")
def home():
    return {
        "message"   : "10-K Risk Classification API",
        "best_model": BEST_MODEL_NAME,
        "endpoints" : {
            "GET  /":        "This message",
            "POST /predict": "Predict risk label from filing text",
        },
    }


@app.post("/predict")
def predict(req: PredictionRequest):
    text = req.text.strip()

    if not text:
        return {"error": "Input text cannot be empty"}

    structured = all([req.risk_section, req.mda_section, req.business_section, req.financials_section])
    check_text = (req.risk_section if structured else text)
    if len(check_text.split()) < 10:
        return {"error": "Text too short for reliable prediction (minimum ~10 words)."}
    df, structured_mode = _prepare_features(req)

    X = feature_builder.transform(df)

    if BEST_MODEL_NAME == "adaboost":
        X = X.toarray()
    elif BEST_MODEL_NAME == "catboost":
        X = X.astype(np.float32).toarray()

    prediction = model.predict(X)
    if hasattr(prediction, "flatten"):
        prediction = prediction.flatten()

    label = label_encoder.inverse_transform(prediction.astype(int))[0]

    confidence = None
    all_probs  = None
    if hasattr(model, "predict_proba"):
        probs      = model.predict_proba(X)[0]
        confidence = float(np.max(probs))
        all_probs  = {
            cls: round(float(p), 4)
            for cls, p in zip(label_encoder.classes_, probs)
        }

    return {
        "label"         : label,
        "confidence"    : round(confidence, 4) if confidence is not None else None,
        "all_probs"     : all_probs,
        "mode"          : "structured" if structured_mode else "unstructured",
    }

@app.get("/health")
def health():
    return {"status": "ok", "model": BEST_MODEL_NAME}