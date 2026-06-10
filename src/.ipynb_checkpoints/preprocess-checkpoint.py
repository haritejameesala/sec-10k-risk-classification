import os
import re

import pandas as pd
import pysentiment2 as ps
from datasets import load_dataset
from scipy.stats import zscore

from .utils import DATA_DIR, get_logger, save_json, Timer

logger = get_logger("preprocess")

lm = ps.LM()

MAX_ROWS = 2400

MIN_SECTION_CHARS = 1000

SECTION_COLS = {
    "risk"      : "Risk Factors",
    "mda"       : "Management’s Discussion and Analysis of Financial Condition and Results of Operations",
    "business"  : "Business",
    "financials": "Financial Statements and Supplementary Data",
}

HIGH_RISK_KEYWORDS = [
    "bankruptcy",
    "insolvency",
    "default",
    "litigation",
    "lawsuit",
    "investigation",
    "regulatory",
    "penalty",
    "fraud",
    "material weakness",
    "restatement",
    "going concern",
    "impairment",
    "restructuring",
    "cybersecurity",
    "data breach",
    "credit risk",
    "liquidity risk",
    "market risk",
]

LOW_RISK_KEYWORDS = [
    "strong growth",
    "growth",
    "revenue growth",
    "revenue increase",
    "increased revenue",
    "record revenue",
    "record sales",
    "profitable",
    "profitability",
    "earnings growth",
    "positive cash flow",
    "strong cash flow",
    "cash generation",
    "strong liquidity",
    "strong balance sheet",
    "healthy balance sheet",
    "solid financial position",
    "financial strength",
    "financial flexibility",
    "reduced debt",
    "low leverage",
    "investment grade",
    "market leader",
    "competitive advantage",
    "strong market position",
    "strong demand",
    "strong customer demand",
    "diversified business",
    "diversified revenue",
    "recurring revenue",
    "customer retention",
    "effective internal controls",
    "successful execution",
    "business momentum",
    "strong operating results",
    "improved margins",
    "margin improvement",
    "strong performance",
    "no material weakness",
    "no material litigation",
]

NEGATIONS = {"no", "not", "never", "without", "none", "cannot" , "neither" ,"nor"}

def load_latest_splits(max_rows: int = MAX_ROWS) -> pd.DataFrame:
    """
    Stream rows from HuggingFace and apply the section-length filter inline.
    Stops only once max_rows GOOD rows have been collected, so the dataset
    size after filtering is guaranteed to be max_rows (or as close as the
    corpus allows).

    Filtering here — rather than after loading — avoids the situation where
    3 000 raw rows collapse to ~300 usable rows because ~85 % of records have
    stub / missing sections.
    """
    logger.info(
        f"Streaming HuggingFace dataset "
        f"(target: {max_rows} good rows, min section length: {MIN_SECTION_CHARS} chars)..."
    )

    KEEP_COLS = list(SECTION_COLS.values()) + ["company_name"]

    with Timer() as t:

        meta = load_dataset("winterForestStump/10-K_sec_filings")

    all_splits = sorted(meta.keys(), key=lambda x: int(x), reverse=True)
    logger.info(f"Discovered {len(all_splits)} splits. Streaming newest-first...")

    RISK_COL = SECTION_COLS["risk"]
    MDA_COL  = SECTION_COLS["mda"]

    rows = []
    skipped = 0

    for split_name in all_splits:
        if len(rows) >= max_rows:
            break

        ds_split = load_dataset(
            "winterForestStump/10-K_sec_filings",
            split=split_name,
        )

        for record in ds_split:
            risk_len = len(record.get(RISK_COL, "") or "")
            mda_len  = len(record.get(MDA_COL,  "") or "")

            if risk_len < MIN_SECTION_CHARS or mda_len < MIN_SECTION_CHARS:
                skipped += 1
                continue

            row = {col: record.get(col, "") for col in KEEP_COLS if col in record}
            rows.append(row)

            if len(rows) >= max_rows:
                break

        logger.info(
            f"  Split {split_name} done — "
            f"{len(rows)} good rows | {skipped} stubs skipped so far"
        )

    df = pd.DataFrame(rows)
    logger.info(
        f"Loaded {len(df)} good rows "
        f"({skipped} stub rows skipped) in {t}"
    )
    return df

_HTML        = re.compile(r"<[^>]+>")
_URL         = re.compile(r"https?://\S+|www\.\S+")
_ITEM_HEADER = re.compile(r"\bitem\s+\d+[a-z]?\b\.?",      re.IGNORECASE)
_EXHIBIT     = re.compile(r"\bexhibit\s+\d+[\.\d]*\b",      re.IGNORECASE)
_BOILERPLATE = re.compile(
    r"(table of contents|annual report on form 10-k|"
    r"securities and exchange commission|"
    r"united states securities|washington,?\s*d\.?c\.?|"
    r"incorporated by reference|see note \d+|f-\d+|page \d+)",
    re.IGNORECASE,
)
_MONEY     = re.compile(r"\$[\d,\.]+\s*(million|billion|thousand)?", re.IGNORECASE)
_PERCENT   = re.compile(r"\d+\.?\d*\s*%")
_DATE      = re.compile(
    r"\b(january|february|march|april|may|june|july|august|"
    r"september|october|november|december)\s+\d{1,2},?\s+\d{4}\b",
    re.IGNORECASE,
)
_NUMBER    = re.compile(r"\b\d[\d,\.]*\b")
_NON_ALPHA = re.compile(r"[^a-z\s]")
_WHITESPACE = re.compile(r"\s+")

MAX_SECTION_CHARS = 30_000

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    if len(text) > MAX_SECTION_CHARS:
        text = text[:MAX_SECTION_CHARS]

    if text.strip().lower() in ("nan", "none", ""):
        return ""

    t = text
    t = _HTML.sub(" ", t)
    t = _URL.sub(" ", t)
    t = _BOILERPLATE.sub(" ", t)
    t = _ITEM_HEADER.sub(" ", t)
    t = _EXHIBIT.sub(" ", t)
    t = _MONEY.sub(" ", t)
    t = _PERCENT.sub(" ", t)
    t = _DATE.sub(" ", t)
    t = _NUMBER.sub(" ", t)
    t = t.lower()
    t = _NON_ALPHA.sub(" ", t)
    t = _WHITESPACE.sub(" ", t).strip()
    return t

def keyword_count_with_negation(tokens, keyword, window=3):

    kw_tokens = keyword.split()
    kw_len = len(kw_tokens)

    high_count = 0
    low_count  = 0

    for i in range(len(tokens) - kw_len + 1):
        if tokens[i : i + kw_len] == kw_tokens:
            prev_words = tokens[max(0, i - window) : i]
            negated = any(w in NEGATIONS for w in prev_words)

            if keyword in HIGH_RISK_KEYWORDS:
                if negated:
                    low_count += 1
                else:
                    high_count += 1
            else:
                if negated:
                    high_count += 1
                else:
                    low_count += 1

    return high_count, low_count

def compute_risk_score(original_text: str, clean_text_: str) -> tuple[float, float]:
 
    if not clean_text_ or len(clean_text_.strip()) < 50:
        return 0.0, 0.0

    tokens = clean_text_.split()

    total_words = max(len(tokens), 1)

    high_count = 0
    low_count  = 0

    for kw in HIGH_RISK_KEYWORDS:
        h, l = keyword_count_with_negation(tokens, kw)
        high_count += h
        low_count  += l

    for kw in LOW_RISK_KEYWORDS:
        h, l = keyword_count_with_negation(tokens, kw)
        high_count += h
        low_count  += l

    high_density  = (high_count / total_words) * 1000
    low_density   = (low_count  / total_words) * 1000
    keyword_score = high_density - low_density

    tokens        = lm.tokenize(original_text)
    lm_scores     = lm.get_score(tokens)
    polarity      = lm_scores["Polarity"]
    sentiment_score = -polarity * 10

    return keyword_score, sentiment_score

def assign_labels_by_quantile(scores: pd.Series) -> pd.Series:
    q1 = scores.quantile(0.33)
    q2 = scores.quantile(0.66)
    logger.info(f"Risk score quantiles — Q33: {q1:.3f} | Q66: {q2:.3f}")

    def label(score):
        if score <= q1:
            return "low_risk"
        elif score <= q2:
            return "medium_risk"
        else:
            return "high_risk"

    return scores.apply(label)

def run_preprocessing(max_rows: int = MAX_ROWS) -> pd.DataFrame:

    df = load_latest_splits(max_rows=max_rows)

    logger.info("Extracting sections...")
    for key, col in SECTION_COLS.items():
        if col in df.columns:
            raw = df[col].astype(str).replace("nan", "")
            df[f"raw_{key}"] = raw.str[:MAX_SECTION_CHARS]
        else:
            df[f"raw_{key}"] = ""

    logger.info("Cleaning text...")
    with Timer() as t:
        for key in SECTION_COLS:
            df[f"clean_{key}"] = df[f"raw_{key}"].apply(clean_text)
    logger.info(f"Cleaning done in {t}")

    logger.info("Building combined text field...")
    df["text"] = (
        df["clean_risk"].fillna("") + " " +
        df["clean_mda"].fillna("") + " " +
        df["clean_business"].fillna("") + " " +
        df["clean_financials"].fillna("")
    ).str.strip()

    logger.info("Computing risk scores (LM + keywords) ...")

    original_label_text = df["raw_risk"].fillna("") + " " + df["raw_mda"].fillna("")
    clean_label_text    = df["clean_risk"].fillna("") + " " + df["clean_mda"].fillna("")

    with Timer() as t:
        scores = [
            compute_risk_score(orig, clean)
            for orig, clean in zip(original_label_text, clean_label_text)
        ]
        df["keyword_score"]   = [s[0] for s in scores]
        df["sentiment_score"] = [s[1] for s in scores]

        df["risk_score"] = (
            0.4 * zscore(df["keyword_score"]) +
            0.6 * zscore(df["sentiment_score"])
        )

    logger.info(f"Scoring done in {t}")
    logger.info(f"Risk score stats:\n{df['risk_score'].describe().round(3)}")

    df = df[df["risk_score"].notna()].reset_index(drop=True)

    logger.info("Assigning labels by quantile...")
    df["label"] = assign_labels_by_quantile(df["risk_score"])

    label_dist = df["label"].value_counts().to_dict()
    logger.info(f"Label distribution: {label_dist}")

    keep_cols = [
        "split",
        "text",
        "clean_risk",
        "clean_mda",
        "clean_business",
        "clean_financials",
        "risk_score",
        "label",
    ]

    if "company_name" in df.columns:
        keep_cols.insert(1, "company_name")

    df_out = df[[c for c in keep_cols if c in df.columns]].copy()

    out_path = DATA_DIR / "processed.csv"
    df_out.to_csv(out_path, index=False)
    logger.info(f"Saved → {out_path}  |  shape: {df_out.shape}")

    save_json(label_dist, DATA_DIR / "label_distribution.json")
    logger.info("Saved label_distribution.json")

    return df_out

if __name__ == "__main__":
    df = run_preprocessing(max_rows=MAX_ROWS)
    logger.info("Preprocessing complete.")