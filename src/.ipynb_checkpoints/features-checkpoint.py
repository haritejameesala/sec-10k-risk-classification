import numpy as np
import pandas as pd
import scipy.sparse as sp
import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from .utils import get_logger, MODELS_DIR, Timer

logger = get_logger("features")

_TFIDF_DEFAULTS = dict(
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.90,
    sublinear_tf=True,
    strip_accents="unicode",
    analyzer="word",
)

RISK_TFIDF_CFG       = dict(**_TFIDF_DEFAULTS, max_features=7000)
MDA_TFIDF_CFG        = dict(**_TFIDF_DEFAULTS, max_features=5000)
BUSINESS_TFIDF_CFG   = dict(**_TFIDF_DEFAULTS, max_features=1500)
FINANCIALS_TFIDF_CFG = dict(**_TFIDF_DEFAULTS, max_features=300)

HIGH_RISK_WORDS = [
    "litigation",
    "default",
    "bankruptcy",
    "impairment",
    "failure",
    "penalty",
    "fraud",
    "breach",
    "investigation",
    "restatement",
    "insolvency",
    "recession",
    "cybersecurity",
    "data breach",
    "going concern",
    "material weakness",
]

LOW_RISK_WORDS = [
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

NEGATIONS = {"not", "no", "never", "neither", "nor", "none", "cannot", "without"}

CUSTOM_FEATURE_NAMES = [
    "doc_length_log",
    "high_risk_density",
    "low_risk_density",
    "risk_ratio",
    "negation_density",
    "avg_word_length",
    "unique_token_ratio",
    "sentence_count_log",

    "risk_len_log",
    "mda_len_log",
    "business_len_log",
    "financials_len_log",
]


def _custom_features(row: pd.Series) -> list:

    text = row["text"] if isinstance(row["text"], str) else ""
    risk_text = row.get("clean_risk", "")
    mda_text = row.get("clean_mda", "")

    risk_text = risk_text if isinstance(risk_text, str) else ""
    mda_text = mda_text if isinstance(mda_text, str) else ""

    combined_text = f"{risk_text} {mda_text}"
    tokens = combined_text.split()
    n = max(len(tokens), 1)
    token_set = set(tokens)

    high = 0
    low = 0

    for kw in HIGH_RISK_WORDS:
	    kw_tokens = kw.split()

	    for i in range(len(tokens) - len(kw_tokens) + 1):
		    if tokens[i:i + len(kw_tokens)] == kw_tokens:
			    prev = tokens[max(0, i - 3):i]
			    negated = any(w in NEGATIONS for w in prev)

			    if negated:
				    low += 1
			    else:
				    high += 1

    for kw in LOW_RISK_WORDS:
	    kw_tokens = kw.split()

	    for i in range(len(tokens) - len(kw_tokens) + 1):
		    if tokens[i:i + len(kw_tokens)] == kw_tokens:
			    prev = tokens[max(0, i - 3):i]
			    negated = any(w in NEGATIONS for w in prev)

			    if negated:
				    high += 1
			    else:
				    low += 1

    neg = sum(1 for t in tokens if t in NEGATIONS)

    avg_wl = float(np.mean([len(t) for t in tokens])) if tokens else 0.0
    unique_ratio = len(token_set) / n
    sent_count = max(text.count(".") + text.count("!") + text.count("?"), 1)

    risk_len       = len(str(row.get("clean_risk",       "")).split())
    mda_len        = len(str(row.get("clean_mda",        "")).split())
    business_len   = len(str(row.get("clean_business",   "")).split())
    financials_len = len(str(row.get("clean_financials", "")).split())

    return [
        np.log1p(n),
        high / n,
        low / n,
        (high + 1) / (low + 1),
        neg / n,
        avg_wl,
        unique_ratio,
        np.log1p(sent_count),
        np.log1p(risk_len),
        np.log1p(mda_len),
        np.log1p(business_len),
        np.log1p(financials_len),
    ]



class FeatureBuilder:
  
    def __init__(self):
        self.risk_tfidf       = TfidfVectorizer(**RISK_TFIDF_CFG)
        self.mda_tfidf        = TfidfVectorizer(**MDA_TFIDF_CFG)
        self.business_tfidf   = TfidfVectorizer(**BUSINESS_TFIDF_CFG)
        self.financials_tfidf = TfidfVectorizer(**FINANCIALS_TFIDF_CFG)
        self.scaler           = StandardScaler()
        self._fitted          = False

    def fit(self, df: pd.DataFrame) -> "FeatureBuilder":

        logger.info("Fitting section TF-IDF vectorizers ...")
        with Timer() as t:
            self.risk_tfidf.fit(df["clean_risk"].fillna(""))
            self.mda_tfidf.fit(df["clean_mda"].fillna(""))
            self.business_tfidf.fit(df["clean_business"].fillna(""))
            self.financials_tfidf.fit(df["clean_financials"].fillna(""))

        n_tfidf = (
            len(self.risk_tfidf.vocabulary_)
            + len(self.mda_tfidf.vocabulary_)
            + len(self.business_tfidf.vocabulary_)
            + len(self.financials_tfidf.vocabulary_)
        )
        logger.info(f"TF-IDF fitted: {n_tfidf} total vocab features in {t}.")

        logger.info("Fitting custom-feature scaler ...")
        custom = np.array(
            [_custom_features(row) for _, row in df.iterrows()],
            dtype=np.float32,
        )
        self.scaler.fit(custom)
        self._fitted = True
        return self
		
    def transform(self, df: pd.DataFrame) -> sp.csr_matrix:
        if not self._fitted:
            raise RuntimeError("FeatureBuilder must be fitted before transform.")

        risk_mat       = self.risk_tfidf.transform(df["clean_risk"].fillna(""))
        mda_mat        = self.mda_tfidf.transform(df["clean_mda"].fillna(""))
        business_mat   = self.business_tfidf.transform(df["clean_business"].fillna(""))
        financials_mat = self.financials_tfidf.transform(df["clean_financials"].fillna(""))

        custom = np.array(
            [_custom_features(row) for _, row in df.iterrows()],
            dtype=np.float32,
        )
        custom_scaled = self.scaler.transform(custom)

        return sp.hstack(
            [risk_mat, mda_mat, business_mat, financials_mat,
             sp.csr_matrix(custom_scaled)],
            format="csr",
        )

    def fit_transform(self, df: pd.DataFrame) -> sp.csr_matrix:
        return self.fit(df).transform(df)

    def save(self, path: Path = None) -> Path:
        path = path or (MODELS_DIR / "feature_builder.joblib")
        joblib.dump(self, path)
        logger.info(f"FeatureBuilder saved -> {path}")
        return path

    @staticmethod
    def load(path: Path = None) -> "FeatureBuilder":
        path = path or (MODELS_DIR / "feature_builder.joblib")
        fb = joblib.load(path)
        logger.info(f"FeatureBuilder loaded <- {path}")
        return fb


    @property
    def feature_names(self) -> list[str]:
        risk_names       = self.risk_tfidf.get_feature_names_out().tolist()
        mda_names        = self.mda_tfidf.get_feature_names_out().tolist()
        business_names   = self.business_tfidf.get_feature_names_out().tolist()
        financials_names = self.financials_tfidf.get_feature_names_out().tolist()
        return risk_names + mda_names + business_names + financials_names + CUSTOM_FEATURE_NAMES


def build_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[sp.csr_matrix, sp.csr_matrix, "FeatureBuilder"]:
  
    fb = FeatureBuilder()
    X_train = fb.fit_transform(train_df)
    X_test  = fb.transform(test_df)
    fb.save()
    logger.info(
        f"Feature matrix shapes -> train: {X_train.shape}, test: {X_test.shape}"
    )
    return X_train, X_test, fb



if __name__ == "__main__":
    sample = pd.DataFrame({
        "text":             ["risk fraud bankruptcy", "growth profit stable", "mixed results"],
        "clean_risk":       ["risk fraud bankruptcy", "no material litigation", "some risk"],
        "clean_mda":        ["adverse volatile decline", "increased profitability", "moderate"],
        "clean_business":   ["disruption inflation",   "market leader", "stable"],
        "clean_financials": ["negative cash flow",     "positive cash flow", ""],
    })
    fb = FeatureBuilder()
    X = fb.fit_transform(sample)
    print(f"Feature matrix shape: {X.shape}") 