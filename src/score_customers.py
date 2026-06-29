import json
from pathlib import Path

import joblib
import pandas as pd

from explain_risk import explain_customer_risk
from feature_engineering import ensure_customer_id, prepare_model_features, prepare_training_frame
from recommend_actions import add_retention_recommendations, risk_level_from_probability


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "raw" / "Customer Churn.csv"
FALLBACK_DATA_PATH = BASE_DIR / "Customer Churn.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODEL_PATH = ARTIFACTS_DIR / "iranian_churn_model.joblib"
METADATA_PATH = ARTIFACTS_DIR / "model_metadata.json"
SEGMENTS_PATH = PROCESSED_DIR / "customer_segments.csv"
CUSTOMERS_SCORED_PATH = PROCESSED_DIR / "customers_scored.csv"
RETENTION_ACTIONS_PATH = PROCESSED_DIR / "retention_actions.csv"


def load_dataset() -> pd.DataFrame:
    path = DATA_PATH if DATA_PATH.exists() else FALLBACK_DATA_PATH
    return pd.read_csv(path)


def load_segments() -> pd.DataFrame:
    if not SEGMENTS_PATH.exists():
        from build_segments import main as build_segments_main

        build_segments_main()
    return pd.read_csv(SEGMENTS_PATH)


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    model = joblib.load(MODEL_PATH)
    threshold = float(metadata.get("threshold", 0.5))

    raw_data = ensure_customer_id(load_dataset())
    data, _ = prepare_training_frame(raw_data)
    features = prepare_model_features(data, metadata)

    probabilities = model.predict_proba(features)[:, 1]
    data["churn_probability"] = probabilities
    data["churn_prediction"] = (data["churn_probability"] >= threshold).astype(int)
    data["risk_level"] = data["churn_probability"].apply(risk_level_from_probability)
    data["threshold"] = threshold
    data["main_reason"] = data.apply(
        lambda row: explain_customer_risk(row, metadata.get("risk_driver_thresholds", {})),
        axis=1,
    )

    segments = load_segments()[["customer_id", "segment"]]
    data = data.merge(segments, on="customer_id", how="left")
    data = add_retention_recommendations(data, metadata)

    output_columns = [
        "customer_id",
        "churn_probability",
        "churn_prediction",
        "risk_level",
        "threshold",
        "customer_value",
        "customer_value_tier",
        "retention_priority_score",
        "segment",
        "main_reason",
        "recommended_action",
        "priority",
    ]
    data[output_columns].to_csv(CUSTOMERS_SCORED_PATH, index=False)
    data[output_columns].to_csv(RETENTION_ACTIONS_PATH, index=False)

    print(f"Saved scored customers to {CUSTOMERS_SCORED_PATH}")
    print(f"Saved retention actions to {RETENTION_ACTIONS_PATH}")


if __name__ == "__main__":
    main()
