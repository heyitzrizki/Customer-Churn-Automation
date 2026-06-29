import pandas as pd


def explain_customer_risk(row: pd.Series, thresholds: dict | None = None) -> str:
    thresholds = thresholds or {}
    reasons = []

    failed_call_cutoff = float(thresholds.get("failed_call_rate_high", 0.20))
    low_usage_cutoff = float(thresholds.get("frequency_of_use_low", 20))
    low_duration_cutoff = float(thresholds.get("seconds_of_use_low", 1200))
    low_value_cutoff = float(thresholds.get("customer_value_low", 80))

    if float(row.get("complains", 0)) >= 1:
        reasons.append("Complaint recorded")
    if float(row.get("failed_call_rate", 0)) >= failed_call_cutoff:
        reasons.append("High failed call rate")
    if float(row.get("frequency_of_use", 0)) <= low_usage_cutoff:
        reasons.append("Low usage frequency")
    if float(row.get("seconds_of_use", 0)) <= low_duration_cutoff:
        reasons.append("Low usage duration")
    if float(row.get("customer_value", 0)) <= low_value_cutoff:
        reasons.append("Low customer value")
    if int(float(row.get("status", 1))) != 1:
        reasons.append("Inactive or higher-risk customer status")

    if not reasons:
        reasons.append("Model detected elevated churn risk based on customer behavior")

    return "; ".join(reasons[:3])
