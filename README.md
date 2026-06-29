# Customer Retention Intelligence System

An end-to-end customer retention intelligence system that combines churn prediction, customer segmentation, business-readable risk explanation, recommended retention actions, and n8n automation.

## Business Problem

Telecom retention teams need more than a churn probability. They need to know which customers deserve attention, why they are at risk, and what action should happen next in CRM or automation workflows.

This project turns Iranian telecom customer churn data into an operational retention workflow that can support business users, portfolio review, and n8n-driven follow-up.

## What This Project Does

- Predicts customer churn risk from customer behavior data.
- Compares Logistic Regression, Random Forest, XGBoost, and LightGBM when available.
- Groups customers into business-readable customer segments.
- Scores retention priority using churn risk, customer value, and service issue signals.
- Explains each risk score with simple risk drivers.
- Recommends a practical retention action for each customer.
- Keeps the existing JSON prediction workflow compatible with n8n.

## Dashboard Pages

The Streamlit app has only two business-facing pages:

1. **Customer Segmentation**: shows customer groups, segment size, average churn risk, average value, complaint rate, suggested strategy, and a value-versus-risk map.
2. **Churn Risk Scoring**: shows who to act on, why, recommended actions, priority level, and a downloadable retention target list.

Run it with:

```bash
streamlit run app/streamlit_app.py
```

## Machine Learning Model Comparison

`src/train_models.py` trains and compares:

- Logistic Regression
- Random Forest
- XGBoost, if installed
- LightGBM, if installed

Models are evaluated using ROC-AUC, PR-AUC, F1-score, precision, recall, and accuracy. The best model is saved to `artifacts/iranian_churn_model.joblib`, and the comparison table is saved to `data/processed/model_comparison.csv`.

If XGBoost or LightGBM are unavailable, the script skips them and continues with the available models.

## Retention Priority and Action Logic

The retention layer is a hybrid business decision layer:

- Churn probability comes from the machine learning model.
- Retention priority combines model risk, customer value, and service issue signals.
- Recommended actions are assigned using transparent business rules.

Priority score:

```text
0.50 * churn risk score
+ 0.30 * customer value score
+ 0.20 * service issue score
```

Priority levels:

- `P1`: score >= 80
- `P2`: 60 <= score < 80
- `P3`: 40 <= score < 60
- `P4`: score < 40

## n8n Automation Workflow

The root `predict.py` remains available for existing n8n Execute Command workflows:

```bash
python /path/to/Customer-Churn-Automation/predict.py --json '{{ JSON.stringify($json) }}'
```

The output keeps the original fields:

- `churn_probability`
- `churn_prediction`
- `threshold`

It also adds business-ready fields for Telegram alerts, HubSpot updates, or CRM routing:

- `risk_level`
- `customer_value_tier`
- `retention_priority_score`
- `main_reason`
- `recommended_action`
- `priority`

See `automation/README_n8n.md` for the automation notes.

## Repository Structure

```text
Customer-Churn-Automation/
├── app/
│   ├── streamlit_app.py
│   └── assets/
│       └── n8n_workflow.png
├── artifacts/
│   ├── iranian_churn_model.joblib
│   ├── model_metadata.json
│   ├── segmentation_model.joblib
│   └── scaler.joblib
├── data/
│   ├── raw/
│   │   └── Customer Churn.csv
│   └── processed/
│       ├── customers_scored.csv
│       ├── customer_segments.csv
│       ├── model_comparison.csv
│       └── retention_actions.csv
├── src/
│   ├── feature_engineering.py
│   ├── train_models.py
│   ├── build_segments.py
│   ├── score_customers.py
│   ├── recommend_actions.py
│   └── explain_risk.py
├── automation/
│   ├── predict.py
│   └── README_n8n.md
├── notebooks/
│   └── Customer_Churn_Prediction.ipynb
├── predict.py
├── sample_customer.json
├── Dockerfile
├── requirements.txt
└── README.md
```

The original root dataset and notebook are preserved so older local workflows are not broken.

## How To Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Train and compare models:

```bash
python src/train_models.py
```

Build customer segments:

```bash
python src/build_segments.py
```

Score all customers and create retention action outputs:

```bash
python src/score_customers.py
```

Test the n8n-compatible JSON predictor:

```bash
python predict.py --input sample_customer.json
```

Launch the Streamlit app:

```bash
streamlit run app/streamlit_app.py
```

## Limitations

- The dataset does not include real campaign history.
- Recommended actions are rule-based business decision logic, not proven causal treatment effects.
- Campaign ROI is not calculated using real post-campaign outcomes.
- Estimated value at risk is a customer value proxy, not actual proven lost revenue.
- The system demonstrates how model outputs can be operationalized for retention workflows.
- If XGBoost or LightGBM are unavailable in the environment, the project falls back to available models.

## Future Improvements

- Add campaign response history and measure retention lift.
- Add treatment/control experiment tracking.
- Connect scored retention targets directly to HubSpot lists.
- Add scheduled batch scoring through n8n or a lightweight job runner.
- Add monitoring for model drift and changes in churn rate.
