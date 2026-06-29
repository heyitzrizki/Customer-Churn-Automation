# n8n Automation Notes

This project keeps the root `predict.py` entry point for existing n8n workflows and also provides `automation/predict.py` for the organized portfolio structure.

Example Execute Command node:

```bash
python /path/to/Customer-Churn-Automation/predict.py --json '{{ JSON.stringify($json) }}'
```

The command returns valid JSON with the original prediction fields plus business-ready fields that can be mapped into Telegram and HubSpot:

- `churn_probability`
- `churn_prediction`
- `threshold`
- `risk_level`
- `customer_value_tier`
- `retention_priority_score`
- `main_reason`
- `recommended_action`
- `priority`

Extra fields such as `customer_id`, `email`, `name`, `nama`, and `phone` are preserved in the output when present.
