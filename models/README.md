# Models Directory

This directory stores trained machine learning models for the ChurnGuard application.

## Generating Models

Model files are not committed to git due to their size. To generate them:

```bash
# Ensure PostgreSQL is running
docker compose up -d postgres

# Ensure data is ingested
python -m src.data.ingestion

# Train models
python -m src.models.train
```

This will create:
- `best_model_production.pkl` - Best performing model used by the API
- `lightgbm_latest.pkl` - LightGBM model
- `xgboost_latest.pkl` - XGBoost model
- `random_forest_latest.pkl` - Random Forest model
- `logistic_regression_latest.pkl` - Logistic Regression model

## Model Details

- **Training Data**: Bank Customer Churn dataset (10,000 customers)
- **Features**: 17 features including credit score, geography, age, tenure, balance, etc.
- **Target**: Customer churn (exited)
- **Best Model**: Typically LightGBM or XGBoost with AUC-ROC ~0.85-0.90

## File Size

Model files are typically 1-5 MB each and are excluded from git via `.gitignore`.
