# LightGBM Model Fine-Tuning Guide

## Overview

This guide explains how to fine-tune the LightGBM model to achieve better performance than the baseline model. The fine-tuning process uses advanced hyperparameter optimization with Optuna to explore a much larger search space.

## What's Different from Baseline Training?

### Baseline (`src/models/train.py`)
- **Basic hyperparameters:** Uses default or simple parameter ranges
- **Limited optimization:** Trains once with fixed parameters
- **Quick training:** ~5-10 minutes
- **Typical AUC-ROC:** 0.84-0.86

### Fine-Tuned (`src/models/finetune_lightgbm.py`)
- **Extensive search space:** 200+ trials exploring 15+ hyperparameters
- **Advanced parameters:** Includes regularization (reg_alpha, reg_lambda), tree complexity controls, sampling strategies
- **Cross-validation:** 5-fold stratified CV for robust evaluation
- **Longer training:** ~30-60 minutes depending on hardware
- **Expected improvement:** +1-3% AUC-ROC (targeting 0.87-0.89)

## Hyperparameters Optimized

The fine-tuning script optimizes the following hyperparameters:

### Tree Structure
- `num_leaves` (20-150): Controls tree complexity
- `max_depth` (3-12): Maximum tree depth
- `min_child_samples` (5-100): Minimum samples per leaf
- `min_child_weight` (1e-5 to 10): Minimum sum of instance weight in a child

### Learning
- `learning_rate` (0.005-0.3): Step size for gradient descent
- `n_estimators` (100-1000): Number of boosting iterations

### Regularization
- `reg_alpha` (1e-8 to 10): L1 regularization
- `reg_lambda` (1e-8 to 10): L2 regularization
- `min_split_gain` (0-1): Minimum gain to make a split
- `path_smooth` (0-1): Path smoothing parameter

### Sampling
- `subsample` (0.5-1.0): Fraction of samples used per tree
- `subsample_freq` (0-7): Frequency of subsampling
- `colsample_bytree` (0.5-1.0): Fraction of features used per tree

## Prerequisites

Ensure you have:
1. Trained the baseline model: `python -m src.models.train`
2. Database populated with customer data
3. PostgreSQL running: `docker compose up -d postgres`

## How to Run Fine-Tuning

### Step 1: Activate Python Environment

```bash
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Step 2: Run Fine-Tuning Script

```bash
python -m src.models.finetune_lightgbm
```

### Step 3: Monitor Progress

The script will display:
- Trial-by-trial progress with AUC-ROC scores
- Best parameters found so far
- Cross-validation results
- Final model comparison with baseline

### Expected Output

```
==============================================================
LightGBM Advanced Fine-Tuning Pipeline
==============================================================
Loading training data from database...
Loaded 10000 customer records from database
Preparing data: 10000 samples, 17 features
...
==============================================================
Starting Hyperparameter Optimization
Number of trials: 200
Cross-validation folds: 5
==============================================================
Trial 0: AUC-ROC = 0.8534 (+/- 0.0123)
Trial 1: AUC-ROC = 0.8612 (+/- 0.0098)
...
Trial 200: AUC-ROC = 0.8789 (+/- 0.0087)
==============================================================
Optimization Complete!
Best CV AUC-ROC: 0.8789
==============================================================
Best Hyperparameters:
  num_leaves: 85
  max_depth: 8
  learning_rate: 0.0234
  ...
==============================================================
Training Final Model with Best Hyperparameters
==============================================================
Final Model Test Set Performance:
  AUC-ROC: 0.8812
  AUC-PR: 0.8456
  Accuracy: 0.8623
  Precision: 0.8145
  Recall: 0.8298
  F1-Score: 0.8221

Backed up existing model to: models/best_model_production_backup_20250127_143022.pkl
Fine-tuned model saved as production model: models/best_model_production.pkl
==============================================================
Comparing with Previous Production Model
==============================================================
Model Comparison:
------------------------------------------------------------
Metric          Baseline     Fine-tuned   Improvement
------------------------------------------------------------
AUC_ROC         0.8600       0.8812       +0.0212 (+2.47%)
AUC_PR          0.8200       0.8456       +0.0256 (+3.12%)
ACCURACY        0.8400       0.8623       +0.0223 (+2.65%)
PRECISION       0.7900       0.8145       +0.0245 (+3.10%)
RECALL          0.8100       0.8298       +0.0198 (+2.44%)
F1              0.8000       0.8221       +0.0221 (+2.76%)
------------------------------------------------------------
==============================================================
Fine-Tuning Complete!
==============================================================
Trials completed: 200
Best CV score: 0.8789
Test AUC-ROC: 0.8812
Production model updated: models/best_model_production.pkl
Previous model backed up to: models/best_model_production_backup_20250127_143022.pkl
==============================================================
✓ The fine-tuned model is now in production!
  Restart your API server to use the new model.
==============================================================
```

## Understanding the Results

### Key Metrics

- **AUC-ROC:** Primary metric for ranking quality. Higher is better (aim for >0.87)
- **AUC-PR:** Important for imbalanced data. Shows precision-recall tradeoff
- **Recall:** Percentage of actual churners identified. Critical for retention campaigns
- **Precision:** Percentage of predicted churners who actually churn. Important for cost efficiency

### What Constitutes Success?

- **Minimal improvement:** +1% AUC-ROC (+0.01)
- **Good improvement:** +2-3% AUC-ROC (+0.02-0.03)
- **Excellent improvement:** +3%+ AUC-ROC (+0.03+)

Even small improvements can have significant business impact when dealing with thousands of customers.

## Using the Fine-Tuned Model

**The fine-tuned model automatically replaces your production model!**

When you run the fine-tuning script:
1. ✅ Your existing `best_model_production.pkl` is backed up with a timestamp
2. ✅ The fine-tuned model is saved as the new `best_model_production.pkl`
3. ✅ Your API will use the new model after restart

### Activate the New Model

Simply restart your API server:

```bash
# Stop the current API (Ctrl+C in the terminal where it's running)

# Start it again
uvicorn src.api.main:app --reload --port 8000
```

The API automatically loads `models/best_model_production.pkl`, which is now your fine-tuned model!

### Rollback to Previous Model (If Needed)

If you want to revert to the previous model:

```bash
# Find the backup file (it has a timestamp)
ls -lt models/best_model_production_backup_*.pkl | head -1

# Copy the backup back to production
cp models/best_model_production_backup_YYYYMMDD_HHMMSS.pkl models/best_model_production.pkl

# Restart API
uvicorn src.api.main:app --reload --port 8000
```

## Customization Options

### Increase Number of Trials

For potentially better results (but longer training):

```python
tuner = LightGBMFineTuner(
    n_trials=500,  # Default is 200
    cv_folds=5,
)
```

### Change Cross-Validation Strategy

For faster iteration (less robust):

```python
tuner = LightGBMFineTuner(
    n_trials=200,
    cv_folds=3,  # Default is 5
)
```

### Modify Hyperparameter Search Space

Edit the `objective` method in `src/models/finetune_lightgbm.py` to adjust ranges:

```python
params = {
    "num_leaves": trial.suggest_int("num_leaves", 50, 200),  # Wider range
    "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1),  # Narrower range
    # ... other params
}
```

## MLflow Tracking

All fine-tuning experiments are tracked in MLflow:

1. **Start MLflow UI** (if not running):
   ```bash
   docker compose up -d mlflow
   ```

2. **View results:** http://localhost:5001

3. **Compare experiments:**
   - Navigate to "Experiments" tab
   - Select "lightgbm-advanced-tuning"
   - View all 200 trials with parameters and metrics
   - Compare with baseline experiments

## Troubleshooting

### Issue: "No customer data found in database"

**Solution:** Run data ingestion first:
```bash
python -m src.data.download
python -m src.data.ingestion
```

### Issue: "Baseline model not found"

**Solution:** Train baseline model first:
```bash
python -m src.models.train
```

### Issue: Out of Memory Error

**Solution:** Reduce trials or CV folds:
```python
tuner = LightGBMFineTuner(n_trials=100, cv_folds=3)
```

### Issue: Training is too slow

**Solution:**
- Reduce `n_trials` to 100 or 50
- Reduce `cv_folds` to 3
- Set `n_jobs=4` instead of `-1` to limit CPU usage

## Performance Tips

1. **Use GPU acceleration** (if available):
   - Install LightGBM with GPU support: `pip install lightgbm --install-option=--gpu`
   - Add `device="gpu"` to model parameters

2. **Parallel trials** (requires Ray):
   ```bash
   pip install optuna[ray]
   ```
   Then modify the study creation to use Ray backend

3. **Early stopping for trials:**
   The script uses `MedianPruner` to stop unpromising trials early, saving time

## Next Steps

After fine-tuning:

1. **Validate on new data:** Test the model on fresh customer data
2. **Monitor in production:** Track prediction accuracy over time
3. **Retrain periodically:** Customer behavior changes; retrain every 3-6 months
4. **A/B testing:** Run fine-tuned model alongside baseline to measure real-world impact

## Advanced: Ensemble Models

For even better performance, consider ensemble approaches:

```python
# Train multiple fine-tuned models with different seeds
# Average their predictions for final output
```

See `src/models/ensemble.py` (to be implemented) for ensemble strategies.

---

**Questions or Issues?** Open an issue on GitHub or consult the MLflow tracking UI for detailed experiment logs.
