# Kaggle API Setup Guide

This guide explains how to set up Kaggle API credentials to download the Bank Customer Churn dataset.

## Dataset Information

**Dataset**: Bank Customer Churn Prediction
**Kaggle URL**: https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction
**File**: `Churn_Modelling.csv`
**Size**: ~2MB, 10,000 customers

### Features:
- **Customer Info**: CustomerId, Surname, Geography, Gender, Age
- **Banking Info**: CreditScore, Balance, Tenure, EstimatedSalary
- **Product Usage**: NumOfProducts, HasCrCard, IsActiveMember
- **Target**: Exited (1 = churned, 0 = retained)

---

## Option 1: Kaggle API (Recommended)

### Step 1: Install Kaggle Package

```bash
pip install kaggle
```

### Step 2: Get Kaggle API Credentials

1. **Go to Kaggle**:
   - Visit: https://www.kaggle.com/account
   - Sign in or create an account (it's free!)

2. **Create API Token**:
   - Scroll down to **"API"** section
   - Click **"Create New API Token"**
   - This downloads `kaggle.json` file

3. **Save Credentials**:

   **On Mac/Linux**:
   ```bash
   # Create .kaggle directory
   mkdir -p ~/.kaggle

   # Move the downloaded kaggle.json
   mv ~/Downloads/kaggle.json ~/.kaggle/

   # Set permissions
   chmod 600 ~/.kaggle/kaggle.json
   ```

   **On Windows**:
   ```powershell
   # Create .kaggle directory
   mkdir $env:USERPROFILE\.kaggle

   # Move the downloaded kaggle.json
   move Downloads\kaggle.json $env:USERPROFILE\.kaggle\
   ```

### Step 3: Verify Setup

```bash
# Test Kaggle API
kaggle datasets list

# If you see a list of datasets, you're good to go!
```

### Step 4: Download Dataset

```bash
# From ChurnGuard directory
python -m src.data.download
```

The script will:
- Authenticate with Kaggle API
- Download the dataset to `data/raw/`
- Process and save to `data/processed/`

---

## Option 2: Manual Download

If you prefer not to use the API or encounter issues:

### Step 1: Download from Kaggle

1. Go to: https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction
2. Click the **"Download"** button (requires Kaggle account)
3. Extract the zip file

### Step 2: Copy to Project

```bash
# Copy the CSV file to data/raw/
cp path/to/Churn_Modelling.csv data/raw/

# Or on Windows:
# copy path\to\Churn_Modelling.csv data\raw\
```

### Step 3: Process Dataset

```bash
# Run the processing script
python -m src.data.download
```

---

## Troubleshooting

### Error: "Kaggle API credentials not found"

**Solution**:
```bash
# Check if kaggle.json exists
# On Mac/Linux:
ls -la ~/.kaggle/kaggle.json

# On Windows:
dir %USERPROFILE%\.kaggle\kaggle.json

# If not found, repeat Step 2 of Kaggle API setup
```

### Error: "401 Unauthorized"

**Solution**:
- Your kaggle.json is invalid or expired
- Create a new API token from https://www.kaggle.com/account
- Replace the old kaggle.json file

### Error: "403 Forbidden"

**Solution**:
- You haven't accepted the dataset's terms
- Visit: https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction
- Click "Download" to accept terms
- Try again

### Kaggle Not Installed

**Solution**:
```bash
pip install --upgrade kaggle
```

---

## Dataset Details

### Columns:

| Column | Type | Description |
|--------|------|-------------|
| RowNumber | int | Row index |
| CustomerId | int | Unique customer ID |
| Surname | str | Customer surname |
| CreditScore | int | Credit score (300-850) |
| Geography | str | Country (France, Germany, Spain) |
| Gender | str | Male/Female |
| Age | int | Customer age |
| Tenure | int | Years with bank |
| Balance | float | Account balance |
| NumOfProducts | int | Number of products (1-4) |
| HasCrCard | int | Has credit card (1=Yes, 0=No) |
| IsActiveMember | int | Active member (1=Yes, 0=No) |
| EstimatedSalary | float | Annual salary estimate |
| Exited | int | **Target** - Churned (1=Yes, 0=No) |

### Statistics:
- **Total Customers**: 10,000
- **Churn Rate**: ~20% (2,037 churned)
- **Geography**: France (50%), Germany (25%), Spain (25%)
- **Gender**: Male (54.6%), Female (45.4%)
- **Age Range**: 18-92 years
- **Tenure**: 0-10 years

---

## Security Best Practices

### ⚠️ Never commit kaggle.json to Git!

The `.gitignore` already excludes:
```gitignore
# Kaggle
.kaggle/
kaggle.json
```

### Rotate API Keys

If you accidentally expose your kaggle.json:
1. Go to https://www.kaggle.com/account
2. Click "Create New API Token" (this invalidates the old one)
3. Update your local kaggle.json

---

## Next Steps

After downloading the dataset:

1. **Verify Download**:
   ```bash
   ls -lh data/raw/Churn_Modelling.csv
   # Should show ~2MB file
   ```

2. **Check Processed Data**:
   ```bash
   ls -lh data/processed/bank_churn_processed.csv
   ```

3. **Continue Setup**:
   - Run database migrations: `alembic upgrade head`
   - Ingest data: `python -m src.data.ingestion`
   - Train models: `python -m src.models.train`

---

## Alternative: Use Sample Data

For testing without Kaggle:

```python
# Create sample data (in Python)
import polars as pl
import numpy as np

np.random.seed(42)
n = 1000

df = pl.DataFrame({
    "customer_id": range(1, n+1),
    "credit_score": np.random.randint(300, 850, n),
    "geography": np.random.choice(["France", "Germany", "Spain"], n),
    "gender": np.random.choice(["Male", "Female"], n),
    "age": np.random.randint(18, 70, n),
    "tenure": np.random.randint(0, 11, n),
    "balance": np.random.uniform(0, 250000, n),
    "num_of_products": np.random.randint(1, 5, n),
    "has_credit_card": np.random.randint(0, 2, n),
    "is_active_member": np.random.randint(0, 2, n),
    "estimated_salary": np.random.uniform(10000, 200000, n),
    "exited": np.random.randint(0, 2, n),
})

df.write_csv("data/raw/Churn_Modelling.csv")
```

---

## Support

- **Kaggle API Docs**: https://github.com/Kaggle/kaggle-api
- **Dataset Page**: https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction
- **Issues**: Open an issue in the repository

Happy modeling! 🚀
