"""Download Bank Customer Churn dataset from Kaggle."""

import logging
import os
from pathlib import Path
from typing import Optional

import polars as pl
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dataset information
KAGGLE_DATASET = "shantanudhakadd/bank-customer-churn-prediction"
DATASET_FILE = "Churn_Modelling.csv"


def download_from_kaggle(output_dir: Optional[Path] = None) -> Path:
    """
    Download dataset from Kaggle using kaggle API.

    Args:
        output_dir: Directory to save dataset (defaults to data/raw)

    Returns:
        Path to downloaded file
    """
    # Set output directory
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / "data" / "raw"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if file already exists
    destination = output_dir / DATASET_FILE
    if destination.exists():
        logger.info(f"Dataset already exists at {destination}")
        return destination

    # Try to import kaggle
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        logger.info("Using Kaggle API to download dataset")

        # Initialize Kaggle API
        api = KaggleApi()
        api.authenticate()

        # Download dataset
        logger.info(f"Downloading dataset: {KAGGLE_DATASET}")
        api.dataset_download_files(
            KAGGLE_DATASET,
            path=str(output_dir),
            unzip=True,
        )

        logger.info(f"Downloaded successfully to {destination}")
        return destination

    except ImportError:
        logger.error("Kaggle package not installed. Install with: pip install kaggle")
        logger.info("Alternative: Download manually from Kaggle")
        raise


def download_manually_instructions() -> str:
    """
    Provide instructions for manual download.

    Returns:
        Instructions string
    """
    instructions = f"""
    Manual Download Instructions:

    1. Go to: https://www.kaggle.com/datasets/{KAGGLE_DATASET}
    2. Click "Download" button
    3. Extract the zip file
    4. Copy {DATASET_FILE} to: data/raw/

    Or use Kaggle API:
    1. Install: pip install kaggle
    2. Set up Kaggle credentials:
       - Go to https://www.kaggle.com/account
       - Click "Create New API Token"
       - Save kaggle.json to ~/.kaggle/kaggle.json (Linux/Mac) or %USERPROFILE%\\.kaggle\\kaggle.json (Windows)
    3. Run this script again
    """
    return instructions


def load_and_validate_dataset(file_path: Path) -> pl.DataFrame:
    """
    Load and perform initial validation of the dataset.

    Args:
        file_path: Path to the CSV file

    Returns:
        Polars DataFrame with the loaded data
    """
    logger.info(f"Loading dataset from {file_path}")

    # Load with Polars for high performance
    df = pl.read_csv(file_path)

    logger.info(f"Dataset loaded successfully")
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Columns: {df.columns}")

    # Basic validation
    if df.height == 0:
        raise ValueError("Dataset is empty")

    # Check for expected columns
    expected_columns = [
        "CustomerId",
        "Surname",
        "CreditScore",
        "Geography",
        "Gender",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
        "Exited",
    ]

    missing_columns = set(expected_columns) - set(df.columns)
    if missing_columns:
        logger.warning(f"Missing expected columns: {missing_columns}")

    logger.info("Dataset validation passed")

    return df


def standardize_column_names(df: pl.DataFrame) -> pl.DataFrame:
    """
    Standardize column names to match our database schema.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with standardized column names
    """
    # Column mapping to snake_case
    column_mapping = {
        "CustomerId": "customer_id",
        "Surname": "surname",
        "CreditScore": "credit_score",
        "Geography": "geography",
        "Gender": "gender",
        "Age": "age",
        "Tenure": "tenure",
        "Balance": "balance",
        "NumOfProducts": "num_of_products",
        "HasCrCard": "has_credit_card",
        "IsActiveMember": "is_active_member",
        "EstimatedSalary": "estimated_salary",
        "Exited": "exited",
        "RowNumber": "row_number",
    }

    # Rename columns that exist in the mapping
    for old_name, new_name in column_mapping.items():
        if old_name in df.columns:
            df = df.rename({old_name: new_name})

    logger.info(f"Standardized columns: {df.columns}")

    return df


def save_processed_dataset(df: pl.DataFrame, output_path: Optional[Path] = None) -> Path:
    """
    Save processed dataset to CSV.

    Args:
        df: DataFrame to save
        output_path: Path to save to (defaults to data/processed/bank_churn_processed.csv)

    Returns:
        Path to saved file
    """
    if output_path is None:
        output_path = Path(__file__).parent.parent.parent / "data" / "processed" / "bank_churn_processed.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.write_csv(output_path)
    logger.info(f"Processed dataset saved to {output_path}")

    return output_path


def main() -> None:
    """Main function to download and process the dataset."""
    logger.info("Starting Bank Customer Churn dataset download and processing")

    try:
        # Download dataset
        raw_file_path = download_from_kaggle()

    except Exception as e:
        logger.error(f"Failed to download from Kaggle: {e}")
        logger.info(download_manually_instructions())

        # Check if file exists manually
        raw_file_path = Path(__file__).parent.parent.parent / "data" / "raw" / DATASET_FILE
        if not raw_file_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {raw_file_path}. "
                f"Please download manually from Kaggle."
            )

    # Load and validate
    df = load_and_validate_dataset(raw_file_path)

    # Standardize column names
    df = standardize_column_names(df)

    # Save processed dataset
    processed_file_path = save_processed_dataset(df)

    logger.info("Dataset download and processing completed successfully")
    logger.info(f"Raw data: {raw_file_path}")
    logger.info(f"Processed data: {processed_file_path}")


if __name__ == "__main__":
    main()
