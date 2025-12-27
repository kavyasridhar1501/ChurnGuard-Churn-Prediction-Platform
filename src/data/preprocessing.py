"""Data preprocessing using Polars for high-performance transformations."""

import logging
from datetime import date, timedelta
from typing import Optional, Tuple

import numpy as np
import polars as pl

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChurnDataPreprocessor:
    """Preprocessor for bank customer churn data."""

    def __init__(self, df: pl.DataFrame):
        """
        Initialize preprocessor with DataFrame.

        Args:
            df: Input DataFrame
        """
        self.df = df
        self.original_shape = df.shape
        logger.info(f"Initialized preprocessor with data shape: {self.original_shape}")

    def handle_missing_values(self, strategy: str = "drop") -> "ChurnDataPreprocessor":
        """
        Handle missing values in the dataset.

        Args:
            strategy: Strategy for handling missing values ('drop', 'mean', 'median', 'mode')

        Returns:
            Self for method chaining
        """
        logger.info(f"Handling missing values with strategy: {strategy}")

        # Check for missing values
        null_counts = self.df.null_count()
        logger.info(f"Null counts: {null_counts}")

        if strategy == "drop":
            self.df = self.df.drop_nulls()
        elif strategy == "mean":
            # Fill numeric columns with mean
            for col in self.df.columns:
                if self.df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]:
                    mean_value = self.df[col].mean()
                    self.df = self.df.with_columns(pl.col(col).fill_null(mean_value))
        elif strategy == "median":
            # Fill numeric columns with median
            for col in self.df.columns:
                if self.df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]:
                    median_value = self.df[col].median()
                    self.df = self.df.with_columns(pl.col(col).fill_null(median_value))
        elif strategy == "mode":
            # Fill with mode (most frequent value)
            for col in self.df.columns:
                mode_value = self.df[col].mode().first()
                self.df = self.df.with_columns(pl.col(col).fill_null(mode_value))

        logger.info(f"After handling missing values, shape: {self.df.shape}")
        return self

    def convert_boolean_columns(self) -> "ChurnDataPreprocessor":
        """
        Convert yes/no and True/False string columns to boolean.

        Returns:
            Self for method chaining
        """
        logger.info("Converting boolean columns")

        # List of potential boolean columns for bank dataset
        boolean_candidates = ["exited", "has_credit_card", "is_active_member"]

        for col in boolean_candidates:
            if col in self.df.columns:
                # Convert integer booleans (0/1) to actual booleans
                if self.df[col].dtype in [pl.Int64, pl.Int32, pl.Int8]:
                    self.df = self.df.with_columns(
                        pl.col(col).cast(pl.Boolean).alias(col)
                    )
                elif self.df[col].dtype == pl.Utf8:
                    # Convert string booleans to actual booleans
                    self.df = self.df.with_columns(
                        pl.when(pl.col(col).str.to_lowercase().is_in(["yes", "true", "1", "t"]))
                        .then(True)
                        .when(pl.col(col).str.to_lowercase().is_in(["no", "false", "0", "f"]))
                        .then(False)
                        .otherwise(None)
                        .alias(col)
                    )

        return self

    def detect_and_handle_outliers(
        self,
        method: str = "iqr",
        threshold: float = 1.5,
    ) -> "ChurnDataPreprocessor":
        """
        Detect and handle outliers in numeric columns.

        Args:
            method: Method for outlier detection ('iqr', 'zscore')
            threshold: Threshold for outlier detection

        Returns:
            Self for method chaining
        """
        logger.info(f"Detecting outliers using {method} method")

        numeric_cols = [col for col in self.df.columns if self.df[col].dtype in [pl.Float64, pl.Int64]]

        outlier_counts = {}

        for col in numeric_cols:
            if method == "iqr":
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR

                # Count outliers
                outliers = self.df.filter((pl.col(col) < lower_bound) | (pl.col(col) > upper_bound))
                outlier_counts[col] = len(outliers)

                # Cap outliers
                self.df = self.df.with_columns(
                    pl.when(pl.col(col) < lower_bound)
                    .then(lower_bound)
                    .when(pl.col(col) > upper_bound)
                    .then(upper_bound)
                    .otherwise(pl.col(col))
                    .alias(col)
                )

            elif method == "zscore":
                mean = self.df[col].mean()
                std = self.df[col].std()

                # Count outliers (|z| > threshold)
                z_scores = (self.df[col] - mean) / std
                outliers = self.df.filter((z_scores.abs() > threshold))
                outlier_counts[col] = len(outliers)

                # Cap at mean ± threshold * std
                lower_bound = mean - threshold * std
                upper_bound = mean + threshold * std

                self.df = self.df.with_columns(
                    pl.when(pl.col(col) < lower_bound)
                    .then(lower_bound)
                    .when(pl.col(col) > upper_bound)
                    .then(upper_bound)
                    .otherwise(pl.col(col))
                    .alias(col)
                )

        logger.info(f"Outliers detected and capped: {outlier_counts}")
        return self

    def add_derived_features(self) -> "ChurnDataPreprocessor":
        """
        Add derived features based on existing columns for bank dataset.

        Returns:
            Self for method chaining
        """
        logger.info("Adding derived features for bank dataset")

        # Balance to salary ratio
        if "balance" in self.df.columns and "estimated_salary" in self.df.columns:
            self.df = self.df.with_columns(
                (pl.col("balance") / pl.col("estimated_salary").clip(lower_bound=1)).alias("balance_salary_ratio")
            )

        # Age grouping
        if "age" in self.df.columns:
            self.df = self.df.with_columns(
                pl.when(pl.col("age") < 30)
                .then(pl.lit("young"))
                .when(pl.col("age") < 45)
                .then(pl.lit("middle_aged"))
                .when(pl.col("age") < 60)
                .then(pl.lit("senior"))
                .otherwise(pl.lit("elderly"))
                .alias("age_group")
            )

        # Tenure to age ratio (loyalty indicator)
        if "tenure" in self.df.columns and "age" in self.df.columns:
            self.df = self.df.with_columns(
                (pl.col("tenure") / pl.col("age").clip(lower_bound=18)).alias("tenure_age_ratio")
            )

        # Products per tenure (engagement)
        if "num_of_products" in self.df.columns and "tenure" in self.df.columns:
            self.df = self.df.with_columns(
                (pl.col("num_of_products") / pl.col("tenure").clip(lower_bound=1)).alias("products_per_tenure")
            )

        # Credit score grouping
        if "credit_score" in self.df.columns:
            self.df = self.df.with_columns(
                pl.when(pl.col("credit_score") < 500)
                .then(pl.lit("poor"))
                .when(pl.col("credit_score") < 650)
                .then(pl.lit("fair"))
                .when(pl.col("credit_score") < 750)
                .then(pl.lit("good"))
                .otherwise(pl.lit("excellent"))
                .alias("credit_score_category")
            )

        # Activity score (based on products, active membership, credit card)
        if all(col in self.df.columns for col in ["num_of_products", "is_active_member", "has_credit_card"]):
            self.df = self.df.with_columns(
                (
                    pl.col("num_of_products").cast(pl.Float64) * 25
                    + pl.col("is_active_member").cast(pl.Int8) * 30
                    + pl.col("has_credit_card").cast(pl.Int8) * 20
                ).alias("engagement_score")
            )

        # Zero balance indicator (risk factor)
        if "balance" in self.df.columns:
            self.df = self.df.with_columns(
                (pl.col("balance") == 0).cast(pl.Int8).alias("zero_balance")
            )

        # High value customer indicator
        if "balance" in self.df.columns and "estimated_salary" in self.df.columns:
            balance_threshold = self.df["balance"].quantile(0.75)
            salary_threshold = self.df["estimated_salary"].quantile(0.75)

            self.df = self.df.with_columns(
                ((pl.col("balance") > balance_threshold) | (pl.col("estimated_salary") > salary_threshold))
                .cast(pl.Int8)
                .alias("high_value_customer")
            )

        # Generate synthetic signup date based on tenure
        if "tenure" in self.df.columns:
            today = date.today()
            self.df = self.df.with_columns(
                pl.lit(today).sub(pl.duration(days=pl.col("tenure") * 365)).alias("signup_date")
            )

        logger.info(f"Derived features added. New shape: {self.df.shape}")
        return self

    def get_processed_data(self) -> pl.DataFrame:
        """
        Get the processed DataFrame.

        Returns:
            Processed DataFrame
        """
        return self.df

    def get_summary(self) -> dict:
        """
        Get summary of preprocessing operations.

        Returns:
            Dictionary with summary statistics
        """
        return {
            "original_shape": self.original_shape,
            "final_shape": self.df.shape,
            "columns": self.df.columns,
            "dtypes": {col: str(dtype) for col, dtype in zip(self.df.columns, self.df.dtypes)},
            "null_counts": {col: self.df[col].null_count() for col in self.df.columns},
        }


def preprocess_churn_data(
    df: pl.DataFrame,
    handle_missing: str = "drop",
    handle_outliers: bool = True,
) -> Tuple[pl.DataFrame, dict]:
    """
    Preprocess bank churn data with standard pipeline.

    Args:
        df: Input DataFrame
        handle_missing: Strategy for missing values
        handle_outliers: Whether to handle outliers

    Returns:
        Tuple of (processed DataFrame, summary dict)
    """
    preprocessor = ChurnDataPreprocessor(df)

    # Apply preprocessing steps
    preprocessor.handle_missing_values(strategy=handle_missing)
    preprocessor.convert_boolean_columns()

    if handle_outliers:
        preprocessor.detect_and_handle_outliers(method="iqr", threshold=1.5)

    preprocessor.add_derived_features()

    # Get processed data and summary
    processed_df = preprocessor.get_processed_data()
    summary = preprocessor.get_summary()

    return processed_df, summary
