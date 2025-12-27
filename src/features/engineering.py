"""Feature engineering for bank customer churn prediction."""

import logging
from typing import List, Tuple

import numpy as np
import polars as pl
from sklearn.preprocessing import LabelEncoder, StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChurnFeatureEngineer:
    """Feature engineering for bank customer churn prediction model."""

    def __init__(self):
        """Initialize feature engineer."""
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.categorical_features = []
        self.numerical_features = []

    def create_financial_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Create financial-based features.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with additional financial features
        """
        logger.info("Creating financial features...")

        # Balance to salary ratio
        if "balance" in df.columns and "estimated_salary" in df.columns:
            df = df.with_columns(
                (pl.col("balance") / pl.col("estimated_salary").clip(lower_bound=1)).alias("balance_salary_ratio")
            )

        # Balance bins (categorize balance levels)
        if "balance" in df.columns:
            df = df.with_columns(
                pl.when(pl.col("balance") == 0)
                .then(0)
                .when(pl.col("balance") < 50000)
                .then(1)
                .when(pl.col("balance") < 100000)
                .then(2)
                .when(pl.col("balance") < 150000)
                .then(3)
                .otherwise(4)
                .alias("balance_category")
            )

        # Salary bins
        if "estimated_salary" in df.columns:
            df = df.with_columns(
                pl.when(pl.col("estimated_salary") < 50000)
                .then(0)
                .when(pl.col("estimated_salary") < 100000)
                .then(1)
                .when(pl.col("estimated_salary") < 150000)
                .then(2)
                .otherwise(3)
                .alias("salary_category")
            )

        # Zero balance indicator (strong churn signal)
        if "balance" in df.columns:
            df = df.with_columns(
                (pl.col("balance") == 0).cast(pl.Int8).alias("zero_balance_flag")
            )

        return df

    def create_demographic_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Create demographic features.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with demographic features
        """
        logger.info("Creating demographic features...")

        # Age groups
        if "age" in df.columns:
            df = df.with_columns(
                [
                    pl.when(pl.col("age") < 30)
                    .then(0)
                    .when(pl.col("age") < 40)
                    .then(1)
                    .when(pl.col("age") < 50)
                    .then(2)
                    .when(pl.col("age") < 60)
                    .then(3)
                    .otherwise(4)
                    .alias("age_group_numeric"),
                    # Is senior citizen
                    (pl.col("age") >= 60).cast(pl.Int8).alias("is_senior"),
                ]
            )

        # Geography-Gender interaction
        if "geography" in df.columns and "gender" in df.columns:
            df = df.with_columns(
                (pl.col("geography") + "_" + pl.col("gender")).alias("geo_gender_combo")
            )

        return df

    def create_product_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Create product-related features.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with product features
        """
        logger.info("Creating product features...")

        # Product engagement score
        if "num_of_products" in df.columns:
            df = df.with_columns(
                [
                    # Single product customer (higher churn risk)
                    (pl.col("num_of_products") == 1).cast(pl.Int8).alias("single_product"),
                    # Multiple products (lower churn risk)
                    (pl.col("num_of_products") >= 3).cast(pl.Int8).alias("multi_product"),
                ]
            )

        # Product-tenure interaction
        if "num_of_products" in df.columns and "tenure" in df.columns:
            df = df.with_columns(
                (pl.col("num_of_products") / pl.col("tenure").clip(lower_bound=1)).alias("products_per_tenure_year")
            )

        # Activity flags combination
        if "has_credit_card" in df.columns and "is_active_member" in df.columns:
            df = df.with_columns(
                [
                    # Both flags true (highly engaged)
                    (pl.col("has_credit_card") & pl.col("is_active_member")).cast(pl.Int8).alias("fully_engaged"),
                    # Neither flag true (disengaged)
                    (~pl.col("has_credit_card") & ~pl.col("is_active_member")).cast(pl.Int8).alias("disengaged"),
                    # Has card but not active
                    (pl.col("has_credit_card") & ~pl.col("is_active_member")).cast(pl.Int8).alias("inactive_cardholder"),
                ]
            )

        return df

    def create_tenure_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Create tenure-based features.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with tenure features
        """
        logger.info("Creating tenure features...")

        if "tenure" in df.columns:
            # Tenure categories
            df = df.with_columns(
                [
                    pl.when(pl.col("tenure") == 0)
                    .then(0)  # new customer
                    .when(pl.col("tenure") <= 2)
                    .then(1)  # recent
                    .when(pl.col("tenure") <= 5)
                    .then(2)  # established
                    .otherwise(3)  # loyal
                    .alias("tenure_category"),
                    # New customer flag (higher churn risk)
                    (pl.col("tenure") == 0).cast(pl.Int8).alias("new_customer"),
                    # Long-term customer flag
                    (pl.col("tenure") >= 5).cast(pl.Int8).alias("long_term_customer"),
                ]
            )

            # Tenure to age ratio
            if "age" in df.columns:
                df = df.with_columns(
                    (pl.col("tenure") / pl.col("age").clip(lower_bound=18)).alias("tenure_age_ratio")
                )

        return df

    def create_credit_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Create credit score features.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with credit features
        """
        logger.info("Creating credit score features...")

        if "credit_score" in df.columns:
            df = df.with_columns(
                [
                    # Credit score categories
                    pl.when(pl.col("credit_score") < 400)
                    .then(0)  # very poor
                    .when(pl.col("credit_score") < 550)
                    .then(1)  # poor
                    .when(pl.col("credit_score") < 650)
                    .then(2)  # fair
                    .when(pl.col("credit_score") < 750)
                    .then(3)  # good
                    .otherwise(4)  # excellent
                    .alias("credit_category"),
                    # Low credit score flag
                    (pl.col("credit_score") < 500).cast(pl.Int8).alias("low_credit_score"),
                    # High credit score flag
                    (pl.col("credit_score") >= 750).cast(pl.Int8).alias("high_credit_score"),
                ]
            )

        return df

    def create_interaction_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Create interaction features between variables.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with interaction features
        """
        logger.info("Creating interaction features...")

        # Balance * Number of products
        if "balance" in df.columns and "num_of_products" in df.columns:
            df = df.with_columns(
                (pl.col("balance") * pl.col("num_of_products") / 100000).alias("balance_products_interaction")
            )

        # Age * Tenure
        if "age" in df.columns and "tenure" in df.columns:
            df = df.with_columns(
                (pl.col("age") * pl.col("tenure") / 100).alias("age_tenure_interaction")
            )

        # Credit score * Balance
        if "credit_score" in df.columns and "balance" in df.columns:
            df = df.with_columns(
                (pl.col("credit_score") * pl.col("balance") / 1000000).alias("credit_balance_interaction")
            )

        # Active member * Products
        if "is_active_member" in df.columns and "num_of_products" in df.columns:
            df = df.with_columns(
                pl.when(pl.col("is_active_member"))
                .then(pl.col("num_of_products"))
                .otherwise(0)
                .alias("active_products")
            )

        return df

    def prepare_features_for_modeling(
        self,
        df: pl.DataFrame,
        target_col: str = "exited",
        exclude_cols: List[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare features for modeling.

        Args:
            df: Input DataFrame
            target_col: Name of target column
            exclude_cols: Columns to exclude from features

        Returns:
            Tuple of (X, y, feature_names)
        """
        logger.info("Preparing features for modeling...")

        if exclude_cols is None:
            exclude_cols = [
                "customer_id",
                "row_number",
                "surname",
                "signup_date",
                "last_interaction_date",
                "created_at",
                "updated_at",
                "age_group",  # Using numeric version instead
                "geo_gender_combo",  # Will be encoded separately
                "credit_score_category",  # Using numeric version instead
            ]

        # Separate target
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame")

        # Identify feature columns
        feature_cols = [col for col in df.columns if col not in exclude_cols + [target_col]]

        # Convert to pandas for sklearn compatibility
        df_pandas = df.to_pandas()

        # Separate categorical and numerical features
        self.categorical_features = [
            col
            for col in feature_cols
            if df_pandas[col].dtype == "object" or df_pandas[col].dtype == "bool" or col.endswith("_category")
        ]

        self.numerical_features = [col for col in feature_cols if col not in self.categorical_features]

        logger.info(f"Categorical features ({len(self.categorical_features)}): {self.categorical_features}")
        logger.info(f"Numerical features ({len(self.numerical_features)}): {self.numerical_features}")

        # Encode categorical variables
        X_parts = []

        if self.categorical_features:
            for col in self.categorical_features:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    X_parts.append(self.label_encoders[col].fit_transform(df_pandas[col].fillna("missing")))
                else:
                    X_parts.append(self.label_encoders[col].transform(df_pandas[col].fillna("missing")))

        # Add numerical features
        if self.numerical_features:
            X_numerical = df_pandas[self.numerical_features].fillna(0).values
            X_parts.append(X_numerical)

        # Combine all features
        if len(X_parts) > 1:
            X = np.hstack(X_parts)
        elif len(X_parts) == 1:
            X = X_parts[0] if len(X_parts[0].shape) == 2 else X_parts[0].reshape(-1, 1)
        else:
            raise ValueError("No features available for modeling")

        # Get target
        if df_pandas[target_col].dtype == bool:
            y = df_pandas[target_col].astype(int).values
        else:
            y = df_pandas[target_col].values

        # Store feature names
        self.feature_names = self.categorical_features + self.numerical_features

        logger.info(f"Prepared {X.shape[1]} features for {X.shape[0]} samples")

        return X, y, self.feature_names

    def fit_transform(self, df: pl.DataFrame, target_col: str = "exited") -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Fit and transform the dataset with all feature engineering steps.

        Args:
            df: Input DataFrame
            target_col: Name of target column

        Returns:
            Tuple of (X, y, feature_names)
        """
        logger.info("Starting feature engineering pipeline for bank dataset...")

        # Create all engineered features
        df = self.create_financial_features(df)
        df = self.create_demographic_features(df)
        df = self.create_product_features(df)
        df = self.create_tenure_features(df)
        df = self.create_credit_features(df)
        df = self.create_interaction_features(df)

        # Prepare for modeling
        X, y, feature_names = self.prepare_features_for_modeling(df, target_col=target_col)

        # Scale numerical features
        num_feature_indices = [i for i, feat in enumerate(feature_names) if feat in self.numerical_features]

        if num_feature_indices:
            X[:, num_feature_indices] = self.scaler.fit_transform(X[:, num_feature_indices])
            logger.info("Scaled numerical features")

        logger.info("Feature engineering pipeline completed")

        return X, y, feature_names


def engineer_features(df: pl.DataFrame, target_col: str = "exited") -> Tuple[np.ndarray, np.ndarray, List[str], ChurnFeatureEngineer]:
    """
    Convenience function for feature engineering.

    Args:
        df: Input DataFrame
        target_col: Name of target column (default: 'exited' for bank dataset)

    Returns:
        Tuple of (X, y, feature_names, engineer)
    """
    engineer = ChurnFeatureEngineer()
    X, y, feature_names = engineer.fit_transform(df, target_col=target_col)

    return X, y, feature_names, engineer
