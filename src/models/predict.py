"""Model prediction utilities."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import polars as pl

from src.models.evaluate import ModelEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChurnPredictor:
    """Churn prediction inference class."""

    def __init__(
        self,
        model_path: str,
        feature_engineer: Optional[Any] = None,
        feature_names: Optional[List[str]] = None,
    ):
        """
        Initialize predictor.

        Args:
            model_path: Path to saved model
            feature_engineer: Feature engineering pipeline
            feature_names: List of feature names
        """
        self.model_path = Path(model_path)
        self.model = self._load_model()
        self.feature_engineer = feature_engineer
        self.feature_names = feature_names
        self.evaluator = None

        if feature_names:
            self.evaluator = ModelEvaluator(self.model, feature_names=feature_names)

        logger.info(f"Predictor initialized with model from {model_path}")

    def _load_model(self) -> Any:
        """Load model from disk."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        model = joblib.load(self.model_path)
        logger.info("Model loaded successfully")

        return model

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Feature matrix

        Returns:
            Binary predictions
        """
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities.

        Args:
            X: Feature matrix

        Returns:
            Probability matrix
        """
        return self.model.predict_proba(X)

    def predict_with_explanation(
        self,
        X: np.ndarray,
        X_background: Optional[np.ndarray] = None,
        top_n: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Predict with SHAP explanations.

        Args:
            X: Feature matrix
            X_background: Background data for SHAP
            top_n: Number of top features to return

        Returns:
            List of prediction dictionaries with explanations
        """
        if self.evaluator is None:
            raise ValueError("Evaluator not initialized. Provide feature_names at initialization.")

        # Initialize SHAP if background data provided
        if X_background is not None and self.evaluator.shap_explainer is None:
            self.evaluator.initialize_shap_explainer(X_background)

        results = []

        for i in range(len(X)):
            X_instance = X[i : i + 1]

            # Get explanation
            explanation = self.evaluator.explain_prediction(X_instance, top_n=top_n)

            results.append(explanation)

        return results

    def predict_dataframe(
        self,
        df: pl.DataFrame,
        return_explanations: bool = False,
        X_background: Optional[np.ndarray] = None,
    ) -> pl.DataFrame:
        """
        Predict on a Polars DataFrame.

        Args:
            df: Input DataFrame
            return_explanations: Whether to include SHAP explanations
            X_background: Background data for SHAP

        Returns:
            DataFrame with predictions
        """
        # Apply feature engineering if available
        if self.feature_engineer is not None:
            X, _, _ = self.feature_engineer.fit_transform(df.clone())
        else:
            # Preprocess the data (one-hot encode categorical variables)
            import pandas as pd

            # Convert to pandas for one-hot encoding (polars doesn't have get_dummies)
            df_pd = df.to_pandas()

            # Add derived features (same as training)
            if "balance" in df_pd.columns and "estimated_salary" in df_pd.columns:
                df_pd["balance_salary_ratio"] = df_pd["balance"] / df_pd["estimated_salary"].clip(lower=1)

            if "tenure" in df_pd.columns and "age" in df_pd.columns:
                df_pd["tenure_age_ratio"] = df_pd["tenure"] / df_pd["age"].clip(lower=18)

            if "num_of_products" in df_pd.columns and "tenure" in df_pd.columns:
                df_pd["products_per_tenure"] = df_pd["num_of_products"] / df_pd["tenure"].clip(lower=1)

            if all(col in df_pd.columns for col in ["num_of_products", "is_active_member", "has_credit_card"]):
                df_pd["engagement_score"] = (
                    df_pd["num_of_products"].astype(float) * 25
                    + df_pd["is_active_member"].astype(int) * 30
                    + df_pd["has_credit_card"].astype(int) * 20
                )

            # One-hot encode geography and gender (same as training)
            if "geography" in df_pd.columns and "gender" in df_pd.columns:
                df_pd = pd.get_dummies(df_pd, columns=["geography", "gender"], drop_first=False)

                # Ensure all expected geography columns exist (even if not in current data)
                for geo in ["France", "Germany", "Spain"]:
                    col_name = f"geography_{geo}"
                    if col_name not in df_pd.columns:
                        df_pd[col_name] = 0

                # Ensure all expected gender columns exist
                for gen in ["Female", "Male"]:
                    col_name = f"gender_{gen}"
                    if col_name not in df_pd.columns:
                        df_pd[col_name] = 0

            # Select only numeric features (exclude the target if present)
            feature_cols = [col for col in df_pd.columns if col not in ["exited", "churn_prediction", "churn_probability", "retention_probability", "risk_segment"]]
            X = df_pd[feature_cols].values

        # Get predictions
        predictions = self.predict(X)
        probabilities = self.predict_proba(X)

        # Create results DataFrame
        result_df = df.clone()
        result_df = result_df.with_columns(
            [
                pl.Series("churn_prediction", predictions),
                pl.Series("churn_probability", probabilities[:, 1]),
                pl.Series("retention_probability", probabilities[:, 0]),
            ]
        )

        # Add risk segment
        result_df = result_df.with_columns(
            pl.when(pl.col("churn_probability") < 0.3)
            .then(pl.lit("low"))
            .when(pl.col("churn_probability") < 0.5)
            .then(pl.lit("medium"))
            .when(pl.col("churn_probability") < 0.7)
            .then(pl.lit("high"))
            .otherwise(pl.lit("critical"))
            .alias("risk_segment")
        )

        # Add explanations if requested
        if return_explanations and self.evaluator is not None:
            explanations = self.predict_with_explanation(X, X_background=X_background)

            # Add top risk factors as JSON
            top_factors = [
                {feat["feature"]: feat["shap_value"] for feat in exp["top_risk_factors"][:5]}
                for exp in explanations
            ]

            result_df = result_df.with_columns(pl.Series("top_risk_factors", top_factors))

        return result_df

    def batch_predict(
        self,
        data_path: str,
        output_path: Optional[str] = None,
        return_explanations: bool = False,
    ) -> pl.DataFrame:
        """
        Batch prediction from CSV file.

        Args:
            data_path: Path to input CSV
            output_path: Path to save output CSV (optional)
            return_explanations: Whether to include SHAP explanations

        Returns:
            DataFrame with predictions
        """
        logger.info(f"Loading data from {data_path}")

        # Load data
        df = pl.read_csv(data_path)

        # Make predictions
        result_df = self.predict_dataframe(df, return_explanations=return_explanations)

        # Save if output path provided
        if output_path:
            result_df.write_csv(output_path)
            logger.info(f"Predictions saved to {output_path}")

        return result_df

    def get_risk_segmentation(self, df: pl.DataFrame) -> Dict[str, Any]:
        """
        Get risk segmentation statistics.

        Args:
            df: DataFrame with predictions

        Returns:
            Dictionary with segmentation stats
        """
        if "risk_segment" not in df.columns:
            raise ValueError("DataFrame must contain 'risk_segment' column")

        total_customers = len(df)

        segmentation = {
            "total_customers": total_customers,
            "segments": {},
        }

        for segment in ["low", "medium", "high", "critical"]:
            count = len(df.filter(pl.col("risk_segment") == segment))
            percentage = (count / total_customers * 100) if total_customers > 0 else 0

            segmentation["segments"][segment] = {
                "count": count,
                "percentage": round(percentage, 2),
            }

        return segmentation


def load_and_predict(
    model_path: str,
    data_path: str,
    feature_engineer: Optional[Any] = None,
    output_path: Optional[str] = None,
) -> pl.DataFrame:
    """
    Convenience function to load model and make predictions.

    Args:
        model_path: Path to saved model
        data_path: Path to input data
        feature_engineer: Feature engineering pipeline
        output_path: Path to save predictions

    Returns:
        DataFrame with predictions
    """
    predictor = ChurnPredictor(model_path, feature_engineer=feature_engineer)
    return predictor.batch_predict(data_path, output_path=output_path)
