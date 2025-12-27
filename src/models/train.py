"""Model training pipeline with MLflow tracking and Optuna optimization."""

import asyncio
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import lightgbm as lgb
import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from database.models.model_performance import ModelPerformance
from src.db.connection import AsyncSessionLocal
from src.models.evaluate import ModelEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set MLflow tracking URI
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


class ChurnModelTrainer:
    """Trainer for churn prediction models."""

    def __init__(
        self,
        experiment_name: str = "churn-prediction",
        model_save_path: str = "./models",
    ):
        """
        Initialize model trainer.

        Args:
            experiment_name: MLflow experiment name
            model_save_path: Path to save trained models
        """
        self.experiment_name = experiment_name
        self.model_save_path = Path(model_save_path)
        self.model_save_path.mkdir(parents=True, exist_ok=True)

        # Set or create MLflow experiment
        try:
            mlflow.set_experiment(experiment_name)
            logger.info(f"Using MLflow experiment: {experiment_name}")
        except Exception as e:
            logger.warning(f"Could not set MLflow experiment: {e}")

        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.best_score = 0.0

    def prepare_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
        handle_imbalance: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training with train/val/test split.

        Args:
            X: Feature matrix
            y: Target vector
            test_size: Proportion of test set
            val_size: Proportion of validation set from training data
            random_state: Random seed
            handle_imbalance: Whether to apply SMOTE

        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        logger.info(f"Preparing data: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"Class distribution: {np.bincount(y.astype(int))}")

        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )

        # Second split: train vs val
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=val_ratio,
            random_state=random_state,
            stratify=y_temp,
        )

        logger.info(f"Train set: {X_train.shape[0]} samples")
        logger.info(f"Validation set: {X_val.shape[0]} samples")
        logger.info(f"Test set: {X_test.shape[0]} samples")

        # Handle class imbalance with SMOTE
        if handle_imbalance:
            logger.info("Applying SMOTE to handle class imbalance...")
            smote = SMOTE(random_state=random_state)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            logger.info(f"After SMOTE - Train set: {X_train.shape[0]} samples")
            logger.info(f"Class distribution: {np.bincount(y_train.astype(int))}")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def train_logistic_regression(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        params: Optional[Dict[str, Any]] = None,
    ) -> LogisticRegression:
        """Train Logistic Regression model."""
        logger.info("Training Logistic Regression...")

        if params is None:
            params = {
                "C": 1.0,
                "max_iter": 1000,
                "random_state": 42,
                "class_weight": "balanced",
            }

        model = LogisticRegression(**params)
        model.fit(X_train, y_train)

        return model

    def train_random_forest(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        params: Optional[Dict[str, Any]] = None,
    ) -> RandomForestClassifier:
        """Train Random Forest model."""
        logger.info("Training Random Forest...")

        if params is None:
            params = {
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "random_state": 42,
                "class_weight": "balanced",
                "n_jobs": -1,
            }

        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        return model

    def train_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> xgb.XGBClassifier:
        """Train XGBoost model."""
        logger.info("Training XGBoost...")

        if params is None:
            # Calculate scale_pos_weight for imbalanced data
            neg_count = np.sum(y_train == 0)
            pos_count = np.sum(y_train == 1)
            scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

            params = {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "scale_pos_weight": scale_pos_weight,
                "random_state": 42,
                "eval_metric": "auc",
            }

        model = xgb.XGBClassifier(**params)

        # Use early stopping if validation set is provided
        if X_val is not None and y_val is not None:
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
        else:
            model.fit(X_train, y_train)

        return model

    def train_lightgbm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> lgb.LGBMClassifier:
        """Train LightGBM model."""
        logger.info("Training LightGBM...")

        if params is None:
            params = {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "num_leaves": 31,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
                "class_weight": "balanced",
                "verbose": -1,
            }

        model = lgb.LGBMClassifier(**params)

        # Use early stopping if validation set is provided
        if X_val is not None and y_val is not None:
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                eval_metric="auc",
                callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)],
            )
        else:
            model.fit(X_train, y_train)

        return model

    def train_all_models(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        y_test: np.ndarray,
        feature_names: list,
    ) -> Dict[str, Any]:
        """
        Train all models and track with MLflow.

        Args:
            X_train, X_val, X_test: Feature matrices
            y_train, y_val, y_test: Target vectors
            feature_names: List of feature names

        Returns:
            Dictionary with model results
        """
        results = {}

        models_to_train = {
            "logistic_regression": lambda: self.train_logistic_regression(X_train, y_train),
            "random_forest": lambda: self.train_random_forest(X_train, y_train),
            "xgboost": lambda: self.train_xgboost(X_train, y_train, X_val, y_val),
            "lightgbm": lambda: self.train_lightgbm(X_train, y_train, X_val, y_val),
        }

        for model_name, train_func in models_to_train.items():
            logger.info(f"\n{'=' * 50}")
            logger.info(f"Training {model_name}")
            logger.info(f"{'=' * 50}")

            try:
                with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                    # Train model
                    model = train_func()

                    # Evaluate model
                    evaluator = ModelEvaluator(model, feature_names=feature_names)
                    metrics = evaluator.evaluate(X_test, y_test)

                    # Log parameters
                    mlflow.log_params(model.get_params())

                    # Log metrics
                    mlflow.log_metrics(metrics)

                    # Log model
                    mlflow.sklearn.log_model(model, "model")

                    # Save model locally
                    model_path = self.model_save_path / f"{model_name}_latest.pkl"
                    joblib.dump(model, model_path)
                    logger.info(f"Model saved to {model_path}")

                    # Store results
                    results[model_name] = {
                        "model": model,
                        "metrics": metrics,
                        "model_path": str(model_path),
                    }

                    # Track best model
                    if metrics["auc_roc"] > self.best_score:
                        self.best_score = metrics["auc_roc"]
                        self.best_model = model
                        self.best_model_name = model_name
                        logger.info(f"New best model: {model_name} with AUC-ROC: {self.best_score:.4f}")

                    logger.info(f"{model_name} - AUC-ROC: {metrics['auc_roc']:.4f}, F1: {metrics['f1']:.4f}")

            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                results[model_name] = {"error": str(e)}

        return results

    def save_best_model(self, version: str = "production") -> Path:
        """Save the best performing model."""
        if self.best_model is None:
            raise ValueError("No best model found. Train models first.")

        model_path = self.model_save_path / f"best_model_{version}.pkl"
        joblib.dump(self.best_model, model_path)
        logger.info(f"Best model ({self.best_model_name}) saved to {model_path}")

        return model_path


def train_churn_models(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list,
    experiment_name: str = "churn-prediction",
) -> Dict[str, Any]:
    """
    Convenience function to train churn prediction models.

    Args:
        X: Feature matrix
        y: Target vector
        feature_names: List of feature names
        experiment_name: MLflow experiment name

    Returns:
        Dictionary with training results
    """
    trainer = ChurnModelTrainer(experiment_name=experiment_name)

    # Prepare data
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data(X, y)

    # Train all models
    results = trainer.train_all_models(
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        feature_names,
    )

    # Save best model
    trainer.save_best_model()

    return {
        "results": results,
        "best_model_name": trainer.best_model_name,
        "best_score": trainer.best_score,
        "trainer": trainer,
    }


async def load_training_data_from_db() -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Load customer data from PostgreSQL database for training.

    Returns:
        Tuple of (X: features, y: target, feature_names: list of feature names)
    """
    from sqlalchemy import text

    logger.info("Loading training data from database...")

    async with AsyncSessionLocal() as session:
        # Query all customer data
        query = text("""
            SELECT
                credit_score,
                geography,
                gender,
                age,
                tenure,
                balance,
                num_of_products,
                has_credit_card,
                is_active_member,
                estimated_salary,
                balance_salary_ratio,
                tenure_age_ratio,
                products_per_tenure,
                engagement_score,
                exited
            FROM customers
            WHERE exited IS NOT NULL
        """)

        result = await session.execute(query)
        rows = result.fetchall()

    if not rows:
        raise ValueError("No customer data found in database. Please run data ingestion first.")

    logger.info(f"Loaded {len(rows)} customer records from database")

    # Convert to numpy arrays
    df = pd.DataFrame(rows, columns=[
        "credit_score",
        "geography",
        "gender",
        "age",
        "tenure",
        "balance",
        "num_of_products",
        "has_credit_card",
        "is_active_member",
        "estimated_salary",
        "balance_salary_ratio",
        "tenure_age_ratio",
        "products_per_tenure",
        "engagement_score",
        "exited",
    ])

    # Handle categorical variables with one-hot encoding
    df = pd.get_dummies(df, columns=["geography", "gender"], drop_first=False)

    # Separate features and target
    feature_cols = [col for col in df.columns if col != "exited"]
    X = df[feature_cols].values
    y = df["exited"].values.astype(int)

    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Target distribution: {np.bincount(y)}")

    return X, y, feature_cols


async def save_model_performance_to_db(results: Dict[str, Any], data_split_sizes: Dict[str, int]):
    """
    Save model performance metrics to database.

    Args:
        results: Training results dictionary
        data_split_sizes: Dictionary with train/val/test sizes
    """
    try:
        best_model_name = results["best_model_name"]
        best_model_result = results["results"][best_model_name]

        if "error" in best_model_result:
            logger.warning("Best model had errors, skipping database save")
            return

        metrics = best_model_result["metrics"]
        confusion = best_model_result.get("confusion_matrix", {})

        # Create model performance record
        model_perf = ModelPerformance(
            model_version="1.0.0",
            model_name=best_model_name,
            training_date=datetime.now(),
            training_samples=data_split_sizes.get("train", None),
            test_samples=data_split_sizes.get("test", None),
            accuracy=metrics.get("accuracy", 0.0),
            precision=metrics.get("precision", 0.0),
            recall=metrics.get("recall", 0.0),
            f1_score=metrics.get("f1", 0.0),
            auc_roc=metrics.get("auc_roc", 0.0),
            auc_pr=metrics.get("auc_pr", 0.0),
            is_production=True,
        )

        async with AsyncSessionLocal() as session:
            # Mark all existing models as non-production
            from sqlalchemy import update
            await session.execute(
                update(ModelPerformance).values(is_production=False)
            )

            # Add new model performance
            session.add(model_perf)
            await session.commit()

        logger.info("Model performance saved to database successfully")

    except Exception as e:
        logger.error(f"Failed to save model performance to database: {e}")


async def main():
    """Main training pipeline."""
    logger.info("=" * 60)
    logger.info("Starting ChurnGuard Model Training Pipeline")
    logger.info("=" * 60)

    try:
        # Load data from database
        X, y, feature_names = await load_training_data_from_db()

        # Train models
        logger.info("\nTraining models...")
        results = train_churn_models(
            X=X,
            y=y,
            feature_names=feature_names,
            experiment_name="bank-churn-prediction",
        )

        # Print results
        logger.info("\n" + "=" * 60)
        logger.info("Training Complete!")
        logger.info("=" * 60)
        logger.info(f"Best Model: {results['best_model_name']}")
        logger.info(f"Best AUC-ROC Score: {results['best_score']:.4f}")
        logger.info("\nAll Model Results:")

        for model_name, model_result in results["results"].items():
            if "error" in model_result:
                logger.error(f"  {model_name}: ERROR - {model_result['error']}")
            else:
                metrics = model_result["metrics"]
                logger.info(
                    f"  {model_name}: "
                    f"AUC-ROC={metrics['auc_roc']:.4f}, "
                    f"F1={metrics['f1']:.4f}, "
                    f"Accuracy={metrics['accuracy']:.4f}"
                )

        # Save model performance to database
        trainer = results["trainer"]
        data_split_sizes = {
            "train": len(trainer.models[results["best_model_name"]]) if hasattr(trainer, 'models') else 0,
            "val": 0,  # Will be calculated from data split
            "test": 0,
        }
        await save_model_performance_to_db(results, data_split_sizes)

        logger.info("\n" + "=" * 60)
        logger.info("Models saved successfully!")
        logger.info(f"Best model saved to: models/best_model_production.pkl")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
