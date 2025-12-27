"""Advanced LightGBM fine-tuning with extensive hyperparameter optimization.

This script performs aggressive hyperparameter tuning on LightGBM using Optuna
to improve upon the baseline model performance.
"""

import asyncio
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import lightgbm as lgb
import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

from database.models.model_performance import ModelPerformance
from src.db.connection import AsyncSessionLocal
from src.models.evaluate import ModelEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set MLflow tracking URI
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


class LightGBMFineTuner:
    """Advanced LightGBM hyperparameter tuning with Optuna."""

    def __init__(
        self,
        n_trials: int = 200,
        cv_folds: int = 5,
        experiment_name: str = "lightgbm-finetuning",
        model_save_path: str = "./models",
        random_state: int = 42,
    ):
        """
        Initialize LightGBM fine-tuner.

        Args:
            n_trials: Number of Optuna optimization trials
            cv_folds: Number of cross-validation folds
            experiment_name: MLflow experiment name
            model_save_path: Path to save trained models
            random_state: Random seed for reproducibility
        """
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.experiment_name = experiment_name
        self.model_save_path = Path(model_save_path)
        self.model_save_path.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state

        # Set MLflow experiment
        try:
            mlflow.set_experiment(experiment_name)
            logger.info(f"Using MLflow experiment: {experiment_name}")
        except Exception as e:
            logger.warning(f"Could not set MLflow experiment: {e}")

        self.best_model = None
        self.best_params = None
        self.best_score = 0.0
        self.study = None

    def prepare_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        handle_imbalance: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training.

        Args:
            X: Feature matrix
            y: Target vector
            test_size: Proportion of test set
            handle_imbalance: Whether to apply SMOTE

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info(f"Preparing data: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"Class distribution: {np.bincount(y.astype(int))}")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y,
        )

        logger.info(f"Train set: {X_train.shape[0]} samples")
        logger.info(f"Test set: {X_test.shape[0]} samples")

        # Handle class imbalance with SMOTE
        if handle_imbalance:
            logger.info("Applying SMOTE to handle class imbalance...")
            smote = SMOTE(random_state=self.random_state)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            logger.info(f"After SMOTE - Train set: {X_train.shape[0]} samples")
            logger.info(f"Class distribution: {np.bincount(y_train.astype(int))}")

        return X_train, X_test, y_train, y_test

    def objective(
        self,
        trial: optuna.Trial,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> float:
        """
        Optuna objective function for hyperparameter optimization.

        Args:
            trial: Optuna trial object
            X_train: Training features
            y_train: Training target

        Returns:
            Cross-validation AUC-ROC score
        """
        # Define hyperparameter search space
        params = {
            # Tree structure parameters
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-5, 10.0, log=True),

            # Learning parameters
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),

            # Regularization parameters
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),

            # Sampling parameters
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "subsample_freq": trial.suggest_int("subsample_freq", 0, 7),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),

            # Additional parameters
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
            "path_smooth": trial.suggest_float("path_smooth", 0.0, 1.0),

            # Fixed parameters
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "class_weight": "balanced",
            "random_state": self.random_state,
            "verbose": -1,
            "n_jobs": -1,
        }

        # Create model
        model = lgb.LGBMClassifier(**params)

        # Perform cross-validation
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        cv_scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1,
        )

        mean_score = cv_scores.mean()
        std_score = cv_scores.std()

        # Log trial to MLflow
        with mlflow.start_run(nested=True):
            mlflow.log_params(params)
            mlflow.log_metric("cv_auc_roc_mean", mean_score)
            mlflow.log_metric("cv_auc_roc_std", std_score)

        logger.info(
            f"Trial {trial.number}: AUC-ROC = {mean_score:.4f} (+/- {std_score:.4f})"
        )

        return mean_score

    def optimize(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Run Optuna hyperparameter optimization.

        Args:
            X_train: Training features
            y_train: Training target

        Returns:
            Dictionary with optimization results
        """
        logger.info("=" * 60)
        logger.info("Starting Hyperparameter Optimization")
        logger.info(f"Number of trials: {self.n_trials}")
        logger.info(f"Cross-validation folds: {self.cv_folds}")
        logger.info("=" * 60)

        # Create study
        self.study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
        )

        # Optimize
        with mlflow.start_run(run_name=f"lightgbm_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            self.study.optimize(
                lambda trial: self.objective(trial, X_train, y_train),
                n_trials=self.n_trials,
                show_progress_bar=True,
            )

            # Log best results
            self.best_params = self.study.best_params
            self.best_score = self.study.best_value

            mlflow.log_params(self.best_params)
            mlflow.log_metric("best_cv_auc_roc", self.best_score)

        logger.info("\n" + "=" * 60)
        logger.info("Optimization Complete!")
        logger.info(f"Best CV AUC-ROC: {self.best_score:.4f}")
        logger.info("=" * 60)
        logger.info("Best Hyperparameters:")
        for param, value in self.best_params.items():
            logger.info(f"  {param}: {value}")

        return {
            "best_params": self.best_params,
            "best_score": self.best_score,
            "n_trials": len(self.study.trials),
        }

    def train_final_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: list,
    ) -> Dict[str, Any]:
        """
        Train final model with best hyperparameters.

        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
            feature_names: List of feature names

        Returns:
            Dictionary with model and metrics
        """
        logger.info("\n" + "=" * 60)
        logger.info("Training Final Model with Best Hyperparameters")
        logger.info("=" * 60)

        if self.best_params is None:
            raise ValueError("No best parameters found. Run optimize() first.")

        # Add fixed parameters
        final_params = {
            **self.best_params,
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "class_weight": "balanced",
            "random_state": self.random_state,
            "verbose": -1,
            "n_jobs": -1,
        }

        # Train model with early stopping
        self.best_model = lgb.LGBMClassifier(**final_params)

        # Use validation set for early stopping
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, test_size=0.15, random_state=self.random_state, stratify=y_train
        )

        self.best_model.fit(
            X_train_split,
            y_train_split,
            eval_set=[(X_val, y_val)],
            eval_metric="auc",
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=100),
            ],
        )

        # Evaluate on test set
        y_pred = self.best_model.predict(X_test)
        y_pred_proba = self.best_model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "auc_roc": roc_auc_score(y_test, y_pred_proba),
            "auc_pr": average_precision_score(y_test, y_pred_proba),
        }

        logger.info("\nFinal Model Test Set Performance:")
        logger.info(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
        logger.info(f"  AUC-PR: {metrics['auc_pr']:.4f}")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall: {metrics['recall']:.4f}")
        logger.info(f"  F1-Score: {metrics['f1']:.4f}")

        # Log to MLflow
        with mlflow.start_run(run_name=f"lightgbm_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            mlflow.log_params(final_params)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(self.best_model, "model")

        return {
            "model": self.best_model,
            "metrics": metrics,
            "params": final_params,
        }

    def save_model(self, backup: bool = True) -> Tuple[Path, Path]:
        """
        Save the fine-tuned model as production model.

        Args:
            backup: Whether to backup existing production model

        Returns:
            Tuple of (production_model_path, backup_path)
        """
        if self.best_model is None:
            raise ValueError("No model to save. Train final model first.")

        production_path = self.model_save_path / "best_model_production.pkl"
        backup_path = None

        # Backup existing production model
        if backup and production_path.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.model_save_path / f"best_model_production_backup_{timestamp}.pkl"

            # Copy existing model to backup
            import shutil
            shutil.copy(production_path, backup_path)
            logger.info(f"Backed up existing model to: {backup_path}")

        # Save fine-tuned model as production
        joblib.dump(self.best_model, production_path)
        logger.info(f"Fine-tuned model saved as production model: {production_path}")

        return production_path, backup_path

    def compare_with_baseline(
        self,
        baseline_model_path: str,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Compare fine-tuned model with baseline model.

        Args:
            baseline_model_path: Path to baseline model (usually the backup)
            X_test: Test features
            y_test: Test target

        Returns:
            Comparison results
        """
        logger.info("\n" + "=" * 60)
        logger.info("Comparing with Previous Production Model")
        logger.info("=" * 60)

        # Load baseline model
        try:
            baseline_model = joblib.load(baseline_model_path)
            logger.info(f"Loaded previous model from: {baseline_model_path}")
        except FileNotFoundError:
            logger.warning(f"Previous model not found at {baseline_model_path}")
            logger.info("Skipping comparison (this might be the first model training)")
            return {"error": "Previous model not found"}

        # Evaluate baseline
        y_pred_baseline = baseline_model.predict(X_test)
        y_pred_proba_baseline = baseline_model.predict_proba(X_test)[:, 1]

        baseline_metrics = {
            "accuracy": accuracy_score(y_test, y_pred_baseline),
            "precision": precision_score(y_test, y_pred_baseline),
            "recall": recall_score(y_test, y_pred_baseline),
            "f1": f1_score(y_test, y_pred_baseline),
            "auc_roc": roc_auc_score(y_test, y_pred_proba_baseline),
            "auc_pr": average_precision_score(y_test, y_pred_proba_baseline),
        }

        # Evaluate fine-tuned model
        y_pred_tuned = self.best_model.predict(X_test)
        y_pred_proba_tuned = self.best_model.predict_proba(X_test)[:, 1]

        tuned_metrics = {
            "accuracy": accuracy_score(y_test, y_pred_tuned),
            "precision": precision_score(y_test, y_pred_tuned),
            "recall": recall_score(y_test, y_pred_tuned),
            "f1": f1_score(y_test, y_pred_tuned),
            "auc_roc": roc_auc_score(y_test, y_pred_proba_tuned),
            "auc_pr": average_precision_score(y_test, y_pred_proba_tuned),
        }

        # Calculate improvements
        improvements = {
            metric: tuned_metrics[metric] - baseline_metrics[metric]
            for metric in baseline_metrics.keys()
        }

        # Print comparison
        logger.info("\nModel Comparison:")
        logger.info("-" * 60)
        logger.info(f"{'Metric':<15} {'Baseline':<12} {'Fine-tuned':<12} {'Improvement':<12}")
        logger.info("-" * 60)
        for metric in baseline_metrics.keys():
            improvement_pct = (improvements[metric] / baseline_metrics[metric] * 100) if baseline_metrics[metric] > 0 else 0
            logger.info(
                f"{metric.upper():<15} "
                f"{baseline_metrics[metric]:<12.4f} "
                f"{tuned_metrics[metric]:<12.4f} "
                f"{improvements[metric]:+.4f} ({improvement_pct:+.2f}%)"
            )
        logger.info("-" * 60)

        return {
            "baseline": baseline_metrics,
            "finetuned": tuned_metrics,
            "improvements": improvements,
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

    # Convert to pandas DataFrame
    df = pd.DataFrame(rows, columns=[
        "credit_score", "geography", "gender", "age", "tenure", "balance",
        "num_of_products", "has_credit_card", "is_active_member",
        "estimated_salary", "balance_salary_ratio", "tenure_age_ratio",
        "products_per_tenure", "engagement_score", "exited",
    ])

    # One-hot encode categorical variables
    df = pd.get_dummies(df, columns=["geography", "gender"], drop_first=False)

    # Separate features and target
    feature_cols = [col for col in df.columns if col != "exited"]
    X = df[feature_cols].values
    y = df["exited"].values.astype(int)

    return X, y, feature_cols


async def save_model_performance_to_db(metrics: Dict[str, float], params: Dict[str, Any]):
    """Save fine-tuned model performance to database."""
    try:
        model_perf = ModelPerformance(
            model_version="2.0.0",
            model_name="lightgbm_finetuned",
            training_date=datetime.now(),
            accuracy=metrics["accuracy"],
            precision=metrics["precision"],
            recall=metrics["recall"],
            f1_score=metrics["f1"],
            auc_roc=metrics["auc_roc"],
            auc_pr=metrics["auc_pr"],
            hyperparameters=params,
            is_production=True,
        )

        async with AsyncSessionLocal() as session:
            # Mark existing models as non-production
            from sqlalchemy import update
            await session.execute(
                update(ModelPerformance).values(is_production=False)
            )

            # Add new model
            session.add(model_perf)
            await session.commit()

        logger.info("Fine-tuned model performance saved to database")

    except Exception as e:
        logger.error(f"Failed to save model performance: {e}")


async def main():
    """Main fine-tuning pipeline."""
    logger.info("=" * 60)
    logger.info("LightGBM Advanced Fine-Tuning Pipeline")
    logger.info("=" * 60)

    try:
        # Load data
        X, y, feature_names = await load_training_data_from_db()

        # Initialize fine-tuner
        tuner = LightGBMFineTuner(
            n_trials=200,  # More trials for better optimization
            cv_folds=5,
            experiment_name="lightgbm-advanced-tuning",
        )

        # Prepare data
        X_train, X_test, y_train, y_test = tuner.prepare_data(X, y)

        # Run optimization
        optimization_results = tuner.optimize(X_train, y_train)

        # Train final model
        final_results = tuner.train_final_model(
            X_train, y_train, X_test, y_test, feature_names
        )

        # Save model (automatically backs up and replaces production model)
        production_path, backup_path = tuner.save_model(backup=True)

        # Compare with baseline (using the backup)
        comparison = {}
        if backup_path:
            comparison = tuner.compare_with_baseline(str(backup_path), X_test, y_test)
        else:
            logger.info("No previous production model found, skipping comparison")

        # Save to database
        await save_model_performance_to_db(
            final_results["metrics"],
            final_results["params"]
        )

        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("Fine-Tuning Complete!")
        logger.info("=" * 60)
        logger.info(f"Trials completed: {optimization_results['n_trials']}")
        logger.info(f"Best CV score: {optimization_results['best_score']:.4f}")
        logger.info(f"Test AUC-ROC: {final_results['metrics']['auc_roc']:.4f}")
        logger.info(f"Production model updated: {production_path}")
        if backup_path:
            logger.info(f"Previous model backed up to: {backup_path}")
        logger.info("=" * 60)
        logger.info("✓ The fine-tuned model is now in production!")
        logger.info("  Restart your API server to use the new model.")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Fine-tuning pipeline failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
