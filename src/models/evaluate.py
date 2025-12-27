"""Model evaluation with comprehensive metrics and SHAP interpretability."""

import logging
from typing import Any, Dict, Optional

import numpy as np
import shap
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation with SHAP."""

    def __init__(self, model: Any, feature_names: Optional[list] = None):
        """
        Initialize evaluator.

        Args:
            model: Trained model
            feature_names: List of feature names
        """
        self.model = model
        self.feature_names = feature_names
        self.shap_explainer = None
        self.shap_values = None

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model and return comprehensive metrics.

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            Dictionary of metrics
        """
        logger.info("Evaluating model...")

        # Get predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]

        # Classification metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        # ROC and AUC
        auc_roc = roc_auc_score(y_test, y_pred_proba)

        # Precision-Recall AUC
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
        auc_pr = auc(recall_curve, precision_curve)

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "auc_roc": float(auc_roc),
            "auc_pr": float(auc_pr),
            "true_positives": int(tp),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
        }

        logger.info(f"Evaluation complete - AUC-ROC: {auc_roc:.4f}, F1: {f1:.4f}")

        return metrics

    def initialize_shap_explainer(self, X_background: np.ndarray, max_samples: int = 100) -> None:
        """
        Initialize SHAP explainer.

        Args:
            X_background: Background data for SHAP
            max_samples: Maximum samples for background
        """
        logger.info("Initializing SHAP explainer...")

        # Use a subset of data as background for efficiency
        if len(X_background) > max_samples:
            background_sample = shap.sample(X_background, max_samples, random_state=42)
        else:
            background_sample = X_background

        # Try TreeExplainer first (for tree-based models)
        try:
            self.shap_explainer = shap.TreeExplainer(self.model)
            logger.info("Using SHAP TreeExplainer")
        except Exception:
            # Fall back to KernelExplainer for other models
            self.shap_explainer = shap.KernelExplainer(
                self.model.predict_proba,
                background_sample,
            )
            logger.info("Using SHAP KernelExplainer")

    def compute_shap_values(self, X: np.ndarray) -> np.ndarray:
        """
        Compute SHAP values for given data.

        Args:
            X: Input data

        Returns:
            SHAP values
        """
        if self.shap_explainer is None:
            raise ValueError("SHAP explainer not initialized. Call initialize_shap_explainer() first.")

        logger.info(f"Computing SHAP values for {len(X)} samples...")

        # Compute SHAP values
        shap_values = self.shap_explainer.shap_values(X)

        # For binary classification, some explainers return values for both classes
        if isinstance(shap_values, list):
            # Take positive class SHAP values
            shap_values = shap_values[1]

        self.shap_values = shap_values

        logger.info("SHAP values computed successfully")

        return shap_values

    def get_feature_importance(self, method: str = "shap") -> Dict[str, float]:
        """
        Get feature importance.

        Args:
            method: Method to use ('shap', 'model', or 'both')

        Returns:
            Dictionary mapping feature names to importance scores
        """
        importance_dict = {}

        if method in ["model", "both"]:
            # Get model's feature importance if available
            if hasattr(self.model, "feature_importances_"):
                importance_dict["model"] = dict(
                    zip(
                        self.feature_names or range(len(self.model.feature_importances_)),
                        self.model.feature_importances_,
                    )
                )

        if method in ["shap", "both"]:
            # Get SHAP feature importance
            if self.shap_values is not None:
                # Mean absolute SHAP value for each feature
                shap_importance = np.abs(self.shap_values).mean(axis=0)
                importance_dict["shap"] = dict(
                    zip(
                        self.feature_names or range(len(shap_importance)),
                        shap_importance,
                    )
                )

        return importance_dict

    def explain_prediction(
        self,
        X_instance: np.ndarray,
        top_n: int = 10,
    ) -> Dict[str, Any]:
        """
        Explain a single prediction using SHAP.

        Args:
            X_instance: Single instance to explain
            top_n: Number of top features to return

        Returns:
            Dictionary with explanation details
        """
        if self.shap_explainer is None:
            raise ValueError("SHAP explainer not initialized")

        # Ensure X_instance is 2D
        if len(X_instance.shape) == 1:
            X_instance = X_instance.reshape(1, -1)

        # Get prediction
        prediction_proba = self.model.predict_proba(X_instance)[0]
        prediction = self.model.predict(X_instance)[0]

        # Get SHAP values for this instance
        shap_values = self.shap_explainer.shap_values(X_instance)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Positive class

        # Get SHAP values for single instance
        if len(shap_values.shape) > 1:
            shap_values = shap_values[0]

        # Get base value (expected value)
        if hasattr(self.shap_explainer, "expected_value"):
            expected_value = self.shap_explainer.expected_value
            if isinstance(expected_value, (list, np.ndarray)):
                expected_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]
        else:
            expected_value = 0.0

        # Create feature contributions
        feature_contributions = []
        feature_names = self.feature_names or [f"feature_{i}" for i in range(len(shap_values))]

        for i, (feat_name, shap_val, feat_val) in enumerate(
            zip(feature_names, shap_values, X_instance[0])
        ):
            feature_contributions.append(
                {
                    "feature": feat_name,
                    "value": float(feat_val),
                    "shap_value": float(shap_val),
                    "abs_shap_value": float(abs(shap_val)),
                }
            )

        # Sort by absolute SHAP value
        feature_contributions.sort(key=lambda x: x["abs_shap_value"], reverse=True)

        # Get top features
        top_features = feature_contributions[:top_n]

        explanation = {
            "prediction": int(prediction),
            "probability_churned": float(prediction_proba[1]),
            "probability_retained": float(prediction_proba[0]),
            "base_value": float(expected_value),
            "top_risk_factors": top_features,
            "all_contributions": feature_contributions,
        }

        return explanation


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_train: Optional[np.ndarray] = None,
    feature_names: Optional[list] = None,
    use_shap: bool = True,
) -> Dict[str, Any]:
    """
    Convenience function to evaluate a model.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        X_train: Training features (for SHAP background)
        feature_names: List of feature names
        use_shap: Whether to compute SHAP values

    Returns:
        Dictionary with evaluation results
    """
    evaluator = ModelEvaluator(model, feature_names=feature_names)

    # Get metrics
    metrics = evaluator.evaluate(X_test, y_test)

    results = {"metrics": metrics}

    # Add SHAP if requested
    if use_shap and X_train is not None:
        try:
            evaluator.initialize_shap_explainer(X_train)
            shap_values = evaluator.compute_shap_values(X_test)
            feature_importance = evaluator.get_feature_importance(method="both")

            results["shap_values"] = shap_values
            results["feature_importance"] = feature_importance
            results["evaluator"] = evaluator
        except Exception as e:
            logger.warning(f"Could not compute SHAP values: {e}")

    return results
