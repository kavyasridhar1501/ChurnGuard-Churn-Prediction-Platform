"""Model prediction model."""

from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database.models.base import Base, TimestampMixin


class ModelPrediction(Base, TimestampMixin):
    """Model predictions table for storing churn predictions."""

    __tablename__ = "model_predictions"

    # Primary Key
    prediction_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Foreign Key
    customer_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("customers.customer_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Model Information
    model_version: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)

    # Prediction Details
    prediction_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
    )
    churn_probability: Mapped[float] = mapped_column(Float, nullable=False)
    churn_prediction: Mapped[bool] = mapped_column(nullable=False)  # binary prediction
    prediction_threshold: Mapped[float] = mapped_column(Float, default=0.5)

    # Risk Segmentation
    risk_segment: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
    )  # low, medium, high, critical

    # Feature Importance and Interpretability
    shap_values: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    top_risk_factors: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    feature_contributions: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    # Business Metrics
    estimated_clv: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    retention_cost: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Prediction Metadata
    prediction_context: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
    )  # batch, realtime, scheduled

    # Actual Outcome (for model performance tracking)
    actual_churn: Mapped[Optional[bool]] = mapped_column(nullable=True)
    outcome_date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # Relationship
    customer = relationship("Customer", back_populates="predictions")

    def __repr__(self) -> str:
        return f"<ModelPrediction(id={self.prediction_id}, customer_id={self.customer_id}, probability={self.churn_probability:.2f})>"
