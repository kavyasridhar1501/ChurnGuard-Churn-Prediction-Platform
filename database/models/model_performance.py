"""Model performance tracking model."""

from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, DateTime, Float, Integer, JSON, String
from sqlalchemy.orm import Mapped, mapped_column

from database.models.base import Base, TimestampMixin


class ModelPerformance(Base, TimestampMixin):
    """Model performance tracking table."""

    __tablename__ = "model_performance"

    # Primary Key
    performance_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Model Information
    model_version: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)

    # Performance Metrics - Classification
    accuracy: Mapped[float] = mapped_column(Float, nullable=False)
    precision: Mapped[float] = mapped_column(Float, nullable=False)
    recall: Mapped[float] = mapped_column(Float, nullable=False)
    f1_score: Mapped[float] = mapped_column(Float, nullable=False)
    auc_roc: Mapped[float] = mapped_column(Float, nullable=False)
    auc_pr: Mapped[float] = mapped_column(Float, nullable=False)

    # Training Information
    training_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
    )
    training_samples: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    test_samples: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Model Status
    is_production: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    deployed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # Additional Metadata
    hyperparameters: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    feature_importance: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    def __repr__(self) -> str:
        return f"<ModelPerformance(id={self.performance_id}, version={self.model_version}, auc_roc={self.auc_roc:.3f})>"
