"""Customer model."""

from datetime import date, datetime
from typing import Optional

from sqlalchemy import Boolean, Date, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database.models.base import Base, TimestampMixin


class Customer(Base, TimestampMixin):
    """Customer table storing bank customer information and churn status."""

    __tablename__ = "customers"

    # Primary Key
    customer_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Basic Customer Information
    credit_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    geography: Mapped[Optional[str]] = mapped_column(String(50), nullable=True, index=True)
    gender: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    age: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    tenure: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # Years with bank

    # Financial Information
    balance: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    estimated_salary: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Product Information
    num_of_products: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    has_credit_card: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    is_active_member: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)

    # Churn Status (target variable)
    exited: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True, index=True)

    # Derived fields (calculated during preprocessing)
    balance_salary_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    age_group: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    tenure_age_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    products_per_tenure: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    engagement_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Date fields
    signup_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    last_interaction_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)

    # Relationships
    usage_history = relationship("CustomerUsageHistory", back_populates="customer", cascade="all, delete-orphan")
    interactions = relationship("CustomerInteraction", back_populates="customer", cascade="all, delete-orphan")
    predictions = relationship("ModelPrediction", back_populates="customer", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Customer(id={self.customer_id}, geography={self.geography}, exited={self.exited})>"
