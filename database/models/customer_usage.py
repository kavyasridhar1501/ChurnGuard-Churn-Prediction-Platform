"""Customer banking activity history model."""

from datetime import date
from typing import Optional

from sqlalchemy import Date, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database.models.base import Base, TimestampMixin


class CustomerUsageHistory(Base, TimestampMixin):
    """Customer banking activity history table for tracking transactions over time."""

    __tablename__ = "customer_usage_history"

    # Primary Key
    usage_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Foreign Key
    customer_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("customers.customer_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Activity Period
    activity_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    activity_period: Mapped[str] = mapped_column(String(20), nullable=False)  # daily, weekly, monthly

    # Transaction Metrics
    transaction_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    total_transaction_amount: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    avg_transaction_amount: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Balance Tracking
    balance_start: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    balance_end: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    balance_change: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Activity Flags
    is_active_period: Mapped[Optional[bool]] = mapped_column(nullable=True)
    product_usage_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Relationship
    customer = relationship("Customer", back_populates="usage_history")

    def __repr__(self) -> str:
        return f"<CustomerUsageHistory(id={self.usage_id}, customer_id={self.customer_id}, date={self.activity_date})>"
