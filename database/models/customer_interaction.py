"""Customer interaction model."""

from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database.models.base import Base, TimestampMixin


class CustomerInteraction(Base, TimestampMixin):
    """Customer interactions table for tracking customer service interactions."""

    __tablename__ = "customer_interactions"

    # Primary Key
    interaction_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Foreign Key
    customer_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("customers.customer_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Interaction Details
    interaction_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
    )
    interaction_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )  # call, email, chat, ticket

    interaction_channel: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    interaction_category: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    # Issue Resolution
    issue_resolved: Mapped[Optional[bool]] = mapped_column(nullable=True)
    resolution_time_minutes: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Sentiment and Satisfaction
    sentiment_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # -1 to 1
    satisfaction_rating: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # 1-5

    # Notes
    interaction_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Relationship
    customer = relationship("Customer", back_populates="interactions")

    def __repr__(self) -> str:
        return f"<CustomerInteraction(id={self.interaction_id}, customer_id={self.customer_id}, type={self.interaction_type})>"
