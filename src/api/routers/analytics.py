"""Analytics endpoints."""

import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from database.models.customer import Customer
from database.models.model_performance import ModelPerformance
from database.models.model_prediction import ModelPrediction
from src.api.dependencies import get_database
from src.api.models import AnalyticsOverview, FeatureImportanceResponse, RiskSegmentationResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/analytics", tags=["analytics"])


@router.get("/overview", response_model=AnalyticsOverview)
async def get_analytics_overview(
    db: AsyncSession = Depends(get_database),
) -> AnalyticsOverview:
    """
    Get dashboard analytics overview.

    Args:
        db: Database session

    Returns:
        Analytics overview
    """
    try:
        # Get total customers
        total_customers_result = await db.execute(select(func.count(Customer.customer_id)))
        total_customers = total_customers_result.scalar() or 0

        # Get churn rate (using 'exited' column for bank dataset)
        churned_result = await db.execute(
            select(func.count(Customer.customer_id)).where(Customer.exited == True)
        )
        churned_count = churned_result.scalar() or 0
        churn_rate = (churned_count / total_customers * 100) if total_customers > 0 else 0

        # Get at-risk customers (from predictions if available)
        # For now, we'll use a simplified calculation
        at_risk_customers = int(total_customers * 0.25)  # Placeholder
        high_risk_customers = int(total_customers * 0.15)  # Placeholder
        critical_risk_customers = int(total_customers * 0.05)  # Placeholder

        # Estimate revenue at risk (using estimated_salary as proxy for customer value)
        avg_salary_result = await db.execute(select(func.avg(Customer.estimated_salary)))
        avg_salary = avg_salary_result.scalar() or 0
        # Estimate annual revenue as 10% of salary (simplified assumption)
        estimated_revenue_at_risk = (avg_salary * 0.1) * at_risk_customers

        return AnalyticsOverview(
            total_customers=total_customers,
            churn_rate=round(churn_rate, 2),
            at_risk_customers=at_risk_customers,
            high_risk_customers=high_risk_customers,
            critical_risk_customers=critical_risk_customers,
            estimated_revenue_at_risk=round(estimated_revenue_at_risk, 2),
            avg_churn_probability=0.25,  # Placeholder
        )

    except Exception as e:
        logger.error(f"Error getting analytics overview: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/segments", response_model=RiskSegmentationResponse)
async def get_risk_segmentation(
    db: AsyncSession = Depends(get_database),
) -> RiskSegmentationResponse:
    """
    Get risk segmentation statistics.

    Args:
        db: Database session

    Returns:
        Risk segmentation response
    """
    try:
        # Get total customers
        total_customers_result = await db.execute(select(func.count(Customer.customer_id)))
        total_customers = total_customers_result.scalar() or 0

        # Simplified segmentation (in production, use actual predictions)
        segments = [
            {
                "segment": "low",
                "count": int(total_customers * 0.50),
                "percentage": 50.0,
                "avg_probability": 0.15,
            },
            {
                "segment": "medium",
                "count": int(total_customers * 0.25),
                "percentage": 25.0,
                "avg_probability": 0.35,
            },
            {
                "segment": "high",
                "count": int(total_customers * 0.15),
                "percentage": 15.0,
                "avg_probability": 0.60,
            },
            {
                "segment": "critical",
                "count": int(total_customers * 0.10),
                "percentage": 10.0,
                "avg_probability": 0.85,
            },
        ]

        return RiskSegmentationResponse(
            total_customers=total_customers,
            segments=segments,
        )

    except Exception as e:
        logger.error(f"Error getting risk segmentation: {e}")
        raise HTTPException(status_code=500, detail=str(e))
