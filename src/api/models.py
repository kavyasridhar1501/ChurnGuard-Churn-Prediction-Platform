"""Pydantic models for API request/response validation."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# Request Models
class CustomerFeatures(BaseModel):
    """Customer features for bank churn prediction."""

    credit_score: int = Field(..., ge=300, le=900, description="Credit score (300-900)")
    geography: str = Field(..., description="Country (France, Germany, Spain)")
    gender: str = Field(..., description="Gender (Male, Female)")
    age: int = Field(..., ge=18, le=100, description="Customer age")
    tenure: int = Field(..., ge=0, le=10, description="Years with bank")
    balance: float = Field(..., ge=0, description="Account balance")
    num_of_products: int = Field(..., ge=1, le=4, description="Number of products (1-4)")
    has_credit_card: bool = Field(..., description="Has credit card")
    is_active_member: bool = Field(..., description="Is active member")
    estimated_salary: float = Field(..., ge=0, description="Estimated annual salary")

    @field_validator("geography")
    @classmethod
    def validate_geography(cls, v):
        """Validate geography is one of the expected values."""
        if v not in ["France", "Germany", "Spain"]:
            raise ValueError("Geography must be France, Germany, or Spain")
        return v

    @field_validator("gender")
    @classmethod
    def validate_gender(cls, v):
        """Validate gender is one of the expected values."""
        if v not in ["Male", "Female"]:
            raise ValueError("Gender must be Male or Female")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "credit_score": 650,
                "geography": "France",
                "gender": "Female",
                "age": 42,
                "tenure": 3,
                "balance": 75000.0,
                "num_of_products": 2,
                "has_credit_card": True,
                "is_active_member": True,
                "estimated_salary": 85000.0,
            }
        }


# Response Models
class PredictionResponse(BaseModel):
    """Single prediction response."""

    prediction: bool = Field(..., description="Churn prediction (True = will churn)")
    churn_probability: float = Field(..., ge=0, le=1, description="Probability of churning")
    retention_probability: float = Field(..., ge=0, le=1, description="Probability of retention")
    risk_segment: str = Field(..., description="Risk segment (low, medium, high, critical)")
    model_version: str = Field(..., description="Model version used for prediction")
    timestamp: datetime = Field(default_factory=datetime.now, description="Prediction timestamp")


class DetailedPredictionResponse(PredictionResponse):
    """Detailed prediction with SHAP explanation."""

    top_risk_factors: List[Dict[str, Any]] = Field(..., description="Top contributing features")
    base_value: float = Field(..., description="Model base value")
    estimated_clv: Optional[float] = Field(None, description="Estimated customer lifetime value")


class BatchPredictionItem(BaseModel):
    """Single item in batch prediction result."""

    customer_id: Optional[int] = None
    prediction: bool
    churn_probability: float
    risk_segment: str


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""

    total_processed: int
    predictions: List[BatchPredictionItem]
    summary: Dict[str, Any]


# Customer Models
class CustomerRiskProfile(BaseModel):
    """Customer risk profile with details."""

    customer_id: int
    current_risk_segment: str
    churn_probability: float
    top_risk_factors: List[Dict[str, Any]]
    customer_metrics: Dict[str, Any]
    recommendations: List[str]


# Analytics Models
class AnalyticsOverview(BaseModel):
    """Dashboard analytics overview."""

    total_customers: int
    churn_rate: float
    at_risk_customers: int
    high_risk_customers: int
    critical_risk_customers: int
    estimated_revenue_at_risk: float
    avg_churn_probability: float


class RiskSegmentation(BaseModel):
    """Risk segmentation statistics."""

    segment: str
    count: int
    percentage: float
    avg_probability: float


class RiskSegmentationResponse(BaseModel):
    """Risk segmentation response."""

    total_customers: int
    segments: List[RiskSegmentation]


# Model Performance Models
class ModelMetrics(BaseModel):
    """Model performance metrics."""

    model_version: str
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    auc_pr: float
    training_date: datetime
    is_production: bool


class FeatureImportance(BaseModel):
    """Feature importance."""

    feature: str
    importance: float
    rank: int


class FeatureImportanceResponse(BaseModel):
    """Feature importance response."""

    model_version: str
    features: List[FeatureImportance]
    method: str  # 'shap' or 'model'


# Health Check
class HealthCheckResponse(BaseModel):
    """API health check response."""

    status: str
    version: str
    timestamp: datetime
    database_status: str
    model_loaded: bool


# Error Response
class ErrorResponse(BaseModel):
    """Error response."""

    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
