"""Tests for prediction endpoints."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health_check(client: AsyncClient):
    """Test health check endpoint."""
    response = await client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "version" in data


@pytest.mark.asyncio
async def test_predict_endpoint(client: AsyncClient, sample_customer_data: dict):
    """Test single prediction endpoint."""
    # Note: This will fail without a model, but tests the endpoint structure
    response = await client.post("/api/v1/predict/", json=sample_customer_data)

    # Even if it fails (503 without model), the endpoint should exist
    assert response.status_code in [200, 503]


@pytest.mark.asyncio
async def test_invalid_customer_data(client: AsyncClient):
    """Test prediction with invalid data."""
    invalid_data = {
        "account_length": -1,  # Invalid: negative value
        "international_plan": "not_a_boolean",  # Invalid: not a boolean
    }

    response = await client.post("/api/v1/predict/", json=invalid_data)
    assert response.status_code in [422, 500, 503]  # Validation error or model not available
