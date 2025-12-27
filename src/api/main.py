"""FastAPI main application."""

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.dependencies import get_model
from src.api.models import ErrorResponse, HealthCheckResponse
from src.api.routers import analytics, models, predictions
from src.db.connection import close_db, init_db

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# API version
API_VERSION = "1.0.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting ChurnGuard API...")

    # Initialize database (for development)
    if os.getenv("ENVIRONMENT") == "development":
        try:
            await init_db()
            logger.info("Database initialized")
        except Exception as e:
            logger.warning(f"Database initialization skipped: {e}")

    # Load ML model
    try:
        get_model()
        logger.info("ML model loaded successfully")
    except Exception as e:
        logger.warning(f"ML model not loaded: {e}")

    logger.info("ChurnGuard API started successfully")

    yield

    # Shutdown
    logger.info("Shutting down ChurnGuard API...")
    await close_db()
    logger.info("ChurnGuard API shut down")


# Create FastAPI app
app = FastAPI(
    title="ChurnGuard API",
    description="Customer Churn Prediction API for SaaS businesses",
    version=API_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
origins = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Global exception: {exc}", exc_info=True)

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal Server Error",
            detail=str(exc),
        ).model_dump(),
    )


# Health check endpoint
@app.get("/api/v1/health", response_model=HealthCheckResponse, tags=["health"])
async def health_check() -> HealthCheckResponse:
    """
    Health check endpoint.

    Returns:
        Health check response
    """
    # Check if model is loaded
    model_loaded = False
    try:
        get_model()
        model_loaded = True
    except Exception:
        pass

    # Check database
    database_status = "unknown"
    try:
        from sqlalchemy import text
        from src.db.connection import engine

        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        database_status = "healthy"
    except Exception as e:
        database_status = f"unhealthy: {str(e)}"
        logger.error(f"Database health check failed: {e}")

    return HealthCheckResponse(
        status="healthy" if model_loaded and database_status == "healthy" else "degraded",
        version=API_VERSION,
        timestamp=datetime.now(),
        database_status=database_status,
        model_loaded=model_loaded,
    )


# Root endpoint
@app.get("/", tags=["root"])
async def root():
    """Root endpoint."""
    return {
        "name": "ChurnGuard API",
        "version": API_VERSION,
        "status": "running",
        "docs": "/docs",
    }


# Include routers
app.include_router(predictions.router)
app.include_router(analytics.router)
app.include_router(models.router)


# Development server
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=os.getenv("API_RELOAD", "True").lower() == "true",
        log_level=os.getenv("API_LOG_LEVEL", "info"),
    )
