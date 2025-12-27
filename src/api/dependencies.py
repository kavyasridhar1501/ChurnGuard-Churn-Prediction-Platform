"""FastAPI dependencies."""

import logging
from pathlib import Path
from typing import AsyncGenerator, Optional

import joblib
from fastapi import Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.connection import get_db
from src.models.predict import ChurnPredictor

logger = logging.getLogger(__name__)

# Global model cache
_model_cache: Optional[ChurnPredictor] = None
_model_path = Path("./models/best_model_production.pkl")


async def get_database() -> AsyncGenerator[AsyncSession, None]:
    """
    Get database session dependency.

    Yields:
        AsyncSession: Database session
    """
    async for session in get_db():
        yield session


def get_model() -> ChurnPredictor:
    """
    Get loaded ML model dependency.

    Returns:
        ChurnPredictor: Loaded model

    Raises:
        HTTPException: If model cannot be loaded
    """
    global _model_cache, _model_path

    if _model_cache is None:
        try:
            # Check if model file exists
            model_path_to_use = _model_path

            if not model_path_to_use.exists():
                # Try alternative paths
                alternative_paths = [
                    Path("./models/best_model_latest.pkl"),
                    Path("./models/lightgbm_latest.pkl"),
                    Path("./models/xgboost_latest.pkl"),
                    Path("./models/random_forest_latest.pkl"),
                ]

                for alt_path in alternative_paths:
                    if alt_path.exists():
                        model_path_to_use = alt_path
                        logger.info(f"Using alternative model path: {model_path_to_use}")
                        break
                else:
                    raise FileNotFoundError(
                        f"Model file not found. Tried: {_model_path} and alternatives. "
                        f"Please run 'python -m src.models.train' to train a model first."
                    )

            logger.info(f"Loading model from {model_path_to_use}")
            _model_cache = ChurnPredictor(str(model_path_to_use))
            logger.info("Model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Model not available: {str(e)}",
            )

    return _model_cache


def reload_model() -> ChurnPredictor:
    """
    Reload model from disk.

    Returns:
        ChurnPredictor: Reloaded model
    """
    global _model_cache
    _model_cache = None
    return get_model()


# Model version tracking
_current_model_version = "1.0.0"


def get_model_version() -> str:
    """Get current model version."""
    return _current_model_version
