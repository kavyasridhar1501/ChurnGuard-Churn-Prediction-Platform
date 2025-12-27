"""Tests for data preprocessing."""

import polars as pl
import pytest

from src.data.preprocessing import ChurnDataPreprocessor


def test_preprocessor_initialization():
    """Test preprocessor initialization."""
    df = pl.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
    preprocessor = ChurnDataPreprocessor(df)

    assert preprocessor.df is not None
    assert preprocessor.original_shape == (3, 2)


def test_convert_boolean_columns():
    """Test boolean column conversion."""
    df = pl.DataFrame(
        {
            "churn": ["yes", "no", "yes"],
            "international_plan": ["True", "False", "True"],
        }
    )

    preprocessor = ChurnDataPreprocessor(df)
    preprocessor.convert_boolean_columns()

    result_df = preprocessor.get_processed_data()

    assert result_df["churn"].dtype == pl.Boolean
    assert result_df["international_plan"].dtype == pl.Boolean


def test_handle_missing_values():
    """Test missing value handling."""
    df = pl.DataFrame({"col1": [1, None, 3], "col2": [4, 5, None]})

    preprocessor = ChurnDataPreprocessor(df)
    preprocessor.handle_missing_values(strategy="drop")

    result_df = preprocessor.get_processed_data()

    assert len(result_df) == 1  # Only one row without nulls
