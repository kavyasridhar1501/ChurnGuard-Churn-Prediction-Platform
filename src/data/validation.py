"""Data validation using Great Expectations."""

import logging
from pathlib import Path
from typing import Dict, Optional

import polars as pl

# Note: Great Expectations integration will be added after basic setup
# For now, we'll use Polars-based validation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataValidator:
    """Data validation class for churn dataset."""

    def __init__(self, df: pl.DataFrame):
        """
        Initialize validator with DataFrame.

        Args:
            df: DataFrame to validate
        """
        self.df = df
        self.validation_results: Dict[str, bool] = {}
        self.validation_messages: Dict[str, str] = {}

    def validate_required_columns(self, required_columns: list[str]) -> bool:
        """
        Validate that required columns are present.

        Args:
            required_columns: List of required column names

        Returns:
            True if all required columns are present
        """
        missing_columns = set(required_columns) - set(self.df.columns)

        if missing_columns:
            self.validation_results["required_columns"] = False
            self.validation_messages["required_columns"] = f"Missing columns: {missing_columns}"
            logger.error(f"Missing required columns: {missing_columns}")
            return False

        self.validation_results["required_columns"] = True
        self.validation_messages["required_columns"] = "All required columns present"
        logger.info("All required columns present")
        return True

    def validate_data_types(self) -> bool:
        """
        Validate expected data types.

        Returns:
            True if data types are valid
        """
        type_checks = []

        # Check numeric columns
        numeric_cols = [
            "account_length",
            "total_day_minutes",
            "total_eve_minutes",
            "total_night_minutes",
            "total_intl_minutes",
        ]

        for col in numeric_cols:
            if col in self.df.columns:
                is_numeric = self.df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]
                type_checks.append(is_numeric)
                if not is_numeric:
                    logger.warning(f"Column {col} is not numeric: {self.df[col].dtype}")

        # Check boolean columns
        boolean_cols = ["churn", "international_plan", "voice_mail_plan"]

        for col in boolean_cols:
            if col in self.df.columns:
                is_boolean = self.df[col].dtype == pl.Boolean
                type_checks.append(is_boolean)
                if not is_boolean:
                    logger.warning(f"Column {col} is not boolean: {self.df[col].dtype}")

        all_valid = all(type_checks)
        self.validation_results["data_types"] = all_valid

        if all_valid:
            self.validation_messages["data_types"] = "All data types valid"
            logger.info("All data types valid")
        else:
            self.validation_messages["data_types"] = "Some data types invalid"

        return all_valid

    def validate_value_ranges(self) -> bool:
        """
        Validate that values are in expected ranges.

        Returns:
            True if all values are in valid ranges
        """
        range_checks = []

        # Check non-negative values
        non_negative_cols = [
            "account_length",
            "total_day_minutes",
            "total_day_calls",
            "total_day_charge",
            "customer_service_calls",
        ]

        for col in non_negative_cols:
            if col in self.df.columns:
                min_value = self.df[col].min()
                is_valid = min_value >= 0 if min_value is not None else True
                range_checks.append(is_valid)

                if not is_valid:
                    logger.warning(f"Column {col} has negative values: min={min_value}")

        all_valid = all(range_checks)
        self.validation_results["value_ranges"] = all_valid

        if all_valid:
            self.validation_messages["value_ranges"] = "All value ranges valid"
            logger.info("All value ranges valid")
        else:
            self.validation_messages["value_ranges"] = "Some values out of range"

        return all_valid

    def validate_no_duplicates(self, key_column: Optional[str] = None) -> bool:
        """
        Validate no duplicate records.

        Args:
            key_column: Column to check for duplicates (if None, checks entire rows)

        Returns:
            True if no duplicates found
        """
        if key_column and key_column in self.df.columns:
            duplicates = self.df.filter(pl.col(key_column).is_duplicated())
            dup_count = len(duplicates)
        else:
            duplicates = self.df.filter(pl.all_horizontal(pl.all().is_duplicated()))
            dup_count = len(duplicates)

        is_valid = dup_count == 0
        self.validation_results["no_duplicates"] = is_valid

        if is_valid:
            self.validation_messages["no_duplicates"] = "No duplicates found"
            logger.info("No duplicates found")
        else:
            self.validation_messages["no_duplicates"] = f"Found {dup_count} duplicates"
            logger.warning(f"Found {dup_count} duplicates")

        return is_valid

    def validate_completeness(self, max_null_percentage: float = 10.0) -> bool:
        """
        Validate data completeness (null percentage threshold).

        Args:
            max_null_percentage: Maximum allowed null percentage

        Returns:
            True if null percentage is below threshold
        """
        total_rows = len(self.df)
        completeness_checks = []

        for col in self.df.columns:
            null_count = self.df[col].null_count()
            null_percentage = (null_count / total_rows) * 100

            is_valid = null_percentage <= max_null_percentage
            completeness_checks.append(is_valid)

            if not is_valid:
                logger.warning(f"Column {col} has {null_percentage:.2f}% null values")

        all_valid = all(completeness_checks)
        self.validation_results["completeness"] = all_valid

        if all_valid:
            self.validation_messages["completeness"] = f"All columns below {max_null_percentage}% null threshold"
            logger.info(f"All columns below {max_null_percentage}% null threshold")
        else:
            self.validation_messages["completeness"] = "Some columns exceed null threshold"

        return all_valid

    def run_all_validations(
        self,
        required_columns: Optional[list[str]] = None,
        key_column: Optional[str] = None,
    ) -> Dict:
        """
        Run all validations and return results.

        Args:
            required_columns: List of required columns
            key_column: Key column for duplicate check

        Returns:
            Dictionary with validation results
        """
        logger.info("Running all validations...")

        # Define required columns if not provided
        if required_columns is None:
            required_columns = [
                "account_length",
                "total_day_minutes",
                "total_eve_minutes",
                "total_night_minutes",
                "customer_service_calls",
                "churn",
            ]

        # Run validations
        self.validate_required_columns(required_columns)
        self.validate_data_types()
        self.validate_value_ranges()
        self.validate_no_duplicates(key_column)
        self.validate_completeness()

        # Overall status
        all_passed = all(self.validation_results.values())

        results = {
            "overall_status": "PASSED" if all_passed else "FAILED",
            "validations": self.validation_results,
            "messages": self.validation_messages,
            "dataset_shape": self.df.shape,
        }

        logger.info(f"Validation complete. Overall status: {results['overall_status']}")

        return results


def validate_churn_dataset(df: pl.DataFrame, required_columns: Optional[list[str]] = None) -> Dict:
    """
    Convenience function to validate churn dataset.

    Args:
        df: DataFrame to validate
        required_columns: List of required columns

    Returns:
        Validation results dictionary
    """
    validator = DataValidator(df)
    return validator.run_all_validations(required_columns=required_columns, key_column="phone_number")
