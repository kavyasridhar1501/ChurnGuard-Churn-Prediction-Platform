"""Data ingestion pipeline to load data into PostgreSQL."""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import polars as pl
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from database.models.customer import Customer
from src.db.connection import AsyncSessionLocal, engine
from src.data.download import load_and_validate_dataset, standardize_column_names
from src.data.preprocessing import preprocess_churn_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def create_tables() -> None:
    """Create all database tables."""
    from database.models.base import Base

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables created successfully")


async def truncate_customers_table() -> None:
    """Truncate customers table before loading new data."""
    async with AsyncSessionLocal() as session:
        await session.execute(text("TRUNCATE TABLE customers RESTART IDENTITY CASCADE"))
        await session.commit()
    logger.info("Customers table truncated")


async def load_customers_to_db(df: pl.DataFrame, batch_size: int = 1000) -> int:
    """
    Load customer data into PostgreSQL database.

    Args:
        df: Polars DataFrame with customer data
        batch_size: Number of records to insert in each batch

    Returns:
        Number of records inserted
    """
    logger.info(f"Loading {len(df)} customers to database...")

    # Convert DataFrame to list of dictionaries
    records = df.to_dicts()

    total_inserted = 0
    async with AsyncSessionLocal() as session:
        try:
            for i in range(0, len(records), batch_size):
                batch = records[i : i + batch_size]

                # Create Customer objects
                customers = []
                for record in batch:
                    # Map DataFrame columns to Customer model for bank dataset
                    customer_data = {
                        "credit_score": record.get("credit_score"),
                        "geography": record.get("geography"),
                        "gender": record.get("gender"),
                        "age": record.get("age"),
                        "tenure": record.get("tenure"),
                        "balance": record.get("balance"),
                        "num_of_products": record.get("num_of_products"),
                        "has_credit_card": record.get("has_credit_card"),
                        "is_active_member": record.get("is_active_member"),
                        "estimated_salary": record.get("estimated_salary"),
                        "exited": record.get("exited"),
                        # Derived fields from preprocessing
                        "balance_salary_ratio": record.get("balance_salary_ratio"),
                        "age_group": record.get("age_group"),
                        "tenure_age_ratio": record.get("tenure_age_ratio"),
                        "products_per_tenure": record.get("products_per_tenure"),
                        "engagement_score": record.get("engagement_score"),
                        "signup_date": record.get("signup_date"),
                    }

                    # Remove None values
                    customer_data = {k: v for k, v in customer_data.items() if v is not None}

                    customers.append(Customer(**customer_data))

                # Bulk insert
                session.add_all(customers)
                await session.commit()

                total_inserted += len(customers)
                logger.info(f"Inserted batch {i // batch_size + 1}: {total_inserted}/{len(records)} records")

            logger.info(f"Successfully loaded {total_inserted} customers to database")
            return total_inserted

        except Exception as e:
            await session.rollback()
            logger.error(f"Error loading customers to database: {e}")
            raise


async def run_full_ingestion_pipeline(
    raw_file_path: Optional[Path] = None,
    truncate_existing: bool = False,
) -> dict:
    """
    Run the complete data ingestion pipeline for bank dataset.

    Args:
        raw_file_path: Path to raw CSV file (if None, uses default path)
        truncate_existing: Whether to truncate existing data

    Returns:
        Dictionary with ingestion statistics
    """
    start_time = datetime.now()
    logger.info("Starting full data ingestion pipeline for bank dataset")

    try:
        # Step 1: Create tables
        await create_tables()

        # Step 2: Load raw file
        if raw_file_path is None:
            raw_file_path = Path(__file__).parent.parent.parent / "data" / "raw" / "Churn_Modelling.csv"

        # Step 3: Load and validate
        df = load_and_validate_dataset(raw_file_path)

        # Step 4: Standardize columns
        df = standardize_column_names(df)

        # Step 5: Preprocess data
        df, summary = preprocess_churn_data(df, handle_missing="drop", handle_outliers=True)

        # Step 6: Truncate if requested
        if truncate_existing:
            await truncate_customers_table()

        # Step 7: Load to database
        records_loaded = await load_customers_to_db(df)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        stats = {
            "status": "success",
            "records_loaded": records_loaded,
            "duration_seconds": duration,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "preprocessing_summary": summary,
        }

        logger.info(f"Data ingestion pipeline completed in {duration:.2f} seconds")
        return stats

    except Exception as e:
        logger.error(f"Data ingestion pipeline failed: {e}")
        raise


async def main() -> None:
    """Main function for bank customer data ingestion."""
    stats = await run_full_ingestion_pipeline(truncate_existing=True)
    logger.info(f"Ingestion statistics: {stats}")


if __name__ == "__main__":
    asyncio.run(main())
