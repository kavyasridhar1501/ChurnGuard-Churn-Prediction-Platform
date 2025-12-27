"""Initial migration for bank customer churn schema.

Revision ID: 001
Revises:
Create Date: 2025-12-25 03:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create all tables for bank customer churn prediction."""

    # Create customers table
    op.create_table(
        'customers',
        sa.Column('customer_id', sa.Integer(), autoincrement=True, nullable=False),

        # Basic Customer Information
        sa.Column('credit_score', sa.Integer(), nullable=True),
        sa.Column('geography', sa.String(length=50), nullable=True),
        sa.Column('gender', sa.String(length=10), nullable=True),
        sa.Column('age', sa.Integer(), nullable=True),
        sa.Column('tenure', sa.Integer(), nullable=True),

        # Financial Information
        sa.Column('balance', sa.Float(), nullable=True),
        sa.Column('estimated_salary', sa.Float(), nullable=True),

        # Product Information
        sa.Column('num_of_products', sa.Integer(), nullable=True),
        sa.Column('has_credit_card', sa.Boolean(), nullable=True),
        sa.Column('is_active_member', sa.Boolean(), nullable=True),

        # Churn Status
        sa.Column('exited', sa.Boolean(), nullable=True),

        # Derived fields
        sa.Column('balance_salary_ratio', sa.Float(), nullable=True),
        sa.Column('age_group', sa.String(length=20), nullable=True),
        sa.Column('tenure_age_ratio', sa.Float(), nullable=True),
        sa.Column('products_per_tenure', sa.Float(), nullable=True),
        sa.Column('engagement_score', sa.Float(), nullable=True),

        # Date fields
        sa.Column('signup_date', sa.Date(), nullable=True),
        sa.Column('last_interaction_date', sa.Date(), nullable=True),

        # Timestamps
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),

        sa.PrimaryKeyConstraint('customer_id')
    )

    # Create indexes
    op.create_index('ix_customers_geography', 'customers', ['geography'])
    op.create_index('ix_customers_exited', 'customers', ['exited'])

    # Create customer_usage_history table
    op.create_table(
        'customer_usage_history',
        sa.Column('usage_id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('customer_id', sa.Integer(), nullable=False),

        # Activity Period
        sa.Column('activity_date', sa.Date(), nullable=False),
        sa.Column('activity_period', sa.String(length=20), nullable=False),

        # Transaction Metrics
        sa.Column('transaction_count', sa.Integer(), nullable=True),
        sa.Column('total_transaction_amount', sa.Float(), nullable=True),
        sa.Column('avg_transaction_amount', sa.Float(), nullable=True),

        # Balance Tracking
        sa.Column('balance_start', sa.Float(), nullable=True),
        sa.Column('balance_end', sa.Float(), nullable=True),
        sa.Column('balance_change', sa.Float(), nullable=True),

        # Activity Flags
        sa.Column('is_active_period', sa.Boolean(), nullable=True),
        sa.Column('product_usage_count', sa.Integer(), nullable=True),

        # Timestamps
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),

        sa.ForeignKeyConstraint(['customer_id'], ['customers.customer_id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('usage_id')
    )

    # Create indexes for usage_history
    op.create_index('ix_customer_usage_history_customer_id', 'customer_usage_history', ['customer_id'])
    op.create_index('ix_customer_usage_history_activity_date', 'customer_usage_history', ['activity_date'])

    # Create customer_interactions table
    op.create_table(
        'customer_interactions',
        sa.Column('interaction_id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('customer_id', sa.Integer(), nullable=False),

        # Interaction Details
        sa.Column('interaction_date', sa.DateTime(timezone=True), nullable=False),
        sa.Column('interaction_type', sa.String(length=50), nullable=False),
        sa.Column('interaction_channel', sa.String(length=50), nullable=True),
        sa.Column('interaction_category', sa.String(length=100), nullable=True),

        # Issue Resolution
        sa.Column('issue_resolved', sa.Boolean(), nullable=True),
        sa.Column('resolution_time_minutes', sa.Integer(), nullable=True),

        # Sentiment and Satisfaction
        sa.Column('sentiment_score', sa.Float(), nullable=True),
        sa.Column('satisfaction_rating', sa.Integer(), nullable=True),

        # Notes
        sa.Column('interaction_notes', sa.Text(), nullable=True),

        # Timestamps
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),

        sa.ForeignKeyConstraint(['customer_id'], ['customers.customer_id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('interaction_id')
    )

    # Create indexes for interactions
    op.create_index('ix_customer_interactions_customer_id', 'customer_interactions', ['customer_id'])
    op.create_index('ix_customer_interactions_interaction_date', 'customer_interactions', ['interaction_date'])

    # Create model_performance table
    op.create_table(
        'model_performance',
        sa.Column('performance_id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('model_version', sa.String(length=50), nullable=False),
        sa.Column('model_name', sa.String(length=100), nullable=False),

        # Performance Metrics
        sa.Column('accuracy', sa.Float(), nullable=False),
        sa.Column('precision', sa.Float(), nullable=False),
        sa.Column('recall', sa.Float(), nullable=False),
        sa.Column('f1_score', sa.Float(), nullable=False),
        sa.Column('auc_roc', sa.Float(), nullable=False),
        sa.Column('auc_pr', sa.Float(), nullable=False),

        # Training Info
        sa.Column('training_date', sa.DateTime(timezone=True), nullable=False),
        sa.Column('training_samples', sa.Integer(), nullable=True),
        sa.Column('test_samples', sa.Integer(), nullable=True),

        # Production Status
        sa.Column('is_production', sa.Boolean(), default=False, nullable=False),
        sa.Column('deployed_at', sa.DateTime(timezone=True), nullable=True),

        # Additional Metadata
        sa.Column('hyperparameters', sa.JSON(), nullable=True),
        sa.Column('feature_importance', sa.JSON(), nullable=True),

        # Timestamps
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),

        sa.PrimaryKeyConstraint('performance_id')
    )

    # Create indexes for model_performance
    op.create_index('ix_model_performance_model_version', 'model_performance', ['model_version'])
    op.create_index('ix_model_performance_is_production', 'model_performance', ['is_production'])

    # Create model_predictions table
    op.create_table(
        'model_predictions',
        sa.Column('prediction_id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('customer_id', sa.Integer(), nullable=False),
        sa.Column('model_version', sa.String(length=50), nullable=False),

        # Prediction Results
        sa.Column('prediction_date', sa.DateTime(timezone=True), nullable=False),
        sa.Column('churn_probability', sa.Float(), nullable=False),
        sa.Column('predicted_churn', sa.Boolean(), nullable=False),
        sa.Column('risk_segment', sa.String(length=20), nullable=True),

        # Actual Outcome (for monitoring)
        sa.Column('actual_churn', sa.Boolean(), nullable=True),
        sa.Column('feedback_date', sa.DateTime(timezone=True), nullable=True),

        # Feature Values at Prediction Time
        sa.Column('feature_values', sa.JSON(), nullable=True),
        sa.Column('shap_values', sa.JSON(), nullable=True),

        # Timestamps
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),

        sa.ForeignKeyConstraint(['customer_id'], ['customers.customer_id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('prediction_id')
    )

    # Create indexes for predictions
    op.create_index('ix_model_predictions_customer_id', 'model_predictions', ['customer_id'])
    op.create_index('ix_model_predictions_prediction_date', 'model_predictions', ['prediction_date'])
    op.create_index('ix_model_predictions_model_version', 'model_predictions', ['model_version'])


def downgrade() -> None:
    """Drop all tables."""
    op.drop_index('ix_model_predictions_model_version', table_name='model_predictions')
    op.drop_index('ix_model_predictions_prediction_date', table_name='model_predictions')
    op.drop_index('ix_model_predictions_customer_id', table_name='model_predictions')
    op.drop_table('model_predictions')

    op.drop_index('ix_model_performance_is_production', table_name='model_performance')
    op.drop_index('ix_model_performance_model_version', table_name='model_performance')
    op.drop_table('model_performance')

    op.drop_index('ix_customer_interactions_interaction_date', table_name='customer_interactions')
    op.drop_index('ix_customer_interactions_customer_id', table_name='customer_interactions')
    op.drop_table('customer_interactions')

    op.drop_index('ix_customer_usage_history_activity_date', table_name='customer_usage_history')
    op.drop_index('ix_customer_usage_history_customer_id', table_name='customer_usage_history')
    op.drop_table('customer_usage_history')

    op.drop_index('ix_customers_exited', table_name='customers')
    op.drop_index('ix_customers_geography', table_name='customers')
    op.drop_table('customers')
