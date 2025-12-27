import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Types
export interface CustomerFeatures {
  credit_score: number;
  geography: string;
  gender: string;
  age: number;
  tenure: number;
  balance: number;
  num_of_products: number;
  has_credit_card: boolean;
  is_active_member: boolean;
  estimated_salary: number;
}

export interface PredictionResponse {
  prediction: boolean;
  churn_probability: number;
  retention_probability: number;
  risk_segment: string;
  model_version: string;
  timestamp: string;
}

export interface DetailedPredictionResponse extends PredictionResponse {
  top_risk_factors: Array<{
    feature: string;
    value: number;
    shap_value: number;
  }>;
  base_value: number;
  estimated_clv?: number;
}

export interface AnalyticsOverview {
  total_customers: number;
  churn_rate: number;
  at_risk_customers: number;
  high_risk_customers: number;
  critical_risk_customers: number;
  estimated_revenue_at_risk: number;
  avg_churn_probability: number;
}

export interface RiskSegmentation {
  segment: string;
  count: number;
  percentage: number;
  avg_probability: number;
}

export interface ModelMetrics {
  model_version: string;
  model_name: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  auc_roc: number;
  auc_pr: number;
  training_date: string;
  is_production: boolean;
}

export interface FeatureImportance {
  feature: string;
  importance: number;
  rank: number;
}

// API Functions
export const predictChurn = async (customer: CustomerFeatures): Promise<PredictionResponse> => {
  const response = await api.post('/api/v1/predict/', customer);
  return response.data;
};

export const predictChurnDetailed = async (
  customer: CustomerFeatures
): Promise<DetailedPredictionResponse> => {
  const response = await api.post('/api/v1/predict/detailed', customer);
  return response.data;
};

export const batchPredict = async (file: File) => {
  const formData = new FormData();
  formData.append('file', file);

  const response = await api.post('/api/v1/predict/batch', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return response.data;
};

export const getAnalyticsOverview = async (): Promise<AnalyticsOverview> => {
  const response = await api.get('/api/v1/analytics/overview');
  return response.data;
};

export const getRiskSegmentation = async () => {
  const response = await api.get('/api/v1/analytics/segments');
  return response.data;
};

export const getCurrentModel = async (): Promise<ModelMetrics> => {
  const response = await api.get('/api/v1/models/current');
  return response.data;
};

export const getModelPerformance = async (): Promise<ModelMetrics[]> => {
  const response = await api.get('/api/v1/models/performance');
  return response.data;
};

export const getFeatureImportance = async (method: string = 'model') => {
  const response = await api.get(`/api/v1/models/features/importance?method=${method}`);
  return response.data;
};

export const healthCheck = async () => {
  const response = await api.get('/api/v1/health');
  return response.data;
};

export default api;
