import { useState } from 'react';
import Card from '../components/Card';
import { predictChurn, CustomerFeatures, PredictionResponse } from '../services/api';
import { AlertCircle, CheckCircle } from 'lucide-react';

const Predict = () => {
  const [formData, setFormData] = useState<CustomerFeatures>({
    credit_score: 650,
    geography: 'France',
    gender: 'Female',
    age: 42,
    tenure: 3,
    balance: 75000,
    num_of_products: 2,
    has_credit_card: true,
    is_active_member: true,
    estimated_salary: 85000,
  });

  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const result = await predictChurn(formData);
      setPrediction(result);
    } catch (err) {
      setError('Failed to get prediction. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value, type, checked } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : parseFloat(value) || 0,
    }));
  };

  const getRiskColor = (segment: string) => {
    const colors = {
      low: 'text-green-600 bg-green-50',
      medium: 'text-yellow-600 bg-yellow-50',
      high: 'text-orange-600 bg-orange-50',
      critical: 'text-red-600 bg-red-50',
    };
    return colors[segment as keyof typeof colors] || colors.medium;
  };

  return (
    <div className="max-w-4xl mx-auto space-y-8">
      <div>
        <h2 className="text-3xl font-bold text-gray-900">Customer Churn Prediction</h2>
        <p className="mt-2 text-gray-600">Enter customer details to predict churn risk</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Form */}
        <Card>
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700">
                Credit Score
              </label>
              <input
                type="number"
                name="credit_score"
                value={formData.credit_score}
                onChange={handleChange}
                min="300"
                max="900"
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500 px-3 py-2 border"
              />
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700">Geography</label>
                <select
                  name="geography"
                  value={formData.geography}
                  onChange={(e) => setFormData((prev) => ({ ...prev, geography: e.target.value }))}
                  className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500 px-3 py-2 border"
                >
                  <option value="France">France</option>
                  <option value="Germany">Germany</option>
                  <option value="Spain">Spain</option>
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700">Gender</label>
                <select
                  name="gender"
                  value={formData.gender}
                  onChange={(e) => setFormData((prev) => ({ ...prev, gender: e.target.value }))}
                  className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500 px-3 py-2 border"
                >
                  <option value="Male">Male</option>
                  <option value="Female">Female</option>
                </select>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700">Age</label>
                <input
                  type="number"
                  name="age"
                  value={formData.age}
                  onChange={handleChange}
                  min="18"
                  max="100"
                  className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500 px-3 py-2 border"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700">Tenure (years)</label>
                <input
                  type="number"
                  name="tenure"
                  value={formData.tenure}
                  onChange={handleChange}
                  min="0"
                  max="10"
                  className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500 px-3 py-2 border"
                />
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700">Account Balance</label>
              <input
                type="number"
                step="0.01"
                name="balance"
                value={formData.balance}
                onChange={handleChange}
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500 px-3 py-2 border"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700">Number of Products</label>
              <input
                type="number"
                name="num_of_products"
                value={formData.num_of_products}
                onChange={handleChange}
                min="1"
                max="4"
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500 px-3 py-2 border"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700">Estimated Salary</label>
              <input
                type="number"
                step="0.01"
                name="estimated_salary"
                value={formData.estimated_salary}
                onChange={handleChange}
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500 px-3 py-2 border"
              />
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    name="has_credit_card"
                    checked={formData.has_credit_card}
                    onChange={handleChange}
                    className="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
                  />
                  <span className="text-sm font-medium text-gray-700">Has Credit Card</span>
                </label>
              </div>
              <div>
                <label className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    name="is_active_member"
                    checked={formData.is_active_member}
                    onChange={handleChange}
                    className="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
                  />
                  <span className="text-sm font-medium text-gray-700">Active Member</span>
                </label>
              </div>
            </div>

            <button
              type="submit"
              disabled={loading}
              className="w-full bg-primary-600 text-white py-2 px-4 rounded-md hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 disabled:opacity-50"
            >
              {loading ? 'Predicting...' : 'Predict Churn'}
            </button>

            {error && (
              <div className="flex items-center space-x-2 text-red-600 text-sm">
                <AlertCircle className="h-4 w-4" />
                <span>{error}</span>
              </div>
            )}
          </form>
        </Card>

        {/* Results */}
        <Card>
          <h3 className="text-xl font-semibold text-gray-900 mb-4">Prediction Result</h3>

          {prediction ? (
            <div className="space-y-4">
              <div className="text-center py-6">
                {prediction.prediction ? (
                  <AlertCircle className="h-16 w-16 text-red-500 mx-auto mb-4" />
                ) : (
                  <CheckCircle className="h-16 w-16 text-green-500 mx-auto mb-4" />
                )}
                <h4 className="text-2xl font-bold text-gray-900 mb-2">
                  {prediction.prediction ? 'Likely to Churn' : 'Likely to Stay'}
                </h4>
                <p className="text-lg text-gray-600">
                  Churn Probability: {(prediction.churn_probability * 100).toFixed(1)}%
                </p>
              </div>

              <div className="border-t pt-4 space-y-3">
                <div className="flex justify-between">
                  <span className="text-gray-600">Risk Segment</span>
                  <span
                    className={`px-3 py-1 rounded-full text-sm font-medium ${getRiskColor(
                      prediction.risk_segment
                    )}`}
                  >
                    {prediction.risk_segment.toUpperCase()}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Retention Probability</span>
                  <span className="font-semibold text-gray-900">
                    {(prediction.retention_probability * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Model Version</span>
                  <span className="font-mono text-sm text-gray-900">{prediction.model_version}</span>
                </div>
              </div>
            </div>
          ) : (
            <div className="text-center py-12 text-gray-500">
              Enter customer details and click predict to see results
            </div>
          )}
        </Card>
      </div>
    </div>
  );
};

export default Predict;
