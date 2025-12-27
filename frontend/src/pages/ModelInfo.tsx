import { useEffect, useState } from 'react';
import Card from '../components/Card';
import { getCurrentModel, getFeatureImportance, ModelMetrics } from '../services/api';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const ModelInfo = () => {
  const [model, setModel] = useState<ModelMetrics | null>(null);
  const [features, setFeatures] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [modelData, featureData] = await Promise.all([
          getCurrentModel(),
          getFeatureImportance(),
        ]);
        setModel(modelData);
        setFeatures(featureData);
        setLoading(false);
      } catch (err) {
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  if (loading) {
    return <div className="text-center py-12">Loading...</div>;
  }

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-3xl font-bold text-gray-900">Model Information</h2>
        <p className="mt-2 text-gray-600">Current production model details</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <h3 className="text-xl font-semibold text-gray-900 mb-4">Model Metrics</h3>
          {model ? (
            <div className="space-y-3">
              <div className="flex justify-between">
                <span className="text-gray-600">Model Name</span>
                <span className="font-semibold text-gray-900">{model.model_name || 'N/A'}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Version</span>
                <span className="font-mono text-sm text-gray-900">{model.model_version || 'N/A'}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">AUC-ROC</span>
                <span className="font-semibold text-gray-900">{(model.auc_roc || 0).toFixed(3)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Precision</span>
                <span className="font-semibold text-gray-900">{(model.precision || 0).toFixed(3)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Recall</span>
                <span className="font-semibold text-gray-900">{(model.recall || 0).toFixed(3)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">F1 Score</span>
                <span className="font-semibold text-gray-900">{(model.f1_score || 0).toFixed(3)}</span>
              </div>
            </div>
          ) : (
            <div className="text-center py-8 text-gray-500">
              <p>No model data available.</p>
              <p className="text-sm mt-2">Train a model to see metrics here.</p>
            </div>
          )}
        </Card>

        <Card>
          <h3 className="text-xl font-semibold text-gray-900 mb-4">Top Features</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={features?.features.slice(0, 5)} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                type="number"
                label={{ value: 'Importance Score', position: 'insideBottom', offset: -5 }}
              />
              <YAxis
                dataKey="feature"
                type="category"
                width={150}
                label={{ value: 'Features', angle: -90, position: 'insideLeft' }}
              />
              <Tooltip />
              <Bar dataKey="importance" fill="#0ea5e9" />
            </BarChart>
          </ResponsiveContainer>
        </Card>
      </div>
    </div>
  );
};

export default ModelInfo;
