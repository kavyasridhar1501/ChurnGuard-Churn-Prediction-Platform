import { useState } from 'react';
import Card from '../components/Card';
import { batchPredict } from '../services/api';
import { Upload, Download } from 'lucide-react';

const BatchUpload = () => {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<any>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) return;

    setLoading(true);
    try {
      const result = await batchPredict(file);
      setResults(result);
    } catch (err) {
      alert('Upload failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const downloadResultsAsCSV = () => {
    if (!results || !results.predictions || results.predictions.length === 0) {
      alert('No prediction data available to download');
      return;
    }

    try {
      // Create CSV content
      const headers = ['Row Number', 'Churn Prediction', 'Churn Probability', 'Risk Segment'];
      const rows = results.predictions.map((pred: any) => [
        pred.customer_id || '',
        pred.prediction ? 'Yes' : 'No',
        ((pred.churn_probability || 0) * 100).toFixed(2) + '%',
        pred.risk_segment || 'unknown'
      ]);

      const csvContent = [
        headers.join(','),
        ...rows.map((row: any[]) => row.join(','))
      ].join('\n');

      // Create download link
      const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
      const link = document.createElement('a');
      const url = URL.createObjectURL(blob);

      link.setAttribute('href', url);
      link.setAttribute('download', `churn_predictions_${new Date().toISOString().slice(0,10)}_${Date.now()}.csv`);
      link.style.visibility = 'hidden';

      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);

      // Clean up the URL object to prevent memory leaks
      setTimeout(() => URL.revokeObjectURL(url), 100);
    } catch (error) {
      console.error('Failed to download CSV:', error);
      alert('Failed to download CSV. Please try again.');
    }
  };

  return (
    <div className="max-w-4xl mx-auto space-y-8">
      <div>
        <h2 className="text-3xl font-bold text-gray-900">Batch Prediction</h2>
        <p className="mt-2 text-gray-600">Upload CSV file for batch churn predictions</p>
      </div>

      <Card>
        <form onSubmit={handleSubmit} className="space-y-6">
          <div className="border-2 border-dashed border-gray-300 rounded-lg p-12 text-center">
            <Upload className="mx-auto h-12 w-12 text-gray-400 mb-4" />
            <label className="cursor-pointer">
              <span className="text-primary-600 hover:text-primary-700 font-medium">
                Choose a file
              </span>
              <input
                type="file"
                accept=".csv"
                onChange={handleFileChange}
                className="hidden"
              />
            </label>
            {file && (
              <p className="mt-2 text-sm text-gray-600">Selected: {file.name}</p>
            )}
          </div>

          <button
            type="submit"
            disabled={!file || loading}
            className="w-full bg-primary-600 text-white py-2 px-4 rounded-md hover:bg-primary-700 disabled:opacity-50"
          >
            {loading ? 'Processing...' : 'Upload and Predict'}
          </button>
        </form>
      </Card>

      {results && (
        <Card>
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-xl font-semibold text-gray-900">Results</h3>
            <button
              onClick={downloadResultsAsCSV}
              className="flex items-center gap-2 bg-primary-600 text-white px-4 py-2 rounded-md hover:bg-primary-700 transition-colors"
            >
              <Download className="w-4 h-4" />
              Download CSV
            </button>
          </div>
          <div className="space-y-4">
            <div className="grid grid-cols-3 gap-4">
              <div className="bg-gray-50 p-4 rounded-lg">
                <div className="text-sm text-gray-600">Total Processed</div>
                <div className="text-2xl font-bold text-gray-900">{results.total_processed}</div>
              </div>
              <div className="bg-red-50 p-4 rounded-lg">
                <div className="text-sm text-gray-600">Predicted to Churn</div>
                <div className="text-2xl font-bold text-red-600">
                  {results.summary.predicted_to_churn}
                </div>
              </div>
              <div className="bg-yellow-50 p-4 rounded-lg">
                <div className="text-sm text-gray-600">At Risk</div>
                <div className="text-2xl font-bold text-yellow-600">
                  {results.summary.at_risk_customers}
                </div>
              </div>
            </div>
          </div>
        </Card>
      )}
    </div>
  );
};

export default BatchUpload;
