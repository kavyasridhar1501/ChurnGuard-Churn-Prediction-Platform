import { useEffect, useState } from 'react';
import Card from '../components/Card';
import { getRiskSegmentation } from '../services/api';
import { PieChart, Pie, Cell, ResponsiveContainer, Legend, Tooltip } from 'recharts';

const Analytics = () => {
  const [segmentation, setSegmentation] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const data = await getRiskSegmentation();
        setSegmentation(data);
        setLoading(false);
      } catch (err) {
        setError('Failed to load analytics data');
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  if (loading) {
    return <div className="text-center py-12">Loading...</div>;
  }

  if (error || !segmentation) {
    return (
      <div className="text-center py-12">
        <p className="text-red-600">{error || 'Failed to load analytics data'}</p>
        <p className="text-sm text-gray-500 mt-2">Please ensure the backend server is running and the database is populated with customer data.</p>
      </div>
    );
  }

  const COLORS = ['#10b981', '#fbbf24', '#f97316', '#ef4444'];

  const pieData = segmentation?.segments.map((seg: any) => ({
    name: seg.segment,
    value: seg.count,
  }));

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-3xl font-bold text-gray-900">Analytics</h2>
        <p className="mt-2 text-gray-600">Deep dive into churn analytics</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <h3 className="text-xl font-semibold text-gray-900 mb-4">Risk Segmentation</h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={pieData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                outerRadius={80}
                fill="#8884d8"
                dataKey="value"
              >
                {pieData?.map((_entry: any, index: number) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </Card>

        <Card>
          <h3 className="text-xl font-semibold text-gray-900 mb-4">Segment Details</h3>
          <div className="space-y-4">
            {segmentation?.segments?.map((seg: any, index: number) => (
              <div key={seg.segment || index} className="flex justify-between items-center">
                <div className="flex items-center space-x-3">
                  <div
                    className="w-4 h-4 rounded-full"
                    style={{ backgroundColor: COLORS[index % COLORS.length] }}
                  />
                  <span className="font-medium capitalize text-gray-900">
                    {seg.segment || 'Unknown'} Risk
                  </span>
                </div>
                <div className="text-right">
                  <div className="font-semibold text-gray-900">
                    {(seg.count || 0).toLocaleString()}
                  </div>
                  <div className="text-sm text-gray-500">
                    {(seg.percentage || 0).toFixed(1)}%
                  </div>
                </div>
              </div>
            )) || <div className="text-gray-500">No segment data available</div>}
          </div>
        </Card>
      </div>
    </div>
  );
};

export default Analytics;
