import { useEffect, useState } from 'react';
import { Users, TrendingDown, AlertTriangle, DollarSign } from 'lucide-react';
import StatCard from '../components/StatCard';
import Card from '../components/Card';
import { getAnalyticsOverview, AnalyticsOverview } from '../services/api';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const Dashboard = () => {
  const [analytics, setAnalytics] = useState<AnalyticsOverview | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchAnalytics = async () => {
      try {
        const data = await getAnalyticsOverview();
        setAnalytics(data);
        setLoading(false);
      } catch (err) {
        setError('Failed to load analytics data');
        setLoading(false);
      }
    };

    fetchAnalytics();
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-lg text-gray-600">Loading...</div>
      </div>
    );
  }

  if (error || !analytics) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-lg text-red-600">{error || 'No data available'}</div>
      </div>
    );
  }

  const riskData = [
    { name: 'Low Risk', value: analytics.total_customers - analytics.at_risk_customers },
    { name: 'Medium Risk', value: analytics.at_risk_customers - analytics.high_risk_customers },
    { name: 'High Risk', value: analytics.high_risk_customers - analytics.critical_risk_customers },
    { name: 'Critical', value: analytics.critical_risk_customers },
  ];

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-3xl font-bold text-gray-900">Dashboard</h2>
        <p className="mt-2 text-gray-600">Customer churn analytics overview</p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatCard
          title="Total Customers"
          value={analytics.total_customers.toLocaleString()}
          icon={Users}
          color="blue"
        />
        <StatCard
          title="Churn Rate"
          value={`${analytics.churn_rate.toFixed(1)}%`}
          icon={TrendingDown}
          color="red"
        />
        <StatCard
          title="At-Risk Customers"
          value={analytics.at_risk_customers.toLocaleString()}
          icon={AlertTriangle}
          color="yellow"
        />
        <StatCard
          title="Revenue at Risk"
          value={`$${(analytics.estimated_revenue_at_risk / 1000).toFixed(1)}K`}
          icon={DollarSign}
          color="green"
        />
      </div>

      {/* Risk Distribution Chart */}
      <Card>
        <h3 className="text-xl font-semibold text-gray-900 mb-4">Risk Distribution</h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={riskData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Bar dataKey="value" fill="#0ea5e9" name="Customers" />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <h3 className="text-xl font-semibold text-gray-900 mb-4">Risk Breakdown</h3>
          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <span className="text-gray-600">Critical Risk</span>
              <span className="font-semibold text-red-600">
                {analytics.critical_risk_customers.toLocaleString()}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-600">High Risk</span>
              <span className="font-semibold text-orange-600">
                {analytics.high_risk_customers.toLocaleString()}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-600">Medium Risk</span>
              <span className="font-semibold text-yellow-600">
                {analytics.at_risk_customers.toLocaleString()}
              </span>
            </div>
          </div>
        </Card>

        <Card>
          <h3 className="text-xl font-semibold text-gray-900 mb-4">Average Metrics</h3>
          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <span className="text-gray-600">Avg Churn Probability</span>
              <span className="font-semibold text-gray-900">
                {((analytics.avg_churn_probability || 0) * 100).toFixed(1)}%
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-600">Avg Revenue per Customer</span>
              <span className="font-semibold text-gray-900">
                $
                {analytics.at_risk_customers > 0
                  ? (analytics.estimated_revenue_at_risk / analytics.at_risk_customers).toFixed(2)
                  : '0.00'}
              </span>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
};

export default Dashboard;
