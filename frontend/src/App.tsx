import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import Predict from './pages/Predict';
import Analytics from './pages/Analytics';
import BatchUpload from './pages/BatchUpload';
import ModelInfo from './pages/ModelInfo';

function App() {
  return (
    <Router>
      <Layout>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/predict" element={<Predict />} />
          <Route path="/analytics" element={<Analytics />} />
          <Route path="/batch" element={<BatchUpload />} />
          <Route path="/model" element={<ModelInfo />} />
        </Routes>
      </Layout>
    </Router>
  );
}

export default App;
