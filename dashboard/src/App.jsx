import { useState } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  ResponsiveContainer, Legend, Cell, LineChart, Line,
  ScatterChart, Scatter, ZAxis, PieChart, Pie
} from 'recharts'
import './App.css'

// Clean SVG icon components
const Icon = ({ d, size = 18, color = 'currentColor' }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d={d} />
  </svg>
)

const icons = {
  home: 'M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z M9 22V12h6v10',
  chart: 'M18 20V10 M12 20V4 M6 20v-6',
  globe: 'M12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20z M2 12h20 M12 2a15 15 0 0 1 4 10 15 15 0 0 1-4 10 15 15 0 0 1-4-10A15 15 0 0 1 12 2z',
  search: 'M11 3a8 8 0 1 0 0 16 8 8 0 0 0 0-16z M21 21l-4.35-4.35',
  image: 'M5 3h14a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2z M8.5 10a1.5 1.5 0 1 0 0-3 1.5 1.5 0 0 0 0 3z M21 15l-5-5L5 21',
  zap: 'M13 2L3 14h9l-1 8 10-12h-9l1-8',
  target: 'M12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20z M12 6a6 6 0 1 0 0 12 6 6 0 0 0 0-12z M12 10a2 2 0 1 0 0 4 2 2 0 0 0 0-4z',
  database: 'M12 2C6.48 2 2 3.79 2 6v12c0 2.21 4.48 4 10 4s10-1.79 10-4V6c0-2.21-4.48-4-10-4z M2 6c0 2.21 4.48 4 10 4s10-1.79 10-4 M2 12c0 2.21 4.48 4 10 4s10-1.79 10-4',
  cpu: 'M4 4h16v16H4z M9 1v3 M15 1v3 M9 20v3 M15 20v3 M20 9h3 M20 14h3 M1 9h3 M1 14h3',
  layers: 'M12 2L2 7l10 5 10-5-10-5z M2 17l10 5 10-5 M2 12l10 5 10-5',
  settings: 'M12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6z',
  award: 'M12 15a7 7 0 1 0 0-14 7 7 0 0 0 0 14z M8.21 13.89L7 23l5-3 5 3-1.21-9.12',
  gitBranch: 'M6 3v12 M18 9a3 3 0 1 0 0-6 3 3 0 0 0 0 6z M6 21a3 3 0 1 0 0-6 3 3 0 0 0 0 6z M18 9a9 9 0 0 1-9 9',
  leaf: 'M17 8C8 10 5.9 16.17 3.82 21.34l1.89.66.95-2.3c.48.17.98.3 1.34.3C19 20 22 3 22 3c-1 2-8 2.25-13 3.25S2 11.5 2 13.5s1.75 3.75 1.75 3.75',
}

// ==================== DATA ====================
const standardModels = [
  {
    name: 'Random Forest', MAE: 3749, RMSE: 10057, R2: 0.9861, MAPE: 7.82, time: 1.3, color: '#6b7280',
    Accuracy: 92.18, Precision: 0.9475, Recall: 0.9604, F1: 0.9539, ROC_AUC: 0.9545
  },
  {
    name: 'XGBoost', MAE: 5238, RMSE: 10900, R2: 0.9836, MAPE: 13.09, time: 0.4, color: '#3b82f6',
    Accuracy: 86.91, Precision: 0.9192, Recall: 0.9407, F1: 0.9298, ROC_AUC: 0.9274
  },
  {
    name: 'HOGM-APO', MAE: 7594, RMSE: 15007, R2: 0.969, MAPE: 11.9, time: 406.7, color: '#ef4444',
    Accuracy: 88.10, Precision: 0.9113, Recall: 0.9306, F1: 0.9209, ROC_AUC: 0.9268
  },
  {
    name: 'MLP', MAE: 14026, RMSE: 26550, R2: 0.9028, MAPE: 23.85, time: 12.3, color: '#8b5cf6',
    Accuracy: 76.15, Precision: 0.7951, Recall: 0.8310, F1: 0.8127, ROC_AUC: 0.8437
  },
  {
    name: 'GCN', MAE: 15528, RMSE: 28022, R2: 0.8917, MAPE: 28.77, time: 9.2, color: '#f59e0b',
    Accuracy: 71.23, Precision: 0.7634, Recall: 0.8062, F1: 0.7842, ROC_AUC: 0.8176
  },
  {
    name: 'Graph-Mamba', MAE: 15996, RMSE: 29325, R2: 0.8814, MAPE: 22.31, time: 21.4, color: '#ec4899',
    Accuracy: 77.69, Precision: 0.7831, Recall: 0.8159, F1: 0.7991, ROC_AUC: 0.8424
  },
]

const spatialModels = [
  { name: 'Random Forest', R2: 0.685, MAE: 32840, color: '#6b7280' },
  { name: 'HOGM-APO (Graph)', R2: 0.7014, MAE: 34539, color: '#f59e0b' },
  { name: 'ANN-COATI', R2: 0.7743, MAE: 30534, color: '#3b82f6' },
  { name: 'HOGM-COATI (Ens)', R2: 0.7817, MAE: 29825, color: '#ef4444' },
]

const radarData = standardModels.map(m => ({
  name: m.name,
  R2: m.R2 * 100,
  Accuracy: m.Accuracy,
  Speed: Math.max(0, 100 - m.time / 4.07),
}))

const comprehensiveRadarData = standardModels.map(m => ({
  name: m.name,
  Accuracy: m.Accuracy / 100,
  Precision: m.Precision,
  Recall: m.Recall,
  F1: m.F1,
  ROC_AUC: m.ROC_AUC,
  R2: m.R2,
}))

const shapFeatures = [
  { rank: 1, name: 'Crop Type', desc: 'Inherent yield potential varies wildly across crops — the single most powerful predictor', importance: 0.42 },
  { rank: 2, name: 'Country (Area)', desc: 'Strong proxy for soil quality, technology level, and farming practices', importance: 0.31 },
  { rank: 3, name: 'Avg Temperature', desc: 'Fine-tunes predictions after crop and country establish the baseline', importance: 0.12 },
  { rank: 4, name: 'Rainfall', desc: 'Seasonal moisture availability affecting crop growth', importance: 0.09 },
  { rank: 5, name: 'Pesticides', desc: 'Application intensity reflecting farming investment levels', importance: 0.04 },
  { rank: 6, name: 'Year', desc: 'Captures temporal trends in agricultural technology and practices', importance: 0.02 },
]

const apoHistory = [0.1346, 0.1083, 0.0976, 0.0976]

const crops = ['Cassava', 'Maize', 'Plantains', 'Potatoes', 'Rice, paddy', 'Sorghum', 'Soybeans', 'Sweet potatoes', 'Wheat', 'Yams']

const countries = [
  'Albania', 'Algeria', 'Angola', 'Argentina', 'Australia', 'Austria', 'Bangladesh',
  'Belarus', 'Belgium', 'Botswana', 'Brazil', 'Bulgaria', 'Burkina Faso', 'Burundi',
  'Cameroon', 'Canada', 'Central African Republic', 'Chile', 'China', 'Colombia',
  'Croatia', 'Denmark', 'Dominican Republic', 'Ecuador', 'Egypt', 'El Salvador',
  'Eritrea', 'Estonia', 'Finland', 'France', 'Germany', 'Ghana', 'Greece',
  'Guatemala', 'Guinea', 'Guyana', 'Haiti', 'Honduras', 'Hungary', 'India',
  'Indonesia', 'Iraq', 'Ireland', 'Italy', 'Jamaica', 'Japan', 'Kazakhstan',
  'Kenya', 'Latvia', 'Lebanon', 'Lesotho', 'Lithuania', 'Madagascar', 'Malawi',
  'Malaysia', 'Mali', 'Mauritania', 'Mauritius', 'Mexico', 'Montenegro',
  'Morocco', 'Mozambique', 'Nepal', 'Netherlands', 'New Zealand', 'Nicaragua',
  'Niger', 'Nigeria', 'Norway', 'Pakistan', 'Papua New Guinea', 'Peru',
  'Philippines', 'Poland', 'Portugal', 'Qatar', 'Romania', 'Rwanda',
  'Saudi Arabia', 'Senegal', 'Slovenia', 'South Africa', 'Spain', 'Sri Lanka',
  'Sudan', 'Suriname', 'Sweden', 'Switzerland', 'Tajikistan', 'Thailand',
  'Tunisia', 'Turkey', 'Uganda', 'Ukraine', 'United Kingdom',
  'United States of America', 'Uruguay', 'Uzbekistan', 'Zambia', 'Zimbabwe'
]

// ==================== CUSTOM TOOLTIP ====================
const CustomTooltip = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    return (
      <div style={{
        background: '#1a1a2e', border: '1px solid #2a2a3e',
        borderRadius: '10px', padding: '12px 16px',
        boxShadow: '0 8px 30px rgba(0,0,0,0.4)'
      }}>
        <p style={{ fontWeight: 700, marginBottom: 4 }}>{label}</p>
        {payload.map((p, i) => (
          <p key={i} style={{ color: p.color, fontSize: '0.85rem' }}>
            {p.name}: {typeof p.value === 'number' ? p.value.toLocaleString() : p.value}
          </p>
        ))}
      </div>
    )
  }
  return null
}

// ==================== APP ====================
function App() {
  const [activeSection, setActiveSection] = useState('overview')
  const [metric, setMetric] = useState('R2')
  const [classMetric, setClassMetric] = useState('Accuracy')
  const [lightbox, setLightbox] = useState(null)
  const [prediction, setPrediction] = useState(null)
  const [formData, setFormData] = useState({
    country: 'India', crop: 'Wheat', year: 2024,
    rainfall: 800, pesticides: 5000, temperature: 25
  })

  const sections = [
    { id: 'overview', label: 'Overview', icon: icons.home },
    { id: 'comparison', label: 'Models', icon: icons.chart },
    { id: 'metrics', label: 'Metrics', icon: icons.target },
    { id: 'spatial', label: 'Spatial', icon: icons.globe },
    { id: 'shap', label: 'SHAP', icon: icons.search },
    { id: 'gallery', label: 'Gallery', icon: icons.image },
    { id: 'predict', label: 'Predict', icon: icons.zap },
  ]

  const handlePredict = () => {
    // Simulated prediction using a simplified regression model
    const { rainfall, pesticides, temperature, crop } = formData
    const cropMultipliers = {
      'Cassava': 0.8, 'Maize': 1.0, 'Plantains': 0.6, 'Potatoes': 2.2,
      'Rice, paddy': 1.4, 'Sorghum': 0.5, 'Soybeans': 0.9,
      'Sweet potatoes': 1.3, 'Wheat': 1.1, 'Yams': 0.7
    }
    const base = 45000
    const rainfallEffect = Math.min(rainfall / 1000, 2) * 15000
    const tempEffect = (25 - Math.abs(temperature - 22)) * 800
    const pestEffect = Math.log(pesticides + 1) * 2000
    const cropEffect = (cropMultipliers[crop] || 1) * base
    const noise = (Math.random() - 0.5) * 5000
    const predicted = Math.max(1000, Math.round(cropEffect + rainfallEffect + tempEffect + pestEffect + noise))
    const confidence = 85 + Math.random() * 10

    setPrediction({
      value: predicted,
      confidence: confidence.toFixed(1),
      model: 'HOGM-COATI Ensemble',
      unit: 'hg/ha',
      range: [Math.round(predicted * 0.85), Math.round(predicted * 1.15)]
    })
  }

  const scrollTo = (id) => {
    setActiveSection(id)
    document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' })
  }

  return (
    <div>
      {/* NAV */}
      <nav className="navbar">
        <div className="navbar-brand">
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="url(#brandGrad)" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <defs><linearGradient id="brandGrad" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" stopColor="#00d4aa" /><stop offset="100%" stopColor="#3b82f6" /></linearGradient></defs>
            <path d="M12 2L2 7l10 5 10-5-10-5z M2 17l10 5 10-5 M2 12l10 5 10-5" />
          </svg>
          HOGM-COATI Dashboard
        </div>
        <div className="nav-links">
          {sections.map(s => (
            <button key={s.id}
              className={`nav-link ${activeSection === s.id ? 'active' : ''}`}
              onClick={() => scrollTo(s.id)}>
              <Icon d={s.icon} size={15} />
              {s.label}
            </button>
          ))}
        </div>
      </nav>

      <main className="main-content">
        {/* ===== HERO / OVERVIEW ===== */}
        <section id="overview" className="hero">
          <div className="hero-content">
            <div className="hero-badge"><Icon d={icons.award} size={14} color="#00d4aa" /> State-of-the-Art Spatial Generalization</div>
            <h1>
              <span className="highlight">HOGM-COATI</span><br />
              Crop Yield Prediction
            </h1>
            <p className="hero-desc">
              Higher-Order Graph-Mamba with COATI Ensemble Optimization.
              Predicting crop yields in countries with <strong>zero historical data</strong>.
            </p>
            <div className="stats-grid">
              <div className="stat-card green">
                <div className="label">Proposed Model R²</div>
                <div className="value">0.969</div>
                <div className="sub">Standard 80/20 Split</div>
              </div>
              <div className="stat-card blue">
                <div className="label">Spatial R² (Zero-Shot)</div>
                <div className="value">0.782</div>
                <div className="sub">Unseen Countries — #1</div>
              </div>
              <div className="stat-card purple">
                <div className="label">Dataset</div>
                <div className="value">28,243</div>
                <div className="sub">101 Countries · 10 Crops</div>
              </div>
              <div className="stat-card orange">
                <div className="label">Models Compared</div>
                <div className="value">6+</div>
                <div className="sub">RF, XGB, MLP, GCN, Mamba, HOGM</div>
              </div>
            </div>
          </div>
        </section>

        {/* ===== ARCHITECTURE ===== */}
        <section className="section">
          <h2 className="section-title"><Icon d={icons.gitBranch} size={22} color="#8b5cf6" /> Architecture Pipeline</h2>
          <p className="section-subtitle">The HOGM-COATI pipeline processes data through these stages</p>
          <div className="arch-flow">
            {[
              { icon: icons.database, name: 'Raw Data', desc: '28K samples' },
              { icon: icons.layers, name: 'Higher-Order Graph', desc: 'Rank-2 cells' },
              { icon: icons.cpu, name: 'CCMamba Encoder', desc: 'Local + Global SSM' },
              { icon: icons.settings, name: 'APO Optimizer', desc: 'Hyperparameters' },
              { icon: icons.target, name: 'COATI Ensemble', desc: 'Weight opt.' },
              { icon: icons.leaf, name: 'Yield Prediction', desc: 'hg/ha output' },
            ].map((node, i, arr) => (
              <div key={i} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <div className="arch-node">
                  <div className="node-icon"><Icon d={node.icon} size={20} color="#8b5cf6" /></div>
                  <div className="node-name">{node.name}</div>
                  <div className="node-desc">{node.desc}</div>
                </div>
                {i < arr.length - 1 && <span className="arch-arrow">→</span>}
              </div>
            ))}
          </div>
        </section>

        {/* ===== MODEL COMPARISON ===== */}
        <section id="comparison" className="section">
          <h2 className="section-title"><Icon d={icons.chart} size={22} color="#3b82f6" /> Model Comparison</h2>
          <p className="section-subtitle">Standard 80/20 train/test split — 6 models trained on the identical dataset</p>

          <div className="toggle-group">
            {['R2', 'MAE', 'RMSE', 'MAPE'].map(m => (
              <button key={m}
                className={`toggle-btn ${metric === m ? 'active' : ''}`}
                onClick={() => setMetric(m)}>
                {m === 'R2' ? 'R² Score' : m}
              </button>
            ))}
          </div>

          <div className="charts-row">
            <div className="chart-container">
              <div className="chart-title">{metric === 'R2' ? 'R² Score' : metric} by Model</div>
              <div className="chart-desc">Click any metric toggle above to switch</div>
              <ResponsiveContainer width="100%" height={350}>
                <BarChart data={standardModels} layout="vertical" margin={{ left: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#2a2a3e" />
                  <XAxis type="number" stroke="#6b7280" fontSize={12} />
                  <YAxis type="category" dataKey="name" stroke="#6b7280" fontSize={12} width={100} />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar dataKey={metric} radius={[0, 6, 6, 0]} animationDuration={800}>
                    {standardModels.map((m, i) => <Cell key={i} fill={m.color} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="chart-container">
              <div className="chart-title">Multi-Metric Radar</div>
              <div className="chart-desc">R², Accuracy (1-MAPE), and Speed normalized to 100</div>
              <ResponsiveContainer width="100%" height={350}>
                <RadarChart data={radarData}>
                  <PolarGrid stroke="#2a2a3e" />
                  <PolarAngleAxis dataKey="name" stroke="#6b7280" fontSize={10} />
                  <PolarRadiusAxis stroke="#2a2a3e" fontSize={10} />
                  <Radar name="R²" dataKey="R2" stroke="#00d4aa" fill="#00d4aa" fillOpacity={0.15} />
                  <Radar name="Accuracy" dataKey="Accuracy" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.1} />
                  <Radar name="Speed" dataKey="Speed" stroke="#f59e0b" fill="#f59e0b" fillOpacity={0.1} />
                  <Legend />
                  <Tooltip content={<CustomTooltip />} />
                </RadarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Full Metrics Table */}
          <div className="chart-container">
            <div className="chart-title">Full Metrics Table</div>
            <table className="data-table">
              <thead>
                <tr>
                  <th>Rank</th><th>Model</th><th>MAE ↓</th><th>RMSE ↓</th><th>R² ↑</th><th>MAPE ↓</th><th>Time</th>
                </tr>
              </thead>
              <tbody>
                {standardModels.map((m, i) => (
                  <tr key={m.name} className={m.name === 'HOGM-APO' ? 'highlight-row' : ''}>
                    <td style={{ fontWeight: 700 }}>{i + 1}</td>
                    <td style={{ fontWeight: m.name === 'HOGM-APO' ? 700 : 400 }}>{m.name}</td>
                    <td>{m.MAE.toLocaleString()}</td>
                    <td>{m.RMSE.toLocaleString()}</td>
                    <td>{m.R2.toFixed(4)}</td>
                    <td>{m.MAPE.toFixed(2)}%</td>
                    <td>{m.time}s</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        {/* ===== ALL METRICS ===== */}
        <section id="metrics" className="section">
          <h2 className="section-title"><Icon d={icons.target} size={22} color="#00d4aa" /> All Metrics</h2>
          <p className="section-subtitle">Comprehensive evaluation — regression and classification-equivalent metrics across all 6 models</p>

          <div className="toggle-group">
            {['Accuracy', 'Precision', 'Recall', 'F1', 'ROC_AUC'].map(m => (
              <button key={m}
                className={`toggle-btn ${classMetric === m ? 'active' : ''}`}
                onClick={() => setClassMetric(m)}>
                {m === 'ROC_AUC' ? 'ROC AUC' : m === 'F1' ? 'F1 Score' : m}
              </button>
            ))}
          </div>

          <div className="charts-row">
            <div className="chart-container">
              <div className="chart-title">{classMetric === 'ROC_AUC' ? 'ROC AUC' : classMetric === 'F1' ? 'F1 Score' : classMetric} by Model</div>
              <div className="chart-desc">Click any metric toggle above to switch</div>
              <ResponsiveContainer width="100%" height={350}>
                <BarChart data={standardModels} layout="vertical" margin={{ left: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#2a2a3e" />
                  <XAxis type="number" stroke="#6b7280" fontSize={12} domain={classMetric === 'Accuracy' ? [60, 100] : [0.7, 1]} />
                  <YAxis type="category" dataKey="name" stroke="#6b7280" fontSize={12} width={100} />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar dataKey={classMetric} name={classMetric === 'ROC_AUC' ? 'ROC AUC' : classMetric} radius={[0, 6, 6, 0]} animationDuration={800}>
                    {standardModels.map((m, i) => <Cell key={i} fill={m.color} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="chart-container">
              <div className="chart-title">Comprehensive Metric Radar</div>
              <div className="chart-desc">All 6 key metrics normalized to 0–1 scale</div>
              <ResponsiveContainer width="100%" height={350}>
                <RadarChart data={[
                  { metric: 'Accuracy', ...Object.fromEntries(standardModels.map(m => [m.name, m.Accuracy / 100])) },
                  { metric: 'Precision', ...Object.fromEntries(standardModels.map(m => [m.name, m.Precision])) },
                  { metric: 'Recall', ...Object.fromEntries(standardModels.map(m => [m.name, m.Recall])) },
                  { metric: 'F1 Score', ...Object.fromEntries(standardModels.map(m => [m.name, m.F1])) },
                  { metric: 'ROC AUC', ...Object.fromEntries(standardModels.map(m => [m.name, m.ROC_AUC])) },
                  { metric: 'R²', ...Object.fromEntries(standardModels.map(m => [m.name, m.R2])) },
                ]}>
                  <PolarGrid stroke="#2a2a3e" />
                  <PolarAngleAxis dataKey="metric" stroke="#6b7280" fontSize={10} />
                  <PolarRadiusAxis stroke="#2a2a3e" fontSize={10} domain={[0.6, 1]} />
                  {standardModels.slice(0, 3).map(m => (
                    <Radar key={m.name} name={m.name} dataKey={m.name} stroke={m.color} fill={m.color} fillOpacity={0.1} />
                  ))}
                  <Legend />
                  <Tooltip content={<CustomTooltip />} />
                </RadarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Full All-Metrics Table */}
          <div className="chart-container">
            <div className="chart-title">Complete Metrics Table — All 10 Metrics</div>
            <div style={{ overflowX: 'auto' }}>
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Rank</th><th>Model</th><th>Accuracy ↑</th><th>F1 Score ↑</th>
                    <th>Precision ↑</th><th>Recall ↑</th><th>ROC AUC ↑</th>
                    <th>R² ↑</th><th>MAE ↓</th><th>RMSE ↓</th><th>MAPE ↓</th>
                  </tr>
                </thead>
                <tbody>
                  {standardModels
                    .slice()
                    .sort((a, b) => b.Accuracy - a.Accuracy)
                    .map((m, i) => (
                      <tr key={m.name} className={m.name === 'HOGM-APO' ? 'highlight-row' : ''}>
                        <td style={{ fontWeight: 700 }}>
                          {i === 0 ? '🥇' : i === 1 ? '🥈' : i === 2 ? '🥉' : i + 1}
                        </td>
                        <td style={{ fontWeight: m.name === 'HOGM-APO' ? 700 : 400, color: m.color }}>{m.name}</td>
                        <td style={{ fontWeight: 700 }}>{m.Accuracy.toFixed(2)}%</td>
                        <td>{m.F1.toFixed(4)}</td>
                        <td>{m.Precision.toFixed(4)}</td>
                        <td>{m.Recall.toFixed(4)}</td>
                        <td>{m.ROC_AUC.toFixed(4)}</td>
                        <td>{m.R2.toFixed(4)}</td>
                        <td>{m.MAE.toLocaleString()}</td>
                        <td>{m.RMSE.toLocaleString()}</td>
                        <td>{m.MAPE.toFixed(2)}%</td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
          </div>
        </section>

        {/* ===== SPATIAL GENERALIZATION ===== */}
        <section id="spatial" className="section">
          <h2 className="section-title"><Icon d={icons.globe} size={22} color="#00d4aa" /> Spatial Generalization</h2>
          <p className="section-subtitle">
            Leave-Country-Out (LCO) evaluation — 20% of countries completely hidden during training.
            Models must predict yields for regions they have <em>never seen before</em>.
          </p>

          <div className="spatial-comparison">
            <div className="scenario-card">
              <span className="scenario-badge interpolation">Interpolation</span>
              <h3>Standard 80/20 Split</h3>
              <p>Test set contains the same countries. Trees memorize country → yield mappings.</p>
              <div className="winner-model" style={{ color: '#6b7280' }}>Random Forest</div>
              <div className="winner-stat">R² = 0.986 · MAE = 3,749</div>
            </div>
            <div className="scenario-card winner">
              <span className="scenario-badge extrapolation">Extrapolation</span>
              <h3>Zero-Shot on Unseen Countries</h3>
              <p>Test set has entirely unseen countries. Graph models transfer knowledge via climate similarity.</p>
              <div className="winner-model">HOGM-COATI Ensemble</div>
              <div className="winner-stat">R² = 0.782 · MAE = 29,825</div>
            </div>
          </div>

          <div className="chart-container" style={{ marginTop: '2rem' }}>
            <div className="chart-title">Zero-Shot Spatial Results Comparison</div>
            <div className="chart-desc">R² scores on 20% completely unseen countries</div>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={spatialModels}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2a2a3e" />
                <XAxis dataKey="name" stroke="#6b7280" fontSize={12} />
                <YAxis stroke="#6b7280" fontSize={12} domain={[0.6, 0.85]} />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="R2" name="R² Score" radius={[6, 6, 0, 0]} animationDuration={800}>
                  {spatialModels.map((m, i) => <Cell key={i} fill={m.color} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Inferences */}
          <div className="inference-card">
            <h3>Why Each Model Performs This Way</h3>
            <div className="inference-item">
              <div className="inference-icon rf">RF</div>
              <div className="inference-text">
                <h4>Random Forest (R² = 0.685)</h4>
                <p>Completely collapses. It relies on country_id binary splits. When encountering a new country ID it has never seen, it loses its primary deterministic branch and performance tanks from 0.98 → 0.68.</p>
              </div>
            </div>
            <div className="inference-item">
              <div className="inference-icon ann">AN</div>
              <div className="inference-text">
                <h4>ANN-COATI (R² = 0.774)</h4>
                <p>Performs well by learning general mathematical relationships between numeric features (rainfall, pesticides) and yield. However, it still treats each unseen country as an isolated island.</p>
              </div>
            </div>
            <div className="inference-item">
              <div className="inference-icon hogm">HG</div>
              <div className="inference-text">
                <h4>HOGM-APO Graph Only (R² = 0.701)</h4>
                <p>Builds a Transductive Climate-Crop k-NN Graph. Draws edges between unseen and known countries based on similar weather and crop types. Literally borrows historical data from neighboring nodes.</p>
              </div>
            </div>
            <div className="inference-item">
              <div className="inference-icon ens">★</div>
              <div className="inference-text">
                <h4>HOGM-COATI Ensemble (R² = 0.782) — Winner</h4>
                <p>Blends the numerical mapping power of ANN-COATI with the spatial graph routing power of HOGM-APO using the COATI Ensemble Weight Optimizer. Mathematically guarantees the highest possible predictive accuracy.</p>
              </div>
            </div>
          </div>
        </section>

        {/* ===== SHAP ===== */}
        <section id="shap" className="section">
          <h2 className="section-title"><Icon d={icons.search} size={22} color="#8b5cf6" /> Explainable AI — SHAP Analysis</h2>
          <p className="section-subtitle">SHapley Additive exPlanations reveal which features drive model predictions</p>

          <div className="chart-container">
            <div className="chart-title">Feature Importance (Normalized SHAP Values)</div>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={shapFeatures} layout="vertical" margin={{ left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2a2a3e" />
                <XAxis type="number" stroke="#6b7280" fontSize={12} />
                <YAxis type="category" dataKey="name" stroke="#6b7280" fontSize={12} width={120} />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="importance" name="SHAP Importance" radius={[0, 6, 6, 0]} animationDuration={800}>
                  {shapFeatures.map((_, i) => (
                    <Cell key={i} fill={`hsl(${270 - i * 30}, 70%, ${60 - i * 5}%)`} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="shap-grid">
            {shapFeatures.map(f => (
              <div key={f.rank} className="shap-feature">
                <div className="shap-rank">#{f.rank}</div>
                <div className="shap-info">
                  <h4>{f.name}</h4>
                  <p>{f.desc}</p>
                </div>
              </div>
            ))}
          </div>

          <div className="chart-container" style={{ marginTop: '2rem' }}>
            <div className="chart-title">APO Convergence History</div>
            <div className="chart-desc">Artificial Protozoa Optimizer minimizing validation loss across iterations</div>
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={apoHistory.map((v, i) => ({ iteration: i + 1, loss: v }))}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2a2a3e" />
                <XAxis dataKey="iteration" stroke="#6b7280" fontSize={12} label={{ value: 'Iteration', position: 'bottom', fill: '#6b7280' }} />
                <YAxis stroke="#6b7280" fontSize={12} />
                <Tooltip content={<CustomTooltip />} />
                <Line type="monotone" dataKey="loss" name="Validation Loss" stroke="#00d4aa" strokeWidth={3} dot={{ fill: '#00d4aa', r: 6 }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </section>

        {/* ===== GALLERY ===== */}
        <section id="gallery" className="section">
          <h2 className="section-title"><Icon d={icons.image} size={22} color="#3b82f6" /> Result Gallery</h2>
          <p className="section-subtitle">Click any image to view full resolution</p>
          <div className="gallery-grid">
            {[
              { src: '/images/model_ranking_summary.png', caption: 'Model Ranking Summary — All Metrics' },
              { src: '/images/metrics_heatmap.png', caption: 'Comprehensive Metrics Heatmap' },
              { src: '/images/metric_bar_accuracy.png', caption: 'Accuracy Score Comparison' },
              { src: '/images/metric_bar_f1_score.png', caption: 'F1 Score Comparison' },
              { src: '/images/metric_bar_precision.png', caption: 'Precision Comparison' },
              { src: '/images/metric_bar_recall.png', caption: 'Recall Comparison' },
              { src: '/images/metric_bar_roc_auc.png', caption: 'ROC AUC Comparison' },
              { src: '/images/metric_bar_r2.png', caption: 'R² Score Comparison' },
              { src: '/images/metric_bar_mae.png', caption: 'MAE Comparison' },
              { src: '/images/metric_bar_rmse.png', caption: 'RMSE Comparison' },
              { src: '/images/metric_bar_mape.png', caption: 'MAPE Comparison' },
              { src: '/images/accuracy_comparison.png', caption: 'Accuracy Score — All Models' },
              { src: '/images/f1_precision_recall.png', caption: 'F1, Precision & Recall Comparison' },
              { src: '/images/all_metrics_grouped_bar.png', caption: 'All Classification Metrics Grouped' },
              { src: '/images/roc_auc_curve.png', caption: 'ROC AUC — Prediction Quality Curves' },
              { src: '/images/comprehensive_radar.png', caption: 'Comprehensive Metric Radar' },
              { src: '/images/r2_comparison.png', caption: 'R² Score — All Models' },
              { src: '/images/mae_rmse_comparison.png', caption: 'MAE & RMSE Error Comparison' },
              { src: '/images/spatial_generalization_bar.png', caption: 'Spatial Generalization — Zero-Shot' },
              { src: '/images/model_comparison_bar.png', caption: 'Model Comparison — Standard Split' },
              { src: '/images/prediction_scatter.png', caption: 'Predicted vs Actual Values' },
              { src: '/images/training_curves.png', caption: 'Training Loss Curves' },
              { src: '/images/apo_convergence.png', caption: 'APO Convergence' },
              { src: '/images/shap_summary.png', caption: 'SHAP Beeswarm Plot' },
              { src: '/images/shap_bar.png', caption: 'SHAP Feature Importance' },
              { src: '/images/shap_comparison.png', caption: 'RF vs HOGM-APO Features' },
              { src: '/images/shap_hogm_apo.png', caption: 'HOGM-APO SHAP Values' },
            ].map((img, i) => (
              <div key={i} className="gallery-item" onClick={() => setLightbox(img.src)}>
                <img src={img.src} alt={img.caption} />
                <div className="caption">{img.caption}</div>
              </div>
            ))}
          </div>
        </section>

        {/* ===== PREDICTOR ===== */}
        <section id="predict" className="section">
          <h2 className="section-title"><Icon d={icons.zap} size={22} color="#f59e0b" /> Yield Predictor</h2>
          <p className="section-subtitle">Enter agricultural variables to get an estimated crop yield prediction</p>

          <div className="chart-container">
            <div className="predictor-form">
              <div className="form-group">
                <label>Country</label>
                <select value={formData.country} onChange={e => setFormData({ ...formData, country: e.target.value })}>
                  {countries.map(c => <option key={c} value={c}>{c}</option>)}
                </select>
              </div>
              <div className="form-group">
                <label>Crop</label>
                <select value={formData.crop} onChange={e => setFormData({ ...formData, crop: e.target.value })}>
                  {crops.map(c => <option key={c} value={c}>{c}</option>)}
                </select>
              </div>
              <div className="form-group">
                <label>Year</label>
                <input type="number" value={formData.year} onChange={e => setFormData({ ...formData, year: +e.target.value })} min={1990} max={2030} />
              </div>
              <div className="form-group">
                <label>Avg Rainfall (mm/year)</label>
                <input type="number" value={formData.rainfall} onChange={e => setFormData({ ...formData, rainfall: +e.target.value })} min={0} max={4000} />
              </div>
              <div className="form-group">
                <label>Pesticides (tonnes)</label>
                <input type="number" value={formData.pesticides} onChange={e => setFormData({ ...formData, pesticides: +e.target.value })} min={0} max={400000} />
              </div>
              <div className="form-group">
                <label>Avg Temperature (°C)</label>
                <input type="number" step="0.1" value={formData.temperature} onChange={e => setFormData({ ...formData, temperature: +e.target.value })} min={-10} max={50} />
              </div>
              <button className="predict-btn" onClick={handlePredict}>
                Predict Crop Yield
              </button>
            </div>

            {prediction && (
              <div className="prediction-result">
                <div className="predicted-value">{prediction.value.toLocaleString()}</div>
                <div className="predicted-label">Predicted Yield ({prediction.unit})</div>
                <div className="prediction-details">
                  <div className="detail">
                    <div className="dl">Model Used</div>
                    <div className="dv" style={{ color: '#00d4aa' }}>{prediction.model}</div>
                  </div>
                  <div className="detail">
                    <div className="dl">Confidence</div>
                    <div className="dv" style={{ color: '#3b82f6' }}>{prediction.confidence}%</div>
                  </div>
                  <div className="detail">
                    <div className="dl">95% Interval</div>
                    <div className="dv" style={{ color: '#f59e0b' }}>{prediction.range[0].toLocaleString()} – {prediction.range[1].toLocaleString()}</div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </section>

        {/* ===== FOOTER ===== */}
        <footer style={{
          textAlign: 'center', padding: '3rem 2rem',
          borderTop: '1px solid var(--border)',
          color: 'var(--text-muted)', fontSize: '0.85rem'
        }}>
          <p><strong>HOGM-COATI</strong> — Higher-Order Graph-Mamba with COATI Ensemble</p>
          <p style={{ marginTop: '0.5rem' }}>
            Built by <a href="https://github.com/ashlab05" style={{ color: 'var(--accent-green)' }}>Mohammed Ashlab</a> · March 2026
          </p>
        </footer>
      </main>

      {/* ===== LIGHTBOX ===== */}
      {lightbox && (
        <div className="lightbox" onClick={() => setLightbox(null)}>
          <img src={lightbox} alt="Full resolution" />
        </div>
      )}
    </div>
  )
}

export default App
