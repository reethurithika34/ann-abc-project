import React, { useEffect, useMemo, useState } from 'react';
import axios from 'axios';
import { motion } from 'framer-motion';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

const API = 'http://localhost:5000/api';
const motionUp = {
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.45 },
};

const chartColors = ['#22c55e', '#a855f7', '#06b6d4', '#f59e0b', '#f43f5e'];

const defaultConfig = {
  hidden_size: 8,
  learning_rate: 0.1,
  batch_size: 64,
  max_epochs: 50,
  patience: 8,
  population_size: 20,
  iterations: 20,
  limit: 4,
  search_runs: 3,
  auto_retry_max: 2,
};

function formatPct(value) {
  return value === null || value === undefined ? '--' : `${(value * 100).toFixed(2)}%`;
}

function formatNumber(value, digits = 4) {
  return value === null || value === undefined ? '--' : Number(value).toFixed(digits);
}

function downloadCsv(rows) {
  if (!rows?.length) return;
  const headers = Object.keys(rows[0]);
  const csv = [
    headers.join(','),
    ...rows.map((row) =>
      headers
        .map((header) => `"${String(row[header] ?? '').replaceAll('"', '""')}"`)
        .join(',')
    ),
  ].join('\n');
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = 'ann_abc_experiment_results.csv';
  link.click();
  URL.revokeObjectURL(url);
}

function MetricCard({ title, value, subtitle, accent = 'neutral' }) {
  const accentClasses =
    accent === 'success'
      ? 'border-emerald-400/30 bg-emerald-500/10'
      : accent === 'primary'
        ? 'border-cyan-400/20 bg-white/10'
        : 'border-white/10 bg-white/5';

  return (
    <motion.div
      {...motionUp}
      className={`rounded-3xl border p-5 shadow-2xl shadow-black/10 backdrop-blur-xl ${accentClasses}`}
    >
      <p className="text-xs uppercase tracking-[0.25em] text-slate-400">{title}</p>
      <div className="mt-3 text-3xl font-semibold text-white">{value}</div>
      <p className="mt-2 text-sm text-slate-300">{subtitle}</p>
    </motion.div>
  );
}

function SectionCard({ title, subtitle, children, action }) {
  return (
    <motion.section
      {...motionUp}
      className="rounded-3xl border border-white/10 bg-white/5 p-6 shadow-2xl shadow-black/10 backdrop-blur-xl"
    >
      <div className="mb-5 flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
        <div>
          <h2 className="text-xl font-semibold text-white">{title}</h2>
          <p className="text-sm text-slate-300">{subtitle}</p>
        </div>
        {action}
      </div>
      {children}
    </motion.section>
  );
}

function InputField({ label, value, onChange, type = 'number', step = '1', title }) {
  return (
    <label className="flex flex-col gap-2">
      <span className="text-sm font-medium text-slate-200" title={title}>
        {label}
      </span>
      <input
        type={type}
        step={step}
        value={value}
        onChange={onChange}
        className="rounded-2xl border border-white/10 bg-slate-950/60 px-4 py-3 text-white outline-none transition focus:border-cyan-400/60"
      />
    </label>
  );
}

export default function App() {
  const [dataset, setDataset] = useState(null);
  const [status, setStatus] = useState({ running: false, progress: 0, stage: 'idle', active_job: null });
  const [results, setResults] = useState({
    baseline: null,
    optimized: null,
    comparison: null,
    comparison_chart: [],
    csv_rows: [],
  });
  const [convergence, setConvergence] = useState({ baseline: [], optimization: [], abc_search: [] });
  const [logs, setLogs] = useState([]);
  const [config, setConfig] = useState(defaultConfig);
  const [isStarting, setIsStarting] = useState(false);

  const fetchAll = async () => {
    const [datasetResponse, resultsResponse, convergenceResponse] = await Promise.all([
      axios.get(`${API}/dataset-info`),
      axios.get(`${API}/get-results`),
      axios.get(`${API}/get-convergence-data`),
    ]);
    const serverResults = resultsResponse.data.results || {
      baseline: null,
      optimized: null,
      comparison: null,
      comparison_chart: [],
      csv_rows: [],
    };
    setDataset(datasetResponse.data.dataset);
    setResults(serverResults);
    setStatus({
      running: resultsResponse.data.running,
      progress: resultsResponse.data.progress,
      stage: resultsResponse.data.stage,
      active_job: resultsResponse.data.active_job,
    });
    setLogs(resultsResponse.data.logs || []);
    setConvergence(convergenceResponse.data.convergence || { baseline: [], optimization: [], abc_search: [] });
  };

  useEffect(() => {
    fetchAll().catch(() => null);
    const interval = setInterval(() => {
      fetchAll().catch(() => null);
    }, 2200);
    return () => clearInterval(interval);
  }, []);

  const setValue = (key, rawValue) => {
    setConfig((current) => ({
      ...current,
      [key]: key.includes('learning') ? parseFloat(rawValue || '0') : parseInt(rawValue || '0', 10),
    }));
  };

  const runBaseline = async () => {
    if (status.running || isStarting) {
      return;
    }

    setIsStarting(true);
    try {
      await axios.post(`${API}/run-baseline`, {
        hidden_size: config.hidden_size,
        learning_rate: config.learning_rate,
        batch_size: config.batch_size,
        max_epochs: config.max_epochs,
        patience: config.patience,
      });
    } catch (error) {
      const message = error.response?.data?.message || error.message || 'Failed to start baseline.';
      setLogs((current) => [...current, `[${new Date().toLocaleTimeString()}] ${message}`].slice(-250));
    } finally {
      setIsStarting(false);
      fetchAll().catch(() => null);
    }
  };

  const runOptimization = async () => {
    if (status.running || isStarting) {
      return;
    }

    setIsStarting(true);
    try {
      await axios.post(`${API}/run-abc-optimization`, {
        population_size: config.population_size,
        iterations: config.iterations,
        limit: config.limit,
        search_runs: config.search_runs,
        auto_retry_max: config.auto_retry_max,
      });
    } catch (error) {
      const message = error.response?.data?.message || error.message || 'Failed to start optimization.';
      setLogs((current) => [...current, `[${new Date().toLocaleTimeString()}] ${message}`].slice(-250));
    } finally {
      setIsStarting(false);
      fetchAll().catch(() => null);
    }
  };

  const rocData = useMemo(() => results.optimized?.roc_curve || results.baseline?.roc_curve || [], [results]);
  const pieData = useMemo(
    () => results.optimized?.prediction_distribution || results.baseline?.prediction_distribution || [],
    [results]
  );

  const summaryRows = useMemo(() => {
    const baseline = results?.baseline?.metrics;
    const optimized = results?.optimized?.metrics;
    return [
      {
        metric: 'Accuracy',
        baseline: baseline?.test_accuracy?.mean != null ? formatPct(baseline.test_accuracy.mean) : '--',
        optimized: optimized?.test_accuracy?.mean != null ? formatPct(optimized.test_accuracy.mean) : '--',
      },
      {
        metric: 'Loss',
        baseline: baseline?.test_loss?.mean != null ? formatNumber(baseline.test_loss.mean, 4) : '--',
        optimized: optimized?.test_loss?.mean != null ? formatNumber(optimized.test_loss.mean, 4) : '--',
      },
      {
        metric: 'F1-score',
        baseline: baseline?.f1_score?.mean != null ? formatNumber(baseline.f1_score.mean, 4) : '--',
        optimized: optimized?.f1_score?.mean != null ? formatNumber(optimized.f1_score.mean, 4) : '--',
      },
      {
        metric: 'Training time',
        baseline: baseline ? `${formatNumber(baseline.training_time.mean, 2)} s` : '--',
        optimized: optimized ? `${formatNumber(optimized.training_time.mean, 2)} s` : '--',
      },
    ];
  }, [results]);

  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_top_left,_rgba(59,130,246,0.20),_transparent_30%),radial-gradient(circle_at_top_right,_rgba(168,85,247,0.18),_transparent_25%),linear-gradient(180deg,_#07111f_0%,_#0f172a_55%,_#020617_100%)] px-4 py-6 text-slate-100 md:px-8">
      <div className="mx-auto flex max-w-7xl flex-col gap-6">
        <motion.header
          {...motionUp}
          className="overflow-hidden rounded-[2rem] border border-white/10 bg-slate-900/40 p-8 shadow-2xl shadow-black/20 backdrop-blur-xl"
        >
          <div className="grid gap-6 lg:grid-cols-[1.5fr_1fr]">
            <div>
              <span className="inline-flex rounded-full border border-cyan-400/20 bg-cyan-400/10 px-4 py-2 text-xs font-semibold uppercase tracking-[0.3em] text-cyan-300">
                Research-grade ANN hyperparameter optimization
              </span>
              <h1 className="mt-5 max-w-4xl text-4xl font-semibold leading-tight text-white md:text-6xl">
                Artificial Bee Colony optimization for Wine dataset neural networks.
              </h1>
              <p className="mt-5 max-w-3xl text-base leading-8 text-slate-300 md:text-lg">
                A reproducible experimentation dashboard with stratified 70/15/15 splitting,
                repeated runs, confidence intervals, significance testing, live logs, and
                convergence analysis.
              </p>
            </div>

            <div className="grid gap-4 sm:grid-cols-2">
              <MetricCard title="Dataset" value={dataset?.name || 'Loading...'} subtitle={dataset?.split || 'Preparing dataset summary'} accent="primary" />
              <MetricCard title="Samples" value={dataset?.total_samples || '--'} subtitle="Train/validation/test stratified split" />
              <MetricCard title="Features" value={dataset?.features || '--'} subtitle="UCI wine chemistry attributes" />
              <MetricCard
                title="Job status"
                value={status.running ? 'Running' : 'Idle'}
                subtitle={status.stage}
                accent={status.running ? 'primary' : 'neutral'}
              />
            </div>
          </div>
        </motion.header>

        <div className="grid gap-6 xl:grid-cols-[360px_minmax(0,1fr)]">
          <div className="flex flex-col gap-6">
            <SectionCard
              title="Experiment Control"
              subtitle="Baseline ANN uses the required single hidden layer with Adam and CrossEntropyLoss."
              action={
                <button
                  onClick={() => downloadCsv(results.csv_rows)}
                  className="rounded-full border border-cyan-400/20 bg-cyan-400/10 px-4 py-2 text-sm font-medium text-cyan-200 transition hover:bg-cyan-400/20"
                >
                  Export CSV
                </button>
              }
            >
              <div className="grid gap-4">
                <InputField label="Baseline hidden size" value={config.hidden_size} onChange={(e) => setValue('hidden_size', e.target.value)} title="Research baseline: single hidden layer width" />
                <InputField label="Learning rate" value={config.learning_rate} step="0.0001" onChange={(e) => setConfig((c) => ({ ...c, learning_rate: parseFloat(e.target.value || '0') }))} title="Adam learning rate for the baseline ANN" />
                <InputField label="Batch size" value={config.batch_size} onChange={(e) => setValue('batch_size', e.target.value)} title="Mini-batch size for baseline training" />
                <InputField label="Max epochs" value={config.max_epochs} onChange={(e) => setValue('max_epochs', e.target.value)} title="Upper bound on epochs before early stopping" />
                <InputField label="Patience" value={config.patience} onChange={(e) => setValue('patience', e.target.value)} title="Early stopping patience on validation loss" />
              </div>
              <button
                onClick={runBaseline}
                disabled={status.running || isStarting}
                className="mt-5 w-full rounded-2xl bg-slate-200 px-5 py-3 font-semibold text-slate-950 transition hover:bg-white disabled:cursor-not-allowed disabled:opacity-50"
              >
                Run baseline
              </button>

              {status.running && (
                <p className="mt-3 text-sm text-amber-300">
                  An experiment is already running: {status.stage}. Wait for it to complete before starting another.
                </p>
              )}

              <div className="mt-6 h-px bg-white/10" />

              <div className="mt-6 grid gap-4">
                <InputField label="ABC population size" value={config.population_size} onChange={(e) => setValue('population_size', e.target.value)} title="Number of food sources / bees in the ABC population" />
                <InputField label="ABC iterations" value={config.iterations} onChange={(e) => setValue('iterations', e.target.value)} title="Optimization cycles across employed/onlooker/scout phases" />
                <InputField label="Scout limit" value={config.limit} onChange={(e) => setValue('limit', e.target.value)} title="Abandonment threshold before scout reset" />
                <InputField label="Search runs per fitness" value={config.search_runs} onChange={(e) => setValue('search_runs', e.target.value)} title="Repeated candidate evaluation count for stability during search" />
                <InputField label="Auto rerun attempts" value={config.auto_retry_max} onChange={(e) => setValue('auto_retry_max', e.target.value)} title="Automatic re-search if optimization fails to improve over baseline" />
              </div>
              <button
                onClick={runOptimization}
                disabled={status.running || isStarting}
                className="mt-5 w-full rounded-2xl bg-emerald-500 px-5 py-3 font-semibold text-emerald-950 transition hover:bg-emerald-400 disabled:cursor-not-allowed disabled:opacity-50"
              >
                Run ABC optimization
              </button>
            </SectionCard>

            <SectionCard title="Live Experiment Status" subtitle="Real-time progress, stage, and training logs.">
              <div className="rounded-2xl border border-white/10 bg-slate-950/40 p-4">
                <div className="flex items-center justify-between text-sm text-slate-300">
                  <span>{status.stage}</span>
                  <span>{Math.round((status.progress || 0) * 100)}%</span>
                </div>
                <div className="mt-3 h-3 rounded-full bg-white/10">
                  <div
                    className="h-3 rounded-full bg-gradient-to-r from-cyan-400 via-violet-500 to-emerald-400 transition-all"
                    style={{ width: `${Math.round((status.progress || 0) * 100)}%` }}
                  />
                </div>
              </div>

              <div className="mt-5 max-h-[420px] space-y-2 overflow-y-auto rounded-2xl border border-white/10 bg-slate-950/50 p-4 font-mono text-xs text-slate-300">
                {logs.length ? logs.slice().reverse().map((log, index) => <div key={`${log}-${index}`}>{log}</div>) : <div>No logs yet. Start an experiment to stream training and optimization updates.</div>}
              </div>
            </SectionCard>
          </div>

          <div className="flex flex-col gap-6">
            <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
              <MetricCard
                title="Baseline Accuracy"
                value={results.baseline?.metrics?.test_accuracy?.mean != null ? formatPct(results.baseline.metrics.test_accuracy.mean) : '--'}
                subtitle="Mean test accuracy across repeated runs"
              />
              <MetricCard
                title="Optimized Accuracy"
                value={results.optimized?.metrics?.test_accuracy?.mean != null ? formatPct(results.optimized.metrics.test_accuracy.mean) : '--'}
                subtitle={
                  status.running && status.active_job === 'optimization'
                    ? 'ABC search is still running'
                    : 'Mean test accuracy after ABC search'
                }
                accent="success"
              />
              <MetricCard
                title="Loss Reduction"
                value={results.comparison?.loss_reduction_percent != null ? `${results.comparison.loss_reduction_percent.toFixed(2)}%` : '--'}
                subtitle="Relative reduction in test loss"
                accent="success"
              />
              <MetricCard
                title="Accuracy Gain"
                value={results.comparison?.accuracy_gain_percent != null ? `${results.comparison.accuracy_gain_percent.toFixed(2)}%` : '--'}
                subtitle="Relative uplift over baseline"
                accent="success"
              />
            </div>

            <SectionCard title="Scientific Summary" subtitle="Repeated-run metrics, standard deviation, confidence interval, and significance indicators.">
              <div className="overflow-hidden rounded-3xl border border-white/10">
                <table className="min-w-full divide-y divide-white/10 text-sm">
                  <thead className="bg-white/5 text-slate-300">
                    <tr>
                      <th className="px-4 py-3 text-left">Metric</th>
                      <th className="px-4 py-3 text-left">Baseline</th>
                      <th className="px-4 py-3 text-left">Optimized</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-white/10 bg-slate-950/20">
                    {summaryRows.map((row) => (
                      <tr key={row.metric}>
                        <td className="px-4 py-3 text-slate-200">{row.metric}</td>
                        <td className="px-4 py-3 text-slate-300">{row.baseline}</td>
                        <td className="px-4 py-3 font-medium text-emerald-300">{row.optimized}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {results.comparison && (
                <div className="mt-5 grid gap-4 md:grid-cols-3">
                  <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
                    <p className="text-xs uppercase tracking-[0.2em] text-slate-400">Accuracy p-value</p>
                    <div className="mt-2 text-2xl font-semibold text-white">{formatNumber(results.comparison?.accuracy_p_value, 4)}</div>
                  </div>
                  <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
                    <p className="text-xs uppercase tracking-[0.2em] text-slate-400">Loss p-value</p>
                    <div className="mt-2 text-2xl font-semibold text-white">{formatNumber(results.comparison?.loss_p_value, 4)}</div>
                  </div>
                  <div className="rounded-2xl border border-emerald-400/20 bg-emerald-500/10 p-4">
                    <p className="text-xs uppercase tracking-[0.2em] text-emerald-200">Improvement status</p>
                    <div className="mt-2 text-lg font-semibold text-emerald-100">
                      {results.comparison?.improved ? 'Optimized model outperformed baseline' : 'Waiting for a stronger optimized result'}
                    </div>
                  </div>
                </div>
              )}

              {!results.optimized && status.running && status.active_job === 'optimization' && (
                <div className="mt-5 rounded-2xl border border-cyan-400/20 bg-cyan-400/10 p-4 text-sm text-cyan-100">
                  ABC optimization is still evaluating candidates. Results will appear automatically when the current search finishes.
                </div>
              )}
            </SectionCard>

            <div className="grid gap-6 xl:grid-cols-2">
              <SectionCard title="ABC Search Convergence" subtitle="Fitness, validation accuracy, and loss over ABC iterations.">
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={convergence.abc_search}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                      <XAxis dataKey="iteration" stroke="#94a3b8" />
                      <YAxis stroke="#94a3b8" />
                      <Tooltip />
                      <Legend />
                      <Line type="monotone" dataKey="best_fitness" stroke="#22c55e" strokeWidth={3} dot={false} />
                      <Line type="monotone" dataKey="best_val_accuracy" stroke="#06b6d4" strokeWidth={2} dot={false} />
                      <Line type="monotone" dataKey="best_val_loss" stroke="#a855f7" strokeWidth={2} dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </SectionCard>

              <SectionCard title="Baseline vs Optimized Accuracy" subtitle="Comparison of final averaged test performance.">
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={results.comparison_chart || []}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                      <XAxis dataKey="name" stroke="#94a3b8" />
                      <YAxis stroke="#94a3b8" />
                      <Tooltip />
                      <Bar dataKey="accuracy" radius={[12, 12, 0, 0]}>
                        {(results.comparison_chart || []).map((entry, index) => (
                          <Cell key={entry.name} fill={index === 1 ? '#22c55e' : '#94a3b8'} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </SectionCard>
            </div>

            <div className="grid gap-6 xl:grid-cols-2">
              <SectionCard title="Prediction Distribution" subtitle="Pie chart of predicted class counts from the current best model.">
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie data={pieData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={110} label>
                        {pieData.map((entry, index) => (
                          <Cell key={entry.name} fill={chartColors[index % chartColors.length]} />
                        ))}
                      </Pie>
                      <Tooltip />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
              </SectionCard>

              <SectionCard title="Multi-class ROC Curve" subtitle="One-vs-rest ROC traces using averaged probabilities.">
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={rocData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                      <XAxis dataKey="fpr" stroke="#94a3b8" />
                      <YAxis stroke="#94a3b8" />
                      <Tooltip />
                      <Legend />
                      {(dataset?.target_names || []).map((name, index) => (
                        <Line key={name} type="monotone" dataKey={name} stroke={chartColors[index % chartColors.length]} dot={false} strokeWidth={2.5} />
                      ))}
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </SectionCard>
            </div>

            <div className="grid gap-6 xl:grid-cols-[1.2fr_0.8fr]">
              <SectionCard title="Confusion Matrix" subtitle="Aggregated predictions from the current evaluation model.">
                <div className="grid gap-2">
                  {results.optimized?.confusion_matrix?.matrix?.map((row, rowIndex) => (
                    <div key={`row-${rowIndex}`} className="grid grid-cols-3 gap-2">
                      {row.map((cell, colIndex) => (
                        <div
                          key={`cell-${rowIndex}-${colIndex}`}
                          className={`rounded-2xl border px-4 py-4 text-center text-lg font-semibold ${
                            rowIndex === colIndex
                              ? 'border-emerald-400/20 bg-emerald-500/15 text-emerald-200'
                              : 'border-white/10 bg-white/5 text-slate-200'
                          }`}
                        >
                          {cell}
                        </div>
                      ))}
                    </div>
                  )) || <div className="text-slate-300">Run an experiment to populate the confusion matrix.</div>}
                </div>
              </SectionCard>

              <SectionCard title="Dataset + Methods" subtitle="Protocol summary for academic reporting.">
                <div className="space-y-4 text-sm leading-7 text-slate-300">
                  <p>Split: {dataset?.split || '70/15/15 stratified'}</p>
                  <p>Preprocessing: {(dataset?.preprocessing || ['StandardScaler']).join(', ')}</p>
                  <p>Baseline architecture: 13 → 64 → 3 with ReLU, Adam, CrossEntropyLoss.</p>
                  <p>ABC search space: hidden size, layers, learning rate, batch size, activation, dropout, and weight initialization.</p>
                  <p>Validation criterion: fitness = validation accuracy - λ × validation loss - overfitting penalty.</p>
                  <p>Statistical reporting: mean, standard deviation, 95% confidence interval, paired t-tests, confusion matrix, ROC, precision, recall, and F1-score.</p>
                </div>
              </SectionCard>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
