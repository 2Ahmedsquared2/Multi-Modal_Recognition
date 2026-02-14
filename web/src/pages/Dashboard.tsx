import { useEffect, useState } from 'react';
import { api } from '../api/client';
import type {
  ConfusionMatrix,
  ClassMetricsResponse,
  ClassMetric,
  TrainingHistory,
  ModelInfo,
} from '../types';
import TrainingCurves from '../components/TrainingCurves';
import ConfusionMatrixChart from '../components/ConfusionMatrixChart';
import ClassMetricsChart from '../components/ClassMetricsChart';

/* ── Helper row type for class metrics ── */
interface ClassMetricRow extends ClassMetric {
  class_name: string;
}

export default function Dashboard() {
  const [confusion, setConfusion] = useState<ConfusionMatrix | null>(null);
  const [metricsResp, setMetricsResp] = useState<ClassMetricsResponse | null>(null);
  const [history, setHistory] = useState<TrainingHistory | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  useEffect(() => {
    Promise.allSettled([
      api.confusionMatrix(),
      api.classMetrics(),
      api.trainingHistory(),
      api.modelInfo(),
    ]).then(([cm, met, hist, info]) => {
      if (cm.status === 'fulfilled') setConfusion(cm.value);
      if (met.status === 'fulfilled') setMetricsResp(met.value);
      if (hist.status === 'fulfilled') setHistory(hist.value);
      if (info.status === 'fulfilled') setModelInfo(info.value);
      setLoading(false);
    }).catch(() => {
      setError(true);
      setLoading(false);
    });
  }, []);

  /* ── Transform per_class Record → sorted array ── */
  let metrics: ClassMetricRow[] | null = null;
  try {
    if (metricsResp && metricsResp.per_class) {
      metrics = Object.entries(metricsResp.per_class).map(([name, m]) => ({
        class_name: name,
        ...m,
      }));
    }
  } catch {
    metrics = null;
  }

  /* ── Compute accuracy from confusion matrix diagonal ── */
  let accuracy: number | null = null;
  try {
    if (confusion && confusion.matrix) {
      const total = confusion.matrix.flat().reduce((a, b) => a + b, 0);
      const correct = confusion.matrix.reduce((sum, row, i) => sum + (row[i] || 0), 0);
      accuracy = total > 0 ? correct / total : 0;
    } else if (metricsResp?.test_accuracy != null) {
      accuracy = metricsResp.test_accuracy;
    }
  } catch {
    accuracy = null;
  }

  return (
    <div className="max-w-6xl mx-auto px-8 py-10 space-y-8">

      {/* Header */}
      <header className="space-y-2">
        <h1 className="text-2xl font-bold tracking-tight text-slate-900 dark:text-white">
          Training Dashboard
        </h1>
        <p className="text-sm text-slate-600 dark:text-slate-400">
          Model performance metrics, interactive training curves, and the confusion matrix.
        </p>
      </header>

      {loading ? (
        <div className="flex items-center gap-3 py-20 justify-center">
          <div className="w-5 h-5 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin" />
          <span className="text-sm text-slate-500">Loading metrics...</span>
        </div>
      ) : error ? (
        <div className="card p-8 text-center">
          <p className="text-sm text-slate-500 dark:text-slate-500">
            Could not load metrics. Make sure the backend is running.
          </p>
        </div>
      ) : (
        <>
          {/* ── Summary Cards Row ── */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {/* Accuracy */}
            <div className="card p-5">
              <p className="text-[11px] font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-500">
                Overall Accuracy
              </p>
              <p className="mt-1 text-3xl font-bold text-slate-900 dark:text-white">
                {accuracy !== null ? `${(accuracy * 100).toFixed(1)}%` : '—'}
              </p>
              <p className="text-xs text-slate-500 dark:text-slate-500 mt-1">
                On test set {confusion ? `(${confusion.class_names.length} classes)` : ''}
              </p>
            </div>

            {/* Macro F1 */}
            <div className="card p-5">
              <p className="text-[11px] font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-500">
                Macro F1
              </p>
              <p className="mt-1 text-3xl font-bold text-slate-900 dark:text-white">
                {metricsResp?.macro_f1 != null ? (metricsResp.macro_f1 * 100).toFixed(1) + '%' : '—'}
              </p>
              <p className="text-xs text-slate-500 dark:text-slate-500 mt-1">
                Across all classes
              </p>
            </div>

            {/* Best Class */}
            <div className="card p-5">
              <p className="text-[11px] font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-500">
                Best Class
              </p>
              {metrics && metrics.length > 0 && (() => {
                const best = metrics.reduce((a, b) => a.f1_score > b.f1_score ? a : b);
                return (
                  <>
                    <p className="mt-1 text-lg font-semibold text-slate-900 dark:text-white">{best.class_name}</p>
                    <p className="text-xs text-emerald-600 dark:text-emerald-500 mt-0.5">F1: {(best.f1_score * 100).toFixed(1)}%</p>
                  </>
                );
              })()}
            </div>

            {/* Model Info */}
            <div className="card p-5">
              <p className="text-[11px] font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-500">
                Model
              </p>
              {modelInfo ? (
                <>
                  <p className="mt-1 text-lg font-semibold text-slate-900 dark:text-white">
                    {modelInfo.parameters.toLocaleString()} params
                  </p>
                  <p className="text-xs text-slate-500 dark:text-slate-500 mt-0.5">
                    {modelInfo.architecture.hidden_sizes.length} hidden layers
                    {' · '}
                    {history ? `${history.epochs} epochs` : ''}
                  </p>
                </>
              ) : (
                <p className="mt-1 text-sm text-slate-400">—</p>
              )}
            </div>
          </div>

          {/* ── Architecture Details (collapsible summary) ── */}
          {modelInfo && (
            <details className="card group">
              <summary className="px-5 py-4 cursor-pointer flex items-center justify-between
                text-[11px] font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-500
                hover:text-slate-700 dark:hover:text-slate-300 transition-colors">
                <span>Architecture Details</span>
                <svg className="w-4 h-4 transition-transform group-open:rotate-180" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
                  <path d="M6 9l6 6 6-6" />
                </svg>
              </summary>
              <div className="px-5 pb-5">
                <div className="flex flex-wrap gap-2">
                  {modelInfo.architecture.layers.map((layer, i) => (
                    <div
                      key={i}
                      className="flex items-center gap-2 px-3 py-2 rounded-lg bg-slate-50 dark:bg-slate-800/50 text-xs"
                    >
                      <span className="font-mono text-indigo-600 dark:text-indigo-400">
                        {layer.input_size}→{layer.output_size}
                      </span>
                      <span className="text-slate-400 dark:text-slate-600">|</span>
                      <span className="text-slate-600 dark:text-slate-400">{layer.activation}</span>
                      <span className="text-slate-400 dark:text-slate-600">|</span>
                      <span className="text-slate-500 dark:text-slate-500 font-mono">
                        {layer.parameters.toLocaleString()}
                      </span>
                    </div>
                  ))}
                </div>
                <p className="mt-3 text-[11px] text-slate-400 dark:text-slate-600">
                  Input: {modelInfo.spectrogram_shape.join('×')} spectrogram
                  {' · '}
                  {modelInfo.audio_duration}s @ {(modelInfo.sample_rate / 1000).toFixed(1)}kHz
                </p>
              </div>
            </details>
          )}

          {/* ── Training Curves ── */}
          {history ? (
            <TrainingCurves history={history} />
          ) : (
            <section className="grid grid-cols-2 gap-3">
              <div className="card p-5 space-y-3">
                <h3 className="text-[11px] font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-500">
                  Loss Over Epochs
                </h3>
                <div className="h-48 rounded-lg bg-slate-100 dark:bg-slate-800/50 flex items-center justify-center">
                  <p className="text-xs text-slate-500 dark:text-slate-500 font-mono">
                    No training history available
                  </p>
                </div>
              </div>
              <div className="card p-5 space-y-3">
                <h3 className="text-[11px] font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-500">
                  Accuracy Over Epochs
                </h3>
                <div className="h-48 rounded-lg bg-slate-100 dark:bg-slate-800/50 flex items-center justify-center">
                  <p className="text-xs text-slate-500 dark:text-slate-500 font-mono">
                    No training history available
                  </p>
                </div>
              </div>
            </section>
          )}

          {/* ── Confusion Matrix + Class Metrics (side-by-side on large screens) ── */}
          <div className="grid grid-cols-1 xl:grid-cols-2 gap-3">
            {/* Confusion Matrix */}
            {confusion ? (
              <ConfusionMatrixChart data={confusion} />
            ) : (
              <div className="card p-5">
                <p className="text-xs text-slate-500 dark:text-slate-500 font-mono text-center py-16">
                  Confusion matrix not available
                </p>
              </div>
            )}

            {/* Per-Class Metrics */}
            {metrics && metrics.length > 0 ? (
              <ClassMetricsChart metrics={metrics} macroF1={metricsResp?.macro_f1 ?? undefined} />
            ) : (
              <div className="card p-5">
                <p className="text-xs text-slate-500 dark:text-slate-500 font-mono text-center py-16">
                  Class metrics not available
                </p>
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}
