import Plot from 'react-plotly.js';
import type { ClassMetric } from '../types';
import { useDarkMode } from '../hooks/useDarkMode';
import { useState, useMemo } from 'react';

type SortMode = 'f1' | 'support' | 'alpha';

interface ClassMetricRow extends ClassMetric {
  class_name: string;
}

interface Props {
  metrics: ClassMetricRow[];
  macroF1?: number;
}

export default function ClassMetricsChart({ metrics, macroF1 }: Props) {
  const isDark = useDarkMode();
  const [sortBy, setSortBy] = useState<SortMode>('f1');

  const sorted = useMemo(() => {
    const copy = [...metrics];
    switch (sortBy) {
      case 'f1':
        return copy.sort((a, b) => b.f1_score - a.f1_score);
      case 'support':
        return copy.sort((a, b) => b.support - a.support);
      case 'alpha':
        return copy.sort((a, b) => a.class_name.localeCompare(b.class_name));
    }
  }, [metrics, sortBy]);

  const names = sorted.map((m) => m.class_name);

  const textColor = isDark ? '#94a3b8' : '#64748b';
  const gridColor = isDark ? 'rgba(148,163,184,0.1)' : 'rgba(148,163,184,0.2)';

  const traces: Plotly.Data[] = [
    {
      x: names,
      y: sorted.map((m) => m.precision),
      type: 'bar' as const,
      name: 'Precision',
      marker: { color: isDark ? '#818cf8' : '#6366f1' }, // indigo
      hovertemplate: '%{x}<br>Precision: %{y:.1%}<extra></extra>',
    },
    {
      x: names,
      y: sorted.map((m) => m.recall),
      type: 'bar' as const,
      name: 'Recall',
      marker: { color: isDark ? '#34d399' : '#10b981' }, // emerald
      hovertemplate: '%{x}<br>Recall: %{y:.1%}<extra></extra>',
    },
    {
      x: names,
      y: sorted.map((m) => m.f1_score),
      type: 'bar' as const,
      name: 'F1',
      marker: { color: isDark ? '#fbbf24' : '#f59e0b' }, // amber
      hovertemplate: '%{x}<br>F1: %{y:.1%}<extra></extra>',
    },
  ];

  /* ── Macro F1 reference line ── */
  const shapes: Partial<Plotly.Shape>[] = [];
  const annotations: Partial<Plotly.Annotations>[] = [];

  if (macroF1 != null) {
    shapes.push({
      type: 'line',
      x0: -0.5,
      x1: names.length - 0.5,
      y0: macroF1,
      y1: macroF1,
      line: { color: isDark ? 'rgba(248,113,113,0.6)' : 'rgba(239,68,68,0.5)', width: 1.5, dash: 'dash' },
    });
    annotations.push({
      x: names.length - 0.5,
      y: macroF1,
      xanchor: 'right' as const,
      text: `Macro F1: ${(macroF1 * 100).toFixed(1)}%`,
      showarrow: false,
      font: { size: 9, color: isDark ? '#f87171' : '#ef4444' },
      yshift: 10,
    });
  }

  const sortOptions: { key: SortMode; label: string }[] = [
    { key: 'f1', label: 'F1' },
    { key: 'support', label: 'Support' },
    { key: 'alpha', label: 'A–Z' },
  ];

  return (
    <div className="card p-5 space-y-4">
      {/* ── Header + Sort Toggle ── */}
      <div className="flex flex-col gap-2">
        <div className="flex items-center justify-between">
          <h3 className="text-[11px] font-semibold uppercase tracking-wider text-warm-500 dark:text-warm-500">
            Per-Class Metrics
          </h3>
          <div className="flex items-center gap-1.5 bg-warm-200 dark:bg-warm-700 rounded-lg p-0.5">
            {sortOptions.map((opt) => (
              <button
                key={opt.key}
                onClick={() => setSortBy(opt.key)}
                className={`px-2.5 py-1 rounded-md text-[10px] font-medium transition-colors
                  ${sortBy === opt.key
                    ? 'bg-white dark:bg-slate-700 text-warm-900 dark:text-warm-100 shadow-sm'
                    : 'text-warm-500 dark:text-warm-400 hover:text-slate-700 dark:hover:text-slate-300'
                  }`}
              >
                {opt.label}
              </button>
            ))}
          </div>
        </div>
        <p className="text-xs text-warm-500 dark:text-warm-400">
          Precision (correct positive predictions), Recall (found all positives), and F1 (balanced score) for each class.
        </p>
      </div>

      {/* ── Grouped Bar Chart ── */}
      <Plot
        data={traces}
        layout={{
          paper_bgcolor: 'rgba(0,0,0,0)',
          plot_bgcolor: 'rgba(0,0,0,0)',
          font: { family: 'Inter, system-ui, sans-serif', color: textColor, size: 11 },
          margin: { t: 12, r: 12, b: 72, l: 48 },
          barmode: 'group' as const,
          bargap: 0.25,
          bargroupgap: 0.08,
          xaxis: {
            tickangle: -45,
            tickfont: { size: 9 },
            gridcolor: gridColor,
          },
          yaxis: {
            title: { text: 'Score', font: { size: 10 } },
            tickformat: '.0%',
            range: [0, 1.08],
            gridcolor: gridColor,
            zeroline: false,
          },
          legend: {
            orientation: 'h' as const,
            y: -0.28,
            x: 0.5,
            xanchor: 'center' as const,
            font: { size: 10 },
            bgcolor: 'rgba(0,0,0,0)',
          },
          shapes,
          annotations,
          hovermode: 'x unified' as const,
        }}
        config={{
          responsive: true,
          displaylogo: false,
          modeBarButtonsToRemove: ['lasso2d', 'select2d', 'autoScale2d'],
        }}
        useResizeHandler
        className="w-full"
        style={{ height: 340 }}
      />
    </div>
  );
}
