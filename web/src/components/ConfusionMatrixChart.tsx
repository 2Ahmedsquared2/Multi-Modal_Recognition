import Plot from 'react-plotly.js';
import type { ConfusionMatrix } from '../types';
import { useDarkMode } from '../hooks/useDarkMode';
import { useState, useMemo, useCallback } from 'react';

interface Props {
  data: ConfusionMatrix;
}

export default function ConfusionMatrixChart({ data }: Props) {
  const isDark = useDarkMode();
  const [normalized, setNormalized] = useState(false);
  const [selected, setSelected] = useState<{ row: number; col: number } | null>(null);

  /* ── Compute normalized matrix (recall %) ── */
  const normalizedMatrix = useMemo(() => {
    return data.matrix.map((row) => {
      const total = row.reduce((a, b) => a + b, 0);
      return row.map((v) => (total > 0 ? v / total : 0));
    });
  }, [data.matrix]);

  const displayMatrix = normalized ? normalizedMatrix : data.matrix;

  /* ── Hover text ── */
  const hoverText = useMemo(() => {
    return data.matrix.map((row, i) => {
      const rowTotal = row.reduce((a, b) => a + b, 0);
      return row.map((val, j) => {
        const pct = rowTotal > 0 ? ((val / rowTotal) * 100).toFixed(1) : '0.0';
        return `True: ${data.class_names[i]}<br>Pred: ${data.class_names[j]}<br>Count: ${val} (${pct}%)`;
      });
    });
  }, [data]);

  /* ── Display text on cells ── */
  const annotationText = useMemo(() => {
    if (normalized) {
      return normalizedMatrix.map((row) =>
        row.map((v) => `${(v * 100).toFixed(0)}%`),
      );
    }
    return data.matrix.map((row) => row.map((v) => String(v)));
  }, [normalized, normalizedMatrix, data.matrix]);

  /* ── Colors ── */
  const textColor = isDark ? '#94a3b8' : '#64748b';

  const colorscale: Array<[number, string]> = isDark
    ? [
        [0, 'rgba(30,41,59,0.6)'],      // slate-800
        [0.25, 'rgba(99,102,241,0.2)'],  // indigo tint
        [0.5, 'rgba(99,102,241,0.45)'],
        [0.75, 'rgba(99,102,241,0.7)'],
        [1, 'rgba(99,102,241,0.95)'],     // indigo-500
      ]
    : [
        [0, 'rgba(241,245,249,1)'],       // slate-100
        [0.25, 'rgba(199,210,254,1)'],    // indigo-200
        [0.5, 'rgba(165,180,252,1)'],     // indigo-300
        [0.75, 'rgba(129,140,248,1)'],    // indigo-400
        [1, 'rgba(99,102,241,1)'],        // indigo-500
      ];

  /* ── Click handler ── */
  const handleClick = useCallback((event: Plotly.PlotMouseEvent) => {
    const pt = event.points[0];
    if (pt) {
      const row = pt.y as number;
      const col = pt.x as number;
      const rowIdx = data.class_names.indexOf(row as unknown as string);
      const colIdx = data.class_names.indexOf(col as unknown as string);
      if (rowIdx >= 0 && colIdx >= 0) {
        setSelected((prev) =>
          prev?.row === rowIdx && prev?.col === colIdx ? null : { row: rowIdx, col: colIdx },
        );
      }
    }
  }, [data.class_names]);

  return (
    <div className="card p-5 space-y-4">
      {/* ── Header + Toggle ── */}
      <div className="flex flex-col gap-2">
        <div className="flex items-center justify-between">
          <h3 className="text-[11px] font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-500">
            Confusion Matrix
          </h3>
          <div className="flex items-center gap-1.5 bg-slate-100 dark:bg-slate-800 rounded-lg p-0.5">
            <button
              onClick={() => setNormalized(false)}
              className={`px-2.5 py-1 rounded-md text-[10px] font-medium transition-colors
                ${!normalized
                  ? 'bg-white dark:bg-slate-700 text-slate-900 dark:text-white shadow-sm'
                  : 'text-slate-500 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-300'
                }`}
            >
              Counts
            </button>
            <button
              onClick={() => setNormalized(true)}
              className={`px-2.5 py-1 rounded-md text-[10px] font-medium transition-colors
                ${normalized
                  ? 'bg-white dark:bg-slate-700 text-slate-900 dark:text-white shadow-sm'
                  : 'text-slate-500 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-300'
                }`}
            >
              Recall %
            </button>
          </div>
        </div>
        <p className="text-xs text-slate-500 dark:text-slate-400">
          Visualizes correct predictions (diagonal) vs misclassifications. Darker colors indicate higher values.
        </p>
      </div>

      {/* ── Heatmap ── */}
      <Plot
        data={[
          {
            z: displayMatrix,
            x: data.class_names,
            y: data.class_names,
            type: 'heatmap' as const,
            colorscale,
            showscale: false,
            hovertext: hoverText as unknown as string[],
            hoverinfo: 'text' as const,
            text: annotationText as unknown as string[],
            texttemplate: '%{text}',
            textfont: { size: 10, color: textColor },
          },
        ]}
        layout={{
          paper_bgcolor: 'rgba(0,0,0,0)',
          plot_bgcolor: 'rgba(0,0,0,0)',
          font: { family: 'Inter, system-ui, sans-serif', color: textColor, size: 10 },
          margin: { t: 8, r: 8, b: 80, l: 80 },
          xaxis: {
            title: { text: 'Predicted', font: { size: 10 }, standoff: 12 },
            side: 'bottom' as const,
            tickangle: -45,
            tickfont: { size: 9 },
          },
          yaxis: {
            title: { text: 'Actual', font: { size: 10 }, standoff: 12 },
            autorange: 'reversed' as const,
            tickfont: { size: 9 },
          },
        }}
        config={{
          responsive: true,
          displaylogo: false,
          modeBarButtonsToRemove: ['lasso2d', 'select2d', 'autoScale2d', 'zoom2d', 'pan2d'],
        }}
        useResizeHandler
        className="w-full"
        style={{ height: 380 }}
        onClick={handleClick}
      />

      {/* ── Click Drill-Down Panel ── */}
      {selected && (
        <div className="rounded-lg bg-slate-50 dark:bg-slate-800/50 p-4 space-y-2 animate-fade-in">
          <div className="flex items-center justify-between">
            <p className="text-xs text-slate-700 dark:text-slate-300">
              <span className="font-medium">True:</span>{' '}
              <span className="text-indigo-600 dark:text-indigo-400">{data.class_names[selected.row]}</span>
              {' → '}
              <span className="font-medium">Predicted:</span>{' '}
              <span className="text-indigo-600 dark:text-indigo-400">{data.class_names[selected.col]}</span>
            </p>
            <button
              onClick={() => setSelected(null)}
              className="text-slate-400 hover:text-slate-600 dark:hover:text-slate-300 transition-colors"
            >
              <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
                <path d="M18 6L6 18M6 6l12 12" />
              </svg>
            </button>
          </div>
          <div className="flex items-center gap-4">
            <div className="text-2xl font-bold text-slate-900 dark:text-white">
              {data.matrix[selected.row][selected.col]}
            </div>
            <div className="text-xs text-slate-500 dark:text-slate-400 space-y-0.5">
              <p>
                samples {selected.row === selected.col ? 'correctly classified' : 'misclassified'}
              </p>
              <p>
                {(() => {
                  const rowTotal = data.matrix[selected.row].reduce((a, b) => a + b, 0);
                  const pct = rowTotal > 0 ? (data.matrix[selected.row][selected.col] / rowTotal * 100).toFixed(1) : '0.0';
                  return `${pct}% of all ${data.class_names[selected.row]} samples`;
                })()}
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
