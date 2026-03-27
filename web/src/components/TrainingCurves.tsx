import Plot from 'react-plotly.js';
import type { TrainingHistory } from '../types';
import { useDarkMode } from '../hooks/useDarkMode';
import { useMemo } from 'react';

interface Props {
  history: TrainingHistory;
}

export default function TrainingCurves({ history }: Props) {
  const isDark = useDarkMode();

  const epochs = useMemo(
    () => Array.from({ length: history.epochs }, (_, i) => i + 1),
    [history.epochs],
  );

  /* ── Detect early-stopping epoch (lowest val loss) ── */
  const earlyStopEpoch = useMemo(() => {
    let minIdx = 0;
    for (let i = 1; i < history.val_loss.length; i++) {
      if (history.val_loss[i] < history.val_loss[minIdx]) minIdx = i;
    }
    return minIdx + 1; // 1-indexed
  }, [history.val_loss]);

  /* ── Colors ── */
  const colors = {
    trainLoss: isDark ? '#818cf8' : '#6366f1',   // indigo
    valLoss: isDark ? '#f472b6' : '#ec4899',       // pink
    trainAcc: isDark ? '#34d399' : '#10b981',       // emerald
    valAcc: isDark ? '#fbbf24' : '#f59e0b',         // amber
    grid: isDark ? 'rgba(148,163,184,0.1)' : 'rgba(148,163,184,0.2)',
    text: isDark ? '#94a3b8' : '#64748b',
    bg: 'rgba(0,0,0,0)',
    paper: 'rgba(0,0,0,0)',
    annotationLine: isDark ? 'rgba(248,113,113,0.5)' : 'rgba(239,68,68,0.4)',
  };

  /* ── Calculate smart tick interval ── */
  const tickInterval = useMemo(() => {
    const numEpochs = history.epochs;
    if (numEpochs <= 20) return 2;
    if (numEpochs <= 50) return 5;
    if (numEpochs <= 100) return 10;
    return 20;
  }, [history.epochs]);

  /* ── Shared layout props ── */
  const baseLayout: Partial<Plotly.Layout> = {
    paper_bgcolor: colors.paper,
    plot_bgcolor: colors.bg,
    font: { family: 'Inter, system-ui, sans-serif', color: colors.text, size: 11 },
    margin: { t: 24, r: 16, b: 50, l: 48 },
    hovermode: 'x unified' as const,
    legend: {
      orientation: 'h' as const,
      y: -0.25,
      x: 0.5,
      xanchor: 'center' as const,
      font: { size: 9 },
      bgcolor: 'rgba(0,0,0,0)',
    },
    xaxis: {
      title: { text: 'Epoch', font: { size: 10 } },
      gridcolor: colors.grid,
      zeroline: false,
      tickmode: 'linear' as const,
      dtick: tickInterval,
      tickfont: { size: 10 },
    },
  };

  const config: Partial<Plotly.Config> = {
    responsive: true,
    displaylogo: false,
    modeBarButtonsToRemove: ['lasso2d', 'select2d', 'autoScale2d'],
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
      {/* ── Loss Chart ── */}
      <div className="card p-5 space-y-3">
        <div>
          <h3 className="text-[11px] font-semibold uppercase tracking-wider text-warm-500 dark:text-warm-500">
            Loss Over Epochs
          </h3>
          <p className="text-xs text-warm-500 dark:text-warm-400 mt-1">
            Tracks how well the model learns over time. Lower loss means better fit to the data.
          </p>
        </div>
        <Plot
          data={[
            {
              x: epochs,
              y: history.train_loss,
              type: 'scatter' as const,
              mode: 'lines' as const,
              name: 'Train Loss',
              line: { color: colors.trainLoss, width: 2 },
              hovertemplate: '%{y:.4f}<extra>Train Loss</extra>',
            },
            {
              x: epochs,
              y: history.val_loss,
              type: 'scatter' as const,
              mode: 'lines' as const,
              name: 'Val Loss',
              line: { color: colors.valLoss, width: 2 },
              hovertemplate: '%{y:.4f}<extra>Val Loss</extra>',
            },
          ]}
          layout={{
            ...baseLayout,
            yaxis: {
              title: { text: 'Loss', font: { size: 10 } },
              gridcolor: colors.grid,
              zeroline: false,
            },
            shapes: [
              {
                type: 'line',
                x0: earlyStopEpoch,
                x1: earlyStopEpoch,
                y0: 0,
                y1: 1,
                yref: 'paper',
                line: { color: colors.annotationLine, width: 1.5, dash: 'dash' },
              },
            ],
            annotations: [
              {
                x: earlyStopEpoch,
                y: 1,
                yref: 'paper',
                text: `Best (ep ${earlyStopEpoch})`,
                showarrow: false,
                font: { size: 9, color: isDark ? '#f87171' : '#ef4444' },
                yanchor: 'bottom',
              },
            ],
          }}
          config={config}
          useResizeHandler
          className="w-full"
          style={{ height: 240 }}
        />
      </div>

      {/* ── Accuracy Chart ── */}
      <div className="card p-5 space-y-3">
        <div>
          <h3 className="text-[11px] font-semibold uppercase tracking-wider text-warm-500 dark:text-warm-500">
            Accuracy Over Epochs
          </h3>
          <p className="text-xs text-warm-500 dark:text-warm-400 mt-1">
            Shows prediction accuracy on training and validation sets. Higher is better.
          </p>
        </div>
        <Plot
          data={[
            {
              x: epochs,
              y: history.train_accuracy,
              type: 'scatter' as const,
              mode: 'lines' as const,
              name: 'Train Acc',
              line: { color: colors.trainAcc, width: 2 },
              hovertemplate: '%{y:.2%}<extra>Train Accuracy</extra>',
            },
            {
              x: epochs,
              y: history.val_accuracy,
              type: 'scatter' as const,
              mode: 'lines' as const,
              name: 'Val Acc',
              line: { color: colors.valAcc, width: 2 },
              hovertemplate: '%{y:.2%}<extra>Val Accuracy</extra>',
            },
          ]}
          layout={{
            ...baseLayout,
            yaxis: {
              title: { text: 'Accuracy', font: { size: 10 } },
              gridcolor: colors.grid,
              zeroline: false,
              tickformat: '.0%',
              range: [0, 1.05],
            },
            shapes: [
              {
                type: 'line',
                x0: earlyStopEpoch,
                x1: earlyStopEpoch,
                y0: 0,
                y1: 1,
                yref: 'paper',
                line: { color: colors.annotationLine, width: 1.5, dash: 'dash' },
              },
            ],
            annotations: [
              {
                x: earlyStopEpoch,
                y: 1,
                yref: 'paper',
                text: `Best (ep ${earlyStopEpoch})`,
                showarrow: false,
                font: { size: 9, color: isDark ? '#f87171' : '#ef4444' },
                yanchor: 'bottom',
              },
            ],
          }}
          config={config}
          useResizeHandler
          className="w-full"
          style={{ height: 240 }}
        />
      </div>
    </div>
  );
}
