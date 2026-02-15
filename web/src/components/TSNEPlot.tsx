import Plot from 'react-plotly.js';
import { useDarkMode } from '../hooks/useDarkMode';
import { useMemo, useCallback } from 'react';
import type { TSNEPoint } from '../types';

/* ── Dynamic color palette (works for any number of classes) ── */
const PALETTE = [
  '#38bdf8', // sky-400
  '#fb7185', // rose-400
  '#fbbf24', // amber-400
  '#fb923c', // orange-400
  '#a78bfa', // violet-400
  '#34d399', // emerald-400
  '#818cf8', // indigo-400
  '#2dd4bf', // teal-400
  '#ef4444', // red-500
  '#94a3b8', // slate-400
  '#f472b6', // pink-400
  '#60a5fa', // blue-400
  '#4ade80', // green-400
  '#c084fc', // purple-400
  '#facc15', // yellow-400
];

const FALLBACK_COLOR = '#94a3b8';

/** Build a color map from class names using the palette */
function buildColorMap(classNames: string[]): Record<string, string> {
  const map: Record<string, string> = {};
  classNames.forEach((cls, i) => {
    map[cls] = PALETTE[i % PALETTE.length];
  });
  return map;
}

/* ── Incorrect marker: diamond shape ── */
const CORRECT_SYMBOL = 'circle';
const INCORRECT_SYMBOL = 'diamond';

interface Props {
  points: TSNEPoint[];
  classNames: string[];
  visibleClasses: Set<string>;
  showOnlyErrors: boolean;
  selectedIdx: number | null;
  onSelect: (idx: number | null) => void;
}

export default function TSNEPlot({
  points,
  classNames,
  visibleClasses,
  showOnlyErrors,
  selectedIdx,
  onSelect,
}: Props) {
  const isDark = useDarkMode();
  const textColor = isDark ? '#94a3b8' : '#64748b';
  const gridColor = isDark ? 'rgba(148,163,184,0.08)' : 'rgba(100,116,139,0.1)';

  /* ── Build dynamic color map from classNames ── */
  const colorMap = useMemo(() => buildColorMap(classNames), [classNames]);

  /* ── Filter points based on class visibility + error filter ── */
  const filtered = useMemo(() => {
    return points
      .map((p, i) => ({ ...p, originalIdx: i }))
      .filter((p) => visibleClasses.has(p.class_name))
      .filter((p) => !showOnlyErrors || !p.correct);
  }, [points, visibleClasses, showOnlyErrors]);

  /* ── Build one Plotly trace per class ── */
  const traces = useMemo(() => {
    return classNames
      .filter((cls) => visibleClasses.has(cls))
      .map((cls) => {
        const classPoints = filtered.filter((p) => p.class_name === cls);
        const color = colorMap[cls] ?? FALLBACK_COLOR;

        return {
          x: classPoints.map((p) => p.x),
          y: classPoints.map((p) => p.y),
          mode: 'markers' as const,
          type: 'scatter' as const,
          name: cls,
          marker: {
            size: 9,
            color,
            opacity: 0.8,
            symbol: classPoints.map((p) => (p.correct ? CORRECT_SYMBOL : INCORRECT_SYMBOL)),
            line: {
              width: classPoints.map((p) => {
                if (p.originalIdx === selectedIdx) return 3;
                return p.correct ? 0.5 : 1.5;
              }),
              color: classPoints.map((p) => {
                if (p.originalIdx === selectedIdx) return isDark ? '#fff' : '#0f172a';
                return p.correct
                  ? (isDark ? 'rgba(255,255,255,0.2)' : 'rgba(0,0,0,0.1)')
                  : '#ef4444';
              }),
            },
          },
          customdata: classPoints.map((p) => ({
            idx: p.originalIdx,
            cls: p.class_name,
            pred: p.predicted_label,
            conf: p.confidence,
            correct: p.correct,
          })),
          hovertemplate:
            '<b>%{customdata.cls}</b><br>' +
            'Predicted: %{customdata.pred}<br>' +
            'Confidence: %{customdata.conf:.1%}<br>' +
            '<extra></extra>',
        };
      });
  }, [classNames, visibleClasses, filtered, selectedIdx, isDark, colorMap]);

  /* ── Click handler ── */
  const handleClick = useCallback(
    (event: Plotly.PlotMouseEvent) => {
      const pt = event.points[0];
      if (pt?.customdata) {
        const cd = pt.customdata as unknown as { idx: number };
        onSelect(cd.idx === selectedIdx ? null : cd.idx);
      }
    },
    [onSelect, selectedIdx],
  );

  return (
    <Plot
      /* eslint-disable-next-line @typescript-eslint/no-explicit-any */
      data={traces as any}
      layout={{
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { family: 'Inter, system-ui, sans-serif', color: textColor, size: 11 },
        margin: { t: 8, r: 8, b: 40, l: 40 },
        xaxis: {
          showticklabels: false,
          showgrid: true,
          gridcolor: gridColor,
          zeroline: false,
          title: { text: 't-SNE 1', font: { size: 10 }, standoff: 4 },
        },
        yaxis: {
          showticklabels: false,
          showgrid: true,
          gridcolor: gridColor,
          zeroline: false,
          title: { text: 't-SNE 2', font: { size: 10 }, standoff: 4 },
        },
        showlegend: false,
        hovermode: 'closest' as const,
        dragmode: 'pan' as const,
      }}
      config={{
        responsive: true,
        displaylogo: false,
        modeBarButtonsToRemove: ['lasso2d', 'select2d', 'autoScale2d'],
        scrollZoom: true,
      }}
      useResizeHandler
      className="w-full"
      style={{ height: '100%' }}
      onClick={handleClick}
    />
  );
}
