import { useState, useEffect, useRef, useMemo } from 'react';

/* ── Magma colormap (matches SpectrogramDisplay) ── */
const MAGMA: [number, number, number][] = [
  [0, 0, 4], [16, 7, 46], [44, 10, 82], [81, 18, 107],
  [120, 28, 109], [156, 44, 96], [189, 63, 72], [217, 89, 52],
  [240, 124, 32], [252, 165, 10], [252, 206, 37], [252, 247, 125],
];

function colorAt(t: number): [number, number, number] {
  const c = Math.max(0, Math.min(1, t));
  const pos = c * (MAGMA.length - 1);
  const lo = Math.floor(pos);
  const hi = Math.min(lo + 1, MAGMA.length - 1);
  const f = pos - lo;
  return [
    Math.round(MAGMA[lo][0] + (MAGMA[hi][0] - MAGMA[lo][0]) * f),
    Math.round(MAGMA[lo][1] + (MAGMA[hi][1] - MAGMA[lo][1]) * f),
    Math.round(MAGMA[lo][2] + (MAGMA[hi][2] - MAGMA[lo][2]) * f),
  ];
}

/* ── Network layer config ── */
const NN_LAYERS = [
  { size: 4096, nodes: 7 },
  { size: 128, nodes: 5 },
  { size: 64, nodes: 4 },
  { size: 10, nodes: 5 },
];

/* ── Props ── */
interface PipelineAnimationProps {
  waveform: number[];
  spectrogram: number[][];
  prediction: string;
  confidence: number;
  onComplete: () => void;
}

/*
 * Animation stages:
 *  0 = init
 *  1 = waveform visible
 *  2 = arrow 1
 *  3 = spectrogram visible
 *  4 = arrow 2
 *  5 = nn input layer lit
 *  6 = nn hidden-1 lit
 *  7 = nn hidden-2 lit
 *  8 = nn output lit
 *  9 = arrow 3
 * 10 = prediction visible
 * 11 = done → onComplete
 */
const SCHEDULE: [number, number][] = [
  [1, 150],
  [2, 950],
  [3, 1250],
  [4, 2250],
  [5, 2550],
  [6, 2850],
  [7, 3150],
  [8, 3450],
  [9, 3750],
  [10, 4050],
  [11, 4850],
];

export default function PipelineAnimation({
  waveform,
  spectrogram,
  prediction,
  confidence,
  onComplete,
}: PipelineAnimationProps) {
  const [stage, setStage] = useState(0);
  const timersRef = useRef<number[]>([]);
  const onCompleteRef = useRef(onComplete);
  onCompleteRef.current = onComplete;
  const specCanvasRef = useRef<HTMLCanvasElement>(null);
  const specDrawnRef = useRef(false);

  /* ── Schedule all stage transitions on mount ── */
  useEffect(() => {
    timersRef.current = SCHEDULE.map(([s, ms]) =>
      window.setTimeout(() => {
        setStage(s);
        if (s === 11) onCompleteRef.current();
      }, ms)
    );
    return () => timersRef.current.forEach(clearTimeout);
  }, []);

  /* ── Skip handler ── */
  const skip = () => {
    timersRef.current.forEach(clearTimeout);
    setStage(11);
    onCompleteRef.current();
  };

  /* ── Waveform SVG path ── */
  const waveformPath = useMemo(() => {
    if (!waveform.length) return '';
    const N = 60;
    const step = Math.max(1, Math.floor(waveform.length / N));
    const pts: number[] = [];
    for (let i = 0; i < waveform.length; i += step) pts.push(waveform[i]);
    const peak = Math.max(...pts.map(Math.abs), 0.01);
    return pts
      .map((v, i) => {
        const x = (i / (pts.length - 1)) * 120;
        const y = 30 - (v / peak) * 24;
        return `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(' ');
  }, [waveform]);

  /* ── Draw mini spectrogram once stage reaches 3 ── */
  useEffect(() => {
    if (stage < 3 || specDrawnRef.current) return;
    const canvas = specCanvasRef.current;
    if (!canvas || !spectrogram.length) return;
    const nF = spectrogram.length;
    const nT = spectrogram[0]?.length ?? 0;
    if (!nF || !nT) return;
    specDrawnRef.current = true;

    canvas.width = nT;
    canvas.height = nF;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let mn = Infinity, mx = -Infinity;
    for (const row of spectrogram)
      for (const v of row) {
        if (v < mn) mn = v;
        if (v > mx) mx = v;
      }
    const range = mx === mn ? 1 : mx - mn;

    const img = ctx.createImageData(nT, nF);
    for (let r = 0; r < nF; r++) {
      for (let c = 0; c < nT; c++) {
        const sr = nF - 1 - r;
        const [red, green, blue] = colorAt(
          (spectrogram[sr][c] - mn) / range
        );
        const idx = (r * nT + c) * 4;
        img.data[idx] = red;
        img.data[idx + 1] = green;
        img.data[idx + 2] = blue;
        img.data[idx + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
  }, [stage, spectrogram]);

  /* ── Derived: which NN layer is active (-1 to 3) ── */
  const activeLayer =
    stage >= 8 ? 3 : stage >= 7 ? 2 : stage >= 6 ? 1 : stage >= 5 ? 0 : -1;

  /* ── Shared arrow SVG ── */
  const Arrow = ({ visible }: { visible: boolean }) => (
    <div className="flex items-center flex-shrink-0 w-6 justify-center">
      <svg
        viewBox="0 0 24 24"
        className={`w-4 h-4 text-warm-400 dark:text-warm-600 transition-opacity duration-300 ${
          visible ? 'opacity-100' : 'opacity-0'
        }`}
      >
        <path
          d="M5 12h14M13 6l6 6-6 6"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.5"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
    </div>
  );

  return (
    <div className="card relative overflow-hidden p-6 space-y-5 animate-fade-in">
      {/* ── Progress bar ── */}
      <div className="absolute top-0 left-0 right-0 h-0.5 bg-warm-200 dark:bg-warm-700 rounded-t-xl overflow-hidden">
        <div
          className="h-full bg-accent/60 dark:bg-accent-light/40 transition-all ease-linear"
          style={{
            width: stage >= 11 ? '100%' : `${Math.round((stage / 11) * 100)}%`,
            transitionDuration: '300ms',
          }}
        />
      </div>

      {/* ── Header ── */}
      <div className="flex items-center justify-between">
        <div>
          <p className="text-[11px] font-semibold uppercase tracking-wider text-warm-500 dark:text-warm-500">
            Processing Pipeline
          </p>
          <p className="text-xs text-warm-400 dark:text-warm-600 mt-0.5">
            Watch how your audio flows through the neural network
          </p>
        </div>
        {stage < 11 && (
          <button
            onClick={skip}
            className="text-xs text-warm-500 hover:text-warm-600 dark:hover:text-warm-300 transition-colors"
          >
            Skip &rarr;
          </button>
        )}
      </div>

      {/* ── Pipeline stages (horizontal flow) ── */}
      <div className="flex items-stretch gap-3">
        {/* ▸ Stage 1 — Raw Audio */}
        <div
          className={`flex-1 min-w-0 rounded-xl border p-3 transition-all duration-700 ease-out ${
            stage >= 1
              ? 'opacity-100 translate-y-0 border-accent/20 dark:border-accent/20 bg-accent-subtle/50 dark:bg-accent/5'
              : 'opacity-0 translate-y-4 border-warm-300 dark:border-warm-700'
          }`}
        >
          <p className="text-[10px] font-semibold uppercase tracking-wider text-accent dark:text-accent-light mb-1.5">
            Raw Audio
          </p>
          <svg
            viewBox="0 0 120 60"
            className="w-full h-12"
            preserveAspectRatio="none"
          >
            <path
              d={waveformPath}
              fill="none"
              stroke="currentColor"
              strokeWidth="1.5"
              strokeLinecap="round"
              strokeLinejoin="round"
              className="text-accent dark:text-accent-light"
            />
          </svg>
          <p className="text-[9px] text-warm-400 dark:text-warm-600 mt-1 text-center">
            Amplitude vs Time
          </p>
        </div>

        <Arrow visible={stage >= 2} />

        {/* ▸ Stage 2 — Spectrogram */}
        <div
          className={`flex-1 min-w-0 rounded-xl border p-3 transition-all duration-700 ease-out ${
            stage >= 3
              ? 'opacity-100 translate-y-0 border-amber-200 dark:border-amber-500/20 bg-amber-50/50 dark:bg-amber-500/5'
              : 'opacity-0 translate-y-4 border-warm-300 dark:border-warm-700'
          }`}
        >
          <p className="text-[10px] font-semibold uppercase tracking-wider text-amber-600 dark:text-amber-400 mb-1.5">
            Spectrogram
          </p>
          <canvas
            ref={specCanvasRef}
            className="w-full h-12 rounded"
            style={{ imageRendering: 'pixelated' }}
          />
          <p className="text-[9px] text-warm-400 dark:text-warm-600 mt-1 text-center">
            64 × 64 mel grid
          </p>
        </div>

        <Arrow visible={stage >= 4} />

        {/* ▸ Stage 3 — Neural Network */}
        <div
          className={`flex-[1.4] min-w-0 rounded-xl border p-3 transition-all duration-700 ease-out ${
            stage >= 5
              ? 'opacity-100 translate-y-0 border-violet-200 dark:border-violet-500/20 bg-violet-50/50 dark:bg-violet-500/5'
              : 'opacity-0 translate-y-4 border-warm-300 dark:border-warm-700'
          }`}
        >
          <p className="text-[10px] font-semibold uppercase tracking-wider text-violet-600 dark:text-violet-400 mb-1.5">
            Neural Network
          </p>
          <NeuralNetDiagram activeLayer={activeLayer} />
          <p className="text-[9px] text-warm-400 dark:text-warm-600 mt-1 text-center">
            4096 → 128 → 64 → 10
          </p>
        </div>

        <Arrow visible={stage >= 9} />

        {/* ▸ Stage 4 — Prediction */}
        <div
          className={`flex-1 min-w-0 rounded-xl border p-3 transition-all duration-700 ease-out
            flex flex-col items-center justify-center ${
            stage >= 10
              ? 'opacity-100 translate-y-0 border-emerald-200 dark:border-emerald-500/20 bg-emerald-50/50 dark:bg-emerald-500/5'
              : 'opacity-0 translate-y-4 border-warm-300 dark:border-warm-700'
          }`}
        >
          <p className="text-[10px] font-semibold uppercase tracking-wider text-emerald-600 dark:text-emerald-400 mb-2">
            Prediction
          </p>
          <p className="text-base font-bold text-warm-900 dark:text-warm-100 capitalize leading-tight">
            {prediction}
          </p>
          <p className="text-sm font-mono text-emerald-600 dark:text-emerald-400 mt-0.5">
            {(confidence * 100).toFixed(1)}%
          </p>
        </div>
      </div>

      {/* ── Step indicators ── */}
      <div className="flex items-center justify-between px-1">
        {[
          { label: 'Load Audio', threshold: 1 },
          { label: 'Generate Spectrogram', threshold: 3 },
          { label: 'Run Inference', threshold: 5 },
          { label: 'Classify', threshold: 10 },
        ].map(({ label, threshold }) => (
          <div key={label} className="flex items-center gap-1.5">
            <div
              className={`w-1.5 h-1.5 rounded-full transition-colors duration-500 ${
                stage >= threshold
                  ? 'bg-accent dark:bg-accent-light'
                  : 'bg-warm-300 dark:bg-warm-700'
              }`}
            />
            <span
              className={`text-[10px] transition-colors duration-500 ${
                stage >= threshold
                  ? 'text-warm-600 dark:text-warm-400'
                  : 'text-warm-400 dark:text-warm-700'
              }`}
            >
              {label}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ══════════════════════════════════════════════════════════
   Neural Network Diagram (SVG)
   ══════════════════════════════════════════════════════════ */

function NeuralNetDiagram({ activeLayer }: { activeLayer: number }) {
  const W = 200;
  const H = 70;
  const xs = [20, 75, 130, 180];

  // Pre-compute node positions for each layer
  const layers = NN_LAYERS.map((layer, li) =>
    Array.from({ length: layer.nodes }, (_, ni) => ({
      x: xs[li],
      y: ((ni + 1) / (layer.nodes + 1)) * (H - 12) + 6,
    }))
  );

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-12">
      {/* Connections between layers */}
      {layers.map((layerNodes, li) => {
        if (li >= layers.length - 1) return null;
        const nextNodes = layers[li + 1];
        const lit = activeLayer >= li + 1;
        return layerNodes.flatMap((from, fi) =>
          nextNodes.map((to, ti) => (
            <line
              key={`c-${li}-${fi}-${ti}`}
              x1={from.x + 3}
              y1={from.y}
              x2={to.x - 3}
              y2={to.y}
              strokeWidth="0.3"
              className={`transition-all duration-500 ${
                lit
                  ? 'stroke-violet-300 dark:stroke-violet-600'
                  : 'stroke-warm-300 dark:stroke-warm-700'
              }`}
            />
          ))
        );
      })}

      {/* Nodes */}
      {layers.map((layerNodes, li) => {
        const lit = activeLayer >= li;
        const isOutput = li === layers.length - 1;
        return layerNodes.map((node, ni) => {
          const isWinner = isOutput && ni === 0 && activeLayer >= 3;
          return (
            <circle
              key={`n-${li}-${ni}`}
              cx={node.x}
              cy={node.y}
              r={2.5}
              className={`transition-all duration-500 ${
                isWinner
                  ? 'fill-emerald-500 dark:fill-emerald-400'
                  : lit
                    ? 'fill-violet-500 dark:fill-violet-400'
                    : 'fill-warm-300 dark:fill-warm-700'
              }`}
            />
          );
        });
      })}

      {/* "..." indicators between visible nodes and total count */}
      {NN_LAYERS.map((layer, li) => {
        if (layer.nodes >= layer.size) return null;
        const x = xs[li];
        return (
          <text
            key={`dots-${li}`}
            x={x}
            y={H - 14}
            textAnchor="middle"
            className="fill-warm-400 dark:fill-warm-700"
            style={{ fontSize: '6px' }}
          >
            ⋮
          </text>
        );
      })}

      {/* Layer size labels */}
      {NN_LAYERS.map((layer, li) => (
        <text
          key={`lbl-${li}`}
          x={xs[li]}
          y={H - 2}
          textAnchor="middle"
          className="fill-warm-500 dark:fill-warm-600"
          style={{ fontSize: '5px' }}
        >
          {layer.size}
        </text>
      ))}
    </svg>
  );
}
