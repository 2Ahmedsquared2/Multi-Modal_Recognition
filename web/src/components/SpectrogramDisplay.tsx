import { useRef, useEffect, useState, useCallback } from 'react';

interface SpectrogramDisplayProps {
  /** 2-D spectrogram array (rows = frequency bins, cols = time frames) */
  spectrogram: number[][];
  /** Audio duration in seconds */
  duration: number;
}

// Magma-inspired colormap: dark → purple → orange → yellow-white
const MAGMA: [number, number, number][] = [
  [0, 0, 4],
  [16, 7, 46],
  [44, 10, 82],
  [81, 18, 107],
  [120, 28, 109],
  [156, 44, 96],
  [189, 63, 72],
  [217, 89, 52],
  [240, 124, 32],
  [252, 165, 10],
  [252, 206, 37],
  [252, 247, 125],
];

function sampleColormap(t: number): [number, number, number] {
  const clamped = Math.max(0, Math.min(1, t));
  const pos = clamped * (MAGMA.length - 1);
  const lo = Math.floor(pos);
  const hi = Math.min(lo + 1, MAGMA.length - 1);
  const frac = pos - lo;
  return [
    Math.round(MAGMA[lo][0] + (MAGMA[hi][0] - MAGMA[lo][0]) * frac),
    Math.round(MAGMA[lo][1] + (MAGMA[hi][1] - MAGMA[lo][1]) * frac),
    Math.round(MAGMA[lo][2] + (MAGMA[hi][2] - MAGMA[lo][2]) * frac),
  ];
}

export default function SpectrogramDisplay({ spectrogram, duration }: SpectrogramDisplayProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [tooltip, setTooltip] = useState<{
    x: number; y: number; freq: string; time: string; energy: string;
  } | null>(null);

  // Precompute flat min/max for normalization
  const { minVal, maxVal, nFreq, nTime } = (() => {
    if (spectrogram.length === 0) return { minVal: 0, maxVal: 1, nFreq: 0, nTime: 0 };
    let mn = Infinity, mx = -Infinity;
    for (const row of spectrogram) {
      for (const v of row) {
        if (v < mn) mn = v;
        if (v > mx) mx = v;
      }
    }
    return { minVal: mn, maxVal: mx === mn ? mn + 1 : mx, nFreq: spectrogram.length, nTime: spectrogram[0]?.length ?? 0 };
  })();

  // Draw spectrogram
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || nFreq === 0) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.scale(dpr, dpr);

    const w = rect.width;
    const h = rect.height;
    const pad = { top: 4, bottom: 20, left: 28, right: 8 };
    const plotW = w - pad.left - pad.right;
    const plotH = h - pad.top - pad.bottom;

    ctx.clearRect(0, 0, w, h);

    // Create ImageData for the heatmap
    const imgCanvas = document.createElement('canvas');
    imgCanvas.width = nTime;
    imgCanvas.height = nFreq;
    const imgCtx = imgCanvas.getContext('2d')!;
    const imgData = imgCtx.createImageData(nTime, nFreq);
    const range = maxVal - minVal;

    for (let row = 0; row < nFreq; row++) {
      for (let col = 0; col < nTime; col++) {
        // Flip row so low freq is at bottom
        const srcRow = nFreq - 1 - row;
        const t = (spectrogram[srcRow][col] - minVal) / range;
        const [r, g, b] = sampleColormap(t);
        const idx = (row * nTime + col) * 4;
        imgData.data[idx] = r;
        imgData.data[idx + 1] = g;
        imgData.data[idx + 2] = b;
        imgData.data[idx + 3] = 255;
      }
    }
    imgCtx.putImageData(imgData, 0, 0);

    // Draw scaled heatmap onto main canvas
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(imgCanvas, pad.left, pad.top, plotW, plotH);

    // Axis labels
    ctx.fillStyle = 'rgba(148, 163, 184, 0.7)';
    ctx.font = '10px ui-monospace, monospace';

    // Time axis
    ctx.textAlign = 'left';
    ctx.fillText('0s', pad.left, h - 4);
    ctx.textAlign = 'center';
    ctx.fillText(`${(duration / 2).toFixed(1)}s`, pad.left + plotW / 2, h - 4);
    ctx.textAlign = 'right';
    ctx.fillText(`${duration.toFixed(1)}s`, w - pad.right, h - 4);

    // Frequency axis (left side)
    ctx.save();
    ctx.textAlign = 'right';
    ctx.fillText('Hi', pad.left - 4, pad.top + 10);
    ctx.fillText('Lo', pad.left - 4, pad.top + plotH);
    ctx.restore();
  }, [spectrogram, duration, nFreq, nTime, minVal, maxVal]);

  // Hover handler
  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas || nFreq === 0) return;

    const rect = canvas.getBoundingClientRect();
    const pad = { top: 4, bottom: 20, left: 28, right: 8 };
    const plotW = rect.width - pad.left - pad.right;
    const plotH = rect.height - pad.top - pad.bottom;
    const mx = e.clientX - rect.left - pad.left;
    const my = e.clientY - rect.top - pad.top;

    if (mx < 0 || mx > plotW || my < 0 || my > plotH) {
      setTooltip(null);
      return;
    }

    const col = Math.floor((mx / plotW) * nTime);
    const row = Math.floor((my / plotH) * nFreq);
    const srcRow = nFreq - 1 - row; // flipped
    const energy = spectrogram[srcRow]?.[col] ?? 0;
    const timeSec = (col / nTime) * duration;
    const freqBin = srcRow;

    setTooltip({
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
      freq: `Bin ${freqBin}/${nFreq}`,
      time: `${timeSec.toFixed(3)}s`,
      energy: energy.toFixed(4),
    });
  }, [spectrogram, duration, nFreq, nTime]);

  return (
    <div className="space-y-3 h-full flex flex-col">
      <div className="flex items-center justify-between">
        <p className="text-[11px] font-semibold uppercase tracking-wider text-warm-500 dark:text-warm-500">
          Mel-Spectrogram
        </p>
        <div className="flex items-center gap-1.5">
          <svg className="w-3 h-3 text-warm-400 dark:text-warm-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
          </svg>
          <p className="text-[10px] text-warm-400 dark:text-warm-600">
            Hover to explore
          </p>
        </div>
      </div>
      <div ref={containerRef} className="relative flex-1">
        <canvas
          ref={canvasRef}
          className="w-full h-40 rounded-lg cursor-crosshair border border-slate-200/50 dark:border-slate-700/50"
          onMouseMove={handleMouseMove}
          onMouseLeave={() => setTooltip(null)}
        />
        {tooltip && (
          <div
            className="absolute pointer-events-none z-10 px-2 py-1 rounded-md text-[10px] font-mono
              bg-warm-800/90 text-slate-200 dark:bg-slate-800/95 dark:text-slate-300
              whitespace-nowrap shadow-lg"
            style={{ left: tooltip.x + 12, top: tooltip.y - 36 }}
          >
            <span className="text-indigo-400">{tooltip.time}</span>
            {' · '}
            <span className="text-amber-400">{tooltip.freq}</span>
            {' · '}
            <span className="text-emerald-400">{tooltip.energy}</span>
          </div>
        )}
      </div>
      <p className="text-[10px] text-warm-400 dark:text-warm-600 text-center italic">
        What the model sees — your audio transformed into a visual pattern
      </p>
    </div>
  );
}
