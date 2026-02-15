import { useRef, useEffect } from 'react';

interface ImageDisplayProps {
  /** Object URL of the original uploaded image */
  originalSrc: string | null;
  /** 2-D preprocessed matrix the model receives (64×64 grayscale or flattened RGB) */
  preprocessed: number[][];
}

/**
 * Viridis-inspired colormap for the preprocessed heatmap.
 * Maps 0→dark-purple, 0.5→teal, 1→yellow.
 */
const VIRIDIS: [number, number, number][] = [
  [68, 1, 84],
  [72, 33, 115],
  [67, 62, 133],
  [56, 89, 140],
  [45, 112, 142],
  [37, 133, 142],
  [30, 155, 138],
  [42, 176, 127],
  [82, 197, 105],
  [134, 213, 73],
  [194, 223, 35],
  [253, 231, 37],
];

function sampleColormap(t: number): [number, number, number] {
  const clamped = Math.max(0, Math.min(1, t));
  const pos = clamped * (VIRIDIS.length - 1);
  const lo = Math.floor(pos);
  const hi = Math.min(lo + 1, VIRIDIS.length - 1);
  const frac = pos - lo;
  return [
    Math.round(VIRIDIS[lo][0] + (VIRIDIS[hi][0] - VIRIDIS[lo][0]) * frac),
    Math.round(VIRIDIS[lo][1] + (VIRIDIS[hi][1] - VIRIDIS[lo][1]) * frac),
    Math.round(VIRIDIS[lo][2] + (VIRIDIS[hi][2] - VIRIDIS[lo][2]) * frac),
  ];
}

export default function ImageDisplay({ originalSrc, preprocessed }: ImageDisplayProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const nRows = preprocessed.length;
  const nCols = preprocessed[0]?.length ?? 0;

  // Draw preprocessed heatmap onto the canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || nRows === 0 || nCols === 0) return;

    const dpr = window.devicePixelRatio || 1;
    const displaySize = 256; // render at a fixed logical size
    canvas.width = displaySize * dpr;
    canvas.height = displaySize * dpr;
    canvas.style.width = `${displaySize}px`;
    canvas.style.height = `${displaySize}px`;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.scale(dpr, dpr);

    // Compute min/max for normalization
    let mn = Infinity;
    let mx = -Infinity;
    for (const row of preprocessed) {
      for (const v of row) {
        if (v < mn) mn = v;
        if (v > mx) mx = v;
      }
    }
    const range = mx === mn ? 1 : mx - mn;

    // Build an off-screen image at native resolution, then scale up
    const imgCanvas = document.createElement('canvas');
    imgCanvas.width = nCols;
    imgCanvas.height = nRows;
    const imgCtx = imgCanvas.getContext('2d')!;
    const imgData = imgCtx.createImageData(nCols, nRows);

    for (let r = 0; r < nRows; r++) {
      for (let c = 0; c < nCols; c++) {
        const t = (preprocessed[r][c] - mn) / range;
        const [red, green, blue] = sampleColormap(t);
        const idx = (r * nCols + c) * 4;
        imgData.data[idx] = red;
        imgData.data[idx + 1] = green;
        imgData.data[idx + 2] = blue;
        imgData.data[idx + 3] = 255;
      }
    }

    imgCtx.putImageData(imgData, 0, 0);

    // Scale up with nearest-neighbor for that crisp pixel look
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(imgCanvas, 0, 0, displaySize, displaySize);
  }, [preprocessed, nRows, nCols]);

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      {/* Original image */}
      <div className="card p-5 flex flex-col">
        <div className="space-y-3 h-full flex flex-col">
          <p className="text-[11px] font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-500">
            Original Image
          </p>
          <div className="flex-1 flex items-center justify-center">
            {originalSrc ? (
              <img
                src={originalSrc}
                alt="Uploaded image"
                className="max-h-64 max-w-full rounded-lg object-contain ring-1 ring-slate-200/50 dark:ring-slate-700/50"
              />
            ) : (
              <div className="w-full h-40 bg-slate-100 dark:bg-slate-800 rounded-lg flex items-center justify-center">
                <span className="text-xs text-slate-400 dark:text-slate-600">No image</span>
              </div>
            )}
          </div>
          <p className="text-[10px] text-slate-400 dark:text-slate-600 text-center italic">
            The image you uploaded
          </p>
        </div>
      </div>

      {/* Preprocessed heatmap */}
      <div className="card p-5 flex flex-col">
        <div className="space-y-3 h-full flex flex-col">
          <div className="flex items-center justify-between">
            <p className="text-[11px] font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-500">
              Preprocessed ({nRows}×{nCols})
            </p>
            <p className="text-[10px] text-slate-400 dark:text-slate-600">
              Viridis colormap
            </p>
          </div>
          <div className="flex-1 flex items-center justify-center">
            <canvas
              ref={canvasRef}
              className="rounded-lg border border-slate-200/50 dark:border-slate-700/50"
            />
          </div>
          <p className="text-[10px] text-slate-400 dark:text-slate-600 text-center italic">
            What the model sees — your image resized and normalized
          </p>
        </div>
      </div>
    </div>
  );
}
