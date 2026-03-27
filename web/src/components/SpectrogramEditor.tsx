import { useRef, useEffect, useCallback } from 'react';

// ── Magma colormap (same as SpectrogramDisplay) ──────────────────────────

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

// ── Types ────────────────────────────────────────────────────────────────

export type BrushTool = 'paint' | 'erase';

interface SpectrogramEditorProps {
  /** 2-D spectrogram array (rows = frequency bins, cols = time frames). */
  spectrogram: number[][];
  /** Called with the full modified spectrogram after each edit stroke (on mouseUp). */
  onChange?: (spectrogram: number[][]) => void;
  /** If true, canvas is non-interactive (used for the "original" view). */
  readOnly?: boolean;
  /** Current brush tool. */
  tool?: BrushTool;
  /** Brush radius in grid cells (1 = small, 3 = medium, 6 = large). */
  brushSize?: number;
  /** Label shown above the canvas. */
  label?: string;
}

// ── Component ────────────────────────────────────────────────────────────

export default function SpectrogramEditor({
  spectrogram,
  onChange,
  readOnly = false,
  tool = 'paint',
  brushSize = 1,
  label,
}: SpectrogramEditorProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const dataRef = useRef<number[][]>([]);
  const paintingRef = useRef(false);
  const hoveredRef = useRef<{ x: number; y: number } | null>(null);

  // Keep latest props in refs so stable callbacks see current values
  const toolRef = useRef(tool);
  toolRef.current = tool;
  const brushRef = useRef(brushSize);
  brushRef.current = brushSize;
  const readOnlyRef = useRef(readOnly);
  readOnlyRef.current = readOnly;
  const onChangeRef = useRef(onChange);
  onChangeRef.current = onChange;

  // ── Render the spectrogram onto the canvas (imperative, stable ref) ──
  const render = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const data = dataRef.current;
    if (data.length === 0) return;

    const nRows = data.length;
    const nCols = data[0]?.length ?? 0;
    if (nCols === 0) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) return;

    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.scale(dpr, dpr);

    const w = rect.width;
    const h = rect.height;

    // Find min/max for normalization
    let mn = Infinity;
    let mx = -Infinity;
    for (const row of data) {
      for (const v of row) {
        if (v < mn) mn = v;
        if (v > mx) mx = v;
      }
    }
    if (mx === mn) mx = mn + 1;
    const range = mx - mn;

    // Build ImageData on an offscreen canvas (64×64 → fast)
    const imgCanvas = document.createElement('canvas');
    imgCanvas.width = nCols;
    imgCanvas.height = nRows;
    const imgCtx = imgCanvas.getContext('2d')!;
    const imgData = imgCtx.createImageData(nCols, nRows);

    for (let row = 0; row < nRows; row++) {
      for (let col = 0; col < nCols; col++) {
        const srcRow = nRows - 1 - row; // flip so low freq is at bottom
        const t = (data[srcRow][col] - mn) / range;
        const [r, g, b] = sampleColormap(t);
        const idx = (row * nCols + col) * 4;
        imgData.data[idx] = r;
        imgData.data[idx + 1] = g;
        imgData.data[idx + 2] = b;
        imgData.data[idx + 3] = 255;
      }
    }
    imgCtx.putImageData(imgData, 0, 0);

    // Draw scaled heatmap (nearest-neighbor for crisp pixels)
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(imgCanvas, 0, 0, w, h);

    // Draw brush cursor on hover
    const hovered = hoveredRef.current;
    if (hovered && !readOnlyRef.current) {
      const cellW = w / nCols;
      const cellH = h / nRows;
      ctx.strokeStyle =
        toolRef.current === 'paint'
          ? 'rgba(252, 211, 77, 0.8)'
          : 'rgba(248, 113, 113, 0.8)';
      ctx.lineWidth = 1.5;
      const cx = (hovered.x + 0.5) * cellW;
      const cy = (nRows - 1 - hovered.y + 0.5) * cellH;
      const radius = (brushRef.current + 0.5) * Math.min(cellW, cellH);
      ctx.beginPath();
      ctx.arc(cx, cy, radius, 0, Math.PI * 2);
      ctx.stroke();
    }
  }, []);

  // ── Sync spectrogram prop → internal data & re-render ──
  useEffect(() => {
    dataRef.current = spectrogram.map((row) => [...row]);
    render();
  }, [spectrogram, render]);

  // Re-render when visual-only props change (cursor style)
  useEffect(() => {
    render();
  }, [tool, brushSize, render]);

  // ── Convert mouse event → grid coordinates ──
  const getGridCoords = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current;
      if (!canvas) return null;
      const data = dataRef.current;
      const nRows = data.length;
      const nCols = data[0]?.length ?? 0;
      if (nRows === 0 || nCols === 0) return null;

      const rect = canvas.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      const col = Math.floor((mx / rect.width) * nCols);
      const row = nRows - 1 - Math.floor((my / rect.height) * nRows);

      if (col < 0 || col >= nCols || row < 0 || row >= nRows) return null;
      return { x: col, y: row };
    },
    [],
  );

  // ── Apply brush at a grid coordinate ──
  const applyBrush = useCallback(
    (gx: number, gy: number) => {
      const data = dataRef.current;
      const nRows = data.length;
      const nCols = data[0]?.length ?? 0;
      const value = toolRef.current === 'paint' ? 1.0 : 0.0;
      const bs = brushRef.current;

      for (let dy = -bs; dy <= bs; dy++) {
        for (let dx = -bs; dx <= bs; dx++) {
          if (dx * dx + dy * dy <= bs * bs) {
            const nx = gx + dx;
            const ny = gy + dy;
            if (nx >= 0 && nx < nCols && ny >= 0 && ny < nRows) {
              data[ny][nx] = value;
            }
          }
        }
      }
      render();
    },
    [render],
  );

  // ── Mouse handlers ──

  const onMouseDown = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      if (readOnlyRef.current) return;
      e.preventDefault();
      paintingRef.current = true;
      const coords = getGridCoords(e);
      if (coords) applyBrush(coords.x, coords.y);
    },
    [getGridCoords, applyBrush],
  );

  const onMouseMove = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const coords = getGridCoords(e);
      hoveredRef.current = coords;
      if (paintingRef.current && !readOnlyRef.current && coords) {
        applyBrush(coords.x, coords.y);
      } else {
        render(); // redraw for brush cursor
      }
    },
    [getGridCoords, applyBrush, render],
  );

  const onMouseUp = useCallback(() => {
    if (paintingRef.current) {
      paintingRef.current = false;
      // Send a deep copy to parent
      onChangeRef.current?.(dataRef.current.map((row) => [...row]));
    }
  }, []);

  const onMouseLeave = useCallback(() => {
    hoveredRef.current = null;
    render();
    if (paintingRef.current) {
      paintingRef.current = false;
      onChangeRef.current?.(dataRef.current.map((row) => [...row]));
    }
  }, [render]);

  return (
    <div className="space-y-1.5">
      {label && (
        <h3 className="text-[11px] font-semibold uppercase tracking-wider text-warm-500 dark:text-warm-500">
          {label}
        </h3>
      )}
      <canvas
        ref={canvasRef}
        className={`w-full aspect-square rounded-lg bg-slate-950 ${
          readOnly ? 'cursor-default' : 'cursor-crosshair'
        }`}
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
        onMouseLeave={onMouseLeave}
      />
    </div>
  );
}
