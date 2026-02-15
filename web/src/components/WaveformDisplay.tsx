import { useRef, useEffect } from 'react';

interface WaveformDisplayProps {
  /** Downsampled waveform samples (min/max envelope, ~500 points) */
  waveform: number[];
  /** Total audio duration in seconds */
  duration: number;
}

export default function WaveformDisplay({ waveform, duration }: WaveformDisplayProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || waveform.length === 0) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.scale(dpr, dpr);

    const w = rect.width;
    const h = rect.height;
    const pad = { top: 8, bottom: 20, left: 4, right: 4 };
    const plotW = w - pad.left - pad.right;
    const plotH = h - pad.top - pad.bottom;
    const midY = pad.top + plotH / 2;

    // Clear
    ctx.clearRect(0, 0, w, h);

    // Find peak for scaling
    const peak = Math.max(...waveform.map(Math.abs), 0.01);

    // Draw filled waveform
    const n = waveform.length;
    const step = plotW / (n - 1);

    // Upper fill (positive half)
    ctx.beginPath();
    ctx.moveTo(pad.left, midY);
    for (let i = 0; i < n; i++) {
      const x = pad.left + i * step;
      const y = midY - (waveform[i] / peak) * (plotH / 2);
      ctx.lineTo(x, y);
    }
    ctx.lineTo(pad.left + (n - 1) * step, midY);
    ctx.closePath();

    // Gradient fill
    const grad = ctx.createLinearGradient(0, pad.top, 0, midY);
    grad.addColorStop(0, 'rgba(99, 102, 241, 0.35)');  // indigo-500
    grad.addColorStop(1, 'rgba(99, 102, 241, 0.05)');
    ctx.fillStyle = grad;
    ctx.fill();

    // Lower fill (negative half — mirror)
    ctx.beginPath();
    ctx.moveTo(pad.left, midY);
    for (let i = 0; i < n; i++) {
      const x = pad.left + i * step;
      const y = midY - (waveform[i] / peak) * (plotH / 2);
      ctx.lineTo(x, y);
    }
    ctx.lineTo(pad.left + (n - 1) * step, midY);
    ctx.closePath();

    const grad2 = ctx.createLinearGradient(0, midY, 0, pad.top + plotH);
    grad2.addColorStop(0, 'rgba(99, 102, 241, 0.05)');
    grad2.addColorStop(1, 'rgba(99, 102, 241, 0.25)');
    ctx.fillStyle = grad2;
    ctx.fill();

    // Stroke the waveform line
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const x = pad.left + i * step;
      const y = midY - (waveform[i] / peak) * (plotH / 2);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = 'rgba(99, 102, 241, 0.7)';
    ctx.lineWidth = 1;
    ctx.stroke();

    // Center line
    ctx.beginPath();
    ctx.moveTo(pad.left, midY);
    ctx.lineTo(pad.left + plotW, midY);
    ctx.strokeStyle = 'rgba(148, 163, 184, 0.25)'; // slate-400
    ctx.lineWidth = 0.5;
    ctx.stroke();

    // Time axis labels
    ctx.fillStyle = 'rgba(148, 163, 184, 0.7)';
    ctx.font = '10px ui-monospace, monospace';
    ctx.textAlign = 'left';
    ctx.fillText('0s', pad.left, h - 4);
    ctx.textAlign = 'center';
    ctx.fillText(`${(duration / 2).toFixed(1)}s`, w / 2, h - 4);
    ctx.textAlign = 'right';
    ctx.fillText(`${duration.toFixed(1)}s`, w - pad.right, h - 4);
  }, [waveform, duration]);

  return (
    <div className="space-y-3 h-full flex flex-col">
      <div className="flex items-center justify-between">
        <p className="text-[11px] font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-500">
          Waveform
        </p>
        <p className="text-[10px] text-slate-400 dark:text-slate-600">
          Amplitude vs Time
        </p>
      </div>
      <canvas
        ref={canvasRef}
        className="w-full h-40 rounded-lg"
      />
    </div>
  );
}
