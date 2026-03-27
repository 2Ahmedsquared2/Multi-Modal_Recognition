import { useState, useRef, useEffect, useCallback, useMemo } from 'react';
import { getWaveformEnvelope, extractSegmentAsWav } from '../utils/audioEncoder';

interface AudioTrimmerProps {
  /** The uploaded audio file */
  file: File;
  /** How many seconds the model needs */
  clipDuration?: number;
  /** Called when the user clicks "Classify This Segment" */
  onClassify: (wavBlob: Blob) => void;
  /** Called when user wants to go back / pick another file */
  onCancel: () => void;
}

export default function AudioTrimmer({
  file,
  clipDuration = 2,
  onClassify,
  onCancel,
}: AudioTrimmerProps) {
  const [audioBuffer, setAudioBuffer] = useState<AudioBuffer | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [startSec, setStartSec] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackTime, setPlaybackTime] = useState(0);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const sourceRef = useRef<AudioBufferSourceNode | null>(null);
  const playStartRef = useRef(0);
  const rafRef = useRef(0);
  const draggingRef = useRef(false);

  const duration = audioBuffer?.duration ?? 0;
  const waveform = useMemo(
    () => (audioBuffer ? getWaveformEnvelope(audioBuffer, 1000) : []),
    [audioBuffer],
  );

  // Decode audio file
  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);

    const ctx = new AudioContext();
    audioContextRef.current = ctx;

    file.arrayBuffer().then((buf) => {
      return ctx.decodeAudioData(buf);
    }).then((decoded) => {
      if (cancelled) return;
      setAudioBuffer(decoded);
      setLoading(false);
    }).catch((err) => {
      if (cancelled) return;
      setError(`Could not decode audio: ${err.message}`);
      setLoading(false);
    });

    return () => {
      cancelled = true;
      ctx.close();
    };
  }, [file]);

  // Clamp startSec when audioBuffer changes
  useEffect(() => {
    if (audioBuffer) {
      const maxStart = Math.max(0, audioBuffer.duration - clipDuration);
      if (startSec > maxStart) setStartSec(maxStart);
    }
  }, [audioBuffer, clipDuration, startSec]);

  // ── Draw waveform + selection ──
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || waveform.length === 0 || duration === 0) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    const ctx = canvas.getContext('2d')!;
    ctx.scale(dpr, dpr);

    const w = rect.width;
    const h = rect.height;
    const pad = { top: 8, bottom: 24, left: 4, right: 4 };
    const plotW = w - pad.left - pad.right;
    const plotH = h - pad.top - pad.bottom;
    const midY = pad.top + plotH / 2;

    ctx.clearRect(0, 0, w, h);

    // Selection region highlight
    const selStartX = pad.left + (startSec / duration) * plotW;
    const selEndX = pad.left + (Math.min(startSec + clipDuration, duration) / duration) * plotW;

    // Dimmed background
    ctx.fillStyle = 'rgba(100, 116, 139, 0.15)'; // slate tint
    ctx.fillRect(pad.left, pad.top, selStartX - pad.left, plotH);
    ctx.fillRect(selEndX, pad.top, w - pad.right - selEndX, plotH);

    // Selection highlight
    ctx.fillStyle = 'rgba(99, 102, 241, 0.12)'; // indigo tint
    ctx.fillRect(selStartX, pad.top, selEndX - selStartX, plotH);

    // Selection borders
    ctx.strokeStyle = 'rgba(99, 102, 241, 0.7)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(selStartX, pad.top);
    ctx.lineTo(selStartX, pad.top + plotH);
    ctx.moveTo(selEndX, pad.top);
    ctx.lineTo(selEndX, pad.top + plotH);
    ctx.stroke();

    // Draw handle grips on selection borders
    for (const x of [selStartX, selEndX]) {
      ctx.fillStyle = 'rgba(99, 102, 241, 0.9)';
      const gripH = 20;
      const gripW = 6;
      ctx.beginPath();
      ctx.roundRect(x - gripW / 2, midY - gripH / 2, gripW, gripH, 3);
      ctx.fill();
    }

    // Draw waveform
    const peak = Math.max(...waveform.map(Math.abs), 0.01);
    const n = waveform.length;
    const step = plotW / (n - 1);

    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const x = pad.left + i * step;
      const y = midY - (waveform[i] / peak) * (plotH / 2) * 0.9;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = 'rgba(99, 102, 241, 0.5)';
    ctx.lineWidth = 1;
    ctx.stroke();

    // Playback position line
    if (isPlaying && playbackTime > 0) {
      const px = pad.left + (playbackTime / duration) * plotW;
      ctx.beginPath();
      ctx.moveTo(px, pad.top);
      ctx.lineTo(px, pad.top + plotH);
      ctx.strokeStyle = 'rgba(244, 63, 94, 0.8)'; // rose-500
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }

    // Center line
    ctx.beginPath();
    ctx.moveTo(pad.left, midY);
    ctx.lineTo(w - pad.right, midY);
    ctx.strokeStyle = 'rgba(148, 163, 184, 0.15)';
    ctx.lineWidth = 0.5;
    ctx.stroke();

    // Time labels
    ctx.fillStyle = 'rgba(148, 163, 184, 0.7)';
    ctx.font = '10px ui-monospace, monospace';
    ctx.textAlign = 'left';
    ctx.fillText('0:00', pad.left, h - 4);
    ctx.textAlign = 'center';
    ctx.fillText(formatTime(duration / 2), w / 2, h - 4);
    ctx.textAlign = 'right';
    ctx.fillText(formatTime(duration), w - pad.right, h - 4);

    // Selection time label
    ctx.textAlign = 'center';
    ctx.fillStyle = 'rgba(99, 102, 241, 0.9)';
    ctx.font = '10px ui-monospace, monospace';
    const labelX = (selStartX + selEndX) / 2;
    ctx.fillText(
      `${formatTime(startSec)} — ${formatTime(Math.min(startSec + clipDuration, duration))}`,
      labelX,
      pad.top - 1,
    );
  }, [waveform, duration, startSec, clipDuration, isPlaying, playbackTime]);

  // ── Drag to reposition selection ──
  const handlePointerDown = useCallback((e: React.PointerEvent) => {
    if (!canvasRef.current || duration === 0) return;
    draggingRef.current = true;
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
    updateStartFromPointer(e);
  }, [duration, clipDuration]);

  const handlePointerMove = useCallback((e: React.PointerEvent) => {
    if (!draggingRef.current) return;
    updateStartFromPointer(e);
  }, [duration, clipDuration]);

  const handlePointerUp = useCallback(() => {
    draggingRef.current = false;
  }, []);

  const updateStartFromPointer = useCallback((e: React.PointerEvent) => {
    const canvas = canvasRef.current;
    if (!canvas || duration === 0) return;
    const rect = canvas.getBoundingClientRect();
    const pad = { left: 4, right: 4 };
    const plotW = rect.width - pad.left - pad.right;
    const mx = e.clientX - rect.left - pad.left;
    const frac = Math.max(0, Math.min(1, mx / plotW));
    const t = frac * duration;
    // Center the selection around the click
    const maxStart = Math.max(0, duration - clipDuration);
    setStartSec(Math.max(0, Math.min(maxStart, t - clipDuration / 2)));
  }, [duration, clipDuration]);

  // ── Playback ──
  const togglePlayback = useCallback(() => {
    if (!audioBuffer || !audioContextRef.current) return;
    const ctx = audioContextRef.current;

    if (isPlaying && sourceRef.current) {
      sourceRef.current.stop();
      sourceRef.current = null;
      cancelAnimationFrame(rafRef.current);
      setIsPlaying(false);
      return;
    }

    const source = ctx.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(ctx.destination);
    source.start(0, startSec, clipDuration);
    sourceRef.current = source;
    playStartRef.current = ctx.currentTime;
    setIsPlaying(true);

    const tick = () => {
      const elapsed = ctx.currentTime - playStartRef.current;
      setPlaybackTime(startSec + elapsed);
      if (elapsed < clipDuration) {
        rafRef.current = requestAnimationFrame(tick);
      } else {
        setIsPlaying(false);
        setPlaybackTime(0);
      }
    };
    rafRef.current = requestAnimationFrame(tick);

    source.onended = () => {
      setIsPlaying(false);
      setPlaybackTime(0);
      cancelAnimationFrame(rafRef.current);
    };
  }, [audioBuffer, isPlaying, startSec, clipDuration]);

  // ── Classify selected segment ──
  const handleClassify = useCallback(() => {
    if (!audioBuffer) return;
    const blob = extractSegmentAsWav(audioBuffer, startSec, clipDuration);
    onClassify(blob);
  }, [audioBuffer, startSec, clipDuration, onClassify]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (sourceRef.current) sourceRef.current.stop();
      cancelAnimationFrame(rafRef.current);
    };
  }, []);

  if (loading) {
    return (
      <div className="card p-8 flex flex-col items-center gap-3 animate-fade-in">
        <div className="w-6 h-6 border-2 border-accent border-t-transparent rounded-full animate-spin" />
        <p className="text-sm text-warm-500 dark:text-warm-400">Decoding audio...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="card p-6 space-y-3 animate-fade-in">
        <p className="text-sm text-rose-600 dark:text-rose-400">{error}</p>
        <button onClick={onCancel} className="text-sm text-warm-500 hover:text-warm-700 dark:hover:text-warm-300">
          Try another file
        </button>
      </div>
    );
  }

  const isShortClip = duration <= clipDuration;

  return (
    <div className="space-y-4 animate-fade-in-up">
      {/* File info */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 min-w-0">
          <span className="text-sm font-medium text-warm-700 dark:text-warm-300 truncate">
            {file.name}
          </span>
          <span className="text-xs text-warm-400 dark:text-warm-600 font-mono shrink-0">
            {formatTime(duration)}
          </span>
        </div>
        <button
          onClick={onCancel}
          className="text-xs text-warm-500 hover:text-warm-700 dark:hover:text-warm-300
            transition-colors shrink-0 ml-2"
        >
          Change file
        </button>
      </div>

      {/* Waveform + selection */}
      {!isShortClip && (
        <div className="card p-4 space-y-2">
          <div className="flex items-center justify-between">
            <p className="text-[11px] font-semibold uppercase tracking-wider text-warm-500 dark:text-warm-500">
              Select a {clipDuration}s segment to classify
            </p>
            <p className="text-[10px] text-warm-400 dark:text-warm-600">
              Click or drag to reposition
            </p>
          </div>
          <div ref={containerRef}>
            <canvas
              ref={canvasRef}
              className="w-full h-36 rounded-lg cursor-col-resize"
              onPointerDown={handlePointerDown}
              onPointerMove={handlePointerMove}
              onPointerUp={handlePointerUp}
            />
          </div>
        </div>
      )}

      {/* Controls */}
      <div className="flex items-center gap-3">
        {/* Play selected segment */}
        <button
          onClick={togglePlayback}
          className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium
            bg-warm-200 text-warm-600 dark:bg-warm-700 dark:text-warm-300
            hover:bg-warm-300 dark:hover:bg-warm-600
            transition-colors duration-150"
        >
          {isPlaying ? (
            <svg className="w-4 h-4" viewBox="0 0 24 24" fill="currentColor">
              <rect x="6" y="4" width="4" height="16" rx="1" />
              <rect x="14" y="4" width="4" height="16" rx="1" />
            </svg>
          ) : (
            <svg className="w-4 h-4" viewBox="0 0 24 24" fill="currentColor">
              <path d="M8 5v14l11-7z" />
            </svg>
          )}
          {isPlaying ? 'Pause' : isShortClip ? 'Preview' : 'Preview Segment'}
        </button>

        {/* Classify */}
        <button
          onClick={handleClassify}
          className="flex items-center gap-2 px-5 py-2 rounded-lg text-sm font-medium
            bg-accent text-white hover:bg-accent
            dark:bg-accent dark:hover:bg-accent
            transition-colors duration-150"
        >
          <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 3v18M6 7v10M18 7v10M3 10v4M21 10v4M9 5v14M15 5v14" />
          </svg>
          {isShortClip ? 'Classify' : 'Classify This Segment'}
        </button>
      </div>

      {!isShortClip && (
        <p className="text-[10px] text-warm-400 dark:text-warm-600 italic">
          The model classifies {clipDuration}-second clips. Pick the section that best represents the instrument.
        </p>
      )}
    </div>
  );
}

function formatTime(sec: number): string {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}
