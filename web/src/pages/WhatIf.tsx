import { useState, useRef, useCallback } from 'react';
import { api } from '../api/client';
import { useModel } from '../contexts/ModelContext';
import type { WhatIfResponse } from '../types';
import SpectrogramEditor from '../components/SpectrogramEditor';
import type { BrushTool } from '../components/SpectrogramEditor';
import AudioTrimmer from '../components/AudioTrimmer';
import MicrophoneRecorder from '../components/MicrophoneRecorder';

// ── Confidence bar list ──────────────────────────────────────────────────

function ConfidenceBars({
  prediction,
  originalConfidences,
}: {
  prediction: WhatIfResponse;
  originalConfidences?: Record<string, number>;
}) {
  const sorted = Object.entries(prediction.all_confidences).sort(
    ([, a], [, b]) => b - a,
  );

  return (
    <div className="space-y-1.5">
      {sorted.slice(0, 5).map(([cls, prob], idx) => {
        const isTop = idx === 0;
        const origProb =
          originalConfidences !== undefined
            ? (originalConfidences[cls] ?? 0)
            : null;
        const delta = origProb !== null ? prob - origProb : null;

        return (
          <div key={cls} className="space-y-0.5">
            <div className="flex justify-between items-center text-xs">
              <span
                className={
                  isTop
                    ? 'font-semibold text-slate-900 dark:text-white'
                    : 'text-slate-500 dark:text-slate-400'
                }
              >
                {cls}
              </span>
              <div className="flex items-center gap-1.5">
                {delta !== null && Math.abs(delta) > 0.005 && (
                  <span
                    className={`text-[10px] font-mono ${
                      delta > 0 ? 'text-emerald-500' : 'text-rose-500'
                    }`}
                  >
                    {delta > 0 ? '+' : ''}
                    {(delta * 100).toFixed(1)}
                  </span>
                )}
                <span
                  className={`font-mono ${
                    isTop
                      ? 'text-indigo-600 dark:text-indigo-400 font-semibold'
                      : 'text-slate-400 dark:text-slate-600'
                  }`}
                >
                  {(prob * 100).toFixed(1)}%
                </span>
              </div>
            </div>
            <div className="h-1.5 bg-slate-100 dark:bg-slate-800 rounded-full overflow-hidden">
              <div
                className={`h-full rounded-full transition-all duration-500 ease-out ${
                  isTop
                    ? 'bg-indigo-500 dark:bg-indigo-400'
                    : 'bg-slate-300 dark:bg-slate-700'
                }`}
                style={{ width: `${prob * 100}%` }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}

// ── Preset definitions ───────────────────────────────────────────────────

const PRESETS = [
  { id: 'remove-low', label: 'Remove Low Freq', icon: '⬇' },
  { id: 'remove-high', label: 'Remove High Freq', icon: '⬆' },
  { id: 'keep-mid', label: 'Keep Mid-Range', icon: '⬌' },
  { id: 'add-noise', label: 'Add Noise', icon: '〰' },
  { id: 'reset', label: 'Reset', icon: '↺' },
] as const;

// ── Phase type ───────────────────────────────────────────────────────────

type Phase = 'upload' | 'trim' | 'recording' | 'loaded';

// ── Main page component ─────────────────────────────────────────────────

export default function WhatIf() {
  const { engine, engineLabel } = useModel();

  // Phase & audio source
  const [phase, setPhase] = useState<Phase>('upload');
  const [audioFile, setAudioFile] = useState<File | null>(null);

  // Spectrogram & predictions
  const [originalSpec, setOriginalSpec] = useState<number[][] | null>(null);
  const [modifiedSpec, setModifiedSpec] = useState<number[][] | null>(null);
  const [originalPred, setOriginalPred] = useState<WhatIfResponse | null>(null);
  const [modifiedPred, setModifiedPred] = useState<WhatIfResponse | null>(null);

  // Editor state
  const [tool, setTool] = useState<BrushTool>('paint');
  const [brushSize, setBrushSize] = useState(3);
  const [loading, setLoading] = useState(false);
  const [predicting, setPredicting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [insight, setInsight] = useState('');

  const fileInputRef = useRef<HTMLInputElement>(null);
  const originalPredRef = useRef<WhatIfResponse | null>(null);

  // ── Phase 1a: User picks a file → go to trimmer ──

  const handleFileSelected = useCallback((file: File) => {
    setError(null);
    setAudioFile(file);
    setPhase('trim');
  }, []);

  // ── Phase 1b: Mic recording finished → go to trimmer ──

  const handleMicRecording = useCallback((file: File) => {
    setError(null);
    setAudioFile(file);
    setPhase('trim');
  }, []);

  // ── Phase 2: User clicks "Classify This Segment" from trimmer ──

  const handleClassifyBlob = useCallback(async (blob: Blob) => {
    setError(null);
    setLoading(true);
    try {
      const wavFile = new File([blob], 'segment.wav', { type: 'audio/wav' });
      const result = await api.classify(wavFile, engine);
      const spec = result.spectrogram;
      const pred: WhatIfResponse = {
        prediction: result.prediction,
        confidence: result.confidence,
        all_confidences: result.all_confidences,
      };

      setOriginalSpec(spec);
      setModifiedSpec(spec.map((row) => [...row]));
      setOriginalPred(pred);
      setModifiedPred(pred);
      originalPredRef.current = pred;
      setInsight('');
      setPhase('loaded');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to process audio');
    } finally {
      setLoading(false);
    }
  }, []);

  // ── Reset everything ──

  const resetState = useCallback(() => {
    setPhase('upload');
    setAudioFile(null);
    setOriginalSpec(null);
    setModifiedSpec(null);
    setOriginalPred(null);
    setModifiedPred(null);
    originalPredRef.current = null;
    setError(null);
    setInsight('');
    if (fileInputRef.current) fileInputRef.current.value = '';
  }, []);

  // ── Re-predict the modified spectrogram via /api/what-if ──

  const predict = useCallback(async (spec: number[][]) => {
    setPredicting(true);
    try {
      const result = await api.whatIf(spec, engine);
      setModifiedPred(result);

      // Generate insight by comparing to original
      const orig = originalPredRef.current;
      if (orig) {
        if (result.prediction !== orig.prediction) {
          setInsight(
            `The prediction shifted from ${orig.prediction} ` +
              `(${(orig.confidence * 100).toFixed(1)}%) to ` +
              `${result.prediction} (${(result.confidence * 100).toFixed(1)}%). ` +
              `This suggests the model relies on the modified frequency regions ` +
              `to distinguish these instruments.`,
          );
        } else if (result.confidence < orig.confidence - 0.1) {
          setInsight(
            `Still ${result.prediction}, but confidence dropped from ` +
              `${(orig.confidence * 100).toFixed(1)}% to ` +
              `${(result.confidence * 100).toFixed(1)}%. ` +
              `The edited regions contribute to the model's certainty.`,
          );
        } else if (result.confidence > orig.confidence + 0.05) {
          setInsight(
            `Confidence increased from ` +
              `${(orig.confidence * 100).toFixed(1)}% to ` +
              `${(result.confidence * 100).toFixed(1)}%. ` +
              `Your edits reinforced the model's decision.`,
          );
        } else {
          setInsight(
            `No significant change — the model's prediction is robust to these modifications.`,
          );
        }
      }
    } catch (err) {
      console.error('What-if prediction failed:', err);
    } finally {
      setPredicting(false);
    }
  }, []);

  // ── Canvas edit handler (called on mouseUp from SpectrogramEditor) ──

  const handleSpecChange = useCallback(
    (newSpec: number[][]) => {
      setModifiedSpec(newSpec);
      predict(newSpec);
    },
    [predict],
  );

  // ── Preset experiments ──

  const applyPreset = useCallback(
    (presetId: string) => {
      if (!originalSpec) return;
      const spec = originalSpec.map((row) => [...row]);
      const nRows = spec.length;
      const nCols = spec[0]?.length ?? 0;

      switch (presetId) {
        case 'remove-low': {
          const cutoff = Math.floor(nRows / 3);
          for (let r = 0; r < cutoff; r++)
            for (let c = 0; c < nCols; c++) spec[r][c] = 0;
          break;
        }
        case 'remove-high': {
          const cutoff = Math.floor((nRows * 2) / 3);
          for (let r = cutoff; r < nRows; r++)
            for (let c = 0; c < nCols; c++) spec[r][c] = 0;
          break;
        }
        case 'keep-mid': {
          const lo = Math.floor(nRows / 3);
          const hi = Math.floor((nRows * 2) / 3);
          for (let r = 0; r < nRows; r++) {
            if (r < lo || r >= hi) {
              for (let c = 0; c < nCols; c++) spec[r][c] = 0;
            }
          }
          break;
        }
        case 'add-noise': {
          for (let r = 0; r < nRows; r++)
            for (let c = 0; c < nCols; c++)
              spec[r][c] = Math.max(
                0,
                Math.min(1, spec[r][c] + (Math.random() - 0.5) * 0.3),
              );
          break;
        }
        case 'reset':
          // spec is already a fresh copy of originalSpec
          break;
      }

      setModifiedSpec(spec);
      predict(spec);
    },
    [originalSpec, predict],
  );

  return (
    <div className="max-w-6xl mx-auto px-8 py-10 space-y-8">
      {/* ── Header ── */}
      <header className="space-y-2">
        <div className="flex items-center gap-2">
          <h1 className="text-2xl font-bold tracking-tight text-slate-900 dark:text-white">
            What-If Tool
          </h1>
          <span className="badge-indigo">Interactive</span>
          <span className={`text-[10px] font-medium px-2 py-0.5 rounded-full
            ${engine === 'custom'
              ? 'bg-indigo-100 text-indigo-700 dark:bg-indigo-500/10 dark:text-indigo-400'
              : 'bg-orange-100 text-orange-700 dark:bg-orange-500/10 dark:text-orange-400'
            }`}
          >
            {engineLabel}
          </span>
        </div>
        <p className="text-sm text-slate-600 dark:text-slate-400">
          Paint on a spectrogram and watch the {engineLabel} model's prediction change in real
          time. Discover which frequency regions matter most for each sound
          class.
        </p>
      </header>

      {/* ═══════════════════════════════════════════════════════════════════
          Phase 1: Upload / source selection
          ═══════════════════════════════════════════════════════════════════ */}
      {phase === 'upload' && (
        <div className="grid grid-cols-1 md:grid-cols-5 gap-6">
          <div className="md:col-span-3 space-y-4">
            {/* Drop zone */}
            <div
              onClick={() => fileInputRef.current?.click()}
              className="relative cursor-pointer rounded-xl border-2 border-dashed p-10
                flex flex-col items-center justify-center text-center
                transition-all duration-200
                border-slate-300 dark:border-slate-800
                hover:border-slate-400 dark:hover:border-slate-700"
            >
              <input
                ref={fileInputRef}
                type="file"
                accept="audio/*,.wav,.mp3,.ogg,.flac"
                className="hidden"
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (file) handleFileSelected(file);
                }}
              />
              <div className="w-12 h-12 rounded-xl bg-slate-100 dark:bg-slate-800 flex items-center justify-center mb-4">
                <svg
                  className="w-6 h-6 text-slate-500"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth={1.5}
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <path d="M12 16V3M12 3l4 4M12 3L8 7" />
                  <path d="M2 17v2a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-2" />
                </svg>
              </div>
              <p className="text-sm font-medium text-slate-700 dark:text-slate-300">
                Drop audio file here
              </p>
              <p className="text-xs text-slate-500 dark:text-slate-500 mt-1">
                or click to browse — WAV, MP3, OGG, FLAC
              </p>
            </div>

            {/* Mic record button */}
            <button
              onClick={() => setPhase('recording')}
              className="w-full py-3 rounded-xl text-sm font-medium
                bg-slate-100 text-slate-600
                dark:bg-slate-800 dark:text-slate-300
                hover:bg-slate-200 dark:hover:bg-slate-700
                transition-colors duration-150
                flex items-center justify-center gap-2"
            >
              <svg
                className="w-4 h-4"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth={1.5}
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <rect x="9" y="2" width="6" height="12" rx="3" />
                <path d="M5 10a7 7 0 0 0 14 0M12 18v4M8 22h8" />
              </svg>
              Record from Microphone
            </button>
          </div>

          {/* Placeholder right panel */}
          <div className="hidden md:block md:col-span-2">
            <div className="card p-8 flex flex-col items-center text-center">
              <div className="w-16 h-16 rounded-2xl bg-slate-50 dark:bg-slate-800 flex items-center justify-center mb-4">
                <svg
                  className="w-8 h-8 text-slate-400 dark:text-slate-600"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth={1}
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <path d="M12 3v18M6 7v10M18 7v10M3 10v4M21 10v4M9 5v14M15 5v14" />
                </svg>
              </div>
              <p className="text-sm text-slate-500 dark:text-slate-500">
                Upload audio or record from your
                <br />
                microphone, then select a 2-second
                <br />
                segment to explore
              </p>
            </div>
          </div>
        </div>
      )}

      {/* ═══════════════════════════════════════════════════════════════════
          Phase 1b: Microphone Recording
          ═══════════════════════════════════════════════════════════════════ */}
      {phase === 'recording' && (
        <MicrophoneRecorder
          onRecordingComplete={handleMicRecording}
          onCancel={resetState}
        />
      )}

      {/* ═══════════════════════════════════════════════════════════════════
          Phase 2: Trim / select 2-second segment
          ═══════════════════════════════════════════════════════════════════ */}
      {phase === 'trim' && audioFile && (
        <div className="space-y-4">
          <AudioTrimmer
            file={audioFile}
            clipDuration={2}
            onClassify={handleClassifyBlob}
            onCancel={resetState}
          />

          {/* Loading indicator */}
          {loading && (
            <div className="card p-6 flex items-center gap-3 animate-fade-in">
              <div className="w-5 h-5 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin" />
              <div>
                <p className="text-sm font-medium text-slate-700 dark:text-slate-300">
                  Analyzing audio…
                </p>
                <p className="text-xs text-slate-500 dark:text-slate-500">
                  Generating spectrogram → Running inference
                </p>
              </div>
            </div>
          )}

          {error && (
            <div className="card p-4 border-rose-200 dark:border-rose-500/20 bg-rose-50 dark:bg-rose-500/5 animate-fade-in">
              <p className="text-sm text-rose-600 dark:text-rose-400">
                {error}
              </p>
            </div>
          )}
        </div>
      )}

      {/* ═══════════════════════════════════════════════════════════════════
          Phase 3: Loaded — spectrogram editor
          ═══════════════════════════════════════════════════════════════════ */}
      {phase === 'loaded' && originalSpec && modifiedSpec && (
        <div className="space-y-6">
          {/* ── Top bar: source controls + predicting indicator ── */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <button
                onClick={() => {
                  setPhase('trim');
                }}
                className="px-3 py-1.5 rounded-lg text-xs font-medium
                  bg-indigo-50 text-indigo-700 dark:bg-indigo-500/10 dark:text-indigo-400
                  hover:bg-indigo-100 dark:hover:bg-indigo-500/20
                  transition-colors duration-150"
              >
                Try Different Segment
              </button>
              <button
                onClick={resetState}
                className="px-3 py-1.5 rounded-lg text-xs font-medium
                  bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-400
                  hover:bg-slate-200 dark:hover:bg-slate-700
                  transition-colors duration-150"
              >
                New Audio
              </button>
            </div>
            {predicting && (
              <div className="flex items-center gap-1.5 text-xs text-indigo-500">
                <div className="w-3 h-3 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin" />
                Predicting…
              </div>
            )}
          </div>

          {/* ── Before / After spectrograms ── */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Original (read-only) */}
            <div className="space-y-4">
              <div className="card p-4">
                <SpectrogramEditor
                  spectrogram={originalSpec}
                  readOnly
                  label="Original"
                />
              </div>
              <div className="card p-4 space-y-3">
                <div className="flex items-baseline gap-2">
                  <span className="text-sm font-bold text-slate-900 dark:text-white capitalize">
                    {originalPred?.prediction ?? '—'}
                  </span>
                  <span className="text-xs font-mono text-indigo-500">
                    {originalPred
                      ? `${(originalPred.confidence * 100).toFixed(1)}%`
                      : ''}
                  </span>
                </div>
                {originalPred && (
                  <ConfidenceBars prediction={originalPred} />
                )}
              </div>
            </div>

            {/* Modified (editable) */}
            <div className="space-y-4">
              <div className="card p-4">
                <SpectrogramEditor
                  spectrogram={modifiedSpec}
                  onChange={handleSpecChange}
                  tool={tool}
                  brushSize={brushSize}
                  label="Modified — draw here"
                />
              </div>
              <div className="card p-4 space-y-3">
                <div className="flex items-baseline gap-2">
                  <span
                    className={`text-sm font-bold capitalize ${
                      modifiedPred?.prediction !== originalPred?.prediction
                        ? 'text-amber-500'
                        : 'text-slate-900 dark:text-white'
                    }`}
                  >
                    {modifiedPred?.prediction ?? '—'}
                  </span>
                  <span className="text-xs font-mono text-indigo-500">
                    {modifiedPred
                      ? `${(modifiedPred.confidence * 100).toFixed(1)}%`
                      : ''}
                  </span>
                  {modifiedPred &&
                    originalPred &&
                    modifiedPred.prediction !== originalPred.prediction && (
                      <span className="text-[10px] px-1.5 py-0.5 rounded bg-amber-100 text-amber-700 dark:bg-amber-500/10 dark:text-amber-400 font-medium">
                        Changed
                      </span>
                    )}
                </div>
                {modifiedPred && originalPred && (
                  <ConfidenceBars
                    prediction={modifiedPred}
                    originalConfidences={originalPred.all_confidences}
                  />
                )}
              </div>
            </div>
          </div>

          {/* ── Tool palette ── */}
          <div className="card p-4">
            <div className="flex flex-wrap items-center gap-6">
              {/* Brush tool */}
              <div className="flex items-center gap-2">
                <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-500 mr-1">
                  Tool
                </span>
                <button
                  onClick={() => setTool('paint')}
                  className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
                    tool === 'paint'
                      ? 'bg-amber-100 text-amber-700 dark:bg-amber-500/15 dark:text-amber-400'
                      : 'bg-slate-100 text-slate-500 dark:bg-slate-800 dark:text-slate-400 hover:bg-slate-200 dark:hover:bg-slate-700'
                  }`}
                >
                  Paint (+)
                </button>
                <button
                  onClick={() => setTool('erase')}
                  className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
                    tool === 'erase'
                      ? 'bg-rose-100 text-rose-700 dark:bg-rose-500/15 dark:text-rose-400'
                      : 'bg-slate-100 text-slate-500 dark:bg-slate-800 dark:text-slate-400 hover:bg-slate-200 dark:hover:bg-slate-700'
                  }`}
                >
                  Erase (−)
                </button>
              </div>

              {/* Brush size */}
              <div className="flex items-center gap-2">
                <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-500">
                  Size
                </span>
                {([1, 3, 6] as const).map((size) => (
                  <button
                    key={size}
                    onClick={() => setBrushSize(size)}
                    className={`w-8 h-8 rounded-lg text-xs font-medium transition-colors
                      flex items-center justify-center ${
                        brushSize === size
                          ? 'bg-indigo-100 text-indigo-700 dark:bg-indigo-500/15 dark:text-indigo-400'
                          : 'bg-slate-100 text-slate-500 dark:bg-slate-800 dark:text-slate-400 hover:bg-slate-200 dark:hover:bg-slate-700'
                      }`}
                  >
                    {size === 1 ? 'S' : size === 3 ? 'M' : 'L'}
                  </button>
                ))}
              </div>

              {/* Divider */}
              <div className="w-px h-6 bg-slate-200 dark:bg-slate-700" />

              {/* Presets */}
              <div className="flex items-center gap-2 flex-wrap">
                <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-500 mr-1">
                  Presets
                </span>
                {PRESETS.map((p) => (
                  <button
                    key={p.id}
                    onClick={() => applyPreset(p.id)}
                    className={`px-2.5 py-1.5 rounded-lg text-xs font-medium transition-colors ${
                      p.id === 'reset'
                        ? 'bg-slate-200 text-slate-700 dark:bg-slate-700 dark:text-slate-300 hover:bg-slate-300 dark:hover:bg-slate-600'
                        : 'bg-slate-100 text-slate-500 dark:bg-slate-800 dark:text-slate-400 hover:bg-slate-200 dark:hover:bg-slate-700'
                    }`}
                  >
                    <span className="mr-1">{p.icon}</span>
                    {p.label}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* ── Insight panel ── */}
          {insight && (
            <div className="card p-4 border-indigo-200 dark:border-indigo-500/20 bg-indigo-50/50 dark:bg-indigo-500/5">
              <div className="flex gap-3">
                <div className="shrink-0 w-8 h-8 rounded-lg bg-indigo-100 dark:bg-indigo-500/10 flex items-center justify-center">
                  <svg
                    className="w-4 h-4 text-indigo-500"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth={2}
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  >
                    <circle cx="12" cy="12" r="10" />
                    <path d="M12 16v-4M12 8h.01" />
                  </svg>
                </div>
                <div>
                  <p className="text-[11px] font-semibold uppercase tracking-wider text-indigo-600 dark:text-indigo-400 mb-1">
                    Insight
                  </p>
                  <p className="text-sm text-slate-700 dark:text-slate-300">
                    {insight}
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
