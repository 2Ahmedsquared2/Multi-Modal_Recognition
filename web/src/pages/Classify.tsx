import { useState, useRef, useCallback, useEffect } from 'react';
import { api } from '../api/client';
import { useModel } from '../contexts/ModelContext';
import type { ClassifyResult } from '../types';
import AudioTrimmer from '../components/AudioTrimmer';
import MicrophoneRecorder from '../components/MicrophoneRecorder';
import PipelineAnimation from '../components/PipelineAnimation';
import WaveformDisplay from '../components/WaveformDisplay';
import SpectrogramDisplay from '../components/SpectrogramDisplay';

type Phase = 'upload' | 'trim' | 'recording' | 'pipeline' | 'results';
type InputSource = 'upload' | 'mic';

export default function Classify() {
  const { engine, engineLabel } = useModel();
  const [phase, setPhase] = useState<Phase>('upload');
  const [result, setResult] = useState<ClassifyResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [fileName, setFileName] = useState<string | null>(null);
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [source, setSource] = useState<InputSource>('upload');
  const fileInputRef = useRef<HTMLInputElement>(null);
  const hasAnimatedRef = useRef(false);

  // Animation: bars start at 0 and grow to real values after mount
  const [animateBars, setAnimateBars] = useState(false);
  useEffect(() => {
    if (result) {
      const id = requestAnimationFrame(() => setAnimateBars(true));
      return () => cancelAnimationFrame(id);
    }
    setAnimateBars(false);
  }, [result]);

  // ── Phase 1a: User picks a file ──
  const handleFileSelected = useCallback((file: File) => {
    setError(null);
    setResult(null);
    setFileName(file.name);
    setAudioFile(file);
    setSource('upload');
    setPhase('trim');
  }, []);

  // ── Phase 2: User clicks "Classify" from trimmer (sends a WAV blob) ──
  const handleClassifyBlob = useCallback(async (blob: Blob) => {
    setError(null);
    setResult(null);
    setLoading(true);
    try {
      const wavFile = new File([blob], 'segment.wav', { type: 'audio/wav' });
      const res = await api.classify(wavFile, engine);
      setResult(res);
      if (!hasAnimatedRef.current) {
        setPhase('pipeline');
      } else {
        setPhase('results');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Classification failed');
    } finally {
      setLoading(false);
    }
  }, []);

  // ── Phase 1b: Mic recording finished → go to trimmer (same as file upload) ──
  const handleMicRecording = useCallback((file: File) => {
    setError(null);
    setResult(null);
    setFileName('Microphone Recording');
    setAudioFile(file);
    setSource('mic');
    setPhase('trim');
  }, []);

  // ── Pipeline animation finished ──
  const handlePipelineComplete = useCallback(() => {
    hasAnimatedRef.current = true;
    setPhase('results');
  }, []);

  // ── Reset everything ──
  const resetState = useCallback(() => {
    setPhase('upload');
    setResult(null);
    setError(null);
    setFileName(null);
    setAudioFile(null);
    setSource('upload');
    setAnimateBars(false);
    if (fileInputRef.current) fileInputRef.current.value = '';
  }, []);

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFileSelected(file);
  }, [handleFileSelected]);

  // Sorted confidences for the bar chart
  const sortedConfidences = result
    ? Object.entries(result.all_confidences).sort(([, a], [, b]) => b - a)
    : [];

  return (
    <div className="max-w-4xl mx-auto px-8 py-10 space-y-8">

      {/* Header */}
      <header className="space-y-2">
        <div className="flex items-center gap-2">
          <h1 className="text-2xl font-bold tracking-tight text-slate-900 dark:text-white">
            Classify Audio
          </h1>
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
          Upload an audio file or record from your microphone. The {engineLabel} neural network will analyze
          the spectrogram and predict what sound it is.
        </p>
      </header>

      {/* ═══════════════════════════════════════════════════════════════
          Phase 1: Upload
          ═══════════════════════════════════════════════════════════════ */}
      {phase === 'upload' && (
        <div className="grid grid-cols-1 md:grid-cols-5 gap-6">
          <div className="md:col-span-3 space-y-4">
            {/* Drop zone */}
            <div
              onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
              onDragLeave={() => setDragOver(false)}
              onDrop={onDrop}
              onClick={() => fileInputRef.current?.click()}
              className={`relative cursor-pointer rounded-xl border-2 border-dashed p-10
                flex flex-col items-center justify-center text-center
                transition-all duration-200
                ${dragOver
                  ? 'border-indigo-400 bg-indigo-50 dark:bg-indigo-500/5'
                  : 'border-slate-300 dark:border-slate-800 hover:border-slate-400 dark:hover:border-slate-700'
                }`}
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
                <svg className="w-6 h-6 text-slate-500" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round">
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
              <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round">
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
                <svg className="w-8 h-8 text-slate-400 dark:text-slate-600" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1} strokeLinecap="round" strokeLinejoin="round">
                  <path d="M12 3v18M6 7v10M18 7v10M3 10v4M21 10v4M9 5v14M15 5v14" />
                </svg>
              </div>
              <p className="text-sm text-slate-500 dark:text-slate-500">
                Upload an audio file to see<br />classification results here
              </p>
            </div>
          </div>
        </div>
      )}

      {/* ═══════════════════════════════════════════════════════════════
          Phase 2: Trim / Preview
          ═══════════════════════════════════════════════════════════════ */}
      {phase === 'trim' && audioFile && (
        <div className="space-y-4">
          <AudioTrimmer
            file={audioFile}
            clipDuration={2}
            onClassify={handleClassifyBlob}
            onCancel={resetState}
          />

          {/* Loading state */}
          {loading && (
            <div className="card p-6 flex items-center gap-3 animate-fade-in">
              <div className="w-5 h-5 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin" />
              <div>
                <p className="text-sm font-medium text-slate-700 dark:text-slate-300">
                  Analyzing audio...
                </p>
                <p className="text-xs text-slate-500 dark:text-slate-500">
                  Generating spectrogram → Running inference
                </p>
              </div>
            </div>
          )}

          {error && (
            <div className="card p-4 border-rose-200 dark:border-rose-500/20 bg-rose-50 dark:bg-rose-500/5 animate-fade-in">
              <p className="text-sm text-rose-600 dark:text-rose-400">{error}</p>
            </div>
          )}
        </div>
      )}

      {/* ═══════════════════════════════════════════════════════════════
          Phase 2b: Microphone Recording
          ═══════════════════════════════════════════════════════════════ */}
      {phase === 'recording' && (
        <MicrophoneRecorder onRecordingComplete={handleMicRecording} onCancel={resetState} />
      )}

      {/* ═══════════════════════════════════════════════════════════════
          Phase 2c: Pipeline Animation (first classify only)
          ═══════════════════════════════════════════════════════════════ */}
      {phase === 'pipeline' && result && (
        <PipelineAnimation
          waveform={result.waveform}
          spectrogram={result.spectrogram}
          prediction={result.prediction}
          confidence={result.confidence}
          onComplete={handlePipelineComplete}
        />
      )}

      {/* ═══════════════════════════════════════════════════════════════
          Phase 3: Results
          ═══════════════════════════════════════════════════════════════ */}
      {phase === 'results' && result && (
        <div className="space-y-8 animate-fade-in-up">

          {/* Top row: Prediction + Probabilities */}
          <div className="grid grid-cols-1 md:grid-cols-5 gap-6">
            <div className="md:col-span-3 space-y-4">
              {/* Prediction card */}
              <div className="card p-5 space-y-3">
                <p className="text-[11px] font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-500">
                  Prediction
                </p>
                <div className="flex items-baseline gap-2">
                  <span className="text-xl font-bold text-slate-900 dark:text-white capitalize">
                    {result.prediction}
                  </span>
                  <span className="text-sm font-mono text-indigo-500">
                    {(result.confidence * 100).toFixed(1)}%
                  </span>
                </div>
                {fileName && (
                  <p className="text-xs text-slate-500 dark:text-slate-500 font-mono truncate">
                    {fileName}
                  </p>
                )}
              </div>

              {/* Waveform + Spectrogram side by side */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="card p-4">
                  <WaveformDisplay
                    waveform={result.waveform}
                    duration={result.waveform_summary.duration}
                  />
                </div>
                <div className="card p-4">
                  <SpectrogramDisplay
                    spectrogram={result.spectrogram}
                    duration={result.waveform_summary.duration}
                  />
                </div>
              </div>
            </div>

            {/* Right: Probabilities */}
            <div className="md:col-span-2 space-y-4">
              <div className="card p-5 space-y-3">
                <p className="text-[11px] font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-500">
                  All Probabilities
                </p>
                <div className="space-y-2">
                  {sortedConfidences.map(([cls, prob], idx) => {
                    const isTop = idx === 0;
                    return (
                      <div key={cls} className="space-y-1">
                        <div className="flex justify-between text-xs">
                          <span className={
                            isTop
                              ? 'font-semibold text-slate-900 dark:text-white'
                              : 'text-slate-500 dark:text-slate-400'
                          }>
                            {cls}
                          </span>
                          <span className={`font-mono ${
                            isTop
                              ? 'text-indigo-600 dark:text-indigo-400 font-semibold'
                              : 'text-slate-400 dark:text-slate-600'
                          }`}>
                            {(prob * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="h-1.5 bg-slate-100 dark:bg-slate-800 rounded-full overflow-hidden">
                          <div
                            className={`h-full rounded-full transition-all duration-700 ease-out ${
                              isTop
                                ? 'bg-indigo-500 dark:bg-indigo-400'
                                : 'bg-slate-300 dark:bg-slate-700'
                            }`}
                            style={{ width: animateBars ? `${prob * 100}%` : '0%' }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Action buttons — context-aware for upload vs mic */}
              <button
                onClick={() => {
                  setResult(null);
                  setPhase('trim');
                }}
                className="w-full py-2.5 rounded-xl text-sm font-medium
                  bg-indigo-50 text-indigo-700 dark:bg-indigo-500/10 dark:text-indigo-400
                  hover:bg-indigo-100 dark:hover:bg-indigo-500/20
                  transition-colors duration-150"
              >
                Try Different Segment
              </button>

              {source === 'mic' && (
                <button
                  onClick={() => {
                    setResult(null);
                    setAudioFile(null);
                    setPhase('recording');
                  }}
                  className="w-full py-2.5 rounded-xl text-sm font-medium
                    border border-slate-300 dark:border-slate-700
                    text-slate-600 dark:text-slate-300
                    hover:bg-slate-50 dark:hover:bg-slate-800
                    transition-colors duration-150
                    flex items-center justify-center gap-2"
                >
                  <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round">
                    <rect x="9" y="2" width="6" height="12" rx="3" />
                    <path d="M5 10a7 7 0 0 0 14 0M12 18v4M8 22h8" />
                  </svg>
                  Record Again
                </button>
              )}

              <button
                onClick={resetState}
                className="w-full py-2.5 rounded-xl text-sm font-medium
                  border border-slate-300 dark:border-slate-700
                  text-slate-600 dark:text-slate-300
                  hover:bg-slate-50 dark:hover:bg-slate-800
                  transition-colors duration-150"
              >
                {source === 'upload' ? 'Upload New File' : 'Back to Upload'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
