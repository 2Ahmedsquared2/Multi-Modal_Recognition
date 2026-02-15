import { useState, useRef, useCallback, useEffect } from 'react';
import { api } from '../api/client';
import { useModel } from '../contexts/ModelContext';
import type { ClassifyResult } from '../types';
import AudioTrimmer from '../components/AudioTrimmer';
import MicrophoneRecorder from '../components/MicrophoneRecorder';
import AudioInputZone from '../components/AudioInputZone';
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
  const [fileName, setFileName] = useState<string | null>(null);
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [source, setSource] = useState<InputSource>('upload');
  const hasAnimatedRef = useRef(false);
  const [showAllProbs, setShowAllProbs] = useState(false);

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
  }, []);

  // Sorted confidences for the bar chart
  const sortedConfidences = result
    ? Object.entries(result.all_confidences).sort(([, a], [, b]) => b - a)
    : [];

  // Top 5 predictions
  const topConfidences = sortedConfidences.slice(0, 5);
  const displayedConfidences = showAllProbs ? sortedConfidences : topConfidences;

  // Color-code confidence levels
  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.7) return 'text-emerald-600 dark:text-emerald-400';
    if (confidence >= 0.4) return 'text-amber-600 dark:text-amber-400';
    return 'text-slate-600 dark:text-slate-400';
  };

  return (
    <div className="max-w-7xl mx-auto px-8 py-10 space-y-8">

      {/* Header */}
      <header className="space-y-2">
        <div className="flex items-center gap-3">
          <h1 className="text-2xl font-bold tracking-tight text-slate-900 dark:text-white">
            Classify Audio
          </h1>
          <span className={`text-xs font-semibold px-3 py-1 rounded-full
            ${engine === 'custom'
              ? 'bg-indigo-100 text-indigo-700 dark:bg-indigo-500/20 dark:text-indigo-300'
              : 'bg-orange-100 text-orange-700 dark:bg-orange-500/20 dark:text-orange-300'
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
        <AudioInputZone
          onFileSelected={handleFileSelected}
          onRecordClick={() => setPhase('recording')}
          statusHint="Upload a file or record from your mic — the neural network will analyze the spectrogram and classify the sound."
        />
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
        <div className="space-y-6 animate-fade-in-up">

          {/* Action buttons at top right */}
          <div className="flex justify-end gap-3">
            <button
              onClick={() => {
                setResult(null);
                setPhase('trim');
              }}
              className="px-4 py-2 rounded-lg text-sm font-medium
                border border-slate-300 dark:border-slate-700
                text-slate-600 dark:text-slate-300
                hover:bg-slate-50 dark:hover:bg-slate-800
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
                className="px-4 py-2 rounded-lg text-sm font-medium
                  border border-slate-300 dark:border-slate-700
                  text-slate-600 dark:text-slate-300
                  hover:bg-slate-50 dark:hover:bg-slate-800
                  transition-colors duration-150
                  flex items-center gap-2"
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
              className="px-4 py-2 rounded-lg text-sm font-medium
                bg-indigo-600 text-white
                hover:bg-indigo-700 dark:bg-indigo-500 dark:hover:bg-indigo-600
                transition-colors duration-150"
            >
              {source === 'upload' ? 'Upload New File' : 'Back to Upload'}
            </button>
          </div>

          {/* Main content grid */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            
            {/* Left side: Prediction + Visualizations */}
            <div className="lg:col-span-2 space-y-6">
              
              {/* Prediction card - more prominent */}
              <div className="card p-6 space-y-3">
                <p className="text-[11px] font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-500">
                  Prediction
                </p>
                <div className="flex items-baseline gap-3">
                  <span className={`text-4xl font-bold capitalize ${getConfidenceColor(result.confidence)}`}>
                    {result.prediction}
                  </span>
                  <span className={`text-2xl font-mono font-semibold ${getConfidenceColor(result.confidence)}`}>
                    {(result.confidence * 100).toFixed(1)}%
                  </span>
                </div>
                {fileName && (
                  <p className="text-xs text-slate-400 dark:text-slate-600 font-mono truncate pt-2">
                    {fileName}
                  </p>
                )}
              </div>

              {/* Waveform + Spectrogram - equal heights */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="card p-5 flex flex-col">
                  <WaveformDisplay
                    waveform={result.waveform}
                    duration={result.waveform_summary.duration}
                  />
                </div>
                <div className="card p-5 flex flex-col">
                  <SpectrogramDisplay
                    spectrogram={result.spectrogram}
                    duration={result.waveform_summary.duration}
                  />
                </div>
              </div>
            </div>

            {/* Right side: Top probabilities */}
            <div className="lg:col-span-1">
              <div className="card p-5 space-y-4 h-full">
                <div className="flex items-center justify-between">
                  <p className="text-[11px] font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-500">
                    {showAllProbs ? 'All Probabilities' : 'Top 5 Predictions'}
                  </p>
                  {sortedConfidences.length > 5 && (
                    <button
                      onClick={() => setShowAllProbs(!showAllProbs)}
                      className="text-xs text-indigo-600 dark:text-indigo-400 hover:text-indigo-700 dark:hover:text-indigo-300 font-medium transition-colors"
                    >
                      {showAllProbs ? 'Show Less' : 'Show All'}
                    </button>
                  )}
                </div>
                <div className="space-y-3">
                  {displayedConfidences.map(([cls, prob], idx) => {
                    const isTop = idx === 0;
                    const isSignificant = prob > 0.01;
                    return (
                      <div key={cls} className="space-y-1.5">
                        <div className="flex justify-between text-xs">
                          <span className={
                            isTop
                              ? 'font-semibold text-slate-900 dark:text-white capitalize'
                              : isSignificant
                              ? 'text-slate-600 dark:text-slate-400 capitalize'
                              : 'text-slate-400 dark:text-slate-600 capitalize'
                          }>
                            {cls}
                          </span>
                          <span className={`font-mono ${
                            isTop
                              ? 'text-indigo-600 dark:text-indigo-400 font-semibold'
                              : isSignificant
                              ? 'text-slate-500 dark:text-slate-500'
                              : 'text-slate-400 dark:text-slate-600'
                          }`}>
                            {(prob * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="h-2 bg-slate-100 dark:bg-slate-800 rounded-full overflow-hidden">
                          <div
                            className={`h-full rounded-full transition-all duration-700 ease-out ${
                              isTop
                                ? 'bg-indigo-500 dark:bg-indigo-400'
                                : isSignificant
                                ? 'bg-slate-400 dark:bg-slate-600'
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
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
