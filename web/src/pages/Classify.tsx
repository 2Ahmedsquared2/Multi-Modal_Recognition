import { useState, useRef, useCallback, useEffect } from 'react';
import { api } from '../api/client';
import { useModel } from '../contexts/ModelContext';
import type { ClassifyResult } from '../types';

/* ── Audio-specific imports ── */
import AudioTrimmer from '../components/AudioTrimmer';
import MicrophoneRecorder from '../components/MicrophoneRecorder';
import AudioInputZone from '../components/AudioInputZone';
import PipelineAnimation from '../components/PipelineAnimation';
import WaveformDisplay from '../components/WaveformDisplay';
import SpectrogramDisplay from '../components/SpectrogramDisplay';

/* ── Image-specific imports ── */
import ImageInputZone from '../components/ImageInputZone';
import ImageDisplay from '../components/ImageDisplay';

type Phase = 'upload' | 'trim' | 'recording' | 'pipeline' | 'results';
type InputSource = 'upload' | 'mic';

export default function Classify() {
  const { engine, engineLabel, modality, dataset, activeDataset } = useModel();
  const [phase, setPhase] = useState<Phase>('upload');
  const [result, setResult] = useState<ClassifyResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [imagePreviewUrl, setImagePreviewUrl] = useState<string | null>(null);
  const [source, setSource] = useState<InputSource>('upload');
  const hasAnimatedRef = useRef(false);
  const [showAllProbs, setShowAllProbs] = useState(false);

  // Reset state when modality or dataset changes
  useEffect(() => {
    resetState();
  }, [modality, dataset]);

  // Animation: bars start at 0 and grow to real values after mount
  const [animateBars, setAnimateBars] = useState(false);
  useEffect(() => {
    if (result) {
      const id = requestAnimationFrame(() => setAnimateBars(true));
      return () => cancelAnimationFrame(id);
    }
    setAnimateBars(false);
  }, [result]);

  // ──────────────────────────────────────────────────────────────────────
  //  Audio flow handlers
  // ──────────────────────────────────────────────────────────────────────

  /** Phase 1a: User picks an audio file */
  const handleAudioFileSelected = useCallback((file: File) => {
    setError(null);
    setResult(null);
    setFileName(file.name);
    setAudioFile(file);
    setSource('upload');
    setPhase('trim');
  }, []);

  /** Phase 2: User clicks "Classify" from trimmer (sends a WAV blob) */
  const handleClassifyBlob = useCallback(async (blob: Blob) => {
    setError(null);
    setResult(null);
    setLoading(true);
    try {
      const wavFile = new File([blob], 'segment.wav', { type: 'audio/wav' });
      const res = await api.classify(wavFile, engine, dataset);
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
  }, [engine, dataset]);

  /** Phase 1b: Mic recording finished → trimmer */
  const handleMicRecording = useCallback((file: File) => {
    setError(null);
    setResult(null);
    setFileName('Microphone Recording');
    setAudioFile(file);
    setSource('mic');
    setPhase('trim');
  }, []);

  /** Pipeline animation finished */
  const handlePipelineComplete = useCallback(() => {
    hasAnimatedRef.current = true;
    setPhase('results');
  }, []);

  // ──────────────────────────────────────────────────────────────────────
  //  Image flow handlers
  // ──────────────────────────────────────────────────────────────────────

  /** User picks an image file → classify immediately */
  const handleImageFileSelected = useCallback(async (file: File) => {
    setError(null);
    setResult(null);
    setFileName(file.name);
    setSource('upload');
    setImagePreviewUrl(URL.createObjectURL(file));
    setLoading(true);

    try {
      const res = await api.classify(file, engine, dataset);
      setResult(res);
      setPhase('results');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Classification failed');
      setPhase('upload');
    } finally {
      setLoading(false);
    }
  }, [engine, dataset]);

  // ──────────────────────────────────────────────────────────────────────
  //  Shared
  // ──────────────────────────────────────────────────────────────────────

  const resetState = useCallback(() => {
    setPhase('upload');
    setResult(null);
    setError(null);
    setFileName(null);
    setAudioFile(null);
    setSource('upload');
    setAnimateBars(false);
    if (imagePreviewUrl) URL.revokeObjectURL(imagePreviewUrl);
    setImagePreviewUrl(null);
  }, [imagePreviewUrl]);

  // Sorted confidences for the bar chart
  const sortedConfidences = result
    ? Object.entries(result.all_confidences).sort(([, a], [, b]) => b - a)
    : [];

  const topConfidences = sortedConfidences.slice(0, 5);
  const displayedConfidences = showAllProbs ? sortedConfidences : topConfidences;

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.7) return 'text-success dark:text-success';
    if (confidence >= 0.4) return 'text-accent dark:text-accent-light';
    return 'text-warm-600 dark:text-warm-400';
  };

  const isAudio = modality === 'audio';
  const datasetLabel = activeDataset?.name ?? dataset;

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-8 py-10 space-y-8">

      {/* Header */}
      <header className="stagger-1 space-y-2">
        <div className="flex items-center gap-3">
          <h1 className="font-display text-2xl font-bold tracking-tight text-warm-900 dark:text-warm-100">
            {isAudio ? 'Classify Audio' : 'Classify Image'}
          </h1>
          <span className={`text-xs font-semibold px-3 py-1 rounded-full
            ${engine === 'custom'
              ? 'bg-accent-subtle text-accent dark:bg-accent/10 dark:text-accent-light'
              : 'bg-warm-200 text-warm-600 dark:bg-warm-700 dark:text-warm-400'
            }`}
          >
            {engineLabel}
          </span>
          <span className="text-xs font-medium px-3 py-1 rounded-full bg-warm-200 text-warm-600 dark:bg-warm-700 dark:text-warm-400">
            {datasetLabel}
          </span>
        </div>
        <p className="text-sm text-warm-600 dark:text-warm-400">
          {isAudio
            ? `Upload an audio file or record from your microphone. The ${engineLabel} neural network will analyze the spectrogram and predict what sound it is.`
            : `Upload an image and the ${engineLabel} neural network will preprocess, analyze, and classify it.`}
        </p>
      </header>

      {/* ═══════════════════════════════════════════════════════════════
          AUDIO FLOW
          ═══════════════════════════════════════════════════════════════ */}
      {isAudio && (
        <>
          {/* Phase 1: Upload */}
          {phase === 'upload' && (
            <AudioInputZone
              onFileSelected={handleAudioFileSelected}
              onRecordClick={() => setPhase('recording')}
              statusHint="Upload a file or record from your mic — the neural network will analyze the spectrogram and classify the sound."
            />
          )}

          {/* Phase 2: Trim / Preview */}
          {phase === 'trim' && audioFile && (
            <div className="space-y-4">
              <AudioTrimmer
                file={audioFile}
                clipDuration={2}
                onClassify={handleClassifyBlob}
                onCancel={resetState}
              />

              {loading && (
                <div className="card p-6 flex items-center gap-3 animate-fade-in">
                  <div className="w-5 h-5 border-2 border-accent border-t-transparent rounded-full animate-spin" />
                  <div>
                    <p className="text-sm font-medium text-warm-700 dark:text-warm-300">
                      Analyzing audio…
                    </p>
                    <p className="text-xs text-warm-500 dark:text-warm-500">
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

          {/* Phase 2b: Microphone Recording */}
          {phase === 'recording' && (
            <MicrophoneRecorder onRecordingComplete={handleMicRecording} onCancel={resetState} />
          )}

          {/* Phase 2c: Pipeline Animation (first classify only) */}
          {phase === 'pipeline' && result && (
            <PipelineAnimation
              waveform={result.waveform ?? []}
              spectrogram={result.spectrogram}
              prediction={result.prediction}
              confidence={result.confidence}
              onComplete={handlePipelineComplete}
            />
          )}
        </>
      )}

      {/* ═══════════════════════════════════════════════════════════════
          IMAGE FLOW
          ═══════════════════════════════════════════════════════════════ */}
      {!isAudio && (
        <>
          {/* Upload */}
          {phase === 'upload' && !loading && (
            <ImageInputZone
              onFileSelected={handleImageFileSelected}
              statusHint={`Upload an image for ${datasetLabel} classification.`}
            />
          )}

          {/* Loading overlay */}
          {loading && (
            <div className="card p-8 flex flex-col items-center gap-4 animate-fade-in">
              <div className="w-6 h-6 border-2 border-accent border-t-transparent rounded-full animate-spin" />
              <div className="text-center">
                <p className="text-sm font-medium text-warm-700 dark:text-warm-300">
                  Analyzing image…
                </p>
                <p className="text-xs text-warm-500 dark:text-warm-500 mt-1">
                  Resizing → Normalizing → Running inference
                </p>
              </div>
              {imagePreviewUrl && (
                <img
                  src={imagePreviewUrl}
                  alt="Processing"
                  className="w-24 h-24 object-cover rounded-lg ring-1 ring-warm-300 dark:ring-warm-700 opacity-60"
                />
              )}
            </div>
          )}

          {/* Error */}
          {error && phase === 'upload' && (
            <div className="card p-4 border-rose-200 dark:border-rose-500/20 bg-rose-50 dark:bg-rose-500/5 animate-fade-in">
              <p className="text-sm text-rose-600 dark:text-rose-400">{error}</p>
            </div>
          )}
        </>
      )}

      {/* ═══════════════════════════════════════════════════════════════
          RESULTS (shared layout, modality-aware visualization)
          ═══════════════════════════════════════════════════════════════ */}
      {phase === 'results' && result && (
        <div className="space-y-6 stagger-2">

          {/* Action buttons */}
          <div className="flex justify-end gap-3">
            {isAudio && (
              <button
                onClick={() => {
                  setResult(null);
                  setPhase('trim');
                }}
                className="px-4 py-2 rounded-lg text-sm font-medium
                  border border-warm-300 dark:border-warm-700
                  text-warm-600 dark:text-warm-300
                  hover:bg-warm-150 dark:hover:bg-warm-700
                  transition-colors duration-150"
              >
                Try Different Segment
              </button>
            )}

            {isAudio && source === 'mic' && (
              <button
                onClick={() => {
                  setResult(null);
                  setAudioFile(null);
                  setPhase('recording');
                }}
                className="px-4 py-2 rounded-lg text-sm font-medium
                  border border-warm-300 dark:border-warm-700
                  text-warm-600 dark:text-warm-300
                  hover:bg-warm-150 dark:hover:bg-warm-700
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
                bg-warm-900 text-warm-50
                hover:bg-warm-800 dark:bg-warm-700 dark:hover:bg-warm-600
                transition-all duration-150 active:scale-[0.98]"
            >
              {isAudio
                ? (source === 'upload' ? 'Upload New File' : 'Back to Upload')
                : 'Upload New Image'}
            </button>
          </div>

          {/* Main content grid */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

            {/* Left side: Prediction + Visualizations */}
            <div className="lg:col-span-2 space-y-6">

              {/* Prediction card */}
              <div className="card p-6 space-y-3">
                <p className="text-[11px] font-semibold uppercase tracking-wider text-warm-500 dark:text-warm-500">
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
                  <p className="text-xs text-warm-400 dark:text-warm-600 font-mono truncate pt-2">
                    {fileName}
                  </p>
                )}
              </div>

              {/* Modality-specific visualization */}
              {isAudio && result.waveform_summary ? (
                /* Audio: Waveform + Spectrogram side-by-side */
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="card p-5 flex flex-col">
                    <WaveformDisplay
                      waveform={result.waveform ?? []}
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
              ) : (
                /* Image: Original + Preprocessed side-by-side */
                <ImageDisplay
                  originalSrc={imagePreviewUrl}
                  preprocessed={result.spectrogram}
                />
              )}
            </div>

            {/* Right side: Top probabilities */}
            <div className="lg:col-span-1">
              <div className="card p-5 space-y-4 h-full">
                <div className="flex items-center justify-between">
                  <p className="text-[11px] font-semibold uppercase tracking-wider text-warm-500 dark:text-warm-500">
                    {showAllProbs ? 'All Probabilities' : 'Top 5 Predictions'}
                  </p>
                  {sortedConfidences.length > 5 && (
                    <button
                      onClick={() => setShowAllProbs(!showAllProbs)}
                      className="text-xs text-accent dark:text-accent-light hover:text-accent-dark dark:hover:text-accent-light font-medium transition-colors"
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
                              ? 'font-semibold text-warm-900 dark:text-warm-100 capitalize'
                              : isSignificant
                              ? 'text-warm-600 dark:text-warm-400 capitalize'
                              : 'text-warm-400 dark:text-warm-600 capitalize'
                          }>
                            {cls}
                          </span>
                          <span className={`font-mono ${
                            isTop
                              ? 'text-accent dark:text-accent-light font-semibold'
                              : isSignificant
                              ? 'text-warm-500 dark:text-warm-500'
                              : 'text-warm-400 dark:text-warm-600'
                          }`}>
                            {(prob * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="h-2 bg-warm-200 dark:bg-warm-700 rounded-full overflow-hidden">
                          <div
                            className={`h-full rounded-full transition-all duration-500 ease-out ${
                              isTop
                                ? 'bg-accent dark:bg-accent-light'
                                : isSignificant
                                ? 'bg-warm-500 dark:bg-warm-600'
                                : 'bg-warm-400 dark:bg-warm-700'
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
