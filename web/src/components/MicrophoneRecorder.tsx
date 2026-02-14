import { useState, useRef, useCallback, useEffect } from 'react';
import { encodeWav } from '../utils/audioEncoder';

/* ─────────────────────────────── types ─────────────────────────────── */

type RecordingState =
  | 'idle'
  | 'requesting'
  | 'recording'
  | 'converting'
  | 'error';

interface MicrophoneRecorderProps {
  /** Called with a WAV File after the user finishes recording. */
  onRecordingComplete: (file: File) => void;
  onCancel: () => void;
}

/* ──────────────────────────── constants ─────────────────────────────── */

const MAX_DURATION = 30; // seconds — user can stop anytime
const MIN_DURATION = 1;  // must record at least 1 s

/* ═══════════════════════════════════════════════════════════════════════
   Component
   ═══════════════════════════════════════════════════════════════════════ */

export default function MicrophoneRecorder({
  onRecordingComplete,
  onCancel,
}: MicrophoneRecorderProps) {
  const [state, setState] = useState<RecordingState>('idle');
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const [audioLevel, setAudioLevel] = useState(0);

  /* refs ─────────────────────────────────────────────────────────────── */
  const streamRef = useRef<MediaStream | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const rafRef = useRef<number>(0);
  const timerRef = useRef<number>(0);
  const startRef = useRef<number>(0);
  const chunksRef = useRef<Blob[]>([]);

  /* cleanup ──────────────────────────────────────────────────────────── */
  const cleanup = useCallback(() => {
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    if (timerRef.current) clearInterval(timerRef.current);
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    recorderRef.current = null;
    analyserRef.current = null;
  }, []);

  useEffect(
    () => () => {
      cleanup();
      audioCtxRef.current?.close();
    },
    [cleanup],
  );

  /* level meter (requestAnimationFrame loop) ─────────────────────────── */
  const updateLevel = useCallback(() => {
    if (!analyserRef.current) return;
    const data = new Uint8Array(analyserRef.current.frequencyBinCount);
    analyserRef.current.getByteFrequencyData(data);
    const avg = data.reduce((s, v) => s + v, 0) / data.length;
    setAudioLevel(avg / 255);
    rafRef.current = requestAnimationFrame(updateLevel);
  }, []);

  /* convert recorded chunks → WAV File ───────────────────────────────── */
  const convertToWav = useCallback(
    async (chunks: Blob[], mimeType: string) => {
      setState('converting');
      setAudioLevel(0);

      try {
        const ctx = audioCtxRef.current ?? new AudioContext();
        audioCtxRef.current = ctx;

        // Decode browser's native format into PCM
        const recordedBlob = new Blob(chunks, { type: mimeType });
        const arrayBuf = await recordedBlob.arrayBuffer();
        const audioBuf = await ctx.decodeAudioData(arrayBuf);

        // Mix to mono
        const length = audioBuf.length;
        const mono = new Float32Array(length);
        const nCh = audioBuf.numberOfChannels;
        for (let ch = 0; ch < nCh; ch++) {
          const chData = audioBuf.getChannelData(ch);
          for (let i = 0; i < length; i++) {
            mono[i] += chData[i] / nCh;
          }
        }

        // Encode as WAV and wrap in a File
        const wavBlob = encodeWav(mono, audioBuf.sampleRate);
        const wavFile = new File([wavBlob], 'recording.wav', {
          type: 'audio/wav',
        });
        onRecordingComplete(wavFile);
      } catch (err) {
        setState('error');
        setErrorMsg(
          err instanceof Error ? err.message : 'Failed to process recording',
        );
      }
    },
    [onRecordingComplete],
  );

  /* start recording ──────────────────────────────────────────────────── */
  const startRecording = useCallback(async () => {
    if (!navigator.mediaDevices?.getUserMedia) {
      setState('error');
      setErrorMsg('Your browser does not support microphone access.');
      return;
    }
    if (typeof MediaRecorder === 'undefined') {
      setState('error');
      setErrorMsg(
        'Your browser does not support audio recording. Try Chrome or Firefox.',
      );
      return;
    }

    setState('requesting');
    setErrorMsg(null);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      const ctx = audioCtxRef.current ?? new AudioContext();
      audioCtxRef.current = ctx;

      // AnalyserNode for the live level meter
      const source = ctx.createMediaStreamSource(stream);
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 256;
      source.connect(analyser);
      analyserRef.current = analyser;
      updateLevel();

      // MediaRecorder to capture audio
      const recorder = new MediaRecorder(stream);
      recorderRef.current = recorder;
      chunksRef.current = [];

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      recorder.onstop = () => {
        if (rafRef.current) cancelAnimationFrame(rafRef.current);
        if (timerRef.current) clearInterval(timerRef.current);
        stream.getTracks().forEach((t) => t.stop());
        convertToWav(chunksRef.current, recorder.mimeType);
      };

      recorder.start();
      setState('recording');
      startRef.current = Date.now();
      setElapsed(0);

      // Timer — auto-stops at MAX_DURATION
      timerRef.current = window.setInterval(() => {
        const secs = (Date.now() - startRef.current) / 1000;
        setElapsed(Math.min(secs, MAX_DURATION));
        if (secs >= MAX_DURATION) {
          recorder.stop();
          clearInterval(timerRef.current);
        }
      }, 50);
    } catch (err: unknown) {
      cleanup();
      setState('error');
      const e = err as { name?: string; message?: string };
      if (e.name === 'NotAllowedError') {
        setErrorMsg(
          'Microphone access was denied. You can still upload audio files.',
        );
      } else if (e.name === 'NotFoundError') {
        setErrorMsg('No microphone found on this device.');
      } else {
        setErrorMsg(
          `Microphone not available: ${e.message || 'Unknown error'}`,
        );
      }
    }
  }, [cleanup, convertToWav, updateLevel]);

  /* manual stop ──────────────────────────────────────────────────────── */
  const stopRecording = useCallback(() => {
    if (recorderRef.current?.state === 'recording') {
      // Enforce minimum duration
      const secs = (Date.now() - startRef.current) / 1000;
      if (secs < MIN_DURATION) return; // ignore — too short
      recorderRef.current.stop();
    }
  }, []);

  /* helpers ───────────────────────────────────────────────────────────── */
  const canStop = elapsed >= MIN_DURATION;

  // Format mm:ss
  const fmt = (s: number) => {
    const m = Math.floor(s / 60);
    const sec = Math.floor(s % 60);
    return `${m}:${sec.toString().padStart(2, '0')}`;
  };

  /* ═══════════════════════════════════════════════════════════════════
     Render
     ═══════════════════════════════════════════════════════════════════ */

  return (
    <div className="space-y-4">
      {/* Back link */}
      <button
        onClick={onCancel}
        className="text-sm text-slate-500 dark:text-slate-400
          hover:text-slate-700 dark:hover:text-slate-200
          transition-colors flex items-center gap-1"
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
          <path d="M19 12H5M5 12l6-6M5 12l6 6" />
        </svg>
        Back to upload
      </button>

      {/* Main card */}
      <div className="card p-10 flex flex-col items-center text-center space-y-6">
        {/* ── Heading ── */}
        <div className="space-y-1">
          <h2 className="text-lg font-semibold text-slate-900 dark:text-white">
            Record from Microphone
          </h2>
          <p className="text-sm text-slate-500 dark:text-slate-400">
            {state === 'idle' &&
              `Record up to ${MAX_DURATION}s, then pick a 2-second clip to classify`}
            {state === 'requesting' && 'Waiting for microphone permission…'}
            {state === 'recording' && 'Press stop when you\'re done'}
            {state === 'converting' && 'Preparing your recording…'}
            {state === 'error' && ''}
          </p>
        </div>

        {/* ── IDLE ── */}
        {state === 'idle' && (
          <button
            onClick={startRecording}
            className="group relative w-24 h-24 rounded-full
              bg-indigo-500 hover:bg-indigo-600
              dark:bg-indigo-600 dark:hover:bg-indigo-500
              transition-colors duration-200
              flex items-center justify-center
              shadow-lg shadow-indigo-500/25"
          >
            <svg
              className="w-10 h-10 text-white"
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
          </button>
        )}

        {/* ── REQUESTING ── */}
        {state === 'requesting' && (
          <div
            className="w-24 h-24 rounded-full
              bg-slate-200 dark:bg-slate-700
              flex items-center justify-center
              animate-pulse"
          >
            <svg
              className="w-10 h-10 text-slate-400 dark:text-slate-500"
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
          </div>
        )}

        {/* ── RECORDING ── */}
        {state === 'recording' && (
          <div className="flex flex-col items-center space-y-5">
            {/* Pulsing ring + stop button */}
            <div className="relative flex items-center justify-center">
              <span className="absolute w-28 h-28 rounded-full bg-rose-500/20 animate-ping" />
              <span className="absolute w-24 h-24 rounded-full bg-rose-500/10" />
              <button
                onClick={stopRecording}
                disabled={!canStop}
                className={`relative z-10 w-20 h-20 rounded-full
                  transition-colors duration-150
                  flex items-center justify-center
                  ${canStop
                    ? 'bg-rose-500 hover:bg-rose-600 shadow-lg shadow-rose-500/30 cursor-pointer'
                    : 'bg-rose-400/60 cursor-not-allowed'
                  }`}
              >
                <div className="w-7 h-7 rounded-sm bg-white" />
              </button>
            </div>

            {/* Elapsed time (large) */}
            <p className="text-3xl font-mono font-semibold text-slate-900 dark:text-white tabular-nums">
              {fmt(elapsed)}
              <span className="text-base text-slate-400 dark:text-slate-500">
                {' '}/ {fmt(MAX_DURATION)}
              </span>
            </p>

            {/* Audio level meter */}
            <div className="w-72 space-y-1">
              <div className="flex justify-between text-[11px] text-slate-500 dark:text-slate-400">
                <span>Level</span>
                <span>{Math.round(audioLevel * 100)}%</span>
              </div>
              <div className="h-2 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
                <div
                  className="h-full rounded-full transition-all duration-75
                    bg-gradient-to-r from-emerald-400 to-emerald-500"
                  style={{ width: `${Math.max(audioLevel * 100, 2)}%` }}
                />
              </div>
            </div>

            {/* Time progress bar */}
            <div className="w-72">
              <div className="h-1 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
                <div
                  className="h-full rounded-full bg-indigo-500 transition-all duration-100 ease-linear"
                  style={{
                    width: `${(elapsed / MAX_DURATION) * 100}%`,
                  }}
                />
              </div>
            </div>

            {!canStop && (
              <p className="text-xs text-slate-400 dark:text-slate-500">
                Record at least {MIN_DURATION}s…
              </p>
            )}
          </div>
        )}

        {/* ── CONVERTING ── */}
        {state === 'converting' && (
          <div className="flex flex-col items-center gap-4 py-4">
            <div
              className="w-12 h-12 border-[3px] border-indigo-500
                border-t-transparent rounded-full animate-spin"
            />
            <p className="text-sm text-slate-500 dark:text-slate-400">
              Converting to WAV…
            </p>
          </div>
        )}

        {/* ── ERROR ── */}
        {state === 'error' && (
          <div className="flex flex-col items-center gap-4 max-w-sm">
            <div
              className="w-14 h-14 rounded-full
                bg-rose-50 dark:bg-rose-500/10
                flex items-center justify-center"
            >
              <svg
                className="w-7 h-7 text-rose-500"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth={1.5}
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <circle cx="12" cy="12" r="10" />
                <path d="M12 8v4M12 16h.01" />
              </svg>
            </div>

            <p className="text-sm text-rose-600 dark:text-rose-400">
              {errorMsg}
            </p>

            <div className="flex gap-3">
              <button
                onClick={() => {
                  setState('idle');
                  setErrorMsg(null);
                }}
                className="px-4 py-2 rounded-lg text-sm font-medium
                  bg-indigo-50 text-indigo-700
                  dark:bg-indigo-500/10 dark:text-indigo-400
                  hover:bg-indigo-100 dark:hover:bg-indigo-500/20
                  transition-colors"
              >
                Try Again
              </button>
              <button
                onClick={onCancel}
                className="px-4 py-2 rounded-lg text-sm font-medium
                  border border-slate-300 dark:border-slate-700
                  text-slate-600 dark:text-slate-300
                  hover:bg-slate-50 dark:hover:bg-slate-800
                  transition-colors"
              >
                Upload a file instead
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
