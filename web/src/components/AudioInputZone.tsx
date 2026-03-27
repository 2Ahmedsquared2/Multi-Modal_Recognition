import { useState, useRef, useCallback } from 'react';

interface AudioInputZoneProps {
  onFileSelected: (file: File) => void;
  onRecordClick: () => void;
  /** Short description shown in the right status panel */
  statusHint?: string;
}

/* ── Animated waveform bars for the "waiting" panel ── */

function WaitingWaveform() {
  return (
    <div className="flex items-end justify-center gap-[3px] h-8">
      {[1, 2, 3, 4, 5, 6, 7].map((i) => (
        <div
          key={i}
          className="w-[3px] rounded-full bg-warm-400 dark:bg-warm-700"
          style={{
            animation: `wave 1.4s ease-in-out ${i * 0.1}s infinite`,
          }}
        />
      ))}
      <style>{`
        @keyframes wave {
          0%, 100% { height: 6px; opacity: 0.4; }
          50%      { height: 24px; opacity: 0.8; }
        }
      `}</style>
    </div>
  );
}

export default function AudioInputZone({
  onFileSelected,
  onRecordClick,
  statusHint = 'Upload a file or record audio, then select a 2-second segment to analyze.',
}: AudioInputZoneProps) {
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const file = e.dataTransfer.files[0];
      if (file) onFileSelected(file);
    },
    [onFileSelected],
  );

  return (
    <div className="grid grid-cols-1 md:grid-cols-5 gap-5">
      {/* ── Primary Input Zone (left) ── */}
      <div className="md:col-span-3">
        <div className="card overflow-hidden">
          {/* Drop zone */}
          <div
            onDragOver={(e) => {
              e.preventDefault();
              setDragOver(true);
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={onDrop}
            onClick={() => fileInputRef.current?.click()}
            className={`relative cursor-pointer px-8 pt-10 pb-8
              flex flex-col items-center text-center
              transition-all duration-300
              ${dragOver
                ? 'bg-accent-subtle/80 dark:bg-accent/5'
                : 'hover:bg-warm-150 dark:hover:bg-warm-700/40'
              }`}
          >
            {/* Glow ring on drag */}
            {dragOver && (
              <div className="absolute inset-0 rounded-t-xl ring-2 ring-accent-light/50 ring-inset pointer-events-none" />
            )}

            <input
              ref={fileInputRef}
              type="file"
              accept="audio/*,.wav,.mp3,.ogg,.flac"
              className="hidden"
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) onFileSelected(file);
              }}
            />

            {/* Upload icon — filled style */}
            <div
              className={`w-14 h-14 rounded-2xl flex items-center justify-center mb-4 transition-all duration-300
                ${dragOver
                  ? 'bg-accent-subtle dark:bg-accent/15 scale-110'
                  : 'bg-warm-200 dark:bg-warm-700 group-hover:scale-105'
                }`}
            >
              <svg
                className={`w-7 h-7 transition-colors duration-300 ${
                  dragOver
                    ? 'text-accent'
                    : 'text-warm-500 dark:text-warm-500'
                }`}
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

            {/* Primary text */}
            <p className="text-sm font-semibold text-warm-700 dark:text-warm-300">
              Drop audio file here
            </p>
            <p className="text-xs text-warm-500 dark:text-warm-500 mt-1">
              or click to browse
            </p>

            {/* File types */}
            <div className="flex items-center gap-1.5 mt-3">
              {['WAV', 'MP3', 'OGG', 'FLAC'].map((fmt) => (
                <span
                  key={fmt}
                  className="px-1.5 py-0.5 text-[10px] font-medium rounded
                    bg-warm-200 text-warm-500
                    dark:bg-warm-700 dark:text-warm-500"
                >
                  {fmt}
                </span>
              ))}
            </div>
          </div>

          {/* Divider with "or" */}
          <div className="flex items-center gap-3 px-8">
            <div className="flex-1 h-px bg-warm-300 dark:bg-warm-700" />
            <span className="text-[10px] font-medium uppercase tracking-wider text-warm-400 dark:text-warm-600">
              or
            </span>
            <div className="flex-1 h-px bg-warm-300 dark:bg-warm-700" />
          </div>

          {/* Record button — pill, centered */}
          <div className="px-8 pt-4 pb-6 flex justify-center">
            <button
              onClick={(e) => {
                e.stopPropagation();
                onRecordClick();
              }}
              className="inline-flex items-center gap-2 px-6 py-2.5 rounded-full text-sm font-medium
                bg-warm-200 text-warm-600
                dark:bg-warm-700 dark:text-warm-300
                hover:bg-warm-300 dark:hover:bg-warm-600
                transition-colors duration-150"
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

          {/* 2-second hint */}
          <div className="px-8 pb-5">
            <p className="text-center text-[11px] text-warm-400 dark:text-warm-600">
              You'll select a 2-second segment after uploading
            </p>
          </div>
        </div>
      </div>

      {/* ── Status / Preview panel (right) ── */}
      <div className="hidden md:flex md:col-span-2">
        <div className="card flex-1 flex flex-col items-center justify-center text-center p-8 gap-5">
          {/* Animated waveform */}
          <div
            className="w-16 h-16 rounded-2xl
              bg-warm-150 dark:bg-warm-700/60
              flex items-center justify-center"
          >
            <WaitingWaveform />
          </div>

          {/* Status text */}
          <div className="space-y-1.5">
            <p className="text-xs font-semibold uppercase tracking-wider text-warm-400 dark:text-warm-600">
              Waiting for audio
            </p>
            <p className="text-sm text-warm-500 dark:text-warm-500 leading-relaxed max-w-[200px]">
              {statusHint}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
