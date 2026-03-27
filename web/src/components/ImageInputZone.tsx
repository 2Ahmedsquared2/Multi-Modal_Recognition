import { useState, useRef, useCallback } from 'react';

interface ImageInputZoneProps {
  onFileSelected: (file: File) => void;
  /** Short description shown in the right status panel */
  statusHint?: string;
}

export default function ImageInputZone({
  onFileSelected,
  statusHint = 'Upload an image and the neural network will classify what it sees.',
}: ImageInputZoneProps) {
  const [dragOver, setDragOver] = useState(false);
  const [preview, setPreview] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback(
    (file: File) => {
      // Generate a preview URL
      const url = URL.createObjectURL(file);
      setPreview(url);
      onFileSelected(file);
    },
    [onFileSelected],
  );

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile],
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
                : 'hover:bg-warm-50 dark:hover:bg-warm-800/40'
              }`}
          >
            {/* Glow ring on drag */}
            {dragOver && (
              <div className="absolute inset-0 rounded-t-xl ring-2 ring-accent-light/50 ring-inset pointer-events-none" />
            )}

            <input
              ref={fileInputRef}
              type="file"
              accept="image/*,.jpg,.jpeg,.png,.webp,.bmp"
              className="hidden"
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) handleFile(file);
              }}
            />

            {/* Upload icon */}
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
                <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
                <circle cx="8.5" cy="8.5" r="1.5" />
                <path d="M21 15l-5-5L5 21" />
              </svg>
            </div>

            {/* Primary text */}
            <p className="text-sm font-semibold text-warm-700 dark:text-warm-300">
              Drop image file here
            </p>
            <p className="text-xs text-warm-500 dark:text-warm-500 mt-1">
              or click to browse
            </p>

            {/* File types */}
            <div className="flex items-center gap-1.5 mt-3">
              {['JPG', 'PNG', 'WEBP', 'BMP'].map((fmt) => (
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

          {/* Hint text */}
          <div className="px-8 py-5">
            <p className="text-center text-[11px] text-warm-400 dark:text-warm-600">
              The image will be resized to 64×64 for classification
            </p>
          </div>
        </div>
      </div>

      {/* ── Status / Preview panel (right) ── */}
      <div className="hidden md:flex md:col-span-2">
        <div className="card flex-1 flex flex-col items-center justify-center text-center p-8 gap-5">
          {preview ? (
            <>
              {/* Show thumbnail preview */}
              <div className="w-28 h-28 rounded-xl overflow-hidden ring-1 ring-warm-300 dark:ring-warm-700">
                <img
                  src={preview}
                  alt="Upload preview"
                  className="w-full h-full object-cover"
                />
              </div>
              <div className="space-y-1.5">
                <p className="text-xs font-semibold uppercase tracking-wider text-emerald-600 dark:text-emerald-400">
                  Image loaded
                </p>
                <p className="text-sm text-warm-500 dark:text-warm-500 leading-relaxed max-w-[200px]">
                  Classifying now…
                </p>
              </div>
            </>
          ) : (
            <>
              {/* Placeholder icon */}
              <div className="w-16 h-16 rounded-2xl bg-warm-50 dark:bg-warm-800/60 flex items-center justify-center">
                <svg
                  className="w-8 h-8 text-warm-400 dark:text-warm-700"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth={1.5}
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
                  <circle cx="8.5" cy="8.5" r="1.5" />
                  <path d="M21 15l-5-5L5 21" />
                </svg>
              </div>
              <div className="space-y-1.5">
                <p className="text-xs font-semibold uppercase tracking-wider text-warm-400 dark:text-warm-600">
                  Waiting for image
                </p>
                <p className="text-sm text-warm-500 dark:text-warm-500 leading-relaxed max-w-[200px]">
                  {statusHint}
                </p>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
