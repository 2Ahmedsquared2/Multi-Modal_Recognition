import { useState } from 'react';

const MODIFICATION_PRESETS = [
  {
    id: 'low-pass',
    label: 'Low-Pass Filter',
    description: 'Remove high frequencies',
    icon: '↘',
  },
  {
    id: 'high-pass',
    label: 'High-Pass Filter',
    description: 'Remove low frequencies',
    icon: '↗',
  },
  {
    id: 'noise',
    label: 'Add Noise',
    description: 'Gaussian noise injection',
    icon: '〰',
  },
  {
    id: 'shift',
    label: 'Time Shift',
    description: 'Shift audio temporally',
    icon: '→',
  },
  {
    id: 'scale',
    label: 'Amplitude Scale',
    description: 'Scale energy levels',
    icon: '↕',
  },
  {
    id: 'mask',
    label: 'Frequency Mask',
    description: 'Zero out frequency bands',
    icon: '▬',
  },
];

export default function WhatIf() {
  const [activePreset, setActivePreset] = useState<string | null>(null);
  const [intensity, setIntensity] = useState(50);

  return (
    <div className="max-w-5xl mx-auto px-8 py-10 space-y-8">

      {/* Header */}
      <header className="space-y-2">
        <div className="flex items-center gap-2">
          <h1 className="text-2xl font-bold tracking-tight text-slate-900 dark:text-white">
            What-If Analysis
          </h1>
          <span className="badge-indigo">Experimental</span>
        </div>
        <p className="text-sm text-slate-600 dark:text-slate-400">
          Modify a spectrogram and watch how the neural network's predictions change in real time.
          Understand which features matter most for each sound class.
        </p>
      </header>

      <div className="grid grid-cols-3 gap-6">

        {/* ── Left: Spectrogram Editor ── */}
        <div className="col-span-2 space-y-4">

          {/* Original vs Modified */}
          <div className="grid grid-cols-2 gap-3">
            <div className="card p-4 space-y-3">
              <h3 className="text-[11px] font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-500">
                Original Spectrogram
              </h3>
              <div className="aspect-square rounded-lg bg-slate-50 dark:bg-slate-800/50 flex items-center justify-center">
                <div className="text-center">
                  <svg className="w-10 h-10 mx-auto text-slate-400 dark:text-slate-600" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1} strokeLinecap="round" strokeLinejoin="round">
                    <rect x="2" y="2" width="20" height="20" rx="2" />
                    <path d="M2 12h20M7 2v20M12 2v20M17 2v20M2 7h20M2 17h20" opacity="0.3" />
                  </svg>
                  <p className="text-xs text-slate-500 dark:text-slate-500 mt-2">
                    Upload audio first
                  </p>
                </div>
              </div>
            </div>
            <div className="card p-4 space-y-3">
              <h3 className="text-[11px] font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-500">
                Modified Spectrogram
              </h3>
              <div className="aspect-square rounded-lg bg-slate-50 dark:bg-slate-800/50 flex items-center justify-center">
                <div className="text-center">
                  <svg className="w-10 h-10 mx-auto text-slate-400 dark:text-slate-600" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1} strokeLinecap="round" strokeLinejoin="round">
                    <rect x="2" y="2" width="20" height="20" rx="2" />
                    <path d="M2 12h20M7 2v20M12 2v20M17 2v20M2 7h20M2 17h20" opacity="0.3" />
                    <path d="M6 6l12 12M18 6L6 18" opacity="0.2" strokeWidth="2" />
                  </svg>
                  <p className="text-xs text-slate-500 dark:text-slate-500 mt-2">
                    Apply a modification
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Prediction Comparison */}
          <div className="card p-5 space-y-3">
            <h3 className="text-[11px] font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-500">
              Prediction Comparison
            </h3>
            <div className="grid grid-cols-2 gap-4">
              <div className="p-3 rounded-lg bg-slate-50 dark:bg-slate-800/30">
                <p className="text-[10px] uppercase tracking-wide text-slate-500 dark:text-slate-500">Original</p>
                <p className="text-sm font-medium text-slate-600 dark:text-slate-400 mt-1">—</p>
              </div>
              <div className="p-3 rounded-lg bg-slate-50 dark:bg-slate-800/30">
                <p className="text-[10px] uppercase tracking-wide text-slate-500 dark:text-slate-500">Modified</p>
                <p className="text-sm font-medium text-slate-600 dark:text-slate-400 mt-1">—</p>
              </div>
            </div>
          </div>
        </div>

        {/* ── Right: Modification Controls ── */}
        <div className="col-span-1 space-y-4">

          {/* Preset buttons */}
          <div className="card p-4 space-y-3">
            <h3 className="text-[11px] font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-500">
              Modifications
            </h3>
            <div className="space-y-1.5">
              {MODIFICATION_PRESETS.map((preset) => (
                <button
                  key={preset.id}
                  onClick={() => setActivePreset(activePreset === preset.id ? null : preset.id)}
                  className={`w-full text-left px-3 py-2.5 rounded-lg text-sm
                    transition-colors duration-150 flex items-center gap-3
                    ${activePreset === preset.id
                      ? 'bg-indigo-50 text-indigo-600 dark:bg-indigo-500/10 dark:text-indigo-400'
                      : 'text-slate-600 dark:text-slate-400 hover:bg-slate-50 dark:hover:bg-slate-800/50'
                    }`}
                >
                  <span className="text-base leading-none w-5 text-center">{preset.icon}</span>
                  <div className="min-w-0">
                    <p className="font-medium text-xs">{preset.label}</p>
                    <p className="text-[10px] text-slate-500 dark:text-slate-500 mt-0.5">
                      {preset.description}
                    </p>
                  </div>
                </button>
              ))}
            </div>
          </div>

          {/* Intensity slider */}
          <div className="card p-4 space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="text-[11px] font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-500">
                Intensity
              </h3>
              <span className="text-xs font-mono text-slate-500 dark:text-slate-500">
                {intensity}%
              </span>
            </div>
            <input
              type="range"
              min={0}
              max={100}
              value={intensity}
              onChange={(e) => setIntensity(Number(e.target.value))}
              className="w-full h-1.5 bg-slate-200 dark:bg-slate-700 rounded-full appearance-none
                [&::-webkit-slider-thumb]:appearance-none
                [&::-webkit-slider-thumb]:w-3.5
                [&::-webkit-slider-thumb]:h-3.5
                [&::-webkit-slider-thumb]:rounded-full
                [&::-webkit-slider-thumb]:bg-indigo-500
                [&::-webkit-slider-thumb]:cursor-pointer
                [&::-webkit-slider-thumb]:transition-transform
                [&::-webkit-slider-thumb]:hover:scale-110"
            />
          </div>

          {/* Apply button */}
          <button
            disabled={!activePreset}
            className="w-full py-2.5 rounded-xl text-sm font-medium
              bg-indigo-500 text-white
              hover:bg-indigo-600
              disabled:opacity-40 disabled:cursor-not-allowed
              transition-colors duration-150"
          >
            Apply Modification
          </button>
        </div>
      </div>
    </div>
  );
}
