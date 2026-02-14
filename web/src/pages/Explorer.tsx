import { useEffect, useState, useMemo } from 'react';
import { api } from '../api/client';
import type { TSNEData, TSNEPoint } from '../types';
import TSNEPlot from '../components/TSNEPlot';

/* ── Color map (hex) — must stay in sync with TSNEPlot ── */
const CLASS_COLORS: Record<string, string> = {
  bass:     '#38bdf8',
  brass:    '#fb7185',
  flute:    '#fbbf24',
  guitar:   '#fb923c',
  keyboard: '#94a3b8',
  mallet:   '#a78bfa',
  organ:    '#ef4444',
  reed:     '#2dd4bf',
  string:   '#34d399',
  vocal:    '#818cf8',
};

const FALLBACK_COLOR = '#94a3b8';

export default function Explorer() {
  /* ── Data ── */
  const [tsne, setTsne] = useState<TSNEData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  /* ── UI state ── */
  const [visibleClasses, setVisibleClasses] = useState<Set<string>>(new Set());
  const [showOnlyErrors, setShowOnlyErrors] = useState(false);
  const [selectedIdx, setSelectedIdx] = useState<number | null>(null);

  /* ── Fetch ── */
  useEffect(() => {
    api.tsne()
      .then((data) => {
        setTsne(data);
        setVisibleClasses(new Set(data.class_names));
      })
      .catch(() => setError(true))
      .finally(() => setLoading(false));
  }, []);

  /* ── Derived stats ── */
  const stats = useMemo(() => {
    if (!tsne) return null;
    const total = tsne.points.length;
    const errors = tsne.points.filter((p) => !p.correct).length;
    return { total, errors, accuracy: tsne.accuracy };
  }, [tsne]);

  /* ── Class-level stats for the filter panel ── */
  const classStats = useMemo(() => {
    if (!tsne) return new Map<string, { total: number; errors: number }>();
    const map = new Map<string, { total: number; errors: number }>();
    for (const cls of tsne.class_names) {
      const pts = tsne.points.filter((p) => p.class_name === cls);
      map.set(cls, {
        total: pts.length,
        errors: pts.filter((p) => !p.correct).length,
      });
    }
    return map;
  }, [tsne]);

  /* ── Selected point ── */
  const selectedPoint: TSNEPoint | null = useMemo(() => {
    if (selectedIdx === null || !tsne) return null;
    return tsne.points[selectedIdx] ?? null;
  }, [selectedIdx, tsne]);

  /* ── Filter helpers ── */
  function toggleClass(cls: string) {
    setVisibleClasses((prev) => {
      const next = new Set(prev);
      if (next.has(cls)) next.delete(cls);
      else next.add(cls);
      return next;
    });
  }

  function showAll() {
    if (!tsne) return;
    setVisibleClasses(new Set(tsne.class_names));
  }

  function showNone() {
    setVisibleClasses(new Set());
  }

  function isolateClass(cls: string) {
    setVisibleClasses(new Set([cls]));
  }

  return (
    <div className="max-w-6xl mx-auto px-8 py-10 space-y-6">

      {/* ── Header ── */}
      <header className="space-y-2">
        <h1 className="text-2xl font-bold tracking-tight text-slate-900 dark:text-white">
          Feature Explorer
        </h1>
        <p className="text-sm text-slate-600 dark:text-slate-400">
          t-SNE visualization of how the neural network organizes different sounds in
          its learned feature space. Points that cluster together sound similar to the network.
        </p>
      </header>

      {/* ── Loading ── */}
      {loading && (
        <div className="flex items-center gap-3 py-20 justify-center">
          <div className="w-5 h-5 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin" />
          <span className="text-sm text-slate-500">Loading t-SNE data...</span>
        </div>
      )}

      {/* ── Error ── */}
      {error && (
        <div className="card p-8 text-center space-y-2">
          <p className="text-sm text-slate-500 dark:text-slate-500">
            t-SNE data not available yet.
          </p>
          <p className="text-xs text-slate-500 dark:text-slate-600">
            Run <code className="font-mono bg-slate-100 dark:bg-slate-800 px-1.5 py-0.5 rounded">python generate_tsne.py</code> to generate the data.
          </p>
        </div>
      )}

      {/* ── Main content ── */}
      {tsne && stats && (
        <>
          {/* ── Summary cards ── */}
          <div className="grid grid-cols-3 gap-3">
            <div className="card p-4">
              <p className="text-[11px] font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-500">
                Test Samples
              </p>
              <p className="mt-1 text-2xl font-bold text-slate-900 dark:text-white">
                {stats.total}
              </p>
              <p className="text-xs text-slate-500 dark:text-slate-500 mt-0.5">
                {tsne.class_names.length} instrument classes
              </p>
            </div>

            <div className="card p-4">
              <p className="text-[11px] font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-500">
                Accuracy
              </p>
              <p className="mt-1 text-2xl font-bold text-slate-900 dark:text-white">
                {(stats.accuracy * 100).toFixed(1)}%
              </p>
              <p className="text-xs text-slate-500 dark:text-slate-500 mt-0.5">
                On held-out test set
              </p>
            </div>

            <div className="card p-4">
              <p className="text-[11px] font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-500">
                Misclassified
              </p>
              <p className="mt-1 text-2xl font-bold text-red-500">
                {stats.errors}
              </p>
              <p className="text-xs text-slate-500 dark:text-slate-500 mt-0.5">
                {((stats.errors / stats.total) * 100).toFixed(1)}% error rate
              </p>
            </div>
          </div>

          {/* ── Scatter + sidebar ── */}
          <div className="grid grid-cols-4 gap-4">

            {/* ── Scatter plot (3/4 width) ── */}
            <div className="col-span-3">
              <div className="card overflow-hidden">
                <div className="p-4 border-b border-slate-200 dark:border-slate-800 flex items-center justify-between">
                  <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-500">
                    t-SNE Projection
                  </h3>
                  <div className="flex items-center gap-3">
                    <span className="flex items-center gap-1.5 text-[10px] text-slate-500 dark:text-slate-500">
                      <span className="inline-block w-2 h-2 rounded-full bg-slate-400" />
                      correct
                    </span>
                    <span className="flex items-center gap-1.5 text-[10px] text-slate-500 dark:text-slate-500">
                      <span className="inline-block w-2 h-2 rotate-45 bg-red-400" />
                      incorrect
                    </span>
                    <span className="text-[10px] text-slate-500 dark:text-slate-600 font-mono ml-2">
                      {tsne.points.filter(p => visibleClasses.has(p.class_name) && (!showOnlyErrors || !p.correct)).length} pts
                    </span>
                  </div>
                </div>

                <div className="h-[520px] bg-slate-50 dark:bg-slate-800/30">
                  <TSNEPlot
                    points={tsne.points}
                    classNames={tsne.class_names}
                    visibleClasses={visibleClasses}
                    showOnlyErrors={showOnlyErrors}
                    selectedIdx={selectedIdx}
                    onSelect={setSelectedIdx}
                  />
                </div>
              </div>
            </div>

            {/* ── Sidebar: Filters ── */}
            <div className="col-span-1 space-y-4">

              {/* Class filters */}
              <div className="card p-4 space-y-3">
                <div className="flex items-center justify-between">
                  <h3 className="text-[11px] font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-500">
                    Classes
                  </h3>
                  <div className="flex gap-1">
                    <button
                      onClick={showAll}
                      className="text-[10px] text-indigo-500 hover:text-indigo-400 transition-colors"
                    >
                      All
                    </button>
                    <span className="text-[10px] text-slate-400">/</span>
                    <button
                      onClick={showNone}
                      className="text-[10px] text-indigo-500 hover:text-indigo-400 transition-colors"
                    >
                      None
                    </button>
                  </div>
                </div>

                <div className="space-y-0.5">
                  {tsne.class_names.map((cls) => {
                    const cs = classStats.get(cls);
                    const active = visibleClasses.has(cls);
                    return (
                      <div key={cls} className="flex items-center gap-2 group">
                        {/* Checkbox */}
                        <button
                          onClick={() => toggleClass(cls)}
                          className={`flex items-center gap-2 flex-1 px-2 py-1.5 rounded-md text-left
                            transition-colors duration-150
                            ${active
                              ? 'hover:bg-slate-50 dark:hover:bg-slate-800/50'
                              : 'opacity-40 hover:opacity-60'
                            }`}
                        >
                          <div
                            className="w-2.5 h-2.5 rounded-full flex-shrink-0 transition-opacity"
                            style={{ backgroundColor: CLASS_COLORS[cls] ?? FALLBACK_COLOR }}
                          />
                          <span className="text-xs text-slate-600 dark:text-slate-300 truncate flex-1">
                            {cls}
                          </span>
                          <span className="text-[10px] text-slate-400 dark:text-slate-600 font-mono">
                            {cs?.total ?? 0}
                          </span>
                        </button>

                        {/* Isolate button */}
                        <button
                          onClick={() => isolateClass(cls)}
                          className="opacity-0 group-hover:opacity-100 text-[9px] text-indigo-500
                            hover:text-indigo-400 transition-opacity px-1"
                          title={`Show only ${cls}`}
                        >
                          only
                        </button>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Show Errors toggle */}
              <div className="card p-4 space-y-3">
                <h3 className="text-[11px] font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-500">
                  Filters
                </h3>
                <button
                  onClick={() => setShowOnlyErrors((v) => !v)}
                  className={`w-full flex items-center justify-between px-3 py-2 rounded-lg text-xs
                    transition-colors duration-150
                    ${showOnlyErrors
                      ? 'bg-red-500/10 text-red-500 dark:text-red-400'
                      : 'bg-slate-50 dark:bg-slate-800/50 text-slate-600 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800'
                    }`}
                >
                  <span>Show Errors Only</span>
                  <span className={`font-mono text-[10px] ${showOnlyErrors ? 'text-red-400' : 'text-slate-400 dark:text-slate-600'}`}>
                    {stats.errors}
                  </span>
                </button>
              </div>

              {/* About */}
              <div className="card p-4 space-y-2">
                <h3 className="text-[11px] font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-500">
                  About
                </h3>
                <p className="text-xs text-slate-600 dark:text-slate-400 leading-relaxed">
                  t-SNE reduces the network's 64-dimensional hidden layer activations
                  into 2D. Points close together were encoded similarly by the neural network.
                </p>
                <p className="text-xs text-slate-500 dark:text-slate-500 leading-relaxed">
                  Click any point to see details. Diamond markers indicate misclassifications.
                </p>
              </div>
            </div>
          </div>

          {/* ── Detail panel (appears on click) ── */}
          {selectedPoint && (
            <div className="card overflow-hidden animate-fade-in">
              <div className="p-4 border-b border-slate-200 dark:border-slate-800 flex items-center justify-between">
                <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-500">
                  Sample Detail
                </h3>
                <button
                  onClick={() => setSelectedIdx(null)}
                  className="text-slate-400 hover:text-slate-600 dark:hover:text-slate-300 transition-colors"
                >
                  <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
                    <path d="M18 6L6 18M6 6l12 12" />
                  </svg>
                </button>
              </div>

              <div className="p-5 flex items-start gap-8">
                {/* Left: spectrogram placeholder + coordinates */}
                <div className="space-y-3">
                  <div className="w-32 h-32 rounded-lg bg-slate-100 dark:bg-slate-800/60 flex items-center justify-center">
                    <span className="text-[10px] text-slate-400 dark:text-slate-600 font-mono">
                      ({selectedPoint.x.toFixed(1)}, {selectedPoint.y.toFixed(1)})
                    </span>
                  </div>
                  <div className="text-center">
                    <span
                      className="inline-block w-3 h-3 rounded-full mr-1.5"
                      style={{ backgroundColor: CLASS_COLORS[selectedPoint.class_name] ?? FALLBACK_COLOR }}
                    />
                    <span className="text-xs text-slate-600 dark:text-slate-300 font-medium">
                      Sample #{selectedIdx}
                    </span>
                  </div>
                </div>

                {/* Right: labels + confidence */}
                <div className="flex-1 space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    {/* True label */}
                    <div>
                      <p className="text-[11px] font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-500 mb-1">
                        True Label
                      </p>
                      <p className="text-lg font-semibold text-slate-900 dark:text-white">
                        {selectedPoint.class_name}
                      </p>
                    </div>

                    {/* Predicted label */}
                    <div>
                      <p className="text-[11px] font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-500 mb-1">
                        Predicted
                      </p>
                      <div className="flex items-center gap-2">
                        <p className="text-lg font-semibold text-slate-900 dark:text-white">
                          {selectedPoint.predicted_label}
                        </p>
                        {selectedPoint.correct ? (
                          <span className="px-1.5 py-0.5 rounded text-[10px] font-medium bg-emerald-500/10 text-emerald-600 dark:text-emerald-400">
                            Correct
                          </span>
                        ) : (
                          <span className="px-1.5 py-0.5 rounded text-[10px] font-medium bg-red-500/10 text-red-500 dark:text-red-400">
                            Wrong
                          </span>
                        )}
                      </div>
                    </div>
                  </div>

                  {/* Confidence */}
                  <div>
                    <p className="text-[11px] font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-500 mb-2">
                      Confidence
                    </p>
                    <div className="flex items-center gap-3">
                      <div className="flex-1 h-2.5 rounded-full bg-slate-100 dark:bg-slate-800 overflow-hidden">
                        <div
                          className="h-full rounded-full transition-all duration-500"
                          style={{
                            width: `${selectedPoint.confidence * 100}%`,
                            backgroundColor: selectedPoint.correct
                              ? '#34d399' // emerald-400
                              : '#ef4444', // red-500
                          }}
                        />
                      </div>
                      <span className="text-sm font-mono font-semibold text-slate-900 dark:text-white w-14 text-right">
                        {(selectedPoint.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>

                  {/* Cluster context */}
                  {!selectedPoint.correct && (
                    <p className="text-xs text-slate-500 dark:text-slate-500 leading-relaxed">
                      The model confused <span className="font-medium text-slate-700 dark:text-slate-300">{selectedPoint.class_name}</span> for{' '}
                      <span className="font-medium text-slate-700 dark:text-slate-300">{selectedPoint.predicted_label}</span>.
                      Check if these classes cluster near each other in the scatter plot — overlapping clusters indicate
                      the network finds them acoustically similar.
                    </p>
                  )}
                </div>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}
