import { useEffect, useState } from 'react';
import { api } from '../api/client';
import type { TSNEData } from '../types';

const CLASS_COLORS: Record<string, string> = {
  'bass': 'bg-sky-400',
  'brass': 'bg-rose-400',
  'flute': 'bg-amber-400',
  'guitar': 'bg-orange-400',
  'keyboard': 'bg-slate-400',
  'mallet': 'bg-zinc-400',
  'organ': 'bg-red-500',
  'reed': 'bg-violet-400',
  'string': 'bg-emerald-400',
  'vocal': 'bg-indigo-400',
};

function getColor(label: string): string {
  return CLASS_COLORS[label] || 'bg-slate-400';
}

export default function Explorer() {
  const [tsne, setTsne] = useState<TSNEData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);
  const [hoveredClass, setHoveredClass] = useState<string | null>(null);

  useEffect(() => {
    api.tsne()
      .then(setTsne)
      .catch(() => setError(true))
      .finally(() => setLoading(false));
  }, []);

  // Use class_names from the API data if available, otherwise fall back to static map
  const classLabels = tsne?.class_names ?? Object.keys(CLASS_COLORS);

  return (
    <div className="max-w-5xl mx-auto px-8 py-10 space-y-8">

      {/* Header */}
      <header className="space-y-2">
        <h1 className="text-2xl font-bold tracking-tight text-slate-900 dark:text-white">
          Feature Explorer
        </h1>
        <p className="text-sm text-slate-600 dark:text-slate-400">
          t-SNE visualization of how the neural network organizes different sounds in
          its learned feature space. Points that cluster together sound similar to the network.
        </p>
      </header>

      <div className="grid grid-cols-4 gap-6">

        {/* ── Main scatter area ── */}
        <div className="col-span-3">
          <div className="card overflow-hidden">
            <div className="p-4 border-b border-slate-200 dark:border-slate-800 flex items-center justify-between">
              <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-500">
                t-SNE Projection
              </h3>
              {tsne && (
                <span className="text-[11px] text-slate-500 dark:text-slate-500 font-mono">
                  {tsne.points.length} samples
                </span>
              )}
            </div>

            <div className="relative h-[500px] bg-slate-50 dark:bg-slate-800/30">
              {loading && (
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="w-5 h-5 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin" />
                </div>
              )}

              {error && (
                <div className="absolute inset-0 flex flex-col items-center justify-center text-center px-8">
                  <p className="text-sm text-slate-500 dark:text-slate-500">
                    t-SNE data not available yet.
                  </p>
                  <p className="text-xs text-slate-500 dark:text-slate-600 mt-1">
                    Generate it by running the t-SNE computation on the backend.
                  </p>
                </div>
              )}

              {tsne && tsne.points.length > 0 && (() => {
                const xs = tsne.points.map(p => p.x);
                const ys = tsne.points.map(p => p.y);
                const minX = Math.min(...xs), maxX = Math.max(...xs);
                const minY = Math.min(...ys), maxY = Math.max(...ys);
                const rangeX = maxX - minX || 1;
                const rangeY = maxY - minY || 1;
                const pad = 0.05;

                return (
                  <div className="absolute inset-4">
                    {tsne.points.map((pt, i) => {
                      const x = ((pt.x - minX) / rangeX) * (1 - 2 * pad) + pad;
                      const y = ((pt.y - minY) / rangeY) * (1 - 2 * pad) + pad;
                      const dimmed = hoveredClass !== null && hoveredClass !== pt.class_name;
                      return (
                        <div
                          key={i}
                          className={`absolute w-2 h-2 rounded-full transition-opacity duration-200
                            ${getColor(pt.class_name)}
                            ${dimmed ? 'opacity-10' : 'opacity-80 hover:opacity-100'}
                          `}
                          style={{
                            left: `${x * 100}%`,
                            top: `${y * 100}%`,
                            transform: 'translate(-50%, -50%)',
                          }}
                          title={`${pt.class_name} (${pt.x.toFixed(2)}, ${pt.y.toFixed(2)})`}
                        />
                      );
                    })}
                  </div>
                );
              })()}
            </div>
          </div>
        </div>

        {/* ── Legend / Controls ── */}
        <div className="col-span-1 space-y-4">
          <div className="card p-4 space-y-3">
            <h3 className="text-[11px] font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-500">
              Classes
            </h3>
            <div className="space-y-1.5">
              {classLabels.map((label) => (
                <button
                  key={label}
                  onMouseEnter={() => setHoveredClass(label)}
                  onMouseLeave={() => setHoveredClass(null)}
                  className={`w-full flex items-center gap-2 px-2 py-1.5 rounded-md text-left
                    transition-colors duration-150
                    ${hoveredClass === label
                      ? 'bg-slate-100 dark:bg-slate-800'
                      : 'hover:bg-slate-50 dark:hover:bg-slate-800/50'
                    }`}
                >
                  <div className={`w-2.5 h-2.5 rounded-full ${getColor(label)} flex-shrink-0`} />
                  <span className="text-xs text-slate-600 dark:text-slate-300 truncate">
                    {label.replace(/_/g, ' ')}
                  </span>
                </button>
              ))}
            </div>
          </div>

          <div className="card p-4 space-y-2">
            <h3 className="text-[11px] font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-500">
              About
            </h3>
            <p className="text-xs text-slate-600 dark:text-slate-400 leading-relaxed">
              t-SNE reduces the network's high-dimensional feature representations
              into 2D coordinates. Points that are close together were encoded similarly
              by the neural network.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
