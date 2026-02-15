import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { api } from '../api/client';
import { useModel } from '../contexts/ModelContext';
import type { HealthResponse, ModelInfo } from '../types';

/* ── Pipeline stage data ── */

const pipelineStages = [
  {
    title: 'Audio Input',
    detail: 'Raw .wav file',
    mono: 'signal[t]',
    color: 'bg-sky-500',
    lightBg: 'bg-sky-50 dark:bg-sky-500/5',
  },
  {
    title: 'Spectrogram',
    detail: 'Mel-frequency',
    mono: '64×64 matrix',
    color: 'bg-violet-500',
    lightBg: 'bg-violet-50 dark:bg-violet-500/5',
  },
  {
    title: 'Neural Network',
    detail: '3-layer dense',
    mono: 'f(Wx + b)',
    color: 'bg-indigo-500',
    lightBg: 'bg-indigo-50 dark:bg-indigo-500/5',
  },
  {
    title: 'Classification',
    detail: '10 categories',
    mono: 'argmax(p)',
    color: 'bg-emerald-500',
    lightBg: 'bg-emerald-50 dark:bg-emerald-500/5',
  },
];

const features = [
  {
    path: '/classify',
    title: 'Classify Audio',
    description: 'Upload a sound or use your microphone to identify what the neural network hears.',
    accent: 'group-hover:text-sky-500',
  },
  {
    path: '/dashboard',
    title: 'Training Dashboard',
    description: 'Explore training curves, loss landscapes, and the confusion matrix interactively.',
    accent: 'group-hover:text-violet-500',
  },
  {
    path: '/explorer',
    title: 'Feature Explorer',
    description: 'Visualize how the network organizes sounds in high-dimensional feature space via t-SNE.',
    accent: 'group-hover:text-indigo-500',
  },
  {
    path: '/what-if',
    title: 'What-If Analysis',
    description: 'Modify spectrograms and observe how small changes shift the network\'s predictions.',
    accent: 'group-hover:text-emerald-500',
  },
];

/* ── Component ── */

export default function Home() {
  const { engine, engineLabel } = useModel();
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    api.health()
      .then(setHealth)
      .catch(() => setError(true));
  }, []);

  useEffect(() => {
    api.modelInfo(engine)
      .then(setModelInfo)
      .catch(() => {});
  }, [engine]);

  const isCustom = engine === 'custom';

  return (
    <div className="max-w-5xl mx-auto px-8 py-10 space-y-12">

      {/* ── Header ── */}
      <header className="space-y-3">
        <h1 className="text-3xl font-bold tracking-tight text-slate-900 dark:text-white">
          Acoustic Pattern Recognition Engine
        </h1>
        <p className="text-base text-slate-600 dark:text-slate-400 max-w-2xl leading-relaxed">
          {isCustom
            ? 'A fully hand-engineered neural network — zero frameworks, zero shortcuts. Raw audio is decomposed into spectrograms and fed through a forward pass, backpropagation loop, and weight update cycle built entirely from first principles in NumPy. Every gradient is computed, every matrix is multiplied, every parameter is tuned — by code written from the ground up.'
            : 'The same architecture, now powered by PyTorch. Compare how an industry-standard ML framework performs against the hand-built NumPy implementation on the same dataset and architecture.'
          }
        </p>
      </header>

      {/* ── Pipeline Visualization ── */}
      <section className="space-y-4">
        <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-500">
          Processing Pipeline
        </h2>
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
          {pipelineStages.map((stage, i) => (
            <div key={stage.title} className="relative">
              <div className={`card p-4 ${stage.lightBg} border-transparent`}>
                <div className="flex items-center gap-2 mb-2">
                  <div className={`w-2 h-2 rounded-full ${stage.color}`} />
                  <span className="text-xs font-medium text-slate-600 dark:text-slate-400">
                    Step {i + 1}
                  </span>
                </div>
                <p className="text-sm font-semibold text-slate-900 dark:text-white">
                  {stage.title}
                </p>
                <p className="text-xs text-slate-600 dark:text-slate-400 mt-0.5">
                  {stage.detail}
                </p>
                <code className="block mt-2 text-[11px] font-mono text-slate-500 dark:text-slate-500">
                  {stage.mono}
                </code>
              </div>
              {/* Arrow connector */}
              {i < pipelineStages.length - 1 && (
                <div className="hidden lg:block absolute -right-2.5 top-1/2 -translate-y-1/2 z-10">
                  <svg width="10" height="10" viewBox="0 0 10 10" className="text-slate-400 dark:text-slate-700">
                    <path d="M1 1l8 4-8 4" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                </div>
              )}
            </div>
          ))}
        </div>
      </section>

      {/* ── Stats Row ── */}
      <section className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        {[
          { label: 'Architecture', value: '3-Layer Dense', sub: 'ReLU + Softmax' },
          { label: 'Classes', value: health ? String(health.num_classes) : '10', sub: 'Instrument families' },
          {
            label: 'Implementation',
            value: isCustom ? 'Pure NumPy' : 'PyTorch',
            sub: isCustom ? 'No ML frameworks' : 'nn.Sequential',
          },
          {
            label: 'Test Accuracy',
            value: modelInfo?.test_accuracy != null
              ? `${(modelInfo.test_accuracy * 100).toFixed(1)}%`
              : '—',
            sub: modelInfo
              ? `${modelInfo.parameters.toLocaleString()} params`
              : 'Loading...',
          },
        ].map((stat) => (
          <div key={stat.label} className="card p-4">
            <p className="text-[11px] font-medium uppercase tracking-wider text-slate-500 dark:text-slate-500">
              {stat.label}
            </p>
            <p className="mt-1 text-lg font-semibold text-slate-900 dark:text-white">
              {stat.value}
            </p>
            <p className="text-xs text-slate-500 dark:text-slate-500 mt-0.5">
              {stat.sub}
            </p>
          </div>
        ))}
      </section>

      {/* ── Feature Cards ── */}
      <section className="space-y-4">
        <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-500">
          Explore
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {features.map((feat) => (
            <Link
              key={feat.path}
              to={feat.path}
              className="group card-hover p-5 flex flex-col justify-between"
            >
              <div>
                <h3 className={`text-sm font-semibold text-slate-900 dark:text-white transition-colors ${feat.accent}`}>
                  {feat.title}
                </h3>
                <p className="text-sm text-slate-600 dark:text-slate-400 mt-1.5 leading-relaxed">
                  {feat.description}
                </p>
              </div>
              <div className="mt-4 flex items-center text-xs font-medium text-slate-500 dark:text-slate-500 group-hover:text-indigo-500 dark:group-hover:text-indigo-400 transition-colors">
                Open
                <svg className="ml-1 w-3 h-3 transition-transform group-hover:translate-x-0.5" viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M2.5 6h7M6.5 3l3 3-3 3" />
                </svg>
              </div>
            </Link>
          ))}
        </div>
      </section>

      {/* ── Classes Preview ── */}
      {health && health.classes && health.classes.length > 0 && (
        <section className="space-y-4">
          <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-500">
            Recognized Sound Classes
          </h2>
          <div className="flex flex-wrap gap-2">
            {health.classes.map((cls) => (
              <span
                key={cls}
                className="px-3 py-1.5 text-xs font-medium rounded-lg
                  bg-slate-100 text-slate-600
                  dark:bg-slate-800 dark:text-slate-300"
              >
                {cls}
              </span>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}
