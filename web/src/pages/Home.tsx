import { useEffect, useState, useMemo } from 'react';
import { Link } from 'react-router-dom';
import { api } from '../api/client';
import { useModel } from '../contexts/ModelContext';
import type { HealthResponse, ModelInfo } from '../types';

const audioPipeline = [
  {
    title: 'Audio Input',
    detail: 'Raw waveform',
    mono: 'signal[t]',
    bg: 'bg-warm-150 dark:bg-warm-700/40',
  },
  {
    title: 'Spectrogram',
    detail: 'Mel-frequency',
    mono: '64×64 matrix',
    bg: 'bg-warm-200 dark:bg-warm-700/50',
  },
  {
    title: 'Neural Network',
    detail: '3-layer dense',
    mono: 'f(Wx + b)',
    bg: 'bg-warm-300/60 dark:bg-warm-700/60',
  },
  {
    title: 'Classification',
    detail: 'Probability output',
    mono: 'argmax(p)',
    bg: 'bg-warm-300/80 dark:bg-warm-700/70',
  },
];

const imagePipeline = [
  {
    title: 'Image Input',
    detail: 'JPG / PNG file',
    mono: 'img[h,w,c]',
    bg: 'bg-warm-150 dark:bg-warm-700/40',
  },
  {
    title: 'Preprocess',
    detail: 'Resize & normalize',
    mono: '64×64 matrix',
    bg: 'bg-warm-200 dark:bg-warm-700/50',
  },
  {
    title: 'Neural Network',
    detail: '3-layer dense',
    mono: 'f(Wx + b)',
    bg: 'bg-warm-300/60 dark:bg-warm-700/60',
  },
  {
    title: 'Classification',
    detail: 'N categories',
    mono: 'argmax(p)',
    bg: 'bg-warm-300/80 dark:bg-warm-700/70',
  },
];

const audioFeatures = [
  {
    path: '/classify',
    title: 'Classify Audio',
    description: 'Upload a sound or use your microphone — the network will identify what it hears.',
    accentHover: 'group-hover:text-accent',
  },
  {
    path: '/dashboard',
    title: 'Training Metrics',
    description: 'Loss curves, confusion matrices, and per-class performance at a glance.',
    accentHover: 'group-hover:text-accent',
  },
  {
    path: '/explorer',
    title: 'Feature Explorer',
    description: 'See how the network organizes sounds in high-dimensional space via t-SNE.',
    accentHover: 'group-hover:text-accent',
  },
  {
    path: '/what-if',
    title: 'What-If Lab',
    description: 'Edit spectrograms directly and watch how small changes shift predictions.',
    accentHover: 'group-hover:text-accent',
  },
];

const imageFeatures = [
  {
    path: '/classify',
    title: 'Classify Image',
    description: 'Upload an image and see what the network identifies in the visual pattern.',
    accentHover: 'group-hover:text-accent',
  },
  {
    path: '/dashboard',
    title: 'Training Metrics',
    description: 'Loss curves, confusion matrices, and per-class performance at a glance.',
    accentHover: 'group-hover:text-accent',
  },
  {
    path: '/explorer',
    title: 'Feature Explorer',
    description: 'Visualize how the network clusters images in high-dimensional feature space.',
    accentHover: 'group-hover:text-accent',
  },
  {
    path: '/what-if',
    title: 'What-If Lab',
    description: 'Paint on preprocessed inputs and observe how changes shift predictions.',
    accentHover: 'group-hover:text-accent',
  },
];

export default function Home() {
  const { engine, modality, dataset, activeDataset } = useModel();
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    api.health()
      .then(setHealth)
      .catch(() => {});
  }, []);

  useEffect(() => {
    let cancelled = false;
    api.modelInfo(engine, dataset)
      .then((info) => { if (!cancelled) setModelInfo(info); })
      .catch(() => { if (!cancelled) setError(true); });
    return () => { cancelled = true; };
  }, [engine, dataset]);

  const isCustom = engine === 'custom';
  const isAudio = modality === 'audio';
  const datasetLabel = activeDataset?.name ?? dataset;
  const numClasses = activeDataset?.num_classes ?? modelInfo?.class_names?.length ?? health?.num_classes ?? '—';
  const classNames = activeDataset?.class_names ?? modelInfo?.class_names ?? health?.classes ?? [];

  const pipelineStages = useMemo(
    () => (isAudio ? audioPipeline : imagePipeline),
    [isAudio],
  );

  const features = useMemo(
    () => (isAudio ? audioFeatures : imageFeatures),
    [isAudio],
  );

  return (
    <div className="max-w-5xl mx-auto px-4 sm:px-8 py-10 space-y-12">

      {/* Header */}
      <header className="stagger-1 space-y-3">
        <h1 className="font-display text-3xl font-bold tracking-tight text-warm-900 dark:text-warm-100">
          {isAudio ? 'Acoustic Pattern Recognition' : 'Visual Pattern Recognition'}
        </h1>
        <p className="text-base text-warm-600 dark:text-warm-400 max-w-2xl leading-relaxed">
          {isAudio
            ? (isCustom
                ? 'A neural network written entirely from scratch in NumPy. Raw audio becomes spectrograms, then passes through hand-coded forward propagation, backprop, and gradient descent — no frameworks involved.'
                : 'The same architecture powered by PyTorch. Compare how an industry-standard framework performs against the hand-built NumPy implementation on identical data.'
              )
            : (isCustom
                ? 'A neural network written entirely from scratch in NumPy for visual data. Images are preprocessed and classified through hand-coded layers — no frameworks involved.'
                : 'The same architecture powered by PyTorch. Compare framework performance against the hand-built implementation on identical image data.'
              )
          }
        </p>
        <div className="flex items-center gap-2 pt-1">
          <span className="text-xs font-medium px-2.5 py-1 rounded-full bg-accent-subtle text-accent dark:bg-accent/10 dark:text-accent-light">
            {isAudio ? 'Audio' : 'Image'}
          </span>
          <span className="text-xs font-medium px-2.5 py-1 rounded-full bg-warm-200 text-warm-600 dark:bg-warm-700 dark:text-warm-400">
            {datasetLabel}
          </span>
        </div>
      </header>

      {/* Pipeline */}
      <section className="stagger-2 space-y-4">
        <h2 className="text-[10px] font-semibold uppercase tracking-wider text-warm-500 dark:text-warm-500">
          Processing Pipeline
        </h2>
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
          {pipelineStages.map((stage, i) => (
            <div key={stage.title} className="relative">
              <div className={`card p-4 ${stage.bg} border-transparent`}>
                <div className="flex items-center gap-2 mb-2">
                  <div className="w-1.5 h-1.5 rounded-full bg-accent" />
                  <span className="text-[10px] font-medium text-warm-500 dark:text-warm-400">
                    Step {i + 1}
                  </span>
                </div>
                <p className="font-display text-sm font-semibold text-warm-900 dark:text-warm-100">
                  {stage.title}
                </p>
                <p className="text-xs text-warm-600 dark:text-warm-400 mt-0.5">
                  {stage.detail}
                </p>
                <code className="block mt-2 text-[11px] font-mono text-warm-500 dark:text-warm-500">
                  {stage.mono}
                </code>
              </div>
              {i < pipelineStages.length - 1 && (
                <div className="hidden lg:block absolute -right-2.5 top-1/2 -translate-y-1/2 z-10">
                  <svg width="10" height="10" viewBox="0 0 10 10" className="text-warm-400 dark:text-warm-600">
                    <path d="M1 1l8 4-8 4" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                </div>
              )}
            </div>
          ))}
        </div>
      </section>

      {/* Stats */}
      <section className="stagger-3 grid grid-cols-2 lg:grid-cols-4 gap-3">
        {[
          { label: 'Architecture', value: '3-Layer Dense', sub: 'ReLU + Softmax' },
          { label: 'Classes', value: String(numClasses), sub: datasetLabel },
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
            sub: modelInfo && modelInfo.parameters != null
              ? `${modelInfo.parameters.toLocaleString()} params`
              : error ? 'Model not loaded' : 'Loading…',
          },
        ].map((stat) => (
          <div key={stat.label} className="card p-4">
            <p className="text-[10px] font-medium uppercase tracking-wider text-warm-500 dark:text-warm-500">
              {stat.label}
            </p>
            <p className="mt-1 font-display text-lg font-semibold text-warm-900 dark:text-warm-100">
              {stat.value}
            </p>
            <p className="text-xs text-warm-500 dark:text-warm-500 mt-0.5">
              {stat.sub}
            </p>
          </div>
        ))}
      </section>

      {/* Features */}
      <section className="stagger-4 space-y-4">
        <h2 className="text-[10px] font-semibold uppercase tracking-wider text-warm-500 dark:text-warm-500">
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
                <h3 className={`font-display text-sm font-semibold text-warm-900 dark:text-warm-100 transition-colors ${feat.accentHover}`}>
                  {feat.title}
                </h3>
                <p className="text-sm text-warm-600 dark:text-warm-400 mt-1.5 leading-relaxed">
                  {feat.description}
                </p>
              </div>
              <div className="mt-4 flex items-center text-xs font-medium text-warm-500 dark:text-warm-500 group-hover:text-accent dark:group-hover:text-accent-light transition-colors duration-150">
                Open
                <svg className="ml-1 w-3 h-3 transition-transform duration-200 group-hover:translate-x-1" viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M2.5 6h7M6.5 3l3 3-3 3" />
                </svg>
              </div>
            </Link>
          ))}
        </div>
      </section>

      {/* Classes */}
      {classNames.length > 0 && (
        <section className="stagger-5 space-y-4">
          <h2 className="text-[10px] font-semibold uppercase tracking-wider text-warm-500 dark:text-warm-500">
            {isAudio ? 'Recognized Sound Classes' : 'Recognized Image Classes'}
          </h2>
          <div className="flex flex-wrap gap-2">
            {classNames.map((cls) => (
              <span
                key={cls}
                className="px-3 py-1.5 text-xs font-medium rounded-lg capitalize
                  bg-warm-200 text-warm-600
                  dark:bg-warm-700 dark:text-warm-300"
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
