import { createContext, useContext, useState, useCallback, useEffect, useMemo } from 'react';
import type { ReactNode } from 'react';
import type { ModelEngine, Modality, DatasetSummary } from '../types';
import { api, clearCache } from '../api/client';

// ── Context shape ────────────────────────────────────────────────────────

interface ModelContextValue {
  /* Engine */
  engine: ModelEngine;
  setEngine: (engine: ModelEngine) => void;
  engineLabel: string;

  /* Modality */
  modality: Modality;
  setModality: (m: Modality) => void;

  /* Dataset */
  dataset: string;
  setDataset: (key: string) => void;
  datasets: DatasetSummary[];
  /** Datasets filtered to the current modality */
  datasetsForModality: DatasetSummary[];
  /** The active DatasetSummary (convenience) */
  activeDataset: DatasetSummary | undefined;
  /** True while the initial dataset list is loading */
  datasetsLoading: boolean;
}

const ModelContext = createContext<ModelContextValue | null>(null);

// ── Static labels ────────────────────────────────────────────────────────

const ENGINE_LABELS: Record<ModelEngine, string> = {
  custom: 'From-Scratch',
  pytorch: 'PyTorch',
};

const DEFAULT_DATASET = 'audio/music';

// ── Provider ─────────────────────────────────────────────────────────────

export function ModelProvider({ children }: { children: ReactNode }) {
  const [engine, setEngineState] = useState<ModelEngine>('custom');
  const [modality, setModalityState] = useState<Modality>('audio');
  const [dataset, setDatasetState] = useState<string>(DEFAULT_DATASET);
  const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
  const [datasetsLoading, setDatasetsLoading] = useState(true);

  // ── Fetch dataset list on mount ──

  useEffect(() => {
    let cancelled = false;
    api.datasets()
      .then((res) => {
        if (cancelled) return;
        setDatasets(res.datasets);
      })
      .catch((err) => {
        console.warn('Failed to fetch datasets:', err);
      })
      .finally(() => {
        if (!cancelled) setDatasetsLoading(false);
      });
    return () => { cancelled = true; };
  }, []);

  // ── Derived: datasets for current modality ──

  const datasetsForModality = useMemo(
    () => datasets.filter((d) => d.modality === modality),
    [datasets, modality],
  );

  const activeDataset = useMemo(
    () => datasets.find((d) => d.key === dataset),
    [datasets, dataset],
  );

  // ── Setters (clear cache on switch) ──

  const setEngine = useCallback((e: ModelEngine) => {
    setEngineState(e);
    clearCache();
  }, []);

  const setDataset = useCallback((key: string) => {
    setDatasetState(key);
    clearCache();
    // Derive modality from the dataset key
    const mod = key.startsWith('image/') ? 'image' : 'audio';
    setModalityState(mod as Modality);
  }, []);

  const setModality = useCallback(
    (m: Modality) => {
      setModalityState(m);
      // Auto-select first ready dataset for new modality, or first dataset if none ready
      const forModality = datasets.filter((d) => d.modality === m);
      const ready = forModality.find((d) => d.ready);
      const fallback = forModality[0];
      const pick = ready || fallback;
      if (pick && pick.key !== dataset) {
        setDatasetState(pick.key);
        clearCache();
      }
    },
    [datasets, dataset],
  );

  return (
    <ModelContext.Provider
      value={{
        engine,
        setEngine,
        engineLabel: ENGINE_LABELS[engine],
        modality,
        setModality,
        dataset,
        setDataset,
        datasets,
        datasetsForModality,
        activeDataset,
        datasetsLoading,
      }}
    >
      {children}
    </ModelContext.Provider>
  );
}

// ── Hook ─────────────────────────────────────────────────────────────────

export function useModel(): ModelContextValue {
  const ctx = useContext(ModelContext);
  if (!ctx) {
    throw new Error('useModel must be used within a ModelProvider');
  }
  return ctx;
}
