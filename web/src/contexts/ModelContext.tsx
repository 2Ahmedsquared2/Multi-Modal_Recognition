import { createContext, useContext, useState, useCallback } from 'react';
import type { ReactNode } from 'react';
import type { ModelEngine } from '../types';

interface ModelContextValue {
  engine: ModelEngine;
  setEngine: (engine: ModelEngine) => void;
  /** Display label for the current engine */
  engineLabel: string;
}

const ModelContext = createContext<ModelContextValue | null>(null);

const ENGINE_LABELS: Record<ModelEngine, string> = {
  custom: 'From-Scratch',
  pytorch: 'PyTorch',
};

export function ModelProvider({ children }: { children: ReactNode }) {
  const [engine, setEngineState] = useState<ModelEngine>('custom');

  const setEngine = useCallback((e: ModelEngine) => {
    setEngineState(e);
  }, []);

  return (
    <ModelContext.Provider
      value={{
        engine,
        setEngine,
        engineLabel: ENGINE_LABELS[engine],
      }}
    >
      {children}
    </ModelContext.Provider>
  );
}

export function useModel(): ModelContextValue {
  const ctx = useContext(ModelContext);
  if (!ctx) {
    throw new Error('useModel must be used within a ModelProvider');
  }
  return ctx;
}
