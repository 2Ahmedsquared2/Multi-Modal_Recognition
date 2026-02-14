import type {
  HealthResponse,
  ModelInfo,
  ClassifyResult,
  ConfusionMatrix,
  ClassMetricsResponse,
  TrainingHistory,
  TSNEData,
  WhatIfResponse,
  ModelEngine,
} from '../types';

const API_BASE = import.meta.env.VITE_API_URL || '';

/** Append ?model=... to a URL if engine is specified */
function withEngine(endpoint: string, engine?: ModelEngine): string {
  if (!engine) return endpoint;
  const sep = endpoint.includes('?') ? '&' : '?';
  return `${endpoint}${sep}model=${engine}`;
}

async function request<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${endpoint}`, options);
  if (!res.ok) {
    let message = `API Error: ${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body.detail) message = body.detail;
    } catch {
      // body wasn't JSON — keep default message
    }
    throw new Error(message);
  }
  return res.json();
}

// ── Cache keyed by full URL (including engine query param) ──

const cache = new Map<string, unknown>();

async function cachedRequest<T>(endpoint: string): Promise<T> {
  if (cache.has(endpoint)) {
    return cache.get(endpoint) as T;
  }
  const data = await request<T>(endpoint);
  cache.set(endpoint, data);
  return data;
}

/** Clear cached data for a specific engine (called on engine switch) */
export function clearCache() {
  cache.clear();
}

export const api = {
  health: () =>
    request<HealthResponse>('/api/health'),

  modelInfo: (engine?: ModelEngine) =>
    cachedRequest<ModelInfo>(withEngine('/api/model/info', engine)),

  classify: (file: File, engine?: ModelEngine) => {
    const form = new FormData();
    form.append('file', file);
    return request<ClassifyResult>(withEngine('/api/classify', engine), {
      method: 'POST',
      body: form,
    });
  },

  classifyLive: (blob: Blob, engine?: ModelEngine) => {
    const form = new FormData();
    form.append('file', blob, 'recording.wav');
    return request<ClassifyResult>(withEngine('/api/classify/live', engine), {
      method: 'POST',
      body: form,
    });
  },

  spectrogram: (file: File) => {
    const form = new FormData();
    form.append('file', file);
    return request<{ spectrogram: number[][] }>('/api/spectrogram', {
      method: 'POST',
      body: form,
    });
  },

  confusionMatrix: (engine?: ModelEngine) =>
    cachedRequest<ConfusionMatrix>(withEngine('/api/model/confusion-matrix', engine)),

  classMetrics: (engine?: ModelEngine) =>
    cachedRequest<ClassMetricsResponse>(withEngine('/api/model/class-metrics', engine)),

  trainingHistory: (engine?: ModelEngine) =>
    cachedRequest<TrainingHistory>(withEngine('/api/model/training-history', engine)),

  tsne: (engine?: ModelEngine) =>
    cachedRequest<TSNEData>(withEngine('/api/model/tsne', engine)),

  whatIf: (spectrogram: number[][], engine?: ModelEngine) =>
    request<WhatIfResponse>(withEngine('/api/what-if', engine), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ spectrogram }),
    }),
};
