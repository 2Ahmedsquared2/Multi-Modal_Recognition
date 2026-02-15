import type {
  HealthResponse,
  ModelInfo,
  ClassifyResult,
  ConfusionMatrix,
  ClassMetricsResponse,
  TrainingHistory,
  TSNEData,
  WhatIfResponse,
  DatasetListResponse,
  ModelEngine,
} from '../types';

const API_BASE = import.meta.env.VITE_API_URL || '';

// ── Query-param helpers ──────────────────────────────────────────────────

/** Build query string with optional model + dataset params */
function withParams(
  endpoint: string,
  engine?: ModelEngine,
  dataset?: string,
): string {
  const params: string[] = [];
  if (engine) params.push(`model=${engine}`);
  if (dataset) params.push(`dataset=${encodeURIComponent(dataset)}`);
  if (params.length === 0) return endpoint;
  const sep = endpoint.includes('?') ? '&' : '?';
  return `${endpoint}${sep}${params.join('&')}`;
}

// ── Core fetch helpers ───────────────────────────────────────────────────

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

// ── Cache keyed by full URL (including query params) ─────────────────────

const cache = new Map<string, unknown>();

async function cachedRequest<T>(endpoint: string): Promise<T> {
  if (cache.has(endpoint)) {
    return cache.get(endpoint) as T;
  }
  const data = await request<T>(endpoint);
  cache.set(endpoint, data);
  return data;
}

/** Clear all cached data (called on engine or dataset switch) */
export function clearCache() {
  cache.clear();
}

// ── Public API ───────────────────────────────────────────────────────────

export const api = {
  health: () =>
    request<HealthResponse>('/api/health'),

  /** Fetch all registered datasets and their readiness status */
  datasets: () =>
    request<DatasetListResponse>('/api/datasets'),

  modelInfo: (engine?: ModelEngine, dataset?: string) =>
    cachedRequest<ModelInfo>(withParams('/api/model/info', engine, dataset)),

  classify: (file: File, engine?: ModelEngine, dataset?: string) => {
    const form = new FormData();
    form.append('file', file);
    return request<ClassifyResult>(withParams('/api/classify', engine, dataset), {
      method: 'POST',
      body: form,
    });
  },

  classifyLive: (blob: Blob, engine?: ModelEngine, dataset?: string) => {
    const form = new FormData();
    form.append('file', blob, 'recording.wav');
    return request<ClassifyResult>(withParams('/api/classify/live', engine, dataset), {
      method: 'POST',
      body: form,
    });
  },

  spectrogram: (file: File, dataset?: string) => {
    const form = new FormData();
    form.append('file', file);
    return request<{ spectrogram: number[][] }>(withParams('/api/spectrogram', undefined, dataset), {
      method: 'POST',
      body: form,
    });
  },

  confusionMatrix: (engine?: ModelEngine, dataset?: string) =>
    cachedRequest<ConfusionMatrix>(withParams('/api/model/confusion-matrix', engine, dataset)),

  classMetrics: (engine?: ModelEngine, dataset?: string) =>
    cachedRequest<ClassMetricsResponse>(withParams('/api/model/class-metrics', engine, dataset)),

  trainingHistory: (engine?: ModelEngine, dataset?: string) =>
    cachedRequest<TrainingHistory>(withParams('/api/model/training-history', engine, dataset)),

  tsne: (engine?: ModelEngine, dataset?: string) =>
    cachedRequest<TSNEData>(withParams('/api/model/tsne', engine, dataset)),

  whatIf: (spectrogram: number[][], engine?: ModelEngine, dataset?: string) =>
    request<WhatIfResponse>(withParams('/api/what-if', engine, dataset), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ spectrogram }),
    }),
};
