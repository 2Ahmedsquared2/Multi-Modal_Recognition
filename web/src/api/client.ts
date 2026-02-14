import type {
  HealthResponse,
  ModelInfo,
  ClassifyResult,
  ConfusionMatrix,
  ClassMetricsResponse,
  TrainingHistory,
  TSNEData,
  WhatIfResponse,
} from '../types';

const API_BASE = import.meta.env.VITE_API_URL || '';

async function request<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${endpoint}`, options);
  if (!res.ok) {
    // Try to extract the detail message from FastAPI's error response
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

// ── Simple in-memory cache for static data (model info, metrics, etc.) ──

const cache = new Map<string, unknown>();

async function cachedRequest<T>(endpoint: string): Promise<T> {
  if (cache.has(endpoint)) {
    return cache.get(endpoint) as T;
  }
  const data = await request<T>(endpoint);
  cache.set(endpoint, data);
  return data;
}

export const api = {
  health: () =>
    request<HealthResponse>('/api/health'),

  modelInfo: () =>
    cachedRequest<ModelInfo>('/api/model/info'),

  classify: (file: File) => {
    const form = new FormData();
    form.append('file', file);
    return request<ClassifyResult>('/api/classify', {
      method: 'POST',
      body: form,
    });
  },

  classifyLive: (blob: Blob) => {
    const form = new FormData();
    form.append('file', blob, 'recording.wav');
    return request<ClassifyResult>('/api/classify/live', {
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

  confusionMatrix: () =>
    cachedRequest<ConfusionMatrix>('/api/model/confusion-matrix'),

  classMetrics: () =>
    cachedRequest<ClassMetricsResponse>('/api/model/class-metrics'),

  trainingHistory: () =>
    cachedRequest<TrainingHistory>('/api/model/training-history'),

  tsne: () =>
    cachedRequest<TSNEData>('/api/model/tsne'),

  whatIf: (spectrogram: number[][]) =>
    request<WhatIfResponse>('/api/what-if', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ spectrogram }),
    }),
};
