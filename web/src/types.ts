// API Response Types

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  num_classes?: number;
  classes?: string[];
}

export interface ModelInfo {
  architecture: {
    input_size: number;
    hidden_sizes: number[];
    num_classes: number;
    layers: Array<{
      type: string;
      input_size: number;
      output_size: number;
      activation: string;
      parameters: number;
    }>;
  };
  parameters: number;
  class_names: string[];
  spectrogram_shape: number[];
  sample_rate: number;
  audio_duration: number;
  test_accuracy?: number;
  macro_f1?: number;
}

export interface ClassifyResult {
  prediction: string;
  confidence: number;
  all_confidences: Record<string, number>;
  spectrogram: number[][];
  waveform: number[];
  waveform_summary: {
    duration: number;
    sample_rate: number;
    peak_amplitude: number;
  };
}

export interface ConfusionMatrix {
  matrix: number[][];
  class_names: string[];
}

export interface ClassMetric {
  precision: number;
  recall: number;
  f1_score: number;
  support: number;
}

export interface ClassMetricsResponse {
  per_class: Record<string, ClassMetric>;
  macro_f1?: number;
  test_accuracy?: number;
}

export interface TrainingHistory {
  train_loss: number[];
  val_loss: number[];
  train_accuracy: number[];
  val_accuracy: number[];
  epochs: number;
}

export interface TSNEPoint {
  x: number;
  y: number;
  class_name: string;
  class_idx: number;
}

export interface TSNEData {
  points: TSNEPoint[];
  class_names: string[];
}

