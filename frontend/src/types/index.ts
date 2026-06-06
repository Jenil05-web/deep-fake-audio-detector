// API Types from Python Backend

export interface BackendAnalyzeResponse {
  job_id: string;
  status: "done" | "error" | string;
}

export interface BackendResultResponse {
  job_id: string;
  filename: string;
  verdict: "REAL" | "FAKE" | string;
  confidence: number;      // 0.0 - 1.0 representation
  proba_cnn: number;       // 0.0 - 1.0 representation
  proba_lstm: number;      // 0.0 - 1.0 representation
  proba_bio: number;       // 0.0 - 1.0 representation
  duration_sec: number;
  inference_ms: number;
  status: string;
  error?: string;
  // Fallbacks for the prompt description schema
  file?: string;
  duration?: number;
  cnn_score?: number;
  lstm_score?: number;
  bio_score?: number;
  inference_time?: number;
}

// Normalized UI Model

export interface NormalizedResult {
  jobId: string;
  filename: string;
  verdict: "REAL" | "FAKE";
  confidence: number;      // 0 - 100 percentage representation
  cnnScore: number;        // 0 - 100 percentage representation
  lstmScore: number;       // 0 - 100 percentage representation
  bioScore: number;        // 0 - 100 percentage representation
  duration: number;        // in seconds
  inferenceTime: number;   // in milliseconds
  timestamp: string;       // when it was scanned
  rawJson: string;         // original raw JSON response string
}

// Scan History Model (Saved in LocalStorage) - matching full results for recall
export type ScanHistoryItem = NormalizedResult;
