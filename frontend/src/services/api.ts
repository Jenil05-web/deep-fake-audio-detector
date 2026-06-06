import axios from 'axios';
import type { 
  BackendAnalyzeResponse, 
  BackendResultResponse, 
  NormalizedResult 
} from '../types';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const client = axios.create({
  baseURL: API_URL,
  timeout: 30000, // 30 seconds
});

/**
 * Health Check API
 * Verifies if the backend is online and the model is fully loaded.
 */
export async function checkHealth(): Promise<{ online: boolean; modelLoaded: boolean }> {
  try {
    const response = await client.get('/health', { timeout: 3000 });
    return {
      online: true,
      modelLoaded: !!response.data.model_loaded,
    };
  } catch (error) {
    console.error("Health check failed:", error);
    return {
      online: false,
      modelLoaded: false,
    };
  }
}

/**
 * Normalizes backend responses from multiple possible structures
 * (supporting both the prompt specification and the actual FastAPI return object).
 */
export function normalizeResponse(data: BackendResultResponse): NormalizedResult {
  // 1. Filename mapping
  const filename = data.filename || data.file || 'unknown_audio_file.wav';
  
  // 2. Verdict mapping
  const verdict = data.verdict?.toUpperCase() === 'REAL' ? 'REAL' : 'FAKE';

  // 3. Confidence mapping (support 0.0-1.0 and 0.0-100.0 formats)
  let confidence = 0;
  if (data.confidence !== undefined) {
    confidence = data.confidence <= 1.0 ? data.confidence * 100 : data.confidence;
  }

  // 4. Model score mappings
  const cnnScore = data.proba_cnn !== undefined 
    ? (data.proba_cnn <= 1.0 ? data.proba_cnn * 100 : data.proba_cnn)
    : (data.cnn_score !== undefined ? data.cnn_score : 0);

  const lstmScore = data.proba_lstm !== undefined
    ? (data.proba_lstm <= 1.0 ? data.proba_lstm * 100 : data.proba_lstm)
    : (data.lstm_score !== undefined ? data.lstm_score : 0);

  const bioScore = data.proba_bio !== undefined
    ? (data.proba_bio <= 1.0 ? data.proba_bio * 100 : data.proba_bio)
    : (data.bio_score !== undefined ? data.bio_score : 0);

  // 5. Duration mapping
  const duration = data.duration_sec !== undefined 
    ? data.duration_sec 
    : (data.duration !== undefined ? data.duration : 0);

  // 6. Inference time mapping
  const inferenceTime = data.inference_ms !== undefined
    ? data.inference_ms
    : (data.inference_time !== undefined ? data.inference_time : 0);

  return {
    jobId: data.job_id || 'manual-eval',
    filename,
    verdict,
    confidence: Math.round(confidence * 10) / 10,
    cnnScore: Math.round(cnnScore * 10) / 10,
    lstmScore: Math.round(lstmScore * 10) / 10,
    bioScore: Math.round(bioScore * 10) / 10,
    duration: Math.round(duration * 100) / 100,
    inferenceTime: Math.round(inferenceTime),
    timestamp: new Date().toISOString(),
    rawJson: JSON.stringify(data, null, 2),
  };
}

/**
 * Upload audio file and run complete analysis.
 * Uses a polling mechanism to fetch results when they are ready.
 */
export async function analyzeAudio(
  file: File,
  onProgress?: (message: string) => void
): Promise<NormalizedResult> {
  const formData = new FormData();
  formData.append('file', file);

  onProgress?.('Uploading audio payload...');
  
  let jobResponse: { data: BackendAnalyzeResponse };
  try {
    jobResponse = await client.post<BackendAnalyzeResponse>('/analyze', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
  } catch (error: unknown) {
    console.error("Upload failed:", error);
    const err = error as { response?: { status: number; data?: { detail?: string } } };
    if (!err.response) {
      throw new Error('Backend Offline: Could not establish a connection to the forensic API server.', { cause: error });
    }
    if (err.response.status === 413) {
      throw new Error('Payload Size Violation: The uploaded audio file exceeds the 10 MB maximum size limit.', { cause: error });
    }
    if (err.response.status === 400) {
      throw new Error(`Unsupported Format: ${err.response.data?.detail || 'The file format is not supported.'}`, { cause: error });
    }
    throw new Error(err.response.data?.detail || 'Inference request failed during file submission.', { cause: error });
  }

  const { job_id, status } = jobResponse.data;
  
  // If the status from post is already indicating an immediate result/done
  if (status === 'done') {
    onProgress?.('Analysis compiled. Preparing report...');
    const res = await client.get<BackendResultResponse>(`/result/${job_id}`);
    return normalizeResponse(res.data);
  }

  // Polling loop
  const maxAttempts = 30;
  const pollInterval = 1000; // 1 second
  let attempt = 0;
  
  const messages = [
    "Extracting Audio Features...",
    "Extracting Mel Spectrogram...",
    "Running CNN Analysis...",
    "Scanning Spectral Patterns...",
    "Running LSTM Analysis...",
    "Analyzing Temporal Sequences...",
    "Running Voice Biometrics...",
    "Verifying Vocal Authenticity...",
    "Calculating Confidence Score...",
    "Formatting Forensic Results..."
  ];

  while (attempt < maxAttempts) {
    attempt++;
    
    // Cycle scanning messages for premium UX
    const msgIndex = Math.min(Math.floor((attempt / maxAttempts) * messages.length), messages.length - 1);
    onProgress?.(messages[msgIndex]);

    try {
      const resultRes = await client.get<BackendResultResponse>(`/result/${job_id}`);
      
      if (resultRes.data.status === 'done') {
        onProgress?.('Authenticity check complete.');
        return normalizeResponse(resultRes.data);
      }
      
      if (resultRes.data.status === 'error') {
        throw new Error(resultRes.data.error || 'Forensic model processing encountered a system exception.');
      }
    } catch (pollError: unknown) {
      const pErr = pollError as { response?: { status: number; data?: { detail?: string } }; message?: string };
      // If result endpoint returns 404, the job is still processing.
      // Other error statuses (500) will be thrown.
      if (pErr.response && pErr.response.status !== 404) {
        throw new Error(pErr.response.data?.detail || 'Polling failed due to server exception.', { cause: pollError });
      }
      if (pErr.message && !pErr.response) {
        // network issue
        throw new Error(pErr.message, { cause: pollError });
      }
    }

    await new Promise(resolve => setTimeout(resolve, pollInterval));
  }

  throw new Error('Analysis Timeout: The ensemble models took longer than 30 seconds to respond.');
}
