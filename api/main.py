"""
FastAPI Backend — Deepfake Audio Detector
==========================================
Endpoints:
  POST /analyze        upload audio file → returns job_id
  GET  /result/{id}    get prediction result
  GET  /health         health check

Run:
  uvicorn api.main:app --reload --port 8000
"""

import uuid
import time
import sys
from pathlib import Path
from typing import Dict

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from api.schemas.request  import AnalyzeResponse, ResultResponse, HealthResponse
from api.predictor        import get_predictor
from src.utils.logger     import get_logger

logger = get_logger("api")

app = FastAPI(
    title="Deepfake Audio Detector",
    description="Detect AI-generated voices using CNN + LSTM + Voice Biometrics",
    version="1.0.0",
)

# Allow frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory result store (replace with Redis/DB for production)
results: Dict[str, dict] = {}

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
MAX_FILE_SIZE_MB   = 10


@app.get("/health", response_model=HealthResponse)
def health():
    """Check if the API and model are ready."""
    predictor = get_predictor()
    return HealthResponse(
        status="ok",
        model_loaded=predictor.is_ready(),
    )


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(file: UploadFile = File(...)):
    """
    Upload an audio file and get a job_id.
    Poll GET /result/{job_id} for the verdict.
    """
    # ── Validate file ──────────────────────────────────────────────────────
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{suffix}'. Allowed: {ALLOWED_EXTENSIONS}"
        )

    audio_bytes = await file.read()
    size_mb = len(audio_bytes) / 1e6
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f} MB). Max: {MAX_FILE_SIZE_MB} MB"
        )

    # ── Save temp file ─────────────────────────────────────────────────────
    job_id   = str(uuid.uuid4())
    tmp_dir  = ROOT / "data" / "uploads"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / f"{job_id}{suffix}"

    with open(tmp_path, "wb") as f:
        f.write(audio_bytes)

    # ── Run inference ──────────────────────────────────────────────────────
    try:
        t_start   = time.time()
        predictor = get_predictor()
        result    = predictor.predict(str(tmp_path))
        elapsed   = round((time.time() - t_start) * 1000, 1)

        results[job_id] = {
            "job_id":       job_id,
            "filename":     file.filename,
            "verdict":      result["verdict"],
            "confidence":   result["confidence"],
            "proba_cnn":    result["proba_cnn"],
            "proba_lstm":   result["proba_lstm"],
            "proba_bio":    result["proba_bio"],
            "duration_sec": result["duration_sec"],
            "inference_ms": elapsed,
            "status":       "done",
        }
        logger.info(f"[{job_id[:8]}] {file.filename} → {result['verdict']} "
                    f"({result['confidence']*100:.1f}%) in {elapsed}ms")

    except Exception as e:
        results[job_id] = {"status": "error", "error": str(e)}
        logger.error(f"[{job_id[:8]}] Inference failed: {e}")
    finally:
        # Clean up temp file
        try:
            tmp_path.unlink()
        except Exception:
            pass

    return AnalyzeResponse(job_id=job_id, status=results[job_id]["status"])


@app.get("/result/{job_id}", response_model=ResultResponse)
def get_result(job_id: str):
    """Get the prediction result for a job_id."""
    if job_id not in results:
        raise HTTPException(status_code=404, detail="Job not found")

    r = results[job_id]
    if r["status"] == "error":
        raise HTTPException(status_code=500, detail=r.get("error", "Unknown error"))

    return ResultResponse(**r)