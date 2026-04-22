from pathlib import Path
import shutil
import uuid
import logging

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

import tensorflow as tf

from solar_fault_detector.config.config import Config
from solar_fault_detector.inference.predictor import Predictor
from solar_fault_detector.utils.download_model import ensure_model_exists

# ======================
# Logging Setup
# ======================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ======================
# App Initialization
# ======================
app = FastAPI(
    title="Solar Panel Fault Detection API",
    description="Real-time inference API for solar panel fault detection",
    version="1.0.0",
)

config = Config()
UPLOAD_DIR = Path("artifacts/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ======================
# CPU Memory / Thread Configuration
# ======================
# Prevent OOM crashes on CPU deployments (like Render) by limiting threads
try:
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    logger.info("✅ Configured CPU thread limits to prevent memory bloat")
except Exception as e:
    logger.error(f"❌ Failed to configure CPU thread limits: {e}")

# ======================
# Model Loading with Fallback
# ======================
try:
    MODEL_PATH = ensure_model_exists(config.model.best_model_path)
    logger.info(f"Loading model from: {MODEL_PATH}")

    predictor = Predictor(
        model_path=MODEL_PATH,
        config=config.model,
    )
    MODEL_READY = True
    logger.info("✅ Model loaded successfully")
except Exception as e:
    logger.error(f"❌ Failed to load model: {e}")
    predictor = None
    MODEL_READY = False

ALLOWED_MIME_TYPES = {"image/jpeg", "image/jpg", "image/png"}


# ======================
# Health Check
# ======================
@app.get("/health")
def health_check():
    return {
        "status": "ok" if MODEL_READY else "model_not_ready",
        "model_loaded": MODEL_READY,
    }


# ======================
# Inference Endpoint
# ======================
@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    if not MODEL_READY:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Service unavailable. Check server logs.",
        )

    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Upload a JPG or PNG image.",
        )

    suffix = Path(file.filename).suffix
    temp_filename = f"{uuid.uuid4()}{suffix}"
    temp_path = UPLOAD_DIR / temp_filename

    try:
        with temp_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        prediction = predictor.predict(temp_path)
        return JSONResponse(content=prediction)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if temp_path.exists():
            temp_path.unlink()
