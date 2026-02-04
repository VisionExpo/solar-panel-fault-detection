from pathlib import Path
import shutil
import uuid

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from solar_fault_detector.config.config import Config
from solar_fault_detector.inference.predictor import Predictor

# ======================
# App Initialization
# ======================
app = FastAPI(
    title="Solar Panel Fault Detection API",
    description="Real-time inference API for solar panel fault detection",
    version="1.0.0",
)

config = Config()

MODEL_PATH = config.model.best_model_path
UPLOAD_DIR = Path("artifacts/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

predictor = Predictor(
    model_path=MODEL_PATH,
    config=config.model,
)

ALLOWED_MIME_TYPES = {"image/jpeg", "image/jpg", "image/png"}


# ======================
# Health Check
# ======================
@app.get("/health")
def health_check():
    return {"status": "ok"}


# ======================
# Inference Endpoint
# ======================
@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
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
