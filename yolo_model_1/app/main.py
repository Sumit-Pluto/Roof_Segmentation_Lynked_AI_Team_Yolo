import os
import io
import logging
import uuid
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from PIL import Image
import cv2  # OpenCV
from ultralytics import YOLO


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Paths are now fixed *inside* the container ---
MODEL_PATH = "models/model_weights.pt"
ARTIFACTS_DIR = "/app/artifacts"
CONFIDENCE_THRESHOLD = 0.5  

# --- Model Loading ---

def load_model(model_path):
    """
    Loads the YOLO model.
    Returns the model object or None if loading fails.
    """
    if not os.path.exists(model_path):
        logger.error(f"Model file not found at {model_path}")
        return None
    try:
        model = YOLO(model_path)
        logger.info(f"Successfully loaded model from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None

# Load model on startup
model = load_model(MODEL_PATH)

# --- FastAPI App ---

app = FastAPI(title="YOLO Segmentation API (Packaged)", version="1.1.0")

@app.on_event("startup")
async def startup_event():
    # Ensure model is loaded on startup
    global model
    if model is None:
        logger.warning("Model not loaded at startup, attempting to reload...")
        model = load_model(MODEL_PATH)
    
    # Ensure artifacts directory exists *inside the container*
    try:
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)
        logger.info(f"Artifacts directory ensured at {ARTIFACTS_DIR}")
    except Exception as e:
        logger.error(f"Could not create artifacts directory: {e}")


@app.get("/", summary="Health Check")
def read_root():
    """
    Health check endpoint to ensure service is running.
    """
    return {"status": "ok", "message": "YOLO segmentation API is running."}


@app.post("/predict", summary="Run Object Segmentation")
async def predict(file: UploadFile = File(..., description="Image file to process")):
    """
    Perform segmentation on an uploaded image.

    - **Receives**: An image file.
    - **Runs**: YOLO segmentation model.
    - **Saves**: The resulting image to an internal 'artifacts' folder.
    - **Returns**: The resulting image as a JPEG.
    """
    global model
    if model is None:
        logger.error("Model is not loaded. Cannot perform prediction.")
        raise HTTPException(status_code=500, detail="Model is not loaded")

    try:
        # Read image file
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents))

        # --- Model Inference ---
        results = model(pil_image, conf=CONFIDENCE_THRESHOLD)
        
        if not results:
            logger.warning("No results found for the input image.")
            raise HTTPException(status_code=404, detail="No objects detected in the image.")

        # --- Plotting Results ---
        # .plot() returns a BGR NumPy array with detections, masks, and labels
        result_image_bgr = results[0].plot()
        result_image_rgb = cv2.cvtColor(result_image_bgr, cv2.COLOR_BGR2RGB)
        pil_result_image = Image.fromarray(result_image_rgb)

        # --- Save Artifact ---
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_id = uuid.uuid4().hex[:8]
        filename = f"result_{timestamp}_{unique_id}.jpg"
        save_path = os.path.join(ARTIFACTS_DIR, filename)

        try:
            pil_result_image.save(save_path, "JPEG")
            logger.info(f"Successfully saved artifact to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save artifact: {e}")

        # --- Return Image to User ---
        img_buffer = io.BytesIO()
        pil_result_image.save(img_buffer, "JPEG")
        img_buffer.seek(0) # Rewind buffer to the beginning

        return StreamingResponse(img_buffer, media_type="image/jpeg", headers={
            "Content-Disposition": f"inline; filename={filename}"
        })

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})