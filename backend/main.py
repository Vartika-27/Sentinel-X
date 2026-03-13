from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from visualize import visualize_attack
from model import get_model, load_model
from utils import preprocess_image, run_inference
from attack import fgsm_attack, pgd_attack

# ---------------------------------------------------------------------------
# Lifespan – load model once on startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    print("Loading ResNet18 weights …")
    load_model()
    print("Model ready.")
    yield
    # (cleanup goes here if needed)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Adversarial ML Demo – Backend",
    description="Upload an image to get a ResNet18 ImageNet prediction.",
    version="1.0.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory="."), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}


@app.get("/health", summary="Health check")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})


@app.post("/predict/image", summary="Classify an uploaded image")
async def predict_image(file: UploadFile = File(...)) -> JSONResponse:
    """
    Accept a JPEG / PNG / WebP image, resize to 224 × 224, run ResNet18
    inference, and return:

    ```json
    {
        "filename": "cat.jpg",
        "class_index": 281,
        "confidence": 0.923456
    }
    ```
    """
    # ── validate content type ──────────────────────────────────────────────
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported media type '{file.content_type}'. "
                f"Accepted: {sorted(ALLOWED_CONTENT_TYPES)}"
            ),
        )

    # ── read & preprocess ──────────────────────────────────────────────────
    try:
        image_bytes = await file.read()
        tensor = preprocess_image(image_bytes)
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Could not decode image: {exc}",
        ) from exc

    # ── inference ──────────────────────────────────────────────────────────
    try:
        model = get_model()
        result = run_inference(model, tensor)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {exc}",
        ) from exc

    return JSONResponse({"filename": file.filename, **result})

@app.post("/attack/image", summary="Run FGSM adversarial attack on an uploaded image")
async def attack_image(file: UploadFile = File(...), epsilon: float = 0.03) -> JSONResponse:

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type '{file.content_type}'."
        )

    try:
        image_bytes = await file.read()
        tensor = preprocess_image(image_bytes)
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Could not decode image: {exc}",
        ) from exc

    try:
        model = get_model()

        # original prediction
        original_prediction = run_inference(model, tensor)

        # FGSM attack
        fgsm_tensor = fgsm_attack(model, tensor.clone(), epsilon=epsilon)
        fgsm_prediction = run_inference(model, fgsm_tensor)

        # PGD attack
        pgd_tensor = pgd_attack(model, tensor.clone(), epsilon=epsilon)
        pgd_prediction = run_inference(model, pgd_tensor)

        # generate visualization using PGD attack (stronger example)
        visualize_attack(tensor, pgd_tensor, save_path="attack_visualization.png")

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Attack pipeline failed: {exc}",
        ) from exc


    return JSONResponse(
    {
        "filename": file.filename,
        "epsilon": epsilon,
        "original_prediction": original_prediction,
        "fgsm_prediction": fgsm_prediction,
        "pgd_prediction": pgd_prediction,
    }
)



