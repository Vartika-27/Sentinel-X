"""
main.py – Sentinel-X FastAPI Backend

Endpoints:
    GET  /               Health ping
    GET  /health         Health check (JSON)
    POST /predict/image  Top-1 classification only
    POST /attack/preview Lightweight FGSM preview (no PGD, base64 response)
    POST /attack/image   Full FGSM + PGD pipeline with caching, DB logging, top-5

Environment variables (see .env.example):
    REDIS_URL       – Redis connection URL  (default: redis://localhost:6379)
    POSTGRES_DSN    – PostgreSQL DSN        (default: local sentinelx DB)
"""

import asyncio
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

import cache as cache_module
import db as db_module
from attack import fgsm_attack, pgd_attack
from model import get_model, load_model
from utils import (
    preprocess_image,
    run_inference,
    run_top5_inference,
    tensor_to_perturbation_b64,
)
from visualize import visualize_attack

# Load .env before anything else
load_dotenv()

# ---------------------------------------------------------------------------
# Allowed image MIME types
# ---------------------------------------------------------------------------

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}


# ---------------------------------------------------------------------------
# Lifespan – startup / shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # ── startup ──────────────────────────────────────────────────────────
    print("Loading ResNet-18 weights …")
    load_model()
    print("Model ready.")

    # Redis (non-fatal if unavailable — caching is best-effort)
    try:
        await cache_module.init_redis()
    except Exception as exc:
        print(f"[WARNING] Redis unavailable — caching disabled: {exc}")

    # PostgreSQL (non-fatal — logging is best-effort)
    try:
        await db_module.init_db()
    except Exception as exc:
        print(f"[WARNING] PostgreSQL unavailable — DB logging disabled: {exc}")

    yield

    # ── shutdown ─────────────────────────────────────────────────────────
    await cache_module.close_redis()
    await db_module.close_db()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Sentinel-X – Adversarial ML Demo",
    description=(
        "Upload an image to classify it with ResNet-18 and see how "
        "FGSM / PGD adversarial attacks change the prediction."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory="."), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _read_and_validate(file: UploadFile) -> bytes:
    """Read upload bytes and validate the MIME type."""
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported media type '{file.content_type}'. "
                f"Accepted: {sorted(ALLOWED_CONTENT_TYPES)}"
            ),
        )
    return await file.read()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    return {"message": "Sentinel-X Backend Running", "version": "2.0.0"}


@app.get("/health", summary="Health check")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})


# ── Standard inference (top-1 only) ─────────────────────────────────────────

@app.post("/predict/image", summary="Classify an uploaded image (top-1)")
async def predict_image(file: UploadFile = File(...)) -> JSONResponse:
    """
    Accept an image, resize to 224×224, run ResNet-18, return top-1 prediction.
    """
    try:
        image_bytes = await _read_and_validate(file)
        tensor = preprocess_image(image_bytes)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not decode image: {exc}") from exc

    try:
        result = run_inference(get_model(), tensor)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    return JSONResponse({"filename": file.filename, **result})


# ── Live FGSM preview (lightweight, no PGD, no DB) ──────────────────────────

@app.post("/attack/preview", summary="Lightweight FGSM preview as ε changes")
async def attack_preview(
    file: UploadFile = File(...),
    epsilon: float = 0.03,
) -> JSONResponse:
    """
    Run a single FGSM pass and return original + FGSM predictions plus a
    Base64-encoded PNG of the amplified perturbation noise map.

    Designed to be called on every epsilon slider update (debounced on client).
    PGD is intentionally skipped — it is multi-step and too slow for real-time use.

    Response:
    ```json
    {
        "epsilon": 0.03,
        "original_prediction": { "class_index": 281, "label": "tabby", "confidence": 0.92 },
        "fgsm_prediction":     { "class_index": 12,  "label": "house finch", "confidence": 0.55 },
        "perturbation_b64":    "<base64-encoded PNG of noise>"
    }
    ```
    """
    if not (0.001 <= epsilon <= 1.0):
        raise HTTPException(status_code=422, detail="epsilon must be in [0.001, 1.0]")

    try:
        image_bytes = await _read_and_validate(file)
        tensor = preprocess_image(image_bytes)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not decode image: {exc}") from exc

    try:
        model = get_model()
        original_prediction = run_inference(model, tensor)

        fgsm_tensor = fgsm_attack(model, tensor.clone(), epsilon=epsilon)
        fgsm_prediction = run_inference(model, fgsm_tensor)

        perturbation_b64 = tensor_to_perturbation_b64(tensor, fgsm_tensor)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Preview pipeline failed: {exc}") from exc

    return JSONResponse({
        "epsilon": epsilon,
        "original_prediction": original_prediction,
        "fgsm_prediction": fgsm_prediction,
        "perturbation_b64": perturbation_b64,
    })


# ── Full attack pipeline (FGSM + PGD, top-5, cache, DB) ─────────────────────

@app.post("/attack/image", summary="Full FGSM + PGD adversarial attack")
async def attack_image(
    file: UploadFile = File(...),
    epsilon: float = 0.03,
) -> JSONResponse:
    """
    Full adversarial attack pipeline:

    1. Check Redis cache for (image_hash, epsilon) — skip ML if hit.
    2. Run ResNet-18 original inference (top-5).
    3. Apply FGSM attack → top-5 inference.
    4. Apply PGD attack  → top-5 inference.
    5. Generate and save attack visualization PNG.
    6. Store result in Redis (TTL 1 h).
    7. Log run asynchronously to PostgreSQL.

    Response includes `top5` arrays for chart rendering in the frontend.
    """
    if not (0.001 <= epsilon <= 1.0):
        raise HTTPException(status_code=422, detail="epsilon must be in [0.001, 1.0]")

    try:
        image_bytes = await _read_and_validate(file)
        tensor = preprocess_image(image_bytes)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not decode image: {exc}") from exc

    image_hash = cache_module.compute_image_hash(image_bytes)

    # ── Cache check ──────────────────────────────────────────────────────────
    cached = await cache_module.get_cached(image_hash, epsilon)
    if cached is not None:
        # Fire-and-forget DB log (mark cache_hit=True)
        asyncio.create_task(
            db_module.log_attack(
                filename=file.filename,
                epsilon=epsilon,
                image_hash=image_hash,
                orig=cached["original_prediction"],
                fgsm=cached["fgsm_prediction"],
                pgd=cached["pgd_prediction"],
                cache_hit=True,
            )
        )
        return JSONResponse({**cached, "cache_hit": True})

    # ── Full pipeline ────────────────────────────────────────────────────────
    try:
        model = get_model()

        original_prediction  = run_top5_inference(model, tensor)

        fgsm_tensor          = fgsm_attack(model, tensor.clone(), epsilon=epsilon)
        fgsm_prediction      = run_top5_inference(model, fgsm_tensor)

        pgd_tensor           = pgd_attack(model, tensor.clone(), epsilon=epsilon)
        pgd_prediction       = run_top5_inference(model, pgd_tensor)

        # Save visualisation PNG (original | adversarial | perturbation)
        visualize_attack(tensor, pgd_tensor, save_path="attack_visualization.png")

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Attack pipeline failed: {exc}") from exc

    result = {
        "filename":             file.filename,
        "epsilon":              epsilon,
        "original_prediction":  original_prediction,
        "fgsm_prediction":      fgsm_prediction,
        "pgd_prediction":       pgd_prediction,
    }

    # ── Cache store ──────────────────────────────────────────────────────────
    await cache_module.set_cached(image_hash, epsilon, result)

    # ── DB logging (fire-and-forget — never blocks the response) ────────────
    asyncio.create_task(
        db_module.log_attack(
            filename=file.filename,
            epsilon=epsilon,
            image_hash=image_hash,
            orig=original_prediction,
            fgsm=fgsm_prediction,
            pgd=pgd_prediction,
            cache_hit=False,
        )
    )

    return JSONResponse({**result, "cache_hit": False})
