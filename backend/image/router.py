import asyncio
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

import cache as cache_module
import db as db_module

from .attack import fgsm_attack, pgd_attack
from .model import get_model
from .utils import (
    preprocess_image,
    run_inference,
    run_top5_inference,
    tensor_to_perturbation_b64,
)
from .visualize import visualize_attack

router = APIRouter()

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}


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


@router.post("/predict/image", summary="Classify an uploaded image (top-1)")
async def predict_image(file: UploadFile = File(...)) -> JSONResponse:
    """
    Accept an image, resize to 224x224, run ResNet-18, return top-1 prediction.
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


@router.post("/attack/preview", summary="Lightweight FGSM preview as epsilon changes")
async def attack_preview(
    file: UploadFile = File(...),
    epsilon: float = 0.03,
) -> JSONResponse:
    """
    Run a single FGSM pass and return original + FGSM predictions plus a
    Base64-encoded PNG of the amplified perturbation noise map.
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


@router.post("/attack/image", summary="Full FGSM + PGD adversarial attack")
async def attack_image(
    file: UploadFile = File(...),
    epsilon: float = 0.03,
) -> JSONResponse:
    """
    Full adversarial attack pipeline.
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

    # Cache check
    cached = await cache_module.get_cached(image_hash, epsilon)
    if cached is not None:
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

    # Full pipeline
    try:
        model = get_model()

        original_prediction  = run_top5_inference(model, tensor)

        fgsm_tensor          = fgsm_attack(model, tensor.clone(), epsilon=epsilon)
        fgsm_prediction      = run_top5_inference(model, fgsm_tensor)

        pgd_tensor           = pgd_attack(model, tensor.clone(), epsilon=epsilon)
        pgd_prediction       = run_top5_inference(model, pgd_tensor)

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

    # Cache store
    await cache_module.set_cached(image_hash, epsilon, result)

    # DB logging
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
