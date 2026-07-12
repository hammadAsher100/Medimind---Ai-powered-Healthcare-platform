from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status

from cnn.preprocessing import ImageValidationError
from cnn.registry import CNNModelRegistry, CNNModelUnavailable


router = APIRouter(prefix="/cnn", tags=["cnn-inference"])


def _registry(request: Request) -> CNNModelRegistry:
    registry = getattr(request.app.state, "cnn_registry", None)
    if registry is None:
        registry = CNNModelRegistry()
        registry.load_all()
        request.app.state.cnn_registry = registry
    return registry


@router.get("/models")
async def list_cnn_models(request: Request):
    try:
        return _registry(request).status()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unable to list CNN models: {exc}") from exc


@router.post("/predict/{model_id}")
async def predict_cnn_model(request: Request, model_id: str, file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        return _registry(request).predict(
            model_id=model_id,
            image_bytes=image_bytes,
            filename=file.filename,
            content_type=file.content_type,
        )
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ImageValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except CNNModelUnavailable as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"CNN inference failed: {exc}") from exc

