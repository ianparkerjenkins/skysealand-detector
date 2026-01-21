import logging
import pathlib

from fastapi import FastAPI, File, HTTPException, UploadFile

from skysealand import inference, logging_setup

app = FastAPI()
logging_setup.setup_logging()
logger = logging.getLogger(__name__)


# TODO: Make this a config setting.
MODEL_PATH = pathlib.Path("yolov8n.pt")


_model = None


def _get_model():
    """Lazily load the model so it can be cached."""
    # Doing this global for simplicity for now.
    global _model  # noqa: PLW0603
    if _model is None:
        logger.info("No model cached. Loading YOLO model from %s", MODEL_PATH)
        _model = inference.load_ultralytics_yolo_model(MODEL_PATH)
    return _model


@app.post("/infer")
async def infer_endpoint(
    files: list[UploadFile] = File(..., description="One or more image files"),
) -> inference.InferenceJsonOutput:
    logger.info("Received inference request with %d file(s)", len(files))

    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    try:
        images, filenames = inference.load_images(*files)
        return inference.run_model_with_timing(_get_model(), images, filenames)

    except ValueError as e:
        logger.warning("Validation error: %s", e, exc_info=True)
        raise HTTPException(status_code=400, detail=str(e)) from None

    except Exception:
        logger.exception("Unhandled inference error")
        raise HTTPException(
            status_code=500,
            detail="Internal inference error",
        ) from None
