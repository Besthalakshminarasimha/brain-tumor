from __future__ import annotations

import base64
import importlib.util
import io
import json
import sqlite3
import threading
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps, UnidentifiedImageError
from pydicom import dcmread
from pydicom.errors import InvalidDicomError
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"
PUBLIC_DATASET_DIR = DATA_DIR / "brain-tumor-mri-cc0"
DATABASE = DATA_DIR / "workspace.sqlite3"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
MODEL_PATH = ARTIFACTS_DIR / "model.pt"
METADATA_PATH = ARTIFACTS_DIR / "metadata.json"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_IMAGE_DIMENSION = 8192
MAX_IMAGE_PIXELS = 40_000_000

app = FastAPI(title="Neuro Imaging Lab API", version="0.1.0")
training_lock = threading.Lock()
training_state: dict[str, Any] = {
    "running": False,
    "message": None,
    "error": None,
}
dataset_state: dict[str, Any] = {
    "preparing": False,
    "message": None,
    "error": None,
}


def get_dataset_state() -> dict[str, Any]:
    metadata_path = PUBLIC_DATASET_DIR / "dataset-source.json"
    metadata = None
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            metadata = None
    classes = metadata.get("classes", []) if metadata else []
    ready = bool(classes) and all(
        (PUBLIC_DATASET_DIR / split / class_name).is_dir()
        and any((PUBLIC_DATASET_DIR / split / class_name).glob("*.png"))
        for split in ("train", "test")
        for class_name in classes
    )
    return {
        "ready": ready,
        "path": str(PUBLIC_DATASET_DIR),
        "preparing": dataset_state["preparing"],
        "message": dataset_state["message"],
        "error": dataset_state["error"],
        "metadata": metadata if ready else None,
    }


def connect_database() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE)
    connection.row_factory = sqlite3.Row
    connection.execute(
        """CREATE TABLE IF NOT EXISTS cases (
            id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            created_at TEXT NOT NULL,
            status TEXT NOT NULL
        )"""
    )
    connection.execute(
        """CREATE TABLE IF NOT EXISTS audit_events (
            id TEXT PRIMARY KEY,
            event_type TEXT NOT NULL,
            filename TEXT,
            role TEXT NOT NULL,
            created_at TEXT NOT NULL
        )"""
    )
    return connection


def audit(event_type: str, filename: str | None, role: str) -> None:
    allowed_roles = {"admin", "radiologist", "patient"}
    safe_role = role if role in allowed_roles else "radiologist"
    with closing(connect_database()) as connection:
        connection.execute(
            "INSERT INTO audit_events (id, event_type, filename, role, created_at) VALUES (?, ?, ?, ?, ?)",
            (uuid.uuid4().hex, event_type, Path(filename or "").name or None, safe_role, datetime.now(timezone.utc).isoformat()),
        )
        connection.commit()


def image_from_bytes(contents: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(contents))
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise HTTPException(status_code=415, detail="Upload a valid JPG or PNG image.") from error
    if image.format not in {"JPEG", "PNG"}:
        raise HTTPException(status_code=415, detail="Only JPG and PNG images are supported.")
    if max(image.size) > MAX_IMAGE_DIMENSION or image.width * image.height > MAX_IMAGE_PIXELS:
        raise HTTPException(status_code=413, detail="Image dimensions exceed the 8192-pixel limit.")
    try:
        image.load()
    except (OSError, Image.DecompressionBombError) as error:
        raise HTTPException(status_code=415, detail="Upload a valid JPG or PNG image.") from error
    return image.convert("L")


def preprocess_image(
    image: Image.Image,
    auto_crop: bool = True,
    noise_reduction: bool = True,
    skull_stripping: bool = False,
) -> Image.Image:
    image = ImageOps.exif_transpose(image)
    if noise_reduction:
        image = image.filter(ImageFilter.MedianFilter(size=3))
    if auto_crop:
        foreground = image.point(lambda value: 255 if value > 8 else 0)
        bbox = foreground.getbbox()
        if bbox and bbox[2] - bbox[0] > 16 and bbox[3] - bbox[1] > 16:
            image = image.crop(bbox)
    if skull_stripping:
        mask = Image.new("L", image.size, 0)
        mask_draw = ImageDraw.Draw(mask)
        inset_x, inset_y = image.width * 0.06, image.height * 0.06
        mask_draw.ellipse((inset_x, inset_y, image.width - inset_x, image.height - inset_y), fill=255)
        masked_image = Image.new("L", image.size, 0)
        masked_image.paste(image, mask=mask)
        image = masked_image
    image.thumbnail((224, 224), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (224, 224), 0)
    canvas.paste(image, ((224 - image.width) // 2, (224 - image.height) // 2))
    return ImageOps.autocontrast(canvas, cutoff=1)


def image_quality_metrics(image: Image.Image) -> dict[str, Any]:
    pixels = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    contrast = float(pixels.std())
    gradient_y, gradient_x = np.gradient(pixels)
    sharpness = float((gradient_x ** 2 + gradient_y ** 2).mean())
    entropy_histogram, _ = np.histogram(pixels, bins=32, range=(0, 1), density=True)
    probabilities = entropy_histogram / max(float(entropy_histogram.sum()), 1e-8)
    entropy = float(-sum(value * np.log2(value) for value in probabilities if value > 0))
    foreground_ratio = float((pixels > 0.05).mean())
    score = int(round(np.clip(35 * min(contrast / 0.2, 1) + 35 * min(sharpness / 0.02, 1) + 30 * min(entropy / 4.5, 1), 0, 100)))
    flags = []
    if contrast < 0.05:
        flags.append("low_contrast")
    if sharpness < 0.001:
        flags.append("low_sharpness")
    if foreground_ratio < 0.15:
        flags.append("limited_foreground")
    return {
        "score": score,
        "contrast": round(contrast, 4),
        "sharpness": round(sharpness, 6),
        "entropy": round(entropy, 3),
        "foreground_ratio": round(foreground_ratio, 3),
        "flags": flags,
        "interpretation": "research quality screen; not a clinical acceptability decision",
    }


def research_robustness_check(image: Image.Image, model_path: Path, metadata_path: Path) -> dict[str, Any]:
    from .modeling import predict

    variants = {
        "original": image,
        "slightly_brighter": ImageEnhance.Brightness(image).enhance(1.08),
        "slightly_darker": ImageEnhance.Brightness(image).enhance(0.92),
        "slightly_higher_contrast": ImageEnhance.Contrast(image).enhance(1.08),
    }
    outputs = []
    for name, variant in variants.items():
        prediction = predict(variant, model_path, metadata_path)
        outputs.append({
            "variant": name,
            "predicted_class": prediction["predicted_class"],
            "confidence": prediction["confidence"],
        })
    classes = [item["predicted_class"] for item in outputs]
    agreement = sum(item == classes[0] for item in classes) / len(classes)
    return {
        "variants": outputs,
        "agreement": round(agreement, 3),
        "stable": agreement == 1.0,
        "interpretation": "research stress test under mild intensity perturbations; not clinical validation",
    }


async def read_scan_upload(file: UploadFile) -> tuple[Image.Image, dict[str, Any]]:
    filename_suffix = Path(file.filename or "").suffix.lower()
    is_nifti = filename_suffix == ".nii" or str(file.filename or "").lower().endswith(".nii.gz")
    is_dicom = file.content_type in {"application/dicom", "application/dicom+json"} or filename_suffix in {".dcm", ".dicom", ".ima"}
    is_raster = file.content_type in {"image/jpeg", "image/png"}
    if not is_dicom and not is_raster and not is_nifti:
        raise HTTPException(status_code=415, detail="Upload an MRI DICOM, NIfTI, JPG, or PNG image.")
    contents = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image must be 20 MB or smaller.")
    if is_dicom:
        try:
            dataset = dcmread(io.BytesIO(contents), force=False)
        except (InvalidDicomError, EOFError, OSError, ValueError) as error:
            raise HTTPException(status_code=415, detail="The file is not a valid DICOM image.") from error
        if str(getattr(dataset, "Modality", "")).upper() != "MR":
            raise HTTPException(status_code=415, detail="Only MR-modality DICOM images are supported.")
        if int(getattr(dataset, "NumberOfFrames", 1)) != 1:
            raise HTTPException(status_code=422, detail="Multi-frame DICOM is not supported. Upload one 2D MR slice.")
        rows = int(getattr(dataset, "Rows", 0))
        columns = int(getattr(dataset, "Columns", 0))
        if rows < 1 or columns < 1 or max(rows, columns) > MAX_IMAGE_DIMENSION or rows * columns > MAX_IMAGE_PIXELS:
            raise HTTPException(status_code=413, detail="DICOM image dimensions exceed the supported limit.")
        if int(getattr(dataset, "SamplesPerPixel", 1)) != 1:
            raise HTTPException(status_code=415, detail="Only single-channel MR DICOM images are supported.")
        try:
            pixels = np.asarray(dataset.pixel_array, dtype=np.float32)
        except (AttributeError, NotImplementedError, RuntimeError, ValueError) as error:
            raise HTTPException(status_code=422, detail="DICOM pixel data could not be decoded. Compressed transfer syntaxes may need an additional codec.") from error
        if pixels.ndim != 2 or not np.isfinite(pixels).any():
            raise HTTPException(status_code=422, detail="DICOM pixel data must contain one valid 2D image.")
        slope = float(getattr(dataset, "RescaleSlope", 1))
        intercept = float(getattr(dataset, "RescaleIntercept", 0))
        pixels = pixels * slope + intercept
        finite_pixels = pixels[np.isfinite(pixels)]
        low, high = np.percentile(finite_pixels, (0.5, 99.5))
        if high <= low:
            low, high = float(finite_pixels.min()), float(finite_pixels.max())
        if high <= low:
            normalized = np.zeros(pixels.shape, dtype=np.uint8)
        else:
            normalized = np.uint8(np.clip((pixels - low) * (255.0 / (high - low)), 0, 255))
        image = Image.fromarray(normalized, mode="L")
        if str(getattr(dataset, "PhotometricInterpretation", "MONOCHROME2")) == "MONOCHROME1":
            image = ImageOps.invert(image)
        spacing_value = getattr(dataset, "PixelSpacing", None)
        metadata = {
            "format": "DICOM",
            "modality": "MR",
            "rows": rows,
            "columns": columns,
            "pixel_spacing_mm": [float(value) for value in spacing_value] if spacing_value else None,
            "anonymized": True,
        }
        return image, metadata

    if is_nifti:
        try:
            import nibabel as nib
            volume = np.asarray(nib.Nifti1Image.from_bytes(contents).get_fdata(), dtype=np.float32)
        except ImportError as error:
            raise HTTPException(status_code=503, detail="NIfTI support is not installed. Install the API requirements and restart.") from error
        except (OSError, ValueError, RuntimeError) as error:
            raise HTTPException(status_code=415, detail="The file is not a valid NIfTI image.") from error
        if volume.ndim not in {2, 3} or not np.isfinite(volume).any():
            raise HTTPException(status_code=422, detail="NIfTI must contain a 2D image or 3D volume.")
        if volume.ndim == 3:
            slice_index = volume.shape[2] // 2
            pixels = volume[:, :, slice_index]
            depth = volume.shape[2]
        else:
            slice_index = 0
            pixels = volume
            depth = 1
        finite_pixels = pixels[np.isfinite(pixels)]
        low, high = np.percentile(finite_pixels, (0.5, 99.5))
        if high <= low:
            high = low + 1
        normalized = np.uint8(np.clip((pixels - low) * (255.0 / (high - low)), 0, 255))
        return Image.fromarray(normalized, mode="L"), {
            "format": "NIfTI",
            "modality": "MR",
            "rows": int(pixels.shape[0]),
            "columns": int(pixels.shape[1]),
            "pixel_spacing_mm": None,
            "slice_count": depth,
            "slice_index": slice_index,
            "anonymized": True,
        }

    image = image_from_bytes(contents)
    return image, {
        "format": "PNG" if file.content_type == "image/png" else "JPEG",
        "modality": None,
        "rows": image.height,
        "columns": image.width,
        "pixel_spacing_mm": None,
        "anonymized": True,
    }


def image_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def model_metadata() -> dict[str, Any] | None:
    if not MODEL_PATH.is_file() or not METADATA_PATH.is_file():
        return None
    try:
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return metadata if isinstance(metadata, dict) else None


def ml_dependencies_ready() -> bool:
    return importlib.util.find_spec("torch") is not None and importlib.util.find_spec("torchvision") is not None


def save_metrics(metrics: dict[str, Any]) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    temporary_path = METRICS_PATH.with_suffix(".tmp")
    temporary_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    temporary_path.replace(METRICS_PATH)


class CaseCreate(BaseModel):
    filename: str = Field(min_length=1, max_length=255)


class TrainRequest(BaseModel):
    dataset_path: str = Field(min_length=1, max_length=1024)
    epochs: int = Field(default=10, ge=5, le=50)
    architecture: Literal["compact_cnn", "resnet18_transfer"] = "resnet18_transfer"
    max_train_samples_per_class: int = Field(default=500, ge=1, le=10000)


@app.get("/api/status")
def get_status() -> dict[str, Any]:
    metadata = model_metadata()
    return {
        "api_ready": True,
        "model_ready": metadata is not None,
        "ml_dependencies_ready": ml_dependencies_ready(),
        "model_name": metadata.get("model_name") if metadata else None,
        "classes": metadata.get("classes", []) if metadata else [],
        "training": training_state["running"],
        "training_message": training_state["message"],
        "training_error": training_state["error"],
    }


@app.get("/api/dataset")
def dataset_status() -> dict[str, Any]:
    return get_dataset_state()


@app.get("/api/datasets/research")
def research_dataset_status() -> dict[str, Any]:
    from .research_sources import research_dataset_sources

    return {"sources": research_dataset_sources()}


@app.post("/api/dataset/prepare", status_code=202)
def start_dataset_preparation() -> JSONResponse:
    with training_lock:
        if training_state["running"] or dataset_state["preparing"]:
            raise HTTPException(status_code=409, detail="A training or dataset preparation job is already running.")
        if get_dataset_state()["ready"]:
            return JSONResponse(status_code=200, content={"message": "The public dataset is already prepared locally."})
        dataset_state.update(preparing=True, message="Downloading the CC0 dataset train and test splits…", error=None)
        thread = threading.Thread(target=run_dataset_preparation, daemon=True)
        thread.start()
    return JSONResponse(status_code=202, content={"message": "Dataset download and conversion started locally."})


def run_dataset_preparation() -> None:
    try:
        from .prepare_dataset import prepare_dataset

        metadata = prepare_dataset(PUBLIC_DATASET_DIR)
        dataset_state["message"] = f"Prepared {metadata['image_count']:,} labeled MRI images."
    except Exception as error:  # shown in the local UI for network or source-data troubleshooting
        dataset_state["error"] = str(error)
        dataset_state["message"] = f"Dataset preparation failed: {error}"
    finally:
        dataset_state["preparing"] = False


@app.get("/api/cases")
def list_cases() -> list[dict[str, str]]:
    with closing(connect_database()) as connection:
        rows = connection.execute(
            "SELECT id, filename, created_at, status FROM cases ORDER BY created_at DESC LIMIT 100"
        ).fetchall()
    return [dict(row) for row in rows]


@app.post("/api/cases", status_code=201)
def create_case(case: CaseCreate) -> dict[str, str]:
    filename = Path(case.filename.replace("\\", "/")).name
    record = {
        "id": uuid.uuid4().hex,
        "filename": filename,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "preview_only",
    }
    with closing(connect_database()) as connection:
        connection.execute(
            "INSERT INTO cases (id, filename, created_at, status) VALUES (?, ?, ?, ?)",
            (record["id"], record["filename"], record["created_at"], record["status"]),
        )
        connection.commit()
    return record


@app.post("/api/preprocess")
async def preprocess(
    file: UploadFile = File(...),
    auto_crop: bool = Form(True),
    noise_reduction: bool = Form(True),
    skull_stripping: bool = Form(False),
    role: str = Header(default="radiologist", alias="X-User-Role"),
) -> dict[str, Any]:
    image, scan_metadata = await read_scan_upload(file)
    original_size = image.size
    processed = preprocess_image(image, auto_crop, noise_reduction, skull_stripping)
    audit("preprocess", file.filename, role)
    steps = ["grayscale"]
    if noise_reduction:
        steps.append("median_denoise")
    if auto_crop:
        steps.append("auto_crop")
    if skull_stripping:
        steps.append("research_skull_mask")
    steps.extend(["resize_pad_224", "contrast_normalization"])
    return {
        "preview_data_url": image_data_url(processed),
        "width": original_size[0],
        "height": original_size[1],
        "scan_metadata": scan_metadata,
        "preprocessed_width": processed.width,
        "preprocessed_height": processed.height,
        "steps": steps,
        "quality_metrics": image_quality_metrics(image),
        "deidentification": {
            "applied": True,
            "stored_patient_identifiers": False,
            "removed_fields": ["name", "identifier", "birth_date", "address"],
        },
    }


@app.get("/api/metrics")
def get_metrics() -> dict[str, Any]:
    if not METRICS_PATH.is_file():
        return {"metrics": None}
    try:
        return {"metrics": json.loads(METRICS_PATH.read_text(encoding="utf-8"))}
    except (OSError, json.JSONDecodeError):
        return {"metrics": None}


@app.post("/api/train", status_code=202)
def start_training(request: TrainRequest) -> JSONResponse:
    if not ml_dependencies_ready():
        raise HTTPException(
            status_code=503,
            detail="Model training is not installed. Run 'pip install -r requirements-ml.txt' in the project environment, then restart the app.",
        )
    dataset_path = Path(request.dataset_path).expanduser().resolve()
    if not dataset_path.is_dir():
        raise HTTPException(status_code=400, detail="Dataset folder was not found on this machine.")
    train_path = dataset_path / "train" if (dataset_path / "train").is_dir() else dataset_path
    class_folders = [path for path in train_path.iterdir() if path.is_dir()]
    if len(class_folders) < 2:
        raise HTTPException(status_code=400, detail="Create at least two class subfolders inside the dataset folder.")
    with training_lock:
        if training_state["running"] or dataset_state["preparing"]:
            raise HTTPException(status_code=409, detail="A training or dataset preparation job is already running.")
        training_state.update(running=True, message="Preparing the local dataset…", error=None)
        thread = threading.Thread(
            target=run_training,
            args=(dataset_path, request.epochs, request.architecture, request.max_train_samples_per_class),
            daemon=True,
        )
        thread.start()
    return JSONResponse(
        status_code=202,
        content={"message": "Training started locally. This page will show progress."},
    )


def run_training(dataset_path: Path, epochs: int, architecture: str, max_train_samples_per_class: int) -> None:
    try:
        from .modeling import train_model

        def update(message: str) -> None:
            training_state["message"] = message

        metrics = train_model(
            dataset_path,
            epochs,
            architecture,
            update,
            max_samples_per_class=max_train_samples_per_class,
        )
        save_metrics(metrics)
        training_state["message"] = "Training and held-out evaluation completed."
    except Exception as error:  # surfaced in the local UI for setup/data troubleshooting
        training_state["error"] = str(error)
        training_state["message"] = f"Training failed: {error}"
    finally:
        training_state["running"] = False


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...), role: str = Header(default="radiologist", alias="X-User-Role")) -> dict[str, Any]:
    if model_metadata() is None:
        raise HTTPException(status_code=503, detail="No trained model is available. Train and validate a model first.")
    image, scan_metadata = await read_scan_upload(file)
    processed = preprocess_image(image)
    audit("analysis", file.filename, role)
    try:
        from .modeling import predict

        result = predict(processed, MODEL_PATH, METADATA_PATH)
        probabilities = result["probabilities"]
        ranked = sorted(probabilities.values(), reverse=True)
        result["uncertainty"] = {
            "flagged": result["confidence"] < 0.6 or (len(ranked) > 1 and ranked[0] - ranked[1] < 0.15),
            "reason": "Low confidence or narrow class margin" if result["confidence"] < 0.6 or (len(ranked) > 1 and ranked[0] - ranked[1] < 0.15) else "No heuristic uncertainty flag",
            "method": "confidence and top-two probability margin; not calibrated clinical uncertainty",
        }
        result["model_provenance"] = {
            "model_name": model_metadata().get("model_name"),
            "architecture": model_metadata().get("architecture"),
            "trained_at": model_metadata().get("trained_at"),
        }
        result["quality_metrics"] = image_quality_metrics(image)
        result["robustness_check"] = research_robustness_check(image, MODEL_PATH, METADATA_PATH)
        result["result_type"] = "research_prediction"
        result["disclaimer"] = (
            "Research decision support only. This model output is not a diagnosis, "
            "does not rule in or rule out disease, and must not guide patient care."
        )
        result["scan_metadata"] = scan_metadata
        result["segmentation"] = {
            "available": False,
            "mask_data_url": None,
            "volume_cm3": None,
            "reason": "No trained segmentation model is configured; classification heatmap is not a tumor contour.",
        }
        result["severity_grade"] = {
            "value": None,
            "status": "unavailable",
            "reason": "Severity and grade require a validated clinical model and complete MRI study.",
        }
        return result
    except ImportError as error:
        raise HTTPException(status_code=503, detail="Install the optional ML dependencies to run inference.") from error


@app.get("/api/audit")
def list_audit_events(role: str = Header(default="radiologist", alias="X-User-Role")) -> list[dict[str, str | None]]:
    if role != "admin":
        raise HTTPException(status_code=403, detail="Audit logs are restricted to administrators.")
    with closing(connect_database()) as connection:
        rows = connection.execute(
            "SELECT event_type, filename, role, created_at FROM audit_events ORDER BY created_at DESC LIMIT 100"
        ).fetchall()
    return [dict(row) for row in rows]
