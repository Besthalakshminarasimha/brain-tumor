from __future__ import annotations

import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageOps

DATASET_ID = "Dharamanand/BrainTumorMRI"
DATASET_URL = "https://huggingface.co/datasets/Dharamanand/BrainTumorMRI"
DATASET_LICENSE = "CC0-1.0"
CITATION = "Dharamanand. BrainTumorMRI dataset. Hugging Face Datasets, 2024."
OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "brain-tumor-mri-cc0"
CLASS_NAMES = {
    0: "glioma",
    1: "meningioma",
    2: "no_tumor",
    3: "pituitary",
}


def export_split(records: Iterable[dict[str, Any]], split: str, output_dir: Path) -> dict[str, int]:
    if split not in {"train", "test"}:
        raise ValueError("Only the source train and test splits are supported.")

    counts = {name: 0 for name in CLASS_NAMES.values()}
    for index, record in enumerate(records):
        try:
            label = int(record["label"])
            class_name = CLASS_NAMES[label]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid source record at {split} index {index}.") from error
        target_dir = output_dir / split / class_name
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / f"{split}_{index:05d}.png"
        if target_path.is_file():
            try:
                with Image.open(target_path) as existing_image:
                    existing_image.verify()
                counts[class_name] += 1
                continue
            except OSError:
                target_path.unlink(missing_ok=True)

        try:
            source_image = record["image"]
            if isinstance(source_image, dict):
                image_bytes = source_image.get("bytes")
                if image_bytes is None:
                    raise ValueError("The source row does not contain decoded image bytes.")
                source_image = Image.open(io.BytesIO(image_bytes))
        except (KeyError, OSError, ValueError) as error:
            raise ValueError(f"Source image at {split} index {index} could not be decoded.") from error
        if not isinstance(source_image, Image.Image):
            raise ValueError(f"Source image at {split} index {index} could not be decoded.")

        image = ImageOps.exif_transpose(source_image).convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", optimize=False)
        temporary_path = target_path.with_name(target_path.name + ".tmp")
        temporary_path.write_bytes(buffer.getvalue())
        temporary_path.replace(target_path)
        counts[class_name] += 1
    return counts


def prepare_dataset(output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    metadata_path = output_dir / "dataset-source.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("dataset_id") == DATASET_ID and all(
            (output_dir / split / class_name).is_dir()
            for split in ("train", "test")
            for class_name in CLASS_NAMES.values()
        ):
            return metadata

    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError("Dataset download requires the optional 'datasets' package. Install requirements-ml.txt first.") from error

    split_counts = {}
    for split in ("train", "test"):
        from datasets import Image as DatasetImage

        source_split = load_dataset(DATASET_ID, split=split, streaming=True)
        source_split = source_split.cast_column("image", DatasetImage(decode=False))
        split_counts[split] = export_split(source_split, split, output_dir)

    metadata = {
        "dataset_id": DATASET_ID,
        "source_url": DATASET_URL,
        "license": DATASET_LICENSE,
        "citation": CITATION,
        "classes": list(CLASS_NAMES.values()),
        "split_counts": split_counts,
        "image_count": sum(sum(counts.values()) for counts in split_counts.values()),
        "patient_level_split": False,
        "prepared_at": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


if __name__ == "__main__":
    result = prepare_dataset()
    print(json.dumps(result, indent=2))
