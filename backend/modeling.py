from __future__ import annotations

import base64
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image


def build_model(torch, number_of_classes: int, architecture: str, pretrained: bool = False):
    nn = torch.nn

    if architecture == "resnet18_transfer":
        from torchvision.models import ResNet18_Weights, resnet18

        model = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        original_conv = model.conv1
        grayscale_conv = nn.Conv2d(
            1,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False,
        )
        with torch.no_grad():
            grayscale_conv.weight.copy_(original_conv.weight.mean(dim=1, keepdim=True))
        model.conv1 = grayscale_conv
        model.fc = nn.Linear(model.fc.in_features, number_of_classes)
        return model

    class BrainCNN(nn.Module):
        def __init__(self, number_of_classes: int):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
            )
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.dropout = nn.Dropout(p=0.35)
            self.classifier = nn.Linear(128, number_of_classes)

        def forward(self, inputs):
            features = self.features(inputs)
            return self.classifier(self.dropout(self.pool(features).flatten(1)))

    if architecture != "compact_cnn":
        raise ValueError(f"Unsupported model architecture: {architecture}")
    return BrainCNN(number_of_classes)


def metric_summary(targets: list[int], predictions: list[int], classes: list[str]) -> dict:
    if len(targets) != len(predictions):
        raise ValueError("Targets and predictions must contain the same number of samples.")
    if not classes:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "classes": [],
            "confusion_matrix": [],
            "per_class": {},
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
        }

    matrix = [[0 for _ in classes] for _ in classes]
    for target, prediction in zip(targets, predictions):
        if target < 0 or prediction < 0 or target >= len(classes) or prediction >= len(classes):
            raise ValueError("Each target and prediction must reference a valid class index.")
        matrix[target][prediction] += 1

    total = sum(sum(row) for row in matrix)
    correct = sum(matrix[index][index] for index in range(len(classes)))
    precisions = []
    recalls = []
    f1_scores = []
    per_class = {}
    for index in range(len(classes)):
        true_positive = matrix[index][index]
        predicted_positive = sum(matrix[row][index] for row in range(len(classes)))
        actual_positive = sum(matrix[index])
        precision = true_positive / predicted_positive if predicted_positive else 0.0
        recall = true_positive / actual_positive if actual_positive else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)
        per_class[classes[index]] = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "support": actual_positive,
        }

    return {
        "accuracy": correct / total if total else 0.0,
        "precision": sum(precisions) / len(classes),
        "recall": sum(recalls) / len(classes),
        "f1_score": sum(f1_scores) / len(classes),
        "classes": classes,
        "confusion_matrix": matrix,
        "per_class": per_class,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }


def select_training_indices(
    samples: list[tuple[str, int]], max_samples_per_class: int, seed: int = 42
) -> list[int]:
    if max_samples_per_class < 1:
        raise ValueError("The per-class training sample limit must be at least one.")

    by_class: dict[int, list[int]] = defaultdict(list)
    for index, (_, label) in enumerate(samples):
        by_class[label].append(index)

    random_generator = random.Random(seed)
    selected_indices = []
    for indices in by_class.values():
        random_generator.shuffle(indices)
        selected_indices.extend(indices[:max_samples_per_class])
    return sorted(selected_indices)


def split_training_indices(
    samples: list[tuple[str, int]],
    selected_indices: list[int],
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    if not 0 < validation_fraction < 1:
        raise ValueError("The validation fraction must be between zero and one.")

    by_class: dict[int, list[int]] = defaultdict(list)
    for index in selected_indices:
        by_class[samples[index][1]].append(index)
    if len(by_class) < 2 or any(len(indices) < 2 for indices in by_class.values()):
        raise ValueError("Every class needs at least two selected images for a stratified validation split.")

    random_generator = random.Random(seed)
    training_indices = []
    validation_indices = []
    for indices in by_class.values():
        random_generator.shuffle(indices)
        validation_count = max(1, round(len(indices) * validation_fraction))
        if validation_count >= len(indices):
            raise ValueError("Each class must have at least one image for training and validation.")
        validation_indices.extend(indices[:validation_count])
        training_indices.extend(indices[validation_count:])
    return sorted(training_indices), sorted(validation_indices)


def negative_log_likelihood(logits: np.ndarray, targets: np.ndarray, temperature: float) -> float:
    if temperature <= 0:
        raise ValueError("The temperature must be positive.")
    if logits.ndim != 2:
        raise ValueError("Class logits must be a 2D array with one row per sample.")
    if targets.ndim != 1 or logits.shape[0] != len(targets):
        raise ValueError("Targets must be a 1D array matching the number of logit rows.")
    if not len(targets):
        raise ValueError("At least one sample is required to compute a loss.")
    if logits.shape[1] < 2 or np.any(targets < 0) or np.any(targets >= logits.shape[1]):
        raise ValueError("Each target must reference a valid class index.")
    if not np.isfinite(logits).all():
        raise ValueError("Logits must be finite numbers.")

    scaled_logits = logits / temperature
    maximum = scaled_logits.max(axis=1, keepdims=True)
    log_normalizer = maximum[:, 0] + np.log(np.exp(scaled_logits - maximum).sum(axis=1))
    target_logits = scaled_logits[np.arange(len(targets)), targets]
    return float(np.mean(log_normalizer - target_logits))


def fit_temperature(logits: np.ndarray, targets: list[int]) -> tuple[float, float, float]:
    logits = np.asarray(logits, dtype=np.float64)
    target_values = np.asarray(targets, dtype=np.int64)
    if (
        logits.ndim != 2
        or logits.shape[0] != len(target_values)
        or logits.shape[1] < 2
        or not len(target_values)
        or not np.isfinite(logits).all()
        or np.any(target_values < 0)
        or np.any(target_values >= logits.shape[1])
    ):
        raise ValueError("Calibration requires finite class logits and matching valid targets.")

    temperatures = np.exp(np.linspace(np.log(0.05), np.log(20.0), 401))
    losses = [negative_log_likelihood(logits, target_values, value) for value in temperatures]
    best_temperature = float(temperatures[int(np.argmin(losses))])
    return (
        best_temperature,
        negative_log_likelihood(logits, target_values, 1.0),
        negative_log_likelihood(logits, target_values, best_temperature),
    )


def expected_calibration_error(
    logits: np.ndarray, targets: list[int], temperature: float, bins: int = 10
) -> float:
    if temperature <= 0:
        raise ValueError("The temperature must be positive.")
    if bins <= 0:
        raise ValueError("The number of confidence bins must be at least one.")

    logits = np.asarray(logits, dtype=np.float64)
    targets_array = np.asarray(targets, dtype=np.int64)
    if logits.ndim != 2:
        raise ValueError("Class logits must be a 2D array with one row per sample.")
    if logits.shape[0] != len(targets_array):
        raise ValueError("Targets must match the number of logit rows.")
    if not len(targets_array):
        raise ValueError("At least one sample is required to compute calibration error.")
    if logits.shape[1] < 2 or np.any(targets_array < 0) or np.any(targets_array >= logits.shape[1]):
        raise ValueError("Each target must reference a valid class index.")
    if not np.isfinite(logits).all():
        raise ValueError("Logits must be finite numbers.")

    logits = logits / temperature
    shifted_logits = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted_logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    confidences = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == targets_array

    error = 0.0
    for bin_index in range(bins):
        lower = bin_index / bins
        upper = (bin_index + 1) / bins
        in_bin = (confidences >= lower) & (
            confidences <= upper if bin_index == bins - 1 else confidences < upper
        )
        if in_bin.any():
            error += float(in_bin.mean()) * abs(float(correct[in_bin].mean()) - float(confidences[in_bin].mean()))
    return error


def calibrate_saved_model(dataset_path: Path, model_path: Path, metadata_path: Path) -> dict:
    try:
        import torch
        from torch.utils.data import DataLoader, Subset
        from torchvision import datasets, transforms
    except ImportError as error:
        raise RuntimeError("Install PyTorch and torchvision from requirements-ml.txt to calibrate confidence.") from error

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    architecture = metadata.get("architecture", "compact_cnn")
    input_size = int(metadata.get("input_size", 128))
    mean, standard_deviation = ((0.449,), (0.226,)) if architecture == "resnet18_transfer" else ((0.5,), (0.5,))
    evaluation_transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, standard_deviation),
        ]
    )
    dataset_root = dataset_path / "train" if (dataset_path / "train").is_dir() else dataset_path
    dataset = datasets.ImageFolder(str(dataset_root), transform=evaluation_transform)
    if dataset.classes != metadata["classes"]:
        raise ValueError("Calibration dataset classes do not match the saved model metadata.")
    selected_indices = select_training_indices(
        dataset.samples, int(metadata.get("max_train_samples_per_class", len(dataset.samples)))
    )
    _, validation_indices = split_training_indices(dataset.samples, selected_indices)
    validation_loader = DataLoader(Subset(dataset, validation_indices), batch_size=48, shuffle=False, num_workers=0)

    model = build_model(torch, len(dataset.classes), architecture).to("cpu")
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()
    logits = []
    targets = []
    with torch.no_grad():
        for images, labels in validation_loader:
            logits.extend(model(images).cpu().numpy())
            targets.extend(labels.tolist())

    temperature, nll_before, nll_after = fit_temperature(np.asarray(logits), targets)
    expected_error = None
    official_test_path = dataset_path / "test"
    if official_test_path.is_dir():
        official_test_dataset = datasets.ImageFolder(str(official_test_path), transform=evaluation_transform)
        if official_test_dataset.classes != dataset.classes:
            raise ValueError("Training and official test folders must contain the same class names.")
        test_loader = DataLoader(official_test_dataset, batch_size=48, shuffle=False, num_workers=0)
        test_logits = []
        test_targets = []
        with torch.no_grad():
            for images, labels in test_loader:
                test_logits.extend(model(images).cpu().numpy())
                test_targets.extend(labels.tolist())
        expected_error = expected_calibration_error(np.asarray(test_logits), test_targets, temperature)

    metadata.update(
        calibration_temperature=temperature,
        calibration_method="scalar temperature scaling on training-only validation images",
        calibration_image_count=len(targets),
        calibration_nll_before=nll_before,
        calibration_nll_after=nll_after,
        calibration_test_image_count=len(test_targets) if official_test_path.is_dir() else None,
        expected_calibration_error=expected_error,
    )
    temporary_metadata = metadata_path.with_name("metadata.tmp")
    temporary_metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    temporary_metadata.replace(metadata_path)
    if expected_error is not None:
        metrics_path = metadata_path.with_name("metrics.json")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics["calibration_temperature"] = temperature
        metrics["calibration_nll_before"] = nll_before
        metrics["calibration_nll_after"] = nll_after
        metrics["expected_calibration_error"] = expected_error
        temporary_metrics = metrics_path.with_name("metrics.tmp")
        temporary_metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        temporary_metrics.replace(metrics_path)
    return metadata


def train_model(
    dataset_path: Path,
    epochs: int,
    architecture: str,
    update: Callable[[str], None],
    max_samples_per_class: int = 500,
) -> dict:
    try:
        import torch
        from torch.utils.data import DataLoader, Subset
        from torchvision import datasets, transforms
    except ImportError as error:
        raise RuntimeError(
            "PyTorch and torchvision are optional. Install them with: pip install -r requirements-ml.txt"
        ) from error

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    input_size = 160 if architecture == "resnet18_transfer" else 128
    if architecture == "resnet18_transfer":
        mean, standard_deviation = (0.449,), (0.226,)
    else:
        mean, standard_deviation = (0.5,), (0.5,)
    evaluation_transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, standard_deviation),
        ]
    )
    training_transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((input_size, input_size)),
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(degrees=0, translate=(0.04, 0.04), scale=(0.96, 1.04)),
            transforms.ToTensor(),
            transforms.Normalize(mean, standard_deviation),
        ]
    )
    training_path = dataset_path / "train" if (dataset_path / "train").is_dir() else dataset_path
    official_test_path = dataset_path / "test"
    training_dataset = datasets.ImageFolder(str(training_path), transform=training_transform)
    validation_dataset = datasets.ImageFolder(str(training_path), transform=evaluation_transform)
    classes = training_dataset.classes
    if len(classes) < 2 or not training_dataset.samples:
        raise ValueError("Training data must contain at least two non-empty class folders.")
    if validation_dataset.classes != classes:
        raise ValueError("Training and validation folders must contain the same class names.")
    selected_indices = select_training_indices(training_dataset.samples, max_samples_per_class)
    training_indices, validation_indices = split_training_indices(training_dataset.samples, selected_indices)
    train_loader = DataLoader(
        Subset(training_dataset, training_indices), batch_size=48, shuffle=True, num_workers=0
    )
    validation_loader = DataLoader(
        Subset(validation_dataset, validation_indices), batch_size=48, shuffle=False, num_workers=0
    )
    validation_count = len(validation_indices)
    validation_split = "stratified 80/20 split from the source training partition"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    if device.type == "cuda":
        train_loader = DataLoader(
            Subset(training_dataset, training_indices), batch_size=64, shuffle=True, num_workers=0
        )
        validation_loader = DataLoader(
            Subset(validation_dataset, validation_indices), batch_size=64, shuffle=False, num_workers=0
        )
    if official_test_path.is_dir():
        official_test_dataset = datasets.ImageFolder(str(official_test_path), transform=evaluation_transform)
        if official_test_dataset.classes != classes:
            raise ValueError("Training and official test folders must contain the same class names.")
        if not official_test_dataset.samples:
            raise ValueError("The official test split contains no supported images.")
    else:
        official_test_dataset = None
    update("Loading pretrained ResNet-18 weights…" if architecture == "resnet18_transfer" else "Initializing compact CNN…")
    model = build_model(
        torch,
        len(classes),
        architecture,
        pretrained=architecture == "resnet18_transfer",
    ).to(device)
    if architecture == "resnet18_transfer":
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.layer4.parameters():
            parameter.requires_grad = True
        for parameter in model.fc.parameters():
            parameter.requires_grad = True
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    learning_rate = 0.0001 if architecture == "resnet18_transfer" else 0.0007
    optimizer = torch.optim.AdamW(trainable_parameters, lr=learning_rate, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)
    loss_function = torch.nn.CrossEntropyLoss()
    best_loss = float("inf")
    best_state = None
    best_targets: list[int] = []
    best_predictions: list[int] = []
    stale_epochs = 0

    for epoch in range(epochs):
        model.train()
        if architecture == "resnet18_transfer":
            model.bn1.eval()
            model.layer1.eval()
            model.layer2.eval()
            model.layer3.eval()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(images), labels)
            loss.backward()
            optimizer.step()

        model.eval()
        validation_loss = 0.0
        targets = []
        predictions = []
        with torch.no_grad():
            for images, labels in validation_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                validation_loss += loss_function(logits, labels).item() * labels.size(0)
                targets.extend(labels.cpu().tolist())
                predictions.extend(logits.argmax(dim=1).cpu().tolist())
        validation_loss /= validation_count
        scheduler.step(validation_loss)
        update(f"Epoch {epoch + 1}/{epochs} · validation loss {validation_loss:.4f} · {device.type.upper()}")

        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= 6:
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid model checkpoint.")

    model.load_state_dict(best_state)
    validation_logits = []
    validation_targets = []
    model.eval()
    with torch.no_grad():
        for images, labels in validation_loader:
            validation_logits.extend(model(images.to(device)).cpu().numpy())
            validation_targets.extend(labels.tolist())
    calibration_temperature, calibration_nll_before, calibration_nll_after = fit_temperature(
        np.asarray(validation_logits), validation_targets
    )

    evaluation_loader = validation_loader
    evaluation_count = validation_count
    evaluation_split = validation_split
    if official_test_dataset is not None:
        evaluation_loader = DataLoader(
            official_test_dataset,
            batch_size=64 if device.type == "cuda" else 48,
            shuffle=False,
            num_workers=0,
        )
        evaluation_count = len(official_test_dataset)
        evaluation_split = "official source test split; patient-level separation is not documented by the source"
    model.eval()
    evaluation_targets = []
    evaluation_predictions = []
    evaluation_logits = []
    with torch.no_grad():
        for images, labels in evaluation_loader:
            images, labels = images.to(device), labels.to(device)
            batch_logits = model(images)
            evaluation_logits.extend(batch_logits.cpu().numpy())
            evaluation_predictions.extend(batch_logits.argmax(dim=1).cpu().tolist())
            evaluation_targets.extend(labels.cpu().tolist())
    if len(evaluation_targets) != evaluation_count:
        raise RuntimeError("Final evaluation did not include every expected image.")

    output_dir = Path(__file__).resolve().parent / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "model.pt"
    temporary_checkpoint = output_dir / "model.tmp"
    torch.save(best_state, temporary_checkpoint)
    temporary_checkpoint.replace(checkpoint_path)
    metrics = metric_summary(evaluation_targets, evaluation_predictions, classes)
    metrics["evaluation_split"] = evaluation_split
    metrics["validation_loss"] = best_loss
    metrics["calibration_temperature"] = calibration_temperature
    metrics["calibration_nll_before"] = calibration_nll_before
    metrics["calibration_nll_after"] = calibration_nll_after
    metrics["expected_calibration_error"] = expected_calibration_error(
        np.asarray(evaluation_logits), evaluation_targets, calibration_temperature
    )
    metadata = {
        "model_name": "Brain MRI ResNet-18" if architecture == "resnet18_transfer" else "Brain MRI CNN",
        "classes": classes,
        "input_size": input_size,
        "architecture": architecture,
        "max_train_samples_per_class": max_samples_per_class,
        "training_image_count": len(training_indices),
        "validation_image_count": len(validation_indices),
        "validation_split": validation_split,
        "evaluation_split": evaluation_split,
        "calibration_temperature": calibration_temperature,
        "calibration_method": "scalar temperature scaling on training-only validation images",
        "calibration_image_count": len(validation_targets),
        "calibration_nll_before": calibration_nll_before,
        "calibration_nll_after": calibration_nll_after,
        "patient_level_separation_documented": False if official_test_dataset is not None else None,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path = output_dir / "metadata.json"
    temporary_metadata = output_dir / "metadata.tmp"
    temporary_metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    temporary_metadata.replace(metadata_path)
    return metrics


def predict(image: Image.Image, model_path: Path, metadata_path: Path) -> dict:
    try:
        import torch
        from torchvision.transforms import Compose, Normalize, Resize, ToTensor
    except ImportError as error:
        raise RuntimeError("Install PyTorch and torchvision from requirements-ml.txt to run inference.") from error

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    classes = metadata["classes"]
    architecture = metadata.get("architecture", "compact_cnn")
    input_size = metadata.get("input_size", 224)
    model = build_model(torch, len(classes), architecture)
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    transform = Compose([Resize((input_size, input_size)), ToTensor(), Normalize((0.5,), (0.5,))])
    inputs = transform(image.convert("L")).unsqueeze(0)
    activation = None
    gradient = None

    def save_activation(_module, _input, output):
        nonlocal activation
        activation = output
        output.register_hook(lambda grad: set_gradient(grad))

    def set_gradient(value):
        nonlocal gradient
        gradient = value

    target_layer = model.features[-1] if architecture == "compact_cnn" else model.layer4[-1]
    hook = target_layer.register_forward_hook(save_activation)
    model.zero_grad(set_to_none=True)
    logits = model(inputs)
    class_index = int(logits.argmax(dim=1).item())
    logits[0, class_index].backward()
    hook.remove()

    calibration_temperature = float(metadata.get("calibration_temperature", 1.0))
    if not np.isfinite(calibration_temperature) or calibration_temperature <= 0:
        calibration_temperature = 1.0
    probabilities = torch.softmax(logits.detach() / calibration_temperature, dim=1)[0].tolist()
    if activation is None or gradient is None:
        raise RuntimeError("Could not generate an explanation map for this model.")
    weights = gradient.mean(dim=(2, 3), keepdim=True)
    heatmap = torch.relu((weights * activation).sum(dim=1, keepdim=True))[0, 0]
    heatmap = heatmap / (heatmap.max() + 1e-8)
    heatmap_pixels = np.uint8(heatmap.detach().numpy() * 255)
    heat_image = Image.fromarray(heatmap_pixels, mode="L").resize(image.size, Image.Resampling.BILINEAR)

    base = np.asarray(image.convert("L"), dtype=np.float32)
    heat = np.asarray(heat_image, dtype=np.float32) / 255.0
    colored = np.stack((255 * heat, 180 * np.maximum(0, 1 - np.abs(heat - 0.55) * 3), 30 * (1 - heat)), axis=-1)
    gray_rgb = np.stack((base, base, base), axis=-1)
    overlay = np.uint8(np.clip(gray_rgb * 0.58 + colored * 0.42, 0, 255))
    overlay_image = Image.fromarray(overlay, mode="RGB")
    buffer = __import__("io").BytesIO()
    overlay_image.save(buffer, format="PNG")
    heatmap_url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")

    predicted_class = classes[class_index]
    return {
        "predicted_class": predicted_class,
        "confidence": probabilities[class_index],
        "probabilities": dict(zip(classes, probabilities)),
        "tumor_present": None if "no_tumor" not in classes else predicted_class != "no_tumor",
        "heatmap_data_url": heatmap_url,
        "explanation_note": "Grad-CAM is an exploratory visualization and is not a segmentation or a clinical finding.",
    }