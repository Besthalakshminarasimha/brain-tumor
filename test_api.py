import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.filewriter import dcmwrite
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

from backend.main import app
from backend.modeling import (
    expected_calibration_error,
    fit_temperature,
    metric_summary,
    negative_log_likelihood,
    select_training_indices,
    split_training_indices,
)
from backend.prepare_dataset import export_split


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "test.sqlite3"
        self.client = TestClient(app)
        self.database_patch = patch("backend.main.DATABASE", self.database_path)
        self.database_patch.start()

    def tearDown(self):
        self.database_patch.stop()
        self.temporary_directory.cleanup()

    def make_mr_dicom(self, frame_count=1, modality="MR"):
        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = MRImageStorage
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        dataset = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)
        dataset.SOPClassUID = MRImageStorage
        dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        dataset.Modality = modality
        dataset.Rows = 16
        dataset.Columns = 16
        dataset.SamplesPerPixel = 1
        dataset.PhotometricInterpretation = "MONOCHROME2"
        dataset.BitsAllocated = 16
        dataset.BitsStored = 16
        dataset.HighBit = 15
        dataset.PixelRepresentation = 0
        dataset.PixelSpacing = [0.7, 0.7]
        dataset.PatientName = "Example^Patient"
        if frame_count > 1:
            dataset.NumberOfFrames = frame_count
        dataset.PixelData = np.arange(256 * frame_count, dtype=np.uint16).tobytes()
        buffer = io.BytesIO()
        dcmwrite(buffer, dataset, enforce_file_format=True)
        return buffer.getvalue()

    def test_training_sample_cap_is_balanced_and_deterministic(self):
        samples = [(f"image-{index}.png", label) for label in range(3) for index in range(8)]

        selected = select_training_indices(samples, max_samples_per_class=3)
        repeated = select_training_indices(samples, max_samples_per_class=3)
        class_counts = {label: sum(samples[index][1] == label for index in selected) for label in range(3)}

        self.assertEqual(selected, repeated)
        self.assertEqual(class_counts, {0: 3, 1: 3, 2: 3})
        self.assertEqual(len(select_training_indices(samples, max_samples_per_class=20)), len(samples))

    def test_research_dataset_sources_include_supplied_references(self):
        response = self.client.get("/api/datasets/research")

        self.assertEqual(response.status_code, 200)
        sources = response.json()["sources"]
        source_urls = {source["url"] for source in sources}
        self.assertIn("https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset", source_urls)
        self.assertIn("https://ieeexplore.ieee.org/abstract/document/10762991", source_urls)

    def test_training_sample_cap_rejects_non_positive_values(self):
        with self.assertRaises(ValueError):
            select_training_indices([], max_samples_per_class=0)

    def test_validation_split_is_stratified_disjoint_and_deterministic(self):
        samples = [(f"image-{index}.png", label) for label in range(3) for index in range(12)]
        selected = list(range(len(samples)))

        training, validation = split_training_indices(samples, selected, validation_fraction=0.25)
        repeated_training, repeated_validation = split_training_indices(samples, selected, validation_fraction=0.25)
        validation_counts = {
            label: sum(samples[index][1] == label for index in validation)
            for label in range(3)
        }

        self.assertEqual((training, validation), (repeated_training, repeated_validation))
        self.assertFalse(set(training) & set(validation))
        self.assertEqual(sorted(training + validation), selected)
        self.assertEqual(validation_counts, {0: 3, 1: 3, 2: 3})

    def test_metric_summary_reports_per_class_support_and_recall(self):
        summary = metric_summary([0, 0, 1, 1], [0, 1, 1, 1], ["glioma", "meningioma"])

        self.assertEqual(summary["per_class"]["glioma"]["support"], 2)
        self.assertEqual(summary["per_class"]["glioma"]["recall"], 0.5)
        self.assertEqual(summary["per_class"]["meningioma"]["recall"], 1.0)

    def test_temperature_scaling_reduces_nll_for_overconfident_logits(self):
        logits = np.asarray([[8.0, 0.0], [0.0, 8.0], [8.0, 0.0], [0.0, 8.0]])
        targets = [0, 1, 1, 0]

        temperature, nll_before, nll_after = fit_temperature(logits, targets)
        calibration_error = expected_calibration_error(logits, targets, temperature)

        self.assertGreater(temperature, 1.0)
        self.assertLess(nll_after, nll_before)
        self.assertGreaterEqual(calibration_error, 0.0)
        self.assertLessEqual(calibration_error, 1.0)

    def test_preprocesses_a_valid_png_in_memory(self):
        image_buffer = io.BytesIO()
        Image.new("RGB", (320, 240), (120, 80, 40)).save(image_buffer, format="PNG")

        response = self.client.post(
            "/api/preprocess",
            files={"file": ("sample.png", image_buffer.getvalue(), "image/png")},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["preprocessed_width"], 224)
        self.assertEqual(response.json()["preprocessed_height"], 224)
        self.assertEqual(response.json()["steps"][-1], "contrast_normalization")
        self.assertIn("quality_metrics", response.json())
        self.assertIn("score", response.json()["quality_metrics"])

    def test_preprocesses_single_frame_mr_dicom_without_returning_phi(self):
        response = self.client.post(
            "/api/preprocess",
            files={"file": ("scan.dcm", self.make_mr_dicom(), "application/dicom")},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["scan_metadata"]["modality"], "MR")
        self.assertEqual(response.json()["scan_metadata"]["rows"], 16)
        self.assertNotIn("PatientName", str(response.json()))

    def test_rejects_multiframe_mr_dicom_with_clear_message(self):
        response = self.client.post(
            "/api/preprocess",
            files={"file": ("volume.dcm", self.make_mr_dicom(frame_count=2), "application/dicom")},
        )

        self.assertEqual(response.status_code, 422)
        self.assertIn("one 2D MR slice", response.json()["detail"])

    def test_rejects_non_mr_dicom(self):
        response = self.client.post(
            "/api/preprocess",
            files={"file": ("scan.dcm", self.make_mr_dicom(modality="CT"), "application/dicom")},
        )

        self.assertEqual(response.status_code, 415)
        self.assertIn("MR-modality", response.json()["detail"])

    def test_rejects_malformed_dicom_without_server_error(self):
        response = self.client.post(
            "/api/preprocess",
            files={"file": ("broken.dcm", b"not a dicom file", "application/dicom")},
        )

        self.assertEqual(response.status_code, 415)

    def test_rejects_non_image_upload(self):
        response = self.client.post(
            "/api/preprocess",
            files={"file": ("notes.txt", b"not an image", "text/plain")},
        )

        self.assertEqual(response.status_code, 415)

    def test_case_metadata_strips_path_components(self):
        response = self.client.post("/api/cases", json={"filename": "../scan.png"})

        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json()["filename"], "scan.png")
        self.assertEqual(len(self.client.get("/api/cases").json()), 1)

    def test_analysis_is_unavailable_without_trained_weights(self):
        missing_model = Path(self.temporary_directory.name) / "missing-model.pt"
        missing_metadata = Path(self.temporary_directory.name) / "missing-metadata.json"
        with patch("backend.main.MODEL_PATH", missing_model), patch("backend.main.METADATA_PATH", missing_metadata):
            response = self.client.post(
                "/api/analyze",
                files={"file": ("sample.png", b"unused", "image/png")},
            )

        self.assertEqual(response.status_code, 503)
        self.assertIn("No trained model", response.json()["detail"])

    def test_status_recognizes_valid_trained_model_metadata(self):
        model_path = Path(self.temporary_directory.name) / "model.pt"
        metadata_path = Path(self.temporary_directory.name) / "metadata.json"
        model_path.write_bytes(b"checkpoint")
        metadata_path.write_text('{"model_name": "Test CNN", "classes": ["a", "b"]}', encoding="utf-8")

        with patch("backend.main.MODEL_PATH", model_path), patch("backend.main.METADATA_PATH", metadata_path):
            response = self.client.get("/api/status")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["model_ready"])
        self.assertEqual(response.json()["model_name"], "Test CNN")
        self.assertEqual(response.json()["classes"], ["a", "b"])

    def test_analysis_labels_output_as_research_prediction(self):
        model_path = Path(self.temporary_directory.name) / "model.pt"
        metadata_path = Path(self.temporary_directory.name) / "metadata.json"
        model_path.write_bytes(b"checkpoint")
        metadata_path.write_text('{"model_name": "Test CNN", "classes": ["a", "b"]}', encoding="utf-8")
        image_buffer = io.BytesIO()
        Image.new("L", (16, 16), 120).save(image_buffer, format="PNG")

        with (
            patch("backend.main.MODEL_PATH", model_path),
            patch("backend.main.METADATA_PATH", metadata_path),
            patch(
                "backend.modeling.predict",
                return_value={
                    "predicted_class": "a",
                    "confidence": 0.8,
                    "probabilities": {"a": 0.8, "b": 0.2},
                    "tumor_present": None,
                    "heatmap_data_url": "data:image/png;base64,",
                    "explanation_note": "Exploratory only.",
                },
            ),
        ):
            response = self.client.post(
                "/api/analyze",
                files={"file": ("sample.png", image_buffer.getvalue(), "image/png")},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["result_type"], "research_prediction")
        self.assertIn("not a diagnosis", response.json()["disclaimer"])

    def test_training_rejects_unknown_architecture(self):
        response = self.client.post(
            "/api/train",
            json={"dataset_path": "unused", "epochs": 10, "architecture": "unknown_model"},
        )

        self.assertEqual(response.status_code, 422)

    def test_training_reports_missing_ml_dependencies_before_starting(self):
        dataset_path = Path(self.temporary_directory.name) / "dataset"
        (dataset_path / "class_a").mkdir(parents=True)
        (dataset_path / "class_b").mkdir()
        with patch("backend.main.ml_dependencies_ready", return_value=False):
            response = self.client.post(
                "/api/train",
                json={"dataset_path": str(dataset_path), "epochs": 10},
            )

        self.assertEqual(response.status_code, 503)
        self.assertIn("requirements-ml.txt", response.json()["detail"])

    def test_dataset_export_preserves_official_split_and_class_labels(self):
        output_path = Path(self.temporary_directory.name) / "dataset"
        records = [
            {"image": Image.new("RGB", (12, 12), (label * 40, 0, 0)), "label": label}
            for label in range(4)
        ]

        train_counts = export_split(records, "train", output_path)
        test_counts = export_split(records[:2], "test", output_path)

        self.assertEqual(sum(train_counts.values()), 4)
        self.assertEqual(sum(test_counts.values()), 2)
        self.assertTrue((output_path / "train" / "no_tumor" / "train_00002.png").is_file())
        self.assertTrue((output_path / "test" / "meningioma" / "test_00001.png").is_file())

    def test_dataset_status_is_not_ready_without_both_complete_splits(self):
        dataset_path = Path(self.temporary_directory.name) / "dataset"
        dataset_path.mkdir()
        (dataset_path / "dataset-source.json").write_text(
            '{"classes": ["glioma"], "image_count": 1}',
            encoding="utf-8",
        )
        (dataset_path / "train" / "glioma").mkdir(parents=True)
        with patch("backend.main.PUBLIC_DATASET_DIR", dataset_path):
            response = self.client.get("/api/dataset")

        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["ready"])

    def test_metric_summary_handles_empty_classes_and_invalid_predictions(self):
        empty_summary = metric_summary([], [], [])
        self.assertEqual(empty_summary["accuracy"], 0.0)
        self.assertEqual(empty_summary["precision"], 0.0)
        with self.assertRaises(ValueError):
            metric_summary([0], [2], ["glioma", "meningioma"])

    def test_temperature_and_bin_validation_reject_invalid_inputs(self):
        logits = np.asarray([[1.0, 2.0], [2.0, 1.0]], dtype=np.float64)
        with self.assertRaises(ValueError):
            negative_log_likelihood(logits, np.asarray([0, 1]), 0.0)
        with self.assertRaises(ValueError):
            expected_calibration_error(logits, [0, 1], 0.0, bins=10)
        with self.assertRaises(ValueError):
            expected_calibration_error(logits, [0, 1], 1.0, bins=0)


if __name__ == "__main__":
    unittest.main()