# Neuro Imaging Lab

A local-first research prototype for reviewing 2D brain MRI images, preprocessing scans, training a CNN from a labeled dataset, and inspecting held-out metrics and Grad-CAM visualizations.

**[Open the website](https://besthalakshminarasimha.github.io/brain-tumor/)**

**This is not a medical device and does not provide a diagnosis.** It ships without pretrained weights. Predictions are unavailable until a model has been trained locally, and training metrics do not establish clinical validity. Do not use the output to make patient-care decisions.

## Run locally

Requirements: Node.js 20.19+ and Python 3.10+.

1. Create and activate a Python virtual environment.
2. Install API dependencies with `pip install -r requirements.txt`.
3. Install frontend dependencies with `npm install`.
4. Run `npm run dev:full` and open the URL printed by Vite.

The command starts the UI and API together. Vite proxies `/api` requests to FastAPI on `127.0.0.1:8000`. The backend binds to loopback; do not expose it to a network without adding authentication and a security review.

Run the API tests with `npm run test:api`. Run the frontend production build with `npm run build`.

## Optional model training

Install the extra ML dependencies with `pip install -r requirements-ml.txt`. Organize images in class-named folders, for example:

```text
brain-mri/
  glioma/
  meningioma/
  no_tumor/
```

Use JPG or PNG images and provide at least two images per class. In **Model training**, enter the dataset folder path and start training. The default uses pretrained ResNet-18 transfer learning, mild training-only image augmentation, and a deterministic stratified validation split drawn only from the training partition. The official test partition is evaluated once after checkpoint selection and is not used to tune epochs. The service cannot infer patient identity, so prepare datasets with patient-level separation before training to prevent leakage. Evaluation reports include macro metrics, per-class precision/recall/F1, support counts, and a confusion matrix. Model checkpoints and metrics are written under `backend/artifacts/`.

The compact grayscale CNN remains available for quick experiments. ResNet-18 downloads ImageNet initialization weights the first time it is selected; this is only an initialization and still requires MRI-specific training and independent validation. Inputs are resized to 160×160 for ResNet-18 and 128×128 for the compact CNN. A Grad-CAM overlay is generated during inference; it is an exploratory explanation, not a tumor segmentation.

Inference returns a **research prediction** with class probabilities and an exploratory Grad-CAM overlay. It is not a diagnosis, does not rule in or rule out disease, and must not be used to guide patient care. Always use the official radiology report and qualified clinical review.

The review workspace also provides a reproducible research-quality screen (contrast, sharpness, entropy, foreground coverage, and heuristic flags), a low-confidence/top-two-margin uncertainty flag, model provenance, structured sequence/laterality/artifact annotations, and a local JSON research-review export. These are engineering and research aids, not clinical image-quality or uncertainty determinations.

It also runs a research robustness stress test across mild intensity perturbations and reports prediction agreement. This is a reproducibility signal, not clinical validation; a model that changes under small perturbations should be treated as requiring further research review.

### Ingestion and clinical-workflow prototype features

- Study review accepts single-frame MR DICOM, NIfTI (`.nii`/`.nii.gz`), JPG, and PNG files by browse or drag-and-drop. NIfTI volumes are previewed slice-by-slice; the current API selects the middle slice for preprocessing.
- DICOM/NIfTI identifiers are not returned to the browser or written to the local case database. The API reports de-identification as applied and records only a sanitized filename, timestamp, role, and event type in its local audit table. This is a technical safeguard, not a complete HIPAA/GDPR compliance program.
- Preprocessing controls expose auto-crop, median noise reduction, and a clearly labeled research skull mask. These are not validated skull stripping or clinical preprocessing algorithms.
- The dashboard provides a local history timeline, structured radiologist notes, role labeling (Admin/Radiologist/Patient), audit-log protection for the Admin role, and browser print-to-PDF for the current research review.
- The existing trained classifier and Grad-CAM explanation remain available when a local model is trained. Segmentation masks, tumor volume, and severity/grade are explicitly reported as unavailable until a validated segmentation/ grading model is configured; a Grad-CAM heatmap is not a tumor contour.

### Public sample dataset

The **Model training** screen can download the public [Dharamanand/BrainTumorMRI](https://huggingface.co/datasets/Dharamanand/BrainTumorMRI) dataset and convert its CC0-1.0 train and test splits to local PNG class folders. The importer keeps the supplied test split out of training, records source attribution under `backend/data/brain-tumor-mri-cc0/dataset-source.json`, and writes only the MRI pixels and class labels needed by `ImageFolder`. The dataset card does not provide patient identifiers or document patient-level split construction, so test metrics may be optimistic if related slices appear across splits. This is a research/demo dataset, not a clinical validation set.

The **Model training** screen also records the supplied [Kaggle Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) and [IEEE reference](https://ieeexplore.ieee.org/abstract/document/10762991) as research sources. The Kaggle source is described as a four-class dataset with 7,023 images (5,712 train and 1,311 test); download it through Kaggle or provide an explicit local export, then preserve its train/test directories when training. The public IEEE page did not expose a verifiable abstract or download during implementation, so the app intentionally displays it as a reference-only source and does not invent dataset counts, labels, or license terms. Verify the paper and any accompanying repository before using it.

## Privacy and limitations

- Uploaded image bytes are processed in memory and are not written to disk by the API.
- The local SQLite database stores only filenames, timestamps and preview status. Filenames can contain identifying information; use anonymized names.
- The application has no authentication or multi-user access controls. Keep the API bound to localhost.
- The included CNN is a training example, not a clinically validated model. Performance depends on dataset quality, labeling, acquisition protocols and independent external validation.
- Single-frame MR DICOM, JPG and PNG are supported. Multi-frame DICOM, series/3D volumes, PACS, segmentation, calibration, audit trails and clinical-system integration are not implemented. Some compressed DICOM transfer syntaxes may require optional pydicom decoding plugins.

## Technical references

- [DICOM Standard](https://www.dicomstandard.org/)
- [pydicom pixel data guide](https://pydicom.github.io/pydicom/stable/tutorials/pixel_data/introduction.html)
- [FDA Clinical Decision Support Software guidance](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/clinical-decision-support-software)
