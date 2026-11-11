from __future__ import annotations

import json
from pathlib import Path

import fitz


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = Path(__file__).resolve().parent / "Neuro_Imaging_Lab_Presentation.pdf"
METRICS = json.loads((ROOT / "backend" / "artifacts" / "metrics.json").read_text(encoding="utf-8"))
METADATA = json.loads((ROOT / "backend" / "artifacts" / "metadata.json").read_text(encoding="utf-8"))
DATASET = json.loads(
    (ROOT / "backend" / "data" / "brain-tumor-mri-cc0" / "dataset-source.json").read_text(
        encoding="utf-8"
    )
)

WIDTH = 960
HEIGHT = 540
INK = (0.10, 0.16, 0.15)
FOREST = (0.10, 0.29, 0.25)
MINT = (0.23, 0.63, 0.50)
CORAL = (0.88, 0.40, 0.32)
GOLD = (0.86, 0.66, 0.29)
PAPER = (0.97, 0.96, 0.92)
WHITE = (1.0, 1.0, 1.0)
PALE_GREEN = (0.89, 0.94, 0.90)
PALE_CORAL = (0.97, 0.90, 0.87)
MUTED = (0.38, 0.45, 0.43)
RULE = (0.84, 0.87, 0.83)


def rect(page: fitz.Page, x: float, y: float, w: float, h: float, color, radius: float = 0):
    shape = page.new_shape()
    bounds = fitz.Rect(x, y, x + w, y + h)
    shape.draw_rect(bounds)
    shape.finish(color=color, fill=color, width=0)
    shape.commit()


def line(page: fitz.Page, x1: float, y1: float, x2: float, y2: float, color=RULE, width=1):
    page.draw_line((x1, y1), (x2, y2), color=color, width=width)


def text(
    page: fitz.Page,
    x: float,
    y: float,
    w: float,
    h: float,
    value: str,
    size: float = 14,
    color=INK,
    font: str = "helv",
    align: int = 0,
    line_height: float = 1.16,
):
    return page.insert_textbox(
        fitz.Rect(x, y, x + w, y + h),
        value,
        fontsize=size,
        fontname=font,
        color=color,
        align=align,
        lineheight=line_height,
    )


def base_slide(doc: fitz.Document, number: int, section: str, title: str, subtitle: str = ""):
    page = doc.new_page(width=WIDTH, height=HEIGHT)
    rect(page, 0, 0, WIDTH, HEIGHT, PAPER)
    rect(page, 0, 0, WIDTH, 8, MINT)
    text(page, 42, 26, 500, 20, section.upper(), 9, FOREST, "hebo")
    text(page, 42, 54, 870, 45, title, 28, INK, "hebo")
    if subtitle:
        text(page, 42, 99, 860, 30, subtitle, 12, MUTED)
    line(page, 42, 507, 918, 507, RULE, 0.8)
    text(page, 42, 514, 700, 14, "NEURO IMAGING LAB  /  RESEARCH PROTOTYPE  /  2026-11-02", 8, MUTED)
    text(page, 868, 512, 48, 16, f"{number:02d}", 9, FOREST, "hebo", align=2)
    return page


def card(page: fitz.Page, x: float, y: float, w: float, h: float, fill=WHITE, stroke=RULE):
    shape = page.new_shape()
    shape.draw_rect(fitz.Rect(x, y, x + w, y + h))
    shape.finish(color=stroke, fill=fill, width=0.8)
    shape.commit()


def metric(page: fitz.Page, x: float, y: float, label: str, value: str, detail: str, accent=MINT):
    card(page, x, y, 200, 104)
    rect(page, x, y, 5, 104, accent)
    text(page, x + 17, y + 13, 170, 15, label.upper(), 9, MUTED, "hebo")
    page.insert_text((x + 17, y + 69), value, fontsize=28, fontname="hebo", color=INK)
    text(page, x + 17, y + 75, 170, 18, detail, 9, MUTED)


def add_notes_page(doc: fitz.Document, page_number: int, heading: str, notes: list[tuple[str, str]]):
    page = base_slide(doc, page_number, "Presenter notes", heading, "Suggested plain-language narration; adapt to your audience.")
    y = 145
    for title, narration in notes:
        text(page, 48, y, 850, 19, title, 12, FOREST, "hebo")
        text(page, 48, y + 23, 850, 52, narration, 10.5, INK, line_height=1.2)
        y += 91
    return page


def build_pdf() -> None:
    doc = fitz.open()
    doc.set_metadata(
        {
            "title": "Neuro Imaging Lab - Research Prototype Presentation",
            "author": "Neuro Imaging Lab",
            "subject": "Brain MRI research workflow, model training, evaluation, and limitations",
            "keywords": "MRI, research prototype, ResNet-18, model evaluation",
        }
    )
    slides: list[tuple[str, str]] = []

    page = doc.new_page(width=WIDTH, height=HEIGHT)
    rect(page, 0, 0, WIDTH, HEIGHT, INK)
    rect(page, 0, 0, 12, HEIGHT, MINT)
    text(page, 54, 54, 410, 20, "RESEARCH PROTOTYPE  /  PRESENTATION", 10, (0.55, 0.82, 0.70), "hebo")
    text(page, 54, 122, 540, 100, "Neuro Imaging\nLab", 47, WHITE, "hebo", line_height=1.0)
    text(page, 57, 245, 475, 62, "A local-first brain MRI review and\nresearch classification workflow", 20, (0.84, 0.90, 0.86))
    text(page, 57, 355, 410, 42, "What it does. How it was evaluated.\nWhat its results do not prove.", 13, (0.69, 0.77, 0.73))
    # A simple process motif reinforces the workflow without implying anatomy.
    centers = [(673, 174), (788, 250), (673, 326)]
    labels = [("01", "IMAGE"), ("02", "MODEL"), ("03", "REVIEW")]
    for (cx, cy), (num, label) in zip(centers, labels):
        page.draw_circle((cx, cy), 42, color=(0.24, 0.43, 0.37), fill=(0.14, 0.34, 0.29), width=1.2)
        text(page, cx - 25, cy - 15, 50, 20, num, 18, (0.55, 0.82, 0.70), "hebo", align=1)
        text(page, cx - 48, cy + 49, 96, 17, label, 9, WHITE, "hebo", align=1)
    line(page, 673, 216, 788, 208, (0.42, 0.62, 0.54), 1.4)
    line(page, 788, 292, 673, 284, (0.42, 0.62, 0.54), 1.4)
    text(page, 54, 500, 830, 18, "NOT A MEDICAL DEVICE  /  NOT FOR DIAGNOSIS OR TREATMENT DECISIONS", 9, (0.93, 0.70, 0.52), "hebo")
    text(page, 872, 500, 40, 18, "01", 9, WHITE, "hebo", align=2)
    slides.append(("Title", "Introduce this as a research prototype for exploring a local MRI review workflow. It is not a medical device and does not make clinical decisions."))

    page = base_slide(doc, 2, "01 / Purpose", "A focused research workflow", "The tool organizes image review and model experiments; it does not replace diagnostic interpretation.")
    cards = [
        (48, "INPUT", "Single-frame MR DICOM, JPG, or PNG", MINT),
        (270, "PREPARE", "Grayscale, denoise, resize, normalize", GOLD),
        (492, "MODEL", "Four-class research prediction", CORAL),
        (714, "REVIEW", "Probabilities and exploratory Grad-CAM", FOREST),
    ]
    for x, label, desc, accent in cards:
        card(page, x, 178, 194, 139)
        rect(page, x, 178, 194, 6, accent)
        text(page, x + 14, 198, 165, 18, label, 9, accent, "hebo")
        text(page, x + 14, 226, 164, 64, desc, 15, INK, "hebo", line_height=1.12)
    for x in (245, 467, 689):
        text(page, x, 230, 20, 30, ">", 20, MUTED, "hebo", align=1)
    rect(page, 48, 355, 860, 71, PALE_CORAL, 4)
    text(page, 67, 372, 825, 41, "The model reports one of four dataset labels. It does not locate or segment a tumor, interpret a full MRI series, or establish a diagnosis.", 13, INK, "hebo")
    slides.append(("Purpose", "Walk through the four stages from a single image to a reviewable model output. Emphasize that the prototype classifies an image into one of four labels and does not perform tumor segmentation or radiology interpretation."))

    page = base_slide(doc, 3, "02 / System", "Local-first architecture", "Image bytes are processed locally; the database stores study metadata, not uploaded image files.")
    blocks = [
        (48, "WEB APP", "React + TypeScript\nUpload, review controls,\nresults, evaluation", MINT),
        (272, "LOCAL API", "FastAPI\nPreprocessing, training,\ninference", GOLD),
        (496, "MODEL", "PyTorch ResNet-18\nSingle-channel MRI input\nCalibrated probabilities", CORAL),
        (720, "LOCAL DATA", "SQLite metadata\nLocal model artifacts\nCC0 sample dataset", FOREST),
    ]
    for x, label, desc, accent in blocks:
        card(page, x, 188, 190, 160)
        rect(page, x + 14, 204, 34, 4, accent)
        text(page, x + 14, 224, 160, 20, label, 10, accent, "hebo")
        text(page, x + 14, 258, 160, 72, desc, 12, INK, line_height=1.2)
    for x in (243, 467, 691):
        line(page, x, 269, x + 34, 269, MUTED, 1.3)
    text(page, 52, 382, 850, 45, "The service binds to loopback for local development. It has no authentication or multi-user access controls; do not expose it to a network without a security review.", 11, MUTED)
    slides.append(("Architecture", "Explain that the browser, API, model, and data artifacts run on the user’s machine. Uploaded images are handled in memory, while case history is limited to filenames and timestamps. The current development service should remain local."))

    page = base_slide(doc, 4, "03 / Workflow", "From scan to review", "A transparent processing path helps distinguish image handling from model interpretation.")
    steps = [
        (48, "01", "Select", "One 2D MRI image"),
        (266, "02", "Preprocess", "Denoise + normalize"),
        (484, "03", "Infer", "Class probabilities"),
        (702, "04", "Inspect", "Output + Grad-CAM"),
    ]
    for x, num, title, desc in steps:
        page.draw_circle((x + 75, 220), 30, color=MINT, fill=PALE_GREEN, width=1.2)
        text(page, x + 50, 210, 50, 20, num, 14, FOREST, "hebo", align=1)
        text(page, x, 270, 150, 24, title, 15, INK, "hebo", align=1)
        text(page, x - 5, 301, 160, 40, desc, 11, MUTED, align=1)
    for x in (202, 420, 638):
        line(page, x, 220, x + 42, 220, RULE, 1.8)
    rect(page, 48, 373, 860, 58, PALE_GREEN, 4)
    text(page, 66, 390, 822, 27, "Grad-CAM is an exploratory attention visualization - not a tumor boundary, segmentation, or clinical finding.", 12, FOREST, "hebo")
    slides.append(("Workflow", "Describe the user path: select a supported image, preprocess it, run inference, and review probabilities. Grad-CAM highlights image regions that influenced the network but is not a tumor mask."))

    page = base_slide(doc, 5, "04 / Data", "A labeled public sample dataset", "The bundled CC0 dataset contains four image classes and separate source train/test folders.")
    split_counts = DATASET["split_counts"]
    class_names = DATASET["classes"]
    chart_x, chart_y, chart_w = 225, 175, 530
    max_count = max(split_counts["train"].values())
    for index, name in enumerate(class_names):
        y = chart_y + index * 54
        train_count = split_counts["train"][name]
        test_count = split_counts["test"][name]
        train_width = chart_w * train_count / max_count
        test_width = chart_w * test_count / max_count
        text(page, 50, y + 4, 160, 20, name.replace("_", " "), 11, INK, "hebo")
        rect(page, chart_x, y, train_width, 17, MINT, 3)
        rect(page, chart_x, y + 20, test_width, 9, CORAL, 3)
        text(page, chart_x + chart_w + 14, y - 1, 150, 18, f"train {train_count:,}", 9, FOREST, "hebo")
        text(page, chart_x + chart_w + 14, y + 17, 150, 18, f"test {test_count:,}", 9, CORAL, "hebo")
    text(page, 52, 405, 860, 24, "7,023 images total  |  5,969 source train  |  1,054 source test  |  CC0-1.0", 12, FOREST, "hebo")
    text(page, 52, 443, 860, 36, "Source: Dharamanand/BrainTumorMRI on Hugging Face. Its card does not provide patient identifiers or document patient-level split construction.", 9.5, MUTED)
    slides.append(("Dataset", "The app uses the 7,023-image public dataset shown here. The license is listed as CC0-1.0. Explain that class labels and image-level splits are available, but the source does not establish that patients are separated across splits."))

    page = base_slide(doc, 6, "05 / Training", "How the current model was trained", "ResNet-18 transfer learning, bounded local compute, and validation separated from final test scoring.")
    training_rows = [
        ("Architecture", "ImageNet-initialized ResNet-18; grayscale first layer; final block and classifier fine-tuned"),
        ("Training cap", "500 sampled images per class; 2,000 selected total"),
        ("Fit / validation", "1,600 fit images + 400 stratified validation images from the source training partition"),
        ("Augmentation", "Training-only small rotation, translation, and scale changes"),
        ("Selection", "Best checkpoint chosen by validation loss; 10 epochs on CPU"),
        ("Final test", "All 1,054 source test images scored after checkpoint selection"),
    ]
    y = 156
    for label, detail in training_rows:
        card(page, 50, y, 858, 44)
        text(page, 64, y + 13, 132, 18, label.upper(), 9, FOREST, "hebo")
        text(page, 205, y + 10, 684, 25, detail, 10.5, INK)
        y += 53
    text(page, 53, 482, 855, 15, "This run used the source test split only after checkpoint selection; patient-level separation remains undocumented.", 9, CORAL, "hebo")
    slides.append(("Training", "The training cap selected up to 500 images per class, then the selected images were split into 1,600 fitting and 400 validation examples. The test partition was not used to select epochs in this run. The older compact-CNN result is not a fair baseline because it was selected against test loss."))

    page = base_slide(doc, 7, "06 / Results", "Held-out evaluation", "Official source test partition; 1,054 images. Results are dataset-specific, not evidence of clinical accuracy.")
    metric(page, 50, 160, "Accuracy", f"{METRICS['accuracy'] * 100:.1f}%", "1,007 / 1,054 correct", MINT)
    metric(page, 270, 160, "Macro precision", f"{METRICS['precision'] * 100:.1f}%", "Unweighted class average", GOLD)
    metric(page, 490, 160, "Macro recall", f"{METRICS['recall'] * 100:.1f}%", "Unweighted class average", CORAL)
    metric(page, 710, 160, "Macro F1", f"{METRICS['f1_score'] * 100:.1f}%", "Unweighted class average", FOREST)
    text(page, 55, 300, 290, 22, "CLASS-WISE RECALL", 10, FOREST, "hebo")
    y = 334
    for name in class_names:
        values = METRICS["per_class"][name]
        text(page, 56, y, 175, 19, name.replace("_", " "), 10, INK, "hebo")
        text(page, 236, y, 80, 19, f"{values['recall'] * 100:.1f}%", 10, FOREST, "hebo", align=2)
        text(page, 326, y, 210, 19, f"support {values['support']}", 9, MUTED)
        y += 32
    rect(page, 575, 310, 333, 126, PALE_CORAL, 4)
    text(page, 594, 329, 292, 20, "WHAT THIS NUMBER MEANS", 9, CORAL, "hebo")
    text(page, 594, 355, 292, 65, "Performance on this source split only. Related slices may cross splits because patient-level separation is not documented.", 11, INK)
    slides.append(("Results", "Read the aggregate metrics, then call out the lowest class recall: meningioma at 91.6 percent. This is measured on the source test partition. Because patient separation is unknown, the result may overestimate performance on new patients or scanners."))

    page = base_slide(doc, 8, "07 / Error analysis", "Where predictions were confused", "Rows are actual labels; columns are predicted labels. Diagonal cells are correct classifications.")
    matrix = METRICS["confusion_matrix"]
    left, top, cell = 330, 175, 56
    text(page, left, 142, 310, 18, "PREDICTED CLASS", 9, MUTED, "hebo", align=1)
    text(page, 52, top + 92, 176, 20, "ACTUAL CLASS", 9, MUTED, "hebo")
    for column, name in enumerate(class_names):
        text(page, left + column * cell, top - 19, cell, 16, name.replace("_", "\n"), 7.3, FOREST, "hebo", align=1)
    for row, name in enumerate(class_names):
        y = top + row * cell
        text(page, 52, y + 18, 245, 18, name.replace("_", " "), 9, INK, "hebo")
        for column, value in enumerate(matrix[row]):
            x = left + column * cell
            fill = PALE_GREEN if row == column else (PALE_CORAL if value >= 8 else WHITE)
            rect(page, x, y, cell - 4, cell - 4, fill, 3)
            text(page, x, y + 17, cell - 4, 24, str(value), 13, FOREST if row == column else INK, "hebo", align=1)
    rect(page, 610, 190, 283, 165, WHITE, 4)
    text(page, 631, 211, 240, 20, "READOUT", 9, FOREST, "hebo")
    text(page, 631, 240, 240, 94, "Meningioma has the lowest recall: 239 of 261 correct. Most errors in this class were predicted as glioma (11) or pituitary (7). This is a useful target for further dataset review.", 11, INK)
    text(page, 50, 444, 860, 28, "Counts describe this test partition; they do not establish clinical sensitivity or generalization.", 10, MUTED)
    slides.append(("Error analysis", "Explain how to read the confusion matrix: rows are actual class, columns predicted. The diagonal is correct. Meningioma has the highest number of missed cases in this split and should be prioritized for independent review."))

    page = base_slide(doc, 9, "08 / Confidence", "Calibration and explanations", "Confidence is adjusted on validation data; Grad-CAM is exploratory and is not segmentation.")
    card(page, 50, 165, 405, 239)
    text(page, 70, 185, 360, 20, "TEMPERATURE SCALING", 10, FOREST, "hebo")
    page.insert_text((70, 250), "1.29", fontsize=24, fontname="hebo", color=INK)
    text(page, 176, 223, 240, 20, "validation-fitted temperature", 10, MUTED)
    text(page, 70, 272, 320, 18, "VALIDATION NEGATIVE LOG-LIKELIHOOD", 8.5, MUTED, "hebo")
    before = METRICS["calibration_nll_before"]
    after = METRICS["calibration_nll_after"]
    max_nll = max(before, after) * 1.15
    for index, (label, value, color) in enumerate((("before", before, CORAL), ("after", after, MINT))):
        y = 308 + index * 39
        text(page, 70, y + 4, 66, 17, label, 9, INK, "hebo")
        rect(page, 142, y, 220 * value / max_nll, 18, color, 3)
        text(page, 370, y + 2, 62, 17, f"{value:.3f}", 9, INK, "hebo", align=2)
    card(page, 485, 165, 423, 239)
    text(page, 507, 185, 370, 20, "HELD-OUT EXPECTED CALIBRATION ERROR", 10, FOREST, "hebo")
    page.insert_text(
        (507, 261),
        f"{METRICS['expected_calibration_error'] * 100:.2f}%",
        fontsize=32,
        fontname="hebo",
        color=INK,
    )
    text(page, 507, 280, 370, 72, "Measured on the source test split after temperature was fitted on the 400-image validation subset. ECE depends on the dataset and binning method.", 10.5, MUTED)
    rect(page, 50, 426, 858, 48, PALE_CORAL, 4)
    text(page, 68, 440, 824, 20, "Grad-CAM highlights model-sensitive regions; it does not prove why the model was right or wrong.", 10, CORAL, "hebo")
    slides.append(("Calibration", "Temperature scaling adjusts the confidence distribution without changing the predicted class. It reduced validation NLL from 0.176 to 0.167. The 1.45 percent test ECE is a dataset-specific calibration measure, not a guarantee. Grad-CAM is only an exploratory visualization."))

    page = base_slide(doc, 10, "09 / Limitations", "What the result does not establish", "A strong score on one public image split is not the same as reliable performance in patient care.")
    limitations = [
        ("Patient leakage", "Patient identities and split construction are undocumented."),
        ("External validity", "One source dataset; no multi-site or prospective evaluation."),
        ("Task scope", "Four image labels; single 2D images only, not full MRI series."),
        ("Clinical readiness", "No clinical workflow validation, regulatory review, or deployment controls."),
    ]
    for index, (title, detail) in enumerate(limitations):
        x = 50 + (index % 2) * 438
        y = 160 + (index // 2) * 134
        card(page, x, y, 414, 108)
        rect(page, x, y, 5, 108, CORAL if index in (0, 3) else GOLD)
        text(page, x + 20, y + 17, 370, 20, title.upper(), 9, CORAL if index in (0, 3) else FOREST, "hebo")
        text(page, x + 20, y + 46, 370, 48, detail, 12, INK)
    rect(page, 50, 444, 852, 35, INK, 4)
    text(page, 66, 454, 820, 18, "Not for diagnosis or treatment decisions. A qualified radiologist must review all findings.", 10, WHITE, "hebo")
    slides.append(("Limitations", "Be direct: the data lacks documented patient-level separation, so related slices could appear across train and test. The sample is from one source and does not represent every scanner, population, or protocol. The app is for research demonstration only."))

    page = base_slide(doc, 11, "10 / Next steps", "A responsible path forward", "Improve the evidence before increasing the claims.")
    next_steps = [
        ("01", "Source stronger data", "Use institution-approved, patient-identified splits and verified labels."),
        ("02", "Lock external evaluation", "Keep an independent site or cohort untouched until final reporting."),
        ("03", "Measure clinical risks", "Review class-wise sensitivity, calibration, scanner shift, and failure cases."),
        ("04", "Require governance", "Add clinical oversight, privacy controls, auditability, and regulatory review before use."),
    ]
    y = 156
    for number, title, detail in next_steps:
        page.draw_circle((77, y + 18), 18, color=MINT, fill=PALE_GREEN, width=1)
        text(page, 59, y + 11, 36, 16, number, 9, FOREST, "hebo", align=1)
        text(page, 112, y + 1, 235, 21, title, 12, INK, "hebo")
        text(page, 345, y + 1, 540, 40, detail, 10.5, MUTED)
        y += 70
    rect(page, 50, 454, 852, 31, PALE_GREEN, 4)
    text(page, 66, 462, 818, 16, "Current takeaway: a working local research prototype with promising source-split results and important validation gaps.", 9.5, FOREST, "hebo")
    slides.append(("Next steps", "Close with the work required before any real-world decision support: patient-level labels, an untouched external cohort, scanner-shift tests, clinical governance, and review by qualified professionals. The current result is promising for a demo, not a clinical claim."))

    notes = [
        ("Slide 1 - Neuro Imaging Lab", slides[0][1]),
        ("Slide 2 - A focused research workflow", slides[1][1]),
        ("Slide 3 - Local-first architecture", slides[2][1]),
        ("Slide 4 - From scan to review", slides[3][1]),
    ]
    add_notes_page(doc, 12, "Opening and workflow", notes)
    notes = [
        ("Slide 5 - Labeled public sample dataset", slides[4][1]),
        ("Slide 6 - How the current model was trained", slides[5][1]),
        ("Slide 7 - Held-out evaluation", slides[6][1]),
    ]
    add_notes_page(doc, 13, "Data and results", notes)
    notes = [
        ("Slide 8 - Where predictions were confused", slides[7][1]),
        ("Slide 9 - Calibration and explanations", slides[8][1]),
        ("Slide 10 - What the result does not establish", slides[9][1]),
        ("Slide 11 - A responsible path forward", slides[10][1]),
    ]
    add_notes_page(doc, 14, "Risks and close", notes)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUTPUT_PATH, garbage=4, deflate=True)
    doc.close()
    print(f"Created {OUTPUT_PATH}")
    print("Pages: 14 (11 slides + 3 presenter-note pages)")


if __name__ == "__main__":
    build_pdf()