from __future__ import annotations

from typing import Any


RESEARCH_DATASETS: list[dict[str, Any]] = [
    {
        "id": "kaggle-brain-tumor-mri",
        "name": "Brain Tumor MRI Dataset",
        "publisher": "Kaggle / Masoud Nickparvar",
        "url": "https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset",
        "access": "Kaggle download or local export",
        "license": "See the Kaggle dataset page",
        "image_count": 7023,
        "split_summary": "5,712 training images and 1,311 test images",
        "classes": ["glioma", "meningioma", "no_tumor", "pituitary"],
        "notes": [
            "The public page describes a four-class brain-tumor classification dataset.",
            "Use the supplied training and test directories as separate partitions.",
            "Kaggle credentials or an explicit local download are required; the app does not upload data to Kaggle.",
        ],
    },
    {
        "id": "ieee-10762991",
        "name": "IEEE reference supplied by the researcher",
        "publisher": "IEEE Xplore",
        "url": "https://ieeexplore.ieee.org/abstract/document/10762991",
        "access": "Reference only",
        "license": "Confirm terms at the source",
        "image_count": None,
        "split_summary": "Dataset size and split details are not available from the public abstract page",
        "classes": [],
        "notes": [
            "The public IEEE page currently exposes an account shell rather than a verifiable abstract or download.",
            "Do not infer dataset counts, labels, or licensing from this link until the paper or accompanying repository is accessible.",
            "Record the paper's actual protocol and cite it alongside any dataset used in a research report.",
        ],
    },
]


def research_dataset_sources() -> list[dict[str, Any]]:
    return RESEARCH_DATASETS
