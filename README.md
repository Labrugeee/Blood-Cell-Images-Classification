# White Blood Cell Classification

**Automated classification of four white blood cell subtypes from peripheral blood smear images, using classical machine learning and deep learning.**

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Status](https://img.shields.io/badge/status-work%20in%20progress-orange)
![License](https://img.shields.io/badge/license-MIT-green)

> Academic project — M2 Bioinformatics. The goal is a methodologically sound pipeline rather than state-of-the-art performance: particular attention is paid to evaluation protocol, data leakage and model interpretability.

---

## Table of contents

- [Background](#background)
- [Dataset](#dataset)
- [A note on data leakage](#a-note-on-data-leakage)
- [Project structure](#project-structure)
- [Getting started](#getting-started)
- [Methodology](#methodology)
- [Results](#results)
- [Interpretability](#interpretability)
- [Limitations](#limitations)
- [References](#references)
- [Authors](#authors)

---

## Background

Diagnosing blood-based disorders often relies on identifying and counting white blood cell (leukocyte) subtypes in a peripheral blood smear — the *differential blood count*. This is traditionally performed manually by a trained cytologist under a microscope: slow, operator-dependent and difficult to scale.

The four subtypes considered here are distinguished by well-established morphological criteria:

| Cell type | Nucleus | Cytoplasm | Typical clinical relevance |
|---|---|---|---|
| **Eosinophil** | Bi-lobed | Coarse orange-red granules | Allergic reactions, parasitic infections |
| **Lymphocyte** | Large, round, dense | Thin rim, scarce | Viral infections, lymphoid leukaemias |
| **Monocyte** | Kidney / horseshoe-shaped | Abundant, greyish | Chronic infections, inflammation |
| **Neutrophil** | Multi-lobed (3–5 lobes) | Fine, pale granules | Bacterial infections, most abundant leukocyte |

Automating this task is a natural image-classification problem, and a good testbed for convolutional neural networks.

---

## Dataset

We use the **BCCD (Blood Cell Count and Detection)** dataset, distributed on Kaggle by Paul Mooney.

- **Format:** JPEG, 320 × 240 px, RGB
- **Total size:** ~129 MB
- **Classes:** 4, approximately balanced
- **Origin:** ~410 original smear images, expanded to ~12,500 images through data augmentation

### Class distribution

| Split | Eosinophil | Lymphocyte | Monocyte | Neutrophil | **Total** |
|---|---:|---:|---:|---:|---:|
| `TRAIN` | 2,497 | 2,483 | 2,478 | 2,499 | **9,957** |
| `TEST` | 623 | 620 | 620 | 624 | **2,487** |
| `TEST_SIMPLE` | 13 | 6 | 4 | 48 | **71** |

Classes are near-perfectly balanced in `TRAIN` and `TEST`, so plain accuracy is a meaningful metric — though macro-F1 and the confusion matrix are reported alongside it. The naive majority-class baseline sits at **25%**.

---

## A note on data leakage

The ~12,500 images are **augmented variants of only ~410 original smears**, i.e. roughly 30 variants per source image. This creates a well-known but frequently overlooked pitfall:

> If augmented variants of the same original image end up on both sides of a train/test split, the reported score measures memorisation, not generalisation.

Filenames (`_0_1169.jpeg`) carry no traceable link back to their source image, so the grouping cannot simply be read off the file paths. This project therefore treats the evaluation protocol as a first-class research question rather than an implementation detail:

1. **Auditing the provided split** — quantifying near-duplicate overlap between `TRAIN` and `TEST`.
2. **Building a grouped validation split** — a random split inside `TRAIN` is guaranteed to leak; variants must be grouped by source image first.
3. **Choosing a trustworthy final test set** — an augmented test set yields optimistic estimates.

Findings and the resulting protocol are documented in [`results/`](results/).

<!-- TODO: link the leakage audit report / figure once phase 2 is complete -->

---

## Project structure

```
.
├── data/              # datasets (not versioned — see Getting started)
├── script/            # numbered, reproducible pipeline scripts
├── notebooks/         # exploratory analysis
├── results/
│   ├── figures/       # plots, confusion matrices, Grad-CAM overlays
│   └── metrics/       # scores and evaluation reports
├── models/            # trained weights (not versioned)
├── requirements.txt
└── README.md
```

---

## Getting started

### 1. Clone and set up the environment

```bash
git clone https://github.com/<user>/<repo>.git
cd <repo>
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download the data

The dataset is **not versioned** in this repository (size, and GitHub's 100 MB per-file limit). Download it manually:

1. Go to [kaggle.com/datasets/paultimothymooney/blood-cells](https://www.kaggle.com/datasets/paultimothymooney/blood-cells) (free account required)
2. Download and unzip the archive
3. Place the contents under `data/` so that the tree looks like:

```
data/
└── dataset2-master/
    └── images/
        ├── TRAIN/{EOSINOPHIL,LYMPHOCYTE,MONOCYTE,NEUTROPHIL}/
        ├── TEST/{EOSINOPHIL,LYMPHOCYTE,MONOCYTE,NEUTROPHIL}/
        └── TEST_SIMPLE/{EOSINOPHIL,LYMPHOCYTE,MONOCYTE,NEUTROPHIL}/
```

### 3. Run the pipeline

<!-- TODO: update as scripts are written -->

```bash
python script/01_eda.py           # exploratory data analysis
python script/02_split.py         # leakage audit + grouped split
python script/03_baseline.py      # classical ML baselines
python script/04_cnn.py           # CNN trained from scratch
python script/05_transfer.py      # transfer learning
python script/06_interpret.py     # Grad-CAM and error analysis
```

---

## Methodology

The pipeline is deliberately staged, from simple and interpretable to complex and performant, so that each gain in performance can be attributed to a specific modelling choice.

| # | Stage | What it does | Why |
|---|---|---|---|
| 1 | **Exploratory analysis** | Class balance, image quality, staining variability, duplicate detection | Understand the data before modelling it |
| 2 | **Evaluation protocol** | Leakage audit, grouped train/validation split, metric selection | Fixed *before* any model is trained |
| 3 | **Classical baselines** | Hand-crafted features (colour histograms, LBP/Haralick texture, nucleus morphology) + logistic regression / SVM / random forest | Establishes a performance floor and makes explicit what a CNN learns implicitly |
| 4 | **CNN from scratch** | Small custom architecture, learning curves, regularisation | Understand the building blocks end to end |
| 5 | **Transfer learning** | ImageNet-pretrained backbone, feature extraction then fine-tuning | Compensates for the small number of *independent* source images |
| 6 | **Interpretability** | Grad-CAM, error analysis against expected cytological confusions | Verify the model looks at the cell, not the background |

---

## Results

<!-- TODO: fill in as experiments are completed. Do not report numbers that
     have not been produced by a script committed to this repository. -->

| Model | Accuracy | Macro-F1 | Notes |
|---|---:|---:|---|
| Majority-class baseline | 0.25 | — | Reference floor |
| Logistic regression (hand-crafted features) | *TBD* | *TBD* | |
| SVM (hand-crafted features) | *TBD* | *TBD* | |
| Random forest (hand-crafted features) | *TBD* | *TBD* | |
| CNN from scratch | *TBD* | *TBD* | |
| Transfer learning (fine-tuned) | *TBD* | *TBD* | |

*All scores are reported on the held-out test set defined in stage 2.*

---

## Interpretability

Model decisions are inspected with **Grad-CAM**, which highlights the image regions that drove a prediction towards a given class. It is architecture-agnostic (so the same analysis applies to both the custom CNN and the pretrained backbone), cheap to compute, and class-discriminative — on a single image one can compare *why eosinophil?* against *why neutrophil?*.

Grad-CAM resolution is coarse by construction, so conclusions are cross-checked with an occlusion-sensitivity analysis on a subset of images.

<!-- TODO: add Grad-CAM overlay figures from results/figures/ -->

---

## Limitations

- Images originate from a **single source dataset and staining protocol**; generalisation to other laboratories is untested.
- Labels were assigned at the level of the original smear, so a degree of **label noise** is plausible.
- Heavy augmentation means the number of **statistically independent samples is far lower** than the raw image count suggests.
- Basophils, the fifth leukocyte subtype, are **absent** from the four-class dataset.

---

## References

- Mooney, P. *Blood Cell Images* dataset — [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/blood-cells)
- Athelas. [*Classifying White Blood Cells With Convolutional Neural Networks*](https://blog.athelas.com/classifying-white-blood-cells-with-convolutional-neural-networks-2ca6da239331)
- Selvaraju, R. R. et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.* ICCV.
- Ribeiro, M. T., Singh, S., Guestrin, C. (2016). *"Why Should I Trust You?": Explaining the Predictions of Any Classifier.* KDD.

---

## Authors

M2 Bioinformatics project.

<!-- TODO: add names -->

## License

Code released under the MIT License. The dataset remains subject to its original licensing terms.
