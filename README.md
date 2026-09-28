# NTMN

### Neural Tumor Morphogenesis Network with Closed-Loop Morphological Refinement for Multimodal Brain Tumor Segmentation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-ee4c2c.svg)](https://pytorch.org/)

## Overview

This repository provides a research-oriented PyTorch implementation of **NTMN**, a Neural Tumor Morphogenesis Network for multimodal brain tumor MRI segmentation.

NTMN uses a closed-loop structural refinement mechanism in which an intermediate segmentation is used to extract differentiable morphological information. This information is compared with a learned morphological expectation, and the resulting discrepancy is transformed into a residual structural correction that is incorporated into subsequent recurrent refinement stages.

The framework investigates morphology-conditioned recurrent refinement for tumor localization, boundary delineation, and structural consistency in multimodal brain MRI segmentation.

---

## Key Components

The repository follows the principal conceptual stages of NTMN:

1. **MBNSS** — Morphology- and Boundary-Normal-Aware Skull Stripping
2. **ASE** — Anatomical State Encoder
3. **ConvGRU structural-state evolution**
4. **Intermediate tumor segmentation**
5. **Differentiable morphological representation**
6. **Learned morphological expectation**
7. **Morphological discrepancy**
8. **Morphology-conditioned residual structural feedback**
9. **Topology-aware refinement**
10. **Local connectivity regularization**
11. **Normalized predictive entropy**
12. **Final segmentation decoding**

The morphology representation includes area, centroid, covariance, elongation, soft boundary, and soft compactness.

---

## Methodological Concept

```text
      Multimodal MRI
            │
            ▼
          MBNSS
            │
            ▼
           ASE
            │
            ▼
      Initial Structural State
            │
            ▼
 ┌─────────────────────────────┐
 │ Recurrent Refinement Stage  │
 │                             │
 │ Intermediate Segmentation   │
 │          │                  │
 │          ▼                  │
 │ Morphology Extraction       │
 │          │                  │
 │          ▼                  │
 │ Learned Morphological       │
 │ Expectation                 │
 │          │                  │
 │          ▼                  │
 │ Morphological Discrepancy   │
 │          │                  │
 │          ▼                  │
 │ Residual Structural         │
 │ Feedback                    │
 └──────────┬──────────────────┘
            │
            ▼
       Next Stage
            │
            ▼
    Topology Refinement
            │
            ▼
    Final Segmentation
```

## Repository Structure

```text
NTMN/
├── README.md
├── LICENSE
├── requirements.txt
├── configs/
│   └── default.yaml
├── train.py
├── evaluate.py
├── src/
│   └── ntmn/
        ├──__init__
│       ├── data.py
│       ├── losses.py
│       ├── model.py
│       ├── morphology.py
│       ├── topology.py
│       └── uncertainty.py
├── scripts/
│   ├── train.sh
│   └── evaluate.sh
├── tests/
│   └── test_smoke.py
└── docs/
    ├── DATA.md
    └── REPRODUCIBILITY.md
```

Datasets, medical images, model checkpoints, and experiment outputs are excluded from version control through `.gitignore`.

---

## Datasets

- **UPENN-GBM:** https://www.cancerimagingarchive.net/collection/upenn-gbm/
- **TCGA-GBM:** https://www.cancerimagingarchive.net/collection/tcga-gbm/
- **MOTUM:** https://doi.gin.g-node.org/10.12751/g-node.tvzqc5/
  
---
## Installation

### 1. Clone the repository

```bash
git clone https://github.com/IndrakumarK/A-Knowledge-Evolving-Neural-Tumor-Morphogenesis-Network-with-Closed-Loop-Morphological-Refinement.git
cd A-Knowledge-Evolving-Neural-Tumor-Morphogenesis-Network-with-Closed-Loop-Morphological-Refinement
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate on Linux/macOS:

```bash
source .venv/bin/activate
```

Activate on Windows:

```powershell
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

The reference environment uses Python 3.10+, PyTorch 2.6.0, NumPy, PyYAML, SciPy, scikit-learn, and tqdm.

---

## Data Preparation

The benchmark datasets are **not distributed with this repository**. The manuscript uses UPenn-GBM, TCGA-GBM, and MOTUM. Users must obtain these datasets independently from their authorized sources and comply with the corresponding dataset terms and usage requirements.

The generic loader expects:

```text
data/
├── train/
│   └── case_id/
│       ├── T1.npy
│       ├── T1ce.npy
│       ├── T2.npy
│       ├── FLAIR.npy
│       └── mask.npy
├── val/
│   └── case_id/
│       ├── T1.npy
│       ├── T1ce.npy
│       ├── T2.npy
│       ├── FLAIR.npy
│       └── mask.npy
└── test/
    └── case_id/
        ├── T1.npy
        ├── T1ce.npy
        ├── T2.npy
        ├── FLAIR.npy
        └── mask.npy
```

Each modality is represented as a 2-D `256 × 256` array. The segmentation mask uses four labels: `0` background and `1–3` tumor classes.

The `.npy` representation is a generic repository format. Original NIfTI/DICOM data require appropriate dataset-specific conversion, registration, resampling, normalization, slice selection, and label mapping.

See [`docs/DATA.md`](docs/DATA.md) for details.

---

## Configuration

The reference configuration is provided in `configs/default.yaml`.

| Setting | Value |
|---|---|
| Modalities | T1, T1ce, T2, FLAIR |
| Dimensionality | 2-D |
| Input size | 256 × 256 |
| Classes | 4 |
| Recurrent stages | 3 |
| Structural feature dimension | 256 |
| Optimizer | AdamW |
| Learning rate | 1 × 10⁻⁴ |
| Weight decay | 1 × 10⁻⁴ |
| Epochs | 150 |
| Batch size | 8 |
| Scheduler | Cosine annealing |
| Minimum learning rate | 1 × 10⁻⁶ |
| Initialization | Xavier |
| Precision | FP32 |
| Random seed | 42 |

---

## Training

After preparing the data and configuration:

```bash
bash scripts/train.sh
```

or:

```bash
python train.py --config configs/default.yaml
```

The best checkpoint is selected using **validation Dice only** and is saved as:

```text
checkpoints/best.pt
```

The test set should not be used for model selection or hyperparameter tuning.

---

## Evaluation

Evaluate the selected checkpoint with:

```bash
bash scripts/evaluate.sh
```

or:

```bash
python evaluate.py \
    --config configs/default.yaml \
    --checkpoint checkpoints/best.pt \
    --split test
```

The evaluator reports patient-level foreground Dice values and their mean and standard deviation.

For publication-level statistical analysis, retain patient-level predictions and analyze them directly rather than reconstructing statistics from aggregate dataset means.

---

## Reproducibility

For reproducible experiments:

- Use fixed patient-level train/validation/test partitions.
- Ensure no patient occurs in more than one partition.
- Record dataset versions and preprocessing procedures.
- Record the exact software environment and actual GPU.
- Keep the random seed fixed.
- Select checkpoints using validation data only.
- Preserve patient-level predictions and metric values.
- Compute pooled statistics from patient-level observations.
- Use paired observations for paired statistical tests.
- Generate bootstrap confidence intervals from paired patient-level observations.
- Clearly distinguish reproduced results from published or externally reported results.

See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

---

## External Clinical Cohort

The independently collected clinical MRI cohort described in the manuscript is **not included in this repository**.

It was used exclusively for external qualitative assessment and was not used for training, validation, parameter selection, quantitative benchmark evaluation, or statistical performance analysis.

Voxel-level tumor annotations were unavailable for these cases. Consequently, quantitative segmentation metrics should not be reported for this cohort.

Private clinical MRI data must remain confidential and must not be uploaded to this repository.

---

## Results and Publication Data

This repository does not fabricate or redistribute manuscript results.

Reported benchmark values should only be associated with this repository when they have been generated and verified using the corresponding experimental pipeline.

- Comparator results should be labeled **Reproduced** only when the comparator was actually executed.
- Published values should be clearly identified as **Published/Reported**.
- Pooled statistics should be calculated from patient-level observations.
- Statistical tests should use the actual paired patient-level values.

---

## Medical Data and Responsible Use

This repository is intended for **research purposes only**.

KE-NTMN is not a clinically validated diagnostic system and should not be used as a substitute for professional medical judgment, diagnosis, treatment planning, or clinical decision-making.

The repository does not contain private clinical MRI data or patient-identifiable information.

---

## Citation

If you use this repository or the KE-NTMN methodology in academic work, please cite the associated manuscript:

```bibtex
@article{KE_NTMN_2026,
  title   = {KE-NTMN: A Neural Tumor Morphogenesis Network with Closed-Loop Morphological Refinement for Multimodal Brain Tumor Segmentation},
  year    = {2026},
  note    = {}
}
```

## License

This project is released under the **MIT License**. See [`LICENSE`](LICENSE) for the complete license text.

---

## Contact

For implementation or methodology questions, please use the GitHub repository issue tracker or the contact information provided with the associated manuscript.
