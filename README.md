## NTMN

### Neural Tumor Morphogenesis Network: Closed-Loop Morphological Feedback for Multimodal Brain Tumor Segmentation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-ee4c2c.svg)](https://pytorch.org/)

## Overview

This repository provides a research-oriented PyTorch implementation of **NTMN**, a Neural Tumor Morphogenesis Network for multimodal brain tumor MRI segmentation.

NTMN uses a closed-loop structural feedback mechanism in which an intermediate segmentation is used to extract differentiable morphological information. This information is compared with a learned morphological expectation, and the resulting discrepancy is transformed into a residual structural correction that is incorporated into subsequent recurrent refinement stages.

The framework investigates morphology-conditioned recurrent refinement for tumor localization, boundary delineation, and structural consistency in multimodal brain MRI segmentation.

---

## Key Components

The central contribution of NTMN is morphology conditioned recurrent
structural feedback. The remaining components support the overall
segmentation framework.

1. MBNSS
2. ASE
3. ConvGRU structural state evolution
4. Intermediate tumor segmentation
5. Differentiable morphological representation
6. Learned morphological expectation
7. Morphological discrepancy
8. Morphology conditioned residual structural feedback
9. Topology aware refinement
10. Local connectivity regularization
11. Normalized predictive entropy
12. Final segmentation decoding


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
        ├──__init__.py
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

• **UPenn-GBM:** [The Cancer Imaging Archive](https://www.cancerimagingarchive.net/collection/upenn-gbm/)
• **TCGA-GBM:** [The Cancer Imaging Archive](https://www.cancerimagingarchive.net/collection/tcga-gbm/)
• **MOTUM:** [MOTUM Dataset](https://doi.gin.g-node.org/10.12751/g-node.tvzqc5/)

  
---
## Installation

### 1. Clone the repository

```bash
git clone https://github.com/IndrakumarK/A-Neural-Tumor-Morphogenesis-Network-with-Closed-Loop-Morphological-Refinement.git
cd A-Neural-Tumor-Morphogenesis-Network-with-Closed-Loop-Morphological-Refinement
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

Each modality is represented as a 2-D `256 × 256` array. The network uses a four class output space with label 0 representing background and labels 1 to 3 representing tumor classes. Dataset specific annotations may differ in the availability of individual foreground classes; unavailable classes are excluded from the corresponding supervised losses and evaluation.

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

The evaluator reports patient level segmentation metrics for the
held out test cases. Depending on the evaluation configuration, these
include Dice, IoU, HD95, precision, sensitivity, specificity, and
the structural measures used in the manuscript.

Patient level predictions and metric values should be retained for statistical analysis. Pooled statistics should be calculated from individual patient level observations rather than reconstructed from dataset level means.

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

The repository does not include fabricated, synthetic, or unverified
experimental results.

Reported manuscript results should only be associated with this
repository when they have been generated and verified using the
corresponding experimental pipeline.

• Comparator results should be labeled **Reproduced** only when the
  comparator was actually executed.
• Results obtained from published literature should be labeled
  **Published/Reported**.
• Pooled statistics should be calculated from patient level observations.
• Statistical tests should use the actual paired patient level values.

---

## Medical Data and Responsible Use

This repository is intended for **research purposes only**.

NTMN is not a clinically validated diagnostic system and should not be used as a substitute for professional medical judgment, diagnosis, treatment planning, or clinical decision making.

The repository does not contain private clinical MRI data or patient-identifiable information.

---

## Citation

@article{NTMN2026,
  title   = {Neural Tumor Morphogenesis Network: Closed-Loop Morphological Feedback for Brain Tumor Segmentation in Multimodal MRI},
  author  = {},
  year    = {2026},
  note    = {Manuscript}
}

## License

This project is released under the **MIT License**. See [`LICENSE`](LICENSE) for the complete license text.

---

## Contact

For implementation or methodology questions, please use the GitHub repository issue tracker or the contact information provided with the associated manuscript.
