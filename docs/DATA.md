# Data Preparation

The datasets used in the NTMN study are not redistributed with this repository. Users must obtain the datasets from their respective official sources and comply with the applicable access, licensing, and usage requirements.

The benchmark datasets used in the manuscript are:

- **UPenn-GBM**
- **TCGA-GBM**
- **MOTUM**

The independently collected clinical MRI cohort used for qualitative external assessment is not included in this repository. These clinical data are private and must not be redistributed.

## Dataset Organization

The generic data loader expects the following directory structure:

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
