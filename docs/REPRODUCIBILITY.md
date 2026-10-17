# Reproducibility Checklist

The following checklist should be completed before reproducing or extending the NTMN experiments.

## Dataset and Splits

- [ ] Record the exact dataset name and version used for each experiment.
- [ ] Record the exact patient-level training, validation, and test partitions.
- [ ] Ensure that no patient occurs in more than one partition.
- [ ] Preserve the same patient-level partitions across comparative experiments.
- [ ] Verify the dataset-specific label mapping and the foreground classes available for evaluation.
- [ ] Ensure that unavailable or unlabeled foreground classes are excluded from the corresponding supervised losses and evaluation.
- [ ] Do not redistribute UPenn-GBM, TCGA-GBM, MOTUM, or private clinical MRI data through this repository.

## Preprocessing and Training

- [ ] Record all dataset-specific preprocessing and data-conversion steps.
- [ ] Record modality alignment, spatial resampling, and modality-specific normalization procedures.
- [ ] Record the input resolution and four MRI modalities used by NTMN.
- [ ] Record all data-augmentation operations and their parameters.
- [ ] Record the random seed used for training and evaluation.
- [ ] Record the exact PyTorch version and other relevant software dependencies.
- [ ] Record the actual GPU model and available memory used for experiments.
- [ ] Keep the input resolution and model configuration consistent with the reported experiments.
- [ ] Select model checkpoints using validation data only.
- [ ] Do not use test-set performance for model selection or hyperparameter tuning.
- [ ] Verify that ground-truth tumor labels are used only for supervised training and evaluation and are not used during inference.

## Evaluation and Results

- [ ] Save patient-level predictions for each evaluated method.
- [ ] Save patient-level metric values used to calculate reported summary statistics.
- [ ] Compute pooled statistics directly from patient-level observations rather than averaging dataset-level means.
- [ ] Preserve patient-level pairing when comparing two methods on the same test cases.
- [ ] Use paired statistical tests only when the observations are appropriately paired.
- [ ] Generate bootstrap confidence intervals from the paired patient-level observations.
- [ ] Record the number of patients contributing to each statistical comparison.
- [ ] Verify that Dice, IoU, precision, sensitivity, and specificity are calculated over the annotated foreground classes available for each dataset, excluding background.
- [ ] Verify that HD95 is calculated using the corresponding in-plane physical pixel spacing and is interpreted as a two-dimensional boundary-distance measure.

## Ablation and Component Analysis

- [ ] Record the exact configuration used for each ablation experiment.
- [ ] Preserve the same held-out test cohort across ablation configurations.
- [ ] Distinguish progressive cumulative ablations from isolated component ablations.
- [ ] Do not interpret cumulative ablation results as independent proof of necessity for every individual component.
- [ ] Record the recurrent refinement stage and component configuration used in each experiment.

## Computational Analysis

- [ ] Record the number of trainable parameters under the reported model configuration.
- [ ] Record the GFLOPs measurement and input resolution used for calculation.
- [ ] Record peak GPU memory consumption under the specified evaluation conditions.
- [ ] Record the inference-time measurement procedure, including batch size and timing conditions where available.
- [ ] Ensure that reported computational measurements correspond to the implementation used for the manuscript results.

## Comparator Provenance

- [ ] Label a comparator as **Reproduced** only when its implementation was actually executed under the stated experimental protocol.
- [ ] Clearly identify results obtained from published or externally reported sources.
- [ ] Do not present published results as independently reproduced results.
- [ ] Record the source and provenance of each comparative result reported in tables.
- [ ] Ensure that comparative methods use equivalent evaluation cases, patient-level splits, and metric definitions where applicable.

## External Clinical Cohort

- [ ] Keep the independently collected clinical MRI cohort separate from the benchmark datasets.
- [ ] Do not use the external clinical cohort for training, validation, parameter selection, or quantitative benchmark evaluation.
- [ ] Do not quantitatively evaluate cases for which voxel-level tumor annotations are unavailable.
- [ ] Use the external clinical cohort only for the qualitative evaluation described in the manuscript.
- [ ] Confirm and document the applicable institutional or ethics approval, exemption, or permission status before public release of the repository or manuscript.
- [ ] Keep private clinical MRI data confidential and do not include them in the public repository.

## Recommended Reproducibility Records

For each experiment, retain:

```text
experiment/
├── config.yaml
├── dataset_version.txt
├── patient_splits.json
├── environment.txt
├── training_log.csv
├── checkpoint/
├── predictions/
├── patient_metrics.csv
└── statistics/
    ├── confidence_intervals.csv
    └── significance_tests.csv
