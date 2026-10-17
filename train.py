"""
NTMN Training Script
====================

Neural Tumor Morphogenesis Network (NTMN)

This script trains the NTMN model for multimodal brain tumor
segmentation using the configuration specified in
configs/default.yaml.

Training configuration
----------------------
- 4 MRI modalities: T1, T1ce, T2, FLAIR
- 2-D input
- 256 x 256 spatial resolution
- 4 output classes
- 3 recurrent refinement stages
- AdamW optimizer
- Initial learning rate: 1e-4
- Weight decay: 1e-4
- Cosine annealing
- Minimum learning rate: 1e-6
- Batch size: 8
- 150 epochs
- FP32
- Xavier initialization
- Fixed random seed

The best checkpoint is selected using validation Dice only.
The test split is not used during training or model selection.
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from src.ntmn.data import BrainTumorDataset
from src.ntmn.losses import (
    connectivity_loss,
    knowledge_consistency_loss,
    morphology_estimation_loss,
    smoothness_loss,
    structural_stability_loss,
    topology_loss,
    uncertainty_consistency_loss,
    dice_loss,
)
from src.ntmn.model import NTMN
from src.ntmn.morphology import morphology


# =====================================================================
# Dataset-specific valid classes
# =====================================================================

DATASET_VALID_CLASSES = {
    "UPenn": np.array(
        [True, True, True, True],
        dtype=bool,
    ),
    "UPenn-GBM": np.array(
        [True, True, True, True],
        dtype=bool,
    ),
    "TCGA": np.array(
        [True, True, True, True],
        dtype=bool,
    ),
    "TCGA-GBM": np.array(
        [True, True, True, True],
        dtype=bool,
    ),
    "MOTUM": np.array(
        [True, True, False, True],
        dtype=bool,
    ),
}


# =====================================================================
# Reproducibility
# =====================================================================

def set_seed(
    seed,
    deterministic=True,
):
    """
    Set random seeds for reproducible experiments.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


# =====================================================================
# Valid-class handling
# =====================================================================

def get_valid_classes(
    batch,
    dataset_name,
    num_classes,
):
    """
    Obtain valid foreground classes for the corresponding dataset.

    Dataset-specific unavailable classes are excluded from supervised
    losses and quantitative evaluation.
    """

    if "valid_classes" in batch:

        valid_classes = batch[
            "valid_classes"
        ]

        if torch.is_tensor(
            valid_classes
        ):
            valid_classes = (
                valid_classes
                .detach()
                .cpu()
                .numpy()
            )

        valid_classes = np.asarray(
            valid_classes
        )

        if valid_classes.ndim > 1:
            valid_classes = valid_classes[0]

        valid_classes = valid_classes.astype(
            bool
        )

    elif dataset_name in DATASET_VALID_CLASSES:

        valid_classes = np.asarray(
            DATASET_VALID_CLASSES[
                dataset_name
            ],
            dtype=bool,
        )

    else:

        valid_classes = np.ones(
            num_classes,
            dtype=bool,
        )

    if len(valid_classes) != num_classes:
        raise ValueError(
            f"Expected {num_classes} valid-class flags, "
            f"but received {len(valid_classes)}."
        )

    return valid_classes


def valid_classes_to_tensor(
    valid_classes,
    device,
):
    """
    Convert valid-class flags to a boolean tensor.
    """

    return torch.as_tensor(
        valid_classes,
        dtype=torch.bool,
        device=device,
    )


# =====================================================================
# Dataset name
# =====================================================================

def get_dataset_name(
    batch,
    default=None,
):
    """
    Obtain dataset name from the batch.
    """

    if "dataset" in batch:

        dataset_name = batch[
            "dataset"
        ]

        if isinstance(
            dataset_name,
            (list, tuple),
        ):
            return str(
                dataset_name[0]
            )

        return str(
            dataset_name
        )

    if default is not None:
        return str(default)

    return None


# =====================================================================
# Validation
# =====================================================================

def evaluate(
    model,
    loader,
    device,
    num_classes,
):
    """
    Evaluate the model on the validation split.

    Validation Dice is computed over valid foreground classes.
    """

    model.eval()

    scores = []

    with torch.no_grad():

        for batch in loader:

            images = batch[
                "image"
            ].to(
                device,
                non_blocking=True,
            )

            target = batch[
                "mask"
            ].to(
                device,
                non_blocking=True,
            )

            dataset_name = get_dataset_name(
                batch
            )

            valid_classes = (
                get_valid_classes(
                    batch,
                    dataset_name,
                    num_classes,
                )
            )

            prediction = (
                model(images)["prob"]
                .argmax(dim=1)
            )

            class_scores = []

            for class_id in range(
                1,
                num_classes,
            ):

                # -----------------------------------------------------
                # Skip unavailable classes
                # -----------------------------------------------------

                if not valid_classes[
                    class_id
                ]:
                    continue

                pred_class = (
                    prediction == class_id
                )

                target_class = (
                    target == class_id
                )

                denominator = (
                    pred_class.sum()
                    + target_class.sum()
                )

                if denominator == 0:

                    score = torch.tensor(
                        1.0,
                        device=device,
                    )

                else:

                    intersection = (
                        pred_class
                        & target_class
                    ).sum()

                    score = (
                        2.0
                        * intersection.float()
                        / denominator.float()
                    )

                class_scores.append(
                    score
                )

            if class_scores:

                scores.append(
                    torch.stack(
                        class_scores
                    ).mean().item()
                )

    if not scores:
        return 0.0

    return float(
        np.mean(scores)
    )


# =====================================================================
# Composite loss
# =====================================================================

def compute_loss(
    output,
    target,
    valid_classes,
    loss_config,
):
    """
    Compute the complete NTMN training objective.

    Components
    ----------
    - Foreground Dice
    - Structural stability
    - Morphology estimation
    - Knowledge consistency
    - Topology
    - Connectivity
    - Spatial smoothness
    - Uncertainty consistency
    """

    prob = output[
        "prob"
    ]

    logits = output[
        "logits"
    ]

    states = output[
        "states"
    ]

    # ---------------------------------------------------------------
    # Foreground Dice
    # ---------------------------------------------------------------

    dice = dice_loss(
        logits,
        target,
        num_classes=prob.shape[1],
        valid_classes=valid_classes,
    )

    # ---------------------------------------------------------------
    # Structural stability
    # ---------------------------------------------------------------

    stability_terms = []

    for i in range(
        1,
        len(states),
    ):

        stability_terms.append(
            structural_stability_loss(
                states[i],
                states[i - 1],
            )
        )

    if stability_terms:

        stability = torch.stack(
            stability_terms
        ).mean()

    else:

        stability = prob.new_tensor(
            0.0
        )

    # ---------------------------------------------------------------
    # Ground-truth foreground morphology
    # ---------------------------------------------------------------

    foreground_target = (
        (target > 0)
        .float()
        .unsqueeze(1)
    )

    target_morphology = morphology(
        foreground_target
    )

    # ---------------------------------------------------------------
    # Prediction-derived morphology
    # ---------------------------------------------------------------

    if "morphology" not in output:
        raise KeyError(
            "NTMN output must contain "
            "'morphology'."
        )

    morphology_pred = output[
        "morphology"
    ]

    morphology_loss_value = (
        morphology_estimation_loss(
            morphology_pred,
            target_morphology.detach(),
        )
    )

    # ---------------------------------------------------------------
    # Learned morphology expectation
    # ---------------------------------------------------------------

    if (
        "expected_morphology"
        not in output
    ):
        raise KeyError(
            "NTMN output must contain "
            "'expected_morphology'."
        )

    expected_morphology = output[
        "expected_morphology"
    ]

    knowledge_loss = (
        knowledge_consistency_loss(
            morphology_pred,
            expected_morphology,
        )
    )

    # ---------------------------------------------------------------
    # Topology
    # ---------------------------------------------------------------

    topology = topology_loss(
        prob,
        target,
        valid_classes=valid_classes,
    )

    # ---------------------------------------------------------------
    # Connectivity
    # ---------------------------------------------------------------

    connectivity = connectivity_loss(
        prob,
        target=target,
        valid_classes=valid_classes,
    )

    # ---------------------------------------------------------------
    # Spatial smoothness
    # ---------------------------------------------------------------

    smoothness = smoothness_loss(
        logits
    )

    # ---------------------------------------------------------------
    # Uncertainty consistency
    # ---------------------------------------------------------------

    uncertainty_logits = output.get(
        "auxiliary_logits",
        logits,
    )

    uncertainty = (
        uncertainty_consistency_loss(
            uncertainty_logits,
            target,
            valid_classes=valid_classes,
        )
    )

    # ---------------------------------------------------------------
    # Weighted total loss
    # ---------------------------------------------------------------

    total = (

        loss_config.get(
            "foreground_dice",
            1.00,
        )
        * dice

        + loss_config.get(
            "structural_stability",
            0.20,
        )
        * stability

        + loss_config.get(
            "morphology_estimation",
            0.20,
        )
        * morphology_loss_value

        + loss_config.get(
            "knowledge_consistency",
            0.20,
        )
        * knowledge_loss

        + loss_config.get(
            "topology",
            0.10,
        )
        * topology

        + loss_config.get(
            "connectivity",
            0.10,
        )
        * connectivity

        + loss_config.get(
            "smoothness",
            0.05,
        )
        * smoothness

        + loss_config.get(
            "uncertainty_consistency",
            0.05,
        )
        * uncertainty
    )

    return total


# =====================================================================
# Xavier initialization
# =====================================================================

def initialize_xavier(
    model,
):
    """
    Apply Xavier uniform initialization to convolutional and
    linear layers.
    """

    for module in model.modules():

        if isinstance(
            module,
            (
                nn.Conv2d,
                nn.Linear,
            ),
        ):

            nn.init.xavier_uniform_(
                module.weight
            )

            if module.bias is not None:

                nn.init.zeros_(
                    module.bias
                )


# =====================================================================
# Main
# =====================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Train the Neural Tumor "
            "Morphogenesis Network (NTMN)."
        )
    )

    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help=(
            "Path to configuration YAML file."
        ),
    )

    args = parser.parse_args()

    # =================================================================
    # Configuration
    # =================================================================

    config_path = Path(
        args.config
    )

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: "
            f"{config_path}"
        )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:

        config = yaml.safe_load(
            file
        )

    if config is None:
        config = {}

    # =================================================================
    # Reproducibility
    # =================================================================

    reproducibility_config = config.get(
        "reproducibility",
        {},
    )

    seed = reproducibility_config.get(
        "random_seed",
        config.get(
            "project",
            {},
        ).get(
            "seed",
            42,
        ),
    )

    deterministic = reproducibility_config.get(
        "deterministic",
        True,
    )

    set_seed(
        seed,
        deterministic=deterministic,
    )

    print(
        f"Random seed: {seed}"
    )

    # =================================================================
    # Device
    # =================================================================

    configured_device = config.get(
        "project",
        {},
    ).get(
        "device",
        "cuda",
    )

    if (
        configured_device == "cuda"
        and torch.cuda.is_available()
    ):

        device = torch.device(
            "cuda"
        )

    else:

        device = torch.device(
            "cpu"
        )

    print(
        f"Using device: {device}"
    )

    if device.type == "cuda":

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    # =================================================================
    # Data configuration
    # =================================================================

    data_config = config[
        "data"
    ]

    data_root = Path(
        data_config.get(
            "root",
            "data",
        )
    )

    image_size = tuple(
        data_config.get(
            "image_size",
            [256, 256],
        )
    )

    num_classes = int(
        data_config.get(
            "num_classes",
            4,
        )
    )

    # =================================================================
    # Datasets
    # =================================================================

    train_dataset = BrainTumorDataset(
        root=data_root / "train",
        image_size=image_size,
        num_classes=num_classes,
        augment=data_config.get(
            "augmentation",
            {},
        ).get(
            "enabled",
            True,
        ),
    )

    val_dataset = BrainTumorDataset(
        root=data_root / "val",
        image_size=image_size,
        num_classes=num_classes,
        augment=False,
    )

    # =================================================================
    # Data loaders
    # =================================================================

    training_config = config[
        "training"
    ]

    batch_size = int(
        training_config.get(
            "batch_size",
            8,
        )
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    print(
        f"Training cases: "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation cases: "
        f"{len(val_dataset)}"
    )

    # =================================================================
    # Model
    # =================================================================

    model_config = config[
        "model"
    ]

    recurrent_config = model_config.get(
        "recurrent",
        {},
    )

    recurrent_steps = int(
        recurrent_config.get(
            "steps",
            3,
        )
    )

    structural_dim = int(
        model_config.get(
            "structural_feature_dim",
            256,
        )
    )

    model = NTMN(
        num_classes=num_classes,
        steps=recurrent_steps,
        structural_dim=structural_dim,
    ).to(device)

    # =================================================================
    # Xavier initialization
    # =================================================================

    initialization_method = (
        training_config.get(
            "initialization",
            {},
        ).get(
            "method",
            "Xavier",
        )
    )

    if (
        str(
            initialization_method
        ).lower()
        == "xavier"
    ):

        initialize_xavier(
            model
        )

    # =================================================================
    # Optimizer
    # =================================================================

    optimizer_config = (
        training_config[
            "optimizer"
        ]
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=optimizer_config.get(
            "learning_rate",
            1e-4,
        ),
        weight_decay=optimizer_config.get(
            "weight_decay",
            1e-4,
        ),
    )

    # =================================================================
    # Scheduler
    # =================================================================

    scheduler_config = (
        training_config.get(
            "scheduler",
            {},
        )
    )

    epochs = int(
        training_config.get(
            "epochs",
            150,
        )
    )

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=scheduler_config.get(
                "min_learning_rate",
                1e-6,
            ),
        )
    )

    # =================================================================
    # Loss configuration
    # =================================================================

    loss_config = config.get(
        "loss",
        {},
    )

    # =================================================================
    # Checkpoint directory
    # =================================================================

    logging_config = config.get(
        "logging",
        {},
    )

    checkpoint_dir = Path(
        logging_config.get(
            "checkpoint_dir",
            "checkpoints",
        )
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # =================================================================
    # Training
    # =================================================================

    best_val_dice = -float(
        "inf"
    )

    for epoch in range(
        1,
        epochs + 1,
    ):

        model.train()

        running_loss = 0.0

        for batch in train_loader:

            images = batch[
                "image"
            ].to(
                device,
                non_blocking=True,
            )

            target = batch[
                "mask"
            ].to(
                device,
                non_blocking=True,
            )

            dataset_name = (
                get_dataset_name(
                    batch
                )
            )

            valid_classes = (
                get_valid_classes(
                    batch,
                    dataset_name,
                    num_classes,
                )
            )

            valid_classes_tensor = (
                valid_classes_to_tensor(
                    valid_classes,
                    device,
                )
            )

            # ---------------------------------------------------------
            # Unavailable class handling
            # ---------------------------------------------------------

            if not valid_classes[2]:

                invalid_ed = (
                    target == 2
                )

                if invalid_ed.any():

                    raise ValueError(
                        "The target contains "
                        "the unavailable class "
                        "(label 2) for "
                        f"{dataset_name}. "
                        "The preprocessing/label "
                        "harmonization step must "
                        "exclude this class rather "
                        "than encode it as a "
                        "foreground target."
                    )

            # ---------------------------------------------------------
            # Forward and loss
            # ---------------------------------------------------------

            optimizer.zero_grad(
                set_to_none=True
            )

            # Target is NOT passed into the model.
            # Ground truth is used by the loss only.
            output = model(
                images
            )

            loss = compute_loss(
                output,
                target,
                valid_classes_tensor,
                loss_config,
            )

            if not torch.isfinite(
                loss
            ):

                raise FloatingPointError(
                    f"Non-finite loss detected "
                    f"at epoch {epoch}: "
                    f"{loss.item()}"
                )

            # ---------------------------------------------------------
            # Backpropagation
            # ---------------------------------------------------------

            loss.backward()

            optimizer.step()

            running_loss += (
                loss.item()
            )

        # =================================================================
        # End epoch
        # =================================================================

        scheduler.step()

        train_loss = (
            running_loss
            / max(
                len(train_loader),
                1,
            )
        )

        # =================================================================
        # Validation
        # =================================================================

        val_dice = evaluate(
            model,
            val_loader,
            device,
            num_classes,
        )

        current_lr = (
            optimizer.param_groups[0][
                "lr"
            ]
        )

        print(
            f"Epoch "
            f"{epoch:03d}/{epochs:03d} | "
            f"Loss: {train_loss:.4f} | "
            f"Val Dice: {val_dice:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        # =================================================================
        # Save best checkpoint
        # =================================================================

        if val_dice > best_val_dice:

            best_val_dice = (
                val_dice
            )

            checkpoint = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_val_dice": best_val_dice,
                "seed": seed,
                "config": config,
            }

            checkpoint_path = (
                checkpoint_dir
                / "best.pt"
            )

            torch.save(
                checkpoint,
                checkpoint_path,
            )

            print(
                f"Saved best checkpoint: "
                f"{checkpoint_path}"
            )

    # =================================================================
    # Training complete
    # =================================================================

    print(
        "\nTraining complete."
    )

    print(
        f"Best validation Dice: "
        f"{best_val_dice:.4f}"
    )


# =====================================================================
# Entry point
# =====================================================================

if __name__ == "__main__":
    main()
