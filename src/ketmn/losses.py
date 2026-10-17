"""
NTMN: Neural Tumor Morphogenesis Network.

Loss functions for multimodal brain tumor segmentation,
morphology-conditioned refinement, topology, connectivity,
uncertainty consistency, and MBNSS.
"""

import torch
import torch.nn.functional as F


# =====================================================================
# Valid-Class Utilities
# =====================================================================

def _prepare_valid_classes(
    valid_classes,
    batch_size,
    num_classes,
    device,
):
    """
    Prepare dataset-specific class-validity information.

    valid_classes:
        None
            All classes are considered valid.

        [C]
            Same validity mask is applied to every sample.

        [B, C]
            Sample-specific validity mask.

    Returns:
        Boolean tensor of shape [B, C].
    """

    if valid_classes is None:
        return torch.ones(
            batch_size,
            num_classes,
            dtype=torch.bool,
            device=device,
        )

    valid_classes = torch.as_tensor(
        valid_classes,
        dtype=torch.bool,
        device=device,
    )

    if valid_classes.ndim == 1:

        if valid_classes.numel() != num_classes:
            raise ValueError(
                "valid_classes must contain one entry per class."
            )

        valid_classes = valid_classes.unsqueeze(0).expand(
            batch_size,
            -1,
        )

    elif valid_classes.ndim == 2:

        if valid_classes.shape != (
            batch_size,
            num_classes,
        ):
            raise ValueError(
                "valid_classes must have shape "
                f"[{batch_size}, {num_classes}], "
                f"got {tuple(valid_classes.shape)}."
            )

    else:

        raise ValueError(
            "valid_classes must be a 1-D or 2-D tensor."
        )

    return valid_classes


def _foreground_valid_mask(
    valid_classes,
    batch_size,
    num_classes,
    device,
):
    """
    Return validity information for foreground classes only.

    Background class 0 is excluded.
    """

    valid_classes = _prepare_valid_classes(
        valid_classes,
        batch_size,
        num_classes,
        device,
    )

    return valid_classes[:, 1:]


# =====================================================================
# Foreground Dice Loss
# =====================================================================

def dice_loss(
    pred,
    target,
    num_classes=4,
    valid_classes=None,
    smooth=1e-6,
):
    """
    Foreground multiclass soft Dice loss.

    Background is excluded.

    Dataset-specific unavailable foreground classes are
    excluded through valid_classes.
    """

    probs = F.softmax(
        pred,
        dim=1,
    )

    target_one_hot = F.one_hot(
        target.long(),
        num_classes=num_classes,
    ).permute(
        0,
        3,
        1,
        2,
    ).float()

    # -------------------------------------------------------------
    # Foreground classes
    # -------------------------------------------------------------

    probs_fg = probs[:, 1:]
    target_fg = target_one_hot[:, 1:]

    valid_fg = _foreground_valid_mask(
        valid_classes,
        pred.shape[0],
        num_classes,
        pred.device,
    )

    # -------------------------------------------------------------
    # Dice per foreground class
    # -------------------------------------------------------------

    intersection = (
        probs_fg * target_fg
    ).sum(
        dim=(0, 2, 3)
    )

    denominator = (
        probs_fg.sum(
            dim=(0, 2, 3)
        )
        + target_fg.sum(
            dim=(0, 2, 3)
        )
    )

    dice = (
        2.0 * intersection
        + smooth
    ) / (
        denominator
        + smooth
    )

    # A foreground class is included when it is valid
    # for at least one sample in the batch.
    class_validity = (
        valid_fg.sum(dim=0) > 0
    )

    if not torch.any(class_validity):
        return pred.new_tensor(0.0)

    return 1.0 - dice[class_validity].mean()


# =====================================================================
# Smoothness Loss
# =====================================================================

def smoothness_loss(pred):
    """
    Spatial smoothness regularization over foreground probabilities.
    """

    probs = F.softmax(
        pred,
        dim=1,
    )

    # Exclude background.
    probs = probs[:, 1:]

    delta_x = (
        probs[:, :, :, 1:]
        - probs[:, :, :, :-1]
    )

    delta_y = (
        probs[:, :, 1:, :]
        - probs[:, :, :-1, :]
    )

    return (
        torch.abs(delta_x).mean()
        + torch.abs(delta_y).mean()
    )


# =====================================================================
# Structural Stability Loss
# =====================================================================

def structural_stability_loss(
    current_state,
    previous_state,
):
    """
    Penalize excessive changes between consecutive recurrent
    structural states.

    The previous state is detached so that this term does not
    propagate gradients backward through the previous recurrent stage.
    """

    if previous_state is None:
        return current_state.new_tensor(0.0)

    return torch.mean(
        torch.abs(
            current_state
            - previous_state.detach()
        )
    )


# =====================================================================
# Morphology Estimation Loss
# =====================================================================

def morphology_estimation_loss(
    predicted_morphology,
    target_morphology,
    valid_classes=None,
    num_classes=4,
):
    """
    Morphology estimation loss.

    Supports:
        [B, C, D]
        [B, C*D]

    Dataset-specific unavailable classes are excluded.
    """

    if valid_classes is None:
        return F.mse_loss(
            predicted_morphology,
            target_morphology,
        )

    valid_classes = _prepare_valid_classes(
        valid_classes,
        predicted_morphology.shape[0],
        num_classes,
        predicted_morphology.device,
    )

    # -------------------------------------------------------------
    # Case 1:
    # [B, C, D]
    # -------------------------------------------------------------

    if predicted_morphology.ndim == 3:

        if predicted_morphology.shape[1] != num_classes:
            raise ValueError(
                "Class-wise morphology tensor must have "
                "shape [B, C, D]."
            )

        mask = valid_classes[:, :, None].to(
            predicted_morphology.dtype
        )

        squared_error = (
            predicted_morphology
            - target_morphology
        ) ** 2

        denominator = mask.sum().clamp_min(1.0)

        return (
            squared_error * mask
        ).sum() / denominator

    # -------------------------------------------------------------
    # Case 2:
    # [B, C*D]
    # -------------------------------------------------------------

    if predicted_morphology.ndim == 2:

        if (
            predicted_morphology.shape[1]
            % num_classes
            != 0
        ):
            return F.mse_loss(
                predicted_morphology,
                target_morphology,
            )

        descriptor_dim = (
            predicted_morphology.shape[1]
            // num_classes
        )

        pred = predicted_morphology.reshape(
            predicted_morphology.shape[0],
            num_classes,
            descriptor_dim,
        )

        target = target_morphology.reshape(
            target_morphology.shape[0],
            num_classes,
            descriptor_dim,
        )

        mask = valid_classes[:, :, None].to(
            pred.dtype
        )

        squared_error = (
            pred - target
        ) ** 2

        denominator = (
            mask.sum()
            * descriptor_dim
        ).clamp_min(1.0)

        return (
            squared_error * mask
        ).sum() / denominator

    return F.mse_loss(
        predicted_morphology,
        target_morphology,
    )


# =====================================================================
# Knowledge Consistency Loss
# =====================================================================

def knowledge_consistency_loss(
    predicted_morphology,
    expected_morphology,
    valid_classes=None,
    num_classes=4,
):
    """
    Knowledge consistency between predicted morphology and
    learned/expected morphology.
    """

    return morphology_estimation_loss(
        predicted_morphology=predicted_morphology,
        target_morphology=expected_morphology,
        valid_classes=valid_classes,
        num_classes=num_classes,
    )


# =====================================================================
# Differentiable Topology
# =====================================================================

def _soft_euler_characteristic(
    class_probability,
):
    """
    Differentiable soft Euler characteristic.

    The implementation uses an 8-connected neighborhood consisting
    of horizontal, vertical, and diagonal adjacency terms.
    """

    # -------------------------------------------------------------
    # Soft vertices
    # -------------------------------------------------------------

    vertices = class_probability.sum(
        dim=(-2, -1)
    )

    # -------------------------------------------------------------
    # Soft edges
    # -------------------------------------------------------------

    right = torch.minimum(
        class_probability[:, :, :, 1:],
        class_probability[:, :, :, :-1],
    ).sum(
        dim=(-2, -1)
    )

    down = torch.minimum(
        class_probability[:, :, 1:, :],
        class_probability[:, :, :-1, :],
    ).sum(
        dim=(-2, -1)
    )

    down_right = torch.minimum(
        class_probability[:, :, 1:, 1:],
        class_probability[:, :, :-1, :-1],
    ).sum(
        dim=(-2, -1)
    )

    down_left = torch.minimum(
        class_probability[:, :, 1:, :-1],
        class_probability[:, :, :-1, 1:],
    ).sum(
        dim=(-2, -1)
    )

    edges = (
        right
        + down
        + down_right
        + down_left
    )

    # -------------------------------------------------------------
    # Soft faces
    # -------------------------------------------------------------

    p1 = class_probability[:, :, :-1, :-1]
    p2 = class_probability[:, :, :-1, 1:]
    p3 = class_probability[:, :, 1:, :-1]
    p4 = class_probability[:, :, 1:, 1:]

    faces = torch.minimum(
        torch.minimum(p1, p2),
        torch.minimum(p3, p4),
    ).sum(
        dim=(-2, -1)
    )

    return vertices - edges + faces


def _hard_euler_characteristic(
    class_mask,
):
    """
    Hard Euler characteristic used for the ground-truth target.
    """

    class_mask = class_mask.float()

    # -------------------------------------------------------------
    # Vertices
    # -------------------------------------------------------------

    vertices = class_mask.sum(
        dim=(-2, -1)
    )

    # -------------------------------------------------------------
    # Edges
    # -------------------------------------------------------------

    right = (
        class_mask[:, :, :, 1:]
        * class_mask[:, :, :, :-1]
    ).sum(
        dim=(-2, -1)
    )

    down = (
        class_mask[:, :, 1:, :]
        * class_mask[:, :, :-1, :]
    ).sum(
        dim=(-2, -1)
    )

    down_right = (
        class_mask[:, :, 1:, 1:]
        * class_mask[:, :, :-1, :-1]
    ).sum(
        dim=(-2, -1)
    )

    down_left = (
        class_mask[:, :, 1:, :-1]
        * class_mask[:, :, :-1, 1:]
    ).sum(
        dim=(-2, -1)
    )

    edges = (
        right
        + down
        + down_right
        + down_left
    )

    # -------------------------------------------------------------
    # Faces
    # -------------------------------------------------------------

    p1 = class_mask[:, :, :-1, :-1]
    p2 = class_mask[:, :, :-1, 1:]
    p3 = class_mask[:, :, 1:, :-1]
    p4 = class_mask[:, :, 1:, 1:]

    faces = (
        p1
        * p2
        * p3
        * p4
    ).sum(
        dim=(-2, -1)
    )

    return vertices - edges + faces


def topology_loss(
    prediction,
    target,
    valid_classes=None,
):
    """
    Topology-aware loss based on soft Euler characteristic.

    The loss is evaluated independently for each foreground class
    and excludes dataset-specific unavailable classes.
    """

    probs = F.softmax(
        prediction,
        dim=1,
    )

    num_classes = probs.shape[1]

    valid_fg = _foreground_valid_mask(
        valid_classes,
        prediction.shape[0],
        num_classes,
        prediction.device,
    )

    total = prediction.new_tensor(0.0)
    count = prediction.new_tensor(0.0)

    for class_idx in range(
        1,
        num_classes,
    ):

        if not torch.any(
            valid_fg[:, class_idx - 1]
        ):
            continue

        pred_class = probs[
            :,
            class_idx:class_idx + 1,
        ]

        target_class = (
            target == class_idx
        ).float().unsqueeze(1)

        pred_chi = _soft_euler_characteristic(
            pred_class
        )

        target_chi = _hard_euler_characteristic(
            target_class
        )

        class_loss = torch.abs(
            pred_chi - target_chi
        )

        sample_valid = valid_fg[
            :,
            class_idx - 1,
        ].float()

        total = total + (
            class_loss * sample_valid
        ).sum()

        count = count + sample_valid.sum()

    return (
        total
        / count.clamp_min(1.0)
    )


# =====================================================================
# Local Connectivity Loss
# =====================================================================

def _local_adjacency(
    class_probability,
):
    """
    Differentiable local adjacency using 8-neighbor relationships.
    """

    right = (
        class_probability[:, :, :, 1:]
        * class_probability[:, :, :, :-1]
    ).mean(
        dim=(-2, -1)
    )

    down = (
        class_probability[:, :, 1:, :]
        * class_probability[:, :, :-1, :]
    ).mean(
        dim=(-2, -1)
    )

    down_right = (
        class_probability[:, :, 1:, 1:]
        * class_probability[:, :, :-1, :-1]
    ).mean(
        dim=(-2, -1)
    )

    down_left = (
        class_probability[:, :, 1:, :-1]
        * class_probability[:, :, :-1, 1:]
    ).mean(
        dim=(-2, -1)
    )

    return (
        right
        + down
        + down_right
        + down_left
    ) / 4.0


def _hard_local_adjacency(
    class_mask,
):
    """
    Hard local adjacency computed from the reference mask.
    """

    class_mask = class_mask.float()

    right = (
        class_mask[:, :, :, 1:]
        * class_mask[:, :, :, :-1]
    ).mean(
        dim=(-2, -1)
    )

    down = (
        class_mask[:, :, 1:, :]
        * class_mask[:, :, :-1, :]
    ).mean(
        dim=(-2, -1)
    )

    down_right = (
        class_mask[:, :, 1:, 1:]
        * class_mask[:, :, :-1, :-1]
    ).mean(
        dim=(-2, -1)
    )

    down_left = (
        class_mask[:, :, 1:, :-1]
        * class_mask[:, :, :-1, 1:]
    ).mean(
        dim=(-2, -1)
    )

    return (
        right
        + down
        + down_right
        + down_left
    ) / 4.0


def connectivity_loss(
    prediction,
    target,
    valid_classes=None,
):
    """
    Local connectivity loss for foreground classes.

    Dataset-specific unavailable classes are excluded.
    """

    probs = F.softmax(
        prediction,
        dim=1,
    )

    num_classes = probs.shape[1]

    valid_fg = _foreground_valid_mask(
        valid_classes,
        prediction.shape[0],
        num_classes,
        prediction.device,
    )

    total = prediction.new_tensor(0.0)
    count = prediction.new_tensor(0.0)

    for class_idx in range(
        1,
        num_classes,
    ):

        if not torch.any(
            valid_fg[:, class_idx - 1]
        ):
            continue

        pred_class = probs[
            :,
            class_idx:class_idx + 1,
        ]

        target_class = (
            target == class_idx
        ).float().unsqueeze(1)

        pred_adj = _local_adjacency(
            pred_class
        )

        target_adj = _hard_local_adjacency(
            target_class
        )

        class_loss = torch.abs(
            pred_adj - target_adj
        ).squeeze(1)

        sample_valid = valid_fg[
            :,
            class_idx - 1,
        ].float()

        total = total + (
            class_loss * sample_valid
        ).sum()

        count = count + sample_valid.sum()

    return (
        total
        / count.clamp_min(1.0)
    )


# =====================================================================
# Uncertainty-Error Consistency Loss
# =====================================================================

def uncertainty_consistency_loss(
    prediction,
    target,
    valid_region=None,
):
    """
    Match normalized predictive entropy with local prediction error.

    Normalized predictive entropy:
        H(p) / log(C)

    Prediction error:
        1 - p(y_i | x_i)

    valid_region can be supplied to exclude invalid or ignored
    pixels from the uncertainty consistency objective.
    """

    probs = F.softmax(
        prediction,
        dim=1,
    )

    num_classes = probs.shape[1]

    # -------------------------------------------------------------
    # Normalized predictive entropy
    # -------------------------------------------------------------

    entropy = -(
        probs
        * torch.log(
            probs.clamp_min(1e-8)
        )
    ).sum(
        dim=1
    )

    max_entropy = torch.log(
        torch.tensor(
            float(num_classes),
            device=probs.device,
            dtype=probs.dtype,
        )
    )

    normalized_entropy = (
        entropy
        / max_entropy.clamp_min(1e-8)
    )

    # -------------------------------------------------------------
    # Ground-truth class probability
    # -------------------------------------------------------------

    target_clamped = target.long().clamp(
        min=0,
        max=num_classes - 1,
    )

    target_probability = probs.gather(
        dim=1,
        index=target_clamped.unsqueeze(1),
    ).squeeze(1)

    # -------------------------------------------------------------
    # Local prediction error
    # -------------------------------------------------------------

    prediction_error = (
        1.0
        - target_probability
    )

    # -------------------------------------------------------------
    # Entropy-error consistency
    # -------------------------------------------------------------

    consistency_error = torch.abs(
        normalized_entropy
        - prediction_error
    )

    # -------------------------------------------------------------
    # Valid region
    # -------------------------------------------------------------

    if valid_region is None:

        valid_region = torch.ones_like(
            consistency_error
        )

    else:

        valid_region = valid_region.to(
            consistency_error.dtype
        )

    denominator = (
        valid_region.sum()
        .clamp_min(1.0)
    )

    return (
        consistency_error
        * valid_region
    ).sum() / denominator


# =====================================================================
# MBNSS Utilities
# =====================================================================

def _soft_boundary(
    probability,
):
    """
    Differentiable boundary magnitude.
    """

    dx = torch.zeros_like(
        probability
    )

    dy = torch.zeros_like(
        probability
    )

    dx[:, :, :, 1:] = (
        probability[:, :, :, 1:]
        - probability[:, :, :, :-1]
    )

    dy[:, :, 1:, :] = (
        probability[:, :, 1:, :]
        - probability[:, :, :-1, :]
    )

    return torch.sqrt(
        dx ** 2
        + dy ** 2
        + 1e-8
    )


def _soft_normal(
    probability,
):
    """
    Differentiable boundary-normal representation.
    """

    dx = torch.zeros_like(
        probability
    )

    dy = torch.zeros_like(
        probability
    )

    dx[:, :, :, 1:] = (
        probability[:, :, :, 1:]
        - probability[:, :, :, :-1]
    )

    dy[:, :, 1:, :] = (
        probability[:, :, 1:, :]
        - probability[:, :, :-1, :]
    )

    magnitude = torch.sqrt(
        dx ** 2
        + dy ** 2
        + 1e-8
    )

    nx = dx / magnitude
    ny = dy / magnitude

    return torch.cat(
        [
            nx,
            ny,
        ],
        dim=1,
    )


def _brain_morphology(
    probability,
):
    """
    Compute differentiable brain-mask morphology descriptors.

    Descriptor order:
        area
        centroid_x
        centroid_y
        covariance_xx
        covariance_xy
        covariance_yx
        covariance_yy
        elongation
        soft_boundary
        soft_compactness

    Soft compactness:
        4*pi*A / (B^2 + epsilon)

    where:
        A = soft area
        B = soft boundary measure
    """

    # Expected input:
    # [B, 1, H, W]
    probability = probability[:, 0]

    batch_size, height, width = (
        probability.shape
    )

    device = probability.device
    dtype = probability.dtype

    # -------------------------------------------------------------
    # Coordinate grid
    # -------------------------------------------------------------

    y_coords, x_coords = torch.meshgrid(
        torch.arange(
            height,
            device=device,
            dtype=dtype,
        ),
        torch.arange(
            width,
            device=device,
            dtype=dtype,
        ),
        indexing="ij",
    )

    coords = torch.stack(
        [
            x_coords,
            y_coords,
        ],
        dim=-1,
    )

    # -------------------------------------------------------------
    # Area
    # -------------------------------------------------------------

    area = probability.sum(
        dim=(1, 2),
        keepdim=True,
    )

    denominator = (
        area + 1e-6
    )

    # -------------------------------------------------------------
    # Centroid
    # -------------------------------------------------------------

    centroid = (
        probability.unsqueeze(-1)
        * coords.unsqueeze(0)
    ).sum(
        dim=(1, 2)
    ) / denominator.squeeze(-1)

    # -------------------------------------------------------------
    # Covariance
    # -------------------------------------------------------------

    centered = (
        coords.unsqueeze(0)
        - centroid[:, None, None, :]
    )

    covariance = (
        probability.unsqueeze(-1).unsqueeze(-1)
        * centered.unsqueeze(-1)
        * centered.unsqueeze(-2)
    ).sum(
        dim=(1, 2)
    ) / denominator

    # -------------------------------------------------------------
    # Elongation
    # -------------------------------------------------------------

    eigenvalues = torch.linalg.eigvalsh(
        covariance
    )

    lambda_min = eigenvalues[:, 0]
    lambda_max = eigenvalues[:, 1]

    elongation = torch.sqrt(
        lambda_max
        / (lambda_min + 1e-6)
    ).unsqueeze(-1)

    # -------------------------------------------------------------
    # Soft boundary
    # -------------------------------------------------------------

    boundary_map = _soft_boundary(
        probability.unsqueeze(1)
    )

    boundary_measure = boundary_map.sum(
        dim=(1, 2, 3),
        keepdim=True,
    )

    # -------------------------------------------------------------
    # Soft compactness
    #
    # Manuscript definition:
    #     C = 4*pi*A / (B^2 + epsilon)
    # -------------------------------------------------------------

    compactness = (
        4.0
        * torch.pi
        * area
    ) / (
        boundary_measure ** 2
        + 1e-6
    )

    # -------------------------------------------------------------
    # Descriptor vector
    # -------------------------------------------------------------

    area_n = area.squeeze(-1)

    centroid_n = centroid

    covariance_n = covariance.reshape(
        batch_size,
        -1,
    )

    elongation_n = elongation

    boundary_n = boundary_measure.squeeze(
        -1
    )

    compactness_n = compactness.squeeze(
        -1
    )

    return torch.cat(
        [
            area_n,
            centroid_n,
            covariance_n,
            elongation_n,
            boundary_n,
            compactness_n,
        ],
        dim=-1,
    )


# =====================================================================
# MBNSS Loss
# =====================================================================

def mbnss_loss(
    brain_prediction,
    brain_target,
    morphology_mean=None,
    morphology_std=None,
):
    """
    Complete MBNSS objective.

    Components:
        Dice
        Boundary-position loss
        Boundary-normal consistency
        Global morphology loss

    Optional morphology_mean and morphology_std allow the
    training-set-only descriptor normalization described in
    the manuscript.

    The normalization statistics must be computed from the
    training data only.
    """

    # -------------------------------------------------------------
    # Convert brain logits to probability of brain foreground
    # -------------------------------------------------------------

    if brain_prediction.shape[1] == 2:

        brain_probability = F.softmax(
            brain_prediction,
            dim=1,
        )[:, 1:2]

    else:

        brain_probability = torch.sigmoid(
            brain_prediction
        )

    brain_target = brain_target.float()

    if brain_target.ndim == 3:
        brain_target = brain_target.unsqueeze(1)

    # -------------------------------------------------------------
    # Dice loss
    # -------------------------------------------------------------

    intersection = (
        brain_probability
        * brain_target
    ).sum(
        dim=(1, 2, 3)
    )

    denominator = (
        brain_probability.sum(
            dim=(1, 2, 3)
        )
        + brain_target.sum(
            dim=(1, 2, 3)
        )
    )

    dice = (
        2.0 * intersection
        + 1e-6
    ) / (
        denominator
        + 1e-6
    )

    dice_loss_value = (
        1.0
        - dice.mean()
    )

    # -------------------------------------------------------------
    # Boundary position loss
    # -------------------------------------------------------------

    predicted_boundary = _soft_boundary(
        brain_probability
    )

    target_boundary = _soft_boundary(
        brain_target
    )

    boundary_position_loss = torch.abs(
        predicted_boundary
        - target_boundary
    ).mean()

    # -------------------------------------------------------------
    # Boundary-normal consistency loss
    # -------------------------------------------------------------

    predicted_normal = _soft_normal(
        brain_probability
    )

    target_normal = _soft_normal(
        brain_target
    )

    normal_cosine = (
        predicted_normal
        * target_normal
    ).sum(
        dim=1
    )

    boundary_normal_loss = (
        1.0
        - normal_cosine
    ).mean()

    # -------------------------------------------------------------
    # Global morphology loss
    # -------------------------------------------------------------

    predicted_morphology = _brain_morphology(
        brain_probability
    )

    target_morphology = _brain_morphology(
        brain_target
    )

    # -------------------------------------------------------------
    # Optional training-set descriptor normalization
    # -------------------------------------------------------------

    if (
        morphology_mean is not None
        and morphology_std is not None
    ):

        morphology_mean = torch.as_tensor(
            morphology_mean,
            dtype=predicted_morphology.dtype,
            device=predicted_morphology.device,
        )

        morphology_std = torch.as_tensor(
            morphology_std,
            dtype=predicted_morphology.dtype,
            device=predicted_morphology.device,
        )

        predicted_morphology = (
            predicted_morphology
            - morphology_mean
        ) / (
            morphology_std
            + 1e-6
        )

        target_morphology = (
            target_morphology
            - morphology_mean
        ) / (
            morphology_std
            + 1e-6
        )

    morphology_loss = torch.abs(
        predicted_morphology
        - target_morphology
    ).mean()

    # -------------------------------------------------------------
    # Complete MBNSS objective
    # -------------------------------------------------------------

    return (
        1.00 * dice_loss_value
        + 0.50 * boundary_position_loss
        + 0.25 * boundary_normal_loss
        + 0.25 * morphology_loss
    )


# =====================================================================
# Complete NTMN Segmentation Loss
# =====================================================================

def total_loss(
    prediction,
    target,
    structural_stability=0.0,
    morphology_estimation=0.0,
    knowledge_consistency=0.0,
    topology=0.0,
    connectivity=0.0,
    smoothness=0.0,
    uncertainty_consistency=0.0,
    valid_classes=None,
):
    """
    Complete NTMN segmentation objective.

    Weighted terms:

        Foreground Dice             1.00
        Structural stability        0.20
        Morphology estimation      0.20
        Knowledge consistency      0.20
        Topology                    0.10
        Connectivity                0.10
        Smoothness                  0.05
        Uncertainty consistency    0.05
    """

    foreground_dice = dice_loss(
        prediction,
        target,
        num_classes=4,
        valid_classes=valid_classes,
    )

    return (
        1.00 * foreground_dice
        + 0.20 * structural_stability
        + 0.20 * morphology_estimation
        + 0.20 * knowledge_consistency
        + 0.10 * topology
        + 0.10 * connectivity
        + 0.05 * smoothness
        + 0.05 * uncertainty_consistency
    )
