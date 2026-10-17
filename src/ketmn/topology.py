"""
NTMN Topology and Connectivity Module.

Differentiable slice-level topology and local connectivity constraints
for the Neural Tumor Morphogenesis Network (NTMN).

The implementation is strictly two-dimensional because NTMN operates
on individual MRI slices.

Components
----------
1. Soft Euler characteristic
2. Topology loss
3. Local soft adjacency
4. Hard local adjacency
5. Connectivity loss
"""

import torch


# =====================================================================
# Soft Euler characteristic
# =====================================================================

def soft_euler(
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute a differentiable soft Euler characteristic.

    Parameters
    ----------
    mask : torch.Tensor
        Soft foreground probability map with shape [B, 1, H, W].

    Returns
    -------
    torch.Tensor
        Soft Euler characteristic for each sample, shape [B].

    Notes
    -----
    The implementation uses an 8-neighborhood relation:

        χ = V - E + F

    where:

        V = soft foreground vertex contribution
        E = soft 8-neighbor edge contribution
        F = soft 2x2 face contribution

    This is a slice-level topological descriptor and does not model
    inter-slice or volumetric topology.
    """

    if not torch.is_tensor(mask):
        raise TypeError(
            "mask must be a torch.Tensor."
        )

    if mask.ndim != 4:
        raise ValueError(
            "mask must have shape [B, 1, H, W], "
            f"got {tuple(mask.shape)}"
        )

    if mask.shape[1] != 1:
        raise ValueError(
            "mask must contain exactly one foreground channel, "
            f"got {mask.shape[1]}."
        )

    if mask.shape[2] < 2 or mask.shape[3] < 2:
        raise ValueError(
            "mask spatial dimensions must be at least 2x2."
        )

    p = mask.clamp(
        min=0.0,
        max=1.0,
    )

    p = p[:, 0]

    # -----------------------------------------------------------------
    # Soft vertex contribution
    # -----------------------------------------------------------------

    vertices = p.sum(
        dim=(1, 2)
    )

    # -----------------------------------------------------------------
    # Soft 8-neighborhood edge contribution
    #
    # Each undirected neighbor relation is counted once.
    # -----------------------------------------------------------------

    horizontal_edges = (
        p[:, :, :-1]
        * p[:, :, 1:]
    )

    vertical_edges = (
        p[:, :-1, :]
        * p[:, 1:, :]
    )

    diagonal_down_right = (
        p[:, :-1, :-1]
        * p[:, 1:, 1:]
    )

    diagonal_down_left = (
        p[:, :-1, 1:]
        * p[:, 1:, :-1]
    )

    edges = (
        horizontal_edges.sum(dim=(1, 2))
        + vertical_edges.sum(dim=(1, 2))
        + diagonal_down_right.sum(dim=(1, 2))
        + diagonal_down_left.sum(dim=(1, 2))
    )

    # -----------------------------------------------------------------
    # Soft 2x2 face contribution
    #
    # A face receives a high contribution when all four pixels in the
    # corresponding 2x2 neighborhood have high foreground probability.
    # -----------------------------------------------------------------

    faces = (
        p[:, :-1, :-1]
        * p[:, :-1, 1:]
        * p[:, 1:, :-1]
        * p[:, 1:, 1:]
    )

    face_count = faces.sum(
        dim=(1, 2)
    )

    # -----------------------------------------------------------------
    # Euler characteristic
    # -----------------------------------------------------------------

    return (
        vertices
        - edges
        + face_count
    )


# =====================================================================
# Topology loss
# =====================================================================

def topology_loss(
    prob: torch.Tensor,
    target: torch.Tensor,
    valid_classes=None,
) -> torch.Tensor:
    """
    Compute the topology loss using soft Euler characteristics.

    Parameters
    ----------
    prob : torch.Tensor
        Segmentation probabilities with shape [B, C, H, W].

    target : torch.Tensor
        Integer segmentation labels with shape [B, H, W].

    valid_classes : optional
        Boolean class-validity mask with length C.

    Returns
    -------
    torch.Tensor
        Scalar topology loss.

    Notes
    -----
    Background class 0 is excluded.

    Classes unavailable in a particular dataset can be excluded
    through valid_classes.
    """

    if not torch.is_tensor(prob):
        raise TypeError(
            "prob must be a torch.Tensor."
        )

    if prob.ndim != 4:
        raise ValueError(
            "prob must have shape [B, C, H, W], "
            f"got {tuple(prob.shape)}"
        )

    if not torch.is_tensor(target):
        raise TypeError(
            "target must be a torch.Tensor."
        )

    if target.ndim != 3:
        raise ValueError(
            "target must have shape [B, H, W], "
            f"got {tuple(target.shape)}"
        )

    batch_size, num_classes, height, width = prob.shape

    if target.shape != (
        batch_size,
        height,
        width,
    ):
        raise ValueError(
            "target spatial dimensions must match prob. "
            f"Got prob={tuple(prob.shape)}, "
            f"target={tuple(target.shape)}."
        )

    # -----------------------------------------------------------------
    # Valid class mask
    # -----------------------------------------------------------------

    if valid_classes is None:

        valid_classes = torch.ones(
            num_classes,
            dtype=torch.bool,
            device=prob.device,
        )

    elif not torch.is_tensor(valid_classes):

        valid_classes = torch.as_tensor(
            valid_classes,
            dtype=torch.bool,
            device=prob.device,
        )

    valid_classes = valid_classes.to(
        device=prob.device,
        dtype=torch.bool,
    )

    if valid_classes.numel() != num_classes:
        raise ValueError(
            "valid_classes must contain one value per class. "
            f"Expected {num_classes}, "
            f"got {valid_classes.numel()}."
        )

    losses = []

    # -----------------------------------------------------------------
    # Foreground classes only
    # -----------------------------------------------------------------

    for class_id in range(
        1,
        num_classes,
    ):

        if not valid_classes[class_id]:
            continue

        predicted_mask = prob[
            :,
            class_id:class_id + 1,
        ]

        target_mask = (
            (target == class_id)
            .to(prob.dtype)
            .unsqueeze(1)
        )

        predicted_euler = soft_euler(
            predicted_mask
        )

        target_euler = soft_euler(
            target_mask
        )

        class_loss = (
            predicted_euler
            - target_euler
        ).abs().mean()

        losses.append(
            class_loss
        )

    # -----------------------------------------------------------------
    # No valid foreground classes
    # -----------------------------------------------------------------

    if not losses:
        return prob.new_tensor(
            0.0
        )

    return torch.stack(
        losses
    ).mean()


# =====================================================================
# Local soft adjacency
# =====================================================================

def local_soft_adjacency(
    probability: torch.Tensor,
) -> torch.Tensor:
    """
    Compute differentiable local 8-neighborhood adjacency.

    Parameters
    ----------
    probability : torch.Tensor
        Single-class probability map [B, 1, H, W].

    Returns
    -------
    torch.Tensor
        Mean local adjacency score for each sample, shape [B].
    """

    if not torch.is_tensor(probability):
        raise TypeError(
            "probability must be a torch.Tensor."
        )

    if probability.ndim != 4:
        raise ValueError(
            "probability must have shape [B, 1, H, W], "
            f"got {tuple(probability.shape)}"
        )

    if probability.shape[1] != 1:
        raise ValueError(
            "probability must contain exactly one class."
        )

    if probability.shape[2] < 2 or probability.shape[3] < 2:
        raise ValueError(
            "probability spatial dimensions must be at least 2x2."
        )

    p = probability.clamp(
        min=0.0,
        max=1.0,
    )

    adjacency_values = []

    # -----------------------------------------------------------------
    # Horizontal neighbors
    # -----------------------------------------------------------------

    horizontal = (
        p[:, :, :, :-1]
        * p[:, :, :, 1:]
    )

    adjacency_values.append(
        horizontal.mean(
            dim=(1, 2, 3)
        )
    )

    # -----------------------------------------------------------------
    # Vertical neighbors
    # -----------------------------------------------------------------

    vertical = (
        p[:, :, :-1, :]
        * p[:, :, 1:, :]
    )

    adjacency_values.append(
        vertical.mean(
            dim=(1, 2, 3)
        )
    )

    # -----------------------------------------------------------------
    # Diagonal down-right neighbors
    # -----------------------------------------------------------------

    diagonal_down_right = (
        p[:, :, :-1, :-1]
        * p[:, :, 1:, 1:]
    )

    adjacency_values.append(
        diagonal_down_right.mean(
            dim=(1, 2, 3)
        )
    )

    # -----------------------------------------------------------------
    # Diagonal down-left neighbors
    # -----------------------------------------------------------------

    diagonal_down_left = (
        p[:, :, :-1, 1:]
        * p[:, :, 1:, :-1]
    )

    adjacency_values.append(
        diagonal_down_left.mean(
            dim=(1, 2, 3)
        )
    )

    adjacency = torch.stack(
        adjacency_values,
        dim=1,
    )

    return adjacency.mean(
        dim=1
    )


# =====================================================================
# Hard local adjacency
# =====================================================================

def hard_local_adjacency(
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute local 8-neighborhood adjacency for a binary mask.

    Parameters
    ----------
    mask : torch.Tensor
        Binary foreground mask with shape [B, 1, H, W].

    Returns
    -------
    torch.Tensor
        Mean local adjacency score for each sample, shape [B].
    """

    if not torch.is_tensor(mask):
        raise TypeError(
            "mask must be a torch.Tensor."
        )

    if mask.ndim != 4:
        raise ValueError(
            "mask must have shape [B, 1, H, W], "
            f"got {tuple(mask.shape)}"
        )

    if mask.shape[1] != 1:
        raise ValueError(
            "mask must contain exactly one class."
        )

    if mask.shape[2] < 2 or mask.shape[3] < 2:
        raise ValueError(
            "mask spatial dimensions must be at least 2x2."
        )

    p = mask.to(
        dtype=torch.float32
    ).clamp(
        min=0.0,
        max=1.0,
    )

    adjacency_values = []

    # Horizontal
    horizontal = (
        p[:, :, :, :-1]
        * p[:, :, :, 1:]
    )

    adjacency_values.append(
        horizontal.mean(
            dim=(1, 2, 3)
        )
    )

    # Vertical
    vertical = (
        p[:, :, :-1, :]
        * p[:, :, 1:, :]
    )

    adjacency_values.append(
        vertical.mean(
            dim=(1, 2, 3)
        )
    )

    # Diagonal down-right
    diagonal_down_right = (
        p[:, :, :-1, :-1]
        * p[:, :, 1:, 1:]
    )

    adjacency_values.append(
        diagonal_down_right.mean(
            dim=(1, 2, 3)
        )
    )

    # Diagonal down-left
    diagonal_down_left = (
        p[:, :, :-1, 1:]
        * p[:, :, 1:, :-1]
    )

    adjacency_values.append(
        diagonal_down_left.mean(
            dim=(1, 2, 3)
        )
    )

    adjacency = torch.stack(
        adjacency_values,
        dim=1,
    )

    return adjacency.mean(
        dim=1
    )


# =====================================================================
# Connectivity loss
# =====================================================================

def connectivity_loss(
    prob: torch.Tensor,
    target: torch.Tensor | None = None,
    valid_classes=None,
) -> torch.Tensor:
    """
    Compute local connectivity consistency loss.

    Parameters
    ----------
    prob : torch.Tensor
        Segmentation probabilities [B, C, H, W].

    target : torch.Tensor, optional
        Ground-truth labels [B, H, W].

    valid_classes : optional
        Boolean class-validity mask with length C.

    Returns
    -------
    torch.Tensor
        Scalar connectivity loss.

    Notes
    -----
    When target is supplied, the predicted local adjacency is compared
    with the corresponding ground-truth adjacency.

    When target is None, the loss encourages local foreground
    connectivity directly.
    """

    if not torch.is_tensor(prob):
        raise TypeError(
            "prob must be a torch.Tensor."
        )

    if prob.ndim != 4:
        raise ValueError(
            "prob must have shape [B, C, H, W], "
            f"got {tuple(prob.shape)}"
        )

    batch_size, num_classes, height, width = prob.shape

    if target is not None:

        if not torch.is_tensor(target):
            raise TypeError(
                "target must be a torch.Tensor."
            )

        if target.ndim != 3:
            raise ValueError(
                "target must have shape [B, H, W]."
            )

        if target.shape != (
            batch_size,
            height,
            width,
        ):
            raise ValueError(
                "target spatial dimensions must match prob."
            )

    # -----------------------------------------------------------------
    # Valid classes
    # -----------------------------------------------------------------

    if valid_classes is None:

        valid_classes = torch.ones(
            num_classes,
            dtype=torch.bool,
            device=prob.device,
        )

    elif not torch.is_tensor(valid_classes):

        valid_classes = torch.as_tensor(
            valid_classes,
            dtype=torch.bool,
            device=prob.device,
        )

    valid_classes = valid_classes.to(
        device=prob.device,
        dtype=torch.bool,
    )

    if valid_classes.numel() != num_classes:
        raise ValueError(
            "valid_classes must contain one value per class. "
            f"Expected {num_classes}, "
            f"got {valid_classes.numel()}."
        )

    losses = []

    # -----------------------------------------------------------------
    # Foreground classes
    # -----------------------------------------------------------------

    for class_id in range(
        1,
        num_classes,
    ):

        if not valid_classes[class_id]:
            continue

        predicted = prob[
            :,
            class_id:class_id + 1,
        ]

        predicted_adjacency = (
            local_soft_adjacency(
                predicted
            )
        )

        if target is not None:

            target_class = (
                (target == class_id)
                .to(prob.dtype)
                .unsqueeze(1)
            )

            target_adjacency = (
                hard_local_adjacency(
                    target_class
                ).to(prob.dtype)
            )

            class_loss = (
                predicted_adjacency
                - target_adjacency
            ).abs().mean()

        else:

            class_loss = (
                1.0
                - predicted_adjacency
            ).mean()

        losses.append(
            class_loss
        )

    # -----------------------------------------------------------------
    # No valid foreground classes
    # -----------------------------------------------------------------

    if not losses:
        return prob.new_tensor(
            0.0
        )

    return torch.stack(
        losses
    ).mean()


# =====================================================================
# Public compatibility alias
# =====================================================================

def soft_euler_characteristic(
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Public alias for soft_euler().
    """

    return soft_euler(
        mask
    )
