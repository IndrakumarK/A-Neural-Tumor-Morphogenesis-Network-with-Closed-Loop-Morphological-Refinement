"""
NTMN Uncertainty Module.

Normalized predictive entropy and uncertainty-error consistency
for the Neural Tumor Morphogenesis Network (NTMN).

The uncertainty formulation is based on normalized predictive entropy:

    U(x) = -sum_c p_c(x) log(p_c(x)) / log(C)

An uncertainty-error consistency loss encourages the predictive
uncertainty to correspond to the model's local prediction error.
"""

import torch


# =====================================================================
# Normalized predictive entropy
# =====================================================================

def entropy(
    prob: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute normalized predictive entropy.

    Parameters
    ----------
    prob : torch.Tensor
        Class probabilities with shape [B, C, H, W].

    eps : float, optional
        Numerical stability constant.

    Returns
    -------
    torch.Tensor
        Normalized predictive entropy map with shape [B, H, W].

    Notes
    -----
    The entropy is normalized by log(C), where C is the number
    of segmentation classes. Therefore, the theoretical maximum
    entropy is approximately 1.
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

    num_classes = prob.shape[1]

    if num_classes < 2:
        raise ValueError(
            "prob must contain at least two classes."
        )

    if not torch.is_floating_point(prob):
        prob = prob.float()

    # -----------------------------------------------------------------
    # Numerical stabilization
    # -----------------------------------------------------------------

    p = prob.clamp(
        min=eps,
        max=1.0,
    )

    # -----------------------------------------------------------------
    # Predictive entropy
    #
    #     H(p) = -sum p_c log(p_c)
    # -----------------------------------------------------------------

    entropy_map = -(
        p * torch.log(p)
    ).sum(
        dim=1
    )

    # -----------------------------------------------------------------
    # Normalize by maximum entropy log(C)
    # -----------------------------------------------------------------

    normalization = torch.log(
        torch.tensor(
            float(num_classes),
            device=prob.device,
            dtype=prob.dtype,
        )
    ).clamp_min(
        eps
    )

    normalized_entropy = (
        entropy_map
        / normalization
    )

    return normalized_entropy


# =====================================================================
# Public normalized predictive entropy alias
# =====================================================================

def normalized_predictive_entropy(
    prob: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Public alias for normalized predictive entropy.
    """

    return entropy(
        prob,
        eps=eps,
    )


# =====================================================================
# Uncertainty-error consistency
# =====================================================================

def consistency(
    prob: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute uncertainty-error consistency.

    Parameters
    ----------
    prob : torch.Tensor
        Segmentation probabilities with shape [B, C, H, W].

    target : torch.Tensor
        Ground-truth class labels with shape [B, H, W].

    valid_mask : torch.Tensor, optional
        Optional pixel-validity mask with shape [B, H, W].

        Values:
            1 -> include pixel
            0 -> exclude pixel

        If None, all pixels are included.

    eps : float, optional
        Numerical stability constant.

    Returns
    -------
    torch.Tensor
        Scalar uncertainty-error consistency loss.

    Formulation
    -----------
    Predictive uncertainty:

        U = normalized predictive entropy

    Ground-truth prediction error:

        e = 1 - p_y

    Consistency loss:

        L_U = mean |U - e|

    When valid_mask is supplied, only valid pixels contribute to
    the loss.
    """

    if not torch.is_tensor(prob):
        raise TypeError(
            "prob must be a torch.Tensor."
        )

    if not torch.is_tensor(target):
        raise TypeError(
            "target must be a torch.Tensor."
        )

    if prob.ndim != 4:
        raise ValueError(
            "prob must have shape [B, C, H, W], "
            f"got {tuple(prob.shape)}"
        )

    if target.ndim != 3:
        raise ValueError(
            "target must have shape [B, H, W], "
            f"got {tuple(target.shape)}"
        )

    batch_size, num_classes, height, width = prob.shape

    # -----------------------------------------------------------------
    # Spatial compatibility
    # -----------------------------------------------------------------

    if target.shape != (
        batch_size,
        height,
        width,
    ):
        raise ValueError(
            "Probability and target spatial dimensions do not match. "
            f"prob={tuple(prob.shape)}, "
            f"target={tuple(target.shape)}"
        )

    # -----------------------------------------------------------------
    # Target class validity
    # -----------------------------------------------------------------

    if target.numel() == 0:
        return prob.new_tensor(
            0.0
        )

    if target.min() < 0:
        raise ValueError(
            "Target contains negative class indices."
        )

    if target.max() >= num_classes:
        raise ValueError(
            "Target contains class indices outside "
            f"[0, {num_classes - 1}]."
        )

    # -----------------------------------------------------------------
    # Normalized predictive entropy
    # -----------------------------------------------------------------

    uncertainty = entropy(
        prob,
        eps=eps,
    )

    # -----------------------------------------------------------------
    # Probability assigned to the ground-truth class
    # -----------------------------------------------------------------

    target_probability = (
        prob.gather(
            dim=1,
            index=target.long().unsqueeze(1),
        )
        .squeeze(1)
    )

    # -----------------------------------------------------------------
    # Soft prediction error
    #
    # e = 1 - p(y | x)
    # -----------------------------------------------------------------

    error = (
        1.0
        - target_probability
    )

    # -----------------------------------------------------------------
    # Absolute uncertainty-error discrepancy
    # -----------------------------------------------------------------

    discrepancy = (
        uncertainty
        - error
    ).abs()

    # -----------------------------------------------------------------
    # Optional valid-pixel masking
    # -----------------------------------------------------------------

    if valid_mask is not None:

        if not torch.is_tensor(valid_mask):
            raise TypeError(
                "valid_mask must be a torch.Tensor."
            )

        if valid_mask.ndim != 3:
            raise ValueError(
                "valid_mask must have shape [B, H, W]."
            )

        if valid_mask.shape != (
            batch_size,
            height,
            width,
        ):
            raise ValueError(
                "valid_mask must have the same spatial shape "
                "as target."
            )

        mask = valid_mask.to(
            device=prob.device,
            dtype=discrepancy.dtype,
        )

        valid_count = mask.sum()

        if valid_count <= 0:
            return prob.new_tensor(
                0.0
            )

        return (
            discrepancy * mask
        ).sum() / (
            valid_count + eps
        )

    # -----------------------------------------------------------------
    # Standard mean when all pixels are valid
    # -----------------------------------------------------------------

    return discrepancy.mean()


# =====================================================================
# Public uncertainty-consistency alias
# =====================================================================

def uncertainty_consistency(
    prob: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Public alias for uncertainty-error consistency.
    """

    return consistency(
        prob=prob,
        target=target,
        valid_mask=valid_mask,
        eps=eps,
    )
