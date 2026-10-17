"""
NTMN Morphology Module.

Differentiable prediction-derived morphological descriptors for
Neural Tumor Morphogenesis Network (NTMN).

Six descriptor groups are represented by nine scalar values:

    1. Area             -> 1
    2. Centroid         -> 2 (cx, cy)
    3. Covariance       -> 3 (Cxx, Cxy, Cyy)
    4. Elongation       -> 1
    5. Soft boundary    -> 1
    6. Soft compactness -> 1

Total morphology dimension = 9.

The function operates on a soft foreground probability map with
shape [B, 1, H, W] and preserves differentiability so that the
morphological representation can participate in end-to-end training.
"""

import torch


def morphology(
    mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Extract differentiable morphology descriptors from a soft
    foreground probability map.

    Parameters
    ----------
    mask : torch.Tensor
        Soft foreground probability map with shape:

            [B, 1, H, W]

        Values are expected to lie in [0, 1].

    eps : float, optional
        Numerical stability constant.

    Returns
    -------
    torch.Tensor
        Nine-dimensional morphology representation with shape:

            [B, 9]

        Descriptor ordering:

            [area,
             centroid_x,
             centroid_y,
             covariance_xx,
             covariance_xy,
             covariance_yy,
             elongation,
             soft_boundary,
             soft_compactness]
    """

    # =================================================================
    # Input validation
    # =================================================================

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
            "mask must have exactly one foreground channel, "
            f"got {mask.shape[1]} channels."
        )

    batch_size, _, height, width = mask.shape

    if height < 2 or width < 2:
        raise ValueError(
            "mask spatial dimensions must be at least 2x2."
        )

    if not torch.is_floating_point(mask):
        mask = mask.float()

    device = mask.device
    dtype = mask.dtype

    # =================================================================
    # Coordinate grid
    # =================================================================

    yy, xx = torch.meshgrid(
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

    xx = xx.unsqueeze(0)
    yy = yy.unsqueeze(0)

    # =================================================================
    # Soft foreground
    # =================================================================

    m = mask[:, 0].clamp(
        min=0.0,
        max=1.0,
    )

    # =================================================================
    # 1. Soft area
    # =================================================================

    area = (
        m.sum(
            dim=(1, 2),
            keepdim=True,
        )
        + eps
    )

    area_scalar = area.flatten(1)

    # =================================================================
    # 2. Soft centroid
    # =================================================================

    centroid_x = (
        (m * xx).sum(
            dim=(1, 2),
            keepdim=True,
        )
        / area
    )

    centroid_y = (
        (m * yy).sum(
            dim=(1, 2),
            keepdim=True,
        )
        / area
    )

    centroid_x = centroid_x.flatten(1)
    centroid_y = centroid_y.flatten(1)

    # =================================================================
    # Centered coordinates
    # =================================================================

    dx = (
        xx
        - centroid_x.unsqueeze(-1).unsqueeze(-1)
    )

    dy = (
        yy
        - centroid_y.unsqueeze(-1).unsqueeze(-1)
    )

    # =================================================================
    # 3. Soft covariance
    #
    # Covariance representation:
    #
    #     Cxx
    #     Cxy
    #     Cyy
    # =================================================================

    covariance_xx = (
        (m * dx.pow(2)).sum(
            dim=(1, 2),
            keepdim=True,
        )
        / area
    )

    covariance_xy = (
        (m * dx * dy).sum(
            dim=(1, 2),
            keepdim=True,
        )
        / area
    )

    covariance_yy = (
        (m * dy.pow(2)).sum(
            dim=(1, 2),
            keepdim=True,
        )
        / area
    )

    covariance = torch.cat(
        [
            covariance_xx.flatten(1),
            covariance_xy.flatten(1),
            covariance_yy.flatten(1),
        ],
        dim=1,
    )

    # =================================================================
    # 4. Elongation
    #
    # Construct the 2 x 2 covariance matrix:
    #
    #     [ Cxx  Cxy ]
    #     [ Cxy  Cyy ]
    #
    # Elongation is defined as:
    #
    #     sqrt(lambda_max / lambda_min)
    # =================================================================

    covariance_matrix = torch.stack(
        [
            covariance_xx.flatten(),
            covariance_xy.flatten(),
            covariance_xy.flatten(),
            covariance_yy.flatten(),
        ],
        dim=1,
    ).reshape(
        batch_size,
        2,
        2,
    )

    eigenvalues = torch.linalg.eigvalsh(
        covariance_matrix
    )

    eigenvalues = eigenvalues.clamp_min(
        eps
    )

    lambda_min = eigenvalues[:, 0]
    lambda_max = eigenvalues[:, 1]

    elongation = (
        lambda_max
        / (lambda_min + eps)
    ).sqrt().unsqueeze(1)

    # =================================================================
    # 5. Soft boundary
    #
    # A differentiable boundary-magnitude surrogate is obtained from
    # horizontal and vertical first-order differences.
    #
    # This is intentionally a normalized soft boundary descriptor
    # rather than a hard binary perimeter.
    # =================================================================

    horizontal_gradient = torch.zeros_like(
        m
    )

    vertical_gradient = torch.zeros_like(
        m
    )

    horizontal_gradient[:, :, :-1] = torch.abs(
        m[:, :, 1:]
        - m[:, :, :-1]
    )

    vertical_gradient[:, :-1, :] = torch.abs(
        m[:, 1:, :]
        - m[:, :-1, :]
    )

    boundary = (
        horizontal_gradient
        + vertical_gradient
    ).mean(
        dim=(1, 2)
    ).unsqueeze(1)

    # =================================================================
    # 6. Soft compactness
    #
    # Compactness follows the area-to-boundary relationship:
    #
    #     4*pi*A / B^2
    #
    # where B is the differentiable soft-boundary magnitude.
    # =================================================================

    perimeter_proxy = (
        boundary + eps
    )

    compactness = (
        4.0
        * torch.pi
        * area_scalar
        / perimeter_proxy.pow(2)
    )

    # =================================================================
    # Final nine-dimensional morphology representation
    # =================================================================

    descriptors = torch.cat(
        [
            # 1: area
            area_scalar,

            # 2: centroid
            centroid_x,
            centroid_y,

            # 3: covariance
            covariance,

            # 4: elongation
            elongation,

            # 5: soft boundary
            boundary,

            # 6: soft compactness
            compactness,
        ],
        dim=1,
    )

    # =================================================================
    # Final consistency check
    # =================================================================

    if descriptors.shape != (
        batch_size,
        9,
    ):
        raise RuntimeError(
            "Expected morphology representation with shape "
            f"[{batch_size}, 9], "
            f"got {tuple(descriptors.shape)}."
        )

    # Ensure numerical validity.
    if not torch.isfinite(
        descriptors
    ).all():
        raise RuntimeError(
            "Non-finite values detected in the morphology "
            "descriptor representation."
        )

    return descriptors
