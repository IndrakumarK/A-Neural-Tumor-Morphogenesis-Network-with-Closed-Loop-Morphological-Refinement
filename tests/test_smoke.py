"""
NTMN Unit Tests.

Basic forward-pass and morphology tests for the
Neural Tumor Morphogenesis Network (NTMN).
"""

import torch

from src.ntmn.model import NTMN
from src.ntmn.morphology import morphology


# =====================================================================
# Model forward-pass test
# =====================================================================

def test_forward():
    """
    Verify that NTMN produces the expected output structure and shapes.
    """

    model = NTMN()

    x = torch.randn(
        2,
        4,
        64,
        64,
    )

    target = torch.randint(
        low=0,
        high=4,
        size=(2, 64, 64),
    )

    output = model(
        x,
        target,
    )

    # Final segmentation
    assert output["prob"].shape == (
        2,
        4,
        64,
        64,
    )

    assert output["logits"].shape == (
        2,
        4,
        64,
        64,
    )

    # MBNSS brain mask
    assert output["brain_mask"].shape == (
        2,
        1,
        64,
        64,
    )

    # Three recurrent refinement stages
    assert len(
        output["stage_probs"]
    ) == 3

    assert len(
        output["states"]
    ) == 3


# =====================================================================
# Morphology test
# =====================================================================

def test_morphology():
    """
    Verify the nine-dimensional morphology representation.
    """

    mask = torch.sigmoid(
        torch.randn(
            2,
            1,
            32,
            32,
        )
    )

    features = morphology(
        mask
    )

    assert features.shape == (
        2,
        9,
    )

    assert torch.isfinite(
        features
    ).all()


# =====================================================================
# Morphology descriptor dimension test
# =====================================================================

def test_morphology_descriptor_dimension():
    """
    Explicitly verify that the morphology representation contains
    nine scalar descriptors:

        area                  1
        centroid              2
        covariance            3
        elongation            1
        soft boundary         1
        soft compactness      1
        --------------------------------
        total                 9
    """

    mask = torch.zeros(
        1,
        1,
        32,
        32,
    )

    mask[:, :, 8:24, 8:24] = 1.0

    features = morphology(
        mask
    )

    assert features.shape[-1] == 9

    assert torch.isfinite(
        features
    ).all()


# =====================================================================
# Gradient-flow test
# =====================================================================

def test_morphology_gradient_flow():
    """
    Verify that morphology descriptors remain differentiable.
    """

    mask = torch.sigmoid(
        torch.randn(
            2,
            1,
            32,
            32,
            requires_grad=True,
        )
    )

    features = morphology(
        mask
    )

    loss = features.mean()

    loss.backward()

    assert mask.grad is not None
