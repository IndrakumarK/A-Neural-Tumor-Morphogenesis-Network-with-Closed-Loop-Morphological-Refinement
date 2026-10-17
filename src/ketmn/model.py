"""
NTMN: Neural Tumor Morphogenesis Network.

Closed-loop morphology-conditioned recurrent refinement for
multimodal brain tumor segmentation in 2-D MRI.

The central mechanism is:

    intermediate prediction
        ↓
    prediction-derived morphology
        ↓
    learned morphological representation
        ↓
    morphological discrepancy
        ↓
    instance-specific structural correction
        ↓
    recurrent refinement

Input:
    T1, T1ce, T2, FLAIR

Input resolution:
    256 × 256

Segmentation classes:
    0 = Background
    1 = NCR/NET
    2 = Edema
    3 = Enhancing Tumor
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .morphology import morphology
from .topology import soft_euler_characteristic
from .uncertainty import normalized_predictive_entropy


# =====================================================================
# ConvGRU
# =====================================================================

class ConvGRU(nn.Module):
    """
    Convolutional GRU for recurrent spatial structural-state refinement.
    """

    def __init__(self, channels):
        super().__init__()

        self.channels = channels

        self.update_gate = nn.Conv2d(
            2 * channels,
            channels,
            kernel_size=3,
            padding=1,
        )

        self.reset_gate = nn.Conv2d(
            2 * channels,
            channels,
            kernel_size=3,
            padding=1,
        )

        self.candidate = nn.Conv2d(
            2 * channels,
            channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x, hidden=None):

        if hidden is None:
            hidden = torch.zeros_like(x)

        combined = torch.cat(
            [x, hidden],
            dim=1,
        )

        z = torch.sigmoid(
            self.update_gate(combined)
        )

        r = torch.sigmoid(
            self.reset_gate(combined)
        )

        candidate_input = torch.cat(
            [x, r * hidden],
            dim=1,
        )

        n = torch.tanh(
            self.candidate(candidate_input)
        )

        hidden = (
            (1.0 - z) * hidden
            + z * n
        )

        return hidden


# =====================================================================
# MBNSS
# =====================================================================

class MBNSS(nn.Module):
    """
    Morphology- and Boundary-Normal-Aware Skull Stripping.

    Produces a soft brain probability map used to restrict the
    multimodal MRI input before anatomical encoding.
    """

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(

            nn.Conv2d(
                4,
                32,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                32,
                32,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                32,
                1,
                kernel_size=1,
            ),
        )

    def forward(self, x):

        return torch.sigmoid(
            self.net(x)
        )


# =====================================================================
# Anatomical / Spatial Enhancement
# =====================================================================

class ASE(nn.Module):
    """
    Multimodal anatomical/spatial encoder.

    The encoder produces a 256-channel anatomical representation
    from the four MRI modalities.
    """

    def __init__(
        self,
        structural_dim=256,
        num_heads=8,
    ):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                4,
                64,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                128,
                structural_dim,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(structural_dim),
            nn.ReLU(inplace=True),
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=structural_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        self.norm1 = nn.LayerNorm(
            structural_dim
        )

        self.mlp = nn.Sequential(
            nn.Linear(
                structural_dim,
                structural_dim * 2,
            ),
            nn.GELU(),

            nn.Linear(
                structural_dim * 2,
                structural_dim,
            ),
        )

        self.norm2 = nn.LayerNorm(
            structural_dim
        )

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        batch_size, channels, height, width = (
            x.shape
        )

        tokens = x.flatten(
            2
        ).transpose(
            1,
            2,
        )

        attended, _ = self.attention(
            tokens,
            tokens,
            tokens,
            need_weights=False,
        )

        tokens = self.norm1(
            tokens + attended
        )

        tokens = self.norm2(
            tokens + self.mlp(tokens)
        )

        x = tokens.transpose(
            1,
            2,
        ).reshape(
            batch_size,
            channels,
            height,
            width,
        )

        return x


# =====================================================================
# Learned Morphological Representation
# =====================================================================

class MorphologicalKnowledgeEncoder(nn.Module):
    """
    Learns the expected morphological representation directly
    from the recurrent structural state.

    This is intentionally separate from the prediction-derived
    morphology extractor.

    Input:
        recurrent structural state [B, 256, H, W]

    Output:
        learned morphology representation [B, morphology_dim]
    """

    def __init__(
        self,
        structural_dim=256,
        morphology_dim=9,
    ):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(
            output_size=1
        )

        self.encoder = nn.Sequential(

            nn.Linear(
                structural_dim,
                128,
            ),
            nn.ReLU(inplace=True),

            nn.Linear(
                128,
                morphology_dim,
            ),
        )

    def forward(self, structural_state):

        pooled = self.pool(
            structural_state
        ).flatten(
            1
        )

        return self.encoder(
            pooled
        )


# =====================================================================
# Knowledge Discrepancy Projection
# =====================================================================

class KnowledgeDiscrepancyProjection(nn.Module):
    """
    Projects morphology discrepancy into the structural feature space.
    """

    def __init__(
        self,
        morphology_dim=9,
        structural_dim=256,
    ):
        super().__init__()

        self.projection = nn.Sequential(

            nn.Linear(
                morphology_dim,
                structural_dim,
            ),
            nn.ReLU(inplace=True),

            nn.Linear(
                structural_dim,
                structural_dim,
            ),
        )

    def forward(self, discrepancy):

        return self.projection(
            discrepancy
        )


# =====================================================================
# Morphology-Conditioned Feedback
# =====================================================================

class MorphologyFeedback(nn.Module):
    """
    Converts morphology discrepancy into an instance-specific
    residual structural correction.

    The discrepancy first generates a structural conditioning
    vector. This vector modulates the current recurrent state,
    after which a convolutional correction is generated.

    Formulation:

        q_t^K = MLP_K(D_t^K)

        F_t^K =
            Sbar_t ⊙ (1 + Broadcast(q_t^K))

        Delta S_t^K =
            Conv_K(F_t^K)

        S_t =
            Sbar_t + Delta S_t^K
    """

    def __init__(
        self,
        morphology_dim=9,
        structural_dim=256,
    ):
        super().__init__()

        self.discrepancy_projection = (
            KnowledgeDiscrepancyProjection(
                morphology_dim=morphology_dim,
                structural_dim=structural_dim,
            )
        )

        self.correction = nn.Sequential(

            nn.Conv2d(
                structural_dim,
                structural_dim,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(
                structural_dim
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                structural_dim,
                structural_dim,
                kernel_size=3,
                padding=1,
            ),
        )

    def forward(
        self,
        structural_state,
        discrepancy,
    ):

        conditioning = (
            self.discrepancy_projection(
                discrepancy
            )
        )

        conditioning = (
            conditioning
            .unsqueeze(-1)
            .unsqueeze(-1)
        )

        feedback_state = (
            structural_state
            * (1.0 + conditioning)
        )

        correction = self.correction(
            feedback_state
        )

        refined_state = (
            structural_state
            + correction
        )

        return (
            refined_state,
            conditioning,
            correction,
        )


# =====================================================================
# Topology-Aware Refinement
# =====================================================================

class TopologyRefinement(nn.Module):
    """
    Structural refinement using a differentiable soft Euler
    characteristic of the foreground prediction.
    """

    def __init__(
        self,
        structural_dim=256,
    ):
        super().__init__()

        self.feature_transform = nn.Sequential(

            nn.Conv2d(
                structural_dim,
                structural_dim,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(
                structural_dim
            ),
            nn.ReLU(inplace=True),
        )

        self.topology_projection = nn.Sequential(

            nn.Linear(
                1,
                structural_dim,
            ),
            nn.ReLU(inplace=True),

            nn.Linear(
                structural_dim,
                structural_dim,
            ),
        )

    def forward(
        self,
        structural_state,
        foreground_probability,
    ):

        transformed = (
            self.feature_transform(
                structural_state
            )
        )

        chi = soft_euler_characteristic(
            foreground_probability
        )

        if chi.ndim == 1:
            chi = chi.unsqueeze(1)

        topology_embedding = (
            self.topology_projection(
                chi
            )
        )

        topology_embedding = (
            topology_embedding
            .unsqueeze(-1)
            .unsqueeze(-1)
        )

        refined_state = (
            transformed
            + topology_embedding
        )

        return refined_state


# =====================================================================
# Connectivity Refinement
# =====================================================================

class ConnectivityRefinement(nn.Module):
    """
    Local 8-neighbour connectivity refinement.
    """

    def __init__(
        self,
        structural_dim=256,
    ):
        super().__init__()

        self.refinement = nn.Sequential(

            nn.Conv2d(
                structural_dim,
                structural_dim,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(
                structural_dim
            ),
            nn.ReLU(inplace=True),
        )

        self.connectivity_gate = nn.Sequential(

            nn.Conv2d(
                1,
                structural_dim,
                kernel_size=3,
                padding=1,
            ),
            nn.Sigmoid(),
        )

    def forward(
        self,
        structural_state,
        foreground_probability,
    ):

        shifts = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

        connectivity = torch.zeros_like(
            foreground_probability
        )

        for dy, dx in shifts:

            shifted = torch.roll(
                foreground_probability,
                shifts=(dy, dx),
                dims=(-2, -1),
            )

            connectivity = (
                connectivity
                + foreground_probability
                * shifted
            )

        connectivity = (
            connectivity
            / len(shifts)
        )

        gate = self.connectivity_gate(
            connectivity
        )

        refined = self.refinement(
            structural_state
        )

        return (
            structural_state
            + gate * refined
        )


# =====================================================================
# Uncertainty-Guided Refinement
# =====================================================================

class UncertaintyRefinement(nn.Module):
    """
    Uncertainty-guided structural refinement.

    Normalized predictive entropy is combined with the final
    morphology discrepancy to generate an instance-specific
    refinement gate.
    """

    def __init__(
        self,
        structural_dim=256,
    ):
        super().__init__()

        self.refinement = nn.Sequential(

            nn.Conv2d(
                structural_dim,
                structural_dim,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(
                structural_dim
            ),
            nn.ReLU(inplace=True),
        )

        self.uncertainty_gate = nn.Sequential(

            nn.Conv2d(
                1,
                structural_dim,
                kernel_size=3,
                padding=1,
            ),
            nn.Sigmoid(),
        )

    def forward(
        self,
        structural_state,
        auxiliary_probability,
        morphology_discrepancy,
    ):

        entropy = (
            normalized_predictive_entropy(
                auxiliary_probability
            )
        )

        # -------------------------------------------------------------
        # Scalar morphology-discrepancy magnitude
        # -------------------------------------------------------------

        discrepancy_magnitude = (
            morphology_discrepancy.abs().mean(
                dim=1,
                keepdim=True,
            )
        )

        discrepancy_map = (
            discrepancy_magnitude
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand(
                -1,
                -1,
                structural_state.shape[-2],
                structural_state.shape[-1],
            )
        )

        # -------------------------------------------------------------
        # Combined uncertainty signal
        # -------------------------------------------------------------

        uncertainty_map = (
            entropy
            + discrepancy_map
        )

        gate = self.uncertainty_gate(
            uncertainty_map
        )

        correction = self.refinement(
            structural_state
        )

        refined_state = (
            structural_state
            + gate * correction
        )

        return (
            refined_state,
            entropy,
        )


# =====================================================================
# NTMN
# =====================================================================

class NTMN(nn.Module):
    """
    Neural Tumor Morphogenesis Network.

    Closed-loop morphology-conditioned recurrent refinement.
    """

    def __init__(
        self,
        num_classes=4,
        steps=3,
        structural_dim=256,
        morphology_dim=9,
    ):
        super().__init__()

        self.steps = steps
        self.structural_dim = structural_dim
        self.num_classes = num_classes
        self.morphology_dim = morphology_dim

        # -------------------------------------------------------------
        # 1. MBNSS
        # -------------------------------------------------------------

        self.mbnss = MBNSS()

        # -------------------------------------------------------------
        # 2. Anatomical / Spatial Encoder
        # -------------------------------------------------------------

        self.ase = ASE(
            structural_dim=structural_dim,
            num_heads=8,
        )

        # -------------------------------------------------------------
        # 3. Initial structural projection
        # -------------------------------------------------------------

        self.initial_projection = nn.Conv2d(
            structural_dim,
            structural_dim,
            kernel_size=1,
        )

        # -------------------------------------------------------------
        # 4. ConvGRU recurrent structural evolution
        # -------------------------------------------------------------

        self.gru = ConvGRU(
            structural_dim
        )

        # -------------------------------------------------------------
        # 5. Intermediate segmentation head
        # -------------------------------------------------------------

        self.segmentation_head = nn.Conv2d(
            structural_dim,
            num_classes,
            kernel_size=1,
        )

        # -------------------------------------------------------------
        # 6. Learned morphological representation
        # -------------------------------------------------------------

        self.morphology_encoder = (
            MorphologicalKnowledgeEncoder(
                structural_dim=structural_dim,
                morphology_dim=morphology_dim,
            )
        )

        # -------------------------------------------------------------
        # 7. Morphology-conditioned feedback
        # -------------------------------------------------------------

        self.morphology_feedback = (
            MorphologyFeedback(
                morphology_dim=morphology_dim,
                structural_dim=structural_dim,
            )
        )

        # -------------------------------------------------------------
        # 8. Topology refinement
        # -------------------------------------------------------------

        self.topology_refinement = (
            TopologyRefinement(
                structural_dim=structural_dim
            )
        )

        # -------------------------------------------------------------
        # 9. Connectivity refinement
        # -------------------------------------------------------------

        self.connectivity_refinement = (
            ConnectivityRefinement(
                structural_dim=structural_dim
            )
        )

        # -------------------------------------------------------------
        # 10. Auxiliary prediction branch
        # -------------------------------------------------------------

        self.auxiliary_head = nn.Conv2d(
            structural_dim,
            num_classes,
            kernel_size=1,
        )

        # -------------------------------------------------------------
        # 11. Uncertainty-guided refinement
        # -------------------------------------------------------------

        self.uncertainty_refinement = (
            UncertaintyRefinement(
                structural_dim=structural_dim
            )
        )

        # -------------------------------------------------------------
        # 12. Final decoder
        # -------------------------------------------------------------

        self.decoder = nn.Sequential(

            nn.Conv2d(
                structural_dim,
                128,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                128,
                num_classes,
                kernel_size=1,
            ),
        )

    # -----------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------

    def forward(
        self,
        x,
        target=None,
    ):

        # =============================================================
        # 1. MBNSS
        # =============================================================

        brain_mask = self.mbnss(
            x
        )

        masked_input = (
            x * brain_mask
        )

        # =============================================================
        # 2. ASE
        # =============================================================

        structural_state = self.ase(
            masked_input
        )

        # =============================================================
        # 3. Initial structural state
        # =============================================================

        structural_state = (
            self.initial_projection(
                structural_state
            )
        )

        # ASE downsamples by a factor of four.
        # Restore the structural state to the original
        # image resolution before recurrent refinement.

        structural_state = F.interpolate(
            structural_state,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        # =============================================================
        # 4. Recurrent morphology-conditioned refinement
        # =============================================================

        states = []

        stage_logits = []
        stage_probs = []

        morphologies = []
        learned_morphologies = []
        discrepancies = []

        feedback_conditionings = []
        feedback_corrections = []

        previous_state = None

        for _ in range(self.steps):

            # ---------------------------------------------------------
            # ConvGRU state evolution
            # ---------------------------------------------------------

            recurrent_state = self.gru(
                structural_state,
                previous_state,
            )

            # ---------------------------------------------------------
            # Intermediate segmentation prediction
            # ---------------------------------------------------------

            logits_t = (
                self.segmentation_head(
                    recurrent_state
                )
            )

            probability_t = torch.softmax(
                logits_t,
                dim=1,
            )

            # ---------------------------------------------------------
            # Foreground probability
            #
            # Combined tumor foreground is used for morphology
            # feedback.
            # ---------------------------------------------------------

            foreground_t = (
                probability_t[:, 1:]
                .sum(
                    dim=1,
                    keepdim=True,
                )
            )

            # ---------------------------------------------------------
            # Prediction-derived morphology
            #
            # morphology() provides the six descriptor groups:
            #
            #   area
            #   centroid
            #   covariance
            #   elongation
            #   soft boundary
            #   soft compactness
            #
            # with the corresponding scalar representation
            # expected by morphology_dim=9.
            # ---------------------------------------------------------

            predicted_morphology = morphology(
                foreground_t
            )

            # ---------------------------------------------------------
            # Learned morphological expectation
            #
            # IMPORTANT:
            # This is derived from the learned recurrent structural
            # state, not from predicted_morphology itself.
            # ---------------------------------------------------------

            learned_morphology = (
                self.morphology_encoder(
                    recurrent_state
                )
            )

            # ---------------------------------------------------------
            # Morphological discrepancy
            # ---------------------------------------------------------

            discrepancy = (
                learned_morphology
                - predicted_morphology
            )

            # ---------------------------------------------------------
            # Closed-loop morphology feedback
            # ---------------------------------------------------------

            (
                refined_state,
                feedback_conditioning,
                feedback_correction,
            ) = self.morphology_feedback(
                recurrent_state,
                discrepancy,
            )

            # ---------------------------------------------------------
            # Save stage outputs
            # ---------------------------------------------------------

            states.append(
                refined_state
            )

            stage_logits.append(
                logits_t
            )

            stage_probs.append(
                probability_t
            )

            morphologies.append(
                predicted_morphology
            )

            learned_morphologies.append(
                learned_morphology
            )

            discrepancies.append(
                discrepancy
            )

            feedback_conditionings.append(
                feedback_conditioning
            )

            feedback_corrections.append(
                feedback_correction
            )

            # ---------------------------------------------------------
            # Corrected state becomes the input to the next recurrent
            # refinement stage.
            # ---------------------------------------------------------

            structural_state = (
                refined_state
            )

            previous_state = (
                refined_state
            )

        # =============================================================
        # 5. Final intermediate prediction
        # =============================================================

        final_stage_probability = (
            stage_probs[-1]
        )

        final_foreground = (
            final_stage_probability[:, 1:]
            .sum(
                dim=1,
                keepdim=True,
            )
        )

        # =============================================================
        # 6. Topology refinement
        # =============================================================

        structural_state = (
            self.topology_refinement(
                structural_state,
                final_foreground,
            )
        )

        # =============================================================
        # 7. Connectivity refinement
        # =============================================================

        structural_state = (
            self.connectivity_refinement(
                structural_state,
                final_foreground,
            )
        )

        # =============================================================
        # 8. Auxiliary prediction
        # =============================================================

        auxiliary_logits = (
            self.auxiliary_head(
                structural_state
            )
        )

        auxiliary_probability = (
            torch.softmax(
                auxiliary_logits,
                dim=1,
            )
        )

        # =============================================================
        # 9. Uncertainty-guided refinement
        # =============================================================

        final_morphology_discrepancy = (
            discrepancies[-1]
        )

        (
            structural_state,
            predictive_entropy,
        ) = self.uncertainty_refinement(
            structural_state,
            auxiliary_probability,
            final_morphology_discrepancy,
        )

        # =============================================================
        # 10. Final decoder
        # =============================================================

        logits = self.decoder(
            structural_state
        )

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        # =============================================================
        # 11. Scalar discrepancy magnitude for reporting
        # =============================================================

        discrepancy_magnitude = (
            final_morphology_discrepancy
            .abs()
            .sum(
                dim=1,
            )
        )

        # =============================================================
        # 12. Return
        # =============================================================

        return {

            # ---------------------------------------------------------
            # Final segmentation
            # ---------------------------------------------------------

            "logits": logits,

            "prob": probabilities,

            # ---------------------------------------------------------
            # MBNSS
            # ---------------------------------------------------------

            "brain_mask": brain_mask,

            # ---------------------------------------------------------
            # Recurrent structural states
            # ---------------------------------------------------------

            "states": states,

            # ---------------------------------------------------------
            # Intermediate segmentation predictions
            # ---------------------------------------------------------

            "stage_logits": stage_logits,

            "stage_probs": stage_probs,

            # ---------------------------------------------------------
            # Prediction-derived morphology
            # ---------------------------------------------------------

            "morphology": morphologies[-1],

            "morphologies": morphologies,

            # ---------------------------------------------------------
            # Learned morphology
            # ---------------------------------------------------------

            "expected_morphology": (
                learned_morphologies[-1]
            ),

            "expected_morphologies": (
                learned_morphologies
            ),

            # ---------------------------------------------------------
            # Vector-valued morphology discrepancy
            # ---------------------------------------------------------

            "knowledge_discrepancy": (
                torch.stack(
                    discrepancies,
                    dim=1,
                )
            ),

            "descriptor_discrepancy": (
                final_morphology_discrepancy
            ),

            # ---------------------------------------------------------
            # Scalar morphology discrepancy magnitude
            # ---------------------------------------------------------

            "morphology_discrepancy": (
                discrepancy_magnitude
            ),

            # ---------------------------------------------------------
            # Feedback information
            # ---------------------------------------------------------

            "feedback_conditionings": (
                feedback_conditionings
            ),

            "feedback_corrections": (
                feedback_corrections
            ),

            # ---------------------------------------------------------
            # Final structural state
            # ---------------------------------------------------------

            "final_structural_state": (
                structural_state
            ),

            # ---------------------------------------------------------
            # Auxiliary uncertainty branch
            # ---------------------------------------------------------

            "auxiliary_logits": (
                auxiliary_logits
            ),

            "auxiliary_prob": (
                auxiliary_probability
            ),

            "uncertainty": (
                predictive_entropy
            ),
        }


# =====================================================================
# Backward-compatible alias
# =====================================================================

KE_NTMN = NTMN
