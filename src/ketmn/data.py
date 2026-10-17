
Required files per slice
------------------------

T1.npy
T1ce.npy
T2.npy
FLAIR.npy
mask.npy
spacing.npy

Optional but required for supervised MBNSS training
---------------------------------------------------

brain_mask.npy

The segmentation mask uses the four-class output space:

    0 = Background
    1 = NCR/NET
    2 = Edema
    3 = Enhancing Tumor

Dataset-specific class availability:

    UPenn-GBM : [True, True, True, True]
    TCGA-GBM  : [True, True, True, True]
    MOTUM     : [True, True, False, True]

The class mapping above must correspond to the actual preprocessing
label mapping used to construct mask.npy. This loader does not
silently remap labels.

Pixel spacing
-------------

spacing.npy must contain:

    [row_spacing_mm, column_spacing_mm]

and must correspond to the final 2-D image/mask arrays stored in
the slice directory.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from scipy.ndimage import affine_transform
except ImportError as exc:
    raise ImportError(
        "SciPy is required for rotation and scaling augmentation. "
        "Install it with: pip install scipy"
    ) from exc


class BrainTumorDataset(Dataset):
    """
    Patient- and slice-aware 2-D multimodal MRI dataset.

    Each Dataset item corresponds to ONE 2-D MRI slice.

    Patient identity is preserved separately through `patient_id`,
    allowing downstream evaluation code to aggregate predictions
    across slices belonging to the same patient.
    """

    MODALITIES = (
        "T1",
        "T1ce",
        "T2",
        "FLAIR",
    )

    # ------------------------------------------------------------------
    # Four-class segmentation space
    # ------------------------------------------------------------------

    NUM_CLASSES = 4

    # IMPORTANT:
    # These labels must already have been created during preprocessing.
    # This loader intentionally performs NO automatic semantic remapping.
    CLASS_NAMES = {
        0: "Background",
        1: "NCR/NET",
        2: "Edema",
        3: "Enhancing Tumor",
    }

    # ------------------------------------------------------------------
    # Dataset-specific class availability
    # ------------------------------------------------------------------

    DATASET_VALID_CLASSES = {
        "UPenn": (
            True,
            True,
            True,
            True,
        ),
        "UPenn-GBM": (
            True,
            True,
            True,
            True,
        ),
        "TCGA": (
            True,
            True,
            True,
            True,
        ),
        "TCGA-GBM": (
            True,
            True,
            True,
            True,
        ),
        "MOTUM": (
            True,
            True,
            False,
            True,
        ),
    }

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    def __init__(
        self,
        root,
        image_size=(256, 256),
        num_classes=4,
        augment=False,
        dataset_name="UPenn-GBM",
        rotation_degrees=10.0,
        scale_range=(0.90, 1.10),
        require_brain_mask=False,
    ):
        super().__init__()

        self.root = Path(root)

        self.image_size = tuple(
            int(x)
            for x in image_size
        )

        self.num_classes = int(
            num_classes
        )

        self.augment = bool(
            augment
        )

        self.dataset_name = str(
            dataset_name
        )

        self.rotation_degrees = float(
            rotation_degrees
        )

        self.scale_range = tuple(
            float(x)
            for x in scale_range
        )

        self.require_brain_mask = bool(
            require_brain_mask
        )

        # --------------------------------------------------------------
        # Configuration validation
        # --------------------------------------------------------------

        if self.num_classes != self.NUM_CLASSES:
            raise ValueError(
                "NTMN currently expects exactly four segmentation "
                "classes: Background, NCR/NET, Edema, and Enhancing Tumor."
            )

        if self.dataset_name not in self.DATASET_VALID_CLASSES:
            raise ValueError(
                f"Unsupported dataset '{self.dataset_name}'. "
                f"Expected one of "
                f"{list(self.DATASET_VALID_CLASSES.keys())}."
            )

        if len(self.image_size) != 2:
            raise ValueError(
                f"image_size must contain exactly two values, "
                f"got {self.image_size}."
            )

        if self.rotation_degrees < 0:
            raise ValueError(
                "rotation_degrees must be non-negative."
            )

        if (
            len(self.scale_range) != 2
            or self.scale_range[0] <= 0
            or self.scale_range[1] <= 0
            or self.scale_range[0]
            > self.scale_range[1]
        ):
            raise ValueError(
                f"Invalid scale_range: {self.scale_range}"
            )

        # --------------------------------------------------------------
        # Root directory
        # --------------------------------------------------------------

        if not self.root.exists():
            raise FileNotFoundError(
                f"Dataset directory does not exist: {self.root}"
            )

        if not self.root.is_dir():
            raise NotADirectoryError(
                f"Dataset root is not a directory: {self.root}"
            )

        # --------------------------------------------------------------
        # Discover patient/slice samples
        # --------------------------------------------------------------

        self.samples = self._discover_samples()

        if not self.samples:
            raise RuntimeError(
                f"No valid patient/slice samples were found under "
                f"{self.root}"
            )

    # ==================================================================
    # Sample discovery
    # ==================================================================

    def _discover_samples(self):
        """
        Discover samples using:

            root/
                patient_id/
                    slice_id/
                        T1.npy
                        ...

        Returns
        -------
        list of dict
            One dictionary per 2-D slice.
        """

        samples = []

        patient_dirs = sorted(
            path
            for path in self.root.iterdir()
            if path.is_dir()
        )

        for patient_dir in patient_dirs:

            patient_id = patient_dir.name

            slice_dirs = sorted(
                path
                for path in patient_dir.iterdir()
                if path.is_dir()
            )

            for slice_dir in slice_dirs:

                slice_id = slice_dir.name

                samples.append(
                    {
                        "patient_id": patient_id,
                        "slice_id": slice_id,
                        "case_id": (
                            f"{patient_id}_{slice_id}"
                        ),
                        "path": slice_dir,
                    }
                )

        return samples

    # ==================================================================
    # Dataset length
    # ==================================================================

    def __len__(self):
        """
        Number of 2-D slices.
        """

        return len(
            self.samples
        )

    # ==================================================================
    # Case/slice loading
    # ==================================================================

    def _load_sample(
        self,
        sample,
    ):
        """
        Load the four MRI modalities, segmentation mask,
        optional brain mask, and physical pixel spacing.
        """

        slice_dir = sample["path"]

        images = []

        # --------------------------------------------------------------
        # MRI modalities
        # --------------------------------------------------------------

        for modality in self.MODALITIES:

            path = (
                slice_dir
                / f"{modality}.npy"
            )

            if not path.exists():
                raise FileNotFoundError(
                    f"Missing modality file: {path}"
                )

            image = np.load(
                path
            ).astype(
                np.float32
            )

            if image.ndim != 2:
                raise ValueError(
                    f"{path} must contain a 2-D array, "
                    f"got shape {image.shape}."
                )

            images.append(
                image
            )

        # --------------------------------------------------------------
        # Segmentation mask
        # --------------------------------------------------------------

        mask_path = (
            slice_dir
            / "mask.npy"
        )

        if not mask_path.exists():
            raise FileNotFoundError(
                f"Missing segmentation mask: {mask_path}"
            )

        mask = np.load(
            mask_path
        )

        if mask.ndim != 2:
            raise ValueError(
                f"{mask_path} must contain a 2-D array, "
                f"got shape {mask.shape}."
            )

        # --------------------------------------------------------------
        # Brain mask
        # --------------------------------------------------------------

        brain_mask_path = (
            slice_dir
            / "brain_mask.npy"
        )

        brain_mask = None

        if brain_mask_path.exists():

            brain_mask = np.load(
                brain_mask_path
            )

            if brain_mask.ndim != 2:
                raise ValueError(
                    f"{brain_mask_path} must contain a 2-D array, "
                    f"got shape {brain_mask.shape}."
                )

        elif self.require_brain_mask:

            raise FileNotFoundError(
                f"MBNSS training requires brain_mask.npy, but it "
                f"was not found in {slice_dir}."
            )

        # --------------------------------------------------------------
        # Physical pixel spacing
        # --------------------------------------------------------------

        spacing_path = (
            slice_dir
            / "spacing.npy"
        )

        if not spacing_path.exists():
            raise FileNotFoundError(
                f"Missing physical pixel-spacing file: "
                f"{spacing_path}\n"
                f"Expected [row_spacing_mm, column_spacing_mm]."
            )

        pixel_spacing = np.asarray(
            np.load(
                spacing_path
            ),
            dtype=np.float32,
        ).reshape(-1)

        if pixel_spacing.size != 2:
            raise ValueError(
                f"{spacing_path} must contain exactly two values: "
                f"[row_spacing_mm, column_spacing_mm]. "
                f"Received {pixel_spacing.tolist()}."
            )

        if not np.all(
            np.isfinite(pixel_spacing)
        ):
            raise ValueError(
                f"{spacing_path} contains non-finite values: "
                f"{pixel_spacing.tolist()}."
            )

        if np.any(
            pixel_spacing <= 0
        ):
            raise ValueError(
                f"{spacing_path} contains non-positive values: "
                f"{pixel_spacing.tolist()}."
            )

        # --------------------------------------------------------------
        # Spatial consistency
        # --------------------------------------------------------------

        image_shape = images[0].shape

        for modality, image in zip(
            self.MODALITIES,
            images,
        ):

            if image.shape != image_shape:

                raise ValueError(
                    f"Spatial mismatch in "
                    f"{sample['case_id']}: "
                    f"{modality} has shape {image.shape}, "
                    f"expected {image_shape}."
                )

        if mask.shape != image_shape:

            raise ValueError(
                f"Mask shape {mask.shape} does not match "
                f"image shape {image_shape} in "
                f"{sample['case_id']}."
            )

        if (
            brain_mask is not None
            and brain_mask.shape != image_shape
        ):

            raise ValueError(
                f"Brain mask shape {brain_mask.shape} does not "
                f"match image shape {image_shape} in "
                f"{sample['case_id']}."
            )

        return (
            np.stack(
                images,
                axis=0,
            ),
            mask,
            brain_mask,
            pixel_spacing,
        )

    # ==================================================================
    # Intensity normalization
    # ==================================================================

    @staticmethod
    def _zscore(
        image,
    ):
        """
        Per-slice, per-modality z-score normalization.
        """

        mean = np.mean(
            image
        )

        std = np.std(
            image
        )

        if std < 1e-8:
            return (
                image - mean
            ).astype(
                np.float32
            )

        return (
            (image - mean)
            / std
        ).astype(
            np.float32
        )

    def _normalize(
        self,
        images,
    ):
        """
        Apply z-score normalization independently
        to each MRI modality.
        """

        normalized = np.empty_like(
            images,
            dtype=np.float32,
        )

        for channel in range(
            images.shape[0]
        ):

            normalized[channel] = (
                self._zscore(
                    images[channel]
                )
            )

        return normalized

    # ==================================================================
    # Affine augmentation
    # ==================================================================

    @staticmethod
    def _affine_transform_2d(
        array,
        angle_degrees,
        scale,
        order,
        mode,
        cval,
    ):
        """
        Apply centered 2-D rotation and scaling.

        The same geometric transformation is applied to all
        modalities and to the segmentation masks, with nearest
        neighbor interpolation used for discrete masks.
        """

        height, width = (
            array.shape
        )

        angle = np.deg2rad(
            angle_degrees
        )

        cos_a = np.cos(
            angle
        )

        sin_a = np.sin(
            angle
        )

        forward = np.array(
            [
                [
                    scale * cos_a,
                    -scale * sin_a,
                ],
                [
                    scale * sin_a,
                    scale * cos_a,
                ],
            ],
            dtype=np.float64,
        )

        inverse = np.linalg.inv(
            forward
        )

        center = np.array(
            [
                (height - 1) / 2.0,
                (width - 1) / 2.0,
            ],
            dtype=np.float64,
        )

        offset = (
            center
            - inverse @ center
        )

        return affine_transform(
            array,
            matrix=inverse,
            offset=offset,
            output_shape=(
                height,
                width,
            ),
            order=order,
            mode=mode,
            cval=cval,
            prefilter=(
                order > 1
            ),
        )

    def _augment(
        self,
        images,
        mask,
        brain_mask,
    ):
        """
        Apply synchronized spatial augmentation.
        """

        # --------------------------------------------------------------
        # Horizontal flip
        # --------------------------------------------------------------

        if np.random.rand() < 0.5:

            images = np.flip(
                images,
                axis=2,
            ).copy()

            mask = np.flip(
                mask,
                axis=1,
            ).copy()

            if brain_mask is not None:
                brain_mask = np.flip(
                    brain_mask,
                    axis=1,
                ).copy()

        # --------------------------------------------------------------
        # Vertical flip
        # --------------------------------------------------------------

        if np.random.rand() < 0.5:

            images = np.flip(
                images,
                axis=1,
            ).copy()

            mask = np.flip(
                mask,
                axis=0,
            ).copy()

            if brain_mask is not None:
                brain_mask = np.flip(
                    brain_mask,
                    axis=0,
                ).copy()

        # --------------------------------------------------------------
        # Rotation
        # --------------------------------------------------------------

        if (
            self.rotation_degrees > 0
            and np.random.rand() < 0.5
        ):

            angle = np.random.uniform(
                -self.rotation_degrees,
                self.rotation_degrees,
            )

            transformed_images = (
                np.empty_like(
                    images
                )
            )

            for channel in range(
                images.shape[0]
            ):

                transformed_images[
                    channel
                ] = self._affine_transform_2d(
                    images[channel],
                    angle_degrees=angle,
                    scale=1.0,
                    order=1,
                    mode="constant",
                    cval=0.0,
                )

            mask = self._affine_transform_2d(
                mask,
                angle_degrees=angle,
                scale=1.0,
                order=0,
                mode="constant",
                cval=0,
            ).astype(
                mask.dtype
            )

            if brain_mask is not None:

                brain_mask = (
                    self._affine_transform_2d(
                        brain_mask,
                        angle_degrees=angle,
                        scale=1.0,
                        order=0,
                        mode="constant",
                        cval=0,
                    )
                    .astype(
                        brain_mask.dtype
                    )
                )

            images = transformed_images

        # --------------------------------------------------------------
        # Scaling
        # --------------------------------------------------------------

        if (
            self.scale_range[0]
            != self.scale_range[1]
            and np.random.rand() < 0.5
        ):

            scale = np.random.uniform(
                self.scale_range[0],
                self.scale_range[1],
            )

            transformed_images = (
                np.empty_like(
                    images
                )
            )

            for channel in range(
                images.shape[0]
            ):

                transformed_images[
                    channel
                ] = self._affine_transform_2d(
                    images[channel],
                    angle_degrees=0.0,
                    scale=scale,
                    order=1,
                    mode="constant",
                    cval=0.0,
                )

            mask = self._affine_transform_2d(
                mask,
                angle_degrees=0.0,
                scale=scale,
                order=0,
                mode="constant",
                cval=0,
            ).astype(
                mask.dtype
            )

            if brain_mask is not None:

                brain_mask = (
                    self._affine_transform_2d(
                        brain_mask,
                        angle_degrees=0.0,
                        scale=scale,
                        order=0,
                        mode="constant",
                        cval=0,
                    )
                    .astype(
                        brain_mask.dtype
                    )
                )

            images = transformed_images

        return (
            images,
            mask,
            brain_mask,
        )

    # ==================================================================
    # Label validation
    # ==================================================================

    def _validate_mask(
        self,
        mask,
        mask_path,
    ):
        """
        Validate the four-class segmentation label space.

        No automatic relabeling is performed here.
        """

        unique_labels = np.unique(
            mask
        )

        if (
            np.any(unique_labels < 0)
            or np.any(
                unique_labels
                >= self.num_classes
            )
        ):

            raise ValueError(
                f"Invalid labels in {mask_path}: "
                f"{unique_labels.tolist()}. "
                f"Expected labels in "
                f"[0, {self.num_classes - 1}]."
            )

        # --------------------------------------------------------------
        # MOTUM class availability
        # --------------------------------------------------------------

        if (
            self.dataset_name == "MOTUM"
            and 2 in unique_labels
        ):

            raise ValueError(
                f"MOTUM mask '{mask_path}' contains label 2 "
                f"(Edema), although the configured MOTUM output "
                f"space marks edema as unavailable. "
                f"Do not create an artificial edema label."
            )

    # ==================================================================
    # Brain-mask validation
    # ==================================================================

    @staticmethod
    def _validate_brain_mask(
        brain_mask,
        path,
    ):
        """
        Validate the binary brain mask used by MBNSS.
        """

        unique_values = np.unique(
            brain_mask
        )

        if np.any(
            ~np.isin(
                unique_values,
                [0, 1],
            )
        ):

            raise ValueError(
                f"Brain mask '{path}' must be binary "
                f"(0/1). Found values: "
                f"{unique_values.tolist()}."
            )

    # ==================================================================
    # Dataset item
    # ==================================================================

    def __getitem__(
        self,
        index,
    ):
        """
        Return one 2-D multimodal MRI slice.

        Returned fields
        ---------------

        image:
            [4, H, W] float32 tensor

        mask:
            [H, W] int64 tensor

        brain_mask:
            [H, W] float32 tensor, or None

        patient_id:
            Patient identifier

        slice_id:
            Slice identifier

        case_id:
            patient_id + slice_id

        dataset:
            Dataset name

        valid_classes:
            [4] boolean class-availability vector

        pixel_spacing:
            [2] tensor containing
            [row_spacing_mm, column_spacing_mm]
        """

        sample = self.samples[
            index
        ]

        (
            images,
            mask,
            brain_mask,
            pixel_spacing,
        ) = self._load_sample(
            sample
        )

        # --------------------------------------------------------------
        # Normalize MRI modalities
        # --------------------------------------------------------------

        images = self._normalize(
            images
        )

        # --------------------------------------------------------------
        # Training augmentation
        # --------------------------------------------------------------

        if self.augment:

            (
                images,
                mask,
                brain_mask,
            ) = self._augment(
                images,
                mask,
                brain_mask,
            )

        # --------------------------------------------------------------
        # Input-size validation
        # --------------------------------------------------------------

        if (
            images.shape[1:]
            != self.image_size
        ):

            raise ValueError(
                f"Expected image size "
                f"{self.image_size}, but "
                f"'{sample['case_id']}' has "
                f"image size "
                f"{images.shape[1:]}. "
                f"Resize/resample the data during "
                f"preprocessing."
            )

        # --------------------------------------------------------------
        # Validate segmentation mask
        # --------------------------------------------------------------

        mask = mask.astype(
            np.int64
        )

        self._validate_mask(
            mask,
            sample["path"] / "mask.npy",
        )

        # --------------------------------------------------------------
        # Validate brain mask
        # --------------------------------------------------------------

        if brain_mask is not None:

            brain_mask = (
                brain_mask.astype(
                    np.float32
                )
            )

            self._validate_brain_mask(
                brain_mask,
                sample["path"]
                / "brain_mask.npy",
            )

        # --------------------------------------------------------------
        # Dataset-specific valid classes
        # --------------------------------------------------------------

        valid_classes = torch.tensor(
            self.DATASET_VALID_CLASSES[
                self.dataset_name
            ],
            dtype=torch.bool,
        )

        # --------------------------------------------------------------
        # Convert to tensors
        # --------------------------------------------------------------

        image_tensor = torch.from_numpy(
            images.astype(
                np.float32
            )
        )

        mask_tensor = torch.from_numpy(
            mask
        )

        if brain_mask is not None:

            brain_mask_tensor = (
                torch.from_numpy(
                    brain_mask
                )
            )

        else:

            brain_mask_tensor = None

        spacing_tensor = (
            torch.from_numpy(
                pixel_spacing.astype(
                    np.float32
                )
            )
        )

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "brain_mask": brain_mask_tensor,
            "patient_id": sample[
                "patient_id"
            ],
            "slice_id": sample[
                "slice_id"
            ],
            "case_id": sample[
                "case_id"
            ],
            "dataset": self.dataset_name,
            "valid_classes": valid_classes,
            "pixel_spacing": spacing_tensor,
        }


# ======================================================================
# Utility
# ======================================================================

def get_dataset_valid_classes(
    dataset_name,
):
    """
    Return the configured class-availability vector for a dataset.
    """

    if dataset_name not in (
        BrainTumorDataset
        .DATASET_VALID_CLASSES
    ):
        raise ValueError(
            f"Unsupported dataset '{dataset_name}'."
        )

    return torch.tensor(
        BrainTumorDataset
        .DATASET_VALID_CLASSES[
            dataset_name
        ],
        dtype=torch.bool,
    )
