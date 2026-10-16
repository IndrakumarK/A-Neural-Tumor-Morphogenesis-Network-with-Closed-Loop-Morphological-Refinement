from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from scipy.ndimage import affine_transform
except ImportError as exc:
    raise ImportError(
        "scipy is required for small-angle rotation and random scaling. "
        "Install it with: pip install scipy"
    ) from exc


class BrainTumorDataset(Dataset):
    """
    Dataset loader for 2-D multimodal brain-tumor MRI segmentation.

    Expected directory structure:

        root/
        ├── case_001/
        │   ├── T1.npy
        │   ├── T1ce.npy
        │   ├── T2.npy
        │   ├── FLAIR.npy
        │   └── mask.npy
        │
        ├── case_002/
        │   ├── T1.npy
        │   ├── T1ce.npy
        │   ├── T2.npy
        │   ├── FLAIR.npy
        │   └── mask.npy
        │
        └── ...

    Each MRI modality and mask must be a 2-D NumPy array.

    Important:
        This loader does NOT invent or assume physical pixel spacing.
        Therefore, HD95 calculated by evaluate.py is expressed in
        image-array pixels unless spacing metadata is explicitly added
        to the dataset and evaluation pipeline.
    """

    MODALITIES = (
        "T1",
        "T1ce",
        "T2",
        "FLAIR",
    )

    # Class order:
    # 0 = Background
    # 1 = NCR/NET
    # 2 = Edema
    # 3 = Enhancing Tumor

    DATASET_VALID_CLASSES = {
        "UPenn-GBM": (
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

    def __init__(
        self,
        root,
        image_size=(256, 256),
        num_classes=4,
        augment=False,
        dataset_name="UPenn-GBM",
        rotation_degrees=10.0,
        scale_range=(0.90, 1.10),
    ):
        self.root = Path(root)

        self.image_size = tuple(image_size)

        self.num_classes = int(num_classes)

        self.augment = bool(augment)

        self.dataset_name = dataset_name

        self.rotation_degrees = float(
            rotation_degrees
        )

        self.scale_range = tuple(
            scale_range
        )

        # ---------------------------------------------------------
        # Basic validation
        # ---------------------------------------------------------

        if self.num_classes != 4:
            raise ValueError(
                "KE-NTMN expects four segmentation classes: "
                "Background, NCR/NET, Edema, and Enhancing Tumor."
            )

        if self.dataset_name not in self.DATASET_VALID_CLASSES:
            raise ValueError(
                f"Unsupported dataset '{self.dataset_name}'. "
                f"Expected one of: "
                f"{list(self.DATASET_VALID_CLASSES.keys())}"
            )

        if len(self.image_size) != 2:
            raise ValueError(
                "image_size must contain exactly two values, "
                f"got {self.image_size}"
            )

        if (
            self.rotation_degrees < 0
        ):
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
                f"Invalid scale_range: "
                f"{self.scale_range}"
            )

        # ---------------------------------------------------------
        # Root directory
        # ---------------------------------------------------------

        if not self.root.exists():
            raise FileNotFoundError(
                f"Dataset directory does not exist: "
                f"{self.root}"
            )

        if not self.root.is_dir():
            raise NotADirectoryError(
                f"Dataset root is not a directory: "
                f"{self.root}"
            )

        # ---------------------------------------------------------
        # Case directories
        # ---------------------------------------------------------

        self.cases = sorted(
            case_dir
            for case_dir in self.root.iterdir()
            if case_dir.is_dir()
        )

        if not self.cases:
            raise RuntimeError(
                f"No case directories found in: "
                f"{self.root}"
            )

    # =============================================================
    # Dataset length
    # =============================================================

    def __len__(self):
        return len(self.cases)

    # =============================================================
    # Load one case
    # =============================================================

    def _load_case(self, case_dir):
        """
        Load four MRI modalities and segmentation mask.

        Returns
        -------
        images : np.ndarray
            Shape: [4, H, W]

        mask : np.ndarray
            Shape: [H, W]
        """

        images = []

        # ---------------------------------------------------------
        # Load modalities
        # ---------------------------------------------------------

        for modality in self.MODALITIES:

            path = (
                case_dir
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
                    f"{path} must be a 2-D array, "
                    f"got shape {image.shape}"
                )

            images.append(image)

        # ---------------------------------------------------------
        # Load segmentation mask
        # ---------------------------------------------------------

        mask_path = (
            case_dir
            / "mask.npy"
        )

        if not mask_path.exists():
            raise FileNotFoundError(
                f"Missing segmentation mask: "
                f"{mask_path}"
            )

        mask = np.load(
            mask_path
        ).astype(
            np.int64
        )

        if mask.ndim != 2:
            raise ValueError(
                f"{mask_path} must be a 2-D array, "
                f"got shape {mask.shape}"
            )

        # ---------------------------------------------------------
        # Stack modalities
        # ---------------------------------------------------------

        images = np.stack(
            images,
            axis=0
        )

        # ---------------------------------------------------------
        # Verify image/mask dimensions
        # ---------------------------------------------------------

        if images.shape[1:] != mask.shape:
            raise ValueError(
                f"Image/mask shape mismatch in "
                f"{case_dir}: "
                f"images={images.shape}, "
                f"mask={mask.shape}"
            )

        # ---------------------------------------------------------
        # Verify all modalities have identical shape
        # ---------------------------------------------------------

        reference_shape = images[0].shape

        for idx in range(
            1,
            images.shape[0]
        ):
            if images[idx].shape != reference_shape:
                raise ValueError(
                    f"Modality shape mismatch in "
                    f"{case_dir}: "
                    f"modality index {idx} has "
                    f"shape {images[idx].shape}, "
                    f"expected {reference_shape}"
                )

        return images, mask

    # =============================================================
    # Per-modality z-score normalization
    # =============================================================

    @staticmethod
    def _zscore_normalize(
        images
    ):
        """
        Apply independent z-score normalization
        to each MRI modality.
        """

        images = images.astype(
            np.float32,
            copy=True
        )

        for c in range(
            images.shape[0]
        ):

            channel = images[c]

            mean = float(
                channel.mean()
            )

            std = float(
                channel.std()
            )

            if std < 1e-8:
                images[c] = (
                    channel - mean
                )

            else:
                images[c] = (
                    channel - mean
                ) / std

        return images

    # =============================================================
    # Random horizontal flip
    # =============================================================

    @staticmethod
    def _horizontal_flip(
        images,
        mask
    ):
        images = np.flip(
            images,
            axis=2
        ).copy()

        mask = np.flip(
            mask,
            axis=1
        ).copy()

        return images, mask

    # =============================================================
    # Random vertical flip
    # =============================================================

    @staticmethod
    def _vertical_flip(
        images,
        mask
    ):
        images = np.flip(
            images,
            axis=1
        ).copy()

        mask = np.flip(
            mask,
            axis=0
        ).copy()

        return images, mask

    # =============================================================
    # Small-angle rotation
    # =============================================================

    def _rotate(
        self,
        images,
        mask,
        angle
    ):
        """
        Rotate images and mask using scipy affine_transform.

        Images use linear interpolation.
        Mask uses nearest-neighbor interpolation.
        """

        h, w = mask.shape

        center = np.array(
            [h / 2.0, w / 2.0],
            dtype=np.float32
        )

        theta = np.deg2rad(
            angle
        )

        cos_theta = np.cos(
            theta
        )

        sin_theta = np.sin(
            theta
        )

        rotation_matrix = np.array(
            [
                [
                    cos_theta,
                    -sin_theta,
                ],
                [
                    sin_theta,
                    cos_theta,
                ],
            ],
            dtype=np.float32
        )

        offset = (
            center
            - rotation_matrix @ center
        )

        rotated_images = []

        for c in range(
            images.shape[0]
        ):

            rotated = affine_transform(
                images[c],
                matrix=rotation_matrix,
                offset=offset,
                order=1,
                mode="constant",
                cval=0.0
            )

            rotated_images.append(
                rotated.astype(
                    np.float32
                )
            )

        rotated_images = np.stack(
            rotated_images,
            axis=0
        )

        rotated_mask = affine_transform(
            mask,
            matrix=rotation_matrix,
            offset=offset,
            order=0,
            mode="constant",
            cval=0
        ).astype(
            np.int64
        )

        return (
            rotated_images,
            rotated_mask
        )

    # =============================================================
    # Random scaling
    # =============================================================

    def _scale(
        self,
        images,
        mask,
        scale
    ):
        """
        Apply isotropic scaling while preserving
        the final image dimensions.

        Images use linear interpolation.
        Mask uses nearest-neighbor interpolation.
        """

        h, w = mask.shape

        center = np.array(
            [h / 2.0, w / 2.0],
            dtype=np.float32
        )

        scale_matrix = np.array(
            [
                [
                    1.0 / scale,
                    0.0
                ],
                [
                    0.0,
                    1.0 / scale
                ],
            ],
            dtype=np.float32
        )

        offset = (
            center
            - scale_matrix @ center
        )

        scaled_images = []

        for c in range(
            images.shape[0]
        ):

            scaled = affine_transform(
                images[c],
                matrix=scale_matrix,
                offset=offset,
                output_shape=(h, w),
                order=1,
                mode="constant",
                cval=0.0
            )

            scaled_images.append(
                scaled.astype(
                    np.float32
                )
            )

        scaled_images = np.stack(
            scaled_images,
            axis=0
        )

        scaled_mask = affine_transform(
            mask,
            matrix=scale_matrix,
            offset=offset,
            output_shape=(h, w),
            order=0,
            mode="constant",
            cval=0
        ).astype(
            np.int64
        )

        return (
            scaled_images,
            scaled_mask
        )

    # =============================================================
    # Data augmentation
    # =============================================================

    def _apply_augmentation(
        self,
        images,
        mask
    ):

        # ---------------------------------------------------------
        # Horizontal flip
        # ---------------------------------------------------------

        if np.random.rand() < 0.5:

            images, mask = (
                self._horizontal_flip(
                    images,
                    mask
                )
            )

        # ---------------------------------------------------------
        # Vertical flip
        # ---------------------------------------------------------

        if np.random.rand() < 0.5:

            images, mask = (
                self._vertical_flip(
                    images,
                    mask
                )
            )

        # ---------------------------------------------------------
        # Small-angle rotation
        # ---------------------------------------------------------

        if np.random.rand() < 0.5:

            angle = np.random.uniform(
                -self.rotation_degrees,
                self.rotation_degrees
            )

            images, mask = (
                self._rotate(
                    images,
                    mask,
                    angle
                )
            )

        # ---------------------------------------------------------
        # Random scaling
        # ---------------------------------------------------------

        if np.random.rand() < 0.5:

            scale = np.random.uniform(
                self.scale_range[0],
                self.scale_range[1]
            )

            images, mask = (
                self._scale(
                    images,
                    mask,
                    scale
                )
            )

        return images, mask

    # =============================================================
    # Get item
    # =============================================================

    def __getitem__(
        self,
        index
    ):

        case_dir = self.cases[index]

        # ---------------------------------------------------------
        # Load
        # ---------------------------------------------------------

        images, mask = (
            self._load_case(
                case_dir
            )
        )

        # ---------------------------------------------------------
        # Verify expected image size
        # ---------------------------------------------------------

        if (
            images.shape[1:]
            != self.image_size
        ):

            raise ValueError(
                f"Case '{case_dir.name}' has "
                f"image shape {images.shape[1:]}, "
                f"but the configured image size is "
                f"{self.image_size}. "
                f"Resize/resample the data during "
                f"preprocessing."
            )

        if (
            mask.shape
            != self.image_size
        ):

            raise ValueError(
                f"Case '{case_dir.name}' has "
                f"mask shape {mask.shape}, "
                f"but the configured image size is "
                f"{self.image_size}. "
                f"Resize/resample the data during "
                f"preprocessing."
            )

        # ---------------------------------------------------------
        # Data augmentation
        # ---------------------------------------------------------

        if self.augment:

            images, mask = (
                self._apply_augmentation(
                    images,
                    mask
                )
            )

        # ---------------------------------------------------------
        # Per-modality z-score normalization
        # ---------------------------------------------------------

        images = (
            self._zscore_normalize(
                images
            )
        )

        # ---------------------------------------------------------
        # Verify mask labels
        # ---------------------------------------------------------

        unique_labels = np.unique(
            mask
        )

        invalid_labels = (
            unique_labels[
                (unique_labels < 0)
                | (
                    unique_labels
                    >= self.num_classes
                )
            ]
        )

        if invalid_labels.size > 0:

            raise ValueError(
                f"Case '{case_dir.name}' "
                f"contains invalid mask labels: "
                f"{invalid_labels.tolist()}"
            )

        # ---------------------------------------------------------
        # Convert to tensors
        # ---------------------------------------------------------

        image_tensor = torch.from_numpy(
            images.astype(
                np.float32
            )
        )

        mask_tensor = torch.from_numpy(
            mask.astype(
                np.int64
            )
        )

        # ---------------------------------------------------------
        # Dataset-specific valid classes
        # ---------------------------------------------------------

        valid_classes = torch.tensor(
            self.DATASET_VALID_CLASSES[
                self.dataset_name
            ],
            dtype=torch.bool
        )

        # ---------------------------------------------------------
        # Return interface expected by evaluate.py
        # ---------------------------------------------------------

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "case_id": case_dir.name,
            "dataset": self.dataset_name,
            "valid_classes": valid_classes,
        }
