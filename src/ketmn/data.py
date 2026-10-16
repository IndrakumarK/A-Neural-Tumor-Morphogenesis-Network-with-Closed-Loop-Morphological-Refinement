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
    Patient/case-level 2-D multimodal MRI dataset.

    Expected directory structure
    ----------------------------
    root/
        case_001/
            T1.npy
            T1ce.npy
            T2.npy
            FLAIR.npy
            mask.npy
            spacing.npy
        case_002/
            T1.npy
            T1ce.npy
            T2.npy
            FLAIR.npy
            mask.npy
            spacing.npy
        ...

    spacing.npy must contain:
        [row_spacing_mm, column_spacing_mm]

    The spacing must correspond to the final image/mask arrays stored
    in the .npy files.

    Class order
    -----------
    0 = Background
    1 = NCR/NET
    2 = Edema
    3 = Enhancing Tumor
    """

    MODALITIES = (
        "T1",
        "T1ce",
        "T2",
        "FLAIR",
    )

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

        self.image_size = tuple(
            int(x) for x in image_size
        )

        self.num_classes = int(num_classes)
        self.augment = bool(augment)
        self.dataset_name = str(dataset_name)

        self.rotation_degrees = float(
            rotation_degrees
        )

        self.scale_range = tuple(
            float(x) for x in scale_range
        )

        # -------------------------------------------------------------
        # Basic configuration checks
        # -------------------------------------------------------------
        if self.num_classes != 4:
            raise ValueError(
                "KE-NTMN expects four segmentation classes: "
                "Background, NCR/NET, ED, and ET."
            )

        if self.dataset_name not in self.DATASET_VALID_CLASSES:
            raise ValueError(
                f"Unsupported dataset '{self.dataset_name}'. "
                f"Expected one of: "
                f"{list(self.DATASET_VALID_CLASSES.keys())}"
            )

        if len(self.image_size) != 2:
            raise ValueError(
                f"image_size must contain two values, "
                f"got {self.image_size}"
            )

        if self.rotation_degrees < 0:
            raise ValueError(
                "rotation_degrees must be non-negative."
            )

        if (
            len(self.scale_range) != 2
            or self.scale_range[0] <= 0
            or self.scale_range[1] <= 0
            or self.scale_range[0] > self.scale_range[1]
        ):
            raise ValueError(
                f"Invalid scale_range: {self.scale_range}"
            )

        # -------------------------------------------------------------
        # Dataset directory
        # -------------------------------------------------------------
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

    def __len__(self):
        return len(self.cases)

    # -----------------------------------------------------------------
    # Case loading
    # -----------------------------------------------------------------
    def _load_case(self, case_dir):
        """
        Load four MRI modalities, segmentation mask,
        and physical in-plane pixel spacing.
        """

        images = []

        # -------------------------------------------------------------
        # MRI modalities
        # -------------------------------------------------------------
        for modality in self.MODALITIES:

            path = case_dir / f"{modality}.npy"

            if not path.exists():
                raise FileNotFoundError(
                    f"Missing modality file: {path}"
                )

            image = np.load(path).astype(
                np.float32
            )

            if image.ndim != 2:
                raise ValueError(
                    f"{path} must contain a 2-D array, "
                    f"got shape {image.shape}"
                )

            images.append(image)

        # -------------------------------------------------------------
        # Segmentation mask
        # -------------------------------------------------------------
        mask_path = case_dir / "mask.npy"

        if not mask_path.exists():
            raise FileNotFoundError(
                f"Missing segmentation mask: "
                f"{mask_path}"
            )

        mask = np.load(mask_path)

        if mask.ndim != 2:
            raise ValueError(
                f"{mask_path} must contain a 2-D array, "
                f"got shape {mask.shape}"
            )

        # -------------------------------------------------------------
        # Physical pixel spacing
        # -------------------------------------------------------------
        spacing_path = case_dir / "spacing.npy"

        if not spacing_path.exists():
            raise FileNotFoundError(
                f"Missing physical pixel-spacing file: "
                f"{spacing_path}\n"
                f"Expected spacing.npy containing "
                f"[row_spacing_mm, column_spacing_mm]."
            )

        pixel_spacing = np.asarray(
            np.load(spacing_path),
            dtype=np.float64,
        ).reshape(-1)

        if pixel_spacing.size != 2:
            raise ValueError(
                f"{spacing_path} must contain exactly "
                f"two values: "
                f"[row_spacing_mm, column_spacing_mm]. "
                f"Received shape {pixel_spacing.shape}."
            )

        if not np.all(
            np.isfinite(pixel_spacing)
        ):
            raise ValueError(
                f"{spacing_path} contains non-finite "
                f"pixel-spacing values: "
                f"{pixel_spacing.tolist()}"
            )

        if np.any(pixel_spacing <= 0):
            raise ValueError(
                f"{spacing_path} contains invalid "
                f"non-positive pixel spacing: "
                f"{pixel_spacing.tolist()}"
            )

        # -------------------------------------------------------------
        # Spatial consistency
        # -------------------------------------------------------------
        image_shape = images[0].shape

        for modality, image in zip(
            self.MODALITIES,
            images,
        ):
            if image.shape != image_shape:
                raise ValueError(
                    f"Spatial size mismatch in case "
                    f"'{case_dir.name}': "
                    f"{modality} has shape {image.shape}, "
                    f"expected {image_shape}."
                )

        if mask.shape != image_shape:
            raise ValueError(
                f"Mask shape {mask.shape} does not "
                f"match image shape {image_shape} "
                f"in case '{case_dir.name}'."
            )

        return (
            np.stack(images, axis=0),
            mask,
            pixel_spacing.astype(np.float32),
        )

    # -----------------------------------------------------------------
    # Normalization
    # -----------------------------------------------------------------
    @staticmethod
    def _zscore(image):
        mean = image.mean()
        std = image.std()

        if std < 1e-8:
            return image - mean

        return (image - mean) / std

    def _normalize(self, images):

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

    # -----------------------------------------------------------------
    # 2-D affine augmentation
    # -----------------------------------------------------------------
    @staticmethod
    def _affine_transform_2d(
        array,
        angle_degrees,
        scale,
        order,
        mode,
        cval,
    ):
        height, width = array.shape

        angle = np.deg2rad(
            angle_degrees
        )

        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        # Forward transformation:
        # rotation followed by scaling.
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

        # scipy.ndimage.affine_transform
        # expects inverse mapping.
        matrix = np.linalg.inv(
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
            center - matrix @ center
        )

        transformed = affine_transform(
            array,
            matrix=matrix,
            offset=offset,
            output_shape=(
                height,
                width,
            ),
            order=order,
            mode=mode,
            cval=cval,
            prefilter=(order > 1),
        )

        return transformed

    def _augment(
        self,
        images,
        mask,
    ):

        # -------------------------------------------------------------
        # Random horizontal flip
        # -------------------------------------------------------------
        if np.random.rand() < 0.5:

            images = np.flip(
                images,
                axis=2,
            ).copy()

            mask = np.flip(
                mask,
                axis=1,
            ).copy()

        # -------------------------------------------------------------
        # Random vertical flip
        # -------------------------------------------------------------
        if np.random.rand() < 0.5:

            images = np.flip(
                images,
                axis=1,
            ).copy()

            mask = np.flip(
                mask,
                axis=0,
            ).copy()

        # -------------------------------------------------------------
        # Small-angle rotation
        # -------------------------------------------------------------
        if np.random.rand() < 0.5:

            angle = np.random.uniform(
                -self.rotation_degrees,
                self.rotation_degrees,
            )

            transformed_images = (
                np.empty_like(images)
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
            ).astype(mask.dtype)

            images = transformed_images

        # -------------------------------------------------------------
        # Random scaling
        # -------------------------------------------------------------
        if np.random.rand() < 0.5:

            scale = np.random.uniform(
                self.scale_range[0],
                self.scale_range[1],
            )

            transformed_images = (
                np.empty_like(images)
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
            ).astype(mask.dtype)

            images = transformed_images

        return images, mask

    # -----------------------------------------------------------------
    # Mask validation
    # -----------------------------------------------------------------
    def _validate_mask(
        self,
        mask,
        mask_path,
    ):

        unique_labels = np.unique(mask)

        if np.any(unique_labels < 0) or np.any(
            unique_labels >= self.num_classes
        ):
            raise ValueError(
                f"Invalid labels in {mask_path}: "
                f"{unique_labels.tolist()}. "
                f"Expected labels in "
                f"[0, {self.num_classes - 1}]."
            )

        # MOTUM does not contain an independently
        # annotated edema class.
        if (
            self.dataset_name == "MOTUM"
            and 2 in unique_labels
        ):
            raise ValueError(
                f"MOTUM mask '{mask_path}' "
                f"contains label 2 (ED). "
                "MOTUM does not provide an "
                "independently annotated edema class. "
                "Do not create an artificial edema "
                "label during preprocessing."
            )

    # -----------------------------------------------------------------
    # Dataset item
    # -----------------------------------------------------------------
    def __getitem__(self, index):

        case_dir = self.cases[index]

        (
            images,
            mask,
            pixel_spacing,
        ) = self._load_case(
            case_dir
        )

        # -------------------------------------------------------------
        # Per-modality z-score normalization
        # -------------------------------------------------------------
        images = self._normalize(
            images
        )

        # -------------------------------------------------------------
        # Training augmentation
        # -------------------------------------------------------------
        if self.augment:

            images, mask = self._augment(
                images,
                mask,
            )

        # -------------------------------------------------------------
        # Input-size validation
        # -------------------------------------------------------------
        if images.shape[1:] != self.image_size:

            raise ValueError(
                f"Expected image size "
                f"{self.image_size}, "
                f"but case '{case_dir.name}' "
                f"has image size "
                f"{images.shape[1:]}. "
                "Resize/resample the data "
                "during preprocessing."
            )

        # -------------------------------------------------------------
        # Mask validation
        # -------------------------------------------------------------
        mask = mask.astype(
            np.int64
        )

        mask_path = (
            case_dir / "mask.npy"
        )

        self._validate_mask(
            mask,
            mask_path,
        )

        # -------------------------------------------------------------
        # Dataset-specific valid classes
        # -------------------------------------------------------------
        valid_classes = torch.tensor(
            self.DATASET_VALID_CLASSES[
                self.dataset_name
            ],
            dtype=torch.bool,
        )

        # -------------------------------------------------------------
        # Convert to tensors
        # -------------------------------------------------------------
        images = torch.from_numpy(
            images.astype(
                np.float32
            )
        )

        mask = torch.from_numpy(
            mask
        )

        pixel_spacing = torch.from_numpy(
            pixel_spacing.astype(
                np.float32
            )
        )

        return {
            "image": images,
            "mask": mask,
            "case_id": case_dir.name,
            "dataset": self.dataset_name,
            "valid_classes": valid_classes,
            "pixel_spacing": pixel_spacing,
        }
