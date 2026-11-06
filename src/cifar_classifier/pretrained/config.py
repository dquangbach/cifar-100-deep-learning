"""Settings carried over from build_efficientnet.ipynb."""

from dataclasses import asdict, dataclass

MODEL_NAME = "tf_efficientnetv2_b0.in1k"
NUM_CLASSES = 100
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class EfficientNetConfig:
    seed: int = 42
    epochs: int = 30
    batch_size: int = 32
    validation_size: int = 5_000
    image_size: int = 224
    patience: int = 4
    num_workers: int = 0
    learning_rate: float = 2e-5
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    drop_rate: float = 0.3
    mixup_alpha: float = 0.5
    cutmix_alpha: float = 1.0
    mixup_probability: float = 0.3
    cutmix_probability: float = 0.3

    def __post_init__(self) -> None:
        if min(self.epochs, self.batch_size, self.image_size, self.patience) < 1:
            raise ValueError("epochs, batch_size, image_size and patience must be positive")
        if not 0 < self.validation_size < 50_000:
            raise ValueError("validation_size must be between 1 and 49,999")
        if self.num_workers < 0 or self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("invalid worker count, learning rate or weight decay")
        if not 0 <= self.label_smoothing < 1 or not 0 <= self.drop_rate < 1:
            raise ValueError("invalid label smoothing or drop rate")
        if self.mixup_alpha <= 0 or self.cutmix_alpha <= 0:
            raise ValueError("MixUp/CutMix alpha must be positive")
        if min(self.mixup_probability, self.cutmix_probability) < 0 or (
            self.mixup_probability + self.cutmix_probability > 1
        ):
            raise ValueError("MixUp/CutMix probabilities must sum to at most 1")

    def to_dict(self) -> dict:
        return asdict(self)
