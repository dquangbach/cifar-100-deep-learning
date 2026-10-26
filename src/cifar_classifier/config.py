"""The experiment settings used by the original WRN notebook."""

from dataclasses import dataclass

IMAGE_SIZE = 32
NUM_CLASSES = 100
LEGACY_CHECKPOINT = "cifar100_wrn16_4_aug200_best.keras"
LEGACY_HISTORIES = (
    "cifar100_wrn16_4_aug20_history.csv",
    "cifar100_wrn16_4_aug200_history.csv",
)


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 200
    batch_size: int = 64
    seed: int = 42
    validation_size: int = 5_000
    schedule: str = "plateau"
    mix_probability: float = 0.30
    cutmix_probability: float = 0.30
    cutout_probability: float = 0.25
    erasing_probability: float = 0.25
    mix_alpha: float = 0.5
    cutmix_alpha: float = 1.0

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        if not 0 < self.validation_size < 50_000:
            raise ValueError("validation_size must be between 1 and 49,999")
        if self.schedule not in {"plateau", "cosine"}:
            raise ValueError("schedule must be 'plateau' or 'cosine'")
        if self.mix_probability < 0 or self.cutmix_probability < 0 or (
            self.mix_probability + self.cutmix_probability > 1
        ):
            raise ValueError("mix and cutmix probabilities must sum to at most 1")
        if self.cutout_probability < 0 or self.erasing_probability < 0 or (
            self.cutout_probability + self.erasing_probability > 1
        ):
            raise ValueError("cutout and erasing probabilities must sum to at most 1")
        if self.mix_alpha <= 0 or self.cutmix_alpha <= 0:
            raise ValueError("mix_alpha and cutmix_alpha must be positive")
