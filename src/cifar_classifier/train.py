"""Train WRN-16-4 using the same defaults as the 200-epoch notebook run."""

import json
import math
from dataclasses import asdict
from pathlib import Path

import tensorflow as tf

from .checkpoints import load_checkpoint
from .config import TrainConfig
from .data import load_train_split, make_train_datasets
from .model import build_wrn_16_4


def train(data_dir: Path, run_dir: Path, config: TrainConfig) -> dict:
    data_dir = Path(data_dir)
    run_dir = Path(run_dir)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Run directory is not empty: {run_dir}")
    tf.keras.utils.set_random_seed(config.seed)
    split = load_train_split(data_dir, config)
    train_ds, validation_ds = make_train_datasets(split, config)
    model = build_wrn_16_4()
    sample_images, sample_labels = next(iter(train_ds))
    tf.debugging.assert_shapes([
        (sample_images, (None, 32, 32, 3)),
        (sample_labels, (None, 100)),
    ])
    tf.debugging.assert_near(
        tf.reduce_sum(sample_labels, axis=1), tf.ones(tf.shape(sample_labels)[0])
    )
    tf.debugging.assert_all_finite(sample_images, "Non-finite augmented pixels")

    steps = math.ceil(len(split.train_labels) / config.batch_size)
    learning_rate = (
        1e-3 if config.schedule == "plateau"
        else tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=1e-3,
            decay_steps=steps * config.epochs,
            alpha=0.01,
        )
    )
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=1e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy")],
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "best.keras"
    history_path = run_dir / "history.csv"
    monitor = "val_loss" if config.schedule == "plateau" else "val_accuracy"
    metadata = {
        "model": "wrn_16_4_cifar100",
        "training": asdict(config),
        "preprocessing": {
            "mean": split.mean.tolist(),
            "std": split.std.tolist(),
            "scale": 255.0,
        },
        "checkpoint_monitor": monitor,
        "tensorflow_version": tf.__version__,
    }
    (run_dir / "run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(checkpoint_path), monitor=monitor,
            mode="min" if monitor == "val_loss" else "max", save_best_only=True,
        ),
    ]
    if config.schedule == "plateau":
        callbacks.extend([
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", mode="min", factor=0.5, patience=3,
                min_delta=0.0, min_lr=1e-5,
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", mode="min", patience=20, restore_best_weights=True,
            ),
        ])
    callbacks.append(tf.keras.callbacks.CSVLogger(str(history_path)))
    print(f"Training {model.count_params():,} parameters; output: {run_dir}")
    model.fit(
        train_ds, validation_data=validation_ds, epochs=config.epochs,
        callbacks=callbacks, verbose=2,
    )
    best_model = load_checkpoint(checkpoint_path)
    best_model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy")],
    )
    validation_metrics = best_model.evaluate(validation_ds, return_dict=True, verbose=0)
    metadata["best_validation"] = {key: float(value) for key, value in validation_metrics.items()}
    (run_dir / "run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata
