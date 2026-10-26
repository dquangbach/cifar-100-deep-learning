"""Regression checks for the CIFAR-100 WRN migration."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from cifar_classifier.config import TrainConfig
from cifar_classifier.data import make_train_datasets, split_train
from cifar_classifier.model import build_wrn_16_4
from cifar_classifier.plots import plot_history


class PipelineTests(unittest.TestCase):
    def test_split_stats_exclude_validation_and_pipeline_shapes(self):
        values = np.arange(10, dtype=np.uint8) * 20
        images = np.broadcast_to(values[:, None, None, None], (10, 32, 32, 3)).copy()
        labels = np.arange(10, dtype=np.int32)
        split = split_train(images, labels, seed=42, validation_size=2)
        expected = np.random.default_rng(42).permutation(10)
        np.testing.assert_array_equal(split.validation_labels, expected[:2])
        np.testing.assert_allclose(
            split.mean, images[expected[2:]].mean(axis=(0, 1, 2)) / 255.0
        )
        train_ds, validation_ds = make_train_datasets(
            split, TrainConfig(epochs=1, batch_size=4, validation_size=2)
        )
        train_images, train_labels = next(iter(train_ds))
        val_images, val_labels = next(iter(validation_ds))
        self.assertEqual(train_images.shape, (4, 32, 32, 3))
        self.assertEqual(val_images.shape, (2, 32, 32, 3))
        self.assertEqual(train_labels.shape, (4, 100))
        self.assertEqual(val_labels.shape, (2, 100))
        np.testing.assert_allclose(train_labels.numpy().sum(axis=1), 1.0)

    def test_model_and_legacy_checkpoint_are_compatible(self):
        model = build_wrn_16_4()
        self.assertEqual(model.input_shape, (None, 32, 32, 3))
        self.assertEqual(model.output_shape, (None, 100))
        self.assertEqual(model.count_params(), 2_775_636)

        checkpoint = Path(__file__).resolve().parents[1] / "checkpoints" / (
            "cifar100_wrn16_4_aug200_best.keras"
        )
        if checkpoint.is_file():
            from cifar_classifier.checkpoints import load_checkpoint

            loaded = load_checkpoint(checkpoint)
            self.assertEqual(loaded.count_params(), model.count_params())

    def test_history_plot_uses_checkpoint_epoch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir)
            history = folder / "history.csv"
            history.write_text(
                "epoch,accuracy,loss,val_accuracy,val_loss\n"
                "0,0.1,3.0,0.2,2.8\n"
                "1,0.2,2.0,0.4,1.9\n"
                "2,0.3,1.5,0.3,2.1\n",
                encoding="utf-8",
            )
            output = plot_history(history, folder / "plots", monitor="val_loss")
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
