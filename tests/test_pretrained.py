"""Regression checks for the PyTorch transfer-learning path."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from cifar_classifier.pretrained.data import split_indices
from cifar_classifier.pretrained.evaluate import _top5_accuracy
from cifar_classifier.pretrained.engine import evaluate_loader
from cifar_classifier.pretrained.engine import (
    apply_cutmix, apply_mixup, build_model, load_checkpoint,
)


class PretrainedPipelineTests(unittest.TestCase):
    def test_split_and_batch_mixing(self):
        train, val = split_indices(50_000, seed=42, validation_size=5_000)
        self.assertEqual((len(train), len(val)), (45_000, 5_000))
        self.assertEqual(len(np.intersect1d(train, val)), 0)
        images = torch.zeros((2, 3, 32, 32))
        images[1] = 1
        labels = torch.tensor([1, 2])
        torch.manual_seed(7)
        mixed, a, b, lam = apply_mixup(images, labels, alpha=0.5)
        self.assertEqual(mixed.shape, images.shape)
        self.assertTrue(0 <= lam <= 1)
        expected = lam * images + (1 - lam) * images[(b - 1).long()]
        torch.testing.assert_close(mixed, expected)
        self.assertEqual(a.shape, b.shape)
        cut, _, _, kept = apply_cutmix(images, labels, alpha=1.0)
        self.assertEqual(cut.shape, images.shape)
        self.assertTrue(0 <= kept <= 1)

    def test_top5_on_small_class_subset(self):
        probabilities = np.zeros((2, 100), dtype=float)
        probabilities[0, [1, 3, 5, 7, 99]] = [0.1, 0.2, 0.3, 0.4, 0.5]
        probabilities[1, [1, 3, 5, 7, 99]] = [0.1, 0.2, 0.3, 0.4, 0.5]
        self.assertEqual(_top5_accuracy(np.array([99, 42]), probabilities), 0.5)

    def test_evaluation_preserves_labels_across_batches(self):
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset

        class ConstantModel(nn.Module):
            def forward(self, images):
                return torch.zeros((len(images), 100))

        expected = np.array([49, 33, 72, 51])
        dataset = TensorDataset(torch.zeros((4, 3, 32, 32)), torch.tensor(expected))
        loader = DataLoader(dataset, batch_size=2)
        result = evaluate_loader(
            ConstantModel(), loader, nn.CrossEntropyLoss(), torch.device("cpu"),
            collect_predictions=True,
        )
        np.testing.assert_array_equal(result["y_true"], expected)
        self.assertEqual(result["y_prob"].shape, (4, 100))

    def test_checkpoint_loads_without_downloading_weights(self):
        model = build_model(pretrained=False, drop_rate=0.3)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "best.pth"
            torch.save({"state_dict": model.state_dict(), "config": {"drop_rate": 0.3}}, path)
            loaded, config = load_checkpoint(path, torch.device("cpu"))
            self.assertEqual(config["drop_rate"], 0.3)
            self.assertEqual(
                sum(p.numel() for p in loaded.parameters()),
                sum(p.numel() for p in model.parameters()),
            )


if __name__ == "__main__":
    unittest.main()
