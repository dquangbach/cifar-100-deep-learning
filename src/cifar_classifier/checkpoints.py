"""Load WRN checkpoints, including the existing Keras 2.15 Windows files."""

import tempfile
import zipfile
from pathlib import Path

import h5py
import tensorflow as tf

from .model import build_wrn_16_4


def load_checkpoint(path: Path) -> tf.keras.Model:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        model = tf.keras.models.load_model(str(path), compile=False)
    except (TypeError, ValueError) as direct_error:
        # Old .keras archives contain Keras 2 module names and HDF5 groups
        # with Windows backslashes. Rebuild the known architecture and load
        # only its weights; the original checkpoint is never changed.
        model = build_wrn_16_4()
        try:
            with tempfile.TemporaryDirectory(prefix="cifar-wrn-") as temp_dir:
                original = Path(temp_dir) / "original.h5"
                compatible = Path(temp_dir) / "compatible.weights.h5"
                with zipfile.ZipFile(path) as archive:
                    original.write_bytes(archive.read("model.weights.h5"))
                with h5py.File(original, "r") as source, h5py.File(compatible, "w") as target:
                    def copy_item(name, item):
                        name = name.replace("\\", "/")
                        if isinstance(item, h5py.Group):
                            output = target.require_group(name)
                        else:
                            parent, _, leaf = name.rpartition("/")
                            output = target.require_group(parent).create_dataset(
                                leaf, data=item[()], dtype=item.dtype
                            )
                        for key, value in item.attrs.items():
                            output.attrs[key] = value

                    source.visititems(copy_item)
                model.load_weights(str(compatible))
        except (KeyError, OSError, ValueError, zipfile.BadZipFile) as error:
            raise RuntimeError(
                f"Could not load {path} directly or as a legacy WRN checkpoint"
            ) from error
        print(f"Loaded legacy checkpoint weights ({type(direct_error).__name__}).")
    if model.input_shape != (None, 32, 32, 3) or model.output_shape != (None, 100):
        raise ValueError(f"Checkpoint is not a CIFAR-100 WRN: {path}")
    return model
