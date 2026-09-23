"""Explicit local setup: convert official MNIST IDX files, optionally pretrain.

Accepts the four IDX or IDX.gz files already acquired by the caller. This script
has no network activity. Training uses the same float32 NumPy reference equations
as mnist.py and records its seed/settings alongside the saved weights.
"""

import argparse
import gzip
import io
import json
import struct
from pathlib import Path

import numpy as np
from mnist import reference


def idx(path):
    with gzip.open(path, "rb") if path.suffix == ".gz" else path.open("rb") as handle:
        raw = handle.read()
    zero, dtype, rank = struct.unpack(">HBB", raw[:4])
    if zero != 0 or dtype != 8 or rank not in (1, 3):
        raise ValueError(f"not a uint8 MNIST IDX file: {path}")
    shape = struct.unpack(">" + "I" * rank, raw[4 : 4 + 4 * rank])
    return np.frombuffer(raw[4 + 4 * rank :], dtype=np.uint8).reshape(shape).copy()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("train-images", "train-labels", "test-images", "test-labels"):
        p.add_argument("--" + name, type=Path)
    p.add_argument(
        "--hf-parquet", type=Path, help="local ylecun/mnist snapshot directory"
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--weights", type=Path)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--rate", type=float, default=0.05)
    a = p.parse_args()
    paths = (a.train_images, a.train_labels, a.test_images, a.test_labels)
    if a.hf_parquet:
        if any(paths):
            p.error("choose local parquet or IDX inputs")
        import pyarrow.parquet as pq
        from PIL import Image

        arrays = []
        for split in ("train", "test"):
            rows = pq.read_table(
                a.hf_parquet / "mnist" / f"{split}-00000-of-00001.parquet"
            ).to_pylist()
            arrays.extend(
                (
                    np.stack(
                        [
                            np.array(Image.open(io.BytesIO(row["image"]["bytes"])))
                            for row in rows
                        ]
                    ),
                    np.array([row["label"] for row in rows], dtype=np.int64),
                )
            )
        x_train, y_train, x_test, y_test = arrays
    else:
        if not all(paths):
            p.error("provide all four IDX paths or --hf-parquet")
        x_train, y_train, x_test, y_test = (idx(v) for v in paths)
    np.savez(a.output, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
    if a.weights is None:
        return
    if a.epochs < 1 or a.batch_size < 1:
        p.error("pretraining requires positive epochs and batch size")
    rng = np.random.default_rng(a.seed)
    x = x_train.reshape(-1, 784).astype(np.float32) / np.float32(255)
    weights = [
        (rng.normal(size=(128, 784)) * np.sqrt(2 / 784)).astype(np.float32),
        np.zeros(128, np.float32),
        (rng.normal(size=(10, 128)) * np.sqrt(2 / 128)).astype(np.float32),
        np.zeros(10, np.float32),
    ]
    for epoch in range(a.epochs):
        order = rng.permutation(len(x))
        total = 0.0
        for start in range(0, len(x), a.batch_size):
            rows = order[start : start + a.batch_size]
            _, _, loss, gradients = reference(
                x[rows], np.eye(10, dtype=np.float32)[y_train[rows]], weights
            )
            for w, g in zip(weights, gradients):
                w -= np.float32(a.rate) * g
            total += float(loss.sum())
        print(f"NumPy setup epoch={epoch + 1} mean_loss={total / len(x):.6f}")
    np.savez(a.weights, **dict(zip(("w1", "b1", "w2", "b2"), weights)))
    a.weights.with_suffix(".json").write_text(
        json.dumps(
            {
                "seed": a.seed,
                "epochs": a.epochs,
                "batch_size": a.batch_size,
                "rate": a.rate,
                "reference": "NumPy float32 SGD",
                "architecture": [784, 128, 10],
                "preprocessing": "uint8 / 255",
                "data": str(a.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
