"""MNIST forward, explicit gradients, SGD, and grouped execution.

--smoke uses deterministic small synthetic data, never claims MNIST accuracy.
Real data is an npz containing x_train/y_train/x_test/y_test; images are uint8
(N,28,28), labels are integers. See prepare_mnist.py for explicit local setup.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import seismic as sm


def reference(x, target, weights):
    w1, b1, w2, b2 = weights
    h = np.maximum(x @ w1.T + b1, 0)
    logits = h @ w2.T + b2
    shifted = logits - logits.max(axis=1, keepdims=True)
    exps = np.exp(shifted)
    probs = exps / exps.sum(axis=1, keepdims=True)
    loss = (target * (np.log(exps.sum(axis=1, keepdims=True)) - shifted)).sum(axis=1)
    dy = (probs - target) / np.float32(len(x))
    dw2 = dy.T @ h
    db2 = dy.sum(axis=0)
    dz = (dy @ w2) * (h > 0)
    return h, logits, loss, (dz.T @ x, dz.sum(axis=0), dw2, db2)


def grouped_forward(kernels, dev, x, w1, b1, w2, b2):
    graph = sm.Workflow(device=dev)
    z = graph.enqueue(kernels["dense"], x, w1, b1)
    hidden = graph.enqueue(kernels["relu"], z)
    logits = graph.enqueue(kernels["dense"], hidden, w2, b2)
    return graph.run((hidden, logits))


def grouped_step(kernels, dev, x, target, w1, b1, w2, b2, rate):
    graph = sm.Workflow(device=dev)
    z = graph.enqueue(kernels["dense"], x, w1, b1)
    hidden = graph.enqueue(kernels["relu"], z)
    logits = graph.enqueue(kernels["dense"], hidden, w2, b2)
    loss, dy = graph.enqueue(kernels["loss"], logits, target)
    dw2, db2, dh = graph.enqueue(kernels["dense_backward"], hidden, w2, dy)
    dz = graph.enqueue(kernels["relu_backward"], hidden, dh)
    dw1, db1, _ = graph.enqueue(kernels["dense_backward"], x, w1, dz)
    graph.enqueue(kernels["sgd"], w1, b1, dw1, db1, rate)
    graph.enqueue(kernels["sgd"], w2, b2, dw2, db2, rate)
    return graph.run(loss)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--route", choices=("workflow", "composed"), default="workflow")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--full-size",
        action="store_true",
        help="use 784→128→10 for synthetic smoke data",
    )
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--rate", type=float, default=0.05)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.epochs < 0:
        parser.error("batch size must be positive and epochs nonnegative")
    rng = np.random.default_rng(args.seed)
    if args.smoke:
        shape = (784, 128, 10) if args.full_size else (4, 3, 2)
        x = rng.normal(size=(5, shape[0])).astype(np.float32)
        labels = np.arange(5) % shape[2]
        x_train, labels_train = x, labels
        provenance = "deterministic synthetic fixture, not MNIST"
    else:
        if args.data is None:
            parser.error("provide --data or --smoke")
        with np.load(args.data, allow_pickle=False) as data:
            x = data["x_test"].reshape(-1, 784).astype(np.float32) / np.float32(255)
            labels = data["y_test"]
            x_train = data["x_train"].reshape(-1, 784).astype(np.float32) / np.float32(
                255
            )
            labels_train = data["y_train"]
        shape = (784, 128, 10)
        provenance = (
            f"dataset sha256={hashlib.sha256(args.data.read_bytes()).hexdigest()}"
        )
    inputs, hidden, classes = shape
    if args.weights:
        with np.load(args.weights, allow_pickle=False) as data:
            weights = [data[n].copy() for n in ("w1", "b1", "w2", "b2")]
        provenance += (
            f"; weights sha256={hashlib.sha256(args.weights.read_bytes()).hexdigest()}"
        )
    else:
        weights = [
            (rng.normal(size=(hidden, inputs)) * 0.02).astype(np.float32),
            np.zeros(hidden, np.float32),
            (rng.normal(size=(classes, hidden)) * 0.02).astype(np.float32),
            np.zeros(classes, np.float32),
        ]
        provenance += "; seeded random initialization (not pretrained)"
    expected_shapes = ((hidden, inputs), (hidden,), (classes, hidden), (classes,))
    if any(
        w.dtype != np.float32 or w.shape != s for w, s in zip(weights, expected_shapes)
    ):
        raise ValueError("weight shape/dtype mismatch")
    dev = sm.device(args.device)
    module = sm.load(Path(__file__).with_suffix(".seismic"), std=False)
    names = (
        "forward",
        "dense",
        "relu",
        "loss",
        "dense_backward",
        "relu_backward",
        "sgd",
        "step",
    )
    kernels = {
        n: module["mnist_" + n].prepare(device=dev, evaluation=sm.Feedback(0))
        for n in names
        if n not in ("forward", "step") or args.route == "composed"
    }
    forward = (
        kernels["forward"]
        if args.route == "composed"
        else lambda *a: grouped_forward(kernels, dev, *a)
    )
    step = (
        kernels["step"]
        if args.route == "composed"
        else lambda *a: grouped_step(kernels, dev, *a)
    )
    resident = [sm.asarray(w, device=dev) for w in weights]
    batch = min(args.batch_size, len(x))
    xb = x[:batch]
    target = np.eye(classes, dtype=np.float32)[labels[:batch]]
    tx, tt = (sm.asarray(v, device=dev) for v in (xb, target))
    h, logits = forward(tx, *resident)
    loss, dy = kernels["loss"](logits, tt)
    dw2, db2, dh = kernels["dense_backward"](h, resident[2], dy)
    dz = kernels["relu_backward"](h, dh)
    dw1, db1, _ = kernels["dense_backward"](tx, resident[0], dz)
    expected = reference(xb, target, weights)
    tolerance = {"atol": 2e-5, "rtol": 2e-4, "preserve_signed_zero": False}
    for actual, want in zip((h, logits, loss, (dw1, db1, dw2, db2)), expected):
        sm.testing.assert_close(actual, want, **tolerance)
    # One complete selected-route step uses independent copies.
    updated = [v.copy() for v in resident]
    step_loss = step(tx, tt, *updated, args.rate)
    sm.testing.assert_close(step_loss, expected[2], **tolerance)
    for actual, w, g in zip(updated, weights, expected[3]):
        sm.testing.assert_close(actual, w - np.float32(args.rate) * g, **tolerance)
    # Explicit grouped host calls exercise producer edges separately.
    workflow = sm.Workflow(device=dev)
    z = workflow.enqueue(kernels["dense"], tx, resident[0], resident[1])
    hidden_pending = workflow.enqueue(kernels["relu"], z)
    logits_pending = workflow.enqueue(
        kernels["dense"], hidden_pending, resident[2], resident[3]
    )
    grouped = workflow.run(logits_pending)
    sm.testing.assert_close(grouped, expected[1], **tolerance)
    print(
        json.dumps(
            {
                "seed": args.seed,
                "shape": shape,
                "dtype": "float32",
                "batch_size": args.batch_size,
                "preprocessing": "uint8 / 255; row-major flatten"
                if not args.smoke
                else "synthetic float32",
                "device": dev.name,
                "preparation": dict(kernels["dense"].preparation_report),
                "route": args.route,
                "policy": "exact",
                "reference": "NumPy float32",
                "tolerances": tolerance,
                "provenance": provenance,
            },
            indent=2,
        )
    )
    for epoch in range(args.epochs):
        order = rng.permutation(len(x_train))
        total = 0.0
        for start in range(0, len(order), args.batch_size):
            indices = order[start : start + args.batch_size]
            bx = sm.asarray(x_train[indices], device=dev)
            bt = sm.asarray(
                np.eye(classes, dtype=np.float32)[labels_train[indices]], device=dev
            )
            total += float(step(bx, bt, *resident, args.rate).numpy().sum())
        print(f"epoch={epoch + 1} mean_loss={total / len(order):.6f}")
    all_x = sm.asarray(x, device=dev)
    correct = 0
    for start in range(0, len(x), args.batch_size):
        _, out = forward(all_x[start : start + args.batch_size], *resident)
        correct += int(
            (
                out.numpy().argmax(axis=1) == labels[start : start + args.batch_size]
            ).sum()
        )
    label = "synthetic fixture agreement" if args.smoke else "MNIST test accuracy"
    print(f"{label}: {correct}/{len(x)} = {correct / len(x):.4%}")


if __name__ == "__main__":
    main()
