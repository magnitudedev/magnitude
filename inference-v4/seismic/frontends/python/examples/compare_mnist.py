"""Compare resident-data training epochs: Seismic workflows and PyTorch autograd.

Build Seismic with maturin develop --release before collecting performance data.
No downloads, mixed precision, or compilation of PyTorch graphs occur here.
"""

import argparse
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from mnist import grouped_forward, grouped_step, reference
from torch import nn

import seismic as sm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "metal"), default="metal")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--limit", type=int, help="explicit training subset; not a full MNIST epoch"
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--rate", type=float, default=0.05)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size) <= 0 or (
        args.limit is not None and args.limit <= 0
    ):
        parser.error("epochs, batch size, and limit must be positive")
    with np.load(args.data, allow_pickle=False) as data:
        x = data["x_train"].reshape(-1, 784).astype(np.float32) / np.float32(255)
        y = data["y_train"].astype(np.int64)
        test_x = data["x_test"].reshape(-1, 784).astype(np.float32) / np.float32(255)
        test_y = data["y_test"].astype(np.int64)
    x, y = x[: args.limit], y[: args.limit]
    if not len(x) or len(x) != len(y):
        raise ValueError("invalid training dataset")
    rng = np.random.default_rng(args.seed)
    weights = [
        (rng.normal(size=(128, 784)) * 0.02).astype(np.float32),
        np.zeros(128, np.float32),
        (rng.normal(size=(10, 128)) * 0.02).astype(np.float32),
        np.zeros(10, np.float32),
    ]
    dev = sm.device(args.device)
    td = torch.device("mps" if args.device == "metal" else "cpu")
    if td.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("PyTorch MPS is unavailable")

    def sync():
        if td.type == "mps":
            torch.mps.synchronize()

    setup = {}
    start = time.perf_counter()
    module = sm.load(Path(__file__).with_name("mnist.seismic"), std=False)
    kernels = {
        n: module["mnist_" + n].prepare(device=dev, evaluation=sm.Feedback(0))
        for n in ("dense", "relu", "loss", "dense_backward", "relu_backward", "sgd")
    }
    setup["seismic_prepare_seconds"] = time.perf_counter() - start
    start = time.perf_counter()
    model = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10)).to(td)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.rate, foreach=False)
    sync()
    setup["torch_prepare_seconds"] = time.perf_counter() - start

    def reset():
        with torch.no_grad():
            for parameter, initial in zip(model.parameters(), weights):
                parameter.copy_(torch.tensor(initial, device=td))
        return [sm.asarray(w, device=dev) for w in weights]

    def torch_step(bx, by):
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(bx), by)
        loss.backward()
        optimizer.step()
        return loss

    # Check forward, every parameter gradient, loss and the actual workflow update.
    # Warm every batch geometry (including the tail), then discard all warmup state.
    start = time.perf_counter()
    for batch in sorted({min(args.batch_size, len(x)), len(x) % args.batch_size} - {0}):
        resident = reset()
        bx, by = x[:batch], y[:batch]
        target = np.eye(10, dtype=np.float32)[by]
        sx, sy = (sm.asarray(v, device=dev) for v in (bx, target))
        tx, ty = torch.tensor(bx, device=td), torch.tensor(by, device=td)
        expected_h, expected_logits, expected_loss, expected_grad = reference(
            bx, target, weights
        )
        h, logits = grouped_forward(kernels, dev, sx, *resident)
        loss, dy = kernels["loss"](logits, sy)
        dw2, db2, dh = kernels["dense_backward"](h, resident[2], dy)
        dz = kernels["relu_backward"](h, dh)
        dw1, db1, _ = kernels["dense_backward"](sx, resident[0], dz)
        optimizer.zero_grad(set_to_none=True)
        th = model[:2](tx)
        tl = model[2](th)
        torch_loss = nn.functional.cross_entropy(tl, ty)
        torch_loss.backward()
        pairs = [
            (h.numpy(), expected_h),
            (logits.numpy(), expected_logits),
            (loss.numpy(), expected_loss),
            (th.detach().cpu().numpy(), expected_h),
            (tl.detach().cpu().numpy(), expected_logits),
            (torch_loss.detach().cpu().numpy(), expected_loss.mean()),
        ]
        pairs += [
            (g.numpy(), ref) for g, ref in zip((dw1, db1, dw2, db2), expected_grad)
        ]
        pairs += [
            (p.grad.cpu().numpy(), ref)
            for p, ref in zip(model.parameters(), expected_grad)
        ]
        for actual, expected in pairs:
            np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-4)
        grouped_step(kernels, dev, sx, sy, *resident, args.rate)
        optimizer.step()
        for sw, tw, w, g in zip(resident, model.parameters(), weights, expected_grad):
            expected = w - np.float32(args.rate) * g
            np.testing.assert_allclose(sw.numpy(), expected, atol=2e-5, rtol=2e-4)
            np.testing.assert_allclose(
                tw.detach().cpu().numpy(), expected, atol=2e-5, rtol=2e-4
            )
        # Warm the timed Torch path as well.
        torch_step(tx, ty)
    sync()
    setup["validation_and_warmup_seconds"] = time.perf_counter() - start
    resident = reset()
    records = []
    for epoch in range(args.epochs):
        order = rng.permutation(len(x))
        bx, by = x[order], y[order]
        start = time.perf_counter()
        sx = sm.asarray(bx, device=dev)
        sy = sm.asarray(np.eye(10, dtype=np.float32)[by], device=dev)
        seismic_upload = time.perf_counter() - start
        sync()
        start = time.perf_counter()
        tx, ty = torch.tensor(bx, device=td), torch.tensor(by, device=td)
        sync()
        torch_upload = time.perf_counter() - start
        durations = {}
        last_batch_loss = {}
        # Alternate first runner to reduce systematic ordering bias.
        for runner in ("seismic", "torch") if epoch % 2 == 0 else ("torch", "seismic"):
            sync()
            start = time.perf_counter()
            for lo in range(0, len(x), args.batch_size):
                hi = min(lo + args.batch_size, len(x))
                if runner == "seismic":
                    last_loss = grouped_step(
                        kernels, dev, sx[lo:hi], sy[lo:hi], *resident, args.rate
                    )
                else:
                    last_loss = torch_step(tx[lo:hi], ty[lo:hi])
            sync()
            durations[runner] = time.perf_counter() - start
            last_batch_loss[runner] = (
                float(last_loss.numpy().mean())
                if runner == "seismic"
                else float(last_loss.detach().cpu())
            )
        record = {
            "epoch": epoch + 1,
            "last_batch_loss": last_batch_loss,
            "seconds": durations,
            "examples_per_second": {k: len(x) / v for k, v in durations.items()},
            "upload_seconds": {"seismic": seismic_upload, "torch": torch_upload},
        }
        records.append(record)
        print(json.dumps(record), flush=True)
    # Evaluate both trained models outside the timers; report actual test accuracy.
    correct = {"seismic": 0, "torch": 0}
    for lo in range(0, len(test_x), args.batch_size):
        bx, by = test_x[lo : lo + args.batch_size], test_y[lo : lo + args.batch_size]
        _, logits = grouped_forward(kernels, dev, sm.asarray(bx, device=dev), *resident)
        correct["seismic"] += int((logits.numpy().argmax(axis=1) == by).sum())
        with torch.no_grad():
            prediction = model(torch.tensor(bx, device=td)).argmax(dim=1).cpu().numpy()
        correct["torch"] += int((prediction == by).sum())
    medians = {k: statistics.median(r["seconds"][k] for r in records) for k in correct}
    report = {
        "shape": [784, 128, 10],
        "dtype": "float32",
        "batch_size": args.batch_size,
        "seed": args.seed,
        "learning_rate": args.rate,
        "training_examples": len(x),
        "test_examples": len(test_x),
        "device": dev.name,
        "torch_device": str(td),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_cpu_threads": torch.get_num_threads(),
        "dataset_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(
            Path(__file__).with_name("mnist.seismic").read_bytes()
        ).hexdigest(),
        "protocol": "resident data; eager PyTorch autograd vs exact Seismic explicit-gradient workflow; synchronized epoch boundaries; uploads/preparation/evaluation excluded",
        "validation": "forward, loss, all gradients and one update passed against NumPy",
        "setup": setup,
        "epochs": records,
        "median_seconds": medians,
        "seismic_over_torch_time": medians["seismic"] / medians["torch"],
        "test_accuracy": {k: v / len(test_x) for k, v in correct.items()},
    }
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
