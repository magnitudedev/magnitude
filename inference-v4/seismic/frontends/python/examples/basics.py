"""Run with: python basics.py --device cpu"""

import argparse

import numpy as np

import seismic as sm

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="cpu")
args = parser.parse_args()
device = sm.device(args.device)
module = sm.load_source(
    """
fn affine[N](x: &tensor[N] f32, scale: f32) -> tensor[N] f32:
    let mut y = zeros_like(x)
    parallel for i in 0..N:
        y[i] = x[i] * scale + 1.0
    return y
""",
    std=False,
)
kernel = module.affine.prepare(device=device, evaluation=sm.Feedback(search_seconds=0))
x = sm.asarray(np.arange(8, dtype=np.float32), device=device)
y = kernel(x, scale=2.0)
sm.testing.assert_close(y, np.arange(8, dtype=np.float32) * 2 + 1)
sm.testing.check(kernel, args=(x, 2.0)).assert_passed()
print(y.numpy())
print(sm.benchmark(kernel, args=(x, 2.0)))
