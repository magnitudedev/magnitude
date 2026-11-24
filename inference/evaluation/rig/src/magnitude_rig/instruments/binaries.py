"""Our served binaries as instruments build them in a checkout: the engine (`magnitude-engine`),
the assessment (`magnitude-assess`), and the host-only template renderer (`magnitude-render`),
built together in one cargo invocation with `--no-default-features` so they share one feature
unification (compiling `magnitude-kernels`, `magnitude-executor`, and `magnitude-engine` once)
without enabling any development feature of `magnitude-harness`."""

from ..engine import Step

ENGINE = "{target}/release/magnitude-engine"
ASSESS = "{target}/release/magnitude-assess"
RENDER = "{target}/release/magnitude-render"
SERVED_BUILD = Step(
    "served",
    ("cargo", "build", "--release", "--no-default-features",
     "-p", "magnitude-engine-cli", "-p", "magnitude-engine", "-p", "magnitude-harness",
     "--bin", "magnitude-engine", "--bin", "magnitude-assess", "--bin", "magnitude-render"),
)
RENDER_BUILD = Step(
    "render",
    ("cargo", "build", "--release", "--no-default-features",
     "-p", "magnitude-harness", "--bin", "magnitude-render"),
)
