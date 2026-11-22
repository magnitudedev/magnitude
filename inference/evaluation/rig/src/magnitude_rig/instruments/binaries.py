"""Our served binaries as instruments build them in a checkout: the engine (`magnitude-engine`)
and the assessment (`magnitude-assess`), built together in one cargo invocation so they share
one feature unification (no redundant recompile of `magnitude-kernels` / `magnitude-executor`),
and separately from `magnitude-harness` so no development feature the harness enables is unified
into them."""

from ..engine import Step

ENGINE = "{target}/release/magnitude-engine"
ASSESS = "{target}/release/magnitude-assess"
SERVED_BUILD = Step(
    "served",
    ("cargo", "build", "--release", "-p", "magnitude-engine-cli", "-p", "magnitude-engine",
     "--bin", "magnitude-engine", "--bin", "magnitude-assess"),
)
