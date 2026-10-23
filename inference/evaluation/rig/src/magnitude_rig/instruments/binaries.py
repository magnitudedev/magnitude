"""Our served binaries as instruments build them in a checkout: the engine (`magnitude-engine`)
and the assessment (`magnitude-assess`), each by its own cargo invocation, so no development
feature another build enables (the harness's) is unified into them."""

from ..engine import Step

ENGINE = "{target}/release/magnitude-engine"
ENGINE_BUILD = Step("engine", ("cargo", "build", "--release", "-p", "magnitude-engine-cli"))
ASSESS = "{target}/release/magnitude-assess"
ASSESS_BUILD = Step("assess", ("cargo", "build", "--release", "-p", "magnitude-engine", "--bin", "magnitude-assess"))
