"""The evaluation harness every instrument that drives the engine in process runs: one build per
checkout, in the checkout's target, with the engine's development tuning tools (pins and the
survey, which only the harness enables). `bench` renders through the same executable, so no
instrument rebuilds another's.

Under `ab`, the base's first engine run records the tuning it ran (`RECORD`) and every later
run of both sides replays it (`REPLAY`), so two builds are compared on one tuning: their
difference is the code's, and bit-identical outputs mean identical computation.
"""

from ..engine import Step

EXECUTABLE = "{target}/release/magnitude-harness"
BUILD = Step("harness", ("cargo", "build", "--release", "-p", "magnitude-harness"))
PINS = "{runs}/tuning-pins.json"
RECORD = ("--tuning-record", PINS)
REPLAY = ("--tuning-replay", PINS)
