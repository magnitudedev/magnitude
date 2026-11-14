"""A whole GGUF set the host already holds, as the subject of a run in place of a catalog model: the one owner of
`--checkpoint DIR` for every instrument that serves, drafts or judges one (`kl`, `draft`, `bench`). The set is read
where it lies (`host/checkpoint.py` writes the run's acquisition object, naming the set by its bytes): nothing is
fetched, converted or copied, so a run has no catalog acquisition (`acquires` is None) and its first job writes
`{artifacts}`."""

import argparse

from ..engine import Step, Weights

# The run's preparation of the set, which writes the run's acquisition object (`{artifacts}`).
CHECKPOINT = "checkpoint"


def option(parser: argparse.ArgumentParser, detail: str = "") -> None:
    """`--checkpoint DIR`, beside the instrument's `--model` (selector `model-or-checkpoint`)."""
    parser.add_argument("--checkpoint", metavar="DIR",
                        help="a whole GGUF set the host already holds in this directory, read where it lies: "
                        f"nothing is fetched, converted or copied{detail}")


def subject(command: str, arguments) -> None:
    """Refuse a run that names both a catalog model and a held set, or neither."""
    if (arguments.model is None) == (arguments.checkpoint is None):
        raise SystemExit(f"{command}: give --model or --checkpoint, one")


def prepare(arguments) -> Step:
    return Step(CHECKPOINT, ("{python}", "-m", "magnitude_rig.host.checkpoint", "--path", arguments.checkpoint,
                             "--output", "{artifacts}"))  # fmt: skip


def jobs(arguments) -> tuple[tuple[str, Step], ...]:
    """The job that writes the run's acquisition object from the held set, first; none for a catalog model."""
    return ((CHECKPOINT, prepare(arguments)),) if arguments.checkpoint is not None else ()


def acquires(arguments) -> Weights | None:
    """What the run's jobs read of the catalog: the model's files; none for a held set."""
    return None if arguments.checkpoint is not None else Weights(arguments.model)


def name(arguments) -> str:
    """The subject as an answer names it."""
    return arguments.model or f"checkpoint {arguments.checkpoint}"
