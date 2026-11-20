"""Template files: what every instrument measures, named `<fixture>-<responsekind>:<context>`.

The fixtures package parses the name (`Template.parse`) and builds the sessions; the rig builds a
run's template file from them, holding the lanes the run needs (lanes 0 … n − 1), and stores it
by its content digest: equal files are one file, the same template at the same fixtures commit
and lanes, which the recorded digests prove. The harness reads the file and sizes each lane to
its context with the model's own chat template.
"""

import argparse
import os
from functools import cache
from pathlib import Path

from magnitude_fixtures import Template

from .state import HOME

TEMPLATES = HOME / "templates"


def name(text: str) -> Template:
    """`--template NAME`, checked before anything queues."""
    try:
        return Template.parse(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from None


@cache
def file(template: Template, lanes: int) -> Path:
    """The template file holding lanes 0 … `lanes` − 1 of `template`, built from this checkout's
    fixtures and stored by its content digest."""
    built = template.build(lanes)
    path = TEMPLATES / f"{built.digest[:16]}.json"
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        staged = path.with_suffix(f".{os.getpid()}.tmp")
        staged.write_bytes(built.encode())
        staged.replace(path)
    return path


def argument(path: Path) -> str:
    """A template file as a job reads it, among the run's inputs."""
    return f"{{inputs}}/{path.name}"
