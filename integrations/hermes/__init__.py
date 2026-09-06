"""Native Hermes entrypoint. Registration deliberately performs no model/service I/O."""

from pathlib import Path

SETUP_PROMPT = (
    "Set up local models for me with the Magnitude CLI. Install it with "
    "`npm i -g @magnitudedev/cli` (or my package manager), then run "
    "`magnitude docs onboarding` and follow the instructions."
)


def register(ctx):
    from .commands import ModelCommands
    from .progress import register_progress

    commands = ModelCommands()
    register_progress(ctx)
    ctx.register_command("magnitude", commands.status, description="Show Magnitude local models")
    ctx.register_command("load-model", commands.load, description="Load an installed Magnitude model")
    ctx.register_command("stop-model", commands.stop, description="Stop the active Magnitude model")
    ctx.register_command(
        "magnitude-setup",
        lambda args: "Send this message to your agent to set up Magnitude:\n\n" + SETUP_PROMPT,
        description="Get the Magnitude setup prompt",
    )

    from hermes_constants import get_hermes_home

    if not (get_hermes_home() / "skills" / "magnitude" / "SKILL.md").is_file():
        ctx.register_skill(
            "usage", Path(__file__).parent / "dist" / "skills" / "magnitude" / "SKILL.md",
            "Set up and use Magnitude local models",
        )
