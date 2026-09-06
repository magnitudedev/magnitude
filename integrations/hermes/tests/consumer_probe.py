"""Run in the installed Hermes interpreter against a native-installed consumer package."""

from pathlib import Path
import os

from hermes_constants import get_hermes_home
from hermes_cli.plugins import (
    discover_plugins, get_plugin_commands, get_plugin_manager, unload_plugins,
)


def main():
    home = get_hermes_home()
    assert "magnitude-integration-acceptance-" in str(home), "Use a disposable acceptance profile"
    assert not Path(os.environ["MAGNITUDE_CLI"]).exists(), "CLI must be absent in this test"
    discover_plugins()
    commands = get_plugin_commands()
    expected = {"magnitude", "load-model", "stop-model", "magnitude-setup"}
    assert expected.issubset(commands), f"Missing native commands: {expected - commands.keys()}"
    prompt = (
        "Set up local models for me with the Magnitude CLI. Install it with "
        "`npm i -g @magnitudedev/cli` (or my package manager), then run "
        "`magnitude docs onboarding` and follow the instructions."
    )
    assert prompt in commands["magnitude-setup"]["handler"]("")
    manager = get_plugin_manager()
    assert manager.list_plugin_skills("magnitude") == ["usage"]
    shared = home / "skills" / "magnitude" / "SKILL.md"
    shared.parent.mkdir(parents=True)
    contents = (home / "plugins" / "magnitude" / "dist" / "skills" / "magnitude" / "SKILL.md").read_text()
    shared.write_text(contents)
    for _ in range(2):
        discover_plugins(force=True)
        assert manager.list_plugin_skills("magnitude") == [], "Shared skill must replace the package fallback"
        assert shared.read_text() == contents
        assert prompt in get_plugin_commands()["magnitude-setup"]["handler"]("")
    assert unload_plugins("magnitude")
    assert manager.list_plugin_skills("magnitude") == []
    assert shared.read_text() == contents
    print("Hermes native consumer passed: commands, CLI-free setup, skill fallback, shared-skill reload, unload")


if __name__ == "__main__":
    main()
