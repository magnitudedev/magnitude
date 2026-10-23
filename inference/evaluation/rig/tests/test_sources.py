"""A ref is measured by this worktree's tools: its `evaluation/` tree is this one's, its
engine its own, and its key names its commit and the harness it carries."""

from magnitude_rig import sources
from magnitude_rig.sync import INFERENCE


def test_a_ref_carries_this_worktrees_tools_and_its_own_engine():
    base = sources.commit("HEAD~1")
    with sources.ref("HEAD~1") as ref:
        assert ref.key.startswith(f"ref-{base[:12]}-")
        assert ref.identity.commit == base and ref.identity.dirty_digest is None
        assert (ref.directory / "evaluation/rig/src/magnitude_rig/host/agent.py").read_bytes() == (
            INFERENCE / "evaluation/rig/src/magnitude_rig/host/agent.py"
        ).read_bytes()
        assert "evaluation/rig/src/magnitude_rig/engine.py" in ref.paths
        assert "engine/Cargo.toml" in ref.paths
        directory = ref.directory
    assert not directory.exists()  # materialized for the run only
