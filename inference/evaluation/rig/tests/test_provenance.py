"""Build provenance: the recording rustc wrapper, the verifier, and the engine's refusal."""

import json
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from magnitude_rig import engine
from magnitude_rig.host import provenance
from magnitude_rig.sources import Source

needs_cargo = pytest.mark.skipif(shutil.which("cargo") is None, reason="needs cargo")
OLD = 1_600_000_000
WORKSPACE = {
    "Cargo.toml": '[workspace]\nmembers = ["app", "lib"]\nresolver = "2"\n',
    "app/Cargo.toml": (
        '[package]\nname = "my-app"\nversion = "0.1.0"\nedition = "2021"\n'
        '[dependencies]\nmy-lib = { path = "../lib" }\n'
    ),
    "app/src/main.rs": 'fn main() { println!("{} {}", my_lib::who(), my_lib::GENERATED); }\n',
    "lib/Cargo.toml": '[package]\nname = "my-lib"\nversion = "0.1.0"\nedition = "2021"\n',
    "lib/src/lib.rs": (
        'mod part;\npub use part::who;\n'
        'pub const GENERATED: &str = include_str!(concat!(env!("OUT_DIR"), "/generated.txt"));\n'
    ),
    "lib/src/part.rs": 'pub fn who() -> &\'static str { "first" }\n',
    "lib/build.rs": (
        'fn main() {\n    let out = std::env::var("OUT_DIR").unwrap();\n'
        '    std::fs::write(format!("{out}/generated.txt"), "generated").unwrap();\n'
        '    println!("cargo:rerun-if-changed=build.rs");\n}\n'
    ),
}
SECOND = 'pub fn who() -> &\'static str { "second" }\n'


def workspace(directory: Path) -> Path:
    for path, text in WORKSPACE.items():
        (directory / path).parent.mkdir(parents=True, exist_ok=True)
        (directory / path).write_text(text)
    return directory


def cargo_build(source: Path, target: Path, wrapper: str | None) -> str:
    env = {k: v for k, v in os.environ.items() if k != engine.COMPILER}
    if wrapper is not None:
        env[engine.COMPILER] = wrapper
    return subprocess.run(
        ["cargo", "build", "--release", "--offline", "--target-dir", str(target)],
        cwd=source, env=env, capture_output=True, text=True, check=True,
    ).stderr


def replace_keeping_cargo_fresh(path: Path, text: str) -> None:
    """Change a source the way the time-preserving mirror did: new bytes, a time older than
    the build outputs, so cargo rebuilds nothing."""
    path.write_text(text)
    os.utime(path, (OLD, OLD))


def test_dep_info_lists_outputs_and_sources_with_escaped_spaces(tmp_path):
    path = tmp_path / "unit.d"
    path.write_text(
        "/t/deps/unit.d: src/lib.rs src/a\\ b.rs\n\n"
        "/t/deps/libunit.rlib: src/lib.rs src/a\\ b.rs /t/build/out/gen.rs\n\n"
        "src/lib.rs:\nsrc/a\\ b.rs:\n/t/build/out/gen.rs:\n\n# env-dep:OUT_DIR=/t/build/out\n"
    )
    assert provenance.dep_info(str(path)) == (
        ["/t/deps/unit.d", "/t/deps/libunit.rlib"],
        ["src/lib.rs", "src/a b.rs", "/t/build/out/gen.rs"],
    )


def test_only_compilations_are_units():
    assert provenance.Unit.parse(["-vV"]) is None
    assert provenance.Unit.parse(["-", "--crate-name", "___", "--print=file-names"]) is None
    unit = provenance.Unit.parse([
        "--crate-name", "my_lib", "src/lib.rs", "--emit=dep-info,metadata,link",
        "-C", "extra-filename=-abc", "--out-dir", "/t/release/deps",
        "--extern", "dep=/t/release/deps/libdep-1.rmeta", "--extern", "proc_macro",
    ])
    assert unit.record == "/t/release/deps/my_lib-abc.provenance.json"
    assert unit.dep_info == "/t/release/deps/my_lib-abc.d"
    assert unit.externs == ["/t/release/deps/libdep-1.rmeta"]


def test_built_executables_are_named_files_of_a_cargo_profile_directory(tmp_path):
    profile = tmp_path / "target" / "release"
    (profile / "deps").mkdir(parents=True)
    binary, script = profile / "tool", tmp_path / "script.sh"
    for path in (binary, script):
        path.write_text("x")
        path.chmod(0o755)
    (profile / "notes.txt").write_text("x")
    texts = [str(binary), f"--harness={binary}", str(script), str(profile / "notes.txt"),
             str(profile / "absent"), "relative/tool", "--flag"]  # fmt: skip
    assert provenance.built_executables(texts) == [os.path.realpath(binary)]
    assert provenance.built_executables([f"ls -l {binary} && '{binary}' run"]) == [os.path.realpath(binary)]
    assert provenance.built_executables(["it's not a command line"]) == []
    assert provenance.verify([str(script)], [])["status"] == "no-built-executable"


@needs_cargo
def test_executable_built_through_the_wrapper_is_verified(tmp_path):
    source, target = workspace(tmp_path / "ws"), tmp_path / "target"
    wrapper = provenance.install(str(tmp_path / "root"))
    cargo_build(source, target, wrapper)
    binary = str(target / "release" / "my-app")
    record = provenance.verify([binary], [str(source)])
    assert record["status"] == "verified"
    (executable,) = record["executables"]
    assert executable["units"] == 2 and executable["reasons"] == []
    assert executable["sources"] == 4  # main.rs, lib.rs, part.rs and the generated include
    # A real change is rebuilt and verified again; its record follows the new bytes.
    (source / "lib/src/part.rs").write_text(SECOND)
    assert "Compiling my-lib" in cargo_build(source, target, wrapper)
    again = provenance.verify([binary], [str(source)])
    assert again["status"] == "verified"
    assert again["executables"][0]["sources_digest"] != executable["sources_digest"]


@needs_cargo
def test_reused_unit_of_other_content_is_contradicted(tmp_path):
    source, target = workspace(tmp_path / "ws"), tmp_path / "target"
    wrapper = provenance.install(str(tmp_path / "root"))
    cargo_build(source, target, wrapper)
    replace_keeping_cargo_fresh(source / "lib/src/part.rs", SECOND)
    assert "Compiling" not in cargo_build(source, target, wrapper)  # cargo's freshness is fooled
    record = provenance.verify([str(target / "release" / "my-app")], [str(source)])
    assert record["status"] == "refused"
    (executable,) = record["executables"]
    assert executable["status"] == "contradicted"
    assert executable["reasons"] == [
        f"my_lib: compiled from other content of {source / 'lib/src/part.rs'}"
    ]


@needs_cargo
def test_executable_not_built_through_the_wrapper_is_unverified(tmp_path):
    source, target = workspace(tmp_path / "ws"), tmp_path / "target"
    cargo_build(source, target, None)
    record = provenance.verify([str(target / "release" / "my-app")], [str(source)])
    assert record["status"] == "refused"
    assert record["executables"][0]["status"] == "unverified"


@needs_cargo
def test_relinking_against_an_unrecorded_workspace_unit_is_unverified(tmp_path):
    source, target = workspace(tmp_path / "ws"), tmp_path / "target"
    wrapper = provenance.install(str(tmp_path / "root"))
    cargo_build(source, target, wrapper)
    for record in (target / "release" / "deps").glob("my_lib-*.provenance.json"):
        record.unlink()
    record = provenance.verify([str(target / "release" / "my-app")], [str(source)])
    assert record["executables"][0]["status"] == "unverified"
    assert "compiles synchronized sources but has no compilation record" in "".join(
        record["executables"][0]["reasons"]
    )


def _with_workspace(source: Source) -> Source:
    workspace(source.directory)
    paths = tuple(sorted(p.relative_to(source.directory).as_posix()
                         for p in source.directory.rglob("*") if p.is_file()))
    return replace(source, paths=paths)


@needs_cargo
def test_a_run_refuses_a_binary_it_cannot_tie_to_the_sources(rig):
    hosts, source, tmp_path = rig
    source = _with_workspace(source)
    build = engine.Step("cargo", ("cargo", "build", "--release", "--offline"))
    job = engine.Step("app", ("{target}/release/my-app",))
    request = engine.Request(
        instrument="t", purpose="p", estimate=600, backend="cpu", host="alpha",
        sides=(engine.Side("source", source, (build,)),), jobs=(engine.Job("source", "app", job),),
        device="device",
    )
    outcome = engine.execute(request, engine.new_run_id("test"), hosts=hosts, announce=lambda line: None)
    assert outcome.codes == {"app": 0}
    assert "first generated" in (outcome.results / "logs" / "job-app.log").read_text()
    proof = json.loads((outcome.results / "provenance-source.json").read_text())
    assert proof["status"] == "verified" and proof["executables"][0]["units"] == 2

    # The same new bytes here and on the host, with a time older than the build outputs: the
    # mirror transfers nothing and cargo rebuilds nothing, so the old unit would be reused.
    host_copy = tmp_path / "alpha" / "checkouts" / source.key / "src" / "lib/src/part.rs"
    for path in (source.directory / "lib/src/part.rs", host_copy):
        replace_keeping_cargo_fresh(path, SECOND)
    with pytest.raises(engine.RunFailed) as failed:
        engine.execute(request, engine.new_run_id("test"), hosts=hosts, announce=lambda line: None)
    assert failed.value.phase == "build-source" and "provenance" in str(failed.value)
    assert not (failed.value.outcome.results / "logs" / "job-app.log").exists()


def test_every_build_step_compiles_through_the_checkouts_recording_wrapper(rig):
    hosts, source, tmp_path = rig
    show = engine.Step("show", (sys.executable, "-c", f"import os; print(os.environ['{engine.COMPILER}'])"))
    request = engine.Request(
        instrument="t", purpose="p", estimate=600, backend="cpu", host="alpha",
        sides=(engine.Side("source", source, (show,)),),
    )
    outcome = engine.execute(request, engine.new_run_id("test"), hosts=hosts, announce=lambda line: None)
    wrapper = tmp_path / "alpha" / "checkouts" / source.key / "provenance" / "rustc"
    assert f"{wrapper}\n" in (outcome.results / "logs" / "build-source.log").read_text()
    assert os.access(wrapper, os.X_OK)
