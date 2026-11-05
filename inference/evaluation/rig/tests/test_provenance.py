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
        "# env-dep:WINDOWS=C:\\\\t\\\\out\n# env-dep:LINES=a\\nb\n# env-dep:UNSET\n"
    )
    assert provenance.dep_info(str(path)) == (
        ["/t/deps/unit.d", "/t/deps/libunit.rlib"],
        ["src/lib.rs", "src/a b.rs", "/t/build/out/gen.rs"],
        {"OUT_DIR": "/t/build/out", "WINDOWS": "C:\\t\\out", "LINES": "a\nb", "UNSET": None},
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
    assert provenance.verify([str(script)], [], str(tmp_path / "target"))["status"] == "no-built-executable"


@needs_cargo
def test_executable_built_through_the_wrapper_is_verified(tmp_path):
    source, target = workspace(tmp_path / "ws"), tmp_path / "target"
    wrapper = provenance.install(str(tmp_path / "root"))
    cargo_build(source, target, wrapper)
    binary = str(target / "release" / "my-app")
    record = provenance.verify([binary], [str(source)], str(target))
    assert record["status"] == "verified"
    (executable,) = record["executables"]
    assert executable["units"] == 2 and executable["reasons"] == []
    assert executable["sources"] == 4  # main.rs, lib.rs, part.rs and the generated include
    # A real change is rebuilt and verified again; its record follows the new bytes.
    (source / "lib/src/part.rs").write_text(SECOND)
    assert "Compiling my-lib" in cargo_build(source, target, wrapper)
    again = provenance.verify([binary], [str(source)], str(target))
    assert again["status"] == "verified"
    assert again["executables"][0]["sources_digest"] != executable["sources_digest"]


def test_a_unit_names_its_source_tree_when_it_reads_a_path_in_the_synchronized_roots(tmp_path, monkeypatch):
    """Any variable counts whose value is a path in the synchronized roots: the manifest
    directory or one a build script sets. A registry crate's own manifest directory, the
    build's output directory beside the source, and values that are no path do not."""
    tmp_path = Path(os.path.realpath(tmp_path))
    monkeypatch.chdir(tmp_path)  # where a relative value would resolve
    source, target = tmp_path / "src", tmp_path / "target"
    env = {"CARGO_MANIFEST_DIR": str(source / "lib"), "ASSETS": str(source / "assets"),
           "REGISTRY": str(tmp_path / "registry" / "dep"), "OUT_DIR": str(target / "release" / "build"),
           "VERSION": "src", "UNSET": None}  # fmt: skip
    assert provenance.source_tree_reads(env, [str(source)]) == ["CARGO_MANIFEST_DIR", "ASSETS"]
    assert provenance.source_tree_reads(env, [str(tmp_path / "elsewhere")]) == []


@needs_cargo
def test_a_dependency_recorded_under_another_schema_is_unverified_until_the_new_wrapper_rebuilds_it(
        tmp_path, monkeypatch):
    """A checkout built by a rig of another schema: its wrapper had another path, so the
    wrapper of this schema compiles every workspace unit again."""
    source, target = workspace(tmp_path / "ws"), tmp_path / "target"
    binary = str(target / "release" / "my-app")
    with monkeypatch.context() as earlier:
        earlier.setattr(provenance, "WRAPPER", "provenance/build-provenance-unit/1/rustc")
        cargo_build(source, target, provenance.install(str(tmp_path / "root")))
    (record,) = (target / "release" / "deps").glob(f"my_lib-*{provenance.RECORD_SUFFIX}")
    record.write_text(json.dumps({**json.loads(record.read_text()), "schema": "build-provenance-unit/1"}))
    (executable,) = provenance.verify([binary], [str(source)], str(target))["executables"]
    assert executable["status"] == "unverified"
    assert executable["reasons"] == [f"my_lib: its record is of build-provenance-unit/1, not {provenance.UNIT_SCHEMA}"]
    assert "Compiling my-lib" in cargo_build(source, target, provenance.install(str(tmp_path / "root")))
    assert provenance.verify([binary], [str(source)], str(target))["status"] == "verified"


@needs_cargo
def test_an_executable_that_names_its_source_tree_is_refused_and_one_that_embeds_it_verified(tmp_path):
    source, target = workspace(tmp_path / "ws"), tmp_path / "target"
    wrapper = provenance.install(str(tmp_path / "root"))
    (source / "lib/data.txt").write_text("data")
    (source / "lib/src/part.rs").write_text(
        'pub fn who() -> String { std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/data.txt")).unwrap() }\n')
    cargo_build(source, target, wrapper)
    binary = str(target / "release" / "my-app")
    record = provenance.verify([binary], [str(source)], str(target))
    assert record["status"] == "refused"
    (executable,) = record["executables"]
    assert executable["status"] == "reads-source-tree"
    assert executable["reasons"] == ["my_lib: names its source tree through CARGO_MANIFEST_DIR: embed what it "
                                     "reads (include_bytes!, include_str!) or take its path at run time"]
    (source / "lib/src/part.rs").write_text('pub fn who() -> String { include_str!("../data.txt").to_owned() }\n')
    cargo_build(source, target, wrapper)
    assert provenance.verify([binary], [str(source)], str(target))["status"] == "verified"


@needs_cargo
def test_reused_unit_of_other_content_is_contradicted(tmp_path):
    source, target = workspace(tmp_path / "ws"), tmp_path / "target"
    wrapper = provenance.install(str(tmp_path / "root"))
    cargo_build(source, target, wrapper)
    replace_keeping_cargo_fresh(source / "lib/src/part.rs", SECOND)
    assert "Compiling" not in cargo_build(source, target, wrapper)  # cargo's freshness is fooled
    record = provenance.verify([str(target / "release" / "my-app")], [str(source)], str(target))
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
    record = provenance.verify([str(target / "release" / "my-app")], [str(source)], str(target))
    assert record["status"] == "refused"
    assert record["executables"][0]["status"] == "unverified"


@needs_cargo
def test_relinking_against_an_unrecorded_workspace_unit_is_unverified(tmp_path):
    source, target = workspace(tmp_path / "ws"), tmp_path / "target"
    wrapper = provenance.install(str(tmp_path / "root"))
    cargo_build(source, target, wrapper)
    for record in (target / "release" / "deps").glob("my_lib-*.provenance.json"):
        record.unlink()
    record = provenance.verify([str(target / "release" / "my-app")], [str(source)], str(target))
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
    work = engine.Work(sides=(engine.Side("source", source, (build,)),), jobs=(engine.Job("source", "app", job),),
                       device="device")
    request = engine.Request(instrument="t", purpose="p", estimate=600, backend="cpu", host="alpha",
                             work=lambda alias: work)
    outcome = engine.execute(request, engine.new_run_id("test"), hosts=hosts, announce=lambda line: None)
    assert outcome.codes == {"app": 0}
    assert "first generated" in (outcome.results / "logs" / "job-app.log").read_text()
    proof = json.loads((outcome.results / "provenance-source.json").read_text())
    assert proof["status"] == "verified" and proof["executables"][0]["units"] == 2

    # The same new bytes here and on the host, with a time older than the build outputs: the
    # mirror transfers nothing and cargo rebuilds nothing, so the old unit would be reused.
    host_copy = tmp_path / "alpha" / "checkouts" / "slot-0" / "src" / "lib/src/part.rs"
    for path in (source.directory / "lib/src/part.rs", host_copy):
        replace_keeping_cargo_fresh(path, SECOND)
    with pytest.raises(engine.RunFailed) as failed:
        engine.execute(request, engine.new_run_id("test"), hosts=hosts, announce=lambda line: None)
    assert failed.value.phase == "build-source" and "provenance" in str(failed.value)
    assert not (failed.value.outcome.results / "logs" / "job-app.log").exists()


@needs_cargo
def test_a_job_finds_only_executables_built_from_its_source(rig):
    """A job that reaches an executable only at run time (a script, through `$MRIG_TARGET`),
    after its source changed: with a build it runs the rebuilt executable; without one the old
    executable is withdrawn before the job, which finds none. It never runs the old one."""
    hosts, source, tmp_path = rig
    source = _with_workspace(source)
    build = engine.Step("cargo", ("cargo", "build", "--release", "--offline"))
    job = engine.Step("app", ("sh", "-c", '"$APP_TARGET/release/my-app"'), (("APP_TARGET", "{target}"),))

    def run(builds: tuple) -> engine.Outcome:
        work = engine.Work(sides=(engine.Side("source", source, builds),), jobs=(engine.Job("source", "app", job),),
                           device="device")
        request = engine.Request(instrument="t", purpose="p", estimate=600, backend="cpu", host="alpha",
                                 work=lambda alias: work)
        return engine.execute(request, engine.new_run_id("test"), hosts=hosts, announce=lambda line: None)

    def output(outcome: engine.Outcome) -> str:
        return (outcome.results / "logs" / "job-app.log").read_text()

    first = run((build,))
    assert first.codes == {"app": 0} and "first generated" in output(first)
    (source.directory / "lib/src/part.rs").write_text(SECOND)
    stale = run(())
    assert stale.codes["app"] != 0 and "first" not in output(stale)
    proof = json.loads((stale.results / "provenance-source.json").read_text())
    assert [Path(e["path"]).name for e in proof["withdrawn"]] == ["my-app"]
    rebuilt = run((build,))
    assert rebuilt.codes == {"app": 0} and "second generated" in output(rebuilt)


def test_every_build_step_compiles_through_the_checkouts_recording_wrapper(rig):
    hosts, source, tmp_path = rig
    show = engine.Step("show", (sys.executable, "-c", f"import os; print(os.environ['{engine.COMPILER}'])"))
    work = engine.Work(sides=(engine.Side("source", source, (show,)),))
    request = engine.Request(instrument="t", purpose="p", estimate=600, backend="cpu", host="alpha",
                             work=lambda alias: work)
    outcome = engine.execute(request, engine.new_run_id("test"), hosts=hosts, announce=lambda line: None)
    wrapper = tmp_path / "alpha" / "checkouts" / "slot-0" / provenance.WRAPPER
    assert f"{wrapper}\n" in (outcome.results / "logs" / "build-source.log").read_text()
    assert os.access(wrapper, os.X_OK)
