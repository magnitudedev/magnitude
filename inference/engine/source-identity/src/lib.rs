//! The build identity of a workspace package, derived in a build script from
//! exactly what can change what the package computes: the sources of every
//! workspace package in its dependency closure (normal and build dependencies
//! of every feature, for the target), the locked version of every other
//! package in it, the workspace manifest, and the compiler and the target it
//! compiles for. Packages that depend on it are outside the closure, so their
//! edits neither change the identity nor rerun the build script; nor do the
//! closure's tests, benchmarks and examples.
//!
//! The closure is read from the two records every build already holds: the
//! workspace members' manifests, through `cargo metadata --no-deps`, which
//! reads no registry, and the lockfile's resolve. No package outside the
//! workspace is opened, so a build that fetched only its own target's
//! dependencies, a vendored build and an offline build derive it alike. The
//! lockfile records a registry package's dependencies for every platform and
//! feature, so all of those enter; a workspace member's dependencies are taken
//! for the target only.
//!
//! Every feature's code is covered, since the manifests that declare features
//! are sources, but not which features a build enables: each feature in the
//! closure only adds development or test code, which changes nothing the rest
//! computes. The profile is left out for the same reason; Rust's arithmetic is
//! the same at every optimization level.

use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::Command;

/// `<version>+src.<digest>` of the closure of the workspace package `package`
/// (the calling one, or one it depends on), where the version is the calling
/// package's. Prints a `cargo:rerun-if-changed` for every input; the compiler
/// and the target are already part of every build script's run.
pub fn of_closure(package: &str) -> String {
    let target = Target::of_build();
    let workspace = Workspace::read(&target);
    let mut digest = Sha256::new();
    let mut files = Vec::new();
    for locked in workspace.closure(package) {
        match locked.source.as_deref() {
            None => {
                let directory = &workspace.members[&locked.name].directory;
                watch(directory);
                collect(directory, directory, &mut files);
            }
            // A registry or git package is fixed by its name, version and source.
            Some(source) => {
                for field in [&locked.name, &locked.version, source] {
                    digest.update(field.as_bytes());
                    digest.update([0]);
                }
            }
        }
    }
    // The lockfile fixes the versions; the workspace manifest the profiles and
    // the workspace dependencies' features.
    println!("cargo:rerun-if-changed={}", workspace.root.join("Cargo.lock").display());
    let manifest = workspace.root.join("Cargo.toml");
    println!("cargo:rerun-if-changed={}", manifest.display());
    files.push(manifest);
    files.sort();
    for file in files {
        // The path with '/' separators on every host, so Windows derives the same identity.
        let relative = file
            .strip_prefix(&workspace.root)
            .expect("a source lies under the workspace root")
            .components()
            .map(|component| component.as_os_str().to_string_lossy())
            .collect::<Vec<_>>()
            .join("/");
        digest.update(relative.as_bytes());
        digest.update([0]);
        digest.update(std::fs::read(&file).unwrap_or_else(|error| panic!("reading {}: {error}", file.display())));
        digest.update([0]);
    }
    digest.update(compiler().as_bytes());
    digest.update([0]);
    digest.update(target.triple.as_bytes());
    let hash = digest
        .finalize()
        .iter()
        .take(12)
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    format!("{}+src.{hash}", variable("CARGO_PKG_VERSION"))
}

fn variable(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("Cargo sets {name} for build scripts"))
}

/// The compiler's full version (`rustc -vV`): its release, commit and LLVM.
fn compiler() -> String {
    let output = Command::new(variable("RUSTC"))
        .arg("-vV")
        .output()
        .unwrap_or_else(|error| panic!("running rustc -vV: {error}"));
    assert!(output.status.success(), "rustc -vV failed: {}", String::from_utf8_lossy(&output.stderr));
    String::from_utf8(output.stdout).expect("rustc -vV prints UTF-8")
}

/// The target a build compiles for: its triple and its `cfg` values.
struct Target {
    triple: String,
    cfg: BTreeMap<String, BTreeSet<String>>,
}

impl Target {
    /// The build script's target, from the variables Cargo sets for it.
    fn of_build() -> Self {
        let cfg = std::env::vars()
            .filter_map(|(name, value)| {
                let key = name.strip_prefix("CARGO_CFG_")?.to_lowercase();
                let values = value.split(',').filter(|value| !value.is_empty()).map(str::to_owned).collect();
                Some((key, values))
            })
            .collect();
        Self { triple: variable("TARGET"), cfg }
    }

    /// Whether a dependency of platform `platform` (a manifest's
    /// `[target.<platform>]`: a triple or a `cfg(...)`) builds for this target.
    fn selects(&self, platform: &str) -> bool {
        match platform.strip_prefix("cfg(").and_then(|rest| rest.strip_suffix(')')) {
            Some(predicate) => {
                let mut parser = Predicate { target: self, rest: predicate };
                let holds = parser.evaluate();
                parser.skip_space();
                assert!(parser.rest.is_empty(), "a cfg predicate ends at its close: {platform}");
                holds
            }
            None => platform == self.triple,
        }
    }
}

/// A `cfg` predicate being evaluated: `all(..)`, `any(..)`, `not(..)`,
/// `name` or `name = "value"`.
struct Predicate<'a> {
    target: &'a Target,
    rest: &'a str,
}

impl<'a> Predicate<'a> {
    fn evaluate(&mut self) -> bool {
        let name = self.identifier();
        if self.eat('(') {
            let mut operands = Vec::new();
            while !self.eat(')') {
                operands.push(self.evaluate());
                if !self.eat(',') {
                    assert!(self.eat(')'), "cfg operands are separated by commas: {}", self.rest);
                    break;
                }
            }
            match name {
                "all" => operands.iter().all(|&holds| holds),
                "any" => operands.iter().any(|&holds| holds),
                "not" => {
                    assert_eq!(operands.len(), 1, "not() takes one predicate");
                    !operands[0]
                }
                _ => panic!("unknown cfg operator {name}"),
            }
        } else if self.eat('=') {
            let value = self.string();
            self.target.cfg.get(name).is_some_and(|values| values.contains(value))
        } else {
            self.target.cfg.contains_key(name)
        }
    }

    fn skip_space(&mut self) {
        self.rest = self.rest.trim_start();
    }

    fn eat(&mut self, token: char) -> bool {
        self.skip_space();
        match self.rest.strip_prefix(token) {
            Some(rest) => {
                self.rest = rest;
                true
            }
            None => false,
        }
    }

    fn identifier(&mut self) -> &'a str {
        self.skip_space();
        let end = self.rest.find(|c: char| !(c.is_ascii_alphanumeric() || c == '_')).unwrap_or(self.rest.len());
        assert!(end > 0, "a cfg predicate names a cfg: {}", self.rest);
        let (name, rest) = self.rest.split_at(end);
        self.rest = rest;
        name
    }

    fn string(&mut self) -> &'a str {
        assert!(self.eat('"'), "a cfg value is a string: {}", self.rest);
        let (value, rest) = self.rest.split_once('"').expect("a cfg string closes");
        self.rest = rest;
        value
    }
}

/// The workspace as its manifests and lockfile record it.
struct Workspace {
    root: PathBuf,
    members: BTreeMap<String, Member>,
    /// The lockfile's packages, in its order.
    locked: Vec<Locked>,
}

struct Member {
    directory: PathBuf,
    /// The packages it depends on through normal or build dependencies for
    /// the target; development dependencies build no part of it.
    builds: BTreeSet<String>,
}

/// A lockfile `[[package]]`.
#[derive(Default)]
struct Locked {
    name: String,
    version: String,
    /// None for a workspace member.
    source: Option<String>,
    /// `name`, `name version` or `name version (source)`.
    dependencies: Vec<String>,
}

impl Workspace {
    /// The workspace of the package Cargo is building (or testing), for `target`.
    fn read(target: &Target) -> Self {
        let manifest = PathBuf::from(variable("CARGO_MANIFEST_DIR")).join("Cargo.toml");
        let output = Command::new(variable("CARGO"))
            .args(["metadata", "--format-version", "1", "--no-deps", "--offline", "--manifest-path"])
            .arg(&manifest)
            .output()
            .unwrap_or_else(|error| panic!("running cargo metadata: {error}"));
        assert!(output.status.success(), "cargo metadata failed: {}", String::from_utf8_lossy(&output.stderr));
        let metadata: Value = serde_json::from_slice(&output.stdout).expect("cargo metadata prints JSON");
        let members = list(&metadata["packages"])
            .iter()
            .map(|package| {
                let directory = Path::new(text(&package["manifest_path"]))
                    .parent()
                    .expect("a manifest lies in its package directory")
                    .to_owned();
                let builds = list(&package["dependencies"])
                    .iter()
                    .filter(|dependency| dependency["kind"].as_str() != Some("dev"))
                    .filter(|dependency| dependency["target"].as_str().is_none_or(|platform| target.selects(platform)))
                    .map(|dependency| text(&dependency["name"]).to_owned())
                    .collect();
                (text(&package["name"]).to_owned(), Member { directory, builds })
            })
            .collect();
        let root = PathBuf::from(text(&metadata["workspace_root"]));
        let locked = read_lock(&root.join("Cargo.lock"));
        Self { root, members, locked }
    }

    /// `package` and every package it builds from, in lockfile order.
    fn closure(&self, package: &str) -> Vec<&Locked> {
        let mut by_name: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
        for (index, locked) in self.locked.iter().enumerate() {
            by_name.entry(locked.name.as_str()).or_default().push(index);
        }
        let find = |reference: &str| -> usize {
            let mut parts = reference.splitn(3, ' ');
            let name = parts.next().expect("a reference names a package");
            let version = parts.next();
            let source = parts.next().map(|source| {
                source
                    .strip_prefix('(')
                    .and_then(|source| source.strip_suffix(')'))
                    .unwrap_or_else(|| panic!("a lockfile source is parenthesized: {reference}"))
            });
            let candidates = by_name.get(name).map(Vec::as_slice).unwrap_or_default();
            let matching = candidates
                .iter()
                .copied()
                .filter(|&index| {
                    let locked = &self.locked[index];
                    version.is_none_or(|version| locked.version == version)
                        && source.is_none_or(|source| locked.source.as_deref() == Some(source))
                })
                .collect::<Vec<_>>();
            assert_eq!(matching.len(), 1, "the lockfile names one package {reference}");
            matching[0]
        };
        let root = find(package);
        assert!(self.locked[root].source.is_none(), "{package} is a workspace member");
        let mut reached = BTreeSet::new();
        let mut pending = vec![root];
        while let Some(index) = pending.pop() {
            if !reached.insert(index) {
                continue;
            }
            let locked = &self.locked[index];
            let member = locked.source.is_none().then(|| {
                self.members
                    .get(&locked.name)
                    .unwrap_or_else(|| panic!("the lockfile's path package {} is a workspace member", locked.name))
            });
            for reference in &locked.dependencies {
                let name = reference.split(' ').next().expect("a reference names a package");
                if member.is_none_or(|member| member.builds.contains(name)) {
                    pending.push(find(reference));
                }
            }
        }
        reached.into_iter().map(|index| &self.locked[index]).collect()
    }
}

/// The packages of a lockfile, in the format Cargo writes (version 4).
fn read_lock(path: &Path) -> Vec<Locked> {
    let contents = std::fs::read_to_string(path).unwrap_or_else(|error| panic!("reading {}: {error}", path.display()));
    let string = |value: &str| -> String {
        let inner = value.strip_prefix('"').and_then(|value| value.strip_suffix('"'));
        match inner {
            Some(inner) if !inner.contains(['"', '\\']) => inner.to_owned(),
            _ => panic!("a lockfile value is a plain string: {value}"),
        }
    };
    let mut packages: Vec<Locked> = Vec::new();
    let mut lines = contents.lines();
    while let Some(line) = lines.next() {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if line == "[[package]]" {
            packages.push(Locked::default());
            continue;
        }
        let (key, value) = line
            .split_once(" = ")
            .unwrap_or_else(|| panic!("a lockfile line is a package or a key: {line}"));
        let Some(package) = packages.last_mut() else {
            assert_eq!((key, value), ("version", "4"), "the lockfile is version 4");
            continue;
        };
        match key {
            "name" => package.name = string(value),
            "version" => package.version = string(value),
            "source" => package.source = Some(string(value)),
            "checksum" => {}
            "dependencies" => {
                assert_eq!(value, "[", "lockfile dependencies are listed one per line");
                for line in lines.by_ref().take_while(|line| *line != "]") {
                    let entry = line.trim().strip_suffix(',').unwrap_or_else(|| panic!("a dependency line ends with a comma: {line}"));
                    package.dependencies.push(string(entry));
                }
            }
            _ => panic!("unknown lockfile key {key}"),
        }
    }
    packages
}

/// Whether `path` is a build source: what a crate compiles, includes or
/// generates code from. Hidden files (editor state, Finder metadata) are not.
fn is_build_source(path: &Path) -> bool {
    let hidden = path.file_name().and_then(|name| name.to_str()).is_none_or(|name| name.starts_with('.'));
    !hidden
        && matches!(
            path.extension().and_then(|extension| extension.to_str()),
            Some(
                "rs" | "toml" | "seismic" | "metal" | "cu" | "cuh" | "glsl" | "comp" | "h" | "hpp" | "c"
                    | "cc" | "cpp"
            )
        )
}

/// Cargo's directories of development-only targets, which build no part of
/// the package's library.
const DEVELOPMENT_TARGETS: [&str; 3] = ["tests", "benches", "examples"];

/// Whether the walk leaves out the directory `path` of `package`: a hidden
/// directory, build output, the package's development-only target
/// directories, and another package nested in this one's directory (its own
/// closure membership decides whether it counts).
fn excluded(package: &Path, path: &Path) -> bool {
    let name = path.file_name().and_then(|name| name.to_str()).unwrap_or("");
    name.starts_with('.')
        || name == "target"
        || (path.parent() == Some(package) && DEVELOPMENT_TARGETS.contains(&name))
        || (path != package && path.join("Cargo.toml").is_file())
}

/// The entries of `directory`, sorted, split into subdirectories and files.
fn entries(directory: &Path) -> (Vec<PathBuf>, Vec<PathBuf>) {
    let mut entries = std::fs::read_dir(directory)
        .unwrap_or_else(|error| panic!("reading {}: {error}", directory.display()))
        .map(|entry| entry.expect("a package directory entry").path())
        .collect::<Vec<_>>();
    entries.sort();
    entries.into_iter().partition(|path| path.is_dir())
}

fn collect(package: &Path, directory: &Path, files: &mut Vec<PathBuf>) {
    let (directories, contents) = entries(directory);
    files.extend(contents.into_iter().filter(|path| is_build_source(path)));
    for path in directories {
        if !excluded(package, &path) {
            collect(package, &path, files);
        }
    }
}

/// Watches `package`'s directory: a subtree containing no directory the walk
/// leaves out is watched whole, so Cargo sees files added to it too;
/// otherwise its build sources one by one and its subdirectories in turn, so
/// no nested package, development target or build output reruns the script.
fn watch(package: &Path) {
    fn clean(package: &Path, directory: &Path) -> bool {
        entries(directory).0.iter().all(|path| !excluded(package, path) && clean(package, path))
    }
    fn visit(package: &Path, directory: &Path) {
        if clean(package, directory) {
            println!("cargo:rerun-if-changed={}", directory.display());
            return;
        }
        let (directories, files) = entries(directory);
        for path in files.iter().filter(|path| is_build_source(path)) {
            println!("cargo:rerun-if-changed={}", path.display());
        }
        for path in directories {
            if !excluded(package, &path) {
                visit(package, &path);
            }
        }
    }
    visit(package, package);
}

fn text(value: &Value) -> &str {
    value.as_str().unwrap_or_else(|| panic!("cargo metadata gives a string, not {value}"))
}

fn list(value: &Value) -> &Vec<Value> {
    value.as_array().unwrap_or_else(|| panic!("cargo metadata gives an array, not {value}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn target(triple: &str, cfg: &[(&str, &[&str])]) -> Target {
        Target {
            triple: triple.to_owned(),
            cfg: cfg
                .iter()
                .map(|(key, values)| (key.to_string(), values.iter().map(|value| value.to_string()).collect()))
                .collect(),
        }
    }

    fn linux() -> Target {
        target(
            "x86_64-unknown-linux-gnu",
            &[
                ("unix", &[]),
                ("target_family", &["unix"]),
                ("target_os", &["linux"]),
                ("target_arch", &["x86_64"]),
                ("target_env", &["gnu"]),
            ],
        )
    }

    fn macos() -> Target {
        target(
            "aarch64-apple-darwin",
            &[("unix", &[]), ("target_family", &["unix"]), ("target_os", &["macos"]), ("target_arch", &["aarch64"])],
        )
    }

    #[test]
    fn platforms_select_as_cargo_evaluates_them() {
        let linux = linux();
        assert!(linux.selects("x86_64-unknown-linux-gnu"));
        assert!(!linux.selects("aarch64-apple-darwin"));
        assert!(linux.selects("cfg(unix)"));
        assert!(!linux.selects("cfg(windows)"));
        assert!(linux.selects(r#"cfg(all(target_os = "linux", target_env = "gnu"))"#));
        assert!(!linux.selects(r#"cfg(all(target_os = "linux", target_arch = "aarch64"))"#));
        assert!(linux.selects(r#"cfg(any(windows, target_family = "unix"))"#));
        assert!(linux.selects(r#"cfg(not(target_os = "macos"))"#));
        assert!(!macos().selects(r#"cfg(not(target_os = "macos"))"#));
    }

    /// The workspace's own manifests and lockfile, read without a registry.
    #[test]
    fn closure_holds_what_builds_the_package_for_its_target() {
        let names = |target: &Target, package: &str| -> BTreeSet<String> {
            Workspace::read(target).closure(package).into_iter().map(|locked| locked.name.clone()).collect()
        };
        let executor = names(&linux(), "magnitude-executor");
        for built in ["magnitude-executor", "magnitude-kernels", "seismic", "seismic-std", "seismic-vulkan", "magnitude-ggml"] {
            assert!(executor.contains(built), "{built} builds the executor");
        }
        // Above it, or only its tests'.
        for apart in ["magnitude-engine", "magnitude-chat", "magnitude-family-qwen35", "seismic-native-tests"] {
            assert!(!executor.contains(apart), "{apart} builds no part of the executor");
        }
        // Seismic discovery opens Vulkan off macOS only.
        assert!(names(&linux(), "seismic-discovery").contains("seismic-vulkan"));
        assert!(!names(&macos(), "seismic-discovery").contains("seismic-vulkan"));
        assert!(names(&linux(), "magnitude-engine").contains("magnitude-chat"));
    }
}
