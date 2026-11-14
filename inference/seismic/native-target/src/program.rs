//! Programs: rendered native source compiled by a backend's toolchain.
//! Metal, CUDA and Vulkan render their native entries and Seismic-IR kernels
//! as source in their languages (CUDA's IR kernels as PTX, a toolchain of
//! their own); the CPU's native entries are compiled into the binary, and
//! its toolchain forms Seismic-IR kernels from their CLIF text.
//!
//! A caller renders a [`ProgramSource`] in its backend's language and names
//! the entries it needs from it. Variants of a templated kernel share one
//! source text and differ only in their entry (a template instance), so one
//! compile serves every variant it names. A toolchain makes each compile's
//! artifact and loads one; the runtime's program former keys, stores and
//! reads them, and never interprets the bytes.

use crate::{ArtifactKey, NativeCompilationError};
use std::sync::Arc;

/// Rendered source in a backend's language and the entries a caller needs
/// from it, in order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProgramSource {
    pub text: String,
    pub entries: Vec<ProgramEntry>,
}

/// One entry of a program: its symbol (a kernel name, or a template
/// instance such as `kernel<16, 2>`), the group size its launches run when the
/// caller fixes it (Vulkan forms its pipeline with it; Metal forms a pipeline
/// that admits it), and the backend's specialization constants (Vulkan's
/// view lengths; empty elsewhere).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProgramEntry {
    pub symbol: String,
    pub group_size: Option<[u32; 3]>,
    pub constants: Vec<u32>,
}

impl ProgramEntry {
    pub fn named(symbol: impl Into<String>) -> Self {
        Self {
            symbol: symbol.into(),
            group_size: None,
            constants: Vec::new(),
        }
    }
}

/// What besides the source determines a toolchain's output.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ToolchainIdentity {
    /// Where the toolchain's cached bytes are kept, apart from every other
    /// toolchain's.
    pub namespace: &'static str,
    /// What decides how the programs it forms measure: its compiler, target
    /// and options, and the device and system facts they consume. It keys
    /// stored artifacts and tuning alike.
    pub material: String,
    /// What keys a stored artifact beyond the material, and nothing else:
    /// the build of the backend's formation code, so no artifact is served
    /// to a build whose formation code differs, and what else the artifact
    /// is code for that the tuning identity already implies (Metal's GPU
    /// architecture, which its device does). Tuning is never keyed by it,
    /// so a rebuild compiles again and keeps every tuning record.
    pub stored: String,
}

/// What a toolchain loads a program from: what the store keeps, as the
/// bytes [`Toolchain::artifact`] made.
pub trait Artifact: Sized + Send + Sync {
    /// The version of the bytes' format, part of every key the artifact is
    /// stored under: a changed encoding takes a new version, so the bytes
    /// kept under a key always decode.
    const ENCODING: u32;
    /// Whether its platform opens it by path (Metal's archives), so a load
    /// gives [`Artifact::decode`] the store's file of it, to open in place.
    const IN_PLACE: bool = false;
    /// The artifact `bytes` encode, which the store also keeps in `file`
    /// where the artifact opens in place ([`Artifact::IN_PLACE`]) and the
    /// store keeps them in one. None when they do not decode.
    fn decode(bytes: &[u8], file: Option<&std::path::Path>) -> Option<Self>;
}

/// A backend's compiler for one opened device. A program is had in one of
/// two ways: [`compile`] turns source into an artifact and the program it
/// formed, and [`load`] makes the program of an artifact.
///
/// The runtime's program former forms every program a process serves
/// through it: native entries, Seismic-IR kernels, graph-cost kernels and
/// analytical-service probes ([`Programs`]). It keys every artifact by the
/// source and the toolchain's identity ([`ToolchainIdentity`]) and stores
/// it, so whenever no live compile holds a program, in this process or a
/// later one, it is loaded, compiling nothing.
///
/// Every formation outside the store, with why, and nothing else:
///
/// Tests of a compiler or driver itself, below the runtime, which has the
/// store (the backend crates cannot reach it without inverting the
/// layering, and each tests the thing the store would wrap):
/// - CUDA's copy-engine tests' spin and sum kernels (`transfer.rs`), its
///   driver tests of frame and stack limits (`driver.rs`), its
///   abandoned-launch test (`direct.rs`), and its graph-cost unit tests of
///   the driver's graph accounting (`graph_cost.rs`).
/// - Metal's executor, device and barrier tests of the driver
///   (`executor.rs`, `device.rs`, `profile_tests.rs`, `tests/barrier.rs`).
/// - The toolchains' own tests, of the compile and load the former calls:
///   Metal's `tests/memory.rs`, `tests/timestamp_pool.rs` and
///   `toolchain.rs`, Vulkan's `toolchain.rs`.
/// - Vulkan's seal-pass test (`lib.rs`): GLSL to SPIR-V only, no device.
/// - The backends' test [`Programs`] (Metal's `test_support.rs` `Compiled`,
///   CUDA's `test_ptx_programs`, the CPU's `toolchain.rs` `test_programs`):
///   each backend crate's own tests form IR kernels through its toolchain
///   directly, because the runtime's former is above them. The CPU one
///   compiles, encodes, decodes and loads every source, so the crate's
///   tests run loaded code.
///
/// Tests in the runtime of the compiler's output, not of a device's
/// formation:
/// - `native/abi.rs`: GLSL compiled to SPIR-V to inspect it; no device.
///
/// Formations whose answer the store keeps, so each runs once per answer
/// key by construction:
/// - Metal's capability probes: run only on a miss of their stored answers
///   (`MetalProbeAnswers`). They precede the toolchain, whose material their
///   answers determine, so they cannot form through it.
/// - Vulkan's measurement-job probes: run only by the measurement job, in a
///   probe process of their own (a probe may crash the driver's compiler),
///   for an answer the store lacks.
///
/// Offline tooling that serves nothing:
/// - The runtime's coverage of whether a program forms
///   (`native/coverage/forming.rs`); its caller keeps the verdicts.
/// - `seismic/hardware/vulkan-facts`: a standalone fact-collection binary.
///
/// [`compile`]: Toolchain::compile
/// [`load`]: Toolchain::load
pub trait Toolchain: Send + Sync + 'static {
    type Artifact: Artifact;
    /// The executable program: one handle per requested entry, in order.
    type Program: Send + Sync + 'static;
    /// What a compile leaves its artifact to be made of
    /// ([`Toolchain::artifact`]), off the path that forms the program.
    type Pending: Send + 'static;

    /// The compiler, target, options, device and driver, and the build of
    /// the formation code: what keys every artifact beside its source.
    fn identity(&self) -> &ToolchainIdentity;

    /// The program of `source`'s entries, which the caller uses at once,
    /// and what their artifact is made of ([`Toolchain::artifact`]).
    fn compile(
        &self,
        source: &ProgramSource,
    ) -> Result<(Self::Pending, Self::Program), NativeCompilationError>;

    /// The bytes of the artifact of a compile that left `pending`
    /// ([`Artifact::decode`]): everything that serves its program on this
    /// device and driver, so a load of it compiles nothing. It runs off the
    /// path that formed the program, on another thread; an error stores
    /// nothing, and a later process compiles the program again.
    fn artifact(&self, pending: Self::Pending) -> Result<Vec<u8>, NativeCompilationError>;

    /// The program of `entries` from `artifact`, compiled of a source
    /// holding them, when no live compile holds them. The artifact is
    /// everything that serves the program on this device and driver, so a
    /// load compiles nothing. A load that does not serve the program, for
    /// any reason, is [`Declined`]: the platform would have to compile
    /// (Vulkan's pipeline cache is a hint, never a promise), the driver
    /// refuses the image (a CUBIN after a driver change), or the artifact
    /// lacks a requested entry. A decline is a miss, never a failure: the
    /// former compiles the entries and stores them again, as it does an
    /// artifact the store dropped, and a failure of the device itself is
    /// then the compile's.
    fn load(&self, artifact: &Self::Artifact, entries: &[ProgramEntry]) -> Result<Self::Program, Declined>;

    /// What the backend's compiler reports `program`'s entry `entry` uses
    /// ([`ProgramStatistic`]), read from the driver when the program was
    /// formed; none for a backend whose compiler reports nothing.
    fn statistics(program: &Self::Program, entry: usize) -> &[ProgramStatistic];
}

/// Why a stored artifact did not serve a load ([`Toolchain::load`]): a
/// miss, which the former reports, counts and compiles again. Any failure
/// met while loading is one: the compile it causes meets it again if it is
/// the device's, and fails with it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Declined(pub String);

impl From<NativeCompilationError> for Declined {
    fn from(error: NativeCompilationError) -> Self {
        Self(error.to_string())
    }
}

impl std::fmt::Display for Declined {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// What one launch's formed program uses: the threads of its workgroup,
/// where formation fixes them, and the statistics its backend's compiler
/// reports of it.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct LaunchProgram {
    pub threads: Option<u64>,
    pub statistics: Vec<ProgramStatistic>,
}

/// One statistic a backend's compiler reports of a formed program, named
/// and described as the backend reports it: CUDA's function attributes,
/// Vulkan's pipeline executable statistics. Both backends read them from
/// the driver of every program they form, compiled or loaded, so no
/// artifact carries them.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ProgramStatistic {
    pub name: String,
    pub description: String,
    pub value: StatisticValue,
}

/// A reported statistic's value, in the type it is reported in.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum StatisticValue {
    Bool(bool),
    Int(i64),
    Uint(u64),
    Float(f64),
}

/// One compile of a source text: the executable program and the entries it
/// holds, in the program's order. A [`Programs`] makes one of each compile
/// or load it forms; a holder of some of a compile's entries may make one of
/// those entries alone, out of a compile it was given, which is no compile
/// of the device's live set.
pub struct Compile<P> {
    entries: Vec<ProgramEntry>,
    program: P,
}

impl<P> Compile<P> {
    /// The compile that holds `entries` in `program`.
    pub fn new(entries: Vec<ProgramEntry>, program: P) -> Self {
        Self { entries, program }
    }

    pub fn program(&self) -> &P {
        &self.program
    }

    pub fn entries(&self) -> &[ProgramEntry] {
        &self.entries
    }
}

/// A formed source: the key of its text and the live compiles that
/// together hold every entry requested of it.
pub struct Formed<P> {
    key: ArtifactKey,
    compiles: Vec<Arc<Compile<P>>>,
}

impl<P> Formed<P> {
    /// The source keyed `key`, its entries held by `compiles`.
    pub fn new(key: ArtifactKey, compiles: Vec<Arc<Compile<P>>>) -> Self {
        Self { key, compiles }
    }

    pub fn key(&self) -> &ArtifactKey {
        &self.key
    }

    /// The compile that holds `entry` and the entry's position in it.
    pub fn entry(&self, entry: &ProgramEntry) -> (&Arc<Compile<P>>, usize) {
        self.compiles
            .iter()
            .find_map(|compile| {
                let position = compile.entries.iter().position(|held| held == entry)?;
                Some((compile, position))
            })
            .expect("a formed source holds every entry it was requested for")
    }

    /// The program of `entry` and its position in it ([`Formed::entry`]).
    pub fn program(&self, entry: &ProgramEntry) -> (&P, usize) {
        let (compile, position) = self.entry(entry);
        (compile.program(), position)
    }
}

/// The programs of one opened device under toolchain `T`: the one way code
/// of a backend forms a program. The runtime's program former implements it
/// over the process's program store, so a program any process of the user
/// formed is loaded, and one formed in this process is held or waited for,
/// never formed twice ([`Toolchain`]). Formed sources come back in order.
pub trait Programs<T: Toolchain>: Send + Sync {
    fn form(&self, sources: &[ProgramSource]) -> Result<Vec<Formed<T::Program>>, NativeCompilationError>;
}
