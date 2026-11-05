//! Programs: rendered native source compiled by a backend's toolchain.
//! Backends that render source (Metal, CUDA, Vulkan) have a toolchain; the
//! CPU backend compiles its kernels into the binary or JIT-compiles IR, and
//! renders no source.
//!
//! A caller renders a [`ProgramSource`] in its backend's language and names
//! the entries it needs from it. Variants of a templated kernel share one
//! source text and differ only in their entry (a template instance), so one
//! compile serves every variant it names. Whether and what a toolchain keeps
//! between processes is its own decision: the runtime hands it a cache for
//! the program's key and never interprets the bytes.

use crate::NativeCompilationError;

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

/// What besides the source determines a toolchain's output: its compiler,
/// target and options, and the device and system facts they consume.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ToolchainIdentity {
    /// Where the toolchain's cached bytes are kept, apart from every other
    /// toolchain's.
    pub namespace: &'static str,
    pub material: String,
}

/// How a backend's stored artifact serves a later process.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Load {
    /// The artifact is everything that serves the program on this device
    /// and driver: loading it compiles nothing, and a load that would have
    /// to compile fails.
    Complete,
    /// The artifact names the program, and loading it forms the program
    /// again through the platform's own cache of what it compiled (Metal's
    /// shader cache). What loading takes is the platform's: the former
    /// counts these loads and the time they take, which is what shows
    /// whether the platform still serves them.
    PlatformCached,
}

/// What a toolchain compiles and the store keeps, as bytes.
pub trait Artifact: Sized + Send + Sync {
    /// The version of [`Artifact::encode`]'s format, part of every key the
    /// artifact is stored under: a changed encoding takes a new version, so
    /// the bytes kept under a key always decode.
    const ENCODING: u32;
    fn encode(&self) -> Vec<u8>;
    /// The artifact `bytes` encode; none when they do not.
    fn decode(bytes: &[u8]) -> Option<Self>;
}

/// A backend's compiler for one opened device. A program is had in one of
/// two ways: [`compile`] turns source into an artifact and the program it
/// formed, and [`load`] makes the program of an artifact.
///
/// The runtime's program former is this trait's one caller outside tests,
/// and nothing else forms the programs a process serves. It keys every artifact by the
/// source and the toolchain's identity ([`ToolchainIdentity`]) and stores
/// it, so whenever no live compile holds a program, in this process or a
/// later one, it is loaded ([`Load`] says what that takes). What compiles
/// outside it serves nothing: the device probes that toolchain identity is
/// made of, which precede any toolchain; the device-property probes, which
/// the measurement job runs in probe processes of their own; the
/// analytical-service probes, whose measurements feed the estimator;
/// offline coverage of whether a
/// program forms; and Seismic's compiler of entries without a native
/// implementation, which no load runs.
///
/// [`compile`]: Toolchain::compile
/// [`load`]: Toolchain::load
pub trait Toolchain: Send + Sync {
    type Artifact: Artifact;
    /// The executable program: one handle per requested entry, in order.
    type Program: Send + Sync + 'static;
    const LOAD: Load;

    /// The compiler, target, options, device and driver: what keys every
    /// artifact beside its source.
    fn identity(&self) -> &ToolchainIdentity;

    /// The artifact of `source`'s entries, and the program the compile
    /// formed of them, which the caller uses instead of loading it again
    /// from the artifact.
    fn compile(
        &self,
        source: &ProgramSource,
    ) -> Result<(Self::Artifact, Self::Program), NativeCompilationError>;

    /// The program of `entries` from `artifact`, compiled of a source
    /// holding them, when no live compile holds them. For a
    /// [`Load::Complete`] toolchain it compiles nothing: what the artifact
    /// does not serve without compiling fails.
    fn load(
        &self,
        artifact: &Self::Artifact,
        entries: &[ProgramEntry],
    ) -> Result<Self::Program, NativeCompilationError>;

    /// What the backend's compiler reports `program`'s entry `entry` uses
    /// ([`ProgramStatistic`]), read from the driver when the program was
    /// formed; none for a backend whose compiler reports nothing.
    fn statistics(program: &Self::Program, entry: usize) -> &[ProgramStatistic];
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
