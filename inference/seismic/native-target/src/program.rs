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

/// Bytes a toolchain keeps between processes for one program source.
pub trait ProgramCache {
    /// The kept bytes, or `None` on a miss or any failure.
    fn get(&self) -> Option<Vec<u8>>;
    /// Keep `bytes`, replacing what was kept. Failures are the cache's to
    /// report; compiling never fails for them.
    fn put(&self, bytes: &[u8]);
}

/// A backend's compiler for one opened device.
pub trait Toolchain: Send + Sync {
    /// The executable program: one handle per requested entry, in order.
    type Program: Send + Sync + 'static;

    fn identity(&self) -> &ToolchainIdentity;

    /// The program of `source`'s entries. `cache`, when given, holds what
    /// this toolchain kept for the same source text; a toolchain that keeps
    /// nothing ignores it.
    fn compile(
        &self,
        source: &ProgramSource,
        cache: Option<&dyn ProgramCache>,
    ) -> Result<Self::Program, NativeCompilationError>;
}
