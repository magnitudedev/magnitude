//! The formations a search needs, and forming one.
//!
//! The search decides what to form and reads only the [`Verdict`]; whether a
//! formation runs or is recalled from an earlier run is the caller's
//! [`Forming`]. A [`Formation`] carries what identifies it, so a caller may
//! key what it establishes by its inputs.

use seismic_native_target::{NativeCompilationError, ProgramSource};

/// What forms a program: the toolchain and the facts it reads.
#[derive(Clone, Copy, Debug)]
pub enum Target {
    Vulkan(seismic_vulkan::FormationFacts),
    Cuda {
        architecture: u32,
    },
    #[cfg(target_os = "macos")]
    Metal {
        language: seismic_metal::facts::LanguageVersion,
    },
}

/// One program to form under one target.
pub struct Formation<'a> {
    /// The toolchain's namespace.
    pub namespace: &'static str,
    /// Everything besides the program that determines the formation: the
    /// compiler's identity and the facts it reads.
    pub material: String,
    pub target: Target,
    pub program: &'a ProgramSource,
}

/// What forming established.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Verdict {
    Formed,
    /// A `static_assert` fired, which is how C++ kernels decline bindings
    /// the preprocessor cannot see.
    Rejected,
    Failed(String),
}

/// How a search's formations run.
pub trait Forming: Sync {
    fn form(&self, formation: Formation<'_>) -> Verdict;
}

/// How far a run has come: the units of work it has, and how many are done.
/// Coverage counts entries searched; request formation counts requests.
#[derive(Default)]
pub struct Progress {
    total: std::sync::atomic::AtomicUsize,
    done: std::sync::atomic::AtomicUsize,
}

impl Progress {
    pub fn total(&self) -> usize {
        self.total.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn done(&self) -> usize {
        self.done.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub(super) fn begin(&self, total: usize) {
        self.done.store(0, std::sync::atomic::Ordering::Relaxed);
        self.total.store(total, std::sync::atomic::Ordering::Relaxed);
    }

    pub(super) fn advance(&self) {
        self.done.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Every formation in this process, as it is asked for.
pub struct InProcess;

impl Forming for InProcess {
    fn form(&self, formation: Formation<'_>) -> Verdict {
        form(formation.target, formation.program)
    }
}

/// Form `program` with `target` in this process, as preparation forms it,
/// short of loading it on a device.
pub fn form(target: Target, program: &ProgramSource) -> Verdict {
    match formed(target, program) {
        Ok(()) => Verdict::Formed,
        Err(error) if asserted(&error) => Verdict::Rejected,
        Err(error) => Verdict::Failed(compilation_message(&error)),
    }
}

fn formed(target: Target, program: &ProgramSource) -> Result<(), NativeCompilationError> {
    match target {
        Target::Vulkan(facts) => {
            for entry in &program.entries {
                seismic_vulkan::formation::compile(
                    &program.text,
                    &entry.symbol,
                    facts.environment(),
                )?;
            }
            Ok(())
        }
        Target::Cuda { architecture } => {
            seismic_cuda::toolchain::form_cubin(program, architecture).map(|_| ())
        }
        #[cfg(target_os = "macos")]
        Target::Metal { language } => {
            let device = seismic_metal::DeviceHandle::system_default()
                .map_err(|error| NativeCompilationError::ToolchainFailure(format!("{error:?}")))?;
            seismic_metal::toolchain::form_functions(&device, program, language).map(|_| ())
        }
    }
}

/// Whether a compilation failed because a `static_assert` fired.
fn asserted(error: &NativeCompilationError) -> bool {
    let text = format!("{error:?}");
    text.contains("static assertion failed") || text.contains("static_assert failed")
}

fn compilation_message(error: &NativeCompilationError) -> String {
    let text = format!("{error:?}");
    text.lines()
        .filter(|line| line.contains("ERROR") || line.contains("error"))
        .take(4)
        .collect::<Vec<_>>()
        .join(" | ")
}
