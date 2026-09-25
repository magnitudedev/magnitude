//! NVRTC: the bundled runtime compiler that forms authored CUDA C++ native
//! kernels for the opened device's exact architecture.
//!
//! NVRTC is an owned native dependency of the CUDA backend. It is loaded from
//! exactly one directory:
//!
//! - an installation's native library directory: `lib/` beside the executable
//!   on Linux, the executable's own directory on Windows; or
//! - the directory named by [`DIRECTORY_SETTING`], an explicit development
//!   setting (for example a CUDA toolkit's library directory) that replaces the
//!   installation directory.
//!
//! The directory holds `libnvrtc.so.<major>` and the builtins library of the
//! same release (`libnvrtc-builtins.so.<major>.<minor>`). NVRTC opens its
//! builtins by file name when it compiles; the loader binds the builtins from
//! the same directory first, so NVRTC never resolves them through the
//! dynamic loader's ambient search.

use libloading::Library;
use seismic_native_target::ToolchainUnavailable;
use std::ffi::{c_char, c_int, CStr, CString};
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

type NvrtcResult = c_int;
type Program = *mut std::ffi::c_void;

/// The NVRTC major version the backend bundles; it sets the driver
/// requirement (CUDA minor-version compatibility within one major).
pub const BUNDLED_MAJOR: u32 = 13;

/// Environment setting naming the directory that holds NVRTC and its
/// builtins, in place of the installation's library directory. It is a
/// development setting, not part of the installed runtime contract.
pub const DIRECTORY_SETTING: &str = "SEISMIC_NVRTC_DIRECTORY";

#[cfg(target_os = "windows")]
const LIBRARY: &str = "nvrtc64_130_0.dll";
#[cfg(not(target_os = "windows"))]
const LIBRARY: &str = "libnvrtc.so.13";

/// The builtins library NVRTC `major.minor` opens by name.
fn builtins_library(major: u32, minor: u32) -> String {
    if cfg!(target_os = "windows") {
        format!("nvrtc-builtins64_{major}{minor}.dll")
    } else {
        format!("libnvrtc-builtins.so.{major}.{minor}")
    }
}

struct Nvrtc {
    error_string: unsafe extern "C" fn(NvrtcResult) -> *const c_char,
    create: unsafe extern "C" fn(
        *mut Program,
        *const c_char,
        *const c_char,
        c_int,
        *const *const c_char,
        *const *const c_char,
    ) -> NvrtcResult,
    destroy: unsafe extern "C" fn(*mut Program) -> NvrtcResult,
    compile: unsafe extern "C" fn(Program, c_int, *const *const c_char) -> NvrtcResult,
    log_size: unsafe extern "C" fn(Program, *mut usize) -> NvrtcResult,
    log: unsafe extern "C" fn(Program, *mut c_char) -> NvrtcResult,
    cubin_size: unsafe extern "C" fn(Program, *mut usize) -> NvrtcResult,
    cubin: unsafe extern "C" fn(Program, *mut c_char) -> NvrtcResult,
    arch_count: unsafe extern "C" fn(*mut c_int) -> NvrtcResult,
    archs: unsafe extern "C" fn(*mut c_int) -> NvrtcResult,
    /// `(major, minor)` as reported by the loaded library.
    release: (u32, u32),
    /// Keeps the builtins bound for NVRTC's by-name open.
    _builtins: Library,
    _library: Library,
}

// Function pointers into libraries kept alive by the same value. NVRTC
// programs are created and destroyed per compilation; the API is thread-safe
// for distinct programs.
unsafe impl Send for Nvrtc {}
unsafe impl Sync for Nvrtc {}

static NVRTC: OnceLock<Result<Arc<Nvrtc>, ToolchainUnavailable>> = OnceLock::new();

/// The one directory NVRTC is loaded from.
fn directory() -> Result<PathBuf, ToolchainUnavailable> {
    if let Some(directory) = std::env::var_os(DIRECTORY_SETTING) {
        return Ok(PathBuf::from(directory));
    }
    let executable = std::env::current_exe().map_err(|error| ToolchainUnavailable::Unlocated {
        reason: format!("the executable path is unavailable: {error}"),
    })?;
    let parent = executable
        .parent()
        .ok_or_else(|| ToolchainUnavailable::Unlocated {
            reason: format!("{} has no parent directory", executable.display()),
        })?;
    Ok(if cfg!(target_os = "windows") {
        parent.to_path_buf()
    } else {
        parent.join("lib")
    })
}

fn open(directory: &Path, library: &str) -> Result<(PathBuf, Library), ToolchainUnavailable> {
    let path = directory.join(library);
    if !path.is_file() {
        return Err(ToolchainUnavailable::Missing {
            library: library.to_owned(),
            directory: directory.to_path_buf(),
        });
    }
    // Both libraries are NVIDIA's documented redistributable C ABI.
    let loaded =
        unsafe { Library::new(&path) }.map_err(|error| ToolchainUnavailable::Unusable {
            library: library.to_owned(),
            path: path.clone(),
            reason: error.to_string(),
        })?;
    Ok((path, loaded))
}

fn load_from(directory: &Path) -> Result<Nvrtc, ToolchainUnavailable> {
    let (path, library) = open(directory, LIBRARY)?;
    macro_rules! symbol {
        ($name:literal) => {
            *unsafe { library.get(concat!($name, "\0").as_bytes()) }.map_err(|error| {
                ToolchainUnavailable::Unusable {
                    library: LIBRARY.to_owned(),
                    path: path.clone(),
                    reason: format!("{}: {error}", $name),
                }
            })?
        };
    }
    let version: unsafe extern "C" fn(*mut c_int, *mut c_int) -> NvrtcResult =
        symbol!("nvrtcVersion");
    let (mut major, mut minor) = (0, 0);
    if unsafe { version(&mut major, &mut minor) } != 0 {
        return Err(ToolchainUnavailable::Unusable {
            library: LIBRARY.to_owned(),
            path,
            reason: "nvrtcVersion failed".into(),
        });
    }
    let release = (major as u32, minor as u32);
    if release.0 != BUNDLED_MAJOR {
        return Err(ToolchainUnavailable::IncompatibleVersion {
            library: LIBRARY.to_owned(),
            path,
            found: format!("{}.{}", release.0, release.1),
            required: format!("{BUNDLED_MAJOR}.x"),
        });
    }
    let (_, builtins) = open(directory, &builtins_library(release.0, release.1))?;
    Ok(Nvrtc {
        error_string: symbol!("nvrtcGetErrorString"),
        create: symbol!("nvrtcCreateProgram"),
        destroy: symbol!("nvrtcDestroyProgram"),
        compile: symbol!("nvrtcCompileProgram"),
        log_size: symbol!("nvrtcGetProgramLogSize"),
        log: symbol!("nvrtcGetProgramLog"),
        cubin_size: symbol!("nvrtcGetCUBINSize"),
        cubin: symbol!("nvrtcGetCUBIN"),
        arch_count: symbol!("nvrtcGetNumSupportedArchs"),
        archs: symbol!("nvrtcGetSupportedArchs"),
        release,
        _builtins: builtins,
        _library: library,
    })
}

impl Nvrtc {
    fn get() -> Result<Arc<Self>, ToolchainUnavailable> {
        NVRTC
            .get_or_init(|| load_from(&directory()?).map(Arc::new))
            .clone()
    }

    fn describe(&self, result: NvrtcResult) -> String {
        let text = unsafe { (self.error_string)(result) };
        if text.is_null() {
            format!("NVRTC error {result}")
        } else {
            unsafe { CStr::from_ptr(text) }
                .to_string_lossy()
                .into_owned()
        }
    }

    fn check(&self, result: NvrtcResult, operation: &'static str) -> Result<(), NvrtcError> {
        if result == 0 {
            Ok(())
        } else {
            Err(NvrtcError::Call {
                operation,
                message: self.describe(result),
            })
        }
    }

    fn supported_architectures(&self) -> Result<Vec<u32>, NvrtcError> {
        let mut count = 0;
        self.check(
            unsafe { (self.arch_count)(&mut count) },
            "architecture count",
        )?;
        let mut archs = vec![0 as c_int; usize::try_from(count).unwrap_or(0)];
        self.check(unsafe { (self.archs)(archs.as_mut_ptr()) }, "architectures")?;
        Ok(archs.into_iter().map(|arch| arch as u32).collect())
    }
}

/// A failed native CUDA formation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NvrtcError {
    Unavailable(ToolchainUnavailable),
    /// The device's architecture is outside the loaded NVRTC's target set.
    UnsupportedArchitecture {
        architecture: u32,
        supported: Vec<u32>,
    },
    /// NVRTC rejected the source; `log` is its program log.
    Compilation {
        log: String,
    },
    /// An NVRTC API call other than compilation failed.
    Call {
        operation: &'static str,
        message: String,
    },
}

impl std::fmt::Display for NvrtcError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unavailable(error) => write!(f, "NVRTC is unavailable: {error}"),
            Self::UnsupportedArchitecture {
                architecture,
                supported,
            } => write!(
                f,
                "architecture sm_{architecture} is not supported by the loaded NVRTC ({supported:?})"
            ),
            Self::Compilation { log } => write!(f, "NVRTC compilation failed:\n{log}"),
            Self::Call { operation, message } => write!(f, "NVRTC {operation}: {message}"),
        }
    }
}

/// Options fixed for every native formation. Numerics-changing defaults are
/// pinned to IEEE-preserving values; kernels contract explicitly with
/// `seismic_fma_rn` (or use the explicit approximate helpers) where their
/// numerical contract allows. These options are part of every CUDA native
/// artifact identity.
pub const COMPILE_OPTIONS: &[&str] = &[
    "--std=c++17",
    "--fmad=false",
    "--ftz=false",
    "--prec-div=true",
    "--prec-sqrt=true",
];

/// Everything about one NVRTC formation that determines its CUBIN besides
/// the source: the compiler release, the target architecture and the options.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Formation {
    pub release: (u32, u32),
    pub architecture: u32,
    pub options: &'static [&'static str],
}

impl std::fmt::Display for Formation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "nvrtc {}.{};sm_{};{}",
            self.release.0,
            self.release.1,
            self.architecture,
            self.options.join(" ")
        )
    }
}

/// A CUBIN and the formation that produced it.
pub struct Cubin {
    pub image: Vec<u8>,
    pub formation: Formation,
}

/// The formation NVRTC of the resolved directory applies for
/// `sm_<architecture>`, without compiling: what, with the source, determines
/// a CUBIN.
pub fn formation(architecture: u32) -> Result<Formation, NvrtcError> {
    let nvrtc = Nvrtc::get().map_err(NvrtcError::Unavailable)?;
    let supported = nvrtc.supported_architectures()?;
    if !supported.contains(&architecture) {
        return Err(NvrtcError::UnsupportedArchitecture {
            architecture,
            supported,
        });
    }
    Ok(Formation {
        release: nvrtc.release,
        architecture,
        options: COMPILE_OPTIONS,
    })
}

/// The `(major, minor)` release of the NVRTC of the resolved directory.
pub fn release() -> Result<(u32, u32), NvrtcError> {
    Nvrtc::get()
        .map(|nvrtc| nvrtc.release)
        .map_err(NvrtcError::Unavailable)
}

/// Compile `source` to a CUBIN for `sm_<architecture>` with the NVRTC of the
/// resolved directory.
pub fn compile_cubin(source: &str, name: &str, architecture: u32) -> Result<Cubin, NvrtcError> {
    Nvrtc::get()
        .map_err(NvrtcError::Unavailable)?
        .compile(source, name, architecture)
}

impl Nvrtc {
    fn compile(&self, source: &str, name: &str, architecture: u32) -> Result<Cubin, NvrtcError> {
        let nvrtc = self;
        let supported = nvrtc.supported_architectures()?;
        if !supported.contains(&architecture) {
            return Err(NvrtcError::UnsupportedArchitecture {
                architecture,
                supported,
            });
        }
        let source = CString::new(source).map_err(|_| NvrtcError::Compilation {
            log: "native source contains a NUL byte".into(),
        })?;
        let name = CString::new(name).map_err(|_| NvrtcError::Compilation {
            log: "native source name contains a NUL byte".into(),
        })?;
        let mut program: Program = std::ptr::null_mut();
        nvrtc.check(
            unsafe {
                (nvrtc.create)(
                    &mut program,
                    source.as_ptr(),
                    name.as_ptr(),
                    0,
                    std::ptr::null(),
                    std::ptr::null(),
                )
            },
            "program creation",
        )?;
        struct Owned<'a>(&'a Nvrtc, Program);
        impl Drop for Owned<'_> {
            fn drop(&mut self) {
                unsafe { (self.0.destroy)(&mut self.1) };
            }
        }
        let owned = Owned(nvrtc, program);
        let arch = CString::new(format!("-arch=sm_{architecture}")).expect("no NUL");
        let fixed = COMPILE_OPTIONS
            .iter()
            .map(|option| CString::new(*option).expect("no NUL"))
            .collect::<Vec<_>>();
        let options = std::iter::once(arch.as_ptr())
            .chain(fixed.iter().map(|option| option.as_ptr()))
            .collect::<Vec<_>>();
        let status = unsafe { (nvrtc.compile)(owned.1, options.len() as c_int, options.as_ptr()) };
        if status != 0 {
            let mut size = 0usize;
            unsafe { (nvrtc.log_size)(owned.1, &mut size) };
            let mut log = vec![0u8; size.max(1)];
            unsafe { (nvrtc.log)(owned.1, log.as_mut_ptr().cast()) };
            let end = log.iter().position(|byte| *byte == 0).unwrap_or(log.len());
            return Err(NvrtcError::Compilation {
                log: format!(
                    "{}\n{}",
                    nvrtc.describe(status),
                    String::from_utf8_lossy(&log[..end])
                ),
            });
        }
        let mut size = 0usize;
        nvrtc.check(
            unsafe { (nvrtc.cubin_size)(owned.1, &mut size) },
            "CUBIN size",
        )?;
        let mut image = vec![0u8; size];
        nvrtc.check(
            unsafe { (nvrtc.cubin)(owned.1, image.as_mut_ptr().cast()) },
            "CUBIN",
        )?;
        Ok(Cubin {
            image,
            formation: Formation {
                release: nvrtc.release,
                architecture,
                options: COMPILE_OPTIONS,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_directory_without_nvrtc_is_typed_missing() {
        let directory = std::env::temp_dir().join("seismic-nvrtc-absent");
        std::fs::create_dir_all(&directory).expect("scratch directory");
        match load_from(&directory) {
            Err(ToolchainUnavailable::Missing {
                library,
                directory: reported,
            }) => {
                assert_eq!(library, LIBRARY);
                assert_eq!(reported, directory);
            }
            other => panic!("expected a missing library, got {:?}", other.err()),
        }
    }

    /// With the development setting naming a directory that holds NVRTC and
    /// its builtins (a CUDA toolkit's, or a copy laid out as an installation
    /// bundles them), formation binds the builtins from that directory, not
    /// through the dynamic loader's search path.
    #[cfg(target_os = "linux")]
    #[test]
    fn formation_binds_the_builtins_of_the_resolved_directory() {
        let Some(directory) = std::env::var_os(DIRECTORY_SETTING) else {
            eprintln!("{DIRECTORY_SETTING} is not set; nothing to verify");
            return;
        };
        let directory = std::fs::canonicalize(PathBuf::from(directory)).expect("directory");
        let nvrtc = load_from(&directory).expect("NVRTC loads from the named directory");
        let architecture = *nvrtc
            .supported_architectures()
            .expect("architectures")
            .last()
            .expect("at least one architecture");
        let cubin = nvrtc
            .compile(
                "extern \"C\" __global__ void probe(float *out) { out[threadIdx.x] = 1.0f; }",
                "probe.cu",
                architecture,
            )
            .expect("probe compiles");
        assert!(!cubin.image.is_empty());
        let builtins = builtins_library(nvrtc.release.0, nvrtc.release.1);
        let maps = std::fs::read_to_string("/proc/self/maps").expect("process maps");
        let mapped = maps
            .lines()
            .filter_map(|line| line.split_whitespace().nth(5))
            .filter(|path| path.contains("libnvrtc-builtins"))
            .map(|path| std::fs::canonicalize(path).expect("mapped path"))
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(
            mapped,
            [std::fs::canonicalize(directory.join(&builtins)).expect("builtins")]
                .into_iter()
                .collect(),
            "builtins mapped from {mapped:?}"
        );
    }
}
