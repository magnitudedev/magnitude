//! NVRTC: the bundled runtime compiler that forms authored CUDA C++ native
//! kernels for the opened device's exact architecture.
//!
//! The library is an owned native dependency of the CUDA backend. It is
//! resolved only relative to the installation (the executable's directory and
//! its `lib/` sibling), never from ambient toolkit paths.

use libloading::Library;
use std::ffi::{c_char, c_int, CStr, CString};
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

type NvrtcResult = c_int;
type Program = *mut std::ffi::c_void;

/// The NVRTC major version the backend bundles; it sets the driver
/// requirement (CUDA minor-version compatibility within one major).
pub const BUNDLED_MAJOR: u32 = 13;

#[cfg(target_os = "windows")]
const LIBRARY_NAMES: &[&str] = &["nvrtc64_130_0.dll"];
#[cfg(not(target_os = "windows"))]
const LIBRARY_NAMES: &[&str] = &["libnvrtc.so.13"];

struct Nvrtc {
    version: unsafe extern "C" fn(*mut c_int, *mut c_int) -> NvrtcResult,
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
    _library: Library,
}

// Function pointers into a library kept alive by the same value. NVRTC
// programs are created and destroyed per compilation; the API is thread-safe
// for distinct programs.
unsafe impl Send for Nvrtc {}
unsafe impl Sync for Nvrtc {}

static NVRTC: OnceLock<Result<Arc<Nvrtc>, NvrtcUnavailable>> = OnceLock::new();

/// Why native CUDA formation is unavailable on this installation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NvrtcUnavailable(pub String);

impl std::fmt::Display for NvrtcUnavailable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "NVRTC is unavailable: {}", self.0)
    }
}

/// Directories owned by the installation, in search order.
fn owned_directories() -> Vec<PathBuf> {
    let Ok(executable) = std::env::current_exe() else {
        return Vec::new();
    };
    let Some(directory) = executable.parent() else {
        return Vec::new();
    };
    vec![directory.to_path_buf(), directory.join("lib")]
}

fn load() -> Result<Arc<Nvrtc>, NvrtcUnavailable> {
    let mut failures = Vec::new();
    for directory in owned_directories() {
        for name in LIBRARY_NAMES {
            let path = directory.join(name);
            if !path.exists() {
                continue;
            }
            // NVRTC exports the documented C ABI below.
            let library = match unsafe { Library::new(&path) } {
                Ok(library) => library,
                Err(error) => {
                    failures.push(format!("{}: {error}", path.display()));
                    continue;
                }
            };
            macro_rules! symbol {
                ($name:literal) => {
                    *unsafe { library.get(concat!($name, "\0").as_bytes()) }
                        .map_err(|error| NvrtcUnavailable(format!("{}: {error}", $name)))?
                };
            }
            let nvrtc = Nvrtc {
                version: symbol!("nvrtcVersion"),
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
                _library: library,
            };
            let (major, _) = nvrtc.version()?;
            if major != BUNDLED_MAJOR {
                return Err(NvrtcUnavailable(format!(
                    "{} is NVRTC {major}, but this build requires NVRTC {BUNDLED_MAJOR}",
                    path.display()
                )));
            }
            return Ok(Arc::new(nvrtc));
        }
    }
    Err(NvrtcUnavailable(if failures.is_empty() {
        format!(
            "{} was not found in the installation directories",
            LIBRARY_NAMES.join(" / ")
        )
    } else {
        failures.join("; ")
    }))
}

impl Nvrtc {
    fn get() -> Result<Arc<Self>, NvrtcUnavailable> {
        NVRTC.get_or_init(load).clone()
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

    fn check(&self, result: NvrtcResult, operation: &str) -> Result<(), NvrtcUnavailable> {
        if result == 0 {
            Ok(())
        } else {
            Err(NvrtcUnavailable(format!(
                "{operation}: {}",
                self.describe(result)
            )))
        }
    }

    fn version(&self) -> Result<(u32, u32), NvrtcUnavailable> {
        let (mut major, mut minor) = (0, 0);
        self.check(unsafe { (self.version)(&mut major, &mut minor) }, "version")?;
        Ok((major as u32, minor as u32))
    }

    fn supported_architectures(&self) -> Result<Vec<u32>, NvrtcUnavailable> {
        let mut count = 0;
        self.check(
            unsafe { (self.arch_count)(&mut count) },
            "architecture count",
        )?;
        let mut archs = vec![0 as c_int; count.max(0) as usize];
        self.check(unsafe { (self.archs)(archs.as_mut_ptr()) }, "architectures")?;
        Ok(archs.into_iter().map(|arch| arch as u32).collect())
    }
}

/// The identity of the NVRTC that formed an artifact.
pub fn version() -> Result<(u32, u32), NvrtcUnavailable> {
    Nvrtc::get()?.version()
}

/// A failed native CUDA formation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NvrtcError {
    Unavailable(NvrtcUnavailable),
    /// The device's architecture is newer than the bundled NVRTC supports.
    UnsupportedArchitecture {
        architecture: u32,
        supported: Vec<u32>,
    },
    /// NVRTC rejected the source; `log` is its program log.
    Compilation {
        log: String,
    },
}

impl std::fmt::Display for NvrtcError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unavailable(error) => write!(f, "{error}"),
            Self::UnsupportedArchitecture {
                architecture,
                supported,
            } => write!(
                f,
                "architecture sm_{architecture} is newer than the bundled NVRTC supports ({supported:?})"
            ),
            Self::Compilation { log } => write!(f, "NVRTC compilation failed:\n{log}"),
        }
    }
}

/// Options fixed for every native formation. Numerics-changing defaults are
/// pinned to IEEE-preserving values; kernels contract explicitly with
/// `__fmaf_rn` where their numerical contract allows.
pub const COMPILE_OPTIONS: &[&str] = &[
    "--std=c++17",
    "--fmad=false",
    "--ftz=false",
    "--prec-div=true",
    "--prec-sqrt=true",
];

/// Compile `source` to a CUBIN for `sm_<architecture>`.
pub fn compile_cubin(source: &str, name: &str, architecture: u32) -> Result<Vec<u8>, NvrtcError> {
    let nvrtc = Nvrtc::get().map_err(NvrtcError::Unavailable)?;
    let supported = nvrtc
        .supported_architectures()
        .map_err(NvrtcError::Unavailable)?;
    if !supported.contains(&architecture) {
        return Err(NvrtcError::UnsupportedArchitecture {
            architecture,
            supported,
        });
    }
    let source = CString::new(source).map_err(|_| NvrtcError::Compilation {
        log: "native source contains a NUL byte".into(),
    })?;
    let name = CString::new(name).expect("native source names contain no NUL byte");
    let mut program: Program = std::ptr::null_mut();
    nvrtc
        .check(
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
        )
        .map_err(NvrtcError::Unavailable)?;
    struct Owned<'a>(&'a Nvrtc, Program);
    impl Drop for Owned<'_> {
        fn drop(&mut self) {
            unsafe { (self.0.destroy)(&mut self.1) };
        }
    }
    let owned = Owned(&*nvrtc, program);
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
    nvrtc
        .check(
            unsafe { (nvrtc.cubin_size)(owned.1, &mut size) },
            "CUBIN size",
        )
        .map_err(NvrtcError::Unavailable)?;
    let mut cubin = vec![0u8; size];
    nvrtc
        .check(
            unsafe { (nvrtc.cubin)(owned.1, cubin.as_mut_ptr().cast()) },
            "CUBIN",
        )
        .map_err(NvrtcError::Unavailable)?;
    Ok(cubin)
}
