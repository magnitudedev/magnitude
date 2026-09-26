#[cfg(target_os = "linux")]
use std::collections::HashSet;
#[cfg(any(target_os = "linux", target_os = "windows"))]
use std::ffi::CStr;
#[cfg(any(target_os = "linux", target_os = "windows", test))]
use std::path::Path;
use std::path::PathBuf;
use std::sync::OnceLock;

use libloading::Library;

#[cfg(any(target_os = "linux", test))]
const MAX_CANDIDATES_PER_DIRECTORY: usize = 32;
#[cfg(target_os = "linux")]
const MAX_CHILD_PROVIDER_DIRECTORIES: usize = 64;

#[cfg(any(target_os = "linux", target_os = "windows", test))]
const CUDA_ERROR_STUB_LIBRARY: i32 = 34;
#[cfg(any(target_os = "linux", target_os = "windows", test))]
const CUDA_ERROR_NO_DEVICE: i32 = 100;
#[cfg(any(target_os = "linux", target_os = "windows"))]
const CUDA_COMPUTE_CAPABILITY_MAJOR: i32 = 75;
#[cfg(any(target_os = "linux", target_os = "windows"))]
const CUDA_COMPUTE_CAPABILITY_MINOR: i32 = 76;

#[cfg(any(target_os = "linux", target_os = "windows"))]
type Init = unsafe extern "C" fn(u32) -> i32;
#[cfg(any(target_os = "linux", target_os = "windows"))]
type DriverVersion = unsafe extern "C" fn(*mut i32) -> i32;
#[cfg(any(target_os = "linux", target_os = "windows"))]
type DeviceCount = unsafe extern "C" fn(*mut i32) -> i32;
#[cfg(any(target_os = "linux", target_os = "windows"))]
type DeviceGet = unsafe extern "C" fn(*mut i32, i32) -> i32;
#[cfg(any(target_os = "linux", target_os = "windows"))]
type DeviceAttribute = unsafe extern "C" fn(*mut i32, i32, i32) -> i32;
#[cfg(any(target_os = "linux", target_os = "windows"))]
type DeviceName = unsafe extern "C" fn(*mut std::ffi::c_char, i32, i32) -> i32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(
    not(any(target_os = "linux", target_os = "windows", test)),
    allow(dead_code)
)]
pub(crate) enum CudaDriverFailureKind {
    Absent,
    Failed,
}

#[derive(Debug, Clone)]
pub(crate) struct CudaDriverFailure {
    pub(crate) kind: CudaDriverFailureKind,
    pub(crate) diagnostic: String,
}

impl CudaDriverFailure {
    fn absent(diagnostic: impl std::fmt::Display) -> Self {
        Self {
            kind: CudaDriverFailureKind::Absent,
            diagnostic: bounded(diagnostic),
        }
    }

    #[cfg(any(target_os = "linux", target_os = "windows", test))]
    fn failed(diagnostic: impl std::fmt::Display) -> Self {
        Self {
            kind: CudaDriverFailureKind::Failed,
            diagnostic: bounded(diagnostic),
        }
    }
}

pub(crate) struct CudaDriver {
    // The handle intentionally lives for the process lifetime. CUDA backend modules and
    // libcudart may subsequently resolve the already-loaded host provider by its SONAME.
    _library: Library,
    pub(crate) path: PathBuf,
    pub(crate) driver_api: i32,
    pub(crate) architectures: Vec<String>,
    pub(crate) hardware_labels: Vec<String>,
}

static DRIVER: OnceLock<Result<CudaDriver, CudaDriverFailure>> = OnceLock::new();

fn bounded(value: impl std::fmt::Display) -> String {
    value
        .to_string()
        .replace(['\r', '\n'], " ")
        .chars()
        .take(240)
        .collect()
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
#[derive(Clone, Copy)]
struct CudaApi {
    init: Init,
    driver_version: DriverVersion,
    device_count: DeviceCount,
    device_get: DeviceGet,
    device_attribute: DeviceAttribute,
    device_name: DeviceName,
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
impl CudaApi {
    fn load(library: &Library) -> Result<Self, CudaDriverFailure> {
        unsafe {
            Ok(Self {
                init: *library
                    .get::<Init>(b"cuInit\0")
                    .map_err(|_| CudaDriverFailure::failed("CUDA driver is missing cuInit"))?,
                driver_version: *library
                    .get::<DriverVersion>(b"cuDriverGetVersion\0")
                    .map_err(|_| {
                        CudaDriverFailure::failed("CUDA driver is missing cuDriverGetVersion")
                    })?,
                device_count: *library
                    .get::<DeviceCount>(b"cuDeviceGetCount\0")
                    .map_err(|_| {
                        CudaDriverFailure::failed("CUDA driver is missing cuDeviceGetCount")
                    })?,
                device_get: *library
                    .get::<DeviceGet>(b"cuDeviceGet\0")
                    .map_err(|_| CudaDriverFailure::failed("CUDA driver is missing cuDeviceGet"))?,
                device_attribute: *library
                    .get::<DeviceAttribute>(b"cuDeviceGetAttribute\0")
                    .map_err(|_| {
                        CudaDriverFailure::failed("CUDA driver is missing cuDeviceGetAttribute")
                    })?,
                device_name: *library
                    .get::<DeviceName>(b"cuDeviceGetName\0")
                    .map_err(|_| {
                        CudaDriverFailure::failed("CUDA driver is missing cuDeviceGetName")
                    })?,
            })
        }
    }

    fn initialize(self) -> i32 {
        unsafe { (self.init)(0) }
    }

    fn driver_version(self, version: &mut i32) -> i32 {
        unsafe { (self.driver_version)(version) }
    }

    fn device_count(self, count: &mut i32) -> i32 {
        unsafe { (self.device_count)(count) }
    }

    fn device(self, device: &mut i32, ordinal: i32) -> i32 {
        unsafe { (self.device_get)(device, ordinal) }
    }

    fn device_attribute(self, value: &mut i32, attribute: i32, device: i32) -> i32 {
        unsafe { (self.device_attribute)(value, attribute, device) }
    }

    fn device_name(self, buffer: &mut [std::ffi::c_char], device: i32) -> i32 {
        unsafe { (self.device_name)(buffer.as_mut_ptr(), buffer.len() as i32, device) }
    }
}

#[cfg(any(target_os = "linux", target_os = "windows", test))]
fn cuda_init_failure(code: i32) -> Option<CudaDriverFailure> {
    match code {
        0 => None,
        CUDA_ERROR_STUB_LIBRARY => Some(CudaDriverFailure::absent(
            "only the CUDA stub driver is available",
        )),
        CUDA_ERROR_NO_DEVICE => Some(CudaDriverFailure::absent("no CUDA device is available")),
        _ => Some(CudaDriverFailure::failed(format!(
            "CUDA initialization failed with code {code}"
        ))),
    }
}

#[cfg(any(target_os = "linux", test))]
fn is_driver_filename(name: &str) -> bool {
    let Some(version) = name.strip_prefix("libcuda.so.") else {
        return false;
    };
    version.split('.').all(|component| {
        !component.is_empty() && component.bytes().all(|byte| byte.is_ascii_digit())
    })
}

#[cfg(any(target_os = "linux", test))]
fn add_driver_files(directory: &Path, candidates: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(directory) else {
        return;
    };
    let mut found = entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(is_driver_filename)
        })
        .collect::<Vec<_>>();
    found.sort();
    candidates.extend(found.into_iter().take(MAX_CANDIDATES_PER_DIRECTORY));
}

#[cfg(target_os = "linux")]
fn add_child_driver_directories(directory: &Path, candidates: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(directory) else {
        return;
    };
    let mut directories = entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.is_dir())
        .collect::<Vec<_>>();
    directories.sort();
    for directory in directories.into_iter().take(MAX_CHILD_PROVIDER_DIRECTORIES) {
        add_driver_files(&directory, candidates);
    }
}

#[cfg(target_os = "linux")]
fn is_wsl() -> bool {
    std::env::var_os("WSL_INTEROP").is_some()
        || Path::new("/usr/lib/wsl/lib").is_dir()
        || std::fs::read_to_string("/proc/sys/kernel/osrelease")
            .is_ok_and(|release| release.to_ascii_lowercase().contains("microsoft"))
}

#[cfg(target_os = "linux")]
fn linux_fallback_candidates() -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    let wsl = is_wsl();
    let add_wsl = |candidates: &mut Vec<PathBuf>| {
        add_driver_files(Path::new("/usr/lib/wsl/lib"), candidates);
        add_child_driver_directories(Path::new("/usr/lib/wsl/drivers"), candidates);
    };

    if wsl {
        add_wsl(&mut candidates);
    }

    let multiarch_name = match std::env::consts::ARCH {
        "x86_64" => Some("x86_64-linux-gnu"),
        "aarch64" => Some("aarch64-linux-gnu"),
        _ => None,
    };
    if let Some(multiarch_name) = multiarch_name {
        let multiarch = Path::new("/usr/lib").join(multiarch_name);
        add_driver_files(&multiarch, &mut candidates);
        add_driver_files(&Path::new("/lib").join(multiarch_name), &mut candidates);
        let nvidia = multiarch.join("nvidia");
        add_driver_files(&nvidia.join("current"), &mut candidates);
        add_child_driver_directories(&nvidia, &mut candidates);
    }

    for directory in [
        "/usr/lib64",
        "/usr/lib",
        "/run/opengl-driver/lib",
        "/usr/local/nvidia/lib",
        "/usr/local/nvidia/lib64",
    ] {
        add_driver_files(Path::new(directory), &mut candidates);
    }

    let mut seen = HashSet::new();
    candidates.retain(|path| {
        let identity = path.canonicalize().unwrap_or_else(|_| path.clone());
        seen.insert(identity)
    });
    candidates
}

#[cfg(target_os = "linux")]
fn open_candidate(path: &Path) -> Result<Library, libloading::Error> {
    use libloading::os::unix::{Library as UnixLibrary, RTLD_LOCAL, RTLD_NOW};

    // Validate locally first so a stub or malformed candidate cannot pollute the process.
    unsafe { UnixLibrary::open(Some(path), RTLD_NOW | RTLD_LOCAL).map(Into::into) }
}

#[cfg(target_os = "linux")]
fn loaded_path(init: Init, fallback: &Path) -> PathBuf {
    unsafe {
        let mut info: libc::Dl_info = std::mem::zeroed();
        if libc::dladdr(init as *const () as *const libc::c_void, &mut info) != 0
            && !info.dli_fname.is_null()
        {
            let path = PathBuf::from(
                CStr::from_ptr(info.dli_fname)
                    .to_string_lossy()
                    .into_owned(),
            );
            return path.canonicalize().unwrap_or(path);
        }
    }
    fallback.to_path_buf()
}

#[cfg(target_os = "linux")]
fn promote_global(path: &Path) -> Result<Library, libloading::Error> {
    use libloading::os::unix::{Library as UnixLibrary, RTLD_GLOBAL, RTLD_NOW};

    unsafe { UnixLibrary::open(Some(path), RTLD_NOW | RTLD_GLOBAL).map(Into::into) }
}

#[cfg(target_os = "windows")]
fn open_candidate(path: &Path) -> Result<Library, libloading::Error> {
    unsafe {
        libloading::os::windows::Library::load_with_flags(
            path,
            windows_sys::Win32::System::LibraryLoader::LOAD_LIBRARY_SEARCH_SYSTEM32,
        )
        .map(Into::into)
    }
}

#[cfg(target_os = "windows")]
fn loaded_path(_init: Init, fallback: &Path) -> PathBuf {
    fallback.to_path_buf()
}

#[cfg(target_os = "windows")]
fn windows_candidate() -> Result<PathBuf, CudaDriverFailure> {
    let mut buffer = vec![0u16; 32_768];
    let length = unsafe {
        windows_sys::Win32::System::SystemInformation::GetSystemDirectoryW(
            buffer.as_mut_ptr(),
            buffer.len() as u32,
        )
    };
    if length == 0 || length as usize >= buffer.len() {
        return Err(CudaDriverFailure::failed(
            "unable to resolve the Windows system directory",
        ));
    }
    buffer.truncate(length as usize);
    Ok(PathBuf::from(String::from_utf16_lossy(&buffer)).join("nvcuda.dll"))
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
fn inspect(
    library: &Library,
    candidate: &Path,
) -> Result<(PathBuf, i32, Vec<String>, Vec<String>), CudaDriverFailure> {
    let api = CudaApi::load(library)?;
    if let Some(failure) = cuda_init_failure(api.initialize()) {
        return Err(failure);
    }

    let mut driver_api = 0;
    let version_code = api.driver_version(&mut driver_api);
    let mut device_count = 0;
    let count_code = api.device_count(&mut device_count);
    if version_code != 0 || count_code != 0 || driver_api <= 0 {
        return Err(CudaDriverFailure::failed(format!(
            "CUDA probe failed (version={version_code}, count={count_code})"
        )));
    }
    if device_count <= 0 {
        return Err(CudaDriverFailure::absent("no CUDA device is available"));
    }

    let mut architectures = Vec::new();
    let mut hardware_labels = Vec::new();
    for ordinal in 0..device_count {
        let mut device = 0;
        let mut major = 0;
        let mut minor = 0;
        if api.device(&mut device, ordinal) == 0
            && api.device_attribute(&mut major, CUDA_COMPUTE_CAPABILITY_MAJOR, device) == 0
            && api.device_attribute(&mut minor, CUDA_COMPUTE_CAPABILITY_MINOR, device) == 0
            && major > 0
        {
            architectures.push(format!("{major}{minor}"));
            let mut name = [0 as std::ffi::c_char; 256];
            if api.device_name(&mut name, device) == 0 {
                let label = unsafe { CStr::from_ptr(name.as_ptr()) }
                    .to_string_lossy()
                    .trim()
                    .to_owned();
                if !label.is_empty() {
                    hardware_labels.push(label);
                }
            }
        }
    }
    if architectures.is_empty() {
        return Err(CudaDriverFailure::failed(
            "CUDA devices did not report a compute architecture",
        ));
    }
    Ok((
        loaded_path(api.init, candidate),
        driver_api,
        architectures,
        hardware_labels,
    ))
}

#[cfg(target_os = "linux")]
fn load() -> Result<CudaDriver, CudaDriverFailure> {
    let mut candidates = vec![PathBuf::from("libcuda.so.1")];
    candidates.extend(linux_fallback_candidates());
    let mut last_failure = None;
    let mut last_load_error = None;

    for candidate in candidates {
        let library = match open_candidate(&candidate) {
            Ok(library) => library,
            Err(error) => {
                last_load_error = Some(format!("{}: {error}", candidate.display()));
                continue;
            }
        };
        match inspect(&library, &candidate) {
            Ok((path, driver_api, architectures, hardware_labels)) => {
                let library = promote_global(&path).map_err(|error| {
                    CudaDriverFailure::failed(format!("unable to expose CUDA driver: {error}"))
                })?;
                return Ok(CudaDriver {
                    _library: library,
                    path,
                    driver_api,
                    architectures,
                    hardware_labels,
                });
            }
            Err(failure) => last_failure = Some(failure),
        }
    }

    Err(last_failure.unwrap_or_else(|| {
        CudaDriverFailure::absent(last_load_error.map_or_else(
            || "CUDA driver library is unavailable".to_owned(),
            |error| format!("CUDA driver library is unavailable ({error})"),
        ))
    }))
}

#[cfg(target_os = "windows")]
fn load() -> Result<CudaDriver, CudaDriverFailure> {
    let candidate = windows_candidate()?;
    let library = open_candidate(&candidate).map_err(|error| {
        CudaDriverFailure::absent(format!("CUDA driver library is unavailable ({error})"))
    })?;
    let (path, driver_api, architectures, hardware_labels) = inspect(&library, &candidate)?;
    Ok(CudaDriver {
        _library: library,
        path,
        driver_api,
        architectures,
        hardware_labels,
    })
}

#[cfg(not(any(target_os = "linux", target_os = "windows")))]
fn load() -> Result<CudaDriver, CudaDriverFailure> {
    Err(CudaDriverFailure::absent(
        "CUDA is unavailable on this platform",
    ))
}

pub(crate) fn resolve() -> Result<&'static CudaDriver, &'static CudaDriverFailure> {
    DRIVER.get_or_init(load).as_ref()
}

pub(crate) fn require() -> anyhow::Result<&'static CudaDriver> {
    resolve().map_err(|failure| anyhow::anyhow!(failure.diagnostic.clone()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_only_versioned_driver_provider_names() {
        assert!(is_driver_filename("libcuda.so.1"));
        assert!(is_driver_filename("libcuda.so.555.42.02"));
        assert!(!is_driver_filename("libcuda.so"));
        assert!(!is_driver_filename("libcuda.so.stub"));
        assert!(!is_driver_filename("libcuda.so.1."));
        assert!(!is_driver_filename("libcuda.so.1..2"));
        assert!(!is_driver_filename("libcudadebugger.so.1"));
    }

    #[test]
    fn classifies_cuda_initialization_results() {
        assert!(cuda_init_failure(0).is_none());
        assert_eq!(
            cuda_init_failure(CUDA_ERROR_STUB_LIBRARY).unwrap().kind,
            CudaDriverFailureKind::Absent,
        );
        assert_eq!(
            cuda_init_failure(CUDA_ERROR_NO_DEVICE).unwrap().kind,
            CudaDriverFailureKind::Absent,
        );
        assert_eq!(
            cuda_init_failure(35).unwrap().kind,
            CudaDriverFailureKind::Failed,
        );
    }

    #[test]
    fn directory_discovery_is_sorted_and_limited_to_driver_files() {
        let root = tempfile::tempdir().unwrap();
        for name in ["libcuda.so.2", "libcuda.so", "libcuda.so.1", "other.so.1"] {
            std::fs::write(root.path().join(name), b"fixture").unwrap();
        }
        let mut candidates = Vec::new();
        add_driver_files(root.path(), &mut candidates);
        assert_eq!(
            candidates,
            [
                root.path().join("libcuda.so.1"),
                root.path().join("libcuda.so.2")
            ]
        );
    }

    #[test]
    fn directory_discovery_has_a_candidate_limit() {
        let root = tempfile::tempdir().unwrap();
        for version in 0..(MAX_CANDIDATES_PER_DIRECTORY + 8) {
            std::fs::write(
                root.path().join(format!("libcuda.so.{version}")),
                b"fixture",
            )
            .unwrap();
        }
        let mut candidates = Vec::new();
        add_driver_files(root.path(), &mut candidates);
        assert_eq!(candidates.len(), MAX_CANDIDATES_PER_DIRECTORY);
        assert!(candidates.windows(2).all(|pair| pair[0] < pair[1]));
    }
}
