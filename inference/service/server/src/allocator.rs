//! The service process's heap policy. Assessment, inspection and load previews parse
//! whole GGUF headers and tokenizers on many blocking threads: hundreds of megabytes that live
//! briefly. On glibc, threads get their own arenas (up to eight per core) and retain free pages,
//! so the service caps arenas, keeps large blocks on mappings of their own, and trims free pages.
//! On macOS, libmalloc retains freed pages in magazine/nanomalloc zones until pressured, so the
//! service calls `malloc_zone_pressure_relief(NULL, 0)` after releasing assessment caches and
//! periodically in the background.

#[cfg(all(target_os = "linux", target_env = "gnu"))]
mod platform {
    const ARENAS: libc::c_int = 2;
    const MMAP_THRESHOLD_BYTES: libc::c_int = 1024 * 1024;

    pub fn configure() {
        // SAFETY: mallopt only changes allocator parameters; it is called before the runtime
        // starts any thread.
        unsafe {
            libc::mallopt(libc::M_ARENA_MAX, ARENAS);
            libc::mallopt(libc::M_MMAP_THRESHOLD, MMAP_THRESHOLD_BYTES);
        }
    }

    pub fn release_now() {
        // SAFETY: malloc_trim only returns free pages to the system.
        unsafe {
            libc::malloc_trim(0);
        }
    }
}

#[cfg(target_os = "macos")]
mod platform {
    unsafe extern "C" {
        fn malloc_zone_pressure_relief(zone: *mut std::ffi::c_void, goal: usize) -> usize;
    }

    pub fn configure() {}

    pub fn release_now() {
        // SAFETY: passing NULL for zone relieves pressure across all malloc zones, and goal=0
        // asks libmalloc to release as many unused pages as possible back to the OS.
        unsafe {
            let _ = malloc_zone_pressure_relief(std::ptr::null_mut(), 0);
        }
    }
}

#[cfg(not(any(all(target_os = "linux", target_env = "gnu"), target_os = "macos")))]
mod platform {
    pub fn configure() {}

    pub fn release_now() {}
}

pub use platform::{configure, release_now};

#[cfg(any(all(target_os = "linux", target_env = "gnu"), target_os = "macos"))]
pub async fn release_free_memory() {
    use std::time::Duration;
    const RELEASE_INTERVAL: Duration = Duration::from_secs(30);

    let mut interval = tokio::time::interval(RELEASE_INTERVAL);
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    loop {
        interval.tick().await;
        magnitude_service_utils::blocking::run(release_now).await;
    }
}

#[cfg(not(any(all(target_os = "linux", target_env = "gnu"), target_os = "macos")))]
pub async fn release_free_memory() {}
