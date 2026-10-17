//! The service process's heap policy on glibc. Assessment, inspection and load previews parse
//! whole GGUF headers and tokenizers on many blocking threads: hundreds of megabytes that live
//! briefly. glibc gives threads their own arenas (up to eight per core), keeps the free pages of
//! each, and raises its mmap threshold with every large free, so those peaks stay resident for
//! the life of the process. The service caps the arenas, keeps large blocks on mappings of their
//! own, and periodically returns the free pages left inside the arenas. Other platforms'
//! allocators return freed memory themselves.

#[cfg(all(target_os = "linux", target_env = "gnu"))]
mod glibc {
    use std::time::Duration;

    const ARENAS: libc::c_int = 2;
    const MMAP_THRESHOLD_BYTES: libc::c_int = 1024 * 1024;
    const RELEASE_INTERVAL: Duration = Duration::from_secs(30);

    pub(crate) fn configure() {
        // SAFETY: mallopt only changes allocator parameters; it is called before the runtime
        // starts any thread.
        unsafe {
            libc::mallopt(libc::M_ARENA_MAX, ARENAS);
            libc::mallopt(libc::M_MMAP_THRESHOLD, MMAP_THRESHOLD_BYTES);
        }
    }

    pub(crate) async fn release_free_memory() {
        let mut interval = tokio::time::interval(RELEASE_INTERVAL);
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            interval.tick().await;
            // SAFETY: malloc_trim only returns free pages to the system.
            let _ = tokio::task::spawn_blocking(|| unsafe { libc::malloc_trim(0) }).await;
        }
    }
}

#[cfg(all(target_os = "linux", target_env = "gnu"))]
pub(crate) use glibc::{configure, release_free_memory};

#[cfg(not(all(target_os = "linux", target_env = "gnu")))]
pub(crate) fn configure() {}

#[cfg(not(all(target_os = "linux", target_env = "gnu")))]
pub(crate) async fn release_free_memory() {}
