use std::sync::Mutex;
use std::time::{Duration, Instant};

use sysinfo::{MemoryRefreshKind, Pid, ProcessRefreshKind, ProcessesToUpdate, RefreshKind, System};

pub(crate) const POLL_INTERVAL: Duration = Duration::from_millis(100);
pub(crate) const IDLE_POLL_INTERVAL: Duration = Duration::from_secs(1);
pub(crate) const MONITOR_LOSS_DEADLINE: Duration = Duration::from_secs(1);
pub(crate) const RECOVERY_STABLE_TIME: Duration = Duration::from_secs(5);
pub(crate) const MINIMUM_RESERVE_BYTES: u64 = 2 * 1024 * 1024 * 1024;
pub(crate) const RECOVERY_MARGIN_BYTES: u64 = 512 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct MemorySample {
    pub(crate) captured_at: Instant,
    pub(crate) total_bytes: u64,
    pub(crate) available_bytes: u64,
    pub(crate) commit_limit_bytes: Option<u64>,
    pub(crate) available_commit_bytes: Option<u64>,
}

impl MemorySample {
    pub(crate) fn reserve_bytes(self) -> u64 {
        MINIMUM_RESERVE_BYTES.max(self.total_bytes / 10)
    }

    pub(crate) fn permits_load(self, estimated_incremental_bytes: u64) -> bool {
        let required = self
            .reserve_bytes()
            .saturating_add(estimated_incremental_bytes);
        self.available_bytes >= required
            && self
                .available_commit_bytes
                .is_none_or(|available| available >= required)
    }

    pub(crate) fn requires_eviction(self) -> bool {
        self.available_bytes <= self.reserve_bytes()
            || self
                .available_commit_bytes
                .is_some_and(|available| available <= self.reserve_bytes())
    }

    pub(crate) fn recovered(self) -> bool {
        self.available_bytes > self.reserve_bytes().saturating_add(RECOVERY_MARGIN_BYTES)
    }
}

pub(crate) struct SystemMemoryObserver {
    system: Mutex<System>,
}

impl SystemMemoryObserver {
    pub(crate) fn new() -> Self {
        Self {
            system: Mutex::new(System::new_with_specifics(
                RefreshKind::nothing().with_memory(MemoryRefreshKind::everything()),
            )),
        }
    }

    pub(crate) fn sample(&self) -> Result<MemorySample, String> {
        let mut system = self
            .system
            .lock()
            .map_err(|_| "system memory observer lock poisoned".to_owned())?;
        system.refresh_memory_specifics(MemoryRefreshKind::everything());
        let total_bytes = system.total_memory();
        let available_bytes = system.available_memory();
        if total_bytes == 0 || available_bytes > total_bytes {
            return Err(format!(
                "invalid system memory observation: total={total_bytes}, available={available_bytes}"
            ));
        }
        let (commit_limit_bytes, available_commit_bytes) = windows_commit()?;
        Ok(MemorySample {
            captured_at: Instant::now(),
            total_bytes,
            available_bytes,
            commit_limit_bytes,
            available_commit_bytes,
        })
    }

    pub(crate) fn worker_resident_bytes(&self, pid: u32) -> Option<u64> {
        let mut system = self.system.lock().ok()?;
        let pid = Pid::from_u32(pid);
        system.refresh_processes_specifics(
            ProcessesToUpdate::Some(&[pid]),
            true,
            ProcessRefreshKind::nothing().with_memory(),
        );
        system.process(pid).map(|process| process.memory())
    }
}

#[cfg(not(windows))]
fn windows_commit() -> Result<(Option<u64>, Option<u64>), String> {
    Ok((None, None))
}

#[cfg(windows)]
fn windows_commit() -> Result<(Option<u64>, Option<u64>), String> {
    use std::mem::{size_of, zeroed};
    use windows_sys::Win32::System::ProcessStatus::{GetPerformanceInfo, PERFORMANCE_INFORMATION};

    // SAFETY: the structure is initialized to the documented size and passed for the duration of
    // the call. GetPerformanceInfo writes no more than the supplied structure.
    unsafe {
        let mut performance: PERFORMANCE_INFORMATION = zeroed();
        performance.cb = size_of::<PERFORMANCE_INFORMATION>() as u32;
        if GetPerformanceInfo(&mut performance, performance.cb) == 0 {
            return Err(format!(
                "GetPerformanceInfo failed: {}",
                std::io::Error::last_os_error()
            ));
        }
        let page_size = performance.PageSize as u64;
        let limit = (performance.CommitLimit as u64).saturating_mul(page_size);
        let total = (performance.CommitTotal as u64).saturating_mul(page_size);
        Ok((Some(limit), Some(limit.saturating_sub(total))))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(total_bytes: u64, available_bytes: u64) -> MemorySample {
        MemorySample {
            captured_at: Instant::now(),
            total_bytes,
            available_bytes,
            commit_limit_bytes: None,
            available_commit_bytes: None,
        }
    }

    #[test]
    fn reserve_uses_larger_of_floor_and_fraction() {
        assert_eq!(
            sample(16 * 1024 * 1024 * 1024, 0).reserve_bytes(),
            MINIMUM_RESERVE_BYTES
        );
        assert_eq!(
            sample(64 * 1024 * 1024 * 1024, 0).reserve_bytes(),
            64 * 1024 * 1024 * 1024 / 10
        );
    }

    #[test]
    fn admission_and_eviction_use_whole_system_headroom() {
        let gib = 1024 * 1024 * 1024;
        assert!(sample(16 * gib, 8 * gib).permits_load(6 * gib));
        assert!(!sample(16 * gib, 7 * gib).permits_load(6 * gib));
        assert!(sample(16 * gib, 2 * gib).requires_eviction());
        assert!(!sample(16 * gib, 2 * gib + 1).requires_eviction());
    }

    #[test]
    fn windows_commit_headroom_is_an_independent_gate() {
        let gib = 1024 * 1024 * 1024;
        let mut observation = sample(16 * gib, 12 * gib);
        observation.commit_limit_bytes = Some(20 * gib);
        observation.available_commit_bytes = Some(2 * gib);
        assert!(observation.requires_eviction());
        assert!(!observation.permits_load(gib));
    }
}
