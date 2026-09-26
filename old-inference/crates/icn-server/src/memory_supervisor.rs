use std::sync::Mutex;
use std::time::{Duration, Instant};

use sysinfo::{MemoryRefreshKind, Pid, ProcessRefreshKind, ProcessesToUpdate, RefreshKind, System};

pub(crate) const POLL_INTERVAL: Duration = Duration::from_millis(100);
pub(crate) const IDLE_POLL_INTERVAL: Duration = Duration::from_secs(1);
pub(crate) const MONITOR_LOSS_DEADLINE: Duration = Duration::from_secs(1);
pub(crate) const RECOVERY_STABLE_TIME: Duration = Duration::from_secs(5);
pub(crate) const RECOVERY_MARGIN_BYTES: u64 = 512 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct MemorySample {
    pub(crate) captured_at: Instant,
    pub(crate) physical_capacity_bytes: u64,
    pub(crate) physical_available_bytes: u64,
    pub(crate) allocation_capacity_bytes: u64,
    pub(crate) allocation_headroom_bytes: u64,
}

impl MemorySample {
    pub(crate) fn abort_reserve_bytes(self) -> u64 {
        icn_hardware::system_memory_thresholds(self.physical_capacity_bytes).abort_reserve_bytes
    }

    pub(crate) fn permits_load(self, required_system_memory_bytes: u64) -> bool {
        let required = self
            .abort_reserve_bytes()
            .saturating_add(required_system_memory_bytes);
        self.allocation_headroom_bytes > required
    }

    pub(crate) fn requires_eviction(self) -> bool {
        self.allocation_headroom_bytes <= self.abort_reserve_bytes()
    }

    pub(crate) fn recovered(self) -> bool {
        let required = self
            .abort_reserve_bytes()
            .saturating_add(RECOVERY_MARGIN_BYTES);
        self.allocation_headroom_bytes > required
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
            .map_err(|_| "system memory observer lock is poisoned".to_owned())?;
        system.refresh_memory_specifics(MemoryRefreshKind::everything());
        let observation = icn_hardware::normalize_system_memory(
            system.total_memory(),
            system.available_memory(),
        )?;
        Ok(MemorySample {
            captured_at: Instant::now(),
            physical_capacity_bytes: observation.physical_capacity_bytes,
            physical_available_bytes: observation.physical_available_bytes,
            allocation_capacity_bytes: observation.allocation_capacity_bytes,
            allocation_headroom_bytes: observation.allocation_headroom_bytes,
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

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(total_bytes: u64, available_bytes: u64) -> MemorySample {
        MemorySample {
            captured_at: Instant::now(),
            physical_capacity_bytes: total_bytes,
            physical_available_bytes: available_bytes,
            allocation_capacity_bytes: total_bytes,
            allocation_headroom_bytes: available_bytes,
        }
    }

    #[test]
    fn abort_reserve_uses_larger_of_floor_and_fraction() {
        assert_eq!(
            sample(16 * 1024 * 1024 * 1024, 0).abort_reserve_bytes(),
            1024 * 1024 * 1024
        );
        assert_eq!(
            sample(64 * 1024 * 1024 * 1024, 0).abort_reserve_bytes(),
            64 * 1024 * 1024 * 1024 / 20
        );
    }

    #[test]
    fn admission_and_eviction_use_whole_system_headroom() {
        let gib = 1024 * 1024 * 1024;
        assert!(sample(16 * gib, 8 * gib).permits_load(6 * gib));
        assert!(!sample(16 * gib, 7 * gib).permits_load(6 * gib));
        assert!(sample(16 * gib, gib).requires_eviction());
        assert!(!sample(16 * gib, gib + 1).requires_eviction());
    }

    #[test]
    fn normalized_allocation_headroom_is_the_admission_gate() {
        let gib = 1024 * 1024 * 1024;
        let mut observation = sample(16 * gib, 12 * gib);
        observation.allocation_capacity_bytes = 20 * gib;
        observation.allocation_headroom_bytes = gib;
        assert!(observation.requires_eviction());
        assert!(!observation.permits_load(gib));
        assert!(!observation.recovered());

        observation.allocation_headroom_bytes = 2 * gib;
        assert!(observation.recovered());
    }

    #[test]
    fn physical_availability_does_not_override_normalized_allocation_headroom() {
        let gib = 1024 * 1024 * 1024;
        let mut observation = sample(16 * gib, 12 * gib);
        observation.allocation_headroom_bytes = 7 * gib;
        assert!(!observation.permits_load(6 * gib));

        observation.physical_available_bytes = 16 * gib;
        assert!(!observation.permits_load(6 * gib));
    }
}
