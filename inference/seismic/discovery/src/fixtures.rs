//! Host samples recorded on real machines, for other crates' tests.

use crate::{ColdPages, Counted, Harm, HostSample, Lag, OwnReclaimed, PressureLevel, Supply, WiredFigures, WiredLimit};
use std::time::{Duration, Instant};

/// The 48 GB M4 Pro (Mac16,11, macOS 15.7.9, build 24G830) idle, read
/// 2026-11-13 by `examples/host_sample.rs` on rig host m4-pro-03: the
/// machine `seismic_topology::fixtures::recorded::m4_pro` describes. Its
/// figures stand at the instant of the call, the counters read as the
/// first snapshot of a process (`host_statistics64`'s whole window). The
/// recording predates the process's reclaimed bytes; a process sampling at
/// its start, with 18 GB free above the target, has none compressed (its
/// own paging is zero), so they stand at zero.
pub fn m4_pro() -> HostSample {
    let at = Instant::now();
    HostSample {
        at,
        page_bytes: 16_384,
        total: 51_539_607_552,
        supply: Supply { free_beyond_target: 18_506_711_040, clean_cache: 29_273_047_040, cold_reclaimable: 0 },
        counted: Counted { at, lag: Duration::from_secs(1) },
        cold: ColdPages { machine: 0, own: None },
        own_reclaimed: Some(OwnReclaimed::NONE),
        wired: Some(WiredFigures { limit: WiredLimit::OsDefault, wire_count: 1_795_801_088, at }),
        harm: Some(Harm { machine: 15_802_606, own: 0 }),
        pressure: Some(PressureLevel::Normal),
        lag: Lag { release: Duration::ZERO, growth: Duration::ZERO },
    }
}

#[cfg(test)]
mod tests {
    /// The recording is a sample of its machine: its supply within its RAM,
    /// the wired count within it, the process's paging within the machine's.
    #[test]
    fn the_m4_pro_recording_lies_within_its_machine() {
        let sample = super::m4_pro();
        assert_eq!(sample.total, seismic_topology::fixtures::M4_PRO_HOST);
        assert!(sample.supply.total() <= sample.total);
        assert!(sample.wired.unwrap().wire_count < sample.total);
        let harm = sample.harm.unwrap();
        assert!(harm.own <= harm.machine);
    }
}
