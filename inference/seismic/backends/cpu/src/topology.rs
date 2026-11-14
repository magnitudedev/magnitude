//! The CPU backend's facts in the topology's terms.

use seismic_topology::{HostReads, KindMemory, KindReport, Reach};

/// The backend's one kind: host memory it allocates, which every contract
/// is allocated in.
pub fn kind() -> KindReport {
    KindReport {
        id: 0,
        name: "host memory".into(),
        reach: Reach::Mapped { reads: HostReads::Cached },
        memory: KindMemory::HostRam,
    }
}

#[cfg(test)]
mod tests {
    use seismic_topology::fixtures::{gb10, m4_pro, rtx3090};

    /// The recorded fixtures' CPU kind and largest allocation are the ones
    /// this backend declares.
    #[test]
    fn the_fixtures_cpu_is_the_backends() {
        for reports in [rtx3090(), m4_pro(), gb10()] {
            assert_eq!(super::kind(), reports.cpu.kind);
            assert_eq!(reports.cpu.max_allocation, crate::MAX_ALLOCATION_BYTES);
        }
    }
}
