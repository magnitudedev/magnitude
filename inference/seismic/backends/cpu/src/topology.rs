//! The CPU backend's facts in the topology's terms.

use seismic_reservation::{host_page, AllocationFailure};
use seismic_topology::{
    Contract, HostReads, KindMemory, KindReport, Obtained, Reach, Reservation, ReservationChunks,
};

/// Whether the host addresses a reservation's backed bytes in place: they
/// are host pages (`Storage::host_address` is the range's base).
pub const RESERVATIONS_HOST_ADDRESSED: bool = true;

/// What device work reaches a reservation's backed bytes through chunks
/// of: none, kernels read the range at its base.
pub const RESERVATION_CHUNKS: Option<u64> = None;

/// The backend's one kind: host memory it allocates, which every contract
/// is allocated in; the host page it reads to reserve may refuse.
pub fn kind() -> Result<KindReport, AllocationFailure> {
    Ok(kind_of(host_page()?))
}

/// The backend's kind on a host of `granule` bytes a page.
fn kind_of(granule: u64) -> KindReport {
    KindReport {
        id: 0,
        name: "host memory".into(),
        reach: Reach::Mapped { reads: HostReads::Cached },
        obtained: Obtained::Allocated,
        memory: KindMemory::HostRam,
        reservation: Ok(Some(Reservation {
            granule,
            host_addressed: RESERVATIONS_HOST_ADDRESSED,
            chunks: RESERVATION_CHUNKS.map(|unit| ReservationChunks { unit, table: Contract::DeviceOnly }),
        })),
    }
}

#[cfg(test)]
mod tests {
    use seismic_topology::fixtures::{gb10, m4_pro, rtx3090};

    /// The recorded fixtures' CPU kind and largest allocation are the ones
    /// this backend declares, on each fixture's host page.
    #[test]
    fn the_fixtures_cpu_is_the_backends() {
        for reports in [rtx3090(), m4_pro(), gb10()] {
            let granule = reports.cpu.kind.reservation.clone().unwrap().unwrap().granule;
            assert_eq!(super::kind_of(granule), reports.cpu.kind);
            assert_eq!(reports.cpu.max_allocation, crate::MAX_ALLOCATION_BYTES);
        }
    }
}
