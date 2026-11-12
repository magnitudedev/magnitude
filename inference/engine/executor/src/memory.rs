//! Engine-level memory: the store ledgers ([`ledger`]), the one record of
//! what the process holds in each store, and the demand vocabulary an
//! allocation's bytes are stated in by the storage they lie in
//! ([`ClaimDemand`]).

pub mod ledger;
pub mod scratch;

/// The contracts a claim's bytes are placed under: a claim's words are its
/// `Shared` bytes.
pub const CLAIMED: [seismic::Contract; 3] =
    [seismic::Contract::DeviceOnly, seismic::Contract::Upload, seismic::Contract::Shared];

/// What one claim demands of its device's storage: bytes of device, upload
/// and host storage, the part of its device bytes the backend stages
/// through host RAM, and device and host bytes a reservation commits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ClaimDemand {
    /// Device bytes allocated as the backend allocates, which the free space
    /// of the blocks the device holds serves first.
    pub device: u64,
    /// Upload storage bytes: what the host only writes (a reservation's
    /// record table).
    pub upload: u64,
    /// Host storage bytes allocated as the backend allocates: shared
    /// tensors (`Tensor::shared`), which the host reads and writes in place
    /// and the free space of the host blocks the device holds serves first.
    pub shared: u64,
    pub staged: u64,
    /// Device bytes a reservation backs: fresh pages of the device's store,
    /// which no block's free space serves.
    pub reserved: u64,
    /// Host storage bytes a reservation backs (a host tier's): fresh host
    /// pages, which no block's free space serves.
    pub host: u64,
}

impl ClaimDemand {
    /// The storage the claim covers: its device, reserved, upload, shared
    /// and host bytes.
    pub fn bytes(self) -> u64 {
        self.device
            .saturating_add(self.reserved)
            .saturating_add(self.upload)
            .saturating_add(self.shared)
            .saturating_add(self.host)
    }

    /// Exact sum for planning, where overflow must refuse the plan.
    pub fn checked_bytes(self) -> Option<u64> {
        self.device
            .checked_add(self.reserved)?
            .checked_add(self.upload)?
            .checked_add(self.shared)?
            .checked_add(self.host)
    }

    /// Exact composition of independently owned storage demands.
    pub fn checked_plus(self, other: Self) -> Option<Self> {
        Some(Self {
            device: self.device.checked_add(other.device)?,
            upload: self.upload.checked_add(other.upload)?,
            shared: self.shared.checked_add(other.shared)?,
            staged: self.staged.checked_add(other.staged)?,
            reserved: self.reserved.checked_add(other.reserved)?,
            host: self.host.checked_add(other.host)?,
        })
    }

    /// Exact composition of every one of `demands` ([`Self::checked_plus`]).
    pub fn checked_sum(demands: impl IntoIterator<Item = Self>) -> Option<Self> {
        demands.into_iter().try_fold(Self::default(), Self::checked_plus)
    }

    /// A reservation's backing in its tier and its record table, an
    /// ordinary allocation in the table's own placement.
    pub fn reservation(tier: crate::planning::BankTier, charges: seismic::ReservationCharges) -> Self {
        let mut demand = Self::default();
        match tier {
            crate::planning::BankTier::Bank => demand.reserved = charges.backing,
            crate::planning::BankTier::Host => demand.host = charges.backing,
        }
        if let Some(table) = charges.table {
            match table.placement() {
                seismic::Contract::Upload => demand.upload = table.bytes(),
                seismic::Contract::Shared | seismic::Contract::Words => demand.shared = table.bytes(),
                seismic::Contract::DeviceOnly => demand.device = table.bytes(),
            }
        }
        demand
    }

    /// Physical placement totals; reserved backing stays distinct in the
    /// claim itself so ordinary allocator free blocks cannot serve it.
    pub fn storage_bytes(self, kind: seismic::Contract) -> Option<u64> {
        match kind {
            seismic::Contract::DeviceOnly => self.device.checked_add(self.reserved),
            seismic::Contract::Upload => Some(self.upload),
            seismic::Contract::Shared => self.shared.checked_add(self.host),
            // A claim's words are its `shared` bytes: they lie where
            // fixed `Shared` buffers do, by the same admission.
            seismic::Contract::Words => Some(0),
        }
    }

    /// This demand and `other` together.
    pub fn plus(self, other: Self) -> Self {
        Self {
            device: self.device.saturating_add(other.device),
            upload: self.upload.saturating_add(other.upload),
            shared: self.shared.saturating_add(other.shared),
            staged: self.staged.saturating_add(other.staged),
            reserved: self.reserved.saturating_add(other.reserved),
            host: self.host.saturating_add(other.host),
        }
    }
}
