//! Buffer contracts and the one placement rule.
//!
//! A contract states what each agent may do with a buffer. [`place`] turns
//! how a buffer is placed under its contract and an executor's kinds into the kinds the
//! buffer is allocated from, from facts alone: it never branches on a
//! backend, vendor, driver or model, and reads no rate. It runs when a
//! revision is built, and a backend runs it over its own kinds for the
//! storage it allocates for itself.

use crate::kind::{HostReads, KindMemory, KindReport, Obtained, Reach, Reserving};
use crate::model::Reservation;

/// What each agent may do with a buffer, which sets the kinds it may lie in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Contract {
    /// Device work reads and writes it in bulk; the host never touches it.
    DeviceOnly,
    /// The host only writes it, and device work reads it: graph inputs,
    /// flight rings, record tables.
    Upload,
    /// The host and device work read and write it in bulk: staging, a host
    /// tier, a bank the host writes in place, outputs the host reads.
    Shared,
    /// The host and device work load and store it a coherent word at a time:
    /// timelines, park and answer words, maps, directories.
    Words,
}

impl Contract {
    /// How many contracts there are.
    pub const COUNT: usize = 4;
    /// Every contract, in declaration order.
    pub const ALL: [Self; Self::COUNT] =
        [Self::DeviceOnly, Self::Upload, Self::Shared, Self::Words];
}

/// A contract an address range backed as it grows may be reserved under:
/// bulk storage, device work's own or the host's and device work's.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ReservedContract {
    DeviceOnly,
    Shared,
}

impl ReservedContract {
    /// How many reserved contracts there are.
    pub const COUNT: usize = 2;
    /// Every reserved contract, in declaration order.
    pub const ALL: [Self; Self::COUNT] = [Self::DeviceOnly, Self::Shared];

    /// The contract it is.
    pub const fn contract(self) -> Contract {
        match self {
            Self::DeviceOnly => Contract::DeviceOnly,
            Self::Shared => Contract::Shared,
        }
    }
}

/// How a buffer is placed: allocated whole under any contract, or reserved,
/// an address range backed as it grows, under a reserved contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Placing {
    Fixed(Contract),
    Reserved(ReservedContract),
}

impl Placing {
    /// How many placings there are.
    pub const COUNT: usize = Contract::COUNT + ReservedContract::COUNT;
    /// Every placing, by contract in declaration order, fixed before
    /// reserved.
    pub const ALL: [Self; Self::COUNT] = [
        Self::Fixed(Contract::DeviceOnly),
        Self::Reserved(ReservedContract::DeviceOnly),
        Self::Fixed(Contract::Upload),
        Self::Fixed(Contract::Shared),
        Self::Reserved(ReservedContract::Shared),
        Self::Fixed(Contract::Words),
    ];

    /// Its position in [`Self::ALL`].
    pub const fn index(self) -> usize {
        match self {
            Self::Fixed(Contract::DeviceOnly) => 0,
            Self::Reserved(ReservedContract::DeviceOnly) => 1,
            Self::Fixed(Contract::Upload) => 2,
            Self::Fixed(Contract::Shared) => 3,
            Self::Reserved(ReservedContract::Shared) => 4,
            Self::Fixed(Contract::Words) => 5,
        }
    }

    /// The contract the buffer is placed under.
    pub const fn contract(self) -> Contract {
        match self {
            Self::Fixed(contract) => contract,
            Self::Reserved(contract) => contract.contract(),
        }
    }
}

/// A device-local heap of at most this many bytes that the host maps is a
/// window onto the device's memory (a BAR without ReBAR), which counts as
/// host RAM.
pub const WINDOW_BYTES: u64 = 256 << 20;

/// Where `kind` lies, in the order the rule prefers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Lies {
    /// In the device's own memory.
    Own,
    /// In host RAM.
    Host,
    /// In a small window onto the device's memory, which counts as host RAM
    /// and is spent after it.
    Window,
}

fn lies(kind: &KindReport) -> Lies {
    match kind.memory {
        KindMemory::DeviceMemory => Lies::Own,
        KindMemory::HostRam => Lies::Host,
        KindMemory::Heap { device_local, bytes, .. } => {
            match (
                device_local,
                kind.reach != Reach::Device && bytes <= WINDOW_BYTES,
            ) {
                (false, _) => Lies::Host,
                (true, true) => Lies::Window,
                (true, false) => Lies::Own,
            }
        }
    }
}

/// Whether the contract's agents reach `kind`'s bytes as it requires. A
/// `DeviceOnly` buffer of a device whose memory is its own lies in that
/// memory.
fn reachable(
    kind: &KindReport,
    memory_is_host: bool,
    contract: Contract,
) -> bool {
    match contract {
        Contract::DeviceOnly => memory_is_host || lies(kind) == Lies::Own,
        Contract::Upload => matches!(kind.reach, Reach::Mapped { .. }),
        Contract::Shared | Contract::Words => {
            kind.reach
                == (Reach::Mapped {
                    reads: HostReads::Cached,
                })
        }
    }
}

/// Whether a reserved buffer of `contract` may use `reservation`: a
/// `DeviceOnly` one any, a `Shared` one only one the host addresses at one
/// base.
fn reserved_by(reservation: &Reservation, contract: ReservedContract) -> bool {
    match contract {
        ReservedContract::DeviceOnly => true,
        ReservedContract::Shared => reservation.host_addressed,
    }
}

fn admitted(
    kind: &KindReport,
    memory_is_host: bool,
    placing: Placing,
) -> bool {
    reachable(kind, memory_is_host, placing.contract())
        && match placing {
            Placing::Fixed(_) => kind.obtained == Obtained::Allocated,
            Placing::Reserved(contract) => {
                matches!(&kind.reservation, Ok(Some(reservation)) if reserved_by(reservation, contract))
            }
        }
}

/// The identifiers of the kinds a buffer placed as `placing` is
/// allocated from, in order, among `kinds` of a device whose memory is the
/// host's or its own, on the heaps their memory names (none on another
/// backend); an allocation lands on the first with room. Empty when no kind
/// is admitted.
///
/// 1. Allocated before imported: imported host pages are admitted only where
///    no allocated kind meets the contract.
/// 2. The least reach the contract needs: `DeviceOnly` takes kinds the host
///    cannot map first.
/// 3. The device's own memory first, then host RAM, then a small window.
/// 4. Ties keep the backend's order.
///
/// One kind is kept per memory: a device whose memory is its own has the
/// first, and one whose memory is the host's the first on each heap.
pub fn place(
    kinds: &[KindReport],
    memory_is_host: bool,
    placing: Placing,
) -> Vec<u32> {
    let contract = placing.contract();
    let mut placed = kinds
        .iter()
        .filter(|kind| admitted(kind, memory_is_host, placing))
        .collect::<Vec<_>>();
    if placed
        .iter()
        .any(|kind| kind.obtained == Obtained::Allocated)
    {
        placed.retain(|kind| kind.obtained == Obtained::Allocated);
    }
    placed.sort_by_key(|kind| {
        (
            contract == Contract::DeviceOnly && kind.reach != Reach::Device,
            lies(kind),
        )
    });
    if !memory_is_host {
        placed.truncate(1);
    }
    let mut memories = Vec::new();
    placed.retain(|kind| {
        let memory = match kind.memory {
            KindMemory::Heap { index, .. } => Some(index),
            KindMemory::DeviceMemory | KindMemory::HostRam => None,
        };
        let first = !memories.contains(&memory);
        memories.push(memory);
        first
    });
    placed.into_iter().map(|kind| kind.id).collect()
}

/// How a buffer of `contract` is reserved among `kinds`: as the first kind
/// [`place`] gives it; else none, or why not where a kind its agents reach
/// could not report its reservation.
pub fn reserving(
    kinds: &[KindReport],
    memory_is_host: bool,
    contract: ReservedContract,
) -> Reserving {
    let first = place(kinds, memory_is_host, Placing::Reserved(contract))
        .first()
        .copied();
    if let Some(id) = first {
        return kinds
            .iter()
            .find(|kind| kind.id == id)
            .expect("a placed kind is one of them")
            .reservation
            .clone();
    }
    kinds
        .iter()
        .filter(|kind| reachable(kind, memory_is_host, contract.contract()))
        .find_map(|kind| kind.reservation.clone().err())
        .map_or(Ok(None), Err)
}

#[cfg(test)]
mod tests {
    use super::Placing;

    /// Each placing's index is its position among them all.
    #[test]
    fn a_placing_indexes_its_position() {
        assert!(Placing::ALL.iter().enumerate().all(|(position, placing)| placing.index() == position));
    }
}
