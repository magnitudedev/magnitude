//! Buffer contracts and the one placement rule.
//!
//! A contract states what each agent may do with a buffer. [`place`] turns
//! a buffer's contract and an executor's kinds into the kinds the buffer is
//! allocated from, from facts alone: it never branches on a
//! backend, vendor, driver or model, and reads no rate. It runs when a
//! revision is built, and a backend runs it over its own kinds for the
//! storage it allocates for itself.

use crate::kind::{HostReads, KindMemory, KindReport, Reach};

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

/// The identifiers of the kinds a buffer of `contract` is allocated from,
/// in order, among `kinds` of a device whose memory is the host's or its
/// own, on the heaps their memory names (none on another backend); an
/// allocation lands on the first with room. Empty when no kind is admitted.
///
/// 1. The least reach the contract needs: `DeviceOnly` takes kinds the host
///    cannot map first.
/// 2. The device's own memory first, then host RAM, then a small window.
/// 3. Ties keep the backend's order.
///
/// One kind is kept per memory: a device whose memory is its own has the
/// first, and one whose memory is the host's the first on each heap.
pub fn place(
    kinds: &[KindReport],
    memory_is_host: bool,
    contract: Contract,
) -> Vec<u32> {
    let mut placed = kinds
        .iter()
        .filter(|kind| reachable(kind, memory_is_host, contract))
        .collect::<Vec<_>>();
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
