//! Grants: room a process's own growth draws on beside the load, its
//! footprint credited against it at each reading so that growth is never
//! read as others' (§3.1).

use super::admission::{AdmitPolicy, Short};
use super::changes::Claiming;
use super::claims::Claim;
use super::state::{EntryId, State};
use super::view::Ledger;
use super::vocabulary::{CacheKind, CommittedKind, GrantRole, Holding, Need, size};

/// The footprint of a process in a store at a reading: the one way the
/// ledger reads a process other than through its own entries. The process's
/// scope reads the system's ([`SystemFootprints`]); a test scripts its own.
pub trait Footprints: Send + Sync {
    fn footprint(&self, process: seismic::ProcessId, store: &seismic::StoreIdentity) -> seismic::ProcessFootprint;
}

/// The system's footprints (`seismic::process_footprint`): host RAM's
/// footprint, and a GPU's own memory as NVML counts the process there.
pub struct SystemFootprints;

impl Footprints for SystemFootprints {
    fn footprint(&self, process: seismic::ProcessId, store: &seismic::StoreIdentity) -> seismic::ProcessFootprint {
        seismic::process_footprint(process, store)
    }
}

/// A grant's process, while named ([`Grant::attach`]), and the footprint its
/// growth is counted from: none for a measurement job, whose whole
/// footprint is its; for a search in this process, the footprint and the
/// store's allocated bytes as it was named, whose growth since, less what
/// the ledger already charges of it, is its.
#[derive(Clone, Copy, Debug)]
pub(super) struct Granted {
    role: GrantRole,
    process: Option<(seismic::ProcessId, u64, u64)>,
    /// Its footprint read unobserved once, logged then.
    unobserved: bool,
}

impl Granted {
    pub(super) fn new(role: GrantRole) -> Self {
        Self { role, process: None, unobserved: false }
    }
}

/// Room a process's growth draws on beside the load
/// ([`CommittedKind::Grant`]): a committed claim, its process's footprint
/// credited against it at each reading, within its bytes. Dropped, the room
/// is free and nothing is credited.
#[must_use]
pub struct Grant {
    claim: Claim,
}

impl Ledger {
    /// Hold `bytes` for a process's growth as `role`, in `claiming`, a
    /// change's claim phase: free room beside the
    /// guarantee within the boundary, and the cache worth nothing; never
    /// the bank's or the retained prefixes' (`State::take_beside`). Short,
    /// nothing is held, and the claim waits for the next reading.
    pub fn grant(&self, claiming: &Claiming, role: GrantRole, bytes: u64) -> Result<Grant, Short> {
        claiming.of_scope(&self.store.changes);
        let need = Need::Grant { role, bytes };
        let (kind, bytes) = (need.kind(), size(&self.shape, &need));
        let id = self.admit_or_reclaim(AdmitPolicy {
            at: None,
            kind,
            bytes,
            of: |ck: CacheKind| !ck.valued(),
            within_boundary: true,
            allow_guarantee: false,
        })?;
        self.state().grants.insert(id, Granted::new(role));
        Ok(Grant { claim: Claim::held(self.clone(), id, kind) })
    }
}

impl Grant {
    /// Name the process whose footprint the grant is credited by, from now
    /// (a measurement job's once it is started, this one for a search),
    /// until the grant is released: a measurement job's grant is released
    /// only once the job is shown gone, so a reused pid is never credited.
    ///
    /// # Panics
    /// Where it names one while another is named.
    pub fn attach(&self, process: seismic::ProcessId) {
        let ledger = self.claim.ledger();
        let footprint = ledger.store.footprints.footprint(process, &ledger.store.identity);
        let mut state = ledger.state();
        let allocated = state.allocated;
        let granted = state.grants.get_mut(&self.claim.id()).expect("a grant is recorded as one");
        assert!(granted.process.is_none(), "invariant violated: a grant names its process twice");
        granted.process = Some(match (granted.role, footprint) {
            (GrantRole::Measurement, _) => (process, 0, 0),
            (GrantRole::Tuning, seismic::ProcessFootprint::Unobserved) => (process, 0, allocated),
            (GrantRole::Tuning, seismic::ProcessFootprint::Observed(bytes)) => (process, bytes, allocated),
        });
        state.holdings += 1;
    }

    /// Hold what its process has grown by since it was named, where that
    /// is more than the grant holds, in `claiming`: growth that already
    /// happened, so never refused, the cache worth nothing (spare granules,
    /// idle graph slots) yielding what it took of the guarantee's room (a
    /// search's heap, which no claim of its own admits). A grant never takes the bank or
    /// the retained prefixes: they yield only to a relief plan.
    pub fn hold_growth(&self, claiming: &Claiming) {
        let ledger = self.claim.ledger();
        claiming.of_scope(&ledger.store.changes);
        let footprints = ledger.footprints();
        let mut state = ledger.state();
        let Some(grown) = state.grown(&ledger.store.identity, self.claim.id(), &footprints) else {
            return;
        };
        let held = state.entry(self.claim.id()).bytes;
        if grown > held {
            state.resize(self.claim.id(), grown);
            state.keep_guarantee(Holding::Committed(self.claim.kind()), grown - held, |kind| !kind.valued());
        }
    }
}

/// The footprints of the grants' named processes, each read once, outside
/// the ledger's lock (a footprint is a system call, NVML's a driver's):
/// what a reading and a grant's growth credit.
pub(super) type Footprinted = std::collections::BTreeMap<EntryId, seismic::ProcessFootprint>;

impl Ledger {
    /// The footprints of the grants' processes named now, read with the
    /// ledger's lock released: a grant named after this reads as none yet.
    pub(super) fn footprints(&self) -> Footprinted {
        let named = self.state().grants.iter().filter_map(|(id, granted)| granted.process.map(|(process, ..)| (*id, process))).collect::<Vec<_>>();
        named.into_iter().map(|(id, process)| (id, self.store.footprints.footprint(process, &self.store.identity))).collect()
    }
}

impl State {
    /// What grant `id`'s process has grown by, its footprint read in
    /// `footprints`; none before it was named or where its footprint is
    /// unobserved.
    fn grown(&mut self, store: &seismic::StoreIdentity, id: EntryId, footprints: &Footprinted) -> Option<u64> {
        let allocated = self.allocated;
        let granted = self.grants.get_mut(&id).expect("a grant is recorded as one");
        let (process, from, charged) = granted.process?;
        let footprint = *footprints.get(&id)?;
        match footprint {
            // A search's allocations in this store are the ledger's own
            // entries' already: only growth beyond them is the grant's.
            seismic::ProcessFootprint::Observed(bytes) => Some(match granted.role {
                GrantRole::Measurement => bytes,
                GrantRole::Tuning => bytes.saturating_sub(from).saturating_sub(allocated.saturating_sub(charged)),
            }),
            seismic::ProcessFootprint::Unobserved => {
                if !std::mem::replace(&mut granted.unobserved, true) {
                    tracing::info!(target: crate::MEMORY_TRACE, ?process, role = ?granted.role, ?store, "a grant's process footprint is unobserved: nothing of it is credited");
                }
                None
            }
        }
    }

    /// What the grants' processes hold of their grants at a reading: each
    /// one's growth, within its bytes. Read beside the store's own figures,
    /// it is the process's own, never others'.
    pub(super) fn credited(&mut self, store: &seismic::StoreIdentity, footprints: &Footprinted) -> u64 {
        let ids = self.grants.keys().copied().collect::<Vec<_>>();
        ids.into_iter()
            .map(|id| {
                let bytes = self.held[&id].bytes;
                debug_assert!(matches!(self.held[&id].holding, Holding::Committed(CommittedKind::Grant(_))));
                self.grown(store, id, footprints).map_or(0, |grown| grown.min(bytes))
            })
            .sum()
    }
}
