//! The host side of expert streaming: a load whose
//! routed experts its plan banks (`ExpertBankPlan`) holds them in bank
//! records the host promotes from its host tier, which the host lands file
//! misses into, and the host serves the device's requests while steps run.
//!
//! - `bank`: the bank's device tensors and words, allocated at load, and a
//!   step's lifecycle (`ExpertBank::begin`, `ExpertStep::complete`).
//! - `compute`: the host compute executor, the one owner of every host
//!   thread doing expert work: the CPU site's units and the landings'
//!   conversion pieces.
//! - `conveyor`: a prompt group's cold experts of a layer, read once as
//!   its steps' requests there are published.
//! - `copies`: the device's transfer queue, a step's batches ahead of the
//!   bank's promotions.
//! - `events`: the load's events, which every host job of streaming waits
//!   on instead of a time.
//! - `directory`: where each expert lives (a bank record, a host tier
//!   record, or the file), and the ranking's choice of what the bank keeps.
//! - `host_records`: one bank group's host tier, every miss's first store,
//!   inclusive of the bank, in its resident form.
//! - `landing`: a file miss's landing from staging into the tier.
//! - `map`: each layer's step-stamped map of its bank records, the host its
//!   one writer.
//! - `records`: the bank's records, the one owner of what each holds and
//!   of the map entries naming them: every change is one of its
//!   transitions.
//! - `reuse`: each expert's reuse as the routes' reads measure it, its
//!   half-life among them: the one ranking of every residency decision,
//!   which the directory owns.
//! - `staging`: the landing pool, records that keep the stored form they
//!   were read into until reused.
//! - `exchange`: the one protocol by which the device asks the host for
//!   anything (file misses, the CPU's misses, scratch rounds) and waits for
//!   the answer: its slot tables, their one wait rule, and its watchdog,
//!   the one owner of the exchange's liveness.
//! - `site`: the CPU site, the executor's work computing whole the misses a
//!   decode route assigned to the CPU, from their tier records in place,
//!   beside the device's hits.
//! - `source`: demand reads of expert records from the model file into
//!   records checked once.
//! - `watcher`: the host's serving of running steps, by owner: the request
//!   service (decode requests, scratch rounds and hints, the one poll of
//!   the device's words), residency (the directory, the tier, landings,
//!   promotions, read-ahead and the policy words) and the transfer queue's
//!   completions, one module per responsibility behind its façade.
//!
//! Word layouts and failure rules are `streaming_words.seismic`'s.

mod bank;
mod compute;
mod conveyor;
mod copies;
mod directory;
mod events;
mod exchange;
mod host_records;
mod landing;
mod map;
mod records;
mod reuse;
mod site;
mod source;
mod staging;
mod watcher;
mod yielding;

pub use bank::{ExpertCosts, GroupCost, GroupLayer, LayerTimeline, Span, StepTimeline, LayerCost, LayerExperts, StepCost, STEP_TRACE};
pub(crate) use directory::Ranks;
pub(crate) use compute::{Compute, ComputeThreads, HostCompute, Machine};
pub(crate) use site::{SiteLayer, SiteRecords, SiteScales, SiteShape, SiteTensors, measure as measure_site, measure_cpu_site};
pub use site::{SITE_TERM_NAMES, SiteServed};
#[cfg(test)]
pub(crate) use site::{Answered, ExpertValues, reference as site_reference};
pub(crate) use exchange::ExchangeTable;
pub(crate) use watcher::{ParkKind, ParkSite, Parks, panic_message};
pub(crate) use bank::ChainServing;
pub(crate) use source::Records;
pub use directory::{LayerCurve, MissCurve, RoutedRead};
#[cfg(test)]
pub(crate) use directory::Stack;
#[cfg(feature = "expert-routing")]
pub use directory::{Recency, replay};
pub(crate) use bank::{ExpertBank, ExpertStep, HeldTable, Recording, ServedBank, StepFailure, StepRoute};

use magnitude_kernels::{constants, list_extent};

/// The records a list's entries may name: each entry's from record below
/// `from` and its to record below `to`, the leading extents of the tensors
/// its device reader binds (`routed_copy`'s S and D, `routed_convert`'s V
/// and S, `routed_move`'s D, `routed_seed_commit`'s BANK).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ListBounds {
    pub from: u64,
    pub to: u64,
}

impl ListBounds {
    fn admits(self, from: i32, to: i32) -> bool {
        u64::try_from(from).is_ok_and(|from| from < self.from) && u64::try_from(to).is_ok_and(|to| to < self.to)
    }
}

/// I32 words of a `Tensor::shared` tensor the host exchanges with running
/// steps: acquire loads and release stores.
#[derive(Clone)]
pub(crate) struct Words {
    tensor: seismic::Tensor,
    len: usize,
}

impl Words {
    /// The words of `tensor`, `len` of them, every one `initial`; 8-byte
    /// aligned, so each even word begins a u64 the host accesses whole.
    pub(crate) fn new(tensor: seismic::Tensor, len: u64, initial: i32) -> Result<Self, String> {
        let len = usize::try_from(len).map_err(|_| "bank words exceed the host")?;
        if let Some(base) = tensor.shared_address().filter(|base| base.as_ptr() as usize % 8 != 0) {
            return Err(format!("bank words at {:p}, not 8-byte aligned", base.as_ptr()));
        }
        let words = Self { tensor, len };
        for index in 0..len {
            words
                .tensor
                .store_release(index as u64 * 4, initial as u32)
                .map_err(|error| format!("bank word {index}: {error}"))?;
        }
        Ok(words)
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// Word `index`, loaded with acquire ordering. Every index the host
    /// names lies within the words, which `new` stored once each: neither
    /// the storage nor the offset can be refused.
    pub(crate) fn load(&self, index: usize) -> i32 {
        assert!(index < self.len, "bank word {index} of {}", self.len);
        self.tensor
            .load_acquire(index as u64 * 4)
            .expect("a stored word of host-shared storage") as i32
    }

    /// The words in place, for stores that order nothing.
    pub(crate) fn in_place(&self) -> &[std::sync::atomic::AtomicU32] {
        self.tensor.shared_words().expect("bank words of host-shared storage")
    }

    /// Publish `value` as word `index` (a release store).
    pub(crate) fn store(&self, index: usize, value: i32) {
        assert!(index < self.len, "bank word {index} of {}", self.len);
        self.tensor
            .store_release(index as u64 * 4, value as u32)
            .expect("a stored word of host-shared storage")
    }

    /// The u64 of words `index` and `index + 1` (little-endian, 8-byte
    /// aligned), loaded as one access with acquire ordering.
    pub(crate) fn load_u64(&self, index: usize) -> u64 {
        assert!(index + 1 < self.len, "bank words {index}.. of {}", self.len);
        self.tensor
            .load_acquire_u64(index as u64 * 4)
            .expect("an aligned pair of words of host-shared storage")
    }

    /// Publish `value` as the u64 of words `index` and `index + 1`
    /// (little-endian, 8-byte aligned) as one release store.
    pub(crate) fn store_u64(&self, index: usize, value: u64) {
        assert!(index + 1 < self.len, "bank words {index}.. of {}", self.len);
        self.tensor
            .store_release_u64(index as u64 * 4, value)
            .expect("an aligned pair of words of host-shared storage")
    }

    /// Publish `update` of the u64 of words `index` and `index + 1` as one
    /// read-modify-write, where it returns a value for what they hold; none
    /// leaves them as they are. Whether it published.
    pub(crate) fn update_u64(&self, index: usize, update: impl FnMut(u64) -> Option<u64>) -> bool {
        assert!(index + 1 < self.len, "bank words {index}.. of {}", self.len);
        self.tensor
            .update_release_u64(index as u64 * 4, update)
            .expect("an aligned pair of words of host-shared storage")
    }

    /// Publish a list for step `sequence` (`routed_copy`,
    /// `routed_convert`): its pairs, then its count ([`Self::write_list`]),
    /// then the step's low word, which the device checks before it reads
    /// the rest.
    pub(crate) fn publish_list(&self, sequence: watcher::Step, pairs: &[(i32, i32)], bounds: ListBounds) {
        self.write_list(pairs, bounds);
        self.store(constants::LIST_SEQUENCE as usize, watcher::low_word(sequence));
    }

    /// Write a list's pairs, then its count: a list the device reads only
    /// in work submitted after it (`routed_move`). The device reads every
    /// entry the count names unchecked, so a pair outside `bounds`, or more
    /// pairs than the list holds, is an invariant violation.
    pub(crate) fn write_list(&self, pairs: &[(i32, i32)], bounds: ListBounds) {
        assert!(
            list_extent(pairs.len() as u64) as usize <= self.len,
            "a list of {} entries in {} words",
            pairs.len(),
            self.len
        );
        for (index, &(from, to)) in pairs.iter().enumerate() {
            assert!(bounds.admits(from, to), "list entry {index}, ({from}, {to}), outside {bounds:?}");
            let entry = list_extent(index as u64) as usize;
            self.store(entry + constants::LIST_FROM as usize, from);
            self.store(entry + constants::LIST_TO as usize, to);
        }
        self.store(constants::LIST_COUNT as usize, pairs.len() as i32);
    }
}
