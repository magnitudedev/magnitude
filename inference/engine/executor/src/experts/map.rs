//! A streamed layer's map (`streaming_words.seismic`'s map): for each of
//! its experts the bank record holding it and the steps it holds it for,
//! `[FROM, UNTIL)`. It has one writer ([`StampedMap`], not `Clone`): beside
//! a host tier the bank's records (`experts::records`), where the host
//! places the one store's landing map ([`LandingMap::Bank`]); every reader
//! holds a [`MapView`]. An entry stored from or until the first step not yet
//! begun is stored under the lock every step begins under
//! ([`Steps::unbegun`]): a step it names as not yet begun begins only after
//! its stores, so the device reads them for every step at or past it.
//!
//! - Map: RECORD, then FROM, then UNTIL all ones, each a release store; the
//!   device reads UNTIL, then FROM, then RECORD, each an acquire load
//!   (`streaming_held`). A step that reads the new UNTIL reads the new FROM
//!   and RECORD; one that reads the old UNTIL either is past it (an
//!   unmapped entry's old record is reused only past every step below it)
//!   or reads a record that holds the expert, old or new. So a mapping may
//!   take effect from a step already begun that has not resolved the
//!   layer yet.
//! - Unmap: UNTIL alone. The record is reused only once every step below
//!   it has ended (the frontier reached it, `Steps::frontier`), or, where the device itself
//!   writes the record (a prompt group's seeds and the conveyor's claims),
//!   once the device is past those steps in its order.
//!
//! A step is one u64 at an even word, stored and loaded as one 64-bit
//! access, so no reader assembles a step the host never held.

use super::Words;
use super::watcher::{Step, Steps, Unbegun};
use magnitude_kernels::constants::{MAP_FROM, MAP_RECORD, MAP_UNTIL, MAP_WORDS};

/// UNTIL while mapped: every step.
const ALWAYS: Step = Step::MAX;

/// A layer's map, `[E, MAP_WORDS]` words, entry `e` its expert `e`'s, as
/// its readers read it: what a step's pass resolves, which the host's hits
/// and telemetry read too. Only its writer ([`StampedMap`]) stores it.
#[derive(Clone)]
pub(in crate::experts) struct MapView {
    words: Words,
}

impl MapView {
    /// Its experts.
    pub fn experts(&self) -> usize {
        self.words.len() / MAP_WORDS as usize
    }

    fn word(expert: usize, word: u64) -> usize {
        expert * MAP_WORDS as usize + word as usize
    }

    fn load(&self, expert: usize, word: u64) -> i32 {
        self.words.load(Self::word(expert, word))
    }

    fn load_step(&self, expert: usize, word: u64) -> Step {
        self.words.load_u64(Self::word(expert, word))
    }

    /// The record the map holds `expert` in while mapped, none unmapped.
    pub fn held(&self, expert: usize) -> Option<usize> {
        let record = usize::try_from(self.load(expert, MAP_RECORD)).ok()?;
        (self.load_step(expert, MAP_UNTIL) == ALWAYS).then_some(record)
    }

    /// The record step `step` holds `expert` in, as the device resolves it
    /// (`streaming_held`): none outside its entry's `[FROM, UNTIL)`. What
    /// the host decides a step's hits by, so it reads no record the
    /// device's step does not, and leaves none unfilled that it reads.
    pub fn resolve(&self, expert: usize, step: Step) -> Option<usize> {
        let until = self.load_step(expert, MAP_UNTIL);
        let from = self.load_step(expert, MAP_FROM);
        (from <= step && step < until)
            .then(|| usize::try_from(self.load(expert, MAP_RECORD)).expect("a mapped entry names a record"))
    }
}

/// A layer's map's one writer. It is not `Clone`: whoever holds it is the
/// one writer of the map, beside a host tier the bank's records
/// (`experts::records`), where the host places the one store's landing map
/// ([`LandingMap::Bank`]); every reader holds a [`MapView`].
pub(in crate::experts) struct StampedMap {
    view: MapView,
}

impl StampedMap {
    /// The map over `words`, every entry unmapped (no record, held for no
    /// step), before any step reads it.
    pub fn new(words: Words) -> Self {
        let map = Self { view: MapView { words } };
        for expert in 0..map.view.experts() {
            map.store(expert, MAP_RECORD, -1);
            map.store_step(expert, MAP_FROM, 0);
            map.store_step(expert, MAP_UNTIL, 0);
        }
        map
    }

    /// What its readers read.
    pub fn view(&self) -> &MapView {
        &self.view
    }

    fn store(&self, expert: usize, word: u64, value: i32) {
        self.view.words.store(MapView::word(expert, word), value);
    }

    fn store_step(&self, expert: usize, word: u64, step: Step) {
        self.view.words.store_u64(MapView::word(expert, word), step);
    }

    /// Map `expert` to `record` for every step from `from` on.
    pub fn map(&self, expert: usize, record: usize, from: Step) {
        let record = i32::try_from(record).expect("bank records fit i32");
        self.store(expert, MAP_RECORD, record);
        self.store_step(expert, MAP_FROM, from);
        self.store_step(expert, MAP_UNTIL, ALWAYS);
    }

    /// Unmap `expert` for every step from `until` on.
    pub fn unmap(&self, expert: usize, until: Step) {
        self.store_step(expert, MAP_UNTIL, until);
    }

    /// Unmap `expert` from `until` on where its entry names `record` for a
    /// later step: an unmapping only ever ends an entry earlier, and never
    /// one that names another record (its expert mapped since, whose old
    /// record no resolution after that store names).
    pub fn unmap_record(&self, expert: usize, record: usize, until: Step) {
        let names = usize::try_from(self.view.load(expert, MAP_RECORD)).is_ok_and(|named| named == record);
        if names && until < self.view.load_step(expert, MAP_UNTIL) {
            self.unmap(expert, until);
        }
    }
}

/// Where a layer's landings map the records they land in: across a link
/// its tier map (`tier[e]`, a record of its group's host tier, or -1); on
/// unified memory its map, which names the bank records the host lands
/// into, each mapped from the first step not yet begun.
#[derive(Clone)]
pub(in crate::experts) enum LandingMap {
    Tier(Words),
    Bank { map: std::sync::Arc<StampedMap>, steps: std::sync::Arc<Steps> },
}

impl LandingMap {
    /// Map `expert` to `record`, or unmap it (none), from the first step
    /// not yet begun.
    pub fn store(&self, expert: usize, record: Option<u32>) {
        match self {
            Self::Tier(_) => self.store_unbegun(expert, record, None),
            Self::Bank { steps, .. } => steps.unbegun(|at| self.store_unbegun(expert, record, Some(at))),
        }
    }

    /// [`Self::store`] where the caller holds the lock every step begins
    /// under, `at` the first step not yet begun (a stamped map's, which a
    /// plain word needs none of).
    pub fn store_unbegun(&self, expert: usize, record: Option<u32>, at: Option<&Unbegun>) {
        match self {
            Self::Tier(tier) => tier.store(expert, record.map_or(-1, |record| record as i32)),
            Self::Bank { map, .. } => {
                let at = at.expect("a stamped map is stored from the first step not yet begun").next();
                match record {
                    Some(record) => map.map(expert, record as usize, at),
                    None => map.unmap(expert, at),
                }
            }
        }
    }

    /// Map `expert` to `record`, its bytes whole, from step `from` on where
    /// given, the first step whose resolution of the map's layer the device
    /// has not reached (`Point::first_unresolved`, a parked step's
    /// landing's), so a step begun that has yet to resolve the layer hits
    /// it; else from the first step not yet begun. A plain word, for every
    /// resolution from now.
    pub fn map_from(&self, expert: usize, record: u32, from: Option<Step>) {
        match self {
            Self::Tier(tier) => tier.store(expert, record as i32),
            Self::Bank { map, steps } => {
                steps.unbegun(|at| map.map(expert, record as usize, from.map_or(at.next(), |from| from.min(at.next()))))
            }
        }
    }

    /// Unmap `expert` from its record `record`, the device parked: from
    /// step `from` on, the first step whose resolution of the map's layer
    /// the device has not reached (`Point::first_unresolved`), or from the
    /// first step not yet begun where that is earlier, where the entry still
    /// names `record` ([`StampedMap::unmap_record`], under the lock a
    /// landing's mapping takes too: a graced record's expert may have been
    /// mapped anew since). A plain word, for every resolution from now: it
    /// is unmapped only from a kept record, which it names (an unmapping of
    /// it is read by every later resolution, so a grace needs no restamp).
    pub fn unmap_from(&self, expert: usize, record: u32, from: Step) {
        match self {
            Self::Tier(tier) => tier.store(expert, -1),
            Self::Bank { map, steps } => steps.unbegun(|at| map.unmap_record(expert, record as usize, from.min(at.next()))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map(experts: u64) -> StampedMap {
        let cpu = crate::test_support::open_backend(&seismic::Devices::discover(&[]).unwrap(), seismic::BackendName::Cpu).unwrap();
        let words = experts * MAP_WORDS;
        let tensor = seismic::Tensor::shared(&cpu, seismic::Element::i32(), &[experts, MAP_WORDS]).unwrap();
        StampedMap::new(Words::new(tensor, words, 7).unwrap())
    }

    /// A new map holds nothing; a mapped entry holds its record until it is
    /// unmapped, its steps held as pairs past 32 bits.
    #[test]
    fn an_entry_holds_its_record_between_its_map_and_its_unmap() {
        let map = map(3);
        let view = map.view().clone();
        assert_eq!((0..3).map(|expert| view.held(expert)).collect::<Vec<_>>(), [None; 3]);
        map.map(1, 5, (1 << 33) + 2);
        assert_eq!(view.held(1), Some(5));
        assert_eq!(view.load_step(1, MAP_FROM), (1 << 33) + 2);
        map.unmap(1, (1 << 33) + 9);
        assert_eq!(view.held(1), None);
        assert_eq!(view.load_step(1, MAP_UNTIL), (1 << 33) + 9);
        assert_eq!(view.load(1, MAP_RECORD), 5, "an unmap leaves the record and FROM");
    }

    /// A map entry's words as the device reads them, against the host's
    /// remapping of an expert unmapped before: its old record `0`, held for
    /// steps `[0, 10)` and reused since (the steps below 10 are past), now
    /// mapped to record `1` from step 5 (a step begun that has not resolved
    /// the layer). Every interleaving of the host's three release stores
    /// with the device's three acquire loads, each word reading old or new
    /// by whether its store came first: a step at 12 must resolve record 1
    /// or nothing, never the reused record 0. The device's order (UNTIL,
    /// FROM, RECORD) against the host's (RECORD, FROM, UNTIL) holds it;
    /// the former order (RECORD read first, FROM and UNTIL stored first)
    /// does not.
    #[test]
    fn no_interleaving_of_a_remap_resolves_a_reused_record() {
        const RECORD: usize = 0;
        const FROM: usize = 1;
        const UNTIL: usize = 2;
        let old = [0u64, 0, 10];
        let new = [1u64, 5, u64::MAX];
        let step = 12;
        // Each merge of the host's stores and the device's loads, as the
        // positions of the loads among the six.
        let torn = |stores: [usize; 3], loads: [usize; 3]| {
            let mut torn = Vec::new();
            for first in 0..6 {
                for second in first + 1..6 {
                    for third in second + 1..6 {
                        let at = [first, second, third];
                        let mut stored = [false; 3];
                        let (mut store, mut load, mut read) = (0, 0, [0u64; 3]);
                        for position in 0..6 {
                            if at.contains(&position) {
                                let word = loads[load];
                                read[word] = if stored[word] { new[word] } else { old[word] };
                                load += 1;
                            } else {
                                stored[stores[store]] = true;
                                store += 1;
                            }
                        }
                        let held = (read[FROM] <= step && step < read[UNTIL]).then_some(read[RECORD]);
                        if held == Some(old[RECORD]) {
                            torn.push(at);
                        }
                    }
                }
            }
            torn
        };
        assert!(torn([RECORD, FROM, UNTIL], [UNTIL, FROM, RECORD]).is_empty(), "the protocol's orders");
        assert!(!torn([FROM, UNTIL, RECORD], [RECORD, UNTIL, FROM]).is_empty(), "the former orders tear");
    }
}
