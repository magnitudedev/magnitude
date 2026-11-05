//! What every part of serving reads: the load's layers and groups, the
//! directory, the source, and how far the device is retired.

use super::{Step, Steps, low_word};
use crate::experts::{ExchangeTable, Words};
use crate::experts::conveyor::Conveyor;
use crate::experts::directory::{Directory, Point};
use crate::experts::landing::{Landing, TierTensor};
use crate::experts::host_records::Grant;
use crate::experts::map::MapView;
use crate::experts::source::{ExpertSource, Read, Records};
use crate::planning::RegionWords;
use magnitude_kernels::constants;
use std::sync::Arc;

/// One routed tensor of a streamed layer as the model file stores it:
/// expert `e`'s record is `bytes` at `base + e · pitch` of file `file`
/// (`planning::StoredStack`), read into staging pool `pool`.
#[derive(Clone, Copy, Debug)]
pub(in crate::experts) struct StoredTensor {
    pub file: usize,
    pub base: u64,
    pub pitch: u64,
    pub bytes: u64,
    pub pool: usize,
}

/// A streamed layer's words and records.
pub(in crate::experts) struct Layer {
    pub group: usize,
    /// Its experts: its map's extent.
    pub experts: usize,
    /// Its map (`BankPart::SlotMap`) as the device reads it: its one
    /// writer is the bank's records beside a host tier
    /// (`experts::records`), the one store's host records where the host
    /// places (`experts::host_records`).
    pub map: MapView,
    /// Its file and scratch kinds' tables of the exchange
    /// (`BankPart::FileTable`, `ScratchTable`).
    pub file: ExchangeTable,
    pub scratch: ExchangeTable,
    /// Its rings of the layers ahead's hints, a row a lead
    /// (`RegionWords::ring_words` each), or none for the last streamed
    /// layer.
    pub rings: Words,
    /// Per ring row, the layer it hints (`ExpertBankPlan::predicted`): a row
    /// a lead its route has.
    pub hints: Vec<usize>,
    /// Its routing records, one for each step in flight
    /// (`RegionWords::flight`), routing words each
    /// (`RegionWords::routing_words`).
    pub routing: Words,
    /// Its policy words (`BankPart::Policy`): those a run's encoding reads
    /// written between flights (`Residency::publish_policy`), its line words
    /// at any turn (`Residency::publish_lines`).
    pub policy: Words,
    /// Its CPU site's table of the exchange, where it has a site.
    pub site: Option<ExchangeTable>,
    /// Each routed tensor of its record, in record order.
    pub stored: Vec<StoredTensor>,
    /// Its barrier's convert's words, where the host places and so the
    /// device may land its misses.
    pub landing: Option<std::sync::Arc<crate::experts::landing::DeviceWords>>,
    /// The records its scratch lists name.
    pub lists: ListRecords,
}

/// The records a layer's scratch lists name, its device readers' extents
/// ([`crate::experts::ListBounds`]): its group's bank records (`BANK`), its
/// group's host tier records (`TR`, none without a tier) and the
/// conveyor's (`V`).
#[derive(Clone, Copy, Debug)]
pub(in crate::experts) struct ListRecords {
    pub bank: u64,
    pub tier: u64,
    pub conveyor: u64,
}

impl Layer {
    /// The stored bytes of one of its experts.
    pub fn expert_bytes(&self) -> u64 {
        self.stored.iter().map(|stored| stored.bytes).sum()
    }
}

/// One step's routing record at a streamed layer (`routed_route_streamed`):
/// the one reader of its words.
pub(super) struct RoutingRecord<'a> {
    words: &'a Words,
    shape: RegionWords,
    base: usize,
    count: usize,
    /// Whether its route wrote it: a whole layer's writes none.
    written: bool,
}

impl<'a> RoutingRecord<'a> {
    /// Step `sequence`'s record at `layer` among `words`, of records shaped
    /// by `shape`. The route publishes it before anything reads it, and it
    /// holds until the step ends: a record of another step, or of a count
    /// that is none of its choices, is the device's broken invariant.
    pub(super) fn read(words: &'a Words, shape: RegionWords, sequence: Step, layer: usize) -> Self {
        let base = (sequence % shape.flight) as usize * shape.routing_words() as usize;
        let written = words.load(base + constants::RING_SEQUENCE as usize);
        assert_eq!(written, low_word(sequence), "step {sequence}'s routing record at layer {layer} is another step's ({written})");
        let count = words.load(base + constants::RING_COUNT as usize);
        let count = usize::try_from(count)
            .ok()
            .filter(|&count| count as u64 <= shape.choices)
            .unwrap_or_else(|| panic!("step {sequence}'s routing record at layer {layer} counts {count} of {} choices", shape.choices));
        Self { words, shape, base, count, written: true }
    }

    /// A whole layer's, of records shaped by `shape`: its route writes none,
    /// so it holds no choice.
    pub(super) fn unwritten(words: &'a Words, shape: RegionWords) -> Self {
        Self { words, shape, base: 0, count: 0, written: false }
    }

    /// Whether its route wrote it ([`Self::unwritten`] is a whole layer's).
    pub(super) fn written(&self) -> bool {
        self.written
    }

    /// The choices it holds.
    pub(super) fn count(&self) -> usize {
        self.count
    }

    /// Each choice's expert and the source its route resolved it to.
    pub(super) fn choices(&self) -> impl Iterator<Item = (i32, i32)> + '_ {
        let word = move |at: u64| self.words.load(self.base + at as usize);
        (0..self.count as u64).map(move |choice| (word(self.shape.routing_expert(choice)), word(self.shape.routing_source(choice))))
    }
}

/// A bank group's per-record words, over every record its tensors reserve,
/// and the records committed at the start.
pub(in crate::experts) struct Group {
    /// Each record's seed commit (`routed_seed_commit`): the step whose
    /// seed the device copied into it.
    pub commits: Words,
    /// The tensors the host lands file misses into, in its record's order:
    /// its host tier's across a link, the bank's own on unified memory (the
    /// one store).
    pub tier: Option<Vec<TierTensor>>,
    /// The bank records the directory places into (none where landings
    /// land into the bank itself).
    pub records: usize,
}

/// The placed parts' words (`BankPart::placed`) of the places the bank
/// holds (`ExpertBank::hold_places`): every place's scratch flags, and of
/// each streamed layer its group reads, `[records, places]` instances of
/// `E` words each.
pub(in crate::experts) struct PlacedWords {
    pub flags: Vec<Words>,
    pub layers: Vec<PlacedLayerWords>,
}

/// A streamed layer's placed words: its group reads, an instance every
/// `stride` words (`ExpertBankPlan::place_stride`).
pub(in crate::experts) struct PlacedLayerWords {
    pub group_reads: Words,
    pub stride: usize,
}

/// The load's words the watcher writes besides its layers'.
pub(in crate::experts) struct LoadWords {
    /// Each chain place's scratch flags.
    pub flags: Vec<Words>,
    /// Each scratch round's convert list.
    pub lists: Vec<Words>,
    /// The tier list: a request's experts the host tier keeps whole, which
    /// the fill's tier copy reads (`Scratch::fix_rounds`).
    pub tiered: Words,
    pub seeds: Words,
}

/// The bank's extents the watcher serves by.
#[derive(Clone, Copy)]
pub(in crate::experts) struct Extents {
    /// The widest streamed layer's experts, every layer's expert `e` in
    /// conveyor record `e`.
    pub widest: usize,
    /// Records of a scratch round.
    pub round_records: usize,
    /// The request regions' layout.
    pub words: RegionWords,
    /// `K`: a decode row's choices, so choice `c` is row `c / K`'s.
    pub selected: usize,
    /// The staging records reads ahead leave free for demand
    /// (`ExpertBankPlan::demand_records`).
    pub demand: usize,
    /// The tier records each group keeps ready or graced for the reads
    /// ahead (`ExpertBankPlan::tier_reserve`), which hold no expert.
    pub reserve: usize,
    /// Who places into the bank (`ExpertBankPlan::placer`): where the
    /// device does, landings are into tier records, answered as -2 - t;
    /// where the host does, into bank records, answered as themselves.
    pub placer: crate::planning::Placer,
}

/// Records taken for reads that have ended: free once the device is past
/// `until`.
pub(super) struct Taken {
    pub(super) records: Vec<Listed>,
    pub(super) until: Point,
}

/// A record a scratch round listed: the conveyor record expert `expert` of
/// `layer` is held in.
pub(super) struct Listed {
    pub(super) layer: usize,
    pub(super) expert: usize,
}

impl Listed {
    /// Free the record into the conveyor.
    pub(super) fn release(self, conveyor: &mut Conveyor) {
        conveyor.release(self.layer, self.expert)
    }
}

/// A staging pool: its records, which reads land in.
pub(in crate::experts) struct Pool {
    pub staging: Records,
}

/// A point in the device's order, with its position there: the layer runs
/// the device makes before it. Groups run in sequence order, each
/// layer-major over its steps, so point `(g, l, q)` of a group of `k` steps
/// of a bank of `L` layers follows the `g·L` runs of the steps before its
/// group (sequences begin at 1, each step `L` runs), then `l·k` of its own
/// group's, then `q − g`.
#[derive(Clone, Copy, Debug)]
pub(super) struct At {
    pub(super) point: Point,
    position: u64,
}

impl At {
    /// `point` of a group of `steps` steps of a bank of `layers` layers.
    pub(super) fn new(point: Point, steps: u64, layers: usize) -> Self {
        let position = point.group * layers as u64 + point.layer as u64 * steps + (point.sequence - point.group);
        Self { point, position }
    }

    /// The layer runs from `self` to `later`: what a read submitted at
    /// `self` hides of a request at `later`, its lead.
    pub(super) fn runs_until(self, later: Self) -> u64 {
        later
            .position
            .checked_sub(self.position)
            .unwrap_or_else(|| panic!("a read submitted at {:?}, after its request at {:?}", self.point, later.point))
    }
}

/// What every part of serving reads and the requests write: the source,
/// the directory, the layers and groups, and the records taken until the
/// device is past them.
pub(super) struct Core {
    pub(super) source: ExpertSource,
    pub(super) directory: Directory,
    pub(super) layers: Vec<Layer>,
    pub(super) groups: Vec<Group>,
    pub(super) pools: Vec<Pool>,
    pub(super) extents: Extents,
    /// The steps begun, under whose lock every map entry is written
    /// (`Steps::unbegun`).
    pub(super) steps: Arc<Steps>,
    /// The frontier retired through: residency's own record of what it
    /// has retired (`Residency::retire`), never whether a step has ended,
    /// which steps end beside serving (`Steps::frontier`).
    pub(super) retired: super::Step,
    pub(super) taken: Vec<Taken>,
}

impl Core {
    /// Each group's records that hold the bank's experts: its own where the
    /// device places into the bank from a tier, the landed records it keeps
    /// where the bank is the one store (unified memory), less the reserve
    /// it keeps ready for the reads ahead ([`Extents::reserve`]): none where
    /// a group holds no more than its reserve, which then keeps nothing
    /// (`HostRecords::refill_victims`; a plan's group holds its reserve
    /// beside a request's records, `ExpertBankPlan::least_records`).
    pub(super) fn bank_records(&self) -> Vec<usize> {
        match self.extents.placer {
            crate::planning::Placer::Tier => self.directory.records(),
            crate::planning::Placer::Bank => (0..self.directory.records().len())
                .map(|group| (self.directory.host(group).records() as usize).saturating_sub(self.extents.reserve))
                .collect(),
        }
    }

    /// The device is at `progress`: the records taken before it, now free.
    pub(super) fn take_past(&mut self, progress: Point) -> Vec<Listed> {
        let (done, kept): (Vec<_>, Vec<_>) = std::mem::take(&mut self.taken)
            .into_iter()
            .partition(|taken| taken.until < progress);
        self.taken = kept;
        done.into_iter().flat_map(|taken| taken.records).collect()
    }

    /// The device is at `progress`: retire the seeds and tier graces before
    /// it, each seed committed mapped from the first step not yet begun but
    /// a tainted one's (a wait of its step or of an earlier step of its
    /// group ended on its still pulse, `Steps::ending`) (`Directory::retire`).
    pub(super) fn retire(&mut self, progress: Point) {
        let (groups, steps) = (&self.groups, &self.steps);
        self.directory.retire(progress, |group, record| groups[group].commits.load(record.index()), |point| steps.ending(point), steps);
        // A taint is read until every seed of its group has retired.
        let read = self.directory.pending_from().map_or(progress.group, |group| group.min(progress.group));
        steps.untaint_before(read);
    }

    /// The reads that bring expert `expert` of `layer` into staging record
    /// `record` of each of its pools.
    pub(super) fn reads(&self, layer: usize, expert: usize, record: u32) -> Vec<Read> {
        self.reads_into(layer, expert, |pool| (&self.pools[pool].staging, u64::from(record)))
    }

    /// The landing of expert `expert` of `layer` from staging record
    /// `record` of each of its pools into tier record `tier` of its group,
    /// mapped in its tier map (`experts::landing`).
    /// Each group's tier tensors address its host tier's records by
    /// `views` from now (a resize's, each group's record tensors' records
    /// committed after it), dropping the views they held.
    pub(super) fn address_tier(&mut self, views: Vec<Vec<Records>>) {
        for (group, views) in self.groups.iter_mut().zip(views) {
            let tensors = group.tier.as_mut().expect("a host tier resizes only beside its tensors");
            for (tensor, records) in tensors.iter_mut().zip(views) {
                tensor.records = records;
            }
        }
    }

    /// Each group's tier tensors' first `records[group]` records, as views
    /// of their own (`Records::range`): what they address once a shrink
    /// cuts them there, holding no record past the cut.
    pub(super) fn tier_views(&self, records: &[u32]) -> Result<Vec<Vec<Records>>, String> {
        self.groups
            .iter()
            .zip(records)
            .map(|(group, &records)| {
                let tensors = group.tier.as_ref().expect("a host tier resizes only beside its tensors");
                tensors.iter().map(|tensor| tensor.records.range(0, u64::from(records))).collect()
            })
            .collect()
    }

    /// The landing `grant` allows (its expert into its tier record) from
    /// staging record `record` of each of its pools, mapped by the grant
    /// (`experts::landing`).
    pub(super) fn landing(&self, grant: Grant, record: u32) -> Landing {
        self.landing_from(grant, |pool| (&self.pools[pool].staging, u64::from(record)))
    }

    /// The landing `grant` allows (its expert into its tier record) from
    /// the record `from` names of each of its pools (its records and index:
    /// a staging record, or a prompt group's conveyor record), mapped by
    /// the grant (`experts::landing`).
    pub(super) fn landing_from<'a>(&self, grant: Grant, from: impl Fn(usize) -> (&'a Records, u64)) -> Landing {
        let watched = &self.layers[grant.staged().0];
        let tensors = self.groups[watched.group].tier.as_ref().expect("a landing's group has a host tier");
        let landed = grant.record();
        Landing {
            tensors: watched
                .stored
                .iter()
                .zip(tensors)
                .map(|(stored, tensor)| {
                    let (records, record) = from(stored.pool);
                    (records.clone(), record, stored.bytes, tensor.landed(landed))
                })
                .collect(),
            grant,
            from: None,
        }
    }

    /// Copy expert `expert` of `layer`, whose stored form conveyor record
    /// `conveyed` holds whole, into staging record `record` of each of its
    /// pools: the bytes its read into staging would have landed there.
    ///
    /// # Safety
    ///
    /// No read lands in either record and no device work reads either
    /// until it returns.
    pub(super) unsafe fn stage_conveyed(&self, conveyor: &[Records], layer: usize, conveyed: u32, record: u32) {
        for stored in &self.layers[layer].stored {
            // SAFETY: the caller's.
            unsafe {
                self.pools[stored.pool]
                    .staging
                    .copy_record(u64::from(record), &conveyor[stored.pool], u64::from(conveyed), stored.bytes)
            };
        }
    }

    /// The reads that bring expert `expert` of `layer` into record `into`
    /// names (its records and index) of each of its pools, by pool.
    pub(super) fn reads_into<'a>(
        &'a self,
        layer: usize,
        expert: usize,
        into: impl Fn(usize) -> (&'a Records, u64),
    ) -> Vec<Read> {
        self.layers[layer]
            .stored
            .iter()
            .map(|stored| {
                let (records, record) = into(stored.pool);
                let offset = stored.base + expert as u64 * stored.pitch;
                Read::new(stored.file, offset, stored.bytes, records, record)
            })
            .collect()
    }
}
