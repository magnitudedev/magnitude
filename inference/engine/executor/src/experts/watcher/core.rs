//! What every part of serving reads: the load's layers and groups, the
//! directory, the source, and how far the device is retired.

use super::{Steps, low_word};
use super::steps::Owned;
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
/// expert `e`'s record is its `parts`, each read into staging pool `pool`,
/// one after another (`planning::StoredStack`).
#[derive(Clone, Debug)]
pub(in crate::experts) struct StoredTensor {
    pub parts: Vec<StoredPart>,
    pub pool: usize,
}

/// One part of a stored record: expert `e`'s `bytes` at `base + e · pitch`
/// of file `file` (`planning::StoredPart`).
#[derive(Clone, Copy, Debug)]
pub(in crate::experts) struct StoredPart {
    pub file: usize,
    pub base: u64,
    pub pitch: u64,
    pub bytes: u64,
}

impl StoredTensor {
    /// One expert's record as stored: its parts' bytes.
    pub fn bytes(&self) -> u64 {
        self.parts.iter().map(|part| part.bytes).sum()
    }

    /// The reads that bring expert `expert`'s record into record `record`
    /// of `records`: one a part, each where the parts before it end.
    pub fn reads<'a>(&'a self, expert: usize, records: &'a Records, record: u64) -> impl Iterator<Item = Read> + 'a {
        self.parts.iter().scan(0, move |within, part| {
            let at = *within;
            *within += part.bytes;
            Some(Read::new(part.file, part.base + expert as u64 * part.pitch, part.bytes, records, record, at))
        })
    }
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
    /// Each routed tensor of its record, in record order, shared with the
    /// load's read table ([`ReadTable`]).
    pub stored: Arc<[StoredTensor]>,
    /// Its scratch view of each routed tensor, in record order, as the host
    /// addresses it, where it does (on unified memory): a slot no pass
    /// reads may carry a stored expert, read into its base
    /// (`Conveyor::take`).
    pub slots: Option<Vec<Records>>,
    /// Its scratch form of each routed tensor, in record order
    /// (`ExpertBankPlan::layer_scratch_forms`): layers of one form lay each
    /// slot in the same bytes.
    pub scratch_form: Vec<usize>,
    /// Its decode landing launch's landing list (`routed_land`), where the
    /// bank places and so the device converts its misses
    /// (`landing::DeviceLanding`).
    pub landing: Option<std::sync::Arc<crate::experts::landing::LandingList>>,
    /// Each scratch round's landing list (`routed_round`): the layer's own,
    /// so a round's launch at the next layer, which a step reaches before
    /// the host appends for it there, finds no entry of this layer's under
    /// the step's head.
    pub rounds: Vec<crate::experts::landing::LandingList>,
    /// The records its scratch lists name.
    pub lists: ListRecords,
    /// Its scratch request's tier list (the request's experts the host tier
    /// keeps whole, which the fill's tier copy reads) and seed list (a
    /// prompt group's last step's seeds, which its seed copy and commit
    /// read), both stamped with the step they were written for
    /// (`Scratch::fix_rounds`). Each layer's own: a step whose request here
    /// was never answered (its wait expired, or the request failed) finds
    /// only lists of earlier steps here, and copies nothing, where a list
    /// shared by the layers would hold an earlier layer's, stamped with the
    /// same step, whose seeds name records of that layer's group.
    pub tiered: Words,
    pub seeds: Words,
}

/// The records a layer's scratch lists name, its device readers' extents
/// ([`crate::experts::ListBounds`]): its group's bank records (`BANK`) and
/// its group's host tier records (`TR`, none without a tier).
#[derive(Clone, Copy, Debug)]
pub(in crate::experts) struct ListRecords {
    pub bank: u64,
    pub tier: u64,
}

impl Layer {
    /// The stored bytes of one of its experts.
    pub fn expert_bytes(&self) -> u64 {
        self.stored.iter().map(StoredTensor::bytes).sum()
    }
}

/// One step's routing record at a streamed layer (`routed_route_streamed`),
/// as the step's route wrote it: the one reader of its words.
pub(super) struct RoutingRecord {
    /// Each choice's expert and the source its route resolved it to.
    choices: Vec<(i32, i32)>,
    /// Whether its route wrote it: a whole layer's writes none.
    written: bool,
}

/// A routing record's words as they were while its step held its slot
/// ([`RoutingRecord::copy`]), not yet checked: a check that fails panics,
/// which must not happen under the steps' lock.
pub(super) struct CopiedRecord {
    sequence: super::Step,
    written: i32,
    count: i32,
    /// Every choice's words the record has room for.
    choices: Vec<(i32, i32)>,
}

impl RoutingRecord {
    /// The words of the record of the step `owned` holds among `words`, of
    /// records shaped by `shape`, copied whole while it holds it: no later
    /// step of its slot begins meanwhile, so the device writes none of them
    /// (`Steps::while_owned`).
    pub(super) fn copy(owned: &Owned, words: &Words, shape: RegionWords) -> CopiedRecord {
        let sequence = owned.sequence();
        let base = (sequence % shape.flight) as usize * shape.routing_words() as usize;
        let word = |at: u64| words.load(base + at as usize);
        CopiedRecord {
            sequence,
            written: word(constants::RING_SEQUENCE),
            count: word(constants::RING_COUNT),
            choices: (0..shape.choices).map(|choice| (word(shape.routing_expert(choice)), word(shape.routing_source(choice)))).collect(),
        }
    }

    /// The record `copied` holds at `layer`, of records shaped by `shape`.
    /// The route publishes it before anything reads it: a record of another
    /// step, or of a count that is none of its choices, is the device's
    /// broken invariant.
    pub(super) fn read(copied: CopiedRecord, shape: RegionWords, layer: usize) -> Self {
        let CopiedRecord { sequence, written, count, mut choices } = copied;
        assert_eq!(written, low_word(sequence), "step {sequence}'s routing record at layer {layer} is another step's ({written})");
        let count = usize::try_from(count)
            .ok()
            .filter(|&count| count as u64 <= shape.choices)
            .unwrap_or_else(|| panic!("step {sequence}'s routing record at layer {layer} counts {count} of {} choices", shape.choices));
        choices.truncate(count);
        Self { choices, written: true }
    }

    /// A whole layer's: its route writes none, so it holds no choice.
    pub(super) fn unwritten() -> Self {
        Self { choices: Vec::new(), written: false }
    }

    /// Whether its route wrote it ([`Self::unwritten`] is a whole layer's).
    pub(super) fn written(&self) -> bool {
        self.written
    }

    /// The choices it holds.
    pub(super) fn count(&self) -> usize {
        self.choices.len()
    }

    /// Each choice's expert and the source its route resolved it to.
    pub(super) fn choices(&self) -> impl Iterator<Item = (i32, i32)> + '_ {
        self.choices.iter().copied()
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
    /// The tier records each group keeps holding no expert mapped (ready,
    /// graced, or taken by a read ahead's landing in flight,
    /// `HostRecords::reserved`) for the reads ahead
    /// (`ExpertBankPlan::tier_reserve`).
    pub reserve: usize,
    /// Who places into the bank (`ExpertBankPlan::placer`): where the
    /// device does, landings are into tier records, answered as -2 - t;
    /// where the host does, into bank records, answered as themselves.
    pub placer: crate::planning::Placer,
}

/// The records a scratch round of the request at `until` listed: free once
/// its converts have read them, which the device's arrival at the round
/// after proves (its lane `after`, of the request's slot of the layer's
/// scratch table: that round's wait binds the conveyor as the host's, so
/// it follows the converts that read it), or, where no round follows the
/// slot's last or the fill is followed (`ExpertBank::fill_followed`: the
/// device never arrives at the round after, but at the sibling's round 0,
/// which advances progress past `until`), once the device is past `until`.
pub(super) struct Taken {
    pub(super) records: Vec<Listed>,
    pub(super) until: Point,
    pub(super) after: Option<u64>,
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

/// Where every stored expert's reads come from and land, fixed at load: the
/// source, each layer's routed tensors' file ranges (`(layer, expert)` to
/// its ranges, [`StoredTensor`]) and each staging pool's records. Residency
/// reads through it, and so does the request service's look, which submits
/// a request's reads before residency's lock (`decode::Look`).
pub(super) struct ReadTable {
    source: ExpertSource,
    staging: Vec<Records>,
    stored: Vec<Arc<[StoredTensor]>>,
}

impl ReadTable {
    /// `source`'s reads of `layers`' experts into `pools`.
    pub(super) fn new(source: ExpertSource, pools: Vec<Pool>, layers: &[Layer]) -> Self {
        Self {
            source,
            staging: pools.into_iter().map(|pool| pool.staging).collect(),
            stored: layers.iter().map(|layer| layer.stored.clone()).collect(),
        }
    }

    /// The model's files, read through.
    pub(super) fn source(&self) -> &ExpertSource {
        &self.source
    }

    /// Staging pool `pool`'s records.
    pub(super) fn staging(&self, pool: usize) -> &Records {
        &self.staging[pool]
    }

    /// The reads that bring expert `expert` of `layer` into staging record
    /// `record` of each of its pools.
    pub(super) fn reads(&self, layer: usize, expert: usize, record: u32) -> Vec<Read> {
        self.reads_into(layer, expert, |pool| (&self.staging[pool], u64::from(record)))
    }

    /// The reads that bring expert `expert` of `layer` into record `into`
    /// names (its records and index) of each of its pools, by pool.
    pub(super) fn reads_into<'a>(&self, layer: usize, expert: usize, into: impl Fn(usize) -> (&'a Records, u64)) -> Vec<Read> {
        self.stored[layer]
            .iter()
            .flat_map(|stored| {
                let (records, record) = into(stored.pool);
                stored.reads(expert, records, record)
            })
            .collect()
    }
}

/// A request's point in the device's order, with its position there: the
/// layer runs the device makes before it (`Steps::at`). Pieces run in
/// submission order, each layer-major over its steps, so point `(p, l, q)`
/// of a piece of a group from `g` of `k` steps, its layers from `f`, follows
/// the runs before the piece, then `(l − f)·k` of its own, then `q − g`. A
/// group run whole is one piece, whose `g·L` runs before it (of a bank of
/// `L` layers) are the steps' before it (sequences begin at 1, each step
/// `L` runs).
#[derive(Clone, Copy, Debug)]
pub(in crate::experts) struct At {
    pub(in crate::experts) point: Point,
    position: u64,
}

impl At {
    /// `point`, `position` runs into the device's order.
    pub(super) fn new(point: Point, position: u64) -> Self {
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
    pub(super) table: Arc<ReadTable>,
    pub(super) directory: Directory,
    pub(super) layers: Vec<Layer>,
    pub(super) groups: Vec<Group>,
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
    /// Whether group `group`'s layers run their resident forms: whole and
    /// at home (`Directory::group_whole`, `Directory::group_at_home`), and
    /// its records every record its reservation holds
    /// (`ExpertBankPlan::records`, the span its commits words cover), which
    /// a resident form binds directly, so only while every granule is
    /// backed. A shrink that cuts only records keeping no expert (the
    /// tier's reserve above the homes, where the host places) leaves the
    /// group whole and at home, but releases granules no step may bind.
    pub(super) fn group_resident(&self, group: usize) -> bool {
        let directory = &self.directory;
        let records = match self.extents.placer {
            crate::planning::Placer::Tier => directory.records()[group],
            crate::planning::Placer::Bank => directory.host(group).records() as usize,
        };
        directory.group_whole(group) && directory.group_at_home(group) && records == self.groups[group].commits.len()
    }

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

    /// The records taken whose converts the device has run: it arrived at
    /// the round after theirs.
    pub(super) fn take_converted(&mut self) -> Vec<Listed> {
        let layers = &self.layers;
        let (done, kept): (Vec<_>, Vec<_>) = std::mem::take(&mut self.taken).into_iter().partition(|taken| {
            let Point { layer, sequence, .. } = taken.until;
            taken.after.is_some_and(|lane| layers[layer].scratch.requested(sequence, lane))
        });
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
        // A taint is read until every seed of its group has retired, and
        // while the device is not past its group.
        let unpassed = steps.first_unpassed(progress);
        let read = self.directory.pending_from().map_or(unpassed, |group| group.min(unpassed));
        steps.untaint_before(read);
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
        self.landing_from(grant, |pool| (self.table.staging(pool), u64::from(record)))
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
                    (records.clone(), record, stored.bytes(), tensor.landed(landed))
                })
                .collect(),
            grant,
        }
    }

    /// The reads that bring expert `expert` of `layer` into conveyor
    /// record `record` (`Conveyor::target`): of each of its pools, or the
    /// base of a slot of its scratch.
    pub(super) fn reads_to(&self, layer: usize, expert: usize, conveyor: &Conveyor, record: u32) -> Vec<Read> {
        match conveyor.slot_of(record) {
            None => self.table.reads_into(layer, expert, |pool| (&conveyor.pools()[pool], u64::from(record))),
            Some(slot) => {
                let slots = self.layers[layer].slots.as_ref().expect("a slot carries an expert only where the host addresses the scratch");
                self.layers[layer]
                    .stored
                    .iter()
                    .zip(slots)
                    .flat_map(|(stored, records)| stored.reads(expert, records, u64::from(slot)))
                    .collect()
            }
        }
    }
}
