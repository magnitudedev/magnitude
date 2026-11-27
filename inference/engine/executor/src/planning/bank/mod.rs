//! A streamed load's expert bank:
//! the target's routed expert stacks are held as bank records that a decode
//! step fills from the model file when it routes to an expert the bank lacks,
//! instead of being imported whole.
//!
//! Layers whose records have one form (the stored and resident element and
//! matrix of each routed tensor) share a bank group: one slab store of
//! `E × layers` records (its ceiling, `BANK`), a record a slab holding a
//! region of each routed tensor, each slab one granule of the device's
//! granule pool, which the KV slabs share; the leading `slots × layers`
//! records are committed. Every member layer's entries address it
//! (group-wide record indices), and a miss takes a free committed record,
//! or else its least recently used one. Record `l·E + e` is expert `e` of
//! the group's `l`-th layer, its home. A resize claims or returns granules,
//! so nothing bound to the store is formed again.

mod build;
mod conveyor;
mod counts;
mod layout;
mod records;
mod residency;
mod sizing;
mod table;

use super::resources::FlightDepth;
use super::weights::WeightPlan;
use crate::memory::ClaimDemand;
use magnitude_kernels::{SlotTable, constants};
use magnitude_family_contracts::{WeightRole, WeightScope};
use seismic::Element;
use std::num::NonZeroU64;
use table::stored_rows;
pub use conveyor::{ConveyorCarrier, Lender, StagingPool};
pub use counts::BankCounts;
pub use layout::{BankAllocation, BankLayout, BankPart, BankSubstrate, BankTier, HostTier, Parking, PlacedAllocations, Placer};
pub use records::{PoolSlab, RecordForm, RecordTensor, RecordTensors, RoutedTensor, routed_experts};
pub(crate) use records::{decode_set, layer_homes};
pub use residency::ResidencyOrder;
pub use sizing::growth_target;
pub(crate) use sizing::releasing;
pub use table::BankTable;

/// Items of which there is at least one: made of one and grown, or refused
/// empty, so every extreme over them exists. A group's member layers
/// ([`BankGroup::layers`]); the bank's layers themselves are
/// [`BankLayers`], which also hold each layer's experts positive.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NonEmpty<T>(Vec<T>);

impl<T> NonEmpty<T> {
    pub fn one(item: T) -> Self {
        Self(vec![item])
    }

    /// `items`, none where there are none.
    pub fn new(items: Vec<T>) -> Option<Self> {
        (!items.is_empty()).then_some(Self(items))
    }

    pub fn push(&mut self, item: T) {
        self.0.push(item);
    }

    /// The greatest `key` of the items.
    pub fn max_of<K: Ord>(&self, key: impl Fn(&T) -> K) -> K {
        self.0.iter().map(key).max().expect("a NonEmpty holds an item by construction")
    }
}

impl<T> std::ops::Deref for NonEmpty<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        &self.0
    }
}

/// Routed layers whose records share one form.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BankGroup {
    pub record: RecordTensors<RecordTensor>,
    /// Its member layers (indices of `ExpertBankPlan::layers`), in decoder
    /// order: the order of their home records.
    pub layers: NonEmpty<usize>,
}

/// A routed layer whose experts the bank holds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BankLayer {
    pub scope: WeightScope,
    pub group: usize,
    /// `E`: its experts, its slot map's extent and its home records' count.
    pub experts: u64,
    /// `K`: the experts a row selects in this layer.
    pub selected: u64,
    /// Its expert stacks as a resident load plans them, one of each routed
    /// tensor of its record: what tuning imports to measure the entries
    /// both loads share.
    pub resident: RecordTensors<WeightPlan>,
    /// For fused routed layers whose `routed_output` kernel fuses `SharedDown`
    /// with `ExpertDown`, the `SharedDown` weight's `(resident, resident_bytes,
    /// shape)` key so layers with distinct `SharedDown` quants/shapes get
    /// distinct `RoutedKind`s (`Self::routed_kinds`). `None` for non-fused
    /// layers where `SharedDown` runs in its own `dense_output` call.
    pub shared_down: Option<(Element, u64, Vec<u64>)>,
}

/// One expert stack as its model file stores it: `experts` experts of one
/// routed tensor of one layer, each one's record its `parts` read in turn
/// (`Element::record_parts`: its codes, then under a grid of block scales
/// its grid, which the file keeps as a tensor of its own).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StoredStack {
    pub experts: u64,
    pub parts: Vec<StoredPart>,
}

impl StoredStack {
    /// The range of its file each part's experts lie in, in order: the path,
    /// the offset and the bytes of the experts' whole stored tensor.
    pub fn file_ranges(&self) -> impl Iterator<Item = (std::path::PathBuf, u64, u64)> + '_ {
        self.parts.iter().map(|part| (part.path.clone(), part.offset, self.experts * part.pitch))
    }
}

/// One part of a stored stack: the stored tensor from `offset` in `path`,
/// each expert's `pitch` bytes in turn; each expert's `bytes` of the part
/// lie `first` bytes into its own. A stack stored whole reads its experts
/// whole (`first` 0); a range of each stored expert's rows
/// (`ImportTransform::Rows`, Gemma 4's gate‖up halves) from the range's
/// first row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StoredPart {
    pub path: std::path::PathBuf,
    pub offset: u64,
    pub pitch: u64,
    pub first: u64,
    pub bytes: u64,
}

/// A streamed load's expert bank.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExpertBankPlan {
    /// Its structure and the lending derived from it (`conveyor::Structured`):
    /// the experts a row selects, its host tier, its groups and layers
    /// ([`Self::selected`], [`Self::host`], [`Self::groups`], [`Self::layers`]), with its conveyor
    /// and scratch ([`Self::conveyor`], [`Self::scratch`]) and its pools'
    /// granules ([`Self::granule`], [`Self::host_granule`]).
    structured: conveyor::Structured,
    /// The backend whose launches serve it ([`Self::substrate`]).
    backend: seismic::BackendName,
    /// Whether the target device exposes a distinct side queue for async
    /// transfers / fill DMA beside compute ([`Self::substrate`]).
    has_side_queue: bool,
    /// How its device makes native allocations ([`Self::substrate`]).
    allocations: seismic::NativeAllocations,
    /// Per-distinct-granule pool blocks for the bank's records pools (and a
    /// linked host tier's, which registers large-page blocks at the same
    /// granularity), indexed by `structured.granules()`: what each pool's
    /// granules are charged in ([`Self::resident_charge`], [`Self::host_charge`]).
    blocks: Vec<seismic::BlockSize>,
    /// `S`: the committed records each routed layer adds to its group, at
    /// least the floor ([`Self::floor`], which may be none: each group then
    /// holds its least records, [`Self::least_records`]) and at most its
    /// experts (a layer of fewer experts holds every one of them,
    /// [`Self::layer_slots`]): a bank of the widest layer's experts in slots
    /// holds every expert.
    pub slots: u64,
    /// The most slots growth reaches: every expert, or what a weight
    /// cap bounds of its weights (`ExecutionPlanDraft::with_held_experts`),
    /// the one owner of the slots' upper bound; at least `slots`.
    pub ceiling: u64,
    /// Whether the load holds its decode blocks' resident form
    /// (`RunStage::Resident`): only a bank planned whole, which starts in
    /// it, its forms reading its home records through their tables, as
    /// every backend's routed launches read expert stacks. A bank planned
    /// banked commits no bytes for a form it may never reach, and serves
    /// banked forms whatever it grows to.
    pub resident: bool,
    /// The most host slots a weight cap leaves its host tier beside the
    /// bank's weights at its ceiling (`ExecutionPlanDraft::bank_ceiling`),
    /// at most the widest layer's experts: a cap bounds every weight byte
    /// the load holds, wherever it lies; none uncapped
    /// ([`Self::host_ceiling`]).
    pub host_cap: Option<u64>,
    /// The host slots of the tier's load-time size: as many as the host's
    /// memory admits beside the load, up to every expert
    /// (`ExecutionPlanDraft::with_host_tier`), which the bank backs from
    /// its open on and the watcher's fill fills as each slot joins; none
    /// without a linked tier. Growth and shrink change it only where the
    /// memory's budget changes.
    pub host_slots: u64,
    /// The most chain places a prompt group may take, one for every chunk of
    /// a whole-context prompt (`ExecutionPlanDraft::prompt_places`). Each
    /// place's own step word, scratch flags and request regions and group reads
    /// (`BankPart::placed`) are held for the places a group holds
    /// (`Self::placed`), the floor's one at the load, so the bank's static
    /// charge is the same whichever chain the plan admits.
    pub places: u64,
    /// The most chained passes one banked native head launch executes
    /// (`PlannedMethod::head_steps` when the bank streams a `HeadSublayer`,
    /// otherwise `1`): ensures the routing-record ring (`Self::flight`) fits
    /// every head group in flight even when `places < head_steps`.
    pub head_steps: u64,
    /// The launches one merged-expert prompt pass covers
    /// ([`crate::PassLaunches`]): what the pass buffers and the pass graphs
    /// are sized by, at most `places`. Built at every place; the plan's
    /// settle sizes it by what its fewer passes save
    /// (`ExecutionPlanDraft::settle_bank`).
    pub pass: crate::PassLaunches,
    /// The in-flight depth its steps fly at: the routed kernels' `FLIGHT`
    /// ([`FlightDepth::steps`]) and every exchange table's slots follow it.
    pub flights: FlightDepth,
    /// The stored tables whose slots the bank holds beside its experts, the
    /// load's one elastic owner pricing each by what it saves: every one of
    /// the load's the device can hold so.
    pub tables: Vec<BankTable>,
    /// Which layers hold a CPU site's words and buffers ([`Self::sited`]).
    pub sites: BankSites,
    /// Its model store's path as the plan priced it ([`BankPaths`]).
    pub paths: BankPaths,
    /// Whether the load traces its routes (`ModelLoadPlan::recorded_routes`,
    /// `watcher::routes`): every layer keeps its key records beside its
    /// routing records (`BankPart::Keys`) whatever it traces, since
    /// placement ranks by them.
    pub recording: bool,
    /// Precomputed 1D marginal-value VRAM residency schedule across all bank
    /// groups (`0..=max_slots`), frozen into `BankCounts` when the bank opens.
    residency: ResidencyOrder,
}

/// A bank's model store's path as its plan priced it: its stored read line
/// and the free records its moves need at the plan's slots
/// (`BankPrices::paths`), its reads ahead's staging
/// ([`ExpertBankPlan::hint_records`]) and the tier's reserve
/// ([`ExpertBankPlan::tier_reserve`]); or unpriced, a bank planned before
/// the machine's figures were measured, which reads nothing ahead.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BankPaths {
    Unpriced,
    Priced { line: magnitude_cost_model::ReadLine, records: magnitude_cost_model::PathRecords },
}

impl BankPaths {
    /// The free records its path needs: none unpriced, or where its line
    /// gives no rate (`PathRecords::of`).
    pub fn records(&self) -> magnitude_cost_model::PathRecords {
        match self {
            Self::Unpriced => magnitude_cost_model::PathRecords::NONE,
            Self::Priced { records, .. } => *records,
        }
    }

    /// The stored read line it was priced on.
    pub fn line(&self) -> Option<magnitude_cost_model::ReadLine> {
        match self {
            Self::Unpriced => None,
            Self::Priced { line, .. } => Some(*line),
        }
    }
}

impl BankLayer {
    /// Whether this streamed layer belongs to a drafter: a separate draft
    /// model (`WeightScope::DraftSublayer`) or an embedded causal head
    /// (`WeightScope::HeadSublayer`).
    pub fn is_draft(&self) -> bool {
        matches!(self.scope, WeightScope::DraftSublayer(_) | WeightScope::HeadSublayer(_))
    }

    /// The VRAM records this layer commits at `slots`: every streamed layer
    /// (target or draft) receives its per-layer share up to its expert count.
    pub fn vram_slots(&self, slots: u64) -> u64 {
        slots.min(self.experts)
    }

    /// The sublayer it routes in: a target routed sublayer, a parallel
    /// block's routed branch, an embedded head's routed sublayer, or a
    /// separate draft's routed sublayer (`routed_layers`).
    pub fn sublayer(&self) -> magnitude_family_contracts::SublayerIndex {
        match self.scope {
            WeightScope::TargetSublayer(sublayer)
            | WeightScope::TargetBranch { sublayer, .. }
            | WeightScope::HeadSublayer(sublayer)
            | WeightScope::DraftSublayer(sublayer) => sublayer,
            other => panic!("{other:?} streams its experts, which only a target, head, or draft routed layer does"),
        }
    }
}

/// Which of a bank's streamed layers have a CPU site, holding its words and
/// buffers (`BankPart::SiteTable`, `SiteRows`, `SiteOutput`). A layer is
/// sited once the precision system admits its CPU
/// arithmetic against the reference (`native::BankSite::admitted`), which
/// only its qualification, after its programs are prepared, decides;
/// the plan the load's assessment fits is formed before that, and admission
/// only removes sites, so the charge follows it from above.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BankSites {
    /// Before qualification: every layer of a bank with a host tier, the
    /// most admission can leave.
    Unadmitted,
    /// After qualification ([`ExpertBankPlan::with_sites`]): each layer's
    /// site.
    Admitted(Vec<LayerSite>),
}

/// A streamed layer's CPU site after its qualification.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LayerSite {
    /// The precision system admitted its CPU arithmetic: the CPU serves the
    /// misses its route assigns it.
    Served,
    /// Its verdict is the measurement job's to store (`SiteRequest`): the
    /// layer holds its site, charged and served as `Served`'s is, and every
    /// miss goes to the GPU until a pricing reads the verdict.
    Pending,
    /// The layer's entries serve a site, but its qualification refused it
    /// (why: the precision system refused its CPU arithmetic against the
    /// reference, the pool failed its request, or the numerical profile
    /// approves no CPU site): the layer has no site and reads every miss
    /// over the link.
    Refused(String),
    /// The layer can hold no site (no host tier across a link, or a backend
    /// without the site's entries).
    Unserved,
}

/// A bank's streamed routed layers, in decoder order: at least one, each of
/// at least one expert, so the most any has is positive ([`Self::widest`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BankLayers {
    layers: Vec<BankLayer>,
    widest: NonZeroU64,
}

impl BankLayers {
    /// `layers`, where there is one and each has experts.
    pub fn new(layers: Vec<BankLayer>) -> Option<Self> {
        let widest = layers.iter().map(|layer| NonZeroU64::new(layer.experts)).collect::<Option<Vec<_>>>()?;
        let widest = widest.into_iter().max()?;
        Some(Self { layers, widest })
    }

    /// The most experts of any of the layers.
    pub fn widest(&self) -> NonZeroU64 {
        self.widest
    }
}

impl std::ops::Deref for BankLayers {
    type Target = [BankLayer];

    fn deref(&self) -> &[BankLayer] {
        &self.layers
    }
}

impl<'a> IntoIterator for &'a BankLayers {
    type Item = &'a BankLayer;
    type IntoIter = std::slice::Iter<'a, BankLayer>;

    fn into_iter(self) -> Self::IntoIter {
        self.layers.iter()
    }
}

/// What a streamed routed layer's bank entries are specialized to: its
/// group's record form and records, and the bank's word and staging
/// extents.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BankBinding {
    pub record: RecordTensors<RecordTensor>,
    /// `BANK`.
    pub records: u64,
    /// `E`: the layer's experts.
    pub experts: u64,
    /// `K`.
    pub selected: u64,
    /// `T`.
    pub staging: u64,
    /// `L` of a convert round.
    pub round_records: u64,
    /// The widest streamed layer's experts: the scratch flags' and the
    /// seed list's extents, which every layer shares in turn.
    pub widest: u64,
    /// `W`: scratch window slots (`ExpertBankPlan::window_slots`).
    pub window_slots: u64,
    /// `RW`: convert rounds per window (`ExpertBankPlan::window_rounds`).
    pub window_rounds: u64,
    /// `V`: the conveyor's records ([`ExpertBankPlan::conveyor_records`]).
    pub conveyor: u64,
    /// `TR`: the group's host tier records ([`ExpertBankPlan::host_extent`]),
    /// none without a tier.
    pub tier: u64,
    /// Who places into the group's bank ([`ExpertBankPlan::placer`]): the
    /// device's tier copies of a prompt group exist only beside a host
    /// tier.
    pub placer: Placer,
    /// The layout of a layer's request regions.
    pub words: RegionWords,
}

impl BankBinding {
    /// `F`: the layer's experts' features, the rows of their up matrix
    /// `[F, H]` (the streamed route's `F`, a bank record's device extent).
    pub fn features(&self) -> u64 {
        self.record.up.matrix[0]
    }
}

/// The sized words of a streamed layer's request regions (`routed.seismic`'s
/// streaming section), at the indices `streaming_words.seismic` declares
/// (`magnitude_kernels::constants`), its tables of the exchange laid out by
/// the exchange's one layout (`magnitude_kernels::SlotTable`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct RegionWords {
    /// `P`: the choices of the decode form's widest launch.
    pub choices: u64,
    /// Scratch rounds, each a convert's.
    pub rounds: u64,
    /// `FLIGHT`: the steps in flight ([`FlightDepth::steps`]), each with a
    /// routing record of its own, the one of its sequence modulo them.
    pub flight: u64,
    /// The site kind's slots: the groups in flight
    /// ([`FlightDepth::groups`]).
    pub sites: u64,
}

impl RegionWords {
    /// The file kind's table (`file[FLIGHT, FW]`): a slot for each step in
    /// flight, as its routing records, of every choice.
    pub fn file(&self) -> SlotTable {
        SlotTable::file(self.flight, self.choices)
    }

    /// The site kind's table (`site[FLIGHT, R]`, `routed_site_request`'s and
    /// the sited combines' `FLIGHT` its slots), one slot for each group in
    /// flight: the device runs steps in order and a sited choice's slot
    /// lives only from its request until its sum has read its answer,
    /// before any later step's request at the layer, so slots need not
    /// cover a prompt group's every place. Not the routing records' steps
    /// in flight ([`Self::flight`]), which grow with the context.
    pub fn site(&self) -> SlotTable {
        SlotTable::site(self.sites, self.choices)
    }

    /// The scratch kind's table (`scratch[FLIGHT, SCRATCH_SLOT LANES]`): a
    /// slot for each step in flight of the request's lane and each round's.
    pub fn scratch(&self) -> SlotTable {
        SlotTable::scratch(self.flight, self.rounds)
    }

    /// `RING`: a hint ring's words, an entry (its expert, rank, batch row
    /// and gate weight, HINT_ENTRY_WORDS) for every choice (HINT_ENTRY).
    pub fn ring_words(&self) -> u64 {
        constants::HINT_ENTRY(self.choices)
    }

    /// A decode landing list's words (LANDING_ENTRY): its head, then an entry
    /// for every choice a decode request may name absent (`routed_land`'s
    /// `landing`).
    pub fn landing_words(&self) -> u64 {
        constants::LANDING_ENTRY(self.choices)
    }

    /// A routing record's words (ROUTING_WORDS): each choice's expert and
    /// each choice's source (`routed_route_streamed`'s `routing`).
    pub fn routing_words(&self) -> u64 {
        constants::ROUTING_WORDS(self.choices)
    }

    /// The word of a routing record holding choice `choice`'s source as its
    /// route resolved it, after every choice's expert (ROUTING_SOURCE).
    pub fn routing_source(&self, choice: u64) -> u64 {
        constants::ROUTING_SOURCE(self.choices, choice)
    }
}

impl ExpertBankPlan {
    /// The urgent request its decode steps wait on: a row's selected experts
    /// of one layer, each a record of the widest group (the most stored
    /// bytes) stored as its routed tensors' ranges, which the read engine
    /// reads its files for (`seismic_io::Urgent`).
    pub fn urgent(&self) -> Result<seismic_io::Urgent, String> {
        let mut widest = None::<(u64, Vec<u64>)>;
        for group in self.groups() {
            let record = &group.record;
            let mut parts = Vec::new();
            for tensor in record.gate.iter().chain([&record.up, &record.down]) {
                parts.extend(tensor.stored_parts()?);
            }
            let bytes = parts.iter().sum::<u64>();
            if widest.as_ref().is_none_or(|(most, _)| bytes > *most) {
                widest = Some((bytes, parts));
            }
        }
        let (_, parts) = widest.ok_or("a bank of no group")?;
        Ok(seismic_io::Urgent { parts, records: self.selected() })
    }
}

impl ExpertBankPlan {
    /// Where each streamed layer's expert stacks are stored in `artifact`,
    /// in decoder order, one of each routed tensor of its record: the file,
    /// and where every expert's record lies in it. A stored tensor holds the
    /// layer's experts at a fixed pitch, and a stack of a range of each
    /// one's rows (`ImportTransform::Rows`, admitted by `record_tensor`)
    /// reads them from the range's first row.
    pub fn stored_stacks(
        &self,
        artifact: &magnitude_artifacts::gguf::GgufArtifact,
        draft: Option<&magnitude_artifacts::gguf::GgufArtifact>,
    ) -> Result<Vec<RecordTensors<StoredStack>>, String> {
        self.layers()
            .iter()
            .map(|layer| {
                let source_artifact = match layer.scope {
                    WeightScope::DraftSublayer(_) => {
                        draft.ok_or("a banked draft layer has no draft artifact")?
                    }
                    _ => artifact,
                };
                let record = &self.groups()[layer.group].record;
                layer.resident.as_ref().try_map(|kind, weight| {
                    let stored = crate::residency::Stored::from_gguf(source_artifact, &weight.descriptor)
                        .map_err(|error| error.to_string())?;
                    let name = &weight.descriptor.name;
                    let tensor = record.get(kind).ok_or("a layer's stacks disagree with its group's record")?;
                    let start = stored_rows(&weight.descriptor)
                        .ok_or_else(|| format!("{name:?}: a banked stack is stored whole or as a row range"))?;
                    // Each part of the record is a stored tensor of its own:
                    // the codes, then under a grid the grid's.
                    let (source, offset, bytes) = stored.file_range();
                    let files = std::iter::once((source, offset, bytes))
                        .chain(stored.grid().map(|grid| (&grid.source, grid.offset, grid.nbytes)));
                    let record_parts = tensor.stored_parts()?;
                    let stored_files = 1 + usize::from(stored.grid().is_some());
                    if stored_files != record_parts.len() {
                        return Err(format!("{name:?} stores {stored_files} parts, its record has {}", record_parts.len()));
                    }
                    let parts = files
                        .zip(&record_parts)
                        .map(|((source, offset, bytes), &record_bytes)| {
                            if bytes % layer.experts != 0 {
                                return Err(format!("{name:?} stores {bytes} bytes, not {} experts", layer.experts));
                            }
                            let first = start
                                .checked_mul(record_bytes / tensor.matrix[0])
                                .ok_or_else(|| format!("{name:?}: its first row's offset overflows"))?;
                            let pitch = bytes / layer.experts;
                            if first.checked_add(record_bytes).is_none_or(|end| end > pitch) {
                                return Err(format!("{name:?}: its rows lie past each stored expert"));
                            }
                            Ok(StoredPart { path: source.path().to_owned(), offset, pitch, first, bytes: record_bytes })
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    Ok(StoredStack { experts: layer.experts, parts })
                })
            })
            .collect()
    }

    /// Whether streamed layer `layer` has a CPU site ([`BankSites`]): only
    /// beside a host tier across a link ([`Self::tiered`]), whose records
    /// the site reads; unified memory has none.
    pub fn sited(&self, layer: usize) -> bool {
        self.tiered()
            && match &self.sites {
                BankSites::Unadmitted => true,
                BankSites::Admitted(sites) => matches!(sites[layer], LayerSite::Served | LayerSite::Pending),
            }
    }

    /// Whether a decode step runs the unified zero-switch 3-tier pipeline on
    /// this bank (`experts::decode_dma_enabled`): only a tier-placed bank that
    /// is not whole. Single owner shared by the bank's transport and graphs.
    pub fn decode_dma(&self, backend: seismic::BackendName) -> bool {
        crate::experts::decode_dma_enabled(backend) && self.placer() == Placer::Tier && !self.whole()
    }
    /// The streamed layers whose CPU site their qualification refused, and
    /// why ([`LayerSite::Refused`]): each reads every miss over the link.
    pub fn refused_sites(&self) -> Vec<(usize, String)> {
        match &self.sites {
            BankSites::Unadmitted => Vec::new(),
            BankSites::Admitted(sites) => sites
                .iter()
                .enumerate()
                .filter_map(|(layer, site)| match site {
                    LayerSite::Refused(reason) => Some((layer, reason.clone())),
                    _ => None,
                })
                .collect(),
        }
    }

    /// The plan with each layer's site as its qualification left it
    /// (`sites`, one for each layer): what its sites' words and buffers are
    /// then charged for. A site is served only where the plan may hold one
    /// ([`Self::sited`]).
    pub fn with_sites(self, sites: Vec<LayerSite>) -> Result<Self, String> {
        if sites.len() != self.layers().len() {
            return Err(format!("{} layers' sites admitted for a bank of {} layers", sites.len(), self.layers().len()));
        }
        if let Some(layer) = (0..sites.len()).find(|&layer| matches!(sites[layer], LayerSite::Served | LayerSite::Pending) && !self.sited(layer)) {
            return Err(format!("a CPU site admitted for streamed layer {layer}, which can hold none"));
        }
        Ok(Self { sites: BankSites::Admitted(sites), ..self })
    }

    /// The binding of streamed layer `layer`'s bank entries.
    pub fn binding(&self, layer: usize) -> Result<BankBinding, String> {
        let group = self.layers()[layer].group;
        let conveyor = &self.conveyor().conveyor;
        Ok(BankBinding {
            record: self.groups()[group].record,
            records: self.slab_records(group),
            experts: self.layers()[layer].experts,
            selected: self.layers()[layer].selected,
            staging: self.unified_staging_records_with_paths(self.paths),
            round_records: self.round_records(),
            widest: self.widest().get(),
            window_slots: self.window_slots(),
            window_rounds: self.window_rounds(),
            conveyor: conveyor.records,
            tier: self.host_extent(group),
            placer: self.placer(),
            words: RegionWords {
                choices: self.table_choices(),
                rounds: self.rounds(),
                flight: self.flight(),
                sites: self.site_slots(),
            },
        })
    }

    /// The most steps any single group submitted to this bank can contain:
    /// either a prompt chain of up to `places` launches or a banked native
    /// head launch of up to `head_steps` chained passes.
    pub fn group_steps(&self) -> u64 {
        self.places.max(self.head_steps)
    }

    /// The steps in flight (`FLIGHT`) this bank's routing records, keys, file
    /// and scratch exchange tables hold at `places` prompt chain places:
    /// `flights.groups()` concurrent groups of up to `places.max(head_steps)`
    /// steps each.
    pub fn flight_at(&self, places: u64) -> u64 {
        self.flights.steps(places.max(self.head_steps))
    }

    /// The steps in flight (`FLIGHT`) this bank holds at its planned `places`.
    pub fn flight(&self) -> u64 {
        self.flight_at(self.places)
    }

    /// The site slots (`RegionWords::sites`) this bank's CPU site tables and
    /// `Parks::publications` hold: `flights.steps(head_steps)`, so every
    /// chained `DraftDecode` step of every concurrent group has its own slot
    /// without clobbering earlier steps in flight.
    pub fn site_slots(&self) -> u64 {
        self.flights.steps(self.head_steps)
    }

    /// Records of group `group`'s tensors (`BANK`): its ceiling, every
    /// expert of its layers, and where the host places (the one store) the
    /// tier's reserve beside them ([`Self::tier_reserve`]); never fewer than its least records
    /// ([`Self::least_records`]): the lender's lent records beside them.
    pub fn records(&self, group: usize) -> u64 {
        let experts = self.member_experts(group).sum::<u64>();
        let records = match self.placer() {
            Placer::Tier => experts,
            Placer::Bank => experts + self.tier_reserve(),
        };
        records.max(self.least_records(group))
    }

    /// Dedicated per-flight VRAM scratch slots appended to each group's bank
    /// `SlabTensor` for on-device 4-CTA coalesced SM staging (`routed_stage_host`)
    /// on discrete (`Placer::Tier`) banks (`DECODE_STAGE_SLOTS = 16`).
    /// Only physically backed when unified 3-tier decode is active (`!whole` on
    /// CUDA); never managed by `Directory`.
    pub fn decode_stage_slots(&self) -> u64 {
        if self.placer() == Placer::Tier {
            u64::from(magnitude_cost_model::DECODE_STAGE_SLOTS)
        } else {
            0
        }
    }

    /// Total logical slab extent of group `group`'s bank `SlabTensor`:
    /// `self.records(group)` (owned by `Directory`) plus `self.decode_stage_slots()`
    /// (tail VRAM scratch at `stage_base = self.records(group)`).
    pub fn slab_records(&self, group: usize) -> u64 {
        self.records(group) + self.decode_stage_slots()
    }

    /// Each member layer of group `group`, in its order.
    pub(super) fn member_layers(&self, group: usize) -> impl Iterator<Item = &BankLayer> + '_ {
        self.groups()[group].layers.iter().map(|&layer| &self.layers()[layer])
    }

    /// The experts of each of group `group`'s layers, in its order.
    fn member_experts(&self, group: usize) -> impl Iterator<Item = u64> + '_ {
        self.member_layers(group).map(|layer| layer.experts)
    }

    /// The most experts of group `group`'s member layers.
    fn widest_member(&self, group: usize) -> u64 {
        self.groups()[group].layers.max_of(|&layer| self.layers()[layer].experts)
    }

    /// The records a layer of `experts` experts holds at `slots` slots:
    /// every expert once the slots reach them.
    pub fn layer_slots(slots: u64, experts: u64) -> u64 {
        slots.min(experts)
    }

    /// Group `group`'s committed records: its slots in each of its layers.
    pub fn committed_records(&self, group: usize) -> u64 {
        self.records_at(group, self.slots)
    }

    /// Whether streamed layer `layer` sets its run's step word: the first
    /// target layer (layer 0), and the first draft layer (which runs in its
    /// own draft graph).
    pub fn sets_step(&self, layer: usize) -> bool {
        layer == 0
            || (self.layers()[layer].is_draft()
                && !self.layers()[..layer].iter().any(BankLayer::is_draft))
    }

    /// The most experts of any streamed layer: what every buffer the layers
    /// share in turn is sized to, each layer binding its leading view.
    pub fn widest(&self) -> NonZeroU64 {
        self.layers().widest()
    }

    /// The record form every streamed layer shares (`with_expert_bank`).
    pub fn form(&self) -> RecordForm {
        self.structure().form()
    }

    /// Choice capacity of the bank's per-step metadata tables (`FileTable`,
    /// `ScratchTable`, `SiteTable`, `HintRing`, `LandingList`) and host-tier
    /// landing floor: always sized for `VERIFY_ROWS` so every sealed decode
    /// graph class `M <= VERIFY_ROWS` (including short prompts and
    /// verification batches) fits the kernel domain and cannot deadlock even
    /// when `decode_rows` is narrower than `VERIFY_ROWS`.
    pub fn table_choices(&self) -> u64 {
        records::decode_choices(self.selected(), magnitude_kernels::constants::VERIFY_ROWS)
    }

    /// Each streamed layer's routed kind, in decoder order: one numbering
    /// of the load's, by what makes two layers' routed calls alike in every
    /// form (their record group, their experts, whether a CPU site serves
    /// them, and for fused `routed_output` layers their `SharedDown` weight),
    /// by which its routed passes are priced (`magnitude_cost_model::RoutedKind`).
    pub fn routed_kinds(&self) -> Vec<magnitude_cost_model::RoutedKind> {
        magnitude_cost_model::RoutedKind::number(self.layers().iter().enumerate().map(|(index, layer)| {
            (layer.group, layer.experts, self.sited(index), layer.shared_down.as_ref())
        }))
    }

    /// Records of each staging pool (`T`): the demand reserve and the ahead
    /// window. A staging record lives only from its read to its landing
    /// (on unified memory, to the device's DONE of its entry,
    /// `experts::landing`), and a request past them waits for one
    /// (`Decode::resume`), so a pool holds what a request must hold at once,
    /// and no more of a cap's bytes.
    pub fn staging_records(&self) -> u64 {
        self.demand_records() + self.ahead_records()
    }

    /// Records of a staging pool at `paths`: pools belonging exclusively to
    /// draft layers (`!has_hints`) never receive cross-layer hints and hold
    /// `demand_records()` only.
    pub fn pool_staging_records_with_paths(&self, has_hints: bool, paths: BankPaths) -> u64 {
        if has_hints {
            self.demand_records() + paths.records().ahead
        } else {
            self.demand_records()
        }
    }

    /// Whether group `group` contains any non-draft (target) layer and
    /// therefore can receive cross-layer hints.
    pub fn group_has_hints(&self, group: usize) -> bool {
        self.groups()[group]
            .layers
            .iter()
            .any(|&layer| !self.layers()[layer].is_draft())
    }

    /// The records of a host tier's group a step holds landing at once: a
    /// request's misses and the hints'. Unlike staging, a landed record stays
    /// until the device has read it, so a request can't wait for one of its
    /// own, and `table_choices()` (`VERIFY_ROWS · K`, the widest launch that
    /// runs the decode form) bounds them.
    pub fn landing_records(&self) -> u64 {
        self.table_choices() + self.ahead_records()
    }

    /// Group `group`'s host tier landing floor at `paths`: bounded by
    /// `request_records(group)` (`min(widest_member(group), VERIFY_ROWS * group_selected)`)
    /// so short `VERIFY_ROWS` prompts never deadlock at floor while narrow or
    /// smaller-`K` groups are never over-allocated; draft-only groups omit
    /// cross-layer hint records.
    pub fn group_landing_records_with_paths(&self, group: usize, paths: BankPaths) -> u64 {
        let demand = self.request_records(group);
        if self.group_has_hints(group) {
            demand + paths.records().ahead
        } else {
            demand
        }
    }

    /// Group `group`'s host tier landing floor at `self.paths`.
    pub fn group_landing_records(&self, group: usize) -> u64 {
        self.group_landing_records_with_paths(group, self.paths)
    }

    /// A linked tier's landing rows ([`Self::landing_records`]), which a
    /// load commits beside its floor: a record of every group each, as
    /// resident; none without a tier.
    pub fn landing(&self) -> Result<Option<crate::memory::ledger::Need>, String> {
        let Some(tier) = self.holding_storage(crate::BankHolding::Host) else {
            return Ok(None);
        };
        let uniform_landing = (0..self.groups().len()).all(|group| {
            self.group_has_hints(group)
                && self.group_landing_records(group) == self.landing_records()
                && self.member_experts(group).sum::<u64>() >= self.landing_records()
        });
        if uniform_landing {
            let slot_bytes = self.groups().iter().try_fold(0u64, |total, group| {
                Self::record_sum(&group.record, RecordTensor::record_bytes)
                    .and_then(|bytes| total.checked_add(bytes).ok_or_else(|| "landing row bytes overflow".to_owned()))
            })?;
            return Ok(Some(crate::memory::ledger::Need::LandingRows {
                tier,
                rows: self.landing_records(),
                slot_bytes,
            }));
        }
        let total_bytes = self.groups().iter().enumerate().try_fold(0u64, |total, (index, group)| {
            let record_bytes = Self::record_sum(&group.record, RecordTensor::record_bytes)?;
            let rows = self
                .group_landing_records(index)
                .min(self.member_experts(index).sum::<u64>());
            record_bytes
                .checked_mul(rows)
                .and_then(|bytes| total.checked_add(bytes))
                .ok_or_else(|| "landing row bytes overflow".to_owned())
        })?;
        Ok(Some(crate::memory::ledger::Need::LandingRows {
            tier,
            rows: 1,
            slot_bytes: total_bytes,
        }))
    }

    /// Whether the bank has a host tier of its own records: a linked one
    /// ([`HostTier::Linked`]); unified memory has none.
    pub fn tiered(&self) -> bool {
        self.host() == HostTier::Linked
    }

    /// Whether prefill fills copy host-tier experts into VRAM scratch via
    /// producer-enqueued hardware DMA (`cuMemcpyHtoDAsync` on CUDA,
    /// `TransferQueue` on Vulkan `side_family`) instead of SM zero-copy
    /// `routed_copy`. Enabled only on tiered (discrete) devices whose backend
    /// supports fills beside compute AND whose device actually exposes a
    /// distinct side queue (never enabled on single-queue Vulkan devices).
    pub fn tier_copy_dma(&self) -> bool {
        self.tiered() && self.backend.fills_beside_compute() && self.has_side_queue
    }

    /// The slab regions of group `group`'s store, a record a slab: each
    /// routed tensor's resident record, in [`RecordTensors::iter`]'s order
    /// (a host tier's store has the same).
    pub fn regions(&self, group: usize) -> Vec<(RoutedTensor, seismic::SlabRegion)> {
        records::record_regions(&self.groups()[group].record)
    }

    /// Who places into the bank ([`HostTier::placer`]): the device, unless
    /// the bank is unified memory's one store.
    pub fn placer(&self) -> Placer {
        self.host().placer()
    }

    /// The pool `holding` lies in: the bank's records and tables the
    /// bank's, a host tier its own; none for a host tier the bank lacks.
    pub fn holding_storage(&self, holding: crate::BankHolding) -> Option<crate::planning::BankTier> {
        match holding {
            crate::BankHolding::Experts | crate::BankHolding::Table { .. } => Some(crate::planning::BankTier::Bank),
            crate::BankHolding::Host => self.tiered().then_some(crate::planning::BankTier::Host),
        }
    }

    /// Group `group`'s lent bank records (`ScratchPlan::lender`): the
    /// bottom records of its store that view its lent allocation
    /// (`BankAllocation::Lent`), which the scratch, and on the one store the
    /// conveyor, lie in; none of another group.
    pub fn lent(&self, group: usize) -> u64 {
        Some(self.scratch().lender).filter(|lender| lender.group == group).map_or(0, |lender| lender.records)
    }

    /// Group `group`'s lent host tier records, which the conveyor lies in
    /// beside a discrete device ([`ConveyorCarrier::HostTier`]): its
    /// lender's, none of another group or where the conveyor lies in the
    /// bank.
    pub fn host_lent(&self, group: usize) -> u64 {
        match self.conveyor().carrier {
            ConveyorCarrier::HostTier(lender) if lender.group == group => lender.records,
            _ => 0,
        }
    }

    /// What the bank's records pools are charged for its records at `slots`:
    /// the whole blocks ([`Self::pooled_charge_for`]) holding each group's
    /// granule for each record it commits beyond its lent ones
    /// ([`Self::lent`], charged at load as the lender's lent allocation), so a
    /// shrink is priced by the blocks it returns.
    pub fn resident_charge(&self, slots: u64) -> Result<u64, String> {
        self.pooled_charge_for(&self.blocks, |group| {
            self.records_at(group, slots)
                .checked_sub(self.lent(group))
                .expect("a group's committed records hold its lent ones")
        })
        .ok_or_else(|| "bank record charge overflows".to_owned())
    }

    /// Each distinct granule's records pool block ([`seismic::block_bytes`]),
    /// indexed by [`Self::granules`].
    pub fn blocks(&self) -> &[seismic::BlockSize] {
        &self.blocks
    }

    /// The pool block of the pool holding banked stored tables
    /// ([`Self::table_pool`]).
    pub fn table_block(&self) -> seismic::BlockSize {
        self.blocks[self.table_pool()]
    }

    /// The largest pool block's held bytes among the bank's records pools: the
    /// minimum resize step for [`crate::experts::derived::BankShape`].
    pub fn tier_block_bytes(&self, tier: BankTier) -> u64 {
        if tier == BankTier::Host && !self.tiered() {
            return 0;
        }
        self.blocks.iter().map(|block| block.held).max().unwrap_or(0)
    }

    /// What each group's pooled records (`pooled_of(group)`) cost across the
    /// deduplicated per-granule pools whose blocks are `blocks`: each pool
    /// sums the granules of every group mapped to it and charges whole blocks
    /// ([`seismic::BlockSize::charge`]), then sums across pools.
    pub(super) fn pooled_charge_for(
        &self,
        blocks: &[seismic::BlockSize],
        pooled_of: impl Fn(usize) -> u64,
    ) -> Option<u64> {
        let granules = self.granules();
        let mut per_pool = vec![0u64; granules.len()];
        for group in 0..self.groups().len() {
            let pool = self.group_pool(group);
            per_pool[pool] = per_pool[pool].checked_add(pooled_of(group))?;
        }
        blocks
            .iter()
            .zip(granules)
            .zip(per_pool)
            .try_fold(0u64, |total, ((block, &granule), count)| {
                total.checked_add(block.charge(granule, count)?)
            })
    }

    /// Staging records only demand takes. Every earlier request's records
    /// are free once the device makes a request (it is past their
    /// landings), so a request always finds these. A single multi-row
    /// verify or front decode pass (up to `VERIFY_ROWS` rows, each
    /// selecting `K` experts) can miss up to `decode_set(K, widest)`
    /// distinct experts at a layer (at most `widest`, the widest streamed
    /// layer's experts). Sizing the demand reserve to
    /// `decode_set(K, widest)` on every host tier lets every miss of a
    /// widest decode request stage at once without waiting on its own
    /// earlier landings (`Decode::waiting` wave serialization), keeps
    /// [`Self::hint_records`] unstarved beyond the reserve for cross-layer
    /// hints (`l + 1`, `l + 2`), and remains equal to `K` when
    /// `widest <= K` (including `widest == 1`).
    pub fn demand_records(&self) -> u64 {
        // Under boundary-parked unified memory (`Parking::AtBoundaries`),
        // `routed_land` does not execute until every absent expert of the step
        // is staged and `file[slot]` is answered, so staging records cannot
        // recycle mid-step and must hold a full `VERIFY_ROWS` decode/prompt
        // step (`min(VERIFY_ROWS · K, widest)`). Under in-kernel parking and
        // linked host tiers, staging records recycle as each expert converts,
        // so `decode_rows()` suffices without deadlocking.
        let rows = match self.host() {
            HostTier::Unified(Parking::AtBoundaries) => magnitude_kernels::constants::VERIFY_ROWS,
            HostTier::Unified(Parking::InKernels) | HostTier::Linked => self.decode_rows(),
        };
        decode_set(self.selected(), self.widest().get(), rows)
    }

    /// The stored bytes of the least move its model store's path makes: an
    /// expert of the group whose experts store fewest, which the path moves
    /// at the most records a second, so the records it sizes
    /// ([`magnitude_cost_model::PathRecords::of`]) cover every group's.
    pub fn least_move_bytes(&self) -> Result<u64, String> {
        let moves = self
            .groups()
            .iter()
            .map(|group| group.record.iter().map(|(_, tensor)| tensor.stored_bytes()).sum::<Result<u64, String>>())
            .collect::<Result<Vec<_>, String>>()?;
        Ok(moves.into_iter().min().expect("a bank has a group"))
    }

    /// Staging records reads ahead may hold beyond the demand reserve: the
    /// moves the path has in flight at its rate, each from its issue to its
    /// landing ([`magnitude_cost_model::PathRecords::ahead`]).
    pub fn ahead_records(&self) -> u64 {
        self.paths.records().ahead
    }

    /// The tier records each group keeps holding no expert mapped (ready,
    /// graced, or taken by a read ahead's landing in flight:
    /// `HostRecords::reserved`): the path's moves in flight and those landed
    /// whose victims the device still reads, until it passes their layer, at
    /// most a decode step at the plan's slots
    /// ([`magnitude_cost_model::PathRecords::reserve`]), so no read ahead's
    /// landing is ever short of a record. A read ahead lands off any park, so it can
    /// take only a record no step reads already; a demand landing, at its
    /// park, reclaims a kept one the parked step does not read
    /// (`Directory::claim_demand`), so it needs none kept for it and takes
    /// none of these while it can reclaim.
    /// Where the host places (unified memory) the tier is the bank's own
    /// records, which each group holds beside its slots: the one count of
    /// what the bank holds empty, which the plan charges and the bank's
    /// telemetry leaves out of the records its experts are held in.
    pub fn tier_reserve(&self) -> u64 {
        self.paths.records().reserve
    }

    /// Records a scratch round converts: half the decode set (the most
    /// distinct experts a request reads at a layer, `decode_set`), so a
    /// round's reads land while the device converts the round before. A
    /// round so never holds more than the widest layer's experts.
    pub fn round_records(&self) -> u64 {
        self.structure().round_records()
    }

    /// `W`: scratch slots held at once for a prompt group's layer
    /// (`round_records <= W <= widest`).
    pub fn window_slots(&self) -> u64 {
        self.structured.window_slots()
    }

    /// `P`: windows per layer (`widest.div_ceil(window_slots)`).
    pub fn windows(&self) -> u64 {
        self.widest().get().div_ceil(self.window_slots())
    }

    /// `RW`: convert rounds per window (`window_slots.div_ceil(round_records)`).
    pub fn window_rounds(&self) -> u64 {
        self.window_slots().div_ceil(self.round_records())
    }

    /// `C_l`: scratch slots used by streamed layer `layer` (`experts.min(window_slots)`).
    pub fn layer_window(&self, layer: usize) -> u64 {
        self.layers()[layer].experts.min(self.window_slots())
    }

    /// Rounds of a scratch-served layer: `P * RW` rounds across all windows.
    /// A hit is read where the bank holds it; a round past a layer's last
    /// records neither parks nor converts.
    pub fn rounds(&self) -> u64 {
        self.windows() * self.window_rounds()
    }

    /// Words of a scratch round's landing list (`routed_round`,
    /// LANDING_ENTRY): its head, then an entry per record it converts,
    /// (conveyor record or scratch slot, scratch slot, DONE).
    pub fn round_list_words(&self) -> u64 {
        magnitude_kernels::constants::LANDING_ENTRY(self.round_records())
    }

    /// Words of the tier list and the seed list (`routed_copy` from the host
    /// tier into scratch slots, and of seeds from their scratch slots into
    /// bank records, then `routed_seed_commit`): the sequence and count,
    /// then a pair per expert, at most a window's slots.
    pub fn expert_list_words(&self) -> u64 {
        constants::LIST_ENTRY(self.window_slots())
    }

    /// The least `slots` a load of this bank holds, and the least a shrink
    /// leaves: what a step needs to make progress ([`Self::needed_slots`]),
    /// the one bound below which a bank cannot serve. A bank of few slots
    /// serves a step mostly by reading it; that is its cost, which the
    /// settle weighs against the slots' bytes, never a refusal.
    pub fn floor(&self) -> u64 {
        self.needed_slots()
    }

    /// The most `slots` at which every group commits only its least records
    /// ([`Self::least_records`]), a slot fewer releasing nothing. A group's
    /// records never fall as the slots grow, so those slots are a prefix of
    /// `0..=widest`, of which none always is one.
    fn needed_slots(&self) -> u64 {
        let least = |slots: u64| (0..self.groups().len()).all(|group| self.records_at(group, slots) == self.least_records(group));
        // The first slots past it: a prefix's end, found by halving.
        let (mut needed, mut past) = (0, self.max_slots() + 1);
        while past - needed > 1 {
            let middle = needed + (past - needed) / 2;
            match least(middle) {
                true => needed = middle,
                false => past = middle,
            }
        }
        needed
    }

    /// Whether the plan holds every expert: such a bank is imported whole at
    /// load, each expert at its home record, as a resident load holds it.
    pub fn whole(&self) -> bool {
        self.whole_at(self.slots)
    }

    /// Maximum 1D bank slot coordinate where every group holds all of its
    /// experts (`whole_at(max_slots()) == true`). Equals [`Self::widest`] for
    /// single-group banks and `Σ_g max_{l ∈ g} E_l` for multi-group banks.
    pub fn max_slots(&self) -> u64 {
        self.residency.max_slots()
    }

    /// O(1) lookup of VRAM slots per layer committed for `group` at 1D bank
    /// coordinate `slots`.
    pub fn group_vram_slots(&self, group: usize, slots: u64) -> u64 {
        self.residency.group_slots(group, slots)
    }

    /// O(1) lookup of VRAM records `layer` commits at 1D bank coordinate
    /// `slots`, ordered by decreasing marginal miss-time reduction per VRAM
    /// byte ([`ResidencyOrder`]).
    pub fn layer_vram_slots(&self, layer: &BankLayer, slots: u64) -> u64 {
        self.residency.layer_slots(layer, slots)
    }

    /// Mean per-layer VRAM slots across target streamed layers (or all layers
    /// when every layer is a drafter) at 1D bank coordinate `slots`, always in
    /// `0..=self.widest().get()`.
    pub fn target_vram_slots(&self, slots: u64) -> u64 {
        let mut sum = 0u64;
        let mut count = 0u64;
        for layer in self.layers() {
            if !layer.is_draft() {
                sum += self.layer_vram_slots(layer, slots);
                count += 1;
            }
        }
        if count == 0 {
            for layer in self.layers() {
                sum += self.layer_vram_slots(layer, slots);
                count += 1;
            }
        }
        if count == 0 { 0 } else { (sum + count / 2) / count }
    }

    /// Precomputed O(1) lookup table mapping raw 1D bank coordinate
    /// `0..=self.max_slots()` to [`Self::target_vram_slots`], for lock-free
    /// reporting in [`crate::experts::watcher::Telemetry`].
    pub fn target_vram_slots_table(&self) -> std::sync::Arc<[u64]> {
        (0..=self.max_slots()).map(|s| self.target_vram_slots(s)).collect::<Vec<_>>().into()
    }

    /// Recompute the 1D marginal-value residency schedule from measured
    /// per-layer miss costs (seconds per byte) before `settle_bank` / `open`.
    pub fn set_residency_costs(&mut self, miss_s_per_byte: &[f64]) {
        self.residency = ResidencyOrder::compute(
            self.groups(),
            self.layers(),
            self.decode_rows(),
            self.head_steps,
            miss_s_per_byte,
        );
    }

    /// Per-layer `(vram, host)` disjoint holdings at `(device_slots, host_slots)`
    /// for cost-model pricing (`BankPrices`, `DecodePrice`, `assessment`).
    pub fn held_slots(&self, device_slots: u64, host_slots: u64) -> magnitude_cost_model::HeldSlots {
        let count = |value: u64| usize::try_from(value).expect("a layer's experts fit usize");
        if self.groups().len() == 1 && self.layers().iter().all(|l| l.experts == self.widest().get()) {
            let layer = &self.layers()[0];
            let vram = self.layer_vram_slots(layer, device_slots);
            let host = self.layer_host_slots(layer, device_slots, host_slots);
            return magnitude_cost_model::HeldSlots::tiered(count(vram), count(host));
        }
        let per_layer = self
            .layers()
            .iter()
            .map(|layer| {
                let vram = self.layer_vram_slots(layer, device_slots);
                let host = self.layer_host_slots(layer, device_slots, host_slots);
                (count(vram), count(host))
            })
            .collect();
        magnitude_cost_model::HeldSlots::per_layer(per_layer)
    }

    /// Per-layer `(vram, host)` disjoint holdings for a live bank at
    /// `device_slots` whose host tier was sized against `host_reference_slots()`
    /// and currently holds `host_slots`: clamps each layer's disjoint host
    /// contribution to `layer.experts - v_l(device_slots)` so `v_l + h_l <= E_l`
    /// always holds even after runtime device resizes.
    pub fn held_slots_runtime(&self, device_slots: u64, host_slots: u64) -> magnitude_cost_model::HeldSlots {
        let count = |value: u64| usize::try_from(value).expect("a layer's experts fit usize");
        let host_ref = self.host_reference_slots();
        if self.groups().len() == 1 && self.layers().iter().all(|l| l.experts == self.widest().get()) {
            let layer = &self.layers()[0];
            let vram = self.layer_vram_slots(layer, device_slots);
            let host = self
                .layer_host_slots(layer, host_ref, host_slots)
                .min(layer.experts.saturating_sub(vram));
            return magnitude_cost_model::HeldSlots::tiered(count(vram), count(host));
        }
        let per_layer = self
            .layers()
            .iter()
            .map(|layer| {
                let vram = self.layer_vram_slots(layer, device_slots);
                let host = self
                    .layer_host_slots(layer, host_ref, host_slots)
                    .min(layer.experts.saturating_sub(vram));
                (count(vram), count(host))
            })
            .collect();
        magnitude_cost_model::HeldSlots::per_layer(per_layer)
    }

    /// Whether a device bank of `slots` holds every expert of every streamed
    /// layer: the one predicate the plan's tier and the running bank's host
    /// target both decide by.
    pub fn whole_at(&self, slots: u64) -> bool {
        slots >= self.max_slots()
            && self
                .layers()
                .iter()
                .all(|layer| self.layer_vram_slots(layer, slots) >= layer.experts)
    }

    /// The stored bytes of the experts neither the bank's slots nor its
    /// disjoint host tier's host slots hold, each as its model file stores it:
    /// what the plan's misses read from the file. None of a whole bank.
    pub fn unheld_bytes(&self) -> Result<u64, String> {
        let overflow = || "unheld expert bytes overflow".to_owned();
        self.layers().iter().try_fold(0u64, |total, layer| {
            let expert = self.groups()[layer.group]
                .record
                .iter()
                .try_fold(0u64, |bytes, (_, tensor)| bytes.checked_add(tensor.stored_bytes()?).ok_or_else(overflow))?;
            let vram = self.layer_vram_slots(layer, self.slots);
            let host = self.layer_host_slots(layer, self.slots, self.host_slots);
            let held = (vram + host).min(layer.experts);
            (layer.experts - held).checked_mul(expert).and_then(|bytes| total.checked_add(bytes)).ok_or_else(overflow)
        })
    }

    /// The stored bytes of the experts the disjoint host tier holds that the
    /// bank's slots do not: what the plan holds in host RAM alone. None
    /// without a tier.
    pub fn host_only_bytes(&self) -> Result<u64, String> {
        let overflow = || "host tier expert bytes overflow".to_owned();
        self.layers().iter().try_fold(0u64, |total, layer| {
            let expert = self.groups()[layer.group]
                .record
                .iter()
                .try_fold(0u64, |bytes, (_, tensor)| bytes.checked_add(tensor.stored_bytes()?).ok_or_else(overflow))?;
            let alone = self.layer_host_slots(layer, self.slots, self.host_slots);
            alone.checked_mul(expert).and_then(|bytes| total.checked_add(bytes)).ok_or_else(overflow)
        })
    }

    /// Whether growth reaches every expert ([`Self::ceiling`]).
    pub fn reaches_whole(&self) -> bool {
        self.whole_at(self.ceiling)
    }

    /// Group `group`'s experts: every member layer's, its map's extent
    /// (`BankPart::Map`).
    pub fn group_experts(&self, group: usize) -> u64 {
        self.groups()[group].layers.iter().map(|&member| self.layers()[member].experts).sum()
    }

    /// The home record of expert 0 of streamed layer `layer` in its group
    /// ([`layer_homes`]); the same word of the group's map (`BankPart::Map`)
    /// is the layer's expert 0's.
    pub fn home(&self, layer: usize) -> u64 {
        layer_homes(self.layers().iter().map(|layer| (layer.group, layer.experts)))[layer]
    }

    /// What one more slot commits at most: a record of every layer. The
    /// load's weights grow by it up to the granules backing each record
    /// tensor, less a group's records while it holds its decode request's
    /// ([`Self::records_at`]).
    pub fn slot_bytes(&self) -> Result<u64, String> {
        self.slot_sum(RecordTensor::record_bytes)
    }

    /// `record` of each routed tensor, summed over every layer.
    fn slot_sum(&self, record: fn(&RecordTensor) -> Result<u64, String>) -> Result<u64, String> {
        self.layers().iter().try_fold(0u64, |total, layer| {
            Self::record_sum(&self.groups()[layer.group].record, record).and_then(|bytes| {
                total
                    .checked_add(bytes)
                    .ok_or_else(|| "bank slot bytes overflow".to_owned())
            })
        })
    }

    /// `bytes` of each routed tensor of `record`, summed.
    fn record_sum(record: &RecordTensors<RecordTensor>, bytes: fn(&RecordTensor) -> Result<u64, String>) -> Result<u64, String> {
        record.iter().try_fold(0u64, |sum, (_, tensor)| {
            sum.checked_add(bytes(tensor)?).ok_or_else(|| "a record's bytes overflow".to_owned())
        })
    }

    /// What `allocations` demand at the load.
    fn demand<'a>(&self, mut allocations: impl Iterator<Item = &'a BankAllocation>) -> Result<ClaimDemand, String> {
        allocations.try_fold(ClaimDemand::default(), |total, allocation| {
            total
                .checked_plus(allocation.storage_demand(self.host())?)
                .ok_or_else(|| "bank storage demand overflows".to_owned())
        })
    }

    /// Every byte the bank is charged at its load: each allocation of its
    /// layout, the floor's placed words, and its pools' granules of the
    /// records it commits there (its slots', [`Self::resident_charge`], and
    /// a host tier's at no host slots, [`Self::host_charge`]); a group's
    /// more places are charged as it holds them ([`Self::placed_bytes`]),
    /// more records as they commit.
    pub fn bytes(&self) -> Result<u64, String> {
        self.storage_demand()?.checked_bytes().ok_or_else(|| "bank storage demand overflows".to_owned())
    }

    /// The physical demand of the layout, its floor's placed words and its
    /// pools' granules at its load ([`Self::bytes`]).
    pub fn storage_demand(&self) -> Result<ClaimDemand, String> {
        let (layout, floor) = (self.layout()?, self.placed_allocations(NonZeroU64::MIN)?);
        self.demand(layout.iter().chain(&floor))?
            .checked_plus(self.pooled_demand()?)
            .ok_or_else(|| "bank storage demand overflows".to_owned())
    }

    /// The pools' granules of the records the bank commits at its load
    /// ([`Self::pooled_charges`]).
    fn pooled_demand(&self) -> Result<ClaimDemand, String> {
        self.pooled_charges()?.into_iter().try_fold(ClaimDemand::default(), |total, (_, demand)| {
            total.checked_plus(demand).ok_or_else(|| "bank storage demand overflows".to_owned())
        })
    }

    /// The absent granules the bank's records pools allocate when they open
    /// (`LoadPools::open_at`), held for the load's life outside any slot count:
    /// one per distinct granule in the bank's store and, for a linked host
    /// tier, one per distinct granule in host RAM.
    pub fn absent_demand(&self) -> ClaimDemand {
        let absent: u64 = self.blocks.iter().map(|block| block.absent).sum();
        let mut demand = ClaimDemand::pooled(BankTier::Bank, absent);
        if self.tiered() {
            demand = demand
                .checked_plus(ClaimDemand::pooled(BankTier::Host, absent))
                .expect("bank absent granule demand fits u64");
        }
        demand
    }

    /// The pools' granules of the records the bank commits at its load, by
    /// what commits them: the bank's for its slots
    /// ([`Self::resident_charge`]), for its stored tables, every record
    /// held ([`Self::tables_charge`]), its records pools' absent granules,
    /// and a host tier's at no host slots ([`Self::host_charge`]) and its
    /// pools' absent granules, where the bank has a tier.
    pub(super) fn pooled_charges(&self) -> Result<Vec<(&'static str, ClaimDemand)>, String> {
        let absent: u64 = self.blocks.iter().map(|block| block.absent).sum();
        let mut pooled = vec![
            ("group: records", ClaimDemand::pooled(BankTier::Bank, self.resident_charge(self.slots)?)),
            ("pool: absent", ClaimDemand::pooled(BankTier::Bank, absent)),
        ];
        if !self.tables.is_empty() {
            pooled.push(("stored tables", ClaimDemand::pooled(BankTier::Bank, self.tables_charge()?)));
        }
        if self.tiered() {
            pooled.push(("host tier", ClaimDemand::pooled(BankTier::Host, self.host_charge(0)?)));
            pooled.push(("host pool: absent", ClaimDemand::pooled(BankTier::Host, absent)));
        }
        Ok(pooled)
    }

    /// What the bank's pool is charged for its stored tables at the load,
    /// every record of each held: each table's whole blocks
    /// ([`BankTable::charge`]), as the load backs them
    /// (`memory::ledger::DeviceLedgers::split_records`). Each table's
    /// records are its own ([`BankTable::held_bytes`]); the rest are the
    /// pool's spare granules.
    pub fn tables_charge(&self) -> Result<u64, String> {
        let (block, granule) = (self.table_block(), self.table_granule());
        self.tables.iter().try_fold(0u64, |sum, table| {
            BankTable::charge(block, granule, table.records.get())?
                .checked_add(sum)
                .ok_or_else(|| "banked table charge overflows".to_owned())
        })
    }

    /// What layer `layer`'s banked expert stack of each routed tensor
    /// charges for its records: its slots' (`ModelLoadPlan::with_expert_bank`).
    pub fn banked_bytes(&self, layer: usize) -> Result<RecordTensors<u64>, String> {
        let slots = self.layer_vram_slots(&self.layers()[layer], self.slots);
        self.groups()[self.layers()[layer].group].record.try_map(|_, tensor| {
            tensor
                .record_bytes()?
                .checked_mul(slots)
                .ok_or_else(|| "banked expert bytes overflow".to_owned())
        })
    }

    /// What the bank is charged beyond what its banked weights charge
    /// ([`Self::storage_beyond_records`]).
    pub fn charged_beyond_records(&self) -> Result<u64, String> {
        self.storage_beyond_records()?.checked_bytes()
            .ok_or_else(|| "bank storage demand overflows".to_owned())
    }

    /// What the bank demands beyond what its banked weights charge (each
    /// layer's slots' records, [`Self::banked_bytes`], and its stacks'
    /// per-expert scales): everything else it holds at its load. A banked
    /// record lies in a granule of the pool or in the lender's lent
    /// records, so the records' bytes are taken from the pool's granules
    /// first and the rest from the lent allocation's storage
    /// (`BankAllocation::Lent`): device storage beside a link, shared on
    /// the one store.
    pub fn storage_beyond_records(&self) -> Result<ClaimDemand, String> {
        let mut demand = self.storage_demand()?;
        let mut records = 0u64;
        for (index, layer) in self.layers().iter().enumerate() {
            let banked = self.banked_bytes(index)?;
            for ((_, plan), (_, bytes)) in layer.resident.iter().zip(banked.iter()) {
                records = records.checked_add(*bytes).ok_or("banked record bytes overflow")?;
                demand.device = demand.device.checked_sub(plan.scale.as_ref().map_or(0, |scale| scale.resident_bytes))
                    .ok_or("a bank holds less than its banked scales charge")?;
            }
        }
        // The bank's pool is charged as device storage (`ClaimDemand::pooled`).
        let pooled = records.min(self.resident_charge(self.slots)?);
        demand.device -= pooled;
        // Each stored table's held records are its weight's, every one in
        // the pool (`BankTable::plan`).
        let table_granule = self.table_granule();
        for table in &self.tables {
            demand.device = demand.device.checked_sub(BankTable::held_bytes(table_granule, table.records.get())?)
                .ok_or("a bank holds less than its stored tables' records charge")?;
        }
        let lent = match self.host().contract(BankTier::Bank) {
            seismic::Contract::DeviceOnly => &mut demand.device,
            _ => &mut demand.shared,
        };
        *lent = lent.checked_sub(records - pooled).ok_or("a bank holds less than its banked records charge")?;
        Ok(demand)
    }

    /// The plan with its streamed layers replaced by `layers`, its
    /// structure's lending and residency schedule derived again
    /// ([`conveyor::Structured::new`], [`ResidencyOrder::default_for`]).
    #[cfg(test)]
    pub(crate) fn with_layers(self, layers: BankLayers) -> Self {
        let window_slots = self.window_slots().min(layers.widest().get());
        let decode_rows = self.decode_rows();
        let head_steps = self.head_steps;
        let tier_copy_dma = self.tier_copy_dma();
        let (selected, host, groups, _) = self.structured.into_parts();
        let structured = conveyor::Structured::new(selected, host, groups, layers, Some(window_slots), decode_rows)
            .expect("the layers keep the bank's lending");
        let blocks = size_pool_blocks(structured.granules(), self.allocations)
            .expect("the layers size valid bank blocks");
        let residency = ResidencyOrder::default_for(
            structured.groups(),
            structured.layers(),
            decode_rows,
            head_steps,
            host.placer(),
            tier_copy_dma,
        );
        Self { structured, blocks, residency, ..self }
    }

    /// The layer whose weights are of `scope` and its index, when the bank
    /// holds its experts.
    pub fn layer(&self, scope: WeightScope) -> Option<(usize, &BankLayer)> {
        self.layers()
            .iter()
            .enumerate()
            .find(|(_, layer)| layer.scope == scope)
    }

    /// The layer of the feed-forward of `scope` (a block's feed-forward
    /// sublayer), its own or its parallel branch's, and its index, when the
    /// bank holds its experts.
    pub fn feed_forward(&self, scope: WeightScope) -> Option<(usize, &BankLayer)> {
        self.layers().iter().enumerate().find(|(_, layer)| match (layer.scope, scope) {
            (WeightScope::TargetBranch { sublayer, .. }, WeightScope::TargetSublayer(of)) => sublayer == of,
            (own, of) => own == of,
        })
    }

    /// The layer of target block `block`'s feed-forward ([`Self::feed_forward`]
    /// of its sublayer 1: the sublayer's own or its parallel routed
    /// branch's), and its index, when the bank holds its experts. The one
    /// lookup of a block's streamed layer.
    pub fn block_layer(&self, block: usize) -> Option<usize> {
        let block = u32::try_from(block).expect("invariant violated: target block index exceeds u32");
        self.feed_forward(WeightScope::TargetSublayer(magnitude_family_contracts::SublayerIndex { block, sublayer: 1 }))
            .map(|(layer, _)| layer)
    }

    /// The resident plan of a banked expert stack: what tuning imports.
    pub fn resident_plan(&self, role: WeightRole) -> Option<&WeightPlan> {
        let kind = RoutedTensor::of(role.kind)?;
        self.layer(role.scope).and_then(|(_, layer)| layer.resident.get(kind))
    }

    /// Group `group`'s least records ([`Self::least_records`]) with `reserve`
    /// tier-reserve records instead of `self.tier_reserve()`.
    fn least_records_with_reserve(&self, group: usize, reserve: u64) -> u64 {
        let decode = match self.placer() {
            Placer::Tier => 0,
            Placer::Bank => self.request_records(group) + reserve,
        };
        let prompt = if group != self.lender_group() {
            0
        } else {
            let lent = self.scratch().lender.records;
            match self.placer() {
                Placer::Bank => lent + reserve,
                Placer::Tier => lent,
            }
        };
        decode.max(prompt).max(1)
    }

    /// Group `group`'s records (`BANK`, [`Self::records`]) at `paths`.
    pub(super) fn records_with_paths(&self, group: usize, paths: BankPaths) -> u64 {
        let reserve = paths.records().reserve;
        let experts = self.member_experts(group).sum::<u64>();
        let records = match self.placer() {
            Placer::Tier => experts,
            Placer::Bank => experts + reserve,
        };
        records.max(self.least_records_with_reserve(group, reserve))
    }

    /// Group `group`'s committed records ([`Self::records_at`]) at `slots`
    /// and `paths`.
    pub fn records_at_with_paths(&self, group: usize, slots: u64, paths: BankPaths) -> u64 {
        let reserve = paths.records().reserve;
        let held = self.member_layers(group).map(|layer| self.layer_vram_slots(layer, slots)).sum::<u64>();
        let landing = match self.placer() {
            Placer::Tier => 0,
            Placer::Bank => reserve,
        };
        (held + landing).max(self.least_records_with_reserve(group, reserve))
    }

    /// The least `slots` ([`Self::floor`]) at `paths`.
    pub fn floor_with_paths(&self, paths: BankPaths) -> u64 {
        let reserve = paths.records().reserve;
        let least = |slots: u64| {
            (0..self.groups().len())
                .all(|group| self.records_at_with_paths(group, slots, paths) == self.least_records_with_reserve(group, reserve))
        };
        let (mut needed, mut past) = (0, self.max_slots() + 1);
        while past - needed > 1 {
            let middle = needed + (past - needed) / 2;
            match least(middle) {
                true => needed = middle,
                false => past = middle,
            }
        }
        needed
    }

    /// What the bank's pools are charged for its records at `slots` and `paths`
    /// (matching [`Self::resident_charge`] with `self.paths = paths`).
    pub fn resident_charge_with_paths(&self, slots: u64, paths: BankPaths) -> Result<u64, String> {
        self.pooled_charge_for(&self.blocks, |group| {
            self.records_at_with_paths(group, slots, paths)
                .checked_sub(self.lent(group))
                .expect("a group's committed records hold its lent ones")
        })
        .ok_or_else(|| "bank record charge overflows".to_owned())
    }

    /// Group `group`'s committed host tier records ([`Self::host_records`])
    /// at `host_slots` and `paths`.
    pub fn host_records_with_paths(&self, group: usize, host_slots: u64, paths: BankPaths) -> u64 {
        self.host_records_at_with_paths(group, self.host_reference_slots(), host_slots, paths)
    }

    /// What the host tier's pools are charged ([`Self::host_charge`]) at
    /// `host_slots` and `paths`.
    pub fn host_charge_with_paths(&self, host_slots: u64, paths: BankPaths) -> Result<u64, String> {
        self.host_charge_at_with_paths(self.host_reference_slots(), host_slots, paths)
    }

    /// The pools' granules demanded at `slots` and `paths` (matching
    /// [`Self::pooled_demand`] with `self.slots = slots` and `self.paths = paths`).
    pub fn pooled_demand_at(&self, slots: u64, paths: BankPaths) -> Result<ClaimDemand, String> {
        let mut demand = ClaimDemand::pooled(BankTier::Bank, self.resident_charge_with_paths(slots, paths)?);
        if !self.tables.is_empty() {
            demand = demand
                .checked_plus(ClaimDemand::pooled(BankTier::Bank, self.tables_charge()?))
                .ok_or_else(|| "bank storage demand overflows".to_owned())?;
        }
        if self.tiered() {
            demand = demand
                .checked_plus(ClaimDemand::pooled(BankTier::Host, self.host_charge_with_paths(0, paths)?))
                .ok_or_else(|| "bank storage demand overflows".to_owned())?;
        }
        Ok(demand)
    }

    /// Exact physical storage demanded by this bank at `(slots, places, paths)`,
    /// matching [`Self::storage_demand`] of a bank built with those parameters.
    pub fn storage_demand_at(&self, slots: u64, places: u64, paths: BankPaths) -> Result<ClaimDemand, String> {
        let sub = |a: ClaimDemand, b: ClaimDemand| -> Result<ClaimDemand, String> {
            a.checked_minus(b).ok_or_else(|| "bank storage demand underflows".to_owned())
        };
        let demand = sub(self.storage_demand()?, self.places_storage_demand(self.places)?)?;
        let demand = sub(demand, self.paths_layout_demand(self.paths)?)?;
        let demand = sub(demand, self.pooled_demand_at(self.slots, self.paths)?)?;
        let places_demand = self.places_storage_demand(places)?;
        let paths_demand = self.paths_layout_demand(paths)?;
        let pooled_demand = self.pooled_demand_at(slots, paths)?;
        demand
            .checked_plus(places_demand)
            .and_then(|d| d.checked_plus(paths_demand))
            .and_then(|d| d.checked_plus(pooled_demand))
            .ok_or_else(|| "bank storage demand overflows".to_owned())
    }
}

/// Size each distinct granule's pool block under `allocations`.
pub(super) fn size_pool_blocks(
    granules: &[u64],
    allocations: seismic::NativeAllocations,
) -> Result<Vec<seismic::BlockSize>, String> {
    granules
        .iter()
        .map(|&granule| {
            seismic::block_bytes(seismic::BlockSizing { granule, allocations })
                .map_err(|error| format!("the bank's records pool blocks: {error:?}"))
        })
        .collect()
}

#[cfg(test)]
mod tests;
