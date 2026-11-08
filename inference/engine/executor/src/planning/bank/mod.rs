//! A streamed load's expert bank:
//! the target's routed expert stacks are held as bank records that a decode
//! step fills from the model file when it routes to an expert the bank lacks,
//! instead of being imported whole.
//!
//! Layers whose records have one form (the stored and resident element and
//! matrix of each routed tensor) share a bank group: one reserved tensor per
//! routed tensor of `E × layers` records (its ceiling, `BANK`), of which the
//! leading `slots × layers` are committed. Every member layer's entries
//! address it (group-wide record indices), and a miss takes a free committed
//! record, or else its least recently used one. Record `l·E + e` is expert
//! `e` of the group's `l`-th layer, its home: a group of every expert is
//! exactly the resident stacks laid end to end. A resize recommits the
//! tensors in place, so nothing bound to them is formed again.

mod build;
mod conveyor;
mod counts;
mod layout;
mod records;
mod sizing;
mod table;

use super::resources::FlightDepth;
use super::weights::WeightPlan;
use crate::memory::ClaimDemand;
use magnitude_kernels::{SlotTable, constants, list_extent};
use magnitude_family_contracts::{WeightKind, WeightRole, WeightScope};
use seismic::Element;
use std::num::NonZeroU64;
use table::stored_rows;
pub use conveyor::{ConveyorCarrier, Lender, StagingPool};
pub use counts::BankCounts;
pub use layout::{BankAllocation, BankLayout, BankPart, BankTier, HostTier, PlacedAllocations, Placer};
pub use records::{RecordForm, RecordTensor, RecordTensors, RoutedTensor, routed_experts};
pub(crate) use records::layer_homes;
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
    /// Its expert stacks as a resident load plans them, one of each routed
    /// tensor of its record: what tuning imports to measure the entries
    /// both loads share.
    pub resident: RecordTensors<WeightPlan>,
    /// What the route of the layer before it reads of it in place of its
    /// own, in the route's hint ports' order (`operators::routed::
    /// hint_kinds`): every streamed layer's are alike, kinds and elements,
    /// as the entries read the next layer's weights as their own.
    pub hinted: Vec<HintWeight>,
}

/// A weight a streamed layer's route reads of the layer it predicts: its
/// kind and resident element, or `None` where the layer has none and the
/// route reads its own port of that kind (an absent norm, zeros for a bias).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct HintWeight {
    pub kind: WeightKind,
    pub element: Option<Element>,
}

/// One expert stack as its model file stores it: the stored tensor of one
/// routed tensor of one layer, from `offset` in `path`, each of its
/// `experts` experts' `pitch` bytes in turn; each expert's record lies
/// `first` bytes into its own, and within it. A stack stored whole reads
/// its experts whole (`first` 0); a range of each stored expert's rows
/// (`ImportTransform::Rows`, Gemma 4's gate‖up halves) from the range's
/// first row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StoredStack {
    pub path: std::path::PathBuf,
    pub offset: u64,
    pub experts: u64,
    pub pitch: u64,
    pub first: u64,
}

impl StoredStack {
    /// The stack's bytes in its file: never past the stored tensor's, whose
    /// bytes `pitch` divides among its experts.
    pub fn bytes(&self) -> u64 {
        self.experts * self.pitch
    }
}

/// A streamed load's expert bank.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExpertBankPlan {
    /// Its structure and the lending derived from it (`conveyor::Structured`):
    /// the experts a row selects, its record reservation, its host tier, its
    /// groups and layers ([`Self::selected`], [`Self::reservation`],
    /// [`Self::host`], [`Self::groups`], [`Self::layers`]), with its conveyor
    /// and scratch ([`Self::conveyor`], [`Self::scratch`]).
    structured: conveyor::Structured,
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
    /// a whole-context prompt (`ExecutionPlanDraft::prompt_places`): the
    /// steps in flight its routing records hold (`FlightDepth::steps`, the
    /// routed kernels' `FLIGHT`). Each place's own step word, scratch flags
    /// and request regions and group reads (`BankPart::placed`) are held
    /// for the places a group holds (`Self::placed`), the floor's one at the
    /// load, so the bank's static charge is the same whichever chain the
    /// plan admits.
    pub places: u64,
    /// The launches one merged-expert prompt pass covers
    /// ([`crate::PassLaunches`]): what the pass buffers and the pass graphs
    /// are sized by, at most `places`. Built at every place; the plan's
    /// settle sizes it by what its fewer passes save
    /// (`ExecutionPlanDraft::with_largest_bank`).
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
}

impl BankLayer {
    /// The target sublayer it routes in: a routed sublayer, or a parallel
    /// block's routed branch's. Only a target routed layer streams
    /// (`routed_layers`).
    pub fn sublayer(&self) -> magnitude_family_contracts::SublayerIndex {
        match self.scope {
            WeightScope::TargetSublayer(sublayer) | WeightScope::TargetBranch { sublayer, .. } => sublayer,
            other => panic!("{other:?} streams its experts, which only a target routed layer does"),
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

/// The leads a streamed layer's route predicts at: the layers one and two
/// ahead, each over the post-attention residual through that layer's router
/// (the gate-ahead), each candidate priced at its own lead
/// (`experts::watcher::telemetry::HintPrice`). A second lead pays where
/// wrong reads are cheap: GB10's read engine on the reference routing's
/// exact hints, 28.9 ms a step against 32.2 at one lead (a third adds 0.6
/// ms); on an M4 Pro it is even (45.0 against 45.1), and its pricing admits
/// what pays.
pub const HINT_LEADS: u64 = 2;

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
    /// `EN`: the experts of the layer its route hints (its own, for the
    /// last streamed layer, whose hint ports bind its own tensors).
    pub next_experts: u64,
    /// `K`.
    pub selected: u64,
    /// `T`.
    pub staging: u64,
    /// `L` of a convert round.
    pub round_records: u64,
    /// The widest streamed layer's experts: the scratch flags' and the
    /// seed list's extents, which every layer shares in turn.
    pub widest: u64,
    /// `V`: the conveyor's records ([`ExpertBankPlan::conveyor_records`]).
    pub conveyor: u64,
    /// `L` of the group's move list ([`ExpertBankPlan::move_records`]).
    pub moves: u64,
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

    /// `RING`: a hint ring's words, an entry (its expert and its rank,
    /// HINT_ENTRY_WORDS) for every choice.
    pub fn ring_words(&self) -> u64 {
        constants::RING_EXPERTS + constants::HINT_ENTRY_WORDS * self.choices
    }

    /// A routing record's words, RING_EXPERTS + 2 P: each choice's expert
    /// and each choice's source (`routed_route_streamed`'s `routing`).
    pub fn routing_words(&self) -> u64 {
        constants::RING_EXPERTS + 2 * self.choices
    }

    /// The word of a routing record holding choice `choice`'s expert.
    pub fn routing_expert(&self, choice: u64) -> u64 {
        constants::RING_EXPERTS + choice
    }

    /// The word of a routing record holding choice `choice`'s source as its
    /// route resolved it, after every choice's expert.
    pub fn routing_source(&self, choice: u64) -> u64 {
        constants::RING_EXPERTS + self.choices + choice
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
            let parts = record.gate.iter().chain([&record.up, &record.down]).map(|tensor| tensor.stored_bytes()).collect::<Result<Vec<_>, _>>()?;
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
    ) -> Result<Vec<RecordTensors<StoredStack>>, String> {
        self.layers()
            .iter()
            .map(|layer| {
                let record = &self.groups()[layer.group].record;
                layer.resident.as_ref().try_map(|kind, weight| {
                    let stored = crate::residency::Stored::from_gguf(artifact, &weight.descriptor)
                        .map_err(|error| error.to_string())?;
                    let (source, offset, bytes) = stored.file_range();
                    let name = &weight.descriptor.name;
                    if bytes % layer.experts != 0 {
                        return Err(format!("{name:?} stores {bytes} bytes, not {} experts", layer.experts));
                    }
                    let tensor = record.get(kind).ok_or("a layer's stacks disagree with its group's record")?;
                    let start = stored_rows(&weight.descriptor)
                        .ok_or_else(|| format!("{name:?}: a banked stack is stored whole or as a row range"))?;
                    let record_bytes = tensor.stored_bytes()?;
                    let first = start
                        .checked_mul(record_bytes / tensor.matrix[0])
                        .ok_or_else(|| format!("{name:?}: its first row's offset overflows"))?;
                    let pitch = bytes / layer.experts;
                    if first.checked_add(record_bytes).is_none_or(|end| end > pitch) {
                        return Err(format!("{name:?}: its rows lie past each stored expert"));
                    }
                    Ok(StoredStack {
                        path: source.path().to_owned(),
                        offset,
                        experts: layer.experts,
                        pitch,
                        first,
                    })
                })
            })
            .collect()
    }

    /// Whether streamed layer `layer` has a CPU site ([`BankSites`]): only
    /// beside a host tier across a link ([`Self::tier_reservation`]), whose
    /// records the site reads; unified memory has none.
    pub fn sited(&self, layer: usize) -> bool {
        self.tier_reservation().is_some()
            && match &self.sites {
                BankSites::Unadmitted => true,
                BankSites::Admitted(sites) => matches!(sites[layer], LayerSite::Served | LayerSite::Pending),
            }
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
        // Every lead binds a layer routing over the next layer's experts
        // (`routed_route_streamed`'s EN); the last layer names its own.
        let next = &self.layers()[self.lead_layer(layer, 0)];
        let conveyor = &self.conveyor().conveyor;
        Ok(BankBinding {
            record: self.groups()[group].record,
            records: self.records(group),
            experts: self.layers()[layer].experts,
            next_experts: next.experts,
            selected: self.selected(),
            staging: self.staging_records(),
            round_records: self.round_records(),
            widest: self.widest().get(),
            conveyor: conveyor.records,
            moves: self.move_records(group),
            tier: self.host_extent(group),
            placer: self.placer(),
            words: RegionWords {
                choices: self.choices(),
                rounds: self.rounds(),
                flight: self.flights.steps(self.places),
                sites: self.flights.steps(1),
            },
        })
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

    /// The experts of each of group `group`'s layers, in its order.
    fn member_experts(&self, group: usize) -> impl Iterator<Item = u64> + '_ {
        self.groups()[group].layers.iter().map(|&layer| self.layers()[layer].experts)
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

    /// Whether streamed layer `layer` sets its run's step word: the first.
    pub fn sets_step(&self, layer: usize) -> bool {
        layer == 0
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

    /// The choices of the decode form's widest launch (`decode_choices`): the
    /// most a decode request names, so the answers' extent (`P`) and the host
    /// tier's landing reserve ([`Self::landing_records`]).
    pub fn choices(&self) -> u64 {
        self.structure().choices()
    }

    /// The layer `layer`'s route predicts at lead `lead`, where that lead
    /// exists: the layer `lead + 1` ahead, while `lead` is below
    /// [`HINT_LEADS`] and that layer is a streamed one routing over the next
    /// layer's experts (one `EN` for every lead of a route). The one rule of
    /// which leads a route has.
    pub fn hints(&self, layer: usize, lead: usize) -> Option<usize> {
        let target = layer + 1 + lead;
        (lead < HINT_LEADS as usize
            && target < self.layers().len()
            && self.layers()[target].experts == self.layers()[layer + 1].experts)
            .then_some(target)
    }

    /// The streamed layer whose hinted weights layer `layer`'s route binds
    /// at lead `lead`: the layer it predicts there ([`Self::hints`]); past
    /// the leads it has, which the route does not read, the next layer it
    /// predicts again, so every lead binds weights over the experts its `EN`
    /// names ([`Self::binding`]); the layer itself where it predicts none
    /// (the last streamed layer). The one rule of what a lead binds.
    pub fn lead_layer(&self, layer: usize, lead: usize) -> usize {
        self.hints(layer, lead).or_else(|| self.hints(layer, 0)).unwrap_or(layer)
    }

    /// The streamed layer each lead of `layer`'s route predicts, by lead (the
    /// layer one ahead, then two), each that exists ([`Self::hints`]).
    pub fn predicted(&self, layer: usize) -> impl Iterator<Item = usize> + '_ {
        (0..HINT_LEADS as usize).map_while(move |lead| self.hints(layer, lead))
    }

    /// The leads a streamed layer's route predicts at (`AHEAD`, its rings'
    /// rows): those that exist ([`Self::predicted`]), up to [`HINT_LEADS`],
    /// none for the last streamed layer.
    pub fn ahead(&self, layer: usize) -> u64 {
        self.predicted(layer).count() as u64
    }

    /// Records of each staging pool (`T`): the demand reserve and the hint
    /// window. A staging record lives only from its read to its landing, and
    /// a request past them waits for one (`Decode::resume`), so a pool holds
    /// what one row reads at once, and no more of a cap's bytes.
    pub fn staging_records(&self) -> u64 {
        self.demand_records() + self.hint_records()
    }

    /// The records of a host tier's group a step holds landing at once: a
    /// request's misses and the hints'. Unlike staging, a landed record stays
    /// until the device has read it, so a request can't wait for one of its
    /// own, and its choices bound them.
    pub fn landing_records(&self) -> u64 {
        self.choices() + self.hint_records()
    }

    /// The reservation of the host tier's own records, where the bank has
    /// a tier ([`HostTier::tier`]: a linked one; unified memory has none).
    pub fn tier_reservation(&self) -> Option<seismic::Reservation> {
        self.host().tier()
    }

    /// Who places into the bank ([`HostTier::placer`]): the device, unless
    /// the bank is unified memory's one store.
    pub fn placer(&self) -> Placer {
        self.host().placer()
    }

    /// The reserved storage `holding` lies in: the bank's records and
    /// tables device storage, a host tier host storage; none for a host
    /// tier the bank lacks.
    pub fn holding_storage(&self, holding: crate::BankHolding) -> Option<crate::planning::BankTier> {
        match holding {
            crate::BankHolding::Experts | crate::BankHolding::Table { .. } => Some(crate::planning::BankTier::Bank),
            crate::BankHolding::Host => self.tier_reservation().map(|_| crate::planning::BankTier::Host),
        }
    }

    /// The host tier records of group `group` the conveyor lies in
    /// ([`ConveyorCarrier::HostTier`]): its lender's lent records, none of
    /// another group or where the conveyor lies in the bank.
    fn tier_lent(&self, group: usize) -> u64 {
        match self.conveyor().carrier {
            ConveyorCarrier::HostTier(lender) if lender.group == group => lender.records,
            _ => 0,
        }
    }

    /// The bank record reservations' charge at `slots`, at their actual
    /// record extents and Seismic's granules: logical slots may share one
    /// commitment, so a shrink is priced by what it releases.
    pub fn resident_charge(&self, slots: u64) -> Result<u64, String> {
        self.record_tensors().try_fold(0u64, |total, (group, kind, tensor)| {
            let held = self.reserved_records(group, kind, tensor, crate::planning::BankTier::Bank)?;
            held.record.get().checked_mul(self.records_at(group, slots))
                .and_then(|bytes| self.reservation().charge(held, self.records(group), bytes))
                .and_then(|charge| total.checked_add(charge))
                .ok_or_else(|| "bank record charge overflows".to_owned())
        })
    }

    /// Staging records only demand takes: a row's choices (`K`). Every
    /// earlier request's records are free once the device makes a request,
    /// so a one-row request always finds these, and a wider request's misses
    /// past them land as each record frees.
    pub fn demand_records(&self) -> u64 {
        self.selected()
    }

    /// Staging records hints may hold beyond the demand reserve: 2K, twice
    /// the experts a token selects per layer, and so the tier's reserve
    /// ([`Self::tier_reserve`]). Underived (ledger L30),
    /// pending the read-ahead design: it is to follow from that design's
    /// lead and precision, the planner reading the stored measurement
    /// records (read line, device peak, resident step). Derived from today's
    /// hints, 40–51% of whose reads go unused, it would grow the wrong
    /// thing.
    pub fn hint_records(&self) -> u64 {
        2 * self.selected()
    }

    /// The tier records each group keeps ready or graced, holding no expert
    /// (`HostRecords::refill`): the reads ahead's, a record for each staging
    /// record beyond demand's ([`Self::hint_records`]), so every read
    /// ahead's landing finds one. A read ahead lands off any park, so it can
    /// take only a record no step reads already; a demand landing, at its
    /// park, takes a ready one or reclaims a kept one the parked step does
    /// not read (`Directory::claim_demand`), so it needs none kept for it.
    /// Where the host places (unified memory) the tier is the bank's own
    /// records, which each group holds beside its slots: the one count of
    /// what the bank holds empty, which the plan charges and the bank's
    /// telemetry leaves out of the records its experts are held in.
    pub fn tier_reserve(&self) -> u64 {
        self.hint_records()
    }

    /// Records a scratch round converts: half a request's choices, so a
    /// round's reads land while the device converts the round before.
    pub fn round_records(&self) -> u64 {
        self.structure().round_records()
    }

    /// Rounds of a scratch-served layer: enough to convert every expert of
    /// the widest layer. A hit is read where the bank holds it; a round past
    /// a layer's last records neither parks nor converts.
    pub fn rounds(&self) -> u64 {
        self.widest().get().div_ceil(self.round_records())
    }

    /// Words of a convert list (`routed_convert`): the sequence and
    /// count, then a (conveyor record, scratch slot) pair per record.
    pub fn list_words(&self) -> u64 {
        list_extent(self.round_records())
    }

    /// Words of the tier list and the seed list (`routed_copy` from the host
    /// tier into scratch slots, and of seeds from their scratch slots into
    /// bank records, then `routed_seed_commit`): the sequence and count,
    /// then a pair per expert, at most a layer's experts.
    pub fn expert_list_words(&self) -> u64 {
        list_extent(self.widest().get())
    }

    /// The most records one shrink moves in group `group` (`routed_move`'s
    /// `L`): a shrink from `s` slots to `t` retires the records between
    /// `records_at(group, t)` and `records_at(group, s)` and moves at most the
    /// expert each holds, and no bank holds more than every expert or less
    /// than its floor, so the list is the bank's at any slots.
    pub fn move_records(&self, group: usize) -> u64 {
        self.records_at(group, self.widest().get()) - self.records_at(group, self.floor())
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
        let (mut needed, mut past) = (0, self.widest().get() + 1);
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
        self.slots == self.widest().get()
    }

    /// Whether growth reaches every expert ([`Self::ceiling`]): only such a
    /// bank is ever whole, each expert at its home record.
    pub fn reaches_whole(&self) -> bool {
        self.ceiling == self.widest().get()
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

    /// Every record tensor: its group, its routed tensor, and its record
    /// form.
    fn record_tensors(&self) -> impl Iterator<Item = (usize, RoutedTensor, RecordTensor)> + '_ {
        self.groups()
            .iter()
            .enumerate()
            .flat_map(|(group, each)| each.record.iter().map(move |(kind, tensor)| (group, kind, *tensor)))
    }

    /// What `allocations` demand, each as its reservation backs it.
    fn demand<'a>(&self, mut allocations: impl Iterator<Item = &'a BankAllocation>) -> Result<ClaimDemand, String> {
        allocations.try_fold(ClaimDemand::default(), |total, allocation| {
            total
                .checked_plus(allocation.storage_demand(&self.reservation_of(allocation)?)?)
                .ok_or_else(|| "bank storage demand overflows".to_owned())
        })
    }

    /// The reservation backing `allocation` where it is reserved, by its
    /// storage kind: host storage the host tier's, device storage the
    /// record tensors' (any other allocation's charge reads none).
    fn reservation_of(&self, allocation: &BankAllocation) -> Result<seismic::Reservation, String> {
        match allocation {
            BankAllocation::Reserved { kind: crate::planning::BankTier::Host, .. } => {
                self.tier_reservation().ok_or_else(|| "host-reserved records without a host tier".to_owned())
            }
            _ => Ok(self.reservation()),
        }
    }

    /// Every byte the bank is charged at its load: each allocation of its
    /// layout and the floor's placed words; a group's more places are
    /// charged as it holds them ([`Self::placed_bytes`]).
    pub fn bytes(&self) -> Result<u64, String> {
        self.storage_demand()?.checked_bytes().ok_or_else(|| "bank storage demand overflows".to_owned())
    }

    /// The physical demand of the layout and its floor's placed words.
    pub fn storage_demand(&self) -> Result<ClaimDemand, String> {
        let (layout, floor) = (self.layout()?, self.placed_allocations(NonZeroU64::MIN)?);
        self.demand(layout.iter().chain(&floor))
    }

    /// What layer `layer`'s banked expert stack of each routed tensor
    /// charges for its records: its slots' (`ModelLoadPlan::with_expert_bank`).
    pub fn banked_bytes(&self, layer: usize) -> Result<RecordTensors<u64>, String> {
        let slots = Self::layer_slots(self.slots, self.layers()[layer].experts);
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
    /// per-expert scales): everything else its layout holds.
    pub fn storage_beyond_records(&self) -> Result<ClaimDemand, String> {
        let mut demand = self.storage_demand()?;
        for (index, layer) in self.layers().iter().enumerate() {
            let records = self.banked_bytes(index)?;
            for ((_, plan), (_, bytes)) in layer.resident.iter().zip(records.iter()) {
                demand.reserved = demand.reserved.checked_sub(*bytes)
                    .ok_or("a bank holds less than its banked records charge")?;
                demand.device = demand.device.checked_sub(plan.scale.as_ref().map_or(0, |scale| scale.resident_bytes))
                    .ok_or("a bank holds less than its banked scales charge")?;
            }
        }
        Ok(demand)
    }

    /// The plan with its streamed layers replaced by `layers`, its
    /// structure's lending derived again ([`conveyor::Structured::new`]).
    #[cfg(test)]
    pub(crate) fn with_layers(self, layers: BankLayers) -> Self {
        let (selected, reservation, host, groups, _) = self.structured.into_parts();
        let structured = conveyor::Structured::new(selected, reservation, host, groups, layers)
            .expect("the layers keep the bank's lending");
        Self { structured, ..self }
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

    /// The resident plan of a banked expert stack: what tuning imports.
    pub fn resident_plan(&self, role: WeightRole) -> Option<&WeightPlan> {
        let kind = RoutedTensor::of(role.kind)?;
        self.layer(role.scope).and_then(|(_, layer)| layer.resident.get(kind))
    }
}

#[cfg(test)]
mod tests;
