use super::{BankGroup, BankLayers, ExpertBankPlan, HostTier, RecordForm, RecordTensor, RecordTensors, RoutedTensor};
use super::records::{decode_choices, decode_set, record_bytes, record_slab};
use seismic::Element;

/// A staging pool: records of one routed tensor in one stored form, which
/// demand reads land in and the fills convert from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StagingPool {
    pub kind: RoutedTensor,
    pub source: Element,
    pub matrix: [u64; 2],
}

impl StagingPool {
    /// The pool routed tensor `kind`'s records of `tensor` are staged in:
    /// one per routed tensor and stored form, so no two of a record's
    /// tensors share one.
    pub fn of(kind: RoutedTensor, tensor: RecordTensor) -> Self {
        Self {
            kind,
            source: tensor.source,
            matrix: tensor.matrix,
        }
    }

    /// One expert as stored: a record of the pool, and of its conveyor.
    pub fn record_bytes(&self) -> Result<u64, String> {
        record_bytes(self.source, self.matrix)
    }
}

/// The group whose bottom `records` records carry the conveyor, which a
/// prompt group holds for its life and returns empty: records of the store
/// the host reads files into ([`ConveyorCarrier`]), which the group's
/// floor commits, so no resize retires them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Lender {
    pub group: usize,
    pub records: u64,
}

/// Whose records carry the conveyor (`ConveyorPlan`), which the host reads
/// a prompt group's cold experts into and the rounds convert from in
/// place: the store the host addresses, by how the bank's misses are held
/// ([`HostTier`]). On unified memory the one store's bank records; beside a
/// discrete device the host tier's records, which the device reads over
/// the link, never its own memory. No other carrier exists.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConveyorCarrier {
    Bank(Lender),
    HostTier(Lender),
}

impl ConveyorCarrier {
    /// The lender, whichever store it lends from.
    pub fn lender(self) -> Lender {
        match self {
            Self::Bank(lender) | Self::HostTier(lender) => lender,
        }
    }
}

/// Where each staging pool's conveyor lies: `records` records of its stored
/// form at `offsets[pool]` bytes into the lender's lent records of its
/// routed tensor's carrier (`RecordForm::carrier`), all pools one tensor
/// carries sharing the same bytes from offset 0 (like [`ScratchLayout`]),
/// the conveyors in each reaching `bytes` of it
/// (`BankStructure::conveyor_layout`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConveyorLayout {
    pub records: u64,
    pub offsets: Vec<u64>,
    pub bytes: RecordTensors<u64>,
}

/// A scratch form: records of routed tensor `kind` in one resident form of
/// `matrix`, which the layers of that form read cold experts from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScratchForm {
    pub kind: RoutedTensor,
    pub resident: Element,
    pub matrix: [u64; 2],
}

/// Where each scratch form's view lies: `experts[f]` records of form `f` at
/// `offsets[f]` bytes into the lender's lent records of its routed tensor, the
/// views of each tensor reaching `bytes` of it (on the one store past the
/// conveyor there) ([`ExpertBankPlan::scratch`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScratchLayout {
    pub experts: Vec<u64>,
    pub offsets: Vec<u64>,
    pub bytes: RecordTensors<u64>,
}

/// Who lends the scratch, its lent records (on the one store the
/// conveyor's too), and where each form's view lies.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScratchPlan {
    pub lender: Lender,
    pub layout: ScratchLayout,
}

/// The lender's records `bytes` of each of its record tensors reach: the
/// most of any tensor, each counted in its own records.
fn lent_records(record: RecordTensors<RecordTensor>, bytes: &RecordTensors<u64>) -> Result<u64, String> {
    record.iter().try_fold(0u64, |most, (kind, tensor)| {
        let record = tensor.record_bytes()?;
        Ok(most.max(bytes.get(kind).map_or(0, |&bytes| bytes.div_ceil(record))))
    })
}

/// What carries the conveyor, and the conveyor the scratch rounds' converts
/// read (`V` its records).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConveyorPlan {
    pub carrier: ConveyorCarrier,
    pub conveyor: ConveyorLayout,
}

/// What a bank lends a prompt group, derived once from its structure as its
/// plan is made ([`BankStructure::lending`]): where its conveyor lies
/// ([`ExpertBankPlan::conveyor`]) and where its scratch does
/// ([`ExpertBankPlan::scratch`]). A bank whose conveyor or scratch has no
/// layout has no plan, so every sizing and binding past planning reads them
/// here.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Lending {
    conveyor: ConveyorPlan,
    scratch: ScratchPlan,
}

/// A bank's structure with what derives from it: made only by
/// [`Self::new`], which derives its lending and its pools' granules, and
/// never changed after, so its conveyor, scratch and granules are always
/// its structure's.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Structured {
    selected: u64,
    host: HostTier,
    groups: Vec<BankGroup>,
    layers: BankLayers,
    /// `W`: scratch window slots (`round_records <= W <= widest`).
    window_slots: u64,
    lending: Lending,
    /// Deduplicated distinct record-slab granules across groups (1 for uniform
    /// models), in order of first appearance.
    granules: Vec<u64>,
    /// Index into `granules` for each bank group.
    group_pool: Vec<usize>,
    /// Index into `granules` of the largest granule, which banked stored tables
    /// (`BankTable`) claim from.
    table_pool: usize,
}

impl Structured {
    /// The structure of `groups` over `layers`, `selected` experts a row,
    /// beside `host`, with its scratch bounded to `window_slots` (defaulting
    /// to `widest`), its lending ([`BankStructure::lending`]) and each group's
    /// record slab granule ([`ExpertBankPlan::granules`]); or why its lending
    /// or a granule cannot be made.
    pub(super) fn new(
        selected: u64,
        host: HostTier,
        groups: Vec<BankGroup>,
        layers: BankLayers,
        window_slots: Option<u64>,
    ) -> Result<Self, String> {
        let widest = layers.widest().get();
        let window_slots = window_slots.unwrap_or(widest);
        let structure = BankStructure { groups: &groups, layers: &layers, selected, host, window_slots };
        let round = structure.round_records();
        if window_slots < round || window_slots > widest {
            return Err(format!("scratch window slots {window_slots} must lie in {round}..={widest}"));
        }
        let lending = structure.lending()?;
        let mut granules = Vec::new();
        let mut group_pool = Vec::with_capacity(groups.len());
        for group in &groups {
            let granule = record_slab(&group.record)?.granule()?;
            let pool = match granules.iter().position(|&known| known == granule) {
                Some(index) => index,
                None => {
                    granules.push(granule);
                    granules.len() - 1
                }
            };
            group_pool.push(pool);
        }
        let table_pool = granules
            .iter()
            .enumerate()
            .max_by_key(|&(_, &g)| g)
            .expect("a bank holds a group")
            .0;
        Ok(Self { selected, host, groups, layers, window_slots, lending, granules, group_pool, table_pool })
    }

    /// [`ExpertBankPlan::granules`].
    pub(super) fn granules(&self) -> &[u64] {
        &self.granules
    }

    /// [`ExpertBankPlan::group_pools`].
    pub(super) fn group_pools(&self) -> &[usize] {
        &self.group_pool
    }

    /// [`ExpertBankPlan::table_pool`].
    pub(super) fn table_pool(&self) -> usize {
        self.table_pool
    }

    /// [`ExpertBankPlan::table_granule`].
    pub(super) fn table_granule(&self) -> u64 {
        self.granules[self.table_pool]
    }

    /// [`ExpertBankPlan::window_slots`].
    pub(super) fn window_slots(&self) -> u64 {
        self.window_slots
    }

    /// Its parts, which [`Self::new`] makes a structure of again.
    #[cfg(test)]
    pub(super) fn into_parts(self) -> (u64, HostTier, Vec<BankGroup>, BankLayers) {
        (self.selected, self.host, self.groups, self.layers)
    }
}

/// The part of a bank's plan its lending derives from: its groups and
/// layers, the experts a row selects, its host tier, and its scratch window
/// slots (`W`), none of which changes once the plan is made.
#[derive(Clone, Copy)]
pub(super) struct BankStructure<'a> {
    pub(super) groups: &'a [BankGroup],
    pub(super) layers: &'a BankLayers,
    pub(super) selected: u64,
    pub(super) host: HostTier,
    pub(super) window_slots: u64,
}

impl ExpertBankPlan {
    /// The structure its lending derives from ([`BankStructure`]).
    pub(super) fn structure(&self) -> BankStructure<'_> {
        let structured = &self.structured;
        BankStructure {
            groups: &structured.groups,
            layers: &structured.layers,
            selected: structured.selected,
            host: structured.host,
            window_slots: structured.window_slots,
        }
    }

    /// `K`: the experts a row selects.
    pub fn selected(&self) -> u64 {
        self.structured.selected
    }

    /// The deduplicated distinct record-slab granules across the bank's groups
    /// (1 for uniform models), in order of first appearance: each group claims
    /// records from the pool of its own granule ([`Self::group_pool`],
    /// [`Self::group_granule`]), so smaller groups are never charged a larger
    /// group's slab size.
    pub fn granules(&self) -> &[u64] {
        self.structured.granules()
    }

    /// The index into [`Self::granules`] (and [`Self::blocks`]) for bank group `group`.
    pub fn group_pool(&self, group: usize) -> usize {
        self.structured.group_pools()[group]
    }

    /// Each bank group's index into [`Self::granules`].
    pub fn group_pools(&self) -> &[usize] {
        self.structured.group_pools()
    }

    /// The bytes of a granule of group `group`'s records pool.
    pub fn group_granule(&self, group: usize) -> u64 {
        self.structured.granules()[self.group_pool(group)]
    }

    /// The index into [`Self::granules`] of the pool holding banked stored
    /// tables (`BankTable`): the largest granule pool.
    pub fn table_pool(&self) -> usize {
        self.structured.table_pool()
    }

    /// The granule bytes of the pool holding banked stored tables (`BankTable`).
    pub fn table_granule(&self) -> u64 {
        self.structured.table_granule()
    }

    /// What its device gives it ([`BankSubstrate`]).
    pub fn substrate(&self) -> super::BankSubstrate {
        super::BankSubstrate { backend: self.backend, allocations: self.allocations, host: self.structured.host }
    }

    /// How the host tier is held on the device (`platform::host_tier`): a
    /// bank streams only beside one, so a device without one holds its
    /// experts resident or is refused (`UnstreamedExperts::Untiered`), and
    /// every bank has a store its conveyor lies in. The tier is every miss's first store,
    /// inclusive of the bank, in the bank's own form: each group's record
    /// tensors again, resident as the device holds them, in the tier's own
    /// granule pool for every expert of the group's layers
    /// ([`Self::host_extent`]) and committed by host slots
    /// ([`Self::host_records`]), so the host lands file misses into it and
    /// the device reads it in place; where it lies across a link the host
    /// also promotes from it by a pure copy on the copy engine
    /// (`watcher::promotions`), and nothing is written back.
    pub fn host(&self) -> HostTier {
        self.structured.host
    }

    /// Its groups of layers of one record form.
    pub fn groups(&self) -> &[BankGroup] {
        &self.structured.groups
    }

    /// Every streamed routed layer, in decoder order.
    pub fn layers(&self) -> &BankLayers {
        &self.structured.layers
    }

    /// [`BankStructure::conveyor_records`].
    pub fn conveyor_records(&self) -> u64 {
        self.structure().conveyor_records()
    }

    /// Where the conveyor lies (`ConveyorPlan`): read for a prompt group's
    /// layer as its steps' requests there are published, into records the
    /// host addresses, which the rounds' converts read in place (over the
    /// link on a discrete device). It works only while a prompt group runs,
    /// which holds its records for its life and returns them empty. Its
    /// records are the bottom records of the group of the most experts
    /// (the lender) in the store the host reads files into: on unified
    /// memory the bank's own (the one store, [`ConveyorCarrier::Bank`]);
    /// beside a discrete device the host tier's
    /// ([`ConveyorCarrier::HostTier`]), as the device's own memory is no
    /// memory the host reads into. The records lent are every one the lent
    /// layout ([`BankStructure::conveyor_layout`]) reaches, contiguous
    /// storage of their own (`BankAllocation::Lent`). On the one store
    /// the lender lends the conveyor and the scratch together, from its
    /// bottom records ([`Self::scratch`]), so its lent records are both's.
    pub fn conveyor(&self) -> &ConveyorPlan {
        &self.structured.lending.conveyor
    }

    /// Where the scratch lies ([`BankStructure::scratch`]).
    pub fn scratch(&self) -> &ScratchPlan {
        &self.structured.lending.scratch
    }

    /// The group whose records carry the conveyor and the scratch
    /// ([`BankStructure::lender_group`]).
    pub(super) fn lender_group(&self) -> usize {
        self.structured.lending.scratch.lender.group
    }

    /// The first record of group `group` a prompt group's seeds take, and
    /// so the first its seed copy and commit bind: the scratch (past the
    /// conveyor on the one store) lies in the lender's bottom lent records
    /// ([`Self::scratch`]), which the group holds claimed while it seeds, so
    /// the lender's seeds lie past them and its seed copy writes a view of
    /// its records disjoint from the scratch it reads; every other group's
    /// from its first.
    pub fn seeded_from(&self, group: usize) -> u64 {
        Some(self.scratch().lender).filter(|lender| lender.group == group).map_or(0, |lender| lender.records)
    }

    /// Each layer's scratch form of each routed tensor
    /// ([`Self::scratch`]).
    pub fn layer_scratch_forms(&self) -> Vec<RecordTensors<usize>> {
        self.structure().layer_forms(|kind, tensor| ScratchForm { kind, resident: tensor.resident, matrix: tensor.matrix }).1
    }

    /// [`BankStructure::staging_pools`].
    pub fn staging_pools(&self) -> (Vec<StagingPool>, Vec<RecordTensors<usize>>) {
        self.structure().staging_pools()
    }
}

impl BankStructure<'_> {
    /// The most experts of any streamed layer ([`ExpertBankPlan::widest`]).
    pub(super) fn widest(self) -> u64 {
        self.layers.widest().get()
    }

    /// [`ExpertBankPlan::choices`].
    pub(super) fn choices(self) -> u64 {
        decode_choices(self.selected)
    }

    /// [`ExpertBankPlan::round_records`].
    pub(super) fn round_records(self) -> u64 {
        decode_set(self.selected, self.widest()).div_ceil(2)
    }

    /// [`ExpertBankPlan::form`].
    pub(super) fn form(self) -> RecordForm {
        self.groups[0].record.form()
    }

    /// The group whose records carry the conveyor (`Lender`): the one of the
    /// most experts, the first of those.
    fn lender_group(self) -> usize {
        (0..self.groups.len())
            .max_by_key(|&group| {
                let experts = self.groups[group].layers.iter().map(|&layer| self.layers[layer].experts).sum::<u64>();
                (experts, std::cmp::Reverse(group))
            })
            .expect("a bank holds a group")
    }

    /// Records of each stored form's conveyor (`V` of `routed_round`,
    /// [`ExpertBankPlan::conveyor`]): two rounds' records, all that is
    /// converted or landing at once, bounded by `window_slots`, uniformly
    /// across both unified memory and discrete hosts (so a round's reads land
    /// while the device converts the round before and a record frees once its
    /// round is past, without stealing hundreds of cached experts from the
    /// host tier).
    pub(super) fn conveyor_records(self) -> u64 {
        (2 * self.round_records()).min(self.window_slots)
    }

    /// The bank's lending ([`Lending`]), or why its conveyor or its scratch
    /// has no layout.
    pub(super) fn lending(self) -> Result<Lending, String> {
        let conveyor = self.conveyor_alone()?;
        let scratch = self.scratch(&conveyor)?;
        let carrier = match self.host {
            HostTier::Unified(_) => ConveyorCarrier::Bank(scratch.lender),
            HostTier::Linked => {
                let group = self.lender_group();
                let records = lent_records(self.groups[group].record, &conveyor.bytes)?;
                ConveyorCarrier::HostTier(Lender { group, records })
            }
        };
        Ok(Lending { conveyor: ConveyorPlan { carrier, conveyor }, scratch })
    }

    /// The conveyor's layout in the lender's lent records of the store the
    /// host reads files into ([`ExpertBankPlan::conveyor`]).
    pub(super) fn conveyor_alone(self) -> Result<ConveyorLayout, String> {
        self.conveyor_layout(&self.staging_pools().0, self.conveyor_records())
    }

    /// Where the scratch lies (`ScratchPlan`) beside the conveyor laid out
    /// as `conveyor`: one widest layer of every
    /// routed tensor's every resident form, which a prompt group's rounds
    /// convert each layer's cold experts into (slot `e` holding expert `e`)
    /// and its pass reads them from, each layer through the view of its own
    /// form: one scratch for every group, whatever its record form. It lies
    /// in the bottom records of the lender, in the bank's lent records the
    /// pass reads (device memory): on the one store past the conveyor there,
    /// beside a host tier from the first; every form of a tensor from the
    /// same byte, as a group converts one layer at a time. A prompt group
    /// holds them for its life and returns them empty, so it evicts no other
    /// layer's expert. Or why a form has no layout.
    fn scratch(self, conveyor: &ConveyorLayout) -> Result<ScratchPlan, String> {
        let group = self.lender_group();
        let (forms, experts) = self.scratch_forms();
        let start = match self.host {
            HostTier::Unified(_) => conveyor.bytes,
            HostTier::Linked => self.form().each(|_| Ok::<_, String>(0))?,
        };
        let mut offsets = vec![0u64; forms.len()];
        // A prompt group converts one layer at a time, so every form of a
        // tensor shares the same bytes: each view lies from the tensor's
        // start, and the region reaches the farthest.
        let bytes = self.form().each(|kind| {
            let from = *start.get(kind).expect("each tensor of the form");
            let mut bytes = from;
            for (index, form) in forms.iter().enumerate().filter(|(_, form)| form.kind == kind) {
                let rows = experts[index];
                let (_, alignment) = seismic::Tensor::canonical_layout(form.resident, &[rows, form.matrix[0], form.matrix[1]])
                    .map_err(|error| format!("{} scratch: {error}", form.resident.name()))?;
                let offset = from.next_multiple_of(alignment.max(1));
                offsets[index] = offset;
                let end = record_bytes(form.resident, form.matrix)?
                    .checked_mul(rows)
                    .and_then(|view| view.checked_add(offset))
                    .ok_or_else(|| format!("{} scratch bytes overflow", form.resident.name()))?;
                bytes = bytes.max(end);
            }
            Ok::<_, String>(bytes)
        })?;
        let records = lent_records(self.groups[group].record, &bytes)?;
        Ok(ScratchPlan { lender: Lender { group, records }, layout: ScratchLayout { experts, offsets, bytes } })
    }

    /// The scratch's forms ([`ScratchForm`]), each with the most experts of
    /// the layers of that form bounded by `window_slots`, its view's slots.
    pub(super) fn scratch_forms(self) -> (Vec<ScratchForm>, Vec<u64>) {
        let (forms, layers) = self.layer_forms(|kind, tensor| ScratchForm { kind, resident: tensor.resident, matrix: tensor.matrix });
        let mut experts = vec![0u64; forms.len()];
        for (layer, of) in self.layers.iter().zip(&layers) {
            for (_, &form) in of.iter() {
                experts[form] = experts[form].max(layer.experts.min(self.window_slots));
            }
        }
        (forms, experts)
    }

    /// Where each staging pool's conveyor of `records` records lies in the
    /// lender's lent records of its tensor's carrier (`RecordForm::carrier`),
    /// and the bytes the conveyors each carrier holds reach: all pools carried
    /// by the same tensor share one region from offset 0 (aligned for each
    /// pool), reaching the largest pool's `records` records, mirroring
    /// [`Self::scratch`].
    pub(super) fn conveyor_layout(self, pools: &[StagingPool], records: u64) -> Result<ConveyorLayout, String> {
        let mut offsets = vec![0u64; pools.len()];
        let form = self.form();
        let bytes = form.each(|kind| {
            let mut bytes = 0u64;
            for (index, pool) in pools.iter().enumerate().filter(|(_, pool)| form.carrier(pool.kind) == kind) {
                let (_, alignment) =
                    seismic::Tensor::canonical_layout(pool.source, &[records, pool.matrix[0], pool.matrix[1]])
                        .map_err(|error| format!("{} conveyor: {error}", pool.source.name()))?;
                let offset = 0u64.next_multiple_of(alignment.max(1));
                offsets[index] = offset;
                let end = pool
                    .record_bytes()?
                    .checked_mul(records)
                    .and_then(|conveyor| conveyor.checked_add(offset))
                    .ok_or_else(|| format!("{} conveyor bytes overflow", pool.source.name()))?;
                bytes = bytes.max(end);
            }
            Ok::<_, String>(bytes)
        })?;
        Ok(ConveyorLayout { records, offsets, bytes })
    }

    /// The staging pools, and each layer's pool of each routed tensor: one
    /// per stored form, each with its conveyor.
    pub(super) fn staging_pools(self) -> (Vec<StagingPool>, Vec<RecordTensors<usize>>) {
        self.layer_forms(StagingPool::of)
    }

    /// The distinct forms `form` makes of each layer's record of each
    /// routed tensor, in order of first use, and each layer's index of its
    /// own of each.
    fn layer_forms<F: PartialEq>(
        self,
        form: impl Fn(RoutedTensor, RecordTensor) -> F,
    ) -> (Vec<F>, Vec<RecordTensors<usize>>) {
        let mut forms: Vec<F> = Vec::new();
        let layers = self
            .layers
            .iter()
            .map(|layer| {
                self.groups[layer.group].record.map(|kind, tensor| {
                    let each = form(kind, tensor);
                    match forms.iter().position(|known| *known == each) {
                        Some(index) => index,
                        None => {
                            forms.push(each);
                            forms.len() - 1
                        }
                    }
                })
            })
            .collect();
        (forms, layers)
    }
}
