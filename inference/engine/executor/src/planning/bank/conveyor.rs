use super::{ExpertBankPlan, HostTier, Placer, RecordTensor, RecordTensors, RoutedTensor};
use super::records::record_bytes;
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
/// form at `offsets[pool]` bytes into the lender's records of its routed
/// tensor's carrier (`RecordForm::carrier`), the pools one tensor carries
/// laid end to end, the conveyors in each reaching `bytes` of it
/// (`ExpertBankPlan::conveyor_layout`).
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
/// `offsets[f]` bytes into the lender's records of its routed tensor, the
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

impl ExpertBankPlan {
    /// Records of each stored form's conveyor (`V` of `routed_convert`,
    /// [`Self::conveyor`]): beside a host tier, a whole layer's experts,
    /// record `e` holding expert `e` of a layer, so every cold expert of a
    /// layer is read at once (the tier's records are plentiful host RAM);
    /// on the one store, whose records are the bank's slots, two rounds'
    /// records, all that is converted or landing at once, so a round's reads
    /// land while the device converts the round before and a record frees
    /// once its round is past (Metal's 4k prefill rose from 424 to 448
    /// tok/s at 6.89 GB with it, its records the bank's); at most the widest
    /// layer's experts, all a layer ever reads.
    pub fn conveyor_records(&self) -> u64 {
        let widest = self.widest().get();
        match self.host {
            HostTier::Linked(_) => widest,
            HostTier::Unified => (2 * self.round_records()).min(widest),
        }
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
    /// layout ([`Self::conveyor_layout`]) reaches, its packing waste
    /// included; a conveyor record lies whole in one native allocation of
    /// the lender's (a chunk of chunked storage), or the bank is refused.
    /// On the one store the lender lends the conveyor and the scratch
    /// together, from its bottom records ([`Self::scratch`]), so its lent
    /// records are both's. The plan refused a bank whose conveyor or scratch
    /// lies in no lender ([`Self::check_lending`]).
    pub fn conveyor(&self) -> ConveyorPlan {
        self.try_conveyor().expect("a planned bank's conveyor lies in its lender")
    }

    /// Where the scratch lies ([`Self::try_scratch`]), which the plan
    /// checked ([`Self::check_lending`]).
    pub fn scratch(&self) -> ScratchPlan {
        self.try_scratch().expect("a planned bank's scratch lies in its lender")
    }

    /// Refuse a bank whose conveyor or scratch no record of its lender holds
    /// whole, so every derivation past planning may rely on them.
    pub(super) fn check_lending(&self) -> Result<(), String> {
        self.try_conveyor()?;
        self.try_scratch().map(drop)
    }

    /// The conveyor ([`Self::conveyor`]), or why no lender holds it.
    fn try_conveyor(&self) -> Result<ConveyorPlan, String> {
        let group = self.lender_group();
        let conveyor = self.conveyor_alone()?;
        Ok(match self.host {
            HostTier::Unified => ConveyorPlan { carrier: ConveyorCarrier::Bank(self.try_scratch()?.lender), conveyor },
            HostTier::Linked(_) => {
                let records = lent_records(self.groups[group].record, &conveyor.bytes)?;
                ConveyorPlan { carrier: ConveyorCarrier::HostTier(Lender { group, records }), conveyor }
            }
        })
    }

    /// The conveyor's layout in the lender's records of the store the host
    /// reads files into ([`Self::conveyor`]).
    pub(super) fn conveyor_alone(&self) -> Result<ConveyorLayout, String> {
        let pools = self.staging_pools().0;
        let group = self.lender_group();
        let (store, reservation) = match self.host {
            HostTier::Unified => (crate::planning::BankTier::Bank, self.reservation),
            HostTier::Linked(tier) => (crate::planning::BankTier::Host, tier),
        };
        let carriers = self.groups[group].record.try_map(|kind, tensor| {
            Ok::<_, String>(reservation.of_records(self.reserved_records(group, kind, tensor, store)?))
        })?;
        self.conveyor_layout(&pools, &carriers, self.conveyor_records())
    }

    /// The first record of group `group` a prompt group's seeds take, and
    /// so the first its seed copy and commit bind: beside a host tier the
    /// scratch lies in the lender's bottom lent records ([`Self::scratch`]),
    /// which the group holds claimed while it seeds, so the lender's seeds
    /// lie past them and its seed copy writes a view of its records
    /// disjoint from the scratch it reads; every other group's, and every
    /// group's on the one store (where nothing seeds), from its first.
    pub fn seeded_from(&self, group: usize) -> u64 {
        match self.placer() {
            Placer::Tier => Some(self.scratch().lender).filter(|lender| lender.group == group).map_or(0, |lender| lender.records),
            Placer::Bank => 0,
        }
    }

    /// Where the scratch lies (`ScratchPlan`): one widest layer of every
    /// routed tensor's every resident form, which a prompt group's rounds
    /// convert each layer's cold experts into (slot `e` holding expert `e`)
    /// and its pass reads them from, each layer through the view of its own
    /// form: one scratch for every group, whatever its record form. It lies
    /// in the bottom records of the lender, in the bank's records the pass
    /// reads (device memory): on the one store past the conveyor there,
    /// beside a host tier from the first; every form of a tensor from the
    /// same byte, as a group converts one layer at a time. A prompt group holds them for its
    /// life and returns them empty, so it evicts no other layer's expert. A
    /// scratch record lies whole in one chunk of the lender's chunked
    /// storage ([`Self::reserved_records`]). Or why no lender holds it.
    fn try_scratch(&self) -> Result<ScratchPlan, String> {
        let group = self.lender_group();
        let (forms, experts) = self.scratch_forms();
        let start = match self.host {
            HostTier::Unified => self.conveyor_alone()?.bytes,
            HostTier::Linked(_) => self.form().each(|_| Ok::<_, String>(0))?,
        };
        let carriers = self.groups[group].record.try_map(|kind, tensor| {
            Ok::<_, String>(self.reservation.of_records(self.reserved_records(group, kind, tensor, crate::planning::BankTier::Bank)?))
        })?;
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
                let packing = carriers
                    .get(kind)
                    .ok_or_else(|| format!("{} scratch of a tensor its lender lacks", form.resident.name()))?
                    .packing(record_bytes(form.resident, form.matrix)?)
                    .ok_or_else(|| format!("{} scratch records fit no chunk of their lender", form.resident.name()))?;
                let offset = packing.at(from.next_multiple_of(alignment.max(1)), 0);
                offsets[index] = offset;
                bytes = bytes.max(packing.end(offset, rows));
            }
            Ok::<_, String>(bytes)
        })?;
        let records = lent_records(self.groups[group].record, &bytes)?;
        Ok(ScratchPlan { lender: Lender { group, records }, layout: ScratchLayout { experts, offsets, bytes } })
    }

    /// The scratch's forms ([`ScratchForm`]), each with the most experts of
    /// the layers of that form, its view's slots.
    pub(super) fn scratch_forms(&self) -> (Vec<ScratchForm>, Vec<u64>) {
        let (forms, layers) = self.layer_forms(|kind, tensor| ScratchForm { kind, resident: tensor.resident, matrix: tensor.matrix });
        let mut experts = vec![0u64; forms.len()];
        for (layer, of) in self.layers.iter().zip(&layers) {
            for (_, &form) in of.iter() {
                experts[form] = experts[form].max(layer.experts);
            }
        }
        (forms, experts)
    }

    /// Each layer's scratch form of each routed tensor
    /// ([`Self::scratch`]).
    pub fn layer_scratch_forms(&self) -> Vec<RecordTensors<usize>> {
        self.layer_forms(|kind, tensor| ScratchForm { kind, resident: tensor.resident, matrix: tensor.matrix }).1
    }

    /// Where each staging pool's conveyor of `records` records lies in the
    /// lender's records of its tensor's carrier (`RecordForm::carrier`), and
    /// the bytes the conveyors each carrier holds reach: those pools end to
    /// end, each `records` records of its
    /// stored form, from the first place past the last that its layout's
    /// alignment admits, where `lent` (each tensor's reservation of its
    /// records) packs them (`seismic::Reservation::packing`): contiguous on
    /// unchunked storage; on chunked storage whole in chunks, none crossing
    /// one, the packing running on across pools, so a chunk wastes less
    /// than one record past its last whole one, and the lending takes it
    /// with the records around it.
    pub(super) fn conveyor_layout(
        &self,
        pools: &[StagingPool],
        lent: &RecordTensors<seismic::Reservation>,
        records: u64,
    ) -> Result<ConveyorLayout, String> {
        let mut offsets = vec![0u64; pools.len()];
        let form = self.form();
        let bytes = form.each(|kind| {
            let mut bytes = 0u64;
            for (index, pool) in pools.iter().enumerate().filter(|(_, pool)| form.carrier(pool.kind) == kind) {
                let (_, alignment) =
                    seismic::Tensor::canonical_layout(pool.source, &[records, pool.matrix[0], pool.matrix[1]])
                        .map_err(|error| format!("{} conveyor: {error}", pool.source.name()))?;
                let aligned = bytes.next_multiple_of(alignment.max(1));
                let packing = lent
                    .get(kind)
                    .ok_or_else(|| format!("{} conveyor of a tensor its lender lacks", pool.source.name()))?
                    .packing(pool.record_bytes()?)
                    .ok_or_else(|| format!("{} conveyor records fit no chunk of their lender", pool.source.name()))?;
                let offset = packing.at(aligned, 0);
                offsets[index] = offset;
                bytes = packing.end(offset, records);
            }
            Ok::<_, String>(bytes)
        })?;
        Ok(ConveyorLayout { records, offsets, bytes })
    }

    /// The staging pools, and each layer's pool of each routed tensor: one
    /// per stored form, each with its conveyor.
    pub fn staging_pools(&self) -> (Vec<StagingPool>, Vec<RecordTensors<usize>>) {
        self.layer_forms(StagingPool::of)
    }

    /// The distinct forms `form` makes of each layer's record of each
    /// routed tensor, in order of first use, and each layer's index of its
    /// own of each.
    fn layer_forms<F: PartialEq>(
        &self,
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
