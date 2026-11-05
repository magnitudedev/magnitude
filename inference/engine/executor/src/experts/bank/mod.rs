//! A streamed load's expert bank on the device: each group's record tensors,
//! every layer's and group's words, the
//! staging, and a discrete device's host tier, all
//! allocated at load and empty (no warming), and the watcher that serves
//! its steps.

mod open;
mod resize;
mod served;

pub use super::watcher::{ExpertCosts, GroupCost, GroupLayer, LayerTimeline, Span, StepTimeline, LayerCost, LayerExperts, StepCost, STEP_TRACE};
pub(crate) use super::watcher::{ExpertStep, Recording, StepFailure, StepRoute};
use super::source::Records;
use super::watcher::{PlacedLayerWords, PlacedWords, Watcher};
use super::Words;
use crate::planning::{BankLayout, BankPart, ExpertBankPlan, PlacedAllocations, Placer, RecordTensors, RoutedTensor};
use crate::domain::{BankResize, DomainError, ExpertResidency, HostResidency};
use crate::programs::Chain;
use magnitude_family_contracts::{WeightRole, WeightScope};
use seismic::{Device, Tensor};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use super::{copies, watcher, HostCompute, Machine};
use resize::{Commitment, HostTier, ResizeThread};
pub(crate) use served::ServedBank;

/// The bytes a shrink of the host tier releases, so that it holds the
/// submitting thread no longer than a decode `step`, in seconds, at the
/// `rate` the last such shrink released them. Without either (a rate no
/// release measured yet, or a step no miss record prices), the least that
/// releases storage, which measures the rate; the ladder takes the rest a
/// shrink at a time, at the boundaries after it.
fn landed_bytes(step: Option<f64>, rate: Option<f64>) -> u64 {
    step.zip(rate).map_or(0, |(step, rate)| (rate * step) as u64)
}

/// The holdings the bank itself resizes (`BankHolding`): its experts'
/// slots and its host tier's host slots. A stored table's records are the
/// served bank's (`ServedBank`).
#[derive(Clone, Copy, Debug)]
pub(crate) enum BankSlots {
    Experts,
    Host,
}

/// A stored table the bank holds: its plan and the table.
pub(crate) struct HeldTable {
    pub plan: crate::planning::BankTable,
    pub table: Arc<crate::stored_table::StoredTable>,
    /// Its records' last finished resize ([`ElasticBank::record_resize`]).
    pub last_resize: Mutex<Option<BankResize>>,
}

/// `value` slots as the record counts them.
fn count(value: u64) -> Result<usize, DomainError> {
    usize::try_from(value).map_err(|_| DomainError::invariant("bank slots exceed the host"))
}

/// Write `blocks` as the first word of a list copy's `grid`, which is its
/// blocks: its row's threadgroups.
fn write_blocks(grid: &seismic::Tensor, blocks: u32) -> Result<(), String> {
    let words = [blocks as i32, 0, 1].iter().flat_map(|word| word.to_le_bytes()).collect::<Vec<_>>();
    grid.clone().write_from_host(&words).map_err(|error| format!("a list grid's blocks: {error}"))
}

/// A staging pool's conveyor, a view of the lender's records, which reads
/// land in, the converts read in place, and the scratch rounds' waits bind
/// as the host's (`BankPart::Conveyor`).
struct PoolConveyor {
    conveyor: Tensor,
}

pub(crate) struct ExpertBank {
    /// Every allocation the plan lays out (`ExpertBankPlan::layout`).
    layout: BankLayout<Tensor>,
    /// The placed parts' words of the floor's one place, held for the load,
    /// and those of a group's more places while one holds them
    /// (`Self::hold_places`), which every step binds then.
    floor: PlacedAllocations<Tensor>,
    held: Mutex<Option<PlacedAllocations<Tensor>>>,
    /// The device the bank's words are held on.
    device: Device,
    /// Each layer's staging pool of each routed tensor
    /// (`ExpertBankPlan::staging_pools`).
    pool_of: Vec<RecordTensors<usize>>,
    /// Each staging pool's conveyor views.
    conveyors: Vec<PoolConveyor>,
    /// Each layer's scratch tier view of each routed tensor: its form's view
    /// of the scratch the lender lends prompt groups, its leading experts
    /// (`BankPart::Scratch`).
    scratch: Vec<RecordTensors<Tensor>>,
    /// A host tier's record tensors' resizers and the host slots they
    /// commit; none on unified memory.
    host: Option<Mutex<HostTier>>,
    /// Each layer's slot map: its view of its group's map
    /// (`BankPart::SlotMap`).
    slot_maps: Vec<Tensor>,
    watcher: Watcher,
    /// Each streamed layer's lines as the load measured them
    /// (`assessment::BankLines::of_bank`), each sited layer's CPU lines and
    /// each layer's landing line among them where the bank measured them.
    lines: std::sync::Arc<[crate::assessment::BankLines]>,
    /// Each group's move list (`routed_move`), over its layout's.
    move_lists: Vec<Words>,
    /// What the bank commits, and the resize running.
    commitment: Mutex<Commitment>,
    /// Its experts' slots' last finished resize
    /// ([`ElasticBank::record_resize`]).
    last_resize: Mutex<Option<BankResize>>,
    /// The thread the storage work of its records' resizes runs on (a host
    /// tier's runs on its own, `HostTier::thread`).
    resize_thread: ResizeThread,
    /// Held across every rise of the bank's charge off the domain's thread
    /// ([`Self::hold_rises`]).
    rises: Arc<Mutex<()>>,
    /// The plan the bank opened with: its floor, its ceiling, its granule
    /// and each group's records at any slots (`ExpertBankPlan::records_at`).
    plan: ExpertBankPlan,
    /// The record tensors the host lands file misses into, group by group
    /// ([`landed_tensors`]); none where it lands into none.
    landed: Option<Arc<[RecordTensors<Tensor>]>>,
}

/// The threads a bank's resizes keep busy beside decode: the host tier's
/// [`ResizeThread`], which backs the tier beside early decode; its records'
/// own runs only a resize's storage work, never for a decode's length.
pub(in crate::experts) const RESIZE_THREADS: usize = 1;

impl Drop for ExpertBank {
    /// A running growth ends at its next slot, so the resize thread, which
    /// the bank's fields join as they drop, backs nothing more for a bank
    /// that is gone.
    fn drop(&mut self) {
        for tier in crate::planning::BankTier::ALL {
            self.stop_growth(tier);
        }
    }
}

impl ExpertBank {
    /// Starve the load's host for `duration` from now: its pulse does not
    /// beat and no answer is written meanwhile, so a wait that spins then
    /// ends before its answer (a test's; `Pulse::starve`).
    #[cfg(test)]
    pub(crate) fn starve(&self, duration: std::time::Duration) {
        self.watcher.starve(duration);
    }

    /// Import every expert of a whole bank (`ExpertBankPlan::whole`) into
    /// its home record before any step, as the watcher started holding it:
    /// each layer's stack of each routed tensor by `import` (its resident
    /// plan and the records it lands in, as the resident stack's
    /// `[E, N, K]`).
    pub fn import_whole(
        &self,
        import: &mut dyn FnMut(&crate::WeightPlan, &Tensor) -> Result<(), String>,
    ) -> Result<(), String> {
        let plan = &self.plan;
        for (index, layer) in plan.layers.iter().enumerate() {
            let records = &self.layout.groups[layer.group].records;
            for (((_, record), (_, records)), (_, resident)) in
                plan.groups[layer.group].record.iter().zip(records.iter()).zip(layer.resident.iter())
            {
                let stack = records
                    .region(
                        record.resident,
                        &[layer.experts, record.matrix[0], record.matrix[1]],
                        plan.home(index) * record.record_bytes()?,
                    )
                    .map_err(|error| format!("a layer's home records: {error}"))?;
                import(resident, &stack)?;
            }
        }
        Ok(())
    }

    fn commitment(&self) -> std::sync::MutexGuard<'_, Commitment> {
        self.commitment.lock().expect("bank commitment")
    }

    /// The plan the bank opened with.
    pub fn plan(&self) -> &ExpertBankPlan {
        &self.plan
    }

    /// A tier copy takes `blocks` from now on (`assessment::TierCopyCosts`,
    /// which the job measured).
    pub fn tier_copy_blocks(&self, blocks: u32) -> Result<(), String> {
        write_blocks(&self.layout.load.tier_grid, blocks)
    }

    /// Each bank group's convert in a decode barrier takes its `lines` from
    /// now on (`assessment::ConvertCosts`, which the job measured): a miss
    /// lands wherever its line is sooner.
    pub fn device_convert(&self, lines: &[crate::assessment::RecordLine]) {
        self.watcher.shared().device_convert(lines);
    }

    /// Each streamed layer's host lines as the load measured them: what the
    /// load's misses are priced along, with the GPU's.
    pub fn lines(&self) -> std::sync::Arc<[crate::assessment::BankLines]> {
        self.lines.clone()
    }

    /// The slots each layer commits now.
    pub fn slots(&self) -> u64 {
        self.commitment().slots.load(Ordering::Acquire)
    }

    /// The host slots the bank's host tier commits now, and the most it
    /// holds; none without a tier.
    pub fn host_tier(&self) -> Option<(u64, std::num::NonZeroU64)> {
        self.host.as_ref().map(|host| {
            let tier = host.lock().expect("host tier");
            (tier.slots.load(Ordering::Acquire), tier.ceiling)
        })
    }

    /// The host slots the bank holds now: its host tier's, and none of a
    /// bank without a tier.
    pub fn host_slots(&self) -> u64 {
        self.host_tier().map_or(0, |(slots, _)| slots)
    }

    /// Record `resize` as the last of `slots`, its experts' or its host
    /// tier's ([`ElasticBank::record_resize`]).
    pub fn record_resize(&self, slots: BankSlots, resize: BankResize) {
        match slots {
            BankSlots::Experts => *self.last_resize.lock().expect("the experts' last resize") = Some(resize),
            BankSlots::Host => {
                let tier = self.host.as_ref().expect("a host tier's resize is of a bank with one");
                tier.lock().expect("host tier").last_resize = Some(resize);
            }
        }
    }

    /// [`ElasticBank::residency`]: the slots committed now, and the record
    /// tensors' charge in each store, as Seismic measures it.
    fn residency(&self) -> Result<ExpertResidency, DomainError> {
        let slots = self.slots();
        let mut stores = std::collections::BTreeMap::<(bool, seismic::StoreIdentity), u64>::new();
        for tensor in self.layout.groups.iter().flat_map(|group| group.records.iter().map(|(_, tensor)| tensor)) {
            let device = tensor.device();
            let topology = device.topology();
            let charges = tensor
                .observe_storage()
                .store_charges()
                .ok_or_else(|| DomainError::invariant("a held allocation is charged"))?;
            for (store, bytes) in charges {
                let identity = topology
                    .store_identity(store)
                    .ok_or_else(|| DomainError::invariant("a charged store belongs to the topology"))?;
                *stores.entry((identity == seismic::StoreIdentity::HostRam, identity)).or_default() += bytes;
            }
        }
        // Each layer's experts the bank holds and those its host tier holds
        // alone, as the watcher published them together: the rest only the
        // file holds.
        let (held, alone) = self.watcher.shared().holdings();
        let mut disk_bytes = 0u64;
        for ((layer, &held), &alone) in self.plan.layers.iter().zip(&held).zip(&alone) {
            disk_bytes += (layer.experts - held as u64 - alone as u64) * self.stored_bytes_of(layer.group)?;
        }
        Ok(ExpertResidency {
            experts: self.plan.widest(),
            slots,
            floor: self.plan.floor(),
            ceiling: self.plan.ceiling,
            stores: stores.into_iter().map(|((_, store), bytes)| (store, bytes)).collect(),
            disk_bytes,
            // A tier at its slots, up to its ceiling.
            host: self
                .host
                .as_ref()
                .map(|tier| {
                    let tier = tier.lock().expect("host tier");
                    (tier.slots.load(Ordering::Acquire), tier.ceiling, tier.last_resize)
                })
                .map(|(slots, ceiling, last_resize)| {
                    let fill = self.watcher.shared().tier_fill();
                    self.host_charged().map(|bytes| HostResidency { slots, ceiling, bytes, fill, last_resize })
                })
                .transpose()?,
            // The tables are the served bank's (`ServedBank::residency`).
            tables: Vec::new(),
            last_resize: *self.last_resize.lock().expect("the experts' last resize"),
        })
    }

    /// The bytes the model file stores one expert of group `group` in,
    /// summed over its record's routed tensors.
    fn stored_bytes_of(&self, group: usize) -> Result<u64, DomainError> {
        self.plan.groups[group]
            .record
            .iter()
            .try_fold(0u64, |sum, (_, tensor)| Ok::<_, String>(sum + tensor.stored_bytes()?))
            .map_err(DomainError::invariant)
    }

    /// Begin the steps of a chain of `launches`, each routed as `route`
    /// says, a group the device runs layer-major (a lone step a group of
    /// one), `scratch` saying whether they are scratch-served, from the whole
    /// words it reads as they hold for the group (`Shared::begin`): from now
    /// until each ends, the watcher serves it. Each launch comes with its
    /// step, and with them each streamed layer's scope and merge set, the
    /// steps the group ends with whose backs run in one pass; why the group
    /// cannot begin comes with the launches.
    pub fn begin<L>(
        &self,
        launches: Chain<L>,
        route: impl Fn(&L) -> StepRoute,
        work: impl Fn(&L) -> crate::operation::LaunchWork,
        scratch: impl FnMut() -> Result<bool, String>,
    ) -> Result<(Chain<(L, ExpertStep)>, Vec<(WeightScope, usize)>), (String, Chain<L>)> {
        let (steps, merges) = self.watcher.shared().begin(launches, route, work, scratch)?;
        Ok((steps, self.plan.layers.iter().map(|layer| layer.scope).zip(merges).collect()))
    }

    /// The steps a prompt group runs while decode waits, at most `capacity`
    /// (`Shared::group_size`).
    pub fn group_size(&self, capacity: usize) -> usize {
        self.watcher.shared().group_size(capacity)
    }

    /// Whether the group `begin` is beginning is resident, which its
    /// resident word then holds for its fronts (`BankPart::Resident`): every
    /// streamed layer's group is whole. Residency publishes the
    /// policy words only while no step runs (`Residency::publish_policy`), under
    /// the lock `begin` reads them in, and only a shrink clears one, which
    /// the owner asks for between flights and waits on: what this reads
    /// holds until the group ends.
    pub fn resident(&self) -> Result<bool, String> {
        let word = |tensor: &Tensor| {
            tensor
                .load_acquire(4 * magnitude_kernels::constants::POLICY_WHOLE)
                .map_err(|error| format!("a policy's whole word: {error}"))
        };
        let mut resident = true;
        for words in &self.layout.layers {
            resident &= word(&words.policy)? != 0;
        }
        self.layout
            .load
            .resident
            .store_release(0, u32::from(resident))
            .map_err(|error| format!("the resident word: {error}"))?;
        Ok(resident)
    }

    /// Step `sequence`'s runs serve the blocks from `block` on over a window
    /// of its rows starting at batch row `first`: the streamed layers there
    /// name their requests' rows within it.
    pub(crate) fn window(&self, sequence: u64, block: usize, first: u64) -> Result<(), String> {
        let from = self.plan.layers.iter().position(|layer| layer.sublayer().block as usize >= block);
        if let Some(layer) = from {
            let first = usize::try_from(first).map_err(|_| "a window's first row exceeds the host")?;
            self.watcher.shared().window(sequence, layer, first);
        }
        Ok(())
    }

    /// The streamed layer of `scope`: its index and its group.
    fn layer(&self, scope: WeightScope) -> Result<(usize, usize), String> {
        self.plan
            .layer(scope)
            .map(|(index, layer)| (index, layer.group))
            .ok_or_else(|| format!("{scope:?} is not a streamed routed layer"))
    }

    /// Whether `role` is a banked expert stack.
    pub fn holds(&self, role: WeightRole) -> bool {
        RoutedTensor::of(role.kind).is_some() && self.plan.layer(role.scope).is_some()
    }

    /// The group tensor a banked expert stack of `role` is bound as.
    pub fn records(&self, role: WeightRole) -> Result<&Tensor, String> {
        let kind = RoutedTensor::of(role.kind).ok_or_else(|| format!("{role:?} is no routed tensor of a record"))?;
        self.layout.groups[self.layer(role.scope)?.1]
            .records
            .get(kind)
            .ok_or_else(|| format!("{role:?} is no routed tensor of its layer's record"))
    }

    /// The per-expert second-level scale of a banked expert stack of
    /// `role`.
    pub fn expert_scale(&self, role: WeightRole) -> Result<&Tensor, String> {
        let kind = RoutedTensor::of(role.kind).ok_or_else(|| format!("{role:?} is no routed tensor of a record"))?;
        self.layout.layers[self.layer(role.scope)?.0]
            .expert_scales
            .get(kind)
            .and_then(Option::as_ref)
            .ok_or_else(|| format!("{role:?} has no second-level scale"))
    }

    /// The instance of the group reads (`BankPart::GroupReads`) of record
    /// `record` at chain place `place`, of the places the bank holds now.
    pub fn group_reads_at(&self, record: usize, place: usize) -> usize {
        record * self.placed_now(|placed| placed.places) as usize + place
    }

    /// What `read` reads of the placed parts' words every step binds now: a
    /// group's places' while one holds them, the floor's otherwise.
    fn placed_now<R>(&self, read: impl FnOnce(&PlacedAllocations<Tensor>) -> R) -> R {
        read(self.held.lock().expect("held places").as_ref().unwrap_or(&self.floor))
    }

    /// The tensor a run at instance `place` binds as placed `part`
    /// (`BankPart::placed`) of the streamed layer of `scope`: a chain
    /// place's, or a record's place's (`Self::group_reads_at`), words of the
    /// part's allocation of the places held now.
    pub fn placed(&self, scope: WeightScope, part: BankPart, place: usize) -> Result<Tensor, String> {
        let index = self.layer(scope)?.0;
        self.placed_now(|placed| {
            // A place's step words are their own allocation, its first
            // instance; the other parts' places lie end to end.
            let (tensor, instance) = match part {
                BankPart::Step => (
                    placed.step.get(place).ok_or_else(|| format!("chain place {place} of {} held", placed.step.len()))?,
                    0,
                ),
                BankPart::ScratchFlags => (&placed.flags, place),
                BankPart::GroupReads => (&placed.layers[index].group_reads, place),
                _ => return Err(format!("bank part {part:?} is bound whole")),
            };
            place_words(&self.plan, index, part, tensor, instance as u64)
        })
    }

    /// The streamed entries' parks, which each run binds and registers with
    /// (`super::Parks`).
    pub(crate) fn parks(&self) -> &super::Parks {
        &self.watcher.shared().parks
    }

    /// Step `sequence`'s run of streamed layer `layer`'s block was sampled
    /// (`spans`, `marks` at its ends), whose device time its completion
    /// observes into the layer's block line.
    pub(crate) fn block_sampled(&self, sequence: u64, layer: usize, spans: seismic::NativeSpans, marks: (usize, usize)) {
        self.watcher.shared().block_sampled(sequence, layer, spans, marks);
    }

    /// The bank's index of the streamed routed layer of `scope`.
    pub(crate) fn streamed_layer(&self, scope: WeightScope) -> Result<usize, String> {
        self.layer(scope).map(|(index, _)| index)
    }

    /// The tensor `part` of the streamed layer of `scope` binds, of a part
    /// bound whole; a placed part is bound at its place (`Self::placed`).
    pub fn part(&self, scope: WeightScope, part: BankPart) -> Result<&Tensor, String> {
        let (index, group_index) = self.layer(scope)?;
        let (layer, group, load) = (&self.layout.layers[index], &self.layout.groups[group_index], &self.layout.load);
        Ok(match part {
            BankPart::Step | BankPart::GroupReads | BankPart::ScratchFlags => {
                return Err(format!("bank part {part:?} is bound at a chain place"));
            }
            BankPart::Resident => &load.resident,
            BankPart::SlotMap => &self.slot_maps[index],
            BankPart::Map => &group.map,
            BankPart::TierMap => layer.tier.as_ref().ok_or_else(|| format!("bank part {part:?} of a bank without a host tier"))?,
            BankPart::Tier { kind } => self
                .layout
                .host
                .as_ref()
                .and_then(|tiers| tiers[group_index].get(kind))
                .ok_or_else(|| format!("bank part {part:?} of a bank without a host tier holding it"))?,
            BankPart::FileTable => &layer.file,
            BankPart::ScratchTable => &layer.scratch,
            BankPart::Rings => &layer.rings,
            BankPart::Routing => &layer.routing,
            BankPart::Policy => &layer.policy,
            BankPart::SiteTable | BankPart::SiteRows | BankPart::SiteOutput => {
                let sited = layer.sited.as_ref().ok_or_else(|| format!("bank part {part:?} of a layer without a CPU site"))?;
                match part {
                    BankPart::SiteTable => &sited.table,
                    BankPart::SiteRows => &sited.rows,
                    _ => &sited.output,
                }
            }
            BankPart::Commits => &group.commits,
            BankPart::Conveyor { kind } => {
                let pool = self.pool_of[index].get(kind).ok_or_else(|| format!("bank part {part:?} of a record without it"))?;
                &self.conveyors[*pool].conveyor
            }
            BankPart::Scratch { kind } => self.scratch[index].get(kind).ok_or_else(|| format!("bank part {part:?} of a record without it"))?,
            BankPart::Staging { kind } => {
                let pool = self.pool_of[index].get(kind).ok_or_else(|| format!("bank part {part:?} of a record without it"))?;
                &self.layout.staging[*pool]
            }
            BankPart::LandingList | BankPart::LandingGrids => {
                match part {
                    BankPart::LandingList => &layer.landing.list,
                    _ => layer.landing.grids.as_ref().ok_or_else(|| format!("bank part {part:?} of a bank the device places"))?,
                }
            }
            BankPart::ConvertList { round } => load
                .lists
                .get(usize::from(round))
                .ok_or_else(|| format!("bank part {part:?} names no convert round"))?,
            BankPart::ConvertStaged => &load.convert_staged,
            BankPart::TierList => &load.tiered,
            BankPart::TierGrid => &load.tier_grid,
            BankPart::TierStaged => &load.tier_staged,
            BankPart::SeedList => &load.seeds,
            BankPart::SeedGrid => &load.seed_grid,
            BankPart::SeedStaged => &load.seed_staged,
            BankPart::ScratchGrids => &load.grids,
        })
    }

    /// Visit every allocation the bank holds: its layout's, the floor's
    /// placed words and a group's more places'.
    pub fn visit_storage(&self, visit: &mut dyn FnMut(&Tensor)) {
        self.layout.iter().chain(self.floor.iter()).for_each(&mut *visit);
        if let Some(held) = self.held.lock().expect("held places").as_ref() {
            held.iter().for_each(visit);
        }
    }

    /// The storage a group of `places` chain places holds of the bank beside
    /// the floor's (`Self::hold_places`): none of one place, which the
    /// floor's words serve.
    pub fn placed_demand(&self, places: std::num::NonZeroU64) -> Result<crate::memory::ClaimDemand, String> {
        match places.get() {
            1 => Ok(Default::default()),
            _ => self.plan.placed_demand(places),
        }
    }

    /// The placed parts' words of a group of `places` chain places, held
    /// once installed (`Self::hold_places`): none of one place, which the
    /// floor's serve. Why the device could not hold them, which leaves
    /// what is held now.
    pub fn place(&self, places: std::num::NonZeroU64) -> Result<Places, crate::resources::AllocationError> {
        Ok(Places(match places.get() {
            1 => None,
            _ => Some(allocate_placed(&self.device, &self.plan, places)?),
        }))
    }

    /// Hold `places` (`Self::place`) for every step from now, between
    /// flights: the watcher serves each place's scratch requests, and the
    /// telemetry reads each record's group reads, from their words.
    pub fn hold_places(&self, places: Places) -> Result<(), String> {
        let words = placed_words(&self.plan, places.0.as_ref().unwrap_or(&self.floor))?;
        self.watcher.shared().place(words)?;
        *self.held.lock().expect("held places") = places.0;
        Ok(())
    }

    /// Every byte the bank is charged: its reserved records' backed
    /// granules, and every other allocation's bytes. Off the domain's
    /// thread its charge rises only under [`Self::hold_rises`], which the
    /// domain holds while it observes Seismic's charge and classifies the
    /// bank against it, so one reading and its classification are of one
    /// state; a release off the thread only ever lowers it.
    pub fn storage_bytes(&self) -> Result<u64, &'static str> {
        let mut bytes = Some(0u64);
        self.visit_storage(&mut |tensor| {
            let charged = tensor.observe_storage().charged_bytes().expect("a held allocation is charged");
            bytes = bytes.and_then(|bytes| bytes.checked_add(charged));
        });
        bytes.ok_or("the expert bank's charge overflows")
    }

    /// Hold every rise of the bank's charge off the domain's thread (a
    /// growth's installs on the resize thread, [`grow_by_slots`]) until the
    /// guard drops: what the domain holds across one observation of
    /// Seismic's charge and the classification read against it.
    pub fn hold_rises(&self) -> std::sync::MutexGuard<'_, ()> {
        self.rises.lock().expect("the bank's charge rises")
    }

    /// What streamed layer `layer`'s CPU site answered of step `step`
    /// (`Compute::answered`).
    #[cfg(test)]
    pub(crate) fn site_answered(&self, layer: usize, step: u64) -> Option<crate::experts::Answered> {
        self.watcher.shared().site_answered(layer, step)
    }

    /// What bank-served and scratch-served steps cost each streamed layer
    /// so far, in decoder order, and the last step completed.
    pub fn costs(&self) -> ExpertCosts {
        self.watcher.shared().costs()
    }

    /// The experts each streamed layer holds, ascending, as its slot map
    /// names them.
    pub fn held(&self) -> Vec<Vec<usize>> {
        self.watcher.shared().held()
    }
}

/// The record tensors the host lands file misses into, group by group: a
/// host tier's where the device places, the bank's own where the host does
/// (the one store); none where it lands into neither.
fn landed_tensors(plan: &ExpertBankPlan, layout: &BankLayout<Tensor>) -> Option<Vec<RecordTensors<Tensor>>> {
    match plan.placer() {
        Placer::Tier => layout.host.clone(),
        Placer::Bank => Some(layout.groups.iter().map(|group| group.records.clone()).collect()),
    }
}

/// The records the host lands into of group `group` at `slots` slots: a
/// host tier's host slots where the device places, the one store's slots
/// where the host does.
fn landed_records(plan: &ExpertBankPlan, group: usize, slots: u64) -> u64 {
    match plan.placer() {
        Placer::Tier => plan.host_records(group, slots),
        Placer::Bank => plan.records_at(group, slots),
    }
}

/// Each group's landed `tensors` ([`landed_tensors`]) at its `committed`
/// records, as the host addresses them: what landings convert into and a
/// resize's moves copy within. A view pins its records backed until it
/// drops (`Tensor::region`), so the host addresses no record a recommit
/// could release.
fn host_views(
    plan: &ExpertBankPlan,
    tensors: &[RecordTensors<Tensor>],
    committed: impl Fn(usize) -> u64,
) -> Result<Vec<Vec<Records>>, String> {
    tensors
        .iter()
        .enumerate()
        .map(|(group, tensors)| {
            let committed = committed(group);
            tensors
                .iter()
                .zip(plan.groups[group].record.iter())
                .map(|((_, tensor), (_, record))| {
                    let view = tensor
                        .region(record.resident, &[committed, record.matrix[0], record.matrix[1]], 0)
                        .map_err(|error| format!("the {} host records: {error}", record.resident.name()))?;
                    Records::new(view, record.record_bytes()?, committed)
                        .map_err(|error| format!("the {} host records: {error}", record.resident.name()))
                })
                .collect()
        })
        .collect()
}

/// Instance `place` of placed part `part` of `layer` of its allocation
/// `tensor`: its planned words, from the instance's start at the part's
/// stride (`ExpertBankPlan::place_stride`, whole u64s). The one slicing of a
/// placed part, the bank's binding and the watcher's.
fn place_words(plan: &ExpertBankPlan, layer: usize, part: BankPart, tensor: &Tensor, place: u64) -> Result<Tensor, String> {
    let words = plan.part_shape(layer, part)?.1[0];
    let start = plan.place_stride(layer, part)? * place;
    tensor
        .slice_leading(start, start + words)
        .map_err(|error| format!("chain place {place} of bank part {part:?}: {error}"))
}

/// The placed parts' words of a group's places (`ExpertBank::place`), the
/// floor's where none: what `ExpertBank::hold_places` holds.
pub(crate) struct Places(Option<PlacedAllocations<Tensor>>);

/// The placed parts' words of `places` chain places on `device`, zeroed,
/// host-shared (`ExpertBankPlan::placed`).
fn allocate_placed(
    device: &Device,
    plan: &ExpertBankPlan,
    places: std::num::NonZeroU64,
) -> Result<PlacedAllocations<Tensor>, crate::resources::AllocationError> {
    let planned = plan.placed(places).map_err(|detail| {
        crate::resources::AllocationError::Plan(crate::error::InvariantError { context: "expert bank places", detail })
    })?;
    planned.try_map(|(element, extents)| {
        Tensor::words(device, *element, extents).map_err(crate::resources::AllocationError::Tensor)
    })
}

/// The watcher's words of `placed`: each place's scratch flags
/// (`place_words`), and each streamed layer's group reads at their stride,
/// every word zero.
fn placed_words(plan: &ExpertBankPlan, placed: &PlacedAllocations<Tensor>) -> Result<PlacedWords, String> {
    let words = |tensor: &Tensor| Words::new(tensor.clone(), tensor.byte_len() / 4, 0);
    let each = |layer: usize, part: BankPart, tensor: &Tensor| {
        (0..placed.places)
            .map(|place| words(&place_words(plan, layer, part, tensor, place)?))
            .collect::<Result<Vec<_>, String>>()
    };
    Ok(PlacedWords {
        // The load's flags, of one shape at every layer.
        flags: each(0, BankPart::ScratchFlags, &placed.flags)?,
        layers: placed
            .layers
            .iter()
            .enumerate()
            .map(|(layer, words_of)| {
                Ok(PlacedLayerWords {
                    group_reads: words(&words_of.group_reads)?,
                    stride: usize::try_from(plan.place_stride(layer, BankPart::GroupReads)?)
                        .map_err(|_| "group reads exceed the host")?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?,
    })
}

/// Each layer's slot map, its view of its group's map: its experts from the
/// map word of its expert 0 (`ExpertBankPlan::home`).
fn slot_maps(plan: &ExpertBankPlan, layout: &BankLayout<Tensor>) -> Result<Vec<Tensor>, String> {
    plan.layers
        .iter()
        .enumerate()
        .map(|(index, layer)| {
            let home = plan.home(index);
            layout.groups[layer.group]
                .map
                .slice_leading(home, home + layer.experts)
                .map_err(|error| format!("a layer's slot map view failed: {error}"))
        })
        .collect()
}

#[cfg(test)]
mod tests;
