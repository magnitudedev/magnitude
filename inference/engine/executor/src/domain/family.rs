//! The worker binds one program family before constructing its numerical domain.
//! Every lane keeps its concrete submission type through completion and finish.

use crate::programs::{
    Chain, CompletedHeadWork, CompletedStateWork, CompletedTargetWork, CompletedVisionWork, HeadProgram,
    ProgramSubmission, StateProgram, SubmittedHead, SubmittedTarget, TargetProgram, VisionProgram,
};
use crate::{
    AttestedPrograms, ResidentHead, ResidentTarget, ResidentVision, SubmitError,
    ValidatedHeadLaunch, ValidatedStateLaunch, ValidatedTargetLaunch, ValidatedVisionLaunch,
};
use magnitude_family_contracts::{Decoder, ModelDefinition};
use std::rc::Rc;

/// A hold of an expert bank's own traffic (`ElasticBank::hold_traffic`):
/// it stands while it lives and releases as it drops, by whichever exit,
/// so a holder that panics leaves no hold behind. Holds are counted: each
/// holder releases only its own.
#[must_use = "a hold stands only while it lives"]
pub struct TrafficHold(Option<Box<dyn FnOnce() + Send>>);

impl TrafficHold {
    /// A hold `release` ends.
    pub(crate) fn new(release: impl FnOnce() + Send + 'static) -> Self {
        Self(Some(Box::new(release)))
    }
}

impl Drop for TrafficHold {
    fn drop(&mut self) {
        if let Some(release) = self.0.take() {
            release();
        }
    }
}

/// A load's elastic expert bank: the one owner of the memory its
/// experts and the stored tables it holds take, each holding sized by its
/// derived target (`ExecutorDomain::follow_cache_targets`). It shrinks toward its floor and
/// grows toward every expert one slot at a time, and each table a record at a
/// time. An expert resize's storage is prepared off the submitting thread,
/// installed at a later flight boundary and its surplus released after the
/// flight that last reached it ([`Self::advance`]), so no flight waits on
/// it; a table's resize is the host's directory change and its records'
/// recommit, done when asked and reported by the next advance. One resize
/// runs at a time.
pub trait ElasticBank {
    /// The slots each layer commits now: what its directory serves from.
    fn slots(&self) -> u64;

    /// Starve the bank's host for `duration` from now: its pulse does not
    /// beat and no answer is written meanwhile, so a step's waits that spin
    /// then end before their answers (a test's). Whether it has a host to
    /// starve: a bank with no waits has none.
    #[cfg(test)]
    fn starve(&self, _duration: std::time::Duration) -> bool {
        false
    }

    /// Hold every rise of the bank's charge off the domain's thread until
    /// the guard drops: the domain holds it across one observation of
    /// memory (`ExecutorDomain::observe_memory`), which then is of one state
    /// (a release off the thread only ever lowers the charge).
    fn hold_rises(&self) -> std::sync::MutexGuard<'_, ()>;

    /// Each streamed layer's host lines as the bank's load measured them
    /// (its CPU pool's and landing lines): what the load's misses are priced
    /// along, with the GPU's.
    fn lines(&self) -> std::sync::Arc<[crate::assessment::BankLines]>;

    /// Each streamed layer's lines from now on, the GPU's priced, or none
    /// where the executor's costs do not price them: what the route's
    /// policy words are written from (`ExpertLines::policy`).
    fn price(&self, lines: Option<std::sync::Arc<[crate::assessment::ExpertLines]>>, pass: Option<crate::assessment::ExpertPass>);
    /// A tier copy costs `costs` from now on: the blocks it takes, and
    /// the price a promotion's copy is weighed at.
    fn tier_copy(&self, costs: &crate::assessment::TierCopyCosts) -> Result<(), super::DomainError>;
    /// End its growths (the experts', the host tier's and its backing at
    /// open) at their next slot: a measurement's claim beside the load
    /// stops them, as its hold would leave their slots waiting on the
    /// resize thread, and every shrink the claim asks behind them.
    fn stop_growths(&self);
    /// Hold the bank's own traffic (its promotions' copies, its reads
    /// ahead and their landings) until the returned hold drops: what a
    /// measurement beside the load holds.
    fn hold_traffic(&self) -> TrafficHold;
    /// Whether its traffic is held and none of it still runs.
    fn traffic_quiet(&self) -> bool;
    /// Wait until its traffic, held, is quiet; why serving ended, if it
    /// ends first, or that the load stops.
    fn wait_traffic_quiet(&self) -> Result<(), super::DomainError>;
    /// Each bank group's convert in a decode barrier takes its `lines` from
    /// now on.
    fn device_convert(&self, lines: &[crate::assessment::RecordLine]);
    /// Measure the host's landings slow from now on
    /// (`Shared::slow_host_landing`).
    #[cfg(test)]
    fn slow_host_landing(&self);
    /// The landings the device has converted.
    #[cfg(test)]
    fn device_landed(&self) -> u64;
    /// The decode steps begun while the bank was whole, which ran the
    /// resident forms (`ExpertBank::begin`).
    #[cfg(test)]
    fn resident_decodes(&self) -> u64;

    /// Begin a shrink of `holding` toward `to` (its slots, host slots or
    /// table records), its derived target
    /// (`ExecutorDomain::follow_cache_targets`), at a flight
    /// boundary: one
    /// resize. Unless a resize of its store runs or it is at `to`,
    /// `between_flights` first brings the device past every begun step (no
    /// step may run while the watcher plans; one still running defers the
    /// shrink, [`BankStep::Deferred`]), then the shrink's evictions are
    /// published, its moves carried on the device in a submission of its
    /// own, its plan committed without waiting for the moves (the device
    /// runs every later flight after them), and its smaller storage prepared
    /// off this thread. `wake` is completed once the preparation ends. One
    /// resize reaches `to`, unless a cut of the records the host lands into
    /// falls below a record in flight, which stays in place: the cut stops
    /// above it, and a later follow takes the rest.
    fn shrink(
        &self,
        holding: BankHolding,
        to: u64,
        between_flights: &mut dyn FnMut() -> Result<(), super::DomainError>,
        wake: crate::CompletionWake,
    ) -> Result<BankStep, super::DomainError>;

    /// Once a flight completed: the device failure of a shrink's moves
    /// submitted before it, which it read, so it fails with them; nothing
    /// read from a record the moves did not write is ever taken.
    fn carried(&self) -> Result<(), super::DomainError>;

    /// Each holding as its derived target reads it
    /// (`experts::derived::HoldingTarget`), in the order its store's room
    /// is shared: between its floor and the lesser of its ceiling and where
    /// its misses stop falling on `price` (its ceiling before the load is
    /// priced). The domain derives their targets from its ledgers
    /// (`experts::derived::derive`).
    fn targets(&self, price: Option<&crate::assessment::DecodePrice>) -> Result<Vec<crate::experts::derived::HoldingTarget<'_>>, super::DomainError>;

    /// The experts' slots a shrink releasing `bytes` reaches: the shallowest
    /// size that does, down to the floor; none at the floor. A test's way
    /// to ask a shrink of a size.
    #[cfg(test)]
    fn released_to(&self, bytes: u64) -> Result<Option<u64>, super::DomainError>;

    /// What the load's counted decode read and ran so far, which its decode
    /// is priced on; none before it counted a step.
    fn live(&self) -> Option<crate::assessment::LiveDecode>;

    /// The bank's streamed layers, each priced by its `lines`, which its
    /// decode is priced on.
    fn streamed_layers(&self, lines: std::sync::Arc<[crate::assessment::ExpertLines]>) -> crate::assessment::StreamedLayers;

    /// Begin one growth of `holding` to `to`, its derived target
    /// (`ExecutorDomain::follow_cache_targets`), at a flight boundary, unless a
    /// resize runs, its cache taken from its store's ledger in `ledgers` first (none there, nothing
    /// begins: [`BankStep::Deferred`]): expert slots' and host slots' storage is backed off this thread
    /// a slot at a time, each slot joining the bank as soon as it is backed,
    /// beside running steps, `wake` completed once the growth ends; table
    /// records' are backed now.
    fn grow(&self, ledgers: &crate::memory::ledger::DeviceLedgers, holding: BankHolding, to: u64, wake: crate::CompletionWake) -> Result<BankStep, super::DomainError>;

    /// The storage `holding` lies in, which its resizes are of: each store
    /// resizes on its own ([`BankStep::Begun`]).
    fn holding_storage(&self, holding: BankHolding) -> Result<crate::planning::BankTier, super::DomainError>;

    /// At a flight boundary: install a running shrink's prepared storage and
    /// begin its release off this thread, `wake` completed once that ends;
    /// or take back the released shrink; or take a growth its resize thread
    /// ended. The resize finished here, if one did.
    fn advance(&self, wake: crate::CompletionWake) -> Result<Option<BankResized>, super::DomainError>;

    /// The cut of the expert bank a chain's encoding began beside its
    /// running steps since this was last asked (`ExpertBank::release_running`),
    /// taken: a resize begun off the domain's decisions, which a later
    /// advance finishes as any shrink.
    fn running_begun(&self) -> Option<RunningBegun>;

    /// Whether no step runs: a gap between flights, where a shrink may ask
    /// the watcher. Steps begin and end on the owner's thread, so a gap it
    /// sees holds until it begins one.
    fn in_gap(&self) -> bool;

    /// Where the load's routed experts are now ([`ExpertResidency`]), each
    /// holding with its last finished resize ([`Self::record_resize`]).
    fn residency(&self) -> Result<ExpertResidency, super::DomainError>;

    /// Record `resize`, a resize of `holding` its advance reported finished,
    /// as that holding's last: kept with the holding, which the bank owns.
    fn record_resize(&self, holding: BankHolding, resize: BankResize);

    /// Wait until the store misses land into is filled
    /// ([`TierFill::Filled`]), as its fill publishes it: the host tier
    /// where the device places, the one store where the host does.
    fn wait_tier_filled(&self) -> Result<(), super::DomainError>;

    /// How many times the bank's fill started or finished
    /// (`Shared::fill_epoch`): odd once it finished, even while it lands
    /// experts beside the steps, which then miss and read what the fill
    /// leaves, not what the bank settles to.
    fn fill_epoch(&self) -> u64;
}

/// Where a load's routed experts are now: the slots its directory serves
/// from, between its floor and ceiling, each a record of every streamed
/// layer; Seismic's charge of those records in each store; and what the
/// slots it does not commit leave on disk.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ExpertResidency {
    /// `E`: the widest routed layer's experts, which `slots` reach to hold
    /// every expert.
    pub experts: std::num::NonZeroU64,
    pub slots: u64,
    pub floor: u64,
    pub ceiling: u64,
    /// The records' charge in each store holding them, host RAM last.
    pub stores: Vec<(seismic::StoreIdentity, u64)>,
    /// The host tier, where the load has one (a discrete device's,
    /// `ExpertBankPlan::host`); none on unified memory.
    pub host: Option<HostResidency>,
    /// The fill of the store misses land into: the host tier where the
    /// device places, the one store where the host does.
    pub fill: TierFill,
    /// The experts of every layer in no slot and not in the host tier, as
    /// the model file stores them.
    pub disk_bytes: u64,
    /// The stored tables the bank holds beside its experts.
    pub tables: Vec<TableResidency>,
    /// The experts' slots' last finished resize; none before one.
    pub last_resize: Option<BankResize>,
}

/// Where a stored table the bank holds is now: the records its slots use,
/// between its floor and ceiling, and Seismic's charge of them.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TableResidency {
    pub records: u64,
    pub floor: u64,
    pub ceiling: std::num::NonZeroU64,
    pub bytes: u64,
    /// The table's records' last finished resize; none before one.
    pub last_resize: Option<BankResize>,
}

/// Where a host tier is now: its host slots (each a record of every
/// streamed layer beside the staging floor), up to its ceiling, and
/// Seismic's charge of its records in host RAM.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct HostResidency {
    pub slots: u64,
    pub ceiling: u64,
    pub bytes: u64,
    /// The host slots' last finished resize; none before one.
    pub last_resize: Option<BankResize>,
}

/// A finished resize of one of the expert bank's holdings, in its units (the
/// experts' slots, a table's records, the host tier's host slots), at
/// `at_unix_ms` (milliseconds since the Unix epoch): shrunk by `by` to `to`
/// because its derived target fell below what it held
/// (`ExecutorDomain::follow_cache_targets`), or the guarantee was short beside
/// a running chain, or grown by `by` from `from` because memory stayed free
/// and the step paid. Each changes its size by at least one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum BankResize {
    Shrank { to: u64, by: std::num::NonZeroU64, at_unix_ms: u64 },
    Grew { from: u64, by: std::num::NonZeroU64, at_unix_ms: u64 },
}

/// What a chain's encoding finds memory short of between its submissions
/// (`ExecutorDomain::submit_chain`'s probe): the bytes free room in the
/// expert bank's store lacks of the speed guarantee now
/// (`memory::ledger::Snapshot::short`), and what the process held as it
/// read them.
#[derive(Clone, Copy, Debug)]
pub struct RunningPressure {
    pub bytes: u64,
    pub held: super::MemoryMoment,
}

/// A cut of the expert bank begun beside a running chain
/// ([`ElasticBank::running_begun`]): when it was decided, how long beginning
/// it held the submitting thread, and what the process held as it was
/// decided.
#[derive(Clone, Copy, Debug)]
pub struct RunningBegun {
    pub decided: std::time::Instant,
    pub held: std::time::Duration,
    pub moment: super::MemoryMoment,
}

/// A component a load binds beside its target that serving may release
/// while no request is resident, and loads again at its next use.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum OptionalComponent {
    /// The head that drafts.
    Drafter,
    /// The projector that encodes images.
    Vision,
}

impl OptionalComponent {
    pub const ALL: [Self; 2] = [Self::Drafter, Self::Vision];
}

/// Whether the fill of the store misses land into (the host tier where
/// the device places, the one store where the host does) has finished:
/// the fill lands experts into every record beyond the landing reserve,
/// from the bank's open and again after a growth. Filled once each such
/// record keeps an expert (or the store keeps every expert) and no fill's
/// landing is in flight; from then the store's members change only by its
/// misses' landings displacing what it keeps, at the same records.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum TierFill {
    Filling,
    Filled,
}

/// What asking an elastic bank for a resize came to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BankStep {
    /// The resize runs in the holdings of this storage; [`ElasticBank::advance`]
    /// finishes it. Each store resizes on its own, so a resize of one never
    /// waits on the other's.
    Begun(crate::planning::BankTier),
    /// Another resize of the store this one is of runs: this one waits for it.
    Running,
    /// No gap to resize in now: a chain under way holds the queue (a
    /// shrink, which would orphan it), a step still runs once the queue is
    /// drained, so there is no gap to ask the watcher in, or every cut of
    /// the records the host lands into falls below a record in flight,
    /// which it keeps in place: nothing began, and the next boundary
    /// decides again.
    Deferred,
    /// The bank is at its floor (a shrink) or holds every expert (a growth).
    AtLimit,
}

/// A finished resize of one holding: its size before it (`from`) and after
/// it (`to`), in its units (the experts' slots, a table's records), and
/// Seismic's measured change of its charge, or why the storage refused it.
#[derive(Debug)]
pub enum BankResized {
    Shrunk { holding: BankHolding, from: u64, to: u64, released: u64, times: ResizeTimes },
    /// Grown by at least one joined slot, to the slots its joined slots
    /// reached (`refused` the storage's refusal of the next slot, where one
    /// cut it short of its target).
    Grown { holding: BankHolding, from: u64, to: u64, grown: u64, refused: Option<seismic::TensorError>, times: ResizeTimes },
    /// A growth stopped before its first slot joined (a shrink wanted its
    /// memory, or the bank dropped): the holding serves `at`, as before it.
    Stopped { holding: BankHolding, at: u64 },
    /// Seismic refused a growth's storage before its first slot (the
    /// device's memory): the holding keeps its storage and serves `at`. Only
    /// a growth is refused: a shrink needs no memory, and its storage's
    /// refusal breaks the bank instead.
    GrowthRefused { holding: BankHolding, at: u64, refusal: seismic::TensorError },
}

impl BankResized {
    /// The holding the resize was of.
    pub fn holding(&self) -> BankHolding {
        match *self {
            Self::Shrunk { holding, .. }
            | Self::Grown { holding, .. }
            | Self::Stopped { holding, .. }
            | Self::GrowthRefused { holding, .. } => holding,
        }
    }
}

/// What an elastic bank's owner resizes: its experts' slots, the records
/// of the stored table `ExpertResidency::tables[table]` describes, or the
/// host tier's host slots; the storage each lies in is its plan's
/// (`ExpertBankPlan::holding_storage`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum BankHolding {
    Experts,
    Table { table: usize },
    Host,
}

/// When a finished resize's steps ran: what a decode stall around it is
/// attributed to, and the latency from its decision to its effect.
#[derive(Clone, Copy, Debug)]
pub struct ResizeTimes {
    /// The bank was asked for the resize.
    pub decided: std::time::Instant,
    /// A shrink's wait for the steps queued on the device, its watcher's
    /// plan and commit, and its moves' submission (and any wait for them),
    /// all on the submitting thread; zero for a growth.
    pub queued: std::time::Duration,
    pub planned: std::time::Duration,
    /// Of `planned`, each of a shrink's two asks of the watcher (its plan,
    /// then its commit): the time until the watcher took it up, and the
    /// watcher's serving of it.
    pub trips: [Trip; 2],
    pub moves: std::time::Duration,
    /// A shrink's storage was prepared off the submitting thread; a
    /// growth's first slot joined.
    pub prepared: std::time::Instant,
    /// A shrink's prepared storage was installed between flights; a
    /// growth's last slot joined.
    pub installed: std::time::Instant,
    /// The surplus storage was released: the resize took effect.
    pub released: std::time::Instant,
    /// Of a host tier's or table's shrink, its release's unregistration of
    /// host pages ([`seismic::BackingTimes::unpin`]: CUDA's, which waits for
    /// the device's work); zero for a device bank's resize and a growth.
    pub unpinned: std::time::Duration,
}

/// An ask's trip through the expert watcher, as the submitting thread waits
/// it out: until the watcher took it up, and its serving.
#[derive(Clone, Copy, Debug, Default)]
pub struct Trip {
    pub picked: std::time::Duration,
    pub served: std::time::Duration,
}

pub trait ProgramFamily: 'static {
    type TargetSubmission: ProgramSubmission<CompletedWork = CompletedTargetWork>
        + SubmittedTarget
        + 'static;
    type HeadSubmission: ProgramSubmission<CompletedWork = CompletedHeadWork>
        + SubmittedHead
        + 'static;
    type VisionSubmission: ProgramSubmission<CompletedWork = CompletedVisionWork> + 'static;
    type StateSubmission: ProgramSubmission<CompletedWork = CompletedStateWork> + 'static;

    fn bind_head(
        &mut self,
        resident: ResidentHead,
        definition: &ModelDefinition,
    ) -> Result<(), String>;
    fn bind_vision(
        &mut self,
        resident: ResidentVision,
        definition: &ModelDefinition,
    ) -> Result<(), String>;
    fn head_is_bound(&self) -> bool;
    fn vision_is_bound(&self) -> bool;
    fn state_is_bound(&self) -> bool;
    /// Unbind `component`: released while idle, or its class's storage
    /// was refused.
    fn unbind_optional(&mut self, component: OptionalComponent);
    /// What the load's graphs cost serving so far, by family: graphs
    /// recorded and replays that updated a graph's parameters, of its own
    /// plans alone beside other loads on the device
    /// (`seismic::Device::graph_replays`).
    fn family_graph_replays(&self) -> Vec<(&'static str, seismic::GraphReplays)>;

    /// Whether the target's decode graphs are recorded ahead of their first
    /// use (CUDA's only), so that a bank shrink's step records none:
    /// elsewhere a graph is recorded at its shape's first use, that step's
    /// included. A family with no native graphs records none.
    fn records_decode_ahead(&self) -> bool {
        false
    }

    /// What the load's graphs cost serving so far, of every family
    /// ([`Self::family_graph_replays`]).
    fn graph_replays(&self) -> seismic::GraphReplays {
        self.family_graph_replays().into_iter().fold(seismic::GraphReplays::default(), |counts, (_, family)| counts.plus(family))
    }

    /// Physical target tensors held by the bound family. Test families with
    /// no resident model have no target charge.
    fn target_weight_bytes(&self) -> Result<u64, &'static str> {
        Ok(0)
    }

    /// Physical allocations classified as model storage, including constants.
    fn visit_model_storage(&self, _visit: &mut dyn FnMut(&seismic::Tensor)) {}

    fn prepared_program_bytes(&self) -> u64 {
        0
    }

    fn target_constant_bytes(&self) -> Result<u64, &'static str> {
        Ok(0)
    }

    fn optional_constant_bytes(&self) -> Result<u64, &'static str> {
        Ok(0)
    }

    /// The storage the buffers of a prompt group of `places` places demand
    /// beside the floor's (`hold_group`).
    fn group_demand(&self, _places: usize) -> Result<crate::memory::ClaimDemand, String> {
        Ok(Default::default())
    }

    /// Hold a prompt group of `places` places' buffers in place of any held
    /// now; the floor's, of one place, are always held. The caller holds a
    /// heap claim for `group_demand`; a failure keeps the buffers held
    /// before. Only in a gap between flights (`quiet`): the expert bank
    /// takes a group's places only while none of its steps runs.
    fn hold_group(&self, _places: usize, _quiet: super::Quiet) -> Result<(), crate::resources::AllocationError> {
        Ok(())
    }

    /// The storage a head bound while a class of `places` places is held
    /// demands for that class beside its floor's: its own streamed tables'
    /// stagings (`hold_bound_head`).
    fn bound_head_demand(&self, _places: usize) -> Result<crate::memory::ClaimDemand, String> {
        Ok(Default::default())
    }

    /// Hold the bound head's storage for a class of `places` places, under a
    /// heap claim for `bound_head_demand`.
    fn hold_bound_head(&self, _places: usize) -> Result<(), String> {
        Ok(())
    }

    /// The load's elastic expert bank; none for a load whose experts are
    /// held whole.
    fn expert_bank(&self) -> Option<&dyn ElasticBank>;

    /// The load's stored tables by the target block whose n-gram input
    /// reads each (`ResidentRoles::stored_blocks`). Test families hold none.
    fn stored_tables(
        &self,
    ) -> Result<std::collections::BTreeMap<usize, std::sync::Arc<crate::stored_table::StoredTable>>, String> {
        Ok(Default::default())
    }

    fn bound_constant_bytes(&self) -> Result<u64, &'static str> {
        self.target_constant_bytes()?
            .checked_add(self.optional_constant_bytes()?)
            .ok_or("bound graph constant charge overflows")
    }

    /// Whether a prompt chunk of `rows` rows may run in a chain of several
    /// launches (`ExecutorDomain::behind`).
    fn chains(&self, rows: usize) -> bool;

    /// The chunks a prompt group runs while decode waits, at most
    /// `capacity` (`ExecutorDomain::group_size`).
    fn waiting_group(&self, capacity: usize) -> usize;

    /// Submit a chain of target launches and its spare output slot
    /// (`TargetProgram::submit`), its encoding reading memory between its
    /// submissions by `pressure`, handed the family's expert bank
    /// (`ExpertBank::release_running`).
    fn submit_target(
        &mut self,
        chain: Chain<ValidatedTargetLaunch>,
        spare: crate::TargetGraphOutputLease,
        pressure: &dyn Fn(&dyn ElasticBank) -> Option<RunningPressure>,
    ) -> Result<Chain<Self::TargetSubmission>, crate::programs::ChainRefusal>;
    fn submit_head(
        &mut self,
        launch: ValidatedHeadLaunch,
    ) -> Result<Self::HeadSubmission, (SubmitError, ValidatedHeadLaunch)>;
    fn submit_vision(
        &mut self,
        launch: ValidatedVisionLaunch,
    ) -> Result<Self::VisionSubmission, (SubmitError, ValidatedVisionLaunch)>;
    fn submit_state(
        &mut self,
        launch: ValidatedStateLaunch,
    ) -> Result<Self::StateSubmission, (SubmitError, ValidatedStateLaunch)>;
}

pub struct NativeFamily {
    programs: Rc<AttestedPrograms>,
    resident: ResidentTarget,
    target: crate::programs::native_target::NativeTargetProgram,
    state: crate::programs::native_state::NativeStateProgram,
    head: Option<crate::programs::native_drafter::NativeDrafterProgram>,
    /// The streamed tables the bound head holds of its own, beside the
    /// target's: their stagings follow the class held as the target's do,
    /// held for the class held when it binds by `hold_bound_head`.
    head_tables: Vec<std::rc::Rc<crate::device_table::DeviceTable>>,
    vision: Option<crate::programs::native_vision::NativeVisionProgram>,
    /// The load's expert bank and its moves' kernels; none of a load whose
    /// experts are held whole.
    bank: Option<crate::experts::ServedBank>,
}

impl NativeFamily {
    /// What steps cost each streamed routed layer so far, and the last step
    /// completed; `None` of a load whose experts are held whole.
    pub fn expert_costs(&self) -> Option<crate::experts::ExpertCosts> {
        self.resident.bank.as_ref().map(|bank| bank.costs())
    }

    /// What streamed layer `layer`'s CPU site answered of step `step`
    /// (`Compute::answered`).
    #[cfg(test)]
    pub(crate) fn site_answered(&self, layer: usize, step: u64) -> Option<crate::experts::Answered> {
        self.resident.bank.as_ref()?.site_answered(layer, step)
    }

    /// The experts each streamed routed layer holds; `None` of a load whose
    /// experts are held whole.
    pub fn experts_held(&self) -> Option<Vec<Vec<usize>>> {
        self.resident.bank.as_ref().map(|bank| bank.held())
    }

    /// The family of `resident`, its bank's memory carved from what the
    /// load recorded allocating as it started (`ServedBank::new`).
    pub(crate) fn new(
        programs: Rc<AttestedPrograms>,
        resident: ResidentTarget,
        geometry: Decoder,
        ledgers: &crate::memory::ledger::DeviceLedgers,
        recorded: &mut crate::memory::ledger::Recorded,
    ) -> Result<Self, super::DomainError> {
        let invariant = |detail: String| super::DomainError::invariant(detail);
        let binding_constants = programs
            .target_graphs()
            .ok_or_else(|| invariant("target graph family was not prepared".into()))?
            .binding_constant_bytes()
            .map_err(invariant)?;
        let target = programs
            .bind_target(resident.clone(), geometry)
            .map_err(|error| invariant(error.to_string()))?;
        if target.constant_bytes().map_err(|error| invariant(error.into()))? != binding_constants {
            return Err(invariant("bound target graph constants differ from the claimed charge".into()));
        }
        let state = programs.bind_state();
        let bank = resident
            .bank
            .clone()
            .map(|bank| {
                // The stored tables the bank's plan holds, as the target imported them.
                let tables = bank
                    .plan()
                    .tables
                    .iter()
                    .map(|plan| {
                        let table = resident
                            .sublayers
                            .stored_tables()
                            .find_map(|(role, table)| (role == plan.role).then(|| table.clone()))
                            .ok_or_else(|| invariant(format!("the bank's table {:?} was not imported", plan.role)))?;
                        Ok(crate::experts::HeldTable { plan: plan.clone(), table, last_resize: std::sync::Mutex::new(None) })
                    })
                    .collect::<Result<Vec<_>, super::DomainError>>()?;
                crate::experts::ServedBank::new(bank, &programs, tables, ledgers, recorded)
            })
            .transpose()?;
        Ok(Self {
            programs,
            resident,
            target,
            state,
            head: None,
            head_tables: Vec::new(),
            vision: None,
            bank,
        })
    }

    /// Every streamed table a lane's launches name rows of, each once: the
    /// target's and the bound head's own.
    fn device_tables(&self) -> Vec<std::rc::Rc<crate::device_table::DeviceTable>> {
        let mut tables: Vec<std::rc::Rc<crate::device_table::DeviceTable>> = Vec::new();
        self.resident.visit_weights(|weight| {
            if let Some(table) = weight.table().device() {
                tables.push(table.clone());
            }
        });
        tables.extend(self.head_tables.iter().cloned());
        let mut unique = Vec::with_capacity(tables.len());
        for table in tables {
            if !unique.iter().any(|held| std::rc::Rc::ptr_eq(held, &table)) {
                unique.push(table);
            }
        }
        unique
    }
}

/// A group's `places`, never none: the floor's one at least.
fn places_of(places: usize) -> Result<std::num::NonZeroU64, String> {
    u64::try_from(places)
        .ok()
        .and_then(std::num::NonZeroU64::new)
        .ok_or_else(|| format!("a group of {places} places"))
}

impl ProgramFamily for NativeFamily {
    type TargetSubmission =
        <crate::programs::native_target::NativeTargetProgram as TargetProgram>::Submission;
    type HeadSubmission =
        <crate::programs::native_drafter::NativeDrafterProgram as HeadProgram>::Submission;
    type VisionSubmission =
        <crate::programs::native_vision::NativeVisionProgram as VisionProgram>::Submission;
    type StateSubmission =
        <crate::programs::native_state::NativeStateProgram as StateProgram>::Submission;

    fn bind_head(
        &mut self,
        resident: ResidentHead,
        definition: &ModelDefinition,
    ) -> Result<(), String> {
        if self.head.is_none() {
            let binding_constants = self
                .programs
                .drafter_graphs()
                .ok_or("head graph family was not prepared")?
                .binding_constant_bytes()?;
            // Its own streamed table, not the target's, which the class's
            // claim already holds for every lane.
            let mut target = Vec::new();
            self.resident.visit_weights(|weight| target.extend(weight.table().device().cloned()));
            let table = resident
                .embedding
                .table()
                .device()
                .filter(|table| !target.iter().any(|held| std::rc::Rc::ptr_eq(held, table)))
                .cloned();
            let head = self
                .programs
                .bind_head(resident, definition)
                .map_err(|error| error.to_string())?;
            if head.constant_bytes()? != binding_constants {
                return Err("bound head graph constants differ from the claimed charge".into());
            }
            // Held at the floor, as charged with the head's weights; a class
            // held beyond it is held under its claim (`hold_bound_head`).
            self.head_tables.extend(table);
            self.head = Some(head);
        }
        Ok(())
    }
    fn bind_vision(
        &mut self,
        resident: ResidentVision,
        definition: &ModelDefinition,
    ) -> Result<(), String> {
        if self.vision.is_none() {
            let binding_constants = self
                .programs
                .vision_graphs()
                .ok_or("vision graph family was not prepared")?
                .binding_constant_bytes()?;
            let vision = self
                .programs
                .bind_vision(resident, definition)
                .map_err(|error| error.to_string())?;
            if vision.constant_bytes()? != binding_constants {
                return Err("bound vision graph constants differ from the claimed charge".into());
            }
            self.vision = Some(vision);
        }
        Ok(())
    }
    fn head_is_bound(&self) -> bool {
        self.head.is_some()
    }
    fn family_graph_replays(&self) -> Vec<(&'static str, seismic::GraphReplays)> {
        let mut families = self.programs.graph_replays();
        families.push(("target overlays", self.target.overlay_graph_replays()));
        families
    }
    fn records_decode_ahead(&self) -> bool {
        self.programs.records_decode_ahead()
    }
    fn vision_is_bound(&self) -> bool {
        self.vision.is_some()
    }
    fn state_is_bound(&self) -> bool {
        true
    }

    fn bound_head_demand(&self, places: usize) -> Result<crate::memory::ClaimDemand, String> {
        let device = self.head_tables.iter().try_fold(0u64, |bytes, table| {
            let grown = table.staging_bytes(places)?.saturating_sub(table.staging_bytes(1)?);
            bytes.checked_add(grown).ok_or_else(|| "a head's stagings overflow".to_owned())
        })?;
        Ok(crate::memory::ClaimDemand { device, ..Default::default() })
    }

    fn hold_bound_head(&self, places: usize) -> Result<(), String> {
        self.head_tables.iter().try_for_each(|table| table.hold_places(places))
    }

    fn unbind_optional(&mut self, component: OptionalComponent) {
        match component {
            OptionalComponent::Drafter => {
                self.head = None;
                self.head_tables.clear();
            }
            OptionalComponent::Vision => self.vision = None,
        }
    }

    fn target_weight_bytes(&self) -> Result<u64, &'static str> {
        self.resident.storage_bytes()
    }

    fn visit_model_storage(&self, visit: &mut dyn FnMut(&seismic::Tensor)) {
        self.resident.visit_storage(visit);
        self.target.visit_constants(visit);
        if let Some(head) = &self.head {
            head.visit_constants(visit);
        }
        if let Some(vision) = &self.vision {
            vision.visit_constants(visit);
        }
    }

    fn prepared_program_bytes(&self) -> u64 {
        self.programs.device_storage_bytes()
    }

    fn expert_bank(&self) -> Option<&dyn ElasticBank> {
        self.bank.as_ref().map(|bank| bank as &dyn ElasticBank)
    }

    fn stored_tables(
        &self,
    ) -> Result<std::collections::BTreeMap<usize, std::sync::Arc<crate::stored_table::StoredTable>>, String> {
        self.resident.sublayers.stored_blocks()
    }

    fn target_constant_bytes(&self) -> Result<u64, &'static str> {
        self.target.constant_bytes()
    }

    /// A group's graph buffers and, of a streamed load, its places' bank
    /// words (`ExpertBank::placed_demand`), each beside the floor's.
    fn group_demand(&self, places: usize) -> Result<crate::memory::ClaimDemand, String> {
        let bank = match &self.resident.bank {
            Some(bank) => bank.placed_demand(places_of(places)?)?,
            None => Default::default(),
        };
        // Each streamed table's stagings beyond the floor's
        // (`DeviceTable::hold_places`).
        let stagings = self.device_tables().iter().try_fold(0u64, |bytes, table| {
            let grown = table.staging_bytes(places)?.saturating_sub(table.staging_bytes(1)?);
            bytes.checked_add(grown).ok_or_else(|| "a group's stagings overflow".to_owned())
        })?;
        let device = self.target.group_bytes(places)?.checked_add(stagings).ok_or("a group's device bytes overflow")?;
        bank.checked_plus(crate::memory::ClaimDemand { device, ..Default::default() })
            .ok_or_else(|| "a group's demand overflows".to_owned())
    }

    /// Hold a group's graph buffers and its places' bank words, each
    /// allocated first, so a failure of either keeps both held before; the
    /// bank's words are the steps' once the buffers are held.
    fn hold_group(&self, places: usize, _quiet: super::Quiet) -> Result<(), crate::resources::AllocationError> {
        let invariant = |detail: String| {
            crate::resources::AllocationError::Plan(crate::error::InvariantError { context: "a group's places", detail })
        };
        let placed = match &self.resident.bank {
            Some(bank) => Some((bank, bank.place(places_of(places).map_err(invariant)?)?)),
            None => None,
        };
        self.target.hold_group(places)?;
        for table in self.device_tables() {
            table.hold_places(places).map_err(invariant)?;
        }
        if let Some((bank, placed)) = placed {
            bank.hold_places(placed).map_err(invariant)?;
        }
        Ok(())
    }

    fn optional_constant_bytes(&self) -> Result<u64, &'static str> {
        let head = self
            .head
            .as_ref()
            .map(|head| head.constant_bytes())
            .transpose()?
            .unwrap_or(0);
        let vision = self
            .vision
            .as_ref()
            .map(|vision| vision.constant_bytes())
            .transpose()?
            .unwrap_or(0);
        head.checked_add(vision)
            .ok_or("optional graph constant charge overflows")
    }

    /// Every launch runs in a chain as it runs alone, except a streamed
    /// load's bank-served ones: their fills interleaved layer-major would
    /// place experts of several steps at once. A streamed load's chain of
    /// scratch-served chunks is one prompt group, as a load's chain of
    /// prompt chunks is (`ExecutorDomain::behind`).
    fn chains(&self, rows: usize) -> bool {
        self.resident.bank.is_none()
            || !crate::operators::routed::fused_graph::decodes(rows as u64)
    }

    /// The size rule (`ExpertBank::group_size`): enough chunks that their
    /// compute hides their cold experts' reads. A load without a bank reads
    /// none, so its rule is its floor, one chunk.
    fn waiting_group(&self, capacity: usize) -> usize {
        match &self.resident.bank {
            Some(bank) => bank.group_size(capacity),
            None => 1,
        }
    }

    fn submit_target(
        &mut self,
        chain: Chain<ValidatedTargetLaunch>,
        spare: crate::TargetGraphOutputLease,
        pressure: &dyn Fn(&dyn ElasticBank) -> Option<RunningPressure>,
    ) -> Result<Chain<Self::TargetSubmission>, crate::programs::ChainRefusal> {
        let bank = self.bank.as_ref().map(|bank| bank as &dyn ElasticBank);
        self.target.submit(chain, spare, &|| bank.and_then(pressure))
    }
    fn submit_head(
        &mut self,
        launch: ValidatedHeadLaunch,
    ) -> Result<Self::HeadSubmission, (SubmitError, ValidatedHeadLaunch)> {
        self.head
            .as_mut()
            .expect("head was bound before submission")
            .submit(launch)
    }
    fn submit_vision(
        &mut self,
        launch: ValidatedVisionLaunch,
    ) -> Result<Self::VisionSubmission, (SubmitError, ValidatedVisionLaunch)> {
        self.vision
            .as_mut()
            .expect("vision was bound before submission")
            .submit(launch)
    }
    fn submit_state(
        &mut self,
        launch: ValidatedStateLaunch,
    ) -> Result<Self::StateSubmission, (SubmitError, ValidatedStateLaunch)> {
        self.state.submit(launch)
    }
}
