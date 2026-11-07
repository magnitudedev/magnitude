//! Model-derived memory charges for metadata-only assessment.
//!
//! The standard assessment workload is one conversation at the fit depth
//! `min(context_limit, 100_000)` on a clean load. Its charge is composed of
//! exact header terms (resident weights, history at the fit depth, recurrent
//! banks, the prepared programs' owner) and upper bounds taken from the same
//! checked class enumeration and slot multipliers production preparation
//! uses (graph pools and bound constants, the startup qualification/import
//! peak). The load planner and the state layout remain the authorities for
//! physical representations and codec planes; the platform policy owns each
//! domain's stable capacity and reserve.

use super::resources::{GraphSlots, NativeGraphCharge, history_row_bytes, recurrent_bank_bytes};
use super::{
    ComponentSelection, ModelLoadPlan, PlannedMethod, ResourceLimits,
};
use crate::assessment::DomainFit;
use crate::memory::ClaimDemand;
use crate::platform::{DomainRole, Fit, FitMode, HostRoom, StorageDemand};
use crate::{AttestedPrograms, GraphError};
use magnitude_family_contracts::ModelDefinition;
use magnitude_state::{BankCapacity, KvCodec, ModelStateLayout};
use seismic::BackendName;
use seismic::{DeviceTopology, Executor, Status, StoreId};
use std::num::NonZeroU64;

/// Exact resident and per-state terms of the standard workload, derived
/// without opening a device or reading tensor payloads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AssessmentMemoryTerms {
    pub target_weights: u64,
    pub head_weights: u64,
    pub vision_weights: u64,
    pub history_per_token: u64,
    pub recurrent_per_bank: u64,
    /// Recurrent banks the workload's conversations hold: each one's
    /// accepted bank, every in-flight successor the planned lookahead keeps
    /// and the checkpoints the prefix cache retains of it, and the pristine
    /// seed.
    pub recurrent_banks: u64,
    pub fit_depth: u64,
    /// The workload's conversations (`ResourceLimits::conversations`), each
    /// at the fit depth.
    pub conversations: u64,
    /// Host-resident gathered tables (model-family plan §3.7): claims in the
    /// system-RAM domain, never device weights.
    pub host_table_bytes: u64,
    /// Complete physical weight demand of every selected component.
    pub weight_demand: ClaimDemand,
}

/// Upper bounds for every nonresident charge of a clean load. The prepared
/// resources (`prepared_resource_bytes`) are all steady native holdings: the
/// prepared programs' owner, the graph pools with their output, upload
/// regions and bound constants, recorded graphs and the worker's device
/// bytes; the startup bound covers the additional qualification/import peak;
/// the component bound is the import peak of an optional component a served
/// request loads; the staging bound is the host upload peak of a dedicated
/// device's imports.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AssessmentMemoryBounds {
    /// Every graph pool and bound constant
    /// (`AssessmentGraphResourceBounds::total_bytes`).
    pub graph_bytes: u64,
    /// What the startup graphs (`AssessmentGraphResourceBounds::recorded_graphs`)
    /// hold on the planned device, in its device and upload storage.
    pub recorded_graph_bytes: u64,
    pub startup_additional_bytes: u64,
    pub component_upload_bytes: u64,
    pub staging_upload_bytes: u64,
    /// Host heap of the sealed target graphs
    /// (`TargetGraphBounds::sealed`): system RAM
    /// like the host tables, held from sealing on.
    pub sealed_graph_host_bytes: SealedGraphHostBytes,
    /// Host heap of the static bindings of those graphs
    /// (`PlannedSealedGraphs::bound`): system RAM, held from the binding of
    /// the resident weights on, after the import.
    pub bound_graph_host_bytes: u64,
    /// Host heap of the drafter's graphs
    /// (`AssessmentGraphResourceBounds::drafter_graph_host_bytes`).
    pub drafter_graph_host_bytes: u64,
    pub host_process: HostProcessBytes,
    pub device_process: DeviceProcessBytes,
}

impl AssessmentMemoryBounds {
    /// Every steady native holding: the prepared programs' owner
    /// ([`AttestedPrograms::DEVICE_BYTES`]), the graph pools and bound
    /// constants, recorded graphs and the worker's device bytes.
    pub fn prepared_resource_bytes(&self) -> Result<u64, String> {
        [
            AttestedPrograms::DEVICE_BYTES,
            self.graph_bytes,
            self.recorded_graph_bytes,
            self.device_process.bytes()?,
        ]
        .into_iter()
        .try_fold(0u64, |total, bytes| total.checked_add(bytes))
        .ok_or_else(|| "assessment prepared resource bound overflow".to_owned())
    }
}

/// What a load's worker holds of its device's memory outside every charged
/// allocation: the driver objects its device's creation makes where the
/// usable capacity does not already exclude them, the driver objects of the
/// native kernels it prepares (Vulkan pipelines; CUDA modules measure
/// none), and the free space of the blocks its allocations are carved from.
/// Each is a cost measured on the backend, the kernels' times a count the
/// plan fixes (`DeviceProcessCosts::of`). A unified device's are part of
/// its worker's host bytes, which are measured with them, so they are zero
/// there.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceProcessBytes {
    pub context: u64,
    pub programs: u64,
    pub slack: u64,
}

/// The measured device cost of a load's worker on one backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DeviceProcessCosts {
    context: u64,
    per_kernel: u64,
    slack: u64,
}

impl DeviceProcessCosts {
    /// Measured by the rig's memory procedure at `--backend cuda|vulkan`
    /// (`inference/evaluation/rig`, `host/memory.py`) on an A10G, an L40S
    /// and an A10 (CUDA) and a Radeon PRO V710 (Vulkan, Linux and Windows).
    ///
    /// CUDA: what the device lost beyond what the engine holds (`held`) is
    /// the process's context (within 20 MB), and the usable capacity
    /// already excludes it: measured on the device at discovery, and under
    /// WDDM the free figure read after the context exists, which also
    /// excludes WDDM's own share. It is not charged again. Kernel modules
    /// showed no device cost. Vulkan: the device's creation (67–74 MB)
    /// takes from the heap or budget that is the capacity, and the driver
    /// objects beyond the charge grow with the kernels (at most 7.5 MB each
    /// on Windows, 3.5 MB on Linux).
    ///
    /// The block allocator holds free space beside the charge at every
    /// moment, the fit depth included; it depends on the order of
    /// allocations, not on a count the plan fixes. On CUDA it never exceeded
    /// one block cap (0.28–1.04 GB measured), which is its cost. On Vulkan,
    /// whose upload storage has blocks of its own, it reached 1.151 GB on
    /// both systems, the most measured. Metal and the CPU are unified: their
    /// kernels and runtime are measured in the worker's host bytes.
    fn of(backend: BackendName) -> Self {
        match backend {
            BackendName::Metal | BackendName::Cpu => Self {
                context: 0,
                per_kernel: 0,
                slack: 0,
            },
            BackendName::Cuda => Self {
                context: 0,
                per_kernel: 0,
                slack: CUDA_BLOCK_SLACK,
            },
            BackendName::Vulkan => Self {
                context: VULKAN_CONTEXT,
                per_kernel: VULKAN_PER_KERNEL,
                slack: VULKAN_BLOCK_SLACK,
            },
        }
    }
}

const CUDA_BLOCK_SLACK: u64 = seismic::DEVICE_BLOCK_CAP_BYTES;
const VULKAN_CONTEXT: u64 = 74_000_000;
const VULKAN_PER_KERNEL: u64 = 7_500_000;
const VULKAN_BLOCK_SLACK: u64 = 1_151_000_000;

impl DeviceProcessBytes {
    /// The device bytes of a load on `backend` that prepares `kernels`
    /// native kernels (`HostProcessBytes::kernels`).
    pub fn derive(backend: BackendName, kernels: u64) -> Result<Self, String> {
        let costs = DeviceProcessCosts::of(backend);
        Ok(Self {
            context: costs.context,
            programs: costs
                .per_kernel
                .checked_mul(kernels)
                .ok_or("prepared program device bytes overflow")?,
            slack: costs.slack,
        })
    }

    pub fn bytes(&self) -> Result<u64, String> {
        self.context
            .checked_add(self.programs)
            .and_then(|bytes| bytes.checked_add(self.slack))
            .ok_or_else(|| "worker device bytes overflow".to_owned())
    }
}

/// What a load's worker holds in host RAM that no planned holding names:
/// the runtime it opens its device in, the model's vocabulary, the programs
/// it prepares and, for a model that reads images, the vision encoder's
/// graphs and programs prepared when a request first sends one. Each term
/// is a count the model's plan fixes times a cost measured on the backend
/// (`HostProcessCosts::of`), held from its preparation to the end of
/// serving, so their sum bounds what the worker holds of them at any moment
/// of the load and while serving.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HostProcessBytes {
    pub runtime: u64,
    pub vocabulary: u64,
    pub programs: u64,
    pub vision: u64,
    /// The native kernels the load prepares (`kernel_inventory`), the count
    /// `programs` is priced on.
    pub kernels: u64,
}

/// The measured host cost of a load's worker on one backend. Every cost is
/// a measurement of real loads on that backend, not a margin: what a worker holds beyond its
/// planned holdings, apportioned to the counts that drive it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct HostProcessCosts {
    runtime: u64,
    per_token: u64,
    per_kernel: u64,
    vision: u64,
}

impl HostProcessCosts {
    /// The host bytes of `kernels` prepared native programs: a load's worker's and a measurement
    /// job's alike.
    fn programs(&self, kernels: u64) -> Result<u64, String> {
        self.per_kernel
            .checked_mul(kernels)
            .ok_or_else(|| "prepared program host bytes overflow".to_owned())
    }

    /// Each backend's costs are measured from its loads: the process
    /// footprint at load end less everything the plan holds then is the
    /// worker's own memory, and the costs are the least-overestimating ones
    /// under which no measured load exceeds its plan.
    ///
    /// Procedure (`inference/evaluation/rig`, `instruments/memory.py`):
    /// the rig's memory instrument at `--backend metal|cpu --model M` loads one
    /// model plain in a worker process (`magnitude-harness memory`) and sets its own memory at
    /// load end against its charge; `python -m magnitude_rig.host.memory
    /// --fit` over the runs' `host-costs.json` fits the costs to every load.
    ///
    /// Runtime, vocabulary and kernel costs: an M4 Pro (macOS 15.7.9),
    /// 2026-10-23, from the Q4 targets of MiniCPM5 1B and 2B, MiniCPM-V 4.7
    /// 1B, LFM2.5 2.6B, Qwen3.5 4B and Gemma 4 E2B QAT, once on Metal and
    /// twice on the CPU (identical CPU loads differ by up to 17 MB), each
    /// load's storage equal to its plan's term by term (a load's own memory
    /// holds none of it), and no prepared kernel holding its entry's
    /// lowered program or a copy of the module's sources. Each measured
    /// load's own memory is at most its charge, by 0 to 104 MB on Metal and
    /// 0 to 268 MB on the CPU, the costs the fit's rounded up to the
    /// kilobyte. Six models do not separate the three costs well: the fit
    /// puts the CPU's runtime cost at zero.
    ///
    /// Metal's vision cost is from an M4 Max (macOS 15.5): what Qwen3.5 2B's
    /// and Gemma 4 E2B's workers held after their first image beyond
    /// everything else their plans hold, the two within 2 %. The CPU's
    /// vision is not measured and takes Metal's.
    ///
    /// CUDA and Vulkan hold their workers' host bytes in host RAM beside
    /// a dedicated device, not measured there; they take Metal's.
    fn of(backend: BackendName) -> Self {
        let metal = Self {
            runtime: METAL_RUNTIME,
            per_token: METAL_PER_TOKEN,
            per_kernel: METAL_PER_KERNEL,
            vision: METAL_VISION,
        };
        match backend {
            BackendName::Metal | BackendName::Vulkan | BackendName::Cuda => metal,
            BackendName::Cpu => Self {
                runtime: CPU_RUNTIME,
                per_token: CPU_PER_TOKEN,
                per_kernel: CPU_PER_KERNEL,
                ..metal
            },
        }
    }
}

/// What a measurement job's worker holds beyond what its points lay out, in its device's
/// storage: its device bytes for `kernels` programs (`DeviceProcessBytes`), and where that
/// storage is host RAM (`host`, a unified device), its runtime and programs' host bytes
/// (`HostProcessCosts`), which a load's worker is charged for the same.
pub fn measurement_process_bytes(backend: BackendName, kernels: u64, host: bool) -> Result<u64, String> {
    let device = DeviceProcessBytes::derive(backend, kernels)?.bytes()?;
    if !host {
        return Ok(device);
    }
    let costs = HostProcessCosts::of(backend);
    costs
        .programs(kernels)?
        .checked_add(costs.runtime)
        .and_then(|bytes| bytes.checked_add(device))
        .ok_or_else(|| "measurement process bytes overflow".to_owned())
}

const METAL_RUNTIME: u64 = 80_992_000;
const METAL_PER_TOKEN: u64 = 612;
const METAL_PER_KERNEL: u64 = 5_443_000;
const METAL_VISION: u64 = 210_000_000;
const CPU_RUNTIME: u64 = 0;
const CPU_PER_TOKEN: u64 = 1_452;
const CPU_PER_KERNEL: u64 = 6_045_000;

impl HostProcessBytes {
    /// The worker's unplanned host bytes for a load of `programs` (the
    /// load's program plan under `limits`) on `backend`.
    pub fn derive(
        definition: &ModelDefinition,
        load: &ModelLoadPlan,
        programs: &super::ProgramPlan,
        limits: ResourceLimits,
        backend: BackendName,
    ) -> Result<Self, String> {
        let costs = HostProcessCosts::of(backend);
        let kernels = crate::native::kernel_requests(
            backend,
            programs,
            definition,
            load,
            crate::TuningLimits::of(limits, definition),
        )
        .map_err(|failure| crate::CatalogError::native(backend, failure).to_string())?
        .len() as u64;
        Ok(Self {
            runtime: costs.runtime,
            vocabulary: costs
                .per_token
                .checked_mul(definition.decoder.vocabulary)
                .ok_or("vocabulary host bytes overflow")?,
            programs: costs.programs(kernels)?,
            vision: if programs.vision().is_some() {
                costs.vision
            } else {
                0
            },
            kernels,
        })
    }

    pub fn bytes(&self) -> Result<u64, String> {
        self.runtime
            .checked_add(self.vocabulary)
            .and_then(|bytes| bytes.checked_add(self.programs))
            .and_then(|bytes| bytes.checked_add(self.vision))
            .ok_or_else(|| "worker host bytes overflow".to_owned())
    }
}

/// The host heap a load's sealed target graphs hold (their plans, the
/// prepared graphs that hold them and the family's constant values) and
/// its state copy graphs, from both sides (`PlannedSealedGraphs`). The
/// readout, head, draft and vision graphs are not in it. The two are one figure wherever the
/// plan fixes which blocks share a sealed plan, and apart where blocks of
/// one shape read layer scales the plan does not know.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SealedGraphHostBytes {
    /// No load holds more: what a load is held to a domain's capacity by
    /// (`AssessmentMemoryCharge::assess_fit`).
    pub at_most: u64,
    /// No load holds fewer: what is held to the load's own planned holding
    /// reads this (`HostPeak::sealed_graphs`).
    pub at_least: u64,
}

/// Header-derived charges that are known before graph formation. The startup
/// charge is the same qualification/import upper bound used by the production
/// resource planner; the component charge is the import window an optional
/// head or vision component claims beside its weights when a request first
/// needs it (`DomainState::component_growth`); the staging charge is the
/// largest selected source tensor's import window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AssessmentHeaderBounds {
    pub startup_additional_bytes: u64,
    pub component_upload_bytes: u64,
    pub staging_upload_bytes: u64,
}

/// The standard workload's charge in each memory role a load touches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AssessmentMemoryCharge {
    /// Weights and slab-rounded state at the fit depth, all exact.
    pub exact_resident_bytes: u64,
    /// The workload's state at the fit depth, part of `exact_resident_bytes`.
    pub state_bytes: u64,
    /// The state a load allocates at startup.
    pub startup_state_bytes: u64,
    /// The state the loaded model holds before its first request
    /// (`StateResourcePlan::loaded_state_bytes`).
    pub loaded_state_bytes: u64,
    pub bounds: AssessmentMemoryBounds,
    /// What the device's allocation domain holds while the loaded model
    /// serves at the fit depth: the weights, every prepared resource and the
    /// workload's state. A first load's tuning holds no more than this.
    pub serving_bytes: u64,
    /// Everything the device's allocation domain holds at the clean-load peak.
    pub allocation_bytes: u64,
    /// Host RAM a dedicated device's staged import holds at its peak.
    pub staging_bytes: u64,
    /// Host-resident tables: system RAM, which is the allocation domain of a
    /// unified-memory device and the staging domain of a dedicated one.
    pub host_table_bytes: u64,
    /// Complete physical weight demand, part of `allocation_bytes`: each
    /// allocation in the storage it lies in (`assess_fit`).
    pub weight_demand: ClaimDemand,
    /// What a prompt group's class of the plan's chain places holds beyond
    /// the floor's, granted as a group opens (`ExecutorDomain::grant_group`)
    /// and so part of no other figure: a bank planned beyond its floor
    /// leaves room for it (`ExecutionPlanDraft::with_bank`).
    pub group_class: ClaimDemand,
}

/// A complete fit result: every domain the load touches, and whether the
/// workload fits all of them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AssessmentFit {
    pub verdict: AssessmentFitVerdict,
    pub domains: Vec<DomainFit>,
}

impl AssessmentFit {
    /// The bytes a plan's holdings settle within: its allocation domain's
    /// stable room, its planning reserve aside (`StableRoom`).
    pub fn settle_bound(&self) -> Result<u64, String> {
        self.domains
            .iter()
            .find(|domain| domain.role == DomainRole::Allocation)
            .map(|domain| domain.capacity_bytes - domain.reserve_bytes)
            .ok_or_else(|| "the fit reaches no allocation domain".to_owned())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AssessmentFitVerdict {
    Fits,
    /// The domain with the largest deficit limits the load.
    DoesNotFit {
        limiting: StoreId,
        deficit_bytes: NonZeroU64,
    },
}

/// A target graph family's checked bounds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TargetGraphBounds {
    /// Its pools at the startup slots.
    pub charge: NativeGraphCharge,
    /// The graphs a load seals and the host heap they hold, from the
    /// sealer's own enumeration: one count, or bounds from both sides where
    /// blocks that read a layer scale could share a plan
    /// (`PlannedSealedGraphs::exact`). Host memory, not part of
    /// `AssessmentGraphResourceBounds::total_bytes`.
    pub sealed: crate::PlannedSealedGraphs,
    /// The constants and floor buffers its binding allocates.
    pub binding_constant_bytes: u64,
    /// The bytes its own buffers of a prompt group of the plan's chain
    /// places hold beyond the floor's (`CheckedTargetFamily::group_bytes`).
    pub group_bytes: u64,
}

impl TargetGraphBounds {
    fn of(
        family: crate::programs::native_target_graph::CheckedTargetFamily,
        slots: super::GraphSlots,
    ) -> Result<Self, String> {
        Ok(Self {
            charge: NativeGraphCharge::from_checked(family.resources.storage, family.upload_regions, slots, family.graphs)?,
            sealed: family.sealed,
            binding_constant_bytes: family.resources.binding_constant_bytes,
            group_bytes: family.group_bytes,
        })
    }
}

/// A projector's vision family's checked bounds. No drafter changes the
/// vision program, so a drafted plan and its target's plan alone hold the
/// same family ([`UndraftedBounds`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VisionGraphBounds {
    /// Its pools at the startup slots.
    pub charge: NativeGraphCharge,
    /// The constants its binding allocates.
    pub binding_constant_bytes: u64,
}

/// What a drafted plan's certification gives of its target's plan alone at
/// the same holding of experts: its target family without the drafter's
/// writes, and the vision family both hold. Every other family of the target
/// alone is its own ([`AssessmentGraphResourceBounds::of_target`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UndraftedBounds {
    pub target: TargetGraphBounds,
    pub vision: Option<VisionGraphBounds>,
}

/// Checked graph-pool demand and distinct bound constants from the same class
/// enumeration and slot multipliers as native preparation: an upper bound for
/// every graph pool a clean load commits on the selected backend. Deriving
/// it builds every graph a load prepares, so it fails with
/// `GraphError::KernelDomain` exactly when the model's program calls a kernel
/// outside its domain on the backend.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AssessmentGraphResourceBounds {
    pub target: TargetGraphBounds,
    pub readout: NativeGraphCharge,
    /// A prompt chunk's readout family, at no slots: a group's class grows
    /// it; none of a readout without a chunk's classes.
    pub chunk_readout: Option<NativeGraphCharge>,
    pub head: Option<NativeGraphCharge>,
    pub vision: Option<NativeGraphCharge>,
    pub state: NativeGraphCharge,
    pub binding_constant_bytes: u64,
    pub total_bytes: u64,
    /// What the startup activations' launches record
    /// (`NativeGraphCharge::launch`): the graphs a load keeps before any
    /// prompt group grows its class, which a device's graph cost prices
    /// (`AssessmentHeaderBounds::with_graph_resource_bound`).
    pub recorded_graphs: seismic::GraphCensus,
    /// The host heap of the state copy graphs a load prepares
    /// (`PreparedStateCopyGraphs::host_heap`): their sealed plans and what
    /// stores them, exact from the classes the load prepares. Host memory,
    /// not part of `total_bytes`.
    pub sealed_state_graphs: seismic::HostHeap,
    /// The host heap of the drafter's graphs (a head's, a shared head's or
    /// a separate draft's): each class's sealed plan and the family's one
    /// copy of each constant value, exact from the classes the load
    /// prepares. Host memory, not part of `total_bytes`.
    pub drafter_graph_host_bytes: u64,
}

impl AssessmentGraphResourceBounds {
    /// What a prompt group's class of `limits`' chain places holds beyond
    /// the floor's of one place, as a group opening grants it
    /// (`ExecutorDomain::grant_group`): its pools' slots
    /// (`ResourceLimits::class_slots`; the target's and the head's beyond
    /// their floor's, a chunk's readout slots beyond the floor's readout
    /// slots), the graphs they record priced by `cost` beyond the startup
    /// graphs', and the target's own group buffers.
    pub fn group_class(&self, limits: ResourceLimits, cost: seismic::GraphCost) -> Result<ClaimDemand, String> {
        let (floor, class) = (limits.class_slots(1), limits.class_slots(limits.chain_capacity.get()));
        let chunks = GraphSlots {
            activations: class.readout.activations.saturating_sub(floor.readout.activations),
            output: class.readout.output.saturating_sub(floor.readout.output),
        };
        let none = GraphSlots { activations: 0, output: 0 };
        let grown = [
            Some((&self.target.charge, floor.target, class.target)),
            self.chunk_readout.as_ref().map(|chunk| (chunk, none, chunks)),
            self.head.as_ref().map(|head| (head, floor.head, class.head)),
        ];
        let (pools, graphs) = grown.into_iter().flatten().try_fold(
            (self.target.group_bytes, seismic::GraphCensus::default()),
            |(bytes, census), (charge, from, to)| {
                let (more, graphs) = charge.growth(from, to)?;
                bytes
                    .checked_add(more)
                    .zip(census.plus(graphs))
                    .ok_or_else(|| "a prompt group's class overflows".to_owned())
            },
        )?;
        super::graph_demand(cost, self.recorded_graphs, graphs)?
            .checked_plus(ClaimDemand { device: pools, ..Default::default() })
            .ok_or_else(|| "a prompt group's class overflows".to_owned())
    }

    pub fn derive(
        definition: &ModelDefinition,
        load: &ModelLoadPlan,
        state: &super::StateResourcePlan,
        method: PlannedMethod,
        codec: KvCodec,
        limits: super::ResourceLimits,
        backend: BackendName,
    ) -> Result<Self, GraphError> {
        let plan = Self::program_plan(definition, load, method, codec)?;
        let target =
            crate::programs::native_target_graph::checked_target_family(
                backend,
                load,
                &definition.decoder,
                state,
                plan.target(),
                limits,
                load.pass(limits),
            )?;
        let target = TargetGraphBounds::of(target, limits.startup_slots().target)?;
        let vision = Self::vision(definition, load, &plan, limits, backend)?;
        Self::of_target(definition, load, &plan, state, method, limits, backend, target, vision)
    }

    /// [`Self::derive`], with the bounds of the same target's family
    /// without its drafter's writes (its draft taps and the residual its
    /// head is conditioned on): the target family a load of the target alone
    /// at the same holding of experts prepares, from the one certification
    /// of the layouts both families seal into.
    pub fn derive_with_undrafted(
        definition: &ModelDefinition,
        load: &ModelLoadPlan,
        state: &super::StateResourcePlan,
        method: PlannedMethod,
        codec: KvCodec,
        limits: super::ResourceLimits,
        backend: BackendName,
    ) -> Result<(Self, UndraftedBounds), GraphError> {
        let plan = Self::program_plan(definition, load, method, codec)?;
        let [target, undrafted] = crate::programs::native_target_graph::checked_target_and_undrafted(
            backend,
            load,
            &definition.decoder,
            state,
            plan.target(),
            limits,
            load.pass(limits),
        )?;
        let slots = limits.startup_slots().target;
        let target = TargetGraphBounds::of(target, slots)?;
        let vision = Self::vision(definition, load, &plan, limits, backend)?;
        let bounds = Self::of_target(definition, load, &plan, state, method, limits, backend, target, vision.clone())?;
        Ok((bounds, UndraftedBounds { target: TargetGraphBounds::of(undrafted, slots)?, vision }))
    }

    /// The bounds of `plan`'s vision family, of a projector's load.
    fn vision(
        definition: &ModelDefinition,
        load: &ModelLoadPlan,
        plan: &super::ProgramPlan,
        limits: super::ResourceLimits,
        backend: BackendName,
    ) -> Result<Option<VisionGraphBounds>, GraphError> {
        match (plan.vision(), definition.vision.as_ref()) {
            (Some(vision_plan), Some(vision_definition)) => {
                let patch_rows = crate::programs::native_vision::image_patch_classes(
                    limits.max_image_cells,
                    vision_definition,
                )?;
                let vision_graph = crate::programs::native_vision::checked_vision_family_resources(
                    backend,
                    load,
                    vision_definition,
                    vision_plan,
                    patch_rows,
                )?;
                Ok(Some(VisionGraphBounds {
                    charge: NativeGraphCharge::serial(vision_graph.storage, limits.startup_slots().vision)?,
                    binding_constant_bytes: vision_graph.binding_constant_bytes,
                }))
            }
            (None, None) => Ok(None),
            _ => Err("vision program and definition disagree".into()),
        }
    }

    /// The program plan the graphs are of, which must draft by `method`.
    pub(crate) fn program_plan(
        definition: &ModelDefinition,
        load: &ModelLoadPlan,
        method: PlannedMethod,
        codec: KvCodec,
    ) -> Result<super::ProgramPlan, GraphError> {
        let plan = load
            .program_plan(definition, codec)
            .map_err(|error| error.to_string())?;
        if plan.drafts_with_head() != matches!(method, PlannedMethod::Mtp { .. })
            || plan.draft().is_some() != matches!(method, PlannedMethod::DFlash { .. })
        {
            return Err("assessment method and drafter program disagree".into());
        }
        Ok(plan)
    }

    /// The bounds of a load of `plan` (its [`Self::program_plan`]) whose
    /// target and vision families' are `target` and `vision`: every other
    /// family's certified here.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn of_target(
        definition: &ModelDefinition,
        load: &ModelLoadPlan,
        plan: &super::ProgramPlan,
        state: &super::StateResourcePlan,
        method: PlannedMethod,
        limits: super::ResourceLimits,
        backend: BackendName,
        target: TargetGraphBounds,
        vision: Option<VisionGraphBounds>,
    ) -> Result<Self, GraphError> {
        // The same startup slots a load commits: one request's need.
        let slots = limits.startup_slots();
        let (readout_storage, chunk_storage) =
            crate::programs::graph::readout::checked_readout_storage(backend, load, &definition.decoder, limits)?;
        let readout = NativeGraphCharge::serial(readout_storage, slots.readout)?;
        // A prompt chunk's readout slots: none until a group's class grants
        // places (`ResourcePlanner::plan_with_state`).
        let chunk_readout = chunk_storage
            .map(|storage| NativeGraphCharge::serial(storage, GraphSlots { activations: 0, output: 0 }))
            .transpose()?;
        let (head, head_constant_bytes, drafter_graph_host_bytes) =
            match (plan.head(), plan.draft(), state.head_state()) {
                // A head reading target history keeps no state of its own.
                (None, None, None) if plan.shared_head().is_some() => {
                    let shared = plan
                        .shared_head()
                        .ok_or("a shared head program is absent")?;
                    let geometry = crate::programs::native_shared_head::SharedHeadGeometry::new(
                        definition, state, shared,
                    )?;
                    let head_graph =
                        crate::programs::native_shared_head::checked_shared_head_family_storage(
                            backend,
                            load,
                            &geometry,
                            crate::programs::native_shared_head::shared_head_graph_classes(
                                limits,
                                method.draft_rows(),
                            )?,
                        )?;
                    (
                        Some(NativeGraphCharge::serial(head_graph.storage, slots.head)?),
                        head_graph.binding_constant_bytes,
                        head_graph.host_heap.bytes,
                    )
                }
                (None, Some(draft_plan), Some(head_state)) => {
                    let draft = definition
                        .draft
                        .as_ref()
                        .ok_or("a draft program without a draft")?;
                    let geometry =
                        crate::programs::native_draft::DraftGeometry::new(definition, head_state)?;
                    let draft_graph = crate::programs::native_draft::checked_draft_family_storage(
                        backend,
                        load,
                        draft_plan,
                        &geometry,
                        crate::programs::native_draft::draft_graph_classes(
                            limits,
                            method.draft_rows(),
                            &method.block_widths(),
                            draft,
                        )?,
                    )?;
                    (
                        Some(NativeGraphCharge::serial(draft_graph.storage, slots.head)?),
                        draft_graph.binding_constant_bytes,
                        draft_graph.host_heap.bytes,
                    )
                }
                (Some(head), None, Some(head_state)) => {
                    let binding = *head.blocks().first().ok_or("head program has no block")?;
                    let head = definition
                        .head
                        .as_ref()
                        .ok_or("head graph requires a draft head")?;
                    let head_block = crate::operators::CausalHeadBlock::of(
                        head,
                        head.blocks
                            .first()
                            .ok_or("head graph requires a draft head block")?,
                    )
                    .map_err(|error| error.to_string())?;
                    let history = head_state.sole_history()?;
                    let history_rows =
                        u64::try_from(history.rows).map_err(|_| "head history rows exceed u64")?;
                    let head_graph = crate::programs::native_head::checked_head_family_storage(
                        backend,
                        load,
                        &definition.decoder,
                        head_block,
                        binding,
                        crate::programs::native_head::head_graph_classes(
                            limits,
                            history_rows,
                            history.slab_rows,
                            head_state.span_limit(),
                            method.draft_rows(),
                        )?,
                    )?;
                    (
                        Some(NativeGraphCharge::serial(head_graph.storage, slots.head)?),
                        head_graph.binding_constant_bytes,
                        head_graph.host_heap.bytes,
                    )
                }
                (None, None, None) => (None, 0, 0),
                _ => return Err("drafter program and state disagree".into()),
            };
        let (vision, vision_constant_bytes) = match vision {
            Some(VisionGraphBounds { charge, binding_constant_bytes }) => (Some(charge), binding_constant_bytes),
            None => (None, 0),
        };
        let row_classes = magnitude_batching::row_classes(limits.max_launch_rows)
            .into_iter()
            .map(|rows| rows as u64)
            .collect::<Vec<_>>();
        let (state_storage, sealed_state_graphs) =
            crate::programs::native_state::checked_copy_family(
                backend,
                crate::programs::native_state::state_copy_classes(
                    state.target_state(),
                    state.head_state(),
                    &row_classes,
                )?,
            )?;
        let state = NativeGraphCharge::serial(state_storage, slots.state)?;
        let binding_constant_bytes = target
            .binding_constant_bytes
            .checked_add(head_constant_bytes)
            .and_then(|bytes| bytes.checked_add(vision_constant_bytes))
            .ok_or("checked graph constant charge overflow")?;
        let charges = [
            Some(&target.charge),
            Some(&readout),
            head.as_ref(),
            vision.as_ref(),
            Some(&state),
        ]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
        let total_bytes = super::resources::graph_scratch_bytes(&charges)?
            .checked_add(binding_constant_bytes)
            .ok_or("checked graph resource bound overflow")?;
        let recorded_graphs = super::resources::startup_graphs(&charges)?;
        Ok(Self {
            target,
            readout,
            chunk_readout,
            head,
            vision,
            state,
            binding_constant_bytes,
            total_bytes,
            recorded_graphs,
            sealed_state_graphs,
            drafter_graph_host_bytes,
        })
    }
}

impl AssessmentHeaderBounds {
    pub fn derive(load: &ModelLoadPlan) -> Result<Self, String> {
        let import_peak = load.imports().peak_bytes()?;
        let startup_additional_bytes =
            AttestedPrograms::qualification_peak_bytes(load)?.max(import_peak);
        let component_upload_bytes = load
            .head_upload_peak_bytes()?
            .max(load.vision_upload_peak_bytes()?);
        Ok(Self {
            startup_additional_bytes,
            component_upload_bytes,
            staging_upload_bytes: import_peak,
        })
    }

    /// Complete the memory bounds with the graph-pool upper bound, what the
    /// startup graphs hold on a device of `graph_cost`, and the worker's
    /// host and device bytes for the same backend and workload limits. The
    /// device bytes are steady holdings of the allocation domain, part of
    /// the prepared resource bound.
    pub fn with_graph_resource_bound(
        self,
        graph: &AssessmentGraphResourceBounds,
        graph_cost: seismic::GraphCost,
        host_process: HostProcessBytes,
        device_process: DeviceProcessBytes,
    ) -> Result<AssessmentMemoryBounds, String> {
        let recorded = graph_cost
            .bytes(graph.recorded_graphs)
            .ok_or("the startup graphs' memory overflows")?;
        Ok(AssessmentMemoryBounds {
            graph_bytes: graph.total_bytes,
            recorded_graph_bytes: recorded
                .device
                .checked_add(recorded.upload)
                .ok_or("the startup graphs' memory overflows")?,
            startup_additional_bytes: self.startup_additional_bytes,
            component_upload_bytes: self.component_upload_bytes,
            staging_upload_bytes: self.staging_upload_bytes,
            // The target's sealed plans, the prepared graphs that hold
            // them and the family's constant values, and the state copy
            // graphs, which the plan fixes.
            sealed_graph_host_bytes: SealedGraphHostBytes {
                at_most: graph
                    .target
                    .sealed
                    .at_most_bytes()
                    .and_then(|bytes| bytes.checked_add(graph.sealed_state_graphs.bytes))
                    .ok_or("assessment sealed graph bytes overflow")?,
                at_least: graph
                    .target
                    .sealed
                    .at_least_bytes()
                    .and_then(|bytes| bytes.checked_add(graph.sealed_state_graphs.bytes))
                    .ok_or("assessment sealed graph bytes overflow")?,
            },
            bound_graph_host_bytes: graph.target.sealed.bound.bytes,
            drafter_graph_host_bytes: graph.drafter_graph_host_bytes,
            host_process,
            device_process,
        })
    }
}

/// The checkpoints the prefix cache retains of one conversation: its latest
/// prompt one row before its end, the deepest point a repeat of the prompt
/// resumes from; its latest whole path once the reply ends, which the next
/// turn extends; and the branch point where its last turn began, where the
/// next turn resumes when the template renders earlier replies anew. Each
/// pins a recurrent bank; their history rows are the conversation's own. The
/// workload's state is the most the stores hold with cached prefixes: the
/// scheduler releases the least recently used prefix beyond these whenever
/// their occupied state exceeds it.
pub const CONVERSATION_CHECKPOINTS: usize = 3;

/// The depth of each of the workload's conversations: the lesser of the
/// supported context and 100,000 tokens.
pub(crate) fn fit_depth(definition: &ModelDefinition) -> u64 {
    definition.decoder.context_limit.min(100_000)
}

/// The state store's own capacity rule for the workload's conversations
/// (`ResourceLimits::conversations`): each one's accepted bank, the
/// successors it has in flight ([`ResourceLimits::target_successor_banks`])
/// and the checkpoints the prefix cache retains of it, and the permanently
/// pristine zero seed.
pub(super) fn conversation_banks(limits: ResourceLimits) -> Result<u64, String> {
    let conversations = limits.conversations.get();
    let each = |banks: usize| {
        banks
            .checked_mul(conversations)
            .ok_or_else(|| "assessment bank count overflows".to_owned())
    };
    let banks = BankCapacity {
        active: conversations,
        in_flight: each(limits.target_successor_banks())?,
        retained: each(CONVERSATION_CHECKPOINTS)?,
    }
    .storage_total()
    .map_err(|error| error.to_string())?;
    u64::try_from(banks).map_err(|_| "assessment bank count exceeds u64".into())
}

impl AssessmentMemoryTerms {
    /// The assessment workload is the load's conversations
    /// (`ResourceLimits::conversations`, one for a served load), each at the
    /// lesser of the supported context and 100,000 tokens. Method state and optional
    /// components are included only when selected by the planned method; the
    /// conversation's in-flight banks follow the planned lookahead.
    pub fn derive(
        definition: &ModelDefinition,
        load: &ModelLoadPlan,
        selection: ComponentSelection,
        codec: KvCodec,
        method: PlannedMethod,
        limits: ResourceLimits,
    ) -> Result<Self, String> {
        if selection.head != method.drafts() {
            return Err("assessment method and head selection disagree".into());
        }
        if selection.head != load.head().is_some() || selection.vision != load.vision().is_some() {
            return Err("assessment selection disagrees with the load plan".into());
        }
        let demands = load.weight_demands()?;
        let [target, head, vision] = demands;
        let bytes = |demand: ClaimDemand| demand.checked_bytes().ok_or("weight demand overflows");
        let weights = [bytes(target)?, bytes(head)?, bytes(vision)?];
        let drafter = if selection.head {
            crate::operators::draft::drafter_blocks(definition)
        } else {
            Vec::new()
        };
        let layout =
            ModelStateLayout::derive(&definition.decoder, &drafter, codec, method.draft_rows())?;
        let history_per_token = history_row_bytes(&layout.target_history)?
            .checked_add(history_row_bytes(&layout.head_history)?)
            .ok_or("history row bytes overflow")?;
        let recurrent_per_bank = recurrent_bank_bytes(&layout.target_recurrent)?;
        Ok(Self {
            target_weights: weights[0],
            head_weights: weights[1],
            vision_weights: weights[2],
            history_per_token,
            recurrent_per_bank,
            recurrent_banks: conversation_banks(limits)?,
            fit_depth: fit_depth(definition),
            conversations: limits.conversations.get() as u64,
            host_table_bytes: load.host_table_bytes()?,
            weight_demand: ClaimDemand::checked_sum(demands).ok_or("weight demand overflows")?,
        })
    }

    pub fn history_at_fit_depth(self) -> Result<u64, String> {
        self.history_per_token
            .checked_mul(self.fit_depth)
            .and_then(|bytes| bytes.checked_mul(self.conversations))
            .ok_or_else(|| "assessment history bytes overflow".to_owned())
    }

    pub fn recurrent_at_fit_workload(self) -> Result<u64, String> {
        self.recurrent_per_bank
            .checked_mul(self.recurrent_banks)
            .ok_or_else(|| "assessment recurrent bytes overflow".to_owned())
    }

    /// Exact resident bytes of the workload. The state charge includes each
    /// store's address table and whole history and bank slabs from Seismic.
    pub fn exact_resident_bytes(self, state_slab_bytes: u64) -> Result<u64, String> {
        [
            self.target_weights,
            self.head_weights,
            self.vision_weights,
            state_slab_bytes,
        ]
        .into_iter()
        .try_fold(0u64, |total, bytes| {
            total
                .checked_add(bytes)
                .ok_or_else(|| "assessment resident bytes overflow".to_owned())
        })
    }

    /// The charge of each memory role. The allocation domain holds the
    /// weights and every prepared resource, plus the larger of two phases
    /// that never coexist: serving, the workload's state at the fit depth,
    /// which grows only after the load is ready, with the import window of
    /// an optional component a request loads there; and the load itself,
    /// its startup state with the qualification/import peak, then its state
    /// once loaded. A dedicated device's staging domain holds the larger
    /// upload window. A prompt group's class beyond the floor's,
    /// `group_class`, is carried beside them, in none of them.
    pub fn charge(
        self,
        bounds: AssessmentMemoryBounds,
        state_slab_bytes: u64,
        startup_state_bytes: u64,
        loaded_state_bytes: u64,
        group_class: ClaimDemand,
    ) -> Result<AssessmentMemoryCharge, String> {
        AssessmentMemoryCharge::compose(
            self.exact_resident_bytes(0)?,
            self.weight_demand,
            bounds,
            state_slab_bytes,
            startup_state_bytes,
            loaded_state_bytes,
            self.host_table_bytes,
            group_class,
        )
    }
}

impl AssessmentMemoryCharge {
    /// The charge of each memory role ([`AssessmentMemoryTerms::charge`])
    /// from the weights' bytes (`weight_bytes`, which `weight_demand`
    /// places), the bounds and the state's phases: the one owner of how the
    /// roles compose.
    fn compose(
        weight_bytes: u64,
        weight_demand: ClaimDemand,
        bounds: AssessmentMemoryBounds,
        state_slab_bytes: u64,
        startup_state_bytes: u64,
        loaded_state_bytes: u64,
        host_table_bytes: u64,
        group_class: ClaimDemand,
    ) -> Result<Self, String> {
        let exact_resident_bytes = weight_bytes
            .checked_add(state_slab_bytes)
            .ok_or("assessment resident bytes overflow")?;
        let loading_bytes = startup_state_bytes
            .checked_add(bounds.startup_additional_bytes)
            .ok_or("assessment startup charge overflow")?
            .max(loaded_state_bytes);
        let prepared_bytes = weight_bytes
            .checked_add(bounds.prepared_resource_bytes()?)
            .ok_or("assessment prepared charge overflow")?;
        let serving_bytes = prepared_bytes
            .checked_add(state_slab_bytes)
            .ok_or("assessment serving charge overflow")?;
        let serving_peak_bytes = state_slab_bytes
            .checked_add(bounds.component_upload_bytes)
            .ok_or("assessment serving peak overflow")?;
        let allocation_bytes = prepared_bytes
            .checked_add(serving_peak_bytes.max(loading_bytes))
            .ok_or("assessment allocation charge overflow")?;
        Ok(Self {
            exact_resident_bytes,
            state_bytes: state_slab_bytes,
            startup_state_bytes,
            loaded_state_bytes,
            bounds,
            serving_bytes,
            allocation_bytes,
            staging_bytes: bounds
                .staging_upload_bytes
                .max(bounds.component_upload_bytes),
            host_table_bytes,
            weight_demand,
            group_class,
        })
    }

    /// The charge of `load`, a load that differs from this charge's only in
    /// its expert bank's slots: of its weight demand (all of its weights, a
    /// bank's committed records among them) and its header bounds' startup
    /// peak, the two terms a bank's slots move. Every other term is of the
    /// bank's structure alone (`assessment::StructuralCharge`); its upload
    /// windows are, which `load`'s must agree with.
    pub fn at_bank_slots(self, load: &ModelLoadPlan) -> Result<Self, String> {
        let weight_demand = ClaimDemand::checked_sum(load.weight_demands()?).ok_or("weight demand overflows")?;
        let header = AssessmentHeaderBounds::derive(load)?;
        assert_eq!(
            (header.component_upload_bytes, header.staging_upload_bytes),
            (self.bounds.component_upload_bytes, self.bounds.staging_upload_bytes),
            "a bank's slots moved an upload window, which is of the bank's structure"
        );
        Self::compose(
            weight_demand.checked_bytes().ok_or("weight demand overflows")?,
            weight_demand,
            AssessmentMemoryBounds { startup_additional_bytes: header.startup_additional_bytes, ..self.bounds },
            self.state_bytes,
            self.startup_state_bytes,
            self.loaded_state_bytes,
            self.host_table_bytes,
            self.group_class,
        )
    }

    /// The charge with `more` weight storage held beside its own: what
    /// [`AssessmentMemoryTerms::charge`] makes of weights that much
    /// larger, whose bytes every weight-inclusive figure carries. A linked
    /// host tier grown past the floor its bank commits at load is such
    /// storage (`ExecutionPlanDraft::with_host_tier`).
    pub fn with_weights(self, more: ClaimDemand) -> Result<Self, String> {
        let bytes = more.checked_bytes().ok_or("weight demand overflows")?;
        let grown = |figure: u64| figure.checked_add(bytes).ok_or("assessment charge overflows");
        Ok(Self {
            exact_resident_bytes: grown(self.exact_resident_bytes)?,
            serving_bytes: grown(self.serving_bytes)?,
            allocation_bytes: grown(self.allocation_bytes)?,
            weight_demand: self.weight_demand.checked_plus(more).ok_or("weight demand overflows")?,
            ..self
        })
    }

    /// The charge of a load whose prompt groups hold their class of the
    /// plan's chain places beside the rest ([`Self::group_class`]): what a
    /// bank planned beyond its floor is confirmed at, so the bank leaves the
    /// room a group's places take. Its device-side bytes, as
    /// `ExecutionPlanDraft::allocated` reads the weights'.
    pub fn with_group_class(self) -> Result<Self, String> {
        let bytes = self
            .group_class
            .checked_bytes()
            .and_then(|bytes| bytes.checked_sub(self.group_class.host))
            .ok_or("a prompt group's class overflows")?;
        let grown = |figure: u64| figure.checked_add(bytes).ok_or("assessment charge overflows");
        Ok(Self {
            serving_bytes: grown(self.serving_bytes)?,
            allocation_bytes: grown(self.allocation_bytes)?,
            ..self
        })
    }

    /// Everything the load holds in system RAM outside its device
    /// allocations: the host tables, the target graphs' sealed plans (the
    /// most a load seals) and static bindings, the drafter's graphs, and the
    /// worker's own host bytes. All are held together from the binding of the weights on.
    pub fn host_bytes(&self) -> Result<u64, String> {
        self.host_table_bytes
            .checked_add(self.bounds.sealed_graph_host_bytes.at_most)
            .and_then(|bytes| bytes.checked_add(self.bounds.bound_graph_host_bytes))
            .and_then(|bytes| bytes.checked_add(self.bounds.drafter_graph_host_bytes))
            .and_then(|bytes| bytes.checked_add(self.bounds.host_process.bytes().ok()?))
            .ok_or_else(|| "host charge overflows".to_owned())
    }

    /// Whether the charge fits `executor` at stable capacity: its device
    /// storage, upload storage, reserved host backing and its host RAM ([`Self::host_bytes`] and the
    /// staging fit charges there) placed by
    /// [`Fit::demand`], which alone interprets placements, stores, grants
    /// and reserves. Returns each store the demand reached, with what it
    /// required and what of its room remains (negative: its deficit), and
    /// the verdict: fits, or the store of the largest deficit. A store's
    /// room is what fit can place there along the executor's placements
    /// beside the same host demand, so a grant below the store (Metal's
    /// working set, CUDA's usable memory) bounds it; its capacity is that
    /// room with the store's planning reserve.
    pub fn assess_fit(
        self,
        topology: &DeviceTopology,
        executor: &Executor,
        status: &Status,
        reserves: &crate::platform::MemoryReserves,
    ) -> Result<AssessmentFit, String> {
        let fit = Fit::new(topology, status, reserves, HostRoom::Allocatable);
        // The staging window is part of `allocation_bytes`
        // (`startup_additional_bytes`); fit decides whether host RAM also
        // holds it as a demand of its own.
        let host = self
            .host_bytes()?
            .checked_add(fit.host_staging(executor, self.staging_bytes))
            .ok_or("host charge overflow")?;
        let other = self.allocation_bytes.checked_sub(self.weight_demand.checked_bytes()
            .ok_or("weight demand overflows")?).ok_or("weight storage exceeds the allocation charge")?;
        let demand = self.weight_demand.checked_plus(ClaimDemand { device: other, ..ClaimDemand::default() })
            .ok_or("assessment storage demand overflows")?;
        let place = |bytes: [u64; crate::memory::CLAIMED.len()]| {
            let demands: [_; crate::memory::CLAIMED.len()] = std::array::from_fn(|index| StorageDemand {
                executor: executor.id(), kind: crate::memory::CLAIMED[index], bytes: bytes[index],
            });
            fit.demand(&demands, &[(topology.host_ram().id(), host)], FitMode::Stable)
                .map_err(|error| error.to_string())
        };
        let mut bytes = [0; { crate::memory::CLAIMED.len() }];
        for (index, kind) in crate::memory::CLAIMED.into_iter().enumerate() {
            bytes[index] = demand.storage_bytes(kind).ok_or("assessment storage demand overflows")?;
        }
        let result = place(bytes)?;
        // Each store's room: what all physical placements can place there.
        let rooms = place(crate::memory::CLAIMED.map(|kind| if executor.paths(seismic::Placing::Fixed(kind)).is_empty() { 0 } else { u64::MAX }))?.placed;
        let mut domains = Vec::new();
        for (store, placed) in result.placed {
            let reserve = fit
                .store(store)
                .map_err(|error| error.to_string())?
                .thresholds
                .planning_bytes;
            let room = rooms
                .iter()
                .find(|(of, _)| *of == store)
                .map(|(_, bytes)| *bytes)
                .expect("a store the demand reached is one its placements reach");
            let missing = result
                .refused
                .iter()
                .filter(|refusal| refusal.store == store)
                .map(|refusal| refusal.missing_bytes)
                .sum::<u64>();
            let required_bytes = placed.checked_add(missing).ok_or("store demand overflow")?;
            let remaining_bytes = if missing == 0 {
                i64::try_from(room.saturating_sub(placed))
                    .map_err(|_| "fit remainder exceeds i64")?
            } else {
                -i64::try_from(missing).map_err(|_| "fit deficit exceeds i64")?
            };
            domains.push(DomainFit {
                role: DomainRole::of(topology, executor, store),
                domain: store,
                capacity_bytes: room.saturating_add(reserve),
                required_bytes,
                reserve_bytes: reserve,
                remaining_bytes,
            });
        }
        // A domain's deficit: none where it has room.
        let verdict = domains
            .iter()
            .filter_map(|domain| Some((domain.domain, NonZeroU64::new(domain.remaining_bytes.min(0).unsigned_abs())?)))
            .max_by_key(|&(_, deficit_bytes)| deficit_bytes)
            .map_or(AssessmentFitVerdict::Fits, |(limiting, deficit_bytes)| {
                AssessmentFitVerdict::DoesNotFit {
                    limiting,
                    deficit_bytes,
                }
            });
        Ok(AssessmentFit { verdict, domains })
    }
}

#[cfg(test)]
mod tests;
