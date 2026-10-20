//! Model-derived memory charges for metadata-only assessment.
//!
//! The standard assessment workload is one conversation at the fit depth
//! `min(context_limit, 100_000)` on a clean load. Its charge is composed of
//! exact header terms (resident weights, history at the fit depth, recurrent
//! banks, native invocation storage) and upper bounds taken from the same
//! checked class enumeration and slot multipliers production preparation
//! uses (graph pools and bound constants, the startup qualification/import
//! peak). The load planner and the state layout remain the authorities for
//! physical representations and codec planes; the platform policy owns each
//! domain's stable capacity and reserve.

use super::resources::{NativeGraphCharge, history_row_bytes, recurrent_bank_bytes};
use super::{
    ComponentSelection, ModelLoadPlan, PlannedMethod, ResourceLimits, source_import_peak_bytes,
    weight_bytes_by_component,
};
use crate::assessment::DomainFit;
use crate::platform::{DomainRole, Fit, FitMode, StorageDemand};
use crate::{AttestedPrograms, GraphError};
use magnitude_family_contracts::ModelDefinition;
use magnitude_state::{BankCapacity, KvCodec, ModelStateLayout};
use seismic::BackendName;
use seismic::{DeviceTopology, Executor, Status, StorageKind, StoreId};

/// Exact resident and per-state terms of the standard workload, derived
/// without opening a device or reading tensor payloads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AssessmentMemoryTerms {
    pub target_weights: u64,
    pub head_weights: u64,
    pub vision_weights: u64,
    pub history_per_token: u64,
    pub recurrent_per_bank: u64,
    /// Recurrent banks one conversation holds: its accepted bank, every
    /// in-flight successor the planned lookahead keeps, the checkpoints the
    /// prefix cache retains of it, and the pristine seed.
    pub recurrent_banks: u64,
    pub fit_depth: u64,
    /// Host-resident gathered tables (model-family plan §3.7): claims in the
    /// system-RAM domain, never device weights.
    pub host_table_bytes: u64,
    /// The target's stored tables' host-shared slots and directories
    /// (`ModelLoadPlan::shared_table_bytes`), part of `target_weights`.
    pub shared_table_bytes: u64,
}

/// Upper bounds for every nonresident charge of a clean load. The prepared
/// resource bound covers all steady native holdings (invocation storage,
/// graph workspace, output, upload regions, bound constants, recorded graphs
/// and the worker's device bytes); the startup
/// bound covers the additional qualification/import peak; the component
/// bound is the import peak of an optional component a served request
/// loads; the staging bound is the host upload peak of a dedicated device's
/// imports.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AssessmentMemoryBounds {
    pub prepared_resource_bytes: u64,
    pub startup_additional_bytes: u64,
    pub component_upload_bytes: u64,
    pub staging_upload_bytes: u64,
    /// Host heap of the sealed target graphs
    /// (`AssessmentGraphResourceBounds::sealed_target_graphs`): system RAM
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
    /// Measured by `inference/scripts/device-memory-costs.py` (Linux) and its
    /// Windows equivalent on an A10G, an L40S and an A10 (CUDA) and a Radeon
    /// PRO V710 (Vulkan, Linux and Windows).
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
/// a measurement of real loads on that backend (`design/inference/engine/memory.md`,
/// "Host memory of a load"), not a margin: what a worker holds beyond its
/// planned holdings, apportioned to the counts that drive it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct HostProcessCosts {
    runtime: u64,
    per_token: u64,
    per_kernel: u64,
    vision: u64,
}

impl HostProcessCosts {
    /// Each backend's costs are measured from its loads: the process
    /// footprint at load end less everything the plan holds then is the
    /// worker's own memory, and the costs are the least-overestimating ones
    /// under which no measured load exceeds its plan.
    ///
    /// Metal: an M4 Max (macOS 15.5), from MiniCPM5 1B, Qwen3.5 0.8B, 2B and
    /// 4B, LFM2.5 1.2B and Gemma 4 E2B. Its vision cost is what Qwen3.5 2B's
    /// and Gemma 4 E2B's workers held after their first image beyond
    /// everything else their plans hold, the two within 2 %.
    ///
    /// CPU: the same machine's CPU backend, from MiniCPM5 1B, Qwen3.5 0.8B
    /// and 2B, LFM2.5 1.2B and Gemma 4 E2B, each loaded twice (identical
    /// loads differ by up to 0.16 GB). Linux CPU workers held the same at
    /// load end (LFM2.5 1.2B: 1.30 GB on both). Its vision is not measured
    /// and takes Metal's.
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

const METAL_RUNTIME: u64 = 175_000_000;
const METAL_PER_TOKEN: u64 = 1_800;
const METAL_PER_KERNEL: u64 = 5_700_000;
const METAL_VISION: u64 = 210_000_000;
const CPU_RUNTIME: u64 = 205_000_000;
const CPU_PER_TOKEN: u64 = 1_400;
const CPU_PER_KERNEL: u64 = 5_300_000;

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
            programs: costs
                .per_kernel
                .checked_mul(kernels)
                .ok_or("prepared program host bytes overflow")?,
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

/// Header-derived charges that are known before graph formation. Prepared
/// native invocation storage is exact; the startup charge is the same
/// qualification/import upper bound used by the production resource planner;
/// the component charge is the import window an optional head or vision
/// component claims beside its weights when a request first needs it
/// (`DomainState::component_growth`); the staging charge is the largest
/// selected source tensor's import window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AssessmentHeaderBounds {
    pub prepared_program_bytes: u64,
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
    /// The state a load allocates at startup, which the loaded model holds
    /// before its first request.
    pub startup_state_bytes: u64,
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
    /// The stored tables' host-shared bytes, part of `allocation_bytes` and
    /// `serving_bytes`: upload storage, held where the executor's upload
    /// placement lands (`assess_fit`), host RAM for CUDA's pinned memory.
    pub shared_table_bytes: u64,
}

/// A complete fit result: every domain the load touches, and whether the
/// workload fits all of them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AssessmentFit {
    pub verdict: AssessmentFitVerdict,
    pub domains: Vec<DomainFit>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AssessmentFitVerdict {
    Fits,
    /// The domain with the largest deficit limits the load.
    DoesNotFit {
        limiting: StoreId,
        deficit_bytes: u64,
    },
}

/// Checked graph-pool demand and distinct bound constants from the same class
/// enumeration and slot multipliers as native preparation: an upper bound for
/// every graph pool a clean load commits on the selected backend. Deriving
/// it builds every graph a load prepares, so it fails with
/// `GraphError::KernelDomain` exactly when the model's program calls a kernel
/// outside its domain on the backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AssessmentGraphResourceBounds {
    pub target: NativeGraphCharge,
    pub readout: NativeGraphCharge,
    pub head: Option<NativeGraphCharge>,
    pub vision: Option<NativeGraphCharge>,
    pub state: NativeGraphCharge,
    pub binding_constant_bytes: u64,
    pub total_bytes: u64,
    /// The most device memory the backend's recorded graphs hold beyond
    /// every allocation (`seismic::recorded_graph_device_bytes`).
    pub recorded_graph_bytes: u64,
    /// The target graphs a load seals and the host heap they hold, from
    /// the sealer's own enumeration: one count, or bounds from both sides
    /// where blocks that read a layer scale could share a plan
    /// (`PlannedSealedGraphs::exact`). Host memory, not part of
    /// `total_bytes`.
    pub sealed_target_graphs: crate::PlannedSealedGraphs,
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
    pub fn derive(
        definition: &ModelDefinition,
        load: &ModelLoadPlan,
        state: &super::StateResourcePlan,
        method: PlannedMethod,
        codec: KvCodec,
        limits: super::ResourceLimits,
        backend: BackendName,
    ) -> Result<Self, GraphError> {
        let plan = load
            .program_plan(definition, codec)
            .map_err(|error| error.to_string())?;
        if plan.drafts_with_head() != matches!(method, PlannedMethod::Mtp { .. })
            || plan.draft().is_some() != matches!(method, PlannedMethod::DFlash { .. })
        {
            return Err("assessment method and drafter program disagree".into());
        }
        // The same startup slots a load commits: one request's need.
        let slots = limits.startup_slots();
        let (target_graph, sealed_target_graphs) =
            crate::programs::native_target_graph::checked_target_family(
                backend,
                load,
                &definition.decoder,
                state,
                plan.target(),
                limits,
            )?;
        let target = NativeGraphCharge::from_checked(
            target_graph.storage,
            crate::programs::native_target_graph::target_runs_per_step(&definition.decoder),
            slots.target,
        )?;
        let readout = NativeGraphCharge::from_checked(
            crate::programs::graph::readout::checked_readout_family_storage(
                backend,
                load,
                &definition.decoder,
                limits,
            )?,
            1,
            slots.readout,
        )?;
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
                        Some(NativeGraphCharge::from_checked(
                            head_graph.storage,
                            1,
                            slots.head,
                        )?),
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
                        Some(NativeGraphCharge::from_checked(
                            draft_graph.storage,
                            1,
                            slots.head,
                        )?),
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
                        Some(NativeGraphCharge::from_checked(
                            head_graph.storage,
                            1,
                            slots.head,
                        )?),
                        head_graph.binding_constant_bytes,
                        head_graph.host_heap.bytes,
                    )
                }
                (None, None, None) => (None, 0, 0),
                _ => return Err("drafter program and state disagree".into()),
            };
        let (vision, vision_constant_bytes) = match (plan.vision(), definition.vision.as_ref()) {
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
                (
                    Some(NativeGraphCharge::from_checked(
                        vision_graph.storage,
                        1,
                        slots.vision,
                    )?),
                    vision_graph.binding_constant_bytes,
                )
            }
            (None, None) => (None, 0),
            _ => return Err("vision program and definition disagree".into()),
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
        let state = NativeGraphCharge::from_checked(state_storage, 1, slots.state)?;
        let binding_constant_bytes = target_graph
            .binding_constant_bytes
            .checked_add(head_constant_bytes)
            .and_then(|bytes| bytes.checked_add(vision_constant_bytes))
            .ok_or("checked graph constant charge overflow")?;
        let total_bytes = super::resources::graph_scratch_bytes(
            &[
                Some(&target),
                Some(&readout),
                head.as_ref(),
                vision.as_ref(),
                Some(&state),
            ]
            .into_iter()
            .flatten()
            .collect::<Vec<_>>(),
        )?
        .checked_add(binding_constant_bytes)
        .ok_or("checked graph resource bound overflow")?;
        Ok(Self {
            target,
            readout,
            head,
            vision,
            state,
            binding_constant_bytes,
            total_bytes,
            recorded_graph_bytes: seismic::recorded_graph_device_bytes(backend),
            sealed_target_graphs,
            sealed_state_graphs,
            drafter_graph_host_bytes,
        })
    }
}

impl AssessmentHeaderBounds {
    pub fn derive(
        definition: &ModelDefinition,
        load: &ModelLoadPlan,
        codec: KvCodec,
        backend: BackendName,
    ) -> Result<Self, String> {
        let programs = load
            .program_plan(definition, codec)
            .map_err(|error| error.to_string())?;
        let prepared_program_bytes =
            AttestedPrograms::planned_invocation_workspace_bytes(&programs, backend)
                .map_err(|error| error.to_string())?;
        let largest_source = load
            .weights()
            .map(|weight| weight.source_bytes)
            .max()
            .unwrap_or(0);
        let import_peak = source_import_peak_bytes(largest_source)?;
        let startup_additional_bytes =
            AttestedPrograms::qualification_peak_bytes(load)?.max(import_peak);
        let component_upload_bytes = load
            .head_upload_peak_bytes()?
            .max(load.vision_upload_peak_bytes()?);
        Ok(Self {
            prepared_program_bytes,
            startup_additional_bytes,
            component_upload_bytes,
            staging_upload_bytes: import_peak,
        })
    }

    /// Complete the memory bounds with the graph-pool upper bound and the
    /// worker's host and device bytes for the same backend and workload
    /// limits. The device bytes are steady holdings of the allocation domain,
    /// part of the prepared resource bound.
    pub fn with_graph_resource_bound(
        self,
        graph: &AssessmentGraphResourceBounds,
        host_process: HostProcessBytes,
        device_process: DeviceProcessBytes,
    ) -> Result<AssessmentMemoryBounds, String> {
        Ok(AssessmentMemoryBounds {
            prepared_resource_bytes: self
                .prepared_program_bytes
                .checked_add(graph.total_bytes)
                .and_then(|bytes| bytes.checked_add(graph.recorded_graph_bytes))
                .and_then(|bytes| bytes.checked_add(device_process.bytes().ok()?))
                .ok_or("assessment prepared resource bound overflow")?,
            startup_additional_bytes: self.startup_additional_bytes,
            component_upload_bytes: self.component_upload_bytes,
            staging_upload_bytes: self.staging_upload_bytes,
            // The target's sealed plans, the prepared graphs that hold
            // them and the family's constant values, and the state copy
            // graphs, which the plan fixes.
            sealed_graph_host_bytes: SealedGraphHostBytes {
                at_most: graph
                    .sealed_target_graphs
                    .at_most_bytes()
                    .and_then(|bytes| bytes.checked_add(graph.sealed_state_graphs.bytes))
                    .ok_or("assessment sealed graph bytes overflow")?,
                at_least: graph
                    .sealed_target_graphs
                    .at_least_bytes()
                    .and_then(|bytes| bytes.checked_add(graph.sealed_state_graphs.bytes))
                    .ok_or("assessment sealed graph bytes overflow")?,
            },
            bound_graph_host_bytes: graph.sealed_target_graphs.bound.bytes,
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

/// The depth of the standard workload's one conversation: the lesser of the
/// supported context and 100,000 tokens.
pub(super) fn fit_depth(definition: &ModelDefinition) -> u64 {
    definition.decoder.context_limit.min(100_000)
}

/// The state store's own capacity rule for the standard workload's one
/// conversation: its accepted bank, its in-flight step (and that step's
/// queued successor under lookahead), the checkpoints the prefix cache
/// retains of it, and the permanently pristine zero seed.
pub(super) fn conversation_banks(lookahead: bool) -> Result<u64, String> {
    let banks = BankCapacity {
        active: 1,
        in_flight: 1 + usize::from(lookahead),
        retained: CONVERSATION_CHECKPOINTS,
    }
    .storage_total()
    .map_err(|error| error.to_string())?;
    u64::try_from(banks).map_err(|_| "assessment bank count exceeds u64".into())
}

impl AssessmentMemoryTerms {
    /// The standard assessment workload is one conversation at the lesser
    /// of the supported context and 100,000 tokens. Method state and optional
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
        let weights = weight_bytes_by_component(load)?;
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
            recurrent_banks: conversation_banks(limits.lookahead)?,
            fit_depth: fit_depth(definition),
            host_table_bytes: load.host_table_bytes()?,
            shared_table_bytes: load.shared_table_bytes()?,
        })
    }

    pub fn history_at_fit_depth(self) -> Result<u64, String> {
        self.history_per_token
            .checked_mul(self.fit_depth)
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
    /// its startup state with the qualification/import peak. A dedicated
    /// device's staging domain holds the larger upload window.
    pub fn charge(
        self,
        bounds: AssessmentMemoryBounds,
        state_slab_bytes: u64,
        startup_state_bytes: u64,
    ) -> Result<AssessmentMemoryCharge, String> {
        let exact_resident_bytes = self.exact_resident_bytes(state_slab_bytes)?;
        let loading_bytes = startup_state_bytes
            .checked_add(bounds.startup_additional_bytes)
            .ok_or("assessment startup charge overflow")?;
        let prepared_bytes = self
            .exact_resident_bytes(0)?
            .checked_add(bounds.prepared_resource_bytes)
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
        Ok(AssessmentMemoryCharge {
            exact_resident_bytes,
            state_bytes: state_slab_bytes,
            startup_state_bytes,
            bounds,
            serving_bytes,
            allocation_bytes,
            staging_bytes: bounds
                .staging_upload_bytes
                .max(bounds.component_upload_bytes),
            host_table_bytes: self.host_table_bytes,
            shared_table_bytes: self.shared_table_bytes,
        })
    }
}

impl AssessmentMemoryCharge {
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
    /// storage (the allocation less the shared tables), its upload storage
    /// (the shared tables) and its host RAM ([`Self::host_bytes`] and the
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
        let fit = Fit::new(topology, status, reserves);
        // The staging window is part of `allocation_bytes`
        // (`startup_additional_bytes`); fit decides whether host RAM also
        // holds it as a demand of its own.
        let host = self
            .host_bytes()?
            .checked_add(fit.host_staging(executor, self.staging_bytes))
            .ok_or("host charge overflow")?;
        let device = self
            .allocation_bytes
            .checked_sub(self.shared_table_bytes)
            .ok_or("shared storage exceeds the allocation charge")?;
        let place = |device, upload| {
            fit.demand(
                &[
                    StorageDemand {
                        executor: executor.id(),
                        kind: StorageKind::Device,
                        bytes: device,
                    },
                    StorageDemand {
                        executor: executor.id(),
                        kind: StorageKind::Upload,
                        bytes: upload,
                    },
                ],
                &[(topology.host_ram().id(), host)],
                FitMode::Stable,
            )
            .map_err(|error| error.to_string())
        };
        let result = place(device, self.shared_table_bytes)?;
        // Each store's room: what the placements can place there at most.
        let rooms = place(u64::MAX, u64::MAX)?.placed;
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
        let verdict = domains
            .iter()
            .filter(|domain| domain.remaining_bytes < 0)
            .max_by_key(|domain| domain.remaining_bytes.unsigned_abs())
            .map_or(AssessmentFitVerdict::Fits, |domain| {
                AssessmentFitVerdict::DoesNotFit {
                    limiting: domain.domain,
                    deficit_bytes: domain.remaining_bytes.unsigned_abs(),
                }
            });
        Ok(AssessmentFit { verdict, domains })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use magnitude_kernels::dense_output;
    use seismic::fixtures::recorded;
    use seismic::{BackendName, Element, Layout, NativeGraphMetadata};

    #[test]
    fn checked_native_declarations_are_inspectable_without_a_device() {
        let implementation = seismic::generated::native_implementation_for_backend::<
            magnitude_kernels::dense_output::Entry,
        >(BackendName::Cpu)
        .unwrap();
        assert!(implementation.is_some());
        let invalid = seismic::generated::checked_native_binding::<dense_output::Entry>(
            BackendName::Cpu,
            &[
                ("DW", Element::f32()),
                ("A", Element::f32()),
                ("BOGUS", Element::f32()),
            ],
        )
        .unwrap();
        assert!(matches!(
            invalid,
            seismic::NativeBindingCheck::Unsupported(_)
        ));

        let weight = seismic::generated::checked_native_tensor_parameter::<dense_output::Entry>(
            BackendName::Cpu,
            &[("DW", Element::f32()), ("A", Element::f32())],
            "down_weight",
            &[
                ("M", 4),
                ("O", 2),
                ("H", 16),
                ("F", 32),
                ("DS", 0),
                ("DB", 0),
            ],
        )
        .unwrap();
        assert_eq!(
            weight,
            seismic::NativeTensorParameterCheck::Checked(seismic::NativeTensorMetadata {
                element: Element::f32(),
                extents: vec![16, 32],
                canonical_bytes: 16 * 32 * 4,
            })
        );
        let results = seismic::generated::checked_native_tensor_results::<dense_output::Entry>(
            BackendName::Cpu,
            &[("DW", Element::f32()), ("A", Element::f32())],
            &[
                ("M", 4),
                ("O", 2),
                ("H", 16),
                ("F", 32),
                ("DS", 0),
                ("DB", 0),
            ],
        )
        .unwrap();
        assert_eq!(
            results,
            seismic::NativeTensorResultsCheck::Checked(vec![Some(seismic::NativeTensorMetadata {
                element: Element::f32(),
                extents: vec![2, 16],
                canonical_bytes: 2 * 16 * 4,
            })])
        );
        let call = seismic::generated::checked_native_graph_call::<dense_output::Entry>(
            BackendName::Cpu,
            &[("DW", Element::f32()), ("A", Element::f32())],
            &[
                ("M", 4),
                ("O", 2),
                ("H", 16),
                ("F", 32),
                ("DS", 0),
                ("DB", 0),
            ],
        )
        .unwrap();
        let seismic::generated::CheckedNativeGraphCall::Checked {
            parameters,
            results: call_results,
            ..
        } = call
        else {
            panic!("checked dense output call is supported on CPU");
        };
        let seismic::NativeTensorParameterCheck::Checked(weight) = weight else {
            unreachable!()
        };
        assert!(parameters.contains(&Some(weight)));
        assert_eq!(
            seismic::NativeTensorResultsCheck::Checked(call_results),
            results
        );
    }

    #[test]
    fn checked_embedding_graph_distinguishes_upload_from_selected_input() {
        let definition = crate::planning::tests::fixture_definition();
        let manifest = crate::planning::tests::fixture_manifest(&definition);
        let load = ModelLoadPlan::derive(
            &manifest,
            &definition,
            ComponentSelection {
                head: false,
                vision: false,
            },
            Layout::Rows16,
        )
        .unwrap();
        let uploaded = crate::programs::native_target_graph::checked_entry_graph_storage(
            BackendName::Cpu,
            &load,
            &definition.decoder,
            2,
            true,
        )
        .unwrap();
        let selected = crate::programs::native_target_graph::checked_entry_graph_storage(
            BackendName::Cpu,
            &load,
            &definition.decoder,
            2,
            false,
        )
        .unwrap();
        assert_eq!(uploaded.workspace, selected.workspace);
        assert_eq!(uploaded.output, selected.output);
        assert_eq!(uploaded.upload, 2 * 2 * 4);
        assert_eq!(selected.upload, 0);
        assert!(uploaded.workspace > 0);
        assert!(uploaded.output > 0);

        let elements = [("EW", Element::f32()), ("A", Element::f32())];
        let dimensions = [
            ("M", 2),
            ("V", definition.decoder.vocabulary),
            ("D", definition.decoder.hidden),
            ("R", 0),
            ("F", 1),
        ];
        let mut invalid = NativeGraphMetadata::new(BackendName::Cpu);
        let wrong_table = invalid.port(Element::f32(), &[1, 1]).unwrap();
        let tokens = invalid
            .input_for::<magnitude_kernels::embedding_rows::Entry>(&elements, "tokens", &dimensions)
            .unwrap();
        let direct = invalid.port(Element::i32(), &[0, 1]).unwrap();
        let mismatch = invalid.enqueue::<magnitude_kernels::embedding_rows::Entry>(
            &elements,
            &dimensions,
            magnitude_kernels::embedding_rows::WorkflowArgs {
                table: wrong_table.tensor().into(),
                tokens: tokens.tensor().into(),
                directory: direct.tensor().into(),
                scale: 1.0,
                normalize: 0,
                epsilon: 0.0,
            },
        );
        assert!(matches!(
            mismatch,
            Err(seismic::NativeGraphMetadataError::Call(
                seismic::CallError::Workflow(seismic::WorkflowError::NativeGraphArgumentMismatch {
                    parameter: 0
                })
            ))
        ));
    }

    #[test]
    fn checked_decoder_family_bounds_every_prepared_class_axis() {
        let definition = crate::planning::tests::fixture_definition();
        let manifest = crate::planning::tests::fixture_manifest(&definition);
        let load = ModelLoadPlan::derive(
            &manifest,
            &definition,
            ComponentSelection {
                head: false,
                vision: false,
            },
            Layout::Rows16,
        )
        .unwrap();
        let limits = crate::ResourceLimits {
            max_launch_rows: 2,
            max_launch_slots: 2,
            max_selected_rows: 2,
            max_drafting_slots: 2,
            exported_logits_rows: 0,
            max_images_per_request: 1,
            max_image_cells: 0,
            lookahead: false,
        };
        let state = crate::ResourcePlanner::state_plan(
            &definition,
            &load,
            PlannedMethod::Plain,
            KvCodec::Dense,
            limits,
            crate::ResourceCapacity {
                domain_bytes: 512 * 1024 * 1024,
                tensor_operations: crate::TensorOperations::Absent,
            },
        )
        .unwrap();
        let plan = load.program_plan(&definition, KvCodec::Dense).unwrap();
        let family = crate::programs::native_target_graph::checked_target_family_storage(
            BackendName::Cpu,
            &load,
            &definition.decoder,
            &state,
            plan.target(),
            limits,
        )
        .unwrap();
        let (block, _) = crate::programs::native_target_graph::checked_block_graph_resources(
            BackendName::Cpu,
            &load,
            &definition.decoder,
            &state,
            plan.target().blocks()[0],
            0,
            2,
            4,
            2,
            0,
            false,
        )
        .unwrap();
        assert!(family.storage.workspace >= block.workspace);
        assert!(family.storage.output >= block.output);
        assert!(family.storage.upload >= block.upload);
        assert!(family.storage.workspace > 0);
        assert!(family.storage.output > 0);
        let pools = AssessmentGraphResourceBounds::derive(
            &definition,
            &load,
            &state,
            PlannedMethod::Plain,
            KvCodec::Dense,
            limits,
            BackendName::Cpu,
        )
        .unwrap();
        assert_eq!(pools.target.workspace_bytes, family.storage.workspace);
        assert_eq!(pools.target.output_bytes, family.storage.output);
        assert_eq!(pools.target.upload_bytes, family.storage.upload);
        assert_eq!(pools.binding_constant_bytes, family.binding_constant_bytes);
        assert!(pools.binding_constant_bytes > 0);
        assert_eq!(
            pools.target.upload_regions,
            1 + definition.decoder.blocks.len()
        );
        assert!(pools.total_bytes >= pools.target.committed_bytes);
    }

    #[test]
    fn checked_head_family_includes_chained_and_shaped_classes() {
        use magnitude_artifacts::gguf::{Encoding, TensorDescriptor};
        use magnitude_family_contracts::{
            ActivationFunction, ExitNorm, ExpertSelection, FeedForwardUp, Head, HeadBlock,
            HeadChain, HeadComponent, HeadHistory, HeadInput, HeadReadout, InputNorm, Operator,
            RmsNorm, RouteNormalization, RoutedFfn, Router, RouterInput, ScoreFunction,
            SharedExpert, SharedExpertGate, VocabularyProjection, WeightDescriptor,
        };

        let mut definition = crate::planning::tests::fixture_definition();
        let descriptor =
            |name: &str, shape: &[u64]| WeightDescriptor::stored(format!("head_{name}"), shape);
        let rms = |name: &str| RmsNorm {
            weight: descriptor(name, &[128]),
            epsilon: 1e-6,
        };
        // The head block is the target block (attention, dense) under head
        // names, between its own norms and combine.
        let mut head_layer = definition.decoder.blocks[0].clone();
        let rename = |weight: &mut WeightDescriptor| weight.name = format!("head_{}", weight.name);
        for sublayer in &mut head_layer.sublayers {
            if let InputNorm::Rms(norm) = &mut sublayer.input {
                rename(&mut norm.weight);
            }
            match &mut sublayer.op {
                Operator::Attention(attention) => {
                    rename(&mut attention.query);
                    rename(&mut attention.output);
                    if let magnitude_family_contracts::HeadNorm::Rms(norm) =
                        &mut attention.query_norm
                    {
                        rename(&mut norm.weight);
                    }
                    if let magnitude_family_contracts::KeyValue::Owned {
                        key,
                        value: magnitude_family_contracts::ValueSource::Projected(value),
                        key_norm: magnitude_family_contracts::HeadNorm::Rms(key_norm),
                        ..
                    } = &mut attention.key_value
                    {
                        rename(key);
                        rename(value);
                        rename(&mut key_norm.weight);
                    }
                }
                Operator::DenseFfn(dense) => {
                    if let FeedForwardUp::Gated { gate, up, .. } = &mut dense.up {
                        rename(gate);
                        rename(up);
                    }
                    rename(&mut dense.down);
                }
                _ => unreachable!(),
            }
        }
        definition.head = Some(Head {
            blocks: vec![HeadBlock {
                residual: magnitude_family_contracts::ResidualForm::Single,
                input: HeadInput::Combine {
                    embedding_norm: rms("embedding_norm"),
                    hidden_norm: rms("hidden_norm"),
                    combine: descriptor("combine", &[128, 256]),
                },
                blocks: vec![head_layer],
                output_norm: ExitNorm::Rms(rms("output_norm")),
                readout: HeadReadout::full(VocabularyProjection::Target),
                chain: HeadChain::Output,
            }],
            history: HeadHistory::Own,
            component: HeadComponent::Target,
        });
        let mut manifest = crate::planning::tests::fixture_manifest(&definition);
        let mut offset = manifest.target.files[0].size;
        let load = ModelLoadPlan::derive(
            &manifest,
            &definition,
            ComponentSelection {
                head: true,
                vision: false,
            },
            Layout::Rows16,
        )
        .unwrap();
        let method = PlannedMethod::Mtp {
            greedy_proposals: 1,
            sampled_proposals: 1,
        };
        let limits = crate::ResourceLimits {
            max_launch_rows: 256,
            max_launch_slots: 16,
            max_selected_rows: 16,
            max_drafting_slots: 16,
            exported_logits_rows: 0,
            max_images_per_request: 1,
            max_image_cells: 0,
            lookahead: false,
        };
        let state = crate::ResourcePlanner::state_plan(
            &definition,
            &load,
            method,
            KvCodec::Dense,
            limits,
            crate::ResourceCapacity {
                domain_bytes: 512 * 1024 * 1024,
                tensor_operations: crate::TensorOperations::Absent,
            },
        )
        .unwrap();
        let plan = load.program_plan(&definition, KvCodec::Dense).unwrap();
        let head = definition.head.clone().unwrap();
        let head_block = crate::operators::CausalHeadBlock::of(&head, &head.blocks[0]).unwrap();
        let head_state = state.head_state().unwrap();
        let head_history = head_state.sole_history().unwrap();
        let classes = crate::programs::native_head::head_graph_classes(
            limits,
            head_history.rows as u64,
            head_history.slab_rows,
            head_state.span_limit(),
            method.draft_rows(),
        )
        .unwrap();
        assert!(classes.iter().any(|class| class.steps == 1 && class.shaped));
        assert!(
            classes
                .iter()
                .any(|class| class.entry_rows > crate::operators::routed::fused_graph::DECODE_ROWS)
        );
        let family = crate::programs::native_head::checked_head_family_storage(
            BackendName::Cpu,
            &load,
            &definition.decoder,
            head_block,
            plan.head().unwrap().blocks()[0],
            classes.clone(),
        )
        .unwrap();
        crate::programs::native_head::verify_head_family_certificates(
            BackendName::Cpu,
            &load,
            &definition.decoder,
            head_block,
            plan.head().unwrap().blocks()[0],
            &classes,
        )
        .unwrap();
        assert!(family.storage.workspace > 0);
        assert!(family.storage.output > 0);
        let pools = AssessmentGraphResourceBounds::derive(
            &definition,
            &load,
            &state,
            method,
            KvCodec::Dense,
            limits,
            BackendName::Cpu,
        )
        .unwrap();
        assert_eq!(
            pools.head.unwrap().workspace_bytes,
            family.storage.workspace
        );
        let target = crate::programs::native_target_graph::checked_target_family_storage(
            BackendName::Cpu,
            &load,
            &definition.decoder,
            &state,
            plan.target(),
            limits,
        )
        .unwrap();
        assert_eq!(
            pools.binding_constant_bytes,
            target.binding_constant_bytes + family.binding_constant_bytes
        );

        // Candidate views do not change resource metadata. This fixture
        // explicitly finalizes a selected-row class before forming both recipes.
        let output = manifest
            .target
            .tensors
            .iter_mut()
            .find(|tensor| tensor.name == definition.decoder.exit.output.name)
            .unwrap();
        let packed = (output.encoding, output.nbytes);
        output.encoding = Encoding::Q8_0;
        output.nbytes = output.shape.iter().product::<u64>() / 32 * 34;
        for backend in [BackendName::Metal, BackendName::Cuda] {
            let load = ModelLoadPlan::derive(
                &manifest,
                &definition,
                ComponentSelection {
                    head: true,
                    vision: false,
                },
                crate::planning::resident_layout(crate::ExecutionPath::Native, backend),
            )
            .unwrap()
            .with_readout_candidates(&definition)
            .unwrap();
            let candidate_plan = load.program_plan(&definition, KvCodec::Dense).unwrap();
            assert!(matches!(
                candidate_plan.head().unwrap().blocks()[0].projection,
                crate::ReadoutHead::Packed { .. }
            ));
            let load = load
                .finalize_readout(&[crate::ReadoutClassChoice {
                    scope: magnitude_family_contracts::WeightScope::Target,
                    norm_scope: magnitude_family_contracts::WeightScope::HeadBlock(0),
                    prepared: true,
                    rows: vec![1],
                }])
                .unwrap();
            let plan = load.program_plan(&definition, KvCodec::Dense).unwrap();
            let binding = plan.head().unwrap().blocks()[0];
            assert!(matches!(
                binding.projection,
                crate::ReadoutHead::Qualified { .. }
            ));
            // The head's packed readers bind the unplaced target
            // projection, and the assessment lists their kernels over it.
            HostProcessBytes::derive(&definition, &load, &plan, limits, backend).unwrap();
            let classes = crate::programs::native_head::head_graph_classes(
                limits,
                head_history.rows as u64,
                head_history.slab_rows,
                head_state.span_limit(),
                method.draft_rows(),
            )
            .unwrap();
            let family = crate::programs::native_head::checked_head_family_storage(
                backend,
                &load,
                &definition.decoder,
                head_block,
                binding,
                classes.clone(),
            )
            .unwrap();
            assert!(family.storage.workspace > 0, "{backend:?}");
            crate::programs::native_head::verify_head_family_certificates(
                backend,
                &load,
                &definition.decoder,
                head_block,
                binding,
                &classes,
            )
            .unwrap();
        }
        let output = manifest
            .target
            .tensors
            .iter_mut()
            .find(|tensor| tensor.name == definition.decoder.exit.output.name)
            .unwrap();
        (output.encoding, output.nbytes) = packed;

        let descriptor = |name: &str, shape: &[u64]| {
            WeightDescriptor::stored(format!("routed_head_{name}"), shape)
        };
        let silu = |gate, up| FeedForwardUp::Gated {
            activation: ActivationFunction::Silu,
            gate,
            up,
        };
        let routed = RoutedFfn {
            experts: 4,
            selected: 2,
            intermediate: 64,
            router: Router {
                weight: descriptor("router", &[4, 128]),
                input: RouterInput::Operator,
                score: ScoreFunction::Softmax,
                selection: ExpertSelection::TopK { bias: None },
                normalization: RouteNormalization::Sum,
                scale: 1.0,
            },
            expert_up: silu(
                descriptor("expert_gate", &[4, 64, 128]),
                descriptor("expert_up", &[4, 64, 128]),
            ),
            expert_down: descriptor("expert_down", &[4, 128, 64]),
            expert_scale: None,
            latent: None,
            shared: Some(SharedExpert {
                intermediate: 64,
                up: silu(
                    descriptor("shared_gate", &[64, 128]),
                    descriptor("shared_up", &[64, 128]),
                ),
                down: descriptor("shared_down", &[128, 64]),
                gate: SharedExpertGate::Sigmoid(descriptor("shared_router", &[128])),
            }),
        };
        for weight in [
            &routed.router.weight,
            routed.expert_up.gate().unwrap(),
            routed.expert_up.up(),
            &routed.expert_down,
            routed.shared.as_ref().unwrap().up.gate().unwrap(),
            routed.shared.as_ref().unwrap().up.up(),
            &routed.shared.as_ref().unwrap().down,
            match &routed.shared.as_ref().unwrap().gate {
                SharedExpertGate::Sigmoid(gate) => gate,
                SharedExpertGate::None => unreachable!(),
            },
        ] {
            let nbytes = weight.shape.iter().product::<u64>() * 2;
            manifest.target.tensors.push(TensorDescriptor {
                name: weight.name.clone(),
                shape: weight.shape.clone(),
                encoding: Encoding::F16,
                offset,
                nbytes,
            });
            offset += nbytes;
        }
        manifest.target.files[0].size = offset;
        definition.head.as_mut().unwrap().blocks[0].blocks[0].sublayers[1].op =
            Operator::RoutedFfn(Box::new(routed));
        let head = definition.head.clone().unwrap();
        let head_block = crate::operators::CausalHeadBlock::of(&head, &head.blocks[0]).unwrap();
        let load = ModelLoadPlan::derive(
            &manifest,
            &definition,
            ComponentSelection {
                head: true,
                vision: false,
            },
            Layout::Rows16,
        )
        .unwrap();
        let plan = load.program_plan(&definition, KvCodec::Dense).unwrap();
        assert!(matches!(
            plan.head().unwrap().blocks()[0].feed_forward,
            crate::FeedForwardProgramSlot::Routed(_)
        ));
        let routed_family = crate::programs::native_head::checked_head_family_storage(
            BackendName::Cpu,
            &load,
            &definition.decoder,
            head_block,
            plan.head().unwrap().blocks()[0],
            crate::programs::native_head::head_graph_classes(
                limits,
                head_history.rows as u64,
                head_history.slab_rows,
                head_state.span_limit(),
                method.draft_rows(),
            )
            .unwrap(),
        )
        .unwrap();
        assert!(routed_family.storage.workspace > 0);
        let routed_pools = AssessmentGraphResourceBounds::derive(
            &definition,
            &load,
            &state,
            method,
            KvCodec::Dense,
            limits,
            BackendName::Cpu,
        )
        .unwrap();
        assert_eq!(
            routed_pools.head.unwrap().workspace_bytes,
            routed_family.storage.workspace
        );
        // Raw stream conditioning has independent target and head readout
        // features. Certify priming, prefix, chained and shaped classes with
        // the same topology that execution uses.
        definition.decoder = crate::planning::tests::hyper_connection_fixture_definition().decoder;
        let mut site = match &definition.decoder.exit.norm {
            ExitNorm::HyperConnectionHead(site) => site.clone(),
            _ => unreachable!(),
        };
        let head = &mut definition.head.as_mut().unwrap().blocks[0];
        head.residual = definition.decoder.residual.clone();
        head.chain = HeadChain::Residual;
        let HeadInput::Combine { hidden_norm, .. } = &mut head.input else {
            unreachable!()
        };
        hidden_norm.weight.shape = vec![512];
        for (index, sublayer) in head.blocks[0].sublayers.iter_mut().enumerate() {
            site.norm.weight.name = format!("head_mix_{index}_norm");
            site.down.name = format!("head_mix_{index}_down");
            site.up.name = format!("head_mix_{index}_up");
            site.inject = Some(descriptor(&format!("mix_{index}_inject"), &[4, 512]));
            sublayer.input = InputNorm::HyperConnectionMix(site.clone());
            sublayer.output = magnitude_family_contracts::OutputForm::Scatter;
        }
        site.norm.weight.name = "head_exit_norm".into();
        site.down.name = "head_exit_down".into();
        site.up.name = "head_exit_up".into();
        site.inject = None;
        head.output_norm = ExitNorm::HyperConnectionHead(site);
        let Operator::RoutedFfn(routed) = &mut head.blocks[0].sublayers[1].op else {
            unreachable!()
        };
        // CUDA routes whole groups of 32 experts.
        routed.experts = 32;
        routed.router.weight.shape[0] = 32;
        if let FeedForwardUp::Gated { gate, up, .. } = &mut routed.expert_up {
            gate.shape[0] = 32;
            up.shape[0] = 32;
        }
        routed.expert_down.shape[0] = 32;
        definition.validate().unwrap();
        let manifest = crate::planning::tests::fixture_manifest(&definition);
        let load = ModelLoadPlan::derive(
            &manifest,
            &definition,
            ComponentSelection {
                head: true,
                vision: false,
            },
            Layout::Rows16,
        )
        .unwrap();
        assert_eq!(load.residual_conditioning(), Some(512));
        let method = PlannedMethod::Mtp {
            greedy_proposals: 3,
            sampled_proposals: 3,
        };
        let state = crate::ResourcePlanner::state_plan(
            &definition,
            &load,
            method,
            KvCodec::Dense,
            limits,
            crate::ResourceCapacity {
                domain_bytes: 512 * 1024 * 1024,
                tensor_operations: crate::TensorOperations::Absent,
            },
        )
        .unwrap();
        let plan = load.program_plan(&definition, KvCodec::Dense).unwrap();
        let binding = plan.head().unwrap().blocks()[0];
        assert_eq!(binding.conditioning, Element::f32());
        assert_eq!(binding.streams.unwrap().geometry.streams, 4);
        let head = definition.head.as_ref().unwrap();
        let block = crate::operators::CausalHeadBlock::of(head, &head.blocks[0]).unwrap();
        let head_state = state.head_state().unwrap();
        let history = head_state.sole_history().unwrap();
        let classes = crate::programs::native_head::head_graph_classes(
            limits,
            history.rows as u64,
            history.slab_rows,
            head_state.span_limit(),
            method.draft_rows(),
        )
        .unwrap();
        for backend in [BackendName::Cpu, BackendName::Cuda] {
            crate::programs::native_head::verify_head_family_certificates(
                backend,
                &load,
                &definition.decoder,
                block,
                binding,
                &classes,
            )
            .unwrap();
            let pools = AssessmentGraphResourceBounds::derive(
                &definition,
                &load,
                &state,
                method,
                KvCodec::Dense,
                limits,
                backend,
            )
            .unwrap();
            assert!(pools.head.unwrap().workspace_bytes > 0);
        }
    }

    /// A draft head of its own width that reads the target's history and is
    /// stored in the draft component (the Gemma 4 assistant's form): the
    /// fixture target's layer under head names at width 96, its attention
    /// Shared with the target's only attention layer.
    fn target_reading_head_fixture() -> (ModelDefinition, magnitude_artifacts::PackageManifest) {
        use magnitude_artifacts::gguf::{Encoding, TensorDescriptor};
        use magnitude_artifacts::{ArtifactIdentity, ComponentFile, ComponentManifest};
        use magnitude_family_contracts::{
            ExitNorm, FeedForwardUp, Head, HeadBlock, HeadChain, HeadComponent, HeadHistory,
            HeadInput, HeadNorm, HeadReadout, InputNorm, KeyValue, Operator, RmsNorm,
            SublayerIndex, VocabularyProjection, WeightDescriptor,
        };
        const WIDTH: u64 = 96;
        let mut definition = crate::planning::tests::fixture_definition();
        let manifest = crate::planning::tests::fixture_manifest(&definition);
        let (hidden, vocabulary) = (definition.decoder.hidden, definition.decoder.vocabulary);
        let descriptor =
            |name: &str, shape: &[u64]| WeightDescriptor::stored(format!("head_{name}"), shape);
        let rms = |name: &str, width: u64| RmsNorm {
            weight: descriptor(name, &[width]),
            epsilon: 1e-6,
        };
        let mut block = definition.decoder.blocks[0].clone();
        for (index, sublayer) in block.sublayers.iter_mut().enumerate() {
            sublayer.input = InputNorm::Rms(rms(&format!("input_norm_{index}"), WIDTH));
            match &mut sublayer.op {
                Operator::Attention(attention) => {
                    let [rows, _] = attention.query.shape[..] else {
                        unreachable!()
                    };
                    attention.query = descriptor("query", &[rows, WIDTH]);
                    let [_, columns] = attention.output.shape[..] else {
                        unreachable!()
                    };
                    attention.output = descriptor("attention_output", &[WIDTH, columns]);
                    if let HeadNorm::Rms(norm) = &mut attention.query_norm {
                        norm.weight = descriptor("query_norm", &norm.weight.shape);
                    }
                    attention.key_value = KeyValue::Shared {
                        source: SublayerIndex {
                            block: 0,
                            sublayer: 0,
                        },
                    };
                }
                Operator::DenseFfn(dense) => {
                    let FeedForwardUp::Gated { gate, up, .. } = &mut dense.up else {
                        unreachable!()
                    };
                    *gate = descriptor("gate", &[dense.intermediate, WIDTH]);
                    *up = descriptor("up", &[dense.intermediate, WIDTH]);
                    dense.down = descriptor("down", &[WIDTH, dense.intermediate]);
                }
                _ => unreachable!(),
            }
        }
        definition.head = Some(Head {
            blocks: vec![HeadBlock {
                residual: magnitude_family_contracts::ResidualForm::Single,
                input: HeadInput::Projected {
                    pre: descriptor("pre_projection", &[WIDTH, 2 * hidden]),
                    embedding_scale: definition.decoder.entry.scale,
                },
                blocks: vec![block],
                output_norm: ExitNorm::Rms(rms("output_norm", WIDTH)),
                readout: HeadReadout::full(VocabularyProjection::Own {
                    table: descriptor("table", &[vocabulary, WIDTH]),
                }),
                chain: HeadChain::Projected {
                    post: descriptor("post_projection", &[hidden, WIDTH]),
                },
            }],
            history: HeadHistory::Target,
            component: HeadComponent::Draft,
        });
        definition.validate().unwrap();
        let mut offset = 0;
        let tensors = crate::operators::head_weights(definition.head.as_ref().unwrap())
            .unwrap()
            .into_iter()
            .map(|(_, descriptor)| {
                let nbytes = descriptor.shape.iter().product::<u64>() * 2;
                let tensor = TensorDescriptor {
                    name: descriptor.name.clone(),
                    shape: descriptor.shape.clone(),
                    encoding: Encoding::F16,
                    offset,
                    nbytes,
                };
                offset += nbytes;
                tensor
            })
            .collect();
        let manifest = magnitude_artifacts::PackageManifest {
            draft: Some(ComponentManifest {
                files: vec![ComponentFile {
                    path: "assistant-fixture.gguf".into(),
                    size: offset,
                }],
                identity: ArtifactIdentity([9; 32]),
                tensors,
            }),
            ..manifest
        };
        (definition, manifest)
    }

    /// A head reading target history plans its own program over the draft
    /// component's weights, keeps no state store, and its checked graph
    /// family is the head charge of header assessment.
    #[test]
    fn target_reading_head_plans_its_program_weights_and_charge() {
        let (definition, manifest) = target_reading_head_fixture();
        let load = ModelLoadPlan::derive(
            &manifest,
            &definition,
            ComponentSelection {
                head: true,
                vision: false,
            },
            Layout::Rows16,
        )
        .unwrap();
        // Every head weight, its own vocabulary table and both projections
        // included, is the draft component's.
        let head_weights = load.head.as_ref().unwrap();
        assert_eq!(
            head_weights.len(),
            crate::operators::head_weights(definition.head.as_ref().unwrap())
                .unwrap()
                .len()
        );
        assert!(
            head_weights
                .iter()
                .all(|weight| weight.component.kind == crate::ArtifactComponentKind::Draft)
        );
        let method = PlannedMethod::Mtp {
            greedy_proposals: 3,
            sampled_proposals: 3,
        };
        let limits = fixture_limits(false);
        let state = crate::ResourcePlanner::state_plan(
            &definition,
            &load,
            method,
            KvCodec::Dense,
            limits,
            crate::ResourceCapacity {
                domain_bytes: 512 * 1024 * 1024,
                tensor_operations: crate::TensorOperations::Absent,
            },
        )
        .unwrap();
        assert!(state.head_state().is_none());
        let plan = load.program_plan(&definition, KvCodec::Dense).unwrap();
        assert!(plan.head().is_none() && plan.draft().is_none());
        assert!(plan.drafts_with_head());
        let shared = plan.shared_head().unwrap();
        assert_eq!((shared.width, shared.blocks.len()), (96, 1));
        let geometry = crate::programs::native_shared_head::SharedHeadGeometry::new(
            &definition,
            &state,
            shared,
        )
        .unwrap();
        let classes =
            crate::programs::native_shared_head::shared_head_graph_classes(limits, 3).unwrap();
        assert!(classes.iter().any(|class| class.steps == 3 && class.shaped));
        let family = crate::programs::native_shared_head::checked_shared_head_family_storage(
            BackendName::Cpu,
            &load,
            &geometry,
            classes,
        )
        .unwrap();
        assert!(family.storage.workspace > 0);
        assert!(family.storage.output > 0);
        let pools = AssessmentGraphResourceBounds::derive(
            &definition,
            &load,
            &state,
            method,
            KvCodec::Dense,
            limits,
            BackendName::Cpu,
        )
        .unwrap();
        assert_eq!(
            pools.head.unwrap().workspace_bytes,
            family.storage.workspace
        );
    }

    /// A separate draft keeps its full packed logits: the projection it
    /// reads, the target's or its own, is stored packed and never as views.
    #[test]
    fn a_projection_a_separate_draft_reads_is_stored_packed() {
        use magnitude_artifacts::gguf::{Encoding, TensorDescriptor};
        use magnitude_artifacts::{ArtifactIdentity, ComponentFile, ComponentManifest};
        use magnitude_family_contracts::{
            BlockAttention, BlockLayout, DraftDefinition, DraftEmbedding, DraftMethod, RmsNorm,
            TapPoint, TokenId, WeightDescriptor,
        };
        for owns_output in [false, true] {
            let mut definition = crate::planning::tests::fixture_definition();
            let mut manifest = crate::planning::tests::fixture_manifest(&definition);
            let (hidden, vocabulary) = (definition.decoder.hidden, definition.decoder.vocabulary);
            let norm = |name: &str| RmsNorm {
                weight: WeightDescriptor::stored(name, &[hidden]),
                epsilon: 1e-6,
            };
            definition.draft = Some(DraftDefinition {
                method: DraftMethod::DFlash,
                taps: vec![TapPoint::Exit],
                fusion: WeightDescriptor::stored("draft_fusion", &[hidden, hidden]),
                fusion_norm: norm("draft_fusion_norm"),
                embedding: DraftEmbedding::Target,
                output: owns_output
                    .then(|| WeightDescriptor::stored("draft_output", &[vocabulary, hidden])),
                embedding_scale: 1.0,
                blocks: definition.decoder.blocks.clone(),
                block_attention: vec![BlockAttention::Bidirectional],
                output_norm: norm("draft_output_norm"),
                logit_softcap: Some(0.1),
                block_size: 2,
                mask_token: TokenId(0),
                layout: BlockLayout::MaskSlots,
            });
            definition.validate().unwrap();
            let draft = definition.draft.as_ref().unwrap();
            let mut offset = 0;
            let tensors = crate::operators::draft::fusion_weights(draft)
                .into_iter()
                .chain(crate::operators::draft::draft_weights(draft).unwrap())
                .map(|(_, descriptor)| {
                    let nbytes = descriptor.shape.iter().product::<u64>() * 2;
                    let tensor = TensorDescriptor {
                        name: descriptor.name.clone(),
                        shape: descriptor.shape.clone(),
                        encoding: Encoding::F16,
                        offset,
                        nbytes,
                    };
                    offset += nbytes;
                    tensor
                })
                .collect();
            manifest.draft = Some(ComponentManifest {
                files: vec![ComponentFile {
                    path: "draft-fixture.gguf".into(),
                    size: offset,
                }],
                identity: ArtifactIdentity([9; 32]),
                tensors,
            });
            let (component, descriptor) = match &draft.output {
                Some(output) => (manifest.draft.as_mut().unwrap(), output),
                None => (&mut manifest.target, &definition.decoder.exit.output),
            };
            let output = component
                .tensors
                .iter_mut()
                .find(|tensor| tensor.name == descriptor.name)
                .unwrap();
            output.encoding = Encoding::Q8_0;
            output.nbytes = output.shape.iter().product::<u64>() / 32 * 34;
            let load = ModelLoadPlan::derive(
                &manifest,
                &definition,
                ComponentSelection {
                    head: true,
                    vision: false,
                },
                Layout::Rows16,
            )
            .unwrap()
            .with_readout_candidates(&definition)
            .unwrap();
            let candidate_plan = load.program_plan(&definition, KvCodec::Dense).unwrap();
            assert!(matches!(
                candidate_plan.draft().unwrap().projection(),
                crate::ReadoutHead::Packed { .. }
            ));
            // A weight has one stored form, and the draft has no kernel for
            // the views: a projection it reads is not offered as them,
            // whether it is the target's or the draft's own, so no reader
            // of it can select a class.
            let read_by_draft = if owns_output {
                magnitude_family_contracts::WeightScope::Draft
            } else {
                magnitude_family_contracts::WeightScope::Target
            };
            assert!(
                !load
                    .readout_candidates()
                    .iter()
                    .any(|candidate| candidate.role.scope == read_by_draft),
                "a projection the draft reads is not offered as views"
            );
            assert!(
                load.clone()
                    .finalize_readout(&[crate::ReadoutClassChoice {
                        scope: magnitude_family_contracts::WeightScope::Target,
                        norm_scope: magnitude_family_contracts::WeightScope::Target,
                        prepared: false,
                        rows: vec![1],
                    }])
                    .is_err()
            );
            let load = load.finalize_readout(&[]).unwrap();
            let limits = fixture_limits(false);
            let state = crate::ResourcePlanner::state_plan(
                &definition,
                &load,
                PlannedMethod::DFlash {
                    proposals: 1,
                    generated: 1,
                    ladder: false,
                },
                KvCodec::Dense,
                limits,
                crate::ResourceCapacity {
                    domain_bytes: 512 << 20,
                    tensor_operations: crate::TensorOperations::Absent,
                },
            )
            .unwrap();
            let plan = load.program_plan(&definition, KvCodec::Dense).unwrap();
            let draft_plan = plan.draft().unwrap();
            // The draft's block pass projects every proposing row through
            // the packed weight, whichever component owns it, and the
            // target's readout reads the same one form.
            assert!(matches!(
                plan.target().readout().head,
                crate::ReadoutHead::Packed { .. }
            ));
            assert!(matches!(
                draft_plan.projection(),
                crate::ReadoutHead::Packed { .. }
            ));
            assert_eq!(
                draft_plan.projection_scope(),
                if owns_output {
                    magnitude_family_contracts::WeightScope::Draft
                } else {
                    magnitude_family_contracts::WeightScope::Target
                }
            );
            let geometry = crate::programs::native_draft::DraftGeometry::new(
                &definition,
                state.head_state().unwrap(),
            )
            .unwrap();
            let class = crate::programs::native_draft::DraftGraphClass {
                entry_rows: 1,
                slots: 1,
                proposals: 1,
                shaped: false,
                windowed: true,
                device: false,
            };
            let family = crate::programs::native_draft::checked_draft_family_storage(
                BackendName::Metal,
                &load,
                draft_plan,
                &geometry,
                [class],
            )
            .unwrap();
            assert!(family.storage.workspace > 0 && family.storage.output > 0);
        }
    }

    /// A head reading target history whose embedding scale differs from
    /// the target's entry transform is refused: its graphs gather rows
    /// through the target's embedding entry.
    #[test]
    fn target_reading_head_must_scale_the_embedding_as_its_target() {
        use magnitude_family_contracts::{EmbeddingScale, HeadInput};
        let (mut definition, _) = target_reading_head_fixture();
        let head = definition.head.as_mut().unwrap();
        let HeadInput::Projected {
            embedding_scale, ..
        } = &mut head.blocks[0].input
        else {
            unreachable!()
        };
        assert_eq!(*embedding_scale, EmbeddingScale::Unit);
        *embedding_scale = EmbeddingScale::SqrtHidden;
        assert!(matches!(
            crate::operators::admit(&definition, true),
            Err(crate::PlanError::Unsupported(
                "a draft head scaling the target's embedding differently"
            ))
        ));
        // An unselected head does not bear on the model.
        crate::operators::admit(&definition, false).unwrap();
    }

    #[test]
    fn checked_feature_readout_charges_only_its_export_and_upload() {
        use magnitude_family_contracts::{WeightKind, WeightRole, WeightScope};

        let definition = crate::planning::tests::fixture_definition();
        let manifest = crate::planning::tests::fixture_manifest(&definition);
        let load = ModelLoadPlan::derive(
            &manifest,
            &definition,
            ComponentSelection {
                head: false,
                vision: false,
            },
            Layout::Rows16,
        )
        .unwrap();
        let norm = load
            .weights()
            .find(|weight| {
                weight.role
                    == WeightRole {
                        scope: WeightScope::Target,
                        kind: WeightKind::OutputNorm,
                    }
            })
            .unwrap();
        let storage = crate::programs::graph::readout::checked_features_graph_storage(
            BackendName::Cpu,
            &definition.decoder,
            Some(norm),
            2,
            1,
        )
        .unwrap();
        assert_eq!(storage.workspace, 0);
        assert_eq!(storage.upload, 4);
        assert!(storage.output > 0);
    }

    #[test]
    fn checked_projected_readout_includes_native_scratch() {
        use magnitude_family_contracts::{WeightKind, WeightRole, WeightScope};

        let definition = crate::planning::tests::fixture_definition();
        let manifest = crate::planning::tests::fixture_manifest(&definition);
        let load = ModelLoadPlan::derive(
            &manifest,
            &definition,
            ComponentSelection {
                head: false,
                vision: false,
            },
            Layout::Rows16,
        )
        .unwrap();
        let weight = |kind| {
            load.weights()
                .find(|weight| {
                    weight.role
                        == WeightRole {
                            scope: WeightScope::Target,
                            kind,
                        }
                })
                .unwrap()
        };
        let features = crate::programs::graph::readout::checked_features_graph_storage(
            BackendName::Cpu,
            &definition.decoder,
            Some(weight(WeightKind::OutputNorm)),
            2,
            1,
        )
        .unwrap();
        let projected = crate::programs::graph::readout::checked_projected_graph_storage(
            BackendName::Cpu,
            &definition.decoder,
            Some(weight(WeightKind::OutputNorm)),
            weight(WeightKind::Output),
            2,
            1,
            1,
        )
        .unwrap();
        assert!(projected.workspace > features.workspace);
        assert!(projected.output > features.output);
        assert!(projected.upload > features.upload);
    }

    #[test]
    fn checked_selection_readout_charges_shaping_controls() {
        use magnitude_family_contracts::{WeightKind, WeightRole, WeightScope};

        let definition = crate::planning::tests::fixture_definition();
        let manifest = crate::planning::tests::fixture_manifest(&definition);
        let load = ModelLoadPlan::derive(
            &manifest,
            &definition,
            ComponentSelection {
                head: false,
                vision: false,
            },
            Layout::Rows16,
        )
        .unwrap();
        let weight = |kind| {
            load.weights()
                .find(|weight| {
                    weight.role
                        == WeightRole {
                            scope: WeightScope::Target,
                            kind,
                        }
                })
                .unwrap()
        };
        let selection = |shaped| {
            crate::programs::graph::readout::checked_selection_graph_storage(
                BackendName::Cpu,
                &definition.decoder,
                Some(weight(WeightKind::OutputNorm)),
                weight(WeightKind::Output),
                2,
                1,
                1,
                shaped,
            )
            .unwrap()
        };
        let plain = selection(false);
        let shaped = selection(true);
        assert!(shaped.workspace > plain.workspace);
        assert_eq!(shaped.output, plain.output);
        assert!(shaped.upload > plain.upload);
    }

    #[test]
    fn checked_readout_family_bounds_its_selection_classes() {
        let definition = crate::planning::tests::fixture_definition();
        let manifest = crate::planning::tests::fixture_manifest(&definition);
        let load = ModelLoadPlan::derive(
            &manifest,
            &definition,
            ComponentSelection {
                head: false,
                vision: false,
            },
            Layout::Rows16,
        )
        .unwrap();
        let limits = crate::ResourceLimits {
            max_launch_rows: 2,
            max_launch_slots: 2,
            max_selected_rows: 2,
            max_drafting_slots: 2,
            exported_logits_rows: 0,
            max_images_per_request: 0,
            max_image_cells: 0,
            lookahead: false,
        };
        let family = crate::programs::graph::readout::checked_readout_family_storage(
            BackendName::Cpu,
            &load,
            &definition.decoder,
            limits,
        )
        .unwrap();
        assert!(family.workspace > 0);
        assert!(family.output > 0);
        assert!(family.upload > 0);
    }

    #[test]
    fn checked_vision_family_bounds_exact_patch_classes() {
        use crate::{ArtifactComponent, ArtifactComponentKind, VisionProgramPlan, WeightPlan};
        use magnitude_artifacts::ArtifactIdentity;
        use magnitude_family_contracts::{
            ActivationDType, CellReduction, MergerStage, PositionSampling, VisionActivation,
            VisionAttention, VisionAttentionScale, VisionAttentionSpan, VisionBlock,
            VisionDescription, VisionFeedForward, VisionLinear, VisionMerger, VisionNorm,
            VisionPositions, VisionPreprocessing, VisionResampling, VisionResize, VisionStem,
            VisionUp, WeightDescriptor,
        };

        let f32 = Element::f32();
        let stored = |name: &str, shape: &[u64]| WeightDescriptor::stored(name, shape);
        let linear = |name: &str, outputs: u64, inputs: u64| VisionLinear {
            weight: stored(&format!("{name}.weight"), &[outputs, inputs]),
            bias: Some(stored(&format!("{name}.bias"), &[outputs])),
            clamp: None,
        };
        let layer_norm = |name: &str, width: u64| VisionNorm::Layer {
            weight: stored(&format!("{name}.weight"), &[width]),
            bias: stored(&format!("{name}.bias"), &[width]),
            epsilon: 1e-6,
        };
        // A Qwen3-VL-shaped tower at width 128: one block of one 128-wide
        // head, 2 x 2 cells.
        let description = VisionDescription {
            activation_dtype: ActivationDType::BF16,
            hidden: 128,
            output_hidden: 128,
            preprocessing: VisionPreprocessing {
                resize: VisionResize::PixelBounds {
                    min_pixels: 16,
                    max_pixels: 64,
                },
                resampling: VisionResampling::Bicubic,
                mean: [0.5; 3],
                std: [0.5; 3],
                channels: 3,
                patch: 2,
                merge: 2,
            },
            stem: VisionStem::Patch {
                frames: vec![
                    stored("patch.0", &[128, 3, 2, 2]),
                    stored("patch.1", &[128, 3, 2, 2]),
                ],
                bias: Some(stored("patch.bias", &[128])),
                positions: VisionPositions {
                    table: stored("position", &[16, 128]),
                    sampling: PositionSampling::AlignedCorners { side: 4 },
                },
                norm: None,
            },
            window: None,
            blocks: vec![VisionBlock {
                attention_norm: layer_norm("ln1", 128),
                attention: VisionAttention {
                    heads: 1,
                    width: 128,
                    query: linear("q", 128, 128),
                    key: linear("k", 128, 128),
                    value: linear("v", 128, 128),
                    query_norm: None,
                    key_norm: None,
                    value_norm: None,
                    rotary_base: Some(10000.0),
                    scale: VisionAttentionScale::InverseSqrtWidth,
                    span: VisionAttentionSpan::Full,
                    output: linear("o", 128, 128),
                },
                attention_post_norm: None,
                feedforward_norm: layer_norm("ln2", 128),
                feedforward: VisionFeedForward {
                    up: VisionUp::Plain(linear("up", 128, 128)),
                    activation: VisionActivation::GeluTanh,
                    down: linear("down", 128, 128),
                },
                feedforward_post_norm: None,
            }],
            merger: VisionMerger {
                norm: Some(layer_norm("post", 128)),
                reduction: CellReduction::Concatenate,
                standardize: None,
                projection_norm: None,
                stages: vec![
                    MergerStage {
                        linear: linear("mm.0", 512, 512),
                        activation: Some(VisionActivation::GeluErf),
                    },
                    MergerStage {
                        linear: linear("mm.2", 128, 512),
                        activation: None,
                    },
                ],
                output_norm: None,
            },
        };
        let component = ArtifactComponent {
            kind: ArtifactComponentKind::Projector,
            identity: ArtifactIdentity([9; 32]),
        };
        let weights = description
            .weights()
            .into_iter()
            .map(|(role, descriptor)| {
                let bytes = descriptor.shape.iter().product::<u64>() * 4;
                WeightPlan {
                    role,
                    component,
                    source: f32,
                    upload: f32,
                    resident: f32,
                    shape: descriptor.shape.clone(),
                    descriptor: descriptor.clone(),
                    source_bytes: bytes,
                    resident_bytes: bytes,
                    scale: None,
                    bound: 0,
                    streamed: None,
                }
            })
            .collect();
        let load = ModelLoadPlan {
            residual_conditioning: None,
            target: Vec::new(),
            head: None,
            vision: Some(weights),
            host_tables: Vec::new(),
            readout_candidates: Vec::new(),
            readout_choices: Vec::new(),
            unplaced_output: Vec::new(),
        };
        let program = crate::operators::vision::vision_program(&description, &|_| Ok(f32)).unwrap();
        let plan = VisionProgramPlan::new(program.kernels().into_iter().cloned().collect());
        let resources = |classes: &[u64]| {
            crate::programs::native_vision::checked_vision_family_resources(
                BackendName::Cpu,
                &load,
                &description,
                &plan,
                classes.iter().copied(),
            )
            .unwrap()
            .storage
        };
        let single = resources(&[4]);
        let double = resources(&[8]);
        let family = resources(&[4, 8]);
        crate::programs::native_vision::verify_vision_family_certificates(
            BackendName::Cpu,
            &load,
            &description,
            &plan,
            &[4, 8],
        )
        .unwrap();
        assert_eq!(family.workspace, single.workspace.max(double.workspace));
        assert_eq!(family.output, single.output.max(double.output));
        assert_eq!(family.upload, single.upload.max(double.upload));
        assert!(family.workspace > 0 && family.output > 0 && family.upload > 0);
    }

    /// A load whose readout selected planes stores its packed output
    /// projection unplaced, and its packed readers still bind it: the
    /// assessment's kernel listing over the finalized plan resolves it.
    #[test]
    fn kernels_of_a_finalized_target_readout_resolve_the_unplaced_projection() {
        let definition = crate::planning::tests::fixture_definition();
        let mut manifest = crate::planning::tests::fixture_manifest(&definition);
        let output = manifest
            .target
            .tensors
            .iter_mut()
            .find(|tensor| tensor.name == definition.decoder.exit.output.name)
            .unwrap();
        output.encoding = magnitude_artifacts::gguf::Encoding::Q8_0;
        output.nbytes = output.shape.iter().product::<u64>() / 32 * 34;
        let backend = BackendName::Metal;
        let load = ModelLoadPlan::derive(
            &manifest,
            &definition,
            ComponentSelection {
                head: false,
                vision: false,
            },
            crate::planning::resident_layout(crate::ExecutionPath::Native, backend),
        )
        .unwrap()
        .with_readout_candidates(&definition)
        .unwrap()
        .finalize_readout(&[crate::ReadoutClassChoice {
            scope: magnitude_family_contracts::WeightScope::Target,
            norm_scope: magnitude_family_contracts::WeightScope::Target,
            prepared: false,
            rows: vec![1],
        }])
        .unwrap();
        assert!(
            !load.weights().any(|placed| placed.role
                == magnitude_family_contracts::WeightRole {
                    scope: magnitude_family_contracts::WeightScope::Target,
                    kind: magnitude_family_contracts::WeightKind::Output,
                }),
            "the selected scope's packed projection is not placed"
        );
        let plan = load.program_plan(&definition, KvCodec::Dense).unwrap();
        HostProcessBytes::derive(&definition, &load, &plan, fixture_limits(false), backend)
            .unwrap();
    }

    fn fixture_limits(lookahead: bool) -> crate::ResourceLimits {
        crate::ResourceLimits {
            max_launch_rows: 2,
            max_launch_slots: 2,
            max_selected_rows: 2,
            max_drafting_slots: 2,
            exported_logits_rows: 0,
            max_images_per_request: 1,
            max_image_cells: 0,
            lookahead,
        }
    }

    #[test]
    fn header_terms_include_exact_selected_weights_history_and_bounds() {
        let mut definition = crate::planning::tests::fixture_definition();
        let manifest = crate::planning::tests::fixture_manifest(&definition);
        let selection = ComponentSelection {
            head: false,
            vision: false,
        };
        let load =
            ModelLoadPlan::derive(&manifest, &definition, selection, Layout::Rows16).unwrap();
        let limits = fixture_limits(false);
        let terms = AssessmentMemoryTerms::derive(
            &definition,
            &load,
            selection,
            KvCodec::Dense,
            PlannedMethod::Plain,
            limits,
        )
        .unwrap();

        assert_eq!(
            terms.target_weights,
            weight_bytes_by_component(&load).unwrap()[0]
        );
        assert_eq!(terms.head_weights, 0);
        assert_eq!(terms.vision_weights, 0);
        assert_eq!(terms.history_per_token, 2 * 64 * 2);
        assert_eq!(terms.recurrent_per_bank, 0);
        // Accepted bank, one in-flight successor, the prompt-boundary,
        // terminal and turn-start checkpoints, pristine seed.
        assert_eq!(terms.recurrent_banks, 6);
        assert_eq!(terms.fit_depth, 128);
        assert_eq!(terms.history_at_fit_depth().unwrap(), 128 * 2 * 64 * 2);
        let pipelined = AssessmentMemoryTerms::derive(
            &definition,
            &load,
            selection,
            KvCodec::Dense,
            PlannedMethod::Plain,
            fixture_limits(true),
        )
        .unwrap();
        assert_eq!(pipelined.recurrent_banks, 7);
        assert!(
            AssessmentMemoryTerms::derive(
                &definition,
                &load,
                selection,
                KvCodec::Dense,
                PlannedMethod::Mtp {
                    greedy_proposals: 1,
                    sampled_proposals: 1,
                },
                limits,
            )
            .is_err()
        );

        let header =
            AssessmentHeaderBounds::derive(&definition, &load, KvCodec::Dense, BackendName::Cpu)
                .unwrap();
        assert_eq!(
            header.prepared_program_bytes,
            AttestedPrograms::planned_invocation_workspace_bytes(
                &load.program_plan(&definition, KvCodec::Dense).unwrap(),
                BackendName::Cpu
            )
            .unwrap()
        );
        assert_eq!(
            header.startup_additional_bytes,
            AttestedPrograms::qualification_peak_bytes(&load)
                .unwrap()
                .max(load.target_upload_peak_bytes().unwrap())
        );
        assert_eq!(
            header.staging_upload_bytes,
            load.target_upload_peak_bytes().unwrap()
        );
        assert_eq!(
            header.component_upload_bytes,
            load.head_upload_peak_bytes()
                .unwrap()
                .max(load.vision_upload_peak_bytes().unwrap())
        );
        let state = crate::ResourcePlanner::state_plan(
            &definition,
            &load,
            PlannedMethod::Plain,
            KvCodec::Dense,
            limits,
            crate::ResourceCapacity {
                domain_bytes: 512 * 1024 * 1024,
                tensor_operations: crate::TensorOperations::Absent,
            },
        )
        .unwrap();
        let graph = AssessmentGraphResourceBounds::derive(
            &definition,
            &load,
            &state,
            PlannedMethod::Plain,
            KvCodec::Dense,
            limits,
            BackendName::Cpu,
        )
        .unwrap();
        let host_process = HostProcessBytes::derive(
            &definition,
            &load,
            &load.program_plan(&definition, KvCodec::Dense).unwrap(),
            limits,
            BackendName::Cpu,
        )
        .unwrap();
        let device_process =
            DeviceProcessBytes::derive(BackendName::Cpu, host_process.kernels).unwrap();
        let bounds = header
            .with_graph_resource_bound(&graph, host_process, device_process)
            .unwrap();
        assert_eq!(bounds.host_process, host_process);
        assert_eq!(bounds.device_process, device_process);
        assert_eq!(
            bounds.prepared_resource_bytes,
            header.prepared_program_bytes + graph.total_bytes + device_process.bytes().unwrap()
        );
        assert_eq!(
            bounds.startup_additional_bytes,
            header.startup_additional_bytes
        );
        let state_bytes = state
            .fit_state_bytes(terms.fit_depth, terms.recurrent_banks)
            .unwrap();
        let history = state.target_state().sole_history().unwrap();
        let history_layout = history.slab_layout().unwrap();
        let history_slabs = terms.fit_depth.div_ceil(u64::from(history.slab_rows));
        assert_eq!(
            state_bytes,
            history_layout.address_table_bytes + history_slabs * history_layout.slab_bytes
        );
        assert!(state_bytes > terms.history_at_fit_depth().unwrap());
        let startup_state = state.startup_state_bytes().unwrap();
        let charge = terms.charge(bounds, state_bytes, startup_state).unwrap();
        assert_eq!(
            charge.allocation_bytes,
            terms.exact_resident_bytes(0).unwrap()
                + bounds.prepared_resource_bytes
                + (state_bytes + bounds.component_upload_bytes)
                    .max(startup_state + bounds.startup_additional_bytes)
        );
        assert_eq!(charge.staging_bytes, header.staging_upload_bytes);

        // The fit depth is the context limit capped at 100,000 tokens.
        definition.decoder.context_limit = 262_144;
        let long = AssessmentMemoryTerms::derive(
            &definition,
            &load,
            selection,
            KvCodec::Dense,
            PlannedMethod::Plain,
            limits,
        )
        .unwrap();
        assert_eq!(long.fit_depth, 100_000);
        assert_eq!(
            long.history_at_fit_depth().unwrap(),
            100_000 * terms.history_per_token
        );
    }

    const NO_HOST_PROCESS: HostProcessBytes = HostProcessBytes {
        runtime: 0,
        vocabulary: 0,
        programs: 0,
        vision: 0,
        kernels: 0,
    };

    const NO_DEVICE_PROCESS: DeviceProcessBytes = DeviceProcessBytes {
        context: 0,
        programs: 0,
        slack: 0,
    };

    fn fixture_charge(allocation_bytes: u64, staging_bytes: u64) -> AssessmentMemoryCharge {
        AssessmentMemoryCharge {
            exact_resident_bytes: allocation_bytes,
            state_bytes: 0,
            startup_state_bytes: 0,
            bounds: AssessmentMemoryBounds {
                prepared_resource_bytes: 0,
                startup_additional_bytes: 0,
                component_upload_bytes: 0,
                staging_upload_bytes: staging_bytes,
                sealed_graph_host_bytes: SealedGraphHostBytes {
                    at_most: 0,
                    at_least: 0,
                },
                bound_graph_host_bytes: 0,
                drafter_graph_host_bytes: 0,
                host_process: NO_HOST_PROCESS,
                device_process: NO_DEVICE_PROCESS,
            },
            serving_bytes: allocation_bytes,
            allocation_bytes,
            staging_bytes,
            host_table_bytes: 0,
            shared_table_bytes: 0,
        }
    }

    #[test]
    fn workload_charge_composes_exact_terms_and_upper_bounds() {
        let terms = AssessmentMemoryTerms {
            target_weights: 100,
            head_weights: 20,
            vision_weights: 5,
            history_per_token: 2,
            recurrent_per_bank: 10,
            recurrent_banks: 3,
            fit_depth: 10,
            host_table_bytes: 30,
            shared_table_bytes: 0,
        };
        assert_eq!(terms.exact_resident_bytes(50).unwrap(), 100 + 20 + 5 + 50);
        let bounds = AssessmentMemoryBounds {
            prepared_resource_bytes: 40,
            startup_additional_bytes: 60,
            component_upload_bytes: 0,
            staging_upload_bytes: 7,
            sealed_graph_host_bytes: SealedGraphHostBytes {
                at_most: 0,
                at_least: 0,
            },
            bound_graph_host_bytes: 0,
            drafter_graph_host_bytes: 0,
            host_process: NO_HOST_PROCESS,
            device_process: NO_DEVICE_PROCESS,
        };
        // The workload's state outgrows the load's startup state and peak:
        // the peak is not charged on top of it.
        let charge = terms.charge(bounds, 90, 10).unwrap();
        assert_eq!(charge.exact_resident_bytes, 100 + 20 + 5 + 90);
        assert_eq!(charge.allocation_bytes, 125 + 40 + 90);
        assert_eq!(charge.serving_bytes, 125 + 40 + 90);
        assert_eq!(charge.staging_bytes, 7);
        // The load's startup state and peak exceed the workload's state:
        // serving holds the workload's state, not the load's peak.
        let loading = terms.charge(bounds, 50, 10).unwrap();
        assert_eq!(loading.allocation_bytes, 125 + 40 + (10 + 60));
        assert_eq!(loading.serving_bytes, 125 + 40 + 50);
        assert!(terms.charge(bounds, u64::MAX, 10).is_err());
        assert!(terms.charge(bounds, 50, u64::MAX).is_err());
        // An optional component a request loads at the fit depth claims its
        // import window beside the workload's state: serving's peak, and a
        // dedicated device stages the larger of the two windows.
        let component = AssessmentMemoryBounds {
            component_upload_bytes: 30,
            ..bounds
        };
        let loaded = terms.charge(component, 90, 10).unwrap();
        assert_eq!(loaded.allocation_bytes, 125 + 40 + 90 + 30);
        assert_eq!(loaded.serving_bytes, 125 + 40 + 90);
        assert_eq!(loaded.staging_bytes, 30);
        // The load's own peak still bounds it where it is the larger.
        let early = terms.charge(component, 20, 10).unwrap();
        assert_eq!(early.allocation_bytes, 125 + 40 + (10 + 60));
        assert!(
            AssessmentMemoryTerms {
                history_per_token: u64::MAX,
                ..terms
            }
            .history_at_fit_depth()
            .is_err()
        );
    }

    const GIB: u64 = 1 << 30;

    /// Status of an unopened machine whose host RAM `by` reports: assessment
    /// plans on stable fit, which no headroom changes.
    fn unopened(by: seismic::ReportedBy) -> Status {
        Status {
            sample: seismic::Sample {
                taken: std::time::SystemTime::now(),
                host_ram: Ok(seismic::HostStatus {
                    headroom: seismic::Reported::new(16 * GIB, by),
                    limits: Vec::new(),
                    limit_visibility: seismic::LimitVisibility::Complete,
                    displacement: None,
                    kernel_pressure: None,
                }),
                volumes: Vec::new(),
                grants: Vec::new(),
            },
            runtime: Default::default(),
        }
    }

    fn executor(topology: &DeviceTopology, backend: BackendName) -> &Executor {
        topology
            .executors()
            .iter()
            .find(|executor| executor.backend() == backend)
            .unwrap()
    }

    /// The machine's one device memory store.
    fn device_memory(topology: &DeviceTopology) -> StoreId {
        topology
            .stores()
            .iter()
            .find(|store| matches!(store.kind(), seismic::StoreKind::DeviceMemory(_)))
            .unwrap()
            .id()
    }

    /// The fit of `store` in `fit`.
    fn domain(fit: &AssessmentFit, store: StoreId) -> DomainFit {
        *fit.domains
            .iter()
            .find(|domain| domain.domain == store)
            .unwrap()
    }

    /// Each store a load charges is fitted on its own against its capacity
    /// less its planning reserve, and the store with the largest deficit
    /// limits the load. On the M4 Pro's CPU the load, its host tables and
    /// its sealed graph plans are one domain, host RAM; the RTX 3090's CUDA
    /// charges its allocation to the 3090's memory and its staged upload,
    /// host tables and sealed graph plans to host RAM.
    #[test]
    fn fit_is_decided_per_domain_against_capacity_less_reserve() {
        let reserves = crate::platform::MemoryReserves::standard();
        let m4_pro = recorded::m4_pro();
        let status = unopened(seismic::ReportedBy::MachMovablePages);
        let cpu = executor(&m4_pro, BackendName::Cpu);
        let host_ram = m4_pro.host_ram().id();
        let host = Fit::new(&m4_pro, &status, &reserves)
            .store(host_ram)
            .unwrap();
        let mut unified = AssessmentMemoryCharge {
            host_table_bytes: 30,
            ..fixture_charge(host.stable_bytes - 50, 0)
        };
        unified.bounds.sealed_graph_host_bytes = SealedGraphHostBytes {
            at_most: 20,
            at_least: 15,
        };
        let fits = unified
            .assess_fit(&m4_pro, cpu, &status, &reserves)
            .unwrap();
        assert_eq!(fits.verdict, AssessmentFitVerdict::Fits);
        assert_eq!(
            fits.domains,
            [DomainFit {
                role: DomainRole::Allocation,
                domain: host_ram,
                capacity_bytes: host.capacity_bytes,
                required_bytes: host.stable_bytes,
                reserve_bytes: host.thresholds.planning_bytes,
                remaining_bytes: 0,
            }]
        );
        let short = AssessmentMemoryCharge {
            host_table_bytes: 31,
            ..unified
        }
        .assess_fit(&m4_pro, cpu, &status, &reserves)
        .unwrap();
        assert_eq!(
            short.verdict,
            AssessmentFitVerdict::DoesNotFit {
                limiting: host_ram,
                deficit_bytes: 1,
            }
        );
        assert_eq!(short.domains[0].remaining_bytes, -1);

        let rtx3090 = recorded::rtx3090();
        let status = unopened(seismic::ReportedBy::LinuxMemAvailable);
        let cuda = executor(&rtx3090, BackendName::Cuda);
        let (gpu, host_ram) = (device_memory(&rtx3090), rtx3090.host_ram().id());
        let fit = Fit::new(&rtx3090, &status, &reserves);
        let (own, host) = (fit.store(gpu).unwrap(), fit.store(host_ram).unwrap());
        // CUDA's room is its usable memory less the store's planning
        // reserve, below the store's own fit.
        let room = fit.placement(cuda, StorageKind::Device).unwrap().stable_bytes;
        assert!(room < own.stable_bytes);
        let dedicated = |allocation, staging| {
            AssessmentMemoryCharge {
                host_table_bytes: 30,
                ..fixture_charge(allocation, staging)
            }
            .assess_fit(&rtx3090, cuda, &status, &reserves)
            .unwrap()
        };
        let both = dedicated(room, host.stable_bytes - 30);
        assert_eq!(both.verdict, AssessmentFitVerdict::Fits);
        assert_eq!(
            domain(&both, gpu),
            DomainFit {
                role: DomainRole::Allocation,
                domain: gpu,
                capacity_bytes: room + own.thresholds.planning_bytes,
                required_bytes: room,
                reserve_bytes: own.thresholds.planning_bytes,
                remaining_bytes: 0,
            }
        );
        assert_eq!(domain(&both, host_ram).role, DomainRole::Staging);
        assert_eq!(domain(&both, host_ram).required_bytes, host.stable_bytes);
        assert_eq!(domain(&both, host_ram).remaining_bytes, 0);
        // Staging 50 bytes beyond host RAM's fit: host RAM limits the load.
        let staging = dedicated(room, host.stable_bytes - 30 + 50);
        assert_eq!(
            staging.verdict,
            AssessmentFitVerdict::DoesNotFit {
                limiting: host_ram,
                deficit_bytes: 50,
            }
        );
        assert_eq!(domain(&staging, gpu).remaining_bytes, 0);
        assert_eq!(domain(&staging, host_ram).remaining_bytes, -50);
        // The largest deficit limits the load.
        let worst = dedicated(room + 100, host.stable_bytes - 30 + 50);
        assert_eq!(
            worst.verdict,
            AssessmentFitVerdict::DoesNotFit {
                limiting: gpu,
                deficit_bytes: 100,
            }
        );
    }

    /// The graphs' static bindings, the drafter's graphs and the worker's
    /// own host bytes are system RAM, held with the host tables and the
    /// sealed plans: on the RTX 3090's CUDA they are host RAM's, beside the
    /// staged upload, and the device's own memory holds none of them.
    #[test]
    fn worker_host_bytes_are_charged_to_host_ram() {
        let reserves = crate::platform::MemoryReserves::standard();
        let mut worker = AssessmentMemoryCharge {
            host_table_bytes: 50,
            ..fixture_charge(800, 300)
        };
        worker.bounds.sealed_graph_host_bytes = SealedGraphHostBytes {
            at_most: 20,
            at_least: 15,
        };
        worker.bounds.bound_graph_host_bytes = 3;
        worker.bounds.drafter_graph_host_bytes = 1;
        worker.bounds.host_process = HostProcessBytes {
            runtime: 4,
            vocabulary: 5,
            programs: 6,
            vision: 7,
            kernels: 2,
        };
        assert_eq!(
            worker.host_bytes().unwrap(),
            50 + 20 + 3 + 1 + 4 + 5 + 6 + 7
        );

        let rtx3090 = recorded::rtx3090();
        let status = unopened(seismic::ReportedBy::LinuxMemAvailable);
        let cuda = executor(&rtx3090, BackendName::Cuda);
        let (gpu, host_ram) = (device_memory(&rtx3090), rtx3090.host_ram().id());
        let fit = worker
            .assess_fit(&rtx3090, cuda, &status, &reserves)
            .unwrap();
        assert_eq!(domain(&fit, gpu).required_bytes, 800);
        assert_eq!(
            domain(&fit, host_ram).required_bytes,
            300 + worker.host_bytes().unwrap()
        );
    }

    /// The worker's device bytes are the backend's context and a cost per
    /// kernel the load prepares; a unified device's are in its host bytes,
    /// and a CUDA context is already outside the device's usable capacity.
    #[test]
    fn worker_device_bytes_price_the_context_and_kernels() {
        for backend in [
            BackendName::Cpu,
            BackendName::Metal,
            BackendName::Vulkan,
            BackendName::Cuda,
        ] {
            let costs = DeviceProcessCosts::of(backend);
            let device = DeviceProcessBytes::derive(backend, 40).unwrap();
            assert_eq!(
                device,
                DeviceProcessBytes {
                    context: costs.context,
                    programs: 40 * costs.per_kernel,
                    slack: costs.slack,
                }
            );
            assert_eq!(
                device.bytes().unwrap(),
                device.context + device.programs + device.slack
            );
        }
        for unified in [BackendName::Cpu, BackendName::Metal] {
            let device = DeviceProcessBytes::derive(unified, 40).unwrap();
            assert_eq!(device.bytes().unwrap(), 0);
        }
        // A CUDA context is outside the usable capacity; a dedicated
        // device's block allocator holds free space whatever the model.
        let cuda = DeviceProcessBytes::derive(BackendName::Cuda, 40).unwrap();
        assert_eq!(cuda.context, 0);
        assert!(cuda.slack > 0);
        let vulkan = DeviceProcessBytes::derive(BackendName::Vulkan, 40).unwrap();
        assert!(vulkan.context > 0 && vulkan.programs > 0 && vulkan.slack > 0);
        assert!(DeviceProcessBytes::derive(BackendName::Vulkan, u64::MAX).is_err());
    }

    /// The worker's host bytes are counts the plan fixes times the costs
    /// measured on the backend: its runtime, a cost per vocabulary entry and
    /// one per native kernel the load prepares, the kernels being the
    /// load's own inventory (`kernel_requests`, which a load's preparation
    /// is tested to prepare exactly).
    #[test]
    fn worker_host_bytes_price_the_plans_vocabulary_and_kernels() {
        let definition = crate::planning::tests::fixture_definition();
        let manifest = crate::planning::tests::fixture_manifest(&definition);
        let selection = ComponentSelection {
            head: false,
            vision: false,
        };
        let load =
            ModelLoadPlan::derive(&manifest, &definition, selection, Layout::Rows16).unwrap();
        let programs = load.program_plan(&definition, KvCodec::Dense).unwrap();
        let limits = fixture_limits(false);
        for backend in [
            BackendName::Cpu,
            BackendName::Metal,
            BackendName::Vulkan,
            BackendName::Cuda,
        ] {
            let costs = HostProcessCosts::of(backend);
            let kernels = crate::native::kernel_requests(
                backend,
                &programs,
                &definition,
                &load,
                crate::TuningLimits::of(limits, &definition),
            )
            .unwrap()
            .len() as u64;
            assert!(kernels > 0);
            let host =
                HostProcessBytes::derive(&definition, &load, &programs, limits, backend).unwrap();
            // The fixture reads no images: no vision term.
            assert!(programs.vision().is_none());
            assert_eq!(
                host,
                HostProcessBytes {
                    runtime: costs.runtime,
                    vocabulary: costs.per_token * definition.decoder.vocabulary,
                    programs: costs.per_kernel * kernels,
                    vision: 0,
                    kernels,
                }
            );
            assert_eq!(
                host.bytes().unwrap(),
                host.runtime + host.vocabulary + host.programs
            );
            assert!(host.runtime > 0 && host.vocabulary > 0 && host.programs > 0);
        }
    }

    /// The staging window is part of the allocation charge. An executor
    /// whose device storage is placed on host RAM (Metal on the M4 Pro,
    /// CUDA on GB10, Vulkan on Strix Halo and Lunar Lake) holds it there
    /// once, inside the allocation: its staging adds nothing. A dedicated
    /// device's staging window (CUDA on the RTX 3090) is host RAM's, beside
    /// the allocation in its own memory.
    #[test]
    fn staging_is_charged_to_host_ram_once() {
        let reserves = crate::platform::MemoryReserves::standard();
        let (allocation, staged, tables) = (8 * GIB, 2 * GIB, 30);
        let assess = |topology: &DeviceTopology, executor: &Executor, status: &Status, staging| {
            AssessmentMemoryCharge {
                host_table_bytes: tables,
                ..fixture_charge(allocation, staging)
            }
            .assess_fit(topology, executor, status, &reserves)
            .unwrap()
        };
        use seismic::ReportedBy::{LinuxMemAvailable, MachMovablePages, WindowsPhysicalAndCommit};
        for (topology, backend, by) in [
            (recorded::m4_pro(), BackendName::Metal, MachMovablePages),
            (recorded::gb10(), BackendName::Cuda, LinuxMemAvailable),
            (
                recorded::strix_halo_linux(),
                BackendName::Vulkan,
                LinuxMemAvailable,
            ),
            (
                recorded::lunar_lake(),
                BackendName::Vulkan,
                WindowsPhysicalAndCommit,
            ),
        ] {
            let status = unopened(by);
            let executor = executor(&topology, backend);
            let host_ram = topology.host_ram().id();
            let staging = assess(&topology, executor, &status, staged);
            assert_eq!(staging.verdict, AssessmentFitVerdict::Fits);
            assert_eq!(
                domain(&staging, host_ram).required_bytes,
                allocation + tables,
                "{backend:?} on {}",
                executor.name()
            );
            assert_eq!(staging, assess(&topology, executor, &status, 0));
        }

        let rtx3090 = recorded::rtx3090();
        let status = unopened(seismic::ReportedBy::LinuxMemAvailable);
        let cuda = executor(&rtx3090, BackendName::Cuda);
        let (gpu, host_ram) = (device_memory(&rtx3090), rtx3090.host_ram().id());
        let staging = assess(&rtx3090, cuda, &status, staged);
        assert_eq!(domain(&staging, gpu).required_bytes, allocation);
        assert_eq!(domain(&staging, host_ram).required_bytes, tables + staged);
    }

    /// Metal's working set bounds only its device storage: host-resident
    /// tables are mapped pages, not Metal buffers, so host RAM holds them
    /// beside it, and the store's room for the load is the tables with the
    /// working set's room.
    #[test]
    fn host_demand_sits_beside_the_metal_working_set() {
        let reserves = crate::platform::MemoryReserves::standard();
        let m4_pro = recorded::m4_pro();
        let status = unopened(seismic::ReportedBy::MachMovablePages);
        let metal = executor(&m4_pro, BackendName::Metal);
        let host_ram = m4_pro.host_ram().id();
        let fit = Fit::new(&m4_pro, &status, &reserves);
        let room = fit.placement(metal, StorageKind::Device).unwrap().stable_bytes;
        let host = fit.store(host_ram).unwrap();
        assert!(room + GIB < host.stable_bytes);
        let assess = |device| {
            AssessmentMemoryCharge {
                host_table_bytes: GIB,
                ..fixture_charge(device, 0)
            }
            .assess_fit(&m4_pro, metal, &status, &reserves)
            .unwrap()
        };
        let full = assess(room);
        assert_eq!(full.verdict, AssessmentFitVerdict::Fits);
        assert_eq!(
            domain(&full, host_ram),
            DomainFit {
                role: DomainRole::Allocation,
                domain: host_ram,
                capacity_bytes: room + GIB + host.thresholds.planning_bytes,
                required_bytes: room + GIB,
                reserve_bytes: host.thresholds.planning_bytes,
                remaining_bytes: 0,
            }
        );
        assert_eq!(
            assess(room + 1).verdict,
            AssessmentFitVerdict::DoesNotFit {
                limiting: host_ram,
                deficit_bytes: 1,
            }
        );
    }

    /// A stored table's storage (upload storage, behind `Tensor::shared`) is
    /// charged where its executor's upload placement lands: on the RTX 3090,
    /// CUDA's pinned host RAM, where a load whose host RAM is full is
    /// refused though the 3090's memory holds the rest, and Vulkan's device
    /// memory (ReBAR), where the same load fits.
    #[test]
    fn stored_tables_are_charged_in_the_domain_of_shared_storage() {
        let reserves = crate::platform::MemoryReserves::standard();
        let topology = recorded::rtx3090();
        let status = unopened(seismic::ReportedBy::LinuxMemAvailable);
        let (gpu, host_ram) = (device_memory(&topology), topology.host_ram().id());
        let fit = Fit::new(&topology, &status, &reserves);
        let (own, host) = (fit.store(gpu).unwrap(), fit.store(host_ram).unwrap());
        let cuda = executor(&topology, BackendName::Cuda);
        let vulkan = executor(&topology, BackendName::Vulkan);
        let charge = |staging, executor| {
            AssessmentMemoryCharge {
                shared_table_bytes: GIB,
                ..fixture_charge(own.stable_bytes, staging)
            }
            .assess_fit(&topology, executor, &status, &reserves)
            .unwrap()
        };
        let pinned = charge(0, cuda);
        assert_eq!(pinned.verdict, AssessmentFitVerdict::Fits);
        assert_eq!(domain(&pinned, gpu).required_bytes, own.stable_bytes - GIB);
        // What remains of CUDA's room there: its usable memory less the
        // reserve, below the store's own fit.
        let room = fit.placement(cuda, StorageKind::Device).unwrap().stable_bytes;
        assert_eq!(
            domain(&pinned, gpu).remaining_bytes,
            i64::try_from(room - (own.stable_bytes - GIB)).unwrap()
        );
        assert_eq!(domain(&pinned, host_ram).required_bytes, GIB);
        let local = charge(0, vulkan);
        assert_eq!(local.verdict, AssessmentFitVerdict::Fits);
        assert_eq!(domain(&local, gpu).required_bytes, own.stable_bytes);
        assert_eq!(domain(&local, host_ram).required_bytes, 0);
        // Staging takes host RAM's fit.
        let crowded = charge(host.stable_bytes, cuda);
        assert_eq!(
            crowded.verdict,
            AssessmentFitVerdict::DoesNotFit {
                limiting: host_ram,
                deficit_bytes: GIB,
            }
        );
        assert_eq!(
            domain(&crowded, gpu).remaining_bytes,
            domain(&pinned, gpu).remaining_bytes
        );
        assert_eq!(
            charge(host.stable_bytes, vulkan).verdict,
            AssessmentFitVerdict::Fits
        );
    }
}
