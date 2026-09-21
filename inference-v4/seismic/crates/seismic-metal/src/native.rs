//! Native assembly of Metal plans: compile every launch, reflect native
//! facts against their declared domains, fold everything into direct
//! handles, and seal one native tree mirroring the physical schedule
//! exactly once.
//!
//! The artifact owns the device and command queue it was assembled with.
//! Construction is private (`assemble` is the only constructor); an empty
//! schedule seals the valid identity artifact (no library, no launches).
//! The arena, the folded native facts, the storage targets, the result and
//! status field tables, the dense guard→status pairing, and the folded
//! join/carry copy records all live here so the executor never looks
//! anything up by a fallible identifier and never observes an absent
//! compiler-owned record: one compiled handle per launch, one direct
//! binding program per launch, dense tables and paired records indexed by
//! the sealed plan's dense indices.

use crate::encode::{self, EncodedLaunch, RenderedBinding};
use crate::intrinsics::MetalDialect;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{MTLCompileOptions, MTLComputePipelineState, MTLDevice, MTLLibrary};
use seismic_compiler::pipeline::{AssemblyFailure, EncodedPlan, EncodedStep, ToolchainReport};
use seismic_realization::failure::{CompilerDefect, Package};
use seismic_realization::ids::{
    GuardIx, LaunchIx, NativeFactIx, ObligationRef, ResultFieldIx, ScalarSlotIx, StatusFieldIx,
    StorageIx,
};
use seismic_realization::physical::{
    ExecutionExpr, GuardPredicate, LaunchResources, PhysicalStep, PlanResources, ResultField,
    ScalarSource, SealedLaunch, SealedValue, StatusField, StorageFact, StoragePlacement,
};
use seismic_realization::strategy::NativeFactKind;
use std::collections::BTreeMap;

/// Where one physical storage lives after resolution, folded into the
/// artifact so the executor binds directly.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StorageTarget {
    /// A public root-ABI buffer (the invocation's validated buffer).
    Abi { slot: seismic_realization::ids::BufferSlot },
    /// A slice of the internal device arena.
    Arena { offset: u64 },
    /// A workgroup or participant staging array: declared inside the
    /// kernel, never bound or filled by the host.
    Staged,
}

/// What one absolute Metal buffer argument of a launch binds.
#[derive(Clone, Debug)]
pub(crate) enum NativeBinding {
    Storage {
        index: u32,
        target: StorageTarget,
    },
    Scalar {
        index: u32,
        source: ScalarSource,
        dtype: seismic_lang::types::DType,
    },
    /// The evaluated work-item total.
    Total { index: u32 },
    /// One runtime extent's evaluated actual value.
    Extent { index: u32, value: ExecutionExpr },
    /// The executor-scalar slot block.
    Slots { index: u32 },
    /// The compiler-owned dense result scalar block.
    Results { index: u32 },
    /// The root status block.
    Status { index: u32 },
}

/// One compiled launch: its native pipeline state and complete direct
/// binding program. No public constructor.
pub struct NativeLaunch {
    kernel: String,
    state: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    workgroup_bytes: u64,
    work_items: ExecutionExpr,
    participants: ExecutionExpr,
    workgroups: [ExecutionExpr; 3],
    /// The block's iteration map serializes the domain onto one
    /// participant; the sealed geometry is `[1, 1, 1]` with one participant.
    serialized: bool,
    bindings: Vec<NativeBinding>,
    resources: LaunchResources,
}

impl NativeLaunch {
    pub fn kernel(&self) -> &str {
        &self.kernel
    }

    pub fn workgroup_bytes(&self) -> u64 {
        self.workgroup_bytes
    }

    pub fn work_items(&self) -> &ExecutionExpr {
        &self.work_items
    }

    pub fn participants(&self) -> &ExecutionExpr {
        &self.participants
    }

    pub fn workgroups(&self) -> &[ExecutionExpr; 3] {
        &self.workgroups
    }

    pub fn serialized(&self) -> bool {
        self.serialized
    }

    pub fn resources(&self) -> &LaunchResources {
        &self.resources
    }

    pub(crate) fn bindings(&self) -> &[NativeBinding] {
        &self.bindings
    }

    pub(crate) fn state(&self) -> &ProtocolObject<dyn MTLComputePipelineState> {
        &self.state
    }
}

// ---------------------------------------------------------------------------
// Folded join/carry copy records
// ---------------------------------------------------------------------------

/// The writable destination of one scalar join/carry value, folded from
/// the sealed route: an executor slot or a result-block field. A caller-
/// owned or invocation-owned scalar is never a join/carry destination.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScalarDestination {
    Slot(ScalarSlotIx),
    Result(ResultFieldIx),
}

/// One side of a folded join/carry copy: a readable scalar source, or a
/// tensor storage whose bytes the executor copies.
#[derive(Clone, Debug)]
pub enum CopySource {
    Scalar { source: ScalarSource },
    Tensor { storage: StorageIx, bytes: u64 },
}

/// The destination of a folded join/carry copy.
#[derive(Clone, Debug)]
pub enum CopyDestination {
    Scalar { destination: ScalarDestination },
    Tensor { storage: StorageIx, bytes: u64 },
}

/// One branch join, folded: after the branch, the destination holds the
/// taken side's value (same storage ⇒ no-op, checked at execution).
#[derive(Clone, Debug)]
pub struct NativeJoin {
    pub then_source: CopySource,
    pub else_source: CopySource,
    pub destination: CopyDestination,
}

/// One repeat carry, folded: `current := initial` before the first visit,
/// `current := update` after each visit, `result := current` after the
/// loop (the executor reads a destination back through the same record).
#[derive(Clone, Debug)]
pub struct NativeCarry {
    pub initial: CopySource,
    pub update: CopySource,
    pub current: CopyDestination,
    pub result: CopyDestination,
}

// ---------------------------------------------------------------------------
// The native tree and artifact
// ---------------------------------------------------------------------------

/// The sealed native execution tree: every `PhysicalStep` of the physical
/// schedule exactly once, launches replaced by their direct handles and
/// join/carry values by folded copy records.
pub enum NativeStep {
    Launch { launch: LaunchIx },
    Guard {
        guard: GuardIx,
        obligation: ObligationRef,
        status: StatusFieldIx,
        predicate: GuardPredicate,
    },
    Call(Vec<NativeStep>),
    If {
        condition: ScalarSource,
        then_steps: Vec<NativeStep>,
        else_steps: Vec<NativeStep>,
        joins: Vec<NativeJoin>,
    },
    Repeat {
        start: ExecutionExpr,
        end: ExecutionExpr,
        bound: ExecutionExpr,
        binder: ScalarSlotIx,
        body: Vec<NativeStep>,
        carries: Vec<NativeCarry>,
    },
    /// A host-side fill: zeroes `bytes` bytes of the arena at `offset`
    /// before the launch that follows it (fills zero internal residences
    /// only; a non-arena fill is rejected at assembly).
    Fill { offset: u64, bytes: u64 },
}

/// The sealed native artifact. Private construction; owns its device and
/// command queue. Structured execution over its tree cannot fail for
/// missing compiler structure.
pub struct NativeArtifact {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn objc2_metal::MTLCommandQueue>>,
    steps: Vec<NativeStep>,
    /// Dense by `LaunchIx::index()` (one handle per schedule launch).
    launches: Vec<NativeLaunch>,
    /// Dense by `StorageIx::index()` over every sealed storage.
    storage_targets: Vec<StorageTarget>,
    /// Bytes at capacity per sealed storage (tensor join/carry copies).
    storage_bytes: Vec<u64>,
    resources: PlanResources,
    /// The sealed plan's estimated cost and optimality verdict (telemetry
    /// facts; assembly consumes the plan and retains these two numbers).
    estimated_cost: u64,
    optimal: bool,
    /// Result fields in ABI order (path, endpoint, dtype).
    result_fields: Vec<ResultField>,
    /// Status fields in plan order (obligation, kind).
    status_fields: Vec<StatusField>,
    /// The status field each sealed guard dominates, dense by
    /// `GuardIx::index()` (every sealed guard is a schedule guard step).
    guard_status_fields: Vec<StatusFieldIx>,
    /// The dominating sealed guard of each status field, when one exists
    /// (a kernel-`Check` field legitimately has none).
    status_guards: Vec<Option<GuardIx>>,
    /// Folded native facts, dense by `NativeFactIx::index()`.
    native_facts: Vec<u64>,
}

impl NativeArtifact {
    pub fn steps(&self) -> &[NativeStep] {
        &self.steps
    }

    /// One compiled handle per launch of the sealed schedule. Infallible:
    /// the tree addresses launches by exactly these identities.
    pub fn launch(&self, launch: LaunchIx) -> &NativeLaunch {
        &self.launches[launch.index()]
    }

    /// The sealed plan's estimated cost and optimality verdict.
    pub fn estimated_cost(&self) -> u64 {
        self.estimated_cost
    }
    pub fn optimal(&self) -> bool {
        self.optimal
    }
    pub fn launch_count(&self) -> usize {
        self.launches.len()
    }

    pub fn resources(&self) -> &PlanResources {
        &self.resources
    }

    pub fn result_fields(&self) -> &[ResultField] {
        &self.result_fields
    }

    pub fn status_fields(&self) -> &[StatusField] {
        &self.status_fields
    }

    /// The dominating sealed guard of one status field, when one exists (a
    /// kernel-`Check` field legitimately has none).
    pub fn status_guard(&self, field: StatusFieldIx) -> Option<GuardIx> {
        self.status_guards.get(field.index()).copied().flatten()
    }

    /// The status field one sealed guard dominates. Infallible over the
    /// sealed guard indices: every sealed guard is a schedule guard step
    /// and the table is dense by `GuardIx`.
    pub(crate) fn status_field_of_guard(&self, guard: GuardIx) -> StatusFieldIx {
        self.guard_status_fields[guard.index()]
    }

    /// One folded native fact. Infallible over the sealed fact indices.
    pub fn native_fact(&self, fact: NativeFactIx) -> u64 {
        self.native_facts[fact.index()]
    }

    /// The storage target of one sealed storage (executor bindings).
    pub fn storage_target(&self, storage: StorageIx) -> StorageTarget {
        self.storage_targets[storage.index()]
    }

    /// The capacity bytes of one sealed storage (tensor join/carry
    /// copies).
    pub fn storage_bytes(&self, storage: StorageIx) -> u64 {
        self.storage_bytes[storage.index()]
    }

    pub(crate) fn device(&self) -> &ProtocolObject<dyn MTLDevice> {
        &self.device
    }

    pub(crate) fn queue(&self) -> &ProtocolObject<dyn objc2_metal::MTLCommandQueue> {
        &self.queue
    }
}

fn invariant(detail: impl Into<String>) -> AssemblyFailure {
    AssemblyFailure::CompilerInvariant(CompilerDefect::new(Package::B1Metal, detail))
}

/// Compile every encoded launch, reflect native facts against their
/// declared domains, fold into direct handles, and seal one native tree
/// mirroring the physical schedule exactly once. The only constructor of
/// `NativeArtifact`.
pub fn assemble(
    device: &crate::runtime::Device,
    encoded: &EncodedPlan<MetalDialect, EncodedLaunch>,
) -> Result<NativeArtifact, AssemblyFailure> {
    let plan = encoded.physical();
    // Every encoded launch of the tree, in tree order, exactly once.
    let mut encoded_launches: BTreeMap<LaunchIx, &EncodedLaunch> = BTreeMap::new();
    collect_launches(encoded.steps(), &mut encoded_launches);
    // Render each launch's MSL once against the sealed plan; a contradicted
    // encoder invariant is a compile-time defect here, never a repair.
    let mut rendered: BTreeMap<LaunchIx, encode::RenderedLaunch> = BTreeMap::new();
    for (index, launch) in &encoded_launches {
        let rendered_launch = encode::render(&launch.launch, plan)
            .map_err(AssemblyFailure::CompilerInvariant)?;
        rendered.insert(*index, rendered_launch);
    }
    let library = compile_library(device, &rendered)?;
    // Compile each launch, reflecting its declared native facts against
    // the domains the catalog declared.
    let mut native_facts: BTreeMap<NativeFactIx, u64> = BTreeMap::new();
    let mut launches = Vec::with_capacity(rendered.len());
    for (index, launch) in &encoded_launches {
        validate_resources(&launch.launch)?;
        let rendered = &rendered[index];
        let compiled = compile_launch(device, &library, plan, rendered, &launch.launch)?;
        for declaration in &launch.launch.native_facts {
            let reflected = match declaration.kind {
                NativeFactKind::MaxResidentParticipants => compiled.max_threads,
                NativeFactKind::NativeSubgroupWidth => compiled.execution_width,
            };
            if !(declaration.min..=declaration.max).contains(&reflected) {
                return Err(invariant(format!(
                    "kernel `{}` reflected {} {reflected} outside its declared domain \
                     [{}, {}]",
                    rendered.kernel,
                    match declaration.kind {
                        NativeFactKind::MaxResidentParticipants => {
                            "resident participant limit"
                        }
                        NativeFactKind::NativeSubgroupWidth => "subgroup width",
                    },
                    declaration.min,
                    declaration.max
                )));
            }
            if native_facts
                .insert(declaration.index, reflected)
                .is_some()
            {
                return Err(invariant(
                    "two launches declare the same native fact index (the seal allocates \
                     them globally dense)",
                ));
            }
        }
        launches.push(compiled.launch);
    }
    // Fold the dense storage tables over every sealed storage.
    let targets: Vec<StorageTarget> = plan
        .storages()
        .values()
        .map(|storage| storage_target_of(storage.placement))
        .collect();
    let storage_bytes: Vec<u64> =
        plan.storages().values().map(|storage| storage.bytes).collect();
    let (guard_status_fields, status_guards) = fold_guard_pairing(plan);
    let fact_count = native_facts.keys().map(|f| f.index() + 1).max().unwrap_or(0);
    let mut fact_table = vec![0u64; fact_count];
    for (fact, value) in native_facts {
        fact_table[fact.index()] = value;
    }
    Ok(NativeArtifact {
        device: device.handle().into(),
        queue: device.queue_handle().into(),
        steps: native_steps(plan, &plan.schedule().steps, encoded.steps())?,
        launches,
        storage_targets: targets,
        storage_bytes,
        resources: plan.resources().clone(),
        estimated_cost: plan.estimated_cost(),
        optimal: plan.optimal(),
        result_fields: plan.result_fields().values().cloned().collect(),
        status_fields: plan.status_fields().values().cloned().collect(),
        guard_status_fields,
        status_guards,
        native_facts: fact_table,
    })
}

fn storage_target_of(placement: StoragePlacement) -> StorageTarget {
    match placement {
        StoragePlacement::Abi { slot } => StorageTarget::Abi { slot },
        StoragePlacement::Arena { offset } => StorageTarget::Arena { offset },
        // Declared inside the kernel as a threadgroup or thread-private
        // array; present in the dense table so indexing is total.
        StoragePlacement::Workgroup | StoragePlacement::Participant => StorageTarget::Staged,
    }
}

/// Validate the selected resource contract of one launch against its
/// sealed binding and storage facts: the direct-binding count, the
/// threadgroup bytes, and the per-participant private bytes. The
/// device-side threadgroup limit is checked against native reflection in
/// `compile_launch`.
fn validate_resources(launch: &SealedLaunch<MetalDialect>) -> Result<(), AssemblyFailure> {
    let resources: &LaunchResources = &launch.resources;
    if std::env::var_os("SEISMIC_DEBUG_LAUNCH").is_some() && launch.id.index() == 0 {
        eprintln!(
            "DEBUG launch 0: inputs={} outputs={} locals={} bindings={} facts={:?} wg_storage={:?} part_storage={:?} resources={:?}",
            launch.kernel.interface().inputs.len(),
            launch.kernel.interface().outputs.len(),
            launch.kernel.interface().locals.len(),
            launch.bindings.len(),
            launch.storage_facts,
            launch.workgroup_storage,
            launch.participant_storage,
            resources,
        );
    }
    if u64::from(resources.direct_bindings) != launch.bindings.len() as u64 {
        return Err(invariant(format!(
            "launch {} selected {} direct bindings but binds {} storages",
            launch.id.index(),
            resources.direct_bindings,
            launch.bindings.len()
        )));
    }
    let mut workgroup_bytes = 0u64;
    let mut participant_bytes = 0u64;
    for fact in &launch.storage_facts {
        match fact {
            StorageFact::Global => {}
            StorageFact::Workgroup { bytes, .. } => workgroup_bytes += *bytes,
            StorageFact::Participant { bytes, .. } => participant_bytes += *bytes,
        }
    }
    if workgroup_bytes != resources.workgroup_bytes {
        return Err(invariant(format!(
            "launch {} selected {} workgroup bytes but binds {workgroup_bytes}",
            launch.id.index(),
            resources.workgroup_bytes
        )));
    }
    if participant_bytes != resources.private_bytes_per_participant {
        return Err(invariant(format!(
            "launch {} selected {} private bytes per participant but binds \
              {participant_bytes}",
            launch.id.index(),
            resources.private_bytes_per_participant
        )));
    }
    Ok(())
}

/// Gather every encoded launch of the tree (tree order, exactly once).
fn collect_launches<'a>(
    steps: &'a [EncodedStep<EncodedLaunch>],
    out: &mut BTreeMap<LaunchIx, &'a EncodedLaunch>,
) {
    for step in steps {
        match step {
            EncodedStep::Launch { launch, encoded } => {
                out.insert(*launch, encoded);
            }
            EncodedStep::Guard { .. } | EncodedStep::Fill { .. } => {}
            EncodedStep::Call { body, .. } => collect_launches(body, out),
            EncodedStep::If {
                then_steps,
                else_steps,
                ..
            } => {
                collect_launches(then_steps, out);
                collect_launches(else_steps, out);
            }
            EncodedStep::Repeat { body, .. } => collect_launches(body, out),
        }
    }
}

fn compile_library(
    device: &crate::runtime::Device,
    rendered: &BTreeMap<LaunchIx, encode::RenderedLaunch>,
) -> Result<Option<Retained<ProtocolObject<dyn MTLLibrary>>>, AssemblyFailure> {
    if rendered.is_empty() {
        // The empty schedule is the valid identity artifact.
        return Ok(None);
    }
    let mut source = String::from(
        "#include <metal_stdlib>\nusing namespace metal;\n#pragma clang fp contract(off)\n\n",
    );
    for launch in rendered.values() {
        source.push_str(&launch.source);
    }
    let source = NSString::from_str(&source);
    let options = MTLCompileOptions::new();
    options.setLanguageVersion(device.language_version().native());
    // Keep the macOS 13 API floor. Default fast math may reassociate
    // explicitly ordered operations and erase publication casts.
    #[allow(deprecated)]
    options.setFastMathEnabled(false);
    device
        .handle()
        .newLibraryWithSource_options_error(&source, Some(&options))
        .map(Some)
        .map_err(|error| {
            AssemblyFailure::Toolchain(ToolchainReport(format!(
                "Metal compile failed: {}",
                error.localizedDescription()
            )))
        })
}

/// One compiled launch plus its reflected native facts.
struct CompiledLaunch {
    launch: NativeLaunch,
    execution_width: u64,
    max_threads: u64,
}

/// Compile one rendered launch and fold its direct binding program.
fn compile_launch(
    device: &crate::runtime::Device,
    library: &Option<Retained<ProtocolObject<dyn MTLLibrary>>>,
    plan: &seismic_realization::physical::PhysicalPlan<MetalDialect>,
    rendered: &encode::RenderedLaunch,
    launch: &SealedLaunch<MetalDialect>,
) -> Result<CompiledLaunch, AssemblyFailure> {
    let library = library.as_ref().ok_or_else(|| {
        invariant(format!(
            "launch {} has a rendered source but no compiled library",
            rendered.id.index()
        ))
    })?;
    let name = NSString::from_str(&rendered.kernel);
    let function = library.newFunctionWithName(&name).ok_or_else(|| {
        invariant(format!(
            "kernel `{}` not found in the compiled library",
            rendered.kernel
        ))
    })?;
    let state = device
        .handle()
        .newComputePipelineStateWithFunction_error(&function)
        .map_err(|error| {
            AssemblyFailure::Toolchain(ToolchainReport(format!(
                "pipeline creation failed: {}",
                error.localizedDescription()
            )))
        })?;
    // The selected resource contract, validated against native
    // reflection: the threadgroup bytes the pipeline declares must fit the
    // device the artifact owns.
    if rendered.workgroup_bytes > device.max_threadgroup_bytes() {
        return Err(invariant(format!(
            "kernel `{}` declares {} threadgroup bytes; the device offers {}",
            rendered.kernel,
            rendered.workgroup_bytes,
            device.max_threadgroup_bytes()
        )));
    }
    let execution_width = state.threadExecutionWidth() as u64;
    let max_threads = state.maxTotalThreadsPerThreadgroup() as u64;
    Ok(CompiledLaunch {
        launch: NativeLaunch {
            kernel: rendered.kernel.clone(),
            state,
            workgroup_bytes: rendered.workgroup_bytes,
            work_items: launch.work_items.clone(),
            participants: launch.participants.clone(),
            workgroups: launch.workgroups.clone(),
            serialized: rendered.serialized,
            bindings: fold_bindings(rendered, plan),
            resources: launch.resources.clone(),
        },
        execution_width,
        max_threads,
    })
}

/// The direct binding program of one launch: absolute Metal argument
/// indices resolved to sealed storage targets, scalar sources, the
/// evaluated total, and the system blocks.
fn fold_bindings(
    rendered: &encode::RenderedLaunch,
    plan: &seismic_realization::physical::PhysicalPlan<MetalDialect>,
) -> Vec<NativeBinding> {
    let mut bindings = Vec::new();
    for binding in &rendered.bindings {
        match binding {
            RenderedBinding::Storage {
                index,
                storage,
                access: _,
            } => bindings.push(NativeBinding::Storage {
                index: *index,
                target: storage_target_of(plan.storages()[*storage].placement),
            }),
            RenderedBinding::Scalar {
                index, source, dtype,
            } => bindings.push(NativeBinding::Scalar {
                index: *index,
                source: source.clone(),
                dtype: *dtype,
            }),
        }
    }
    bindings.push(NativeBinding::Total {
        index: rendered.total_index,
    });
    for (index, extent) in &rendered.extent_args {
        bindings.push(NativeBinding::Extent {
            index: *index,
            value: plan.runtime_extent(*extent).clone(),
        });
    }
    if let Some(index) = rendered.slots_index {
        bindings.push(NativeBinding::Slots { index });
    }
    bindings.push(NativeBinding::Results {
        index: rendered.results_index,
    });
    if let Some(index) = rendered.status_index {
        bindings.push(NativeBinding::Status { index });
    }
    bindings
}

// ---------------------------------------------------------------------------
// Folded join/carry records
// ---------------------------------------------------------------------------

/// The writable side of one sealed join/carry value. A scalar destination
/// must be an executor slot or a result field (the route seal never joins
/// into a caller-owned or invocation-owned scalar); a tensor destination
/// is its first plane's storage with its capacity bytes.
fn fold_destination(
    value: &SealedValue,
    plan: &seismic_realization::physical::PhysicalPlan<MetalDialect>,
) -> Result<CopyDestination, AssemblyFailure> {
    match value {
        SealedValue::Scalar(source) => {
            let destination = match source {
                ScalarSource::Abi { .. } | ScalarSource::Invocation(_) => {
                    return Err(invariant(
                        "a scalar join or carry targets a caller-owned or invocation-owned \
                         scalar; the route seal never routes one there",
                    ))
                }
                ScalarSource::Executor(slot) => ScalarDestination::Slot(*slot),
                ScalarSource::Result(field) => ScalarDestination::Result(*field),
            };
            Ok(CopyDestination::Scalar { destination })
        }
        SealedValue::Tensor(views) => {
            let storage = views.first().storage;
            Ok(CopyDestination::Tensor {
                storage,
                bytes: plan.storages()[storage].bytes,
            })
        }
    }
}

/// The readable side of one sealed join/carry value.
fn fold_source(
    value: &SealedValue,
    plan: &seismic_realization::physical::PhysicalPlan<MetalDialect>,
) -> Result<CopySource, AssemblyFailure> {
    match value {
        SealedValue::Scalar(source) => Ok(CopySource::Scalar {
            source: source.clone(),
        }),
        SealedValue::Tensor(views) => {
            let storage = views.first().storage;
            Ok(CopySource::Tensor {
                storage,
                bytes: plan.storages()[storage].bytes,
            })
        }
    }
}

/// One branch's joins, folded from the sealed schedule.
fn fold_joins(
    joins: &[seismic_realization::physical::SealedJoin],
    plan: &seismic_realization::physical::PhysicalPlan<MetalDialect>,
) -> Result<Vec<NativeJoin>, AssemblyFailure> {
    let mut folded = Vec::with_capacity(joins.len());
    for join in joins {
        folded.push(NativeJoin {
            then_source: fold_source(&join.then_value, plan)?,
            else_source: fold_source(&join.else_value, plan)?,
            destination: fold_destination(&join.joined, plan)?,
        });
    }
    Ok(folded)
}

/// One repeat's carries, folded from the sealed schedule.
fn fold_carries(
    carries: &[seismic_realization::physical::SealedCarry],
    plan: &seismic_realization::physical::PhysicalPlan<MetalDialect>,
) -> Result<Vec<NativeCarry>, AssemblyFailure> {
    let mut folded = Vec::with_capacity(carries.len());
    for carry in carries {
        folded.push(NativeCarry {
            initial: fold_source(&carry.initial, plan)?,
            update: fold_source(&carry.update, plan)?,
            current: fold_destination(&carry.current, plan)?,
            result: fold_destination(&carry.result, plan)?,
        });
    }
    Ok(folded)
}

/// The dense guard→status pairing and its per-field reverse (a
/// kernel-`Check` field legitimately has no dominating guard). Every
/// sealed guard is a schedule guard step, so the forward table is dense by
/// `GuardIx`.
fn fold_guard_pairing(
    plan: &seismic_realization::physical::PhysicalPlan<MetalDialect>,
) -> (Vec<StatusFieldIx>, Vec<Option<GuardIx>>) {
    let field_count = plan.status_fields().len();
    let mut forward: Vec<StatusFieldIx> = Vec::new();
    let mut reverse = vec![None; field_count];
    fn walk(
        steps: &[PhysicalStep<MetalDialect>],
        forward: &mut Vec<StatusFieldIx>,
        reverse: &mut Vec<Option<GuardIx>>,
    ) {
        for step in steps {
            match step {
                PhysicalStep::Guard(guard) => {
                    if reverse[guard.status.index()].is_none() {
                        reverse[guard.status.index()] = Some(guard.id);
                    }
                    if forward.len() <= guard.id.index() {
                        forward.resize(guard.id.index() + 1, guard.status);
                    }
                    forward[guard.id.index()] = guard.status;
                }
                PhysicalStep::Launch(_) | PhysicalStep::Fill(_) => {}
                PhysicalStep::Call(call) => walk(&call.body.steps, forward, reverse),
                PhysicalStep::If(branch) => {
                    walk(&branch.then_schedule.steps, forward, reverse);
                    walk(&branch.else_schedule.steps, forward, reverse);
                }
                PhysicalStep::Repeat(repeat) => walk(&repeat.body.steps, forward, reverse),
            }
        }
    }
    walk(&plan.schedule().steps, &mut forward, &mut reverse);
    (forward, reverse)
}

// ---------------------------------------------------------------------------
// The native tree
// ---------------------------------------------------------------------------

/// Build the native tree by walking the physical schedule and the encoded
/// tree in lockstep (they mirror each other exactly; the physical steps
/// carry the payloads, the encoded steps the launch identities). Any
/// divergence is a `CompilerInvariant` defect, never silent.
fn native_steps(
    plan: &seismic_realization::physical::PhysicalPlan<MetalDialect>,
    physical: &[PhysicalStep<MetalDialect>],
    encoded: &[EncodedStep<EncodedLaunch>],
) -> Result<Vec<NativeStep>, AssemblyFailure> {
    if physical.len() != encoded.len() {
        return Err(invariant(format!(
            "the physical schedule has {} steps but the encoded tree has {}",
            physical.len(),
            encoded.len()
        )));
    }
    let mut out = Vec::with_capacity(encoded.len());
    for (physical, encoded) in physical.iter().zip(encoded) {
        out.push(match (physical, encoded) {
            (PhysicalStep::Launch(sealed), EncodedStep::Launch { launch, .. }) => {
                if launch.index() != sealed.id.index() {
                    return Err(invariant(
                        "the encoded and physical launch identities disagree",
                    ));
                }
                NativeStep::Launch { launch: *launch }
            }
            (PhysicalStep::Guard(guard), EncodedStep::Guard { .. }) => NativeStep::Guard {
                guard: guard.id,
                obligation: guard.obligation.clone(),
                status: guard.status,
                predicate: guard.predicate.clone(),
            },
            (PhysicalStep::Call(call), EncodedStep::Call { body, .. }) => {
                NativeStep::Call(native_steps(plan, &call.body.steps, body)?)
            }
            (
                PhysicalStep::If(branch),
                EncodedStep::If {
                    then_steps,
                    else_steps,
                    ..
                },
            ) => NativeStep::If {
                condition: branch.condition.clone(),
                then_steps: native_steps(plan, &branch.then_schedule.steps, then_steps)?,
                else_steps: native_steps(plan, &branch.else_schedule.steps, else_steps)?,
                joins: fold_joins(&branch.joins, plan)?,
            },
            (PhysicalStep::Repeat(repeat), EncodedStep::Repeat { body, .. }) => {
                NativeStep::Repeat {
                    start: repeat.start.clone(),
                    end: repeat.end.clone(),
                    bound: repeat.bound.clone(),
                    binder: repeat.binder,
                    body: native_steps(plan, &repeat.body.steps, body)?,
                    carries: fold_carries(&repeat.carries, plan)?,
                }
            }
            (PhysicalStep::Fill(fill), EncodedStep::Fill { storage }) => {
                if storage.index() != fill.storage.index() {
                    return Err(invariant(
                        "the encoded fill names a storage the schedule does not",
                    ));
                }
                let StoragePlacement::Arena { offset } =
                    plan.storages()[fill.storage].placement
                else {
                    return Err(invariant(
                        "a fill names a non-arena storage; fills zero internal residences \
                         only",
                    ));
                };
                NativeStep::Fill {
                    offset,
                    bytes: fill.bytes,
                }
            }
            _ => {
                return Err(invariant(
                    "the encoded tree diverged from the physical schedule at one step",
                ))
            }
        });
    }
    Ok(out)
}
