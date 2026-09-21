//! Native assembly of the encoded CUDA plan: compile every launch
//! (PTX→cubin→module), reflect native facts against their declared
//! domains, fold them into direct handles, and seal one native tree that
//! mirrors the physical schedule exactly once.
//!
//! The artifact owns its driver resources: every loaded module retains the
//! private context, and dropping the artifact unloads them. An empty
//! schedule is the valid identity artifact (no launches, no allocations
//! beyond the compiler-owned blocks).
//!
//! Failure classes: a reflected fact outside its declared domain, or
//! another contradicted invariant, is `CompilerInvariant` (a B1Cuda
//! defect); PTX compilation and image loading are `Toolchain`; driver and
//! context operations are `SystemPreparation`.

use crate::encode::{CudaLaunch, MAX_KERNEL_PARAMETER_BYTES};
use crate::intrinsics::Dialect;
use crate::runtime::{Device, NativeResources};
use seismic_compiler::pipeline::{AssemblyFailure, EncodedPlan, EncodedStep};
use seismic_lang::types::RuntimeExtentId;
use seismic_realization::failure::{CompilerDefect, Package};
use seismic_realization::ids::{BranchIx, DenseIndex, LaunchIx, NativeFactIx, RepeatIx};
use seismic_realization::invocation::CheckedInvocationExpr;
use seismic_realization::physical::{
    ExecutionExpr, PhysicalPlan, PhysicalStep, ScalarSource, SealedCarry, SealedFill, SealedGuard,
    SealedJoin, SealedLaunch, StorageFact, StoragePlacement,
};
use seismic_realization::strategy::NativeFactKind;
use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

/// One encoded launch's compiled form: the direct function handle, its
/// retained encoded launch, and the reflected native resources.
pub struct NativeLaunch {
    /// Dense launch identity (the encoded tree names it).
    pub id: LaunchIx,
    /// The driver function handle; valid while `module` lives.
    pub function: crate::driver::Handle,
    /// The loaded module retaining the private context.
    /// Ownership only: retains the loaded module (and its context) for the
    /// function handle's lifetime; never read.
    #[allow(dead_code)]
    pub(crate) module: Rc<crate::driver::Module>,
    pub launch: CudaLaunch,
    pub native: NativeResources,
}

/// Mirror of one resolved storage for the executor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StorageMirror {
    pub placement: StoragePlacement,
    pub bytes: u64,
    pub alignment: u64,
}

/// One step of the sealed native tree: the physical schedule mirrored
/// exactly once, with every fact the executor needs inlined (no lookup by
/// id at execution).
#[derive(Clone, Debug)]
pub enum NativeStep {
    Launch(LaunchIx),
    Guard(SealedGuard),
    Call(Vec<NativeStep>),
    If {
        branch: BranchIx,
        condition: ScalarSource,
        then_steps: Vec<NativeStep>,
        else_steps: Vec<NativeStep>,
        joins: Vec<SealedJoin>,
    },
    Repeat {
        repeat: RepeatIx,
        start: ExecutionExpr,
        end: ExecutionExpr,
        bound: ExecutionExpr,
        binder: seismic_realization::ids::ScalarSlotIx,
        body: Vec<NativeStep>,
        carries: Vec<SealedCarry>,
    },
    /// A host-side fill: zeroes `bytes` bytes of one storage.
    Fill(SealedFill),
}

/// The sealed native artifact: one compiled handle per launch, the sealed
/// execution tree, the storage mirror, the reflected native facts, and the
/// native temporaries' sizes. Construction is private to `assemble`.
pub struct NativeArtifact {
    name: String,
    launches: Vec<NativeLaunch>,
    root: Vec<NativeStep>,
    storages: Vec<StorageMirror>,
    /// Every runtime extent any launch reads (the value-block domain).
    runtime_extents: Vec<RuntimeExtentId>,
    /// Grid-barrier scratch words (two per site).
    barrier_words: usize,
    /// Reflected native facts, dense per `NativeFactIx`.
    native_facts: Vec<u64>,
    toolchain_fingerprint: String,
}

impl NativeArtifact {
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn launches(&self) -> &[NativeLaunch] {
        &self.launches
    }
    pub fn root(&self) -> &[NativeStep] {
        &self.root
    }
    pub fn storages(&self) -> &[StorageMirror] {
        &self.storages
    }
    pub fn runtime_extents(&self) -> &[RuntimeExtentId] {
        &self.runtime_extents
    }
    pub fn barrier_words(&self) -> usize {
        self.barrier_words
    }
    /// One reflected native fact; dense and in-bounds by construction.
    pub fn native_fact(&self, index: NativeFactIx) -> u64 {
        self.native_facts[index.index()]
    }
    pub fn toolchain_fingerprint(&self) -> &str {
        &self.toolchain_fingerprint
    }
    /// The launch of one dense index (the encoded tree names it).
    pub fn launch(&self, id: LaunchIx) -> &NativeLaunch {
        &self.launches[id.index()]
    }

    /// The solved participant constant of one launch (the sealed
    /// `participants` expression is the seal's constant node); a
    /// contradicted constant shape is a typed B1Cuda defect.
    pub fn participants_of(
        &self,
        plan: &PhysicalPlan<Dialect>,
        id: LaunchIx,
    ) -> Result<u64, AssemblyFailure> {
        solved_constant(
            plan,
            &self.launches[id.index()].launch.participants,
            "the participant count",
        )
    }
}

fn solved_constant(
    plan: &PhysicalPlan<Dialect>,
    expr: &ExecutionExpr,
    what: &str,
) -> Result<u64, AssemblyFailure> {
    let ExecutionExpr::Invocation(id) = expr else {
        return Err(defect(format!(
            "{what} of a sealed launch is not the seal's constant node"
        )));
    };
    match plan.contract().derived()[*id] {
        CheckedInvocationExpr::Const(value) => Ok(value),
        _ => Err(defect(format!(
            "{what} of a sealed launch is not a solved constant"
        ))),
    }
}

/// Validate the selected resource contract of one launch against its
/// sealed binding and storage facts: the direct-binding count (the
/// parameter-byte ABI bound is checked after compilation), the dynamic
/// workgroup bytes the encoder declared for the `.extern .shared`
/// declaration, and the per-participant private bytes. The block-thread
/// bound is validated against native reflection in `reflect_fact`.
fn validate_resources(launch: &SealedLaunch<Dialect>) -> Result<(), AssemblyFailure> {
    let resources = &launch.resources;
    if u64::from(resources.direct_bindings) != launch.bindings.len() as u64 {
        return Err(defect(format!(
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
        return Err(defect(format!(
            "launch {} selected {} workgroup bytes but binds {workgroup_bytes}",
            launch.id.index(),
            resources.workgroup_bytes
        )));
    }
    if participant_bytes != resources.private_bytes_per_participant {
        return Err(defect(format!(
            "launch {} selected {} private bytes per participant but binds \
             {participant_bytes}",
            launch.id.index(),
            resources.private_bytes_per_participant
        )));
    }
    Ok(())
}

fn defect(invariant: impl Into<String>) -> AssemblyFailure {
    AssemblyFailure::CompilerInvariant(CompilerDefect::new(Package::B1Cuda, invariant))
}

/// Assemble the encoded plan on one CUDA device: compile each launch once,
/// reflect and validate every declared native fact, and seal the native
/// tree mirroring the physical schedule exactly once. An empty schedule is
/// the valid identity artifact.
pub fn assemble(
    device: &Device,
    encoded: EncodedPlan<Dialect, CudaLaunch>,
) -> Result<NativeArtifact, AssemblyFailure> {
    let plan = encoded.physical();
    let steps = encoded.steps();
    let mut facts = NativeFactTable::new(plan);
    let mut launches = Vec::new();
    let mut runtime_extents: Vec<RuntimeExtentId> = Vec::new();
    let mut barrier_words = 0usize;
    let mut by_id: BTreeMap<usize, CudaLaunch> = BTreeMap::new();
    for step in steps {
        collect_launches(step, &mut by_id);
    }
    for (ordinal, encoded_launch) in by_id.into_iter() {
        if ordinal != launches.len() {
            return Err(defect(format!(
                "launch {} appears out of tree order (expected ordinal {})",
                ordinal,
                launches.len()
            )));
        }
        let sealed_launches = plan.launches();
        let Some(sealed) = sealed_launches.get(ordinal) else {
            return Err(defect(format!(
                "the encoded tree names launch {ordinal} but the sealed plan has {}",
                plan.launches().len()
            )));
        };
        if sealed.id.index() != ordinal {
            return Err(defect(
                "the sealed plan's launch order disagrees with its dense launch identities",
            ));
        }
        validate_resources(sealed)?;
        let native = device.compile_launch(&encoded_launch)?;
        // Reflect every declared native fact against its domain and fold
        // it into the dense fact table.
        for declaration in &sealed.native_facts {
            let reflected = reflect_fact(
                device,
                &native,
                declaration.kind,
                solved_constant(plan, &sealed.participants, "the participant count")?,
            )?;
            facts.record(declaration, reflected)?;
        }
        for id in encoded_launch.runtime_extents.iter().copied() {
            if !runtime_extents.contains(&id) {
                runtime_extents.push(id);
            }
        }
        barrier_words += encoded_launch.grid_barriers * 2;
        launches.push(NativeLaunch {
            id: LaunchIx::from_index(ordinal),
            function: native.function,
            module: Rc::new(native.module),
            launch: encoded_launch,
            native: native.resources,
        });
    }
    if launches.len() != plan.launches().len() {
        return Err(defect(format!(
            "the encoded tree carries {} launches but the sealed plan has {}",
            launches.len(),
            plan.launches().len()
        )));
    }
    // The mechanical parameter count is bounded by the solver's
    // `max_direct_bindings` constraint; an overflow contradicts it.
    for launch in &launches {
        if launch.launch.params.len() * std::mem::size_of::<u64>() > MAX_KERNEL_PARAMETER_BYTES {
            return Err(defect(format!(
                "launch {} needs {} parameter bytes beyond the kernel ABI",
                launch.id.index(),
                launch.launch.params.len() * std::mem::size_of::<u64>()
            )));
        }
    }
    let root = mirror_steps(&steps, &plan.schedule().steps)?;
    let storages = plan
        .storages()
        .ids()
        .map(|id| {
            let storage = &plan.storages()[id];
            StorageMirror {
                placement: storage.placement,
                bytes: storage.bytes,
                alignment: storage.alignment,
            }
        })
        .collect();
    Ok(NativeArtifact {
        name: plan.identity().entry.clone(),
        launches,
        root,
        storages,
        runtime_extents,
        barrier_words,
        native_facts: facts.finish()?,
        toolchain_fingerprint: plan.identity().toolchain_fingerprint.clone(),
    })
}

/// Collect the encoded launches in tree order.
fn collect_launches(step: &EncodedStep<CudaLaunch>, out: &mut BTreeMap<usize, CudaLaunch>) {
    match step {
        EncodedStep::Launch { launch, encoded } => {
            out.insert(launch.index(), encoded.clone());
        }
        EncodedStep::Guard { .. } | EncodedStep::Fill { .. } => {}
        EncodedStep::Call { body, .. } | EncodedStep::Repeat { body, .. } => {
            for step in body {
                collect_launches(step, out);
            }
        }
        EncodedStep::If {
            then_steps,
            else_steps,
            ..
        } => {
            for step in then_steps.iter().chain(else_steps) {
                collect_launches(step, out);
            }
        }
    }
}

/// Mirror the encoded and physical steps into one native tree; the two
/// trees mirror the same schedule exactly once.
fn mirror_steps(
    encoded: &[EncodedStep<CudaLaunch>],
    physical: &[PhysicalStep<Dialect>],
) -> Result<Vec<NativeStep>, AssemblyFailure> {
    if encoded.len() != physical.len() {
        return Err(defect(format!(
            "the encoded tree has {} steps but the physical schedule has {}",
            encoded.len(),
            physical.len()
        )));
    }
    let mut out = Vec::with_capacity(encoded.len());
    for (encoded, physical) in encoded.iter().zip(physical) {
        let step = match (encoded, physical) {
            (EncodedStep::Launch { launch, .. }, PhysicalStep::Launch(sealed)) => {
                if launch.index() != sealed.id.index() {
                    return Err(defect(
                        "the encoded and physical launch identities disagree",
                    ));
                }
                NativeStep::Launch(*launch)
            }
            (EncodedStep::Guard { guard }, PhysicalStep::Guard(sealed)) => {
                if guard.index() != sealed.id.index() {
                    return Err(defect("the encoded and physical guard identities disagree"));
                }
                NativeStep::Guard(sealed.clone())
            }
            (EncodedStep::Call { body, .. }, PhysicalStep::Call(sealed)) => {
                NativeStep::Call(mirror_steps(body, &sealed.body.steps)?)
            }
            (
                EncodedStep::If {
                    branch,
                    then_steps,
                    else_steps,
                    ..
                },
                PhysicalStep::If(sealed),
            ) => {
                if branch.index() != sealed.id.index() {
                    return Err(defect(
                        "the encoded and physical branch identities disagree",
                    ));
                }
                NativeStep::If {
                    branch: *branch,
                    condition: sealed.condition.clone(),
                    then_steps: mirror_steps(then_steps, &sealed.then_schedule.steps)?,
                    else_steps: mirror_steps(else_steps, &sealed.else_schedule.steps)?,
                    joins: sealed.joins.clone(),
                }
            }
            (EncodedStep::Repeat { repeat, body, .. }, PhysicalStep::Repeat(sealed)) => {
                if repeat.index() != sealed.id.index() {
                    return Err(defect(
                        "the encoded and physical repeat identities disagree",
                    ));
                }
                NativeStep::Repeat {
                    repeat: *repeat,
                    start: sealed.start.clone(),
                    end: sealed.end.clone(),
                    bound: sealed.bound.clone(),
                    binder: sealed.binder,
                    body: mirror_steps(body, &sealed.body.steps)?,
                    carries: sealed.carries.clone(),
                }
            }
            (EncodedStep::Fill { storage }, PhysicalStep::Fill(sealed)) => {
                if storage.index() != sealed.storage.index() {
                    return Err(defect("the encoded and physical fill storages disagree"));
                }
                NativeStep::Fill(*sealed)
            }
            (encoded, physical) => {
                return Err(defect(format!(
                    "the encoded step {} does not mirror the physical step {}",
                    step_name(encoded),
                    physical_name(physical),
                )))
            }
        };
        out.push(step);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Native fact reflection
// ---------------------------------------------------------------------------

/// The dense native-fact table under construction.
struct NativeFactTable {
    values: BTreeMap<usize, u64>,
    total: usize,
}

impl NativeFactTable {
    fn new(plan: &PhysicalPlan<Dialect>) -> Self {
        // `NativeFactIx` is globally dense over every launch's declarations.
        let indices: BTreeSet<usize> = plan
            .launches()
            .iter()
            .flat_map(|launch| launch.native_facts.iter().map(|fact| fact.index.index()))
            .collect();
        // Dense in `[0, total)`; the set's cardinality is the total.
        NativeFactTable {
            values: BTreeMap::new(),
            total: indices.len(),
        }
    }

    fn record(
        &mut self,
        declaration: &seismic_realization::physical::NativeFactDeclaration,
        reflected: u64,
    ) -> Result<(), AssemblyFailure> {
        if reflected < declaration.min || reflected > declaration.max {
            return Err(defect(format!(
                "reflected native fact {kind:?} ({reflected}) lies outside its declared \
                 domain [{min}, {max}]",
                kind = declaration.kind,
                min = declaration.min,
                max = declaration.max,
            )));
        }
        if self
            .values
            .insert(declaration.index.index(), reflected)
            .is_some()
        {
            return Err(defect(
                "two launches reflect the same native fact index (the seal allocates \
                 them globally dense)",
            ));
        }
        Ok(())
    }

    fn finish(self) -> Result<Vec<u64>, AssemblyFailure> {
        let mut out = vec![0u64; self.total];
        for (index, value) in self.values {
            if index >= self.total {
                return Err(defect(format!(
                    "native fact index {index} exceeds the reflected domain {total}",
                    total = self.total
                )));
            }
            out[index] = value;
        }
        Ok(out)
    }
}

/// Reflect one native fact of a compiled launch, validating it against the
/// selected resource contract.
fn reflect_fact(
    device: &Device,
    native: &CompiledLaunch,
    kind: NativeFactKind,
    participants: u64,
) -> Result<u64, AssemblyFailure> {
    match kind {
        NativeFactKind::MaxResidentParticipants => {
            // The resident-participant capacity of this function: its
            // per-multiprocessor block residency times the device's
            // multiprocessor count times the launch's block width.
            let blocks = device.occupancy_blocks(native, participants)?;
            let reflected =
                u64::from(blocks) * u64::from(device.info.multiprocessors) * participants;
            let capacity = native.resources.max_threads_per_block.max(0) as u32;
            if u64::from(capacity) < participants {
                return Err(defect(format!(
                    "the compiled function admits {max} threads per block but the launch \
                     selected {participants} participants",
                    max = native.resources.max_threads_per_block,
                )));
            }
            Ok(reflected)
        }
        NativeFactKind::NativeSubgroupWidth => {
            // The warp is the subgroup on every supported CUDA target.
            let reflected = u64::from(device.info.warp_size);
            if reflected != 32 {
                return Err(defect(format!(
                    "the device reports a {reflected}-lane warp but the CUDA dialect \
                     requires the 32-lane subgroup"
                )));
            }
            Ok(reflected)
        }
    }
}

/// One compiled launch before it is folded into the artifact.
pub(crate) struct CompiledLaunch {
    pub function: crate::driver::Handle,
    pub module: crate::driver::Module,
    pub resources: NativeResources,
}

fn step_name(step: &EncodedStep<CudaLaunch>) -> &'static str {
    match step {
        EncodedStep::Launch { .. } => "launch",
        EncodedStep::Guard { .. } => "guard",
        EncodedStep::Call { .. } => "call",
        EncodedStep::If { .. } => "if",
        EncodedStep::Repeat { .. } => "repeat",
        EncodedStep::Fill { .. } => "fill",
    }
}

fn physical_name(step: &PhysicalStep<Dialect>) -> &'static str {
    match step {
        PhysicalStep::Launch(_) => "launch",
        PhysicalStep::Guard(_) => "guard",
        PhysicalStep::Call(_) => "call",
        PhysicalStep::If(_) => "if",
        PhysicalStep::Repeat(_) => "repeat",
        PhysicalStep::Fill(_) => "fill",
    }
}
