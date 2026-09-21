//! Native assembly: one Cranelift function per encoded launch, reflected
//! native facts validated against their declared domains and folded into
//! the sealed artifact, and one native tree mirroring the physical schedule
//! exactly once.
//!
//! Emission is mechanical over the encoded instruction program: every arm
//! accepts its instruction, arithmetic follows the registry reference model
//! (binary64 computation rounded once at the result dtype, transcendentals
//! through the versioned `seismic_math` host sequences), atomics under
//! `Serialized` are plain load/combine/round/store and under `Device` a
//! compare/exchange loop on the element word, and a dynamic-pull traversal
//! is a worker claim loop over the pull counter. The only failures are
//! toolchain/system conditions and contradicted internal invariants. No
//! emission arm rejects a semantic case; view-side shapes and storage
//! facts come launch-locally from the sealed views (`view.ty`) and
//! `storage_facts`.
//!
//! Per-launch native ABI (the worker entry):
//! `fn(buffers: *const *mut u8, scalars: *mut u64, scratch: *mut u8, participant: u64) -> i32`
//! — the buffer table is the launch's shared bound storages in binding-slot
//! order (participant-scoped storages are addressed from the worker's
//! private scratch instead) plus the status area as one extra entry when
//! the launch writes status; the scalar word table is
//! `[words][runtime extent values][work items][participants]` exactly as the
//! descriptor records.

use crate::encode::{Binding, CheckKind, CarrySlot, EncodedLaunch, Instruction, JoinSlot, LaunchDescriptor};
use crate::intrinsics::CpuDialect;
use crate::{host_symbols, workers};
use cranelift_codegen::ir::{
    self,
    condcodes::{FloatCC, IntCC},
    types, AbiParam, InstBuilder, MemFlags, StackSlotData, Value,
};
use cranelift_codegen::isa::CallConv;
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{default_libcall_names, Linkage, Module};
use seismic_compiler::pipeline::{
    AssemblyFailure, EncodedPlan, EncodedStep, SystemReport, ToolchainReport,
};
use seismic_lang::{
    intrinsics::{AtomicOp, MathOp, PlaneField, ReduceOp as R, ReduceSchema},
    repr,
    syntax::ast::{BinaryOp, UnaryOp},
    types::{DType, ExtentExpr, NonEmpty},
};
use seismic_realization::failure::{CompilerDefect, Package};
use seismic_realization::ids::{LaunchIx, NativeFactIx, StatusFieldIx, StorageIx};
use seismic_realization::kernel::{AtomicMode, ConstantValue, RelOp};
use seismic_realization::physical::{
    ExecutionExpr, PhysicalPlan, PhysicalStep, ScalarSource, SealedCarry, SealedFill, SealedGuard,
    SealedJoin, SealedLaunch,
};
use seismic_realization::residence::StoragePlane;
use seismic_realization::routes::{SliceAxisTemplate, ViewStepKind};
use seismic_realization::strategy::{MappingCatalog, NativeFactKind};
use seismic_realization::physical::StorageFact;
use std::collections::{BTreeMap, BTreeSet};

fn invariant(reason: impl Into<String>) -> AssemblyFailure {
    AssemblyFailure::CompilerInvariant(CompilerDefect::new(Package::B1Cpu, reason.into()))
}

// ---------------------------------------------------------------------------
// The sealed native artifact
// ---------------------------------------------------------------------------

/// The executable CPU artifact: JIT memory, one compiled handle per launch
/// (dense by `LaunchIx`, exactly the schedule's launches in tree order), the
/// folded native facts (dense by `NativeFactIx`), and the native tree that
/// mirrors the physical schedule exactly once. Construction is private to
/// `assemble`; an empty schedule is the valid identity artifact.
pub struct NativeArtifact {
    memory: Option<JITModule>,
    launches: Vec<LaunchHandle>,
    native_facts: Vec<u64>,
    tree: Vec<NativeStep>,
}

/// One compiled launch: its worker entry and its native ABI tables (which
/// carry the launch-local storage split from the sealed `storage_facts`).
#[derive(Clone)]
pub(crate) struct LaunchHandle {
    pub entry: workers::PhaseEntry,
    pub descriptor: LaunchDescriptor,
}

/// The native execution tree, one node per `PhysicalStep`.
#[derive(Clone)]
pub(crate) enum NativeStep {
    /// Direct dense handle: `NativeArtifact::launches[launch.index()]`.
    Launch(LaunchIx),
    Guard(SealedGuard),
    Call(Vec<NativeStep>),
    If {
        condition: ScalarSource,
        then_steps: Vec<NativeStep>,
        else_steps: Vec<NativeStep>,
        joins: Vec<SealedJoin>,
    },
    Repeat {
        start: ExecutionExpr,
        end: ExecutionExpr,
        binder: seismic_realization::ids::ScalarSlotIx,
        carries: Vec<SealedCarry>,
        body: Vec<NativeStep>,
    },
    Fill(SealedFill),
}

impl NativeArtifact {
    /// Launches compiled into this artifact (the sealed plan's launch count).
    pub fn launch_count(&self) -> usize {
        self.launches.len()
    }

    /// One folded native fact (dense by `NativeFactIx`, reflected and
    /// validated at assembly). Direct dense index: the fact table is sized
    /// over every declared fact, and the seal allocates the ids densely
    /// over exactly those declarations.
    pub fn native_fact(&self, index: NativeFactIx) -> u64 {
        self.native_facts[index.index()]
    }

    /// The native execution tree (the executor's schedule).
    pub(crate) fn tree(&self) -> &[NativeStep] {
        &self.tree
    }

    /// One compiled launch handle. Direct dense index: the handle table is
    /// built in tree order, which is `LaunchIx` order, one handle per
    /// sealed launch.
    pub(crate) fn launch(&self, launch: LaunchIx) -> &LaunchHandle {
        &self.launches[launch.index()]
    }
}

impl Drop for NativeArtifact {
    fn drop(&mut self) {
        if let Some(module) = self.memory.take() {
            // No generated pointer escapes its artifact owner.
            unsafe {
                module.free_memory();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Imports (the versioned host sequences)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Import {
    ExpF64,
    LogF64,
    SinF64,
    CosF64,
    Fmax,
    Fmin,
    Fmod,
    RoundTo,
    F16Load,
    F16Store,
}

impl Import {
    fn symbol(self) -> &'static str {
        match self {
            Import::ExpF64 => "seismic_exp_f64",
            Import::LogF64 => "seismic_log_f64",
            Import::SinF64 => "seismic_sin_f64",
            Import::CosF64 => "seismic_cos_f64",
            Import::Fmax => "seismic_fmax",
            Import::Fmin => "seismic_fmin",
            Import::Fmod => "seismic_fmod",
            Import::RoundTo => "seismic_round_to",
            Import::F16Load => "seismic_f16_load",
            Import::F16Store => "seismic_f16_store",
        }
    }

    fn signature(self, call_conv: CallConv) -> ir::Signature {
        let mut signature = ir::Signature::new(call_conv);
        let param = |signature: &mut ir::Signature, ty| signature.params.push(AbiParam::new(ty));
        match self {
            Import::ExpF64 | Import::LogF64 | Import::SinF64 | Import::CosF64 => {
                param(&mut signature, types::F64);
                signature.returns.push(AbiParam::new(types::F64));
            }
            Import::Fmax | Import::Fmin | Import::Fmod => {
                param(&mut signature, types::F64);
                param(&mut signature, types::F64);
                signature.returns.push(AbiParam::new(types::F64));
            }
            Import::RoundTo => {
                param(&mut signature, types::I32);
                param(&mut signature, types::F64);
                signature.returns.push(AbiParam::new(types::F64));
            }
            Import::F16Load => {
                param(&mut signature, types::I32);
                signature.returns.push(AbiParam::new(types::F32));
            }
            Import::F16Store => {
                param(&mut signature, types::F32);
                signature.returns.push(AbiParam::new(types::I32));
            }
        }
        signature
    }
}

// ---------------------------------------------------------------------------
// Assembly
// ---------------------------------------------------------------------------

/// Assemble one encoded plan into the sealed native artifact: compile every
/// launch, reflect and validate the native facts, and mirror the physical
/// schedule exactly once. An empty schedule is the valid identity artifact.
pub fn assemble(
    encoded: EncodedPlan<CpuDialect, EncodedLaunch>,
    policy: &crate::codegen::Policy,
    catalog: &crate::catalog::CpuCatalog,
) -> Result<NativeArtifact, AssemblyFailure> {
    let plan = encoded.physical();
    let call_conv = policy.call_conv();
    let isa = policy
        .target()
        .map_err(|reason| AssemblyFailure::SystemPreparation(SystemReport(reason)))?
        .isa;
    let mut jit = JITBuilder::with_isa(isa, default_libcall_names());
    for (name, address) in host_symbols() {
        jit.symbol(name, address);
    }
    let mut module = JITModule::new(jit);
    if module.target_config().pointer_type() != types::I64 {
        return Err(AssemblyFailure::SystemPreparation(SystemReport(
            "the CPU backend requires 64-bit pointers".into(),
        )));
    }

    // Compile every launch (tree order) and build the native tree that
    // mirrors the physical schedule exactly once.
    let mut compiled: Vec<CompiledLaunch> = Vec::new();
    let tree = build_tree(encoded.steps(), &plan.schedule().steps, &plan, &mut compiled, call_conv)?;

    // Declare, define, and finalize every compiled function.
    let mut finalized: Vec<(cranelift_module::FuncId, LaunchDescriptor)> = Vec::new();
    for launch in compiled {
        let CompiledLaunch {
            id,
            function,
            imports,
            descriptor,
        } = launch;
        let mut context = module.make_context();
        context.func = function;
        for (reference, import) in &imports {
            let signature = import.signature(call_conv);
            let imported = module
                .declare_function(import.symbol(), Linkage::Import, &signature)
                .map_err(|error| AssemblyFailure::Toolchain(ToolchainReport(error.to_string())))?;
            let native = module.declare_func_in_func(imported, &mut context.func);
            context.func.dfg.ext_funcs[*reference] = context.func.dfg.ext_funcs[native].clone();
        }
        let declared = module
            .declare_function(
                &format!("seismic_cpu_launch_{}", id.index()),
                Linkage::Local,
                &context.func.signature,
            )
            .map_err(|error| AssemblyFailure::Toolchain(ToolchainReport(error.to_string())))?;
        context.func.name = ir::UserFuncName::user(0, declared.as_u32());
        module
            .define_function(declared, &mut context)
            .map_err(|error| {
                AssemblyFailure::Toolchain(ToolchainReport(format!(
                    "CPU compilation of launch {}: {error:?}",
                    id.index()
                )))
            })?;
        module.clear_context(&mut context);
        finalized.push((declared, descriptor));
    }
    module
        .finalize_definitions()
        .map_err(|error| AssemblyFailure::Toolchain(ToolchainReport(error.to_string())))?;
    // Launch handles are dense in tree order, which is `LaunchIx` order.
    let launches = finalized
        .into_iter()
        .map(|(declared, descriptor)| LaunchHandle {
            entry: unsafe {
                std::mem::transmute::<*const u8, workers::PhaseEntry>(
                    module.get_finalized_function(declared),
                )
            },
            descriptor,
        })
        .collect();

    // Reflect and validate every declared native fact, then fold it.
    let limits = catalog.limits();
    let mut fact_count = 0usize;
    for launch in plan.launches() {
        for fact in &launch.native_facts {
            fact_count = fact_count.max(fact.index.index() + 1);
        }
    }
    let mut native_facts = vec![0u64; fact_count];
    let mut reflected_indices: BTreeSet<NativeFactIx> = BTreeSet::new();
    for launch in plan.launches() {
        for fact in &launch.native_facts {
            if !reflected_indices.insert(fact.index) {
                return Err(invariant(
                    "two launches declare the same native fact index (the seal allocates \
                     them globally dense)",
                ));
            }
            let value = match fact.kind {
                NativeFactKind::MaxResidentParticipants => limits.max_participants,
                NativeFactKind::NativeSubgroupWidth => 1,
            };
            if value < fact.min || value > fact.max {
                return Err(invariant(format!(
                    "reflected native fact {:?} of launch {} is {value}, outside its declared \
                     domain [{}, {}]",
                    fact.kind,
                    launch.id.index(),
                    fact.min,
                    fact.max
                )));
            }
            native_facts[fact.index.index()] = value;
        }
    }

    Ok(NativeArtifact {
        memory: Some(module),
        launches,
        native_facts,
        tree,
    })
}

/// One compiled launch awaiting JIT definition: its function, imports, and
/// ABI descriptor (which carries the launch-local storage split).
struct CompiledLaunch {
    id: LaunchIx,
    function: ir::Function,
    imports: Vec<(ir::FuncRef, Import)>,
    descriptor: LaunchDescriptor,
}

/// Build the native tree in lockstep with the physical schedule, compiling
/// every launch on the way.
fn build_tree(
    encoded: &[EncodedStep<EncodedLaunch>],
    physical: &[PhysicalStep<CpuDialect>],
    plan: &PhysicalPlan<CpuDialect>,
    compiled: &mut Vec<CompiledLaunch>,
    call_conv: CallConv,
) -> Result<Vec<NativeStep>, AssemblyFailure> {
    if encoded.len() != physical.len() {
        return Err(invariant(
            "the encoded tree does not mirror the physical schedule",
        ));
    }
    let mut steps = Vec::with_capacity(physical.len());
    for (encoded_step, physical_step) in encoded.iter().zip(physical) {
        let step = match (encoded_step, physical_step) {
            (EncodedStep::Launch { launch, encoded }, PhysicalStep::Launch(sealed)) => {
                if *launch != sealed.id {
                    return Err(invariant(
                        "the encoded tree names a launch the schedule does not",
                    ));
                }
                // The selected resource contract, validated identically
                // against the sealed storage facts.
                validate_resources(sealed, &encoded.descriptor)?;
                let emission = emit_launch(plan, encoded, call_conv)?;
                compiled.push(CompiledLaunch {
                    id: sealed.id,
                    function: emission.function,
                    imports: emission.imports,
                    descriptor: encoded.descriptor.clone(),
                });
                NativeStep::Launch(sealed.id)
            }
            (EncodedStep::Guard { .. }, PhysicalStep::Guard(guard)) => {
                NativeStep::Guard(guard.clone())
            }
            (EncodedStep::Call { body, .. }, PhysicalStep::Call(call)) => {
                NativeStep::Call(build_tree(body, &call.body.steps, plan, compiled, call_conv)?)
            }
            (EncodedStep::If { then_steps, else_steps, .. }, PhysicalStep::If(branch)) => {
                NativeStep::If {
                    condition: branch.condition.clone(),
                    then_steps: build_tree(
                        then_steps,
                        &branch.then_schedule.steps,
                        plan,
                        compiled,
                        call_conv,
                    )?,
                    else_steps: build_tree(
                        else_steps,
                        &branch.else_schedule.steps,
                        plan,
                        compiled,
                        call_conv,
                    )?,
                    joins: branch.joins.clone(),
                }
            }
            (EncodedStep::Repeat { body, .. }, PhysicalStep::Repeat(repeat)) => NativeStep::Repeat {
                start: repeat.start.clone(),
                end: repeat.end.clone(),
                binder: repeat.binder,
                carries: repeat.carries.clone(),
                body: build_tree(body, &repeat.body.steps, plan, compiled, call_conv)?,
            },
            (EncodedStep::Fill { storage }, PhysicalStep::Fill(fill)) => {
                if *storage != fill.storage {
                    return Err(invariant(
                        "the encoded fill names a storage the schedule does not",
                    ));
                }
                NativeStep::Fill(*fill)
            }
            _ => {
                return Err(invariant(
                    "the encoded tree disagrees with the physical schedule at one step",
                ))
            }
        };
        steps.push(step);
    }
    Ok(steps)
}

/// Validate the selected resource contract of one launch against its
/// sealed storage facts and encoded split: binding count, workgroup bytes
/// (the target declares none), per-participant scratch span, and subgroup
/// width. Workgroup-scope storage does not exist on this target
/// (`TargetLimits::max_workgroup_bytes` is 0).
fn validate_resources(
    launch: &SealedLaunch<CpuDialect>,
    descriptor: &LaunchDescriptor,
) -> Result<(), AssemblyFailure> {
    let resources = &launch.resources;
    if u64::from(resources.direct_bindings) != launch.bindings.len() as u64 {
        return Err(invariant(format!(
            "launch {} selected {} direct bindings but binds {} storages",
            launch.id.index(),
            resources.direct_bindings,
            launch.bindings.len()
        )));
    }
    if let Some(width) = resources.required_subgroup_width {
        if width != 1 {
            return Err(invariant(format!(
                "launch {} requires a subgroup width of {width} on a target with width 1",
                launch.id.index()
            )));
        }
    }
    let mut workgroup_bytes = 0u64;
    for fact in &launch.storage_facts {
        if let StorageFact::Workgroup { bytes, .. } = fact {
            if *bytes > 0 {
                return Err(invariant(format!(
                    "launch {} binds workgroup storage of {bytes} bytes on a target that \
                     declares none",
                    launch.id.index()
                )));
            }
            workgroup_bytes += *bytes;
        }
    }
    if workgroup_bytes != resources.workgroup_bytes {
        return Err(invariant(format!(
            "launch {} selected {} workgroup bytes but binds {workgroup_bytes}",
            launch.id.index(),
            resources.workgroup_bytes
        )));
    }
    if descriptor.scratch_bytes != resources.private_bytes_per_participant {
        return Err(invariant(format!(
            "launch {} selected {} private bytes per participant but its participant \
             storages span {}",
            launch.id.index(),
            resources.private_bytes_per_participant,
            descriptor.scratch_bytes
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// One launch's emission
// ---------------------------------------------------------------------------

struct LaunchEmission {
    function: ir::Function,
    imports: Vec<(ir::FuncRef, Import)>,
}

/// The per-launch emission context: everything the instruction program's
/// arms need, with no fallible structural lookup.
struct Emit<'a> {
    plan: &'a PhysicalPlan<CpuDialect>,
    encoded: &'a EncodedLaunch,
    /// Bound storage -> buffer-table index (shared storages only).
    table: BTreeMap<StorageIx, usize>,
    /// Participant-scoped storage -> scratch byte offset.
    participant: BTreeMap<StorageIx, u64>,
    import_refs: BTreeMap<Import, ir::FuncRef>,
    builder: FunctionBuilder<'a>,
    table_ptr: Value,
    scalars_ptr: Value,
    scratch_ptr: Value,
    status_ptr: Option<Value>,
    storage_ptrs: BTreeMap<StorageIx, Value>,
    /// Kernel-local SSA values by binding position (f64 S-values).
    locals: BTreeMap<usize, Value>,
    /// Serial-loop binder values by binding position.
    loop_vars: BTreeMap<usize, Value>,
    /// Delinearized coordinates of the current point (i64, one per axis).
    coords: Vec<Value>,
    /// The linear coordinate of the current participant visit.
    linear: Value,
    /// The participant count of this launch.
    participants: Value,
}

fn emit_launch(
    plan: &PhysicalPlan<CpuDialect>,
    encoded: &EncodedLaunch,
    call_conv: CallConv,
) -> Result<LaunchEmission, AssemblyFailure> {
    let descriptor = &encoded.descriptor;
    // The buffer-table index of every shared storage (binding-slot order).
    let table: BTreeMap<StorageIx, usize> = descriptor
        .table_order
        .iter()
        .enumerate()
        .map(|(index, storage)| (*storage, index))
        .collect();
    // Every bound storage's resolved layout must fit its allocation.
    for storage in descriptor.table_order.iter().copied().chain(
        descriptor.participant_spans.keys().copied(),
    ) {
        check_layout(plan, storage)?;
    }
    let mut signature = ir::Signature::new(call_conv);
    signature
        .params
        .extend((0..4).map(|_| AbiParam::new(types::I64)));
    signature.returns.push(AbiParam::new(types::I32));
    let mut function = ir::Function::with_name_signature(ir::UserFuncName::user(0, 0), signature);
    let mut context = FunctionBuilderContext::new();
    let mut builder = FunctionBuilder::new(&mut function, &mut context);
    let entry_block = builder.create_block();
    builder.append_block_params_for_function_params(entry_block);
    builder.switch_to_block(entry_block);
    builder.seal_block(entry_block);
    let table_ptr = builder.block_params(entry_block)[0];
    let scalars_ptr = builder.block_params(entry_block)[1];
    let scratch_ptr = builder.block_params(entry_block)[2];
    let participant = builder.block_params(entry_block)[3];

    // Import references (created once, before any block needs them).
    let mut import_refs: BTreeMap<Import, ir::FuncRef> = BTreeMap::new();
    let mut imports: Vec<(ir::FuncRef, Import)> = Vec::new();
    for import in [
        Import::ExpF64,
        Import::LogF64,
        Import::SinF64,
        Import::CosF64,
        Import::Fmax,
        Import::Fmin,
        Import::Fmod,
        Import::RoundTo,
        Import::F16Load,
        Import::F16Store,
    ] {
        let signature = import.signature(call_conv);
        let sig_ref = builder.import_signature(signature);
        let data = ir::ExtFuncData {
            name: ir::ExternalName::user(ir::UserExternalNameRef::from_u32(0)),
            signature: sig_ref,
            colocated: false,
        };
        let func_ref = builder.import_function(data);
        import_refs.insert(import, func_ref);
        imports.push((func_ref, import));
    }

    // The status area is one extra buffer-table entry.
    let status_ptr = if descriptor.has_status {
        let address = builder
            .ins()
            .iadd_imm(table_ptr, (descriptor.table_order.len() * 8) as i64);
        Some(
            builder
                .ins()
                .load(types::I64, MemFlags::trusted(), address, 0),
        )
    } else {
        None
    };

    let mut emit = Emit {
        plan,
        encoded,
        table,
        participant: descriptor.participant_spans.clone(),
        import_refs,
        builder,
        table_ptr,
        scalars_ptr,
        scratch_ptr,
        status_ptr,
        storage_ptrs: BTreeMap::new(),
        locals: BTreeMap::new(),
        loop_vars: BTreeMap::new(),
        coords: Vec::new(),
        linear: participant,
        participants: participant,
    };

    let total = emit.word_value(descriptor.work_index);
    let participants = emit.word_value(descriptor.participants_index);
    emit.participants = participants;

    let ops = encoded.instructions.clone();
    if descriptor.pull_loop {
        emit.emit_pull_loop(total, &ops)?;
    } else {
        emit.emit_stride_loop(participant, total, &ops)?;
    }

    let status_zero = emit.builder.ins().iconst(types::I32, 0);
    emit.builder.ins().return_(&[status_zero]);
    drop(emit);
    Ok(LaunchEmission { function, imports })
}

/// A storage's resolved layout must fit its sealed byte allocation.
fn check_layout(plan: &PhysicalPlan<CpuDialect>, storage: StorageIx) -> Result<(), AssemblyFailure> {
    let record = &plan.storages()[storage];
    let mut elements: u64 = 1;
    for axis in &record.layout.shape {
        elements = elements
            .checked_mul(*axis)
            .ok_or_else(|| invariant(format!("storage {storage:?}'s layout overflows")))?;
    }
    let bytes = elements
        .checked_mul(u64::from(record.layout.dtype.bytes()))
        .ok_or_else(|| invariant(format!("storage {storage:?}'s layout overflows")))?;
    if bytes > record.bytes {
        return Err(invariant(format!(
            "storage {storage:?}'s layout needs {bytes} bytes but {} were allocated",
            record.bytes
        )));
    }
    Ok(())
}

impl<'a> Emit<'a> {
    // -- traversal loops -----------------------------------------------------

    /// Grid-stride traversal (OnePass/GridStride/serialized): this
    /// participant visits `participant, participant + P, …` below `total`.
    fn emit_stride_loop(
        &mut self,
        participant: Value,
        total: Value,
        ops: &[Instruction],
    ) -> Result<(), AssemblyFailure> {
        let header = self.builder.create_block();
        self.builder.append_block_param(header, types::I64);
        self.builder.ins().jump(header, &[participant.into()]);
        self.builder.switch_to_block(header);
        let linear = self.builder.block_params(header)[0];
        let in_range = self
            .builder
            .ins()
            .icmp(IntCC::UnsignedLessThan, linear, total);
        let body = self.builder.create_block();
        self.builder.append_block_param(body, types::I64);
        let exit = self.builder.create_block();
        self.builder
            .ins()
            .brif(in_range, body, &[linear.into()], exit, &[]);
        self.builder.switch_to_block(body);
        self.builder.seal_block(body);
        let linear = self.builder.block_params(body)[0];
        self.linear = linear;
        self.delinearize()?;
        self.emit_ops(ops)?;
        let next = self.builder.ins().iadd(linear, self.participants);
        self.builder.ins().jump(header, &[next.into()]);
        self.builder.seal_block(header);
        self.builder.switch_to_block(exit);
        self.builder.seal_block(exit);
        Ok(())
    }

    /// Dynamic-pull traversal: a worker claim loop over the pull counter —
    /// `while ((lin = atomic_fetch_add(counter, 1)) < total) { body }`. The
    /// Fill step before this launch zeroed the counter.
    fn emit_pull_loop(
        &mut self,
        total: Value,
        ops: &[Instruction],
    ) -> Result<(), AssemblyFailure> {
        let counter = self
            .encoded
            .descriptor
            .pull_counter
            .ok_or_else(|| invariant("a pull-traversal launch has no pull counter binding"))?;
        let counter_ptr = self.storage_ptr(counter)?;
        let one = self.builder.ins().iconst(types::I32, 1);
        let claim = self.builder.create_block();
        self.builder.ins().jump(claim, &[]);
        self.builder.switch_to_block(claim);
        let claimed = self.builder.ins().atomic_rmw(
            types::I32,
            MemFlags::trusted(),
            ir::AtomicRmwOp::Add,
            counter_ptr,
            one,
        );
        let linear = self.builder.ins().uextend(types::I64, claimed);
        let header = self.builder.create_block();
        self.builder.append_block_param(header, types::I64);
        self.builder.ins().jump(header, &[linear.into()]);
        self.builder.switch_to_block(header);
        let linear = self.builder.block_params(header)[0];
        let in_range = self
            .builder
            .ins()
            .icmp(IntCC::UnsignedLessThan, linear, total);
        let body = self.builder.create_block();
        self.builder.append_block_param(body, types::I64);
        let exit = self.builder.create_block();
        self.builder
            .ins()
            .brif(in_range, body, &[linear.into()], exit, &[]);
        self.builder.switch_to_block(body);
        self.builder.seal_block(body);
        let linear = self.builder.block_params(body)[0];
        self.linear = linear;
        self.delinearize()?;
        self.emit_ops(ops)?;
        self.builder.ins().jump(claim, &[]);
        self.builder.seal_block(claim);
        self.builder.seal_block(header);
        self.builder.switch_to_block(exit);
        self.builder.seal_block(exit);
        Ok(())
    }

    /// Delinearize the current linear coordinate over the launch's
    /// independent-axis extents (outermost first, last axis fastest).
    fn delinearize(&mut self) -> Result<(), AssemblyFailure> {
        let axes = self.encoded.descriptor.work_axes.clone();
        let rank = axes.len();
        if rank == 0 {
            self.coords = Vec::new();
            return Ok(());
        }
        let extents: Vec<Value> = axes
            .iter()
            .map(|expr| self.extent_value(expr))
            .collect::<Result<_, _>>()?;
        let mut strides = vec![self.builder.ins().iconst(types::I64, 1); rank];
        for axis in (0..rank - 1).rev() {
            strides[axis] = self.builder.ins().imul(strides[axis + 1], extents[axis + 1]);
        }
        let mut coords = Vec::with_capacity(rank);
        let mut rest = self.linear;
        for axis in 0..rank {
            let coord = self.builder.ins().udiv(rest, strides[axis]);
            let coord = if axis + 1 == rank {
                coord
            } else {
                self.builder.ins().urem(coord, extents[axis])
            };
            coords.push(coord);
            rest = if axis + 1 == rank {
                rest
            } else {
                self.builder.ins().urem(rest, strides[axis])
            };
        }
        self.coords = coords;
        Ok(())
    }

    // -- the instruction program ----------------------------------------------

    fn emit_ops(&mut self, ops: &[Instruction]) -> Result<(), AssemblyFailure> {
        for op in ops {
            self.instruction(op)?;
        }
        Ok(())
    }

    fn instruction(&mut self, op: &Instruction) -> Result<(), AssemblyFailure> {
        match op {
            Instruction::Const { into, value } => {
                let v = match value.value {
                    ConstantValue::Int(bits) => {
                        let c = self.builder.ins().iconst(types::I64, bits);
                        self.builder.ins().fcvt_from_sint(types::F64, c)
                    }
                    ConstantValue::Float { bits } => self
                        .builder
                        .ins()
                        .f64const(ir::immediates::Ieee64::with_bits(bits)),
                    ConstantValue::Bool(flag) => {
                        let c = self.builder.ins().iconst(types::I64, i64::from(flag));
                        self.builder.ins().fcvt_from_uint(types::F64, c)
                    }
                };
                self.store_result(*into, v)
            }
            Instruction::Extent { into, extent } => {
                let index = self.extent_word_index(*extent)?;
                let word = self.word_value(index);
                let v = self.builder.ins().fcvt_from_sint(types::F64, word);
                self.store_result(*into, v)
            }
            Instruction::Unary { into, op, operand, dtype } => {
                let v = self.fetch(*operand)?;
                let out = match (op, dtype.is_float()) {
                    (UnaryOp::Neg, true) => self.builder.ins().fneg(v),
                    (UnaryOp::Neg, false) => {
                        let bits = int_bits(&mut self.builder, v);
                        let bits = self.builder.ins().ireduce(types::I32, bits);
                        let bits = self.builder.ins().ineg(bits);
                        int_value_of(&mut self.builder, bits, *dtype)
                    }
                    (UnaryOp::BitNot, false) => {
                        let bits = int_bits(&mut self.builder, v);
                        let bits = self.builder.ins().ireduce(types::I32, bits);
                        let bits = self.builder.ins().bnot(bits);
                        int_value_of(&mut self.builder, bits, *dtype)
                    }
                    (UnaryOp::Not, _) => {
                        let zero = self
                            .builder
                            .ins()
                            .f64const(ir::immediates::Ieee64::with_bits(0));
                        let bit = self.builder.ins().fcmp(FloatCC::Equal, v, zero);
                        f64_of_bool(&mut self.builder, bit)
                    }
                    (UnaryOp::BitNot, true) => {
                        return Err(invariant("a bitwise not of a float operand"))
                    }
                };
                self.store_result(*into, out)
            }
            Instruction::Binary { into, op, left, right, dtype } => {
                let a = self.fetch(*left)?;
                let b = self.fetch(*right)?;
                let out = if dtype.is_float() {
                    let raw = match op {
                        BinaryOp::Add => self.builder.ins().fadd(a, b),
                        BinaryOp::Sub => self.builder.ins().fsub(a, b),
                        BinaryOp::Mul => self.builder.ins().fmul(a, b),
                        BinaryOp::Div => self.builder.ins().fdiv(a, b),
                        BinaryOp::Rem => self.call_import(Import::Fmod, &[a.into(), b.into()]),
                        _ => return Err(invariant("a non-float binary reached float emission")),
                    };
                    self.round_s(raw, *dtype)
                } else if *dtype == DType::Bool {
                    let x = bool_of(&mut self.builder, a);
                    let y = bool_of(&mut self.builder, b);
                    let bit = match op {
                        BinaryOp::And => self.builder.ins().band(x, y),
                        BinaryOp::Or => self.builder.ins().bor(x, y),
                        _ => {
                            return Err(invariant("a non-boolean binary reached boolean emission"))
                        }
                    };
                    f64_of_bool(&mut self.builder, bit)
                } else {
                    let wide_a = int_bits(&mut self.builder, a);
                    let wide_b = int_bits(&mut self.builder, b);
                    let x = self.builder.ins().ireduce(types::I32, wide_a);
                    let y = self.builder.ins().ireduce(types::I32, wide_b);
                    let bits = self.int_binary(*op, *dtype, x, y)?;
                    int_value_of(&mut self.builder, bits, *dtype)
                };
                self.store_result(*into, out)
            }
            Instruction::Compare { into, op, left, right, dtype } => {
                let a = self.fetch(*left)?;
                let b = self.fetch(*right)?;
                let bit = if dtype.is_float() {
                    let cc = match op {
                        RelOp::Eq => FloatCC::Equal,
                        RelOp::Ne => FloatCC::NotEqual,
                        RelOp::Lt => FloatCC::LessThan,
                        RelOp::Le => FloatCC::LessThanOrEqual,
                        RelOp::Gt => FloatCC::GreaterThan,
                        RelOp::Ge => FloatCC::GreaterThanOrEqual,
                    };
                    self.builder.ins().fcmp(cc, a, b)
                } else if *dtype == DType::Bool {
                    let x = bool_of(&mut self.builder, a);
                    let y = bool_of(&mut self.builder, b);
                    let cc = match op {
                        RelOp::Eq => IntCC::Equal,
                        RelOp::Ne => IntCC::NotEqual,
                        _ => return Err(invariant("an ordered boolean comparison")),
                    };
                    self.builder.ins().icmp(cc, x, y)
                } else {
                    let x = int_bits(&mut self.builder, a);
                    let y = int_bits(&mut self.builder, b);
                    let cc = match op {
                        RelOp::Eq => IntCC::Equal,
                        RelOp::Ne => IntCC::NotEqual,
                        RelOp::Lt => IntCC::SignedLessThan,
                        RelOp::Le => IntCC::SignedLessThanOrEqual,
                        RelOp::Gt => IntCC::SignedGreaterThan,
                        RelOp::Ge => IntCC::SignedGreaterThanOrEqual,
                    };
                    self.builder.ins().icmp(cc, x, y)
                };
                let out = f64_of_bool(&mut self.builder, bit);
                self.store_result(*into, out)
            }
            Instruction::Math { into, op, operands, dtype } => {
                self.math(*into, *op, operands, *dtype)
            }
            Instruction::Fma { into, a, b, c, dtype } => {
                let a = self.fetch(*a)?;
                let b = self.fetch(*b)?;
                let c = self.fetch(*c)?;
                let out = if *dtype == DType::F32 {
                    // Single-rounded f32 contraction (the registry rule).
                    let a = self.builder.ins().fdemote(types::F32, a);
                    let b = self.builder.ins().fdemote(types::F32, b);
                    let c = self.builder.ins().fdemote(types::F32, c);
                    let f = self.builder.ins().fma(a, b, c);
                    self.builder.ins().fpromote(types::F64, f)
                } else {
                    let f = self.builder.ins().fma(a, b, c);
                    self.round_s(f, *dtype)
                };
                self.store_result(*into, out)
            }
            Instruction::Cast { into, operand, from, to } => {
                let v = self.fetch(*operand)?;
                let out = if from.is_int()
                    && *from != DType::Bool
                    && to.is_int()
                    && *to != DType::Bool
                    || (*from == DType::Bool && to.is_int())
                {
                    // Integer casts preserve bits.
                    let wide = int_bits(&mut self.builder, v);
                    let bits = self.builder.ins().ireduce(types::I32, wide);
                    int_value_of(&mut self.builder, bits, *to)
                } else {
                    self.round_s(v, *to)
                };
                self.store_result(*into, out)
            }
            Instruction::Select { into, condition, then_value, else_value, .. } => {
                let c = self.fetch(*condition)?;
                let a = self.fetch(*then_value)?;
                let b = self.fetch(*else_value)?;
                let cond = bool_of(&mut self.builder, c);
                let v = self.builder.ins().select(cond, a, b);
                self.store_result(*into, v)
            }
            Instruction::TableLookup { into, index, table } => {
                let v = self.fetch(*index)?;
                let ordinal = int_bits(&mut self.builder, v);
                let slot = self.builder.create_sized_stack_slot(StackSlotData::new(
                    ir::StackSlotKind::ExplicitSlot,
                    (4 * table.len()) as u32,
                    3,
                ));
                let base = self.builder.ins().stack_addr(types::I64, slot, 0);
                for (position, entry) in table.iter().enumerate() {
                    let value = self.builder.ins().iconst(types::I32, i64::from(*entry));
                    let address = self.builder.ins().iadd_imm(base, (position * 4) as i64);
                    self.builder
                        .ins()
                        .store(MemFlags::trusted(), value, address, 0);
                }
                let scaled = self.builder.ins().imul_imm(ordinal, 4);
                let address = self.builder.ins().iadd(base, scaled);
                let entry = self
                    .builder
                    .ins()
                    .load(types::I32, MemFlags::trusted(), address, 0);
                let out = int_value_of(&mut self.builder, entry, DType::I32);
                self.store_result(*into, out)
            }
            Instruction::Load { into, place, coords, .. } => {
                let (address, dtype) = self.place_address(*place, coords)?;
                let v = self.load_element(address, dtype);
                self.store_result(*into, v)
            }
            Instruction::Store { place, coords, value, .. } => {
                let (address, dtype) = self.place_address(*place, coords)?;
                let v = self.fetch(*value)?;
                self.store_element(address, v, dtype);
                Ok(())
            }
            Instruction::PackedPlaneRead { into, place, coords, repr, plane, entry } => {
                self.packed_plane_read(*into, *place, coords, repr, *plane, *entry)
            }
            Instruction::PlaneLoad { into, place, coords, repr, plane } => {
                let schema = self.plane_schema(repr, *plane)?;
                let (address, storage_dtype) =
                    self.plane_element_address(*place, coords, &schema, None)?;
                let v = self.load_element(address, storage_dtype);
                self.store_result(*into, v)
            }
            Instruction::PlaneStore { place, coords, repr, plane, value } => {
                let schema = self.plane_schema(repr, *plane)?;
                let (address, storage_dtype) =
                    self.plane_element_address(*place, coords, &schema, None)?;
                let v = self.fetch(*value)?;
                self.store_element(address, v, storage_dtype);
                Ok(())
            }
            Instruction::Atomic { place, coords, value, op, dtype, mode } => {
                self.atomic(*place, coords, *value, *op, *dtype, *mode)
            }
            Instruction::Repeat { binder, start, end, carries, body } => {
                self.repeat(*binder, *start, *end, carries, body)
            }
            Instruction::Branch { condition, then_body, else_body, joins } => {
                self.branch(*condition, then_body, else_body, joins)
            }
            Instruction::Fold { into, place, axis, coords, op, schema } => {
                self.fold(*into, *place, *axis, coords, *op, *schema)
            }
            Instruction::Check { kind, status, guarded } => self.check(kind, *status, guarded),
            Instruction::Publish { value, destination } => {
                let v = self.fetch(*value)?;
                self.store_result(*destination, v)
            }
            Instruction::Barrier => {
                // One flat participant pool: a full fence orders it.
                self.builder.ins().fence();
                Ok(())
            }
            Instruction::GridBarrier => Err(invariant(
                "a grid barrier reached a CPU launch: cooperative-grid proposals are never \
                 made on this target",
            )),
        }
    }

    /// Integer binary emission: 32-bit wrapping arithmetic, Euclidean
    /// division/remainder, and checked shifts (the predicate guards them).
    fn int_binary(
        &mut self,
        op: BinaryOp,
        dtype: DType,
        x: Value,
        y: Value,
    ) -> Result<Value, AssemblyFailure> {
        let builder = &mut self.builder;
        Ok(match op {
            BinaryOp::Add => builder.ins().iadd(x, y),
            BinaryOp::Sub => builder.ins().isub(x, y),
            BinaryOp::Mul => builder.ins().imul(x, y),
            BinaryOp::BitOr => builder.ins().bor(x, y),
            BinaryOp::BitXor => builder.ins().bxor(x, y),
            BinaryOp::BitAnd => builder.ins().band(x, y),
            BinaryOp::Shl => builder.ins().ishl(x, y),
            BinaryOp::Shr => {
                if dtype == DType::U32 {
                    builder.ins().ushr(x, y)
                } else {
                    builder.ins().sshr(x, y)
                }
            }
            BinaryOp::Div | BinaryOp::Rem => {
                // Euclidean: the remainder is non-negative and below |rhs|.
                let r0 = builder.ins().srem(x, y);
                let zero = builder.ins().iconst(types::I32, 0);
                let r_negative = builder.ins().icmp(IntCC::SignedLessThan, r0, zero);
                let y_negative = builder.ins().icmp(IntCC::SignedLessThan, y, zero);
                let disagree = builder.ins().bxor(r_negative, y_negative);
                let adjusted = builder.ins().iadd(r0, y);
                let r = builder.ins().select(disagree, adjusted, r0);
                if op == BinaryOp::Rem {
                    r
                } else {
                    let numerator = builder.ins().isub(x, r);
                    builder.ins().sdiv(numerator, y)
                }
            }
            _ => return Err(invariant("a non-integer binary reached integer emission")),
        })
    }

    /// The versioned `seismic_math` sequence: binary64 host evaluation
    /// rounded once (identical code path to the reference interpreter).
    fn math(
        &mut self,
        into: usize,
        math: MathOp,
        operands: &[usize],
        dtype: DType,
    ) -> Result<(), AssemblyFailure> {
        let mut args = Vec::with_capacity(3);
        for position in operands {
            args.push(self.fetch(*position)?);
        }
        let arg = |n: usize| -> Value { args[n] };
        let out = match math {
            MathOp::Fma => {
                let (a, b, c) = (arg(0), arg(1), arg(2));
                if dtype == DType::F32 {
                    let a = self.builder.ins().fdemote(types::F32, a);
                    let b = self.builder.ins().fdemote(types::F32, b);
                    let c = self.builder.ins().fdemote(types::F32, c);
                    let f = self.builder.ins().fma(a, b, c);
                    self.builder.ins().fpromote(types::F64, f)
                } else {
                    let f = self.builder.ins().fma(a, b, c);
                    self.round_s(f, dtype)
                }
            }
            MathOp::Exp | MathOp::ExpFast => {
                let v = self.call_import(Import::ExpF64, &[arg(0).into()]);
                self.round_s(v, dtype)
            }
            MathOp::Log => {
                let v = self.call_import(Import::LogF64, &[arg(0).into()]);
                self.round_s(v, dtype)
            }
            MathOp::Sin => {
                let v = self.call_import(Import::SinF64, &[arg(0).into()]);
                self.round_s(v, dtype)
            }
            MathOp::Cos => {
                let v = self.call_import(Import::CosF64, &[arg(0).into()]);
                self.round_s(v, dtype)
            }
            MathOp::Sqrt => {
                let v = self.builder.ins().sqrt(arg(0));
                self.round_s(v, dtype)
            }
            MathOp::Rsqrt => {
                let one = self
                    .builder
                    .ins()
                    .f64const(ir::immediates::Ieee64::with_bits(1.0f64.to_bits()));
                let root = self.builder.ins().sqrt(arg(0));
                let v = self.builder.ins().fdiv(one, root);
                self.round_s(v, dtype)
            }
            MathOp::Abs => {
                if dtype.is_int() {
                    let wide = int_bits(&mut self.builder, arg(0));
                    let bits = self.builder.ins().ireduce(types::I32, wide);
                    let abs = self.builder.ins().iabs(bits);
                    int_value_of(&mut self.builder, abs, dtype)
                } else {
                    self.builder.ins().fabs(arg(0))
                }
            }
            MathOp::Max => {
                let v = self.call_import(Import::Fmax, &[arg(0).into(), arg(1).into()]);
                self.round_s(v, dtype)
            }
            MathOp::Min => {
                let v = self.call_import(Import::Fmin, &[arg(0).into(), arg(1).into()]);
                self.round_s(v, dtype)
            }
        };
        self.store_result(into, out)
    }

    // -- control ----------------------------------------------------------------

    fn repeat(
        &mut self,
        binder: usize,
        start: usize,
        end: usize,
        carries: &[CarrySlot],
        body: &[Instruction],
    ) -> Result<(), AssemblyFailure> {
        let start_v = self.fetch(start)?;
        let end_v = self.fetch(end)?;
        let start = int_bits(&mut self.builder, start_v);
        let end = int_bits(&mut self.builder, end_v);
        // Ascending serial loop over [start, end); the discharged range
        // predicate guards the bound. Carry lanes are loop-header params:
        // `current` is rebound from `initial` on the first visit and from
        // `update` after each visit; `result` is the final value.
        let header = self.builder.create_block();
        for _ in carries {
            self.builder.append_block_param(header, types::F64);
        }
        self.builder.append_block_param(header, types::I64);
        let mut initial: Vec<ir::Value> = Vec::with_capacity(carries.len());
        for carry in carries {
            initial.push(self.fetch(carry.initial)?);
        }
        let mut jump_args: Vec<ir::BlockArg> = initial.into_iter().map(Into::into).collect();
        jump_args.push(start.into());
        self.builder.ins().jump(header, &jump_args);
        self.builder.switch_to_block(header);
        let coordinate = self
            .builder
            .block_params(header)
            .last()
            .copied()
            .expect("the repeat header has a coordinate");
        let currents: Vec<Value> = self.builder.block_params(header)[..carries.len()].to_vec();
        let in_range = self
            .builder
            .ins()
            .icmp(IntCC::UnsignedLessThan, coordinate, end);
        let body_block = self.builder.create_block();
        for _ in carries {
            self.builder.append_block_param(body_block, types::F64);
        }
        self.builder.append_block_param(body_block, types::I64);
        let exit = self.builder.create_block();
        for _ in carries {
            self.builder.append_block_param(exit, types::F64);
        }
        let mut body_args: Vec<ir::BlockArg> =
            currents.iter().copied().map(Into::into).collect();
        body_args.push(coordinate.into());
        let exit_args: Vec<ir::BlockArg> = currents.iter().copied().map(Into::into).collect();
        self.builder
            .ins()
            .brif(in_range, body_block, &body_args, exit, &exit_args);
        self.builder.switch_to_block(body_block);
        self.builder.seal_block(body_block);
        let coordinate = self
            .builder
            .block_params(body_block)
            .last()
            .copied()
            .expect("the repeat body has a coordinate");
        let currents: Vec<Value> = self.builder.block_params(body_block)[..carries.len()].to_vec();
        self.loop_vars.insert(binder, coordinate);
        for (carry, current) in carries.iter().zip(currents) {
            self.locals.insert(carry.current, current);
        }
        self.emit_ops(body)?;
        self.loop_vars.remove(&binder);
        let mut updates: Vec<ir::Value> = Vec::with_capacity(carries.len());
        for carry in carries {
            updates.push(self.fetch(carry.update)?);
        }
        let one = self.builder.ins().iconst(types::I64, 1);
        let next = self.builder.ins().iadd(coordinate, one);
        let mut back_args: Vec<ir::BlockArg> = updates.into_iter().map(Into::into).collect();
        back_args.push(next.into());
        self.builder.ins().jump(header, &back_args);
        self.builder.seal_block(header);
        self.builder.switch_to_block(exit);
        self.builder.seal_block(exit);
        let finals: Vec<Value> = self.builder.block_params(exit)[..carries.len()].to_vec();
        for (carry, final_value) in carries.iter().zip(finals) {
            self.locals.insert(carry.result, final_value);
        }
        Ok(())
    }

    fn branch(
        &mut self,
        condition: usize,
        then_body: &[Instruction],
        else_body: &[Instruction],
        joins: &[JoinSlot],
    ) -> Result<(), AssemblyFailure> {
        let value = self.fetch(condition)?;
        let bits = int_bits(&mut self.builder, value);
        let zero_bits = self.builder.ins().iconst(types::I64, 0);
        let taken = self.builder.ins().icmp(IntCC::NotEqual, bits, zero_bits);
        let then_block = self.builder.create_block();
        let else_block = self.builder.create_block();
        let join = self.builder.create_block();
        for _ in joins {
            self.builder.append_block_param(join, types::F64);
        }
        self.builder
            .ins()
            .brif(taken, then_block, &[], else_block, &[]);
        self.builder.switch_to_block(then_block);
        self.builder.seal_block(then_block);
        self.emit_ops(then_body)?;
        let mut then_args: Vec<ir::Value> = Vec::with_capacity(joins.len());
        for join_slot in joins {
            then_args.push(self.fetch(join_slot.then_value)?);
        }
        let then_args: Vec<ir::BlockArg> = then_args.into_iter().map(Into::into).collect();
        self.builder.ins().jump(join, &then_args);
        self.builder.switch_to_block(else_block);
        self.builder.seal_block(else_block);
        self.emit_ops(else_body)?;
        let mut else_args: Vec<ir::Value> = Vec::with_capacity(joins.len());
        for join_slot in joins {
            else_args.push(self.fetch(join_slot.else_value)?);
        }
        let else_args: Vec<ir::BlockArg> = else_args.into_iter().map(Into::into).collect();
        self.builder.ins().jump(join, &else_args);
        self.builder.switch_to_block(join);
        self.builder.seal_block(join);
        let joined: Vec<Value> = self.builder.block_params(join).to_vec();
        for (join_slot, value) in joins.iter().zip(joined) {
            self.locals.insert(join_slot.joined, value);
        }
        Ok(())
    }

    /// A guarded safety check: on failure the status field is written (the
    /// first error wins) and the guarded instructions are skipped, whose
    /// kernel-local results then hold an unspecified value of their type.
    fn check(
        &mut self,
        kind: &CheckKind,
        status: StatusFieldIx,
        guarded: &[Instruction],
    ) -> Result<(), AssemblyFailure> {
        let ok = self.check_predicate(kind)?;
        let kernel_results: Vec<usize> = guarded
            .iter()
            .flat_map(instruction_results)
            .filter(|position| matches!(self.encoded.bindings[*position], Binding::Local))
            .collect();
        let word_results: Vec<(usize, DType)> = guarded
            .iter()
            .flat_map(instruction_results)
            .filter_map(|position| match &self.encoded.bindings[position] {
                Binding::Word { index, dtype } => Some((*index, *dtype)),
                _ => None,
            })
            .collect();
        let continue_block = self.builder.create_block();
        let fail_block = self.builder.create_block();
        let skip_block = self.builder.create_block();
        let mut skip_params: Vec<Value> = Vec::with_capacity(kernel_results.len());
        for _ in &kernel_results {
            skip_params.push(self.builder.append_block_param(skip_block, types::F64));
        }
        self.builder
            .ins()
            .brif(ok, continue_block, &[], fail_block, &[]);
        self.builder.switch_to_block(fail_block);
        self.builder.seal_block(fail_block);
        {
            // First-error status write (the code is a nonzero constant).
            let base = self
                .status_ptr
                .ok_or_else(|| invariant("a check reached a launch without a status entry"))?;
            let offset = (status.index() * 4) as i64;
            let address = self.builder.ins().iadd_imm(base, offset);
            let code = self.builder.ins().iconst(types::I32, 1);
            self.builder
                .ins()
                .store(MemFlags::trusted(), code, address, 0);
            // Skipped word results take a neutral value.
            for (index, dtype) in word_results {
                let neutral = self
                    .builder
                    .ins()
                    .f64const(ir::immediates::Ieee64::with_bits(0));
                let word = self.encode_word(neutral, dtype);
                store_word(&mut self.builder, self.scalars_ptr, index, word);
            }
            let zeros: Vec<Value> = skip_params
                .iter()
                .map(|_| {
                    self.builder
                        .ins()
                        .f64const(ir::immediates::Ieee64::with_bits(0))
                })
                .collect();
            let zero_args: Vec<ir::BlockArg> = zeros.into_iter().map(Into::into).collect();
            self.builder.ins().jump(skip_block, &zero_args);
        }
        self.builder.switch_to_block(continue_block);
        self.builder.seal_block(continue_block);
        self.emit_ops(guarded)?;
        let computed: Vec<Value> = kernel_results
            .iter()
            .map(|position| self.locals[position])
            .collect();
        let computed_args: Vec<ir::BlockArg> = computed.into_iter().map(Into::into).collect();
        self.builder.ins().jump(skip_block, &computed_args);
        self.builder.switch_to_block(skip_block);
        self.builder.seal_block(skip_block);
        for (position, param) in kernel_results.iter().zip(skip_params) {
            self.locals.insert(*position, param);
        }
        Ok(())
    }

    fn check_predicate(&mut self, kind: &CheckKind) -> Result<Value, AssemblyFailure> {
        let zero = self.builder.ins().iconst(types::I64, 0);
        Ok(match kind {
            CheckKind::IndexInBounds { index, extent } => {
                let v = self.fetch(*index)?;
                let v = int_bits(&mut self.builder, v);
                let e = self.extent_value(extent)?;
                let non_negative = self
                    .builder
                    .ins()
                    .icmp(IntCC::SignedGreaterThanOrEqual, v, zero);
                let below = self.builder.ins().icmp(IntCC::UnsignedLessThan, v, e);
                self.builder.ins().band(non_negative, below)
            }
            CheckKind::RangeInBounds { start, end, extent } => {
                let s = self.fetch(*start)?;
                let e = self.fetch(*end)?;
                let s = int_bits(&mut self.builder, s);
                let e_v = int_bits(&mut self.builder, e);
                let bound = self.extent_value(extent)?;
                let s_ok = self
                    .builder
                    .ins()
                    .icmp(IntCC::SignedGreaterThanOrEqual, s, zero);
                let ordered = self
                    .builder
                    .ins()
                    .icmp(IntCC::SignedLessThanOrEqual, s, e_v);
                let e_ok = self
                    .builder
                    .ins()
                    .icmp(IntCC::UnsignedLessThanOrEqual, e_v, bound);
                let both = self.builder.ins().band(s_ok, ordered);
                self.builder.ins().band(both, e_ok)
            }
            CheckKind::DivisorNonZero { value, .. } => {
                let v = self.fetch(*value)?;
                let bits = int_bits(&mut self.builder, v);
                self.builder.ins().icmp(IntCC::NotEqual, bits, zero)
            }
            CheckKind::SignedDivisionNoOverflow { lhs, rhs } => {
                let l = self.fetch(*lhs)?;
                let r = self.fetch(*rhs)?;
                let l = int_bits(&mut self.builder, l);
                let r = int_bits(&mut self.builder, r);
                let min = self.builder.ins().iconst(types::I64, i64::from(i32::MIN));
                let minus_one = self.builder.ins().iconst(types::I64, -1);
                let is_min = self.builder.ins().icmp(IntCC::Equal, l, min);
                let is_m1 = self.builder.ins().icmp(IntCC::Equal, r, minus_one);
                let overflow = self.builder.ins().band(is_min, is_m1);
                let one = self.builder.ins().iconst(types::I8, 1);
                self.builder.ins().bxor(overflow, one)
            }
            CheckKind::ShiftInRange { value } => {
                let v = self.fetch(*value)?;
                let v = int_bits(&mut self.builder, v);
                let low = self
                    .builder
                    .ins()
                    .icmp(IntCC::SignedGreaterThanOrEqual, v, zero);
                let limit = self.builder.ins().iconst(types::I64, 32);
                let high = self.builder.ins().icmp(IntCC::SignedLessThan, v, limit);
                self.builder.ins().band(low, high)
            }
        })
    }

    // -- tensor access --------------------------------------------------------

    /// The views of one place (one per plane), cloned for emission.
    fn views_of(
        &self,
        place: usize,
    ) -> Result<NonEmpty<seismic_realization::physical::PhysicalStorageView>, AssemblyFailure> {
        match self.encoded.bindings.get(place) {
            Some(Binding::Tensor { views }) => Ok(views.clone()),
            _ => Err(invariant("a tensor access names a non-tensor binding")),
        }
    }

    /// The dense-plane view of one place, cloned for emission.
    fn dense_view(
        &self,
        place: usize,
    ) -> Result<seismic_realization::physical::PhysicalStorageView, AssemblyFailure> {
        let views = self.views_of(place)?;
        for view in views.iter() {
            if matches!(self.plan.storages()[view.storage].plane, StoragePlane::Dense) {
                return Ok(view.clone());
            }
        }
        Err(invariant(
            "a dense element access names a place with no dense plane",
        ))
    }

    /// Row-major storage-element strides of one storage's resolved layout.
    fn storage_strides(&self, storage: StorageIx) -> Vec<u64> {
        let shape = &self.plan.storages()[storage].layout.shape;
        let mut strides = vec![1u64; shape.len()];
        for axis in (0..shape.len().saturating_sub(1)).rev() {
            strides[axis] = strides[axis + 1] * shape[axis + 1];
        }
        strides
    }

    fn layout_dtype(&self, storage: StorageIx) -> DType {
        self.plan.storages()[storage].layout.dtype
    }

    /// The host pointer of one bound storage: the buffer table for shared
    /// storages, the worker's private scratch plus its span offset for
    /// participant-scoped storages.
    fn storage_ptr(&mut self, storage: StorageIx) -> Result<Value, AssemblyFailure> {
        if let Some(pointer) = self.storage_ptrs.get(&storage) {
            return Ok(*pointer);
        }
        let pointer = if let Some(offset) = self.participant.get(&storage) {
            self.builder.ins().iadd_imm(self.scratch_ptr, *offset as i64)
        } else {
            let index = self
                .table
                .get(&storage)
                .copied()
                .ok_or_else(|| invariant(format!("storage {storage:?} is not bound to the launch")))?;
            let address = self.builder.ins().iadd_imm(self.table_ptr, (index * 8) as i64);
            self.builder
                .ins()
                .load(types::I64, MemFlags::trusted(), address, 0)
        };
        self.storage_ptrs.insert(storage, pointer);
        Ok(pointer)
    }

    /// Apply one view transform in residence coordinates: view coordinates
    /// in, residence coordinates out. Steps apply outermost first, so they
    /// fold view coordinates to residence coordinates innermost (last) step
    /// first; a `Reshape` step needs its child extents — the last step's
    /// child is the view-side shape `view.ty` (including any trailing
    /// reshape), every earlier step's is the next step's `source_shape`.
    fn transform_coords(
        &mut self,
        view: &seismic_realization::physical::PhysicalStorageView,
        coords: &[Value],
    ) -> Result<Vec<Value>, AssemblyFailure> {
        let steps = &view.transform.steps;
        let mut current = coords.to_vec();
        for (index, step) in steps.iter().enumerate().rev() {
            current = match &step.kind {
                ViewStepKind::Reshape => {
                    let child = if index + 1 == steps.len() {
                        view.ty.axes.clone()
                    } else {
                        steps[index + 1].source_shape.clone()
                    };
                    let mut flat = if child.is_empty() {
                        self.linear
                    } else {
                        let mut flat = self.builder.ins().iconst(types::I64, 0);
                        for (axis, coord) in current.iter().enumerate() {
                            let extent = self.extent_value(
                                child
                                    .get(axis)
                                    .ok_or_else(|| invariant("a reshape view lacks an axis"))?,
                            )?;
                            flat = self.builder.ins().imul(flat, extent);
                            flat = self.builder.ins().iadd(flat, *coord);
                        }
                        flat
                    };
                    let mut parent =
                        vec![self.builder.ins().iconst(types::I64, 0); step.source_shape.len()];
                    for axis in (0..step.source_shape.len()).rev() {
                        let extent = self.extent_value(&step.source_shape[axis])?;
                        parent[axis] = self.builder.ins().urem(flat, extent);
                        flat = self.builder.ins().udiv(flat, extent);
                    }
                    parent
                }
                ViewStepKind::Transpose { permutation } => {
                    let zero = self.builder.ins().iconst(types::I64, 0);
                    let mut parent = vec![zero; step.source_shape.len()];
                    for (view_axis, coord) in current.iter().enumerate() {
                        let source_axis = *permutation
                            .get(view_axis)
                            .ok_or_else(|| invariant("a transpose permutation lacks an axis"))?
                            as usize;
                        let slot = parent
                            .get_mut(source_axis)
                            .ok_or_else(|| invariant("a transpose names an absent source axis"))?;
                        *slot = *coord;
                    }
                    parent
                }
                ViewStepKind::Slice { axes } => {
                    let mut parent = Vec::with_capacity(step.source_shape.len());
                    let mut view_axis = 0usize;
                    for source_axis in 0..step.source_shape.len() {
                        let axis = axes
                            .get(source_axis)
                            .ok_or_else(|| invariant("a slice view lacks an axis"))?;
                        let coord = match axis {
                            SliceAxisTemplate::Point(leaf) => {
                                let position = self
                                    .encoded
                                    .endpoint_of
                                    .get(leaf)
                                    .copied()
                                    .ok_or_else(|| {
                                        invariant("a slice point endpoint is not a kernel input")
                                    })?;
                                let v = self.fetch(position)?;
                                int_bits(&mut self.builder, v)
                            }
                            SliceAxisTemplate::Range { start, .. } => {
                                let coord = current
                                    .get(view_axis)
                                    .copied()
                                    .ok_or_else(|| invariant("a slice view lacks a view axis"))?;
                                view_axis += 1;
                                if let Some(start) = start {
                                    let position = self
                                        .encoded
                                        .endpoint_of
                                        .get(start)
                                        .copied()
                                        .ok_or_else(|| {
                                            invariant(
                                                "a slice start endpoint is not a kernel input",
                                            )
                                        })?;
                                    let v = self.fetch(position)?;
                                    let start = int_bits(&mut self.builder, v);
                                    self.builder.ins().iadd(coord, start)
                                } else {
                                    coord
                                }
                            }
                            SliceAxisTemplate::Full => {
                                let coord = current
                                    .get(view_axis)
                                    .copied()
                                    .ok_or_else(|| invariant("a slice view lacks a view axis"))?;
                                view_axis += 1;
                                coord
                            }
                        };
                        parent.push(coord);
                    }
                    parent
                }
            };
        }
        Ok(current)
    }

    /// The byte address and element dtype of one dense-plane place at
    /// explicit coordinate bindings.
    fn place_address(
        &mut self,
        place: usize,
        coords: &[usize],
    ) -> Result<(Value, DType), AssemblyFailure> {
        let view = self.dense_view(place)?;
        let storage = view.storage;
        let dtype = self.layout_dtype(storage);
        let strides = self.storage_strides(storage);
        let mut coord_values = Vec::with_capacity(coords.len());
        for position in coords {
            let v = self.fetch(*position)?;
            coord_values.push(int_bits(&mut self.builder, v));
        }
        let residence = self.transform_coords(&view, &coord_values)?;
        let mut offset = self.builder.ins().iconst(types::I64, 0);
        for (axis, coord) in residence.iter().enumerate() {
            let stride = strides
                .get(axis)
                .copied()
                .ok_or_else(|| invariant("a view names an absent storage axis"))?;
            let stride = i64::try_from(stride)
                .map_err(|_| invariant("a storage stride exceeds the addressable size domain"))?;
            let scaled = self.builder.ins().imul_imm(*coord, stride);
            offset = self.builder.ins().iadd(offset, scaled);
        }
        let base = self.storage_ptr(storage)?;
        let bytes = u64::from(dtype.bytes());
        let byte_offset = if bytes == 1 {
            offset
        } else {
            self.builder.ins().imul_imm(offset, bytes as i64)
        };
        Ok((self.builder.ins().iadd(base, byte_offset), dtype))
    }

    /// The plane of one representation a plane op addresses.
    fn plane_schema(
        &self,
        repr_name: &str,
        plane: PlaneField,
    ) -> Result<repr::PlaneSchema, AssemblyFailure> {
        let representation = repr::lookup(repr_name)
            .ok_or_else(|| invariant(format!("unknown representation `{repr_name}`")))?;
        let index = representation
            .plane_index(plane.name())
            .ok_or_else(|| invariant(format!("`{repr_name}` has no `{}` plane", plane.name())))?;
        Ok(representation.plane_schemas()[index].clone())
    }

    /// The storage-element address of one plane row: the outer residence
    /// coordinates plus the storage-element ordinal along the packing axis
    /// (the last view axis). `PlaneLoad`/`PlaneStore` pass the ordinal as
    /// the packing coordinate directly; `PackedPlaneRead` passes an entry
    /// index converted to its storage element.
    fn plane_element_address(
        &mut self,
        place: usize,
        coords: &[usize],
        schema: &repr::PlaneSchema,
        within_row: Option<Value>,
    ) -> Result<(Value, DType), AssemblyFailure> {
        let views = self.views_of(place)?;
        let view = views
            .as_slice()
            .get(schema.ordinal as usize)
            .cloned()
            .ok_or_else(|| invariant("a plane of the representation is unbound"))?;
        let storage = view.storage;
        let dtype = self.layout_dtype(storage);
        let strides = self.storage_strides(storage);
        let rank = strides.len();
        let mut coord_values = Vec::with_capacity(coords.len());
        for position in coords {
            let v = self.fetch(*position)?;
            coord_values.push(int_bits(&mut self.builder, v));
        }
        let residence = self.transform_coords(&view, &coord_values)?;
        let mut offset = self.builder.ins().iconst(types::I64, 0);
        for (axis, coord) in residence.iter().take(rank.saturating_sub(1)).enumerate() {
            let stride = strides
                .get(axis)
                .copied()
                .ok_or_else(|| invariant("a plane access names an absent storage axis"))?;
            let stride = i64::try_from(stride)
                .map_err(|_| invariant("a storage stride exceeds the addressable size domain"))?;
            let scaled = self.builder.ins().imul_imm(*coord, stride);
            offset = self.builder.ins().iadd(offset, scaled);
        }
        let within = match within_row {
            Some(ordinal) => ordinal,
            // The packing-axis residence coordinate is the storage-element
            // ordinal within the plane row.
            None => *residence
                .last()
                .ok_or_else(|| invariant("a plane access of a scalar place"))?,
        };
        let element = self.builder.ins().iadd(offset, within);
        let base = self.storage_ptr(storage)?;
        let bytes = u64::from(dtype.bytes());
        let byte_offset = if bytes == 1 {
            element
        } else {
            self.builder.ins().imul_imm(element, bytes as i64)
        };
        Ok((self.builder.ins().iadd(base, byte_offset), dtype))
    }

    /// Read entry `entry` of the group of the logical element at `coords`
    /// in one plane; a packed plane yields the raw code zero-extended.
    fn packed_plane_read(
        &mut self,
        into: usize,
        place: usize,
        coords: &[usize],
        repr_name: &str,
        plane: PlaneField,
        entry: u32,
    ) -> Result<(), AssemblyFailure> {
        let schema = self.plane_schema(repr_name, plane)?;
        let v = self.fetch(
            *coords
                .last()
                .ok_or_else(|| invariant("a packed read of a scalar place"))?,
        )?;
        let v = int_bits(&mut self.builder, v);
        // Entry index of the group of the packing-axis coordinate:
        // `(v / group) * fields + entry`.
        let group = self.builder.ins().iconst(types::I64, i64::from(schema.group));
        let fields = self.builder.ins().iconst(types::I64, i64::from(schema.fields));
        let group_index = self.builder.ins().udiv(v, group);
        let group_base = self.builder.ins().imul(group_index, fields);
        let entry_index = self.builder.ins().iadd_imm(group_base, i64::from(entry));
        match schema.encoding {
            repr::PlaneEncoding::Dense(_) => {
                // One storage element per entry.
                let (address, storage_dtype) =
                    self.plane_element_address(place, coords, &schema, Some(entry_index))?;
                let value = self.load_element(address, storage_dtype);
                self.store_result(into, value)
            }
            repr::PlaneEncoding::Packed { bits, .. } => {
                // Bits `[entry_index * bits, +bits)` of the row's u32 words,
                // little-endian within words. An entry that straddles a word
                // boundary spans two allocated words (rows are padded to
                // whole storage groups, so the next word exists whenever
                // the entry crosses).
                let bit = self.builder.ins().imul_imm(entry_index, i64::from(bits));
                let thirty_two = self.builder.ins().iconst(types::I64, 32);
                let word_index = self.builder.ins().udiv(bit, thirty_two);
                let (address, _) =
                    self.plane_element_address(place, coords, &schema, Some(word_index))?;
                let shift = self.builder.ins().urem(bit, thirty_two);
                let word = self
                    .builder
                    .ins()
                    .load(types::I32, MemFlags::trusted(), address, 0);
                let mask = if bits >= 32 {
                    self.builder.ins().iconst(types::I32, -1)
                } else {
                    self.builder.ins().iconst(types::I32, (1i64 << bits) - 1)
                };
                let low = self.builder.ins().ushr(word, shift);
                let single = self.builder.ins().band(low, mask);
                let end = self.builder.ins().iadd_imm(shift, i64::from(bits));
                let straddles = self
                    .builder
                    .ins()
                    .icmp(IntCC::UnsignedGreaterThan, end, thirty_two);
                // The next word is read only when the entry straddles; the
                // address is selected so the unconditional load never leaves
                // the plane's allocated words.
                let plus_four = self.builder.ins().iadd_imm(address, 4);
                let next_address = self.builder.ins().select(straddles, plus_four, address);
                let next = self
                    .builder
                    .ins()
                    .load(types::I32, MemFlags::trusted(), next_address, 0);
                let carry = self.builder.ins().isub(thirty_two, shift);
                let high = self.builder.ins().ishl(next, carry);
                let spliced = self.builder.ins().bor(low, high);
                let straddled = self.builder.ins().band(spliced, mask);
                let code = self.builder.ins().select(straddles, straddled, single);
                let value = self.builder.ins().fcvt_from_uint(types::F64, code);
                self.store_result(into, value)
            }
        }
    }

    /// Atomic read-modify-write of one dense element. `Serialized` is a
    /// plain load/combine/round/store (one participant owns the domain);
    /// `Device` is a compare/exchange loop on the element's 32-bit word.
    fn atomic(
        &mut self,
        place: usize,
        coords: &[usize],
        value: usize,
        op: AtomicOp,
        dtype: DType,
        mode: AtomicMode,
    ) -> Result<(), AssemblyFailure> {
        let (address, element_dtype) = self.place_address(place, coords)?;
        let operand = self.fetch(value)?;
        match mode {
            AtomicMode::Serialized => {
                let current = self.load_element(address, element_dtype);
                let combined = self.combine(current, operand, op, dtype)?;
                self.store_element(address, combined, element_dtype);
                Ok(())
            }
            AtomicMode::Device => {
                if !matches!(dtype, DType::F32 | DType::I32 | DType::U32) {
                    return Err(invariant(format!(
                        "a device atomic of {} elements reached emission",
                        dtype.name()
                    )));
                }
                // Compare/exchange loop on the element's 32-bit word: read
                // the word, combine at the element dtype exactly as the
                // serialized form does, and install the result only if the
                // word is unchanged.
                let retry = self.builder.create_block();
                let exit = self.builder.create_block();
                self.builder.ins().jump(retry, &[]);
                self.builder.switch_to_block(retry);
                let observed_bits = self
                    .builder
                    .ins()
                    .load(types::I32, MemFlags::trusted(), address, 0);
                let current = word_to_s_value(&mut self.builder, observed_bits, dtype);
                let combined = self.combine(current, operand, op, dtype)?;
                let next_bits = s_value_to_word(&mut self.builder, combined, dtype);
                let installed = self.builder.ins().atomic_cas(
                    MemFlags::trusted(),
                    address,
                    observed_bits,
                    next_bits,
                );
                let unchanged = self
                    .builder
                    .ins()
                    .icmp(IntCC::Equal, installed, observed_bits);
                self.builder.ins().brif(unchanged, exit, &[], retry, &[]);
                self.builder.seal_block(retry);
                self.builder.switch_to_block(exit);
                self.builder.seal_block(exit);
                Ok(())
            }
        }
    }

    /// The registry atomic law: `add` rounds once at the element dtype per
    /// update; `max`/`min` are exact and ignore a NaN operand.
    fn combine(
        &mut self,
        current: Value,
        operand: Value,
        op: AtomicOp,
        dtype: DType,
    ) -> Result<Value, AssemblyFailure> {
        Ok(match op {
            AtomicOp::Add => {
                let sum = self.builder.ins().fadd(current, operand);
                self.round_s(sum, dtype)
            }
            AtomicOp::Max | AtomicOp::Min => {
                let picked = if op == AtomicOp::Max {
                    self.builder.ins().fmax(current, operand)
                } else {
                    self.builder.ins().fmin(current, operand)
                };
                // Cranelift `fmax`/`fmin` propagate NaN; the registry
                // ignores a NaN operand, so a NaN side yields the other.
                let current_nan = self.builder.ins().fcmp(FloatCC::NotEqual, current, current);
                let operand_nan = self.builder.ins().fcmp(FloatCC::NotEqual, operand, operand);
                let unless_current_nan =
                    self.builder.ins().select(current_nan, operand, picked);
                self.builder
                    .ins()
                    .select(operand_nan, current, unless_current_nan)
            }
        })
    }

    /// The registry serial reduction of one axis of a place at fixed outer
    /// coordinates, under the schema's identity, accumulator, and tie rule.
    /// The operand's view-side shape is the place view's `ty`.
    fn fold(
        &mut self,
        into: usize,
        place: usize,
        axis: u32,
        coords: &[usize],
        op: R,
        schema: ReduceSchema,
    ) -> Result<(), AssemblyFailure> {
        let view = self.dense_view(place)?;
        let storage = view.storage;
        let element_dtype = self.layout_dtype(storage);
        let strides = self.storage_strides(storage);
        let len = self.extent_value(
            view.ty
                .axes
                .get(axis as usize)
                .ok_or_else(|| invariant("a fold names an axis outside its shape"))?,
        )?;
        let bytes = u64::from(element_dtype.bytes());
        let base = self.storage_ptr(storage)?;
        // The element address at reduced-axis coordinate `c`: the outer
        // coordinates (in axis order, the folded axis absent) plus `c`.
        let outer_positions: Vec<usize> = coords.to_vec();
        let element_address = |emit: &mut Self, c: Value| -> Result<Value, AssemblyFailure> {
            let mut coord_values: Vec<Value> = Vec::with_capacity(view.ty.axes.len());
            let mut outer_iter = outer_positions.iter();
            for axis_index in 0..view.ty.axes.len() {
                if axis_index == axis as usize {
                    coord_values.push(c);
                } else {
                    let position = outer_iter
                        .next()
                        .ok_or_else(|| invariant("a fold lacks an outer coordinate"))?;
                    let v = emit.fetch(*position)?;
                    coord_values.push(int_bits(&mut emit.builder, v));
                }
            }
            let residence = emit.transform_coords(&view, &coord_values)?;
            let mut offset = emit.builder.ins().iconst(types::I64, 0);
            for (axis_index, coord) in residence.iter().enumerate() {
                let stride = strides
                    .get(axis_index)
                    .copied()
                    .ok_or_else(|| invariant("a fold names an absent storage axis"))?;
                let stride = i64::try_from(stride)
                    .map_err(|_| invariant("a storage stride exceeds the addressable size domain"))?;
                let scaled = emit.builder.ins().imul_imm(*coord, stride);
                offset = emit.builder.ins().iadd(offset, scaled);
            }
            let byte_offset = if bytes == 1 {
                offset
            } else {
                emit.builder.ins().imul_imm(offset, bytes as i64)
            };
            Ok(emit.builder.ins().iadd(base, byte_offset))
        };
        let is_int = element_dtype.is_int();
        // The fold: ascending serial over the reduced axis, with the
        // registry identity and per-step rounding. Zero identities start at
        // coordinate 0; first-element identities start from coordinate 0's
        // element and fold from 1.
        let fhead = self.builder.create_block();
        let fbody = self.builder.create_block();
        let fexit = self.builder.create_block();
        let zero = self.builder.ins().iconst(types::I64, 0);
        let one = self.builder.ins().iconst(types::I64, 1);
        let (start_c, acc_dtypes, header_args): (Value, Vec<ir::Type>, Vec<ir::Value>) = match op
        {
            R::Sum if is_int => (
                zero,
                vec![types::I32],
                vec![self.builder.ins().iconst(types::I32, 0)],
            ),
            R::Sum => (
                zero,
                vec![types::F64],
                vec![self
                    .builder
                    .ins()
                    .f64const(ir::immediates::Ieee64::with_bits(0))],
            ),
            R::Max | R::Min => {
                let first = element_address(self, zero)?;
                if is_int {
                    (
                        one,
                        vec![types::I32],
                        vec![self
                            .builder
                            .ins()
                            .load(types::I32, MemFlags::trusted(), first, 0)],
                    )
                } else {
                    (
                        one,
                        vec![types::F64],
                        vec![self.load_element(first, element_dtype)],
                    )
                }
            }
            // (best value, best coordinate) — ascending order keeps the
            // smaller coordinate on ties.
            R::Argmax => {
                let first = element_address(self, zero)?;
                (
                    one,
                    vec![types::F64, types::I64],
                    vec![self.load_element(first, element_dtype), zero],
                )
            }
        };
        for ty in &acc_dtypes {
            self.builder.append_block_param(fhead, *ty);
            self.builder.append_block_param(fbody, *ty);
            self.builder.append_block_param(fexit, *ty);
        }
        self.builder.append_block_param(fhead, types::I64);
        self.builder.append_block_param(fbody, types::I64);
        let mut jump_args: Vec<ir::BlockArg> =
            header_args.iter().copied().map(Into::into).collect();
        jump_args.push(start_c.into());
        self.builder.ins().jump(fhead, &jump_args);
        self.builder.switch_to_block(fhead);
        let coordinate = self
            .builder
            .block_params(fhead)
            .last()
            .copied()
            .expect("the fold header has a coordinate");
        let accs: Vec<Value> = self.builder.block_params(fhead)[..acc_dtypes.len()].to_vec();
        let more = self
            .builder
            .ins()
            .icmp(IntCC::UnsignedLessThan, coordinate, len);
        let mut body_args: Vec<ir::BlockArg> = accs.iter().copied().map(Into::into).collect();
        body_args.push(coordinate.into());
        let exit_args: Vec<ir::BlockArg> = accs.iter().copied().map(Into::into).collect();
        self.builder
            .ins()
            .brif(more, fbody, &body_args, fexit, &exit_args);
        self.builder.switch_to_block(fexit);
        self.builder.seal_block(fexit);
        self.builder.switch_to_block(fbody);
        let coordinate = self
            .builder
            .block_params(fbody)
            .last()
            .copied()
            .expect("the fold body has a coordinate");
        let accs: Vec<Value> = self.builder.block_params(fbody)[..acc_dtypes.len()].to_vec();
        let address = element_address(self, coordinate)?;
        let mut next_accs: Vec<ir::Value> = Vec::with_capacity(accs.len());
        match op {
            R::Sum if is_int => {
                let raw = self
                    .builder
                    .ins()
                    .load(types::I32, MemFlags::trusted(), address, 0);
                let sum = self.builder.ins().iadd(accs[0], raw);
                next_accs.push(sum.into());
            }
            R::Sum => {
                let v = self.load_element(address, element_dtype);
                let sum = self.builder.ins().fadd(accs[0], v);
                // Rounds each step at the accumulator dtype.
                let rounded = self.round_s(sum, schema.accumulator);
                next_accs.push(rounded.into());
            }
            R::Max | R::Min => {
                if is_int {
                    let raw = self
                        .builder
                        .ins()
                        .load(types::I32, MemFlags::trusted(), address, 0);
                    let combined = if op == R::Max {
                        if element_dtype == DType::U32 {
                            self.builder.ins().umax(accs[0], raw)
                        } else {
                            self.builder.ins().smax(accs[0], raw)
                        }
                    } else if element_dtype == DType::U32 {
                        self.builder.ins().umin(accs[0], raw)
                    } else {
                        self.builder.ins().smin(accs[0], raw)
                    };
                    next_accs.push(combined.into());
                } else {
                    let v = self.load_element(address, element_dtype);
                    let import = if op == R::Max {
                        Import::Fmax
                    } else {
                        Import::Fmin
                    };
                    let combined =
                        self.call_import(import, &[accs[0].into(), v.into()]);
                    next_accs.push(combined.into());
                }
            }
            R::Argmax => {
                let v = self.load_element(address, element_dtype);
                let better = self.builder.ins().fcmp(FloatCC::GreaterThan, v, accs[0]);
                let value = self.builder.ins().select(better, v, accs[0]);
                let index = self.builder.ins().select(better, coordinate, accs[1]);
                next_accs.push(value.into());
                next_accs.push(index.into());
            }
        }
        let next_coordinate = self.builder.ins().iadd(coordinate, one);
        let mut back_args: Vec<ir::BlockArg> = next_accs.iter().copied().map(Into::into).collect();
        back_args.push(next_coordinate.into());
        self.builder.ins().jump(fhead, &back_args);
        self.builder.seal_block(fbody);
        self.builder.seal_block(fhead);
        // Finalize in the exit block.
        self.builder.switch_to_block(fexit);
        let final_accs: Vec<Value> = self.builder.block_params(fexit)[..acc_dtypes.len()].to_vec();
        let s_value = match op {
            R::Sum if is_int => int_value_of(&mut self.builder, final_accs[0], element_dtype),
            R::Sum => self.round_s(final_accs[0], schema.result),
            R::Max | R::Min if is_int => {
                int_value_of(&mut self.builder, final_accs[0], element_dtype)
            }
            R::Max | R::Min => self.round_s(final_accs[0], schema.result),
            R::Argmax => self
                .builder
                .ins()
                .fcvt_from_sint(types::F64, final_accs[1]),
        };
        self.store_result(into, s_value)
    }

    // -- values ---------------------------------------------------------------

    fn fetch(&mut self, position: usize) -> Result<Value, AssemblyFailure> {
        let binding = self
            .encoded
            .bindings
            .get(position)
            .ok_or_else(|| invariant("an operand position is not bound"))?;
        match binding {
            Binding::Word { index, dtype } => {
                let raw = word_value(&mut self.builder, self.scalars_ptr, *index);
                Ok(self.decode_word(raw, *dtype))
            }
            Binding::Local => {
                // An `Axis` binding reads the coordinate of its ordinal; a
                // serial-loop binder reads the loop variable; every other
                // local was defined exactly once before this use (K1
                // def/use seal).
                if let Some(axis) = self.encoded.axis_of.get(&position) {
                    let coordinate = self
                        .coords
                        .get(*axis)
                        .copied()
                        .ok_or_else(|| invariant("an axis value is read outside its domain"))?;
                    return Ok(self.builder.ins().fcvt_from_sint(types::F64, coordinate));
                }
                if let Some(var) = self.loop_vars.get(&position) {
                    return Ok(*var);
                }
                Ok(self.locals[&position])
            }
            Binding::Tensor { .. } => Err(invariant(
                "a whole-tensor place was used as a kernel scalar",
            )),
        }
    }

    fn store_result(
        &mut self,
        position: usize,
        value: Value,
    ) -> Result<(), AssemblyFailure> {
        let binding = self
            .encoded
            .bindings
            .get(position)
            .ok_or_else(|| invariant("a result position is not bound"))?
            .clone();
        match binding {
            Binding::Word { index, dtype } => {
                let word = self.encode_word(value, dtype);
                store_word(&mut self.builder, self.scalars_ptr, index, word);
                Ok(())
            }
            Binding::Local => {
                self.locals.insert(position, value);
                Ok(())
            }
            Binding::Tensor { .. } => Err(invariant(
                "a whole-tensor place was written as a kernel scalar",
            )),
        }
    }

    fn extent_value(&mut self, expr: &ExtentExpr) -> Result<Value, AssemblyFailure> {
        Ok(match expr {
            ExtentExpr::Static(n) => {
                let signed = i64::try_from(*n)
                    .map_err(|_| invariant("a static extent exceeds i64 at emission"))?;
                self.builder.ins().iconst(types::I64, signed)
            }
            ExtentExpr::Sym(sym) => match sym.as_constant() {
                Some(value) => self.builder.ins().iconst(types::I64, value),
                None => {
                    return Err(invariant(
                        "an unresolved planning symbol survived the seal into emission",
                    ))
                }
            },
            ExtentExpr::Runtime(id) => {
                let index = self.extent_word_index(*id)?;
                word_value(&mut self.builder, self.scalars_ptr, index)
            }
        })
    }

    fn extent_word_index(&self, id: seismic_lang::types::RuntimeExtentId) -> Result<usize, AssemblyFailure> {
        let descriptor = &self.encoded.descriptor;
        let position = descriptor
            .runtime_extents
            .iter()
            .position(|candidate| *candidate == id)
            .ok_or_else(|| invariant("a runtime extent read by the kernel was not registered"))?;
        Ok(descriptor.words.len() + position)
    }

    fn word_value(&mut self, index: usize) -> Value {
        word_value(&mut self.builder, self.scalars_ptr, index)
    }

    /// Call one host import and return its single result.
    fn call_import(&mut self, import: Import, args: &[ir::Value]) -> Value {
        let reference = self
            .import_refs
            .get(&import)
            .copied()
            .expect("imports are created before emission");
        let call = self.builder.ins().call(reference, args);
        self.builder.inst_results(call)[0]
    }

    /// Decode a raw word into an f64 S-value at `dtype` (bits in the low bytes).
    fn decode_word(&mut self, word: Value, dtype: DType) -> Value {
        match dtype {
            DType::F32 => {
                let bits = self.builder.ins().ireduce(types::I32, word);
                let bits = self.builder.ins().bitcast(types::F32, MemFlags::new(), bits);
                self.builder.ins().fpromote(types::F64, bits)
            }
            DType::I32 => {
                let v = self.builder.ins().ireduce(types::I32, word);
                let v = self.builder.ins().sextend(types::I64, v);
                self.builder.ins().fcvt_from_sint(types::F64, v)
            }
            DType::U32 => {
                let v = self.builder.ins().ireduce(types::I32, word);
                let v = self.builder.ins().uextend(types::I64, v);
                self.builder.ins().fcvt_from_uint(types::F64, v)
            }
            DType::Bool => {
                let bit = self.builder.ins().band_imm(word, 1);
                self.builder.ins().fcvt_from_uint(types::F64, bit)
            }
            DType::F16 => {
                let bits = self.builder.ins().ireduce(types::I32, word);
                let bits = self.builder.ins().band_imm(bits, 0xffff);
                let f16 = self.call_import(Import::F16Load, &[bits.into()]);
                self.builder.ins().fpromote(types::F64, f16)
            }
            DType::BF16 => {
                let bits = self.builder.ins().ireduce(types::I32, word);
                let bits = self.builder.ins().band_imm(bits, 0xffff);
                let shifted = self.builder.ins().ishl_imm(bits, 16);
                let f32 = self.builder.ins().bitcast(types::F32, MemFlags::new(), shifted);
                self.builder.ins().fpromote(types::F64, f32)
            }
        }
    }

    /// Encode an f64 S-value into a raw word at `dtype`.
    fn encode_word(&mut self, value: Value, dtype: DType) -> Value {
        match dtype {
            DType::F32 => {
                let f32 = self.builder.ins().fdemote(types::F32, value);
                let bits = self.builder.ins().bitcast(types::I32, MemFlags::new(), f32);
                self.builder.ins().uextend(types::I64, bits)
            }
            DType::I32 => {
                let v = self.builder.ins().fcvt_to_sint(types::I64, value);
                let v = self.builder.ins().ireduce(types::I32, v);
                self.builder.ins().uextend(types::I64, v)
            }
            DType::U32 => {
                let v = self.builder.ins().fcvt_to_uint(types::I64, value);
                let v = self.builder.ins().ireduce(types::I32, v);
                self.builder.ins().uextend(types::I64, v)
            }
            DType::Bool => {
                let zero = self
                    .builder
                    .ins()
                    .f64const(ir::immediates::Ieee64::with_bits(0));
                let bit = self.builder.ins().fcmp(FloatCC::NotEqual, value, zero);
                self.builder.ins().uextend(types::I64, bit)
            }
            DType::F16 => {
                let f32 = self.builder.ins().fdemote(types::F32, value);
                let bits = self.call_import(Import::F16Store, &[f32.into()]);
                self.builder.ins().uextend(types::I64, bits)
            }
            DType::BF16 => {
                let f32 = self.builder.ins().fdemote(types::F32, value);
                let bits = self.builder.ins().bitcast(types::I32, MemFlags::new(), f32);
                let high = self.builder.ins().ushr_imm(bits, 16);
                self.builder.ins().uextend(types::I64, high)
            }
        }
    }

    /// Round one f64 S-value once at `dtype` (the registry reference model).
    fn round_s(&mut self, value: Value, dtype: DType) -> Value {
        match dtype {
            DType::F32 => {
                let f32 = self.builder.ins().fdemote(types::F32, value);
                self.builder.ins().fpromote(types::F64, f32)
            }
            _ => {
                let tag = self
                    .builder
                    .ins()
                    .iconst(types::I32, i64::from(dtype_tag(dtype)));
                self.call_import(Import::RoundTo, &[tag.into(), value.into()])
            }
        }
    }

    fn load_element(&mut self, address: Value, dtype: DType) -> Value {
        match dtype {
            DType::F32 => {
                let v = self
                    .builder
                    .ins()
                    .load(types::F32, MemFlags::trusted(), address, 0);
                self.builder.ins().fpromote(types::F64, v)
            }
            DType::I32 => {
                let v = self
                    .builder
                    .ins()
                    .load(types::I32, MemFlags::trusted(), address, 0);
                let v = self.builder.ins().sextend(types::I64, v);
                self.builder.ins().fcvt_from_sint(types::F64, v)
            }
            DType::U32 => {
                let v = self
                    .builder
                    .ins()
                    .load(types::I32, MemFlags::trusted(), address, 0);
                let v = self.builder.ins().uextend(types::I64, v);
                self.builder.ins().fcvt_from_uint(types::F64, v)
            }
            DType::Bool => {
                let v = self
                    .builder
                    .ins()
                    .load(types::I8, MemFlags::trusted(), address, 0);
                let zero = self.builder.ins().iconst(types::I8, 0);
                let bit = self.builder.ins().icmp(IntCC::NotEqual, v, zero);
                f64_of_bool(&mut self.builder, bit)
            }
            DType::F16 => {
                let bits = self
                    .builder
                    .ins()
                    .load(types::I16, MemFlags::trusted(), address, 0);
                let bits = self.builder.ins().uextend(types::I32, bits);
                let v = self.call_import(Import::F16Load, &[bits.into()]);
                self.builder.ins().fpromote(types::F64, v)
            }
            DType::BF16 => {
                let bits = self
                    .builder
                    .ins()
                    .load(types::I16, MemFlags::trusted(), address, 0);
                let bits = self.builder.ins().uextend(types::I32, bits);
                let shifted = self.builder.ins().ishl_imm(bits, 16);
                let v = self.builder.ins().bitcast(types::F32, MemFlags::new(), shifted);
                self.builder.ins().fpromote(types::F64, v)
            }
        }
    }

    fn store_element(&mut self, address: Value, value: Value, dtype: DType) {
        match dtype {
            DType::F32 => {
                let v = self.builder.ins().fdemote(types::F32, value);
                self.builder.ins().store(MemFlags::trusted(), v, address, 0);
            }
            DType::I32 => {
                let v = self.builder.ins().fcvt_to_sint(types::I64, value);
                let v = self.builder.ins().ireduce(types::I32, v);
                self.builder.ins().store(MemFlags::trusted(), v, address, 0);
            }
            DType::U32 => {
                let v = self.builder.ins().fcvt_to_uint(types::I64, value);
                let v = self.builder.ins().ireduce(types::I32, v);
                self.builder.ins().store(MemFlags::trusted(), v, address, 0);
            }
            DType::Bool => {
                let bit = bool_of(&mut self.builder, value);
                self.builder.ins().store(MemFlags::trusted(), bit, address, 0);
            }
            DType::F16 => {
                let f = self.builder.ins().fdemote(types::F32, value);
                let bits = self.call_import(Import::F16Store, &[f.into()]);
                let bits = self.builder.ins().ireduce(types::I16, bits);
                self.builder.ins().store(MemFlags::trusted(), bits, address, 0);
            }
            DType::BF16 => {
                let f = self.builder.ins().fdemote(types::F32, value);
                let bits = self.builder.ins().bitcast(types::I32, MemFlags::new(), f);
                let high = self.builder.ins().ushr_imm(bits, 16);
                let high = self.builder.ins().ireduce(types::I16, high);
                self.builder.ins().store(MemFlags::trusted(), high, address, 0);
            }
        }
    }
}

/// The SSA result positions one instruction defines (nested bodies'
/// inner definitions belong to their own emission).
fn instruction_results(instruction: &Instruction) -> Vec<usize> {
    match instruction {
        Instruction::Const { into, .. }
        | Instruction::Extent { into, .. }
        | Instruction::Unary { into, .. }
        | Instruction::Binary { into, .. }
        | Instruction::Compare { into, .. }
        | Instruction::Math { into, .. }
        | Instruction::Fma { into, .. }
        | Instruction::Cast { into, .. }
        | Instruction::Select { into, .. }
        | Instruction::TableLookup { into, .. }
        | Instruction::Load { into, .. }
        | Instruction::PackedPlaneRead { into, .. }
        | Instruction::PlaneLoad { into, .. }
        | Instruction::Fold { into, .. } => vec![*into],
        Instruction::Branch { joins, .. } => joins.iter().map(|join| join.joined).collect(),
        Instruction::Repeat { carries, .. } => carries
            .iter()
            .flat_map(|carry| [carry.current, carry.result])
            .collect(),
        Instruction::Check { guarded, .. } => {
            guarded.iter().flat_map(instruction_results).collect()
        }
        Instruction::Store { .. }
        | Instruction::PlaneStore { .. }
        | Instruction::Atomic { .. }
        | Instruction::Publish { .. }
        | Instruction::Barrier
        | Instruction::GridBarrier => Vec::new(),
    }
}

// ---------------------------------------------------------------------------
// Word / value helpers (builder-only)
// ---------------------------------------------------------------------------

/// A stable host tag per dtype (the `seismic_round_to` convention).
fn dtype_tag(dtype: DType) -> i32 {
    match dtype {
        DType::F32 => 0,
        DType::BF16 => 1,
        DType::F16 => 2,
        DType::I32 => 3,
        DType::U32 => 4,
        DType::Bool => 5,
    }
}

/// Load one scalar word (raw i64 bits at 8-byte slot `index`).
fn word_value(builder: &mut FunctionBuilder<'_>, scalars: Value, index: usize) -> Value {
    let address = builder.ins().iadd_imm(scalars, (index * 8) as i64);
    builder
        .ins()
        .load(types::I64, MemFlags::trusted(), address, 0)
}

/// Store one scalar word.
fn store_word(builder: &mut FunctionBuilder<'_>, scalars: Value, index: usize, word: Value) {
    let address = builder.ins().iadd_imm(scalars, (index * 8) as i64);
    builder.ins().store(MemFlags::trusted(), word, address, 0);
}

/// Decode one observed 32-bit word into an f64 S-value at `dtype`.
fn word_to_s_value(builder: &mut FunctionBuilder<'_>, bits: Value, dtype: DType) -> Value {
    match dtype {
        DType::F32 => {
            let f = builder.ins().bitcast(types::F32, MemFlags::new(), bits);
            builder.ins().fpromote(types::F64, f)
        }
        DType::I32 => {
            let wide = builder.ins().sextend(types::I64, bits);
            builder.ins().fcvt_from_sint(types::F64, wide)
        }
        DType::U32 => {
            let wide = builder.ins().uextend(types::I64, bits);
            builder.ins().fcvt_from_uint(types::F64, wide)
        }
        // Device atomics are 32-bit elements only, checked at emission.
        _ => unreachable!("a device atomic of a non-32-bit element reached word decoding"),
    }
}

/// Encode an f64 S-value into its 32-bit word at `dtype`.
fn s_value_to_word(builder: &mut FunctionBuilder<'_>, value: Value, dtype: DType) -> Value {
    match dtype {
        DType::F32 => {
            let f = builder.ins().fdemote(types::F32, value);
            builder.ins().bitcast(types::I32, MemFlags::new(), f)
        }
        DType::I32 => {
            let wide = builder.ins().fcvt_to_sint(types::I64, value);
            builder.ins().ireduce(types::I32, wide)
        }
        _ => {
            let wide = builder.ins().fcvt_to_uint(types::I64, value);
            builder.ins().ireduce(types::I32, wide)
        }
    }
}

/// A `bool` value (I8) from an f64 S-value.
fn bool_of(builder: &mut FunctionBuilder<'_>, value: Value) -> Value {
    let zero = builder
        .ins()
        .f64const(ir::immediates::Ieee64::with_bits(0));
    builder.ins().fcmp(FloatCC::NotEqual, value, zero)
}

/// An f64 S-value (0.0/1.0) from a `bool` value.
fn f64_of_bool(builder: &mut FunctionBuilder<'_>, bit: Value) -> Value {
    let wide = builder.ins().uextend(types::I64, bit);
    builder.ins().fcvt_from_uint(types::F64, wide)
}

/// An i64 bit pattern from an integer S-value (exact for representable ints).
fn int_bits(builder: &mut FunctionBuilder<'_>, value: Value) -> Value {
    builder.ins().fcvt_to_sint(types::I64, value)
}

/// An f64 S-value from i32 bits reinterpreted at `dtype`.
fn int_value_of(builder: &mut FunctionBuilder<'_>, bits: Value, dtype: DType) -> Value {
    match dtype {
        DType::U32 => {
            let u = builder.ins().uextend(types::I64, bits);
            builder.ins().fcvt_from_uint(types::F64, u)
        }
        _ => {
            let s = builder.ins().sextend(types::I64, bits);
            builder.ins().fcvt_from_sint(types::F64, s)
        }
    }
}
