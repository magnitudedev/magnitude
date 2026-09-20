//! Native encoding: one Cranelift function per resolved launch, assembled
//! from the whole resolved plan.
//!
//! Emission is mechanical: every `CpuOp` is accepted — the match
//! over the sealed enum is exhaustive and no selected opcode is rejected —
//! and emitters make no allocation, geometry, synchronization, algorithm, or
//! precision decision. Geometry comes from the resolved launch (the common
//! `LinearIterationMap`, grid-stride over the exact retained total), safety
//! checks are exactly the planned opcodes (none added, none omitted), and
//! arithmetic follows the registry reference model: binary64 computation
//! rounded once at the result dtype, with transcendentals through the
//! versioned `seismic_math` host sequences — the same code path the
//! reference interpreter runs, so the bits agree by construction.
//!
//! Per-launch native ABI (the worker entry):
//! `fn(buffers: *const *mut u8, scalars: *mut u64, scratch: *mut u8, participant: u64) -> i32`
//! — the buffer table is the launch's resolved binding groups in slot order
//! (plus the status area as one extra entry when the launch writes status),
//! and the scalar word table is `[slot words][ABI words][extent values]`
//! followed by the participant count, exactly as `LaunchDescriptor` records.

use crate::physical::{CheckKind, ConstValue, CpuDialect, CpuOp, CpuOpKind, EncodedLaunch};
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
use seismic_compiler::pipeline::EncodedPlan;
use seismic_lang::{
    intrinsics::MathOp,
    logical::{GraphValueId, RuntimeScalarExpr, SliceAxis, ViewTransform},
    repr,
    sym::Sym,
    syntax::ast::{BinaryOp, UnaryOp},
    types::{DType, ExtentExpr, RuntimeExtentId, ValuePath},
};
use seismic_realization::dispatch::{LinearIterationMap, LinearTotal};
use seismic_realization::executable::{
    self as exec, ExecutionExpr, ResolvedExecutorScalar, ResolvedKernelStep, ResolvedLaunch,
    ResolvedLaunchId, ResolvedSchedule, ResolvedStep, ResolvedStorageId, ResolvedTransport,
    StatusFieldId,
};
use std::collections::{BTreeMap, BTreeSet};

// ---------------------------------------------------------------------------
// Launch descriptors (the runtime glue's view of one compiled launch)
// ---------------------------------------------------------------------------

/// Where one scalar word of a launch's scalar table comes from (and goes to).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum WordSource {
    Slot(exec::ResolvedExecutorScalarId),
    /// Root ABI scalar input field ordinal (`abi.scalars.fields`).
    AbiInput {
        field: usize,
    },
    /// Compiler-owned result scalar block word ordinal.
    AbiResult {
        field: usize,
    },
}

/// One compiled launch: its worker entry and its tables.
pub(crate) struct LaunchDescriptor {
    /// Storage ids in buffer-table order (binding groups by slot).
    pub storage_table: Vec<ResolvedStorageId>,
    /// Whether the status area is appended as one extra buffer-table entry.
    pub has_status: bool,
    /// Scalar word table sources (slots then ABI words); extent values and
    /// the participant count follow in this order.
    pub words: Vec<WordSource>,
    pub extent_words: Vec<RuntimeExtentId>,
    pub work_items: ExecutionExpr,
    pub participants: ExecutionExpr,
}

/// The executable CPU artifact: JIT memory plus one entry per launch id.
pub struct Kernel {
    pub(crate) plan: std::sync::Arc<exec::ResolvedPlan<CpuDialect>>,
    pub(crate) memory: Option<JITModule>,
    pub(crate) launches: BTreeMap<ResolvedLaunchId, LaunchEntry>,
    /// Result scalar block words (field id and dtype, in ordinal order).
    pub(crate) result_words: Vec<(exec::ResultScalarFieldId, DType)>,
    /// Number of resolved executor scalar slots.
    pub(crate) slot_count: u64,
}

pub(crate) struct LaunchEntry {
    pub entry: workers::PhaseEntry,
    pub descriptor: LaunchDescriptor,
}

impl Kernel {
    pub fn launch_count(&self) -> usize {
        self.launches.len()
    }
}

impl Drop for Kernel {
    fn drop(&mut self) {
        if let Some(module) = self.memory.take() {
            // No generated pointer escapes its Kernel owner.
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
    Decode,
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
            Import::Decode => "seismic_decode",
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
            Import::Decode => {
                param(&mut signature, types::I32);
                param(&mut signature, types::I64);
                param(&mut signature, types::I64);
                signature.returns.push(AbiParam::new(types::F32));
            }
        }
        signature
    }
}

// ---------------------------------------------------------------------------
// Assembly
// ---------------------------------------------------------------------------

/// Assemble the encoded plan into one executable `Kernel`: JIT-compile one
/// function per resolved launch and record its tables.
pub fn assemble(encoded: EncodedPlan<CpuDialect, EncodedLaunch>) -> Result<Kernel, String> {
    let plan = encoded.resolved.clone();
    let mut launches: Vec<(ResolvedLaunchId, ResolvedLaunch<CpuDialect>)> = Vec::new();
    collect_launches(&encoded.steps, &mut launches)?;
    if launches.is_empty() {
        return Err("the CPU plan contains no native launch".into());
    }
    let policy = crate::codegen::Policy::host()?;
    let call_conv = policy.call_conv();
    let isa = policy.target()?.isa;
    let mut jit = JITBuilder::with_isa(isa, default_libcall_names());
    for (name, address) in host_symbols() {
        jit.symbol(name, address);
    }
    let mut module = JITModule::new(jit);
    if module.target_config().pointer_type() != types::I64 {
        return Err("the CPU backend requires 64-bit pointers".into());
    }
    let result_words = result_word_list(&plan);
    let slot_count = count_slots(&plan.entry.schedule);
    let mut entries = BTreeMap::new();
    let mut declared_launches = Vec::new();
    for (ordinal, (id, launch)) in launches.iter().enumerate() {
        let (function, imports, descriptor) = encode_launch(&plan, launch, call_conv)?;
        let mut context = module.make_context();
        context.func = function;
        for (reference, import) in imports {
            let signature = import.signature(call_conv);
            let imported = module
                .declare_function(import.symbol(), Linkage::Import, &signature)
                .map_err(|error| error.to_string())?;
            let native = module.declare_func_in_func(imported, &mut context.func);
            context.func.dfg.ext_funcs[reference] = context.func.dfg.ext_funcs[native].clone();
        }
        let declared = module
            .declare_function(
                &format!("seismic_cpu_launch_{ordinal}"),
                Linkage::Local,
                &context.func.signature,
            )
            .map_err(|error| error.to_string())?;
        context.func.name = ir::UserFuncName::user(0, declared.as_u32());
        module
            .define_function(declared, &mut context)
            .map_err(|error| format!("CPU compilation of launch {}: {error:?}", id.0))?;
        module.clear_context(&mut context);
        declared_launches.push((*id, declared, descriptor));
    }
    module
        .finalize_definitions()
        .map_err(|error| error.to_string())?;
    for (id, declared, descriptor) in declared_launches {
        entries.insert(
            id,
            LaunchEntry {
                entry: unsafe {
                    std::mem::transmute::<*const u8, workers::PhaseEntry>(
                        module.get_finalized_function(declared),
                    )
                },
                descriptor,
            },
        );
    }
    Ok(Kernel {
        plan,
        memory: Some(module),
        launches: entries,
        result_words,
        slot_count,
    })
}

/// Result scalar block words in ordinal order (ranges contribute two).
pub(crate) fn result_word_list(
    plan: &exec::ResolvedPlan<CpuDialect>,
) -> Vec<(exec::ResultScalarFieldId, DType)> {
    let mut words = Vec::new();
    for binding in &plan.abi.results {
        match binding {
            exec::ResultBinding::Scalar { field, dtype, .. } => words.push((*field, *dtype)),
            exec::ResultBinding::Range { start, end, .. } => {
                words.push((*start, DType::I32));
                words.push((*end, DType::I32));
            }
            exec::ResultBinding::Buffer { .. } => {}
        }
    }
    words
}

fn collect_launches(
    steps: &[seismic_compiler::pipeline::EncodedStep<CpuDialect, EncodedLaunch>],
    out: &mut Vec<(ResolvedLaunchId, ResolvedLaunch<CpuDialect>)>,
) -> Result<(), String> {
    for step in steps {
        match step {
            seismic_compiler::pipeline::EncodedStep::Launch { resolved, .. } => {
                out.push((resolved.id, resolved.clone()));
            }
            seismic_compiler::pipeline::EncodedStep::Call { encoded, .. } => {
                collect_launches(&encoded.steps, out)?;
            }
            seismic_compiler::pipeline::EncodedStep::If {
                then_steps,
                else_steps,
                ..
            } => {
                collect_launches(then_steps, out)?;
                collect_launches(else_steps, out)?;
            }
            seismic_compiler::pipeline::EncodedStep::Repeat { body, .. } => {
                collect_launches(body, out)?;
            }
        }
    }
    Ok(())
}

fn count_slots(schedule: &ResolvedSchedule<CpuDialect>) -> u64 {
    let mut max = 0u64;
    fn walk(schedule: &ResolvedSchedule<CpuDialect>, max: &mut u64) {
        for step in schedule.steps.iter() {
            match step {
                ResolvedStep::Launch(launch) => {
                    for kernel_step in launch.kernel.steps.iter() {
                        if let ResolvedKernelStep::Mapped { bindings, .. } = kernel_step {
                            for (_, transport) in bindings {
                                record_transport(transport, max);
                            }
                        }
                    }
                }
                ResolvedStep::Call(call) => walk(&call.body.schedule, max),
                ResolvedStep::If(if_step) => {
                    record_scalar(&if_step.condition, max);
                    walk(&if_step.then_schedule, max);
                    walk(&if_step.else_schedule, max);
                }
                ResolvedStep::Repeat(repeat) => {
                    *max = (*max).max(repeat.binder.0 + 1);
                    record_scalar(&repeat.range.start, max);
                    record_scalar(&repeat.range.end, max);
                    for carry in &repeat.carried {
                        record_transport(carry, max);
                    }
                    walk(&repeat.body, max);
                }
            }
        }
    }
    fn record_transport(transport: &ResolvedTransport, max: &mut u64) {
        match transport {
            ResolvedTransport::ExecutorScalar(scalar) => record_scalar(scalar, max),
            ResolvedTransport::Tuple(items) => {
                for item in items.iter() {
                    record_transport(item, max);
                }
            }
            _ => {}
        }
    }
    fn record_scalar(scalar: &ResolvedExecutorScalar, max: &mut u64) {
        if let ResolvedExecutorScalar::Slot { slot, .. } = scalar {
            *max = (*max).max(slot.0 + 1);
        }
    }
    walk(schedule, &mut max);
    max
}

// ---------------------------------------------------------------------------
// One launch's emission
// ---------------------------------------------------------------------------

/// Classification of one launch binding (word / tensor / kernel / tuple).
#[derive(Clone)]
enum Binding {
    Word {
        index: usize,
        dtype: DType,
    },
    Tensor {
        storage: ResolvedStorageId,
        transform: ViewTransform,
    },
    Kernel,
    Tuple(Vec<Binding>),
    /// A computed control scalar: a retained execution expression.
    Computed {
        expr: ExecutionExpr,
        dtype: DType,
    },
    Void,
}

struct LaunchEmit<'a> {
    plan: &'a exec::ResolvedPlan<CpuDialect>,
    storage_slots: BTreeMap<ResolvedStorageId, usize>,
    words: Vec<WordSource>,
    word_of_slot: BTreeMap<exec::ResolvedExecutorScalarId, usize>,
    /// Deduplicated ABI input words by (path, dtype); range endpoints
    /// (leaf > 0) always occupy their own word.
    abi_word_of: BTreeMap<(ValuePath, u8), usize>,
    extent_words: Vec<RuntimeExtentId>,
    extent_word_of: BTreeMap<RuntimeExtentId, usize>,
    status_offsets: BTreeMap<StatusFieldId, u32>,
    import_refs: BTreeMap<Import, ir::FuncRef>,
    imports: Vec<(ir::FuncRef, Import)>,
}

/// Per-point emission state of one `Mapped` kernel step.
struct PointEmit<'a, 'f, 'p> {
    builder: &'a mut FunctionBuilder<'f>,
    emit: &'a mut LaunchEmit<'p>,
    table: Value,
    scalars: Value,
    status_ptr: Option<Value>,
    storage_ptrs: BTreeMap<ResolvedStorageId, Value>,
    /// Kernel-local values by binding position.
    locals: BTreeMap<usize, Value>,
    /// Coordinates of the current point (i64, one per iteration axis).
    coords: Vec<Value>,
    /// The linear coordinate of the current participant visit.
    linear: Value,
    /// The participant count of this launch.
    participants: Value,
    /// Serial-loop binders by binding position (the loop variable).
    loop_vars: BTreeMap<usize, Value>,
    /// Axis-coordinate binders by binding position (absorbed independent
    /// loop binders resolve to their iteration axis coordinate).
    axis_vars: BTreeMap<usize, usize>,
}

impl<'a> LaunchEmit<'a> {
    fn extent_index(&mut self, id: RuntimeExtentId) -> usize {
        if let Some(index) = self.extent_word_of.get(&id) {
            return *index;
        }
        let index = self.extent_words.len();
        self.extent_words.push(id);
        self.extent_word_of.insert(id, index);
        index
    }

    /// Classify one transport, registering scalar words as a side effect.
    /// `leaf` is the tuple-leaf ordinal (range endpoint selection).
    fn classify(&mut self, transport: &ResolvedTransport, leaf: usize) -> Result<Binding, String> {
        Ok(match transport {
            ResolvedTransport::Void => Binding::Void,
            ResolvedTransport::Kernel(_) => Binding::Kernel,
            ResolvedTransport::ExecutorScalar(scalar) => match scalar {
                ResolvedExecutorScalar::Computed { expr, dtype } => Binding::Computed {
                    expr: expr.clone(),
                    dtype: *dtype,
                },
                ResolvedExecutorScalar::Slot { slot, dtype } => {
                    let index = match self.word_of_slot.get(slot) {
                        Some(index) => *index,
                        None => {
                            let index = self.words.len();
                            self.words.push(WordSource::Slot(*slot));
                            self.word_of_slot.insert(*slot, index);
                            index
                        }
                    };
                    Binding::Word {
                        index,
                        dtype: *dtype,
                    }
                }
                ResolvedExecutorScalar::Abi { path, dtype, .. } => {
                    if leaf == 0 {
                        if let Some(index) =
                            self.abi_word_of.get(&(path.clone(), dtype_key(*dtype)))
                        {
                            return Ok(Binding::Word {
                                index: *index,
                                dtype: *dtype,
                            });
                        }
                    }
                    let field = self.abi_field(path, *dtype, leaf)?;
                    let index = self.words.len();
                    self.words.push(field);
                    if leaf == 0 {
                        self.abi_word_of
                            .insert(((*path).clone(), dtype_key(*dtype)), index);
                    }
                    Binding::Word {
                        index,
                        dtype: *dtype,
                    }
                }
                ResolvedExecutorScalar::Result { .. } => {
                    return Err("a result scalar cannot be used as an input binding".into());
                }
            },
            ResolvedTransport::Storage(views) => {
                let mut storage = None;
                let mut transform = ViewTransform::Identity;
                for view in views.iter() {
                    if !self.storage_slots.contains_key(&view.storage) {
                        return Err(format!(
                            "CPU launch omits a binding for storage#{} (the family builder \
                             must bind every storage a launch reads or writes)",
                            view.storage.0
                        ));
                    }
                    if storage.is_none() {
                        storage = Some(view.storage);
                        transform = view.transform.clone();
                    }
                }
                Binding::Tensor {
                    storage: storage.ok_or("a tensor transport has no plane")?,
                    transform,
                }
            }
            ResolvedTransport::Tuple(items) => {
                let mut leaves = Vec::new();
                for (ordinal, item) in items.iter().enumerate() {
                    leaves.push(self.classify(item, ordinal)?);
                }
                Binding::Tuple(leaves)
            }
        })
    }

    /// The ABI word source of one scalar leaf: an input field, or (for
    /// result leaves) the compiler-owned result scalar block.
    fn abi_field(
        &mut self,
        path: &ValuePath,
        dtype: DType,
        leaf: usize,
    ) -> Result<WordSource, String> {
        // A top-level leaf path renders as "()" ; match plain parameter names.
        let suffix = if path.0.is_empty() {
            String::new()
        } else {
            path.to_string()
        };
        let fields = &self.plan.abi.scalars.fields;
        // Range leaves: two adjacent start/end fields of one parameter.
        let range_start = fields.iter().position(|field| {
            field.parameter.dtype == dtype
                && field.parameter.range.is_some()
                && (suffix.is_empty() || field.parameter.name.ends_with(&suffix))
        });
        if let Some(start) = range_start {
            let field = if leaf % 2 == 0 { start } else { start + 1 };
            return Ok(WordSource::AbiInput { field });
        }
        let mut matches = fields
            .iter()
            .enumerate()
            .filter(|(_, field)| {
                field.parameter.dtype == dtype
                    && (suffix.is_empty() || field.parameter.name.ends_with(&suffix))
            })
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        if suffix.is_empty() {
            // Top-level leaves: names are plain parameter names.
            matches.retain(|&index| {
                !fields[index].parameter.name.contains('.')
                    && fields[index].parameter.range.is_none()
                    && fields[index].parameter.index_bound.is_none()
            });
        }
        if matches.len() == 1 {
            return Ok(WordSource::AbiInput { field: matches[0] });
        }
        // Not an input: a result scalar leaf of the root ABI.
        let ordinal = self.result_ordinal(path, dtype)?;
        Ok(WordSource::AbiResult { field: ordinal })
    }

    fn result_ordinal(&self, path: &ValuePath, dtype: DType) -> Result<usize, String> {
        let mut ordinal = 0usize;
        for binding in &self.plan.abi.results {
            match binding {
                exec::ResultBinding::Buffer { .. } => {}
                exec::ResultBinding::Scalar {
                    path: p, dtype: d, ..
                } => {
                    if p == path && d == &dtype {
                        return Ok(ordinal);
                    }
                    ordinal += 1;
                }
                exec::ResultBinding::Range { path: p, .. } => {
                    if p == path && dtype == DType::I32 {
                        return Ok(ordinal);
                    }
                    ordinal += 2;
                }
            }
        }
        Err(format!(
            "CPU scalar leaf {path} of {} is neither an ABI input nor a result field",
            dtype.name()
        ))
    }
}

/// One classified mapped step.
struct MappedStep {
    iteration: LinearIterationMap,
    bindings: Vec<(GraphValueId, Binding)>,
    ops: Vec<CpuOp>,
}

fn encode_launch(
    plan: &exec::ResolvedPlan<CpuDialect>,
    launch: &ResolvedLaunch<CpuDialect>,
    call_conv: CallConv,
) -> Result<(ir::Function, Vec<(ir::FuncRef, Import)>, LaunchDescriptor), String> {
    // Buffer table: binding groups by slot, members in order.
    let mut groups = launch.bindings.clone();
    groups.sort_by_key(|group| group.slot);
    let storage_table: Vec<ResolvedStorageId> = groups
        .iter()
        .flat_map(|group| group.members.iter().map(|member| member.storage))
        .collect();
    let storage_slots: BTreeMap<ResolvedStorageId, usize> = storage_table
        .iter()
        .copied()
        .enumerate()
        .map(|(index, storage)| (storage, index))
        .collect();

    // Status fields referenced by planned checks.
    let mut status_ids: BTreeSet<StatusFieldId> = BTreeSet::new();
    for kernel_step in launch.kernel.steps.iter() {
        if let ResolvedKernelStep::Mapped { ops, .. } = kernel_step {
            for op in ops.iter() {
                if let CpuOpKind::Check { status, .. } = &op.kind {
                    status_ids.insert(*status);
                }
            }
        }
    }
    let mut status_offsets = BTreeMap::new();
    for id in status_ids {
        let index = plan
            .abi
            .status
            .as_ref()
            .and_then(|status| status.fields.iter().position(|field| field.id == id))
            .ok_or_else(|| {
                format!(
                    "CPU launch {} writes status field {:?} absent from the root status \
                     binding",
                    launch.id.0, id
                )
            })?;
        status_offsets.insert(id, (index as u32) * 4);
    }
    let has_status = !status_offsets.is_empty();

    let mut emit = LaunchEmit {
        plan,
        storage_slots,
        words: Vec::new(),
        word_of_slot: BTreeMap::new(),
        abi_word_of: BTreeMap::new(),
        extent_words: Vec::new(),
        extent_word_of: BTreeMap::new(),
        status_offsets,
        import_refs: BTreeMap::new(),
        imports: Vec::new(),
    };

    // Pass 1: classify every mapped step's bindings (registering words) and
    // collect the extent references of the op stream.
    let mut mapped: Vec<MappedStep> = Vec::new();
    for kernel_step in launch.kernel.steps.iter() {
        if let ResolvedKernelStep::Mapped {
            iteration,
            bindings,
            ops,
        } = kernel_step
        {
            let mut classified_bindings = Vec::new();
            for (value, transport) in bindings {
                let binding = emit.classify(transport, 0)?;
                classified_bindings.push((*value, binding));
            }
            for op in ops.iter() {
                register_op_extents(&mut emit, op);
            }
            mapped.push(MappedStep {
                iteration: iteration.clone(),
                bindings: classified_bindings,
                ops: ops.iter().cloned().collect(),
            });
        }
    }

    // The Cranelift function.
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
    let table = builder.block_params(entry_block)[0];
    let scalars = builder.block_params(entry_block)[1];
    let participant = builder.block_params(entry_block)[3];

    // Import references (created once, before any block needs them).
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
        Import::Decode,
    ] {
        let signature = import.signature(call_conv);
        let sig_ref = builder.import_signature(signature);
        let data = ir::ExtFuncData {
            name: ir::ExternalName::user(ir::UserExternalNameRef::from_u32(0)),
            signature: sig_ref,
            colocated: false,
        };
        let func_ref = builder.import_function(data);
        emit.import_refs.insert(import, func_ref);
        emit.imports.push((func_ref, import));
    }

    // Status pointer (the extra buffer-table entry).
    let status_ptr = if has_status {
        let address = builder
            .ins()
            .iadd_imm(table, (storage_table.len() * 8) as i64);
        Some(
            builder
                .ins()
                .load(types::I64, MemFlags::trusted(), address, 0),
        )
    } else {
        None
    };

    // The exact retained total (never a capacity) and the participant count.
    let total = launch_total(&mut builder, &emit, scalars, &mapped)?;
    let participants_index = emit.words.len() + emit.extent_words.len();
    let participants = word_value(&mut builder, &emit, scalars, participants_index);
    let zero = builder.ins().iconst(types::I64, 0);

    // Grid-stride traversal: participant p visits p, p+P, …  (P = 1 for
    // serialized/serial maps, which is the ascending serial order).
    let header = builder.create_block();
    builder.append_block_param(header, types::I64);
    builder.ins().jump(header, &[participant.into()]);
    builder.switch_to_block(header);
    let linear = builder.block_params(header)[0];
    let in_range = builder.ins().icmp(IntCC::UnsignedLessThan, linear, total);
    let body = builder.create_block();
    builder.append_block_param(body, types::I64);
    let exit = builder.create_block();
    builder
        .ins()
        .brif(in_range, body, &[linear.into()], exit, &[]);
    builder.switch_to_block(body);
    builder.seal_block(body);

    // The point program.
    for step in &mapped {
        let coords = delinearize(&mut builder, &mut emit, scalars, &step.iteration, linear)?;
        let mut point = PointEmit {
            builder: &mut builder,
            emit: &mut emit,
            table,
            scalars,
            status_ptr,
            storage_ptrs: BTreeMap::new(),
            locals: BTreeMap::new(),
            coords,
            linear,
            participants,
            loop_vars: BTreeMap::new(),
            axis_vars: BTreeMap::new(),
        };
        point.op_stream(&step.bindings, &step.ops)?;
    }

    let _ = zero;
    let next = builder.ins().iadd(linear, participants);
    builder.ins().jump(header, &[next.into()]);
    builder.seal_block(header);
    builder.switch_to_block(exit);
    let status_zero = builder.ins().iconst(types::I32, 0);
    builder.ins().return_(&[status_zero]);
    builder.seal_block(exit);

    let descriptor = LaunchDescriptor {
        storage_table,
        has_status,
        words: emit.words.clone(),
        extent_words: emit.extent_words.clone(),
        work_items: launch.work_items.clone(),
        participants: launch.geometry.participants_per_workgroup[0].clone(),
    };
    Ok((function, std::mem::take(&mut emit.imports), descriptor))
}

fn register_op_extents(emit: &mut LaunchEmit, op: &CpuOp) {
    fn register_expr(emit: &mut LaunchEmit, expr: &ExtentExpr) {
        if let ExtentExpr::Runtime(id) = expr {
            emit.extent_index(*id);
        }
    }
    fn register_exprs(emit: &mut LaunchEmit, exprs: &[ExtentExpr]) {
        for expr in exprs {
            register_expr(emit, expr);
        }
    }
    match &op.kind {
        CpuOpKind::ExtentOf { extent } | CpuOpKind::ValidExtentOf { extent } => {
            register_expr(emit, extent)
        }
        CpuOpKind::Check {
            kind: CheckKind::IndexInBounds { extent, .. },
            ..
        }
        | CpuOpKind::Check {
            kind: CheckKind::RangeInBounds { extent, .. },
            ..
        } => register_expr(emit, extent),
        CpuOpKind::Check {
            kind: CheckKind::ProductFits { factors, .. },
            ..
        } => register_exprs(emit, factors),
        CpuOpKind::Check {
            kind: CheckKind::ExtentPositive { extent },
            ..
        } => register_expr(emit, extent),
        CpuOpKind::LoadElement { shape, .. }
        | CpuOpKind::StoreElement { shape, .. }
        | CpuOpKind::LinearElementLoop { shape, .. }
        | CpuOpKind::PackedDecode { shape, .. }
        | CpuOpKind::PackedPlaneRead { shape, .. }
        | CpuOpKind::PackedElementRead { shape, .. }
        | CpuOpKind::SerializedAtomic { shape, .. }
        | CpuOpKind::AtomicDevice { shape, .. } => register_exprs(emit, shape),
        CpuOpKind::SerialFor { length, body, .. } => {
            register_expr(emit, length);
            for nested in body {
                register_op_extents(emit, nested);
            }
        }
        CpuOpKind::Branch {
            then_ops, else_ops, ..
        } => {
            for nested in then_ops.iter().chain(else_ops) {
                register_op_extents(emit, nested);
            }
        }
        CpuOpKind::ReduceFold { outer, length, .. } => {
            register_exprs(emit, outer);
            register_expr(emit, length);
        }
        _ => {}
    }
}

/// The exact retained total of the first mapped iteration.
fn launch_total(
    builder: &mut FunctionBuilder<'_>,
    emit: &LaunchEmit,
    scalars: Value,
    mapped: &[MappedStep],
) -> Result<Value, String> {
    for step in mapped {
        return match &step.iteration.total {
            LinearTotal::Static(total) => {
                let signed = i64::try_from(*total)
                    .map_err(|_| "the static iteration total exceeds i64".to_string())?;
                Ok(builder.ins().iconst(types::I64, signed))
            }
            LinearTotal::Runtime { product, .. } => {
                runtime_scalar_expr(builder, emit, scalars, product)
            }
        };
    }
    Err("CPU launch has no mapped iteration (compiler bug)".into())
}

fn runtime_scalar_expr(
    builder: &mut FunctionBuilder<'_>,
    emit: &LaunchEmit,
    scalars: Value,
    expr: &RuntimeScalarExpr,
) -> Result<Value, String> {
    Ok(match expr {
        RuntimeScalarExpr::Const(value) => builder.ins().iconst(types::I64, *value),
        RuntimeScalarExpr::Extent(id) => {
            let word = *emit
                .extent_word_of
                .get(id)
                .ok_or_else(|| format!("runtime extent {} is not in the word table", id.0))?;
            word_value(builder, emit, scalars, word)
        }
        RuntimeScalarExpr::Value(value) => {
            return Err(format!(
                "a retained runtime total references graph value#{} (compiler bug)",
                value.0
            ));
        }
        RuntimeScalarExpr::Add(left, right) => {
            let a = runtime_scalar_expr(builder, emit, scalars, left)?;
            let b = runtime_scalar_expr(builder, emit, scalars, right)?;
            builder.ins().iadd(a, b)
        }
        RuntimeScalarExpr::Sub(left, right) => {
            let a = runtime_scalar_expr(builder, emit, scalars, left)?;
            let b = runtime_scalar_expr(builder, emit, scalars, right)?;
            builder.ins().isub(a, b)
        }
        RuntimeScalarExpr::Mul(left, right) => {
            let a = runtime_scalar_expr(builder, emit, scalars, left)?;
            let b = runtime_scalar_expr(builder, emit, scalars, right)?;
            builder.ins().imul(a, b)
        }
        RuntimeScalarExpr::Div(left, right) => {
            let numerator = runtime_scalar_expr(builder, emit, scalars, left)?;
            let denominator = runtime_scalar_expr(builder, emit, scalars, right)?;
            builder.ins().sdiv(numerator, denominator)
        }
        RuntimeScalarExpr::Rem(left, right) => {
            let numerator = runtime_scalar_expr(builder, emit, scalars, left)?;
            let denominator = runtime_scalar_expr(builder, emit, scalars, right)?;
            builder.ins().srem(numerator, denominator)
        }
    })
}

// ---------------------------------------------------------------------------
// Word / extent / value helpers
// ---------------------------------------------------------------------------

/// A stable host tag per dtype (the `seismic_round_to` convention).
fn dtype_tag(dtype: DType) -> i32 {
    i32::from(dtype_key(dtype))
}

/// A stable ordering key per dtype (map keys).
fn dtype_key(dtype: DType) -> u8 {
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
fn word_value(
    builder: &mut FunctionBuilder<'_>,
    emit: &LaunchEmit,
    scalars: Value,
    index: usize,
) -> Value {
    let _ = emit;
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

/// Decode a raw word into an f64 S-value at `dtype` (bits in the low bytes).
fn decode_word(
    builder: &mut FunctionBuilder<'_>,
    emit: &LaunchEmit,
    word: Value,
    dtype: DType,
) -> Value {
    match dtype {
        DType::F32 => {
            let bits = builder.ins().ireduce(types::I32, word);
            let bits = builder.ins().bitcast(types::F32, MemFlags::new(), bits);
            builder.ins().fpromote(types::F64, bits)
        }
        DType::I32 => {
            let v = builder.ins().ireduce(types::I32, word);
            let v = builder.ins().sextend(types::I64, v);
            builder.ins().fcvt_from_sint(types::F64, v)
        }
        DType::U32 => {
            let v = builder.ins().ireduce(types::I32, word);
            let v = builder.ins().uextend(types::I64, v);
            builder.ins().fcvt_from_uint(types::F64, v)
        }
        DType::Bool => {
            let bit = builder.ins().band_imm(word, 1);
            builder.ins().fcvt_from_uint(types::F64, bit)
        }
        DType::F16 => {
            let bits = builder.ins().ireduce(types::I32, word);
            let bits = builder.ins().band_imm(bits, 0xffff);
            let f16 = call_import(builder, emit, Import::F16Load, &[bits.into()]);
            builder.ins().fpromote(types::F64, f16)
        }
        DType::BF16 => {
            let bits = builder.ins().ireduce(types::I32, word);
            let bits = builder.ins().band_imm(bits, 0xffff);
            let shifted = builder.ins().ishl_imm(bits, 16);
            let f32 = builder.ins().bitcast(types::F32, MemFlags::new(), shifted);
            builder.ins().fpromote(types::F64, f32)
        }
    }
}

/// Encode an f64 S-value into a raw word at `dtype`.
fn encode_word(
    builder: &mut FunctionBuilder<'_>,
    emit: &LaunchEmit,
    value: Value,
    dtype: DType,
) -> Value {
    match dtype {
        DType::F32 => {
            let f32 = builder.ins().fdemote(types::F32, value);
            let bits = builder.ins().bitcast(types::I32, MemFlags::new(), f32);
            builder.ins().uextend(types::I64, bits)
        }
        DType::I32 => {
            let v = builder.ins().fcvt_to_sint(types::I64, value);
            let v = builder.ins().ireduce(types::I32, v);
            builder.ins().uextend(types::I64, v)
        }
        DType::U32 => {
            let v = builder.ins().fcvt_to_uint(types::I64, value);
            let v = builder.ins().ireduce(types::I32, v);
            builder.ins().uextend(types::I64, v)
        }
        DType::Bool => {
            let zero = builder.ins().f64const(ir::immediates::Ieee64::with_bits(0));
            let bit = builder.ins().fcmp(FloatCC::NotEqual, value, zero);
            builder.ins().uextend(types::I64, bit)
        }
        DType::F16 => {
            let f32 = builder.ins().fdemote(types::F32, value);
            let bits = call_import(builder, emit, Import::F16Store, &[f32.into()]);
            builder.ins().uextend(types::I64, bits)
        }
        DType::BF16 => {
            let f32 = builder.ins().fdemote(types::F32, value);
            let bits = builder.ins().bitcast(types::I32, MemFlags::new(), f32);
            let high = builder.ins().ushr_imm(bits, 16);
            builder.ins().uextend(types::I64, high)
        }
    }
}

/// Round one f64 S-value once at `dtype` (the registry reference model).
fn round_s(
    builder: &mut FunctionBuilder<'_>,
    emit: &LaunchEmit,
    value: Value,
    dtype: DType,
) -> Value {
    match dtype {
        DType::F32 => {
            let f32 = builder.ins().fdemote(types::F32, value);
            builder.ins().fpromote(types::F64, f32)
        }
        _ => {
            let tag = builder
                .ins()
                .iconst(types::I32, i64::from(dtype_tag(dtype)));
            call_import(builder, emit, Import::RoundTo, &[tag.into(), value.into()])
        }
    }
}

/// Call one host import and return its single result.
fn call_import(
    builder: &mut FunctionBuilder<'_>,
    emit: &LaunchEmit,
    import: Import,
    args: &[ir::Value],
) -> Value {
    let reference = emit
        .import_refs
        .get(&import)
        .copied()
        .expect("imports are created before emission");
    let call = builder.ins().call(reference, args);
    let result = builder.inst_results(call)[0];
    result
}

/// A `bool` value (I8) from an f64 S-value.
fn bool_of(builder: &mut FunctionBuilder<'_>, value: Value) -> Value {
    let zero = builder.ins().f64const(ir::immediates::Ieee64::with_bits(0));
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

/// An f64 S-value from i64 bits reinterpreted at `dtype`.
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

/// Delinearize one row-major coordinate over the iteration extents.
fn delinearize(
    builder: &mut FunctionBuilder<'_>,
    emit: &mut LaunchEmit,
    scalars: Value,
    iteration: &LinearIterationMap,
    linear: Value,
) -> Result<Vec<Value>, String> {
    let rank = iteration.extents.len();
    if rank == 0 {
        return Ok(Vec::new());
    }
    let extents: Vec<Value> = iteration
        .extents
        .iter()
        .map(|expr| extent_value(builder, emit, scalars, expr))
        .collect::<Result<_, _>>()?;
    // Strides: stride[axis] = product of the following extents.
    let mut strides = vec![builder.ins().iconst(types::I64, 1); rank];
    for axis in (0..rank - 1).rev() {
        strides[axis] = builder.ins().imul(strides[axis + 1], extents[axis + 1]);
    }
    let mut coords = Vec::with_capacity(rank);
    let mut rest = linear;
    for axis in 0..rank {
        let coord = builder.ins().udiv(rest, strides[axis]);
        let coord = if axis + 1 == rank {
            coord
        } else {
            builder.ins().urem(coord, extents[axis])
        };
        coords.push(coord);
        rest = if axis + 1 == rank {
            rest
        } else {
            builder.ins().urem(rest, strides[axis])
        };
    }
    Ok(coords)
}

/// One extent expression as an i64 value (static, solved symbol, or the
/// retained runtime-extent word).
fn extent_value(
    builder: &mut FunctionBuilder<'_>,
    emit: &LaunchEmit,
    scalars: Value,
    expr: &ExtentExpr,
) -> Result<Value, String> {
    Ok(match expr {
        ExtentExpr::Static(n) => {
            let signed = i64::try_from(*n)
                .map_err(|_| "a static extent exceeds i64 at emission".to_string())?;
            builder.ins().iconst(types::I64, signed)
        }
        ExtentExpr::Sym(sym) => constant_of_sym(sym)?
            .map(|value| builder.ins().iconst(types::I64, value))
            .ok_or_else(|| {
                "an unresolved planning symbol survived into emission (compiler bug)".to_string()
            })?,
        ExtentExpr::Runtime(id) => {
            let word = *emit
                .extent_word_of
                .get(id)
                .ok_or_else(|| format!("runtime extent {} is not in the word table", id.0))?;
            word_value(builder, emit, scalars, word)
        }
    })
}

fn constant_of_sym(sym: &Sym) -> Result<Option<i64>, String> {
    Ok(sym.as_constant())
}

// ---------------------------------------------------------------------------
// The point program: exhaustive CpuOp emission
// ---------------------------------------------------------------------------

impl<'a, 'f, 'p> PointEmit<'a, 'f, 'p> {
    /// The op stream: a planned check guards the following opcode (on
    /// failure it writes the first error into its status field and the
    /// guarded opcode is skipped).
    fn op_stream(
        &mut self,
        bindings: &[(GraphValueId, Binding)],
        ops: &[CpuOp],
    ) -> Result<(), String> {
        let mut index = 0;
        while index < ops.len() {
            let op = &ops[index];
            if let CpuOpKind::Check { kind, status } = &op.kind {
                let guarded = ops
                    .get(index + 1)
                    .ok_or("a planned check guards no opcode (compiler bug)")?;
                self.emit_guarded(bindings, kind, *status, guarded)?;
                index += 2;
            } else {
                self.op(bindings, op)?;
                index += 1;
            }
        }
        Ok(())
    }

    /// Emit `guarded` under the check predicate: on failure the status field
    /// is written and the opcode is skipped.
    fn emit_guarded(
        &mut self,
        bindings: &[(GraphValueId, Binding)],
        kind: &CheckKind,
        status: StatusFieldId,
        guarded: &CpuOp,
    ) -> Result<(), String> {
        let ok = self.check_predicate(bindings, kind)?;
        // Kernel-local results must dominate uses after the merge: give the
        // skip block one parameter per kernel-local result position.
        let kernel_results: Vec<usize> = guarded
            .results
            .iter()
            .map(|position| usize::from(*position))
            .filter(|position| {
                bindings
                    .get(*position)
                    .is_some_and(|(_, binding)| matches!(binding, Binding::Kernel))
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
            let offset = *self
                .emit
                .status_offsets
                .get(&status)
                .ok_or_else(|| format!("status field {status:?} has no planned offset"))?;
            let base = self
                .status_ptr
                .ok_or("a check opcode reached a launch without a status entry")?;
            let address = self.builder.ins().iadd_imm(base, i64::from(offset));
            let code = self.builder.ins().iconst(types::I32, 1);
            self.builder
                .ins()
                .store(MemFlags::trusted(), code, address, 0);
            // Skipped word results take a neutral value; guarded tensor
            // stores are skipped entirely (that is the unsafe access).
            for position in guarded
                .results
                .iter()
                .map(|position| usize::from(*position))
            {
                if let Some((_, Binding::Word { index, dtype })) = bindings.get(position) {
                    let neutral = self
                        .builder
                        .ins()
                        .f64const(ir::immediates::Ieee64::with_bits(0));
                    let word = encode_word(self.builder, self.emit, neutral, *dtype);
                    store_word(self.builder, self.scalars, *index, word);
                }
            }
            // Kernel-local results merge through the skip-block parameters.
            let zeros: Vec<Value> = skip_params
                .iter()
                .map(|_| {
                    self.builder
                        .ins()
                        .f64const(ir::immediates::Ieee64::with_bits(0))
                })
                .collect();
            let zero_args: Vec<_> = zeros.into_iter().map(|value| value.into()).collect();
            self.builder.ins().jump(skip_block, &zero_args);
        }
        self.builder.switch_to_block(continue_block);
        self.builder.seal_block(continue_block);
        self.op(bindings, guarded)?;
        let computed: Vec<Value> = kernel_results
            .iter()
            .map(|position| {
                self.locals.get(position).copied().unwrap_or_else(|| {
                    self.builder
                        .ins()
                        .f64const(ir::immediates::Ieee64::with_bits(0))
                })
            })
            .collect();
        let computed_args: Vec<_> = computed.into_iter().map(|value| value.into()).collect();
        self.builder.ins().jump(skip_block, &computed_args);
        self.builder.switch_to_block(skip_block);
        self.builder.seal_block(skip_block);
        for (position, param) in kernel_results.iter().zip(skip_params) {
            self.locals.insert(*position, param);
        }
        Ok(())
    }

    /// One planned runtime-check predicate.
    fn check_predicate(
        &mut self,
        bindings: &[(GraphValueId, Binding)],
        kind: &CheckKind,
    ) -> Result<Value, String> {
        let zero = self.builder.ins().iconst(types::I64, 0);
        Ok(match kind {
            CheckKind::IndexInBounds { index, extent } => {
                let (v, _) = self.fetch(bindings, usize::from(*index))?;
                let v = int_bits(self.builder, v);
                let e = self.extent(extent)?;
                let non_negative =
                    self.builder
                        .ins()
                        .icmp(IntCC::SignedGreaterThanOrEqual, v, zero);
                let below = self.builder.ins().icmp(IntCC::UnsignedLessThan, v, e);
                self.builder.ins().band(non_negative, below)
            }
            CheckKind::RangeInBounds { start, end, extent } => {
                let (s, _) = self.fetch(bindings, usize::from(*start))?;
                let (e, _) = self.fetch(bindings, usize::from(*end))?;
                let s = int_bits(self.builder, s);
                let e_v = int_bits(self.builder, e);
                let bound = self.extent(extent)?;
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
            CheckKind::DivisorNonZero { value } => {
                let (v, _) = self.fetch(bindings, usize::from(*value))?;
                let bits = int_bits(self.builder, v);
                self.builder.ins().icmp(IntCC::NotEqual, bits, zero)
            }
            CheckKind::DivisionSafe { lhs, rhs } => {
                let (l, _) = self.fetch(bindings, usize::from(*lhs))?;
                let (r, _) = self.fetch(bindings, usize::from(*rhs))?;
                let l = int_bits(self.builder, l);
                let r = int_bits(self.builder, r);
                let nonzero = self.builder.ins().icmp(IntCC::NotEqual, r, zero);
                let min = self.builder.ins().iconst(types::I64, i64::from(i32::MIN));
                let minus_one = self.builder.ins().iconst(types::I64, -1);
                let is_min = self.builder.ins().icmp(IntCC::Equal, l, min);
                let is_m1 = self.builder.ins().icmp(IntCC::Equal, r, minus_one);
                let overflow = self.builder.ins().band(is_min, is_m1);
                let one = self.builder.ins().iconst(types::I8, 1);
                let not_overflow = self.builder.ins().bxor(overflow, one);
                self.builder.ins().band(nonzero, not_overflow)
            }
            CheckKind::ShiftInRange { value } => {
                let (v, _) = self.fetch(bindings, usize::from(*value))?;
                let v = int_bits(self.builder, v);
                let low = self
                    .builder
                    .ins()
                    .icmp(IntCC::SignedGreaterThanOrEqual, v, zero);
                let limit = self.builder.ins().iconst(types::I64, 32);
                let high = self.builder.ins().icmp(IntCC::SignedLessThan, v, limit);
                self.builder.ins().band(low, high)
            }
            CheckKind::ProductFits { factors, bits } => {
                let mut acc = self.builder.ins().iconst(types::I64, 1);
                let mut ok = self.builder.ins().iconst(types::I8, 1);
                for factor in factors {
                    let f = self.extent(factor)?;
                    let product = self.builder.ins().imul(acc, f);
                    let zero_factor = self.builder.ins().icmp(IntCC::Equal, f, zero);
                    let divided = self.builder.ins().sdiv(product, f);
                    let consistent = self.builder.ins().icmp(IntCC::Equal, divided, acc);
                    let no_overflow = self.builder.ins().bor(zero_factor, consistent);
                    ok = self.builder.ins().band(ok, no_overflow);
                    acc = product;
                }
                if *bits < 64 {
                    let limit = self.builder.ins().iconst(types::I64, 1 << *bits);
                    let within = self.builder.ins().icmp(IntCC::UnsignedLessThan, acc, limit);
                    ok = self.builder.ins().band(ok, within);
                }
                ok
            }
            CheckKind::ExtentPositive { extent } => {
                let e = self.extent(extent)?;
                self.builder.ins().icmp(IntCC::UnsignedGreaterThan, e, zero)
            }
        })
    }

    /// The exhaustive opcode emission. No selected opcode is rejected.
    fn op(&mut self, bindings: &[(GraphValueId, Binding)], op: &CpuOp) -> Result<(), String> {
        let operands = &op.operands;
        let results = &op.results;
        let result_pos = |n: usize| -> Result<usize, String> {
            results
                .get(n)
                .copied()
                .map(|p| p as usize)
                .ok_or_else(|| format!("CPU opcode {:?} lacks result {n}", op.kind))
        };
        let operand_pos = |n: usize| -> Result<usize, String> {
            operands
                .get(n)
                .copied()
                .map(|p| p as usize)
                .ok_or_else(|| format!("CPU opcode {:?} lacks operand {n}", op.kind))
        };
        match &op.kind {
            CpuOpKind::Const { value, dtype } => {
                let v = match value {
                    ConstValue::Int(v) => {
                        let bits = self.builder.ins().iconst(types::I64, *v);
                        self.builder.ins().fcvt_from_sint(types::F64, bits)
                    }
                    ConstValue::FloatBits(bits) => self
                        .builder
                        .ins()
                        .f64const(ir::immediates::Ieee64::with_bits(*bits)),
                    ConstValue::Bool(v) => {
                        let c = self.builder.ins().iconst(types::I64, i64::from(*v));
                        self.builder.ins().fcvt_from_uint(types::F64, c)
                    }
                };
                self.store_result(bindings, result_pos(0)?, v, *dtype)
            }
            CpuOpKind::RuntimeExtent { extent } => {
                let word = self.extent_word(*extent)?;
                let value = self.builder.ins().fcvt_from_sint(types::F64, word);
                self.store_result(bindings, result_pos(0)?, value, DType::I32)
            }
            CpuOpKind::TuplePack => {
                let position = result_pos(0)?;
                let leaves = tuple_leaves(bindings, position)?;
                for (ordinal, leaf) in leaves.iter().enumerate() {
                    let (v, dtype) = self.fetch(bindings, operand_pos(ordinal)?)?;
                    let word = encode_word(self.builder, self.emit, v, dtype);
                    match leaf {
                        Binding::Word { index, .. } => {
                            store_word(self.builder, self.scalars, *index, word)
                        }
                        _ => return Err("a tuple leaf is not scalar-backed".into()),
                    }
                }
                Ok(())
            }
            CpuOpKind::TupleGet { index } => {
                let position = operand_pos(0)?;
                let leaves = tuple_leaves(bindings, position)?;
                match leaves.get(*index) {
                    Some(Binding::Word { index: word, dtype }) => {
                        let raw = word_value(self.builder, self.emit, self.scalars, *word);
                        let v = decode_word(self.builder, self.emit, raw, *dtype);
                        self.store_result(bindings, result_pos(0)?, v, *dtype)
                    }
                    _ => Err("a tuple.get leaf is absent".into()),
                }
            }
            CpuOpKind::RangeMake => {
                let position = result_pos(0)?;
                let leaves = tuple_leaves(bindings, position)?;
                for (ordinal, leaf) in leaves.iter().enumerate().take(2) {
                    let (v, dtype) = self.fetch(bindings, operand_pos(ordinal)?)?;
                    let word = encode_word(self.builder, self.emit, v, dtype);
                    match leaf {
                        Binding::Word { index, .. } => {
                            store_word(self.builder, self.scalars, *index, word)
                        }
                        _ => return Err("a range leaf is not scalar-backed".into()),
                    }
                }
                Ok(())
            }
            CpuOpKind::RangeStart | CpuOpKind::RangeEnd => {
                let position = operand_pos(0)?;
                let leaves = tuple_leaves(bindings, position)?;
                let leaf = if matches!(op.kind, CpuOpKind::RangeStart) {
                    0
                } else {
                    1
                };
                match leaves.get(leaf) {
                    Some(Binding::Word { index, dtype }) => {
                        let raw = word_value(self.builder, self.emit, self.scalars, *index);
                        let v = decode_word(self.builder, self.emit, raw, *dtype);
                        self.store_result(bindings, result_pos(0)?, v, *dtype)
                    }
                    _ => Err("a range leaf is absent".into()),
                }
            }
            CpuOpKind::Select { dtype } => {
                let (c, _) = self.fetch(bindings, operand_pos(0)?)?;
                let (a, _) = self.fetch(bindings, operand_pos(1)?)?;
                let (b, _) = self.fetch(bindings, operand_pos(2)?)?;
                let cond = bool_of(self.builder, c);
                let v = self.builder.ins().select(cond, a, b);
                self.store_result(bindings, result_pos(0)?, v, *dtype)
            }
            CpuOpKind::ExtentOf { extent } | CpuOpKind::ValidExtentOf { extent } => {
                let e = self.extent(extent)?;
                let v = self.builder.ins().fcvt_from_sint(types::F64, e);
                self.store_result(bindings, result_pos(0)?, v, DType::I32)
            }
            CpuOpKind::Not => {
                let (v, _) = self.fetch(bindings, operand_pos(0)?)?;
                let zero = self
                    .builder
                    .ins()
                    .f64const(ir::immediates::Ieee64::with_bits(0));
                let bit = self.builder.ins().fcmp(FloatCC::Equal, v, zero);
                let out = f64_of_bool(self.builder, bit);
                self.store_result(bindings, result_pos(0)?, out, DType::Bool)
            }
            CpuOpKind::Compare { op: cmp, left, .. } => {
                let (a, _) = self.fetch(bindings, operand_pos(0)?)?;
                let (b, _) = self.fetch(bindings, operand_pos(1)?)?;
                let bit = if left.is_float() {
                    let cc = match cmp {
                        BinaryOp::Eq => FloatCC::Equal,
                        BinaryOp::Ne => FloatCC::NotEqual,
                        BinaryOp::Lt => FloatCC::LessThan,
                        BinaryOp::Le => FloatCC::LessThanOrEqual,
                        BinaryOp::Gt => FloatCC::GreaterThan,
                        BinaryOp::Ge => FloatCC::GreaterThanOrEqual,
                        _ => return Err("a non-comparison reached compare emission".into()),
                    };
                    self.builder.ins().fcmp(cc, a, b)
                } else if *left == DType::Bool {
                    let x = bool_of(self.builder, a);
                    let y = bool_of(self.builder, b);
                    let cc = match cmp {
                        BinaryOp::Eq => IntCC::Equal,
                        BinaryOp::Ne => IntCC::NotEqual,
                        _ => return Err("an invalid boolean comparison".into()),
                    };
                    self.builder.ins().icmp(cc, x, y)
                } else {
                    let x = int_bits(self.builder, a);
                    let y = int_bits(self.builder, b);
                    let cc = match cmp {
                        BinaryOp::Eq => IntCC::Equal,
                        BinaryOp::Ne => IntCC::NotEqual,
                        BinaryOp::Lt => IntCC::SignedLessThan,
                        BinaryOp::Le => IntCC::SignedLessThanOrEqual,
                        BinaryOp::Gt => IntCC::SignedGreaterThan,
                        BinaryOp::Ge => IntCC::SignedGreaterThanOrEqual,
                        _ => return Err("a non-comparison reached compare emission".into()),
                    };
                    self.builder.ins().icmp(cc, x, y)
                };
                let out = f64_of_bool(self.builder, bit);
                self.store_result(bindings, result_pos(0)?, out, DType::Bool)
            }
            CpuOpKind::BoolBinary { op: bin } => {
                let (a, _) = self.fetch(bindings, operand_pos(0)?)?;
                let (b, _) = self.fetch(bindings, operand_pos(1)?)?;
                let x = bool_of(self.builder, a);
                let y = bool_of(self.builder, b);
                let bit = match bin {
                    BinaryOp::And => self.builder.ins().band(x, y),
                    BinaryOp::Or => self.builder.ins().bor(x, y),
                    _ => return Err("a non-boolean op reached boolean emission".into()),
                };
                let out = f64_of_bool(self.builder, bit);
                self.store_result(bindings, result_pos(0)?, out, DType::Bool)
            }
            CpuOpKind::IntUnary { op: unary, dtype } => {
                let (v, _) = self.fetch(bindings, operand_pos(0)?)?;
                let wide = int_bits(self.builder, v);
                let bits = self.builder.ins().ireduce(types::I32, wide);
                let bits = match unary {
                    UnaryOp::Neg => self.builder.ins().ineg(bits),
                    UnaryOp::BitNot => self.builder.ins().bnot(bits),
                    UnaryOp::Not => return Err("boolean not is not integer unary".into()),
                };
                let out = int_value_of(self.builder, bits, *dtype);
                self.store_result(bindings, result_pos(0)?, out, *dtype)
            }
            CpuOpKind::IntBinary { op: bin, dtype } => {
                let (a, _) = self.fetch(bindings, operand_pos(0)?)?;
                let (b, _) = self.fetch(bindings, operand_pos(1)?)?;
                let wide_a = int_bits(self.builder, a);
                let wide_b = int_bits(self.builder, b);
                let x = self.builder.ins().ireduce(types::I32, wide_a);
                let y = self.builder.ins().ireduce(types::I32, wide_b);
                let bits = self.int_binary(*bin, *dtype, x, y)?;
                let out = int_value_of(self.builder, bits, *dtype);
                self.store_result(bindings, result_pos(0)?, out, *dtype)
            }
            CpuOpKind::FloatUnary { op: unary, dtype } => {
                let (v, _) = self.fetch(bindings, operand_pos(0)?)?;
                let out = match unary {
                    UnaryOp::Neg => self.builder.ins().fneg(v),
                    _ => return Err("a non-float unary reached float emission".into()),
                };
                self.store_result(bindings, result_pos(0)?, out, *dtype)
            }
            CpuOpKind::FloatBinary { op: bin, dtype } => {
                let (a, _) = self.fetch(bindings, operand_pos(0)?)?;
                let (b, _) = self.fetch(bindings, operand_pos(1)?)?;
                let raw = match bin {
                    BinaryOp::Add => self.builder.ins().fadd(a, b),
                    BinaryOp::Sub => self.builder.ins().fsub(a, b),
                    BinaryOp::Mul => self.builder.ins().fmul(a, b),
                    BinaryOp::Div => self.builder.ins().fdiv(a, b),
                    BinaryOp::Rem => {
                        call_import(self.builder, self.emit, Import::Fmod, &[a.into(), b.into()])
                    }
                    _ => return Err("a non-float binary reached float emission".into()),
                };
                let out = round_s(self.builder, self.emit, raw, *dtype);
                self.store_result(bindings, result_pos(0)?, out, *dtype)
            }
            CpuOpKind::SeismicMath {
                op: math, dtype, ..
            } => self.seismic_math(bindings, *math, *dtype, operands, result_pos(0)?),
            CpuOpKind::Cast { source, target } => {
                let (v, _) = self.fetch(bindings, operand_pos(0)?)?;
                let out = if source.is_int()
                    && *source != DType::Bool
                    && target.is_int()
                    && *target != DType::Bool
                    || (*source == DType::Bool && target.is_int())
                {
                    // Integer↔integer casts preserve bits.
                    let wide = int_bits(self.builder, v);
                    let bits = self.builder.ins().ireduce(types::I32, wide);
                    int_value_of(self.builder, bits, *target)
                } else if *source == DType::Bool {
                    let wide = int_bits(self.builder, v);
                    let bits = self.builder.ins().ireduce(types::I32, wide);
                    int_value_of(self.builder, bits, *target)
                } else {
                    round_s(self.builder, self.emit, v, *target)
                };
                self.store_result(bindings, result_pos(0)?, out, *target)
            }
            CpuOpKind::LayoutAddress { .. } | CpuOpKind::StorageAllocation => Ok(()),
            CpuOpKind::LoadElement { dtype, shape } => {
                let view = operand_pos(0)?;
                let indices = (1..operands.len())
                    .map(|n| operand_pos(n))
                    .collect::<Result<Vec<_>, _>>()?;
                let (address, view_dtype) = self.view_address(bindings, view, &indices, shape)?;
                let v = self.load_element(address, view_dtype);
                self.store_result(bindings, result_pos(0)?, v, *dtype)
            }
            CpuOpKind::StoreElement { dtype, shape } => {
                let view = operand_pos(0)?;
                let value = operand_pos(
                    operands
                        .len()
                        .checked_sub(1)
                        .ok_or("a store has no value")?,
                )?;
                let indices = (1..operands.len() - 1)
                    .map(|n| operand_pos(n))
                    .collect::<Result<Vec<_>, _>>()?;
                let (address, view_dtype) = self.view_address(bindings, view, &indices, shape)?;
                let (v, _) = self.fetch(bindings, value)?;
                self.store_element(address, v, view_dtype);
                let _ = dtype;
                Ok(())
            }
            CpuOpKind::LinearElementLoop {
                op: loop_op,
                dtype,
                shape,
                fill_bits,
            } => {
                match loop_op {
                    seismic_compiler::terminal::LinearLoopOp::Fill => {
                        let value = self
                            .builder
                            .ins()
                            .f64const(ir::immediates::Ieee64::with_bits(*fill_bits));
                        let destination = result_pos(0)?;
                        // The op traverses its own element domain: grid-stride
                        // from this participant's first linear coordinate.
                        let total = self.shape_total(shape)?;
                        let value = value.into();
                        self.flat_loop(total, |point, flat| {
                            let (address, element_dtype) =
                                point.view_flat_address(bindings, destination, shape, flat)?;
                            point.store_element(address, value, element_dtype);
                            Ok(())
                        })?;
                        let _ = dtype;
                        Ok(())
                    }
                    seismic_compiler::terminal::LinearLoopOp::Copy => {
                        // `copy.into`: operands are [destination view, source].
                        let destination = operand_pos(0)?;
                        let source = operand_pos(1)?;
                        let total = self.shape_total(shape)?;
                        self.flat_loop(total, |point, flat| {
                            let (source_address, source_dtype) =
                                point.view_flat_address(bindings, source, shape, flat)?;
                            let v = point.load_element(source_address, source_dtype);
                            let (destination_address, destination_dtype) =
                                point.view_flat_address(bindings, destination, shape, flat)?;
                            point.store_element(destination_address, v, destination_dtype);
                            Ok(())
                        })
                    }
                    seismic_compiler::terminal::LinearLoopOp::Materialize
                    | seismic_compiler::terminal::LinearLoopOp::Clone
                    | seismic_compiler::terminal::LinearLoopOp::Load => {
                        let source = operand_pos(0)?;
                        let destination = result_pos(0)?;
                        let total = self.shape_total(shape)?;
                        self.flat_loop(total, |point, flat| {
                            let (source_address, source_dtype) =
                                point.view_flat_address(bindings, source, shape, flat)?;
                            let v = point.load_element(source_address, source_dtype);
                            let (destination_address, destination_dtype) =
                                point.view_flat_address(bindings, destination, shape, flat)?;
                            point.store_element(destination_address, v, destination_dtype);
                            Ok(())
                        })
                    }
                }
            }
            CpuOpKind::PackedDecode { repr, shape } => {
                self.packed_decode(bindings, repr, shape, operand_pos(0)?, result_pos(0)?)
            }
            CpuOpKind::PackedPlaneRead { plane, repr, shape } => {
                self.packed_plane_read(bindings, plane, repr, shape, operands, result_pos(0)?)
            }
            CpuOpKind::PackedElementRead { repr, shape } => {
                self.packed_element_read(bindings, repr, shape, operands, result_pos(0)?)
            }
            CpuOpKind::AtomicDevice { op, dtype, shape } => {
                use seismic_lang::intrinsics::AtomicOp;
                let view = operand_pos(0)?;
                let value = operand_pos(
                    operands
                        .len()
                        .checked_sub(1)
                        .ok_or("an atomic has no value")?,
                )?;
                let indices = (1..operands.len() - 1)
                    .map(|n| operand_pos(n))
                    .collect::<Result<Vec<_>, _>>()?;
                let (address, _) = self.view_address(bindings, view, &indices, shape)?;
                let (operand, _) = self.fetch(bindings, value)?;
                // Compare/exchange loop on the element's 32-bit word: read the
                // word, combine at the element dtype exactly as the serialized
                // form does, and install the result only if the word is unchanged.
                let retry = self.builder.create_block();
                let exit = self.builder.create_block();
                self.builder.ins().jump(retry, &[]);
                self.builder.switch_to_block(retry);
                let observed_bits =
                    self.builder
                        .ins()
                        .load(types::I32, MemFlags::trusted(), address, 0);
                let current = match dtype {
                    DType::F32 => {
                        let f =
                            self.builder
                                .ins()
                                .bitcast(types::F32, MemFlags::new(), observed_bits);
                        self.builder.ins().fpromote(types::F64, f)
                    }
                    DType::I32 => {
                        let wide = self.builder.ins().sextend(types::I64, observed_bits);
                        self.builder.ins().fcvt_from_sint(types::F64, wide)
                    }
                    DType::U32 => {
                        let wide = self.builder.ins().uextend(types::I64, observed_bits);
                        self.builder.ins().fcvt_from_uint(types::F64, wide)
                    }
                    other => {
                        return Err(format!(
                            "compiler bug: device atomics are planned for 32-bit elements only, not {}",
                            other.name()
                        ));
                    }
                };
                let combined = match op {
                    AtomicOp::Add => {
                        let sum = self.builder.ins().fadd(current, operand);
                        round_s(self.builder, self.emit, sum, *dtype)
                    }
                    AtomicOp::Max | AtomicOp::Min => {
                        let picked = match op {
                            AtomicOp::Max => self.builder.ins().fmax(current, operand),
                            _ => self.builder.ins().fmin(current, operand),
                        };
                        let current_nan =
                            self.builder.ins().fcmp(FloatCC::NotEqual, current, current);
                        let operand_nan =
                            self.builder.ins().fcmp(FloatCC::NotEqual, operand, operand);
                        let unless_current_nan =
                            self.builder.ins().select(current_nan, operand, picked);
                        self.builder
                            .ins()
                            .select(operand_nan, current, unless_current_nan)
                    }
                };
                let next_bits = match dtype {
                    DType::F32 => {
                        let f = self.builder.ins().fdemote(types::F32, combined);
                        self.builder.ins().bitcast(types::I32, MemFlags::new(), f)
                    }
                    DType::I32 => {
                        let wide = self.builder.ins().fcvt_to_sint(types::I64, combined);
                        self.builder.ins().ireduce(types::I32, wide)
                    }
                    _ => {
                        let wide = self.builder.ins().fcvt_to_uint(types::I64, combined);
                        self.builder.ins().ireduce(types::I32, wide)
                    }
                };
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
            CpuOpKind::SerializedAtomic { op, dtype, shape } => {
                let view = operand_pos(0)?;
                let value = operand_pos(
                    operands
                        .len()
                        .checked_sub(1)
                        .ok_or("an atomic has no value")?,
                )?;
                let indices = (1..operands.len() - 1)
                    .map(|n| operand_pos(n))
                    .collect::<Result<Vec<_>, _>>()?;
                let (address, _) = self.view_address(bindings, view, &indices, shape)?;
                let current = self.load_element(address, *dtype);
                let (operand, _) = self.fetch(bindings, value)?;
                let combined = match op {
                    // Registry `RoundsOnce`: load / add / round / store.
                    seismic_lang::intrinsics::AtomicOp::Add => {
                        let sum = self.builder.ins().fadd(current, operand);
                        round_s(self.builder, self.emit, sum, *dtype)
                    }
                    // Exact selection of one operand. Cranelift `fmax`/`fmin`
                    // propagate NaN; the reference ignores a NaN operand, so a
                    // NaN side yields the other side.
                    seismic_lang::intrinsics::AtomicOp::Max
                    | seismic_lang::intrinsics::AtomicOp::Min => {
                        let picked = match op {
                            seismic_lang::intrinsics::AtomicOp::Max => {
                                self.builder.ins().fmax(current, operand)
                            }
                            _ => self.builder.ins().fmin(current, operand),
                        };
                        let current_nan =
                            self.builder.ins().fcmp(FloatCC::NotEqual, current, current);
                        let operand_nan =
                            self.builder.ins().fcmp(FloatCC::NotEqual, operand, operand);
                        let unless_current_nan =
                            self.builder.ins().select(current_nan, operand, picked);
                        self.builder
                            .ins()
                            .select(operand_nan, current, unless_current_nan)
                    }
                };
                self.store_element(address, combined, *dtype);
                Ok(())
            }
            CpuOpKind::Check { .. } => {
                return Err(
                    "a planned check reached plain emission without a guarded opcode".into(),
                );
            }
            CpuOpKind::AxisBinder { position, axis } => {
                self.axis_vars.insert(usize::from(*position), *axis);
                Ok(())
            }
            CpuOpKind::SerialFor {
                binder,
                length,
                body,
            } => {
                let total = self.extent(length)?;
                let binder_position = usize::from(*binder);
                let zero_index = self.builder.ins().iconst(types::I64, 0);
                // Loop blocks: header(i) -> body -> step -> header.
                let header = self.builder.create_block();
                self.builder.append_block_param(header, types::I64);
                self.builder.ins().jump(header, &[zero_index.into()]);
                self.builder.switch_to_block(header);
                let index = self.builder.block_params(header)[0];
                let in_range = self
                    .builder
                    .ins()
                    .icmp(IntCC::UnsignedLessThan, index, total);
                let body_block = self.builder.create_block();
                self.builder.append_block_param(body_block, types::I64);
                let exit = self.builder.create_block();
                self.builder
                    .ins()
                    .brif(in_range, body_block, &[index.into()], exit, &[]);
                self.builder.switch_to_block(body_block);
                self.builder.seal_block(body_block);
                let current = self.builder.block_params(body_block)[0];
                self.loop_vars.insert(binder_position, current);
                for nested in body.iter() {
                    self.op(bindings, nested)?;
                }
                self.loop_vars.remove(&binder_position);
                let one_step = self.builder.ins().iconst(types::I64, 1);
                let step = self.builder.ins().iadd(current, one_step);
                self.builder.ins().jump(header, &[step.into()]);
                self.builder.seal_block(header);
                self.builder.switch_to_block(exit);
                self.builder.seal_block(exit);
                Ok(())
            }
            CpuOpKind::Branch {
                condition,
                then_ops,
                else_ops,
            } => {
                let (value, _) = self.fetch(bindings, usize::from(*condition))?;
                let bits = int_bits(self.builder, value);
                let zero_bits = self.builder.ins().iconst(types::I64, 0);
                let taken = self.builder.ins().icmp(IntCC::NotEqual, bits, zero_bits);
                let then_block = self.builder.create_block();
                let else_block = self.builder.create_block();
                let join = self.builder.create_block();
                self.builder
                    .ins()
                    .brif(taken, then_block, &[], else_block, &[]);
                self.builder.switch_to_block(then_block);
                self.builder.seal_block(then_block);
                for nested in then_ops.iter() {
                    self.op(bindings, nested)?;
                }
                self.builder.ins().jump(join, &[]);
                self.builder.switch_to_block(else_block);
                self.builder.seal_block(else_block);
                for nested in else_ops.iter() {
                    self.op(bindings, nested)?;
                }
                self.builder.ins().jump(join, &[]);
                self.builder.switch_to_block(join);
                self.builder.seal_block(join);
                Ok(())
            }
            CpuOpKind::ReduceFold { .. } => self.reduce_fold(bindings, op),
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
    ) -> Result<Value, String> {
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
            _ => return Err("a non-integer binary reached integer emission".into()),
        })
    }

    /// The versioned `seismic_math` sequence: binary64 host evaluation
    /// rounded once (identical code path to the reference interpreter).
    fn seismic_math(
        &mut self,
        bindings: &[(GraphValueId, Binding)],
        math: MathOp,
        dtype: DType,
        operands: &[u16],
        result: usize,
    ) -> Result<(), String> {
        let mut args = Vec::with_capacity(3);
        for n in 0..3 {
            match operands.get(n).copied() {
                Some(position) => args.push(self.fetch(bindings, usize::from(position))?.0),
                None => break,
            }
        }
        let builder = &mut self.builder;
        let arg = |n: usize| -> Value { args[n] };
        let out = match math {
            MathOp::Fma => {
                let (a, b, c) = (arg(0), arg(1), arg(2));
                if dtype == DType::F32 {
                    // Single-rounded f32 contraction (the registry rule).
                    let a = builder.ins().fdemote(types::F32, a);
                    let b = builder.ins().fdemote(types::F32, b);
                    let c = builder.ins().fdemote(types::F32, c);
                    let f = builder.ins().fma(a, b, c);
                    builder.ins().fpromote(types::F64, f)
                } else {
                    let f = builder.ins().fma(a, b, c);
                    round_s(builder, self.emit, f, dtype)
                }
            }
            MathOp::Exp | MathOp::ExpFast => {
                let v = call_import(builder, self.emit, Import::ExpF64, &[arg(0).into()]);
                round_s(builder, self.emit, v, dtype)
            }
            MathOp::Log => {
                let v = call_import(builder, self.emit, Import::LogF64, &[arg(0).into()]);
                round_s(builder, self.emit, v, dtype)
            }
            MathOp::Sin => {
                let v = call_import(builder, self.emit, Import::SinF64, &[arg(0).into()]);
                round_s(builder, self.emit, v, dtype)
            }
            MathOp::Cos => {
                let v = call_import(builder, self.emit, Import::CosF64, &[arg(0).into()]);
                round_s(builder, self.emit, v, dtype)
            }
            MathOp::Sqrt => {
                let v = builder.ins().sqrt(arg(0));
                round_s(builder, self.emit, v, dtype)
            }
            MathOp::Rsqrt => {
                let one = builder
                    .ins()
                    .f64const(ir::immediates::Ieee64::with_bits(1.0f64.to_bits()));
                let root = builder.ins().sqrt(arg(0));
                let v = builder.ins().fdiv(one, root);
                round_s(builder, self.emit, v, dtype)
            }
            MathOp::Abs => {
                if dtype.is_int() {
                    let wide = int_bits(builder, arg(0));
                    let bits = builder.ins().ireduce(types::I32, wide);
                    let abs = builder.ins().iabs(bits);
                    int_value_of(builder, abs, dtype)
                } else {
                    builder.ins().fabs(arg(0))
                }
            }
            MathOp::Max => {
                let v = call_import(
                    builder,
                    self.emit,
                    Import::Fmax,
                    &[arg(0).into(), arg(1).into()],
                );
                round_s(builder, self.emit, v, dtype)
            }
            MathOp::Min => {
                let v = call_import(
                    builder,
                    self.emit,
                    Import::Fmin,
                    &[arg(0).into(), arg(1).into()],
                );
                round_s(builder, self.emit, v, dtype)
            }
        };
        self.store_result(bindings, result, out, dtype)
    }

    // -- operand and result plumbing ---------------------------------------

    /// Fetch one operand as an f64 S-value: a scalar word, a kernel-local
    /// value, or the element of a tensor binding at the current point.
    fn fetch(
        &mut self,
        bindings: &[(GraphValueId, Binding)],
        position: usize,
    ) -> Result<(Value, DType), String> {
        let binding = bindings
            .get(position)
            .map(|(_, binding)| binding.clone())
            .ok_or_else(|| format!("CPU operand position {position} is not bound"))?;
        match binding {
            Binding::Word { index, dtype } => {
                let raw = word_value(self.builder, self.emit, self.scalars, index);
                Ok((decode_word(self.builder, self.emit, raw, dtype), dtype))
            }
            Binding::Kernel => {
                if let Some(axis) = self.axis_vars.get(&position) {
                    let coordinate = self
                        .coords
                        .get(*axis)
                        .copied()
                        .ok_or_else(|| format!("axis {axis} is absent at this point"))?;
                    let value = self.builder.ins().fcvt_from_sint(types::F64, coordinate);
                    return Ok((value, DType::I32));
                }
                if let Some(var) = self.loop_vars.get(&position) {
                    return Ok((*var, DType::I32));
                }
                let value =
                    self.locals.get(&position).copied().ok_or_else(|| {
                        format!("kernel-local binding {position} has no value yet")
                    })?;
                Ok((value, DType::F32))
            }
            Binding::Tensor { .. } => {
                let (address, dtype) = self.view_coords_address(bindings, position)?;
                Ok((self.load_element(address, dtype), dtype))
            }
            Binding::Tuple(_) => Err("a tuple binding is not a scalar operand".into()),
            Binding::Computed { expr, dtype } => {
                let value = self.execution_value(&expr)?;
                Ok((value, dtype))
            }
            Binding::Void => Err("a void binding is not an operand".into()),
        }
    }

    /// Fetch one operand by its graph value (slice point and range-start
    /// offsets are values, not operand positions of the consuming node).
    fn fetch_value(
        &mut self,
        bindings: &[(GraphValueId, Binding)],
        value: GraphValueId,
    ) -> Result<(Value, DType), String> {
        let position = bindings
            .iter()
            .position(|(bound, _)| *bound == value)
            .ok_or_else(|| format!("value#{value:?} is not bound in this launch"))?;
        self.fetch(bindings, position)
    }

    /// Store one result: a scalar word, a kernel-local value, or the element
    /// of an output tensor binding at the current point.
    fn store_result(
        &mut self,
        bindings: &[(GraphValueId, Binding)],
        position: usize,
        value: Value,
        dtype: DType,
    ) -> Result<(), String> {
        let binding = bindings
            .get(position)
            .map(|(_, binding)| binding.clone())
            .ok_or_else(|| {
                format!(
                    "CPU result position {position} is not bound; the family builder must bind \
                 node outputs after inputs (see AlternativeBuilder::launch)"
                )
            })?;
        match binding {
            Binding::Word {
                index,
                dtype: word_dtype,
            } => {
                let word = encode_word(self.builder, self.emit, value, word_dtype);
                store_word(self.builder, self.scalars, index, word);
                let _ = dtype;
                Ok(())
            }
            Binding::Kernel => {
                self.locals.insert(position, value);
                Ok(())
            }
            Binding::Tensor { .. } => {
                let (address, element_dtype) = self.view_coords_address(bindings, position)?;
                self.store_element(address, value, element_dtype);
                let _ = dtype;
                Ok(())
            }
            Binding::Tuple(_) => Err("a tuple result is written leaf by leaf".into()),
            Binding::Computed { .. } => Err("a computed scalar is read-only".into()),
            Binding::Void => Ok(()),
        }
    }

    /// The extent word of one runtime extent.
    fn extent_word(&mut self, id: RuntimeExtentId) -> Result<Value, String> {
        let index = *self
            .emit
            .extent_word_of
            .get(&id)
            .ok_or_else(|| format!("runtime extent {} is not in the word table", id.0))?;
        Ok(word_value(self.builder, self.emit, self.scalars, index))
    }

    /// One extent expression as an i64 value.
    fn extent(&mut self, expr: &ExtentExpr) -> Result<Value, String> {
        extent_value(self.builder, self.emit, self.scalars, expr)
    }

    /// One retained execution expression as an f64 S-value (registering its
    /// scalar/extent leaves in the word table on demand).
    fn execution_value(&mut self, expr: &ExecutionExpr) -> Result<Value, String> {
        let i64_of = |value: u64| -> Result<i64, String> {
            i64::try_from(value)
                .map_err(|_| format!("a runtime constant {value} exceeds the 64-bit signed range"))
        };
        Ok(match expr {
            ExecutionExpr::Const(value) => {
                let bits = self.builder.ins().iconst(types::I64, i64_of(*value)?);
                self.builder.ins().fcvt_from_sint(types::F64, bits)
            }
            ExecutionExpr::Extent(id) => {
                let word = self.extent_word(*id)?;
                self.builder.ins().fcvt_from_sint(types::F64, word)
            }
            ExecutionExpr::AbiScalar { path, dtype } => {
                let word = self.abi_scalar_word(path, *dtype)?;
                let raw = word_value(self.builder, self.emit, self.scalars, word);
                decode_word(self.builder, self.emit, raw, *dtype)
            }
            ExecutionExpr::ExecutorScalar(slot) => {
                let index = self.slot_word(slot)?;
                let raw = word_value(self.builder, self.emit, self.scalars, index);
                // The dtype is carried by the referencing transport.
                self.builder.ins().fcvt_from_sint(types::F64, raw)
            }
            ExecutionExpr::Add(left, right) => {
                let (a, b) = (self.execution_i64(left)?, self.execution_i64(right)?);
                let sum = self.builder.ins().iadd(a, b);
                self.builder.ins().fcvt_from_sint(types::F64, sum)
            }
            ExecutionExpr::Sub(left, right) => {
                let (a, b) = (self.execution_i64(left)?, self.execution_i64(right)?);
                let difference = self.builder.ins().isub(a, b);
                self.builder.ins().fcvt_from_sint(types::F64, difference)
            }
            ExecutionExpr::Mul(left, right) => {
                let (a, b) = (self.execution_i64(left)?, self.execution_i64(right)?);
                let product = self.builder.ins().imul(a, b);
                self.builder.ins().fcvt_from_sint(types::F64, product)
            }
            ExecutionExpr::CeilDiv(left, right) => {
                let (a, b) = (self.execution_i64(left)?, self.execution_i64(right)?);
                let quotient = self.builder.ins().sdiv(a, b);
                let remainder = self.builder.ins().srem(a, b);
                let zero = self.builder.ins().iconst(types::I64, 0);
                let exact = self.builder.ins().icmp(IntCC::Equal, remainder, zero);
                let plus = self.builder.ins().iadd_imm(quotient, 1);
                let value = self.builder.ins().select(exact, quotient, plus);
                self.builder.ins().fcvt_from_sint(types::F64, value)
            }
            ExecutionExpr::Div(left, right) => {
                let (a, b) = (self.execution_i64(left)?, self.execution_i64(right)?);
                let quotient = self.builder.ins().sdiv(a, b);
                self.builder.ins().fcvt_from_sint(types::F64, quotient)
            }
            ExecutionExpr::Rem(left, right) => {
                let (a, b) = (self.execution_i64(left)?, self.execution_i64(right)?);
                let remainder = self.builder.ins().srem(a, b);
                self.builder.ins().fcvt_from_sint(types::F64, remainder)
            }
            ExecutionExpr::Min(left, right) => {
                let (a, b) = (self.execution_i64(left)?, self.execution_i64(right)?);
                let value = self.builder.ins().smin(a, b);
                self.builder.ins().fcvt_from_sint(types::F64, value)
            }
        })
    }

    /// One retained execution expression as an i64 value.
    fn execution_i64(&mut self, expr: &ExecutionExpr) -> Result<Value, String> {
        Ok(match expr {
            ExecutionExpr::Const(value) => {
                let constant = i64::try_from(*value).map_err(|_| {
                    format!("a runtime constant {value} exceeds the 64-bit signed range")
                })?;
                self.builder.ins().iconst(types::I64, constant)
            }
            ExecutionExpr::Extent(id) => self.extent_word(*id)?,
            ExecutionExpr::AbiScalar { path, dtype } => {
                let index = self.abi_scalar_word(path, *dtype)?;
                word_value(self.builder, self.emit, self.scalars, index)
            }
            ExecutionExpr::ExecutorScalar(slot) => {
                let index = self.slot_word(slot)?;
                word_value(self.builder, self.emit, self.scalars, index)
            }
            ExecutionExpr::Add(left, right) => {
                let (a, b) = (self.execution_i64(left)?, self.execution_i64(right)?);
                self.builder.ins().iadd(a, b)
            }
            ExecutionExpr::Sub(left, right) => {
                let (a, b) = (self.execution_i64(left)?, self.execution_i64(right)?);
                self.builder.ins().isub(a, b)
            }
            ExecutionExpr::Mul(left, right) => {
                let (a, b) = (self.execution_i64(left)?, self.execution_i64(right)?);
                self.builder.ins().imul(a, b)
            }
            ExecutionExpr::CeilDiv(left, right) => {
                let (a, b) = (self.execution_i64(left)?, self.execution_i64(right)?);
                let quotient = self.builder.ins().sdiv(a, b);
                let remainder = self.builder.ins().srem(a, b);
                let zero = self.builder.ins().iconst(types::I64, 0);
                let exact = self.builder.ins().icmp(IntCC::Equal, remainder, zero);
                let plus = self.builder.ins().iadd_imm(quotient, 1);
                self.builder.ins().select(exact, quotient, plus)
            }
            ExecutionExpr::Div(left, right) => {
                let (a, b) = (self.execution_i64(left)?, self.execution_i64(right)?);
                self.builder.ins().sdiv(a, b)
            }
            ExecutionExpr::Rem(left, right) => {
                let (a, b) = (self.execution_i64(left)?, self.execution_i64(right)?);
                self.builder.ins().srem(a, b)
            }
            ExecutionExpr::Min(left, right) => {
                let (a, b) = (self.execution_i64(left)?, self.execution_i64(right)?);
                self.builder.ins().smin(a, b)
            }
        })
    }

    /// The word index of one ABI scalar leaf (registering on demand).
    fn abi_scalar_word(&mut self, path: &ValuePath, dtype: DType) -> Result<usize, String> {
        if let Some(index) = self.emit.abi_word_of.get(&(path.clone(), dtype_key(dtype))) {
            return Ok(*index);
        }
        let field = self.emit.abi_field(path, dtype, 0)?;
        let index = self.emit.words.len();
        self.emit.words.push(field);
        self.emit
            .abi_word_of
            .insert((path.clone(), dtype_key(dtype)), index);
        Ok(index)
    }

    /// The word index of one executor scalar slot (registering on demand).
    fn slot_word(&mut self, slot: &exec::ResolvedExecutorScalarId) -> Result<usize, String> {
        if let Some(index) = self.emit.word_of_slot.get(slot) {
            return Ok(*index);
        }
        let index = self.emit.words.len();
        self.emit.words.push(WordSource::Slot(*slot));
        self.emit.word_of_slot.insert(*slot, index);
        Ok(index)
    }

    /// The storage pointer of one bound storage.
    fn storage_ptr(&mut self, storage: ResolvedStorageId) -> Result<Value, String> {
        if let Some(pointer) = self.storage_ptrs.get(&storage) {
            return Ok(*pointer);
        }
        let index = self
            .emit
            .storage_slots
            .get(&storage)
            .copied()
            .ok_or_else(|| format!("storage#{} is not in the launch buffer table", storage.0))?;
        let address = self.builder.ins().iadd_imm(self.table, (index * 8) as i64);
        let pointer = self
            .builder
            .ins()
            .load(types::I64, MemFlags::trusted(), address, 0);
        self.storage_ptrs.insert(storage, pointer);
        Ok(pointer)
    }

    /// The resolved dense layout of one storage (shape and dtype).
    fn dense_layout(&self, storage: ResolvedStorageId) -> Result<(&[u64], DType), String> {
        let resolved = self
            .emit
            .plan
            .storage
            .get(storage)
            .ok_or_else(|| format!("storage#{} has no resolved layout", storage.0))?;
        match &resolved.layout {
            crate::physical::CpuResolvedLayout::Dense { dtype, shape, .. } => {
                Ok((shape.as_slice(), *dtype))
            }
            _ => Err(format!("storage#{} is not a dense plane", storage.0)),
        }
    }

    /// Base offset and per-axis stride application of one view transform.
    /// `coords` holds one i64 coordinate per view axis.
    fn transform_offset(
        &mut self,
        bindings: &[(GraphValueId, Binding)],
        transform: &ViewTransform,
        storage_strides: &[u64],
        shape: &[ExtentExpr],
        coords: &[Value],
    ) -> Result<Value, String> {
        let mut offset = self.builder.ins().iconst(types::I64, 0);
        let fold = |point: &mut Self, offset: &mut Value, coord: Value, stride: u64| {
            let scaled = point.builder.ins().imul_imm(coord, stride as i64);
            *offset = point.builder.ins().iadd(*offset, scaled);
        };
        match transform {
            ViewTransform::Reshape { .. } => {
                // Row-major flat on both sides of the reshape.
                for (axis, coord) in coords.iter().enumerate() {
                    let extent = self.extent(
                        shape
                            .get(axis)
                            .ok_or_else(|| format!("a reshape view lacks axis {axis}"))?,
                    )?;
                    let scaled = self.builder.ins().imul(offset, extent);
                    offset = self.builder.ins().iadd(scaled, *coord);
                }
            }
            ViewTransform::Transpose { permutation } => {
                for (view_axis, coord) in coords.iter().enumerate() {
                    let storage_axis = *permutation
                        .get(view_axis)
                        .ok_or_else(|| format!("a transpose permutation lacks axis {view_axis}"))?
                        as usize;
                    let stride = storage_strides.get(storage_axis).copied().ok_or_else(|| {
                        format!("a transpose names absent storage axis {storage_axis}")
                    })?;
                    fold(self, &mut offset, *coord, stride);
                }
            }
            ViewTransform::Slice { axes } => {
                let mut view_axis = 0usize;
                for (storage_axis, axis) in axes.iter().enumerate() {
                    let stride = storage_strides.get(storage_axis).copied().ok_or_else(|| {
                        format!("a slice names absent storage axis {storage_axis}")
                    })?;
                    match axis {
                        SliceAxis::Full => {
                            let coord = *coords.get(view_axis).ok_or_else(|| {
                                format!("a slice view lacks view axis {view_axis}")
                            })?;
                            fold(self, &mut offset, coord, stride);
                            view_axis += 1;
                        }
                        SliceAxis::Point(point) => {
                            let (index, _) = self.fetch_value(bindings, *point)?;
                            let index = int_bits(self.builder, index);
                            fold(self, &mut offset, index, stride);
                        }
                        SliceAxis::Range { start, .. } => {
                            if let Some(start) = start {
                                let (index, _) = self.fetch_value(bindings, *start)?;
                                let index = int_bits(self.builder, index);
                                fold(self, &mut offset, index, stride);
                            }
                            let coord = *coords.get(view_axis).ok_or_else(|| {
                                format!("a slice view lacks view axis {view_axis}")
                            })?;
                            fold(self, &mut offset, coord, stride);
                            view_axis += 1;
                        }
                    }
                }
            }
            ViewTransform::Identity => {
                for (axis, coord) in coords.iter().enumerate() {
                    let stride = storage_strides
                        .get(axis)
                        .copied()
                        .ok_or_else(|| format!("an identity view names absent axis {axis}"))?;
                    fold(self, &mut offset, *coord, stride);
                }
            }
        }
        Ok(offset)
    }

    /// Element count of one shape as a runtime value.
    fn shape_total(&mut self, shape: &[ExtentExpr]) -> Result<Value, String> {
        let mut total = self.builder.ins().iconst(types::I64, 1);
        for axis in shape {
            let extent = self.extent(axis)?;
            total = self.builder.ins().imul(total, extent);
        }
        Ok(total)
    }

    /// Grid-stride traversal of [0, total): this participant starts at its
    /// current linear coordinate and advances by the participant count.
    fn flat_loop(
        &mut self,
        total: Value,
        mut body: impl FnMut(&mut Self, Value) -> Result<(), String>,
    ) -> Result<(), String> {
        let header = self.builder.create_block();
        self.builder.append_block_param(header, types::I64);
        let start = self.linear;
        self.builder.ins().jump(header, &[start.into()]);
        self.builder.switch_to_block(header);
        let flat = self.builder.block_params(header)[0];
        let in_range = self
            .builder
            .ins()
            .icmp(IntCC::UnsignedLessThan, flat, total);
        let loop_body = self.builder.create_block();
        self.builder.append_block_param(loop_body, types::I64);
        let exit = self.builder.create_block();
        self.builder
            .ins()
            .brif(in_range, loop_body, &[flat.into()], exit, &[]);
        self.builder.switch_to_block(loop_body);
        self.builder.seal_block(loop_body);
        let current = self.builder.block_params(loop_body)[0];
        body(self, current)?;
        let step = self.builder.ins().iadd(current, self.participants);
        self.builder.ins().jump(header, &[step.into()]);
        self.builder.seal_block(header);
        self.builder.switch_to_block(exit);
        self.builder.seal_block(exit);
        Ok(())
    }

    /// Storage strides and dtype of one bound view's backing storage.
    fn view_layout(
        &mut self,
        bindings: &[(GraphValueId, Binding)],
        position: usize,
    ) -> Result<(ResolvedStorageId, ViewTransform, Vec<u64>, DType), String> {
        let (storage, transform) = match bindings.get(position) {
            Some((_, Binding::Tensor { storage, transform })) => (*storage, transform.clone()),
            _ => return Err(format!("binding position {position} is not a tensor view")),
        };
        let (storage_shape, dtype) = self.dense_layout(storage)?;
        let mut storage_strides = vec![1u64; storage_shape.len()];
        for axis in (0..storage_shape.len().saturating_sub(1)).rev() {
            storage_strides[axis] = storage_strides[axis + 1] * storage_shape[axis + 1];
        }
        Ok((storage, transform, storage_strides, dtype))
    }

    /// The byte address of one view element at explicit index bindings.
    fn view_address(
        &mut self,
        bindings: &[(GraphValueId, Binding)],
        position: usize,
        indices: &[usize],
        shape: &[ExtentExpr],
    ) -> Result<(Value, DType), String> {
        let (storage, transform, storage_strides, dtype) = self.view_layout(bindings, position)?;
        let mut coords = Vec::with_capacity(indices.len());
        for index_position in indices {
            let (index, _) = self.fetch(bindings, *index_position)?;
            coords.push(int_bits(self.builder, index));
        }
        let offset =
            self.transform_offset(bindings, &transform, &storage_strides, shape, &coords)?;
        let base = self.storage_ptr(storage)?;
        let bytes = u64::from(dtype.bytes());
        let byte_offset = if bytes == 1 {
            offset
        } else {
            self.builder.ins().imul_imm(offset, bytes as i64)
        };
        Ok((self.builder.ins().iadd(base, byte_offset), dtype))
    }

    /// The byte address of one view element at the current launch point's
    /// coordinates (a tensor operand consumed inside a tensor domain).
    fn view_coords_address(
        &mut self,
        bindings: &[(GraphValueId, Binding)],
        position: usize,
    ) -> Result<(Value, DType), String> {
        let coords: Vec<Value> = self.coords.clone();
        let (storage, transform, storage_strides, dtype) = self.view_layout(bindings, position)?;
        // A reshape consumed at the launch point addresses row-major flat:
        // the launch coordinate is the flat coordinate on both sides.
        let offset = if matches!(transform, ViewTransform::Reshape { .. }) {
            self.linear
        } else {
            let shape: Vec<ExtentExpr> = Vec::new();
            self.transform_offset(bindings, &transform, &storage_strides, &shape, &coords)?
        };
        let base = self.storage_ptr(storage)?;
        let bytes = u64::from(dtype.bytes());
        let byte_offset = if bytes == 1 {
            offset
        } else {
            self.builder.ins().imul_imm(offset, bytes as i64)
        };
        Ok((self.builder.ins().iadd(base, byte_offset), dtype))
    }

    /// The byte address of one view element at a flat row-major coordinate
    /// over the view's own domain (`shape`).
    fn view_flat_address(
        &mut self,
        bindings: &[(GraphValueId, Binding)],
        position: usize,
        shape: &[ExtentExpr],
        flat: Value,
    ) -> Result<(Value, DType), String> {
        let (storage, transform, storage_strides, dtype) = self.view_layout(bindings, position)?;
        // Delinearize the flat coordinate over the view shape (last axis
        // fastest): coordinate[axis] = (flat / stride) % extent.
        let mut view_strides = vec![1u64; shape.len().max(1)];
        let mut coords = Vec::with_capacity(shape.len());
        let mut rest = flat;
        for axis in (0..shape.len()).rev() {
            let extent = self.extent(&shape[axis])?;
            let coord = if axis + 1 == shape.len() {
                rest
            } else {
                let coord = self.builder.ins().urem(rest, extent);
                rest = self.builder.ins().udiv(rest, extent);
                coord
            };
            coords.push(coord);
        }
        coords.reverse();
        let _ = &mut view_strides;
        let offset =
            self.transform_offset(bindings, &transform, &storage_strides, shape, &coords)?;
        let base = self.storage_ptr(storage)?;
        let bytes = u64::from(dtype.bytes());
        let byte_offset = if bytes == 1 {
            offset
        } else {
            self.builder.ins().imul_imm(offset, bytes as i64)
        };
        Ok((self.builder.ins().iadd(base, byte_offset), dtype))
    }

    /// Load one typed element (converted to an f64 S-value).
    fn load_element(&mut self, address: Value, dtype: DType) -> Value {
        let builder = &mut self.builder;
        match dtype {
            DType::F32 => {
                let v = builder
                    .ins()
                    .load(types::F32, MemFlags::trusted(), address, 0);
                builder.ins().fpromote(types::F64, v)
            }
            DType::I32 => {
                let v = builder
                    .ins()
                    .load(types::I32, MemFlags::trusted(), address, 0);
                let v = builder.ins().sextend(types::I64, v);
                builder.ins().fcvt_from_sint(types::F64, v)
            }
            DType::U32 => {
                let v = builder
                    .ins()
                    .load(types::I32, MemFlags::trusted(), address, 0);
                let v = builder.ins().uextend(types::I64, v);
                builder.ins().fcvt_from_uint(types::F64, v)
            }
            DType::Bool => {
                let v = builder
                    .ins()
                    .load(types::I8, MemFlags::trusted(), address, 0);
                let zero = builder.ins().iconst(types::I8, 0);
                let bit = builder.ins().icmp(IntCC::NotEqual, v, zero);
                f64_of_bool(builder, bit)
            }
            DType::F16 => {
                let bits = builder
                    .ins()
                    .load(types::I16, MemFlags::trusted(), address, 0);
                let bits = builder.ins().uextend(types::I32, bits);
                let v = call_import(builder, self.emit, Import::F16Load, &[bits.into()]);
                builder.ins().fpromote(types::F64, v)
            }
            DType::BF16 => {
                let bits = builder
                    .ins()
                    .load(types::I16, MemFlags::trusted(), address, 0);
                let bits = builder.ins().uextend(types::I32, bits);
                let shifted = builder.ins().ishl_imm(bits, 16);
                let v = builder.ins().bitcast(types::F32, MemFlags::new(), shifted);
                builder.ins().fpromote(types::F64, v)
            }
        }
    }

    /// Store one typed element (from an f64 S-value, with the dtype's
    /// encoding; the value is already dtype-rounded).
    fn store_element(&mut self, address: Value, value: Value, dtype: DType) {
        let builder = &mut self.builder;
        match dtype {
            DType::F32 => {
                let v = builder.ins().fdemote(types::F32, value);
                builder.ins().store(MemFlags::trusted(), v, address, 0);
            }
            DType::I32 => {
                let v = builder.ins().fcvt_to_sint(types::I64, value);
                let v = builder.ins().ireduce(types::I32, v);
                builder.ins().store(MemFlags::trusted(), v, address, 0);
            }
            DType::U32 => {
                let v = builder.ins().fcvt_to_uint(types::I64, value);
                let v = builder.ins().ireduce(types::I32, v);
                builder.ins().store(MemFlags::trusted(), v, address, 0);
            }
            DType::Bool => {
                let bit = bool_of(builder, value);
                let v = bit;
                builder.ins().store(MemFlags::trusted(), v, address, 0);
            }
            DType::F16 => {
                let f = builder.ins().fdemote(types::F32, value);
                let bits = call_import(builder, self.emit, Import::F16Store, &[f.into()]);
                let bits = builder.ins().ireduce(types::I16, bits);
                builder.ins().store(MemFlags::trusted(), bits, address, 0);
            }
            DType::BF16 => {
                let f = builder.ins().fdemote(types::F32, value);
                let bits = builder.ins().bitcast(types::I32, MemFlags::new(), f);
                let high = builder.ins().ushr_imm(bits, 16);
                let high = builder.ins().ireduce(types::I16, high);
                builder.ins().store(MemFlags::trusted(), high, address, 0);
            }
        }
    }

    // -- packed representations -------------------------------------------

    /// Packed decode through the intrinsic registry: dense f32 elements
    /// decoded from the representation planes.
    fn packed_decode(
        &mut self,
        bindings: &[(GraphValueId, Binding)],
        repr: &str,
        _shape: &[ExtentExpr],
        source: usize,
        destination: usize,
    ) -> Result<(), String> {
        let storage = match bindings.get(source) {
            Some((_, Binding::Tensor { storage, .. })) => *storage,
            _ => return Err("a packed decode source is not a tensor binding".into()),
        };
        let (plane_offsets, plane_count, ordinal) = self.packed_planes(storage, repr)?;
        // A stack array of the plane pointers (planned, minimal, and the
        // only scratch the decode sequence needs).
        let slot = self.builder.create_sized_stack_slot(StackSlotData::new(
            ir::StackSlotKind::ExplicitSlot,
            8 * plane_count as u32,
            3,
        ));
        let array = self.builder.ins().stack_addr(types::I64, slot, 0);
        let base = self.storage_ptr(storage)?;
        for (index, offset) in plane_offsets.iter().enumerate() {
            let pointer = self.builder.ins().iadd_imm(base, *offset as i64);
            let address = self.builder.ins().iadd_imm(array, (index * 8) as i64);
            self.builder
                .ins()
                .store(MemFlags::trusted(), pointer, address, 0);
        }
        let ordinal_value = self.builder.ins().iconst(types::I32, i64::from(ordinal));
        // The flat row-major coordinate of the current point: the loop's
        // linear coordinate is exactly that over the element domain.
        let decoded = call_import(
            self.builder,
            self.emit,
            Import::Decode,
            &[ordinal_value.into(), array.into(), self.linear.into()],
        );
        let value = self.builder.ins().fpromote(types::F64, decoded);
        self.store_result(bindings, destination, value, DType::F32)
    }

    /// Raw readable representation-plane element.
    fn packed_plane_read(
        &mut self,
        bindings: &[(GraphValueId, Binding)],
        plane: &seismic_lang::intrinsics::PlaneField,
        repr: &str,
        shape: &[ExtentExpr],
        operands: &[u16],
        result: usize,
    ) -> Result<(), String> {
        let storage = match bindings.get(operands[0] as usize) {
            Some((_, Binding::Tensor { storage, .. })) => *storage,
            _ => return Err("a packed read source is not a tensor binding".into()),
        };
        let (plane_offsets, _, _) = self.packed_planes(storage, repr)?;
        let plane_name = plane_field_name(plane);
        let (plane_index, plane_dtype) = {
            let resolved = self
                .emit
                .plan
                .storage
                .get(storage)
                .ok_or_else(|| format!("storage#{} has no resolved layout", storage.0))?;
            match &resolved.layout {
                crate::physical::CpuResolvedLayout::Packed { planes, repr, .. } => {
                    let representation = repr::lookup(repr)
                        .ok_or_else(|| format!("unknown representation `{repr}`"))?;
                    let index = representation
                        .plane_index(&plane_name)
                        .ok_or_else(|| format!("`{repr}` has no plane `{plane_name}`"))?;
                    (index, planes[index].dtype)
                }
                _ => return Err("a packed read source is not packed".into()),
            }
        };
        // The raw accessor entry: the index operands fold over the view
        // shape (a flat entry index for the usual one-dimensional accessor).
        let mut entry = self.builder.ins().iconst(types::I64, 0);
        for (axis, position) in operands.iter().skip(1).enumerate() {
            let (index, _) = self.fetch(bindings, *position as usize)?;
            let index = int_bits(self.builder, index);
            let extent =
                self.extent(shape.get(axis).ok_or_else(|| {
                    format!("a packed read index exceeds the view rank ({axis})")
                })?)?;
            let scaled = self.builder.ins().imul(entry, extent);
            entry = self.builder.ins().iadd(scaled, index);
        }
        let base = self.storage_ptr(storage)?;
        let plane_base = self
            .builder
            .ins()
            .iadd_imm(base, plane_offsets[plane_index] as i64);
        let bytes = u64::from(plane_dtype.bytes());
        let byte_offset = if bytes == 1 {
            entry
        } else {
            self.builder.ins().imul_imm(entry, bytes as i64)
        };
        let address = self.builder.ins().iadd(plane_base, byte_offset);
        let value = self.load_element(address, plane_dtype);
        self.store_result(bindings, result, value, plane_dtype)
    }

    /// One decoded element of a packed view: the flat entry folds over the
    /// view shape, then the registry decode sequence reads it through the
    /// representation planes.
    fn packed_element_read(
        &mut self,
        bindings: &[(GraphValueId, Binding)],
        repr: &str,
        shape: &[ExtentExpr],
        operands: &[u16],
        result: usize,
    ) -> Result<(), String> {
        let storage = match bindings.get(operands[0] as usize) {
            Some((_, Binding::Tensor { storage, .. })) => *storage,
            _ => return Err("a packed read source is not a tensor binding".into()),
        };
        let (plane_offsets, plane_count, ordinal) = self.packed_planes(storage, repr)?;
        let mut entry = self.builder.ins().iconst(types::I64, 0);
        for (axis, position) in operands.iter().skip(1).enumerate() {
            let (index, _) = self.fetch(bindings, *position as usize)?;
            let index = int_bits(self.builder, index);
            let extent =
                self.extent(shape.get(axis).ok_or_else(|| {
                    format!("a packed read index exceeds the view rank ({axis})")
                })?)?;
            let scaled = self.builder.ins().imul(entry, extent);
            entry = self.builder.ins().iadd(scaled, index);
        }
        let slot = self.builder.create_sized_stack_slot(StackSlotData::new(
            ir::StackSlotKind::ExplicitSlot,
            8 * plane_count as u32,
            3,
        ));
        let array = self.builder.ins().stack_addr(types::I64, slot, 0);
        let base = self.storage_ptr(storage)?;
        for (index, offset) in plane_offsets.iter().enumerate() {
            let pointer = self.builder.ins().iadd_imm(base, *offset as i64);
            let address = self.builder.ins().iadd_imm(array, (index * 8) as i64);
            self.builder
                .ins()
                .store(MemFlags::trusted(), pointer, address, 0);
        }
        let ordinal_value = self.builder.ins().iconst(types::I32, i64::from(ordinal));
        let decoded = call_import(
            self.builder,
            self.emit,
            Import::Decode,
            &[ordinal_value.into(), array.into(), entry.into()],
        );
        let value = self.builder.ins().fpromote(types::F64, decoded);
        self.store_result(bindings, result, value, DType::F32)
    }

    /// The plane byte offsets (sequential, each aligned to its dtype), the
    /// plane count, and the registry ordinal of one representation.
    fn packed_planes(
        &mut self,
        storage: ResolvedStorageId,
        repr: &str,
    ) -> Result<(Vec<u64>, usize, i32), String> {
        let resolved = self
            .emit
            .plan
            .storage
            .get(storage)
            .ok_or_else(|| format!("storage#{} has no resolved layout", storage.0))?;
        let planes = match &resolved.layout {
            crate::physical::CpuResolvedLayout::Packed { planes, .. } => planes,
            _ => return Err(format!("storage#{} is not packed", storage.0)),
        };
        let ordinal = repr::REPRS
            .iter()
            .position(|candidate| candidate.name == repr)
            .ok_or_else(|| format!("unknown representation `{repr}`"))?;
        let mut offsets = Vec::with_capacity(planes.len());
        let mut cursor = 0u64;
        for plane in planes {
            let alignment = u64::from(plane.dtype.bytes()).max(1);
            cursor = cursor.div_ceil(alignment) * alignment;
            offsets.push(cursor);
            cursor += plane.bytes;
        }
        Ok((offsets, planes.len(), ordinal as i32))
    }

    // -- the ordered universal reduction -----------------------------------

    /// Ascending serial fold with registry accumulator/identity/tie
    /// semantics over the reduced axis, at the current outer point.
    fn reduce_fold(
        &mut self,
        bindings: &[(GraphValueId, Binding)],
        op: &CpuOp,
    ) -> Result<(), String> {
        let CpuOpKind::ReduceFold {
            op: reduce,
            input: _input,
            accumulator,
            axis,
            outer,
            length,
            nonempty_precondition,
            result_dtype,
        } = &op.kind
        else {
            unreachable!("the opcode was matched as a reduction")
        };
        let operand_storage = match bindings.get(usize::from(op.operands[0])) {
            Some((_, Binding::Tensor { storage, .. })) => *storage,
            _ => return Err("a reduction operand is not a tensor binding".into()),
        };
        let (shape, element_dtype) = self.dense_layout(operand_storage)?;
        let shape: Vec<u64> = shape.to_vec();
        let element_dtype = element_dtype;
        if shape.len() != outer.len() + 1 {
            return Err(format!(
                "the reduction operand has rank {} but {} outer axes and one reduced axis",
                shape.len(),
                outer.len()
            ));
        }
        // Nonempty precondition (`argmax`): retained and evaluated; a zero
        // length reports the first error and skips the fold.
        if *nonempty_precondition {
            let len = self.extent(length)?;
            let zero = self.builder.ins().iconst(types::I64, 0);
            let positive = self
                .builder
                .ins()
                .icmp(IntCC::UnsignedGreaterThan, len, zero);
            let cont = self.builder.create_block();
            let failure = self.builder.create_block();
            self.builder.ins().brif(positive, cont, &[], failure, &[]);
            self.builder.switch_to_block(failure);
            self.builder.seal_block(failure);
            let code = self.builder.ins().iconst(types::I32, 1);
            self.builder.ins().return_(&[code]);
            self.builder.switch_to_block(cont);
            self.builder.seal_block(cont);
        }
        let len = self.extent(length)?;
        // Outer coordinate offset (all axes but the reduced one).
        let mut strides = vec![1u64; shape.len()];
        for a in (0..shape.len().saturating_sub(1)).rev() {
            strides[a] = strides[a + 1] * shape[a + 1];
        }
        let mut base_offset = self.builder.ins().iconst(types::I64, 0);
        let mut outer_axis = 0usize;
        for a in 0..shape.len() {
            if a == *axis {
                continue;
            }
            let coordinate = self
                .coords
                .get(outer_axis)
                .copied()
                .ok_or("the launch point lacks an outer coordinate")?;
            outer_axis += 1;
            let product = self.builder.ins().imul_imm(coordinate, strides[a] as i64);
            base_offset = self.builder.ins().iadd(base_offset, product);
        }
        let base = self.storage_ptr(operand_storage)?;
        let axis_stride = strides[*axis] as i64;
        let bytes = u64::from(element_dtype.bytes());
        let element_address = |point: &mut Self, c: Value| -> Value {
            let axis_offset = point.builder.ins().imul_imm(c, axis_stride * bytes as i64);
            let offset = point.builder.ins().iadd(base_offset, axis_offset);
            point.builder.ins().iadd(base, offset)
        };
        use seismic_lang::intrinsics::ReduceOp as R;
        let is_int = element_dtype.is_int();
        // The fold: ascending serial over the reduced axis, with the
        // registry identity and per-step rounding.
        let fhead = self.builder.create_block();
        let fbody = self.builder.create_block();
        let fexit = self.builder.create_block();
        let zero = self.builder.ins().iconst(types::I64, 0);
        let one = self.builder.ins().iconst(types::I64, 1);
        let len_zero = self
            .builder
            .ins()
            .icmp(IntCC::UnsignedGreaterThan, len, zero);
        let _ = len_zero;
        // Zero identities start at coordinate 0; first-element identities
        // start from coordinate 0's element and fold from 1.
        let start_c;
        let mut header_args: Vec<ir::Value> = Vec::new();
        let mut acc_dtypes: Vec<ir::Type> = Vec::new();
        match reduce {
            R::Sum if is_int => {
                // Integer sum retains the dtype and wraps.
                start_c = zero;
                acc_dtypes.push(types::I32);
                header_args.push(self.builder.ins().iconst(types::I32, 0).into());
            }
            R::Sum => {
                start_c = zero;
                acc_dtypes.push(types::F64);
                let init = self
                    .builder
                    .ins()
                    .f64const(ir::immediates::Ieee64::with_bits(0));
                header_args.push(init.into());
            }
            R::Max | R::Min => {
                start_c = one;
                let first = element_address(self, zero);
                if is_int {
                    acc_dtypes.push(types::I32);
                    let raw = self
                        .builder
                        .ins()
                        .load(types::I32, MemFlags::trusted(), first, 0);
                    header_args.push(raw.into());
                } else {
                    acc_dtypes.push(types::F64);
                    header_args.push(self.load_element(first, element_dtype).into());
                }
            }
            R::Argmax => {
                // (best value, best coordinate) — ascending order keeps the
                // smaller coordinate on ties.
                start_c = one;
                acc_dtypes.push(types::F64);
                acc_dtypes.push(types::I64);
                let first = element_address(self, zero);
                let value = self.load_element(first, element_dtype);
                header_args.push(value.into());
                header_args.push(zero.into());
            }
        }
        for ty in &acc_dtypes {
            self.builder.append_block_param(fhead, *ty);
            self.builder.append_block_param(fbody, *ty);
            self.builder.append_block_param(fexit, *ty);
        }
        self.builder.append_block_param(fhead, types::I64);
        self.builder.append_block_param(fbody, types::I64);
        // The coordinate is the final block parameter.
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
        let final_accs: Vec<Value> = self.builder.block_params(fexit)[..acc_dtypes.len()].to_vec();
        let _ = &final_accs;
        self.builder.switch_to_block(fbody);
        let coordinate = self
            .builder
            .block_params(fbody)
            .last()
            .copied()
            .expect("the fold body has a coordinate");
        let accs: Vec<Value> = self.builder.block_params(fbody)[..acc_dtypes.len()].to_vec();
        let address = element_address(self, coordinate);
        let mut next_accs: Vec<ir::Value> = Vec::with_capacity(accs.len());
        match reduce {
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
                let rounded = round_s(self.builder, self.emit, sum, *accumulator);
                next_accs.push(rounded.into());
            }
            R::Max | R::Min => {
                if is_int {
                    let raw = self
                        .builder
                        .ins()
                        .load(types::I32, MemFlags::trusted(), address, 0);
                    let combined = if *reduce == R::Max {
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
                    let import = if *reduce == R::Max {
                        Import::Fmax
                    } else {
                        Import::Fmin
                    };
                    let combined =
                        call_import(self.builder, self.emit, import, &[accs[0].into(), v.into()]);
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
        let mut jump_args: Vec<ir::BlockArg> = next_accs.iter().copied().map(Into::into).collect();
        jump_args.push(next_coordinate.into());
        self.builder.ins().jump(fhead, &jump_args);
        self.builder.seal_block(fbody);
        self.builder.seal_block(fhead);
        // Finalize in the exit block.
        self.builder.switch_to_block(fexit);
        let final_accs: Vec<Value> = self.builder.block_params(fexit)[..acc_dtypes.len()].to_vec();
        let result_position = usize::from(*op.results.first().ok_or("a reduction has no result")?);
        let s_value = match reduce {
            R::Sum if is_int => int_value_of(self.builder, final_accs[0], element_dtype),
            R::Sum => round_s(self.builder, self.emit, final_accs[0], *result_dtype),
            R::Max | R::Min if is_int => int_value_of(self.builder, final_accs[0], element_dtype),
            R::Max | R::Min => round_s(self.builder, self.emit, final_accs[0], *result_dtype),
            R::Argmax => self.builder.ins().fcvt_from_sint(types::F64, final_accs[1]),
        };
        self.store_result(bindings, result_position, s_value, *result_dtype)
    }
}

/// The tuple leaves of one binding position.
fn tuple_leaves(
    bindings: &[(GraphValueId, Binding)],
    position: usize,
) -> Result<Vec<Binding>, String> {
    match bindings.get(position) {
        Some((_, Binding::Tuple(leaves))) => Ok(leaves.clone()),
        _ => Err(format!("binding position {position} is not a tuple")),
    }
}

/// The registry plane name of one plane field.
fn plane_field_name(plane: &seismic_lang::intrinsics::PlaneField) -> String {
    plane.name().to_string()
}
