//! Mechanical MSL encoding of sealed Metal launches.
//!
//! `encode` is the total transform the pipeline `Backend` trait names: it
//! takes one sealed launch and returns its encoded form. The frozen trait
//! signature is launch-local, while residence layouts and placements live
//! in the sealed plan, so the exhaustive MSL emission itself is
//! `render(launch, plan)`, invoked once per launch by `native::assemble`
//! (which holds the plan); it lives here because it is encoding: a total,
//! exhaustive match over `KernelOp<MetalIntrinsic>` with named arms only.
//! An impossible structure contradicts a construction invariant of the
//! seal and is returned as a `CompilerDefect` — never a repair.
//!
//! The renderer makes no allocation, geometry, synchronization, algorithm,
//! or precision decision. Iteration comes from the launch's iteration map
//! (the serialized single-participant loop, one-pass, grid stride with its
//! loop-bound tail, or the dynamic-pull claim loop `while ((lin =
//! atomic_fetch_add(counter, 1)) < total)`); addresses come from the
//! launch's route/view transforms applied to each place's view coordinates
//! (the binding's `ty` carries the view-side extents); guards are the
//! planned `Check` discharges; storage comes from the launch's direct
//! bindings and its `storage_facts` (Global pointers, workgroup and
//! participant threadgroup/thread arrays sized from the facts). Native
//! temporaries are introduced freely; no semantic array absent from the
//! sealed block is ever created. `Fill` is host-side and never reaches this
//! module.

use crate::intrinsics::{MatrixOperand, MetalDialect, MetalIntrinsic, PhysicalMetalLayout};
use seismic_lang::{
    intrinsics::{AtomicOp, MathOp, PlaneField, ReduceIdentity, ReduceOp, ReduceSchema},
    repr::{self, PlaneEncoding},
    syntax::ast::{BinaryOp, UnaryOp},
    types::{DType, ExtentExpr, NonEmpty, RuntimeExtentId, TensorType},
};
use seismic_realization::dispatch::Traversal;
use seismic_realization::failure::{CompilerDefect, Package};
use seismic_realization::ids::{
    CanonicalLeafId, KernelInputId, KernelLocalId, KernelSsaId, LaunchIx, StatusFieldTemplateId,
    StorageIx,
};
use seismic_realization::kernel::{
    BarrierScope, CheckPredicate, ConstantValue, CoreKernelOp, KernelJoin, KernelOp, KernelPlaceRef,
    KernelSliceAxis, KernelSsaDecl, KernelValueRef, KernelValueType, KernelViewStep, TypedConstant,
};
use seismic_realization::physical::{
    AccessMode, LaunchInput, LaunchOutput, PhysicalPlan, PhysicalStorageView, ScalarSource,
    SealedLaunch, StorageFact,
};
use seismic_realization::routes::{
    SliceAxisTemplate, ViewStepKind, ViewTransformTemplate,
};
use std::collections::{BTreeMap, BTreeSet};

/// The encoded form of one sealed launch: the launch itself (sealed, dense,
/// complete) plus the encoder-derived facts the assembler folds into the
/// native handle.
#[derive(Clone, Debug)]
pub struct EncodedLaunch {
    pub launch: SealedLaunch<MetalDialect>,
    /// The block's iteration map serializes the domain onto one participant
    /// (the single-loop prologue; the sealed geometry is `[1, 1, 1]`).
    pub serialized: bool,
}

/// Encode one sealed launch exactly once. Total.
pub fn encode(launch: &SealedLaunch<MetalDialect>) -> EncodedLaunch {
    EncodedLaunch {
        launch: launch.clone(),
        serialized: launch.kernel.interface().iteration.serialized,
    }
}

// ---------------------------------------------------------------------------
// The rendered launch (consumed by native assembly)
// ---------------------------------------------------------------------------

/// One absolute Metal buffer index of the rendered kernel and what the
/// executor binds there.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum RenderedBinding {
    /// A direct global storage plane: the executor binds the caller buffer
    /// (ABI placement) or the arena at its offset.
    Storage {
        index: u32,
        storage: StorageIx,
        access: AccessMode,
    },
    /// A by-value kernel scalar input: the executor encodes the source's
    /// current value.
    Scalar {
        index: u32,
        source: ScalarSource,
        dtype: DType,
    },
}

/// The fully rendered launch: MSL source plus the direct binding program.
#[derive(Clone, Debug)]
pub(crate) struct RenderedLaunch {
    pub id: LaunchIx,
    pub kernel: String,
    pub source: String,
    pub workgroup_bytes: u64,
    pub bindings: Vec<RenderedBinding>,
    /// The `seismic_total` argument index (the evaluated work-item total).
    pub total_index: u32,
    /// The runtime-extent arguments: absolute index and extent id; the
    /// executor evaluates the plan's expression for each and binds the word.
    pub extent_args: Vec<(u32, RuntimeExtentId)>,
    /// The executor-scalar slot block argument, when the plan has slots.
    pub slots_index: Option<u32>,
    /// The compiler-owned dense result scalar block argument.
    pub results_index: u32,
    /// The root status block argument, when the launch has status fields.
    pub status_index: Option<u32>,
    pub serialized: bool,
}

fn defect(invariant: impl Into<String>) -> CompilerDefect {
    CompilerDefect::new(Package::B1Metal, invariant)
}

fn counter_not_bound(counter: seismic_realization::ids::ResidenceId) -> CompilerDefect {
    defect(format!(
        "the dynamic-pull counter residence {counter:?} has no sealed storage binding"
    ))
}

// ---------------------------------------------------------------------------
// Runtime-extent collection (the pre-pass that allocates extent arguments)
// ---------------------------------------------------------------------------

fn collect_extent_expr(extent: &ExtentExpr, out: &mut BTreeSet<RuntimeExtentId>) {
    match extent {
        ExtentExpr::Static(_) => {}
        ExtentExpr::Runtime(id) => {
            out.insert(*id);
        }
        ExtentExpr::Sym(_) => {}
    }
}

fn collect_axes(axes: &[ExtentExpr], out: &mut BTreeSet<RuntimeExtentId>) {
    for axis in axes {
        collect_extent_expr(axis, out);
    }
}

fn collect_transform(transform: &ViewTransformTemplate, out: &mut BTreeSet<RuntimeExtentId>) {
    for step in &transform.steps {
        collect_axes(&step.source_shape, out);
    }
}

fn collect_chain(operand: &MatrixOperand, out: &mut BTreeSet<RuntimeExtentId>) {
    collect_axes(&operand.ty.axes, out);
    for step in &operand.view.steps {
        match step {
            KernelViewStep::Reshape { source_shape } => collect_axes(source_shape, out),
            KernelViewStep::Transpose { .. } | KernelViewStep::Slice { .. } => {}
        }
    }
}

fn collect_op_extents(op: &KernelOp<MetalIntrinsic>, out: &mut BTreeSet<RuntimeExtentId>) {
    match op {
        KernelOp::Core(core) => match core {
            CoreKernelOp::RuntimeExtent { extent, .. } => {
                out.insert(*extent);
            }
            CoreKernelOp::Fold { shape, .. } => collect_axes(&shape.axes, out),
            CoreKernelOp::Check { predicate, .. } => match predicate {
                CheckPredicate::IndexInBounds { extent, .. }
                | CheckPredicate::RangeInBounds { extent, .. } => {
                    collect_extent_expr(extent, out);
                }
                CheckPredicate::DivisorNonZero { .. }
                | CheckPredicate::SignedDivisionNoOverflow { .. }
                | CheckPredicate::ShiftInRange { .. } => {}
            },
            CoreKernelOp::Const { .. }
            | CoreKernelOp::Unary { .. }
            | CoreKernelOp::Binary { .. }
            | CoreKernelOp::Compare { .. }
            | CoreKernelOp::Math { .. }
            | CoreKernelOp::Fma { .. }
            | CoreKernelOp::Cast { .. }
            | CoreKernelOp::Select { .. }
            | CoreKernelOp::TableLookup { .. }
            | CoreKernelOp::Load { .. }
            | CoreKernelOp::Store { .. }
            | CoreKernelOp::PackedPlaneRead { .. }
            | CoreKernelOp::PlaneLoad { .. }
            | CoreKernelOp::PlaneStore { .. }
            | CoreKernelOp::Atomic { .. }
            | CoreKernelOp::Repeat { .. }
            | CoreKernelOp::Carry { .. }
            | CoreKernelOp::Branch { .. }
            | CoreKernelOp::Publish { .. }
            | CoreKernelOp::Barrier { .. }
            | CoreKernelOp::GridBarrier => {}
        },
        KernelOp::Intrinsic(intrinsic) => match intrinsic {
            MetalIntrinsic::MatrixMatmul {
                left,
                right,
                rows,
                columns,
                inner,
                ..
            } => {
                collect_chain(left, out);
                collect_chain(right, out);
                collect_extent_expr(rows, out);
                collect_extent_expr(columns, out);
                collect_extent_expr(inner, out);
            }
            MetalIntrinsic::MatrixMatmulAdd {
                left,
                right,
                addend,
                rows,
                columns,
                inner,
                ..
            } => {
                collect_chain(left, out);
                collect_chain(right, out);
                collect_chain(addend, out);
                collect_extent_expr(rows, out);
                collect_extent_expr(columns, out);
                collect_extent_expr(inner, out);
            }
            MetalIntrinsic::ParticipantIndex { .. }
            | MetalIntrinsic::Exchange { .. }
            | MetalIntrinsic::SubgroupReduce { .. } => {}
        },
    }
    for body in op.bodies() {
        for nested in body {
            collect_op_extents(nested, out);
        }
    }
}

// ---------------------------------------------------------------------------
// The renderer
// ---------------------------------------------------------------------------

struct Renderer<'a> {
    plan: &'a PhysicalPlan<MetalDialect>,
    launch: &'a SealedLaunch<MetalDialect>,
    /// Every bound global storage → its pointer name `b{index}`.
    pointers: BTreeMap<StorageIx, String>,
    /// Kernel-local places → one pointer name per plane.
    local_arrays: BTreeMap<KernelLocalId, Vec<String>>,
    /// Workgroup storages → threadgroup array names.
    workgroup_arrays: BTreeMap<StorageIx, String>,
    /// Participant storages → thread-private array names.
    participant_arrays: BTreeMap<StorageIx, String>,
    /// Canonical slice-endpoint leaves → their scalar kernel input.
    endpoint_inputs: BTreeMap<CanonicalLeafId, KernelInputId>,
    /// Runtime-extent arguments, allocated up front.
    extent_args: BTreeMap<RuntimeExtentId, u32>,
    next_index: u32,
    /// The `seismic_total` argument index.
    total_index: u32,
    slots_index: Option<u32>,
    results_index: u32,
    status_index: Option<u32>,
    declarations: Vec<String>,
    bindings: Vec<RenderedBinding>,
    /// Arrays and SSA declarations, emitted before the traversal.
    preamble: Vec<String>,
    temporaries: u32,
}

/// Render one sealed launch into MSL source and its direct binding program.
/// Total over the sealed kernel algebra; an impossible structure
/// contradicts a construction invariant of the seal and is returned as a
/// compiler defect of this package (native assembly maps it into its
/// `AssemblyFailure` channel).
pub(crate) fn render(
    launch: &SealedLaunch<MetalDialect>,
    plan: &PhysicalPlan<MetalDialect>,
) -> Result<RenderedLaunch, CompilerDefect> {
    let mut renderer = Renderer::new(launch, plan);
    renderer.declare_storage()?;
    renderer.declare_scalar_inputs()?;
    renderer.collect_extent_args()?;
    renderer.allocate_system_blocks()?;
    renderer.declare_local_arrays()?;
    renderer.declare_ssa()?;
    let body = renderer.body()?;
    let kernel = format!("seismic_metal_{}", launch.id.index());
    let source = renderer.assemble(&kernel, &body);
    Ok(RenderedLaunch {
        id: launch.id,
        kernel,
        source,
        workgroup_bytes: launch.resources.workgroup_bytes,
        bindings: renderer.bindings,
        total_index: renderer.total_index,
        extent_args: renderer
            .extent_args
            .into_iter()
            .map(|(id, index)| (index, id))
            .collect(),
        slots_index: renderer.slots_index,
        results_index: renderer.results_index,
        status_index: renderer.status_index,
        serialized: launch.kernel.interface().iteration.serialized,
    })
}

impl<'a> Renderer<'a> {
    fn new(launch: &'a SealedLaunch<MetalDialect>, plan: &'a PhysicalPlan<MetalDialect>) -> Self {
        let mut endpoint_inputs = BTreeMap::new();
        for decl in launch.kernel.interface().inputs.iter() {
            endpoint_inputs.insert(decl.leaf, decl.id);
        }
        Renderer {
            plan,
            launch,
            pointers: BTreeMap::new(),
            local_arrays: BTreeMap::new(),
            workgroup_arrays: BTreeMap::new(),
            participant_arrays: BTreeMap::new(),
            endpoint_inputs,
            extent_args: BTreeMap::new(),
            next_index: 0,
            total_index: 0,
            slots_index: None,
            results_index: 0,
            status_index: None,
            declarations: Vec::new(),
            bindings: Vec::new(),
            preamble: Vec::new(),
            temporaries: 0,
        }
    }

    fn allocate_index(&mut self) -> Result<u32, CompilerDefect> {
        let index = self.next_index;
        self.next_index = index
            .checked_add(1)
            .filter(|next| *next < u32::from(crate::target::MAX_KERNEL_BUFFERS))
            .ok_or_else(|| {
                defect(format!(
                    "launch {} needs more than {} Metal buffer arguments",
                    self.launch.id.index(),
                    crate::target::MAX_KERNEL_BUFFERS
                ))
            })?;
        Ok(index)
    }

    fn storage_argument_count(&self) -> u32 {
        self.launch.bindings.len() as u32
    }

    // -- argument declarations ----------------------------------------------

    /// Every storage binding of the launch, in binding-slot order with its
    /// address-space fact: `Global` storages become device pointers at
    /// their ordinal as the absolute Metal index; `Workgroup`/`Participant`
    /// storages become the launch-local threadgroup/thread-private arrays
    /// (sized from the fact's bytes) and never bind a device buffer.
    fn declare_storage(&mut self) -> Result<(), CompilerDefect> {
        let bindings = self.launch.bindings.clone();
        let facts = self.launch.storage_facts.clone();
        if bindings.len() != facts.len() {
            return Err(defect(format!(
                "launch {} carries {} bindings but {} storage facts",
                self.launch.id.index(),
                bindings.len(),
                facts.len()
            )));
        }
        let mut workgroup_ordinal = 0usize;
        let mut participant_ordinal = 0usize;
        for (ordinal, (binding, fact)) in bindings.iter().zip(&facts).enumerate() {
            if binding.slot != ordinal as u32 {
                return Err(defect(format!(
                    "storage binding {} carries slot {} (the ordinals are dense)",
                    ordinal, binding.slot
                )));
            }
            match fact {
                StorageFact::Global => {
                    let dtype = self.plan.storages()[binding.storage].layout.dtype();
                    let name = format!("b{}", binding.slot);
                    let address = match binding.access {
                        AccessMode::Read => "const device",
                        _ => "device",
                    };
                    self.declarations.push(format!(
                        "{address} {}* {name} [[buffer({})]]",
                        dtype_name(dtype),
                        binding.slot
                    ));
                    self.bindings.push(RenderedBinding::Storage {
                        index: binding.slot,
                        storage: binding.storage,
                        access: binding.access,
                    });
                    self.pointers.insert(binding.storage, name);
                }
                StorageFact::Workgroup { bytes, .. } => {
                    let dtype = self.plan.storages()[binding.storage].layout.dtype();
                    let words = bytes / dtype_size(dtype);
                    if words == 0 {
                        return Err(defect(format!(
                            "workgroup storage {storage:?} has no elements",
                            storage = binding.storage
                        )));
                    }
                    let name = format!("seismic_wg{workgroup_ordinal}");
                    workgroup_ordinal += 1;
                    self.preamble.push(format!(
                        "threadgroup {} {name}[{words}];",
                        dtype_name(dtype)
                    ));
                    self.preamble.push(format!("(void){name}[0];"));
                    self.workgroup_arrays.insert(binding.storage, name);
                }
                StorageFact::Participant { bytes, .. } => {
                    let dtype = self.plan.storages()[binding.storage].layout.dtype();
                    let elements = bytes / dtype_size(dtype);
                    if elements == 0 {
                        return Err(defect(format!(
                            "participant storage {storage:?} has no elements",
                            storage = binding.storage
                        )));
                    }
                    let name = format!("seismic_pr{participant_ordinal}");
                    participant_ordinal += 1;
                    self.preamble.push(format!(
                        "thread {} {name}[{elements}];",
                        dtype_name(dtype)
                    ));
                    self.preamble.push(format!("(void){name}[0];"));
                    self.participant_arrays.insert(binding.storage, name);
                }
            }
        }
        self.next_index = self.next_index.max(self.launch.bindings.len() as u32);
        Ok(())
    }

    /// Every by-value scalar kernel input: one constant reference per
    /// input, indexed after the storage bindings.
    fn declare_scalar_inputs(&mut self) -> Result<(), CompilerDefect> {
        let storage_count = self.storage_argument_count();
        for (ordinal, input) in self.launch.inputs.iter().enumerate() {
            let LaunchInput::Scalar { slot, source, dtype } = input else {
                continue;
            };
            let index = storage_count + *slot;
            let input_id = KernelInputId(
                u32::try_from(ordinal).map_err(|_| {
                    defect(format!("kernel input ordinal {ordinal} exceeds the id domain"))
                })?,
            );
            self.declarations.push(format!(
                "constant {}& k{} [[buffer({index})]]",
                dtype_name(*dtype),
                input_id.0
            ));
            self.bindings.push(RenderedBinding::Scalar {
                index,
                source: source.clone(),
                dtype: *dtype,
            });
            self.next_index = self.next_index.max(index + 1);
        }
        Ok(())
    }

    /// Every runtime extent the kernel reads: the iteration axes, every
    /// extent of every view transform and intrinsic operand, every
    /// `RuntimeExtent` op, every check predicate, and fold operand axes.
    fn collect_extent_args(&mut self) -> Result<(), CompilerDefect> {
        let mut extents = BTreeSet::new();
        let iteration = self.launch.kernel.interface().iteration.clone();
        collect_axes(&iteration.extents, &mut extents);
        for input in self.launch.inputs.iter() {
            if let LaunchInput::Storage(views) = input {
                for view in views.iter() {
                    collect_transform(&view.transform, &mut extents);
                }
            }
        }
        for output in self.launch.outputs.iter() {
            if let LaunchOutput::Storage(views) = output {
                for view in views.iter() {
                    collect_transform(&view.transform, &mut extents);
                }
            }
        }
        for local in self.launch.locals.iter() {
            for view in local.iter() {
                collect_transform(&view.transform, &mut extents);
            }
        }
        for op in self.launch.kernel.ops().iter() {
            collect_op_extents(op, &mut extents);
        }
        let ids: Vec<RuntimeExtentId> = extents.into_iter().collect();
        for id in ids {
            let index = self.allocate_index()?;
            self.declarations.push(format!(
                "constant ulong& seismic_extent_{} [[buffer({index})]]",
                id.0
            ));
            self.extent_args.insert(id, index);
        }
        // The evaluated work-item total bound as one 64-bit word: the
        // native dispatch domain never truncates it.
        let total = self.allocate_index()?;
        self.declarations
            .push(format!("constant ulong& seismic_total [[buffer({total})]]"));
        self.total_index = total;
        Ok(())
    }

    /// The system blocks: executor slots, the dense result scalar block,
    /// and the root status block.
    fn allocate_system_blocks(&mut self) -> Result<(), CompilerDefect> {
        if self.plan.resources().scalar_slots > 0 {
            let index = self.allocate_index()?;
            self.declarations
                .push(format!("device uint* seismic_slots [[buffer({index})]]"));
            self.slots_index = Some(index);
        }
        let results = self.allocate_index()?;
        self.declarations.push(format!(
            "device ulong* seismic_results [[buffer({results})]]"
        ));
        self.results_index = results;
        if !self.launch.status_fields.is_empty() {
            let index = self.allocate_index()?;
            self.declarations.push(format!(
                "device atomic_uint* seismic_status [[buffer({index})]]"
            ));
            self.status_index = Some(index);
        }
        Ok(())
    }

    // -- staged arrays and kernel-local pointers ----------------------------

    /// Kernel-local place pointers: one per plane, resolved through the
    /// plane's storage (a bound device plane, a threadgroup array, or a
    /// thread-private array — the arrays themselves are declared from the
    /// bindings' storage facts).
    fn declare_local_arrays(&mut self) -> Result<(), CompilerDefect> {
        let locals = self.launch.locals.clone();
        for (ordinal, views) in locals.iter().enumerate() {
            let id = KernelLocalId(
                u32::try_from(ordinal)
                    .map_err(|_| defect(format!("local ordinal {ordinal} exceeds the id domain")))?,
            );
            let mut planes = Vec::with_capacity(views.len());
            for (plane, view) in views.iter().enumerate() {
                let name = self.storage_pointer(view.storage).map_err(|invariant| {
                    defect(format!(
                        "kernel-local {} plane {plane} storage: {invariant}",
                        invariant
                    ))
                })?;
                planes.push(name);
            }
            self.local_arrays.insert(id, planes);
        }
        Ok(())
    }

    fn storage_pointer(&self, storage: StorageIx) -> Result<String, CompilerDefect> {
        if let Some(name) = self.pointers.get(&storage) {
            return Ok(name.clone());
        }
        if let Some(name) = self.workgroup_arrays.get(&storage) {
            return Ok(name.clone());
        }
        if let Some(name) = self.participant_arrays.get(&storage) {
            return Ok(name.clone());
        }
        Err(defect(format!(
            "storage {storage:?} has no launch binding or staged array"
        )))
    }

    /// Typed declarations of every kernel SSA value, in declaration order.
    fn declare_ssa(&mut self) -> Result<(), CompilerDefect> {
        let mut types: BTreeMap<u32, KernelValueType> = BTreeMap::new();
        let ops: Vec<KernelOp<MetalIntrinsic>> =
            self.launch.kernel.ops().iter().cloned().collect();
        for op in &ops {
            self.type_ops(op, &mut types);
        }
        let ssa: Vec<KernelSsaDecl> = self.launch.kernel.ssa().iter().cloned().collect();
        for decl in &ssa {
            let ty = types.get(&decl.id.0).cloned().ok_or_else(|| {
                defect(format!(
                    "kernel SSA {} has no defining Metal opcode",
                    decl.id.0
                ))
            })?;
            let name = match ty {
                KernelValueType::Scalar(dtype) => dtype_name(dtype).to_string(),
                KernelValueType::Capability(_) => {
                    return Err(defect(format!(
                        "kernel SSA {} holds a capability value; the Metal intrinsic \
                         families produce scalar and tensor results only",
                        decl.id.0
                    )))
                }
            };
            self.preamble
                .push(format!("{} s{} = {}(0);", name, decl.id.0, name));
        }
        Ok(())
    }

    fn type_ops(&self, op: &KernelOp<MetalIntrinsic>, types: &mut BTreeMap<u32, KernelValueType>) {
        match op {
            KernelOp::Core(core) => self.type_core(core, types),
            KernelOp::Intrinsic(intrinsic) => match intrinsic {
                MetalIntrinsic::ParticipantIndex { into, .. } => {
                    types.insert(into.0, KernelValueType::Scalar(DType::I32));
                }
                MetalIntrinsic::Exchange { into, dtype, .. }
                | MetalIntrinsic::SubgroupReduce { into, dtype, .. } => {
                    types.insert(into.0, KernelValueType::Scalar(*dtype));
                }
                MetalIntrinsic::MatrixMatmul { .. } | MetalIntrinsic::MatrixMatmulAdd { .. } => {}
            },
        }
        for body in op.bodies() {
            for nested in body {
                self.type_ops(nested, types);
            }
        }
    }

    fn type_core(&self, op: &CoreKernelOp<MetalIntrinsic>, types: &mut BTreeMap<u32, KernelValueType>) {
        use CoreKernelOp as C;
        let mut scalar = |id: &KernelSsaId, dtype: DType| {
            types.insert(id.0, KernelValueType::Scalar(dtype));
        };
        match op {
            C::Const { into, value } => scalar(into, value.dtype),
            C::RuntimeExtent { into, .. } => scalar(into, DType::I32),
            C::Unary { into, dtype, .. }
            | C::Binary { into, dtype, .. }
            | C::Math { into, dtype, .. }
            | C::Fma { into, dtype, .. }
            | C::Select { into, dtype, .. }
            | C::Load { into, dtype, .. }
            | C::PackedPlaneRead { into, dtype, .. }
            | C::PlaneLoad { into, dtype, .. } => scalar(into, *dtype),
            C::Compare { into, .. } => scalar(into, DType::Bool),
            C::Cast { into, to, .. } => scalar(into, *to),
            C::TableLookup { into, .. } => scalar(into, DType::I32),
            C::Fold { into, schema, .. } => scalar(into, schema.result),
            C::Repeat { binder, .. } => scalar(binder, DType::I32),
            C::Carry { current, result, ty, .. } => {
                types.insert(current.0, ty.clone());
                types.insert(result.0, ty.clone());
            }
            C::Branch { joins, .. } => {
                for join in joins {
                    types.insert(join.joined.0, join.ty.clone());
                }
            }
            C::Check { .. } | C::Store { .. } | C::PlaneStore { .. } | C::Atomic { .. }
            | C::Publish { .. }
            | C::Barrier { .. }
            | C::GridBarrier => {}
        }
    }

    // -- value and address helpers -------------------------------------------

    fn value(&self, reference: KernelValueRef) -> Result<String, CompilerDefect> {
        match reference {
            KernelValueRef::Ssa(id) => Ok(format!("s{}", id.0)),
            KernelValueRef::Axis(id) => Ok(format!("c{}", id.0)),
            KernelValueRef::Input(id) => match self.launch.inputs.get(id.0 as usize) {
                Some(LaunchInput::Scalar { .. }) => Ok(format!("k{}", id.0)),
                _ => Err(defect(format!(
                    "kernel input {} does not transport a scalar",
                    id.0
                ))),
            },
        }
    }

    fn ssa_name(&self, id: KernelSsaId) -> String {
        format!("s{}", id.0)
    }

    fn temporary(&mut self) -> String {
        let temp = format!("seismic_t{}", self.temporaries);
        self.temporaries += 1;
        temp
    }

    /// One extent as an MSL expression; runtime extents read their
    /// pre-allocated argument, an unresolved symbolic extent is a named
    /// defect.
    fn extent_expr(&self, extent: &ExtentExpr) -> Result<String, CompilerDefect> {
        match extent {
            ExtentExpr::Static(value) => Ok(format!("{value}ul")),
            ExtentExpr::Runtime(id) => Ok(format!("ulong(seismic_extent_{})", id.0)),
            ExtentExpr::Sym(sym) => sym
                .as_constant()
                .and_then(|value| u64::try_from(value).ok())
                .map(|value| Ok(format!("{value}ul")))
                .unwrap_or_else(|| {
                    Err(defect(format!(
                        "an unresolved symbolic extent `{sym}` reached Metal emission"
                    )))
                }),
        }
    }

    /// The plane views of one place, in plane order.
    fn place_views(&self, place: KernelPlaceRef) -> Result<&'a NonEmpty<PhysicalStorageView>, CompilerDefect> {
        match place {
            KernelPlaceRef::Input(id) => match self.launch.inputs.get(id.0 as usize) {
                Some(LaunchInput::Storage(views)) => Ok(views),
                _ => Err(defect(format!(
                    "kernel input {} does not transport storage",
                    id.0
                ))),
            },
            KernelPlaceRef::Output(id) => match self.launch.outputs.get(id.0 as usize) {
                Some(LaunchOutput::Storage(views)) => Ok(views),
                _ => Err(defect(format!(
                    "kernel output {} is scalar; it is written by Publish, not by places",
                    id.0
                ))),
            },
            KernelPlaceRef::Local(id) => self
                .launch
                .locals
                .get(id.0 as usize)
                .ok_or_else(|| defect(format!("kernel local {} is absent", id.0))),
        }
    }

    /// The pointer of one plane of one place.
    fn place_pointer(
        &self,
        place: KernelPlaceRef,
        plane: usize,
    ) -> Result<String, CompilerDefect> {
        match place {
            KernelPlaceRef::Input(id) => {
                let view = self
                    .place_views(place)?
                    .iter()
                    .nth(plane)
                    .ok_or_else(|| defect(format!("kernel input {} has no plane {plane}", id.0)))?;
                self.storage_pointer(view.storage)
            }
            KernelPlaceRef::Output(id) => {
                let view = self
                    .place_views(place)?
                    .iter()
                    .nth(plane)
                    .ok_or_else(|| defect(format!("kernel output {} has no plane {plane}", id.0)))?;
                self.storage_pointer(view.storage)
            }
            KernelPlaceRef::Local(id) => self
                .local_arrays
                .get(&id)
                .and_then(|planes| planes.get(plane))
                .cloned()
                .ok_or_else(|| {
                    defect(format!(
                        "kernel-local storage {} plane {plane} is absent",
                        id.0
                    ))
                }),
        }
    }

    /// The route transform of one place; every plane must agree on it.
    fn place_transform(&self, place: KernelPlaceRef) -> Result<ViewTransformTemplate, CompilerDefect> {
        let views = self.place_views(place)?;
        let transform = &views.first().transform;
        if views.iter().any(|view| view.transform != *transform) {
            return Err(defect(format!(
                "the planes of one physical place {place:?} disagree on their view transform"
            )));
        }
        Ok(transform.clone())
    }

    /// The residence shape of one storage, from its resolved layout.
    fn storage_shape(&self, storage: StorageIx) -> &[u64] {
        self.plan.storages()[storage].layout.shape()
    }

    /// Row-major flat index of coordinates over a shape (rendered extents).
    fn row_major_flat(coords: &[String], shape: &[String]) -> String {
        coords
            .iter()
            .zip(shape)
            .fold("0ul".to_string(), |flat, (coord, extent)| {
                format!("(({flat}) * ({extent}) + ulong({coord}))")
            })
    }

    /// Invert one composed view transform: view coordinates (shape
    /// `value_axes`, the binding's view-side tensor type) → residence
    /// coordinates. A terminal `Reshape`'s result shape is the binding's
    /// `ty.axes`.
    fn invert_transform(
        &self,
        transform: &ViewTransformTemplate,
        coords: &[String],
        value_axes: &[ExtentExpr],
    ) -> Result<Vec<String>, CompilerDefect> {
        let mut current = coords.to_vec();
        let mut current_shape: Vec<ExtentExpr> = value_axes.to_vec();
        for step in transform.steps.iter().rev() {
            let source_shape: Vec<String> = step
                .source_shape
                .iter()
                .map(|extent| self.extent_expr(extent))
                .collect::<Result<Vec<_>, _>>()?;
            current = match &step.kind {
                ViewStepKind::Reshape => {
                    let rendered: Vec<String> = current_shape
                        .iter()
                        .map(|extent| self.extent_expr(extent))
                        .collect::<Result<Vec<_>, _>>()?;
                    let flat = Self::row_major_flat(&current, &rendered);
                    let mut parent = Vec::with_capacity(source_shape.len());
                    for axis in 0..source_shape.len() {
                        let stride = source_shape
                            .iter()
                            .skip(axis + 1)
                            .fold("1ul".to_string(), |product, extent| {
                                format!("(({product}) * ({extent}))")
                            });
                        parent.push(format!(
                            "ulong((({flat}) / ({stride})) % ({}))",
                            source_shape[axis]
                        ));
                    }
                    parent
                }
                ViewStepKind::Transpose { permutation } => {
                    if permutation.len() != current.len()
                        || step.source_shape.len() != current.len()
                    {
                        return Err(defect(
                            "transpose ranks do not match its intermediate view",
                        ));
                    }
                    let mut parent = vec![None; step.source_shape.len()];
                    for (view_axis, coordinate) in current.into_iter().enumerate() {
                        let source_axis = *permutation.get(view_axis).ok_or_else(|| {
                            defect(format!("transpose lacks axis {view_axis}"))
                        })? as usize;
                        let slot = parent.get_mut(source_axis).ok_or_else(|| {
                            defect(format!("transpose names absent source axis {source_axis}"))
                        })?;
                        if slot.replace(coordinate).is_some() {
                            return Err(defect(
                                "transpose permutation repeats a source axis",
                            ));
                        }
                    }
                    let mut filled = Vec::with_capacity(parent.len());
                    for (axis, slot) in parent.into_iter().enumerate() {
                        filled.push(slot.ok_or_else(|| {
                            defect(format!("transpose omits source axis {axis}"))
                        })?);
                    }
                    filled
                }
                ViewStepKind::Slice { axes } => {
                    if axes.len() != step.source_shape.len() {
                        return Err(defect(format!(
                            "slice axis count {} does not match source rank {}",
                            axes.len(),
                            step.source_shape.len()
                        )));
                    }
                    let expected = axes
                        .iter()
                        .filter(|axis| !matches!(axis, SliceAxisTemplate::Point(_)))
                        .count();
                    if current.len() != expected {
                        return Err(defect(format!(
                            "slice intermediate rank {} does not match expected rank {expected}",
                            current.len()
                        )));
                    }
                    let mut view = current.into_iter();
                    let mut parent = Vec::with_capacity(axes.len());
                    for axis in axes {
                        parent.push(match axis {
                            SliceAxisTemplate::Point(leaf) => {
                                let input = self.endpoint_inputs.get(leaf).ok_or_else(|| {
                                    defect(format!(
                                        "slice endpoint leaf {leaf:?} has no kernel input"
                                    ))
                                })?;
                                format!("k{}", input.0)
                            }
                            SliceAxisTemplate::Range { start, .. } => {
                                let coordinate = view
                                    .next()
                                    .ok_or_else(|| defect("slice range lacks a coordinate"))?;
                                match start {
                                    Some(leaf) => {
                                        let input = self.endpoint_inputs.get(leaf).ok_or_else(|| {
                                            defect(format!(
                                                "slice endpoint leaf {leaf:?} has no kernel \
                                                 input"
                                            ))
                                        })?;
                                        format!("(({coordinate}) + (k{}))", input.0)
                                    }
                                    None => coordinate,
                                }
                            }
                            SliceAxisTemplate::Full => view
                                .next()
                                .ok_or_else(|| defect("full slice axis lacks a coordinate"))?,
                        });
                    }
                    if view.next().is_some() {
                        return Err(defect("slice left unused intermediate coordinates"));
                    }
                    parent
                }
            };
            current_shape = step.source_shape.clone();
        }
        Ok(current)
    }

    /// Flat residence index of residence coordinates over the storage's
    /// resolved shape.
    fn flat_in_storage(
        &self,
        storage: StorageIx,
        coords: &[String],
    ) -> Result<String, CompilerDefect> {
        let shape = self.storage_shape(storage);
        if coords.len() != shape.len() {
            return Err(defect(format!(
                "resolved view rank {} does not match backing storage rank {}",
                coords.len(),
                shape.len()
            )));
        }
        let mut stride = 1u64;
        let mut terms = Vec::with_capacity(coords.len());
        for (coordinate, extent) in coords.iter().zip(shape).rev() {
            terms.push(if stride == 1 {
                format!("ulong({coordinate})")
            } else {
                format!("{stride}ul * ulong({coordinate})")
            });
            stride = stride
                .checked_mul(*extent)
                .ok_or_else(|| defect("backing storage strides overflow u64"))?;
        }
        terms.reverse();
        Ok(if terms.is_empty() {
            "0ul".into()
        } else {
            terms.join(" + ")
        })
    }

    /// The (pointer, address) of one plane of one place at view
    /// coordinates.
    fn place_address(
        &self,
        place: KernelPlaceRef,
        plane: usize,
        coords: &[KernelValueRef],
    ) -> Result<(String, String), CompilerDefect> {
        let rendered: Vec<String> = coords
            .iter()
            .map(|v| self.value(*v))
            .collect::<Result<Vec<_>, _>>()?;
        let views = self.place_views(place)?;
        let view = views
            .iter()
            .nth(plane)
            .ok_or_else(|| defect(format!("place {place:?} has no plane {plane}")))?;
        let transform = self.place_transform(place)?;
        let residence = self.invert_transform(&transform, &rendered, &view.ty.axes)?;
        let address = self.flat_in_storage(view.storage, &residence)?;
        Ok((self.place_pointer(place, plane)?, address))
    }

    /// The registry representation the planes of one place hold, from the
    /// residence's resolved layout.
    fn representation_of(&self, place: KernelPlaceRef) -> Result<&'static repr::Repr, CompilerDefect> {
        let views = self.place_views(place)?;
        let storage = views.first().storage;
        match &self.plan.storages()[storage].layout {
            PhysicalMetalLayout::PackedPlane { representation, .. } => {
                repr::lookup(representation).ok_or_else(|| {
                    defect(format!("unknown representation `{representation}`"))
                })
            }
            PhysicalMetalLayout::Dense { .. } => Err(defect(format!(
                "place {place:?} is dense; it has no representation planes"
            ))),
        }
    }

    /// The (pointer, address) of one representation plane of a place,
    /// addressed by the outer view coordinates plus the packing-axis
    /// storage-element ordinal.
    fn plane_address(
        &self,
        place: KernelPlaceRef,
        plane: PlaneField,
        coords: &[KernelValueRef],
    ) -> Result<(String, String), CompilerDefect> {
        let representation = self.representation_of(place)?;
        let plane_ordinal = representation.plane_index(plane.name()).ok_or_else(|| {
            defect(format!("representation has no `{}` plane", plane.name()))
        })?;
        let views = self.place_views(place)?;
        let view = views
            .iter()
            .nth(plane_ordinal)
            .ok_or_else(|| defect(format!("place {place:?} has no plane {plane_ordinal}")))?;
        let rendered: Vec<String> = coords
            .iter()
            .map(|v| self.value(*v))
            .collect::<Result<Vec<_>, _>>()?;
        let transform = self.place_transform(place)?;
        let residence = self.invert_transform(&transform, &rendered, &view.ty.axes)?;
        let (outer, ordinal) = match residence.split_last() {
            Some((last, outer)) => (outer, last.clone()),
            None => return Err(defect("a representation plane needs a packing axis")),
        };
        let shape = self.storage_shape(view.storage).to_vec();
        let (packing_extent, outer_shape) = shape
            .split_last()
            .map(|(last, outer)| (*last, outer.to_vec()))
            .ok_or_else(|| defect("a representation plane needs a packing axis"))?;
        let schema = plane_schema(representation, plane)?;
        let per_row = per_row_entries(packing_extent, &schema);
        let outer_rendered: Vec<String> =
            outer_shape.iter().map(|extent| format!("{extent}ul")).collect();
        let flat = Self::row_major_flat(outer, &outer_rendered);
        let address = format!("(({flat}) * {per_row}ul + ulong({ordinal}))");
        Ok((self.place_pointer(place, plane_ordinal)?, address))
    }

    /// The raw entry expression of one representation plane field for the
    /// logical element at `coords`: entry `planes[plane].entry(v, entry)`
    /// of the packing-axis coordinate, zero-extended.
    fn packed_plane_entry(
        &mut self,
        place: KernelPlaceRef,
        coords: &[KernelValueRef],
        repr: &str,
        plane: PlaneField,
        entry: u32,
        out: &mut Vec<String>,
    ) -> Result<String, CompilerDefect> {
        let representation = self.representation_of(place)?;
        let plane_ordinal = representation.plane_index(plane.name()).ok_or_else(|| {
            defect(format!(
                "representation `{repr}` has no `{}` plane",
                plane.name()
            ))
        })?;
        let views = self.place_views(place)?;
        let view = views
            .iter()
            .nth(plane_ordinal)
            .ok_or_else(|| defect(format!("place {place:?} has no plane {plane_ordinal}")))?;
        let storage = view.storage;
        let rendered: Vec<String> = coords
            .iter()
            .map(|v| self.value(*v))
            .collect::<Result<Vec<_>, _>>()?;
        let transform = self.place_transform(place)?;
        let residence = self.invert_transform(&transform, &rendered, &view.ty.axes)?;
        let (outer, v) = match residence.split_last() {
            Some((last, outer)) => (outer.to_vec(), last.clone()),
            None => return Err(defect("a representation plane needs a packing axis")),
        };
        let shape = self.storage_shape(storage).to_vec();
        let (packing_extent, outer_shape) = shape
            .split_last()
            .map(|(last, outer)| (*last, outer.to_vec()))
            .ok_or_else(|| defect("a representation plane needs a packing axis"))?;
        let outer_rendered: Vec<String> =
            outer_shape.iter().map(|extent| format!("{extent}ul")).collect();
        let flat = Self::row_major_flat(&outer, &outer_rendered);
        let schema = plane_schema(representation, plane)?;
        let per_row = per_row_entries(packing_extent, &schema);
        // The entry ordinal within the plane row: the group of the logical
        // element times the plane's field count, plus the field.
        let base = match schema.fields {
            1 => format!("(({v}) / {}ul)", schema.group),
            fields => format!("((({v}) / {}ul) * {fields}ul + {entry}ul)", schema.group),
        };
        let pointer = self.place_pointer(place, plane_ordinal)?;
        Ok(match &schema.encoding {
            PlaneEncoding::Dense(_) => format!("{pointer}[(({flat}) * {per_row}ul + {base})]"),
            PlaneEncoding::Packed { bits, .. } => {
                let per_word = 32 / bits;
                let mask = (1u32 << bits) - 1;
                let temp = self.temporary();
                out.push(format!(
                    "uint {temp} = ({pointer}[(({flat}) * {per_row}ul + \
                     (({base}) / {per_word}ul))] >> (((({base}) % {per_word}ul) * {bits}ul))) \
                     & {mask}u;"
                ));
                temp
            }
        })
    }

    /// The (pointer, address) of one matrix operand at value coordinates:
    /// the in-block view chain inverts first (the operand type carries the
    /// value extents), then the route transform.
    fn chain_address(
        &self,
        operand: &MatrixOperand,
        coords: &[String],
    ) -> Result<(String, String), CompilerDefect> {
        let (place_coords, place_shape) =
            self.invert_kernel_chain(&operand.view, coords, &operand.ty.axes)?;
        let views = self.place_views(operand.place)?;
        let view = views.first();
        let transform = self.place_transform(operand.place)?;
        let residence = self.invert_transform(&transform, &place_coords, &place_shape)?;
        let address = self.flat_in_storage(view.storage, &residence)?;
        Ok((self.place_pointer(operand.place, 0)?, address))
    }

    /// Invert one in-block view chain: value coordinates → place
    /// coordinates, together with the place view's extents. The value
    /// extents are the operand type's axes; transpose and slice steps
    /// reconstruct their source extents structurally, and a chain that
    /// composes a reshape over a sliced intermediate is rejected (the
    /// chain does not carry those extents).
    fn invert_kernel_chain(
        &self,
        chain: &seismic_realization::kernel::KernelViewChain,
        coords: &[String],
        value_axes: &[ExtentExpr],
    ) -> Result<(Vec<String>, Vec<ExtentExpr>), CompilerDefect> {
        let has_reshape = chain
            .steps
            .iter()
            .any(|step| matches!(step, KernelViewStep::Reshape { .. }));
        let has_sliced_axis = chain.steps.iter().any(|step| {
            matches!(
                step,
                KernelViewStep::Slice { axes }
                    if axes
                        .iter()
                        .any(|axis| !matches!(axis, KernelSliceAxis::Full))
            )
        });
        if has_reshape && has_sliced_axis {
            return Err(defect(
                "an in-block view chain composes a reshape over a sliced intermediate; the \
                 chain does not carry those extents",
            ));
        }
        let mut current = coords.to_vec();
        let mut current_shape: Vec<ExtentExpr> = value_axes.to_vec();
        for step in chain.steps.iter().rev() {
            let rendered_shape: Vec<String> = current_shape
                .iter()
                .map(|extent| self.extent_expr(extent))
                .collect::<Result<Vec<_>, _>>()?;
            current = match step {
                KernelViewStep::Reshape { source_shape } => {
                    let source_rendered: Vec<String> = source_shape
                        .iter()
                        .map(|extent| self.extent_expr(extent))
                        .collect::<Result<Vec<_>, _>>()?;
                    let flat = Self::row_major_flat(&current, &rendered_shape);
                    let mut parent = Vec::with_capacity(source_shape.len());
                    for axis in 0..source_shape.len() {
                        let stride = source_rendered
                            .iter()
                            .skip(axis + 1)
                            .fold("1ul".to_string(), |product, extent| {
                                format!("(({product}) * ({extent}))")
                            });
                        parent.push(format!(
                            "ulong((({flat}) / ({stride})) % ({}))",
                            source_rendered[axis]
                        ));
                    }
                    parent
                }
                KernelViewStep::Transpose { permutation } => {
                    if permutation.len() != current.len() {
                        return Err(defect(
                            "transpose ranks do not match its intermediate view",
                        ));
                    }
                    let mut parent = vec![None; permutation.len()];
                    for (view_axis, coordinate) in current.into_iter().enumerate() {
                        let source_axis = *permutation.get(view_axis).ok_or_else(|| {
                            defect(format!("transpose lacks axis {view_axis}"))
                        })? as usize;
                        let slot = parent.get_mut(source_axis).ok_or_else(|| {
                            defect(format!("transpose names absent source axis {source_axis}"))
                        })?;
                        if slot.replace(coordinate).is_some() {
                            return Err(defect(
                                "transpose permutation repeats a source axis",
                            ));
                        }
                    }
                    let mut filled = Vec::with_capacity(parent.len());
                    for (axis, slot) in parent.into_iter().enumerate() {
                        filled.push(slot.ok_or_else(|| {
                            defect(format!("transpose omits source axis {axis}"))
                        })?);
                    }
                    filled
                }
                KernelViewStep::Slice { axes } => {
                    if axes.len() != current_shape.len() {
                        return Err(defect(
                            "slice axis count does not match its intermediate rank",
                        ));
                    }
                    let mut view = current.into_iter();
                    let mut parent = Vec::with_capacity(axes.len());
                    for axis in axes {
                        parent.push(match axis {
                            KernelSliceAxis::Full => view
                                .next()
                                .ok_or_else(|| defect("full slice axis lacks a coordinate"))?,
                            KernelSliceAxis::Point(value) => self.value(*value)?,
                            KernelSliceAxis::Range { start } => {
                                let coordinate = view
                                    .next()
                                    .ok_or_else(|| defect("slice range lacks a coordinate"))?;
                                match start {
                                    Some(value) => {
                                        format!("(({coordinate}) + ({}))", self.value(*value)?)
                                    }
                                    None => coordinate,
                                }
                            }
                        });
                    }
                    if view.next().is_some() {
                        return Err(defect("slice left unused intermediate coordinates"));
                    }
                    parent
                }
            };
            current_shape = match step {
                KernelViewStep::Reshape { source_shape } => source_shape.clone(),
                KernelViewStep::Transpose { permutation } => {
                    let mut shape = vec![ExtentExpr::Static(1); permutation.len()];
                    for (view_axis, axis) in permutation.iter().enumerate() {
                        shape[*axis as usize] = current_shape[view_axis].clone();
                    }
                    shape
                }
                KernelViewStep::Slice { axes } => {
                    // Full axes keep their extent; point and range axes are
                    // excluded from reshape composition by the guard above,
                    // so their placeholder extent never enters a
                    // linearization. Point axes produce no result axis;
                    // range axes do.
                    let mut view = current_shape.iter();
                    let mut shape = Vec::with_capacity(axes.len());
                    for axis in axes {
                        shape.push(match axis {
                            KernelSliceAxis::Full => view
                                .next()
                                .cloned()
                                .ok_or_else(|| defect("full slice axis lacks an extent"))?,
                            KernelSliceAxis::Range { .. } => {
                                view.next()
                                    .ok_or_else(|| defect("slice range lacks an extent"))?;
                                ExtentExpr::Static(1)
                            }
                            KernelSliceAxis::Point(_) => ExtentExpr::Static(1),
                        });
                    }
                    shape
                }
            };
        }
        Ok((current, current_shape))
    }

    // -- the traversal and op emission ---------------------------------------

    /// The traversal prologue and every op of the sealed kernel.
    fn body(&mut self) -> Result<Vec<String>, CompilerDefect> {
        let iteration = self.launch.kernel.interface().iteration.clone();
        let mut out = vec![
            "uint seismic_gid = tg_pos.x * tpg.x + tid.x;".into(),
            "uint seismic_stride = tpg.x * tgn.x;".into(),
            "bool seismic_live = true;".into(),
        ];
        let ops: Vec<KernelOp<MetalIntrinsic>> =
            self.launch.kernel.ops().iter().cloned().collect();
        match iteration.traversal {
            Traversal::OnePass if iteration.serialized => {
                // One participant traverses everything in ascending order:
                // the serialized witness (plain load/combine/round/store
                // atomics are exact under it).
                out.push(
                    "for (uint seismic_lin = 0u; seismic_lin < seismic_total; \
                     seismic_lin++) {"
                        .into(),
                );
                self.delinearize(&iteration, &mut out)?;
                self.emit_ops(&ops, &mut out)?;
                out.push("}".into());
            }
            Traversal::OnePass => {
                // Participant count equals the total: one coordinate each.
                out.push("uint seismic_lin = seismic_gid;".into());
                out.push("seismic_live = seismic_lin < seismic_total;".into());
                out.push("if (seismic_live) {".into());
                self.delinearize(&iteration, &mut out)?;
                self.emit_ops(&ops, &mut out)?;
                out.push("}".into());
            }
            Traversal::GridStride => {
                out.push(
                    "for (uint seismic_lin = seismic_gid; \
                     seismic_lin < seismic_total && seismic_live; \
                     seismic_lin += seismic_stride) {"
                        .into(),
                );
                self.delinearize(&iteration, &mut out)?;
                self.emit_ops(&ops, &mut out)?;
                out.push("}".into());
            }
            Traversal::DynamicPull { counter } => {
                let storage = self
                    .launch
                    .pull_counter
                    .ok_or_else(|| counter_not_bound(counter))?;
                let pointer = self
                    .pointers
                    .get(&storage)
                    .cloned()
                    .ok_or_else(|| counter_not_bound(counter))?;
                out.push(format!(
                    "device atomic_uint* seismic_pull = \
                     reinterpret_cast<device atomic_uint*>(&{pointer}[0ul]);"
                ));
                out.push("uint seismic_lin;".into());
                out.push(
                    "while ((seismic_lin = atomic_fetch_add_explicit(\
                     seismic_pull, 1u, memory_order_relaxed)) < seismic_total) {"
                        .into(),
                );
                self.delinearize(&iteration, &mut out)?;
                self.emit_ops(&ops, &mut out)?;
                out.push("}".into());
            }
        }
        Ok(out)
    }

    /// Per-axis coordinate variables: delinearization of the linear
    /// coordinate into every logical axis (the last axis is fastest).
    fn delinearize(
        &mut self,
        iteration: &seismic_realization::dispatch::LinearIterationMap,
        out: &mut Vec<String>,
    ) -> Result<(), CompilerDefect> {
        let rank = iteration.extents.len();
        if rank == 0 {
            return Ok(());
        }
        let extents = iteration.extents.clone();
        let mut rest = "seismic_lin".to_string();
        for axis in 0..rank {
            let divisor = extents[axis + 1..]
                .iter()
                .map(|extent| self.extent_expr(extent))
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .reduce(|left, right| format!("({left} * {right})"))
                .unwrap_or_else(|| "1ul".into());
            if divisor == "1ul" {
                out.push(format!("uint c{axis} = {rest}; uint seismic_r{axis} = 0u;"));
            } else {
                out.push(format!(
                    "uint c{axis} = {rest} / {divisor}; \
                     uint seismic_r{axis} = {rest} % {divisor};"
                ));
            }
            rest = format!("seismic_r{axis}");
        }
        Ok(())
    }

    fn emit_ops(
        &mut self,
        ops: &[KernelOp<MetalIntrinsic>],
        out: &mut Vec<String>,
    ) -> Result<(), CompilerDefect> {
        for op in ops {
            self.emit_op(op, out)?;
        }
        Ok(())
    }

    fn emit_op(
        &mut self,
        op: &KernelOp<MetalIntrinsic>,
        out: &mut Vec<String>,
    ) -> Result<(), CompilerDefect> {
        match op {
            KernelOp::Core(core) => self.emit_core(core, out),
            KernelOp::Intrinsic(intrinsic) => self.emit_intrinsic(intrinsic, out),
        }
    }

    fn emit_core(
        &mut self,
        op: &CoreKernelOp<MetalIntrinsic>,
        out: &mut Vec<String>,
    ) -> Result<(), CompilerDefect> {
        use CoreKernelOp as C;
        match op {
            C::Const { into, value } => {
                out.push(format!("s{} = {};", into.0, const_expr(value)?));
            }
            C::RuntimeExtent { into, extent } => {
                out.push(format!("s{} = int(seismic_extent_{});", into.0, extent.0));
            }
            C::Unary {
                into,
                op: unary,
                operand,
                dtype,
            } => {
                let name = self.ssa_name(*into);
                let operand = self.value(*operand)?;
                let body = match (unary, dtype.is_float()) {
                    (UnaryOp::Neg, true) => format!("{name} = -{operand};"),
                    (UnaryOp::Neg, false) => format!(
                        "{name} = as_type<{}>(0u - as_type<uint>({operand}));",
                        dtype_name(*dtype)
                    ),
                    (UnaryOp::BitNot, _) => format!("{name} = ~{operand};"),
                    (UnaryOp::Not, _) => format!("{name} = !{operand};"),
                };
                out.push(body);
            }
            C::Binary {
                into,
                op: binary,
                left,
                right,
                dtype,
            } => self.emit_binary(*into, *binary, *left, *right, *dtype, out)?,
            C::Compare {
                into,
                op: rel,
                left,
                right,
                dtype,
            } => {
                let name = self.ssa_name(*into);
                let left = self.value(*left)?;
                let right = self.value(*right)?;
                let cast = comparison_cast(*dtype);
                out.push(format!(
                    "{name} = {cast}({left}) {} {cast}({right});",
                    rel_symbol(*rel)
                ));
            }
            C::Math {
                into,
                op: math,
                operands,
                dtype,
            } => {
                let name = self.ssa_name(*into);
                let args: Vec<String> = operands
                    .iter()
                    .map(|v| self.value(*v))
                    .collect::<Result<Vec<_>, _>>()?;
                out.push(format!("{name} = {};", math_call(*math, &args, *dtype)));
            }
            C::Fma {
                into,
                a,
                b,
                c,
                dtype: _,
            } => {
                let name = self.ssa_name(*into);
                out.push(format!(
                    "{name} = fma({}, {}, {});",
                    self.value(*a)?,
                    self.value(*b)?,
                    self.value(*c)?
                ));
            }
            C::Cast {
                into,
                operand,
                from,
                to,
            } => {
                let name = self.ssa_name(*into);
                out.push(format!(
                    "{name} = {};",
                    cast_expr(*from, *to, &self.value(*operand)?)
                ));
            }
            C::Select {
                into,
                condition,
                then_value,
                else_value,
                dtype: _,
            } => {
                let name = self.ssa_name(*into);
                out.push(format!(
                    "{name} = ({}) ? {} : {};",
                    self.value(*condition)?,
                    self.value(*then_value)?,
                    self.value(*else_value)?
                ));
            }
            C::TableLookup { into, index, table } => {
                let temp = self.temporary();
                let values = table
                    .iter()
                    .map(|value| format!("int({value})"))
                    .collect::<Vec<_>>()
                    .join(", ");
                out.push(format!("constant int {temp}[] = {{ {values} }};"));
                out.push(format!(
                    "s{} = int({temp}[uint({})]);",
                    into.0,
                    self.value(*index)?
                ));
            }
            C::Load {
                into,
                place,
                coords,
                dtype: _,
            } => {
                let name = self.ssa_name(*into);
                let (pointer, address) = self.place_address(*place, 0, coords)?;
                out.push(format!("{name} = {pointer}[{address}];"));
            }
            C::Store {
                place,
                coords,
                value,
                dtype,
            } => {
                let (pointer, address) = self.place_address(*place, 0, coords)?;
                out.push(format!(
                    "{pointer}[{address}] = {}({});",
                    dtype_name(*dtype),
                    self.value(*value)?
                ));
            }
            C::PackedPlaneRead {
                into,
                place,
                coords,
                repr,
                plane,
                entry,
                dtype,
            } => {
                let name = self.ssa_name(*into);
                let value =
                    self.packed_plane_entry(*place, coords, repr, *plane, *entry, out)?;
                out.push(format!("{name} = {}({value});", dtype_name(*dtype)));
            }
            C::PlaneLoad {
                into,
                place,
                coords,
                repr: _,
                plane,
                dtype,
            } => {
                let name = self.ssa_name(*into);
                let (pointer, address) = self.plane_address(*place, *plane, coords)?;
                out.push(format!(
                    "{name} = {}({pointer}[{address}]);",
                    dtype_name(*dtype)
                ));
            }
            C::PlaneStore {
                place,
                coords,
                repr: _,
                plane,
                value,
                dtype,
            } => {
                let (pointer, address) = self.plane_address(*place, *plane, coords)?;
                out.push(format!(
                    "{pointer}[{address}] = {}({});",
                    dtype_name(*dtype),
                    self.value(*value)?
                ));
            }
            C::Atomic {
                place,
                coords,
                op: atomic,
                value,
                dtype,
                mode,
            } => self.emit_atomic(*place, coords, *atomic, *value, *dtype, *mode, out)?,
            C::Repeat {
                binder,
                start,
                end,
                body,
            } => self.emit_repeat(*binder, *start, *end, body, out)?,
            C::Carry { .. } => {
                return Err(defect(
                    "a Carry op reached emission outside the head of its Repeat body",
                ))
            }
            C::Branch {
                condition,
                then_body,
                else_body,
                joins,
            } => self.emit_branch(*condition, then_body, else_body, joins, out)?,
            C::Fold {
                into,
                op,
                place,
                axis,
                coords,
                schema,
                shape,
            } => self.emit_fold(*into, *op, *place, *axis, coords, schema, shape, out)?,
            C::Check {
                obligation: _,
                predicate,
                status,
                guarded,
            } => self.emit_check(predicate, *status, guarded, out)?,
            C::Publish { value, output } => {
                let output = self
                    .launch
                    .outputs
                    .get(output.0 as usize)
                    .ok_or_else(|| defect(format!("kernel output {} is absent", output.0)))?;
                let rendered = self.value(*value)?;
                match output {
                    LaunchOutput::ExecutorSlot { slot, dtype } => {
                        out.push(format!(
                            "if (seismic_live) {{ seismic_slots[{}] = {}; }}",
                            slot.index(),
                            slot_write(&rendered, *dtype)
                        ));
                    }
                    LaunchOutput::ResultField { field, dtype } => {
                        out.push(format!(
                            "if (seismic_live) {{ seismic_results[{}] = ulong({}); }}",
                            field.index(),
                            slot_write(&rendered, *dtype)
                        ));
                    }
                    LaunchOutput::Storage(_) => {
                        return Err(defect(
                            "a tensor output publishes through its store sites, not a Publish \
                             op",
                        ))
                    }
                }
            }
            C::Barrier { scope } => match scope {
                BarrierScope::Subgroup => out.push("simdgroup_barrier();".into()),
                BarrierScope::Workgroup => {
                    out.push("threadgroup_barrier(mem_flags::mem_threadgroup);".into())
                }
            },
            C::GridBarrier => {
                return Err(defect(
                    "a grid barrier reached the Metal encoder; Metal never proposes a \
                     grid-cooperative launch",
                ))
            }
        }
        Ok(())
    }

    fn emit_binary(
        &mut self,
        into: KernelSsaId,
        op: BinaryOp,
        left: KernelValueRef,
        right: KernelValueRef,
        dtype: DType,
        out: &mut Vec<String>,
    ) -> Result<(), CompilerDefect> {
        use BinaryOp as B;
        let name = self.ssa_name(into);
        let left = self.value(left)?;
        let right = self.value(right)?;
        match op {
            B::And => out.push(format!("{name} = {left} && {right};")),
            B::Or => out.push(format!("{name} = {left} || {right};")),
            B::Eq | B::Ne | B::Lt | B::Le | B::Gt | B::Ge => {
                let cast = comparison_cast(dtype);
                out.push(format!(
                    "{name} = {cast}({left}) {} {cast}({right});",
                    rel_symbol(rel_of_binary(op)?)
                ));
            }
            B::Shl => out.push(format!(
                "{name} = as_type<{}>(as_type<uint>({left}) << uint({right}));",
                dtype_name(dtype)
            )),
            B::Shr => out.push(format!(
                "{name} = as_type<{}>(as_type<{}>({left}) >> uint({right}));",
                dtype_name(dtype),
                if dtype == DType::U32 { "uint" } else { "int" }
            )),
            B::BitAnd | B::BitOr | B::BitXor => out.push(format!(
                "{name} = as_type<{}>(as_type<uint>({left}) {} as_type<uint>({right}));",
                dtype_name(dtype),
                int_symbol(op)?
            )),
            B::Add | B::Sub | B::Mul if dtype.is_float() => {
                out.push(format!("{name} = {left} {} {right};", float_symbol(op)?))
            }
            B::Add | B::Sub | B::Mul => out.push(format!(
                "{name} = as_type<{}>(as_type<uint>({left}) {} as_type<uint>({right}));",
                dtype_name(dtype),
                int_symbol(op)?
            )),
            B::Div | B::Rem if dtype.is_float() => {
                let expression = if op == B::Div {
                    format!("{left} / {right}")
                } else {
                    format!("fmod({left}, {right})")
                };
                out.push(format!("{name} = {expression};"));
            }
            B::Div | B::Rem => {
                // Euclidean division/remainder (r in 0..|rhs|), exactly the
                // registry semantics; divisor safety is a separate Check.
                let temp = self.temporary();
                if op == B::Div {
                    out.push(format!(
                        "int {temp} = as_type<int>({left}) % as_type<int>({right}); \
                         if ({temp} < 0) {temp} += (as_type<int>({right}) < 0 ? \
                         -as_type<int>({right}) : as_type<int>({right})); \
                         {name} = {}((as_type<int>({left}) - {temp}) / as_type<int>({right}));",
                        dtype_name(dtype),
                    ));
                } else {
                    out.push(format!(
                        "int {temp} = as_type<int>({left}) % as_type<int>({right}); \
                         if ({temp} < 0) {temp} += (as_type<int>({right}) < 0 ? \
                         -as_type<int>({right}) : as_type<int>({right})); \
                         {name} = {}({temp});",
                        dtype_name(dtype),
                    ));
                }
            }
        }
        Ok(())
    }

    /// The atomic read-modify-write. `Serialized` (one participant owns the
    /// domain) is the exact load/combine/round/store sequence; `Device` is
    /// the native fetch op for 32-bit integers and a compare/exchange loop
    /// on the bits for `f32` (float `max`/`min` ignore a NaN operand; `add`
    /// rounds once per update and reassociates across participants).
    fn emit_atomic(
        &mut self,
        place: KernelPlaceRef,
        coords: &[KernelValueRef],
        op: AtomicOp,
        value: KernelValueRef,
        dtype: DType,
        mode: seismic_realization::kernel::AtomicMode,
        out: &mut Vec<String>,
    ) -> Result<(), CompilerDefect> {
        let (pointer, address) = self.place_address(place, 0, coords)?;
        let value = self.value(value)?;
        let name = dtype_name(dtype);
        match mode {
            seismic_realization::kernel::AtomicMode::Serialized => {
                let combined = match (op, dtype) {
                    (AtomicOp::Add, _) => format!("{name}(seismic_a + {value})"),
                    (AtomicOp::Max, DType::I32 | DType::U32) => {
                        format!("max(seismic_a, {value})")
                    }
                    (AtomicOp::Min, DType::I32 | DType::U32) => {
                        format!("min(seismic_a, {value})")
                    }
                    (AtomicOp::Max, _) => {
                        format!("{name}(max(float(seismic_a), float({value})))")
                    }
                    (AtomicOp::Min, _) => {
                        format!("{name}(min(float(seismic_a), float({value})))")
                    }
                };
                out.push(format!(
                    "{{ {name} seismic_a = {pointer}[{address}]; \
                     {name} seismic_b = {combined}; \
                     {pointer}[{address}] = seismic_b; }}"
                ));
            }
            seismic_realization::kernel::AtomicMode::Device => {
                let body = match dtype {
                    DType::I32 | DType::U32 => {
                        let atomic = if dtype == DType::I32 {
                            "atomic_int"
                        } else {
                            "atomic_uint"
                        };
                        let operation = match op {
                            AtomicOp::Add => "add",
                            AtomicOp::Max => "max",
                            AtomicOp::Min => "min",
                        };
                        format!(
                            "{{ device {atomic}* seismic_p = \
                             reinterpret_cast<device {atomic}*>(&{pointer}[{address}]); \
                             atomic_fetch_{operation}_explicit(\
                             seismic_p, {value}, memory_order_relaxed); }}"
                        )
                    }
                    DType::F32 => {
                        let combined = match op {
                            AtomicOp::Add => "seismic_a + float(seismic_v)",
                            AtomicOp::Max => "max(seismic_a, float(seismic_v))",
                            AtomicOp::Min => "min(seismic_a, float(seismic_v))",
                        };
                        format!(
                            "{{ device atomic_uint* seismic_p = \
                             reinterpret_cast<device atomic_uint*>(&{pointer}[{address}]); \
                             float seismic_v = float({value}); \
                             uint seismic_expected = atomic_load_explicit(\
                             seismic_p, memory_order_relaxed); \
                             while (true) {{ float seismic_a = \
                             as_type<float>(seismic_expected); \
                             uint seismic_desired = as_type<uint>({combined}); \
                             if (atomic_compare_exchange_weak_explicit(\
                             seismic_p, &seismic_expected, seismic_desired, \
                             memory_order_relaxed, memory_order_relaxed)) break; }} }}"
                        )
                    }
                    other => {
                        return Err(defect(format!(
                            "device atomic emission received {} (the concurrent peer admits \
                             32-bit elements only)",
                            other.name()
                        )))
                    }
                };
                out.push(body);
            }
        }
        Ok(())
    }

    /// An ascending serial loop; `Carry` ops at the head of the body thread
    /// their values through the visits and define the result after it.
    fn emit_repeat(
        &mut self,
        binder: KernelSsaId,
        start: KernelValueRef,
        end: KernelValueRef,
        body: &[KernelOp<MetalIntrinsic>],
        out: &mut Vec<String>,
    ) -> Result<(), CompilerDefect> {
        let (carries, rest): (Vec<&KernelOp<MetalIntrinsic>>, Vec<&KernelOp<MetalIntrinsic>>) =
            body.iter().partition(|op| {
                matches!(op, KernelOp::Core(CoreKernelOp::Carry { .. }))
            });
        let carry_of = |op: &KernelOp<MetalIntrinsic>| -> Result<
            (
                KernelValueRef,
                KernelSsaId,
                KernelValueRef,
                KernelSsaId,
                KernelValueType,
            ),
            CompilerDefect,
        > {
            match op {
                KernelOp::Core(CoreKernelOp::Carry {
                    initial,
                    current,
                    update,
                    result,
                    ty,
                }) => Ok((*initial, *current, *update, *result, ty.clone())),
                _ => Err(defect(
                    "a Carry op appears outside the head of its Repeat body",
                )),
            }
        };
        for carry in &carries {
            let (initial, current, _, _, ty) = carry_of(carry)?;
            let ty_name = scalar_type_name(&ty)?;
            out.push(format!(
                "{ty_name} s{} = {};",
                current.0,
                self.value(initial)?
            ));
        }
        let binder_name = self.ssa_name(binder);
        out.push(format!(
            "for (int {binder_name} = int({}); \
             {binder_name} < int({}) && seismic_live; {binder_name}++) {{",
            self.value(start)?,
            self.value(end)?
        ));
        for op in &rest {
            self.emit_op(op, out)?;
        }
        for carry in &carries {
            let (_, current, update, _, _) = carry_of(carry)?;
            out.push(format!(
                "if (seismic_live) {{ s{} = {}; }}",
                current.0,
                self.value(update)?
            ));
        }
        out.push("}".into());
        for carry in &carries {
            let (_, current, _, result, ty) = carry_of(carry)?;
            let ty_name = scalar_type_name(&ty)?;
            out.push(format!(
                "if (seismic_live) {{ {ty_name} s{} = s{}; }}",
                result.0, current.0
            ));
        }
        Ok(())
    }

    fn emit_branch(
        &mut self,
        condition: KernelValueRef,
        then_body: &[KernelOp<MetalIntrinsic>],
        else_body: &[KernelOp<MetalIntrinsic>],
        joins: &[KernelJoin],
        out: &mut Vec<String>,
    ) -> Result<(), CompilerDefect> {
        out.push(format!("if ({}) {{", self.value(condition)?));
        self.emit_ops(then_body, out)?;
        for join in joins {
            out.push(format!(
                "if (seismic_live) {{ s{} = {}; }}",
                join.joined.0,
                self.value(join.then_value)?
            ));
        }
        if else_body.is_empty() && joins.is_empty() {
            out.push("}".into());
        } else {
            out.push("} else {".into());
            self.emit_ops(else_body, out)?;
            for join in joins {
                out.push(format!(
                    "if (seismic_live) {{ s{} = {}; }}",
                    join.joined.0,
                    self.value(join.else_value)?
                ));
            }
            out.push("}".into());
        }
        Ok(())
    }

    /// The registry serial reduction fold: ascending visits over the
    /// reduced axis at the fixed outer coordinates, under the schema's
    /// identity, accumulator, result, and tie rule.
    #[allow(clippy::too_many_arguments)]
    fn emit_fold(
        &mut self,
        into: KernelSsaId,
        op: ReduceOp,
        place: KernelPlaceRef,
        axis: u32,
        coords: &[KernelValueRef],
        schema: &ReduceSchema,
        shape: &TensorType,
        out: &mut Vec<String>,
    ) -> Result<(), CompilerDefect> {
        let axis = axis as usize;
        if axis >= shape.axes.len() {
            return Err(defect(format!(
                "a fold reduces axis {axis} of a rank-{} tensor",
                shape.axes.len()
            )));
        }
        let axis_length = self.extent_expr(&shape.axes[axis])?;
        // Operand coordinates in axis order: the fixed outer coordinates
        // with the fold variable at the reduced position (`None`).
        let mut with_k: Vec<Option<KernelValueRef>> = Vec::with_capacity(shape.axes.len());
        let mut outer = 0usize;
        for position in 0..shape.axes.len() {
            if position == axis {
                with_k.push(None);
            } else {
                let coordinate = coords.get(outer).ok_or_else(|| {
                    defect("the fold's fixed coordinates cover every other axis")
                })?;
                with_k.push(Some(*coordinate));
                outer += 1;
            }
        }
        let acc = dtype_name(schema.accumulator);
        let result_name = self.ssa_name(into);
        let load = |k: &str| -> Result<String, CompilerDefect> {
            let coords: Vec<String> = with_k
                .iter()
                .map(|slot| match slot {
                    None => Ok(k.to_string()),
                    Some(value) => self.value(*value),
                })
                .collect::<Result<Vec<_>, _>>()?;
            let views = self.place_views(place)?;
            let view = views.first();
            let transform = self.place_transform(place)?;
            let residence = self.invert_transform(&transform, &coords, &view.ty.axes)?;
            let address = self.flat_in_storage(view.storage, &residence)?;
            Ok(format!("{}[{address}]", self.place_pointer(place, 0)?))
        };
        let body = match schema.identity {
            ReduceIdentity::Zero => format!(
                "{acc} seismic_acc = 0; \
                 for (uint seismic_k = 0u; seismic_k < {axis_length}ul; seismic_k++) \
                 {{ seismic_acc = {acc}(seismic_acc + {}); }}",
                load("seismic_k")?
            ),
            ReduceIdentity::FirstElement => format!(
                "{acc} seismic_acc = {}; \
                 for (uint seismic_k = 1u; seismic_k < {axis_length}ul; seismic_k++) \
                 {{ seismic_acc = {}(seismic_acc, {}); }}",
                load("0u")?,
                match op {
                    ReduceOp::Max => "max",
                    _ => "min",
                },
                load("seismic_k")?
            ),
            ReduceIdentity::FirstElementNonEmpty => format!(
                // Argmax: the running extremum with its coordinate; a
                // strictly greater value replaces it, so the smaller
                // coordinate wins ties.
                "{acc} seismic_acc = {}; int seismic_best = 0; \
                 for (uint seismic_k = 1u; seismic_k < {axis_length}ul; seismic_k++) {{ \
                 {acc} seismic_v = {}; \
                 if (seismic_v > seismic_acc) {{ seismic_acc = seismic_v; \
                 seismic_best = int(seismic_k); }} }}",
                load("0u")?,
                load("seismic_k")?
            ),
        };
        let publish = match schema.identity {
            ReduceIdentity::FirstElementNonEmpty => format!("{result_name} = seismic_best;"),
            _ => {
                if schema.result != schema.accumulator {
                    format!("{result_name} = {}(seismic_acc);", dtype_name(schema.result))
                } else {
                    format!("{result_name} = seismic_acc;")
                }
            }
        };
        out.push(format!("{{ {body} {publish} }}"));
        Ok(())
    }

    /// One guarded safety check: when the predicate holds the guarded ops
    /// execute; when it fails the first error is recorded in the status
    /// field (compare/exchange, zero word wins) and the guarded ops are
    /// skipped, their SSAs holding unspecified values of their type.
    fn emit_check(
        &mut self,
        predicate: &CheckPredicate,
        status: StatusFieldTemplateId,
        guarded: &[KernelOp<MetalIntrinsic>],
        out: &mut Vec<String>,
    ) -> Result<(), CompilerDefect> {
        let condition = self.predicate(predicate)?;
        let field = self.status_index(status)?;
        let code = guard_code(predicate);
        let mut body = Vec::new();
        self.emit_ops(guarded, &mut body)?;
        let body = body.join(" ");
        out.push(format!(
            "if ({condition}) {{ {body} }} else {{ uint expected = 0u; \
             atomic_compare_exchange_weak_explicit(&seismic_status[{field}], &expected, \
             {code}, memory_order_relaxed, memory_order_relaxed); \
             seismic_live = false; }}"
        ));
        Ok(())
    }

    /// The dense index of one status field: its position in the kernel's
    /// template order mapped through the launch's sealed field list.
    fn status_index(&self, template: StatusFieldTemplateId) -> Result<usize, CompilerDefect> {
        let position = self
            .launch
            .kernel
            .status_fields()
            .iter()
            .position(|(id, _)| *id == template)
            .ok_or_else(|| {
                defect(format!(
                    "status template {} is absent from the sealed kernel",
                    template.0
                ))
            })?;
        self.launch
            .status_fields
            .get(position)
            .map(|field| field.index())
            .ok_or_else(|| {
                defect(format!(
                    "status template {} has no sealed status field",
                    template.0
                ))
            })
    }

    fn predicate(&self, predicate: &CheckPredicate) -> Result<String, CompilerDefect> {
        Ok(match predicate {
            CheckPredicate::IndexInBounds { index, extent } => {
                let index = self.value(*index)?;
                format!(
                    "({index} >= 0 && uint({index}) < {})",
                    self.extent_expr(extent)?
                )
            }
            CheckPredicate::RangeInBounds { start, end, extent } => {
                let start = self.value(*start)?;
                let end = self.value(*end)?;
                format!(
                    "({start} >= 0 && {start} <= {end} && uint({end}) <= {})",
                    self.extent_expr(extent)?
                )
            }
            CheckPredicate::DivisorNonZero { value, dtype: _ } => {
                format!("({} != 0)", self.value(*value)?)
            }
            CheckPredicate::SignedDivisionNoOverflow { lhs, rhs } => format!(
                "({rhs} != 0 && ({lhs} != (-2147483647 - 1) || {rhs} != -1))",
                lhs = self.value(*lhs)?,
                rhs = self.value(*rhs)?,
            ),
            CheckPredicate::ShiftInRange { value } => {
                let value = self.value(*value)?;
                format!("({value} >= 0 && {value} < 32)")
            }
        })
    }

    // -- intrinsics -----------------------------------------------------------

    fn emit_intrinsic(
        &mut self,
        intrinsic: &MetalIntrinsic,
        out: &mut Vec<String>,
    ) -> Result<(), CompilerDefect> {
        match intrinsic {
            MetalIntrinsic::ParticipantIndex { into, .. } => {
                out.push(format!("s{} = int(simd_lane);", into.0));
            }
            MetalIntrinsic::Exchange {
                value,
                lane,
                into,
                dtype: _,
                ..
            } => {
                out.push(format!(
                    "s{} = simd_shuffle({}, uint({}));",
                    into.0,
                    self.value(*value)?,
                    self.value(*lane)?
                ));
            }
            MetalIntrinsic::SubgroupReduce {
                op, value, into, ..
            } => {
                let collective = match op {
                    ReduceOp::Sum => "simd_sum",
                    ReduceOp::Max => "simd_max",
                    ReduceOp::Min => "simd_min",
                    ReduceOp::Argmax => {
                        return Err(defect(
                            "argmax never reassociates; no subgroup collective is proposed \
                             for it",
                        ))
                    }
                };
                out.push(format!(
                    "s{} = {}({});",
                    into.0,
                    collective,
                    self.value(*value)?
                ));
            }
            MetalIntrinsic::MatrixMatmul {
                left,
                right,
                into,
                inner,
                elem,
                accumulator,
                ..
            } => self.emit_matmul(
                left, right, None, *into, inner, *elem, *accumulator, out,
            )?,
            MetalIntrinsic::MatrixMatmulAdd {
                left,
                right,
                addend,
                into,
                inner,
                elem,
                accumulator,
                ..
            } => self.emit_matmul(
                left,
                right,
                Some(addend),
                *into,
                inner,
                *elem,
                *accumulator,
                out,
            )?,
        }
        Ok(())
    }

    /// One output element per participant (the block's axis coordinates
    /// `c0`/`c1` over `[rows, columns]`); the exact ascending-k fma chain
    /// the reference body defines. The fragment/staging realization is
    /// later optimization work through new registry families.
    fn emit_matmul(
        &mut self,
        left: &MatrixOperand,
        right: &MatrixOperand,
        addend: Option<&MatrixOperand>,
        into: KernelPlaceRef,
        inner: &ExtentExpr,
        elem: DType,
        accumulator: DType,
        out: &mut Vec<String>,
    ) -> Result<(), CompilerDefect> {
        let (left_pointer, left_address) =
            self.chain_address(left, &["c0".into(), "mm_k".into()])?;
        let (right_pointer, right_address) =
            self.chain_address(right, &["c1".into(), "mm_k".into()])?;
        let (into_pointer, into_address) =
            self.place_address_string(into, 0, &["c0".into(), "c1".into()])?;
        let k_length = self.extent_expr(inner)?;
        let initial = match addend {
            Some(addend) => {
                let (pointer, address) = self.chain_address(addend, &["c0".into(), "c1".into()])?;
                format!("{pointer}[{address}]")
            }
            None => format!("{}(0)", dtype_name(accumulator)),
        };
        let temp = self.temporary();
        let accumulator_name = dtype_name(accumulator);
        let elem_name = dtype_name(elem);
        out.push(format!("{accumulator_name} {temp} = {initial};"));
        out.push(format!(
            "for (uint mm_k = 0u; mm_k < {k_length}ul; mm_k++) \
             {temp} = fma({elem_name}({left_pointer}[{left_address}]), \
             {elem_name}({right_pointer}[{right_address}]), {temp});"
        ));
        out.push(format!(
            "{into_pointer}[{into_address}] = {accumulator_name}({temp});"
        ));
        Ok(())
    }

    /// The (pointer, address) of one plane of one place at view
    /// coordinates already rendered as MSL expressions.
    fn place_address_string(
        &self,
        place: KernelPlaceRef,
        plane: usize,
        coords: &[String],
    ) -> Result<(String, String), CompilerDefect> {
        let views = self.place_views(place)?;
        let view = views
            .iter()
            .nth(plane)
            .ok_or_else(|| defect(format!("place {place:?} has no plane {plane}")))?;
        let transform = self.place_transform(place)?;
        let residence = self.invert_transform(&transform, coords, &view.ty.axes)?;
        let address = self.flat_in_storage(view.storage, &residence)?;
        Ok((self.place_pointer(place, plane)?, address))
    }

    // -- assembly ---------------------------------------------------------------

    fn assemble(&self, kernel: &str, body: &[String]) -> String {
        // The library preamble (`#include`, namespace, fast-math contract)
        // is emitted once by native assembly; a rendered kernel is one
        // self-contained function.
        let mut declarations = self.declarations.clone();
        declarations.push("uint3 tg_pos [[threadgroup_position_in_grid]]".into());
        declarations.push("uint3 tid [[thread_position_in_threadgroup]]".into());
        declarations.push("uint3 tpg [[threads_per_threadgroup]]".into());
        declarations.push("uint3 tgn [[threadgroups_per_grid]]".into());
        declarations.push("uint simd_lane [[thread_index_in_simdgroup]]".into());
        let mut source = format!(
            "kernel void {kernel}(\n    {}\n) {{\n",
            declarations.join(",\n    ")
        );
        for line in self.preamble.iter().chain(body) {
            source.push_str("    ");
            source.push_str(line);
            source.push('\n');
        }
        source.push_str("}\n\n");
        source
    }
}

// ---------------------------------------------------------------------------
// Representation planes
// ---------------------------------------------------------------------------

fn plane_schema(
    representation: &repr::Repr,
    plane: PlaneField,
) -> Result<repr::Plane, CompilerDefect> {
    representation
        .planes()
        .into_iter()
        .find(|candidate| candidate.name == plane.name())
        .ok_or_else(|| defect(format!("representation has no `{}` plane", plane.name())))
}

/// Storage elements per plane row: groups per row (over the residence's
/// resolved packing-axis extent) times fields per group.
fn per_row_entries(packing_extent: u64, plane: &repr::Plane) -> u64 {
    packing_extent
        .div_ceil(u64::from(plane.group))
        .saturating_mul(u64::from(plane.fields))
}

// ---------------------------------------------------------------------------
// MSL emission helpers (ported from the former printer)
// ---------------------------------------------------------------------------

fn dtype_size(dtype: DType) -> u64 {
    u64::from(dtype.bytes())
}

fn dtype_name(dtype: DType) -> &'static str {
    match dtype {
        DType::Bool => "bool",
        DType::I32 => "int",
        DType::U32 => "uint",
        DType::F16 => "half",
        DType::BF16 => "bfloat",
        DType::F32 => "float",
    }
}

/// The MSL type name of one kernel SSA type; a capability-typed SSA has no
/// Metal scalar name (the Metal intrinsic families produce scalar and
/// tensor results only).
fn scalar_type_name(ty: &KernelValueType) -> Result<&'static str, CompilerDefect> {
    match ty {
        KernelValueType::Scalar(dtype) => Ok(dtype_name(*dtype)),
        KernelValueType::Capability(_) => Err(defect(
            "a carried capability value is not a kernel scalar; the seal rejects it",
        )),
    }
}

fn const_expr(value: &TypedConstant) -> Result<String, CompilerDefect> {
    let mismatch = || {
        defect(format!(
            "a {} constant cannot carry a {} value",
            value.dtype.name(),
            match value.value {
                ConstantValue::Int(_) => "integer",
                ConstantValue::Float { .. } => "float",
                ConstantValue::Bool(_) => "boolean",
            }
        ))
    };
    Ok(match (value.value, value.dtype) {
        (ConstantValue::Int(v), DType::I32) => format!("int({v})"),
        (ConstantValue::Int(v), DType::U32) => format!("uint({v})"),
        (ConstantValue::Int(v), DType::Bool) => format!("{}", v != 0),
        (ConstantValue::Float { bits }, DType::F32) => {
            format!("as_type<float>(uint({:#010x}u))", bits as u32)
        }
        (ConstantValue::Float { bits }, DType::F16) => {
            format!("as_type<half>(ushort({:#010x}u & 0xffffu))", bits)
        }
        (ConstantValue::Float { bits }, DType::BF16) => {
            format!("as_type<bfloat>(ushort({:#010x}u & 0xffffu))", bits)
        }
        (ConstantValue::Bool(b), DType::Bool) => format!("{b}"),
        (ConstantValue::Int(_), DType::F32 | DType::F16 | DType::BF16)
        | (ConstantValue::Float { .. }, DType::I32 | DType::U32 | DType::Bool)
        | (
            ConstantValue::Bool(_),
            DType::I32 | DType::U32 | DType::F32 | DType::F16 | DType::BF16,
        ) => return Err(mismatch()),
    })
}

fn slot_write(value: &str, dtype: DType) -> String {
    match dtype {
        DType::F32 => format!("as_type<uint>({value})"),
        DType::F16 => format!("uint(as_type<ushort>(half({value})))"),
        DType::BF16 => format!("uint(as_type<ushort>(bfloat({value})))"),
        DType::I32 => format!("as_type<uint>({value})"),
        DType::U32 => format!("{value}"),
        DType::Bool => format!("uint({value})"),
    }
}

fn rel_symbol(op: seismic_realization::kernel::RelOp) -> &'static str {
    use seismic_realization::kernel::RelOp as R;
    match op {
        R::Eq => "==",
        R::Ne => "!=",
        R::Lt => "<",
        R::Le => "<=",
        R::Gt => ">",
        R::Ge => ">=",
    }
}

fn rel_of_binary(op: BinaryOp) -> Result<seismic_realization::kernel::RelOp, CompilerDefect> {
    use seismic_realization::kernel::RelOp as R;
    Ok(match op {
        BinaryOp::Eq => R::Eq,
        BinaryOp::Ne => R::Ne,
        BinaryOp::Lt => R::Lt,
        BinaryOp::Le => R::Le,
        BinaryOp::Gt => R::Gt,
        BinaryOp::Ge => R::Ge,
        BinaryOp::And
        | BinaryOp::Or
        | BinaryOp::Shl
        | BinaryOp::Shr
        | BinaryOp::BitAnd
        | BinaryOp::BitOr
        | BinaryOp::BitXor
        | BinaryOp::Add
        | BinaryOp::Sub
        | BinaryOp::Mul
        | BinaryOp::Div
        | BinaryOp::Rem => {
            return Err(defect(format!("{op:?} is not a comparison")))
        }
    })
}

fn comparison_cast(dtype: DType) -> &'static str {
    match dtype {
        DType::U32 => "uint",
        _ => "int",
    }
}

fn int_symbol(op: BinaryOp) -> Result<&'static str, CompilerDefect> {
    Ok(match op {
        BinaryOp::Add => "+",
        BinaryOp::Sub => "-",
        BinaryOp::Mul => "*",
        BinaryOp::BitAnd => "&",
        BinaryOp::BitOr => "|",
        BinaryOp::BitXor => "^",
        BinaryOp::Eq
        | BinaryOp::Ne
        | BinaryOp::Lt
        | BinaryOp::Le
        | BinaryOp::Gt
        | BinaryOp::Ge
        | BinaryOp::And
        | BinaryOp::Or
        | BinaryOp::Shl
        | BinaryOp::Shr
        | BinaryOp::Div
        | BinaryOp::Rem => return Err(defect(format!("{op:?} is not a bitwise integer op"))),
    })
}

fn float_symbol(op: BinaryOp) -> Result<&'static str, CompilerDefect> {
    Ok(match op {
        BinaryOp::Add => "+",
        BinaryOp::Sub => "-",
        BinaryOp::Mul => "*",
        BinaryOp::Eq
        | BinaryOp::Ne
        | BinaryOp::Lt
        | BinaryOp::Le
        | BinaryOp::Gt
        | BinaryOp::Ge
        | BinaryOp::And
        | BinaryOp::Or
        | BinaryOp::Shl
        | BinaryOp::Shr
        | BinaryOp::BitAnd
        | BinaryOp::BitOr
        | BinaryOp::BitXor
        | BinaryOp::Div
        | BinaryOp::Rem => return Err(defect(format!("{op:?} is not a float arithmetic op"))),
    })
}

fn math_call(op: MathOp, args: &[String], dtype: DType) -> String {
    use MathOp as M;
    let name = dtype_name(dtype);
    match op {
        M::Fma => format!("fma({}, {}, {})", args[0], args[1], args[2]),
        M::Exp => format!("exp({name}({}))", args[0]),
        M::ExpFast => format!("fast::exp({name}({}))", args[0]),
        M::Rsqrt => format!("rsqrt({name}({}))", args[0]),
        M::Sqrt => format!("sqrt({name}({}))", args[0]),
        M::Log => format!("log({name}({}))", args[0]),
        M::Sin => format!("sin({name}({}))", args[0]),
        M::Cos => format!("cos({name}({}))", args[0]),
        M::Abs => match dtype {
            d if d.is_int() => format!("abs({})", args[0]),
            _ => format!("fabs({name}({}))", args[0]),
        },
        M::Max => format!("max({}, {})", args[0], args[1]),
        M::Min => format!("min({}, {})", args[0], args[1]),
    }
}

fn cast_expr(from: DType, to: DType, operand: &str) -> String {
    if from.is_int() && to.is_int() {
        // Integer-to-integer casts preserve the low 32 bits.
        return format!("as_type<{}>(as_type<uint>({operand}))", dtype_name(to));
    }
    match to {
        DType::Bool => format!("{operand} != 0"),
        DType::I32 => format!(
            "int(clamp(trunc(float({operand})), -2147483648.0f, 2147483647.0f))"
        ),
        DType::U32 => format!(
            "uint(clamp(trunc(float({operand})), 0.0f, 4294967295.0f))"
        ),
        DType::F32 | DType::F16 | DType::BF16 => format!("{}({operand})", dtype_name(to)),
    }
}

/// Status codes mirror the safety-kind taxonomy (first error wins).
fn guard_code(predicate: &CheckPredicate) -> u32 {
    match predicate {
        CheckPredicate::IndexInBounds { .. } => 1,
        CheckPredicate::RangeInBounds { .. } => 2,
        CheckPredicate::DivisorNonZero { .. } => 3,
        CheckPredicate::SignedDivisionNoOverflow { .. } => 4,
        CheckPredicate::ShiftInRange { .. } => 5,
    }
}
