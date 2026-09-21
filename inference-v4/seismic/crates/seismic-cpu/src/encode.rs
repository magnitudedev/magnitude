//! The exhaustive mechanical CPU encoder: one `SealedLaunch<CpuDialect>` to
//! one encoded launch program, total over the sealed kernel algebra.
//!
//! Every `KernelOp<CpuIntrinsic>` has exactly one named arm; the CPU
//! intrinsic enum is uninhabited, so the `Intrinsic` arm is vacuous. The
//! encoder resolves kernel value/place references to binding positions over
//! the launch's own interface tables (inputs, outputs, locals, status
//! fields, pull counter) — dense indices over records the seal guarantees
//! (one launch binding per interface id, exact def/use equality, dense
//! status-field templates), never a fallible lookup, and never a
//! reconstructed dtype, shape, axis, representation, or operand: a
//! reference the seal did not define is unrepresentable, named as such.
//!
//! The output program is position-based: operands and results index the
//! binding list built here. The launch-local address-space split comes
//! from the sealed `storage_facts` and the view-side shapes from `view.ty`,
//! both carried in the encoded descriptor; the storage strides (resolved
//! layouts) stay plan-side and are folded mechanically at assembly, where
//! the physical plan is available.

use crate::intrinsics::{CpuDialect, CpuIntrinsic};
use seismic_lang::{
    intrinsics::{AtomicOp, MathOp, PlaneField, ReduceOp, ReduceSchema},
    logical::IdIndex,
    syntax::ast::{BinaryOp, UnaryOp},
    types::{DType, ExtentExpr, NonEmpty, RuntimeExtentId, TensorType},
};
use seismic_realization::{
    dispatch::Traversal,
    ids::{
        CanonicalLeafId, InvocationValueId, ResultFieldIx, ScalarSlot, ScalarSlotIx,
        StatusFieldIx, StatusFieldTemplateId, StorageIx,
    },
    kernel::{
        AtomicMode, CheckPredicate, CoreKernelOp, KernelJoin, KernelOp, KernelPlaceRef,
        KernelValueRef, RelOp, TypedConstant,
    },
    physical::{
        LaunchBinding, LaunchInput, LaunchOutput, PhysicalStorageView, ScalarSource,
        SealedLaunch, StorageFact,
    },
};
use std::collections::BTreeMap;

/// One encoded launch: the CPU program plus the native ABI tables assembly
/// compiles it against. Construction is private to this module's `encode`.
pub struct EncodedLaunch {
    pub(crate) bindings: Vec<Binding>,
    pub(crate) instructions: Vec<Instruction>,
    /// Binding position -> iteration axis ordinal (an `Axis` value read).
    pub(crate) axis_of: BTreeMap<usize, usize>,
    /// Canonical transform-endpoint leaf -> scalar input binding position.
    pub(crate) endpoint_of: BTreeMap<CanonicalLeafId, usize>,
    pub(crate) descriptor: LaunchDescriptor,
}

/// How one scalar word of the launch's native scalar table is sourced (and,
/// for executor slots, written back) by the executor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum WordSource {
    Abi(ScalarSlot),
    Executor(ScalarSlotIx),
    Invocation(InvocationValueId),
    Result(ResultFieldIx),
}

/// One binding position: a scalar word, a tensor place (one view per plane),
/// or a kernel-local SSA value.
#[derive(Clone, Debug)]
pub(crate) enum Binding {
    Word { index: usize, dtype: DType },
    Tensor { views: NonEmpty<PhysicalStorageView> },
    Local,
}

/// The native ABI of one encoded launch. Word layout:
/// `[words][runtime extent values][work items][participants]`. The storage
/// split is launch-local, from the sealed `storage_facts` in binding-slot
/// order: `Global` (and zero-byte workgroup) storages form the shared
/// buffer table in that order; `Participant` storages become per-worker
/// scratch spans (offsets folded here, bytes and alignment from the facts).
#[derive(Clone, Debug)]
pub(crate) struct LaunchDescriptor {
    /// Shared storages in buffer-table order (binding-slot order).
    pub table_order: Vec<StorageIx>,
    /// Participant-scoped storage -> scratch byte offset.
    pub participant_spans: BTreeMap<StorageIx, u64>,
    /// Total per-participant scratch bytes.
    pub scratch_bytes: u64,
    pub has_status: bool,
    pub words: Vec<WordSource>,
    pub runtime_extents: Vec<RuntimeExtentId>,
    /// `GridStride`/`OnePass` (a stride loop) or `DynamicPull` (a worker
    /// claim loop over the pull counter).
    pub pull_loop: bool,
    /// The launch's pull-counter storage (`pull_loop` only).
    pub pull_counter: Option<StorageIx>,
    /// The independent-axis extents, outermost first, for delinearization.
    pub work_axes: Vec<ExtentExpr>,
    pub work_index: usize,
    pub participants_index: usize,
}

/// One check predicate with operand positions resolved.
#[derive(Clone, Debug)]
pub(crate) enum CheckKind {
    IndexInBounds { index: usize, extent: ExtentExpr },
    RangeInBounds { start: usize, end: usize, extent: ExtentExpr },
    DivisorNonZero { value: usize },
    SignedDivisionNoOverflow { lhs: usize, rhs: usize },
    ShiftInRange { value: usize },
}

/// One ordered scalar carry of a `Repeat`, at the head of its body.
#[derive(Clone, Copy, Debug)]
pub(crate) struct CarrySlot {
    pub initial: usize,
    pub current: usize,
    pub update: usize,
    pub result: usize,
}

/// One branch join with positions resolved.
#[derive(Clone, Copy, Debug)]
pub(crate) struct JoinSlot {
    pub then_value: usize,
    pub else_value: usize,
    pub joined: usize,
}

/// The encoded CPU instruction program. Operand and result fields are
/// binding positions; nested control bodies nest their instructions.
#[derive(Clone, Debug)]
pub(crate) enum Instruction {
    Const { into: usize, value: TypedConstant },
    Extent { into: usize, extent: RuntimeExtentId },
    Unary { into: usize, op: UnaryOp, operand: usize, dtype: DType },
    Binary { into: usize, op: BinaryOp, left: usize, right: usize, dtype: DType },
    Compare { into: usize, op: RelOp, left: usize, right: usize, dtype: DType },
    Math { into: usize, op: MathOp, operands: Vec<usize>, dtype: DType },
    Fma { into: usize, a: usize, b: usize, c: usize, dtype: DType },
    Cast { into: usize, operand: usize, from: DType, to: DType },
    Select { into: usize, condition: usize, then_value: usize, else_value: usize },
    TableLookup { into: usize, index: usize, table: Vec<i32> },
    Load { into: usize, place: usize, coords: Vec<usize> },
    Store { place: usize, coords: Vec<usize>, value: usize },
    PackedPlaneRead {
        into: usize,
        place: usize,
        coords: Vec<usize>,
        repr: String,
        plane: PlaneField,
        entry: u32,
    },
    PlaneLoad { into: usize, place: usize, coords: Vec<usize>, repr: String, plane: PlaneField },
    PlaneStore { place: usize, coords: Vec<usize>, repr: String, plane: PlaneField, value: usize },
    Atomic { place: usize, coords: Vec<usize>, value: usize, op: AtomicOp, dtype: DType, mode: AtomicMode },
    Repeat { binder: usize, start: usize, end: usize, carries: Vec<CarrySlot>, body: Vec<Instruction> },
    Branch { condition: usize, then_body: Vec<Instruction>, else_body: Vec<Instruction>, joins: Vec<JoinSlot> },
    Fold {
        into: usize,
        place: usize,
        axis: u32,
        coords: Vec<usize>,
        op: ReduceOp,
        schema: ReduceSchema,
    },
    Check { kind: CheckKind, status: StatusFieldIx, guarded: Vec<Instruction> },
    /// Write one kernel scalar to its scalar-routed output binding.
    Publish { value: usize, destination: usize },
    /// A participant-wide ordering fence (`Subgroup`/`Workgroup` scope are
    /// the same flat participant pool on this target).
    Barrier,
    /// Grid-wide synchronization of a cooperative launch: no CPU proposal
    /// can produce it (`TargetLimits::cooperative_grid` is `None`); assembly
    /// reports the contradicted invariant.
    GridBarrier,
}

/// Encode one sealed launch. Total over the sealed kernel algebra.
pub fn encode(launch: &SealedLaunch<CpuDialect>) -> EncodedLaunch {
    Encoder::run(launch)
}

struct Encoder<'a> {
    launch: &'a SealedLaunch<CpuDialect>,
    bindings: Vec<Binding>,
    value_of: BTreeMap<KernelValueRef, usize>,
    place_of: BTreeMap<KernelPlaceRef, usize>,
    /// Scalar input position by canonical leaf (transform endpoints).
    leaf_of: BTreeMap<CanonicalLeafId, usize>,
    axis_of: BTreeMap<usize, usize>,
    words: Vec<WordSource>,
    word_of_source: BTreeMap<WordSource, usize>,
    runtime_extents: Vec<RuntimeExtentId>,
    extent_index: BTreeMap<RuntimeExtentId, usize>,
}

impl Encoder<'_> {
    fn run(launch: &SealedLaunch<CpuDialect>) -> EncodedLaunch {
        let interface = launch.kernel.interface();
        let mut encoder = Encoder {
            launch,
            bindings: Vec::new(),
            value_of: BTreeMap::new(),
            place_of: BTreeMap::new(),
            leaf_of: BTreeMap::new(),
            axis_of: BTreeMap::new(),
            words: Vec::new(),
            word_of_source: BTreeMap::new(),
            runtime_extents: Vec::new(),
            extent_index: BTreeMap::new(),
        };
        encoder.bind_interface();
        // Every runtime extent the program can read at execution: transform
        // source shapes, the iteration axes, and the op stream.
        encoder.register_binding_extents();
        for extent in &interface.iteration.extents {
            encoder.register_extent_expr(extent);
        }
        for op in launch.kernel.ops().iter() {
            encoder.register_op(op);
        }
        let instructions = launch
            .kernel
            .ops()
            .iter()
            .map(|op| encoder.translate(op))
            .collect();
        // The launch-local storage split: one fact per binding, in
        // binding-slot order. Shared (`Global`, and workgroup storages the
        // target declares absent) form the buffer table; `Participant`
        // storages become per-worker scratch spans.
        // One fact per binding, in binding-slot order (P1 seal).
        if launch.storage_facts.len() != launch.bindings.len() {
            unreachable!("a launch's storage facts do not cover its bindings (P1)")
        }
        let mut ordered: Vec<(&LaunchBinding, &StorageFact)> = launch
            .bindings
            .iter()
            .zip(launch.storage_facts.iter())
            .collect();
        ordered.sort_by_key(|(binding, _)| binding.slot);
        let mut table_order: Vec<StorageIx> = Vec::with_capacity(ordered.len());
        let mut participant_spans: BTreeMap<StorageIx, u64> = BTreeMap::new();
        let mut scratch_bytes = 0u64;
        for (binding, fact) in ordered {
            match fact {
                StorageFact::Global => table_order.push(binding.storage),
                // Workgroup-scope storage does not exist on this target
                // (`TargetLimits::max_workgroup_bytes` is 0); a zero-byte
                // binding keeps a null table entry, a nonzero one is
                // rejected against the selected contract at assembly.
                StorageFact::Workgroup { .. } => table_order.push(binding.storage),
                StorageFact::Participant { bytes, alignment } => {
                    let alignment = (*alignment).max(1);
                    let offset = scratch_bytes.div_ceil(alignment) * alignment;
                    scratch_bytes = offset + *bytes;
                    participant_spans.insert(binding.storage, offset);
                }
            }
        }
        let has_status = !launch.status_fields.is_empty();
        let work_index = encoder.words.len() + encoder.runtime_extents.len();
        let participants_index = work_index + 1;
        let pull_loop = matches!(
            interface.iteration.traversal,
            Traversal::DynamicPull { .. }
        );
        let descriptor = LaunchDescriptor {
            table_order,
            participant_spans,
            scratch_bytes,
            has_status,
            words: encoder.words.clone(),
            runtime_extents: encoder.runtime_extents.clone(),
            pull_loop,
            pull_counter: launch.pull_counter,
            work_axes: interface.iteration.extents.clone(),
            work_index,
            participants_index,
        };
        EncodedLaunch {
            bindings: encoder.bindings,
            instructions,
            axis_of: encoder.axis_of,
            endpoint_of: encoder.leaf_of,
            descriptor,
        }
    }

    // -- binding classification --------------------------------------------

    fn bind_interface(&mut self) {
        let interface = self.launch.kernel.interface();
        for decl in interface.inputs.iter() {
            let input = &self.launch.inputs[decl.id.index()];
            let position = self.bindings.len();
            match input {
                LaunchInput::Storage(views) => {
                    self.bindings.push(Binding::Tensor { views: views.clone() });
                    self.place_of.insert(KernelPlaceRef::Input(decl.id), position);
                }
                LaunchInput::Scalar { source, dtype, .. } => {
                    let index = self.word(source);
                    self.bindings.push(Binding::Word { index, dtype: *dtype });
                    self.value_of.insert(KernelValueRef::Input(decl.id), position);
                    self.leaf_of.insert(decl.leaf, position);
                }
            }
        }
        for decl in interface.outputs.iter() {
            let output = &self.launch.outputs[decl.id.index()];
            let position = self.bindings.len();
            match output {
                LaunchOutput::Storage(views) => {
                    self.bindings.push(Binding::Tensor { views: views.clone() });
                }
                LaunchOutput::ExecutorSlot { slot, dtype } => {
                    let index = self.word(&ScalarSource::Executor(*slot));
                    self.bindings.push(Binding::Word { index, dtype: *dtype });
                }
                LaunchOutput::ResultField { field, dtype } => {
                    let index = self.word(&ScalarSource::Result(*field));
                    self.bindings.push(Binding::Word { index, dtype: *dtype });
                }
            }
            self.place_of.insert(KernelPlaceRef::Output(decl.id), position);
        }
        for decl in interface.locals.iter() {
            let views = &self.launch.locals[decl.id.index()];
            let position = self.bindings.len();
            self.bindings.push(Binding::Tensor { views: views.clone() });
            self.place_of.insert(KernelPlaceRef::Local(decl.id), position);
        }
        for decl in self.launch.kernel.ssa().iter() {
            let position = self.bindings.len();
            self.bindings.push(Binding::Local);
            self.value_of.insert(KernelValueRef::Ssa(decl.id), position);
        }
        for decl in interface.axes.iter() {
            let position = self.bindings.len();
            self.bindings.push(Binding::Local);
            self.value_of.insert(KernelValueRef::Axis(decl.id), position);
            self.axis_of.insert(position, decl.id.index());
        }
    }

    /// The word index of one scalar source, registering it on demand.
    fn word(&mut self, source: &ScalarSource) -> usize {
        let key = match source {
            ScalarSource::Abi(slot) => WordSource::Abi(*slot),
            ScalarSource::Executor(slot) => WordSource::Executor(*slot),
            ScalarSource::Invocation(id) => WordSource::Invocation(*id),
            ScalarSource::Result(field) => WordSource::Result(*field),
        };
        if let Some(index) = self.word_of_source.get(&key) {
            return *index;
        }
        let index = self.words.len();
        self.words.push(key);
        self.word_of_source.insert(key, index);
        index
    }

    // -- extent registration --------------------------------------------------

    fn register_extent_expr(&mut self, extent: &ExtentExpr) {
        if let ExtentExpr::Runtime(id) = extent {
            if !self.extent_index.contains_key(id) {
                self.extent_index.insert(*id, self.runtime_extents.len());
                self.runtime_extents.push(*id);
            }
        }
    }

    fn register_shape(&mut self, shape: &TensorType) {
        for axis in &shape.axes {
            self.register_extent_expr(axis);
        }
    }

    fn register_binding_extents(&mut self) {
        // Collect every extent expression first (immutable pass), then
        // register them (mutable pass).
        let mut extents: Vec<ExtentExpr> = Vec::new();
        for binding in &self.bindings {
            if let Binding::Tensor { views } = binding {
                for view in views.iter() {
                    for step in &view.transform.steps {
                        for axis in &step.source_shape {
                            extents.push(axis.clone());
                        }
                    }
                    // The view-side shape (a trailing reshape's result shape)
                    // is evaluated at emission from `view.ty`.
                    for axis in &view.ty.axes {
                        extents.push(axis.clone());
                    }
                }
            }
        }
        for extent in &extents {
            self.register_extent_expr(extent);
        }
    }

    fn register_op(&mut self, op: &KernelOp<CpuIntrinsic>) {
        match op {
            KernelOp::Core(core) => self.register_core(core),
            // The CPU intrinsic enum is uninhabited.
            KernelOp::Intrinsic(intrinsic) => match *intrinsic {},
        }
    }

    fn register_core(&mut self, core: &CoreKernelOp<CpuIntrinsic>) {
        match core {
            CoreKernelOp::RuntimeExtent { extent, .. } => {
                let id = *extent;
                if !self.extent_index.contains_key(&id) {
                    self.extent_index.insert(id, self.runtime_extents.len());
                    self.runtime_extents.push(id);
                }
            }
            CoreKernelOp::Fold { shape, .. } => self.register_shape(shape),
            CoreKernelOp::Check { predicate, guarded, .. } => {
                match predicate {
                    CheckPredicate::IndexInBounds { extent, .. }
                    | CheckPredicate::RangeInBounds { extent, .. } => {
                        self.register_extent_expr(extent)
                    }
                    CheckPredicate::DivisorNonZero { .. }
                    | CheckPredicate::SignedDivisionNoOverflow { .. }
                    | CheckPredicate::ShiftInRange { .. } => {}
                }
                for nested in guarded {
                    self.register_op(nested);
                }
            }
            CoreKernelOp::Repeat { body, .. } => {
                for nested in body {
                    self.register_op(nested);
                }
            }
            CoreKernelOp::Branch { then_body, else_body, .. } => {
                for nested in then_body.iter().chain(else_body) {
                    self.register_op(nested);
                }
            }
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
            | CoreKernelOp::Carry { .. }
            | CoreKernelOp::Publish { .. }
            | CoreKernelOp::Barrier { .. }
            | CoreKernelOp::GridBarrier => {}
        }
    }

    // -- reference resolution ---------------------------------------------------

    fn value_pos(&self, value: KernelValueRef) -> usize {
        self.value_of[&value]
    }

    fn place_pos(&self, place: KernelPlaceRef) -> usize {
        self.place_of[&place]
    }

    fn status_ix(&self, template: StatusFieldTemplateId) -> StatusFieldIx {
        // Template ids are dense per block in kernel order, and the launch
        // lists the sealed fields in that order (K1/P1).
        self.launch.status_fields[template.0 as usize]
    }

    // -- the exhaustive translation ----------------------------------------------

    fn translate(&mut self, op: &KernelOp<CpuIntrinsic>) -> Instruction {
        match op {
            KernelOp::Core(core) => self.translate_core(core),
            KernelOp::Intrinsic(intrinsic) => match *intrinsic {},
        }
    }

    fn translate_core(&mut self, core: &CoreKernelOp<CpuIntrinsic>) -> Instruction {
        match core {
            CoreKernelOp::Const { into, value } => Instruction::Const {
                into: self.value_pos(KernelValueRef::Ssa(*into)),
                value: *value,
            },
            CoreKernelOp::RuntimeExtent { into, extent } => Instruction::Extent {
                into: self.value_pos(KernelValueRef::Ssa(*into)),
                extent: *extent,
            },
            CoreKernelOp::Unary { into, op, operand, dtype } => Instruction::Unary {
                into: self.value_pos(KernelValueRef::Ssa(*into)),
                op: *op,
                operand: self.value_pos(*operand),
                dtype: *dtype,
            },
            CoreKernelOp::Binary { into, op, left, right, dtype } => Instruction::Binary {
                into: self.value_pos(KernelValueRef::Ssa(*into)),
                op: *op,
                left: self.value_pos(*left),
                right: self.value_pos(*right),
                dtype: *dtype,
            },
            CoreKernelOp::Compare { into, op, left, right, dtype } => Instruction::Compare {
                into: self.value_pos(KernelValueRef::Ssa(*into)),
                op: *op,
                left: self.value_pos(*left),
                right: self.value_pos(*right),
                dtype: *dtype,
            },
            CoreKernelOp::Math { into, op, operands, dtype } => Instruction::Math {
                into: self.value_pos(KernelValueRef::Ssa(*into)),
                op: *op,
                operands: operands.iter().map(|o| self.value_pos(*o)).collect(),
                dtype: *dtype,
            },
            CoreKernelOp::Fma { into, a, b, c, dtype } => Instruction::Fma {
                into: self.value_pos(KernelValueRef::Ssa(*into)),
                a: self.value_pos(*a),
                b: self.value_pos(*b),
                c: self.value_pos(*c),
                dtype: *dtype,
            },
            CoreKernelOp::Cast { into, operand, from, to } => Instruction::Cast {
                into: self.value_pos(KernelValueRef::Ssa(*into)),
                operand: self.value_pos(*operand),
                from: *from,
                to: *to,
            },
            CoreKernelOp::Select { into, condition, then_value, else_value, .. } => {
                Instruction::Select {
                    into: self.value_pos(KernelValueRef::Ssa(*into)),
                    condition: self.value_pos(*condition),
                    then_value: self.value_pos(*then_value),
                    else_value: self.value_pos(*else_value),
                }
            }
            CoreKernelOp::TableLookup { into, index, table } => Instruction::TableLookup {
                into: self.value_pos(KernelValueRef::Ssa(*into)),
                index: self.value_pos(*index),
                table: table.clone(),
            },
            CoreKernelOp::Load { into, place, coords, .. } => Instruction::Load {
                into: self.value_pos(KernelValueRef::Ssa(*into)),
                place: self.place_pos(*place),
                coords: coords.iter().map(|c| self.value_pos(*c)).collect(),
            },
            CoreKernelOp::Store { place, coords, value, .. } => Instruction::Store {
                place: self.place_pos(*place),
                coords: coords.iter().map(|c| self.value_pos(*c)).collect(),
                value: self.value_pos(*value),
            },
            CoreKernelOp::PackedPlaneRead { into, place, coords, repr, plane, entry, .. } => {
                Instruction::PackedPlaneRead {
                    into: self.value_pos(KernelValueRef::Ssa(*into)),
                    place: self.place_pos(*place),
                    coords: coords.iter().map(|c| self.value_pos(*c)).collect(),
                    repr: repr.clone(),
                    plane: *plane,
                    entry: *entry,
                }
            }
            CoreKernelOp::PlaneLoad { into, place, coords, repr, plane, .. } => {
                Instruction::PlaneLoad {
                    into: self.value_pos(KernelValueRef::Ssa(*into)),
                    place: self.place_pos(*place),
                    coords: coords.iter().map(|c| self.value_pos(*c)).collect(),
                    repr: repr.clone(),
                    plane: *plane,
                }
            }
            CoreKernelOp::PlaneStore { place, coords, repr, plane, value, .. } => {
                Instruction::PlaneStore {
                    place: self.place_pos(*place),
                    coords: coords.iter().map(|c| self.value_pos(*c)).collect(),
                    repr: repr.clone(),
                    plane: *plane,
                    value: self.value_pos(*value),
                }
            }
            CoreKernelOp::Atomic { place, coords, op, value, dtype, mode } => Instruction::Atomic {
                place: self.place_pos(*place),
                coords: coords.iter().map(|c| self.value_pos(*c)).collect(),
                value: self.value_pos(*value),
                op: *op,
                dtype: *dtype,
                mode: *mode,
            },
            CoreKernelOp::Repeat { binder, start, end, body } => {
                // `Carry` ops are listed at the head of the body; the rest
                // translates as the loop body.
                let mut carries = Vec::new();
                let mut rest = body.iter();
                loop {
                    match rest.next() {
                        Some(KernelOp::Core(CoreKernelOp::Carry {
                            initial,
                            current,
                            update,
                            result,
                            ..
                        })) => carries.push(CarrySlot {
                            initial: self.value_pos(*initial),
                            current: self.value_pos(KernelValueRef::Ssa(*current)),
                            update: self.value_pos(*update),
                            result: self.value_pos(KernelValueRef::Ssa(*result)),
                        }),
                        Some(op) => {
                            let mut body = vec![self.translate(op)];
                            body.extend(rest.map(|op| self.translate(op)));
                            return Instruction::Repeat {
                                binder: self.value_pos(KernelValueRef::Ssa(*binder)),
                                start: self.value_pos(*start),
                                end: self.value_pos(*end),
                                carries,
                                body,
                            };
                        }
                        None => {
                            return Instruction::Repeat {
                                binder: self.value_pos(KernelValueRef::Ssa(*binder)),
                                start: self.value_pos(*start),
                                end: self.value_pos(*end),
                                carries,
                                body: Vec::new(),
                            };
                        }
                    }
                }
            }
            CoreKernelOp::Carry { .. } => unreachable!(
                "an ordered carry appears outside the head of its repeat body (K1)"
            ),
            CoreKernelOp::Branch { condition, then_body, else_body, joins } => {
                Instruction::Branch {
                    condition: self.value_pos(*condition),
                    then_body: then_body.iter().map(|op| self.translate(op)).collect(),
                    else_body: else_body.iter().map(|op| self.translate(op)).collect(),
                    joins: joins
                        .iter()
                        .map(|join: &KernelJoin| JoinSlot {
                            then_value: self.value_pos(join.then_value),
                            else_value: self.value_pos(join.else_value),
                            joined: self.value_pos(KernelValueRef::Ssa(join.joined)),
                        })
                        .collect(),
                }
            }
            CoreKernelOp::Fold { into, op, place, axis, coords, schema, .. } => {
                // The operand place's view carries the view-side tensor
                // type (`view.ty`), including any trailing reshape, so the
                // fold reads its rank and reduced-axis extent from there.
                Instruction::Fold {
                    into: self.value_pos(KernelValueRef::Ssa(*into)),
                    place: self.place_pos(*place),
                    axis: *axis,
                    coords: coords.iter().map(|c| self.value_pos(*c)).collect(),
                    op: *op,
                    schema: *schema,
                }
            }
            CoreKernelOp::Check { predicate, status, guarded, .. } => Instruction::Check {
                kind: match predicate {
                    CheckPredicate::IndexInBounds { index, extent } => CheckKind::IndexInBounds {
                        index: self.value_pos(*index),
                        extent: extent.clone(),
                    },
                    CheckPredicate::RangeInBounds { start, end, extent } => {
                        CheckKind::RangeInBounds {
                            start: self.value_pos(*start),
                            end: self.value_pos(*end),
                            extent: extent.clone(),
                        }
                    }
                    CheckPredicate::DivisorNonZero { value, .. } => CheckKind::DivisorNonZero {
                        value: self.value_pos(*value),
                    },
                    CheckPredicate::SignedDivisionNoOverflow { lhs, rhs } => {
                        CheckKind::SignedDivisionNoOverflow {
                            lhs: self.value_pos(*lhs),
                            rhs: self.value_pos(*rhs),
                        }
                    }
                    CheckPredicate::ShiftInRange { value } => CheckKind::ShiftInRange {
                        value: self.value_pos(*value),
                    },
                },
                status: self.status_ix(*status),
                guarded: guarded.iter().map(|op| self.translate(op)).collect(),
            },
            CoreKernelOp::Publish { value, output } => Instruction::Publish {
                value: self.value_pos(*value),
                destination: self.place_pos(KernelPlaceRef::Output(*output)),
            },
            CoreKernelOp::Barrier { scope } => {
                // One flat participant pool: both scopes order the same set.
                let _ = scope;
                Instruction::Barrier
            }
            CoreKernelOp::GridBarrier => Instruction::GridBarrier,
        }
    }
}
