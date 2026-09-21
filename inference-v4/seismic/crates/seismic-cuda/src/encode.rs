//! Exhaustive mechanical PTX emission over the sealed kernel operation
//! algebra (`KernelOp<CudaIntrinsic>`), one encoded launch per
//! `SealedLaunch<Dialect>`.
//!
//! The encoder is total: it makes no allocation, geometry, synchronization,
//! algorithm, or precision decision, never rejects a sealed operation, and
//! adds or omits no check — guards come only from `Check` operations sealed
//! by K1. It may introduce instructions and native temporaries (kernel
//! parameters, the grid-barrier scratch, dynamic shared/local
//! declarations); it never introduces a semantic array or scratch absent
//! from the sealed block.
//!
//! All scalar floating values live in `.f32` registers: narrow (f16/bf16)
//! loads widen on read and stores round on write, which is exactly the
//! registry's load/add/round/store contract. The exact transcendental
//! reference is the versioned `seismic_math` software sequence (`exp`,
//! `log`, `sin`, `cos` calls; `sqrt`/`rsqrt` as one correctly rounded f64
//! step); `exp_fast` is the registry's approximate-form operator and emits
//! the `ex2.approx` sequence.
//!
//! Geometry is retained, not decided: the encoded launch carries the sealed
//! `work_items` / `participants` / `workgroups` execution expressions and
//! the PTX reads `%ntid`/`%nctaid` dynamically. A serialized traversal
//! (one participant) loops over `[0, total)` internally; a grid-stride
//! traversal masks its tail; a dynamic-pull traversal claims coordinates
//! from the launch's pull-counter storage with
//! `while ((lin = atom.global.add.u32 counter, 1) < total)`.
//!
//! `GridBarrier` emits a grid-wide software barrier over native scratch
//! (two u32 words per site: an arrive counter and a release flag); it is
//! sealed only into `GridCooperative` blocks, whose launches are submitted
//! cooperatively.
//!
//! # Sealed launch facts consumed here (landed)
//!
//! - `storage_facts`: one `StorageFact` per `bindings` entry, in
//!   binding-slot order — global (pointer binding), workgroup, or
//!   participant kernel-local, the latter two with the byte size and
//!   alignment the native declaration needs.
//! - `PhysicalStorageView.ty`: the view-side tensor type the encoder
//!   addresses (the routed value's type, including any trailing reshape).
//!   Address linearization takes its extents per binding; the transform
//!   maps `ty`-coordinates to residence coordinates.
//! - Serialized-policy launches seal `workgroups = [1,1,1]` with
//!   `participants = 1`; the encoder emits the single participant's
//!   internal ascending `[0, total)` loop and the executor submits the
//!   sealed geometry uniformly.

use crate::intrinsics::{CudaIntrinsic, Dialect};
use seismic_lang::{
    intrinsics::{AtomicOp, MathOp, PlaneField, ReduceOp},
    logical::IdIndex,
    repr,
    syntax::ast::{BinaryOp, UnaryOp},
    types::{DType, ExtentExpr, NonEmpty, RuntimeExtentId, TensorType},
};
use seismic_realization::dispatch::Traversal;
use seismic_realization::ids::{
    CanonicalLeafId, KernelAxisId, KernelInputId, KernelLocalId, KernelOutputId, KernelSsaId,
    LaunchIx, StatusFieldTemplateId, StorageIx,
};
use seismic_realization::kernel::{
    AtomicMode, BarrierScope, CheckPredicate, ConstantValue, CoreKernelOp, KernelJoin, KernelOp,
    KernelPlaceRef, KernelValueRef, KernelValueType, RelOp, TypedConstant,
};
use seismic_realization::physical::{
    ExecutionExpr, LaunchInput, LaunchOutput, PhysicalStorageView, ScalarSource, SealedLaunch,
};
use seismic_realization::routes::{SliceAxisTemplate, ViewStepKind, ViewStepTemplate};
use std::collections::BTreeMap;

/// The CUDA kernel parameter ABI limit (bytes).
pub const MAX_KERNEL_PARAMETER_BYTES: usize = 32_764;

/// One emitted kernel value: a typed scalar register (narrow floats live
/// widened in `.f32`) or a u64 index register.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Scalar { register: String, dtype: DType },
    Index(String),
}

/// One (possibly runtime-computed) stride factor.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Stride {
    Static(u64),
    /// A product of trailing axis extents.
    Runtime(Vec<ExtentExpr>),
}

/// The facts of one addressable place: its storage views and its routed
/// value shape in view coordinates.
#[derive(Clone, Debug)]
struct PlaceInfo {
    views: NonEmpty<PhysicalStorageView>,
    shape: Vec<ExtentExpr>,
}

/// One pending ordered-carry update — the update source and the current
/// register — flushed at the enclosing `Repeat`'s visit tail.
struct PendingCarry {
    update: KernelValueRef,
    register: Value,
}

/// One encoded kernel parameter of the launch ABI. Every parameter rides a
/// 64-bit word; by-value scalars are converted to their typed register
/// representation in the prologue.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CudaParam {
    /// Device pointer of one bound storage (a caller buffer or an arena
    /// offset; the executor resolves it from the plan's storage table).
    Storage(StorageIx),
    /// One by-value kernel scalar input, at its scalar-binding ordinal.
    Scalar {
        slot: u32,
        source: ScalarSource,
        dtype: DType,
    },
    /// The exact runtime work-item total of the traversal (present when the
    /// iteration domain is not static).
    WorkTotal,
    /// Base pointer of the launch's grid-barrier scratch: `2 * grid_barriers`
    /// u32 words, two per barrier site.
    GridBarrier,
    /// Pointer to the runtime-extent value block: one u64 per
    /// `RuntimeExtentId` the kernel reads.
    RuntimeExtents,
    /// Pointer to the executor scalar slot block: one u64 per slot.
    ExecutorSlots,
    /// Pointer to the status block: one u32 per status field.
    Status,
    /// Pointer to the compiler-owned result scalar block: one u64 per field.
    Results,
}

/// One encoded launch: the mechanical parameter list, the retained geometry
/// expressions, and the PTX text.
#[derive(Clone, Debug, PartialEq)]
pub struct CudaLaunch {
    pub id: LaunchIx,
    pub name: String,
    pub params: Vec<CudaParam>,
    /// The exact runtime work-item total; evaluated at submission.
    pub work_items: ExecutionExpr,
    /// The solved participant count (one native workgroup's width).
    pub participants: ExecutionExpr,
    /// The per-axis workgroup grid, derived by the seal.
    pub workgroups: [ExecutionExpr; 3],
    /// The traversal is serialized: one participant visits `[0, total)`
    /// ascending (the launch is one workgroup of one participant).
    pub serialized: bool,
    /// The launch is a cooperative grid (`GridCooperative` policy): submit
    /// through `cuLaunchCooperativeKernel`.
    pub cooperative: bool,
    /// Grid-barrier sites in the PTX (native scratch words: two per site).
    pub grid_barriers: usize,
    /// Every runtime extent this kernel reads (the value-block domain).
    pub runtime_extents: Vec<RuntimeExtentId>,
    /// The pull-counter storage of a `DynamicPull` traversal.
    pub pull_counter: Option<StorageIx>,
    pub ptx: String,
}

/// Encode one sealed launch into its mechanical CUDA artifact. Total over
/// the sealed algebra; every case is named.
pub fn encode(launch: &SealedLaunch<Dialect>) -> CudaLaunch {
    Ptx::new(launch).emit()
}

fn bug(message: impl Into<String>) -> ! {
    panic!("compiler defect (B1Cuda): {}", message.into())
}

fn is_cooperative(launch: &SealedLaunch<Dialect>) -> bool {
    // A `MaxResidentParticipants` declaration is present exactly when the
    // sealed block's participant policy is `GridCooperative` (the seal adds
    // no other source of the fact).
    launch.native_facts.iter().any(|fact| {
        matches!(
            fact.kind,
            seismic_realization::strategy::NativeFactKind::MaxResidentParticipants
        )
    })
}

/// The per-launch PTX emitter.
struct Ptx<'a> {
    launch: &'a SealedLaunch<Dialect>,
    params: Vec<CudaParam>,
    storage_registers: BTreeMap<usize, String>,
    scalar_registers: BTreeMap<u32, String>,
    extent_registers: BTreeMap<u32, String>,
    work_total_param: Option<String>,
    extents_param: Option<String>,
    slots_param: Option<String>,
    status_param: Option<String>,
    results_param: Option<String>,
    barrier_param: Option<String>,
    pull_counter_register: Option<String>,
    r16: usize,
    r32: usize,
    r64: usize,
    f32: usize,
    f64: usize,
    pred: usize,
    labels: usize,
    body: Vec<String>,
    ssa: BTreeMap<u32, Value>,
    bound: BTreeMap<KernelValueRef, Value>,
    places: BTreeMap<KernelPlaceRef, PlaceInfo>,
    endpoint_inputs: BTreeMap<CanonicalLeafId, KernelValueRef>,
    axis_registers: Vec<String>,
    traversal: Traversal,
    serialized: bool,
    tail_mask: bool,
    total_static: Option<u64>,
    linear: Option<String>,
    total: Option<String>,
    stride: Option<String>,
    loop_label: Option<String>,
    done_label: Option<String>,
    pull_labels: Option<(String, String)>,
    grid_barriers: usize,
    runtime_extents: Vec<RuntimeExtentId>,
    math_calls: Vec<(String, String, String, String)>,
    exact_math: bool,
    fold_accumulator: Option<String>,
    fold_tracked_index: Option<String>,
    local_decls: Vec<String>,
    /// The storage fact of every bound storage, keyed by storage index
    /// (filled from `storage_facts`, which is parallel to `bindings` in
    /// slot order).
    binding_facts: BTreeMap<usize, seismic_realization::physical::StorageFact>,
    shared_decls: BTreeMap<usize, (String, u64, u64)>,
    shared_offset: u64,
    private_decls: BTreeMap<usize, String>,
    carry_updates: Vec<PendingCarry>,
    /// `(result, current)` of every carry defined inside an open `Repeat`:
    /// the result is bound to the current register after the loop.
    carry_definitions: Vec<(KernelSsaId, KernelSsaId)>,
}

impl<'a> Ptx<'a> {
    fn new(launch: &'a SealedLaunch<Dialect>) -> Self {
        let iteration = &launch.kernel.interface().iteration;
        Ptx {
            launch,
            params: Vec::new(),
            storage_registers: BTreeMap::new(),
            scalar_registers: BTreeMap::new(),
            extent_registers: BTreeMap::new(),
            work_total_param: None,
            extents_param: None,
            slots_param: None,
            status_param: None,
            results_param: None,
            barrier_param: None,
            pull_counter_register: None,
            r16: 0,
            r32: 0,
            r64: 0,
            f32: 0,
            f64: 0,
            pred: 0,
            labels: 0,
            body: Vec::new(),
            ssa: BTreeMap::new(),
            bound: BTreeMap::new(),
            places: BTreeMap::new(),
            endpoint_inputs: BTreeMap::new(),
            axis_registers: Vec::new(),
            traversal: iteration.traversal.clone(),
            serialized: iteration.serialized,
            tail_mask: iteration.tail_mask,
            total_static: iteration.total.as_static(),
            linear: None,
            total: None,
            stride: None,
            loop_label: None,
            done_label: None,
            pull_labels: None,
            grid_barriers: 0,
            runtime_extents: Vec::new(),
            math_calls: Vec::new(),
            exact_math: false,
            fold_accumulator: None,
            fold_tracked_index: None,
            local_decls: Vec::new(),
            binding_facts: BTreeMap::new(),
            shared_decls: BTreeMap::new(),
            shared_offset: 0,
            private_decls: BTreeMap::new(),
            carry_updates: Vec::new(),
            carry_definitions: Vec::new(),
        }
    }

    // -- register allocation ------------------------------------------------

    fn r16(&mut self) -> String {
        let v = format!("%rs{}", self.r16);
        self.r16 += 1;
        v
    }
    fn r32(&mut self) -> String {
        let v = format!("%r{}", self.r32);
        self.r32 += 1;
        v
    }
    fn r64(&mut self) -> String {
        let v = format!("%rd{}", self.r64);
        self.r64 += 1;
        v
    }
    fn f32(&mut self) -> String {
        let v = format!("%f{}", self.f32);
        self.f32 += 1;
        v
    }
    fn f64_register(&mut self) -> String {
        let v = format!("%fd{}", self.f64);
        self.f64 += 1;
        v
    }
    fn pred(&mut self) -> String {
        let v = format!("%p{}", self.pred);
        self.pred += 1;
        v
    }
    fn label(&mut self) -> String {
        let v = format!("L{}", self.labels);
        self.labels += 1;
        v
    }
    fn push(&mut self, line: impl Into<String>) {
        self.body.push(format!("  {}", line.into()));
    }

    /// Push one 64-bit kernel parameter and load it into a register.
    fn push_param(&mut self, param: CudaParam) -> String {
        let slot = self.params.len();
        self.params.push(param);
        let register = self.r64();
        self.push(format!("ld.param.u64 {register}, [__kernel_{slot}];"));
        register
    }

    /// The pointer register of one bound storage (global bindings and
    /// declared shared/private storage).
    fn storage_pointer(&mut self, storage: StorageIx) -> String {
        let index = storage.index();
        if let Some((symbol, base, _)) = self.shared_decls.get(&index) {
            let symbol = symbol.clone();
            let base = *base;
            if base == 0 {
                return symbol;
            }
            let register = self.r64();
            self.push(format!("add.u64 {register}, {symbol}, {base};"));
            return register;
        }
        if let Some(register) = self.private_decls.get(&index) {
            return register.clone();
        }
        if let Some(register) = self.storage_registers.get(&index) {
            return register.clone();
        }
        bug(format!(
            "bound storage {index} has no launch parameter (the seal binds every \
             addressed storage)"
        ))
    }

    /// The value register of one runtime extent (adding the value-block
    /// parameter and recording the extent on first use).
    fn extent_register(&mut self, extent: RuntimeExtentId) -> String {
        if let Some(register) = self.extent_registers.get(&extent.0) {
            return register.clone();
        }
        let base = match self.extents_param.clone() {
            Some(base) => base,
            None => {
                let base = self.push_param(CudaParam::RuntimeExtents);
                self.extents_param = Some(base.clone());
                base
            }
        };
        if !self.runtime_extents.contains(&extent) {
            self.runtime_extents.push(extent);
        }
        let address = self.r64();
        self.push(format!(
            "add.u64 {address}, {base}, {};",
            u64::from(extent.0) * 8
        ));
        let register = self.r64();
        self.push(format!("ld.global.u64 {register}, [{address}];"));
        self.extent_registers.insert(extent.0, register.clone());
        register
    }

    fn slots_base(&mut self) -> String {
        match self.slots_param.clone() {
            Some(base) => base,
            None => {
                let base = self.push_param(CudaParam::ExecutorSlots);
                self.slots_param = Some(base.clone());
                base
            }
        }
    }

    fn status_base(&mut self) -> String {
        match self.status_param.clone() {
            Some(base) => base,
            None => {
                let base = self.push_param(CudaParam::Status);
                self.status_param = Some(base.clone());
                base
            }
        }
    }

    fn results_base(&mut self) -> String {
        match self.results_param.clone() {
            Some(base) => base,
            None => {
                let base = self.push_param(CudaParam::Results);
                self.results_param = Some(base.clone());
                base
            }
        }
    }

    fn barrier_base(&mut self) -> String {
        match self.barrier_param.clone() {
            Some(base) => base,
            None => {
                let base = self.push_param(CudaParam::GridBarrier);
                self.barrier_param = Some(base.clone());
                base
            }
        }
    }

    /// The work-item total: the sealed static total, or the retained-total
    /// parameter the executor evaluates from `work_items` at submission.
    fn total_register(&mut self) -> String {
        if let Some(total) = self.total_static {
            let register = self.r64();
            self.push(format!("mov.u64 {register}, {total};"));
            return register;
        }
        match self.work_total_param.clone() {
            Some(base) => base,
            None => {
                let base = self.push_param(CudaParam::WorkTotal);
                self.work_total_param = Some(base.clone());
                base
            }
        }
    }

    // -- places and views ----------------------------------------------------

    /// Seed the place table from the sealed inputs, outputs, and locals,
    /// and bind every scalar input to its typed register.
    fn seed_places(&mut self) {
        // Dynamic view-transform endpoints resolve to the block's scalar
        // inputs (D1 routes every endpoint leaf the block consumes).
        for (id, decl) in self.launch.kernel.interface().inputs.entries() {
            self.endpoint_inputs
                .insert(decl.leaf, KernelValueRef::Input(id));
        }
        for (ordinal, input) in self.launch.inputs.iter().enumerate() {
            let id = KernelInputId(ordinal as u32);
            match input {
                LaunchInput::Storage(views) => {
                    self.places.insert(
                        KernelPlaceRef::Input(id),
                        PlaceInfo {
                            shape: views.first().ty.axes.clone(),
                            views: views.clone(),
                        },
                    );
                }
                LaunchInput::Scalar { slot, dtype, .. } => {
                    let register = match self.scalar_registers.get(slot) {
                        Some(register) => register.clone(),
                        None => bug(format!(
                            "scalar input binding {slot} has no parameter (the seal binds \
                             every kernel input)"
                        )),
                    };
                    self.bound.insert(
                        KernelValueRef::Input(id),
                        Value::Scalar {
                            register,
                            dtype: *dtype,
                        },
                    );
                }
            }
        }
        for (ordinal, output) in self.launch.outputs.iter().enumerate() {
            let id = KernelOutputId(ordinal as u32);
            if let LaunchOutput::Storage(views) = output {
                self.places.insert(
                    KernelPlaceRef::Output(id),
                    PlaceInfo {
                        shape: views.first().ty.axes.clone(),
                        views: views.clone(),
                    },
                );
            }
        }
        for (ordinal, local) in self.launch.locals.iter().enumerate() {
            let id = KernelLocalId(ordinal as u32);
            self.places.insert(
                KernelPlaceRef::Local(id),
                PlaceInfo {
                    shape: local.first().ty.axes.clone(),
                    views: local.clone(),
                },
            );
            for view in local.as_slice() {
                self.declare_local_storage(view.storage);
            }
        }
        // Every axis ordinal's coordinate is bound after the traversal head
        // delinearizes the linear coordinate.
        if let Some(counter) = self.launch.pull_counter {
            let register = self.storage_pointer(counter);
            self.pull_counter_register = Some(register);
        }
    }

    /// The sealed storage fact of one bound storage.
    fn storage_fact(&self, storage: StorageIx) -> seismic_realization::physical::StorageFact {
        match self.binding_facts.get(&storage.index()) {
            Some(fact) => *fact,
            None => bug(format!(
                "bound storage {} has no sealed storage fact (the seal facts cover \
                 every binding)",
                storage.index()
            )),
        }
    }

    /// Declare one kernel-local storage: dynamic shared memory for a
    /// workgroup scope, per-participant memory for a participant scope,
    /// and nothing for a global (arena/ABI) placement.
    fn declare_local_storage(&mut self, storage: StorageIx) {
        let index = storage.index();
        if self.shared_decls.contains_key(&index)
            || self.private_decls.contains_key(&index)
            || self.storage_registers.contains_key(&index)
        {
            return;
        }
        match self.storage_fact(storage) {
            seismic_realization::physical::StorageFact::Global => {
                // A global placement is addressed through its launch
                // binding parameter; nothing to declare.
            }
            seismic_realization::physical::StorageFact::Workgroup { bytes, alignment } => {
                let alignment = alignment.max(1);
                let symbol = format!("__seismic_shared_{index}");
                self.local_decls.push(format!(
                    "  .extern .shared .align {alignment} .b8 {symbol}[];"
                ));
                let register = self.r64();
                self.push(format!("cvta.shared.u64 {register}, {symbol};"));
                let base = self.shared_offset.div_ceil(alignment) * alignment;
                self.shared_offset = base + bytes;
                self.shared_decls.insert(index, (register, base, bytes));
            }
            seismic_realization::physical::StorageFact::Participant { bytes, alignment } => {
                let alignment = alignment.max(1);
                let symbol = format!("__seismic_private_{index}");
                self.local_decls.push(format!(
                    "  .local .align {alignment} .b8 {symbol}[{bytes}];"
                ));
                let register = self.r64();
                self.push(format!("cvta.local.u64 {register}, {symbol};"));
                self.private_decls.insert(index, register);
            }
        }
    }

    /// Bind every axis ordinal's coordinate register (the traversal head
    /// has delinearized them).
    fn bind_axes(&mut self) {
        for ordinal in 0..self.launch.kernel.interface().axes.len() {
            self.bind_axis(ordinal);
        }
    }

    /// Bind one axis ordinal's coordinate register.
    fn bind_axis(&mut self, ordinal: usize) {
        let coordinate = match self.axis_registers.get(ordinal) {
            Some(register) => register.clone(),
            None => bug(format!(
                "axis ordinal {ordinal} has no traversal coordinate (the seal binds \
                 every axis)"
            )),
        };
        self.bound.insert(
            KernelValueRef::Axis(KernelAxisId(ordinal as u32)),
            Value::Index(coordinate),
        );
    }

    /// The pointer of one place's first (dense) storage.
    fn place_pointer(&mut self, place: KernelPlaceRef) -> String {
        let info = match self.places.get(&place) {
            Some(info) => info.clone(),
            None => bug(format!("kernel place {place:?} is not sealed")),
        };
        if info.views.as_slice().len() != 1 {
            bug(format!(
                "a dense access names the multi-plane place {place:?} (packed access \
                 uses the plane operations)"
            ));
        }
        self.storage_pointer(info.views.first().storage)
    }

    // -- entry ---------------------------------------------------------------

    fn emit(mut self) -> CudaLaunch {
        // Materialize the exact sealed launch ABI before emitting any op:
        // storage bindings in slot order, then scalar inputs in scalar-slot
        // order (both dense by the seal).
        let mut bindings: BTreeMap<u32, StorageIx> = BTreeMap::new();
        for binding in &self.launch.bindings {
            if bindings.insert(binding.slot, binding.storage).is_some() {
                bug("two storage bindings claim the same launch parameter slot");
            }
        }
        for (expected, (slot, storage)) in bindings.into_iter().enumerate() {
            if slot as usize != expected {
                bug("the sealed storage binding slots are not dense");
            }
            // `storage_facts` is parallel to `bindings` in slot order.
            let fact = match self.launch.storage_facts.get(slot as usize) {
                Some(fact) => *fact,
                None => bug(format!(
                    "binding slot {slot} has no sealed storage fact (the seal facts \
                     cover every binding)",
                )),
            };
            self.binding_facts.insert(storage.index(), fact);
            // A global placement (ABI buffer or arena offset) is bound as a
            // launch parameter; workgroup and participant storage is
            // declared inside the kernel and never parameterized.
            match fact {
                seismic_realization::physical::StorageFact::Global => {
                    let register = self.push_param(CudaParam::Storage(storage));
                    self.storage_registers.insert(storage.index(), register);
                }
                seismic_realization::physical::StorageFact::Workgroup { .. }
                | seismic_realization::physical::StorageFact::Participant { .. } => {}
            }
        }
        let mut scalar_bindings: BTreeMap<u32, (ScalarSource, DType)> = BTreeMap::new();
        for input in &self.launch.inputs {
            if let LaunchInput::Scalar {
                slot,
                source,
                dtype,
            } = input
            {
                if scalar_bindings
                    .insert(*slot, (source.clone(), *dtype))
                    .is_some()
                {
                    bug("two scalar input bindings claim the same scalar slot");
                }
            }
        }
        for (expected, (slot, (source, dtype))) in scalar_bindings.into_iter().enumerate() {
            if slot as usize != expected {
                bug("the sealed scalar input slots are not dense");
            }
            let word = self.push_param(CudaParam::Scalar {
                slot,
                source,
                dtype,
            });
            // Convert the parameter word into the typed register
            // representation (narrow floats widen).
            let typed = match dtype {
                DType::F32 => {
                    let bits = self.r32();
                    self.push(format!("cvt.u32.u64 {bits}, {word};"));
                    let register = self.f32();
                    self.push(format!("mov.b32 {register}, {bits};"));
                    register
                }
                DType::F16 | DType::BF16 => {
                    let bits = self.r32();
                    self.push(format!("cvt.u32.u64 {bits}, {word};"));
                    let raw = self.r16();
                    self.push(format!("cvt.u16.u32 {raw}, {bits};"));
                    let register = self.f32();
                    self.push(format!(
                        "cvt.f32.{} {register}, {raw};",
                        if dtype == DType::F16 { "f16" } else { "bf16" }
                    ));
                    register
                }
                DType::I32 => {
                    let register = self.r32();
                    self.push(format!("cvt.rzi.s32.s64 {register}, {word};"));
                    register
                }
                DType::U32 | DType::Bool => {
                    let register = self.r32();
                    self.push(format!("cvt.rzi.u32.u64 {register}, {word};"));
                    register
                }
            };
            self.scalar_registers.insert(slot, typed);
        }
        self.seed_places();
        self.emit_traversal_head();
        for op in self.launch.kernel.ops().as_slice() {
            self.op(op);
        }
        self.emit_traversal_tail();
        // The kernel parameter ABI is a solver constraint
        // (`LaunchResources::direct_bindings` against
        // `TargetLimits::max_direct_bindings`); the encoder reports the
        // mechanical count it produced.
        let exact_math = if self.exact_math {
            format!(
                "{}\n{}\n",
                include_str!("math/exp.ptx"),
                include_str!("math/portable_math.ptx")
            )
        } else {
            String::new()
        };
        let name = format!("seismic_launch_{}", self.launch.id.index());
        let parameters = (0..self.params.len())
            .map(|slot| format!(".param .u64 __kernel_{slot}"))
            .collect::<Vec<_>>()
            .join(", ");
        let mut call_decls = String::new();
        for (_, _, input, output) in &self.math_calls {
            call_decls.push_str(&format!(
                "  .param .b32 {input};\n  .param .b32 {output};\n"
            ));
        }
        let local_decls = self.local_decls.join("\n");
        let text = format!(
            ".version 7.1\n.target sm_80\n.address_size 64\n{exact_math}.visible .entry \
             {name}({parameters}) {{\n  .reg .b16 %rs<{}>;\n  .reg .b32 %r<{}>;\n  .reg .b64 \
             %rd<{}>;\n  .reg .f32 %f<{}>;\n  .reg .f64 %fd<{}>;\n  .reg .pred \
             %p<{}>;\n{local_decls}\n{call_decls}{}\n  ret;\n}}\n",
            self.r16.max(1),
            self.r32.max(1),
            self.r64.max(1),
            self.f32.max(1),
            self.f64.max(1),
            self.pred.max(1),
            self.body.join("\n")
        );
        CudaLaunch {
            id: self.launch.id,
            name,
            params: self.params,
            work_items: self.launch.work_items.clone(),
            participants: self.launch.participants.clone(),
            workgroups: self.launch.workgroups.clone(),
            serialized: self.serialized,
            cooperative: is_cooperative(self.launch),
            grid_barriers: self.grid_barriers,
            runtime_extents: self.runtime_extents,
            pull_counter: self.launch.pull_counter,
            ptx: text,
        }
    }

    // -- traversal -----------------------------------------------------------

    /// Emit the visit-loop head: the linear participant coordinate, the
    /// tail mask, and the delinearized axis coordinates.
    fn emit_traversal_head(&mut self) {
        let total = self.total_register();
        match self.traversal.clone() {
            Traversal::OnePass => {
                if self.serialized {
                    // One participant traverses `[0, total)` ascending.
                    let linear = self.r64();
                    self.push(format!("mov.u64 {linear}, 0;"));
                    let top = self.label();
                    let done = self.label();
                    self.body.push(format!("{top}:"));
                    let exhausted = self.pred();
                    self.push(format!("setp.ge.u64 {exhausted}, {linear}, {total};"));
                    self.push(format!("@{exhausted} bra {done};"));
                    self.delinearize(&linear);
                    self.bind_axes();
                    self.linear = Some(linear);
                    self.total = Some(total);
                    self.loop_label = Some(top);
                    self.done_label = Some(done);
                } else {
                    // Each participant visits exactly one linear coordinate.
                    let linear = self.global_thread_id();
                    if self.tail_mask {
                        let done = self.label();
                        let invalid = self.pred();
                        self.push(format!("setp.ge.u64 {invalid}, {linear}, {total};"));
                        self.push(format!("@{invalid} bra {done};"));
                        self.delinearize(&linear);
                        self.bind_axes();
                        self.linear = Some(linear);
                        self.total = Some(total);
                        self.done_label = Some(done);
                    } else {
                        self.delinearize(&linear);
                        self.bind_axes();
                        self.linear = Some(linear);
                        self.total = Some(total);
                    }
                }
            }
            Traversal::GridStride => {
                let linear = self.global_thread_id();
                let stride = self.launch_stride();
                let top = self.label();
                let done = self.label();
                self.body.push(format!("{top}:"));
                let invalid = self.pred();
                self.push(format!("setp.ge.u64 {invalid}, {linear}, {total};"));
                self.push(format!("@{invalid} bra {done};"));
                self.delinearize(&linear);
                self.bind_axes();
                self.linear = Some(linear);
                self.total = Some(total);
                self.stride = Some(stride);
                self.loop_label = Some(top);
                self.done_label = Some(done);
            }
            Traversal::DynamicPull { .. } => {
                // `while ((lin = atom.global.add.u32 counter, 1) < total)`:
                // every coordinate is claimed exactly once, no tail mask.
                let counter = match self.pull_counter_register.clone() {
                    Some(register) => register,
                    None => bug("a dynamic-pull traversal has no pull-counter storage"),
                };
                let top = self.label();
                let done = self.label();
                self.body.push(format!("{top}:"));
                let claimed = self.r32();
                self.push(format!("atom.global.add.u32 {claimed}, [{counter}], 1;"));
                let linear = self.r64();
                self.push(format!("cvt.u64.u32 {linear}, {claimed};"));
                let exhausted = self.pred();
                self.push(format!("setp.ge.u64 {exhausted}, {linear}, {total};"));
                self.push(format!("@{exhausted} bra {done};"));
                self.delinearize(&linear);
                self.bind_axes();
                self.linear = Some(linear);
                self.total = Some(total);
                self.pull_labels = Some((top, done));
            }
        }
    }

    fn emit_traversal_tail(&mut self) {
        if let Some((top, done)) = self.pull_labels.take() {
            self.push(format!("bra {top};"));
            self.body.push(format!("{done}:"));
            return;
        }
        if self.serialized {
            // The ascending serial visit loop; its head (under the same
            // flag) opened it.
            let (Some(linear), Some(top), Some(done)) = (
                self.linear.clone(),
                self.loop_label.clone(),
                self.done_label.clone(),
            ) else {
                bug("the serialized traversal tail follows an open serial loop")
            };
            self.push(format!("add.u64 {linear}, {linear}, 1;"));
            self.push(format!("bra {top};"));
            self.body.push(format!("{done}:"));
            return;
        }
        if let (Some(linear), Some(stride), Some(total), Some(top), Some(done)) = (
            self.linear.clone(),
            self.stride.clone(),
            self.total.clone(),
            self.loop_label.clone(),
            self.done_label.clone(),
        ) {
            self.push(format!("add.u64 {linear}, {linear}, {stride};"));
            let again = self.pred();
            self.push(format!("setp.lt.u64 {again}, {linear}, {total};"));
            self.push(format!("@{again} bra {top};"));
            self.body.push(format!("{done}:"));
            return;
        }
        if let Some(done) = self.done_label.take() {
            self.body.push(format!("{done}:"));
        }
    }

    /// The global linear thread coordinate of this participant.
    fn global_thread_id(&mut self) -> String {
        let block = self.r32();
        let tid = self.r32();
        let cta = self.r32();
        self.push(format!("mov.u32 {tid}, %tid.x;"));
        self.push(format!("mov.u32 {cta}, %ctaid.x;"));
        self.push(format!("mov.u32 {block}, %ntid.x;"));
        let block64 = self.r64();
        let tid64 = self.r64();
        let cta64 = self.r64();
        self.push(format!("cvt.u64.u32 {block64}, {block};"));
        self.push(format!("cvt.u64.u32 {tid64}, {tid};"));
        self.push(format!("cvt.u64.u32 {cta64}, {cta};"));
        let linear = self.r64();
        self.push(format!("mad.lo.u64 {linear}, {cta64}, {block64}, {tid64};"));
        linear
    }

    /// The grid-stride step (all participants of the launch).
    fn launch_stride(&mut self) -> String {
        let block = self.r32();
        let grid = self.r32();
        self.push(format!("mov.u32 {block}, %ntid.x;"));
        self.push(format!("mov.u32 {grid}, %nctaid.x;"));
        let block64 = self.r64();
        let grid64 = self.r64();
        self.push(format!("cvt.u64.u32 {block64}, {block};"));
        self.push(format!("cvt.u64.u32 {grid64}, {grid};"));
        let stride = self.r64();
        self.push(format!("mul.lo.u64 {stride}, {grid64}, {block64};"));
        stride
    }

    /// Delinearize the row-major linear coordinate into per-axis
    /// coordinates (outermost first, last axis fastest), registering each
    /// axis ordinal's coordinate.
    fn delinearize(&mut self, linear: &str) {
        let extents = self.launch.kernel.interface().iteration.extents.clone();
        self.axis_registers = Vec::with_capacity(extents.len());
        let rest = linear.to_string();
        for axis in 0..extents.len() {
            let divisor = self.trailing_stride(&extents[axis + 1..]);
            let coordinate = self.r64();
            self.push(format!("div.u64 {coordinate}, {rest}, {divisor};"));
            let modulus = self.extent_operand(&extents[axis]);
            self.push(format!("rem.u64 {rest}, {rest}, {modulus};"));
            self.axis_registers.push(coordinate);
        }
    }

    /// The stride divisor of a trailing extent suffix.
    fn trailing_stride(&mut self, extents: &[ExtentExpr]) -> String {
        let mut register = self.r64();
        self.push(format!("mov.u64 {register}, 1;"));
        for factor in extents {
            let value = self.extent_operand(factor);
            let next = self.r64();
            self.push(format!("mul.lo.u64 {next}, {register}, {value};"));
            register = next;
        }
        register
    }

    /// The register holding one extent's value (a constant or a runtime
    /// extent value).
    fn extent_operand(&mut self, extent: &ExtentExpr) -> String {
        match extent {
            ExtentExpr::Static(n) => {
                let register = self.r64();
                self.push(format!("mov.u64 {register}, {n};"));
                register
            }
            ExtentExpr::Sym(sym) => match sym.as_constant() {
                Some(value) => {
                    let register = self.r64();
                    self.push(format!("mov.u64 {register}, {value};"));
                    register
                }
                None => bug("an unresolved symbolic extent survived specialization"),
            },
            ExtentExpr::Runtime(id) => self.extent_register(*id),
        }
    }

    // -- operands ------------------------------------------------------------

    fn operand(&mut self, value: KernelValueRef) -> Value {
        match value {
            KernelValueRef::Ssa(id) => match self.ssa.get(&id.0) {
                Some(value) => value.clone(),
                None => bug(format!(
                    "SSA register {} is used before its definition (the seal proves \
                     domination)",
                    id.0
                )),
            },
            KernelValueRef::Input(_) | KernelValueRef::Axis(_) => match self.bound.get(&value) {
                Some(existing) => existing.clone(),
                None => bug(format!(
                    "kernel value {value:?} is not bound (the seal binds every input)"
                )),
            },
        }
    }

    fn scalar_of(&mut self, value: KernelValueRef) -> String {
        let operand = self.operand(value);
        self.scalar_of_value(operand)
    }

    fn scalar_of_value(&mut self, value: Value) -> String {
        match value {
            Value::Scalar { register, .. } => register,
            Value::Index(register) => {
                let converted = self.r32();
                self.push(format!("cvt.u32.u64 {converted}, {register};"));
                converted
            }
        }
    }

    fn index_of(&mut self, value: KernelValueRef) -> String {
        let operand = self.operand(value);
        self.index_of_value(operand)
    }

    fn index_of_value(&mut self, value: Value) -> String {
        match value {
            Value::Index(register) => register,
            Value::Scalar { register, dtype } => {
                let converted = self.r64();
                let (to, from) = if dtype == DType::I32 {
                    ("s64", "s32")
                } else {
                    ("u64", "u32")
                };
                self.push(format!("cvt.{to}.{from} {converted}, {register};"));
                converted
            }
        }
    }

    fn typed_register(&mut self, dtype: DType) -> String {
        if dtype.is_float() {
            self.f32()
        } else {
            self.r32()
        }
    }

    fn constant(&mut self, value: &TypedConstant) -> Value {
        match value.value {
            ConstantValue::Int(bits) => {
                let register = self.r32();
                let raw = if value.dtype == DType::Bool {
                    (bits & 1) as u32
                } else {
                    bits as u32
                };
                self.push(format!("mov.u32 {register}, {raw};"));
                Value::Scalar {
                    register,
                    dtype: value.dtype,
                }
            }
            ConstantValue::Float { bits } => {
                let register = self.f32();
                self.push(format!("mov.b32 {register}, 0x{bits:08x};"));
                Value::Scalar {
                    register,
                    dtype: value.dtype,
                }
            }
            ConstantValue::Bool(flag) => {
                let register = self.r32();
                self.push(format!("mov.u32 {register}, {};", u8::from(flag)));
                Value::Scalar {
                    register,
                    dtype: value.dtype,
                }
            }
        }
    }

    // -- addressing ----------------------------------------------------------

    /// The flat element offset of one place's coordinates, applying the
    /// route/view transform in residence coordinates (steps applied in
    /// reverse, outermost last).
    fn view_offset(&mut self, place: KernelPlaceRef, coords: &[KernelValueRef]) -> String {
        let info = match self.places.get(&place) {
            Some(info) => info.clone(),
            None => bug(format!("kernel place {place:?} is not sealed")),
        };
        let mut coordinates = coords
            .iter()
            .map(|value| self.index_of(*value))
            .collect::<Vec<_>>();
        if coordinates.len() != info.shape.len() {
            bug(format!(
                "place {place:?} is addressed with {} coordinates for view rank {}",
                coordinates.len(),
                info.shape.len()
            ));
        }
        let transform = info.views.first().transform.clone();
        if info.views.iter().any(|view| view.transform != transform) {
            bug("one logical kernel place has inconsistent plane transforms");
        }
        let mut current_shape = info.shape.clone();
        for step in transform.steps.iter().rev() {
            coordinates = self.reverse_step(&coordinates, &current_shape, step);
            current_shape = step.source_shape.clone();
        }
        self.linearize(&coordinates, &current_shape)
    }

    /// Reverse-apply one view step: coordinates in the step's result shape
    /// become coordinates in its source shape.
    fn reverse_step(
        &mut self,
        coordinates: &[String],
        result_shape: &[ExtentExpr],
        step: &ViewStepTemplate,
    ) -> Vec<String> {
        match &step.kind {
            ViewStepKind::Reshape => {
                // Row-major linearization over the result shape, then
                // delinearization over the source shape.
                let linear = self.linearize(coordinates, result_shape);
                self.delinearize_shape(&linear, &step.source_shape)
            }
            ViewStepKind::Transpose { permutation } => {
                if permutation.len() != coordinates.len()
                    || permutation.len() != step.source_shape.len()
                {
                    bug("transpose view rank is inconsistent");
                }
                let mut source = vec![String::new(); step.source_shape.len()];
                for (view_axis, source_axis) in permutation.iter().enumerate() {
                    let slot = source
                        .get_mut(*source_axis as usize)
                        .unwrap_or_else(|| bug("transpose axis is outside source rank"));
                    *slot = coordinates[view_axis].clone();
                }
                if source.iter().any(String::is_empty) {
                    bug("transpose permutation is incomplete");
                }
                source
            }
            ViewStepKind::Slice { axes } => {
                if axes.len() != step.source_shape.len() {
                    bug("slice axis count does not match its source rank");
                }
                let mut source = Vec::with_capacity(step.source_shape.len());
                let mut current = coordinates.iter();
                for axis in axes {
                    match axis {
                        SliceAxisTemplate::Full => source.push(
                            current
                                .next()
                                .cloned()
                                .unwrap_or_else(|| bug("slice is missing a result coordinate")),
                        ),
                        SliceAxisTemplate::Point(leaf) => {
                            let point = self.endpoint_index(*leaf);
                            source.push(point);
                        }
                        SliceAxisTemplate::Range { start, .. } => {
                            let coordinate = current
                                .next()
                                .cloned()
                                .unwrap_or_else(|| bug("range slice is missing a coordinate"));
                            match start {
                                Some(leaf) => {
                                    let start = self.endpoint_index(*leaf);
                                    let shifted = self.r64();
                                    self.push(format!("add.u64 {shifted}, {coordinate}, {start};"));
                                    source.push(shifted);
                                }
                                None => source.push(coordinate),
                            }
                        }
                    }
                }
                if current.next().is_some() {
                    bug("slice has excess result coordinates");
                }
                source
            }
        }
    }

    /// The kernel value of one dynamic view endpoint (a scalar input).
    fn endpoint_index(&mut self, leaf: CanonicalLeafId) -> String {
        match self.endpoint_inputs.get(&leaf) {
            Some(value) => self.index_of(*value),
            None => bug(format!(
                "view endpoint leaf {} is not a scalar input of the block (D1 routes \
                 every consumed endpoint)",
                leaf.0
            )),
        }
    }

    /// Row-major linearization of coordinates over one shape.
    fn linearize(&mut self, coordinates: &[String], shape: &[ExtentExpr]) -> String {
        if coordinates.len() != shape.len() {
            bug("view coordinate rank does not match its shape");
        }
        let strides = self.contiguous_strides(shape);
        let linear = self.r64();
        self.push(format!("mov.u64 {linear}, 0;"));
        for (coordinate, stride) in coordinates.iter().zip(&strides) {
            let stride = self.stride_register(stride);
            let term = self.r64();
            self.push(format!("mul.lo.u64 {term}, {coordinate}, {stride};"));
            self.push(format!("add.u64 {linear}, {linear}, {term};"));
        }
        linear
    }

    /// Row-major delinearization over one shape.
    fn delinearize_shape(&mut self, linear: &str, shape: &[ExtentExpr]) -> Vec<String> {
        let strides = self.contiguous_strides(shape);
        let mut coordinates = Vec::with_capacity(shape.len());
        for (axis, stride) in strides.iter().enumerate() {
            let stride = self.stride_register(stride);
            let coordinate = self.r64();
            self.push(format!("div.u64 {coordinate}, {linear}, {stride};"));
            if axis + 1 < shape.len() {
                let extent = self.extent_operand(&shape[axis]);
                self.push(format!("rem.u64 {coordinate}, {coordinate}, {extent};"));
            }
            coordinates.push(coordinate);
        }
        coordinates
    }

    /// One (possibly runtime-computed) stride per axis.
    fn contiguous_strides(&mut self, shape: &[ExtentExpr]) -> Vec<Stride> {
        let mut strides = vec![Stride::Static(1); shape.len()];
        for axis in (0..shape.len().saturating_sub(1)).rev() {
            let factor = &shape[axis + 1];
            strides[axis] = match (&strides[axis + 1], factor.as_static()) {
                (Stride::Static(inner), Some(n)) => Stride::Static(inner * n),
                _ => {
                    let mut extents = vec![factor.clone()];
                    if let Stride::Runtime(mut tail) = strides[axis + 1].clone() {
                        extents.append(&mut tail);
                    }
                    Stride::Runtime(extents)
                }
            };
        }
        strides
    }

    fn stride_register(&mut self, stride: &Stride) -> String {
        match stride {
            Stride::Static(bytes) => {
                let register = self.r64();
                self.push(format!("mov.u64 {register}, {bytes};"));
                register
            }
            Stride::Runtime(extents) => {
                let mut register = self.r64();
                self.push(format!("mov.u64 {register}, 1;"));
                for extent in extents {
                    let value = self.extent_operand(extent);
                    let next = self.r64();
                    self.push(format!("mul.lo.u64 {next}, {register}, {value};"));
                    register = next;
                }
                register
            }
        }
    }

    /// The element address of one place at the given view coordinates.
    fn element_address(
        &mut self,
        place: KernelPlaceRef,
        coords: &[KernelValueRef],
        dtype: DType,
    ) -> String {
        let pointer = self.place_pointer(place);
        let elements = self.view_offset(place, coords);
        let offset = self.r64();
        self.push(format!(
            "mul.lo.u64 {offset}, {elements}, {};",
            u64::from(dtype.bytes())
        ));
        let address = self.r64();
        self.push(format!("add.u64 {address}, {pointer}, {offset};"));
        address
    }

    /// The pointer of one representation plane of a place.
    fn plane_pointer(&mut self, place: KernelPlaceRef, ordinal: usize) -> String {
        let info = match self.places.get(&place) {
            Some(info) => info.clone(),
            None => bug(format!("kernel place {place:?} is not sealed")),
        };
        let view = info
            .views
            .as_slice()
            .get(ordinal)
            .cloned()
            .unwrap_or_else(|| bug("a representation plane is unbound"));
        self.storage_pointer(view.storage)
    }

    /// The address of one plane element: the outer coordinates' flat row
    /// times the plane's per-row storage extent plus the packing-axis
    /// storage element (the packing axis is the last axis).
    fn plane_element_address(
        &mut self,
        place: KernelPlaceRef,
        coords: &[KernelValueRef],
        repr: &str,
        plane: PlaneField,
    ) -> String {
        let info = match self.places.get(&place) {
            Some(info) => info.clone(),
            None => bug(format!("kernel place {place:?} is not sealed")),
        };
        let representation = repr_lookup(repr);
        let ordinal = plane_ordinal(representation, plane);
        let plane_descriptor = plane_info(representation, ordinal);
        // The plane's own binding carries the view-side type and transform.
        let view = info
            .views
            .as_slice()
            .get(ordinal)
            .cloned()
            .unwrap_or_else(|| bug("a representation plane is unbound"));
        if coords.len() != view.ty.axes.len() || coords.is_empty() {
            bug("a plane access does not match its plane's view rank");
        }
        // Delinearize the view coordinates into the residence: the last
        // axis is the packing axis.
        let mut coordinates = coords
            .iter()
            .map(|value| self.index_of(*value))
            .collect::<Vec<_>>();
        let transform = view.transform.clone();
        let mut current_shape = view.ty.axes.clone();
        for step in transform.steps.iter().rev() {
            coordinates = self.reverse_step(&coordinates, &current_shape, step);
            current_shape = step.source_shape.clone();
        }
        let Some(packing) = coordinates.pop() else {
            bug("a plane access has at least the packing axis")
        };
        let outer_flat = self.linearize(&coordinates, &current_shape[..current_shape.len() - 1]);
        // Per-row storage elements of this plane at the packing extent.
        let packed_extent = self.extent_operand(&current_shape[current_shape.len() - 1]);
        let per_row = self.plane_row_extent(&plane_descriptor, &packed_extent);
        let row_offset = self.r64();
        self.push(format!("mul.lo.u64 {row_offset}, {outer_flat}, {per_row};"));
        let element = self.r64();
        self.push(format!("add.u64 {element}, {row_offset}, {packing};"));
        let base = self.plane_pointer(place, ordinal);
        let address = self.r64();
        self.push(format!(
            "mad.lo.u64 {address}, {element}, {}, {base};",
            u64::from(plane_descriptor.dtype().bytes())
        ));
        address
    }

    /// One packed-plane read: the raw code of entry `entry` of the group of
    /// the logical element at `coords`, zero-extended.
    fn packed_plane_read(
        &mut self,
        place: KernelPlaceRef,
        coords: &[KernelValueRef],
        repr: &str,
        plane: PlaneField,
        entry: u32,
        dtype: DType,
    ) -> String {
        let info = match self.places.get(&place) {
            Some(info) => info.clone(),
            None => bug(format!("kernel place {place:?} is not sealed")),
        };
        let representation = repr_lookup(repr);
        let ordinal = plane_ordinal(representation, plane);
        let plane_descriptor = plane_info(representation, ordinal);
        // The plane's own binding carries the view-side type and transform.
        let view = info
            .views
            .as_slice()
            .get(ordinal)
            .cloned()
            .unwrap_or_else(|| bug("a representation plane is unbound"));
        if coords.len() != view.ty.axes.len() || coords.is_empty() {
            bug("a packed read does not match its plane's view rank");
        }
        let mut coordinates = coords
            .iter()
            .map(|value| self.index_of(*value))
            .collect::<Vec<_>>();
        let transform = view.transform.clone();
        let mut current_shape = view.ty.axes.clone();
        for step in transform.steps.iter().rev() {
            coordinates = self.reverse_step(&coordinates, &current_shape, step);
            current_shape = step.source_shape.clone();
        }
        let Some(packing) = coordinates.pop() else {
            bug("a plane access has at least the packing axis")
        };
        let outer_flat = self.linearize(&coordinates, &current_shape[..current_shape.len() - 1]);
        let packed_extent = self.extent_operand(&current_shape[current_shape.len() - 1]);
        // The entry index within the row: `v / group * fields + entry`.
        let group = u64::from(plane_descriptor.group);
        let fields = u64::from(plane_descriptor.fields);
        let within = self.r64();
        if group > 1 {
            self.push(format!("div.u64 {within}, {packing}, {group};"));
            self.push(format!("mul.lo.u64 {within}, {within}, {fields};"));
        } else {
            self.push(format!("mov.u64 {within}, {packing};"));
        }
        let entry_index = self.r64();
        self.push(format!("add.u64 {entry_index}, {within}, {entry};"));
        let per_row = self.plane_row_extent(&plane_descriptor, &packed_extent);
        let row_offset = self.r64();
        self.push(format!("mul.lo.u64 {row_offset}, {outer_flat}, {per_row};"));
        let element = self.r64();
        self.push(format!("add.u64 {element}, {row_offset}, {entry_index};"));
        let base = self.plane_pointer(place, ordinal);
        match plane_descriptor.encoding {
            repr::PlaneEncoding::Dense(_) => {
                let address = self.r64();
                self.push(format!(
                    "mad.lo.u64 {address}, {element}, {}, {base};",
                    u64::from(plane_descriptor.dtype().bytes())
                ));
                self.load_typed(address, dtype, None)
            }
            repr::PlaneEncoding::Packed { bits, .. } => {
                // The raw code field out of the containing 32-bit word,
                // zero-extended (the plane's storage dtype is u32).
                let bits = u64::from(bits);
                let bit_position = self.r64();
                self.push(format!("mul.lo.u64 {bit_position}, {element}, {bits};"));
                let word_index = self.r64();
                self.push(format!("div.u64 {word_index}, {bit_position}, 32;"));
                let word_address = self.r64();
                self.push(format!(
                    "mad.lo.u64 {word_address}, {word_index}, 4, {base};"
                ));
                let word = self.r32();
                self.push(format!("ld.global.u32 {word}, [{word_address}];"));
                let shift = self.r32();
                let shift64 = self.r64();
                self.push(format!("rem.u64 {shift64}, {bit_position}, 32;"));
                self.push(format!("cvt.u32.u64 {shift}, {shift64};"));
                let raw = self.r32();
                self.push(format!("shr.b32 {raw}, {word}, {shift};"));
                self.push(format!(
                    "and.b32 {raw}, {raw}, {};",
                    (1u32 << bits) & u32::MAX
                ));
                raw
            }
        }
    }

    // -- memory helpers ------------------------------------------------------

    /// Typed load; narrow floats widen into f32 registers.
    fn load_typed(&mut self, address: String, dtype: DType, predicate: Option<&str>) -> String {
        let out = self.typed_register(dtype);
        let prefix = match predicate {
            Some(guard) => format!("@{guard} "),
            None => String::new(),
        };
        match dtype {
            DType::F32 => self.push(format!("{prefix}ld.f32 {out}, [{address}];")),
            DType::F16 | DType::BF16 => {
                let raw = self.r16();
                self.push(format!(
                    "{prefix}ld.{} {raw}, [{address}];",
                    storage_type(dtype)
                ));
                self.push(format!(
                    "cvt.f32.{} {out}, {raw};",
                    if dtype == DType::F16 { "f16" } else { "bf16" }
                ));
            }
            _ => self.push(format!(
                "{prefix}ld.{} {out}, [{address}];",
                storage_type(dtype)
            )),
        }
        out
    }

    /// Typed store; narrow floats round once on the store.
    fn store_typed(
        &mut self,
        address: String,
        value: String,
        dtype: DType,
        predicate: Option<&str>,
    ) {
        let prefix = match predicate {
            Some(guard) => format!("@{guard} "),
            None => String::new(),
        };
        match dtype {
            DType::F32 => self.push(format!("{prefix}st.f32 [{address}], {value};")),
            DType::F16 | DType::BF16 => {
                let raw = self.r16();
                self.push(format!(
                    "cvt.rn.{}.f32 {raw}, {value};",
                    if dtype == DType::F16 { "f16" } else { "bf16" }
                ));
                self.push(format!(
                    "{prefix}st.{} [{address}], {raw};",
                    storage_type(dtype)
                ));
            }
            _ => self.push(format!(
                "{prefix}st.{} [{address}], {value};",
                storage_type(dtype)
            )),
        }
    }

    /// Store one scalar into an 8-byte executor slot word.
    fn store_word(&mut self, address: String, value: String, dtype: DType) {
        let wide = self.r64();
        match dtype {
            DType::F32 => {
                let raw = self.r32();
                self.push(format!("mov.b32 {raw}, {value};"));
                self.push(format!("cvt.u64.u32 {wide}, {raw};"));
            }
            DType::F16 | DType::BF16 => {
                let raw = self.r16();
                let narrow = if dtype == DType::F16 { "f16" } else { "bf16" };
                self.push(format!("cvt.rn.{narrow}.f32 {raw}, {value};"));
                self.push(format!("cvt.u64.u16 {wide}, {raw};"));
            }
            DType::I32 => self.push(format!("cvt.s64.s32 {wide}, {value};")),
            DType::U32 | DType::Bool => {
                self.push(format!("cvt.u64.u32 {wide}, {value};"));
            }
        }
        self.push(format!("st.global.u64 [{address}], {wide};"));
    }

    // -- scalar helpers ------------------------------------------------------

    fn unary(&mut self, op: UnaryOp, value: String, dtype: DType) -> Value {
        let out = self.typed_register(dtype);
        let instruction = match op {
            UnaryOp::Neg if dtype.is_float() => "neg.f32",
            UnaryOp::Neg => "neg.s32",
            UnaryOp::Not => "xor.b32",
            UnaryOp::BitNot => "not.b32",
        };
        if op == UnaryOp::Not {
            self.push(format!("{instruction} {out}, {value}, 1;"));
        } else {
            self.push(format!("{instruction} {out}, {value};"));
        }
        Value::Scalar {
            register: out,
            dtype,
        }
    }

    fn binary(&mut self, op: BinaryOp, lhs: String, rhs: String, dtype: DType) -> Value {
        let comparison = matches!(
            op,
            BinaryOp::Eq | BinaryOp::Ne | BinaryOp::Lt | BinaryOp::Le | BinaryOp::Gt | BinaryOp::Ge
        );
        if comparison {
            let predicate = self.pred();
            self.push(format!(
                "setp.{}.{} {predicate}, {lhs}, {rhs};",
                comparison_name(op),
                ptx_type(dtype)
            ));
            let out = self.r32();
            self.push(format!("selp.u32 {out}, 1, 0, {predicate};"));
            return Value::Scalar {
                register: out,
                dtype: DType::Bool,
            };
        }
        // Integer division and remainder are Euclidean: the quotient is
        // floored and the remainder is always non-negative (the divisor
        // obligation is a sealed `Check` guarding this operation).
        if matches!(op, BinaryOp::Div | BinaryOp::Rem) && !dtype.is_float() {
            return self.euclidean_division(op, lhs, rhs, dtype);
        }
        if matches!(op, BinaryOp::Div | BinaryOp::Rem) && dtype.is_float() {
            bug("floating remainder is undefined (the registry admits no such operation)");
        }
        let out = self.typed_register(dtype);
        let instruction = binary_instruction(op, dtype);
        self.push(format!("{instruction} {out}, {lhs}, {rhs};"));
        Value::Scalar {
            register: out,
            dtype,
        }
    }

    /// Euclidean `a div b` / `a rem b`.
    fn euclidean_division(
        &mut self,
        op: BinaryOp,
        lhs: String,
        rhs: String,
        dtype: DType,
    ) -> Value {
        let signed = dtype == DType::I32;
        let (div, rem) = if signed {
            ("div.s32", "rem.s32")
        } else {
            ("div.u32", "rem.u32")
        };
        let quotient = self.r32();
        let remainder = self.r32();
        self.push(format!("{div} {quotient}, {lhs}, {rhs};"));
        self.push(format!("{rem} {remainder}, {lhs}, {rhs};"));
        if signed {
            // If the remainder is nonzero and the operands' signs differ,
            // the truncated quotient is one too high: subtract one.
            let nonzero = self.pred();
            self.push(format!("setp.ne.s32 {nonzero}, {remainder}, 0;"));
            let lhs_sign = self.pred();
            self.push(format!("setp.lt.s32 {lhs_sign}, {lhs}, 0;"));
            let rhs_sign = self.pred();
            self.push(format!("setp.lt.s32 {rhs_sign}, {rhs}, 0;"));
            let signs_differ = self.pred();
            self.push(format!("xor.pred {signs_differ}, {lhs_sign}, {rhs_sign};"));
            let adjust = self.pred();
            self.push(format!("and.pred {adjust}, {nonzero}, {signs_differ};"));
            self.push(format!("@{adjust} add.s32 {quotient}, {quotient}, -1;"));
        }
        let out = self.r32();
        match op {
            BinaryOp::Div => self.push(format!("mov.b32 {out}, {quotient};")),
            _ => {
                self.push(format!("mul.lo.s32 {out}, {quotient}, {rhs};"));
                self.push(format!("sub.s32 {out}, {lhs}, {out};"));
            }
        }
        Value::Scalar {
            register: out,
            dtype,
        }
    }

    fn cast(&mut self, value: String, source: DType, target: DType) -> Value {
        let source_type = ptx_type(source);
        let out = if target == DType::Bool {
            let predicate = self.pred();
            let zero = if source.is_float() { "0f00000000" } else { "0" };
            self.push(format!(
                "setp.ne.{source_type} {predicate}, {value}, {zero};"
            ));
            let out = self.r32();
            self.push(format!("selp.u32 {out}, 1, 0, {predicate};"));
            out
        } else if target == DType::F32 {
            let out = self.f32();
            if source.is_float() {
                self.push(format!("mov.b32 {out}, {value};"));
            } else {
                self.push(format!("cvt.rn.f32.{source_type} {out}, {value};"));
            }
            out
        } else if matches!(target, DType::F16 | DType::BF16) {
            let float = if source.is_float() {
                value
            } else {
                let converted = self.f32();
                self.push(format!("cvt.rn.f32.{source_type} {converted}, {value};"));
                converted
            };
            let narrow_type = if target == DType::F16 { "f16" } else { "bf16" };
            let raw = self.r16();
            self.push(format!("cvt.rn.{narrow_type}.f32 {raw}, {float};"));
            let out = self.f32();
            self.push(format!("cvt.f32.{narrow_type} {out}, {raw};"));
            out
        } else {
            let out = self.r32();
            if source.is_float() {
                self.push(format!("cvt.rni.{}.f32 {out}, {value};", ptx_type(target)));
            } else {
                // All integer data registers are 32-bit. Signedness changes
                // interpretation, not representation.
                self.push(format!("mov.b32 {out}, {value};"));
            }
            out
        };
        Value::Scalar {
            register: out,
            dtype: target,
        }
    }

    fn select(
        &mut self,
        condition: String,
        then_value: String,
        else_value: String,
        dtype: DType,
    ) -> Value {
        let predicate = self.pred();
        self.push(format!("setp.ne.u32 {predicate}, {condition}, 0;"));
        let out = self.typed_register(dtype);
        self.push(format!(
            "selp.{} {out}, {then_value}, {else_value}, {predicate};",
            ptx_type(dtype)
        ));
        Value::Scalar {
            register: out,
            dtype,
        }
    }

    /// The exact `seismic_math` reference. `exp_fast` is the registry's
    /// approximate-form operator and emits the `ex2.approx` sequence.
    fn math(&mut self, op: MathOp, arguments: &[String]) -> Value {
        // The registry fixes every math operator's arity; the sealed block
        // presents exactly that many operands. An operand list of another
        // length contradicts the sealed algebra.
        if arguments.len() != op.arity() {
            bug(format!(
                "math `{}` takes {} registry operands; the sealed block presents {}",
                op.name(),
                op.arity(),
                arguments.len()
            ));
        }
        let out = self.f32();
        match op {
            MathOp::Fma => {
                let [a, b, c] = arguments else {
                    bug("fma presents its three registry operands")
                };
                let (a, b, c) = (a.clone(), b.clone(), c.clone());
                self.push(format!("fma.rn.f32 {out}, {a}, {b}, {c};"));
            }
            MathOp::Exp | MathOp::Log | MathOp::Sin | MathOp::Cos => {
                let symbol = match op {
                    MathOp::Exp => "seismic_exp",
                    MathOp::Log => "seismic_log",
                    MathOp::Sin => "seismic_sin",
                    MathOp::Cos => "seismic_cos",
                    _ => bug("the transcendental symbol arm is exhaustive"),
                };
                let result = self.call_math(symbol, &arguments[0]);
                self.push(format!("mov.b32 {out}, {result};"));
            }
            MathOp::ExpFast => {
                let scaled = self.f32();
                self.push(format!("mul.f32 {scaled}, {}, 0f3fb8aa3b;", arguments[0]));
                self.push(format!("ex2.approx.f32 {out}, {scaled};"));
            }
            MathOp::Sqrt | MathOp::Rsqrt => {
                // One correctly rounded f64 step: widen, compute in f64,
                // round once back to f32 — the exact reference bits.
                let wide = self.f64_register();
                self.push(format!("cvt.f64.f32 {wide}, {};", arguments[0]));
                let wide_result = self.f64_register();
                match op {
                    MathOp::Sqrt => {
                        self.push(format!("sqrt.rn.f64 {wide_result}, {wide};"));
                    }
                    _ => {
                        let reciprocal = self.f64_register();
                        self.push(format!("sqrt.rn.f64 {reciprocal}, {wide};"));
                        self.push(format!(
                            "div.rn.f64 {wide_result}, 0d3ff0000000000000, {reciprocal};"
                        ));
                    }
                }
                self.push(format!("cvt.rn.f32.f64 {out}, {wide_result};"));
            }
            MathOp::Abs => self.push(format!("abs.f32 {out}, {};", arguments[0])),
            MathOp::Max | MathOp::Min => {
                let [a, b] = arguments else {
                    bug("an extremum presents its two registry operands")
                };
                let (a, b) = (a.clone(), b.clone());
                self.push(format!(
                    "{}.f32 {out}, {a}, {b};",
                    if op == MathOp::Max { "max" } else { "min" }
                ));
            }
        }
        Value::Scalar {
            register: out,
            dtype: DType::F32,
        }
    }

    /// One `seismic_math` software call (exact reference bits).
    fn call_math(&mut self, symbol: &str, argument: &str) -> String {
        self.exact_math = true;
        let ordinal = self.math_calls.len();
        let input = format!("__math_arg_{ordinal}");
        let output = format!("__math_result_{ordinal}");
        self.math_calls
            .push((symbol.to_string(), argument.to_string(), input, output));
        let out = self.f32();
        let (_, _, input, output) = self.math_calls[ordinal].clone();
        self.push(format!("st.param.f32 [{input}], {argument};"));
        self.push(format!("call.uni ({output}), {symbol}, ({input});"));
        self.push(format!("ld.param.f32 {out}, [{output}];"));
        out
    }

    /// The per-row storage element count of one plane at the packing
    /// extent: `ceil(v / group) * fields` entries for a dense plane, and
    /// that scaled by the packing width and rounded up to 32-bit words for
    /// a packed plane (the same arithmetic as `Plane::extent`).
    fn plane_row_extent(&mut self, plane: &repr::Plane, packed_extent: &str) -> String {
        let mut entries = self.r64();
        if plane.group > 1 {
            self.push(format!(
                "div.u64 {entries}, {packed_extent}, {};",
                plane.group
            ));
        } else {
            self.push(format!("mov.u64 {entries}, {packed_extent};"));
        }
        if plane.fields > 1 {
            let scaled = self.r64();
            self.push(format!("mul.lo.u64 {scaled}, {entries}, {};", plane.fields));
            entries = scaled;
        }
        match &plane.encoding {
            repr::PlaneEncoding::Dense(_) => entries,
            repr::PlaneEncoding::Packed { bits, .. } => {
                let bits_register = self.r64();
                self.push(format!("mul.lo.u64 {bits_register}, {entries}, {bits};"));
                let adjusted = self.r64();
                self.push(format!("add.u64 {adjusted}, {bits_register}, 31;"));
                let words = self.r64();
                self.push(format!("div.u64 {words}, {adjusted}, 32;"));
                words
            }
        }
    }

    // -- operations ----------------------------------------------------------

    /// Emit one kernel operation. The match is exhaustive over the sealed
    /// algebra; no selected operation is rejected and no check is added or
    /// omitted.
    fn op(&mut self, op: &KernelOp<CudaIntrinsic>) {
        match op {
            KernelOp::Core(core) => self.core_op(core),
            KernelOp::Intrinsic(intrinsic) => self.intrinsic_op(intrinsic),
        }
    }

    fn intrinsic_op(&mut self, intrinsic: &CudaIntrinsic) {
        match intrinsic {
            CudaIntrinsic::LaneIndex { into } => {
                let register = self.r32();
                self.push(format!("mov.u32 {register}, %laneid;"));
                self.ssa.insert(
                    into.0,
                    Value::Scalar {
                        register,
                        dtype: DType::U32,
                    },
                );
            }
            CudaIntrinsic::Shuffle {
                into,
                value,
                index,
                dtype,
            } => {
                let value = self.scalar_of(*value);
                let index = self.scalar_of(*index);
                let out = self.typed_register(*dtype);
                self.push(format!(
                    "shfl.sync.idx.b32 {out}, {value}, {index}, 31, 0xffffffff;"
                ));
                self.ssa.insert(
                    into.0,
                    Value::Scalar {
                        register: out,
                        dtype: *dtype,
                    },
                );
            }
            CudaIntrinsic::SubgroupReduce {
                into,
                op,
                value,
                dtype,
            } => {
                // The warp butterfly: five descending exchanges cover the
                // 32-lane subgroup exactly.
                let mut current = self.scalar_of(*value);
                for offset in [16u32, 8, 4, 2, 1] {
                    let exchanged = self.typed_register(*dtype);
                    self.push(format!(
                        "shfl.sync.down.b32 {exchanged}, {current}, {offset}, 31, \
                         0xffffffff;"
                    ));
                    let combined = self.typed_register(*dtype);
                    let instruction = match op {
                        ReduceOp::Sum => {
                            format!("add.rn.f32 {combined}, {current}, {exchanged};")
                        }
                        ReduceOp::Max => format!("max.f32 {combined}, {current}, {exchanged};"),
                        ReduceOp::Min => format!("min.f32 {combined}, {current}, {exchanged};"),
                        // The registry declares no `simd_argmax`, so `lower`
                        // never constructs this reduction.
                        ReduceOp::Argmax => bug("argmax has no subgroup reduction"),
                    };
                    self.push(instruction);
                    current = combined;
                }
                self.ssa.insert(
                    into.0,
                    Value::Scalar {
                        register: current,
                        dtype: *dtype,
                    },
                );
            }
        }
    }

    fn core_op(&mut self, op: &CoreKernelOp<CudaIntrinsic>) {
        match op {
            CoreKernelOp::Const { into, value } => {
                let emitted = self.constant(value);
                self.ssa.insert(into.0, emitted);
            }
            CoreKernelOp::RuntimeExtent { into, extent } => {
                let register = self.extent_register(*extent);
                self.ssa.insert(into.0, Value::Index(register));
            }
            CoreKernelOp::Unary {
                into,
                op: kind,
                operand,
                dtype,
            } => {
                let value = self.scalar_of(*operand);
                let emitted = self.unary(*kind, value, *dtype);
                self.ssa.insert(into.0, emitted);
            }
            CoreKernelOp::Binary {
                into,
                op: kind,
                left,
                right,
                dtype,
            } => {
                let (lhs, rhs) = (self.scalar_of(*left), self.scalar_of(*right));
                let emitted = self.binary(*kind, lhs, rhs, *dtype);
                self.ssa.insert(into.0, emitted);
            }
            CoreKernelOp::Compare {
                into,
                op: kind,
                left,
                right,
                dtype,
            } => {
                let (lhs, rhs) = (self.scalar_of(*left), self.scalar_of(*right));
                let predicate = self.pred();
                self.push(format!(
                    "setp.{}.{} {predicate}, {lhs}, {rhs};",
                    rel_name(*kind),
                    ptx_type(*dtype)
                ));
                let out = self.r32();
                self.push(format!("selp.u32 {out}, 1, 0, {predicate};"));
                self.ssa.insert(
                    into.0,
                    Value::Scalar {
                        register: out,
                        dtype: DType::Bool,
                    },
                );
            }
            CoreKernelOp::Math {
                into,
                op: kind,
                operands,
                ..
            } => {
                let arguments = operands
                    .iter()
                    .map(|operand| self.scalar_of(*operand))
                    .collect::<Vec<_>>();
                let emitted = self.math(*kind, &arguments);
                self.ssa.insert(into.0, emitted);
            }
            CoreKernelOp::Fma {
                into,
                a,
                b,
                c,
                dtype,
            } => {
                let (a, b, c) = (self.scalar_of(*a), self.scalar_of(*b), self.scalar_of(*c));
                let out = self.f32();
                self.push(format!("fma.rn.f32 {out}, {a}, {b}, {c};"));
                self.ssa.insert(
                    into.0,
                    Value::Scalar {
                        register: out,
                        dtype: *dtype,
                    },
                );
            }
            CoreKernelOp::Cast {
                into,
                operand,
                from,
                to,
            } => {
                let value = self.scalar_of(*operand);
                let emitted = self.cast(value, *from, *to);
                self.ssa.insert(into.0, emitted);
            }
            CoreKernelOp::Select {
                into,
                condition,
                then_value,
                else_value,
                dtype,
            } => {
                let (condition, then_value, else_value) = (
                    self.scalar_of(*condition),
                    self.scalar_of(*then_value),
                    self.scalar_of(*else_value),
                );
                let emitted = self.select(condition, then_value, else_value, *dtype);
                self.ssa.insert(into.0, emitted);
            }
            CoreKernelOp::TableLookup { into, index, table } => {
                // `index` is a u32 code in `[0, table.len())` (the sealed
                // obligation; a sealed table is nonempty); the lookup is a
                // select chain over the table.
                let first = table
                    .first()
                    .copied()
                    .unwrap_or_else(|| bug("a sealed table lookup has an empty table"));
                let code = self.scalar_of(*index);
                let decoded = self.r32();
                self.push(format!("mov.s32 {decoded}, {first};"));
                for (position, value) in table.iter().copied().enumerate().skip(1) {
                    let selected = self.pred();
                    self.push(format!("setp.eq.u32 {selected}, {code}, {position};"));
                    self.push(format!(
                        "selp.s32 {decoded}, {value}, {decoded}, {selected};"
                    ));
                }
                self.ssa.insert(
                    into.0,
                    Value::Scalar {
                        register: decoded,
                        dtype: DType::I32,
                    },
                );
            }
            CoreKernelOp::Load {
                into,
                place,
                coords,
                dtype,
            } => {
                let address = self.element_address(*place, coords, *dtype);
                let loaded = self.load_typed(address, *dtype, None);
                self.ssa.insert(
                    into.0,
                    Value::Scalar {
                        register: loaded,
                        dtype: *dtype,
                    },
                );
            }
            CoreKernelOp::Store {
                place,
                coords,
                value,
                dtype,
            } => {
                let address = self.element_address(*place, coords, *dtype);
                let value = self.scalar_of(*value);
                self.store_typed(address, value, *dtype, None);
            }
            CoreKernelOp::PackedPlaneRead {
                into,
                place,
                coords,
                repr,
                plane,
                entry,
                dtype,
            } => {
                let value = self.packed_plane_read(*place, coords, repr, *plane, *entry, *dtype);
                self.ssa.insert(
                    into.0,
                    Value::Scalar {
                        register: value,
                        dtype: *dtype,
                    },
                );
            }
            CoreKernelOp::PlaneLoad {
                into,
                place,
                coords,
                repr,
                plane,
                dtype,
            } => {
                let address = self.plane_element_address(*place, coords, repr, *plane);
                let loaded = self.load_typed(address, *dtype, None);
                self.ssa.insert(
                    into.0,
                    Value::Scalar {
                        register: loaded,
                        dtype: *dtype,
                    },
                );
            }
            CoreKernelOp::PlaneStore {
                place,
                coords,
                repr,
                plane,
                value,
                dtype,
            } => {
                let address = self.plane_element_address(*place, coords, repr, *plane);
                let value = self.scalar_of(*value);
                self.store_typed(address, value, *dtype, None);
            }
            CoreKernelOp::Atomic {
                place,
                coords,
                op,
                value,
                dtype,
                mode,
            } => {
                let address = self.element_address(*place, coords, *dtype);
                let value = self.scalar_of(*value);
                match mode {
                    AtomicMode::Serialized => self.serialized_atomic(*op, address, value, *dtype),
                    AtomicMode::Device => self.device_atomic(*op, address, value, *dtype),
                }
            }
            CoreKernelOp::Repeat {
                binder,
                start,
                end,
                body,
            } => {
                let binder_register = self.r64();
                let start_value = self.index_of(*start);
                self.push(format!("mov.u64 {binder_register}, {start_value};"));
                self.ssa
                    .insert(binder.0, Value::Index(binder_register.clone()));
                let top = self.label();
                let done = self.label();
                self.body.push(format!("{top}:"));
                let end_value = self.index_of(*end);
                let exhausted = self.pred();
                self.push(format!(
                    "setp.ge.u64 {exhausted}, {binder_register}, {end_value};"
                ));
                self.push(format!("@{exhausted} bra {done};"));
                let carry_base = self.carry_updates.len();
                let definitions_base = self.carry_definitions.len();
                for nested in body {
                    self.op(nested);
                }
                // Each visit's ordered-carry updates, then the ascending
                // binder step.
                for carry in self.carry_updates.split_off(carry_base) {
                    let update = self.operand(carry.update);
                    match (&carry.register, update) {
                        (Value::Index(destination), Value::Index(source)) => {
                            self.push(format!("mov.u64 {destination}, {source};"));
                        }
                        (
                            Value::Scalar {
                                register: destination,
                                ..
                            },
                            Value::Scalar {
                                register: source, ..
                            },
                        ) => {
                            self.push(format!("mov.b32 {destination}, {source};"));
                        }
                        (destination, source) => bug(format!(
                            "an ordered scalar carry changes its value representation \
                             ({destination:?} := {source:?})"
                        )),
                    }
                }
                self.push(format!("add.u64 {binder_register}, {binder_register}, 1;"));
                self.push(format!("bra {top};"));
                self.body.push(format!("{done}:"));
                // `result` holds the final value of each carry of this
                // loop: the current register after the last visit.
                for (result, current) in self.carry_definitions.split_off(definitions_base) {
                    let final_value = match self.ssa.get(&current.0) {
                        Some(value) => value.clone(),
                        None => bug("an ordered scalar carry has no current value"),
                    };
                    self.ssa.insert(result.0, final_value);
                }
            }
            CoreKernelOp::Carry {
                initial,
                current,
                update,
                result,
                ..
            } => {
                // Listed at the head of its `Repeat` body: `current` is
                // rebound from `initial` here and from `update` at each
                // visit's tail (flushed by the enclosing loop); `result`
                // is defined after the loop as the final value.
                let initial_value = self.operand(*initial);
                let current_register = match initial_value {
                    Value::Index(source) => {
                        let register = self.r64();
                        self.push(format!("mov.u64 {register}, {source};"));
                        Value::Index(register)
                    }
                    Value::Scalar {
                        register: source,
                        dtype,
                    } => {
                        let register = self.typed_register(dtype);
                        self.push(format!("mov.b32 {register}, {source};"));
                        Value::Scalar { register, dtype }
                    }
                };
                self.ssa.insert(current.0, current_register.clone());
                self.carry_updates.push(PendingCarry {
                    update: *update,
                    register: current_register,
                });
                self.carry_definitions.push((*result, *current));
            }
            CoreKernelOp::Branch {
                condition,
                then_body,
                else_body,
                joins,
            } => {
                let condition = self.scalar_of(*condition);
                let predicate = self.pred();
                self.push(format!("setp.ne.u32 {predicate}, {condition}, 0;"));
                let else_label = self.label();
                let done_label = self.label();
                let destinations = joins
                    .iter()
                    .map(|join| self.fresh_join_destination(join))
                    .collect::<Vec<_>>();
                self.push(format!("@!{predicate} bra {else_label};"));
                for nested in then_body {
                    self.op(nested);
                }
                for (join, destination) in joins.iter().zip(&destinations) {
                    let source = self.operand(join.then_value);
                    self.move_join_value(destination, source);
                }
                self.push(format!("bra {done_label};"));
                self.body.push(format!("{else_label}:"));
                for nested in else_body {
                    self.op(nested);
                }
                for (join, destination) in joins.iter().zip(&destinations) {
                    let source = self.operand(join.else_value);
                    self.move_join_value(destination, source);
                }
                self.body.push(format!("{done_label}:"));
                for (join, destination) in joins.iter().zip(destinations) {
                    self.ssa.insert(join.joined.0, destination);
                }
            }
            CoreKernelOp::Fold {
                into,
                op,
                place,
                axis,
                coords,
                schema,
                shape,
            } => {
                let result = self.fold(*op, *place, *axis, coords, schema, shape);
                self.ssa.insert(
                    into.0,
                    Value::Scalar {
                        register: result,
                        dtype: schema.result,
                    },
                );
            }
            CoreKernelOp::Check {
                predicate,
                status,
                guarded,
                ..
            } => {
                let guard = self.check_predicate(predicate);
                // The first-error status write: only when the word is
                // currently zero, so the first violation wins.
                self.record_status_failure(*status, &guard);
                // The guarded operations execute only when the predicate
                // holds; when it fails they are skipped and their SSA
                // values hold unspecified values of their types (the
                // invocation fails after completion).
                let end = self.label();
                self.push(format!("@!{guard} bra {end};"));
                for nested in guarded {
                    self.op(nested);
                }
                self.body.push(format!("{end}:"));
            }
            CoreKernelOp::Publish { value, output } => {
                let value = self.operand(*value);
                let (register, dtype) = match value {
                    Value::Scalar { register, dtype } => (register, dtype),
                    Value::Index(index) => {
                        let register = self.r32();
                        self.push(format!("cvt.u32.u64 {register}, {index};"));
                        (register, DType::I32)
                    }
                };
                match self.launch.outputs.get(output.index()) {
                    Some(LaunchOutput::ExecutorSlot {
                        slot,
                        dtype: out_dtype,
                    }) => {
                        if dtype != *out_dtype {
                            bug("a scalar publication changes its declared dtype");
                        }
                        let base = self.slots_base();
                        let address = self.r64();
                        self.push(format!(
                            "add.u64 {address}, {base}, {};",
                            u64::from(slot.index() as u32) * 8
                        ));
                        self.store_word(address, register, dtype);
                    }
                    Some(LaunchOutput::ResultField {
                        field,
                        dtype: out_dtype,
                    }) => {
                        if dtype != *out_dtype {
                            bug("a scalar publication changes its declared dtype");
                        }
                        let base = self.results_base();
                        let address = self.r64();
                        self.push(format!(
                            "add.u64 {address}, {base}, {};",
                            u64::from(field.index() as u32) * 8
                        ));
                        self.store_word(address, register, dtype);
                    }
                    Some(LaunchOutput::Storage(_)) => {
                        bug("a scalar publication targets tensor storage")
                    }
                    None => bug("a kernel publication names an absent output"),
                }
            }
            CoreKernelOp::Barrier { scope } => match scope {
                BarrierScope::Subgroup => {
                    self.push("bar.warp.sync 0xffffffff;");
                }
                BarrierScope::Workgroup => {
                    self.push("bar.sync 0;");
                }
            },
            CoreKernelOp::GridBarrier => {
                self.grid_barrier();
            }
        }
    }

    // -- joins ---------------------------------------------------------------

    /// A fresh destination register for one branch join value.
    fn fresh_join_destination(&mut self, join: &KernelJoin) -> Value {
        match &join.ty {
            KernelValueType::Scalar(dtype) => {
                let register = self.typed_register(*dtype);
                Value::Scalar {
                    register,
                    dtype: *dtype,
                }
            }
            KernelValueType::Capability(_) => {
                bug("a branch join names a capability value the CUDA dialect has no family for")
            }
        }
    }

    fn move_join_value(&mut self, destination: &Value, source: Value) {
        match (destination, source) {
            (Value::Index(destination), Value::Index(source)) => {
                self.push(format!("mov.u64 {destination}, {source};"));
            }
            (
                Value::Scalar {
                    register: destination,
                    ..
                },
                Value::Scalar {
                    register: source, ..
                },
            ) => {
                self.push(format!("mov.b32 {destination}, {source};"));
            }
            _ => bug("a branch join changes its value representation"),
        }
    }

    // -- checks and folds ----------------------------------------------------

    /// Emit one sealed safety predicate; the result is the guard predicate
    /// register (true when the obligation holds).
    fn check_predicate(&mut self, predicate: &CheckPredicate) -> String {
        let guard = self.pred();
        match predicate {
            CheckPredicate::IndexInBounds { index, extent } => {
                let index = self.index_of(*index);
                let bound = self.extent_operand(extent);
                // `0 <= index` is the representation's invariant; the
                // signed index must also be non-negative.
                let non_negative = self.pred();
                let index32 = self.r32();
                self.push(format!("cvt.u32.u64 {index32}, {index};"));
                self.push(format!("setp.ge.s32 {non_negative}, {index32}, 0;"));
                self.push(format!("setp.lt.u64 {guard}, {index}, {bound};"));
                self.push(format!("and.pred {guard}, {guard}, {non_negative};"));
            }
            CheckPredicate::RangeInBounds { start, end, extent } => {
                let start = self.index_of(*start);
                let end = self.index_of(*end);
                let bound = self.extent_operand(extent);
                let ordered = self.pred();
                self.push(format!("setp.le.u64 {ordered}, {start}, {end};"));
                let within = self.pred();
                self.push(format!("setp.le.u64 {within}, {end}, {bound};"));
                let non_negative = self.pred();
                let start32 = self.r32();
                self.push(format!("cvt.u32.u64 {start32}, {start};"));
                self.push(format!("setp.ge.s32 {non_negative}, {start32}, 0;"));
                self.push(format!("and.pred {guard}, {ordered}, {within};"));
                self.push(format!("and.pred {guard}, {guard}, {non_negative};"));
            }
            CheckPredicate::DivisorNonZero { value, dtype } => {
                let divisor = self.scalar_of(*value);
                let suffix = ptx_type(*dtype);
                let zero = if dtype.is_float() { "0f00000000" } else { "0" };
                self.push(format!("setp.ne.{suffix} {guard}, {divisor}, {zero};"));
            }
            CheckPredicate::SignedDivisionNoOverflow { lhs, rhs } => {
                let (lhs, rhs) = (self.scalar_of(*lhs), self.scalar_of(*rhs));
                let overflow = self.pred();
                self.push(format!("setp.eq.s32 {overflow}, {rhs}, -1;"));
                let is_minimum = self.pred();
                self.push(format!("setp.eq.s32 {is_minimum}, {lhs}, -2147483648;"));
                let both = self.pred();
                self.push(format!("and.pred {both}, {overflow}, {is_minimum};"));
                self.push(format!("not.pred {guard}, {both};"));
            }
            CheckPredicate::ShiftInRange { value } => {
                let amount = self.scalar_of(*value);
                let low = self.pred();
                self.push(format!("setp.ge.s32 {low}, {amount}, 0;"));
                let high = self.pred();
                self.push(format!("setp.lt.s32 {high}, {amount}, 32;"));
                self.push(format!("and.pred {guard}, {low}, {high};"));
            }
        }
        guard
    }

    /// The first-error status write for one failed check: only when the
    /// field's word is currently zero, so the first violation wins.
    fn record_status_failure(&mut self, template: StatusFieldTemplateId, guard: &str) {
        let field = match self.launch.status_fields.get(template.0 as usize) {
            Some(field) => *field,
            None => bug(format!(
                "status template {} has no sealed status field (the seal covers every \
                 template)",
                template.0
            )),
        };
        let base = self.status_base();
        let address = self.r64();
        self.push(format!(
            "add.u64 {address}, {base}, {};",
            u64::from(field.index() as u32) * 4
        ));
        let current = self.r32();
        self.push(format!("ld.global.u32 {current}, [{address}];"));
        let untouched = self.pred();
        self.push(format!("setp.eq.u32 {untouched}, {current}, 0;"));
        let violated = self.pred();
        self.push(format!("not.pred {violated}, {guard};"));
        let record = self.pred();
        self.push(format!("and.pred {record}, {untouched}, {violated};"));
        let code = status_code();
        self.push(format!("@{record} st.global.u32 [{address}], {code};"));
    }

    /// The parallel-outer serial fold: each participant folds the reduced
    /// axis ascending with the registry accumulator/identity/tie semantics
    /// and defines exactly one result.
    #[allow(clippy::too_many_arguments)]
    fn fold(
        &mut self,
        op: ReduceOp,
        place: KernelPlaceRef,
        axis: u32,
        coords: &[KernelValueRef],
        schema: &seismic_lang::intrinsics::ReduceSchema,
        shape: &TensorType,
    ) -> String {
        let info = match self.places.get(&place) {
            Some(info) => info.clone(),
            None => bug(format!("kernel place {place:?} is not sealed")),
        };
        let axis = axis as usize;
        if axis >= shape.axes.len() || coords.len() + 1 != shape.axes.len() {
            bug("a fold's axis and outer coordinates do not match its operand shape");
        }
        // The reduced axis's coordinates come from the fold's own binder;
        // every other axis takes its fixed coordinate (in axis order).
        let mut coordinates: Vec<String> = Vec::with_capacity(shape.axes.len());
        let mut outer = coords.iter();
        for current in 0..shape.axes.len() {
            if current == axis {
                continue;
            }
            let coordinate = outer
                .next()
                .cloned()
                .unwrap_or_else(|| bug("a fold is missing an outer coordinate"));
            coordinates.push(self.index_of(coordinate));
        }
        coordinates.insert(axis, String::new());
        let binder = self.r64();
        self.push(format!("mov.u64 {binder}, 0;"));
        let length = self.extent_operand(&shape.axes[axis]);
        let top = self.label();
        let done = self.label();
        self.body.push(format!("{top}:"));
        let exhausted = self.pred();
        self.push(format!("setp.ge.u64 {exhausted}, {binder}, {length};"));
        self.push(format!("@{exhausted} bra {done};"));
        coordinates[axis] = binder.clone();
        let transform = info.views.first().transform.clone();
        let mut current_shape = info.shape.clone();
        let mut mapped = coordinates.clone();
        for step in transform.steps.iter().rev() {
            mapped = self.reverse_step(&mapped, &current_shape, step);
            current_shape = step.source_shape.clone();
        }
        let elements = self.linearize(&mapped, &current_shape);
        let read_dtype = fold_read_dtype(op, schema);
        let pointer = self.place_pointer(place);
        let offset = self.r64();
        self.push(format!(
            "mul.lo.u64 {offset}, {elements}, {};",
            u64::from(read_dtype.bytes())
        ));
        let address = self.r64();
        self.push(format!("add.u64 {address}, {pointer}, {offset};"));
        let element = self.load_typed(address, read_dtype, None);
        self.fold_step(op, schema, element, &binder);
        self.push(format!("add.u64 {binder}, {binder}, 1;"));
        self.push(format!("bra {top};"));
        self.body.push(format!("{done}:"));
        match op {
            ReduceOp::Argmax => match self.fold_tracked_index.clone() {
                Some(index) => index,
                None => self.unspecified_fold_result(),
            },
            _ => match self.fold_accumulator.clone() {
                Some(accumulator) => accumulator,
                None => self.unspecified_fold_result(),
            },
        }
    }

    /// The result register of a fold over an empty domain: no element was
    /// ever combined, so the published result holds the unspecified value
    /// of its type (the sealed empty-domain guard reports the violation).
    fn unspecified_fold_result(&mut self) -> String {
        self.r64()
    }

    /// One ascending fold step with the registry accumulator/identity/tie
    /// semantics: `sum` accumulates in the accumulator dtype; `max`/`min`
    /// keep the extremum; `argmax` tracks the value and the smaller
    /// coordinate on ties.
    fn fold_step(
        &mut self,
        op: ReduceOp,
        schema: &seismic_lang::intrinsics::ReduceSchema,
        element: String,
        binder: &str,
    ) {
        let accumulator = schema.accumulator;
        match op {
            ReduceOp::Sum => {
                let current = match self.fold_accumulator.clone() {
                    Some(register) => register,
                    None => {
                        // The additive identity of the accumulator dtype.
                        let zero = self.typed_register(accumulator);
                        match accumulator {
                            DType::F32 | DType::F16 | DType::BF16 => {
                                self.push(format!("mov.b32 {zero}, 0x00000000;"));
                            }
                            _ => self.push(format!("mov.u32 {zero}, 0;")),
                        }
                        self.fold_accumulator = Some(zero.clone());
                        zero
                    }
                };
                let out = self.typed_register(accumulator);
                if accumulator.is_float() {
                    self.push(format!("add.rn.f32 {out}, {current}, {element};"));
                } else {
                    self.push(format!("add.u32 {out}, {current}, {element};"));
                }
                self.fold_accumulator = Some(out);
            }
            ReduceOp::Max | ReduceOp::Min => {
                let instruction = if op == ReduceOp::Max {
                    "max.f32"
                } else {
                    "min.f32"
                };
                match self.fold_accumulator.clone() {
                    Some(current) => {
                        let out = self.f32();
                        self.push(format!("{instruction} {out}, {current}, {element};"));
                        self.fold_accumulator = Some(out);
                    }
                    // No identity: the fold starts from the first
                    // (ascending) element.
                    None => self.fold_accumulator = Some(element),
                }
            }
            ReduceOp::Argmax => {
                match (
                    self.fold_accumulator.clone(),
                    self.fold_tracked_index.clone(),
                ) {
                    (Some(current), Some(index)) => {
                        let strictly = self.pred();
                        self.push(format!("setp.gt.f32 {strictly}, {element}, {current};"));
                        let tie = self.pred();
                        self.push(format!("setp.eq.f32 {tie}, {element}, {current};"));
                        let smaller = self.pred();
                        self.push(format!("setp.lt.u64 {smaller}, {binder}, {index};"));
                        let tied_smaller = self.pred();
                        self.push(format!("and.pred {tied_smaller}, {tie}, {smaller};"));
                        let better = self.pred();
                        self.push(format!("or.pred {better}, {strictly}, {tied_smaller};"));
                        let value = self.f32();
                        self.push(format!("selp.f32 {value}, {element}, {current}, {better};"));
                        let chosen = self.r64();
                        self.push(format!("selp.u64 {chosen}, {binder}, {index}, {better};"));
                        self.fold_accumulator = Some(value);
                        self.fold_tracked_index = Some(chosen);
                    }
                    _ => {
                        self.fold_accumulator = Some(element);
                        let start = self.r64();
                        self.push(format!("mov.u64 {start}, {binder};"));
                        self.fold_tracked_index = Some(start);
                    }
                }
            }
        }
    }

    /// The grid-wide software barrier over native scratch: every workgroup
    /// arrives (one atomic ticket per block), the last block releases, and
    /// every block spins until the release flag is set. Legal only in a
    /// `GridCooperative` block, whose launch is co-resident by contract.
    fn grid_barrier(&mut self) {
        if !is_cooperative(self.launch) {
            bug(
                "a grid barrier is sealed into a non-cooperative block (K1 admits it \
                 only in GridCooperative blocks)",
            );
        }
        let site = self.grid_barriers;
        self.grid_barriers += 1;
        let base = self.barrier_base();
        let arrive_address = self.r64();
        self.push(format!(
            "add.u64 {arrive_address}, {base}, {};",
            site as u64 * 8
        ));
        let release_address = self.r64();
        self.push(format!(
            "add.u64 {release_address}, {base}, {};",
            site as u64 * 8 + 4
        ));
        // Align this workgroup: every participant of the block reaches the
        // barrier site structurally (the phases are top-level).
        self.push("bar.sync 0;");
        // One ticket per workgroup, from its first participant.
        let is_leader = self.pred();
        let tid = self.r32();
        self.push(format!("mov.u32 {tid}, %tid.x;"));
        self.push(format!("setp.eq.u32 {is_leader}, {tid}, 0;"));
        let spin = self.label();
        let released = self.label();
        self.push(format!("@!{is_leader} bra {released};"));
        let blocks = self.r32();
        self.push(format!("mov.u32 {blocks}, %nctaid.x;"));
        let previous = self.r32();
        self.push(format!(
            "atom.global.add.u32 {previous}, [{arrive_address}], 1;"
        ));
        let last = self.pred();
        let total_blocks = self.r32();
        self.push(format!("sub.u32 {total_blocks}, {blocks}, 1;"));
        self.push(format!("setp.eq.u32 {last}, {previous}, {total_blocks};"));
        // The last workgroup publishes the release after making its
        // arrival visible device-wide.
        self.push("membar.gl;");
        let one = self.r32();
        self.push(format!("mov.u32 {one}, 1;"));
        self.push(format!(
            "@{last} st.volatile.global.u32 [{release_address}], {one};"
        ));
        self.push(format!("bra {released};"));
        self.body.push(format!("{spin}:"));
        let flag = self.r32();
        self.push(format!(
            "ld.volatile.global.u32 {flag}, [{release_address}];"
        ));
        let ready = self.pred();
        self.push(format!("setp.ne.u32 {ready}, {flag}, 0;"));
        self.push(format!("@{ready} bra {released};"));
        self.push(format!("bra {spin};"));
        self.body.push(format!("{released}:"));
        // Align again: no participant proceeds past the barrier until its
        // whole workgroup has observed the release.
        self.push("bar.sync 0;");
    }

    // -- atomics -------------------------------------------------------------

    /// Serialized exact load/combine/round/store: one participant owns the
    /// domain (the sealed `AtomicMode::Serialized`). Floats widen, combine
    /// in f32, and round on the store; PTX `max.f32`/`min.f32` return the
    /// non-NaN operand, matching the registry's NaN-ignoring rule.
    fn serialized_atomic(&mut self, op: AtomicOp, address: String, value: String, dtype: DType) {
        match dtype {
            DType::F32 | DType::F16 | DType::BF16 => {
                let loaded = self.load_typed(address.clone(), dtype, None);
                let out = self.f32();
                let instruction = match op {
                    AtomicOp::Add => "add.rn.f32",
                    AtomicOp::Max => "max.f32",
                    AtomicOp::Min => "min.f32",
                };
                self.push(format!("{instruction} {out}, {loaded}, {value};"));
                self.store_typed(address, out, dtype, None);
            }
            DType::I32 | DType::U32 => {
                let loaded = self.r32();
                let suffix = if dtype == DType::I32 { "s32" } else { "u32" };
                self.push(format!("ld.{} {loaded}, [{address}];", suffix));
                let out = self.r32();
                let instruction = match op {
                    AtomicOp::Add => format!("add.{suffix}"),
                    AtomicOp::Max => format!("max.{suffix}"),
                    AtomicOp::Min => format!("min.{suffix}"),
                };
                self.push(format!("{instruction} {out}, {loaded}, {value};"));
                self.push(format!("st.{suffix} [{address}], {out};"));
            }
            DType::Bool => bug("a bool atomic update is undefined (the registry admits none)"),
        }
    }

    /// The device atomic: native fetch-add/max/min for 32-bit integers,
    /// native `atom.global.add.f32` for f32 `add`, and a compare/exchange
    /// loop over the bits for float `max`/`min` (NaN-ignoring) and narrow
    /// floats. `add` rounds once per update; `max`/`min` are exact.
    fn device_atomic(&mut self, op: AtomicOp, address: String, value: String, dtype: DType) {
        match (op, dtype) {
            (AtomicOp::Add, DType::I32) => {
                let out = self.r32();
                self.push(format!("atom.global.add.s32 {out}, [{address}], {value};"));
            }
            (AtomicOp::Add, DType::U32) => {
                let out = self.r32();
                self.push(format!("atom.global.add.u32 {out}, [{address}], {value};"));
            }
            (AtomicOp::Max, DType::I32) => {
                let out = self.r32();
                self.push(format!("atom.global.max.s32 {out}, [{address}], {value};"));
            }
            (AtomicOp::Max, DType::U32) => {
                let out = self.r32();
                self.push(format!("atom.global.max.u32 {out}, [{address}], {value};"));
            }
            (AtomicOp::Min, DType::I32) => {
                let out = self.r32();
                self.push(format!("atom.global.min.s32 {out}, [{address}], {value};"));
            }
            (AtomicOp::Min, DType::U32) => {
                let out = self.r32();
                self.push(format!("atom.global.min.u32 {out}, [{address}], {value};"));
            }
            // f32 add rounds once per update natively.
            (AtomicOp::Add, DType::F32) => {
                let out = self.f32();
                self.push(format!("atom.global.add.f32 {out}, [{address}], {value};"));
            }
            // f32 max/min: compare/exchange on the bits, ignoring a NaN
            // operand (`max.f32`/`min.f32` return the non-NaN operand).
            (AtomicOp::Max | AtomicOp::Min, DType::F32) => {
                let retry = self.label();
                self.body.push(format!("{retry}:"));
                let assumed_bits = self.r32();
                self.push(format!("ld.global.b32 {assumed_bits}, [{address}];"));
                let assumed = self.f32();
                self.push(format!("mov.b32 {assumed}, {assumed_bits};"));
                let combined = self.f32();
                let instruction = if op == AtomicOp::Max {
                    "max.f32"
                } else {
                    "min.f32"
                };
                self.push(format!("{instruction} {combined}, {assumed}, {value};"));
                let combined_bits = self.r32();
                self.push(format!("mov.b32 {combined_bits}, {combined};"));
                let stored_bits = self.r32();
                self.push(format!(
                    "atom.global.cas.b32 {stored_bits}, [{address}], {assumed_bits}, \
                     {combined_bits};"
                ));
                let failed = self.pred();
                self.push(format!(
                    "setp.ne.b32 {failed}, {stored_bits}, {assumed_bits};"
                ));
                self.push(format!("@{failed} bra {retry};"));
            }
            // Narrow floats: a compare/exchange loop over the containing
            // 32-bit word (the word lies inside the tensor's own
            // allocation; the layout admitted it).
            (AtomicOp::Add | AtomicOp::Max | AtomicOp::Min, DType::F16 | DType::BF16) => {
                let word = self.r64();
                self.push(format!("and.b64 {word}, {address}, -4;"));
                let half_selector = self.r64();
                self.push(format!("rem.u64 {half_selector}, {address}, 4;"));
                let shift = self.r64();
                self.push(format!("mul.lo.u64 {shift}, {half_selector}, 8;"));
                let retry = self.label();
                self.body.push(format!("{retry}:"));
                let assumed = self.r32();
                self.push(format!("ld.global.b32 {assumed}, [{word}];"));
                // Extract the target half, widen.
                let extracted = self.r32();
                self.push(format!("shr.b32 {extracted}, {assumed}, {shift};"));
                self.push(format!("and.b32 {extracted}, {extracted}, 0xffff;"));
                let extracted16 = self.r16();
                self.push(format!("cvt.u16.u32 {extracted16}, {extracted};"));
                let widened = self.f32();
                self.push(format!(
                    "cvt.f32.{} {widened}, {extracted16};",
                    if dtype == DType::F16 { "f16" } else { "bf16" }
                ));
                // Combine under the registry law; `add` rounds once at the
                // element dtype, `max`/`min` ignore a NaN operand.
                let combined = self.f32();
                let instruction = match op {
                    AtomicOp::Add => "add.rn.f32",
                    AtomicOp::Max => "max.f32",
                    AtomicOp::Min => "min.f32",
                };
                self.push(format!("{instruction} {combined}, {widened}, {value};"));
                // Repack into the containing word.
                let narrow = self.r16();
                self.push(format!(
                    "cvt.rn.{}.f32 {narrow}, {combined};",
                    if dtype == DType::F16 { "f16" } else { "bf16" }
                ));
                let narrow32 = self.r32();
                self.push(format!("cvt.u32.u16 {narrow32}, {narrow};"));
                let packed = self.r32();
                let mask = self.r32();
                self.push(format!("shl.b32 {mask}, 65535, {shift};"));
                self.push(format!("not.b32 {mask}, {mask};"));
                self.push(format!("and.b32 {packed}, {assumed}, {mask};"));
                let shifted = self.r32();
                self.push(format!("shl.b32 {shifted}, {narrow32}, {shift};"));
                self.push(format!("or.b32 {packed}, {packed}, {shifted};"));
                let stored = self.r32();
                self.push(format!(
                    "atom.global.cas.b32 {stored}, [{word}], {assumed}, {packed};"
                ));
                let failed = self.pred();
                self.push(format!("setp.ne.u32 {failed}, {stored}, {assumed};"));
                self.push(format!("@{failed} bra {retry};"));
            }
            (AtomicOp::Add | AtomicOp::Max | AtomicOp::Min, DType::Bool) => {
                bug("a bool atomic update is undefined (the registry admits none)")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Free helpers
// ---------------------------------------------------------------------------

/// The status code recorded by a failed check (nonzero; first error wins).
/// The typed safety kind of the failing field is carried by the plan's
/// status-field table, not by the word.
fn status_code() -> u32 {
    1
}

/// The load dtype of one fold element: the accumulator dtype governs
/// floating sums (f32 for narrow inputs); integers keep theirs.
fn fold_read_dtype(op: ReduceOp, schema: &seismic_lang::intrinsics::ReduceSchema) -> DType {
    let _ = op;
    schema.accumulator
}

fn repr_lookup(name: &str) -> &'static repr::Repr {
    repr::lookup(name).unwrap_or_else(|| {
        bug(format!(
            "representation `{name}` is not registered (the checker admitted it)"
        ))
    })
}

fn plane_ordinal(representation: &repr::Repr, plane: PlaneField) -> usize {
    representation.plane_index(plane.name()).unwrap_or_else(|| {
        bug(format!(
            "representation `{}` has no plane `{}`",
            representation.name,
            plane.name()
        ))
    })
}

fn plane_info(representation: &repr::Repr, ordinal: usize) -> repr::Plane {
    representation
        .planes()
        .into_iter()
        .nth(ordinal)
        .unwrap_or_else(|| bug("a plane ordinal exceeds the representation"))
}

/// The comparison mnemonic of one relational operator.
fn comparison_name(op: BinaryOp) -> &'static str {
    match op {
        BinaryOp::Eq => "eq",
        BinaryOp::Ne => "ne",
        BinaryOp::Lt => "lt",
        BinaryOp::Le => "le",
        BinaryOp::Gt => "gt",
        BinaryOp::Ge => "ge",
        _ => bug("a comparison reached arithmetic emission"),
    }
}

fn rel_name(op: RelOp) -> &'static str {
    match op {
        RelOp::Eq => "eq",
        RelOp::Ne => "ne",
        RelOp::Lt => "lt",
        RelOp::Le => "le",
        RelOp::Gt => "gt",
        RelOp::Ge => "ge",
    }
}

/// The PTX instruction of one non-relational operator.
fn binary_instruction(op: BinaryOp, dtype: DType) -> &'static str {
    match op {
        BinaryOp::Or | BinaryOp::BitOr => "or.b32",
        BinaryOp::And | BinaryOp::BitAnd => "and.b32",
        BinaryOp::BitXor => "xor.b32",
        BinaryOp::Shl => "shl.b32",
        BinaryOp::Shr => {
            if dtype == DType::I32 {
                "shr.s32"
            } else {
                "shr.u32"
            }
        }
        BinaryOp::Add => {
            if dtype.is_float() {
                "add.rn.f32"
            } else {
                "add.u32"
            }
        }
        BinaryOp::Sub => {
            if dtype.is_float() {
                "sub.rn.f32"
            } else {
                "sub.u32"
            }
        }
        BinaryOp::Mul => {
            if dtype.is_float() {
                "mul.rn.f32"
            } else {
                "mul.lo.u32"
            }
        }
        BinaryOp::Div => {
            if dtype.is_float() {
                "div.rn.f32"
            } else {
                bug("integer division is emitted by the Euclidean sequence")
            }
        }
        BinaryOp::Rem => bug("remainder is emitted by the Euclidean sequence"),
        _ => bug("a comparison reached arithmetic emission"),
    }
}

/// Storage type spelling of one dtype in PTX.
pub(crate) fn storage_type(dtype: DType) -> &'static str {
    match dtype {
        DType::F32 => "f32",
        DType::F16 | DType::BF16 => "b16",
        DType::I32 => "s32",
        DType::U32 => "u32",
        DType::Bool => "u8",
    }
}

/// Register type spelling of one dtype (narrow floats live widened).
pub(crate) fn ptx_type(dtype: DType) -> &'static str {
    match dtype {
        DType::F32 | DType::F16 | DType::BF16 => "f32",
        DType::I32 => "s32",
        DType::U32 | DType::Bool => "u32",
    }
}
