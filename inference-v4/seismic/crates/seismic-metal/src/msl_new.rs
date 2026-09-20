//! Mechanical MSL encoding of resolved Metal launches.
//!
//! Each launch is encoded once. The printer accepts `MetalOp` exhaustively
//! and makes no allocation, geometry, synchronization, algorithm, or
//! precision decision: iteration comes from the resolved launch's geometry
//! (the common arbitrary-rank grid-stride mapping), storage from the resolved
//! binding groups, and guards from the planned discharges. There is no
//! assembly alias union: boundaries already name caller storage, and alias
//! validation is the root-ABI `validate_alias_rules`. Produced (output)
//! values resolve through the launch's own value bindings, which carry node
//! outputs after inputs.

use crate::physical::{
    AccessLayout, AddrExpr, BoolOp, ConstValue, DivRemOp, Dst, Guard, GuardPredicate, IndexRef,
    MetalDialect, MetalOp, ReduceResult, ShiftOp, Src, UnaryOp, ValueRef,
};
use seismic_compiler::pipeline::EncodedPlan;
use seismic_lang::{
    intrinsics::AtomicOp,
    logical::GraphValueId,
    types::{DType, RuntimeExtentId},
};
use seismic_realization::executable::{
    AccessMode, BarrierScope, BufferBindingId, ExecutionExpr, ResolvedAbi, ResolvedExecutorScalar,
    ResolvedKernelStep, ResolvedLaunch, ResolvedSchedule, ResolvedStep, ResolvedStorage,
    ResolvedStorageId,
};
use std::collections::{BTreeMap, BTreeSet};

/// Fixed extra buffer indices after the binding-group slots.
pub const SLOT_BUFFER_INDEX: u32 = 26;
pub const EXTENT_BUFFER_INDEX: u32 = 27;
pub const STATUS_BUFFER_INDEX: u32 = 28;
pub const MAX_KERNEL_BUFFERS: usize = 31;

/// Per-launch encoding: the resolved launch itself (geometry and binding
/// groups are retained execution data); MSL is rendered at assembly, when the
/// full plan context is available.
#[derive(Clone, Debug)]
pub struct EncodedLaunch {
    pub launch: ResolvedLaunch<MetalDialect>,
}

/// One native buffer binding of an emitted launch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LaunchBinding {
    /// A public root-ABI buffer.
    Buffer {
        binding: BufferBindingId,
        access: AccessMode,
    },
    /// A slice of the internal device arena.
    Arena { offset: u64, access: AccessMode },
    /// The planned executor-scalar slot block (4 bytes per slot).
    Slots,
    /// The retained runtime-extent block (8 bytes per extent id).
    Extents,
    /// The root status block.
    Status,
}

/// The encoded native artifact: sources, launches, the structured execution
/// tree, the root ABI, and the planned blocks the runtime owns.
#[derive(Clone, Debug)]
pub struct Emitted {
    pub source: String,
    pub launches: Vec<EmittedLaunch>,
    pub execution: Vec<ExecutionItem>,
    pub abi: ResolvedAbi<MetalDialect>,
    pub arena_bytes: u64,
    /// Resolved executor-scalar slot count (4 bytes each).
    pub slot_count: usize,
    /// Runtime-extent count (8 bytes each).
    pub extent_count: usize,
    /// Retained runtime-extent values (execution expressions, never
    /// capacities), for runtime evaluation.
    pub runtime_extents: Vec<(RuntimeExtentId, ExecutionExpr)>,
    /// Per launch: which resolved executor slots its ops write (diagnostics).
    pub launch_slots_written: Vec<BTreeSet<u64>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EmittedLaunch {
    pub kernel: String,
    pub work_items: ExecutionExpr,
    pub participants: ExecutionExpr,
    pub bindings: Vec<LaunchBinding>,
    pub threadgroup_bytes: u64,
    /// A statically skipped launch (zero work): retained identity, no
    /// native pipeline.
    pub skipped: bool,
}

/// Encoded mirror of the resolved structured schedule (never flattened away).
/// `If`/`Repeat` retain their control scalars so the runtime evaluates them
/// directly; nothing is reconstructed from kernel names.
#[derive(Clone, Debug, PartialEq)]
pub enum ExecutionItem {
    Launch(usize),
    Call(Vec<ExecutionItem>),
    If {
        condition: ResolvedExecutorScalar,
        then_steps: Vec<ExecutionItem>,
        else_steps: Vec<ExecutionItem>,
    },
    Repeat {
        binder: seismic_realization::executable::ResolvedExecutorScalarId,
        start: ResolvedExecutorScalar,
        end: ResolvedExecutorScalar,
        body: Vec<ExecutionItem>,
    },
}

/// Encode one resolved launch (retained; MSL rendering happens in
/// `assemble` with full plan context).
pub fn encode_launch(launch: &ResolvedLaunch<MetalDialect>) -> Result<EncodedLaunch, String> {
    if launch.bindings.len() > MAX_KERNEL_BUFFERS {
        return Err(format!(
            "launch#{} binds {} groups; Metal permits {MAX_KERNEL_BUFFERS} buffers",
            launch.id.0,
            launch.bindings.len()
        ));
    }
    for group in &launch.bindings {
        let top = group.slot as usize + group.members.len();
        let reserved = [SLOT_BUFFER_INDEX, EXTENT_BUFFER_INDEX, STATUS_BUFFER_INDEX]
            .into_iter()
            .map(|index| index as usize)
            .min()
            .unwrap();
        if top > reserved {
            return Err(format!(
                "launch#{} binding group at slot {} exceeds the reserved Metal buffer table",
                launch.id.0, group.slot
            ));
        }
    }
    Ok(EncodedLaunch {
        launch: launch.clone(),
    })
}

/// Assemble the native artifact: render MSL for every encoded launch, mirror
/// the structured execution tree, and collect the planned blocks.
pub fn assemble(encoded: EncodedPlan<MetalDialect, EncodedLaunch>) -> Result<Emitted, String> {
    let resolved = &encoded.resolved;
    let mut sources = String::from(
        "#include <metal_stdlib>\nusing namespace metal;\n#pragma clang fp contract(off)\n\n",
    );
    // Rendered launches keyed by resolved launch id (the execution tree
    // addresses launches by that identity).
    let mut rendered: BTreeMap<u64, (EmittedLaunch, BTreeSet<u64>)> = BTreeMap::new();
    for step in &encoded.steps {
        assemble_step(step, resolved, &mut sources, &mut rendered)?;
    }
    let mut launches = Vec::new();
    let mut launch_slots_written = Vec::new();
    let mut next = 0u64;
    for (id, (launch, slots)) in rendered {
        while next < id {
            launches.push(placeholder_launch(next));
            launch_slots_written.push(BTreeSet::new());
            next += 1;
        }
        launches.push(launch);
        launch_slots_written.push(slots);
        next = id + 1;
    }
    let extent_count = resolved.runtime_extents.len();
    let slot_count = schedule_slot_max(&resolved.entry.schedule);
    Ok(Emitted {
        source: sources,
        launches,
        execution: execution_tree(&encoded.steps)?,
        abi: resolved.abi.clone(),
        arena_bytes: resolved.internal_arena.bytes,
        slot_count,
        extent_count,
        runtime_extents: resolved
            .runtime_extents
            .ids()
            .zip(resolved.runtime_extents.iter())
            .map(|(id, expr)| (id, expr.clone()))
            .collect(),
        launch_slots_written,
    })
}

fn placeholder_launch(_id: u64) -> EmittedLaunch {
    // A statically skipped launch (zero work) retains its identity without a
    // native pipeline.
    EmittedLaunch {
        kernel: String::new(),
        work_items: ExecutionExpr::Const(0),
        participants: ExecutionExpr::Const(1),
        bindings: Vec::new(),
        threadgroup_bytes: 0,
        skipped: true,
    }
}

/// The highest resolved executor-scalar slot id referenced anywhere in one
/// schedule tree (runtime-written binder slots included).
fn schedule_slot_max(schedule: &ResolvedSchedule<MetalDialect>) -> usize {
    let mut max = 0usize;
    fn transport_max(
        transport: &seismic_realization::executable::ResolvedTransport,
        max: &mut usize,
    ) {
        match transport {
            seismic_realization::executable::ResolvedTransport::ExecutorScalar(
                ResolvedExecutorScalar::Slot { slot, .. },
            ) => *max = (*max).max(slot.0 as usize + 1),
            seismic_realization::executable::ResolvedTransport::Tuple(items) => {
                for item in items.iter() {
                    transport_max(item, max);
                }
            }
            _ => {}
        }
    }
    fn walk(schedule: &ResolvedSchedule<MetalDialect>, max: &mut usize) {
        for step in schedule.steps.iter() {
            match step {
                ResolvedStep::Launch(launch) => {
                    for group in &launch.bindings {
                        for kernel_step in launch.kernel.steps.iter() {
                            if let ResolvedKernelStep::Mapped { bindings, .. } = kernel_step {
                                for (_, transport) in bindings {
                                    transport_max(transport, max);
                                }
                            }
                        }
                        let _ = group;
                    }
                }
                ResolvedStep::Call(call) => {
                    for transport in call
                        .boundary
                        .inputs
                        .values()
                        .chain(call.boundary.results.values())
                    {
                        transport_max(transport, max);
                    }
                    walk(&call.body.schedule, max);
                }
                ResolvedStep::If(if_step) => {
                    transport_max(
                        &seismic_realization::executable::ResolvedTransport::ExecutorScalar(
                            if_step.condition.clone(),
                        ),
                        max,
                    );
                    walk(&if_step.then_schedule, max);
                    walk(&if_step.else_schedule, max);
                }
                ResolvedStep::Repeat(repeat) => {
                    *max = (*max).max(repeat.binder.0 as usize + 1);
                    walk(&repeat.body, max);
                }
            }
        }
    }
    walk(schedule, &mut max);
    max
}

/// Mirror the encoded plan's structured steps (launch identities retained,
/// dynamic control never flattened away).
fn execution_tree(
    steps: &[seismic_compiler::pipeline::EncodedStep<MetalDialect, EncodedLaunch>],
) -> Result<Vec<ExecutionItem>, String> {
    let mut out = Vec::new();
    for step in steps {
        out.push(match step {
            seismic_compiler::pipeline::EncodedStep::Launch { .. } => {
                ExecutionItem::Launch(launch_index_of(step)?)
            }
            seismic_compiler::pipeline::EncodedStep::Call { encoded, .. } => {
                ExecutionItem::Call(execution_tree(&encoded.steps)?)
            }
            seismic_compiler::pipeline::EncodedStep::If {
                resolved,
                then_steps,
                else_steps,
            } => ExecutionItem::If {
                condition: resolved.condition.clone(),
                then_steps: execution_tree(then_steps)?,
                else_steps: execution_tree(else_steps)?,
            },
            seismic_compiler::pipeline::EncodedStep::Repeat { resolved, body } => {
                ExecutionItem::Repeat {
                    binder: resolved.binder,
                    start: resolved.range.start.clone(),
                    end: resolved.range.end.clone(),
                    body: execution_tree(body)?,
                }
            }
        });
    }
    Ok(out)
}

fn launch_index_of(
    step: &seismic_compiler::pipeline::EncodedStep<MetalDialect, EncodedLaunch>,
) -> Result<usize, String> {
    match step {
        seismic_compiler::pipeline::EncodedStep::Launch { encoded, .. } => {
            Ok(encoded.launch.id.0 as usize)
        }
        _ => Err("internal: a launch step carries an encoded launch".into()),
    }
}

/// Render one step (recursively), appending kernel sources and launch specs.
fn assemble_step(
    step: &seismic_compiler::pipeline::EncodedStep<MetalDialect, EncodedLaunch>,
    resolved: &seismic_realization::executable::ResolvedPlan<MetalDialect>,
    sources: &mut String,
    rendered: &mut BTreeMap<u64, (EmittedLaunch, BTreeSet<u64>)>,
) -> Result<(), String> {
    match step {
        seismic_compiler::pipeline::EncodedStep::Launch {
            resolved: launch, ..
        } => {
            let output = render_launch(launch, resolved)?;
            sources.push_str(&output.source);
            rendered.insert(
                launch.id.0,
                (
                    EmittedLaunch {
                        kernel: output.kernel,
                        work_items: launch.work_items.clone(),
                        participants: launch.geometry.participants_per_workgroup[0].clone(),
                        bindings: output.bindings,
                        threadgroup_bytes: launch.kernel.resources.workgroup_bytes,
                        skipped: false,
                    },
                    output.slots_written,
                ),
            );
            Ok(())
        }
        seismic_compiler::pipeline::EncodedStep::Call { encoded, .. } => {
            for child in &encoded.steps {
                assemble_step(child, resolved, sources, rendered)?;
            }
            Ok(())
        }
        seismic_compiler::pipeline::EncodedStep::If {
            then_steps,
            else_steps,
            ..
        } => {
            for child in then_steps.iter().chain(else_steps) {
                assemble_step(child, resolved, sources, rendered)?;
            }
            Ok(())
        }
        seismic_compiler::pipeline::EncodedStep::Repeat { body, .. } => {
            for child in body {
                assemble_step(child, resolved, sources, rendered)?;
            }
            Ok(())
        }
    }
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

struct RenderedLaunch {
    source: String,
    kernel: String,
    bindings: Vec<LaunchBinding>,
    slots_written: BTreeSet<u64>,
}

/// One launch input binding: (value, resolved transport).
type ResolvedBinding = (
    GraphValueId,
    seismic_realization::executable::ResolvedTransport,
);

struct Renderer {
    /// Every resolved storage bound by this launch → pointer name.
    pointers: BTreeMap<ResolvedStorageId, String>,
    bindings: Vec<LaunchBinding>,
    lines: Vec<String>,
    mapped: Vec<ResolvedBinding>,
    /// Resolved executor slots this launch's ops write.
    slots_written: SlotsWritten,
    /// Counter for decode temporaries.
    temporaries: u32,
}

fn render_launch(
    launch: &ResolvedLaunch<MetalDialect>,
    plan: &seismic_realization::executable::ResolvedPlan<MetalDialect>,
) -> Result<RenderedLaunch, String> {
    let kernel = format!("seismic_metal_{}", launch.id.0);
    let mut renderer = Renderer {
        pointers: BTreeMap::new(),
        bindings: Vec::new(),
        lines: Vec::new(),
        mapped: Vec::new(),
        slots_written: SlotsWritten::default(),
        temporaries: 0,
    };
    // Resource declarations: one pointer per binding-group member at
    // successive indices, then the shared slot/extent/status blocks. The
    // member's access mode comes from the resolved binding group (parameter
    // ownership on the root transports).
    let mut declarations = Vec::new();
    let mut next_index = 0u32;
    for group in &launch.bindings {
        match group.kind {
            seismic_realization::executable::BindingGroupKind::Direct => {
                for member in group.members.iter() {
                    let storage = plan.storage.get(member.storage).ok_or_else(|| {
                        format!("resolved storage#{} is absent", member.storage.0)
                    })?;
                    let name = format!("b{next_index}");
                    let dtype = layout_dtype(&storage.layout);
                    let address = match member.access {
                        AccessMode::Read => "const device",
                        _ => "device",
                    };
                    declarations.push(format!(
                        "{address} {}* {name} [[buffer({next_index})]]",
                        dtype_name(dtype)
                    ));
                    renderer.bindings.push(binding_of(storage, member.access)?);
                    renderer.pointers.insert(storage.id, name.clone());
                    next_index += 1;
                }
            }
            seismic_realization::executable::BindingGroupKind::ArgumentTable => {
                return Err(format!(
                    "compiler bug: launch#{} binds an argument table; no current Metal \
                     strategy emits one (a table is a hard resource of the alternative that \
                     declares it)",
                    launch.id.0
                ));
            }
        }
    }
    declarations.push(format!(
        "constant uint* seismic_slots [[buffer({SLOT_BUFFER_INDEX})]]"
    ));
    renderer.bindings.push(LaunchBinding::Slots);
    declarations.push(format!(
        "constant ulong* seismic_extents [[buffer({EXTENT_BUFFER_INDEX})]]"
    ));
    renderer.bindings.push(LaunchBinding::Extents);
    if plan.abi.status.is_some() {
        declarations.push(format!(
            "device uint* seismic_status [[buffer({STATUS_BUFFER_INDEX})]]"
        ));
        renderer.bindings.push(LaunchBinding::Status);
    }
    declarations.push("uint3 tg_pos [[threadgroup_position_in_grid]]".into());
    declarations.push("uint3 tid [[thread_position_in_threadgroup]]".into());
    declarations.push("uint3 tpg [[threads_per_threadgroup]]".into());
    declarations.push("uint3 tgn [[threadgroups_per_grid]]".into());
    declarations.push("uint simd_lane [[thread_index_in_simdgroup]]".into());
    let mut source = format!(
        "kernel void {kernel}(\n    {}\n) {{\n",
        declarations.join(",\n    ")
    );
    // Planned workgroup storage: a statically sized threadgroup array the
    // resolved plan's staged allocations own; emitters address it by offset.
    if launch.kernel.resources.workgroup_bytes > 0 {
        let words = launch.kernel.resources.workgroup_bytes / 4;
        source.push_str(&format!(
            "  threadgroup float seismic_wg[{words}];\n  \
             (void)seismic_wg[0];\n"
        ));
    }
    // Grid-stride prologue over the launch's retained work-item total (the
    // common arbitrary-rank mapping; rank is not the native grid rank).
    let total = expr_string(&launch.work_items);
    source.push_str(&format!(
        "  uint seismic_gid = tg_pos.x * tpg.x + tid.x;\n  \
         uint seismic_stride = tpg.x * tgn.x;\n  \
         for (uint seismic_lin = seismic_gid; seismic_lin < {total}; \
         seismic_lin += seismic_stride) {{\n"
    ));
    for step in launch.kernel.steps.iter() {
        renderer.step(step)?;
    }
    for line in &renderer.lines {
        source.push_str("    ");
        source.push_str(line);
        source.push('\n');
    }
    source.push_str("  }\n}\n\n");
    Ok(RenderedLaunch {
        source,
        kernel,
        bindings: renderer.bindings,
        slots_written: renderer.slots_written.0.clone(),
    })
}

fn binding_of(
    storage: &ResolvedStorage<MetalDialect>,
    access: AccessMode,
) -> Result<LaunchBinding, String> {
    Ok(match storage.placement {
        seismic_realization::executable::ResolvedStoragePlacement::Abi { binding } => {
            LaunchBinding::Buffer { binding, access }
        }
        seismic_realization::executable::ResolvedStoragePlacement::Arena { offset } => {
            LaunchBinding::Arena { offset, access }
        }
        placement => {
            return Err(format!(
                "compiler bug: {:?} storage entered a Metal device binding group",
                placement.scope()
            ));
        }
    })
}

fn layout_dtype(layout: &crate::physical::ResolvedMetalLayout) -> DType {
    match layout {
        crate::physical::ResolvedMetalLayout::Dense { dtype, .. } => *dtype,
        crate::physical::ResolvedMetalLayout::PackedPlane { dtype, .. } => *dtype,
    }
}

/// A tiny cell for the slots-written set (cleared into the result).
#[derive(Default)]
struct SlotsWritten(BTreeSet<u64>);

impl Renderer {
    fn step(&mut self, step: &ResolvedKernelStep<MetalDialect>) -> Result<(), String> {
        match step {
            ResolvedKernelStep::Mapped {
                iteration,
                bindings,
                ops,
            } => {
                self.delinearize(iteration)?;
                self.mapped = bindings.clone();
                for op in ops.iter() {
                    self.op(op)?;
                }
                Ok(())
            }
            ResolvedKernelStep::Barrier { scope } => {
                match scope {
                    BarrierScope::Subgroup => self.line("simdgroup_barrier();"),
                    BarrierScope::Workgroup => {
                        self.line("threadgroup_barrier(mem_flags::mem_threadgroup);")
                    }
                }
                Ok(())
            }
            ResolvedKernelStep::Publish { storage } => {
                let name = self
                    .pointers
                    .get(storage)
                    .cloned()
                    .unwrap_or_else(|| format!("storage#{}", storage.0));
                self.line(format!("/* publish {name} (retained transport) */"));
                Ok(())
            }
            ResolvedKernelStep::PublishScalar { slot } => {
                self.slots_written.0.insert(slot.0);
                Ok(())
            }
        }
    }

    /// Per-axis coordinate variables: delinearization of the linear coordinate
    /// into every logical axis (arbitrary rank; the last axis is fastest).
    fn delinearize(
        &mut self,
        iteration: &seismic_realization::dispatch::LinearIterationMap,
    ) -> Result<(), String> {
        let rank = iteration.extents.len();
        if rank == 0 {
            return Ok(());
        }
        let mut rest = "seismic_lin".to_string();
        for axis in 0..rank {
            // The row-major stride of this axis: the product of the following
            // extents.
            let mut stride = AddrExpr::Const(1);
            for following in iteration.extents[axis + 1..].iter() {
                stride = AddrExpr::Mul(Box::new(stride), Box::new(extent_addr_of(following)?));
            }
            let divisor = addr_expr_string(&stride);
            if divisor == "1" {
                self.line(format!("uint c{axis} = {rest}; uint seismic_r{axis} = 0u;"));
            } else {
                self.line(format!(
                    "uint c{axis} = {rest} / {divisor}; uint seismic_r{axis} = {rest} % {divisor};"
                ));
            }
            rest = format!("seismic_r{axis}");
        }
        Ok(())
    }

    /// Flat row-major entry index of one view element over its axes: the
    /// element address expression (slice offsets included).
    fn entry_expr(
        &mut self,
        layout: &AccessLayout,
        indices: &[IndexRef],
    ) -> Result<String, String> {
        self.address(layout, indices)
    }

    /// One operand's device pointer plus its layout's base offset (slice
    /// points and range starts), as a single addressable expression.
    fn based_pointer(&mut self, binding: &(usize, AccessLayout)) -> Result<String, String> {
        let pointer = self.operand_pointer(binding.0)?;
        if binding.1.offset.is_empty() {
            return Ok(pointer);
        }
        let mut offset = String::new();
        for (stride, value) in &binding.1.offset {
            let term = format!("{} * ({})", addr_expr_string(stride), self.value(*value)?);
            offset = if offset.is_empty() {
                term
            } else {
                format!("{offset} + {term}")
            };
        }
        Ok(format!("{pointer} + {offset}"))
    }

    /// Decoded f32 value of one packed element: the code from the words
    /// plane combined with the representation's coefficient planes. The
    /// single-rounding `fma` matches the reference combine exactly.
    fn packed_decode_value(
        &mut self,
        operand: usize,
        repr: &str,
        entry: String,
    ) -> Result<String, String> {
        use seismic_lang::repr::{CodeInterpretation, Coefficient, PlaneEncoding};
        let representation = seismic_lang::repr::lookup(repr)
            .ok_or_else(|| format!("unknown representation `{repr}`"))?;
        let plane_ordinal = |name: &str| {
            representation
                .plane_index(name)
                .ok_or_else(|| format!("representation `{repr}` has no `{name}` plane"))
        };
        let bits = representation.bits;
        let per_word = 32 / bits;
        let mask = (1u32 << bits) - 1;
        // Raw code field from the words plane.
        let words = self.plane_pointer(operand, plane_ordinal("words")?)?;
        let temp = self.temporaries;
        self.temporaries += 1;
        let raw = format!("seismic_pk{temp}");
        self.line(format!(
            "uint {raw} = ({words}[({entry}) / {per_word}] >> ((({entry}) % {per_word}) * {bits})) & {mask}u;"
        ));
        let code = match &representation.code {
            CodeInterpretation::Unsigned => format!("float({raw})"),
            CodeInterpretation::TwosComplement => {
                format!("float((int({raw}) << (32 - {bits})) >> (32 - {bits}))")
            }
            CodeInterpretation::Offset(zero) => format!("float(int({raw}) - {zero})"),
            CodeInterpretation::Table(table) => {
                let values = table
                    .iter()
                    .map(|value| format!("int({value})"))
                    .collect::<Vec<_>>()
                    .join(", ");
                let table_name = format!("seismic_tbl{temp}");
                self.line(format!("constant int {table_name}[] = {{ {values} }};"));
                format!("float({table_name}[{raw}])")
            }
        };
        // One coefficient plane read (dense scale/factor plane or packed
        // coefficients plane), converted to f32.
        let plane_read = |renderer: &mut Renderer, plane_name: &str, plane_entry: String| {
            let ordinal = plane_ordinal(plane_name)?;
            let pointer = renderer.plane_pointer(operand, ordinal)?;
            let representation = seismic_lang::repr::lookup(repr).expect("looked up above");
            let plane = representation
                .planes()
                .into_iter()
                .find(|candidate| candidate.name == plane_name)
                .ok_or_else(|| format!("`{repr}` has no plane `{plane_name}`"))?;
            Ok(match &plane.encoding {
                PlaneEncoding::Dense(dtype) => match dtype {
                    seismic_lang::types::DType::F32 => {
                        format!("float({pointer}[{plane_entry}])")
                    }
                    seismic_lang::types::DType::F16 => {
                        format!("float({pointer}[{plane_entry}])")
                    }
                    seismic_lang::types::DType::BF16 => {
                        format!("as_type<float>(as_type<uint>({pointer}[{plane_entry}]) << 16)")
                    }
                    other => {
                        return Err(format!(
                            "a coefficient plane of dtype {other:?} is not floating"
                        ));
                    }
                },
                PlaneEncoding::Packed { bits, .. } => {
                    let per_word = 32 / bits;
                    let mask = (1u32 << bits) - 1;
                    format!(
                        "float(({pointer}[({plane_entry}) / {per_word}] >> ((({plane_entry}) % {per_word}) * {bits})) & {mask}u)"
                    )
                }
            })
        };
        let scale = match representation.coefficient(false) {
            Some(Coefficient::Direct { plane }) => {
                let plane_entry = format!("(({entry}) / {})", plane.group);
                plane_read(self, plane.name, plane_entry)?
            }
            Some(Coefficient::Product {
                factor,
                coefficients,
                field,
                sign,
            }) => {
                let factor_entry = format!("(({entry}) / {})", factor.group);
                let factor_value = plane_read(self, factor.name, factor_entry)?;
                let coeff_entry = format!(
                    "((({entry}) / {}) * {} + {field})",
                    coefficients.group, coefficients.fields
                );
                let coeff_value = plane_read(self, coefficients.name, coeff_entry)?;
                format!("({factor_value} * {coeff_value} * float({sign}))")
            }
            None => "1.0".into(),
        };
        Ok(match representation.coefficient(true) {
            Some(Coefficient::Direct { plane }) => {
                let plane_entry = format!("(({entry}) / {})", plane.group);
                let bias = plane_read(self, plane.name, plane_entry)?;
                format!("fma({scale}, {code}, {bias})")
            }
            Some(Coefficient::Product { factor, .. }) => {
                let plane_entry = format!("(({entry}) / {})", factor.group);
                let bias = plane_read(self, factor.name, plane_entry)?;
                format!("fma({scale}, {code}, {bias})")
            }
            None => format!("({scale} * {code})"),
        })
    }

    fn line(&mut self, line: impl Into<String>) {
        self.lines.push(line.into());
    }

    fn op(&mut self, op: &MetalOp) -> Result<(), String> {
        use MetalOp::*;
        match op {
            View | Alloc => Ok(()),
            Const { into, value, dtype } => {
                let name = self.fresh_ssa(*into)?;
                self.line(format!(
                    "{} {name} = {};",
                    dtype_name(*dtype),
                    const_expr(*value, *dtype)
                ));
                Ok(())
            }
            ExtentOf { value, into } => {
                let name = self.fresh_ssa(*into)?;
                self.line(format!("int {name} = int({});", addr_expr_string(value)));
                Ok(())
            }
            Select {
                condition,
                then,
                otherwise,
                into,
                dtype,
            } => {
                let name = self.fresh_ssa(*into)?;
                let condition = self.value(*condition)?;
                let then = self.src(then)?;
                let otherwise = self.src(otherwise)?;
                self.line(format!(
                    "{} {name} = ({condition}) ? {then} : {otherwise};",
                    dtype_name(*dtype)
                ));
                Ok(())
            }
            Compare {
                op: rel,
                lhs,
                rhs,
                into,
                dtype,
            } => {
                let name = self.fresh_ssa(*into)?;
                let lhs = self.src(lhs)?;
                let rhs = self.src(rhs)?;
                let cast = comparison_cast(*dtype);
                self.line(format!(
                    "bool {name} = {cast}({lhs}) {} {cast}({rhs});",
                    rel_symbol(*rel)
                ));
                Ok(())
            }
            BoolLogic { op, operands, into } => {
                let name = self.fresh_ssa(*into)?;
                let expression = match (op, operands.as_slice()) {
                    (BoolOp::Not, [operand]) => format!("!{}", self.value(*operand)?),
                    (BoolOp::And, operands) => self.values(operands)?.join(" && "),
                    (BoolOp::Or, operands) => self.values(operands)?.join(" || "),
                    _ => return Err("compiler bug: bool logic arity".into()),
                };
                self.line(format!("bool {name} = {expression};"));
                Ok(())
            }
            IntOp {
                op,
                lhs,
                rhs,
                into,
                dtype,
            } => {
                let name = self.fresh_ssa(*into)?;
                let lhs = self.src(lhs)?;
                let rhs = self.src(rhs)?;
                self.line(format!(
                    "{} {name} = as_type<{}>(as_type<uint>({lhs}) {} as_type<uint>({rhs}));",
                    dtype_name(*dtype),
                    dtype_name(*dtype),
                    int_symbol(*op),
                ));
                Ok(())
            }
            IntDivRem {
                op,
                lhs,
                rhs,
                into,
                dtype,
                guard,
            } => {
                let name = self.fresh_ssa(*into)?;
                let lhs = self.src(lhs)?;
                let rhs = self.src(rhs)?;
                // Euclidean division/remainder (r in 0..|rhs|), exactly the
                // registry semantics.
                let body = match op {
                    DivRemOp::Div => format!(
                        "int seismic_r = as_type<int>({lhs}) % as_type<int>({rhs}); \
                         if (seismic_r < 0) seismic_r += (as_type<int>({rhs}) < 0 ? \
                         -as_type<int>({rhs}) : as_type<int>({rhs})); \
                         {} {name} = {}((as_type<int>({lhs}) - seismic_r) / as_type<int>({rhs}));",
                        dtype_name(*dtype),
                        dtype_name(*dtype),
                    ),
                    DivRemOp::Rem => format!(
                        "int seismic_r = as_type<int>({lhs}) % as_type<int>({rhs}); \
                         if (seismic_r < 0) seismic_r += (as_type<int>({rhs}) < 0 ? \
                         -as_type<int>({rhs}) : as_type<int>({rhs})); \
                         {} {name} = {}(seismic_r);",
                        dtype_name(*dtype),
                        dtype_name(*dtype),
                    ),
                };
                self.guarded(guard, body)
            }
            Shift {
                op,
                value,
                amount,
                into,
                dtype,
                guard,
            } => {
                let name = self.fresh_ssa(*into)?;
                let value = self.src(value)?;
                let amount = self.src(amount)?;
                let body = match op {
                    ShiftOp::Shl => format!(
                        "{} {name} = as_type<{}>(as_type<uint>({value}) << uint({amount}));",
                        dtype_name(*dtype),
                        dtype_name(*dtype),
                    ),
                    ShiftOp::Shr => match dtype {
                        DType::U32 => {
                            format!("uint {name} = as_type<uint>({value}) >> uint({amount});")
                        }
                        _ => format!("int {name} = as_type<int>({value}) >> uint({amount});"),
                    },
                };
                self.guarded(guard, body)
            }
            Unary {
                op,
                operand,
                into,
                dtype,
            } => {
                let name = self.fresh_ssa(*into)?;
                let operand = self.src(operand)?;
                let body = match (op, dtype.is_float()) {
                    (UnaryOp::Neg, true) => {
                        format!("{} {name} = -{operand};", dtype_name(*dtype))
                    }
                    (UnaryOp::Neg, false) => format!(
                        "{dtype} {name} = as_type<{dtype}>(0u - as_type<uint>({operand}));",
                        dtype = dtype_name(*dtype),
                    ),
                    (UnaryOp::BitNot, _) => {
                        format!("{} {name} = ~{operand};", dtype_name(*dtype))
                    }
                    (UnaryOp::Not, _) => format!("bool {name} = !{operand};"),
                };
                self.line(body);
                Ok(())
            }
            FloatOp {
                op,
                lhs,
                rhs,
                into,
                dtype,
            } => {
                let name = self.fresh_ssa(*into)?;
                let lhs = self.src(lhs)?;
                let rhs = self.src(rhs)?;
                let expression = match op {
                    crate::physical::FloatArith::Add => format!("{lhs} + {rhs}"),
                    crate::physical::FloatArith::Sub => format!("{lhs} - {rhs}"),
                    crate::physical::FloatArith::Mul => format!("{lhs} * {rhs}"),
                    crate::physical::FloatArith::Div => format!("{lhs} / {rhs}"),
                    crate::physical::FloatArith::Rem => format!("fmod({lhs}, {rhs})"),
                };
                self.line(format!("{} {name} = {expression};", dtype_name(*dtype)));
                Ok(())
            }
            Fma {
                a,
                b,
                c,
                into,
                dtype,
            } => {
                let name = self.fresh_ssa(*into)?;
                let a = self.src(a)?;
                let b = self.src(b)?;
                let c = self.src(c)?;
                self.line(format!(
                    "{} {name} = fma({a}, {b}, {c});",
                    dtype_name(*dtype)
                ));
                Ok(())
            }
            Math {
                op,
                args,
                into,
                dtype,
                reference,
            } => {
                let name = self.fresh_ssa(*into)?;
                let mut rendered = Vec::new();
                for arg in args {
                    rendered.push(self.src(arg)?);
                }
                let call = math_call(*op, &rendered, *dtype)?;
                self.line(format!(
                    "{} {name} = {call}; /* seismic_math {}-v{} */",
                    dtype_name(*dtype),
                    reference.identity,
                    reference.version,
                ));
                Ok(())
            }
            Cast {
                from,
                to,
                operand,
                into,
            } => {
                let name = self.fresh_ssa(*into)?;
                let operand = self.src(operand)?;
                self.line(format!(
                    "{} {name} = {};",
                    dtype_name(*to),
                    cast_expr(*from, *to, &operand)
                ));
                Ok(())
            }
            RangeEndpoint {
                start,
                operand,
                into,
            } => {
                let name = self.fresh_ssa(*into)?;
                let slot = self.tuple_slot(*operand, usize::from(!*start))?;
                let slot = slot.0.to_string();
                self.line(format!("int {name} = {};", slot_read(&slot, DType::I32)));
                Ok(())
            }
            TupleGet {
                operand,
                index,
                into,
            } => {
                let name = self.fresh_ssa(*into)?;
                let slot = self.tuple_slot(*operand, *index)?;
                let slot = slot.0.to_string();
                let dtype = self.binding_dtype(*operand)?;
                self.line(format!(
                    "{} {name} = {};",
                    dtype_name(dtype),
                    slot_read(&slot, dtype)
                ));
                Ok(())
            }
            MatrixMatmul {
                left,
                right,
                into,
                accumulate,
                k,
                dtype,
                ..
            } => {
                // One output element per participant (launch coordinates
                // c0/c1 over [M, N]); the exact ascending-k fma chain the
                // reference body defines.
                let left_pointer = self.based_pointer(left)?;
                let right_pointer = self.based_pointer(right)?;
                let into_pointer = self.based_pointer(into)?;
                let left_stride = addr_expr_string(&left.1.strides[0]);
                let right_stride = addr_expr_string(&right.1.strides[0]);
                let into_stride = addr_expr_string(&into.1.strides[1]);
                let into_row_stride = addr_expr_string(&into.1.strides[0]);
                let k_length = addr_expr_string(k);
                let dtype_name = dtype_name(*dtype);
                let initial = if *accumulate {
                    format!("float({into_pointer}[c0 * {into_row_stride} + c1 * {into_stride}])")
                } else {
                    "0.0f".into()
                };
                self.line(format!("float mm_acc = {initial};"));
                self.line(format!(
                    "for (uint mm_k = 0; mm_k < {k_length}u; mm_k++) \
                     mm_acc = fma(float({left_pointer}[c0 * {left_stride} + mm_k]), \
                     float({right_pointer}[c1 * {right_stride} + mm_k]), mm_acc);"
                ));
                self.line(format!(
                    "{into_pointer}[c0 * {into_row_stride} + c1 * {into_stride}] \
                     = {dtype_name}(mm_acc);"
                ));
                Ok(())
            }
            PackedDecode { .. } => {
                Err("compiler bug: packed decode emission is not implemented".into())
            }
            PackedElementRead {
                operand,
                layout,
                indices,
                repr,
                into,
                guard,
            } => {
                let name = self.fresh_ssa(*into)?;
                // Flat entry over the view's axes at the given indices.
                let entry = self.entry_expr(layout, indices)?;
                let value = self.packed_decode_value(*operand, repr, entry)?;
                let body = format!("float {name} = {value};");
                self.guarded(guard, body)
            }
            SerialFor {
                binder,
                length,
                body,
            } => {
                let binder_name = self.fresh_ssa(*binder)?;
                let length = addr_expr_string(length);
                self.line(format!(
                    "for (int {binder_name} = 0; {binder_name} < {length}; {binder_name}++) {{"
                ));
                for op in body.iter() {
                    self.op(op)?;
                }
                self.line("}");
                Ok(())
            }
            Branch {
                condition,
                then_ops,
                else_ops,
            } => {
                let condition = self.value(*condition)?;
                self.line(format!("if ({condition}) {{"));
                for op in then_ops.iter() {
                    self.op(op)?;
                }
                if else_ops.is_empty() {
                    self.line("}");
                } else {
                    self.line("} else {");
                    for op in else_ops.iter() {
                        self.op(op)?;
                    }
                    self.line("}");
                }
                Ok(())
            }
            PackedPlaneRead {
                operand,
                plane,
                dtype,
                into,
            } => {
                let name = self.fresh_ssa(*into)?;
                let pointer = self.plane_pointer(*operand, *plane)?;
                self.line(format!(
                    "{} {name} = {}[seismic_lin];",
                    dtype_name(*dtype),
                    pointer
                ));
                Ok(())
            }
            ReadElement {
                operand,
                layout,
                indices,
                dtype,
                into,
                guard,
            } => {
                let name = self.fresh_ssa(*into)?;
                let address = self.address(layout, indices)?;
                let pointer = self.operand_pointer(*operand)?;
                let body = format!("{} {name} = {}[{address}];", dtype_name(*dtype), pointer);
                self.guarded(guard, body)
            }
            WriteElement {
                operand,
                layout,
                indices,
                value,
                dtype,
                guard,
            } => {
                let value = self.src(value)?;
                let address = self.address(layout, indices)?;
                let pointer = self.operand_pointer(*operand)?;
                let body = format!("{pointer}[{address}] = {value};");
                let _ = dtype;
                self.guarded(guard, body)
            }
            StoreResult {
                value,
                dst,
                layout,
                indices,
                dtype,
            } => {
                let value = self.value(*value)?;
                let (pointer, address) = self.dst_location(dst, layout, indices)?;
                self.line(format!("{pointer}[{address}] = {value};"));
                let _ = dtype;
                Ok(())
            }
            LinearLoop {
                op: loop_op,
                source,
                dst,
                dst_layout,
                dtype,
                fill,
            } => {
                let dst_address = self.address(
                    dst_layout,
                    &(0..dst_layout.strides.len())
                        .map(IndexRef::Axis)
                        .collect::<Vec<_>>(),
                )?;
                let dst_pointer = self.dst_base(dst)?;
                match (source, fill) {
                    (Some((source_operand, source_layout)), _) => {
                        let source_address = self.address(
                            source_layout,
                            &(0..source_layout.strides.len())
                                .map(IndexRef::Axis)
                                .collect::<Vec<_>>(),
                        )?;
                        let source_pointer = self.operand_pointer(*source_operand)?;
                        self.line(format!(
                            "{dst_pointer}[{dst_address}] = {source_pointer}[{source_address}];"
                        ));
                    }
                    (None, Some(value)) => {
                        self.line(format!(
                            "{dst_pointer}[{dst_address}] = {};",
                            const_expr(*value, *dtype)
                        ));
                    }
                    (None, None) => match loop_op {
                        seismic_compiler::terminal::LinearLoopOp::Fill => {
                            return Err("compiler bug: a fill loop carries its value".into());
                        }
                        _ => return Err("compiler bug: a snapshot loop carries its source".into()),
                    },
                }
                Ok(())
            }
            AtomicSerial {
                op,
                operand,
                layout,
                indices,
                value,
                dtype,
                guard,
            } => {
                // Serialized domain: the exact load/combine/round/store sequence.
                let value = self.src(value)?;
                let address = self.address(layout, indices)?;
                let pointer = self.operand_pointer(*operand)?;
                let name = dtype_name(*dtype);
                // `add` rounds once at the element type. `max`/`min` select one
                // operand; floats compare widened so `bfloat` needs no overload,
                // and Metal `max`/`min` ignore a NaN operand like the reference.
                let combined = match (op, dtype) {
                    (seismic_lang::intrinsics::AtomicOp::Add, _) => {
                        format!("{name}(seismic_a + {value})")
                    }
                    (seismic_lang::intrinsics::AtomicOp::Max, DType::I32 | DType::U32) => {
                        format!("max(seismic_a, {value})")
                    }
                    (seismic_lang::intrinsics::AtomicOp::Min, DType::I32 | DType::U32) => {
                        format!("min(seismic_a, {value})")
                    }
                    (seismic_lang::intrinsics::AtomicOp::Max, _) => {
                        format!("{name}(max(float(seismic_a), float({value})))")
                    }
                    (seismic_lang::intrinsics::AtomicOp::Min, _) => {
                        format!("{name}(min(float(seismic_a), float({value})))")
                    }
                };
                let body = format!(
                    "{{ {name} seismic_a = {pointer}[{address}]; \
                     {name} seismic_b = {combined}; \
                     {pointer}[{address}] = seismic_b; }}"
                );
                self.guarded(guard, body)
            }
            AtomicDevice {
                op,
                operand,
                layout,
                indices,
                value,
                dtype,
                guard,
            } => {
                let value = self.src(value)?;
                let address = self.address(layout, indices)?;
                let pointer = self.operand_pointer(*operand)?;
                let body = match dtype {
                    DType::I32 | DType::U32 => {
                        let atomic = if *dtype == DType::I32 {
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
                            "{{ device {atomic}* seismic_p = reinterpret_cast<device {atomic}*>(&{pointer}[{address}]); \
                             atomic_fetch_{operation}_explicit(seismic_p, {value}, memory_order_relaxed); }}"
                        )
                    }
                    DType::F32 => {
                        let combined = match op {
                            AtomicOp::Add => "seismic_a + float(seismic_v)",
                            AtomicOp::Max => "max(seismic_a, float(seismic_v))",
                            AtomicOp::Min => "min(seismic_a, float(seismic_v))",
                        };
                        format!(
                            "{{ device atomic_uint* seismic_p = reinterpret_cast<device atomic_uint*>(&{pointer}[{address}]); \
                             float seismic_v = float({value}); \
                             uint seismic_expected = atomic_load_explicit(seismic_p, memory_order_relaxed); \
                             while (true) {{ float seismic_a = as_type<float>(seismic_expected); \
                               uint seismic_desired = as_type<uint>({combined}); \
                               if (atomic_compare_exchange_weak_explicit(seismic_p, &seismic_expected, seismic_desired, memory_order_relaxed, memory_order_relaxed)) break; }} }}"
                        )
                    }
                    _ => {
                        return Err(format!(
                            "compiler bug: device atomic emission received {}",
                            dtype.name()
                        ));
                    }
                };
                self.guarded(guard, body)
            }
            ReduceSerial {
                op,
                operand,
                layout,
                axis,
                axis_length,
                input_dtype,
                accumulator,
                result,
                guard,
            } => self.reduce(
                op,
                *operand,
                layout,
                *axis,
                axis_length,
                *input_dtype,
                *accumulator,
                result,
                guard,
                false,
            ),
            BlockedReduce {
                op,
                operand,
                layout,
                axis,
                axis_length,
                lanes,
                input_dtype: _,
                accumulator,
                result,
                nonempty,
            } => self.blocked_reduce(
                op,
                *operand,
                layout,
                *axis,
                axis_length,
                *lanes,
                *accumulator,
                result,
                nonempty,
            ),
            SubgroupReduce {
                op,
                operand,
                layout,
                axis,
                axis_length,
                input_dtype,
                accumulator,
                result,
                guard,
            } => self.reduce(
                op,
                *operand,
                layout,
                *axis,
                axis_length,
                *input_dtype,
                *accumulator,
                result,
                guard,
                true,
            ),
            SubgroupLaneIndex { into } => {
                let name = self.fresh_ssa(*into)?;
                self.line(format!("int {name} = int(simd_lane);"));
                Ok(())
            }
            SubgroupShuffle {
                value,
                index,
                into,
                dtype,
            } => {
                let name = self.fresh_ssa(*into)?;
                let value = self.src(value)?;
                let index = self.src(index)?;
                self.line(format!(
                    "{} {name} = simd_shuffle({value}, uint({index}));",
                    dtype_name(*dtype)
                ));
                Ok(())
            }
        }
    }

    /// The universal (serial) or subgroup reduction fold. One participant per
    /// output; the reduced axis is folded in ascending coordinates with
    /// registry accumulator/identity/tie semantics; exactly one publication.
    #[allow(clippy::too_many_arguments)]
    fn reduce(
        &mut self,
        op: &seismic_lang::intrinsics::ReduceOp,
        operand: usize,
        layout: &AccessLayout,
        axis: usize,
        axis_length: &AddrExpr,
        input_dtype: DType,
        accumulator: DType,
        result: &ReduceResult,
        guard: &Option<Guard>,
        subgroup: bool,
    ) -> Result<(), String> {
        use seismic_lang::intrinsics::ReduceOp::*;
        let operand_pointer = self.operand_pointer(operand)?;
        // Operand coordinates: the iteration's outer axes plus the
        // reduced-axis fold variable at the reduced position.
        let mut indices = Vec::new();
        let mut outer = 0usize;
        for position in 0..layout.strides.len() {
            if position == axis {
                indices.push(IndexRef::ReducedAxis);
            } else {
                indices.push(IndexRef::Axis(outer));
                outer += 1;
            }
        }
        let address = self.address(layout, &indices)?;
        let load = format!("{operand_pointer}[{address}]");
        let acc = dtype_name(accumulator);
        let length = addr_expr_string(axis_length);
        let lane = if subgroup { "simd_lane" } else { "0u" };
        let lane_stride = if subgroup { " + 32u" } else { "" };
        let fold_load = format!("{acc}({load})");
        let body = match op {
            Sum => format!(
                "{acc} seismic_acc = 0; \
                 for (uint seismic_k = {lane}; seismic_k < {length}; seismic_k{lane_stride}) \
                 {{ seismic_acc = {acc}(seismic_acc + {fold_load}); }}"
            ),
            Max => {
                if subgroup {
                    return Err(
                        "compiler bug: subgroup simd_max emission is pending; only simd_sum \
                         subgroup reductions are offered"
                            .into(),
                    );
                }
                format!(
                    "{acc} seismic_acc = {fold_load}; \
                     for (uint seismic_k = 1u; seismic_k < {length}; seismic_k += 1u) \
                     {{ seismic_acc = max(seismic_acc, {fold_load}); }}"
                )
            }
            Min => {
                if subgroup {
                    return Err(
                        "compiler bug: subgroup simd_min emission is pending; only simd_sum \
                         subgroup reductions are offered"
                            .into(),
                    );
                }
                format!(
                    "{acc} seismic_acc = {fold_load}; \
                     for (uint seismic_k = 1u; seismic_k < {length}; seismic_k += 1u) \
                     {{ seismic_acc = min(seismic_acc, {fold_load}); }}"
                )
            }
            Argmax => {
                if subgroup {
                    unreachable!("argmax never reassociates")
                }
                format!(
                    "{acc} seismic_acc = {fold_load}; int seismic_best = 0; \
                     for (uint seismic_k = 1u; seismic_k < {length}; seismic_k += 1u) \
                     {{ {acc} seismic_v = {fold_load}; \
                     if (seismic_v > seismic_acc) {{ seismic_acc = seismic_v; \
                     seismic_best = int(seismic_k); }} }}"
                )
            }
        };
        let collective = if subgroup {
            format!("seismic_acc = simd_sum(seismic_acc);")
        } else {
            String::new()
        };
        let publish = match result {
            ReduceResult::Scalar { value } => {
                let resolved = self.resolved_slot_of(*value)?;
                self.slots_written.0.insert(resolved.0);
                format!(
                    "seismic_slots[{}] = {};",
                    resolved.0,
                    slot_write("seismic_acc", accumulator)
                )
            }
            ReduceResult::Tensor {
                dst,
                layout: result_layout,
                indices,
                dtype,
            } => {
                let value = match op {
                    Argmax => "int(seismic_best)".to_string(),
                    _ => {
                        if *dtype != accumulator {
                            format!("{dtype}(seismic_acc)", dtype = dtype_name(*dtype))
                        } else {
                            "seismic_acc".to_string()
                        }
                    }
                };
                let (pointer, address) = self.dst_location(dst, result_layout, indices)?;
                format!("{pointer}[{address}] = {value};")
            }
        };
        let _ = input_dtype;
        self.guarded(guard, format!("{{ {body} {collective} {publish} }}"))
    }

    /// The interleaved blocked cover: this visit's lane (the iteration's
    /// last axis) folds its strided share ascending, publishes its partial
    /// into workgroup storage, and lane zero combines all partials in lane
    /// order (a one-level tree).
    fn blocked_reduce(
        &mut self,
        op: &seismic_lang::intrinsics::ReduceOp,
        operand: usize,
        layout: &AccessLayout,
        axis: usize,
        axis_length: &AddrExpr,
        lanes: u32,
        accumulator: DType,
        result: &ReduceResult,
        nonempty: &Option<Guard>,
    ) -> Result<(), String> {
        use seismic_lang::intrinsics::ReduceOp::*;
        let operand_pointer = self.operand_pointer(operand)?;
        let rank = layout.strides.len();
        let lane_axis = rank.saturating_sub(1);
        let mut indices = Vec::new();
        let mut outer = 0usize;
        for position in 0..rank {
            if position == axis {
                indices.push(IndexRef::ReducedAxis);
            } else {
                indices.push(IndexRef::Axis(outer));
                outer += 1;
            }
        }
        let address = self.address(layout, &indices)?;
        let load = format!("{operand_pointer}[{address}]");
        let acc = dtype_name(accumulator);
        let length = addr_expr_string(axis_length);
        let lane = format!("c{lane_axis}");
        let fold_load = format!("{acc}({load})");
        let identity = match op {
            Sum => "0".to_string(),
            // An empty lane share folds to the identity; the nonempty
            // precondition guard covers a fully empty axis.
            Max => "-INFINITY".to_string(),
            Min => "INFINITY".to_string(),
            Argmax => unreachable!("argmax is rejected before blocked reduction emission"),
        };
        let fold = match op {
            Sum => format!(
                "{acc} seismic_acc = {identity}; \
                 for (uint seismic_k = {lane}; seismic_k < {length}; seismic_k += {lanes}u) \
                 {{ seismic_acc = {acc}(seismic_acc + {fold_load}); }}"
            ),
            Max => format!(
                "{acc} seismic_acc = {identity}; \
                 for (uint seismic_k = {lane}; seismic_k < {length}; seismic_k += {lanes}u) \
                 {{ seismic_acc = max(seismic_acc, {fold_load}); }}"
            ),
            Min => format!(
                "{acc} seismic_acc = {identity}; \
                 for (uint seismic_k = {lane}; seismic_k < {length}; seismic_k += {lanes}u) \
                 {{ seismic_acc = min(seismic_acc, {fold_load}); }}"
            ),
            Argmax => {
                return Err(
                    "compiler bug: argmax never reassociates; a blocked cover is not offered"
                        .into(),
                );
            }
        };
        let combine = match op {
            Sum => "seismic_acc = seismic_acc + seismic_wg[seismic_i];".to_string(),
            Max => "seismic_acc = max(seismic_acc, seismic_wg[seismic_i]);".to_string(),
            Min => "seismic_acc = min(seismic_acc, seismic_wg[seismic_i]);".to_string(),
            Argmax => unreachable!("checked above"),
        };
        let publish = match result {
            ReduceResult::Scalar { value } => {
                let resolved = self.resolved_slot_of(*value)?;
                self.slots_written.0.insert(resolved.0);
                format!(
                    "seismic_slots[{}] = {};",
                    resolved.0,
                    slot_write("seismic_acc", accumulator)
                )
            }
            ReduceResult::Tensor {
                dst,
                layout: result_layout,
                indices,
                dtype,
            } => {
                let value = if *dtype != accumulator {
                    format!("{dtype}(seismic_acc)", dtype = dtype_name(*dtype))
                } else {
                    "seismic_acc".to_string()
                };
                let (pointer, address) = self.dst_location(dst, result_layout, indices)?;
                format!("{pointer}[{address}] = {value};")
            }
        };
        let body = format!(
            "{{ {fold} \
             seismic_wg[{lane}] = seismic_acc; \
             threadgroup_barrier(mem_flags::mem_threadgroup); \
             if ({lane} == 0u) {{ \
               {acc} seismic_acc = 0; \
               for (uint seismic_i = 0u; seismic_i < {lanes}u; seismic_i += 1u) {{ {combine} }} \
               {publish} \
             }} }}"
        );
        self.guarded(nonempty, body)
    }

    // -- expression helpers ---------------------------------------------------

    fn fresh_ssa(&mut self, into: ValueRef) -> Result<String, String> {
        match into {
            ValueRef::Ssa(n) => Ok(format!("s{n}")),
            ValueRef::Operand(_) | ValueRef::Axis(_) => {
                Err("compiler bug: an opcode result must be a fresh SSA slot".into())
            }
        }
    }

    fn values(&self, references: &[ValueRef]) -> Result<Vec<String>, String> {
        references.iter().map(|r| self.value(*r)).collect()
    }

    fn value(&self, reference: ValueRef) -> Result<String, String> {
        match reference {
            ValueRef::Ssa(n) => Ok(format!("s{n}")),
            ValueRef::Axis(n) => Ok(format!("c{n}")),
            ValueRef::Operand(k) => {
                let (_, transport) = self
                    .mapped
                    .get(k)
                    .ok_or_else(|| format!("operand binding#{k} is absent"))?;
                match transport {
                    seismic_realization::executable::ResolvedTransport::Kernel(_) => Err(format!(
                        "compiler bug: operand binding#{k} is kernel-local SSA; the \
                             strategy must reference it as SSA"
                    )),
                    seismic_realization::executable::ResolvedTransport::ExecutorScalar(
                        ResolvedExecutorScalar::Slot { slot, dtype },
                    ) => Ok(slot_read(&slot.0.to_string(), *dtype)),
                    seismic_realization::executable::ResolvedTransport::ExecutorScalar(
                        ResolvedExecutorScalar::Abi { .. },
                    ) => Err(
                        "compiler bug: an ABI-scalar transport reached a kernel; ABI scalar \
                         fields need ordinal-qualified paths (interface request)"
                            .into(),
                    ),
                    _ => Err(format!("operand binding#{k} does not transport a scalar")),
                }
            }
        }
    }

    fn src(&self, source: &Src) -> Result<String, String> {
        match source {
            Src::Scalar(reference) => self.value(*reference),
            Src::Element {
                operand,
                layout,
                indices,
            } => {
                let address = self.address(layout, indices)?;
                let pointer = self.operand_pointer(*operand)?;
                Ok(format!("{pointer}[{address}]"))
            }
        }
    }

    fn operand_pointer(&self, operand: usize) -> Result<String, String> {
        let (_, transport) = self
            .mapped
            .get(operand)
            .cloned()
            .ok_or_else(|| format!("operand binding#{operand} is absent"))?;
        match transport {
            seismic_realization::executable::ResolvedTransport::Storage(views) => {
                let view = views.first();
                self.pointer_of(view.storage)
            }
            _ => Err(format!(
                "compiler bug: tensor operand binding#{operand} does not transport storage"
            )),
        }
    }

    fn plane_pointer(&self, operand: usize, plane: usize) -> Result<String, String> {
        let (_, transport) = self
            .mapped
            .get(operand)
            .cloned()
            .ok_or_else(|| format!("operand binding#{operand} is absent"))?;
        match transport {
            seismic_realization::executable::ResolvedTransport::Storage(views) => {
                let view = views
                    .iter()
                    .nth(plane)
                    .ok_or_else(|| format!("plane#{plane} is absent from the operand transport"))?;
                self.pointer_of(view.storage)
            }
            _ => Err("compiler bug: a packed operand transports storage planes".into()),
        }
    }

    fn pointer_of(&self, storage: ResolvedStorageId) -> Result<String, String> {
        self.pointers
            .get(&storage)
            .cloned()
            .ok_or_else(|| format!("resolved storage#{} has no launch binding", storage.0))
    }

    fn binding_dtype(&self, operand: usize) -> Result<DType, String> {
        let (_, transport) = self
            .mapped
            .get(operand)
            .cloned()
            .ok_or_else(|| format!("operand binding#{operand} is absent"))?;
        match transport {
            seismic_realization::executable::ResolvedTransport::ExecutorScalar(
                ResolvedExecutorScalar::Slot { dtype, .. },
            ) => Ok(dtype),
            _ => Err("compiler bug: a scalar operand transports through a slot".into()),
        }
    }

    fn tuple_slot(
        &self,
        operand: usize,
        index: usize,
    ) -> Result<seismic_realization::executable::ResolvedExecutorScalarId, String> {
        let (_, transport) = self
            .mapped
            .get(operand)
            .cloned()
            .ok_or_else(|| format!("operand binding#{operand} is absent"))?;
        match transport {
            seismic_realization::executable::ResolvedTransport::Tuple(items) => {
                match items.iter().nth(index) {
                    Some(seismic_realization::executable::ResolvedTransport::ExecutorScalar(
                        ResolvedExecutorScalar::Slot { slot, .. },
                    )) => Ok(*slot),
                    _ => Err("compiler bug: a tuple leaf transports through a slot".into()),
                }
            }
            _ => Err("compiler bug: a tuple operand transports as a tuple".into()),
        }
    }

    /// The resolved executor slot of one produced scalar value, from the
    /// launch's own value bindings.
    fn resolved_slot_of(
        &self,
        value: GraphValueId,
    ) -> Result<seismic_realization::executable::ResolvedExecutorScalarId, String> {
        match self.transport_of_value(value)? {
            seismic_realization::executable::ResolvedTransport::ExecutorScalar(
                ResolvedExecutorScalar::Slot { slot, .. },
            ) => Ok(*slot),
            _ => Err(format!(
                "compiler bug: produced scalar value#{value:?} does not transport through a \
                 slot"
            )),
        }
    }

    /// The resolved transport bound for one value of the mapped block.
    fn transport_of_value(
        &self,
        value: GraphValueId,
    ) -> Result<&seismic_realization::executable::ResolvedTransport, String> {
        self.mapped
            .iter()
            .find(|(bound, _)| *bound == value)
            .map(|(_, transport)| transport)
            .ok_or_else(|| {
                format!(
                    "compiler bug: produced value#{value:?} has no binding in its launch; the \
                     resolver binds node outputs after inputs"
                )
            })
    }

    /// The (pointer, address) of one destination.
    fn dst_location(
        &self,
        dst: &Dst,
        layout: &AccessLayout,
        indices: &[IndexRef],
    ) -> Result<(String, String), String> {
        match dst {
            Dst::Operand(operand) => {
                let address = self.address(layout, indices)?;
                let pointer = self.operand_pointer(*operand)?;
                Ok((pointer, address))
            }
            Dst::Produced(value) => {
                let address = self.address(layout, indices)?;
                match self.transport_of_value(*value)? {
                    seismic_realization::executable::ResolvedTransport::Storage(views) => {
                        let view = views.first();
                        let pointer = self.pointer_of(view.storage)?;
                        Ok((pointer, address))
                    }
                    _ => Err(format!(
                        "compiler bug: produced tensor value#{value:?} does not transport \
                         storage"
                    )),
                }
            }
        }
    }

    fn dst_base(&self, dst: &Dst) -> Result<String, String> {
        match dst {
            Dst::Operand(operand) => self.operand_pointer(*operand),
            Dst::Produced(value) => match self.transport_of_value(*value)? {
                seismic_realization::executable::ResolvedTransport::Storage(views) => {
                    self.pointer_of(views.first().storage)
                }
                _ => Err(format!(
                    "compiler bug: produced tensor value#{value:?} does not transport storage"
                )),
            },
        }
    }

    /// The element address of one layout at the given indices.
    fn address(&self, layout: &AccessLayout, indices: &[IndexRef]) -> Result<String, String> {
        let mut terms = Vec::new();
        for (stride, value) in &layout.offset {
            terms.push(format!(
                "{} * ({})",
                addr_expr_string(stride),
                self.value(*value)?
            ));
        }
        for (axis, index) in indices.iter().enumerate() {
            let stride = layout
                .strides
                .get(axis)
                .ok_or_else(|| format!("index#{axis} has no layout stride"))?;
            let factor = addr_expr_string(stride);
            let coordinate = match index {
                IndexRef::Value(value) => self.value(*value)?,
                IndexRef::Axis(n) => format!("c{n}"),
                IndexRef::ReducedAxis => "seismic_k".into(),
            };
            if factor == "1" {
                terms.push(format!("({coordinate})"));
            } else {
                terms.push(format!("{factor} * ({coordinate})"));
            }
        }
        if terms.is_empty() {
            Ok("0".into())
        } else {
            Ok(terms.join(" + "))
        }
    }

    /// Emit one guarded statement: every predicate must hold, else the first
    /// error is recorded in the planned status field.
    fn guarded(&mut self, guard: &Option<Guard>, body: String) -> Result<(), String> {
        match guard {
            None => {
                self.line(body);
                Ok(())
            }
            Some(guard) => {
                let mut predicates = Vec::new();
                for predicate in &guard.predicates {
                    predicates.push(self.predicate(predicate)?);
                }
                let condition = if predicates.is_empty() {
                    "true".into()
                } else {
                    predicates.join(" && ")
                };
                self.line(format!(
                    "if ({condition}) {{ {body} }} else {{ \
                     if (seismic_status[{}] == 0u) seismic_status[{}] = {}; }}",
                    guard.status,
                    guard.status,
                    guard_code(&guard.predicates)
                ));
                Ok(())
            }
        }
    }

    fn predicate(&self, predicate: &GuardPredicate) -> Result<String, String> {
        Ok(match predicate {
            GuardPredicate::IndexInBounds { index, extent } => {
                let index = self.value(*index)?;
                format!(
                    "({index} >= 0 && uint({index}) < {})",
                    addr_expr_string(extent)
                )
            }
            GuardPredicate::RangeInBounds { start, end, extent } => {
                let start = self.value(*start)?;
                let end = self.value(*end)?;
                let bound = addr_expr_string(extent);
                format!(
                    "({start} >= 0 && uint({start}) < {bound} && {end} >= 0 && uint({end}) < {bound})"
                )
            }
            GuardPredicate::DivisorNonZero { value } => {
                format!("({} != 0)", self.value(*value)?)
            }
            GuardPredicate::DivisionSafe { lhs, rhs } => format!(
                "({rhs} != 0 && ({lhs} != (-2147483647 - 1) || {rhs} != -1))",
                lhs = self.value(*lhs)?,
                rhs = self.value(*rhs)?,
            ),
            GuardPredicate::ShiftInRange { value } => {
                let value = self.value(*value)?;
                format!("({value} >= 0 && {value} < 32)")
            }
            GuardPredicate::ProductFits { factors, bits } => {
                if *bits >= 64 {
                    "true".into()
                } else {
                    let product = factors
                        .iter()
                        .map(addr_expr_string)
                        .collect::<Vec<_>>()
                        .join(" * ");
                    format!("((1u{product}) < (1ul << {bits}))")
                }
            }
            GuardPredicate::ExtentPositive { extent } => {
                format!("({} > 0)", addr_expr_string(extent))
            }
        })
    }
}
fn extent_addr_of(extent: &seismic_lang::types::ExtentExpr) -> Result<AddrExpr, String> {
    match extent {
        seismic_lang::types::ExtentExpr::Static(n) => Ok(AddrExpr::Const(*n)),
        seismic_lang::types::ExtentExpr::Runtime(id) => Ok(AddrExpr::Extent(*id)),
        seismic_lang::types::ExtentExpr::Sym(sym) => sym
            .as_constant()
            .and_then(|c| u64::try_from(c).ok())
            .map(AddrExpr::Const)
            .ok_or_else(|| "unresolved symbolic extent in the iteration map".to_string()),
    }
}

fn addr_expr_string(expr: &AddrExpr) -> String {
    match expr {
        AddrExpr::Const(n) => format!("{n}u"),
        AddrExpr::Extent(id) => format!("uint(seismic_extents[{}])", id.0),
        AddrExpr::Mul(a, b) => format!("({} * {})", addr_expr_string(a), addr_expr_string(b)),
    }
}

fn expr_string(expr: &ExecutionExpr) -> String {
    match expr {
        ExecutionExpr::Const(n) => format!("{n}u"),
        ExecutionExpr::Extent(id) => format!("uint(seismic_extents[{}])", id.0),
        ExecutionExpr::AbiScalar { .. } => "0u /* abi scalar */".into(),
        ExecutionExpr::ExecutorScalar(slot) => format!("seismic_slots[{}]", slot.0),
        ExecutionExpr::Add(a, b) => format!("({} + {})", expr_string(a), expr_string(b)),
        ExecutionExpr::Sub(a, b) => format!("({} - {})", expr_string(a), expr_string(b)),
        ExecutionExpr::Mul(a, b) => format!("({} * {})", expr_string(a), expr_string(b)),
        ExecutionExpr::CeilDiv(a, b) => format!(
            "({} + {} - 1) / {}",
            expr_string(a),
            expr_string(b),
            expr_string(b)
        ),
        ExecutionExpr::Div(a, b) => format!("({} / {})", expr_string(a), expr_string(b)),
        ExecutionExpr::Rem(a, b) => format!("({} % {})", expr_string(a), expr_string(b)),
        ExecutionExpr::Min(a, b) => format!("min({}, {})", expr_string(a), expr_string(b)),
    }
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

fn const_expr(value: ConstValue, dtype: DType) -> String {
    match (value, dtype) {
        (ConstValue::Int(v), DType::I32) => format!("int({v})"),
        (ConstValue::Int(v), DType::U32) => format!("uint({v})"),
        (ConstValue::Int(v), DType::Bool) => format!("{}", v != 0),
        (ConstValue::Float(bits), DType::F32) => {
            format!("as_type<float>(uint({bits:#010x}u))")
        }
        (ConstValue::Float(bits), DType::F16) => {
            format!("as_type<half>(ushort({bits:#010x} & 0xffffu))")
        }
        (ConstValue::Bool(b), _) => format!("{}", b),
        _ => "0".into(),
    }
}

fn slot_read(slot: &str, dtype: DType) -> String {
    match dtype {
        DType::F32 => format!("as_type<float>(seismic_slots[{slot}])"),
        DType::F16 => format!("as_type<half>(ushort(seismic_slots[{slot}] & 0xffffu))"),
        DType::I32 => format!("as_type<int>(seismic_slots[{slot}])"),
        DType::U32 => format!("seismic_slots[{slot}]"),
        DType::Bool => format!("seismic_slots[{slot}] != 0u"),
        DType::BF16 => format!("as_type<bfloat>(ushort(seismic_slots[{slot}] & 0xffffu))"),
    }
}

fn slot_write(value: &str, dtype: DType) -> String {
    match dtype {
        DType::F32 => format!("as_type<uint>({value})"),
        DType::F16 => format!("uint(as_type<ushort>(half({value})))"),
        DType::I32 => format!("as_type<uint>({value})"),
        DType::U32 => format!("{value}"),
        DType::Bool => format!("uint({value})"),
        DType::BF16 => format!("uint(as_type<ushort>(bfloat({value})))"),
    }
}

fn rel_symbol(op: crate::physical::RelOp) -> &'static str {
    use crate::physical::RelOp::*;
    match op {
        Eq => "==",
        Ne => "!=",
        Lt => "<",
        Le => "<=",
        Gt => ">",
        Ge => ">=",
    }
}

fn comparison_cast(dtype: DType) -> &'static str {
    match dtype {
        DType::U32 => "uint",
        _ => "int",
    }
}

fn int_symbol(op: crate::physical::IntOp) -> &'static str {
    use crate::physical::IntOp::*;
    match op {
        Add => "+",
        Sub => "-",
        Mul => "*",
        BitAnd => "&",
        BitOr => "|",
        BitXor => "^",
    }
}

fn math_call(
    op: seismic_lang::intrinsics::MathOp,
    args: &[String],
    dtype: DType,
) -> Result<String, String> {
    use seismic_lang::intrinsics::MathOp::*;
    let name = dtype_name(dtype);
    Ok(match op {
        Fma => format!("fma({}, {}, {})", args[0], args[1], args[2]),
        Exp => format!("exp({}({}))", name, args[0]),
        ExpFast => format!("fast::exp({}({}))", name, args[0]),
        Rsqrt => format!("rsqrt({}({}))", name, args[0]),
        Sqrt => format!("sqrt({}({}))", name, args[0]),
        Log => format!("log({}({}))", name, args[0]),
        Sin => format!("sin({}({}))", name, args[0]),
        Cos => format!("cos({}({}))", name, args[0]),
        Abs => match dtype {
            d if d.is_int() => format!("abs({})", args[0]),
            _ => format!("fabs({}({}))", name, args[0]),
        },
        Max => format!("max({}, {})", args[0], args[1]),
        Min => format!("min({}, {})", args[0], args[1]),
    })
}

fn cast_expr(from: DType, to: DType, operand: &str) -> String {
    if from.is_int() && to.is_int() {
        // Integer-to-integer casts preserve the low 32 bits.
        return format!("as_type<{}>(as_type<uint>({operand}))", dtype_name(to));
    }
    match to {
        DType::Bool => format!("{operand} != 0"),
        DType::I32 => format!("int(clamp(trunc(float({operand})), -2147483648.0f, 2147483647.0f))"),
        DType::U32 => format!("uint(clamp(trunc(float({operand})), 0.0f, 4294967295.0f))"),
        DType::F32 | DType::F16 | DType::BF16 => format!("{}({operand})", dtype_name(to)),
    }
}

fn guard_code(predicates: &[GuardPredicate]) -> u32 {
    // Status codes mirror the safety-kind taxonomy (first error wins).
    for predicate in predicates {
        return match predicate {
            GuardPredicate::IndexInBounds { .. } => 1,
            GuardPredicate::RangeInBounds { .. } => 2,
            GuardPredicate::DivisorNonZero { .. } => 3,
            GuardPredicate::DivisionSafe { .. } => 4,
            GuardPredicate::ShiftInRange { .. } => 5,
            GuardPredicate::ProductFits { .. } => 6,
            GuardPredicate::ExtentPositive { .. } => 7,
        };
    }
    0
}
