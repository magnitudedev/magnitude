//! Encoding boundary for one fully resolved CUDA launch: exhaustive PTX
//! emission over `CudaOp`, and assembly of
//! the native artifact mirroring the resolved structured schedule tree.
//!
//! The emitter is mechanical: it makes no allocation, geometry,
//! synchronization, algorithm, or precision decision, and it never rejects
//! a selected opcode — every guard in an opcode comes from a discharge
//! decided during alternative construction. Emission failures are compiler
//! bugs only: an unresolved identity or a launch beyond the kernel parameter ABI.
//!
//! `Backend::encode_launch` returns the resolved launch id as its encoded
//! form; the real encoding happens once in `assemble`. One recursive walk
//! mirrors the resolved structured execution tree and encodes each launch once.

use crate::physical::CudaDialect;
use seismic_compiler::pipeline::EncodedPlan;
use seismic_lang::{
    logical::GraphValueId,
    types::{DType, RuntimeExtentId},
};
use seismic_realization::executable::{
    self, BufferBinding, ResolvedExecutorScalar, ResolvedKernelStep, ResolvedLaunch,
    ResolvedLaunchId, ResolvedPlan, ResolvedStep, ResolvedStorageId, ResolvedTransport,
};
use std::collections::{BTreeMap, BTreeSet};

/// The CUDA kernel parameter ABI limit (bytes).
pub const MAX_KERNEL_PARAMETER_BYTES: usize = 32_764;

/// Encoded form of one launch: its resolved identity (the mechanical PTX
/// emission happens once, in `assemble`).
pub type EncodedLaunch = ResolvedLaunchId;

/// Encode one resolved launch: the identity marker.
pub fn encode_launch(launch: &ResolvedLaunch<CudaDialect>) -> Result<EncodedLaunch, String> {
    Ok(launch.id)
}

/// One encoded kernel parameter.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CudaParam {
    /// Device pointer of one resolved storage (a caller buffer or an arena
    /// offset; the runtime resolves it from the plan's storage table).
    Storage(ResolvedStorageId),
    /// A root-ABI scalar passed by value, located by byte offset in the
    /// invocation scalar block.
    AbiScalar { offset: u64, dtype: DType },
    /// The retained runtime value of one runtime extent, passed by value.
    Extent(RuntimeExtentId),
    /// Base pointer of the executor scalar slot block (8 bytes per slot).
    SlotBlock,
    /// Base pointer of the status block (4 bytes per status field).
    StatusBlock,
    /// Base pointer of the compiler-owned result scalar block (8 bytes per
    /// result field).
    ResultBlock,
    /// One `seismic_math` software call: two `.b32` parameter blocks.
    MathCall { symbol: String },
}

/// One encoded launch: its mechanical parameter list and PTX text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CudaLaunch {
    pub id: ResolvedLaunchId,
    pub name: String,
    pub params: Vec<CudaParam>,
    pub block: u64,
    /// The retained zero-work launch condition; the runtime skips the
    /// launch when it evaluates to zero (zero grids are never submitted).
    pub work_items: executable::ExecutionExpr,
    pub ptx: String,
}

/// Mirror of one resolved storage for the runtime.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StorageMirror {
    pub placement: executable::ResolvedStoragePlacement,
    pub bytes: u64,
    pub alignment: u64,
}

/// Where one encoded control scalar reads its value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ControlSource {
    /// A fixed constant (an absorbed single-visit repeat endpoint).
    Const(i64),
    /// A root-ABI scalar field, located by byte offset.
    Abi { offset: u64, dtype: DType },
    /// An executor scalar slot (device-produced; read back synchronously).
    Slot(u64),
    /// A computed control scalar: a retained execution expression the
    /// runtime evaluates against extents, ABI scalars, and slot words.
    Computed(executable::ExecutionExpr),
}

/// One encoded step of the execution tree mirror.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EncodedBodyStep {
    Launch(ResolvedLaunchId),
    Call(Vec<EncodedBodyStep>),
    If {
        condition: ControlSource,
        then_steps: Vec<EncodedBodyStep>,
        else_steps: Vec<EncodedBodyStep>,
    },
    Repeat {
        start: ControlSource,
        end: ControlSource,
        /// The executor slot word the runtime rebinds to each visit's
        /// binder value.
        binder: u64,
        body: Vec<EncodedBodyStep>,
    },
}

/// The native artifact: encoded launches plus the retained runtime facts
/// (storage table, ABI mirror, runtime extent expressions, slot and status
/// sizing). There is no retry or candidate surface.
#[derive(Clone, Debug)]
pub struct Emitted {
    pub name: String,
    pub launches: Vec<CudaLaunch>,
    pub root: Vec<EncodedBodyStep>,
    pub storages: BTreeMap<u64, StorageMirror>,
    pub abi_buffers: Vec<BufferBinding>,
    pub abi_scalar_fields: Vec<(String, DType, usize)>,
    pub abi_scalar_bytes: usize,
    /// Discharged runtime-check and strategy-declared status fields.
    pub status_words: u64,
    pub slot_words: u64,
    pub arena_bytes: u64,
    pub runtime_extents: Vec<(RuntimeExtentId, executable::ExecutionExpr)>,
    pub alias_rules: Vec<executable::AliasRule>,
    pub toolchain_fingerprint: String,
}

fn bug(message: String) -> String {
    format!("compiler bug: {message}")
}

// ---------------------------------------------------------------------------
// One recursive walk: mirror + per-launch encoding
// ---------------------------------------------------------------------------

struct Assembly {
    launches: Vec<CudaLaunch>,
    /// Every graph value's resolved transport, collected from all mapped
    /// launches of the plan (operands of one launch may be produced by
    /// another).
    values: BTreeMap<GraphValueId, ResolvedTransport>,
}

/// Assemble the native artifact: encode each
/// resolved launch once, and mirror the structured execution tree.
pub fn assemble(encoded: EncodedPlan<CudaDialect, EncodedLaunch>) -> Result<Emitted, String> {
    let plan = &encoded.resolved;
    let mut assembly = Assembly {
        launches: Vec::new(),
        values: BTreeMap::new(),
    };
    collect_plan_values(&plan.entry.schedule, &mut assembly.values);
    let root = assembly.schedule(&plan.entry.schedule)?;
    let storages = plan
        .storage
        .ids()
        .zip(plan.storage.iter())
        .map(|(id, storage)| {
            (
                id.0,
                StorageMirror {
                    placement: storage.placement,
                    bytes: storage.bytes,
                    alignment: storage.alignment,
                },
            )
        })
        .collect();
    let discharged = plan
        .abi
        .status
        .as_ref()
        .map(|s| s.fields.len() as u64)
        .unwrap_or(0);
    Ok(Emitted {
        name: plan.identity.entry.clone(),
        launches: assembly.launches,
        root,
        storages,
        abi_buffers: plan.abi.buffers.clone(),
        abi_scalar_fields: plan
            .abi
            .scalars
            .fields
            .iter()
            .map(|field| {
                (
                    field.parameter.name.clone(),
                    field.parameter.dtype,
                    field.offset,
                )
            })
            .collect(),
        abi_scalar_bytes: plan.abi.scalars.bytes,
        status_words: discharged,
        slot_words: count_slot_words(plan),
        arena_bytes: plan.internal_arena.bytes,
        runtime_extents: plan
            .runtime_extents
            .ids()
            .zip(plan.runtime_extents.iter())
            .map(|(id, expr)| (id, expr.clone()))
            .collect(),
        alias_rules: plan.abi.alias_rules.clone(),
        toolchain_fingerprint: plan.identity.toolchain_fingerprint.clone(),
    })
}

impl Assembly {
    fn schedule(
        &mut self,
        resolved: &executable::ResolvedSchedule<CudaDialect>,
    ) -> Result<Vec<EncodedBodyStep>, String> {
        let mut out = Vec::new();
        for resolved_step in resolved.steps.iter() {
            out.push(match resolved_step {
                ResolvedStep::Launch(resolved_launch) => {
                    self.launch(resolved_launch)?;
                    EncodedBodyStep::Launch(resolved_launch.id)
                }
                ResolvedStep::Call(resolved_call) => {
                    EncodedBodyStep::Call(self.schedule(&resolved_call.body.schedule)?)
                }
                ResolvedStep::If(resolved_if) => EncodedBodyStep::If {
                    condition: self.control(&resolved_if.condition)?,
                    then_steps: self.schedule(&resolved_if.then_schedule)?,
                    else_steps: self.schedule(&resolved_if.else_schedule)?,
                },
                ResolvedStep::Repeat(resolved_repeat) => EncodedBodyStep::Repeat {
                    start: self.control(&resolved_repeat.range.start)?,
                    end: self.control(&resolved_repeat.range.end)?,
                    binder: resolved_repeat.binder.0,
                    body: self.schedule(&resolved_repeat.body)?,
                },
            });
        }
        Ok(out)
    }

    /// Resolve one control scalar to its encoded source: an ABI scalar to
    /// its invocation-block offset; a device slot to its slot word; a
    /// computed expression to its retained expression.
    fn control(&self, resolved: &ResolvedExecutorScalar) -> Result<ControlSource, String> {
        match resolved {
            ResolvedExecutorScalar::Abi { offset, dtype, .. } => Ok(ControlSource::Abi {
                offset: *offset,
                dtype: *dtype,
            }),
            ResolvedExecutorScalar::Result { .. } => Err(bug(
                "a result scalar cannot be used as an executor control value".into(),
            )),
            ResolvedExecutorScalar::Slot { slot, .. } => Ok(ControlSource::Slot(slot.0)),
            ResolvedExecutorScalar::Computed { expr, .. } => {
                Ok(ControlSource::Computed(expr.clone()))
            }
        }
    }

    fn launch(&mut self, resolved: &ResolvedLaunch<CudaDialect>) -> Result<(), String> {
        let encoded = encode_resolved_launch(resolved, &self.values)?;
        self.launches.push(encoded);
        Ok(())
    }
}

/// The number of 8-byte executor slot words the plan references.
fn count_slot_words(plan: &ResolvedPlan<CudaDialect>) -> u64 {
    let mut slots: BTreeSet<u64> = BTreeSet::new();
    collect_plan_slots(&plan.entry.schedule, &mut slots);
    slots.len() as u64
}

fn collect_plan_slots(
    schedule: &executable::ResolvedSchedule<CudaDialect>,
    slots: &mut BTreeSet<u64>,
) {
    for step in schedule.steps.iter() {
        match step {
            ResolvedStep::Launch(launch) => {
                for kernel_step in launch.kernel.steps.iter() {
                    match kernel_step {
                        ResolvedKernelStep::PublishScalar { slot } => {
                            slots.insert(slot.0);
                        }
                        ResolvedKernelStep::Mapped { bindings, .. } => {
                            for (_, transport) in bindings {
                                collect_transport_slots(transport, slots);
                            }
                        }
                        _ => {}
                    }
                }
            }
            ResolvedStep::Call(call) => {
                for transport in call.boundary.inputs.values() {
                    collect_transport_slots(transport, slots);
                }
                for transport in call.boundary.results.values() {
                    collect_transport_slots(transport, slots);
                }
                collect_plan_slots(&call.body.schedule, slots);
            }
            ResolvedStep::If(if_step) => {
                if let ResolvedExecutorScalar::Slot { slot, .. } = &if_step.condition {
                    slots.insert(slot.0);
                }
                collect_plan_slots(&if_step.then_schedule, slots);
                collect_plan_slots(&if_step.else_schedule, slots);
            }
            ResolvedStep::Repeat(repeat) => {
                for scalar in [&repeat.range.start, &repeat.range.end] {
                    if let ResolvedExecutorScalar::Slot { slot, .. } = scalar {
                        slots.insert(slot.0);
                    }
                }
                slots.insert(repeat.binder.0);
                for carry in &repeat.carried {
                    collect_transport_slots(carry, slots);
                }
                collect_plan_slots(&repeat.body, slots);
            }
        }
    }
}

fn collect_transport_slots(transport: &ResolvedTransport, slots: &mut BTreeSet<u64>) {
    match transport {
        ResolvedTransport::ExecutorScalar(ResolvedExecutorScalar::Slot { slot, .. }) => {
            slots.insert(slot.0);
        }
        ResolvedTransport::Tuple(items) => {
            for item in items.iter() {
                collect_transport_slots(item, slots);
            }
        }
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// The PTX emitter (crate::emitter, declared in lib.rs)
// ---------------------------------------------------------------------------

/// Encode one resolved launch into PTX, delegating to the emitter module.
fn encode_resolved_launch(
    launch: &ResolvedLaunch<CudaDialect>,
    values: &BTreeMap<GraphValueId, ResolvedTransport>,
) -> Result<CudaLaunch, String> {
    crate::emitter::encode(launch, values)
}

/// Collect every graph value's resolved transport from all mapped launches
/// and call boundaries of the plan.
fn collect_plan_values(
    schedule: &executable::ResolvedSchedule<CudaDialect>,
    values: &mut BTreeMap<GraphValueId, ResolvedTransport>,
) {
    for step in schedule.steps.iter() {
        match step {
            ResolvedStep::Launch(launch) => {
                for kernel_step in launch.kernel.steps.iter() {
                    if let ResolvedKernelStep::Mapped { bindings, .. } = kernel_step {
                        for (value, transport) in bindings {
                            values.insert(*value, transport.clone());
                        }
                    }
                }
            }
            ResolvedStep::Call(call) => {
                for (path, transport) in &call.boundary.inputs {
                    let _ = path;
                    collect_boundary_values(transport, values);
                }
                for (path, transport) in &call.boundary.results {
                    let _ = path;
                    collect_boundary_values(transport, values);
                }
                collect_plan_values(&call.body.schedule, values);
            }
            ResolvedStep::If(if_step) => {
                collect_plan_values(&if_step.then_schedule, values);
                collect_plan_values(&if_step.else_schedule, values);
            }
            ResolvedStep::Repeat(repeat) => {
                collect_plan_values(&repeat.body, values);
            }
        }
    }
}

fn collect_boundary_values(
    _transport: &ResolvedTransport,
    values: &mut BTreeMap<GraphValueId, ResolvedTransport>,
) {
    let _ = values;
}

/// Public re-exports used by the runtime glue.
pub use crate::emitter::{ptx_type, storage_type};
