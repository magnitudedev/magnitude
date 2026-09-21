//! The CPU executor: one prepared invocation against the sealed native
//! artifact, over direct dense handles, with only `ExecutionFailure`.
//!
//! Invoked from runtime submission: `CpuExecutor::new(&mut Workers).execute(
//! &ExecutionInputs { physical, native, buffers, values })`. The backend
//! crate cannot name the runtime crate's prepared-invocation types (the
//! dependency runs the other way); submission builds `ExecutionInputs` from
//! the sealed artifact's CPU arm and the invocation's validated buffers and
//! values. Nothing is looked up fallibly: launches, guards, slots, fields,
//! and facts are dense by construction, the schedule-guard kind comes from
//! the sealed status-field table, and the only failures are a retained
//! safety obligation or the external system.
//!
//! Geometry is evaluated host-side per launch from the sealed execution
//! expressions (invocation values, folded native facts, guarded executor
//! arithmetic, result fields — the sealed grid is authoritative and
//! serialized launches seal `participants = 1`, which the stride loop
//! renders as ascending serial order); zero work skips submission. Runtime
//! extents are evaluated the same way and passed to the kernel as scalar
//! words.

use crate::encode::WordSource;
use crate::native::{NativeArtifact, NativeStep};
use crate::workers::Workers;
use seismic_lang::{
    abi::RangeEndpoint,
    types::{DType, ValuePath},
};
use seismic_realization::failure::{
    ExecutionFailure, ExternalFailure, ExternalStage, SafetyKind, SafetyViolation,
    SafetyViolationSource,
};
use seismic_realization::ids::{GuardIx, LaunchIx, ScalarSlotIx, StorageIx};
use seismic_realization::invocation::InvocationValues;
use seismic_realization::physical::{
    ExecutionExpr, GuardPredicate, PhysicalPlan, ScalarSource, SealedGuard, SealedJoin,
    SealedValue,
};

/// One decoded scalar result of the compiler-owned result block, in ABI
/// result order (range endpoints stay explicitly distinguished).
#[derive(Clone, Debug, PartialEq)]
pub struct ScalarOutput {
    pub path: ValuePath,
    pub endpoint: Option<RangeEndpoint>,
    pub dtype: DType,
    pub value: f64,
}

/// The outputs of one executed invocation: decoded scalar results. Result
/// tensor planes are written in place through the validated result buffers
/// preparation owns; submission returns them alongside these.
#[derive(Clone, Debug, Default)]
pub struct InvocationOutputs {
    pub scalars: Vec<ScalarOutput>,
}

/// Everything execution needs from one prepared invocation.
pub struct ExecutionInputs<'a> {
    pub physical: &'a PhysicalPlan<crate::intrinsics::CpuDialect>,
    pub native: &'a NativeArtifact,
    /// Host pointer of every validated ABI buffer, dense by `BufferSlot`
    /// (parameters and preparation-allocated results alike).
    pub buffers: &'a [*mut u8],
    pub values: &'a InvocationValues,
}

/// The CPU executor over one worker pool. Execution is synchronous; one
/// launch runs at a time.
pub struct CpuExecutor<'a> {
    workers: &'a mut Workers,
}

impl<'a> CpuExecutor<'a> {
    pub fn new(workers: &'a mut Workers) -> CpuExecutor<'a> {
        CpuExecutor { workers }
    }

    /// Execute one prepared invocation: evaluate every launch's geometry,
    /// run the native tree in schedule order, and report the first recorded
    /// safety error after synchronous completion.
    pub fn execute(
        &mut self,
        invocation: &ExecutionInputs<'_>,
    ) -> Result<InvocationOutputs, ExecutionFailure> {
        let plan = invocation.physical;
        let resources = plan.resources();
        let mut run = Run {
            plan,
            native: invocation.native,
            buffers: invocation.buffers,
            values: invocation.values,
            slots: vec![0; resources.scalar_slots as usize],
            result_words: vec![0; plan.result_fields().len()],
            status: vec![0; resources.status_bytes as usize],
            arena: vec![0; resources.arena_bytes as usize],
            // Dense by `GuardIx`: every guard of the plan is a guard step,
            // and the native tree mirrors the schedule exactly once.
            guards: guard_table(invocation.native.tree()),
            workers: &mut *self.workers,
        };
        run.schedule(invocation.native.tree())?;
        // The first recorded safety error, reported after completion.
        let failure = run.first_status_failure();
        let scalars = plan
            .result_fields()
            .iter()
            .map(|(index, field)| ScalarOutput {
                path: field.path.clone(),
                endpoint: field.endpoint,
                dtype: field.dtype,
                value: decode_word(run.result_words[index.index()], field.dtype),
            })
            .collect();
        match failure {
            Some(violation) => Err(violation.into()),
            None => Ok(InvocationOutputs { scalars }),
        }
    }
}

/// The per-invocation execution state.
struct Run<'a> {
    plan: &'a PhysicalPlan<crate::intrinsics::CpuDialect>,
    native: &'a NativeArtifact,
    buffers: &'a [*mut u8],
    values: &'a InvocationValues,
    slots: Vec<u64>,
    result_words: Vec<u64>,
    status: Vec<u8>,
    arena: Vec<u8>,
    /// Every sealed guard, dense by `GuardIx` (schedule tree order).
    guards: Vec<SealedGuard>,
    workers: &'a mut Workers,
}

/// Collect every guard of the native tree, dense by `GuardIx`: the seal
/// allocates guard ids in schedule tree order, and the native tree mirrors
/// the schedule exactly once, so the pre-order walk produces the ids in
/// allocation order.
fn guard_table(steps: &[NativeStep]) -> Vec<SealedGuard> {
    fn walk(steps: &[NativeStep], out: &mut Vec<SealedGuard>) {
        for step in steps {
            match step {
                NativeStep::Guard(guard) => out.push(guard.clone()),
                NativeStep::Call(body)
                | NativeStep::Repeat { body, .. } => walk(body, out),
                NativeStep::If { then_steps, else_steps, .. } => {
                    walk(then_steps, out);
                    walk(else_steps, out);
                }
                NativeStep::Launch(_) | NativeStep::Fill(_) => {}
            }
        }
    }
    let mut out = Vec::new();
    walk(steps, &mut out);
    out
}

impl Run<'_> {
    // -- the schedule ---------------------------------------------------------

    fn schedule(&mut self, steps: &[NativeStep]) -> Result<(), ExecutionFailure> {
        for step in steps {
            match step {
                NativeStep::Launch(launch) => self.launch(*launch)?,
                NativeStep::Guard(guard) => {
                    if !self.guard_holds(&guard.predicate)? {
                        // The kind of the guard's status field, from the
                        // sealed status-field table.
                        return Err(SafetyViolation {
                            source: SafetyViolationSource::Guard(guard.id),
                            obligation: guard.obligation.clone(),
                            kind: self.plan.status_fields()[guard.status].kind,
                        }
                        .into());
                    }
                }
                NativeStep::Call(body) => self.schedule(body)?,
                NativeStep::If { condition, then_steps, else_steps, joins } => {
                    let word = self.scalar_source_word(condition);
                    if word != 0 {
                        self.schedule(then_steps)?;
                        self.join_values(joins, true)?;
                    } else {
                        self.schedule(else_steps)?;
                        self.join_values(joins, false)?;
                    }
                }
                NativeStep::Repeat { start, end, binder, carries, body, .. } => {
                    let start = self.eval_expr(start)?;
                    let end = self.eval_expr(end)?;
                    // The discharged `RangeOrdered` predicate (an invocation
                    // relation or the dominating guard step that already ran)
                    // established `0 <= start <= end <= bound`. `current` is
                    // rebound from `initial` before the first visit and from
                    // `update` after each visit; `result` is the final value.
                    for carry in carries {
                        self.copy_sealed_value(&carry.initial, &carry.current)?;
                    }
                    for coordinate in start..end {
                        self.slots[binder.index()] = coordinate;
                        self.schedule(body)?;
                        for carry in carries {
                            self.copy_sealed_value(&carry.update, &carry.current)?;
                        }
                    }
                    for carry in carries {
                        self.copy_sealed_value(&carry.current, &carry.result)?;
                    }
                }
                NativeStep::Fill(fill) => {
                    let pointer = self.storage_pointer(fill.storage);
                    if !pointer.is_null() {
                        unsafe {
                            std::ptr::write_bytes(pointer, 0, fill.bytes as usize);
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// One launch: evaluate the geometry, skip zero work, build the tables,
    /// and distribute the participants across the worker pool.
    fn launch(&mut self, launch: LaunchIx) -> Result<(), ExecutionFailure> {
        let handle = self.native.launch(launch);
        let descriptor = &handle.descriptor;
        // The sealed launches mirror the schedule in tree order, which is
        // `LaunchIx` order — direct dense index.
        let sealed = &self.plan.launches()[launch.index()];
        // Zero work is a retained launch condition: never submitted. The
        // seal guarantees a present launch with work resolves at least one
        // participant.
        let work = self.eval_expr(&sealed.work_items)?;
        if work == 0 {
            return Ok(());
        }
        let participants = self.eval_expr(&sealed.participants)?;
        // The buffer table: shared storages in binding-slot order, plus the
        // status area as one extra entry when the launch writes status.
        let mut buffers: Vec<*mut u8> =
            Vec::with_capacity(descriptor.table_order.len() + 1);
        for storage in &descriptor.table_order {
            buffers.push(self.storage_pointer(*storage));
        }
        if descriptor.has_status {
            buffers.push(self.status.as_mut_ptr());
        }
        // The scalar word table: [words][runtime extents][work][participants].
        let mut table: Vec<u64> =
            Vec::with_capacity(descriptor.words.len() + descriptor.runtime_extents.len() + 2);
        for source in &descriptor.words {
            table.push(self.word_source(source));
        }
        for extent in &descriptor.runtime_extents {
            let expr = self.plan.runtime_extent(*extent);
            table.push(self.eval_expr(expr)?);
        }
        table.push(work);
        table.push(participants);
        self.workers
            .run(
                handle.entry,
                &buffers,
                &mut table,
                participants,
                descriptor.scratch_bytes as usize,
            )
            .map_err(|reason| {
                ExecutionFailure::External(ExternalFailure {
                    stage: ExternalStage::Allocation,
                    detail: reason,
                })
            })?;
        // Published executor slots persist across launches.
        for (index, source) in descriptor.words.iter().enumerate() {
            if let WordSource::Executor(slot) = source {
                self.slots[slot.index()] = table[index];
            }
        }
        Ok(())
    }

    // -- values ----------------------------------------------------------------

    fn word_source(&self, source: &WordSource) -> u64 {
        match source {
            WordSource::Abi(slot) => self.values.scalars[*slot].bits,
            WordSource::Executor(slot) => self.slots[slot.index()],
            WordSource::Invocation(id) => self.values.derived[*id],
            WordSource::Result(field) => self.result_words[field.index()],
        }
    }

    fn scalar_source_word(&self, source: &ScalarSource) -> u64 {
        match source {
            ScalarSource::Abi(slot) => self.values.scalars[*slot].bits,
            ScalarSource::Executor(slot) => self.slots[slot.index()],
            ScalarSource::Invocation(id) => self.values.derived[*id],
            ScalarSource::Result(field) => self.result_words[field.index()],
        }
    }

    /// One retained execution expression: invocation-known values, folded
    /// native facts, guarded executor arithmetic, or a result-field read.
    /// Guarded arithmetic consults its dominating guard (whose kind is the
    /// sealed kind of the guard's status field) and reads result-block
    /// words through the results closure.
    fn eval_expr(&self, expr: &ExecutionExpr) -> Result<u64, SafetyViolation> {
        match expr {
            ExecutionExpr::Invocation(id) => Ok(self.values.derived[*id]),
            ExecutionExpr::NativeFact(index) => Ok(self.native.native_fact(*index)),
            ExecutionExpr::ResultField(field) => Ok(self.result_words[field.index()]),
            ExecutionExpr::Guarded(guarded) => {
                let values = self.values;
                let slots = &self.slots;
                let results = &self.result_words;
                let this: &Run<'_> = self;
                guarded.evaluate(
                    values,
                    &|slot: ScalarSlotIx| slots[slot.index()],
                    &|field: seismic_realization::ids::ResultFieldIx| {
                        results[field.index()]
                    },
                    &mut |guard: GuardIx, _kind: SafetyKind| {
                        // The dominating guard step, dense by id; the kind
                        // of its status field is the sealed kind.
                        let record = &this.guards[guard.index()];
                        if this.guard_holds(&record.predicate)? {
                            Ok(())
                        } else {
                            Err(SafetyViolation {
                                source: SafetyViolationSource::Guard(guard),
                                obligation: record.obligation.clone(),
                                kind: this.plan.status_fields()[record.status].kind,
                            })
                        }
                    },
                )
            }
        }
    }

    fn guard_holds(&self, predicate: &GuardPredicate) -> Result<bool, SafetyViolation> {
        Ok(match predicate {
            GuardPredicate::ProductFits { factors, bits } => {
                let mut product: u64 = 1;
                for factor in factors {
                    product = match product.checked_mul(self.eval_expr(factor)?) {
                        Some(value) => value,
                        // An overflowing product never fits.
                        None => return Ok(false),
                    };
                }
                *bits >= 64 || product < (1u64 << *bits)
            }
            GuardPredicate::ExtentPositive { extent } => self.eval_expr(extent)? > 0,
            GuardPredicate::RangeOrdered { start, end, bound } => {
                let (start, end, bound) = (
                    self.eval_expr(start)?,
                    self.eval_expr(end)?,
                    self.eval_expr(bound)?,
                );
                start <= end && end <= bound
            }
        })
    }

    // -- joins, carries, and fills ----------------------------------------------

    /// Thread every join of one branch: `joined` holds the taken side's
    /// value after the branch.
    fn join_values(
        &mut self,
        joins: &[SealedJoin],
        taken: bool,
    ) -> Result<(), ExecutionFailure> {
        for join in joins {
            let source = if taken { &join.then_value } else { &join.else_value };
            self.copy_sealed_value(source, &join.joined)?;
        }
        Ok(())
    }

    /// Thread one joined or carried sealed value: a scalar word copy, or a
    /// device copy between distinct storages (same storage: no-op).
    fn copy_sealed_value(
        &mut self,
        source: &SealedValue,
        destination: &SealedValue,
    ) -> Result<(), ExecutionFailure> {
        match (source, destination) {
            (SealedValue::Scalar(source), SealedValue::Scalar(destination)) => {
                let word = self.scalar_source_word(source);
                match destination {
                    ScalarSource::Executor(slot) => self.slots[slot.index()] = word,
                    ScalarSource::Result(field) => self.result_words[field.index()] = word,
                    // A joined or carried destination is always a mutable
                    // location (P1 route table); the ABI and invocation
                    // leaves are inputs.
                    ScalarSource::Abi(_) | ScalarSource::Invocation(_) => unreachable!(
                        "a joined or carried scalar destination is an immutable location (P1)"
                    ),
                }
                Ok(())
            }
            (SealedValue::Tensor(source), SealedValue::Tensor(destination)) => {
                // Both sides are the same routed value across the boundary:
                // one view per plane, in the same plane order (P1).
                if source.len() != destination.len() {
                    unreachable!("a joined tensor's plane counts disagree (P1)");
                }
                for (source_view, destination_view) in source.iter().zip(destination.iter()) {
                    if source_view.storage == destination_view.storage {
                        continue;
                    }
                    let bytes = self.plan.storages()[source_view.storage].bytes;
                    let from = self.storage_pointer(source_view.storage);
                    let to = self.storage_pointer(destination_view.storage);
                    if !from.is_null() && !to.is_null() && bytes > 0 {
                        // SAFETY: both storages are sealed, distinct, and at
                        // least `bytes` long; execution is synchronous.
                        unsafe {
                            std::ptr::copy_nonoverlapping(from, to, bytes as usize);
                        }
                    }
                }
                Ok(())
            }
            // A joined value keeps its kind across the boundary (P1).
            _ => unreachable!("a joined value changes kind across the boundary (P1)"),
        }
    }

    /// The host pointer of one sealed storage: ABI buffers are the caller's
    /// (or preparation's); internal residences use their arena offsets;
    /// zero-sized storages keep identity without an interval. Every index
    /// below is dense and in-bounds by the seal.
    fn storage_pointer(&mut self, storage: StorageIx) -> *mut u8 {
        let record = &self.plan.storages()[storage];
        match record.placement {
            seismic_realization::physical::StoragePlacement::Abi { slot } => {
                if record.bytes == 0 {
                    std::ptr::null_mut()
                } else {
                    // Dense by `BufferSlot`: submission built the pointer
                    // table in slot order from the validated buffers.
                    self.buffers[slot.index()]
                }
            }
            seismic_realization::physical::StoragePlacement::Arena { offset } => {
                if record.bytes == 0 {
                    std::ptr::null_mut()
                } else {
                    // SAFETY: the arena holds `arena_bytes` and the seal's
                    // arena packing places every offset+bytes inside it.
                    unsafe { self.arena.as_mut_ptr().add(offset as usize) }
                }
            }
            // Assembly validated that no nonzero workgroup storage exists.
            seismic_realization::physical::StoragePlacement::Workgroup => {
                std::ptr::null_mut()
            }
            // Participant-scoped storages are addressed from each worker's
            // private scratch inside the kernel; the encode split never
            // puts one in the shared table or a cross-boundary join.
            seismic_realization::physical::StoragePlacement::Participant => unreachable!(
                "a participant-scoped storage reached a shared host pointer (B1Cpu encode split)"
            ),
        }
    }

    /// The first recorded safety error of the root status block, if any:
    /// a kernel-`Check` failure, named by its dense status field.
    fn first_status_failure(&self) -> Option<SafetyViolation> {
        for (index, field) in self.plan.status_fields().iter() {
            let offset = index.index() * 4;
            // The status block is `4 bytes per field` by the sealed
            // `PlanResources` identity.
            let raw: [u8; 4] = self.status[offset..offset + 4].try_into().unwrap();
            if i32::from_le_bytes(raw) != 0 {
                return Some(SafetyViolation {
                    source: SafetyViolationSource::KernelCheck(index),
                    obligation: field.obligation.clone(),
                    kind: field.kind,
                });
            }
        }
        None
    }
}

/// Decode one result word at its dtype (the ABI presentation).
fn decode_word(word: u64, dtype: DType) -> f64 {
    match dtype {
        DType::F32 => f32::from_bits(word as u32) as f64,
        DType::I32 => f64::from((word as u32) as i32),
        DType::U32 => f64::from(word as u32),
        DType::Bool => f64::from((word & 1) as u8),
        DType::F16 => seismic_lang::numeric::f16_to_f32((word & 0xffff) as u16) as f64,
        DType::BF16 => f32::from_bits(((word & 0xffff) as u32) << 16) as f64,
    }
}
