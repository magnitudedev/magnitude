//! The CPU backend: universal physical strategies over the common family
//! builder, mechanical Cranelift emission of the sealed `CpuOp` vocabulary,
//! and the runtime glue that interprets the retained structured
//! `ResolvedStep` tree (Launch / Call / If / Repeat) directly.
//!
//! There is no retry, candidate, or repair API: the runtime validates the
//! root ABI binding, evaluates the retained runtime expressions, skips
//! zero-work launches (zero native dispatches are never submitted),
//! distributes launch participants across the worker pool in retained
//! order, and reports the first recorded safety error after synchronous
//! completion.

use seismic_lang::{
    abi::ScalarParameter,
    repr,
    types::{DType, RuntimeExtentId, ValuePath},
};
use seismic_realization::executable::{
    self as exec, ExecutionExpr, ResolvedExecutorScalar, ResolvedSchedule, ResolvedStep,
};

mod buffer;
pub mod codegen;
pub mod mapping;
pub mod native;
pub mod physical;
mod workers;

pub use buffer::Buffer;
pub use mapping::{Cpu, Limits, TARGET};
pub use native::Kernel;
pub use physical::{CpuDialect, CpuOp, CpuOpKind};
pub use workers::Workers;

/// The decoded outputs of one invocation: result buffers by path and plane,
/// and result scalars in root-ABI result order (ranges contribute two).
pub struct Outputs {
    pub buffers: Vec<(ValuePath, String, Vec<u8>)>,
    pub scalars: Vec<(ValuePath, DType, f64)>,
}

/// Why an invocation failed: an invalid binding, a safety violation (the
/// first recorded error, with its provenance), or a system failure.
#[derive(Debug)]
pub enum InvocationFailure {
    InvalidBinding(String),
    SafetyViolation { field: u64, node: String },
    SystemFailure(String),
}

impl std::fmt::Display for InvocationFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InvocationFailure::InvalidBinding(reason) => {
                write!(f, "invalid binding: {reason}")
            }
            InvocationFailure::SafetyViolation { field, node } => write!(
                f,
                "safety violation: the first error was recorded in status field {field} \
                 (provenance: {node}); outputs may be partially written"
            ),
            InvocationFailure::SystemFailure(reason) => write!(f, "system failure: {reason}"),
        }
    }
}
impl std::error::Error for InvocationFailure {}

impl native::Kernel {
    /// Execute one invocation. `params` supplies the root ABI parameter
    /// buffers in binding order and `scalars` the ABI scalar values in
    /// `abi.scalars.fields` order. Tensor results are runtime-allocated by
    /// path/plane and returned together with the decoded result scalars.
    pub fn run(
        &mut self,
        workers: &mut Workers,
        params: &[&Buffer],
        scalars: &[f64],
    ) -> Result<Outputs, InvocationFailure> {
        let system = InvocationFailure::SystemFailure;
        let plan = std::sync::Arc::clone(&self.plan);
        let abi = &plan.abi;
        let parameter_buffers: Vec<&exec::BufferBinding> = abi
            .buffers
            .iter()
            .filter(|buffer| matches!(buffer.role, exec::AbiRole::Parameter { .. }))
            .collect();
        if params.len() != parameter_buffers.len() {
            return Err(InvocationFailure::InvalidBinding(format!(
                "the CPU entry takes {} parameter buffers; {} were supplied",
                parameter_buffers.len(),
                params.len()
            )));
        }
        for (buffer, binding) in params.iter().zip(parameter_buffers.iter()) {
            if buffer.len() < binding.bytes as usize {
                return Err(InvocationFailure::InvalidBinding(format!(
                    "buffer {}.{} needs {} bytes, has {}",
                    binding.path,
                    binding.plane,
                    binding.bytes,
                    buffer.len()
                )));
            }
            if binding.alignment > 1 && buffer.offset % binding.alignment as usize != 0 {
                return Err(InvocationFailure::InvalidBinding(format!(
                    "buffer {}.{} violates its typed storage alignment",
                    binding.path, binding.plane
                )));
            }
        }
        // Root ABI alias validation: the root rules are the whole contract.
        let param_locations: Vec<(u64, u64, u64)> = params
            .iter()
            .zip(parameter_buffers.iter())
            .map(|(buffer, binding)| (buffer.root_id(), buffer.offset as u64, binding.bytes))
            .collect();
        let parameter_count = parameter_buffers.len();
        seismic_realization::validate_alias_rules(abi, |binding| {
            // ABI bindings are numbered in buffer order (parameters, then
            // results); results are runtime-allocated distinct buffers.
            let ordinal = binding.0 as usize;
            if ordinal < parameter_count {
                Ok(Some(param_locations[ordinal]))
            } else {
                Ok(None)
            }
        })
        .map_err(InvocationFailure::InvalidBinding)?;
        // Scalar encoding (bool/dtype/index/range validation included).
        let schema: Vec<ScalarParameter> = abi
            .scalars
            .fields
            .iter()
            .map(|field| field.parameter.clone())
            .collect();
        let words = seismic_realization::encode_scalars(&schema, scalars)
            .map_err(InvocationFailure::InvalidBinding)?;
        // Runtime-allocated result buffers, by path and plane.
        let mut result_allocations: Vec<(ValuePath, String, Vec<u8>)> = Vec::new();
        for binding in &abi.buffers {
            if matches!(binding.role, exec::AbiRole::Result) {
                result_allocations.push((
                    binding.path.clone(),
                    binding.plane.clone(),
                    vec![0u8; binding.bytes as usize],
                ));
            }
        }
        let mut extents: Vec<(RuntimeExtentId, u64)> = Vec::new();
        let mut arena = vec![0u8; plan.internal_arena.bytes as usize];
        let mut slots = vec![0u64; self.slot_count as usize];
        let mut result_words = vec![0u64; self.result_words.len()];
        let mut status = vec![0u8; abi.status.as_ref().map(|s| s.bytes as usize).unwrap_or(0)];
        {
            let mut executor = Executor {
                plan: &plan,
                kernel: self,
                workers,
                params,
                words: &words,
                result_words: &mut result_words,
                slots: &mut slots,
                extents: &mut extents,
                status: &mut status,
                arena: &mut arena,
                result_allocations: &mut result_allocations,
            };
            // Retained runtime extent values (never capacities).
            let extent_exprs: Vec<(RuntimeExtentId, ExecutionExpr)> = plan
                .runtime_extents
                .ids()
                .zip(plan.runtime_extents.iter())
                .map(|(id, expr)| (id, expr.clone()))
                .collect();
            for (id, expr) in extent_exprs {
                let value = executor
                    .eval(&expr)
                    .map_err(|reason| system(format!("runtime extent {id:?}: {reason}")))?;
                executor.extents.push((id, value));
            }
            executor
                .schedule(&plan.entry.schedule)
                .map_err(InvocationFailure::SystemFailure)?;
        }
        // The first recorded safety error, reported after completion.
        if let Some(binding) = &abi.status {
            for (index, field) in binding.fields.iter().enumerate() {
                let offset = index * 4;
                if offset + 4 <= status.len() {
                    let raw: [u8; 4] = status[offset..offset + 4].try_into().unwrap();
                    if i32::from_le_bytes(raw) != 0 {
                        return Err(InvocationFailure::SafetyViolation {
                            field: field.id.0,
                            node: format!(
                                "node#{} in region {:?}, obligation #{}",
                                field.node.node.0, field.node.region, field.index
                            ),
                        });
                    }
                }
            }
        }
        // Decode the result scalars.
        let mut result_scalars = Vec::new();
        for binding in &abi.results {
            match binding {
                exec::ResultBinding::Buffer { .. } => {}
                exec::ResultBinding::Scalar { path, field, dtype } => {
                    let ordinal = result_ordinal(&self.result_words, *field);
                    let word = result_words[ordinal];
                    result_scalars.push((path.clone(), *dtype, decode_word_value(word, *dtype)));
                }
                exec::ResultBinding::Range {
                    path, start, end, ..
                } => {
                    let start_word = result_words[result_ordinal(&self.result_words, *start)];
                    let end_word = result_words[result_ordinal(&self.result_words, *end)];
                    result_scalars.push((
                        path.clone(),
                        DType::I32,
                        decode_word_value(start_word, DType::I32),
                    ));
                    result_scalars.push((
                        path.clone(),
                        DType::I32,
                        decode_word_value(end_word, DType::I32),
                    ));
                }
            }
        }
        Ok(Outputs {
            buffers: result_allocations,
            scalars: result_scalars,
        })
    }
}

/// The ordinal of one result scalar field in the result word list.
fn result_ordinal(
    schema: &[(exec::ResultScalarFieldId, DType)],
    field: exec::ResultScalarFieldId,
) -> usize {
    schema
        .iter()
        .position(|(id, _)| *id == field)
        .unwrap_or(usize::MAX)
}

fn decode_word_value(word: u64, dtype: DType) -> f64 {
    match dtype {
        DType::F32 => f32::from_bits(word as u32) as f64,
        DType::I32 => f64::from((word as u32) as i32),
        DType::U32 => f64::from(word as u32),
        DType::Bool => f64::from((word & 1) as u8),
        DType::F16 => seismic_lang::numeric::f16_to_f32((word & 0xffff) as u16) as f64,
        DType::BF16 => f32::from_bits(((word & 0xffff) as u32) << 16) as f64,
    }
}

// ---------------------------------------------------------------------------
// The executor: direct interpretation of the retained structured tree
// ---------------------------------------------------------------------------

struct Executor<'a> {
    plan: &'a exec::ResolvedPlan<physical::CpuDialect>,
    kernel: &'a mut native::Kernel,
    workers: &'a mut Workers,
    params: &'a [&'a Buffer],
    words: &'a [u64],
    result_words: &'a mut Vec<u64>,
    slots: &'a mut Vec<u64>,
    extents: &'a mut Vec<(RuntimeExtentId, u64)>,
    status: &'a mut Vec<u8>,
    arena: &'a mut Vec<u8>,
    result_allocations: &'a mut Vec<(ValuePath, String, Vec<u8>)>,
}

impl Executor<'_> {
    /// Execute one schedule: each sibling step completes before the next.
    fn schedule(
        &mut self,
        schedule: &ResolvedSchedule<physical::CpuDialect>,
    ) -> Result<(), String> {
        for step in schedule.steps.iter() {
            match step {
                ResolvedStep::Launch(launch) => self.launch(launch)?,
                // A call remains a nested plan: its transports are ids in the
                // same global storage table, so the body executes directly.
                ResolvedStep::Call(call) => self.schedule(&call.body.schedule)?,
                ResolvedStep::If(if_step) => {
                    // Exactly one retained predicate; exactly one branch.
                    let condition = self.scalar(&if_step.condition)?;
                    if condition != 0.0 {
                        self.schedule(&if_step.then_schedule)?;
                    } else {
                        self.schedule(&if_step.else_schedule)?;
                    }
                    // The strategy's value joins share leaf transports, so
                    // nothing is copied; state joins name the joined storage.
                }
                ResolvedStep::Repeat(repeat) => {
                    // The retained half-open ascending range; the discharged
                    // range predicate is evaluated here.
                    let start = self.scalar(&repeat.range.start)? as i64;
                    let end = self.scalar(&repeat.range.end)? as i64;
                    let bound = self.eval(&repeat.range.bound)? as i64;
                    if start < 0 || end < start || bound < end {
                        return Err(format!(
                            "range violation: [{start}, {end}) against bound {bound}"
                        ));
                    }
                    for coordinate in start..end {
                        self.slots[repeat.binder.0 as usize] = coordinate as u64;
                        self.schedule(&repeat.body)?;
                    }
                }
            }
        }
        Ok(())
    }

    /// One launch: evaluate the retained geometry, skip zero work, build the
    /// tables, and distribute the participants across the workers.
    fn launch(
        &mut self,
        launch: &exec::ResolvedLaunch<physical::CpuDialect>,
    ) -> Result<(), String> {
        let entry_fn = self
            .kernel
            .launches
            .get(&launch.id)
            .ok_or_else(|| format!("launch {} has no compiled entry", launch.id.0))?
            .entry;
        let (
            work_items_expr,
            participants_expr,
            has_status,
            storage_table,
            word_sources,
            extent_words,
        ) = {
            let descriptor = &self
                .kernel
                .launches
                .get(&launch.id)
                .expect("the entry exists")
                .descriptor;
            (
                descriptor.work_items.clone(),
                descriptor.participants.clone(),
                descriptor.has_status,
                descriptor.storage_table.clone(),
                descriptor.words.clone(),
                descriptor.extent_words.clone(),
            )
        };
        // Zero work is a retained launch condition: never submitted.
        let work_items = self.eval(&work_items_expr)?;
        if work_items == 0 {
            return Ok(());
        }
        let participants = self.eval(&participants_expr)?;
        if participants == 0 {
            return Err(format!(
                "launch {} resolved a zero participant count (compiler bug)",
                launch.id.0
            ));
        }
        // The buffer table (plus the status area when the launch writes one).
        let mut buffers: Vec<*mut u8> = Vec::with_capacity(storage_table.len() + 1);
        for storage in &storage_table {
            buffers.push(self.storage_pointer(*storage)?);
        }
        if has_status {
            buffers.push(self.status.as_mut_ptr());
        }
        // The scalar word table: [slots][ABI words][extent values][participants].
        let mut table: Vec<u64> = Vec::with_capacity(word_sources.len() + extent_words.len() + 1);
        for source in &word_sources {
            let word = match source {
                native::WordSource::Slot(slot) => *self
                    .slots
                    .get(slot.0 as usize)
                    .ok_or_else(|| format!("slot {} is absent", slot.0))?,
                native::WordSource::AbiInput { field } => *self
                    .words
                    .get(*field)
                    .ok_or_else(|| format!("ABI scalar field {field} is absent"))?,
                native::WordSource::AbiResult { field } => *self
                    .result_words
                    .get(*field)
                    .ok_or_else(|| format!("result word {field} is absent"))?,
            };
            table.push(word);
        }
        for extent in &extent_words {
            let value = self
                .extents
                .iter()
                .find(|(id, _)| id == extent)
                .map(|(_, value)| *value)
                .ok_or_else(|| format!("runtime extent {} was not evaluated", extent.0))?;
            table.push(value);
        }
        table.push(participants);
        let status = self
            .workers
            .run(entry_fn, &buffers, &mut table, participants, 0)?;
        if status != 0 {
            return Err(format!(
                "launch {} reported status {status} (an empty reduction input)",
                launch.id.0
            ));
        }
        // Published executor slots persist across launches.
        for (index, source) in word_sources.iter().enumerate() {
            if let native::WordSource::Slot(slot) = source {
                if let (Some(value), Some(target)) =
                    (table.get(index), self.slots.get_mut(slot.0 as usize))
                {
                    *target = *value;
                }
            }
        }
        Ok(())
    }

    /// One retained runtime expression (never a planning symbol).
    fn eval(&mut self, expr: &ExecutionExpr) -> Result<u64, String> {
        Ok(match expr {
            ExecutionExpr::Const(value) => *value,
            ExecutionExpr::Extent(id) => self
                .extents
                .iter()
                .find(|(candidate, _)| candidate == id)
                .map(|(_, value)| *value)
                .ok_or_else(|| format!("runtime extent {} was not evaluated", id.0))?,
            ExecutionExpr::AbiScalar { path, dtype } => self.abi_word(path, *dtype)?,
            ExecutionExpr::ExecutorScalar(slot) => *self
                .slots
                .get(slot.0 as usize)
                .ok_or_else(|| format!("slot {} is absent", slot.0))?,
            ExecutionExpr::Add(left, right) => self
                .eval(left)?
                .checked_add(self.eval(right)?)
                .ok_or("an execution expression overflows")?,
            ExecutionExpr::Sub(left, right) => self
                .eval(left)?
                .checked_sub(self.eval(right)?)
                .ok_or("an execution expression underflows")?,
            ExecutionExpr::Mul(left, right) => self
                .eval(left)?
                .checked_mul(self.eval(right)?)
                .ok_or("an execution expression overflows")?,
            ExecutionExpr::CeilDiv(left, right) => {
                let (left, right) = (self.eval(left)?, self.eval(right)?);
                if right == 0 {
                    return Err("a ceil-div expression divides by zero".into());
                }
                left.div_ceil(right)
            }
            ExecutionExpr::Div(left, right) => {
                let (left, right) = (self.eval(left)?, self.eval(right)?);
                if right == 0 {
                    return Err("an execution expression divides by zero".into());
                }
                left / right
            }
            ExecutionExpr::Rem(left, right) => {
                let (left, right) = (self.eval(left)?, self.eval(right)?);
                if right == 0 {
                    return Err("an execution expression divides by zero".into());
                }
                left % right
            }
            ExecutionExpr::Min(left, right) => self.eval(left)?.min(self.eval(right)?),
        })
    }

    /// One scalar as its f64 value.
    fn scalar(&mut self, scalar: &ResolvedExecutorScalar) -> Result<f64, String> {
        let (word, dtype) = match scalar {
            ResolvedExecutorScalar::Abi { path, dtype, .. } => {
                (self.abi_word(path, *dtype)?, *dtype)
            }
            ResolvedExecutorScalar::Result { .. } => {
                return Err("a result scalar cannot be read as an executor control value".into());
            }
            ResolvedExecutorScalar::Slot { slot, dtype } => (
                *self
                    .slots
                    .get(slot.0 as usize)
                    .ok_or_else(|| format!("slot {} is absent", slot.0))?,
                *dtype,
            ),
            ResolvedExecutorScalar::Computed { expr, dtype } => {
                // A retained computed control scalar; its value decodes at
                // its dtype (correct signedness for i32 leaves).
                (self.eval(expr)?, *dtype)
            }
        };
        Ok(decode_word_value(word, dtype))
    }

    fn abi_word(&mut self, path: &ValuePath, dtype: DType) -> Result<u64, String> {
        let suffix = path.to_string();
        let fields = &self.plan.abi.scalars.fields;
        let position = fields
            .iter()
            .position(|field| {
                field.parameter.dtype == dtype
                    && (suffix.is_empty()
                        || field.parameter.name.ends_with(&suffix)
                        || field.parameter.name == suffix)
            })
            .ok_or_else(|| format!("the root ABI has no scalar {path} of {}", dtype.name()))?;
        Ok(self.words[position])
    }

    /// The host pointer of one resolved storage: ABI buffers are the
    /// caller's (or the runtime-allocated results); the internal arena uses
    /// its retained offsets. Zero-sized storages keep identity without an
    /// interval.
    fn storage_pointer(&mut self, storage: exec::ResolvedStorageId) -> Result<*mut u8, String> {
        let resolved = self
            .plan
            .storage
            .get(storage)
            .ok_or_else(|| format!("storage#{} is absent from the plan", storage.0))?;
        match resolved.placement {
            exec::ResolvedStoragePlacement::Abi { binding } => {
                let parameters = self
                    .plan
                    .abi
                    .buffers
                    .iter()
                    .filter(|buffer| matches!(buffer.role, exec::AbiRole::Parameter { .. }))
                    .count();
                let ordinal = binding.0 as usize;
                if resolved.bytes == 0 {
                    Ok(std::ptr::null_mut())
                } else if ordinal < parameters {
                    Ok(self.params[ordinal].data_pointer())
                } else {
                    let result = ordinal - parameters;
                    let allocation = self
                        .result_allocations
                        .get_mut(result)
                        .ok_or_else(|| format!("result buffer {result} is absent"))?;
                    Ok(allocation.2.as_mut_ptr())
                }
            }
            exec::ResolvedStoragePlacement::Arena { offset } => {
                if resolved.bytes == 0 {
                    Ok(std::ptr::null_mut())
                } else if offset as usize + resolved.bytes as usize > self.arena.len() {
                    Err(format!(
                        "storage#{} exceeds the internal arena ({} + {} > {})",
                        storage.0,
                        offset,
                        resolved.bytes,
                        self.arena.len()
                    ))
                } else {
                    // SAFETY: the arena outlives the launch; offsets are
                    // aligned and non-overlapping in the solved plan.
                    Ok(unsafe { self.arena.as_mut_ptr().add(offset as usize) })
                }
            }
            placement => Err(format!(
                "the universal CPU strategy declares no {:?} storage; optimized \
                 strategies must plan their own worker scratch",
                placement.scope()
            )),
        }
    }
}

// ---------------------------------------------------------------------------
// The versioned host sequences (the software math reference)
// ---------------------------------------------------------------------------

/// The `seismic_math` sequences: binary64 host evaluation rounded once.
/// These are the same code paths the reference interpreter runs, so the
/// reference bits agree by construction.
extern "C" fn seismic_exp_f64(x: f64) -> f64 {
    x.exp()
}
extern "C" fn seismic_log_f64(x: f64) -> f64 {
    x.ln()
}
extern "C" fn seismic_sin_f64(x: f64) -> f64 {
    x.sin()
}
extern "C" fn seismic_cos_f64(x: f64) -> f64 {
    x.cos()
}
extern "C" fn seismic_fmax(a: f64, b: f64) -> f64 {
    a.max(b)
}
extern "C" fn seismic_fmin(a: f64, b: f64) -> f64 {
    a.min(b)
}
extern "C" fn seismic_fmod(a: f64, b: f64) -> f64 {
    a % b
}
extern "C" fn seismic_round_to(tag: i32, x: f64) -> f64 {
    let dtype = match tag {
        0 => DType::F32,
        1 => DType::BF16,
        2 => DType::F16,
        3 => DType::I32,
        4 => DType::U32,
        _ => DType::Bool,
    };
    seismic_lang::interp::round_to(dtype, x)
}
extern "C" fn seismic_f16_load(bits: i32) -> f32 {
    seismic_lang::numeric::f16_to_f32((bits & 0xffff) as u16)
}
extern "C" fn seismic_f16_store(x: f32) -> i32 {
    seismic_lang::numeric::f16_bits(x) as i32
}

/// Packed decode through the intrinsic registry:
/// `seismic_decode(repr ordinal, plane pointer array, flat index)`.
extern "C" fn seismic_decode(ordinal: i32, planes: *const *mut u8, flat: i64) -> f32 {
    match repr::REPRS.get(ordinal as usize) {
        Some(representation) => unsafe { decode_raw(representation, planes, flat.max(0) as usize) },
        None => f32::NAN,
    }
}

/// The registry decode sequence, mirroring the reference tensor decode
/// exactly (f32 plane values, binary64 combine, one rounding to f32).
unsafe fn decode_raw(representation: &repr::Repr, planes: *const *mut u8, flat: usize) -> f32 {
    let plane_base = |plane: &repr::Plane| {
        *planes.add(
            representation
                .plane_index(plane.name)
                .expect("a registry representation contains each declared plane"),
        )
    };
    let plane_entry = |plane: &repr::Plane, entry: usize| -> f32 {
        let base = plane_base(plane);
        match &plane.encoding {
            repr::PlaneEncoding::Packed {
                bits,
                interpretation,
            } => {
                let raw = read_packed_raw(base, entry, *bits);
                interpretation.decode(raw, *bits) as f32
            }
            repr::PlaneEncoding::Dense(dtype) => read_dense_raw(base, entry, *dtype),
        }
    };
    let coefficient = |bias: bool| -> f32 {
        match representation.coefficient(bias) {
            None => 0.0,
            Some(repr::Coefficient::Direct { plane }) => {
                plane_entry(&plane, flat / plane.group as usize)
            }
            Some(repr::Coefficient::Product {
                factor,
                coefficients,
                field,
                sign,
            }) => {
                let code = plane_entry(
                    &coefficients,
                    flat / coefficients.group as usize * coefficients.fields as usize
                        + field as usize,
                );
                (plane_entry(&factor, flat / factor.group as usize) * code) * sign as f32
            }
        }
    };
    let code = read_packed_raw(
        plane_base(&representation.planes()[0]),
        flat,
        representation.bits,
    );
    (coefficient(false) as f64 * representation.decode_code(code) as f64 + coefficient(true) as f64)
        as f32
}

unsafe fn read_packed_raw(base: *mut u8, entry: usize, bits: u32) -> u32 {
    let first = entry * bits as usize;
    let mut value = 0u32;
    for bit in 0..bits as usize {
        let byte = *base.add((first + bit) / 8);
        let flag = ((byte >> ((first + bit) % 8)) & 1) as u32;
        value |= flag << bit;
    }
    value
}

unsafe fn read_dense_raw(base: *mut u8, entry: usize, dtype: DType) -> f32 {
    match dtype {
        DType::F32 => {
            let bytes = [
                *base.add(entry * 4),
                *base.add(entry * 4 + 1),
                *base.add(entry * 4 + 2),
                *base.add(entry * 4 + 3),
            ];
            f32::from_le_bytes(bytes)
        }
        DType::F16 => {
            let bytes = [*base.add(entry * 2), *base.add(entry * 2 + 1)];
            seismic_lang::numeric::f16_to_f32(u16::from_le_bytes(bytes))
        }
        DType::BF16 => {
            let bytes = [*base.add(entry * 2), *base.add(entry * 2 + 1)];
            f32::from_bits(u32::from(u16::from_le_bytes(bytes)) << 16)
        }
        _ => f32::NAN,
    }
}

/// The host symbols the JIT resolves (the versioned sequences and the
/// registry decode).
pub(crate) fn host_symbols() -> Vec<(&'static str, *const u8)> {
    vec![
        ("seismic_exp_f64", seismic_exp_f64 as *const u8),
        ("seismic_log_f64", seismic_log_f64 as *const u8),
        ("seismic_sin_f64", seismic_sin_f64 as *const u8),
        ("seismic_cos_f64", seismic_cos_f64 as *const u8),
        ("seismic_fmax", seismic_fmax as *const u8),
        ("seismic_fmin", seismic_fmin as *const u8),
        ("seismic_fmod", seismic_fmod as *const u8),
        ("seismic_round_to", seismic_round_to as *const u8),
        ("seismic_f16_load", seismic_f16_load as *const u8),
        ("seismic_f16_store", seismic_f16_store as *const u8),
        ("seismic_decode", seismic_decode as *const u8),
    ]
}
