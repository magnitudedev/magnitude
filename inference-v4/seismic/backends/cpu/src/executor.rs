//! Synchronous execution of the core-owned executable command vocabulary.

use crate::buffer::{AllocationFailure, Buffer};
use crate::workers::{LaunchFailure, LaunchFrame, NativeSteps, Workers};
use crate::Cpu;
use seismic_compiler::errors::ExecutionError;
use seismic_compiler::executable::{
    CompiledBufferView, CompiledLocalClassTotals, CompiledLocalLayout, DeviceService,
    ExecutableCommand, ExecutionEnvironment, NativeExecution, NativeExecutor, NativeSubmission,
};
use seismic_ir::schedule::FillValue;
use seismic_lang::expr::compiled::InvocationValues;
use seismic_lang::expr::SymbolValue;
use std::sync::{Arc, Mutex};

#[derive(Clone, Copy, Debug, Default)]
pub struct Device;

impl DeviceService<Cpu> for Device {
    type Buffer = Buffer;

    fn allocate(&self, bytes: u64, alignment: u64) -> Result<Buffer, ExecutionError> {
        Buffer::new(bytes, alignment).map_err(allocation_error)
    }

    fn write(&self, buffer: &Buffer, offset: u64, bytes: &[u8]) -> Result<(), ExecutionError> {
        buffer.write(offset, bytes);
        Ok(())
    }

    fn read(&self, buffer: &Buffer, offset: u64, into: &mut [u8]) -> Result<(), ExecutionError> {
        buffer.read(offset, into);
        Ok(())
    }

    fn buffer_len(&self, buffer: &Buffer) -> u64 {
        buffer.len()
    }
}

pub struct Executor {
    /// Shared by every CPU device of the process (`open_host`).
    workers: Arc<Mutex<Workers>>,
    participants: usize,
}

impl Executor {
    pub fn analytical(
        &self,
        device: Arc<seismic_native_target::DeviceDescription<Cpu>>,
    ) -> Result<
        seismic_compiler::evaluation::AnalyticalEvaluationContext<Cpu>,
        seismic_compiler::errors::TargetError,
    > {
        let mut workers = self.workers.lock().expect("CPU worker pool lock poisoned");
        crate::profile::profile_for_workers(&mut workers, device)
    }

    /// Participants of the worker pool, the submitting thread included.
    pub fn workers(&self) -> usize {
        self.participants
    }

    /// Run a sequence of authored native launches as one pool job and
    /// return after all complete (see [`NativeSteps`]), with the interval
    /// the job ran on `clock`. The pool is shared by the process's CPU
    /// devices, so the interval starts once this job holds it.
    pub fn run_native(
        &self,
        steps: &dyn NativeSteps,
        clock: fn() -> f64,
    ) -> (Result<(), ExecutionError>, (f64, f64)) {
        let mut workers = self.workers.lock().expect("CPU worker-pool lock poisoned");
        let started = clock();
        let outcome = workers.run_native(steps).map_err(launch_error);
        (outcome, (started, clock()))
    }

    pub(crate) fn from_workers(workers: Arc<Mutex<Workers>>) -> Self {
        let participants = workers.lock().expect("CPU worker-pool lock poisoned").count();
        Self { workers, participants }
    }
}

impl NativeExecutor<Cpu> for Executor {
    type Handle = crate::command::CompiledKernel;
    type Device = Device;
    type Submission = Submission;

    fn begin_submission(&self) -> Result<Self::Submission, ExecutionError> {
        Ok(Submission {
            workers: self.workers.clone(),
        })
    }
}

pub struct Submission {
    workers: Arc<Mutex<Workers>>,
}

pub struct Completed;

impl NativeSubmission<Cpu> for Submission {
    type Handle = crate::command::CompiledKernel;
    type Device = Device;
    type Execution = Completed;

    fn execute(
        &mut self,
        command: &ExecutableCommand<Cpu>,
        env: &mut ExecutionEnvironment<'_, Cpu, crate::command::CompiledKernel, Device>,
    ) -> Result<(), ExecutionError> {
        match command {
            ExecutableCommand::Launch {
                kernel,
                descriptor: _,
                grid,
                workgroup,
                empty,
                bindings,
                nat_args,
                scalar_args,
                result_slots,
                locals,
                addressable_resources,
                local_totals,
                scratch,
                abi,
            } => self.launch(
                *kernel,
                grid,
                workgroup,
                empty,
                bindings,
                nat_args,
                scalar_args,
                result_slots,
                locals,
                addressable_resources,
                local_totals,
                *scratch,
                abi,
                env,
            ),
            ExecutableCommand::Copy {
                source,
                destination,
                bytes,
            } => {
                let bytes = evaluate(bytes, env.values());
                let source = view_range(source, bytes, env)?;
                let destination = view_range(destination, bytes, env)?;
                let bytes = usize::try_from(bytes)
                    .unwrap_or_else(|_| panic!("prepared CPU copy byte count exceeds usize"));
                unsafe { std::ptr::copy(source, destination, bytes) };
                Ok(())
            }
            ExecutableCommand::Fill {
                destination,
                value,
                bytes,
            } => {
                let bytes = evaluate(bytes, env.values());
                let destination = view_range(destination, bytes, env)?;
                fill(destination, bytes, *value);
                Ok(())
            }
            ExecutableCommand::ScalarMove { from, to } => {
                let value = env.values().get(from.symbol()).unwrap_or_else(|| {
                    panic!("prepared scalar move reads an unbound schedule slot")
                });
                env.set_slot(*to, value);
                Ok(())
            }
            ExecutableCommand::ScalarRead {
                source,
                bounds: _,
                byte_offset,
                to,
            } => {
                let offset = evaluate(byte_offset, env.values());
                let width = u64::from(to.kind().bytes());
                let base = view_range(
                    source,
                    offset
                        .checked_add(width)
                        .unwrap_or_else(|| panic!("prepared scalar-read range overflows u64")),
                    env,
                )?;
                let pointer = unsafe { base.add(to_usize(offset, "scalar read byte offset")) };
                let mut bytes = [0u8; 8];
                unsafe {
                    std::ptr::copy_nonoverlapping(pointer, bytes.as_mut_ptr(), width as usize);
                }
                env.set_slot(*to, to.kind().decode_word(u64::from_le_bytes(bytes)));
                Ok(())
            }
        }
    }

    fn complete_prefix(&mut self) -> Result<(), ExecutionError> {
        Ok(())
    }

    fn submit(self) -> Self::Execution {
        Completed
    }
}

impl NativeExecution for Completed {
    fn complete(&mut self) -> Result<(), ExecutionError> {
        Ok(())
    }
}

impl Submission {
    #[allow(clippy::too_many_arguments)]
    fn launch(
        &mut self,
        kernel_id: seismic_compiler::executable::ExecutableKernelId,
        grid_exprs: &[seismic_lang::expr::compiled::CompiledNat; 3],
        workgroup_exprs: &[seismic_lang::expr::compiled::CompiledNat; 3],
        empty: &seismic_lang::expr::compiled::CompiledPredicate,
        bindings: &[CompiledBufferView],
        nat_args: &[seismic_lang::expr::compiled::CompiledNat],
        scalar_args: &[seismic_lang::expr::SymbolId],
        result_slots: &[seismic_ir::schedule::AnyScalarSlot],
        locals: &[CompiledLocalLayout],
        addressable_resources: &[seismic_compiler::executable::CompiledAddressableResource],
        local_totals: &CompiledLocalClassTotals,
        scratch: seismic_compiler::executable::LaunchScratchBindings,
        _abi: &seismic_compiler::executable::KernelAbiBindings,
        env: &mut ExecutionEnvironment<'_, Cpu, crate::command::CompiledKernel, Device>,
    ) -> Result<(), ExecutionError> {
        if scratch != seismic_compiler::executable::LaunchScratchBindings::default() {
            panic!("CPU native-dynamic local policy produced an invocation scratch binding");
        }
        assert!(
            addressable_resources.is_empty(),
            "CPU executable contains an addressable resource excluded by its target profile"
        );
        if env.predicate(empty) {
            return Ok(());
        }
        let kernel = env.kernel_handle(kernel_id);
        assert!(
            kernel.layout.words.addressable_resources.is_empty(),
            "CPU kernel ABI contains an addressable-resource word layout"
        );
        let grid = grid_exprs
            .each_ref()
            .map(|value| evaluate(value, env.values()));
        let workgroup = workgroup_exprs
            .each_ref()
            .map(|value| evaluate(value, env.values()));
        let workgroups = checked_product(grid, "grid");
        let participants = checked_product(workgroup, "workgroup");
        if workgroups == 0 || participants == 0 {
            return Ok(());
        }

        let mut resolved_bindings = Vec::with_capacity(bindings.len());
        for binding in bindings {
            resolved_bindings.push(env.resolve_view(binding)?);
        }
        let buffer_table = resolved_bindings
            .iter()
            .map(|resolved| unsafe {
                resolved.buffer.data_pointer().add(to_usize(
                    resolved.byte_offset,
                    "resolved buffer byte offset",
                ))
            })
            .collect::<Vec<_>>();
        let mut words = vec![0u64; kernel.layout.words.total as usize];
        for (index, value) in nat_args.iter().enumerate() {
            words[kernel.layout.words.nat_first as usize + index] = evaluate(value, env.values());
        }
        for (index, symbol) in scalar_args.iter().copied().enumerate() {
            let value = env
                .values()
                .get(symbol)
                .unwrap_or_else(|| panic!("prepared kernel scalar argument is unbound"));
            words[kernel.layout.words.scalar_first as usize + index] = encode_symbol(value)?;
        }
        for (position, binding) in resolved_bindings.iter().enumerate() {
            let layout = kernel.layout.words.bindings[position];
            for (axis, extent) in binding.extents.iter().copied().enumerate() {
                words[layout.first as usize + axis] = extent;
            }
            for (axis, stride) in binding.strides.iter().copied().enumerate() {
                words[layout.first as usize + layout.rank as usize + axis] = stride;
            }
        }

        let workgroup_bytes = evaluate(&local_totals.workgroup_bytes, env.values());
        let participant_bytes = evaluate(&local_totals.participant_bytes, env.values());
        let register_bytes = evaluate(&local_totals.register_bytes, env.values());
        for (position, compiled) in locals.iter().enumerate() {
            let words_layout = kernel.layout.words.locals[position];
            let offset = evaluate(&compiled.byte_offset, env.values());
            words[words_layout.first as usize] = offset;
            for (axis, extent) in compiled.extents.iter().enumerate() {
                words[words_layout.first as usize + 1 + axis] = evaluate(extent, env.values());
            }
            for (axis, stride) in compiled.strides.iter().enumerate() {
                words[words_layout.first as usize + 1 + words_layout.rank as usize + axis] =
                    evaluate(stride, env.values());
            }
        }
        words[kernel.layout.words.grid_first as usize..kernel.layout.words.grid_first as usize + 3]
            .copy_from_slice(&grid);
        words[kernel.layout.words.workgroup_first as usize
            ..kernel.layout.words.workgroup_first as usize + 3]
            .copy_from_slice(&workgroup);

        let mut results = vec![0u64; result_slots.len()];
        let frame = LaunchFrame {
            buffers: buffer_table.as_ptr(),
            words: words.as_ptr(),
            results: results.as_mut_ptr(),
        };
        self.workers
            .lock()
            .expect("CPU worker-pool lock poisoned")
            .run(
                kernel.entry,
                &frame,
                workgroups,
                participants,
                workgroup_bytes,
                participant_bytes,
                register_bytes,
            )
            .map_err(launch_error)?;
        for (slot, word) in result_slots.iter().zip(results) {
            env.set_slot(*slot, slot.kind().decode_word(word));
        }
        Ok(())
    }
}

fn evaluate(value: &seismic_lang::expr::compiled::CompiledNat, values: &InvocationValues) -> u64 {
    value.evaluate_u64(values).unwrap_or_else(|error| {
        panic!("prepared CPU command expression failed evaluation: {error:?}")
    })
}

fn view_range(
    view: &CompiledBufferView,
    bytes: u64,
    env: &ExecutionEnvironment<'_, Cpu, crate::command::CompiledKernel, Device>,
) -> Result<*mut u8, ExecutionError> {
    let resolved = env.resolve_view(view)?;
    if bytes > resolved.byte_span {
        return Err(ExecutionError::ConstructionContradiction("CPU transfer exceeds resolved view".into()));
    }
    Ok(unsafe {
        resolved.buffer.data_pointer().add(to_usize(
            resolved.byte_offset,
            "resolved buffer byte offset",
        ))
    })
}

fn checked_product(values: [u64; 3], name: &str) -> u64 {
    values
        .into_iter()
        .try_fold(1u64, u64::checked_mul)
        .unwrap_or_else(|| panic!("prepared CPU {name} product overflows u64"))
}

fn to_usize(value: u64, what: &str) -> usize {
    usize::try_from(value).unwrap_or_else(|_| panic!("prepared CPU {what} exceeds usize"))
}

fn fill(destination: *mut u8, bytes: u64, value: FillValue) {
    let bytes = to_usize(bytes, "fill byte count");
    if bytes == 0 {
        return;
    }
    let width = to_usize(value.width(), "fill scalar width");
    if bytes % width != 0 {
        panic!("prepared fill byte count is not a multiple of its scalar width");
    }
    for offset in (0..bytes).step_by(width) {
        unsafe {
            std::ptr::copy_nonoverlapping(value.pattern().as_ptr(), destination.add(offset), width)
        };
    }
}

fn encode_symbol(value: SymbolValue) -> Result<u64, ExecutionError> {
    value.try_word64().map_err(|error| ExecutionError::ConstructionContradiction(
        format!("native scalar ABI quantity does not fit its planned word: {error:?}")))
}

fn allocation_error(error: AllocationFailure) -> ExecutionError {
    ExecutionError::AllocationFailed(error.to_string())
}

fn launch_error(error: LaunchFailure) -> ExecutionError {
    match error {
        LaunchFailure::Scratch(error) => allocation_error(error),
    }
}
