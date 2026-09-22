//! Metal execution of the core-owned command vocabulary.

use crate::{Metal, MetalBuffer, MetalDevice, MetalLaunchMode, Pipeline};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder, MTLSize,
};
use seismic_compiler::errors::ExecutionError;
use seismic_compiler::executable::{
    CompiledBufferView, CompiledLocalClassTotals, CompiledLocalLayout, DeviceService,
    ExecutableAllocationId, ExecutableCommand, ExecutableKernelId, ExecutionEnvironment,
    KernelAbiBindings, NativeExecution, NativeExecutor, NativeSubmission, RuntimeBuffer,
};
use seismic_ir::schedule::{AnyScalarSlot, FillValue};
use seismic_ir::storage::LaunchLocalKind;
use seismic_ir::target::KernelAbiAllocationRole;
use seismic_lang::expr::SymbolValue;
use seismic_lang::types::DType;

pub struct MetalExecutor {
    device: MetalDevice,
}

impl MetalExecutor {
    pub fn new(device: MetalDevice) -> Self {
        Self { device }
    }
    pub fn device(&self) -> &MetalDevice {
        &self.device
    }
}

impl NativeExecutor<Metal> for MetalExecutor {
    type Handle = Pipeline;
    type Device = MetalDevice;
    type Submission = MetalSubmission;

    fn begin_submission(&self) -> Result<Self::Submission, ExecutionError> {
        Ok(MetalSubmission {
            device: self.device.clone(),
            pending: Vec::new(),
        })
    }
}

pub struct MetalSubmission {
    device: MetalDevice,
    pending: Vec<Retained<ProtocolObject<dyn MTLCommandBuffer>>>,
}

pub struct MetalExecution {
    pending: Vec<Retained<ProtocolObject<dyn MTLCommandBuffer>>>,
}

// Command buffers are Metal synchronization objects and may be completed on
// another runtime thread after submission.
unsafe impl Send for MetalSubmission {}
unsafe impl Send for MetalExecution {}

impl NativeSubmission<Metal> for MetalSubmission {
    type Handle = Pipeline;
    type Device = MetalDevice;
    type Execution = MetalExecution;

    fn execute(
        &mut self,
        command: &ExecutableCommand<Metal>,
        env: &mut ExecutionEnvironment<'_, Metal, Pipeline, MetalDevice>,
    ) -> Result<(), ExecutionError> {
        match command {
            ExecutableCommand::Launch {
                kernel,
                mode,
                grid,
                workgroup,
                empty,
                bindings,
                nat_args,
                scalar_args,
                locals,
                local_totals,
                scratch,
                abi,
                addressable_resources: _,
            } => self.launch(
                *kernel,
                *mode,
                grid,
                workgroup,
                empty,
                bindings,
                nat_args,
                scalar_args,
                locals,
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
                self.synchronize()?;
                self.copy(source, destination, env.nat(bytes), env)
            }
            ExecutableCommand::Fill {
                destination,
                value,
                bytes,
            } => {
                self.synchronize()?;
                self.fill(destination, *value, env.nat(bytes), env)
            }
            ExecutableCommand::ScalarMove { from, to } => {
                self.synchronize()?;
                let value = env.symbol(from.symbol());
                env.set_slot(*to, value);
                Ok(())
            }
            ExecutableCommand::ScalarRead {
                source,
                bounds: _,
                byte_offset,
                to,
            } => {
                self.synchronize()?;
                self.scalar_read(source, env.nat(byte_offset), *to, env)
            }
        }
    }

    fn submit(self) -> Self::Execution {
        MetalExecution {
            pending: self.pending,
        }
    }
}

impl NativeExecution for MetalExecution {
    fn complete(&mut self) -> Result<(), ExecutionError> {
        complete_pending(&mut self.pending)
    }
}

impl MetalSubmission {
    fn synchronize(&mut self) -> Result<(), ExecutionError> {
        complete_pending(&mut self.pending)
    }

    #[allow(clippy::too_many_arguments)]
    fn launch(
        &mut self,
        kernel_id: ExecutableKernelId,
        _mode: MetalLaunchMode,
        grid_exprs: &[seismic_lang::expr::compiled::CompiledNat; 3],
        workgroup_exprs: &[seismic_lang::expr::compiled::CompiledNat; 3],
        empty: &seismic_lang::expr::compiled::CompiledPredicate,
        bindings: &[CompiledBufferView],
        nat_args: &[seismic_lang::expr::compiled::CompiledNat],
        scalar_args: &[seismic_lang::expr::SymbolId],
        locals: &[CompiledLocalLayout],
        totals: &CompiledLocalClassTotals,
        scratch: seismic_compiler::executable::LaunchScratchBindings,
        abi: &KernelAbiBindings,
        env: &mut ExecutionEnvironment<'_, Metal, Pipeline, MetalDevice>,
    ) -> Result<(), ExecutionError> {
        if env.predicate(empty) {
            return Ok(());
        }
        let kernel = env.kernel_handle(kernel_id);
        let grid = grid_exprs.each_ref().map(|value| env.nat(value));
        let workgroup = workgroup_exprs.each_ref().map(|value| env.nat(value));
        if grid.contains(&0) || workgroup.contains(&0) {
            return Ok(());
        }
        let mut words = vec![0u64; kernel.words.total as usize];
        let mut resolved = Vec::with_capacity(bindings.len());
        for (position, binding) in bindings.iter().enumerate() {
            let view = env.resolve_view(binding);
            let layout = kernel.words.bindings[position];
            words[layout.first as usize..layout.first as usize + layout.rank as usize]
                .copy_from_slice(&view.extents);
            words[layout.first as usize + layout.rank as usize
                ..layout.first as usize + 2 * layout.rank as usize]
                .copy_from_slice(&view.strides);
            resolved.push((view.buffer.clone(), view.byte_offset));
        }
        for (index, value) in nat_args.iter().enumerate() {
            words[kernel.words.nat_first as usize + index] = env.nat(value);
        }
        for (index, symbol) in scalar_args.iter().copied().enumerate() {
            words[kernel.words.scalar_first as usize + index] = encode_symbol(env.symbol(symbol));
        }
        let total_values = [
            env.nat(&totals.workgroup_bytes),
            env.nat(&totals.participant_bytes),
            env.nat(&totals.register_bytes),
        ];
        for (position, local) in locals.iter().enumerate() {
            let layout = kernel.words.locals[position];
            let offset = env.nat(&local.byte_offset);
            words[layout.first as usize] = offset;
            for (axis, value) in local.extents.iter().enumerate() {
                words[layout.first as usize + 1 + axis] = env.nat(value);
            }
            for (axis, value) in local.strides.iter().enumerate() {
                words[layout.first as usize + 1 + layout.rank as usize + axis] = env.nat(value);
            }
        }
        words[kernel.words.grid_first as usize..kernel.words.grid_first as usize + 3]
            .copy_from_slice(&grid);
        words[kernel.words.workgroup_first as usize..kernel.words.workgroup_first as usize + 3]
            .copy_from_slice(&workgroup);
        words[kernel.words.local_total_first as usize..kernel.words.local_total_first as usize + 3]
            .copy_from_slice(&total_values);
        let word_table = abi_buffer(abi, KernelAbiAllocationRole::WordTable, env);
        write_words(word_table, &words)?;
        let results = abi_buffer(abi, KernelAbiAllocationRole::ScalarResults, env);
        let result_buffer = results.buffer.clone();
        let result_base_offset = results.base_offset;
        let participant = scratch_buffer(scratch.allocation(LaunchLocalKind::Participant), env);
        let registers = scratch_buffer(scratch.allocation(LaunchLocalKind::Register), env);

        let command = self.device.queue().commandBuffer().ok_or_else(|| {
            ExecutionError::SubmissionFailed("Metal could not create a command buffer".into())
        })?;
        let encoder = command.computeCommandEncoder().ok_or_else(|| {
            ExecutionError::SubmissionFailed("Metal could not create a compute encoder".into())
        })?;
        encoder.setComputePipelineState(&kernel.state);
        for (index, (buffer, offset)) in resolved.iter().enumerate() {
            unsafe { encoder.setBuffer_offset_atIndex(Some(buffer.raw()), *offset as usize, index) }
        }
        let first_aux = bindings.len();
        unsafe {
            encoder.setBuffer_offset_atIndex(
                Some(word_table.buffer.raw()),
                word_table.base_offset as usize,
                first_aux,
            );
            encoder.setBuffer_offset_atIndex(
                Some(results.buffer.raw()),
                results.base_offset as usize,
                first_aux + 1,
            );
            set_optional_buffer(&encoder, participant, first_aux + 2);
            set_optional_buffer(&encoder, registers, first_aux + 3);
            encoder.setThreadgroupMemoryLength_atIndex(total_values[0] as usize, 0);
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: grid[0] as usize,
                height: grid[1] as usize,
                depth: grid[2] as usize,
            },
            MTLSize {
                width: workgroup[0] as usize,
                height: workgroup[1] as usize,
                depth: workgroup[2] as usize,
            },
        );
        encoder.endEncoding();
        command.commit();
        self.pending.push(command);
        let result_slots = kernel.result_slots.clone();
        if !result_slots.is_empty() {
            self.synchronize()?;
        }
        for (index, (slot, dtype)) in result_slots.iter().enumerate() {
            let mut bytes = [0u8; 8];
            self.device.read(
                &result_buffer,
                result_base_offset + (index as u64) * 8,
                &mut bytes,
            )?;
            env.set_slot(*slot, decode_symbol(u64::from_le_bytes(bytes), *dtype));
        }
        Ok(())
    }

    fn copy(
        &self,
        source: &CompiledBufferView,
        destination: &CompiledBufferView,
        bytes: u64,
        env: &ExecutionEnvironment<'_, Metal, Pipeline, MetalDevice>,
    ) -> Result<(), ExecutionError> {
        let source = env.resolve_view(source);
        let destination = env.resolve_view(destination);
        let mut data = vec![0u8; bytes as usize];
        self.device
            .read(source.buffer, source.byte_offset, &mut data)?;
        self.device
            .write(destination.buffer, destination.byte_offset, &data)
    }
    fn fill(
        &self,
        destination: &CompiledBufferView,
        value: FillValue,
        bytes: u64,
        env: &ExecutionEnvironment<'_, Metal, Pipeline, MetalDevice>,
    ) -> Result<(), ExecutionError> {
        let destination = env.resolve_view(destination);
        let mut data = vec![0u8; bytes as usize];
        let pattern = value.pattern();
        for (index, byte) in data.iter_mut().enumerate() {
            *byte = pattern[index % pattern.len()];
        }
        self.device
            .write(destination.buffer, destination.byte_offset, &data)
    }
    fn scalar_read(
        &self,
        source: &CompiledBufferView,
        offset: u64,
        to: AnyScalarSlot,
        env: &mut ExecutionEnvironment<'_, Metal, Pipeline, MetalDevice>,
    ) -> Result<(), ExecutionError> {
        let source = env.resolve_view(source);
        let mut bytes = [0u8; 4];
        let width = to.dtype().bytes() as usize;
        self.device.read(
            source.buffer,
            source
                .byte_offset
                .checked_add(offset)
                .expect("CompiledBufferView violated the MetalBuffer host-range FFI precondition: scalar byte offset overflowed"),
            &mut bytes[..width],
        )?;
        env.set_slot(to, decode_bytes(bytes, to.dtype()));
        Ok(())
    }
}

fn complete_pending(
    pending: &mut Vec<Retained<ProtocolObject<dyn MTLCommandBuffer>>>,
) -> Result<(), ExecutionError> {
    let completed = complete_all(pending.iter(), |command| {
        command.waitUntilCompleted();
        if let Some(error) = command.error() {
            Err(ExecutionError::SynchronizationFailed(
                error.localizedDescription().to_string(),
            ))
        } else {
            Ok(())
        }
    });
    // Keep every command-buffer owner in `pending` until every wait returns.
    // If a foreign call panics, the runtime leaks the still-populated
    // execution owner and therefore cannot release in-flight ownership.
    pending.clear();
    completed
}

/// Reaches the terminal boundary for every pending operation before exposing
/// the first failure. Later operations may depend on resources owned by the
/// same admitted run, so observing one failure cannot shorten their lifetime.
fn complete_all<T>(
    pending: impl IntoIterator<Item = T>,
    mut complete: impl FnMut(T) -> Result<(), ExecutionError>,
) -> Result<(), ExecutionError> {
    let mut first_error = None;
    for item in pending {
        if let Err(error) = complete(item) {
            if first_error.is_none() {
                first_error = Some(error);
            }
        }
    }
    first_error.map_or(Ok(()), Err)
}

#[cfg(test)]
mod completion_tests {
    use super::*;

    #[test]
    fn completion_waits_every_pending_operation_before_returning_first_error() {
        let mut completed = Vec::new();
        let error = complete_all([0, 1, 2, 3], |item| {
            completed.push(item);
            match item {
                1 => Err(ExecutionError::SynchronizationFailed("first".into())),
                2 => Err(ExecutionError::DeviceLost("second".into())),
                _ => Ok(()),
            }
        })
        .unwrap_err();

        assert_eq!(completed, vec![0, 1, 2, 3]);
        assert_eq!(error, ExecutionError::SynchronizationFailed("first".into()));
    }
}

unsafe fn set_optional_buffer(
    encoder: &objc2::runtime::ProtocolObject<dyn MTLComputeCommandEncoder>,
    buffer: Option<&RuntimeBuffer<MetalBuffer>>,
    index: usize,
) {
    match buffer {
        Some(buffer) => encoder.setBuffer_offset_atIndex(
            Some(buffer.buffer.raw()),
            buffer.base_offset as usize,
            index,
        ),
        None => encoder.setBuffer_offset_atIndex(None, 0, index),
    }
}
fn scratch_buffer<'a>(
    allocation: Option<ExecutableAllocationId>,
    env: &'a ExecutionEnvironment<'_, Metal, Pipeline, MetalDevice>,
) -> Option<&'a RuntimeBuffer<MetalBuffer>> {
    allocation.map(|allocation| env.buffer(allocation))
}
fn abi_buffer<'a>(
    abi: &KernelAbiBindings,
    role: KernelAbiAllocationRole,
    env: &'a ExecutionEnvironment<'_, Metal, Pipeline, MetalDevice>,
) -> &'a RuntimeBuffer<MetalBuffer> {
    env.buffer(abi.allocation(role))
}
fn write_words(buffer: &RuntimeBuffer<MetalBuffer>, words: &[u64]) -> Result<(), ExecutionError> {
    let bytes = words
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    buffer.buffer.write_bytes(buffer.base_offset, &bytes);
    Ok(())
}
fn encode_symbol(value: SymbolValue) -> u64 {
    match value {
        SymbolValue::Nat(v) => v,
        SymbolValue::Int(v) => v as u64,
        SymbolValue::F32(v) => u64::from(v.to_bits()),
        SymbolValue::F16(v) | SymbolValue::BF16(v) => u64::from(v),
        SymbolValue::I32(v) => u64::from(v as u32),
        SymbolValue::U32(v) => u64::from(v),
        SymbolValue::Bool(v) => u64::from(v),
    }
}
fn decode_symbol(value: u64, dtype: DType) -> SymbolValue {
    match dtype {
        DType::F32 => SymbolValue::F32(f32::from_bits(value as u32)),
        DType::F16 => SymbolValue::F16(value as u16),
        DType::BF16 => SymbolValue::BF16(value as u16),
        DType::I32 => SymbolValue::I32(value as u32 as i32),
        DType::U32 => SymbolValue::U32(value as u32),
        DType::Bool => SymbolValue::Bool(value != 0),
    }
}
fn decode_bytes(value: [u8; 4], dtype: DType) -> SymbolValue {
    match dtype {
        DType::F32 => SymbolValue::F32(f32::from_le_bytes(value)),
        DType::F16 => SymbolValue::F16(u16::from_le_bytes([value[0], value[1]])),
        DType::BF16 => SymbolValue::BF16(u16::from_le_bytes([value[0], value[1]])),
        DType::I32 => SymbolValue::I32(i32::from_le_bytes(value)),
        DType::U32 => SymbolValue::U32(u32::from_le_bytes(value)),
        DType::Bool => SymbolValue::Bool(value[0] != 0),
    }
}
