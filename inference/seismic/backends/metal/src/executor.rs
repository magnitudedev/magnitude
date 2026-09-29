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
use seismic_ir::physical_target::KernelAbiAllocationRole;
use seismic_ir::schedule::{AnyScalarSlot, FillValue};
use seismic_ir::storage::LaunchLocalKind;
use seismic_lang::expr::SymbolValue;

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
    pending: Vec<PendingCommand>,
}

pub struct MetalExecution {
    pending: Vec<PendingCommand>,
}

/// One native command owns the exact host-written input that remains readable
/// until completion. The same range supplies the encoder's binding below.
struct PendingCommand {
    command: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    words: WordTableRead,
}

struct WordTableRead {
    buffer: MetalBuffer,
    bytes: std::ops::Range<u64>,
}

impl WordTableRead {
    fn overlaps(&self, other: &Self) -> bool {
        std::ptr::eq(self.buffer.raw(), other.buffer.raw())
            && self.bytes.start < other.bytes.end
            && other.bytes.start < self.bytes.end
    }
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
                descriptor,
                grid,
                workgroup,
                empty,
                bindings,
                nat_args,
                scalar_args,
                result_slots,
                locals,
                local_totals,
                scratch,
                abi,
                addressable_resources: _,
            } => self.launch(
                *kernel,
                *descriptor,
                grid,
                workgroup,
                empty,
                bindings,
                nat_args,
                scalar_args,
                result_slots,
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

    fn complete_prefix(&mut self) -> Result<(), ExecutionError> {
        self.synchronize()
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

    fn write_words(
        &mut self,
        buffer: &RuntimeBuffer<MetalBuffer>,
        words: &[u64],
    ) -> Result<WordTableRead, ExecutionError> {
        let bytes = words
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let length = u64::try_from(bytes.len()).expect("native word table length exceeds u64");
        assert!(
            length <= buffer.accessible_bytes,
            "native word table exceeds its ABI allocation"
        );
        let end = buffer
            .base_offset
            .checked_add(length)
            .expect("native word table range exceeds u64");
        let words = WordTableRead {
            buffer: buffer.buffer.clone(),
            bytes: buffer.base_offset..end,
        };
        // Queue ordering does not order a host write after an earlier GPU read.
        // Wait only for native readers of these bytes. In particular, logical
        // allocation ordinals from another variant are not physical identity.
        let completed = complete_all(
            self.pending
                .iter()
                .filter(|pending| pending.words.overlaps(&words)),
            |pending| complete_command(&pending.command),
        );
        // Preserve every owner if a native wait panics, and preserve unrelated
        // issued work on an ordinary completion error.
        self.pending
            .retain(|pending| !pending.words.overlaps(&words));
        completed?;
        words.buffer.write_bytes(words.bytes.start, &bytes);
        Ok(words)
    }

    fn commit(
        &mut self,
        command: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        words: WordTableRead,
    ) {
        // All fallible command/encoder formation is already finished. Retain
        // the input before the infallible native commit; no uncommitted command
        // can reach a normal failure path between registration and commit.
        self.pending.push(PendingCommand { command, words });
        self.pending.last().unwrap().command.commit();
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
        result_slots: &[seismic_ir::schedule::AnyScalarSlot],
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
            let view = env.resolve_view(binding)?;
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
            words[kernel.words.scalar_first as usize + index] = encode_symbol(env.symbol(symbol))?;
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
        let word_table = self.write_words(
            abi_buffer(abi, KernelAbiAllocationRole::WordTable, env),
            &words,
        )?;
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
                word_table.bytes.start as usize,
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
        self.commit(command, word_table);
        if !result_slots.is_empty() {
            self.synchronize()?;
        }
        for (index, slot) in result_slots.iter().enumerate() {
            let mut bytes = [0u8; 8];
            self.device.read(
                &result_buffer,
                result_base_offset + (index as u64) * 8,
                &mut bytes,
            )?;
            env.set_slot(*slot, slot.kind().decode_word(u64::from_le_bytes(bytes)));
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
        let source = env.resolve_view(source)?;
        let destination = env.resolve_view(destination)?;
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
        let destination = env.resolve_view(destination)?;
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
        let source = env.resolve_view(source)?;
        let mut bytes = [0u8; 8];
        let width = to.kind().bytes() as usize;
        self.device.read(
            source.buffer,
            source
                .byte_offset
                .checked_add(offset)
                .expect("CompiledBufferView violated the MetalBuffer host-range FFI precondition: scalar byte offset overflowed"),
            &mut bytes[..width],
        )?;
        env.set_slot(to, to.kind().decode_word(u64::from_le_bytes(bytes)));
        Ok(())
    }
}

fn complete_command(command: &ProtocolObject<dyn MTLCommandBuffer>) -> Result<(), ExecutionError> {
    command.waitUntilCompleted();
    if let Some(error) = command.error() {
        Err(ExecutionError::SynchronizationFailed(
            error.localizedDescription().to_string(),
        ))
    } else {
        Ok(())
    }
}

fn complete_pending(pending: &mut Vec<PendingCommand>) -> Result<(), ExecutionError> {
    let completed = complete_all(pending.iter(), |pending| complete_command(&pending.command));
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
    use crate::test_support::metal_device;

    fn capture_pipeline(
        device: &MetalDevice,
    ) -> Retained<ProtocolObject<dyn objc2_metal::MTLComputePipelineState>> {
        use objc2_metal::{MTLDevice, MTLLibrary};
        let source = objc2_foundation::NSString::from_str(
            "#include <metal_stdlib>\nusing namespace metal;\n\
             kernel void capture(constant ulong* words [[buffer(0)]], device ulong* output [[buffer(1)]]) {\n\
                 output[words[0]] = words[1];\n\
             }",
        );
        let library = device
            .handle()
            .raw()
            .newLibraryWithSource_options_error(&source, None)
            .unwrap();
        let function = library
            .newFunctionWithName(&objc2_foundation::NSString::from_str("capture"))
            .unwrap();
        device
            .handle()
            .raw()
            .newComputePipelineStateWithFunction_error(&function)
            .unwrap()
    }

    fn capture_words(
        submission: &mut MetalSubmission,
        pipeline: &ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
        table: &RuntimeBuffer<MetalBuffer>,
        output: &MetalBuffer,
        words: [u64; 2],
    ) {
        let words = submission.write_words(table, &words).unwrap();
        let command = submission.device.queue().commandBuffer().unwrap();
        let encoder = command.computeCommandEncoder().unwrap();
        encoder.setComputePipelineState(pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(
                Some(words.buffer.raw()),
                words.bytes.start as usize,
                0,
            );
            encoder.setBuffer_offset_atIndex(Some(output.raw()), 0, 1);
        }
        let one = MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        };
        encoder.dispatchThreadgroups_threadsPerThreadgroup(one, one);
        encoder.endEncoding();
        submission.commit(command, words);
    }

    fn captured_words(device: &MetalDevice, buffer: &MetalBuffer) -> Vec<u64> {
        let mut bytes = vec![0; buffer.len() as usize];
        device.read(buffer, 0, &mut bytes).unwrap();
        bytes
            .chunks_exact(8)
            .map(|word| u64::from_le_bytes(word.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn repeated_launches_preserve_each_word_table_version_until_its_native_read() {
        let device = metal_device();
        let pipeline = capture_pipeline(&device);
        let output = device.allocate_bytes(64 * 8).unwrap();
        device.write(&output, 0, &vec![0xff; 64 * 8]).unwrap();
        let table = RuntimeBuffer {
            tensor: None,
            buffer: device.allocate_bytes(32).unwrap(),
            base_offset: 8,
            accessible_bytes: 16,
        };
        let mut submission = MetalExecutor::new(device.clone())
            .begin_submission()
            .unwrap();
        for index in 0..64u64 {
            capture_words(
                &mut submission,
                &pipeline,
                &table,
                &output,
                [index, index * 17 + 3],
            );
            assert_eq!(
                submission.pending.len(),
                1,
                "reuse completes only the preceding reader"
            );
        }
        // No scalar-result read or explicit terminal completion occurs between
        // dispatches. The production input ownership protocol protects reuse.
        submission.submit().complete().unwrap();
        assert_eq!(
            captured_words(&device, &output),
            (0..64).map(|index| index * 17 + 3).collect::<Vec<_>>()
        );
    }

    #[test]
    fn host_word_write_completes_overlapping_ranges_and_retains_independent_readers() {
        let device = metal_device();
        let pipeline = capture_pipeline(&device);
        let output = device.allocate_bytes(3 * 8).unwrap();
        let shared = device.allocate_bytes(32).unwrap();
        let separate = device.allocate_bytes(16).unwrap();
        let mut submission = MetalExecutor::new(device.clone())
            .begin_submission()
            .unwrap();
        for (index, buffer, offset) in [(0, &shared, 0), (1, &shared, 16), (2, &separate, 0)] {
            capture_words(
                &mut submission,
                &pipeline,
                &RuntimeBuffer {
                    tensor: None,
                    buffer: buffer.clone(),
                    base_offset: offset,
                    accessible_bytes: 16,
                },
                &output,
                [index, index + 11],
            );
        }
        assert_eq!(
            submission.pending.len(),
            3,
            "adjacent ranges and other buffers do not conflict"
        );
        let middle = RuntimeBuffer {
            tensor: None,
            buffer: shared,
            base_offset: 8,
            accessible_bytes: 16,
        };
        let _next_input = submission
            .write_words(&middle, &[u64::MAX, u64::MAX])
            .unwrap();
        assert_eq!(
            submission.pending.len(),
            1,
            "only the two overlapping readers are completed"
        );
        assert!(std::ptr::eq(
            submission.pending[0].words.buffer.raw(),
            separate.raw()
        ));
        submission.submit().complete().unwrap();
        assert_eq!(captured_words(&device, &output), vec![11, 12, 13]);
    }

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
fn encode_symbol(value: SymbolValue) -> Result<u64, ExecutionError> {
    value.try_word64().map_err(|error| {
        ExecutionError::ConstructionContradiction(format!(
            "native scalar ABI quantity does not fit its planned word: {error:?}"
        ))
    })
}
