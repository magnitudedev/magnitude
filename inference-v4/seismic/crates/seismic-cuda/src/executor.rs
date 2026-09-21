//! CUDA execution of the core-owned command vocabulary.

use crate::buffer::Buffer;
use crate::command::{CompiledKernel, LaunchFrame};
use crate::driver::{Allocation, Context, Driver, DriverError, PinnedUpload, Stream};
use crate::{Cuda, CudaLaunchMode};
use seismic_compiler::errors::ExecutionError;
use seismic_compiler::executable::{
    CompiledBufferView, CompiledLocalClassTotals, CompiledLocalLayout, DeviceService,
    ExecutableAllocationId, ExecutableCommand, ExecutableKernelId, ExecutionEnvironment,
    KernelAbiBindings, NativeExecution, NativeExecutor, NativeSubmission, RuntimeBuffer,
};
use seismic_compiler::schedule::{AnyScalarSlot, FillValue};
use seismic_compiler::target::KernelAbiAllocationRole;
use seismic_compiler::target::{DeviceContract, ExecutionProfile};
use seismic_lang::expr::compiled::{Compiled, InvocationValues};
use seismic_lang::expr::SymbolValue;
use seismic_lang::types::DType;
use std::ffi::c_void;
use std::sync::Arc;

#[derive(Clone)]
pub struct Device {
    context: Arc<Context>,
    stream: Arc<Stream>,
    contract: Arc<DeviceContract<Cuda>>,
    execution: Arc<ExecutionProfile<Cuda>>,
}

impl Device {
    pub fn open(ordinal: u32) -> Result<Self, ExecutionError> {
        let driver = Driver::load().map_err(ExecutionError::SubmissionFailed)?;
        let ordinal = i32::try_from(ordinal).map_err(|_| {
            ExecutionError::SubmissionFailed("CUDA device ordinal exceeds driver ABI".into())
        })?;
        let context = Context::retain(driver, ordinal).map_err(submission_error)?;
        let stream = Stream::new(&context).map_err(submission_error)?;
        let (contract, execution) =
            crate::profile::profile_for_opened(ordinal as u32, &context, &stream)
                .map_err(|error| ExecutionError::SubmissionFailed(error.to_string()))?;
        Ok(Self {
            context,
            stream,
            contract: Arc::new(contract),
            execution: Arc::new(execution),
        })
    }
    pub fn ordinal(&self) -> u32 {
        self.context.ordinal() as u32
    }
    pub fn contract(&self) -> &Arc<DeviceContract<Cuda>> {
        &self.contract
    }
    pub fn execution_profile(&self) -> &Arc<ExecutionProfile<Cuda>> {
        &self.execution
    }
}

impl DeviceService<Cuda> for Device {
    type Buffer = Buffer;
    fn allocate(&self, bytes: u64, alignment: u64) -> Result<Buffer, ExecutionError> {
        if alignment > 256 || !alignment.is_power_of_two() {
            panic!("prepared CUDA allocation asks for unsupported alignment")
        }
        let bytes = usize::try_from(bytes).map_err(|_| {
            ExecutionError::AllocationFailed("CUDA allocation exceeds host address space".into())
        })?;
        Allocation::new(&self.context, bytes)
            .map(Buffer::new)
            .map_err(allocation_error)
    }
    fn write(&self, buffer: &Buffer, offset: u64, bytes: &[u8]) -> Result<(), ExecutionError> {
        let offset = usize::try_from(offset)
            .unwrap_or_else(|_| panic!("prepared CUDA upload offset exceeds usize"));
        buffer
            .allocation
            .upload_at(offset, bytes)
            .map_err(submission_error)
    }
    fn read(&self, buffer: &Buffer, offset: u64, into: &mut [u8]) -> Result<(), ExecutionError> {
        let offset = usize::try_from(offset)
            .unwrap_or_else(|_| panic!("prepared CUDA download offset exceeds usize"));
        buffer
            .allocation
            .download_at(offset, into)
            .map_err(submission_error)
    }
    fn buffer_len(&self, buffer: &Buffer) -> u64 {
        buffer.len()
    }
}

pub struct Executor {
    device: Device,
}
impl Executor {
    pub fn open(ordinal: u32) -> Result<Self, ExecutionError> {
        Ok(Self {
            device: Device::open(ordinal)?,
        })
    }
    pub fn new(device: Device) -> Self {
        Self { device }
    }
    pub fn device(&self) -> &Device {
        &self.device
    }
}

impl NativeExecutor<Cuda> for Executor {
    type Device = Device;
    type Submission = Submission;

    fn begin_submission(&self) -> Result<Self::Submission, ExecutionError> {
        Ok(Submission {
            device: self.device.clone(),
            in_flight_uploads: Vec::new(),
        })
    }
}

pub struct Submission {
    device: Device,
    in_flight_uploads: Vec<PinnedUpload>,
}

pub struct Execution {
    device: Device,
    in_flight_uploads: Vec<PinnedUpload>,
}

impl NativeSubmission<Cuda> for Submission {
    type Device = Device;
    type Execution = Execution;
    fn execute(
        &mut self,
        command: &ExecutableCommand<Cuda>,
        env: &mut ExecutionEnvironment<'_, Cuda, Device>,
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
                addressable_resources,
                local_totals,
                scratch,
                abi,
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
            } => self.copy(source, destination, evaluate(bytes, env.values), env),
            ExecutableCommand::Fill {
                destination,
                value,
                bytes,
            } => self.fill(destination, *value, evaluate(bytes, env.values), env),
            ExecutableCommand::ScalarMove { from, to } => {
                let value = env
                    .values
                    .get(from.symbol)
                    .unwrap_or_else(|| panic!("prepared scalar move reads an unbound slot"));
                env.set_slot(*to, value);
                Ok(())
            }
            ExecutableCommand::ScalarRead {
                source,
                bounds: _,
                byte_offset,
                to,
            } => self.scalar_read(source, evaluate(byte_offset, env.values), *to, env),
        }
    }
    fn submit(mut self) -> Result<Self::Execution, ExecutionError> {
        Ok(Execution {
            device: self.device.clone(),
            in_flight_uploads: std::mem::take(&mut self.in_flight_uploads),
        })
    }
}

impl NativeExecution for Execution {
    fn complete(mut self) -> Result<(), ExecutionError> {
        synchronize(&self.device, &mut self.in_flight_uploads)
    }
}

impl Submission {
    #[allow(clippy::too_many_arguments)]
    fn launch(
        &mut self,
        kernel_id: ExecutableKernelId,
        mode: CudaLaunchMode,
        grid_exprs: &[seismic_lang::expr::compiled::CompiledNat; 3],
        workgroup_exprs: &[seismic_lang::expr::compiled::CompiledNat; 3],
        empty: &seismic_lang::expr::compiled::CompiledPredicate,
        bindings: &[CompiledBufferView],
        nat_args: &[seismic_lang::expr::compiled::CompiledNat],
        scalar_args: &[seismic_lang::expr::SymbolId],
        locals: &[CompiledLocalLayout],
        addressable_resources: &[seismic_compiler::executable::CompiledAddressableResource],
        totals: &CompiledLocalClassTotals,
        scratch_bindings: seismic_compiler::executable::LaunchScratchBindings,
        abi: &KernelAbiBindings,
        env: &mut ExecutionEnvironment<'_, Cuda, Device>,
    ) -> Result<(), ExecutionError> {
        if evaluate(empty, env.values) {
            return Ok(());
        }
        let kernel = env.kernel(kernel_id).handle();
        let grid = grid_exprs
            .each_ref()
            .map(|value| evaluate(value, env.values));
        let workgroup = workgroup_exprs
            .each_ref()
            .map(|value| evaluate(value, env.values));
        if grid.contains(&0) || workgroup.contains(&0) {
            return Ok(());
        }
        let mut buffer_words = Vec::with_capacity(bindings.len());
        let mut words = vec![0u64; kernel.layout.words.total as usize];
        for (position, binding) in bindings.iter().enumerate() {
            let resolved = env.resolve_view(binding);
            buffer_words.push(
                resolved
                    .buffer
                    .pointer()
                    .checked_add(resolved.byte_offset)
                    .unwrap_or_else(|| panic!("prepared CUDA buffer address overflows")),
            );
            let layout = kernel.layout.words.bindings[position];
            words[layout.first as usize..layout.first as usize + layout.rank as usize]
                .copy_from_slice(&resolved.extents);
            words[layout.first as usize + layout.rank as usize
                ..layout.first as usize + 2 * layout.rank as usize]
                .copy_from_slice(&resolved.strides);
        }
        for (index, value) in nat_args.iter().enumerate() {
            words[kernel.layout.words.nat_first as usize + index] = evaluate(value, env.values);
        }
        for (index, symbol) in scalar_args.iter().copied().enumerate() {
            words[kernel.layout.words.scalar_first as usize + index] = encode_symbol(
                env.values
                    .get(symbol)
                    .unwrap_or_else(|| panic!("prepared CUDA scalar argument is unbound")),
            );
        }
        let workgroup_bytes = evaluate(&totals.workgroup_bytes, env.values);
        let participant_bytes = evaluate(&totals.participant_bytes, env.values);
        let register_bytes = evaluate(&totals.register_bytes, env.values);
        for (position, local) in locals.iter().enumerate() {
            let layout = kernel.layout.words.locals[position];
            let offset = evaluate(&local.byte_offset, env.values);
            words[layout.first as usize] = offset;
            for (axis, value) in local.extents.iter().enumerate() {
                words[layout.first as usize + 1 + axis] = evaluate(value, env.values);
            }
            for (axis, value) in local.strides.iter().enumerate() {
                words[layout.first as usize + 1 + layout.rank as usize + axis] =
                    evaluate(value, env.values);
            }
        }
        for (position, resource) in addressable_resources.iter().enumerate() {
            let layout = kernel.layout.words.addressable_resources[position];
            words[layout.offset_units as usize] = evaluate(&resource.offset_units, env.values);
            words[layout.units as usize] = evaluate(&resource.units, env.values);
        }
        words[kernel.layout.words.grid_first as usize..kernel.layout.words.grid_first as usize + 3]
            .copy_from_slice(&grid);
        words[kernel.layout.words.workgroup_first as usize
            ..kernel.layout.words.workgroup_first as usize + 3]
            .copy_from_slice(&workgroup);
        words[kernel.layout.words.local_total_first as usize
            ..kernel.layout.words.local_total_first as usize + 3]
            .copy_from_slice(&[workgroup_bytes, participant_bytes, register_bytes]);

        let buffers = abi_buffer(abi, KernelAbiAllocationRole::BufferTable, env);
        let word_table = abi_buffer(abi, KernelAbiAllocationRole::WordTable, env);
        let results = abi_buffer(abi, KernelAbiAllocationRole::ScalarResults, env);
        if scratch_bindings
            .allocation(seismic_compiler::storage::LaunchLocalKind::Workgroup)
            .is_some()
        {
            panic!("CUDA native-dynamic workgroup storage received an invocation scratch binding");
        }
        let participant = scratch_pointer(
            scratch_bindings.allocation(seismic_compiler::storage::LaunchLocalKind::Participant),
            env,
        );
        let registers = scratch_pointer(
            scratch_bindings.allocation(seismic_compiler::storage::LaunchLocalKind::Register),
            env,
        );
        let frame = LaunchFrame {
            buffers: abi_pointer(buffers),
            words: abi_pointer(word_table),
            results: abi_pointer(results),
            participant_scratch: participant,
            register_scratch: registers,
        };
        let frame_alloc = abi_buffer(abi, KernelAbiAllocationRole::LaunchFrame, env);
        let buffer_bytes = words_bytes(&buffer_words);
        let word_bytes = words_bytes(&words);
        let frame_bytes = unsafe {
            std::slice::from_raw_parts(
                (&frame as *const LaunchFrame).cast::<u8>(),
                std::mem::size_of::<LaunchFrame>(),
            )
        };
        self.enqueue_uploads(&[
            (buffers, buffer_bytes.as_slice()),
            (word_table, word_bytes.as_slice()),
            (frame_alloc, frame_bytes),
        ])?;
        launch(
            &self.device,
            kernel,
            mode,
            abi_pointer(frame_alloc),
            grid,
            workgroup,
            workgroup_bytes,
        )?;
        // Ordinary launches remain asynchronous in the opened device's
        // production stream. A
        // schedule scalar is a host-side dependency for subsequent command
        // evaluation, so only a publishing launch synchronizes and reads its
        // canonical result table here. The prepared invocation's final
        // synchronize covers every non-publishing launch.
        let result_slots = kernel.layout.result_slots.clone();
        if !result_slots.is_empty() {
            self.synchronize_pending()?;
            let mut result_words = vec![0u8; result_slots.len() * 8];
            results
                .buffer
                .allocation
                .download_at(abi_offset(results), &mut result_words)
                .map_err(synchronization_error)?;
            for (index, (slot, dtype)) in result_slots.iter().enumerate() {
                let raw = u64::from_le_bytes(
                    result_words[index * 8..index * 8 + 8]
                        .try_into()
                        .expect("fixed result word"),
                );
                env.set_slot(*slot, decode_symbol(raw, *dtype));
            }
        }
        Ok(())
    }

    fn enqueue_uploads(
        &mut self,
        uploads: &[(&RuntimeBuffer<Buffer>, &[u8])],
    ) -> Result<(), ExecutionError> {
        let total = uploads
            .iter()
            .try_fold(0usize, |total, (_, bytes)| total.checked_add(bytes.len()))
            .unwrap_or_else(|| panic!("closed CUDA ABI upload batch exceeds host address space"));
        if total == 0 {
            return Ok(());
        }
        let mut staging =
            PinnedUpload::new(&self.device.context, total).map_err(allocation_error)?;
        let mut segments = Vec::with_capacity(uploads.len());
        let mut source_offset = 0usize;
        for (buffer, bytes) in uploads {
            if u64::try_from(bytes.len()).unwrap_or(u64::MAX) > buffer.accessible_bytes {
                panic!("closed CUDA ABI allocation is smaller than its canonical upload")
            }
            let end = source_offset + bytes.len();
            staging.bytes_mut()[source_offset..end].copy_from_slice(bytes);
            segments.push((source_offset, abi_pointer(buffer), bytes.len()));
            source_offset = end;
        }
        for (source_offset, destination, bytes) in segments {
            if let Err(error) =
                staging.upload(&self.device.stream, source_offset, destination, bytes)
            {
                // A successfully enqueued prefix may still read this memory.
                // Transfer ownership before reporting the real driver error.
                self.in_flight_uploads.push(staging);
                return Err(submission_error(error));
            }
        }
        self.in_flight_uploads.push(staging);
        Ok(())
    }

    fn synchronize_pending(&mut self) -> Result<(), ExecutionError> {
        synchronize(&self.device, &mut self.in_flight_uploads)
    }

    fn copy(
        &self,
        source: &CompiledBufferView,
        destination: &CompiledBufferView,
        bytes: u64,
        env: &ExecutionEnvironment<'_, Cuda, Device>,
    ) -> Result<(), ExecutionError> {
        let source = env.resolve_view(source);
        let destination = env.resolve_view(destination);
        let count =
            usize::try_from(bytes).unwrap_or_else(|_| panic!("prepared CUDA copy exceeds usize"));
        let _current = self.device.context.enter().map_err(submission_error)?;
        unsafe {
            self.device
                .context
                .driver
                .check(
                    (self.device.context.driver.memcpy_device_async)(
                        destination.buffer.pointer() + destination.byte_offset,
                        source.buffer.pointer() + source.byte_offset,
                        count,
                        self.device.stream.raw(),
                    ),
                    "device copy",
                )
                .map_err(submission_error)
        }
    }
    fn fill(
        &self,
        destination: &CompiledBufferView,
        value: FillValue,
        bytes: u64,
        env: &ExecutionEnvironment<'_, Cuda, Device>,
    ) -> Result<(), ExecutionError> {
        let destination = env.resolve_view(destination);
        let pointer = destination.buffer.pointer() + destination.byte_offset;
        let count = usize::try_from(bytes / value.width())
            .unwrap_or_else(|_| panic!("prepared CUDA fill exceeds usize"));
        if bytes % value.width() != 0 {
            panic!("prepared CUDA fill is not element aligned")
        }
        let _current = self.device.context.enter().map_err(submission_error)?;
        let driver = &self.device.context.driver;
        unsafe {
            match value {
                FillValue::U8(pattern) => driver.check(
                    (driver.memset_d8_async)(pointer, pattern[0], count, self.device.stream.raw()),
                    "device fill",
                ),
                FillValue::U16(pattern) => driver.check(
                    (driver.memset_d16_async)(
                        pointer,
                        u16::from_le_bytes(pattern),
                        count,
                        self.device.stream.raw(),
                    ),
                    "device fill",
                ),
                FillValue::U32(pattern) => driver.check(
                    (driver.memset_d32_async)(
                        pointer,
                        u32::from_le_bytes(pattern),
                        count,
                        self.device.stream.raw(),
                    ),
                    "device fill",
                ),
            }
            .map_err(submission_error)
        }
    }
    fn scalar_read(
        &mut self,
        source: &CompiledBufferView,
        offset: u64,
        to: AnyScalarSlot,
        env: &mut ExecutionEnvironment<'_, Cuda, Device>,
    ) -> Result<(), ExecutionError> {
        // A schedule scalar is a host dependency. Complete the exact
        // production stream before the synchronous host read and release
        // any pinned uploads whose DMA has now completed.
        self.synchronize_pending()?;
        let source = env.resolve_view(source);
        let mut bytes = [0u8; 4];
        let width = to.dtype.bytes() as usize;
        let absolute = source
            .byte_offset
            .checked_add(offset)
            .unwrap_or_else(|| panic!("prepared CUDA scalar-read offset overflows"));
        source
            .buffer
            .allocation
            .download_at(
                usize::try_from(absolute)
                    .unwrap_or_else(|_| panic!("prepared CUDA scalar-read offset exceeds usize")),
                &mut bytes[..width],
            )
            .map_err(synchronization_error)?;
        env.set_slot(to, decode_bytes(bytes, to.dtype));
        Ok(())
    }
}

fn launch(
    device: &Device,
    kernel: &CompiledKernel,
    mode: CudaLaunchMode,
    frame: u64,
    grid: [u64; 3],
    block: [u64; 3],
    shared: u64,
) -> Result<(), ExecutionError> {
    let _keep_module_alive = &kernel.module;
    if kernel.module.context.ordinal() != device.context.ordinal() {
        panic!("compiled CUDA kernel and execution device use different primary contexts")
    }
    let grid = grid.map(|v| {
        u32::try_from(v).unwrap_or_else(|_| panic!("prepared CUDA grid exceeds driver ABI"))
    });
    let block = block.map(|v| {
        u32::try_from(v).unwrap_or_else(|_| panic!("prepared CUDA workgroup exceeds driver ABI"))
    });
    let shared = u32::try_from(shared)
        .unwrap_or_else(|_| panic!("prepared CUDA workgroup storage exceeds driver ABI"));
    let mut frame = frame;
    let mut params = [(&mut frame as *mut u64).cast::<c_void>()];
    let context = &kernel.module.context;
    let _current = context.enter().map_err(submission_error)?;
    unsafe {
        let submit = match mode {
            CudaLaunchMode::Independent => context.driver.launch,
            CudaLaunchMode::CooperativeGrid => context.driver.launch_cooperative,
        };
        context
            .driver
            .check(
                submit(
                    kernel.function,
                    grid[0],
                    grid[1],
                    grid[2],
                    block[0],
                    block[1],
                    block[2],
                    shared,
                    device.stream.raw(),
                    params.as_mut_ptr(),
                    std::ptr::null_mut(),
                ),
                "kernel launch",
            )
            .map_err(submission_error)
    }
}

fn scratch_pointer(
    allocation: Option<ExecutableAllocationId>,
    env: &ExecutionEnvironment<'_, Cuda, Device>,
) -> u64 {
    let Some(allocation) = allocation else {
        return 0;
    };
    let runtime = env.buffer(allocation);
    runtime
        .buffer
        .pointer()
        .checked_add(runtime.base_offset)
        .unwrap_or_else(|| panic!("core scratch binding address overflows"))
}
fn abi_buffer<'a>(
    bindings: &KernelAbiBindings,
    role: KernelAbiAllocationRole,
    env: &'a ExecutionEnvironment<'_, Cuda, Device>,
) -> &'a RuntimeBuffer<Buffer> {
    env.buffer(bindings.allocation(role))
}

fn synchronize(
    device: &Device,
    in_flight_uploads: &mut Vec<PinnedUpload>,
) -> Result<(), ExecutionError> {
    device.stream.synchronize().map_err(synchronization_error)?;
    in_flight_uploads.clear();
    Ok(())
}

fn abi_offset(buffer: &RuntimeBuffer<Buffer>) -> usize {
    usize::try_from(buffer.base_offset)
        .unwrap_or_else(|_| panic!("prepared CUDA ABI offset exceeds usize"))
}

fn abi_pointer(buffer: &RuntimeBuffer<Buffer>) -> u64 {
    buffer
        .buffer
        .pointer()
        .checked_add(buffer.base_offset)
        .unwrap_or_else(|| panic!("prepared CUDA ABI address overflows"))
}

fn words_bytes(values: &[u64]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}
fn evaluate<T: Copy>(value: &Compiled<T>, values: &InvocationValues) -> T {
    value
        .evaluate(values)
        .unwrap_or_else(|error| panic!("prepared CUDA expression failed evaluation: {error:?}"))
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
        DType::F16 => SymbolValue::F16(u16::from_le_bytes(
            value[..2].try_into().expect("two bytes"),
        )),
        DType::BF16 => SymbolValue::BF16(u16::from_le_bytes(
            value[..2].try_into().expect("two bytes"),
        )),
        DType::I32 => SymbolValue::I32(i32::from_le_bytes(value)),
        DType::U32 => SymbolValue::U32(u32::from_le_bytes(value)),
        DType::Bool => SymbolValue::Bool(value[0] != 0),
    }
}
fn allocation_error(error: DriverError) -> ExecutionError {
    if error.is_device_loss() {
        ExecutionError::DeviceLost(error.to_string())
    } else {
        ExecutionError::AllocationFailed(error.to_string())
    }
}
fn submission_error(error: DriverError) -> ExecutionError {
    if error.is_device_loss() {
        ExecutionError::DeviceLost(error.to_string())
    } else {
        ExecutionError::SubmissionFailed(error.to_string())
    }
}
fn synchronization_error(error: DriverError) -> ExecutionError {
    if error.is_device_loss() {
        ExecutionError::DeviceLost(error.to_string())
    } else {
        ExecutionError::SynchronizationFailed(error.to_string())
    }
}

impl Drop for Submission {
    fn drop(&mut self) {
        if self.in_flight_uploads.is_empty() {
            return;
        }
        if self.device.stream.synchronize().is_ok() {
            self.in_flight_uploads.clear();
        } else {
            // Never free pinned memory that a live DMA may still read. On a
            // lost device/context this is a bounded process-lifetime leak;
            // it is safer than violating the CUDA async-copy contract.
            std::mem::forget(std::mem::take(&mut self.in_flight_uploads));
        }
    }
}
