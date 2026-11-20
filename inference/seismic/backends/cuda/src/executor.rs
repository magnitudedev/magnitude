//! CUDA execution of the core-owned command vocabulary.

use crate::blocks::Blocks;
use crate::buffer::Buffer;
use crate::command::{CompiledKernel, LaunchFrame};
use crate::driver::{Context, Driver, DriverError, HostMapped, PinnedUpload, Stream};
use crate::{Cuda, CudaLaunchMode};
use seismic_compiler::errors::{ExecutionError, RefusalReading, TargetError};
use seismic_compiler::executable::{
    CompiledBufferView, CompiledLocalClassTotals, CompiledLocalLayout, DeviceService,
    ExecutableAllocationId, ExecutableCommand, ExecutableKernelId, ExecutionEnvironment,
    KernelAbiBindings, NativeExecution, NativeExecutor, NativeSubmission, RuntimeBuffer,
};
use seismic_ir::physical_target::{KernelAbiAllocationRole, KernelWordValues, LocalWordValues};
use seismic_ir::schedule::{AnyScalarSlot, FillValue};
use seismic_lang::expr::compiled::InvocationValues;
use seismic_memory::entry::{Entry, Store};
use std::ffi::c_void;
use std::sync::Arc;

/// A native stream of the device: the one its native work runs on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NativeStream {
    /// Compute.
    Compute,
    /// Fill work beside compute: moves into device memory and their waits,
    /// whose spins hold only later fills.
    Fill,
}

/// The embedder's ledgers of the stores a device's memory lies in, which it
/// is opened with: its device memory (`device`: blocks, graphs' driver
/// memory, the stack the driver grows) and host RAM (`host`: pinned host
/// memory, staging and the words device work signals). Where the device's
/// memory is the host's, one store twice.
#[derive(Clone)]
pub struct CudaStores {
    pub device: Arc<dyn Store>,
    pub host: Arc<dyn Store>,
}

#[derive(Clone)]
pub struct Device {
    context: Arc<Context>,
    stores: CudaStores,
    stream: Arc<Stream>,
    /// The fill stream ([`NativeStream::Fill`]).
    fill: Arc<Stream>,
    blocks: Arc<Blocks>,
    graphs: Arc<crate::graph_cost::GraphPool>,
    /// The word a graph's word nodes name as it is recorded, before a
    /// replay binds its run's own ([`Self::placeholder_word`]).
    placeholder: Arc<crate::direct::HostWord>,
    launch_limits: crate::profile::DeviceLaunchLimits,
}

/// One `cuMemGetInfo` sample.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryInfo {
    pub free_bytes: u64,
    pub total_bytes: u64,
}

/// `context`'s device memory as a refusal reads it ([`memory_info`]).
pub(crate) fn device_free(context: &Context) -> RefusalReading {
    memory_info(context).map_or(RefusalReading::Unread, |info| RefusalReading::DeviceFree {
        free: info.free_bytes,
        total: info.total_bytes,
    })
}

/// `cuMemGetInfo` on `context`: device-wide free memory and the total
/// memory of the exposed device ([`Device::memory_info`]).
pub(crate) fn memory_info(context: &Context) -> Result<MemoryInfo, ExecutionError> {
    let _current = context.enter().map_err(submission_error)?;
    let driver = &context.driver;
    let (mut free, mut total) = (0usize, 0usize);
    unsafe {
        driver
            .check((driver.memory_info)(&mut free, &mut total), "memory info")
            .map_err(submission_error)?;
    }
    Ok(MemoryInfo {
        free_bytes: free as u64,
        total_bytes: total as u64,
    })
}

impl Device {
    /// Device `ordinal`, what it holds that no caller names charged to its
    /// stores' own entries.
    pub fn open(ordinal: u32, stores: CudaStores) -> Result<Self, TargetError> {
        let unavailable = |error: ExecutionError| TargetError::DeviceUnavailable(error.to_string());
        let driver = Driver::load().map_err(|error| unavailable(ExecutionError::SubmissionFailed(error)))?;
        let ordinal = i32::try_from(ordinal).map_err(|_| {
            unavailable(ExecutionError::SubmissionFailed("CUDA device ordinal exceeds driver ABI".into()))
        })?;
        let context = Context::retain(driver, ordinal).map_err(opening_error)?;
        let stream = Stream::new(&context).map_err(opening_error)?;
        let fill = Stream::new(&context).map_err(opening_error)?;
        let blocks = Arc::new(Blocks::new(context.clone(), stores.device.own()).map_err(opening_error)?);
        if let Some(memory) = counted_driver_memory(context.ordinal() as u32).map_err(unavailable)? {
            assert!(
                context.driver_memory.set(memory).is_ok(),
                "a context's driver memory is counted once, at its opening"
            );
        }
        let placeholder = Arc::new(crate::direct::HostWord::on(&context, &stores.host.own()).map_err(unavailable)?);
        let graphs = Arc::new(crate::graph_cost::GraphPool::new(stores.device.own(), context.clone()));
        let launch_limits = crate::profile::DeviceLaunchLimits::query(&context.driver, context.ordinal())?;
        context.charge_own_to(stores.device.own());
        Ok(Self {
            context,
            stores,
            stream,
            fill,
            blocks,
            graphs,
            placeholder,
            launch_limits,
        })
    }

    /// Whether the process's footprint holds this device's allocations
    /// (`seismic_native_target::process::ProcessMemory`): where the device's
    /// memory is the host's, its context counts what the driver holds for
    /// the process.
    pub fn allocations_in_process_footprint(&self) -> bool {
        self.context.driver_memory.get().is_some()
    }
    pub fn ordinal(&self) -> u32 {
        self.context.ordinal() as u32
    }

    /// The device's memory as a refusal reads it ([`device_free`]).
    pub(crate) fn refusal_reading(&self) -> RefusalReading {
        device_free(&self.context)
    }

    /// `cuMemGetInfo` on this opened context: device-wide free memory (other
    /// processes' and this process's allocations already excluded) and the
    /// total memory of the exposed device.
    pub fn memory_info(&self) -> Result<MemoryInfo, ExecutionError> {
        memory_info(&self.context)
    }

    /// Device storage of `bytes` as a range of a block, its bytes carved
    /// for `entry`, an entry of the device store.
    pub fn allocate_device(&self, bytes: u64, alignment: u64, entry: &Arc<dyn Entry>) -> Result<Buffer, ExecutionError> {
        if alignment > 256 || !alignment.is_power_of_two() {
            panic!("prepared CUDA allocation asks for unsupported alignment")
        }
        self.blocks.allocate(bytes, alignment, entry).map(Buffer::device)
    }

    /// The word every graph's word nodes name as it is recorded, held from
    /// the device's opening so a recording allocates no memory of the
    /// device's ledger.
    pub fn placeholder_word(&self) -> &crate::direct::HostWord {
        &self.placeholder
    }

    /// What the device's instantiated graphs hold
    /// ([`crate::graph_cost::GraphPool`]).
    pub(crate) fn graphs(&self) -> &Arc<crate::graph_cost::GraphPool> {
        &self.graphs
    }

    pub(crate) fn context(&self) -> &Arc<Context> {
        &self.context
    }

    pub(crate) fn launch_contract(
        &self,
        max_threads_per_block: u64,
    ) -> seismic_compiler::direct_launch::DirectLaunchContract {
        seismic_compiler::direct_launch::DirectLaunchContract {
            max_buffer_slots: None,
            max_grid: self.launch_limits.max_grid,
            max_group_size: Some(self.launch_limits.max_block),
            max_group_threads: Some(max_threads_per_block),
            max_shared_bytes: Some(self.launch_limits.max_shared_bytes),
            max_param_bytes: Some(self.launch_limits.max_param_bytes),
        }
    }

    /// The stores the device was opened with.
    pub(crate) fn stores(&self) -> &CudaStores {
        &self.stores
    }

    /// The compute stream.
    pub(crate) fn stream(&self) -> &Arc<Stream> {
        &self.stream
    }

    /// Native stream `which`.
    pub(crate) fn native_stream(&self, which: NativeStream) -> &Arc<Stream> {
        match which {
            NativeStream::Compute => &self.stream,
            NativeStream::Fill => &self.fill,
        }
    }

    /// Wait until every launch queued so far on the device's native streams
    /// has completed, sleeping ([`crate::driver::Stream::synchronize`]).
    pub fn drain(&self) -> Result<(), ExecutionError> {
        self.stream.synchronize().map_err(submission_error)?;
        self.fill.synchronize().map_err(submission_error)
    }

    /// Whether every launch queued on the device's native streams has
    /// completed, as of now: it waits for nothing.
    pub fn idle(&self) -> Result<bool, ExecutionError> {
        Ok(self.stream.query().map_err(submission_error)? && self.fill.query().map_err(submission_error)?)
    }

    /// Pinned host memory mapped into the device address space, for inputs
    /// the host fills before each submission: the host writes it in place,
    /// without a driver copy that would wait for queued device work. Its
    /// whole host pages are charged to `entry`, an entry of host RAM.
    pub fn allocate_mapped(&self, bytes: u64, entry: &Arc<dyn Entry>) -> Result<Buffer, ExecutionError> {
        let bytes = usize::try_from(bytes).map_err(|_| {
            ExecutionError::AllocationFailed("CUDA allocation exceeds host address space".into())
        })?;
        HostMapped::new(&self.context, entry, bytes).map(Buffer::mapped)
    }

    /// One block of a granule pool, of `kind`: a native allocation of its
    /// own (never a range of a shared block), every byte backed and zero,
    /// charged exactly to `entry` and returned to the driver and the OS as
    /// it drops. Device memory is one `cuMemAlloc`; host memory is a range
    /// of host pages (large pages where the host has them) backed whole and
    /// registered for the device (`cuMemHostRegister`), which reads it in
    /// place.
    pub fn allocate_block(&self, kind: u32, bytes: u64, entry: &Arc<dyn Entry>, fill: seismic_memory::SlabFill) -> Result<Buffer, ExecutionError> {
        match kind {
            crate::topology::DEVICE_KIND => self.blocks.allocate_block(bytes, entry, fill).map(Buffer::block),
            crate::topology::PINNED_KIND => {
                if self.allocations_in_process_footprint()
                    && crate::driver::can_create_host_node(&self.context).map_err(submission_error)?
                {
                    crate::driver::HostNodeBlock::new(&self.context, entry, bytes).map(Buffer::host_node)
                } else {
                    crate::registered::RegisteredBlock::new(&self.context, entry, bytes).map(Buffer::registered)
                }
            }
            kind => panic!("invariant violated: CUDA has no memory kind {kind}"),
        }
    }

}

impl DeviceService<Cuda> for Device {
    type Buffer = Buffer;
    /// CUDA maps no read-only host memory: every buffer is one it allocated.
    type Writable = Buffer;
    /// Storage of `kind` (`crate::topology`): a range of a device block, or
    /// pinned host memory.
    fn allocate(&self, kind: u32, bytes: u64, alignment: u64, entry: &Arc<dyn Entry>) -> Result<Buffer, ExecutionError> {
        match kind {
            crate::topology::DEVICE_KIND => self.allocate_device(bytes, alignment, entry),
            crate::topology::PINNED_KIND => self.allocate_mapped(bytes, entry),
            kind => panic!("invariant violated: CUDA has no memory kind {kind}"),
        }
    }
    fn write(&self, buffer: &Buffer, offset: u64, bytes: &[u8]) -> Result<(), ExecutionError> {
        let offset = usize::try_from(offset)
            .unwrap_or_else(|_| panic!("prepared CUDA upload offset exceeds usize"));
        buffer.upload_at(offset, bytes).map_err(submission_error)
    }
    fn read(&self, buffer: &Buffer, offset: u64, into: &mut [u8]) -> Result<(), ExecutionError> {
        let offset = usize::try_from(offset)
            .unwrap_or_else(|_| panic!("prepared CUDA download offset exceeds usize"));
        buffer.download_at(offset, into).map_err(submission_error)
    }
    fn buffer_len(&self, buffer: &Buffer) -> u64 {
        buffer.len()
    }
    fn host_address(&self, buffer: &Buffer) -> Option<std::ptr::NonNull<u8>> {
        buffer.host_address()
    }
}

pub struct Executor {
    device: Device,
}
impl Executor {
    pub fn new(device: Device) -> Self {
        Self { device }
    }
    pub fn device(&self) -> &Device {
        &self.device
    }
}

impl NativeExecutor<Cuda> for Executor {
    type Handle = CompiledKernel;
    type Device = Device;
    type Submission = Submission;

    fn begin_submission(&self) -> Result<Self::Submission, ExecutionError> {
        Ok(Submission {
            device: self.device.clone(),
            in_flight_uploads: InFlightUploads::new(self.device.stream.clone()),
        })
    }
}

trait PendingStream {
    fn synchronize(&self) -> Result<(), DriverError>;
}

impl PendingStream for Arc<Stream> {
    fn synchronize(&self) -> Result<(), DriverError> {
        Stream::synchronize(self)
    }
}

struct InFlightUploads<S: PendingStream = Arc<Stream>, U = PinnedUpload> {
    stream: S,
    uploads: Vec<U>,
    enqueued: bool,
}

impl<S: PendingStream, U> InFlightUploads<S, U> {
    fn new(stream: S) -> Self {
        Self {
            stream,
            uploads: Vec::new(),
            enqueued: false,
        }
    }

    fn mark_enqueued(&mut self) {
        self.enqueued = true;
    }

    fn push(&mut self, upload: U) {
        self.enqueued = true;
        self.uploads.push(upload);
    }

    fn complete(&mut self) -> Result<(), ExecutionError> {
        self.stream
            .synchronize()
            .map_err(synchronization_error)?;
        self.uploads.clear();
        self.enqueued = false;
        Ok(())
    }
}

impl<S: PendingStream, U> Drop for InFlightUploads<S, U> {
    fn drop(&mut self) {
        if !self.enqueued && self.uploads.is_empty() {
            return;
        }
        if self.stream.synchronize().is_ok() {
            self.uploads.clear();
            self.enqueued = false;
        } else if !self.uploads.is_empty() {
            // Never free pinned memory that a live DMA may still read. On a
            // lost device/context this is a bounded process-lifetime leak;
            // it is safer than violating the CUDA async-copy contract.
            std::mem::forget(std::mem::take(&mut self.uploads));
        }
    }
}

pub struct Submission {
    device: Device,
    in_flight_uploads: InFlightUploads,
}

pub struct Execution {
    in_flight_uploads: InFlightUploads,
}

impl NativeSubmission<Cuda> for Submission {
    type Handle = CompiledKernel;
    type Device = Device;
    type Execution = Execution;
    fn execute(
        &mut self,
        command: &ExecutableCommand<Cuda>,
        env: &mut ExecutionEnvironment<'_, Cuda, CompiledKernel, Device>,
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
                addressable_resources,
                local_totals,
                scratch,
                abi,
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
            } => self.copy(source, destination, evaluate(bytes, env.values()), env),
            ExecutableCommand::Fill {
                destination,
                value,
                bytes,
            } => self.fill(destination, *value, evaluate(bytes, env.values()), env),
            ExecutableCommand::ScalarMove { from, to } => {
                let value = env
                    .values()
                    .get(from.symbol())
                    .unwrap_or_else(|| panic!("prepared scalar move reads an unbound slot"));
                env.set_slot(*to, value);
                Ok(())
            }
            ExecutableCommand::ScalarRead {
                source,
                bounds: _,
                byte_offset,
                to,
            } => self.scalar_read(source, evaluate(byte_offset, env.values()), *to, env),
        }
    }
    fn complete_prefix(&mut self) -> Result<(), ExecutionError> {
        self.synchronize_pending()
    }

    fn submit(self) -> Self::Execution {
        Execution {
            in_flight_uploads: self.in_flight_uploads,
        }
    }
}

impl NativeExecution for Execution {
    fn complete(&mut self) -> Result<(), ExecutionError> {
        self.in_flight_uploads.complete()
    }
}

impl Submission {
    /// One fill of `bytes` bytes at device address `pointer`
    /// ([`Submission::fill`]).
    fn fill_piece(
        device: &Device,
        in_flight: &mut InFlightUploads,
        issue: &crate::enqueue::Enqueue,
        pointer: u64,
        value: FillValue,
        bytes: u64,
    ) -> Result<(), ExecutionError> {
        let count = usize::try_from(bytes / value.width())
            .unwrap_or_else(|_| panic!("prepared CUDA fill exceeds usize"));
        let driver = &device.context.driver;
        unsafe {
            match value {
                FillValue::U8(pattern) => {
                    driver
                        .check(
                            issue.memset_d8_async(pointer, pattern[0], count, device.stream.raw()),
                            "device fill",
                        )
                        .map_err(submission_error)?;
                    in_flight.mark_enqueued();
                    Ok(())
                }
                FillValue::U16(pattern) => {
                    driver
                        .check(
                            issue.memset_d16_async(
                                pointer,
                                u16::from_le_bytes(pattern),
                                count,
                                device.stream.raw(),
                            ),
                            "device fill",
                        )
                        .map_err(submission_error)?;
                    in_flight.mark_enqueued();
                    Ok(())
                }
                FillValue::U32(pattern) => {
                    driver
                        .check(
                            issue.memset_d32_async(
                                pointer,
                                u32::from_le_bytes(pattern),
                                count,
                                device.stream.raw(),
                            ),
                            "device fill",
                        )
                        .map_err(submission_error)?;
                    in_flight.mark_enqueued();
                    Ok(())
                }
                // Each 32-bit half of every element: one strided pass per half,
                // or a single pass when the halves agree.
                FillValue::U64(pattern) => {
                    let low = u32::from_le_bytes([pattern[0], pattern[1], pattern[2], pattern[3]]);
                    let high = u32::from_le_bytes([pattern[4], pattern[5], pattern[6], pattern[7]]);
                    if low == high {
                        driver
                            .check(
                                issue.memset_d32_async(
                                    pointer,
                                    low,
                                    count * 2,
                                    device.stream.raw(),
                                ),
                                "device fill",
                            )
                            .map_err(submission_error)?;
                        in_flight.mark_enqueued();
                        Ok(())
                    } else {
                        for (offset, half) in [(0, low), (4, high)] {
                            driver
                                .check(
                                    issue.memset_d2d32_async(
                                        pointer + offset,
                                        8,
                                        half,
                                        1,
                                        count,
                                        device.stream.raw(),
                                    ),
                                    "device fill",
                                )
                                .map_err(submission_error)?;
                            in_flight.mark_enqueued();
                        }
                        Ok(())
                    }
                }
            }
        }
    }

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
        result_slots: &[seismic_ir::schedule::AnyScalarSlot],
        locals: &[CompiledLocalLayout],
        addressable_resources: &[seismic_compiler::executable::CompiledAddressableResource],
        totals: &CompiledLocalClassTotals,
        scratch_bindings: seismic_compiler::executable::LaunchScratchBindings,
        abi: &KernelAbiBindings,
        env: &mut ExecutionEnvironment<'_, Cuda, CompiledKernel, Device>,
    ) -> Result<(), ExecutionError> {
        if env.predicate(empty) {
            return Ok(());
        }
        let kernel = env.kernel_handle(kernel_id);
        let grid = grid_exprs
            .each_ref()
            .map(|value| evaluate(value, env.values()));
        let workgroup = workgroup_exprs
            .each_ref()
            .map(|value| evaluate(value, env.values()));
        if grid.contains(&0) || workgroup.contains(&0) {
            return Ok(());
        }
        let mut resolved_views = Vec::with_capacity(bindings.len());
        for binding in bindings {
            resolved_views.push(env.resolve_view(binding)?);
        }
        let buffer_words = resolved_views
            .iter()
            .map(|resolved| {
                resolved
                    .buffer
                    .pointer()
                    .checked_add(resolved.byte_offset)
                    .unwrap_or_else(|| panic!("prepared CUDA buffer address overflows"))
            })
            .collect::<Vec<_>>();
        let values = env.values();
        let workgroup_bytes = evaluate(&totals.workgroup_bytes, values);
        let participant_bytes = evaluate(&totals.participant_bytes, values);
        let register_bytes = evaluate(&totals.register_bytes, values);
        let words = kernel
            .layout
            .words
            .populate(KernelWordValues {
                nat_args: nat_args.iter().map(|value| evaluate(value, values)),
                scalar_args: scalar_args.iter().copied().map(|symbol| {
                    values
                        .get(symbol)
                        .unwrap_or_else(|| panic!("prepared CUDA scalar argument is unbound"))
                }),
                bindings: resolved_views
                    .iter()
                    .map(|view| (view.extents.as_slice(), view.strides.as_slice())),
                locals: locals.iter().map(|local| LocalWordValues {
                    byte_offset: evaluate(&local.byte_offset, values),
                    extents: local
                        .extents
                        .iter()
                        .map(move |value| evaluate(value, values)),
                    strides: local
                        .strides
                        .iter()
                        .map(move |value| evaluate(value, values)),
                }),
                addressable_resources: addressable_resources.iter().map(|resource| {
                    (
                        evaluate(&resource.offset_units, values),
                        evaluate(&resource.units, values),
                    )
                }),
                grid,
                workgroup,
                local_totals: [workgroup_bytes, participant_bytes, register_bytes],
            })
            .map_err(|error| {
                ExecutionError::ConstructionContradiction(format!(
                    "native scalar ABI quantity does not fit its planned word: {error:?}"
                ))
            })?;

        let buffers = abi_buffer(abi, KernelAbiAllocationRole::BufferTable, env);
        let word_table = abi_buffer(abi, KernelAbiAllocationRole::WordTable, env);
        let results = abi_buffer(abi, KernelAbiAllocationRole::ScalarResults, env);
        if scratch_bindings
            .allocation(seismic_ir::storage::LaunchLocalKind::Workgroup)
            .is_some()
        {
            panic!("CUDA native-dynamic workgroup storage received an invocation scratch binding");
        }
        let participant = scratch_pointer(
            scratch_bindings.allocation(seismic_ir::storage::LaunchLocalKind::Participant),
            env,
        );
        let registers = scratch_pointer(
            scratch_bindings.allocation(seismic_ir::storage::LaunchLocalKind::Register),
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
        self.in_flight_uploads.mark_enqueued();
        // Ordinary launches remain asynchronous in the opened device's
        // production stream. A
        // schedule scalar is a host-side dependency for subsequent command
        // evaluation, so only a publishing launch synchronizes and reads its
        // canonical result table here. The prepared invocation's final
        // synchronize covers every non-publishing launch.
        if !result_slots.is_empty() {
            self.synchronize_pending()?;
            let mut result_words = vec![0u8; result_slots.len() * 8];
            results
                .buffer
                .download_at(abi_offset(results), &mut result_words)
                .map_err(synchronization_error)?;
            for (index, slot) in result_slots.iter().enumerate() {
                let raw = u64::from_le_bytes(
                    result_words[index * 8..index * 8 + 8]
                        .try_into()
                        .expect("fixed result word"),
                );
                env.set_slot(*slot, slot.kind().decode_word(raw));
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
        let mut staging = PinnedUpload::new(&self.device.context, &self.device.stores.host.own(), total)?;
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
        self.in_flight_uploads.complete()
    }

    fn copy(
        &mut self,
        source: &CompiledBufferView,
        destination: &CompiledBufferView,
        bytes: u64,
        env: &ExecutionEnvironment<'_, Cuda, CompiledKernel, Device>,
    ) -> Result<(), ExecutionError> {
        let source = env.resolve_view(source)?;
        let destination = env.resolve_view(destination)?;
        let _current = self.device.context.enter().map_err(submission_error)?;
        let issue = self.device.context.enqueue();
        let count =
            usize::try_from(bytes).unwrap_or_else(|_| panic!("prepared CUDA copy exceeds usize"));
        unsafe {
            self.device
                .context
                .driver
                .check(
                    issue.memcpy_device_async(
                        destination.buffer.pointer() + destination.byte_offset,
                        source.buffer.pointer() + source.byte_offset,
                        count,
                        self.device.stream.raw(),
                    ),
                    "device copy",
                )
                .map_err(submission_error)?;
        }
        self.in_flight_uploads.mark_enqueued();
        Ok(())
    }
    fn fill(
        &mut self,
        destination: &CompiledBufferView,
        value: FillValue,
        bytes: u64,
        env: &ExecutionEnvironment<'_, Cuda, CompiledKernel, Device>,
    ) -> Result<(), ExecutionError> {
        let destination = env.resolve_view(destination)?;
        if bytes % value.width() != 0 || destination.byte_offset % value.width() != 0 {
            panic!("prepared CUDA fill is not element aligned")
        }
        let _current = self.device.context.enter().map_err(submission_error)?;
        let issue = self.device.context.enqueue();
        Self::fill_piece(
            &self.device,
            &mut self.in_flight_uploads,
            issue,
            destination.buffer.pointer() + destination.byte_offset,
            value,
            bytes,
        )
    }
    fn scalar_read(
        &mut self,
        source: &CompiledBufferView,
        offset: u64,
        to: AnyScalarSlot,
        env: &mut ExecutionEnvironment<'_, Cuda, CompiledKernel, Device>,
    ) -> Result<(), ExecutionError> {
        // A schedule scalar is a host dependency. Complete the exact
        // production stream before the synchronous host read and release
        // any pinned uploads whose DMA has now completed.
        self.synchronize_pending()?;
        let source = env.resolve_view(source)?;
        let mut bytes = [0u8; 8];
        let width = to.kind().bytes() as usize;
        let absolute = source
            .byte_offset
            .checked_add(offset)
            .unwrap_or_else(|| panic!("prepared CUDA scalar-read offset overflows"));
        source
            .buffer
            .download_at(
                usize::try_from(absolute)
                    .unwrap_or_else(|_| panic!("prepared CUDA scalar-read offset exceeds usize")),
                &mut bytes[..width],
            )
            .map_err(synchronization_error)?;
        env.set_slot(to, to.kind().decode_word(u64::from_le_bytes(bytes)));
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
    let module = kernel.module();
    if module.context().ordinal() != device.context.ordinal() {
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
    let context = module.context();
    let _current = context.enter().map_err(submission_error)?;
    let status = unsafe {
        context.enqueue().launch(
            mode == CudaLaunchMode::CooperativeGrid,
            module.function(kernel.function),
            grid,
            block,
            shared,
            device.stream.raw(),
            params.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    };
    context.driver.check(status, "kernel launch").map_err(submission_error)
}

fn scratch_pointer(
    allocation: Option<ExecutableAllocationId>,
    env: &ExecutionEnvironment<'_, Cuda, CompiledKernel, Device>,
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
    env: &'a ExecutionEnvironment<'_, Cuda, CompiledKernel, Device>,
) -> &'a RuntimeBuffer<Buffer> {
    env.buffer(bindings.allocation(role))
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
fn evaluate(value: &seismic_lang::expr::compiled::CompiledNat, values: &InvocationValues) -> u64 {
    value
        .evaluate_u64(values)
        .unwrap_or_else(|error| panic!("prepared CUDA expression failed evaluation: {error:?}"))
}
/// The error of an allocation of `bytes` the driver failed: a device loss,
/// a capacity refusal where the driver is out of memory
/// (`CUDA_ERROR_OUT_OF_MEMORY`), as every backend reports one, else a
/// failure.
/// `CUDA_ERROR_OUT_OF_MEMORY`.
pub(crate) const OUT_OF_MEMORY: i32 = 2;

pub(crate) fn allocation_error(
    bytes: u64,
    reading: impl FnOnce() -> RefusalReading,
) -> impl FnOnce(DriverError) -> ExecutionError {
    move |error| {
        if error.is_device_loss() {
            ExecutionError::DeviceLost(error.to_string())
        } else if error.code == OUT_OF_MEMORY {
            ExecutionError::DeviceRefused { required: bytes, reading: reading() }
        } else {
            ExecutionError::AllocationFailed(error.to_string())
        }
    }
}
/// Where device `ordinal`'s memory is the host's, its driver's memory of
/// this process, counted in the process's footprint while held: what its
/// context holds, read from NVML once, and every allocation made on it.
/// Windows reports no footprint at all.
#[cfg(not(target_os = "windows"))]
fn counted_driver_memory(
    ordinal: u32,
) -> Result<Option<Arc<seismic_native_target::process::DriverMemory>>, ExecutionError> {
    let descriptor = crate::profile::describe(ordinal)
        .map_err(|error| ExecutionError::SubmissionFailed(error.to_string()))?;
    if !descriptor.memory_is_host() {
        return Ok(None);
    }
    seismic_native_target::process::count_driver_memory(u128::from_le_bytes(descriptor.uuid), || {
        crate::nvml::context_bytes(descriptor.uuid)
    })
    .map(Some)
    .map_err(|error| {
        ExecutionError::SubmissionFailed(format!("the context's driver memory is unread: {error}"))
    })
}

#[cfg(target_os = "windows")]
fn counted_driver_memory(
    _ordinal: u32,
) -> Result<Option<Arc<seismic_native_target::process::DriverMemory>>, ExecutionError> {
    Ok(None)
}

/// Opening the device failed on `error`: busy where the driver is out of
/// memory or another process holds the device ([`DriverError::is_busy`]),
/// stated as the submission's failure.
fn opening_error(error: DriverError) -> TargetError {
    let busy = error.is_busy();
    let reason = submission_error(error).to_string();
    match busy {
        true => TargetError::DeviceBusy(reason),
        false => TargetError::DeviceUnavailable(reason),
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

#[cfg(test)]
mod tests {
    use super::{InFlightUploads, PendingStream};
    use crate::driver::DriverError;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::Arc;

    #[derive(Clone)]
    struct MockStream {
        syncs: Arc<AtomicUsize>,
        fail: Arc<AtomicBool>,
    }

    impl MockStream {
        fn new(fail: bool) -> Self {
            Self {
                syncs: Arc::new(AtomicUsize::new(0)),
                fail: Arc::new(AtomicBool::new(fail)),
            }
        }
    }

    impl PendingStream for MockStream {
        fn synchronize(&self) -> Result<(), DriverError> {
            self.syncs.fetch_add(1, Ordering::SeqCst);
            if self.fail.load(Ordering::SeqCst) {
                Err(DriverError {
                    operation: "stream synchronization",
                    code: 700,
                    description: "illegal address".into(),
                })
            } else {
                Ok(())
            }
        }
    }

    struct DropCounter(Arc<AtomicUsize>);

    impl Drop for DropCounter {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn empty_in_flight_does_not_synchronize_on_drop() {
        let stream = MockStream::new(false);
        let in_flight = InFlightUploads::<MockStream, DropCounter>::new(stream.clone());
        drop(in_flight);
        assert_eq!(stream.syncs.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn enqueued_work_without_uploads_synchronizes_on_early_drop() {
        let stream = MockStream::new(false);
        let mut in_flight = InFlightUploads::<MockStream, DropCounter>::new(stream.clone());
        in_flight.mark_enqueued();
        drop(in_flight);
        assert_eq!(stream.syncs.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn completed_work_does_not_synchronize_again_on_drop() {
        let stream = MockStream::new(false);
        let mut in_flight = InFlightUploads::<MockStream, DropCounter>::new(stream.clone());
        in_flight.mark_enqueued();
        in_flight.complete().unwrap();
        assert_eq!(stream.syncs.load(Ordering::SeqCst), 1);
        drop(in_flight);
        assert_eq!(stream.syncs.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn drop_frees_uploads_when_sync_succeeds_and_leaks_when_sync_fails() {
        let drops = Arc::new(AtomicUsize::new(0));

        let ok_stream = MockStream::new(false);
        let mut ok_in_flight = InFlightUploads::new(ok_stream.clone());
        ok_in_flight.push(DropCounter(drops.clone()));
        drop(ok_in_flight);
        assert_eq!(ok_stream.syncs.load(Ordering::SeqCst), 1);
        assert_eq!(drops.load(Ordering::SeqCst), 1);

        let err_stream = MockStream::new(true);
        let mut err_in_flight = InFlightUploads::new(err_stream.clone());
        err_in_flight.push(DropCounter(drops.clone()));
        drop(err_in_flight);
        assert_eq!(err_stream.syncs.load(Ordering::SeqCst), 1);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn failed_complete_retains_enqueued_flag_so_drop_retries_sync_and_leaks_uploads() {
        let drops = Arc::new(AtomicUsize::new(0));
        let stream = MockStream::new(true);
        let mut in_flight = InFlightUploads::new(stream.clone());
        in_flight.push(DropCounter(drops.clone()));
        assert!(in_flight.complete().is_err());
        assert_eq!(stream.syncs.load(Ordering::SeqCst), 1);
        drop(in_flight);
        assert_eq!(stream.syncs.load(Ordering::SeqCst), 2);
        assert_eq!(drops.load(Ordering::SeqCst), 0);
    }
}
