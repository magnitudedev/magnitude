//! CUDA execution of the core-owned command vocabulary.

use crate::blocks::Blocks;
use crate::buffer::Buffer;
use crate::command::{CompiledKernel, LaunchFrame};
use crate::driver::{
    Context, Driver, DriverError, HostMapped, PinnedUpload, RangeMemory, ReservedMemory, Stream,
};
use crate::{Cuda, CudaLaunchMode};
use seismic_compiler::errors::{ExecutionError, RefusalReading};
use seismic_compiler::executable::{
    CompiledBufferView, CompiledLocalClassTotals, CompiledLocalLayout, DeviceService,
    ExecutableAllocationId, ExecutableCommand, ExecutableKernelId, ExecutionEnvironment,
    KernelAbiBindings, NativeExecution, NativeExecutor, NativeSubmission, RuntimeBuffer,
};
use seismic_ir::physical_target::KernelAbiAllocationRole;
use seismic_ir::schedule::{AnyScalarSlot, FillValue};
use seismic_lang::expr::compiled::InvocationValues;
use seismic_lang::expr::SymbolValue;
use seismic_reservation::{time_backing, Backing, BackingPhase, Reserved, Resizer};
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

/// The accounts a device's own allocations are charged to, which it is
/// opened with: device memory (blocks, graphs' driver memory), pinned host
/// memory the device reads (`upload`) and pinned host staging (`host`).
#[derive(Clone)]
pub struct MemoryAccounts {
    pub device: Arc<dyn seismic_compiler::executable::MemoryAccount>,
    pub upload: Arc<dyn seismic_compiler::executable::MemoryAccount>,
    pub host: Arc<dyn seismic_compiler::executable::MemoryAccount>,
}

#[derive(Clone)]
pub struct Device {
    context: Arc<Context>,
    accounts: MemoryAccounts,
    stream: Arc<Stream>,
    /// The fill stream ([`NativeStream::Fill`]).
    fill: Arc<Stream>,
    blocks: Arc<Blocks>,
    graphs: Arc<crate::graph_cost::GraphPool>,
    /// The word a graph's word nodes name as it is recorded, before a
    /// replay binds its run's own ([`Self::placeholder_word`]).
    placeholder: Arc<crate::direct::HostWord>,
    stack_growth: Arc<crate::driver::StackGrowth>,
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
    /// Device `ordinal`, its own allocations charged to `accounts`.
    pub fn open(ordinal: u32, accounts: MemoryAccounts) -> Result<Self, ExecutionError> {
        let driver = Driver::load().map_err(ExecutionError::SubmissionFailed)?;
        let ordinal = i32::try_from(ordinal).map_err(|_| {
            ExecutionError::SubmissionFailed("CUDA device ordinal exceeds driver ABI".into())
        })?;
        let context = Context::retain(driver, ordinal).map_err(submission_error)?;
        let stream = Stream::new(&context).map_err(submission_error)?;
        let fill = Stream::new(&context).map_err(submission_error)?;
        let blocks = Arc::new(Blocks::new(context.clone(), accounts.device.clone()).map_err(submission_error)?);
        if let Some(memory) = counted_driver_memory(context.ordinal() as u32)? {
            assert!(
                context.driver_memory.set(memory).is_ok(),
                "a context's driver memory is counted once, at its opening"
            );
        }
        let placeholder = Arc::new(crate::direct::HostWord::on(&context, &accounts.upload)?);
        let graphs = Arc::new(crate::graph_cost::GraphPool::new(accounts.device.clone(), context.clone()));
        context.charge_stack_to(accounts.device.clone());
        Ok(Self {
            context,
            accounts,
            stream,
            fill,
            blocks,
            graphs,
            placeholder,
            stack_growth: crate::driver::stack_growth(ordinal),
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

    /// Device storage of `bytes` as a range of a block. `room`, when given,
    /// is the device memory this allocation may add, before the free space
    /// the device's blocks already hold is taken off it.
    pub fn allocate_within(
        &self,
        bytes: u64,
        alignment: u64,
        room: Option<u64>,
    ) -> Result<Buffer, ExecutionError> {
        if alignment > 256 || !alignment.is_power_of_two() {
            panic!("prepared CUDA allocation asks for unsupported alignment")
        }
        let room = room.map(|room| room.saturating_sub(self.blocks.unallocated_bytes()));
        self.blocks
            .allocate(bytes, alignment, room)
            .map(Buffer::device)
    }

    /// Device memory held beyond the live ranges: the blocks' free space,
    /// the instantiated graphs' driver memory and the stack the driver
    /// reserved beyond what every context reserves. Held by this process,
    /// charged to no allocation.
    pub fn held_beyond_ranges(&self) -> u64 {
        self.blocks.unallocated_bytes() + self.graphs.bytes() + self.stack_growth.bytes()
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

    /// The accounts the device was opened with.
    pub(crate) fn accounts(&self) -> &MemoryAccounts {
        &self.accounts
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

    /// Storage of `kind` (`crate::topology`) that is allocated whole: device
    /// memory or pinned host memory. `room` as for [`Self::allocate_within`].
    pub fn allocate_kind(&self, kind: u32, bytes: u64, alignment: u64, room: Option<u64>) -> Result<Buffer, ExecutionError> {
        match kind {
            crate::topology::DEVICE_KIND => self.allocate_within(bytes, alignment, room),
            crate::topology::PINNED_KIND => self.allocate_mapped(bytes),
            kind => Err(ExecutionError::AllocationFailed(format!("CUDA allocates no whole storage of kind {kind}"))),
        }
    }

    /// Pinned host memory mapped into the device address space, for inputs
    /// the host fills before each submission: the host writes it in place,
    /// without a driver copy that would wait for queued device work.
    pub fn allocate_mapped(&self, bytes: u64) -> Result<Buffer, ExecutionError> {
        let bytes = usize::try_from(bytes).map_err(|_| {
            ExecutionError::AllocationFailed("CUDA allocation exceeds host address space".into())
        })?;
        HostMapped::new(&self.context, &self.accounts.upload, bytes).map(Buffer::mapped)
    }

    /// `memory` over a reserved address range of `reserved` bytes in
    /// `granule`-byte granules ([`crate::reservation_granule`]), charged to
    /// `account`, with the leading `committed` bytes backed and zeroed, and
    /// its resizer ([`Resizer::prepare`]), which changes the backed prefix
    /// without moving it.
    pub fn allocate_reserved(
        &self,
        committed: u64,
        reserved: u64,
        granule: u64,
        memory: RangeMemory,
        account: Arc<dyn seismic_compiler::executable::MemoryAccount>,
    ) -> Result<(Buffer, CudaResizer), ExecutionError> {
        let host = |bytes: u64| {
            usize::try_from(bytes).map_err(|_| {
                ExecutionError::AllocationFailed(
                    "CUDA reservation exceeds host address space".into(),
                )
            })
        };
        let (reserved, granule) = (host(reserved)?, host(granule)?);
        let memory = ReservedMemory::new(&self.context, reserved, granule, memory, account)
            .map_err(allocation_error(reserved as u64, || device_free(&self.context)))?;
        let (reservation, resizer) = Reserved::new(memory, committed)?;
        Ok((Buffer::reserved(reservation), resizer))
    }
}

/// The right to resize one reserved buffer ([`Device::allocate_reserved`]).
pub type CudaResizer = Resizer<ReservedMemory>;

/// Granules mapped in one address range, in place: nothing device work binds
/// changes with them.
impl Backing for ReservedMemory {
    type Error = ExecutionError;
    type View = ();
    type Change = ();
    type Retired = ();

    fn granule(&self) -> usize {
        self.granularity
    }

    fn granules(&self) -> usize {
        self.reserved / self.granularity
    }

    /// Map granules `[from, to)`, zero: those their backing did not give
    /// zeroed (created granules, or registered ones it kept) are zeroed.
    fn back(&self, from: usize, to: usize) -> Result<(), ExecutionError> {
        let (start, end) = (from * self.granularity, to * self.granularity);
        let zeroed = self.commit(end)?;
        let stale = start as u64..end.min(zeroed) as u64;
        if stale.is_empty() {
            return Ok(());
        }
        // Nothing is published: the new granules are retracted.
        time_backing(BackingPhase::Zero, || self.zero(stale)).inspect_err(|_| self.retract(from, to))
    }

    /// Unmap the granules from `from`, the tail. Driver failures after the
    /// first unmap cannot leave the backing valid, so `commit` treats them
    /// as fatal; an unmap is refused nothing else.
    fn unback(&self, from: usize, to: usize) {
        if from < to {
            self.commit(from * self.granularity).expect("a CUDA reservation unmaps what it backs");
        }
    }

    /// Registered host granules stay registered and charged, uncommitted
    /// (their unregistration waits for the device, beside the growth's
    /// flights); created granules are unmapped.
    fn retract(&self, from: usize, to: usize) {
        if from >= to {
            return;
        }
        match self.registered() {
            Some(registered) => registered.uncommit(from),
            None => self.unback(from, to),
        }
    }

    /// Zero `bytes`, which no queued device work reaches: host memory in
    /// place (the driver refuses a memset across its separately created
    /// granules: `cuMemsetD8Async` returned `CUDA_ERROR_INVALID_VALUE` on an
    /// RTX 3090), device memory on the reservation's own stream, waited
    /// for: behind none of the device's queued work, which a resize prepared
    /// while decode runs would otherwise wait out.
    fn zero(&self, bytes: std::ops::Range<u64>) -> Result<(), ExecutionError> {
        let (pointer, length) = (self.base + bytes.start, (bytes.end - bytes.start) as usize);
        if self.memory.host_addressed() {
            // SAFETY: backed bytes of the reservation, which the host maps
            // and no device work reaches.
            unsafe { std::ptr::write_bytes(pointer as *mut u8, 0, length) };
            return Ok(());
        }
        let context = self.context();
        {
            let _current = context.enter().map_err(submission_error)?;
            let status = unsafe { context.enqueue().memset_d8_async(pointer, 0, length, self.stream.raw()) };
            context.driver.check(status, "reservation zero fill").map_err(submission_error)?;
        }
        self.stream.synchronize().map_err(submission_error)
    }

    fn view(&self, _: usize) -> Result<(), ExecutionError> {
        Ok(())
    }

    fn change(&self, _: usize, _: usize) -> Result<(), ExecutionError> {
        Ok(())
    }

    fn apply(&self, _: &mut (), _: ()) {}
}

impl DeviceService<Cuda> for Device {
    type Buffer = Buffer;
    /// CUDA maps no read-only host memory: every buffer is one it allocated.
    type Writable = Buffer;
    fn allocate(&self, bytes: u64, alignment: u64) -> Result<Buffer, ExecutionError> {
        Device::allocate_within(self, bytes, alignment, None)
    }
    fn allocate_within(
        &self,
        bytes: u64,
        alignment: u64,
        room: Option<u64>,
    ) -> Result<Buffer, ExecutionError> {
        Device::allocate_within(self, bytes, alignment, room)
    }
    fn held_beyond_allocations(&self) -> u64 {
        self.held_beyond_ranges()
    }
    fn unallocated(&self) -> u64 {
        self.blocks.unallocated_bytes()
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

    fn submit(mut self) -> Self::Execution {
        Execution {
            device: self.device.clone(),
            in_flight_uploads: std::mem::take(&mut self.in_flight_uploads),
        }
    }
}

impl NativeExecution for Execution {
    fn complete(&mut self) -> Result<(), ExecutionError> {
        synchronize(&self.device, &mut self.in_flight_uploads)
    }
}

impl Submission {
    /// One fill of `bytes` bytes at device address `pointer`, within one of
    /// its buffer's copy spans ([`Submission::fill`]).
    fn fill_piece(
        &self,
        issue: &crate::enqueue::Enqueue,
        pointer: u64,
        value: FillValue,
        bytes: u64,
    ) -> Result<(), ExecutionError> {
        let count = usize::try_from(bytes / value.width())
            .unwrap_or_else(|_| panic!("prepared CUDA fill exceeds usize"));
        let driver = &self.device.context.driver;
        unsafe {
            match value {
                FillValue::U8(pattern) => driver.check(
                    issue.memset_d8_async(pointer, pattern[0], count, self.device.stream.raw()),
                    "device fill",
                ),
                FillValue::U16(pattern) => driver.check(
                    issue.memset_d16_async(
                        pointer,
                        u16::from_le_bytes(pattern),
                        count,
                        self.device.stream.raw(),
                    ),
                    "device fill",
                ),
                FillValue::U32(pattern) => driver.check(
                    issue.memset_d32_async(
                        pointer,
                        u32::from_le_bytes(pattern),
                        count,
                        self.device.stream.raw(),
                    ),
                    "device fill",
                ),
                // Each 32-bit half of every element: one strided pass per half,
                // or a single pass when the halves agree.
                FillValue::U64(pattern) => {
                    let low = u32::from_le_bytes([pattern[0], pattern[1], pattern[2], pattern[3]]);
                    let high = u32::from_le_bytes([pattern[4], pattern[5], pattern[6], pattern[7]]);
                    if low == high {
                        driver.check(
                            issue.memset_d32_async(
                                pointer,
                                low,
                                count * 2,
                                self.device.stream.raw(),
                            ),
                            "device fill",
                        )
                    } else {
                        [(0, low), (4, high)].into_iter().try_for_each(|(offset, half)| {
                            driver.check(
                                issue.memset_d2d32_async(
                                    pointer + offset,
                                    8,
                                    half,
                                    1,
                                    count,
                                    self.device.stream.raw(),
                                ),
                                "device fill",
                            )
                        })
                    }
                }
            }
            .map_err(submission_error)
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
        let mut buffer_words = Vec::with_capacity(bindings.len());
        let mut words = vec![0u64; kernel.layout.words.total as usize];
        for (position, binding) in bindings.iter().enumerate() {
            let resolved = env.resolve_view(binding)?;
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
            words[kernel.layout.words.nat_first as usize + index] = evaluate(value, env.values());
        }
        for (index, symbol) in scalar_args.iter().copied().enumerate() {
            words[kernel.layout.words.scalar_first as usize + index] = encode_symbol(
                env.values()
                    .get(symbol)
                    .unwrap_or_else(|| panic!("prepared CUDA scalar argument is unbound")),
            )?;
        }
        let workgroup_bytes = evaluate(&totals.workgroup_bytes, env.values());
        let participant_bytes = evaluate(&totals.participant_bytes, env.values());
        let register_bytes = evaluate(&totals.register_bytes, env.values());
        for (position, local) in locals.iter().enumerate() {
            let layout = kernel.layout.words.locals[position];
            let offset = evaluate(&local.byte_offset, env.values());
            words[layout.first as usize] = offset;
            for (axis, value) in local.extents.iter().enumerate() {
                words[layout.first as usize + 1 + axis] = evaluate(value, env.values());
            }
            for (axis, value) in local.strides.iter().enumerate() {
                words[layout.first as usize + 1 + layout.rank as usize + axis] =
                    evaluate(value, env.values());
            }
        }
        for (position, resource) in addressable_resources.iter().enumerate() {
            let layout = kernel.layout.words.addressable_resources[position];
            words[layout.offset_units as usize] = evaluate(&resource.offset_units, env.values());
            words[layout.units as usize] = evaluate(&resource.units, env.values());
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
        let mut staging = PinnedUpload::new(&self.device.context, &self.device.accounts.host, total)?;
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
        env: &ExecutionEnvironment<'_, Cuda, CompiledKernel, Device>,
    ) -> Result<(), ExecutionError> {
        let source = env.resolve_view(source)?;
        let destination = env.resolve_view(destination)?;
        let _current = self.device.context.enter().map_err(submission_error)?;
        let issue = self.device.context.enqueue();
        // No copy crosses a host reservation's granule (`copy_pieces`).
        for (from, to, piece) in crate::buffer::copy_pieces(
            source.buffer,
            source.byte_offset,
            destination.buffer,
            destination.byte_offset,
            bytes,
        ) {
            let count =
                usize::try_from(piece).unwrap_or_else(|_| panic!("prepared CUDA copy exceeds usize"));
            unsafe {
                self.device
                    .context
                    .driver
                    .check(
                        issue.memcpy_device_async(
                            destination.buffer.pointer() + to,
                            source.buffer.pointer() + from,
                            count,
                            self.device.stream.raw(),
                        ),
                        "device copy",
                    )
                    .map_err(submission_error)?;
            }
        }
        Ok(())
    }
    fn fill(
        &self,
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
        // No fill crosses a host reservation's granule (`fill_pieces`); a
        // granule is whole elements, so each piece is.
        for (offset, piece) in crate::buffer::fill_pieces(destination.buffer, destination.byte_offset, bytes) {
            self.fill_piece(issue, destination.buffer.pointer() + offset, value, piece)?;
        }
        Ok(())
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
    let status = unsafe {
        context.enqueue().launch(
            mode == CudaLaunchMode::CooperativeGrid,
            kernel.function,
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
fn evaluate(value: &seismic_lang::expr::compiled::CompiledNat, values: &InvocationValues) -> u64 {
    value
        .evaluate_u64(values)
        .unwrap_or_else(|error| panic!("prepared CUDA expression failed evaluation: {error:?}"))
}
fn encode_symbol(value: SymbolValue) -> Result<u64, ExecutionError> {
    value.try_word64().map_err(|error| {
        ExecutionError::ConstructionContradiction(format!(
            "native scalar ABI quantity does not fit its planned word: {error:?}"
        ))
    })
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
