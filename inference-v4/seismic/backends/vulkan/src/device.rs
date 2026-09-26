//! One opened Vulkan device (§8.1–§8.5): the logical device with exactly
//! the floor's features plus the gated ones it has, one queue (compute-only
//! when the device has such a family), a timeline semaphore that numbers
//! every submission, the block allocator, and host access to storage.

use crate::facts::{self, Description, Facts};
use crate::instance::{Instance, LoaderError};
use crate::memory::{Buffer, Memory, Range, StorageKind};
use ash::vk;
use seismic_compiler::errors::ExecutionError;
use std::ffi::CStr;
use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard};

/// Map a Vulkan call's failure: device loss is its own error.
pub(crate) fn call<T>(result: Result<T, vk::Result>, what: &str) -> Result<T, ExecutionError> {
    result.map_err(|error| match error {
        vk::Result::ERROR_DEVICE_LOST => ExecutionError::DeviceLost(format!("{what}: {error}")),
        other => ExecutionError::SubmissionFailed(format!("{what} failed: {other}")),
    })
}

/// Why a device could not be opened.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OpenError {
    Loader(LoaderError),
    /// No physical device has the requested UUID any more.
    Missing,
    /// The device does not meet the floor.
    Floor(String),
    /// Creating the device or one of its objects, or running its fma probe,
    /// failed.
    Creation(String),
    /// Without `VK_KHR_shader_fma`, GLSL `fma` beside a `NoContraction`
    /// `a * b + c` of the same operands is not one fused operation (or the
    /// expression is contracted), so `seismic_fma_rn` cannot be bound: the
    /// device fails the floor (§16.1). The bits of both witness results.
    MultiplyAdd {
        fma: u32,
        separate: u32,
    },
    /// Where modules do not declare `RoundingModeRTE 32`, the device's
    /// default fp32 rounding of `operation` is not round-to-nearest-even on
    /// the witness `operands` (a, b, integer): the device fails the floor
    /// (§16.1). The expected and actual result bits.
    Rounding {
        operation: Rounded,
        operands: [u32; 3],
        expected: u32,
        actual: u32,
    },
}

/// An fp32 operation whose rounding the open-time probe checks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rounded {
    Add,
    Multiply,
    SignedConversion,
    UnsignedConversion,
}

impl fmt::Display for OpenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Loader(error) => write!(f, "{error}"),
            Self::Missing => f.write_str("the Vulkan device is no longer present"),
            Self::Floor(reason) => write!(f, "the Vulkan device does not meet the floor: {reason}"),
            Self::Creation(reason) => write!(f, "opening the Vulkan device failed: {reason}"),
            Self::MultiplyAdd { fma, separate } => write!(
                f,
                "the Vulkan device does not meet the floor: it lacks VK_KHR_shader_fma and its fma is not fused \
                 beside a NoContraction multiply-add (witness fma {fma:#010x}, multiply-add {separate:#010x}; \
                 fused 0x33800000, rounded twice 0x00000000)"
            ),
            Self::Rounding { operation, operands, expected, actual } => write!(
                f,
                "the Vulkan device does not meet the floor: without RoundingModeRTE 32 its fp32 {operation:?} of \
                 witness {operands:#010x?} gives {actual:#010x}, not the nearest-even {expected:#010x}"
            ),
        }
    }
}

impl std::error::Error for OpenError {}

/// Staging of transfers to and from unmapped device storage.
struct Staging {
    pool: vk::CommandPool,
    command: vk::CommandBuffer,
    range: Range,
}

/// Bytes of the staging range; larger transfers go in pieces.
const STAGING_BYTES: u64 = 16 << 20;

/// Resources of one submission (or of a dropped graph), reusable once the
/// timeline reaches `value`.
pub(crate) struct Retired {
    pub(crate) value: u64,
    pub(crate) command: RetiredCommand,
    pub(crate) queries: Option<(vk::QueryPool, u32)>,
    pub(crate) arguments: Vec<Range>,
}

pub(crate) enum RetiredCommand {
    /// A batch's primary buffer, pooled for reuse.
    Primary(vk::CommandBuffer),
    /// A graph's secondary buffer, freed.
    Secondary(vk::CommandBuffer),
}

/// Submission-side state recorded under one lock: command buffers, query
/// pools and argument chunks of finished submissions awaiting reuse.
pub(crate) struct Recording {
    pub(crate) pool: vk::CommandPool,
    free_commands: Vec<vk::CommandBuffer>,
    free_queries: Vec<(vk::QueryPool, u32)>,
    free_chunks: Vec<Range>,
    retired: Vec<Retired>,
}

impl Recording {
    /// Return an unused primary command buffer.
    pub(crate) fn free_command(&mut self, command: vk::CommandBuffer) {
        self.free_commands.push(command);
    }
}

/// Bytes of one argument chunk.
pub(crate) const ARGUMENT_CHUNK_BYTES: u64 = 256 << 10;

pub(crate) struct Queue {
    raw: vk::Queue,
    next: u64,
}

pub(crate) struct Inner {
    pub(crate) device: ash::Device,
    pub(crate) facts: Facts,
    pub(crate) memory: Memory,
    pub(crate) layout: vk::PipelineLayout,
    pub(crate) cache: vk::PipelineCache,
    pub(crate) timeline: vk::Semaphore,
    queue: Mutex<Queue>,
    recording: Mutex<Recording>,
    staging: Mutex<Option<Staging>>,
    physical: vk::PhysicalDevice,
    instance: &'static Instance,
}

/// An opened Vulkan device.
#[derive(Clone)]
pub struct Device {
    pub(crate) inner: Arc<Inner>,
}

/// Memory-budget sample of the device-local heap (`VK_EXT_memory_budget`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryBudget {
    pub heap_budget_bytes: u64,
    pub heap_usage_bytes: u64,
}

impl Device {
    /// Open the physical device with this UUID, revalidating it against
    /// the floor; without `RoundingModeRTE 32`, its default fp32 rounding
    /// must pass the rounding probe, and without `VK_KHR_shader_fma`, its
    /// `fma` must pass the multiply-add probe.
    ///
    /// A physical device has one logical device for the process, created by
    /// the first open and shared by every later one, like CUDA's primary
    /// context: NVIDIA 580 deadlocks when one logical device is destroyed
    /// while another compiles pipelines, which independent opens in one
    /// process would do.
    pub fn open(uuid: [u8; 16]) -> Result<Self, OpenError> {
        static OPENED: std::sync::OnceLock<Mutex<Vec<Device>>> = std::sync::OnceLock::new();
        let mut opened = OPENED
            .get_or_init(|| Mutex::new(Vec::new()))
            .lock()
            .expect("the opened-device registry lock is never poisoned");
        if let Some(device) = opened.iter().find(|device| device.facts().uuid == uuid) {
            return Ok(device.clone());
        }
        let instance = crate::instance::instance().map_err(OpenError::Loader)?;
        let (physical, Description { facts, floor }) = instance
            .physical_devices()
            .map_err(OpenError::Loader)?
            .into_iter()
            .map(|physical| (physical, facts::describe(instance, physical)))
            .find(|(_, description)| description.facts.uuid == uuid)
            .ok_or(OpenError::Missing)?;
        floor.map_err(OpenError::Floor)?;
        let device = Self {
            inner: Arc::new(Inner::create(instance, physical, facts)?),
        };
        if !device.facts().rounding_rte_32 {
            crate::probe::rounding(&device)?;
        }
        if !device.facts().shader_fma.float32 {
            crate::probe::multiply_add(&device)?;
        }
        opened.push(device.clone());
        Ok(device)
    }

    pub fn facts(&self) -> &Facts {
        &self.inner.facts
    }

    /// Device-local storage.
    pub fn allocate(&self, bytes: u64, alignment: u64) -> Result<Buffer, ExecutionError> {
        self.allocate_kind(StorageKind::Device, bytes, alignment)
    }

    /// Storage the host writes before each submission with plain memory
    /// writes that never wait for device work.
    pub fn allocate_upload(&self, bytes: u64, alignment: u64) -> Result<Buffer, ExecutionError> {
        self.allocate_kind(StorageKind::Upload, bytes, alignment)
    }

    fn allocate_kind(
        &self,
        kind: StorageKind,
        bytes: u64,
        alignment: u64,
    ) -> Result<Buffer, ExecutionError> {
        let range = self
            .inner
            .memory
            .allocate(&self.inner.device, kind, bytes, alignment)?;
        Ok(Buffer::new(self.inner.clone(), range))
    }

    /// Host write: through the mapping when the storage is host-visible,
    /// else through staging and a transfer on the queue, waited on. The
    /// caller orders it after device work using the storage.
    pub fn write(&self, buffer: &Buffer, offset: u64, bytes: &[u8]) -> Result<(), ExecutionError> {
        let range = buffer.range();
        if range.mapped().is_some() {
            range.write_mapped(offset, bytes);
            return Ok(());
        }
        let mut done = 0usize;
        while done < bytes.len() {
            let length = (bytes.len() - done).min(STAGING_BYTES as usize);
            let _staging = self.transfer(
                |staging| {
                    staging.write_mapped(0, &bytes[done..done + length]);
                    vk::BufferCopy {
                        src_offset: staging.offset(),
                        dst_offset: range.offset() + offset + done as u64,
                        size: length as u64,
                    }
                },
                range,
                true,
            )?;
            done += length;
        }
        Ok(())
    }

    /// Host read; see [`Device::write`].
    pub fn read(
        &self,
        buffer: &Buffer,
        offset: u64,
        into: &mut [u8],
    ) -> Result<(), ExecutionError> {
        let range = buffer.range();
        if range.mapped().is_some() {
            range.read_mapped(offset, into);
            return Ok(());
        }
        let mut done = 0usize;
        while done < into.len() {
            let length = (into.len() - done).min(STAGING_BYTES as usize);
            let staging = self.transfer(
                |staging| vk::BufferCopy {
                    src_offset: range.offset() + offset + done as u64,
                    dst_offset: staging.offset(),
                    size: length as u64,
                },
                range,
                false,
            )?;
            staging
                .as_ref()
                .expect("a transfer leaves its staging in place")
                .range
                .read_mapped(0, &mut into[done..done + length]);
            done += length;
        }
        Ok(())
    }

    /// One staged copy between `range` and the staging range, submitted and
    /// waited on. `upload` copies staging to `range`, else `range` to
    /// staging. Returns the staging lock so a read can copy out.
    fn transfer(
        &self,
        region: impl FnOnce(&Range) -> vk::BufferCopy,
        range: &Range,
        upload: bool,
    ) -> Result<MutexGuard<'_, Option<Staging>>, ExecutionError> {
        let inner = &self.inner;
        let device = &inner.device;
        let mut guard = inner
            .staging
            .lock()
            .expect("staging lock is never poisoned");
        if guard.is_none() {
            *guard = Some(inner.create_staging()?);
        }
        let staging = guard.as_mut().expect("staging was created above");
        let copy = region(&staging.range);
        let (source, destination) = if upload {
            (staging.range.raw(), range.raw())
        } else {
            (range.raw(), staging.range.raw())
        };
        let command = staging.command;
        unsafe {
            call(
                device.reset_command_buffer(command, vk::CommandBufferResetFlags::empty()),
                "vkResetCommandBuffer",
            )?;
            call(
                device.begin_command_buffer(
                    command,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                ),
                "vkBeginCommandBuffer",
            )?;
            barrier(
                device,
                command,
                vk::PipelineStageFlags2::ALL_COMMANDS,
                vk::AccessFlags2::MEMORY_WRITE,
                vk::PipelineStageFlags2::TRANSFER,
                vk::AccessFlags2::TRANSFER_READ | vk::AccessFlags2::TRANSFER_WRITE,
            );
            device.cmd_copy_buffer(command, source, destination, &[copy]);
            barrier(
                device,
                command,
                vk::PipelineStageFlags2::TRANSFER,
                vk::AccessFlags2::TRANSFER_WRITE,
                vk::PipelineStageFlags2::HOST | vk::PipelineStageFlags2::ALL_COMMANDS,
                vk::AccessFlags2::HOST_READ
                    | vk::AccessFlags2::MEMORY_READ
                    | vk::AccessFlags2::MEMORY_WRITE,
            );
            call(device.end_command_buffer(command), "vkEndCommandBuffer")?;
        }
        let value = self.submit(command)?;
        self.wait(value)?;
        Ok(guard)
    }

    /// Submit `command`, signalling the timeline with the next value.
    pub(crate) fn submit(&self, command: vk::CommandBuffer) -> Result<u64, ExecutionError> {
        let inner = &self.inner;
        let mut queue = inner.queue.lock().expect("queue lock is never poisoned");
        let value = queue.next;
        let signal = [vk::SemaphoreSubmitInfo::default()
            .semaphore(inner.timeline)
            .value(value)
            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
        let commands = [vk::CommandBufferSubmitInfo::default().command_buffer(command)];
        let submit = vk::SubmitInfo2::default()
            .command_buffer_infos(&commands)
            .signal_semaphore_infos(&signal);
        call(
            unsafe {
                inner
                    .device
                    .queue_submit2(queue.raw, &[submit], vk::Fence::null())
            },
            "vkQueueSubmit2",
        )?;
        queue.next += 1;
        Ok(value)
    }

    /// The timeline value every completed submission has reached. A failed
    /// query means the work cannot make progress: it reads as complete.
    pub(crate) fn completed(&self) -> u64 {
        unsafe {
            self.inner
                .device
                .get_semaphore_counter_value(self.inner.timeline)
        }
        .unwrap_or(u64::MAX)
    }

    pub(crate) fn wait(&self, value: u64) -> Result<(), ExecutionError> {
        let semaphores = [self.inner.timeline];
        let values = [value];
        let info = vk::SemaphoreWaitInfo::default()
            .semaphores(&semaphores)
            .values(&values);
        call(
            unsafe { self.inner.device.wait_semaphores(&info, u64::MAX) },
            "vkWaitSemaphores",
        )
    }

    pub(crate) fn recording(&self) -> MutexGuard<'_, Recording> {
        let mut recording = self
            .inner
            .recording
            .lock()
            .expect("recording lock is never poisoned");
        let completed = self.completed();
        let mut index = 0;
        while index < recording.retired.len() {
            if recording.retired[index].value <= completed {
                let retired = recording.retired.swap_remove(index);
                match retired.command {
                    RetiredCommand::Primary(command) => recording.free_commands.push(command),
                    RetiredCommand::Secondary(command) => unsafe {
                        self.inner
                            .device
                            .free_command_buffers(recording.pool, &[command])
                    },
                }
                recording.free_queries.extend(retired.queries);
                recording.free_chunks.extend(retired.arguments);
            } else {
                index += 1;
            }
        }
        recording
    }

    /// A primary command buffer, reset and ready to begin.
    pub(crate) fn command(
        &self,
        recording: &mut Recording,
        level: vk::CommandBufferLevel,
    ) -> Result<vk::CommandBuffer, ExecutionError> {
        if level == vk::CommandBufferLevel::PRIMARY {
            if let Some(command) = recording.free_commands.pop() {
                call(
                    unsafe {
                        self.inner
                            .device
                            .reset_command_buffer(command, vk::CommandBufferResetFlags::empty())
                    },
                    "vkResetCommandBuffer",
                )?;
                return Ok(command);
            }
        }
        let info = vk::CommandBufferAllocateInfo::default()
            .command_pool(recording.pool)
            .level(level)
            .command_buffer_count(1);
        call(
            unsafe { self.inner.device.allocate_command_buffers(&info) },
            "vkAllocateCommandBuffers",
        )
        .map(|commands| commands[0])
    }

    /// A timestamp query pool of at least `queries` queries.
    pub(crate) fn queries(
        &self,
        recording: &mut Recording,
        queries: u32,
    ) -> Result<(vk::QueryPool, u32), ExecutionError> {
        if let Some(index) = recording
            .free_queries
            .iter()
            .position(|(_, capacity)| *capacity >= queries)
        {
            return Ok(recording.free_queries.swap_remove(index));
        }
        let info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(queries);
        call(
            unsafe { self.inner.device.create_query_pool(&info, None) },
            "vkCreateQueryPool",
        )
        .map(|pool| (pool, queries))
    }

    /// An argument chunk in upload storage.
    pub(crate) fn argument_chunk(
        &self,
        recording: &mut Recording,
    ) -> Result<Range, ExecutionError> {
        match recording.free_chunks.pop() {
            Some(chunk) => Ok(chunk),
            None => self.inner.memory.allocate(
                &self.inner.device,
                StorageKind::Upload,
                ARGUMENT_CHUNK_BYTES,
                256,
            ),
        }
    }

    pub(crate) fn retire(&self, retired: Retired) {
        self.inner
            .recording
            .lock()
            .expect("recording lock is never poisoned")
            .retired
            .push(retired);
    }

    /// `VK_EXT_memory_budget` of the device-local heap.
    pub fn memory_budget(&self) -> MemoryBudget {
        let inner = &self.inner;
        let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
        let mut properties = vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut budget);
        unsafe {
            inner
                .instance
                .raw()
                .get_physical_device_memory_properties2(inner.physical, &mut properties)
        };
        let memory = properties.memory_properties;
        let heap = (0..memory.memory_heap_count as usize)
            .filter(|index| {
                memory.memory_heaps[*index]
                    .flags
                    .contains(vk::MemoryHeapFlags::DEVICE_LOCAL)
            })
            .max_by_key(|index| memory.memory_heaps[*index].size)
            .expect("the floor requires a device-local heap");
        MemoryBudget {
            heap_budget_bytes: budget.heap_budget[heap],
            heap_usage_bytes: budget.heap_usage[heap],
        }
    }
}

/// One global memory barrier.
pub(crate) unsafe fn barrier(
    device: &ash::Device,
    command: vk::CommandBuffer,
    source_stage: vk::PipelineStageFlags2,
    source_access: vk::AccessFlags2,
    destination_stage: vk::PipelineStageFlags2,
    destination_access: vk::AccessFlags2,
) {
    let barriers = [vk::MemoryBarrier2::default()
        .src_stage_mask(source_stage)
        .src_access_mask(source_access)
        .dst_stage_mask(destination_stage)
        .dst_access_mask(destination_access)];
    device.cmd_pipeline_barrier2(
        command,
        &vk::DependencyInfo::default().memory_barriers(&barriers),
    );
}

fn creation<T>(result: Result<T, vk::Result>, what: &str) -> Result<T, OpenError> {
    result.map_err(|error| OpenError::Creation(format!("{what} failed: {error}")))
}

impl Inner {
    fn create(
        instance: &'static Instance,
        physical: vk::PhysicalDevice,
        facts: Facts,
    ) -> Result<Self, OpenError> {
        let raw = instance.raw();
        let mut extensions: Vec<&CStr> = facts::FLOOR_EXTENSIONS.to_vec();
        if facts.matrix {
            extensions.push(ash::khr::cooperative_matrix::NAME);
        }
        if facts.f32_atomic_add {
            extensions.push(ash::ext::shader_atomic_float::NAME);
        }
        if facts.shader_fma.float32 {
            extensions.push(facts::SHADER_FMA_EXTENSION);
        }
        let extension_names = extensions
            .iter()
            .map(|name| name.as_ptr())
            .collect::<Vec<_>>();
        let mut shader_fma = facts::ShaderFmaFeatures::default();
        shader_fma.float16 = facts.shader_fma.float16.into();
        shader_fma.float32 = facts.shader_fma.float32.into();
        shader_fma.float64 = facts.shader_fma.float64.into();

        let mut explicit = vk::PhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR::default()
            .workgroup_memory_explicit_layout(true)
            .workgroup_memory_explicit_layout_scalar_block_layout(true)
            .workgroup_memory_explicit_layout8_bit_access(true)
            .workgroup_memory_explicit_layout16_bit_access(true);
        let mut cooperative =
            vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default().cooperative_matrix(true);
        let mut atomic_float = vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT::default()
            .shader_buffer_float32_atomic_add(true);
        let mut f11 =
            vk::PhysicalDeviceVulkan11Features::default().storage_buffer16_bit_access(true);
        let mut f12 = vk::PhysicalDeviceVulkan12Features::default()
            .buffer_device_address(true)
            .shader_int8(true)
            .shader_float16(facts.float16)
            .storage_buffer8_bit_access(true)
            .scalar_block_layout(true)
            .timeline_semaphore(true)
            .vulkan_memory_model(true)
            .vulkan_memory_model_device_scope(true)
            .shader_subgroup_extended_types(true)
            .shader_shared_int64_atomics(facts.shared_int64_atomics);
        let mut f13 = vk::PhysicalDeviceVulkan13Features::default()
            .synchronization2(true)
            .maintenance4(true)
            .pipeline_creation_cache_control(true)
            .subgroup_size_control(true)
            .compute_full_subgroups(true)
            .shader_integer_dot_product(true);
        let mut features = vk::PhysicalDeviceFeatures2::default()
            .features(
                vk::PhysicalDeviceFeatures::default()
                    .shader_int64(true)
                    .shader_int16(true),
            )
            .push_next(&mut f11)
            .push_next(&mut f12)
            .push_next(&mut f13)
            .push_next(&mut explicit);
        if facts.matrix {
            features = features.push_next(&mut cooperative);
        }
        if facts.f32_atomic_add {
            features = features.push_next(&mut atomic_float);
        }
        if facts.shader_fma.float32 {
            features = features.push_next(&mut shader_fma);
        }
        let priorities = [1.0f32];
        let queues = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(facts.queue_family)
            .queue_priorities(&priorities)];
        let info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queues)
            .enabled_extension_names(&extension_names)
            .push_next(&mut features);
        let device = creation(
            unsafe { raw.create_device(physical, &info, None) },
            "vkCreateDevice",
        )?;
        // From here every created object is destroyed by `Drop` (null
        // handles are ignored), so partial construction cleans up.
        let queue = unsafe { device.get_device_queue(facts.queue_family, 0) };
        let memory_properties = unsafe { raw.get_physical_device_memory_properties(physical) };
        let memory = match Memory::new(&memory_properties, &facts) {
            Ok(memory) => memory,
            Err(reason) => {
                unsafe { device.destroy_device(None) };
                return Err(OpenError::Creation(reason.into()));
            }
        };
        let mut inner = Self {
            device,
            facts,
            memory,
            layout: vk::PipelineLayout::null(),
            cache: vk::PipelineCache::null(),
            timeline: vk::Semaphore::null(),
            queue: Mutex::new(Queue {
                raw: queue,
                next: 1,
            }),
            recording: Mutex::new(Recording {
                pool: vk::CommandPool::null(),
                free_commands: Vec::new(),
                free_queries: Vec::new(),
                free_chunks: Vec::new(),
                retired: Vec::new(),
            }),
            staging: Mutex::new(None),
            physical,
            instance,
        };
        let device = &inner.device;
        let ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(8)];
        inner.layout = creation(
            unsafe {
                device.create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&ranges),
                    None,
                )
            },
            "vkCreatePipelineLayout",
        )?;
        inner.cache = creation(
            unsafe { device.create_pipeline_cache(&vk::PipelineCacheCreateInfo::default(), None) },
            "vkCreatePipelineCache",
        )?;
        let mut timeline = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0);
        inner.timeline = creation(
            unsafe {
                device.create_semaphore(
                    &vk::SemaphoreCreateInfo::default().push_next(&mut timeline),
                    None,
                )
            },
            "vkCreateSemaphore",
        )?;
        let pool = creation(
            unsafe {
                device.create_command_pool(&command_pool_info(inner.facts.queue_family), None)
            },
            "vkCreateCommandPool",
        )?;
        inner
            .recording
            .get_mut()
            .expect("recording lock is never poisoned")
            .pool = pool;
        Ok(inner)
    }

    fn create_staging(&self) -> Result<Staging, ExecutionError> {
        let device = &self.device;
        let pool = call(
            unsafe {
                device.create_command_pool(&command_pool_info(self.facts.queue_family), None)
            },
            "vkCreateCommandPool",
        )?;
        let info = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let command = match call(
            unsafe { device.allocate_command_buffers(&info) },
            "vkAllocateCommandBuffers",
        ) {
            Ok(commands) => commands[0],
            Err(error) => {
                unsafe { device.destroy_command_pool(pool, None) };
                return Err(error);
            }
        };
        let range = match self
            .memory
            .allocate(device, StorageKind::Staging, STAGING_BYTES, 256)
        {
            Ok(range) => range,
            Err(error) => {
                unsafe { device.destroy_command_pool(pool, None) };
                return Err(error);
            }
        };
        Ok(Staging {
            pool,
            command,
            range,
        })
    }
}

fn command_pool_info<'a>(family: u32) -> vk::CommandPoolCreateInfo<'a> {
    vk::CommandPoolCreateInfo::default()
        .queue_family_index(family)
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
}

impl Drop for Inner {
    fn drop(&mut self) {
        let device = &self.device;
        unsafe {
            let _ = device.device_wait_idle();
            let recording = self
                .recording
                .get_mut()
                .expect("recording lock is never poisoned");
            for retired in recording.retired.drain(..) {
                recording.free_queries.extend(retired.queries);
            }
            for (pool, _) in recording.free_queries.drain(..) {
                device.destroy_query_pool(pool, None);
            }
            device.destroy_command_pool(recording.pool, None);
            if let Some(staging) = self
                .staging
                .get_mut()
                .expect("staging lock is never poisoned")
                .take()
            {
                device.destroy_command_pool(staging.pool, None);
            }
            self.memory.destroy(device);
            device.destroy_semaphore(self.timeline, None);
            device.destroy_pipeline_cache(self.cache, None);
            device.destroy_pipeline_layout(self.layout, None);
            device.destroy_device(None);
        }
    }
}
