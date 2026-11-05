//! Physical-device facts and the device floor.
//!
//! Discovery and opening read the same facts through [`describe`]: the floor
//! decides availability (the first missing item is the reason), and the gated
//! facts select prefix macros, the numerical environment and the formation and
//! tuning identities.

use crate::device::DeviceHandle;
use crate::instance::Instance;
use ash::vk;
use std::ffi::CStr;
use seismic_topology::Unreservable;
use std::sync::Arc;

/// Minimum group memory, invocations, grid and group extents and push
/// constants of the floor.
const FLOOR_SHARED_BYTES: u32 = 32 * 1024;
const FLOOR_INVOCATIONS: u32 = 1024;
const FLOOR_GROUP_COUNT: u32 = 65_535;
const FLOOR_GROUP_SIZE: [u32; 3] = [1024, 1024, 64];
const FLOOR_PUSH_CONSTANT_BYTES: u32 = 128;
/// The width of the logical subgroup every kernel is written for.
pub const SUBGROUP_WIDTH: u32 = 32;

/// The device facts formation reads: what the generated prefix
/// exposes to sources and what the seal pass declares. Two devices with
/// equal formation facts form every source identically.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FormationFacts {
    /// Lanes of one hardware subgroup: 32, or 64 holding two logical
    /// subgroups.
    pub subgroup_lanes: u32,
    pub float16: bool,
    pub matrix: bool,
    pub wide_accumulators: bool,
    pub mixed_dot: bool,
    pub f32_atomic_add: bool,
    pub shared_int64_atomics: bool,
    pub rounding_rte_32: bool,
    pub denorm_preserve_32: bool,
}

impl FormationFacts {
    /// Whether a device meeting the floor can have these facts: its subgroup
    /// has 32 or 64 lanes, and cooperative matrix needs fp16 arithmetic and
    /// 32-lane subgroups (see `describe`).
    pub fn consistent(&self) -> bool {
        matches!(self.subgroup_lanes, 32 | 64)
            && (!self.matrix || (self.float16 && self.subgroup_lanes == SUBGROUP_WIDTH))
    }

    /// The environment the seal pass declares.
    pub fn environment(&self) -> crate::seal::Environment {
        crate::seal::Environment {
            float16: self.float16,
            rounding_rte_32: self.rounding_rte_32,
            denorm_preserve_32: self.denorm_preserve_32,
        }
    }

    /// The consistent facts that differ from these in exactly one field,
    /// each field's other values in turn.
    pub fn variations(&self) -> Vec<Self> {
        let Self {
            subgroup_lanes,
            float16,
            matrix,
            wide_accumulators,
            mixed_dot,
            f32_atomic_add,
            shared_int64_atomics,
            rounding_rte_32,
            denorm_preserve_32,
        } = *self;
        let other_lanes = if subgroup_lanes == SUBGROUP_WIDTH {
            2 * SUBGROUP_WIDTH
        } else {
            SUBGROUP_WIDTH
        };
        [
            Self {
                subgroup_lanes: other_lanes,
                ..*self
            },
            Self {
                float16: !float16,
                ..*self
            },
            Self {
                matrix: !matrix,
                ..*self
            },
            Self {
                wide_accumulators: !wide_accumulators,
                ..*self
            },
            Self {
                mixed_dot: !mixed_dot,
                ..*self
            },
            Self {
                f32_atomic_add: !f32_atomic_add,
                ..*self
            },
            Self {
                shared_int64_atomics: !shared_int64_atomics,
                ..*self
            },
            Self {
                rounding_rte_32: !rounding_rte_32,
                ..*self
            },
            Self {
                denorm_preserve_32: !denorm_preserve_32,
                ..*self
            },
        ]
        .into_iter()
        .filter(Self::consistent)
        .collect()
    }
}

/// How a device runs the 32-lane logical subgroup.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SubgroupWidth {
    /// Pipelines require a subgroup size of 32.
    Required32,
    /// Every subgroup has 32 lanes; pipelines request no size (NVK).
    Fixed32,
    /// Every subgroup has 64 lanes and holds two logical subgroups (AMD
    /// GCN).
    Fixed64,
}

impl SubgroupWidth {
    /// Lanes of one hardware subgroup.
    pub fn lanes(self) -> u32 {
        match self {
            Self::Required32 | Self::Fixed32 => SUBGROUP_WIDTH,
            Self::Fixed64 => 2 * SUBGROUP_WIDTH,
        }
    }

    /// The width a device with subgroup sizes `min..=max` runs, where
    /// `requirable` says whether compute pipelines may require a size.
    fn of(min: u32, max: u32, requirable: bool) -> Option<Self> {
        if requirable && min <= SUBGROUP_WIDTH && SUBGROUP_WIDTH <= max {
            Some(Self::Required32)
        } else if (min, max) == (SUBGROUP_WIDTH, SUBGROUP_WIDTH) {
            Some(Self::Fixed32)
        } else if (min, max) == (2 * SUBGROUP_WIDTH, 2 * SUBGROUP_WIDTH) {
            Some(Self::Fixed64)
        } else {
            None
        }
    }
}

/// PCI vendor of AMD, whose Windows driver misreports cooperative matrix on
/// RDNA2.
const AMD_VENDOR: u32 = 0x1002;

/// `VK_KHR_shader_fma` (newer than ash 0.38).
pub(crate) const SHADER_FMA_EXTENSION: &CStr = c"VK_KHR_shader_fma";

/// `VkPhysicalDeviceShaderFmaFeaturesKHR` (newer than ash 0.38).
#[repr(C)]
pub(crate) struct ShaderFmaFeatures {
    s_type: vk::StructureType,
    p_next: *mut std::ffi::c_void,
    pub(crate) float16: vk::Bool32,
    pub(crate) float32: vk::Bool32,
    pub(crate) float64: vk::Bool32,
}

impl Default for ShaderFmaFeatures {
    fn default() -> Self {
        Self {
            s_type: vk::StructureType::from_raw(1_000_579_000),
            p_next: std::ptr::null_mut(),
            float16: vk::FALSE,
            float32: vk::FALSE,
            float64: vk::FALSE,
        }
    }
}

// SAFETY: a Vulkan structure with `sType` and `pNext` leading, extending
// both chains per the extension's registry entry.
unsafe impl vk::ExtendsPhysicalDeviceFeatures2 for ShaderFmaFeatures {}
unsafe impl vk::ExtendsDeviceCreateInfo for ShaderFmaFeatures {}

/// `VkPhysicalDeviceType`, as the backend reads it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceType {
    Discrete,
    Integrated,
    /// A software device (lavapipe).
    Cpu,
    /// A virtual GPU or a type the backend does not know; the floor
    /// refuses it.
    Other,
}

impl DeviceType {
    /// Whether the device's memory is the host's, as its type reports it:
    /// an integrated GPU's or a CPU device's is.
    pub fn memory_is_host(self) -> bool {
        matches!(self, Self::Integrated | Self::Cpu)
    }
}

/// Limits the runtime checks launches and allocations against.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Limits {
    pub max_shared_bytes: u64,
    pub max_invocations: u64,
    pub max_group_count: [u64; 3],
    pub max_group_size: [u64; 3],
    /// Largest single allocation: min(`maxMemoryAllocationSize`,
    /// `maxBufferSize`).
    pub max_allocation_bytes: u64,
    /// `maxMemoryAllocationCount`: the device memory objects that may exist
    /// at once.
    pub max_memory_allocations: u64,
    /// Nanoseconds per timestamp tick.
    pub timestamp_period: f64,
    /// Valid bits of the chosen queue family's timestamps.
    pub timestamp_valid_bits: u32,
}

/// The widths `OpFmaKHR` supports (`VK_KHR_shader_fma`); all false without
/// the extension.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ShaderFma {
    pub float16: bool,
    pub float32: bool,
    pub float64: bool,
}

/// How a device pairs its timestamps with the host clock.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Calibration {
    Khr,
    Ext,
}

/// Everything Seismic uses about one physical device.
#[derive(Clone, Debug)]
pub struct Facts {
    pub name: String,
    /// `VkPhysicalDeviceIDProperties::deviceUUID`.
    pub uuid: [u8; 16],
    pub vendor_id: u32,
    pub device_id: u32,
    pub device_type: DeviceType,
    /// `VkDriverId` (raw).
    pub driver_id: i32,
    pub driver_name: String,
    pub driver_info: String,
    pub driver_version: u32,
    pub api_version: u32,
    pub pipeline_cache_uuid: [u8; 16],
    pub limits: Limits,
    /// The queue family work is submitted to: compute-only when one exists.
    pub queue_family: u32,
    pub dedicated_compute_queue: bool,
    /// The queues of `queue_family`.
    pub queue_count: u32,
    /// The family of the queue beside the first: a transfer-only family
    /// where the device has one, whose copies run on a copy engine beside
    /// compute; else `queue_family`, where it has a second queue, whose
    /// copies take turns with compute on one engine.
    pub side_family: Option<u32>,
    /// `minImportedHostPointerAlignment`, where the device imports host
    /// memory (`VK_EXT_external_memory_host`).
    pub host_import_alignment: Option<u64>,
    /// `minPlacedMemoryMapAlignment`, where the device maps memory at a
    /// host address it is given and unmaps it back to a reservation
    /// (`VK_EXT_map_memory_placed`'s `memoryMapPlaced` and
    /// `memoryUnmapReserve`, over `VK_KHR_map_memory2`).
    pub placed_map_alignment: Option<u64>,
    /// How the device runs the logical subgroup; `None` fails the floor.
    pub subgroup_width: Option<SubgroupWidth>,
    /// fp16 arithmetic with RTE rounding and signed-zero/Inf/NaN
    /// preservation (`SEISMIC_HAS_FLOAT16`). Without it, kernels compute in
    /// fp32 over 16-bit storage.
    pub float16: bool,
    /// Subgroup-scope 16x16x16 cooperative matrix with f16xf16->f32
    /// (`SEISMIC_HAS_MATRIX`), never on RDNA2 or the NVIDIA proprietary
    /// driver (slower than the FMA path there). Requires fp16
    /// arithmetic and 32-lane hardware subgroups.
    pub matrix: bool,
    /// The compiler handles a subgroup's cooperative-matrix accumulator
    /// arrays beyond 4 fragments in the flash-attention body
    /// (`SEISMIC_HAS_WIDE_ACCUMULATORS`). False on the NVIDIA proprietary
    /// driver: 580.159.03 returns wrong outputs for 8- and 16-fragment
    /// output accumulators there, while 4 are reliable.
    pub wide_accumulators: bool,
    /// `DenormPreserve 32` is emitted when true.
    pub denorm_preserve_32: bool,
    /// `RoundingModeRTE 32` is emitted when true. False on the NVIDIA
    /// proprietary driver, whose compiler traps on fp32 `OpFDiv` under it
    /// (580.159.03); there fp32 rounds by the device's default.
    pub rounding_rte_32: bool,
    pub mixed_dot_accelerated: bool,
    /// `VK_KHR_shader_fma` with fp32 `OpFmaKHR` (fp16/fp64 as reported):
    /// `seismic_fma_rn` is bound to `OpFmaKHR`, so it is fused. Without it,
    /// `seismic_fma_rn` is GLSL `fma`, which the driver may fuse or round twice.
    pub shader_fma: ShaderFma,
    pub f32_atomic_add: bool,
    pub shared_int64_atomics: bool,
    pub calibration: Option<Calibration>,
    /// `VkPhysicalDeviceIDProperties::deviceLUID`, when `deviceLUIDValid`:
    /// the OS adapter's identity, in `LUID` memory layout.
    pub luid: Option<[u8; 8]>,
    /// The device's memory heaps and types.
    pub memory: MemoryLayout,
    /// The memory types the backend allocates each storage kind from.
    pub allocation: Allocation,
}

/// `VkPhysicalDeviceMemoryProperties`: the heaps and the memory types, in
/// the driver's order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryLayout {
    /// The heaps, indexed by heap number.
    pub heaps: Vec<MemoryHeap>,
    /// The memory types, indexed by memory type number.
    pub types: Vec<MemoryType>,
}

/// One `VkMemoryHeap`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryHeap {
    pub bytes: u64,
    /// `VK_MEMORY_HEAP_DEVICE_LOCAL_BIT`.
    pub device_local: bool,
}

/// One `VkMemoryType`: its heap and the property flags the backend reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryType {
    /// The heap the type allocates from (`heapIndex`).
    pub heap: u32,
    /// `VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT`.
    pub device_local: bool,
    /// `VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT`.
    pub host_visible: bool,
    /// `VK_MEMORY_PROPERTY_HOST_COHERENT_BIT`.
    pub host_coherent: bool,
    /// `VK_MEMORY_PROPERTY_HOST_CACHED_BIT`.
    pub host_cached: bool,
}

impl MemoryType {
    /// Host-visible and coherent: the host maps it and its writes need no
    /// flush.
    pub fn mappable(&self) -> bool {
        self.host_visible && self.host_coherent
    }
}

/// What the backend reads of a device's memory types for its own use:
/// where it lands host pages and its staging. Which kinds a buffer lies in is
/// the topology's placement rule, over the kinds it reports
/// (`crate::topology`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Allocation {
    /// The heaps of the device's own memory, whose usage its budget reads:
    /// its first device-local type's, or every heap with a type on a device
    /// whose memory is the host's, the device-local ones first.
    pub device_heaps: Vec<u32>,
    /// The types host pages are imported as, and staging's: every mappable
    /// type that is not device-local (any mappable type, on a device whose
    /// memory is the host's), host-cached ones first, so the host's reads
    /// are cached and its stores are ordinary write-back stores.
    pub host: Vec<u32>,
    /// The backend's transfer staging, which is host RAM's: the first of
    /// [`Self::host`]; none when there is none.
    pub staging: Option<u32>,
}

impl Allocation {
    /// What `layout` of a device whose memory is the host's or its own gives.
    pub fn of(layout: &MemoryLayout, memory_is_host: bool) -> Self {
        let types = || layout.types.iter().enumerate().map(|(index, kind)| (index as u32, kind));
        let mut host = types()
            .filter(|(_, kind)| kind.mappable() && (memory_is_host || !kind.device_local))
            .collect::<Vec<_>>();
        host.sort_by_key(|(_, kind)| !kind.host_cached);
        let host = host.into_iter().map(|(index, _)| index).collect::<Vec<_>>();
        let device_heaps = if memory_is_host {
            let mut heaps = (0..layout.heaps.len() as u32)
                .filter(|&heap| types().any(|(_, kind)| kind.heap == heap))
                .collect::<Vec<_>>();
            heaps.sort_by_key(|&heap| !layout.heaps[heap as usize].device_local);
            heaps
        } else {
            types().find(|(_, kind)| kind.device_local).map(|(_, kind)| kind.heap).into_iter().collect()
        };
        Self { device_heaps, staging: host.first().copied(), host }
    }
}

impl Facts {
    pub fn is_gpu(&self) -> bool {
        matches!(
            self.device_type,
            DeviceType::Discrete | DeviceType::Integrated
        )
    }

    /// The formation-identity part the device determines, including
    /// the sealed environment and how `seismic_fma_rn` is bound.
    pub fn formation_identity(&self) -> String {
        format!(
            "driver {} {};cache {};subgroup {:?};f16 {};matrix {};denorm {};rte32 {};fma {}",
            self.driver_id,
            self.driver_version,
            hex(&self.pipeline_cache_uuid),
            self.subgroup_width(),
            u8::from(self.float16),
            u8::from(self.matrix),
            if self.denorm_preserve_32 {
                "preserve"
            } else {
                "default"
            },
            if self.rounding_rte_32 {
                "declared"
            } else {
                "default"
            },
            if self.shader_fma.float32 {
                "khr"
            } else {
                "glsl"
            }
        )
    }

    /// How a device meeting the floor runs the logical subgroup.
    pub fn subgroup_width(&self) -> SubgroupWidth {
        self.subgroup_width
            .expect("a device meeting the floor has a logical subgroup width")
    }

    /// The environment the seal pass declares for this device.
    pub fn environment(&self) -> crate::seal::Environment {
        self.formation().environment()
    }

    /// The facts formation reads.
    pub fn formation(&self) -> FormationFacts {
        let facts = FormationFacts {
            subgroup_lanes: self.subgroup_width().lanes(),
            float16: self.float16,
            matrix: self.matrix,
            wide_accumulators: self.wide_accumulators,
            mixed_dot: self.mixed_dot_accelerated,
            f32_atomic_add: self.f32_atomic_add,
            shared_int64_atomics: self.shared_int64_atomics,
            rounding_rte_32: self.rounding_rte_32,
            denorm_preserve_32: self.denorm_preserve_32,
        };
        assert!(
            facts.consistent(),
            "device description produced inconsistent formation facts: {facts:?}"
        );
        facts
    }

    /// What backs a reservation in `kind` ([`ChunkBacking`]): imported host
    /// pages where the device has no placed mapping for its host-addressed
    /// ranges or its memory is its own; a device-local type's chunks on a
    /// device whose memory is its own; a mappable type's, mapped in place,
    /// on one whose memory is the host's and that maps at a placed address;
    /// else none.
    pub(crate) fn chunk_backing(&self, kind: u32) -> Option<ChunkBacking> {
        chunk_backing(&self.memory, self.device_type.memory_is_host(), self.placed_map_alignment.is_some(), kind)
    }

    /// How the backend reserves address ranges in `kind`: in chunks, each
    /// its own memory object and buffer, which device work reaches through
    /// record tables (`crate::reservation`), backed as
    /// [`Self::chunk_backing`] says; and why not where this device cannot
    /// ([`reservation_unit`]). Chunks the host addresses lie in one reserved
    /// host range, so the host addresses the reservation in place at one
    /// base, as CUDA's host-addressed storage is.
    pub fn reservation(&self, kind: u32) -> Result<Option<ReservationForm>, Unreservable> {
        let Some(backing) = self.chunk_backing(kind) else {
            return Ok(None);
        };
        reservation_form(
            backing,
            Reserving {
                queues: 1 + u32::from(self.side_family.is_some()),
                host_import_alignment: self.host_import_alignment,
                placed_map_alignment: self.placed_map_alignment,
                heap_bytes: self.memory.heaps.iter().map(|heap| heap.bytes).sum::<u64>(),
                max_allocations: self.limits.max_memory_allocations,
            },
        )
        .map(Some)
    }

    /// Tuning identity: vendor and device, name, driver ID and
    /// version, so a driver update changes the key.
    pub fn tuning_identity(&self) -> String {
        format!(
            "vulkan;vendor {:04x};device {:04x};{};driver {} {} ({})",
            self.vendor_id,
            self.device_id,
            self.name,
            self.driver_id,
            self.driver_version,
            self.driver_info
        )
    }
}

/// [`Facts::chunk_backing`] of a device of memory layout `layout`, whose
/// memory is the host's or its own, that maps at a `placed` address or not.
pub(crate) fn chunk_backing(layout: &MemoryLayout, memory_is_host: bool, placed: bool, kind: u32) -> Option<ChunkBacking> {
    if kind == crate::topology::IMPORTED_KIND {
        return (!memory_is_host || !placed).then_some(ChunkBacking::HostPages);
    }
    let memory = layout.types[kind as usize];
    match memory_is_host {
        true => (placed && memory.mappable()).then_some(ChunkBacking::Placed),
        false => memory.device_local.then_some(ChunkBacking::Device),
    }
}

/// The facts of a device that decide how it reserves address ranges
/// ([`reservation_form`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Reserving {
    /// Queues of its work queue family.
    pub queues: u32,
    /// `minImportedHostPointerAlignment`, where it imports host memory.
    pub host_import_alignment: Option<u64>,
    /// `minPlacedMemoryMapAlignment`, where it maps memory at a placed host
    /// address ([`Facts::placed_map_alignment`]).
    pub placed_map_alignment: Option<u64>,
    /// Bytes of every memory heap.
    pub heap_bytes: u64,
    /// `maxMemoryAllocationCount`.
    pub max_allocations: u64,
}

/// How a device of `reserving` reserves address ranges backed as `backing`;
/// why not where it cannot.
fn reservation_form(backing: ChunkBacking, reserving: Reserving) -> Result<ReservationForm, Unreservable> {
    let unit = reservation_unit(backing, reserving, seismic_reservation::host_page)?;
    Ok(ReservationForm {
        granule: reservation_granule(unit, reserving.heap_bytes, reserving.max_allocations),
        chunk_unit: unit,
        host_addressed: backing != ChunkBacking::Device,
    })
}

/// How the backend reserves one kind's address ranges
/// ([`Facts::reservation`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReservationForm {
    /// The least chunk, in which committed bytes are backed and charged.
    pub granule: u64,
    /// The bytes every chunk is a whole number of: `vkCmdFillBuffer`'s
    /// 4-byte unit for device memory chunks, which are zeroed with it; for
    /// chunks the host addresses the host page (a power of two, of the
    /// import or placed mapping alignment's multiples), where each starts
    /// and ends.
    pub chunk_unit: u64,
    /// Whether the host addresses the backed bytes in place, at one base:
    /// whether its chunks lie in one reserved host range ([`ChunkBacking`]).
    pub host_addressed: bool,
}

/// What backs a reservation's chunks ([`chunk_backing`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ChunkBacking {
    /// Device memory of its kind, which the host does not address.
    Device,
    /// Host pages of one reserved host range, imported
    /// (`VK_EXT_external_memory_host`): host RAM, which the host addresses
    /// at the range's base.
    HostPages,
    /// Device memory of its kind, each chunk mapped at its place in one
    /// reserved host range (`VK_EXT_map_memory_placed`), so the host
    /// addresses it at the range's base while device work reads it as the
    /// device's own (GB10: scattered reads at 238 GB/s against 19.5 for
    /// imported host pages).
    Placed,
}

/// The bytes every chunk of a reservation backed as `backing` is a whole
/// number of ([`Facts::reservation`]), on a device of `reserving`; `page`
/// reads the host page; why not where the device cannot reserve so. Device
/// memory chunks need a second queue in the work queue family, on which a
/// growth's chunks are zeroed beside work parked on the first; chunks in a
/// host range start on host pages, imported at a host page, or mapped in
/// place at an alignment no coarser than the least chunk (the host zeroes
/// them through its range).
fn reservation_unit<E: std::fmt::Display>(
    backing: ChunkBacking,
    reserving: Reserving,
    page: impl FnOnce() -> Result<u64, E>,
) -> Result<u64, Unreservable> {
    let host_page = || page().map_err(|error| Unreservable::Unread(format!("the host page size: {error}")));
    match backing {
        ChunkBacking::Device => match reserving.queues {
            0 | 1 => Err(Unreservable::OneQueue { queues: reserving.queues }),
            _ => Ok(FILL_UNIT),
        },
        ChunkBacking::HostPages => {
            let alignment = reserving.host_import_alignment.ok_or(Unreservable::NoHostImport)?;
            let page = host_page()?;
            match alignment <= page {
                true => Ok(page),
                false => Err(Unreservable::HostImportAlignment { alignment, page }),
            }
        }
        ChunkBacking::Placed => {
            let alignment = reserving.placed_map_alignment.ok_or(Unreservable::NoPlacedMapping)?;
            match alignment <= RESERVATION_CHUNK_BYTES {
                true => host_page(),
                false => Err(Unreservable::PlacedMappingAlignment { alignment, least: RESERVATION_CHUNK_BYTES }),
            }
        }
    }
}

/// The unit `vkCmdFillBuffer` writes: its offset and size are whole words.
pub(crate) const FILL_UNIT: u64 = 4;

/// The least chunk a reservation backs: 2 MiB, the large page of the GPUs
/// the backend runs (CUDA's reservation granule on the same devices,
/// `cuMemGetAllocationGranularity`), so a chunk is no smaller than the
/// pages its memory is mapped in.
const RESERVATION_CHUNK_BYTES: u64 = 2 << 20;

/// The granule of chunks of `unit` bytes on a device of `total` bytes of
/// heaps that admits `allocations` memory objects at once: the least power
/// of two of at least [`RESERVATION_CHUNK_BYTES`] and `unit` (a power of
/// two) whose chunks over every heap take at most half of them, leaving the
/// rest to the block allocator and the driver (Windows NVIDIA drivers admit
/// 4096; a 24 GiB device in 2 MiB chunks would need 12288).
fn reservation_granule(unit: u64, total: u64, allocations: u64) -> u64 {
    let mut granule = RESERVATION_CHUNK_BYTES.max(unit);
    while total.div_ceil(granule) > allocations / 2 && granule < total {
        granule *= 2;
    }
    granule
}

pub(crate) fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// One enumerated physical device: its handle, its facts, and whether it
/// meets the floor (`Err` names the first missing item).
#[derive(Clone, Debug)]
pub struct Description {
    pub handle: DeviceHandle,
    pub facts: Facts,
    pub floor: Result<(), String>,
}

/// Device extensions of the floor.
pub(crate) const FLOOR_EXTENSIONS: [&CStr; 2] = [
    ash::khr::workgroup_memory_explicit_layout::NAME,
    ash::ext::memory_budget::NAME,
];

fn text(raw: &[std::ffi::c_char]) -> String {
    // SAFETY: Vulkan fixed-size name arrays are NUL-terminated.
    unsafe { CStr::from_ptr(raw.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

/// Describe one physical device.
pub(crate) fn describe(instance: &Arc<Instance>, physical: vk::PhysicalDevice) -> Description {
    let raw = instance.raw();
    let extensions = unsafe { raw.enumerate_device_extension_properties(physical) }
        .unwrap_or_default()
        .iter()
        .map(|extension| text(&extension.extension_name))
        .collect::<Vec<_>>();
    let has = |name: &CStr| {
        extensions
            .iter()
            .any(|extension| extension.as_bytes() == name.to_bytes())
    };

    let mut v11 = vk::PhysicalDeviceVulkan11Properties::default();
    let mut v12 = vk::PhysicalDeviceVulkan12Properties::default();
    let mut v13 = vk::PhysicalDeviceVulkan13Properties::default();
    let mut host_import = vk::PhysicalDeviceExternalMemoryHostPropertiesEXT::default();
    let imports_host = has(ash::ext::external_memory_host::NAME);
    let mut properties = vk::PhysicalDeviceProperties2::default()
        .push_next(&mut v11)
        .push_next(&mut v12)
        .push_next(&mut v13);
    if imports_host {
        properties = properties.push_next(&mut host_import);
    }
    unsafe { raw.get_physical_device_properties2(physical, &mut properties) };
    let base = properties.properties;

    let mut explicit = vk::PhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR::default();
    let mut atomic_float = vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT::default();
    let mut cooperative = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default();
    let mut shader_fma = ShaderFmaFeatures::default();
    let mut f11 = vk::PhysicalDeviceVulkan11Features::default();
    let mut f12 = vk::PhysicalDeviceVulkan12Features::default();
    let mut f13 = vk::PhysicalDeviceVulkan13Features::default();
    let mut features = vk::PhysicalDeviceFeatures2::default()
        .push_next(&mut f11)
        .push_next(&mut f12)
        .push_next(&mut f13);
    if has(ash::khr::workgroup_memory_explicit_layout::NAME) {
        features = features.push_next(&mut explicit);
    }
    if has(ash::ext::shader_atomic_float::NAME) {
        features = features.push_next(&mut atomic_float);
    }
    if has(ash::khr::cooperative_matrix::NAME) {
        features = features.push_next(&mut cooperative);
    }
    if has(SHADER_FMA_EXTENSION) {
        features = features.push_next(&mut shader_fma);
    }
    unsafe { raw.get_physical_device_features2(physical, &mut features) };
    let core = features.features;
    // Placed mapping, where the device maps memory at a given host address
    // and unmaps it back to a reservation: its alignment.
    let placed_map_alignment = (has(ash::ext::map_memory_placed::NAME) && has(ash::khr::map_memory2::NAME))
        .then(|| {
            let mut placed = vk::PhysicalDeviceMapMemoryPlacedFeaturesEXT::default();
            let mut features = vk::PhysicalDeviceFeatures2::default().push_next(&mut placed);
            unsafe { raw.get_physical_device_features2(physical, &mut features) };
            let mut alignment = vk::PhysicalDeviceMapMemoryPlacedPropertiesEXT::default();
            let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut alignment);
            unsafe { raw.get_physical_device_properties2(physical, &mut properties) };
            (placed.memory_map_placed == vk::TRUE && placed.memory_unmap_reserve == vk::TRUE)
                .then_some(alignment.min_placed_memory_map_alignment)
        })
        .flatten();

    let families = unsafe { raw.get_physical_device_queue_family_properties(physical) };
    let compute_only = families.iter().position(|family| {
        family.queue_flags.contains(vk::QueueFlags::COMPUTE)
            && !family.queue_flags.contains(vk::QueueFlags::GRAPHICS)
            && family.timestamp_valid_bits > 0
    });
    let universal = families.iter().position(|family| {
        family.queue_flags.contains(vk::QueueFlags::COMPUTE) && family.timestamp_valid_bits > 0
    });
    let queue_family = compute_only.or(universal);
    let transfer_only = families.iter().position(|family| {
        family.queue_flags.contains(vk::QueueFlags::TRANSFER)
            && !family.queue_flags.intersects(vk::QueueFlags::COMPUTE | vk::QueueFlags::GRAPHICS)
    });
    let side_family = transfer_only.or(queue_family.filter(|&family| families[family].queue_count > 1)).map(|family| family as u32);

    let memory = memory_layout(&unsafe { raw.get_physical_device_memory_properties(physical) });
    let device_type = match base.device_type {
        vk::PhysicalDeviceType::DISCRETE_GPU => DeviceType::Discrete,
        vk::PhysicalDeviceType::INTEGRATED_GPU => DeviceType::Integrated,
        vk::PhysicalDeviceType::CPU => DeviceType::Cpu,
        _ => DeviceType::Other,
    };
    let allocation = Allocation::of(&memory, device_type.memory_is_host());
    let mixed_dot_accelerated =
        v13.integer_dot_product4x8_bit_packed_mixed_signedness_accelerated == vk::TRUE;
    let subgroup_width = SubgroupWidth::of(
        v13.min_subgroup_size,
        v13.max_subgroup_size,
        v13.required_subgroup_size_stages
            .contains(vk::ShaderStageFlags::COMPUTE),
    );
    let float16 = f12.shader_float16 == vk::TRUE
        && v12.shader_rounding_mode_rte_float16 == vk::TRUE
        && v12.shader_signed_zero_inf_nan_preserve_float16 == vk::TRUE;
    let matrix = cooperative.cooperative_matrix == vk::TRUE
        && float16
        && subgroup_width.is_some_and(|width| width.lanes() == SUBGROUP_WIDTH)
        && matrix_shapes(instance, physical)
        // RDNA2 has no WMMA units whatever its driver reports; accelerated
        // mixed-signedness packed dot identifies RDNA3+.
        && (base.vendor_id != AMD_VENDOR || mixed_dot_accelerated)
        // The NVIDIA proprietary compiler (580.159.03, GB10) runs the
        // projection GEMM's cooperative-matrix path 1.3x (paired) to over
        // 30x (single weight) slower than its FMA path.
        && v12.driver_id != vk::DriverId::NVIDIA_PROPRIETARY;
    let calibration = if has(ash::khr::calibrated_timestamps::NAME) {
        Some(Calibration::Khr)
    } else if has(ash::ext::calibrated_timestamps::NAME) {
        Some(Calibration::Ext)
    } else {
        None
    };
    let limits = base.limits;
    let facts = Facts {
        name: text(&base.device_name),
        uuid: v11.device_uuid,
        vendor_id: base.vendor_id,
        device_id: base.device_id,
        device_type,
        driver_id: v12.driver_id.as_raw(),
        driver_name: text(&v12.driver_name),
        driver_info: text(&v12.driver_info),
        driver_version: base.driver_version,
        api_version: base.api_version,
        pipeline_cache_uuid: base.pipeline_cache_uuid,
        limits: Limits {
            max_shared_bytes: limits.max_compute_shared_memory_size.into(),
            max_invocations: limits.max_compute_work_group_invocations.into(),
            max_group_count: limits.max_compute_work_group_count.map(u64::from),
            max_group_size: limits.max_compute_work_group_size.map(u64::from),
            max_allocation_bytes: v11.max_memory_allocation_size.min(v13.max_buffer_size),
            max_memory_allocations: limits.max_memory_allocation_count.into(),
            timestamp_period: f64::from(limits.timestamp_period),
            timestamp_valid_bits: queue_family
                .map_or(0, |family| families[family].timestamp_valid_bits),
        },
        queue_family: queue_family.map_or(0, |family| family as u32),
        dedicated_compute_queue: compute_only.is_some(),
        queue_count: queue_family.map_or(0, |family| families[family].queue_count),
        side_family,
        host_import_alignment: imports_host.then_some(host_import.min_imported_host_pointer_alignment),
        placed_map_alignment,
        subgroup_width,
        float16,
        matrix,
        wide_accumulators: v12.driver_id != vk::DriverId::NVIDIA_PROPRIETARY,
        denorm_preserve_32: v12.shader_denorm_preserve_float32 == vk::TRUE,
        rounding_rte_32: v12.driver_id != vk::DriverId::NVIDIA_PROPRIETARY,
        mixed_dot_accelerated,
        shader_fma: ShaderFma {
            float16: shader_fma.float16 == vk::TRUE,
            float32: shader_fma.float32 == vk::TRUE,
            float64: shader_fma.float64 == vk::TRUE,
        },
        f32_atomic_add: atomic_float.shader_buffer_float32_atomic_add == vk::TRUE,
        shared_int64_atomics: f12.shader_shared_int64_atomics == vk::TRUE,
        calibration,
        luid: (v11.device_luid_valid == vk::TRUE).then_some(v11.device_luid),
        memory,
        allocation,
    };

    let on = |value: vk::Bool32| value == vk::TRUE;
    let subgroup_operations = vk::SubgroupFeatureFlags::BASIC
        | vk::SubgroupFeatureFlags::VOTE
        | vk::SubgroupFeatureFlags::ARITHMETIC
        | vk::SubgroupFeatureFlags::BALLOT
        | vk::SubgroupFeatureFlags::SHUFFLE
        | vk::SubgroupFeatureFlags::SHUFFLE_RELATIVE
        | vk::SubgroupFeatureFlags::CLUSTERED
        | vk::SubgroupFeatureFlags::QUAD;
    let requirements = [
        ("Vulkan 1.3", vk::api_version_major(base.api_version) > 1 || vk::api_version_minor(base.api_version) >= 3),
        ("a compute queue with timestamps", queue_family.is_some()),
        ("bufferDeviceAddress", on(f12.buffer_device_address)),
        ("shaderInt64", on(core.shader_int64)),
        ("shaderInt16", on(core.shader_int16)),
        ("shaderInt8", on(f12.shader_int8)),
        ("storageBuffer16BitAccess", on(f11.storage_buffer16_bit_access)),
        ("storageBuffer8BitAccess", on(f12.storage_buffer8_bit_access)),
        ("scalarBlockLayout", on(f12.scalar_block_layout)),
        ("timelineSemaphore", on(f12.timeline_semaphore)),
        ("synchronization2", on(f13.synchronization2)),
        ("maintenance4", on(f13.maintenance4)),
        ("pipelineCreationCacheControl", on(f13.pipeline_creation_cache_control)),
        ("vulkanMemoryModel", on(f12.vulkan_memory_model)),
        ("vulkanMemoryModelDeviceScope", on(f12.vulkan_memory_model_device_scope)),
        ("shaderBufferInt64Atomics", on(f12.shader_buffer_int64_atomics)),
        ("shaderSubgroupExtendedTypes", on(f12.shader_subgroup_extended_types)),
        ("shaderIntegerDotProduct", on(f13.shader_integer_dot_product)),
        ("subgroupSizeControl", on(f13.subgroup_size_control)),
        ("computeFullSubgroups", on(f13.compute_full_subgroups)),
        ("a requirable subgroup size of 32, or a fixed subgroup size of 32 or 64", subgroup_width.is_some()),
        ("subgroup operations in compute", v11.subgroup_supported_stages.contains(vk::ShaderStageFlags::COMPUTE)),
        ("basic, vote, arithmetic, ballot, shuffle, shuffle-relative, clustered and quad subgroup operations", v11.subgroup_supported_operations.contains(subgroup_operations)),
        ("VK_KHR_workgroup_memory_explicit_layout", has(FLOOR_EXTENSIONS[0])),
        ("workgroupMemoryExplicitLayout (base, 8-bit, 16-bit, scalar)", on(explicit.workgroup_memory_explicit_layout) && on(explicit.workgroup_memory_explicit_layout8_bit_access) && on(explicit.workgroup_memory_explicit_layout16_bit_access) && on(explicit.workgroup_memory_explicit_layout_scalar_block_layout)),
        ("VK_EXT_memory_budget", has(FLOOR_EXTENSIONS[1])),
        ("RTE rounding for fp32", on(v12.shader_rounding_mode_rte_float32)),
        ("SignedZeroInfNanPreserve for fp32", on(v12.shader_signed_zero_inf_nan_preserve_float32)),
        ("32 KiB of shared memory", limits.max_compute_shared_memory_size >= FLOOR_SHARED_BYTES),
        ("1024 invocations per workgroup", limits.max_compute_work_group_invocations >= FLOOR_INVOCATIONS),
        ("65535 workgroups on every axis", limits.max_compute_work_group_count.iter().all(|count| *count >= FLOOR_GROUP_COUNT)),
        ("workgroup size (1024, 1024, 64)", limits.max_compute_work_group_size.iter().zip(FLOOR_GROUP_SIZE).all(|(size, floor)| *size >= floor)),
        ("128 bytes of push constants", limits.max_push_constants_size >= FLOOR_PUSH_CONSTANT_BYTES),
        ("a GPU or CPU device type", !matches!(facts.device_type, DeviceType::Other)),
        (
            "a device-local heap",
            facts.memory.heaps.iter().any(|heap| heap.device_local && heap.bytes > 0),
        ),
        ("host-visible coherent memory", facts.memory.types.iter().any(MemoryType::mappable)),
        ("device memory", !facts.allocation.device_heaps.is_empty()),
        ("host staging memory", facts.allocation.staging.is_some()),
        ("a nonzero allocation limit", facts.limits.max_allocation_bytes > 0),
    ];
    let floor = match requirements.iter().find(|(_, met)| !met) {
        Some((missing, _)) => Err(format!("missing {missing}")),
        None => Ok(()),
    };
    Description {
        handle: DeviceHandle { instance: instance.clone(), physical },
        facts,
        floor,
    }
}

/// The heaps and types of `properties`, in the driver's order.
pub(crate) fn memory_layout(properties: &vk::PhysicalDeviceMemoryProperties) -> MemoryLayout {
    let heaps = &properties.memory_heaps[..properties.memory_heap_count as usize];
    let types = &properties.memory_types[..properties.memory_type_count as usize];
    MemoryLayout {
        heaps: heaps
            .iter()
            .map(|heap| MemoryHeap {
                bytes: heap.size,
                device_local: heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL),
            })
            .collect(),
        types: types
            .iter()
            .map(|kind| {
                let has = |flag| kind.property_flags.contains(flag);
                MemoryType {
                    heap: kind.heap_index,
                    device_local: has(vk::MemoryPropertyFlags::DEVICE_LOCAL),
                    host_visible: has(vk::MemoryPropertyFlags::HOST_VISIBLE),
                    host_coherent: has(vk::MemoryPropertyFlags::HOST_COHERENT),
                    host_cached: has(vk::MemoryPropertyFlags::HOST_CACHED),
                }
            })
            .collect(),
    }
}

/// Whether the device offers subgroup-scope 16x16x16 cooperative matrix
/// with f16 x f16 -> f32, the one shape the kernels multiply with. (Integer
/// shapes are not required: no kernel uses them, and NVIDIA offers s8 only
/// as 16x16x32.)
fn matrix_shapes(instance: &Instance, physical: vk::PhysicalDevice) -> bool {
    let loader = ash::khr::cooperative_matrix::Instance::new(instance.entry(), instance.raw());
    let Ok(shapes) =
        (unsafe { loader.get_physical_device_cooperative_matrix_properties(physical) })
    else {
        return false;
    };
    shapes.iter().any(|shape| {
        shape.scope == vk::ScopeKHR::SUBGROUP
            && (shape.m_size, shape.n_size, shape.k_size) == (16, 16, 16)
            && shape.a_type == vk::ComponentTypeKHR::FLOAT16
            && shape.b_type == vk::ComponentTypeKHR::FLOAT16
            && shape.c_type == vk::ComponentTypeKHR::FLOAT32
            && shape.result_type == vk::ComponentTypeKHR::FLOAT32
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn memory_type(heap: u32, device_local: bool, mappable: bool) -> MemoryType {
        MemoryType {
            heap,
            device_local,
            host_visible: mappable,
            host_coherent: mappable,
            host_cached: false,
        }
    }

    /// A discrete device's layout: its device-local heap, a host heap, and
    /// a device-local mappable heap of `bar` bytes.
    fn discrete(bar: u64) -> MemoryLayout {
        MemoryLayout {
            heaps: vec![
                MemoryHeap {
                    bytes: 24 << 30,
                    device_local: true,
                },
                MemoryHeap {
                    bytes: 64 << 30,
                    device_local: false,
                },
                MemoryHeap {
                    bytes: bar,
                    device_local: true,
                },
            ],
            types: vec![
                memory_type(0, true, false),
                memory_type(1, false, true),
                memory_type(2, true, true),
            ],
        }
    }

    /// Imported pages and staging lie in host RAM, host-cached types first;
    /// the device's budget reads its first device-local type's heap.
    #[test]
    fn the_backends_own_storage_is_host_ram_cached_first() {
        let mut layout = discrete(1 << 30);
        let mut cached = memory_type(1, false, true);
        cached.host_cached = true;
        layout.types.push(cached);
        let allocation = Allocation::of(&layout, false);
        assert_eq!(allocation.host, vec![3, 1]);
        assert_eq!(allocation.staging, Some(3));
        assert_eq!(allocation.device_heaps, vec![0]);
    }

    /// Device memory chunks need a second queue, which zeroes a growth, and
    /// say why not on one; imported host pages say why not where host memory
    /// imports at no host page or not at all; chunks mapped in place need a
    /// placed alignment no coarser than the least chunk (no second queue: the
    /// host zeroes them).
    #[test]
    fn a_reservation_needs_its_backing_own_requirement() {
        let page = || Ok::<_, String>(16384);
        let reserving = |queues, import, placed| Reserving {
            queues,
            host_import_alignment: import,
            placed_map_alignment: placed,
            heap_bytes: 1 << 30,
            max_allocations: u64::from(u32::MAX),
        };
        let unit = |backing, queues, import, placed| reservation_unit(backing, reserving(queues, import, placed), page);
        assert_eq!(unit(ChunkBacking::Device, 2, None, None), Ok(FILL_UNIT));
        assert_eq!(unit(ChunkBacking::Device, 1, Some(4096), None), Err(Unreservable::OneQueue { queues: 1 }));
        assert_eq!(unit(ChunkBacking::HostPages, 1, Some(4096), None), Ok(16384));
        assert_eq!(unit(ChunkBacking::Placed, 1, None, Some(65536)), Ok(16384), "mapped in place, the host page");
        assert_eq!(
            unit(ChunkBacking::Placed, 2, None, Some(4 << 20)),
            Err(Unreservable::PlacedMappingAlignment { alignment: 4 << 20, least: RESERVATION_CHUNK_BYTES })
        );
        assert_eq!(unit(ChunkBacking::HostPages, 2, None, None), Err(Unreservable::NoHostImport));
        assert_eq!(
            unit(ChunkBacking::HostPages, 2, Some(65536), None),
            Err(Unreservable::HostImportAlignment { alignment: 65536, page: 16384 })
        );
        let unread = reservation_unit(ChunkBacking::HostPages, reserving(2, Some(4096), None), || Err("no sysconf")).unwrap_err();
        assert!(matches!(&unread, Unreservable::Unread(why) if why.contains("no sysconf")), "{unread}");
    }

    /// A reservation's chunks are 2 MiB where the device admits the memory
    /// objects to chunk every heap twice over, and double until they fit
    /// half its memory objects; a unit above 2 MiB is the least chunk.
    #[test]
    fn reservation_chunks_fit_half_the_memory_objects() {
        const GIB: u64 = 1 << 30;
        // A Linux NVIDIA driver admits u32::MAX objects; a 24 GiB device
        // with 64 GiB of host heap is chunked at 2 MiB.
        assert_eq!(reservation_granule(4, 88 * GIB, u64::from(u32::MAX)), 2 << 20);
        // A Windows NVIDIA driver admits 4096: 88 GiB in at most 2048
        // chunks is 44 MiB, so 64 MiB.
        assert_eq!(reservation_granule(4, 88 * GIB, 4096), 64 << 20);
        assert!(88 * GIB / (64 << 20) <= 2048 && 88 * GIB / (32 << 20) > 2048);
        assert_eq!(reservation_granule(4 << 20, 8 * GIB, u64::from(u32::MAX)), 4 << 20);
    }

    /// The RTX 3090 as one reports it (vulkaninfo, 2026-11-02: the
    /// compute-only family's 8 queues, host import at 4 KiB on its 4 KiB
    /// host page, maxMemoryAllocationCount 2^32 − 1, the fixture's heaps)
    /// reserves as its fixture records: device storage in fill-unit chunks,
    /// host storage in host pages, both at the 2 MiB least chunk.
    #[test]
    fn the_recorded_rtx3090_reserves_as_its_fixture_records() {
        use seismic_topology::fixtures::{rtx3090_uuid, rtx3090_vulkan};
        let report = rtx3090_vulkan(rtx3090_uuid());
        let heap_bytes = report.memory.heaps.iter().map(|heap| heap.bytes).sum::<u64>();
        let reserving = Reserving {
            queues: 8,
            host_import_alignment: Some(4096),
            placed_map_alignment: None,
            heap_bytes,
            max_allocations: 4_294_967_295,
        };
        let form = |backing, kind| {
            let unit = reservation_unit(backing, reserving, || Ok::<_, String>(4096)).unwrap();
            let granule = reservation_granule(unit, heap_bytes, 4_294_967_295);
            let recorded = report.kinds.iter().find(|each| each.id == kind).unwrap();
            let recorded = recorded.reservation.clone().unwrap().unwrap();
            assert_eq!(
                recorded,
                seismic_topology::Reservation {
                    granule,
                    host_addressed: backing != ChunkBacking::Device,
                    chunks: Some(seismic_topology::ReservationChunks { unit, table: seismic_topology::Contract::Upload }),
                }
            );
            (granule, unit)
        };
        assert_eq!(form(ChunkBacking::Device, 0), (2 << 20, FILL_UNIT));
        assert_eq!(form(ChunkBacking::HostPages, crate::topology::IMPORTED_KIND), (2 << 20, 4096));
    }
}
