//! `MetalFacts`: every Metal-specific target fact planning and native
//! compilation consume, gathered once at device open (spec §4.1).
//!
//! These are immutable observations, not a live device or native artifact.
//! Pipeline-specific resource limits still come from authoritative reflection
//! after compilation. Device discovery alone does not establish that every
//! constructible kernel satisfies those limits.

use seismic_lang::types::DType;
use std::collections::BTreeSet;

/// Vendor GPU families are observations used to derive facts; they never
/// appear in kernel source.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MetalFamily {
    Metal3,
    Metal4,
    Apple(u8),
    Mac2,
}

impl MetalFamily {
    pub fn label(self) -> String {
        match self {
            Self::Metal3 => "metal3".into(),
            Self::Metal4 => "metal4".into(),
            Self::Apple(generation) => format!("apple{generation}"),
            Self::Mac2 => "mac2".into(),
        }
    }
}

/// The Metal Shading Language version the device's compiler accepts, as
/// established by a compile probe.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LanguageVersion {
    V2_3,
    V2_4,
    V3_0,
    V3_1,
    V3_2,
    V4_0,
}

impl LanguageVersion {
    pub fn label(self) -> &'static str {
        match self {
            Self::V2_3 => "2.3",
            Self::V2_4 => "2.4",
            Self::V3_0 => "3.0",
            Self::V3_1 => "3.1",
            Self::V3_2 => "3.2",
            Self::V4_0 => "4.0",
        }
    }

    pub(crate) fn native(self) -> objc2_metal::MTLLanguageVersion {
        match self {
            Self::V2_3 => objc2_metal::MTLLanguageVersion::Version2_3,
            Self::V2_4 => objc2_metal::MTLLanguageVersion::Version2_4,
            Self::V3_0 => objc2_metal::MTLLanguageVersion::Version3_0,
            Self::V3_1 => objc2_metal::MTLLanguageVersion::Version3_1,
            Self::V3_2 => objc2_metal::MTLLanguageVersion::Version3_2,
            Self::V4_0 => objc2_metal::MTLLanguageVersion::Version4_0,
        }
    }
}

/// One multiply-accumulate dtype combination the native compiler accepts
/// for `simdgroup_multiply_accumulate`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MatrixCombination {
    pub accumulator: DType,
    pub left: DType,
    pub right: DType,
}

/// Complete Metal facts of one device.
#[derive(Clone, Debug, PartialEq)]
pub struct MetalFacts {
    pub(crate) device_name: String,
    pub(crate) architecture: String,
    /// Exact operating-system build string returned by `NSProcessInfo`; the
    /// Metal compiler/runtime and display driver ship as part of this image.
    pub(crate) operating_system: String,
    pub(crate) registry_id: u64,
    pub(crate) unified_memory: bool,
    pub(crate) families: BTreeSet<MetalFamily>,
    pub(crate) language: LanguageVersion,
    /// `MTLDevice.maxThreadsPerThreadgroup`, per axis.
    pub(crate) max_threads_per_threadgroup: [u64; 3],
    /// `MTLDevice.maxThreadgroupMemoryLength`.
    pub(crate) max_threadgroup_bytes: u64,
    /// `MTLDevice.maxBufferLength`.
    pub(crate) max_buffer_bytes: u64,
    /// `MTLDevice.heapBufferSizeAndAlignWithLength:options:` alignment for
    /// the shared buffer class used by the production allocator.
    pub(crate) buffer_alignment: u64,
    /// Scalar dtypes the compiler accepts in `simd_sum/max/min/shuffle`.
    pub(crate) scalar_collective_dtypes: BTreeSet<DType>,
    /// Whether `bfloat` arithmetic compiles at the probed language version.
    pub(crate) bfloat_arithmetic: bool,
    /// Element dtypes accepted for `simdgroup_matrix` declaration, load and
    /// store.
    pub(crate) matrix_dtypes: BTreeSet<DType>,
    /// Multiply-accumulate combinations the compiler accepts.
    pub(crate) matrix_combinations: BTreeSet<MatrixCombination>,
    /// Buffer argument-table entries of one compute pipeline (Metal: 31).
    pub(crate) argument_table_entries: u32,
    /// Argument-table entries this backend reserves for its own kernel
    /// arguments (parameter words, side/status, participant scratch, and
    /// register scratch).
    pub(crate) reserved_argument_entries: u32,
    pub(crate) backend_revision: &'static str,
}

/// Buffer entries of one compute pipeline's argument table.
pub const ARGUMENT_TABLE_ENTRIES: u32 = 31;
/// Entries this backend reserves beyond the kernel's bindings: the
/// parameter word block and the side (status + result slots) block.
pub const RESERVED_ARGUMENT_ENTRIES: u32 = 4;
/// The emitted MSL launch ABI represents grid coordinates with `uint3`.
/// This is a language-level representability bound, not a guessed hardware
/// limit.
pub const MSL_GRID_INDEX_MAX: u64 = u32::MAX as u64;

impl MetalFacts {
    /// Every fact, serialized deterministically for the profile fingerprint.
    pub fn fingerprint_material(&self) -> String {
        let families = self
            .families
            .iter()
            .map(|family| family.label())
            .collect::<Vec<_>>()
            .join(",");
        let dtypes = |set: &BTreeSet<DType>| {
            set.iter()
                .map(|dtype| dtype.name())
                .collect::<Vec<_>>()
                .join(",")
        };
        let combinations = self
            .matrix_combinations
            .iter()
            .map(|c| {
                format!(
                    "{}:{}:{}",
                    c.accumulator.name(),
                    c.left.name(),
                    c.right.name()
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        format!(
            "backend={};arch={};os={};unified={};families={families};msl={};threads={:?};\
             threadgroup={};buffer={};alignment={};scalar={};bfloat={};matrix={};mma={combinations};\
             args={};reserved={}",
            self.backend_revision,
            self.architecture,
            self.operating_system,
            self.unified_memory,
            self.language.label(),
            self.max_threads_per_threadgroup,
            self.max_threadgroup_bytes,
            self.max_buffer_bytes,
            self.buffer_alignment,
            dtypes(&self.scalar_collective_dtypes),
            self.bfloat_arithmetic,
            dtypes(&self.matrix_dtypes),
            self.argument_table_entries,
            self.reserved_argument_entries,
        )
    }
}
