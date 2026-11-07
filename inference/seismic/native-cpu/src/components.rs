//! Component instances: the element-specific inner loops, compiled once
//! for the program per tier × component × row block, and resolved into
//! typed handles when a kernel is prepared.
//!
//! The components are the registry's: build.rs instantiates one per decode
//! of the weight elements (`COMPONENT_OF`), which elements of one decode in
//! different layouts share. Each instance is a `#[target_feature]` function
//! of its tier wrapping a generic body of `weights`; the body inlines into it
//! and compiles with the tier's features. A call covers a whole row block
//! across the reduction dimension, so the one indirect call per block is
//! amortized.

use crate::isa::Tier;
use crate::quant::Q8Block;
use crate::weights::{
    self, codes, scales, Carried, Delta, DenseRows, Direct, Format, Grid, Hier, Minimum, NoDelta,
    PackedRows, Planes, RowGeometry, Scales, SignedBias, Unbiased,
};
use seismic_lang::ids::RepresentationId;
#[cfg(test)]
use seismic_lang::registry;

include!(concat!(env!("OUT_DIR"), "/elements.rs"));

/// Row blocks of the dot component, in the order of [`WeightKernels::dot`]:
/// up to the largest row group a layout's rows share their bytes in
/// (`weights::Placement::row_block`).
pub const ROW_BLOCKS: [usize; 5] = [1, 2, 4, 8, 16];

/// Activation rows of one quantized dot component call, in the order of
/// [`WeightKernels::dot_q8`]'s instances: what `Format::dot_q8` decodes a
/// weight block once for.
const ACTIVATION_GROUPS: [usize; 3] = [1, 2, 4];

type DecodeFn = unsafe fn(*const u8, &RowGeometry, usize, *mut f32);
type DotFn = unsafe fn(*const u8, &RowGeometry, *const f32, usize, *mut f32);
type DotQ8Fn = unsafe fn(*const u8, &RowGeometry, *const Q8Block, usize, *mut f32);
type DotQ8TilesFn = unsafe fn(*const u8, &RowGeometry, *const Q8Block, usize, *mut f32, usize) -> bool;

/// The instances of one component on one tier.
#[derive(Clone, Copy)]
pub struct WeightKernels {
    pub tier: Tier,
    decode: DecodeFn,
    /// One instance per entry of [`ROW_BLOCKS`].
    dot: [DotFn; 5],
    /// The same against quantized activation rows, one instance per entry
    /// of [`ROW_BLOCKS`] by entry of [`ACTIVATION_GROUPS`].
    dot_q8: [[DotQ8Fn; 3]; 5],
    /// Whole 16-row tiles against quantized activations, where the element
    /// has a whole-tile form (`Format::dot_q8_tiles`).
    dot_q8_tiles: DotQ8TilesFn,
}

impl std::fmt::Debug for WeightKernels {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WeightKernels")
            .field("tier", &self.tier)
            .finish()
    }
}

/// The index of `block` in [`ROW_BLOCKS`].
fn row_block(block: usize) -> usize {
    ROW_BLOCKS
        .iter()
        .position(|known| *known == block)
        .unwrap_or_else(|| panic!("a dot row block has {block} rows; blocks are {ROW_BLOCKS:?}"))
}

impl WeightKernels {
    /// Decodes the `k` values of the row at `row` into `out`.
    ///
    /// # Safety
    /// `row` addresses a row of this component's elements with `geometry`.
    #[inline(always)]
    pub unsafe fn decode(&self, row: *const u8, geometry: &RowGeometry, k: usize, out: &mut [f32]) {
        assert!(out.len() >= k);
        unsafe { (self.decode)(row, geometry, k, out.as_mut_ptr()) }
    }

    /// `out[r] = row r · x` for the `out.len()` rows from `rows` (an entry
    /// of [`ROW_BLOCKS`]), `geometry.stride` bytes apart.
    ///
    /// # Safety
    /// `rows` addresses that many rows with `geometry` and `x.len()` values.
    #[inline(always)]
    pub unsafe fn dot(&self, rows: *const u8, geometry: &RowGeometry, x: &[f32], out: &mut [f32]) {
        let block = row_block(out.len());
        unsafe { (self.dot[block])(rows, geometry, x.as_ptr(), x.len(), out.as_mut_ptr()) }
    }

    /// As [`WeightKernels::dot`] of a block of `R = out.len() / m` rows,
    /// against each of the `m` activation rows of `k` values quantized into
    /// `x` (`quant::blocks(k)` blocks each, row after row), activation-major:
    /// `out[a R + r] = row r · activation a`. The activation rows go in the
    /// largest groups that fit ([`ACTIVATION_GROUPS`]), each one pass over
    /// the weight rows decoding every block once for the group
    /// (`weights::dot_q8_rows`); every activation row's sums are the ones it
    /// has alone.
    ///
    /// # Safety
    /// `rows` addresses that many rows with `geometry` and `k` values.
    #[inline(always)]
    pub unsafe fn dot_q8(
        &self,
        rows: *const u8,
        geometry: &RowGeometry,
        x: &[Q8Block],
        k: usize,
        out: &mut [f32],
    ) {
        let blocks = crate::quant::blocks(k);
        let activations = x.len() / blocks;
        assert!(
            activations > 0 && x.len() == activations * blocks && out.len() % activations == 0,
            "{} quantized activation blocks of {k}-value rows for {} sums",
            x.len(),
            out.len()
        );
        let width = out.len() / activations;
        let block = row_block(width);
        let mut done = 0;
        while done < activations {
            let group = ACTIVATION_GROUPS
                .iter()
                .rposition(|group| *group <= activations - done)
                .expect("the one-row group fits any remainder");
            let (x, out) = (x[done * blocks..].as_ptr(), out[done * width..].as_mut_ptr());
            unsafe { (self.dot_q8[block][group])(rows, geometry, x, k, out) };
            done += ACTIVATION_GROUPS[group];
        }
    }

    /// `out.len()` rows from `rows` (a multiple of 16, each 16 a tile)
    /// against quantized activations `x`, where the element has a
    /// whole-tile form for them (`Format::dot_q8_tiles`): the sums
    /// [`Self::dot_q8`] gives, bit for bit. False, `out` untouched, where it
    /// has none.
    ///
    /// # Safety
    /// `rows` addresses `out.len()` rows with `geometry` and `k` values.
    pub unsafe fn dot_q8_tiles(&self, rows: *const u8, geometry: &RowGeometry, x: &[Q8Block], k: usize, out: &mut [f32]) -> bool {
        assert_eq!(x.len(), crate::quant::blocks(k), "quantized activation blocks of a {k}-value row");
        unsafe { (self.dot_q8_tiles)(rows, geometry, x.as_ptr(), k, out.as_mut_ptr(), out.len()) }
    }
}

/// `[kernels::<C>(), ..]` of the listed components, in the tier module that
/// expands it.
macro_rules! kernel_table {
    ($($component:ty),* $(,)?) => {
        [$(kernels::<$component>()),*]
    };
}

/// `[inlined dots of C per row block, ..]`, in the tier module that expands
/// it.
#[cfg(test)]
macro_rules! inlined_table {
    ($($component:ty),* $(,)?) => {
        [$([
            inlined::<$component, 1>,
            inlined::<$component, 2>,
            inlined::<$component, 4>,
            inlined::<$component, 8>,
            inlined::<$component, 16>,
        ]),*]
    };
}

macro_rules! tier_components {
    ($module:ident, $tier:ident, $features:literal) => {
        pub(crate) mod $module {
            use super::*;

            #[cfg(test)]
            pub(crate) const FEATURES: &str = $features;

            #[target_feature(enable = $features)]
            unsafe fn decode<W: Format>(
                row: *const u8,
                geometry: &RowGeometry,
                k: usize,
                out: *mut f32,
            ) {
                let out = unsafe { std::slice::from_raw_parts_mut(out, k) };
                unsafe { weights::decode_row::<W, crate::isa::$tier>(row, geometry, k, out) }
            }

            /// Every block of `R` rows of `rows` rows dotted with `x`, the
            /// dot body inlined: the reference a component call is measured
            /// against.
            #[cfg(test)]
            #[target_feature(enable = $features)]
            unsafe fn inlined<W: Format, const R: usize>(
                data: *const u8,
                rows: usize,
                geometry: &RowGeometry,
                x: &[f32],
                out: &mut [f32],
            ) {
                for first in (0..rows).step_by(R) {
                    let result = unsafe {
                        weights::dot_rows::<W, R, crate::isa::$tier>(
                            data.add(first * geometry.stride),
                            geometry,
                            x,
                            x.len(),
                        )
                    };
                    out[first..first + R].copy_from_slice(&result);
                }
            }

            /// Each component's inlined dots, per row block.
            #[cfg(test)]
            pub(crate) static INLINED: [[super::tests::InlinedFn; ROW_BLOCKS.len()]; COMPONENTS] =
                with_components!(inlined_table);

            #[target_feature(enable = $features)]
            unsafe fn dot<W: Format, const R: usize>(
                rows: *const u8,
                geometry: &RowGeometry,
                x: *const f32,
                k: usize,
                out: *mut f32,
            ) {
                let x = unsafe { std::slice::from_raw_parts(x, k) };
                let result =
                    unsafe { weights::dot_rows::<W, R, crate::isa::$tier>(rows, geometry, x, k) };
                unsafe { std::ptr::copy_nonoverlapping(result.as_ptr(), out, R) };
            }

            #[target_feature(enable = $features)]
            unsafe fn dot_q8<W: Format, const R: usize, const M: usize>(
                rows: *const u8,
                geometry: &RowGeometry,
                x: *const Q8Block,
                k: usize,
                out: *mut f32,
            ) {
                let x = unsafe { std::slice::from_raw_parts(x, M * crate::quant::blocks(k)) };
                let result = unsafe {
                    weights::dot_q8_rows::<W, R, M, crate::isa::$tier>(rows, geometry, x, k)
                };
                unsafe { std::ptr::copy_nonoverlapping(result.as_ptr().cast::<f32>(), out, M * R) };
            }

            #[target_feature(enable = $features)]
            unsafe fn dot_q8_tiles<W: Format>(
                rows: *const u8,
                geometry: &RowGeometry,
                x: *const Q8Block,
                k: usize,
                out: *mut f32,
                len: usize,
            ) -> bool {
                let x = unsafe { std::slice::from_raw_parts(x, crate::quant::blocks(k)) };
                let out = unsafe { std::slice::from_raw_parts_mut(out, len) };
                unsafe { W::dot_q8_tiles::<crate::isa::$tier>(rows, geometry, x, k, out) }
            }

            const fn kernels<W: Format>() -> WeightKernels {
                WeightKernels {
                    tier: Tier::$tier,
                    decode: decode::<W>,
                    dot: [dot::<W, 1>, dot::<W, 2>, dot::<W, 4>, dot::<W, 8>, dot::<W, 16>],
                    dot_q8: [
                        [dot_q8::<W, 1, 1>, dot_q8::<W, 1, 2>, dot_q8::<W, 1, 4>],
                        [dot_q8::<W, 2, 1>, dot_q8::<W, 2, 2>, dot_q8::<W, 2, 4>],
                        [dot_q8::<W, 4, 1>, dot_q8::<W, 4, 2>, dot_q8::<W, 4, 4>],
                        [dot_q8::<W, 8, 1>, dot_q8::<W, 8, 2>, dot_q8::<W, 8, 4>],
                        [dot_q8::<W, 16, 1>, dot_q8::<W, 16, 2>, dot_q8::<W, 16, 4>],
                    ],
                    dot_q8_tiles: dot_q8_tiles::<W>,
                }
            }

            pub(crate) static KERNELS: [WeightKernels; COMPONENTS] = with_components!(kernel_table);
        }
    };
}

#[cfg(target_arch = "x86_64")]
tier_components!(x86v2, X86V2, "sse3,ssse3,sse4.1,sse4.2,popcnt");
#[cfg(target_arch = "x86_64")]
tier_components!(
    x86v3,
    X86V3,
    "sse3,ssse3,sse4.1,sse4.2,popcnt,avx,avx2,fma,f16c,bmi1,bmi2,lzcnt,movbe"
);
#[cfg(target_arch = "x86_64")]
tier_components!(
    x86v4,
    X86V4,
    "sse3,ssse3,sse4.1,sse4.2,popcnt,avx,avx2,fma,f16c,bmi1,bmi2,lzcnt,movbe,avx512f,avx512cd,avx512bw,avx512dq,avx512vl"
);
#[cfg(target_arch = "x86_64")]
tier_components!(
    x86v4vnni,
    X86V4Vnni,
    "sse3,ssse3,sse4.1,sse4.2,popcnt,avx,avx2,fma,f16c,bmi1,bmi2,lzcnt,movbe,avx512f,avx512cd,avx512bw,avx512dq,avx512vl,avx512vnni"
);
#[cfg(target_arch = "aarch64")]
tier_components!(neon, Neon, "neon");

/// Every tier's component table of this architecture.
#[cfg(target_arch = "x86_64")]
static TABLES: [&[WeightKernels; COMPONENTS]; 4] = [
    &x86v2::KERNELS,
    &x86v3::KERNELS,
    &x86v4::KERNELS,
    &x86v4vnni::KERNELS,
];
#[cfg(target_arch = "aarch64")]
static TABLES: [&[WeightKernels; COMPONENTS]; 1] = [&neon::KERNELS];

/// The name of every element with a component: the floating dense elements
/// and the storage elements of the CPU row layouts.
#[cfg(test)]
pub(crate) fn elements() -> impl Iterator<Item = &'static str> {
    registry::representations()
        .iter()
        .zip(COMPONENT_OF)
        .filter(|(_, component)| component.is_some())
        .map(|(info, _)| info.name)
}

/// The component index of element `element`, or `None` when it has none.
fn component(element: RepresentationId) -> Option<usize> {
    COMPONENT_OF[element.position()]
}

/// The components of storage element `element` (`registry::storage_of`) on
/// `tier`, or `None` when it has none.
pub(crate) fn resolve(tier: Tier, element: RepresentationId) -> Option<&'static WeightKernels> {
    let component = component(element)?;
    TABLES
        .iter()
        .map(|table| &table[component])
        .find(|kernels| kernels.tier == tier)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::element::{f32_to_bf16, f32_to_f16};
    use seismic_lang::registry::{
        self, CodingAxes, Layout, RepresentationKind, RowPlaneRole,
    };
    use seismic_lang::types::DType;

    /// A cross-check of build.rs's deduplication: the components it emits
    /// are as many as the distinct decodes (a dense element's dtype, a
    /// packed storage's decode axes and storage group) of the elements that
    /// have one, each compiled once per tier, row block and body.
    #[test]
    fn components_are_the_distinct_decodes_of_the_elements() {
        let mut decodes: Vec<(Option<DType>, Option<(registry::DecodeAxes, u32)>)> = Vec::new();
        for name in elements() {
            let info = registry::representation_info(registry::representation(name).unwrap());
            let decode = match &info.kind {
                RepresentationKind::Dense(dtype) => (Some(*dtype), None),
                RepresentationKind::PackedRows(rows) => (
                    None,
                    Some((registry::decode_axes(info.id).unwrap(), rows.group())),
                ),
                _ => unreachable!("`{name}` is a dense or packed rows element"),
            };
            if !decodes.contains(&decode) {
                decodes.push(decode);
            }
        }
        assert_eq!(decodes.len(), COMPONENTS);
    }

    #[test]
    fn the_component_table_is_the_registry_table() {
        assert_eq!(COMPONENT_OF.len(), registry::representations().len());
    }

    #[test]
    fn instance_features_are_the_tier_features() {
        #[cfg(target_arch = "x86_64")]
        {
            assert_eq!(x86v2::FEATURES, Tier::X86V2.features());
            assert_eq!(x86v3::FEATURES, Tier::X86V3.features());
            assert_eq!(x86v4::FEATURES, Tier::X86V4.features());
            assert_eq!(x86v4vnni::FEATURES, Tier::X86V4Vnni.features());
        }
        #[cfg(target_arch = "aarch64")]
        assert_eq!(neon::FEATURES, Tier::Neon.features());
    }

    /// Rows of the largest component block.
    const BLOCK_ROWS: usize = ROW_BLOCKS[ROW_BLOCKS.len() - 1];

    /// A deterministic pseudo-random byte stream.
    fn bytes(seed: u64, count: usize) -> Vec<u8> {
        let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
        (0..count)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state >> 24) as u8
            })
            .collect()
    }

    /// One line of a [`Placed`] buffer.
    #[derive(Clone, Copy)]
    #[repr(C, align(256))]
    struct Line([u8; crate::tensor::BUFFER_ALIGNMENT]);

    const _: () = assert!(std::mem::align_of::<Line>() == crate::tensor::BUFFER_ALIGNMENT);

    /// Values placed as a launch's buffers are (`tensor::BUFFER_ALIGNMENT`),
    /// so a measurement's vector accesses line up as they do in a launch,
    /// wherever the allocator puts the storage.
    pub(crate) struct Placed<T> {
        lines: Vec<Line>,
        len: usize,
        values: std::marker::PhantomData<T>,
    }

    impl<T: Copy> Placed<T> {
        /// `len` zero values.
        fn zeroed(len: usize) -> Self {
            let lines = (len * std::mem::size_of::<T>()).div_ceil(crate::tensor::BUFFER_ALIGNMENT);
            Self {
                lines: vec![Line([0; crate::tensor::BUFFER_ALIGNMENT]); lines],
                len,
                values: std::marker::PhantomData,
            }
        }

        /// A copy of `values`.
        fn of(values: &[T]) -> Self {
            let mut placed = Self::zeroed(values.len());
            placed.copy_from_slice(values);
            placed
        }
    }

    impl<'a, T: Copy> IntoIterator for &'a Placed<T> {
        type Item = &'a T;
        type IntoIter = std::slice::Iter<'a, T>;
        fn into_iter(self) -> Self::IntoIter {
            self.iter()
        }
    }

    impl<T: Copy> std::ops::Deref for Placed<T> {
        type Target = [T];
        fn deref(&self) -> &[T] {
            unsafe { std::slice::from_raw_parts(self.lines.as_ptr().cast(), self.len) }
        }
    }

    impl<T: Copy> std::ops::DerefMut for Placed<T> {
        fn deref_mut(&mut self) -> &mut [T] {
            unsafe { std::slice::from_raw_parts_mut(self.lines.as_mut_ptr().cast(), self.len) }
        }
    }

    /// The components of the element named `name` on `tier`.
    fn kernels_of(tier: Tier, name: &str) -> &'static WeightKernels {
        resolve(tier, registry::representation(name).unwrap()).unwrap()
    }

    /// The inlined dots of one component and row block.
    pub(crate) type InlinedFn = unsafe fn(*const u8, usize, &RowGeometry, &[f32], &mut [f32]);

    /// The row layout of packed rows `id`.
    fn packed_rows(id: RepresentationId) -> &'static registry::PackedRowLayout {
        match &registry::representation_info(id).kind {
            RepresentationKind::PackedRows(layout) => layout,
            _ => panic!("`{}` is no row layout", registry::representation_info(id).name),
        }
    }

    /// Every row-layout storage element has components (a device's layout
    /// too, `mma16`'s records included), and each, beside its rows16
    /// element's: the same packets read in either layout are the same
    /// values (decoded bit for bit) and the same products (quantized dots
    /// bit for bit), every row block and tier, at every column window (its
    /// storage groups from the first a multiple of its group on, and up to
    /// one).
    #[test]
    fn every_layout_reads_its_packets_as_rows16_does() {
        let tier = Tier::detected().unwrap();
        for info in registry::representations() {
            if matches!(info.kind, RepresentationKind::PackedRows(_)) && registry::storage_of(info.id) == info.id {
                assert!(resolve(tier, info.id).is_some(), "`{}` has no components", info.name);
            }
        }
        let mut compared = 0;
        for name in elements() {
            let id = registry::representation(name).unwrap();
            let info = registry::representation_info(id);
            if info.layout == Layout::Rows16 || !matches!(info.kind, RepresentationKind::PackedRows(_)) {
                continue;
            }
            let row_major = registry::storage(info.representation, Layout::Rows16).unwrap();
            let group = packed_rows(id).group() as usize;
            // Two tiles of rows32 and a partial mma16 tile; whole k-blocks,
            // and an odd count of them (K = 64 x odd).
            for (rows, k) in [(40, 1024usize), (19, 576)] {
                if k % group != 0 {
                    continue;
                }
                let (data, geometry) = self::rows(name, rows, k, 5);
                // The reference: the same packets placed in rows16.
                let extents = [rows as u64, k as u64];
                let packets = packed_rows(id).packets(&extents, &data);
                let reference = packed_rows(row_major).place(&extents, &packets);
                let reference_geometry = RowGeometry::of(registry::representation_info(row_major), k, 0);
                let x = bytes(9, k).iter().map(|byte| (*byte as f32 - 128.0) / 100.0).collect::<Vec<_>>();
                let mut q8 = vec![crate::quant::Q8Block::ZERO; crate::quant::blocks(k)];
                crate::quant::quantize(&x, &mut q8);
                for tier in tier.at_or_below() {
                    let (kernels, reference_kernels) = (resolve(tier, id).unwrap(), resolve(tier, row_major).unwrap());
                    for row in 0..rows {
                        let (mut values, mut expected) = (vec![0.0f32; k], vec![0.0f32; k]);
                        unsafe {
                            kernels.decode(data.as_ptr().add(row * geometry.stride), &geometry, k, &mut values);
                            reference_kernels.decode(
                                reference.as_ptr().add(row * reference_geometry.stride),
                                &reference_geometry,
                                k,
                                &mut expected,
                            );
                        }
                        let bits = |values: &[f32]| values.iter().map(|value| value.to_bits()).collect::<Vec<_>>();
                        assert_eq!(bits(&values), bits(&expected), "{name} {tier:?} k {k} row {row}");
                    }
                    for block in ROW_BLOCKS {
                        for first in (0..rows - block + 1).step_by(block) {
                            let (mut out, mut expected) = (vec![0.0f32; block], vec![0.0f32; block]);
                            unsafe {
                                kernels.dot_q8(data.as_ptr().add(first * geometry.stride), &geometry, &q8, k, &mut out);
                                reference_kernels.dot_q8(
                                    reference.as_ptr().add(first * reference_geometry.stride),
                                    &reference_geometry,
                                    &q8,
                                    k,
                                    &mut expected,
                                );
                            }
                            let bits = |values: &[f32]| values.iter().map(|value| value.to_bits()).collect::<Vec<_>>();
                            assert_eq!(bits(&out), bits(&expected), "{name} {tier:?} k {k} block {block} rows {first}..");
                        }
                    }
                }
                // Column windows, read where they lie, as the same columns of
                // the rows16 rows.
                for start in (0..k).step_by(group) {
                    let end = (start + 2 * group).min(k);
                    let mut window = geometry;
                    window.first_group += start / group;
                    let mut reference_window = reference_geometry;
                    reference_window.first_group += start / group;
                    let kernels = resolve(tier, id).unwrap();
                    let reference_kernels = resolve(tier, row_major).unwrap();
                    for row in [0, rows - 1] {
                        let (mut values, mut expected) = (vec![0.0f32; end - start], vec![0.0f32; end - start]);
                        unsafe {
                            kernels.decode(data.as_ptr().add(row * geometry.stride), &window, end - start, &mut values);
                            reference_kernels.decode(
                                reference.as_ptr().add(row * reference_geometry.stride),
                                &reference_window,
                                end - start,
                                &mut expected,
                            );
                        }
                        assert_eq!(values, expected, "{name} k {k} columns {start}..{end} row {row}");
                    }
                }
                compared += 1;
            }
        }
        assert!(compared >= 20, "{compared} layouts compared");
    }

    /// Rows of `representation` whose scales and factors are finite and
    /// small.
    fn rows(representation: &str, rows: usize, k: usize, seed: u64) -> (Placed<u8>, RowGeometry) {
        let id = registry::representation(representation).unwrap();
        let info = registry::representation_info(id);
        if info.layout != Layout::Rows16 && matches!(info.kind, RepresentationKind::PackedRows(_)) {
            // The rows16 rows' packets, placed in this layout (one matrix).
            let row_major = registry::storage(info.representation, Layout::Rows16).unwrap();
            let (source, _) = self::rows(registry::representation_info(row_major).name, rows, k, seed);
            let extents = [rows as u64, k as u64];
            let data = Placed::of(&packed_rows(id).place(&extents, &packed_rows(row_major).packets(&extents, &source)));
            let mut geometry = RowGeometry::of(info, k, 0);
            geometry.base = data.as_ptr() as usize;
            geometry.matrix_rows = rows;
            return (data, geometry);
        }
        let dense = match &info.kind {
            RepresentationKind::Dense(dtype) => dtype.bytes() as usize,
            _ => 0,
        };
        let geometry = RowGeometry::of(info, k, k * dense);
        let mut data = bytes(seed, rows * geometry.stride);
        let half = |value: f32| f32_to_f16(value).to_le_bytes();
        for row in 0..rows {
            let base = row * geometry.stride;
            match &info.kind {
                RepresentationKind::Dense(dtype) => {
                    for i in 0..k {
                        let at = base + dense * i;
                        let value = (data[at] as f32 - 128.0) / 64.0;
                        match dtype {
                            DType::F32 => data[at..at + 4].copy_from_slice(&value.to_le_bytes()),
                            DType::BF16 => {
                                data[at..at + 2].copy_from_slice(&f32_to_bf16(value).to_le_bytes())
                            }
                            _ => data[at..at + 2].copy_from_slice(&half(value)),
                        }
                    }
                }
                RepresentationKind::PackedRows(layout) => {
                    // The supers plane: the direct scales (and biases) or the
                    // factors, each a small finite value of its encoding.
                    let encoding = match registry::decode_axes(id).unwrap().coding {
                        CodingAxes::Direct { scale, .. } => scale,
                        CodingAxes::Hierarchical { factor, .. } => factor,
                    };
                    let supers = RowPlaneRole::Supers as usize;
                    let start = base + geometry.offsets[supers];
                    let bytes =
                        layout.row_groups(k as u64).unwrap() as usize * geometry.groups[supers];
                    let plane = &mut data[start..start + bytes];
                    crate::element::finite_scales(plane, encoding, false);
                }
                _ => unreachable!("a weight element is dense or packed rows"),
            }
        }
        (Placed::of(&data), geometry)
    }

    #[test]
    fn every_row_block_and_tier_gives_the_single_row_result() {
        let k = 2 * 256 + 32 * 3;
        let x = Placed::of(
            &bytes(7, k)
                .iter()
                .map(|byte| (*byte as f32 - 128.0) / 100.0)
                .collect::<Vec<_>>(),
        );
        for representation in elements() {
            let (data, geometry) = rows(representation, BLOCK_ROWS, k, 11);
            let reference = kernels_of(Tier::detected().unwrap(), representation);
            // The single-row dot against the decoded row, in the defined order.
            let mut expected = [0.0f32; BLOCK_ROWS];
            let mut decoded = vec![0.0f32; k];
            for (row, expected) in expected.iter_mut().enumerate() {
                unsafe {
                    reference.decode(
                        data.as_ptr().add(row * geometry.stride),
                        &geometry,
                        k,
                        &mut decoded,
                    )
                };
                *expected = crate::reduce::dot(&decoded, &x);
            }
            for tier in Tier::detected().unwrap().at_or_below() {
                let kernels = kernels_of(tier, representation);
                for block in ROW_BLOCKS {
                    for first in (0..BLOCK_ROWS).step_by(block) {
                        let mut out = vec![0.0f32; block];
                        unsafe {
                            kernels.dot(
                                data.as_ptr().add(first * geometry.stride),
                                &geometry,
                                &x,
                                &mut out,
                            )
                        };
                        for (r, value) in out.iter().enumerate() {
                            assert_eq!(
                                value.to_bits(),
                                expected[first + r].to_bits(),
                                "{representation} {tier:?} block {block} row {}",
                                first + r
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn rows8_weight_view_addresses_padded_matrices() {
        let k = 256;
        let tier = Tier::detected().unwrap();
        let representation = "q4k@rows8";
        let kernels = kernels_of(tier, representation);
        let (first, first_geometry) = rows(representation, 5, k, 41);
        let (second, second_geometry) = rows(representation, 5, k, 43);
        let bytes = Placed::of(&[&first[..], &second[..]].concat());
        let view = unsafe {
            crate::Weights::from_tensor(
                bytes.as_ptr(),
                &[2, 5, k as u64],
                &[8, 1, 1],
                crate::WeightOperand::of(tier, registry::representation(representation).unwrap())
                    .unwrap(),
            )
        };
        for row in 0..10 {
            let (source, geometry, local) = if row < 5 {
                (&first, &first_geometry, row)
            } else {
                (&second, &second_geometry, row - 5)
            };
            let mut expected = vec![0.0f32; k];
            let mut actual = vec![0.0f32; k];
            unsafe {
                kernels.decode(
                    source.as_ptr().add(local * geometry.stride),
                    geometry,
                    k,
                    &mut expected,
                )
            };
            view.decode_row(row, &mut actual);
            assert_eq!(
                actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "row {row}"
            );
        }
    }

    /// A quantized dot of `m` activation rows gives each row the sums it has
    /// alone, bit for bit, for every element, row block and `m` (each `M`
    /// grouping of `Format::dot_q8` the body takes).
    #[test]
    fn many_activation_rows_match_individual_dots() {
        let k = 2 * 256 + 32;
        let tier = Tier::detected().unwrap();
        const MOST: usize = 9;
        let mut activations = vec![0.0f32; MOST * k];
        for (i, value) in activations.iter_mut().enumerate() {
            *value = ((i * 37 % 251) as f32 - 125.0) / 113.0;
        }
        let blocks = crate::quant::blocks(k);
        let mut quantized = vec![Q8Block::ZERO; MOST * blocks];
        for m in 0..MOST {
            crate::quant::quantize(
                &activations[m * k..(m + 1) * k],
                &mut quantized[m * blocks..(m + 1) * blocks],
            );
        }
        for representation in elements() {
            let (data, geometry) = rows(representation, 16, k, 31);
            let kernels = kernels_of(tier, representation);
            for block in ROW_BLOCKS {
                let alone = (0..MOST)
                    .map(|m| {
                        let mut expected = vec![0.0f32; block];
                        unsafe {
                            kernels.dot_q8(
                                data.as_ptr(),
                                &geometry,
                                &quantized[m * blocks..(m + 1) * blocks],
                                k,
                                &mut expected,
                            )
                        };
                        expected
                    })
                    .collect::<Vec<_>>();
                for count in 1..=MOST {
                    let mut together = vec![0.0f32; count * block];
                    unsafe {
                        kernels.dot_q8(data.as_ptr(), &geometry, &quantized[..count * blocks], k, &mut together)
                    };
                    for m in 0..count {
                        for r in 0..block {
                            assert_eq!(
                                together[m * block + r].to_bits(),
                                alone[m][r].to_bits(),
                                "{representation} block {block} of {count} rows: m {m} r {r}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// The component throughput rule: a component call per row block across
    /// the whole row runs within a bound of the same loop with the body
    /// inlined, for every representation and row block, on the detected tier
    /// (weights cache-resident, so call overhead is most exposed). Measured
    /// interleaved, best of several, so host noise affects both alike.
    #[test]
    fn components_keep_the_throughput_of_the_inlined_body() {
        /// A component call may cost at most this much over the inlined body.
        /// Instances measure within a few percent of it; the bound catches a
        /// structural loss (a component that stops vectorizing, or keeps a
        /// per-element bounds check, measured at 5x) with room for the few
        /// instances the inlined loop schedules better across rows.
        const BOUND: f64 = 1.35;
        if cfg!(debug_assertions) {
            // Unoptimized code measures the compiler, not the component.
            return;
        }
        let tier = Tier::detected().unwrap();
        let (rows, k) = (64usize, 2048usize);
        let x = Placed::of(
            &bytes(5, k)
                .iter()
                .map(|byte| (*byte as f32 - 128.0) / 100.0)
                .collect::<Vec<_>>(),
        );
        for representation in elements() {
            let (data, geometry) = self::rows(representation, rows, k, 9);
            let kernels = kernels_of(tier, representation);
            for block in ROW_BLOCKS {
                let mut out = vec![0.0f32; rows];
                let inlined = |out: &mut [f32]| unsafe {
                    inline_for(
                        tier,
                        representation,
                        block,
                        data.as_ptr(),
                        rows,
                        &geometry,
                        &x,
                        out,
                    )
                };
                let component = |out: &mut [f32]| {
                    for first in (0..rows).step_by(block) {
                        unsafe {
                            kernels.dot(
                                data.as_ptr().add(first * geometry.stride),
                                &geometry,
                                &x,
                                &mut out[first..first + block],
                            )
                        };
                    }
                };
                let time = |run: &dyn Fn(&mut [f32]), out: &mut [f32]| {
                    let started = std::time::Instant::now();
                    for _ in 0..20 {
                        run(out);
                        std::hint::black_box(&*out);
                    }
                    started.elapsed().as_secs_f64()
                };
                let (mut best_inlined, mut best_component) = (f64::INFINITY, f64::INFINITY);
                for _ in 0..9 {
                    best_inlined = best_inlined.min(time(&inlined, &mut out));
                    best_component = best_component.min(time(&component, &mut out));
                }
                let ratio = best_component / best_inlined;
                eprintln!("{representation:>14} rows {block}: {ratio:.3}");
                assert!(
                    ratio <= BOUND,
                    "{representation} {tier:?} rows {block}: a component call runs {ratio:.2}x the inlined body"
                );
            }
        }
    }

    /// The inlined reference loop of `representation` and row block `block` on `tier`.
    #[allow(clippy::too_many_arguments)]
    unsafe fn inline_for(
        tier: Tier,
        representation: &str,
        block: usize,
        data: *const u8,
        rows: usize,
        geometry: &RowGeometry,
        x: &[f32],
        out: &mut [f32],
    ) {
        let table: &[[InlinedFn; ROW_BLOCKS.len()]; COMPONENTS] = match tier {
            #[cfg(target_arch = "x86_64")]
            Tier::X86V2 => &x86v2::INLINED,
            #[cfg(target_arch = "x86_64")]
            Tier::X86V3 => &x86v3::INLINED,
            #[cfg(target_arch = "x86_64")]
            Tier::X86V4 => &x86v4::INLINED,
            #[cfg(target_arch = "x86_64")]
            Tier::X86V4Vnni => &x86v4vnni::INLINED,
            #[cfg(target_arch = "aarch64")]
            Tier::Neon => &neon::INLINED,
            #[allow(unreachable_patterns)]
            other => panic!("tier {other:?} is not of this architecture"),
        };
        let inlined = table[component(registry::representation(representation).unwrap()).unwrap()]
            [row_block(block)];
        unsafe { inlined(data, rows, geometry, x, out) }
    }

    /// Aggregate streaming rate of the eight-row quantized dot over a matrix
    /// far larger than the caches, split across `CPU_DOT_THREADS` threads in
    /// contiguous ranges or, with `CPU_DOT_ITEMS=1`, in eight-row items
    /// claimed from a shared counter as the pool does.
    #[test]
    #[ignore = "measurement; run with --release --ignored --nocapture"]
    fn parallel_dot_streaming() {
        use crate::quant::{blocks, quantize, Q8Block};
        use std::sync::atomic::{AtomicUsize, Ordering};
        let tier = Tier::detected().unwrap();
        let threads = std::env::var("CPU_DOT_THREADS").map_or(10, |t| t.parse().unwrap());
        let items = std::env::var("CPU_DOT_ITEMS").is_ok_and(|v| v == "1");
        // Items per claim when claiming; 1 is the pool's current claim.
        let chunk: usize = std::env::var("CPU_DOT_CHUNK").map_or(1, |c| c.parse().unwrap());
        let (rows, k) = (98304usize, 4096usize);
        for representation in ["q4k@rows8", "q6k@rows8"] {
            let (data, geometry) = self::rows(representation, rows, k, 9);
            let kernels = kernels_of(tier, representation);
            let x = bytes(5, k)
                .iter()
                .map(|b| (*b as f32 - 128.0) / 100.0)
                .collect::<Vec<_>>();
            let mut xq = vec![Q8Block::ZERO; blocks(k)];
            quantize(&x, &mut xq);
            let mut best = f64::INFINITY;
            for _ in 0..5 {
                let next = AtomicUsize::new(0);
                let started = std::time::Instant::now();
                std::thread::scope(|scope| {
                    for t in 0..threads {
                        let (data, geometry, xq, next) = (&data, &geometry, &xq, &next);
                        scope.spawn(move || {
                            let mut out = [0.0f32; 8];
                            let mut run = |first: usize| unsafe {
                                kernels.dot_q8(
                                    data.as_ptr().add(first * geometry.stride),
                                    geometry,
                                    xq,
                                    k,
                                    &mut out,
                                );
                                std::hint::black_box(&out);
                            };
                            if items {
                                loop {
                                    let first = next.fetch_add(chunk, Ordering::Relaxed);
                                    if first * 8 >= rows {
                                        break;
                                    }
                                    for item in first..(first + chunk).min(rows / 8) {
                                        run(item * 8);
                                    }
                                }
                            } else {
                                let span = rows / threads / 8 * 8;
                                for first in (t * span..(t + 1) * span).step_by(8) {
                                    run(first);
                                }
                            }
                        });
                    }
                });
                best = best.min(started.elapsed().as_secs_f64());
            }
            eprintln!(
                "{representation} threads {threads} items {items}: {:.1} Gweights/s, {:.1} GB/s",
                (rows * k) as f64 / best / 1e9,
                (rows * geometry.stride) as f64 / best / 1e9
            );
        }
    }

    #[test]
    #[ignore = "measurement; run with --release --ignored --nocapture; CPU_DOT_REPRESENTATIONS and CPU_DOT_BLOCKS narrow it, CPU_DOT_ROWS sets the matrix rows (default 4096, which caches)"]
    fn dot_component_throughput() {
        let tier = Tier::detected().unwrap();
        let rows = std::env::var("CPU_DOT_ROWS").ok().map_or(4096, |rows| {
            rows.parse::<usize>().expect("CPU_DOT_ROWS is a row count")
        });
        let k = 4096usize;
        let x = Placed::of(
            &bytes(5, k)
                .iter()
                .map(|byte| (*byte as f32 - 128.0) / 100.0)
                .collect::<Vec<_>>(),
        );
        let selected = std::env::var("CPU_DOT_REPRESENTATIONS").ok();
        let selected_blocks = std::env::var("CPU_DOT_BLOCKS").ok();
        for representation in elements() {
            if selected
                .as_ref()
                .is_some_and(|names| !names.split(',').any(|name| name.trim() == representation))
            {
                continue;
            }
            let (data, geometry) = self::rows(representation, rows, k, 9);
            let kernels = kernels_of(tier, representation);
            for block in ROW_BLOCKS {
                if selected_blocks.as_ref().is_some_and(|blocks| {
                    !blocks
                        .split(',')
                        .any(|value| value.trim().parse::<usize>().ok() == Some(block))
                }) {
                    continue;
                }
                let mut out = vec![0.0f32; block];
                let mut best = f64::INFINITY;
                for _ in 0..5 {
                    let started = std::time::Instant::now();
                    for first in (0..rows).step_by(block) {
                        unsafe {
                            kernels.dot(
                                data.as_ptr().add(first * geometry.stride),
                                &geometry,
                                &x,
                                &mut out,
                            )
                        };
                        std::hint::black_box(&out);
                    }
                    best = best.min(started.elapsed().as_secs_f64());
                }
                eprintln!(
                    "{representation:>14} {tier:?} rows {block}: {:6.2} Gweights/s, {:6.2} GB/s of weights",
                    (rows * k) as f64 / best / 1e9,
                    (rows * geometry.stride) as f64 / best / 1e9
                );
                let mut xq = vec![crate::quant::Q8Block::ZERO; crate::quant::blocks(k)];
                crate::quant::quantize(&x, &mut xq);
                let mut best = f64::INFINITY;
                for _ in 0..5 {
                    let started = std::time::Instant::now();
                    for first in (0..rows).step_by(block) {
                        unsafe {
                            kernels.dot_q8(
                                data.as_ptr().add(first * geometry.stride),
                                &geometry,
                                &xq,
                                k,
                                &mut out,
                            )
                        };
                        std::hint::black_box(&out);
                    }
                    best = best.min(started.elapsed().as_secs_f64());
                }
                eprintln!(
                    "{representation:>14} {tier:?} rows {block} quantized: {:6.2} Gweights/s",
                    (rows * k) as f64 / best / 1e9
                );
                if block == 8 {
                    let four = xq.repeat(4);
                    let mut tile = [0.0f32; 32];
                    let mut best = f64::INFINITY;
                    for _ in 0..5 {
                        let started = std::time::Instant::now();
                        for first in (0..rows).step_by(8) {
                            unsafe {
                                kernels.dot_q8(
                                    data.as_ptr().add(first * geometry.stride),
                                    &geometry,
                                    &four,
                                    k,
                                    &mut tile,
                                )
                            };
                            std::hint::black_box(&tile);
                        }
                        best = best.min(started.elapsed().as_secs_f64());
                    }
                    eprintln!(
                        "{representation:>14} {tier:?} rows 8 quantized, four activation rows: {:6.2} Gweights/s",
                        (4 * rows * k) as f64 / best / 1e9
                    );
                }
            }
        }
    }

    /// Quantized dots are identical on every tier and row block, and differ
    /// from the exact dot of the decoded row by at most the activation
    /// rounding: half a quantization step per value, times |w|.
    #[test]
    fn quantized_dots_are_tier_invariant_and_within_the_activation_rounding() {
        use crate::quant::{blocks, quantize, Q8Block, Q8_BLOCK};
        let k = 2 * 256 + 32 * 3;
        let x = Placed::of(
            &bytes(7, k)
                .iter()
                .map(|byte| (*byte as f32 - 128.0) / 100.0)
                .collect::<Vec<_>>(),
        );
        let mut xq = vec![Q8Block::ZERO; blocks(k)];
        quantize(&x, &mut xq);
        let detected = Tier::detected().unwrap();
        for representation in elements() {
            let (data, geometry) = rows(representation, BLOCK_ROWS, k, 11);
            let reference = kernels_of(detected, representation);
            let mut first_tier = [0.0f32; BLOCK_ROWS];
            unsafe { reference.dot_q8(data.as_ptr(), &geometry, &xq, k, &mut first_tier) };
            let mut decoded = vec![0.0f32; k];
            for (row, value) in first_tier.iter().enumerate() {
                unsafe {
                    reference.decode(
                        data.as_ptr().add(row * geometry.stride),
                        &geometry,
                        k,
                        &mut decoded,
                    )
                };
                let exact: f64 = decoded
                    .iter()
                    .zip(&x)
                    .map(|(w, x)| f64::from(*w) * f64::from(*x))
                    .sum();
                let bound: f64 = decoded
                    .iter()
                    .enumerate()
                    .map(|(i, w)| f64::from(w.abs()) * f64::from(xq[i / Q8_BLOCK].d) * 0.5)
                    .sum::<f64>()
                    + 1e-4
                        * decoded
                            .iter()
                            .zip(&x)
                            .map(|(w, x)| f64::from((w * x).abs()))
                            .sum::<f64>();
                assert!(
                    (f64::from(*value) - exact).abs() <= bound,
                    "{representation} row {row}: {value} against {exact} (bound {bound})"
                );
            }
            for tier in detected.at_or_below() {
                let kernels = kernels_of(tier, representation);
                for block in ROW_BLOCKS {
                    for first in (0..BLOCK_ROWS).step_by(block) {
                        let mut out = vec![0.0f32; block];
                        unsafe {
                            kernels.dot_q8(
                                data.as_ptr().add(first * geometry.stride),
                                &geometry,
                                &xq,
                                k,
                                &mut out,
                            )
                        };
                        for (r, value) in out.iter().enumerate() {
                            assert_eq!(
                                value.to_bits(),
                                first_tier[first + r].to_bits(),
                                "{representation} {tier:?} block {block}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// Whole 16-row tiles in one call (`dot_q8_tiles`) give the sums the
    /// same tier's quantized dots give eight rows at a time (the rows16
    /// projection's block), bit for bit, at the expert
    /// shapes of a mixture layer (k 2048 and 512) and with a partial block,
    /// on every tier where an element has a whole-tile form; on x86 with
    /// quantized dots `q4k` and `q5k` have one in `mma16` and `rows16`.
    #[test]
    fn whole_tiles_match_the_row_block_dots() {
        use crate::quant::{blocks, quantize, Q8Block};
        let detected = Tier::detected().unwrap();
        let mut whole = std::collections::BTreeSet::new();
        for (k, tiles) in [(2048, 4), (512, 4), (2 * 256 + 128, 3)] {
            let x = bytes(5, k).iter().map(|byte| (*byte as f32 - 128.0) / 90.0).collect::<Vec<_>>();
            let mut xq = vec![Q8Block::ZERO; blocks(k)];
            quantize(&x, &mut xq);
            for representation in elements() {
                let (data, geometry) = rows(representation, 16 * tiles, k, 13);
                // Eight rows a call: never a whole tile.
                let block = 8;
                for tier in detected.at_or_below() {
                    let kernels = kernels_of(tier, representation);
                    let mut out = vec![0.0f32; 16 * tiles];
                    if !unsafe { kernels.dot_q8_tiles(data.as_ptr(), &geometry, &xq, k, &mut out) } {
                        continue;
                    }
                    whole.insert(representation);
                    let mut expected = vec![0.0f32; 16 * tiles];
                    for (index, expected) in expected.chunks_exact_mut(block).enumerate() {
                        unsafe { kernels.dot_q8(data.as_ptr().add(block * index * geometry.stride), &geometry, &xq, k, expected) };
                    }
                    for (row, (value, expected)) in out.iter().zip(&expected).enumerate() {
                        assert_eq!(value.to_bits(), expected.to_bits(), "{representation} {tier:?} k {k} row {row}");
                    }
                }
            }
        }
        if cfg!(target_arch = "x86_64") && detected.quantized_dots() >= 1 {
            for representation in ["q4k@mma16", "q5k@mma16", "q4k@rows16", "q5k@rows16"] {
                assert!(whole.contains(representation), "{representation} has no whole-tile form");
            }
        }
    }

    /// Every tier's quantized row dots and four-row tile reproduce the
    /// lowest tier's scalar integer recipe bit for bit, for every
    /// representation and row block, across reductions with partial packets
    /// and partial blocks, and activations with saturated and zero codes.
    #[test]
    fn quantized_components_match_the_lowest_tier_bit_for_bit() {
        use crate::quant::{blocks, quantize};
        let lowest = Tier::all()[0];
        let detected = Tier::detected().unwrap();
        for k in [32usize, 96, 256, 2 * 256 + 96, 704, 2688] {
            let blocks = blocks(k);
            let mut quantized = vec![Q8Block::ZERO; 4 * blocks];
            for m in 0..4 {
                let mut x = bytes(100 + (m + k) as u64, k)
                    .iter()
                    .map(|byte| (*byte as f32 - 128.0) / 100.0)
                    .collect::<Vec<_>>();
                if m == 3 {
                    for value in x.iter_mut().step_by(7) {
                        *value = 0.0;
                    }
                    x[0] = 50.0;
                }
                quantize(&x, &mut quantized[m * blocks..(m + 1) * blocks]);
            }
            for representation in elements() {
                let (data, geometry) = rows(representation, BLOCK_ROWS, k, 17 + k as u64);
                let reference = kernels_of(lowest, representation);
                let mut expected = [[0.0f32; BLOCK_ROWS]; 4];
                for (m, expected) in expected.iter_mut().enumerate() {
                    let x = &quantized[m * blocks..(m + 1) * blocks];
                    unsafe { reference.dot_q8(data.as_ptr(), &geometry, x, k, expected) };
                }
                for tier in detected.at_or_below() {
                    let kernels = kernels_of(tier, representation);
                    for (m, expected) in expected.iter().enumerate() {
                        let x = &quantized[m * blocks..(m + 1) * blocks];
                        for block in ROW_BLOCKS {
                            for first in (0..BLOCK_ROWS).step_by(block) {
                                let mut out = vec![0.0f32; block];
                                let rows = unsafe { data.as_ptr().add(first * geometry.stride) };
                                unsafe { kernels.dot_q8(rows, &geometry, x, k, &mut out) };
                                for (r, value) in out.iter().enumerate() {
                                    assert_eq!(
                                        value.to_bits(),
                                        expected[first + r].to_bits(),
                                        "{representation} {tier:?} k {k} m {m} block {block} row {}",
                                        first + r
                                    );
                                }
                            }
                        }
                    }
                    // The four activation rows at once, each block's.
                    for block in ROW_BLOCKS {
                        let mut together = vec![0.0f32; 4 * block];
                        unsafe { kernels.dot_q8(data.as_ptr(), &geometry, &quantized, k, &mut together) };
                        for (m, expected) in expected.iter().enumerate() {
                            for (r, expected) in expected.iter().enumerate().take(block) {
                                assert_eq!(
                                    together[m * block + r].to_bits(),
                                    expected.to_bits(),
                                    "{representation} {tier:?} k {k} four rows, block {block}: m {m} row {r}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn dense_rows_with_a_partial_packet_decode_their_values() {
        let k = 45;
        let (data, geometry) = rows("bf16", 1, k, 3);
        let kernels = kernels_of(Tier::detected().unwrap(), "bf16");
        let mut out = vec![f32::NAN; k];
        unsafe { kernels.decode(data.as_ptr(), &geometry, k, &mut out) };
        for (i, value) in out.iter().enumerate() {
            let bits = u16::from_le_bytes([data[2 * i], data[2 * i + 1]]);
            assert_eq!(value.to_bits(), crate::element::bf16_to_f32(bits).to_bits());
        }
    }
}
