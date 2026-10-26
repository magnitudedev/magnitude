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
    self, codes, scales, Carried, DenseRows, Direct, Format, Grid, Hier, PackedRows, Planes,
    RowGeometry, Scales,
};
use seismic_lang::ids::RepresentationId;
#[cfg(test)]
use seismic_lang::registry;

include!(concat!(env!("OUT_DIR"), "/elements.rs"));

/// Row blocks of the dot component, in the order of [`WeightKernels::dot`].
pub const ROW_BLOCKS: [usize; 4] = [1, 2, 4, 8];

type DecodeFn = unsafe fn(*const u8, &RowGeometry, usize, *mut f32);
type DotFn = unsafe fn(*const u8, &RowGeometry, *const f32, usize, *mut f32);
type DotQ8Fn = unsafe fn(*const u8, &RowGeometry, *const Q8Block, usize, *mut f32);
type GemmQ8Fn = unsafe fn(*const u8, &RowGeometry, *const Q8Block, usize, *mut f32);

/// The instances of one component on one tier.
#[derive(Clone, Copy)]
pub struct WeightKernels {
    pub tier: Tier,
    decode: DecodeFn,
    /// One instance per entry of [`ROW_BLOCKS`].
    dot: [DotFn; 4],
    /// The same against quantized activations.
    dot_q8: [DotQ8Fn; 4],
    /// Four activation rows by eight weight rows, all reduced over K.
    gemm_q8: GemmQ8Fn,
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

    /// `out[r] = row r · x` for the `out.len()` rows from `rows` (1, 2, 4 or
    /// 8), `geometry.stride` bytes apart.
    ///
    /// # Safety
    /// `rows` addresses that many rows with `geometry` and `x.len()` values.
    #[inline(always)]
    pub unsafe fn dot(&self, rows: *const u8, geometry: &RowGeometry, x: &[f32], out: &mut [f32]) {
        let block = row_block(out.len());
        unsafe { (self.dot[block])(rows, geometry, x.as_ptr(), x.len(), out.as_mut_ptr()) }
    }

    /// As [`WeightKernels::dot`], against the activations of a `k`-value row
    /// quantized into `x` (`quant::blocks(k)` blocks).
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
        assert_eq!(
            x.len(),
            crate::quant::blocks(k),
            "quantized activation blocks of a {k}-value row"
        );
        let block = row_block(out.len());
        unsafe { (self.dot_q8[block])(rows, geometry, x.as_ptr(), k, out.as_mut_ptr()) }
    }

    /// Four quantized activation rows by eight weight rows. Output is
    /// activation-major: four consecutive blocks of eight projections.
    ///
    /// # Safety
    /// `rows` addresses eight rows with `geometry` and `k` values.
    pub unsafe fn gemm_q8(
        &self,
        rows: *const u8,
        geometry: &RowGeometry,
        x: &[Q8Block],
        k: usize,
        out: &mut [f32; 32],
    ) {
        assert_eq!(x.len(), 4 * crate::quant::blocks(k));
        unsafe { (self.gemm_q8)(rows, geometry, x.as_ptr(), k, out.as_mut_ptr()) }
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
                unsafe {
                    weights::decode_row::<W, { Tier::$tier.quantized_dots() }>(
                        row, geometry, k, out,
                    )
                }
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
                        weights::dot_rows::<W, R, { Tier::$tier.quantized_dots() }>(
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
            pub(crate) static INLINED: [[super::tests::InlinedFn; 4]; COMPONENTS] =
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
                let result = unsafe {
                    weights::dot_rows::<W, R, { Tier::$tier.quantized_dots() }>(
                        rows, geometry, x, k,
                    )
                };
                unsafe { std::ptr::copy_nonoverlapping(result.as_ptr(), out, R) };
            }

            #[target_feature(enable = $features)]
            unsafe fn dot_q8<W: Format, const R: usize>(
                rows: *const u8,
                geometry: &RowGeometry,
                x: *const Q8Block,
                k: usize,
                out: *mut f32,
            ) {
                let x = unsafe { std::slice::from_raw_parts(x, crate::quant::blocks(k)) };
                let result = unsafe {
                    weights::dot_q8_rows::<W, R, { Tier::$tier.quantized_dots() }>(
                        rows, geometry, x, k,
                    )
                };
                unsafe { std::ptr::copy_nonoverlapping(result.as_ptr(), out, R) };
            }

            /// A whole K reduction for a 4×8 tile: the same weight block
            /// serves four activation rows.
            #[target_feature(enable = $features)]
            unsafe fn gemm_q8<W: Format>(
                rows: *const u8,
                geometry: &RowGeometry,
                x: *const Q8Block,
                k: usize,
                out: *mut f32,
            ) {
                let x = unsafe { std::slice::from_raw_parts(x, 4 * crate::quant::blocks(k)) };
                let accumulators = unsafe {
                    weights::gemm_q8_rows::<W, { Tier::$tier.quantized_dots() }>(
                        rows, geometry, x, k,
                    )
                };
                unsafe {
                    std::ptr::copy_nonoverlapping(accumulators.as_ptr().cast::<f32>(), out, 32)
                };
            }

            const fn kernels<W: Format>() -> WeightKernels {
                WeightKernels {
                    tier: Tier::$tier,
                    decode: decode::<W>,
                    dot: [dot::<W, 1>, dot::<W, 2>, dot::<W, 4>, dot::<W, 8>],
                    dot_q8: [
                        dot_q8::<W, 1>,
                        dot_q8::<W, 2>,
                        dot_q8::<W, 4>,
                        dot_q8::<W, 8>,
                    ],
                    gemm_q8: gemm_q8::<W>,
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
    COMPONENT_OF[element.index()]
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

/// Compiled component instances in this build: decode, dot and quantized dot
/// instances of every component on every tier of the architecture.
pub fn instances() -> usize {
    TABLES.len() * COMPONENTS * (2 + 2 * ROW_BLOCKS.len())
}

/// The recorded ceiling on compiled component instances. Raising it is a
/// reviewed change with a stated reason (see the CPU authoring design).
pub const INSTANCE_CEILING: usize = 1_000;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::element::{f32_to_bf16, f32_to_f16};
    use seismic_lang::registry::{
        self, CoefficientAxes, Layout, RepresentationKind, RowPlaneRole,
    };
    use seismic_lang::types::DType;

    #[test]
    fn instance_count_stays_under_the_ceiling() {
        assert!(
            instances() <= INSTANCE_CEILING,
            "{} component instances",
            instances()
        );
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

    /// The components of the element named `name` on `tier`.
    fn kernels_of(tier: Tier, name: &str) -> &'static WeightKernels {
        resolve(tier, registry::representation(name).unwrap()).unwrap()
    }

    /// The inlined dots of one component and row block.
    pub(crate) type InlinedFn = unsafe fn(*const u8, usize, &RowGeometry, &[f32], &mut [f32]);

    /// Rows of `representation` whose scales and factors are finite and
    /// small.
    fn rows(representation: &str, rows: usize, k: usize, seed: u64) -> (Vec<u8>, RowGeometry) {
        let id = registry::representation(representation).unwrap();
        let info = registry::representation_info(id);
        if info.layout == Layout::Rows8 {
            // The rows16 rows, each plane's storage groups interleaved by tile.
            let row_major = registry::storage(info.representation, Layout::Rows16).unwrap();
            let (source, source_geometry) =
                self::rows(registry::representation_info(row_major).name, rows, k, seed);
            let mut geometry = RowGeometry::of(info, k, 0);
            let mut data = vec![0u8; rows.div_ceil(8) * 8 * geometry.stride];
            let RepresentationKind::PackedRows(layout) = &info.kind else {
                unreachable!("a rows8 element is packed rows")
            };
            let groups = layout.row_groups(k as u64).unwrap() as usize;
            for row in 0..rows {
                let (tile, lane) = (row / 8, row % 8);
                for (offset, bytes) in geometry.offsets.iter().zip(geometry.groups) {
                    if bytes == 0 {
                        continue;
                    }
                    for group in 0..groups {
                        let from = row * geometry.stride + offset + group * bytes;
                        let to = tile * 8 * geometry.stride
                            + offset * 8
                            + group * bytes * 8
                            + lane * bytes;
                        data[to..to + bytes].copy_from_slice(&source[from..from + bytes]);
                    }
                }
            }
            assert_eq!(source_geometry.offsets, geometry.offsets);
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
                    let encoding = match registry::decode_axes(id).unwrap().coefficients {
                        CoefficientAxes::Direct { scale, .. } => scale,
                        CoefficientAxes::Hierarchical { factor, .. } => factor,
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
        (data, geometry)
    }

    #[test]
    fn every_row_block_and_tier_gives_the_single_row_result() {
        let k = 2 * 256 + 32 * 3;
        let x = bytes(7, k)
            .iter()
            .map(|byte| (*byte as f32 - 128.0) / 100.0)
            .collect::<Vec<_>>();
        for representation in elements() {
            let (data, geometry) = rows(representation, 8, k, 11);
            let reference = kernels_of(Tier::detected().unwrap(), representation);
            // The single-row dot against the decoded row, in the defined order.
            let mut expected = [0.0f32; 8];
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
                    for first in (0..8).step_by(block) {
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
        let mut bytes = first.clone();
        bytes.extend_from_slice(&second);
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

    #[test]
    fn four_by_eight_q8_tile_matches_individual_dots() {
        let k = 2 * 256 + 32;
        let tier = Tier::detected().unwrap();
        let mut activations = vec![0.0f32; 4 * k];
        for (i, value) in activations.iter_mut().enumerate() {
            *value = ((i * 37 % 251) as f32 - 125.0) / 113.0;
        }
        let blocks = crate::quant::blocks(k);
        let mut quantized = vec![Q8Block::ZERO; 4 * blocks];
        for m in 0..4 {
            crate::quant::quantize(
                &activations[m * k..(m + 1) * k],
                &mut quantized[m * blocks..(m + 1) * blocks],
            );
        }
        for representation in elements() {
            let (data, geometry) = rows(representation, 8, k, 31);
            let kernels = kernels_of(tier, representation);
            let mut tile = [0.0f32; 32];
            unsafe { kernels.gemm_q8(data.as_ptr(), &geometry, &quantized, k, &mut tile) };
            for m in 0..4 {
                let mut expected = [0.0f32; 8];
                unsafe {
                    kernels.dot_q8(
                        data.as_ptr(),
                        &geometry,
                        &quantized[m * blocks..(m + 1) * blocks],
                        k,
                        &mut expected,
                    )
                };
                for r in 0..8 {
                    assert_eq!(
                        tile[m * 8 + r].to_bits(),
                        expected[r].to_bits(),
                        "{representation} m {m} r {r}"
                    );
                }
            }
        }
    }

    #[test]
    #[ignore = "measurement; run in release mode to compare the four-row tile with separate row dots"]
    fn four_by_eight_q8_tile_throughput() {
        let tier = Tier::detected().unwrap();
        let k = 4096;
        let blocks = crate::quant::blocks(k);
        let activations = bytes(37, 4 * k)
            .iter()
            .map(|byte| (*byte as f32 - 128.0) / 113.0)
            .collect::<Vec<_>>();
        let mut quantized = vec![Q8Block::ZERO; 4 * blocks];
        for m in 0..4 {
            crate::quant::quantize(
                &activations[m * k..(m + 1) * k],
                &mut quantized[m * blocks..(m + 1) * blocks],
            );
        }
        for representation in ["q4k@rows8", "q5k@rows8", "q6k@rows8"] {
            let (data, geometry) = rows(representation, 8, k, 31);
            let kernels = kernels_of(tier, representation);
            let mut tile = [0.0f32; 32];
            let measure = |f: &dyn Fn(&mut [f32; 32]), tile: &mut [f32; 32]| {
                (0..5)
                    .map(|_| {
                        let began = std::time::Instant::now();
                        for _ in 0..1000 {
                            f(std::hint::black_box(tile));
                        }
                        began.elapsed().as_secs_f64()
                    })
                    .fold(f64::INFINITY, f64::min)
            };
            let separate = measure(
                &|tile| {
                    for m in 0..4 {
                        unsafe {
                            kernels.dot_q8(
                                data.as_ptr(),
                                &geometry,
                                &quantized[m * blocks..(m + 1) * blocks],
                                k,
                                &mut tile[m * 8..(m + 1) * 8],
                            )
                        }
                    }
                },
                &mut tile,
            );
            let fused = measure(
                &|tile| unsafe { kernels.gemm_q8(data.as_ptr(), &geometry, &quantized, k, tile) },
                &mut tile,
            );
            eprintln!(
                "{representation}: separate {separate:.3} s, fused {fused:.3} s, speedup {:.2}x",
                separate / fused
            );
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
        let x = bytes(5, k)
            .iter()
            .map(|byte| (*byte as f32 - 128.0) / 100.0)
            .collect::<Vec<_>>();
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
        let table: &[[InlinedFn; 4]; COMPONENTS] = match tier {
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
        let x = bytes(5, k)
            .iter()
            .map(|byte| (*byte as f32 - 128.0) / 100.0)
            .collect::<Vec<_>>();
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
                                kernels.gemm_q8(
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
                        "{representation:>14} {tier:?} rows 8 quantized 4-row tile: {:6.2} Gweights/s",
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
        let x = bytes(7, k)
            .iter()
            .map(|byte| (*byte as f32 - 128.0) / 100.0)
            .collect::<Vec<_>>();
        let mut xq = vec![Q8Block::ZERO; blocks(k)];
        quantize(&x, &mut xq);
        let detected = Tier::detected().unwrap();
        for representation in elements() {
            let (data, geometry) = rows(representation, 8, k, 11);
            let reference = kernels_of(detected, representation);
            let mut first_tier = [0.0f32; 8];
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
                    for first in (0..8).step_by(block) {
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
                let (data, geometry) = rows(representation, 8, k, 17 + k as u64);
                let reference = kernels_of(lowest, representation);
                let mut expected = [[0.0f32; 8]; 4];
                for (m, expected) in expected.iter_mut().enumerate() {
                    let x = &quantized[m * blocks..(m + 1) * blocks];
                    unsafe { reference.dot_q8(data.as_ptr(), &geometry, x, k, expected) };
                }
                for tier in detected.at_or_below() {
                    let kernels = kernels_of(tier, representation);
                    for (m, expected) in expected.iter().enumerate() {
                        let x = &quantized[m * blocks..(m + 1) * blocks];
                        for block in ROW_BLOCKS {
                            for first in (0..8).step_by(block) {
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
                    let mut tile = [0.0f32; 32];
                    unsafe { kernels.gemm_q8(data.as_ptr(), &geometry, &quantized, k, &mut tile) };
                    for (m, expected) in expected.iter().enumerate() {
                        for (r, expected) in expected.iter().enumerate() {
                            assert_eq!(
                                tile[m * 8 + r].to_bits(),
                                expected.to_bits(),
                                "{representation} {tier:?} k {k} tile m {m} row {r}"
                            );
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
