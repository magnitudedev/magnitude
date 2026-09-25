//! Component instances: the representation-specific inner loops, compiled
//! once for the program per tier × weight representation × row block, and
//! resolved into typed handles when a kernel is prepared.
//!
//! Each instance is a `#[target_feature]` function of its tier wrapping a
//! generic body of `weights`; the body inlines into it and compiles with the
//! tier's features. A call covers a whole row block across the reduction
//! dimension, so the one indirect call per block is amortized.

use crate::isa::Tier;
use crate::quant::Q8Block;
use crate::weights::{self, DenseRows, Format, Iq4, RowGeometry, Rows8, Q4K, Q5K, Q6K, Q8};

/// Weight representations with components.
const REPRESENTATIONS: usize = 13;

/// Row blocks of the dot component, in the order of [`WeightKernels::dot`].
pub const ROW_BLOCKS: [usize; 4] = [1, 2, 4, 8];

type DecodeFn = unsafe fn(*const u8, &RowGeometry, usize, *mut f32);
type DotFn = unsafe fn(*const u8, &RowGeometry, *const f32, usize, *mut f32);
type DotQ8Fn = unsafe fn(*const u8, &RowGeometry, *const Q8Block, usize, *mut f32);
type GemmQ8Fn = unsafe fn(*const u8, &RowGeometry, *const Q8Block, usize, *mut f32);
type GeometryFn = fn(usize, usize) -> RowGeometry;

/// The components of one weight representation on one tier.
#[derive(Clone, Copy)]
pub struct WeightKernels {
    pub representation: &'static str,
    pub tier: Tier,
    /// Bytes of one element of a dense representation; 0 for a packed one.
    pub dense_bytes: usize,
    geometry: GeometryFn,
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
            .field("representation", &self.representation)
            .field("tier", &self.tier)
            .finish()
    }
}

impl WeightKernels {
    /// The row geometry of `k`-value rows; `dense_stride` is the bound
    /// tensor's row stride in bytes.
    pub fn geometry(&self, k: usize, dense_stride: usize) -> RowGeometry {
        (self.geometry)(k, dense_stride)
    }

    /// Decodes the `k` values of the row at `row` into `out`.
    ///
    /// # Safety
    /// `row` addresses a row of this representation with `geometry`.
    #[inline(always)]
    pub unsafe fn decode(&self, row: *const u8, geometry: &RowGeometry, k: usize, out: &mut [f32]) {
        assert!(out.len() >= k);
        unsafe { (self.decode)(row, geometry, k, out.as_mut_ptr()) }
    }

    /// `out[r] = row r · x` for the `out.len()` rows from `rows` (1, 2, 4 or
    /// 8), `geometry.stride` bytes apart.
    ///
    /// # Safety
    /// `rows` addresses that many rows of this representation with
    /// `geometry` and `x.len()` values.
    #[inline(always)]
    pub unsafe fn dot(&self, rows: *const u8, geometry: &RowGeometry, x: &[f32], out: &mut [f32]) {
        let block = ROW_BLOCKS
            .iter()
            .position(|block| *block == out.len())
            .unwrap_or_else(|| {
                panic!(
                    "a dot row block has {} rows; blocks are {ROW_BLOCKS:?}",
                    out.len()
                )
            });
        unsafe { (self.dot[block])(rows, geometry, x.as_ptr(), x.len(), out.as_mut_ptr()) }
    }

    /// As [`WeightKernels::dot`], against the activations of a `k`-value row
    /// quantized into `x` (`quant::blocks(k)` blocks).
    ///
    /// # Safety
    /// `rows` addresses that many rows of this representation with
    /// `geometry` and `k` values.
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
        let block = ROW_BLOCKS
            .iter()
            .position(|block| *block == out.len())
            .unwrap_or_else(|| {
                panic!(
                    "a dot row block has {} rows; blocks are {ROW_BLOCKS:?}",
                    out.len()
                )
            });
        unsafe { (self.dot_q8[block])(rows, geometry, x.as_ptr(), k, out.as_mut_ptr()) }
    }

    /// Four quantized activation rows by eight weight rows. Output is
    /// activation-major: four consecutive blocks of eight projections.
    ///
    /// # Safety
    /// `rows` addresses eight rows of this representation.
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

macro_rules! tier_components {
    ($module:ident, $tier:ident, $features:literal, $mode:expr) => {
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
                unsafe { weights::decode_row::<W>(row, geometry, k, out) }
            }

            /// Every block of `R` rows of `rows` rows dotted with `x`, the
            /// dot body inlined: the reference a component call is measured
            /// against.
            #[cfg(test)]
            #[target_feature(enable = $features)]
            pub(crate) unsafe fn inlined<W: Format, const R: usize>(
                data: *const u8,
                rows: usize,
                geometry: &RowGeometry,
                x: &[f32],
                out: &mut [f32],
            ) {
                for first in (0..rows).step_by(R) {
                    let result = unsafe {
                        weights::dot_rows::<W, R>(
                            data.add(first * geometry.stride),
                            geometry,
                            x,
                            x.len(),
                        )
                    };
                    out[first..first + R].copy_from_slice(&result);
                }
            }

            #[target_feature(enable = $features)]
            unsafe fn dot<W: Format, const R: usize>(
                rows: *const u8,
                geometry: &RowGeometry,
                x: *const f32,
                k: usize,
                out: *mut f32,
            ) {
                let x = unsafe { std::slice::from_raw_parts(x, k) };
                let result = unsafe { weights::dot_rows::<W, R>(rows, geometry, x, k) };
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
                let result = unsafe { weights::dot_q8_rows::<W, R, $mode>(rows, geometry, x, k) };
                unsafe { std::ptr::copy_nonoverlapping(result.as_ptr(), out, R) };
            }

            /// A whole K reduction for a 4×8 tile. Thirty-two accumulators
            /// stay local while the same weight block serves four inputs.
            #[target_feature(enable = $features)]
            unsafe fn gemm_q8<W: Format>(
                rows: *const u8,
                geometry: &RowGeometry,
                x: *const Q8Block,
                k: usize,
                out: *mut f32,
            ) {
                let blocks = crate::quant::blocks(k);
                let x = unsafe { std::slice::from_raw_parts(x, 4 * blocks) };
                let mut accumulators = [[0.0f32; 8]; 4];
                let resolved = std::array::from_fn::<_, 8, _>(|r| {
                    geometry.for_row(unsafe { rows.add(r * geometry.stride) })
                });
                for b in 0..blocks {
                    let activations = std::array::from_fn(|m| &x[m * blocks + b]);
                    for r in 0..8 {
                        let row = unsafe { rows.add(r * geometry.stride) };
                        let products = unsafe {
                            W::dot_q8_four::<$mode>(row, &resolved[r], b, k, activations)
                        };
                        for m in 0..4 {
                            accumulators[m][r] += products[m];
                        }
                    }
                }
                unsafe {
                    std::ptr::copy_nonoverlapping(accumulators.as_ptr().cast::<f32>(), out, 32)
                };
            }

            const fn kernels<W: Format>() -> WeightKernels {
                WeightKernels {
                    representation: W::NAME,
                    tier: Tier::$tier,
                    dense_bytes: W::DENSE_BYTES,
                    geometry: W::geometry,
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

            pub(crate) static KERNELS: [WeightKernels; REPRESENTATIONS] = [
                kernels::<DenseRows<crate::element::F32>>(),
                kernels::<DenseRows<crate::element::Bf16>>(),
                kernels::<DenseRows<crate::element::F16>>(),
                kernels::<Q4K>(),
                kernels::<Q5K>(),
                kernels::<Q6K>(),
                kernels::<Q8>(),
                kernels::<Iq4>(),
                kernels::<Rows8<Q4K>>(),
                kernels::<Rows8<Q5K>>(),
                kernels::<Rows8<Q6K>>(),
                kernels::<Rows8<Q8>>(),
                kernels::<Rows8<Iq4>>(),
            ];
        }
    };
}

#[cfg(target_arch = "x86_64")]
tier_components!(x86v2, X86V2, "sse3,ssse3,sse4.1,sse4.2,popcnt", 0);
#[cfg(target_arch = "x86_64")]
tier_components!(
    x86v3,
    X86V3,
    "sse3,ssse3,sse4.1,sse4.2,popcnt,avx,avx2,fma,f16c,bmi1,bmi2,lzcnt,movbe",
    1
);
#[cfg(target_arch = "x86_64")]
tier_components!(
    x86v4,
    X86V4,
    "sse3,ssse3,sse4.1,sse4.2,popcnt,avx,avx2,fma,f16c,bmi1,bmi2,lzcnt,movbe,avx512f,avx512cd,avx512bw,avx512dq,avx512vl",
    1
);
#[cfg(target_arch = "x86_64")]
tier_components!(
    x86v4vnni,
    X86V4Vnni,
    "sse3,ssse3,sse4.1,sse4.2,popcnt,avx,avx2,fma,f16c,bmi1,bmi2,lzcnt,movbe,avx512f,avx512cd,avx512bw,avx512dq,avx512vl,avx512vnni",
    2
);
#[cfg(target_arch = "aarch64")]
tier_components!(neon, Neon, "neon", 0);

/// Every tier's component table of this architecture.
#[cfg(target_arch = "x86_64")]
static TABLES: [&[WeightKernels; REPRESENTATIONS]; 4] = [
    &x86v2::KERNELS,
    &x86v3::KERNELS,
    &x86v4::KERNELS,
    &x86v4vnni::KERNELS,
];
#[cfg(target_arch = "aarch64")]
static TABLES: [&[WeightKernels; REPRESENTATIONS]; 1] = [&neon::KERNELS];

fn tables() -> &'static [&'static [WeightKernels; REPRESENTATIONS]] {
    &TABLES
}

/// Every weight representation a CPU weight operand can be bound to: the
/// set every native backend's weight slots bind.
pub fn representations() -> impl Iterator<Item = &'static str> {
    tables()[0].iter().map(|kernels| kernels.representation)
}

/// The components of `representation` on `tier`, or `None` when the
/// representation is not a weight representation.
pub fn resolve(tier: Tier, representation: &str) -> Option<&'static WeightKernels> {
    tables()
        .iter()
        .flat_map(|table| table.iter())
        .find(|kernels| kernels.tier == tier && kernels.representation == representation)
}

/// Compiled component instances in this build: decode, dot and quantized dot
/// instances of every representation on every tier of the architecture.
pub fn instances() -> usize {
    tables()
        .iter()
        .map(|table| table.len() * (2 + 2 * ROW_BLOCKS.len()))
        .sum()
}

/// The recorded ceiling on compiled component instances. Raising it is a
/// reviewed change with a stated reason (see the CPU authoring design).
pub const INSTANCE_CEILING: usize = 1_000;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::element::{f32_to_bf16, f32_to_f16};

    #[test]
    fn instance_count_stays_under_the_ceiling() {
        assert!(
            instances() <= INSTANCE_CEILING,
            "{} component instances",
            instances()
        );
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

    /// Rows of `representation` whose f16 factors are finite and small.
    fn rows(representation: &str, rows: usize, k: usize, seed: u64) -> (Vec<u8>, RowGeometry) {
        if let Some(format) = representation.strip_suffix("@rows8") {
            let row_major = format!("{format}@rows16");
            let (source, source_geometry) = self::rows(&row_major, rows, k, seed);
            let kernels = resolve(Tier::detected().unwrap(), representation).unwrap();
            let mut geometry = kernels.geometry(k, 0);
            let mut data = vec![0u8; rows.div_ceil(8) * 8 * geometry.stride];
            let groups = match format {
                "q8g32s" => k.div_ceil(32),
                _ => k.div_ceil(256),
            };
            let offsets = [
                source_geometry.codes,
                source_geometry.high,
                source_geometry.scales,
                source_geometry.supers,
            ];
            for row in 0..rows {
                let tile = row / 8;
                let lane = row % 8;
                for (plane, bytes) in geometry.groups.iter().enumerate() {
                    if *bytes == 0 {
                        continue;
                    }
                    for group in 0..groups {
                        let from = row * geometry.stride + offsets[plane] + group * bytes;
                        let to = tile * 8 * geometry.stride
                            + offsets[plane] * 8
                            + group * bytes * 8
                            + lane * bytes;
                        data[to..to + bytes].copy_from_slice(&source[from..from + bytes]);
                    }
                }
            }
            geometry.base = data.as_ptr() as usize;
            geometry.matrix_rows = rows;
            return (data, geometry);
        }
        let kernels = resolve(Tier::detected().unwrap(), representation).unwrap();
        let dense = match representation {
            "f32" => 4,
            "bf16" | "f16" => 2,
            _ => 0,
        };
        let geometry = kernels.geometry(k, k * dense);
        let mut data = bytes(seed, rows * geometry.stride);
        let half = |value: f32| f32_to_f16(value).to_le_bytes();
        for row in 0..rows {
            let base = row * geometry.stride;
            match representation {
                "f32" => {
                    for i in 0..k {
                        let value = (data[base + 4 * i] as f32 - 128.0) / 64.0;
                        data[base + 4 * i..base + 4 * i + 4].copy_from_slice(&value.to_le_bytes());
                    }
                }
                "bf16" => {
                    for i in 0..k {
                        let value = (data[base + 2 * i] as f32 - 128.0) / 64.0;
                        data[base + 2 * i..base + 2 * i + 2]
                            .copy_from_slice(&f32_to_bf16(value).to_le_bytes());
                    }
                }
                "f16" => {
                    for i in 0..k {
                        let value = (data[base + 2 * i] as f32 - 128.0) / 64.0;
                        data[base + 2 * i..base + 2 * i + 2].copy_from_slice(&half(value));
                    }
                }
                "q4k@rows16" | "q5k@rows16" => {
                    for block in 0..k.div_ceil(256) {
                        let at = base + geometry.supers + 4 * block;
                        data[at..at + 2].copy_from_slice(&half(0.01 + block as f32 * 1e-3));
                        data[at + 2..at + 4].copy_from_slice(&half(0.02));
                    }
                }
                "q6k@rows16" => {
                    for block in 0..k.div_ceil(256) {
                        let at = base + geometry.supers + 2 * block;
                        data[at..at + 2].copy_from_slice(&half(0.003));
                    }
                }
                "q8g32s@rows16" => {
                    for group in 0..k.div_ceil(32) {
                        let at = base + geometry.supers + 2 * group;
                        data[at..at + 2].copy_from_slice(&half(0.004));
                    }
                }
                "iq4g32@rows16" => {
                    for packet in 0..k.div_ceil(256) * 8 {
                        let at = base + geometry.supers + 4 * packet;
                        data[at..at + 4]
                            .copy_from_slice(&(0.0003 * (packet % 3 + 1) as f32).to_le_bytes());
                    }
                }
                other => panic!("no generator for {other}"),
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
        for representation in representations() {
            let (data, geometry) = rows(representation, 8, k, 11);
            let reference = resolve(Tier::detected().unwrap(), representation).unwrap();
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
                let kernels = resolve(tier, representation).unwrap();
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
        let kernels = resolve(tier, representation).unwrap();
        let (first, first_geometry) = rows(representation, 5, k, 41);
        let (second, second_geometry) = rows(representation, 5, k, 43);
        let mut bytes = first.clone();
        bytes.extend_from_slice(&second);
        let view = unsafe {
            crate::Weights::from_tensor(bytes.as_ptr(), &[2, 5, k as u64], &[8, 1, 1], kernels)
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
        for representation in representations() {
            let (data, geometry) = rows(representation, 8, k, 31);
            let kernels = resolve(tier, representation).unwrap();
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
            let kernels = resolve(tier, representation).unwrap();
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
        for representation in representations() {
            let (data, geometry) = self::rows(representation, rows, k, 9);
            let kernels = resolve(tier, representation).unwrap();
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
        macro_rules! dispatch {
            ($module:ident) => {{
                macro_rules! by_block {
                    ($format:ty) => {
                        match block {
                            1 => unsafe {
                                $module::inlined::<$format, 1>(data, rows, geometry, x, out)
                            },
                            2 => unsafe {
                                $module::inlined::<$format, 2>(data, rows, geometry, x, out)
                            },
                            4 => unsafe {
                                $module::inlined::<$format, 4>(data, rows, geometry, x, out)
                            },
                            _ => unsafe {
                                $module::inlined::<$format, 8>(data, rows, geometry, x, out)
                            },
                        }
                    };
                }
                match representation {
                    "f32" => by_block!(DenseRows<crate::element::F32>),
                    "bf16" => by_block!(DenseRows<crate::element::Bf16>),
                    "f16" => by_block!(DenseRows<crate::element::F16>),
                    "q4k@rows16" => by_block!(Q4K),
                    "q5k@rows16" => by_block!(Q5K),
                    "q6k@rows16" => by_block!(Q6K),
                    "q8g32s@rows16" => by_block!(Q8),
                    "iq4g32@rows16" => by_block!(Iq4),
                    "q4k@rows8" => by_block!(Rows8<Q4K>),
                    "q5k@rows8" => by_block!(Rows8<Q5K>),
                    "q6k@rows8" => by_block!(Rows8<Q6K>),
                    "q8g32s@rows8" => by_block!(Rows8<Q8>),
                    "iq4g32@rows8" => by_block!(Rows8<Iq4>),
                    other => panic!("no format for {other}"),
                }
            }};
        }
        match tier {
            #[cfg(target_arch = "x86_64")]
            Tier::X86V2 => dispatch!(x86v2),
            #[cfg(target_arch = "x86_64")]
            Tier::X86V3 => dispatch!(x86v3),
            #[cfg(target_arch = "x86_64")]
            Tier::X86V4 => dispatch!(x86v4),
            #[cfg(target_arch = "x86_64")]
            Tier::X86V4Vnni => dispatch!(x86v4vnni),
            #[cfg(target_arch = "aarch64")]
            Tier::Neon => dispatch!(neon),
            #[allow(unreachable_patterns)]
            other => panic!("tier {other:?} is not of this architecture"),
        }
    }

    #[test]
    #[ignore = "measurement; run with --release --ignored --nocapture; CPU_DOT_REPRESENTATIONS and CPU_DOT_BLOCKS narrow it"]
    fn dot_component_throughput() {
        let tier = Tier::detected().unwrap();
        let (rows, k) = (4096usize, 4096usize);
        let x = bytes(5, k)
            .iter()
            .map(|byte| (*byte as f32 - 128.0) / 100.0)
            .collect::<Vec<_>>();
        let selected = std::env::var("CPU_DOT_REPRESENTATIONS").ok();
        let selected_blocks = std::env::var("CPU_DOT_BLOCKS").ok();
        for representation in representations() {
            if selected
                .as_ref()
                .is_some_and(|names| !names.split(',').any(|name| name.trim() == representation))
            {
                continue;
            }
            let (data, geometry) = self::rows(representation, rows, k, 9);
            let kernels = resolve(tier, representation).unwrap();
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
        for representation in representations() {
            let (data, geometry) = rows(representation, 8, k, 11);
            let reference = resolve(detected, representation).unwrap();
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
                let kernels = resolve(tier, representation).unwrap();
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

    #[test]
    fn dense_rows_with_a_partial_packet_decode_their_values() {
        let k = 45;
        let (data, geometry) = rows("bf16", 1, k, 3);
        let kernels = resolve(Tier::detected().unwrap(), "bf16").unwrap();
        let mut out = vec![f32::NAN; k];
        unsafe { kernels.decode(data.as_ptr(), &geometry, k, &mut out) };
        for (i, value) in out.iter().enumerate() {
            let bits = u16::from_le_bytes([data[2 * i], data[2 * i + 1]]);
            assert_eq!(value.to_bits(), crate::element::bf16_to_f32(bits).to_bits());
        }
    }
}
