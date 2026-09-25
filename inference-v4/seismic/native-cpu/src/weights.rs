//! Weight representations a CPU weight operand can be bound to, their row
//! geometry, and the bodies of their components.
//!
//! A weight row of `k` logical values is read in packets of 32 consecutive
//! values. Dense rows (`f32`, `bf16`, `f16`) hold the values contiguously; a
//! partial final packet decodes its missing values as zero. Packed rows use
//! the registry's `rows16` layout: per row, a low-code plane (16 bytes per
//! packet, codes 2i and 2i + 1 in the low and high nibble of byte i), a
//! high-code plane (q5k: one bit per value, q6k: two), local coefficients and
//! super factors, each plane starting at a 16-byte aligned offset. A value is
//! `scale * code + bias` for its coefficient group, rounded once to `f32` as
//! the registry's decode recipe rounds it.
//!
//! The dot bodies accumulate `w[i] * x[i]` (fused) into lane `i % LANES` in
//! index order and combine the lanes in the fixed tree of `reduce`, so a
//! result depends only on the values, never on the tier or the row block.

use crate::element::{f16_to_f32, Dense};
use crate::quant::{Q8Block, Q8_BLOCK};
use crate::reduce::{combine, LANES};

/// Values of one packet.
pub const PACKET: usize = 32;

/// Byte alignment of every `rows16` plane and of the row stride.
pub const ROW_ALIGNMENT: usize = 16;

/// Byte geometry of the rows of one weight operand.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RowGeometry {
    /// Bytes between the starts of adjacent rows.
    pub stride: usize,
    /// Plane offsets within a row; planes a representation lacks are zero.
    pub codes: usize,
    pub high: usize,
    pub scales: usize,
    pub supers: usize,
    /// Bytes per storage group in code, high, local and super planes.
    pub groups: [usize; 4],
    /// Row addressing of the bound tensor. Zero base is used for unbound geometry.
    pub base: usize,
    pub matrix_rows: usize,
    pub rows8: bool,
}

#[derive(Clone, Copy)]
enum Plane { Codes, High, Scales, Supers }

impl RowGeometry {
    #[inline(always)]
    unsafe fn address(&self, row: *const u8, plane: Plane, offset: usize) -> *const u8 {
        let (start, bytes) = match plane {
            Plane::Codes => (self.codes, self.groups[0]),
            Plane::High => (self.high, self.groups[1]),
            Plane::Scales => (self.scales, self.groups[2]),
            Plane::Supers => (self.supers, self.groups[3]),
        };
        if !self.rows8 { return unsafe { row.add(start + offset) }; }
        let index = (row as usize - self.base) / self.stride;
        let n = self.matrix_rows;
        let stored = index / n * n.div_ceil(8) * 8 + index % n;
        let tile = stored / 8;
        let lane = stored % 8;
        let byte = tile * self.stride * 8 + start * 8
            + offset / bytes * bytes * 8 + lane * bytes + offset % bytes;
        unsafe { (self.base as *const u8).add(byte) }
    }
}

fn align(bytes: usize) -> usize {
    bytes.div_ceil(ROW_ALIGNMENT) * ROW_ALIGNMENT
}

/// The `rows16` geometry of rows of `k` values: storage groups of `group`
/// values, planes of `plane_bytes` bytes per group in storage order.
fn rows16(k: usize, group: usize, plane_bytes: &[usize]) -> ([usize; 4], usize) {
    let groups = k.div_ceil(group);
    let mut offsets = [0usize; 4];
    let mut end = 0usize;
    for (plane, bytes) in plane_bytes.iter().enumerate() {
        offsets[plane] = align(end);
        end = offsets[plane] + groups * bytes;
    }
    (offsets, align(end))
}

/// One weight representation: its geometry and packet decode.
pub trait Format: Copy + Send + Sync + 'static {
    /// The registry representation name a bound tensor reports.
    const NAME: &'static str;
    /// Bytes of one element of a dense representation; 0 for a packed one.
    const DENSE_BYTES: usize;

    /// The geometry of rows of `k` values; `dense_stride` is the bound
    /// tensor's row stride in bytes, which only dense rows use.
    fn geometry(k: usize, dense_stride: usize) -> RowGeometry;

    /// The 32 values of packet `p` of the row at `row`.
    ///
    /// # Safety
    /// `row` addresses a row of this representation with `geometry` and at
    /// least `32 * p + 1` values.
    unsafe fn decode(row: *const u8, geometry: &RowGeometry, p: usize, k: usize, out: &mut [f32; PACKET]);

    /// Values `256 b .. 256 b + 256` (below `k`) of the row at `row` dotted
    /// with the quantized activation block `x`: exact integer sums per
    /// coefficient group, combined in `f32` in group order.
    ///
    /// # Safety
    /// `row` addresses a row of this representation with `geometry` and `k`
    /// values, and `256 b < k`.
    unsafe fn dot_q8<const MODE: u8>(row: *const u8, geometry: &RowGeometry, b: usize, k: usize, x: &Q8Block) -> f32;
}

pub trait Rows8Format: Format {
    const ROWS8_NAME: &'static str;
}

/// Eight-row interleaved storage of a packed weight format. Each plane's
/// storage groups are adjacent across the eight rows of a tile.
#[derive(Clone, Copy, Debug)]
pub struct Rows8<W: Rows8Format>(W);

impl<W: Rows8Format> Format for Rows8<W> {
    const NAME: &'static str = W::ROWS8_NAME;
    const DENSE_BYTES: usize = 0;

    fn geometry(k: usize, dense_stride: usize) -> RowGeometry {
        let mut geometry = W::geometry(k, dense_stride);
        geometry.rows8 = true;
        geometry
    }

    unsafe fn decode(row: *const u8, geometry: &RowGeometry, p: usize, k: usize, out: &mut [f32; PACKET]) {
        unsafe { W::decode(row, geometry, p, k, out) }
    }

    unsafe fn dot_q8<const MODE: u8>(row: *const u8, geometry: &RowGeometry, b: usize, k: usize, x: &Q8Block) -> f32 {
        unsafe { W::dot_q8::<MODE>(row, geometry, b, k, x) }
    }
}

/// Packets of super block `b` of a row of `k` values.
#[inline(always)]
fn block_packets(b: usize, k: usize) -> std::ops::Range<usize> {
    8 * b..(8 * b + 8).min(k.div_ceil(PACKET))
}

/// Dense rows of element `E`.
#[derive(Clone, Copy, Debug)]
pub struct DenseRows<E: Dense>(E);

impl<E: Dense> Format for DenseRows<E> {
    const NAME: &'static str = E::NAME;
    const DENSE_BYTES: usize = E::BYTES;

    fn geometry(_k: usize, dense_stride: usize) -> RowGeometry {
        RowGeometry { stride: dense_stride, codes: 0, high: 0, scales: 0, supers: 0, groups: [0; 4], base: 0, matrix_rows: 0, rows8: false }
    }

    #[inline(always)]
    unsafe fn decode(row: *const u8, _geometry: &RowGeometry, p: usize, k: usize, out: &mut [f32; PACKET]) {
        let first = p * PACKET;
        let valid = (k - first).min(PACKET);
        let values = unsafe { std::slice::from_raw_parts(row.cast::<E::Storage>().add(first), valid) };
        for (target, value) in out.iter_mut().zip(values) {
            *target = E::widen(*value);
        }
        for target in &mut out[valid..] {
            *target = 0.0;
        }
    }

    #[inline(always)]
    unsafe fn dot_q8<const MODE: u8>(row: *const u8, _geometry: &RowGeometry, b: usize, k: usize, x: &Q8Block) -> f32 {
        let first = b * Q8_BLOCK;
        let valid = (k - first).min(Q8_BLOCK);
        let values = unsafe { std::slice::from_raw_parts(row.cast::<E::Storage>().add(first), valid) };
        let mut lanes = [0.0f32; LANES];
        for (i, (value, code)) in values.iter().zip(&x.codes).enumerate() {
            lanes[i % LANES] = E::widen(*value).mul_add(f32::from(*code), lanes[i % LANES]);
        }
        x.d * combine(lanes)
    }
}

/// The exact dot of a packet's 32 codes (codes 2i and 2i + 1 in the low and
/// high nibble of byte i, each joined with `high(value)`) with its 32
/// activation `codes`.
#[inline(always)]
fn nibble_dot<const MODE: u8>(bytes: &[u8; 16], high: impl Fn(usize) -> u8, codes: &[i8]) -> i32 {
    let codes = &codes[..PACKET];
    let mut weights = [0u8; PACKET];
    for (i, byte) in bytes.iter().enumerate() {
        weights[2 * i] = (byte & 15) | high(2 * i);
        weights[2 * i + 1] = (byte >> 4) | high(2 * i + 1);
    }
    let activations: &[i8; PACKET] = codes.try_into().expect("one packet");
    dot_u8_i8::<MODE>(&weights, activations)
}

/// Integer dot of 32 unsigned weight codes and signed activation codes.
/// The mode is fixed by the component's tier, so lower tiers never execute
/// instructions they did not advertise.
#[inline(always)]
fn dot_u8_i8<const MODE: u8>(weights: &[u8; 32], activations: &[i8; 32]) -> i32 {
    #[cfg(target_arch = "x86_64")]
    {
        if MODE == 2 { return unsafe { dot_u8_i8_vnni(weights, activations) }; }
        if MODE == 1 { return unsafe { dot_u8_i8_avx2(weights, activations) }; }
    }
    weights.iter().zip(activations).map(|(w, a)| i32::from(*w) * i32::from(*a)).sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_u8_i8_avx2(weights: &[u8; 32], activations: &[i8; 32]) -> i32 {
    use std::arch::x86_64::*;
    let w = unsafe { _mm256_loadu_si256(weights.as_ptr().cast()) };
    let a = unsafe { _mm256_loadu_si256(activations.as_ptr().cast()) };
    let pairs = _mm256_maddubs_epi16(w, a);
    let sums = _mm256_madd_epi16(pairs, _mm256_set1_epi16(1));
    let mut lanes = [0i32; 8];
    unsafe { _mm256_storeu_si256(lanes.as_mut_ptr().cast(), sums) };
    lanes.into_iter().sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,avx512vnni,avx512vl")]
unsafe fn dot_u8_i8_vnni(weights: &[u8; 32], activations: &[i8; 32]) -> i32 {
    use std::arch::x86_64::*;
    let w = unsafe { _mm256_loadu_si256(weights.as_ptr().cast()) };
    let a = unsafe { _mm256_loadu_si256(activations.as_ptr().cast()) };
    let sums = _mm256_dpbusd_epi32(_mm256_setzero_si256(), w, a);
    let mut lanes = [0i32; 8];
    unsafe { _mm256_storeu_si256(lanes.as_mut_ptr().cast(), sums) };
    lanes.into_iter().sum()
}

/// Signed packed weights use widening AVX2 products, or VNNI's unsigned
/// form with a 128-code offset and the exact activation sum correction.
#[inline(always)]
fn dot_i8_i8<const MODE: u8>(weights: &[i8; 32], activations: &[i8; 32]) -> i32 {
    #[cfg(target_arch = "x86_64")]
    {
        if MODE == 2 {
            let unsigned = weights.map(|w| (i16::from(w) + 128) as u8);
            let correction: i32 = activations.iter().map(|a| i32::from(*a)).sum();
            return unsafe { dot_u8_i8_vnni(&unsigned, activations) } - 128 * correction;
        }
        if MODE == 1 { return unsafe { dot_i8_i8_avx2(weights, activations) }; }
    }
    weights.iter().zip(activations).map(|(w, a)| i32::from(*w) * i32::from(*a)).sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_i8_i8_avx2(weights: &[i8; 32], activations: &[i8; 32]) -> i32 {
    use std::arch::x86_64::*;
    let mut lanes = [0i32; 8];
    for half in 0..2 {
        let w = unsafe { _mm_loadu_si128(weights.as_ptr().add(half * 16).cast()) };
        let a = unsafe { _mm_loadu_si128(activations.as_ptr().add(half * 16).cast()) };
        let pairs = _mm256_madd_epi16(_mm256_cvtepi8_epi16(w), _mm256_cvtepi8_epi16(a));
        let mut part = [0i32; 8];
        unsafe { _mm256_storeu_si256(part.as_mut_ptr().cast(), pairs) };
        for (lane, value) in lanes.iter_mut().zip(part) { *lane += value; }
    }
    lanes.into_iter().sum()
}

/// The 6-bit (scale, min) of group `local` of a q4k/q5k super block's 12
/// packed coefficient bytes.
#[inline(always)]
fn k_scale_min(fields: &[u8; 12], local: usize) -> (i32, i32) {
    let at = (3 * local) >> 1;
    let pair = (u32::from(fields[at]) | (u32::from(fields[at + 1]) << 8)) >> ((local & 1) * 4);
    ((pair & 63) as i32, ((pair >> 6) & 63) as i32)
}

/// The integer block dot of a q4k or q5k row: `high(p, i)` is the high code
/// bit of value `i` of packet `p`, already shifted to bit 4.
#[inline(always)]
unsafe fn k_dot_q8<const MODE: u8>(row: *const u8, geometry: &RowGeometry, b: usize, k: usize, x: &Q8Block, high: impl Fn(usize, usize) -> u8) -> f32 {
    let factors: [u16; 2] = unsafe { read(geometry.address(row, Plane::Supers, 4 * b)) };
    let fields: [u8; 12] = unsafe { read(geometry.address(row, Plane::Scales, 12 * b)) };
    let (mut scaled, mut minimums) = (0i32, 0i32);
    for p in block_packets(b, k) {
        let local = p - 8 * b;
        let (scale, minimum) = k_scale_min(&fields, local);
        let bytes: [u8; 16] = unsafe { read(geometry.address(row, Plane::Codes, 16 * p)) };
        scaled += scale * nibble_dot::<MODE>(&bytes, |i| high(p, i), &x.codes[32 * local..]);
        minimums += minimum * (i32::from(x.sums[2 * local]) + i32::from(x.sums[2 * local + 1]));
    }
    x.d * (f16_to_f32(factors[0]) * scaled as f32 - f16_to_f32(factors[1]) * minimums as f32)
}

#[inline(always)]
unsafe fn read<T: Copy>(pointer: *const u8) -> T {
    unsafe { pointer.cast::<T>().read_unaligned() }
}

/// The 16 low-code bytes of packet `p` as 32 nibble codes.
#[inline(always)]
unsafe fn nibbles(row: *const u8, geometry: &RowGeometry, p: usize) -> [u8; PACKET] {
    let bytes: [u8; 16] = unsafe { read(geometry.address(row, Plane::Codes, 16 * p)) };
    let mut codes = [0u8; PACKET];
    for (i, byte) in bytes.iter().enumerate() {
        codes[2 * i] = byte & 15;
        codes[2 * i + 1] = byte >> 4;
    }
    codes
}

/// The (scale, bias) of packet `p` of a q4k or q5k row: 6-bit scale and min
/// of the packet's group of 32 and the (d, dmin) of its group of 256.
#[inline(always)]
unsafe fn k_coefficients(row: *const u8, geometry: &RowGeometry, p: usize) -> (f32, f32) {
    let (block, local) = (p >> 3, p & 7);
    let fields: [u8; 2] = unsafe { read(geometry.address(row, Plane::Scales, 12 * block + ((3 * local) >> 1))) };
    let pair = (u32::from(fields[0]) | (u32::from(fields[1]) << 8)) >> ((local & 1) * 4);
    let factors: [u16; 2] = unsafe { read(geometry.address(row, Plane::Supers, 4 * block)) };
    let scale = f16_to_f32(factors[0]) * (pair & 63) as f32;
    let bias = -(f16_to_f32(factors[1]) * ((pair >> 6) & 63) as f32);
    (scale, bias)
}

/// Registry `q4k` in the `rows16` layout.
#[derive(Clone, Copy, Debug)]
pub struct Q4K;

impl Format for Q4K {
    const NAME: &'static str = "q4k@rows16";
    const DENSE_BYTES: usize = 0;

    fn geometry(k: usize, _dense_stride: usize) -> RowGeometry {
        let ([codes, scales, supers, _], stride) = rows16(k, 256, &[128, 12, 4]);
        RowGeometry { stride, codes, high: 0, scales, supers, groups: [128, 0, 12, 4], base: 0, matrix_rows: 0, rows8: false }
    }

    #[inline(always)]
    unsafe fn decode(row: *const u8, geometry: &RowGeometry, p: usize, _k: usize, out: &mut [f32; PACKET]) {
        let codes = unsafe { nibbles(row, geometry, p) };
        let (scale, bias) = unsafe { k_coefficients(row, geometry, p) };
        for (target, code) in out.iter_mut().zip(codes) {
            *target = scale.mul_add(f32::from(code), bias);
        }
    }

    #[inline(always)]
    unsafe fn dot_q8<const MODE: u8>(row: *const u8, geometry: &RowGeometry, b: usize, k: usize, x: &Q8Block) -> f32 {
        unsafe { k_dot_q8::<MODE>(row, geometry, b, k, x, |_, _| 0) }
    }
}

/// Registry `q5k` in the `rows16` layout.
#[derive(Clone, Copy, Debug)]
pub struct Q5K;

impl Format for Q5K {
    const NAME: &'static str = "q5k@rows16";
    const DENSE_BYTES: usize = 0;

    fn geometry(k: usize, _dense_stride: usize) -> RowGeometry {
        let ([codes, high, scales, supers], stride) = rows16(k, 256, &[128, 32, 12, 4]);
        RowGeometry { stride, codes, high, scales, supers, groups: [128, 32, 12, 4], base: 0, matrix_rows: 0, rows8: false }
    }

    #[inline(always)]
    unsafe fn decode(row: *const u8, geometry: &RowGeometry, p: usize, _k: usize, out: &mut [f32; PACKET]) {
        let mut codes = unsafe { nibbles(row, geometry, p) };
        let high: u32 = unsafe { read(geometry.address(row, Plane::High, 4 * p)) };
        for (i, code) in codes.iter_mut().enumerate() {
            *code |= (((high >> i) & 1) as u8) << 4;
        }
        let (scale, bias) = unsafe { k_coefficients(row, geometry, p) };
        for (target, code) in out.iter_mut().zip(codes) {
            *target = scale.mul_add(f32::from(code), bias);
        }
    }

    #[inline(always)]
    unsafe fn dot_q8<const MODE: u8>(row: *const u8, geometry: &RowGeometry, b: usize, k: usize, x: &Q8Block) -> f32 {
        let high = |p: usize, i: usize| {
            // SAFETY: packet `p` of this row exists (`k_dot_q8` visits only those).
            let bits: u32 = unsafe { read(geometry.address(row, Plane::High, 4 * p)) };
            (((bits >> i) & 1) as u8) << 4
        };
        unsafe { k_dot_q8::<MODE>(row, geometry, b, k, x, high) }
    }
}

/// Registry `q6k` in the `rows16` layout.
#[derive(Clone, Copy, Debug)]
pub struct Q6K;

impl Format for Q6K {
    const NAME: &'static str = "q6k@rows16";
    const DENSE_BYTES: usize = 0;

    fn geometry(k: usize, _dense_stride: usize) -> RowGeometry {
        let ([codes, high, scales, supers], stride) = rows16(k, 256, &[128, 64, 16, 2]);
        RowGeometry { stride, codes, high, scales, supers, groups: [128, 64, 16, 2], base: 0, matrix_rows: 0, rows8: false }
    }

    #[inline(always)]
    unsafe fn decode(row: *const u8, geometry: &RowGeometry, p: usize, _k: usize, out: &mut [f32; PACKET]) {
        let mut codes = unsafe { nibbles(row, geometry, p) };
        let high: u64 = unsafe { read(geometry.address(row, Plane::High, 8 * p)) };
        for (i, code) in codes.iter_mut().enumerate() {
            *code |= (((high >> (2 * i)) & 3) as u8) << 4;
        }
        let local: [i8; 2] = unsafe { read(geometry.address(row, Plane::Scales, 2 * p)) };
        let d = f16_to_f32(unsafe { read::<u16>(geometry.address(row, Plane::Supers, 2 * (p >> 3))) });
        let scales = [d * f32::from(local[0]), d * f32::from(local[1])];
        for (i, (target, code)) in out.iter_mut().zip(codes).enumerate() {
            *target = scales[i / 16] * (f32::from(code) - 32.0);
        }
    }

    #[inline(always)]
    unsafe fn dot_q8<const MODE: u8>(row: *const u8, geometry: &RowGeometry, b: usize, k: usize, x: &Q8Block) -> f32 {
        let d = f16_to_f32(unsafe { read::<u16>(geometry.address(row, Plane::Supers, 2 * b)) });
        let mut scaled = 0i32;
        for p in block_packets(b, k) {
            let local: [i8; 2] = unsafe { read(geometry.address(row, Plane::Scales, 2 * p)) };
            let bytes: [u8; 16] = unsafe { read(geometry.address(row, Plane::Codes, 16 * p)) };
            let high: u64 = unsafe { read(geometry.address(row, Plane::High, 8 * p)) };
            let activations: &[i8; 32] =
                x.codes[32 * (p - 8 * b)..32 * (p - 8 * b) + 32].try_into().expect("a packet of codes");
            // The packet's signed codes (low nibble, two high bits, less 32).
            let mut codes = [0i8; PACKET];
            for (i, byte) in bytes.iter().enumerate() {
                let high = (high >> (4 * i)) as u8;
                codes[2 * i] = ((byte & 15) | ((high & 3) << 4)) as i8 - 32;
                codes[2 * i + 1] = ((byte >> 4) | (((high >> 2) & 3) << 4)) as i8 - 32;
            }
            let mut first = [0i8; 32];
            let mut second = [0i8; 32];
            first[..16].copy_from_slice(&activations[..16]);
            second[16..].copy_from_slice(&activations[16..]);
            scaled += i32::from(local[0]) * dot_i8_i8::<MODE>(&codes, &first)
                + i32::from(local[1]) * dot_i8_i8::<MODE>(&codes, &second);
        }
        x.d * (d * scaled as f32)
    }
}

/// Registry `q8g32s` in the `rows16` layout.
#[derive(Clone, Copy, Debug)]
pub struct Q8;

impl Format for Q8 {
    const NAME: &'static str = "q8g32s@rows16";
    const DENSE_BYTES: usize = 0;

    fn geometry(k: usize, _dense_stride: usize) -> RowGeometry {
        let ([codes, supers, _, _], stride) = rows16(k, 32, &[32, 2]);
        RowGeometry { stride, codes, high: 0, scales: 0, supers, groups: [32, 0, 0, 2], base: 0, matrix_rows: 0, rows8: false }
    }

    #[inline(always)]
    unsafe fn decode(row: *const u8, geometry: &RowGeometry, p: usize, _k: usize, out: &mut [f32; PACKET]) {
        let codes: [i8; PACKET] = unsafe { read(geometry.address(row, Plane::Codes, 32 * p)) };
        let scale = f16_to_f32(unsafe { read::<u16>(geometry.address(row, Plane::Supers, 2 * p)) });
        for (target, code) in out.iter_mut().zip(codes) {
            *target = scale * f32::from(code);
        }
    }

    #[inline(always)]
    unsafe fn dot_q8<const MODE: u8>(row: *const u8, geometry: &RowGeometry, b: usize, k: usize, x: &Q8Block) -> f32 {
        let mut sum = 0.0f32;
        for p in block_packets(b, k) {
            let codes: [i8; PACKET] = unsafe { read(geometry.address(row, Plane::Codes, 32 * p)) };
            let scale = f16_to_f32(unsafe { read::<u16>(geometry.address(row, Plane::Supers, 2 * p)) });
            let activations: &[i8; 32] = x.codes[32 * (p - 8 * b)..32 * (p - 8 * b) + 32].try_into().expect("one packet");
            let exact = dot_i8_i8::<MODE>(&codes, activations);
            sum = scale.mul_add(exact as f32, sum);
        }
        x.d * sum
    }
}

/// The code values of registry `iq4g32`: a 4-bit code selects one.
pub const IQ4_VALUES: [i8; 16] = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113];

/// Registry `iq4g32` in the `rows16` layout: nibble codes into
/// [`IQ4_VALUES`], one `f32` scale per packet (eight per 256-value group).
#[derive(Clone, Copy, Debug)]
pub struct Iq4;

impl Format for Iq4 {
    const NAME: &'static str = "iq4g32@rows16";
    const DENSE_BYTES: usize = 0;

    fn geometry(k: usize, _dense_stride: usize) -> RowGeometry {
        let ([codes, supers, _, _], stride) = rows16(k, 256, &[128, 32]);
        RowGeometry { stride, codes, high: 0, scales: 0, supers, groups: [128, 0, 0, 32], base: 0, matrix_rows: 0, rows8: false }
    }

    #[inline(always)]
    unsafe fn decode(row: *const u8, geometry: &RowGeometry, p: usize, _k: usize, out: &mut [f32; PACKET]) {
        let codes = unsafe { nibbles(row, geometry, p) };
        let scale: f32 = unsafe { read(geometry.address(row, Plane::Supers, 4 * p)) };
        for (target, code) in out.iter_mut().zip(codes) {
            *target = scale * f32::from(IQ4_VALUES[usize::from(code)]);
        }
    }

    #[inline(always)]
    unsafe fn dot_q8<const MODE: u8>(row: *const u8, geometry: &RowGeometry, b: usize, k: usize, x: &Q8Block) -> f32 {
        let mut sum = 0.0f32;
        for p in block_packets(b, k) {
            let codes = unsafe { nibbles(row, geometry, p) };
            let scale: f32 = unsafe { read(geometry.address(row, Plane::Supers, 4 * p)) };
            let activations: &[i8; 32] = x.codes[32 * (p - 8 * b)..32 * (p - 8 * b) + 32].try_into().expect("one packet");
            let weights = codes.map(|code| IQ4_VALUES[usize::from(code)]);
            let exact = dot_i8_i8::<MODE>(&weights, activations);
            sum = scale.mul_add(exact as f32, sum);
        }
        x.d * sum
    }
}

impl Rows8Format for Q4K { const ROWS8_NAME: &'static str = "q4k@rows8"; }
impl Rows8Format for Q5K { const ROWS8_NAME: &'static str = "q5k@rows8"; }
impl Rows8Format for Q6K { const ROWS8_NAME: &'static str = "q6k@rows8"; }
impl Rows8Format for Q8 { const ROWS8_NAME: &'static str = "q8g32s@rows8"; }
impl Rows8Format for Iq4 { const ROWS8_NAME: &'static str = "iq4g32@rows8"; }

/// The body of the row-decode component: the `k` values of one row.
///
/// # Safety
/// `row` addresses a row of `W` with `geometry` and `k` values; `out` holds
/// at least `k` values.
#[inline(always)]
pub unsafe fn decode_row<W: Format>(row: *const u8, geometry: &RowGeometry, k: usize, out: &mut [f32]) {
    let out = &mut out[..k];
    let mut packet = [0.0f32; PACKET];
    for (p, chunk) in out.chunks_mut(PACKET).enumerate() {
        unsafe { W::decode(row, geometry, p, k, &mut packet) };
        chunk.copy_from_slice(&packet[..chunk.len()]);
    }
}

/// The body of the dot component: rows `0..R` (from `rows`, `geometry.stride`
/// apart) against the `k` values of `x`.
///
/// # Safety
/// `rows` addresses `R` rows of `W` with `geometry` and `k` values.
#[inline(always)]
pub unsafe fn dot_rows<W: Format, const R: usize>(
    rows: *const u8,
    geometry: &RowGeometry,
    x: &[f32],
    k: usize,
) -> [f32; R] {
    let x = &x[..k];
    let mut lanes = [[0.0f32; LANES]; R];
    let mut decoded = [[0.0f32; PACKET]; R];
    let full = k / PACKET;
    for p in 0..full {
        for (r, packet) in decoded.iter_mut().enumerate() {
            unsafe { W::decode(rows.add(r * geometry.stride), geometry, p, k, packet) };
        }
        let xs: &[f32; PACKET] = x[p * PACKET..(p + 1) * PACKET].try_into().expect("a whole packet");
        for (lanes, packet) in lanes.iter_mut().zip(&decoded) {
            for chunk in 0..PACKET / LANES {
                for lane in 0..LANES {
                    let i = chunk * LANES + lane;
                    lanes[lane] = packet[i].mul_add(xs[i], lanes[lane]);
                }
            }
        }
    }
    let tail = k - full * PACKET;
    if tail > 0 {
        for (r, packet) in decoded.iter_mut().enumerate() {
            unsafe { W::decode(rows.add(r * geometry.stride), geometry, full, k, packet) };
        }
        let xs = &x[full * PACKET..];
        for (lanes, packet) in lanes.iter_mut().zip(&decoded) {
            for (i, value) in xs.iter().enumerate() {
                lanes[i % LANES] = packet[i].mul_add(*value, lanes[i % LANES]);
            }
        }
    }
    lanes.map(combine)
}

/// The body of the quantized dot component: rows `0..R` (from `rows`,
/// `geometry.stride` apart) of `k` values against quantized activations
/// `x`, accumulating each row's block dots in block order.
///
/// # Safety
/// `rows` addresses `R` rows of `W` with `geometry` and `k` values, and `x`
/// holds `quant::blocks(k)` blocks.
#[inline(always)]
pub unsafe fn dot_q8_rows<W: Format, const R: usize, const MODE: u8>(
    rows: *const u8,
    geometry: &RowGeometry,
    x: &[Q8Block],
    k: usize,
) -> [f32; R] {
    let x = &x[..crate::quant::blocks(k)];
    let mut sums = [0.0f32; R];
    for (b, block) in x.iter().enumerate() {
        for (r, sum) in sums.iter_mut().enumerate() {
            *sum += unsafe { W::dot_q8::<MODE>(rows.add(r * geometry.stride), geometry, b, k, block) };
        }
    }
    sums
}

#[cfg(test)]
mod tests {
    use super::*;
    use seismic_lang::registry::{representation, representation_info, RepresentationKind};

    fn registry_geometry(name: &str, k: u64) -> (u64, Vec<u64>) {
        let RepresentationKind::PackedRows(layout) = &representation_info(representation(name).unwrap()).kind
        else {
            panic!("`{name}` is a row layout")
        };
        let geometry = layout.geometry(k).unwrap();
        (geometry.stride, geometry.offsets)
    }

    #[test]
    fn rows16_geometry_matches_the_registry() {
        for k in [32usize, 256, 512, 2560, 2816, 4096, 9216, 9472, 17408] {
            let q4k = Q4K::geometry(k, 0);
            assert_eq!(
                registry_geometry(Q4K::NAME, k as u64),
                (q4k.stride as u64, vec![q4k.codes as u64, q4k.scales as u64, q4k.supers as u64])
            );
            let q5k = Q5K::geometry(k, 0);
            assert_eq!(
                registry_geometry(Q5K::NAME, k as u64),
                (q5k.stride as u64, vec![q5k.codes as u64, q5k.high as u64, q5k.scales as u64, q5k.supers as u64])
            );
            let q6k = Q6K::geometry(k, 0);
            assert_eq!(
                registry_geometry(Q6K::NAME, k as u64),
                (q6k.stride as u64, vec![q6k.codes as u64, q6k.high as u64, q6k.scales as u64, q6k.supers as u64])
            );
            let q8 = Q8::geometry(k, 0);
            assert_eq!(
                registry_geometry(Q8::NAME, k as u64),
                (q8.stride as u64, vec![q8.codes as u64, q8.supers as u64])
            );
            let iq4 = Iq4::geometry(k, 0);
            assert_eq!(
                registry_geometry(Iq4::NAME, k as u64),
                (iq4.stride as u64, vec![iq4.codes as u64, iq4.supers as u64])
            );
        }
    }

    /// Every packed format decodes a row to exactly the registry's decode of
    /// the same bytes.
    #[test]
    fn packed_decode_matches_the_registry() {
        use seismic_lang::interp::TensorData;
        fn check<W: Format>(k: usize) {
            let geometry = W::geometry(k, 0);
            let rows = 3;
            let mut state = 0x2545_f491_4f6c_dd1du64 ^ k as u64;
            let mut data = (0..rows * geometry.stride)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    (state >> 24) as u8
                })
                .collect::<Vec<_>>();
            // Finite, moderate coefficients: f16 factors and f32 scales of
            // random bytes would be NaN or overflow.
            for row in 0..rows {
                let base = row * geometry.stride;
                let supers = geometry.stride - geometry.supers;
                let span = &mut data[base + geometry.supers..base + geometry.supers + supers];
                if W::NAME.starts_with("iq4") {
                    for (i, chunk) in span.chunks_exact_mut(4).enumerate() {
                        chunk.copy_from_slice(&(0.001 * (i % 7 + 1) as f32).to_le_bytes());
                    }
                } else {
                    for (i, chunk) in span.chunks_exact_mut(2).enumerate() {
                        chunk.copy_from_slice(&crate::element::f32_to_f16(0.002 * (i % 5 + 1) as f32).to_le_bytes());
                    }
                }
            }
            let id = representation(W::NAME).unwrap();
            let reference = TensorData::encoded(id, vec![rows, k], data.clone()).unwrap().values().unwrap();
            let mut out = vec![0.0f32; k];
            for row in 0..rows {
                unsafe { decode_row::<W>(data.as_ptr().add(row * geometry.stride), &geometry, k, &mut out) };
                for (i, value) in out.iter().enumerate() {
                    assert_eq!(
                        f64::from(*value),
                        reference[row * k + i],
                        "{} k {k} row {row} value {i}",
                        W::NAME
                    );
                }
            }
        }
        for k in [256usize, 512, 2560] {
            check::<Q4K>(k);
            check::<Q5K>(k);
            check::<Q6K>(k);
            check::<Q8>(k);
            check::<Iq4>(k);
        }
    }
}
