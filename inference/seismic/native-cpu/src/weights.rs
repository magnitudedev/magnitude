//! Weight elements a CPU weight operand can be bound to, their row
//! geometry, and the bodies of their components.
//!
//! A weight row of `k` logical values is read in packets of 32 consecutive
//! values. Dense rows (`f32`, `bf16`, `f16`) hold the values contiguously; a
//! partial final packet decodes its missing values as zero. Packed rows are
//! the registry's row layouts (`rows16`, `rows8`) with the registry's
//! geometry ([`RowGeometry::of`]), decoded by the components the registry's
//! decode axes name (`registry::decode_axes`, instantiated by build.rs): a
//! code component ([`Planes`], [`Grid`]) gives a packet's exact integer
//! values, and a coefficient component ([`Direct`], [`Hier`]) the scale (and
//! bias) of each of its coefficient groups. A value is `scale * value + bias`
//! (fused) or `scale * value`, rounded once to `f32` as the registry's
//! decode recipe rounds it.
//!
//! The dot bodies accumulate `w[i] * x[i]` (fused) into lane `i % LANES` in
//! index order and combine the lanes in the fixed tree of `reduce`, so a
//! result depends only on the values, never on the tier or the row block.
//! The quantized dot bodies sum exact integer products and combine them in
//! `f32` in the fixed order of their coefficient component, so they too
//! depend only on the values: the instructions forming the integer sums are
//! free to differ by tier.

use crate::element::{bf16_to_f32, f16_to_f32, Dense};
use crate::quant::{Q8Block, Q8_BLOCK};
use crate::reduce::{combine, LANES};
use magnitude_ggml::tables;
use seismic_lang::registry::{Layout, RepresentationInfo, RepresentationKind, RowPlaneRole};
use std::marker::PhantomData;

/// Values of one packet.
pub const PACKET: usize = 32;

/// Byte geometry of the rows of one weight operand.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RowGeometry {
    /// Bytes between the starts of adjacent rows.
    pub stride: usize,
    /// Per plane role (`RowPlaneRole::ALL` order), its offset within a row
    /// and its bytes per storage group; zero for a role the element lacks.
    pub offsets: [usize; 4],
    pub groups: [usize; 4],
    /// Row addressing of the bound tensor. Zero base is used for unbound geometry.
    pub base: usize,
    pub matrix_rows: usize,
    pub rows8: bool,
}

impl RowGeometry {
    /// The geometry of rows of `k` values of weight element `element` (dense,
    /// or packed `rows16` or `rows8`); `dense_stride` is the bound tensor's
    /// row stride in bytes, which only dense rows use. Packed rows follow the
    /// registry's row layout (`PackedRowLayout::geometry`), computed without
    /// allocating.
    pub fn of(element: &RepresentationInfo, k: usize, dense_stride: usize) -> Self {
        let mut geometry = Self {
            stride: dense_stride,
            offsets: [0; 4],
            groups: [0; 4],
            base: 0,
            matrix_rows: 0,
            rows8: false,
        };
        match &element.kind {
            RepresentationKind::Dense(_) => {}
            RepresentationKind::PackedRows(rows) => {
                let k = k as u64;
                geometry.stride = rows.row_stride_bytes(k).expect("a weight row's stride") as usize;
                for (plane, row_plane) in rows.planes.iter().enumerate() {
                    let role = row_plane.role() as usize;
                    geometry.offsets[role] =
                        rows.plane_row_offset(plane, k)
                            .expect("a weight plane's offset") as usize;
                    geometry.groups[role] = row_plane.bytes_per_group as usize;
                }
                geometry.rows8 = match element.layout {
                    Layout::Rows16 => false,
                    Layout::Rows8 => true,
                    other => panic!(
                        "CPU weight rows are rows16 or rows8, not {}",
                        other.as_str()
                    ),
                };
            }
            RepresentationKind::Packed(_) | RepresentationKind::External(_) => {
                panic!("`{}` is no CPU weight element", element.name)
            }
        }
        geometry
    }

    pub(crate) fn with_base(mut self, base: *const u8, matrix_rows: usize) -> Self {
        self.base = base as usize;
        self.matrix_rows = matrix_rows;
        self
    }

    /// The rows8 tile base and lane of the row at `row`.
    #[inline(always)]
    fn tile_lane(&self, row: *const u8) -> (usize, usize) {
        let index = (row as usize - self.base) / self.stride;
        let n = self.matrix_rows;
        let stored = index / n * n.div_ceil(8) * 8 + index % n;
        (self.base + stored / 8 * self.stride * 8, stored % 8)
    }

    /// The row at `row`, resolved once for the component call that reads
    /// it.
    #[inline(always)]
    pub(crate) fn row(&self, row: *const u8) -> Row {
        let (base, scale, lane, tile) = if self.rows8 {
            let (tile, lane) = self.tile_lane(row);
            (tile as *const u8, 8, lane, Some((tile, lane)))
        } else {
            (row, 1, 0, None)
        };
        let mut planes = [base; 4];
        let mut steps = self.groups;
        for plane in 0..4 {
            let (start, bytes) = (self.offsets[plane], self.groups[plane]);
            planes[plane] = base.wrapping_add(start * scale + lane * bytes);
            steps[plane] = bytes * scale;
        }
        Row {
            start: row,
            planes,
            steps,
            groups: self.groups,
            tile,
        }
    }
}

/// One row of a weight operand, resolved once per component call: where
/// each plane's storage groups lie, the same for every layout.
#[derive(Clone, Copy, Debug)]
pub struct Row {
    /// The row's first byte: a dense row's values.
    start: *const u8,
    /// Per plane role (`RowPlaneRole` order), the first byte of storage
    /// group 0 and the bytes from one storage group to the next; a group's
    /// bytes are contiguous in every layout.
    planes: [*const u8; 4],
    steps: [usize; 4],
    /// Per plane role, the bytes of a storage group: in a rows8 tile, from
    /// one lane's group to the next.
    groups: [usize; 4],
    /// The rows8 tile base and lane.
    tile: Option<(usize, usize)>,
}

impl Row {
    /// Whether eight rows are lanes 0..8 of one rows8 tile, in order: row
    /// `r`'s storage group of a plane is then row 0's plus `r` groups
    /// (`At::lane`).
    #[inline(always)]
    fn same_tile<const R: usize>(rows: &[Row; R]) -> bool {
        R == 8
            && rows[0].tile.is_some_and(|(base, lane)| {
                lane == 0 && (1..R).all(|r| rows[r].tile == Some((base, r)))
            })
    }

    /// The first byte of storage group `index` of every plane.
    #[inline(always)]
    fn group(&self, index: usize) -> [*const u8; 4] {
        let mut planes = self.planes;
        for plane in 0..4 {
            planes[plane] = planes[plane].wrapping_add(index * self.steps[plane]);
        }
        planes
    }
}

/// One weight element's component bodies.
pub trait Format: Copy + Send + Sync + 'static {
    /// The 32 values of packet `p` of `row`, on a tier of quantized-dot mode
    /// `MODE`.
    ///
    /// # Safety
    /// `row` is a row of at least `32 * p + 1` values.
    unsafe fn decode<const MODE: u8>(row: &Row, p: usize, k: usize, out: &mut [f32; PACKET]);

    /// Values `256 b .. 256 b + 256` (below `k`) of each of the `R` rows
    /// `rows` dotted with each of the `M` quantized activation blocks `x`: per row and activation block, the
    /// same exact integer sums and the same `f32` combine for every `R` and
    /// `M`.
    ///
    /// # Safety
    /// Each row is a row of `k` values, and `256 b < k`.
    unsafe fn dot_q8<const R: usize, const M: usize, const MODE: u8>(
        rows: &[Row; R],
        b: usize,
        k: usize,
        x: [&Q8Block; M],
    ) -> [[f32; M]; R];
}

/// Packets `0..count` of a block (`count` at most 8) as a loop bounded by
/// the constant 8, which the compiler unrolls.
#[inline(always)]
fn packets_of(count: usize) -> impl Iterator<Item = usize> {
    (0..Q8_BLOCK / PACKET).take_while(move |local| *local < count)
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
    #[inline(always)]
    unsafe fn decode<const MODE: u8>(row: &Row, p: usize, k: usize, out: &mut [f32; PACKET]) {
        let first = p * PACKET;
        // A whole packet widens in the tier's vectors; a row's tail and a
        // tier without them widen value by value.
        if k - first >= PACKET {
            let values = unsafe { row.start.cast::<E::Storage>().add(first) };
            #[cfg(target_arch = "x86_64")]
            if MODE >= 1 {
                for chunk in 0..PACKET / 8 {
                    unsafe {
                        let widened = E::widen_x86(values.add(8 * chunk));
                        std::arch::x86_64::_mm256_storeu_ps(
                            out.as_mut_ptr().add(8 * chunk),
                            widened,
                        );
                    }
                }
                return;
            }
            #[cfg(target_arch = "aarch64")]
            {
                for chunk in 0..PACKET / 8 {
                    unsafe {
                        let [low, high] = E::widen_neon(values.add(8 * chunk));
                        let out = out.as_mut_ptr().add(8 * chunk);
                        std::arch::aarch64::vst1q_f32(out, low);
                        std::arch::aarch64::vst1q_f32(out.add(4), high);
                    }
                }
                return;
            }
        }
        let valid = (k - first).min(PACKET);
        let values =
            unsafe { std::slice::from_raw_parts(row.start.cast::<E::Storage>().add(first), valid) };
        for (target, value) in out.iter_mut().zip(values) {
            *target = E::widen(*value);
        }
        for target in &mut out[valid..] {
            *target = 0.0;
        }
    }

    #[inline(always)]
    unsafe fn dot_q8<const R: usize, const M: usize, const MODE: u8>(
        rows: &[Row; R],
        b: usize,
        k: usize,
        x: [&Q8Block; M],
    ) -> [[f32; M]; R] {
        let first = b * Q8_BLOCK;
        let valid = (k - first).min(Q8_BLOCK);
        let mut out = [[0.0f32; M]; R];
        if valid == Q8_BLOCK {
            // A whole block: each packet widens once for every activation
            // block and accumulates in the tier's vectors, value `i` into lane
            // `i % LANES` in index order as below.
            for (row, out) in rows.iter().zip(&mut out) {
                let mut lanes = [[0.0f32; LANES]; M];
                for local in 0..Q8_BLOCK / PACKET {
                    let mut values = [0.0f32; PACKET];
                    unsafe { Self::decode::<MODE>(row, b * (Q8_BLOCK / PACKET) + local, k, &mut values) };
                    for (lanes, x) in lanes.iter_mut().zip(x.iter()) {
                        let mut codes = [0.0f32; PACKET];
                        for (code, x) in codes.iter_mut().zip(&x.codes[PACKET * local..]) {
                            *code = f32::from(*x);
                        }
                        unsafe { accumulate::<MODE>(lanes, &values, &codes) };
                    }
                }
                for ((out, lanes), x) in out.iter_mut().zip(lanes).zip(x.iter()) {
                    *out = x.d * combine(lanes);
                }
            }
            return out;
        }
        for (row, out) in rows.iter().zip(&mut out) {
            let values = unsafe {
                std::slice::from_raw_parts(row.start.cast::<E::Storage>().add(first), valid)
            };
            for (out, x) in out.iter_mut().zip(x.iter()) {
                let mut lanes = [0.0f32; LANES];
                for (i, (value, code)) in values.iter().zip(&x.codes).enumerate() {
                    lanes[i % LANES] = E::widen(*value).mul_add(f32::from(*code), lanes[i % LANES]);
                }
                *out = x.d * combine(lanes);
            }
        }
        out
    }
}

/// Where a packet lies: the first byte of its storage group in each plane
/// (`RowPlaneRole` order), resolved once per row and storage group, and the
/// packet within the group.
#[derive(Clone, Copy, Debug)]
pub struct At {
    planes: [*const u8; 4],
    pub packet: usize,
}

impl At {
    /// Packet 0 of storage group `index` of `row`.
    #[inline(always)]
    fn group(row: &Row, index: usize) -> Self {
        Self {
            planes: row.group(index),
            packet: 0,
        }
    }

    /// Packet `p` of `row`, of `STORAGE`-value storage groups.
    #[inline(always)]
    fn of<const STORAGE: usize>(row: &Row, p: usize) -> Self {
        let packets = STORAGE / PACKET;
        Self::group(row, p / packets).packet(p % packets)
    }

    /// Packet `packet` of the same storage group.
    #[inline(always)]
    fn packet(self, packet: usize) -> Self {
        Self { packet, ..self }
    }

    /// The same packet of lane `lane` of the rows8 tile whose lane 0 is
    /// `row`, this a packet of it.
    #[inline(always)]
    fn lane(self, lane: usize, row: &Row) -> Self {
        let mut at = self;
        for plane in 0..4 {
            at.planes[plane] = at.planes[plane].wrapping_add(lane * row.groups[plane]);
        }
        at
    }

    /// The storage group's first byte in plane `role`.
    #[inline(always)]
    fn plane(self, role: RowPlaneRole) -> *const u8 {
        self.planes[role as usize]
    }

    /// Coefficient group of `GROUP` values (within the storage group) of
    /// this packet's half `half`.
    #[inline(always)]
    fn local<const GROUP: usize>(self, half: usize) -> usize {
        (PACKET * self.packet + 16 * half) / GROUP
    }
}

/// The packets of activation block `b` of a row of `STORAGE`-value storage
/// groups (a divisor of the block): its first storage group and the bytes
/// from one group to the next in each plane.
#[derive(Clone, Copy, Debug)]
struct Block {
    first: At,
    steps: [usize; 4],
}

impl Block {
    #[inline(always)]
    fn of<const STORAGE: usize>(row: &Row, b: usize) -> Self {
        const {
            assert!(
                Q8_BLOCK % STORAGE == 0 && STORAGE % PACKET == 0,
                "storage groups divide a block"
            )
        };
        Self {
            first: At::group(row, b * (Q8_BLOCK / STORAGE)),
            steps: row.steps,
        }
    }

    /// Packet `local` of the block.
    #[inline(always)]
    fn at<const STORAGE: usize>(&self, local: usize) -> At {
        let packets = STORAGE / PACKET;
        let group = local / packets;
        let mut at = self.first.packet(local % packets);
        for plane in 0..4 {
            at.planes[plane] = at.planes[plane].wrapping_add(group * self.steps[plane]);
        }
        at
    }
}

/// The interpretations of codes (`registry::CodeInterpretation`): a raw
/// `bits`-bit code's exact integer value. Which interpretations a code plane
/// reads are the [`Codes`] implementations.
pub mod codes {
    /// The code itself.
    #[derive(Clone, Copy, Debug)]
    pub struct Unsigned;
    /// Two's complement.
    #[derive(Clone, Copy, Debug)]
    pub struct Twos;
    /// The code minus `C`.
    #[derive(Clone, Copy, Debug)]
    pub struct Offset<const C: i32>;
    /// The odd value `2 c + 1`.
    #[derive(Clone, Copy, Debug)]
    pub struct Odd;
    /// Value `c` of codebook `ID` (its position in `tables::CODEBOOKS`).
    #[derive(Clone, Copy, Debug)]
    pub struct Table<const ID: usize>;
}
use codes::{Odd, Offset, Table, Twos, Unsigned};

/// A raw code's exact integer value.
pub trait Interpretation: Copy + Send + Sync + 'static {
    fn value<const BITS: u32>(raw: u32) -> i32;
}

impl Interpretation for Unsigned {
    #[inline(always)]
    fn value<const BITS: u32>(raw: u32) -> i32 {
        raw as i32
    }
}

impl Interpretation for Twos {
    #[inline(always)]
    fn value<const BITS: u32>(raw: u32) -> i32 {
        ((raw << (32 - BITS)) as i32) >> (32 - BITS)
    }
}

impl<const C: i32> Interpretation for Offset<C> {
    #[inline(always)]
    fn value<const BITS: u32>(raw: u32) -> i32 {
        raw as i32 - C
    }
}

impl Interpretation for Odd {
    #[inline(always)]
    fn value<const BITS: u32>(raw: u32) -> i32 {
        2 * raw as i32 + 1
    }
}

impl<const ID: usize> Table<ID> {
    const VALUES: [i8; 16] = {
        let values = tables::CODEBOOKS[ID].values;
        assert!(values.len() == 16, "a 4-bit codebook");
        let mut out = [0i8; 16];
        let mut i = 0;
        while i < 16 {
            out[i] = values[i];
            i += 1;
        }
        out
    };
    const LEAST: i8 = {
        let mut least = Self::VALUES[0];
        let mut i = 1;
        while i < 16 {
            if Self::VALUES[i] < least {
                least = Self::VALUES[i];
            }
            i += 1;
        }
        least
    };
    const GREATEST: i8 = {
        let mut greatest = Self::VALUES[0];
        let mut i = 1;
        while i < 16 {
            if Self::VALUES[i] > greatest {
                greatest = Self::VALUES[i];
            }
            i += 1;
        }
        greatest
    };
    /// The codebook shifted to start at zero, as unsigned bytes.
    #[cfg(target_arch = "x86_64")]
    const SHIFTED: [i8; 16] = {
        let mut shifted = [0i8; 16];
        let mut i = 0;
        while i < 16 {
            shifted[i] = (Self::VALUES[i] as i32 - Self::LEAST as i32) as u8 as i8;
            i += 1;
        }
        shifted
    };
}

impl<const ID: usize> Interpretation for Table<ID> {
    #[inline(always)]
    fn value<const BITS: u32>(raw: u32) -> i32 {
        i32::from(Self::VALUES[raw as usize])
    }
}

/// A packet's 32 exact integer values: a code component.
pub trait Codes: Copy + Send + Sync + 'static {
    /// The least and the greatest value.
    const LEAST: i32;
    const GREATEST: i32;

    /// The values of packet `at`, in element order.
    ///
    /// # Safety
    /// `at` is a packet of a row.
    unsafe fn values(at: At) -> [i8; PACKET];

    /// The values in registers, values `0..16` then `16..32`.
    ///
    /// # Safety
    /// As [`Codes::values`].
    #[cfg(all(target_arch = "aarch64", target_feature = "dotprod"))]
    #[inline(always)]
    unsafe fn neon(at: At) -> [std::arch::aarch64::int8x16_t; 2] {
        use std::arch::aarch64::*;
        let values = unsafe { Self::values(at) };
        unsafe { [vld1q_s8(values.as_ptr()), vld1q_s8(values.as_ptr().add(16))] }
    }

    /// The values in one register.
    ///
    /// # Safety
    /// As [`Codes::values`]; the caller's tier has 256-bit integer vectors.
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn x86(at: At) -> std::arch::x86_64::__m256i {
        let values = unsafe { Self::values(at) };
        unsafe { packed_x86::load(values.as_ptr()) }
    }

    /// The values minus [`Codes::LEAST`] in one register, as unsigned bytes.
    ///
    /// # Safety
    /// As [`Codes::x86`].
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn x86_shifted(at: At) -> std::arch::x86_64::__m256i {
        use std::arch::x86_64::*;
        unsafe { _mm256_sub_epi8(Self::x86(at), _mm256_set1_epi8(Self::LEAST as i8)) }
    }
}

/// Codes in a `LOW`-bit low plane and a `HIGH`-bit high plane
/// (`registry::code_planes`), read through interpretation `I`: code `i` of
/// a packet is bits `LOW i ..` of the low plane (the low nibble of byte
/// `i / 2` first for 4-bit codes; byte `i` of an 8-bit plane) joined with
/// bits `HIGH i ..` of the high plane shifted to bit `LOW`.
#[derive(Clone, Copy, Debug)]
pub struct Planes<const LOW: u32, const HIGH: u32, I>(PhantomData<I>);

impl<const LOW: u32, const HIGH: u32, I> Planes<LOW, HIGH, I> {
    /// The packet's `4 LOW` low-plane bytes.
    #[inline(always)]
    unsafe fn low_bytes(at: At) -> *const u8 {
        unsafe {
            at.plane(RowPlaneRole::Codes)
                .add(4 * LOW as usize * at.packet)
        }
    }

    /// The packet's `4 HIGH` high-plane bytes.
    #[inline(always)]
    unsafe fn high_bytes(at: At) -> *const u8 {
        unsafe {
            at.plane(RowPlaneRole::High)
                .add(4 * HIGH as usize * at.packet)
        }
    }

    /// The packet's raw codes.
    #[inline(always)]
    unsafe fn raw(at: At) -> [u8; PACKET] {
        const {
            assert!(
                matches!(LOW, 2 | 4) && HIGH <= 2,
                "a 2- or 4-bit low plane and a high plane of at most 2 bits"
            )
        };
        let per_byte = (8 / LOW) as usize;
        let mask = (1u8 << LOW) - 1;
        let mut codes = [0u8; PACKET];
        let low = unsafe { Self::low_bytes(at) };
        for i in 0..PACKET / per_byte {
            let byte = unsafe { *low.add(i) };
            for j in 0..per_byte {
                codes[per_byte * i + j] = (byte >> (LOW as usize * j)) & mask;
            }
        }
        if HIGH > 0 {
            let high: u64 = match HIGH {
                1 => u64::from(unsafe { read::<u32>(Self::high_bytes(at)) }),
                _ => unsafe { read(Self::high_bytes(at)) },
            };
            let mask = (1u64 << HIGH) - 1;
            for (i, code) in codes.iter_mut().enumerate() {
                *code |= (((high >> (HIGH as usize * i)) & mask) as u8) << LOW;
            }
        }
        codes
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "dotprod"))]
    #[inline(always)]
    unsafe fn raw_neon(at: At) -> [std::arch::aarch64::int8x16_t; 2] {
        const { assert!(HIGH <= 1 || LOW == 4, "a 2-bit high plane joins a nibble low plane") };
        let low = unsafe { Self::low_bytes(at) };
        match HIGH {
            0 => unsafe { packed_neon::codes::<LOW>(low, 0) },
            1 => unsafe { packed_neon::codes::<LOW>(low, read(Self::high_bytes(at))) },
            _ => unsafe { packed_neon::nibbles2(low, Self::high_bytes(at)) },
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn raw_x86(at: At) -> std::arch::x86_64::__m256i {
        const { assert!(HIGH <= 1 || LOW == 4, "a 2-bit high plane joins a nibble low plane") };
        let low = unsafe { Self::low_bytes(at) };
        match HIGH {
            0 => unsafe { packed_x86::codes::<LOW>(low) },
            1 => unsafe { packed_x86::codes1::<LOW>(low, read(Self::high_bytes(at))) },
            _ => unsafe { packed_x86::nibbles2(low, Self::high_bytes(at)) },
        }
    }
}

impl<const LOW: u32, const HIGH: u32> Codes for Planes<LOW, HIGH, Unsigned> {
    const LEAST: i32 = 0;
    const GREATEST: i32 = (1 << (LOW + HIGH)) - 1;

    #[inline(always)]
    unsafe fn values(at: At) -> [i8; PACKET] {
        let raw = unsafe { Self::raw(at) };
        let mut values = [0i8; PACKET];
        for i in 0..PACKET {
            values[i] = raw[i] as i8;
        }
        values
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "dotprod"))]
    #[inline(always)]
    unsafe fn neon(at: At) -> [std::arch::aarch64::int8x16_t; 2] {
        unsafe { Self::raw_neon(at) }
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn x86(at: At) -> std::arch::x86_64::__m256i {
        unsafe { Self::raw_x86(at) }
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn x86_shifted(at: At) -> std::arch::x86_64::__m256i {
        unsafe { Self::raw_x86(at) }
    }
}

impl<const LOW: u32, const HIGH: u32, const C: i32> Codes for Planes<LOW, HIGH, Offset<C>> {
    const LEAST: i32 = {
        assert!(C >= 0 && C <= 128, "values fit a signed byte");
        -C
    };
    const GREATEST: i32 = (1 << (LOW + HIGH)) - 1 - C;

    #[inline(always)]
    unsafe fn values(at: At) -> [i8; PACKET] {
        let raw = unsafe { Self::raw(at) };
        let mut values = [0i8; PACKET];
        for i in 0..PACKET {
            values[i] = (i32::from(raw[i]) + Self::LEAST) as i8;
        }
        values
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "dotprod"))]
    #[inline(always)]
    unsafe fn neon(at: At) -> [std::arch::aarch64::int8x16_t; 2] {
        use std::arch::aarch64::*;
        let offset = unsafe { vdupq_n_s8(-Self::LEAST as i8) };
        unsafe { Self::raw_neon(at) }.map(|codes| unsafe { vsubq_s8(codes, offset) })
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn x86(at: At) -> std::arch::x86_64::__m256i {
        use std::arch::x86_64::*;
        let offset = _mm256_set1_epi8(-Self::LEAST as i8);
        unsafe { _mm256_sub_epi8(Self::raw_x86(at), offset) }
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn x86_shifted(at: At) -> std::arch::x86_64::__m256i {
        unsafe { Self::raw_x86(at) }
    }
}

impl<const ID: usize> Codes for Planes<4, 0, Table<ID>> {
    const LEAST: i32 = Table::<ID>::LEAST as i32;
    const GREATEST: i32 = Table::<ID>::GREATEST as i32;

    #[inline(always)]
    unsafe fn values(at: At) -> [i8; PACKET] {
        let raw = unsafe { Self::raw(at) };
        let mut values = [0i8; PACKET];
        for i in 0..PACKET {
            values[i] = Table::<ID>::value::<4>(u32::from(raw[i])) as i8;
        }
        values
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "dotprod"))]
    #[inline(always)]
    unsafe fn neon(at: At) -> [std::arch::aarch64::int8x16_t; 2] {
        use std::arch::aarch64::*;
        let table = unsafe { vld1q_s8(Table::<ID>::VALUES.as_ptr()) };
        unsafe { Self::raw_neon(at) }
            .map(|codes| unsafe { vqtbl1q_s8(table, vreinterpretq_u8_s8(codes)) })
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn x86(at: At) -> std::arch::x86_64::__m256i {
        unsafe { packed_x86::lookup(packed_x86::table(&Table::<ID>::VALUES), Self::raw_x86(at)) }
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn x86_shifted(at: At) -> std::arch::x86_64::__m256i {
        unsafe { packed_x86::lookup(packed_x86::table(&Table::<ID>::SHIFTED), Self::raw_x86(at)) }
    }
}

impl Codes for Planes<8, 0, Twos> {
    const LEAST: i32 = i8::MIN as i32;
    const GREATEST: i32 = i8::MAX as i32;

    #[inline(always)]
    unsafe fn values(at: At) -> [i8; PACKET] {
        unsafe { read(Self::bytes(at)) }
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "dotprod"))]
    #[inline(always)]
    unsafe fn neon(at: At) -> [std::arch::aarch64::int8x16_t; 2] {
        use std::arch::aarch64::*;
        let bytes = unsafe { Self::bytes(at) }.cast::<i8>();
        unsafe { [vld1q_s8(bytes), vld1q_s8(bytes.add(16))] }
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn x86(at: At) -> std::arch::x86_64::__m256i {
        unsafe { packed_x86::load(Self::bytes(at).cast()) }
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn x86_shifted(at: At) -> std::arch::x86_64::__m256i {
        use std::arch::x86_64::*;
        unsafe { _mm256_xor_si256(Self::x86(at), _mm256_set1_epi8(i8::MIN)) }
    }
}

impl Planes<8, 0, Twos> {
    /// The packet's 32 code bytes.
    #[inline(always)]
    unsafe fn bytes(at: At) -> *const u8 {
        unsafe { at.plane(RowPlaneRole::Codes).add(PACKET * at.packet) }
    }
}

/// Grid codes (`registry::GridCodes`): `BYTES`-byte entries of grid `GRID`
/// (its position in `tables::GRIDS`), one per eight values, each holding
/// `8 / lanes` `INDEX_BITS`-bit point indices, first lowest, then seven sign
/// bits completed by an eighth of even parity. The value at lane `j` of an
/// entry is byte `j % lanes` of its point `j / lanes`, negated by sign `j`.
#[derive(Clone, Copy, Debug)]
pub struct Grid<const GRID: usize, const INDEX_BITS: u32, const BYTES: usize>;

impl<const GRID: usize, const INDEX_BITS: u32, const BYTES: usize> Grid<GRID, INDEX_BITS, BYTES> {
    const POINTS: tables::GridPoints = tables::GRIDS[GRID].points;
    /// The greatest magnitude of a point lane.
    const MAGNITUDE: i32 = {
        let mut greatest = 0u8;
        let mut index = 0;
        while index < Self::POINTS.len() {
            let mut lane = 0;
            while lane < Self::POINTS.lanes() {
                let byte = Self::POINTS.byte(index, lane);
                if byte > greatest {
                    greatest = byte;
                }
                lane += 1;
            }
            index += 1;
        }
        assert!(
            greatest <= i8::MAX as u8,
            "grid magnitudes fit a signed byte"
        );
        greatest as i32
    };

    /// Entry `e` of the storage group of `at`.
    #[inline(always)]
    unsafe fn entry(at: At, e: usize) -> u32 {
        let at = unsafe { at.plane(RowPlaneRole::Codes).add(BYTES * e) };
        (0..BYTES).fold(0, |entry, byte| {
            entry | (u32::from(unsafe { *at.add(byte) }) << (8 * byte))
        })
    }
}

/// The eight values one grid entry gives: `point`'s magnitudes (byte `j` the
/// lane-`j` magnitude), negated where the seven sign bits `signs` and their
/// even-parity eighth are set. Bit `j` of the signs becomes byte `j` of a
/// mask; a magnitude is below 128, so `(m ^ 0xff) + 1 = -m` bytewise without
/// carries.
#[inline(always)]
fn grid_values(point: u64, signs: u32) -> [i8; 8] {
    const ONES: u64 = 0x0101_0101_0101_0101;
    let signs = u64::from(signs | ((signs.count_ones() & 1) << 7));
    let spread = signs.wrapping_mul(ONES) & 0x8040_2010_0804_0201;
    let negative = (spread.wrapping_add(0x7f7f_7f7f_7f7f_7f7f) >> 7) & ONES;
    ((point ^ (negative * 0xff)).wrapping_add(negative))
        .to_le_bytes()
        .map(|byte| byte as i8)
}

impl<const GRID: usize, const INDEX_BITS: u32, const BYTES: usize> Codes
    for Grid<GRID, INDEX_BITS, BYTES>
{
    const LEAST: i32 = -Self::MAGNITUDE;
    const GREATEST: i32 = Self::MAGNITUDE;

    #[inline(always)]
    unsafe fn values(at: At) -> [i8; PACKET] {
        let mask = (1u32 << INDEX_BITS) - 1;
        let mut values = [0i8; PACKET];
        for e in 0..PACKET / 8 {
            let entry = unsafe { Self::entry(at, 4 * at.packet + e) };
            let (point, signs) = match Self::POINTS {
                tables::GridPoints::Lanes8(points) => {
                    (points[(entry & mask) as usize], entry >> INDEX_BITS)
                }
                tables::GridPoints::Lanes4(points) => (
                    u64::from(points[(entry & mask) as usize])
                        | (u64::from(points[((entry >> INDEX_BITS) & mask) as usize]) << 32),
                    entry >> (2 * INDEX_BITS),
                ),
            };
            values[8 * e..8 * e + 8].copy_from_slice(&grid_values(point, signs & 127));
        }
        values
    }
}

/// The encodings of a stored scale, bias or factor
/// (`registry::ScaleEncoding`), each read as its exact value.
pub mod scales {
    #[derive(Clone, Copy, Debug)]
    pub struct F16;
    #[derive(Clone, Copy, Debug)]
    pub struct Bf16;
    #[derive(Clone, Copy, Debug)]
    pub struct F32;
    /// OCP E8M0: `2^(e - 127)`; 0xff is NaN.
    #[derive(Clone, Copy, Debug)]
    pub struct E8m0;
    /// NVIDIA unsigned E4M3 (the sign bit ignored); 0x7f is NaN.
    #[derive(Clone, Copy, Debug)]
    pub struct Ue4m3;
}

/// A stored scale's encoding.
pub trait Scale: Copy + Send + Sync + 'static {
    const BYTES: usize;

    /// The scale stored at `at`.
    ///
    /// # Safety
    /// `at` addresses `BYTES` bytes.
    unsafe fn read(at: *const u8) -> f32;

    /// The eight scales at `first`, `stride` bytes apart, one a lane, each
    /// exactly as [`Scale::read`] reads it.
    ///
    /// # Safety
    /// Each addresses `BYTES` bytes; the caller's tier has 256-bit vectors
    /// and FMA.
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn read8(first: *const u8, stride: usize) -> std::arch::x86_64::__m256 {
        let scales = unsafe { read8_each::<Self>(first, stride) };
        unsafe { std::arch::x86_64::_mm256_loadu_ps(scales.as_ptr()) }
    }

    /// [`Scale::read8`]'s NEON form: lanes 0..4 and 4..8.
    ///
    /// # Safety
    /// Each addresses `BYTES` bytes.
    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    unsafe fn read8_neon(
        first: *const u8,
        stride: usize,
    ) -> [std::arch::aarch64::float32x4_t; 2] {
        use std::arch::aarch64::*;
        let scales = unsafe { read8_each::<Self>(first, stride) };
        unsafe { [vld1q_f32(scales.as_ptr()), vld1q_f32(scales.as_ptr().add(4))] }
    }
}

/// The eight scales of [`Scale::read8`], each read alone.
///
/// # Safety
/// As [`Scale::read8_neon`].
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[inline(always)]
unsafe fn read8_each<S: Scale>(first: *const u8, stride: usize) -> [f32; 8] {
    let mut scales = [0.0f32; 8];
    for (r, scale) in scales.iter_mut().enumerate() {
        *scale = unsafe { S::read(first.add(r * stride)) };
    }
    scales
}

impl scales::F16 {
    /// The eight stored words at `first`, `stride` bytes apart.
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    #[inline(always)]
    unsafe fn words8(first: *const u8, stride: usize) -> [u16; 8] {
        let mut words = [0u16; 8];
        for (r, word) in words.iter_mut().enumerate() {
            *word = unsafe { read(first.add(r * stride)) };
        }
        words
    }
}

/// Eight scales widen as one vector (`Dense::widen_*`), straight from the
/// plane where they are adjacent.
impl Scale for scales::F16 {
    const BYTES: usize = 2;
    #[inline(always)]
    unsafe fn read(at: *const u8) -> f32 {
        f16_to_f32(unsafe { read(at) })
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn read8(first: *const u8, stride: usize) -> std::arch::x86_64::__m256 {
        type Widen = crate::element::F16;
        if stride == Self::BYTES {
            return unsafe { Widen::widen_x86(first.cast()) };
        }
        let words = unsafe { Self::words8(first, stride) };
        unsafe { Widen::widen_x86(words.as_ptr()) }
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    unsafe fn read8_neon(
        first: *const u8,
        stride: usize,
    ) -> [std::arch::aarch64::float32x4_t; 2] {
        type Widen = crate::element::F16;
        if stride == Self::BYTES {
            return unsafe { Widen::widen_neon(first.cast()) };
        }
        let words = unsafe { Self::words8(first, stride) };
        unsafe { Widen::widen_neon(words.as_ptr()) }
    }
}

impl Scale for scales::Bf16 {
    const BYTES: usize = 2;
    #[inline(always)]
    unsafe fn read(at: *const u8) -> f32 {
        bf16_to_f32(unsafe { read(at) })
    }
}

impl Scale for scales::F32 {
    const BYTES: usize = 4;
    #[inline(always)]
    unsafe fn read(at: *const u8) -> f32 {
        f32::from_bits(unsafe { read(at) })
    }
}

impl Scale for scales::E8m0 {
    const BYTES: usize = 1;
    #[inline(always)]
    unsafe fn read(at: *const u8) -> f32 {
        match unsafe { *at } {
            0 => f32::from_bits(0x0040_0000),
            0xff => f32::NAN,
            exponent => f32::from_bits(u32::from(exponent) << 23),
        }
    }
}

impl Scale for scales::Ue4m3 {
    const BYTES: usize = 1;
    #[inline(always)]
    unsafe fn read(at: *const u8) -> f32 {
        let raw = unsafe { *at } & 0x7f;
        let (exponent, mantissa) = (u32::from(raw >> 3), u32::from(raw & 7));
        if raw == 0x7f {
            f32::NAN
        } else if exponent == 0 {
            // mantissa * 2^-9.
            mantissa as f32 * f32::from_bits(0x3b00_0000)
        } else {
            f32::from_bits(((exponent + 120) << 23) | (mantissa << 20))
        }
    }
}

/// `2^exponent`, exactly.
const fn power2(exponent: i32) -> f32 {
    assert!(-126 <= exponent && exponent <= 127, "a normal power of two");
    f32::from_bits(((127 + exponent) as u32) << 23)
}

/// `value * 2^EXPONENT`, exact for every value that stays normal; the value
/// itself at exponent 0.
#[inline(always)]
fn scaled<const EXPONENT: i32>(value: f32) -> f32 {
    if EXPONENT == 0 {
        value
    } else {
        value * const { power2(EXPONENT) }
    }
}

/// A packet's coefficients: a coefficient component.
pub trait Coefficients: Copy + Send + Sync + 'static {
    /// Whether a value is `scale * value + bias` (fused) rather than
    /// `scale * value`.
    const BIASED: bool;

    /// The (scale, bias) of values `0..16` and `16..32` of packet `at` of
    /// rows of `STORAGE`-value storage groups.
    ///
    /// # Safety
    /// `at` is a packet of a row.
    unsafe fn decode<const STORAGE: usize>(at: At) -> [(f32, f32); 2];

    /// [`Format::dot_q8`] of rows whose codes are `C`, over the first
    /// `packets` packets of block `b`.
    ///
    /// # Safety
    /// As [`Format::dot_q8`], with `packets` the block's packets below `k`.
    unsafe fn dot_q8<
        C: Codes,
        const STORAGE: usize,
        const R: usize,
        const M: usize,
        const MODE: u8,
    >(
        rows: &[Row; R],
        b: usize,
        packets: usize,
        x: [&Q8Block; M],
    ) -> [[f32; M]; R];
}

/// The activation code sum of values `16 h .. 16 h + 16` of packet `local`
/// of `x`'s block, or with `whole` of the whole packet.
#[inline(always)]
fn activation_sum(x: &Q8Block, local: usize, half: usize, whole: bool) -> i32 {
    if whole {
        i32::from(x.sums[2 * local]) + i32::from(x.sums[2 * local + 1])
    } else {
        i32::from(x.sums[2 * local + half])
    }
}

/// Coefficient groups of a packet under `GROUP`-value groups: 1 or 2.
const fn groups_per_packet(group: usize) -> usize {
    assert!(
        group == 16 || group % PACKET == 0,
        "coefficient groups of 16 or whole packets"
    );
    if group == 16 {
        2
    } else {
        1
    }
}

/// Whether x86 dots take a code component's values as unsigned bytes rather
/// than their magnitudes against sign-adjusted activations: when they are
/// unsigned within the unsigned byte products (`limit`: 128 for the AVX2
/// pair sums, 255 for VNNI's).
#[cfg(target_arch = "x86_64")]
const fn unsigned<C: Codes>(limit: i32) -> bool {
    C::LEAST >= 0 && C::GREATEST <= limit
}

/// Whether x86 dots take a code component's values shifted to unsigned
/// bytes and add back its least value times the activation sums: when the
/// span fits the unsigned byte products (as [`unsigned`]), for a dot whose
/// one correction serves several rows.
#[cfg(target_arch = "x86_64")]
const fn shifted<C: Codes>(limit: i32) -> bool {
    C::GREATEST - C::LEAST <= limit
}

/// The exact integer dots of the first `packets` packets of `block` with
/// each activation block of `x`: `[m][h][local]`, per `GROUP`-value
/// coefficient group of a packet (the whole packet in `h = 0` when a group
/// covers it), as `f32`: a dot of 32 products of at most 128 x 255 is below
/// 2^24 in magnitude, so its `f32` is exact, and it converts in the vectors
/// that reduce it.
#[inline(always)]
unsafe fn packet_dots<
    C: Codes,
    const STORAGE: usize,
    const M: usize,
    const GROUP: usize,
    const MODE: u8,
>(
    block: &Block,
    packets: usize,
    x: [&Q8Block; M],
) -> [[[f32; 8]; 2]; M] {
    let halves = const { groups_per_packet(GROUP) };
    let mut exact = [[[0.0f32; 8]; 2]; M];
    #[cfg(all(target_arch = "aarch64", target_feature = "dotprod"))]
    {
        use std::arch::aarch64::*;
        // A packet's products are four lanes; pairwise adds reduce four
        // packets' into one vector of their sums, which stays a vector (no
        // lane-by-lane reduction into general registers).
        let zero = unsafe { vdupq_n_s32(0) };
        let mut unpaired = [[zero; 2]; M];
        let mut pairs = [[[zero; 4]; 2]; M];
        for local in packets_of(packets) {
            let weights = unsafe { C::neon(block.at::<STORAGE>(local)) };
            for m in 0..M {
                let activations = x[m].codes[PACKET * local..].as_ptr();
                let (first, second) =
                    unsafe { (vld1q_s8(activations), vld1q_s8(activations.add(16))) };
                let products = if halves == 2 {
                    [
                        packed_neon::sdot(zero, weights[0], first),
                        packed_neon::sdot(zero, weights[1], second),
                    ]
                } else {
                    let products = packed_neon::sdot(
                        packed_neon::sdot(zero, weights[0], first),
                        weights[1],
                        second,
                    );
                    [products, zero]
                };
                for h in 0..halves {
                    if local % 2 == 0 {
                        unpaired[m][h] = products[h];
                    } else {
                        pairs[m][h][local / 2] = unsafe { vpaddq_s32(unpaired[m][h], products[h]) };
                    }
                }
            }
        }
        for m in 0..M {
            for h in 0..halves {
                if packets % 2 == 1 {
                    pairs[m][h][packets / 2] = unsafe { vpaddq_s32(unpaired[m][h], zero) };
                }
                let pairs = pairs[m][h];
                unsafe {
                    let quads = [vpaddq_s32(pairs[0], pairs[1]), vpaddq_s32(pairs[2], pairs[3])];
                    vst1q_f32(exact[m][h].as_mut_ptr(), vcvtq_f32_s32(quads[0]));
                    vst1q_f32(exact[m][h].as_mut_ptr().add(4), vcvtq_f32_s32(quads[1]));
                }
            }
        }
        return exact;
    }
    #[cfg(target_arch = "x86_64")]
    if MODE >= 1 {
        unsafe {
            use std::arch::x86_64::*;
            let unsigned = unsigned::<C>(if MODE == 2 { 255 } else { 128 });
            // Per activation block, a block's eight packets reduce together:
            // the sums of their low and high halves (values 0..16, 16..32).
            for m in 0..M {
                let mut products = [_mm256_setzero_si256(); 8];
                for local in packets_of(packets) {
                    let weights = C::x86(block.at::<STORAGE>(local));
                    products[local] = packed_x86::product::<MODE>(weights, x[m], local, unsigned);
                }
                let (first, second) = packed_x86::halves8(products);
                if halves == 1 {
                    let whole = _mm256_add_epi32(first, second);
                    _mm256_storeu_ps(exact[m][0].as_mut_ptr(), _mm256_cvtepi32_ps(whole));
                } else {
                    _mm256_storeu_ps(exact[m][0].as_mut_ptr(), _mm256_cvtepi32_ps(first));
                    _mm256_storeu_ps(exact[m][1].as_mut_ptr(), _mm256_cvtepi32_ps(second));
                }
            }
        }
        return exact;
    }
    #[allow(unreachable_code)]
    {
        for local in packets_of(packets) {
            let values = unsafe { C::values(block.at::<STORAGE>(local)) };
            let dots = scalar_dots::<M, GROUP, MODE>(&values, x, local);
            for m in 0..M {
                for h in 0..halves {
                    exact[m][h][local] = dots[m][h] as f32;
                }
            }
        }
        exact
    }
}

/// The exact integer dots of one packet's `values` with packet `local` of
/// each activation block of `x`, per `GROUP`-value coefficient group (the
/// whole packet in `[0]` when a group covers it), by [`dot_i8_i8`].
#[inline(always)]
fn scalar_dots<const M: usize, const GROUP: usize, const MODE: u8>(
    values: &[i8; PACKET],
    x: [&Q8Block; M],
    local: usize,
) -> [[i32; 2]; M] {
    let halves = const { groups_per_packet(GROUP) };
    let mut exact = [[0i32; 2]; M];
    for (exact, x) in exact.iter_mut().zip(x.iter()) {
        let activations: &[i8; PACKET] = x.codes[PACKET * local..PACKET * (local + 1)]
            .try_into()
            .expect("a packet of codes");
        for (h, exact) in exact.iter_mut().enumerate().take(halves) {
            *exact = if halves == 1 {
                dot_i8_i8::<MODE>(values, activations)
            } else {
                // Each sixteen-value group alone: the other half is zero.
                let mut half = [0i8; PACKET];
                half[16 * h..16 * h + 16].copy_from_slice(&values[16 * h..16 * h + 16]);
                dot_i8_i8::<MODE>(&half, activations)
            };
        }
    }
    exact
}

/// Direct coefficients (`CoefficientAxes::Direct`): per `GROUP` values a
/// stored scale of encoding `S` (and with `BIAS` a bias), the scales of a
/// storage group then its biases in the supers plane; values are scaled by
/// `2^CODE_EXPONENT`, folded into each scale exactly. A quantized dot sums
/// each group's exact integer products and accumulates `scale * products`
/// (and `bias * activation sum`), fused, in group order.
#[derive(Clone, Copy, Debug)]
pub struct Direct<S, const GROUP: usize, const BIAS: bool, const CODE_EXPONENT: i32>(
    PhantomData<S>,
);

impl<S: Scale, const GROUP: usize, const BIAS: bool, const CODE_EXPONENT: i32>
    Direct<S, GROUP, BIAS, CODE_EXPONENT>
{
    /// [`Coefficients::dot_q8`] of the eight rows of one rows8 tile
    /// (`Row::same_tile`) against one activation block on x86: per
    /// packet the eight rows' exact sums of a group in one vector, a row a
    /// lane, and each lane's combine a fused multiply-add in group order, as
    /// a row's own.
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn dot_q8_eight<
        C: Codes,
        const STORAGE: usize,
        const R: usize,
        const M: usize,
        const MODE: u8,
    >(
        rows: &[Row; R],
        b: usize,
        packets: usize,
        x: [&Q8Block; M],
    ) -> [[f32; M]; R] {
        use std::arch::x86_64::*;
        let halves = const { groups_per_packet(GROUP) };
        let x = x[0];
        let shift = shifted::<C>(if MODE == 2 { 255 } else { 128 });
        let block = Block::of::<STORAGE>(&rows[0], b);
        let mut sums = unsafe { _mm256_setzero_ps() };
        for local in packets_of(packets) {
            let activations = unsafe { packed_x86::load(x.codes.as_ptr().add(PACKET * local)) };
            let at = block.at::<STORAGE>(local);
            let mut products = [unsafe { _mm256_setzero_si256() }; 8];
            for (r, products) in products.iter_mut().enumerate() {
                let at = at.lane(r, &rows[0]);
                *products = unsafe {
                    if shift {
                        packed_x86::dot::<MODE>(*products, C::x86_shifted(at), activations)
                    } else {
                        packed_x86::dot_signed::<MODE>(*products, C::x86(at), activations)
                    }
                };
            }
            let exact = unsafe {
                if halves == 1 {
                    [packed_x86::sums8(products), _mm256_setzero_si256()]
                } else {
                    let (first, second) = packed_x86::halves8(products);
                    [first, second]
                }
            };
            for h in 0..halves {
                let codes = activation_sum(x, local, h, halves == 1);
                let mut exact = exact[h];
                if shift && C::LEAST != 0 {
                    exact = unsafe { _mm256_add_epi32(exact, _mm256_set1_epi32(C::LEAST * codes)) };
                }
                // Lane r's fields are a supers group further on.
                let (fields, stride) = (
                    at.plane(RowPlaneRole::Supers),
                    rows[0].groups[RowPlaneRole::Supers as usize],
                );
                let local = at.local::<GROUP>(h);
                let scales = unsafe { S::read8(fields.add(S::BYTES * local), stride) };
                let scales = if CODE_EXPONENT == 0 {
                    scales
                } else {
                    unsafe {
                        _mm256_mul_ps(scales, _mm256_set1_ps(const { power2(CODE_EXPONENT) }))
                    }
                };
                sums = unsafe { _mm256_fmadd_ps(scales, _mm256_cvtepi32_ps(exact), sums) };
                if BIAS {
                    let biases = unsafe {
                        S::read8(fields.add(S::BYTES * (STORAGE / GROUP + local)), stride)
                    };
                    sums = unsafe { _mm256_fmadd_ps(biases, _mm256_set1_ps(codes as f32), sums) };
                }
            }
        }
        let mut lanes = [0.0f32; 8];
        unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), sums) };
        let mut out = [[0.0f32; M]; R];
        for r in 0..R {
            out[r][0] = x.d * lanes[r];
        }
        out
    }

    /// [`Coefficients::dot_q8`] of rows against four activation blocks on
    /// x86: per packet the four blocks' exact sums of a group in one vector,
    /// a block a lane, and each lane's combine a fused multiply-add in group
    /// order, as a block's own.
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn dot_q8_four<
        C: Codes,
        const STORAGE: usize,
        const R: usize,
        const M: usize,
        const MODE: u8,
    >(
        rows: &[Row; R],
        b: usize,
        packets: usize,
        x: [&Q8Block; M],
    ) -> [[f32; M]; R] {
        use std::arch::x86_64::*;
        let halves = const { groups_per_packet(GROUP) };
        let unsigned = unsigned::<C>(if MODE == 2 { 255 } else { 128 });
        let mut out = [[0.0f32; M]; R];
        for (out, row) in out.iter_mut().zip(rows) {
            let block = Block::of::<STORAGE>(row, b);
            let mut sums = unsafe { _mm_setzero_ps() };
            for local in packets_of(packets) {
                let at = block.at::<STORAGE>(local);
                let weights = unsafe { C::x86(at) };
                let mut products = [unsafe { _mm256_setzero_si256() }; 4];
                for (products, x) in products.iter_mut().zip(x.iter()) {
                    *products = unsafe { packed_x86::product::<MODE>(weights, x, local, unsigned) };
                }
                // Low and high halves of the four blocks' sums.
                let halves4 = unsafe { packed_x86::halves4(products) };
                let (low, high) = unsafe {
                    (
                        _mm256_castsi256_si128(halves4),
                        _mm256_extracti128_si256::<1>(halves4),
                    )
                };
                let exact = if halves == 1 {
                    [unsafe { _mm_add_epi32(low, high) }, low]
                } else {
                    [low, high]
                };
                let coefficients = unsafe { Self::decode::<STORAGE>(at) };
                for h in 0..halves {
                    let (exact, (scale, bias)) = (exact[h], coefficients[h]);
                    let mut codes = [0i32; 4];
                    for m in 0..4 {
                        codes[m] = activation_sum(x[m], local, h, halves == 1);
                    }
                    let codes = unsafe { _mm_loadu_si128(codes.as_ptr().cast()) };
                    sums =
                        unsafe { _mm_fmadd_ps(_mm_set1_ps(scale), _mm_cvtepi32_ps(exact), sums) };
                    if BIAS {
                        sums = unsafe {
                            _mm_fmadd_ps(_mm_set1_ps(bias), _mm_cvtepi32_ps(codes), sums)
                        };
                    }
                }
            }
            let mut lanes = [0.0f32; 4];
            unsafe { _mm_storeu_ps(lanes.as_mut_ptr(), sums) };
            for ((out, lane), x) in out.iter_mut().zip(lanes).zip(x.iter()) {
                *out = x.d * lane;
            }
        }
        out
    }

    /// The (scale, bias) of coefficient group `local` of the storage group
    /// whose supers fields are at `fields`.
    #[inline(always)]
    unsafe fn group<const STORAGE: usize>(fields: *const u8, local: usize) -> (f32, f32) {
        let scale = scaled::<CODE_EXPONENT>(unsafe { S::read(fields.add(S::BYTES * local)) });
        let bias = if BIAS {
            unsafe { S::read(fields.add(S::BYTES * (STORAGE / GROUP + local))) }
        } else {
            0.0
        };
        (scale, bias)
    }
}

impl<S: Scale, const GROUP: usize, const BIAS: bool, const CODE_EXPONENT: i32> Coefficients
    for Direct<S, GROUP, BIAS, CODE_EXPONENT>
{
    const BIASED: bool = BIAS;

    #[inline(always)]
    unsafe fn decode<const STORAGE: usize>(at: At) -> [(f32, f32); 2] {
        let fields = at.plane(RowPlaneRole::Supers);
        let first = unsafe { Self::group::<STORAGE>(fields, at.local::<GROUP>(0)) };
        if GROUP == 16 {
            [first, unsafe {
                Self::group::<STORAGE>(fields, at.local::<GROUP>(1))
            }]
        } else {
            [first, first]
        }
    }

    #[inline(always)]
    unsafe fn dot_q8<
        C: Codes,
        const STORAGE: usize,
        const R: usize,
        const M: usize,
        const MODE: u8,
    >(
        rows: &[Row; R],
        b: usize,
        packets: usize,
        x: [&Q8Block; M],
    ) -> [[f32; M]; R] {
        let halves = const { groups_per_packet(GROUP) };
        #[cfg(target_arch = "x86_64")]
        if M == 1 && MODE >= 1 && Row::same_tile(rows) {
            return unsafe { Self::dot_q8_eight::<C, STORAGE, R, M, MODE>(rows, b, packets, x) };
        }
        #[cfg(target_arch = "x86_64")]
        if M == 4 && MODE >= 1 {
            return unsafe { Self::dot_q8_four::<C, STORAGE, R, M, MODE>(rows, b, packets, x) };
        }
        let mut out = [[0.0f32; M]; R];
        for (out, row) in out.iter_mut().zip(rows) {
            let block = Block::of::<STORAGE>(row, b);
            let exact = unsafe { packet_dots::<C, STORAGE, M, GROUP, MODE>(&block, packets, x) };
            let mut sums = [0.0f32; M];
            for local in packets_of(packets) {
                let coefficients = unsafe { Self::decode::<STORAGE>(block.at::<STORAGE>(local)) };
                for m in 0..M {
                    for h in 0..halves {
                        let (scale, bias) = coefficients[h];
                        sums[m] = scale.mul_add(exact[m][h][local], sums[m]);
                        if BIAS {
                            let codes = activation_sum(x[m], local, h, halves == 1);
                            sums[m] = bias.mul_add(codes as f32, sums[m]);
                        }
                    }
                }
            }
            for ((out, sum), x) in out.iter_mut().zip(sums).zip(x.iter()) {
                *out = x.d * sum;
            }
        }
        out
    }
}

/// The local codes of hierarchical coefficients: per coefficient group a
/// scale code (and with a minimum a minimum code).
pub trait Locals: Copy + Send + Sync + 'static {
    /// The sign of the minimum's term in a value's bias: -1 or 1, or 0
    /// without a minimum.
    const MIN_SIGN: i32;

    /// The (scale, minimum) codes' values of `GROUP`-value coefficient group
    /// `local` of the storage group of `at`; the minimum is 0 without one.
    ///
    /// # Safety
    /// `at` is a packet of a row.
    unsafe fn read<const GROUP: usize>(at: At, local: usize) -> (i32, i32);

    /// The codes of every packet of the 256-value storage group of `at`,
    /// per packet and half (a group's codes in both halves of a packet it
    /// covers): what a quantized dot reads once per row and block.
    ///
    /// # Safety
    /// As [`Locals::read`].
    #[inline(always)]
    unsafe fn block<const GROUP: usize>(at: At) -> [[(i32, i32); 2]; 8] {
        let mut codes = [[(0, 0); 2]; 8];
        for (packet, codes) in codes.iter_mut().enumerate() {
            for (half, codes) in codes.iter_mut().enumerate() {
                *codes = unsafe { Self::read::<GROUP>(at, at.packet(packet).local::<GROUP>(half)) };
            }
        }
        codes
    }
}

/// Local codes in the scales plane: per coefficient group a `BITS`-bit
/// scale code (then, with a minimum of sign `MIN_SIGN`, a minimum code) read
/// through `L`, group `j`'s at bit `j BITS (1 + minimum)` of its storage
/// group's bytes.
#[derive(Clone, Copy, Debug)]
pub struct Scales<L, const BITS: u32, const MIN_SIGN: i32>(PhantomData<L>);

impl<L: Interpretation, const BITS: u32, const MIN_SIGN: i32> Scales<L, BITS, MIN_SIGN> {
    /// Whether the groups have a minimum code.
    const MIN: bool = {
        assert!(MIN_SIGN.abs() <= 1, "a minimum's sign is -1, 0 or 1");
        MIN_SIGN != 0
    };

    /// Bits of one group's codes; every group's lie within two bytes.
    const WIDTH: usize = {
        let width = BITS as usize * (1 + Self::MIN as usize);
        let mut group = 0;
        while group < 8 {
            assert!(
                (group * width) % 8 + width <= 16,
                "a group's codes lie within two bytes"
            );
            group += 1;
        }
        width
    };

    /// The (scale, minimum) codes' values of the field at bit `bit % 8` of
    /// the low bits of `bytes`.
    #[inline(always)]
    fn codes(bytes: u32, bit: usize) -> (i32, i32) {
        Self::split(bytes >> (bit % 8))
    }

    /// The (scale, minimum) codes' values of the field in the low bits of
    /// `field`.
    #[inline(always)]
    fn split(field: u32) -> (i32, i32) {
        let mask = (1u32 << BITS) - 1;
        let minimum = if Self::MIN {
            L::value::<BITS>((field >> BITS) & mask)
        } else {
            0
        };
        (L::value::<BITS>(field & mask), minimum)
    }
}

impl<L: Interpretation, const BITS: u32, const MIN_SIGN: i32> Locals for Scales<L, BITS, MIN_SIGN> {
    const MIN_SIGN: i32 = MIN_SIGN;

    #[inline(always)]
    unsafe fn block<const GROUP: usize>(at: At) -> [[(i32, i32); 2]; 8] {
        // The storage group's fields as one word, read once, and each
        // coefficient group's codes from it once.
        let bytes = const {
            let bytes = (Q8_BLOCK / GROUP * Self::WIDTH).div_ceil(8);
            assert!(bytes <= 16, "a storage group's fields fit 16 bytes");
            bytes
        };
        let mut fields = [0u8; 16];
        unsafe {
            std::ptr::copy_nonoverlapping(
                at.plane(RowPlaneRole::Scales),
                fields.as_mut_ptr(),
                bytes,
            )
        };
        let fields = u128::from_le_bytes(fields);
        let mut groups = [(0, 0); 16];
        for (group, codes) in groups.iter_mut().enumerate().take(Q8_BLOCK / GROUP) {
            *codes = Self::split((fields >> (group * Self::WIDTH)) as u32);
        }
        let mut codes = [[(0, 0); 2]; 8];
        for (packet, codes) in codes.iter_mut().enumerate() {
            for (half, codes) in codes.iter_mut().enumerate() {
                *codes = groups[at.packet(packet).local::<GROUP>(half)];
            }
        }
        codes
    }

    #[inline(always)]
    unsafe fn read<const GROUP: usize>(at: At, local: usize) -> (i32, i32) {
        let bit = local * Self::WIDTH;
        let bytes = unsafe { at.plane(RowPlaneRole::Scales).add(bit / 8) };
        let mut pair = u32::from(unsafe { *bytes });
        if bit % 8 + Self::WIDTH > 8 {
            pair |= u32::from(unsafe { *bytes.add(1) }) << 8;
        }
        Self::codes(pair, bit)
    }
}

/// Local scale codes carried by grid entries of `ENTRY_BYTES` bytes: bit
/// `j` of a group's `BITS`-bit code at bit `BIT` of the group's entry `j`,
/// read through `L`.
#[derive(Clone, Copy, Debug)]
pub struct Carried<const BIT: u32, const ENTRY_BYTES: usize, L, const BITS: u32>(PhantomData<L>);

impl<const BIT: u32, const ENTRY_BYTES: usize, L: Interpretation, const BITS: u32> Locals
    for Carried<BIT, ENTRY_BYTES, L, BITS>
{
    const MIN_SIGN: i32 = 0;

    #[inline(always)]
    unsafe fn read<const GROUP: usize>(at: At, local: usize) -> (i32, i32) {
        let entries = unsafe {
            at.plane(RowPlaneRole::Codes)
                .add(ENTRY_BYTES * (GROUP / 8) * local)
        };
        let (byte, bit) = (BIT as usize / 8, BIT % 8);
        let code = (0..BITS as usize).fold(0u32, |code, j| {
            code | (((u32::from(unsafe { *entries.add(ENTRY_BYTES * j + byte) }) >> bit) & 1) << j)
        });
        (L::value::<BITS>(code), 0)
    }
}

/// Hierarchical coefficients (`CoefficientAxes::Hierarchical`): per storage
/// group (the factor group) a factor of encoding `F` (then with a minimum a
/// minimum factor) in the supers plane, scaled by `2^EXPONENT`; per `GROUP`
/// values local codes `L`. A value's scale is `factor * 2^CODE_EXPONENT *
/// scale code` and its bias `L::MIN_SIGN * minimum factor * minimum code`. A
/// quantized dot sums `scale code * products` (and `minimum code *
/// activation sums`) over the factor group exactly, then combines
/// `factor * scaled ± minimum factor * minimums` once.
#[derive(Clone, Copy, Debug)]
pub struct Hier<F, const EXPONENT: i32, const CODE_EXPONENT: i32, L, const GROUP: usize>(
    PhantomData<(F, L)>,
);

impl<F: Scale, const EXPONENT: i32, const CODE_EXPONENT: i32, L: Locals, const GROUP: usize>
    Hier<F, EXPONENT, CODE_EXPONENT, L, GROUP>
{
    /// Whether the coefficients have a minimum.
    const MIN: bool = L::MIN_SIGN != 0;

    /// The scale factor (`factor * 2^(EXPONENT + CODE_EXPONENT)`) and the
    /// minimum factor (the stored minimum factor times `2^EXPONENT`) of the
    /// storage group of `at`.
    #[inline(always)]
    unsafe fn factors(at: At) -> (f32, f32) {
        let supers = at.plane(RowPlaneRole::Supers);
        let minimum = if Self::MIN {
            unsafe { F::read(supers.add(F::BYTES)) }
        } else {
            0.0
        };
        Self::factors_of(unsafe { F::read(supers) }, minimum)
    }

    /// [`Hier::factors`] of the stored factor and minimum factor.
    #[inline(always)]
    fn factors_of(factor: f32, minimum: f32) -> (f32, f32) {
        let minimum = if Self::MIN {
            scaled::<EXPONENT>(minimum)
        } else {
            0.0
        };
        (scaled::<CODE_EXPONENT>(scaled::<EXPONENT>(factor)), minimum)
    }

    /// [`Hier::factors`] of the eight lanes of the rows8 tile whose lane 0
    /// is `row`, `group` its storage group: lane `r`'s fields are `r` supers
    /// groups further on, so each field's eight read as one vector.
    #[cfg(any(
        all(target_arch = "aarch64", target_feature = "dotprod"),
        target_arch = "x86_64"
    ))]
    #[inline(always)]
    unsafe fn tile_factors(row: &Row, group: At) -> [(f32, f32); 8] {
        let (supers, stride) = (
            group.plane(RowPlaneRole::Supers),
            row.groups[RowPlaneRole::Supers as usize],
        );
        let mut fields = [[0.0f32; 8]; 2];
        for (field, values) in fields.iter_mut().enumerate().take(1 + Self::MIN as usize) {
            let first = unsafe { supers.add(F::BYTES * field) };
            #[cfg(target_arch = "aarch64")]
            unsafe {
                use std::arch::aarch64::*;
                let [low, high] = F::read8_neon(first, stride);
                vst1q_f32(values.as_mut_ptr(), low);
                vst1q_f32(values.as_mut_ptr().add(4), high);
            }
            #[cfg(target_arch = "x86_64")]
            unsafe {
                std::arch::x86_64::_mm256_storeu_ps(values.as_mut_ptr(), F::read8(first, stride));
            }
        }
        let mut factors = [(0.0, 0.0); 8];
        for (r, factors) in factors.iter_mut().enumerate() {
            *factors = Self::factors_of(fields[0][r], fields[1][r]);
        }
        factors
    }

    /// The scale codes of packet `at`'s halves (its one group's in both
    /// with one group a packet), read where used.
    ///
    /// # Safety
    /// As [`Locals::read`].
    #[inline(always)]
    unsafe fn packet_scales(at: At) -> [i32; 2] {
        let first = unsafe { L::read::<GROUP>(at, at.local::<GROUP>(0)).0 };
        let second = if groups_per_packet(GROUP) == 1 {
            first
        } else {
            unsafe { L::read::<GROUP>(at, at.local::<GROUP>(1)).0 }
        };
        [first, second]
    }

    /// One row's [`Coefficients::dot_q8`] over the block's first
    /// `packets` packets, with the activation blocks' code sums `sums`.
    #[inline(always)]
    unsafe fn dot_row<C: Codes, const M: usize, const MODE: u8>(
        row: &Row,
        b: usize,
        packets: usize,
        x: [&Q8Block; M],
        sums: &[[[i32; 2]; 8]; M],
    ) -> [f32; M] {
        let halves = const { groups_per_packet(GROUP) };
        let whole = halves == 1;
        let group = At::group(row, b);
        // The block's codes, read once: the minimum terms', and the scale
        // codes that x86 broadcasts from memory (a load operand). NEON reads
        // each packet's scale codes where used instead: its broadcast from a
        // general register replaces the block's round trip through memory.
        let codes = unsafe { L::block::<GROUP>(group) };
        let mut scaled = [0i32; M];
        let mut minimums = [0i32; M];
        if Self::MIN {
            for m in 0..M {
                for local in packets_of(packets) {
                    for h in 0..halves {
                        minimums[m] += codes[local][h].1 * sums[m][local][h];
                    }
                }
            }
        }
        #[cfg(all(target_arch = "aarch64", target_feature = "dotprod"))]
        unsafe {
            use std::arch::aarch64::*;
            // Each half accumulates in its own lanes: one multiply-add per
            // half, which the compiler cannot reassociate into a multiply,
            // a multiply-add and an add.
            let mut lanes = [[vdupq_n_s32(0); 2]; M];
            for local in packets_of(packets) {
                let weights = C::neon(group.packet(local));
                let [first_code, second_code] = Self::packet_scales(group.packet(local));
                let zero = vdupq_n_s32(0);
                for (lanes, x) in lanes.iter_mut().zip(x.iter()) {
                    let activations = x.codes[PACKET * local..].as_ptr();
                    let (first, second) = (vld1q_s8(activations), vld1q_s8(activations.add(16)));
                    if whole {
                        let products = packed_neon::sdot(
                            packed_neon::sdot(zero, weights[0], first),
                            weights[1],
                            second,
                        );
                        lanes[0] = vmlaq_n_s32(lanes[0], products, first_code);
                    } else {
                        lanes[0] = vmlaq_n_s32(
                            lanes[0],
                            packed_neon::sdot(zero, weights[0], first),
                            first_code,
                        );
                        lanes[1] = vmlaq_n_s32(
                            lanes[1],
                            packed_neon::sdot(zero, weights[1], second),
                            second_code,
                        );
                    }
                }
            }
            for (scaled, [first, second]) in scaled.iter_mut().zip(lanes) {
                *scaled += vaddvq_s32(vaddq_s32(first, second));
            }
        }
        #[cfg(target_arch = "x86_64")]
        if MODE >= 1 {
            unsafe {
                use std::arch::x86_64::*;
                // The pair products take one 16-bit scale code per half.
                let unsigned = unsigned::<C>(128);
                let mut lanes = [_mm256_setzero_si256(); M];
                for local in packets_of(packets) {
                    let (first, second) = (codes[local][0].0, codes[local][1].0);
                    let scales = if whole {
                        _mm256_set1_epi16(first as i16)
                    } else {
                        _mm256_set_m128i(
                            _mm_set1_epi16(second as i16),
                            _mm_set1_epi16(first as i16),
                        )
                    };
                    let weights = C::x86(group.packet(local));
                    let magnitudes = _mm256_abs_epi8(weights);
                    for m in 0..M {
                        let activations = packed_x86::load(x[m].codes.as_ptr().add(PACKET * local));
                        lanes[m] = if unsigned {
                            packed_x86::dot_scaled(lanes[m], weights, activations, scales)
                        } else {
                            packed_x86::dot_scaled(
                                lanes[m],
                                magnitudes,
                                _mm256_sign_epi8(activations, weights),
                                scales,
                            )
                        };
                    }
                }
                for (scaled, lanes) in scaled.iter_mut().zip(lanes) {
                    *scaled += packed_x86::sum(lanes);
                }
            }
        }
        #[cfg(not(all(target_arch = "aarch64", target_feature = "dotprod")))]
        if !(cfg!(target_arch = "x86_64") && MODE >= 1) {
            for local in packets_of(packets) {
                let values = unsafe { C::values(group.packet(local)) };
                let exact = scalar_dots::<M, GROUP, MODE>(&values, x, local);
                for m in 0..M {
                    for h in 0..halves {
                        scaled[m] += codes[local][h].0 * exact[m][h];
                    }
                }
            }
        }
        let factors = unsafe { Self::factors(group) };
        let mut out = [0.0f32; M];
        for (((out, scaled), minimums), x) in out.iter_mut().zip(scaled).zip(minimums).zip(x.iter())
        {
            *out = x.d * Self::combine(factors, scaled, minimums);
        }
        out
    }

    /// The eight rows of one rows8 tile (`Row::same_tile`, lane 0's
    /// at `row`) against one activation block, interleaved per packet so an
    /// activation packet loads once for the eight; each row's arithmetic is
    /// [`Hier::dot_row`]'s.
    #[cfg(any(
        all(target_arch = "aarch64", target_feature = "dotprod"),
        target_arch = "x86_64"
    ))]
    #[inline(always)]
    unsafe fn dot_tile<C: Codes, const MODE: u8>(
        row: &Row,
        b: usize,
        packets: usize,
        x: &Q8Block,
        sums: &[[i32; 2]; 8],
    ) -> [f32; 8] {
        let halves = const { groups_per_packet(GROUP) };
        let whole = halves == 1;
        let group = At::group(row, b);
        // Each row's minimum term; its scale codes are read where used.
        let mut minimums = [0i32; 8];
        if Self::MIN {
            for r in 0..8 {
                let codes = unsafe { L::block::<GROUP>(group.lane(r, row)) };
                for local in packets_of(packets) {
                    for h in 0..halves {
                        minimums[r] += codes[local][h].1 * sums[local][h];
                    }
                }
            }
        }
        let mut scaled = [0i32; 8];
        #[cfg(all(target_arch = "aarch64", target_feature = "dotprod"))]
        unsafe {
            use std::arch::aarch64::*;
            let mut lanes = [vdupq_n_s32(0); 8];
            for local in packets_of(packets) {
                let activations = x.codes[PACKET * local..].as_ptr();
                let (first, second) = (vld1q_s8(activations), vld1q_s8(activations.add(16)));
                let at = group.packet(local);
                let zero = vdupq_n_s32(0);
                for r in 0..8 {
                    let at = at.lane(r, row);
                    let weights = C::neon(at);
                    let [first_scale, second_scale] = Self::packet_scales(at);
                    if whole {
                        let products = packed_neon::sdot(
                            packed_neon::sdot(zero, weights[0], first),
                            weights[1],
                            second,
                        );
                        lanes[r] = vmlaq_n_s32(lanes[r], products, first_scale);
                    } else {
                        lanes[r] = vmlaq_n_s32(
                            lanes[r],
                            packed_neon::sdot(zero, weights[0], first),
                            first_scale,
                        );
                        lanes[r] = vmlaq_n_s32(
                            lanes[r],
                            packed_neon::sdot(zero, weights[1], second),
                            second_scale,
                        );
                    }
                }
            }
            for r in 0..8 {
                scaled[r] = vaddvq_s32(lanes[r]);
            }
        }
        #[cfg(target_arch = "x86_64")]
        unsafe {
            use std::arch::x86_64::*;
            let unsigned = unsigned::<C>(128);
            let mut lanes = [_mm256_setzero_si256(); 8];
            for local in packets_of(packets) {
                let activations = packed_x86::load(x.codes.as_ptr().add(PACKET * local));
                let at = group.packet(local);
                for r in 0..8 {
                    let at_r = at.lane(r, row);
                    let [first, second] = Self::packet_scales(at_r);
                    let scales = if whole {
                        _mm256_set1_epi16(first as i16)
                    } else {
                        _mm256_set_m128i(
                            _mm_set1_epi16(second as i16),
                            _mm_set1_epi16(first as i16),
                        )
                    };
                    let weights = C::x86(at_r);
                    lanes[r] = if unsigned {
                        packed_x86::dot_scaled(lanes[r], weights, activations, scales)
                    } else {
                        packed_x86::dot_scaled(
                            lanes[r],
                            _mm256_abs_epi8(weights),
                            _mm256_sign_epi8(activations, weights),
                            scales,
                        )
                    };
                }
            }
            // The eight rows' lane sums together.
            let mut sums = [0i32; 8];
            _mm256_storeu_si256(sums.as_mut_ptr().cast(), packed_x86::sums8(lanes));
            for r in 0..8 {
                scaled[r] += sums[r];
            }
        }
        let factors = unsafe { Self::tile_factors(row, group) };
        let mut out = [0.0f32; 8];
        for r in 0..8 {
            out[r] = x.d * Self::combine(factors[r], scaled[r], minimums[r]);
        }
        out
    }

    /// [`Coefficients::dot_q8`] over the block's first `packets` packets.
    #[inline(always)]
    unsafe fn dot_block<C: Codes, const R: usize, const M: usize, const MODE: u8>(
        rows: &[Row; R],
        b: usize,
        packets: usize,
        x: [&Q8Block; M],
    ) -> [[f32; M]; R] {
        let whole = const { groups_per_packet(GROUP) } == 1;
        // Each activation block's code sums, per packet and half (the whole
        // packet in both halves with one group).
        let mut sums = [[[0i32; 2]; 8]; M];
        for (sums, x) in sums.iter_mut().zip(x.iter()) {
            for (local, sums) in sums.iter_mut().enumerate().take(packets) {
                for (h, sum) in sums.iter_mut().enumerate() {
                    *sum = activation_sum(x, local, h, whole);
                }
            }
        }
        #[cfg(any(
            all(target_arch = "aarch64", target_feature = "dotprod"),
            target_arch = "x86_64"
        ))]
        if M == 1 && (cfg!(target_arch = "aarch64") || MODE >= 1) && Row::same_tile(rows) {
            let tile = unsafe { Self::dot_tile::<C, MODE>(&rows[0], b, packets, x[0], &sums[0]) };
            let mut out = [[0.0f32; M]; R];
            for r in 0..R {
                out[r][0] = tile[r];
            }
            return out;
        }
        // Row by row, each whole.
        let mut out = [[0.0f32; M]; R];
        for (out, row) in out.iter_mut().zip(rows) {
            *out = unsafe { Self::dot_row::<C, M, MODE>(row, b, packets, x, &sums) };
        }
        out
    }

    /// The (scale, bias) of a group of codes `(scale, minimum)` under
    /// `factors`.
    #[inline(always)]
    fn coefficient(factors: (f32, f32), (scale, minimum): (i32, i32)) -> (f32, f32) {
        let bias = if Self::MIN {
            let bias = factors.1 * minimum as f32;
            if L::MIN_SIGN < 0 {
                -bias
            } else {
                bias
            }
        } else {
            0.0
        };
        (factors.0 * scale as f32, bias)
    }

    /// `factor * scaled + L::MIN_SIGN * minimum * minimums`, the minimum
    /// term's product rounded on its own.
    #[inline(always)]
    fn combine(factors: (f32, f32), scaled: i32, minimums: i32) -> f32 {
        let scaled = factors.0 * scaled as f32;
        if !Self::MIN {
            scaled
        } else if L::MIN_SIGN < 0 {
            scaled - factors.1 * minimums as f32
        } else {
            scaled + factors.1 * minimums as f32
        }
    }
}

impl<F: Scale, const EXPONENT: i32, const CODE_EXPONENT: i32, L: Locals, const GROUP: usize>
    Coefficients for Hier<F, EXPONENT, CODE_EXPONENT, L, GROUP>
{
    const BIASED: bool = Self::MIN;

    #[inline(always)]
    unsafe fn decode<const STORAGE: usize>(at: At) -> [(f32, f32); 2] {
        let factors = unsafe { Self::factors(at) };
        let first = unsafe { L::read::<GROUP>(at, at.local::<GROUP>(0)) };
        let first = Self::coefficient(factors, first);
        if GROUP == 16 {
            let second = unsafe { L::read::<GROUP>(at, at.local::<GROUP>(1)) };
            [first, Self::coefficient(factors, second)]
        } else {
            [first, first]
        }
    }

    #[inline(always)]
    unsafe fn dot_q8<
        C: Codes,
        const STORAGE: usize,
        const R: usize,
        const M: usize,
        const MODE: u8,
    >(
        rows: &[Row; R],
        b: usize,
        packets: usize,
        x: [&Q8Block; M],
    ) -> [[f32; M]; R] {
        const {
            assert!(
                STORAGE == Q8_BLOCK,
                "a factor group is one activation block"
            )
        };
        // A factor group is the whole block. Every block but a row's last
        // has all its packets: its own instance bounds the packet loops by
        // a constant, which unrolls the block into registers.
        let full = Q8_BLOCK / PACKET;
        if packets == full {
            unsafe { Self::dot_block::<C, R, M, MODE>(rows, b, full, x) }
        } else {
            unsafe { Self::dot_block::<C, R, M, MODE>(rows, b, packets, x) }
        }
    }
}

/// Packed rows of `STORAGE`-value storage groups whose packets' values are
/// codes `C` scaled by coefficients `K`.
#[derive(Clone, Copy, Debug)]
pub struct PackedRows<C, K, const STORAGE: usize>(PhantomData<(C, K)>);

impl<C: Codes, K: Coefficients, const STORAGE: usize> Format for PackedRows<C, K, STORAGE> {
    #[inline(always)]
    unsafe fn decode<const MODE: u8>(row: &Row, p: usize, _k: usize, out: &mut [f32; PACKET]) {
        const { assert!(STORAGE % PACKET == 0, "storage groups of whole packets") };
        let at = At::of::<STORAGE>(row, p);
        let coefficients = unsafe { K::decode::<STORAGE>(at) };
        #[cfg(all(target_arch = "aarch64", target_feature = "dotprod"))]
        unsafe {
            let values = C::neon(at);
            for half in 0..2 {
                let (scale, bias) = coefficients[half];
                let out = out.as_mut_ptr().add(16 * half);
                if K::BIASED {
                    packed_neon::affine_store(values[half], scale, bias, out);
                } else {
                    packed_neon::scaled_store(values[half], scale, out);
                }
            }
        }
        #[cfg(not(all(target_arch = "aarch64", target_feature = "dotprod")))]
        {
            let values = unsafe { C::values(at) };
            for half in 0..2 {
                let (scale, bias) = coefficients[half];
                for i in 16 * half..16 * half + 16 {
                    out[i] = if K::BIASED {
                        scale.mul_add(f32::from(values[i]), bias)
                    } else {
                        scale * f32::from(values[i])
                    };
                }
            }
        }
    }

    #[inline(always)]
    unsafe fn dot_q8<const R: usize, const M: usize, const MODE: u8>(
        rows: &[Row; R],
        b: usize,
        k: usize,
        x: [&Q8Block; M],
    ) -> [[f32; M]; R] {
        let packets = block_packets(b, k).len();
        unsafe { K::dot_q8::<C, STORAGE, R, M, MODE>(rows, b, packets, x) }
    }
}

/// Register-resident packet dots for targets whose baseline includes the
/// dot-product instructions (every Apple silicon target). A packet's values
/// are formed in two vectors in element order and never leave registers.
#[cfg(all(target_arch = "aarch64", target_feature = "dotprod"))]
mod packed_neon {
    use std::arch::aarch64::*;

    /// `acc` plus the four-way signed byte dot products of `a` and `b`.
    #[inline(always)]
    pub(super) fn sdot(mut acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
        // Rust's dot-product intrinsic is not stable yet; the target
        // baseline guarantees the instruction.
        unsafe {
            std::arch::asm!(
                "sdot {acc:v}.4s, {a:v}.16b, {b:v}.16b",
                acc = inout(vreg) acc,
                a = in(vreg) a,
                b = in(vreg) b,
                options(pure, nomem, nostack, preserves_flags),
            );
        }
        acc
    }

    /// The 32-value dot of two signed packets.
    ///
    /// # Safety
    /// Both pointers address 32 values.
    #[inline(always)]
    pub(super) unsafe fn dot32(weights: *const i8, activations: *const i8) -> i32 {
        unsafe {
            let zero = vdupq_n_s32(0);
            let first = sdot(zero, vld1q_s8(weights), vld1q_s8(activations));
            vaddvq_s32(sdot(
                first,
                vld1q_s8(weights.add(16)),
                vld1q_s8(activations.add(16)),
            ))
        }
    }

    /// Sixteen values widened to F32.
    #[inline(always)]
    fn widen(values: int8x16_t) -> [float32x4_t; 4] {
        unsafe {
            let (low, high) = (
                vmovl_s8(vget_low_s8(values)),
                vmovl_s8(vget_high_s8(values)),
            );
            [
                vcvtq_f32_s32(vmovl_s16(vget_low_s16(low))),
                vcvtq_f32_s32(vmovl_s16(vget_high_s16(low))),
                vcvtq_f32_s32(vmovl_s16(vget_low_s16(high))),
                vcvtq_f32_s32(vmovl_s16(vget_high_s16(high))),
            ]
        }
    }

    /// `out[i] = scale * values[i] + bias`, fused, for sixteen values.
    ///
    /// # Safety
    /// `out` addresses 16 values.
    #[inline(always)]
    pub(super) unsafe fn affine_store(values: int8x16_t, scale: f32, bias: f32, out: *mut f32) {
        unsafe {
            let (scale, bias) = (vdupq_n_f32(scale), vdupq_n_f32(bias));
            for (quarter, values) in widen(values).into_iter().enumerate() {
                vst1q_f32(out.add(4 * quarter), vfmaq_f32(bias, values, scale));
            }
        }
    }

    /// `out[i] = scale * values[i]` for sixteen values.
    ///
    /// # Safety
    /// `out` addresses 16 values.
    #[inline(always)]
    pub(super) unsafe fn scaled_store(values: int8x16_t, scale: f32, out: *mut f32) {
        unsafe {
            let scale = vdupq_n_f32(scale);
            for (quarter, values) in widen(values).into_iter().enumerate() {
                vst1q_f32(out.add(4 * quarter), vmulq_f32(scale, values));
            }
        }
    }

    /// Bit `i` of `mask` as the value `2^LOW` in byte `i`.
    #[inline(always)]
    fn high_bits<const LOW: u32>(mask: u16) -> uint8x16_t {
        const BITS: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
        unsafe {
            let bytes = vcombine_u8(vdup_n_u8(mask as u8), vdup_n_u8((mask >> 8) as u8));
            vandq_u8(vtstq_u8(bytes, vld1q_u8(BITS.as_ptr())), vdupq_n_u8(1 << LOW))
        }
    }

    /// A packet's codes of a `LOW`-bit low plane (`4 LOW` bytes) in element
    /// order: code `(8 / LOW) i + j` is bits `LOW j ..` of byte `i`, joined
    /// with its bit of `high`, a 1-bit high plane's packet word (0 for none),
    /// at bit `LOW`. Codes are at most 31, so they are also signed bytes.
    ///
    /// # Safety
    /// `bytes` addresses the packet's `4 LOW` code bytes.
    #[inline(always)]
    pub(super) unsafe fn codes<const LOW: u32>(bytes: *const u8, high: u32) -> [int8x16_t; 2] {
        const { assert!(LOW == 2 || LOW == 4, "a 2- or 4-bit low plane") };
        unsafe {
            let (first, second) = if LOW == 4 {
                let packed = vld1q_u8(bytes);
                let low = vandq_u8(packed, vdupq_n_u8(15));
                let upper = vshrq_n_u8(packed, 4);
                (vzip1q_u8(low, upper), vzip2q_u8(low, upper))
            } else {
                // Codes 4i..4i + 4 of byte i: its crumbs at bits 0, 2, 4, 6.
                let packed = vld1_u8(bytes);
                let three = vdup_n_u8(3);
                let crumbs = [
                    vand_u8(packed, three),
                    vand_u8(vshr_n_u8(packed, 2), three),
                    vand_u8(vshr_n_u8(packed, 4), three),
                    vshr_n_u8(packed, 6),
                ];
                let even = vcombine_u8(vzip1_u8(crumbs[0], crumbs[2]), vzip2_u8(crumbs[0], crumbs[2]));
                let odd = vcombine_u8(vzip1_u8(crumbs[1], crumbs[3]), vzip2_u8(crumbs[1], crumbs[3]));
                (vzip1q_u8(even, odd), vzip2q_u8(even, odd))
            };
            let (first, second) = if high != 0 {
                (
                    vorrq_u8(first, high_bits::<LOW>(high as u16)),
                    vorrq_u8(second, high_bits::<LOW>((high >> 16) as u16)),
                )
            } else {
                (first, second)
            };
            [vreinterpretq_s8_u8(first), vreinterpretq_s8_u8(second)]
        }
    }

    /// A packet's codes in element order under a 2-bit high plane: codes 2i
    /// and 2i + 1 join the low and high nibble of byte i with bits 0-1 and
    /// 2-3 of the high plane's nibble i. Codes are at most 63.
    ///
    /// # Safety
    /// `codes` addresses the packet's 16 code bytes and `high` its 8 high
    /// bytes.
    #[inline(always)]
    pub(super) unsafe fn nibbles2(codes: *const u8, high: *const u8) -> [int8x16_t; 2] {
        unsafe {
            let packed = vld1q_u8(codes);
            let high = vld1_u8(high);
            let (low_nibbles, high_nibbles) = (vand_u8(high, vdup_n_u8(15)), vshr_n_u8(high, 4));
            let nibbles = vcombine_u8(
                vzip1_u8(low_nibbles, high_nibbles),
                vzip2_u8(low_nibbles, high_nibbles),
            );
            let even = vorrq_u8(
                vandq_u8(packed, vdupq_n_u8(15)),
                vshlq_n_u8(vandq_u8(nibbles, vdupq_n_u8(3)), 4),
            );
            let odd = vorrq_u8(vshrq_n_u8(packed, 4), vshlq_n_u8(vshrq_n_u8(nibbles, 2), 4));
            [
                vreinterpretq_s8_u8(vzip1q_u8(even, odd)),
                vreinterpretq_s8_u8(vzip2q_u8(even, odd)),
            ]
        }
    }
}

/// Register-resident packet dots for the x86 tiers with 256-bit integer
/// vectors: mode 1 (AVX2) and mode 2 (AVX-512 VNNI on 256-bit registers).
/// A packet's 32 values form one vector in element order and its products
/// stay in eight `i32` lanes until the caller's block ends, so integer sums
/// are exact and independent of the lane order. Only components of a tier
/// whose features cover the mode reach these bodies; they inline into them.
///
/// Weight bytes passed as unsigned are at most 128 (255 to mode 2's `dot`)
/// and activations are quantization codes in `[-127, 127]`, so the AVX2 pair
/// sums of 16-bit products never saturate (`2 * 128 * 127 < 2^15`).
#[cfg(target_arch = "x86_64")]
mod packed_x86 {
    use std::arch::x86_64::*;

    /// 32 bytes at `bytes`.
    #[inline(always)]
    pub(super) unsafe fn load(bytes: *const i8) -> __m256i {
        unsafe { _mm256_loadu_si256(bytes.cast()) }
    }

    /// `acc` plus, per 32-bit lane, the four products of unsigned weight
    /// bytes `weights` and signed activation bytes `activations`.
    #[inline(always)]
    pub(super) unsafe fn dot<const MODE: u8>(
        acc: __m256i,
        weights: __m256i,
        activations: __m256i,
    ) -> __m256i {
        unsafe {
            if MODE == 2 {
                _mm256_dpbusd_epi32(acc, weights, activations)
            } else {
                let pairs = _mm256_maddubs_epi16(weights, activations);
                _mm256_add_epi32(acc, _mm256_madd_epi16(pairs, _mm256_set1_epi16(1)))
            }
        }
    }

    /// `acc` plus the four-way products of signed weights and activations:
    /// `|w| * (a * sign(w))` is `w * a` exactly, with `|w|` at most 128.
    #[inline(always)]
    pub(super) unsafe fn dot_signed<const MODE: u8>(
        acc: __m256i,
        weights: __m256i,
        activations: __m256i,
    ) -> __m256i {
        unsafe {
            dot::<MODE>(
                acc,
                _mm256_abs_epi8(weights),
                _mm256_sign_epi8(activations, weights),
            )
        }
    }

    /// The products of one packet's `weights` (unsigned bytes with
    /// `unsigned`, else signed) with packet `local` of the activation block
    /// `x`.
    #[inline(always)]
    pub(super) unsafe fn product<const MODE: u8>(
        weights: __m256i,
        x: &crate::quant::Q8Block,
        local: usize,
        unsigned: bool,
    ) -> __m256i {
        unsafe {
            let activations = load(x.codes.as_ptr().add(super::PACKET * local));
            if unsigned {
                dot::<MODE>(_mm256_setzero_si256(), weights, activations)
            } else {
                dot_signed::<MODE>(_mm256_setzero_si256(), weights, activations)
            }
        }
    }

    /// `acc` plus the pair products of unsigned weights and activations,
    /// each pair multiplied by its 16-bit lane of `scales` (at most 128 in
    /// magnitude, so a lane stays far inside `i32`).
    #[inline(always)]
    pub(super) unsafe fn dot_scaled(
        acc: __m256i,
        weights: __m256i,
        activations: __m256i,
        scales: __m256i,
    ) -> __m256i {
        unsafe {
            let pairs = _mm256_maddubs_epi16(weights, activations);
            _mm256_add_epi32(acc, _mm256_madd_epi16(pairs, scales))
        }
    }

    /// The sum of the eight lanes.
    #[inline(always)]
    pub(super) unsafe fn sum(lanes: __m256i) -> i32 {
        unsafe {
            let half = _mm_add_epi32(
                _mm256_castsi256_si128(lanes),
                _mm256_extracti128_si256::<1>(lanes),
            );
            let pair = _mm_add_epi32(half, _mm_shuffle_epi32::<0b01_00_11_10>(half));
            _mm_cvtsi128_si32(_mm_add_epi32(
                pair,
                _mm_shuffle_epi32::<0b10_11_00_01>(pair),
            ))
        }
    }

    /// Per 128-bit half of four vectors: `[v0 low, v1 low, v2 low, v3 low |
    /// v0 high, v1 high, v2 high, v3 high]`, each the sum of four lanes.
    #[inline(always)]
    pub(super) unsafe fn halves4(v: [__m256i; 4]) -> __m256i {
        unsafe { _mm256_hadd_epi32(_mm256_hadd_epi32(v[0], v[1]), _mm256_hadd_epi32(v[2], v[3])) }
    }

    /// The sums of the low and of the high halves of eight vectors, in
    /// vector order.
    #[inline(always)]
    pub(super) unsafe fn halves8(v: [__m256i; 8]) -> (__m256i, __m256i) {
        unsafe {
            let first = halves4([v[0], v[1], v[2], v[3]]);
            let second = halves4([v[4], v[5], v[6], v[7]]);
            (
                _mm256_permute2x128_si256::<0x20>(first, second),
                _mm256_permute2x128_si256::<0x31>(first, second),
            )
        }
    }

    /// The lane sums of eight vectors, in vector order.
    #[inline(always)]
    pub(super) unsafe fn sums8(v: [__m256i; 8]) -> __m256i {
        unsafe {
            let (low, high) = halves8(v);
            _mm256_add_epi32(low, high)
        }
    }

    /// A packet's `4 LOW` bytes of `LOW`-bit codes as 32 codes in element
    /// order: code `(8 / LOW) i + j` is bits `LOW j ..` of byte `i`.
    #[inline(always)]
    pub(super) unsafe fn codes<const LOW: u32>(bytes: *const u8) -> __m256i {
        const { assert!(LOW == 2 || LOW == 4, "a 2- or 4-bit low plane") };
        unsafe {
            if LOW == 4 {
                let packed = _mm_loadu_si128(bytes.cast());
                let mask = _mm_set1_epi8(15);
                let low = _mm_and_si128(packed, mask);
                let high = _mm_and_si128(_mm_srli_epi16::<4>(packed), mask);
                _mm256_set_m128i(_mm_unpackhi_epi8(low, high), _mm_unpacklo_epi8(low, high))
            } else {
                // Codes 4i..4i + 4 of byte i: its crumbs at bits 0, 2, 4, 6.
                let packed = _mm_loadl_epi64(bytes.cast());
                let three = _mm_set1_epi8(3);
                let crumb0 = _mm_and_si128(packed, three);
                let crumb1 = _mm_and_si128(_mm_srli_epi16::<2>(packed), three);
                let crumb2 = _mm_and_si128(_mm_srli_epi16::<4>(packed), three);
                let crumb3 = _mm_and_si128(_mm_srli_epi16::<6>(packed), three);
                let pairs01 = _mm_unpacklo_epi8(crumb0, crumb1);
                let pairs23 = _mm_unpacklo_epi8(crumb2, crumb3);
                _mm256_set_m128i(
                    _mm_unpackhi_epi16(pairs01, pairs23),
                    _mm_unpacklo_epi16(pairs01, pairs23),
                )
            }
        }
    }

    /// Bit `i` of `mask` as the value `2^LOW` in byte `i`.
    #[inline(always)]
    unsafe fn high_bits<const LOW: u32>(mask: u32) -> __m256i {
        unsafe {
            let spread = _mm256_shuffle_epi8(
                _mm256_set1_epi32(mask as i32),
                _mm256_setr_epi8(
                    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3,
                    3, 3, 3, 3, 3, 3,
                ),
            );
            let bits = _mm256_set1_epi64x(0x8040_2010_0804_0201u64 as i64);
            let set = _mm256_cmpeq_epi8(_mm256_and_si256(spread, bits), bits);
            _mm256_and_si256(set, _mm256_set1_epi8(1 << LOW))
        }
    }

    /// A packet's `LOW`-bit codes (`codes`), each joined with its bit of
    /// `high` (a 1-bit high plane's packet word) at bit `LOW`.
    #[inline(always)]
    pub(super) unsafe fn codes1<const LOW: u32>(bytes: *const u8, high: u32) -> __m256i {
        unsafe { _mm256_or_si256(codes::<LOW>(bytes), high_bits::<LOW>(high)) }
    }

    /// A packet's codes under a 2-bit high plane: codes 2i and 2i + 1 join
    /// the low and high nibble of byte i with bits 0-1 and 2-3 of the high
    /// plane's nibble i.
    #[inline(always)]
    pub(super) unsafe fn nibbles2(codes: *const u8, high: *const u8) -> __m256i {
        unsafe {
            let packed = _mm_loadu_si128(codes.cast());
            let high = _mm_loadl_epi64(high.cast());
            let fifteen = _mm_set1_epi8(15);
            let nibbles = _mm_unpacklo_epi8(
                _mm_and_si128(high, fifteen),
                _mm_and_si128(_mm_srli_epi16::<4>(high), fifteen),
            );
            let three = _mm_set1_epi8(3);
            let even = _mm_or_si128(
                _mm_and_si128(packed, fifteen),
                _mm_slli_epi16::<4>(_mm_and_si128(nibbles, three)),
            );
            let odd = _mm_or_si128(
                _mm_and_si128(_mm_srli_epi16::<4>(packed), fifteen),
                _mm_slli_epi16::<4>(_mm_and_si128(_mm_srli_epi16::<2>(nibbles), three)),
            );
            _mm256_set_m128i(_mm_unpackhi_epi8(even, odd), _mm_unpacklo_epi8(even, odd))
        }
    }

    /// The 32 values of a 16-entry table `table` (broadcast to both halves)
    /// at the nibble codes `codes`.
    #[inline(always)]
    pub(super) unsafe fn lookup(table: __m256i, codes: __m256i) -> __m256i {
        unsafe { _mm256_shuffle_epi8(table, codes) }
    }

    /// A 16-entry byte table in both halves of a vector.
    #[inline(always)]
    pub(super) unsafe fn table(values: &[i8; 16]) -> __m256i {
        unsafe { _mm256_broadcastsi128_si256(_mm_loadu_si128(values.as_ptr().cast())) }
    }
}

/// Integer dot of 32 signed weight values and signed activation codes.
#[inline(always)]
fn dot_i8_i8<const MODE: u8>(weights: &[i8; 32], activations: &[i8; 32]) -> i32 {
    #[cfg(target_arch = "x86_64")]
    if MODE >= 1 {
        return unsafe {
            let (w, a) = (
                packed_x86::load(weights.as_ptr()),
                packed_x86::load(activations.as_ptr()),
            );
            packed_x86::sum(packed_x86::dot_signed::<MODE>(
                std::arch::x86_64::_mm256_setzero_si256(),
                w,
                a,
            ))
        };
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "dotprod"))]
    return unsafe { packed_neon::dot32(weights.as_ptr(), activations.as_ptr()) };
    #[cfg(all(target_arch = "aarch64", not(target_feature = "dotprod")))]
    if std::arch::is_aarch64_feature_detected!("dotprod") {
        return unsafe { dot_i8_i8_dotprod(weights.as_ptr(), activations.as_ptr()) };
    }
    #[allow(unreachable_code)]
    weights
        .iter()
        .zip(activations)
        .map(|(w, a)| i32::from(*w) * i32::from(*a))
        .sum()
}

/// ARM CPUs whose target baseline lacks the dot-product instructions but
/// that report them at run time reduce two 16-byte signed packets without
/// scalar widening or a temporary sum array.
#[cfg(all(target_arch = "aarch64", not(target_feature = "dotprod")))]
#[target_feature(enable = "dotprod")]
unsafe fn dot_i8_i8_dotprod(weights: *const i8, activations: *const i8) -> i32 {
    use std::arch::aarch64::*;
    let mut sums = vdupq_n_s32(0);
    for half in 0..2 {
        let weight = unsafe { vld1q_s8(weights.add(16 * half)) };
        let activation = unsafe { vld1q_s8(activations.add(16 * half)) };
        // Rust's dotprod intrinsic is not stable yet. The target feature
        // restricts this instruction to CPUs that advertise it.
        unsafe {
            std::arch::asm!(
                "sdot {sum:v}.4s, {weight:v}.16b, {activation:v}.16b",
                sum = inout(vreg) sums,
                weight = in(vreg) weight,
                activation = in(vreg) activation,
                options(nostack, preserves_flags),
            );
        }
    }
    vaddvq_s32(sums)
}

#[inline(always)]
unsafe fn read<T: Copy>(pointer: *const u8) -> T {
    unsafe { pointer.cast::<T>().read_unaligned() }
}

/// The body of the row-decode component: the `k` values of one row.
///
/// # Safety
/// `row` addresses a row of `W` with `geometry` and `k` values; `out` holds
/// at least `k` values.
#[inline(always)]
pub unsafe fn decode_row<W: Format, const MODE: u8>(
    row: *const u8,
    geometry: &RowGeometry,
    k: usize,
    out: &mut [f32],
) {
    let out = &mut out[..k];
    let row = geometry.row(row);
    let mut packet = [0.0f32; PACKET];
    for (p, chunk) in out.chunks_mut(PACKET).enumerate() {
        unsafe { W::decode::<MODE>(&row, p, k, &mut packet) };
        chunk.copy_from_slice(&packet[..chunk.len()]);
    }
}

/// `lanes` plus the products of a packet's values and activations, value
/// `i` fused into lane `i % LANES` in index order: the tier's vector
/// multiply-adds where it has them, so the accumulation is the same
/// instructions in every component.
#[inline(always)]
unsafe fn accumulate<const MODE: u8>(
    lanes: &mut [f32; LANES],
    values: &[f32; PACKET],
    x: &[f32; PACKET],
) {
    const { assert!(LANES == 8 && PACKET % LANES == 0, "a lane vector of eight") };
    #[cfg(target_arch = "x86_64")]
    if MODE >= 1 {
        use std::arch::x86_64::*;
        unsafe {
            let mut sum = _mm256_loadu_ps(lanes.as_ptr());
            for chunk in 0..PACKET / LANES {
                let at = LANES * chunk;
                sum = _mm256_fmadd_ps(
                    _mm256_loadu_ps(values.as_ptr().add(at)),
                    _mm256_loadu_ps(x.as_ptr().add(at)),
                    sum,
                );
            }
            _mm256_storeu_ps(lanes.as_mut_ptr(), sum);
        }
        return;
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::*;
        let (mut low, mut high) = (vld1q_f32(lanes.as_ptr()), vld1q_f32(lanes.as_ptr().add(4)));
        for chunk in 0..PACKET / LANES {
            let at = LANES * chunk;
            low = vfmaq_f32(
                low,
                vld1q_f32(values.as_ptr().add(at)),
                vld1q_f32(x.as_ptr().add(at)),
            );
            high = vfmaq_f32(
                high,
                vld1q_f32(values.as_ptr().add(at + 4)),
                vld1q_f32(x.as_ptr().add(at + 4)),
            );
        }
        vst1q_f32(lanes.as_mut_ptr(), low);
        vst1q_f32(lanes.as_mut_ptr().add(4), high);
    }
    #[cfg(not(target_arch = "aarch64"))]
    for chunk in 0..PACKET / LANES {
        for lane in 0..LANES {
            let i = chunk * LANES + lane;
            lanes[lane] = values[i].mul_add(x[i], lanes[lane]);
        }
    }
}

/// The body of the dot component: rows `0..R` (from `rows`, `geometry.stride`
/// apart) against the `k` values of `x`.
///
/// # Safety
/// `rows` addresses `R` rows of `W` with `geometry` and `k` values.
#[inline(always)]
pub unsafe fn dot_rows<W: Format, const R: usize, const MODE: u8>(
    rows: *const u8,
    geometry: &RowGeometry,
    x: &[f32],
    k: usize,
) -> [f32; R] {
    let x = &x[..k];
    let mut lanes = [[0.0f32; LANES]; R];
    // One packet buffer, consumed by its row before the next row decodes:
    // with every row's packet live at once, x86 vectorizers transpose the
    // accumulation across rows into a gather per multiply-add.
    let mut packet = [0.0f32; PACKET];
    let rows =
        std::array::from_fn::<_, R, _>(|r| geometry.row(rows.wrapping_add(r * geometry.stride)));
    let full = k / PACKET;
    for p in 0..full {
        let xs: &[f32; PACKET] = x[p * PACKET..(p + 1) * PACKET]
            .try_into()
            .expect("a whole packet");
        for (r, lanes) in lanes.iter_mut().enumerate() {
            unsafe { W::decode::<MODE>(&rows[r], p, k, &mut packet) };
            unsafe { accumulate::<MODE>(lanes, &packet, xs) };
        }
    }
    let tail = k - full * PACKET;
    if tail > 0 {
        let xs = &x[full * PACKET..];
        for (r, lanes) in lanes.iter_mut().enumerate() {
            unsafe { W::decode::<MODE>(&rows[r], full, k, &mut packet) };
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
    let rows =
        std::array::from_fn::<_, R, _>(|r| geometry.row(rows.wrapping_add(r * geometry.stride)));
    let mut sums = [0.0f32; R];
    for (b, block) in x.iter().enumerate() {
        let values = unsafe { W::dot_q8::<R, 1, MODE>(&rows, b, k, [block]) };
        for (sum, value) in sums.iter_mut().zip(values) {
            *sum += value[0];
        }
    }
    sums
}

/// The body of the quantized GEMM component: eight rows (from `rows`,
/// `geometry.stride` apart) of `k` values against four quantized activation
/// rows (`x`, `quant::blocks(k)` blocks each), activation-major: `out[m][r]`
/// accumulates row `r`'s block dots with activation row `m` in block order.
///
/// # Safety
/// `rows` addresses eight rows of `W` with `geometry` and `k` values, and
/// `x` holds `4 quant::blocks(k)` blocks.
#[inline(always)]
pub unsafe fn gemm_q8_rows<W: Format, const MODE: u8>(
    rows: *const u8,
    geometry: &RowGeometry,
    x: &[Q8Block],
    k: usize,
) -> [[f32; 8]; 4] {
    let blocks = crate::quant::blocks(k);
    let mut accumulators = [[0.0f32; 8]; 4];
    let rows =
        std::array::from_fn::<_, 8, _>(|r| geometry.row(rows.wrapping_add(r * geometry.stride)));
    for b in 0..blocks {
        let activations = std::array::from_fn(|m| &x[m * blocks + b]);
        for r in 0..8 {
            let [products] = unsafe { W::dot_q8::<1, 4, MODE>(&[rows[r]], b, k, activations) };
            for m in 0..4 {
                accumulators[m][r] += products[m];
            }
        }
    }
    accumulators
}

#[cfg(test)]
mod tests {
    use crate::components::{elements, resolve};
    use crate::isa::Tier;
    use seismic_lang::registry::{
        self, CoefficientAxes, Layout, RepresentationKind, RowPlaneRole,
    };

    /// Every packed weight element decodes rows of random codes to exactly
    /// the registry's decode of the same bytes.
    #[test]
    fn packed_decode_matches_the_registry() {
        use seismic_lang::interp::TensorData;
        let tier = Tier::detected().unwrap();
        for name in elements() {
            let id = registry::representation(name).unwrap();
            let info = registry::representation_info(id);
            let RepresentationKind::PackedRows(layout) = &info.kind else {
                continue;
            };
            if info.layout != Layout::Rows16 {
                continue;
            }
            let kernels = resolve(tier, id).unwrap();
            let axes = registry::decode_axes(id).unwrap();
            let encoding = match axes.coefficients {
                CoefficientAxes::Direct { scale, .. } => scale,
                CoefficientAxes::Hierarchical { factor, .. } => factor,
            };
            let group = layout.group() as usize;
            // Also the Gemma K = 704 and the Nemotron K = 2688 where whole
            // storage groups make them.
            let lengths = [1, 3, 10].map(|groups| groups * group);
            for k in lengths
                .into_iter()
                .chain([704, 2688].into_iter().filter(|k| k % group == 0))
            {
                let geometry = super::RowGeometry::of(info, k, 0);
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
                // Finite coefficients: moderate floats (random f16 and f32
                // bytes would be NaN or overflow), and every finite code of
                // the scale codes.
                let supers = RowPlaneRole::Supers as usize;
                let bytes = layout.row_groups(k as u64).unwrap() as usize * geometry.groups[supers];
                for row in 0..rows {
                    let start = row * geometry.stride + geometry.offsets[supers];
                    let plane = &mut data[start..start + bytes];
                    crate::element::finite_scales(plane, encoding, true);
                }
                let reference = TensorData::encoded(id, vec![rows, k], data.clone())
                    .unwrap()
                    .values()
                    .unwrap();
                let mut out = vec![0.0f32; k];
                for row in 0..rows {
                    unsafe {
                        kernels.decode(
                            data.as_ptr().add(row * geometry.stride),
                            &geometry,
                            k,
                            &mut out,
                        )
                    };
                    for (i, value) in out.iter().enumerate() {
                        assert_eq!(
                            f64::from(*value),
                            reference[row * k + i],
                            "{name} k {k} row {row} value {i}"
                        );
                    }
                }
            }
        }
    }
}
