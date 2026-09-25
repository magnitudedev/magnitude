//! Exact conversions of external GGUF weights into resident `rows16` storage:
//! the registry's registered `repack`. Codes and coefficients move bit for
//! bit (only `iq4g32` scales are formed, each an exact product); nothing is
//! decoded. One source packet becomes one storage group of its row.
//!
//! A kernel converts through [`External`] and [`Packed`] views of its bound
//! tensors; [`rows`] resolves the conversion from their representations.

use crate::element::f16_to_f32;
use crate::weights::{Format, Iq4, RowGeometry, Q4K, Q5K, Q6K, Q8};
use std::marker::PhantomData;

/// One registered conversion of an external packet format into a `rows16`
/// format whose storage group is one source packet.
trait Conversion {
    /// The registry name of the external source.
    const SOURCE: &'static str;
    type Target: Format;
    const PACKET_BYTES: usize;
    /// Values of one packet.
    const GROUP: usize;
    /// Bits of one code.
    const BITS: u32;
    /// Bytes per group of the target's local-coefficient and super planes.
    const SCALE_BYTES: usize;
    const SUPER_BYTES: usize;

    /// The raw code of value `p` of `packet`.
    fn code(packet: &[u8], p: usize) -> u8;
    /// The group's local-coefficient plane bytes.
    fn scales(_packet: &[u8], _out: &mut [u8]) {}
    /// The group's super-plane bytes.
    fn supers(packet: &[u8], out: &mut [u8]);
}

/// The six-bit local coefficient `field` (0 scale, 1 minimum) of sub-block
/// `j` of a GGUF q4_K or q5_K packet.
fn k_local(packet: &[u8], j: usize, field: usize) -> u128 {
    let low = packet[4 + field * 4 + j % 4];
    let high = packet[12 + j % 4];
    u128::from(if j < 4 { low & 63 } else { ((high >> (4 * field)) & 15) | ((low >> 6) << 4) })
}

/// The registry's packed local coefficients of a k-quant group: sixteen
/// six-bit fields, field `2j + f` of sub-block `j` at bit `12j + 6f`.
fn k_scales(packet: &[u8], out: &mut [u8]) {
    let mut bits = 0u128;
    for field in 0..16 {
        bits |= k_local(packet, field / 2, field % 2) << (6 * field);
    }
    out.copy_from_slice(&bits.to_le_bytes()[..12]);
}

struct GgufQ4K;

impl Conversion for GgufQ4K {
    const SOURCE: &'static str = "gguf_q4_k";
    type Target = Q4K;
    const PACKET_BYTES: usize = 144;
    const GROUP: usize = 256;
    const BITS: u32 = 4;
    const SCALE_BYTES: usize = 12;
    const SUPER_BYTES: usize = 4;

    fn code(packet: &[u8], p: usize) -> u8 {
        (packet[16 + (p / 64) * 32 + p % 32] >> ((p % 64 / 32) * 4)) & 15
    }
    fn scales(packet: &[u8], out: &mut [u8]) {
        k_scales(packet, out);
    }
    fn supers(packet: &[u8], out: &mut [u8]) {
        out.copy_from_slice(&packet[..4]);
    }
}

struct GgufQ5K;

impl Conversion for GgufQ5K {
    const SOURCE: &'static str = "gguf_q5_k";
    type Target = Q5K;
    const PACKET_BYTES: usize = 176;
    const GROUP: usize = 256;
    const BITS: u32 = 5;
    const SCALE_BYTES: usize = 12;
    const SUPER_BYTES: usize = 4;

    fn code(packet: &[u8], p: usize) -> u8 {
        let low = (packet[48 + (p / 64) * 32 + p % 32] >> ((p % 64 / 32) * 4)) & 15;
        let high = (packet[16 + p % 32] >> (p / 32)) & 1;
        low | (high << 4)
    }
    fn scales(packet: &[u8], out: &mut [u8]) {
        k_scales(packet, out);
    }
    fn supers(packet: &[u8], out: &mut [u8]) {
        out.copy_from_slice(&packet[..4]);
    }
}

struct GgufQ6K;

impl Conversion for GgufQ6K {
    const SOURCE: &'static str = "gguf_q6_k";
    type Target = Q6K;
    const PACKET_BYTES: usize = 210;
    const GROUP: usize = 256;
    const BITS: u32 = 6;
    const SCALE_BYTES: usize = 16;
    const SUPER_BYTES: usize = 2;

    fn code(packet: &[u8], p: usize) -> u8 {
        let low = (packet[(p / 128) * 64 + p % 64] >> ((p % 128 / 64) * 4)) & 15;
        let high = (packet[128 + (p / 128) * 32 + p % 32] >> ((p % 128 / 32) * 2)) & 3;
        low | (high << 4)
    }
    fn scales(packet: &[u8], out: &mut [u8]) {
        out.copy_from_slice(&packet[192..208]);
    }
    fn supers(packet: &[u8], out: &mut [u8]) {
        out.copy_from_slice(&packet[208..210]);
    }
}

struct GgufQ8;

impl Conversion for GgufQ8 {
    const SOURCE: &'static str = "gguf_q8_0";
    type Target = Q8;
    const PACKET_BYTES: usize = 34;
    const GROUP: usize = 32;
    const BITS: u32 = 8;
    const SCALE_BYTES: usize = 0;
    const SUPER_BYTES: usize = 2;

    fn code(packet: &[u8], p: usize) -> u8 {
        packet[2 + p]
    }
    fn supers(packet: &[u8], out: &mut [u8]) {
        out.copy_from_slice(&packet[..2]);
    }
}

struct GgufIq4Xs;

impl Conversion for GgufIq4Xs {
    const SOURCE: &'static str = "gguf_iq4_xs";
    type Target = Iq4;
    const PACKET_BYTES: usize = 136;
    const GROUP: usize = 256;
    const BITS: u32 = 4;
    const SCALE_BYTES: usize = 0;
    const SUPER_BYTES: usize = 32;

    fn code(packet: &[u8], p: usize) -> u8 {
        (packet[8 + (p / 32) * 16 + p % 16] >> ((p % 32 / 16) * 4)) & 15
    }
    /// The `f32` scale of each 32-value sub-block: the f16 base times the
    /// sub-block's six-bit scale less 32, exact in `f32`.
    fn supers(packet: &[u8], out: &mut [u8]) {
        let base = f16_to_f32(u16::from_le_bytes([packet[0], packet[1]]));
        let high = u16::from_le_bytes([packet[2], packet[3]]);
        for (j, scale) in out.chunks_exact_mut(4).enumerate() {
            let low = (packet[4 + j / 2] >> ((j % 2) * 4)) & 15;
            let local = i32::from(low | ((((high >> (2 * j)) & 3) as u8) << 4)) - 32;
            scale.copy_from_slice(&(base * local as f32).to_le_bytes());
        }
    }
}

/// Converts the source row at `source` (whole packets of `k` values) into the
/// target row at `row`, zeroing every stored byte no plane occupies.
///
/// # Safety
/// `source` addresses `ceil(k / GROUP)` packets and `row` one target row of
/// `k` values.
unsafe fn convert_row<C: Conversion>(source: *const u8, row: *mut u8, k: usize) {
    let geometry: RowGeometry = C::Target::geometry(k, 0);
    let groups = k.div_ceil(C::GROUP);
    let source = unsafe { std::slice::from_raw_parts(source, groups * C::PACKET_BYTES) };
    let out = unsafe { std::slice::from_raw_parts_mut(row, geometry.stride) };
    out.fill(0);
    let mut codes = [0u8; 256];
    for (g, packet) in source.chunks_exact(C::PACKET_BYTES).enumerate() {
        let codes = &mut codes[..C::GROUP];
        for (p, code) in codes.iter_mut().enumerate() {
            *code = C::code(packet, p);
        }
        if C::BITS == 8 {
            out[geometry.codes + g * C::GROUP..][..C::GROUP].copy_from_slice(codes);
        } else {
            let low = &mut out[geometry.codes + g * C::GROUP / 2..][..C::GROUP / 2];
            for (i, byte) in low.iter_mut().enumerate() {
                *byte = (codes[2 * i] & 15) | ((codes[2 * i + 1] & 15) << 4);
            }
        }
        match C::BITS {
            5 => {
                let high = &mut out[geometry.high + g * C::GROUP / 8..][..C::GROUP / 8];
                for (i, code) in codes.iter().enumerate() {
                    high[i / 8] |= ((code >> 4) & 1) << (i % 8);
                }
            }
            6 => {
                let high = &mut out[geometry.high + g * C::GROUP / 4..][..C::GROUP / 4];
                for (i, code) in codes.iter().enumerate() {
                    high[i / 4] |= ((code >> 4) & 3) << (2 * (i % 4));
                }
            }
            _ => {}
        }
        if C::SCALE_BYTES > 0 {
            C::scales(packet, &mut out[geometry.scales + g * C::SCALE_BYTES..][..C::SCALE_BYTES]);
        }
        C::supers(packet, &mut out[geometry.supers + g * C::SUPER_BYTES..][..C::SUPER_BYTES]);
    }
}

/// One registered conversion as a kernel resolves it.
#[derive(Clone, Copy, Debug)]
pub struct RowConversion {
    pub source: &'static str,
    pub target: &'static str,
    /// Bytes and values of one source packet.
    pub packet_bytes: usize,
    pub group: usize,
    geometry: fn(usize, usize) -> RowGeometry,
    /// Eight-row block interleaving rather than contiguous rows.
    rows8: bool,
    /// Bytes per storage group of each plane, in storage order.
    planes: &'static [usize],
    convert: unsafe fn(*const u8, *mut u8, usize),
}

const fn conversion<C: Conversion>(target: &'static str, rows8: bool, planes: &'static [usize]) -> RowConversion {
    RowConversion {
        source: C::SOURCE,
        target,
        packet_bytes: C::PACKET_BYTES,
        group: C::GROUP,
        geometry: C::Target::geometry,
        rows8,
        planes,
        convert: convert_row::<C>,
    }
}

static CONVERSIONS: [RowConversion; 10] = [
    conversion::<GgufQ4K>("q4k@rows16", false, &[128, 12, 4]),
    conversion::<GgufQ5K>("q5k@rows16", false, &[128, 32, 12, 4]),
    conversion::<GgufQ6K>("q6k@rows16", false, &[128, 64, 16, 2]),
    conversion::<GgufQ8>("q8g32s@rows16", false, &[32, 2]),
    conversion::<GgufIq4Xs>("iq4g32@rows16", false, &[128, 32]),
    conversion::<GgufQ4K>("q4k@rows8", true, &[128, 12, 4]),
    conversion::<GgufQ5K>("q5k@rows8", true, &[128, 32, 12, 4]),
    conversion::<GgufQ6K>("q6k@rows8", true, &[128, 64, 16, 2]),
    conversion::<GgufQ8>("q8g32s@rows8", true, &[32, 2]),
    conversion::<GgufIq4Xs>("iq4g32@rows8", true, &[128, 32]),
];

/// Every registered conversion into a `rows16` representation.
pub fn conversions() -> &'static [RowConversion] {
    &CONVERSIONS
}

/// The conversion of `source` into `target`, when one is registered.
pub fn resolve(source: &str, target: &str) -> Option<&'static RowConversion> {
    CONVERSIONS.iter().find(|conversion| conversion.source == source && conversion.target == target)
}

/// Leading-axis row addressing shared by [`External`] and [`Packed`]: a
/// tensor's rows are the positions of every axis before the last, which a
/// canonical view stores `row_bytes` apart.
fn canonical_rows(extents: &[u64], strides: &[u64], row_stride: u64) -> usize {
    let rank = extents.len();
    assert!(rank >= 1, "a converted operand has a value axis");
    let mut expected = row_stride;
    for axis in (0..rank - 1).rev() {
        assert_eq!(strides[axis], expected, "a converted operand's rows are canonical");
        expected *= extents[axis];
    }
    extents[..rank - 1].iter().product::<u64>() as usize
}

/// A bound external source tensor: whole packets per row, read only by a
/// conversion.
#[derive(Clone, Copy, Debug)]
pub struct External<'a> {
    base: *const u8,
    rows: usize,
    k: usize,
    representation: &'static str,
    matrix_rows: usize,
    life: PhantomData<&'a u8>,
}

unsafe impl Send for External<'_> {}
unsafe impl Sync for External<'_> {}

impl<'a> External<'a> {
    /// The external tensor at `base` with `extents` in values and `strides`
    /// in packets.
    ///
    /// # Safety
    /// `base`, `extents` and `strides` describe a tensor of `representation`
    /// valid for `'a`.
    pub unsafe fn from_tensor(base: *const u8, extents: &[u64], strides: &[u64], representation: &'static str) -> Self {
        let k = extents[extents.len() - 1];
        let group = CONVERSIONS
            .iter()
            .find(|conversion| conversion.source == representation)
            .unwrap_or_else(|| panic!("`{representation}` is not a registered external source"))
            .group as u64;
        let rows = canonical_rows(extents, strides, k.div_ceil(group));
        Self { base, rows, k: k as usize, representation, matrix_rows: extents[extents.len() - 2] as usize, life: PhantomData }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn representation(&self) -> &'static str {
        self.representation
    }
}

/// A bound result tensor of a packed `rows16` representation, written only
/// by a conversion.
#[derive(Clone, Copy, Debug)]
pub struct Packed<'a> {
    base: *mut u8,
    rows: usize,
    k: usize,
    representation: &'static str,
    matrix_rows: usize,
    life: PhantomData<&'a u8>,
}

unsafe impl Send for Packed<'_> {}
unsafe impl Sync for Packed<'_> {}

impl<'a> Packed<'a> {
    /// The packed tensor at `base` with `extents` in values and `strides` in
    /// rows.
    ///
    /// # Safety
    /// `base`, `extents` and `strides` describe a tensor of `representation`
    /// valid for `'a`.
    pub unsafe fn from_tensor(base: *mut u8, extents: &[u64], strides: &[u64], representation: &'static str) -> Self {
        let rows = if representation.ends_with("@rows8") {
            let rank = extents.len();
            assert!(rank >= 2, "Rows8 requires a row axis");
            assert_eq!(strides[rank - 1], 1);
            let mut expected = 1u64;
            for axis in (0..rank - 1).rev() {
                assert_eq!(strides[axis], expected, "Rows8 tile strides");
                expected *= if axis == rank - 2 { extents[axis].div_ceil(8) * 8 } else { extents[axis] };
            }
            extents[..rank - 1].iter().product::<u64>() as usize
        } else {
            canonical_rows(extents, strides, 1)
        };
        Self { base, rows, k: extents[extents.len() - 1] as usize, representation, matrix_rows: extents[extents.len() - 2] as usize, life: PhantomData }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn representation(&self) -> &'static str {
        self.representation
    }
}

/// Converts rows `rows` of `source` into the same rows of `destination`.
///
/// # Safety
/// No other work item writes these rows of `destination` in the launch.
pub unsafe fn rows(source: &External<'_>, destination: &Packed<'_>, rows: std::ops::Range<usize>) {
    let conversion = resolve(source.representation, destination.representation).unwrap_or_else(|| {
        panic!("no CPU conversion of `{}` into `{}`", source.representation, destination.representation)
    });
    assert_eq!(source.k, destination.k, "a conversion keeps the row length");
    assert!(rows.end <= source.rows && rows.end <= destination.rows, "rows {rows:?} outside the operands");
    let source_row = source.k.div_ceil(conversion.group) * conversion.packet_bytes;
    let target_row = (conversion.geometry)(destination.k, 0).stride;
    assert_eq!(source.matrix_rows, destination.matrix_rows);
    if conversion.rows8 {
        let n = source.matrix_rows;
        assert!(rows.start / n == (rows.end - 1) / n, "a Rows8 work item stays within one matrix");
        assert_eq!(rows.start % n % 8, 0, "a Rows8 work item starts on a tile");
        assert!(rows.end % n == 0 || rows.end % n % 8 == 0, "a Rows8 work item ends on a tile");
        let groups = source.k.div_ceil(conversion.group);
        let geometry = (conversion.geometry)(destination.k, 0);
        let mut offsets = [0usize; 4];
        let mut end = 0usize;
        for (plane, bytes) in conversion.planes.iter().enumerate() {
            offsets[plane] = end.div_ceil(16) * 16;
            end = offsets[plane] + groups * bytes;
        }
        for first in (rows.start..rows.end).step_by(8) {
            let matrix = first / n;
            let in_matrix = first % n;
            let tile = matrix * n.div_ceil(8) + in_matrix / 8;
            // SAFETY: this work item owns the entire tile, including its padded rows.
            let output = unsafe { std::slice::from_raw_parts_mut(destination.base.add(tile * 8 * target_row), 8 * target_row) };
            output.fill(0);
            for r in 0..(rows.end - first).min(8) {
                let mut packed = vec![0u8; geometry.stride];
                unsafe { (conversion.convert)(source.base.add((first + r) * source_row), packed.as_mut_ptr(), source.k) };
                for (plane, bytes) in conversion.planes.iter().enumerate() {
                    for group in 0..groups {
                        let from = offsets[plane] + group * bytes;
                        let to = offsets[plane] * 8 + group * bytes * 8 + r * bytes;
                        output[to..to + bytes].copy_from_slice(&packed[from..from + bytes]);
                    }
                }
            }
        }
    } else {
        for row in rows {
            // SAFETY: the row lies inside both operands (`from_tensor`); the
            // caller owns its destination row.
            unsafe { (conversion.convert)(source.base.add(row * source_row), destination.base.add(row * target_row), source.k) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use seismic_lang::registry::representation;

    /// Every conversion reproduces the registry's host repack bit for bit,
    /// on rows whose last packet is partial.
    #[test]
    fn conversions_match_the_registry_repack() {
        for conversion in conversions() {
            for k in [conversion.group, 3 * conversion.group, 2560, 2560 + 40] {
                let rows = 3;
                let packets = k.div_ceil(conversion.group);
                let mut state = 0x9e37_79b9_7f4a_7c15u64 ^ k as u64;
                let mut source = (0..rows * packets * conversion.packet_bytes)
                    .map(|_| {
                        state ^= state << 13;
                        state ^= state >> 7;
                        state ^= state << 17;
                        (state >> 24) as u8
                    })
                    .collect::<Vec<_>>();
                if conversion.source == "gguf_iq4_xs" {
                    // Finite f16 bases (exponent below 16): the scales are
                    // formed products, and NaN payloads are not part of the
                    // conversion contract.
                    for packet in source.chunks_exact_mut(conversion.packet_bytes) {
                        packet[1] &= 0xbf;
                    }
                }
                let id = seismic_lang::registry::representation_conversion(
                    representation(conversion.source).unwrap(),
                    representation(conversion.target).unwrap(),
                )
                .expect("registered conversion")
                .id;
                let expected = seismic_lang::interp::repack(id, &[rows, k], &source).expect("registry repack");
                let stride = (conversion.geometry)(k, 0).stride;
                let stored_rows = if conversion.rows8 { rows.div_ceil(8) * 8 } else { rows };
                let mut out = vec![0xa5u8; stored_rows * stride];
                if conversion.rows8 {
                    let from = unsafe { External::from_tensor(source.as_ptr(), &[rows as u64, k as u64], &[packets as u64, 1], conversion.source) };
                    let to = unsafe { Packed::from_tensor(out.as_mut_ptr(), &[rows as u64, k as u64], &[1, 1], conversion.target) };
                    unsafe { super::rows(&from, &to, 0..rows) };
                } else {
                    for row in 0..rows {
                        unsafe { (conversion.convert)(source.as_ptr().add(row * packets * conversion.packet_bytes), out.as_mut_ptr().add(row * stride), k) };
                    }
                }
                assert_eq!(out.len(), expected.len(), "{} -> {} k {k}", conversion.source, conversion.target);
                if let Some(at) = out.iter().zip(&expected).position(|(a, b)| a != b) {
                    let word = at / 4 * 4;
                    panic!(
                        "{} -> {} k {k}: byte {at} (row {}, offset {}) is {:#04x}, the registry's {:#04x}; words {:?} {:?}",
                        conversion.source,
                        conversion.target,
                        at / stride,
                        at % stride,
                        out[at],
                        expected[at],
                        f32::from_le_bytes(out[word..word + 4].try_into().unwrap()),
                        f32::from_le_bytes(expected[word..word + 4].try_into().unwrap()),
                    );
                }
            }
        }
    }
}
