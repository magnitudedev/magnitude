//! Exact conversions of external GGUF weights into resident `rows16` and
//! `rows8` storage:
//! the registry's registered `repack`. Codes and coefficients move bit for
//! bit, except that `iq4g32` scales are formed (each an exact product or
//! widening) and q3_K and iq3_s codes and scales are re-expressed as q6k's
//! (every value is unchanged); nothing is decoded. One source packet becomes
//! one storage group of its row.
//!
//! A kernel converts through [`External`] and [`Packed`] views of its bound
//! tensors; [`rows`] resolves the conversion from their representations.

use crate::element::f16_to_f32;
use crate::weights::RowGeometry;
use seismic_lang::registry::{self, Layout, RepresentationInfo, RowPlaneRole};
use std::marker::PhantomData;

/// One registered conversion of an external packet format into a resident
/// format whose storage group is one source packet.
trait Conversion {
    /// The registry name of the external source.
    const SOURCE: &'static str;
    const PACKET_BYTES: usize;
    /// Values of one packet.
    const GROUP: usize;
    /// Bits of one code.
    const BITS: u32;
    /// Bytes per group of the target's local-coefficient and super planes.
    const SCALE_BYTES: usize;
    const SUPER_BYTES: usize;

    /// Bytes of the target's vector-code entries per group, for a grid
    /// target, whose entries `grids` writes whole; 0 for per-value codes.
    const GRID_BYTES: usize = 0;

    /// The raw code of value `p` of `packet`.
    fn code(_packet: &[u8], _p: usize) -> u8 {
        unreachable!("a grid conversion moves whole entries")
    }
    /// The group's vector-code entries.
    fn grids(_packet: &[u8], _out: &mut [u8]) {
        unreachable!("a per-value conversion moves codes")
    }
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
    u128::from(if j < 4 {
        low & 63
    } else {
        ((high >> (4 * field)) & 15) | ((low >> 6) << 4)
    })
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

/// GGUF q3_K into q6k: codes and coefficients are re-based exactly (see the
/// registry conversion); the f16 super-scale moves bit for bit.
struct GgufQ3K;

impl Conversion for GgufQ3K {
    const SOURCE: &'static str = "gguf_q3_k";
    const PACKET_BYTES: usize = 110;
    const GROUP: usize = 256;
    const BITS: u32 = 6;
    const SCALE_BYTES: usize = 16;
    const SUPER_BYTES: usize = 2;

    /// `q + 32` for q = low two bits + 4 * high bit - 4.
    fn code(packet: &[u8], p: usize) -> u8 {
        let low = (packet[32 + (p / 128) * 32 + p % 32] >> ((p % 128 / 32) * 2)) & 3;
        let high = (packet[p % 32] >> (p / 32)) & 1;
        (low | (high << 2)) + 28
    }
    /// Each sub-block's six-bit scale less 32, as a two's complement byte.
    fn scales(packet: &[u8], out: &mut [u8]) {
        for (j, scale) in out.iter_mut().enumerate() {
            let low = (packet[96 + j % 8] >> (4 * (j / 8))) & 15;
            let high = (packet[104 + j % 4] >> (2 * (j / 4))) & 3;
            *scale = ((low | (high << 4)) as i8 - 32) as u8;
        }
    }
    fn supers(packet: &[u8], out: &mut [u8]) {
        out.copy_from_slice(&packet[108..110]);
    }
}

struct GgufQ4K;

impl Conversion for GgufQ4K {
    const SOURCE: &'static str = "gguf_q4_k";
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

/// GGUF iq3_s into q6k: each value's signed grid magnitude re-based by 32,
/// each 32-value sub-block's scale `1 + 2 s` on both its sixteen-value
/// halves; the f16 super-scale moves bit for bit.
struct GgufIq3S;

impl Conversion for GgufIq3S {
    const SOURCE: &'static str = "gguf_iq3_s";
    const PACKET_BYTES: usize = 110;
    const GROUP: usize = 256;
    const BITS: u32 = 6;
    const SCALE_BYTES: usize = 16;
    const SUPER_BYTES: usize = 2;

    fn code(packet: &[u8], p: usize) -> u8 {
        let (block, group, slot) = (p / 32, p % 32 / 8, p % 8);
        let point = usize::from(packet[2 + 8 * block + 2 * group + slot / 4])
            | usize::from((packet[66 + block] >> (2 * group + slot / 4)) & 1) << 8;
        let magnitude = magnitude_ggml::tables::IQ3S
            .points
            .byte(point, (slot % 4) as u32);
        if (packet[74 + 4 * block + group] >> slot) & 1 != 0 {
            32 - magnitude
        } else {
            32 + magnitude
        }
    }
    fn scales(packet: &[u8], out: &mut [u8]) {
        for (j, scale) in out.iter_mut().enumerate() {
            *scale = 1 + 2 * ((packet[106 + j / 4] >> (4 * (j / 2 % 2))) & 15);
        }
    }
    fn supers(packet: &[u8], out: &mut [u8]) {
        out.copy_from_slice(&packet[..2]);
    }
}

/// GGUF iq2_xs into iq2xs: its sixteen-bit entries and 4-bit scale codes
/// move bit for bit; the f16 `d` becomes the f32 factor `d / 8`, exactly.
struct GgufIq2Xs;

impl Conversion for GgufIq2Xs {
    const SOURCE: &'static str = "gguf_iq2_xs";
    const PACKET_BYTES: usize = 74;
    const GROUP: usize = 256;
    const BITS: u32 = 0;
    const GRID_BYTES: usize = 64;
    const SCALE_BYTES: usize = 8;
    const SUPER_BYTES: usize = 4;

    fn grids(packet: &[u8], out: &mut [u8]) {
        out.copy_from_slice(&packet[2..66]);
    }
    fn scales(packet: &[u8], out: &mut [u8]) {
        out.copy_from_slice(&packet[66..74]);
    }
    fn supers(packet: &[u8], out: &mut [u8]) {
        let d = f16_to_f32(u16::from_le_bytes([packet[0], packet[1]]));
        out.copy_from_slice(&(d * 0.125).to_le_bytes());
    }
}

/// GGUF iq3_xxs into iq3xxs: entry `e` (values `8 e ..`) is its sub-block's
/// grid indices `2 (e % 4)` and `2 (e % 4) + 1` and sign group `e % 4`, a
/// three-byte entry; each sub-block's 4-bit scale code moves; the f16 `d`
/// becomes the f32 factor `d / 4`, exactly.
struct GgufIq3Xxs;

impl Conversion for GgufIq3Xxs {
    const SOURCE: &'static str = "gguf_iq3_xxs";
    const PACKET_BYTES: usize = 98;
    const GROUP: usize = 256;
    const BITS: u32 = 0;
    const GRID_BYTES: usize = 96;
    const SCALE_BYTES: usize = 4;
    const SUPER_BYTES: usize = 4;

    fn grids(packet: &[u8], out: &mut [u8]) {
        for (entry, out) in out.chunks_exact_mut(3).enumerate() {
            let (block, group) = (entry / 4, entry % 4);
            let word =
                u32::from_le_bytes(packet[66 + 4 * block..70 + 4 * block].try_into().unwrap());
            out[0] = packet[2 + 8 * block + 2 * group];
            out[1] = packet[3 + 8 * block + 2 * group];
            out[2] = ((word >> (7 * group)) & 127) as u8;
        }
    }
    fn scales(packet: &[u8], out: &mut [u8]) {
        out.fill(0);
        for block in 0..8 {
            let scale = packet[69 + 4 * block] >> 4;
            out[block / 2] |= scale << (4 * (block % 2));
        }
    }
    fn supers(packet: &[u8], out: &mut [u8]) {
        let d = f16_to_f32(u16::from_le_bytes([packet[0], packet[1]]));
        out.copy_from_slice(&(d * 0.25).to_le_bytes());
    }
}

/// GGUF iq2_xxs into iq2xxs: entry `e` (values `8 e ..`) is its sub-block's
/// grid index `e % 4` and sign group `e % 4`, a sixteen-bit entry carrying
/// at bit 15 bit `e % 4` of its sub-block's 4-bit scale code; the f16 `d`
/// moves bit for bit.
struct GgufIq2Xxs;

impl Conversion for GgufIq2Xxs {
    const SOURCE: &'static str = "gguf_iq2_xxs";
    const PACKET_BYTES: usize = 66;
    const GROUP: usize = 256;
    const BITS: u32 = 0;
    const GRID_BYTES: usize = 64;
    const SCALE_BYTES: usize = 0;
    const SUPER_BYTES: usize = 2;

    fn grids(packet: &[u8], out: &mut [u8]) {
        for (entry, out) in out.chunks_exact_mut(2).enumerate() {
            let (block, group) = (entry / 4, entry % 4);
            let word =
                u32::from_le_bytes(packet[6 + 8 * block..10 + 8 * block].try_into().unwrap());
            out[0] = packet[2 + 8 * block + group];
            out[1] =
                ((word >> (7 * group)) & 127) as u8 | (((word >> (28 + group)) & 1) << 7) as u8;
        }
    }
    fn supers(packet: &[u8], out: &mut [u8]) {
        out.copy_from_slice(&packet[..2]);
    }
}

/// Eight consecutive GGUF iq4_nl blocks (one iq4g32 storage group).
struct GgufIq4Nl;

impl Conversion for GgufIq4Nl {
    const SOURCE: &'static str = "gguf_iq4_nl";
    const PACKET_BYTES: usize = 144;
    const GROUP: usize = 256;
    const BITS: u32 = 4;
    const SCALE_BYTES: usize = 0;
    const SUPER_BYTES: usize = 32;

    fn code(packet: &[u8], p: usize) -> u8 {
        (packet[(p / 32) * 18 + 2 + p % 16] >> ((p % 32 / 16) * 4)) & 15
    }
    /// Each block's f16 scale, widened exactly to `f32`.
    fn supers(packet: &[u8], out: &mut [u8]) {
        for (block, scale) in out.chunks_exact_mut(4).enumerate() {
            let bits = u16::from_le_bytes([packet[block * 18], packet[block * 18 + 1]]);
            scale.copy_from_slice(&f16_to_f32(bits).to_le_bytes());
        }
    }
}

/// The nibble of value `p` of a 32-value GGUF block whose `qs[16]` start at
/// byte `at`: the low (p < 16) or high nibble of `qs[p % 16]`.
fn block_nibble(packet: &[u8], at: usize, p: usize) -> u8 {
    (packet[at + p % 16] >> ((p / 16) * 4)) & 15
}

/// One GGUF iq4_nl block (`d | qs[16]`) into iq4g32s: the codes and the f16
/// scale move bit for bit.
struct GgufIq4NlBlock;

impl Conversion for GgufIq4NlBlock {
    const SOURCE: &'static str = "gguf_iq4_nl_block";
    const PACKET_BYTES: usize = 18;
    const GROUP: usize = 32;
    const BITS: u32 = 4;
    const SCALE_BYTES: usize = 0;
    const SUPER_BYTES: usize = 2;

    fn code(packet: &[u8], p: usize) -> u8 {
        block_nibble(packet, 2, p)
    }
    fn supers(packet: &[u8], out: &mut [u8]) {
        out.copy_from_slice(&packet[..2]);
    }
}

/// GGUF q4_0 (`d | qs[16]`) into q4g32s.
struct GgufQ4_0;

impl Conversion for GgufQ4_0 {
    const SOURCE: &'static str = "gguf_q4_0";
    const PACKET_BYTES: usize = 18;
    const GROUP: usize = 32;
    const BITS: u32 = 4;
    const SCALE_BYTES: usize = 0;
    const SUPER_BYTES: usize = 2;

    fn code(packet: &[u8], p: usize) -> u8 {
        block_nibble(packet, 2, p)
    }
    fn supers(packet: &[u8], out: &mut [u8]) {
        out.copy_from_slice(&packet[..2]);
    }
}

/// GGUF q5_0 (`d | qh[4] | qs[16]`) and q5_1 (`d | m | qh[4] | qs[16]`) into
/// q5g32s and q5g32: bit p of `qh` is the fifth bit of value p.
struct GgufQ5<const MINIMUM: bool>;

impl<const MINIMUM: bool> Conversion for GgufQ5<MINIMUM> {
    const SOURCE: &'static str = if MINIMUM { "gguf_q5_1" } else { "gguf_q5_0" };
    const PACKET_BYTES: usize = if MINIMUM { 24 } else { 22 };
    const GROUP: usize = 32;
    const BITS: u32 = 5;
    const SCALE_BYTES: usize = 0;
    const SUPER_BYTES: usize = if MINIMUM { 4 } else { 2 };

    fn code(packet: &[u8], p: usize) -> u8 {
        let high = Self::SUPER_BYTES;
        block_nibble(packet, high + 4, p) | (((packet[high + p / 8] >> (p % 8)) & 1) << 4)
    }
    fn supers(packet: &[u8], out: &mut [u8]) {
        out.copy_from_slice(&packet[..Self::SUPER_BYTES]);
    }
}

/// The resident E2M1 code of a GGUF MXFP4 / NVFP4 code: code 8 (E2M1 -0),
/// which ggml decodes as +0, becomes code 0.
fn e2m1_positive_zero(code: u8) -> u8 {
    if code == 8 {
        0
    } else {
        code
    }
}

/// GGUF mxfp4 (`e | qs[16]`) into mxfp4g32.
struct GgufMxfp4;

impl Conversion for GgufMxfp4 {
    const SOURCE: &'static str = "gguf_mxfp4";
    const PACKET_BYTES: usize = 17;
    const GROUP: usize = 32;
    const BITS: u32 = 4;
    const SCALE_BYTES: usize = 0;
    const SUPER_BYTES: usize = 1;

    fn code(packet: &[u8], p: usize) -> u8 {
        e2m1_positive_zero(block_nibble(packet, 1, p))
    }
    fn supers(packet: &[u8], out: &mut [u8]) {
        out.copy_from_slice(&packet[..1]);
    }
}

/// GGUF nvfp4 (`d[4] | qs[32]`) into nvfp4g16: value p of sixteen-value
/// sub-block s = p / 16 is the low (p % 16 < 8) or high nibble of
/// `qs[8 s + p % 8]`.
struct GgufNvfp4;

impl Conversion for GgufNvfp4 {
    const SOURCE: &'static str = "gguf_nvfp4";
    const PACKET_BYTES: usize = 36;
    const GROUP: usize = 64;
    const BITS: u32 = 4;
    const SCALE_BYTES: usize = 0;
    const SUPER_BYTES: usize = 4;

    fn code(packet: &[u8], p: usize) -> u8 {
        let (block, within) = (p / 16, p % 16);
        e2m1_positive_zero((packet[4 + 8 * block + within % 8] >> ((within / 8) * 4)) & 15)
    }
    fn supers(packet: &[u8], out: &mut [u8]) {
        out.copy_from_slice(&packet[..4]);
    }
}

/// Converts the source row at `source` (whole packets of `k` values) into one
/// row of `geometry`'s planes at `row`, each storage group contiguous (a
/// `rows16` row; `rows` interleaves it into a `rows8` tile), zeroing every
/// stored byte no plane occupies.
///
/// # Safety
/// `source` addresses `ceil(k / GROUP)` packets and `row` one target row of
/// `k` values.
unsafe fn convert_row<C: Conversion>(
    source: *const u8,
    row: *mut u8,
    k: usize,
    geometry: &RowGeometry,
) {
    let [codes_plane, high_plane, scales_plane, supers_plane] =
        RowPlaneRole::ALL.map(|role| geometry.offsets[role as usize]);
    let groups = k.div_ceil(C::GROUP);
    let source = unsafe { std::slice::from_raw_parts(source, groups * C::PACKET_BYTES) };
    let out = unsafe { std::slice::from_raw_parts_mut(row, geometry.stride) };
    out.fill(0);
    let mut codes = [0u8; 256];
    for (g, packet) in source.chunks_exact(C::PACKET_BYTES).enumerate() {
        if C::GRID_BYTES > 0 {
            C::grids(
                packet,
                &mut out[codes_plane + g * C::GRID_BYTES..][..C::GRID_BYTES],
            );
            C::scales(
                packet,
                &mut out[scales_plane + g * C::SCALE_BYTES..][..C::SCALE_BYTES],
            );
            C::supers(
                packet,
                &mut out[supers_plane + g * C::SUPER_BYTES..][..C::SUPER_BYTES],
            );
            continue;
        }
        let codes = &mut codes[..C::GROUP];
        for (p, code) in codes.iter_mut().enumerate() {
            *code = C::code(packet, p);
        }
        if C::BITS == 8 {
            out[codes_plane + g * C::GROUP..][..C::GROUP].copy_from_slice(codes);
        } else {
            let low = &mut out[codes_plane + g * C::GROUP / 2..][..C::GROUP / 2];
            for (i, byte) in low.iter_mut().enumerate() {
                *byte = (codes[2 * i] & 15) | ((codes[2 * i + 1] & 15) << 4);
            }
        }
        match C::BITS {
            5 => {
                let high = &mut out[high_plane + g * C::GROUP / 8..][..C::GROUP / 8];
                for (i, code) in codes.iter().enumerate() {
                    high[i / 8] |= ((code >> 4) & 1) << (i % 8);
                }
            }
            6 => {
                let high = &mut out[high_plane + g * C::GROUP / 4..][..C::GROUP / 4];
                for (i, code) in codes.iter().enumerate() {
                    high[i / 4] |= ((code >> 4) & 3) << (2 * (i % 4));
                }
            }
            _ => {}
        }
        if C::SCALE_BYTES > 0 {
            C::scales(
                packet,
                &mut out[scales_plane + g * C::SCALE_BYTES..][..C::SCALE_BYTES],
            );
        }
        C::supers(
            packet,
            &mut out[supers_plane + g * C::SUPER_BYTES..][..C::SUPER_BYTES],
        );
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
    convert: unsafe fn(*const u8, *mut u8, usize, &RowGeometry),
}

const fn conversion<C: Conversion>(target: &'static str) -> RowConversion {
    RowConversion {
        source: C::SOURCE,
        target,
        packet_bytes: C::PACKET_BYTES,
        group: C::GROUP,
        convert: convert_row::<C>,
    }
}

static CONVERSIONS: [RowConversion; 34] = [
    conversion::<GgufIq4NlBlock>("iq4g32s@rows16"),
    conversion::<GgufIq4NlBlock>("iq4g32s@rows8"),
    conversion::<GgufIq2Xs>("iq2xs@rows16"),
    conversion::<GgufIq3Xxs>("iq3xxs@rows16"),
    conversion::<GgufIq2Xxs>("iq2xxs@rows16"),
    conversion::<GgufIq2Xs>("iq2xs@rows8"),
    conversion::<GgufIq3Xxs>("iq3xxs@rows8"),
    conversion::<GgufIq2Xxs>("iq2xxs@rows8"),
    conversion::<GgufQ4_0>("q4g32s@rows16"),
    conversion::<GgufQ5<false>>("q5g32s@rows16"),
    conversion::<GgufQ5<true>>("q5g32@rows16"),
    conversion::<GgufMxfp4>("mxfp4g32@rows16"),
    conversion::<GgufNvfp4>("nvfp4g16@rows16"),
    conversion::<GgufQ4_0>("q4g32s@rows8"),
    conversion::<GgufQ5<false>>("q5g32s@rows8"),
    conversion::<GgufQ5<true>>("q5g32@rows8"),
    conversion::<GgufMxfp4>("mxfp4g32@rows8"),
    conversion::<GgufNvfp4>("nvfp4g16@rows8"),
    conversion::<GgufQ3K>("q6k@rows16"),
    conversion::<GgufIq3S>("q6k@rows16"),
    conversion::<GgufQ4K>("q4k@rows16"),
    conversion::<GgufQ5K>("q5k@rows16"),
    conversion::<GgufQ6K>("q6k@rows16"),
    conversion::<GgufQ8>("q8g32s@rows16"),
    conversion::<GgufIq4Nl>("iq4g32@rows16"),
    conversion::<GgufIq4Xs>("iq4g32@rows16"),
    conversion::<GgufQ3K>("q6k@rows8"),
    conversion::<GgufIq3S>("q6k@rows8"),
    conversion::<GgufQ4K>("q4k@rows8"),
    conversion::<GgufQ5K>("q5k@rows8"),
    conversion::<GgufQ6K>("q6k@rows8"),
    conversion::<GgufQ8>("q8g32s@rows8"),
    conversion::<GgufIq4Nl>("iq4g32@rows8"),
    conversion::<GgufIq4Xs>("iq4g32@rows8"),
];

/// Every registered conversion into a resident row layout.
pub fn conversions() -> &'static [RowConversion] {
    &CONVERSIONS
}

/// The conversion of `source` into `target`, when one is registered.
pub fn resolve(source: &str, target: &str) -> Option<&'static RowConversion> {
    CONVERSIONS
        .iter()
        .find(|conversion| conversion.source == source && conversion.target == target)
}

/// Leading-axis row addressing shared by [`External`] and [`Packed`]: a
/// tensor's rows are the positions of every axis before the last, which a
/// canonical view stores `row_bytes` apart.
fn canonical_rows(extents: &[u64], strides: &[u64], row_stride: u64) -> usize {
    let rank = extents.len();
    assert!(rank >= 1, "a converted operand has a value axis");
    let mut expected = row_stride;
    for axis in (0..rank - 1).rev() {
        assert_eq!(
            strides[axis], expected,
            "a converted operand's rows are canonical"
        );
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
    pub unsafe fn from_tensor(
        base: *const u8,
        extents: &[u64],
        strides: &[u64],
        representation: &'static str,
    ) -> Self {
        let k = extents[extents.len() - 1];
        let (group, _) = external_packet(representation);
        let rows = canonical_rows(extents, strides, k.div_ceil(group as u64));
        Self {
            base,
            rows,
            k: k as usize,
            representation,
            matrix_rows: extents[extents.len() - 2] as usize,
            life: PhantomData,
        }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn representation(&self) -> &'static str {
        self.representation
    }

    /// The stored bytes of row `row`: its whole packets, as stored.
    pub fn row(&self, row: usize) -> &'a [u8] {
        assert!(row < self.rows, "row {row} outside the operand");
        let (group, packet_bytes) = external_packet(self.representation);
        let bytes = self.k.div_ceil(group) * packet_bytes;
        // SAFETY: the operand's rows are canonical and lie within it.
        unsafe { std::slice::from_raw_parts(self.base.add(row * bytes), bytes) }
    }
}

/// The values and bytes of a packet of the external encoding
/// `representation`.
fn external_packet(representation: &str) -> (usize, usize) {
    CONVERSIONS
        .iter()
        .find(|conversion| conversion.source == representation)
        .map(|conversion| (conversion.group, conversion.packet_bytes))
        .unwrap_or_else(|| panic!("`{representation}` is not a registered external source"))
}

/// A bound result tensor of a packed `rows16` or `rows8` element, written
/// only by a conversion.
#[derive(Clone, Copy, Debug)]
pub struct Packed<'a> {
    base: *mut u8,
    rows: usize,
    k: usize,
    element: &'static RepresentationInfo,
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
    pub unsafe fn from_tensor(
        base: *mut u8,
        extents: &[u64],
        strides: &[u64],
        representation: &'static str,
    ) -> Self {
        let element = registry::representation_info(
            registry::representation(representation)
                .unwrap_or_else(|| panic!("`{representation}` is no registered element")),
        );
        let rows = if element.layout == Layout::Rows8 {
            let rank = extents.len();
            assert!(rank >= 2, "Rows8 requires a row axis");
            assert_eq!(strides[rank - 1], 1);
            let mut expected = 1u64;
            for axis in (0..rank - 1).rev() {
                assert_eq!(strides[axis], expected, "Rows8 tile strides");
                expected *= if axis == rank - 2 {
                    extents[axis].div_ceil(8) * 8
                } else {
                    extents[axis]
                };
            }
            extents[..rank - 1].iter().product::<u64>() as usize
        } else {
            canonical_rows(extents, strides, 1)
        };
        Self {
            base,
            rows,
            k: extents[extents.len() - 1] as usize,
            element,
            matrix_rows: extents[extents.len() - 2] as usize,
            life: PhantomData,
        }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }
}

/// Converts rows `rows` of `source` into the same rows of `destination`.
///
/// # Safety
/// No other work item writes these rows of `destination` in the launch.
pub unsafe fn rows(source: &External<'_>, destination: &Packed<'_>, rows: std::ops::Range<usize>) {
    let conversion =
        resolve(source.representation, destination.element.name).unwrap_or_else(|| {
            panic!(
                "no CPU conversion of `{}` into `{}`",
                source.representation, destination.element.name
            )
        });
    assert_eq!(source.k, destination.k, "a conversion keeps the row length");
    assert!(
        rows.end <= source.rows && rows.end <= destination.rows,
        "rows {rows:?} outside the operands"
    );
    let source_row = source.k.div_ceil(conversion.group) * conversion.packet_bytes;
    let geometry = RowGeometry::of(destination.element, destination.k, 0);
    let target_row = geometry.stride;
    assert_eq!(source.matrix_rows, destination.matrix_rows);
    if geometry.rows8 {
        let n = source.matrix_rows;
        assert!(
            rows.start / n == (rows.end - 1) / n,
            "a Rows8 work item stays within one matrix"
        );
        assert_eq!(rows.start % n % 8, 0, "a Rows8 work item starts on a tile");
        assert!(
            rows.end % n == 0 || rows.end % n % 8 == 0,
            "a Rows8 work item ends on a tile"
        );
        let groups = source.k.div_ceil(conversion.group);
        for first in (rows.start..rows.end).step_by(8) {
            let matrix = first / n;
            let in_matrix = first % n;
            let tile = matrix * n.div_ceil(8) + in_matrix / 8;
            // SAFETY: this work item owns the entire tile, including its padded rows.
            let output = unsafe {
                std::slice::from_raw_parts_mut(
                    destination.base.add(tile * 8 * target_row),
                    8 * target_row,
                )
            };
            output.fill(0);
            for r in 0..(rows.end - first).min(8) {
                let mut packed = vec![0u8; geometry.stride];
                unsafe {
                    (conversion.convert)(
                        source.base.add((first + r) * source_row),
                        packed.as_mut_ptr(),
                        source.k,
                        &geometry,
                    )
                };
                for (offset, bytes) in geometry.offsets.iter().zip(geometry.groups) {
                    if bytes == 0 {
                        continue;
                    }
                    for group in 0..groups {
                        let from = offset + group * bytes;
                        let to = offset * 8 + group * bytes * 8 + r * bytes;
                        output[to..to + bytes].copy_from_slice(&packed[from..from + bytes]);
                    }
                }
            }
        }
    } else {
        for row in rows {
            // SAFETY: the row lies inside both operands (`from_tensor`); the
            // caller owns its destination row.
            unsafe {
                (conversion.convert)(
                    source.base.add(row * source_row),
                    destination.base.add(row * target_row),
                    source.k,
                    &geometry,
                )
            };
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
                let registered = seismic_lang::registry::representation_conversion(
                    representation(conversion.source).unwrap(),
                    representation(conversion.target).unwrap(),
                )
                .expect("registered conversion");
                // The scales a conversion converts as numbers are finite: NaN
                // payloads are not part of the conversion contract.
                registered.admit_source(&mut source);
                let id = registered.id;
                let expected =
                    seismic_lang::interp::repack(id, &[rows, k], &source).expect("registry repack");
                let geometry = RowGeometry::of(
                    registry::representation_info(representation(conversion.target).unwrap()),
                    k,
                    0,
                );
                let stride = geometry.stride;
                let stored_rows = if geometry.rows8 {
                    rows.div_ceil(8) * 8
                } else {
                    rows
                };
                let mut out = vec![0xa5u8; stored_rows * stride];
                if geometry.rows8 {
                    let from = unsafe {
                        External::from_tensor(
                            source.as_ptr(),
                            &[rows as u64, k as u64],
                            &[packets as u64, 1],
                            conversion.source,
                        )
                    };
                    let to = unsafe {
                        Packed::from_tensor(
                            out.as_mut_ptr(),
                            &[rows as u64, k as u64],
                            &[1, 1],
                            conversion.target,
                        )
                    };
                    unsafe { super::rows(&from, &to, 0..rows) };
                } else {
                    for row in 0..rows {
                        unsafe {
                            (conversion.convert)(
                                source.as_ptr().add(row * packets * conversion.packet_bytes),
                                out.as_mut_ptr().add(row * stride),
                                k,
                                &geometry,
                            )
                        };
                    }
                }
                assert_eq!(
                    out.len(),
                    expected.len(),
                    "{} -> {} k {k}",
                    conversion.source,
                    conversion.target
                );
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
