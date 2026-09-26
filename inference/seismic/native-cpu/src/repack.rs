//! Exact conversions of external GGUF weights into resident `rows16` storage:
//! the registry's registered `repack`. Codes and coefficients move bit for
//! bit, except that `iq4g32` scales are formed (each an exact product or
//! widening) and q3_K and iq3_s codes and scales are re-expressed as q6k's
//! (every value is unchanged); nothing is decoded. One source packet becomes
//! one storage group of its row.
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
    type Target = Q6K;
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

/// GGUF iq3_s into q6k: each value's signed grid magnitude re-based by 32,
/// each 32-value sub-block's scale `1 + 2 s` on both its sixteen-value
/// halves; the f16 super-scale moves bit for bit.
struct GgufIq3S;

impl Conversion for GgufIq3S {
    const SOURCE: &'static str = "gguf_iq3_s";
    type Target = Q6K;
    const PACKET_BYTES: usize = 110;
    const GROUP: usize = 256;
    const BITS: u32 = 6;
    const SCALE_BYTES: usize = 16;
    const SUPER_BYTES: usize = 2;

    fn code(packet: &[u8], p: usize) -> u8 {
        let (block, group, slot) = (p / 32, p % 32 / 8, p % 8);
        let point = usize::from(packet[2 + 8 * block + 2 * group + slot / 4])
            | usize::from((packet[66 + block] >> (2 * group + slot / 4)) & 1) << 8;
        let magnitude = (IQ3S_GRID[point] >> (8 * (slot % 4))) as u8;
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

/// ggml's `iq3s_grid`: byte `j` of point `i` is magnitude `j` (odd, 1..=15).
static IQ3S_GRID: [u32; 512] = [
    0x01010101, 0x01010103, 0x01010105, 0x0101010b, 0x0101010f, 0x01010301, 0x01010303, 0x01010305,
    0x01010309, 0x0101030d, 0x01010501, 0x01010503, 0x0101050b, 0x01010707, 0x01010901, 0x01010905,
    0x0101090b, 0x0101090f, 0x01010b03, 0x01010b07, 0x01010d01, 0x01010d05, 0x01010f03, 0x01010f09,
    0x01010f0f, 0x01030101, 0x01030103, 0x01030105, 0x01030109, 0x01030301, 0x01030303, 0x0103030b,
    0x01030501, 0x01030507, 0x0103050f, 0x01030703, 0x0103070b, 0x01030909, 0x01030d03, 0x01030d0b,
    0x01030f05, 0x01050101, 0x01050103, 0x0105010b, 0x0105010f, 0x01050301, 0x01050307, 0x0105030d,
    0x01050503, 0x0105050b, 0x01050701, 0x01050709, 0x01050905, 0x0105090b, 0x0105090f, 0x01050b03,
    0x01050b07, 0x01050f01, 0x01050f07, 0x01070107, 0x01070303, 0x0107030b, 0x01070501, 0x01070505,
    0x01070703, 0x01070707, 0x0107070d, 0x01070909, 0x01070b01, 0x01070b05, 0x01070d0f, 0x01070f03,
    0x01070f0b, 0x01090101, 0x01090307, 0x0109030f, 0x01090503, 0x01090509, 0x01090705, 0x01090901,
    0x01090907, 0x01090b03, 0x01090f01, 0x010b0105, 0x010b0109, 0x010b0501, 0x010b0505, 0x010b050d,
    0x010b0707, 0x010b0903, 0x010b090b, 0x010b090f, 0x010b0d0d, 0x010b0f07, 0x010d010d, 0x010d0303,
    0x010d0307, 0x010d0703, 0x010d0b05, 0x010d0f03, 0x010f0101, 0x010f0105, 0x010f0109, 0x010f0501,
    0x010f0505, 0x010f050d, 0x010f0707, 0x010f0b01, 0x010f0b09, 0x03010101, 0x03010103, 0x03010105,
    0x03010109, 0x03010301, 0x03010303, 0x03010307, 0x0301030b, 0x0301030f, 0x03010501, 0x03010505,
    0x03010703, 0x03010709, 0x0301070d, 0x03010b09, 0x03010b0d, 0x03010d03, 0x03010f05, 0x03030101,
    0x03030103, 0x03030107, 0x0303010d, 0x03030301, 0x03030309, 0x03030503, 0x03030701, 0x03030707,
    0x03030903, 0x03030b01, 0x03030b05, 0x03030f01, 0x03030f0d, 0x03050101, 0x03050305, 0x0305030b,
    0x0305030f, 0x03050501, 0x03050509, 0x03050705, 0x03050901, 0x03050907, 0x03050b0b, 0x03050d01,
    0x03050f05, 0x03070103, 0x03070109, 0x0307010f, 0x03070301, 0x03070307, 0x03070503, 0x0307050f,
    0x03070701, 0x03070709, 0x03070903, 0x03070d05, 0x03070f01, 0x03090107, 0x0309010b, 0x03090305,
    0x03090309, 0x03090703, 0x03090707, 0x03090905, 0x0309090d, 0x03090b01, 0x03090b09, 0x030b0103,
    0x030b0301, 0x030b0307, 0x030b0503, 0x030b0701, 0x030b0705, 0x030b0b03, 0x030d0501, 0x030d0509,
    0x030d050f, 0x030d0909, 0x030d090d, 0x030f0103, 0x030f0107, 0x030f0301, 0x030f0305, 0x030f0503,
    0x030f070b, 0x030f0903, 0x030f0d05, 0x030f0f01, 0x05010101, 0x05010103, 0x05010107, 0x0501010b,
    0x0501010f, 0x05010301, 0x05010305, 0x05010309, 0x0501030d, 0x05010503, 0x05010507, 0x0501050f,
    0x05010701, 0x05010705, 0x05010903, 0x05010907, 0x0501090b, 0x05010b01, 0x05010b05, 0x05010d0f,
    0x05010f01, 0x05010f07, 0x05010f0b, 0x05030101, 0x05030105, 0x05030301, 0x05030307, 0x0503030f,
    0x05030505, 0x0503050b, 0x05030703, 0x05030709, 0x05030905, 0x05030b03, 0x05050103, 0x05050109,
    0x0505010f, 0x05050503, 0x05050507, 0x05050701, 0x0505070f, 0x05050903, 0x05050b07, 0x05050b0f,
    0x05050f03, 0x05050f09, 0x05070101, 0x05070105, 0x0507010b, 0x05070303, 0x05070505, 0x05070509,
    0x05070703, 0x05070707, 0x05070905, 0x05070b01, 0x05070d0d, 0x05090103, 0x0509010f, 0x05090501,
    0x05090507, 0x05090705, 0x0509070b, 0x05090903, 0x05090f05, 0x05090f0b, 0x050b0109, 0x050b0303,
    0x050b0505, 0x050b070f, 0x050b0901, 0x050b0b07, 0x050b0f01, 0x050d0101, 0x050d0105, 0x050d010f,
    0x050d0503, 0x050d0b0b, 0x050d0d03, 0x050f010b, 0x050f0303, 0x050f050d, 0x050f0701, 0x050f0907,
    0x050f0b01, 0x07010105, 0x07010303, 0x07010307, 0x0701030b, 0x0701030f, 0x07010505, 0x07010703,
    0x07010707, 0x0701070b, 0x07010905, 0x07010909, 0x0701090f, 0x07010b03, 0x07010d07, 0x07010f03,
    0x07030103, 0x07030107, 0x0703010b, 0x07030309, 0x07030503, 0x07030507, 0x07030901, 0x07030d01,
    0x07030f05, 0x07030f0d, 0x07050101, 0x07050305, 0x07050501, 0x07050705, 0x07050709, 0x07050b01,
    0x07070103, 0x07070301, 0x07070309, 0x07070503, 0x07070507, 0x0707050f, 0x07070701, 0x07070903,
    0x07070907, 0x0707090f, 0x07070b0b, 0x07070f07, 0x07090107, 0x07090303, 0x0709030d, 0x07090505,
    0x07090703, 0x07090b05, 0x07090d01, 0x07090d09, 0x070b0103, 0x070b0301, 0x070b0305, 0x070b050b,
    0x070b0705, 0x070b0909, 0x070b0b0d, 0x070b0f07, 0x070d030d, 0x070d0903, 0x070f0103, 0x070f0107,
    0x070f0501, 0x070f0505, 0x070f070b, 0x09010101, 0x09010109, 0x09010305, 0x09010501, 0x09010509,
    0x0901050f, 0x09010705, 0x09010903, 0x09010b01, 0x09010f01, 0x09030105, 0x0903010f, 0x09030303,
    0x09030307, 0x09030505, 0x09030701, 0x0903070b, 0x09030907, 0x09030b03, 0x09030b0b, 0x09050103,
    0x09050107, 0x09050301, 0x0905030b, 0x09050503, 0x09050707, 0x09050901, 0x09050b0f, 0x09050d05,
    0x09050f01, 0x09070109, 0x09070303, 0x09070307, 0x09070501, 0x09070505, 0x09070703, 0x0907070b,
    0x09090101, 0x09090105, 0x09090509, 0x0909070f, 0x09090901, 0x09090f03, 0x090b010b, 0x090b010f,
    0x090b0503, 0x090b0d05, 0x090d0307, 0x090d0709, 0x090d0d01, 0x090f0301, 0x090f030b, 0x090f0701,
    0x090f0907, 0x090f0b03, 0x0b010105, 0x0b010301, 0x0b010309, 0x0b010505, 0x0b010901, 0x0b010909,
    0x0b01090f, 0x0b010b05, 0x0b010d0d, 0x0b010f09, 0x0b030103, 0x0b030107, 0x0b03010b, 0x0b030305,
    0x0b030503, 0x0b030705, 0x0b030f05, 0x0b050101, 0x0b050303, 0x0b050507, 0x0b050701, 0x0b05070d,
    0x0b050b07, 0x0b070105, 0x0b07010f, 0x0b070301, 0x0b07050f, 0x0b070909, 0x0b070b03, 0x0b070d0b,
    0x0b070f07, 0x0b090103, 0x0b090109, 0x0b090501, 0x0b090705, 0x0b09090d, 0x0b0b0305, 0x0b0b050d,
    0x0b0b0b03, 0x0b0b0b07, 0x0b0d0905, 0x0b0f0105, 0x0b0f0109, 0x0b0f0505, 0x0d010303, 0x0d010307,
    0x0d01030b, 0x0d010703, 0x0d010707, 0x0d010d01, 0x0d030101, 0x0d030501, 0x0d03050f, 0x0d030d09,
    0x0d050305, 0x0d050709, 0x0d050905, 0x0d050b0b, 0x0d050d05, 0x0d050f01, 0x0d070101, 0x0d070309,
    0x0d070503, 0x0d070901, 0x0d09050b, 0x0d090907, 0x0d090d05, 0x0d0b0101, 0x0d0b0107, 0x0d0b0709,
    0x0d0b0d01, 0x0d0d010b, 0x0d0d0901, 0x0d0f0303, 0x0d0f0307, 0x0f010101, 0x0f010109, 0x0f01010f,
    0x0f010501, 0x0f010505, 0x0f01070d, 0x0f010901, 0x0f010b09, 0x0f010d05, 0x0f030105, 0x0f030303,
    0x0f030509, 0x0f030907, 0x0f03090b, 0x0f050103, 0x0f050109, 0x0f050301, 0x0f05030d, 0x0f050503,
    0x0f050701, 0x0f050b03, 0x0f070105, 0x0f070705, 0x0f07070b, 0x0f070b07, 0x0f090103, 0x0f09010b,
    0x0f090307, 0x0f090501, 0x0f090b01, 0x0f0b0505, 0x0f0b0905, 0x0f0d0105, 0x0f0d0703, 0x0f0f0101,
];

/// Eight consecutive GGUF iq4_nl blocks (one iq4g32 storage group).
struct GgufIq4Nl;

impl Conversion for GgufIq4Nl {
    const SOURCE: &'static str = "gguf_iq4_nl";
    type Target = Iq4;
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
            C::scales(
                packet,
                &mut out[geometry.scales + g * C::SCALE_BYTES..][..C::SCALE_BYTES],
            );
        }
        C::supers(
            packet,
            &mut out[geometry.supers + g * C::SUPER_BYTES..][..C::SUPER_BYTES],
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
    geometry: fn(usize, usize) -> RowGeometry,
    /// Eight-row block interleaving rather than contiguous rows.
    rows8: bool,
    /// Bytes per storage group of each plane, in storage order.
    planes: &'static [usize],
    convert: unsafe fn(*const u8, *mut u8, usize),
}

const fn conversion<C: Conversion>(
    target: &'static str,
    rows8: bool,
    planes: &'static [usize],
) -> RowConversion {
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

static CONVERSIONS: [RowConversion; 16] = [
    conversion::<GgufQ3K>("q6k@rows16", false, &[128, 64, 16, 2]),
    conversion::<GgufIq3S>("q6k@rows16", false, &[128, 64, 16, 2]),
    conversion::<GgufQ4K>("q4k@rows16", false, &[128, 12, 4]),
    conversion::<GgufQ5K>("q5k@rows16", false, &[128, 32, 12, 4]),
    conversion::<GgufQ6K>("q6k@rows16", false, &[128, 64, 16, 2]),
    conversion::<GgufQ8>("q8g32s@rows16", false, &[32, 2]),
    conversion::<GgufIq4Nl>("iq4g32@rows16", false, &[128, 32]),
    conversion::<GgufIq4Xs>("iq4g32@rows16", false, &[128, 32]),
    conversion::<GgufQ3K>("q6k@rows8", true, &[128, 64, 16, 2]),
    conversion::<GgufIq3S>("q6k@rows8", true, &[128, 64, 16, 2]),
    conversion::<GgufQ4K>("q4k@rows8", true, &[128, 12, 4]),
    conversion::<GgufQ5K>("q5k@rows8", true, &[128, 32, 12, 4]),
    conversion::<GgufQ6K>("q6k@rows8", true, &[128, 64, 16, 2]),
    conversion::<GgufQ8>("q8g32s@rows8", true, &[32, 2]),
    conversion::<GgufIq4Nl>("iq4g32@rows8", true, &[128, 32]),
    conversion::<GgufIq4Xs>("iq4g32@rows8", true, &[128, 32]),
];

/// Every registered conversion into a `rows16` representation.
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
        let group = CONVERSIONS
            .iter()
            .find(|conversion| conversion.source == representation)
            .unwrap_or_else(|| panic!("`{representation}` is not a registered external source"))
            .group as u64;
        let rows = canonical_rows(extents, strides, k.div_ceil(group));
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
    pub unsafe fn from_tensor(
        base: *mut u8,
        extents: &[u64],
        strides: &[u64],
        representation: &'static str,
    ) -> Self {
        let rows = if representation.ends_with("@rows8") {
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
}

/// Converts rows `rows` of `source` into the same rows of `destination`.
///
/// # Safety
/// No other work item writes these rows of `destination` in the launch.
pub unsafe fn rows(source: &External<'_>, destination: &Packed<'_>, rows: std::ops::Range<usize>) {
    let conversion =
        resolve(source.representation, destination.representation).unwrap_or_else(|| {
            panic!(
                "no CPU conversion of `{}` into `{}`",
                source.representation, destination.representation
            )
        });
    assert_eq!(source.k, destination.k, "a conversion keeps the row length");
    assert!(
        rows.end <= source.rows && rows.end <= destination.rows,
        "rows {rows:?} outside the operands"
    );
    let source_row = source.k.div_ceil(conversion.group) * conversion.packet_bytes;
    let target_row = (conversion.geometry)(destination.k, 0).stride;
    assert_eq!(source.matrix_rows, destination.matrix_rows);
    if conversion.rows8 {
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
                    )
                };
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
            unsafe {
                (conversion.convert)(
                    source.base.add(row * source_row),
                    destination.base.add(row * target_row),
                    source.k,
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
                if conversion.source == "gguf_iq4_xs" {
                    // Finite f16 bases (exponent below 16): the scales are
                    // formed products, and NaN payloads are not part of the
                    // conversion contract.
                    for packet in source.chunks_exact_mut(conversion.packet_bytes) {
                        packet[1] &= 0xbf;
                    }
                }
                if conversion.source == "gguf_iq4_nl" {
                    // Finite f16 block scales: NaN payloads are not part of
                    // the conversion contract.
                    for packet in source.chunks_exact_mut(conversion.packet_bytes) {
                        for block in packet.chunks_exact_mut(18) {
                            block[1] &= 0xbf;
                        }
                    }
                }
                let id = seismic_lang::registry::representation_conversion(
                    representation(conversion.source).unwrap(),
                    representation(conversion.target).unwrap(),
                )
                .expect("registered conversion")
                .id;
                let expected =
                    seismic_lang::interp::repack(id, &[rows, k], &source).expect("registry repack");
                let stride = (conversion.geometry)(k, 0).stride;
                let stored_rows = if conversion.rows8 {
                    rows.div_ceil(8) * 8
                } else {
                    rows
                };
                let mut out = vec![0xa5u8; stored_rows * stride];
                if conversion.rows8 {
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
