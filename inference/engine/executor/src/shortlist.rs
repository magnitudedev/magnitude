//! Narrow coarse vocabulary projection (`ImportTransform::Shortlist`) for
//! draft-head shortlist readout.
//!
//! A draft head that projects through the target's full-precision vocabulary
//! table (`q5k`, `q6k`, `q8_0`, `bf16`, etc.) runs sequential `M = 1` draft
//! passes whose DRAM cost is dominated by reading the entire vocabulary table
//! per pass. A `Shortlist` weight stores a narrow 2-bit (`gguf_q2_0` /
//! `q2g64`, 2.25 bits/value) coarse copy of the target's vocabulary table:
//! each draft pass scans the coarse copy to select the top [`SHORTLIST_K`]
//! candidate tokens (`draft_top_k` -> `shortlist_token_rows`), then evaluates
//! the **exact** target vocabulary rows for those candidates via gathered
//! `readout_project_rows` (`I = 1`) and `place_logits_rows`.
//!
//! Because draft tokens are strictly proposals verified by the exact target
//! model (`readout_head_rows`), shortlist drafting has zero effect on the
//! target output distribution while preserving near-exact draft proposals and
//! exact proposal confidence on the shortlist.

use seismic::{DType, Element};

/// Number of top coarse candidates refined against the exact target LM head.
pub(crate) const SHORTLIST_K: u64 = 64;

/// Minimum vocabulary size for which a shortlist projection is offered.
/// Small synthetic test vocabularies keep direct full projection.
pub(crate) const MIN_VOCABULARY: u64 = 4096;

/// Values per `gguf_q2_0` (`q2g64`) block.
pub(crate) const GROUP: u64 = 64;

/// Bytes per `gguf_q2_0` (`q2g64`) block: 2-byte f16 scale + 16 bytes of 2-bit codes.
const Q2_0_BLOCK_BYTES: usize = 18;

/// Values per `gguf_q4_0` (`q4g32s`) block.
pub(crate) const Q4_0_GROUP: u64 = 32;

/// Bytes per `gguf_q4_0` (`q4g32s`) block: 2-byte f16 scale + 16 bytes of 4-bit codes.
const Q4_0_BLOCK_BYTES: usize = 18;

/// The GGUF upload element of a shortlist projection (`gguf_q2_0`, repacked
/// into `q2g64@rows16` / `q2g64@mma16` / `q2g64` by the standard importer).
pub(crate) fn upload_element() -> Element {
    Element::named("gguf_q2_0").expect("gguf_q2_0 is registered")
}

/// The GGUF upload element of a narrowed draft-head weight (`gguf_q4_0`,
/// repacked into `q4g32s@rows32` / `q4g32s@mma16` / `q4g32s@rows16` by the
/// standard importer).
pub(crate) fn q4_0_upload_element() -> Element {
    Element::named("gguf_q4_0").expect("gguf_q4_0 is registered")
}

/// Whether a 2D `[rows, columns]` or 3D `[experts, rows, columns]` tensor
/// stored as `source` admits `gguf_q4_0` draft requantization.
pub(crate) fn admits_draft_q4_0(source: Element, shape: &[u64]) -> bool {
    if !(shape.len() == 2 || shape.len() == 3) || shape.contains(&0) {
        return false;
    }
    let columns = *shape.last().expect("2D/3D shape");
    if !columns.is_multiple_of(Q4_0_GROUP) {
        return false;
    }
    source.dtype().is_some() || crate::planning::value_packets(source).is_some()
}

/// Whether a draft-head weight stored as `source` with `shape` benefits from
/// `gguf_q4_0` import requantization (strictly wider than `gguf_q4_0` by at
/// least 1.5x, e.g. `q8_0`, `f16`, `bf16`, `f32`).
pub(crate) fn benefits_from_draft_q4_0(source: Element, shape: &[u64]) -> bool {
    if !admits_draft_q4_0(source, shape) {
        return false;
    }
    let Ok(source_bytes) = source.canonical_byte_len(shape) else {
        return false;
    };
    let Ok(q4_bytes) = q4_0_upload_element().canonical_byte_len(shape) else {
        return false;
    };
    source_bytes.saturating_mul(2) >= q4_bytes.saturating_mul(3)
}

/// Whether a `[rows, columns]` matrix stored as `source` admits shortlist
/// quantization (independent of vocabulary-size / bandwidth-gain thresholds).
pub(crate) fn admits_matrix(source: Element, shape: &[u64]) -> bool {
    let [rows, columns] = shape else {
        return false;
    };
    if *rows == 0 || *columns == 0 || !columns.is_multiple_of(GROUP) {
        return false;
    }
    source.dtype().is_some() || crate::planning::value_packets(source).is_some()
}

/// Whether a target output projection stored as `source` with `shape` should
/// be accompanied by a narrow shortlist projection for a draft head:
/// - vocabulary is large enough (`V >= MIN_VOCABULARY`, `D % 64 == 0`),
/// - `source` is decodable on the host,
/// - `source` is strictly wider than `gguf_q2_0` by at least 1.5x (e.g.,
///   `q4k`, `q5k`, `q6k`, `q8_0`, `f16`, `bf16`, `f32`).
pub(crate) fn benefits_from_shortlist(source: Element, shape: &[u64]) -> bool {
    if !admits_matrix(source, shape) {
        return false;
    }
    let [vocabulary, _] = shape else {
        return false;
    };
    if *vocabulary < MIN_VOCABULARY {
        return false;
    }
    let Ok(source_bytes) = source.canonical_byte_len(shape) else {
        return false;
    };
    let Ok(coarse_bytes) = upload_element().canonical_byte_len(shape) else {
        return false;
    };
    source_bytes.saturating_mul(2) >= coarse_bytes.saturating_mul(3)
}

/// Quantize a `[rows, columns]` matrix stored as `source` into `gguf_q2_0`
/// bytes, parallelized across rows with bounded per-thread scratch.
pub(crate) fn quantize(source: Element, shape: &[u64], bytes: &[u8]) -> Result<Vec<u8>, String> {
    if !admits_matrix(source, shape) {
        return Err(format!(
            "shortlist quantization does not admit {} of shape {shape:?}",
            source.name()
        ));
    }
    quantize_rows(source, shape, bytes, GROUP as usize, Q2_0_BLOCK_BYTES, quantize_row_q2_0)
}

/// Quantize a 2D `[rows, columns]` or 3D `[experts, rows, columns]` tensor
/// stored as `source` into `gguf_q4_0` bytes, parallelized across rows.
pub(crate) fn quantize_q4_0(source: Element, shape: &[u64], bytes: &[u8]) -> Result<Vec<u8>, String> {
    if !admits_draft_q4_0(source, shape) {
        return Err(format!(
            "draft q4_0 quantization does not admit {} of shape {shape:?}",
            source.name()
        ));
    }
    quantize_rows(source, shape, bytes, Q4_0_GROUP as usize, Q4_0_BLOCK_BYTES, quantize_row_q4_0)
}

fn quantize_rows(
    source: Element,
    shape: &[u64],
    bytes: &[u8],
    group: usize,
    block_bytes: usize,
    encode_row: fn(&[f32], &mut [u8]),
) -> Result<Vec<u8>, String> {
    let columns = usize::try_from(*shape.last().expect("2D/3D shape"))
        .map_err(|_| "quantize columns exceed usize")?;
    let rows_u64 = shape[..shape.len() - 1]
        .iter()
        .try_fold(1u64, |acc, &dim| acc.checked_mul(dim))
        .ok_or("quantize rows overflow u64")?;
    let rows = usize::try_from(rows_u64).map_err(|_| "quantize rows exceed usize")?;
    let expected_src = usize::try_from(
        source
            .canonical_byte_len(shape)
            .map_err(|error| error.to_string())?,
    )
    .map_err(|_| "quantize source byte length exceeds usize")?;
    if bytes.len() != expected_src {
        return Err(format!(
            "quantize source byte length {} differs from expected {expected_src}",
            bytes.len()
        ));
    }
    let src_row_bytes = expected_src / rows;
    let dst_row_bytes = (columns / group) * block_bytes;
    let mut out = vec![0u8; rows * dst_row_bytes];

    let threads = std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(1)
        .min(rows.max(1));
    let rows_per_thread = rows.div_ceil(threads);
    let fallback_packet = if FastDecoder::of(source).is_some() {
        None
    } else {
        Some(
            crate::planning::value_packets(source)
                .ok_or_else(|| format!("{} has no packet representation", source.name()))?,
        )
    };
    let fast = FastDecoder::of(source);

    std::thread::scope(|scope| -> Result<(), String> {
        let mut handles = Vec::with_capacity(threads);
        for (chunk_idx, dst_chunk) in out
            .chunks_mut(rows_per_thread * dst_row_bytes)
            .enumerate()
        {
            let start_row = chunk_idx * rows_per_thread;
            let chunk_rows = dst_chunk.len() / dst_row_bytes;
            let src_chunk = &bytes[start_row * src_row_bytes..(start_row + chunk_rows) * src_row_bytes];
            handles.push(
                seismic::threads::spawn_scoped(scope, "draft-quantize", move || -> Result<(), String> {
                    let mut row_f32 = vec![0.0f32; columns];
                    for r in 0..chunk_rows {
                        let src_row = &src_chunk[r * src_row_bytes..(r + 1) * src_row_bytes];
                        let dst_row = &mut dst_chunk[r * dst_row_bytes..(r + 1) * dst_row_bytes];
                        if let Some(decoder) = fast {
                            decoder.decode_row(src_row, &mut row_f32);
                        } else {
                            let packet = fallback_packet.expect("fallback packet present");
                            let row_shape = [1u64, columns as u64];
                            let canonical = packet
                                .repack_host(source, &row_shape, src_row)
                                .ok_or_else(|| {
                                    format!("{} has no conversion into {}", source.name(), packet.name())
                                })?;
                            let values = packet
                                .decode_host(&row_shape, &canonical)
                                .ok_or_else(|| format!("{} does not decode on the host", packet.name()))?;
                            for (dst, src) in row_f32.iter_mut().zip(values) {
                                *dst = src as f32;
                            }
                        }
                        encode_row(&row_f32, dst_row);
                    }
                    Ok(())
                })
                .map_err(|error| format!("draft quantization worker spawn failed: {error}"))?,
            );
        }
        for handle in handles {
            handle
                .join()
                .map_err(|_| "draft quantization worker panicked".to_string())??;
        }
        Ok(())
    })?;

    Ok(out)
}

/// Quantize one row of F32 values (multiple of 32) into `gguf_q4_0` blocks.
fn quantize_row_q4_0(row: &[f32], out: &mut [u8]) {
    for (block_vals, block_out) in row
        .chunks_exact(Q4_0_GROUP as usize)
        .zip(out.chunks_exact_mut(Q4_0_BLOCK_BYTES))
    {
        let vals: &[f32; 32] = block_vals.try_into().expect("32 values");
        let dst: &mut [u8; 18] = block_out.try_into().expect("18 bytes");
        quantize_block_q4_0(vals, dst);
    }
}

/// Quantize 32 F32 values into one `gguf_q4_0` block (`d (f16) | qs[16]`,
/// value `p` = `(q - 8) * d`, low nibble = `p < 16`, high nibble = `p >= 16`).
fn quantize_block_q4_0(values: &[f32; 32], out: &mut [u8; 18]) {
    let mut amax = 0.0f32;
    let mut signed_max = 0.0f32;
    for &v in values {
        let a = v.abs();
        if a > amax {
            amax = a;
            signed_max = v;
        }
    }
    if amax == 0.0 || !amax.is_finite() {
        *out = [0u8; 18];
        return;
    }
    let d = signed_max / -8.0;
    let d_bits = seismic::f16_bits(d);
    let d_f16 = seismic::exact_f64(DType::F16, u32::from(d_bits)) as f32;
    let inv = if d_f16 != 0.0 { 1.0 / d_f16 } else { 0.0 };
    out[0..2].copy_from_slice(&d_bits.to_le_bytes());
    for j in 0..16 {
        let lo = (values[j] * inv + 8.5).floor().clamp(0.0, 15.0) as u8;
        let hi = (values[j + 16] * inv + 8.5).floor().clamp(0.0, 15.0) as u8;
        out[2 + j] = lo | (hi << 4);
    }
}

/// Quantize one row of F32 values (multiple of 64) into `gguf_q2_0` blocks.
fn quantize_row_q2_0(row: &[f32], out: &mut [u8]) {
    for (block_vals, block_out) in row
        .chunks_exact(GROUP as usize)
        .zip(out.chunks_exact_mut(Q2_0_BLOCK_BYTES))
    {
        let vals: &[f32; 64] = block_vals.try_into().expect("64 values");
        let dst: &mut [u8; 18] = block_out.try_into().expect("18 bytes");
        quantize_block_q2_0(vals, dst);
    }
}

/// Quantize 64 F32 values into one `gguf_q2_0` block (`d (f16) | qs[16]`,
/// value `p` = `(q - 1) * d` with `q in {0, 1, 2, 3}`).
///
/// Because `{-d, 0, +d, +2d}` is asymmetric, both signs of `d` are tried with
/// one closed-form least-squares scale refinement step and exact f16 scale
/// rounding; the lower-MSE sign wins.
fn quantize_block_q2_0(values: &[f32; 64], out: &mut [u8; 18]) {
    let mut sum_sq = 0.0f32;
    for &v in values {
        sum_sq += v * v;
    }
    if sum_sq == 0.0 || !sum_sq.is_finite() {
        *out = [0u8; 18];
        return;
    }
    let rms = (sum_sq / 64.0).sqrt();
    let mut best_err = f32::INFINITY;
    let mut best_d_bits = 0u16;
    let mut best_q = [1u8; 64];

    for sign in [1.0f32, -1.0f32] {
        let d0 = sign * 0.9 * rms;
        let inv0 = 1.0 / d0;
        let mut dot = 0.0f32;
        let mut denom = 0.0f32;
        for &v in values {
            let q = (v * inv0 + 1.0).round().clamp(0.0, 3.0);
            let c = q - 1.0;
            dot += v * c;
            denom += c * c;
        }
        let d_raw = if denom > 0.0 { dot / denom } else { d0 };
        let d_bits = seismic::f16_bits(d_raw);
        let d = seismic::exact_f64(DType::F16, u32::from(d_bits)) as f32;
        if d == 0.0 {
            continue;
        }
        let inv = 1.0 / d;
        let mut err = 0.0f32;
        let mut q_arr = [1u8; 64];
        for (i, &v) in values.iter().enumerate() {
            let q = (v * inv + 1.0).round().clamp(0.0, 3.0) as u8;
            q_arr[i] = q;
            let diff = v - (f32::from(q) - 1.0) * d;
            err += diff * diff;
        }
        if err < best_err {
            best_err = err;
            best_d_bits = d_bits;
            best_q = q_arr;
        }
    }

    out[0..2].copy_from_slice(&best_d_bits.to_le_bytes());
    out[2..18].fill(0);
    for (p, &q) in best_q.iter().enumerate() {
        out[2 + p / 4] |= q << (2 * (p % 4));
    }
}

#[derive(Clone, Copy)]
enum FastDecoder {
    F32,
    F16,
    BF16,
    GgufQ8_0,
    GgufQ4K,
    GgufQ5K,
    GgufQ6K,
}

impl FastDecoder {
    fn of(source: Element) -> Option<Self> {
        match source.name() {
            "f32" => Some(Self::F32),
            "f16" => Some(Self::F16),
            "bf16" => Some(Self::BF16),
            "gguf_q8_0" => Some(Self::GgufQ8_0),
            "gguf_q4_k" => Some(Self::GgufQ4K),
            "gguf_q5_k" => Some(Self::GgufQ5K),
            "gguf_q6_k" => Some(Self::GgufQ6K),
            _ => None,
        }
    }

    fn decode_row(self, src: &[u8], out: &mut [f32]) {
        match self {
            Self::F32 => {
                for (dst, chunk) in out.iter_mut().zip(src.chunks_exact(4)) {
                    *dst = f32::from_le_bytes(chunk.try_into().expect("4 bytes"));
                }
            }
            Self::F16 => {
                for (dst, chunk) in out.iter_mut().zip(src.chunks_exact(2)) {
                    let bits = u16::from_le_bytes(chunk.try_into().expect("2 bytes"));
                    *dst = seismic::exact_f64(DType::F16, u32::from(bits)) as f32;
                }
            }
            Self::BF16 => {
                for (dst, chunk) in out.iter_mut().zip(src.chunks_exact(2)) {
                    let bits = u16::from_le_bytes(chunk.try_into().expect("2 bytes"));
                    *dst = f32::from_bits(u32::from(bits) << 16);
                }
            }
            Self::GgufQ8_0 => {
                for (dst_block, block) in out.chunks_exact_mut(32).zip(src.chunks_exact(34)) {
                    let d = f16_le(&block[0..2]);
                    for (dst, &q) in dst_block.iter_mut().zip(&block[2..34]) {
                        *dst = d * f32::from(q as i8);
                    }
                }
            }
            Self::GgufQ4K => {
                for (dst_block, block) in out.chunks_exact_mut(256).zip(src.chunks_exact(144)) {
                    let d = f16_le(&block[0..2]);
                    let dmin = f16_le(&block[2..4]);
                    let scales = &block[4..16];
                    let qs = &block[16..144];
                    for chunk in 0..4 {
                        let (sc1, m1) = get_scale_min_k4(2 * chunk, scales);
                        let (sc2, m2) = get_scale_min_k4(2 * chunk + 1, scales);
                        let (d1, min1) = (d * f32::from(sc1), dmin * f32::from(m1));
                        let (d2, min2) = (d * f32::from(sc2), dmin * f32::from(m2));
                        let q = &qs[chunk * 32..chunk * 32 + 32];
                        let base = chunk * 64;
                        for l in 0..32 {
                            dst_block[base + l] = d1 * f32::from(q[l] & 0x0f) - min1;
                            dst_block[base + 32 + l] = d2 * f32::from(q[l] >> 4) - min2;
                        }
                    }
                }
            }
            Self::GgufQ5K => {
                for (dst_block, block) in out.chunks_exact_mut(256).zip(src.chunks_exact(176)) {
                    let d = f16_le(&block[0..2]);
                    let dmin = f16_le(&block[2..4]);
                    let scales = &block[4..16];
                    let qh = &block[16..48];
                    let qs = &block[48..176];
                    for chunk in 0..4 {
                        let (sc1, m1) = get_scale_min_k4(2 * chunk, scales);
                        let (sc2, m2) = get_scale_min_k4(2 * chunk + 1, scales);
                        let (d1, min1) = (d * f32::from(sc1), dmin * f32::from(m1));
                        let (d2, min2) = (d * f32::from(sc2), dmin * f32::from(m2));
                        let q = &qs[chunk * 32..chunk * 32 + 32];
                        let u1 = 1u8 << (2 * chunk);
                        let u2 = 1u8 << (2 * chunk + 1);
                        let base = chunk * 64;
                        for l in 0..32 {
                            let h1 = if (qh[l] & u1) != 0 { 16u8 } else { 0u8 };
                            let h2 = if (qh[l] & u2) != 0 { 16u8 } else { 0u8 };
                            dst_block[base + l] = d1 * f32::from((q[l] & 0x0f) + h1) - min1;
                            dst_block[base + 32 + l] = d2 * f32::from((q[l] >> 4) + h2) - min2;
                        }
                    }
                }
            }
            Self::GgufQ6K => {
                for (dst_block, block) in out.chunks_exact_mut(256).zip(src.chunks_exact(210)) {
                    let ql = &block[0..128];
                    let qh = &block[128..192];
                    let sc = &block[192..208];
                    let d = f16_le(&block[208..210]);
                    for half in 0..2 {
                        let ql_h = &ql[half * 64..half * 64 + 64];
                        let qh_h = &qh[half * 32..half * 32 + 32];
                        let sc_h = &sc[half * 8..half * 8 + 8];
                        let base = half * 128;
                        for l in 0..32 {
                            let is = l / 16;
                            let q1 = ((ql_h[l] & 0x0f) | ((qh_h[l] & 0x03) << 4)) as i8 - 32;
                            let q2 = ((ql_h[l + 32] & 0x0f) | (((qh_h[l] >> 2) & 0x03) << 4)) as i8 - 32;
                            let q3 = ((ql_h[l] >> 4) | (((qh_h[l] >> 4) & 0x03) << 4)) as i8 - 32;
                            let q4 = ((ql_h[l + 32] >> 4) | (((qh_h[l] >> 6) & 0x03) << 4)) as i8 - 32;
                            dst_block[base + l] = d * f32::from(sc_h[is] as i8) * f32::from(q1);
                            dst_block[base + l + 32] = d * f32::from(sc_h[is + 2] as i8) * f32::from(q2);
                            dst_block[base + l + 64] = d * f32::from(sc_h[is + 4] as i8) * f32::from(q3);
                            dst_block[base + l + 96] = d * f32::from(sc_h[is + 6] as i8) * f32::from(q4);
                        }
                    }
                }
            }
        }
    }
}

#[inline]
fn f16_le(bytes: &[u8]) -> f32 {
    let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
    seismic::exact_f64(DType::F16, u32::from(bits)) as f32
}

#[inline]
fn get_scale_min_k4(j: usize, scales: &[u8]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        (
            (scales[j + 4] & 0x0f) | ((scales[j - 4] >> 6) << 4),
            (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn next_u32(state: &mut u32) -> u32 {
        *state ^= *state << 13;
        *state ^= *state >> 17;
        *state ^= *state << 5;
        *state
    }

    fn random_bytes(len: usize, seed: u32) -> Vec<u8> {
        let mut state = seed | 1;
        (0..len).map(|_| (next_u32(&mut state) >> 24) as u8).collect()
    }

    /// Every fast-path GGUF decoder matches the registry's reference
    /// `import_transforms::dequantize` on random blocks.
    #[test]
    fn fast_decoders_match_reference_dequantize() {
        let cases: &[(&str, usize, usize, &[usize])] = &[
            ("gguf_q8_0", 34, 256, &[0]),
            ("gguf_q4_k", 144, 256, &[0, 2]),
            ("gguf_q5_k", 176, 256, &[0, 2]),
            ("gguf_q6_k", 210, 256, &[208]),
        ];
        for &(name, block_bytes, cols, f16_offsets) in cases {
            let source = Element::named(name).unwrap();
            let blocks = cols / if name == "gguf_q8_0" { 32 } else { 256 };
            let mut bytes = random_bytes(blocks * block_bytes, 0x1234_5678);
            // Set finite, reasonable f16 scales at each block's scale offsets.
            let scale_bits = seismic::f16_bits(0.03125).to_le_bytes();
            for b in 0..blocks {
                for &off in f16_offsets {
                    bytes[b * block_bytes + off..b * block_bytes + off + 2]
                        .copy_from_slice(&scale_bits);
                }
            }
            let ref_bytes = crate::import_transforms::dequantize(source, &[1, cols as u64], &bytes).unwrap();
            let ref_vals: Vec<f32> = ref_bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            let mut fast_vals = vec![0.0f32; cols];
            FastDecoder::of(source).unwrap().decode_row(&bytes, &mut fast_vals);
            for (i, (&a, &b)) in fast_vals.iter().zip(&ref_vals).enumerate() {
                assert!(
                    (a - b).abs() <= 1e-5,
                    "{name} col {i}: fast={a} ref={b}"
                );
            }
        }
    }

    /// `quantize` produces valid `gguf_q2_0` blocks with high cosine similarity
    /// to the original F32 rows.
    #[test]
    fn quantize_q2_0_preserves_direction() {
        let (rows, cols) = (8usize, 256usize);
        let mut state = 0xdead_beefu32;
        let orig: Vec<f32> = (0..rows * cols)
            .map(|_| {
                let u = (next_u32(&mut state) as f32) / (u32::MAX as f32);
                (u - 0.5) * 2.0
            })
            .collect();
        let src_bytes: Vec<u8> = orig.iter().flat_map(|v| v.to_le_bytes()).collect();
        let q2_bytes = quantize(Element::f32(), &[rows as u64, cols as u64], &src_bytes).unwrap();
        let decoded_bytes =
            crate::import_transforms::dequantize(upload_element(), &[rows as u64, cols as u64], &q2_bytes)
                .unwrap();
        let decoded: Vec<f32> = decoded_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        for r in 0..rows {
            let a = &orig[r * cols..(r + 1) * cols];
            let b = &decoded[r * cols..(r + 1) * cols];
            let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
            let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
            let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
            let cos = dot / (na * nb);
            assert!(cos > 0.90, "row {r} cosine similarity too low: {cos}");
        }
    }

    /// `quantize_q4_0` produces valid `gguf_q4_0` blocks on 3D routed expert
    /// stacks with >0.995 cosine similarity to the original F32 rows.
    #[test]
    fn quantize_q4_0_preserves_direction_on_3d_routed_stack() {
        let (experts, rows, cols) = (4usize, 6usize, 128usize);
        let shape = [experts as u64, rows as u64, cols as u64];
        let mut state = 0xfeed_faceu32;
        let orig: Vec<f32> = (0..experts * rows * cols)
            .map(|_| {
                let u = (next_u32(&mut state) as f32) / (u32::MAX as f32);
                (u - 0.5) * 2.0
            })
            .collect();
        let src_bytes: Vec<u8> = orig.iter().flat_map(|v| v.to_le_bytes()).collect();
        let q4_bytes = quantize_q4_0(Element::f32(), &shape, &src_bytes).unwrap();
        let decoded_bytes =
            crate::import_transforms::dequantize(q4_0_upload_element(), &shape, &q4_bytes)
                .unwrap();
        let decoded: Vec<f32> = decoded_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        for r in 0..experts * rows {
            let a = &orig[r * cols..(r + 1) * cols];
            let b = &decoded[r * cols..(r + 1) * cols];
            let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
            let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
            let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
            let cos = dot / (na * nb);
            assert!(cos > 0.995, "row {r} q4_0 cosine similarity too low: {cos}");
        }
    }
}
