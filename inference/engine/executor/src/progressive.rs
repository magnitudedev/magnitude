//! Progressive placement of symmetric 32-value Q4_0 and Q8_0 matrices (`ImportTransform::Progressive`):
//! offset-binary codes `u = c + 2^(bits-1)` in significance-ordered bit planes,
//! and per-row radii for the half-depth coarse view and its next-bit refinement
//! (`readout.seismic`, the progressive head readout).
//!
//! A view of the top `b` bits takes each code's missing low bits at their
//! midpoint. For stored row `w`, view row `ŵ` and any activation row `x`,
//! `|w·x − ŵ·x| ≤ |w − ŵ|₂ |x|₂` (Cauchy–Schwarz), and an F32 dot product of
//! `K` terms differs from the exact one by at most `K u |w|₂ |x|₂` with
//! `u = 2⁻²⁴`. A radius covers both products' rounding, so the F32 logits a
//! device computes from the row and from its view stay within `radius |x|₂`
//! of each other.

use magnitude_family_contracts::ProgressivePlane;
use seismic::Element;

/// Values per Q8_0 block, and its bytes: an f16 scale, then 32 int8 codes.
const GROUP: usize = 32;
fn block_bytes(bits: u8) -> usize {
    2 + GROUP * usize::from(bits) / 8
}

fn coarse_bits(bits: u8) -> u8 {
    assert!(
        matches!(bits, 4 | 8),
        "progressive code depth must be 4 or 8"
    );
    bits / 2
}

/// The dense element a plane is held in.
pub(crate) fn element(plane: ProgressivePlane) -> Element {
    match plane {
        ProgressivePlane::Top | ProgressivePlane::Next | ProgressivePlane::Rest => Element::u32(),
        ProgressivePlane::Scales => Element::f16(),
        ProgressivePlane::Radius => Element::f32(),
    }
}

/// The bytes of `plane` of the Q8_0 matrix `bytes` (`rows` rows of
/// `columns` values), in the plane's dense element (`ProgressivePlane`).
pub(crate) fn plane(
    plane: ProgressivePlane,
    bytes: &[u8],
    rows: usize,
    columns: usize,
    bits: u8,
) -> Vec<u8> {
    let groups = columns / GROUP;
    let block_bytes = block_bytes(bits);
    let coarse = coarse_bits(bits);
    let remaining = bits - coarse;
    let row_bytes = groups * block_bytes;
    let blocks = bytes.len() / block_bytes;
    match plane {
        ProgressivePlane::Radius => placed(bytes, rows, row_bytes, 8, |row, radius| {
            let (radius4, radius5) = radii(row, columns, bits);
            radius[..4].copy_from_slice(&radius4.to_le_bytes());
            radius[4..].copy_from_slice(&radius5.to_le_bytes());
        }),
        ProgressivePlane::Scales => placed(bytes, blocks, block_bytes, 2, |block, scale| {
            scale.copy_from_slice(&block[..2])
        }),
        ProgressivePlane::Top => {
            let per_word = 32 / usize::from(coarse);
            let width = 4 * GROUP / per_word;
            placed(bytes, blocks, block_bytes, width, |block, words| {
                let codes = codes(block, bits);
                for (word, codes) in words.chunks_exact_mut(4).zip(codes.chunks_exact(per_word)) {
                    let packed = codes.iter().enumerate().fold(0u32, |packed, (e, code)| {
                        packed | (u32::from(code >> remaining) << (usize::from(coarse) * e))
                    });
                    word.copy_from_slice(&packed.to_le_bytes());
                }
            })
        }
        ProgressivePlane::Next => placed(bytes, blocks, block_bytes, 4, |block, word| {
            let next = bit_plane(&codes(block, bits), u32::from(remaining - 1));
            word.copy_from_slice(&next.to_le_bytes());
        }),
        ProgressivePlane::Rest => {
            let width = 4 * usize::from(remaining - 1);
            placed(bytes, blocks, block_bytes, width, |block, words| {
                let codes = codes(block, bits);
                for (word, bit) in words.chunks_exact_mut(4).zip((0..remaining - 1).rev()) {
                    word.copy_from_slice(&bit_plane(&codes, u32::from(bit)).to_le_bytes());
                }
            })
        }
    }
}

/// A plane's bytes: `width` of them for each of the first `pieces`
/// consecutive `piece`-byte pieces of `bytes` (a block, or a row for the
/// radii), written in place by `place(piece, its bytes)`, the pieces divided
/// across the host's cores. A head is hundreds of millions of codes and its
/// planes are placed while the model loads, so a plane is written once, whole
/// words at a time, into its final storage.
fn placed(
    bytes: &[u8],
    pieces: usize,
    piece: usize,
    width: usize,
    place: impl Fn(&[u8], &mut [u8]) + Sync,
) -> Vec<u8> {
    let mut plane = vec![0u8; pieces * width];
    let workers = std::thread::available_parallelism()
        .map_or(1, usize::from)
        .min(pieces.max(1));
    let per_worker = pieces.div_ceil(workers).max(1);
    std::thread::scope(|scope| {
        let place = &place;
        for (worker, part) in plane.chunks_mut(per_worker * width).enumerate() {
            scope.spawn(move || {
                let first = worker * per_worker;
                for (index, target) in part.chunks_exact_mut(width).enumerate() {
                    let start = (first + index) * piece;
                    place(&bytes[start..start + piece], target);
                }
            });
        }
    });
    plane
}

/// The stored values of one matrix row from that row of its planes (`plane`'s
/// inverse): `scale · (u − 2^(bits−1))` for each of its `columns` codes.
pub(crate) fn row_values(
    top: &[u8],
    next: &[u8],
    rest: &[u8],
    scales: &[u8],
    columns: usize,
    bits: u8,
) -> Vec<f32> {
    let coarse = usize::from(coarse_bits(bits));
    let remaining = usize::from(bits) - coarse;
    let per_word = 32 / coarse;
    let word = |plane: &[u8], index: usize| {
        u32::from_le_bytes(plane[4 * index..4 * index + 4].try_into().unwrap())
    };
    let sign = f32::from(1u16 << (bits - 1));
    (0..columns)
        .map(|column| {
            let (group, within) = (column / GROUP, column % GROUP);
            let shift = 4 * (within % 8) + within / 8;
            let high = (word(top, column / per_word) >> (coarse * (column % per_word)))
                & ((1 << coarse) - 1);
            let middle = (word(next, group) >> shift) & 1;
            let low = (0..remaining - 1).fold(0, |low, plane| {
                (low << 1) | ((word(rest, group * (remaining - 1) + plane) >> shift) & 1)
            });
            let code = (high << remaining) | (middle << (remaining - 1)) | low;
            let scale = seismic::f16_to_f32(u16::from_le_bytes([
                scales[2 * group],
                scales[2 * group + 1],
            ]));
            scale * (code as f32 - sign)
        })
        .collect()
}

/// A block's 32 offset-binary codes.
fn codes(block: &[u8], bits: u8) -> [u8; GROUP] {
    let mut codes = [0u8; GROUP];
    let stored = &block[2..];
    if bits == 8 {
        for (code, stored) in codes.iter_mut().zip(stored) {
            *code = stored ^ 0x80;
        }
    } else {
        // Code `i` is the low nibble of byte `i`, code `16 + i` its high one.
        let (low, high) = codes.split_at_mut(GROUP / 2);
        for ((low, high), stored) in low.iter_mut().zip(high).zip(stored) {
            *low = stored & 15;
            *high = stored >> 4;
        }
    }
    codes
}

/// Bit `bit` of a group's codes, code `8q + e` in bit `4e + q`.
fn bit_plane(codes: &[u8; GROUP], bit: u32) -> u32 {
    (0..GROUP).fold(0u32, |word, i| {
        word | (u32::from((codes[i] >> bit) & 1) << (4 * (i % 8) + i / 8))
    })
}

/// The 4- and 5-bit views' radii of one row: `|w − ŵ|₂ + 4 K u |w|₂`,
/// rounded up to F32.
fn radii(row: &[u8], columns: usize, bits: u8) -> (f32, f32) {
    let block_bytes = block_bytes(bits);
    let coarse = coarse_bits(bits);
    let step = f64::from(1u32 << (bits - coarse));
    let sign = f64::from(1u32 << (bits - 1));
    let rounding = 4.0 * columns as f64 * 2f64.powi(-24);
    let (mut error4, mut error5, mut length) = (0f64, 0f64, 0f64);
    for block in row.chunks_exact(block_bytes) {
        let scale = f64::from(seismic::f16_to_f32(u16::from_le_bytes([
            block[0], block[1],
        ])));
        for u in codes(block, bits) {
            let exact = f64::from(u) - sign;
            let four = f64::from(u >> (bits - coarse)) * step + (step - 1.0) * 0.5 - sign;
            let five = f64::from(u >> (bits - coarse - 1)) * (step * 0.5)
                + (step * 0.5 - 1.0) * 0.5
                - sign;
            error4 += (scale * (exact - four)).powi(2);
            error5 += (scale * (exact - five)).powi(2);
            length += (scale * exact).powi(2);
        }
    }
    let slack = rounding * length.sqrt();
    (
        round_up(error4.sqrt() + slack),
        round_up(error5.sqrt() + slack),
    )
}

/// The least F32 at or above `value`.
fn round_up(value: f64) -> f32 {
    let nearest = value as f32;
    if f64::from(nearest) >= value {
        nearest
    } else {
        f32::from_bits(nearest.to_bits() + 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const BLOCK_BYTES: usize = 34;

    /// A Q8_0 row of `groups` blocks with codes from `seed`.
    fn row(seed: u32, groups: usize) -> Vec<u8> {
        let mut state = seed | 1;
        (0..groups)
            .flat_map(|group| {
                let scale = seismic::f16_bits(0.01 + 0.001 * group as f32).to_le_bytes();
                let codes = (0..GROUP).map(|_| {
                    state ^= state << 13;
                    state ^= state >> 17;
                    state ^= state << 5;
                    (state >> 24) as u8
                });
                scale.into_iter().chain(codes).collect::<Vec<_>>()
            })
            .collect()
    }

    fn words(bytes: &[u8]) -> Vec<u32> {
        bytes
            .chunks_exact(4)
            .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
            .collect()
    }

    /// The planes hold every code bit for bit, as the kernels read them.
    #[test]
    fn the_planes_hold_every_code() {
        let (rows, columns) = (3usize, 64usize);
        let bytes = [row(1, 2), row(2, 2), row(3, 2)].concat();
        let top = words(&plane(ProgressivePlane::Top, &bytes, rows, columns, 8));
        let bit3 = words(&plane(ProgressivePlane::Next, &bytes, rows, columns, 8));
        let rest = words(&plane(ProgressivePlane::Rest, &bytes, rows, columns, 8));
        let scales = plane(ProgressivePlane::Scales, &bytes, rows, columns, 8);
        let groups = columns / GROUP;
        for r in 0..rows {
            for k in 0..columns {
                let (g, within) = (k / GROUP, k % GROUP);
                let block = &bytes[(r * groups + g) * BLOCK_BYTES..];
                let u = u32::from(block[2 + within] ^ 0x80);
                let shift = 4 * (within % 8) + within / 8;
                let high = (top[r * columns / 8 + k / 8] >> (4 * (k % 8))) & 15;
                let third = (bit3[r * groups + g] >> shift) & 1;
                let low = (0..3).fold(0, |low, j| {
                    (low << 1) | ((rest[(r * groups + g) * 3 + j] >> shift) & 1)
                });
                assert_eq!((high << 4) | (third << 3) | low, u, "row {r} column {k}");
                assert_eq!(&scales[(r * groups + g) * 2..][..2], &block[..2]);
            }
        }
    }

    /// Each depth's placement is the placement written one code at a time
    /// from the layout's definition: code `i` of a group has its top half in
    /// `top` (code `i` of the group at bits `c·(i mod (32/c))` of word
    /// `i / (32/c)`), and bit `b` of its low half in the plane word of that
    /// bit at position `4·(i mod 8) + i / 8`, the next bit in `next` and the
    /// rest from the highest down in `rest`.
    #[test]
    fn each_depth_places_every_plane_as_the_layout_defines() {
        // More blocks than any host has cores, and not a multiple of their
        // count: the cores' shares of a plane differ in length.
        let (rows, columns) = (211usize, 96usize);
        for bits in [4u8, 8u8] {
            let block_bytes = block_bytes(bits);
            let mut state = 0x1234_5678u32 | u32::from(bits);
            let bytes = (0..rows * (columns / GROUP) * block_bytes)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 17;
                    state ^= state << 5;
                    (state >> 24) as u8
                })
                .collect::<Vec<_>>();
            let coarse = usize::from(bits / 2);
            let low = usize::from(bits) - coarse;
            let per_word = 32 / coarse;
            let blocks = rows * columns / GROUP;
            let mut top = vec![0u32; blocks * GROUP / per_word];
            let mut next = vec![0u32; blocks];
            let mut rest = vec![0u32; blocks * (low - 1)];
            let mut scales = Vec::new();
            for block in 0..blocks {
                let stored = &bytes[block * block_bytes..(block + 1) * block_bytes];
                scales.extend([stored[0], stored[1]]);
                for i in 0..GROUP {
                    let code = if bits == 8 {
                        u32::from(stored[2 + i] ^ 0x80)
                    } else {
                        u32::from((stored[2 + i % 16] >> (4 * (i / 16))) & 15)
                    };
                    let position = 4 * (i % 8) + i / 8;
                    top[block * GROUP / per_word + i / per_word] |=
                        (code >> low) << (coarse * (i % per_word));
                    next[block] |= ((code >> (low - 1)) & 1) << position;
                    for plane in 0..low - 1 {
                        rest[block * (low - 1) + plane] |=
                            ((code >> (low - 2 - plane)) & 1) << position;
                    }
                }
            }
            let expected = |words: &[u32]| {
                words
                    .iter()
                    .flat_map(|word| word.to_le_bytes())
                    .collect::<Vec<_>>()
            };
            let placed = |part| plane(part, &bytes, rows, columns, bits);
            assert_eq!(placed(ProgressivePlane::Top), expected(&top), "top, depth {bits}");
            assert_eq!(placed(ProgressivePlane::Next), expected(&next), "next, depth {bits}");
            assert_eq!(placed(ProgressivePlane::Rest), expected(&rest), "rest, depth {bits}");
            assert_eq!(placed(ProgressivePlane::Scales), scales, "scales, depth {bits}");
        }
    }

    /// A row read back from its planes is the stored row, at both depths.
    #[test]
    fn a_row_of_the_planes_decodes_to_its_stored_values() {
        let columns = 64usize;
        let eight = [row(5, 2), row(6, 2)].concat();
        let four = (0..4usize)
            .flat_map(|group| {
                let mut block = seismic::f16_bits(0.01 * (group + 1) as f32)
                    .to_le_bytes()
                    .to_vec();
                block.extend((0..16).map(|i| ((3 * i + group) % 16) as u8 | (((i + 7) % 16) << 4) as u8));
                block
            })
            .collect::<Vec<_>>();
        for (bytes, bits) in [(&eight, 8u8), (&four, 4u8)] {
            let rows = 2usize;
            let planes = [
                ProgressivePlane::Top,
                ProgressivePlane::Next,
                ProgressivePlane::Rest,
                ProgressivePlane::Scales,
            ]
            .map(|part| plane(part, bytes, rows, columns, bits));
            let block_bytes = block_bytes(bits);
            for r in 0..rows {
                let [top, next, rest, scales] =
                    std::array::from_fn(|i| &planes[i][r * planes[i].len() / rows..][..planes[i].len() / rows]);
                let values = row_values(top, next, rest, scales, columns, bits);
                for (k, value) in values.iter().enumerate() {
                    let block = &bytes[(r * columns / GROUP + k / GROUP) * block_bytes..];
                    let scale = seismic::f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
                    let code = f32::from(codes(block, bits)[k % GROUP]);
                    let expected = scale * (code - f32::from(1u16 << (bits - 1)));
                    assert_eq!(*value, expected, "depth {bits} row {r} column {k}");
                }
            }
        }
    }

    /// Q4 uses two coarse bits, one refinement and one residual bit. Every
    /// source code survives, and the planes together charge the original
    /// packed weight bytes plus the two radii per vocabulary row.
    #[test]
    fn four_bit_planes_preserve_every_code_and_bound_views() {
        let (rows, columns) = (2usize, 64usize);
        let bytes = (0..rows * columns / GROUP)
            .flat_map(|group| {
                let mut block = seismic::f16_bits(0.01 * (group + 1) as f32)
                    .to_le_bytes()
                    .to_vec();
                block
                    .extend((0..16).map(|i| {
                        ((i + group) % 16) as u8 | ((((15 - i + group) % 16) as u8) << 4)
                    }));
                block
            })
            .collect::<Vec<_>>();
        for part in ProgressivePlane::ALL {
            let shape = part.shape(
                rows as u64,
                columns as u64,
                magnitude_family_contracts::ProgressiveCodeDepth::Four,
            );
            let expected = shape.iter().product::<u64>() as usize
                * if part == ProgressivePlane::Scales {
                    2
                } else {
                    4
                };
            assert_eq!(
                plane(part, &bytes, rows, columns, 4).len(),
                expected,
                "physical placement must match the declared depth-aware plane shape"
            );
        }
        let top = words(&plane(ProgressivePlane::Top, &bytes, rows, columns, 4));
        let next = words(&plane(ProgressivePlane::Next, &bytes, rows, columns, 4));
        let rest = words(&plane(ProgressivePlane::Rest, &bytes, rows, columns, 4));
        let scales = plane(ProgressivePlane::Scales, &bytes, rows, columns, 4);
        assert_eq!(
            4 * (top.len() + next.len() + rest.len()) + scales.len(),
            bytes.len()
        );
        let radius = plane(ProgressivePlane::Radius, &bytes, rows, columns, 4)
            .chunks_exact(4)
            .map(|w| f32::from_le_bytes(w.try_into().unwrap()))
            .collect::<Vec<_>>();
        for row in 0..rows {
            let mut exact = 0f64;
            let mut coarse = 0f64;
            let mut refined = 0f64;
            let mut length = 0f64;
            for k in 0..columns {
                let group = row * columns / GROUP + k / GROUP;
                let shift = 4 * (k % 8) + (k % 32) / 8;
                let high = (top[row * columns / 16 + k / 16] >> (2 * (k % 16))) & 3;
                let middle = (next[group] >> shift) & 1;
                let low = (rest[group] >> shift) & 1;
                let code = (high << 2) | (middle << 1) | low;
                let source = codes(&bytes[group * 18..], 4)[k % GROUP];
                assert_eq!(code, u32::from(source));
                let scale = f64::from(seismic::f16_to_f32(u16::from_le_bytes([
                    bytes[group * 18],
                    bytes[group * 18 + 1],
                ])));
                let x = ((k * 7 % 19) as f64 - 9.0) * 0.125;
                exact += scale * (f64::from(source) - 8.0) * x;
                coarse += scale * (f64::from(high) * 4.0 - 6.5) * x;
                refined += scale * (f64::from(high * 2 + middle) * 2.0 - 7.5) * x;
                length += x * x;
            }
            assert!((exact - coarse).abs() <= f64::from(radius[2 * row]) * length.sqrt());
            assert!((exact - refined).abs() <= f64::from(radius[2 * row + 1]) * length.sqrt());
        }
    }

    /// Every row's dot product with any activation stays within its radius of
    /// each view's, in F32.
    #[test]
    fn the_radii_bound_every_dot_product_with_the_views() {
        let (rows, columns) = (4usize, 256usize);
        let bytes = (0..rows as u32)
            .flat_map(|r| row(r + 7, columns / GROUP))
            .collect::<Vec<_>>();
        let radius = plane(ProgressivePlane::Radius, &bytes, rows, columns, 8)
            .chunks_exact(4)
            .map(|word| f32::from_le_bytes(word.try_into().unwrap()))
            .collect::<Vec<_>>();
        let mut state = 99u32;
        for _ in 0..8 {
            let x = (0..columns)
                .map(|_| {
                    state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                    (state >> 8) as f32 / (1u32 << 24) as f32 - 0.5
                })
                .collect::<Vec<_>>();
            let length = x.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>().sqrt();
            for r in 0..rows {
                let (mut exact, mut four, mut five) = (0f32, 0f32, 0f32);
                for k in 0..columns {
                    let block = &bytes[(r * columns / GROUP + k / GROUP) * BLOCK_BYTES..];
                    let scale = seismic::f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
                    let u = block[2 + k % GROUP] ^ 0x80;
                    exact = (scale * (f32::from(u) - 128.0)).mul_add(x[k], exact);
                    four = (scale * (f32::from(u >> 4) * 16.0 - 120.5)).mul_add(x[k], four);
                    five = (scale * (f32::from(u >> 3) * 8.0 - 124.5)).mul_add(x[k], five);
                }
                for (level, view) in [(0, four), (1, five)] {
                    let reach = f64::from(radius[2 * r + level]) * length;
                    assert!(
                        f64::from((exact - view).abs()) <= reach,
                        "row {r} level {level}: |{exact} - {view}| exceeds {reach}"
                    );
                }
            }
        }
    }
}
