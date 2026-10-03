//! The progressive placement of a Q8_0 matrix (`ImportTransform::Progressive`):
//! its offset-binary codes `u = c + 128` in significance-ordered bit planes,
//! and per row the radii that bound its 4- and 5-bit views
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
const BLOCK_BYTES: usize = 34;

/// The dense element a plane is held in.
pub(crate) fn element(plane: ProgressivePlane) -> Element {
    match plane {
        ProgressivePlane::Top | ProgressivePlane::Bit3 | ProgressivePlane::Rest => Element::u32(),
        ProgressivePlane::Scales => Element::f16(),
        ProgressivePlane::Radius => Element::f32(),
    }
}

/// The bytes of `plane` of the Q8_0 matrix `bytes` (`rows` rows of
/// `columns` values), in the plane's dense element (`ProgressivePlane`).
pub(crate) fn plane(plane: ProgressivePlane, bytes: &[u8], rows: usize, columns: usize) -> Vec<u8> {
    let groups = columns / GROUP;
    let row_bytes = groups * BLOCK_BYTES;
    match plane {
        ProgressivePlane::Radius => {
            let mut radius = vec![0f32; 2 * rows];
            parallel_rows(rows, |first, count| {
                (first..first + count)
                    .map(|row| radii(&bytes[row * row_bytes..(row + 1) * row_bytes], columns))
                    .collect::<Vec<_>>()
            })
            .into_iter()
            .enumerate()
            .for_each(|(row, (radius4, radius5))| {
                radius[2 * row] = radius4;
                radius[2 * row + 1] = radius5;
            });
            radius
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect()
        }
        ProgressivePlane::Scales => bytes
            .chunks_exact(BLOCK_BYTES)
            .flat_map(|block| [block[0], block[1]])
            .collect(),
        ProgressivePlane::Top => bytes
            .chunks_exact(BLOCK_BYTES)
            .flat_map(|block| {
                let codes = codes(block);
                (0..4).flat_map(move |word| {
                    (0..8)
                        .fold(0u32, |packed, e| {
                            packed | (u32::from(codes[8 * word + e] >> 4) << (4 * e))
                        })
                        .to_le_bytes()
                })
            })
            .collect(),
        ProgressivePlane::Bit3 => bytes
            .chunks_exact(BLOCK_BYTES)
            .flat_map(|block| bit_plane(&codes(block), 3).to_le_bytes())
            .collect(),
        ProgressivePlane::Rest => bytes
            .chunks_exact(BLOCK_BYTES)
            .flat_map(|block| {
                let codes = codes(block);
                [2, 1, 0]
                    .into_iter()
                    .flat_map(move |bit| bit_plane(&codes, bit).to_le_bytes())
            })
            .collect(),
    }
}

/// A block's 32 offset-binary codes.
fn codes(block: &[u8]) -> [u8; GROUP] {
    std::array::from_fn(|i| block[2 + i] ^ 0x80)
}

/// Bit `bit` of a group's codes, code `8q + e` in bit `4e + q`.
fn bit_plane(codes: &[u8; GROUP], bit: u32) -> u32 {
    (0..GROUP).fold(0u32, |word, i| {
        word | (u32::from((codes[i] >> bit) & 1) << (4 * (i % 8) + i / 8))
    })
}

/// The 4- and 5-bit views' radii of one row: `|w − ŵ|₂ + 4 K u |w|₂`,
/// rounded up to F32.
fn radii(row: &[u8], columns: usize) -> (f32, f32) {
    let rounding = 4.0 * columns as f64 * 2f64.powi(-24);
    let (mut error4, mut error5, mut length) = (0f64, 0f64, 0f64);
    for block in row.chunks_exact(BLOCK_BYTES) {
        let scale = f64::from(seismic::f16_to_f32(u16::from_le_bytes([
            block[0], block[1],
        ])));
        for u in codes(block) {
            let exact = f64::from(u) - 128.0;
            let four = f64::from(u >> 4) * 16.0 + 7.5 - 128.0;
            let five = f64::from(u >> 3) * 8.0 + 3.5 - 128.0;
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

/// `work(first, count)` over consecutive row ranges across the host's cores,
/// its results in row order.
fn parallel_rows<T: Send>(rows: usize, work: impl Fn(usize, usize) -> Vec<T> + Sync) -> Vec<T> {
    let workers = std::thread::available_parallelism()
        .map_or(1, usize::from)
        .min(rows.max(1));
    let per_worker = rows.div_ceil(workers);
    std::thread::scope(|scope| {
        let work = &work;
        (0..rows)
            .step_by(per_worker.max(1))
            .map(|first| scope.spawn(move || work(first, per_worker.min(rows - first))))
            .collect::<Vec<_>>()
            .into_iter()
            .flat_map(|task| task.join().expect("a radius thread panicked"))
            .collect()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let top = words(&plane(ProgressivePlane::Top, &bytes, rows, columns));
        let bit3 = words(&plane(ProgressivePlane::Bit3, &bytes, rows, columns));
        let rest = words(&plane(ProgressivePlane::Rest, &bytes, rows, columns));
        let scales = plane(ProgressivePlane::Scales, &bytes, rows, columns);
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

    /// Every row's dot product with any activation stays within its radius of
    /// each view's, in F32.
    #[test]
    fn the_radii_bound_every_dot_product_with_the_views() {
        let (rows, columns) = (4usize, 256usize);
        let bytes = (0..rows as u32)
            .flat_map(|r| row(r + 7, columns / GROUP))
            .collect::<Vec<_>>();
        let radius = plane(ProgressivePlane::Radius, &bytes, rows, columns)
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
