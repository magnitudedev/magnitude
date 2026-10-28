//! Progressive placement of matrices of symmetric integer codes with an f16
//! scale per 32 values (`ImportTransform::Progressive`), as the registry reads
//! the codes of their resident form ([`Placement::of`]): offset-binary codes
//! `u = c + 2^(bits-1)` in significance-ordered bit planes,
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

use magnitude_family_contracts::{ProgressiveCodeDepth, ProgressivePlane};
use seismic::{DType, Element, IntegerCodes, ScaledCodeRows};
use std::num::NonZeroU32;

/// Values per group (`ProgressivePlane::GROUP`).
const GROUP: usize = ProgressivePlane::GROUP as usize;
/// Bits of a plane word: the bit planes are `u32` words.
const WORD_BITS: u32 = u32::BITS;
// A plane's bits of one group's codes are one plane word ([`bit_of`]).
const _: () = assert!(GROUP == WORD_BITS as usize);

/// Where a bit plane (`Next`, each of `Rest`'s) holds its bit of code
/// `column`: in the word of its group, `column / GROUP`, at bit
/// `4 (i mod 8) + i / 8` for code `i` of the group.
fn bit_of(column: u32) -> (u32, u32) {
    let group = GROUP as u32;
    let within = column % group;
    (column / group, group / 8 * (within % 8) + within / 8)
}
/// The planes' scale (`ProgressivePlane::Scales`), as the source stores it.
const SCALE: DType = DType::F16;

/// The value of a stored [`SCALE`] (little-endian), as its dtype defines it.
fn scale_value(stored: &[u8]) -> f64 {
    let bytes = SCALE.bytes() as usize;
    let mut bits = [0; 4];
    bits[..bytes].copy_from_slice(&stored[..bytes]);
    seismic::exact_f64(SCALE, u32::from_le_bytes(bits))
}

/// A matrix whose stored rows hold codes of `depth`, its rows admitted for
/// placement as scaled codes (`ScaledCodeRows`), as its import places its
/// planes ([`plane`]). Admitted by `import_transforms::progressive_matrix`.
pub(crate) struct Placement {
    depth: ProgressiveCodeDepth,
    rows: ScaledCodeRows,
}

impl Placement {
    /// A matrix of `columns` values per row stored as `source`, its values
    /// those of its packet storage (`planning::value_packets`); `None` unless
    /// they are the `2^depth` codes `−2^(depth−1) .. 2^(depth−1)` of a group
    /// of [`GROUP`] values, each times its group's stored scale, the rows
    /// place as such scaled codes (`Element::scaled_code_rows`), their scale
    /// is the planes' ([`SCALE`]), and they place into every plane's rows
    /// ([`placed_bytes`], `ScaledCodeRows::places`).
    pub(crate) fn of(
        depth: ProgressiveCodeDepth,
        source: Element,
        columns: NonZeroU32,
    ) -> Option<Self> {
        let packets = crate::planning::value_packets(source)?;
        let half = 1i32 << (depth.bits() - 1);
        let codes = IntegerCodes {
            min: -half,
            max: half - 1,
            group: GROUP as u64,
        };
        let rows = (packets.integer_codes()? == codes)
            .then(|| packets.scaled_code_rows(source, columns))??;
        let places = ProgressivePlane::ALL.into_iter().all(|plane| {
            placed_bytes(plane, depth, columns.get()).is_some_and(|bytes| rows.places(bytes))
        });
        (rows.scale_dtype() == SCALE && places).then_some(Self { depth, rows })
    }
}

/// A plane's bytes of one row of `columns` values at `depth`: its shape's
/// values in its dtype; `None` beyond `u32`.
fn row_bytes(plane: ProgressivePlane, depth: ProgressiveCodeDepth, columns: u32) -> Option<u32> {
    let values = plane
        .shape(1, u64::from(columns), depth)
        .iter()
        .product::<u64>();
    u32::try_from(values * u64::from(dtype(plane).bytes())).ok()
}

/// The bytes of one row a plane's placement places ([`plane`]): the
/// plane's row, but for the radii a byte per code, then the scales, which
/// the row is reduced to as it is placed; `None` beyond `u32`.
fn placed_bytes(plane: ProgressivePlane, depth: ProgressiveCodeDepth, columns: u32) -> Option<u32> {
    match plane {
        ProgressivePlane::Radius => {
            columns.checked_add(row_bytes(ProgressivePlane::Scales, depth, columns)?)
        }
        _ => row_bytes(plane, depth, columns),
    }
}

/// The dtype a plane is held in.
fn dtype(plane: ProgressivePlane) -> DType {
    match plane {
        ProgressivePlane::Top | ProgressivePlane::Next | ProgressivePlane::Rest => DType::U32,
        ProgressivePlane::Scales => SCALE,
        ProgressivePlane::Radius => DType::F32,
    }
}

/// The dense element a plane is held in.
pub(crate) fn element(plane: ProgressivePlane) -> Element {
    Element::dense(dtype(plane))
}

/// The bytes of `plane` of the matrix `bytes` placed as `placement`, in the
/// plane's dense element (`ProgressivePlane`). The registry's conversion of
/// the matrix's source places each plane's bits of the codes and scales of
/// its resident form; the radii are computed from codes and scales so
/// placed.
pub(crate) fn plane(plane: ProgressivePlane, placement: &Placement, bytes: &[u8]) -> Vec<u8> {
    let Placement { depth, rows } = placement;
    let (depth, columns) = (*depth, rows.columns().get());
    let bits = u32::from(depth.bits());
    let coarse = u32::from(depth.coarse_bits());
    let remaining = bits - coarse;
    let rest = u32::from(depth.rest_planes());
    // The bytes of a row the plane's placement places, and of the plane's
    // row: within u32, as the placement admitted them (`Placement::of`).
    let placed_bytes = |plane: ProgressivePlane| {
        placed_bytes(plane, depth, columns).expect("an admitted placement places every plane")
    };
    let row_bytes = |plane: ProgressivePlane| {
        row_bytes(plane, depth, columns).expect("an admitted placement places every plane")
    };
    let scale_bits = 8 * dtype(ProgressivePlane::Scales).bytes();
    let place = |row_bytes,
                 code: &dyn Fn(u32, u32) -> Option<u32>,
                 scale: &dyn Fn(u32, u32) -> Option<u32>| {
        rows.placement(row_bytes, code, scale).execute(bytes)
    };
    let none = |_, _| None;
    match plane {
        ProgressivePlane::Top => {
            let per_word = WORD_BITS / coarse;
            place(
                placed_bytes(plane),
                &|column, bit| {
                    (bit >= remaining).then(|| {
                        column / per_word * WORD_BITS + coarse * (column % per_word) + bit
                            - remaining
                    })
                },
                &none,
            )
        }
        ProgressivePlane::Next => place(
            placed_bytes(plane),
            &|column, bit| {
                (bit == rest).then(|| {
                    let (group, at) = bit_of(column);
                    group * WORD_BITS + at
                })
            },
            &none,
        ),
        ProgressivePlane::Rest => place(
            placed_bytes(plane),
            &|column, bit| {
                // The rest's planes run from its highest bit down.
                (bit < rest).then(|| {
                    let (group, at) = bit_of(column);
                    (group * rest + rest - 1 - bit) * WORD_BITS + at
                })
            },
            &none,
        ),
        ProgressivePlane::Scales => place(placed_bytes(plane), &none, &|group, bit| {
            Some(scale_bits * group + bit)
        }),
        ProgressivePlane::Radius => {
            // A byte per code, then the scales, each row reduced to its
            // radii as it is placed.
            let views = Views::of(depth);
            rows.placement(
                placed_bytes(plane),
                |column, bit| Some(8 * column + bit),
                |group, bit| Some(8 * columns + scale_bits * group + bit),
            )
            .execute_mapped(bytes, row_bytes(plane) as usize, |row, radius| {
                let (codes, scales) = row.split_at(columns as usize);
                let (radius4, radius5) = views.radii(codes, scales);
                radius[..4].copy_from_slice(&radius4.to_le_bytes());
                radius[4..].copy_from_slice(&radius5.to_le_bytes());
            })
        }
    }
}

/// The stored values of one matrix row from that row of its planes (`plane`'s
/// inverse): `scale · (u − 2^(bits−1))` for each of its `columns` codes.
pub(crate) fn row_values(
    top: &[u8],
    next: &[u8],
    rest: &[u8],
    scales: &[u8],
    columns: usize,
    depth: ProgressiveCodeDepth,
) -> Vec<f32> {
    let bits = depth.bits();
    let coarse = usize::from(depth.coarse_bits());
    let remaining = usize::from(bits) - coarse;
    let rest_planes = usize::from(depth.rest_planes());
    let per_word = WORD_BITS as usize / coarse;
    let word = |plane: &[u8], index: usize| {
        u32::from_le_bytes(plane[4 * index..4 * index + 4].try_into().unwrap())
    };
    let sign = f32::from(1u16 << (bits - 1));
    (0..columns)
        .map(|column| {
            let (group, shift) = bit_of(column as u32);
            let group = group as usize;
            let high = (word(top, column / per_word) >> (coarse * (column % per_word)))
                & ((1 << coarse) - 1);
            let middle = (word(next, group) >> shift) & 1;
            let low = (0..rest_planes).fold(0, |low, plane| {
                (low << 1) | ((word(rest, group * rest_planes + plane) >> shift) & 1)
            });
            let code = (high << remaining) | (middle << rest_planes) | low;
            let scale = scale_value(&scales[group * SCALE.bytes() as usize..]) as f32;
            scale * (code as f32 - sign)
        })
        .collect()
}

/// Per offset-binary code `u` of a depth, its value `u − 2^(bits−1)` and
/// that value's differences from the coarse and the refined views' value of
/// `u`, which take the views' missing low bits at their midpoint.
struct Views {
    exact: [f64; 256],
    coarse: [f64; 256],
    refined: [f64; 256],
}

impl Views {
    fn of(depth: ProgressiveCodeDepth) -> Self {
        let (bits, coarse) = (depth.bits(), depth.coarse_bits());
        let step = f64::from(1u32 << (bits - coarse));
        let sign = f64::from(1u32 << (bits - 1));
        let mut views = Self {
            exact: [0.0; 256],
            coarse: [0.0; 256],
            refined: [0.0; 256],
        };
        for u in 0..=u8::MAX >> (8 - bits) {
            let exact = f64::from(u) - sign;
            let four = f64::from(u >> (bits - coarse)) * step + (step - 1.0) * 0.5 - sign;
            let five = f64::from(u >> (bits - coarse - 1)) * (step * 0.5)
                + (step * 0.5 - 1.0) * 0.5
                - sign;
            let u = usize::from(u);
            (views.exact[u], views.coarse[u], views.refined[u]) =
                (exact, exact - four, exact - five);
        }
        views
    }

    /// The coarse and refined views' radii of one row of offset-binary
    /// `codes` and its groups' stored `scales`: `|w − ŵ|₂ + 4 K u |w|₂`,
    /// rounded up to F32.
    fn radii(&self, codes: &[u8], scales: &[u8]) -> (f32, f32) {
        let rounding = 4.0 * codes.len() as f64 * 2f64.powi(-24);
        let (mut error4, mut error5, mut length) = (0f64, 0f64, 0f64);
        let scales = scales.chunks_exact(SCALE.bytes() as usize);
        for (codes, scale) in codes.chunks_exact(GROUP).zip(scales) {
            let scale = scale_value(scale);
            for &u in codes {
                let u = usize::from(u);
                error4 += (scale * self.coarse[u]).powi(2);
                error5 += (scale * self.refined[u]).powi(2);
                length += (scale * self.exact[u]).powi(2);
            }
        }
        let slack = rounding * length.sqrt();
        (
            round_up(error4.sqrt() + slack),
            round_up(error5.sqrt() + slack),
        )
    }
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
    use magnitude_artifacts::gguf::Encoding;
    const BLOCK_BYTES: usize = 34;

    /// `part` of the Q4_0 (`bits` 4) or Q8_0 (`bits` 8) matrix `bytes`.
    fn plane_of(part: ProgressivePlane, bytes: &[u8], columns: usize, bits: u8) -> Vec<u8> {
        let (encoding, depth) = match bits {
            4 => (Encoding::Q4_0, ProgressiveCodeDepth::Four),
            _ => (Encoding::Q8_0, ProgressiveCodeDepth::Eight),
        };
        let source = crate::source_element(encoding, columns as u64).unwrap();
        let columns = NonZeroU32::new(u32::try_from(columns).unwrap()).unwrap();
        plane(part, &Placement::of(depth, source, columns).unwrap(), bytes)
    }

    /// A matrix is admitted only where every plane's placed rows keep their
    /// bit positions within `u32`, checked without its data: the radii's
    /// placement, a byte per column and the scales (a sixteenth), fits up to
    /// 2^28 columns, not 2^29.
    #[test]
    fn placement_admits_only_rows_whose_positions_fit() {
        let of = |columns: u32| {
            let source = crate::source_element(Encoding::Q8_0, u64::from(columns)).unwrap();
            Placement::of(
                ProgressiveCodeDepth::Eight,
                source,
                NonZeroU32::new(columns).unwrap(),
            )
        };
        assert!(of(1 << 28).is_some());
        assert!(of(1 << 29).is_none());
        assert!(of((1 << 30) - 32).is_none());
    }

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
        let top = words(&plane_of(ProgressivePlane::Top, &bytes, columns, 8));
        let bit3 = words(&plane_of(ProgressivePlane::Next, &bytes, columns, 8));
        let rest = words(&plane_of(ProgressivePlane::Rest, &bytes, columns, 8));
        let scales = plane_of(ProgressivePlane::Scales, &bytes, columns, 8);
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
            let block_bytes = 2 + GROUP * usize::from(bits) / 8;
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
            let placed = |part| plane_of(part, &bytes, columns, bits);
            assert_eq!(placed(ProgressivePlane::Top), expected(&top), "top, depth {bits}");
            assert_eq!(placed(ProgressivePlane::Next), expected(&next), "next, depth {bits}");
            assert_eq!(placed(ProgressivePlane::Rest), expected(&rest), "rest, depth {bits}");
            assert_eq!(placed(ProgressivePlane::Scales), scales, "scales, depth {bits}");
        }
    }

    /// A row read back from its planes is the stored row, as the registry
    /// decodes it, at both depths.
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
            .map(|part| plane_of(part, bytes, columns, bits));
            let encoding = if bits == 4 {
                Encoding::Q4_0
            } else {
                Encoding::Q8_0
            };
            let source = crate::source_element(encoding, columns as u64).unwrap();
            let depth = if bits == 4 {
                ProgressiveCodeDepth::Four
            } else {
                ProgressiveCodeDepth::Eight
            };
            let stored =
                crate::import_transforms::dequantize(source, &[rows as u64, columns as u64], bytes)
                    .unwrap()
                    .chunks_exact(4)
                    .map(|value| f32::from_le_bytes(value.try_into().unwrap()))
                    .collect::<Vec<_>>();
            for r in 0..rows {
                let [top, next, rest, scales] =
                    std::array::from_fn(|i| &planes[i][r * planes[i].len() / rows..][..planes[i].len() / rows]);
                let values = row_values(top, next, rest, scales, columns, depth);
                for (k, value) in values.iter().enumerate() {
                    assert_eq!(
                        *value,
                        stored[r * columns + k],
                        "depth {bits} row {r} column {k}"
                    );
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
                plane_of(part, &bytes, columns, 4).len(),
                expected,
                "physical placement must match the declared depth-aware plane shape"
            );
        }
        let top = words(&plane_of(ProgressivePlane::Top, &bytes, columns, 4));
        let next = words(&plane_of(ProgressivePlane::Next, &bytes, columns, 4));
        let rest = words(&plane_of(ProgressivePlane::Rest, &bytes, columns, 4));
        let scales = plane_of(ProgressivePlane::Scales, &bytes, columns, 4);
        assert_eq!(
            4 * (top.len() + next.len() + rest.len()) + scales.len(),
            bytes.len()
        );
        let radius = plane_of(ProgressivePlane::Radius, &bytes, columns, 4)
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
                let i = k % GROUP;
                let source = (bytes[group * 18 + 2 + i % 16] >> (4 * (i / 16))) & 15;
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
        let radius = plane_of(ProgressivePlane::Radius, &bytes, columns, 8)
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
