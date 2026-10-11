use super::*;
use crate::interp::TensorData;
use crate::reference_math::conversion::{canonical_nan, exact_f64, quantize};
use crate::registry::{
    CodeInterpretation, DecodeComponent, DecodeRecipe, DecodeStep, FloatCodeFormat, Layout,
    QuantizedComponent, RepresentationInfo,
};

const FLOATS: [DType; 3] = [DType::F32, DType::F16, DType::BF16];

/// Deterministic SplitMix64 test data.
struct Samples(u64);
impl Samples {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn bytes(&mut self, count: usize) -> Vec<u8> {
        (0..count).map(|_| self.next() as u8).collect()
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn read_bits(bytes: &[u8], first: usize, width: u32) -> u32 {
    (0..width as usize).fold(0, |value, bit| {
        value | (u32::from((bytes[(first + bit) / 8] >> ((first + bit) % 8)) & 1) << bit)
    })
}

fn write_bits(bytes: &mut [u8], first: usize, width: u32, value: u32) {
    for bit in 0..width as usize {
        let (byte, shift) = ((first + bit) / 8, (first + bit) % 8);
        bytes[byte] = (bytes[byte] & !(1 << shift)) | (((value >> bit) as u8 & 1) << shift);
    }
}

fn dither(seed: u64, up: f64, down: f64) -> Dither {
    Dither::new(seed, up, down).unwrap()
}

/// Every value draws a move in one direction.
fn all(up: bool) -> Dither {
    if up {
        dither(7, 1.0, 0.0)
    } else {
        dither(7, 0.0, 1.0)
    }
}

fn one(tallies: Vec<Tally>) -> Tally {
    assert_eq!(tallies.len(), 1);
    tallies[0]
}

/// The independent definition of adjacency: the next representable value by
/// exact value comparison.
enum Oracle {
    /// Every finite value of the lattice, ascending and distinct.
    Table(Vec<f64>),
    /// The finite `f32` values.
    F32,
}

impl Oracle {
    fn table(values: impl Iterator<Item = f64>) -> Oracle {
        let mut values: Vec<f64> = values.filter(|value| value.is_finite()).collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        values.dedup();
        Oracle::Table(values)
    }

    fn dense(dtype: DType) -> Oracle {
        match dtype {
            DType::F32 => Oracle::F32,
            _ => Oracle::table((0..=u32::from(u16::MAX)).map(|bits| exact_f64(dtype, bits))),
        }
    }

    fn plane(encoding: &PlaneEncoding) -> Oracle {
        match encoding {
            PlaneEncoding::Dense(dtype) => Oracle::dense(*dtype),
            PlaneEncoding::Packed { bits, .. } => {
                Oracle::table((0..1u32 << bits).map(|raw| plane_value(encoding, raw)))
            }
            PlaneEncoding::FloatCode { format } => {
                Oracle::table((0..1u32 << format.bits()).map(|raw| plane_value(encoding, raw)))
            }
        }
    }

    /// The adjacent finite lattice value above or below finite `value`.
    fn neighbour(&self, value: f64, up: bool) -> Option<f64> {
        match self {
            Oracle::Table(values) => {
                let index = values
                    .binary_search_by(|level| level.partial_cmp(&value).unwrap())
                    .unwrap_or_else(|_| panic!("{value} is not a level of its lattice"));
                if up {
                    values.get(index + 1).copied()
                } else {
                    index.checked_sub(1).map(|index| values[index])
                }
            }
            Oracle::F32 => {
                let value = value as f32;
                let next = if up {
                    value.next_up()
                } else {
                    value.next_down()
                };
                next.is_finite().then_some(f64::from(next))
            }
        }
    }
}

/// The exact value of raw entry `raw` of a plane, written here from the
/// encoding's definition and independent of the reference recipes the
/// lattice is built from.
fn plane_value(encoding: &PlaneEncoding, raw: u32) -> f64 {
    match encoding {
        PlaneEncoding::Dense(dtype) => exact_f64(*dtype, raw),
        PlaneEncoding::Packed {
            bits,
            interpretation,
        } => match interpretation {
            CodeInterpretation::Unsigned => f64::from(raw),
            CodeInterpretation::TwosComplement => {
                f64::from(raw)
                    - if raw >> (bits - 1) == 1 {
                        2f64.powi(*bits as i32)
                    } else {
                        0.0
                    }
            }
            CodeInterpretation::Offset(offset) => f64::from(raw) - f64::from(*offset),
            CodeInterpretation::Table(table) => f64::from(table[raw as usize]),
        },
        PlaneEncoding::FloatCode { format } => f64::from(format.decode(raw)),
    }
}

fn step_of(steps: &Steps, raw: u32, up: bool) -> Step {
    steps.step(raw, if up { Direction::Up } else { Direction::Down })
}

// ---------------------------------------------------------------- floats

#[test]
fn narrow_floats_step_to_the_adjacent_value_exhaustively() {
    for dtype in [DType::F16, DType::BF16] {
        let lattice = Lattice::dense(dtype).unwrap();
        assert_eq!(lattice.chunk_bytes(), 2);
        let oracle = Oracle::dense(dtype);
        let before: Vec<u8> = (0..=u16::MAX).flat_map(u16::to_le_bytes).collect();
        for up in [true, false] {
            let mut after = before.clone();
            let tally = one(lattice.perturb(&all(up), 0, &mut after));
            let mut expected = Tally {
                values: 1 << 16,
                ..Tally::default()
            };
            for payload in 0..=u32::from(u16::MAX) {
                let stepped = read_bits(&after, payload as usize * 16, 16);
                let value = exact_f64(dtype, payload);
                if !value.is_finite() {
                    assert_eq!(stepped, payload, "{dtype} {payload:#x} is not finite");
                    expected.off_lattice += 1;
                    continue;
                }
                match oracle.neighbour(value, up) {
                    Some(neighbour) => {
                        assert_eq!(
                            exact_f64(dtype, stepped),
                            neighbour,
                            "{dtype} {payload:#x} up={up}"
                        );
                        if up {
                            expected.up += 1;
                        } else {
                            expected.down += 1;
                        }
                    }
                    None => {
                        assert_eq!(stepped, payload, "{dtype} {payload:#x} has no neighbour");
                        expected.blocked += 1;
                    }
                }
            }
            assert_eq!(tally, expected, "{dtype} up={up}");
            assert_eq!(tally.blocked, 1, "only the largest magnitude is blocked");
        }
    }
}

#[test]
fn f32_steps_to_the_adjacent_value_across_every_binade() {
    let mut samples = Samples(11);
    let mut payloads = Vec::new();
    for sign in [0u32, 1 << 31] {
        for exponent in 0..=255u32 {
            let random = [(); 4].map(|_| samples.next() as u32 & 0x7f_ffff);
            for fraction in [0, 1, 2, 0x3f_ffff, 0x40_0000, 0x7f_fffe, 0x7f_ffff]
                .into_iter()
                .chain(random)
            {
                payloads.push(sign | (exponent << 23) | fraction);
            }
        }
    }
    let lattice = Lattice::dense(DType::F32).unwrap();
    assert_eq!(lattice.chunk_bytes(), 4);
    let before: Vec<u8> = payloads
        .iter()
        .flat_map(|bits| bits.to_le_bytes())
        .collect();
    for up in [true, false] {
        let mut after = before.clone();
        let tally = one(lattice.perturb(&all(up), 0, &mut after));
        let mut expected = Tally {
            values: payloads.len() as u64,
            ..Tally::default()
        };
        for (index, payload) in payloads.iter().copied().enumerate() {
            let stepped = read_bits(&after, index * 32, 32);
            let value = exact_f64(DType::F32, payload);
            if !value.is_finite() {
                assert_eq!(stepped, payload);
                expected.off_lattice += 1;
                continue;
            }
            if stepped == payload {
                // No move: only the largest magnitude, moving away from zero.
                assert_eq!(
                    value,
                    if up {
                        f64::from(f32::MAX)
                    } else {
                        f64::from(f32::MIN)
                    }
                );
                expected.blocked += 1;
                continue;
            }
            let moved = exact_f64(DType::F32, stepped);
            assert!(
                moved.is_finite(),
                "{payload:#x} stepped to a non-finite value"
            );
            assert!(
                if up { moved > value } else { moved < value },
                "{payload:#x}"
            );
            // Adjacent by exact value: the midpoint is nearer to any value
            // strictly between the two than to either, so it rounds to one
            // of them only when no representable value lies between.
            let nearest = exact_f64(DType::F32, quantize(DType::F32, (value + moved) / 2.0));
            assert!(nearest == value || nearest == moved, "{payload:#x} up={up}");
            assert_eq!(Oracle::F32.neighbour(value, up), Some(moved));
            if up {
                expected.up += 1;
            } else {
                expected.down += 1;
            }
        }
        assert_eq!(tally, expected, "up={up}");
        assert_eq!(tally.blocked, 1);
    }
}

#[test]
fn floating_edges_follow_the_stated_rules() {
    for dtype in FLOATS {
        let lattice = Lattice::dense(dtype).unwrap();
        let width = dtype.bytes() * 8;
        let sign = 1u32 << (width - 1);
        let step = |payload: u32, up: bool| {
            let mut bytes = payload.to_le_bytes()[..dtype.bytes() as usize].to_vec();
            let tally = one(lattice.perturb(&all(up), 0, &mut bytes));
            (read_bits(&bytes, 0, width), tally)
        };
        let moved = |up: bool| Tally {
            values: 1,
            up: u64::from(up),
            down: u64::from(!up),
            ..Tally::default()
        };
        let blocked = Tally {
            values: 1,
            blocked: 1,
            ..Tally::default()
        };
        let off_lattice = Tally {
            values: 1,
            off_lattice: 1,
            ..Tally::default()
        };

        // Either zero: the smallest subnormal of each sign.
        for zero in [0, sign] {
            assert_eq!(step(zero, true), (1, moved(true)), "{dtype}");
            assert_eq!(step(zero, false), (sign | 1, moved(false)), "{dtype}");
        }
        assert!(exact_f64(dtype, 1) > 0.0 && exact_f64(dtype, sign | 1) < 0.0);
        assert_eq!(exact_f64(dtype, 1), -exact_f64(dtype, sign | 1));
        // A smallest subnormal moves toward zero to the zero of its own sign.
        assert_eq!(step(1, false), (0, moved(false)), "{dtype}");
        assert_eq!(step(sign | 1, true), (sign, moved(true)), "{dtype}");

        // A binade boundary: the spacing above is twice the spacing below.
        for boundary in [1.0, 2.0, 0.25, 1024.0] {
            for negate in [1.0, -1.0] {
                let value = boundary * negate;
                let payload = quantize(dtype, value);
                assert_eq!(exact_f64(dtype, payload), value);
                let above = exact_f64(dtype, step(payload, true).0);
                let below = exact_f64(dtype, step(payload, false).0);
                let (outward, inward) = if negate > 0.0 {
                    (above - value, value - below)
                } else {
                    (value - below, above - value)
                };
                assert!(outward > 0.0 && inward > 0.0, "{dtype} {value}");
                assert_eq!(outward, 2.0 * inward, "{dtype} {value}");
            }
        }

        // The largest finite magnitude: no move away from zero, never an
        // infinity; the move toward zero is an ordinary step.
        let infinity = quantize(dtype, f64::INFINITY);
        let largest = infinity - 1;
        assert!(exact_f64(dtype, largest).is_finite());
        assert_eq!(step(largest, true), (largest, blocked), "{dtype}");
        assert_eq!(
            step(sign | largest, false),
            (sign | largest, blocked),
            "{dtype}"
        );
        assert_eq!(step(largest, false), (largest - 1, moved(false)), "{dtype}");
        assert_eq!(
            step(sign | largest, true),
            (sign | (largest - 1), moved(true)),
            "{dtype}"
        );

        // Non-finite values never move, and keep their payload.
        let quiet = canonical_nan(dtype);
        for payload in [
            infinity,
            sign | infinity,
            infinity + 1,
            sign | (infinity + 1),
            quiet,
        ] {
            assert!(!exact_f64(dtype, payload).is_finite());
            for up in [true, false] {
                assert_eq!(
                    step(payload, up),
                    (payload, off_lattice),
                    "{dtype} {payload:#x}"
                );
            }
        }
    }
}

// ----------------------------------------------------------- code planes

fn pack_codes(bits: u32, codes: &[u32]) -> Vec<u8> {
    let per_word = (32 / bits) as usize;
    let mut words = vec![0u32; codes.len().div_ceil(per_word)];
    for (index, code) in codes.iter().enumerate() {
        words[index / per_word] |= code << (bits * (index % per_word) as u32);
    }
    words.into_iter().flat_map(u32::to_le_bytes).collect()
}

fn unpack_codes(bits: u32, bytes: &[u8]) -> Vec<u32> {
    let per_word = 32 / bits;
    bytes
        .chunks_exact(4)
        .flat_map(|word| {
            let word = u32::from_le_bytes(word.try_into().unwrap());
            (0..per_word).map(move |slot| (word >> (bits * slot)) & (u32::MAX >> (32 - bits)))
        })
        .collect()
}

#[test]
fn codes_step_to_the_adjacent_code_within_their_range() {
    let mut samples = Samples(3);
    for (bits, range) in [
        (8, 0..=255),
        (4, 0..=15),
        (4, 0..=9),
        (8, 3..=200),
        (2, 1..=2),
        (1, 0..=1),
        (16, 100..=40_000),
        (32, 5..=u32::MAX - 1),
        (32, 0..=u32::MAX),
        (8, 77..=77),
    ] {
        let lattice = Lattice::codes(bits, range.clone()).unwrap();
        assert_eq!(lattice.chunk_bytes(), 4);
        let mask = u32::MAX >> (32 - bits);
        let codes: Vec<u32> = if bits <= 8 {
            // Every code, repeated to fill whole words.
            (0..=mask)
                .cycle()
                .take((1usize << bits).max(32 / bits as usize))
                .collect()
        } else {
            let edges = [
                0,
                1,
                *range.start(),
                *range.end(),
                range.start().saturating_sub(1),
                (*range.end()).min(mask - 1) + 1,
                mask - 1,
                mask,
            ];
            edges
                .into_iter()
                .chain((0..992).map(|_| samples.next() as u32 & mask))
                .collect()
        };
        let before = pack_codes(bits, &codes);
        assert_eq!(unpack_codes(bits, &before), codes);
        for up in [true, false] {
            let mut after = before.clone();
            let tally = one(lattice.perturb(&all(up), 0, &mut after));
            let mut expected = Tally {
                values: codes.len() as u64,
                ..Tally::default()
            };
            for (code, stepped) in codes.iter().copied().zip(unpack_codes(bits, &after)) {
                let wanted = if !range.contains(&code) {
                    expected.off_lattice += 1;
                    code
                } else if up && code < *range.end() {
                    expected.up += 1;
                    code + 1
                } else if !up && code > *range.start() {
                    expected.down += 1;
                    code - 1
                } else {
                    expected.blocked += 1;
                    code
                };
                assert_eq!(stepped, wanted, "{bits}-bit {code} in {range:?} up={up}");
            }
            assert_eq!(tally, expected, "{bits}-bit {range:?} up={up}");
        }
    }
}

#[test]
fn descriptions_the_storage_cannot_have_are_refused() {
    for dtype in [DType::I32, DType::U32, DType::Bool] {
        assert_eq!(
            Lattice::dense(dtype).err(),
            Some(LatticeError::NotFloating(dtype))
        );
    }
    for bits in [0, 3, 5, 6, 12, 33, 64] {
        assert_eq!(
            Lattice::codes(bits, 0..=1).err(),
            Some(LatticeError::CodeWidth(bits))
        );
    }
    assert_eq!(
        Lattice::codes(4, 0..=16).err(),
        Some(LatticeError::CodeRange)
    );
    #[allow(clippy::reversed_empty_ranges)]
    let empty = 3..=2;
    assert_eq!(
        Lattice::codes(4, empty).err(),
        Some(LatticeError::CodeRange)
    );
    for (up, down) in [
        (-0.1, 0.0),
        (0.0, -0.1),
        (0.6, 0.5),
        (f64::NAN, 0.0),
        (0.0, f64::NAN),
    ] {
        assert_eq!(
            Dither::new(0, up, down).err(),
            Some(LatticeError::Probabilities)
        );
    }
    let mut refused = 0;
    for info in registry::representations() {
        match &info.kind {
            RepresentationKind::Dense(_) | RepresentationKind::External(_) => {
                assert_eq!(
                    Lattice::packed(info.id, &[4, 256]).err(),
                    Some(LatticeError::NotPacked(info.name))
                );
                refused += 1;
            }
            RepresentationKind::Packed(_) | RepresentationKind::PackedRows(_) => {
                for extents in [&[][..], &[0, 256], &[4, 0], &[u64::MAX, u64::MAX]] {
                    assert_eq!(
                        Lattice::packed(info.id, extents).err(),
                        Some(LatticeError::Geometry),
                        "{} {extents:?}",
                        info.name
                    );
                }
            }
        }
    }
    assert!(
        refused > DType::ALL.len(),
        "the registry has external representations"
    );
}

// ---------------------------------------------------------- level tables

#[test]
fn level_tables_order_codes_by_decoded_value() {
    // Sign-magnitude E2M1: raw 0..=7 are 0, .5, 1, 1.5, 2, 3, 4, 6 and raw
    // 8..=15 their negations, so code order is not value order and zero has
    // two codes. The stored code of a level is its lowest raw code.
    let e2m1 = Steps::of(&PlaneEncoding::FloatCode {
        format: FloatCodeFormat::E2M1,
    })
    .unwrap();
    assert_eq!(step_of(&e2m1, 9, true), Step::To(0));
    assert_eq!(step_of(&e2m1, 0, false), Step::To(9));
    assert_eq!(step_of(&e2m1, 8, false), Step::To(9));
    assert_eq!(step_of(&e2m1, 8, true), Step::To(1));
    assert_eq!(step_of(&e2m1, 0, true), Step::To(1));
    assert_eq!(step_of(&e2m1, 4, true), Step::To(5));
    assert_eq!(step_of(&e2m1, 12, true), Step::To(11));
    assert_eq!(step_of(&e2m1, 7, true), Step::Blocked);
    assert_eq!(step_of(&e2m1, 15, false), Step::Blocked);

    // UE4M3 ignores its sign bit: a code with the bit set steps to the
    // lowest code of the adjacent value. The all-ones magnitude is NaN.
    let ue4m3 = Steps::of(&PlaneEncoding::FloatCode {
        format: FloatCodeFormat::UE4M3,
    })
    .unwrap();
    assert_eq!(step_of(&ue4m3, 0x85, true), Step::To(0x06));
    assert_eq!(step_of(&ue4m3, 0x85, false), Step::To(0x04));
    assert_eq!(step_of(&ue4m3, 0x00, false), Step::Blocked);
    assert_eq!(step_of(&ue4m3, 0x7e, true), Step::Blocked);
    for raw in [0x7f, 0xff] {
        for up in [true, false] {
            assert_eq!(step_of(&ue4m3, raw, up), Step::OffLattice);
        }
    }

    // E8M0: powers of two in code order; 0xff is NaN.
    let e8m0 = Steps::of(&PlaneEncoding::FloatCode {
        format: FloatCodeFormat::E8M0,
    })
    .unwrap();
    assert_eq!(step_of(&e8m0, 0, false), Step::Blocked);
    assert_eq!(step_of(&e8m0, 0, true), Step::To(1));
    assert_eq!(step_of(&e8m0, 254, true), Step::Blocked);
    assert_eq!(step_of(&e8m0, 0xff, true), Step::OffLattice);

    // Two's complement wraps in code order at the sign.
    let twos = Steps::of(&PlaneEncoding::Packed {
        bits: 8,
        interpretation: CodeInterpretation::TwosComplement,
    })
    .unwrap();
    assert_eq!(step_of(&twos, 0x7f, true), Step::Blocked);
    assert_eq!(step_of(&twos, 0x80, false), Step::Blocked);
    assert_eq!(step_of(&twos, 0xff, true), Step::To(0x00));
    assert_eq!(step_of(&twos, 0x00, false), Step::To(0xff));
    assert_eq!(step_of(&twos, 0x80, true), Step::To(0x81));

    // A table whose values are neither ordered nor distinct.
    const TABLE: &[i32] = &[3, -1, 7, 0, 3, -5, 7, 2];
    let table = Steps::of(&PlaneEncoding::Packed {
        bits: 3,
        interpretation: CodeInterpretation::Table(TABLE),
    })
    .unwrap();
    // Ascending levels: -5 (5), -1 (1), 0 (3), 2 (7), 3 (0, 4), 7 (2, 6).
    for (raw, down, up) in [
        (5, Step::Blocked, Step::To(1)),
        (1, Step::To(5), Step::To(3)),
        (3, Step::To(1), Step::To(7)),
        (7, Step::To(3), Step::To(0)),
        (0, Step::To(7), Step::To(2)),
        (4, Step::To(7), Step::To(2)),
        (2, Step::To(0), Step::Blocked),
        (6, Step::To(0), Step::Blocked),
    ] {
        assert_eq!(step_of(&table, raw, false), down, "raw {raw} down");
        assert_eq!(step_of(&table, raw, true), up, "raw {raw} up");
    }

    assert!(Steps::of(&PlaneEncoding::Packed {
        bits: ENUMERABLE_CODE_BITS + 1,
        interpretation: CodeInterpretation::Unsigned,
    })
    .is_none());
    assert!(Steps::of(&PlaneEncoding::Dense(DType::U32)).is_none());
}

#[test]
fn every_registry_level_table_matches_the_plane_encoding() {
    let mut tables = 0;
    for info in packed_storages() {
        for plane in &packet_form(info).planes {
            let Some(Steps::Levels(steps)) = Steps::of(&plane.encoding) else {
                assert!(matches!(plane.encoding, PlaneEncoding::Dense(_)));
                continue;
            };
            tables += 1;
            let oracle = Oracle::plane(&plane.encoding);
            assert_eq!(steps.len(), 1 << plane.entry_bits);
            for raw in 0..1u32 << plane.entry_bits {
                let value = plane_value(&plane.encoding, raw);
                for up in [true, false] {
                    let step = steps[raw as usize][usize::from(up)];
                    if !value.is_finite() {
                        assert_eq!(step, Step::OffLattice);
                        continue;
                    }
                    match oracle.neighbour(value, up) {
                        None => assert_eq!(step, Step::Blocked),
                        Some(neighbour) => {
                            let Step::To(next) = step else {
                                panic!("`{}` {} raw {raw} up={up}: {step:?}", info.name, plane.name)
                            };
                            assert_eq!(plane_value(&plane.encoding, next), neighbour);
                            assert!(
                                (0..next).all(|lower| {
                                    plane_value(&plane.encoding, lower) != neighbour
                                }),
                                "the stored code of a level is its lowest raw code"
                            );
                        }
                    }
                }
            }
        }
    }
    assert!(tables > 0);
}

// ------------------------------------------------- packed representations

fn packed_storages() -> impl Iterator<Item = &'static RepresentationInfo> {
    registry::representations().iter().filter(|info| {
        matches!(
            info.kind,
            RepresentationKind::Packed(_) | RepresentationKind::PackedRows(_)
        )
    })
}

fn packet_form(info: &RepresentationInfo) -> &PackedPacketLayout {
    match &info.kind {
        RepresentationKind::Packed(packet) => packet,
        RepresentationKind::PackedRows(rows) => &rows.packet,
        _ => unreachable!("a packed storage"),
    }
}

/// A packed tensor under test: its storage and its extents.
struct Stored {
    info: &'static RepresentationInfo,
    extents: Vec<u64>,
}

impl Stored {
    fn rows(&self) -> u64 {
        self.extents[..self.extents.len() - 1].iter().product()
    }
    fn extent(&self) -> u64 {
        *self.extents.last().unwrap()
    }
    fn packet(&self) -> &'static PackedPacketLayout {
        packet_form(self.info)
    }
    fn packets_per_row(&self) -> u64 {
        self.packet().packet_extent(self.extent())
    }
    fn packet_form_bytes(&self) -> usize {
        (self.rows() * self.packets_per_row() * u64::from(self.packet().packet_size)) as usize
    }
    /// Canonical storage bytes holding the packet-form bytes `packets`.
    fn place(&self, packets: &[u8]) -> Vec<u8> {
        match &self.info.kind {
            RepresentationKind::PackedRows(rows) => rows.place(&self.extents, packets),
            _ => packets.to_vec(),
        }
    }
    /// The packet-form bytes of canonical storage bytes.
    fn packets(&self, bytes: &[u8]) -> Vec<u8> {
        match &self.info.kind {
            RepresentationKind::PackedRows(rows) => rows.packets(&self.extents, bytes),
            _ => bytes.to_vec(),
        }
    }
    /// Packet-form bytes with every bit pattern in every plane, exactly as
    /// the storage holds them: the `packet` layout keeps every byte of a
    /// packet, a row layout keeps the plane bytes and drops a packet's
    /// alignment padding.
    fn random_packets(&self, samples: &mut Samples) -> Vec<u8> {
        self.packets(&self.place(&samples.bytes(self.packet_form_bytes())))
    }
    fn lattice(&self) -> Lattice {
        Lattice::packed(self.info.id, &self.extents).unwrap()
    }
    /// Every stored entry: (plane, cell, entry within the packet).
    fn entries(&self) -> Vec<(usize, u64, u64)> {
        let packet = self.packet();
        let mut entries = Vec::new();
        for cell in 0..self.rows() * self.packets_per_row() {
            let first = cell % self.packets_per_row() * u64::from(packet.group);
            let values = (self.extent() - first).min(u64::from(packet.group));
            for (ordinal, plane) in packet.planes.iter().enumerate() {
                // The entries the values of the packet read (`PlaneSchema::entry`).
                let mut read: Vec<u64> = (0..values)
                    .flat_map(|value| {
                        (0..u64::from(plane.fields)).map(move |field| {
                            value / u64::from(plane.group) * u64::from(plane.fields) + field
                        })
                    })
                    .collect();
                read.sort_unstable();
                read.dedup();
                entries.extend(read.into_iter().map(|entry| (ordinal, cell, entry)));
            }
        }
        entries
    }
    /// First packet-form bit of an entry.
    fn entry_bit(&self, plane: usize, cell: u64, entry: u64) -> usize {
        let packet = self.packet();
        let schema = &packet.planes[plane];
        (cell as usize * packet.packet_size as usize + schema.offset as usize) * 8
            + entry as usize * schema.entry_bits as usize
    }
    /// A logical value (flat index) that reads the entry, and the field.
    fn reader(&self, plane: usize, cell: u64, entry: u64) -> (usize, usize) {
        let packet = self.packet();
        let schema = &packet.planes[plane];
        let (row, index) = (cell / self.packets_per_row(), cell % self.packets_per_row());
        let column = index * u64::from(packet.group)
            + entry / u64::from(schema.fields) * u64::from(schema.group);
        (
            (row * self.extent() + column) as usize,
            (entry % u64::from(schema.fields)) as usize,
        )
    }
}

/// The semantic role holding the decoded value of each (plane, field): the
/// temporary the recipe computes directly from the plane read.
fn entry_roles(recipe: &DecodeRecipe) -> Vec<Vec<QuantizedComponent>> {
    recipe
        .planes()
        .iter()
        .map(|schema| {
            (0..schema.fields)
                .map(|wanted| {
                    let read = recipe
                        .steps()
                        .iter()
                        .find_map(|step| match step {
                            DecodeStep::ReadPlaneField { into, plane, field }
                                if *plane == schema.ordinal && *field == wanted =>
                            {
                                Some(*into)
                            }
                            _ => None,
                        })
                        .expect("the recipe reads every plane field");
                    let decoded = recipe
                        .steps()
                        .iter()
                        .find(|step| step.uses() == [read])
                        .expect("the recipe decodes every plane read")
                        .defines();
                    QuantizedComponent::ALL
                        .into_iter()
                        .find(|role| {
                            recipe.component(*role) == Some(DecodeComponent::Temporary(decoded))
                        })
                        .expect("every decoded plane entry is a semantic component")
                })
                .collect()
        })
        .collect()
}

/// The reference decoder's value of every (plane, field) for every logical
/// value of the stored tensor.
fn observe(stored: &Stored, bytes: &[u8]) -> Vec<Vec<Vec<f64>>> {
    let recipe = registry::decode_recipe(stored.info.id, stored.info.decoded).unwrap();
    let shape: Vec<usize> = stored
        .extents
        .iter()
        .map(|extent| *extent as usize)
        .collect();
    let tensor = TensorData::encoded(stored.info.id, shape, bytes.to_vec()).unwrap();
    entry_roles(&recipe)
        .into_iter()
        .map(|fields| {
            fields
                .into_iter()
                .map(|role| tensor.components(role).unwrap())
                .collect()
        })
        .collect()
}

/// What happened to the entries of one plane, by the reference decoder.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Observed {
    values: u64,
    up: u64,
    down: u64,
    /// Unchanged finite entries with no adjacent level above / below.
    top: u64,
    bottom: u64,
    /// Unchanged finite entries with adjacent levels on both sides.
    interior: u64,
    non_finite: u64,
}

/// Perturb the canonical bytes of `packets`, whose reference decode is
/// `decoded_before`, and check, through the reference decoder, that every
/// entry is unchanged or at exactly the adjacent decoded level of its own
/// plane, and that no bit outside a moved entry changed.
fn perturb_and_examine(
    stored: &Stored,
    packets: &[u8],
    decoded_before: &[Vec<Vec<f64>>],
    dither: &Dither,
) -> (Vec<Tally>, Vec<Observed>) {
    let name = stored.info.name;
    let packet = stored.packet();
    let mut after = stored.place(packets);
    let tallies = stored.lattice().perturb(dither, 0, &mut after);
    assert_eq!(tallies.len(), packet.planes.len());
    let after_packets = stored.packets(&after);
    // Nothing outside the packet-form bits of logical rows and packets
    // (alignment padding, padding groups, padding rows) changed.
    assert_eq!(
        stored.place(&after_packets),
        after,
        "`{name}` padding changed"
    );

    let decoded_after = observe(stored, &after);
    let oracles: Vec<Oracle> = packet
        .planes
        .iter()
        .map(|plane| Oracle::plane(&plane.encoding))
        .collect();
    let mut observed = vec![Observed::default(); packet.planes.len()];
    let mut expected_packets = packets.to_vec();
    for (plane, cell, entry) in stored.entries() {
        let schema = &packet.planes[plane];
        let bit = stored.entry_bit(plane, cell, entry);
        let (flat, field) = stored.reader(plane, cell, entry);
        let (raw_before, raw_after) = (
            read_bits(packets, bit, schema.entry_bits),
            read_bits(&after_packets, bit, schema.entry_bits),
        );
        let (value_before, value_after) = (
            decoded_before[plane][field][flat],
            decoded_after[plane][field][flat],
        );
        let context = format!("`{name}` plane `{}` cell {cell} entry {entry}", schema.name);
        // The entry addressed here is the one the reference decoder reads.
        let direct = plane_value(&schema.encoding, raw_before);
        assert!(
            direct == value_before || (direct.is_nan() && value_before.is_nan()),
            "{context}: {direct} != {value_before}"
        );
        write_bits(&mut expected_packets, bit, schema.entry_bits, raw_after);
        let seen = &mut observed[plane];
        seen.values += 1;
        if !value_before.is_finite() {
            assert_eq!(raw_after, raw_before, "{context}: a non-finite entry moved");
            seen.non_finite += 1;
            continue;
        }
        assert!(value_after.is_finite(), "{context}: became non-finite");
        let above = oracles[plane].neighbour(value_before, true);
        let below = oracles[plane].neighbour(value_before, false);
        if value_after == value_before {
            assert_eq!(
                raw_after, raw_before,
                "{context}: bits changed without a move"
            );
            match (below, above) {
                (_, None) => seen.top += 1,
                (None, _) => seen.bottom += 1,
                (Some(_), Some(_)) => seen.interior += 1,
            }
        } else if Some(value_after) == above {
            seen.up += 1;
        } else if Some(value_after) == below {
            seen.down += 1;
        } else {
            panic!("{context}: {value_before} moved to {value_after}, which is not adjacent");
        }
    }
    // The only packet-form bits that changed are bits of stored entries.
    assert_eq!(
        expected_packets, after_packets,
        "`{name}` bits outside entries changed"
    );
    for (tally, seen) in tallies.iter().zip(&observed) {
        assert_eq!(tally.values, seen.values, "`{name}`");
        assert_eq!(tally.up, seen.up, "`{name}`");
        assert_eq!(tally.down, seen.down, "`{name}`");
    }
    (tallies, observed)
}

/// Extents with several packets per row, a partial last packet for every
/// registered group, and row counts that are not whole row tiles.
const EXTENTS: [u64; 3] = [2, 3, 300];

/// The reference decoder evaluates a whole decode recipe per logical value
/// and component, so the decoded checks use the smallest tensor that still
/// has two rows, two packets per row for every registered group, and a
/// partial last packet.
const DECODED_EXTENTS: [u64; 2] = [2, 270];

/// Mixed moves through the reference decoder: both directions and unmoved
/// entries in the code plane, every plane checked entry by entry.
fn assert_mixed_moves_are_adjacent(stored: &Stored, packets: &[u8], decoded: &[Vec<Vec<f64>>]) {
    let (tallies, observed) = perturb_and_examine(stored, packets, decoded, &dither(41, 0.3, 0.3));
    let name = stored.info.name;
    assert!(
        observed[0].up > 0 && observed[0].down > 0 && observed[0].interior > 0,
        "`{name}`: {:?}",
        observed[0]
    );
    for tally in tallies {
        assert!(
            tally.up + tally.down + tally.blocked + tally.off_lattice < tally.values,
            "`{name}`: every entry drew a move"
        );
    }
}

#[test]
fn every_packed_representation_steps_each_plane_to_its_adjacent_decoded_level() {
    let mut samples = Samples(29);
    let mut representations = 0;
    for info in packed_storages() {
        if info.layout != Layout::Packet {
            continue;
        }
        representations += 1;
        let stored = Stored {
            info,
            extents: DECODED_EXTENTS.to_vec(),
        };
        let packets = stored.random_packets(&mut samples);
        let decoded = observe(&stored, &stored.place(&packets));
        // Every entry draws a move in one direction: each one moves to the
        // adjacent level on that side, or has none, or is not finite.
        for up in [true, false] {
            let (tallies, observed) = perturb_and_examine(&stored, &packets, &decoded, &all(up));
            assert!(
                observed[0].up + observed[0].down > 0,
                "`{}` up={up}",
                info.name
            );
            for (plane, (tally, seen)) in tallies.iter().zip(&observed).enumerate() {
                let context = format!("`{}` plane {plane} up={up}", info.name);
                assert_eq!(seen.interior, 0, "{context}: an interior entry stayed");
                let (moved, ended, other_end) = if up {
                    (seen.up, seen.top, seen.bottom)
                } else {
                    (seen.down, seen.bottom, seen.top)
                };
                assert_eq!(other_end, 0, "{context}: an entry with room stayed");
                assert_eq!(
                    *tally,
                    Tally {
                        values: seen.values,
                        up: if up { moved } else { 0 },
                        down: if up { 0 } else { moved },
                        blocked: ended,
                        off_lattice: seen.non_finite,
                    },
                    "{context}"
                );
            }
        }
        assert_mixed_moves_are_adjacent(&stored, &packets, &decoded);
    }
    assert_eq!(
        representations,
        packed_storages()
            .map(|info| info.representation)
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
        "every packed representation has packet storage"
    );
}

/// One storage of every row layout through the reference decoder. The other
/// row storages are held to the packet layout bit for bit
/// (`a_row_layout_receives_the_packet_layout_perturbation`).
#[test]
fn every_row_layout_steps_each_plane_to_its_adjacent_decoded_level() {
    let mut samples = Samples(31);
    for layout in Layout::ALL {
        if layout == Layout::Packet {
            continue;
        }
        let info = packed_storages()
            .find(|info| info.layout == layout)
            .unwrap_or_else(|| panic!("no storage in `{}`", layout.as_str()));
        let stored = Stored {
            info,
            extents: DECODED_EXTENTS.to_vec(),
        };
        let packets = stored.random_packets(&mut samples);
        let decoded = observe(&stored, &stored.place(&packets));
        assert_mixed_moves_are_adjacent(&stored, &packets, &decoded);
    }
}

#[test]
fn a_row_layout_receives_the_packet_layout_perturbation() {
    let mut samples = Samples(53);
    let mut layouts = std::collections::BTreeSet::new();
    for info in packed_storages() {
        let RepresentationKind::PackedRows(rows) = &info.kind else {
            continue;
        };
        layouts.insert(rows.layout);
        let packet = registry::storage(info.representation, Layout::Packet)
            .map(registry::representation_info)
            .expect("a resident representation has packet storage");
        for extents in [&EXTENTS[..], &[1, 40, 1000], &[37, 64], &[3, 2, 33, 257]] {
            let (resident, portable) = (
                Stored {
                    info,
                    extents: extents.to_vec(),
                },
                Stored {
                    info: packet,
                    extents: extents.to_vec(),
                },
            );
            let dither = dither(samples.next(), 0.2, 0.1);
            let mut packets = samples.bytes(portable.packet_form_bytes());
            let mut bytes = resident.place(&packets);
            let expected = portable.lattice().perturb(&dither, 0, &mut packets);
            let tallies = resident.lattice().perturb(&dither, 0, &mut bytes);
            assert_eq!(tallies, expected, "`{}` {extents:?}", info.name);
            assert!(
                bytes == resident.place(&packets),
                "`{}` {extents:?}",
                info.name
            );
        }
    }
    assert_eq!(
        layouts.into_iter().collect::<Vec<_>>(),
        Layout::ALL
            .into_iter()
            .filter(|layout| *layout != Layout::Packet)
            .collect::<Vec<_>>(),
        "every row layout is exercised"
    );
}

// ------------------------------------------------------------ determinism

/// Perturb `bytes` in runs of chunks of varying length, last run first.
fn perturb_in_runs(lattice: &Lattice, dither: &Dither, bytes: &mut [u8]) -> Vec<Tally> {
    let chunk_bytes = lattice.chunk_bytes();
    let chunks = bytes.len() / chunk_bytes;
    let mut runs = Vec::new();
    let (mut first, mut length) = (0, 1);
    while first < chunks {
        let run = length.min(chunks - first);
        runs.push((first, run));
        first += run;
        length = length * 3 % 7 + 1;
    }
    let mut total: Vec<Tally> = Vec::new();
    for (first, run) in runs.into_iter().rev() {
        let tallies = lattice.perturb(
            dither,
            first as u64,
            &mut bytes[first * chunk_bytes..(first + run) * chunk_bytes],
        );
        total.resize(tallies.len(), Tally::default());
        for (sum, tally) in total.iter_mut().zip(tallies) {
            *sum += tally;
        }
    }
    total
}

fn assert_chunking_is_invisible(lattice: &Lattice, bytes: &[u8], name: &str) {
    let dither = dither(97, 0.2, 0.15);
    let mut whole = bytes.to_vec();
    let expected = lattice.perturb(&dither, 0, &mut whole);
    assert!(whole != bytes, "`{name}`: nothing moved");
    let mut again = bytes.to_vec();
    assert_eq!(
        lattice.perturb(&dither, 0, &mut again),
        expected,
        "`{name}`"
    );
    assert!(again == whole, "`{name}`: a repeated perturbation differs");
    let mut chunked = bytes.to_vec();
    assert_eq!(
        perturb_in_runs(lattice, &dither, &mut chunked),
        expected,
        "`{name}`"
    );
    assert!(
        chunked == whole,
        "`{name}`: chunking changed the perturbation"
    );
    let mut reseeded = bytes.to_vec();
    lattice.perturb(&Dither::new(98, 0.2, 0.15).unwrap(), 0, &mut reseeded);
    assert!(reseeded != whole, "`{name}`: the seed has no effect");
}

#[test]
fn the_perturbation_depends_on_seed_and_position_not_on_chunking() {
    let mut samples = Samples(71);
    for dtype in FLOATS {
        let bytes = samples.bytes(4096 * dtype.bytes() as usize);
        assert_chunking_is_invisible(&Lattice::dense(dtype).unwrap(), &bytes, dtype.name());
    }
    for (bits, range) in [(8, 0..=255), (4, 0..=15), (4, 2..=11)] {
        let bytes = samples.bytes(4096);
        assert_chunking_is_invisible(&Lattice::codes(bits, range).unwrap(), &bytes, "codes");
    }
    for info in packed_storages() {
        let stored = Stored {
            info,
            extents: vec![3, 70, 300],
        };
        let bytes = stored.place(&samples.bytes(stored.packet_form_bytes()));
        let lattice = stored.lattice();
        assert_eq!(
            bytes.len() as u64,
            registry::canonical_bytes(info.id, &stored.extents).unwrap()
        );
        assert!(
            bytes.len() / lattice.chunk_bytes() > 3,
            "`{}` has several chunks",
            info.name
        );
        assert_chunking_is_invisible(&lattice, &bytes, info.name);
    }
}

#[test]
#[should_panic(expected = "not whole")]
fn a_partial_chunk_is_a_caller_contradiction() {
    Lattice::dense(DType::F32)
        .unwrap()
        .perturb(&all(true), 0, &mut [0; 6]);
}

#[test]
#[should_panic(expected = "outside the tensor")]
fn chunks_beyond_a_packed_tensor_are_a_caller_contradiction() {
    let info = packed_storages().next().unwrap();
    let lattice = Lattice::packed(info.id, &EXTENTS).unwrap();
    let mut bytes = vec![0; lattice.chunk_bytes()];
    let chunks = registry::canonical_bytes(info.id, &EXTENTS).unwrap() / bytes.len() as u64;
    lattice.perturb(&all(true), chunks, &mut bytes);
}

// ------------------------------------------------------------- statistics

/// Five standard deviations of a binomial count: a fixed seed either passes
/// or shows a defect, and a correct generator fails one such bound with
/// probability below 6e-7.
fn binomial_bound(trials: u64, probability: f64) -> f64 {
    5.0 * (trials as f64 * probability * (1.0 - probability)).sqrt()
}

fn assert_binomial(count: u64, trials: u64, probability: f64, context: &str) {
    let expected = trials as f64 * probability;
    assert!(
        (count as f64 - expected).abs() <= binomial_bound(trials, probability),
        "{context}: {count} of {trials} at probability {probability} (expected {expected})"
    );
}

#[test]
fn realised_move_fractions_match_the_supplied_probabilities() {
    const VALUES: u64 = 1 << 20;
    let mut samples = Samples(113);
    // Finite, normal, far from the largest magnitude: nothing is blocked.
    let floats: Vec<u8> = (0..VALUES)
        .flat_map(|_| {
            let bits = samples.next() as u32;
            ((bits & 0x807f_ffff) | ((100 + (bits >> 23) % 50) << 23)).to_le_bytes()
        })
        .collect();
    for (seed, up, down) in [
        (1, 1.0 / 24.0, 1.0 / 24.0),
        (2, 0.2, 0.05),
        (3, 0.0, 0.5),
        (4, 0.001, 0.0),
        (5, 0.0, 0.0),
        (6, 0.5, 0.5),
    ] {
        let mut bytes = floats.clone();
        let tally = one(Lattice::dense(DType::F32).unwrap().perturb(
            &dither(seed, up, down),
            0,
            &mut bytes,
        ));
        let context = format!("f32 up={up} down={down}");
        assert_eq!(
            (tally.values, tally.blocked, tally.off_lattice),
            (VALUES, 0, 0)
        );
        assert_binomial(tally.up, VALUES, up, &context);
        assert_binomial(tally.down, VALUES, down, &context);
        // The tally is the realised perturbation.
        let (mut moved_up, mut moved_down) = (0, 0);
        for (before, after) in floats.chunks_exact(4).zip(bytes.chunks_exact(4)) {
            let before = f32::from_le_bytes(before.try_into().unwrap());
            let after = f32::from_le_bytes(after.try_into().unwrap());
            moved_up += u64::from(after > before);
            moved_down += u64::from(after < before);
        }
        assert_eq!((moved_up, moved_down), (tally.up, tally.down), "{context}");
    }

    // Every plane of every packed representation draws moves at the supplied
    // total rate, whatever the stored values make of the draw.
    for info in packed_storages() {
        if !matches!(info.kind, RepresentationKind::Packed(_)) {
            continue;
        }
        let stored = Stored {
            info,
            extents: vec![64, 4096],
        };
        let mut bytes = samples.bytes(stored.packet_form_bytes());
        let (up, down) = (0.07, 0.13);
        for (plane, tally) in stored
            .lattice()
            .perturb(&dither(17, up, down), 0, &mut bytes)
            .into_iter()
            .enumerate()
        {
            assert_binomial(
                tally.up + tally.down + tally.blocked + tally.off_lattice,
                tally.values,
                up + down,
                &format!("`{}` plane {plane}", info.name),
            );
        }
    }
}

#[test]
fn a_uniform_lattice_realises_the_variance_of_its_move_probability() {
    const VALUES: usize = 1 << 20;
    // An affine by-value codec: decoded = code * scale + zero, so the
    // lattice is uniform with step `scale`.
    let (scale, zero) = (0.37f64, -11.5f64);
    let mut samples = Samples(131);
    // Codes away from both ends of the range: nothing is blocked.
    let codes: Vec<u32> = (0..VALUES)
        .map(|_| 64 + samples.next() as u32 % 128)
        .collect();
    let before = pack_codes(8, &codes);
    let lattice = Lattice::codes(8, 0..=255).unwrap();
    for (seed, probability) in [(1, 1.0 / 24.0), (2, 0.25), (3, 0.01)] {
        let mut after = before.clone();
        let tally = one(lattice.perturb(&dither(seed, probability, probability), 0, &mut after));
        assert_eq!(tally.blocked + tally.off_lattice, 0);
        let (mut sum, mut squares) = (0.0, 0.0);
        for (code, stepped) in codes.iter().zip(unpack_codes(8, &after)) {
            let difference =
                (f64::from(stepped) * scale + zero) - (f64::from(*code) * scale + zero);
            sum += difference;
            squares += difference * difference;
        }
        let count = VALUES as f64;
        let variance = squares / count;
        let expected = 2.0 * probability * scale * scale;
        // A squared difference is `scale^2` with probability 2p, else 0.
        let bound = scale * scale * binomial_bound(VALUES as u64, 2.0 * probability) / count;
        assert!(
            (variance - expected).abs() <= bound,
            "p={probability}: variance {variance}, expected {expected} +- {bound}"
        );
        // The mean is zero: its standard deviation is sqrt(variance / n).
        assert!(
            (sum / count).abs() <= 5.0 * (expected / count).sqrt(),
            "p={probability}: mean {}",
            sum / count
        );
        if probability == 1.0 / 24.0 {
            // One move in 24 each way is the variance of uniform noise over
            // one resolution cell.
            assert!((expected - scale * scale / 12.0).abs() < 1e-15);
        }
    }
}

// --------------------------------------------------------------- property

#[test]
fn perturbing_never_leaves_the_lattice() {
    let mut samples = Samples(151);
    for round in 0..24 {
        let up = samples.unit();
        let down = samples.unit() * (1.0 - up);
        let dither = dither(samples.next(), up, down);

        for dtype in FLOATS {
            let width = dtype.bytes() as usize;
            let before = samples.bytes(8192 * width);
            let mut after = before.clone();
            Lattice::dense(dtype)
                .unwrap()
                .perturb(&dither, 0, &mut after);
            for index in 0..8192 {
                let before = read_bits(&before, index * width * 8, width as u32 * 8);
                let after = read_bits(&after, index * width * 8, width as u32 * 8);
                if exact_f64(dtype, before).is_finite() {
                    assert!(exact_f64(dtype, after).is_finite(), "{dtype} {before:#x}");
                } else {
                    assert_eq!(after, before, "{dtype} {before:#x}");
                }
            }
        }

        for (bits, range) in [
            (8, 0..=255),
            (4, 0..=15),
            (4, 3..=12),
            (8, 0..=100),
            (16, 9..=9000),
        ] {
            let before = samples.bytes(4096);
            let mut after = before.clone();
            Lattice::codes(bits, range.clone())
                .unwrap()
                .perturb(&dither, 0, &mut after);
            for (before, after) in unpack_codes(bits, &before)
                .into_iter()
                .zip(unpack_codes(bits, &after))
            {
                if range.contains(&before) {
                    assert!(range.contains(&after), "{bits}-bit {before} -> {after}");
                } else {
                    assert_eq!(after, before);
                }
            }
        }

        // Every sixth storage per round, each one in four rounds.
        for info in packed_storages().skip(round as usize % 6).step_by(6) {
            let stored = Stored {
                info,
                extents: vec![1 + round % 3, 5, 150 + 13 * round],
            };
            let packets = stored.random_packets(&mut samples);
            let mut bytes = stored.place(&packets);
            stored.lattice().perturb(&dither, 0, &mut bytes);
            let after = stored.packets(&bytes);
            assert!(
                stored.place(&after) == bytes,
                "`{}` padding changed",
                info.name
            );
            let mut expected = packets.clone();
            for (plane, cell, entry) in stored.entries() {
                let schema = &stored.packet().planes[plane];
                let bit = stored.entry_bit(plane, cell, entry);
                let raw_before = read_bits(&packets, bit, schema.entry_bits);
                let raw_after = read_bits(&after, bit, schema.entry_bits);
                write_bits(&mut expected, bit, schema.entry_bits, raw_after);
                if plane_value(&schema.encoding, raw_before).is_finite() {
                    assert!(
                        plane_value(&schema.encoding, raw_after).is_finite(),
                        "`{}` plane `{}`: {raw_before:#x} -> {raw_after:#x}",
                        info.name,
                        schema.name
                    );
                } else {
                    assert_eq!(
                        raw_after, raw_before,
                        "`{}` plane `{}`",
                        info.name, schema.name
                    );
                }
            }
            assert!(
                expected == after,
                "`{}` bits outside entries changed",
                info.name
            );
        }
    }
}

// ------------------------------------------------------------- throughput

/// Bytes per second of `perturb` at the design probabilities, per storage
/// family. A measurement, not a check:
/// `cargo test --release --locked -p seismic-lang --lib lattice::tests::throughput -- --ignored --nocapture`
#[test]
#[ignore = "throughput measurement; run in release"]
fn throughput() {
    let dither = dither(1, 1.0 / 24.0, 1.0 / 24.0);
    let mut samples = Samples(1);
    let mut measure = |name: &str, lattice: Lattice, bytes: usize| {
        let bytes = bytes / lattice.chunk_bytes() * lattice.chunk_bytes();
        let mut buffer: Vec<u8> = (0..bytes.div_ceil(8))
            .flat_map(|_| samples.next().to_le_bytes())
            .take(bytes)
            .collect();
        let start = std::time::Instant::now();
        let tallies = lattice.perturb(&dither, 0, &mut buffer);
        let seconds = start.elapsed().as_secs_f64();
        let values: u64 = tallies.iter().map(|tally| tally.values).sum();
        println!(
            "{name:28} {:9.1} MB/s {:8.1} M values/s ({bytes} bytes)",
            bytes as f64 / seconds / 1e6,
            values as f64 / seconds / 1e6,
        );
    };
    for dtype in FLOATS {
        measure(dtype.name(), Lattice::dense(dtype).unwrap(), 256 << 20);
    }
    measure(
        "codes 8-bit",
        Lattice::codes(8, 0..=255).unwrap(),
        256 << 20,
    );
    measure("codes 4-bit", Lattice::codes(4, 0..=15).unwrap(), 256 << 20);
    for info in packed_storages() {
        let extents = [2048, 8192];
        let bytes = registry::canonical_bytes(info.id, &extents).unwrap() as usize;
        measure(
            info.name,
            Lattice::packed(info.id, &extents).unwrap(),
            bytes,
        );
    }
}
