//! Measuring a candidate's observation against the default's on identical
//! inputs: per subject, how far the candidate moved it and how coarse the
//! floating storage it is judged against is. A pure function of the two
//! observations; it reaches no verdict ([`super::admission`] does).
//!
//! The scale is the rounding step of floating storage. Rounding a value
//! into floating type `T` moves it within the rounding cell of the result,
//! whose width `w_T(v)` is half the gap to the adjacent representable value
//! below plus half the gap to the one above
//! ([`seismic_lang::lattice::RoundingCells`]); the rounding error has
//! variance `w² / 12`. For a floating subject with default elements `d` and
//! candidate elements `c`, over the elements finite in both:
//!
//! ```text
//! error_power  = Σ (c - d)²
//! step_power_T = Σ w_T(d)² / 12
//! ```
//!
//! Admission reads `sqrt(error_power / step_power_T)`: the candidate's error
//! in units of what rounding the subject into `T` adds.
use super::observation::{Observation, Page, Subject};
use seismic_compiler::prepared::ArgumentValue;
use seismic_lang::{
    lattice::RoundingCells,
    reference_math::conversion::exact_f64,
    registry::{self, RepresentationKind},
    types::DType,
};
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

/// The version of this comparison: what is measured and how. Part of the
/// identity of every measurement, wherever one is keyed, stored or traced.
pub const COMPARISON_VERSION: u32 = 2;

/// The rounding step power of a subject against one floating type.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct StepPower {
    pub dtype: DType,
    pub power: f64,
}

/// How a candidate's subject differs from the default's.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Difference {
    /// The floating type the subject is stored as; none for integer,
    /// Boolean and packed storage, which admit no difference.
    pub stored: Option<DType>,
    pub elements: u64,
    /// Elements whose stored bits differ.
    pub differing: u64,
    /// Differing elements of storage that admits no difference.
    pub exact_mismatches: u64,
    /// Elements not finite in the default or the candidate whose bits
    /// differ.
    pub non_finite_mismatches: u64,
    /// Over the elements finite in both: Σ (c - d)², Σ d² and the largest
    /// |c - d|.
    pub error_power: f64,
    pub reference_power: f64,
    pub maximum_absolute_error: f64,
    /// Σ w_T(d)² / 12 over the same elements, against the stored type
    /// first and then every other type asked for.
    pub step_power: Vec<StepPower>,
}

impl Difference {
    /// The error in rounding steps of `dtype`: sqrt(error_power /
    /// step_power). A zero step power permits no error. None when the
    /// subject was not measured against `dtype`.
    pub fn ratio(&self, dtype: DType) -> Option<f64> {
        let step = self.step_power.iter().find(|step| step.dtype == dtype)?;
        Some(if self.error_power == 0. {
            0.
        } else if step.power == 0. {
            f64::INFINITY
        } else {
            (self.error_power / step.power).sqrt()
        })
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum SubjectComparison {
    /// Byte for byte the default's.
    Identical,
    /// Another representation, shape or kind of value: nothing to measure.
    Mismatched,
    Differs(Difference),
}

/// One subject of one call, measured.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SubjectMeasurement {
    pub subject: String,
    pub comparison: SubjectComparison,
}

/// Every finite magnitude of a floating type narrow enough to enumerate, in
/// ascending order (which is payload order), each with the `w² / 12` of its
/// cell. Built from the type's own adjacency.
struct Enumerated {
    magnitudes: Box<[f64]>,
    terms: Box<[f64]>,
}

impl Enumerated {
    fn of(cells: &RoundingCells) -> Self {
        let (magnitudes, terms) = (0..=u32::from(u16::MAX))
            .map_while(|bits| {
                let width = cells.width(bits)?;
                Some((exact_f64(cells.dtype(), bits), width * width / 12.))
            })
            .unzip::<_, _, Vec<_>, Vec<_>>();
        Self {
            magnitudes: magnitudes.into(),
            terms: terms.into(),
        }
    }

    /// The payload of the magnitude whose rounding cell holds `magnitude`:
    /// the nearest, the even payload of two equally near. None beyond the
    /// cell of the largest, from where rounding overflows.
    fn containing(&self, magnitude: f64) -> Option<usize> {
        let above = self.magnitudes.partition_point(|value| *value <= magnitude);
        let below = above - 1;
        let lower = self.magnitudes[below];
        match self.magnitudes.get(above) {
            Some(upper) => {
                let midpoint = lower + (upper - lower) / 2.;
                Some(
                    if magnitude < midpoint || (magnitude == midpoint && below % 2 == 0) {
                        below
                    } else {
                        above
                    },
                )
            }
            // The largest magnitude's cell extends half its inward gap
            // beyond it.
            None => (magnitude - lower < (lower - self.magnitudes[below - 1]) / 2.)
                .then_some(below),
        }
    }
}

/// The `w² / 12` terms of one floating type.
enum Steps {
    /// A sixteen-bit type, enumerated once.
    Enumerated(DType, &'static Enumerated),
    /// `f32`, computed per value.
    Wide(RoundingCells),
}

impl Steps {
    fn of(dtype: DType) -> Self {
        static F16: OnceLock<Enumerated> = OnceLock::new();
        static BF16: OnceLock<Enumerated> = OnceLock::new();
        let cells = RoundingCells::of(dtype).expect("a floating subject has rounding cells");
        match dtype {
            DType::F16 => Self::Enumerated(dtype, F16.get_or_init(|| Enumerated::of(&cells))),
            DType::BF16 => Self::Enumerated(dtype, BF16.get_or_init(|| Enumerated::of(&cells))),
            DType::F32 => Self::Wide(cells),
            DType::I32 | DType::U32 | DType::Bool => {
                unreachable!("only floating types have rounding cells")
            }
        }
    }

    fn dtype(&self) -> DType {
        match self {
            Self::Enumerated(dtype, _) => *dtype,
            Self::Wide(cells) => cells.dtype(),
        }
    }

    /// The term of the finite value `value` stored as `stored` with
    /// payload `bits`: of its own cell when this is its stored type, else
    /// of the cell it rounds into; nothing for a value beyond the type's
    /// largest cell.
    fn term(&self, stored: DType, bits: u32, value: f64) -> f64 {
        match self {
            Self::Enumerated(dtype, enumerated) => {
                let payload = if stored == *dtype {
                    // Its magnitude's payload: the bits below the sign.
                    Some((bits & ((1 << (dtype.bytes() * 8 - 1)) - 1)) as usize)
                } else {
                    enumerated.containing(value.abs())
                };
                payload.map_or(0., |payload| enumerated.terms[payload])
            }
            Self::Wide(cells) => {
                let payload = if stored == cells.dtype() {
                    bits
                } else {
                    // Every finite value of a narrower type is an `f32`.
                    (value as f32).to_bits()
                };
                cells.width(payload).map_or(0., |width| width * width / 12.)
            }
        }
    }
}

fn element_bits(bytes: &[u8]) -> u32 {
    let mut bits = [0; 4];
    bits[..bytes.len()].copy_from_slice(bytes);
    u32::from_le_bytes(bits)
}

/// Measure the floating elements `(default bits, candidate bits)`.
fn measure_floating(
    stored: DType,
    elements: impl Iterator<Item = (u32, u32)>,
    activation: Option<DType>,
) -> Difference {
    let steps = std::iter::once(stored)
        .chain(activation.filter(|activation| *activation != stored))
        .map(Steps::of)
        .collect::<Vec<_>>();
    let mut difference = Difference {
        stored: Some(stored),
        elements: 0,
        differing: 0,
        exact_mismatches: 0,
        non_finite_mismatches: 0,
        error_power: 0.,
        reference_power: 0.,
        maximum_absolute_error: 0.,
        step_power: steps
            .iter()
            .map(|steps| StepPower {
                dtype: steps.dtype(),
                power: 0.,
            })
            .collect(),
    };
    for (default, candidate) in elements {
        difference.elements += 1;
        difference.differing += u64::from(default != candidate);
        let (d, c) = (exact_f64(stored, default), exact_f64(stored, candidate));
        if !d.is_finite() || !c.is_finite() {
            difference.non_finite_mismatches += u64::from(default != candidate);
            continue;
        }
        let error = (c - d).abs();
        difference.error_power += error * error;
        difference.reference_power += d * d;
        difference.maximum_absolute_error = difference.maximum_absolute_error.max(error);
        for (steps, power) in steps.iter().zip(&mut difference.step_power) {
            power.power += steps.term(stored, default, d);
        }
    }
    difference
}

/// Count the differing elements of storage that admits no difference.
fn measure_exact(elements: u64, differing: u64) -> Difference {
    Difference {
        stored: None,
        elements,
        differing,
        exact_mismatches: differing,
        non_finite_mismatches: 0,
        error_power: 0.,
        reference_power: 0.,
        maximum_absolute_error: 0.,
        step_power: Vec::new(),
    }
}

fn scalar_bits(value: &ArgumentValue) -> Option<(DType, u32)> {
    Some(match value {
        ArgumentValue::F32(v) => (DType::F32, v.to_bits()),
        ArgumentValue::F16(v) => (DType::F16, u32::from(*v)),
        ArgumentValue::BF16(v) => (DType::BF16, u32::from(*v)),
        _ => return None,
    })
}

fn compare_subject(
    default: &Observation,
    candidate: &Observation,
    unwritten: bool,
    activation: Option<DType>,
) -> SubjectComparison {
    match (default, candidate) {
        (
            Observation::Tensor {
                representation: dr,
                shape: ds,
                bytes: db,
            },
            Observation::Tensor {
                representation: cr,
                shape: cs,
                bytes: cb,
            },
        ) => {
            if dr != cr
                || ds != cs
                || db.len() != cb.len()
                || db.iter().zip(cb).any(|(d, c)| d.len() != c.len())
            {
                return SubjectComparison::Mismatched;
            }
            if db == cb {
                return SubjectComparison::Identical;
            }
            let pairs = |width: usize| {
                db.iter()
                    .zip(cb)
                    .flat_map(move |(d, c): (&Page, &Page)| {
                        d.chunks_exact(width).zip(c.chunks_exact(width))
                    })
            };
            match registry::representation_info(*dr).kind {
                RepresentationKind::Dense(dtype) if dtype.is_float() && !unwritten => {
                    SubjectComparison::Differs(measure_floating(
                        dtype,
                        pairs(dtype.bytes() as usize)
                            .map(|(d, c)| (element_bits(d), element_bits(c))),
                        activation,
                    ))
                }
                RepresentationKind::Dense(dtype) => {
                    let width = dtype.bytes() as usize;
                    SubjectComparison::Differs(measure_exact(
                        pairs(width).count() as u64,
                        pairs(width).filter(|(d, c)| d != c).count() as u64,
                    ))
                }
                // Packed storage has no elements to tell apart: its bytes.
                _ => SubjectComparison::Differs(measure_exact(
                    pairs(1).count() as u64,
                    pairs(1).filter(|(d, c)| d != c).count() as u64,
                )),
            }
        }
        (Observation::Scalar(d), Observation::Scalar(c)) => {
            match (scalar_bits(d), scalar_bits(c)) {
                (Some((dd, db)), Some((cd, cb))) if dd == cd => {
                    if db == cb {
                        SubjectComparison::Identical
                    } else {
                        SubjectComparison::Differs(measure_floating(
                            dd,
                            std::iter::once((db, cb)),
                            activation,
                        ))
                    }
                }
                (None, None) if std::mem::discriminant(d) == std::mem::discriminant(c) => {
                    if d == c {
                        SubjectComparison::Identical
                    } else {
                        SubjectComparison::Differs(measure_exact(1, 1))
                    }
                }
                _ => SubjectComparison::Mismatched,
            }
        }
        _ => SubjectComparison::Mismatched,
    }
}

/// Measure every subject of a candidate's call against the default's call
/// on the same inputs. A floating subject is measured against its stored
/// type and `activation`, the unit's activation type, when that is another.
pub(super) fn compare(
    default: &[Observation],
    candidate: &[Observation],
    subjects: &[Subject],
    activation: Option<DType>,
) -> Vec<SubjectMeasurement> {
    assert!(
        default.len() == subjects.len() && candidate.len() == subjects.len(),
        "an entry's calls publish its subjects"
    );
    default
        .iter()
        .zip(candidate)
        .zip(subjects)
        .map(|((default, candidate), subject)| SubjectMeasurement {
            subject: subject.name.clone(),
            comparison: compare_subject(default, candidate, subject.unwritten, activation),
        })
        .collect()
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    pub(in crate::native) fn tensor(dtype: DType, values: &[u32]) -> Observation {
        let width = dtype.bytes() as usize;
        Observation::Tensor {
            representation: registry::dense(dtype),
            shape: vec![values.len() as u64],
            bytes: super::super::observation::pages_of(
                &values
                    .iter()
                    .flat_map(|bits| bits.to_le_bytes()[..width].to_vec())
                    .collect::<Vec<_>>(),
            )
            .expect("the host maps an observation's pages"),
        }
    }

    pub(in crate::native) fn subject(name: &str) -> Subject {
        Subject {
            name: name.into(),
            unwritten: false,
        }
    }

    fn one(default: Observation, candidate: Observation, activation: Option<DType>) -> SubjectComparison {
        compare(&[default], &[candidate], &[subject("value")], activation)
            .remove(0)
            .comparison
    }

    fn differs(comparison: SubjectComparison) -> Difference {
        match comparison {
            SubjectComparison::Differs(difference) => difference,
            other => panic!("expected a difference, not {other:?}"),
        }
    }

    fn bf16(value: f32) -> u32 {
        value.to_bits() >> 16
    }

    /// Sums of many equal terms round; everything else here is exact.
    fn close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() <= 1e-12 * expected.abs(),
            "{actual} is not {expected}"
        );
    }

    /// A cell is half the gap below plus half the gap above, by the type's
    /// own adjacency: the two gaps differ at a binade boundary, zero's
    /// neighbours are the smallest subnormals, and the largest finite
    /// value's cell extends half its inward gap beyond it.
    #[test]
    fn rounding_cells_are_exact_at_zero_binade_boundaries_and_the_largest_value() {
        let width = |dtype, bits| RoundingCells::of(dtype).unwrap().width(bits).unwrap();
        // Inside a binade the cell is the spacing.
        assert_eq!(width(DType::F32, 1.5f32.to_bits()), 2f64.powi(-23));
        assert_eq!(width(DType::F16, 0x3e00), 2f64.powi(-10));
        assert_eq!(width(DType::BF16, bf16(1.5)), 2f64.powi(-7));
        // At a binade boundary: half the finer spacing below, half the
        // coarser above, on either sign.
        assert_eq!(
            width(DType::F32, 1f32.to_bits()),
            0.5 * 2f64.powi(-24) + 0.5 * 2f64.powi(-23)
        );
        assert_eq!(
            width(DType::F32, (-1f32).to_bits()),
            width(DType::F32, 1f32.to_bits())
        );
        assert_eq!(
            width(DType::F16, 0x3c00),
            0.5 * 2f64.powi(-11) + 0.5 * 2f64.powi(-10)
        );
        assert_eq!(
            width(DType::BF16, bf16(1.)),
            0.5 * 2f64.powi(-8) + 0.5 * 2f64.powi(-7)
        );
        assert_eq!(
            width(DType::BF16, bf16(-2.)),
            0.5 * 2f64.powi(-7) + 0.5 * 2f64.powi(-6)
        );
        // Either zero: half the smallest subnormal on each side.
        for (dtype, sign, subnormal) in [
            (DType::F32, 1u32 << 31, 2f64.powi(-149)),
            (DType::F16, 1 << 15, 2f64.powi(-24)),
            (DType::BF16, 1 << 15, 2f64.powi(-133)),
        ] {
            assert_eq!(width(dtype, 0), subnormal, "{dtype}");
            assert_eq!(width(dtype, sign), subnormal, "{dtype}");
            // The smallest subnormal sits between zero and the next.
            assert_eq!(width(dtype, 1), subnormal, "{dtype}");
        }
        // The largest finite value: its inward gap on both sides.
        assert_eq!(width(DType::F32, f32::MAX.to_bits()), 2f64.powi(104));
        assert_eq!(width(DType::F16, 0x7bff), 2f64.powi(5));
        assert_eq!(width(DType::BF16, 0x7f7f), 2f64.powi(120));
        assert_eq!(width(DType::BF16, 0xff7f), 2f64.powi(120));
        // Non-finite payloads have no cell; an integer type has no cells.
        for bits in [f32::INFINITY.to_bits(), f32::NAN.to_bits()] {
            assert_eq!(RoundingCells::of(DType::F32).unwrap().width(bits), None);
        }
        assert_eq!(RoundingCells::of(DType::BF16).unwrap().width(0x7f80), None);
        assert!(RoundingCells::of(DType::U32).is_err());
    }

    /// Constructed arrays with known powers: every element of a binade
    /// interior moved by a whole number of steps.
    #[test]
    fn powers_and_ratios_of_constructed_arrays_are_exact() {
        // 1024 bf16 elements at 1.5 (spacing 2^-7), each moved two steps.
        let default = vec![bf16(1.5); 1024];
        let candidate = vec![bf16(1.5) + 2; 1024];
        let difference = differs(one(
            tensor(DType::BF16, &default),
            tensor(DType::BF16, &candidate),
            None,
        ));
        let step = 2f64.powi(-7);
        assert_eq!(difference.stored, Some(DType::BF16));
        assert_eq!((difference.elements, difference.differing), (1024, 1024));
        assert_eq!(difference.error_power, 1024. * (2. * step) * (2. * step));
        assert_eq!(difference.reference_power, 1024. * 2.25);
        assert_eq!(difference.maximum_absolute_error, 2. * step);
        assert_eq!(difference.step_power.len(), 1);
        assert_eq!(difference.step_power[0].dtype, DType::BF16);
        close(difference.step_power[0].power, 1024. * step * step / 12.);
        // Two steps of error against a step's rounding deviation, 1/sqrt(12).
        close(difference.ratio(DType::BF16).unwrap(), 2. * 12f64.sqrt());
        assert_eq!(difference.ratio(DType::F32), None);

        // One element of 1024 moved one step: the others still count
        // toward the step power.
        let mut candidate = default.clone();
        candidate[731] += 1;
        let difference = differs(one(
            tensor(DType::BF16, &default),
            tensor(DType::BF16, &candidate),
            None,
        ));
        assert_eq!(difference.differing, 1);
        close(
            difference.ratio(DType::BF16).unwrap(),
            (12f64 / 1024.).sqrt(),
        );
    }

    /// A subject stored wider than the activation type is measured against
    /// both: its own step, and the step of the cell each default value
    /// rounds into in the activation type.
    #[test]
    fn a_wider_subject_is_measured_against_its_own_and_the_activation_step() {
        // f32 values at 1.5, each moved 2^-9: a quarter of a bf16 step,
        // 2^14 f32 steps.
        let default = vec![1.5f32.to_bits(); 256];
        let candidate = vec![(1.5f32 + 2f32.powi(-9)).to_bits(); 256];
        let difference = differs(one(
            tensor(DType::F32, &default),
            tensor(DType::F32, &candidate),
            Some(DType::BF16),
        ));
        assert_eq!(
            difference
                .step_power
                .iter()
                .map(|step| step.dtype)
                .collect::<Vec<_>>(),
            [DType::F32, DType::BF16]
        );
        close(
            difference.ratio(DType::F32).unwrap(),
            2f64.powi(14) * 12f64.sqrt(),
        );
        close(difference.ratio(DType::BF16).unwrap(), 0.25 * 12f64.sqrt());
        // A value between two bf16 values takes the cell it rounds into:
        // 1 + 2^-10 rounds to 1, whose cell straddles the binade boundary.
        let difference = differs(one(
            tensor(DType::F32, &[(1f32 + 2f32.powi(-10)).to_bits()]),
            tensor(DType::F32, &[1f32.to_bits()]),
            Some(DType::BF16),
        ));
        let cell = 0.5 * 2f64.powi(-8) + 0.5 * 2f64.powi(-7);
        assert_eq!(difference.step_power[1].power, cell * cell / 12.);
        // A finite f32 beyond the activation type's range has no cell there.
        let difference = differs(one(
            tensor(DType::F32, &[1e30f32.to_bits()]),
            tensor(DType::F32, &[1.0001e30f32.to_bits()]),
            Some(DType::F16),
        ));
        assert_eq!(difference.step_power[1].power, 0.);
        assert_eq!(difference.ratio(DType::F16), Some(f64::INFINITY));
    }

    /// The cell a wider value falls into is the one the type's own
    /// round-to-nearest-even conversion puts it in, ties and the overflow
    /// threshold included.
    #[test]
    fn the_cell_of_a_wider_value_is_the_one_it_rounds_into() {
        use seismic_lang::reference_math::conversion::quantize;
        for dtype in [DType::F16, DType::BF16] {
            let cells = RoundingCells::of(dtype).unwrap();
            let enumerated = Enumerated::of(&cells);
            let largest = *enumerated.magnitudes.last().unwrap();
            let gap = largest - enumerated.magnitudes[enumerated.magnitudes.len() - 2];
            let mut values = vec![0., f64::from(f32::MIN_POSITIVE), 1e-30, 0.1, 1., 3.1415926, 1e4];
            // Around a binade boundary, an exact tie on either side of an
            // even payload, and either side of the overflow threshold.
            for payload in [0x3bfe, 0x3bff, 0x3c00, 0x3c01, 1, 2] {
                let (below, above) = (
                    enumerated.magnitudes[payload],
                    enumerated.magnitudes[payload + 1],
                );
                values.extend([
                    below,
                    below + (above - below) / 4.,
                    below + (above - below) / 2.,
                    below + 3. * (above - below) / 4.,
                ]);
            }
            values.extend([largest, largest + gap / 4., largest + gap / 2., largest + gap]);
            for value in values {
                let rounded = quantize(dtype, value);
                let expected = exact_f64(dtype, rounded)
                    .is_finite()
                    .then_some(rounded as usize);
                assert_eq!(enumerated.containing(value), expected, "{dtype} {value:e}");
            }
        }
    }

    #[test]
    fn identical_bytes_need_no_arithmetic_and_a_zero_step_power_permits_no_error() {
        let values = [f32::NAN.to_bits(), 1f32.to_bits(), f32::INFINITY.to_bits()];
        assert_eq!(
            one(tensor(DType::F32, &values), tensor(DType::F32, &values), None),
            SubjectComparison::Identical
        );
        // The zeros differ in sign alone: no error, and nothing to divide.
        let difference = differs(one(
            tensor(DType::F32, &[0]),
            tensor(DType::F32, &[(-0f32).to_bits()]),
            None,
        ));
        assert_eq!((difference.differing, difference.error_power), (1, 0.));
        assert_eq!(difference.ratio(DType::F32), Some(0.));
        // No element finite in both: no step power, so any error is beyond it.
        let empty = Difference {
            error_power: 1e-30,
            ..difference
        };
        assert_eq!(
            Difference {
                step_power: vec![StepPower {
                    dtype: DType::F32,
                    power: 0.
                }],
                ..empty
            }
            .ratio(DType::F32),
            Some(f64::INFINITY)
        );
    }

    #[test]
    fn non_finite_integer_and_unwritten_differences_are_counted_not_scaled() {
        // A non-finite value on either side must be the same bits.
        let default = [1f32.to_bits(), f32::INFINITY.to_bits(), f32::NAN.to_bits(), 2f32.to_bits()];
        let candidate = [
            1f32.to_bits(),
            f32::MAX.to_bits(),
            f32::NAN.to_bits() | 1,
            f32::NEG_INFINITY.to_bits(),
        ];
        let difference = differs(one(
            tensor(DType::F32, &default),
            tensor(DType::F32, &candidate),
            None,
        ));
        assert_eq!(difference.non_finite_mismatches, 3);
        assert_eq!((difference.differing, difference.error_power), (3, 0.));
        // Integer and Boolean subjects have no scale.
        for dtype in [DType::U32, DType::I32, DType::Bool] {
            let difference = differs(one(
                tensor(dtype, &[1, 0, 1]),
                tensor(dtype, &[1, 1, 0]),
                Some(DType::BF16),
            ));
            assert_eq!(difference.stored, None, "{dtype}");
            assert_eq!(
                (difference.elements, difference.differing, difference.exact_mismatches),
                (3, 2, 2),
                "{dtype}"
            );
            assert!(difference.step_power.is_empty());
        }
        // Rows the entry may not write are exact whatever they store.
        let unwritten = Subject {
            name: "i0:unwritten".into(),
            unwritten: true,
        };
        let measured = compare(
            &[tensor(DType::F32, &[1f32.to_bits()])],
            &[tensor(DType::F32, &[1.0000001f32.to_bits()])],
            &[unwritten],
            None,
        );
        assert_eq!(differs(measured[0].comparison.clone()).exact_mismatches, 1);
        // Scalars: floating ones are one-element subjects, others exact.
        let difference = differs(
            compare(
                &[Observation::Scalar(ArgumentValue::F32(1.5))],
                &[Observation::Scalar(ArgumentValue::F32(1.5 + f32::EPSILON))],
                &[subject("r")],
                None,
            )
            .remove(0)
            .comparison,
        );
        close(difference.ratio(DType::F32).unwrap(), 12f64.sqrt());
        let discrete = compare(
            &[Observation::Scalar(ArgumentValue::U32(1))],
            &[Observation::Scalar(ArgumentValue::U32(2))],
            &[subject("r")],
            None,
        );
        assert_eq!(differs(discrete[0].comparison.clone()).exact_mismatches, 1);
    }

    #[test]
    fn another_shape_representation_or_kind_is_a_mismatch() {
        let default = tensor(DType::F32, &[0, 0]);
        let mut reshaped = default.clone();
        if let Observation::Tensor { shape, .. } = &mut reshaped {
            *shape = vec![1, 2];
        }
        for candidate in [
            reshaped,
            tensor(DType::U32, &[0, 0]),
            tensor(DType::F32, &[0]),
            Observation::Scalar(ArgumentValue::F32(0.)),
        ] {
            assert_eq!(
                one(default.clone(), candidate, None),
                SubjectComparison::Mismatched
            );
        }
        assert_eq!(
            compare(
                &[Observation::Scalar(ArgumentValue::F32(0.))],
                &[Observation::Scalar(ArgumentValue::U32(0))],
                &[subject("r")],
                None,
            )[0]
            .comparison,
            SubjectComparison::Mismatched
        );
    }
}
