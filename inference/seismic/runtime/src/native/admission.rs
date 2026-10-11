//! Whether a candidate's measurements against the default admit it: the
//! caller's requirement applied to what [`super::comparison`] measured.
//! Measurements are facts of the candidate and the default; admission is a
//! pure function of them and the requirement, so the same stored
//! measurements answer for any requirement.
//!
//! A candidate is admitted when, at every point and rotation it was
//! observed at, every subject keeps the default's representation and shape,
//! every integer, Boolean and packed subject and every unwritten row is the
//! default's bit for bit, every value not finite in either is the same
//! bits, and every floating subject's error is within α rounding steps of
//! its governing type: `sqrt(error_power / step_power) <= α`.
use super::comparison::{Difference, SubjectComparison, SubjectMeasurement};
use super::search::PointKey;
use super::tune::{Configuration, Exclusion, Outcome, TuningDisposition, TuningResult};
use seismic_lang::types::DType;
use serde::{Deserialize, Serialize};

/// What a unit's candidates are held to against its native default.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum NumericalRequirement {
    /// Every subject is the default's byte for byte.
    BitExact,
    /// Every floating subject's error is within `alpha` rounding steps.
    /// `activation` is the floating type the unit's activations are stored
    /// as, when it binds one: subjects are measured against its step beside
    /// their own.
    RoundingSteps {
        alpha: f64,
        activation: Option<DType>,
    },
}

impl NumericalRequirement {
    /// The activation type subjects are measured against beside their own.
    pub fn activation(&self) -> Option<DType> {
        match self {
            Self::BitExact => None,
            Self::RoundingSteps { activation, .. } => *activation,
        }
    }
}

/// Which floating type's rounding step a subject's error is divided by.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GoverningStep {
    /// The unit's activation type where it binds one, else the subject's
    /// own stored type. A subject stored wider than the activations (an
    /// `f32` result or state beside `bf16` activations) is then held to the
    /// precision the model's activations carry.
    Activation,
    /// The subject's own stored type, always. A subject stored wider than
    /// the activations then admits only error at its own resolution.
    Subject,
}

/// The step that governs admission. Both ratios of every subject are
/// measured and recorded whatever this is.
pub const GOVERNING_STEP: GoverningStep = GoverningStep::Activation;

impl GoverningStep {
    fn of(self, stored: DType, activation: Option<DType>) -> DType {
        match self {
            Self::Activation => activation.unwrap_or(stored),
            Self::Subject => stored,
        }
    }
}

/// One candidate's measurements at one point: every subject of every
/// rotation. `key` is what determines them there ([`PointKey`]): candidates
/// with the same key at a point run the same work and share the observation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NumericalObservation {
    pub point: String,
    pub key: PointKey,
    pub rotations: Vec<Vec<SubjectMeasurement>>,
}

/// The constraint a subject violates.
#[derive(Clone, Debug, PartialEq)]
pub enum Constraint {
    /// Another representation, shape or kind of value.
    Layout,
    /// `differing` elements of a subject that admits no difference.
    Exact { differing: u64 },
    /// `mismatches` values not finite in either that are not the same bits.
    NonFinite { mismatches: u64 },
    /// An error of `ratio` rounding steps of `step`, beyond `alpha`.
    Ratio { ratio: f64, alpha: f64, step: DType },
}

/// The first constraint a candidate violates.
#[derive(Clone, Debug, PartialEq)]
pub struct Violation {
    pub point: String,
    pub rotation: usize,
    pub subject: String,
    pub constraint: Constraint,
}

impl std::fmt::Display for Violation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "subject {} at point {} rotation {}: ",
            self.subject, self.point, self.rotation
        )?;
        match &self.constraint {
            Constraint::Layout => f.write_str("representation, shape or kind differs"),
            Constraint::Exact { differing } => {
                write!(f, "{differing} elements differ where none may")
            }
            Constraint::NonFinite { mismatches } => {
                write!(f, "{mismatches} non-finite values are not the default's bits")
            }
            Constraint::Ratio { ratio, alpha, step } => write!(
                f,
                "error ratio {ratio} exceeds alpha {alpha} ({step} rounding steps)"
            ),
        }
    }
}

impl Violation {
    /// The exclusion of a candidate this violation rejects.
    pub fn exclusion(&self) -> Exclusion {
        Exclusion::Validation {
            point: self.point.clone(),
            detail: self.to_string(),
        }
    }
}

/// The error of a differing floating subject in the rounding steps that
/// govern its admission under `requirement`; none for a subject with no
/// floating scale.
pub fn governing_ratio(difference: &Difference, requirement: &NumericalRequirement) -> Option<f64> {
    governing_ratio_under(difference, requirement.activation(), GOVERNING_STEP).map(|(ratio, _)| ratio)
}

fn governing_ratio_under(
    difference: &Difference,
    activation: Option<DType>,
    governing: GoverningStep,
) -> Option<(f64, DType)> {
    let step = governing.of(difference.stored?, activation);
    let ratio = difference
        .ratio(step)
        .expect("a floating subject is measured against its own and the activation type");
    Some((ratio, step))
}

fn violated(
    measurement: &SubjectMeasurement,
    requirement: &NumericalRequirement,
    governing: GoverningStep,
) -> Option<Constraint> {
    let difference = match &measurement.comparison {
        SubjectComparison::Identical => return None,
        SubjectComparison::Mismatched => return Some(Constraint::Layout),
        SubjectComparison::Differs(difference) => difference,
    };
    if difference.exact_mismatches > 0 {
        return Some(Constraint::Exact {
            differing: difference.exact_mismatches,
        });
    }
    if difference.non_finite_mismatches > 0 {
        return Some(Constraint::NonFinite {
            mismatches: difference.non_finite_mismatches,
        });
    }
    match requirement {
        NumericalRequirement::BitExact => Some(Constraint::Exact {
            differing: difference.differing,
        }),
        NumericalRequirement::RoundingSteps { alpha, activation } => {
            let (ratio, step) = governing_ratio_under(difference, *activation, governing)?;
            (!(ratio <= *alpha)).then_some(Constraint::Ratio {
                ratio,
                alpha: *alpha,
                step,
            })
        }
    }
}

/// Admit `observation` under `requirement`, or name the first constraint it
/// violates.
pub fn admit(
    observation: &NumericalObservation,
    requirement: &NumericalRequirement,
) -> Result<(), Violation> {
    admit_under(observation, requirement, GOVERNING_STEP)
}

fn admit_under(
    observation: &NumericalObservation,
    requirement: &NumericalRequirement,
    governing: GoverningStep,
) -> Result<(), Violation> {
    for (rotation, subjects) in observation.rotations.iter().enumerate() {
        for measurement in subjects {
            if let Some(constraint) = violated(measurement, requirement, governing) {
                return Err(Violation {
                    point: observation.point.clone(),
                    rotation,
                    subject: measurement.subject.clone(),
                    constraint,
                });
            }
        }
    }
    Ok(())
}

/// How a stored record was served under a requirement other than the one
/// it was searched under.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Reselection {
    /// What the search held its candidates to.
    pub searched: NumericalRequirement,
    /// Candidates the search rejected whose recorded measurements this
    /// requirement admits. The search took no timing, confirmation or
    /// coverage of them: none can be served, and none is tried again at
    /// this load.
    pub unserved: Vec<Configuration>,
}

/// Serve a stored record under `requirement`: of the choices its search
/// served in turn (its default, then each promotion, every one timed,
/// confirmed faster than the one before and covered), the last whose
/// measurements `requirement` admits. Nothing is formed or measured.
pub fn reselect(result: &mut TuningResult, requirement: &NumericalRequirement) {
    if result.requirement == *requirement {
        return;
    }
    let admitted = |configuration: &Configuration| {
        result
            .configurations
            .iter()
            .find(|record| record.configuration == *configuration)
            .is_none_or(|record| {
                record
                    .observations
                    .iter()
                    .all(|index| admit(&result.observations[*index], requirement).is_ok())
            })
    };
    let default = result
        .served
        .first()
        .expect("a unit serves its canonical default first")
        .clone();
    let chosen = result
        .served
        .iter()
        .rev()
        .find(|configuration| admitted(configuration))
        .unwrap_or(&default)
        .clone();
    let unserved = result
        .configurations
        .iter()
        .filter(|record| {
            matches!(
                record.outcome,
                Outcome::Excluded(Exclusion::Validation { .. })
            ) && admitted(&record.configuration)
        })
        .map(|record| record.configuration.clone())
        .collect();
    if chosen != result.overall {
        result.disposition = if chosen == default {
            TuningDisposition::CanonicalFallback {
                artifact: result
                    .configurations
                    .iter()
                    .find(|record| record.configuration == chosen)
                    .and_then(|record| record.artifact.clone())
                    .expect("a searched unit records its default's artifact"),
            }
        } else {
            TuningDisposition::QualifiedCandidate
        };
        result.overall = chosen;
    }
    result.reselection = Some(Reselection {
        searched: std::mem::replace(&mut result.requirement, requirement.clone()),
        unserved,
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native::comparison::StepPower;

    /// A floating subject stored as `stored` whose error is `own` steps of
    /// its own type and `activation` steps of bf16.
    fn floating(stored: DType, own: f64, activation: f64) -> SubjectMeasurement {
        SubjectMeasurement {
            subject: "r0".into(),
            comparison: SubjectComparison::Differs(Difference {
                stored: Some(stored),
                elements: 4,
                differing: 4,
                exact_mismatches: 0,
                non_finite_mismatches: 0,
                error_power: 1.,
                reference_power: 4.,
                maximum_absolute_error: 0.5,
                step_power: std::iter::once((stored, own))
                    .chain((stored != DType::BF16).then_some((DType::BF16, activation)))
                    .map(|(dtype, ratio)| StepPower {
                        dtype,
                        power: 1. / (ratio * ratio),
                    })
                    .collect(),
            }),
        }
    }

    fn observation(subjects: Vec<SubjectMeasurement>) -> NumericalObservation {
        NumericalObservation {
            point: "m512".into(),
            key: PointKey {
                launches: vec![0],
                values: [("PACK".to_owned(), 1)].into(),
            },
            rotations: vec![Vec::new(), subjects],
        }
    }

    fn steps(alpha: f64) -> NumericalRequirement {
        NumericalRequirement::RoundingSteps {
            alpha,
            activation: Some(DType::BF16),
        }
    }

    /// The verdict turns exactly at α, and its message states the subject,
    /// the point, the ratio and α.
    #[test]
    fn a_ratio_just_under_alpha_is_admitted_and_just_over_is_not() {
        let candidate = observation(vec![floating(DType::BF16, 4., 4.)]);
        assert_eq!(admit(&candidate, &steps(4.)), Ok(()));
        assert_eq!(admit(&candidate, &steps(4.000001)), Ok(()));
        let violation = admit(&candidate, &steps(3.999999)).unwrap_err();
        assert_eq!(
            violation,
            Violation {
                point: "m512".into(),
                rotation: 1,
                subject: "r0".into(),
                constraint: Constraint::Ratio {
                    ratio: 4.,
                    alpha: 3.999999,
                    step: DType::BF16
                },
            }
        );
        assert_eq!(
            violation.to_string(),
            "subject r0 at point m512 rotation 1: error ratio 4 exceeds alpha 3.999999 (bf16 rounding steps)"
        );
        assert!(matches!(
            violation.exclusion(),
            Exclusion::Validation { point, .. } if point == "m512"
        ));
        // A ratio that is not a number is never within α.
        let mut undefined = candidate;
        if let SubjectComparison::Differs(difference) = &mut undefined.rotations[1][0].comparison {
            difference.error_power = f64::NAN;
        }
        assert!(admit(&undefined, &steps(f64::MAX)).is_err());
    }

    /// The governing choice decides only subjects stored wider than the
    /// activation type: there the activation step admits what the
    /// subject's own step does not.
    #[test]
    fn the_governing_step_changes_the_verdict_only_for_a_subject_wider_than_the_activations() {
        let requirement = steps(8.);
        let verdict = |subject: SubjectMeasurement, governing| {
            admit_under(&observation(vec![subject]), &requirement, governing).is_ok()
        };
        // An f32 subject beside bf16 activations, a quarter of a bf16 step
        // off: 16384 of its own steps.
        let wider = || floating(DType::F32, 16384., 0.25);
        assert!(verdict(wider(), GoverningStep::Activation));
        assert!(!verdict(wider(), GoverningStep::Subject));
        // A subject stored as the activation type: one step, either way.
        for (ratio, admitted) in [(7., true), (9., false)] {
            for governing in [GoverningStep::Activation, GoverningStep::Subject] {
                assert_eq!(
                    verdict(floating(DType::BF16, ratio, ratio), governing),
                    admitted
                );
            }
        }
        // A unit that binds no activation type: the subject's own step.
        let unbound = NumericalRequirement::RoundingSteps {
            alpha: 8.,
            activation: None,
        };
        let own = SubjectMeasurement {
            subject: "r0".into(),
            comparison: SubjectComparison::Differs(Difference {
                stored: Some(DType::F32),
                elements: 1,
                differing: 1,
                exact_mismatches: 0,
                non_finite_mismatches: 0,
                error_power: 4.,
                reference_power: 1.,
                maximum_absolute_error: 2.,
                step_power: vec![StepPower {
                    dtype: DType::F32,
                    power: 1.,
                }],
            }),
        };
        assert_eq!(GOVERNING_STEP, GoverningStep::Activation);
        assert_eq!(admit(&observation(vec![own.clone()]), &unbound), Ok(()));
        let SubjectComparison::Differs(difference) = &own.comparison else {
            unreachable!()
        };
        assert_eq!(governing_ratio(difference, &unbound), Some(2.));
    }

    /// A record searched under α = 8 whose search served its default, then
    /// `ROWS` 2 (two steps from the default), then `ROWS` 4 (five steps),
    /// measured `ROWS` 8 (one step) without ever promoting it, and
    /// rejected `ROWS` 16 at its first point (twelve steps).
    fn searched() -> TuningResult {
        use crate::native::tune::{
            ConfigurationRecord, PointMeasurement, PointRecord, TuningMethod, TuningTime,
        };
        let configuration = |rows: u64| Configuration {
            statics: Default::default(),
            params: [("ROWS".to_owned(), rows)].into(),
            launches: Vec::new(),
        };
        let mut observations = Vec::new();
        let mut record = |rows: u64, ratios: &[f64], outcome: Outcome| ConfigurationRecord {
            formation: None,
            configuration: configuration(rows),
            artifact: Some(format!("artifact-{rows}")),
            observations: ["m1", "m512"]
                .iter()
                .zip(ratios)
                .map(|(point, ratio)| {
                    observations.push(NumericalObservation {
                        point: (*point).into(),
                        key: PointKey {
                            launches: vec![0],
                            values: [("ROWS".to_owned(), rows)].into(),
                        },
                        rotations: vec![vec![floating(DType::BF16, *ratio, *ratio)]],
                    });
                    observations.len() - 1
                })
                .collect(),
            outcome,
        };
        let timed = |confirmed: bool| {
            let measured = vec![PointMeasurement {
                point: "m1".into(),
                key: PointKey {
                    launches: vec![0],
                    values: Default::default(),
                },
                median_seconds: 1e-3,
                deviation_seconds: 0.,
                samples: vec![1e-3],
                repetitions: 1,
                rotation_bytes: 0,
            }];
            Outcome::Measured {
                confirmed: if confirmed { measured.clone() } else { Vec::new() },
                points: measured,
            }
        };
        let configurations = vec![
            record(1, &[], timed(true)),
            record(2, &[2., 2.], timed(true)),
            record(4, &[5., 3.], timed(true)),
            record(8, &[1.], timed(false)),
            record(
                16,
                &[12.],
                Outcome::Excluded(Exclusion::Validation {
                    point: "m1".into(),
                    detail: "beyond 8".into(),
                }),
            ),
        ];
        TuningResult {
            disposition: TuningDisposition::QualifiedCandidate,
            tuning_identity: "device".into(),
            entry: "dense".into(),
            backend: "metal".into(),
            points: ["m1", "m512"]
                .map(|label| PointRecord {
                    label: label.into(),
                    weight: 1.,
                    class: None,
                })
                .into(),
            requirement: steps(8.),
            measurement_identity: "measurements".into(),
            observations,
            served: [1, 2, 4].map(configuration).into(),
            reselection: None,
            implementation_identity: "implementation".into(),
            parameters: Vec::new(),
            configurations,
            overall: configuration(4),
            method: TuningMethod::Census,
            time: TuningTime::default(),
            coverage: None,
        }
    }

    /// A stored record is served under another α from its measurements
    /// alone: the last choice its search served that the α admits. A
    /// candidate the search measured or rejected without promoting it is
    /// never served, whatever the α admits of it; the ones it rejected that
    /// are now admissible are named.
    #[test]
    fn reselection_serves_the_last_promoted_choice_the_requirement_admits() {
        let rows = |result: &TuningResult| result.overall.params["ROWS"];
        let reselected = |alpha: f64| {
            let mut result = searched();
            reselect(&mut result, &steps(alpha));
            result
        };
        // The α it was searched under: the record as it stands.
        assert_eq!(reselected(8.), searched());
        // Stricter than the last promotion's five steps: the one before.
        let stricter = reselected(4.);
        assert_eq!(rows(&stricter), 2);
        assert_eq!(stricter.disposition, TuningDisposition::QualifiedCandidate);
        assert_eq!(stricter.requirement, steps(4.));
        assert_eq!(
            stricter.reselection,
            Some(Reselection {
                searched: steps(8.),
                unserved: Vec::new()
            })
        );
        // `ROWS` 8 is within every α here and was timed, but its search
        // never promoted it: stricter than both promotions, the default.
        let strictest = reselected(1.5);
        assert_eq!(rows(&strictest), 1);
        assert_eq!(
            strictest.disposition,
            TuningDisposition::CanonicalFallback {
                artifact: "artifact-1".into()
            }
        );
        // Looser: the choice stands, and the candidate rejected at twelve
        // steps is admissible now and left unserved.
        let looser = reselected(16.);
        assert_eq!(rows(&looser), 4);
        let unserved = &looser.reselection.as_ref().unwrap().unserved;
        assert_eq!(unserved.len(), 1);
        assert_eq!(unserved[0].params["ROWS"], 16);
        // Bit-exactness admits none of them.
        let mut exact = searched();
        reselect(&mut exact, &NumericalRequirement::BitExact);
        assert_eq!(rows(&exact), 1);
        // Nothing but the choice, its disposition and the requirement
        // changes: every measurement and timing is the stored one.
        let stored = searched();
        for result in [&stricter, &strictest, &looser, &exact] {
            assert_eq!(result.observations, stored.observations);
            assert_eq!(result.configurations, stored.configurations);
            assert_eq!(result.served, stored.served);
        }
    }

    /// The constraints outside the ratio hold under every α; bit-exactness
    /// admits identical subjects alone.
    #[test]
    fn hard_constraints_hold_whatever_alpha_is() {
        let loose = steps(f64::MAX);
        let exact = |differing| Difference {
            stored: None,
            elements: 8,
            differing,
            exact_mismatches: differing,
            non_finite_mismatches: 0,
            error_power: 0.,
            reference_power: 0.,
            maximum_absolute_error: 0.,
            step_power: Vec::new(),
        };
        let subject = |comparison| SubjectMeasurement {
            subject: "i2".into(),
            comparison,
        };
        let constraint = |comparison, requirement: &NumericalRequirement| {
            admit(&observation(vec![subject(comparison)]), requirement)
                .err()
                .map(|violation| violation.constraint)
        };
        assert_eq!(constraint(SubjectComparison::Identical, &loose), None);
        assert_eq!(
            constraint(SubjectComparison::Identical, &NumericalRequirement::BitExact),
            None
        );
        assert_eq!(
            constraint(SubjectComparison::Mismatched, &loose),
            Some(Constraint::Layout)
        );
        assert_eq!(
            constraint(SubjectComparison::Differs(exact(3)), &loose),
            Some(Constraint::Exact { differing: 3 })
        );
        let mut non_finite = floating(DType::BF16, 0.5, 0.5);
        if let SubjectComparison::Differs(difference) = &mut non_finite.comparison {
            difference.non_finite_mismatches = 2;
        }
        assert_eq!(
            constraint(non_finite.comparison, &loose),
            Some(Constraint::NonFinite { mismatches: 2 })
        );
        // Within α, yet not the default's bits.
        assert_eq!(
            constraint(
                floating(DType::BF16, 0.5, 0.5).comparison,
                &NumericalRequirement::BitExact
            ),
            Some(Constraint::Exact { differing: 4 })
        );
    }
}
