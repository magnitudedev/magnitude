//! One comparison of the completed reference invocation and native observations.
//! Geometry is checked before values, independently of numerical association.
use super::{compare_element, ElementComparison};
use seismic_lang::entry::TensorAccess;
use seismic_lang::expr::SymbolValue;
use seismic_lang::failure::SourceTermination;
use seismic_lang::ids::RepresentationId;
use seismic_lang::interp::{OracleOutcome, OutcomeValue, TensorReader};
use seismic_lang::precision::PrecisionPolicy;
use seismic_lang::{registry, types::DType};

#[derive(Debug)]
pub struct ObservedInvocation {
    pub termination: SourceTermination<Vec<ObservedResult>>,
    pub inputs: Vec<ObservedInput>,
}
#[derive(Debug)]
pub struct ObservedResult {
    pub path: Vec<u32>,
    pub value: ObservedValue,
}
#[derive(Debug)]
pub struct ObservedInput {
    pub ordinal: usize,
    pub tensor: ObservedTensor,
}
#[derive(Debug)]
pub struct ObservedTensor {
    pub representation: RepresentationId,
    pub shape: Vec<usize>,
    pub bytes: Vec<u8>,
}
#[derive(Debug)]
pub enum ObservedValue {
    Scalar(SymbolValue),
    Index(SymbolValue),
    Range {
        start: SymbolValue,
        end: SymbolValue,
    },
    Tensor(ObservedTensor),
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ComparisonSubject {
    Result(Vec<u32>),
    Input(usize),
    Invocation,
}
#[derive(Clone, Debug, PartialEq)]
pub struct Difference {
    pub subject: ComparisonSubject,
    pub coordinate: Option<Vec<usize>>,
    pub message: String,
    pub metrics: Option<ElementComparison>,
}
#[derive(Clone, Debug, PartialEq)]
pub enum Comparison {
    Match,
    Mismatch(Difference),
    Unsupported(String),
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ComparisonError {
    Resource(String),
    Contract(String),
}

/// The caller debits the existing observation allowance, including work spent
/// by the interpreter and earlier trials. No independent budget is created here.
pub fn compare_outcome(
    reference: &OracleOutcome,
    actual: &ObservedInvocation,
    policy: &PrecisionPolicy,
    allowance: &mut impl FnMut(u64) -> Result<(), ComparisonError>,
) -> Result<Comparison, ComparisonError> {
    allowance(1)?;
    let results = match &actual.termination {
        SourceTermination::Returned(results) => results.as_slice(),
        SourceTermination::Failed(_) => &[],
    };
    let both_returned = matches!(
        (reference.termination(), &actual.termination),
        (
            SourceTermination::Returned(_),
            SourceTermination::Returned(_)
        )
    );
    if (both_returned && reference.results().count() != results.len())
        || reference.inputs().count() != actual.inputs.len()
    {
        return Ok(mismatch(
            ComparisonSubject::Invocation,
            "result or tensor input count differs",
        ));
    }
    // A numerical ambiguity may never hide a later structural violation.
    let mut unsupported = None;
    for (expected, observed) in reference.results().zip(results) {
        allowance(1)?;
        let subject = || ComparisonSubject::Result(expected.path().to_vec());
        if expected.path() != observed.path {
            return Ok(mismatch(subject(), "result path differs"));
        }
        let geometry = match (expected.value(), &observed.value) {
            (OutcomeValue::Scalar(expected), ObservedValue::Scalar(value))
                if scalar(value.clone())
                    .is_some_and(|(actual_dtype, _)| actual_dtype == expected.dtype()) =>
            {
                None
            }
            (OutcomeValue::Index(_), ObservedValue::Index(SymbolValue::Nat(_))) => None,
            (
                OutcomeValue::Range(_, _),
                ObservedValue::Range {
                    start: SymbolValue::Nat(_),
                    end: SymbolValue::Nat(_),
                },
            ) => None,
            (OutcomeValue::Tensor(tensor), ObservedValue::Tensor(observed)) => {
                tensor_geometry(&tensor, observed)?
            }
            _ => Some("result kind or scalar dtype differs".into()),
        };
        if let Some(reason) = geometry {
            return Ok(mismatch(subject(), &reason));
        }
    }
    for (expected, observed) in reference.inputs().zip(&actual.inputs) {
        allowance(1)?;
        if expected.ordinal() != observed.ordinal {
            return Ok(mismatch(
                ComparisonSubject::Input(expected.ordinal()),
                "tensor input ordinal differs",
            ));
        }
        if let Some(reason) = tensor_geometry(&expected.tensor(), &observed.tensor)? {
            return Ok(mismatch(
                ComparisonSubject::Input(expected.ordinal()),
                &reason,
            ));
        }
    }
    // Shared input identity is a storage contract, independently of floating
    // tolerance and of any allowed numerical associations in the invocation.
    for (expected, observed) in reference.inputs().zip(&actual.inputs) {
        let tensor = expected.tensor();
        if expected.access() == TensorAccess::Shared {
            allowance(tensor.element_count() as u64)?;
            let bytes = tensor
                .canonical_bytes()
                .map_err(ComparisonError::Contract)?
                .ok_or_else(|| {
                    ComparisonError::Contract(
                        "shared input does not expose its complete canonical backing".into(),
                    )
                })?;
            for (left, right) in bytes.chunks(1024).zip(observed.tensor.bytes.chunks(1024)) {
                allowance(left.len() as u64)?;
                if left != right {
                    return Ok(mismatch(
                        ComparisonSubject::Input(expected.ordinal()),
                        "read-only input bytes changed",
                    ));
                }
            }
        } else if !matches!(
            registry::representation_info(tensor.representation()).kind,
            registry::RepresentationKind::Dense(_)
        ) {
            unsupported.get_or_insert_with(|| {
                "writable encoded state comparison is unavailable".to_owned()
            });
        }
    }
    let parallel_prefix = reference
        .relation()
        .is_some_and(|relation| !relation.parallel_regions().is_empty())
        && (!matches!(reference.termination(), SourceTermination::Returned(_))
            || !matches!(actual.termination, SourceTermination::Returned(_)));
    let mut difference = match (reference.termination(), &actual.termination) {
        (SourceTermination::Returned(_), SourceTermination::Returned(_)) => None,
        (SourceTermination::Failed(expected), SourceTermination::Failed(observed)) => {
            if expected.cause != observed.cause {
                Some(Difference {
                    subject: ComparisonSubject::Invocation,
                    coordinate: None,
                    message: "source failure cause differs".into(),
                    metrics: None,
                })
            } else if expected.event.body() == observed.event.body()
                && expected.event != observed.event
            {
                Some(Difference {
                    subject: ComparisonSubject::Invocation,
                    coordinate: None,
                    message: "source failure event differs within the same checked body".into(),
                    metrics: None,
                })
            } else {
                if expected.event != observed.event {
                    unsupported.get_or_insert_with(|| {
                        "source failure event correspondence is not established".into()
                    });
                }
                None
            }
        }
        _ => Some(Difference {
            subject: ComparisonSubject::Invocation,
            coordinate: None,
            message: "source termination differs".into(),
            metrics: None,
        }),
    };
    for (expected, observed) in reference.results().zip(results) {
        let name = output_name(expected.path());
        let subject = || ComparisonSubject::Result(expected.path().to_vec());
        let result = match (expected.value(), &observed.value) {
            (OutcomeValue::Scalar(value), ObservedValue::Scalar(actual)) => {
                allowance(1)?;
                let actual = scalar(actual.clone()).expect("geometry checked").1;
                element_difference(policy, &name, value.dtype(), value.to_f64(), actual).map(
                    |(message, metrics)| Difference {
                        subject: subject(),
                        coordinate: None,
                        message,
                        metrics: Some(metrics),
                    },
                )
            }
            (OutcomeValue::Index(value), ObservedValue::Index(SymbolValue::Nat(actual))) => {
                allowance(1)?;
                (value != actual).then(|| Difference {
                    subject: subject(),
                    coordinate: None,
                    message: format!("expected index {value}, actual {actual}"),
                    metrics: None,
                })
            }
            (
                OutcomeValue::Range(start, end),
                ObservedValue::Range {
                    start: SymbolValue::Nat(actual_start),
                    end: SymbolValue::Nat(actual_end),
                },
            ) => {
                allowance(2)?;
                (start != actual_start || end != actual_end).then(|| Difference {
                    subject: subject(),
                    coordinate: None,
                    message: format!(
                        "expected range {start}..{end}, actual {actual_start}..{actual_end}"
                    ),
                    metrics: None,
                })
            }
            (OutcomeValue::Tensor(tensor), ObservedValue::Tensor(actual)) => {
                match tensor_difference(&tensor, actual, policy, &name, allowance)? {
                    TensorComparison::Match => None,
                    TensorComparison::Difference(index, message, metrics) => Some(Difference {
                        subject: subject(),
                        coordinate: Some(coordinate(tensor.shape(), index)),
                        message,
                        metrics: Some(metrics),
                    }),
                    TensorComparison::Unsupported(reason) => {
                        unsupported.get_or_insert(reason);
                        None
                    }
                }
            }
            _ => unreachable!("geometry checked"),
        };
        if difference.is_none() {
            difference = result;
        }
    }
    for (expected, observed) in reference.inputs().zip(&actual.inputs) {
        let tensor = expected.tensor();
        if expected.access() == TensorAccess::Shared
            || !matches!(
                registry::representation_info(tensor.representation()).kind,
                registry::RepresentationKind::Dense(_)
            )
        {
            continue;
        }
        match tensor_difference(&tensor, &observed.tensor, policy, "value", allowance)? {
            TensorComparison::Match => {}
            TensorComparison::Difference(index, message, metrics) => {
                difference.get_or_insert_with(|| Difference {
                    subject: ComparisonSubject::Input(expected.ordinal()),
                    coordinate: Some(coordinate(tensor.shape(), index)),
                    message,
                    metrics: Some(metrics),
                });
            }
            TensorComparison::Unsupported(reason) => {
                unsupported.get_or_insert(reason);
            }
        }
    }
    if let Some(difference) = difference {
        let numerical_freedom = difference.metrics.is_some()
            && reference
                .relation()
                .is_some_and(|relation| !relation.associations().is_empty());
        if parallel_prefix || numerical_freedom {
            return Ok(Comparison::Unsupported(format!("{:?}: disagreement with an allowed representative cannot decide relation membership", difference.subject)));
        }
        return Ok(Comparison::Mismatch(difference));
    }
    Ok(unsupported.map_or(Comparison::Match, Comparison::Unsupported))
}
fn mismatch(subject: ComparisonSubject, message: &str) -> Comparison {
    Comparison::Mismatch(Difference {
        subject,
        coordinate: None,
        message: message.into(),
        metrics: None,
    })
}
fn tensor_geometry(
    reference: &TensorReader<'_>,
    actual: &ObservedTensor,
) -> Result<Option<String>, ComparisonError> {
    if reference.shape() != actual.shape || reference.representation() != actual.representation {
        return Ok(Some("tensor shape or representation differs".into()));
    }
    let bytes = reference
        .canonical_byte_len()
        .map_err(ComparisonError::Contract)?;
    if bytes != actual.bytes.len() {
        return Ok(Some(format!(
            "expected {bytes} tensor bytes, actual {}",
            actual.bytes.len()
        )));
    }
    Ok(None)
}
enum TensorComparison {
    Match,
    Difference(usize, String, ElementComparison),
    Unsupported(String),
}
fn tensor_difference(
    reference: &TensorReader<'_>,
    actual: &ObservedTensor,
    policy: &PrecisionPolicy,
    name: &str,
    allowance: &mut impl FnMut(u64) -> Result<(), ComparisonError>,
) -> Result<TensorComparison, ComparisonError> {
    let registry::RepresentationKind::Dense(dtype) =
        registry::representation_info(reference.representation()).kind
    else {
        return Ok(TensorComparison::Unsupported(
            "encoded numerical comparison is unavailable".into(),
        ));
    };
    let mut first = None;
    for (index, bytes) in actual
        .bytes
        .chunks_exact(dtype.bytes() as usize)
        .enumerate()
    {
        allowance(1)?;
        let expected = reference.read(index).map_err(ComparisonError::Contract)?;
        let value = decode(dtype, bytes);
        if let Some((message, metrics)) = element_difference(policy, name, dtype, expected, value) {
            if first.is_none() {
                first = Some(TensorComparison::Difference(index, message, metrics));
            }
        }
    }
    Ok(first.unwrap_or(TensorComparison::Match))
}
fn element_difference(
    policy: &PrecisionPolicy,
    name: &str,
    dtype: DType,
    expected: f64,
    actual: f64,
) -> Option<(String, ElementComparison)> {
    let metrics = compare_element(policy, name, dtype, expected, actual);
    (!metrics.accepted).then(|| (format!("expected {expected:?}, actual {actual:?}"), metrics))
}
fn scalar(value: SymbolValue) -> Option<(DType, f64)> {
    Some(match value {
        SymbolValue::F32(v) => (DType::F32, v as f64),
        SymbolValue::F16(v) => (DType::F16, registry::f16_to_f32(v) as f64),
        SymbolValue::BF16(v) => (DType::BF16, f32::from_bits((v as u32) << 16) as f64),
        SymbolValue::I32(v) => (DType::I32, v as f64),
        SymbolValue::U32(v) => (DType::U32, v as f64),
        SymbolValue::Bool(v) => (DType::Bool, f64::from(v)),
        _ => return None,
    })
}
fn decode(dtype: DType, bytes: &[u8]) -> f64 {
    match dtype {
        DType::F32 => f32::from_le_bytes(bytes.try_into().unwrap()) as f64,
        DType::F16 => registry::f16_to_f32(u16::from_le_bytes(bytes.try_into().unwrap())) as f64,
        DType::BF16 => {
            f32::from_bits((u16::from_le_bytes(bytes.try_into().unwrap()) as u32) << 16) as f64
        }
        DType::I32 => i32::from_le_bytes(bytes.try_into().unwrap()) as f64,
        DType::U32 => u32::from_le_bytes(bytes.try_into().unwrap()) as f64,
        DType::Bool => f64::from(bytes[0] != 0),
    }
}
fn output_name(path: &[u32]) -> String {
    if path.is_empty() {
        "value".into()
    } else {
        format!(
            "r{}",
            path.iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join("_")
        )
    }
}
fn coordinate(shape: &[usize], mut index: usize) -> Vec<usize> {
    let mut result = vec![0; shape.len()];
    for (coordinate, extent) in result.iter_mut().zip(shape).rev() {
        *coordinate = index % extent;
        index /= extent;
    }
    result
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use seismic_lang::checked::{check_source, SourceFile, SourceSet};
    use seismic_lang::entry::ElementBindings;
    use seismic_lang::interp::{Arg, Interpreter, TensorData};
    use seismic_lang::reference_math::ReferenceScalar;
    pub(crate) fn scalar_outcome(dtype: DType, value: f64) -> OracleOutcome {
        let source = format!(
            "fn probe(x: {}) -> {}:\n    return x\n",
            match dtype {
                DType::F32 => "f32",
                DType::F16 => "f16",
                DType::BF16 => "bf16",
                DType::I32 => "i32",
                DType::U32 => "u32",
                DType::Bool => "bool",
            },
            match dtype {
                DType::F32 => "f32",
                DType::F16 => "f16",
                DType::BF16 => "bf16",
                DType::I32 => "i32",
                DType::U32 => "u32",
                DType::Bool => "bool",
            }
        );
        let value = if dtype.is_float() {
            seismic_lang::reference_math::float_literal(dtype, value)
        } else {
            use seismic_lang::reference_math::ReferenceScalar as R;
            match dtype {
                DType::I32 => R::I32(value as i32),
                DType::U32 => R::U32(value as u32),
                DType::Bool => R::Bool(value != 0.),
                _ => unreachable!(),
            }
        };
        run(&source, &[Arg::Scalar(value)], vec![])
    }
    fn run(source: &str, args: &[Arg], tensors: Vec<TensorData>) -> OracleOutcome {
        let module = check_source(SourceSet::new(vec![SourceFile {
            path: "comparison.seismic".into(),
            text: source.into(),
        }]))
        .unwrap();
        let entry = module
            .entry(
                module.entry_named("probe").unwrap(),
                &ElementBindings::default(),
            )
            .unwrap();
        let mut interpreter = Interpreter::new(&entry);
        for tensor in tensors {
            interpreter.add_tensor(tensor);
        }
        interpreter.run(args).unwrap()
    }
    fn returned(actual: &mut ObservedInvocation) -> &mut Vec<ObservedResult> {
        let SourceTermination::Returned(values) = &mut actual.termination else {
            panic!("expected returned fixture")
        };
        values
    }
    pub(crate) fn scalar_actual(value: SymbolValue) -> ObservedInvocation {
        ObservedInvocation {
            termination: seismic_lang::failure::SourceTermination::Returned(vec![ObservedResult {
                path: vec![],
                value: ObservedValue::Scalar(value),
            }]),
            inputs: vec![],
        }
    }
    fn compare(reference: &OracleOutcome, actual: &ObservedInvocation) -> Comparison {
        compare_outcome(reference, actual, &PrecisionPolicy::Exact, &mut |_| Ok(())).unwrap()
    }
    fn dense(values: &[f32]) -> ObservedTensor {
        ObservedTensor {
            representation: registry::dense(DType::F32),
            shape: vec![values.len()],
            bytes: values.iter().flat_map(|x| x.to_le_bytes()).collect(),
        }
    }
    #[test]
    fn complete_inputs_geometry_and_resource_failures_are_observable() {
        let outcome = run(
            "fn probe[N](x: &tensor[N] f32) -> f32:\n    return x[0]\n",
            &[Arg::Tensor(0)],
            vec![TensorData::dense(DType::F32, vec![2], vec![1., 2.])],
        );
        let mut actual = scalar_actual(SymbolValue::F32(1.));
        assert!(matches!(
            compare(&outcome, &actual),
            Comparison::Mismatch(_)
        ));
        actual.inputs.push(ObservedInput {
            ordinal: 0,
            tensor: dense(&[1., 2.]),
        });
        assert_eq!(compare(&outcome, &actual), Comparison::Match);
        actual.inputs[0].tensor.bytes[4..].copy_from_slice(&3f32.to_le_bytes());
        assert!(matches!(
            compare(&outcome, &actual),
            Comparison::Mismatch(Difference {
                subject: ComparisonSubject::Input(0),
                ..
            })
        ));
        actual.inputs[0].tensor.bytes.pop();
        assert!(matches!(
            compare(&outcome, &actual),
            Comparison::Mismatch(Difference { metrics: None, .. })
        ));
        assert!(matches!(
            compare_outcome(&outcome, &actual, &PrecisionPolicy::Exact, &mut |_| Err(
                ComparisonError::Resource("deadline".into())
            )),
            Err(ComparisonError::Resource(_))
        ));
    }
    #[test]
    fn readonly_dense_bytes_are_exact_under_every_policy_and_allowed_association() {
        use seismic_lang::precision::{Limit, Tolerance};
        let outcome = run(
            "fn probe[N](x: &tensor[N] f32) -> f32:\n    return reduce(x, 0, sum, unordered=true)\n",
            &[Arg::Tensor(0)],
            vec![TensorData::dense(DType::F32, vec![2], vec![1., 2.])],
        );
        assert!(outcome.relation().is_some());
        let mut actual = scalar_actual(SymbolValue::F32(3.));
        actual.inputs.push(ObservedInput {
            ordinal: 0,
            tensor: dense(&[1., 2.]),
        });
        let tolerant = PrecisionPolicy::bounded(Tolerance {
            absolute: Limit::new(0.001).unwrap(),
            relative: Limit::ZERO,
            relative_floor: Limit::ZERO,
            ulps: None,
        });
        for policy in [
            PrecisionPolicy::Exact,
            tolerant,
            PrecisionPolicy::Unconstrained,
        ] {
            actual.inputs[0].tensor = dense(&[1., 2.]);
            assert_eq!(
                compare_outcome(&outcome, &actual, &policy, &mut |_| Ok(())).unwrap(),
                Comparison::Match
            );
            actual.inputs[0].tensor.bytes[..4]
                .copy_from_slice(&f32::from_bits(1f32.to_bits() + 1).to_le_bytes());
            assert!(matches!(
                compare_outcome(&outcome, &actual, &policy, &mut |_| Ok(())).unwrap(),
                Comparison::Mismatch(Difference {
                    subject: ComparisonSubject::Input(0),
                    metrics: None,
                    ..
                })
            ));
        }
        returned(&mut actual)[0].value = ObservedValue::Scalar(SymbolValue::F32(4.));
        assert!(matches!(
            compare(&outcome, &actual),
            Comparison::Mismatch(_)
        ));
    }

    #[test]
    fn readonly_dense_signed_zero_and_nan_payloads_are_storage_identity() {
        for (dtype, name, words) in [
            (
                DType::F32,
                "f32",
                vec![0x8000_0000u32, 0x7f80_0001, 0x7fc1_2345],
            ),
            (DType::F16, "f16", vec![0x8000, 0x7c01, 0x7e45]),
            (DType::BF16, "bf16", vec![0x8000, 0x7f81, 0x7fc5]),
        ] {
            let width = dtype.bytes() as usize;
            let bytes = words
                .iter()
                .flat_map(|word| word.to_le_bytes()[..width].to_vec())
                .collect::<Vec<_>>();
            let outcome = run(
                &format!("fn probe[N](x: &tensor[N] {name}) -> f32:\n    return 0.0\n"),
                &[Arg::Tensor(0)],
                vec![TensorData::dense_from_bytes(dtype, vec![3], bytes.clone()).unwrap()],
            );
            let mut actual = scalar_actual(SymbolValue::F32(0.));
            actual.inputs.push(ObservedInput {
                ordinal: 0,
                tensor: ObservedTensor {
                    representation: registry::dense(dtype),
                    shape: vec![3],
                    bytes: bytes.clone(),
                },
            });
            assert_eq!(compare(&outcome, &actual), Comparison::Match);
            for changed_byte in [width - 1, width, width * 2] {
                actual.inputs[0].tensor.bytes.clone_from(&bytes);
                actual.inputs[0].tensor.bytes[changed_byte] ^= 0x80;
                assert!(matches!(
                    compare_outcome(
                        &outcome,
                        &actual,
                        &PrecisionPolicy::Unconstrained,
                        &mut |_| Ok(())
                    )
                    .unwrap(),
                    Comparison::Mismatch(Difference {
                        subject: ComparisonSubject::Input(0),
                        ..
                    })
                ));
            }
        }
    }

    #[test]
    fn writable_dense_state_retains_numerical_policy_comparison() {
        use seismic_lang::precision::{Limit, Tolerance};
        let outcome = run(
            "fn probe[N](x: &mut tensor[N] f32) -> f32:\n    x[0] = x[0] + 1.0\n    return x[0]\n",
            &[Arg::Tensor(0)],
            vec![TensorData::dense(DType::F32, vec![1], vec![1.])],
        );
        let mut actual = scalar_actual(SymbolValue::F32(2.));
        actual.inputs.push(ObservedInput {
            ordinal: 0,
            tensor: dense(&[f32::from_bits(2f32.to_bits() + 1)]),
        });
        assert!(matches!(
            compare(&outcome, &actual),
            Comparison::Mismatch(_)
        ));
        let policy = PrecisionPolicy::bounded(Tolerance {
            absolute: Limit::new(0.001).unwrap(),
            relative: Limit::ZERO,
            relative_floor: Limit::ZERO,
            ulps: None,
        });
        assert_eq!(
            compare_outcome(&outcome, &actual, &policy, &mut |_| Ok(())).unwrap(),
            Comparison::Match
        );
    }

    #[test]
    fn allowed_disagreement_does_not_hide_later_geometry() {
        let outcome=run("fn probe[N](x: &tensor[N] f32) -> f32:\n    return reduce(x, 0, sum, unordered=true)\n",&[Arg::Tensor(0)],vec![TensorData::dense(DType::F32,vec![2],vec![1.,2.])]);
        assert!(outcome.relation().is_some());
        let mut actual = scalar_actual(SymbolValue::F32(3.));
        actual.inputs.push(ObservedInput {
            ordinal: 0,
            tensor: dense(&[1., 2.]),
        });
        assert_eq!(compare(&outcome, &actual), Comparison::Match);
        returned(&mut actual)[0].value = ObservedValue::Scalar(SymbolValue::F32(4.));
        assert!(matches!(
            compare(&outcome, &actual),
            Comparison::Unsupported(_)
        ));
        actual.inputs[0].tensor.shape[0] = 1;
        assert!(matches!(
            compare(&outcome, &actual),
            Comparison::Mismatch(Difference {
                subject: ComparisonSubject::Input(0),
                ..
            })
        ));
    }
    #[test]
    fn scalar_kinds_and_integer_extrema_remain_exact() {
        for (dtype, value, word) in [
            (DType::I32, i32::MIN as f64, SymbolValue::I32(i32::MIN)),
            (DType::U32, u32::MAX as f64, SymbolValue::U32(u32::MAX)),
            (DType::Bool, 1., SymbolValue::Bool(true)),
        ] {
            assert_eq!(
                compare(&scalar_outcome(dtype, value), &scalar_actual(word)),
                Comparison::Match
            );
        }
        let outcome = scalar_outcome(DType::I32, 1.);
        let actual = ObservedInvocation {
            termination: seismic_lang::failure::SourceTermination::Returned(vec![ObservedResult {
                path: vec![],
                value: ObservedValue::Index(SymbolValue::Nat(1u64.into())),
            }]),
            inputs: vec![],
        };
        assert!(matches!(
            compare(&outcome, &actual),
            Comparison::Mismatch(_)
        ));
        assert!(matches!(
            compare_outcome(
                &outcome,
                &scalar_actual(SymbolValue::I32(2)),
                &PrecisionPolicy::Unconstrained,
                &mut |_| Ok(())
            )
            .unwrap(),
            Comparison::Mismatch(_)
        ));
    }
    #[test]
    fn scalar_specials_use_existing_element_policy() {
        for (reference, actual, accepted) in [
            (f32::NAN, f32::NAN, true),
            (-0., 0., false),
            (f32::INFINITY, f32::NEG_INFINITY, false),
            (f32::from_bits(1), 0., false),
        ] {
            assert_eq!(
                matches!(
                    compare(
                        &scalar_outcome(DType::F32, reference as f64),
                        &scalar_actual(SymbolValue::F32(actual))
                    ),
                    Comparison::Match
                ),
                accepted
            );
        }
        for (dtype, word, value) in [
            (DType::F16, SymbolValue::F16(registry::f16_bits(1.5)), 1.5),
            (
                DType::BF16,
                SymbolValue::BF16((1.5f32.to_bits() >> 16) as u16),
                1.5,
            ),
        ] {
            assert_eq!(
                compare(&scalar_outcome(dtype, value), &scalar_actual(word)),
                Comparison::Match
            );
        }
    }
    #[test]
    fn tensor_result_views_compare_in_logical_order() {
        let outcome = run(
            "fn probe[W](x: &tensor[2,W] f32) -> tensor[W] f32:\n    return to_owned(x[1])\n",
            &[Arg::Tensor(0)],
            vec![TensorData::dense(
                DType::F32,
                vec![2, 2],
                vec![1., 2., 3., 4.],
            )],
        );
        let mut input = dense(&[1., 2., 3., 4.]);
        input.shape = vec![2, 2];
        let actual = ObservedInvocation {
            termination: seismic_lang::failure::SourceTermination::Returned(vec![ObservedResult {
                path: vec![],
                value: ObservedValue::Tensor(dense(&[3., 4.])),
            }]),
            inputs: vec![ObservedInput {
                ordinal: 0,
                tensor: input,
            }],
        };
        assert_eq!(compare(&outcome, &actual), Comparison::Match);
    }
    #[test]
    fn full_width_native_index_and_range_words_are_not_float_rounded() {
        let index = run(
            "fn probe(i: index[2147483647]) -> index[2147483647]:\n    return i\n",
            &[Arg::Index(7u8.into())],
            vec![],
        );
        let actual = |word: u64| ObservedInvocation {
            termination: seismic_lang::failure::SourceTermination::Returned(vec![ObservedResult {
                path: vec![],
                value: ObservedValue::Index(SymbolValue::Nat(word.into())),
            }]),
            inputs: vec![],
        };
        assert_eq!(compare(&index, &actual(7)), Comparison::Match);
        assert!(matches!(
            compare(&index, &actual((1u64 << 54) + 1)),
            Comparison::Mismatch(_)
        ));
        let range = run(
            "fn probe(r: range[2147483647]) -> range[2147483647]:\n    return r\n",
            &[Arg::Range(3u8.into(), 7u8.into())],
            vec![],
        );
        let actual = ObservedInvocation {
            termination: seismic_lang::failure::SourceTermination::Returned(vec![ObservedResult {
                path: vec![],
                value: ObservedValue::Range {
                    start: SymbolValue::Nat(3u64.into()),
                    end: SymbolValue::Nat(((1u64 << 54) + 1).into()),
                },
            }]),
            inputs: vec![],
        };
        assert!(matches!(compare(&range, &actual), Comparison::Mismatch(_)));
    }
    #[test]
    fn readonly_encoded_bytes_are_exact_even_with_allowed_association() {
        let representation = registry::representation("q4g64").unwrap();
        let module=check_source(SourceSet::new(vec![SourceFile{path:"encoded-comparison.seismic".into(),text:"fn probe[N](x: &tensor[N] Q, y: &tensor[N] f32) -> f32:\n    return reduce(y, 0, sum, unordered=true)\n".into()}])).unwrap();
        let entry = module
            .entry(
                module.entry_named("probe").unwrap(),
                &ElementBindings::new().bind("Q", representation),
            )
            .unwrap();
        let registry::RepresentationKind::Packed(layout) =
            &registry::representation_info(representation).kind
        else {
            panic!()
        };
        let bytes = vec![0; layout.bytes(1, 64).unwrap() as usize];
        let mut interpreter = Interpreter::new(&entry);
        interpreter
            .add_tensor(TensorData::encoded(representation, vec![64], bytes.clone()).unwrap());
        interpreter.add_tensor(TensorData::dense(DType::F32, vec![64], vec![0.; 64]));
        let outcome = interpreter.run(&[Arg::Tensor(0), Arg::Tensor(1)]).unwrap();
        let mut actual = scalar_actual(SymbolValue::F32(0.));
        actual.inputs = vec![
            ObservedInput {
                ordinal: 0,
                tensor: ObservedTensor {
                    representation,
                    shape: vec![64],
                    bytes,
                },
            },
            ObservedInput {
                ordinal: 1,
                tensor: dense(&[0.; 64]),
            },
        ];
        assert_eq!(compare(&outcome, &actual), Comparison::Match);
        returned(&mut actual)[0].value = ObservedValue::Scalar(SymbolValue::F32(1.));
        assert!(matches!(
            compare(&outcome, &actual),
            Comparison::Unsupported(_)
        ));
        actual.inputs[0].tensor.bytes[0] = 1;
        assert!(matches!(
            compare(&outcome, &actual),
            Comparison::Mismatch(Difference {
                subject: ComparisonSubject::Input(0),
                ..
            })
        ));
    }
    #[test]
    fn mutated_input_state_is_checked_even_when_returned_value_matches() {
        let outcome = run(
            "fn probe[N](x: &mut tensor[N] f32) -> f32:\n    x[0] = x[0] + 1.0\n    return x[0]\n",
            &[Arg::Tensor(0)],
            vec![TensorData::dense(DType::F32, vec![2], vec![1., 2.])],
        );
        let mut actual = scalar_actual(SymbolValue::F32(2.));
        actual.inputs.push(ObservedInput {
            ordinal: 0,
            tensor: dense(&[2., 2.]),
        });
        assert_eq!(compare(&outcome, &actual), Comparison::Match);
        actual.inputs[0].tensor = dense(&[1., 2.]);
        assert!(matches!(
            compare(&outcome, &actual),
            Comparison::Mismatch(Difference {
                subject: ComparisonSubject::Input(0),
                ..
            })
        ));
    }
    #[test]
    fn correlated_outputs_match_one_complete_representative() {
        let outcome=run("fn probe[N](x: &tensor[N] f32) -> (f32, f32):\n    let total = reduce(x, 0, sum, unordered=true)\n    return (total, total)\n",&[Arg::Tensor(0)],vec![TensorData::dense(DType::F32,vec![2],vec![1.,2.])]);
        let mut actual = ObservedInvocation {
            termination: seismic_lang::failure::SourceTermination::Returned(vec![
                ObservedResult {
                    path: vec![0],
                    value: ObservedValue::Scalar(SymbolValue::F32(3.)),
                },
                ObservedResult {
                    path: vec![1],
                    value: ObservedValue::Scalar(SymbolValue::F32(3.)),
                },
            ]),
            inputs: vec![ObservedInput {
                ordinal: 0,
                tensor: dense(&[1., 2.]),
            }],
        };
        assert_eq!(compare(&outcome, &actual), Comparison::Match);
        returned(&mut actual)[1].value = ObservedValue::Scalar(SymbolValue::F32(4.));
        assert!(matches!(
            compare(&outcome, &actual),
            Comparison::Unsupported(_)
        ));
        returned(&mut actual)[0].path = vec![1];
        assert!(matches!(
            compare(&outcome, &actual),
            Comparison::Mismatch(_)
        ));
    }
    #[test]
    fn owned_encoded_state_requires_comparison_semantics() {
        let representation = registry::representation("q4g64").unwrap();
        let module = check_source(SourceSet::new(vec![SourceFile {
            path: "encoded-state.seismic".into(),
            text: "fn probe[N](x: tensor[N] Q) -> f32:\n    return 0.0\n".into(),
        }]))
        .unwrap();
        let entry = module
            .entry(
                module.entry_named("probe").unwrap(),
                &ElementBindings::new().bind("Q", representation),
            )
            .unwrap();
        let registry::RepresentationKind::Packed(layout) =
            &registry::representation_info(representation).kind
        else {
            panic!()
        };
        let bytes = vec![0; layout.bytes(1, 64).unwrap() as usize];
        let mut interpreter = Interpreter::new(&entry);
        interpreter
            .add_tensor(TensorData::encoded(representation, vec![64], bytes.clone()).unwrap());
        let outcome = interpreter.run(&[Arg::Tensor(0)]).unwrap();
        let mut actual = scalar_actual(SymbolValue::F32(0.));
        actual.inputs = vec![ObservedInput {
            ordinal: 0,
            tensor: ObservedTensor {
                representation,
                shape: vec![64],
                bytes,
            },
        }];
        assert!(matches!(
            compare(&outcome, &actual),
            Comparison::Unsupported(_)
        ));
        actual.inputs[0].tensor.bytes.pop();
        assert!(matches!(
            compare(&outcome, &actual),
            Comparison::Mismatch(_)
        ));
    }
    #[test]
    fn encoded_result_remains_unsupported_without_comparison_semantics() {
        let representation = registry::representation("q4g64").unwrap();
        let module = check_source(SourceSet::new(vec![SourceFile {
            path: "encoded-result.seismic".into(),
            text: "fn probe[N](x: tensor[N] Q) -> tensor[N] Q:\n    return x\n".into(),
        }]))
        .unwrap();
        let entry = module
            .entry(
                module.entry_named("probe").unwrap(),
                &ElementBindings::new().bind("Q", representation),
            )
            .unwrap();
        let registry::RepresentationKind::Packed(layout) =
            &registry::representation_info(representation).kind
        else {
            panic!()
        };
        let bytes = vec![0; layout.bytes(1, 64).unwrap() as usize];
        let mut interpreter = Interpreter::new(&entry);
        interpreter
            .add_tensor(TensorData::encoded(representation, vec![64], bytes.clone()).unwrap());
        let outcome = interpreter.run(&[Arg::Tensor(0)]).unwrap();
        let tensor = || ObservedTensor {
            representation,
            shape: vec![64],
            bytes: bytes.clone(),
        };
        let actual = ObservedInvocation {
            termination: seismic_lang::failure::SourceTermination::Returned(vec![ObservedResult {
                path: vec![],
                value: ObservedValue::Tensor(tensor()),
            }]),
            inputs: vec![ObservedInput {
                ordinal: 0,
                tensor: tensor(),
            }],
        };
        assert!(matches!(
            compare(&outcome, &actual),
            Comparison::Unsupported(_)
        ));
        let policy = PrecisionPolicy::bounded(seismic_lang::precision::Tolerance::EXACT);
        assert!(matches!(
            compare_outcome(&outcome, &actual, &policy, &mut |_| Ok(())).unwrap(),
            Comparison::Unsupported(_)
        ));
    }
    fn failed_actual(reference: &OracleOutcome) -> ObservedInvocation {
        let SourceTermination::Failed(failure) = reference.termination() else {
            panic!("expected failed fixture")
        };
        ObservedInvocation {
            termination: SourceTermination::Failed(failure.clone()),
            inputs: reference
                .inputs()
                .map(|input| {
                    let tensor = input.tensor();
                    ObservedInput {
                        ordinal: input.ordinal(),
                        tensor: ObservedTensor {
                            representation: tensor.representation(),
                            shape: tensor.shape().to_vec(),
                            bytes: tensor.canonical_bytes().unwrap().unwrap().to_vec(),
                        },
                    }
                })
                .collect(),
        }
    }
    #[test]
    fn source_failure_compares_cause_site_and_complete_prefix_state() {
        use seismic_lang::failure::SourceFailureCause;
        use seismic_lang::reference_math::ScalarFailure;
        let source="fn probe(dst: &mut tensor[2] i32, divisor: i32):\n    dst[0] = 7\n    let unused = 42 / divisor\n    dst[1] = 9\n";
        let args = [Arg::Tensor(0), Arg::Scalar(ReferenceScalar::I32(0))];
        let reference = run(
            source,
            &args,
            vec![TensorData::dense(DType::I32, vec![2], vec![0.; 2])],
        );
        let mut actual = failed_actual(&reference);
        assert_eq!(compare(&reference, &actual), Comparison::Match);
        actual.inputs[0].tensor.bytes[4..].copy_from_slice(&9i32.to_le_bytes());
        assert!(matches!(
            compare(&reference, &actual),
            Comparison::Mismatch(_)
        ));
        actual = failed_actual(&reference);
        let SourceTermination::Failed(failure) = &mut actual.termination else {
            unreachable!()
        };
        failure.cause = SourceFailureCause::Scalar(ScalarFailure::ShiftCount);
        assert!(matches!(
            compare(&reference, &actual),
            Comparison::Mismatch(_)
        ));
        actual.termination = SourceTermination::Returned(vec![]);
        assert!(matches!(
            compare(&reference, &actual),
            Comparison::Mismatch(_)
        ));
        // Rebuilding the same checked body preserves its canonical event.
        let rebuilt = run(
            source,
            &args,
            vec![TensorData::dense(DType::I32, vec![2], vec![0.; 2])],
        );
        assert_eq!(
            compare(&reference, &failed_actual(&rebuilt)),
            Comparison::Match
        );
        // A distinct checked body has no established event mapping.
        let alternative = source.replace("42 / divisor", "(40 + 2) / divisor");
        let other = run(
            &alternative,
            &args,
            vec![TensorData::dense(DType::I32, vec![2], vec![0.; 2])],
        );
        actual = failed_actual(&other);
        assert!(matches!(
            compare(&reference, &actual),
            Comparison::Unsupported(_)
        ));
    }
    #[test]
    fn legal_parallel_prefix_disagreement_is_not_a_false_mismatch() {
        let reference=run("fn probe(dst: &mut tensor[2] i32, divisor: i32):\n    parallel for i in 0..2:\n        dst[i] = 7\n        let unused = 42 / divisor\n",&[Arg::Tensor(0),Arg::Scalar(ReferenceScalar::I32(0))],vec![TensorData::dense(DType::I32,vec![2],vec![0.;2])]);
        let mut actual = failed_actual(&reference);
        assert_eq!(compare(&reference, &actual), Comparison::Match);
        actual.inputs[0].tensor.bytes[4..].copy_from_slice(&7i32.to_le_bytes());
        assert!(matches!(
            compare(&reference, &actual),
            Comparison::Unsupported(_)
        ));
        // Geometry remains obligatory even if prefix membership is unknown.
        actual.inputs[0].tensor.bytes.pop();
        assert!(matches!(
            compare(&reference, &actual),
            Comparison::Mismatch(_)
        ));
    }
    #[test]
    fn successful_parallel_execution_does_not_make_all_values_ambiguous() {
        let reference=run("fn probe(dst: &mut tensor[2] f32):\n    parallel for i in 0..2:\n        dst[i] = 7.0\n",&[Arg::Tensor(0)],vec![TensorData::dense(DType::F32,vec![2],vec![0.;2])]);
        let actual = ObservedInvocation {
            termination: SourceTermination::Returned(vec![]),
            inputs: vec![ObservedInput {
                ordinal: 0,
                tensor: dense(&[7., 8.]),
            }],
        };
        assert!(matches!(
            compare(&reference, &actual),
            Comparison::Mismatch(_)
        ));
    }

    #[test]
    fn a_different_event_in_the_same_deterministic_body_is_a_mismatch() {
        let source="fn probe(dst: &mut tensor[2] i32, divisor: i32):\n    dst[0] = 7\n    let first = 42 / divisor\n    let second = 42 / (divisor - 1)\n    dst[1] = 9\n";
        let reference = run(
            source,
            &[Arg::Tensor(0), Arg::Scalar(ReferenceScalar::I32(0))],
            vec![TensorData::dense(DType::I32, vec![2], vec![0.; 2])],
        );
        let second = run(
            source,
            &[Arg::Tensor(0), Arg::Scalar(ReferenceScalar::I32(1))],
            vec![TensorData::dense(DType::I32, vec![2], vec![0.; 2])],
        );
        assert!(matches!(
            compare(&reference, &failed_actual(&second)),
            Comparison::Mismatch(_)
        ));
    }
}
