//! Controlled resident trials use ordinary selected-executable admission.
use seismic_lang::failure::SourceTermination;
use super::*;
use seismic_compiler::feedback::{
    CaseArgument, ControlledObserver, Observation, ObservationError, ObservationRequest,
};
use seismic_compiler::numerics::{
    compare_outcome, Comparison, ComparisonError, ObservedInput, ObservedInvocation,
    ObservedResult, ObservedTensor, ObservedValue, ValidationCase, ValidationObservation,
};
use seismic_lang::interp::{Arg, Interpreter, OracleOutcome, TensorData};
use seismic_lang::types::DType;
use std::time::{Duration, Instant};

pub(super) struct Observer<T: TargetFamily, E: NativeExecutor<T>> {
    device: Arc<Opened<T, E>>,
    public_device: Arc<crate::api::device::DeviceInner>,
}
impl<T: TargetFamily, E: NativeExecutor<T>> Observer<T, E> {
    pub(super) fn new(
        device: Arc<Opened<T, E>>,
        public_device: Arc<crate::api::device::DeviceInner>,
    ) -> Self {
        Self {
            device,
            public_device,
        }
    }
}

enum Input {
    Tensor {
        tensor: Arc<TensorInner>,
        bytes: Vec<u8>,
    },
    Scalar(crate::api::kernel::EncodedScalar),
}
impl Input {
    fn append(&self, args: &mut EncodedArgs) -> Result<(), ObservationError> {
        match self {
            Self::Tensor { tensor, bytes } => {
                tensor.write_from_host(bytes).map_err(tensor_error)?;
                args.push_tensor(tensor.clone());
            }
            Self::Scalar(value) => args.push_scalar(value.clone()),
        }
        Ok(())
    }
}

impl<T: TargetFamily, E: NativeExecutor<T>> ControlledObserver<T, E::Handle> for Observer<T, E> {
    fn environment(&mut self) -> Result<[u8; 32], ObservationError> {
        let mut digest =
            seismic_ir::identity::StructureDigest::new("seismic-controlled-observer-v6");
        digest.bytes(&self.device.device_description().identity().fingerprint);
        Ok(digest.finish())
    }
    fn observe(
        &mut self,
        request: ObservationRequest<'_, T, E::Handle>,
    ) -> Result<Observation, ObservationError> {
        self.run_case(request, None)
            .map(|(observation, _)| observation)
    }
    fn validate(
        &mut self,
        request: ObservationRequest<'_, T, E::Handle>,
        case: ValidationCase,
    ) -> Result<ValidationObservation, ObservationError> {
        self.run_case(request, Some(case))?.1.ok_or_else(|| {
            ObservationError::InvalidCase(
                "diagnostic validation requires one completed trial".into(),
            )
        })
    }
}

impl<T: TargetFamily, E: NativeExecutor<T>> Observer<T, E> {
    fn run_case(
        &mut self,
        request: ObservationRequest<'_, T, E::Handle>,
        validation: Option<ValidationCase>,
    ) -> Result<(Observation, Option<ValidationObservation>), ObservationError> {
        if validation.is_some() && (request.protocol.warmup != 0 || request.protocol.trials != 1) {
            return Err(ObservationError::InvalidCase(
                "diagnostic validation executes exactly one corpus case".into(),
            ));
        }
        if request.case.arguments.len() != request.invocation().schema().parameters().len() {
            return Err(ObservationError::InvalidCase(
                "case argument arity differs from the entry schema".into(),
            ));
        }
        check_deadline(&request)?;
        let mut diagnostic = None;
        let support = seismic_compiler::feedback::assess_observation_support(request.reference)?;
        let started = Instant::now();
        let mut required = 0u64;
        let mut reference_input_bytes = 0u64;
        let mut construction_temporary = 0u64;
        for argument in &request.case.arguments {
            if let CaseArgument::Tensor {
                representation,
                extents,
            } = argument
            {
                if matches!(
                    registry::representation_info(*representation).kind,
                    registry::RepresentationKind::External(_)
                ) {
                    return Err(ObservationError::Unsupported(
                        "external conversion-source input needs a registered finite-content recipe"
                            .into(),
                    ));
                }
                let elements = extents
                    .iter()
                    .try_fold(1u64, |count, extent| count.checked_mul(*extent))
                    .ok_or_else(|| {
                        ObservationError::InvalidCase("input element count overflow".into())
                    })?;
                if elements > request.reference_work_limit {
                    return Err(ObservationError::ReferenceLimit {
                        work_limit: request.reference_work_limit,
                    });
                }
                let layout = crate::layout::canonical(*representation, extents)
                    .map_err(ObservationError::Execution)?;
                let shape = extents
                    .iter()
                    .map(|extent| usize::try_from(*extent))
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|_| {
                        ObservationError::InvalidCase(
                            "input rank exceeds host address space".into(),
                        )
                    })?;
                let reference_bytes = TensorData::allocation_bytes(*representation, &shape)
                    .map_err(ObservationError::InvalidCase)?;
                // Persistent input payloads: native buffer, host restore image,
                // and reference backing. Oracle-owned view maps are charged by
                // the interpreter's own live allocation ledger during replay.
                required = required.saturating_add(layout.byte_len.saturating_mul(2));
                reference_input_bytes = reference_input_bytes.saturating_add(reference_bytes);
                let temporary = match &registry::representation_info(*representation).kind {
                    registry::RepresentationKind::Dense(_) => 0,
                    registry::RepresentationKind::Packed(packet) => {
                        let info = registry::representation_info(*representation);
                        let recipe = registry::decode_recipe(*representation, info.decoded)
                            .ok_or_else(|| {
                                ObservationError::Unsupported(
                                    "packed input has no reference decoder".into(),
                                )
                            })?;
                        // The packet clone, decoded group and decoder workspace
                        // coexist during registry validation of one packet.
                        let validation = u64::from(packet.packet_size)
                            .saturating_add(
                                u64::from(packet.group)
                                    .saturating_mul(std::mem::size_of::<f64>() as u64),
                            )
                            .saturating_add(
                                (recipe.temporary_count() as u64)
                                    .saturating_mul(std::mem::size_of::<f64>() as u64),
                            );
                        layout.byte_len.max(validation)
                    }
                    registry::RepresentationKind::External(_) => unreachable!(),
                };
                construction_temporary = construction_temporary.max(temporary);
            }
        }
        let construction_peak = required
            .saturating_add(reference_input_bytes)
            .saturating_add(construction_temporary);
        if construction_peak > request.memory_limit {
            return Err(ObservationError::Capacity {
                required: construction_peak.into(),
                limit: request.memory_limit,
            });
        }
        let mut reference = Interpreter::from_view(request.reference);
        let mut reference_args = Vec::new();
        let mut inputs = Vec::new();
        let mut random = request.case.seed;
        for (ordinal, argument) in request.case.arguments.iter().enumerate() {
            check_deadline(&request)?;
            match argument {
                CaseArgument::Tensor {
                    representation,
                    extents,
                } => {
                    let name = &request.invocation().schema().parameters()[ordinal].name;
                    let assumption = match request.precision {
                        PrecisionPolicy::Bounded { inputs, .. } => inputs
                            .get(name)
                            .map(|range| (range.minimum.get(), range.maximum.get())),
                        _ => None,
                    };
                    if let registry::RepresentationKind::Packed(layout) =
                        &registry::representation_info(*representation).kind
                    {
                        let bytes = packed_input(
                            *representation,
                            layout,
                            extents,
                            assumption,
                            &mut random,
                        )?;
                        let data = TensorData::encoded(
                            *representation,
                            extents.iter().map(|n| *n as usize).collect(),
                            bytes.clone(),
                        )
                        .map_err(ObservationError::InvalidCase)?;
                        reference_args.push(Arg::Tensor(reference.add_tensor(data)));
                        let tensor = Arc::new(
                            TensorInner::from_host(
                                &self.public_device,
                                *representation,
                                extents,
                                &bytes,
                            )
                            .map_err(tensor_error)?,
                        );
                        inputs.push(Input::Tensor { tensor, bytes });
                        continue;
                    }
                    let registry::RepresentationKind::Dense(dtype) =
                        registry::representation_info(*representation).kind
                    else {
                        unreachable!()
                    };
                    let count = extents
                        .iter()
                        .try_fold(1usize, |count, extent| {
                            count.checked_mul(usize::try_from(*extent).ok()?)
                        })
                        .ok_or_else(|| {
                            ObservationError::InvalidCase(
                                "input geometry overflows host address space".into(),
                            )
                        })?;
                    let mut bytes = Vec::with_capacity(count * dtype.bytes() as usize);
                    for element in 0..count {
                        let rounded = match validation {
                            Some(case) => validation_input_value(
                                dtype,
                                assumption,
                                case,
                                element,
                                &mut random,
                            )?,
                            None => dense_input_value(dtype, assumption, &mut random)?,
                        };
                        encode(dtype, rounded, &mut bytes);
                    }
                    let shape = extents.iter().map(|extent| *extent as usize).collect();
                    let data = TensorData::dense_from_bytes(dtype, shape, bytes.clone())
                        .map_err(ObservationError::InvalidCase)?;
                    reference_args.push(Arg::Tensor(reference.add_tensor(data)));
                    let tensor = Arc::new(
                        TensorInner::from_host(
                            &self.public_device,
                            *representation,
                            extents,
                            &bytes,
                        )
                        .map_err(tensor_error)?,
                    );
                    inputs.push(Input::Tensor { tensor, bytes });
                }
                CaseArgument::Scalar(value) => {
                    let (encoded, value) = scalar(value.clone())?;
                    inputs.push(Input::Scalar(encoded));
                    reference_args.push(Arg::Scalar(value));
                }
                CaseArgument::Index(value) => {
                    inputs.push(Input::Scalar(crate::api::kernel::EncodedScalar::Index(
                        value.clone(),
                    )));
                    reference_args.push(Arg::Index(value.clone()));
                }
                CaseArgument::Range { start, end } => {
                    inputs.push(Input::Scalar(crate::api::kernel::EncodedScalar::Range {
                        start: start.clone(),
                        end: end.clone(),
                    }));
                    reference_args.push(Arg::Range(start.clone(), end.clone()));
                }
            }
        }
        let mut setup_time = started.elapsed();
        let mut checking_time = Duration::ZERO;
        let mut samples = Vec::new();
        // Reference execution occurs after ordinary admission establishes the
        // complete implementation's resource footprint, outside the timing endpoint.
        let mut expected: Option<OracleOutcome> = None;
        let mut reference = Some(reference);
        let mut remaining_work = request.reference_work_limit;
        for trial in 0..request
            .protocol
            .warmup
            .saturating_add(request.protocol.trials)
        {
            check_deadline(&request)?;
            let setup = Instant::now();
            let mut args = EncodedArgs::new();
            for input in &inputs {
                input.append(&mut args)?;
            }
            let admitted = workflow::native::admit_trial(
                &self.device,
                &self.public_device,
                request.executable,
                args,
                request
                    .memory_limit
                    .saturating_sub(required)
                    .saturating_sub(
                        expected
                            .as_ref()
                            .map(|o| o.retained_payload_bytes())
                            .unwrap_or(reference_input_bytes),
                    ),
            )
            .map_err(call_error)?;
            {
                let mut invocations = admitted.invocation_values();
                let actual = invocations.next().expect("a trial admits one invocation");
                assert!(
                    invocations.next().is_none(),
                    "a trial admits one invocation"
                );
                validate_case_bindings(
                    request.invocation().schema(),
                    &request.case.values,
                    actual,
                )?;
            }
            let total = required.saturating_add(admitted.allocated_bytes());
            if total > request.memory_limit {
                return Err(ObservationError::Capacity {
                    required: total.into(),
                    limit: request.memory_limit,
                });
            }
            setup_time += setup.elapsed();
            if expected.is_none() {
                let check = Instant::now();
                let outcome = reference
                    .take()
                    .expect("reference runs once")
                    .run_bounded_with_memory(
                        &reference_args,
                        request.reference_work_limit,
                        request.memory_limit.saturating_sub(total),
                    )
                    .map_err(|error| match error {
                        seismic_lang::interp::OracleError::InterpreterDefect(reason) => {
                            ObservationError::Contract(reason)
                        }
                        seismic_lang::interp::OracleError::InvalidInvocation(reason) => {
                            ObservationError::InvalidCase(reason)
                        }
                        seismic_lang::interp::OracleError::WorkLimit { limit } => {
                            ObservationError::ReferenceLimit { work_limit: limit }
                        }
                        seismic_lang::interp::OracleError::MemoryLimit { required, .. } => {
                            ObservationError::Capacity {
                                required: total.saturating_add(required).into(),
                                limit: request.memory_limit,
                            }
                        }
                    })?;
                remaining_work = remaining_work.checked_sub(outcome.work_units()).ok_or(
                    ObservationError::ReferenceLimit {
                        work_limit: request.reference_work_limit,
                    },
                )?;
                expected = Some(outcome);
                checking_time += check.elapsed();
            }
            check_deadline(&request)?;
            let timer = Instant::now();
            let completed = admitted
                .submit()
                .map_err(call_error)?
                .complete()
                .map_err(call_error)?;
            let elapsed = timer.elapsed();
            let check = Instant::now();
            let termination = match completed.outcome().map_err(call_error)? {
                SourceTermination::Returned(mut groups) => {
                    assert_eq!(groups.len(), 1, "single trial has one output group");
                    SourceTermination::Returned(groups.pop().unwrap())
                }
                SourceTermination::Failed(failure) => SourceTermination::Failed(failure.failure),
            };
            let stopped = matches!(&termination, SourceTermination::Failed(_));
            let live = total.saturating_add(expected.as_ref().unwrap().retained_payload_bytes());
            // The completed owner retains admission permits and backing until
            // all observable prefix state has been copied into this observation.
            let actual = capture_invocation(&request, termination, &inputs, live, |tensor| completed.read_tensor(tensor))?;
            drop(completed);
            if validation.is_some() {
                diagnostic = Some(ValidationObservation {
                    reference: expected.take().unwrap(),
                    actual,
                    remaining_work,
                });
            } else {
                let mut allowance = |units| {
                    if request
                        .deadline
                        .is_some_and(|deadline| Instant::now() >= deadline)
                    {
                        return Err(ComparisonError::Resource(
                            "observation deadline expired".into(),
                        ));
                    }
                    remaining_work = remaining_work.checked_sub(units).ok_or_else(|| {
                        ComparisonError::Resource("checking work limit exceeded".into())
                    })?;
                    Ok(())
                };
                match compare_outcome(
                    expected.as_ref().unwrap(),
                    &actual,
                    request.precision,
                    &mut allowance,
                )
                .map_err(|error| match error {
                    ComparisonError::Resource(_)
                        if request
                            .deadline
                            .is_some_and(|deadline| Instant::now() >= deadline) =>
                    {
                        ObservationError::DeadlineExpired
                    }
                    ComparisonError::Resource(_) => ObservationError::ReferenceLimit {
                        work_limit: request.reference_work_limit,
                    },
                    ComparisonError::Contract(reason) => ObservationError::Contract(reason),
                })? {
                    Comparison::Match => {}
                    Comparison::Mismatch(reason) => {
                        return Err(ObservationError::IncorrectResult(format!("{reason:?}")))
                    }
                    Comparison::Unsupported(reason) => {
                        return Err(ObservationError::Unsupported(reason))
                    }
                }
            }
            checking_time += check.elapsed();
            if stopped {
                if validation.is_none() {
                    return Err(ObservationError::Unsupported("completed source failure has no timing sample".into()));
                }
                break;
            }
            if trial >= request.protocol.warmup {
                samples.push(elapsed);
            }
        }
        Ok((
            Observation {
                support,
                samples,
                setup_time,
                checking_time,
                environment: self.environment()?,
            },
            diagnostic,
        ))
    }
}

fn same_binding(expected: SymbolValue, actual: SymbolValue) -> bool {
    match (expected, actual) {
        (SymbolValue::F32(a), SymbolValue::F32(b)) => a.to_bits() == b.to_bits(),
        (a, b) => a == b,
    }
}

fn validate_case_bindings(
    schema: &seismic_lang::entry::CallSchema,
    requested: &seismic_lang::expr::compiled::InvocationValues,
    inferred: &seismic_lang::expr::compiled::InvocationValues,
) -> Result<(), ObservationError> {
    use seismic_lang::entry::ParameterKind;
    let mut symbols = schema
        .dimensions()
        .iter()
        .map(|dimension| dimension.symbol)
        .collect::<Vec<_>>();
    for parameter in schema.parameters() {
        match &parameter.kind {
            ParameterKind::Scalar { symbol, .. } | ParameterKind::Index { symbol, .. } => {
                symbols.push(*symbol)
            }
            ParameterKind::Range { start, end, .. } => {
                symbols.push(*start);
                symbols.push(*end);
            }
            ParameterKind::Tensor { .. } => {}
        }
    }
    for symbol in symbols {
        if !matches!((requested.get(symbol), inferred.get(symbol)), (Some(a), Some(b)) if same_binding(a.clone(), b.clone()))
        {
            return Err(ObservationError::InvalidCase(
                "constructed arguments do not reproduce the requested invocation point".into(),
            ));
        }
    }
    Ok(())
}

fn random_bits(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut value = *state;
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
    value ^ (value >> 31)
}

/// Generate in the representation's value domain. Integer and Boolean inputs
/// must not be obtained by truncating a continuous floating-point distribution.
fn dense_input_value(
    dtype: DType,
    assumption: Option<(f64, f64)>,
    random: &mut u64,
) -> Result<f64, ObservationError> {
    let bits = random_bits(random);
    if matches!(dtype, DType::I32 | DType::U32 | DType::Bool) {
        let (minimum, maximum) = match dtype {
            DType::I32 => (i32::MIN as f64, i32::MAX as f64),
            DType::U32 => (0.0, u32::MAX as f64),
            DType::Bool => (0.0, 1.0),
            _ => unreachable!(),
        };
        // Small finite defaults are content recipes, not inferred index bounds.
        // Ordinary reference admission still checks all content-based accesses.
        let (low, high) = assumption.unwrap_or((0.0, 1.0));
        let low = low.ceil().max(minimum);
        let high = high.floor().min(maximum);
        if !low.is_finite() || !high.is_finite() || low > high {
            return Err(ObservationError::Unsupported(format!(
                "input assumption contains no representable {dtype:?} integer"
            )));
        }
        let low = low as i64;
        let count = (high as i64 - low + 1) as u64;
        let offset = ((u128::from(bits) * u128::from(count)) >> 64) as i64;
        return Ok((low + offset) as f64);
    }
    let (low, high) = assumption.unwrap_or((-1.0, 1.0));
    let unit = (bits >> 11) as f64 / ((1u64 << 53) as f64);
    let value = seismic_lang::interp::round_to(dtype, low * (1.0 - unit) + high * unit);
    if !value.is_finite() || value < low || value > high {
        return Err(ObservationError::Unsupported(format!(
            "input assumption needs a representable {dtype:?} recipe"
        )));
    }
    Ok(value)
}

/// The registry defines packet structure and interpretation. Generate plane
/// contents, then require its own semantic decoder to confirm finite values and
/// any declared input assumption. No representation names enter the recipe.
fn packed_input(
    representation: RepresentationId,
    layout: &registry::PackedPacketLayout,
    extents: &[u64],
    assumption: Option<(f64, f64)>,
    random: &mut u64,
) -> Result<Vec<u8>, ObservationError> {
    let canonical =
        crate::layout::canonical(representation, extents).map_err(ObservationError::Execution)?;
    let byte_len = usize::try_from(canonical.byte_len).map_err(|_| {
        ObservationError::Unsupported("packet input exceeds host address space".into())
    })?;
    let mut bytes = vec![0; byte_len];
    for packet in bytes.chunks_exact_mut(layout.packet_size as usize) {
        let mut accepted = false;
        for _ in 0..32 {
            for plane in &layout.planes {
                let storage = &mut packet
                    [plane.offset as usize..(plane.offset + plane.bytes_per_group) as usize];
                match plane.encoding {
                    registry::PlaneEncoding::Dense(dtype) => {
                        for element in storage.chunks_exact_mut(dtype.bytes() as usize) {
                            let unit = (random_bits(random) >> 11) as f64 / ((1u64 << 53) as f64);
                            let value = seismic_lang::interp::round_to(dtype, 2.0 * unit - 1.0);
                            let mut encoded = Vec::with_capacity(element.len());
                            encode(dtype, value, &mut encoded);
                            element.copy_from_slice(&encoded);
                        }
                    }
                    registry::PlaneEncoding::Packed { .. }
                    | registry::PlaneEncoding::FloatCode { .. } => {
                        for byte in storage {
                            *byte = random_bits(random) as u8;
                        }
                    }
                }
            }
            let reference =
                TensorData::encoded(representation, vec![layout.group as usize], packet.to_vec())
                    .map_err(ObservationError::InvalidCase)?;
            if reference
                .values()
                .map_err(ObservationError::InvalidCase)?
                .iter()
                .all(|value| {
                    value.is_finite()
                        && assumption.is_none_or(|(low, high)| *value >= low && *value <= high)
                })
            {
                accepted = true;
                break;
            }
        }
        if !accepted {
            return Err(ObservationError::Unsupported("no finite packed input satisfying the declared range was found within the recipe limit".into()));
        }
    }
    Ok(bytes)
}

fn call_error(error: CallError) -> ObservationError {
    match error {
        CallError::Execution(ExecutionError::AllocationCapacity { required, available }) => ObservationError::Capacity { required, limit: available },
        CallError::Execution(error) => ObservationError::Execution(error),
        CallError::Invocation(InvocationError::AllocationCapacity {
            required,
            available,
        }) => ObservationError::Capacity {
            required,
            limit: available,
        },
        other => ObservationError::InvalidCase(format!("trial admission: {other:?}")),
    }
}
fn tensor_error(error: crate::api::TensorError) -> ObservationError {
    match error {
        crate::api::TensorError::Execution(error) => ObservationError::Execution(error),
        other => ObservationError::InvalidCase(other.to_string()),
    }
}
fn scalar(
    value: SymbolValue,
) -> Result<
    (
        crate::api::kernel::EncodedScalar,
        seismic_lang::reference_math::ReferenceScalar,
    ),
    ObservationError,
> {
    use crate::api::kernel::EncodedScalar as S;
    use seismic_lang::reference_math::ReferenceScalar as R;
    Ok(match value {
        SymbolValue::F32(value) => (S::F32Bits(value.to_bits()), R::F32(value.to_bits())),
        SymbolValue::F16(value) => (S::F16(value), R::F16(value)),
        SymbolValue::BF16(value) => (S::BF16(value), R::BF16(value)),
        SymbolValue::I32(value) => (S::I32(value), R::I32(value)),
        SymbolValue::U32(value) => (S::U32(value), R::U32(value)),
        SymbolValue::Bool(value) => (S::Bool(value), R::Bool(value)),
        _ => {
            return Err(ObservationError::InvalidCase(
                "index sort cannot encode a scalar parameter".into(),
            ))
        }
    })
}

fn encode(dtype: DType, value: f64, bytes: &mut Vec<u8>) {
    match dtype {
        DType::F32 => bytes.extend_from_slice(&(value as f32).to_le_bytes()),
        DType::F16 => bytes.extend_from_slice(&registry::f16_bits(value as f32).to_le_bytes()),
        DType::BF16 => bytes.extend_from_slice(
            &((registry::bf16_round(value as f32).to_bits() >> 16) as u16).to_le_bytes(),
        ),
        DType::I32 => bytes.extend_from_slice(&(value as i32).to_le_bytes()),
        DType::U32 => bytes.extend_from_slice(&(value as u32).to_le_bytes()),
        DType::Bool => bytes.push(u8::from(value != 0.0)),
    }
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
fn validation_input_value(
    dtype: DType,
    assumption: Option<(f64, f64)>,
    case: ValidationCase,
    ordinal: usize,
    random: &mut u64,
) -> Result<f64, ObservationError> {
    if case == ValidationCase::FiniteRandom {
        return dense_input_value(dtype, assumption, random);
    }
    let integer = matches!(dtype, DType::I32 | DType::U32 | DType::Bool);
    let default = match dtype {
        DType::I32 => (i32::MIN as f64, i32::MAX as f64),
        DType::U32 => (0.0, u32::MAX as f64),
        DType::Bool => (0.0, 1.0),
        DType::F16 => (-65504.0, 65504.0),
        DType::BF16 => {
            let maximum = f32::from_bits(0x7f7f0000) as f64;
            (-maximum, maximum)
        }
        DType::F32 => (-(f32::MAX as f64), f32::MAX as f64),
    };
    let (low, high) = assumption.unwrap_or(default);
    let smallest = match dtype {
        DType::F16 => registry::f16_to_f32(1) as f64,
        DType::BF16 => f32::from_bits(1 << 16) as f64,
        DType::F32 => f32::from_bits(1) as f64,
        _ => 1.0,
    };
    let value = match case {
        ValidationCase::FiniteRandom => unreachable!(),
        ValidationCase::Zeros => 0.0,
        ValidationCase::AlternatingExtremes => {
            if ordinal % 2 == 0 {
                low
            } else {
                high
            }
        }
        ValidationCase::SmallMagnitude => {
            if ordinal % 2 == 0 {
                smallest
            } else {
                -smallest
            }
        }
        ValidationCase::SpecialValues if !integer && assumption.is_none() => {
            [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.0, smallest][ordinal % 5]
        }
        ValidationCase::SpecialValues => {
            if ordinal % 2 == 0 {
                -0.0
            } else {
                smallest
            }
        }
    };
    if !integer && !value.is_finite() {
        return Ok(seismic_lang::interp::round_to(dtype, value));
    }
    if integer {
        // Sampling a singleton uses the same representability validation as
        // random inputs, including fractional/empty bounded integer domains.
        let min = low.ceil();
        let max = high.floor();
        if min > max {
            return Err(ObservationError::Unsupported(
                "validation range has no integer".into(),
            ));
        }
        let chosen = value.round().max(min).min(max);
        return dense_input_value(dtype, Some((chosen, chosen)), random);
    }
    let bounded = if value < low {
        low
    } else if value > high {
        high
    } else {
        value
    };
    let rounded = seismic_lang::interp::round_to(dtype, bounded);
    if !rounded.is_finite() || rounded < low || rounded > high {
        return Err(ObservationError::Unsupported(
            "validation stress case has no representable bounded value".into(),
        ));
    }
    Ok(rounded)
}

fn check_deadline<T: TargetFamily, H>(
    request: &ObservationRequest<'_, T, H>,
) -> Result<(), ObservationError> {
    if request
        .deadline
        .is_some_and(|deadline| Instant::now() >= deadline)
    {
        return Err(ObservationError::DeadlineExpired);
    }
    Ok(())
}

/// Capture native observations once, preserving typed words and canonical bytes.
fn capture_invocation<T: TargetFamily, H>(
    request: &ObservationRequest<'_, T, H>,
    termination: SourceTermination<Vec<DecodedValue>>,
    inputs: &[Input],
    live: u64,
    mut read: impl FnMut(&TensorInner) -> Result<Vec<u8>, ExecutionError>,
) -> Result<ObservedInvocation, ObservationError> {
    let outputs = match &termination {
        SourceTermination::Returned(values) => values.as_slice(),
        SourceTermination::Failed(_) => &[],
    };
    let mut bytes = (outputs.len() * std::mem::size_of::<ObservedResult>()
        + inputs.len() * std::mem::size_of::<ObservedInput>()) as u64;
    for output in outputs {
        if let DecodedValue::Tensor(t) = output {
            bytes = bytes
                .saturating_add(t.byte_len())
                .saturating_add((t.extents().len() * std::mem::size_of::<usize>()) as u64);
        }
    }
    for input in inputs {
        if let Input::Tensor { tensor, .. } = input {
            bytes = bytes
                .saturating_add(tensor.byte_len())
                .saturating_add((tensor.extents().len() * std::mem::size_of::<usize>()) as u64);
        }
    }
    if matches!(&termination, SourceTermination::Returned(_)) {
        for r in request.invocation().schema().results() {
            bytes = bytes.saturating_add((r.path.len() * 4) as u64);
        }
    } else if let SourceTermination::Failed(failure) = &termination {
        bytes = bytes.saturating_add(std::mem::size_of_val(failure) as u64);
        if let seismic_lang::failure::SourceFailureCause::Check(seismic_lang::entry::CheckReason::Custom(text)) = &failure.cause {
            bytes = bytes.saturating_add(text.capacity() as u64);
        }
    }
    let required = live.checked_add(bytes).unwrap_or(u64::MAX);
    if required > request.memory_limit {
        return Err(ObservationError::Capacity {
            required: required.into(),
            limit: request.memory_limit,
        });
    }
    let mut tensor = |t: &TensorInner| -> Result<ObservedTensor, ObservationError> {
        Ok(ObservedTensor {
            representation: t.representation(),
            shape: t.extents().iter().map(|n| *n as usize).collect(),
            bytes: read(t).map_err(ObservationError::Execution)?,
        })
    };
    let mut results = Vec::with_capacity(outputs.len());
    for (ordinal, output) in outputs.iter().enumerate() {
        check_deadline(request)?;
        let value = match output {
            DecodedValue::Tensor(t) => ObservedValue::Tensor(tensor(t)?),
            DecodedValue::Scalar(value) => match value {
                ArgumentValue::Index(v) => ObservedValue::Index(SymbolValue::Nat(v.clone())),
                ArgumentValue::Range { start, end } => ObservedValue::Range {
                    start: SymbolValue::Nat(start.clone()),
                    end: SymbolValue::Nat(end.clone()),
                },
                ArgumentValue::F32(v) => ObservedValue::Scalar(SymbolValue::F32(*v)),
                ArgumentValue::F16(v) => ObservedValue::Scalar(SymbolValue::F16(*v)),
                ArgumentValue::BF16(v) => ObservedValue::Scalar(SymbolValue::BF16(*v)),
                ArgumentValue::I32(v) => ObservedValue::Scalar(SymbolValue::I32(*v)),
                ArgumentValue::U32(v) => ObservedValue::Scalar(SymbolValue::U32(*v)),
                ArgumentValue::Bool(v) => ObservedValue::Scalar(SymbolValue::Bool(*v)),
                _ => {
                    return Err(ObservationError::InvalidCase(
                        "non-value in completed scalar result".into(),
                    ))
                }
            },
        };
        results.push(ObservedResult {
            path: request
                .invocation()
                .schema()
                .results()
                .get(ordinal)
                .map(|r| r.path.clone())
                .unwrap_or_default(),
            value,
        });
    }
    let mut final_inputs = Vec::with_capacity(inputs.len());
    for (ordinal, input) in inputs.iter().enumerate() {
        check_deadline(request)?;
        if let Input::Tensor { tensor: t, .. } = input {
            final_inputs.push(ObservedInput {
                ordinal,
                tensor: tensor(t)?,
            });
        }
    }
    Ok(ObservedInvocation {
        termination: match termination {
            SourceTermination::Returned(_) => SourceTermination::Returned(results),
            SourceTermination::Failed(failure) => SourceTermination::Failed(failure),
        },
        inputs: final_inputs,
    })
}

#[cfg(all(test, target_os = "macos"))]
mod tests {

    #[test]
    fn constructed_case_must_match_every_requested_invocation_binding() {
        let module = check_source(SourceSet::new(vec![SourceFile {
            path: "case-identity.seismic".into(),
            text: "fn probe[N](x: &tensor[N] f32, value: f32) -> f32:\n    return value\n".into(),
        }]))
        .unwrap();
        let entry = module
            .entry(
                module.entry_named("probe").unwrap(),
                &ElementBindings::new(),
            )
            .unwrap();
        let schema = entry.schema();
        let dimension = schema.dimensions()[0].symbol;
        let seismic_lang::entry::ParameterKind::Scalar { symbol, .. } = schema.parameters()[1].kind
        else {
            panic!()
        };
        let mut inferred = seismic_lang::expr::compiled::InvocationValues::new();
        inferred.bind(dimension, SymbolValue::Nat(5_u64.into()));
        inferred.bind(symbol, SymbolValue::F32(-0.0));
        assert!(validate_case_bindings(schema, &inferred, &inferred).is_ok());
        let mut wrong = inferred.clone();
        wrong.bind(dimension, SymbolValue::Nat(6_u64.into()));
        assert!(validate_case_bindings(schema, &wrong, &inferred).is_err());
        wrong.bind(dimension, SymbolValue::Nat(5_u64.into()));
        wrong.bind(symbol, SymbolValue::F32(0.0));
        assert!(validate_case_bindings(schema, &wrong, &inferred).is_err());
    }

    #[test]
    fn observed_invocation_identity_preserves_float_bits() {
        assert!(same_binding(
            SymbolValue::F32(f32::from_bits(0x7fc00001)),
            SymbolValue::F32(f32::from_bits(0x7fc00001))
        ));
        assert!(!same_binding(
            SymbolValue::F32(f32::from_bits(0x7fc00001)),
            SymbolValue::F32(f32::from_bits(0x7fc00002))
        ));
        assert!(!same_binding(SymbolValue::F32(0.0), SymbolValue::F32(-0.0)));
    }

    #[test]
    fn validation_recipes_preserve_specials_and_declared_bounds() {
        let mut random = 91;
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            assert!(validation_input_value(
                dtype,
                None,
                ValidationCase::SpecialValues,
                0,
                &mut random
            )
            .unwrap()
            .is_nan());
            assert_eq!(
                validation_input_value(dtype, None, ValidationCase::SpecialValues, 1, &mut random)
                    .unwrap(),
                f64::INFINITY
            );
            let zero =
                validation_input_value(dtype, None, ValidationCase::SpecialValues, 3, &mut random)
                    .unwrap();
            assert_eq!(zero, 0.0);
            assert!(zero.is_sign_negative());
            for (case, _) in ValidationCase::CORPUS {
                for ordinal in 0..8 {
                    let value =
                        validation_input_value(dtype, Some((0.5, 2.0)), case, ordinal, &mut random)
                            .unwrap();
                    assert!(value.is_finite() && (0.5..=2.0).contains(&value));
                }
            }
        }
    }

    #[test]
    fn integer_content_recipes_are_discrete_diverse_and_reproducible() {
        for dtype in [DType::I32, DType::U32, DType::Bool] {
            let mut first = 42;
            let mut second = 42;
            let values = (0..128)
                .map(|_| dense_input_value(dtype, None, &mut first).unwrap())
                .collect::<Vec<_>>();
            let repeated = (0..128)
                .map(|_| dense_input_value(dtype, None, &mut second).unwrap())
                .collect::<Vec<_>>();
            assert_eq!(values, repeated);
            assert!(values.contains(&0.0));
            assert!(values.contains(&1.0));
        }
        let mut random = 19;
        for _ in 0..128 {
            let value = dense_input_value(DType::I32, Some((-3.5, -0.5)), &mut random).unwrap();
            assert!([-3.0, -2.0, -1.0].contains(&value));
        }
        assert!(dense_input_value(DType::I32, Some((0.25, 0.75)), &mut random).is_err());
        assert!(dense_input_value(DType::U32, Some((-3.0, -1.0)), &mut random).is_err());
        assert!(dense_input_value(DType::Bool, Some((2.0, 3.0)), &mut random).is_err());
    }
    use super::*;
    use seismic_compiler::feedback::{
        FeedbackOptions, InvocationParameter, InvocationScope, PreparationOptions,
    };
    use seismic_lang::checked::{check_source, SourceFile, SourceSet};
    #[test]
    fn packed_recipes_follow_every_registered_resident_layout() {
        for info in registry::representations() {
            let registry::RepresentationKind::Packed(layout) = &info.kind else {
                continue;
            };
            let extents = [2, layout.group as u64 + 1];
            let a = packed_input(info.id, layout, &extents, None, &mut 1).unwrap();
            let again = packed_input(info.id, layout, &extents, None, &mut 1).unwrap();
            let b = packed_input(info.id, layout, &extents, None, &mut 2).unwrap();
            assert_eq!(a, again, "{}", info.name);
            assert_ne!(a, b, "{}", info.name);
            let data =
                TensorData::encoded(info.id, extents.iter().map(|n| *n as usize).collect(), a)
                    .unwrap();
            assert!(
                data.values().unwrap().iter().all(|value| value.is_finite()),
                "{}",
                info.name
            );
        }
    }

    #[test]
    #[ignore = "requires Metal for complete-entry numerical corpus execution"]
    fn metal_validation_collects_complete_mutated_input_state() {
        let catalog = crate::api::catalog::Catalog::discover().unwrap();
        let info = catalog
            .devices()
            .iter()
            .find(|device| device.backend == registry::BackendName::Metal)
            .unwrap();
        let device = catalog.open(info.id).unwrap();
        let module = check_source(SourceSet::new(vec![SourceFile {
            path: "validation-state.seismic".into(),
            text: "fn update[N](x: &tensor[N] f32, output: &mut tensor[N] f32):\n    parallel for i in 0..N:\n        output[i] = x[i] + 1.0\n".into(),
        }])).unwrap();
        let entry = module.entry_named("update").unwrap();
        let logical = module.entry(entry, &ElementBindings::new()).unwrap();
        let kernel = crate::api::kernel::prepare(
            &module,
            entry,
            ElementBindings::new(),
            &device,
            PreparationOptions::feedback(
                PrecisionPolicy::Exact,
                FeedbackOptions {
                    search_time: Duration::ZERO,
                    ..Default::default()
                },
            ),
        )
        .unwrap();
        let crate::backends::PreparedKind::Metal(prepared) = &kernel.inner else {
            panic!()
        };
        let prepared = &prepared.prepared;
        let mut observer = Observer::new(prepared.device.clone(), device.clone());
        let mut values = seismic_lang::expr::compiled::InvocationValues::new();
        values.bind(
            prepared.kernel.schema().dimensions()[0].symbol,
            SymbolValue::Nat(5_u64.into()),
        );
        for (corpus_case, seed) in ValidationCase::CORPUS {
            let case = seismic_compiler::feedback::ObservationCase {
                values: values.clone(),
                seed,
                arguments: (0..2)
                    .map(|_| CaseArgument::Tensor {
                        representation: registry::dense(DType::F32),
                        extents: vec![5],
                    })
                    .collect(),
            };
            let request = ObservationRequest {
                candidate: prepared.kernel.candidate_for_variant(prepared.kernel.select(&values)),
                reference: logical.as_view(),
                executable: prepared.kernel.variants().first(),
                deadline: None,
                precision: &PrecisionPolicy::Exact,
                case: &case,
                protocol: seismic_compiler::feedback::ObservationProtocol {
                    warmup: 0,
                    trials: 1,
                },
                memory_limit: 16 * 1024 * 1024,
                reference_work_limit: 10000,
            };
            if corpus_case == ValidationCase::CORPUS[0].0 {
                let before = device.memory_usage();
                let mut wrong = case.clone();
                wrong.values.bind(
                    prepared.kernel.schema().dimensions()[0].symbol,
                    SymbolValue::Nat(6_u64.into()),
                );
                assert!(matches!(
                    observer.validate(
                        ObservationRequest {
                            case: &wrong,
                            ..request
                        },
                        corpus_case
                    ),
                    Err(ObservationError::InvalidCase(_))
                ));
                assert_eq!(
                    device.memory_usage(),
                    before,
                    "rejected point releases admitted resources"
                );
                assert!(matches!(
                    observer.validate(
                        ObservationRequest {
                            deadline: Some(Instant::now()),
                            ..request
                        },
                        corpus_case
                    ),
                    Err(ObservationError::DeadlineExpired)
                ));
                assert_eq!(
                    device.memory_usage(),
                    before,
                    "expired observation allocates no resources"
                );
                assert!(matches!(
                    observer.validate(
                        ObservationRequest {
                            memory_limit: 1,
                            ..request
                        },
                        corpus_case
                    ),
                    Err(ObservationError::Capacity { .. })
                ));
                assert_eq!(
                    device.memory_usage(),
                    before,
                    "capacity rejection retains no resources"
                );
            }
            let observation = observer.validate(request, corpus_case).unwrap();
            assert_eq!(observation.reference.inputs().len(), 2);
            assert_eq!(observation.actual.inputs.len(), 2);
            assert!(matches!(
                compare_outcome(
                    &observation.reference,
                    &observation.actual,
                    &PrecisionPolicy::Exact,
                    &mut |_| Ok(())
                )
                .unwrap(),
                Comparison::Match
            ));
        }
    }

    #[test]
    #[ignore = "requires Metal for participant-local portable calls"]
    fn metal_feedback_inlines_portable_reference_calls_in_parallel_segments() {
        let catalog = crate::api::catalog::Catalog::discover().unwrap();
        let info = catalog
            .devices()
            .iter()
            .find(|device| device.backend == registry::BackendName::Metal)
            .unwrap();
        let device = catalog.open(info.id).unwrap();
        let module = check_source(SourceSet::new(vec![SourceFile {
            path: "feedback-reduce.seismic".into(),
            text: "fn sum_values[W](row: tensor[W] f32) -> f32:\n    return reduce(row, 0, sum)\n\nfn row_sums[N,W](x: &tensor[N,W] f32) -> (tensor[N] f32, tensor[N,W] bf16):\n    let mut result = tensor[N] f32\n    let mut copied = tensor[N,W] bf16\n    parallel for row in 0..N:\n        let mut local = to_owned(x[row])\n        let mut total = f32(0.0)\n        for column in 0..W:\n            if local[column] > 0.0:\n                total = total + local[column]\n            else:\n                total = total - local[column]\n            local[column] = local[column] + 1.0\n        copied[row] = local\n        result[row] = sum_values(local) + total\n    return result, copied\n".into(),
        }])).unwrap();
        let entry = module.entry_named("row_sums").unwrap();
        let bindings = ElementBindings::default();
        let logical = module.entry(entry, &bindings).unwrap();
        let mut scope = InvocationScope::for_entry(logical.identity());
        for (ordinal, value) in [2_u64, 4].into_iter().enumerate() {
            scope.constrain(
                InvocationParameter::Dimension(ordinal as u32),
                SymbolValue::Nat(value.into()),
                SymbolValue::Nat(value.into()),
            );
        }
        let kernel = crate::api::kernel::prepare(
            &module,
            entry,
            bindings,
            &device,
            PreparationOptions::feedback(
                PrecisionPolicy::Exact,
                FeedbackOptions {
                    optimize_for: Some(scope),
                    // This checks complete native observations, including cold
                    // compilation of the helper's scalar reference recipes.
                    search_time: Duration::from_secs(30),
                    ..Default::default()
                },
            ),
        )
        .unwrap();
        assert_eq!(
            kernel.feedback_report().unwrap().measured_points,
            1,
            "expected an observation after cold native formation; budget exhaustion and unresolved numerical reasoning are reported separately: {:?}",
            kernel.feedback_report().unwrap()
        );
    }

    #[test]
    #[ignore = "requires Metal to validate many source checks within its argument limit"]
    fn metal_feedback_shares_status_storage_without_dropping_checks() {
        let catalog = crate::api::catalog::Catalog::discover().unwrap();
        let info = catalog
            .devices()
            .iter()
            .find(|device| device.backend == registry::BackendName::Metal)
            .unwrap();
        let device = catalog.open(info.id).unwrap();
        let mut source = "fn checked_sum[N,W](x: &tensor[N,W] f32, indices: &tensor[N] i32) -> tensor[N] f32:\n    let mut result = tensor[N] f32\n    parallel for row in 0..N:\n        let mut total = f32(0.0)\n".to_string();
        for offset in 0..40 {
            source.push_str(&format!(
                "        total = total + x[row, indices[row] + {offset}]\n"
            ));
        }
        source.push_str("        result[row] = total\n    return result\n");
        let module = check_source(SourceSet::new(vec![SourceFile {
            path: "many-checks.seismic".into(),
            text: source,
        }]))
        .unwrap();
        let entry = module.entry_named("checked_sum").unwrap();
        let bindings = ElementBindings::default();
        let logical = module.entry(entry, &bindings).unwrap();
        let mut scope = InvocationScope::for_entry(logical.identity());
        for (ordinal, value) in [2_u64, 64].into_iter().enumerate() {
            scope.constrain(
                InvocationParameter::Dimension(ordinal as u32),
                SymbolValue::Nat(value.into()),
                SymbolValue::Nat(value.into()),
            );
        }
        let kernel = Arc::new(
            crate::api::kernel::prepare(
                &module,
                entry,
                bindings,
                &device,
                PreparationOptions::feedback(
                    PrecisionPolicy::Exact,
                    FeedbackOptions {
                        optimize_for: Some(scope),
                        // Forty checked reads expand substantial reference code;
                        // leave time for observation after cold native formation.
                        search_time: Duration::from_secs(30),
                        ..Default::default()
                    },
                ),
            )
            .unwrap(),
        );
        assert_eq!(
            kernel.feedback_report().unwrap().measured_points,
            1,
            "status observation must run after native formation; a closed time budget and unresolved numerical applicability are different outcomes: {:?}",
            kernel.feedback_report().unwrap()
        );
        let x = Arc::new(
            TensorInner::from_host(
                &device,
                registry::dense(DType::F32),
                &[2, 64],
                &vec![0u8; 2 * 64 * 4],
            )
            .unwrap(),
        );
        // The early indexed reads fit. Later checks must still reject the call.
        let indices = Arc::new(
            TensorInner::from_host(
                &device,
                registry::dense(DType::I32),
                &[2],
                &[32i32.to_le_bytes(), 32i32.to_le_bytes()].concat(),
            )
            .unwrap(),
        );
        let mut args = EncodedArgs::new();
        args.push_tensor(x);
        args.push_tensor(indices);
        assert!(crate::api::kernel::call(&kernel, args).is_err());
    }

    #[test]
    #[ignore = "requires Metal for complete-entry packed gather replay"]
    fn metal_feedback_observes_checked_content_addressing_of_packed_inputs() {
        let catalog = crate::api::catalog::Catalog::discover().unwrap();
        let info = catalog
            .devices()
            .iter()
            .find(|device| device.backend == registry::BackendName::Metal)
            .unwrap();
        let device = catalog.open(info.id).unwrap();
        let module = check_source(SourceSet::new(vec![SourceFile {
            path: "feedback-gather.seismic".into(),
            text: "fn gather[M,V,D](table: &tensor[V,D] Q, indices: &tensor[M] i32) -> tensor[M,D] f32:\n    let mut result = tensor[M,D] f32\n    parallel for row in 0..M:\n        parallel for col in 0..D:\n            result[row,col] = f32(table[indices[row],col])\n    return result\n".into(),
        }])).unwrap();
        let entry = module.entry_named("gather").unwrap();
        let bindings = ElementBindings::new().bind("Q", registry::representation("q4g64").unwrap());
        let logical = module.entry(entry, &bindings).unwrap();
        let mut scope = InvocationScope::for_entry(logical.identity());
        for (ordinal, value) in [2_u64, 3, 64].into_iter().enumerate() {
            scope.constrain(
                InvocationParameter::Dimension(ordinal as u32),
                SymbolValue::Nat(value.into()),
                SymbolValue::Nat(value.into()),
            );
        }
        let kernel = crate::api::kernel::prepare(
            &module,
            entry,
            bindings,
            &device,
            PreparationOptions::feedback(
                PrecisionPolicy::Exact,
                FeedbackOptions {
                    optimize_for: Some(scope),
                    search_time: Duration::from_secs(3),
                    ..Default::default()
                },
            ),
        )
        .unwrap();
        let report = kernel.feedback_report().unwrap();
        assert_eq!(report.measured_points, 1);
        assert!(report.content_dependent_observations > 0);
    }

    #[test]
    #[ignore = "requires an available Metal device; run explicitly for hardware qualification"]
    fn metal_feedback_measures_complete_entries_without_an_analytical_profile() {
        measures_without_profile(registry::BackendName::Metal, false);
    }
    #[test]
    #[ignore = "executes compiled native kernels; run explicitly for hardware qualification"]
    fn cpu_feedback_measures_complete_entries_without_an_analytical_profile() {
        measures_without_profile(registry::BackendName::Cpu, false);
    }
    #[test]
    #[ignore = "executes native kernels and explicit continuation"]
    fn cpu_feedback_continuation_preserves_independent_kernel_snapshots() {
        measures_without_profile(registry::BackendName::Cpu, true);
    }
    #[test]
    #[ignore = "requires CPU native compilation"]
    fn ordinary_and_trial_binding_preserve_tensor_device_identity() {
        let catalog = crate::api::catalog::Catalog::discover().unwrap();
        let info = catalog
            .devices()
            .iter()
            .find(|device| device.backend == registry::BackendName::Cpu)
            .unwrap();
        let device = catalog.open(info.id).unwrap();
        let foreign_catalog = crate::api::catalog::Catalog::discover().unwrap();
        let foreign_info = foreign_catalog
            .devices()
            .iter()
            .find(|device| device.backend == registry::BackendName::Cpu)
            .unwrap();
        let foreign = foreign_catalog.open(foreign_info.id).unwrap();
        assert_ne!(device.kind.identity(), foreign.kind.identity());
        let module = check_source(SourceSet::new(vec![SourceFile {
            path: "binding-device.seismic".into(),
            text: "fn double[N](x: &tensor[N] f32) -> tensor[N] f32:\n    let mut result = zeros_like(x)\n    parallel for i in 0..N:\n        result[i] = x[i] + x[i]\n    return result\n".into(),
        }]))
        .unwrap();
        let kernel = crate::api::kernel::prepare(
            &module,
            module.entry_named("double").unwrap(),
            ElementBindings::default(),
            &device,
            PreparationOptions::feedback(
                PrecisionPolicy::Exact,
                FeedbackOptions {
                    search_time: Duration::ZERO,
                    ..Default::default()
                },
            ),
        )
        .unwrap();
        let kernel = Arc::new(kernel);
        let crate::backends::PreparedKind::Cpu(prepared) = &kernel.inner else {
            panic!()
        };
        let tensor = Arc::new(
            TensorInner::from_host(
                &foreign,
                registry::dense(DType::F32),
                &[1],
                &3.0f32.to_le_bytes(),
            )
            .unwrap(),
        );
        let arguments = || {
            let mut args = EncodedArgs::new();
            args.push_tensor(tensor.clone());
            args
        };
        let error = crate::api::kernel::call(&kernel, arguments())
            .err()
            .expect("foreign tensor rejected");
        assert!(
            matches!(&error, CallError::Invocation(InvocationError::WrongDevice { parameter }) if parameter == "x"),
            "{error:?}"
        );
        assert!(matches!(workflow::native::admit_trial(
            &prepared.prepared.device, &device, prepared.prepared.kernel.variants().as_slice().first().unwrap(),
            arguments(), u64::MAX),
            Err(CallError::Invocation(InvocationError::WrongDevice { parameter })) if parameter == "x"));
        let local = Arc::new(
            TensorInner::from_host(
                &device,
                registry::dense(DType::F32),
                &[1],
                &3.0f32.to_le_bytes(),
            )
            .unwrap(),
        );
        let mut args = EncodedArgs::new();
        args.push_tensor(local);
        let result = crate::api::kernel::call(&kernel, args)
            .unwrap()
            .take_tensor();
        assert_eq!(result.read_to_host().unwrap(), 6.0f32.to_le_bytes());
    }

    fn measures_without_profile(backend: registry::BackendName, continuation: bool) {
        let catalog = crate::api::catalog::Catalog::discover().unwrap();
        let info = catalog
            .devices()
            .iter()
            .find(|device| device.backend == backend)
            .expect("hardware qualification needs the requested backend");
        let device = catalog.open(info.id).unwrap();
        let profile_is_absent = || match &device.kind {
            crate::backends::DeviceKind::Metal(opened) => opened.analytical.get().is_none(),
            crate::backends::DeviceKind::Cpu(opened) => opened.analytical.get().is_none(),
            crate::backends::DeviceKind::Cuda(opened) => opened.analytical.get().is_none(),
        };
        assert!(profile_is_absent());
        let module = check_source(SourceSet::new(vec![SourceFile {
            path: "feedback-add.seismic".into(),
            text: "fn add[N](x: &tensor[N] f32, y: &tensor[N] f32) -> tensor[N] f32 where N >= 0:\n    let mut result = zeros_like(x)\n    parallel for i in 0..N:\n        result[i] = x[i] + y[i]\n    return result\n".into(),
        }])).unwrap();
        let entry = module.entry_named("add").unwrap();
        let bindings = ElementBindings::default();
        let logical = module.entry(entry, &bindings).unwrap();
        let mut scope = InvocationScope::for_entry(logical.identity());
        scope.constrain(
            InvocationParameter::Dimension(0),
            SymbolValue::Nat(4_u64.into()),
            SymbolValue::Nat(5_u64.into()),
        );
        let options = FeedbackOptions {
            optimize_for: Some(scope),
            search_time: Duration::from_secs(3),
            ..Default::default()
        };
        let mut snapshots = Vec::new();
        let kernel = if continuation {
            let (mut campaign, first) = crate::api::kernel::start_feedback(
                &module,
                entry,
                bindings,
                &device,
                PrecisionPolicy::Exact,
                FeedbackOptions {
                    search_time: Duration::ZERO,
                    ..options
                },
            )
            .unwrap();
            let fingerprint = first.feedback_report().unwrap().evaluation_fingerprint;
            assert_eq!(first.feedback_report().unwrap().measured_points, 0);
            let second = campaign.continue_for(Duration::from_secs(3)).unwrap();
            assert_eq!(
                first.feedback_report().unwrap().evaluation_fingerprint,
                fingerprint
            );
            assert_eq!(campaign.report().measured_points, 2);
            drop(campaign);
            snapshots.push(Arc::new(first));
            Arc::new(second)
        } else {
            Arc::new(
                crate::api::kernel::prepare(
                    &module,
                    entry,
                    bindings,
                    &device,
                    PreparationOptions::feedback(PrecisionPolicy::Exact, options),
                )
                .unwrap(),
            )
        };
        assert!(profile_is_absent());
        let report = kernel.feedback_report().unwrap();
        assert_eq!(report.measured_points, 2);
        assert!(report.costs.observation > Duration::ZERO);
        snapshots.push(kernel);
        // The campaign observes only N=4 and N=5. Exercise ordinary dispatch
        // at unseen points on both sides, including empty and multi-group calls.
        for n in [0u64, 1, 6, 17, 257] {
            let values = (0..n).map(|i| i as f32 - 7.0).collect::<Vec<_>>();
            let bytes = values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>();
            let tensor = Arc::new(
                TensorInner::from_host(&device, registry::dense(DType::F32), &[n], &bytes).unwrap(),
            );
            for kernel in &snapshots {
                let mut args = EncodedArgs::new();
                args.push_tensor(tensor.clone());
                args.push_tensor(tensor.clone());
                let output = crate::api::kernel::call(kernel, args)
                    .unwrap()
                    .take_tensor();
                let actual = output.read_to_host().unwrap();
                assert_eq!(actual.len(), bytes.len());
                for (actual, expected) in actual.chunks_exact(4).zip(&values) {
                    assert_eq!(decode(DType::F32, actual), (2.0 * expected) as f64);
                }
            }
        }
        assert!(profile_is_absent());
    }
}

#[cfg(all(test, target_os = "macos"))]
#[path = "feedback/reduction_tests.rs"]
mod reduction_tests;

#[cfg(test)]
#[path = "feedback/literal_tests.rs"]
mod literal_tests;
#[cfg(all(test, target_os = "macos"))]
#[path = "feedback/natural_tests.rs"]
mod natural_tests;

#[cfg(test)]
#[path = "feedback/payload_tests.rs"]
mod payload_tests;

#[cfg(test)]
#[path = "feedback/packed_tests.rs"]
mod packed_tests;

#[cfg(test)]
#[path = "feedback/binding_tests.rs"]
mod binding_tests;

#[cfg(test)]
#[path = "feedback/cohort_tests.rs"]
mod cohort_tests;

#[cfg(test)]
#[path = "feedback/failure_tests.rs"]
mod failure_tests;

#[cfg(all(test, target_os = "macos"))]
#[path = "feedback/evaluator_tests.rs"]
mod evaluator_tests;
