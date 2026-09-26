//! The public-API runner (design A10 §2.2.3).
//!
//! It targets today's `seismic::dynamic` surface: `Kernel::check` returns a
//! `CheckReport` whose `status` is `"passed"` on a match, errors carry a
//! `kind` string, and ranking is a feedback session continued for
//! `RANKING_TIME`. A9-L9.8 rewrites this file to §2.2.3 in the same change
//! that introduces the typed `CheckOutcome`, `dynamic::Error`, `CHECK_LIMITS`
//! and `Kernel::rank` (rule T1): `verdict`, `invalid_invocation`,
//! `refused_index_width` and `prepare` then match typed errors, and
//! `CheckOutcome::Undecided` is accepted exactly when the scenario's
//! `# undecided:` names its reason.
use seismic::dynamic::{
    CheckReport, FeedbackSession, Function, Kernel, Module, Scalar, SignatureType, Tensor,
    TensorAccess, Value,
};
use seismic::{
    BackendName, BigUint, Device, DeviceCatalog, Element, FeedbackOptions, PrecisionPolicy,
    PreparationOptions,
};
use seismic_corpus::inputs::{generate, literal_bits, GeneratedArgument};
use seismic_corpus::scenario::{
    self, ArgumentSpec, Invocation, Literal, Pin, PinSubject, Scenario, ScenarioClass,
    ScenarioName, Termination,
};
use seismic_lang::checked::{check_source, SourceFile, SourceSet};
use seismic_lang::entry::ElementBindings;
use seismic_lang::interp::{Arg, Interpreter, OracleError, OutcomeValue, TensorData, TensorReader};
use seismic_lang::reference_math::ReferenceScalar;
use seismic_lang::registry::{self, RepresentationKind};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

pub enum Selection {
    Baseline,
    Ranked,
}

pub const CHECK_MEMORY_BYTES: u64 = 1 << 30;
pub const CHECK_WORK_UNITS: u64 = 1 << 28;
/// The ranking budget of `Selection::Ranked` (A6 `RankingOptions { time: 2 s, points: 4, seed: 0 }`).
pub const RANKING_TIME: Duration = Duration::from_secs(2);

pub fn corpus_path(relative: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(relative)
}

/// The backend's device; a missing device is a test failure, never a skip.
pub fn device(backend: BackendName) -> Device {
    DeviceCatalog::discover()
        .expect("device discovery")
        .open_backend(backend)
        .unwrap_or_else(|e| panic!("{} device required: {e}", backend.as_str()))
}

pub fn run_public(name: &str, backend: BackendName, selection: Selection) {
    let scenario = scenario::load(&corpus_path("scenarios"), ScenarioName::new(name));
    // build.rs reads `# backends:` itself to generate tests; the parser is the owner.
    assert!(
        scenario.backends.contains(backend),
        "{name}: build.rs generated a {} test, but the scenario's `# backends:` excludes it",
        backend.as_str()
    );
    let label = format!("{name}.seismic");
    let loaded = Module::source(&scenario.source, &label, scenario.include_std, None);
    let invocations = match &scenario.class {
        ScenarioClass::Rejected { diagnostic } => {
            match loaded {
                Err(e) if e.kind == "SourceError" && e.message.contains(diagnostic.as_str()) => {}
                Err(e) => panic!(
                    "{name}: expected a source error containing {diagnostic:?}, got {}: {}",
                    e.kind, e.message
                ),
                Ok(_) => {
                    panic!("{name}: the checker accepts a program it must reject ({diagnostic:?})")
                }
            }
            return;
        }
        ScenarioClass::Executes(invocations) => invocations,
    };
    let module = loaded.unwrap_or_else(|e| panic!("{name}: the module must load: {e}"));
    let device = device(backend);
    let mut failures = Vec::new();
    for policy in scenario.policies.policies() {
        let mut entries: Vec<&str> = Vec::new();
        for invocation in invocations {
            if !entries.contains(&invocation.entry.as_str()) {
                entries.push(&invocation.entry);
            }
        }
        for entry in entries {
            let calls: Vec<&Invocation> = invocations.iter().filter(|i| i.entry == entry).collect();
            let context = format!("{} {entry}", policy_name(&policy));
            let function = match module.function(entry) {
                Ok(function) => function,
                Err(e) => {
                    failures.push(format!("{context}: {e}"));
                    continue;
                }
            };
            let start = Instant::now();
            let preparation = prepare(&function, &device, BTreeMap::new(), &policy, &selection);
            let elapsed = start.elapsed();
            if let Some(bound) = scenario.preparation_bound.filter(|bound| elapsed > *bound) {
                failures.push(format!(
                    "{context}: preparation took {elapsed:?}, bound {bound:?}"
                ));
            }
            let (kernel, session) = match preparation {
                Ok(prepared) => prepared,
                Err(e) => {
                    failures.push(format!("{context}: preparation: {e}"));
                    continue;
                }
            };
            let run = |kernel: &Kernel, stage: &str, failures: &mut Vec<String>| {
                for (index, call) in calls.iter().enumerate() {
                    if let Err(e) =
                        check_call(kernel, &device, &call.arguments, call.termination, &policy)
                    {
                        failures.push(format!("{context} invocation {index}{stage}: {e}"));
                    }
                }
            };
            run(&kernel, "", &mut failures);
            if let Some(mut session) = session {
                match session.continue_for(RANKING_TIME) {
                    Ok(ranked) => run(&ranked, " (ranked)", &mut failures),
                    Err(e) => failures.push(format!("{context}: ranking: {e}")),
                }
            }
        }
    }
    for (index, invocation) in invocations.iter().enumerate() {
        if let Err(e) = check_oracle(&scenario, &device, invocation) {
            failures.push(format!(
                "oracle {} invocation {index}: {e}",
                invocation.entry
            ));
        }
    }
    assert!(failures.is_empty(), "{name}:\n{}", failures.join("\n"));
}

pub fn policy_name(policy: &PrecisionPolicy) -> &'static str {
    match policy {
        PrecisionPolicy::Exact => "exact",
        PrecisionPolicy::Bounded { .. } => "bounded",
        PrecisionPolicy::Unconstrained => "unconstrained",
    }
}

/// Today's preparation: baseline selection is feedback preparation with no
/// search time (the general construction); ranked selection also returns the
/// feedback session, which the caller continues after the first round of checks.
pub fn prepare(
    function: &Function,
    device: &Device,
    elements: BTreeMap<String, Element>,
    policy: &PrecisionPolicy,
    selection: &Selection,
) -> Result<(Kernel, Option<FeedbackSession>), seismic::dynamic::Error> {
    let options = PreparationOptions::feedback(
        policy.clone(),
        FeedbackOptions {
            search_time: Duration::ZERO,
            seed: 0,
            ..FeedbackOptions::default()
        },
    );
    match selection {
        Selection::Baseline => Ok((function.prepare(device, elements, options)?, None)),
        Selection::Ranked => {
            let (session, kernel) = function.start_feedback(device, elements, options)?;
            Ok((kernel, Some(session)))
        }
    }
}

/// Step 3: one differential check of one invocation.
pub fn check_call(
    kernel: &Kernel,
    device: &Device,
    specs: &[ArgumentSpec],
    termination: Option<Termination>,
    policy: &PrecisionPolicy,
) -> Result<(), String> {
    let args = arguments(device, kernel.function(), specs)?;
    if termination == Some(Termination::RefusedIndexWidth) {
        return refused_index_width(kernel.call(&args));
    }
    // An unconstrained preparation admits no value comparison (the observer
    // refuses it); its checkable outcome is the invocation's termination.
    if matches!(policy, PrecisionPolicy::Unconstrained) {
        return unconstrained_termination(kernel.call(&args), termination);
    }
    let checked = kernel.check(&args, policy.clone(), CHECK_MEMORY_BYTES, CHECK_WORK_UNITS);
    if termination == Some(Termination::InvalidInvocation) {
        return invalid_invocation(checked);
    }
    verdict(&checked.map_err(|e| format!("{}: {}", e.kind, e.message))?)
}

/// A7 `ResourceRefusal::IndexWidth`: a device limit, so neither the differential
/// check nor the interpreter runs (the source outcome may exceed the reference budget).
fn refused_index_width(called: Result<Value, seismic::dynamic::Error>) -> Result<(), String> {
    match called {
        Err(e) if e.message.contains("IndexWidth") => Ok(()),
        Err(e) => Err(format!(
            "expected an IndexWidth refusal, got {}: {}",
            e.kind, e.message
        )),
        Ok(_) => Err("expected an IndexWidth refusal, the call returned".into()),
    }
}

fn unconstrained_termination(
    called: Result<Value, seismic::dynamic::Error>,
    termination: Option<Termination>,
) -> Result<(), String> {
    let source_failure = |e: &seismic::dynamic::Error| {
        e.kind == "ExecutionError" && e.message.starts_with("check failed at ")
    };
    match (called, termination) {
        (Ok(_), None | Some(Termination::Returned)) => Ok(()),
        (Err(e), None | Some(Termination::Failed)) if source_failure(&e) => Ok(()),
        (Err(e), Some(Termination::InvalidInvocation)) if e.kind == "InvocationError" => Ok(()),
        (Ok(_), expected) => Err(format!("expected {expected:?}, the call returned")),
        (Err(e), expected) => Err(format!(
            "expected {expected:?}, got {}: {}",
            e.kind, e.message
        )),
    }
}

fn verdict(report: &CheckReport) -> Result<(), String> {
    if report.status == "passed" {
        Ok(())
    } else {
        Err(format!("check {}: {}", report.status, report.diagnostic))
    }
}

fn invalid_invocation(checked: Result<CheckReport, seismic::dynamic::Error>) -> Result<(), String> {
    match checked {
        Err(e) if e.kind == "InvocationError" => Ok(()),
        Err(e) => Err(format!(
            "expected an invalid invocation, got {}: {}",
            e.kind, e.message
        )),
        Ok(report) => Err(format!(
            "expected an invalid invocation, got check {}: {}",
            report.status, report.diagnostic
        )),
    }
}

/// The public tensor storage size of `element[shape]` (C15-5: never computed by the corpus).
fn storage_bytes(device: &Device, element: &str, shape: &[u64]) -> Result<u64, String> {
    seismic::Tensor::zeros(device, element_named(element), shape)
        .map(|tensor| tensor.storage_bytes())
        .map_err(|e| format!("{element}{shape:?}: {e}"))
}

pub fn element_named(name: &str) -> Element {
    Element::named(name).unwrap_or_else(|| panic!("`{name}` is not a registered element"))
}

fn generate_all(device: &Device, specs: &[ArgumentSpec]) -> Result<Vec<GeneratedArgument>, String> {
    specs
        .iter()
        .map(|spec| generate(spec, |element, shape| storage_bytes(device, element, shape)))
        .collect()
}

/// Builds the entry's argument values from flat specs, in parameter order;
/// tuple parameters consume one spec per leaf. Type agreement is the API's to check.
pub fn arguments(
    device: &Device,
    function: &Function,
    specs: &[ArgumentSpec],
) -> Result<Vec<Value>, String> {
    let mut generated = generate_all(device, specs)?.into_iter();
    let values = function
        .parameters()
        .iter()
        .map(|(name, ty)| value(device, name, ty, &mut generated))
        .collect::<Result<Vec<_>, _>>()?;
    assert!(
        generated.next().is_none(),
        "{} takes fewer arguments than the invocation lists",
        function.name()
    );
    Ok(values)
}

fn value(
    device: &Device,
    parameter: &str,
    ty: &SignatureType,
    generated: &mut impl Iterator<Item = GeneratedArgument>,
) -> Result<Value, String> {
    Ok(match ty {
        SignatureType::Unit => Value::Unit,
        SignatureType::Tuple(types) => Value::Tuple(
            types
                .iter()
                .map(|ty| value(device, parameter, ty, generated))
                .collect::<Result<_, _>>()?,
        ),
        _ => match generated
            .next()
            .unwrap_or_else(|| panic!("the invocation has no argument for `{parameter}`"))
        {
            GeneratedArgument::Tensor {
                element,
                shape,
                bytes,
            } => {
                let tensor = Tensor::from_host(device, element_named(&element), &shape, &bytes)
                    .map_err(|e| format!("{parameter}: {e}"))?;
                if matches!(
                    ty,
                    SignatureType::Tensor {
                        access: TensorAccess::Owned,
                        ..
                    }
                ) {
                    Value::Move(tensor)
                } else {
                    Value::Tensor(tensor)
                }
            }
            GeneratedArgument::Scalar { dtype, bits } => Value::Scalar(match dtype {
                scenario::ScalarDtype::F32 => Scalar::F32(bits as u32),
                scenario::ScalarDtype::F16 => Scalar::F16(bits as u16),
                scenario::ScalarDtype::Bf16 => Scalar::BF16(bits as u16),
                scenario::ScalarDtype::I32 => Scalar::I32(bits as u32 as i32),
                scenario::ScalarDtype::U32 => Scalar::U32(bits as u32),
                scenario::ScalarDtype::Bool => Scalar::Bool(bits != 0),
            }),
            GeneratedArgument::Index(value) => Value::Scalar(Scalar::Index(BigUint::from(value))),
            GeneratedArgument::Range { start, end } => {
                Value::Scalar(Scalar::Range(BigUint::from(start), BigUint::from(end)))
            }
        },
    })
}

/// Step 5: the interpreter's own termination class and pinned values, compared
/// bit for bit. This checks the oracle, which the differential check cannot.
fn check_oracle(
    scenario: &Scenario,
    device: &Device,
    invocation: &Invocation,
) -> Result<(), String> {
    match invocation.termination {
        None if invocation.pins.is_empty() => return Ok(()),
        // A device limit, not a source outcome: the interpreter is not run.
        Some(Termination::RefusedIndexWidth) => return Ok(()),
        _ => {}
    }
    let mut sources = if scenario.include_std {
        seismic_std::sources()
    } else {
        SourceSet::default()
    };
    sources.push(SourceFile {
        path: format!("{}.seismic", scenario.name),
        text: scenario.source.clone(),
    });
    let checked = check_source(sources).map_err(|e| e.to_string())?;
    let info = checked
        .entries()
        .iter()
        .find(|info| info.name == invocation.entry)
        .ok_or_else(|| format!("no entry `{}`", invocation.entry))?;
    let entry = checked
        .entry(info.id, &ElementBindings::new())
        .map_err(|e| e.to_string())?;
    let mut interpreter = Interpreter::new(&entry);
    let mut oracle_args = Vec::new();
    for argument in generate_all(device, &invocation.arguments)? {
        oracle_args.push(match argument {
            GeneratedArgument::Tensor {
                element,
                shape,
                bytes,
            } => {
                let representation = registry::representation(&element).expect("parsed element");
                let shape = shape
                    .iter()
                    .map(|extent| usize::try_from(*extent).map_err(|e| e.to_string()))
                    .collect::<Result<Vec<_>, _>>()?;
                let data = match registry::representation_info(representation).kind {
                    RepresentationKind::Dense(dtype) => {
                        TensorData::dense_from_bytes(dtype, shape, bytes)
                    }
                    _ => TensorData::encoded(representation, shape, bytes),
                }?;
                Arg::Tensor(interpreter.add_tensor(data))
            }
            GeneratedArgument::Scalar { dtype, bits } => {
                Arg::Scalar(ReferenceScalar::from_bits(dtype.dtype(), bits as u32))
            }
            GeneratedArgument::Index(value) => Arg::Index(BigUint::from(value)),
            GeneratedArgument::Range { start, end } => {
                Arg::Range(BigUint::from(start), BigUint::from(end))
            }
        });
    }
    let outcome = match interpreter.run_bounded_with_memory(
        &oracle_args,
        CHECK_WORK_UNITS,
        CHECK_MEMORY_BYTES,
    ) {
        Err(OracleError::InvalidInvocation(reason)) => {
            return match (invocation.termination, invocation.pins.is_empty()) {
                (Some(Termination::InvalidInvocation), true) => Ok(()),
                _ => Err(format!("the interpreter rejects the invocation: {reason}")),
            };
        }
        Err(e) => return Err(e.to_string()),
        Ok(outcome) => outcome,
    };
    let termination = match outcome.termination() {
        seismic_lang::failure::SourceTermination::Returned(_) => Termination::Returned,
        seismic_lang::failure::SourceTermination::Failed(_) => Termination::Failed,
    };
    if let Some(expected) = invocation
        .termination
        .filter(|expected| *expected != termination)
    {
        return Err(format!(
            "the interpreter terminates {termination:?}, expected {expected:?}"
        ));
    }
    for pin in &invocation.pins {
        let actual = match &pin.subject {
            PinSubject::Parameter(name) => {
                let ordinal = parameter_ordinal(&info.parameter_types, name)
                    .ok_or_else(|| format!("`{name}` is not a tensor parameter"))?;
                let input = outcome
                    .inputs()
                    .find(|input| input.ordinal() == ordinal)
                    .ok_or_else(|| format!("the interpreter observed no final `{name}`"))?;
                let reader = input.tensor();
                let bits = tensor_bits(pin, &reader)?;
                bits
            }
            PinSubject::Result(path) => {
                let result = outcome
                    .results()
                    .find(|result| result.path() == path.as_slice())
                    .ok_or_else(|| format!("the interpreter returned no result at {path:?}"))?;
                let value = result.value();
                let bits = match value {
                    OutcomeValue::Tensor(reader) => tensor_bits(pin, &reader)?,
                    OutcomeValue::Scalar(scalar) => {
                        if !pin.shape.is_empty()
                            || registry::dense(scalar.dtype()) != pin_representation(pin)
                        {
                            return Err(format!(
                                "result {path:?} is a {} scalar",
                                scalar.dtype().name()
                            ));
                        }
                        vec![u64::from(scalar.bits())]
                    }
                    OutcomeValue::Index(_) | OutcomeValue::Range(..) => {
                        return Err(format!("result {path:?} is not an element value"))
                    }
                };
                bits
            }
        };
        let expected = pin_bits(pin);
        if actual != expected {
            return Err(format!(
                "pin {:?}: the interpreter gives {}, expected {}",
                pin.subject,
                hex(&actual),
                hex(&expected)
            ));
        }
    }
    Ok(())
}

/// The flat argument ordinal of a top-level tensor parameter.
fn parameter_ordinal(parameters: &[(String, SignatureType)], name: &str) -> Option<usize> {
    fn leaves(ty: &SignatureType) -> usize {
        match ty {
            SignatureType::Unit => 0,
            SignatureType::Tuple(types) => types.iter().map(leaves).sum(),
            _ => 1,
        }
    }
    let mut ordinal = 0;
    for (parameter, ty) in parameters {
        if parameter == name {
            return matches!(ty, SignatureType::Tensor { .. }).then_some(ordinal);
        }
        ordinal += leaves(ty);
    }
    None
}

fn pin_representation(pin: &Pin) -> seismic_lang::ids::RepresentationId {
    registry::representation(&pin.element).expect("parsed element")
}

fn pin_dtype(pin: &Pin) -> seismic_lang::types::DType {
    match registry::representation_info(pin_representation(pin)).kind {
        RepresentationKind::Dense(dtype) => dtype,
        _ => unreachable!("the parser admits only dense pins"),
    }
}

fn pin_bits(pin: &Pin) -> Vec<u64> {
    pin.values
        .iter()
        .map(|value| literal_bits(pin_dtype(pin), *value).expect("validated by the parser"))
        .collect()
}

/// Every element's bit pattern, after checking representation and shape.
fn tensor_bits(pin: &Pin, reader: &TensorReader<'_>) -> Result<Vec<u64>, String> {
    let shape: Vec<u64> = reader.shape().iter().map(|extent| *extent as u64).collect();
    if reader.representation() != pin_representation(pin) || shape != pin.shape {
        return Err(format!(
            "the interpreter's value is {}{shape:?}, the pin states {}{:?}",
            registry::representation_info(reader.representation()).name,
            pin.element,
            pin.shape
        ));
    }
    let dtype = pin_dtype(pin);
    let width = dtype.bytes() as usize;
    if let Some(bytes) = reader.canonical_bytes()? {
        return Ok(bytes
            .chunks_exact(width)
            .map(|element| {
                let mut word = [0; 8];
                word[..width].copy_from_slice(element);
                u64::from_le_bytes(word)
            })
            .collect());
    }
    (0..reader.element_count())
        .map(|index| {
            let value = reader.read(index)?;
            // Views expose only element values, and a NaN payload does not survive
            // the f64 read; such an element cannot be compared bit for bit.
            if value.is_nan() {
                return Err(format!(
                    "element {index} of the interpreter's view result is NaN, whose payload a \
                     view does not expose; a pin on a view result must not cover NaN elements"
                ));
            }
            let literal = if dtype.is_float() {
                Literal::Decimal(value)
            } else if dtype == seismic_lang::types::DType::Bool {
                Literal::Bool(value != 0.0)
            } else {
                Literal::Integer(value as i128)
            };
            literal_bits(dtype, literal)
        })
        .collect()
}

fn hex(bits: &[u64]) -> String {
    let words: Vec<String> = bits.iter().map(|bits| format!("{bits:#x}")).collect();
    format!("[{}]", words.join(", "))
}
