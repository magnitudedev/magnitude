//! Resource tests through the public API (design A10 §2.9.1).
//!
//! The scenario format cannot state a memory budget, and some construction
//! decisions are observable only under a small one: whether a refusal is actual
//! (DoD-3) and whether a launch's resource demand precedes a failure (DoD-4).
//! Each test builds its inputs, reads the device's charge `base`, limits the
//! device to `base + budget`, prepares under `exact`, calls, compares the
//! outcome with the interpreter's on the same source and inputs, asserts the
//! stated values, then asserts that every run-owned backing was released
//! (`charged == base`, IF §2.8) and restores the limit.
//!
//! Every test opens its own device, and an open creates a private memory
//! domain, so concurrently running tests never share a limit or a charge.
//! Budgets keep at least 4 KiB of margin for the run's own small acquisitions
//! (failure record, scalar cells; each at most 64 bytes, IF §2.6).
use crate::common::{device, prepare, Selection, CHECK_MEMORY_BYTES, CHECK_WORK_UNITS};
use seismic::dynamic::{Module, Scalar, Tensor, Value};
use seismic::{BackendName, Element, PrecisionPolicy};
use seismic_lang::checked::{check_source, SourceFile, SourceSet};
use seismic_lang::entry::ElementBindings;
use seismic_lang::failure::SourceTermination;
use seismic_lang::interp::{Arg, Interpreter, TensorData};
use seismic_lang::reference_math::ReferenceScalar;
use seismic_lang::types::DType;
use std::collections::BTreeMap;

const MODULE_PATH: &str = "resources.seismic";

const MODULE: &str = "\
fn unvisited_iterations(out: &mut tensor[1] i32, divisor: i32):
    for i in 0..65536:
        let mut scratch = tensor[i+1] i32
        scratch[0] = 7
        out[0] = scratch[0]
        let quotient = 1 / divisor
        out[0] = quotient
fn backing_after_failure(out: &mut tensor[1] i32, divisor: i32):
    let mut first = tensor[1] i32
    first[0] = 7
    out[0] = first[0]
    let quotient = 1 / divisor
    let mut later = tensor[65536] i32
    later[0] = quotient
    out[0] = later[0]
fn participant_copies[R, N](x: &tensor[N] f32, s: &tensor[R] f32, out: &mut tensor[R] f32):
    for i in 0..R:
        let t = to_owned(x)
        out[i] = reduce(t, 0, sum) + s[i]
";

/// One entry argument, built on the device and in the interpreter from the same bytes.
enum Input {
    Tensor {
        dtype: DType,
        element: Element,
        shape: Vec<u64>,
        bytes: Vec<u8>,
    },
    I32(i32),
}

fn i32_tensor(values: &[i32]) -> Input {
    Input::Tensor {
        dtype: DType::I32,
        element: Element::i32(),
        shape: vec![values.len() as u64],
        bytes: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
    }
}

fn f32_tensor(values: &[f32]) -> Input {
    Input::Tensor {
        dtype: DType::F32,
        element: Element::f32(),
        shape: vec![values.len() as u64],
        bytes: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
    }
}

fn i32_bytes(values: &[i32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// How a budgeted run ended, after it agreed with the interpreter.
struct Run {
    failed: bool,
    /// The final bytes of each tensor argument, in argument order.
    tensors: Vec<Vec<u8>>,
}

/// The interpreter's termination (`Some(message)` when the source failed) and the
/// final bytes of each tensor argument it reports, keyed by argument ordinal.
fn interpret(entry: &str, inputs: &[Input]) -> (Option<String>, Vec<(usize, Vec<u8>)>) {
    let mut sources = SourceSet::default();
    sources.push(SourceFile {
        path: MODULE_PATH.into(),
        text: MODULE.into(),
    });
    let checked = check_source(sources).unwrap_or_else(|e| panic!("the resource module must check: {e}"));
    let info = checked
        .entries()
        .iter()
        .find(|info| info.name == entry)
        .unwrap_or_else(|| panic!("no entry `{entry}`"));
    let monomorphized = checked
        .entry(info.id, &ElementBindings::new())
        .unwrap_or_else(|e| panic!("{entry}: {e}"));
    let mut interpreter = Interpreter::new(&monomorphized);
    let arguments: Vec<Arg> = inputs
        .iter()
        .map(|input| match input {
            Input::Tensor {
                dtype, shape, bytes, ..
            } => {
                let shape = shape.iter().map(|extent| *extent as usize).collect();
                let data = TensorData::dense_from_bytes(*dtype, shape, bytes.clone())
                    .unwrap_or_else(|e| panic!("{entry}: {e}"));
                Arg::Tensor(interpreter.add_tensor(data))
            }
            Input::I32(value) => Arg::Scalar(ReferenceScalar::from_bits(DType::I32, *value as u32)),
        })
        .collect();
    let outcome = interpreter
        .run_bounded_with_memory(&arguments, CHECK_WORK_UNITS, CHECK_MEMORY_BYTES)
        .unwrap_or_else(|e| panic!("{entry}: the interpreter: {e}"));
    let failure = match outcome.termination() {
        SourceTermination::Returned(_) => None,
        SourceTermination::Failed(failure) => Some(failure.to_string()),
    };
    let tensors = outcome
        .inputs()
        .map(|input| {
            let reader = input.tensor();
            let bytes = reader
                .canonical_bytes()
                .unwrap_or_else(|e| panic!("{entry}: {e}"))
                .expect("a dense argument tensor has canonical bytes")
                .to_vec();
            (input.ordinal(), bytes)
        })
        .collect();
    (failure, tensors)
}

/// Prepares and calls `entry` on `backend` with the device limited to
/// `charged + budget`, checks the outcome and every tensor argument against the
/// interpreter, and checks that the run released everything it acquired.
fn run_under_budget(backend: BackendName, entry: &str, inputs: &[Input], budget: u64) -> Run {
    let device = device(backend);
    let module = Module::source(MODULE, MODULE_PATH, false, None)
        .unwrap_or_else(|e| panic!("the resource module must load: {e}"));
    let values: Vec<Value> = inputs
        .iter()
        .map(|input| match input {
            Input::Tensor {
                element, shape, bytes, ..
            } => Value::Tensor(
                Tensor::from_host(&device, element.clone(), shape, bytes)
                    .unwrap_or_else(|e| panic!("{entry}: {e}")),
            ),
            Input::I32(value) => Value::Scalar(Scalar::I32(*value)),
        })
        .collect();
    let usage = device.memory_usage();
    let base = usage.charged;
    device
        .set_memory_limit(Some(base + budget))
        .unwrap_or_else(|e| panic!("{entry}: {e}"));
    let function = module.function(entry).expect("declared entry");
    let (kernel, _) = prepare(
        &function,
        &device,
        BTreeMap::new(),
        &PrecisionPolicy::Exact,
        &Selection::Baseline,
    )
    .unwrap_or_else(|e| panic!("{entry}: preparation under a {budget}-byte budget: {e}"));
    let called = kernel.call(&values);
    let (oracle_failure, oracle_tensors) = interpret(entry, inputs);
    let failed = match (&oracle_failure, &called) {
        (None, Ok(_)) => false,
        // Today's surface: a source failure is an `ExecutionError` whose message is
        // the located check (`ExecutionError::DataCheckFailed`); A9-L9.8 rewrites this
        // arm to `Err(Error::Call(CallError::SourceFailure(_)))` (T1).
        (Some(cause), Err(error))
            if error.kind == "ExecutionError"
                && error.message.starts_with("check failed at")
                && error.message.contains(cause.as_str()) =>
        {
            true
        }
        (None, Err(error)) => panic!(
            "{entry}: the interpreter returns, the call under a {budget}-byte budget gives {}: {}",
            error.kind, error.message
        ),
        (Some(cause), Err(error)) => panic!(
            "{entry}: the interpreter fails with `{cause}`, the call under a {budget}-byte budget gives {}: {}",
            error.kind, error.message
        ),
        (Some(cause), Ok(_)) => panic!("{entry}: the interpreter fails with `{cause}`, the call returned"),
    };
    let tensors: Vec<Vec<u8>> = values
        .iter()
        .filter_map(|value| match value {
            Value::Tensor(tensor) => Some(tensor.read().unwrap_or_else(|e| panic!("{entry}: {e}"))),
            _ => None,
        })
        .collect();
    let ordinals: Vec<usize> = inputs
        .iter()
        .enumerate()
        .filter_map(|(ordinal, input)| matches!(input, Input::Tensor { .. }).then_some(ordinal))
        .collect();
    for (ordinal, expected) in &oracle_tensors {
        let position = ordinals
            .iter()
            .position(|tensor_ordinal| tensor_ordinal == ordinal)
            .unwrap_or_else(|| panic!("{entry}: argument {ordinal} is not a tensor"));
        assert_eq!(
            &tensors[position], expected,
            "{entry}: argument {ordinal} differs from the interpreter's final value"
        );
    }
    assert_eq!(
        device.memory_usage().charged,
        base,
        "{entry}: the run did not release every backing it acquired"
    );
    device
        .set_memory_limit(usage.limit)
        .unwrap_or_else(|e| panic!("{entry}: restoring the limit: {e}"));
    Run { failed, tensors }
}

/// R1: a reached allocation does not reserve unvisited iterations (IF §2.8; A3).
/// Iteration 0 allocates 4 bytes, writes 7 and fails on `1 / 0`; the loop's
/// envelope (Σ 4·(i+1) over 65536 iterations, far above 4 KiB) must never be
/// reserved. From `git show ab921348:./seismic/runtime/src/feedback/failure_tests.rs:179`.
fn reached_allocation_does_not_reserve_unvisited_iterations(backend: BackendName) {
    let run = run_under_budget(
        backend,
        "unvisited_iterations",
        &[i32_tensor(&[0]), Input::I32(0)],
        4096,
    );
    assert!(run.failed, "`1 / divisor` with divisor 0 is a source failure");
    assert_eq!(run.tensors[0], i32_bytes(&[7]));
}

/// R2: no backing is acquired after a source failure (IF §2.6.2 precedence; A3, A7).
/// The 256 KiB `later` allocation follows the failing `1 / divisor` and must never
/// be acquired. From `failure_tests.rs:184` at ab921348.
fn no_backing_after_source_failure(backend: BackendName) {
    let run = run_under_budget(
        backend,
        "backing_after_failure",
        &[i32_tensor(&[0]), Input::I32(0)],
        4096,
    );
    assert!(run.failed, "`1 / divisor` with divisor 0 is a source failure");
    assert_eq!(run.tensors[0], i32_bytes(&[7]));
}

/// `participant_copies` with `x` all 1.0 and `s[i] = i`: `out[i] == N + i`.
fn participant_copies(backend: BackendName, rows: usize, columns: usize, budget: u64) {
    let s: Vec<f32> = (0..rows).map(|i| i as f32).collect();
    let run = run_under_budget(
        backend,
        "participant_copies",
        &[f32_tensor(&vec![1.0; columns]), f32_tensor(&s), f32_tensor(&vec![0.0; rows])],
        budget,
    );
    assert!(!run.failed, "participant_copies has no failing operation");
    let expected: Vec<f32> = (0..rows).map(|i| (columns + i) as f32).collect();
    assert_eq!(run.tensors[2], f32_bytes(&expected));
}

/// R3: an `IndependentTotal` loop runs by lowering participants per group (IF I-42; A3).
/// Shapes assume Metal's `general_workgroup()` of 32 (A4's definition: Metal 32, CPU 1):
/// N = 4096 gives 16 KiB per participant, so one group of 32 needs 512 KiB, which a
/// 64 KiB budget refuses, while one participant fits. The launch runs only if the
/// grant lowers participants per group; on CPU it runs through the same grant.
fn independent_loop_reduces_participants(backend: BackendName) {
    participant_copies(backend, 64, 4096, 64 * 1024);
}

/// R4: participant scratch splits the dispatch (IF I-12, §2.7.4; A3).
/// Shapes assume Metal's `general_workgroup()` of 32: N = 256 gives 1 KiB per
/// participant and 32 KiB per group; R = 128 is 4 groups (128 KiB), which a 48 KiB
/// budget refuses, while one group fits. The launch runs as ascending dispatch ranges
/// with `participant_base = first_group × p`; a wrong base writes the wrong `out` rows.
fn participant_scratch_splits_dispatch(backend: BackendName) {
    participant_copies(backend, 128, 256, 48 * 1024);
}

macro_rules! resource_tests {
    ($($test:ident),* $(,)?) => {
        mod cpu {
            $(#[test]
            fn $test() {
                super::$test(seismic::BackendName::Cpu)
            })*
        }
        #[cfg(target_os = "macos")]
        mod metal {
            $(#[test]
            fn $test() {
                super::$test(seismic::BackendName::Metal)
            })*
        }
    };
}

resource_tests!(
    reached_allocation_does_not_reserve_unvisited_iterations,
    no_backing_after_source_failure,
    independent_loop_reduces_participants,
    participant_scratch_splits_dispatch,
);
