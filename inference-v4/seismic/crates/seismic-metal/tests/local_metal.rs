//! One targeted local Metal execution: construct → family → plan → encode →
//! execute representative kernels on this device and assert bit-agreement
//! with the reference interpreter.

#![cfg(target_os = "macos")]

use seismic_compiler::pipeline::{self, Workload};
use seismic_compiler::planning::Budget;
use seismic_lang::{
    interp::{Arg, Bindings, Interpreter, TensorData},
    logical::EffectiveTargetIdentity,
    precision::PrecisionPolicy,
    program::{self, SourceFile},
    sir::Program,
};
use seismic_metal::runtime::{Device, Invocation};
use seismic_metal::{mapping::MetalCompiler, physical::MetalDialect};

/// The representative kernels: an exclusive-borrow inout tensor and a
/// moved-in owned tensor returned as the result, each with an independent
/// outer loop, an ordered inner loop with the carried tensor state, and
/// point reads and writes through views.
const NEGATE: &str = "fn negate[M, N](io: &mut tensor[M, N] f32):\n    parallel for row in 0..M:\n        for col in 0..N:\n            io[row, col] = 0.0 - io[row, col]\n    return\n";

const NEGATE_OWNED: &str = "fn negate[M, N](io: tensor[M, N] f32) -> tensor[M, N] f32:\n    let mut o = io\n    parallel for row in 0..M:\n        for col in 0..N:\n            o[row, col] = 0.0 - o[row, col]\n    return o\n";

fn check(source: &str) -> Result<Program, String> {
    let files = vec![SourceFile {
        path: "kernel.seismic".into(),
        text: source.to_string(),
    }];
    program::compile(&files).map_err(|diagnostics| {
        diagnostics
            .iter()
            .map(|d| d.render())
            .collect::<Vec<_>>()
            .join("\n")
    })
}

fn input_data(rows: usize, columns: usize) -> Vec<f32> {
    let mut data = Vec::with_capacity(rows * columns);
    let mut state = 0x9e3779b97f4a7c15u64;
    for _ in 0..rows * columns {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let unit = (state % 1_000_003) as f32 / 1_000_003.0;
        data.push(unit * 4.0 - 2.0);
    }
    data
}

#[test]
fn local_metal_execution_agrees_bitwise_with_the_reference_interpreter() {
    let (rows, columns) = (7usize, 5usize);
    let program = check(NEGATE).expect("the kernel checks");
    // Reference: the interpreter evaluates the reference body.
    let mut interpreter = Interpreter::new(&program);
    let data = input_data(rows, columns);
    let tensor = TensorData::Dense {
        dtype: seismic_lang::types::DType::F32,
        shape: vec![rows, columns],
        data: data.iter().map(|v| f64::from(*v)).collect(),
    };
    let reference_id = interpreter.add_tensor(tensor);
    interpreter
        .run(
            "negate",
            &[Arg::Tensor(reference_id)],
            &Bindings {
                shapes: [("M", rows as i64), ("N", columns as i64)]
                    .into_iter()
                    .map(|(k, v)| (k.to_string(), v))
                    .collect(),
                elems: Default::default(),
            },
        )
        .expect("the interpreter runs the reference body");
    // The inout tensor is updated in place (the void result carries no data).
    let expected: Vec<u32> = (0..rows * columns)
        .map(|index| interpreter.tensors[reference_id].get(index) as f32)
        .map(|v: f32| v.to_bits())
        .collect();
    // Native: construct → elaborate → plan → encode → assemble → execute.
    let device = Device::open().expect("a local Metal device exists");
    let compiler = MetalCompiler::from_device(&device).expect("the device profile builds");
    let compiled: pipeline::Compiled<MetalDialect, seismic_metal::runtime::Pipeline> =
        pipeline::compile(
            &program,
            "negate",
            &Workload {
                shapes: [("M", rows as i64), ("N", columns as i64)]
                    .into_iter()
                    .map(|(k, v)| (k.to_string(), v))
                    .collect(),
                elems: Default::default(),
                precision: PrecisionPolicy::default(),
                extents: Default::default(),
            },
            &compiler,
            &[],
            Budget::default(),
        )
        .expect("the representative kernel compiles for the local device");
    // The logical specialization targeted this device.
    assert_eq!(
        compiled.logical.target,
        EffectiveTargetIdentity {
            backend: "metal".into(),
            capability_fingerprint: seismic_compiler::pipeline::Backend::capability_fingerprint(
                &compiler
            ),
        }
    );
    // Caller-supplied ABI parameter buffer: the input tensor bits.
    let input_bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    let input = device.buffer_from(&input_bytes).expect("allocates");
    let pipeline = &compiled.native;
    let outcome = device
        .run(&Invocation {
            pipeline,
            buffers: vec![&input],
            scalars: Vec::new(),
        })
        .expect("the native execution completes");
    assert!(outcome.dispatches > 0, "work was submitted");
    // A void inout kernel: the caller's parameter buffer holds the result.
    assert!(outcome.results.is_empty());
    let bytes = input.read(rows * columns * 4);
    let observed: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
        .collect();
    assert_eq!(observed.len(), expected.len());
    for (index, (observed, expected)) in observed.iter().zip(&expected).enumerate() {
        assert_eq!(
            observed, expected,
            "element#{index}: native {observed:#010x} vs reference {expected:#010x}"
        );
    }
}

#[test]
fn local_metal_owned_result_execution_agrees_bitwise() {
    // A moved-in owned parameter returned as the result: the entry result
    // resolves to a public ABI result buffer distinct from the parameter.
    let (rows, columns) = (5usize, 3usize);
    let program = check(NEGATE_OWNED).expect("the kernel checks");
    let data = input_data(rows, columns);
    let mut interpreter = Interpreter::new(&program);
    let tensor = TensorData::Dense {
        dtype: seismic_lang::types::DType::F32,
        shape: vec![rows, columns],
        data: data.iter().map(|v| f64::from(*v)).collect(),
    };
    let reference_id = interpreter.add_tensor(tensor);
    interpreter
        .run(
            "negate",
            &[Arg::Tensor(reference_id)],
            &Bindings {
                shapes: [("M", rows as i64), ("N", columns as i64)]
                    .into_iter()
                    .map(|(k, v)| (k.to_string(), v))
                    .collect(),
                elems: Default::default(),
            },
        )
        .expect("the interpreter runs the reference body");
    // The reference body computes `0.0 - x` in f64 and rounds once to f32;
    // for f32 inputs that is exactly `(0.0 - x) as f32`.
    let expected: Vec<u32> = data
        .iter()
        .map(|v| (0.0 - f64::from(*v)) as f32)
        .map(|v: f32| v.to_bits())
        .collect();
    let device = Device::open().expect("a local Metal device exists");
    let compiler = MetalCompiler::from_device(&device).expect("the device profile builds");
    let compiled: pipeline::Compiled<MetalDialect, seismic_metal::runtime::Pipeline> =
        pipeline::compile(
            &program,
            "negate",
            &Workload {
                shapes: [("M", rows as i64), ("N", columns as i64)]
                    .into_iter()
                    .map(|(k, v)| (k.to_string(), v))
                    .collect(),
                elems: Default::default(),
                precision: PrecisionPolicy::default(),
                extents: Default::default(),
            },
            &compiler,
            &[],
            Budget::default(),
        )
        .expect("the owned-result kernel compiles for the local device");
    let input_bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    let input = device.buffer_from(&input_bytes).expect("allocates");
    let outcome = device
        .run(&Invocation {
            pipeline: &compiled.native,
            buffers: vec![&input],
            scalars: Vec::new(),
        })
        .expect("the native execution completes");
    assert!(outcome.dispatches > 0, "work was submitted");
    // The result is a distinct runtime-allocated ABI buffer.
    assert_eq!(outcome.results.len(), 1);
    let bytes = outcome.results[0].read(rows * columns * 4);
    let observed: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
        .collect();
    assert_eq!(observed, expected);
}
