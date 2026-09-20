//! End-to-end integration: construct → elaborate → plan → encode → execute
//! on the CPU backend, asserting bit-agreement with the seismic-lang
//! reference interpreter for kernels whose dataflow the resolved artifact
//! expresses (elementwise read/compute/write inside an independent domain).

use seismic_compiler::pipeline::{self, Backend as _, Workload};
use seismic_cpu::{Buffer, Cpu, InvocationFailure, Workers, TARGET};
use seismic_lang::{
    interp::{Arg, Bindings, Interpreter, TensorData},
    precision::PrecisionPolicy,
    program::{compile, SourceFile},
    types::DType,
};
use std::collections::BTreeMap;

fn compile_program(sources: &[&str]) -> seismic_lang::sir::Program {
    compile(
        &sources
            .iter()
            .enumerate()
            .map(|(index, source)| SourceFile {
                path: format!("test-{index}.seismic"),
                text: (*source).to_string(),
            })
            .collect::<Vec<_>>(),
    )
    .expect("the test program checks")
}

fn workload(shapes: &[(&str, i64)]) -> Workload {
    Workload {
        shapes: shapes
            .iter()
            .map(|(name, value)| ((*name).to_string(), *value))
            .collect(),
        elems: BTreeMap::new(),
        precision: PrecisionPolicy::default(),
        extents: BTreeMap::new(),
    }
}

#[test]
fn elementwise_write_agrees_with_the_reference_interpreter() {
    let source = "\
fn scale[N](x: &tensor[N] f32, y: &mut tensor[N] f32, a: f32):
    parallel for i in 0..N:
        y[i] = x[i] * a + 0.5 / a
    return
";
    let program = compile_program(&[source]);
    let n = 24usize;
    let input: Vec<f64> = (0..n).map(|i| (i as f64) * 0.25 - 1.5).collect();
    let a = 3.0f64;
    // Reference: the interpreter evaluates the same body.
    let mut interpreter = Interpreter::new(&program);
    let x = interpreter.add_tensor(TensorData::dense(DType::F32, vec![n], input.clone()));
    let y = interpreter.add_tensor(TensorData::dense(DType::F32, vec![n], vec![0.0; n]));
    let bindings = Bindings {
        shapes: [("N".to_string(), n as i64)].into_iter().collect(),
        ..Bindings::default()
    };
    interpreter
        .run(
            "scale",
            &[Arg::Tensor(x), Arg::Tensor(y), Arg::Scalar(a)],
            &bindings,
        )
        .expect("the reference executes");
    let reference = match &interpreter.tensors[y] {
        TensorData::Dense { data, .. } => data.clone(),
        _ => panic!("the reference output is dense"),
    };

    // CPU: construct → elaborate → plan → encode → assemble → execute.
    let backend = Cpu::host(4).expect("the CPU device opens");
    let compiled = pipeline::compile::<Cpu>(
        &program,
        "scale",
        &workload(&[("N", n as i64)]),
        &backend,
        &[],
        seismic_compiler::planning::Budget::default(),
    )
    .unwrap_or_else(|error| panic!("CPU pipeline failure: {error}"));
    let mut kernel = compiled.native.kernel;
    let mut workers = Workers::new(4).expect("the worker pool opens");
    let input_bytes: Vec<u8> = input
        .iter()
        .flat_map(|v| (*v as f32).to_le_bytes())
        .collect();
    let x_buffer = Buffer::from_bytes(&input_bytes).expect("the input buffer allocates");
    let y_buffer = Buffer::new(n * 4).expect("the output buffer allocates");
    let outputs = kernel
        .run(&mut workers, &[&x_buffer, &y_buffer], &[a])
        .map_err(|failure: InvocationFailure| failure.to_string())
        .expect("the CPU kernel executes");
    // The exclusive parameter `y` is written in place.
    let _ = outputs;
    let mut result_bytes = vec![0u8; n * 4];
    y_buffer.read(&mut result_bytes).expect("the output reads");
    let result: Vec<f64> = result_bytes
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()) as f64)
        .collect();
    assert_eq!(result.len(), reference.len());
    for (index, (left, right)) in result.iter().zip(&reference).enumerate() {
        assert_eq!(
            (*left as f32).to_bits() as u64,
            (*right as f32).to_bits() as u64,
            "element {index}: CPU {} vs reference {}",
            left,
            right
        );
    }
}

#[test]
fn tensor_copy_agrees_with_the_reference_interpreter() {
    // A whole-tensor copy into an exclusive place: both storages are launch
    // inputs, so the resolved artifact expresses the whole dataflow and the
    // CPU backend executes it — JIT emission, worker distribution, scalar
    // and buffer tables, and bit-exact element transfer.
    let source = "\
fn copyinto[N](x: &tensor[N] f32, y: &mut tensor[N] f32):
    y = x
    return
";
    let program = compile_program(&[source]);
    let n = 24usize;
    let input: Vec<f64> = (0..n).map(|i| (i as f64) * 0.5 - 3.0).collect();
    // Reference.
    let mut interpreter = Interpreter::new(&program);
    let x = interpreter.add_tensor(TensorData::dense(DType::F32, vec![n], input.clone()));
    let y = interpreter.add_tensor(TensorData::dense(DType::F32, vec![n], vec![0.0; n]));
    let bindings = Bindings {
        shapes: [("N".to_string(), n as i64)].into_iter().collect(),
        ..Bindings::default()
    };
    interpreter
        .run("copyinto", &[Arg::Tensor(x), Arg::Tensor(y)], &bindings)
        .expect("the reference executes");
    let reference = match &interpreter.tensors[y] {
        TensorData::Dense { data, .. } => data.clone(),
        _ => panic!("the reference output is dense"),
    };
    // CPU.
    let backend = Cpu::host(4).expect("the CPU device opens");
    let compiled = pipeline::compile::<Cpu>(
        &program,
        "copyinto",
        &workload(&[("N", n as i64)]),
        &backend,
        &[],
        seismic_compiler::planning::Budget::default(),
    )
    .expect("the CPU pipeline compiles and assembles");
    let mut kernel = compiled.native.kernel;
    assert!(kernel.launch_count() >= 1, "the plan retains its launches");
    let mut workers = Workers::new(4).expect("the worker pool opens");
    let input_bytes: Vec<u8> = input
        .iter()
        .flat_map(|v| (*v as f32).to_le_bytes())
        .collect();
    let x_buffer = Buffer::from_bytes(&input_bytes).expect("the input buffer allocates");
    let y_buffer = Buffer::new(n * 4).expect("the output buffer allocates");
    kernel
        .run(&mut workers, &[&x_buffer, &y_buffer], &[])
        .map_err(|failure: InvocationFailure| failure.to_string())
        .expect("the CPU kernel executes");
    let mut result_bytes = vec![0u8; n * 4];
    y_buffer.read(&mut result_bytes).expect("the output reads");
    let result: Vec<f64> = result_bytes
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()) as f64)
        .collect();
    for (index, (left, right)) in result.iter().zip(&reference).enumerate() {
        // Both sides are exact f32 values widened to f64: f64 equality is
        // bit equality here (the interpreter's `S` model).
        assert_eq!(
            left, right,
            "element {index}: CPU {} vs reference {}",
            left, right
        );
    }
}

#[test]
fn family_and_plan_resolve_for_a_reduction_kernel() {
    // A full reduction: the family must cover the alternative and the one
    // global model must resolve a plan with the structured schedule and the
    // root ABI, even where launch emission is blocked upstream.
    let source = "\
fn sum[N](x: &tensor[N] f32) -> f32:
    return reduce(f32(x), 0, sum)
";
    let program = compile_program(&[source]);
    let logical = seismic_lang::logical::construct(
        &program,
        "sum",
        &seismic_lang::logical::EffectiveTargetIdentity {
            backend: TARGET.to_string(),
            capability_fingerprint: "test".into(),
        },
        &|_| Ok(()),
        [("N".to_string(), 16)].into_iter().collect(),
        BTreeMap::new(),
    )
    .expect("the logical program constructs");
    let backend = Cpu::host(4).expect("the CPU device opens");
    let family = backend
        .elaborate(&logical)
        .expect("the CPU family elaborates");
    assert!(
        !family.choices.is_empty() && !family.storages.is_empty(),
        "the family carries choices and the storage table"
    );
    let plan = seismic_compiler::planning::plan(
        &logical,
        &family,
        &seismic_compiler::planning::Context {
            target: backend.target_profile(),
            precision: &PrecisionPolicy::default(),
            numerical_evidence: &[],
        },
        seismic_compiler::planning::Budget::default(),
    );
    match plan {
        Ok(resolved) => {
            assert!(
                matches!(
                    resolved.entry.schedule.steps.iter().next(),
                    Some(seismic_realization::executable::ResolvedStep::Launch(_))
                ),
                "the entry schedule is a structured step tree"
            );
        }
        Err(error) => panic!("the reduction kernel must plan: {error}"),
    }
}
