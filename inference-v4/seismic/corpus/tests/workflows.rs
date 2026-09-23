//! Workflow composition through the public dynamic workflow API (design A10 §2.9).
use crate::common::{device, prepare, Selection};
use seismic::dynamic::{Kernel, Module, Scalar, Tensor, Value, Workflow, WorkflowValue};
use seismic::{BackendName, Device, Element, PrecisionPolicy};
use std::collections::BTreeMap;
use std::sync::Arc;

const MODULE: &str = "\
fn twice[N](x: &tensor[N] f32) -> tensor[N] f32:
    return x + x
fn total[N](x: &tensor[N] f32) -> f32:
    return reduce(x, 0, sum)
fn first(x: &tensor[4] f32) -> f32:
    return x[0]
fn scale(x: &tensor[4] f32, s: f32) -> tensor[4] f32:
    return x * s
fn composed(x: &tensor[4] f32) -> f32:
    return total(twice(x))
fn write_then_divide(out: &mut tensor[2] i32, d: i32):
    out[0] = 7
    out[1] = 10 / d
";

/// The index-result programs of W7/W8, kept apart so that the checker's
/// verdict on `index(...)` does not decide the other workflows.
const INDEX_MODULE: &str = "\
fn pick8(x: &tensor[8] i32) -> index[8]:
    return index(x[0])
fn at4(t: &tensor[4] f32, i: index[4]) -> f32:
    return t[i]
";

struct Fixture {
    device: Device,
    module: Module,
}

impl Fixture {
    fn new(backend: BackendName, source: &str) -> Self {
        Self {
            device: device(backend),
            module: Module::source(source, "workflows.seismic", false, None)
                .unwrap_or_else(|e| panic!("the workflow module must load: {e}")),
        }
    }

    fn kernel(&self, entry: &str) -> Arc<Kernel> {
        let function = self.module.function(entry).expect("declared entry");
        let (kernel, _) = prepare(
            &function,
            &self.device,
            BTreeMap::new(),
            &PrecisionPolicy::Exact,
            &Selection::Baseline,
        )
        .unwrap_or_else(|e| panic!("{entry}: preparation: {e}"));
        Arc::new(kernel)
    }

    fn f32s(&self, values: &[f32]) -> Tensor {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        Tensor::from_host(&self.device, Element::f32(), &[values.len() as u64], &bytes)
            .expect("host tensor")
    }

    fn i32s(&self, values: &[i32]) -> Tensor {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        Tensor::from_host(&self.device, Element::i32(), &[values.len() as u64], &bytes)
            .expect("host tensor")
    }
}

fn external(tensor: &Tensor) -> WorkflowValue {
    WorkflowValue::External(Value::Tensor(tensor.clone()))
}

/// The bits of a resolved or returned f32 scalar.
fn f32_bits(value: &Value) -> u32 {
    let Value::Scalar(Scalar::F32(bits)) = value else {
        panic!("expected an f32 scalar");
    };
    *bits
}

fn resolved(value: WorkflowValue) -> Value {
    let WorkflowValue::External(value) = value else {
        panic!("a resolved workflow output is external");
    };
    value
}

fn tensor_bytes(value: &Value) -> Vec<u8> {
    let Value::Tensor(tensor) = value else {
        panic!("expected a tensor");
    };
    tensor.read().expect("readable result")
}

/// W1: `twice -> total` resolves to the bits of `composed`.
fn chain_matches_composed_entry(backend: BackendName) {
    let f = Fixture::new(backend, MODULE);
    let x = f.f32s(&[1.5, -2.0, 3.25, 0.5]);
    let mut workflow = Workflow::new(&f.device);
    let doubled = workflow.enqueue(f.kernel("twice"), vec![external(&x)]).unwrap();
    let sum = workflow.enqueue(f.kernel("total"), vec![doubled]).unwrap();
    let chained = resolved(workflow.run(sum).unwrap());
    let direct = f.kernel("composed").call(&[Value::Tensor(x)]).unwrap();
    assert_eq!(f32_bits(&chained), f32_bits(&direct));
}

/// W2: a device scalar result feeds the next node (C5-2, X1 P39).
fn device_scalar_result_feeds_next_node(backend: BackendName) {
    let f = Fixture::new(backend, MODULE);
    let x = f.f32s(&[1.5, -2.0, 3.25, 0.5]);
    let mut workflow = Workflow::new(&f.device);
    let head = workflow.enqueue(f.kernel("first"), vec![external(&x)]).unwrap();
    let scaled = workflow.enqueue(f.kernel("scale"), vec![external(&x), head]).unwrap();
    let chained = resolved(workflow.run(scaled).unwrap());
    let direct = f
        .kernel("scale")
        .call(&[Value::Tensor(x), Value::Scalar(Scalar::F32(1.5f32.to_bits()))])
        .unwrap();
    assert_eq!(tensor_bytes(&chained), tensor_bytes(&direct));
}

/// W3: `twice(x) -> {total, first}` plus `scale(x, first)`; every result matches a sequential call.
fn diamond_resolves(backend: BackendName) {
    let f = Fixture::new(backend, MODULE);
    let x = f.f32s(&[1.5, -2.0, 3.25, 0.5]);
    let mut workflow = Workflow::new(&f.device);
    let doubled = workflow.enqueue(f.kernel("twice"), vec![external(&x)]).unwrap();
    let sum = workflow.enqueue(f.kernel("total"), vec![doubled.clone()]).unwrap();
    let head = workflow.enqueue(f.kernel("first"), vec![doubled]).unwrap();
    let scaled = workflow.enqueue(f.kernel("scale"), vec![external(&x), head.clone()]).unwrap();
    let WorkflowValue::Tuple(outputs) =
        workflow.run(WorkflowValue::Tuple(vec![sum, head, scaled])).unwrap()
    else {
        panic!("a tuple of outputs resolves to a tuple");
    };
    let outputs: Vec<Value> = outputs.into_iter().map(resolved).collect();

    let doubled = f.kernel("twice").call(&[Value::Tensor(x.clone())]).unwrap();
    let sum = f.kernel("total").call(&[doubled.clone()]).unwrap();
    let head = f.kernel("first").call(&[doubled]).unwrap();
    let scaled = f.kernel("scale").call(&[Value::Tensor(x), head.clone()]).unwrap();
    assert_eq!(f32_bits(&outputs[0]), f32_bits(&sum));
    assert_eq!(f32_bits(&outputs[1]), f32_bits(&head));
    assert_eq!(tensor_bytes(&outputs[2]), tensor_bytes(&scaled));
}

/// W5: a failed node reports the source failure, keeps its effect prefix and stops the workflow.
/// `twice` has no effect and a failed workflow returns no result, so "did not run" is not
/// observable through this node; W9 observes it through a later node's writable input.
fn failed_node_keeps_prefix_and_stops(backend: BackendName) {
    let f = Fixture::new(backend, MODULE);
    let out = f.i32s(&[0, 0]);
    let x = f.f32s(&[1.0, 2.0, 3.0, 4.0]);
    let mut workflow = Workflow::new(&f.device);
    workflow
        .enqueue(
            f.kernel("write_then_divide"),
            vec![external(&out), WorkflowValue::External(Value::Scalar(Scalar::I32(0)))],
        )
        .unwrap();
    let doubled = workflow.enqueue(f.kernel("twice"), vec![external(&x)]).unwrap();
    let error = match workflow.run(doubled) {
        Err(error) => error,
        Ok(_) => panic!("the workflow must report node 0's source failure"),
    };
    assert_eq!(error.kind, "ExecutionError", "{}", error.message);
    let expected: Vec<u8> = [7i32, 0].iter().flat_map(|v| v.to_le_bytes()).collect();
    assert_eq!(out.read().unwrap(), expected);
}

fn pick_then_read(backend: BackendName, picked: i32) -> Result<Value, seismic::dynamic::Error> {
    let f = Fixture::new(backend, INDEX_MODULE);
    let x = f.i32s(&[picked, 0, 0, 0, 0, 0, 0, 0]);
    let t = f.f32s(&[10.0, 20.0, 30.0, 40.0]);
    let mut workflow = Workflow::new(&f.device);
    let index = workflow.enqueue(f.kernel("pick8"), vec![external(&x)]).unwrap();
    let read = workflow.enqueue(f.kernel("at4"), vec![external(&t), index]).unwrap();
    workflow.run(read).map(resolved)
}

/// W7: a device index result feeds an index parameter (X1 P40, A7 C5-2(b)).
fn device_index_result_feeds_index_parameter(backend: BackendName) {
    let value = pick_then_read(backend, 2).unwrap();
    assert_eq!(f32_bits(&value), 30.0f32.to_bits());
}

/// W8: an index outside the consumer's domain is refused when node 1 is issued.
fn device_scalar_decided_validation_fails_at_issue(backend: BackendName) {
    match pick_then_read(backend, 6) {
        Err(error) => assert!(
            error.kind == "InvocationError" && error.message.contains("outside"),
            "expected node 1's outside-domain invocation error, got {}: {}",
            error.kind,
            error.message
        ),
        Ok(_) => panic!("index 6 is outside `at4`'s index[4] domain"),
    }
}

/// W9: node 0 fails on the device; node 1, whose arguments are all host values and so is
/// selected at issue, never executes, and its writable input is unchanged (X1 P41).
fn failed_node_skips_at_issue_nodes(backend: BackendName) {
    let f = Fixture::new(backend, MODULE);
    let out = f.i32s(&[0, 0]);
    let out2 = f.i32s(&[0, 0]);
    let mut workflow = Workflow::new(&f.device);
    let divide = f.kernel("write_then_divide");
    let host_i32 = |value| WorkflowValue::External(Value::Scalar(Scalar::I32(value)));
    workflow
        .enqueue(divide.clone(), vec![external(&out), host_i32(0)])
        .unwrap();
    workflow
        .enqueue(divide, vec![external(&out2), host_i32(2)])
        .unwrap();
    let error = match workflow.run(WorkflowValue::Unit) {
        Err(error) => error,
        Ok(_) => panic!("the workflow must report node 0's source failure"),
    };
    assert_eq!(error.kind, "ExecutionError", "{}", error.message);
    let bytes = |values: [i32; 2]| -> Vec<u8> { values.iter().flat_map(|v| v.to_le_bytes()).collect() };
    assert_eq!(out.read().unwrap(), bytes([7, 0]));
    assert_eq!(out2.read().unwrap(), bytes([0, 0]), "node 1 executed after node 0 failed");
}

macro_rules! workflow_tests {
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

workflow_tests!(
    chain_matches_composed_entry,
    device_scalar_result_feeds_next_node,
    diamond_resolves,
    failed_node_keeps_prefix_and_stops,
    device_index_result_feeds_index_parameter,
    device_scalar_decided_validation_fails_at_issue,
    failed_node_skips_at_issue_nodes,
);
