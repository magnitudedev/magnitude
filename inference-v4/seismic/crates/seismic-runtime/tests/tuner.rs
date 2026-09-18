//! Integration of selection, independent certificate replay and native execution.
//! The cost contract here is deliberately synthetic; it proves plumbing and
//! preservation, not a CPU hardware performance model.
use seismic_accounting::selection::{Context, Cost, Node, Realization, Space};
use seismic_lang::{
    lowered_ir::LoweredIr,
    program::{compile, SourceFile},
    Scope,
};
use seismic_realization::LoadStrategy;
use seismic_runtime::{
    execution::Execution,
    tuner::{self, BackendSpace},
    Candidate, Device, DeviceFacts,
};
use std::{cell::Cell, collections::HashMap};

struct Fixture {
    lowered: LoweredIr,
    hardware: DeviceFacts,
    context: Context,
    prepared: Cell<usize>,
}
impl Space for Fixture {
    type Execution = Execution;
    fn context(&self) -> &Context {
        &self.context
    }
    fn expand(&self, prefix: &[usize]) -> Result<Node<Execution>, String> {
        let loads = match prefix {
            [] => {
                return Ok(Node::Choice {
                    name: "load realization".into(),
                    alternatives: vec!["materialize".into(), "proven borrow".into()],
                })
            }
            [0] => LoadStrategy::Materialize,
            [1] => LoadStrategy::BorrowProvenReadOnly,
            _ => return Err("invalid fixture prefix".into()),
        };
        let execution =
            Execution::prepare(&self.lowered, Candidate::Cpu { loads }, &self.hardware)?;
        self.prepared.set(self.prepared.get() + 1);
        let Execution::Cpu(program) = &execution else {
            unreachable!()
        };
        // Synthetic contract: one model tick per byte of invocation scratch.
        // This is intentionally not a physical CPU latency prediction.
        let cost = Cost::exact(program.scratch_bytes as u64);
        Ok(Node::Realization(Realization { execution, cost }))
    }
}
impl BackendSpace for Fixture {
    fn hardware(&self) -> &DeviceFacts {
        &self.hardware
    }
}
#[test]
fn selected_program_survives_accounting_verification_and_native_compilation() {
    let program = compile(
        &[SourceFile {
            path: "tuner.seismic.portable".into(),
            scope: Scope::Portable,
            text:
                "fn copy(x: tensor[7] f32, out: tensor[7] f32):\n  a = load(x)\n  store(a, out)\n"
                    .into(),
        }],
        &[],
    )
    .unwrap();
    let device = Device::cpu();
    let space = Fixture {
        lowered: seismic_lang::lower::lower(&program, "copy", "cpu", &HashMap::new()).unwrap(),
        hardware: device.facts(),
        prepared: Cell::new(0),
        context: Context {
            program: "copy source above".into(),
            workload: "seven f32 elements".into(),
            target: "synthetic scratch-cost fixture".into(),
            contracts: "one tick per scratch byte".into(),
            execution_form: "sequential scalar, two load strategies".into(),
            objective: "synthetic duration".into(),
            seconds_numerator: 1,
            seconds_denominator: 1,
        },
    };
    let tuned = tuner::tune(&space, 3).unwrap();
    assert_eq!(space.prepared.get(), 2);
    tuner::verify(&space, &tuned, 3).unwrap();
    let prepared = space.prepared.get();
    tuned.execution().account().unwrap();
    let mut kernel = device.compile_tuned(tuned).unwrap();
    assert_eq!(
        space.prepared.get(),
        prepared,
        "native compilation must not re-enter selection"
    );
    let input: Vec<u8> = (0..7).flat_map(|i| (i as f32).to_le_bytes()).collect();
    let source = device.buffer_from(&input).unwrap();
    let target = device.buffer(input.len()).unwrap();
    kernel.execute(&[source, target.clone()], &[]).unwrap();
    let mut output = vec![0; input.len()];
    target.read(&mut output).unwrap();
    assert_eq!(input, output);
}
