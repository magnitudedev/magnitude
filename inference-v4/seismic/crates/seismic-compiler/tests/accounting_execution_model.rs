use seismic_accounting::workload::{
    Allocation, BufferBinding, DerivationError, DerivationLimit, DerivationLimits, ScalarWorkload,
};
use seismic_accounting::{
    execution_model::*,
    schedule::{CapacityUnit, Reservation, Resource, Timebase},
};
use seismic_lang::{
    Scope as LanguageScope,
    program::{SourceFile, compile},
};
use seismic_realization::{CallConv, Dispatch, ScalarProgram};
use std::collections::{BTreeMap, HashMap};

fn program(source: &str, name: &str, n: i64, dispatch: Dispatch) -> ScalarProgram {
    let p = compile(
        &[SourceFile {
            path: "model.seismic.portable".into(),
            scope: LanguageScope::Portable,
            text: source.into(),
        }],
        &[],
    )
    .unwrap();
    let lowered =
        seismic_lang::lower::lower(&p, name, "cpu", &HashMap::from([("N".into(), n)])).unwrap();
    seismic_compiler::scalar_with(&lowered, CallConv::SystemV, dispatch).unwrap()
}

// Explicit hypothetical machine. Every primitive, including checks, addressing,
// and branches consumes this synthetic service. Block transfers are structural;
// imported math bodies remain unresolved. The pool
// pool. These numbers are an oracle fixture, not a native CPU or GPU profile.
fn fixture(program: &ScalarProgram) -> (ScalarHardware, ScalarWorkload) {
    let mut patterns = Vec::new();
    for required in requirements(program).unwrap() {
        if !matches!(required.kind, PrimitiveKind::Instruction(_)) {
            continue;
        }
        let pattern = required.signature();
        if !patterns.contains(&pattern) {
            patterns.push(pattern);
        }
    }
    let contract = ScalarHardware {
        identity: "synthetic one-service-per-tick direct scalar".into(),
        scope: Scope::HypotheticalDirectScalarV1,
        timebase: Timebase {
            seconds_numerator: 1,
            seconds_denominator: 1,
        },
        resources: vec![Resource {
            name: "synthetic scalar service".into(),
            capacity: 1,
            unit: CapacityUnit::Slots,
        }],
        timings: patterns
            .into_iter()
            .map(|primitive| PrimitiveTiming {
                primitive,
                latency: 1,
                services: vec![Reservation {
                    resource: 0,
                    offset: 0,
                    duration: 1,
                    units: 1,
                }],
            })
            .collect(),
    };
    let workload = ScalarWorkload { integer_domains: Vec::new(),
        identity: "exact allocation topology; unconstrained external contents".into(),
        allocations: program
            .buffers
            .iter()
            .enumerate()
            .map(|(id, b)| Allocation {
                id: id as u64,
                bytes: b.bytes as u64,
                alignment: b.alignment as u64,
                known_bytes: BTreeMap::new(),
            })
            .collect(),
        buffers: program
            .buffers
            .iter()
            .enumerate()
            .map(|(id, b)| BufferBinding {
                allocation: id as u64,
                offset: 0,
                bytes: b.bytes as u64,
            })
            .collect(),
        scalars: vec![0; program.scalars.len() * 8],
    };
    (contract, workload)
}
fn derive(p: &ScalarProgram, c: &ScalarHardware, w: &ScalarWorkload) -> DerivedModel {
    derive_scalar(
        p,
        c,
        w,
        DerivationLimits {
            instructions: 20_000,
            operations: 60_000,
        },
    )
    .unwrap()
}
const COPY: &str = "fn copy[N](x: tensor[N] f32, out: tensor[N] f32):\n  for row in parallel:\n    t = load(x[row : row + 1])\n    store(t, out[row : row + 1])\n";

#[test]
fn actual_loop_instances_and_alias_versions_generate_checked_schedules() {
    let p = program(COPY, "copy", 3, Dispatch::Sequential);
    let (c, w) = fixture(&p);
    let d = derive(&p, &c, &w);
    assert_eq!(d.model.operations.len(), d.origins.len());
    assert_eq!(
        d.origins
            .iter()
            .filter(|o| matches!(o, Origin::Instruction { .. }))
            .count() as u64,
        d.instructions
    );
    for (origin, operation) in d.origins.iter().zip(&d.model.operations) {
        if matches!(origin, Origin::EdgeTransfer { .. }) {
            assert_eq!(operation.latency, 0);
            assert!(operation.reservations.is_empty());
        }
    }
    let reads: Vec<_> = d
        .accesses
        .iter()
        .filter(|a| a.allocation == AllocationIdentity::External(0))
        .collect();
    assert_eq!(
        reads.iter().map(|a| a.offset).collect::<Vec<_>>(),
        [0, 4, 8]
    );
    assert!(
        d.origins
            .iter()
            .any(|o| matches!(o, Origin::Instruction { occurrence: 2, .. }))
    );
    assert!(
        d.origins
            .iter()
            .any(|o| matches!(o, Origin::EdgeTransfer { .. }))
    );
    let solution = d.model.solve(0).unwrap();
    assert!(solution.is_optimal());
    // Resource total and serialized feasible witness meet for the declared model.
    assert_eq!(
        solution.schedule().completion,
        d.model.operations.iter().map(|op| op.latency).sum()
    );
    let mut alias = w;
    alias.buffers[1].allocation = 0;
    let aliased = derive(&p, &c, &alias);
    assert!(
        aliased
            .accesses
            .iter()
            .filter(|a| a.write && a.allocation == AllocationIdentity::External(0))
            .count()
            == 3
    );
    for write in aliased
        .accesses
        .iter()
        .filter(|a| a.write && a.allocation == AllocationIdentity::External(0))
    {
        let read = aliased
            .accesses
            .iter()
            .find(|a| !a.write && a.allocation == write.allocation && a.offset == write.offset)
            .unwrap();
        assert!(
            aliased.model.operations[write.completion]
                .predecessors
                .contains(&read.completion)
        );
    }
}

#[test]
fn parallel_invocations_share_resources_but_own_scratch_and_must_not_race() {
    let p = program(COPY, "copy", 3, Dispatch::ParallelRoot);
    let (c, w) = fixture(&p);
    let d = derive(&p, &c, &w);
    assert!(
        d.accesses
            .iter()
            .any(|a| a.allocation == AllocationIdentity::Scratch(2))
    );
    assert_eq!(
        d.accesses
            .iter()
            .filter(|a| a.write && a.allocation == AllocationIdentity::External(1))
            .count(),
        3
    );
    // A one-element shifted overlapping output turns per-invocation independence
    // into a real write/read conflict even though parameter names remain distinct.
    let mut overlap = w;
    overlap.allocations[0].bytes += 4;
    overlap.buffers[1].allocation = 0;
    overlap.buffers[1].offset = 4;
    assert!(
        derive_scalar(
            &p,
            &c,
            &overlap,
            DerivationLimits {
                instructions: 20_000,
                operations: 60_000
            }
        )
        .unwrap_err()
        .to_string()
        .contains("conflicting aliased")
    );
}

#[test]
fn branches_follow_typed_scalar_conditions_and_do_not_charge_both_paths() {
    let p = program(
        "fn branch[N](out: tensor[1] f32, choose: bool):\n  y = tile[1] f32\n  for i in owned(y): y[i] = 7.0\n  if choose:\n    store(y, out)\n",
        "branch",
        1,
        Dispatch::Sequential,
    );
    let (c, mut w) = fixture(&p);
    let off = derive(&p, &c, &w);
    assert!(
        !off.accesses
            .iter()
            .any(|a| a.write && a.allocation == AllocationIdentity::External(0))
    );
    w.scalars[0] = 1;
    let on = derive(&p, &c, &w);
    assert_eq!(
        on.accesses
            .iter()
            .filter(|a| a.write && a.allocation == AllocationIdentity::External(0))
            .count(),
        1
    );
    assert!(on.instructions > off.instructions);
    w.scalars[0] = 2;
    assert!(
        derive_scalar(
            &p,
            &c,
            &w,
            DerivationLimits {
                instructions: 20_000,
                operations: 60_000
            }
        )
        .is_err()
    );
}

#[test]
fn data_dependent_extents_require_real_workload_conditions() {
    let source = "fn stream[N](x: tensor[N] f32, visible: tensor[2] i32, out: tensor[1] f32):\n  acc = tile[1] f32\n  for i in owned(acc): acc[i] = 0.0\n  t = load(x[visible[0]:visible[1]])\n  acc[0] = reduce(t, 0, sum, ordered=true)\n  store(acc, out)\n";
    let p = program(source, "stream", 7, Dispatch::Sequential);
    let (c, mut w) = fixture(&p);
    let limits = DerivationLimits {
        instructions: 20_000,
        operations: 60_000,
    };
    let Err(error) = derive_scalar(&p, &c, &w, limits) else {
        panic!("unknown slice endpoints must leave the executed path undetermined");
    };
    assert!(
        error.to_string().contains("data-dependent branch"),
        "{error}"
    );

    // Source data-dependent slices clamp into the parent extent. Accounting
    // must follow the resulting executed window, including empty windows,
    // rather than rejecting an out-of-range raw endpoint or charging capacity.
    for (bounds, expected) in [
        ([1i32, 4], vec![4, 8, 12]),
        ([1, 9], vec![4, 8, 12, 16, 20, 24]),
        ([-3, 3], vec![0, 4, 8]),
        ([5, 2], vec![]),
        ([-4, -1], vec![]),
        ([9, 12], vec![]),
    ] {
        w.allocations[1].known_bytes = bounds
            .into_iter()
            .flat_map(i32::to_le_bytes)
            .enumerate()
            .map(|(i, b)| (i as u64, b))
            .collect();
        let d = derive(&p, &c, &w);
        assert_eq!(
            d.accesses
                .iter()
                .filter(|a| !a.write && a.allocation == AllocationIdentity::External(0))
                .map(|a| a.offset)
                .collect::<Vec<_>>(),
            expected,
            "bounds={bounds:?}"
        );
        assert_eq!(
            d.accesses
                .iter()
                .filter(|a| a.write && a.allocation == AllocationIdentity::External(2))
                .count(),
            1,
            "even an empty reduction publishes its result: bounds={bounds:?}"
        );
    }
}

#[test]
fn hardware_timings_are_unambiguous_and_missing_information_remains_incomplete() {
    let p = program(COPY, "copy", 1, Dispatch::Sequential);
    let (mut c, w) = fixture(&p);
    let required = requirements(&p).unwrap();
    assert!(
        required.len() > c.timings.len(),
        "one reusable implementation should cover multiple immediate values"
    );
    let last = c.timings.pop().unwrap();
    assert!(
        derive_scalar(
            &p,
            &c,
            &w,
            DerivationLimits {
                instructions: 20_000,
                operations: 60_000
            }
        )
        .unwrap()
        .model
        .unmapped
        .len()
            > 0
    );
    c.timings.push(last.clone());
    c.timings.push(last);
    assert!(c.validate().unwrap_err().contains("overlapping"));
    c.timings.pop();
    for (limits, expected) in [
        (
            DerivationLimits {
                instructions: 2,
                operations: 60_000,
            },
            DerivationLimit::Instructions(2),
        ),
        (
            DerivationLimits {
                instructions: 20_000,
                operations: 2,
            },
            DerivationLimit::Operations(2),
        ),
    ] {
        assert_eq!(
            derive_scalar(&p, &c, &w, limits).unwrap_err(),
            DerivationError::Exhausted(expected)
        );
    }
    let mut bad = c;
    bad.timings[0].services.clear();
    assert!(bad.validate().unwrap_err().contains("service fact"));
}

#[test]
fn branch_and_address_specialization_matches_the_independent_language_interpreter() {
    use seismic_lang::{
        interp::{Arg, Interpreter, TensorData},
        types::DType,
    };
    let cases = [
        (
            "i32",
            "a / b < 0",
            vec![
                (-17.0, 3.0),
                (-17.0, -3.0),
                (17.0, -3.0),
                (1.0, 0.0),
                (i32::MIN as f64, -1.0),
            ],
        ),
        (
            "i32",
            "a % b == 0",
            vec![(-18.0, 3.0), (-17.0, 3.0), (-17.0, -3.0), (17.0, -3.0)],
        ),
        (
            "i32",
            "(a >> b) < 0",
            vec![
                (-2147483648.0, 31.0),
                (-1.0, 0.0),
                (2147483647.0, 31.0),
                (1.0, 32.0),
            ],
        ),
        (
            "u32",
            "(a >> b) == 0",
            vec![(4294967295.0, 31.0), (2147483647.0, 31.0), (1.0, 1.0)],
        ),
        (
            "f32",
            "a < b",
            vec![
                (f64::NAN, 0.0),
                (f64::NEG_INFINITY, 1.0),
                (-0.0, 0.0),
                (1.0, f64::INFINITY),
            ],
        ),
    ];
    for (dtype, condition, inputs) in cases {
        let text = format!(
            "fn choose(out: tensor[2] f32, a: {dtype}, b: {dtype}):\n  y = tile[1] f32\n  for i in owned(y): y[i] = 1.0\n  if {condition}:\n    store(y,out[0:1])\n  else:\n    store(y,out[1:2])\n"
        );
        let source = compile(
            &[SourceFile {
                path: "independent.seismic.portable".into(),
                scope: LanguageScope::Portable,
                text,
            }],
            &[],
        )
        .unwrap();
        let lowered =
            seismic_lang::lower::lower(&source, "choose", "cpu", &HashMap::new()).unwrap();
        let scalar =
            seismic_compiler::scalar_with(&lowered, CallConv::SystemV, Dispatch::Sequential)
                .unwrap();
        let (contract, mut workload) = fixture(&scalar);
        for (a, b) in inputs {
            let mut interpreter = Interpreter::new(&source);
            let out = interpreter.add_tensor(TensorData::dense(DType::F32, vec![2], vec![0.0; 2]));
            let expected = interpreter.run(
                "choose",
                &[Arg::Tensor(out), Arg::Scalar(a), Arg::Scalar(b)],
                &HashMap::new(),
            );
            workload.scalars = seismic_lang::abi::ScalarLayout::words(&scalar.scalars)
                .unwrap()
                .encode(&[a, b])
                .unwrap();
            let actual = derive_scalar(
                &scalar,
                &contract,
                &workload,
                DerivationLimits {
                    instructions: 20_000,
                    operations: 60_000,
                },
            );
            assert_eq!(
                expected.is_ok(),
                actual.is_ok(),
                "{condition}: {a}, {b}; model={actual:?}; interpreter={expected:?}"
            );
            if let Ok(model) = actual {
                let writes: Vec<_> = model
                    .accesses
                    .iter()
                    .filter(|a| a.write && a.allocation == AllocationIdentity::External(0))
                    .collect();
                assert_eq!(writes.len(), 1);
                let expected_index = (0..2)
                    .find(|i| interpreter.tensors[out].get(*i) == 1.0)
                    .unwrap();
                assert_eq!(
                    writes[0].offset,
                    expected_index as u64 * 4,
                    "{condition}: {a}, {b}"
                );
            }
        }
    }
}

#[test]
fn imported_helpers_cannot_be_priced_as_hardware_instructions() {
    let p = program(
        "fn math(out: tensor[1] f32, x: f32):\n  y = tile[1] f32\n  for i in owned(y): y[i] = exp(x)\n  store(y,out)\n",
        "math",
        1,
        Dispatch::Sequential,
    );
    let (mut hardware, workload) = fixture(&p);
    let derived = derive(&p, &hardware, &workload);
    assert!(!derived.model.unmapped.is_empty());
    assert!(derived.model.solve(100).is_err());
    assert!(
        derived
            .accesses
            .iter()
            .any(|a| a.write && matches!(a.allocation, AllocationIdentity::External(_)))
    );
    let helper = requirements(&p)
        .unwrap()
        .into_iter()
        .find(|p| matches!(p.kind, PrimitiveKind::Math(_)))
        .unwrap();
    hardware.timings.push(PrimitiveTiming {
        primitive: helper.signature(),
        latency: 1,
        services: vec![Reservation {
            resource: 0,
            offset: 0,
            duration: 1,
            units: 1,
        }],
    });
    assert!(hardware.validate().unwrap_err().contains("helper call"));
}

#[test]
fn original_parallel_alias_admission_precedes_scalar_model_construction() {
    let source = "construct point[K](x: tile[K] f32, y: tile[K] f32):\n  for j in owned(y): y[j] = x[j]\nfn copy[N](x: tensor[N+1] f32, out: tensor[N] f32):\n  for i in parallel:\n    a = load(x[i+1:i+2])\n    b = tile[1] f32\n    point(a,b)\n    store(b,out[i:i+1])\n";
    let checked = compile(
        &[
            SourceFile {
                path: "alias.seismic.portable".into(),
                scope: LanguageScope::Portable,
                text: source.into(),
            },
            SourceFile {
                path: "alias.seismic.cpu".into(),
                scope: LanguageScope::Backend("cpu".into()),
                text: "lower point: portable\n".into(),
            },
        ],
        &[],
    )
    .unwrap();
    let lowered =
        seismic_lang::lower::lower(&checked, "copy", "cpu", &HashMap::from([("N".into(), 4)]))
            .unwrap();
    let p =
        seismic_compiler::scalar_with(&lowered, CallConv::SystemV, Dispatch::Sequential).unwrap();
    assert!(!p.conditions.alias_pairs().is_empty());
    let (hardware, mut workload) = fixture(&p);
    derive(&p, &hardware, &workload);
    workload.buffers[1].allocation = workload.buffers[0].allocation;
    let Err(error) = derive_scalar(
        &p,
        &hardware,
        &workload,
        DerivationLimits {
            instructions: 20_000,
            operations: 60_000,
        },
    ) else {
        panic!("source alias violation acquired a modeled execution");
    };
    assert!(
        matches!(error, DerivationError::Analysis(message) if message.contains("source parallel binding"))
    );
}
