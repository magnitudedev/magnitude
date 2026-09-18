use cranelift_codegen::ir::{InstructionData, Opcode};
use seismic_accounting::workload::{Allocation, BufferBinding, DerivationLimits, ScalarWorkload};
use seismic_accounting::{execution_model::*, schedule::*};
use seismic_lang::{
    program::{compile, SourceFile},
    Scope as LanguageScope,
};
use seismic_realization::{
    scheduling::{self, Expansion, Order, Space},
    CallConv, Dispatch, ScalarProgram,
};
use std::collections::{BTreeMap, HashMap};

const SOURCE: &str = "fn scalar(out: tensor[2] i32, a: i32, b: i32):\n  y = tile[2] i32\n  for i in owned(y):\n    if i == 0: y[i] = a + b\n    else: y[i] = a - b\n  store(y,out)\n";
fn program() -> ScalarProgram {
    let source = compile(
        &[SourceFile {
            path: "schedule.seismic.portable".into(),
            scope: LanguageScope::Portable,
            text: SOURCE.into(),
        }],
        &[],
    )
    .unwrap();
    let lowered = seismic_lang::lower::lower(&source, "scalar", "cpu", &HashMap::new()).unwrap();
    seismic_compiler::scalar_with(&lowered, CallConv::SystemV, Dispatch::Sequential).unwrap()
}
fn choose_last(space: &Space) -> Order {
    let mut prefix = Vec::new();
    loop {
        match space.expand(&prefix).unwrap() {
            Expansion::Choice(domain) => prefix.push(domain.alternatives.len() - 1),
            Expansion::Order { order, consumed } => {
                assert_eq!(consumed, prefix.len());
                return order;
            }
        }
    }
}
fn fixture(program: &ScalarProgram) -> (ScalarHardware, ScalarWorkload) {
    let mut signatures = Vec::new();
    for primitive in requirements(program).unwrap() {
        let signature = primitive.signature();
        if !signatures.contains(&signature) {
            signatures.push(signature);
        }
    }
    (
        ScalarHardware {
            identity: "hypothetical issued scalar instructions".into(),
            scope: Scope::HypotheticalDirectScalarV1,
            timebase: Timebase {
                seconds_numerator: 1,
                seconds_denominator: 1,
            },
            resources: vec![Resource {
                name: "issue".into(),
                capacity: 1,
                unit: CapacityUnit::Slots,
            }],
            timings: signatures
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
        },
        ScalarWorkload { integer_domains: Vec::new(),
            identity: "two integer scalar operands".into(),
            allocations: vec![Allocation {
                id: 0,
                bytes: 8,
                alignment: 8,
                known_bytes: BTreeMap::new(),
            }],
            buffers: vec![BufferBinding {
                allocation: 0,
                offset: 0,
                bytes: 8,
            }],
            scalars: seismic_lang::abi::ScalarLayout::words(&program.scalars)
                .unwrap()
                .encode(&[11.0, 4.0])
                .unwrap(),
        },
    )
}

#[test]
fn selected_static_order_changes_emitted_ssa_without_unrolling_or_changing_computation() {
    let before = program();
    let space = Space::new(&before).unwrap();
    let order = choose_last(&space);
    assert_ne!(order, Order::current(&before));
    let mut after = program();
    scheduling::apply(&mut after, &order).unwrap();
    scheduling::check_materialization(&before, &after, &order).unwrap();
    assert!(before.function.dfg == after.function.dfg);
    assert_eq!(
        before.function.layout.blocks().collect::<Vec<_>>(),
        after.function.layout.blocks().collect::<Vec<_>>()
    );
    let (contract, workload) = fixture(&after);
    let model = derive_scalar(
        &after,
        &contract,
        &workload,
        DerivationLimits {
            instructions: 20_000,
            operations: 60_000,
        },
    )
    .unwrap();
    assert_eq!(model.order, order);
    assert!(model
        .model
        .static_orders
        .iter()
        .any(|block| block.visits.len() > 1));
    let optimum = model.model.solve(0).unwrap();
    assert!(optimum.is_optimal());
    // Multiple occurrences do not acquire independently selected issue orders.
    assert!(model
        .origins
        .iter()
        .any(|origin| matches!(origin, Origin::Instruction { occurrence: 2, .. })));
    let selected = Order {
        blocks: static_order::orders(&model.model, optimum.schedule()).unwrap(),
    };
    selected.check(&after).unwrap();
    scheduling::apply(&mut after, &selected).unwrap();
}

#[test]
fn correspondence_rejects_changed_arithmetic_missing_instructions_and_illegal_orders() {
    let before = program();
    let mut order = choose_last(&Space::new(&before).unwrap());
    let mut after = program();
    scheduling::apply(&mut after, &order).unwrap();
    let arithmetic = after
        .function
        .layout
        .blocks()
        .flat_map(|block| after.function.layout.block_insts(block))
        .find(|inst| {
            matches!(
                after.function.dfg.insts[*inst],
                InstructionData::Binary {
                    opcode: Opcode::Iadd,
                    ..
                }
            )
        })
        .unwrap();
    let InstructionData::Binary { args, .. } = after.function.dfg.insts[arithmetic] else {
        panic!()
    };
    after.function.dfg.insts[arithmetic] = InstructionData::Binary {
        opcode: Opcode::Isub,
        args,
    };
    assert!(scheduling::check_materialization(&before, &after, &order)
        .unwrap_err()
        .contains("changed the computation"));
    order.blocks[0].instructions.pop();
    assert!(order.check(&before).is_err());
    let mut illegal = Order::current(&before);
    illegal.blocks[0].instructions.reverse();
    let mut untouched = program();
    assert!(scheduling::apply(&mut untouched, &illegal).is_err());
    assert_eq!(
        untouched.function, before.function,
        "invalid proposal must not partially mutate executable SSA"
    );
}

#[test]
fn every_exposed_alternative_remains_in_the_same_complete_static_domain() {
    let program = program();
    let space = Space::new(&program).unwrap();
    let Expansion::Choice(first) = space.expand(&[]).unwrap() else {
        panic!("expected independent entry definitions")
    };
    assert!(first.alternatives.len() > 1);
    for branch in 0..first.alternatives.len() {
        let mut prefix = vec![branch];
        loop {
            match space.expand(&prefix).unwrap() {
                Expansion::Choice(domain) => {
                    assert!(!domain.alternatives.is_empty());
                    prefix.push(0);
                }
                Expansion::Order { order, consumed } => {
                    assert_eq!(consumed, prefix.len());
                    order.check(&program).unwrap();
                    assert_eq!(
                        order.blocks[0].instructions[first.position],
                        first.alternatives[branch]
                    );
                    break;
                }
            }
        }
    }
    assert!(space.expand(&[first.alternatives.len()]).is_err());
}

fn visit_model() -> Model {
    use cranelift_codegen::ir::{Block, Inst};
    Model {
        relationship: seismic_accounting::authority::ModelRelationship::hypothetical_execution(),
        identity: "two loop visits".into(),
        timebase: Timebase {
            seconds_numerator: 1,
            seconds_denominator: 1,
        },
        resources: vec![],
        operations: (0..4)
            .map(|i| Operation {
                name: format!("root{i}"),
                predecessors: vec![],
                start_predecessors: vec![],
                latency: 1,
                reservations: vec![],
            })
            .collect(),
        lifetimes: vec![],
        unmapped: vec![],
        static_orders: vec![static_order::Constraint {
            block: Block::from_u32(0),
            instructions: vec![Inst::from_u32(0), Inst::from_u32(1)],
            predecessors: vec![],
            visits: vec![
                static_order::Visit {
                    invocation: 0,
                    occurrence: 0,
                    roots: vec![vec![0], vec![1]],
                },
                static_order::Visit {
                    invocation: 0,
                    occurrence: 1,
                    roots: vec![vec![2], vec![3]],
                },
            ],
        }],
    }
}

#[test]
fn joint_timing_acceptance_equals_explicit_static_order_union() {
    let model = visit_model();
    // Independently enumerate the tiny entire timing domain and both source
    // permutations. Cross-visit reversal must never be certified as emittable.
    for a in 0..3 {
        for b in 0..3 {
            for c in 0..3 {
                for d in 0..3 {
                    let starts = vec![a, b, c, d];
                    let expected = (a <= b && c <= d) || (b <= a && d <= c);
                    let schedule = Schedule {
                        completion: *starts.iter().max().unwrap() + 1,
                        starts,
                    };
                    assert_eq!(model.check_schedule(&schedule).is_ok(), expected);
                    if expected {
                        let orders = static_order::orders(&model, &schedule).unwrap();
                        let forward = orders[0].instructions == model.static_orders[0].instructions;
                        assert!(if forward {
                            a <= b && c <= d
                        } else {
                            b <= a && d <= c
                        });
                    }
                }
            }
        }
    }
    let mut ordered = model.clone();
    ordered.static_orders[0].predecessors.push((0, 1));
    assert!(ordered
        .check_schedule(&Schedule {
            starts: vec![1, 0, 1, 0],
            completion: 2
        })
        .is_err());
}

#[test]
fn multi_root_interleaving_and_repeated_issue_roots_are_rejected() {
    let mut model = visit_model();
    model.static_orders[0].visits.truncate(1);
    model.static_orders[0].visits[0].roots = vec![vec![0, 2], vec![1, 3]];
    assert!(model
        .check_schedule(&Schedule {
            starts: vec![0, 1, 2, 3],
            completion: 4
        })
        .is_err());
    assert!(static_order::fits(&model, &[Some(0), None, Some(2), None]).unwrap());
    model.static_orders[0].visits[0].roots[1].push(2);
    assert!(model.lower_bound().unwrap_err().contains("repeated"));
}

#[test]
fn joint_solver_selects_a_different_emittable_order_when_it_improves_the_model() {
    use cranelift_codegen::ir::Inst;
    let mut model = visit_model();
    model.resources = vec![Resource {
        name: "issue".into(),
        capacity: 1,
        unit: CapacityUnit::Slots,
    }];
    model.operations = vec![
        Operation {
            name: "independent".into(),
            predecessors: vec![],
            start_predecessors: vec![],
            latency: 5,
            reservations: vec![Reservation {
                resource: 0,
                offset: 0,
                duration: 5,
                units: 1,
            }],
        },
        Operation {
            name: "critical input".into(),
            predecessors: vec![],
            start_predecessors: vec![],
            latency: 1,
            reservations: vec![Reservation {
                resource: 0,
                offset: 0,
                duration: 1,
                units: 1,
            }],
        },
        Operation {
            name: "critical result".into(),
            predecessors: vec![1],
            start_predecessors: vec![],
            latency: 5,
            reservations: vec![],
        },
    ];
    model.static_orders[0].instructions.push(Inst::from_u32(2));
    model.static_orders[0].predecessors.push((1, 2));
    model.static_orders[0].visits.truncate(1);
    model.static_orders[0].visits[0].roots = vec![vec![0], vec![1], vec![2]];
    let solution = model.solve(100_000).unwrap();
    assert!(solution.is_optimal());
    assert_eq!(solution.schedule().completion, 6);
    let order = static_order::orders(&model, solution.schedule()).unwrap();
    assert_eq!(order[0].instructions[0], Inst::from_u32(1));
}
