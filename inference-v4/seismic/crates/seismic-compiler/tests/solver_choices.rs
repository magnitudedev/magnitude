use magnitude_solver::{
    Limits, Options, Outcome, Search,
    model::{Constraint, Domain, ModelBuilder},
};
use seismic_compiler::tuner::choices::Choice;
use seismic_lang::lowered_ir::{Alternative, Alternatives, Decision, DecisionKind, FoldWindows};
use std::sync::Arc;

fn decision(alternatives: Alternatives) -> Decision {
    Decision {
        kind: DecisionKind::FoldPreparation { segment: 64 },
        alternatives,
    }
}
fn solve(builder: ModelBuilder) -> (Arc<magnitude_solver::Model>, magnitude_solver::Solution) {
    let model = Arc::new(builder.build().unwrap());
    let mut search = Search::new(model.clone(), Options::default()).unwrap();
    let Outcome::Optimal(solution) = search.advance(Limits::default()).unwrap() else {
        panic!("source choice equations must complete")
    };
    (model, solution)
}

#[test]
fn every_original_numeric_choice_replays_from_its_symbolic_value() {
    let domains = vec![
        Alternatives::output_widths(5).unwrap(),
        Alternatives::PacketWidths { maximum: 5 },
        Alternatives::reduction_cuts(2, 5).unwrap(),
        Alternatives::reduction_segments(5).unwrap(),
        Alternatives::UnrollWidths { maximum: 5 },
        Alternatives::MatrixPanelWidths { maximum: 5 },
        Alternatives::stream_capacities(5).unwrap(),
        Alternatives::FoldWindows(FoldWindows::new(60, &[6, 10]).unwrap()),
    ];
    for domain in domains {
        for index in 0..domain.len() {
            let source = decision(domain.clone());
            let mut builder = ModelBuilder::new();
            let choice = Choice::append(&mut builder, "choice", &source, None).unwrap();
            builder.constraint(Constraint::InDomain {
                variable: choice.ordinal,
                domain: Domain::singleton(index as i64),
            });
            let (model, solution) = solve(builder);
            let selected = choice.reconstruct(solution.values()).unwrap().unwrap();
            assert_eq!(selected, (index, domain.get(index).unwrap()));
            let actual = solution.values()[choice.numeric.unwrap().id().0];
            // Independent expected ordering, including diagnostic descending widths.
            let expected = match &domain {
                Alternatives::PacketWidths { .. }
                | Alternatives::ReductionSegments { .. }
                | Alternatives::StreamCapacities(_) => 5 - index as i64,
                Alternatives::ReductionCuts { .. } => 2 + index as i64,
                Alternatives::FoldWindows(_) => {
                    let expected: Vec<_> = (1..=60)
                        .filter(|&width| {
                            [6, 10].into_iter().all(|group| {
                                let full = (width % group == 0 && 60 % group == 0)
                                    || (group % width == 0 && 60 % width == 0);
                                let tail = 60 % width;
                                full && (tail == 0
                                    || (tail % group == 0 && (60 - tail) % group == 0)
                                    || (group % tail == 0
                                        && 60 % tail == 0
                                        && (60 - tail) % tail == 0))
                            })
                        })
                        .collect();
                    expected[index]
                }
                _ => index as i64 + 1,
            };
            assert_eq!(actual, expected);
            let mut wrong = solution.values().to_vec();
            wrong[choice.numeric.unwrap().id().0] += 1;
            assert!(
                model
                    .validate_assignment(&wrong)
                    .map_or(true, |v| v.infeasible)
            );
            assert!(choice.reconstruct(&wrong).is_err());
        }
    }
}

#[test]
fn large_domains_keep_a_constant_number_of_variables_and_factors() {
    for windows in [false, true] {
        let mut sizes = Vec::new();
        for maximum in [1024, 1i64 << 40] {
            let source = decision(if windows {
                Alternatives::FoldWindows(FoldWindows::new(maximum, &[64, 256]).unwrap())
            } else {
                Alternatives::stream_capacities(maximum).unwrap()
            });
            let mut builder = ModelBuilder::new();
            let choice = Choice::append(&mut builder, "large", &source, None).unwrap();
            builder.constraint(Constraint::InDomain {
                variable: choice.ordinal,
                domain: Domain::singleton(source.alternatives.len() as i64 - 1),
            });
            let (model, solution) = solve(builder);
            sizes.push((model.variables().len(), model.factors().len()));
            assert_eq!(
                choice.reconstruct(solution.values()).unwrap().unwrap().1,
                if windows {
                    Alternative::PreparationWindow(maximum)
                } else {
                    Alternative::StreamCapacity(1)
                }
            );
        }
        assert_eq!(sizes[0], sizes[1]);
    }
}

#[test]
fn topology_guards_keep_inactive_choices_absent_and_private() {
    for present in [0, 1] {
        let mut builder = ModelBuilder::new();
        let presence = builder.variable("parent topology", Domain::singleton(present));
        let source = decision(Alternatives::FoldWindows(
            FoldWindows::new(60, &[6, 10]).unwrap(),
        ));
        let choice = Choice::append(&mut builder, "child", &source, Some(presence)).unwrap();
        let (model, solution) = solve(builder);
        assert_eq!(
            choice.reconstruct(solution.values()).unwrap().is_some(),
            present == 1
        );
        if present == 0 {
            let mut wrong = solution.values().to_vec();
            wrong[choice.ordinal.0] = 1;
            assert!(model.validate_assignment(&wrong).unwrap().infeasible);
        }
    }
    let mut builder = ModelBuilder::new();
    let source = decision(vec![Alternative::Encoded, Alternative::Decoded].into());
    let choice = Choice::append(&mut builder, "representation", &source, None).unwrap();
    assert!(choice.numeric.is_none());
    builder.constraint(Constraint::InDomain {
        variable: choice.ordinal,
        domain: Domain::singleton(1),
    });
    let (_, solution) = solve(builder);
    assert_eq!(
        choice.reconstruct(solution.values()).unwrap().unwrap(),
        (1, Alternative::Decoded)
    );
}

#[test]
fn an_actual_stream_choice_replays_the_selected_capacity_through_lowering() {
    use seismic_lang::{
        Scope,
        program::{SourceFile, compile},
    };
    let program = compile(&[SourceFile {
        path: "symbolic_stream.seismic.portable".into(), scope: Scope::Portable,
        text: "fn stream[T](x: tensor[T] f32, out: tensor[1] f32):\n  t = load(x)\n  acc = tile[1] f32\n  for i in owned(acc): acc[i] = 0.0\n  acc[0] = reduce(t,0,sum,ordered=true)\n  store(acc,out)\n".into(),
    }], &[]).unwrap();
    let mut streams = 0;
    let lowered = seismic_lang::lower::lower_selected(
        &program,
        "stream",
        "cpu",
        &std::collections::HashMap::from([("T".into(), 137)]),
        &Default::default(),
        &Default::default(),
        &mut |source| {
            if matches!(source.kind, DecisionKind::Stream { maximum: 137, .. }) {
                streams += 1;
                let mut builder = ModelBuilder::new();
                let choice = Choice::append(&mut builder, "original stream", source, None).unwrap();
                use magnitude_solver::model::{Cost, LinearTerm};
                use seismic_accounting::algebra::{Algebra, Symbolic};
                use seismic_compiler::tuner::geometry::{Geometry, Tile};
                let one = Symbolic::new(&mut builder, "launch").constant(1).unwrap();
                let capacity = choice.numeric.unwrap();
                let geometry = Geometry::from_values(
                    &mut builder,
                    "stream geometry",
                    &[137],
                    &[capacity],
                    one,
                    one,
                    &[Tile {
                        symbol: "stream piece".into(),
                        dtype: seismic_lang::types::DType::F32,
                        capacity,
                        placement: seismic_realization::dispatch::TilePlacement::GroupShared,
                    }],
                )
                .unwrap();
                // A declared storage constraint acts on the unresolved source
                // capacity through the actual layout equations. This fixture's
                // objective is piece count, not a claim about native latency.
                builder.constraint(Constraint::LinearLe {
                    terms: vec![LinearTerm::new(
                        geometry.storage[0].shared_bytes_per_group.id(),
                        1,
                    )],
                    rhs: 20,
                });
                builder.cost(Cost::Linear {
                    constant: 0,
                    terms: vec![LinearTerm::new(geometry.work_items.id(), 1)],
                });
                let (_, solution) = solve(builder);
                let realized = geometry.reconstruct(solution.values()).unwrap();
                assert_eq!(realized.mapping.work_items(), 28);
                assert_eq!(realized.tiles[0].capacity, 5);
                assert_eq!(realized.storage[0].shared_bytes_per_group, 20);
                let (ordinal, alternative) =
                    choice.reconstruct(solution.values()).unwrap().unwrap();
                assert_eq!(ordinal, 132);
                assert_eq!(alternative, Alternative::StreamCapacity(5));
                Ok(alternative)
            } else {
                // Other transformations are diagnostic choices in this focused
                // replay test; it does not claim full source-family selection.
                source
                    .alternatives
                    .get(0)
                    .ok_or_else(|| "empty diagnostic domain".into())
            }
        },
    )
    .unwrap();
    assert_eq!(streams, 1);
    assert!(
        lowered
            .decisions
            .iter()
            .any(|d| d.selected == Alternative::StreamCapacity(5))
    );
    seismic_lang::verify::lowered(&lowered, seismic_lang::verify::Stage::Expanded).unwrap();
}
