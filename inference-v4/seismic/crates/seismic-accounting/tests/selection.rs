use seismic_accounting::{schedule::*, selection::*};

#[derive(Clone, Debug, PartialEq)]
struct FixtureChoice {
    site: u8,
}
impl Choices for FixtureChoice {
    type Alternative = u8;
    fn len(&self) -> usize {
        2
    }
    fn get(&self, index: usize) -> Option<u8> {
        (index < 2).then_some(index as u8)
    }
}
struct Tree {
    context: Context,
    models: [Model; 3],
    reject_materialization: bool,
}
impl Space for Tree {
    type Execution = usize;
    fn context(&self) -> &Context {
        &self.context
    }
    fn expand(&self, path: &[usize]) -> Result<Node<usize>, String> {
        Ok(match path {
            [] => Node::Choice {
                name: "implementation".into(),
                alternatives: Domain::new(FixtureChoice { site: 0 })?,
            },
            [0] => Node::Choice {
                name: "layout".into(),
                alternatives: Domain::new(FixtureChoice { site: 1 })?,
            },
            [0, 0] => Node::Realization(0),
            [0, 1] => Node::Realization(1),
            [1] => Node::Realization(2),
            _ => return Err("invalid decision path".into()),
        })
    }
    fn analyze(&self, execution: &usize) -> Result<Model, String> {
        Ok(self.models[*execution].clone())
    }
    fn materialize(&self, execution: &usize, _: &Objective) -> Result<usize, String> {
        if self.reject_materialization {
            Err("selected order does not preserve the execution".into())
        } else {
            Ok(*execution)
        }
    }
}
fn model(latency: u64) -> Model {
    Model {
        relationship: seismic_accounting::authority::ModelRelationship::hypothetical_execution(),
        identity: format!("conditional operation {latency}"),
        timebase: Timebase {
            seconds_numerator: 1,
            seconds_denominator: 1_000_000_000,
        },
        resources: vec![Resource {
            name: "issue".into(),
            capacity: 1,
            unit: CapacityUnit::Slots,
        }],
        operations: vec![Operation {
            name: "operation".into(),
            predecessors: vec![],
            start_predecessors: vec![],
            latency,
            reservations: vec![Reservation {
                resource: 0,
                offset: 0,
                duration: 1,
                units: 1,
            }],
        }],
        lifetimes: vec![],
        static_orders: vec![],
        unmapped: vec![],
    }
}
fn tree() -> Tree {
    Tree {
        context: Context {
            program: "fixture".into(),
            workload: "fixed".into(),
            target: "hypothetical".into(),
            contracts: "conditional issue capacities".into(),
            execution_form: "dependent tree".into(),
            objective: "completion".into(),
            seconds_numerator: 1,
            seconds_denominator: 1_000_000_000,
        },
        models: [model(9), model(3), model(7)],
        reject_materialization: false,
    }
}
fn budget(nodes: usize) -> Budget {
    Budget {
        nodes,
        schedule_assignments: 10_000,
    }
}
fn optimal<E>(outcome: Outcome<E>) -> Selected<E> {
    match outcome {
        Outcome::Optimal(s) => s,
        _ => panic!("expected completed search"),
    }
}
#[test]
fn selection_resolves_dependent_choices_and_materializes_checked_execution() {
    let mut space = tree();
    let selected = optimal(select(&space, budget(5)).unwrap());
    assert_eq!(*selected.execution(), 1);
    assert_eq!(selected.selected_path(), &[0, 1]);
    assert_eq!(selected.cost().lower(), 3);
    assert_eq!(selected.cost().upper(), 3);
    selected
        .objective()
        .model()
        .check_schedule(selected.objective().schedule())
        .unwrap();
    space.reject_materialization = true;
    assert!(matches!(select(&space, budget(5)), Err(error) if error.contains("does not preserve")));
}
#[test]
fn interruptions_preserve_incumbent_and_complete_frontier() {
    let space = tree();
    let Outcome::Incomplete(p) = select(&space, budget(3)).unwrap() else {
        panic!()
    };
    assert_eq!(
        p.frontier()
            .iter()
            .flat_map(Region::paths)
            .collect::<Vec<_>>(),
        vec![vec![1], vec![0, 1]]
    );
    assert_eq!(p.incumbent(), Some(&0));
    assert_eq!(p.feasible_upper(), Some(9));
    assert_eq!(p.lower_bound().unwrap(), 0);
    assert_eq!(
        *optimal(resume(&space, p, budget(2)).unwrap()).execution(),
        1
    );
}
#[test]
fn cached_analyses_are_bound_to_execution_constraints_not_labels() {
    let mut space = tree();
    let Outcome::Incomplete(p) = select(&space, budget(3)).unwrap() else {
        panic!()
    };
    space.models[0].operations[0].latency = 8;
    assert!(
        matches!(resume(&space, p, budget(2)), Err(error) if error.contains("analysis changed"))
    );
}
#[test]
fn infeasible_or_optimistic_analysis_cannot_supply_an_incumbent() {
    let mut space = tree();
    space.models[1].operations[0].reservations[0].units = 2;
    // Exclude only that impossible realization; another legal choice remains.
    assert_eq!(*optimal(select(&space, budget(5)).unwrap()).execution(), 2);
    space = tree();
    space.models[1].timebase.seconds_denominator = 1;
    assert!(select(&space, budget(5)).is_err());
    space = tree();
    space.models[1].relationship =
        seismic_accounting::authority::ModelRelationship::OptimisticRelaxation;
    assert!(
        matches!(select(&space, budget(5)), Err(error) if error.contains("optimistic relaxation"))
    );
}
#[test]
fn incomplete_schedule_search_is_retained_and_resumed() {
    let mut space = tree();
    let mut m = model(3);
    m.resources.push(Resource {
        name: "second".into(),
        capacity: 1,
        unit: CapacityUnit::Slots,
    });
    let mut second = m.operations[0].clone();
    second.name = "second".into();
    second.reservations[0].resource = 1;
    m.operations.push(second);
    space.models = [m.clone(), m.clone(), m];
    let Outcome::Incomplete(p) = select(
        &space,
        Budget {
            nodes: 5,
            schedule_assignments: 0,
        },
    )
    .unwrap() else {
        panic!()
    };
    assert!(p.frontier().is_empty());
    assert_eq!(p.lower_bound().unwrap(), 3);
    assert_eq!(p.feasible_upper(), Some(6));
    assert_eq!(
        optimal(resume(&space, p, budget(0)).unwrap())
            .cost()
            .lower(),
        3
    );
}
#[test]
fn large_domains_retain_exact_frontier_without_expanding_each_alternative() {
    struct Wide(Tree);
    impl Space for Wide {
        type Execution = usize;
        fn context(&self) -> &Context {
            self.0.context()
        }
        fn expand(&self, path: &[usize]) -> Result<Node<usize>, String> {
            if path.is_empty() {
                Ok(Node::Choice {
                    name: "piece capacity".into(),
                    alternatives: Domain::new(IntegerRange::new(
                        FixtureChoice { site: 2 },
                        1,
                        1_000_000_000_000,
                    )?)?,
                })
            } else {
                Ok(Node::Realization(path[0]))
            }
        }
        fn analyze(&self, _: &usize) -> Result<Model, String> {
            Ok(self.0.models[0].clone())
        }
        fn materialize(&self, execution: &usize, _: &Objective) -> Result<usize, String> {
            Ok(*execution)
        }
    }
    let space = Wide(tree());
    let Outcome::Incomplete(p) = select(&space, budget(2)).unwrap() else {
        panic!()
    };
    assert_eq!(p.nodes_visited(), 2);
    assert_eq!(p.frontier().len(), 1);
    assert_eq!(p.frontier()[0].len(), 999_999_999_999);
    assert_eq!(p.frontier()[0].paths().next(), Some(vec![1]));
    let Outcome::Incomplete(p) = resume(&space, p, budget(2)).unwrap() else {
        panic!()
    };
    assert_eq!(p.frontier()[0].paths().next(), Some(vec![3]));
    assert!(IntegerRange::new(FixtureChoice { site: 0 }, 0, u64::MAX).is_err());
}

#[test]
fn typed_domains_keep_semantic_identity_and_alternatives() {
    let a = Domain::new(FixtureChoice { site: 0 }).unwrap();
    let b = Domain::new(FixtureChoice { site: 1 }).unwrap();
    assert_eq!(a.label(0), b.label(0));
    assert_ne!(a, b); // identical display labels do not erase the decision site
    let source = a.owner::<FixtureChoice>().unwrap();
    assert_eq!(source.site, 0);
    assert_eq!(source.get(1), Some(1));
    let range = IntegerRange::new(FixtureChoice { site: 3 }, 9, 2).unwrap();
    assert_eq!(range.get(7), Some(2));
    assert_eq!(range.index(2), Some(7));
    assert_eq!(range.index(10), None);
}

#[test]
fn resumed_incumbent_uses_the_current_execution_even_when_its_bound_is_exact() {
    struct Current {
        tree: Tree,
        generation: u64,
    }
    impl Space for Current {
        type Execution = (usize, u64);
        fn context(&self) -> &Context {
            self.tree.context()
        }
        fn expand(&self, path: &[usize]) -> Result<Node<Self::Execution>, String> {
            Ok(match self.tree.expand(path)? {
                Node::Choice { name, alternatives } => Node::Choice { name, alternatives },
                Node::Realization(index) => Node::Realization((index, self.generation)),
                Node::Infeasible(reason) => Node::Infeasible(reason),
            })
        }
        fn analyze(&self, execution: &Self::Execution) -> Result<Model, String> {
            self.tree.analyze(&execution.0)
        }
        fn materialize(
            &self,
            execution: &Self::Execution,
            _: &Objective,
        ) -> Result<Self::Execution, String> {
            if execution.1 != self.generation {
                return Err("stale source execution".into());
            }
            Ok(*execution)
        }
    }
    let mut space = Current {
        tree: tree(),
        generation: 0,
    };
    space.tree.models = [model(3), model(9), model(7)];
    let Outcome::Incomplete(progress) = select(&space, budget(3)).unwrap() else {
        panic!()
    };
    assert_eq!(progress.incumbent(), Some(&(0, 0)));
    space.generation = 1;
    let selected = optimal(resume(&space, progress, budget(2)).unwrap());
    assert_eq!(selected.execution(), &(0, 1));
    assert_eq!(selected.cost().upper(), 3);
}

#[test]
fn missing_initial_witness_is_resumable_and_not_infeasibility() {
    let mut space = tree();
    for m in &mut space.models {
        m.operations[0].reservations[0].units = 2;
    }
    let Outcome::Incomplete(progress) = select(
        &space,
        Budget {
            nodes: 5,
            schedule_assignments: 0,
        },
    )
    .unwrap() else {
        panic!("no initial witness must leave the schedule search unresolved");
    };
    assert!(progress.incumbent().is_none());
    assert!(progress.frontier().is_empty());
    assert_eq!(progress.lower_bound().unwrap(), 3);
    assert!(matches!(
        resume(&space, progress, budget(0)).unwrap(),
        Outcome::Infeasible
    ));
}

#[test]
fn incomplete_resource_information_keeps_known_bounds_without_an_upper() {
    let mut space = tree();
    for model in &mut space.models {
        model.unmapped.push("remaining instruction service".into());
    }
    let Outcome::Incomplete(progress) = select(&space, budget(5)).unwrap() else {
        panic!("missing service facts cannot complete selection");
    };
    assert_eq!(progress.lower_bound().unwrap(), 3);
    assert_eq!(progress.feasible_upper(), None);
    assert!(progress.frontier().is_empty());
}
