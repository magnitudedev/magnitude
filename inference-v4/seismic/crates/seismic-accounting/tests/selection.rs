use seismic_accounting::{
    schedule::*,
    selection::*,
    workload::{DerivationError, DerivationLimit, DerivationLimits},
};

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
    type Identity = ([Model; 3], bool);
    fn identity(&self) -> Self::Identity {
        (self.models.clone(), self.reject_materialization)
    }
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
    fn analyze(&self, execution: &usize) -> Result<Model, DerivationError> {
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
fn optimal<E, I>(outcome: Outcome<E, I>) -> Selected<E> {
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
    assert!(matches!(resume(&space, p, budget(2)), Err(error) if error.contains("inputs changed")));
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
    assert_eq!(
        p.unresolved(),
        Unresolved {
            schedules: 3,
            ..Unresolved::default()
        }
    );
    assert_eq!(p.lower_bound().unwrap(), 3);
    assert_eq!(p.feasible_upper(), Some(6));
    assert_eq!(
        optimal(resume(&space, p, budget(0)).unwrap())
            .cost()
            .lower(),
        3
    );
}

#[derive(Clone, Copy, PartialEq)]
enum Refinement {
    All,
    NestedOnly,
    Fail,
}
struct RetainedTree {
    tree: Tree,
    mode: Refinement,
    expanded: std::cell::RefCell<Vec<Vec<usize>>>,
    refined: std::cell::RefCell<Vec<(u8, usize)>>,
}
impl RetainedTree {
    fn new(mode: Refinement) -> Self {
        Self {
            tree: tree(),
            mode,
            expanded: Default::default(),
            refined: Default::default(),
        }
    }
}
impl Space for RetainedTree {
    type Execution = usize;
    type Identity = (<Tree as Space>::Identity, Refinement);
    fn identity(&self) -> Self::Identity {
        (self.tree.identity(), self.mode)
    }
    fn context(&self) -> &Context {
        self.tree.context()
    }
    fn expand(&self, path: &[usize]) -> Result<Node<usize>, String> {
        self.expanded.borrow_mut().push(path.to_vec());
        self.tree.expand(path)
    }
    fn refine(&self, domain: &Domain, index: usize) -> Result<Option<Node<usize>>, String> {
        let Some(owner) = domain.owner::<FixtureChoice>() else {
            return Ok(None);
        };
        if self.mode == Refinement::NestedOnly && owner.site == 0 {
            return Ok(None);
        }
        if self.mode == Refinement::Fail {
            return Err("retained construction failed".into());
        }
        self.refined.borrow_mut().push((owner.site, index));
        Ok(Some(match (owner.site, index) {
            (0, 0) => Node::Choice {
                name: "layout".into(),
                alternatives: Domain::new(FixtureChoice { site: 1 })?,
            },
            (0, 1) => Node::Realization(2),
            (1, 0) => Node::Realization(0),
            (1, 1) => Node::Realization(1),
            _ => return Err("invalid retained choice".into()),
        }))
    }
    fn analyze(&self, execution: &usize) -> Result<Model, DerivationError> {
        self.tree.analyze(execution)
    }
    fn materialize(&self, execution: &usize, objective: &Objective) -> Result<usize, String> {
        self.tree.materialize(execution, objective)
    }
}
#[test]
fn retained_dependent_owners_preserve_optimum_and_interrupted_coverage() {
    let expected = optimal(select(&tree(), budget(5)).unwrap());
    let space = RetainedTree::new(Refinement::All);
    let Outcome::Incomplete(progress) = select(&space, budget(3)).unwrap() else {
        panic!("expected retained frontier");
    };
    assert_eq!(
        progress
            .frontier()
            .iter()
            .flat_map(Region::paths)
            .collect::<Vec<_>>(),
        vec![vec![1], vec![0, 1]]
    );
    assert_eq!(progress.incumbent(), Some(&0));
    let actual = optimal(resume(&space, progress, budget(2)).unwrap());
    assert_eq!(actual.execution(), expected.execution());
    assert_eq!(actual.selected_path(), expected.selected_path());
    assert_eq!(actual.objective(), expected.objective());
    assert_eq!(&*space.expanded.borrow(), &[Vec::<usize>::new()]);
    assert_eq!(&*space.refined.borrow(), &[(0, 0), (1, 0), (1, 1), (0, 1)]);
}
#[test]
fn improved_schedule_recovers_nonincumbent_through_its_retained_owner() {
    let mut space = RetainedTree::new(Refinement::All);
    let mut parallel = model(3);
    parallel.resources.push(Resource {
        name: "independent issue".into(),
        capacity: 1,
        unit: CapacityUnit::Slots,
    });
    let mut second = parallel.operations[0].clone();
    second.name = "independent operation".into();
    second.reservations[0].resource = 1;
    parallel.operations.push(second);
    space.tree.models = [model(5), parallel, model(7)];
    let Outcome::Incomplete(progress) = select(
        &space,
        Budget {
            nodes: 5,
            schedule_assignments: 0,
        },
    )
    .unwrap() else {
        panic!("parallel schedule needs further search");
    };
    assert!(progress.frontier().is_empty());
    assert_eq!(progress.incumbent(), Some(&0));
    assert_eq!(progress.feasible_upper(), Some(5));
    assert_eq!(progress.lower_bound().unwrap(), 3);
    let selected = optimal(resume(&space, progress, budget(0)).unwrap());
    assert_eq!(selected.execution(), &1);
    assert_eq!(selected.selected_path(), &[0, 1]);
    assert_eq!(selected.cost().upper(), 3);
    assert_eq!(&*space.expanded.borrow(), &[Vec::<usize>::new()]);
    assert_eq!(space.refined.borrow().last(), Some(&(1, 1)));
    assert_eq!(
        space
            .refined
            .borrow()
            .iter()
            .filter(|&&v| v == (1, 1))
            .count(),
        2
    );
}
#[test]
fn unhandled_owner_uses_full_path_but_construction_errors_propagate() {
    let space = RetainedTree::new(Refinement::NestedOnly);
    let selected = optimal(select(&space, budget(5)).unwrap());
    assert_eq!(selected.execution(), &1);
    assert_eq!(&*space.expanded.borrow(), &[vec![], vec![0], vec![1]]);
    assert_eq!(&*space.refined.borrow(), &[(1, 0), (1, 1)]);
    let space = RetainedTree::new(Refinement::Fail);
    assert!(
        matches!(select(&space, budget(5)), Err(error) if error == "retained construction failed")
    );
    assert_eq!(&*space.expanded.borrow(), &[Vec::<usize>::new()]);
}
#[test]
fn changed_inputs_reject_retained_frontier_before_refinement() {
    let mut space = RetainedTree::new(Refinement::All);
    let Outcome::Incomplete(progress) = select(&space, budget(2)).unwrap() else {
        panic!("expected nested retained domain");
    };
    let refinements = space.refined.borrow().clone();
    space.tree.models[1].operations[0].latency = 4;
    assert!(
        matches!(resume(&space, progress, budget(3)), Err(error) if error.contains("inputs changed"))
    );
    assert_eq!(*space.refined.borrow(), refinements);
    assert_eq!(&*space.expanded.borrow(), &[Vec::<usize>::new()]);
}
#[test]
fn large_domains_retain_exact_frontier_without_expanding_each_alternative() {
    struct Wide(Tree);
    impl Space for Wide {
        type Execution = usize;
        type Identity = <Tree as Space>::Identity;
        fn identity(&self) -> Self::Identity {
            self.0.identity()
        }
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
        fn analyze(&self, _: &usize) -> Result<Model, DerivationError> {
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
    assert!(p.frontier().len() <= 64);
    assert_eq!(
        p.frontier().iter().map(Region::len).sum::<usize>(),
        999_999_999_999
    );
    let mut intervals = p
        .frontier()
        .iter()
        .map(|r| r.alternatives().unwrap().1)
        .collect::<Vec<_>>();
    intervals.sort_by_key(|r| r.start);
    assert_eq!(intervals[0].start, 1);
    assert_eq!(intervals.last().unwrap().end, 1_000_000_000_000);
    assert!(intervals.windows(2).all(|r| r[0].end == r[1].start));
    let Outcome::Incomplete(p) = resume(&space, p, budget(2)).unwrap() else {
        panic!()
    };
    assert_eq!(
        p.frontier().iter().filter_map(|r| r.paths().next()).min(),
        Some(vec![3])
    );
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
fn unchanged_inputs_reuse_visited_choices_and_models_on_resumption() {
    struct Counted {
        tree: Tree,
        expansions: std::cell::Cell<usize>,
        analyses: std::cell::Cell<usize>,
    }
    impl Space for Counted {
        type Execution = usize;
        type Identity = <Tree as Space>::Identity;
        fn context(&self) -> &Context {
            self.tree.context()
        }
        fn identity(&self) -> Self::Identity {
            self.tree.identity()
        }
        fn expand(&self, path: &[usize]) -> Result<Node<usize>, String> {
            self.expansions.set(self.expansions.get() + 1);
            self.tree.expand(path)
        }
        fn analyze(&self, execution: &usize) -> Result<Model, DerivationError> {
            self.analyses.set(self.analyses.get() + 1);
            self.tree.analyze(execution)
        }
        fn materialize(&self, execution: &usize, objective: &Objective) -> Result<usize, String> {
            self.tree.materialize(execution, objective)
        }
    }
    let space = Counted {
        tree: tree(),
        expansions: Default::default(),
        analyses: Default::default(),
    };
    let Outcome::Incomplete(progress) = select(&space, budget(3)).unwrap() else {
        panic!()
    };
    assert_eq!(
        progress
            .decision(&[])
            .unwrap()
            .owner::<FixtureChoice>()
            .unwrap()
            .site,
        0
    );
    assert_eq!(
        *optimal(resume(&space, progress, budget(2)).unwrap()).execution(),
        1
    );
    assert_eq!(space.expansions.get(), 5);
    assert_eq!(space.analyses.get(), 3);
}

#[test]
fn changed_execution_identity_invalidates_cached_costs_even_when_the_cost_is_equal() {
    struct Current {
        tree: Tree,
        generation: u64,
    }
    impl Space for Current {
        type Execution = (usize, u64);
        type Identity = (<Tree as Space>::Identity, u64);
        fn identity(&self) -> Self::Identity {
            (self.tree.identity(), self.generation)
        }
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
        fn analyze(&self, execution: &Self::Execution) -> Result<Model, DerivationError> {
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
    assert!(
        matches!(resume(&space, progress, budget(2)), Err(error) if error.contains("inputs changed"))
    );
    let selected = optimal(select(&space, budget(5)).unwrap());
    assert_eq!(selected.execution(), &(0, 1));
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
    assert_eq!(
        progress.unresolved(),
        Unresolved {
            unmapped_models: 3,
            ..Unresolved::default()
        }
    );
}

#[test]
fn symbolic_region_bounds_find_an_interior_optimum_without_enumerating_the_domain() {
    struct Family {
        context: Context,
        optimum: usize,
        analyses: std::cell::Cell<usize>,
    }
    impl Family {
        fn operation(&self, index: usize) -> Model {
            model(index.abs_diff(self.optimum) as u64 + 1)
        }
    }
    impl Space for Family {
        type Execution = usize;
        type Identity = usize;
        fn identity(&self) -> usize {
            self.optimum
        }
        fn context(&self) -> &Context {
            &self.context
        }
        fn expand(&self, path: &[usize]) -> Result<Node<usize>, String> {
            Ok(match path {
                [] => Node::Choice {
                    name: "retained implementation interval".into(),
                    alternatives: Domain::new(IntegerRange::new(
                        FixtureChoice { site: 0 },
                        0,
                        999_999_999_999,
                    )?)?,
                },
                [index] => Node::Realization(*index),
                _ => return Err("invalid path".into()),
            })
        }
        fn relax(
            &self,
            domain: &Domain,
            range: std::ops::Range<usize>,
        ) -> Result<Option<Demand>, String> {
            let owner = domain
                .owner::<IntegerRange<FixtureChoice>>()
                .ok_or("wrong typed decision")?;
            let first = owner.get(range.start).ok_or("invalid first index")? as usize;
            let last = owner.get(range.end - 1).ok_or("invalid last index")? as usize;
            // In this retained family latency is distance from the optimum plus
            // one. Projection onto the interval minimizes distance for every member.
            let witness = self.operation(self.optimum.clamp(first, last));
            let mut demand = Demand::new(witness.timebase, witness.resources)?;
            demand.include(&witness.operations[0], 1)?;
            Ok(Some(demand))
        }
        fn analyze(&self, execution: &usize) -> Result<Model, DerivationError> {
            self.analyses.set(self.analyses.get() + 1);
            Ok(self.operation(*execution))
        }
        fn materialize(&self, execution: &usize, _: &Objective) -> Result<usize, String> {
            Ok(*execution)
        }
    }
    let family = Family {
        context: tree().context,
        optimum: 743_987_654_321,
        analyses: Default::default(),
    };
    let selected = optimal(select(&family, budget(2)).unwrap());
    assert_eq!(*selected.execution(), family.optimum);
    assert_eq!(selected.cost().upper(), 1);
    assert_eq!(family.analyses.get(), 1);
}

struct LimitedDerivation {
    tree: Tree,
    limits: std::cell::Cell<DerivationLimits>,
    analyses: std::cell::RefCell<[usize; 3]>,
    expansions: std::cell::Cell<usize>,
    unsupported: bool,
}
impl Space for LimitedDerivation {
    type Execution = usize;
    type Identity = (<Tree as Space>::Identity, bool);
    fn identity(&self) -> Self::Identity {
        (self.tree.identity(), self.unsupported)
    }
    fn context(&self) -> &Context {
        self.tree.context()
    }
    fn expand(&self, path: &[usize]) -> Result<Node<usize>, String> {
        self.expansions.set(self.expansions.get() + 1);
        self.tree.expand(path)
    }
    fn analyze(&self, execution: &usize) -> Result<Model, DerivationError> {
        self.analyses.borrow_mut()[*execution] += 1;
        if *execution == 1 {
            if self.unsupported {
                // Similar diagnostic text must not turn an ordinary compiler
                // failure into an exhausted construction budget.
                return Err("unsupported operation named budget exhausted".into());
            }
            let limits = self.limits.get();
            if limits.instructions < 10 {
                return Err(DerivationError::Exhausted(DerivationLimit::Instructions(
                    limits.instructions,
                )));
            }
            if limits.operations < 10 {
                return Err(DerivationError::Exhausted(DerivationLimit::Operations(
                    limits.operations,
                )));
            }
        }
        self.tree.analyze(execution)
    }
    fn materialize(&self, execution: &usize, objective: &Objective) -> Result<usize, String> {
        self.tree.materialize(execution, objective)
    }
}

#[test]
fn exhausted_model_derivation_retains_incumbent_leaf_and_completed_models() {
    for limits in [
        DerivationLimits {
            instructions: 2,
            operations: 100,
        },
        DerivationLimits {
            instructions: 100,
            operations: 2,
        },
    ] {
        let space = LimitedDerivation {
            tree: tree(),
            limits: std::cell::Cell::new(limits),
            analyses: Default::default(),
            expansions: Default::default(),
            unsupported: false,
        };
        let Outcome::Incomplete(progress) = select(&space, budget(5)).unwrap() else {
            panic!("the cheaper unresolved execution must remain in the frontier");
        };
        assert_eq!(progress.incumbent(), Some(&2));
        assert_eq!(progress.feasible_upper(), Some(7));
        assert_eq!(progress.lower_bound().unwrap(), 0);
        assert!(progress.frontier().is_empty());
        assert_eq!(
            progress.unresolved(),
            Unresolved {
                derivations: 1,
                ..Unresolved::default()
            }
        );
        let expected = if limits.instructions < 10 {
            DerivationLimit::Instructions(limits.instructions)
        } else {
            DerivationLimit::Operations(limits.operations)
        };
        assert_eq!(
            progress.exhausted_derivations().collect::<Vec<_>>(),
            vec![(&[0, 1][..], expected)]
        );
        assert_eq!(*space.analyses.borrow(), [1, 1, 1]);

        let Outcome::Incomplete(progress) = resume(&space, progress, budget(0)).unwrap() else {
            panic!("unchanged limits do not establish completion");
        };
        assert_eq!(*space.analyses.borrow(), [1, 2, 1]);
        assert_eq!(progress.feasible_upper(), Some(7));
        space.limits.set(DerivationLimits {
            instructions: 100,
            operations: 100,
        });
        let selected = optimal(resume(&space, progress, budget(0)).unwrap());
        assert_eq!(*selected.execution(), 1);
        assert_eq!(
            selected.cost(),
            optimal(select(&space.tree, budget(5)).unwrap()).cost()
        );
        assert_eq!(*space.analyses.borrow(), [1, 3, 1]);
        assert_eq!(
            space.expansions.get(),
            5,
            "deferred execution is retained, not rebuilt"
        );
    }
}

#[test]
fn deferred_derivation_still_rejects_changed_semantics_and_unsupported_analysis() {
    let mut space = LimitedDerivation {
        tree: tree(),
        limits: std::cell::Cell::new(DerivationLimits {
            instructions: 1,
            operations: 1,
        }),
        analyses: Default::default(),
        expansions: Default::default(),
        unsupported: false,
    };
    let Outcome::Incomplete(progress) = select(&space, budget(5)).unwrap() else {
        panic!()
    };
    space.tree.models[1] = model(2);
    assert!(matches!(resume(&space, progress, budget(0)), Err(e) if e.contains("inputs changed")));
    space.unsupported = true;
    assert!(matches!(select(&space, budget(5)), Err(e) if e.contains("unsupported operation")));
}
