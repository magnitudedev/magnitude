use seismic_accounting::selection::*;

struct Tree {
    context: Context,
    costs: [Cost; 3],
}
impl Space for Tree {
    type Execution = usize;
    fn context(&self) -> &Context {
        &self.context
    }
    fn expand(&self, prefix: &[usize]) -> Result<Node<usize>, String> {
        Ok(match prefix {
            [] => Node::Choice {
                name: "implementation".into(),
                alternatives: vec!["a".into(), "b".into()],
            },
            [0] => Node::Choice {
                name: "dependent layout".into(),
                alternatives: vec!["x".into(), "y".into()],
            },
            [0, 0] => Node::Realization(Realization {
                execution: 0,
                cost: self.costs[0],
            }),
            [0, 1] => Node::Realization(Realization {
                execution: 1,
                cost: self.costs[1],
            }),
            [1] => Node::Realization(Realization {
                execution: 2,
                cost: self.costs[2],
            }),
            _ => return Err("invalid decision path".into()),
        })
    }
}
fn tree(costs: [Cost; 3]) -> Tree {
    Tree {
        context: Context {
            program: "fixture program".into(),
            workload: "fixture workload".into(),
            target: "synthetic target".into(),
            contracts: "fixture contracts".into(),
            execution_form: "dependent two-level tree".into(),
            objective: "latency".into(),
            seconds_numerator: 1,
            seconds_denominator: 1_000_000_000,
        },
        costs,
    }
}
#[test]
fn complete_selection_and_independent_replay_reject_forged_coverage_cost_and_assumptions() {
    let space = tree([Cost::exact(9), Cost::exact(3), Cost::exact(7)]);
    let selected = select(&space, 5).unwrap();
    assert_eq!(*selected.execution(), 1);
    assert_eq!(
        verify(&space, selected.certificate(), 5).unwrap(),
        Cost::exact(3)
    );
    assert!(select(&space, 4).is_err());
    assert!(verify(&space, selected.certificate(), 4).is_err());
    let mut forged = selected.certificate().clone();
    forged.records.pop();
    assert!(verify(&space, &forged, 5).is_err());
    let mut forged = selected.certificate().clone();
    forged.selected = vec![1];
    assert!(verify(&space, &forged, 5).is_err());
    let mut forged = selected.certificate().clone();
    forged.records[3].evidence = Evidence::Realization(Cost::exact(1));
    assert!(verify(&space, &forged, 5).is_err());
    let mut forged = selected.certificate().clone();
    forged.context.contracts = "different model".into();
    assert!(verify(&space, &forged, 5).is_err());
}
#[test]
fn bounds_require_dominance_and_equal_upper_bounds_do_not_hide_a_winner() {
    let space = tree([Cost::exact(5), Cost::bounded(1, 5).unwrap(), Cost::exact(9)]);
    let selected = select(&space, 5).unwrap();
    assert_eq!(*selected.execution(), 1);
    verify(&space, selected.certificate(), 5).unwrap();
    let ambiguous = tree([
        Cost::bounded(1, 6).unwrap(),
        Cost::bounded(3, 5).unwrap(),
        Cost::exact(9),
    ]);
    assert!(select(&ambiguous, 5).is_err());
}
