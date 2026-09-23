//! Evaluator-owned traversal of source construction choices. The domain owns
//! the suspended builders; this cursor chooses which path receives work next.

use crate::candidate_domain::{
    CandidateDomain, ConstructionAllowance, ConstructionCoordinate, ConstructionPending,
    Materialization,
};
use seismic_target::TargetFamily;
use std::collections::VecDeque;
use std::time::Duration;

pub(crate) struct ConstructionTraversal {
    frontier: VecDeque<ConstructionCoordinate>,
    unresolved: Vec<ConstructionCoordinate>,
}

impl ConstructionTraversal {
    pub(crate) fn new<B: TargetFamily>(domain: &CandidateDomain<'_, B>) -> Self {
        let general = domain.general_construction();
        Self {
            frontier: domain
                .root_selections()
                .into_iter()
                .map(ConstructionCoordinate::root)
                .filter(|coordinate| coordinate != &general)
                .collect(),
            unresolved: Vec::new(),
        }
    }
    pub(crate) fn complete(&self) -> bool {
        self.frontier.is_empty() && self.unresolved.is_empty()
    }
    pub(crate) fn pending_paths(&self) -> usize {
        self.frontier.len()
    }
    pub(crate) fn initialization_pending_paths(&self) -> usize {
        self.unresolved.len()
    }
    pub(crate) fn has_work(&self) -> bool {
        !self.frontier.is_empty()
    }

    pub(crate) fn advance<B: TargetFamily>(
        &mut self,
        domain: &mut CandidateDomain<'_, B>,
        allowance: &mut ConstructionAllowance,
    ) {
        while allowance.work_units > 0 && allowance.wall_time > Duration::ZERO {
            let Some(coordinate) = self.frontier.pop_front() else {
                break;
            };
            let result = domain.advance(&coordinate, *allowance);
            allowance.work_units = allowance.work_units.saturating_sub(result.work_units);
            allowance.wall_time = allowance.wall_time.saturating_sub(result.elapsed);
            match result.state {
                Materialization::Choice(choice) => {
                    // Queue every alternative fairly. Whichever successor is
                    // visited first consumes the retained parent; later siblings
                    // reconstruct their prefix under charged allowance.
                    for selected in &choice.alternatives {
                        self.frontier
                            .push_back(coordinate.select(&choice, selected.clone()));
                    }
                }
                Materialization::Pending(ConstructionPending::WorkAllowance) => {
                    self.frontier.push_back(coordinate);
                    break;
                }
                Materialization::Pending(ConstructionPending::Initialization(_) | ConstructionPending::Capacity(_) | ConstructionPending::RepresentationView { .. }) => {
                    self.unresolved.push(coordinate);
                }
                Materialization::Ready(_) | Materialization::Excluded(_) => {}
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn allowance_pause_rotates_pending_paths_without_declaring_completion() {
        use crate::realization::demand_driven_tests::registry;
        use crate::realization::demand_driven_tests::device;
        use seismic_lang::checked::{check_source, SourceFile, SourceSet};
        let module = check_source(SourceSet::new(vec![SourceFile {
            path: "construction-fairness.seismic".into(),
            text: "fn helper(x: f32) -> f32:\n    return x\n\nfn probe(x: f32) -> f32:\n    return helper(x)\n".into(),
        }])).unwrap();
        let target = device();
        let registry = registry();
        let entry = module
            .entry(module.entry_named("probe").unwrap(), &Default::default())
            .unwrap();
        let mut domain = crate::candidate_domain::construct_candidate_domain(
            entry,
            &target,
            &registry,
            &seismic_lang::precision::PrecisionPolicy::Exact,
        )
        .unwrap();
        let mut traversal = ConstructionTraversal::new(&domain);
        assert_eq!(traversal.pending_paths(), 2);
        let first = traversal.frontier[0].clone();
        let second = traversal.frontier[1].clone();
        traversal.advance(
            &mut domain,
            &mut ConstructionAllowance {
                work_units: 1,
                wall_time: Duration::from_secs(30),
            },
        );
        assert_eq!(traversal.frontier.front(), Some(&second));
        assert_eq!(traversal.frontier.back(), Some(&first));
        assert!(!traversal.complete());
        traversal.advance(
            &mut domain,
            &mut ConstructionAllowance {
                work_units: 1,
                wall_time: Duration::from_secs(30),
            },
        );
        assert_eq!(traversal.frontier.front(), Some(&first));
        assert!(!traversal.complete());
    }
}
