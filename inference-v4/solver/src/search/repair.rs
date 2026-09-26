//! Search a restriction of an immutable model; all proof status stays local.
use super::{engine, Limits, Options, Stats};
use crate::{Domain, Error, FeasibleSolution, Model, Result};
use std::sync::Arc;

pub(super) struct Repair {
    search: engine::Search,
    pub unrestricted: bool,
}

impl Repair {
    #[cfg(test)]
    pub fn new(
        model: Arc<Model>,
        options: Options,
        domains: Vec<Domain>,
        preferred: Vec<i64>,
        seed: Option<&FeasibleSolution>,
    ) -> Result<Self> {
        let knowledge = super::knowledge::Knowledge::shared(model.clone());
        Self::with_knowledge(model, options, domains, preferred, seed, knowledge)
    }
    pub fn with_knowledge(
        model: Arc<Model>,
        options: Options,
        domains: Vec<Domain>,
        preferred: Vec<i64>,
        seed: Option<&FeasibleSolution>,
        knowledge: super::knowledge::SharedKnowledge,
    ) -> Result<Self> {
        if domains.len() != model.variables().len() || preferred.len() != domains.len() {
            return Err(Error::InvalidModel("repair context has wrong arity".into()));
        }
        let mut unrestricted = true;
        for (domain, variable) in domains.iter().zip(model.variables()) {
            domain.validate()?;
            if domain.intersect(&variable.domain)? != *domain {
                return Err(Error::InvalidModel(
                    "repair expands an original domain".into(),
                ));
            }
            unrestricted &= *domain == variable.domain;
        }
        let incumbent = if let Some(seed) = seed {
            if !Arc::ptr_eq(&model, &seed.model) {
                return Err(Error::InvalidModel(
                    "repair seed belongs to another model".into(),
                ));
            }
            if seed
                .values()
                .iter()
                .zip(&domains)
                .all(|(v, d)| d.contains(*v))
            {
                let checked = model.validate_assignment(seed.values())?;
                if checked.infeasible || checked.exact_cost != Some(seed.cost()) {
                    return Err(Error::Invariant("invalid repair seed".into()));
                }
                Some(seed.clone())
            } else {
                None
            }
        } else {
            None
        };
        Ok(Self {
            search: engine::Search::with_knowledge(
                model,
                options,
                domains,
                Some(preferred),
                incumbent,
                knowledge,
            )?,
            unrestricted,
        })
    }
    pub fn advance(&mut self, limits: Limits) -> Result<engine::Outcome> {
        self.search.advance_to_witness(limits)
    }
    pub fn update_memory(&mut self) {
        self.search.update_memory();
    }
    pub fn stats(&self) -> &Stats {
        self.search.stats()
    }
    pub fn advance_peak_retained_bytes(&self) -> usize {
        self.search.advance_peak_retained_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Constraint, Cost, Literal, ModelBuilder};
    use crate::result::StopReason;
    use crate::search::Policy;

    fn model() -> Arc<Model> {
        let mut b = ModelBuilder::new();
        let x = b.variable("x", Domain::boolean());
        b.cost(Cost::Table {
            variables: vec![x],
            entries: vec![(vec![0], 10), (vec![1], 1)],
        });
        // No factors mention this variable. Its fixed assumption must survive
        // witness reconstruction too.
        b.variable("unused", Domain::boolean());
        Arc::new(b.build().unwrap())
    }
    #[test]
    fn local_optimum_has_local_scope_and_preserves_unused_assumptions() {
        let m = model();
        let mut r = Repair::new(
            m.clone(),
            Options::default(),
            vec![Domain::singleton(0), Domain::singleton(1)],
            vec![0, 1],
            None,
        )
        .unwrap();
        assert!(!r.unrestricted);
        match r.advance(Limits::default()).unwrap() {
            engine::Outcome::Optimal(w) => {
                assert_eq!(w.cost(), 10);
                assert_eq!(w.values(), &[0, 1]);
            }
            _ => panic!("repair should complete"),
        }
        let mut other = Repair::new(
            m,
            Options::default(),
            vec![Domain::singleton(1), Domain::singleton(0)],
            vec![1, 0],
            None,
        )
        .unwrap();
        match other.advance(Limits::default()).unwrap() {
            engine::Outcome::Optimal(w) => assert_eq!(w.cost(), 1),
            _ => panic!("different assumptions must not inherit old proof"),
        }
    }
    #[test]
    fn impossible_local_context_does_not_poison_unrestricted_search() {
        let mut b = ModelBuilder::new();
        let x = b.variable("x", Domain::boolean());
        let y = b.variable("y", Domain::boolean());
        b.constraint(Constraint::Equal { left: x, right: y });
        let m = Arc::new(b.build().unwrap());
        let mut r = Repair::new(
            m.clone(),
            Options::default(),
            vec![Domain::singleton(0), Domain::singleton(1)],
            vec![0, 1],
            None,
        )
        .unwrap();
        assert!(!r.unrestricted);
        assert!(matches!(
            r.advance(Limits::default()).unwrap(),
            engine::Outcome::Infeasible
        ));
        let mut global =
            Repair::new(m.clone(), Options::default(), m.domains(), vec![0, 1], None).unwrap();
        assert!(global.unrestricted);
        assert!(matches!(
            global.advance(Limits::default()).unwrap(),
            engine::Outcome::Optimal(_)
        ));
    }

    #[test]
    fn best_first_repair_finishes_preferred_branch_before_cheaper_siblings() {
        let m = model();
        let options = Options {
            policy: Policy::BestFirst,
            ..Default::default()
        };
        let mut repair =
            Repair::new(m.clone(), options.clone(), m.domains(), vec![0, 0], None).unwrap();
        let engine::Outcome::Incomplete(first) = repair.advance(Limits::default()).unwrap() else {
            panic!("preferred feasible proposal must arrive before the cheaper sibling");
        };
        assert_eq!(first.incumbent.unwrap().cost(), 10);
        assert!(first.lower_bound <= 1);
        assert!(matches!(
            repair.advance(Limits::default()).unwrap(),
            engine::Outcome::Optimal(w) if w.cost() == 1
        ));

        // An unavailable proposal creates no preference; bounds still choose
        // the cheap branch. Unrestricted exact BestFirst keeps that behavior.
        let mut absent =
            Repair::new(m.clone(), options.clone(), m.domains(), vec![2, 0], None).unwrap();
        assert!(matches!(
            absent.advance(Limits::default()).unwrap(),
            engine::Outcome::Optimal(w) if w.cost() == 1
        ));
        let mut exact = engine::Search::new(m, options).unwrap();
        assert!(matches!(
            exact.advance_to_witness(Limits::default()).unwrap(),
            engine::Outcome::Optimal(w) if w.cost() == 1
        ));
    }

    #[test]
    fn interval_memory_peak_excludes_previous_storage_reclaimed_by_eviction() {
        let mut b = ModelBuilder::new();
        let gate = b.variable("known or unresolved", Domain::boolean());
        let choice = b.variable("known choice", Domain::boolean());
        b.guarded_cost(
            vec![Literal::new(gate, 0)],
            Cost::Table {
                variables: vec![choice],
                entries: vec![(vec![0], 10), (vec![1], 1)],
            },
        );
        b.unresolved(vec![Literal::new(gate, 1)], "competitive unresolved region");
        let model = Arc::new(b.build().unwrap());
        let mut repair = Repair::new(
            model.clone(),
            Options::default(),
            model.domains(),
            vec![0, 0],
            None,
        )
        .unwrap();
        let mut blocked = false;
        for _ in 0..10 {
            let engine::Outcome::Incomplete(progress) = repair.advance(Limits::default()).unwrap()
            else {
                panic!("competitive unresolved region must remain open");
            };
            if matches!(progress.reason, StopReason::Coverage(_)) {
                assert_eq!(progress.incumbent.unwrap().cost(), 1);
                blocked = true;
                break;
            }
        }
        assert!(blocked);
        let engine::Outcome::Incomplete(pressure) = repair
            .advance(Limits {
                memory_bytes: Some(1),
                ..Default::default()
            })
            .unwrap()
        else {
            panic!("storage pressure cannot resolve coverage");
        };
        assert_eq!(pressure.reason, StopReason::Memory);
        assert!(repair.stats().evictions > 0);
        let lifetime_peak = repair.stats().peak_retained_bytes;
        let compacted_bytes = repair.stats().retained_bytes;
        assert!(compacted_bytes < lifetime_peak);

        // A later outer caller can have more storage of its own. It must add
        // this interval's peak, not the discarded search tree's lifetime peak.
        repair
            .advance(Limits {
                work: 0,
                ..Default::default()
            })
            .unwrap();
        assert_eq!(repair.advance_peak_retained_bytes(), compacted_bytes);
        assert_eq!(repair.stats().peak_retained_bytes, lifetime_peak);
    }
}

#[cfg(test)]
mod restricted_oracle {
    use super::*;
    use crate::model::{Constraint, Cost, Literal, ModelBuilder};
    #[test]
    fn restricted_context_bounds_and_terminal_claims_match_exhaustive_assignments() {
        for seed in 0..12u64 {
            let mut b = ModelBuilder::new();
            let v: Vec<_> = (0..3)
                .map(|i| b.variable(format!("v{i}"), Domain::interval(0, 2).unwrap()))
                .collect();
            b.guarded_constraint(
                vec![Literal::new(v[2], 1)],
                Constraint::NotEqual {
                    left: v[0],
                    right: v[1],
                },
            );
            for i in 0..2 {
                b.cost(Cost::Table {
                    variables: vec![v[i], v[i + 1]],
                    entries: (0..3)
                        .flat_map(|x| {
                            (0..3).map(move |y| {
                                (
                                    vec![x, y],
                                    (seed + x as u64 * 7 + y as u64 * 11 + i as u64 * 17) % 19,
                                )
                            })
                        })
                        .collect(),
                });
            }
            let m = Arc::new(b.build().unwrap());
            for mut restriction in 0..64 {
                let domains: Vec<_> = (0..3)
                    .map(|_| {
                        let p = restriction % 4;
                        restriction /= 4;
                        if p == 3 {
                            Domain::interval(0, 2).unwrap()
                        } else {
                            Domain::singleton(p)
                        }
                    })
                    .collect();
                let expected = (0..27)
                    .filter_map(|mut a| {
                        let x: Vec<_> = (0..3)
                            .map(|_| {
                                let v = a % 3;
                                a /= 3;
                                v
                            })
                            .collect();
                        if x.iter().zip(&domains).any(|(v, d)| !d.contains(*v))
                            || (x[2] == 1 && x[0] == x[1])
                        {
                            return None;
                        }
                        Some(
                            (0..2)
                                .map(|i| {
                                    (seed + x[i] as u64 * 7 + x[i + 1] as u64 * 11 + i as u64 * 17)
                                        % 19
                                })
                                .sum::<u64>(),
                        )
                    })
                    .min();
                let mut repair = Repair::new(
                    m.clone(),
                    Options::default(),
                    domains.clone(),
                    vec![2, 2, 2],
                    None,
                )
                .unwrap();
                let mut complete = false;
                for _ in 0..1000 {
                    match repair
                        .advance(Limits {
                            work: 7,
                            ..Default::default()
                        })
                        .unwrap()
                    {
                        engine::Outcome::Optimal(w) => {
                            assert_eq!(Some(w.cost()), expected);
                            assert!(w.values().iter().zip(&domains).all(|(v, d)| d.contains(*v)));
                            complete = true;
                            break;
                        }
                        engine::Outcome::Infeasible => {
                            assert_eq!(expected, None);
                            complete = true;
                            break;
                        }
                        engine::Outcome::Incomplete(p) => {
                            if let Some(opt) = expected {
                                assert!(p.lower_bound <= opt);
                                if let Some(w) = p.incumbent {
                                    assert!(w.cost() >= opt);
                                    assert!(w
                                        .values()
                                        .iter()
                                        .zip(&domains)
                                        .all(|(v, d)| d.contains(*v)));
                                }
                            }
                        }
                    }
                }
                assert!(complete);
            }
        }
    }
}

#[cfg(test)]
mod reuse_tests {
    use super::*;
    use crate::model::{Constraint, Cost, ModelBuilder};
    use crate::search::knowledge::Knowledge;
    #[test]
    fn repair_reuses_completed_work_without_leaking_fixed_assumptions() {
        let mut b = ModelBuilder::new();
        let x = b.variable("x", Domain::boolean());
        let y = b.variable("y", Domain::boolean());
        b.constraint(Constraint::Equal { left: x, right: y });
        b.cost(Cost::Table {
            variables: vec![x],
            entries: vec![(vec![0], 4), (vec![1], 1)],
        });
        let model = Arc::new(b.build().unwrap());
        let knowledge = Knowledge::shared(model.clone());
        for (fixed, cost) in [(0, 4), (0, 4), (1, 1)] {
            let domains = vec![Domain::singleton(fixed), Domain::boolean()];
            let mut r = Repair::with_knowledge(
                model.clone(),
                Options::default(),
                domains,
                vec![fixed, 0],
                None,
                knowledge.clone(),
            )
            .unwrap();
            let result = r.advance(Limits::default()).unwrap();
            match result {
                engine::Outcome::Optimal(w) => assert_eq!(w.cost(), cost),
                engine::Outcome::Incomplete(p) => assert_eq!(p.incumbent.unwrap().cost(), cost),
                _ => panic!(),
            }
        }
        let mut r = Repair::with_knowledge(
            model.clone(),
            Options::default(),
            vec![Domain::singleton(0), Domain::boolean()],
            vec![0, 0],
            None,
            knowledge,
        )
        .unwrap();
        let _ = r.advance(Limits::default()).unwrap();
        assert!(r.stats().structural_hits > 0);
    }
}

#[cfg(test)]
mod conflict_reuse_tests {
    use super::*;
    use crate::model::{Constraint, ModelBuilder};
    use crate::search::knowledge::Knowledge;
    #[test]
    fn proved_conflicts_apply_to_narrower_repairs_but_not_wider_ones() {
        let mut b = ModelBuilder::new();
        let x = b.variable("x", Domain::interval(0, 3).unwrap());
        b.constraint(Constraint::InDomain {
            variable: x,
            domain: Domain::singleton(3),
        });
        let model = Arc::new(b.build().unwrap());
        let shared = Knowledge::shared(model.clone());
        let mut first = Repair::with_knowledge(
            model.clone(),
            Options::default(),
            vec![Domain::interval(0, 2).unwrap()],
            vec![0],
            None,
            shared.clone(),
        )
        .unwrap();
        assert!(matches!(
            first.advance(Limits::default()).unwrap(),
            engine::Outcome::Infeasible
        ));
        assert!(first.stats().conflicts_learned > 0);
        let mut narrower = Repair::with_knowledge(
            model.clone(),
            Options::default(),
            vec![Domain::singleton(1)],
            vec![1],
            None,
            shared.clone(),
        )
        .unwrap();
        assert!(matches!(
            narrower.advance(Limits::default()).unwrap(),
            engine::Outcome::Infeasible
        ));
        assert!(narrower.stats().conflict_hits > 0);
        let mut wider = Repair::with_knowledge(
            model.clone(),
            Options::default(),
            model.domains(),
            vec![0],
            None,
            shared,
        )
        .unwrap();
        match wider.advance(Limits::default()).unwrap() {
            engine::Outcome::Optimal(w) => assert_eq!(w.values(), &[3]),
            engine::Outcome::Incomplete(p) => assert_eq!(p.incumbent.unwrap().values(), &[3]),
            _ => panic!("restricted conflict escaped its premise"),
        }
    }
}

#[cfg(test)]
mod shared_oracle {
    use super::*;
    use crate::{
        model::{Constraint, Cost, Literal, ModelBuilder},
        search::knowledge::Knowledge,
    };
    #[test]
    fn shared_repair_proofs_match_exhaustive_results_across_boundary_contexts() {
        let mut b = ModelBuilder::new();
        let x = b.variable("x", Domain::interval(0, 2).unwrap());
        let y = b.variable("y", Domain::interval(0, 2).unwrap());
        let z = b.variable("z", Domain::interval(0, 2).unwrap());
        b.constraint(Constraint::NotEqual { left: x, right: y });
        b.constraint(Constraint::Implies {
            premise: Literal::new(z, 2),
            consequence: Literal::new(x, 1),
        });
        let mut entries = vec![];
        for a in 0..3 {
            for c in 0..3 {
                for d in 0..3 {
                    entries.push((vec![a, c, d], (a + 2 * c + 3 * d) as u64));
                }
            }
        }
        b.cost(Cost::Table {
            variables: vec![x, y, z],
            entries,
        });
        let model = Arc::new(b.build().unwrap());
        let shared = Knowledge::shared(model.clone());
        let choices = [
            Domain::singleton(0),
            Domain::singleton(1),
            Domain::singleton(2),
            Domain::interval(0, 1).unwrap(),
            Domain::interval(0, 2).unwrap(),
        ];
        for _ in 0..2 {
            for a in &choices {
                for c in &choices {
                    for d in &choices {
                        let domains = vec![a.clone(), c.clone(), d.clone()];
                        let mut expected = None;
                        for x in 0..3 {
                            for y in 0..3 {
                                for z in 0..3 {
                                    if a.contains(x)
                                        && c.contains(y)
                                        && d.contains(z)
                                        && x != y
                                        && (z != 2 || x == 1)
                                    {
                                        let cost = (x + 2 * y + 3 * z) as u64;
                                        expected =
                                            Some(expected.map_or(cost, |v: u64| v.min(cost)));
                                    }
                                }
                            }
                        }
                        let mut repair = Repair::with_knowledge(
                            model.clone(),
                            Options::default(),
                            domains,
                            vec![0, 0, 0],
                            None,
                            shared.clone(),
                        )
                        .unwrap();
                        let mut completed = false;
                        for _ in 0..30 {
                            match repair.advance(Limits::default()).unwrap() {
                                engine::Outcome::Optimal(w) => {
                                    assert_eq!(Some(w.cost()), expected);
                                    completed = true;
                                    break;
                                }
                                engine::Outcome::Infeasible => {
                                    assert_eq!(expected, None);
                                    completed = true;
                                    break;
                                }
                                engine::Outcome::Incomplete(p) => {
                                    if let Some(optimum) = expected {
                                        assert!(p.lower_bound <= optimum);
                                    }
                                    if let Some(w) = p.incumbent {
                                        assert!(expected.is_some_and(|optimum| w.cost() >= optimum));
                                    }
                                }
                            }
                        }
                        assert!(completed, "finite tiny context must finish");
                    }
                }
            }
        }
    }
}
