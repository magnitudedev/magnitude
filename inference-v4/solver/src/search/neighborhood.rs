//! Complete-candidate search. Restricted repair proofs never escape their scope.
use super::{
    engine, repair::Repair, structure::Structure, Limits, NeighborhoodOptions, Options, Stats,
};
use crate::{
    result::{Progress, StopReason},
    Domain, Error, FeasibleSolution, Model, Outcome, Result, Solution,
};
use std::{sync::Arc, time::Instant};

struct Random(u64);
impl Random {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut x = self.0;
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
        x ^ (x >> 31)
    }
    fn index(&mut self, n: usize) -> usize {
        (self.next() as u128 % n as u128) as usize
    }
    fn value(&mut self, d: &Domain) -> i64 {
        d.nth(self.next() as u128 % d.cardinality())
            .expect("nonempty domain")
    }
    fn unit(&mut self) -> f64 {
        ((self.next() >> 11) as f64 + 0.5) / ((1_u64 << 53) as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Constraint, Cost, ModelBuilder};

    fn seeded(config: NeighborhoodOptions, disconnected: bool) -> Search {
        let mut b = ModelBuilder::new();
        let a = b.variable("tile", Domain::boolean());
        let c = b.variable("staging", Domain::boolean());
        // An external choice ensures a repair cannot prove global optimality.
        let outside = b.variable("outside", Domain::boolean());
        b.cost(Cost::Table {
            variables: vec![a, c],
            entries: vec![
                (vec![0, 0], 10),
                (vec![0, 1], 13),
                (vec![1, 0], 14),
                (vec![1, 1], 6),
            ],
        });
        b.cost(Cost::Table {
            variables: vec![outside],
            entries: vec![(vec![0], 2), (vec![1], 1)],
        });
        if disconnected {
            b.constraint(Constraint::Equal { left: a, right: c });
        }
        let model = Arc::new(b.build().unwrap());
        let mut s = Search::new(model.clone(), Options::default(), config).unwrap();
        let w = FeasibleSolution {
            model,
            values: vec![0, 0, 0],
            cost: 12,
        };
        s.consider(w, None, 0.0).unwrap();
        s
    }

    #[test]
    fn joint_repairs_cross_barriers_that_coordinate_greedy_cannot() {
        for disconnected in [false, true] {
            let options = NeighborhoodOptions {
                seed: 2,
                exploration: false,
                max_neighborhood_variables: 1,
                restart_after: u64::MAX,
                ..Default::default()
            };
            let mut scalar = seeded(options.clone(), disconnected);
            let result = scalar
                .advance(Limits {
                    work: 15_000,
                    ..Default::default()
                })
                .unwrap();
            let Outcome::Incomplete(p) = result else {
                panic!("coordinate fixed point is not a global optimum");
            };
            assert_eq!(p.incumbent.unwrap().cost(), 11);
            assert!(p.lower_bound <= 7);
            let mut joint = seeded(
                NeighborhoodOptions {
                    max_neighborhood_variables: 2,
                    ..options
                },
                disconnected,
            );
            let result = joint
                .advance(Limits {
                    work: 15_000,
                    ..Default::default()
                })
                .unwrap();
            let cost = match result {
                Outcome::Optimal(w) => w.cost(),
                Outcome::Incomplete(p) => p.incumbent.unwrap().cost(),
                _ => panic!(),
            };
            assert_eq!(cost, 7);
            assert!(joint.stats.successful_repairs > 0);
        }
    }
    #[test]
    fn split_work_has_identical_candidate_and_rng_trajectory() {
        let options = NeighborhoodOptions {
            seed: 89,
            repair_work: 93,
            restart_after: 3,
            ..Default::default()
        };
        let mut split = seeded(options.clone(), true);
        let mut whole = seeded(options, true);
        split
            .advance(Limits {
                work: 1337,
                ..Default::default()
            })
            .unwrap();
        split
            .advance(Limits {
                work: 2663,
                ..Default::default()
            })
            .unwrap();
        whole
            .advance(Limits {
                work: 4000,
                ..Default::default()
            })
            .unwrap();
        assert_eq!(
            split.best.as_ref().unwrap().values(),
            whole.best.as_ref().unwrap().values()
        );
        assert_eq!(split.rng.0, whole.rng.0);
        assert_eq!(split.stats.work, whole.stats.work);
        assert_eq!(
            split.stats.neighborhood_attempts,
            whole.stats.neighborhood_attempts
        );
    }
    #[test]
    fn zero_work_and_memory_interruptions_preserve_pending_repair() {
        let mut search = seeded(NeighborhoodOptions::default(), false);
        search
            .advance(Limits {
                work: 1,
                ..Default::default()
            })
            .unwrap();
        let random = search.rng.0;
        let work = search.stats.work;
        search
            .advance(Limits {
                work: 0,
                ..Default::default()
            })
            .unwrap();
        assert_eq!(random, search.rng.0);
        assert_eq!(work, search.stats.work);
        let Outcome::Incomplete(p) = search
            .advance(Limits {
                memory_bytes: Some(1),
                ..Default::default()
            })
            .unwrap()
        else {
            panic!()
        };
        assert_eq!(p.reason, StopReason::Memory);
        assert!(search.active.is_some());
        assert_eq!(work, search.stats.work);
    }
    #[test]
    fn shared_proof_cache_is_reclaimed_before_outer_memory_stop() {
        let mut s = seeded(NeighborhoodOptions::default(), false);
        let empty = super::super::knowledge::retained_bytes(&s.knowledge);
        let mut repair = Repair::with_knowledge(
            s.model.clone(),
            s.options.clone(),
            s.model.domains(),
            vec![0; s.model.variables().len()],
            None,
            s.knowledge.clone(),
        )
        .unwrap();
        for _ in 0..30 {
            if matches!(
                repair.advance(Limits::default()).unwrap(),
                engine::Outcome::Optimal(_)
            ) {
                break;
            }
        }
        drop(repair);
        assert!(super::super::knowledge::retained_bytes(&s.knowledge) > empty);
        s.memory();
        let cap = s.stats.retained_bytes - 1;
        let _ = s
            .advance(Limits {
                work: 1,
                memory_bytes: Some(cap),
                ..Default::default()
            })
            .unwrap();
        assert!(super::super::knowledge::retained_bytes(&s.knowledge) <= empty);
        assert_eq!(
            s.best.as_ref().unwrap().cost(),
            12,
            "eviction preserves the incumbent"
        );
    }

    #[test]
    fn repeated_inner_memory_stops_do_not_consume_the_local_repair_budget() {
        let mut s = seeded(
            NeighborhoodOptions {
                seed: 5,
                repair_work: 128,
                exploration: false,
                max_neighborhood_variables: 2,
                ..Default::default()
            },
            false,
        );
        // Install a genuine multi-variable repair directly. This regression
        // tests memory suspension, independently of heuristic move selection.
        let base = s.best.clone();
        let repair = Repair::with_knowledge(
            s.model.clone(),
            s.options.clone(),
            s.model.domains(),
            base.as_ref().unwrap().values().to_vec(),
            base.as_ref(),
            s.knowledge.clone(),
        )
        .unwrap();
        s.active = Some(Active {
            repair,
            base,
            remaining: 128,
            initialization: false,
        });
        s.memory();
        let cap = s.stats.retained_bytes + 1;
        let Outcome::Incomplete(p) = s
            .advance(Limits {
                work: 100,
                memory_bytes: Some(cap),
                ..Default::default()
            })
            .unwrap()
        else {
            panic!()
        };
        assert_eq!(p.reason, StopReason::Memory);
        let remaining = s.active.as_ref().unwrap().remaining;
        let work = s.stats.repair_work;
        for _ in 0..128 {
            let Outcome::Incomplete(p) = s
                .advance(Limits {
                    work: 100,
                    memory_bytes: Some(cap),
                    ..Default::default()
                })
                .unwrap()
            else {
                panic!()
            };
            assert_eq!(p.reason, StopReason::Memory);
            assert_eq!(s.stats.repair_work, work);
            assert_eq!(s.active.as_ref().unwrap().remaining, remaining);
        }
    }

    #[test]
    fn requested_width_and_dependency_release_are_reported_separately() {
        let mut b = ModelBuilder::new();
        let x = b.variable("choice", Domain::boolean());
        let one = b.variable("one", Domain::singleton(1));
        let y = b.variable("derived", Domain::boolean());
        b.constraint(Constraint::Arithmetic(crate::model::Arithmetic::Product {
            left: x,
            right: one,
            product: y,
        }));
        b.cost(Cost::Constant(5));
        let mut search = Search::new(
            Arc::new(b.build().unwrap()),
            Options::default(),
            NeighborhoodOptions {
                max_neighborhood_variables: 1,
                exploration: false,
                ..Default::default()
            },
        )
        .unwrap();
        // Seed validation and local-repair construction, without executing it.
        search
            .advance(Limits {
                work: 2,
                ..Default::default()
            })
            .unwrap();
        assert_eq!(search.stats.requested_variables, 1);
        assert_eq!(search.stats.max_requested_variables, 1);
        assert_eq!(search.stats.released_variables, 2);
        assert_eq!(search.stats.max_released_variables, 2);
    }
}
struct Guidance {
    values: Vec<i64>,
    priorities: Vec<u64>,
}

struct Active {
    repair: Repair,
    base: Option<FeasibleSolution>,
    remaining: u64,
    initialization: bool,
}
pub(super) struct Search {
    model: Arc<Model>,
    options: Options,
    config: NeighborhoodOptions,
    structure: Structure,
    guidance: Option<Guidance>,
    knowledge: super::knowledge::SharedKnowledge,
    rng: Random,
    active: Option<Active>,
    population: Vec<FeasibleSolution>,
    best: Option<FeasibleSolution>,
    lower: u64,
    stale: u64,
    stats: Stats,
    terminal: Option<Outcome>,
    failure: Option<Error>,
}
impl Search {
    pub fn new(model: Arc<Model>, options: Options, config: NeighborhoodOptions) -> Result<Self> {
        model.validate()?;
        if config.max_neighborhood_variables == 0
            || config.repair_work == 0
            || config.population_size == 0
            || config.restart_after == 0
        {
            return Err(Error::InvalidModel(
                "neighborhood sizes and work limits must be positive".into(),
            ));
        }
        let structure = Structure::new(&model);
        let rng = Random(config.seed);
        let knowledge = super::knowledge::Knowledge::shared(model.clone());
        let mut s = Self {
            model,
            options,
            config,
            structure,
            guidance: None,
            knowledge,
            rng,
            active: None,
            population: vec![],
            best: None,
            lower: 0,
            stale: 0,
            stats: Stats::default(),
            terminal: None,
            failure: None,
        };
        s.memory();
        Ok(s)
    }
    pub fn model(&self) -> &Arc<Model> {
        &self.model
    }
    pub fn stats(&self) -> &Stats {
        &self.stats
    }

    fn prepare(&mut self, elapsed: f64) -> Result<()> {
        let preparation = Instant::now();
        let initial = self.best.is_none();
        let restart = !initial && self.stale >= self.config.restart_after;
        let mut domains = self.model.domains();
        let mut preferred: Vec<_> = domains.iter().map(|d| d.min().unwrap_or(0)).collect();
        let mut base = None;
        if initial && domains.iter().all(|d| !d.is_empty()) {
            // A proposal, not a proof: the same original-model validator
            // decides whether this cheap complete assignment is executable.
            let checked = self.model.validate_assignment(&preferred)?;
            self.stats.assessments += 1;
            if !checked.infeasible {
                if let Some(cost) = checked.exact_cost {
                    let candidate = FeasibleSolution {
                        model: self.model.clone(),
                        values: preferred,
                        cost,
                    };
                    self.consider(
                        candidate,
                        None,
                        elapsed + preparation.elapsed().as_secs_f64(),
                    )?;
                    return Ok(());
                }
            }
        }
        if initial || restart {
            if restart {
                self.stats.restarts += 1;
                self.stale = 0;
                for (p, d) in preferred.iter_mut().zip(&domains) {
                    if !d.is_empty() {
                        *p = self.rng.value(d);
                    }
                }
            }
        } else {
            let candidate = if self.config.exploration {
                self.population[self.rng.index(self.population.len())].clone()
            } else {
                // Greedy restarts improve their own current candidate; the
                // best-ever witness is retained independently.
                self.population[0].clone()
            };
            preferred.copy_from_slice(candidate.values());
            let n = domains.len();
            let mut selected = vec![false; n];
            let mut chosen = Vec::new();
            let roots = &self.structure.roots;
            if !roots.is_empty() {
                // Alternate guided and unweighted moves: incumbent bottlenecks
                // deserve attention without starving different strategies.
                let guided = self.rng.index(2) == 0;
                if guided
                    && self
                        .guidance
                        .as_ref()
                        .is_none_or(|g| g.values != candidate.values())
                {
                    self.guidance = Some(Guidance {
                        values: candidate.values().to_vec(),
                        priorities: self.structure.priorities(&self.model, candidate.values()),
                    });
                }
                let priorities = guided.then(|| &self.guidance.as_ref().unwrap().priorities);
                let mut first = roots[self.rng.index(roots.len())];
                if let Some(scores) = &priorities {
                    for _ in 0..3 {
                        let trial = roots[self.rng.index(roots.len())];
                        if scores[trial] > scores[first] {
                            first = trial;
                        }
                    }
                }
                selected[first] = true;
                chosen.push(first);
                let wanted = 1 + self
                    .rng
                    .index(self.config.max_neighborhood_variables.min(roots.len()));
                self.stats.requested_variables += wanted as u64;
                self.stats.max_requested_variables = self.stats.max_requested_variables.max(wanted);
                // Pick a factor incident on an already selected root, then a
                // root in its scope. Global factors are sampled, not expanded.
                for _ in 1..wanted {
                    let from = chosen[self.rng.index(chosen.len())];
                    let incident = &self.structure.incident[from];
                    let mut next = roots[self.rng.index(roots.len())];
                    if !incident.is_empty() {
                        let scope =
                            &self.structure.scopes[incident[self.rng.index(incident.len())]];
                        let neighbors: Vec<_> = scope
                            .iter()
                            .copied()
                            .filter(|v| !selected[*v] && roots.binary_search(v).is_ok())
                            .collect();
                        if !neighbors.is_empty() {
                            next = neighbors[self.rng.index(neighbors.len())];
                            if let Some(scores) = &priorities {
                                for _ in 0..3 {
                                    let trial = neighbors[self.rng.index(neighbors.len())];
                                    if scores[trial] > scores[next] {
                                        next = trial;
                                    }
                                }
                            }
                        }
                    }
                    if !selected[next] {
                        selected[next] = true;
                        chosen.push(next);
                    }
                }
                self.structure.close(&mut selected, &mut chosen);
                let proposals = self
                    .structure
                    .proposals(&self.model, candidate.values(), first);
                if !proposals.is_empty() && self.rng.index(2) == 0 {
                    preferred[first] = proposals[self.rng.index(proposals.len())];
                }
                if self.config.exploration && self.rng.index(3) == 0 {
                    // Force a change so repair can return an uphill candidate;
                    // a repair seeded with the old optimum cannot do that.
                    let alternative = domains[first].without(candidate.values()[first]);
                    if !alternative.is_empty() {
                        let proposal = match self.rng.index(4) {
                            0 => alternative.min().unwrap(),
                            1 => alternative.max().unwrap(),
                            2 => alternative
                                .restrict(
                                    preferred[first].saturating_sub(1),
                                    preferred[first].saturating_add(1),
                                )?
                                .min()
                                .unwrap_or_else(|| self.rng.value(&alternative)),
                            _ => self.rng.value(&alternative),
                        };
                        domains[first] = Domain::singleton(proposal);
                        preferred[first] = proposal;
                    }
                }
            }
            for (i, d) in domains.iter_mut().enumerate() {
                if !selected[i] {
                    *d = Domain::singleton(candidate.values()[i]);
                }
            }
            self.stats.released_variables += chosen.len() as u64;
            self.stats.max_released_variables = self.stats.max_released_variables.max(chosen.len());
            base = Some(candidate);
        }
        self.stats.neighborhood_attempts += u64::from(!initial);
        let repair = Repair::with_knowledge(
            self.model.clone(),
            self.options.clone(),
            domains,
            preferred,
            base.as_ref(),
            self.knowledge.clone(),
        )?;
        self.aggregate(&Stats::default(), repair.stats());
        self.active = Some(Active {
            repair,
            base,
            remaining: self.config.repair_work,
            initialization: initial,
        });
        Ok(())
    }

    fn consider(
        &mut self,
        candidate: FeasibleSolution,
        base: Option<&FeasibleSolution>,
        elapsed: f64,
    ) -> Result<()> {
        let validation = Instant::now();
        let assessment = self.model.validate_assignment(candidate.values())?;
        if assessment.infeasible || assessment.exact_cost != Some(candidate.cost()) {
            return Err(Error::Invariant(
                "neighborhood witness failed original model validation".into(),
            ));
        }
        self.stats.assessments += 1;
        let elapsed = elapsed + validation.elapsed().as_secs_f64();
        let improved = self
            .best
            .as_ref()
            .is_none_or(|b| candidate.cost() < b.cost());
        if improved {
            self.best = Some(candidate.clone());
            self.stale = 0;
            self.stats.improving_moves += 1;
            self.stats
                .first_feasible_seconds
                .get_or_insert(self.stats.solve_seconds + elapsed);
            self.stats.winner_found_seconds = Some(self.stats.solve_seconds + elapsed);
        } else {
            self.stale += 1;
        }
        let accept = base.is_none_or(|b| candidate.cost() <= b.cost())
            || (self.config.exploration && {
                let b = base.unwrap();
                let temperature = (b.cost().max(1) as f64)
                    * (0.02
                        + 0.18
                            * (self.stale.min(self.config.restart_after) as f64
                                / self.config.restart_after as f64));
                self.rng.unit() < (-((candidate.cost() - b.cost()) as f64) / temperature).exp()
            });
        if accept || improved {
            self.stats.accepted_moves += 1;
            if !self.config.exploration {
                if self.population.is_empty() {
                    self.population.push(candidate);
                } else {
                    self.population[0] = candidate;
                }
                return Ok(());
            }
            if !self
                .population
                .iter()
                .any(|p| p.values() == candidate.values())
            {
                if self.population.len() < self.config.population_size {
                    self.population.push(candidate);
                } else {
                    // Replace the closest candidate, weighting guard choices.
                    let index = self
                        .population
                        .iter()
                        .enumerate()
                        .min_by_key(|(_, p)| {
                            let distance = p
                                .values()
                                .iter()
                                .zip(candidate.values())
                                .filter(|(a, b)| a != b)
                                .count();
                            let modes = self
                                .structure
                                .guards
                                .iter()
                                .filter(|&&v| p.values()[v] != candidate.values()[v])
                                .count();
                            distance + modes * self.model.variables().len()
                        })
                        .map(|(i, _)| i)
                        .unwrap();
                    self.population[index] = candidate;
                }
            }
        }
        Ok(())
    }
    fn memory(&mut self) {
        let witness_bytes = |w: &FeasibleSolution| w.values.capacity() * std::mem::size_of::<i64>();
        let mut bytes = self.structure.bytes()
            + self.guidance.as_ref().map_or(0, |g| {
                g.values.capacity() * std::mem::size_of::<i64>()
                    + g.priorities.capacity() * std::mem::size_of::<u64>()
            })
            + self.population.capacity() * std::mem::size_of::<FeasibleSolution>()
            + self.population.iter().map(witness_bytes).sum::<usize>()
            + self.best.as_ref().map_or(0, witness_bytes);
        if let Some(a) = &self.active {
            bytes += a.repair.stats().retained_bytes + a.base.as_ref().map_or(0, witness_bytes);
        } else {
            bytes += super::knowledge::retained_bytes(&self.knowledge);
        }
        if let Some(Outcome::Optimal(solution)) = &self.terminal {
            bytes += witness_bytes(solution.feasible());
        }
        self.stats.retained_bytes = bytes;
        self.stats.peak_retained_bytes = self.stats.peak_retained_bytes.max(bytes);
    }
    fn aggregate(&mut self, old: &Stats, new: &Stats) {
        macro_rules! sum { ($($f:ident),*) => { $(self.stats.$f += new.$f - old.$f;)* }; }
        sum!(
            nodes,
            branches,
            materializations,
            decompositions,
            memo_hits,
            structural_hits,
            proof_stores,
            conflict_hits,
            conflicts_learned,
            evictions,
            propagations,
            assessments
        );
        self.stats.max_frontier = self.stats.max_frontier.max(new.max_frontier);
        self.stats.max_context_variables = self
            .stats
            .max_context_variables
            .max(new.max_context_variables);
        self.stats.max_context_log10 = self.stats.max_context_log10.max(new.max_context_log10);
        self.stats.pruned = new.pruned;
    }
    pub fn advance(&mut self, limits: Limits) -> Result<Outcome> {
        if let Some(e) = &self.failure {
            return Err(e.clone());
        }
        let result = self.advance_inner(limits);
        if let Err(e) = &result {
            self.failure = Some(e.clone());
        }
        result
    }
    fn advance_inner(&mut self, limits: Limits) -> Result<Outcome> {
        if let Some(terminal) = &self.terminal {
            return Ok(terminal.clone());
        }
        let start = Instant::now();
        let work_start = self.stats.work;
        let reason = loop {
            if self.stats.work - work_start >= limits.work {
                break StopReason::Work;
            }
            if limits.time.is_some_and(|t| start.elapsed() >= t) {
                break StopReason::Time;
            }
            self.memory();
            if limits
                .memory_bytes
                .is_some_and(|cap| self.stats.retained_bytes >= cap)
            {
                // Reusable proofs own no live search coverage. Reclaim them
                // before a cache alone can prevent resuming the caller's work.
                self.knowledge
                    .lock()
                    .expect("solver knowledge lock poisoned")
                    .clear();
                if let Some(active) = &mut self.active {
                    active.repair.update_memory();
                }
                self.memory();
                if limits
                    .memory_bytes
                    .is_some_and(|cap| self.stats.retained_bytes >= cap)
                {
                    break StopReason::Memory;
                }
            }
            if self.active.is_none() {
                // Conservative admission before cloning domains/factors into
                // a repair. The exact engine then accounts actual retained
                // state and checks each subsequent expansion.
                let setup = self
                    .model
                    .retained_bytes()
                    .saturating_mul(6)
                    .saturating_add(self.model.variables().len().saturating_mul(128))
                    .saturating_add(1024);
                if limits
                    .memory_bytes
                    .is_some_and(|cap| self.stats.retained_bytes.saturating_add(setup) > cap)
                {
                    self.knowledge
                        .lock()
                        .expect("solver knowledge lock poisoned")
                        .clear();
                    self.memory();
                    if limits
                        .memory_bytes
                        .is_some_and(|cap| self.stats.retained_bytes.saturating_add(setup) > cap)
                    {
                        break StopReason::Memory;
                    }
                }
                self.prepare(start.elapsed().as_secs_f64())?;
                self.stats.work += 1;
                continue;
            }
            let mut active = self.active.take().unwrap();
            let old = active.repair.stats().clone();
            // Yield on witness events, not arbitrary polling chunks. This keeps
            // RNG transitions independent of callers' work-budget partitioning
            // without rescanning retained memory after every factor operation.
            let remaining = limits.work - (self.stats.work - work_start);
            let outcome = active.repair.advance(Limits {
                work: if active.initialization {
                    remaining
                } else {
                    remaining.min(active.remaining)
                },
                time: limits.time.map(|t| t.saturating_sub(start.elapsed())),
                memory_bytes: limits.memory_bytes.map(|cap| {
                    cap.saturating_sub(self.stats.retained_bytes.saturating_sub(old.retained_bytes))
                }),
            })?;
            let new = active.repair.stats().clone();
            self.stats.peak_retained_bytes = self.stats.peak_retained_bytes.max(
                self.stats
                    .retained_bytes
                    .saturating_sub(old.retained_bytes)
                    .saturating_add(active.repair.advance_peak_retained_bytes()),
            );
            self.aggregate(&old, &new);
            let work = new.work - old.work;
            let suspended = matches!(&outcome, engine::Outcome::Incomplete(p)
                if matches!(p.reason, StopReason::Memory | StopReason::Time));
            self.stats.work += if suspended { work } else { work.max(1) };
            self.stats.repair_work += work;
            active.remaining = active.remaining.saturating_sub(work);
            let (witness, complete, stop) = match outcome {
                engine::Outcome::Optimal(w) => {
                    if active.repair.unrestricted {
                        self.terminal = Some(Outcome::Optimal(Solution {
                            feasible: w.clone(),
                        }));
                        self.lower = w.cost();
                    }
                    (Some(w), true, None)
                }
                engine::Outcome::Infeasible => {
                    if active.repair.unrestricted {
                        self.terminal = Some(Outcome::Infeasible);
                    }
                    (None, true, None)
                }
                engine::Outcome::Incomplete(p) => {
                    if active.repair.unrestricted {
                        self.lower = self.lower.max(p.lower_bound);
                    }
                    let stop = match p.reason {
                        StopReason::Work => None,
                        reason => Some(reason),
                    };
                    (p.incumbent, false, stop)
                }
            };
            let done = complete
                || (active.initialization && witness.is_some())
                || (!active.initialization && active.remaining == 0 && !suspended);
            if done {
                if !complete && !active.initialization {
                    self.stats.exhausted_repairs += 1;
                }
                if let Some(w) = witness {
                    self.stats.successful_repairs += 1;
                    self.consider(w, active.base.as_ref(), start.elapsed().as_secs_f64())?;
                } else {
                    self.stats.failed_repairs += 1;
                    self.stale += 1;
                }
            } else {
                // Keep an improvement visible even while the repair continues.
                if let Some(w) = witness {
                    if self.best.as_ref().is_none_or(|b| w.cost() < b.cost()) {
                        self.consider(w, active.base.as_ref(), start.elapsed().as_secs_f64())?;
                    }
                }
                self.active = Some(active);
            }
            if let Some(outcome) = self.terminal.clone() {
                self.stats.solve_seconds += start.elapsed().as_secs_f64();
                self.memory();
                return Ok(outcome);
            }
            if self.best.as_ref().is_some_and(|b| b.cost() == self.lower) {
                let outcome = Outcome::Optimal(Solution {
                    feasible: self.best.as_ref().unwrap().clone(),
                });
                self.terminal = Some(outcome.clone());
                self.stats.solve_seconds += start.elapsed().as_secs_f64();
                self.memory();
                return Ok(outcome);
            }
            if let Some(reason) = stop {
                // Coverage blocks the affected repair; restart it next time if
                // a feasible base exists, retaining the obligation in this result.
                if matches!(reason, StopReason::Coverage(_)) && self.best.is_some() {
                    self.active = None;
                }
                break reason;
            }
        };
        self.stats.solve_seconds += start.elapsed().as_secs_f64();
        self.memory();
        Ok(Outcome::Incomplete(Progress {
            lower_bound: self.lower,
            incumbent: self.best.clone(),
            reason,
            stats: self.stats.clone(),
        }))
    }
}
