//! Seeded joint selection: export once, validate the seed, then improve it under the
//! budget by the budget's strategy (exact search then neighborhood improvement, or greedy
//! coordinate sweeps), reconstruct exactly the chosen witness.
use super::export::{solver_error, Export};
use super::{greedy, Backend, Interval, Phase, ProofStatus, Qualification, SearchStats, Selected, SelectionError, Timings};
use magnitude_solver::{Algorithm, FeasibleSolution, Limits, Model, NeighborhoodOptions, Options, Outcome, Search};
use seismic_lang::family::{self, Family, SiteId, Witness, Workload};
use seismic_lang::instantiate::instantiate;
use seismic_lang::precision::NumericalAssessment;
use seismic_lang::sir::Program;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// How the validated seed is improved. Both rank by the exported objective only.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Strategy {
    /// Exact search for half the budget, then neighborhood search; best checked witness.
    #[default]
    Exact,
    /// Diagnostic: no solver search, deterministic coordinate sweeps from the seed
    /// (`greedy.rs`). Never claims more than `ProofStatus::Feasible`; ignores `work`/`time`.
    Greedy,
}

#[derive(Clone, Copy, Debug)]
pub struct Budget {
    pub work: u64,
    pub time: Option<Duration>,
    pub strategy: Strategy,
}

impl Default for Budget {
    fn default() -> Budget {
        Budget { work: 200_000, time: Some(Duration::from_secs(2)), strategy: Strategy::Exact }
    }
}

/// Construct the family, validate the backend seed, improve under `budget`, instantiate,
/// verify reconstruction, realize.
pub fn select<B: Backend>(program: &Program, entry: &str, workload: &Workload, backend: &B, budget: Budget) -> Result<Selected<B::Execution>, SelectionError> {
    let started = Instant::now();
    let family = construct(program, entry, workload, backend)?;
    let constructed = started.elapsed();
    let mut chosen = select_family(program, &family, backend, budget, &|w| family.validate(w))?;
    chosen.timings.family = constructed;
    finish(program, family, backend, chosen, None)
}

/// Validate and realize a supplied complete witness without search (qualified-witness
/// replay). The result is `Feasible` with no bound: nothing was compared.
pub fn replay<B: Backend>(program: &Program, entry: &str, workload: &Workload, backend: &B, witness: &Witness) -> Result<Selected<B::Execution>, SelectionError> {
    let started = Instant::now();
    let family = construct(program, entry, workload, backend)?;
    let mut timings = Timings { family: started.elapsed(), ..Timings::default() };
    let (export, ..) = exported(program, &family, backend, &mut timings, None)?;
    let (_, estimate) = audit(&export, &|w| family.validate(w), witness).map_err(|why| SelectionError::Reconstruction(format!("replayed witness rejected: {why}")))?;
    let stats = SearchStats { variables: export.model.variables().len(), factors: export.model.factors().len(), ..SearchStats::default() };
    drop(export);
    let chosen = Chosen { witness: witness.clone(), estimate, seed: witness.clone(), seed_estimate: estimate, lower_bound: 0, status: ProofStatus::Feasible, timings, stats };
    finish(program, family, backend, chosen, None)
}

/// Select the cheapest matching, numerically accepted whole-witness qualification. This does not
/// infer a bound for a new configuration: every candidate result is replayed and audited exactly.
pub fn select_qualified<B: Backend>(program: &Program, entry: &str, workload: &Workload, backend: &B, budget: Budget, qualifications: &[Qualification]) -> Result<Selected<B::Execution>, SelectionError> {
    let identity = program.identity();
    let numerical_environment = backend.numerical_environment();
    let mut accepted: Vec<&Qualification> = qualifications.iter().filter(|qualification| {
        qualification.program == identity
            && qualification.target == backend.target()
            && qualification.numerical_environment == numerical_environment
            && qualification.entry == entry
            && qualification.shapes == workload.shapes
            && qualification.elems == workload.elems
            && qualification.assessment.satisfies(&workload.precision)
    }).collect();
    accepted.sort_by(|left, right| left.corpus.cmp(&right.corpus).then_with(|| left.method.cmp(&right.method)));
    let mut best = Some(select(program, entry, workload, backend, budget)?);
    for qualification in accepted {
        let started = Instant::now();
        let mut family = construct(program, entry, workload, backend)?;
        family.allow_numerical_effects = true;
        let mut timings = Timings { family: started.elapsed(), ..Timings::default() };
        let (export, ..) = exported(program, &family, backend, &mut timings, Some(&qualification.witness))?;
        let (_, estimate) = audit(&export, &|w| family.validate(w), &qualification.witness)
            .map_err(|why| SelectionError::Reconstruction(format!("qualified witness rejected: {why}")))?;
        let stats = SearchStats { variables: export.model.variables().len(), factors: export.model.factors().len(), ..SearchStats::default() };
        drop(export);
        let chosen = Chosen { witness: qualification.witness.clone(), estimate, seed: qualification.witness.clone(), seed_estimate: estimate, lower_bound: 0, status: ProofStatus::Feasible, timings, stats };
        let selected = finish(program, family, backend, chosen, Some(qualification))?;
        if best.as_ref().is_none_or(|current: &Selected<B::Execution>| selected.estimate < current.estimate) {
            best = Some(selected);
        }
    }
    let identity = identity.iter().map(|byte| format!("{byte:02x}")).collect::<String>();
    best.ok_or_else(|| SelectionError::MissingQualification(format!("no accepted record or reference path matches program {identity}, entry `{entry}` and target `{}`", backend.target())))
}

fn construct<B: Backend>(program: &Program, entry: &str, workload: &Workload, backend: &B) -> Result<Family, SelectionError> {
    family::construct(program, entry, backend.target(), workload).map_err(|e| if e.contains("coverage") { SelectionError::MissingCoverage(e) } else { SelectionError::InvalidSource(e) })
}

/// The backend hooks and the solver export, each timed; the domains and intervals the seed reads.
#[allow(clippy::type_complexity)]
fn exported<'f, B: Backend>(program: &Program, family: &'f Family, backend: &B, timings: &mut Timings, qualified: Option<&Witness>) -> Result<(Export<'f>, BTreeMap<SiteId, Vec<i64>>, Vec<Interval>), SelectionError> {
    let started = Instant::now();
    let domains = backend.bind_structure(program, family)?;
    let constraints = backend.constraints(program, family)?;
    let intervals = backend.intervals(program, family)?;
    let factors = backend.factors(program, family, &intervals)?;
    timings.backend_hooks = started.elapsed();
    let started = Instant::now();
    let export = Export::build(family, &domains, &constraints, &intervals, &factors, qualified)?;
    timings.export = started.elapsed();
    Ok((export, domains, intervals))
}

fn finish<B: Backend>(program: &Program, family: Family, backend: &B, chosen: Chosen, qualified: Option<&Qualification>) -> Result<Selected<B::Execution>, SelectionError> {
    let mut timings = chosen.timings;
    let started = Instant::now();
    let lowered = instantiate(program, &family, &chosen.witness).map_err(SelectionError::Reconstruction)?;
    timings.instantiate = started.elapsed();
    let started = Instant::now();
    let execution = backend.realize(lowered, &family, &chosen.witness)?;
    timings.realize = started.elapsed();
    let unresolved = family.obligations.iter().map(|o| format!("occurrence {} definition {}: {}", o.occurrence.0, o.definition.0, o.reason)).collect();
    let numerical_deviations: Vec<String> = family.occurrences.iter().filter_map(|occurrence| {
        let selected = *chosen.witness.choices.get(&occurrence.id)?;
        let candidate = occurrence.candidates.get(selected as usize)?;
        let active_effect = !candidate.reference
            || (family.allow_numerical_effects && !candidate.numerical_effects.is_empty());
        active_effect.then(|| format!("occurrence {} selected definition {} with effects {:?}", occurrence.id.0, candidate.via.0, candidate.numerical_effects))
    }).collect();
    let numerical_assessment = qualified.map(|record| record.assessment.clone()).unwrap_or_else(|| if numerical_deviations.is_empty() {
        NumericalAssessment::exact()
    } else {
        NumericalAssessment::unknown(numerical_deviations.join("; "))
    });
    Ok(Selected {
        execution,
        witness: chosen.witness,
        estimate: chosen.estimate,
        seed: chosen.seed,
        seed_estimate: chosen.seed_estimate,
        status: chosen.status,
        estimate_model: backend.estimate_model(),
        numerical_assessment,
        qualification: qualified.map(Qualification::identity),
        lower_bound: chosen.lower_bound,
        unresolved,
        timings,
        search: chosen.stats,
        family: Arc::new(family),
    })
}

#[derive(Debug)]
struct Chosen {
    witness: Witness,
    estimate: u64,
    seed: Witness,
    seed_estimate: u64,
    lower_bound: u64,
    status: ProofStatus,
    /// `family`, `instantiate` and `realize` are filled by the callers that run them.
    timings: Timings,
    stats: SearchStats,
}

/// Selection over an already constructed family. `check` is the family's structural
/// witness validation.
fn select_family<B: Backend>(program: &Program, family: &Family, backend: &B, budget: Budget, check: &dyn Fn(&Witness) -> Result<(), String>) -> Result<Chosen, SelectionError> {
    let mut timings = Timings::default();
    let (export, domains, intervals) = exported(program, family, backend, &mut timings, None)?;
    let started = Instant::now();
    let seed = backend.seed(program, family, &domains, &intervals)?;
    let seeded = audit(&export, check, &seed);
    timings.seed = started.elapsed();
    let mut stats = SearchStats { strategy: budget.strategy, variables: export.model.variables().len(), factors: export.model.factors().len(), ..SearchStats::default() };

    if budget.strategy == Strategy::Greedy {
        // Without a solver there is no infeasibility proof: a rejected seed is reported as such.
        let (_, seed_estimate) = seeded.map_err(|why| SelectionError::Reconstruction(format!("seed policy is defective: {why}")))?;
        let started = Instant::now();
        let improved = greedy::improve(&export, &|w| audit(&export, check, w).ok().map(|(_, cost)| cost), &seed, seed_estimate);
        timings.search = started.elapsed();
        (stats.greedy_sweeps, stats.greedy_trials) = (improved.sweeps, improved.trials);
        let (_, estimate) = audit(&export, check, &improved.witness).map_err(|why| SelectionError::Reconstruction(format!("greedy witness rejected: {why}")))?;
        if estimate != improved.estimate || estimate > seed_estimate {
            return Err(SelectionError::Reconstruction("greedy witness does not reproduce its estimate".into()));
        }
        return Ok(Chosen { witness: improved.witness, estimate, seed, seed_estimate, lower_bound: 0, status: ProofStatus::Feasible, timings, stats });
    }

    // A rejected seed is a defective seed policy unless the family itself is proved empty.
    let started = Instant::now();
    let searched = search(&export.model, budget, seeded.is_ok(), &mut stats);
    timings.search = started.elapsed();
    let Some(searched) = searched? else { return Err(SelectionError::Infeasible) };
    let (_, seed_estimate) = seeded.map_err(|why| SelectionError::Reconstruction(format!("seed policy is defective: {why}")))?;

    let improved = searched.incumbent.as_ref().filter(|s| s.cost() < seed_estimate);
    if searched.optimal && improved.is_none() && searched.incumbent.as_ref().is_some_and(|s| s.cost() > seed_estimate) {
        return Err(SelectionError::Reconstruction("solver optimum is costlier than the validated seed".into()));
    }
    let (witness, estimate) = match improved {
        None => (seed.clone(), seed_estimate),
        Some(solution) => {
            let witness = export.witness(solution.values());
            let (values, estimate) = audit(&export, check, &witness).map_err(|why| SelectionError::Reconstruction(format!("selected witness rejected: {why}")))?;
            if values != solution.values() || estimate != solution.cost() {
                return Err(SelectionError::Reconstruction("witness does not reproduce the solver assignment".into()));
            }
            (witness, estimate)
        }
    };
    let status = if searched.optimal && family.obligations.is_empty() { ProofStatus::ModelOptimal } else { ProofStatus::Feasible };
    Ok(Chosen { witness, estimate, seed, seed_estimate, lower_bound: searched.lower_bound.min(estimate), status, timings, stats })
}

/// Structural check, full assignment, and exact cost of a complete witness.
fn audit(export: &Export, check: &dyn Fn(&Witness) -> Result<(), String>, witness: &Witness) -> Result<(Vec<i64>, u64), String> {
    check(witness)?;
    let values = export.assignment(witness)?;
    let assessment = export.model.validate_assignment(&values).map_err(|e| e.to_string())?;
    if assessment.infeasible {
        return Err("violates the joint family's constraints".into());
    }
    let cost = assessment.exact_cost.ok_or("has no exact estimate")?;
    if export.witness(&values) != *witness {
        return Err("does not survive export round trip".into());
    }
    Ok((values, cost))
}

struct Searched {
    incumbent: Option<FeasibleSolution>,
    lower_bound: u64,
    optimal: bool,
}

fn phase(search: &Search, started: Instant) -> Phase {
    Phase { time: started.elapsed(), work: search.stats().work, nodes: search.stats().nodes }
}

/// Exact search for half the budget, then neighborhood improvement. `None` is proved
/// infeasibility. Without `improve` only the exact slice runs.
fn search(model: &Arc<Model>, budget: Budget, improve: bool, stats: &mut SearchStats) -> Result<Option<Searched>, SelectionError> {
    let started = Instant::now();
    let exact_work = (budget.work / 2).max(1);
    let mut exact = Search::new(model.clone(), Options::default()).map_err(solver_error)?;
    let outcome = exact.advance(Limits { work: exact_work, time: budget.time.map(|t| t / 2), memory_bytes: None });
    stats.exact = phase(&exact, started);
    let first = match outcome.map_err(solver_error)? {
        Outcome::Optimal(s) => return Ok(Some(Searched { lower_bound: s.cost(), incumbent: Some(s.feasible().clone()), optimal: true })),
        Outcome::Infeasible => return Ok(None),
        Outcome::Incomplete(progress) => progress,
    };
    drop(exact);
    let rest = Limits { work: budget.work.saturating_sub(exact_work), time: budget.time.map(|t| t.saturating_sub(started.elapsed())), memory_bytes: None };
    if !improve || rest.work == 0 {
        return Ok(Some(Searched { incumbent: first.incumbent, lower_bound: first.lower_bound, optimal: false }));
    }
    let options = Options { algorithm: Algorithm::Neighborhood(NeighborhoodOptions::default()), ..Options::default() };
    let restarted = Instant::now();
    let mut neighborhood = Search::new(model.clone(), options).map_err(solver_error)?;
    let outcome = neighborhood.advance(rest);
    stats.neighborhood = Some(phase(&neighborhood, restarted));
    Ok(match outcome.map_err(solver_error)? {
        Outcome::Optimal(s) => Some(Searched { lower_bound: s.cost(), incumbent: Some(s.feasible().clone()), optimal: true }),
        Outcome::Infeasible => None,
        Outcome::Incomplete(second) => {
            let incumbent = match (first.incumbent, second.incumbent) {
                (Some(a), Some(b)) => Some(if b.cost() < a.cost() { b } else { a }),
                (a, b) => a.or(b),
            };
            Some(Searched { incumbent, lower_bound: first.lower_bound.max(second.lower_bound), optimal: false })
        }
    })
}

#[cfg(test)]
mod tests {
    use super::super::{analyze, Constraint, Factor, Interval, IntervalRef};
    use super::*;
    use seismic_lang::exec::lowered_ir::LoweredIr;
    use seismic_lang::family::{Candidate, CandidateRef, Occurrence, OccurrenceId, ScopeStep, Sequence, SequenceId, Site, SiteId, SiteKind, TemplateId, Unit, UnitKind};
    use seismic_lang::sir::DefId;
    use seismic_lang::types::{RegionId, SliceId};
    use std::collections::BTreeMap;

    struct Mock {
        domains: BTreeMap<SiteId, Vec<i64>>,
        constraints: fn() -> Vec<Constraint>,
        intervals: Vec<Interval>,
        factors: fn() -> Vec<Factor>,
        seed: Witness,
    }

    impl Backend for Mock {
        type Execution = ();
        fn target(&self) -> &'static str {
            "cpu"
        }
        fn estimate_model(&self) -> String {
            "mock-v0".into()
        }
        fn numerical_environment(&self) -> String {
            "mock-environment-v0".into()
        }
        fn bind_structure(&self, _: &Program, _: &Family) -> Result<BTreeMap<SiteId, Vec<i64>>, SelectionError> {
            Ok(self.domains.clone())
        }
        fn constraints(&self, _: &Program, _: &Family) -> Result<Vec<Constraint>, SelectionError> {
            Ok((self.constraints)())
        }
        fn intervals(&self, _: &Program, _: &Family) -> Result<Vec<Interval>, SelectionError> {
            Ok(self.intervals.clone())
        }
        fn factors(&self, _: &Program, _: &Family, _: &[Interval]) -> Result<Vec<Factor>, SelectionError> {
            Ok((self.factors)())
        }
        fn seed(&self, _: &Program, _: &Family, _: &BTreeMap<SiteId, Vec<i64>>, _: &[Interval]) -> Result<Witness, SelectionError> {
            Ok(self.seed.clone())
        }
        fn realize(&self, _: LoweredIr, _: &Family, _: &Witness) -> Result<(), SelectionError> {
            Ok(())
        }
    }

    fn program() -> Program {
        Program { definitions: Vec::new(), families: Vec::new(), files: Vec::new() }
    }

    fn cand(occurrence: u32, candidate: u32) -> CandidateRef {
        CandidateRef { occurrence: OccurrenceId(occurrence), candidate }
    }

    fn candidate(children: Vec<u32>, sites: Vec<u32>, sequences: Vec<u32>) -> Candidate {
        Candidate {
            template: TemplateId(0),
            via: DefId(0),
            reference: true,
            numerical_effects: Vec::new(),
            structural: Vec::new(),
            requirements: Vec::new(),
            children: children.into_iter().map(OccurrenceId).collect(),
            sites: sites.into_iter().map(SiteId).collect(),
            sequences: sequences.into_iter().map(SequenceId).collect(),
        }
    }

    fn occurrence(id: u32, parent: Option<CandidateRef>, candidates: Vec<Candidate>) -> Occurrence {
        Occurrence { id: OccurrenceId(id), parent, call: None, family: 0, candidates, rejected: Vec::new() }
    }

    fn site(id: u32, owner: CandidateRef) -> Site {
        Site { id: SiteId(id), owner, kind: SiteKind::Width { region: RegionId(0), slice: SliceId(id) }, extent: 8 }
    }

    fn family(occurrences: Vec<Occurrence>, sites: Vec<Site>, sequences: Vec<Sequence>) -> Family {
        Family { entry: "entry".into(), target: "mock".into(), workload: Workload::default(), allow_numerical_effects: false, templates: Vec::new(), occurrences, sites, refinements: Vec::new(), sequences, obligations: Vec::new() }
    }

    fn factor(guard: Vec<CandidateRef>, intervals: Vec<u32>, scope: Vec<u32>, cost: impl Fn(&[i64]) -> u64 + Send + Sync + 'static) -> Factor {
        Factor {
            guard,
            intervals: intervals.into_iter().map(IntervalRef).collect(),
            scope: scope.into_iter().map(SiteId).collect(),
            cost: Box::new(move |v| Ok(cost(v))),
            label: "mock".into(),
        }
    }

    fn run(family: &Family, backend: &Mock) -> Result<Chosen, SelectionError> {
        select_family(&program(), family, backend, Budget::default(), &|_| Ok(()))
    }

    fn run_greedy(family: &Family, backend: &Mock) -> Result<Chosen, SelectionError> {
        select_family(&program(), family, backend, Budget { strategy: Strategy::Greedy, ..Budget::default() }, &|_| Ok(()))
    }

    const TWENTY: u32 = 20;

    // Occurrence i in 1..=20: candidate 0 owns site i-1 over {1,2,4}; 1 and 2 are constants.
    fn twenty_factors() -> Vec<Factor> {
        let mut factors = Vec::new();
        for i in 1..=TWENTY {
            let base = if i % 3 == 0 { 10 } else { 60 };
            factors.push(factor(vec![cand(i, 0)], vec![], vec![i - 1], move |v| base + [5, 1, 9][v[0].trailing_zeros() as usize]));
            factors.push(factor(vec![cand(i, 1)], vec![], vec![], move |_| if i % 3 == 1 { 30 } else { 70 }));
            factors.push(factor(vec![cand(i, 2)], vec![], vec![], move |_| if i % 3 == 2 { 30 } else { 80 }));
        }
        factors
    }

    fn twenty() -> (Family, Mock) {
        let mut occurrences = vec![occurrence(0, None, vec![candidate((1..=TWENTY).collect(), vec![], vec![])])];
        let mut sites = Vec::new();
        let mut seed = Witness::default();
        seed.choices.insert(OccurrenceId(0), 0);
        for i in 1..=TWENTY {
            occurrences.push(occurrence(i, Some(cand(0, 0)), vec![candidate(vec![], vec![i - 1], vec![]), candidate(vec![], vec![], vec![]), candidate(vec![], vec![], vec![])]));
            sites.push(site(i - 1, cand(i, 0)));
            seed.choices.insert(OccurrenceId(i), 0);
            seed.sites.insert(SiteId(i - 1), 1);
        }
        let domains: BTreeMap<SiteId, Vec<i64>> = (0..TWENTY).map(|s| (SiteId(s), vec![1, 2, 4])).collect();
        (family(occurrences, sites, Vec::new()), Mock { domains, constraints: Vec::new, intervals: Vec::new(), factors: twenty_factors, seed })
    }

    fn twenty_optimum() -> u64 {
        (1..=TWENTY).map(|i| if i % 3 == 0 { 11 } else { 30 }).sum()
    }

    #[test]
    fn independent_choices_solve_locally() {
        let (family, backend) = twenty();
        let domains = backend.domains.clone();

        let chosen = run(&family, &backend).expect("feasible");
        assert_eq!(chosen.status, ProofStatus::ModelOptimal);
        assert_eq!((chosen.estimate, chosen.lower_bound), (twenty_optimum(), twenty_optimum()));
        assert_eq!(chosen.seed_estimate, (1..=TWENTY).map(|i| if i % 3 == 0 { 15 } else { 65 }).sum::<u64>());
        for i in 1..=TWENTY {
            assert_eq!(chosen.witness.choices[&OccurrenceId(i)], i % 3);
            assert_eq!(chosen.witness.sites.get(&SiteId(i - 1)), (i % 3 == 0).then_some(&2));
        }

        // True independence is visible: no factor spans more than one occurrence's own
        // choice and site.
        let export = Export::build(&family, &domains, &[], &[], &twenty_factors(), None).expect("export");
        assert!(export.model.factors().iter().all(|f| f.scope().len() <= 2));
        assert_eq!(analyze(&family).independent_components, TWENTY as usize);
    }

    #[test]
    fn precision_is_a_hard_candidate_constraint_and_qualification_is_witness_scoped() {
        let mut reference = candidate(vec![], vec![], vec![]);
        reference.reference = true;
        let mut alternative = candidate(vec![], vec![], vec![]);
        alternative.reference = false;
        let mut exact = family(vec![occurrence(0, None, vec![reference.clone(), alternative.clone()])], vec![], vec![]);
        let factors = || vec![factor(vec![cand(0, 0)], vec![], vec![], |_| 100), factor(vec![cand(0, 1)], vec![], vec![], |_| 10)];
        let seed = Witness { choices: [(OccurrenceId(0), 0)].into(), ..Witness::default() };
        let backend = Mock { domains: BTreeMap::new(), constraints: Vec::new, intervals: Vec::new(), factors, seed: seed.clone() };

        let chosen = run(&exact, &backend).expect("exact reference is feasible");
        assert_eq!(chosen.witness.choices[&OccurrenceId(0)], 0);

        exact.workload.precision = seismic_lang::precision::PrecisionPolicy::Unconstrained;
        let chosen = run(&exact, &backend).expect("exploration admits the alternative");
        assert_eq!(chosen.witness.choices[&OccurrenceId(0)], 1);

        exact.workload.precision = seismic_lang::precision::PrecisionPolicy::Exact;
        let qualified = Witness { choices: [(OccurrenceId(0), 1)].into(), ..Witness::default() };
        let export = Export::build(&exact, &BTreeMap::new(), &[], &[], &(backend.factors)(), Some(&qualified)).expect("qualified export");
        let assignment = export.assignment(&qualified).expect("qualified alternative is admitted only for this witness");
        assert!(!export.model.validate_assignment(&assignment).expect("valid model").infeasible);
    }

    #[test]
    fn changing_the_precision_threshold_changes_the_selected_qualified_witness() {
        use seismic_lang::precision::{qualify_f32, EvidenceRequirement, Limit, PrecisionPolicy, Tolerance};
        use seismic_lang::program::{compile, SourceFile};

        let source = SourceFile {
            path: "precision.seismic".into(),
            text: "fn copy[N](x: tensor[N] f32, out y: tensor[N] f32):\n    publish x to y\n\nlower copy[N](x: tensor[N] f32, out y: tensor[N] f32) for cpu:\n    publish x to y\n".into(),
        };
        let program = compile(&[source]).unwrap();
        let factors = || vec![
            factor(vec![cand(0, 0)], vec![], vec![], |_| 100),
            factor(vec![cand(0, 1)], vec![], vec![], |_| 10),
        ];
        let reference = Witness { choices: [(OccurrenceId(0), 0)].into(), ..Witness::default() };
        let backend = Mock { domains: BTreeMap::new(), constraints: Vec::new, intervals: Vec::new(), factors, seed: reference };
        let loose = PrecisionPolicy::Bounded {
            default: Tolerance {
                absolute: Limit::new(0.01).unwrap(),
                relative: Limit::new(0.1).unwrap(),
                relative_floor: Limit::new(0.001).unwrap(),
                ulps: None,
            },
            outputs: BTreeMap::new(),
            evidence: EvidenceRequirement::Qualified,
            specials: Default::default(),
            inputs: BTreeMap::new(),
        };
        let workload = Workload { shapes: [("N".into(), 1)].into(), elems: BTreeMap::new(), precision: loose.clone() };
        let alternative = Witness { choices: [(OccurrenceId(0), 1)].into(), ..Witness::default() };
        let assessment = qualify_f32("y", &[1.0], &[1.05], loose, vec!["mock alternative".into()]).unwrap();
        let qualification = Qualification::new(
            &program,
            &backend,
            "copy",
            &workload,
            alternative,
            assessment,
            "threshold-test",
            "common-comparator-v1",
        ).unwrap();

        let selected = select_qualified(&program, "copy", &workload, &backend, Budget::default(), &[qualification.clone()]).unwrap();
        assert_eq!(selected.witness.choices[&OccurrenceId(0)], 1);
        assert_eq!(selected.numerical_assessment.evidence, seismic_lang::precision::EvidenceClass::Qualified);

        let mut strict = workload.clone();
        strict.precision = PrecisionPolicy::Bounded {
            default: Tolerance { absolute: Limit::new(0.001).unwrap(), relative: Limit::ZERO, relative_floor: Limit::ZERO, ulps: None },
            outputs: BTreeMap::new(),
            evidence: EvidenceRequirement::Qualified,
            specials: Default::default(),
            inputs: BTreeMap::new(),
        };
        let selected = select_qualified(&program, "copy", &strict, &backend, Budget::default(), &[qualification]).unwrap();
        assert_eq!(selected.witness.choices[&OccurrenceId(0)], 0);
        assert_eq!(selected.numerical_assessment.evidence, seismic_lang::precision::EvidenceClass::Exact);
    }

    // Root owns site 0 and a two-unit sequence [local, call]. The call's fast body (0) owns
    // site 1; its slower body (1) owns site 2 and is the only one the fused interval admits,
    // at a common width.
    fn coupled_factors() -> Vec<Factor> {
        vec![
            factor(vec![cand(0, 0)], vec![], vec![0], |v| (80 / v[0]) as u64),
            factor(vec![cand(1, 0)], vec![], vec![1], |v| (4 + 8 / v[0]) as u64),
            factor(vec![cand(1, 1)], vec![], vec![2], |v| (30 / v[0]) as u64),
            factor(vec![], vec![0], vec![], |_| 40),
            factor(vec![], vec![1], vec![], |_| 40),
            factor(vec![], vec![2], vec![], |_| 10),
        ]
    }

    fn coupled() -> (Family, Mock) {
        let unit = |kind| Unit { statements: 0..1, kind, completion_after: false };
        let sequence = Sequence { id: SequenceId(0), owner: cand(0, 0), scope: vec![ScopeStep::Region(RegionId(0))], units: vec![unit(UnitKind::Local), unit(UnitKind::Call(OccurrenceId(1)))] };
        let family = family(
            vec![
                occurrence(0, None, vec![candidate(vec![1], vec![0], vec![0])]),
                occurrence(1, Some(cand(0, 0)), vec![candidate(vec![], vec![1], vec![]), candidate(vec![], vec![2], vec![])]),
            ],
            vec![site(0, cand(0, 0)), site(1, cand(1, 0)), site(2, cand(1, 1))],
            vec![sequence],
        );
        let interval = |start, end, requires: Vec<CandidateRef>, equal_sites| Interval { sequence: SequenceId(0), start, end, requires, equal_sites };
        let intervals = vec![interval(0, 1, vec![], vec![]), interval(1, 2, vec![], vec![]), interval(0, 2, vec![cand(1, 1)], vec![(SiteId(0), SiteId(2))])];
        let seed = Witness {
            choices: [(OccurrenceId(0), 0), (OccurrenceId(1), 0)].into(),
            sites: [(SiteId(0), 8), (SiteId(1), 8)].into(),
            covers: [(SequenceId(0), vec![(0, 1), (1, 2)])].into(),
        };
        let domains = [(SiteId(0), vec![1, 2, 4, 8]), (SiteId(1), vec![1, 2, 4, 8]), (SiteId(2), vec![1, 2])].into();
        (family, Mock { domains, constraints: Vec::new, intervals, factors: coupled_factors, seed })
    }

    #[test]
    fn parent_total_selects_locally_slower_child() {
        let (family, backend) = coupled();
        let chosen = run(&family, &backend).expect("feasible");

        assert_eq!((chosen.seed_estimate, chosen.estimate, chosen.status), (95, 65, ProofStatus::ModelOptimal));
        let expected = Witness {
            choices: [(OccurrenceId(0), 0), (OccurrenceId(1), 1)].into(),
            sites: [(SiteId(0), 2), (SiteId(2), 2)].into(),
            covers: [(SequenceId(0), vec![(0, 2)])].into(),
        };
        assert_eq!(chosen.witness, expected);
    }

    // Greedy is a diagnostic strategy: feasible, never worse than the seed, never a proof.
    // Independent decisions are its best case (it reaches the optimum); the coupled family
    // needs a choice, a cover and two sites to move together, which coordinate moves cannot.
    #[test]
    fn greedy_improves_the_seed_without_search() {
        let (family, backend) = twenty();
        let chosen = run_greedy(&family, &backend).expect("feasible");
        assert_eq!((chosen.estimate, chosen.status, chosen.lower_bound), (twenty_optimum(), ProofStatus::Feasible, 0));
        assert!(chosen.stats.greedy_trials > 0 && chosen.stats.exact.work == 0);

        let (family, backend) = coupled();
        let chosen = run_greedy(&family, &backend).expect("feasible");
        assert!(chosen.estimate <= chosen.seed_estimate && chosen.estimate >= 65);
        assert_eq!(chosen.status, ProofStatus::Feasible);

        let (family, backend) = one_site(|| constraint(|_| true));
        let chosen = run_greedy(&family, &backend).expect("feasible");
        assert_eq!((chosen.estimate, chosen.seed_estimate), (0, 0));
    }

    fn one_site(constraints: fn() -> Vec<Constraint>) -> (Family, Mock) {
        let family = family(vec![occurrence(0, None, vec![candidate(vec![], vec![0], vec![])])], vec![site(0, cand(0, 0))], Vec::new());
        let seed = Witness { choices: [(OccurrenceId(0), 0)].into(), sites: [(SiteId(0), 1)].into(), covers: BTreeMap::new() };
        (family, Mock { domains: [(SiteId(0), vec![1, 2])].into(), constraints, intervals: Vec::new(), factors: Vec::new, seed })
    }

    fn constraint(holds: fn(&[i64]) -> bool) -> Vec<Constraint> {
        vec![Constraint { guard: vec![cand(0, 0)], scope: vec![SiteId(0)], holds: Box::new(holds), reason: "capacity".into() }]
    }

    #[test]
    fn unsatisfiable_constraint_is_infeasible() {
        let (family, backend) = one_site(|| constraint(|_| false));
        assert!(matches!(run(&family, &backend), Err(SelectionError::Infeasible)));
    }

    #[test]
    fn infeasible_seed_is_a_reconstruction_defect() {
        let (family, backend) = one_site(|| constraint(|v| v[0] == 2));
        assert!(matches!(run(&family, &backend), Err(SelectionError::Reconstruction(_))));
    }
}
