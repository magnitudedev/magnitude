//! Feedback owns its sparse populations and all search decisions. Preparation
//! remains the authority for candidates, numerical admission and finalization.
use super::confirmation::{Arm, Comparison, Verdict};
use super::evolution::{finite_coordinates, fresh_in_family, propose, Operator, Random};
use super::invocation::{Navigator, Point};
use super::scheduling::{
    affordable_sweep, order_cost, FormationCosts, IntentWindow, Lane, Ledger, ObservedCost,
};
use super::statistics::median;
use super::{
    ControlledObserver, FeedbackCosts, FeedbackError, FeedbackOptions, Observation,
    ObservationCase, ObservationError, ObservationProtocol, ObservationRequest,
};
use crate::candidate_domain::{CandidateCoordinate, CandidatePoint, NonEmpty};
use crate::errors::PreparationError;
use crate::evaluation::{CandidateEvaluator, EvaluationIdentity, EvaluationProvenance};
use crate::evaluation_session::{EvaluationSession, PreparedCandidateId, RealizationAdmission};
use crate::planning::{
    OptimizationCompletion, PlanningBudgetReport, PlanningLimit, PlanningReport, SelectionPolicy,
};
use crate::prepared::{CandidateIndex, SelectionFunction};
use seismic_target::{NativeCompiler, TargetFamily};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet, VecDeque};
use std::time::{Duration, Instant};

#[derive(Clone, Debug, Default)]
pub struct FeedbackReport {
    pub elapsed: Duration,
    pub evaluation_fingerprint: [u8; 32],
    pub measurement_environment: Option<[u8; 32]>,
    pub costs: FeedbackCosts,
    pub attempted_points: usize,
    pub measured_points: usize,
    pub content_dependent_observations: u64,
    pub prepared_candidates: usize,
    pub native_code_bytes: u64,
    pub native_metadata_bytes: u64,
    pub retained_native_code_bytes: u64,
    pub retained_native_metadata_bytes: u64,
    pub native_budget_exhausted: bool,
    pub retained_variants: u64,
    pub retained_metadata_bytes: u64,
    pub native_hits: u64,
    pub native_misses: u64,
    pub confirmation_events: u64,
    pub confirmed_points: usize,
    pub environment_changes: u64,
    pub deferred_cases: u64,
    pub unresolved_witness_searches: u64,
    pub navigation_resource_limited: bool,
    pub inconclusive_comparisons: u64,
    pub shared_setup_batches: u64,
    pub enumerated_screens: u64,
    /// Preparation attempts left pending by unresolved numerical applicability.
    pub numerical_pending_attempts: u64,
    pub construction_work_units: u64,
    pub construction_pending_paths: usize,
    pub initialization_pending_paths: usize,
}

impl FeedbackReport {
    pub fn timing_endpoint(&self) -> &'static str {
        "resident complete-entry submission through native completion; setup, reset and checks excluded"
    }
    pub fn confidence_assumptions(&self) -> &'static str {
        "independent identically distributed timings within each fixed arm/event; randomized interleaving does not prove stationarity"
    }
}

struct Candidate {
    coordinate: CandidateCoordinate,
    executable: Option<PreparedCandidateId>,
    rejected: bool,
}
enum ExperimentOutcome {
    Measured(Observation),
    Inapplicable,
    Deferred,
}
struct PointPopulation {
    point: Point,
    cases: [ObservationCase; 2],
    incumbent: usize,
    screens: HashMap<usize, f64>,
    attempted: HashSet<usize>,
    deferred: HashSet<usize>,
    refinement_deferred: bool,
    refinement_visits: u64,
    pending: VecDeque<usize>,
    comparison: Option<Promotion>,
    cost: ObservedCost,
    confirmation_cost: ObservedCost,
}
impl PointPopulation {
    fn next_stage_cost(&self) -> Option<Duration> {
        if self.comparison.is_some() || !self.pending.is_empty() {
            self.confirmation_cost.admission()
        } else {
            self.cost.admission()
        }
    }
    fn retain_donors(&mut self) {
        let mut protected = HashSet::from([0, self.incumbent]);
        if let Some(comparison) = &self.comparison {
            protected.extend([comparison.incumbent, comparison.challenger]);
        }
        let mut scores = self
            .screens
            .iter()
            .map(|(candidate, score)| (*candidate, *score))
            .collect::<Vec<_>>();
        scores.sort_by(|(a, x), (b, y)| x.total_cmp(y).then(a.cmp(b)));
        for (candidate, _) in scores {
            if protected.len() >= 8 {
                break;
            }
            protected.insert(candidate);
        }
        self.screens
            .retain(|candidate, _| protected.contains(candidate));
        self.pending
            .retain(|candidate| self.screens.contains_key(candidate));
    }
}

struct Promotion {
    incumbent: usize,
    challenger: usize,
    case: usize,
    comparison: Comparison,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug)]
enum TestStrategy {
    Evolution,
    Fresh,
    Enumeration,
}

#[cfg(test)]
struct TestObservationPolicy {
    fixed_points: Option<VecDeque<Point>>,
    screen_trials: usize,
    allowance: Option<usize>,
    spent: usize,
    exhausted: bool,
}
#[cfg(test)]
impl Default for TestObservationPolicy {
    fn default() -> Self {
        Self {
            fixed_points: None,
            screen_trials: 3,
            allowance: None,
            spent: 0,
            exhausted: false,
        }
    }
}

pub(crate) struct FeedbackEvaluator<O> {
    observer: O,
    construction: crate::evaluation::construction::ConstructionTraversal,
    construction_allowance: crate::candidate_domain::ConstructionAllowance,
    options: FeedbackOptions,
    navigator: Navigator,
    random: Random,
    ledger: Ledger,
    candidates: Vec<Candidate>,
    coordinates: HashMap<CandidateCoordinate, usize>,
    points: Vec<PointPopulation>,
    point_cursor: usize,
    refinement_round: Vec<usize>,
    fresh_point: bool,
    discovery_window: IntentWindow<(usize, usize)>,
    formation_costs: FormationCosts,
    finite_candidates: Option<Vec<CandidateCoordinate>>,
    next_operator: usize,
    family_cursor: usize,
    next_event: u64,
    environment: Option<[u8; 32]>,
    evidence: Sha256,
    report: FeedbackReport,
    #[cfg(test)]
    strategy: TestStrategy,
    #[cfg(test)]
    cost_ordering: bool,
    #[cfg(test)]
    observation_policy: TestObservationPolicy,
}

impl<O> FeedbackEvaluator<O> {
    pub(crate) fn new<T: TargetFamily, C: NativeCompiler<T>>(
        session: &EvaluationSession<'_, T, C>,
        observer: O,
        options: FeedbackOptions,
        started: Instant,
    ) -> Result<Self, PreparationError> {
        if options.archive_points == 0
            || options.archive_candidates == 0
            || options.experiment_memory_bytes == 0
            || options.reference_work_limit == 0
        {
            return Err(PreparationError::Feedback(FeedbackError::InvalidOptions(
                "archive and experiment resource limits must be positive".into(),
            )));
        }
        let navigator = Navigator::new(
            session.domain(),
            options.optimize_for.as_ref(),
            options.archive_points.saturating_mul(128),
        )
        .map_err(|error| PreparationError::Feedback(FeedbackError::InvalidOptions(error)))?;
        let mut ledger = Ledger::new(started, options.search_time);
        ledger.costs.domain = started.elapsed();
        let mut evidence = Sha256::new();
        evidence.update(b"seismic-feedback-evidence-v1");
        evidence.update(super::EvaluationMethod::Feedback(options.clone()).fingerprint());
        Ok(Self {
            observer,
            construction: crate::evaluation::construction::ConstructionTraversal::new(
                session.domain(),
            ),
            construction_allowance: session.construction_allowance(),
            random: Random(options.seed),
            options: options.clone(),
            navigator,
            ledger,
            candidates: Vec::new(),
            coordinates: HashMap::new(),
            points: Vec::new(),
            point_cursor: 0,
            refinement_round: Vec::new(),
            fresh_point: true,
            discovery_window: IntentWindow::default(),
            formation_costs: FormationCosts::default(),
            finite_candidates: None,
            next_operator: 0,
            family_cursor: 0,
            next_event: 0,
            environment: None,
            evidence,
            report: FeedbackReport::default(),
            #[cfg(test)]
            strategy: TestStrategy::Evolution,
            #[cfg(test)]
            cost_ordering: true,
            #[cfg(test)]
            observation_policy: TestObservationPolicy::default(),
        })
    }

    pub(crate) fn resume(&mut self, additional: Duration) {
        self.ledger.resume(additional);
        self.refinement_round.clear();
        self.discovery_window.resume();
        for point in &mut self.points {
            point.deferred.clear();
            point.refinement_deferred = false;
        }
    }
    pub(crate) fn finalized(
        &mut self,
        elapsed: Duration,
        planning_report: Option<&PlanningReport>,
    ) {
        self.ledger.published(elapsed);
        self.report.costs = self.ledger.costs.clone();
        if let Some(planning_report) = planning_report {
            self.report.retained_variants = planning_report.budget.executable_variants;
            self.report.retained_metadata_bytes = planning_report.budget.retained_metadata_bytes;
        }
    }

    pub(crate) fn suspend(&mut self) {
        self.ledger.suspend();
        self.report.elapsed = self.ledger.elapsed();
        self.report.navigation_resource_limited = self.navigator.resource_limited;
    }

    pub(crate) fn report(&self) -> &FeedbackReport {
        &self.report
    }

    fn add_candidate(&mut self, coordinate: CandidateCoordinate) -> Option<usize> {
        if let Some(index) = self.coordinates.get(&coordinate) {
            return Some(*index);
        }
        if self.candidates.len() == self.options.archive_candidates {
            return None;
        }
        let index = self.candidates.len();
        self.coordinates.insert(coordinate.clone(), index);
        self.candidates.push(Candidate {
            coordinate,
            executable: None,
            rejected: false,
        });
        Some(index)
    }

    fn prepare<T: TargetFamily, C: NativeCompiler<T>>(
        &mut self,
        session: &mut EvaluationSession<'_, T, C>,
        index: usize,
    ) -> Result<Option<PreparedCandidateId>, PreparationError> {
        let candidate = &mut self.candidates[index];
        if candidate.rejected {
            return Ok(None);
        }
        if let Some(executable) = candidate.executable {
            return Ok(Some(executable));
        }
        let start = Instant::now();
        let requirements = session.inspect(&candidate.coordinate)?;
        self.report.native_hits +=
            (requirements.required.len() - requirements.missing.len()) as u64;
        self.report.native_misses += requirements.missing.len() as u64;
        let admitted = session.realize_checked(&candidate.coordinate)?;
        let elapsed = start.elapsed();
        self.ledger.costs.native_and_executable += elapsed;
        self.formation_costs.record(&requirements.missing, elapsed);
        match admitted {
            RealizationAdmission::Ready(executable) => {
                candidate.executable = Some(executable);
                Ok(Some(executable))
            }
            RealizationAdmission::Rejected(_) => {
                candidate.rejected = true;
                Ok(None)
            }
            RealizationAdmission::Pending(crate::evaluation::RealizationPending::NumericalAnalysis) => {
                self.report.numerical_pending_attempts += 1;
                Ok(None)
            }
            RealizationAdmission::Pending(crate::evaluation::RealizationPending::NativeBudget) => Ok(None),
        }
    }

    fn observe<T: TargetFamily, C: NativeCompiler<T>>(
        &mut self,
        session: &mut EvaluationSession<'_, T, C>,
        point: usize,
        candidate: usize,
        case: usize,
        protocol: ObservationProtocol,
    ) -> Result<ExperimentOutcome, PreparationError>
    where
        O: ControlledObserver<T, C::Handle>,
    {
        if !session
            .domain()
            .contains(&CandidatePoint::new(
                self.candidates[candidate].coordinate.clone(),
                self.points[point].cases[case].values.clone(),
            ))
            .unwrap_or(false)
        {
            return Ok(ExperimentOutcome::Inapplicable);
        }
        let observation_cost = if protocol.trials == ObservationProtocol::SCREEN.trials {
            self.points[point].cost.admission()
        } else {
            self.points[point].confirmation_cost.admission()
        };
        let formation_cost = if self.candidates[candidate].executable.is_some() {
            Some(Duration::ZERO)
        } else {
            let requirements = session.inspect(&self.candidates[candidate].coordinate)?;
            self.formation_costs.estimate(&requirements.missing, true)
        };
        if !self.ledger.allows(observation_cost)
            || !self.ledger.allows(formation_cost)
            || !self.ledger.allows(
                observation_cost
                    .zip(formation_cost)
                    .map(|(a, b)| a.saturating_add(b)),
            )
        {
            return Ok(ExperimentOutcome::Deferred);
        }
        let Some(executable) = self.prepare(session, candidate)? else {
            return Ok(if self.candidates[candidate].rejected {
                ExperimentOutcome::Inapplicable
            } else {
                ExperimentOutcome::Deferred
            });
        };
        if !self.ledger.allows(None) {
            return Ok(ExperimentOutcome::Deferred);
        }
        if !session.is_admitted(executable, &self.points[point].cases[case].values)? {
            self.report.numerical_pending_attempts += 1;
            return Ok(ExperimentOutcome::Deferred);
        }
        let prepared_candidate = executable;
        let executable = session.executable(prepared_candidate)?;
        #[cfg(test)]
        if self.observation_policy.allowance.is_some_and(|allowance| {
            self.observation_policy
                .spent
                .saturating_add(protocol.warmup)
                .saturating_add(protocol.trials)
                > allowance
        }) {
            self.observation_policy.exhausted = true;
            return Ok(ExperimentOutcome::Deferred);
        }
        let started = Instant::now();
        let reference = session.domain().reference();
        let outcome = self.observer.observe(ObservationRequest {
            candidate: prepared_candidate,
            reference: reference.view(),
            executable,
            precision: session.domain().precision(),
            case: &self.points[point].cases[case],
            protocol,
            memory_limit: self.options.experiment_memory_bytes,
            reference_work_limit: self.options.reference_work_limit,
            deadline: self.ledger.deadline(),
        });
        drop(reference);
        let elapsed = started.elapsed();
        self.ledger.costs.observation += elapsed;
        if protocol.trials == ObservationProtocol::SCREEN.trials {
            self.points[point].cost.record(elapsed);
        } else {
            self.points[point].confirmation_cost.record(elapsed);
        }
        let observation = match outcome {
            Ok(observation) => observation,
            Err(
                ObservationError::Capacity { .. }
                | ObservationError::ReferenceLimit { .. }
                | ObservationError::DeadlineExpired,
            ) => {
                self.report.deferred_cases += 1;
                return Ok(ExperimentOutcome::Deferred);
            }
            Err(error) => {
                return Err(PreparationError::Feedback(FeedbackError::Observation(
                    error,
                )));
            }
        };
        #[cfg(test)]
        {
            self.observation_policy.spent += protocol.warmup + protocol.trials;
        }
        if observation.samples.len() != protocol.trials {
            return Err(PreparationError::Feedback(FeedbackError::Observation(
                ObservationError::InvalidCase(
                    "observer returned an incomplete timing block".into(),
                ),
            )));
        }
        self.ledger.costs.input_and_admission += observation.setup_time;
        self.ledger.costs.reference_and_checks += observation.checking_time;
        self.accept_environment(observation.environment);
        if observation.support.depends_on_contents() {
            self.report.content_dependent_observations += 1;
        }
        self.evidence.update([
            observation.support.content_control as u8,
            observation.support.content_addressing as u8,
            observation.support.content_checks as u8,
        ]);
        self.evidence.update(observation.environment);
        self.evidence
            .update(self.candidates[candidate].coordinate.family().digest());
        self.evidence
            .update((self.candidates[candidate].coordinate.choices().len() as u64).to_le_bytes());
        for (_, value) in self.candidates[candidate].coordinate.choices() {
            self.evidence.update(value.to_le_bytes());
        }
        self.evidence
            .update((self.points[point].point.0.len() as u64).to_le_bytes());
        for rank in &self.points[point].point.0 {
            self.evidence.update(rank.to_le_bytes());
        }
        self.evidence
            .update(self.points[point].cases[case].seed.to_le_bytes());
        self.evidence.update((protocol.warmup as u64).to_le_bytes());
        self.evidence.update((protocol.trials as u64).to_le_bytes());
        for sample in &observation.samples {
            self.evidence.update(sample.as_nanos().to_le_bytes());
        }
        Ok(ExperimentOutcome::Measured(observation))
    }

    fn accept_environment(&mut self, environment: [u8; 32]) {
        if self
            .environment
            .is_some_and(|previous| previous != environment)
        {
            self.report.environment_changes += 1;
            self.evidence = Sha256::new();
            self.evidence
                .update(b"seismic-feedback-environment-change-v1");
            self.evidence.update(environment);
            for population in &mut self.points {
                population.incumbent = 0;
                population.screens.clear();
                population.attempted.clear();
                population.deferred.clear();
                population.refinement_deferred = false;
                population.pending.clear();
                population.comparison = None;
            }
        }
        self.environment = Some(environment);
    }

    fn screen<T: TargetFamily, C: NativeCompiler<T>>(
        &mut self,
        session: &mut EvaluationSession<'_, T, C>,
        point: usize,
        candidate: usize,
    ) -> Result<(), PreparationError>
    where
        O: ControlledObserver<T, C::Handle>,
    {
        if self.points[point].attempted.contains(&candidate)
            || self.points[point].deferred.contains(&candidate)
        {
            return Ok(());
        }
        let protocol = ObservationProtocol::SCREEN;
        #[cfg(test)]
        let protocol = ObservationProtocol {
            trials: self.observation_policy.screen_trials,
            ..protocol
        };
        let observation = match self.observe(session, point, candidate, 0, protocol)? {
            ExperimentOutcome::Measured(observation) => Some(observation),
            ExperimentOutcome::Inapplicable => None,
            ExperimentOutcome::Deferred => {
                self.points[point].deferred.insert(candidate);
                return Ok(());
            }
        };
        if let Some(observation) = observation {
            let score = median(
                &observation
                    .samples
                    .iter()
                    .map(Duration::as_secs_f64)
                    .collect::<Vec<_>>(),
            );
            let population = &mut self.points[point];
            population.screens.insert(candidate, score);
            if population
                .screens
                .get(&population.incumbent)
                .is_some_and(|incumbent| score < *incumbent)
            {
                population.pending.push_back(candidate);
            }
            population.retain_donors();
        }
        let coordinate = self.candidates[candidate].coordinate.clone();
        self.navigator
            .prioritize_boundary(&self.points[point].point, |values| {
                session
                    .domain()
                    .contains(&CandidatePoint::new(coordinate.clone(), values.clone()))
                    .ok()
            });
        self.points[point].attempted.insert(candidate);
        Ok(())
    }

    fn propose_discovery<T: TargetFamily, C: NativeCompiler<T>>(
        &mut self,
        session: &mut EvaluationSession<'_, T, C>,
    ) -> Result<Option<(usize, usize)>, PreparationError>
    where
        O: ControlledObserver<T, C::Handle>,
    {
        self.fresh_point = !self.fresh_point;
        if (self.points.is_empty() || self.fresh_point)
            && self.points.len() < self.options.archive_points
            && !self.navigator.exhausted()
            && !self.navigator.resource_limited
        {
            #[cfg(test)]
            let next = match self.observation_policy.fixed_points.as_mut() {
                Some(points) => points.pop_front(),
                None => self.navigator.next(&mut self.random),
            };
            #[cfg(not(test))]
            let next = self.navigator.next(&mut self.random);
            if let Some(point) = next {
                let first = self
                    .navigator
                    .case(session.domain(), &point, self.random.next());
                let second = self
                    .navigator
                    .case(session.domain(), &point, self.random.next());
                let cases = [first, second]
                    .into_iter()
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|error| {
                        PreparationError::Feedback(FeedbackError::Observation(
                            ObservationError::InvalidCase(error),
                        ))
                    })?;
                self.points.push(PointPopulation {
                    point,
                    cases: cases.try_into().unwrap(),
                    incumbent: 0,
                    screens: HashMap::new(),
                    attempted: HashSet::new(),
                    deferred: HashSet::new(),
                    refinement_deferred: false,
                    refinement_visits: 0,
                    pending: VecDeque::new(),
                    comparison: None,
                    cost: ObservedCost::default(),
                    confirmation_cost: ObservedCost::default(),
                });
                return Ok(Some((self.points.len() - 1, 0)));
            }
        }
        if self.points.is_empty() {
            return Ok(None);
        }
        let point = self.point_cursor % self.points.len();
        self.point_cursor += 1;
        let family = self.family_cursor;
        self.family_cursor = self.family_cursor.wrapping_add(1);
        if let Some(coordinate) = fresh_in_family(session.domain(), family, &mut self.random) {
            if let Some(candidate) = self.add_candidate(coordinate) {
                return Ok(Some((point, candidate)));
            }
        }
        Ok(None)
    }

    fn discover<T: TargetFamily, C: NativeCompiler<T>>(
        &mut self,
        session: &mut EvaluationSession<'_, T, C>,
    ) -> Result<(), PreparationError>
    where
        O: ControlledObserver<T, C::Handle>,
    {
        self.discovery_window.refill();
        if self.discovery_window.is_empty() {
            // Bound proposal work as well as the retained experiment window.
            for _ in 0..32 {
                if !self.ledger.allows(None) || self.discovery_window.len() == 8 {
                    break;
                }
                if let Some(intent) = self.propose_discovery(session)? {
                    let (point, candidate) = intent;
                    if !self.points[point].attempted.contains(&candidate)
                        && !self.points[point].deferred.contains(&candidate)
                        && !self
                            .discovery_window
                            .iter()
                            .any(|(_, _, queued)| *queued == intent)
                    {
                        self.discovery_window.push(intent);
                    }
                }
            }
        }
        // Inspection never compiles. Reinspect every visit: earlier experiments
        // may have made a shared unit resident since the intent was proposed.
        let mut ready = Vec::new();
        for (index, age, &(point, candidate)) in self.discovery_window.iter() {
            let requirements = session.inspect(&self.candidates[candidate].coordinate)?;
            let formation = if self.candidates[candidate].executable.is_some() {
                Some(Duration::ZERO)
            } else {
                self.formation_costs.estimate(&requirements.missing, false)
            };
            let observation = self.points[point].cost.ordering();
            let cost = formation.zip(observation).map(|(a, b)| a.saturating_add(b));
            ready.push((index, age, point, candidate, cost, requirements.missing));
        }
        #[cfg(not(test))]
        let cost_ordering = true;
        #[cfg(test)]
        let cost_ordering = self.cost_ordering;
        if cost_ordering {
            ready.sort_by(|a, b| order_cost(a.4, a.1, b.4, b.1));
        }
        let Some(first) = ready.first() else {
            return Ok(());
        };
        let selected = (first.2, first.3);
        // A batch contains only already eligible intents from this round. Share
        // native setup, not submissions: each invocation keeps its own endpoint.
        let shared = ready
            .iter()
            .skip(1)
            .find(|item| !first.5.is_empty() && item.5.iter().any(|key| first.5.contains(key)))
            .map(|item| (item.2, item.3));
        let mut batch = vec![selected];
        if let Some(shared) = shared.filter(|_| cost_ordering) {
            self.report.shared_setup_batches += 1;
            batch.push(shared);
        }
        for intent in batch {
            if !self.ledger.allows(None) {
                break;
            }
            let index = self
                .discovery_window
                .iter()
                .find(|(_, _, queued)| **queued == intent)
                .map(|(index, _, _)| index)
                .unwrap();
            self.screen(session, intent.0, intent.1)?;
            if self.points[intent.0].deferred.contains(&intent.1) {
                self.discovery_window.defer(index);
            } else {
                self.discovery_window.remove(index);
            }
        }
        Ok(())
    }

    fn enumerated_candidate<T: TargetFamily, C: NativeCompiler<T>>(
        &self,
        session: &EvaluationSession<'_, T, C>,
        point: usize,
    ) -> Result<Option<CandidateCoordinate>, PreparationError> {
        #[cfg(test)]
        if matches!(self.strategy, TestStrategy::Fresh) {
            return Ok(None);
        }
        let Some(coordinates) = &self.finite_candidates else {
            return Ok(None);
        };
        let remaining = coordinates
            .iter()
            .filter(|coordinate| {
                self.coordinates.get(*coordinate).is_none_or(|candidate| {
                    !self.points[point].attempted.contains(candidate)
                        && !self.points[point].deferred.contains(candidate)
                })
            })
            .collect::<Vec<_>>();
        let mut missing = Vec::new();
        let mut needs_formation = false;
        for coordinate in &remaining {
            if self
                .coordinates
                .get(*coordinate)
                .is_some_and(|index| self.candidates[*index].executable.is_some())
            {
                continue;
            }
            needs_formation = true;
            for key in session.inspect(coordinate)?.missing {
                if !missing.contains(&key) {
                    missing.push(key);
                }
            }
        }
        let formation = if needs_formation {
            self.formation_costs.estimate(&missing, true)
        } else {
            Some(Duration::ZERO)
        };
        let affordable = affordable_sweep(
            self.ledger.remaining(),
            formation,
            remaining
                .iter()
                .map(|_| self.points[point].cost.admission()),
            self.points[point].confirmation_cost.admission(),
        );
        // A bounded deterministic prefix obtains evidence; subsequent enumeration
        // requires observed affordability and is reconsidered after every visit.
        let enumerate = affordable || self.points[point].attempted.len() < 2;
        #[cfg(test)]
        let enumerate = enumerate || matches!(self.strategy, TestStrategy::Enumeration);
        if enumerate {
            Ok(remaining.first().map(|coordinate| (*coordinate).clone()))
        } else {
            Ok(None)
        }
    }

    fn next_refinement_point(&mut self) -> Option<usize> {
        for point in &mut self.points {
            if !self.ledger.allows(point.next_stage_cost()) {
                point.refinement_deferred = true;
            }
        }
        self.refinement_round
            .retain(|index| !self.points[*index].refinement_deferred);
        if self.refinement_round.is_empty() {
            let oldest = self
                .points
                .iter()
                .filter(|point| !point.refinement_deferred)
                .map(|point| point.refinement_visits)
                .min()?;
            self.refinement_round = self
                .points
                .iter()
                .enumerate()
                .filter(|(_, point)| {
                    !point.refinement_deferred && point.refinement_visits == oldest
                })
                .map(|(index, _)| index)
                .take(8)
                .collect();
        }
        let priority = |index: usize| {
            let point = &self.points[index];
            let class = if point.comparison.is_some() {
                0
            } else if !point.pending.is_empty() {
                1
            } else {
                2
            };
            let cost = if point.comparison.is_some() || !point.pending.is_empty() {
                let stage = point
                    .comparison
                    .as_ref()
                    .map(|promotion| promotion.comparison.stage())
                    .unwrap_or(0);
                point
                    .confirmation_cost
                    .ordering()
                    .map(|cost| cost.saturating_mul(16u32.checked_shl(stage).unwrap_or(u32::MAX)))
            } else {
                point.cost.ordering()
            };
            let gain = point
                .screens
                .get(&point.incumbent)
                .map(|incumbent| {
                    point
                        .pending
                        .iter()
                        .filter_map(|candidate| point.screens.get(candidate))
                        .map(|score| (incumbent - score) / incumbent.max(f64::MIN_POSITIVE))
                        .fold(0.0_f64, f64::max)
                })
                .unwrap_or(0.0);
            let gain_per_cost = cost
                .map(|cost| gain / cost.as_secs_f64().max(f64::MIN_POSITIVE))
                .unwrap_or(0.0);
            (class, cost, gain_per_cost)
        };
        let selected = self
            .refinement_round
            .iter()
            .enumerate()
            .filter(|(_, point)| {
                !self.points[**point].refinement_deferred
                    && self.ledger.allows(self.points[**point].next_stage_cost())
            })
            .min_by(|(_, a), (_, b)| {
                #[cfg(test)]
                if !self.cost_ordering {
                    return a.cmp(b);
                }
                let (ac, at, ag) = priority(**a);
                let (bc, bt, bg) = priority(**b);
                ac.cmp(&bc)
                    // Unknown costs receive one bounded attempt in this round.
                    .then(at.is_some().cmp(&bt.is_some()))
                    .then_with(|| bg.total_cmp(&ag))
                    .then(at.cmp(&bt))
            })
            .map(|(ordinal, _)| ordinal)?;
        let point = self.refinement_round.remove(selected);
        self.points[point].refinement_visits =
            self.points[point].refinement_visits.saturating_add(1);
        Some(point)
    }

    fn refine<T: TargetFamily, C: NativeCompiler<T>>(
        &mut self,
        session: &mut EvaluationSession<'_, T, C>,
    ) -> Result<(), PreparationError>
    where
        O: ControlledObserver<T, C::Handle>,
    {
        let Some(point) = self.next_refinement_point() else {
            return Ok(());
        };
        if !self.points[point].screens.contains_key(&0) {
            return self.screen(session, point, 0);
        }
        if self.points[point].comparison.is_some() || !self.points[point].pending.is_empty() {
            if !self.points[point].refinement_deferred {
                return self.confirm(session, point);
            }
            return Ok(());
        }
        if let Some(coordinate) = self.enumerated_candidate(session, point)? {
            if let Some(candidate) = self.add_candidate(coordinate) {
                self.report.enumerated_screens += 1;
                return self.screen(session, point, candidate);
            }
        }
        // Resident transfer is one sparse edge, never a point-by-candidate product.
        let resident = self
            .candidates
            .iter()
            .enumerate()
            .find(|(index, candidate)| {
                candidate.executable.is_some()
                    && !self.points[point].attempted.contains(index)
                    && !self.points[point].deferred.contains(index)
            })
            .map(|(index, _)| index);
        if let Some(candidate) = resident {
            return self.screen(session, point, candidate);
        }
        let parent = self.points[point]
            .screens
            .iter()
            .min_by(|(a, x), (b, y)| x.total_cmp(y).then(a.cmp(b)))
            .map(|(index, _)| *index)
            .unwrap_or(0);
        let mut donors = self.points[point]
            .screens
            .keys()
            .copied()
            .filter(|index| {
                *index != parent
                    && self.candidates[*index].coordinate.family()
                        == self.candidates[parent].coordinate.family()
            })
            .collect::<Vec<_>>();
        donors.sort_unstable();
        let donor = (!donors.is_empty()).then(|| donors[self.random.index(donors.len())]);
        let mutation = if (self.next_operator / 4) % 2 == 0 {
            Operator::Mutate
        } else {
            Operator::Compound
        };
        let operator = [
            mutation,
            Operator::Crossover,
            Operator::Mutate,
            Operator::Fresh,
        ][self.next_operator % 4];
        self.next_operator += 1;
        let operator = if matches!(operator, Operator::Crossover) && donor.is_none() {
            Operator::Mutate
        } else {
            operator
        };
        #[cfg(test)]
        let operator = if matches!(self.strategy, TestStrategy::Fresh) {
            Operator::Fresh
        } else {
            operator
        };
        if let Some(coordinate) = propose(
            session.domain(),
            Some(&self.candidates[parent].coordinate),
            donor.map(|i| &self.candidates[i].coordinate),
            operator,
            &mut self.random,
        ) {
            if let Some(candidate) = self.add_candidate(coordinate) {
                self.screen(session, point, candidate)?;
            }
        }
        Ok(())
    }

    fn event(&mut self, point: usize, case: usize) -> Comparison {
        self.next_event += 1;
        Comparison::new(
            self.next_event,
            self.points[point].cases[case].seed,
            self.environment
                .expect("confirmation requires a measured environment"),
        )
    }

    fn confirm<T: TargetFamily, C: NativeCompiler<T>>(
        &mut self,
        session: &mut EvaluationSession<'_, T, C>,
        point: usize,
    ) -> Result<(), PreparationError>
    where
        O: ControlledObserver<T, C::Handle>,
    {
        let mut promotion = if let Some(promotion) = self.points[point].comparison.take() {
            promotion
        } else {
            let challenger = self.points[point].pending.pop_front().unwrap();
            Promotion {
                incumbent: self.points[point].incumbent,
                challenger,
                case: 0,
                comparison: self.event(point, 0),
            }
        };
        debug_assert_eq!(
            promotion.comparison.seed(),
            self.points[point].cases[promotion.case].seed
        );
        if promotion.comparison.stage_complete() {
            promotion.comparison.advance();
        }
        while self
            .ledger
            .allows(self.points[point].confirmation_cost.admission())
        {
            let Some(arm) = promotion.comparison.next(&mut self.random) else {
                break;
            };
            let candidate = match arm {
                Arm::Incumbent => promotion.incumbent,
                Arm::Challenger => promotion.challenger,
            };
            let environment = self.environment;
            let observation = match self.observe(
                session,
                point,
                candidate,
                promotion.case,
                ObservationProtocol {
                    warmup: 1,
                    trials: 1,
                },
            )? {
                ExperimentOutcome::Measured(observation) => observation,
                ExperimentOutcome::Inapplicable => return Ok(()),
                ExperimentOutcome::Deferred => {
                    self.points[point].refinement_deferred = true;
                    self.points[point].comparison = Some(promotion);
                    return Ok(());
                }
            };
            if environment != self.environment {
                return Ok(());
            }
            promotion
                .comparison
                .record(observation.samples[0], observation.environment);
        }
        match promotion.comparison.verdict() {
            Verdict::Improved if promotion.case == 0 => {
                promotion.case = 1;
                promotion.comparison = self.event(point, 1);
                self.points[point].comparison = Some(promotion);
            }
            Verdict::Improved => {
                self.points[point].incumbent = promotion.challenger;
                self.points[point].pending.clear();
                self.points[point].retain_donors();
                for other in &self.points {
                    if other.incumbent != promotion.challenger {
                        self.navigator
                            .prioritize_between(&self.points[point].point, &other.point);
                    }
                }
            }
            Verdict::Pending => {
                self.points[point].comparison = Some(promotion);
            }
            Verdict::Inconclusive => self.report.inconclusive_comparisons += 1,
            Verdict::Inferior | Verdict::EnvironmentChanged => {}
        }
        Ok(())
    }

    fn publish<T: TargetFamily, C: NativeCompiler<T>>(
        &mut self,
        session: &mut EvaluationSession<'_, T, C>,
    ) -> Result<(SelectionPolicy, EvaluationIdentity, PlanningReport), PreparationError> {
        let started = Instant::now();
        let mut retained = vec![0];
        let mut cases = Vec::new();
        for population in &self.points {
            if population.incumbent == 0 {
                continue;
            }
            let Some(predicate) = self
                .navigator
                .predicate(&mut session.domain_mut().arena_mut(), &population.point)
            else {
                continue;
            };
            let ordinal = retained
                .iter()
                .position(|candidate| *candidate == population.incumbent)
                .unwrap_or_else(|| {
                    retained.push(population.incumbent);
                    retained.len() - 1
                });
            cases.push((
                session.domain().arena().compile_bool(predicate),
                CandidateIndex::from_usize(ordinal),
            ));
        }
        let policy = loop {
            let requests = retained
                .iter()
                .map(|index| {
                    (
                        self.candidates[*index]
                            .executable
                            .expect("only prepared winners are retained"),
                        (),
                    )
                })
                .collect();
            match session.policy(NonEmpty::new(requests).unwrap(), |_, candidates| {
                SelectionFunction::ordered_decision(candidates, cases.clone())
            }) {
                Ok(policy) => break policy,
                Err(PreparationError::Planning(_)) if retained.len() > 1 => {
                    retained.pop();
                    cases.retain(|(_, index)| index.as_usize() < retained.len());
                }
                Err(error) => return Err(error),
            }
        };
        let mut evidence = self.evidence.clone();
        evidence.update(session.domain().module().digest());
        evidence.update(session.domain().entry().digest());
        evidence.update(super::EvaluationMethod::Feedback(self.options.clone()).fingerprint());
        evidence.update((self.points.len() as u64).to_le_bytes());
        for population in &self.points {
            if retained.contains(&population.incumbent) {
                evidence.update((population.point.0.len() as u64).to_le_bytes());
                for value in &population.point.0 {
                    evidence.update(value.to_le_bytes());
                }
                let coordinate = &self.candidates[population.incumbent].coordinate;
                evidence.update(coordinate.family().digest());
                evidence.update((coordinate.choices().len() as u64).to_le_bytes());
                for (_, choice) in coordinate.choices() {
                    evidence.update(choice.to_le_bytes());
                }
            }
        }
        let protocol = Sha256::digest(b"seismic-feedback-v2-applicable-only-resident-submission-completion-two-seeds-median-delta-0.05-epsilon-0").into();
        let identity = EvaluationIdentity::new(
            session.domain().device_identity().clone(),
            EvaluationProvenance::new(protocol, evidence.finalize().into()),
        );
        self.report.evaluation_fingerprint = identity.evidence_fingerprint();
        self.report.measurement_environment = self.environment;
        let report = PlanningReport {
            optimization: OptimizationCompletion::Limited(PlanningLimit::FeedbackSearch),
            budget: PlanningBudgetReport::default(),
        };
        self.ledger.published(started.elapsed());
        let (native_code_bytes, native_metadata_bytes, native_budget_exhausted) =
            session.native_usage();
        let (retained_native_code_bytes, retained_native_metadata_bytes) =
            session.retained_native_sizes(&policy)?;
        self.report = FeedbackReport {
            native_code_bytes,
            native_metadata_bytes,
            retained_native_code_bytes,
            retained_native_metadata_bytes,
            native_budget_exhausted,
            elapsed: self.ledger.elapsed(),
            costs: self.ledger.costs.clone(),
            attempted_points: self.points.len(),
            measured_points: self
                .points
                .iter()
                .filter(|point| !point.screens.is_empty())
                .count(),
            prepared_candidates: self
                .candidates
                .iter()
                .filter(|candidate| candidate.executable.is_some())
                .count(),
            confirmation_events: self.next_event,
            construction_pending_paths: self.construction.pending_paths(),
            initialization_pending_paths: self.construction.initialization_pending_paths(),
            confirmed_points: cases.len(),
            unresolved_witness_searches: self.navigator.unresolved,
            ..self.report.clone()
        };
        Ok((policy, identity, report))
    }
}

impl<T: TargetFamily, C: NativeCompiler<T>, O: ControlledObserver<T, C::Handle>>
    CandidateEvaluator<T, C> for FeedbackEvaluator<O>
{
    fn evaluate(
        &mut self,
        session: &mut EvaluationSession<'_, T, C>,
    ) -> Result<(SelectionPolicy, EvaluationIdentity, PlanningReport), PreparationError> {
        let environment = self
            .observer
            .environment()
            .map_err(|error| PreparationError::Feedback(FeedbackError::Observation(error)))?;
        self.accept_environment(environment);
        if self.candidates.is_empty() {
            let coordinate = session
                .domain()
                .canonicalize(session.domain().universal_proposal())
                .expect("universal coordinate is canonical");
            self.add_candidate(coordinate);
            if self.prepare(session, 0)?.is_none() {
                return Err(PreparationError::UniversalClosure(
                    "feedback could not establish its general implementation".into(),
                ));
            }
            // Establish an immediately publishable incumbent and measure its reserve.
            self.publish(session)?;
        }
        while self.ledger.allows(None) {
            #[cfg(test)]
            if self.observation_policy.exhausted {
                break;
            }
            let discovery = !self.discovery_window.is_empty()
                || (!self.navigator.exhausted()
                    && !self.navigator.resource_limited
                    && self.points.len() < self.options.archive_points)
                || self.candidates.len() < self.options.archive_candidates
                || (self.construction.has_work()
                    && self.construction_allowance.work_units > 0
                    && self.construction_allowance.wall_time > Duration::ZERO);
            let refinement = self.points.iter().any(|point| {
                !point.refinement_deferred && self.ledger.allows(point.next_stage_cost())
            });
            let Some(lane) = self.ledger.next_lane(discovery, refinement) else {
                break;
            };
            let started = Instant::now();
            match lane {
                Lane::Discovery => {
                    if self.construction.has_work() && self.construction_allowance.work_units > 0 {
                        let mut slice = crate::candidate_domain::ConstructionAllowance {
                            work_units: self.construction_allowance.work_units.min(64),
                            wall_time: self
                                .construction_allowance
                                .wall_time
                                .min(Duration::from_millis(2)),
                        };
                        let before = slice;
                        self.construction.advance(session.domain_mut(), &mut slice);
                        let used = before.work_units - slice.work_units;
                        self.construction_allowance.work_units -= used;
                        self.report.construction_work_units += used;
                        self.construction_allowance.wall_time = self
                            .construction_allowance
                            .wall_time
                            .saturating_sub(before.wall_time - slice.wall_time);
                        self.finite_candidates = self
                            .construction
                            .complete()
                            .then(|| {
                                finite_coordinates(
                                    session.domain(),
                                    self.options.archive_candidates.min(4096),
                                )
                            })
                            .flatten();
                    }
                    self.discover(session)?;
                }
                Lane::Refinement => self.refine(session)?,
            }
            self.ledger.charge(lane, started.elapsed());
        }
        let policy = self.publish(session)?;
        Ok(policy)
    }
}

#[cfg(test)]
mod tests {
    use super::super::{InvocationParameter, InvocationScope};
    use super::*;
    use crate::evaluation_session::boundary_tests::domain_with_optional;
    use crate::realization::demand_driven_tests::registry;
    use crate::preparation_budget::{PlanningBudget, PreparationBudget};
    use crate::realization::demand_driven_tests::{device, CountingCompiler, FakeTarget};
    use seismic_lang::expr::{compiled::InvocationValues, SymbolValue};

    #[derive(Default)]
    struct Observer {
        general: Option<crate::implementation::ImplementationIdentity>,
        calls: usize,
        defer_next: bool,
        next_error: Option<ObservationError>,
        slow_seed: Option<u64>,
        environment: u8,
    }
    impl<H> ControlledObserver<FakeTarget, H> for Observer {
        fn environment(&mut self) -> Result<[u8; 32], ObservationError> {
            Ok([self.environment.max(1); 32])
        }
        fn observe(
            &mut self,
            request: ObservationRequest<'_, FakeTarget, H>,
        ) -> Result<Observation, ObservationError> {
            self.calls += 1;
            if let Some(error) = self.next_error.take() {
                return Err(error);
            }
            if std::mem::take(&mut self.defer_next) {
                return Err(ObservationError::Capacity {
                    required: 2u64.into(),
                    limit: 1,
                });
            }
            let identity = &request.executable.identity().implementation;
            let general = self.general.get_or_insert_with(|| identity.clone());
            Ok(Observation {
                support: Default::default(),
                samples: vec![
                    Duration::from_nanos(if identity == general {
                        100
                    } else if self.slow_seed == Some(request.case.seed) {
                        200
                    } else {
                        10
                    });
                    request.protocol.trials
                ],
                setup_time: Duration::ZERO,
                checking_time: Duration::ZERO,
                environment: [self.environment.max(1); 32],
            })
        }
    }

    fn population(index: usize) -> PointPopulation {
        PointPopulation {
            point: Point(vec![index as u64]),
            cases: std::array::from_fn(|seed| ObservationCase {
                values: InvocationValues::new(),
                arguments: Vec::new(),
                seed: seed as u64,
            }),
            incumbent: 0,
            screens: HashMap::new(),
            attempted: HashSet::new(),
            deferred: HashSet::new(),
            refinement_deferred: false,
            refinement_visits: 0,
            pending: VecDeque::new(),
            comparison: None,
            cost: ObservedCost::default(),
            confirmation_cost: ObservedCost::default(),
        }
    }

    /// Equal wall-time checkpoints exercise the delivered policy, not a later
    /// remeasurement. This small warm-cache fixture establishes the harness;
    /// it does not establish superiority on large coupled candidate domains.
    #[test]
    fn equal_budget_controller_qualification_reports_delivered_policies() {
        struct SurfaceObserver {
            candidates: HashMap<crate::evaluation_session::PreparedCandidateId, usize>,
        }
        fn latency(candidate: usize, point: usize) -> u64 {
            match candidate {
                0 => 1000,
                1 if point <= 1 || point >= 7 => 100,
                2 if (3..=5).contains(&point) => 80,
                _ => 1400,
            }
        }
        impl<H> ControlledObserver<FakeTarget, H> for SurfaceObserver {
            fn environment(&mut self) -> Result<[u8; 32], ObservationError> {
                Ok([9; 32])
            }
            fn observe(
                &mut self,
                request: ObservationRequest<'_, FakeTarget, H>,
            ) -> Result<Observation, ObservationError> {
                let value = match request.case.arguments[0] {
                    super::super::CaseArgument::Scalar(SymbolValue::F32(value)) => value,
                    _ => panic!("scalar fixture"),
                };
                let point = value.to_bits().saturating_sub(1.0f32.to_bits()) as usize;
                // Discovery can materialize a third native family after this
                // fixture preloads its two initial implementations.
                let candidate = self
                    .candidates
                    .get(&request.candidate)
                    .copied()
                    .unwrap_or(2);
                // Heterogeneous complete-observation work is charged to the real
                // ledger. Returned kernel timings define the known truth surface.
                std::thread::sleep(Duration::from_micros(if point % 3 == 0 { 150 } else { 40 }));
                Ok(Observation {
                    support: Default::default(),
                    samples: vec![
                        Duration::from_nanos(latency(candidate, point));
                        request.protocol.trials
                    ],
                    setup_time: Duration::ZERO,
                    checking_time: Duration::ZERO,
                    environment: [9; 32],
                })
            }
        }
        for seed in [1, 7, 29] {
            for (strategy, cost_ordering) in [
                (TestStrategy::Evolution, true),
                (TestStrategy::Fresh, true),
                (TestStrategy::Enumeration, true),
                (TestStrategy::Evolution, false),
            ] {
                let device = device();
                let registry = registry();
                let compiler = CountingCompiler::new();
                let (domain, _) = domain_with_optional(&device, &registry);
                let mut scope = InvocationScope::for_entry(domain.entry());
                scope.constrain(
                    InvocationParameter::Scalar(0),
                    SymbolValue::F32(1.0),
                    SymbolValue::F32(f32::from_bits(1.0f32.to_bits() + 8)),
                );
                let symbol = match domain.schema().parameters()[0].kind {
                    seismic_lang::entry::ParameterKind::Scalar { symbol, .. } => symbol,
                    _ => panic!(),
                };
                let mut session = EvaluationSession::from_domain(
                    domain,
                    &registry,
                    &compiler,
                    &(),
                    &device,
                    &PreparationBudget::default(),
                    &PlanningBudget::default(),
                )
                .unwrap();
                let coordinates = finite_coordinates(session.domain(), 8).unwrap();
                let mut candidates = HashMap::new();
                for (index, coordinate) in coordinates.iter().enumerate() {
                    let RealizationAdmission::Ready(id) =
                        session.realize_checked(coordinate).unwrap()
                    else {
                        panic!()
                    };
                    candidates.insert(id, index);
                }
                let options = FeedbackOptions {
                    seed,
                    search_time: Duration::from_millis(80),
                    optimize_for: Some(scope),
                    ..Default::default()
                };
                let mut evaluator = FeedbackEvaluator::new(
                    &session,
                    SurfaceObserver { candidates },
                    options,
                    Instant::now(),
                )
                .unwrap();
                evaluator.strategy = strategy;
                evaluator.cost_ordering = cost_ordering;
                for checkpoint in [80, 160] {
                    let policy = evaluator.evaluate(&mut session).unwrap();
                    let kernel = session.finalize(policy.0, policy.1, policy.2).unwrap();
                    let mut latencies = Vec::new();
                    for point in 0..9 {
                        let mut values = InvocationValues::new();
                        values.bind(
                            symbol,
                            SymbolValue::F32(f32::from_bits(1.0f32.to_bits() + point as u32)),
                        );
                        let selected = kernel.select(&values).as_usize();
                        let selected_variant = kernel.variants().iter().nth(selected).unwrap();
                        let candidate = evaluator
                            .observer
                            .candidates
                            .iter()
                            .find_map(|(id, index)| {
                                (session.executable(*id).ok()?.identity()
                                    == selected_variant.identity())
                                .then_some(*index)
                            })
                            .unwrap_or(2);
                        let measured = latency(candidate, point);
                        assert!(
                            measured <= latency(0, point),
                            "confirmed policy regressed on known surface"
                        );
                        latencies.push(measured);
                    }
                    println!(
                        "feedback-qualification seed={seed} strategy={strategy:?} cost_ordering={cost_ordering} cache=warm checkpoint_ms={checkpoint} elapsed_us={} confirmed={} measured={} native_forms={} point_latencies_ns={latencies:?}",
                        evaluator.report.elapsed.as_micros(),
                        evaluator.report.confirmed_points,
                        evaluator.report.measured_points,
                        compiler.form_count()
                    );
                    evaluator.suspend();
                    if checkpoint == 80 {
                        evaluator.resume(Duration::from_millis(80));
                    }
                }
            }
        }
    }

    #[test]
    fn point_archive_keeps_incumbent_and_active_comparison_with_bounded_donors() {
        let mut point = population(0);
        point.incumbent = 18;
        point.comparison = Some(Promotion {
            incumbent: 18,
            challenger: 19,
            case: 0,
            comparison: Comparison::new(1, 0, [1; 32]),
        });
        for candidate in 0..20 {
            point.screens.insert(candidate, candidate as f64);
            point.pending.push_back(candidate);
        }
        point.retain_donors();
        assert_eq!(point.screens.len(), 8);
        for protected in [0, 18, 19] {
            assert!(point.screens.contains_key(&protected));
        }
        assert!(point
            .pending
            .iter()
            .all(|candidate| point.screens.contains_key(candidate)));
    }

    #[test]
    fn costly_old_round_does_not_block_later_feasible_work_and_resumes() {
        let device = device();
        let registry = registry();
        let compiler = CountingCompiler::new();
        let (domain, _) = domain_with_optional(&device, &registry);
        let session = EvaluationSession::from_domain(
            domain,
            &registry,
            &compiler,
            &(),
            &device,
            &PreparationBudget::default(),
            &PlanningBudget::default(),
        )
        .unwrap();
        let mut evaluator = FeedbackEvaluator::new(
            &session,
            Observer::default(),
            FeedbackOptions::default(),
            Instant::now(),
        )
        .unwrap();
        evaluator.points = (0..10).map(population).collect();
        for point in &mut evaluator.points[..8] {
            point.cost.record(Duration::from_secs(100));
        }
        evaluator.refinement_round = (0..8).collect();
        assert_eq!(evaluator.next_refinement_point(), Some(8));
        assert_eq!(evaluator.next_refinement_point(), Some(9));
        assert!(evaluator.points[..8]
            .iter()
            .all(|point| point.refinement_deferred));
        evaluator.suspend();
        evaluator.resume(Duration::from_secs(1000));
        assert_eq!(evaluator.next_refinement_point(), Some(0));
    }

    #[test]
    fn observation_deadline_defers_but_contract_and_numerical_defects_abort() {
        let device = device();
        let registry = registry();
        let compiler = CountingCompiler::new();
        let (domain, optional) = domain_with_optional(&device, &registry);
        let mut scope = InvocationScope::for_entry(domain.entry());
        scope.constrain(
            InvocationParameter::Scalar(0),
            SymbolValue::F32(1.0),
            SymbolValue::F32(1.0),
        );
        let mut session = EvaluationSession::from_domain(
            domain,
            &registry,
            &compiler,
            &(),
            &device,
            &PreparationBudget::default(),
            &PlanningBudget::default(),
        )
        .unwrap();
        let mut evaluator = FeedbackEvaluator::new(
            &session,
            Observer::default(),
            FeedbackOptions {
                optimize_for: Some(scope),
                search_time: Duration::from_secs(2),
                ..Default::default()
            },
            Instant::now(),
        )
        .unwrap();
        let general = session
            .domain()
            .canonicalize(session.domain().universal_proposal())
            .unwrap();
        evaluator.add_candidate(general);
        evaluator.prepare(&mut session, 0).unwrap();
        evaluator.discover(&mut session).unwrap();
        evaluator.observer.next_error = Some(ObservationError::DeadlineExpired);
        assert!(matches!(
            evaluator.observe(&mut session, 0, 0, 0, ObservationProtocol::SCREEN),
            Ok(ExperimentOutcome::Deferred)
        ));
        evaluator.observer.next_error = Some(ObservationError::Contract("broken outcome".into()));
        assert!(matches!(
            evaluator.observe(&mut session, 0, 0, 0, ObservationProtocol::SCREEN),
            Err(PreparationError::Feedback(FeedbackError::Observation(
                ObservationError::Contract(reason)
            ))) if reason == "broken outcome"
        ));
        let optional = evaluator.add_candidate(optional).unwrap();
        evaluator.observer.next_error = Some(ObservationError::IncorrectResult(
            "wrong final state".into(),
        ));
        assert!(matches!(
            evaluator.observe(&mut session, 0, optional, 0, ObservationProtocol::SCREEN),
            Err(PreparationError::Feedback(FeedbackError::Observation(
                ObservationError::IncorrectResult(reason)
            ))) if reason == "wrong final state"
        ));
        assert!(
            !evaluator.candidates[optional].rejected,
            "a compiler/backend defect must not be hidden as ordinary candidate rejection"
        );
    }

    #[test]
    fn feedback_promotes_only_after_both_cases_and_dispatches_only_the_measured_point() {
        let device = device();
        let registry = registry();
        let compiler = CountingCompiler::new();
        let (domain, optional) = domain_with_optional(&device, &registry);
        let mut scope = InvocationScope::for_entry(domain.entry());
        scope.constrain(
            InvocationParameter::Scalar(0),
            SymbolValue::F32(1.0),
            SymbolValue::F32(1.0),
        );
        let symbol = match domain.schema().parameters()[0].kind {
            seismic_lang::entry::ParameterKind::Scalar { symbol, .. } => symbol,
            _ => panic!(),
        };
        let mut session = EvaluationSession::from_domain(
            domain,
            &registry,
            &compiler,
            &(),
            &device,
            &PreparationBudget::default(),
            &PlanningBudget::default(),
        )
        .unwrap();
        let options = FeedbackOptions {
            optimize_for: Some(scope),
            search_time: Duration::from_secs(2),
            ..Default::default()
        };
        let mut evaluator =
            FeedbackEvaluator::new(&session, Observer::default(), options, Instant::now()).unwrap();
        let general = session
            .domain()
            .canonicalize(session.domain().universal_proposal())
            .unwrap();
        evaluator.add_candidate(general);
        evaluator.prepare(&mut session, 0).unwrap();
        evaluator.observer.defer_next = true;
        evaluator.discover(&mut session).unwrap();
        assert!(evaluator.points[0].deferred.contains(&0));
        assert!(!evaluator.points[0].attempted.contains(&0));
        let calls = evaluator.observer.calls;
        evaluator.screen(&mut session, 0, 0).unwrap();
        assert_eq!(
            evaluator.observer.calls, calls,
            "deferred work waits for continuation"
        );
        evaluator.suspend();
        evaluator.resume(Duration::from_secs(1));
        evaluator.screen(&mut session, 0, 0).unwrap();
        assert!(evaluator.points[0].screens.contains_key(&0));
        let optional = evaluator.add_candidate(optional).unwrap();
        evaluator.screen(&mut session, 0, optional).unwrap();
        assert_eq!(
            evaluator.points[0].incumbent, 0,
            "screening must not promote"
        );
        evaluator.confirm(&mut session, 0).unwrap();
        evaluator.confirm(&mut session, 0).unwrap();
        assert_eq!(
            evaluator.points[0].incumbent, 0,
            "first seed alone must not promote"
        );
        let second_seed = evaluator.points[0].cases[1].seed;
        assert_ne!(evaluator.points[0].cases[0].seed, second_seed);
        assert_eq!(
            evaluator.points[0]
                .comparison
                .as_ref()
                .unwrap()
                .comparison
                .seed(),
            second_seed
        );
        evaluator.observer.slow_seed = Some(second_seed);
        evaluator.confirm(&mut session, 0).unwrap();
        evaluator.confirm(&mut session, 0).unwrap();
        assert_eq!(
            evaluator.points[0].incumbent, 0,
            "a loss on the second content case defeats promotion"
        );
        assert!(evaluator.points[0].comparison.is_none());
        assert_eq!(evaluator.next_event, 2);
        // A new nomination cannot reuse the first event's winning samples.
        evaluator.observer.slow_seed = None;
        evaluator.points[0].pending.push_back(optional);
        evaluator.confirm(&mut session, 0).unwrap();
        assert_eq!(evaluator.next_event, 3);
        assert_eq!(evaluator.points[0].incumbent, 0);
        evaluator.confirm(&mut session, 0).unwrap();
        assert_eq!(evaluator.next_event, 4);
        // Changing environment between cases discards both the frozen
        // comparison and its nomination; event allowance is never reset.
        evaluator.accept_environment([2; 32]);
        evaluator.observer.environment = 2;
        assert!(evaluator.points[0].comparison.is_none());
        assert!(evaluator.points[0].pending.is_empty());
        assert_eq!(evaluator.points[0].incumbent, 0);
        assert_eq!(evaluator.next_event, 4);
        evaluator.screen(&mut session, 0, 0).unwrap();
        evaluator.screen(&mut session, 0, optional).unwrap();
        for _ in 0..4 {
            evaluator.confirm(&mut session, 0).unwrap();
        }
        assert_eq!(evaluator.points[0].incumbent, optional);
        let policy = evaluator.publish(&mut session).unwrap();
        let kernel = session.finalize(policy.0, policy.1, policy.2).unwrap();
        let mut values = InvocationValues::new();
        values.bind(symbol, SymbolValue::F32(1.0));
        assert_eq!(kernel.select(&values).as_usize(), 1);
        values.bind(symbol, SymbolValue::F32(2.0));
        assert_eq!(kernel.select(&values).as_usize(), 0);
        assert_eq!(evaluator.report().confirmation_events, 6);
    }

    #[test]
    fn zero_optional_allowance_returns_general_and_explicit_resume_keeps_candidates() {
        let device = device();
        let registry = registry();
        let compiler = CountingCompiler::new();
        let (domain, _) = domain_with_optional(&device, &registry);
        let mut session = EvaluationSession::from_domain(
            domain,
            &registry,
            &compiler,
            &(),
            &device,
            &PreparationBudget::default(),
            &PlanningBudget::default(),
        )
        .unwrap();
        let options = FeedbackOptions {
            search_time: Duration::ZERO,
            ..Default::default()
        };
        let mut evaluator =
            FeedbackEvaluator::new(&session, Observer::default(), options, Instant::now()).unwrap();
        let policy =
            <FeedbackEvaluator<_> as CandidateEvaluator<FakeTarget, CountingCompiler>>::evaluate(
                &mut evaluator,
                &mut session,
            )
            .unwrap();
        let first = session.finalize(policy.0, policy.1, policy.2).unwrap();
        let formations = compiler.form_count();
        evaluator.suspend();
        evaluator.resume(Duration::ZERO);
        let policy =
            <FeedbackEvaluator<_> as CandidateEvaluator<FakeTarget, CountingCompiler>>::evaluate(
                &mut evaluator,
                &mut session,
            )
            .unwrap();
        let second = session.finalize(policy.0, policy.1, policy.2).unwrap();
        assert_eq!(first.variants().len(), 1);
        assert_eq!(second.variants().len(), 1);
        assert_eq!(compiler.form_count(), formations);
        assert_eq!(evaluator.observer.calls, 0);
    }
    include!("controller_qualification.rs");
}
