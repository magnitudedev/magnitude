//! Scheduling a unit's numerical observations: when the native default and
//! a candidate run at a point, what those runs are charged to, and which
//! observations already made stand for a candidate's. Capturing a call is
//! [`super::observation`], measuring two captures against each other
//! [`super::comparison`], and judging the measurements
//! [`super::admission`].
//!
//! This is empirical validation of direct-native entries against the
//! declaration's native default. It does not establish compiler
//! applicability for a native implementation.
use super::NativePrepared;
use super::admission::{NumericalObservation, NumericalRequirement, admit};
use super::comparison::{COMPARISON_VERSION, SubjectMeasurement, compare};
use super::deadline::{Deadline, WorkError};
use super::observation::{Observation, ReferencePages, StateRegion, Subject, observe, subjects};
use super::search::PointKey;
use super::timing::{PointPrices, PointTiming};
use super::tune::{Exclusion, TuneError, TuningInitializer, TuningPoint};
use crate::api::{
    device::DeviceInner,
    kernel::{DecodedValue, EncodedArgs},
};
use seismic_lang::entry::LogicalEntry;
use sha2::{Digest, Sha256};
use std::{
    cell::{Cell, RefCell},
    collections::HashMap,
    rc::Rc,
    sync::Arc,
    time::{Duration, Instant},
};

pub(super) type Initializer<'a> = Rc<RefCell<TuningInitializer<'a>>>;

/// One point of a unit, prepared for observation. What identifies it and
/// what was learned at it (its label, its demand, the prices of its
/// operation classes, the unit's evidence) stays for the load. Its inputs
/// do not: they are resident only while work of the unit uses them
/// ([`PreparedPoint::release`], [`Validator::restore`]).
pub(super) struct PreparedPoint<'a> {
    pub label: String,
    /// Invocation demand for this workload in the search's objective.
    pub weight: f64,
    pub class: Option<String>,
    shared: Rc<Shared>,
    prices: Rc<Cell<PointPrices>>,
    comparing_seconds: Cell<f64>,
    inputs: Option<Inputs<'a>>,
}

/// A point's built inputs and what is derived from them alone: the
/// default's observation at each witness rotation.
struct Inputs<'a> {
    rotation: Vec<EncodedArgs>,
    workload: Vec<usize>,
    initialize: Option<Initializer<'a>>,
    states: Vec<StateRegion>,
    subjects: Vec<Subject>,
    reference: RefCell<Vec<Option<Vec<Observation>>>>,
}

/// Every observation a unit has made of its candidates, each made once.
///
/// What a candidate publishes at a point is determined by the work it runs
/// there (the launches active at the point and the values of the parameters
/// they read: its [`PointKey`]), the point's inputs, the default it is
/// measured against and the comparison. The last three are fixed for a
/// point of a unit, so an observation is keyed by point and key, and a
/// candidate whose key at a point was observed before takes that
/// observation.
#[derive(Default)]
struct Evidence {
    /// In the order made.
    observations: Vec<NumericalObservation>,
    by_key: HashMap<(String, PointKey), usize>,
    /// The observations each artifact was held to, in the order reached.
    by_artifact: HashMap<String, Vec<usize>>,
}

impl Evidence {
    fn attribute(&mut self, artifact: &str, observation: usize) {
        let held = self.by_artifact.entry(artifact.to_owned()).or_default();
        if !held.contains(&observation) {
            held.push(observation);
        }
    }
}

/// What every point of a unit shares: the default, what candidates are
/// held to against it, and what was observed so far.
struct Shared {
    /// The declaration's native default: the canonical kernel itself.
    default: Arc<NativePrepared>,
    requirement: NumericalRequirement,
    logical: Arc<LogicalEntry>,
    pages: RefCell<ReferencePages>,
    /// Time spent executing the default as the reference.
    seconds: RefCell<f64>,
    evidence: RefCell<Evidence>,
}

impl<'a> PreparedPoint<'a> {
    fn inputs(&self) -> &Inputs<'a> {
        self.inputs
            .as_ref()
            .expect("a point's inputs are resident while work uses them")
    }

    /// Whether the point's inputs are built and held.
    pub(super) fn resident(&self) -> bool {
        self.inputs.is_some()
    }

    /// Every argument set of the point: its numerical witnesses.
    pub(super) fn rotation(&self) -> &[EncodedArgs] {
        &self.inputs().rotation
    }

    /// The argument sets of the point's performance workload.
    pub(super) fn workload_rotation(&self) -> Vec<EncodedArgs> {
        let inputs = self.inputs();
        inputs
            .workload
            .iter()
            .map(|index| inputs.rotation[*index].clone())
            .collect()
    }

    /// Give up the point's inputs and the default's observations of them:
    /// its argument sets, its initializer and the reference pages nothing
    /// else shares. What was measured and observed here stands; work that
    /// needs the point again has its inputs built again
    /// ([`Validator::restore`]) and the default observed again where a
    /// candidate is yet to be compared.
    pub(super) fn release(&mut self) {
        self.inputs = None;
        self.shared.pages.borrow_mut().release_unshared();
    }

    /// Execute the default at the witness rotations it is not observed at
    /// in the point's resident inputs: it never runs twice at a rotation
    /// of inputs that stay.
    fn ensure_reference(&self, deadline: Deadline) -> Result<(), WorkError<Exclusion>> {
        let inputs = self.inputs();
        let missing = (0..inputs.rotation.len())
            .filter(|rotation| inputs.reference.borrow()[*rotation].is_none())
            .collect::<Vec<_>>();
        if missing.is_empty() {
            return Ok(());
        }
        let started = Instant::now();
        let shared = &self.shared;
        let execution = |error: &dyn std::fmt::Display| Exclusion::Execution(error.to_string());
        let result = (|| {
            for rotation in missing {
                deadline.check(Duration::ZERO)?;
                let args = &inputs.rotation[rotation];
                let witness = Instant::now();
                // The reset restores every written row before the default
                // and before each candidate, so neither sees the other's
                // state.
                if let Some(reset) = &inputs.initialize {
                    reset.borrow_mut()().map_err(|e| execution(&e))?;
                }
                deadline.check(Duration::ZERO)?;
                let results = shared
                    .default
                    .call(args.clone())
                    .map_err(|e| execution(&e))?;
                let expected = observe(
                    results.into_values(),
                    args,
                    &inputs.states,
                    Some(&mut *shared.pages.borrow_mut()),
                )?;
                inputs.reference.borrow_mut()[rotation] = Some(expected);
                let mut prices = self.prices.get();
                prices.witness.observe(witness.elapsed().as_secs_f64());
                self.prices.set(prices);
            }
            Ok(())
        })();
        *shared.seconds.borrow_mut() += started.elapsed().as_secs_f64();
        result
    }

    /// The census's run of the native default at this point: its first
    /// pass over the workload rotation is observed and retained as the
    /// reference there, so no candidate ever needs another default run at
    /// those witnesses.
    pub(super) fn retain_default(
        &self,
        timing: &mut PointTiming<'a>,
        deadline: Deadline,
    ) -> Result<(), WorkError<Exclusion>> {
        let shared = &self.shared;
        let inputs = self.inputs();
        if timing.prepared().artifact() != shared.default.artifact()
            || inputs
                .workload
                .iter()
                .all(|rotation| inputs.reference.borrow()[*rotation].is_some())
        {
            return Ok(());
        }
        timing.initialize = inputs.initialize.clone();
        let started = Instant::now();
        let workload = &inputs.workload;
        let result = timing.observe_first(deadline, |position, values, args| {
            let expected = observe(
                values,
                args,
                &inputs.states,
                Some(&mut *shared.pages.borrow_mut()),
            )?;
            inputs.reference.borrow_mut()[workload[position]].get_or_insert(expected);
            Ok(())
        });
        *shared.seconds.borrow_mut() += started.elapsed().as_secs_f64();
        result
    }

    /// Place a kernel's timing at this point: its operations are observed
    /// in the point's classes, and its state is restored.
    pub(super) fn price_before(
        &self,
        timing: &mut PointTiming<'a>,
        deadline: Deadline,
    ) -> Result<(), WorkError<Exclusion>> {
        deadline.check(Duration::ZERO)?;
        let inputs = self.inputs();
        timing.track(self.prices.clone());
        let began = Instant::now();
        timing.initialize = inputs.initialize.clone();
        if let Some(reset) = &inputs.initialize {
            reset.borrow_mut()().map_err(|error| Exclusion::Execution(error.to_string()))?;
        }
        timing.placed(began.elapsed());
        Ok(())
    }

    /// `rotations` of `kernel` placed for observation alone.
    fn witnesses(
        &self,
        kernel: &Arc<NativePrepared>,
        rotations: &[usize],
    ) -> Result<PointTiming<'a>, Exclusion> {
        let inputs = self.inputs();
        let mut witnesses = PointTiming::new(
            kernel,
            rotations
                .iter()
                .map(|rotation| inputs.rotation[*rotation].clone())
                .collect(),
        )
        .map_err(|error| Exclusion::Execution(error.to_string()))?;
        witnesses.track(self.prices.clone());
        witnesses.initialize = inputs.initialize.clone();
        Ok(witnesses)
    }

    /// Measure `kernel` against the retained native default at every
    /// witness of this point, unless its work here was observed before:
    /// `key` gives the key of a candidate whose working launches here are
    /// those given. The index of its observation among the unit's; none
    /// for the default itself, which is the reference and is measured
    /// against nothing. No verdict is reached.
    ///
    /// With the candidate's timing placement, the observation is that
    /// placement's first pass over the workload rotation. Witnesses outside
    /// the workload rotation run in a placement of their own.
    pub(super) fn observe_before(
        &self,
        mut timing: Option<&mut PointTiming<'a>>,
        kernel: &Arc<NativePrepared>,
        key: &dyn Fn(Vec<usize>) -> PointKey,
        deadline: Deadline,
    ) -> Result<Option<usize>, WorkError<Exclusion>> {
        if let Some(timing) = timing.as_deref_mut() {
            self.price_before(timing, deadline)?;
        }
        let shared = &self.shared;
        let inputs = self.inputs();
        let artifact = &kernel.artifact().0;
        if *artifact == shared.default.artifact().0 {
            return Ok(None);
        }
        let all = (0..inputs.rotation.len()).collect::<Vec<_>>();
        // The launches that work here are known once the calls are placed.
        let (working, mut untimed) = match timing.as_deref() {
            Some(timing) => (timing.working_launches(), None),
            None => {
                let witnesses = self.witnesses(kernel, &all)?;
                (witnesses.working_launches(), Some(witnesses))
            }
        };
        let id = (self.label.clone(), key(working));
        let observed = shared.evidence.borrow().by_key.get(&id).copied();
        if let Some(observed) = observed {
            shared.evidence.borrow_mut().attribute(artifact, observed);
            return Ok(Some(observed));
        }
        self.ensure_reference(deadline)?;
        let mut rotations: Vec<Option<Vec<SubjectMeasurement>>> = vec![None; all.len()];
        let mut measure_at = |rotation: usize,
                              values: Vec<DecodedValue>,
                              args: &EncodedArgs|
         -> Result<(), Exclusion> {
            let started = Instant::now();
            let measured = observe(values, args, &inputs.states, None).map(|actual| {
                compare(
                    inputs.reference.borrow()[rotation]
                        .as_ref()
                        .expect("the default is observed at every witness first"),
                    &actual,
                    &inputs.subjects,
                    shared.requirement.activation(),
                )
            });
            self.comparing_seconds
                .set(self.comparing_seconds.get() + started.elapsed().as_secs_f64());
            rotations[rotation] = Some(measured?);
            Ok(())
        };
        let workload = &inputs.workload;
        let covered = match timing {
            Some(timing) => {
                timing.observe_first(deadline, |position, values, args| {
                    measure_at(workload[position], values, args)
                })?;
                workload.clone()
            }
            None => Vec::new(),
        };
        let rest = all
            .iter()
            .copied()
            .filter(|rotation| !covered.contains(rotation))
            .collect::<Vec<_>>();
        if !rest.is_empty() {
            let mut witnesses = match untimed.take() {
                Some(witnesses) => witnesses,
                None => self.witnesses(kernel, &rest)?,
            };
            witnesses.observe_first(deadline, |position, values, args| {
                measure_at(rest[position], values, args)
            })?;
        }
        let mut evidence = shared.evidence.borrow_mut();
        let observed = evidence.observations.len();
        evidence.observations.push(NumericalObservation {
            point: id.0.clone(),
            key: id.1.clone(),
            rotations: rotations
                .into_iter()
                .map(|rotation| rotation.expect("every witness was measured"))
                .collect(),
        });
        evidence.by_key.insert(id, observed);
        evidence.attribute(artifact, observed);
        Ok(Some(observed))
    }

    /// [`PreparedPoint::observe_before`], then the unit's requirement
    /// applied to what was measured: the candidate is admitted here, or
    /// excluded by the first constraint it violates.
    pub(super) fn admit_before(
        &self,
        timing: Option<&mut PointTiming<'a>>,
        kernel: &Arc<NativePrepared>,
        key: &dyn Fn(Vec<usize>) -> PointKey,
        deadline: Deadline,
    ) -> Result<(), WorkError<Exclusion>> {
        let Some(observed) = self.observe_before(timing, kernel, key, deadline)? else {
            return Ok(());
        };
        let shared = &self.shared;
        admit(
            &shared.evidence.borrow().observations[observed],
            &shared.requirement,
        )
        .map_err(|violation| WorkError::Failed(violation.exclusion()))
    }

    /// Whether a candidate formed as `artifact` whose work here is `key`
    /// owes no witness here ([`PreparedPoint::observe_before`]): it is the
    /// reference itself, or that work was observed here before.
    pub(super) fn witnessed(&self, artifact: &str, key: &PointKey) -> bool {
        let shared = &self.shared;
        artifact == shared.default.artifact().0
            || shared
                .evidence
                .borrow()
                .by_key
                .contains_key(&(self.label.clone(), key.clone()))
    }

    /// What this point's operation classes have been observed to take.
    pub(super) fn prices(&self) -> PointPrices {
        self.prices.get()
    }

    /// Host time spent capturing and measuring candidates' calls here.
    pub(super) fn validation_seconds(&self) -> f64 {
        self.comparing_seconds.get()
    }

    /// Weigh this point by `weight` in the search's objective.
    pub(super) fn reweigh(&mut self, weight: f64) {
        self.weight = weight;
    }
}

/// Prepares a unit's points for observation as their inputs arrive, and
/// holds everything observed at them.
pub(super) struct Validator {
    mutable: Vec<(usize, String)>,
    shared: Rc<Shared>,
    identity: String,
}

impl Validator {
    pub(super) fn new(
        device: &Arc<DeviceInner>,
        logical: &Arc<LogicalEntry>,
        mutable: &[(usize, String)],
        requirement: &NumericalRequirement,
        default: &Arc<NativePrepared>,
    ) -> Self {
        // What every measurement of the unit is a measurement of, besides
        // its point and key: the comparison, the default it is against, the
        // entry and the device. Tensor contents are generated test data;
        // the caller's tuning key names what they are generated from. The
        // requirement's α is not part of it: it judges measurements and
        // changes none.
        let mut identity = Sha256::new();
        identity.update(b"native-measurement");
        identity.update(COMPARISON_VERSION.to_le_bytes());
        identity.update(format!("{:?}", requirement.activation()).as_bytes());
        identity.update(default.artifact().0.as_bytes());
        identity.update(seismic_lang::reference_math::VERSION);
        identity.update(logical.identity().digest());
        identity.update(logical.module_hash().digest());
        identity.update(device.tuning_identity());
        Self {
            mutable: mutable.to_vec(),
            shared: Rc::new(Shared {
                default: default.clone(),
                requirement: requirement.clone(),
                logical: logical.clone(),
                pages: RefCell::new(ReferencePages::default()),
                seconds: RefCell::new(0.),
                evidence: RefCell::new(Evidence::default()),
            }),
            identity: crate::telemetry::hex(&identity.finalize()),
        }
    }

    /// The identity of the unit's measurements.
    pub(super) fn identity(&self) -> &str {
        &self.identity
    }

    /// Time spent executing the default as the reference so far.
    pub(super) fn reference_seconds(&self) -> f64 {
        *self.shared.seconds.borrow()
    }

    /// Host bytes of the default's observations the unit holds: the
    /// distinct reference pages its resident points share.
    pub(super) fn retained_bytes(&self) -> u64 {
        self.shared.pages.borrow().retained()
    }

    /// The most [`Validator::retained_bytes`] has been since the mark was
    /// last restarted.
    pub(super) fn retained_high_water(&self) -> u64 {
        self.shared.pages.borrow().high_water()
    }

    pub(super) fn restart_retained_high_water(&self) {
        self.shared.pages.borrow_mut().restart_high_water();
    }

    /// Every observation made of the unit's candidates, in the order made.
    pub(super) fn observations(&self) -> Vec<NumericalObservation> {
        self.shared.evidence.borrow().observations.clone()
    }

    /// The observations `artifact` was held to, as indices into
    /// [`Validator::observations`]: one per point it was checked at,
    /// whatever its verdict there.
    pub(super) fn held(&self, artifact: &str) -> Vec<usize> {
        self.shared
            .evidence
            .borrow()
            .by_artifact
            .get(artifact)
            .cloned()
            .unwrap_or_default()
    }

    /// Whether the unit's requirement admits `artifact` at every point it
    /// was held to.
    pub(super) fn admits(&self, artifact: &str) -> bool {
        let evidence = self.shared.evidence.borrow();
        self.held(artifact).iter().all(|observation| {
            admit(&evidence.observations[*observation], &self.shared.requirement).is_ok()
        })
    }

    /// `point` ready for observation; the default executes there on first
    /// use.
    pub(super) fn point<'a>(
        &self,
        point: TuningPoint<'a>,
    ) -> Result<PreparedPoint<'a>, TuneError> {
        Ok(PreparedPoint {
            label: point.label.clone(),
            weight: point.weight,
            class: point.class.clone(),
            shared: self.shared.clone(),
            prices: Rc::new(Cell::new(PointPrices::default())),
            comparing_seconds: Cell::new(0.),
            inputs: Some(self.inputs(point)?),
        })
    }

    /// Hold `point`'s inputs, built again, at `prepared`, which gave its
    /// own up ([`PreparedPoint::release`]).
    pub(super) fn restore<'a>(
        &self,
        prepared: &mut PreparedPoint<'a>,
        point: TuningPoint<'a>,
    ) -> Result<(), TuneError> {
        prepared.inputs = Some(self.inputs(point)?);
        Ok(())
    }

    fn inputs<'a>(&self, mut point: TuningPoint<'a>) -> Result<Inputs<'a>, TuneError> {
        if let Some((_, parameter)) = self.mutable.first().filter(|_| point.initialize.is_none()) {
            return Err(TuneError::SharedMutableState {
                point: point.label.clone(),
                parameter: parameter.clone(),
            });
        }
        if let Some(name) = point
            .written
            .keys()
            .find(|name| !self.mutable.iter().any(|(_, parameter)| parameter == *name))
        {
            return Err(TuneError::Declaration(format!(
                "point `{}` declares written rows of `{name}`, which is not a `&mut` parameter",
                point.label
            )));
        }
        let states: Vec<StateRegion> = self
            .mutable
            .iter()
            .map(|(ordinal, name)| StateRegion {
                ordinal: *ordinal,
                written: point.written.get(name).cloned(),
            })
            .collect();
        let initialize = point.initialize.take().map(|f| Rc::new(RefCell::new(f)));
        Ok(Inputs {
            initialize,
            subjects: subjects(&self.shared.logical, &states),
            states,
            reference: RefCell::new((0..point.rotation.len()).map(|_| None).collect()),
            rotation: point.rotation,
            workload: point.workload,
        })
    }
}
