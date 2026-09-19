//! Compiler/solver boundary tests: one retained model, guarded accounting,
//! checked reconstruction and proof-gated execution. Native semantics live in
//! the backend acceptance tests rather than in this contract fixture.
use magnitude_solver::{model::{Constraint, Cost, Domain, LinearTerm, ModelBuilder, ObligationKind, VarId}, Algorithm, FeasibleSolution, Limits, Model, Options};
use seismic_accounting::{authority::{InstructionOrder, ModelRelationship, ScheduleInterpretation, TimingSemantics}, objective::Objective, schedule::{self, export::Binding, symbolic::Encoding}, workload::{DerivationLimits, ScalarWorkload}};
use seismic_compiler::tuner::{self, family::{Decision, Export, Reconstructed, Reconstruction}, Backend, Description, Input, Outcome, Request, Settings};
use seismic_lang::lowered_ir::LoweredIr;
use std::{cell::Cell, rc::Rc, sync::Arc};

fn timebase() -> schedule::Timebase { schedule::Timebase { seconds_numerator: 1, seconds_denominator: 1 } }
fn source() -> LoweredIr {
    let program = seismic_lang::program::compile(&[seismic_lang::program::SourceFile {
        path: "session.seismic.portable".into(), scope: seismic_lang::Scope::Portable,
        text: "fn empty():\n  x = 1\n".into(),
    }], &[]).unwrap();
    seismic_lang::lower::lower(&program, "empty", "cpu", &Default::default()).unwrap()
}
fn workload() -> ScalarWorkload {
    ScalarWorkload { identity: "fixed fixture".into(), allocations: vec![], buffers: vec![], scalars: vec![], integer_domains: vec![] }
}
fn limits(work: u64) -> Limits { Limits { work, ..Default::default() } }
fn settings(algorithm: Algorithm, work: u64) -> Settings {
    Settings { options: Options { algorithm, ..Default::default() }, limits: limits(work) }
}
fn operation(latency: u64) -> schedule::Model {
    schedule::Model {
        relationship: ModelRelationship::hypothetical_execution(), identity: format!("original operation {latency}"), timebase: timebase(),
        resources: vec![schedule::Resource { name: "shared issue".into(), capacity: 1, unit: schedule::CapacityUnit::Slots }],
        operations: vec![schedule::Operation { name: "operation".into(), latency, predecessors: vec![], start_predecessors: vec![],
            reservations: vec![schedule::Reservation { resource: 0, offset: 0, duration: latency, units: 1 }] }],
        lifetimes: vec![], static_orders: vec![], unmapped: vec![],
    }
}
struct Fixture {
    model: Arc<Model>,
    choice: VarId,
    arms: Rc<Vec<Binding>>,
    exports: Cell<usize>,
    reconstructed: Rc<Cell<usize>>,
    conditions: u8,
    wrong_cost: bool,
}
impl Fixture {
    fn new(coverage: bool, impossible: bool) -> Self {
        let mut builder = ModelBuilder::new();
        builder.units("seconds");
        let choice = builder.variable("implementation", Domain::boolean());
        let inverse = builder.variable("other implementation", Domain::boolean());
        builder.constraint(Constraint::NotEqual { left: choice, right: inverse });
        let mut encoding = Encoding::new(&mut builder, "whole family", &operation(9).resources, 12).unwrap();
        let arms = [9, 3].into_iter().zip([inverse, choice]).map(|(latency, presence)| {
            Binding::append(&mut builder, &mut encoding, Arc::new(operation(latency).into()), Some(presence)).unwrap()
        }).collect();
        if coverage { builder.obligation(vec![], ObligationKind::Construction, "fixture retains unexported computation"); }
        if impossible { builder.constraint(Constraint::NotEqual { left: choice, right: choice }); }
        let completion = encoding.finish(&mut builder).unwrap();
        builder.cost(Cost::Linear { constant: 0, terms: vec![LinearTerm::new(completion, 1)] });
        Self { model: Arc::new(builder.build().unwrap()), choice, arms: Rc::new(arms), exports: Cell::new(0), reconstructed: Rc::new(Cell::new(0)), conditions: 0, wrong_cost: false }
    }
}
struct Rebuild {
    source: LoweredIr, choice: VarId, arms: Rc<Vec<Binding>>, reconstructed: Rc<Cell<usize>>, wrong_cost: bool,
}
impl Reconstruction<usize> for Rebuild {
    fn reconstruct(&self, witness: &FeasibleSolution, lower_bound: u64) -> Result<Reconstructed<usize>, String> {
        self.reconstructed.set(self.reconstructed.get() + 1);
        let choice = usize::try_from(witness.values()[self.choice.0]).map_err(|_| "bad fixture choice")?;
        let objective = self.arms[choice].reconstruct(witness.values(), lower_bound)?.ok_or("selected fixture absent")?;
        let objective = if self.wrong_cost {
            let latency = witness.cost() + 1;
            Objective::from_flat(Arc::new(operation(latency)), schedule::Schedule { starts: vec![0], completion: latency }, lower_bound)?
        } else { objective };
        Ok(Reconstructed { source: self.source.clone(), execution: choice, objective,
            decisions: vec![Decision { identity: "implementation".into(), value: choice as i64 }] })
    }
}
impl Backend for Fixture {
    type Execution = usize;
    type Conditions = u8;
    fn name(&self) -> &'static str { "cpu" }
    fn conditions(&self) -> u8 { self.conditions }
    fn description(&self) -> Description {
        Description { target: "test target".into(), contracts: "test contract".into(), form: "guarded operations".into(), objective: "completion".into(), timebase: timebase(),
            scheduling: ScheduleInterpretation { instruction_order: InstructionOrder::FixedByRealization, timing: TimingSemantics::IdealResourceFeasible } }
    }
    fn export(&self, input: Input<'_>, _: &ScalarWorkload, _: DerivationLimits) -> Result<Export<usize>, String> {
        self.exports.set(self.exports.get()+1);
        let Input::Lowered(source) = input else { return Err("fixture needs explicit source".into()); };
        Export::new(self.model.clone(), Rebuild { source: source.clone(), choice: self.choice, arms: self.arms.clone(), reconstructed: self.reconstructed.clone(), wrong_cost: self.wrong_cost })
    }
}
fn request<'a>(backend: &'a Fixture, source: &'a LoweredIr, workload: &'a ScalarWorkload) -> Request<'a, Fixture> {
    Request { input: Input::Lowered(source), backend, workload, derivation_limits: DerivationLimits { instructions: 100, operations: 100 } }
}

#[test]
fn algorithms_share_model_and_reconstruction_and_resume_does_not_reexport() {
    let backend = Fixture::new(false, false);
    let source = source();
    let workload = workload();
    let request = request(&backend, &source, &workload);
    for algorithm in [Algorithm::Exact, Algorithm::Neighborhood(Default::default())] {
        let Outcome::Incomplete(progress) = tuner::tune(&request, settings(algorithm, 0)).unwrap() else { panic!("zero work retains the search"); };
        assert!(Arc::ptr_eq(progress.model(), &backend.model));
        let exports = backend.exports.get();
        match tuner::resume(&request, progress, limits(100_000)).unwrap() {
            Outcome::Optimal(selected) => {
                assert_eq!(*selected.execution(), 1);
                assert_eq!(selected.modeled_cost().upper(), 3);
                assert!(selected.modeled_cost().is_exact());
                let (_, artifact) = selected.into_parts();
                assert_eq!(artifact.solution().model(), backend.model.as_ref());
            }
            Outcome::Incomplete(progress) => {
                if let Some(incumbent) = progress.reconstruct_incumbent().unwrap() {
                    incumbent.objective.check_execution_upper().unwrap();
                    assert!(incumbent.execution < 2);
                    assert!(incumbent.objective.cost().upper() >= 3);
                }
            }
            Outcome::Infeasible => panic!("both alternatives have original feasible witnesses"),
        }
        assert_eq!(backend.exports.get(), exports, "resumption must retain export identity");
    }
}

#[test]
fn unresolved_coverage_and_zero_work_never_reconstruct_an_execution() {
    for coverage in [false, true] {
        let backend = Fixture::new(coverage, false);
        let source = source();
        let workload = workload();
        let work = if coverage { 10_000 } else { 0 };
        let Outcome::Incomplete(progress) = tuner::tune(&request(&backend, &source, &workload), settings(Algorithm::Exact, work)).unwrap() else { panic!("unfinished family cannot be executable"); };
        assert_eq!(backend.reconstructed.get(), 0);
        assert!(progress.incumbent().is_none());
        if coverage { assert!(matches!(progress.reason(), magnitude_solver::result::StopReason::Coverage(_))); }
    }
}

#[test]
fn infeasibility_and_reconstruction_defects_are_distinct() {
    let source = source();
    let workload = workload();
    let backend = Fixture::new(false, true);
    assert!(matches!(tuner::tune(&request(&backend, &source, &workload), settings(Algorithm::Exact, 100_000)).unwrap(), Outcome::Infeasible));
    assert_eq!(backend.reconstructed.get(), 0);
    let mut backend = Fixture::new(false, false);
    backend.wrong_cost = true;
    assert!(matches!(tuner::tune(&request(&backend, &source, &workload), settings(Algorithm::Exact, 100_000)), Err(error) if error.contains("objective differs")));
}

#[test]
fn resume_rejects_changed_source_workload_conditions_and_construction_limits() {
    let source = source();
    let workload = workload();
    for changed in 0..4 {
        let mut backend = Fixture::new(false, false);
        let Outcome::Incomplete(progress) = tuner::tune(&request(&backend, &source, &workload), settings(Algorithm::Exact, 0)).unwrap() else { panic!(); };
        let mut altered_source = source.clone();
        let mut altered_workload = workload.clone();
        if changed == 0 { altered_source.name.push_str(" changed"); }
        if changed == 1 { altered_workload.identity.push_str(" changed"); }
        if changed == 2 { backend.conditions += 1; }
        let mut request = request(&backend, &altered_source, &altered_workload);
        if changed == 3 { request.derivation_limits.operations += 1; }
        assert!(matches!(tuner::resume(&request, progress, limits(100_000)), Err(error) if error.contains("inputs changed")));
        assert_eq!(backend.exports.get(), 1);
        assert_eq!(backend.reconstructed.get(), 0);
    }
}
