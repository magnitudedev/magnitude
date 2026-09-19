//! Automatic selection through the shared domain-independent solver contract.
//! Seismic constructs one immutable family model and validates reconstruction;
//! optimization strategy and completion belong exclusively to `Search`.
pub mod choices;
pub mod expressions;
pub mod family;
pub mod geometry;
pub mod source;

use magnitude_solver::{Limits, Options, Search, Stats};
use seismic_accounting::{objective::{Cost, Objective}, schedule, workload::{DerivationLimits, ScalarWorkload}};
use seismic_lang::{ir, lower, lowered_ir::LoweredIr, program::Program, types::Elem};
use std::{collections::{BTreeMap, HashMap}, sync::Arc};

/// Complete unresolved backend family. Neither native compilation nor a search
/// over completed realizations is permitted during export or reconstruction.
pub trait Backend {
    type Execution: 'static;
    type Conditions: Clone + PartialEq;
    fn name(&self) -> &'static str;
    fn conditions(&self) -> Self::Conditions;
    fn description(&self) -> Description;
    fn export(&self, input: Input<'_>, workload: &ScalarWorkload, limits: DerivationLimits)
        -> Result<family::Export<Self::Execution>, String>;
}

/// Compiler settings configure the common solver implementation, never a second
/// selection algorithm. Limits are incremental and options are fixed per session.
#[derive(Clone, Debug, Default)]
pub struct Settings {
    pub options: Options,
    pub limits: Limits,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Description {
    pub target: String,
    pub contracts: String,
    pub form: String,
    pub objective: String,
    pub scheduling: seismic_accounting::authority::ScheduleInterpretation,
    pub timebase: schedule::Timebase,
}
#[derive(Clone, Copy)]
pub enum Input<'a> {
    Lowered(&'a LoweredIr),
    Portable {
        program: &'a Program,
        entry: &'a str,
        shapes: &'a HashMap<String, i64>,
        elements: &'a HashMap<String, Elem>,
        options: &'a lower::Options,
    },
}
#[derive(Clone, Debug, PartialEq)]
enum Source {
    Lowered(LoweredIr),
    Portable {
        functions: Vec<ir::Function>,
        lowerings: Vec<ir::Lowering>,
        entry: String,
        shapes: BTreeMap<String, i64>,
        elements: BTreeMap<String, Elem>,
        options: lower::Options,
    },
}
pub struct Request<'a, B: Backend> {
    pub input: Input<'a>,
    pub backend: &'a B,
    pub workload: &'a ScalarWorkload,
    pub derivation_limits: DerivationLimits,
}
#[derive(Clone, PartialEq)]
struct Inputs<C> {
    source: Source,
    conditions: C,
    workload: ScalarWorkload,
    description: Description,
    construction: DerivationLimits,
}
impl<B: Backend> Request<'_, B> {
    fn inputs(&self) -> Inputs<B::Conditions> {
        Inputs {
            source: match &self.input {
                Input::Lowered(function) => Source::Lowered((*function).clone()),
                Input::Portable {
                    program,
                    entry,
                    shapes,
                    elements,
                    options,
                } => Source::Portable {
                    functions: program.functions.clone(),
                    lowerings: program.lowerings.clone(),
                    entry: (*entry).into(),
                    shapes: shapes.iter().map(|(k, v)| (k.clone(), *v)).collect(),
                    elements: elements
                        .iter()
                        .map(|(k, v)| (k.clone(), v.clone()))
                        .collect(),
                    options: (*options).clone(),
                },
            },
            conditions: self.backend.conditions(),
            description: self.backend.description(),
            workload: self.workload.clone(),
            construction: self.derivation_limits,
        }
    }
}

pub struct TunedIr<E, C> {
    execution: E,
    objective: Objective,
    decisions: Vec<family::Decision>,
    inputs: Arc<Inputs<C>>,
    /// Retaining the global solution keeps its model/proof scope attached.
    solution: magnitude_solver::Solution,
}
impl<E, C> TunedIr<E, C> {
    pub fn execution(&self) -> &E { &self.execution }
    pub fn decisions(&self) -> &[family::Decision] { &self.decisions }
    pub fn modeled_cost(&self) -> Cost { self.objective.cost() }
    pub fn objective(&self) -> &Objective { &self.objective }
    pub fn description(&self) -> &Description { &self.inputs.description }
    pub fn conditions(&self) -> &C { &self.inputs.conditions }
    pub fn workload(&self) -> &ScalarWorkload { &self.inputs.workload }
    pub fn into_parts(self) -> (E, Artifact<C>) {
        (self.execution, Artifact { decisions: self.decisions, objective: self.objective,
            inputs: self.inputs, solution: self.solution })
    }
}
pub struct Artifact<C> {
    decisions: Vec<family::Decision>,
    objective: Objective,
    inputs: Arc<Inputs<C>>,
    solution: magnitude_solver::Solution,
}
impl<C> Artifact<C> {
    pub fn decisions(&self) -> &[family::Decision] { &self.decisions }
    pub fn modeled_cost(&self) -> Cost { self.objective.cost() }
    pub fn objective(&self) -> &Objective { &self.objective }
    pub fn description(&self) -> &Description { &self.inputs.description }
    pub fn conditions(&self) -> &C { &self.inputs.conditions }
    pub fn workload(&self) -> &ScalarWorkload { &self.inputs.workload }
    pub fn solution(&self) -> &magnitude_solver::Solution { &self.solution }
}

/// Retains the original model, reconstruction and engine state together. There
/// are no compiler-owned choice regions or independent scheduling frontiers.
pub struct Progress<E, C> {
    export: family::Export<E>,
    search: Search,
    inputs: Arc<Inputs<C>>,
    result: magnitude_solver::Progress,
}
impl<E: 'static, C> Progress<E, C> {
    pub fn stats(&self) -> &Stats { self.search.stats() }
    pub fn model(&self) -> &Arc<magnitude_solver::Model> { self.search.model() }
    pub fn feasible_upper(&self) -> Option<u64> { self.result.incumbent.as_ref().map(|w| w.cost()) }
    pub fn lower_bound(&self) -> u64 { self.result.lower_bound }
    pub fn reason(&self) -> &magnitude_solver::result::StopReason { &self.result.reason }
    pub fn incumbent(&self) -> Option<&magnitude_solver::FeasibleSolution> { self.result.incumbent.as_ref() }
    /// Diagnostics have no conversion into TunedIr or an executable artifact.
    pub fn reconstruct_incumbent(&self) -> Result<Option<family::Reconstructed<E>>, String> {
        self.result.incumbent.as_ref().map(|w| self.export.reconstruct(w, self.result.lower_bound)).transpose()
    }
}
pub enum Outcome<E, C> {
    Optimal(TunedIr<E, C>),
    Incomplete(Progress<E, C>),
    Infeasible,
}

fn advance<E: 'static, C>(export: family::Export<E>, mut search: Search,
    inputs: Arc<Inputs<C>>, limits: Limits) -> Result<Outcome<E, C>, String> {
    match search.advance(limits).map_err(|e| e.to_string())? {
        magnitude_solver::Outcome::Optimal(solution) => {
            let selected = export.reconstruct(solution.feasible(), solution.cost())?;
            check_source_workload(&selected.source, &inputs.workload)?;
            let timebase = selected.objective.timebase();
            if u128::from(timebase.seconds_numerator) * u128::from(inputs.description.timebase.seconds_denominator)
                != u128::from(inputs.description.timebase.seconds_numerator) * u128::from(timebase.seconds_denominator) {
                return Err("reconstructed execution has a different objective time unit".into());
            }
            Ok(Outcome::Optimal(TunedIr { execution: selected.execution, objective: selected.objective,
                decisions: selected.decisions, inputs, solution }))
        }
        magnitude_solver::Outcome::Incomplete(result) => Ok(Outcome::Incomplete(Progress { export, search, inputs, result })),
        magnitude_solver::Outcome::Infeasible => Ok(Outcome::Infeasible),
    }
}

pub fn tune<B: Backend>(request: &Request<'_, B>, settings: Settings)
    -> Result<Outcome<B::Execution, B::Conditions>, String> {
    request.workload.validate()?;
    let inputs = Arc::new(request.inputs());
    let description = &inputs.description;
    if [&description.target, &description.contracts, &description.form, &description.objective]
        .iter().any(|field| field.is_empty()) || description.timebase.seconds_numerator == 0
        || description.timebase.seconds_denominator == 0 {
        return Err("selection requires identified conditions and a positive objective time unit".into());
    }
    let export = request.backend.export(request.input, request.workload, request.derivation_limits)?;
    let search = Search::new(export.model().clone(), settings.options).map_err(|e| e.to_string())?;
    advance(export, search, inputs, settings.limits)
}
pub fn resume<B: Backend>(request: &Request<'_, B>, progress: Progress<B::Execution, B::Conditions>, limits: Limits)
    -> Result<Outcome<B::Execution, B::Conditions>, String> {
    if *progress.inputs != request.inputs() {
        return Err("selection inputs changed: source, workload, backend, objective or construction limits differ; build a new immutable model".into());
    }
    advance(progress.export, progress.search, progress.inputs, limits)
}

/// Private publications must retain their nonaliasing contract in the selected
/// source, including when family reconstruction changes phase/storage topology.
pub fn check_source_workload(function: &LoweredIr, workload: &ScalarWorkload) -> Result<(), String> {
    function.ownership.validate(function)?;
    if function.ownership.intermediates.is_empty() { return Ok(()); }
    let (parameters, _) = seismic_realization::storage::parameters(function)?;
    if parameters.len() != workload.buffers.len() {
        return Err("composition workload does not match the entry storage ABI".into());
    }
    for (index, parameter) in parameters.iter().enumerate() {
        if function.ownership.intermediates.contains(&parameter.parameter) {
            let allocation = workload.buffers[index].allocation;
            if workload.buffers.iter().enumerate().any(|(other, b)| other != index && b.allocation == allocation) {
                return Err(format!("private intermediate {} aliases another entry argument", parameter.parameter));
            }
        }
    }
    Ok(())
}
