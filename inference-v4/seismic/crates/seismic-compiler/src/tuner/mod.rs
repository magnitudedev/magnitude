//! Resolve the implementation's own choices using analysis derived from that IR.
//! Native compilation is downstream and never participates in candidate selection.
use seismic_accounting::{
    schedule,
    selection::{self, Cost, Objective, Selected},
    workload::{DerivationError, DerivationLimits, ScalarWorkload},
};
use seismic_lang::{ir, lower, lowered_ir::LoweredIr, program::Program, types::Elem};
use std::{collections::{BTreeMap, HashMap}, sync::Arc};

/// Backend-owned implementation boundary. Implementations prepare IR and derive
/// models from it; native compilation and timing feedback are forbidden here.
pub trait Backend {
    type Execution;
    type Conditions: Clone + PartialEq;
    fn name(&self) -> &'static str;
    fn conditions(&self) -> Self::Conditions;
    fn description(&self) -> Description;
    fn prepare(
        &self,
        function: &LoweredIr,
        path: &[usize],
    ) -> Result<Preparation<Self::Execution>, String>;
    /// Refine the actual retained owner without repeating frontend preparation.
    /// The owner must come from this backend under the same request identity;
    /// unsupported owner types continue through ordinary path construction.
    fn refine(
        &self,
        _alternatives: &selection::Domain,
        _index: usize,
    ) -> Result<Option<Preparation<Self::Execution>>, String> {
        Ok(None)
    }
    /// Relax resource constraints of the partial implementation retained by the
    /// typed domain. Counts must hold for every alternative in this interval.
    /// No numeric bound, independent execution graph or candidate sample enters.
    fn relax(
        &self,
        _alternatives: &selection::Domain,
        _indices: std::ops::Range<usize>,
        _workload: &ScalarWorkload,
        _limits: DerivationLimits,
    ) -> Result<Option<schedule::Demand>, String> {
        Ok(None)
    }
    /// Demand of a completed execution without constructing its full schedule.
    fn relax_execution(
        &self,
        _execution: &Self::Execution,
        _workload: &ScalarWorkload,
        _limits: DerivationLimits,
    ) -> Result<Option<schedule::Demand>, String> {
        Ok(None)
    }
    fn analyze(
        &self,
        execution: &Self::Execution,
        workload: &ScalarWorkload,
        limits: DerivationLimits,
    ) -> Result<schedule::evaluation::Model, DerivationError>;
    fn materialize(
        &self,
        execution: &Self::Execution,
        objective: &Objective,
    ) -> Result<Self::Execution, String>;
    fn check_materialization(
        &self,
        source: &Self::Execution,
        selected: &Self::Execution,
        objective: &Objective,
    ) -> Result<(), String>;
}
pub struct Description {
    pub target: String,
    pub contracts: String,
    pub form: String,
    pub objective: String,
    pub timebase: schedule::Timebase,
}
pub enum Preparation<E> {
    Choice {
        name: String,
        alternatives: selection::Domain,
    },
    Execution(E),
    Infeasible(selection::CapacityViolation),
    Unresolved(String),
}
fn preparation_node<E>(preparation: Preparation<E>) -> selection::Node<E> {
    match preparation {
        Preparation::Choice { name, alternatives } => {
            selection::Node::Choice { name, alternatives }
        }
        Preparation::Execution(execution) => selection::Node::Realization(execution),
        Preparation::Infeasible(violation) => selection::Node::Infeasible(violation),
        Preparation::Unresolved(reason) => selection::Node::Unresolved(reason),
    }
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
            workload: self.workload.clone(),
        }
    }
}
struct Space<'a, 'b, B: Backend> {
    request: &'a Request<'b, B>,
    context: selection::Context,
    inputs: Arc<Inputs<B::Conditions>>,
}
impl<'a, 'b, B: Backend> Space<'a, 'b, B> {
    fn new(request: &'a Request<'b, B>) -> Self {
        let name = match &request.input {
            Input::Lowered(f) => f.name.as_str(),
            Input::Portable { entry, .. } => entry,
        };
        let d = request.backend.description();
        Self {
            request,
            inputs: Arc::new(request.inputs()),
            context: selection::Context {
                program: name.to_string(),
                workload: request.workload.identity.clone(),
                target: d.target,
                contracts: d.contracts,
                execution_form: d.form,
                objective: d.objective,
                seconds_numerator: d.timebase.seconds_numerator,
                seconds_denominator: d.timebase.seconds_denominator,
            },
        }
    }
    fn backend_node(
        &self,
        function: &LoweredIr,
        path: &[usize],
    ) -> Result<selection::Node<B::Execution>, String> {
        function.ownership.validate(function)?;
        if !function.ownership.intermediates.is_empty() {
            let (parameters, _) = seismic_realization::storage::parameters(function)?;
            if parameters.len() != self.request.workload.buffers.len() {
                return Err("composition workload does not match the entry storage ABI".into());
            }
            for (index, parameter) in parameters.iter().enumerate() {
                if !function
                    .ownership
                    .intermediates
                    .contains(&parameter.parameter)
                {
                    continue;
                }
                let allocation = self.request.workload.buffers[index].allocation;
                if self
                    .request
                    .workload
                    .buffers
                    .iter()
                    .enumerate()
                    .any(|(other, b)| other != index && b.allocation == allocation)
                {
                    return Err(format!(
                        "private intermediate {} aliases another entry argument",
                        parameter.parameter
                    ));
                }
            }
        }
        Ok(preparation_node(
            self.request.backend.prepare(function, path)?,
        ))
    }
}
impl<B: Backend> selection::Space for Space<'_, '_, B> {
    type Execution = B::Execution;
    type Identity = Arc<Inputs<B::Conditions>>;
    fn identity(&self) -> Self::Identity {
        self.inputs.clone()
    }
    fn refine(
        &self,
        alternatives: &selection::Domain,
        index: usize,
    ) -> Result<Option<selection::Node<B::Execution>>, String> {
        if let Some(owner) = alternatives.owner::<lower::alternatives::LoweringChoice>() {
            return Ok(Some(match owner.refine(index)? {
                lower::alternatives::Expansion::RetainedChoice(choice) => selection::Node::Choice {
                    name: format!("{:?}", choice.decision().kind),
                    alternatives: selection::Domain::new(choice)?,
                },
                lower::alternatives::Expansion::Lowered { function, .. } => self.backend_node(&function, &[])?,
                lower::alternatives::Expansion::Choice(_) => return Err("retained refinement returned an earlier lowering stage".into()),
            }));
        }
        self.request
            .backend
            .refine(alternatives, index)
            .map(|preparation| preparation.map(preparation_node))
    }
    fn relax(
        &self,
        alternatives: &selection::Domain,
        indices: std::ops::Range<usize>,
    ) -> Result<Option<schedule::Demand>, String> {
        self.request
            .backend
            .relax(alternatives, indices, self.request.workload, self.request.derivation_limits)
    }
    fn context(&self) -> &selection::Context {
        &self.context
    }
    fn analyze(&self, execution: &B::Execution) -> Result<schedule::evaluation::Model, DerivationError> {
        self.request.backend.analyze(
            execution,
            self.request.workload,
            self.request.derivation_limits,
        )
    }
    fn relax_execution(&self, execution: &B::Execution) -> Result<Option<schedule::Demand>, String> {
        self.request.backend.relax_execution(execution, self.request.workload, self.request.derivation_limits)
    }
    fn materialize(
        &self,
        execution: &B::Execution,
        objective: &Objective,
    ) -> Result<B::Execution, String> {
        let selected = self.request.backend.materialize(execution, objective)?;
        self.request
            .backend
            .check_materialization(execution, &selected, objective)?;
        Ok(selected)
    }
    fn expand(&self, path: &[usize]) -> Result<selection::Node<B::Execution>, String> {
        match &self.request.input {
            Input::Lowered(function) => self.backend_node(function, path),
            Input::Portable {
                program,
                entry,
                shapes,
                elements,
                options,
            } => {
                match lower::alternatives::expand(
                    lower::alternatives::Specialization {
                        program,
                        entry,
                        backend: self.request.backend.name(),
                        shapes,
                        elements,
                        options,
                    },
                    path,
                )? {
                    lower::alternatives::Expansion::Choice(d) => Ok(selection::Node::Choice {
                        name: format!("{:?}", d.kind),
                        alternatives: selection::Domain::new(d)?,
                    }),
                    lower::alternatives::Expansion::RetainedChoice(choice) => Ok(selection::Node::Choice {
                        name: format!("{:?}", choice.decision().kind),
                        alternatives: selection::Domain::new(choice)?,
                    }),
                    lower::alternatives::Expansion::Lowered { function, consumed } => {
                        self.backend_node(&function, &path[consumed..])
                    }
                }
            }
        }
    }
}
pub struct TunedIr<E, C> {
    selected: Selected<E>,
    inputs: Arc<Inputs<C>>,
}
impl<E, C> TunedIr<E, C> {
    pub fn execution(&self) -> &E {
        self.selected.execution()
    }
    pub fn selected_path(&self) -> &[usize] {
        self.selected.selected_path()
    }
    pub fn modeled_cost(&self) -> Cost {
        self.selected.cost()
    }
    pub fn objective(&self) -> &Objective { self.selected.objective() }
    pub fn conditions(&self) -> &C {
        &self.inputs.conditions
    }
    pub fn workload(&self) -> &ScalarWorkload {
        &self.inputs.workload
    }
    pub fn into_parts(self) -> (E, Artifact<C>) {
        let objective = self.selected.objective().clone();
        debug_assert!(
            objective.cost().is_exact(),
            "a completed global minimum resolves its selected execution's objective"
        );
        let artifact = Artifact {
            selected_path: self.selected.selected_path().to_vec(),
            objective,
            inputs: self.inputs,
        };
        (self.selected.into_execution(), artifact)
    }
}
pub struct Artifact<C> {
    selected_path: Vec<usize>,
    objective: Objective,
    inputs: Arc<Inputs<C>>,
}
impl<C> Artifact<C> {
    pub fn selected_path(&self) -> &[usize] {
        &self.selected_path
    }
    pub fn modeled_cost(&self) -> Cost {
        self.objective.cost()
    }
    pub fn objective(&self) -> &Objective { &self.objective }
    pub fn conditions(&self) -> &C {
        &self.inputs.conditions
    }
    pub fn workload(&self) -> &ScalarWorkload {
        &self.inputs.workload
    }
}
pub struct Progress<E, C> {
    search: selection::Progress<E, Arc<Inputs<C>>>,
}
impl<E, C> Progress<E, C> {
    pub fn nodes_visited(&self) -> usize {
        self.search.nodes_visited()
    }
    pub fn missing_mappings(&self) -> impl Iterator<Item = (&[usize], &[String])> {
        self.search.missing_mappings()
    }
    pub fn frontier(&self) -> &[selection::Region] {
        self.search.frontier()
    }
    pub fn feasible_upper(&self) -> Option<u64> {
        self.search.feasible_upper()
    }
    pub fn lower_bound(&self) -> Result<u64, String> {
        self.search.lower_bound()
    }
    pub fn unresolved(&self) -> selection::Unresolved {
        self.search.unresolved()
    }
    pub fn unsupported_analyses(&self) -> impl Iterator<Item = (&[usize], &str)> {
        self.search.unsupported_analyses()
    }
    pub fn exhausted_derivations(
        &self,
    ) -> impl Iterator<Item = (&[usize], seismic_accounting::workload::DerivationLimit)> {
        self.search.exhausted_derivations()
    }
}
pub enum Outcome<E, C> {
    Optimal(TunedIr<E, C>),
    Incomplete(Progress<E, C>),
    Infeasible,
}
fn finish<E, C>(outcome: selection::Outcome<E, Arc<Inputs<C>>>, inputs: Arc<Inputs<C>>) -> Outcome<E, C> {
    match outcome {
        selection::Outcome::Optimal(selected) => Outcome::Optimal(TunedIr { selected, inputs }),
        selection::Outcome::Incomplete(search) => Outcome::Incomplete(Progress { search }),
        selection::Outcome::Infeasible => Outcome::Infeasible,
    }
}
pub fn tune<B: Backend>(
    request: &Request<'_, B>,
    budget: selection::Budget,
) -> Result<Outcome<B::Execution, B::Conditions>, String> {
    let space = Space::new(request);
    Ok(finish(selection::select(&space, budget)?, space.inputs))
}
pub fn resume<B: Backend>(
    request: &Request<'_, B>,
    progress: Progress<B::Execution, B::Conditions>,
    budget: selection::Budget,
) -> Result<Outcome<B::Execution, B::Conditions>, String> {
    let space = Space::new(request);
    Ok(finish(selection::resume(&space, progress.search, budget)?, space.inputs))
}
