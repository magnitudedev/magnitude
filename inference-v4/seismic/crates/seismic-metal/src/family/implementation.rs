//! Typed implementation definitions and assignment consumption. These bindings
//! never replay an ordinal path or substitute an unrepresented default choice.
use crate::{
    choices::{Decision, Domain},
    execution::{self, Config, Execution, Stage},
};
use magnitude_solver::model::{Domain as NumericDomain, ModelBuilder, VarId};
use seismic_accounting::choices::Choices;
use seismic_lang::{ir::LoadMode, lowered_ir::LoweredIr};
use seismic_realization::dispatch::WorkMapping;
use std::{
    collections::BTreeMap,
    sync::Arc,
};

pub struct Registry {
    name: String,
    entries: BTreeMap<String, Entry>,
    retained: Option<Execution>,
    selectors: BTreeMap<String, VarId>,
    private_guards: Vec<(seismic_lang::ir::VarId, magnitude_solver::model::Literal)>,
    launch_parameters: Option<Arc<Vec<crate::msl::parameters::Launch>>>,
    numeric_parameters: BTreeMap<String, seismic_accounting::algebra::Value>,
    numeric_definitions: BTreeMap<String, seismic_lang::sym::Sym>,
    retained_split: Option<execution::RetainedSplit>,
    phase_presence: Vec<Option<VarId>>,
    storage_binding: Option<crate::storage::family::Binding>,
}
struct Entry {
    domain: Domain,
    variable: VarId,
}
pub struct Template {
    pub execution: Execution,
    pub operations: Arc<crate::terminal::family::Family>,
    pub emission: crate::msl::GroupingTemplate,
}
fn identity(decision: &Decision) -> String {
    match decision {
        Decision::Fold(choice) => format!("fold.site{}", choice.site),
        Decision::Load(choice) => format!("load.site{}.variable{}", choice.site, choice.variable),
        Decision::Storage(choice) => format!("storage.variable{}", choice.variable),
        Decision::Reduction(choice) => format!(
            "reduction.operation{:?}.output{}",
            choice.site.operation, choice.site.output
        ),
        Decision::Allocation(choice) => format!("allocation.{:?}", choice.allocation),
        Decision::Transfer(choice) => format!(
            "transfer.launch{}.operation{:?}.site{}",
            choice.launch, choice.operation, choice.site
        ),
        Decision::Traversal(choice) => format!(
            "traversal.launch{}.operation{:?}.site{}",
            choice.launch, choice.operation, choice.site
        ),
    }
}
impl Registry {
    pub fn new(name: String) -> Self {
        Self {
            name,
            entries: BTreeMap::new(),
            retained: None,
            selectors: BTreeMap::new(),
            private_guards: Vec::new(),
            launch_parameters: None,
            numeric_parameters: BTreeMap::new(),
            numeric_definitions: BTreeMap::new(),
            retained_split: None,
            phase_presence: Vec::new(),
            storage_binding: None,
        }
    }
    pub fn append(&mut self, builder: &mut ModelBuilder, domain: Domain) -> Result<String, String> {
        let identity = format!("{}.{}", self.name, identity(&domain.decision));
        if let Some(previous) = self.entries.get(&identity) {
            if previous.domain != domain {
                return Err(format!(
                    "Metal decision identity changed its defining domain: {identity}"
                ));
            }
            return Ok(identity);
        }
        let maximum = i64::try_from(
            domain
                .len()
                .checked_sub(1)
                .ok_or("empty Metal implementation domain")?,
        )
        .map_err(|_| "Metal domain exceeds shared integer range")?;
        let variable = builder
            .local_variable(
                identity.clone(),
                NumericDomain::interval(0, maximum).map_err(|e| e.to_string())?,
            )
            .map_err(|e| e.to_string())?;
        self.entries
            .insert(identity.clone(), Entry { domain, variable });
        Ok(identity)
    }
    pub(crate) fn storage_parameters(&mut self, builder: &mut ModelBuilder, storage: &crate::storage::StorageFamily)
        -> Result<BTreeMap<seismic_lang::ir::VarId, VarId>, String> {
        let mut parameters = BTreeMap::new();
        for decision in storage.decisions() {
            let identity = self.append(builder, Domain { decision: Decision::Storage(decision.clone()) })?;
            let ordinal = self.entries[&identity].variable;
            for (_, guard) in self.private_guards.iter().filter(|(variable, _)| *variable == decision.variable) {
                let replicated = decision.alternatives.iter().position(|placement| *placement == seismic_realization::dispatch::TilePlacement::Replicated);
                match replicated {
                    Some(value) => builder.guarded_constraint(vec![guard.clone()], magnitude_solver::model::Constraint::InDomain {
                        variable: ordinal, domain: NumericDomain::singleton(value as i64) }),
                    None => builder.guarded_constraint(vec![guard.clone()], magnitude_solver::model::Constraint::LinearLe { terms: Vec::new(), rhs: -1 }),
                };
            }
            parameters.insert(decision.variable, ordinal);
        }
        Ok(parameters)
    }
    pub(crate) fn complete_work(
        &mut self, builder: &mut ModelBuilder, work: &super::decomposition::WorkTemplate,
    ) -> Result<Template, String> {
        use seismic_accounting::algebra::{Algebra, Symbolic};
        let one = Symbolic::new(builder, &self.name).constant(1).map_err(|error| error.to_string())?;
        let mut launches = Vec::new();
        for mapping in &work.phases {
            for (work, parts) in std::iter::once((mapping.work_items, mapping.parts))
                .chain(mapping.merge_work.map(|work| (work, one))) {
                let launch = crate::msl::parameters::Launch::new(builder, &format!("{}.launch{}", self.name, launches.len()), work,
                    parts, mapping.axes.iter().map(|axis| (axis.extent, axis.count, axis.step)))
                    .map_err(|error| error.to_string())?;
                launches.push(launch);
            }
        }
        self.launch_parameters = Some(Arc::new(launches));
        self.numeric_parameters = work.source.parameters.clone();
        self.numeric_definitions = work.source.numeric_definitions.clone();
        self.retained_split = work.retained_split.clone();
        self.phase_presence = work.source.phase_presence.clone();
        self.complete_retained(builder, &work.source, work.config.clone(), &work.structure)
    }
    fn retain_layout(&mut self, builder: &mut ModelBuilder, prepared: &execution::Prepared) -> Result<Execution, String> {
        use seismic_accounting::algebra::{Algebra, Symbolic};
        let mut prepared = prepared.clone();
        let mut layout = super::layout::Family { source_predicates: self.selectors.clone(), ..Default::default() };
        let mut operation = 0;
        seismic_lang::normalize::identify(&mut prepared.function.body, &mut operation);
        layout.retain_phase_presence(&prepared.function, &self.phase_presence)?;
        layout.bind_source(builder, &self.name)?;
        layout.retain_split_guards(&prepared.function, &prepared.phases)?;
        let load_occurrences = super::layout::load_occurrences(&prepared.function.body);
        let mut modes = Vec::new();
        for (site, definition) in seismic_lang::normalize::loads::sites(&prepared.function.body).into_iter().enumerate() {
            let mode = if let Some(mode) = definition.selected { mode } else if definition.can_borrow {
                let domain = Domain { decision: Decision::Load(seismic_lang::normalize::loads::Choice { site, variable: definition.variable }) };
                let identity = self.append(builder, domain)?;
                let choice = super::layout::Choice::new(builder, &identity, self.entries[&identity].variable,
                    &[LoadMode::Materialize, LoadMode::Borrow])?;
                let occurrence = *load_occurrences.get(site).ok_or("retained load definition lacks an original operation")?;
                layout.load_sites.insert(occurrence, choice.clone());
                // The union's allocation envelope contains the snapshot request.
                // Its original load parameter controls presence and the native
                // value-binding alternatives; this is not a selected policy.
                LoadMode::Materialize
            } else { LoadMode::Materialize };
            modes.push(mode);
        }
        seismic_lang::normalize::loads::resolve(&mut prepared.function.body, &modes)?;
        for phase in &mut prepared.phases {
            if let Some(split) = &mut phase.split {
                seismic_lang::normalize::select_loads(&mut split.validation_bindings, false);
                seismic_lang::normalize::identify(&mut split.validation_bindings, &mut operation);
            }
        }
        let extra = prepared.phases.iter().filter_map(|phase| phase.split.as_ref().map(|split| split.validation_bindings.as_slice())).collect::<Vec<_>>();
        let mut storage = crate::storage::StorageFamily::derive_parameterized(&prepared.function.vars, &prepared.function.body, &extra, &self.numeric_parameters)?;
        storage.force_replicated(&prepared.private_values)?;
        let storage = Arc::new(storage);
        let ordinals = self.storage_parameters(builder, &storage)?;
        for decision in storage.decisions() {
            let ordinal = ordinals[&decision.variable];
            layout.storage.insert(decision.variable, super::layout::Choice::new(builder, &format!("{}.storage.variable{}", self.name, decision.variable), ordinal, &decision.alternatives)?);
        }
        let mut algebra = Symbolic::new(builder, &self.name);
        let lanes = algebra.constant(execution::SUBGROUP as u64).map_err(|error| error.to_string())?;
        let items = algebra.constant(1).map_err(|error| error.to_string())?;
        let variable_guards = layout.variable_guards(builder, &self.name, &prepared.function)?;
        let variable_guards = layout.storage_guards(builder, &self.name, &variable_guards)?;
        self.storage_binding = Some(storage.append_guarded_bound(builder, &format!("{}.storage", self.name), lanes, items, &ordinals, &variable_guards).map_err(|error| error.to_string())?);
        layout.bind_owners(builder, &self.name, self.storage_binding.as_ref().ok_or("retained storage binding is absent")?)?;
        let storage = storage.layout_template()?;
        let reductions = crate::reduction::plan_template(&prepared.function.vars, &prepared.function.body, &prepared.phases, &storage)?;
        layout.append_reductions(builder, &self.name, &reductions)?;
        layout.bind_activation(builder, &self.name, &prepared.function, &variable_guards)?;
        let memory = crate::memory::plan_template(&prepared.function.vars, &prepared.function.body, &prepared.phases, &storage, &reductions)?
            .with_retained(&prepared.retained, &prepared.phases)?;
        layout.append_allocations(builder, &self.name, &memory, &prepared.function,
            self.storage_binding.as_ref().ok_or("retained storage binding is absent")?, &storage, &self.numeric_parameters)?;
        self.selectors = layout.predicates();
        Ok(self.bind_execution(&Execution {
            implementation: Some(Arc::new(layout)), launch_parameters: None, numeric_parameters: Default::default(), numeric_definitions: Default::default(),
            terminal: Default::default(), transfers: Vec::new(), traversals: Vec::new(),
            function: prepared.function, source: prepared.source, config: prepared.config, phases: prepared.phases,
            retained: prepared.retained, partition_parameters: prepared.partition_parameters,
            memory, support: crate::support::Plan::new(), storage, reductions,
        }))
    }
    fn bind_execution(&self, execution: &Execution) -> Execution {
        let mut execution = execution.clone();
        execution.launch_parameters = self.launch_parameters.clone();
        execution.numeric_parameters = self.numeric_parameters.clone();
        execution.numeric_definitions = self.numeric_definitions.clone();
        execution.invalidate_terminal();
        execution
    }
    /// The checked union carries compiler predicates separately from its
    /// placeholder initializer expressions. All arms remain present through
    /// stage construction and become compile-time target regions at emission.
    pub(crate) fn complete_retained(
        &mut self,
        builder: &mut ModelBuilder,
        source: &super::source::Template,
        config: Config,
        mappings: &[WorkMapping],
    ) -> Result<Template, String> {
        for predicate in &source.predicates {
            let variable = source.function.vars.get(predicate.variable)
                .ok_or("Metal source predicate has no typed variable")?;
            if variable.ty != seismic_lang::types::Ty::Scalar(seismic_lang::types::DType::Bool) {
                return Err("Metal source predicate is not boolean".into());
            }
            let symbol = crate::msl::variable_symbol(variable, predicate.variable);
            if self.selectors.insert(symbol, predicate.presence).is_some() {
                return Err("Metal source predicate has multiple definitions".into());
            }
        }
        self.complete(builder, &source.function, config, mappings)
    }
    pub fn complete(
        &mut self,
        builder: &mut ModelBuilder,
        source: &LoweredIr,
        config: Config,
        mappings: &[WorkMapping],
    ) -> Result<Template, String> {
        let selectors = source.vars.iter().enumerate().filter_map(|(variable, definition)|
            self.selectors.contains_key(&crate::msl::variable_symbol(definition, variable)).then_some(variable)).collect();
        let mut stage = execution::prepare_retained(source, config, mappings, &self.numeric_parameters, &selectors, self.retained_split.as_ref())?;
        loop {
            if let Stage::Loads(prepared) = &stage {
                let execution = self.retain_layout(builder, prepared)?;
                let template = crate::msl::GroupingTemplate::new(&execution)?;
                if let Some(layout) = &execution.implementation {
                    layout.append_native_constraints(builder, &self.name, &self.numeric_parameters)?;
                }
                let operations = crate::terminal::family::Family::with_selectors(&template.emitted().terminal, &self.selectors)?;
                self.retained = Some(execution.clone());
                return Ok(Template { execution, operations, emission: template });
            }
            let mut folds = BTreeMap::new();
            for definition in stage.local_domains()? {
                let fold = match &definition.decision { Decision::Fold(choice) => Some(choice.site), _ => None };
                let identity = self.append(builder, definition)?;
                if let Some(site) = fold { folds.insert(site, self.entries[&identity].variable); }
            }
            if let Some(retained) = super::folds::construct(&stage, &folds, builder, &self.name, &self.selectors, &self.phase_presence)? {
                for (variable, guard) in retained.predicates {
                    if guard.value != 1 { return Err("retained fold predicate is not boolean".into()); }
                    self.selectors.insert(crate::msl::variable_symbol(&retained.stage.function().vars[variable], variable), guard.variable);
                }
                self.private_guards.extend(retained.private_values);
                stage = retained.stage;
                continue;
            }
            return Err("retained Metal construction did not reach its load/layout family boundary".into());
        }
    }
    /// Consume the original local definitions against a terminal implementation
    /// retained at export time. No source transform, stage discovery, planning,
    /// or target emission is permitted at the witness boundary.
    pub fn instantiate(
        &self,
        values: &[i64],
    ) -> Result<(Execution, Vec<seismic_compiler::tuner::family::Decision>), String> {
        let execution = self.retained.as_ref()
            .ok_or("Metal implementation family has no retained terminal realization")?;
        let mut decisions = Vec::with_capacity(self.entries.len());
        for (identity, entry) in &self.entries {
            if execution.implementation.as_ref().and_then(|layout| layout.decision_presence.get(&entry.variable))
                .is_some_and(|active| values.get(active.0) != Some(&1)) { continue; }
            let value = *values.get(entry.variable.0)
                .ok_or("missing Metal implementation parameter")?;
            let ordinal = usize::try_from(value).map_err(|_| "negative Metal implementation ordinal")?;
            entry.domain.get(ordinal)
                .ok_or("Metal implementation parameter lies outside its original definition")?;
            decisions.push(seismic_compiler::tuner::family::Decision { identity: identity.clone(), value });
        }
        let mut execution = execution.clone();
        if let Some(layout) = execution.implementation.clone() {
            let unit = seismic_realization::dispatch::GroupDispatch::new(1, execution::SUBGROUP as u64, 1)?;
            execution.storage = self.storage_binding.as_ref().ok_or("retained layout has no original storage binding")?
                .reconstruct(values, &unit).map_err(|error| error.to_string())?;
            execution.memory = layout.instantiate_memory(&execution.memory, values)?;
            execution.reductions = layout.instantiate_reductions(values, &self.numeric_parameters, &execution.function.vars)?;
            decisions.extend(layout.decisions(values, &self.name)?);
        }
        Ok((execution, decisions))
    }

}
