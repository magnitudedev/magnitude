//! Original implementation choices retained as local value-binding alternatives.
//! The typed emitter consumes these bindings at the operation which needs them;
//! independent values do not enumerate complete backend implementations.
use magnitude_solver::model::{Constraint, Domain, Literal, ModelBuilder, VarId};
use seismic_lang::ir;
use seismic_realization::dispatch::TilePlacement;
use std::collections::BTreeMap;

#[derive(Clone, Debug)]
pub(crate) struct Arm<T> {
    pub predicate: String,
    pub active: VarId,
    pub value: T,
}
#[derive(Clone, Debug)]
pub(crate) struct Choice<T> {
    pub ordinal: VarId,
    pub arms: Vec<Arm<T>>,
}
impl<T: Clone> Choice<T> {
    pub(crate) fn new(builder: &mut ModelBuilder, name: &str, ordinal: VarId, alternatives: &[T]) -> Result<Self, String> {
        let mut arms = Vec::with_capacity(alternatives.len());
        for (index, value) in alternatives.iter().enumerate() {
            let predicate = format!("seismic_implementation_{}_arm{index}__", ordinal.0);
            let active = builder.local_variable(format!("{name}.arm{index}"), Domain::boolean()).map_err(|error| error.to_string())?;
            let expected = builder.variable(format!("{name}.ordinal{index}"), Domain::singleton(index as i64));
            builder.guarded_constraint(vec![Literal::new(active, 1)], Constraint::Equal { left: ordinal, right: expected });
            builder.guarded_constraint(vec![Literal::new(active, 0)], Constraint::NotEqual { left: ordinal, right: expected });
            arms.push(Arm { predicate, active, value: value.clone() });
        }
        Ok(Self { ordinal, arms })
    }
    pub(crate) fn selected(&self, values: &[i64]) -> Result<&T, String> {
        let ordinal = *values.get(self.ordinal.0).ok_or("missing retained layout ordinal")?;
        self.arms.get(usize::try_from(ordinal).map_err(|_| "negative retained layout ordinal")?)
            .map(|arm| &arm.value).ok_or_else(|| "retained layout ordinal is outside its original domain".into())
    }
    pub(crate) fn predicates(&self) -> impl Iterator<Item = (String, VarId)> + '_ {
        self.arms.iter().map(|arm| (arm.predicate.clone(), arm.active))
    }
}
#[derive(Clone, Debug, Default)]
pub(crate) struct Family {
    pub load_sites: BTreeMap<(ir::OperationId, ir::VarId), Choice<ir::LoadMode>>,
    pub storage: BTreeMap<ir::VarId, Choice<TilePlacement>>,
    pub owners: BTreeMap<ir::VarId, Choice<bool>>,
    /// Original storage ownership equations, indexed by the local emission
    /// predicates which establish an actual binding. This checks conjunctions
    /// of known facts; it neither searches nor treats missing analysis as false.
    pub ownership_requirements: BTreeMap<String, Vec<(VarId, bool)>>,
    pub reductions: BTreeMap<(ir::OperationId, ir::VarId), Choice<crate::reduction::Algorithm>>,
    /// Compile-time active binding facts while emitting one guarded local arm.
    /// Presence variables are shared with source/fold selectors, never runtime.
    pub source_predicates: BTreeMap<String, VarId>,
    /// Split prefix, narrowed stream and merge tail share a compiler guard
    /// carried by phase topology rather than a source-level conditional.
    pub split_guards: BTreeMap<ir::OperationId, Literal>,
    pub phase_guards: BTreeMap<ir::OperationId, Literal>,
    pub phase_predicates: Vec<Option<String>>,
    pub source_negations: BTreeMap<String, String>,
    pub reduction_bindings: std::sync::Arc<std::sync::Mutex<BTreeMap<(ir::OperationId, ir::VarId), Vec<ReductionBinding>>>>,
    pub reduction_definitions: BTreeMap<(ir::OperationId, ir::VarId), crate::reduction::Selected>,
    pub allocations: Vec<AllocationRequest>,
    pub impossible: std::sync::Arc<std::sync::Mutex<Vec<Vec<Literal>>>>,
    pub storage_demand: BTreeMap<ir::VarId, VarId>,
    pub decision_presence: BTreeMap<VarId, VarId>,
    pub barrier_conditions: BTreeMap<(usize, crate::memory::BarrierSite), Arm<bool>>,
    pub barrier_uses: std::sync::Arc<std::sync::Mutex<BTreeMap<(usize, crate::memory::BarrierSite), Vec<Vec<Literal>>>>>,
    pub allocation_uses: std::sync::Arc<std::sync::Mutex<BTreeMap<(usize, crate::memory::AllocationId), Vec<Vec<Literal>>>>>,
    pub native_requirements: std::sync::Arc<std::sync::Mutex<Vec<(Vec<Literal>, seismic_lang::sym::Sym)>>>,
}
impl Family {
    pub(crate) fn retain_phase_presence(&mut self, function: &seismic_lang::lowered_ir::LoweredIr,
        presence: &[Option<VarId>]) -> Result<(), String> {
        if !presence.is_empty() && presence.len() != function.body.len() { return Err("retained phase presence differs from launch topology".into()); }
        self.phase_predicates = vec![None; function.body.len()];
        for (phase, (&active, statement)) in presence.iter().zip(&function.body).enumerate() {
            let Some(active) = active else { continue; };
            let predicate = format!("seismic_phase_{}__", active.0);
            self.source_predicates.insert(predicate.clone(), active);
            self.phase_predicates[phase] = Some(predicate);
            for operation in source_guards(std::slice::from_ref(statement), &self.source_predicates, &function.vars).keys() {
                self.phase_guards.insert(*operation, Literal::new(active, 1));
            }
        }
        Ok(())
    }
    pub(crate) fn retain_split_guards(&mut self, function: &seismic_lang::lowered_ir::LoweredIr,
        phases: &[crate::execution::Phase]) -> Result<(), String> {
        for (statement, phase) in function.body.iter().zip(phases) {
            let Some(retained) = phase.split.as_ref().and_then(|split| split.retained.as_ref()) else { continue; };
            let ir::StmtKind::Parallel { body, .. } = &statement.kind else { return Err("retained split lost its phase body".into()); };
            let split_body = body.get(..retained.ordinary_at).ok_or("retained split ordinary boundary is invalid")?;
            let active = *self.source_predicates.get(&retained.selector).ok_or("retained split has no original compiler selector")?;
            if let Some(operation) = statement.id { self.split_guards.insert(operation, Literal::new(active, 1)); }
            for operation in source_guards(split_body, &self.source_predicates, &function.vars).keys() {
                self.split_guards.insert(*operation, Literal::new(active, 1));
            }
        }
        Ok(())
    }
    fn operation_guards(&self, function: &seismic_lang::lowered_ir::LoweredIr) -> BTreeMap<ir::OperationId, Vec<Literal>> {
        let mut guards = source_guards(&function.body, &self.source_predicates, &function.vars);
        for (&operation, &guard) in &self.phase_guards { guards.entry(operation).or_default().push(guard); }
        for (&operation, &guard) in &self.split_guards { guards.entry(operation).or_default().push(guard); }
        guards
    }
    pub(crate) fn bind_source(&mut self, builder: &mut ModelBuilder, name: &str) -> Result<(), String> {
        for (symbol, original) in self.source_predicates.clone() {
            let complement = builder.local_variable(format!("{name}.source_not{}", original.0), Domain::boolean()).map_err(|error| error.to_string())?;
            builder.constraint(Constraint::LinearLe { terms: vec![magnitude_solver::model::LinearTerm::new(original, 1), magnitude_solver::model::LinearTerm::new(complement, 1)], rhs: 1 });
            builder.constraint(Constraint::LinearLe { terms: vec![magnitude_solver::model::LinearTerm::new(original, -1), magnitude_solver::model::LinearTerm::new(complement, -1)], rhs: -1 });
            let negative = format!("seismic_source_not_{}__", original.0);
            self.source_predicates.insert(negative.clone(), complement);
            self.source_negations.insert(symbol, negative);
        }
        Ok(())
    }
    pub(crate) fn variable_guards(&self, builder: &mut ModelBuilder, name: &str,
        function: &seismic_lang::lowered_ir::LoweredIr) -> Result<BTreeMap<ir::VarId, Vec<Literal>>, String> {
        let guards = self.operation_guards(function);
        fn visit(body: &[ir::Stmt], guards: &BTreeMap<ir::OperationId, Vec<Literal>>, out: &mut BTreeMap<ir::VarId, Vec<Vec<Literal>>>) {
            for statement in body {
                let guard = statement.id.and_then(|operation| guards.get(&operation)).cloned().unwrap_or_default();
                let mut define = |variable| { out.entry(variable).or_default().push(guard.clone()); };
                match &statement.kind {
                    ir::StmtKind::Assign { target: ir::Expr { kind: ir::ExprKind::Var(variable), .. }, .. } => define(*variable),
                    ir::StmtKind::LoadLoop { vars, body, .. } => { for &variable in vars { define(variable); } visit(body, guards, out); },
                    ir::StmtKind::Parallel { body, .. } | ir::StmtKind::Owned { body, .. } | ir::StmtKind::Range { body, .. } | ir::StmtKind::Lanes { body, .. } => visit(body, guards, out),
                    ir::StmtKind::If { then, els, .. } => { visit(then, guards, out); visit(els, guards, out); },
                    _ => {},
                }
            }
        }
        let mut definitions = BTreeMap::new(); visit(&function.body, &guards, &mut definitions);
        definitions.into_iter().map(|(variable, arms)| {
            let active = disjunction(builder, &format!("{name}.variable{variable}.present"), &arms)?;
            Ok((variable, vec![Literal::new(active, 1)]))
        }).collect()
    }
    /// Close original request presence over the exact guarded native uses. The
    /// emitter records local binding guards before backing-owner alternatives;
    /// repeated source definitions therefore retain their own load choices.
    pub(crate) fn storage_guards(&mut self, builder: &mut ModelBuilder, name: &str,
        definitions: &BTreeMap<ir::VarId, Vec<Literal>>) -> Result<BTreeMap<ir::VarId, Vec<Literal>>, String> {
        let mut guards = definitions.clone();
        for &variable in self.storage.keys() {
            let demand = builder.local_variable(format!("{name}.storage.variable{variable}.demand"), Domain::boolean()).map_err(|error| error.to_string())?;
            guards.entry(variable).or_default().push(Literal::new(demand, 1));
            self.storage_demand.insert(variable, demand);
        }
        Ok(guards)
    }
    pub(crate) fn bind_owners(&mut self, builder: &mut ModelBuilder, name: &str,
        storage: &crate::storage::family::Binding) -> Result<(), String> {
        let mut groups = BTreeMap::<VarId, Choice<bool>>::new();
        for (&variable, &owner) in storage.owners() {
            let choice = match groups.get(&owner) {
                Some(choice) => choice.clone(),
                None => {
                    let choice = Choice::new(builder, &format!("{name}.ownership{}", owner.0), owner, &[false, true])?;
                    groups.insert(owner, choice.clone()); choice
                }
            };
            for arm in &choice.arms {
                self.ownership_requirements.insert(arm.predicate.clone(), vec![(owner, arm.value)]);
            }
            self.owners.insert(variable, choice);
        }
        for (&variable, placement) in &storage.placements {
            let choice = self.storage.get(&variable).ok_or("storage ownership lost its placement choice")?;
            if choice.arms.len() != placement.ownership.len() { return Err("storage ownership alternative count differs".into()); }
            for (arm, requirements) in choice.arms.iter().zip(&placement.ownership) {
                self.ownership_requirements.insert(arm.predicate.clone(), requirements.clone());
            }
        }
        Ok(())
    }
    pub(crate) fn compatible_ownership(&self, active: &BTreeMap<String, bool>) -> bool {
        let mut values = BTreeMap::new();
        for (predicate, requirements) in &self.ownership_requirements {
            if active.get(predicate) != Some(&true) { continue; }
            for &(variable, required) in requirements {
                if values.insert(variable, required).is_some_and(|previous| previous != required) {
                    return false;
                }
            }
        }
        true
    }
    pub(crate) fn append_native_constraints(&self, builder: &mut ModelBuilder, name: &str,
        numeric: &BTreeMap<String, seismic_accounting::algebra::Value>) -> Result<(), String> {
        let uses = self.allocation_uses.lock().map_err(|_| "retained allocation uses were poisoned")?;
        for (index, request) in self.allocations.iter().enumerate() {
            let alternatives = uses.get(&(request.launch, request.allocation.id)).map(Vec::as_slice).unwrap_or(&[]);
            let active = disjunction(builder, &format!("{name}.allocation{index}.used"), alternatives)?;
            builder.constraint(Constraint::Equal { left: request.active, right: active });
        }
        for (&variable, &demand) in &self.storage_demand {
            let arms = self.allocations.iter().filter(|request| request.allocation.id.variable == variable)
                .map(|request| vec![Literal::new(request.active, 1)]).collect::<Vec<_>>();
            let present = disjunction(builder, &format!("{name}.storage.variable{variable}.used"), &arms)?;
            builder.constraint(Constraint::Equal { left: demand, right: present });
        }
        for guard in self.impossible.lock().map_err(|_| "retained layout applicability was poisoned")?.iter() {
            builder.guarded_constraint(guard.clone(), Constraint::LinearLe { terms: Vec::new(), rhs: -1 });
        }
        for (index, (guards, expression)) in self.native_requirements.lock().map_err(|_| "retained native requirements were poisoned")?.iter().enumerate() {
            let active = conjunction(builder, &format!("{name}.native_requirement{index}.active"), guards)?;
            let mut expressions = seismic_compiler::tuner::expressions::Expressions::new(numeric.clone());
            let remainder = builder.when(Literal::new(active, 1), |builder| expressions.nonnegative(builder,
                &format!("{name}.native_requirement{index}"), expression)).map_err(|error| error.to_string())?;
            builder.guarded_constraint(vec![Literal::new(active, 1)], Constraint::InDomain { variable: remainder.id(), domain: Domain::singleton(0) });
        }
        Ok(())
    }
    pub(crate) fn bind_activation(&mut self, builder: &mut ModelBuilder, name: &str,
        function: &seismic_lang::lowered_ir::LoweredIr, variables: &BTreeMap<ir::VarId, Vec<Literal>>) -> Result<(), String> {
        let operations = self.operation_guards(function);
        let choices = self.load_sites.iter().map(|(&(operation, _), choice)| (choice.ordinal, operations.get(&operation).cloned().unwrap_or_default()))
            .chain(self.reductions.iter().map(|(&(operation, _), choice)| (choice.ordinal, operations.get(&operation).cloned().unwrap_or_default())))
            .chain(self.storage.iter().map(|(&variable, choice)| (choice.ordinal, variables.get(&variable).cloned().unwrap_or_default()))).collect::<Vec<_>>();
        for (ordinal, guard) in choices {
            let active = conjunction(builder, &format!("{name}.choice{}.active", ordinal.0), &guard)?;
            builder.constraint(Constraint::InactiveValue { active: Literal::new(active, 1), variable: ordinal, inactive: 0 });
            self.decision_presence.insert(ordinal, active);
        }
        Ok(())
    }
    pub(crate) fn append_reductions(&mut self, builder: &mut ModelBuilder, name: &str, plan: &crate::reduction::ReductionPlan) -> Result<(), String> {
        use crate::reduction::Algorithm;
        let mut definitions = plan.selections().values().cloned().collect::<Vec<_>>();
        definitions.sort_by_key(|selected| selected.decision.site.operation);
        for selected in definitions {
            let definition = &selected.decision;
            let site = (definition.site.operation, definition.site.output);
            let argmax = definition.contract.operation == ir::ReduceOp::Argmax;
            let ordered = definition.contract.ordered || matches!(definition.contract.input, seismic_lang::types::DType::F16 | seismic_lang::types::DType::BF16)
                || definition.contract.combination() == seismic_lang::reduction::Combination::SaturatingAdd;
            let placements = self.storage.get(&definition.input).map(|choice| choice.arms.iter().map(|arm| Some(arm.value.clone())).collect::<Vec<_>>())
                .unwrap_or_else(|| vec![definition.input_placement.clone(), Some(TilePlacement::Replicated), Some(TilePlacement::Distributed)]);
            let mut algorithms = Vec::new();
            for placement in placements.into_iter().chain(std::iter::once(None)) {
                let domain = definition.domain.placement_variant(placement, argmax, true, definition.contract.input, ordered)?;
                for &algorithm in domain.algorithms() { if !algorithms.contains(&algorithm) { algorithms.push(algorithm); } }
            }
            algorithms.sort_by_key(|algorithm| match algorithm { Algorithm::Ordered => 0, Algorithm::LaneLocal => 1, Algorithm::Collective => 2 });
            let label = format!("{name}.reduction.operation{}.output{}", site.0.0, site.1);
            let ordinal = builder.local_variable(label.clone(), Domain::interval(0, algorithms.len().checked_sub(1).ok_or("empty retained reduction family")? as i64).map_err(|error| error.to_string())?).map_err(|error| error.to_string())?;
            let choice = Choice::new(builder, &label, ordinal, &algorithms)?;
            self.reductions.insert(site, choice);
            self.reduction_definitions.insert(site, selected);
        }
        Ok(())
    }
    pub(crate) fn predicates(&self) -> BTreeMap<String, VarId> {
        let mut predicates = self.source_predicates.clone();
        for choice in self.load_sites.values() { predicates.extend(choice.predicates()); }
        for choice in self.storage.values() { predicates.extend(choice.predicates()); }
        for choice in self.owners.values() { predicates.extend(choice.predicates()); }
        for choice in self.reductions.values() { predicates.extend(choice.predicates()); }
        for barrier in self.barrier_conditions.values() { predicates.insert(barrier.predicate.clone(), barrier.active); }
        for allocation in &self.allocations {
            predicates.extend(allocation.owner.predicates());
            predicates.extend(allocation.placements.iter().map(|arm| (arm.predicate.clone(), arm.active)));
            predicates.insert(format!("seismic_reuse_{}__", allocation.shared_reuse.0), allocation.shared_reuse);
            predicates.insert(format!("seismic_allocation_active_{}__", allocation.active.0), allocation.active);
        }
        predicates
    }
}

#[derive(Clone, Debug)]
pub(crate) struct ReductionBinding {
    pub guards: Vec<Literal>,
    pub selected: crate::reduction::Selected,
    pub shape: Vec<seismic_lang::sym::Sym>,
    pub axis: usize,
    pub lane_domains: Vec<(seismic_lang::sym::Sym, i64)>,
}
#[derive(Clone, Debug)]
pub(crate) struct AllocationRequest {
    pub launch: usize,
    pub allocation: crate::memory::ArrayAllocation,
    pub active: VarId,
    pub guards: Vec<Literal>,
    pub capacity: seismic_accounting::algebra::Value,
    pub placements: Vec<Arm<TilePlacement>>,
    pub owner: Choice<usize>,
    pub backing_capacity: seismic_accounting::algebra::Value,
    pub shared_reuse: VarId,
}
pub(super) fn conjunction(builder: &mut ModelBuilder, name: &str, guards: &[Literal]) -> Result<VarId, String> {
    let mut inputs = Vec::new();
    for guard in guards {
        if guard.value == 1 { inputs.push(guard.variable); continue; }
        let variable = builder.local_variable(format!("{name}.predicate{}", inputs.len()), Domain::boolean()).map_err(|error| error.to_string())?;
        let expected = builder.variable(format!("{name}.literal{}", inputs.len()), Domain::singleton(guard.value));
        builder.guarded_constraint(vec![Literal::new(variable, 1)], Constraint::Equal { left: guard.variable, right: expected });
        builder.guarded_constraint(vec![Literal::new(variable, 0)], Constraint::NotEqual { left: guard.variable, right: expected });
        inputs.push(variable);
    }
    let output = builder.local_variable(name.to_owned(), Domain::boolean()).map_err(|error| error.to_string())?;
    if inputs.is_empty() { builder.constraint(Constraint::InDomain { variable: output, domain: Domain::singleton(1) }); }
    else { builder.constraint(Constraint::BoolAnd { output, inputs }); }
    Ok(output)
}
fn disjunction(builder: &mut ModelBuilder, name: &str, guards: &[Vec<Literal>]) -> Result<VarId, String> {
    let inputs = guards.iter().enumerate().map(|(index, guard)| conjunction(builder, &format!("{name}.arm{index}"), guard)).collect::<Result<Vec<_>, _>>()?;
    let output = builder.local_variable(name.to_owned(), Domain::boolean()).map_err(|error| error.to_string())?;
    let mut terms = vec![magnitude_solver::model::LinearTerm::new(output, 1)];
    for input in inputs {
        builder.constraint(Constraint::Implies { premise: Literal::new(input, 1), consequence: Literal::new(output, 1) });
        terms.push(magnitude_solver::model::LinearTerm::new(input, -1));
    }
    builder.constraint(Constraint::LinearLe { terms, rhs: 0 });
    Ok(output)
}
fn source_guards(body: &[ir::Stmt], predicates: &BTreeMap<String, VarId>, vars: &[ir::Var]) -> BTreeMap<ir::OperationId, Vec<Literal>> {
    fn visit(body: &[ir::Stmt], predicates: &BTreeMap<String, VarId>, vars: &[ir::Var], enclosing: &[Literal], output: &mut BTreeMap<ir::OperationId, Vec<Literal>>) {
        for statement in body {
            if let Some(operation) = statement.id { output.insert(operation, enclosing.to_vec()); }
            match &statement.kind {
                ir::StmtKind::Parallel { body, .. } | ir::StmtKind::Owned { body, .. } | ir::StmtKind::Range { body, .. }
                | ir::StmtKind::Lanes { body, .. } | ir::StmtKind::LoadLoop { body, .. } => visit(body, predicates, vars, enclosing, output),
                ir::StmtKind::If { cond, then, els } => {
                    let predicate = match cond.kind { ir::ExprKind::Var(variable) => predicates.get(&crate::msl::variable_symbol(&vars[variable], variable)), _ => None };
                    if let Some(&predicate) = predicate {
                        let mut yes = enclosing.to_vec(); yes.push(Literal::new(predicate, 1)); visit(then, predicates, vars, &yes, output);
                        let mut no = enclosing.to_vec(); no.push(Literal::new(predicate, 0)); visit(els, predicates, vars, &no, output);
                    } else { visit(then, predicates, vars, enclosing, output); visit(els, predicates, vars, enclosing, output); }
                }
                _ => {},
            }
        }
    }
    let mut output = BTreeMap::new(); visit(body, predicates, vars, &[], &mut output); output
}
impl Family {
    pub(crate) fn append_allocations(&mut self, builder: &mut ModelBuilder, name: &str, memory: &crate::memory::MemoryPlan,
        function: &seismic_lang::lowered_ir::LoweredIr, storage: &crate::storage::family::Binding,
        template: &crate::storage::StoragePlan, numeric: &BTreeMap<String, seismic_accounting::algebra::Value>) -> Result<(), String> {
        use seismic_accounting::algebra::{Algebra, Symbolic, Value};
        let presence = self.operation_guards(function);
        let zero = Symbolic::new(builder, name).constant(0).map_err(|error| error.to_string())?;
        fn publications(body: &[ir::Stmt], output: &mut BTreeMap<ir::OperationId, Vec<ir::VarId>>) {
            for statement in body {
                match &statement.kind {
                    ir::StmtKind::Owned { tile, body, .. } => {
                        let mut variables = std::collections::HashSet::new();
                        for statement in body { seismic_lang::rewrite::writes(statement, &mut variables); }
                        if let Some(variable) = crate::storage::tile_root(tile) { variables.insert(variable); }
                        if let Some(operation) = statement.id { output.insert(operation, variables.into_iter().collect()); }
                        publications(body, output);
                    }
                    ir::StmtKind::Parallel { body, .. } | ir::StmtKind::Range { body, .. } | ir::StmtKind::Lanes { body, .. }
                    | ir::StmtKind::LoadLoop { body, .. } => publications(body, output),
                    ir::StmtKind::If { then, els, .. } => { publications(then, output); publications(els, output); },
                    _ => {},
                }
            }
        }
        let mut publication_values = BTreeMap::new(); publications(&function.body, &mut publication_values);
        for (launch, plan) in memory.launches().iter().enumerate() {
            for &site in plan.barriers.keys() {
                if matches!(site.purpose, crate::memory::BarrierPurpose::Reuse(_)) { continue; }
                let variables = if site.purpose == crate::memory::BarrierPurpose::Owned {
                    publication_values.get(&site.operation).cloned().unwrap_or_else(|| vec![site.variable])
                } else { vec![site.variable] };
                let alternatives = variables.into_iter().filter_map(|variable| self.storage.get(&variable))
                    .flat_map(|choice| choice.arms.iter().filter(|arm| arm.value == TilePlacement::GroupShared)
                        .map(|arm| vec![Literal::new(arm.active, 1)])).collect::<Vec<_>>();
                let active = disjunction(builder, &format!("{name}.launch{launch}.barrier{site:?}"), &alternatives)?;
                self.barrier_conditions.insert((launch, site), Arm { predicate: format!("seismic_publication_{}__", active.0), active, value: true });
            }
            let begin = self.allocations.len();
            for array in &plan.arrays {
                let index = self.allocations.len();
                let label = format!("{name}.launch{launch}.allocation{index}");
                let mut guards = presence.get(&array.id.operation).cloned().unwrap_or_default();
                if matches!(array.id.purpose, crate::memory::Purpose::Value | crate::memory::Purpose::PacketPlane(_)) {
                    if let Some(load) = self.load_sites.get(&(array.id.operation, array.id.variable)) {
                        let snapshot = load.arms.iter().find(|arm| arm.value == ir::LoadMode::Materialize).ok_or("load lost its snapshot arm")?;
                        guards.push(Literal::new(snapshot.active, 1));
                    }
                }
                let active = builder.local_variable(format!("{label}.present"), Domain::boolean()).map_err(|error| error.to_string())?;
                for &guard in &guards { builder.constraint(Constraint::Implies { premise: Literal::new(active, 1), consequence: guard }); }
                let capacity = if let Some(physical) = storage.layouts.get(&array.id.variable) {
                    match array.id.purpose {
                        crate::memory::Purpose::PacketPlane(plane) => physical.packets.as_ref().and_then(|packet| packet.planes.get(plane))
                            .map(|plane| plane.elements).ok_or("packet allocation lost its original geometry")?,
                        _ => physical.capacity,
                    }
                } else {
                    let shape = function.vars[array.id.variable].ty.shaped().ok_or("allocation has no typed logical shape")?;
                    let count = shape.shape.iter().fold(seismic_lang::sym::Sym::constant(1), |count, extent| count.mul(&template.capacity_expression(extent)));
                    let mut expressions = seismic_compiler::tuner::expressions::Expressions::new(numeric.clone());
                    let expression = builder.when(Literal::new(active, 1), |builder| expressions.nonnegative(builder, &label, &count)).map_err(|error| error.to_string())?;
                    let capacity = Symbolic::new(builder, &label).variable("capacity", Domain::interval(0, expression.bounds().1 as i64).map_err(|error| error.to_string())?).map_err(|error| error.to_string())?;
                    builder.guarded_constraint(vec![Literal::new(active, 1)], Constraint::Equal { left: capacity.id(), right: expression.id() });
                    builder.constraint(Constraint::InactiveValue { active: Literal::new(active, 1), variable: capacity.id(), inactive: 0 });
                    capacity
                };
                let placements = if let Some(choice) = self.storage.get(&array.id.variable) { choice.arms.clone() }
                    else if let Some(choice) = self.reductions.get(&(array.id.operation, array.id.variable)) {
                        let mut variants = Vec::new();
                        for placement in [TilePlacement::Replicated, TilePlacement::Distributed] {
                            let inputs = choice.arms.iter().filter(|arm| (arm.value == crate::reduction::Algorithm::LaneLocal) == (placement == TilePlacement::Distributed))
                                .map(|arm| arm.active).collect::<Vec<_>>();
                            if inputs.is_empty() { continue; }
                            let active = if inputs.len() == 1 { inputs[0] } else {
                                let active = builder.local_variable(format!("{label}.output_placement{placement:?}"), Domain::boolean()).map_err(|error| error.to_string())?;
                                for &input in &inputs { builder.constraint(Constraint::Implies { premise: Literal::new(input, 1), consequence: Literal::new(active, 1) }); }
                                let mut terms = vec![magnitude_solver::model::LinearTerm::new(active, 1)];
                                terms.extend(inputs.iter().map(|&input| magnitude_solver::model::LinearTerm::new(input, -1)));
                                builder.constraint(Constraint::LinearLe { terms, rhs: 0 }); active
                            };
                            variants.push(Arm { predicate: format!("seismic_output_placement_{}__", active.0), active, value: placement });
                        }
                        variants
                    } else {
                        let present = builder.variable(format!("{label}.placement_fixed"), Domain::singleton(1));
                        vec![Arm { predicate: format!("seismic_fixed_{}__", present.0), active: present, value: array.declaration.placement.clone() }]
                    };
                let alternatives = (begin..index).filter(|&other| {
                    let previous = &self.allocations[other].allocation;
                    previous.declaration.dtype == array.declaration.dtype && !previous.lifetime.overlaps(array.lifetime)
                }).chain(std::iter::once(index)).collect::<Vec<_>>();
                let ordinal = builder.local_variable(format!("{label}.backing"), Domain::interval(0, alternatives.len() as i64 - 1).map_err(|error| error.to_string())?).map_err(|error| error.to_string())?;
                let owner = Choice::new(builder, &format!("{label}.backing"), ordinal, &alternatives)?;
                builder.constraint(Constraint::InactiveValue { active: Literal::new(active, 1), variable: ordinal, inactive: alternatives.len() as i64 - 1 });
                let shared_reuse = builder.local_variable(format!("{label}.shared_reuse"), Domain::boolean()).map_err(|error| error.to_string())?;
                builder.constraint(Constraint::InactiveValue { active: Literal::new(active, 1), variable: shared_reuse, inactive: 0 });
                self.allocations.push(AllocationRequest { launch, allocation: array.clone(), active, guards, capacity, placements, owner, backing_capacity: zero, shared_reuse });
            }
            let end = self.allocations.len();
            for index in begin..end {
                let request = &self.allocations[index];
                for arm in &request.owner.arms {
                    if arm.value == index { continue; }
                    let backing = &self.allocations[arm.value];
                    let root = backing.owner.arms.iter().find(|candidate| candidate.value == arm.value).ok_or("backing owner lost its defining arm")?;
                    let guard = vec![Literal::new(request.active, 1), Literal::new(arm.active, 1)];
                    builder.guarded_constraint(guard.clone(), Constraint::InDomain { variable: backing.active, domain: Domain::singleton(1) });
                    builder.guarded_constraint(guard.clone(), Constraint::InDomain { variable: root.active, domain: Domain::singleton(1) });
                    for left in &request.placements { for right in &backing.placements {
                        if left.value != right.value || (left.value == TilePlacement::GroupShared && (!request.allocation.uniform || !backing.allocation.uniform)) {
                            let mut guard = guard.clone(); guard.push(Literal::new(left.active, 1)); guard.push(Literal::new(right.active, 1));
                            builder.guarded_constraint(guard, Constraint::LinearLe { terms: Vec::new(), rhs: -1 });
                        }
                    } }
                }
                for other in begin..index {
                    if !request.allocation.lifetime.overlaps(self.allocations[other].allocation.lifetime) { continue; }
                    for left in &request.owner.arms { for right in &self.allocations[other].owner.arms {
                        if left.value == right.value {
                            builder.guarded_constraint(vec![Literal::new(request.active, 1), Literal::new(self.allocations[other].active, 1), Literal::new(left.active, 1), Literal::new(right.active, 1)],
                                Constraint::LinearLe { terms: Vec::new(), rhs: -1 });
                        }
                    } }
                }
            }
            for root in begin..end {
                let mut capacity = zero;
                let mut members = Vec::new();
                for index in root..end {
                    let request = &self.allocations[index];
                    let Some(owner) = request.owner.arms.iter().find(|arm| arm.value == root) else { continue; };
                    let active = conjunction(builder, &format!("{name}.backing{root}.member{index}"), &[Literal::new(request.active, 1), Literal::new(owner.active, 1)])?;
                    let mut algebra = Symbolic::new(builder, name);
                    let contribution = algebra.variable(&format!("backing{root}.capacity{index}"), Domain::interval(0, request.capacity.bounds().1 as i64).map_err(|error| error.to_string())?).map_err(|error| error.to_string())?;
                    builder.guarded_constraint(vec![Literal::new(active, 1)], Constraint::Equal { left: contribution.id(), right: request.capacity.id() });
                    builder.guarded_constraint(vec![Literal::new(active, 0)], Constraint::InDomain { variable: contribution.id(), domain: Domain::singleton(0) });
                    capacity = Symbolic::new(builder, name).maximum(capacity, contribution).map_err(|error| error.to_string())?;
                    members.push((index, active));
                }
                self.allocations[root].backing_capacity = capacity;
                for &(index, active) in &members {
                    let others = members.iter().filter(|(other, _)| *other != index).map(|(_, member)| *member).collect::<Vec<_>>();
                    let output = self.allocations[index].shared_reuse;
                    // A request may reference several possible roots. Under its
                    // selected root, reuse is exactly whether another live
                    // allocation shares that root, including later loop visits.
                    for &other in &others {
                        builder.guarded_constraint(vec![Literal::new(active, 1), Literal::new(other, 1)], Constraint::InDomain { variable: output, domain: Domain::singleton(1) });
                    }
                    let mut guard = vec![Literal::new(active, 1)]; guard.extend(others.iter().map(|&other| Literal::new(other, 0)));
                    builder.guarded_constraint(guard, Constraint::InDomain { variable: output, domain: Domain::singleton(0) });
                }
            }
        }
        Ok(())
    }
}
impl Family {
    pub(crate) fn instantiate_reductions(&self, values: &[i64], parameters: &BTreeMap<String, seismic_accounting::algebra::Value>,
        variables: &[ir::Var]) -> Result<crate::reduction::ReductionPlan, String> {
        let parameters = parameters.iter().map(|(name, value)| values.get(value.id().0).copied().map(|value| (name.clone(), value))
            .ok_or("missing retained reduction shape parameter")).collect::<Result<BTreeMap<_, _>, _>>()?;
        let mut plan = crate::reduction::ReductionPlan::default();
        let bindings = self.reduction_bindings.lock().map_err(|_| "retained reduction bindings were poisoned")?;
        for (&site, choice) in &self.reductions {
            if self.decision_presence.get(&choice.ordinal).is_some_and(|active| values.get(active.0) != Some(&1)) { continue; }
            let binding = bindings.get(&site).and_then(|arms| arms.iter().find(|binding| binding.guards.iter()
                .all(|guard| values.get(guard.variable.0) == Some(&guard.value))))
                .ok_or("selected reduction has no retained native input binding")?;
            let mut selected = binding.selected.clone();
            let shape = binding.shape.iter().map(|extent| extent.eval(&|name| parameters.get(name).copied())
                .ok_or("selected reduction capacity is unresolved")).collect::<Result<Vec<_>, _>>()?;
            selected.decision.full_lanes &= binding.lane_domains.iter().all(|(extent, run)|
                extent.eval(&|name| parameters.get(name).copied()).is_some_and(|extent| extent % run == 0));
            let contract = selected.decision.contract;
            let ordered = contract.ordered || matches!(contract.input, seismic_lang::types::DType::F16 | seismic_lang::types::DType::BF16)
                || contract.combination() == seismic_lang::reduction::Combination::SaturatingAdd;
            selected.decision.domain = if contract.operation == ir::ReduceOp::Argmax {
                crate::reduction::ReductionDomain::argmax(&shape, binding.axis, contract.input, selected.decision.input_placement.clone(),
                    selected.decision.full_lanes, crate::execution::SUBGROUP as u64)?
            } else {
                crate::reduction::ReductionDomain::new(&shape, binding.axis, contract.input, ordered,
                    selected.decision.input_placement.clone().ok_or("selected reduction has no owned input")?, crate::execution::SUBGROUP as u64)?
            };
            selected.output = selected.decision.domain.output(selected.algorithm, variables[site.1].name.clone())?;
            plan = plan.with_selection(selected);
        }
        Ok(plan)
    }
    pub(crate) fn instantiate_memory(&self, template: &crate::memory::MemoryPlan, values: &[i64]) -> Result<crate::memory::MemoryPlan, String> {
        use crate::memory::{Barrier, BarrierPurpose, BarrierSite, MemorySpace};
        use seismic_realization::dispatch::{GroupDispatch, TileDeclaration};
        let read = |variable: VarId| values.get(variable.0).copied().ok_or("missing retained allocation assignment".to_string());
        let mut launches = template.launches().to_vec();
        for (launch_index, launch) in launches.iter_mut().enumerate() {
            let requests = self.allocations.iter().enumerate().filter(|(_, request)| request.launch == launch_index && read(request.active) == Ok(1)).collect::<Vec<_>>();
            let mut owners = BTreeMap::new();
            let mut slots = Vec::new();
            for &(index, request) in &requests {
                if *request.owner.selected(values)? != index { continue; }
                let placement = request.placements.iter().find(|arm| read(arm.active) == Ok(1)).ok_or("selected allocation has no placement")?.value.clone();
                let capacity = u64::try_from(read(request.backing_capacity.id())?).map_err(|_| "negative retained backing capacity")?;
                owners.insert(index, slots.len());
                slots.push(TileDeclaration { symbol: Self::backing_symbol(index, &placement), dtype: request.allocation.declaration.dtype, capacity, placement });
            }
            let mut arrays = Vec::new();
            let uses = self.barrier_uses.lock().map_err(|_| "retained publication uses were poisoned")?;
            let mut barriers = launch.barriers.iter().filter(|(site, _)| uses.get(&(launch_index, **site)).is_some_and(|arms|
                arms.iter().any(|guard| guard.iter().all(|literal| read(literal.variable) == Ok(literal.value)))))
                .map(|(site, barrier)| (*site, barrier.clone())).collect::<BTreeMap<_, _>>();
            for &(_, request) in &requests {
                let mut array = request.allocation.clone();
                array.declaration.placement = request.placements.iter().find(|arm| read(arm.active) == Ok(1)).ok_or("allocation placement assignment is missing")?.value.clone();
                array.declaration.capacity = u64::try_from(read(request.capacity.id())?).map_err(|_| "negative retained array capacity")?;
                array.slot = *owners.get(request.owner.selected(values)?).ok_or("selected allocation refers to an absent backing")?;
                if array.declaration.placement == TilePlacement::GroupShared {
                    if read(request.shared_reuse)? == 1 {
                        barriers.insert(BarrierSite { operation: array.id.operation, variable: array.id.variable,
                            purpose: BarrierPurpose::Reuse(array.id.purpose) }, Barrier { memory: MemorySpace::Threadgroup,
                                scope: array.scope.clone(), executions: array.executions.clone() });
                    }
                }
                arrays.push(array);
            }
            launch.arrays = arrays;
            launch.slots = slots;
            launch.barriers = barriers;
            let dispatch = GroupDispatch::new(1, crate::execution::SUBGROUP as u64, 1)?;
            let mut private = 0u64; let mut shared = 0u64;
            for slot in &launch.slots {
                let layout = slot.layout(&dispatch)?;
                private = private.checked_add(layout.private_bytes_per_lane).ok_or("selected private backing overflow")?;
                shared = shared.checked_add(layout.shared_bytes_per_group).ok_or("selected shared backing overflow")?;
            }
            launch.declared_private_bytes_per_lane = private;
            launch.shared_bytes_per_group = shared;
        }
        crate::memory::MemoryPlan::new(launches, template.scratch().to_vec())
    }
    pub(crate) fn backing_symbol(index: usize, placement: &TilePlacement) -> String {
        let class = match placement { TilePlacement::Replicated => "replicated", TilePlacement::Distributed => "distributed", TilePlacement::GroupShared => "shared" };
        format!("seismic_backing_{index}_{class}__")
    }
    pub(crate) fn decisions(&self, values: &[i64], name: &str) -> Result<Vec<seismic_compiler::tuner::family::Decision>, String> {
        let mut decisions = Vec::new();
        for (&(operation, output), choice) in &self.reductions {
            if self.decision_presence.get(&choice.ordinal).is_some_and(|active| values.get(active.0) != Some(&1)) { continue; }
            choice.selected(values)?;
            decisions.push(seismic_compiler::tuner::family::Decision { identity: format!("{name}.reduction.operation{}.output{output}", operation.0),
                value: values[choice.ordinal.0] });
        }
        for request in &self.allocations {
            if values.get(request.active.0) == Some(&1) {
                request.owner.selected(values)?;
                decisions.push(seismic_compiler::tuner::family::Decision { identity: format!("{name}.allocation.{:?}", request.allocation.id), value: values[request.owner.ordinal.0] });
            }
        }
        Ok(decisions)
    }
}
/// Load identity includes its defining operation. Repeated assignments to the
/// same source value and copies in distinct retained arms keep separate choices.
pub(crate) fn load_occurrences(body: &[ir::Stmt]) -> Vec<(ir::OperationId, ir::VarId)> {
    fn visit(body: &[ir::Stmt], output: &mut Vec<(ir::OperationId, ir::VarId)>) {
        for statement in body {
            match &statement.kind {
                ir::StmtKind::Assign { target: ir::Expr { kind: ir::ExprKind::Var(variable), .. }, value, .. }
                    if matches!(value.kind, ir::ExprKind::Load { .. } | ir::ExprKind::Builtin { name: ir::Builtin::Load, .. }) => {
                    if let Some(operation) = statement.id { output.push((operation, *variable)); }
                }
                ir::StmtKind::LoadLoop { vars, body, .. } => {
                    if let Some(operation) = statement.id { output.extend(vars.iter().map(|&variable| (operation, variable))); }
                    visit(body, output);
                }
                ir::StmtKind::Parallel { body, .. } | ir::StmtKind::Owned { body, .. } | ir::StmtKind::Range { body, .. }
                | ir::StmtKind::Lanes { body, .. } => visit(body, output),
                ir::StmtKind::If { then, els, .. } => { visit(then, output); visit(els, output); },
                _ => {},
            }
        }
    }
    let mut output = Vec::new(); visit(body, &mut output); output
}
