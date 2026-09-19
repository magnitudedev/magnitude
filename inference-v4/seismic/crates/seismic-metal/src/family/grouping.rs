//! Parameterized launch membership and resident lifetimes. The operation graph
//! comes from the same typed terminal template used by native reconstruction.
//! Only dynamic occurrences are expanded; implementation assignments are not.
use super::decomposition::Launch;
use crate::{execution::Execution, model, msl::GroupingTemplate};
use magnitude_solver::model::{Arithmetic, Constraint, Domain, LinearTerm, Literal, ModelBuilder, VarId};
use magnitude_solver::scheduling::{Demand, Event, Lifetime, SchedulingConstraint};
use seismic_accounting::{algebra::{Algebra, ResourceUse, Symbolic, Value}, objective::Objective, schedule::{self, symbolic::Encoding}, workload::{DerivationError, DerivationLimits, ScalarWorkload}};
use seismic_realization::dispatch::GroupDispatch;
use std::{collections::BTreeMap, sync::Arc};

struct Edge { before: usize, after: usize, presence: Option<VarId> }
struct Resident { begin: usize, end: usize, resource: usize, units: Value, presence: Option<VarId> }
/// A contribution retains native reconstruction and operation identities before
/// the caller chooses the single horizon/resource boundary for the whole family.
pub(crate) struct Prepared {
    name: String,
    template: GroupingTemplate,
    terminal: Option<Arc<crate::terminal::family::Binding>>,
    original: schedule::Model,
    presence: Vec<Option<VarId>>,
    edges: Vec<Edge>,
    residents: Vec<Resident>,
}
pub(crate) struct Binding {
    template: GroupingTemplate,
    terminal: Option<Arc<crate::terminal::family::Binding>>,
    original: schedule::Model,
    presence: Vec<Option<VarId>>,
    starts: Vec<VarId>,
    edges: Vec<Edge>,
    residents: Vec<Resident>,
}
fn error(error: impl ToString) -> String { error.to_string() }
fn boolean(builder: &mut ModelBuilder, name: &str) -> VarId { builder.variable(name, Domain::boolean()) }
/// Fixed hardware quantities have no activation-dependent value. Sharing their
/// singleton bindings avoids both duplicate variables and inactive-value factors.
#[derive(Default)]
struct Constants(BTreeMap<u64, Value>);
impl Constants {
    fn get(&mut self, builder: &mut ModelBuilder, name: &str, value: u64) -> Result<Value, String> {
        if let Some(&binding) = self.0.get(&value) { return Ok(binding); }
        let domain = Domain::singleton(i64::try_from(value).map_err(error)?);
        let variable = builder.variable(format!("{name}.constant{value}"), domain.clone());
        let binding = Value::binding(variable, &domain).map_err(error)?;
        self.0.insert(value, binding);
        Ok(binding)
    }
}
/// Share functional predicate definitions within one lexical model activation
/// scope. Operation, event, dependency and resource identities remain distinct.
/// A cache must never be carried into or out of a `builder.when` scope.
#[derive(Default)]
struct Predicates {
    literals: BTreeMap<i64, VarId>,
    equalities: BTreeMap<(VarId, i64), VarId>,
    thresholds: BTreeMap<(VarId, u64), VarId>,
    conjunctions: BTreeMap<Vec<VarId>, VarId>,
}
impl Predicates {
    fn literal(&mut self, builder: &mut ModelBuilder, name: &str, value: i64) -> VarId {
        *self.literals.entry(value).or_insert_with(||
            builder.variable(format!("{name}.literal{value}"), Domain::singleton(value)))
    }
    fn equal_to(&mut self, builder: &mut ModelBuilder, name: &str, value: VarId, expected: i64) -> VarId {
        let key = (value, expected);
        if let Some(&output) = self.equalities.get(&key) { return output; }
        let output = boolean(builder, name);
        let literal = self.literal(builder, name, expected);
        builder.guarded_constraint(vec![Literal::new(output, 1)], Constraint::Equal { left: value, right: literal });
        builder.guarded_constraint(vec![Literal::new(output, 0)], Constraint::NotEqual { left: value, right: literal });
        self.equalities.insert(key, output);
        output
    }
    fn above(&mut self, builder: &mut ModelBuilder, name: &str, value: Value, limit: u64) -> Result<VarId, String> {
        let key = (value.id(), limit);
        if let Some(&output) = self.thresholds.get(&key) { return Ok(output); }
        let (minimum, maximum) = value.bounds();
        if minimum > limit || maximum <= limit {
            let output = self.literal(builder, name, i64::from(minimum > limit));
            self.thresholds.insert(key, output);
            return Ok(output);
        }
        let limit = i128::from(limit);
        let output = boolean(builder, name);
        builder.guarded_constraint(vec![Literal::new(output, 1)], Constraint::LinearLe { terms: vec![LinearTerm::new(value.id(), -1)], rhs: -limit - 1 });
        builder.guarded_constraint(vec![Literal::new(output, 0)], Constraint::LinearLe { terms: vec![LinearTerm::new(value.id(), 1)], rhs: limit });
        self.thresholds.insert(key, output);
        Ok(output)
    }
    fn all(&mut self, builder: &mut ModelBuilder, name: &str, mut inputs: Vec<VarId>) -> VarId {
        if let Some(&absent) = self.literals.get(&0) {
            if inputs.contains(&absent) { return absent; }
        }
        if let Some(&present) = self.literals.get(&1) {
            inputs.retain(|&input| input != present);
        }
        inputs.sort_unstable();
        inputs.dedup();
        if inputs.len() == 1 { return inputs[0]; }
        if let Some(&output) = self.conjunctions.get(&inputs) { return output; }
        let output = if inputs.is_empty() {
            self.literal(builder, name, 1)
        } else {
            let output = boolean(builder, name);
            builder.constraint(Constraint::BoolAnd { output, inputs: inputs.clone() });
            output
        };
        self.conjunctions.insert(inputs, output);
        output
    }
}
fn join(name: String) -> schedule::Operation { schedule::Operation { name, predecessors: Vec::new(), start_predecessors: Vec::new(), latency: 0, reservations: Vec::new() } }

/// Resource quantities describe actual backing slots, with the same guards and
/// original capacities as native declarations. Construction envelopes only bound
/// occurrences; they never become the selected shared-memory demand.
fn shared_storage(
    builder: &mut ModelBuilder,
    predicates: &mut Predicates,
    name: &str,
    template: &GroupingTemplate,
    launch: usize,
    geometry: &Launch,
    pooled_capacity: u64,
) -> Result<Value, String> {
    let allocations = template.operands().allocations.iter()
        .filter(|allocation| allocation.launch == launch).collect::<Vec<_>>();
    if allocations.is_empty() {
        let bytes = template.shared_per_item()[launch];
        if bytes != 0 && pooled_capacity == 0 {
            builder.obligation(geometry.presence.into_iter().map(|variable| Literal::new(variable, 1)).collect(),
                magnitude_solver::model::ObligationKind::Analysis,
                "shared allocation has no declared pooled resident capacity");
        }
        let mut algebra = Symbolic::new(builder, name);
        let bytes = algebra.constant(bytes).map_err(error)?;
        return algebra.product(bytes, geometry.items_per_group).map_err(error);
    }
    let mut total = Symbolic::new(builder, name).constant(0).map_err(error)?;
    for (index, allocation) in allocations.into_iter().enumerate() {
        if allocation.placement != seismic_realization::dispatch::TilePlacement::GroupShared { continue; }
        let prefix = format!("{name}.allocation{index}");
        let mut guards = allocation.guards.clone();
        if let Some(active) = geometry.presence { guards.push(Literal::new(active, 1)); }
        if pooled_capacity == 0 {
            builder.obligation(guards.clone(), magnitude_solver::model::ObligationKind::Analysis,
                format!("shared backing {} has no declared pooled resident capacity", allocation.symbol));
        }
        let terms = guards.iter().enumerate().map(|(index, guard)|
            predicates.equal_to(builder, &format!("{prefix}.guard{index}"), guard.variable, guard.value)).collect::<Vec<_>>();
        let active = predicates.all(builder, &format!("{prefix}.active"), terms);
        let bytes = builder.when(Literal::new(active, 1), |builder| -> Result<Value, String> {
            let mut algebra = Symbolic::new(builder, &prefix);
            let lanes = algebra.constant(crate::execution::SUBGROUP as u64).map_err(error)?;
            let storage = seismic_realization::dispatch::geometry::storage(&mut algebra,
                allocation.capacity, allocation.dtype.bytes() as u64, &allocation.placement,
                lanes, geometry.items_per_group).map_err(error)?;
            Ok(storage.shared_bytes_per_group)
        })?;
        let domain = Domain::interval(0, i64::try_from(bytes.bounds().1).map_err(error)?).map_err(error)?;
        let contribution = Symbolic::new(builder, &prefix).variable("selected_bytes", domain).map_err(error)?;
        builder.guarded_constraint(vec![Literal::new(active, 1)], Constraint::Equal { left: contribution.id(), right: bytes.id() });
        builder.guarded_constraint(vec![Literal::new(active, 0)], Constraint::LinearLe {
            terms: vec![LinearTerm::new(contribution.id(), 1)], rhs: 0 });
        total = Symbolic::new(builder, &prefix).sum(total, contribution).map_err(error)?;
    }
    Ok(total)
}

impl Prepared {
    pub(crate) fn derive(
        builder: &mut ModelBuilder,
        name: &str,
        execution: &Execution,
        template: GroupingTemplate,
        terminal: Option<Arc<crate::terminal::family::Binding>>,
        launches: &[&Launch],
        hardware: &model::Hardware,
        workload: &ScalarWorkload,
        limits: DerivationLimits,
        presence: Option<VarId>,
    ) -> Result<Self, DerivationError> {
        let prepare = |builder: &mut ModelBuilder| Self::derive_active(
            builder, name, execution, template, terminal, launches, hardware, workload, limits);
        let mut prepared = match presence {
            Some(active) => builder.when(Literal::new(active, 1), prepare),
            None => prepare(builder),
        }?;
        if let Some(active) = presence {
            let mut predicates = Predicates::default();
            for (index, operation) in prepared.presence.iter_mut().enumerate() {
                *operation = Some(match *operation {
                    Some(inner) => predicates.all(builder, &format!("{name}.operation{index}.enclosing"), vec![active, inner]),
                    None => active,
                });
            }
            for (index, edge) in prepared.edges.iter_mut().enumerate() {
                edge.presence = Some(match edge.presence {
                    Some(inner) => predicates.all(builder, &format!("{name}.edge{index}.enclosing"), vec![active, inner]),
                    None => active,
                });
            }
            for (index, resident) in prepared.residents.iter_mut().enumerate() {
                resident.presence = Some(match resident.presence {
                    Some(inner) => predicates.all(builder, &format!("{name}.resident{index}.enclosing"), vec![active, inner]),
                    None => active,
                });
            }
        }
        Ok(prepared)
    }
    fn derive_active(
        builder: &mut ModelBuilder,
        name: &str,
        execution: &Execution,
        template: GroupingTemplate,
        terminal: Option<Arc<crate::terminal::family::Binding>>,
        launches: &[&Launch],
        hardware: &model::Hardware,
        workload: &ScalarWorkload,
        limits: DerivationLimits,
    ) -> Result<Self, DerivationError> {
        if launches.len() != template.emitted().launches.len() { return Err("Metal grouping/template launch arity differs".into()); }
        let mut predicates = Predicates::default();
        let maximum = launches.iter().map(|launch| launch.items_per_group.bounds().1).collect::<Vec<_>>();
        let launch_presence = launches.iter().enumerate().map(|(index, launch)| -> Result<VarId, String> {
            let name = format!("{name}.launch{index}.submitted");
            let nonempty = predicates.above(builder, &name, launch.work_items, 0)?;
            Ok(match launch.presence {
                Some(present) => predicates.all(builder, &name, vec![present, nonempty]),
                None => nonempty,
            })
        }).collect::<Result<Vec<_>, _>>()?;
        let traces = model::grouping_traces(execution, &template, terminal.as_deref(), builder, &maximum, &launch_presence, hardware, workload, limits)?;
        let mut original = traces.model;
        for reason in original.unmapped.drain(..) {
            builder.obligation(vec![], magnitude_solver::model::ObligationKind::Analysis, reason);
        }
        let mut presence = traces.presence.iter().enumerate().map(|(index, guards)| {
            if guards.is_empty() { return None; }
            let inputs = guards.iter().enumerate().map(|(guard_index, guard)|
                predicates.equal_to(builder, &format!("{name}.operation{index}.guard{guard_index}"), guard.variable, guard.value)).collect();
            Some(predicates.all(builder, &format!("{name}.operation{index}.present"), inputs))
        }).collect::<Vec<_>>();
        let mut edges = Vec::new();
        let mut residents = Vec::new();
        let resident_groups = original.resources.len();
        original.resources.push(schedule::Resource { name: "Metal resident threadgroups".into(), capacity: hardware.resident_groups, unit: schedule::CapacityUnit::Slots });
        // Resource identity is common even when this alternative uses no shared
        // storage. Other alternatives reserve from exactly this same pool.
        let resident_shared = if hardware.resident_shared_bytes != 0 {
            let resource = original.resources.len();
            original.resources.push(schedule::Resource { name: "Metal resident shared bytes".into(), capacity: hardware.resident_shared_bytes, unit: schedule::CapacityUnit::Bytes });
            Some(resource)
        } else { None };
        let mut previous_launch = None;
        let one = Symbolic::new(builder, name).constant(1).map_err(error)?;
        for (index, (trace, geometry)) in traces.launches.iter().zip(launches).enumerate() {
            if geometry.work_items.bounds().1 > trace.work_items { return Err("retained work-item occurrence domain does not cover its original mapping".into()); }
            let prefix = format!("{name}.launch{index}");
            let launch_presence = Some(launch_presence[index]);
            if let Some(active) = launch_presence {
                presence[trace.submission] = Some(active);
            }
            if let Some(previous) = previous_launch { edges.push(Edge { before: previous, after: trace.submission, presence: launch_presence }); }
            let shared = shared_storage(builder, &mut predicates, &prefix, &template, index, geometry, hardware.resident_shared_bytes)?;
            builder.guarded_constraint(launch_presence.into_iter().map(|variable| Literal::new(variable, 1)).collect(),
                Constraint::LinearLe { terms: vec![LinearTerm::new(shared.id(), 1)], rhs: i128::from(execution.config.max_threadgroup_bytes) });
            let mut item_active = Vec::new();
            let padded_items = {
                let mut algebra = Symbolic::new(builder, &prefix);
                algebra.product(geometry.groups, geometry.items_per_group).map_err(error)?
            };
            for (item, operations) in trace.items.iter().enumerate() {
                let active = predicates.above(builder, &format!("{prefix}.item{item}.active"), padded_items, item as u64)?;
                let active = match launch_presence {
                    Some(launch) => predicates.all(builder, &format!("{prefix}.item{item}.launch"), vec![launch, active]),
                    None => active,
                };
                for operation in operations.clone() {
                    presence[operation] = Some(match presence[operation] {
                        Some(terminal) => predicates.all(builder, &format!("{prefix}.operation{operation}.active"), vec![active, terminal]),
                        None => active,
                    });
                }
                item_active.push(active);
            }
            let completion = original.operations.len();
            original.operations.push(join(format!("{prefix}.completion")));
            presence.push(None);
            edges.push(Edge { before: trace.submission, after: completion, presence: launch_presence });
            if let Some(previous) = previous_launch {
                edges.push(Edge { before: previous, after: completion, presence: None });
            }
            // A group is identified by its first logical work-item slot. Its
            // activation is exactly divisibility by the selected grouping.
            for first in 0..trace.work_items {
                let group = format!("{prefix}.group_at_{first}");
                let numerator = Symbolic::new(builder, &group).constant(first).map_err(error)?;
                let quotient = builder.variable(format!("{group}.index"), Domain::interval(0, i64::try_from(first).map_err(error)?).map_err(error)?);
                let remainder = builder.variable(format!("{group}.remainder"), Domain::interval(0, i64::try_from(geometry.items_per_group.bounds().1 - 1).map_err(error)?).map_err(error)?);
                builder.constraint(Constraint::Arithmetic(Arithmetic::DivRem { numerator: numerator.id(), denominator: geometry.items_per_group.id(), quotient, remainder }));
                let divisible = predicates.equal_to(builder, &format!("{group}.divisible"), remainder, 0);
                let nonempty = predicates.above(builder, &format!("{group}.nonempty"), geometry.work_items, first)?;
                let active = predicates.all(builder, &format!("{group}.active"), vec![divisible, nonempty]);
                let active = match launch_presence {
                    Some(launch) => predicates.all(builder, &format!("{group}.launch"), vec![launch, active]),
                    None => active,
                };
                let begin = original.operations.len();
                let mut operation = traces.group.clone(); operation.name = format!("{group}.admission");
                original.operations.push(operation); presence.push(Some(active));
                let end = original.operations.len();
                original.operations.push(join(format!("{group}.completion"))); presence.push(Some(active));
                edges.push(Edge { before: trace.submission, after: begin, presence: Some(active) });
                edges.push(Edge { before: begin, after: end, presence: Some(active) });
                edges.push(Edge { before: end, after: completion, presence: Some(active) });
                let upper = first.checked_add(geometry.items_per_group.bounds().1).ok_or("group membership domain overflow")?.min(trace.items.len() as u64);
                for item in first..upper {
                    let offset = item - first;
                    let admitted = predicates.above(builder, &format!("{group}.member{item}.width"), geometry.items_per_group, offset)?;
                    let member = predicates.all(builder, &format!("{group}.member{item}"), vec![active, admitted, item_active[item as usize]]);
                    let operations = &trace.items[item as usize];
                    if !operations.is_empty() {
                        edges.push(Edge { before: begin, after: operations.start, presence: Some(member) });
                        edges.push(Edge { before: operations.end - 1, after: end, presence: Some(member) });
                    }
                }
                residents.push(Resident { begin, end, resource: resident_groups, units: one, presence: Some(active) });
                if let Some(resource) = resident_shared { if shared.bounds().1 != 0 { residents.push(Resident { begin, end, resource, units: shared, presence: Some(active) }); } }
            }
            previous_launch = Some(completion);
        }
        Ok(Self { name: name.into(), template, terminal, original, presence, edges, residents })
    }
    pub(crate) fn resources(&self) -> &[schedule::Resource] { &self.original.resources }
    pub(crate) fn horizon(&self) -> Result<i64, String> {
        let duration = self.original.operations.iter().try_fold(0u64, |sum, operation| {
            let mut occupied = operation.latency;
            for reservation in &operation.reservations {
                occupied = occupied.max(reservation.offset.checked_add(reservation.duration)
                    .ok_or("Metal service horizon overflow")?);
            }
            sum.checked_add(occupied).ok_or("Metal family horizon overflow")
        })?;
        i64::try_from(duration).map_err(|_| "Metal family horizon exceeds shared integer range".into())
    }
    /// Attach to the enclosing search's resource relation. This never closes the
    /// boundary or installs a per-alternative objective.
    pub(crate) fn append(self, builder: &mut ModelBuilder, encoding: &mut Encoding) -> Result<Binding, String> {
        let Self { name, template, terminal, original, presence, edges, residents } = self;
        let name = name.as_str();
        if encoding.resources() != original.resources.as_slice() {
            return Err("Metal alternatives disagree on the shared resource identities".into());
        }
        encoding.bind_timebase(&original.timebase).map_err(error)?;
        let mut constants = Constants::default();
        let mut starts = Vec::new();
        let mut ends = Vec::new();
        for (index, operation) in original.operations.iter().enumerate() {
            let active = presence[index];
            let duration = constants.get(builder, name, operation.latency)?;
            let uses = operation.reservations.iter().map(|reservation| -> Result<_, String> {
                Ok(ResourceUse {
                    resource: reservation.resource,
                    offset: reservation.offset,
                    duration: constants.get(builder, name, reservation.duration)?,
                    units: constants.get(builder, name, reservation.units)?,
                })
            }).collect::<Result<Vec<_>, _>>()?;
            let activity = encoding.operation(builder, &format!("{name}.operation{index}"),
                duration, active).map_err(error)?;
            let mut partial = Vec::new();
            for service in &uses {
                if service.offset == 0 && service.duration.id() == activity.duration {
                    // These intervals are identical by construction. Keep
                    // each reservation, sharing only its defining activity.
                    encoding.whole_activity(service.resource, activity.clone(),
                        Demand::Variable(service.units.id())).map_err(error)?;
                } else {
                    partial.push(service.clone());
                }
            }
            encoding.service(builder, &activity, &partial).map_err(error)?;
            starts.push(activity.start); ends.push(activity.end);
        }
        for (index, operation) in original.operations.iter().enumerate() {
            for &before in &operation.predecessors {
                builder.constraint(Constraint::Schedule(SchedulingConstraint::Precedence { before: Event { time: ends[before], presence: presence[before] }, after: Event { time: starts[index], presence: presence[index] }, lag: 0 }));
            }
            for &before in &operation.start_predecessors {
                builder.constraint(Constraint::Schedule(SchedulingConstraint::Precedence { before: Event { time: starts[before], presence: presence[before] }, after: Event { time: starts[index], presence: presence[index] }, lag: 0 }));
            }
        }
        for edge in &edges {
            let relation = Constraint::Schedule(SchedulingConstraint::Precedence { before: Event { time: ends[edge.before], presence: presence[edge.before] }, after: Event { time: starts[edge.after], presence: presence[edge.after] }, lag: 0 });
            builder.guarded_constraint(edge.presence.into_iter().map(|variable| Literal::new(variable, 1)).collect(), relation);
        }
        for resident in &residents {
            encoding.reservation(resident.resource, Lifetime {
                begin: Event { time: starts[resident.begin], presence: resident.presence },
                end: Event { time: ends[resident.end], presence: resident.presence },
                demand: Demand::Variable(resident.units.id()),
            }.reservation()).map_err(error)?;
        }
        Ok(Binding { template, terminal, original, presence, starts, edges, residents })
    }
}
impl Binding {
    pub(crate) fn reconstruct(&self, values: &[i64], lower_bound: u64, dispatches: &[GroupDispatch]) -> Result<(crate::msl::Emitted, Objective, Vec<seismic_compiler::tuner::family::Decision>), String> {
        let (emitted, decisions) = match &self.terminal {
            Some(terminal) => {
                let (program, decisions) = terminal.instantiate(values)?;
                (self.template.instantiate_program(dispatches, &program, values)?, decisions)
            },
            None => (self.template.instantiate(dispatches, values)?, Vec::new()),
        };
        let active = |presence: Option<VarId>| -> Result<bool, String> { match presence {
            None => Ok(true), Some(variable) => match values.get(variable.0) { Some(0) => Ok(false), Some(1) => Ok(true), _ => Err("missing grouping activation assignment".into()) }
        } };
        let mut model = self.original.clone();
        model.operations.clear(); model.lifetimes.clear();
        let mut remap = vec![None; self.original.operations.len()];
        let mut starts = Vec::new();
        for (index, operation) in self.original.operations.iter().enumerate() {
            if !active(self.presence[index])? { continue; }
            remap[index] = Some(model.operations.len());
            model.operations.push(operation.clone());
            starts.push(values.get(self.starts[index].0).and_then(|&value| u64::try_from(value).ok()).ok_or("missing grouping operation start")?);
        }
        for operation in &mut model.operations {
            operation.predecessors = operation.predecessors.iter().filter_map(|&index| remap[index]).collect();
            operation.start_predecessors = operation.start_predecessors.iter().filter_map(|&index| remap[index]).collect();
        }
        for edge in &self.edges {
            if active(edge.presence)? {
                let before = remap[edge.before].ok_or("active grouping edge has absent predecessor")?;
                let after = remap[edge.after].ok_or("active grouping edge has absent successor")?;
                model.operations[after].predecessors.push(before);
            }
        }
        for resident in &self.residents {
            if !active(resident.presence)? { continue; }
            model.lifetimes.push(schedule::Lifetime {
                resource: resident.resource,
                units: values.get(resident.units.id().0).and_then(|&value| u64::try_from(value).ok()).ok_or("missing resident grouping units")?,
                begin: schedule::Event { operation: remap[resident.begin].ok_or("resident group missing admission")?, point: schedule::Point::Start },
                end: schedule::Event { operation: remap[resident.end].ok_or("resident group missing completion")?, point: schedule::Point::Completion },
            });
        }
        for operation in &mut model.operations { operation.predecessors.sort_unstable(); operation.predecessors.dedup(); }
        let completion = model.operations.iter().zip(&starts).try_fold(0u64, |completion, (operation, start)| start.checked_add(operation.latency).map(|end| completion.max(end)).ok_or("grouping completion overflow"))?;
        let objective = Objective::from_flat(Arc::new(model), schedule::Schedule { starts, completion }, lower_bound)?;
        Ok((emitted, objective, decisions))
    }
}
