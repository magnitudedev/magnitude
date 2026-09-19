//! Work-item operation traces from a retained grouping template. Group ownership
//! is deliberately absent here and is attached by the enclosing shared model.
use super::*;

pub(crate) struct Traces {
    pub model: Model,
    pub launches: Vec<Launch>,
    pub group: Operation,
    pub presence: Vec<Vec<magnitude_solver::model::Literal>>,
}
pub(crate) struct Launch {
    pub submission: usize,
    pub items: Vec<std::ops::Range<usize>>,
    pub work_items: u64,
}

pub(crate) fn derive(
    execution: &crate::execution::Execution,
    template: &crate::msl::GroupingTemplate,
    terminal: Option<&crate::terminal::family::Binding>,
    builder: &mut magnitude_solver::model::ModelBuilder,
    maximum_items: &[u64],
    launch_presence: &[magnitude_solver::model::VarId],
    hardware: &Hardware,
    workload: &ScalarWorkload,
    limits: DerivationLimits,
) -> Result<Traces, DerivationError> {
    hardware.validate()?;
    let emitted = template.canonical()?;
    validate_workload(&emitted, workload)?;
    if maximum_items.len() != emitted.launches.len() || launch_presence.len() != emitted.launches.len() { return Err("grouping domain differs from the terminal launch count".into()); }
    let mut model = Model {
        relationship: ModelRelationship::hypothetical_execution(),
        identity: format!("{}:{}:{}:retained-grouping", execution.function().name, workload.identity, hardware.identity),
        timebase: hardware.timebase.clone(), resources: hardware.resources.clone(), operations: Vec::new(), lifetimes: Vec::new(), static_orders: Vec::new(), unmapped: Vec::new(),
    };
    let group = hardware.operation(&Primitive::Group, 1, None)?.unwrap_or_else(|| {
        for &active in launch_presence {
            builder.obligation(vec![magnitude_solver::model::Literal::new(active, 1)], magnitude_solver::model::ObligationKind::Analysis,
                "Metal Group primitive has no hardware mapping");
        }
        Operation { name: "Group".into(), predecessors: Vec::new(), start_predecessors: Vec::new(), latency: 0, reservations: Vec::new() }
    });
    let mut state = Derivation::new(Sink::Schedule { hardware, model, groups_resource: usize::MAX, shared_resource: None }, limits);
    let scalar_values = canonical_scalars(&emitted, workload)?;
    state.memory.initialize(&emitted, workload)?;
    state.next_coordinate = workload.integer_domains.len() as u64 + 1;
    let mut launches = Vec::new();
    let mut presence = Vec::new();
    let mut predicates = family::Predicates::default();
    let parameter_symbols = super::parameters::Symbols::with_definitions(
        template.operands().bindings.clone(), template.operands().definitions.clone())?;
    for (index, ((metadata, body), &maximum_items)) in emitted.launches.iter().zip(emitted.terminal.launches()).zip(maximum_items).enumerate() {
        if maximum_items == 0 { return Err("grouping domain must be positive".into()); }
        let dispatch = metadata.dispatch.as_ref().ok_or("retained terminal launch has no dispatch")?;
        let launch_guard = magnitude_solver::model::Literal::new(launch_presence[index], 1);
        state.scope = format!("launch {index}");
        state.last = None;
        let submission = state.issue(Primitive::Launch, 1)?;
        if let Sink::Schedule { model, .. } = &mut state.sink {
            presence.resize(model.operations.len(), Vec::new());
            for reason in model.unmapped.drain(..) {
                builder.obligation(vec![launch_guard], magnitude_solver::model::ObligationKind::Analysis, reason);
            }
        }
        state.memory.launch(index, dispatch.work_items == 1);
        let maximum_dispatched = if dispatch.work_items == 0 { 0 } else {
            dispatch.work_items.checked_add(maximum_items - 1).ok_or("grouping trace domain overflow")?.min(u64::from(u32::MAX))
        };
        let mut items = Vec::new();
        for item in 0..maximum_dispatched {
            let begin = match &state.sink { Sink::Schedule { model, .. } => model.operations.len(), _ => unreachable!() };
            state.scope = format!("launch {index} logical item {item}");
            state.last = None;
            state.env = scalar_values.clone();
            state.ranges.clear();
            state.affine.clear();
            for (coordinate, domain) in workload.integer_domains.iter().enumerate() {
                if let seismic_accounting::workload::IntegerInput::Scalar { slot } = domain.input {
                    let name = format!("sc.{}", emitted.scalars[slot].name);
                    let value = affine::Value::domain(coordinate as u64 + 1, domain)?;
                    state.affine.insert(name.clone(), std::array::from_fn(|_| Some(value.clone())));
                    state.ranges.insert(name, [Some((domain.range.min, domain.range.max)); 32]);
                }
            }
            state.memory.subgroup();
            let allocations = template.operands().allocations.iter().filter(|allocation| allocation.launch == index).collect::<Vec<_>>();
            if allocations.is_empty() {
                for slot in &execution.memory().launches()[index].slots {
                    if slot.placement == seismic_realization::dispatch::TilePlacement::GroupShared {
                        let layout = slot.layout(dispatch)?;
                        state.memory.array(&slot.symbol, u64::from(slot.dtype.bytes()), layout.shared_elements_per_item, crate::terminal::Space::Threadgroup)?;
                    }
                }
            } else {
                // These names are the retained native backing identities. The
                // maximum capacity bounds address interpretation; actual shared
                // resource demand is the original guarded capacity expression.
                for allocation in allocations {
                    if allocation.placement == seismic_realization::dispatch::TilePlacement::GroupShared {
                        state.memory.array(&allocation.symbol, u64::from(allocation.dtype.bytes()),
                            allocation.capacity.bounds().1.max(1), crate::terminal::Space::Threadgroup)?;
                    }
                }
            }
            state.active = u32::MAX;
            state.alive = u32::MAX;
            state.returned = [None; 32];
            state.env.insert("lane".into(), std::array::from_fn(|lane| Some(lane as u64)));
            state.env.insert("tg_pos.x".into(), [Some(item); 32]);
            state.env.insert("sg_id".into(), [Some(0); 32]);
            state.env.insert(format!("seismic_family_grouping_{index}__"), [Some(1); 32]);
            for binding in emitted.buffers.iter().chain(&emitted.scratch_bindings) {
                let name = if binding.plane.is_empty() { binding.parameter.clone() } else { format!("{}_{}", binding.parameter, binding.plane) };
                state.env.insert(name, [None; 32]);
            }
            let mut symbols = parameter_symbols.clone();
            symbols.initialize(&mut state);
            if let Some(terminal) = terminal {
                family::Recorder { builder, binding: terminal, guards: vec![launch_guard], presence: &mut presence, predicates: &mut predicates, symbols: &mut symbols }
                    .launch(&mut state, &terminal.family().launches()[index])?;
            } else {
                if let Err(reason) = state.block(body, 0, body.len())? {
                    state.memory.unfinished_publication();
                    state.sink.gap(format!("{}: {reason}", state.scope));
                }
                if let Sink::Schedule { model, .. } = &mut state.sink {
                    presence.resize(model.operations.len(), vec![launch_guard]);
                    for reason in model.unmapped.drain(..) {
                        builder.obligation(vec![launch_guard], magnitude_solver::model::ObligationKind::Analysis, reason);
                    }
                }
            }
            let end = match &state.sink { Sink::Schedule { model, .. } => model.operations.len(), _ => unreachable!() };
            items.push(begin..end);
        }
        launches.push(Launch { submission, items, work_items: dispatch.work_items });
    }
    let Sink::Schedule { mut model, .. } = state.sink else { unreachable!() };
    model.unmapped.sort(); model.unmapped.dedup();
    for reason in model.unmapped.drain(..) {
        builder.obligation(vec![], magnitude_solver::model::ObligationKind::Analysis, reason);
    }
    Ok(Traces { model, launches, group, presence })
}
