//! Defensive verification of a logical program.
//!
//! Normal semantic discovery happens during construction; the builder checks
//! every locally decidable invariant eagerly. This pass re-validates cached
//! (deserialized or stored) programs: structural integrity, origin
//! domination, branch schema equality, identical carry types, legal
//! independent-loop joins, call/interface leaf agreement, result
//! completeness, and the moved-storage rule.

use super::*;
use std::collections::{BTreeMap, BTreeSet};

pub(super) fn verify(program: &LogicalProgram) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();

    // Global value/token origin table.
    let mut value_types: BTreeMap<GraphValueId, ValueType> = BTreeMap::new();
    let mut value_views: BTreeMap<GraphValueId, LogicalViewId> = BTreeMap::new();
    let mut token_storages: BTreeMap<StateTokenId, LogicalStorageId> = BTreeMap::new();

    for graph in program.graphs.iter() {
        for view in graph.views.iter() {
            if !graph.storages.get(view.storage).is_some() {
                errors.push(format!(
                    "graph {} view {} names storage {} which does not exist",
                    graph.choice.0, view.storage.0, view.storage.0
                ));
            }
        }
        verify_region(
            program,
            graph,
            &graph.root,
            &BTreeMap::new(),
            &BTreeMap::new(),
            &mut value_types,
            &mut value_views,
            &mut token_storages,
            &mut errors,
        );
        for result in &graph.results {
            match result {
                RegionResult::Value { id, .. } => {
                    if let Some(view) = value_views.get(id) {
                        let storage = graph.views[*view].storage;
                        if !matches!(
                            graph.storages[storage].initialization,
                            Initialization::FullyInitialized
                        ) {
                            errors.push(format!(
                                "graph {} result value {} reads storage {} that is not fully initialized",
                                graph.choice.0, id.0, storage.0
                            ));
                        }
                    }
                }
                RegionResult::State { id, storage, join } => {
                    if token_storages.get(id) != Some(storage) {
                        errors.push(format!(
                            "graph {} result state {} does not version storage {}",
                            graph.choice.0, id.0, storage.0
                        ));
                    }
                    if let Some(join) = join {
                        if !matches!(
                            join,
                            StateJoin::DisjointWrite { .. } | StateJoin::Atomic { .. }
                        ) {
                            errors.push(format!(
                                "graph {} result state {} carries an illegal join",
                                graph.choice.0, id.0
                            ));
                        }
                    }
                }
            }
        }
    }

    // Choices reference existing graphs and alternatives.
    for (ordinal, choice) in program.choices.iter().enumerate() {
        let id = ChoiceId(ordinal as u32);
        for (index, alternative) in choice.alternatives.iter().enumerate() {
            match program.graphs.get(alternative.graph) {
                Some(graph) => {
                    if graph.choice != id || graph.alternative != index as u32 {
                        errors.push(format!(
                            "graph {} claims to be alternative {} of choice {} but is stored as alternative {} of choice {}",
                            alternative.graph.0,
                            index,
                            id.0,
                            graph.alternative,
                            graph.choice.0
                        ));
                    }
                }
                None => errors.push(format!(
                    "choice {} alternative {} names graph {} which does not exist",
                    id.0, index, alternative.graph.0
                )),
            }
        }
    }
    if program.choices.get(program.entry_choice).is_none() {
        errors.push(format!(
            "the entry choice {} does not exist",
            program.entry_choice.0
        ));
    }

    // Runtime extents reference existing values.
    for extent in program.runtime_extents.iter() {
        verify_scalar_expr(&extent.value, &value_types, &mut errors);
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

fn verify_scalar_expr(
    expr: &RuntimeScalarExpr,
    values: &BTreeMap<GraphValueId, ValueType>,
    errors: &mut Vec<String>,
) {
    match expr {
        RuntimeScalarExpr::Value(id) => {
            if !values.contains_key(id) {
                errors.push(format!(
                    "a runtime extent reads value {} which has no origin",
                    id.0
                ));
            }
        }
        RuntimeScalarExpr::Add(a, b)
        | RuntimeScalarExpr::Sub(a, b)
        | RuntimeScalarExpr::Mul(a, b)
        | RuntimeScalarExpr::Div(a, b)
        | RuntimeScalarExpr::Rem(a, b) => {
            verify_scalar_expr(a, values, errors);
            verify_scalar_expr(b, values, errors);
        }
        _ => {}
    }
}

fn verify_region(
    program: &LogicalProgram,
    graph: &TaskGraph,
    region: &GraphRegion,
    parent_scope: &BTreeMap<GraphValueId, ValueType>,
    parent_states: &BTreeMap<StateTokenId, LogicalStorageId>,
    value_types: &mut BTreeMap<GraphValueId, ValueType>,
    value_views: &mut BTreeMap<GraphValueId, LogicalViewId>,
    token_storages: &mut BTreeMap<StateTokenId, LogicalStorageId>,
    errors: &mut Vec<String>,
) {
    // A region sees its own parameters and nodes plus everything its
    // enclosing scope defined before the construct was built.
    let mut scope: BTreeMap<GraphValueId, ValueType> = parent_scope.clone();
    let mut state_scope: BTreeMap<StateTokenId, LogicalStorageId> = parent_states.clone();
    for parameter in &region.parameters {
        match parameter {
            RegionParameter::Value { id, ty } => {
                record_value(*id, ty.clone(), value_types, &mut scope, errors);
            }
            RegionParameter::State { id, storage } => {
                record_token(*id, *storage, token_storages, &mut state_scope, errors);
            }
        }
    }
    let mut moved: BTreeSet<LogicalStorageId> = BTreeSet::new();
    for node in region.nodes.iter() {
        for input in &node.inputs {
            if !scope.contains_key(input) {
                errors.push(format!(
                    "graph {} reads value {} which does not dominate its use",
                    graph.choice.0, input.0
                ));
            }
            if let Some(view) = value_views.get(input) {
                let storage = graph.views[*view].storage;
                if moved.contains(&storage) {
                    errors.push(format!(
                        "graph {} uses storage {} after it was moved",
                        graph.choice.0, storage.0
                    ));
                }
            }
        }
        for token in &node.state_inputs {
            if !state_scope.contains_key(token) && !token_storages.contains_key(token) {
                errors.push(format!(
                    "graph {} node consumes state {} which is not in scope",
                    graph.choice.0, token.0
                ));
            }
        }
        for output in &node.outputs {
            record_value(
                output.id,
                output.ty.clone(),
                value_types,
                &mut scope,
                errors,
            );
            if let Some(view) = output.view {
                value_views.insert(output.id, view);
                if graph.views.get(view).is_none() {
                    errors.push(format!(
                        "graph {} output {} names view {} which does not exist",
                        graph.choice.0, output.id.0, view.0
                    ));
                }
            }
        }
        for token in &node.state_outputs {
            record_token(
                token.id,
                token.storage,
                token_storages,
                &mut state_scope,
                errors,
            );
        }
        // Per-kind checks.
        match &node.kind {
            LogicalNodeKind::Primitive(_) | LogicalNodeKind::Reduction(_) => {}
            LogicalNodeKind::If(if_node) => {
                if if_node.then_region.parameters != if_node.else_region.parameters {
                    errors.push(format!(
                        "graph {} has an `if` whose branches do not share parameter schemas",
                        graph.choice.0
                    ));
                }
                for (branch, join_index) in [(&if_node.then_region, 0), (&if_node.else_region, 1)] {
                    for join in &if_node.joins {
                        let result = match join {
                            JoinSlot::Value { then_result, .. } if join_index == 0 => *then_result,
                            JoinSlot::Value { else_result, .. } if join_index == 1 => *else_result,
                            JoinSlot::State { then_result, .. } if join_index == 0 => *then_result,
                            JoinSlot::State { else_result, .. } if join_index == 1 => *else_result,
                            _ => continue,
                        };
                        if region_result(branch, result).is_none() {
                            errors.push(format!(
                                "graph {} has an `if` join that names a result its branch does not have",
                                graph.choice.0
                            ));
                        }
                    }
                    verify_region(
                        program,
                        graph,
                        branch,
                        &scope,
                        &state_scope,
                        value_types,
                        value_views,
                        token_storages,
                        errors,
                    );
                }
            }
            LogicalNodeKind::Loop(loop_node) => {
                verify_region(
                    program,
                    graph,
                    &loop_node.body,
                    &scope,
                    &state_scope,
                    value_types,
                    value_views,
                    token_storages,
                    errors,
                );
                for slot in &loop_node.carried {
                    let parameter = loop_node
                        .body
                        .parameters
                        .get(slot.body_parameter.index())
                        .cloned();
                    let result = loop_node
                        .body
                        .results
                        .get(slot.body_result.index())
                        .cloned();
                    let (initial_value, initial_state) = match slot.initial {
                        RegionInput::Value(id) => (Some(id), None),
                        RegionInput::State(token) => (None, Some(token)),
                    };
                    match (&parameter, &result, initial_value, initial_state) {
                        (
                            Some(RegionParameter::Value { ty: param_ty, .. }),
                            Some(RegionResult::Value { ty: result_ty, .. }),
                            Some(initial),
                            None,
                        ) => {
                            let initial_ty = value_types.get(&initial);
                            if initial_ty != Some(param_ty) || initial_ty != Some(result_ty) {
                                errors.push(format!(
                                    "graph {} has a carried value slot whose initial, parameter and result types differ",
                                    graph.choice.0
                                ));
                            }
                        }
                        (
                            Some(RegionParameter::State {
                                storage: param_storage,
                                ..
                            }),
                            Some(RegionResult::State {
                                storage: result_storage,
                                ..
                            }),
                            None,
                            Some(initial),
                        ) => {
                            if token_storages.get(&initial) != Some(param_storage)
                                || param_storage != result_storage
                            {
                                errors.push(format!(
                                    "graph {} has a carried state slot whose storages differ",
                                    graph.choice.0
                                ));
                            }
                        }
                        _ => errors.push(format!(
                            "graph {} has a carried slot that mixes value and state shapes",
                            graph.choice.0
                        )),
                    }
                }
                if loop_node.kind == LoopKind::Independent {
                    if !loop_node.carried.is_empty() {
                        errors.push(format!(
                            "graph {} has an independent loop with data carries",
                            graph.choice.0
                        ));
                    }
                    for token in &node.state_outputs {
                        if let Some(join) = &token.join {
                            match join {
                                StateJoin::DisjointWrite { .. } | StateJoin::Atomic { .. } => {}
                            }
                        } else {
                            errors.push(format!(
                                "graph {} has an independent loop state output without a join",
                                graph.choice.0
                            ));
                        }
                    }
                } else {
                    for token in &node.state_outputs {
                        if token.join.is_some() {
                            errors.push(format!(
                                "graph {} has an ordered loop state output with a visit join",
                                graph.choice.0
                            ));
                        }
                    }
                }
            }
            LogicalNodeKind::Call(call_node) => {
                let Some(choice) = program.choices.get(call_node.choice) else {
                    errors.push(format!(
                        "graph {} has a call to choice {} which does not exist",
                        graph.choice.0, call_node.choice.0
                    ));
                    continue;
                };
                // Boundary paths must be the canonical leaves of the
                // interface, in order.
                let mut input_paths = Vec::new();
                for (ordinal, param) in choice.interface.params.iter().enumerate() {
                    for (path, _) in super::normalize::boundary_leaf_paths(&param.ty) {
                        input_paths.push((path, ordinal as u32));
                    }
                }
                let actual_inputs = call_node
                    .boundary_inputs
                    .iter()
                    .map(|input| (input.path.clone(), input.param))
                    .collect::<Vec<_>>();
                if actual_inputs != input_paths {
                    errors.push(format!(
                        "graph {} has a call whose boundary inputs are not the interface's canonical leaves",
                        graph.choice.0
                    ));
                }
                let mut result_paths = Vec::new();
                for (path, _) in super::normalize::boundary_leaf_paths(&choice.interface.result) {
                    result_paths.push(path);
                }
                for (_ordinal, param) in choice.interface.params.iter().enumerate() {
                    if param.mode == Mode::Inout {
                        for (path, leaf) in super::normalize::boundary_leaf_paths(&param.ty) {
                            if matches!(leaf, super::normalize::BoundaryLeafRef::Tensor) {
                                result_paths.push(path);
                            }
                        }
                    }
                }
                let actual_results = call_node
                    .boundary_results
                    .iter()
                    .map(|result| result.path.clone())
                    .collect::<Vec<_>>();
                if actual_results != result_paths {
                    errors.push(format!(
                        "graph {} has a call whose boundary results are not the canonical result leaves",
                        graph.choice.0
                    ));
                }
                // Moves mark storage; later uses were checked above by scope
                // scanning — record for subsequent nodes.
                for input in &call_node.boundary_inputs {
                    if let BoundaryInputKind::Move { state, .. } = &input.kind {
                        if let Some(storage) = token_storages.get(state) {
                            moved.insert(*storage);
                        }
                    }
                }
            }
        }
    }
}

fn region_result(region: &GraphRegion, id: RegionResultId) -> Option<&RegionResult> {
    region.results.get(id.index())
}

fn record_value(
    id: GraphValueId,
    ty: ValueType,
    global: &mut BTreeMap<GraphValueId, ValueType>,
    scope: &mut BTreeMap<GraphValueId, ValueType>,
    errors: &mut Vec<String>,
) {
    if let Some(existing) = global.get(&id) {
        if existing != &ty {
            errors.push(format!(
                "value {} is defined twice with different types: {} vs {}",
                id.0, existing, ty
            ));
        }
    } else {
        global.insert(id, ty.clone());
    }
    scope.insert(id, ty);
}

fn record_token(
    id: StateTokenId,
    storage: LogicalStorageId,
    global: &mut BTreeMap<StateTokenId, LogicalStorageId>,
    scope: &mut BTreeMap<StateTokenId, LogicalStorageId>,
    errors: &mut Vec<String>,
) {
    if let Some(existing) = global.get(&id) {
        if existing != &storage {
            errors.push(format!(
                "state token {} is defined twice with different storages",
                id.0
            ));
        }
    } else {
        global.insert(id, storage);
    }
    scope.insert(id, storage);
}
