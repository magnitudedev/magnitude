//! Defensive verification of a logical program.
//!
//! Normal semantic discovery happens during construction; the builder checks
//! every locally decidable invariant eagerly and `seal` establishes the
//! graph-wide ones. This pass re-validates cached (deserialized or stored)
//! programs against the same invariants: every value and state token of a
//! graph is defined exactly once with an exhaustive kind, every view names
//! one base (a storage of the graph or a computed tensor value of it),
//! origin domination, branch schema equality, identical carry
//! types, legal independent-loop joins, boundary/interface leaf agreement for
//! roots and calls, the moved-storage rule, and referential integrity of
//! choices, graphs, runtime extents and shape fields. It rejects no ordinary
//! builder output; every error here is a `CompilerDefect` of cached data.

use super::normalize::{boundary_leaves, LeafKind};
use super::*;
use crate::sir::ParamOwnership;
use std::collections::{BTreeMap, BTreeSet};

pub(super) fn verify(program: &LogicalProgram) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();

    // Every value of every graph, for runtime-extent references.
    let mut all_values: BTreeSet<GraphValueId> = BTreeSet::new();

    for (graph_id, graph) in program.graphs() {
        let tag = format!("graph {}", graph_id.0);
        verify_tables(&tag, graph, &mut errors);
        for value in graph.values() {
            all_values.insert(value.id());
        }

        // Origins: exactly one per value and per state token, all in the
        // tables and agreeing with them.
        let mut value_origins: BTreeMap<GraphValueId, usize> = BTreeMap::new();
        let mut state_origins: BTreeMap<StateTokenId, usize> = BTreeMap::new();
        collect_origins(&tag, graph, graph.root(), &mut value_origins, &mut state_origins, &mut errors);
        for (id, count) in &value_origins {
            if *count != 1 {
                errors.push(format!("{tag}: value {} has {count} origins", id.0));
            }
        }
        for value in graph.values() {
            if !value_origins.contains_key(&value.id()) {
                errors.push(format!(
                    "{tag}: value {} is in the table but never originates",
                    value.id().0
                ));
            }
        }
        for (id, count) in &state_origins {
            if *count != 1 {
                errors.push(format!("{tag}: state token {} has {count} origins", id.0));
            }
        }
        for (token, _) in graph.states() {
            if !state_origins.contains_key(&token) {
                errors.push(format!(
                    "{tag}: state token {} is in the table but never originates",
                    token.0
                ));
            }
        }

        if !graph.root().results.is_empty() {
            errors.push(format!(
                "{tag}: the root region has positional results; the boundary is the only result authority"
            ));
        }

        let mut root_scope = BTreeSet::new();
        let mut root_states = BTreeSet::new();
        verify_region(
            program,
            &tag,
            graph,
            graph.root(),
            &BTreeSet::new(),
            &BTreeSet::new(),
            &mut root_scope,
            &mut root_states,
            &mut errors,
        );

        match program.choices.get(graph.choice) {
            Some(choice) => verify_root_boundary(
                &tag,
                graph,
                &choice.interface,
                &root_scope,
                &root_states,
                &mut errors,
            ),
            None => errors.push(format!(
                "{tag} belongs to choice {} which does not exist",
                graph.choice.0
            )),
        }
    }

    // Choices reference existing graphs and alternatives.
    for (id, choice) in program.choices() {
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

    // Shape fields are dense and self-identified.
    for (id, field) in program.shape_fields.entries() {
        if field.id != id {
            errors.push(format!(
                "shape field {} records id {}",
                id.0, field.id.0
            ));
        }
    }

    // Runtime extents reference existing values, extents and shape fields.
    for (id, extent) in program.runtime_extents.entries() {
        if extent.id != id {
            errors.push(format!(
                "runtime extent {} records id {}",
                id.0, extent.id.0
            ));
        }
        verify_scalar_expr(program, &extent.value, &all_values, &mut errors);
        match &extent.value {
            RuntimeScalarExpr::ShapeField(field) => {
                if program.shape_fields.get(*field).is_none() {
                    errors.push(format!(
                        "runtime extent {} reads shape field {} which does not exist",
                        id.0, field.0
                    ));
                } else if extent.expected.is_none() {
                    errors.push(format!(
                        "runtime extent {} is a shape field without its expected value",
                        id.0
                    ));
                }
            }
            RuntimeScalarExpr::Const(_)
            | RuntimeScalarExpr::Value(_)
            | RuntimeScalarExpr::Extent(_)
            | RuntimeScalarExpr::Add(..)
            | RuntimeScalarExpr::Sub(..)
            | RuntimeScalarExpr::Mul(..)
            | RuntimeScalarExpr::Div(..)
            | RuntimeScalarExpr::Rem(..) => {
                if extent.expected.is_some() {
                    errors.push(format!(
                        "runtime extent {} states an expected value but is not a shape field",
                        id.0
                    ));
                }
            }
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// Table integrity: every value has an exhaustive, well-formed kind; every
/// view names one base — a storage of the graph, or a computed tensor value
/// of the graph, read-only; every state token names a storage.
fn verify_tables(tag: &str, graph: &TaskGraph, errors: &mut Vec<String>) {
    for (id, view) in graph.views() {
        match view.base {
            ViewBase::Storage(storage) => {
                if graph.storages.get(storage).is_none() {
                    errors.push(format!(
                        "{tag}: view {} names storage {} which does not exist",
                        id.0, storage.0
                    ));
                }
            }
            ViewBase::Value(value) => {
                match graph.values.get(value).map(|value| value.kind()) {
                    Some(GraphValueKind::Tensor {
                        source: TensorSource::Computed,
                        ..
                    }) => {}
                    Some(_) => errors.push(format!(
                        "{tag}: view {} names value {} which is not a computed tensor",
                        id.0, value.0
                    )),
                    None => errors.push(format!(
                        "{tag}: view {} names value {} which does not exist",
                        id.0, value.0
                    )),
                }
                if view.access != Access::Shared {
                    errors.push(format!(
                        "{tag}: view {} of a computed value is not read-only",
                        id.0
                    ));
                }
            }
        }
    }
    for (token, storage) in graph.states() {
        if graph.storages.get(storage).is_none() {
            errors.push(format!(
                "{tag}: state token {} names storage {} which does not exist",
                token.0, storage.0
            ));
        }
    }
    for (id, value) in graph.values.entries() {
        if value.id() != id {
            errors.push(format!("{tag}: value {} records id {}", id.0, value.id().0));
        }
        match value.kind() {
            GraphValueKind::Void => {
                errors.push(format!("{tag}: value {} is void", id.0));
            }
            GraphValueKind::Tensor {
                ty,
                source: TensorSource::View(view),
            } => match graph.views.get(*view) {
                Some(declared) => {
                    if declared.shape != *ty {
                        errors.push(format!(
                            "{tag}: value {} has type {} but its view {} has shape {}",
                            id.0,
                            value.ty(),
                            view.0,
                            ValueType::Tensor(declared.shape.clone())
                        ));
                    }
                }
                None => errors.push(format!(
                    "{tag}: value {} names view {} which does not exist",
                    id.0, view.0
                )),
            },
            GraphValueKind::Tensor {
                source: TensorSource::Computed,
                ..
            }
            | GraphValueKind::Tuple(_)
            | GraphValueKind::Scalar(_)
            | GraphValueKind::Index { .. }
            | GraphValueKind::Range { .. }
            | GraphValueKind::Capability(_) => {}
        }
    }
}

fn collect_origins(
    tag: &str,
    graph: &TaskGraph,
    region: &GraphRegion,
    values: &mut BTreeMap<GraphValueId, usize>,
    states: &mut BTreeMap<StateTokenId, usize>,
    errors: &mut Vec<String>,
) {
    for parameter in &region.parameters {
        match parameter {
            RegionParameter::Value { id, ty } => {
                *values.entry(*id).or_insert(0) += 1;
                match graph.values.get(*id) {
                    Some(value) => {
                        if value.ty() != *ty {
                            errors.push(format!(
                                "{tag}: region parameter {} is declared as {ty} but the table holds {}",
                                id.0,
                                value.ty()
                            ));
                        }
                    }
                    None => errors.push(format!(
                        "{tag}: region parameter {} is not in the value table",
                        id.0
                    )),
                }
            }
            RegionParameter::State { id, storage } => {
                *states.entry(*id).or_insert(0) += 1;
                match graph.states.get(*id) {
                    Some(bound) => {
                        if bound != storage {
                            errors.push(format!(
                                "{tag}: region state parameter {} is declared on storage {} but the table holds {}",
                                id.0, storage.0, bound.0
                            ));
                        }
                    }
                    None => errors.push(format!(
                        "{tag}: region state parameter {} is not in the state table",
                        id.0
                    )),
                }
            }
        }
    }
    for node in region.nodes.iter() {
        for output in &node.outputs {
            *values.entry(output.id()).or_insert(0) += 1;
            match graph.values.get(output.id()) {
                Some(value) => {
                    if value != output {
                        errors.push(format!(
                            "{tag}: node output {} disagrees with the value table",
                            output.id().0
                        ));
                    }
                }
                None => errors.push(format!(
                    "{tag}: node output {} is not in the value table",
                    output.id().0
                )),
            }
        }
        for token in &node.state_outputs {
            *states.entry(token.id).or_insert(0) += 1;
            match graph.states.get(token.id) {
                Some(bound) => {
                    if *bound != token.storage {
                        errors.push(format!(
                            "{tag}: node state output {} is on storage {} but the table holds {}",
                            token.id.0, token.storage.0, bound.0
                        ));
                    }
                }
                None => errors.push(format!(
                    "{tag}: node state output {} is not in the state table",
                    token.id.0
                )),
            }
        }
        match &node.kind {
            LogicalNodeKind::If(if_node) => {
                collect_origins(tag, graph, &if_node.then_region, values, states, errors);
                collect_origins(tag, graph, &if_node.else_region, values, states, errors);
            }
            LogicalNodeKind::Loop(loop_node) => {
                collect_origins(tag, graph, &loop_node.body, values, states, errors);
            }
            LogicalNodeKind::Primitive(_)
            | LogicalNodeKind::Reduction(_)
            | LogicalNodeKind::Call(_) => {}
        }
    }
}

fn verify_scalar_expr(
    program: &LogicalProgram,
    expr: &RuntimeScalarExpr,
    values: &BTreeSet<GraphValueId>,
    errors: &mut Vec<String>,
) {
    match expr {
        RuntimeScalarExpr::Const(_) => {}
        RuntimeScalarExpr::Value(id) => {
            if !values.contains(id) {
                errors.push(format!(
                    "a runtime extent reads value {} which has no origin",
                    id.0
                ));
            }
        }
        RuntimeScalarExpr::Extent(id) => {
            if program.runtime_extents.get(*id).is_none() {
                errors.push(format!(
                    "a runtime extent reads extent {} which does not exist",
                    id.0
                ));
            }
        }
        RuntimeScalarExpr::ShapeField(id) => {
            if program.shape_fields.get(*id).is_none() {
                errors.push(format!(
                    "a runtime extent reads shape field {} which does not exist",
                    id.0
                ));
            }
        }
        RuntimeScalarExpr::Add(a, b)
        | RuntimeScalarExpr::Sub(a, b)
        | RuntimeScalarExpr::Mul(a, b)
        | RuntimeScalarExpr::Div(a, b)
        | RuntimeScalarExpr::Rem(a, b) => {
            verify_scalar_expr(program, a, values, errors);
            verify_scalar_expr(program, b, values, errors);
        }
    }
}

/// The storage a value reads through a view, when its view's base is
/// storage. A computed tensor and a view of one read no storage.
fn view_storage(graph: &TaskGraph, id: GraphValueId) -> Option<LogicalStorageId> {
    match graph.values.get(id)?.kind() {
        GraphValueKind::Tensor {
            source: TensorSource::View(view),
            ..
        } => match graph.views.get(*view)?.base {
            ViewBase::Storage(storage) => Some(storage),
            ViewBase::Value(_) => None,
        },
        GraphValueKind::Tensor {
            source: TensorSource::Computed,
            ..
        }
        | GraphValueKind::Void
        | GraphValueKind::Scalar(_)
        | GraphValueKind::Index { .. }
        | GraphValueKind::Range { .. }
        | GraphValueKind::Tuple(_)
        | GraphValueKind::Capability(_) => None,
    }
}

#[allow(clippy::too_many_arguments)]
fn verify_region(
    program: &LogicalProgram,
    tag: &str,
    graph: &TaskGraph,
    region: &GraphRegion,
    parent_scope: &BTreeSet<GraphValueId>,
    parent_states: &BTreeSet<StateTokenId>,
    exit_scope: &mut BTreeSet<GraphValueId>,
    exit_states: &mut BTreeSet<StateTokenId>,
    errors: &mut Vec<String>,
) {
    // A region sees its own parameters and nodes plus everything its
    // enclosing scope defined before the construct was built.
    let mut scope: BTreeSet<GraphValueId> = parent_scope.clone();
    let mut state_scope: BTreeSet<StateTokenId> = parent_states.clone();
    for parameter in &region.parameters {
        match parameter {
            RegionParameter::Value { id, .. } => {
                scope.insert(*id);
            }
            RegionParameter::State { id, .. } => {
                state_scope.insert(*id);
            }
        }
    }
    let mut moved: BTreeSet<LogicalStorageId> = BTreeSet::new();
    for node in region.nodes.iter() {
        for input in &node.inputs {
            if !scope.contains(input) {
                errors.push(format!(
                    "{tag}: a node reads value {} which does not dominate its use",
                    input.0
                ));
            }
            if let Some(storage) = view_storage(graph, *input) {
                if moved.contains(&storage) {
                    errors.push(format!(
                        "{tag}: a node uses storage {} after it was moved",
                        storage.0
                    ));
                }
            }
        }
        for token in &node.state_inputs {
            if !state_scope.contains(token) {
                errors.push(format!(
                    "{tag}: a node consumes state {} which is not in scope",
                    token.0
                ));
            }
        }
        for output in &node.outputs {
            scope.insert(output.id());
        }
        for token in &node.state_outputs {
            state_scope.insert(token.id);
        }
        // Per-kind checks.
        match &node.kind {
            LogicalNodeKind::Primitive(_) | LogicalNodeKind::Reduction(_) => {}
            LogicalNodeKind::If(if_node) => {
                if if_node.then_region.parameters != if_node.else_region.parameters {
                    errors.push(format!(
                        "{tag}: an `if` has branches which do not share parameter schemas"
                    ));
                }
                for (branch, is_then) in [(&if_node.then_region, true), (&if_node.else_region, false)] {
                    for join in &if_node.joins {
                        let result = match join {
                            JoinSlot::Value {
                                then_result,
                                else_result,
                                ..
                            }
                            | JoinSlot::State {
                                then_result,
                                else_result,
                                ..
                            } => {
                                if is_then {
                                    *then_result
                                } else {
                                    *else_result
                                }
                            }
                        };
                        if branch.results.get(result.index()).is_none() {
                            errors.push(format!(
                                "{tag}: an `if` join names a result its branch does not have"
                            ));
                        }
                    }
                    let mut branch_scope = BTreeSet::new();
                    let mut branch_states = BTreeSet::new();
                    verify_region(
                        program,
                        tag,
                        graph,
                        branch,
                        &scope,
                        &state_scope,
                        &mut branch_scope,
                        &mut branch_states,
                        errors,
                    );
                }
            }
            LogicalNodeKind::Loop(loop_node) => {
                let mut body_scope = BTreeSet::new();
                let mut body_states = BTreeSet::new();
                verify_region(
                    program,
                    tag,
                    graph,
                    &loop_node.body,
                    &scope,
                    &state_scope,
                    &mut body_scope,
                    &mut body_states,
                    errors,
                );
                for slot in &loop_node.carried {
                    let parameter = loop_node.body.parameters.get(slot.body_parameter.index());
                    let result = loop_node.body.results.get(slot.body_result.index());
                    match (parameter, result, slot.initial) {
                        (
                            Some(RegionParameter::Value { ty: param_ty, .. }),
                            Some(RegionResult::Value { ty: result_ty, .. }),
                            RegionInput::Value(initial),
                        ) => {
                            let initial_ty = graph.values.get(initial).map(|value| value.ty());
                            if initial_ty.as_ref() != Some(param_ty)
                                || initial_ty.as_ref() != Some(result_ty)
                            {
                                errors.push(format!(
                                    "{tag}: a carried value slot's initial, parameter and result types differ"
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
                            RegionInput::State(initial),
                        ) => {
                            if graph.states.get(initial) != Some(param_storage)
                                || param_storage != result_storage
                            {
                                errors.push(format!(
                                    "{tag}: a carried state slot's storages differ"
                                ));
                            }
                        }
                        (
                            Some(RegionParameter::Value { .. }) | Some(RegionParameter::State { .. }),
                            Some(RegionResult::Value { .. }) | Some(RegionResult::State { .. }),
                            RegionInput::Value(_) | RegionInput::State(_),
                        ) => errors.push(format!(
                            "{tag}: a carried slot mixes value and state shapes"
                        )),
                        (None, _, _) | (_, None, _) => errors.push(format!(
                            "{tag}: a carried slot names a body parameter or result that does not exist"
                        )),
                    }
                }
                match loop_node.kind {
                    LoopKind::Independent => {
                        if !loop_node.carried.is_empty() {
                            errors.push(format!(
                                "{tag}: an independent loop has data carries"
                            ));
                        }
                        for token in &node.state_outputs {
                            match &token.join {
                                Some(StateJoin::DisjointWrite { .. })
                                | Some(StateJoin::Atomic { .. }) => {}
                                None => errors.push(format!(
                                    "{tag}: an independent loop state output has no join"
                                )),
                            }
                        }
                    }
                    LoopKind::Ordered => {
                        for token in &node.state_outputs {
                            if token.join.is_some() {
                                errors.push(format!(
                                    "{tag}: an ordered loop state output carries a visit join"
                                ));
                            }
                        }
                    }
                }
            }
            LogicalNodeKind::Call(call_node) => {
                let Some(choice) = program.choices.get(call_node.choice) else {
                    errors.push(format!(
                        "{tag}: a call names choice {} which does not exist",
                        call_node.choice.0
                    ));
                    continue;
                };
                verify_call_boundary(tag, graph, node, call_node, &choice.interface, errors);
                for input in call_node.boundary.inputs.values() {
                    if let CallInput::Tensor {
                        state,
                        ownership: ParamOwnership::Owned,
                        ..
                    } = input
                    {
                        if let Some(storage) = graph.states.get(*state) {
                            moved.insert(*storage);
                        }
                    }
                }
            }
        }
    }
    for result in &region.results {
        match result {
            RegionResult::Value { id, ty } => {
                if !scope.contains(id) {
                    errors.push(format!(
                        "{tag}: region result value {} is not in scope",
                        id.0
                    ));
                }
                match graph.values.get(*id) {
                    Some(value) => {
                        if value.ty() != *ty {
                            errors.push(format!(
                                "{tag}: region result value {} is declared as {ty} but the table holds {}",
                                id.0,
                                value.ty()
                            ));
                        }
                    }
                    None => errors.push(format!(
                        "{tag}: region result value {} is not in the value table",
                        id.0
                    )),
                }
            }
            RegionResult::State { id, storage, join } => {
                if !state_scope.contains(id) {
                    errors.push(format!(
                        "{tag}: region result state {} is not in scope",
                        id.0
                    ));
                }
                if graph.states.get(*id) != Some(storage) {
                    errors.push(format!(
                        "{tag}: region result state {} does not version storage {}",
                        id.0, storage.0
                    ));
                }
                match join {
                    None | Some(StateJoin::DisjointWrite { .. }) | Some(StateJoin::Atomic { .. }) => {}
                }
            }
        }
    }
    *exit_scope = scope;
    *exit_states = state_scope;
}

/// The canonical input leaves of an interface, with their declared ownership.
fn interface_input_leaves(
    interface: &FunctionInterface,
) -> Vec<(BoundaryLeaf, ParamOwnership, bool)> {
    let mut out = Vec::new();
    for (ordinal, param) in interface.params.iter().enumerate() {
        for (path, leaf) in boundary_leaves(&param.ty) {
            out.push((
                BoundaryLeaf::Input {
                    param: ordinal as u32,
                    leaf: path,
                },
                param.ownership,
                matches!(leaf, LeafKind::Tensor(_)),
            ));
        }
    }
    out
}

fn interface_result_leaves(interface: &FunctionInterface) -> Vec<BoundaryLeaf> {
    boundary_leaves(&interface.result)
        .into_iter()
        .map(|(path, _)| BoundaryLeaf::Result { leaf: path })
        .collect()
}

fn verify_root_boundary(
    tag: &str,
    graph: &TaskGraph,
    interface: &FunctionInterface,
    root_scope: &BTreeSet<GraphValueId>,
    root_states: &BTreeSet<StateTokenId>,
    errors: &mut Vec<String>,
) {
    let boundary = &graph.boundary;
    let expected_inputs = interface_input_leaves(interface);
    let expected_keys: Vec<&BoundaryLeaf> = expected_inputs.iter().map(|(leaf, _, _)| leaf).collect();
    let actual_keys: Vec<&BoundaryLeaf> = boundary.inputs().keys().collect();
    if expected_keys != actual_keys {
        errors.push(format!(
            "{tag}: the boundary inputs are not the interface's canonical leaves"
        ));
    }
    let root_value_params: BTreeSet<GraphValueId> = graph
        .root()
        .parameters
        .iter()
        .filter_map(|parameter| match parameter {
            RegionParameter::Value { id, .. } => Some(*id),
            RegionParameter::State { .. } => None,
        })
        .collect();
    let root_state_params: BTreeMap<StateTokenId, LogicalStorageId> = graph
        .root()
        .parameters
        .iter()
        .filter_map(|parameter| match parameter {
            RegionParameter::State { id, storage } => Some((*id, *storage)),
            RegionParameter::Value { .. } => None,
        })
        .collect();
    let mut exclusive_leaves = BTreeSet::new();
    for (leaf, ownership, is_tensor) in &expected_inputs {
        let Some(input) = boundary.inputs().get(leaf) else {
            continue;
        };
        match (input, *is_tensor) {
            (LogicalBoundaryInput::Value(id), false) => {
                if !root_value_params.contains(id) {
                    errors.push(format!(
                        "{tag}: boundary input {} is not a root parameter",
                        id.0
                    ));
                }
            }
            (
                LogicalBoundaryInput::Tensor {
                    value,
                    state,
                    ownership: actual,
                },
                true,
            ) => {
                if actual != ownership {
                    errors.push(format!(
                        "{tag}: boundary input {} declares {actual:?} but the interface declares {ownership:?}",
                        value.0
                    ));
                }
                if !root_value_params.contains(value) {
                    errors.push(format!(
                        "{tag}: boundary tensor input {} is not a root parameter",
                        value.0
                    ));
                }
                match view_storage(graph, *value) {
                    Some(storage) => {
                        if graph.storages.get(storage).map(|s| &s.owner)
                            != Some(&LogicalStorageOwner::Parameter(leaf.clone()))
                        {
                            errors.push(format!(
                                "{tag}: boundary tensor input {} views storage {} which its leaf does not own",
                                value.0, storage.0
                            ));
                        }
                        if root_state_params.get(state) != Some(&storage) {
                            errors.push(format!(
                                "{tag}: boundary tensor input {} names entry state {} which is not the root state of its storage",
                                value.0, state.0
                            ));
                        }
                        if *actual == ParamOwnership::Exclusive {
                            exclusive_leaves.insert(leaf.clone());
                            match boundary.final_states().get(leaf) {
                                Some(token) => {
                                    if !root_states.contains(token) {
                                        errors.push(format!(
                                            "{tag}: final state {} is not in root scope",
                                            token.0
                                        ));
                                    }
                                    if graph.states.get(*token) != Some(&storage) {
                                        errors.push(format!(
                                            "{tag}: final state {} does not version storage {}",
                                            token.0, storage.0
                                        ));
                                    }
                                }
                                None => errors.push(format!(
                                    "{tag}: exclusive boundary input {} has no final state",
                                    value.0
                                )),
                            }
                        }
                    }
                    None => errors.push(format!(
                        "{tag}: boundary tensor input {} is not a view of storage",
                        value.0
                    )),
                }
            }
            (LogicalBoundaryInput::Value(id), true) => errors.push(format!(
                "{tag}: boundary input {} is a tensor leaf passed as a value",
                id.0
            )),
            (LogicalBoundaryInput::Tensor { value, .. }, false) => errors.push(format!(
                "{tag}: boundary input {} is a non-tensor leaf passed as a tensor",
                value.0
            )),
        }
    }
    let final_keys: BTreeSet<BoundaryLeaf> = boundary.final_states().keys().cloned().collect();
    if final_keys != exclusive_leaves {
        errors.push(format!(
            "{tag}: the boundary final states are not exactly the exclusive tensor leaves"
        ));
    }
    let expected_results = interface_result_leaves(interface);
    let actual_results: Vec<BoundaryLeaf> = boundary.results().keys().cloned().collect();
    if expected_results != actual_results {
        errors.push(format!(
            "{tag}: the boundary results are not the interface's canonical result leaves"
        ));
    }
    for result in boundary.results().values() {
        if !root_scope.contains(&result.value) {
            errors.push(format!(
                "{tag}: boundary result {} is not in root scope",
                result.value.0
            ));
        }
        if let Some(storage) = view_storage(graph, result.value) {
            match graph.storages.get(storage) {
                Some(declared) => {
                    if declared.initialization != Initialization::FullyInitialized {
                        errors.push(format!(
                            "{tag}: boundary result {} views storage {} which is not fully initialized",
                            result.value.0, storage.0
                        ));
                    }
                }
                None => errors.push(format!(
                    "{tag}: boundary result {} views storage {} which does not exist",
                    result.value.0, storage.0
                )),
            }
        }
    }
}

fn verify_call_boundary(
    tag: &str,
    graph: &TaskGraph,
    node: &LogicalNode,
    call: &CallNode,
    interface: &FunctionInterface,
    errors: &mut Vec<String>,
) {
    let boundary = &call.boundary;
    let expected_inputs = interface_input_leaves(interface);
    let expected_keys: Vec<&BoundaryLeaf> = expected_inputs.iter().map(|(leaf, _, _)| leaf).collect();
    let actual_keys: Vec<&BoundaryLeaf> = boundary.inputs.keys().collect();
    if expected_keys != actual_keys {
        errors.push(format!(
            "{tag}: a call's boundary inputs are not the callee's canonical leaves"
        ));
    }
    let mut exclusive_leaves = BTreeSet::new();
    for (leaf, ownership, is_tensor) in &expected_inputs {
        let Some(input) = boundary.inputs.get(leaf) else {
            continue;
        };
        match (input, *is_tensor) {
            (CallInput::Value(_), false) => {}
            (
                CallInput::Computed {
                    value,
                    ownership: actual,
                },
                true,
            ) => {
                if actual != ownership {
                    errors.push(format!(
                        "{tag}: call input {} declares {actual:?} but the callee declares {ownership:?}",
                        value.0
                    ));
                }
                match actual {
                    ParamOwnership::Shared | ParamOwnership::Owned => {}
                    ParamOwnership::Exclusive | ParamOwnership::Value => {
                        errors.push(format!(
                            "{tag}: call input {} is a computed value with {actual:?} ownership",
                            value.0
                        ))
                    }
                }
                if boundary.final_states.contains_key(leaf) {
                    errors.push(format!(
                        "{tag}: computed call input {} has a final state",
                        value.0
                    ));
                }
                match graph.values.get(*value).map(|value| value.kind()) {
                    Some(GraphValueKind::Tensor {
                        source: TensorSource::Computed,
                        ..
                    }) => {}
                    Some(GraphValueKind::Tensor {
                        source: TensorSource::View(view),
                        ..
                    }) => match graph.views.get(*view).map(|view| view.base) {
                        Some(ViewBase::Value(_)) => {}
                        Some(ViewBase::Storage(_)) | None => {
                            errors.push(format!(
                                "{tag}: call tensor input {} views storage but is passed as a computed value",
                                value.0
                            ))
                        }
                    },
                    Some(_) | None => {
                        errors.push(format!(
                            "{tag}: call tensor input {} is not a computed tensor value",
                            value.0
                        ))
                    }
                }
            }
            (
                CallInput::Tensor {
                    value,
                    state,
                    ownership: actual,
                },
                true,
            ) => {
                if actual != ownership {
                    errors.push(format!(
                        "{tag}: call input {} declares {actual:?} but the callee declares {ownership:?}",
                        value.0
                    ));
                }
                match (view_storage(graph, *value), graph.states.get(*state)) {
                    (Some(viewed), Some(versioned)) => {
                        if viewed != *versioned {
                            errors.push(format!(
                                "{tag}: call input {} views storage {} but consumes a state of storage {}",
                                value.0, viewed.0, versioned.0
                            ));
                        }
                        if *actual == ParamOwnership::Exclusive {
                            exclusive_leaves.insert(leaf.clone());
                            match boundary.final_states.get(leaf) {
                                Some(token) => {
                                    if graph.states.get(*token) != Some(&viewed) {
                                        errors.push(format!(
                                            "{tag}: call final state {} does not version storage {}",
                                            token.0, viewed.0
                                        ));
                                    }
                                }
                                None => errors.push(format!(
                                    "{tag}: exclusive call input {} has no final state",
                                    value.0
                                )),
                            }
                        }
                    }
                    (None, _) => errors.push(format!(
                        "{tag}: call tensor input {} is not a view of storage",
                        value.0
                    )),
                    (Some(_), None) => errors.push(format!(
                        "{tag}: call input {} consumes state {} which does not exist",
                        value.0, state.0
                    )),
                }
            }
            (CallInput::Value(id), true) => errors.push(format!(
                "{tag}: call input {} is a tensor leaf passed as a value",
                id.0
            )),
            (CallInput::Tensor { value, .. }, false) | (CallInput::Computed { value, .. }, false) => {
                errors.push(format!(
                    "{tag}: call input {} is a non-tensor leaf passed as a tensor",
                    value.0
                ))
            }
        }
    }
    let final_keys: BTreeSet<BoundaryLeaf> = boundary.final_states.keys().cloned().collect();
    if final_keys != exclusive_leaves {
        errors.push(format!(
            "{tag}: a call's final states are not exactly its exclusive tensor leaves"
        ));
    }
    let expected_results = interface_result_leaves(interface);
    let actual_results: Vec<BoundaryLeaf> = boundary.results.keys().cloned().collect();
    if expected_results != actual_results {
        errors.push(format!(
            "{tag}: a call's boundary results are not the callee's canonical result leaves"
        ));
    }
    let output_ids: Vec<GraphValueId> = node.outputs.iter().map(|value| value.id()).collect();
    let result_ids: Vec<GraphValueId> = boundary.results.values().copied().collect();
    if output_ids != result_ids {
        errors.push(format!(
            "{tag}: a call's outputs are not its boundary result values"
        ));
    }
    for value in &result_ids {
        if view_storage(graph, *value).is_some() {
            errors.push(format!(
                "{tag}: call result {} is a view; a call result is a produced value",
                value.0
            ));
        }
    }
    let state_output_ids: Vec<StateTokenId> = node.state_outputs.iter().map(|token| token.id).collect();
    let final_ids: Vec<StateTokenId> = boundary.final_states.values().copied().collect();
    if state_output_ids != final_ids {
        errors.push(format!(
            "{tag}: a call's state outputs are not its boundary final states"
        ));
    }
}
