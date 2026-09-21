//! Occurrence expansion (package O1).
//!
//! `expand` is the body of `OccurrenceForest::expand`: it walks the sealed
//! logical program from the entry choice, assigns one `OccurrenceId` per
//! reachable (parent occurrence, call node) path, instantiates the one
//! boundary schema for the root and every call, records pass-through and
//! capture equality in a temporary alias forest (a union-find whose every
//! union is directed alias -> producer), flattens every identity to
//! canonical ids, builds the canonical leaf registry, and hands the finished
//! tables to `OccurrenceFacts::new`. Nothing here outlives `expand`.
//!
//! Forbidden here: any retained alias map, any `Option` for a required
//! identity, any bare node id keyed outside its graph.

use crate::failure::CompilerDefect;
use crate::ids::{
    CanonicalLeafId, CanonicalStorageId, CanonicalValueId, OccurrenceId, OwnedGraphKey,
    OwnedNodeRef, OwnedStateRef, OwnedStorageRef, OwnedValueRef,
};
use crate::occurrence::{
    defect, CanonicalLeaf, InstantiatedBoundary, InstantiatedFinalState, InstantiatedInput,
    InstantiatedResult, OccurrenceAlternative, OccurrenceFacts, OccurrenceForest, OccurrenceParent,
    OccurrenceRecord,
};
use seismic_lang::abi::RangeEndpoint;
use seismic_lang::logical::boundary::{CallBoundary, CallInput, LogicalBoundaryInput};
use seismic_lang::logical::value::{GraphValueKind, LogicalStorageOwner};
use seismic_lang::logical::{
    ChoiceId, GraphRegion, IdVec, LogicalNode, LogicalNodeKind, LogicalProgram, LoopNode, NodeRef,
    RegionInput, RegionParameter, RegionPath, RegionStep, TaskGraph,
};
use seismic_lang::sir::ParamOwnership;
use seismic_lang::types::{NonEmpty, ValuePath, ValueType};
use std::collections::BTreeMap;
use std::fmt::Debug;

pub(crate) fn expand(logical: &LogicalProgram) -> Result<OccurrenceForest<'_>, CompilerDefect> {
    let mut expander = Expander {
        logical,
        next: 0,
        records: BTreeMap::new(),
        values: AliasForest::new(),
        storages: AliasForest::new(),
        ancestors: Vec::new(),
    };
    let entry = expander.visit(logical.entry_choice, None)?;
    let Expander {
        records,
        values,
        storages,
        ..
    } = expander;
    let occurrences: IdVec<OccurrenceId, OccurrenceRecord> = IdVec::from_iter(records);

    // Every call node of every alternative reaches exactly one occurrence:
    // alternatives of one occurrence have distinct keys and `graph_nodes`
    // yields each node of a graph exactly once, so a duplicate key would
    // contradict the expansion invariant above.
    let mut calls = BTreeMap::new();
    for record in occurrences.iter() {
        for alternative in record.alternatives.iter() {
            for (call, occurrence) in &alternative.calls {
                if calls.insert(call.clone(), *occurrence).is_some() {
                    return Err(defect(format!("{call:?} reaches more than one occurrence")));
                }
            }
        }
    }

    // Flatten values: canonical ids in first-encounter order over every
    // owned graph, the class producer (alias-forest root) first in members.
    let mut canonical_values = BTreeMap::new();
    let mut members: Vec<Vec<OwnedValueRef>> = Vec::new();
    let mut kinds: Vec<GraphValueKind> = Vec::new();
    let mut types: Vec<ValueType> = Vec::new();
    for record in occurrences.iter() {
        for alternative in record.alternatives.iter() {
            let key = alternative.key;
            let graph = logical.graph(alternative.graph);
            for value in graph.values() {
                let owned = OwnedValueRef {
                    graph: key,
                    value: value.id(),
                };
                let producer = values.root(owned)?;
                let canonical = match canonical_values.get(&producer) {
                    Some(id) => *id,
                    None => {
                        let id = CanonicalValueId(members.len() as u32);
                        canonical_values.insert(producer, id);
                        members.push(vec![producer]);
                        let defining =
                            graph_of(logical, &occurrences, producer.graph).value(producer.value);
                        kinds.push(defining.kind().clone());
                        types.push(defining.ty());
                        id
                    }
                };
                if owned != producer {
                    canonical_values.insert(owned, canonical);
                    members[canonical.0 as usize].push(owned);
                    let ty = value.ty();
                    if ty != types[canonical.0 as usize] {
                        return Err(defect(format!(
                            "{owned:?} of type {ty} shares canonical value {} with {producer:?} of type {}",
                            canonical.0,
                            types[canonical.0 as usize]
                        )));
                    }
                }
            }
        }
    }

    // Flatten storages the same way.
    let mut canonical_storages = BTreeMap::new();
    let mut storage_members: Vec<Vec<OwnedStorageRef>> = Vec::new();
    for record in occurrences.iter() {
        for alternative in record.alternatives.iter() {
            let key = alternative.key;
            let graph = logical.graph(alternative.graph);
            for (storage, _) in graph.storages() {
                let owned = OwnedStorageRef {
                    graph: key,
                    storage,
                };
                let owner = storages.root(owned)?;
                let canonical = match canonical_storages.get(&owner) {
                    Some(id) => *id,
                    None => {
                        let id = CanonicalStorageId(storage_members.len() as u32);
                        canonical_storages.insert(owner, id);
                        storage_members.push(vec![owner]);
                        id
                    }
                };
                if owned != owner {
                    canonical_storages.insert(owned, canonical);
                    storage_members[canonical.0 as usize].push(owned);
                }
            }
        }
    }

    // The canonical leaf registry: one semantic leaf per scalar, index,
    // tensor, or capability leaf; two per range (start/end); tuples by
    // ordinal path; void none.
    let mut leaf_table: Vec<CanonicalLeaf> = Vec::new();
    let mut leaves: Vec<Vec<CanonicalLeafId>> = Vec::new();
    for (index, ty) in types.iter().enumerate() {
        let value = CanonicalValueId(index as u32);
        let mut ids = Vec::new();
        for (path, endpoint) in semantic_leaves(ty) {
            let id = CanonicalLeafId(leaf_table.len() as u32);
            leaf_table.push(CanonicalLeaf {
                value,
                path,
                endpoint,
            });
            ids.push(id);
        }
        leaves.push(ids);
    }

    Ok(OccurrenceForest::from_facts(OccurrenceFacts::new(
        logical,
        entry,
        occurrences,
        canonical_values,
        canonical_storages,
        IdVec::new(members),
        IdVec::new(storage_members),
        IdVec::new(leaves),
        IdVec::new(leaf_table),
        IdVec::new(kinds),
        calls,
    )))
}

/// The graph an owned key instantiates, before `OccurrenceFacts` exists.
/// Total: the key was minted by this expansion, whose occurrence ids are
/// dense and whose alternative ordinals index their records.
fn graph_of<'l>(
    logical: &'l LogicalProgram,
    occurrences: &IdVec<OccurrenceId, OccurrenceRecord>,
    key: OwnedGraphKey,
) -> &'l TaskGraph {
    let record = &occurrences[key.occurrence];
    logical.graph(record.alternatives.as_slice()[key.logical_alternative as usize].graph)
}

// ---------------------------------------------------------------------------
// Temporary alias forest
// ---------------------------------------------------------------------------

/// A union-find whose every union is directed: `alias` is declared equal to
/// `producer`, and an identity is the alias side of at most one union (a
/// boundary leaf or capture maps exactly once). Every class is therefore a
/// tree whose root is the class's producer. Discarded before `expand`
/// returns.
struct AliasForest<K> {
    producer_of: BTreeMap<K, K>,
}

impl<K: Ord + Copy + Debug> AliasForest<K> {
    fn new() -> AliasForest<K> {
        AliasForest {
            producer_of: BTreeMap::new(),
        }
    }

    fn alias(&mut self, alias: K, producer: K) -> Result<(), CompilerDefect> {
        if alias == producer {
            return Err(defect(format!("{alias:?} is aliased to itself")));
        }
        if let Some(existing) = self.producer_of.insert(alias, producer) {
            return Err(defect(format!(
                "{alias:?} is aliased twice: to {existing:?} and to {producer:?}"
            )));
        }
        Ok(())
    }

    fn root(&self, mut key: K) -> Result<K, CompilerDefect> {
        let mut steps = 0usize;
        while let Some(next) = self.producer_of.get(&key) {
            steps += 1;
            if steps > self.producer_of.len() {
                return Err(defect(format!("{key:?} is in an alias cycle")));
            }
            key = *next;
        }
        Ok(key)
    }
}

// ---------------------------------------------------------------------------
// Expansion
// ---------------------------------------------------------------------------

struct Expander<'l> {
    logical: &'l LogicalProgram,
    next: u32,
    records: BTreeMap<OccurrenceId, OccurrenceRecord>,
    values: AliasForest<OwnedValueRef>,
    storages: AliasForest<OwnedStorageRef>,
    ancestors: Vec<ChoiceId>,
}

/// The caller side of a non-root occurrence: its parent, the parent's graph,
/// and the call boundary instantiated at the parent's call node.
struct CallerSide<'l> {
    parent: OccurrenceParent,
    graph: &'l TaskGraph,
    boundary: &'l CallBoundary,
}

impl<'l> Expander<'l> {
    fn visit(
        &mut self,
        choice: ChoiceId,
        caller: Option<CallerSide<'l>>,
    ) -> Result<OccurrenceId, CompilerDefect> {
        if self.ancestors.contains(&choice) {
            return Err(defect(format!(
                "choice#{} recurs on its own call path {:?}; a sealed program is acyclic",
                choice.0, self.ancestors
            )));
        }
        let occurrence = OccurrenceId(self.next);
        self.next += 1;
        self.ancestors.push(choice);

        let logical_choice = self.logical.choice(choice);
        let mut alternatives = Vec::new();
        for (ordinal, alternative) in logical_choice.alternatives.iter().enumerate() {
            let key = OwnedGraphKey {
                occurrence,
                logical_alternative: ordinal as u32,
            };
            let graph = self.logical.graph(alternative.graph);
            if graph.choice != choice || graph.alternative != key.logical_alternative {
                return Err(defect(format!(
                    "graph#{} is filed under choice#{} alternative {} but records choice#{} alternative {}",
                    alternative.graph.0, choice.0, ordinal, graph.choice.0, graph.alternative
                )));
            }

            let boundary = match &caller {
                None => root_boundary(key, graph),
                Some(side) => self.call_boundary(key, graph, side)?,
            };
            self.capture_aliases(key, graph.root())?;

            let mut calls = Vec::new();
            for (node, logical) in graph_nodes(graph) {
                let LogicalNodeKind::Call(call) = &logical.kind else {
                    continue;
                };
                let call_ref = OwnedNodeRef { graph: key, node };
                let child = self.visit(
                    call.choice,
                    Some(CallerSide {
                        parent: OccurrenceParent {
                            occurrence,
                            call: call_ref.clone(),
                        },
                        graph,
                        boundary: &call.boundary,
                    }),
                )?;
                calls.push((call_ref, child));
            }

            alternatives.push(OccurrenceAlternative {
                key,
                graph: alternative.graph,
                kind: alternative.kind,
                boundary,
                calls,
                authored_numerics: alternative.authored_numerical_effects.clone(),
            });
        }

        let alternatives = match NonEmpty::new(alternatives) {
            Some(alternatives) => alternatives,
            None => return Err(defect(format!("choice#{} has no alternative", choice.0))),
        };
        self.records.insert(
            occurrence,
            OccurrenceRecord {
                choice,
                parent: caller.map(|side| side.parent),
                interface: logical_choice.interface.clone(),
                alternatives,
            },
        );
        self.ancestors.pop();
        Ok(occurrence)
    }

    /// Instantiate the callee's boundary with the caller's identities and
    /// record the equalities: callee parameter value = caller argument
    /// value, callee parameter storage = caller argument storage (tensor
    /// arguments that carry caller storage), caller call output = callee
    /// result value. Every leaf of either side must be present on the other.
    /// A computed tensor argument aliases the value only: it has no caller
    /// storage, so the callee's parameter storage stays its own canonical
    /// class.
    fn call_boundary(
        &mut self,
        key: OwnedGraphKey,
        graph: &'l TaskGraph,
        side: &CallerSide<'l>,
    ) -> Result<InstantiatedBoundary, CompilerDefect> {
        let caller_key = side.parent.call.graph;
        let caller_graph = side.graph;
        let call = side.boundary;
        let callee = &graph.boundary;

        let mut inputs = BTreeMap::new();
        for (leaf, input) in callee.inputs() {
            let Some(caller_input) = call.inputs.get(leaf) else {
                return Err(defect(format!(
                    "callee input leaf {leaf:?} of {key:?} is absent from the call at {:?}",
                    side.parent.call
                )));
            };
            let instantiated = match (input, caller_input) {
                (LogicalBoundaryInput::Value(callee_value), CallInput::Value(caller_value)) => {
                    let callee_ref = OwnedValueRef {
                        graph: key,
                        value: *callee_value,
                    };
                    let caller_ref = OwnedValueRef {
                        graph: caller_key,
                        value: *caller_value,
                    };
                    self.values.alias(callee_ref, caller_ref)?;
                    InstantiatedInput::Call {
                        callee: callee_ref,
                        caller: caller_ref,
                        states: None,
                        ownership: ParamOwnership::Value,
                    }
                }
                (
                    LogicalBoundaryInput::Tensor {
                        value: callee_value,
                        state: callee_state,
                        ownership,
                    },
                    CallInput::Tensor {
                        value: caller_value,
                        state: caller_state,
                        ownership: caller_ownership,
                    },
                ) => {
                    if ownership != caller_ownership {
                        return Err(defect(format!(
                            "input leaf {leaf:?} of {key:?} is {ownership:?} at the callee and {caller_ownership:?} at the call"
                        )));
                    }
                    let callee_ref = OwnedValueRef {
                        graph: key,
                        value: *callee_value,
                    };
                    let caller_ref = OwnedValueRef {
                        graph: caller_key,
                        value: *caller_value,
                    };
                    self.values.alias(callee_ref, caller_ref)?;
                    let callee_state_ref = OwnedStateRef {
                        graph: key,
                        state: *callee_state,
                    };
                    let caller_state_ref = OwnedStateRef {
                        graph: caller_key,
                        state: *caller_state,
                    };
                    let callee_storage = OwnedStorageRef {
                        graph: key,
                        storage: graph.state_storage(*callee_state),
                    };
                    let caller_storage = OwnedStorageRef {
                        graph: caller_key,
                        storage: caller_graph.state_storage(*caller_state),
                    };
                    let owner = &graph.storage(callee_storage.storage).owner;
                    if *owner != LogicalStorageOwner::Parameter(leaf.clone()) {
                        return Err(defect(format!(
                            "storage {callee_storage:?} behind input leaf {leaf:?} is owned by {owner:?}, not by that leaf"
                        )));
                    }
                    self.storages.alias(callee_storage, caller_storage)?;
                    InstantiatedInput::Call {
                        callee: callee_ref,
                        caller: caller_ref,
                        states: Some((callee_state_ref, caller_state_ref)),
                        ownership: *ownership,
                    }
                }
                (
                    LogicalBoundaryInput::Tensor {
                        value: callee_value,
                        state: _,
                        ownership,
                    },
                    CallInput::Computed {
                        value: caller_value,
                        ownership: caller_ownership,
                    },
                ) => {
                    if ownership != caller_ownership {
                        return Err(defect(format!(
                            "input leaf {leaf:?} of {key:?} is {ownership:?} at the callee and {caller_ownership:?} at the call"
                        )));
                    }
                    if *ownership == ParamOwnership::Exclusive {
                        return Err(defect(format!(
                            "input leaf {leaf:?} of {key:?} is exclusively borrowed but passed as a computed value"
                        )));
                    }
                    let callee_ref = OwnedValueRef {
                        graph: key,
                        value: *callee_value,
                    };
                    let caller_ref = OwnedValueRef {
                        graph: caller_key,
                        value: *caller_value,
                    };
                    self.values.alias(callee_ref, caller_ref)?;
                    InstantiatedInput::Call {
                        callee: callee_ref,
                        caller: caller_ref,
                        states: None,
                        ownership: *ownership,
                    }
                }
                (LogicalBoundaryInput::Value(_), CallInput::Tensor { .. })
                | (LogicalBoundaryInput::Value(_), CallInput::Computed { .. })
                | (LogicalBoundaryInput::Tensor { .. }, CallInput::Value(_)) => {
                    return Err(defect(format!(
                        "input leaf {leaf:?} of {key:?} is a value leaf on one side of the call and a tensor leaf on the other"
                    )))
                }
            };
            inputs.insert(leaf.clone(), instantiated);
        }
        if call.inputs.len() != inputs.len() {
            return Err(defect(format!(
                "the call at {:?} passes {} input leaves but the callee {key:?} declares {}",
                side.parent.call,
                call.inputs.len(),
                inputs.len()
            )));
        }

        let mut results = BTreeMap::new();
        for (leaf, result) in callee.results() {
            let Some(caller_value) = call.results.get(leaf) else {
                return Err(defect(format!(
                    "callee result leaf {leaf:?} of {key:?} is absent from the call at {:?}",
                    side.parent.call
                )));
            };
            let callee_ref = OwnedValueRef {
                graph: key,
                value: result.value,
            };
            let caller_ref = OwnedValueRef {
                graph: caller_key,
                value: *caller_value,
            };
            self.values.alias(caller_ref, callee_ref)?;
            results.insert(
                leaf.clone(),
                InstantiatedResult::Call {
                    callee: callee_ref,
                    caller: caller_ref,
                },
            );
        }
        if call.results.len() != results.len() {
            return Err(defect(format!(
                "the call at {:?} produces {} result leaves but the callee {key:?} declares {}",
                side.parent.call,
                call.results.len(),
                results.len()
            )));
        }

        let mut final_states = BTreeMap::new();
        for (leaf, callee_state) in callee.final_states() {
            let Some(caller_state) = call.final_states.get(leaf) else {
                return Err(defect(format!(
                    "callee final state of leaf {leaf:?} of {key:?} is absent from the call at {:?}",
                    side.parent.call
                )));
            };
            final_states.insert(
                leaf.clone(),
                InstantiatedFinalState::Call {
                    callee: OwnedStateRef {
                        graph: key,
                        state: *callee_state,
                    },
                    caller: OwnedStateRef {
                        graph: caller_key,
                        state: *caller_state,
                    },
                },
            );
        }
        if call.final_states.len() != final_states.len() {
            return Err(defect(format!(
                "the call at {:?} receives {} final states but the callee {key:?} declares {}",
                side.parent.call,
                call.final_states.len(),
                final_states.len()
            )));
        }

        Ok(InstantiatedBoundary {
            inputs,
            results,
            final_states,
        })
    }

    /// Region captures within one owned graph: an `if` capture parameter
    /// equals its outer value; a loop's invariant value parameters equal the
    /// loop's `invariant_values`, paired in order after excluding the binder
    /// and every value carry. Carries are never aliased: a carried value is a
    /// distinct SSA per visit, and a carried state already names its storage.
    fn capture_aliases(
        &mut self,
        key: OwnedGraphKey,
        region: &GraphRegion,
    ) -> Result<(), CompilerDefect> {
        for node in region.nodes.iter() {
            match &node.kind {
                LogicalNodeKind::If(if_node) => {
                    for capture in &if_node.captured {
                        self.values.alias(
                            OwnedValueRef {
                                graph: key,
                                value: capture.parameter,
                            },
                            OwnedValueRef {
                                graph: key,
                                value: capture.outer,
                            },
                        )?;
                    }
                    self.capture_aliases(key, &if_node.then_region)?;
                    self.capture_aliases(key, &if_node.else_region)?;
                }
                LogicalNodeKind::Loop(loop_node) => {
                    self.loop_invariant_aliases(key, loop_node)?;
                    self.capture_aliases(key, &loop_node.body)?;
                }
                LogicalNodeKind::Primitive(_)
                | LogicalNodeKind::Reduction(_)
                | LogicalNodeKind::Call(_) => {}
            }
        }
        Ok(())
    }

    fn loop_invariant_aliases(
        &mut self,
        key: OwnedGraphKey,
        loop_node: &LoopNode,
    ) -> Result<(), CompilerDefect> {
        let mut carried_parameters = Vec::new();
        for slot in &loop_node.carried {
            match slot.initial {
                RegionInput::Value(_) => carried_parameters.push(slot.body_parameter.index()),
                RegionInput::State(_) => {}
            }
        }
        let mut invariants = loop_node.invariant_values.iter().copied();
        for (index, parameter) in loop_node.body.parameters.iter().enumerate() {
            let RegionParameter::Value { id, .. } = parameter else {
                continue;
            };
            if *id == loop_node.binder || carried_parameters.contains(&index) {
                continue;
            }
            let Some(outer) = invariants.next() else {
                return Err(defect(format!(
                    "loop body value parameter {} of {key:?} has no invariant outer value",
                    id.0
                )));
            };
            self.values.alias(
                OwnedValueRef {
                    graph: key,
                    value: *id,
                },
                OwnedValueRef {
                    graph: key,
                    value: outer,
                },
            )?;
        }
        if let Some(outer) = invariants.next() {
            return Err(defect(format!(
                "loop invariant value {} of {key:?} has no body parameter",
                outer.0
            )));
        }
        Ok(())
    }
}

/// The root boundary: the graph's own contract instantiated with ABI leaves.
fn root_boundary(key: OwnedGraphKey, graph: &TaskGraph) -> InstantiatedBoundary {
    let boundary = &graph.boundary;
    let inputs = boundary
        .inputs()
        .iter()
        .map(|(leaf, input)| {
            let instantiated = match input {
                LogicalBoundaryInput::Value(value) => InstantiatedInput::Root {
                    callee: OwnedValueRef {
                        graph: key,
                        value: *value,
                    },
                    callee_state: None,
                    ownership: ParamOwnership::Value,
                },
                LogicalBoundaryInput::Tensor {
                    value,
                    state,
                    ownership,
                } => InstantiatedInput::Root {
                    callee: OwnedValueRef {
                        graph: key,
                        value: *value,
                    },
                    callee_state: Some(OwnedStateRef {
                        graph: key,
                        state: *state,
                    }),
                    ownership: *ownership,
                },
            };
            (leaf.clone(), instantiated)
        })
        .collect();
    let results = boundary
        .results()
        .iter()
        .map(|(leaf, result)| {
            (
                leaf.clone(),
                InstantiatedResult::Root {
                    callee: OwnedValueRef {
                        graph: key,
                        value: result.value,
                    },
                },
            )
        })
        .collect();
    let final_states = boundary
        .final_states()
        .iter()
        .map(|(leaf, state)| {
            (
                leaf.clone(),
                InstantiatedFinalState::Root {
                    callee: OwnedStateRef {
                        graph: key,
                        state: *state,
                    },
                },
            )
        })
        .collect();
    InstantiatedBoundary {
        inputs,
        results,
        final_states,
    }
}

/// Every node of a graph in graph order (pre-order over nested regions),
/// with its region-qualified reference.
pub(crate) fn graph_nodes(graph: &TaskGraph) -> Vec<(NodeRef, &LogicalNode)> {
    fn walk<'g>(
        region: &'g GraphRegion,
        path: &RegionPath,
        out: &mut Vec<(NodeRef, &'g LogicalNode)>,
    ) {
        for (id, node) in region.nodes.entries() {
            out.push((
                NodeRef {
                    region: path.clone(),
                    node: id,
                },
                node,
            ));
            match &node.kind {
                LogicalNodeKind::If(if_node) => {
                    let mut then_path = path.clone();
                    then_path.push(RegionStep::IfThen(id));
                    walk(&if_node.then_region, &then_path, out);
                    let mut else_path = path.clone();
                    else_path.push(RegionStep::IfElse(id));
                    walk(&if_node.else_region, &else_path, out);
                }
                LogicalNodeKind::Loop(loop_node) => {
                    let mut body_path = path.clone();
                    body_path.push(RegionStep::LoopBody(id));
                    walk(&loop_node.body, &body_path, out);
                }
                LogicalNodeKind::Primitive(_)
                | LogicalNodeKind::Reduction(_)
                | LogicalNodeKind::Call(_) => {}
            }
        }
    }
    let mut out = Vec::new();
    walk(graph.root(), &Vec::new(), &mut out);
    out
}

/// The canonical semantic leaves of one value type, as physical leaf
/// references: the ordinal-path traversal of `seismic_lang::types::
/// canonical_leaves` (tuples by ordinal, void none) with a range expanded to
/// its two endpoints and a capability value admitted as one leaf. The
/// registry must be total over every graph value, and a capability value
/// (alone or as a tuple component) is a graph value even though it never
/// crosses the public ABI.
fn semantic_leaves(ty: &ValueType) -> Vec<(ValuePath, Option<RangeEndpoint>)> {
    fn walk(ty: &ValueType, path: &ValuePath, out: &mut Vec<(ValuePath, Option<RangeEndpoint>)>) {
        match ty {
            ValueType::Scalar(_)
            | ValueType::Index { .. }
            | ValueType::Tensor(_)
            | ValueType::CapabilityValue(_) => out.push((path.clone(), None)),
            ValueType::Range { .. } => {
                out.push((path.clone(), Some(RangeEndpoint::Start)));
                out.push((path.clone(), Some(RangeEndpoint::End)));
            }
            ValueType::Tuple(items) => {
                for (index, item) in items.iter().enumerate() {
                    walk(item, &path.extend(index as u32), out);
                }
            }
            ValueType::Void => {}
        }
    }
    let mut out = Vec::new();
    walk(ty, &ValuePath::default(), &mut out);
    out
}
