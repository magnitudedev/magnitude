//! Occurrence-qualified identities and instantiated boundaries (package O1).
//!
//! `OccurrenceForest::expand` is the only constructor. It expands every
//! static call occurrence of the sealed logical program, assigns
//! occurrence-qualified identities, instantiates the one boundary schema for
//! the root (ABI residences) and every call (caller identities), resolves
//! pass-through/capture equality once with a temporary union-find, and emits
//! only canonical identities. No alias map, union-find, or unqualified node
//! reference escapes.
//!
//! Expansion validates every structural agreement of the program it walks
//! (boundary pairing, capture pairing, alias direction, graph filing) and
//! reports a contradicted invariant as a `CompilerDefect` return of this
//! package (spec §4), never a panic. Lookups over the finished forest are
//! total for every reference expansion minted: densely minted tables are
//! indexed infallibly, and keyed or structural lookups return a
//! `CompilerDefect` for a reference no producer could have minted.

use crate::failure::{CompilerDefect, Package};
use crate::ids::{
    CanonicalLeafId, CanonicalStorageId, CanonicalValueId, OccurrenceId, OwnedGraphKey,
    OwnedNodeRef, OwnedRegionRef, OwnedStateRef, OwnedStorageRef, OwnedValueLeafRef, OwnedValueRef,
    OwnedViewRef,
};
use seismic_lang::abi::RangeEndpoint;
use seismic_lang::logical::boundary::BoundaryLeaf;
use seismic_lang::logical::specialization::ShapeField;
use seismic_lang::logical::value::{GraphValue, GraphValueKind};
use seismic_lang::logical::{
    ChoiceId, FunctionInterface, GraphId, GraphRegion, IdVec, ImplementationKind, LogicalNode,
    LogicalNodeKind, LogicalProgram, LogicalStorage, LogicalView, NumericalTransfer, RegionStep,
    RuntimeExtent, ShapeFieldId, TaskGraph,
};
use seismic_lang::sir::ParamOwnership;
use seismic_lang::types::{NonEmpty, RuntimeExtentId, ValuePath, ValueType};
use std::collections::BTreeMap;
use std::fmt::Display;

/// A contradicted O1 invariant, as a returned defect of this package.
pub(crate) fn defect(invariant: impl Display) -> CompilerDefect {
    CompilerDefect::new(Package::O1, invariant.to_string())
}

/// The parent call of a non-root occurrence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OccurrenceParent {
    pub occurrence: OccurrenceId,
    pub call: OwnedNodeRef,
}

/// One input leaf of an instantiated boundary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InstantiatedInput {
    /// A root input leaf: the callee value and its ABI leaf. A tensor leaf
    /// names its entry state; a value leaf has none (`ownership` is `Value`).
    Root {
        callee: OwnedValueRef,
        callee_state: Option<OwnedStateRef>,
        ownership: ParamOwnership,
    },
    /// A call input leaf: the callee value equals the caller argument value
    /// (one canonical identity). Tensor leaves that carry caller storage on
    /// both sides also pair entry states `(callee, caller)`, whose storages
    /// share one canonical identity; a value leaf and a computed tensor
    /// argument have none (`states` is `None`).
    Call {
        callee: OwnedValueRef,
        caller: OwnedValueRef,
        states: Option<(OwnedStateRef, OwnedStateRef)>,
        ownership: ParamOwnership,
    },
}

/// One result leaf of an instantiated boundary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InstantiatedResult {
    /// A root result leaf: the callee value published to the ABI result.
    Root { callee: OwnedValueRef },
    /// A call result leaf: the callee value equals the caller's call output
    /// (one canonical identity).
    Call {
        callee: OwnedValueRef,
        caller: OwnedValueRef,
    },
}

/// One final mutable state of an instantiated boundary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InstantiatedFinalState {
    Root {
        callee: OwnedStateRef,
    },
    Call {
        callee: OwnedStateRef,
        caller: OwnedStateRef,
    },
}

/// The one boundary schema instantiated for a root or a call.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InstantiatedBoundary {
    pub inputs: BTreeMap<BoundaryLeaf, InstantiatedInput>,
    pub results: BTreeMap<BoundaryLeaf, InstantiatedResult>,
    pub final_states: BTreeMap<BoundaryLeaf, InstantiatedFinalState>,
}

/// One logical alternative of one occurrence.
#[derive(Clone, Debug, PartialEq)]
pub struct OccurrenceAlternative {
    pub key: OwnedGraphKey,
    /// The immutable logical graph this alternative instantiates.
    pub graph: GraphId,
    pub kind: ImplementationKind,
    pub boundary: InstantiatedBoundary,
    /// Every call node of this alternative's graph with the occurrence it
    /// reaches, in graph order.
    pub calls: Vec<(OwnedNodeRef, OccurrenceId)>,
    pub authored_numerics: Vec<NumericalTransfer>,
}

/// One statically expanded occurrence.
#[derive(Clone, Debug, PartialEq)]
pub struct OccurrenceRecord {
    pub choice: ChoiceId,
    pub parent: Option<OccurrenceParent>,
    pub interface: FunctionInterface,
    pub alternatives: NonEmpty<OccurrenceAlternative>,
}

/// One canonical semantic leaf: the leaf of a canonical value, with its path
/// and (for ranges) endpoint. Tuple and range values expand through this
/// registry; every non-void value has at least one leaf.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CanonicalLeaf {
    pub value: CanonicalValueId,
    pub path: ValuePath,
    pub endpoint: Option<RangeEndpoint>,
}

/// The immutable occurrence facts every later phase and every mapping rule
/// queries. Lookups are total over the sealed forest for every reference a
/// producer minted: occurrence and alternative keys are minted densely by
/// expansion (indexed infallibly); keyed and structural lookups return a
/// `CompilerDefect` for a reference expansion did not mint.
pub struct OccurrenceFacts<'l> {
    logical: &'l LogicalProgram,
    entry: OccurrenceId,
    occurrences: IdVec<OccurrenceId, OccurrenceRecord>,
    canonical_values: BTreeMap<OwnedValueRef, CanonicalValueId>,
    canonical_storages: BTreeMap<OwnedStorageRef, CanonicalStorageId>,
    /// Canonical value -> its members, the producer first.
    members: IdVec<CanonicalValueId, Vec<OwnedValueRef>>,
    /// Canonical storage -> its members, the owning (outermost) storage first.
    storage_members: IdVec<CanonicalStorageId, Vec<OwnedStorageRef>>,
    /// Canonical value -> its leaves, in canonical traversal order.
    leaves: IdVec<CanonicalValueId, Vec<CanonicalLeafId>>,
    leaf_table: IdVec<CanonicalLeafId, CanonicalLeaf>,
    /// Canonical value -> the value kind of the class's producer.
    kinds: IdVec<CanonicalValueId, GraphValueKind>,
    /// Owned call node -> the one occurrence it reaches.
    calls: BTreeMap<OwnedNodeRef, OccurrenceId>,
}

impl<'l> OccurrenceFacts<'l> {
    pub fn logical(&self) -> &'l LogicalProgram {
        self.logical
    }

    pub fn entry(&self) -> OccurrenceId {
        self.entry
    }

    /// The record of one occurrence. Total: expansion mints occurrence ids
    /// densely (`0..len`), and every id in circulation was minted there.
    pub fn occurrence(&self, id: OccurrenceId) -> &OccurrenceRecord {
        &self.occurrences[id]
    }

    pub fn occurrences(&self) -> impl Iterator<Item = (OccurrenceId, &OccurrenceRecord)> + '_ {
        self.occurrences.entries()
    }

    /// The alternative instantiated at this occurrence-qualified key. Total:
    /// every `OwnedGraphKey` in circulation is the `key` of an alternative
    /// this forest minted, whose ordinal indexes that record's alternatives.
    pub fn alternative(&self, key: OwnedGraphKey) -> &OccurrenceAlternative {
        let record = self.occurrence(key.occurrence);
        &record.alternatives.as_slice()[key.logical_alternative as usize]
    }

    /// The immutable graph instantiated at this occurrence-qualified key.
    pub fn graph(&self, key: OwnedGraphKey) -> &'l TaskGraph {
        self.logical.graph(self.alternative(key).graph)
    }

    /// The region at `path` of an owned graph, walked from its root. Every
    /// region path in circulation was minted by logical construction while
    /// walking the regions it names, so each step matches its node kind; a
    /// path that does not is a defect return.
    fn region_at(
        &self,
        graph: OwnedGraphKey,
        path: &[RegionStep],
    ) -> Result<&'l GraphRegion, CompilerDefect> {
        let mut current = self.graph(graph).root();
        for step in path {
            let Some(node) = current.nodes.get(step.node()) else {
                return Err(defect(format!(
                    "region step {step:?} of {graph:?} names a node outside its region"
                )));
            };
            current = match (step, &node.kind) {
                (RegionStep::IfThen(_), LogicalNodeKind::If(if_node)) => &if_node.then_region,
                (RegionStep::IfElse(_), LogicalNodeKind::If(if_node)) => &if_node.else_region,
                (RegionStep::LoopBody(_), LogicalNodeKind::Loop(loop_node)) => &loop_node.body,
                (
                    RegionStep::IfThen(_) | RegionStep::IfElse(_),
                    LogicalNodeKind::Loop(_)
                    | LogicalNodeKind::Primitive(_)
                    | LogicalNodeKind::Reduction(_)
                    | LogicalNodeKind::Call(_),
                )
                | (
                    RegionStep::LoopBody(_),
                    LogicalNodeKind::If(_)
                    | LogicalNodeKind::Primitive(_)
                    | LogicalNodeKind::Reduction(_)
                    | LogicalNodeKind::Call(_),
                ) => {
                    return Err(defect(format!(
                        "region step {step:?} of {graph:?} does not match its node kind"
                    )))
                }
            };
        }
        Ok(current)
    }

    pub fn region(&self, region: &OwnedRegionRef) -> Result<&'l GraphRegion, CompilerDefect> {
        self.region_at(region.graph, &region.region)
    }

    pub fn node(&self, node: &OwnedNodeRef) -> Result<&'l LogicalNode, CompilerDefect> {
        let region = self.region_at(node.graph, &node.node.region)?;
        match region.nodes.get(node.node.node) {
            Some(logical) => Ok(logical),
            None => Err(defect(format!("{node:?} names a node outside its region"))),
        }
    }

    /// The logical value an owned reference names.
    pub fn value(&self, value: OwnedValueRef) -> &'l GraphValue {
        self.graph(value.graph).value(value.value)
    }

    pub fn storage(&self, storage: OwnedStorageRef) -> &'l LogicalStorage {
        self.graph(storage.graph).storage(storage.storage)
    }

    pub fn view(&self, view: OwnedViewRef) -> &'l LogicalView {
        self.graph(view.graph).view(view.view)
    }

    /// The storage a state token names.
    pub fn state_storage(&self, state: OwnedStateRef) -> OwnedStorageRef {
        OwnedStorageRef {
            graph: state.graph,
            storage: self.graph(state.graph).state_storage(state.state),
        }
    }

    /// The canonical identity of one owned value. Every owned value
    /// reference in circulation names a value of a minted alternative, and
    /// expansion canonicalized every such value.
    pub fn canonical_value(
        &self,
        value: OwnedValueRef,
    ) -> Result<CanonicalValueId, CompilerDefect> {
        match self.canonical_values.get(&value) {
            Some(id) => Ok(*id),
            None => Err(defect(format!("{value:?} has no canonical value"))),
        }
    }

    /// The canonical identity of one owned storage. Every owned storage
    /// reference in circulation names a storage of a minted alternative, and
    /// expansion canonicalized every such storage.
    pub fn canonical_storage(
        &self,
        storage: OwnedStorageRef,
    ) -> Result<CanonicalStorageId, CompilerDefect> {
        match self.canonical_storages.get(&storage) {
            Some(id) => Ok(*id),
            None => Err(defect(format!("{storage:?} has no canonical storage"))),
        }
    }

    /// The members of one canonical value class, its producer first.
    pub fn members(&self, value: CanonicalValueId) -> &[OwnedValueRef] {
        &self.members[value]
    }

    /// The members of one canonical storage class, the owning storage first.
    pub fn storage_members(&self, storage: CanonicalStorageId) -> &[OwnedStorageRef] {
        &self.storage_members[storage]
    }

    pub fn value_kind(&self, value: CanonicalValueId) -> &GraphValueKind {
        &self.kinds[value]
    }

    /// The value type shared by every member of the class.
    pub fn value_type(&self, value: CanonicalValueId) -> ValueType {
        self.value(self.members[value][0]).ty()
    }

    /// The canonical leaves of one canonical value (empty for void).
    pub fn leaves(&self, value: CanonicalValueId) -> &[CanonicalLeafId] {
        &self.leaves[value]
    }

    pub fn leaf(&self, leaf: CanonicalLeafId) -> &CanonicalLeaf {
        &self.leaf_table[leaf]
    }

    /// The canonical leaf of one owned leaf reference. Every owned leaf in
    /// circulation combines a minted owned value with a path and endpoint of
    /// that value's canonical traversal.
    pub fn canonical_leaf(
        &self,
        leaf: &OwnedValueLeafRef,
    ) -> Result<CanonicalLeafId, CompilerDefect> {
        let value = self.canonical_value(leaf.value)?;
        let found = self.leaves[value].iter().copied().find(|id| {
            let canonical = &self.leaf_table[*id];
            canonical.path == leaf.path && canonical.endpoint == leaf.endpoint
        });
        match found {
            Some(id) => Ok(id),
            None => Err(defect(format!(
                "{leaf:?} names no leaf of canonical value {}",
                value.0
            ))),
        }
    }

    /// The occurrence reached by an owned call node. Every owned call node
    /// in circulation was recorded by the alternative that owns it.
    pub fn call_occurrence(&self, call: &OwnedNodeRef) -> Result<OccurrenceId, CompilerDefect> {
        match self.calls.get(call) {
            Some(occurrence) => Ok(*occurrence),
            None => Err(defect(format!("{call:?} is not a call node of the forest"))),
        }
    }

    pub fn runtime_extent(&self, id: RuntimeExtentId) -> &'l RuntimeExtent {
        self.logical.runtime_extent(id)
    }

    pub fn shape_fields(&self) -> &'l IdVec<ShapeFieldId, ShapeField> {
        &self.logical.shape_fields
    }
}

/// The occurrence forest of one logical program.
pub struct OccurrenceForest<'l> {
    facts: OccurrenceFacts<'l>,
}

impl<'l> OccurrenceForest<'l> {
    /// The only constructor. Validates every structural agreement of the
    /// sealed logical program; a contradiction is a `CompilerDefect` of this
    /// package (spec §4).
    pub fn expand(logical: &'l LogicalProgram) -> Result<OccurrenceForest<'l>, CompilerDefect> {
        crate::formation::occurrence::expand(logical)
    }

    pub fn facts(&self) -> &OccurrenceFacts<'l> {
        &self.facts
    }

    pub(crate) fn from_facts(facts: OccurrenceFacts<'l>) -> OccurrenceForest<'l> {
        OccurrenceForest { facts }
    }
}

impl<'l> OccurrenceFacts<'l> {
    /// Private constructor for `formation::occurrence`. The call index is
    /// the one derived table (every call node of every alternative reaches
    /// exactly one occurrence); formation validates its uniqueness before
    /// calling this.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        logical: &'l LogicalProgram,
        entry: OccurrenceId,
        occurrences: IdVec<OccurrenceId, OccurrenceRecord>,
        canonical_values: BTreeMap<OwnedValueRef, CanonicalValueId>,
        canonical_storages: BTreeMap<OwnedStorageRef, CanonicalStorageId>,
        members: IdVec<CanonicalValueId, Vec<OwnedValueRef>>,
        storage_members: IdVec<CanonicalStorageId, Vec<OwnedStorageRef>>,
        leaves: IdVec<CanonicalValueId, Vec<CanonicalLeafId>>,
        leaf_table: IdVec<CanonicalLeafId, CanonicalLeaf>,
        kinds: IdVec<CanonicalValueId, GraphValueKind>,
        calls: BTreeMap<OwnedNodeRef, OccurrenceId>,
    ) -> OccurrenceFacts<'l> {
        OccurrenceFacts {
            logical,
            entry,
            occurrences,
            canonical_values,
            canonical_storages,
            members,
            storage_members,
            leaves,
            leaf_table,
            kinds,
            calls,
        }
    }
}

impl seismic_lang::logical::IdIndex for OccurrenceId {
    fn from_index(index: usize) -> Self {
        OccurrenceId(index as u32)
    }
    fn index(self) -> usize {
        self.0 as usize
    }
}

impl seismic_lang::logical::IdIndex for CanonicalValueId {
    fn from_index(index: usize) -> Self {
        CanonicalValueId(index as u32)
    }
    fn index(self) -> usize {
        self.0 as usize
    }
}

impl seismic_lang::logical::IdIndex for CanonicalLeafId {
    fn from_index(index: usize) -> Self {
        CanonicalLeafId(index as u32)
    }
    fn index(self) -> usize {
        self.0 as usize
    }
}

impl seismic_lang::logical::IdIndex for CanonicalStorageId {
    fn from_index(index: usize) -> Self {
        CanonicalStorageId(index as u32)
    }
    fn index(self) -> usize {
        self.0 as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formation::occurrence::graph_nodes;
    use seismic_lang::logical::specialization::{ShapeBinding, SpecializationDomain};
    use seismic_lang::logical::value::{LogicalStorageOwner, TensorSource, ViewBase};
    use seismic_lang::logical::{
        construct, EffectiveTargetIdentity, LoopNode, RegionInput, RegionParameter,
    };
    use seismic_lang::program::{compile, SourceFile};
    use seismic_lang::sir::Program;
    use std::collections::BTreeSet;

    fn check(source: &str) -> Program {
        let files = vec![SourceFile {
            path: "test.seismic".to_string(),
            text: source.to_string(),
        }];
        compile(&files).unwrap_or_else(|diagnostics| {
            panic!(
                "the source checks: {}",
                diagnostics
                    .iter()
                    .map(|d| d.render())
                    .collect::<Vec<_>>()
                    .join("\n")
            )
        })
    }

    fn supports_all(_: &seismic_lang::sir::IntrinsicUse) -> Result<(), String> {
        Ok(())
    }

    fn exact(program: &Program, entry: &str, entries: &[(&str, u64)]) -> SpecializationDomain {
        let shapes = entries
            .iter()
            .map(|(name, value)| (name.to_string(), ShapeBinding::Exact(*value)))
            .collect();
        SpecializationDomain::new(program, entry, shapes, BTreeMap::new())
            .expect("the exact domain binds every entry parameter")
    }

    fn logical(source: &str, entry: &str, shapes: &[(&str, u64)]) -> LogicalProgram {
        let program = check(source);
        let domain = exact(&program, entry, shapes);
        let target = EffectiveTargetIdentity {
            backend: "cpu".to_string(),
            capability_fingerprint: "test-fingerprint".to_string(),
        };
        construct(&program, &target, &supports_all, &domain).expect("construction succeeds")
    }

    fn key(occurrence: u32, alternative: u32) -> OwnedGraphKey {
        OwnedGraphKey {
            occurrence: OccurrenceId(occurrence),
            logical_alternative: alternative,
        }
    }

    /// Every value of every owned graph has a canonical identity, an agreed
    /// type, and one leaf per semantic leaf; every storage has a canonical
    /// storage; every call node reaches its recorded occurrence.
    fn assert_total(facts: &OccurrenceFacts<'_>) {
        for (id, record) in facts.occurrences() {
            for alternative in record.alternatives.iter() {
                assert_eq!(alternative.key.occurrence, id);
                let graph = facts.graph(alternative.key);
                for value in graph.values() {
                    let owned = OwnedValueRef {
                        graph: alternative.key,
                        value: value.id(),
                    };
                    let canonical = facts.canonical_value(owned).unwrap();
                    assert_eq!(facts.value_type(canonical), value.ty());
                    assert!(facts.members(canonical).contains(&owned));
                    let leaf_count = facts.leaves(canonical).len();
                    match value.kind() {
                        GraphValueKind::Void => assert_eq!(leaf_count, 0),
                        GraphValueKind::Range { .. } => assert_eq!(leaf_count, 2),
                        GraphValueKind::Scalar(_)
                        | GraphValueKind::Index { .. }
                        | GraphValueKind::Capability(_)
                        | GraphValueKind::Tensor { .. } => {
                            assert_eq!(leaf_count, 1);
                            let leaf = facts
                                .canonical_leaf(&OwnedValueLeafRef {
                                    value: owned,
                                    path: ValuePath::default(),
                                    endpoint: None,
                                })
                                .unwrap();
                            assert_eq!(facts.leaf(leaf).value, canonical);
                        }
                        GraphValueKind::Tuple(items) => assert!(leaf_count >= items.len()),
                    }
                }
                for (storage, _) in graph.storages() {
                    let owned = OwnedStorageRef {
                        graph: alternative.key,
                        storage,
                    };
                    let canonical = facts.canonical_storage(owned).unwrap();
                    assert!(facts.storage_members(canonical).contains(&owned));
                }
                for (call, occurrence) in &alternative.calls {
                    assert_eq!(facts.call_occurrence(call).unwrap(), *occurrence);
                    assert!(matches!(
                        facts.node(call).unwrap().kind,
                        LogicalNodeKind::Call(_)
                    ));
                    let parent = facts
                        .occurrence(*occurrence)
                        .parent
                        .as_ref()
                        .expect("a called occurrence has a parent");
                    assert_eq!(parent.occurrence, id);
                    assert_eq!(&parent.call, call);
                }
            }
        }
        // Canonicalization is idempotent: every member of a class maps back
        // to that class, and the producer is the first member.
        for (_, record) in facts.occurrences() {
            for alternative in record.alternatives.iter() {
                for value in facts.graph(alternative.key).values() {
                    let owned = OwnedValueRef {
                        graph: alternative.key,
                        value: value.id(),
                    };
                    let canonical = facts.canonical_value(owned).unwrap();
                    for member in facts.members(canonical) {
                        assert_eq!(facts.canonical_value(*member).unwrap(), canonical);
                    }
                    let producer = facts.members(canonical)[0];
                    assert_eq!(facts.value_kind(canonical), facts.value(producer).kind());
                }
            }
        }
    }

    fn loop_nodes<'g>(graph: &'g TaskGraph) -> Vec<&'g LoopNode> {
        graph_nodes(graph)
            .into_iter()
            .filter_map(|(_, node)| match &node.kind {
                LogicalNodeKind::Loop(loop_node) => Some(loop_node),
                _ => None,
            })
            .collect()
    }

    const KERNEL: &str = "fn add[M, N](x: &tensor[M, N] f32, y: &tensor[M, N] f32, result: tensor[M, N] f32) -> tensor[M, N] f32:\n    let mut output = result\n    parallel for row in 0..M:\n        for col in 0..N:\n            output[row, col] = x[row, col] + y[row, col]\n    return output\n\nfn linear[M, N](x: &tensor[M, N] f32, y: &tensor[M, N] f32, result: tensor[M, N] f32) -> tensor[M, N] f32:\n    return add(x, y, result)\n";

    #[test]
    fn nested_call_shares_one_identity_across_the_boundary() {
        let logical = logical(KERNEL, "linear", &[("M", 4), ("N", 8)]);
        let forest = OccurrenceForest::expand(&logical).expect("expansion succeeds");
        let facts = forest.facts();
        assert_total(facts);

        assert_eq!(facts.entry(), OccurrenceId(0));
        assert_eq!(facts.occurrences().count(), 2);
        let entry = facts.occurrence(OccurrenceId(0));
        assert_eq!(entry.choice, logical.entry_choice);
        assert!(entry.parent.is_none());
        assert_eq!(entry.interface.name, "linear");
        let child = facts.occurrence(OccurrenceId(1));
        assert_eq!(child.interface.name, "add");
        let parent = child.parent.as_ref().expect("the call has a parent");
        assert_eq!(parent.occurrence, OccurrenceId(0));

        // The entry alternative has one call reaching the child.
        let entry_alternative = facts.alternative(key(0, 0));
        assert_eq!(entry_alternative.calls.len(), 1);
        assert_eq!(entry_alternative.calls[0].1, OccurrenceId(1));
        assert_eq!(entry_alternative.calls[0].0, parent.call);

        // The root boundary is instantiated with ABI leaves: three tensor
        // inputs with entry states, one result, no final state.
        assert_eq!(entry_alternative.boundary.inputs.len(), 3);
        for input in entry_alternative.boundary.inputs.values() {
            let InstantiatedInput::Root {
                callee,
                callee_state,
                ownership,
            } = input
            else {
                panic!("the entry instantiates root inputs");
            };
            let state = callee_state.expect("a tensor leaf has an entry state");
            assert_eq!(callee.graph, key(0, 0));
            assert_eq!(state.graph, key(0, 0));
            assert_ne!(*ownership, ParamOwnership::Value);
            let storage = facts.state_storage(state);
            assert!(matches!(
                facts.storage(storage).owner,
                LogicalStorageOwner::Parameter(_)
            ));
        }
        assert_eq!(entry_alternative.boundary.results.len(), 1);
        assert!(entry_alternative.boundary.final_states.is_empty());

        // The call boundary pairs callee and caller identities into one
        // canonical value per input leaf and one canonical storage per
        // tensor leaf; the result leaf's class is produced by the callee.
        let child_alternative = facts.alternative(key(1, 0));
        assert_eq!(child_alternative.boundary.inputs.len(), 3);
        for input in child_alternative.boundary.inputs.values() {
            let InstantiatedInput::Call {
                callee,
                caller,
                states,
                ..
            } = input
            else {
                panic!("the child instantiates call inputs");
            };
            assert_eq!(callee.graph, key(1, 0));
            assert_eq!(caller.graph, key(0, 0));
            let canonical = facts.canonical_value(*callee).unwrap();
            assert_eq!(facts.canonical_value(*caller).unwrap(), canonical);
            assert_eq!(facts.members(canonical)[0], *caller);
            assert!(facts.members(canonical).contains(callee));
            let (callee_state, caller_state) = states.expect("tensor leaves pair states");
            assert_eq!(
                facts
                    .canonical_storage(facts.state_storage(callee_state))
                    .unwrap(),
                facts
                    .canonical_storage(facts.state_storage(caller_state))
                    .unwrap()
            );
            assert_eq!(
                facts.storage_members(
                    facts
                        .canonical_storage(facts.state_storage(callee_state))
                        .unwrap()
                )[0],
                facts.state_storage(caller_state)
            );
        }
        let (_, result) = child_alternative
            .boundary
            .results
            .iter()
            .next()
            .expect("one result leaf");
        let InstantiatedResult::Call { callee, caller } = result else {
            panic!("the child instantiates call results");
        };
        let canonical = facts.canonical_value(*callee).unwrap();
        assert_eq!(facts.canonical_value(*caller).unwrap(), canonical);
        assert_eq!(facts.members(canonical)[0], *callee);
        assert_eq!(facts.value_kind(canonical), facts.value(*callee).kind());
        assert!(matches!(
            facts.value_kind(canonical),
            GraphValueKind::Tensor {
                source: TensorSource::View(_),
                ..
            }
        ));
        assert_eq!(
            facts.value(*caller).tensor_source(),
            Some(TensorSource::Computed)
        );

        // Loop invariants share the outer identity: the body's captured
        // value parameters (binder and carries excluded) are exactly the
        // classes of the loop's invariant values.
        let add_graph = facts.graph(key(1, 0));
        let loops = loop_nodes(add_graph);
        assert!(!loops.is_empty());
        for loop_node in loops {
            assert!(!loop_node.invariant_values.is_empty());
            let carried: BTreeSet<usize> = loop_node
                .carried
                .iter()
                .filter(|slot| matches!(slot.initial, RegionInput::Value(_)))
                .map(|slot| slot.body_parameter.index())
                .collect();
            let parameters: BTreeSet<CanonicalValueId> = loop_node
                .body
                .parameters
                .iter()
                .enumerate()
                .filter_map(|(index, parameter)| match parameter {
                    RegionParameter::Value { id, .. }
                        if *id != loop_node.binder && !carried.contains(&index) =>
                    {
                        Some(
                            facts
                                .canonical_value(OwnedValueRef {
                                    graph: key(1, 0),
                                    value: *id,
                                })
                                .unwrap(),
                        )
                    }
                    RegionParameter::Value { .. } | RegionParameter::State { .. } => None,
                })
                .collect();
            let outers: BTreeSet<CanonicalValueId> = loop_node
                .invariant_values
                .iter()
                .map(|outer| {
                    facts
                        .canonical_value(OwnedValueRef {
                            graph: key(1, 0),
                            value: *outer,
                        })
                        .unwrap()
                })
                .collect();
            assert_eq!(parameters, outers);
            // The binder is the producer of its class (a nested loop may
            // capture it as an invariant, never the other way round).
            let binder = OwnedValueRef {
                graph: key(1, 0),
                value: loop_node.binder,
            };
            assert_eq!(
                facts.members(facts.canonical_value(binder).unwrap())[0],
                binder
            );
        }
    }

    const BRANCH: &str = "fn pick(a: f32, choose: bool, result: tensor[1] f32) -> tensor[1] f32:\n    let mut output = result\n    let mut value = a\n    if choose:\n        value = value + 1.0\n    else:\n        value = value * 2.0\n    output[0] = value\n    return output\n";

    #[test]
    fn branch_captures_share_the_outer_identity() {
        let logical = logical(BRANCH, "pick", &[]);
        let forest = OccurrenceForest::expand(&logical).expect("expansion succeeds");
        let facts = forest.facts();
        assert_total(facts);
        assert_eq!(facts.occurrences().count(), 1);

        let graph = facts.graph(key(0, 0));
        let if_nodes: Vec<_> = graph_nodes(graph)
            .into_iter()
            .filter_map(|(_, node)| match &node.kind {
                LogicalNodeKind::If(if_node) => Some((if_node, node)),
                _ => None,
            })
            .collect();
        assert_eq!(if_nodes.len(), 1);
        let (if_node, node) = if_nodes[0];
        assert!(!if_node.captured.is_empty());
        for capture in &if_node.captured {
            let parameter = OwnedValueRef {
                graph: key(0, 0),
                value: capture.parameter,
            };
            let outer = OwnedValueRef {
                graph: key(0, 0),
                value: capture.outer,
            };
            let canonical = facts.canonical_value(parameter).unwrap();
            assert_eq!(facts.canonical_value(outer).unwrap(), canonical);
            assert_eq!(facts.members(canonical)[0], outer);
            assert_eq!(facts.value_kind(canonical), facts.value(outer).kind());
        }
        // The joined value the `if` produces is a distinct class.
        for output in &node.outputs {
            let joined = facts
                .canonical_value(OwnedValueRef {
                    graph: key(0, 0),
                    value: output.id(),
                })
                .unwrap();
            assert_eq!(facts.members(joined).len(), 1);
        }

        // Root inputs: two value leaves without state, one tensor leaf with
        // its entry state.
        let boundary = &facts.alternative(key(0, 0)).boundary;
        let mut value_leaves = 0;
        let mut tensor_leaves = 0;
        for input in boundary.inputs.values() {
            let InstantiatedInput::Root {
                callee_state,
                ownership,
                ..
            } = input
            else {
                panic!("the entry instantiates root inputs");
            };
            match callee_state {
                None => {
                    assert_eq!(*ownership, ParamOwnership::Value);
                    value_leaves += 1;
                }
                Some(_) => {
                    assert_eq!(*ownership, ParamOwnership::Owned);
                    tensor_leaves += 1;
                }
            }
        }
        assert_eq!((value_leaves, tensor_leaves), (2, 1));
    }

    const TWICE: &str = "fn copy[N](x: &tensor[N] f32) -> tensor[N] f32:\n    return to_owned(x)\n\nfn twice[N](x: &tensor[N] f32, y: &tensor[N] f32) -> (tensor[N] f32, tensor[N] f32):\n    let a = copy(x)\n    let b = copy(y)\n    return a, b\n";

    #[test]
    fn repeated_calls_are_distinct_occurrences_with_distinct_locals() {
        let logical = logical(TWICE, "twice", &[("N", 4)]);
        let forest = OccurrenceForest::expand(&logical).expect("expansion succeeds");
        let facts = forest.facts();
        assert_total(facts);
        assert_eq!(facts.occurrences().count(), 3);

        let entry = facts.alternative(key(0, 0));
        assert_eq!(entry.calls.len(), 2);
        let first = entry.calls[0].1;
        let second = entry.calls[1].1;
        assert_ne!(first, second);
        assert_ne!(
            facts.occurrence(first).choice,
            facts.occurrence(second).choice
        );
        assert_eq!(facts.occurrence(first).interface.name, "copy");
        assert_eq!(facts.occurrence(second).interface.name, "copy");

        // Each occurrence's local storage is its own canonical class; the
        // parameter storages join the caller's classes, which differ.
        let local_of = |occurrence: OccurrenceId| {
            let graph_key = OwnedGraphKey {
                occurrence,
                logical_alternative: 0,
            };
            let locals: Vec<_> = facts
                .graph(graph_key)
                .storages()
                .filter(|(_, storage)| storage.owner == LogicalStorageOwner::Local)
                .map(|(storage, _)| {
                    facts
                        .canonical_storage(OwnedStorageRef {
                            graph: graph_key,
                            storage,
                        })
                        .unwrap()
                })
                .collect();
            assert_eq!(locals.len(), 1);
            locals[0]
        };
        let first_local = local_of(first);
        let second_local = local_of(second);
        assert_ne!(first_local, second_local);
        assert_eq!(facts.storage_members(first_local).len(), 1);
        assert_eq!(facts.storage_members(second_local).len(), 1);

        let parameter_class = |occurrence: OccurrenceId| {
            let alternative = facts.alternative(OwnedGraphKey {
                occurrence,
                logical_alternative: 0,
            });
            let input = alternative
                .boundary
                .inputs
                .values()
                .next()
                .expect("copy has one input leaf");
            let InstantiatedInput::Call {
                callee,
                states: Some((callee_state, _)),
                ..
            } = input
            else {
                panic!("the copy input is a call tensor leaf");
            };
            (
                facts.canonical_value(*callee).unwrap(),
                facts
                    .canonical_storage(facts.state_storage(*callee_state))
                    .unwrap(),
            )
        };
        let (first_value, first_storage) = parameter_class(first);
        let (second_value, second_storage) = parameter_class(second);
        assert_ne!(first_value, second_value);
        assert_ne!(first_storage, second_storage);
        assert_eq!(facts.storage_members(first_storage).len(), 2);

        // Each call's result leaf is one class produced by its callee: a
        // view of that occurrence's own local storage, with the caller's
        // computed output as the other member.
        for (occurrence, local) in [(first, first_local), (second, second_local)] {
            let graph_key = OwnedGraphKey {
                occurrence,
                logical_alternative: 0,
            };
            let alternative = facts.alternative(graph_key);
            assert_eq!(alternative.boundary.results.len(), 1);
            let InstantiatedResult::Call { callee, caller } = alternative
                .boundary
                .results
                .values()
                .next()
                .expect("copy returns one leaf")
            else {
                panic!("the copy result is a call leaf");
            };
            let canonical = facts.canonical_value(*callee).unwrap();
            assert_eq!(facts.members(canonical), &[*callee, *caller]);
            let GraphValueKind::Tensor {
                source: TensorSource::View(view),
                ..
            } = facts.value_kind(canonical)
            else {
                panic!("to_owned returns a view");
            };
            let ViewBase::Storage(storage) = facts
                .view(OwnedViewRef {
                    graph: graph_key,
                    view: *view,
                })
                .base
            else {
                panic!("to_owned views local storage");
            };
            let storage = OwnedStorageRef {
                graph: graph_key,
                storage,
            };
            assert_eq!(facts.canonical_storage(storage).unwrap(), local);
        }

        // The entry's tuple result decomposes into two root leaves of the
        // entry graph, keyed by their ordinal paths.
        let boundary = &entry.boundary;
        let leaves: Vec<&BoundaryLeaf> = boundary.results.keys().collect();
        assert_eq!(
            leaves,
            vec![
                &BoundaryLeaf::Result {
                    leaf: ValuePath(vec![0])
                },
                &BoundaryLeaf::Result {
                    leaf: ValuePath(vec![1])
                },
            ]
        );
        for result in boundary.results.values() {
            let InstantiatedResult::Root { callee } = result else {
                panic!("the entry instantiates root results");
            };
            assert_eq!(callee.graph, key(0, 0));
            assert!(matches!(
                facts.value_kind(facts.canonical_value(*callee).unwrap()),
                GraphValueKind::Tensor { .. }
            ));
        }
    }
}
