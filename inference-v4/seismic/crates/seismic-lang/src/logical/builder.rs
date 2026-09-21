//! The private graph builder: validity by construction.
//!
//! All graph, node, value, state, loop and call assembly goes through
//! `GraphBuilder`; raw constructors do not exist outside this module.
//! Constructors create outputs and dependencies immediately, and every locally
//! decidable invariant is checked at that moment:
//!
//! - every value has one exhaustive kind; a view value names a view of this
//!   graph whose shape is the value's type; void creates no value;
//! - every state token names one storage of this graph from allocation;
//! - every input is a value in scope, so its origin dominates the use;
//! - every consumed state is the current token of its storage;
//! - moved storage has no later use;
//! - uninitialized storage is never read;
//! - writes and atomics consume one state and produce the next while merging
//!   initialization coverage structurally;
//! - branch regions share parameter schemas and produce one result per join;
//! - carried slots have identical initial/body-parameter/body-result types;
//! - independent loops admit only `DisjointWrite`/`Atomic` joins and no
//!   carries;
//! - a call instantiates its callee contract with current caller states, and
//!   every exclusively borrowed leaf has exactly one final state.
//!
//! The consuming `seal` is the only constructor of a `TaskGraph`. It admits a
//! graph only when every region is closed, every value and state token of the
//! graph has exactly one origin, and the boundary names root-scope values and
//! states with the structure its ownership modes require.
//!
//! Every `Err` of this module is a construction defect of the normalizer
//! (the `CompilerDefect` class); it never classifies source semantics.

use super::{
    Access, BoundaryLeaf, CallBoundary, CallInput, CallNode, CarriedSlot, ChoiceId, Coverage,
    GraphRegion, GraphValue, GraphValueId, GraphValueKind, IdVec, IfCapture, IfNode,
    Initialization, JoinSlot, LogicalBoundary, LogicalBoundaryInput, LogicalBoundaryResult,
    LogicalNode, LogicalNodeKind, LogicalRange, LogicalStorage, LogicalStorageId,
    LogicalStorageOwner, LogicalView, LogicalViewId, LoopKind, LoopNode, NodeId,
    PrimitiveApplication, PrimitiveOp, ReductionNode, ReductionOrder, RegionInput,
    RegionParameter, RegionResult, SafetyObligation, StateJoin, StateToken, StateTokenId,
    TaskGraph, TensorSource, ViewBase, ViewTransform,
};
use crate::intrinsics::ReduceOp;
use crate::sir::ParamOwnership;
use crate::span::Span;
use crate::types::{DType, TensorType, ValueType};
use std::collections::{BTreeMap, BTreeSet};

/// Global id allocator shared by every graph of one logical program. Ids are
/// unique across the whole program, so retained runtime expressions and
/// diagnostics never conflate values of different graphs. Moved, never
/// copied: exactly one allocator is live at a time (a suspended graph's
/// builder hands it to the child graph it is building).
#[derive(Clone, Debug, Default)]
pub struct Ids {
    next_value: u32,
    next_state: u32,
    next_storage: u32,
    next_view: u32,
}

impl Ids {
    fn value(&mut self) -> GraphValueId {
        let id = GraphValueId(self.next_value);
        self.next_value += 1;
        id
    }

    fn state(&mut self) -> StateTokenId {
        let id = StateTokenId(self.next_state);
        self.next_state += 1;
        id
    }

    fn storage(&mut self) -> LogicalStorageId {
        let id = LogicalStorageId(self.next_storage);
        self.next_storage += 1;
        id
    }

    fn view(&mut self) -> LogicalViewId {
        let id = LogicalViewId(self.next_view);
        self.next_view += 1;
        id
    }
}

/// One primitive application submitted to the builder.
pub struct PrimitiveSpec {
    pub op: PrimitiveOp,
    pub inputs: Vec<GraphValueId>,
    /// Storages read through operand views; resolved to their current tokens.
    pub reads: Vec<LogicalStorageId>,
    /// The state transition of a writing primitive: consumes the current
    /// token of `storage` and produces the next, merging `coverage`
    /// structurally. A pure primitive has none.
    pub write: Option<WriteEffect>,
    pub outputs: Vec<Output>,
    pub safety: Vec<SafetyObligation>,
    pub span: Span,
}

/// A whole or partial write to one storage.
pub struct WriteEffect {
    pub storage: LogicalStorageId,
    /// Axes this write proves covered.
    pub coverage: Coverage,
    /// An atomic read-modify-write (the admitted operation).
    pub atomic: bool,
    /// The write creates the storage's first version from nothing (a fresh
    /// allocation); a nothing-covering initializing write leaves the storage
    /// `Uninitialized` while still producing its initial state token.
    pub initializing: bool,
}

/// One output of a primitive: a produced value (a non-tensor value, or a
/// computed tensor without storage) or a declared view of storage.
pub enum Output {
    Computed(ValueType),
    View(LogicalViewId),
}

/// What `add_primitive` produced.
pub struct PrimitiveOutcome {
    pub outputs: Vec<GraphValueId>,
    pub next_state: Option<StateTokenId>,
}

/// One loop submitted to the builder.
pub struct LoopSpec {
    pub kind: LoopKind,
    pub range: LogicalRange,
    pub binder: GraphValueId,
    pub invariant_values: Vec<GraphValueId>,
    /// Carried slots in loop order (ordered loops only).
    pub carried: Vec<CarriedSlot>,
    /// Cross-visit state joins of an independent loop, one per mutated storage.
    pub joins: Vec<(LogicalStorageId, StateJoin)>,
    pub body: GraphRegion,
    pub span: Span,
}

/// What `add_loop` produced.
pub struct LoopOutcome {
    /// The loop's exit values, one per value carry, in carried order.
    pub exit_values: Vec<GraphValueId>,
}

/// The exhaustive kind of a produced (non-view) value of type `ty`: tensors
/// are computed, `Void` creates no value, tuples are nonempty.
pub fn computed_kind(ty: ValueType) -> Result<GraphValueKind, String> {
    Ok(match ty {
        ValueType::Void => return Err("void creates no graph value".into()),
        ValueType::Scalar(dtype) => GraphValueKind::Scalar(dtype),
        ValueType::Index { bound } => GraphValueKind::Index { bound },
        ValueType::Range { bound } => GraphValueKind::Range { bound },
        ValueType::Tensor(ty) => GraphValueKind::Tensor {
            ty,
            source: TensorSource::Computed,
        },
        ValueType::Tuple(items) => GraphValueKind::Tuple(items),
        ValueType::CapabilityValue(ty) => GraphValueKind::Capability(ty),
    })
}

struct RegionFrame {
    parameters: Vec<RegionParameter>,
    nodes: Vec<LogicalNode>,
    /// Values visible in this region (ancestor scopes included).
    scope: BTreeSet<GraphValueId>,
    /// State tokens visible in this region (ancestor scopes included).
    state_scope: BTreeSet<StateTokenId>,
}

struct GraphCore {
    choice: ChoiceId,
    alternative: u32,
    ids: Ids,
    /// Every value allocated in this graph, with its exhaustive kind.
    values: BTreeMap<GraphValueId, GraphValue>,
    storages: BTreeMap<LogicalStorageId, LogicalStorage>,
    views: BTreeMap<LogicalViewId, LogicalView>,
    /// The storage every allocated state token versions.
    states: BTreeMap<StateTokenId, LogicalStorageId>,
    /// Current state token per storage.
    current: BTreeMap<LogicalStorageId, StateTokenId>,
    /// Storages moved into a call (no later use).
    moved: BTreeSet<LogicalStorageId>,
    frames: Vec<RegionFrame>,
    /// The closed root region with the value/state scope it ended with;
    /// present once the root frame closed.
    root: Option<ClosedRoot>,
}

struct ClosedRoot {
    region: GraphRegion,
    scope: BTreeSet<GraphValueId>,
    state_scope: BTreeSet<StateTokenId>,
}

/// Builder of one task graph (one alternative of one occurrence).
pub struct GraphBuilder {
    core: GraphCore,
}

impl GraphBuilder {
    /// Begin the graph of `alternative` of `choice`, continuing id allocation
    /// from `ids`.
    pub fn new(choice: ChoiceId, alternative: u32, ids: Ids) -> GraphBuilder {
        GraphBuilder {
            core: GraphCore {
                choice,
                alternative,
                ids,
                values: BTreeMap::new(),
                storages: BTreeMap::new(),
                views: BTreeMap::new(),
                states: BTreeMap::new(),
                current: BTreeMap::new(),
                moved: BTreeSet::new(),
                frames: Vec::new(),
                root: None,
            },
        }
    }

    /// Open the root region with the interface parameters.
    pub fn begin_root(&mut self, parameters: Vec<RegionParameter>) -> Result<(), String> {
        if !self.core.frames.is_empty() || self.core.root.is_some() {
            return Err("the root region is already open".into());
        }
        self.open_frame(parameters)
    }

    /// Open a nested region. `State` parameters become the current tokens of
    /// their storages inside the region; `Value` parameters extend the scope.
    pub fn begin_region(&mut self, parameters: Vec<RegionParameter>) -> Result<(), String> {
        if self.core.root.is_some() {
            return Err("the graph's root region is already closed".into());
        }
        if self.core.frames.is_empty() {
            return Err("a nested region must be opened inside the root region".into());
        }
        self.open_frame(parameters)
    }

    fn open_frame(&mut self, parameters: Vec<RegionParameter>) -> Result<(), String> {
        let mut scope = BTreeSet::new();
        let mut state_scope = BTreeSet::new();
        for frame in &self.core.frames {
            scope.extend(frame.scope.iter().copied());
            state_scope.extend(frame.state_scope.iter().copied());
        }
        for parameter in &parameters {
            match parameter {
                RegionParameter::Value { id, ty } => {
                    let value = self.value(*id)?;
                    if value.ty() != *ty {
                        return Err(format!(
                            "region parameter {} is declared as {ty} but the value is {}",
                            id.0,
                            value.ty()
                        ));
                    }
                    scope.insert(*id);
                }
                RegionParameter::State { id, storage } => {
                    let bound = self.storage_of_token(*id)?;
                    if bound != *storage {
                        return Err(format!(
                            "region state parameter {} versions storage {} but is declared on storage {}",
                            id.0, bound.0, storage.0
                        ));
                    }
                    state_scope.insert(*id);
                    self.core.current.insert(*storage, *id);
                }
            }
        }
        self.core.frames.push(RegionFrame {
            parameters,
            nodes: Vec::new(),
            scope,
            state_scope,
        });
        Ok(())
    }

    /// Close the current region, validating that every result is in scope and
    /// fully initialized where it is a view of storage. The root region takes
    /// no results: the graph boundary is stated at `seal`.
    pub fn end_region(&mut self, results: Vec<RegionResult>) -> Result<GraphRegion, String> {
        let Some(frame) = self.core.frames.pop() else {
            return Err("no region is open".into());
        };
        if self.core.frames.is_empty() && !results.is_empty() {
            return Err("the root region has no positional results; state the boundary at seal".into());
        }
        for result in &results {
            match result {
                RegionResult::Value { id, ty } => {
                    if !frame.scope.contains(id) {
                        return Err(format!("region result value {} is not in scope", id.0));
                    }
                    let value = self.value(*id)?;
                    if value.ty() != *ty {
                        return Err(format!(
                            "region result value {} is declared as {ty} but the value is {}",
                            id.0,
                            value.ty()
                        ));
                    }
                    if let Some(TensorSource::View(view)) = value.tensor_source() {
                        // A view result names storage only when its base is
                        // storage; a view of a computed value is a value.
                        if let ViewBase::Storage(storage) = self.view(view).base {
                            self.check_result_storage(storage)?;
                        }
                    }
                }
                RegionResult::State { id, storage, .. } => {
                    if !frame.state_scope.contains(id) {
                        return Err(format!("region result state {} is not in scope", id.0));
                    }
                    if self.storage_of_token(*id)? != *storage {
                        return Err(format!(
                            "region result state {} does not version storage {}",
                            id.0, storage.0
                        ));
                    }
                }
            }
        }
        let region = GraphRegion {
            parameters: frame.parameters,
            nodes: IdVec::<NodeId, _>::new(frame.nodes),
            results,
        };
        if self.core.frames.is_empty() {
            self.core.root = Some(ClosedRoot {
                region: region.clone(),
                scope: frame.scope,
                state_scope: frame.state_scope,
            });
        }
        Ok(region)
    }

    fn check_result_storage(&self, storage: LogicalStorageId) -> Result<(), String> {
        if self.core.moved.contains(&storage) {
            return Err(format!(
                "a result leaf uses storage {} after it was moved",
                storage.0
            ));
        }
        let initialization = &self.storage(storage).initialization;
        if !matches!(initialization, Initialization::FullyInitialized) {
            return Err(format!(
                "a result leaf on storage {} is not fully initialized",
                storage.0
            ));
        }
        Ok(())
    }

    /// Declare one semantic storage.
    pub fn declare_storage(
        &mut self,
        shape: TensorType,
        owner: LogicalStorageOwner,
        initialization: Initialization,
    ) -> LogicalStorageId {
        let id = self.core.ids.storage();
        self.core.storages.insert(
            id,
            LogicalStorage {
                shape,
                owner,
                initialization,
            },
        );
        id
    }

    /// Declare one semantic view over one base: a storage of this graph, or a
    /// computed tensor value of this graph (a view of a computed value is
    /// read-only and flattens view chains to the bottom computed value).
    pub fn declare_view(
        &mut self,
        base: ViewBase,
        shape: TensorType,
        access: Access,
        transform: ViewTransform,
    ) -> Result<LogicalViewId, String> {
        match base {
            ViewBase::Storage(storage) => {
                if !self.core.storages.contains_key(&storage) {
                    return Err(format!(
                        "a view names storage {} which this graph does not declare",
                        storage.0
                    ));
                }
            }
            ViewBase::Value(value) => {
                if access != Access::Shared {
                    return Err("a view of a computed value is read-only".into());
                }
                match self.value(value)?.kind() {
                    GraphValueKind::Tensor {
                        source: TensorSource::Computed,
                        ..
                    } => {}
                    GraphValueKind::Tensor {
                        source: TensorSource::View(_),
                        ..
                    } => {
                        return Err(format!(
                            "a view's value base {} is itself a view; views flatten to the bottom computed value",
                            value.0
                        ))
                    }
                    kind => {
                        return Err(format!(
                            "a view's value base {} is not a computed tensor ({:?})",
                            value.0, kind
                        ))
                    }
                }
            }
        }
        let id = self.core.ids.view();
        self.core.views.insert(
            id,
            LogicalView {
                base,
                shape,
                access,
                transform,
            },
        );
        Ok(id)
    }

    /// The storage a view of storage names; a named error for a view whose
    /// base is a computed value.
    pub fn storage_base(&self, view: LogicalViewId) -> Result<LogicalStorageId, String> {
        match self.view(view).base {
            ViewBase::Storage(storage) => Ok(storage),
            ViewBase::Value(_) => Err(format!(
                "view {} reads a computed value, not storage",
                view.0
            )),
        }
    }

    /// The kind of a value backed by `view`: a tensor of the view's shape.
    pub fn view_kind(&self, view: LogicalViewId) -> Result<GraphValueKind, String> {
        let declared = self
            .core
            .views
            .get(&view)
            .ok_or_else(|| format!("view {} is not declared by this graph", view.0))?;
        Ok(GraphValueKind::Tensor {
            ty: declared.shape.clone(),
            source: TensorSource::View(view),
        })
    }

    /// Allocate a graph value of one exhaustive kind. A view-backed tensor
    /// must name a view of this graph with exactly the value's shape. The
    /// value enters the open region's scope (or becomes a region parameter
    /// of the region opened next).
    pub fn fresh_value(&mut self, kind: GraphValueKind) -> Result<GraphValueId, String> {
        match &kind {
            GraphValueKind::Tensor {
                ty,
                source: TensorSource::View(view),
            } => {
                let declared = self
                    .core
                    .views
                    .get(view)
                    .ok_or_else(|| format!("view {} is not declared by this graph", view.0))?;
                if declared.shape != *ty {
                    return Err(format!(
                        "view {} has shape {} but backs a value of type {}",
                        view.0,
                        ValueType::Tensor(declared.shape.clone()),
                        ValueType::Tensor(ty.clone())
                    ));
                }
            }
            GraphValueKind::Void => return Err("void creates no graph value".into()),
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
        let id = self.core.ids.value();
        self.core.values.insert(id, GraphValue::new(id, kind));
        if let Some(frame) = self.core.frames.last_mut() {
            frame.scope.insert(id);
        }
        Ok(id)
    }

    /// Allocate a state token versioning `storage`. The token enters scope
    /// when the region parameter or node that originates it is added.
    pub fn fresh_state(&mut self, storage: LogicalStorageId) -> Result<StateTokenId, String> {
        if !self.core.storages.contains_key(&storage) {
            return Err(format!(
                "a state token names storage {} which this graph does not declare",
                storage.0
            ));
        }
        let id = self.core.ids.state();
        self.core.states.insert(id, storage);
        Ok(id)
    }

    /// The current state token of one storage.
    pub fn current_state(&self, storage: LogicalStorageId) -> Result<StateTokenId, String> {
        self.core
            .current
            .get(&storage)
            .copied()
            .ok_or_else(|| format!("storage {} has no current state", storage.0))
    }

    /// Install a specific token as the current state of a storage (used after
    /// region parameters, joins, calls and loop exits).
    pub fn set_current_state(
        &mut self,
        storage: LogicalStorageId,
        token: StateTokenId,
    ) -> Result<(), String> {
        if self.storage_of_token(token)? != storage {
            return Err(format!(
                "token {} does not version storage {}",
                token.0, storage.0
            ));
        }
        self.core.current.insert(storage, token);
        Ok(())
    }

    pub fn initialization(&self, storage: LogicalStorageId) -> &Initialization {
        &self.storage(storage).initialization
    }

    pub fn view(&self, id: LogicalViewId) -> &LogicalView {
        &self.core.views[&id]
    }

    pub fn storage(&self, id: LogicalStorageId) -> &LogicalStorage {
        &self.core.storages[&id]
    }

    /// The value with this id; an error names a value this graph did not
    /// allocate (a normalizer defect).
    pub fn value(&self, id: GraphValueId) -> Result<&GraphValue, String> {
        self.core
            .values
            .get(&id)
            .ok_or_else(|| format!("value {} is not defined by this graph", id.0))
    }

    pub fn value_type(&self, id: GraphValueId) -> Result<ValueType, String> {
        Ok(self.value(id)?.ty())
    }

    /// The tensor source of a tensor value; an error names a non-tensor.
    pub fn tensor_source(&self, id: GraphValueId) -> Result<TensorSource, String> {
        let value = self.value(id)?;
        value
            .tensor_source()
            .ok_or_else(|| format!("value {} is not a tensor ({})", id.0, value.ty()))
    }

    /// The storage a value reads through its view: `Some` exactly for a view
    /// of logical storage. A computed tensor, a view of one, and a non-tensor
    /// value read no storage (an exhaustive statement about the kind, not a
    /// missing fact).
    pub fn read_storage(&self, id: GraphValueId) -> Result<Option<LogicalStorageId>, String> {
        Ok(match self.value(id)?.kind() {
            GraphValueKind::Tensor {
                source: TensorSource::View(view),
                ..
            } => match self.view(*view).base {
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
        })
    }

    /// The distinct storages the given values read through views, in first
    /// occurrence order.
    pub fn reads_of(&self, values: &[GraphValueId]) -> Result<Vec<LogicalStorageId>, String> {
        let mut reads = Vec::new();
        for value in values {
            if let Some(storage) = self.read_storage(*value)? {
                if !reads.contains(&storage) {
                    reads.push(storage);
                }
            }
        }
        Ok(reads)
    }

    /// Mark one storage fully initialized. Used for storages handed to a
    /// call through an exclusive borrow: the checker admits an unassigned
    /// argument only when every applicable implementation initializes the
    /// whole parameter, and an assigned argument stays initialized.
    pub fn mark_fully_initialized(&mut self, storage: LogicalStorageId) {
        self.core
            .storages
            .get_mut(&storage)
            .expect("the storage exists")
            .initialization = Initialization::FullyInitialized;
    }

    /// Add one primitive node.
    pub fn add_primitive(&mut self, spec: PrimitiveSpec) -> Result<PrimitiveOutcome, String> {
        let PrimitiveSpec {
            op,
            inputs,
            reads,
            write,
            outputs,
            safety,
            span,
        } = spec;
        self.check_inputs(&inputs)?;
        let write_storage = write.as_ref().map(|w| w.storage);
        let mut state_inputs = Vec::new();
        let initializing = write.as_ref().map(|w| w.initializing).unwrap_or(false);
        for storage in reads.iter().copied().chain(write_storage) {
            if !self.core.storages.contains_key(&storage) {
                return Err(format!(
                    "a primitive names storage {} which this graph does not declare",
                    storage.0
                ));
            }
            // A write to fresh storage consumes no prior state: the node is
            // the origin of the storage's first version.
            if let Ok(token) = self.current_state(storage) {
                if !state_inputs.contains(&token) {
                    state_inputs.push(token);
                }
                let plain_write = write_storage == Some(storage)
                    && write.as_ref().map(|w| !w.atomic).unwrap_or(false);
                if !plain_write {
                    self.check_read(storage)?;
                }
            } else if write_storage != Some(storage) || !initializing {
                return Err(format!("storage {} has no current state", storage.0));
            }
            if self.core.moved.contains(&storage) {
                return Err(format!("storage {} is used after it was moved", storage.0));
            }
        }
        let mut out_ids = Vec::new();
        for output in outputs {
            let kind = match output {
                Output::Computed(ty) => computed_kind(ty)?,
                Output::View(view) => self.view_kind(view)?,
            };
            out_ids.push(self.fresh_value(kind)?);
        }
        let mut state_outputs = Vec::new();
        if let Some(write) = &write {
            let token = self.fresh_state(write.storage)?;
            state_outputs.push(StateToken {
                id: token,
                storage: write.storage,
                join: None,
            });
            self.apply_coverage(write.storage, &write.coverage, write.initializing);
            self.core.current.insert(write.storage, token);
        }
        let outputs = self.values_of(&out_ids)?;
        let next_state = state_outputs.first().map(|t| t.id);
        let node = LogicalNode {
            inputs,
            state_inputs,
            kind: LogicalNodeKind::Primitive(PrimitiveApplication { op }),
            outputs,
            state_outputs,
            safety,
            span,
        };
        self.append_node(node);
        Ok(PrimitiveOutcome {
            outputs: out_ids,
            next_state,
        })
    }

    /// Add one reduction node (never a scalar primitive payload). A computed
    /// operand is reduced directly; a view operand reads its storage.
    #[allow(clippy::too_many_arguments)]
    pub fn add_reduction(
        &mut self,
        operand: GraphValueId,
        axis: usize,
        op: ReduceOp,
        order: ReductionOrder,
        accumulator: DType,
        result: ValueType,
        span: Span,
    ) -> Result<GraphValueId, String> {
        self.check_inputs(&[operand])?;
        let value = self.value(operand)?;
        let (tensor, source) = match value.kind() {
            GraphValueKind::Tensor { ty, source } => (ty.clone(), *source),
            GraphValueKind::Void
            | GraphValueKind::Scalar(_)
            | GraphValueKind::Index { .. }
            | GraphValueKind::Range { .. }
            | GraphValueKind::Tuple(_)
            | GraphValueKind::Capability(_) => {
                return Err(format!(
                    "reduction operand must be a tensor, found {}",
                    value.ty()
                ))
            }
        };
        let reduced_extent = tensor.axes.get(axis).cloned().ok_or_else(|| {
            format!(
                "reduction axis {axis} is outside rank {}",
                tensor.axes.len()
            )
        })?;
        let mut state_inputs = Vec::new();
        match source {
            TensorSource::View(view) => match self.view(view).base {
                ViewBase::Storage(storage) => {
                    self.check_read(storage)?;
                    state_inputs.push(self.current_state(storage)?);
                }
                // A view of a computed value reads the value; no storage.
                ViewBase::Value(_) => {}
            },
            TensorSource::Computed => {}
        }
        let out = self.fresh_value(computed_kind(result.clone())?)?;
        let node = LogicalNode {
            inputs: vec![operand],
            state_inputs,
            kind: LogicalNodeKind::Reduction(ReductionNode {
                operand,
                axis,
                op,
                order,
                accumulator,
                result,
            }),
            outputs: self.values_of(&[out])?,
            state_outputs: Vec::new(),
            safety: if matches!(op, ReduceOp::Max | ReduceOp::Min | ReduceOp::Argmax) {
                vec![SafetyObligation::ExtentPositive {
                    extent: reduced_extent,
                }]
            } else {
                Vec::new()
            },
            span,
        };
        self.append_node(node);
        Ok(out)
    }

    /// Add one call node: the callee contract instantiated with the caller's
    /// identities. Input values must be in scope; a storage-backed tensor
    /// input's state must be the current token of its storage (a moved
    /// storage has no later use); a computed tensor input carries the value
    /// itself, admitted for shared reads and owned moves; every exclusively
    /// borrowed leaf has exactly one final state versioning the same storage,
    /// which becomes current; result values are produced by the node.
    pub fn add_call(
        &mut self,
        choice: ChoiceId,
        boundary: CallBoundary,
        span: Span,
    ) -> Result<(), String> {
        let mut inputs = Vec::new();
        let mut state_inputs = Vec::new();
        for input in boundary.inputs.values() {
            match input {
                CallInput::Value(id) => inputs.push(*id),
                CallInput::Computed { value, .. } => inputs.push(*value),
                CallInput::Tensor { value, state, .. } => {
                    inputs.push(*value);
                    state_inputs.push(*state);
                }
            }
        }
        self.check_inputs(&inputs)?;
        for (leaf, input) in &boundary.inputs {
            let BoundaryLeaf::Input { .. } = leaf else {
                return Err("a call input leaf is keyed as a result".into());
            };
            match input {
                CallInput::Value(id) => {
                    if self.value(*id)?.tensor_source().is_some() {
                        return Err(format!(
                            "call input value {} is a tensor passed as a plain value",
                            id.0
                        ));
                    }
                }
                CallInput::Computed { value, ownership } => match ownership {
                    // An exclusive borrow mutates through caller storage; a
                    // computed value has none to version.
                    ParamOwnership::Exclusive => {
                        return Err(format!(
                            "exclusively borrowed call input {} is a computed value; an exclusive borrow requires a tensor place with storage",
                            value.0
                        ))
                    }
                    ParamOwnership::Value => {
                        return Err(format!(
                            "call tensor input {} carries value ownership",
                            value.0
                        ))
                    }
                    ParamOwnership::Shared | ParamOwnership::Owned => {
                        if boundary.final_states.contains_key(leaf) {
                            return Err("a computed call input has no final state".into());
                        }
                        match self.value(*value)?.tensor_source() {
                            Some(TensorSource::Computed) => {}
                            Some(TensorSource::View(view)) => match self.view(view).base {
                                // A storage-backed tensor goes through its
                                // state so the call can borrow or move the
                                // storage.
                                ViewBase::Storage(_) => {
                                    return Err(format!(
                                        "call tensor input {} views storage but is passed as a computed value",
                                        value.0
                                    ))
                                }
                                ViewBase::Value(_) => {}
                            },
                            None => {
                                return Err(format!(
                                    "call tensor input {} is not a tensor",
                                    value.0
                                ))
                            }
                        }
                    }
                },
                CallInput::Tensor {
                    value,
                    state,
                    ownership,
                } => {
                    let storage = self.storage_of_current(*state)?;
                    let Some(TensorSource::View(view)) = self.value(*value)?.tensor_source()
                    else {
                        return Err(format!(
                            "call tensor input {} is not a view of storage",
                            value.0
                        ));
                    };
                    if self.storage_base(view)? != storage {
                        return Err(format!(
                            "call tensor input {} views storage {} but consumes a state of storage {}",
                            value.0,
                            self.storage_base(view)?.0,
                            storage.0
                        ));
                    }
                    if self.core.moved.contains(&storage) {
                        return Err(format!(
                            "storage {} is used by this call after it was moved",
                            storage.0
                        ));
                    }
                    match ownership {
                        ParamOwnership::Value => {
                            return Err(format!(
                                "call tensor input {} carries value ownership",
                                value.0
                            ));
                        }
                        ParamOwnership::Shared => {
                            if boundary.final_states.contains_key(leaf) {
                                return Err("a shared borrow has no final state".into());
                            }
                        }
                        ParamOwnership::Owned => {
                            if boundary.final_states.contains_key(leaf) {
                                return Err("a moved tensor has no final state".into());
                            }
                        }
                        ParamOwnership::Exclusive => {
                            let final_state = boundary.final_states.get(leaf).ok_or_else(|| {
                                format!(
                                    "exclusively borrowed call input {} has no final state",
                                    value.0
                                )
                            })?;
                            if self.storage_of_token(*final_state)? != storage {
                                return Err(format!(
                                    "final state {} does not version storage {}",
                                    final_state.0, storage.0
                                ));
                            }
                        }
                    }
                }
            }
        }
        for leaf in boundary.final_states.keys() {
            match boundary.inputs.get(leaf) {
                Some(CallInput::Tensor {
                    ownership: ParamOwnership::Exclusive,
                    ..
                }) => {}
                Some(_) | None => {
                    return Err("a final state names a leaf that is not an exclusive tensor input".into());
                }
            }
        }
        for (leaf, value) in &boundary.results {
            let BoundaryLeaf::Result { .. } = leaf else {
                return Err("a call result leaf is keyed as an input".into());
            };
            match self.value(*value)?.tensor_source() {
                Some(TensorSource::View(_)) => {
                    return Err(format!(
                        "call result {} is a view; a call result is a produced value",
                        value.0
                    ));
                }
                Some(TensorSource::Computed) | None => {}
            }
        }
        // Moves happen after every check so a failing call leaves no trace.
        for input in boundary.inputs.values() {
            if let CallInput::Tensor {
                state,
                ownership: ParamOwnership::Owned,
                ..
            } = input
            {
                let storage = self.storage_of_token(*state)?;
                self.core.moved.insert(storage);
            }
        }
        let result_ids: Vec<GraphValueId> = boundary.results.values().copied().collect();
        let outputs = self.values_of(&result_ids)?;
        let mut state_outputs = Vec::new();
        for token in boundary.final_states.values() {
            let storage = self.storage_of_token(*token)?;
            state_outputs.push(StateToken {
                id: *token,
                storage,
                join: None,
            });
        }
        let node = LogicalNode {
            inputs,
            state_inputs,
            kind: LogicalNodeKind::Call(CallNode { choice, boundary }),
            outputs,
            state_outputs: state_outputs.clone(),
            safety: Vec::new(),
            span,
        };
        self.append_node(node);
        for token in &state_outputs {
            self.core.current.insert(token.storage, token.id);
        }
        Ok(())
    }

    /// Add one `if` node with explicit joins. Branch regions must share
    /// parameter schemas; every join names one result of each branch.
    pub fn add_if(
        &mut self,
        condition: GraphValueId,
        then_region: GraphRegion,
        else_region: GraphRegion,
        joins: Vec<JoinSlot>,
        captured: Vec<IfCapture>,
        span: Span,
    ) -> Result<(), String> {
        self.check_inputs(&[condition])?;
        if then_region.parameters != else_region.parameters {
            return Err("if branches must share parameter schemas".into());
        }
        // Every captured parameter names its outer value; both branch
        // schemas share them.
        let parameter_ids: BTreeSet<_> = then_region
            .parameters
            .iter()
            .filter_map(|parameter| match parameter {
                RegionParameter::Value { id, .. } => Some(*id),
                RegionParameter::State { .. } => None,
            })
            .collect();
        for capture in &captured {
            if !parameter_ids.contains(&capture.parameter) {
                return Err("an if capture names a parameter its branches lack".into());
            }
            self.check_inputs(&[capture.outer])?;
        }
        let then_len = then_region.results.len();
        let else_len = else_region.results.len();
        for join in &joins {
            let (then_result, else_result) = match join {
                JoinSlot::Value {
                    then_result,
                    else_result,
                    ..
                } => (then_result, else_result),
                JoinSlot::State {
                    then_result,
                    else_result,
                    ..
                } => (then_result, else_result),
            };
            if then_result.index() >= then_len || else_result.index() >= else_len {
                return Err("a join references a result its branch does not have".into());
            }
        }
        let mut outputs = Vec::new();
        let mut state_outputs = Vec::new();
        for join in &joins {
            match join {
                JoinSlot::Value { joined, ty, .. } => {
                    let value = self.value(*joined)?;
                    if value.ty() != *ty {
                        return Err(format!(
                            "join value {} is declared as {ty} but the value is {}",
                            joined.0,
                            value.ty()
                        ));
                    }
                    outputs.push(value.clone());
                }
                JoinSlot::State {
                    joined, storage, ..
                } => {
                    if self.storage_of_token(*joined)? != *storage {
                        return Err(format!(
                            "join state {} does not version storage {}",
                            joined.0, storage.0
                        ));
                    }
                    state_outputs.push(StateToken {
                        id: *joined,
                        storage: *storage,
                        join: None,
                    });
                }
            }
        }
        let node = LogicalNode {
            inputs: vec![condition],
            state_inputs: Vec::new(),
            kind: LogicalNodeKind::If(IfNode {
                condition,
                then_region,
                else_region,
                joins: joins.clone(),
                captured,
            }),
            outputs,
            state_outputs,
            safety: Vec::new(),
            span,
        };
        self.append_node(node);
        for join in &joins {
            if let JoinSlot::State {
                joined, storage, ..
            } = join
            {
                self.core.current.insert(*storage, *joined);
            }
        }
        Ok(())
    }

    /// Add one loop node. Carried slots must have identical types across
    /// initial value, body parameter and body result; independent loops admit
    /// only `DisjointWrite`/`Atomic` joins and no carries. Returns the loop's
    /// exit values (one fresh produced value per value carry, in carried
    /// order).
    pub fn add_loop(&mut self, spec: LoopSpec) -> Result<LoopOutcome, String> {
        let LoopSpec {
            kind,
            range,
            binder,
            invariant_values,
            carried,
            joins,
            body,
            span,
        } = spec;
        self.check_inputs(&[range.start, range.end])?;
        self.check_inputs(&invariant_values)?;
        for slot in &carried {
            match slot.initial {
                RegionInput::Value(initial) => {
                    let Some(RegionParameter::Value { ty, .. }) =
                        body.parameters.get(slot.body_parameter.index())
                    else {
                        return Err(format!(
                            "carried value slot {} has no value body parameter",
                            slot.body_parameter.0
                        ));
                    };
                    let Some(RegionResult::Value { ty: result_ty, .. }) =
                        body.results.get(slot.body_result.index())
                    else {
                        return Err(format!(
                            "carried value slot {} has no value body result",
                            slot.body_result.0
                        ));
                    };
                    let initial_ty = self.value_type(initial)?;
                    if initial_ty != *ty {
                        return Err(format!(
                            "carried slot {} has initial type {initial_ty} but body parameter type {ty}",
                            slot.body_parameter.0
                        ));
                    }
                    if initial_ty != *result_ty {
                        return Err(format!(
                            "carried slot {} has initial type {initial_ty} but body result type {result_ty}",
                            slot.body_result.0
                        ));
                    }
                }
                RegionInput::State(token) => {
                    let initial_storage = self.storage_of_current(token)?;
                    let Some(RegionParameter::State { storage, .. }) =
                        body.parameters.get(slot.body_parameter.index())
                    else {
                        return Err(format!(
                            "carried state slot {} has no state body parameter",
                            slot.body_parameter.0
                        ));
                    };
                    if initial_storage != *storage {
                        return Err(format!(
                            "carried state slot {} changes storage",
                            slot.body_parameter.0
                        ));
                    }
                    let Some(RegionResult::State {
                        storage: result_storage,
                        ..
                    }) = body.results.get(slot.body_result.index())
                    else {
                        return Err(format!(
                            "carried state slot {} has no state body result",
                            slot.body_result.0
                        ));
                    };
                    if initial_storage != *result_storage {
                        return Err(format!(
                            "carried state slot {} changes storage",
                            slot.body_result.0
                        ));
                    }
                }
            }
        }
        match kind {
            LoopKind::Independent => {
                if !carried.is_empty() {
                    return Err("an independent loop has no data carries".into());
                }
            }
            LoopKind::Ordered => {
                if !joins.is_empty() {
                    return Err("an ordered loop joins carried states, not visit states".into());
                }
            }
        }
        let mut initial_values = Vec::new();
        let mut initial_states = Vec::new();
        for slot in &carried {
            match slot.initial {
                RegionInput::Value(id) => initial_values.push(id),
                RegionInput::State(token) => initial_states.push(token),
            }
        }
        for (storage, _) in &joins {
            initial_states.push(self.current_state(*storage)?);
        }
        let mut state_outputs = Vec::new();
        match kind {
            LoopKind::Ordered => {
                for slot in &carried {
                    if let RegionInput::State(token) = slot.initial {
                        let storage = self.storage_of_current(token)?;
                        let next = self.fresh_state(storage)?;
                        state_outputs.push(StateToken {
                            id: next,
                            storage,
                            join: None,
                        });
                    }
                }
            }
            LoopKind::Independent => {
                for (storage, join) in &joins {
                    let next = self.fresh_state(*storage)?;
                    state_outputs.push(StateToken {
                        id: next,
                        storage: *storage,
                        join: Some(join.clone()),
                    });
                }
            }
        }
        // The loop's exit values are fresh produced values (the value of the
        // last visit), one per value carry, in carried order.
        let mut exit_values = Vec::new();
        for id in &initial_values {
            let ty = self.value_type(*id)?;
            exit_values.push(self.fresh_value(computed_kind(ty)?)?);
        }
        let node = LogicalNode {
            inputs: Vec::new(),
            state_inputs: Vec::new(),
            kind: LogicalNodeKind::Loop(LoopNode {
                kind,
                range,
                binder,
                invariant_values,
                initial_values,
                initial_states,
                body,
                carried,
            }),
            outputs: self.values_of(&exit_values)?,
            state_outputs: state_outputs.clone(),
            safety: Vec::new(),
            span,
        };
        self.append_node(node);
        for token in &state_outputs {
            self.core.current.insert(token.storage, token.id);
        }
        Ok(LoopOutcome { exit_values })
    }

    /// Take the id allocator out (a child graph is built with it while this
    /// graph's construction is suspended).
    pub fn take_ids(&mut self) -> Ids {
        std::mem::take(&mut self.core.ids)
    }

    /// Restore the id allocator after a child graph finished.
    pub fn restore_ids(&mut self, ids: Ids) {
        self.core.ids = ids;
    }

    /// Abort construction, recovering the id allocator.
    pub fn abort(self) -> Ids {
        self.core.ids
    }

    /// Close construction: the only constructor of a `TaskGraph`. Requires
    /// every region closed, and establishes by checking:
    ///
    /// - every value and state token of the graph has exactly one origin
    ///   (region parameter or node output) and every origin is in the table;
    /// - every view names a storage of the graph and every view value has
    ///   the view's shape (by construction of `fresh_value`/`declare_view`);
    /// - every boundary input is a root region parameter: a tensor input is a
    ///   view of parameter-owned storage whose entry state is the root state
    ///   parameter of that storage; a value input is a non-tensor;
    /// - every exclusive input has exactly one final state, in root scope and
    ///   versioning its storage; shared, owned and value inputs have none;
    /// - every result value is in root scope; a view result is fully
    ///   initialized and unmoved; a computed result needs no storage.
    ///
    /// Returns the sealed graph and the id allocator for the next graph.
    pub fn seal(
        self,
        inputs: BTreeMap<BoundaryLeaf, LogicalBoundaryInput>,
        results: BTreeMap<BoundaryLeaf, LogicalBoundaryResult>,
        final_states: BTreeMap<BoundaryLeaf, StateTokenId>,
    ) -> Result<(TaskGraph, Ids), String> {
        if !self.core.frames.is_empty() {
            return Err("a region is still open".into());
        }
        let Some(root) = &self.core.root else {
            return Err("the root region was never closed".into());
        };

        // Origins: exactly one per value and per state token.
        let mut value_origins: BTreeMap<GraphValueId, usize> = BTreeMap::new();
        let mut state_origins: BTreeMap<StateTokenId, (usize, LogicalStorageId)> = BTreeMap::new();
        collect_origins(&root.region, &mut value_origins, &mut state_origins);
        for (id, count) in &value_origins {
            if *count != 1 {
                return Err(format!("value {} has {count} origins", id.0));
            }
            if !self.core.values.contains_key(id) {
                return Err(format!("value {} originates but is not in the table", id.0));
            }
        }
        for id in self.core.values.keys() {
            if !value_origins.contains_key(id) {
                return Err(format!("value {} is allocated but never originates", id.0));
            }
        }
        for (id, (count, storage)) in &state_origins {
            if *count != 1 {
                return Err(format!("state token {} has {count} origins", id.0));
            }
            match self.core.states.get(id) {
                Some(bound) if bound == storage => {}
                Some(bound) => {
                    return Err(format!(
                        "state token {} originates on storage {} but versions storage {}",
                        id.0, storage.0, bound.0
                    ))
                }
                None => {
                    return Err(format!(
                        "state token {} originates but is not in the table",
                        id.0
                    ))
                }
            }
        }
        for id in self.core.states.keys() {
            if !state_origins.contains_key(id) {
                return Err(format!(
                    "state token {} is allocated but never originates",
                    id.0
                ));
            }
        }

        // Boundary inputs.
        let root_value_params: BTreeSet<GraphValueId> = root
            .region
            .parameters
            .iter()
            .filter_map(|parameter| match parameter {
                RegionParameter::Value { id, .. } => Some(*id),
                RegionParameter::State { .. } => None,
            })
            .collect();
        let root_state_params: BTreeMap<StateTokenId, LogicalStorageId> = root
            .region
            .parameters
            .iter()
            .filter_map(|parameter| match parameter {
                RegionParameter::State { id, storage } => Some((*id, *storage)),
                RegionParameter::Value { .. } => None,
            })
            .collect();
        for (leaf, input) in &inputs {
            let BoundaryLeaf::Input { .. } = leaf else {
                return Err("a boundary input leaf is keyed as a result".into());
            };
            match input {
                LogicalBoundaryInput::Value(id) => {
                    if !root_value_params.contains(id) {
                        return Err(format!(
                            "boundary input value {} is not a root parameter",
                            id.0
                        ));
                    }
                    if self.value(*id)?.tensor_source().is_some() {
                        return Err(format!(
                            "boundary input value {} is a tensor without a state",
                            id.0
                        ));
                    }
                    if final_states.contains_key(leaf) {
                        return Err("a value input has no final state".into());
                    }
                }
                LogicalBoundaryInput::Tensor {
                    value,
                    state,
                    ownership,
                } => {
                    if !root_value_params.contains(value) {
                        return Err(format!(
                            "boundary tensor input {} is not a root parameter",
                            value.0
                        ));
                    }
                    let Some(TensorSource::View(view)) = self.value(*value)?.tensor_source()
                    else {
                        return Err(format!(
                            "boundary tensor input {} is not a view of parameter storage",
                            value.0
                        ));
                    };
                    let storage = match self.view(view).base {
                        ViewBase::Storage(storage) => storage,
                        ViewBase::Value(_) => {
                            return Err(format!(
                                "boundary tensor input {} is not a view of parameter storage",
                                value.0
                            ))
                        }
                    };
                    if self.storage(storage).owner != LogicalStorageOwner::Parameter(leaf.clone())
                    {
                        return Err(format!(
                            "boundary tensor input {} views storage {} which is not owned by its leaf",
                            value.0, storage.0
                        ));
                    }
                    match root_state_params.get(state) {
                        Some(bound) if *bound == storage => {}
                        Some(_) | None => {
                            return Err(format!(
                                "boundary tensor input {} names entry state {} which is not the root state of storage {}",
                                value.0, state.0, storage.0
                            ))
                        }
                    }
                    match ownership {
                        ParamOwnership::Value => {
                            return Err(format!(
                                "boundary tensor input {} carries value ownership",
                                value.0
                            ))
                        }
                        ParamOwnership::Shared | ParamOwnership::Owned => {
                            if final_states.contains_key(leaf) {
                                return Err(format!(
                                    "boundary input {} is not exclusive but has a final state",
                                    value.0
                                ));
                            }
                        }
                        ParamOwnership::Exclusive => {
                            let token = final_states.get(leaf).ok_or_else(|| {
                                format!(
                                    "exclusive boundary input {} has no final state",
                                    value.0
                                )
                            })?;
                            if !root.state_scope.contains(token) {
                                return Err(format!(
                                    "final state {} is not in root scope",
                                    token.0
                                ));
                            }
                            if self.storage_of_token(*token)? != storage {
                                return Err(format!(
                                    "final state {} does not version storage {}",
                                    token.0, storage.0
                                ));
                            }
                        }
                    }
                }
            }
        }
        for leaf in final_states.keys() {
            if !inputs.contains_key(leaf) {
                return Err("a final state names a leaf that is not a boundary input".into());
            }
        }

        // Boundary results.
        for (leaf, result) in &results {
            let BoundaryLeaf::Result { .. } = leaf else {
                return Err("a boundary result leaf is keyed as an input".into());
            };
            if !root.scope.contains(&result.value) {
                return Err(format!(
                    "boundary result {} is not in root scope",
                    result.value.0
                ));
            }
            if let Some(TensorSource::View(view)) = self.value(result.value)?.tensor_source() {
                // A returned view of a computed value is a legitimate value
                // result; only a view of storage names storage to check.
                if let ViewBase::Storage(storage) = self.view(view).base {
                    self.check_result_storage(storage)?;
                }
            }
        }

        let core = self.core;
        let root = core.root.expect("checked above");
        let graph = TaskGraph {
            choice: core.choice,
            alternative: core.alternative,
            boundary: LogicalBoundary::new(inputs, results, final_states),
            root: root.region,
            values: IdVec::from_iter(core.values),
            storages: IdVec::from_iter(core.storages),
            views: IdVec::from_iter(core.views),
            states: IdVec::from_iter(core.states),
        };
        Ok((graph, core.ids))
    }

    // -- internals ---------------------------------------------------------

    fn check_inputs(&self, inputs: &[GraphValueId]) -> Result<(), String> {
        for id in inputs {
            let visible = self
                .core
                .frames
                .iter()
                .any(|frame| frame.scope.contains(id));
            if !visible {
                return Err(format!("input value {} does not dominate this use", id.0));
            }
            if let Some(storage) = self.read_storage(*id)? {
                if self.core.moved.contains(&storage) {
                    return Err(format!(
                        "value {} uses storage {} after it was moved",
                        id.0, storage.0
                    ));
                }
            }
        }
        Ok(())
    }

    fn check_read(&self, storage: LogicalStorageId) -> Result<(), String> {
        let initialization = &self.storage(storage).initialization;
        if matches!(initialization, Initialization::Uninitialized) {
            return Err(format!(
                "storage {} is read before any element is written",
                storage.0
            ));
        }
        Ok(())
    }

    /// The storage a token versions (recorded at allocation).
    fn storage_of_token(&self, token: StateTokenId) -> Result<LogicalStorageId, String> {
        self.core
            .states
            .get(&token)
            .copied()
            .ok_or_else(|| format!("token {} is not a state token of this graph", token.0))
    }

    /// The storage of a token that must be its storage's current state.
    fn storage_of_current(&self, token: StateTokenId) -> Result<LogicalStorageId, String> {
        let storage = self.storage_of_token(token)?;
        if self.core.current.get(&storage) != Some(&token) {
            return Err(format!(
                "token {} is not the current state of storage {}",
                token.0, storage.0
            ));
        }
        Ok(storage)
    }

    fn values_of(&self, ids: &[GraphValueId]) -> Result<Vec<GraphValue>, String> {
        ids.iter().map(|id| self.value(*id).cloned()).collect()
    }

    fn apply_coverage(
        &mut self,
        storage: LogicalStorageId,
        coverage: &Coverage,
        initializing: bool,
    ) {
        let initialization = self.storage(storage).initialization.clone();
        self.core
            .storages
            .get_mut(&storage)
            .expect("the storage exists")
            .initialization = match initialization {
            Initialization::Uninitialized => {
                if coverage.is_full() {
                    Initialization::FullyInitialized
                } else if initializing && coverage.axes.iter().all(|axis| !axis) {
                    Initialization::Uninitialized
                } else {
                    Initialization::PartiallyInitialized(coverage.clone())
                }
            }
            Initialization::PartiallyInitialized(existing) => {
                let merged = existing.union(coverage);
                if merged.is_full() {
                    Initialization::FullyInitialized
                } else {
                    Initialization::PartiallyInitialized(merged)
                }
            }
            Initialization::FullyInitialized => Initialization::FullyInitialized,
        };
    }

    fn append_node(&mut self, node: LogicalNode) {
        let frame = self.core.frames.last_mut().expect("a region is open");
        for output in &node.outputs {
            frame.scope.insert(output.id());
        }
        for token in &node.state_outputs {
            frame.state_scope.insert(token.id);
        }
        frame.nodes.push(node);
    }
}

/// Count the origins of every value and state token in a region tree.
fn collect_origins(
    region: &GraphRegion,
    values: &mut BTreeMap<GraphValueId, usize>,
    states: &mut BTreeMap<StateTokenId, (usize, LogicalStorageId)>,
) {
    for parameter in &region.parameters {
        match parameter {
            RegionParameter::Value { id, .. } => *values.entry(*id).or_insert(0) += 1,
            RegionParameter::State { id, storage } => {
                let entry = states.entry(*id).or_insert((0, *storage));
                entry.0 += 1;
            }
        }
    }
    for node in region.nodes.iter() {
        for output in &node.outputs {
            *values.entry(output.id()).or_insert(0) += 1;
        }
        for token in &node.state_outputs {
            let entry = states.entry(token.id).or_insert((0, token.storage));
            entry.0 += 1;
        }
        match &node.kind {
            LogicalNodeKind::If(if_node) => {
                collect_origins(&if_node.then_region, values, states);
                collect_origins(&if_node.else_region, values, states);
            }
            LogicalNodeKind::Loop(loop_node) => collect_origins(&loop_node.body, values, states),
            LogicalNodeKind::Primitive(_)
            | LogicalNodeKind::Reduction(_)
            | LogicalNodeKind::Call(_) => {}
        }
    }
}
