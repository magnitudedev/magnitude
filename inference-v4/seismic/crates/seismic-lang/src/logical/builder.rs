//! The type-state graph builder: validity by construction.
//!
//! All graph, node, value, state, loop and call assembly inside this crate
//! goes through `GraphBuilder`; raw constructors do not exist. Constructors
//! create outputs and dependencies immediately, and every locally decidable
//! invariant is checked at that moment:
//!
//! - every input is a value in scope, so its origin dominates the use;
//! - every consumed state is the current token of its storage;
//! - moved storage has no later use;
//! - uninitialized storage is never read;
//! - writes and atomics consume one state and produce the next while merging
//!   initialization coverage structurally;
//! - branch regions share parameter schemas and produce one result per join;
//! - carried slots have identical initial/body-parameter/body-result types;
//! - independent loops admit only `DisjointWrite`/`Atomic` joins and no
//!   carries.
//!
//! `seal` is available only when every region is closed and the root region
//! produced its results; `finish` is available only on the sealed builder.

use super::{
    Access, BoundaryInput, BoundaryInputKind, BoundaryResult, BoundaryResultKind, CallNode,
    CarriedSlot, ChoiceId, Coverage, GraphRegion, GraphValue, GraphValueId, IdVec, IfCapture,
    IfNode, Initialization, JoinSlot, LogicalNode, LogicalNodeKind, LogicalRange, LogicalStorage,
    LogicalStorageId, LogicalView, LogicalViewId, LoopKind, LoopNode, NodeId, PrimitiveApplication,
    PrimitiveOp, ReductionNode, ReductionOrder, RegionInput, RegionParameter, RegionResult,
    SafetyObligation, StateJoin, StateToken, StateTokenId, StorageOrigin, TaskGraph, ViewTransform,
};
use crate::intrinsics::ReduceOp;
use crate::span::Span;
use crate::types::{DType, TensorType, ValueType};
use std::collections::{BTreeMap, BTreeSet};
use std::marker::PhantomData;

/// Construction phase marker: regions may still be open.
pub struct Building;
/// Construction phase marker: everything is consumed; only `finish` remains.
pub struct Complete;

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
    pub fn value(&mut self) -> GraphValueId {
        let id = GraphValueId(self.next_value);
        self.next_value += 1;
        id
    }

    pub fn state(&mut self) -> StateTokenId {
        let id = StateTokenId(self.next_state);
        self.next_state += 1;
        id
    }

    pub fn storage(&mut self) -> LogicalStorageId {
        let id = LogicalStorageId(self.next_storage);
        self.next_storage += 1;
        id
    }

    pub fn view(&mut self) -> LogicalViewId {
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
    /// An optional state transition: consumes the current token of `storage`
    /// and produces the next, merging `coverage` structurally.
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

/// One output of a primitive: a plain typed value or a declared view.
pub enum Output {
    Value(ValueType),
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

struct RegionFrame {
    parameters: Vec<RegionParameter>,
    nodes: Vec<LogicalNode>,
    /// Values visible in this region (ancestor scopes included).
    scope: BTreeMap<GraphValueId, ValueType>,
    /// State tokens visible in this region (ancestor scopes included).
    state_scope: BTreeMap<StateTokenId, LogicalStorageId>,
}

struct GraphCore {
    choice: ChoiceId,
    alternative: u32,
    ids: Ids,
    parameters: Vec<RegionParameter>,
    storages: BTreeMap<u32, LogicalStorage>,
    views: BTreeMap<u32, LogicalView>,
    /// Type of every value allocated in this graph.
    value_types: BTreeMap<GraphValueId, ValueType>,
    /// The view backing each tensor value.
    value_views: BTreeMap<GraphValueId, LogicalViewId>,
    /// Current state token per storage.
    current: BTreeMap<LogicalStorageId, StateTokenId>,
    /// Storages moved into a call (no later use).
    moved: BTreeMap<LogicalStorageId, ()>,
    frames: Vec<RegionFrame>,
    root: Option<GraphRegion>,
    results: Vec<RegionResult>,
}

/// Builder of one task graph (one alternative of one occurrence).
pub struct GraphBuilder<P = Building> {
    core: GraphCore,
    phase: PhantomData<P>,
}

impl GraphBuilder<Building> {
    /// Begin the graph of `alternative` of `choice`, continuing id allocation
    /// from `ids`.
    pub fn new(choice: ChoiceId, alternative: u32, ids: Ids) -> GraphBuilder<Building> {
        GraphBuilder {
            core: GraphCore {
                choice,
                alternative,
                ids,
                parameters: Vec::new(),
                storages: BTreeMap::new(),
                views: BTreeMap::new(),
                value_types: BTreeMap::new(),
                value_views: BTreeMap::new(),
                current: BTreeMap::new(),
                moved: BTreeMap::new(),
                frames: Vec::new(),
                root: None,
                results: Vec::new(),
            },
            phase: PhantomData,
        }
    }

    /// Open the root region with the interface parameters.
    pub fn begin_root(&mut self, parameters: Vec<RegionParameter>) -> Result<(), String> {
        if !self.core.frames.is_empty() || self.core.root.is_some() {
            return Err("the root region is already open".into());
        }
        self.core.parameters = parameters.clone();
        self.open_frame(parameters);
        Ok(())
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
        self.open_frame(parameters);
        Ok(())
    }

    fn open_frame(&mut self, parameters: Vec<RegionParameter>) {
        let mut scope = BTreeMap::new();
        let mut state_scope = BTreeMap::new();
        for frame in &self.core.frames {
            for (id, ty) in &frame.scope {
                scope.insert(*id, ty.clone());
            }
            for (token, storage) in &frame.state_scope {
                state_scope.insert(*token, *storage);
            }
        }
        for parameter in &parameters {
            match parameter {
                RegionParameter::Value { id, ty } => {
                    scope.insert(*id, ty.clone());
                }
                RegionParameter::State { id, storage } => {
                    state_scope.insert(*id, *storage);
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
    }

    /// Close the current region, validating that every result is in scope and
    /// fully initialized where it is a storage leaf.
    pub fn end_region(&mut self, results: Vec<RegionResult>) -> Result<GraphRegion, String> {
        let Some(frame) = self.core.frames.pop() else {
            return Err("no region is open".into());
        };
        for result in &results {
            match result {
                RegionResult::Value { id, .. } => {
                    if !frame.scope.contains_key(id) && !self.in_ancestor_scope(*id) {
                        return Err(format!("region result value {} is not in scope", id.0));
                    }
                    if let Some(view) = self.core.value_views.get(id) {
                        let storage = self.core.views[&view.0].storage;
                        self.check_result_storage(storage)?;
                    }
                }
                RegionResult::State { id, .. } => {
                    if !frame.state_scope.contains_key(id) && !self.in_ancestor_states(*id) {
                        return Err(format!("region result state {} is not in scope", id.0));
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
            self.core.results = region.results.clone();
            self.core.root = Some(region.clone());
        }
        Ok(region)
    }

    fn in_ancestor_scope(&self, id: GraphValueId) -> bool {
        self.core
            .frames
            .iter()
            .any(|frame| frame.scope.contains_key(&id))
    }

    fn in_ancestor_states(&self, id: StateTokenId) -> bool {
        self.core
            .frames
            .iter()
            .any(|frame| frame.state_scope.contains_key(&id))
    }

    fn check_result_storage(&self, storage: LogicalStorageId) -> Result<(), String> {
        if self.core.moved.contains_key(&storage) {
            return Err(format!(
                "a result leaf uses storage {} after it was moved",
                storage.0
            ));
        }
        let initialization = &self.core.storages[&storage.0].initialization;
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
        origin: StorageOrigin,
        initialization: Initialization,
    ) -> LogicalStorageId {
        let id = self.core.ids.storage();
        self.core.storages.insert(
            id.0,
            LogicalStorage {
                shape,
                origin,
                initialization,
            },
        );
        id
    }

    /// Declare one semantic view of a storage.
    pub fn declare_view(
        &mut self,
        storage: LogicalStorageId,
        shape: TensorType,
        access: Access,
        transform: ViewTransform,
    ) -> LogicalViewId {
        let id = self.core.ids.view();
        self.core.views.insert(
            id.0,
            LogicalView {
                storage,
                shape,
                access,
                transform,
            },
        );
        id
    }

    /// Allocate a graph value; `view` backs tensor values.
    pub fn fresh_value(
        &mut self,
        ty: ValueType,
        view: Option<LogicalViewId>,
    ) -> Result<GraphValueId, String> {
        if let Some(view) = view {
            let shape = self.core.views[&view.0].shape.clone();
            if ValueType::Tensor(shape) != ty {
                return Err(format!("a declared view cannot back a value of type {ty}"));
            }
        }
        let id = self.core.ids.value();
        self.core.value_types.insert(id, ty);
        if let Some(view) = view {
            self.core.value_views.insert(id, view);
        }
        if let Some(frame) = self.core.frames.last_mut() {
            frame.scope.insert(id, self.core.value_types[&id].clone());
        }
        Ok(id)
    }

    /// Allocate a state token id (the `StateToken` record is assembled by the
    /// construct that owns the transition; `bind_state` records its storage).
    pub fn fresh_state(&mut self) -> StateTokenId {
        self.core.ids.state()
    }

    /// Record which storage a freshly allocated token versions.
    pub fn bind_state(&mut self, token: StateTokenId, storage: LogicalStorageId) {
        if let Some(frame) = self.core.frames.last_mut() {
            frame.state_scope.insert(token, storage);
        }
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
    pub fn set_current_state(&mut self, storage: LogicalStorageId, token: StateTokenId) {
        self.core.current.insert(storage, token);
    }

    pub fn initialization(&self, storage: LogicalStorageId) -> &Initialization {
        &self.core.storages[&storage.0].initialization
    }

    pub fn view(&self, id: LogicalViewId) -> &LogicalView {
        &self.core.views[&id.0]
    }

    pub fn storage(&self, id: LogicalStorageId) -> &LogicalStorage {
        &self.core.storages[&id.0]
    }

    pub fn value_type(&self, id: GraphValueId) -> Result<ValueType, String> {
        self.core
            .value_types
            .get(&id)
            .cloned()
            .ok_or_else(|| format!("value {} has no recorded type", id.0))
    }

    /// The view backing a tensor value.
    pub fn view_of_value(&self, id: GraphValueId) -> Option<LogicalViewId> {
        self.core.value_views.get(&id).copied()
    }

    /// The storage a tensor value reads.
    pub fn storage_of_value(&self, id: GraphValueId) -> Option<LogicalStorageId> {
        self.view_of_value(id).map(|view| self.view(view).storage)
    }

    /// Mark one storage fully initialized. Used for storages handed to a
    /// call through an exclusive borrow: the checker admits an unassigned
    /// argument only when every applicable implementation initializes the
    /// whole parameter, and an assigned argument stays initialized.
    pub fn mark_fully_initialized(&mut self, storage: LogicalStorageId) {
        self.core
            .storages
            .get_mut(&storage.0)
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
            // A write to fresh storage consumes no prior state: the node is
            // the origin of the storage's first version.
            if let Some(token) = self.current_state(storage).ok() {
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
            if self.core.moved.contains_key(&storage) {
                return Err(format!("storage {} is used after it was moved", storage.0));
            }
        }
        let mut out_ids = Vec::new();
        for output in outputs {
            let id = match output {
                Output::Value(ty) => self.fresh_value(ty, None)?,
                Output::View(view) => {
                    let shape = self.core.views[&view.0].shape.clone();
                    self.fresh_value(ValueType::Tensor(shape), Some(view))?
                }
            };
            out_ids.push(id);
        }
        let mut state_outputs = Vec::new();
        if let Some(write) = &write {
            let token = self.fresh_state();
            self.bind_state(token, write.storage);
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

    /// Add one reduction node (never a scalar primitive payload).
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
        if let Some(storage) = self.storage_of_value(operand) {
            self.check_read(storage)?;
        }
        let out = self.fresh_value(result.clone(), None)?;
        let node = LogicalNode {
            inputs: vec![operand],
            state_inputs: Vec::new(),
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
            safety: Vec::new(),
            span,
        };
        self.append_node(node);
        Ok(out)
    }

    /// Add one call node. Inputs must be in scope; borrowed and moved storages
    /// must be at their current token; a move marks the caller storage moved.
    /// `result_values` are the call's data-result values (scalar leaves and
    /// the view-backed values of result storages): the node produces them.
    /// Result storages and `inout` result states become current.
    pub fn add_call(
        &mut self,
        choice: ChoiceId,
        boundary_inputs: Vec<BoundaryInput>,
        boundary_results: Vec<BoundaryResult>,
        result_values: Vec<GraphValueId>,
        span: Span,
    ) -> Result<(), String> {
        let mut inputs = Vec::new();
        let mut state_inputs = Vec::new();
        for input in &boundary_inputs {
            match &input.kind {
                BoundaryInputKind::Value(id) => inputs.push(*id),
                BoundaryInputKind::Shared { value, state }
                | BoundaryInputKind::Exclusive { value, state }
                | BoundaryInputKind::Move { value, state } => {
                    inputs.push(*value);
                    state_inputs.push(*state);
                }
            }
        }
        self.check_inputs(&inputs)?;
        for input in &boundary_inputs {
            match &input.kind {
                BoundaryInputKind::Shared { state, .. } => {
                    self.check_borrowed(*state)?;
                }
                BoundaryInputKind::Exclusive { state, .. }
                | BoundaryInputKind::Move { state, .. } => {
                    self.check_borrowed(*state)?;
                    let storage = self.storage_of_current(*state)?;
                    if self.core.moved.contains_key(&storage) {
                        return Err(format!(
                            "storage {} is used by this call after it was moved",
                            storage.0
                        ));
                    }
                    if matches!(input.kind, BoundaryInputKind::Move { .. }) {
                        self.core.moved.insert(storage, ());
                    }
                }
                BoundaryInputKind::Value(_) => {}
            }
        }
        let outputs = self.values_of(&result_values)?;
        let mut state_outputs = Vec::new();
        for result in &boundary_results {
            match &result.kind {
                BoundaryResultKind::Value(_) => {}
                BoundaryResultKind::Storage { storage, token, .. } => {
                    self.bind_state(*token, *storage);
                    state_outputs.push(StateToken {
                        id: *token,
                        storage: *storage,
                        join: None,
                    });
                    self.core.current.insert(*storage, *token);
                }
                BoundaryResultKind::State(token) => {
                    let storage = self.storage_of_bound(*token)?;
                    state_outputs.push(StateToken {
                        id: *token,
                        storage,
                        join: None,
                    });
                    self.core.current.insert(storage, *token);
                }
            }
        }
        let node = LogicalNode {
            inputs,
            state_inputs,
            kind: LogicalNodeKind::Call(CallNode {
                choice,
                boundary_inputs,
                boundary_results,
            }),
            outputs,
            state_outputs,
            safety: Vec::new(),
            span,
        };
        self.append_node(node);
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
                    let view = self.core.value_views.get(joined).copied();
                    outputs.push(GraphValue {
                        id: *joined,
                        ty: ty.clone(),
                        view,
                    });
                }
                JoinSlot::State {
                    joined, storage, ..
                } => {
                    self.bind_state(*joined, *storage);
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
    /// exit values (one fresh id per value carry, in carried order).
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
            let initial_kind = matches!(slot.initial, RegionInput::Value(_));
            let parameter_kind = matches!(
                body.parameters.get(slot.body_parameter.index()),
                Some(RegionParameter::Value { .. })
            );
            let result_kind = matches!(
                body.results.get(slot.body_result.index()),
                Some(RegionResult::Value { .. })
            );
            if initial_kind != parameter_kind || initial_kind != result_kind {
                return Err(format!(
                    "carried slot {} mixes value and state shapes",
                    slot.body_parameter.0
                ));
            }
            if initial_kind {
                let initial_ty = match slot.initial {
                    RegionInput::Value(id) => self.value_type(id)?,
                    RegionInput::State(_) => unreachable!(),
                };
                let Some(RegionParameter::Value { ty, .. }) =
                    body.parameters.get(slot.body_parameter.index())
                else {
                    unreachable!()
                };
                if initial_ty != *ty {
                    return Err(format!(
                        "carried slot {} has initial type {initial_ty} but body parameter type {ty}",
                        slot.body_parameter.0
                    ));
                }
                let Some(RegionResult::Value { ty: result_ty, .. }) =
                    body.results.get(slot.body_result.index())
                else {
                    unreachable!()
                };
                if initial_ty != *result_ty {
                    return Err(format!(
                        "carried slot {} has initial type {initial_ty} but body result type {result_ty}",
                        slot.body_result.0
                    ));
                }
            } else {
                let initial_storage = match slot.initial {
                    RegionInput::State(token) => self.storage_of_current(token)?,
                    RegionInput::Value(_) => unreachable!(),
                };
                let Some(RegionParameter::State { storage, .. }) =
                    body.parameters.get(slot.body_parameter.index())
                else {
                    unreachable!()
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
                    unreachable!()
                };
                if initial_storage != *result_storage {
                    return Err(format!(
                        "carried state slot {} changes storage",
                        slot.body_result.0
                    ));
                }
            }
        }
        if kind == LoopKind::Independent {
            if !carried.is_empty() {
                return Err("an independent loop has no data carries".into());
            }
        } else if !joins.is_empty() {
            return Err("an ordered loop joins carried states, not visit states".into());
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
        if kind == LoopKind::Ordered {
            for slot in &carried {
                if let RegionInput::State(token) = slot.initial {
                    let storage = self.storage_of_current(token)?;
                    let next = self.fresh_state();
                    self.bind_state(next, storage);
                    state_outputs.push(StateToken {
                        id: next,
                        storage,
                        join: None,
                    });
                }
            }
        } else {
            for (storage, join) in &joins {
                let next = self.fresh_state();
                self.bind_state(next, *storage);
                state_outputs.push(StateToken {
                    id: next,
                    storage: *storage,
                    join: Some(join.clone()),
                });
            }
        }
        let carried_value_ids = carried
            .iter()
            .filter(|slot| matches!(slot.initial, RegionInput::Value(_)))
            .filter_map(|slot| match slot.initial {
                RegionInput::Value(id) => Some(id),
                RegionInput::State(_) => None,
            })
            .collect::<Vec<_>>();
        // The loop's exit values are fresh ids (the value of the last visit),
        // one per value carry, in carried order.
        let mut exit_values = Vec::new();
        for id in &carried_value_ids {
            let ty = self.value_type(*id)?;
            exit_values.push(self.fresh_value(ty, None)?);
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

    /// Whether construction is complete enough to seal (every region closed).
    pub fn seal_check(&self) -> Result<(), String> {
        if !self.core.frames.is_empty() {
            return Err("a region is still open".into());
        }
        if self.core.root.is_none() {
            return Err("the root region was never closed".into());
        }
        Ok(())
    }

    /// Close construction. Available only when every region is closed (for a
    /// void function the result list is legitimately empty); afterwards only
    /// `finish` remains. Call `seal_check` first to recover the builder on
    /// failure.
    pub fn seal(self) -> Result<GraphBuilder<Complete>, String> {
        if !self.core.frames.is_empty() {
            return Err("a region is still open".into());
        }
        if self.core.root.is_none() {
            return Err("the root region was never closed".into());
        }
        Ok(GraphBuilder {
            core: self.core,
            phase: PhantomData,
        })
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

    // -- internals ---------------------------------------------------------

    fn check_inputs(&mut self, inputs: &[GraphValueId]) -> Result<(), String> {
        for id in inputs {
            let visible = self
                .core
                .frames
                .iter()
                .any(|frame| frame.scope.contains_key(id));
            if !visible {
                return Err(format!("input value {} does not dominate this use", id.0));
            }
            if let Some(view) = self.core.value_views.get(id) {
                let storage = self.core.views[&view.0].storage;
                if self.core.moved.contains_key(&storage) {
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
        let initialization = &self.core.storages[&storage.0].initialization;
        if matches!(initialization, Initialization::Uninitialized) {
            return Err(format!(
                "storage {} is read before any element is written",
                storage.0
            ));
        }
        Ok(())
    }

    fn check_borrowed(&self, token: StateTokenId) -> Result<(), String> {
        if !self.core.current.values().any(|current| *current == token) {
            return Err(format!(
                "token {} is not the current state of its storage",
                token.0
            ));
        }
        Ok(())
    }

    fn storage_of_current(&self, token: StateTokenId) -> Result<LogicalStorageId, String> {
        self.core
            .current
            .iter()
            .find(|(_, current)| **current == token)
            .map(|(storage, _)| *storage)
            .ok_or_else(|| format!("token {} is not a current state", token.0))
    }

    /// The storage a bound token versions (recorded by `bind_state`).
    fn storage_of_bound(&self, token: StateTokenId) -> Result<LogicalStorageId, String> {
        for frame in self.core.frames.iter().rev() {
            if let Some(storage) = frame.state_scope.get(&token) {
                return Ok(*storage);
            }
        }
        Err(format!("token {} is not bound to a storage", token.0))
    }

    fn values_of(&self, ids: &[GraphValueId]) -> Result<Vec<GraphValue>, String> {
        let mut out = Vec::new();
        for id in ids {
            out.push(GraphValue {
                id: *id,
                ty: self.value_type(*id)?,
                view: self.core.value_views.get(id).copied(),
            });
        }
        Ok(out)
    }

    fn apply_coverage(
        &mut self,
        storage: LogicalStorageId,
        coverage: &Coverage,
        initializing: bool,
    ) {
        let initialization = self.core.storages[&storage.0].initialization.clone();
        self.core
            .storages
            .get_mut(&storage.0)
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
            frame.scope.insert(output.id, output.ty.clone());
            if let Some(view) = output.view {
                self.core.value_views.insert(output.id, view);
            }
        }
        for token in &node.state_outputs {
            frame.state_scope.insert(token.id, token.storage);
        }
        frame.nodes.push(node);
    }
}

impl GraphBuilder<Complete> {
    /// Finish the sealed graph, returning the id allocator for the next graph.
    pub fn finish(self) -> (TaskGraph, Ids) {
        let core = self.core;
        let graph = TaskGraph {
            choice: core.choice,
            alternative: core.alternative,
            parameters: core.parameters,
            storages: IdVec::from_iter(
                core.storages
                    .into_iter()
                    .map(|(id, storage)| (LogicalStorageId(id), storage)),
            ),
            views: IdVec::from_iter(
                core.views
                    .into_iter()
                    .map(|(id, view)| (LogicalViewId(id), view)),
            ),
            root: core.root.expect("seal guarantees a root region"),
            results: core.results,
        };
        (graph, core.ids)
    }
}
