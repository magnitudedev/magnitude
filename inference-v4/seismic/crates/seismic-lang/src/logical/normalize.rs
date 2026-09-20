//! Semantic-to-logical construction: a single forward pass over the checked
//! body.
//!
//! One lexical environment maps each local to its current value/view/state.
//! Constructors create outputs and dependencies immediately; nothing is
//! collected by recursive scanning afterwards. Arguments are region
//! parameters, never
//! pseudo-values; entry results are compiler-owned logical storage; call
//! results are occurrence-owned storage. Initialization follows
//! `Uninitialized -> PartiallyInitialized(coverage proof) -> FullyInitialized`
//! with structurally composing disjoint coverage.

use super::builder::{GraphBuilder, Ids, LoopSpec, Output, PrimitiveSpec, WriteEffect};
use super::*;
use crate::check::atom_var;
use crate::family;
use crate::intrinsics::{accumulator_dtype, PrimitiveId};
use crate::sir::{
    BlockTerminator, CheckedBlock, CheckedCall, CheckedExpr, CheckedExprKind, CheckedIndex,
    CheckedPlace, CheckedRange, CheckedStmt, DefId, Definition, IntrinsicUse, Literal, LocalId,
    LoopKind, LoopMutationSummary, Mode, ParamOwnership, Pattern, Program,
};
use crate::span::Span;
use crate::sym::{Atom, Sym};
use crate::syntax::ast::{AssignOp, BinaryOp};
use crate::types::{
    DType, Elem, ExtentExpr, NonEmpty, RuntimeExtentId, TensorType, ValuePath, ValueType,
};
use std::collections::{BTreeMap, BTreeSet};

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Why one occurrence has no applicable implementation.
#[derive(Clone, Debug)]
pub struct OccurrenceRejection {
    /// Human-readable occurrence location (definition and span).
    pub occurrence: String,
    pub rejections: Vec<(DefId, String)>,
}

/// The applicability report of a failed construction: every occurrence that
/// had no applicable semantic implementation, with per-definition reasons.
#[derive(Clone, Debug, Default)]
pub struct ApplicabilityReport {
    pub entry: String,
    pub occurrences: Vec<OccurrenceRejection>,
}

/// Construction failure of a checked program. `InvalidProgram` names a
/// representability defect (the `CompilerBug` failure class): checked input
/// the logical layer could not construct.
#[derive(Clone, Debug)]
pub enum LogicalConstructionError {
    NoApplicableImplementation(ApplicabilityReport),
    InvalidProgram(String),
}

pub(super) enum BuildError {
    NoImplementation(ApplicabilityReport),
    Invalid(String),
}

// ---------------------------------------------------------------------------
// Boundary leaf decomposition
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
enum BoundaryLeaf {
    Scalar(DType),
    Index(ExtentExpr),
    Range(ExtentExpr),
    Tensor(TensorType),
    /// An opaque capability value. Checking keeps these inside one backend's
    /// alternatives; they never cross a portable boundary or the public ABI.
    Capability,
}

fn boundary_leaves(ty: &ValueType) -> Vec<(ValuePath, BoundaryLeaf)> {
    let mut out = Vec::new();
    fn walk(ty: &ValueType, path: &ValuePath, out: &mut Vec<(ValuePath, BoundaryLeaf)>) {
        match ty {
            ValueType::Scalar(d) => out.push((path.clone(), BoundaryLeaf::Scalar(*d))),
            ValueType::Index { bound } => {
                out.push((path.clone(), BoundaryLeaf::Index(bound.clone())))
            }
            ValueType::Range { bound } => {
                out.push((path.clone(), BoundaryLeaf::Range(bound.clone())))
            }
            ValueType::Tensor(s) => out.push((path.clone(), BoundaryLeaf::Tensor(s.clone()))),
            ValueType::Tuple(items) => {
                for (i, item) in items.iter().enumerate() {
                    walk(item, &path.extend(i as u32), out);
                }
            }
            ValueType::CapabilityValue(_) => out.push((path.clone(), BoundaryLeaf::Capability)),
            ValueType::Void => {}
        }
    }
    walk(ty, &ValuePath::default(), &mut out);
    out
}

fn leaf_value_type(leaf: &BoundaryLeaf) -> ValueType {
    match leaf {
        BoundaryLeaf::Scalar(d) => ValueType::Scalar(*d),
        BoundaryLeaf::Index(bound) => ValueType::Index {
            bound: bound.clone(),
        },
        BoundaryLeaf::Range(bound) => ValueType::Range {
            bound: bound.clone(),
        },
        BoundaryLeaf::Tensor(s) => ValueType::Tensor(s.clone()),
        BoundaryLeaf::Capability => ValueType::CapabilityValue(crate::types::CapabilityValueType {
            target: String::new(),
            name: String::new(),
            shape: Vec::new(),
            elem: None,
        }),
    }
}

// ---------------------------------------------------------------------------
// Block scans
// ---------------------------------------------------------------------------

fn referenced_locals(block: &CheckedBlock) -> BTreeSet<LocalId> {
    let mut out = BTreeSet::new();
    scan_block(block, &mut |expr| {
        if let CheckedExprKind::Local(id) = &expr.kind {
            out.insert(*id);
        }
    });
    // A mutation place's storage root is referenced even though it never
    // appears as an operand expression.
    fn place_roots(block: &CheckedBlock, out: &mut BTreeSet<LocalId>) {
        for stmt in &block.statements {
            match stmt {
                CheckedStmt::Assign { place, .. } => scan_place_roots(place, out),
                CheckedStmt::Loop { body, .. } => place_roots(body, out),
                CheckedStmt::If {
                    then_body,
                    else_body,
                    ..
                } => {
                    place_roots(then_body, out);
                    place_roots(else_body, out);
                }
                CheckedStmt::Let { .. } | CheckedStmt::Evaluate(_) => {}
            }
        }
    }
    place_roots(block, &mut out);
    out
}

fn scan_place_roots(place: &CheckedPlace, out: &mut BTreeSet<LocalId>) {
    match place {
        CheckedPlace::Local { root } => {
            out.insert(*root);
        }
        CheckedPlace::Element { root, .. } => {
            out.insert(*root);
        }
        CheckedPlace::Tuple(places) => places.iter().for_each(|p| scan_place_roots(p, out)),
    }
}

fn scan_block<'a>(block: &'a CheckedBlock, visit: &mut dyn FnMut(&'a CheckedExpr)) {
    for stmt in &block.statements {
        scan_stmt(stmt, visit);
    }
    if let BlockTerminator::Return(values) = &block.terminator {
        for value in values {
            scan_expr(value, visit);
        }
    }
}

fn scan_stmt<'a>(stmt: &'a CheckedStmt, visit: &mut dyn FnMut(&'a CheckedExpr)) {
    match stmt {
        CheckedStmt::Let { value, .. } => scan_expr(value, visit),
        CheckedStmt::Assign { place, value, .. } => {
            scan_place(place, visit);
            scan_expr(value, visit);
        }
        CheckedStmt::Loop { range, body, .. } => {
            scan_expr(&range.start, visit);
            scan_expr(&range.end, visit);
            scan_block(body, visit);
        }
        CheckedStmt::If {
            condition,
            then_body,
            else_body,
        } => {
            scan_expr(condition, visit);
            scan_block(then_body, visit);
            scan_block(else_body, visit);
        }
        CheckedStmt::Evaluate(expr) => scan_expr(expr, visit),
    }
}

fn scan_place<'a>(place: &'a CheckedPlace, visit: &mut dyn FnMut(&'a CheckedExpr)) {
    match place {
        CheckedPlace::Local { .. } => {}
        CheckedPlace::Element { indices, .. } => {
            for index in indices {
                match index {
                    CheckedIndex::Point(p) => scan_expr(p, visit),
                    CheckedIndex::Range { start, end } => {
                        start.iter().chain(end).for_each(|e| scan_expr(e, visit));
                    }
                }
            }
        }
        CheckedPlace::Tuple(places) => places.iter().for_each(|p| scan_place(p, visit)),
    }
}

fn scan_expr<'a>(expr: &'a CheckedExpr, visit: &mut dyn FnMut(&'a CheckedExpr)) {
    visit(expr);
    match &expr.kind {
        CheckedExprKind::Primitive { operands, .. } => {
            operands.iter().for_each(|o| scan_expr(o, visit))
        }
        CheckedExprKind::Capability { args, .. } => args.iter().for_each(|a| scan_expr(a, visit)),
        CheckedExprKind::Call { args, .. } => args.iter().for_each(|a| scan_expr(a, visit)),
        CheckedExprKind::Literal(_) | CheckedExprKind::Local(_) => {}
    }
}

/// Locals bound by `let`/`let mut` and loop binders inside this block.
fn declared_locals(block: &CheckedBlock) -> BTreeSet<LocalId> {
    let mut out = BTreeSet::new();
    fn declare_pattern(pattern: &Pattern, out: &mut BTreeSet<LocalId>) {
        match pattern {
            Pattern::Local(id) => {
                out.insert(*id);
            }
            Pattern::Tuple(items) => items.iter().for_each(|p| declare_pattern(p, out)),
        }
    }
    fn walk(block: &CheckedBlock, out: &mut BTreeSet<LocalId>) {
        for stmt in &block.statements {
            match stmt {
                CheckedStmt::Let { pattern, .. } => declare_pattern(pattern, out),
                CheckedStmt::Loop { binder, body, .. } => {
                    out.insert(*binder);
                    walk(body, out);
                }
                CheckedStmt::If {
                    then_body,
                    else_body,
                    ..
                } => {
                    walk(then_body, out);
                    walk(else_body, out);
                }
                CheckedStmt::Assign { .. } | CheckedStmt::Evaluate(_) => {}
            }
        }
    }
    walk(block, &mut out);
    out
}

fn free_locals(block: &CheckedBlock) -> BTreeSet<LocalId> {
    referenced_locals(block)
        .difference(&declared_locals(block))
        .copied()
        .collect()
}

/// The storage root of a place or argument expression, following view
/// transforms: the local whose current binding owns the storage.
fn root_var(expr: &CheckedExpr) -> Option<LocalId> {
    match &expr.kind {
        CheckedExprKind::Local(id) => Some(*id),
        CheckedExprKind::Primitive {
            id: PrimitiveId::SliceView { .. } | PrimitiveId::Reshape | PrimitiveId::Transpose,
            operands,
        } => root_var(operands.first()?),
        _ => None,
    }
}

/// Locals whose storage this block may write: assignment places, atomic
/// bases, and call arguments borrowed into callees. Over-approximated.
fn written_roots(block: &CheckedBlock) -> BTreeSet<LocalId> {
    let mut out = BTreeSet::new();
    fn place_root(place: &CheckedPlace, out: &mut BTreeSet<LocalId>) {
        match place {
            CheckedPlace::Local { root } => {
                out.insert(*root);
            }
            CheckedPlace::Element { root, .. } => {
                out.insert(*root);
            }
            CheckedPlace::Tuple(places) => places.iter().for_each(|p| place_root(p, out)),
        }
    }
    fn walk(block: &CheckedBlock, out: &mut BTreeSet<LocalId>) {
        for stmt in &block.statements {
            match stmt {
                CheckedStmt::Assign { place, .. } => place_root(place, out),
                CheckedStmt::Loop { body, .. } => walk(body, out),
                CheckedStmt::If {
                    then_body,
                    else_body,
                    ..
                } => {
                    walk(then_body, out);
                    walk(else_body, out);
                }
                CheckedStmt::Let { .. } | CheckedStmt::Evaluate(_) => {}
            }
        }
        scan_block(block, &mut |expr| {
            if let CheckedExprKind::Primitive {
                id: PrimitiveId::Atomic { .. },
                operands,
            } = &expr.kind
            {
                if let Some(root) = root_var(operands.first().expect("atomic add has a base")) {
                    out.insert(root);
                }
            }
            if let CheckedExprKind::Call { args, .. } = &expr.kind {
                for arg in args {
                    if let Some(root) = root_var(arg) {
                        out.insert(root);
                    }
                }
            }
        });
    }
    walk(block, &mut out);
    out
}

/// Locals rebound (whole-assignment targets) inside this block.
fn rebound_roots(block: &CheckedBlock) -> BTreeSet<LocalId> {
    let mut out = BTreeSet::new();
    fn walk(block: &CheckedBlock, out: &mut BTreeSet<LocalId>) {
        for stmt in &block.statements {
            match stmt {
                CheckedStmt::Assign { place, .. } => {
                    if let CheckedPlace::Local { root } = place {
                        out.insert(*root);
                    }
                }
                CheckedStmt::Loop { body, .. } => walk(body, out),
                CheckedStmt::If {
                    then_body,
                    else_body,
                    ..
                } => {
                    walk(then_body, out);
                    walk(else_body, out);
                }
                CheckedStmt::Let { .. } | CheckedStmt::Evaluate(_) => {}
            }
        }
    }
    walk(block, &mut out);
    out
}

fn returns_directly(block: &CheckedBlock) -> bool {
    matches!(block.terminator, BlockTerminator::Return(_))
}

// ---------------------------------------------------------------------------
// Program-level factory
// ---------------------------------------------------------------------------

type ShapeEnv = BTreeMap<String, ExtentExpr>;
type ElemEnv = BTreeMap<String, Elem>;

struct Factory<'a> {
    program: &'a Program,
    target: &'a EffectiveTargetIdentity,
    supports: &'a dyn Fn(&IntrinsicUse) -> Result<(), String>,
    entry: String,
    ids: Ids,
    next_choice: u32,
    next_graph: u32,
    choices: Vec<Option<ImplementationChoice>>,
    graphs: Vec<Option<TaskGraph>>,
    runtime_extents: Vec<RuntimeExtent>,
    rejections: Vec<OccurrenceRejection>,
    /// Set when any occurrence had no applicable implementation; the error
    /// cascades to the top as `NoImplementation`.
    no_implementation: bool,
    depth: usize,
}

impl<'a> Factory<'a> {
    fn new(
        program: &'a Program,
        target: &'a EffectiveTargetIdentity,
        supports: &'a dyn Fn(&IntrinsicUse) -> Result<(), String>,
        entry: &str,
    ) -> Factory<'a> {
        Factory {
            program,
            target,
            supports,
            entry: entry.to_string(),
            ids: Ids::default(),
            next_choice: 0,
            next_graph: 0,
            choices: Vec::new(),
            graphs: Vec::new(),
            runtime_extents: Vec::new(),
            rejections: Vec::new(),
            no_implementation: false,
            depth: 0,
        }
    }

    fn alloc_choice(&mut self) -> ChoiceId {
        let id = ChoiceId(self.next_choice);
        self.next_choice += 1;
        self.choices.push(None);
        id
    }

    fn install_choice(&mut self, id: ChoiceId, choice: ImplementationChoice) {
        self.choices[id.index()] = Some(choice);
    }

    fn alloc_graph_slot(&mut self) -> GraphId {
        let id = GraphId(self.next_graph);
        self.next_graph += 1;
        self.graphs.push(None);
        id
    }

    fn runtime_extent(&mut self, value: RuntimeScalarExpr, capacity: u64) -> RuntimeExtentId {
        let id = RuntimeExtentId(self.runtime_extents.len() as u32);
        self.runtime_extents.push(RuntimeExtent {
            id,
            value,
            capacity,
            expected: None,
        });
        id
    }

    /// Build the task graph of one alternative of one occurrence by the
    /// forward pass over the definition's checked body. The graph slot is
    /// reserved first so `LogicalAlternative`s can reference the id before
    /// the (recursive) construction completes.
    fn build_graph(
        &mut self,
        definition: &Definition,
        shapes: ShapeEnv,
        elems: ElemEnv,
        choice: ChoiceId,
        alternative: u32,
        is_entry: bool,
    ) -> Result<GraphId, BuildError> {
        if self.depth > self.program.definitions.len() {
            return Err(BuildError::Invalid(
                "the static call graph is not acyclic".into(),
            ));
        }
        self.depth += 1;
        let graph_id = self.alloc_graph_slot();
        let builder = GraphBuilder::new(choice, alternative, std::mem::take(&mut self.ids));
        let result = (|| -> Result<(TaskGraph, Ids), String> {
            let invalid = |reason: String| reason;
            let body = &definition.body;
            let mut work = Work {
                builder,
                shapes,
                elems,
                local_tys: &body.locals,
                locals: vec![None; body.locals.len()],
                constants: BTreeMap::new(),
                results: result_slots(&definition.result),
                result_states: vec![
                    None;
                    definition
                        .params
                        .iter()
                        .filter(|p| p.mode == Mode::Inout)
                        .count()
                ],
                inout: definition
                    .params
                    .iter()
                    .enumerate()
                    .filter(|(_, p)| p.mode == Mode::Inout)
                    .map(|(ordinal, p)| (ordinal as u32, p.local))
                    .collect(),
                loops: Vec::new(),
                if_depth: 0,
                extent_memo: BTreeMap::new(),
                is_entry,
            };

            // Root region parameters: one value (or state) per interface leaf.
            let mut root_params = Vec::new();
            // (param local, leaf value id, is_tuple_component)
            let mut param_leaf_values: Vec<Vec<(ValuePath, GraphValueId)>> =
                vec![Vec::new(); definition.params.len()];
            for (ordinal, param) in definition.params.iter().enumerate() {
                let ty = work.convert_type(self, &param.ty).map_err(invalid)?;
                for (path, leaf) in boundary_leaves(&ty) {
                    match leaf {
                        BoundaryLeaf::Tensor(shape) => {
                            let storage = work.builder.declare_storage(
                                shape.clone(),
                                StorageOrigin::Parameter {
                                    ordinal: ordinal as u32,
                                    path: path.clone(),
                                    name: param.name.clone(),
                                },
                                Initialization::FullyInitialized,
                            );
                            let view = work.builder.declare_view(
                                storage,
                                shape,
                                match param.ownership {
                                    ParamOwnership::Shared => Access::Shared,
                                    _ => Access::Exclusive,
                                },
                                ViewTransform::Identity,
                            );
                            let value = work
                                .builder
                                .fresh_value(
                                    ValueType::Tensor(work.builder.view(view).shape.clone()),
                                    Some(view),
                                )
                                .map_err(invalid)?;
                            let token = work.builder.fresh_state();
                            work.builder.bind_state(token, storage);
                            root_params.push(RegionParameter::Value {
                                id: value,
                                ty: ValueType::Tensor(work.builder.view(view).shape.clone()),
                            });
                            root_params.push(RegionParameter::State { id: token, storage });
                            param_leaf_values[ordinal].push((path, value));
                        }
                        other => {
                            let ty = leaf_value_type(&other);
                            let value = work
                                .builder
                                .fresh_value(ty.clone(), None)
                                .map_err(invalid)?;
                            root_params.push(RegionParameter::Value { id: value, ty });
                            param_leaf_values[ordinal].push((path, value));
                        }
                    }
                }
            }
            work.builder.begin_root(root_params).map_err(invalid)?;

            // Bind each parameter local: a single leaf binds directly; a
            // tuple parameter packs its leaf values.
            for (ordinal, param) in definition.params.iter().enumerate() {
                let leaves = &param_leaf_values[ordinal];
                let value = if leaves.len() == 1 {
                    leaves[0].1
                } else {
                    let components = leaves.iter().map(|(_, id)| *id).collect::<Vec<_>>();
                    let tys: Vec<ValueType> = components
                        .iter()
                        .map(|id| work.builder.value_type(*id).map_err(invalid))
                        .collect::<Result<_, _>>()?;
                    let ty = ValueType::Tuple(
                        NonEmpty::new(tys).ok_or_else(|| invalid("a tuple is empty".into()))?,
                    );
                    let spec = PrimitiveSpec {
                        op: PrimitiveOp::Primitive(PrimitiveId::TuplePack),
                        inputs: components,
                        reads: Vec::new(),
                        write: None,
                        outputs: vec![Output::Value(ty)],
                        safety: Vec::new(),
                        span: definition.span,
                    };
                    work.add(self, spec)?.expect("tuple pack has a value")
                };
                work.locals[param.local] = Some(value);
            }

            let flow = work.block(self, &body.root)?;
            // A void body may fall through: its terminator retains completion
            // and final states without an explicit `return`.
            if !matches!(flow, Flow::Returned) && !definition.result.is_void() {
                return Err(format!(
                    "`{}` does not end every path in `return`",
                    definition.name
                ));
            }

            // Boundary results: the function result values (materialized into
            // compiler-owned storage at the entry) and the final states of
            // `inout` parameters.
            let mut region_results = Vec::new();
            let result_values: Vec<GraphValueId> = work
                .results
                .iter()
                .map(|slot| {
                    slot.ok_or_else(|| {
                        "a `return` path does not bind every result value".to_string()
                    })
                })
                .collect::<Result<_, String>>()
                .map_err(invalid)?;
            for value in result_values {
                let ty = work.builder.value_type(value).map_err(invalid)?;
                let value = if work.is_entry {
                    work.materialize_result_leaf(self, value, &ty, definition.span)
                        .map_err(invalid)?
                } else {
                    value
                };
                region_results.push(RegionResult::Value { id: value, ty });
            }
            let inout: Vec<(u32, usize)> = work.inout.clone();
            for (ordinal, local) in &inout {
                let storage = work
                    .storage_of_local(*local)
                    .ok_or_else(|| format!("inout parameter `{ordinal}` has no storage"))
                    .map_err(invalid)?;
                let token = work
                    .builder
                    .current_state(storage)
                    .map_err(|reason| invalid(reason))?;
                region_results.push(RegionResult::State {
                    id: token,
                    storage,
                    join: None,
                });
            }
            if region_results.is_empty() && !definition.result.is_void() {
                return Err(format!(
                    "`{}` produces no boundary results",
                    definition.name
                ));
            }
            work.builder.end_region(region_results).map_err(invalid)?;
            work.builder.seal_check().map_err(invalid)?;
            let builder = work.builder;
            let sealed = builder.seal().expect("seal_check passed");
            let (graph, ids) = sealed.finish();
            Ok((graph, ids))
        })();
        match result {
            Ok((graph, ids)) => {
                self.ids = ids;
                self.graphs[graph_id.index()] = Some(graph);
            }
            Err(reason) => {
                // Construction errors are fatal for the whole program, so the
                // allocator is not recovered.
                if self.no_implementation {
                    self.no_implementation = false;
                    self.depth -= 1;
                    return Err(BuildError::NoImplementation(self.report()));
                }
                self.depth -= 1;
                return Err(BuildError::Invalid(reason));
            }
        }
        self.depth -= 1;
        Ok(graph_id)
    }

    fn report(&self) -> ApplicabilityReport {
        ApplicabilityReport {
            entry: self.entry.clone(),
            occurrences: self.rejections.clone(),
        }
    }
}

fn result_slots(result: &ValueType) -> Vec<Option<GraphValueId>> {
    let count = match result {
        ValueType::Void => 0,
        ValueType::Tuple(items) => items.len(),
        _ => 1,
    };
    vec![None; count]
}

// ---------------------------------------------------------------------------
// Per-graph forward pass
// ---------------------------------------------------------------------------

enum Flow {
    Next,
    Returned,
}

/// Constant-fold one retained runtime expression.
fn fold_scalar(expr: RuntimeScalarExpr) -> RuntimeScalarExpr {
    fn binary(
        a: RuntimeScalarExpr,
        b: RuntimeScalarExpr,
        op: fn(i64, i64) -> Option<i64>,
        rebuild: fn(Box<RuntimeScalarExpr>, Box<RuntimeScalarExpr>) -> RuntimeScalarExpr,
    ) -> RuntimeScalarExpr {
        let a = fold_scalar(a);
        let b = fold_scalar(b);
        if let (RuntimeScalarExpr::Const(x), RuntimeScalarExpr::Const(y)) = (&a, &b) {
            if let Some(value) = op(*x, *y) {
                return RuntimeScalarExpr::Const(value);
            }
        }
        rebuild(Box::new(a), Box::new(b))
    }
    match expr {
        RuntimeScalarExpr::Add(a, b) => binary(
            *a,
            *b,
            |x, y| x.checked_add(y),
            |a, b| RuntimeScalarExpr::Add(a, b),
        ),
        RuntimeScalarExpr::Sub(a, b) => binary(
            *a,
            *b,
            |x, y| x.checked_sub(y),
            |a, b| RuntimeScalarExpr::Sub(a, b),
        ),
        RuntimeScalarExpr::Mul(a, b) => binary(
            *a,
            *b,
            |x, y| x.checked_mul(y),
            |a, b| RuntimeScalarExpr::Mul(a, b),
        ),
        RuntimeScalarExpr::Div(a, b) => binary(
            *a,
            *b,
            |x, y| x.checked_div(y),
            |a, b| RuntimeScalarExpr::Div(a, b),
        ),
        RuntimeScalarExpr::Rem(a, b) => binary(
            *a,
            *b,
            |x, y| x.checked_rem_euclid(y),
            |a, b| RuntimeScalarExpr::Rem(a, b),
        ),
        other => other,
    }
}

/// What one symbolic atom resolves to in the current environment.
enum ResolvedAtom {
    Value(GraphValueId),
    Extent(ExtentExpr),
}

#[derive(Default, Clone, Copy)]
struct LoopWrite {
    atomic: bool,
    plain: bool,
}

struct LoopInfo {
    binder: GraphValueId,
    start: ExtentExpr,
    end: ExtentExpr,
    /// `if`-nesting depth at loop entry: writes at the same depth are
    /// unconditional within the loop.
    if_depth: usize,
    /// Storages captured (free) by this loop, for write classification.
    captured: BTreeSet<LogicalStorageId>,
    /// Writes seen so far, classified.
    writes: BTreeMap<LogicalStorageId, LoopWrite>,
    /// Atomic operations admitted inside, for the join record.
    atomics: Vec<AtomicOperation>,
}

struct Work<'w> {
    builder: GraphBuilder<Building>,
    shapes: ShapeEnv,
    elems: ElemEnv,
    /// Checked local types (indexed by `LocalId`).
    local_tys: &'w [crate::sir::CheckedLocal],
    /// Lexical environment: local -> current value (view-backed for tensors).
    locals: Vec<Option<GraphValueId>>,
    /// Integer constants by producing value, for static coverage proofs.
    constants: BTreeMap<GraphValueId, i64>,
    /// Function result slots, bound by `return`.
    results: Vec<Option<GraphValueId>>,
    /// Final states of `inout` parameters, bound by `return`.
    result_states: Vec<Option<StateTokenId>>,
    /// (interface ordinal, local) of `inout` parameters.
    inout: Vec<(u32, usize)>,
    loops: Vec<LoopInfo>,
    if_depth: usize,
    extent_memo: BTreeMap<String, RuntimeExtentId>,
    is_entry: bool,
}

impl<'w> Work<'w> {
    fn add(
        &mut self,
        f: &mut Factory,
        spec: PrimitiveSpec,
    ) -> Result<Option<GraphValueId>, String> {
        let outcome = self.builder.add_primitive(spec)?;
        let _ = f;
        Ok(outcome.outputs.first().copied())
    }

    fn storage_of_local(&self, local: usize) -> Option<LogicalStorageId> {
        self.locals
            .get(local)
            .and_then(|binding| *binding)
            .and_then(|id| {
                self.builder
                    .view_of_value(id)
                    .map(|view| self.builder.view(view).storage)
            })
    }

    // -- symbolic extent resolution ----------------------------------------

    /// Convert a checked type to the logical level: `Sym` extents become
    /// `Static` or fresh `Runtime` extents, element parameters resolve.
    fn convert_type(&mut self, f: &mut Factory, ty: &ValueType) -> Result<ValueType, String> {
        match ty {
            ValueType::Scalar(d) => Ok(ValueType::Scalar(*d)),
            ValueType::Index { bound } => Ok(ValueType::Index {
                bound: self.convert_extent(f, bound)?,
            }),
            ValueType::Range { bound } => Ok(ValueType::Range {
                bound: self.convert_extent(f, bound)?,
            }),
            ValueType::Tensor(s) => Ok(ValueType::Tensor(TensorType {
                axes: s
                    .axes
                    .iter()
                    .map(|axis| self.convert_extent(f, axis))
                    .collect::<Result<_, _>>()?,
                elem: self.convert_elem(&s.elem)?,
                packed_axis: s.packed_axis,
            })),
            ValueType::Tuple(items) => Ok(ValueType::Tuple(
                NonEmpty::new(
                    items
                        .iter()
                        .map(|item| self.convert_type(f, item))
                        .collect::<Result<_, _>>()?,
                )
                .expect("the source tuple is nonempty"),
            )),
            ValueType::CapabilityValue(n) => Ok(ValueType::CapabilityValue(n.clone())),
            ValueType::Void => Ok(ValueType::Void),
        }
    }

    fn convert_extent(
        &mut self,
        f: &mut Factory,
        extent: &ExtentExpr,
    ) -> Result<ExtentExpr, String> {
        match extent {
            ExtentExpr::Static(n) => Ok(ExtentExpr::Static(*n)),
            ExtentExpr::Runtime(id) => Ok(ExtentExpr::Runtime(*id)),
            ExtentExpr::Sym(sym) => self.resolve_sym(f, sym),
        }
    }

    fn convert_elem(&self, elem: &Elem) -> Result<Elem, String> {
        match elem {
            Elem::Param(p) => self.elems.get(p).cloned().ok_or_else(|| {
                format!("element parameter `{p}` is not bound at this specialization")
            }),
            other => Ok(other.clone()),
        }
    }

    /// Resolve one symbolic extent: shape parameters through the environment,
    /// value atoms to the current graph value (allocating a runtime extent).
    fn resolve_sym(&mut self, f: &mut Factory, sym: &Sym) -> Result<ExtentExpr, String> {
        if let Some(value) = sym.as_constant() {
            return Ok(ExtentExpr::Static(u64::try_from(value).unwrap_or(0)));
        }
        let memo_key = self.sym_key(sym)?;
        if let Some(id) = self.extent_memo.get(&memo_key) {
            return Ok(ExtentExpr::Runtime(*id));
        }
        let expr = self.scalar_expr(f, sym)?;
        if let RuntimeScalarExpr::Const(value) = expr {
            return Ok(ExtentExpr::Static(value as u64));
        }
        let capacity = sym
            .eval(&|name| {
                self.shape_or_value(name)
                    .and_then(|resolved| match resolved {
                        ResolvedAtom::Extent(ExtentExpr::Static(value)) => Some(value as i64),
                        _ => None,
                    })
            })
            .and_then(|value| u64::try_from(value).ok())
            .unwrap_or(u64::MAX);
        let id = f.runtime_extent(expr, capacity);
        self.extent_memo.insert(memo_key, id);
        Ok(ExtentExpr::Runtime(id))
    }

    /// A memo key for one symbolic extent: its display plus the current value
    /// ids of every value atom it mentions (SSA rebinding changes the key).
    fn sym_key(&self, sym: &Sym) -> Result<String, String> {
        let mut key = sym.to_string();
        for atom in sym.atoms() {
            if let Atom::Param(name) = &atom {
                if let Some(local) = atom_var(name) {
                    let value = self
                        .locals
                        .get(local)
                        .and_then(|binding| *binding)
                        .ok_or_else(|| format!("`{name}` has no current value"))?;
                    key.push_str(&format!("#{}", value.0));
                }
            }
        }
        Ok(key)
    }

    fn shape_or_value(&self, name: &str) -> Option<ResolvedAtom> {
        if let Some(local) = atom_var(name) {
            return self
                .locals
                .get(local)
                .and_then(|binding| *binding)
                .map(ResolvedAtom::Value);
        }
        self.shapes.get(name).cloned().map(ResolvedAtom::Extent)
    }

    /// Build the retained runtime expression of one symbolic extent, with
    /// constant folding so fully static symbols collapse to `Const`.
    fn scalar_expr(&mut self, f: &mut Factory, sym: &Sym) -> Result<RuntimeScalarExpr, String> {
        let mut total = RuntimeScalarExpr::Const(0);
        for (monomial, coefficient) in sym.monomials() {
            let mut term = RuntimeScalarExpr::Const(coefficient);
            for (atom, power) in monomial {
                let atom_expr = self.atom_expr(f, atom)?;
                for _ in 0..*power {
                    term = RuntimeScalarExpr::Mul(Box::new(term), Box::new(atom_expr.clone()));
                }
            }
            total = RuntimeScalarExpr::Add(Box::new(total), Box::new(term));
        }
        Ok(fold_scalar(total))
    }

    fn atom_expr(&mut self, f: &mut Factory, atom: &Atom) -> Result<RuntimeScalarExpr, String> {
        match atom {
            Atom::Param(name) => match self.shape_or_value(name) {
                Some(ResolvedAtom::Value(id)) => Ok(RuntimeScalarExpr::Value(id)),
                Some(ResolvedAtom::Extent(ExtentExpr::Static(n))) => {
                    Ok(RuntimeScalarExpr::Const(n as i64))
                }
                Some(ResolvedAtom::Extent(ExtentExpr::Runtime(id))) => {
                    Ok(RuntimeScalarExpr::Extent(id))
                }
                Some(ResolvedAtom::Extent(ExtentExpr::Sym(sym))) => {
                    let resolved = self.resolve_sym(f, &sym)?;
                    self.atom_expr(f, &Atom::Param(format!("__resolved_{}", resolved)))
                        .or_else(|_| {
                            Ok(match resolved {
                                ExtentExpr::Static(n) => RuntimeScalarExpr::Const(n as i64),
                                ExtentExpr::Runtime(id) => RuntimeScalarExpr::Extent(id),
                                ExtentExpr::Sym(_) => {
                                    unreachable!("resolve_sym returns logical extents")
                                }
                            })
                        })
                }
                None => {
                    // A runtime-length atom (`@dyn#n`) realizes to the runtime
                    // extent allocated when its slice view was built; look it up
                    // so compound extents mentioning it resolve.
                    if name.starts_with('@') {
                        if let Some(id) = self.extent_memo.get(name) {
                            return Ok(RuntimeScalarExpr::Extent(*id));
                        }
                    }
                    Err(format!(
                        "`{name}` is neither a shape parameter nor a bound value here"
                    ))
                }
            },
            Atom::Quot(numerator, denominator) => {
                let n = self.scalar_expr(f, numerator)?;
                let d = self.scalar_expr(f, denominator)?;
                Ok(RuntimeScalarExpr::Div(Box::new(n), Box::new(d)))
            }
            Atom::Rem(numerator, denominator) => {
                let n = self.scalar_expr(f, numerator)?;
                let d = self.scalar_expr(f, denominator)?;
                Ok(RuntimeScalarExpr::Rem(Box::new(n), Box::new(d)))
            }
        }
    }

    // -- statements ---------------------------------------------------------

    fn block(&mut self, f: &mut Factory, block: &CheckedBlock) -> Result<Flow, String> {
        self.statements(f, &block.statements, &block.terminator)
    }

    fn statements(
        &mut self,
        f: &mut Factory,
        statements: &[CheckedStmt],
        terminator: &BlockTerminator,
    ) -> Result<Flow, String> {
        for (index, stmt) in statements.iter().enumerate() {
            if let CheckedStmt::If {
                condition,
                then_body,
                else_body,
            } = stmt
            {
                let then_returns = returns_directly(then_body);
                let else_returns = returns_directly(else_body);
                if then_returns != else_returns {
                    // One arm returns early: the continuing arm's region
                    // swallows the rest of this block, so both arms end in
                    // `return` and their results join.
                    return self.if_with_continuation(
                        f,
                        condition,
                        then_body,
                        else_body,
                        &statements[index + 1..],
                        terminator,
                    );
                }
            }
            match self.stmt(f, stmt)? {
                Flow::Next => {}
                Flow::Returned => return Ok(Flow::Returned),
            }
        }
        match terminator {
            BlockTerminator::Continue => Ok(Flow::Next),
            BlockTerminator::Return(values) => {
                if !values.is_empty() {
                    for slot in 0..self.results.len() {
                        if self.results[slot].is_none() {
                            let value = self
                                .expr(f, values.get(slot).ok_or("a `return` arity mismatch")?)
                                .map_err(|reason| reason)?;
                            self.results[slot] = value;
                        }
                    }
                } else if self.results.iter().any(|slot| slot.is_none()) {
                    // The values were installed by a terminal `if` inside this
                    // block (the checker's empty `return` marker); require them.
                    if self.results.iter().any(|slot| slot.is_none()) {
                        return Err("a `return` path does not bind every result value".into());
                    }
                }
                for (index, (_, local)) in self.inout.iter().enumerate() {
                    if self.result_states[index].is_none() {
                        let storage = self
                            .storage_of_local(*local)
                            .ok_or("an inout parameter has no storage")?;
                        self.result_states[index] = Some(self.builder.current_state(storage)?);
                    }
                }
                Ok(Flow::Returned)
            }
        }
    }

    fn stmt(&mut self, f: &mut Factory, stmt: &CheckedStmt) -> Result<Flow, String> {
        match stmt {
            CheckedStmt::Let {
                pattern,
                value,
                mutable,
            } => {
                let _ = mutable;
                let value = self.expr(f, value)?.expect("`let` binds a value");
                self.bind_pattern(f, pattern, value)?;
                Ok(Flow::Next)
            }
            CheckedStmt::Assign { place, op, value } => {
                self.assign(f, place, *op, value)?;
                Ok(Flow::Next)
            }
            CheckedStmt::Loop {
                kind,
                binder,
                range,
                body,
                mutation,
            } => {
                self.loop_stmt(f, *kind, *binder, range, body, mutation)?;
                Ok(Flow::Next)
            }
            CheckedStmt::If {
                condition,
                then_body,
                else_body,
            } => {
                self.if_stmt(f, condition, then_body, else_body)?;
                Ok(Flow::Next)
            }
            CheckedStmt::Evaluate(expr) => {
                let value = self.expr(f, expr)?;
                if value.is_some() {
                    return Err(format!(
                        "an expression statement must be void; this one produces {}",
                        self.builder.value_type(value.unwrap())?
                    ));
                }
                Ok(Flow::Next)
            }
        }
    }

    fn bind_pattern(
        &mut self,
        f: &mut Factory,
        pattern: &Pattern,
        value: GraphValueId,
    ) -> Result<(), String> {
        match pattern {
            Pattern::Local(id) => {
                self.locals[*id] = Some(value);
                Ok(())
            }
            Pattern::Tuple(items) => {
                let ty = self.builder.value_type(value)?;
                let ValueType::Tuple(components) = ty else {
                    return Err("a tuple pattern needs a tuple value".into());
                };
                if components.len() != items.len() {
                    return Err("a tuple pattern does not match its value".into());
                }
                for (index, item) in items.iter().enumerate() {
                    let component = self.tuple_get(f, value, index)?;
                    self.bind_pattern(f, item, component)?;
                }
                Ok(())
            }
        }
    }

    fn tuple_get(
        &mut self,
        f: &mut Factory,
        value: GraphValueId,
        index: usize,
    ) -> Result<GraphValueId, String> {
        let ty = self.builder.value_type(value)?;
        let ValueType::Tuple(components) = &ty else {
            return Err("tuple.get needs a tuple".into());
        };
        let component = components
            .as_slice()
            .get(index)
            .cloned()
            .ok_or("tuple.get is out of bounds")?;
        let spec = PrimitiveSpec {
            op: PrimitiveOp::Primitive(PrimitiveId::TupleGet(index)),
            inputs: vec![value],
            reads: Vec::new(),
            write: None,
            outputs: vec![Output::Value(component)],
            safety: Vec::new(),
            span: Span::default(),
        };
        Ok(self.add(f, spec)?.expect("tuple.get has a value"))
    }

    // -- expressions -------------------------------------------------------

    fn expr(&mut self, f: &mut Factory, e: &CheckedExpr) -> Result<Option<GraphValueId>, String> {
        match &e.kind {
            CheckedExprKind::Literal(literal) => {
                let op = match literal {
                    Literal::Int(value) => PrimitiveOp::Constant(Literal::Int(*value)),
                    Literal::Float(value) => PrimitiveOp::Constant(Literal::Float(*value)),
                    Literal::Bool(value) => PrimitiveOp::Constant(Literal::Bool(*value)),
                    Literal::ShapeParam(name) => match self.shapes.get(name) {
                        Some(ExtentExpr::Static(n)) => {
                            PrimitiveOp::Constant(Literal::Int(*n as i64))
                        }
                        Some(ExtentExpr::Runtime(id)) => PrimitiveOp::RuntimeExtent(*id),
                        _ => {
                            return Err(format!(
                                "shape parameter `{name}` is not bound at this specialization"
                            ));
                        }
                    },
                };
                let ty = self.convert_type(f, &e.ty)?;
                let spec = PrimitiveSpec {
                    op,
                    inputs: Vec::new(),
                    reads: Vec::new(),
                    write: None,
                    outputs: vec![Output::Value(ty)],
                    safety: Vec::new(),
                    span: e.span,
                };
                let out = self.add(f, spec)?;
                if let (Some(id), Literal::Int(value)) = (out, literal) {
                    self.constants.insert(id, *value);
                }
                Ok(out)
            }
            CheckedExprKind::Local(id) => self
                .locals
                .get(*id)
                .and_then(|binding| *binding)
                .map(Some)
                .ok_or_else(|| format!("local `{}` is not bound", id)),
            CheckedExprKind::Primitive { id, operands } => self.primitive(f, e, id, operands),
            CheckedExprKind::Capability { id, args } => {
                let mut inputs = Vec::new();
                let mut reads = Vec::new();
                for arg in args {
                    let value = self
                        .expr(f, arg)?
                        .expect("a capability argument is a value");
                    if let Some(storage) = self.builder.storage_of_value(value) {
                        reads.push(storage);
                    }
                    inputs.push(value);
                }
                let ty = self.convert_type(f, &e.ty)?;
                let (output, write) = self.computed_output(f, ty, e.span)?;
                let spec = PrimitiveSpec {
                    op: PrimitiveOp::Capability(id.clone()),
                    inputs,
                    reads,
                    write,
                    outputs: vec![output],
                    safety: Vec::new(),
                    span: e.span,
                };
                Ok(self.add(f, spec)?)
            }
            CheckedExprKind::Call { call, args } => self.call(f, e, call, args),
        }
    }

    fn operand_values(
        &mut self,
        f: &mut Factory,
        operands: &[CheckedExpr],
    ) -> Result<Vec<GraphValueId>, String> {
        let mut out = Vec::new();
        for operand in operands {
            out.push(
                self.expr(f, operand)?
                    .expect("a primitive operand is a value"),
            );
        }
        Ok(out)
    }

    fn operand_reads(&self, values: &[GraphValueId]) -> Vec<LogicalStorageId> {
        let mut reads = Vec::new();
        for value in values {
            if let Some(storage) = self.builder.storage_of_value(*value) {
                if !reads.contains(&storage) {
                    reads.push(storage);
                }
            }
        }
        reads
    }

    /// Allocate fresh owned storage with an initializing write, returning the
    /// backing view.
    fn fresh_owned(
        &mut self,
        f: &mut Factory,
        shape: TensorType,
        _span: Span,
    ) -> Result<LogicalViewId, String> {
        let storage = self.builder.declare_storage(
            shape.clone(),
            StorageOrigin::Owned,
            Initialization::Uninitialized,
        );
        let view =
            self.builder
                .declare_view(storage, shape, Access::Exclusive, ViewTransform::Identity);
        let _ = f;
        Ok(view)
    }

    /// Give every computed tensor an explicit logical storage identity.
    /// Scalar-like results remain SSA values; structural tuple operations are
    /// handled separately because they preserve their component transports.
    fn computed_output(
        &mut self,
        f: &mut Factory,
        ty: ValueType,
        span: Span,
    ) -> Result<(Output, Option<WriteEffect>), String> {
        let ValueType::Tensor(shape) = ty else {
            return Ok((Output::Value(ty), None));
        };
        let view = self.fresh_owned(f, shape.clone(), span)?;
        let storage = self.builder.view(view).storage;
        Ok((
            Output::View(view),
            Some(WriteEffect {
                storage,
                coverage: Coverage::full(shape.axes.len()),
                atomic: false,
                initializing: true,
            }),
        ))
    }

    #[allow(clippy::too_many_lines)]
    fn primitive(
        &mut self,
        f: &mut Factory,
        e: &CheckedExpr,
        id: &PrimitiveId,
        operands: &[CheckedExpr],
    ) -> Result<Option<GraphValueId>, String> {
        let mut values = self.operand_values(f, operands)?;
        let reads = self.operand_reads(&values);
        // A slice view's realized axes are built by its own arm; converting the
        // checked type up front would resolve the runtime-length atom before
        // the view that defines it exists.
        let ty = match id {
            PrimitiveId::SliceView { .. } => ValueType::Void,
            _ => self.convert_type(f, &e.ty)?,
        };
        let span = e.span;
        match id {
            PrimitiveId::TuplePack => {
                let spec = PrimitiveSpec {
                    op: PrimitiveOp::Primitive(id.clone()),
                    inputs: values,
                    reads: Vec::new(),
                    write: None,
                    outputs: vec![Output::Value(ty)],
                    safety: Vec::new(),
                    span,
                };
                Ok(self.add(f, spec)?)
            }
            PrimitiveId::TupleGet(_index) => {
                let spec = PrimitiveSpec {
                    op: PrimitiveOp::Primitive(id.clone()),
                    inputs: values,
                    reads,
                    write: None,
                    outputs: vec![Output::Value(ty)],
                    safety: Vec::new(),
                    span,
                };
                Ok(self.add(f, spec)?)
            }
            PrimitiveId::RangeMake => {
                let ValueType::Range { bound } = &ty else {
                    return Err("range.make produces a range".into());
                };
                let safety = vec![SafetyObligation::RangeInBounds {
                    start: values[0],
                    end: values[1],
                    extent: bound.clone(),
                }];
                let spec = PrimitiveSpec {
                    op: PrimitiveOp::Primitive(id.clone()),
                    inputs: values,
                    reads: Vec::new(),
                    write: None,
                    outputs: vec![Output::Value(ty)],
                    safety,
                    span,
                };
                Ok(self.add(f, spec)?)
            }
            PrimitiveId::RangeStart | PrimitiveId::RangeEnd => {
                let spec = PrimitiveSpec {
                    op: PrimitiveOp::Primitive(id.clone()),
                    inputs: values,
                    reads: Vec::new(),
                    write: None,
                    outputs: vec![Output::Value(ty)],
                    safety: Vec::new(),
                    span,
                };
                Ok(self.add(f, spec)?)
            }
            PrimitiveId::Unary(_) | PrimitiveId::Cast(_) | PrimitiveId::Math(_) => {
                // A tensor-level cast of a packed operand decodes first: the
                // decoded f32 elements then cast elementwise.
                let values = if let (PrimitiveId::Cast(_), Some((first, rest))) =
                    (id, values.split_first())
                {
                    match self.builder.value_type(*first) {
                        Ok(ValueType::Tensor(source)) if matches!(source.elem, Elem::Repr(_)) => {
                            let decoded_ty = ValueType::Tensor(TensorType {
                                elem: Elem::Dtype(DType::F32),
                                axes: source.axes,
                                packed_axis: source.packed_axis,
                            });
                            let (output, write) = self.computed_output(f, decoded_ty, span)?;
                            let spec = PrimitiveSpec {
                                op: PrimitiveOp::Primitive(PrimitiveId::Decode),
                                inputs: vec![*first],
                                reads: self.operand_reads(&[*first]),
                                write,
                                outputs: vec![output],
                                safety: Vec::new(),
                                span,
                            };
                            let decoded = self
                                .add(f, spec)?
                                .ok_or("packed tensor decoding did not produce a logical value")?;
                            vec![decoded]
                                .into_iter()
                                .chain(rest.iter().copied())
                                .collect()
                        }
                        _ => vec![*first]
                            .into_iter()
                            .chain(rest.iter().copied())
                            .collect(),
                    }
                } else {
                    values
                };
                let (output, write) = self.computed_output(f, ty, span)?;
                let spec = PrimitiveSpec {
                    op: PrimitiveOp::Primitive(id.clone()),
                    inputs: values,
                    reads,
                    write,
                    outputs: vec![output],
                    safety: Vec::new(),
                    span,
                };
                Ok(self.add(f, spec)?)
            }
            PrimitiveId::Binary(op) => {
                let mut safety = Vec::new();
                match op {
                    BinaryOp::Div | BinaryOp::Rem => {
                        safety.push(SafetyObligation::DivisorNonZero { value: values[1] });
                        let signed = matches!(operands[0].ty.scalar_dtype(), Some(DType::I32));
                        if signed {
                            safety.push(SafetyObligation::SignedDivisionNoOverflow {
                                lhs: values[0],
                                rhs: values[1],
                            });
                        }
                    }
                    BinaryOp::Shl | BinaryOp::Shr => {
                        safety.push(SafetyObligation::ShiftInRange { value: values[1] });
                    }
                    _ => {}
                }
                let (output, write) = self.computed_output(f, ty, span)?;
                let spec = PrimitiveSpec {
                    op: PrimitiveOp::Primitive(id.clone()),
                    inputs: values,
                    reads,
                    write,
                    outputs: vec![output],
                    safety,
                    span,
                };
                Ok(self.add(f, spec)?)
            }
            PrimitiveId::Select => {
                let (output, write) = self.computed_output(f, ty, span)?;
                let spec = PrimitiveSpec {
                    op: PrimitiveOp::Primitive(id.clone()),
                    inputs: values,
                    reads,
                    write,
                    outputs: vec![output],
                    safety: Vec::new(),
                    span,
                };
                Ok(self.add(f, spec)?)
            }
            PrimitiveId::TensorAlloc { .. } => {
                let ValueType::Tensor(shape) = &ty else {
                    return Err("tensor.alloc produces a tensor".into());
                };
                let mut safety = Vec::new();
                if shape
                    .axes
                    .iter()
                    .any(|axis| !matches!(axis, ExtentExpr::Static(_)))
                {
                    safety.push(SafetyObligation::ShapeProductFits {
                        factors: shape.axes.clone(),
                        bits: 64,
                    });
                } else {
                    let product = shape.axes.iter().try_fold(1u128, |acc, axis| match axis {
                        ExtentExpr::Static(n) => acc.checked_mul(*n as u128),
                        _ => None,
                    });
                    if !matches!(product, Some(p) if p <= u64::MAX as u128) {
                        safety.push(SafetyObligation::ShapeProductFits {
                            factors: shape.axes.clone(),
                            bits: 64,
                        });
                    }
                }
                let view = self.fresh_owned(f, shape.clone(), span)?;
                let storage = self.builder.view(view).storage;
                let spec = PrimitiveSpec {
                    op: PrimitiveOp::Primitive(id.clone()),
                    inputs: values,
                    reads: Vec::new(),
                    write: Some(WriteEffect {
                        storage,
                        coverage: Coverage::none(shape.axes.len()),
                        atomic: false,
                        initializing: true,
                    }),
                    outputs: vec![Output::View(view)],
                    safety,
                    span,
                };
                Ok(self.add(f, spec)?)
            }
            PrimitiveId::Fill { .. } => {
                let ValueType::Tensor(shape) = &ty else {
                    return Err("tensor.fill produces a tensor".into());
                };
                let view = self.fresh_owned(f, shape.clone(), span)?;
                let storage = self.builder.view(view).storage;
                let spec = PrimitiveSpec {
                    op: PrimitiveOp::Primitive(id.clone()),
                    inputs: values,
                    reads,
                    write: Some(WriteEffect {
                        storage,
                        coverage: Coverage::full(shape.axes.len()),
                        atomic: false,
                        initializing: true,
                    }),
                    outputs: vec![Output::View(view)],
                    safety: Vec::new(),
                    span,
                };
                Ok(self.add(f, spec)?)
            }
            PrimitiveId::Materialize
            | PrimitiveId::Clone
            | PrimitiveId::Load
            | PrimitiveId::Decode
            | PrimitiveId::PackedRead(_) => {
                let ValueType::Tensor(shape) = &ty else {
                    return Err("a tensor-producing primitive produces a tensor".into());
                };
                let access = if matches!(id, PrimitiveId::PackedRead(_)) {
                    Access::Shared
                } else {
                    Access::Exclusive
                };
                let storage = self.builder.declare_storage(
                    shape.clone(),
                    StorageOrigin::Owned,
                    Initialization::Uninitialized,
                );
                let view = self.builder.declare_view(
                    storage,
                    shape.clone(),
                    access,
                    ViewTransform::Identity,
                );
                let spec = PrimitiveSpec {
                    op: PrimitiveOp::Primitive(id.clone()),
                    inputs: values,
                    reads,
                    write: Some(WriteEffect {
                        storage,
                        coverage: Coverage::full(shape.axes.len()),
                        atomic: false,
                        initializing: true,
                    }),
                    outputs: vec![Output::View(view)],
                    safety: Vec::new(),
                    span,
                };
                Ok(self.add(f, spec)?)
            }
            PrimitiveId::Transpose => {
                let view = match self.builder.view_of_value(values[0]) {
                    Some(view) => view,
                    None => {
                        values[0] = self.materialize_value(f, values[0], span)?;
                        self.builder
                            .view_of_value(values[0])
                            .expect("materialize produces a view")
                    }
                };
                let source = self.builder.view(view);
                let storage = source.storage;
                let access = source.access;
                let rank = source.shape.rank();
                let permutation: Vec<u32> = (0..rank as u32).rev().collect();
                let new_view = self.builder.declare_view(
                    storage,
                    match &ty {
                        ValueType::Tensor(s) => s.clone(),
                        _ => return Err("transpose produces a tensor".into()),
                    },
                    access,
                    ViewTransform::Transpose { permutation },
                );
                let spec = PrimitiveSpec {
                    op: PrimitiveOp::Primitive(id.clone()),
                    inputs: values,
                    reads: Vec::new(),
                    write: None,
                    outputs: vec![Output::View(new_view)],
                    safety: Vec::new(),
                    span,
                };
                Ok(self.add(f, spec)?)
            }
            PrimitiveId::Reshape => {
                let view = match self.builder.view_of_value(values[0]) {
                    Some(view) => view,
                    None => {
                        values[0] = self.materialize_value(f, values[0], span)?;
                        self.builder
                            .view_of_value(values[0])
                            .expect("materialize produces a view")
                    }
                };
                let source = self.builder.view(view);
                let storage = source.storage;
                let access = source.access;
                let source_shape = source.shape.axes.clone();
                let new_view = self.builder.declare_view(
                    storage,
                    match &ty {
                        ValueType::Tensor(s) => s.clone(),
                        _ => return Err("reshape produces a tensor".into()),
                    },
                    access,
                    ViewTransform::Reshape { source_shape },
                );
                let spec = PrimitiveSpec {
                    op: PrimitiveOp::Primitive(id.clone()),
                    inputs: values,
                    reads: Vec::new(),
                    write: None,
                    outputs: vec![Output::View(new_view)],
                    safety: Vec::new(),
                    span,
                };
                Ok(self.add(f, spec)?)
            }
            PrimitiveId::SliceView { indices } => {
                let view = match self.builder.view_of_value(values[0]) {
                    Some(view) => view,
                    None => {
                        values[0] = self.materialize_value(f, values[0], span)?;
                        self.builder
                            .view_of_value(values[0])
                            .expect("materialize produces a view")
                    }
                };
                let source = self.builder.view(view);
                let storage = source.storage;
                let access = source.access;
                let base_axes = source.shape.axes.clone();
                let mut axes = Vec::new();
                let mut safety = Vec::new();
                let mut cursor = 1; // operands after the base
                for slot in indices {
                    match slot {
                        crate::intrinsics::IndexSlot::Point => {
                            axes.push(SliceAxis::Point(values[cursor]));
                            safety.push(SafetyObligation::IndexInBounds {
                                index: values[cursor],
                                extent: base_axes[axes.len() - 1].clone(),
                            });
                            cursor += 1;
                        }
                        crate::intrinsics::IndexSlot::Range { start, end } => {
                            let start_value = if *start {
                                cursor += 1;
                                Some(values[cursor - 1])
                            } else {
                                None
                            };
                            let end_value = if *end {
                                cursor += 1;
                                Some(values[cursor - 1])
                            } else {
                                None
                            };
                            if let (Some(s), Some(e)) = (start_value, end_value) {
                                safety.push(SafetyObligation::RangeInBounds {
                                    start: s,
                                    end: e,
                                    extent: base_axes[axes.len()].clone(),
                                });
                            }
                            axes.push(SliceAxis::Range {
                                start: start_value,
                                end: end_value,
                            });
                        }
                    }
                }
                // View axes: points drop the axis, ranges keep their
                // realized length. A runtime-bounded range realizes to a
                // runtime extent of `end - start`; the checker's `@dyn`
                // atom for that axis maps to the same extent so later type
                // conversions of this expression's type resolve.
                let mut shape_axes: Vec<ExtentExpr> = Vec::new();
                let mut axis_cursor = 0usize;
                let mut range_values = axes.iter().filter_map(|axis| match axis {
                    SliceAxis::Range { start, end } => Some((*start, *end)),
                    SliceAxis::Point(_) | SliceAxis::Full => None,
                });
                for slot in indices {
                    match slot {
                        crate::intrinsics::IndexSlot::Point => {
                            axis_cursor += 1;
                        }
                        crate::intrinsics::IndexSlot::Range { .. } => {
                            let (start, end) = range_values.next().unwrap_or((None, None));
                            let base_axis = base_axes[axis_cursor].clone();
                            let extent = match (start, end) {
                                (None, None) => base_axis,
                                (Some(start), Some(end)) => {
                                    let expr = RuntimeScalarExpr::Sub(
                                        Box::new(RuntimeScalarExpr::Value(end)),
                                        Box::new(RuntimeScalarExpr::Value(start)),
                                    );
                                    let capacity = match &base_axis {
                                        ExtentExpr::Static(value) => *value,
                                        _ => u64::MAX,
                                    };
                                    let id = f.runtime_extent(expr, capacity);
                                    if let ValueType::Tensor(source) = &e.ty {
                                        if let Some(ExtentExpr::Sym(sym)) =
                                            source.axes.get(shape_axes.len())
                                        {
                                            if sym.atoms().iter().all(|atom| matches!(atom, Atom::Param(name) if name.starts_with('@')))
                                            {
                                                self.extent_memo
                                                    .insert(self.sym_key(sym)?, id);
                                            }
                                        }
                                    }
                                    ExtentExpr::Runtime(id)
                                }
                                _ => {
                                    return Err(
                                        "a one-sided runtime slice bound is not representable"
                                            .into(),
                                    );
                                }
                            };
                            shape_axes.push(extent);
                            axis_cursor += 1;
                        }
                    }
                }
                while axis_cursor < base_axes.len() {
                    shape_axes.push(base_axes[axis_cursor].clone());
                    axis_cursor += 1;
                }
                let source_elem = source.shape.elem.clone();
                let packed_axis = source.shape.packed_axis;
                let new_view = self.builder.declare_view(
                    storage,
                    TensorType {
                        axes: shape_axes.clone(),
                        elem: source_elem.clone(),
                        packed_axis,
                    },
                    access,
                    ViewTransform::Slice { axes },
                );
                let spec = PrimitiveSpec {
                    op: PrimitiveOp::Primitive(id.clone()),
                    inputs: values,
                    reads: vec![storage],
                    write: None,
                    outputs: vec![Output::View(new_view)],
                    safety,
                    span,
                };
                Ok(self.add(f, spec)?)
            }
            PrimitiveId::ElementRead { arity } => {
                let base = match self.builder.view_of_value(values[0]) {
                    Some(_) => values[0],
                    // A computed dense value is explicitly stored before a
                    // point read.
                    None => self.materialize_value(f, values[0], span)?,
                };
                values[0] = base;
                let view = self
                    .builder
                    .view_of_value(base)
                    .ok_or("element read needs a storage-backed view")?;
                let base = self.builder.view(view);
                let storage = base.storage;
                let indices: Vec<GraphValueId> = values[1..1 + arity].to_vec();
                let mut safety = Vec::new();
                for (axis, index) in indices.iter().enumerate() {
                    safety.push(SafetyObligation::IndexInBounds {
                        index: *index,
                        extent: base.shape.axes[axis].clone(),
                    });
                }
                let spec = PrimitiveSpec {
                    op: PrimitiveOp::Primitive(id.clone()),
                    inputs: values,
                    reads: vec![storage],
                    write: None,
                    outputs: vec![Output::Value(ty)],
                    safety,
                    span,
                };
                Ok(self.add(f, spec)?)
            }
            PrimitiveId::ElementWrite { arity } => {
                // Reached from assignments (the normal path) or a checked
                // expression position.
                let view = self
                    .builder
                    .view_of_value(values[0])
                    .ok_or("element write needs a storage-backed view")?;
                self.element_write(
                    f,
                    values[0],
                    view,
                    &values[1..1 + arity],
                    values[values.len() - 1],
                    span,
                )?;
                Ok(None)
            }
            PrimitiveId::CopyInto => {
                let view = self
                    .builder
                    .view_of_value(values[0])
                    .ok_or("copy.into needs a storage-backed destination")?;
                self.copy_into(f, values[0], view, values[1], span)?;
                Ok(None)
            }
            PrimitiveId::Extent { axis } | PrimitiveId::ValidExtent { axis } => {
                let _ = axis;
                let spec = PrimitiveSpec {
                    op: PrimitiveOp::Primitive(id.clone()),
                    inputs: values,
                    reads,
                    write: None,
                    outputs: vec![Output::Value(ty)],
                    safety: Vec::new(),
                    span,
                };
                Ok(self.add(f, spec)?)
            }
            PrimitiveId::Atomic { op, arity } => {
                let view = self
                    .builder
                    .view_of_value(values[0])
                    .ok_or("atomic update needs a storage-backed view")?;
                let base = self.builder.view(view);
                let storage = base.storage;
                let indices: Vec<GraphValueId> = values[1..1 + arity].to_vec();
                let mut safety = Vec::new();
                for (axis, index) in indices.iter().enumerate() {
                    safety.push(SafetyObligation::IndexInBounds {
                        index: *index,
                        extent: base.shape.axes[axis].clone(),
                    });
                }
                let coverage = self.write_coverage(&base.shape.axes, &indices);
                let dtype = base
                    .shape
                    .elem
                    .read_dtype()
                    .ok_or("atomic update needs a dense element")?;
                self.record_write(storage);
                self.record_atomic(
                    storage,
                    AtomicOperation {
                        op: *op,
                        dtype,
                        arity: *arity,
                    },
                );
                let spec = PrimitiveSpec {
                    op: PrimitiveOp::Primitive(id.clone()),
                    inputs: values,
                    reads: Vec::new(),
                    write: Some(WriteEffect {
                        storage,
                        coverage,
                        atomic: true,
                        initializing: false,
                    }),
                    outputs: Vec::new(),
                    safety,
                    span,
                };
                self.add(f, spec)?;
                Ok(None)
            }
            PrimitiveId::Reduce {
                op,
                axis,
                unordered,
            } => {
                if self.builder.view_of_value(values[0]).is_none() {
                    values[0] = self.materialize_value(f, values[0], span)?;
                }
                let operand_elem = match &operands[0].ty {
                    ValueType::Tensor(s) => s.elem.clone(),
                    _ => return Err("reduce needs a tensor operand".into()),
                };
                let elem = self.convert_elem(&operand_elem)?;
                let input = match &elem {
                    Elem::Dtype(d) => *d,
                    _ => return Err("reduce needs a dense element".into()),
                };
                let accumulator = accumulator_dtype(*op, input);
                let order = if *unordered {
                    ReductionOrder::Unordered
                } else {
                    ReductionOrder::Ascending
                };
                let value = self.builder.add_reduction(
                    values[0],
                    *axis,
                    *op,
                    order,
                    accumulator,
                    ty.clone(),
                    span,
                )?;
                Ok(Some(value))
            }
        }
    }
}

impl<'w> Work<'w> {
    // -- writes and coverage ------------------------------------------------

    /// Structural coverage proof of a point write: an axis is covered when
    /// the index is an enclosing loop's binder, the write is unconditional
    /// within that loop, and the loop's range covers the axis (the checker's
    /// disjoint-write proof covers independent loops; ascending order covers
    /// ordered ones). Zero-sized axes are trivially covered.
    fn write_coverage(&self, axes: &[ExtentExpr], indices: &[GraphValueId]) -> Coverage {
        let mut covered = axes
            .iter()
            .map(|axis| *axis == ExtentExpr::Static(0))
            .collect::<Vec<bool>>();
        for (axis, index) in indices.iter().enumerate() {
            if let Some(info) = self.loops.iter().rev().find(|info| info.binder == *index) {
                if self.if_depth == info.if_depth
                    && info.start == ExtentExpr::Static(0)
                    && info.end == axes[axis]
                {
                    covered[axis] = true;
                }
            }
        }
        Coverage { axes: covered }
    }

    /// Structural coverage of writing through one view: identity/reshape/
    /// transpose views are whole-storage writes; slice views cover the axes
    /// their static bounds provably span.
    fn transform_coverage(&self, view: LogicalViewId) -> Coverage {
        let view = self.builder.view(view);
        let rank = self.builder.storage(view.storage).shape.axes.len();
        match &view.transform {
            ViewTransform::Identity
            | ViewTransform::Reshape { .. }
            | ViewTransform::Transpose { .. } => Coverage::full(rank),
            ViewTransform::Slice { axes } => {
                let mut covered = Vec::with_capacity(rank);
                for slot in axes {
                    let axis = covered.len();
                    let extent = self.builder.storage(view.storage).shape.axes[axis].clone();
                    let covers = match slot {
                        SliceAxis::Full => true,
                        SliceAxis::Point(_) => false,
                        SliceAxis::Range { start, end } => {
                            let start_ok = start
                                .map(|v| self.constants.get(&v) == Some(&0))
                                .unwrap_or(true);
                            let end_ok = end
                                .map(|v| match &extent {
                                    ExtentExpr::Static(n) => {
                                        self.constants.get(&v) == Some(&(*n as i64))
                                    }
                                    _ => false,
                                })
                                .unwrap_or(true);
                            start_ok && end_ok
                        }
                    };
                    covered.push(covers);
                }
                while covered.len() < rank {
                    covered.push(true);
                }
                Coverage { axes: covered }
            }
        }
    }

    /// Range-safety obligations of writing through a slice view.
    fn transform_safety(&self, view: LogicalViewId) -> Vec<SafetyObligation> {
        let logical = self.builder.view(view);
        let storage = self.builder.storage(logical.storage);
        match &logical.transform {
            ViewTransform::Slice { axes } => {
                let mut safety = Vec::new();
                for (axis, slot) in axes.iter().enumerate() {
                    if let SliceAxis::Range {
                        start: Some(start),
                        end: Some(end),
                    } = slot
                    {
                        safety.push(SafetyObligation::RangeInBounds {
                            start: *start,
                            end: *end,
                            extent: storage.shape.axes[axis].clone(),
                        });
                    }
                }
                safety
            }
            _ => Vec::new(),
        }
    }

    /// Record a write for every enclosing loop that captured the storage.
    fn record_write(&mut self, storage: LogicalStorageId) {
        for info in self.loops.iter_mut() {
            if info.captured.contains(&storage) {
                info.writes.entry(storage).or_default().plain = true;
            }
        }
    }

    /// Record an admitted atomic operation for every enclosing loop that
    /// captured the storage.
    fn record_atomic(&mut self, storage: LogicalStorageId, operation: AtomicOperation) {
        for info in self.loops.iter_mut() {
            if info.captured.contains(&storage) {
                let entry = info.writes.entry(storage).or_default();
                entry.atomic = true;
                if !info.atomics.contains(&operation) {
                    info.atomics.push(operation.clone());
                }
            }
        }
    }

    fn element_write(
        &mut self,
        f: &mut Factory,
        base_value: GraphValueId,
        view: LogicalViewId,
        indices: &[GraphValueId],
        value: GraphValueId,
        span: Span,
    ) -> Result<(), String> {
        let base = self.builder.view(view);
        let storage = base.storage;
        let mut safety = Vec::new();
        for (axis, index) in indices.iter().enumerate() {
            safety.push(SafetyObligation::IndexInBounds {
                index: *index,
                extent: base.shape.axes[axis].clone(),
            });
        }
        let coverage = self.write_coverage(&base.shape.axes, indices);
        self.record_write(storage);
        let mut inputs = vec![base_value];
        inputs.extend(indices.iter().copied());
        inputs.push(value);
        let spec = PrimitiveSpec {
            op: PrimitiveOp::Primitive(PrimitiveId::ElementWrite {
                arity: indices.len(),
            }),
            inputs,
            reads: Vec::new(),
            write: Some(WriteEffect {
                storage,
                coverage,
                atomic: false,
                initializing: false,
            }),
            outputs: Vec::new(),
            safety,
            span,
        };
        self.add(f, spec)?;
        Ok(())
    }

    fn copy_into(
        &mut self,
        f: &mut Factory,
        dst_value: GraphValueId,
        view: LogicalViewId,
        src_value: GraphValueId,
        span: Span,
    ) -> Result<(), String> {
        let storage = self.builder.view(view).storage;
        let coverage = self.transform_coverage(view);
        let safety = self.transform_safety(view);
        let reads = self
            .builder
            .storage_of_value(src_value)
            .into_iter()
            .collect::<Vec<_>>();
        self.record_write(storage);
        let spec = PrimitiveSpec {
            op: PrimitiveOp::Primitive(PrimitiveId::CopyInto),
            inputs: vec![dst_value, src_value],
            reads,
            write: Some(WriteEffect {
                storage,
                coverage,
                atomic: false,
                initializing: false,
            }),
            outputs: Vec::new(),
            safety,
            span,
        };
        self.add(f, spec)?;
        Ok(())
    }

    fn element_read(
        &mut self,
        f: &mut Factory,
        base_value: GraphValueId,
        view: LogicalViewId,
        indices: &[GraphValueId],
        span: Span,
    ) -> Result<GraphValueId, String> {
        let base = self.builder.view(view);
        let storage = base.storage;
        let mut safety = Vec::new();
        for (axis, index) in indices.iter().enumerate() {
            safety.push(SafetyObligation::IndexInBounds {
                index: *index,
                extent: base.shape.axes[axis].clone(),
            });
        }
        let dtype = base
            .shape
            .elem
            .read_dtype()
            .ok_or("an element read needs a readable element")?;
        let mut inputs = vec![base_value];
        inputs.extend(indices.iter().copied());
        let spec = PrimitiveSpec {
            op: PrimitiveOp::Primitive(PrimitiveId::ElementRead {
                arity: indices.len(),
            }),
            inputs,
            reads: vec![storage],
            write: None,
            outputs: vec![Output::Value(ValueType::Scalar(dtype))],
            safety,
            span,
        };
        Ok(self
            .add(f, spec)?
            .expect("an element read produces a scalar"))
    }

    /// Materialize one computed or borrowed tensor value into fresh owned
    /// storage.
    fn materialize_value(
        &mut self,
        f: &mut Factory,
        value: GraphValueId,
        span: Span,
    ) -> Result<GraphValueId, String> {
        let ty = self.builder.value_type(value)?;
        let ValueType::Tensor(shape) = &ty else {
            return Err("materialize needs a tensor".into());
        };
        let storage = self.builder.declare_storage(
            shape.clone(),
            StorageOrigin::Owned,
            Initialization::Uninitialized,
        );
        let view = self.builder.declare_view(
            storage,
            shape.clone(),
            Access::Exclusive,
            ViewTransform::Identity,
        );
        let reads = self
            .builder
            .storage_of_value(value)
            .into_iter()
            .collect::<Vec<_>>();
        let spec = PrimitiveSpec {
            op: PrimitiveOp::Primitive(PrimitiveId::Materialize),
            inputs: vec![value],
            reads,
            write: Some(WriteEffect {
                storage,
                coverage: Coverage::full(shape.axes.len()),
                atomic: false,
                initializing: true,
            }),
            outputs: vec![Output::View(view)],
            safety: Vec::new(),
            span,
        };
        Ok(self.add(f, spec)?.expect("materialize produces a value"))
    }

    /// At the entry, every tensor leaf of a result becomes compiler-owned
    /// storage: borrowed or computed values are materialized; tuples are
    /// repacked from their materialized components.
    fn materialize_result_leaf(
        &mut self,
        f: &mut Factory,
        value: GraphValueId,
        ty: &ValueType,
        span: Span,
    ) -> Result<GraphValueId, String> {
        match ty {
            ValueType::Tensor(_) => self.materialize_value(f, value, span),
            ValueType::Tuple(items) => {
                let mut components = Vec::new();
                for (index, item) in items.iter().enumerate() {
                    let component = self.tuple_get(f, value, index)?;
                    components.push(self.materialize_result_leaf(f, component, item, span)?);
                }
                let tys = components
                    .iter()
                    .map(|id| self.builder.value_type(*id))
                    .collect::<Result<_, _>>()?;
                let packed = ValueType::Tuple(NonEmpty::new(tys).ok_or("a tuple result is empty")?);
                let spec = PrimitiveSpec {
                    op: PrimitiveOp::Primitive(PrimitiveId::TuplePack),
                    inputs: components,
                    reads: Vec::new(),
                    write: None,
                    outputs: vec![Output::Value(packed)],
                    safety: Vec::new(),
                    span,
                };
                Ok(self.add(f, spec)?.expect("tuple pack produces a value"))
            }
            _ => Ok(value),
        }
    }

    // -- assignment ---------------------------------------------------------

    fn assign(
        &mut self,
        f: &mut Factory,
        place: &CheckedPlace,
        op: AssignOp,
        value_expr: &CheckedExpr,
    ) -> Result<(), String> {
        let span = value_expr.span;
        let value = self
            .expr(f, value_expr)?
            .expect("an assignment value is a value");
        self.assign_place(f, place, op, value, span)
    }

    fn assign_place(
        &mut self,
        f: &mut Factory,
        place: &CheckedPlace,
        op: AssignOp,
        value: GraphValueId,
        span: Span,
    ) -> Result<(), String> {
        match place {
            CheckedPlace::Local { root } => {
                let current = self.locals[*root].ok_or("an assignment target is not bound")?;
                let current_ty = self.builder.value_type(current)?;
                let value_ty = self.builder.value_type(value)?;
                if matches!(current_ty, ValueType::Tensor(_)) {
                    if current_ty == value_ty {
                        // Whole-tensor assignment writes into the target's
                        // storage (a compound op computes elementwise first).
                        let view = self
                            .builder
                            .view_of_value(current)
                            .ok_or("a whole-tensor assignment needs a storage-backed target")?;
                        if op == AssignOp::Assign {
                            self.copy_into(f, current, view, value, span)?;
                        } else {
                            let computed =
                                self.binary_assign_op(f, op, current, value, &current_ty, span)?;
                            self.copy_into(f, current, view, computed, span)?;
                        }
                    } else {
                        if op != AssignOp::Assign {
                            return Err(
                                "a compound assignment cannot change the target's type".into()
                            );
                        }
                        self.locals[*root] = Some(value);
                    }
                } else if op == AssignOp::Assign {
                    self.locals[*root] = Some(value);
                } else {
                    let computed =
                        self.binary_assign_op(f, op, current, value, &current_ty, span)?;
                    self.locals[*root] = Some(computed);
                }
                Ok(())
            }
            CheckedPlace::Element { root, indices } => {
                let current = self.locals[*root].ok_or("an assignment target is not bound")?;
                // A computed target is explicitly stored first; the local's
                // current origin becomes that storage so later reads see
                // these writes.
                let current = match self.builder.view_of_value(current) {
                    Some(_) => current,
                    None => {
                        let materialized = self.materialize_value(f, current, span)?;
                        self.locals[*root] = Some(materialized);
                        materialized
                    }
                };
                let view = self
                    .builder
                    .view_of_value(current)
                    .ok_or("an element assignment needs a storage-backed target")?;
                // Convert indices: values for points, optional bounds for
                // ranges.
                let mut point_indices = Vec::new();
                let mut slots = Vec::new();
                let mut slot_args = Vec::new();
                for index in indices {
                    match index {
                        CheckedIndex::Point(p) => {
                            let value = self.expr(f, p)?.expect("an index is a value");
                            point_indices.push(value);
                            slots.push(crate::intrinsics::IndexSlot::Point);
                            slot_args.push(SlotArg::Point(value));
                        }
                        CheckedIndex::Range { start, end } => {
                            let start_value = match start {
                                Some(e) => Some(self.expr(f, e)?.expect("a bound is a value")),
                                None => None,
                            };
                            let end_value = match end {
                                Some(e) => Some(self.expr(f, e)?.expect("a bound is a value")),
                                None => None,
                            };
                            slots.push(crate::intrinsics::IndexSlot::Range {
                                start: start_value.is_some(),
                                end: end_value.is_some(),
                            });
                            slot_args.push(SlotArg::Range {
                                start: start_value,
                                end: end_value,
                            });
                        }
                    }
                }
                let all_points = slot_args.iter().all(|arg| matches!(arg, SlotArg::Point(_)));
                if all_points {
                    if op == AssignOp::Assign {
                        self.element_write(f, current, view, &point_indices, value, span)?;
                    } else {
                        let read = self.element_read(f, current, view, &point_indices, span)?;
                        let read_ty = self.builder.value_type(read)?;
                        let computed = self.binary_assign_op(f, op, read, value, &read_ty, span)?;
                        self.element_write(f, current, view, &point_indices, computed, span)?;
                    }
                } else {
                    if op != AssignOp::Assign {
                        return Err("a compound assignment cannot target a slice".into());
                    }
                    // Build the destination view, then copy the source into it.
                    let sliced = self.make_slice_view(f, current, view, slots, slot_args, span)?;
                    self.copy_into(f, sliced.0, sliced.1, value, span)?;
                }
                Ok(())
            }
            CheckedPlace::Tuple(places) => {
                for (index, place) in places.iter().enumerate() {
                    let component = self.tuple_get(f, value, index)?;
                    self.assign_place(f, place, op, component, span)?;
                }
                Ok(())
            }
        }
    }

    fn binary_assign_op(
        &mut self,
        f: &mut Factory,
        op: AssignOp,
        lhs: GraphValueId,
        rhs: GraphValueId,
        ty: &ValueType,
        span: Span,
    ) -> Result<GraphValueId, String> {
        let binary = match op {
            AssignOp::Assign => return Err("a plain assignment is not compound".into()),
            AssignOp::Add => BinaryOp::Add,
            AssignOp::Sub => BinaryOp::Sub,
            AssignOp::Mul => BinaryOp::Mul,
        };
        let (output, write) = self.computed_output(f, ty.clone(), span)?;
        let spec = PrimitiveSpec {
            op: PrimitiveOp::Primitive(PrimitiveId::Binary(binary)),
            inputs: vec![lhs, rhs],
            reads: self.operand_reads(&[lhs, rhs]),
            write,
            outputs: vec![output],
            safety: Vec::new(),
            span,
        };
        Ok(self
            .add(f, spec)?
            .expect("a compound assignment computes a value"))
    }

    /// Construct the destination view of a slice write.
    fn make_slice_view(
        &mut self,
        f: &mut Factory,
        base_value: GraphValueId,
        base_view: LogicalViewId,
        slots: Vec<crate::intrinsics::IndexSlot>,
        args: Vec<SlotArg>,
        span: Span,
    ) -> Result<(GraphValueId, LogicalViewId), String> {
        let base = self.builder.view(base_view);
        let storage = base.storage;
        let access = base.access;
        let base_axes = base.shape.axes.clone();
        let mut inputs = vec![base_value];
        let mut axes = Vec::new();
        let mut safety = Vec::new();
        for (axis, (slot, arg)) in slots.iter().zip(&args).enumerate() {
            match (slot, arg) {
                (crate::intrinsics::IndexSlot::Point, SlotArg::Point(value)) => {
                    inputs.push(*value);
                    axes.push(SliceAxis::Point(*value));
                    safety.push(SafetyObligation::IndexInBounds {
                        index: *value,
                        extent: base_axes[axis].clone(),
                    });
                }
                (
                    crate::intrinsics::IndexSlot::Range { start, end },
                    SlotArg::Range { start: s, end: e },
                ) => {
                    if let (true, Some(value)) = (start, s) {
                        inputs.push(*value);
                    }
                    if let (true, Some(value)) = (end, e) {
                        inputs.push(*value);
                    }
                    if let (Some(start), Some(end)) = (s, e) {
                        safety.push(SafetyObligation::RangeInBounds {
                            start: *start,
                            end: *end,
                            extent: base_axes[axis].clone(),
                        });
                    }
                    axes.push(SliceAxis::Range { start: *s, end: *e });
                }
                _ => return Err("a slice slot's arguments do not match its shape".into()),
            }
        }
        // The result type keeps the base element; its axes shrink per the
        // checker's recorded expression type, which the caller does not pass.
        // Derive it structurally: points drop the axis, ranges keep it (their
        // runtime extent), trailing axes are kept.
        let mut shape_axes = Vec::new();
        let mut axis_cursor = 0;
        for slot in &slots {
            match slot {
                crate::intrinsics::IndexSlot::Point => {
                    axis_cursor += 1;
                }
                crate::intrinsics::IndexSlot::Range { .. } => {
                    shape_axes.push(base_axes[axis_cursor].clone());
                    axis_cursor += 1;
                }
            }
        }
        while axis_cursor < base_axes.len() {
            shape_axes.push(base_axes[axis_cursor].clone());
            axis_cursor += 1;
        }
        let elem = base.shape.elem.clone();
        let packed_axis = base.shape.packed_axis;
        let shape = TensorType {
            axes: shape_axes,
            elem,
            packed_axis,
        };
        let view = self
            .builder
            .declare_view(storage, shape, access, ViewTransform::Slice { axes });
        let spec = PrimitiveSpec {
            op: PrimitiveOp::Primitive(PrimitiveId::SliceView { indices: slots }),
            inputs,
            reads: vec![storage],
            write: None,
            outputs: vec![Output::View(view)],
            safety,
            span,
        };
        let value = self.add(f, spec)?.expect("a slice view produces a value");
        Ok((value, view))
    }

    // -- loops --------------------------------------------------------------

    fn loop_stmt(
        &mut self,
        f: &mut Factory,
        kind: LoopKind,
        binder: usize,
        range: &CheckedRange,
        body: &CheckedBlock,
        mutation: &LoopMutationSummary,
    ) -> Result<(), String> {
        let span = range.start.span;
        let start = self
            .expr(f, &range.start)?
            .expect("a range bound is a value");
        let end = self.expr(f, &range.end)?.expect("a range bound is a value");
        let binder_ty = self.convert_type(f, &self.local_tys[binder].ty)?;
        let ValueType::Index { bound } = &binder_ty else {
            return Err("a loop binder is an index".into());
        };
        let range_logical = LogicalRange {
            start,
            end,
            bound: bound.clone(),
        };
        let binder_id = self.builder.fresh_value(
            ValueType::Index {
                bound: bound.clone(),
            },
            None,
        )?;

        // Symbolic range endpoints, for coverage proofs.
        let ctx_start = match &range.start.sym {
            Some(sym) => self.resolve_sym(f, sym)?,
            None => ExtentExpr::Static(0),
        };
        let ctx_end = match &range.end.sym {
            Some(sym) => self.resolve_sym(f, sym)?,
            None => bound.clone(),
        };

        let mut free = free_locals(body);
        free.remove(&binder);
        let rebound = rebound_roots(body);

        // Storages threaded through the body region: ordered loops carry the
        // captured tensor locals the checker marked; independent loops thread
        // every captured storage the body writes (their visits join).
        let mut captured_storages = BTreeSet::new();
        if kind == LoopKind::Ordered {
            for local in &mutation.carried {
                if let Some(storage) = self.storage_of_local(*local) {
                    captured_storages.insert(storage);
                }
            }
        } else {
            // The checker's summary is exact; the syntactic write scan
            // over-approximates call arguments borrowed into callees.
            for local in mutation.carried.iter().chain(&mutation.atomics) {
                if free.contains(local) {
                    if let Some(storage) = self.storage_of_local(*local) {
                        captured_storages.insert(storage);
                    }
                }
            }
        }

        // Carried slots (ordered loops only): every changed captured local.
        let mut carried_value_locals = Vec::new();
        let mut carried_state_locals = Vec::new();
        if kind == LoopKind::Ordered {
            for local in &mutation.carried {
                let binding = self
                    .locals
                    .get(*local)
                    .and_then(|binding| *binding)
                    .ok_or("a carried local is not bound")?;
                if self.builder.view_of_value(binding).is_some() {
                    if rebound.contains(local) {
                        return Err(format!(
                            "captured tensor local {} is rebound to a different value inside an ordered loop; this is not representable",
                            self.local_tys[*local].name
                        ));
                    }
                    carried_state_locals.push(*local);
                } else {
                    carried_value_locals.push(*local);
                }
            }
        } else {
            // Independent loops admit no scalar or owned carry: the checker's
            // exact written-capture set, not the over-approximating scan.
            for local in mutation.carried.iter().chain(&mutation.atomics) {
                if free.contains(local) {
                    let binding = self
                        .locals
                        .get(*local)
                        .and_then(|binding| *binding)
                        .ok_or("a written local is not bound")?;
                    if self.builder.view_of_value(binding).is_none()
                        && !matches!(self.builder.value_type(binding)?, ValueType::Tensor(_))
                    {
                        return Err(format!(
                            "independent loop writes captured scalar local {}",
                            self.local_tys[*local].name
                        ));
                    }
                }
            }
        }

        // Body region parameters: binder, every captured value (carried
        // tensors' views included — the view itself never changes), then the
        // carried/joined states.
        let mut params = Vec::new();
        let mut param_ids = BTreeMap::new();
        params.push(RegionParameter::Value {
            id: binder_id,
            ty: binder_ty,
        });
        let mut invariant_values = Vec::new();
        for local in &free {
            let binding = self
                .locals
                .get(*local)
                .and_then(|binding| *binding)
                .ok_or("a captured local is not bound")?;
            let ty = self.builder.value_type(binding)?;
            let view = self.builder.view_of_value(binding);
            let param = self.builder.fresh_value(ty.clone(), view)?;
            param_ids.insert(*local, param);
            params.push(RegionParameter::Value { id: param, ty });
            if !carried_value_locals.contains(local) {
                invariant_values.push(binding);
            }
        }
        let mut state_params = Vec::new();
        for local in carried_state_locals.iter().copied() {
            let storage = self
                .storage_of_local(local)
                .ok_or("a carried tensor local has no storage")?;
            let token = self.builder.fresh_state();
            self.builder.bind_state(token, storage);
            params.push(RegionParameter::State { id: token, storage });
            state_params.push((local, storage, token));
        }
        if kind == LoopKind::Independent {
            for storage in &captured_storages {
                let token = self.builder.fresh_state();
                self.builder.bind_state(token, *storage);
                params.push(RegionParameter::State {
                    id: token,
                    storage: *storage,
                });
                state_params.push((usize::MAX, *storage, token));
            }
        }

        self.loops.push(LoopInfo {
            binder: binder_id,
            start: ctx_start,
            end: ctx_end,
            if_depth: self.if_depth,
            captured: captured_storages.clone(),
            writes: BTreeMap::new(),
            atomics: Vec::new(),
        });

        // Pre-loop current states, captured before the body mutates them.
        let mut pre_states: BTreeMap<LogicalStorageId, StateTokenId> = BTreeMap::new();
        for storage in &captured_storages {
            pre_states.insert(*storage, self.builder.current_state(*storage)?);
        }

        // Open the body region with the assembled parameters, then build it
        // with captured locals rebound to the parameters.
        self.builder.begin_region(params)?;
        let saved_locals = self.locals.clone();
        for (local, param) in &param_ids {
            self.locals[*local] = Some(*param);
        }
        self.locals[binder] = Some(binder_id);
        let flow = self.block(f, body)?;
        // Capture the body-exit bindings of captured locals before restoring
        // the enclosing environment.
        let mut body_exit: BTreeMap<usize, GraphValueId> = BTreeMap::new();
        for local in &free {
            body_exit.insert(
                *local,
                self.locals
                    .get(*local)
                    .and_then(|binding| *binding)
                    .ok_or("a captured local is not bound")?,
            );
        }
        self.locals = saved_locals;
        if !matches!(flow, Flow::Next) {
            return Err("`return` inside a loop is rejected while checking".into());
        }
        // Restore the pre-loop current states: the loop node consumes those
        // tokens and installs its own exit tokens.
        for (storage, token) in &pre_states {
            self.builder.set_current_state(*storage, *token);
        }
        let info = self.loops.pop().expect("the loop stack is balanced");

        // Body region results and carried slots. A carried value's body
        // result is its body-exit binding; unchanged locals pass their
        // parameter through.
        let mut body_results = Vec::new();
        let mut carried = Vec::new();
        let mut value_ordinal = 0;
        for local in &carried_value_locals {
            let exit = body_exit[local];
            let ty = self.builder.value_type(exit)?;
            body_results.push(RegionResult::Value { id: exit, ty });
            carried.push(CarriedSlot {
                initial: RegionInput::Value(
                    self.locals
                        .get(*local)
                        .and_then(|binding| *binding)
                        .ok_or("a carried local is not bound")?,
                ),
                body_parameter: RegionParameterId(
                    1 + free
                        .iter()
                        .position(|l| l == local)
                        .expect("carried is free") as u32,
                ),
                body_result: RegionResultId(body_results.len() as u32 - 1),
                loop_result: RegionResultId(value_ordinal),
            });
            value_ordinal += 1;
        }
        for (local, storage, _) in &state_params {
            if *local == usize::MAX {
                // Independent-loop state parameter: its result is assembled
                // with the joins below.
                continue;
            }
            let token = self.builder.current_state(*storage)?;
            body_results.push(RegionResult::State {
                id: token,
                storage: *storage,
                join: None,
            });
            carried.push(CarriedSlot {
                initial: RegionInput::State(pre_states[storage]),
                body_parameter: RegionParameterId(
                    (1 + free.len()
                        + state_params
                            .iter()
                            .position(|(l, _, _)| l == local)
                            .expect("the state param exists")) as u32,
                ),
                body_result: RegionResultId(body_results.len() as u32 - 1),
                loop_result: RegionResultId(value_ordinal),
            });
            value_ordinal += 1;
        }
        if kind == LoopKind::Independent {
            for storage in &captured_storages {
                let token = self.builder.current_state(*storage)?;
                body_results.push(RegionResult::State {
                    id: token,
                    storage: *storage,
                    join: None,
                });
            }
        }
        let body_region = self.builder.end_region(body_results)?;

        // Cross-visit joins of an independent loop: the admitted atomic
        // operation, or the checker's disjoint-write proof.
        let mut joins = Vec::new();
        if kind == LoopKind::Independent {
            for (storage, write) in &info.writes {
                if write.atomic {
                    joins.push((
                        *storage,
                        StateJoin::Atomic {
                            operations: info.atomics.clone(),
                        },
                    ));
                } else if mutation.disjoint_writes {
                    joins.push((
                        *storage,
                        StateJoin::DisjointWrite {
                            binder: binder_id,
                            range: range_logical.clone(),
                        },
                    ));
                } else {
                    return Err(format!(
                        "independent loop writes storage {} without the disjoint-write proof or an atomic operation",
                        storage.0
                    ));
                }
            }
        }

        let outcome = self.builder.add_loop(LoopSpec {
            kind,
            range: range_logical,
            binder: binder_id,
            invariant_values,
            carried,
            joins,
            body: body_region,
            span,
        })?;

        // Rebind carried values to the loop's exit values; carried storages'
        // current states were installed by the loop node.
        for (ordinal, local) in carried_value_locals.iter().enumerate() {
            self.locals[*local] = Some(outcome.exit_values[ordinal]);
        }
        Ok(())
    }
}

enum SlotArg {
    Point(GraphValueId),
    Range {
        start: Option<GraphValueId>,
        end: Option<GraphValueId>,
    },
}

struct ArmBuilt {
    region: GraphRegion,
    bindings: BTreeMap<usize, GraphValueId>,
    results: Vec<Option<GraphValueId>>,
    result_states: Vec<Option<StateTokenId>>,
}

impl<'w> Work<'w> {
    // -- conditionals -------------------------------------------------------

    fn if_stmt(
        &mut self,
        f: &mut Factory,
        condition: &CheckedExpr,
        then_body: &CheckedBlock,
        else_body: &CheckedBlock,
    ) -> Result<(), String> {
        self.if_core(f, condition, then_body, else_body, None)
    }

    /// An `if` where exactly one arm returns early: the continuing arm's
    /// region swallows the rest of the enclosing block, so both arms end in
    /// `return` and their results join explicitly.
    fn if_with_continuation(
        &mut self,
        f: &mut Factory,
        condition: &CheckedExpr,
        then_body: &CheckedBlock,
        else_body: &CheckedBlock,
        rest: &[CheckedStmt],
        terminator: &BlockTerminator,
    ) -> Result<Flow, String> {
        let (then_rest, else_rest) = if returns_directly(then_body) {
            (None, Some((rest, terminator)))
        } else {
            (Some((rest, terminator)), None)
        };
        self.if_core(f, condition, then_body, else_body, then_rest.or(else_rest))?;
        Ok(Flow::Returned)
    }

    fn if_core(
        &mut self,
        f: &mut Factory,
        condition: &CheckedExpr,
        then_body: &CheckedBlock,
        else_body: &CheckedBlock,
        continuation: Option<(&[CheckedStmt], &BlockTerminator)>,
    ) -> Result<(), String> {
        let span = condition.span;
        let cond = self.expr(f, condition)?.expect("a condition is a value");
        let then_free = free_locals(then_body);
        let else_free = free_locals(else_body);
        let union_free: BTreeSet<usize> = then_free.union(&else_free).copied().collect();
        let union_writes: BTreeSet<usize> = written_roots(then_body)
            .union(&written_roots(else_body))
            .copied()
            .filter(|local| union_free.contains(local))
            // Only storage-backed locals join as states; a written scalar
            // local rebinds and joins as a value.
            .filter(|local| self.storage_of_local(*local).is_some())
            .collect();
        let union_rebinds: BTreeSet<usize> = rebound_roots(then_body)
            .union(&rebound_roots(else_body))
            .copied()
            .filter(|local| union_free.contains(local))
            .collect();
        let returning =
            returns_directly(then_body) || returns_directly(else_body) || continuation.is_some();

        // Shared parameter schema: one value per captured local, one state
        // per written storage.
        let mut params = Vec::new();
        let mut param_ids = BTreeMap::new();
        let mut captured = Vec::new();
        for local in &union_free {
            let binding = self
                .locals
                .get(*local)
                .and_then(|binding| *binding)
                .ok_or("a captured local is not bound")?;
            let ty = self.builder.value_type(binding)?;
            let view = self.builder.view_of_value(binding);
            let param = self.builder.fresh_value(ty.clone(), view)?;
            param_ids.insert(*local, param);
            captured.push(IfCapture {
                parameter: param,
                outer: binding,
            });
            params.push(RegionParameter::Value { id: param, ty });
        }
        let mut state_params = BTreeMap::new();
        for local in &union_writes {
            let storage = self
                .storage_of_local(*local)
                .ok_or("a written local has no storage")?;
            let token = self.builder.fresh_state();
            self.builder.bind_state(token, storage);
            params.push(RegionParameter::State { id: token, storage });
            state_params.insert(*local, (token, storage));
        }

        let then_cont = if returns_directly(then_body) {
            None
        } else {
            continuation
        };
        let else_cont = if returns_directly(else_body) {
            None
        } else {
            continuation
        };
        let then_arm = self.build_arm(
            f,
            &params,
            &param_ids,
            then_body,
            then_cont,
            &union_free,
            &union_rebinds,
            &union_writes,
            returning,
            span,
        )?;
        let else_arm = self.build_arm(
            f,
            &params,
            &param_ids,
            else_body,
            else_cont,
            &union_free,
            &union_rebinds,
            &union_writes,
            returning,
            span,
        )?;

        // Explicit joins. Ordinals follow the arm result layout: rebound
        // values, written states, then (when returning) result values and
        // inout states.
        let mut joins = Vec::new();
        let mut joined_bindings = BTreeMap::new();
        let mut ordinal = 0u32;
        for local in &union_rebinds {
            let then_value = then_arm.bindings[local];
            let else_value = else_arm.bindings[local];
            let ty = self.builder.value_type(then_value)?;
            let view = match (
                self.builder.view_of_value(then_value),
                self.builder.view_of_value(else_value),
            ) {
                (Some(then_view), Some(else_view))
                    if self.builder.view(then_view).storage
                        == self.builder.view(else_view).storage =>
                {
                    Some(then_view)
                }
                _ => None,
            };
            let joined = self.builder.fresh_value(ty.clone(), view)?;
            joins.push(JoinSlot::Value {
                then_result: RegionResultId(ordinal),
                else_result: RegionResultId(ordinal),
                joined,
                ty,
            });
            joined_bindings.insert(*local, joined);
            ordinal += 1;
        }
        for local in &union_writes {
            let (_, storage) = state_params[local];
            let joined = self.builder.fresh_state();
            self.builder.bind_state(joined, storage);
            joins.push(JoinSlot::State {
                then_result: RegionResultId(ordinal),
                else_result: RegionResultId(ordinal),
                joined,
                storage,
            });
            ordinal += 1;
        }
        if returning {
            for slot in 0..self.results.len() {
                let then_value =
                    then_arm.results[slot].ok_or("a returning arm binds every result value")?;
                let else_value =
                    else_arm.results[slot].ok_or("a returning arm binds every result value")?;
                let ty = self.builder.value_type(then_value)?;
                let view = match (
                    self.builder.view_of_value(then_value),
                    self.builder.view_of_value(else_value),
                ) {
                    (Some(then_view), Some(else_view))
                        if self.builder.view(then_view).storage
                            == self.builder.view(else_view).storage =>
                    {
                        Some(then_view)
                    }
                    _ => None,
                };
                let joined = self.builder.fresh_value(ty.clone(), view)?;
                joins.push(JoinSlot::Value {
                    then_result: RegionResultId(ordinal),
                    else_result: RegionResultId(ordinal),
                    joined,
                    ty,
                });
                self.results[slot] = Some(joined);
                ordinal += 1;
            }
            for (index, (_, local)) in self.inout.iter().enumerate() {
                let _then_token = then_arm.result_states[index]
                    .ok_or("a returning arm binds every inout state")?;
                let _else_token = else_arm.result_states[index]
                    .ok_or("a returning arm binds every inout state")?;
                let storage = self
                    .storage_of_local(*local)
                    .ok_or("an inout parameter has no storage")?;
                let joined = self.builder.fresh_state();
                self.builder.bind_state(joined, storage);
                joins.push(JoinSlot::State {
                    then_result: RegionResultId(ordinal),
                    else_result: RegionResultId(ordinal),
                    joined,
                    storage,
                });
                self.result_states[index] = Some(joined);
                ordinal += 1;
            }
        }

        self.builder.add_if(
            cond,
            then_arm.region,
            else_arm.region,
            joins,
            captured,
            span,
        )?;
        for (local, joined) in &joined_bindings {
            self.locals[*local] = Some(*joined);
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn build_arm(
        &mut self,
        f: &mut Factory,
        params: &[RegionParameter],
        param_ids: &BTreeMap<usize, GraphValueId>,
        body: &CheckedBlock,
        continuation: Option<(&[CheckedStmt], &BlockTerminator)>,
        union_free: &BTreeSet<usize>,
        union_rebinds: &BTreeSet<usize>,
        union_writes: &BTreeSet<usize>,
        returning: bool,
        span: Span,
    ) -> Result<ArmBuilt, String> {
        self.builder.begin_region(params.to_vec())?;
        let saved_locals = self.locals.clone();
        let saved_results = self.results.clone();
        let saved_states = self.result_states.clone();
        for (local, param) in param_ids {
            self.locals[*local] = Some(*param);
        }
        self.if_depth += 1;
        let mut returned = matches!(self.block(f, body)?, Flow::Returned);
        if !returned {
            if let Some((rest, terminator)) = continuation {
                returned = matches!(self.statements(f, rest, terminator)?, Flow::Returned);
            }
        }
        self.if_depth -= 1;
        if returning && !returned {
            return Err(format!(
                "an arm of the `if` at {span:?} does not end in `return`"
            ));
        }
        if !returning && returned {
            return Err(format!(
                "an arm of the `if` at {span:?} returns while the construct continues"
            ));
        }

        // Arm results: the fixed join layout.
        let mut arm_results = Vec::new();
        let mut bindings = BTreeMap::new();
        for local in union_rebinds {
            let value = self
                .locals
                .get(*local)
                .and_then(|binding| *binding)
                .ok_or("an arm binding is not bound")?;
            let ty = self.builder.value_type(value)?;
            arm_results.push(RegionResult::Value { id: value, ty });
            bindings.insert(*local, value);
        }
        for local in union_writes {
            let storage = self
                .storage_of_local(*local)
                .ok_or("a written local has no storage")?;
            let token = self.builder.current_state(storage)?;
            arm_results.push(RegionResult::State {
                id: token,
                storage,
                join: None,
            });
        }
        let mut results = Vec::new();
        let mut result_states = Vec::new();
        if returning {
            results = self.results.clone();
            result_states = self.result_states.clone();
            for value in results.iter().flatten() {
                let ty = self.builder.value_type(*value)?;
                arm_results.push(RegionResult::Value { id: *value, ty });
            }
            for (index, (_, local)) in self.inout.iter().enumerate() {
                let token =
                    self.result_states[index].ok_or("a returning arm binds every inout state")?;
                let storage = self
                    .storage_of_local(*local)
                    .ok_or("an inout parameter has no storage")?;
                arm_results.push(RegionResult::State {
                    id: token,
                    storage,
                    join: None,
                });
            }
        }
        let region = self.builder.end_region(arm_results)?;
        self.locals = saved_locals;
        self.results = saved_results;
        self.result_states = saved_states;
        let _ = union_free;
        Ok(ArmBuilt {
            region,
            bindings,
            results,
            result_states,
        })
    }

    // -- calls --------------------------------------------------------------

    fn call(
        &mut self,
        f: &mut Factory,
        e: &CheckedExpr,
        call: &CheckedCall,
        args: &[CheckedExpr],
    ) -> Result<Option<GraphValueId>, String> {
        let span = e.span;
        let mut arg_values = Vec::new();
        for arg in args {
            arg_values.push(self.expr(f, arg)?.expect("a call argument is a value"));
        }

        self.call_inner(f, e, call, args, arg_values, span)
    }

    fn call_inner(
        &mut self,
        f: &mut Factory,
        e: &CheckedExpr,
        call: &CheckedCall,
        args: &[CheckedExpr],
        arg_values: Vec<GraphValueId>,
        span: Span,
    ) -> Result<Option<GraphValueId>, String> {
        let family_name = {
            let index = program_index(f, call.family);
            f.program.families[index].name.clone()
        };
        let candidates = family::candidates_of_call(f.program, call, &f.target.backend);
        if candidates.is_empty() {
            f.rejections.push(OccurrenceRejection {
                occurrence: format!("call of `{family_name}` at {}..{}", span.start, span.end),
                rejections: Vec::new(),
            });
            f.no_implementation = true;
            return Err(format!(
                "no implementation of `{family_name}` is applicable on `{}`",
                f.target.backend
            ));
        }
        let shapes = self.shapes.clone();
        let elems = self.elems.clone();
        let caller_shape = |name: &str| {
            shapes
                .get(name)
                .and_then(|extent| extent.as_static())
                .map(|value| value as i64)
        };
        let caller_elem = |name: &str| elems.get(name).cloned();
        let app = family::applicable(
            f.program,
            &f.target.backend,
            f.supports,
            &candidates,
            &caller_shape,
            &caller_elem,
        );
        if app.alternatives.is_empty() {
            f.rejections.push(OccurrenceRejection {
                occurrence: format!("call of `{family_name}` at {}..{}", span.start, span.end),
                rejections: app.rejected.clone(),
            });
            f.no_implementation = true;
            return Err(format!(
                "no implementation of `{family_name}` is applicable at this occurrence on `{}`",
                f.target.backend
            ));
        }

        // One choice for this occurrence; the interface comes from the
        // contract with the occurrence's concrete argument types.
        let choice = f.alloc_choice();
        let contract = f
            .program
            .definition(f.program.families[program_index(f, call.family)].contract);
        if contract.params.len() != args.len() {
            return Err(format!(
                "the contract of `{family_name}` has {} parameters but the call passes {} arguments",
                contract.params.len(),
                args.len()
            ));
        }
        let mut interface_params = Vec::new();
        for (param, value) in contract.params.iter().zip(&arg_values) {
            interface_params.push(InterfaceParam {
                name: param.name.clone(),
                mode: param.mode,
                ownership: param.ownership,
                ty: self.builder.value_type(*value)?,
            });
        }
        let interface = FunctionInterface {
            name: family_name.clone(),
            params: interface_params,
            result: self.convert_type(f, &e.ty)?,
        };

        // Alternatives: one graph per (occurrence, alternative). The graph's
        // allocator is handed to the factory while the callee graphs are
        // built, then restored for the boundary construction.
        f.ids = self.builder.take_ids();
        let mut alternatives = Vec::new();
        for (ordinal, resolved) in app.alternatives.iter().enumerate() {
            let candidate = &resolved.candidate;
            if candidate.arg_order != (0..args.len()).collect::<Vec<_>>() {
                return Err(format!(
                    "the candidates of `{family_name}` do not share the contract's argument order at this occurrence"
                ));
            }
            let definition = f.program.definition(candidate.definition);
            let mut child_shapes = ShapeEnv::new();
            for (param, sym) in &candidate.shape_args {
                child_shapes.insert(param.clone(), self.resolve_sym(f, sym)?);
            }
            let mut child_elems = ElemEnv::new();
            for (param, elem) in &candidate.elem_args {
                let resolved = match elem {
                    Elem::Param(p) => self
                        .elems
                        .get(p)
                        .cloned()
                        .ok_or_else(|| format!("element parameter `{p}` is not bound here"))?,
                    other => other.clone(),
                };
                child_elems.insert(param.clone(), resolved);
            }
            let graph = f
                .build_graph(
                    definition,
                    child_shapes,
                    child_elems,
                    choice,
                    ordinal as u32,
                    false,
                )
                .map_err(|error| match error {
                    BuildError::Invalid(reason) => reason,
                    BuildError::NoImplementation(_) => {
                        "an inner occurrence has no applicable implementation".to_string()
                    }
                })?;
            alternatives.push(LogicalAlternative {
                definition: candidate.definition,
                kind: candidate.kind,
                graph,
                required_capabilities: resolved.required_capabilities.clone(),
                authored_numerical_effects: resolved.authored_numerical_effects.clone(),
            });
        }
        f.install_choice(
            choice,
            ImplementationChoice {
                interface,
                alternatives: NonEmpty::new(alternatives)
                    .ok_or("the occurrence has alternatives")?,
            },
        );
        self.builder.restore_ids(std::mem::take(&mut f.ids));

        // Boundary inputs: one per interface leaf, at its canonical path.
        let interface = f.choices[choice.index()]
            .clone()
            .expect("the choice was just installed")
            .interface;
        let mut boundary_inputs = Vec::new();
        let mut exclusively_borrowed = Vec::new();
        for (ordinal, param) in interface.params.iter().enumerate() {
            let arg = arg_values[ordinal];
            for (path, leaf) in boundary_leaves(&param.ty) {
                let leaf_value = self.decompose_value(f, arg, &path.0)?;
                match leaf {
                    BoundaryLeaf::Tensor(_) => {
                        // A tensor leaf must be storage-backed; computed
                        // values are materialized first.
                        let leaf_value = match self.builder.view_of_value(leaf_value) {
                            Some(_) => leaf_value,
                            None => self.materialize_value(f, leaf_value, span)?,
                        };
                        let view = self
                            .builder
                            .view_of_value(leaf_value)
                            .expect("the value is view-backed");
                        let storage = self.builder.view(view).storage;
                        let token = self.builder.current_state(storage)?;
                        let kind = match param.ownership {
                            ParamOwnership::Value => {
                                return Err("a tensor parameter must be borrowed or owned".into());
                            }
                            ParamOwnership::Shared => BoundaryInputKind::Shared {
                                value: leaf_value,
                                state: token,
                            },
                            ParamOwnership::Exclusive => {
                                exclusively_borrowed.push(storage);
                                BoundaryInputKind::Exclusive {
                                    value: leaf_value,
                                    state: token,
                                }
                            }
                            ParamOwnership::Owned => BoundaryInputKind::Move {
                                value: leaf_value,
                                state: token,
                            },
                        };
                        boundary_inputs.push(BoundaryInput {
                            path,
                            param: ordinal as u32,
                            kind,
                        });
                    }
                    _ => boundary_inputs.push(BoundaryInput {
                        path,
                        param: ordinal as u32,
                        kind: BoundaryInputKind::Value(leaf_value),
                    }),
                }
            }
        }

        // Boundary results: one per result leaf at its canonical path, plus
        // the next state of every `inout` parameter storage.
        let mut boundary_results = Vec::new();
        let mut result_leaf_values = Vec::new();
        for (path, leaf) in boundary_leaves(&interface.result) {
            match leaf {
                BoundaryLeaf::Tensor(shape) => {
                    let storage = self.builder.declare_storage(
                        shape.clone(),
                        StorageOrigin::Result {
                            owner: Some(choice),
                            path: path.clone(),
                        },
                        Initialization::FullyInitialized,
                    );
                    let token = self.builder.fresh_state();
                    self.builder.bind_state(token, storage);
                    let view = self.builder.declare_view(
                        storage,
                        shape,
                        Access::Exclusive,
                        ViewTransform::Identity,
                    );
                    let value = self.builder.fresh_value(
                        ValueType::Tensor(self.builder.view(view).shape.clone()),
                        Some(view),
                    )?;
                    result_leaf_values.push((path.clone(), value));
                    boundary_results.push(BoundaryResult {
                        path,
                        kind: BoundaryResultKind::Storage {
                            storage,
                            ty: self.builder.view(view).shape.clone(),
                            token,
                        },
                    });
                }
                _ => {
                    let ty = leaf_value_type(&leaf);
                    let value = self.builder.fresh_value(ty, None)?;
                    result_leaf_values.push((path.clone(), value));
                    boundary_results.push(BoundaryResult {
                        path,
                        kind: BoundaryResultKind::Value(value),
                    });
                }
            }
        }
        for (ordinal, param) in interface.params.iter().enumerate() {
            if param.mode != Mode::Inout {
                continue;
            }
            let arg = arg_values[ordinal];
            for (path, leaf) in boundary_leaves(&param.ty) {
                if matches!(leaf, BoundaryLeaf::Tensor(_)) {
                    let leaf_value = self.decompose_value(f, arg, &path.0)?;
                    let view = self
                        .builder
                        .view_of_value(leaf_value)
                        .ok_or("an inout argument leaf is storage-backed")?;
                    let storage = self.builder.view(view).storage;
                    let token = self.builder.fresh_state();
                    self.builder.bind_state(token, storage);
                    boundary_results.push(BoundaryResult {
                        path,
                        kind: BoundaryResultKind::State(token),
                    });
                }
            }
        }

        let result_value_ids = result_leaf_values
            .iter()
            .map(|(_, value)| *value)
            .collect::<Vec<_>>();
        self.builder.add_call(
            choice,
            boundary_inputs,
            boundary_results,
            result_value_ids,
            span,
        )?;
        // The callee returns every exclusively borrowed storage fully
        // initialized: the checker admits an unassigned argument only on
        // that condition, and an assigned one stays initialized.
        for storage in exclusively_borrowed {
            self.builder.mark_fully_initialized(storage);
        }

        // The call's value: a single leaf passes through; tuples are repacked
        // from their leaf values; void calls retain completion only.
        let value = match &interface.result {
            ValueType::Void => None,
            ValueType::Tuple(_) if result_leaf_values.len() > 1 => {
                let components = result_leaf_values
                    .iter()
                    .map(|(_, value)| *value)
                    .collect::<Vec<_>>();
                let tys = components
                    .iter()
                    .map(|id| self.builder.value_type(*id))
                    .collect::<Result<_, _>>()?;
                let ty = ValueType::Tuple(NonEmpty::new(tys).expect("leaves are nonempty"));
                let spec = PrimitiveSpec {
                    op: PrimitiveOp::Primitive(PrimitiveId::TuplePack),
                    inputs: components,
                    reads: Vec::new(),
                    write: None,
                    outputs: vec![Output::Value(ty)],
                    safety: Vec::new(),
                    span,
                };
                Some(self.add(f, spec)?.expect("tuple pack produces a value"))
            }
            _ => Some(result_leaf_values[0].1),
        };
        Ok(value)
    }

    /// Decompose one value along an ordinal path with `tuple.get` nodes.
    fn decompose_value(
        &mut self,
        f: &mut Factory,
        value: GraphValueId,
        path: &[u32],
    ) -> Result<GraphValueId, String> {
        let mut current = value;
        for index in path {
            current = self.tuple_get(f, current, *index as usize)?;
        }
        Ok(current)
    }
}

fn program_index(f: &Factory, family: usize) -> usize {
    let _ = f;
    family
}

// ---------------------------------------------------------------------------
// Entry construction
// ---------------------------------------------------------------------------

fn entry_convert_type(
    ty: &ValueType,
    shapes: &ShapeEnv,
    elems: &ElemEnv,
    context: &str,
) -> Result<ValueType, String> {
    fn convert_extent(
        extent: &ExtentExpr,
        shapes: &ShapeEnv,
        context: &str,
    ) -> Result<ExtentExpr, String> {
        match extent {
            ExtentExpr::Static(n) => Ok(ExtentExpr::Static(*n)),
            ExtentExpr::Runtime(id) => Ok(ExtentExpr::Runtime(*id)),
            ExtentExpr::Sym(sym) => {
                let value = sym.eval(&|name| {
                    shapes
                        .get(name)
                        .and_then(|extent| extent.as_static())
                        .map(|value| value as i64)
                });
                match value {
                    Some(value) if value >= 0 => Ok(ExtentExpr::Static(value as u64)),
                    Some(_) => Err(format!("{context} has a negative extent")),
                    None => Err(format!(
                        "{context} depends on `{sym}`, which is not a concrete entry shape"
                    )),
                }
            }
        }
    }
    fn convert_elem(elem: &Elem, elems: &ElemEnv, context: &str) -> Result<Elem, String> {
        match elem {
            Elem::Param(p) => elems
                .get(p)
                .cloned()
                .ok_or_else(|| format!("element parameter `{p}` of {context} is not bound")),
            other => Ok(other.clone()),
        }
    }
    match ty {
        ValueType::Scalar(d) => Ok(ValueType::Scalar(*d)),
        ValueType::Index { bound } => Ok(ValueType::Index {
            bound: convert_extent(bound, shapes, context)?,
        }),
        ValueType::Range { bound } => Ok(ValueType::Range {
            bound: convert_extent(bound, shapes, context)?,
        }),
        ValueType::Tensor(s) => Ok(ValueType::Tensor(TensorType {
            axes: s
                .axes
                .iter()
                .map(|axis| convert_extent(axis, shapes, context))
                .collect::<Result<_, _>>()?,
            elem: convert_elem(&s.elem, elems, context)?,
            packed_axis: s.packed_axis,
        })),
        ValueType::Tuple(items) => Ok(ValueType::Tuple(
            NonEmpty::new(
                items
                    .iter()
                    .map(|item| entry_convert_type(item, shapes, elems, context))
                    .collect::<Result<_, _>>()?,
            )
            .expect("the source tuple is nonempty"),
        )),
        ValueType::CapabilityValue(n) => Ok(ValueType::CapabilityValue(n.clone())),
        ValueType::Void => Ok(ValueType::Void),
    }
}

pub(super) fn construct(
    program: &Program,
    entry: &str,
    target: &EffectiveTargetIdentity,
    supports: &dyn Fn(&IntrinsicUse) -> Result<(), String>,
    spec: Specialization,
) -> Result<LogicalProgram, BuildError> {
    let family_index = program.family_index(entry).map_err(BuildError::Invalid)?;
    let mut shapes = ShapeEnv::new();
    for (name, value) in &spec.shapes {
        if *value < 0 {
            return Err(BuildError::Invalid(format!(
                "shape parameter `{name}` is negative"
            )));
        }
        shapes.insert(name.clone(), ExtentExpr::Static(*value as u64));
    }
    let elems = spec.elems.clone();
    let (candidates, unreachable) =
        family::entry_candidates(program, family_index, &target.backend);
    let mut factory = Factory::new(program, target, supports, entry);
    let caller_shape = |name: &str| {
        shapes
            .get(name)
            .and_then(|extent| extent.as_static())
            .map(|value| value as i64)
    };
    let caller_elem = |name: &str| elems.get(name).cloned();
    let app = family::applicable(
        program,
        &target.backend,
        supports,
        &candidates,
        &caller_shape,
        &caller_elem,
    );
    if app.alternatives.is_empty() {
        let mut rejections = unreachable.clone();
        rejections.extend(app.rejected.clone());
        return Err(BuildError::NoImplementation(ApplicabilityReport {
            entry: entry.to_string(),
            occurrences: vec![OccurrenceRejection {
                occurrence: format!("entry `{entry}`"),
                rejections,
            }],
        }));
    }

    let entry_choice = factory.alloc_choice();
    let contract = program.definition(program.families[family_index].contract);
    let context = format!("the contract of `{}`", contract.name);
    let interface = FunctionInterface {
        name: contract.name.clone(),
        params: {
            let mut out = Vec::new();
            for param in &contract.params {
                out.push(InterfaceParam {
                    name: param.name.clone(),
                    mode: param.mode,
                    ownership: param.ownership,
                    ty: entry_convert_type(&param.ty, &shapes, &elems, &context)
                        .map_err(BuildError::Invalid)?,
                });
            }
            out
        },
        result: entry_convert_type(&contract.result, &shapes, &elems, &context)
            .map_err(BuildError::Invalid)?,
    };

    let mut alternatives = Vec::new();
    for (ordinal, resolved) in app.alternatives.iter().enumerate() {
        let candidate = &resolved.candidate;
        let definition = program.definition(candidate.definition);
        let mut child_shapes = ShapeEnv::new();
        for (param, sym) in &candidate.shape_args {
            let value = sym.eval(&|name| {
                shapes
                    .get(name)
                    .and_then(|extent| extent.as_static())
                    .map(|value| value as i64)
            });
            match value {
                Some(value) if value >= 0 => {
                    child_shapes.insert(param.clone(), ExtentExpr::Static(value as u64));
                }
                Some(_) => {
                    return Err(BuildError::Invalid(format!(
                        "shape parameter `{param}` of `{}` binds a negative extent",
                        definition.name
                    )));
                }
                None => {
                    return Err(BuildError::Invalid(format!(
                        "shape parameter `{param}` of `{}` is not concrete at the entry",
                        definition.name
                    )));
                }
            }
        }
        let mut child_elems = ElemEnv::new();
        for (param, elem) in &candidate.elem_args {
            let resolved = match elem {
                Elem::Param(p) => elems.get(p).cloned().ok_or_else(|| {
                    BuildError::Invalid(format!(
                        "element parameter `{p}` of `{}` is not bound at the entry",
                        definition.name
                    ))
                })?,
                other => other.clone(),
            };
            child_elems.insert(param.clone(), resolved);
        }
        let graph = factory.build_graph(
            definition,
            child_shapes,
            child_elems,
            entry_choice,
            ordinal as u32,
            true,
        )?;
        alternatives.push(LogicalAlternative {
            definition: candidate.definition,
            kind: candidate.kind,
            graph,
            required_capabilities: resolved.required_capabilities.clone(),
            authored_numerical_effects: resolved.authored_numerical_effects.clone(),
        });
    }
    factory.install_choice(
        entry_choice,
        ImplementationChoice {
            interface,
            alternatives: NonEmpty::new(alternatives).expect("applicability is nonempty"),
        },
    );

    let graphs = factory
        .graphs
        .into_iter()
        .collect::<Option<Vec<TaskGraph>>>()
        .ok_or_else(|| BuildError::Invalid("a graph slot was not filled".into()))?;
    let choices = factory
        .choices
        .into_iter()
        .collect::<Option<Vec<ImplementationChoice>>>()
        .ok_or_else(|| BuildError::Invalid("a choice slot was not filled".into()))?;
    let identity = logical_identity(program, entry, &target, &spec);
    Ok(LogicalProgram {
        identity,
        entry: entry.to_string(),
        target: target.clone(),
        shapes: spec.shapes,
        elements: elems,
        entry_choice,
        choices: IdVec::new(choices),
        graphs: IdVec::new(graphs),
        runtime_extents: IdVec::new(factory.runtime_extents),
    })
}

/// The logical identity: source identity, registry revision, entry and the
/// concrete specialization. Changing any of them changes every downstream
/// plan, cache and evidence identity.
fn logical_identity(
    program: &Program,
    entry: &str,
    target: &EffectiveTargetIdentity,
    spec: &Specialization,
) -> LogicalIdentity {
    use sha2::{Digest, Sha256};
    let mut hash = Sha256::new();
    hash.update(program.identity());
    hash.update(crate::intrinsics::REGISTRY_REVISION.as_bytes());
    hash.update(entry.as_bytes());
    hash.update(target.backend.as_bytes());
    hash.update(target.capability_fingerprint.as_bytes());
    for (name, value) in &spec.shapes {
        hash.update(name.as_bytes());
        hash.update(value.to_le_bytes());
    }
    for (name, elem) in &spec.elems {
        hash.update(name.as_bytes());
        hash.update(format!("{elem}").as_bytes());
    }
    LogicalIdentity(hash.finalize().into())
}

/// Borrowed leaf classification for defensive verification.
pub(super) enum BoundaryLeafRef {
    Scalar,
    Index,
    Range,
    Tensor,
    Capability,
}

/// The canonical (path, leaf) list of one boundary type, for verification of
/// call boundaries.
pub(super) fn boundary_leaf_paths(ty: &ValueType) -> Vec<(ValuePath, BoundaryLeafRef)> {
    fn walk(ty: &ValueType, path: &ValuePath, out: &mut Vec<(ValuePath, BoundaryLeafRef)>) {
        match ty {
            ValueType::Scalar(_) => out.push((path.clone(), BoundaryLeafRef::Scalar)),
            ValueType::Index { .. } => out.push((path.clone(), BoundaryLeafRef::Index)),
            ValueType::Range { .. } => out.push((path.clone(), BoundaryLeafRef::Range)),
            ValueType::Tensor(_) => out.push((path.clone(), BoundaryLeafRef::Tensor)),
            ValueType::Tuple(items) => {
                for (i, item) in items.iter().enumerate() {
                    walk(item, &path.extend(i as u32), out);
                }
            }
            ValueType::CapabilityValue(_) => out.push((path.clone(), BoundaryLeafRef::Capability)),
            ValueType::Void => {}
        }
    }
    let mut out = Vec::new();
    walk(ty, &ValuePath::default(), &mut out);
    out
}
