//! Semantic-to-logical construction: a single forward pass over the checked
//! body.
//!
//! One lexical environment maps each local to its current value. Constructors
//! create outputs and dependencies immediately; nothing is collected by
//! recursive scanning afterwards. Arguments are region parameters, never
//! pseudo-values; a computed tensor is a value without logical storage and is
//! returned, reduced, read, viewed or passed into a call as such — a view of
//! a computed tensor names the value as its base, and a computed call
//! argument is shared or moved as a value. Storage exists only where the
//! source names it: parameters, allocation-family primitives, and the mutable
//! state of a `let mut` local written in place (the one materialization
//! construction performs, because the write itself names state).
//! Initialization follows `Uninitialized -> PartiallyInitialized(coverage
//! proof) -> FullyInitialized` with structurally composing disjoint coverage.
//!
//! Domain specialization (package W1 behavior under W1's frozen types): exact entry
//! shape bindings are static extents; bounded bindings are one retained
//! invocation-sourced runtime extent each; every implementation predicate is
//! decided over the whole domain; every shape expression of every occurrence
//! is resolved in the entry's symbol space so equal expressions share one
//! runtime extent program-wide.

use super::builder::{computed_kind, GraphBuilder, Ids, LoopSpec, Output, PrimitiveSpec, WriteEffect};
use super::specialization::{predicate_verdict, shape_interval, PredicateVerdict, ShapeBinding};
use super::*;
use crate::check::atom_var;
use crate::family;
use crate::intrinsics::{accumulator_dtype, IndexSlot, PrimitiveId};
use crate::sir::{
    BlockTerminator, CheckedBlock, CheckedCall, CheckedExpr, CheckedExprKind, CheckedIndex,
    CheckedPlace, CheckedRange, CheckedStmt, DefId, Definition, IntrinsicUse, Literal, LocalId,
    LoopKind, LoopMutationSummary, ParamOwnership, Pattern, Predicate, Program,
};
use crate::span::Span;
use crate::sym::{Atom, Sym};
use crate::syntax::ast::{AssignOp, BinaryOp};
use crate::types::{
    CapabilityValueType, DType, Elem, ExtentExpr, NonEmpty, RuntimeExtentId, TensorType,
    ValuePath, ValueType,
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

/// One canonical leaf of a boundary type. This is `types::canonical_leaves`'
/// path scheme extended with capability values: they never cross the public
/// ABI or a portable boundary, but a same-backend call passes them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum LeafKind<'a> {
    Scalar(DType),
    Index(&'a ExtentExpr),
    Range(&'a ExtentExpr),
    Tensor(&'a TensorType),
    Capability(&'a CapabilityValueType),
}

impl LeafKind<'_> {
    fn ty(&self) -> ValueType {
        match self {
            LeafKind::Scalar(dtype) => ValueType::Scalar(*dtype),
            LeafKind::Index(bound) => ValueType::Index {
                bound: (*bound).clone(),
            },
            LeafKind::Range(bound) => ValueType::Range {
                bound: (*bound).clone(),
            },
            LeafKind::Tensor(ty) => ValueType::Tensor((*ty).clone()),
            LeafKind::Capability(ty) => ValueType::CapabilityValue((*ty).clone()),
        }
    }

    /// The kind of a value this leaf produces at a boundary where it is not a
    /// view: a tensor leaf is computed.
    fn produced_kind(&self) -> GraphValueKind {
        match self {
            LeafKind::Scalar(dtype) => GraphValueKind::Scalar(*dtype),
            LeafKind::Index(bound) => GraphValueKind::Index {
                bound: (*bound).clone(),
            },
            LeafKind::Range(bound) => GraphValueKind::Range {
                bound: (*bound).clone(),
            },
            LeafKind::Tensor(ty) => GraphValueKind::Tensor {
                ty: (*ty).clone(),
                source: TensorSource::Computed,
            },
            LeafKind::Capability(ty) => GraphValueKind::Capability((*ty).clone()),
        }
    }
}

/// The canonical (path, leaf) list of one boundary type. `Void` has no leaves.
pub(super) fn boundary_leaves(ty: &ValueType) -> Vec<(ValuePath, LeafKind<'_>)> {
    fn walk<'a>(ty: &'a ValueType, path: &ValuePath, out: &mut Vec<(ValuePath, LeafKind<'a>)>) {
        match ty {
            ValueType::Scalar(d) => out.push((path.clone(), LeafKind::Scalar(*d))),
            ValueType::Index { bound } => out.push((path.clone(), LeafKind::Index(bound))),
            ValueType::Range { bound } => out.push((path.clone(), LeafKind::Range(bound))),
            ValueType::Tensor(s) => out.push((path.clone(), LeafKind::Tensor(s))),
            ValueType::Tuple(items) => {
                for (i, item) in items.iter().enumerate() {
                    walk(item, &path.extend(i as u32), out);
                }
            }
            ValueType::CapabilityValue(n) => out.push((path.clone(), LeafKind::Capability(n))),
            ValueType::Void => {}
        }
    }
    let mut out = Vec::new();
    walk(ty, &ValuePath::default(), &mut out);
    out
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
        CheckedExprKind::Primitive { .. }
        | CheckedExprKind::Capability { .. }
        | CheckedExprKind::Call { .. }
        | CheckedExprKind::Literal(_) => None,
    }
}

/// Locals whose storage this block writes in place: element and slice
/// assignment places, atomic bases, and call arguments borrowed into callees.
/// Over-approximated (a shared borrow is counted).
fn storage_written_roots(block: &CheckedBlock) -> BTreeSet<LocalId> {
    let mut out = BTreeSet::new();
    fn walk(block: &CheckedBlock, out: &mut BTreeSet<LocalId>) {
        for stmt in &block.statements {
            match stmt {
                CheckedStmt::Assign { place, .. } => match place {
                    CheckedPlace::Local { .. } => {}
                    CheckedPlace::Element { root, .. } => {
                        out.insert(*root);
                    }
                    CheckedPlace::Tuple(places) => {
                        for place in places {
                            if let CheckedPlace::Element { root, .. } = place {
                                out.insert(*root);
                            }
                        }
                    }
                },
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
                if let Some(root) = root_var(operands.first().expect("an atomic has a base")) {
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

/// Every local this block may write: in place or by rebinding.
fn written_roots(block: &CheckedBlock) -> BTreeSet<LocalId> {
    storage_written_roots(block)
        .union(&rebound_roots(block))
        .copied()
        .collect()
}

fn returns_directly(block: &CheckedBlock) -> bool {
    matches!(block.terminator, BlockTerminator::Return(_))
}

// ---------------------------------------------------------------------------
// Predicates over the specialization domain
// ---------------------------------------------------------------------------

fn predicate_display(predicate: &Predicate) -> String {
    match predicate {
        Predicate::NonNegative(expr) => format!("0 <= {expr}"),
        Predicate::Zero(expr) => format!("{expr} == 0"),
        Predicate::NonZero(expr) => format!("{expr} != 0"),
    }
}

/// Decide one implementation predicate (already substituted into the
/// caller's symbol space) over the entry domain: it is rewritten into the
/// entry's symbol space through `caller_to_entry`, must mention entry shape
/// parameters only, and admits the alternative only under `Always`.
fn judge_predicate(
    predicate: &Predicate,
    caller_to_entry: &dyn Fn(&str) -> Option<Sym>,
    domain: &SpecializationDomain,
) -> Result<(), String> {
    let rewrite = |expr: &Sym| family::substitute(expr, caller_to_entry);
    let entry_form = match predicate {
        Predicate::NonNegative(expr) => Predicate::NonNegative(rewrite(expr)),
        Predicate::Zero(expr) => Predicate::Zero(rewrite(expr)),
        Predicate::NonZero(expr) => Predicate::NonZero(rewrite(expr)),
    };
    let expr = match &entry_form {
        Predicate::NonNegative(expr) | Predicate::Zero(expr) | Predicate::NonZero(expr) => expr,
    };
    let display = predicate_display(&entry_form);
    for name in expr.params() {
        if !domain.shapes().contains_key(&name) {
            return Err(format!(
                "the `where` predicate `{display}` depends on `{name}`, which is not a shape parameter of the entry"
            ));
        }
    }
    match predicate_verdict(&entry_form, domain) {
        PredicateVerdict::Always => Ok(()),
        PredicateVerdict::Never => Err(format!(
            "predicate `{display}` fails on the whole specialization domain"
        )),
        PredicateVerdict::Mixed => Err(format!(
            "predicate `{display}` is true on only part of the domain; declare a workload partition"
        )),
    }
}

// ---------------------------------------------------------------------------
// Program-level factory
// ---------------------------------------------------------------------------

/// One resolved shape parameter of an occurrence: static, or one retained
/// runtime extent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ShapeExtent {
    Static(u64),
    Runtime(RuntimeExtentId),
}

impl ShapeExtent {
    fn extent(self) -> ExtentExpr {
        match self {
            ShapeExtent::Static(n) => ExtentExpr::Static(n),
            ShapeExtent::Runtime(id) => ExtentExpr::Runtime(id),
        }
    }
}

fn static_extent(value: i64) -> Result<ShapeExtent, String> {
    u64::try_from(value)
        .map(ShapeExtent::Static)
        .map_err(|_| format!("extent `{value}` is negative"))
}

/// Shape parameter name -> resolved extent, in one occurrence's symbol space.
type ShapeEnv = BTreeMap<String, ShapeExtent>;
/// Shape parameter name -> its expression over the entry's shape parameters.
type SymEnv = BTreeMap<String, Sym>;
type ElemEnv = BTreeMap<String, Elem>;

struct Factory<'a> {
    program: &'a Program,
    target: &'a EffectiveTargetIdentity,
    supports: &'a dyn Fn(&IntrinsicUse) -> Result<(), String>,
    domain: &'a SpecializationDomain,
    /// The entry's shape parameters: exact ones static, bounded ones runtime.
    entry_shapes: ShapeEnv,
    shape_fields: Vec<ShapeField>,
    ids: Ids,
    next_choice: u32,
    next_graph: u32,
    choices: Vec<Option<ImplementationChoice>>,
    graphs: Vec<Option<TaskGraph>>,
    runtime_extents: Vec<RuntimeExtent>,
    /// Runtime extents of entry-space shape expressions, by display form:
    /// equal expressions share one extent program-wide.
    shape_memo: BTreeMap<String, RuntimeExtentId>,
    rejections: Vec<OccurrenceRejection>,
    /// Set when any occurrence had no applicable implementation; the error
    /// cascades to the top as `NoImplementation`.
    no_implementation: bool,
    /// Definitions whose graphs are under construction (the static call
    /// stack); re-entering one is recursion, which cannot terminate under
    /// whole-program instantiation.
    building: Vec<DefId>,
}

impl<'a> Factory<'a> {
    fn new(
        program: &'a Program,
        target: &'a EffectiveTargetIdentity,
        supports: &'a dyn Fn(&IntrinsicUse) -> Result<(), String>,
        domain: &'a SpecializationDomain,
    ) -> Factory<'a> {
        Factory {
            program,
            target,
            supports,
            domain,
            entry_shapes: ShapeEnv::new(),
            shape_fields: Vec::new(),
            ids: Ids::default(),
            next_choice: 0,
            next_graph: 0,
            choices: Vec::new(),
            graphs: Vec::new(),
            runtime_extents: Vec::new(),
            shape_memo: BTreeMap::new(),
            rejections: Vec::new(),
            no_implementation: false,
            building: Vec::new(),
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

    fn runtime_extent(
        &mut self,
        value: RuntimeScalarExpr,
        capacity: u64,
        expected: Option<u64>,
    ) -> RuntimeExtentId {
        let id = RuntimeExtentId(self.runtime_extents.len() as u32);
        self.runtime_extents.push(RuntimeExtent {
            id,
            value,
            capacity,
            expected,
        });
        id
    }

    /// Whether a symbolic expression mentions entry shape parameters only.
    fn is_entry_shape_sym(&self, sym: &Sym) -> bool {
        sym.params()
            .iter()
            .all(|name| self.entry_shapes.contains_key(name))
    }

    /// Resolve one entry-space shape expression: static when it folds to a
    /// constant, otherwise one runtime extent shared by every equal
    /// expression of the program.
    fn shape_extent(&mut self, sym: &Sym) -> Result<ShapeExtent, String> {
        if let Some(value) = sym.as_constant() {
            return static_extent(value);
        }
        let key = sym.to_string();
        if let Some(id) = self.shape_memo.get(&key) {
            return Ok(ShapeExtent::Runtime(*id));
        }
        let expr = fold_scalar(self.shape_scalar_expr(sym)?);
        let resolved = match expr {
            RuntimeScalarExpr::Const(value) => static_extent(value)?,
            RuntimeScalarExpr::Extent(id) => ShapeExtent::Runtime(id),
            expr => {
                // A shape expression mentions no graph value; its bound is
                // proved from the shape fields' finite domains, or by the
                // domain's own interval arithmetic. No proof is a
                // construction error, never a substituted capacity.
                let no_value = |id: GraphValueId| -> Result<GraphValueKind, String> {
                    Err(format!("shape expression mentions graph value {}", id.0))
                };
                let capacity = match runtime_scalar_bound(
                    &expr,
                    &self.runtime_extents,
                    &self.shape_fields,
                    &no_value,
                ) {
                    Ok(capacity) => capacity,
                    Err(reason) => match shape_interval(sym, self.domain) {
                        Some((_, max)) => max,
                        None => {
                            return Err(format!(
                                "shape expression `{sym}` has no finite bound over the specialization domain because {reason}"
                            ))
                        }
                    },
                };
                ShapeExtent::Runtime(self.runtime_extent(expr, capacity, None))
            }
        };
        if let ShapeExtent::Runtime(id) = resolved {
            self.shape_memo.insert(key, id);
        }
        Ok(resolved)
    }

    fn shape_scalar_expr(&self, sym: &Sym) -> Result<RuntimeScalarExpr, String> {
        let mut total = RuntimeScalarExpr::Const(0);
        for (monomial, coefficient) in sym.monomials() {
            let mut term = RuntimeScalarExpr::Const(coefficient);
            for (atom, power) in monomial {
                let atom_expr = self.shape_atom_expr(atom)?;
                for _ in 0..*power {
                    term = RuntimeScalarExpr::Mul(Box::new(term), Box::new(atom_expr.clone()));
                }
            }
            total = RuntimeScalarExpr::Add(Box::new(total), Box::new(term));
        }
        Ok(total)
    }

    fn shape_atom_expr(&self, atom: &Atom) -> Result<RuntimeScalarExpr, String> {
        match atom {
            Atom::Param(name) => match self.entry_shapes.get(name) {
                Some(ShapeExtent::Static(n)) => Ok(RuntimeScalarExpr::Const(*n as i64)),
                Some(ShapeExtent::Runtime(id)) => Ok(RuntimeScalarExpr::Extent(*id)),
                None => Err(format!("`{name}` is not a shape parameter of the entry")),
            },
            Atom::Quot(numerator, denominator) => Ok(RuntimeScalarExpr::Div(
                Box::new(self.shape_scalar_expr(numerator)?),
                Box::new(self.shape_scalar_expr(denominator)?),
            )),
            Atom::Rem(numerator, denominator) => Ok(RuntimeScalarExpr::Rem(
                Box::new(self.shape_scalar_expr(numerator)?),
                Box::new(self.shape_scalar_expr(denominator)?),
            )),
        }
    }

    /// Convert an interface type of the entry contract: symbolic extents are
    /// entry shape expressions, element parameters resolve in `elems`.
    fn convert_interface_type(
        &mut self,
        ty: &ValueType,
        elems: &ElemEnv,
        context: &str,
    ) -> Result<ValueType, String> {
        let convert_extent = |f: &mut Factory, extent: &ExtentExpr| -> Result<ExtentExpr, String> {
            match extent {
                ExtentExpr::Static(n) => Ok(ExtentExpr::Static(*n)),
                ExtentExpr::Runtime(id) => Ok(ExtentExpr::Runtime(*id)),
                ExtentExpr::Sym(sym) => {
                    if !f.is_entry_shape_sym(sym) {
                        return Err(format!(
                            "{context} depends on `{sym}`, which is not an entry shape expression"
                        ));
                    }
                    Ok(f.shape_extent(sym)?.extent())
                }
            }
        };
        match ty {
            ValueType::Scalar(d) => Ok(ValueType::Scalar(*d)),
            ValueType::Index { bound } => Ok(ValueType::Index {
                bound: convert_extent(self, bound)?,
            }),
            ValueType::Range { bound } => Ok(ValueType::Range {
                bound: convert_extent(self, bound)?,
            }),
            ValueType::Tensor(s) => {
                let mut axes = Vec::new();
                for axis in &s.axes {
                    axes.push(convert_extent(self, axis)?);
                }
                Ok(ValueType::Tensor(s.specialize_elem(
                    axes,
                    convert_elem(&s.elem, elems, context)?,
                )?))
            }
            ValueType::Tuple(items) => Ok(ValueType::Tuple(
                NonEmpty::new(
                    items
                        .iter()
                        .map(|item| self.convert_interface_type(item, elems, context))
                        .collect::<Result<_, _>>()?,
                )
                .expect("the source tuple is nonempty"),
            )),
            ValueType::CapabilityValue(n) => Ok(ValueType::CapabilityValue(n.clone())),
            ValueType::Void => Ok(ValueType::Void),
        }
    }

    fn report(&self) -> ApplicabilityReport {
        ApplicabilityReport {
            entry: self.domain.entry().to_string(),
            occurrences: self.rejections.clone(),
        }
    }
}

fn convert_elem(elem: &Elem, elems: &ElemEnv, context: &str) -> Result<Elem, String> {
    match elem {
        Elem::Param(p) => elems
            .get(p)
            .cloned()
            .ok_or_else(|| format!("element parameter `{p}` of {context} is not bound")),
        Elem::Dtype(_) | Elem::Repr(_) => Ok(elem.clone()),
    }
}

impl<'a> Factory<'a> {
    /// Build the task graph of one alternative of one occurrence by the
    /// forward pass over the definition's checked body. The graph slot is
    /// reserved first so `LogicalAlternative`s can reference the id before
    /// the (nested) construction completes. The graph is sealed with its
    /// boundary before the slot is filled.
    fn build_graph(
        &mut self,
        definition: &Definition,
        shapes: ShapeEnv,
        sym_env: SymEnv,
        elems: ElemEnv,
        choice: ChoiceId,
        alternative: u32,
    ) -> Result<GraphId, BuildError> {
        if self.building.contains(&definition.id) {
            return Err(BuildError::Invalid(format!(
                "`{}` is reachable from its own body: recursion cannot terminate when every occurrence is instantiated",
                definition.name
            )));
        }
        self.building.push(definition.id);
        let graph_id = self.alloc_graph_slot();
        let builder = GraphBuilder::new(choice, alternative, std::mem::take(&mut self.ids));
        let result = self.build_body(definition, shapes, sym_env, elems, builder);
        self.building.pop();
        match result {
            Ok((graph, ids)) => {
                self.ids = ids;
                self.graphs[graph_id.index()] = Some(graph);
                Ok(graph_id)
            }
            Err(reason) => {
                // Construction errors are fatal for the whole program, so the
                // allocator is not recovered.
                if self.no_implementation {
                    self.no_implementation = false;
                    return Err(BuildError::NoImplementation(self.report()));
                }
                Err(BuildError::Invalid(reason))
            }
        }
    }

    fn build_body(
        &mut self,
        definition: &Definition,
        shapes: ShapeEnv,
        sym_env: SymEnv,
        elems: ElemEnv,
        builder: GraphBuilder,
    ) -> Result<(TaskGraph, Ids), String> {
        let body = &definition.body;
        let mut work = Work {
            builder,
            shapes,
            sym_env,
            elems,
            local_tys: &body.locals,
            locals: vec![None; body.locals.len()],
            constants: BTreeMap::new(),
            returned: None,
            exclusive: Vec::new(),
            loops: Vec::new(),
            if_depth: 0,
            extent_memo: BTreeMap::new(),
        };

        // Root region parameters: one value (plus one state for tensors) per
        // canonical interface leaf, and the boundary inputs they originate.
        let mut root_params = Vec::new();
        let mut inputs = BTreeMap::new();
        let mut param_leaves: Vec<(ValueType, BTreeMap<ValuePath, GraphValueId>)> = Vec::new();
        for (ordinal, param) in definition.params.iter().enumerate() {
            let ty = work.convert_type(self, &param.ty)?;
            let mut leaves = BTreeMap::new();
            for (path, leaf) in boundary_leaves(&ty) {
                let key = BoundaryLeaf::Input {
                    param: ordinal as u32,
                    leaf: path.clone(),
                };
                match leaf {
                    LeafKind::Tensor(shape) => {
                        let access = match param.ownership {
                            ParamOwnership::Value => {
                                return Err(format!(
                                    "tensor leaf {path} of parameter `{}` is passed by value; tensor parameters are owned or borrowed",
                                    param.name
                                ));
                            }
                            ParamOwnership::Shared => Access::Shared,
                            ParamOwnership::Owned | ParamOwnership::Exclusive => Access::Exclusive,
                        };
                        let storage = work.builder.declare_storage(
                            shape.clone(),
                            LogicalStorageOwner::Parameter(key.clone()),
                            Initialization::FullyInitialized,
                        );
                        let view = work.builder.declare_view(
                            ViewBase::Storage(storage),
                            shape.clone(),
                            access,
                            ViewTransform::Identity,
                        )?;
                        let value = work.builder.fresh_value(work.builder.view_kind(view)?)?;
                        let token = work.builder.fresh_state(storage)?;
                        root_params.push(RegionParameter::Value {
                            id: value,
                            ty: ValueType::Tensor(shape.clone()),
                        });
                        root_params.push(RegionParameter::State { id: token, storage });
                        if param.ownership == ParamOwnership::Exclusive {
                            work.exclusive.push((key.clone(), storage));
                        }
                        inputs.insert(
                            key,
                            LogicalBoundaryInput::Tensor {
                                value,
                                state: token,
                                ownership: param.ownership,
                            },
                        );
                        leaves.insert(path, value);
                    }
                    LeafKind::Scalar(_)
                    | LeafKind::Index(_)
                    | LeafKind::Range(_)
                    | LeafKind::Capability(_) => {
                        let value = work.builder.fresh_value(leaf.produced_kind())?;
                        root_params.push(RegionParameter::Value {
                            id: value,
                            ty: leaf.ty(),
                        });
                        inputs.insert(key, LogicalBoundaryInput::Value(value));
                        leaves.insert(path, value);
                    }
                }
            }
            param_leaves.push((ty, leaves));
        }
        work.builder.begin_root(root_params)?;

        // Bind each parameter local: leaves are repacked along the type's
        // tuple structure (a single leaf binds directly).
        for (param, (ty, leaves)) in definition.params.iter().zip(&param_leaves) {
            let value = work.repack(ty, &ValuePath::default(), &|path| {
                leaves.get(path).copied()
            }, definition.span)?;
            work.locals[param.local] = Some(value);
        }

        let flow = work.block(self, &body.root)?;
        // A void body may fall through: its final states are its current
        // states without an explicit `return`.
        if !matches!(flow, Flow::Returned) && !definition.result.is_void() {
            return Err(format!(
                "`{}` does not end every path in `return`",
                definition.name
            ));
        }

        // Boundary results: one value per canonical leaf of the result type,
        // decomposed from the returned components. A computed tensor is
        // returned as it is; a view stays a view.
        let result_ty = work.convert_type(self, &definition.result)?;
        let mut results = BTreeMap::new();
        if !result_ty.is_void() {
            let returned = work
                .returned
                .clone()
                .ok_or_else(|| format!("`{}` returns no values", definition.name))?;
            for (path, leaf) in boundary_leaves(&result_ty) {
                let (component, rest) = match &result_ty {
                    ValueType::Tuple(_) => {
                        let index = path.0[0] as usize;
                        let component = returned.get(index).copied().ok_or_else(|| {
                            format!("`{}` returns fewer components than its result", definition.name)
                        })?;
                        (component, &path.0[1..])
                    }
                    ValueType::Scalar(_)
                    | ValueType::Index { .. }
                    | ValueType::Range { .. }
                    | ValueType::Tensor(_)
                    | ValueType::CapabilityValue(_)
                    | ValueType::Void => (
                        returned.first().copied().ok_or_else(|| {
                            format!("`{}` returns no value", definition.name)
                        })?,
                        &path.0[..],
                    ),
                };
                let value = work.decompose_value(component, rest, definition.span)?;
                let actual = work.builder.value_type(value)?;
                if actual != leaf.ty() {
                    return Err(format!(
                        "`{}` returns {actual} at {path} where its result declares {}",
                        definition.name,
                        leaf.ty()
                    ));
                }
                results.insert(
                    BoundaryLeaf::Result { leaf: path },
                    LogicalBoundaryResult { value },
                );
            }
        }
        let mut final_states = BTreeMap::new();
        for (leaf, storage) in &work.exclusive {
            final_states.insert(leaf.clone(), work.builder.current_state(*storage)?);
        }
        work.builder.end_region(Vec::new())?;
        work.builder.seal(inputs, results, final_states)
    }
}

// ---------------------------------------------------------------------------
// Per-graph forward pass
// ---------------------------------------------------------------------------

enum Flow {
    Next,
    Returned,
}

/// Constant-fold one retained runtime expression, eliminating arithmetic
/// identities (`0 + x`, `x + 0`, `1 * x`, `x * 1`) so a pure parameter
/// reference folds to the referenced expression itself.
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
        RuntimeScalarExpr::Add(a, b) => {
            let a = fold_scalar(*a);
            let b = fold_scalar(*b);
            let folded = match (&a, &b) {
                (RuntimeScalarExpr::Const(x), RuntimeScalarExpr::Const(y)) => x.checked_add(*y),
                _ => None,
            };
            match folded {
                Some(value) => RuntimeScalarExpr::Const(value),
                None => match (&a, &b) {
                    (RuntimeScalarExpr::Const(0), _) => b,
                    (_, RuntimeScalarExpr::Const(0)) => a,
                    _ => RuntimeScalarExpr::Add(Box::new(a), Box::new(b)),
                },
            }
        }
        RuntimeScalarExpr::Mul(a, b) => {
            let a = fold_scalar(*a);
            let b = fold_scalar(*b);
            let folded = match (&a, &b) {
                (RuntimeScalarExpr::Const(x), RuntimeScalarExpr::Const(y)) => x.checked_mul(*y),
                _ => None,
            };
            match folded {
                Some(value) => RuntimeScalarExpr::Const(value),
                None => match (&a, &b) {
                    (RuntimeScalarExpr::Const(1), _) => b,
                    (_, RuntimeScalarExpr::Const(1)) => a,
                    _ => RuntimeScalarExpr::Mul(Box::new(a), Box::new(b)),
                },
            }
        }
        RuntimeScalarExpr::Sub(a, b) => binary(
            *a,
            *b,
            |x, y| x.checked_sub(y),
            |a, b| RuntimeScalarExpr::Sub(a, b),
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
        RuntimeScalarExpr::Const(_)
        | RuntimeScalarExpr::Value(_)
        | RuntimeScalarExpr::Extent(_)
        | RuntimeScalarExpr::ShapeField(_) => expr,
    }
}

/// What one symbolic atom resolves to in the current environment.
enum ResolvedAtom {
    Value(GraphValueId),
    Shape(ShapeExtent),
}

#[derive(Default, Clone, Copy)]
struct LoopWrite {
    atomic: bool,
    plain: bool,
}

struct LoopInfo {
    binder: GraphValueId,
    /// The binder's local identity: nested body regions rebind the binder
    /// (and captured outer binders) to fresh region parameters, so a write
    /// inside a nested region addresses the loop binder through the local's
    /// current value, not the original binder value.
    binder_local: LocalId,
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
    builder: GraphBuilder,
    shapes: ShapeEnv,
    sym_env: SymEnv,
    elems: ElemEnv,
    /// Checked local types (indexed by `LocalId`).
    local_tys: &'w [crate::sir::CheckedLocal],
    /// Lexical environment: local -> current value; `None` before binding.
    locals: Vec<Option<GraphValueId>>,
    /// Integer constants by producing value, for static coverage proofs.
    constants: BTreeMap<GraphValueId, i64>,
    /// The returned top-level components once a `return` executed on the
    /// current path.
    returned: Option<Vec<GraphValueId>>,
    /// Exclusively borrowed tensor parameter leaves with their storages; the
    /// boundary records their final states.
    exclusive: Vec<(BoundaryLeaf, LogicalStorageId)>,
    loops: Vec<LoopInfo>,
    if_depth: usize,
    /// Runtime extents of value-dependent symbolic extents of this graph.
    extent_memo: BTreeMap<String, RuntimeExtentId>,
}

/// The interval of one leaf or operation of a retained runtime scalar, and
/// its upper bound as a capacity. Total over `RuntimeScalarExpr`: every leaf
/// has a representation-derived interval (a constant is exact, a runtime
/// extent spans `[0, capacity]`, a shape field spans its finite domain, a
/// graph value spans its kind's representation: `i32`/`u32` scalars their
/// full range, a bounded index `[0, capacity(bound) - 1]`), and every
/// operation propagates with checked `i128` arithmetic over the defined
/// domain (a divisor interval admitting zero is clamped to its positive part;
/// the source's `DivisorNonZero` obligation guarantees definedness at
/// runtime). An `Err` names a construction defect: a leaf without a
/// representation range, a divisor or modulus that is never positive, or a
/// bound outside `u64`.
fn runtime_scalar_bound(
    expr: &RuntimeScalarExpr,
    extents: &[RuntimeExtent],
    fields: &[ShapeField],
    value_kind: &dyn Fn(GraphValueId) -> Result<GraphValueKind, String>,
) -> Result<u64, String> {
    let (_, upper) = runtime_scalar_interval(expr, extents, fields, value_kind)?;
    u64::try_from(upper).map_err(|_| format!("its bound {upper} is outside u64"))
}

fn runtime_scalar_interval(
    expr: &RuntimeScalarExpr,
    extents: &[RuntimeExtent],
    fields: &[ShapeField],
    value_kind: &dyn Fn(GraphValueId) -> Result<GraphValueKind, String>,
) -> Result<(i128, i128), String> {
    fn field_bounds(id: ShapeFieldId, fields: &[ShapeField]) -> Result<(i128, i128), String> {
        let field = fields
            .get(id.index())
            .ok_or_else(|| format!("shape field {} does not exist", id.0))?;
        Ok((
            i128::from(field.domain.min()),
            i128::from(field.domain.max()),
        ))
    }
    fn extent_capacity(extent: &ExtentExpr, extents: &[RuntimeExtent]) -> Result<u64, String> {
        match extent {
            ExtentExpr::Static(n) => Ok(*n),
            ExtentExpr::Runtime(id) => extents
                .get(id.0 as usize)
                .map(|extent| extent.capacity)
                .ok_or_else(|| format!("runtime extent {} does not exist", id.0)),
            ExtentExpr::Sym(sym) => Err(format!(
                "index bound `{sym}` is still symbolic at the logical level"
            )),
        }
    }
    let binary = |left: &RuntimeScalarExpr, right: &RuntimeScalarExpr| {
        Ok::<_, String>((
            runtime_scalar_interval(left, extents, fields, value_kind)?,
            runtime_scalar_interval(right, extents, fields, value_kind)?,
        ))
    };
    let overflow = || "the interval arithmetic overflows".to_string();
    match expr {
        RuntimeScalarExpr::Const(value) => {
            let value = i128::from(*value);
            Ok((value, value))
        }
        RuntimeScalarExpr::Value(id) => match value_kind(*id)? {
            GraphValueKind::Scalar(DType::I32) => {
                Ok((i128::from(i32::MIN), i128::from(i32::MAX)))
            }
            GraphValueKind::Scalar(DType::U32) => Ok((0, i128::from(u32::MAX))),
            GraphValueKind::Index { bound } => {
                // `0 <= i < capacity(bound)`; an empty index domain is
                // vacuous and contributes `[0, 0]`.
                let capacity = extent_capacity(&bound, extents)?;
                Ok((0, i128::from(capacity.saturating_sub(1))))
            }
            GraphValueKind::Scalar(DType::F32)
            | GraphValueKind::Scalar(DType::BF16)
            | GraphValueKind::Scalar(DType::F16)
            | GraphValueKind::Scalar(DType::Bool)
            | GraphValueKind::Void
            | GraphValueKind::Range { .. }
            | GraphValueKind::Tuple(_)
            | GraphValueKind::Capability(_)
            | GraphValueKind::Tensor { .. } => Err(format!(
                "graph value {} is not an integer scalar or index",
                id.0
            )),
        },
        RuntimeScalarExpr::ShapeField(id) => field_bounds(*id, fields),
        RuntimeScalarExpr::Extent(id) => {
            let extent = extents
                .get(id.0 as usize)
                .ok_or_else(|| format!("runtime extent {} does not exist", id.0))?;
            match &extent.value {
                RuntimeScalarExpr::ShapeField(field) => field_bounds(*field, fields),
                RuntimeScalarExpr::Const(_)
                | RuntimeScalarExpr::Value(_)
                | RuntimeScalarExpr::Extent(_)
                | RuntimeScalarExpr::Add(..)
                | RuntimeScalarExpr::Sub(..)
                | RuntimeScalarExpr::Mul(..)
                | RuntimeScalarExpr::Div(..)
                | RuntimeScalarExpr::Rem(..) => Ok((0, i128::from(extent.capacity))),
            }
        }
        RuntimeScalarExpr::Add(left, right) => {
            let ((left_min, left_max), (right_min, right_max)) = binary(left, right)?;
            Ok((
                left_min.checked_add(right_min).ok_or_else(overflow)?,
                left_max.checked_add(right_max).ok_or_else(overflow)?,
            ))
        }
        RuntimeScalarExpr::Sub(left, right) => {
            let ((left_min, left_max), (right_min, right_max)) = binary(left, right)?;
            Ok((
                left_min.checked_sub(right_max).ok_or_else(overflow)?,
                left_max.checked_sub(right_min).ok_or_else(overflow)?,
            ))
        }
        RuntimeScalarExpr::Mul(left, right) => {
            let ((left_min, left_max), (right_min, right_max)) = binary(left, right)?;
            let products = [
                left_min.checked_mul(right_min).ok_or_else(overflow)?,
                left_min.checked_mul(right_max).ok_or_else(overflow)?,
                left_max.checked_mul(right_min).ok_or_else(overflow)?,
                left_max.checked_mul(right_max).ok_or_else(overflow)?,
            ];
            Ok((
                *products.iter().min().expect("four products"),
                *products.iter().max().expect("four products"),
            ))
        }
        RuntimeScalarExpr::Div(left, right) => {
            // The bound is over the defined domain: the source's
            // `DivisorNonZero` obligation (retained on the node) guarantees a
            // nonzero divisor at runtime, so a divisor interval that merely
            // admits zero (an `index[N]` is `[0, N-1]`) is clamped to
            // `[max(min, 1), max]`. Only a divisor that is never positive is
            // a construction error.
            let ((left_min, left_max), (right_min, right_max)) = binary(left, right)?;
            if right_max <= 0 {
                return Err(format!(
                    "the divisor of `{}` is never positive",
                    describe(expr)
                ));
            }
            let divisor_min = right_min.max(1);
            let quotients = [
                left_min / divisor_min,
                left_min / right_max,
                left_max / divisor_min,
                left_max / right_max,
            ];
            Ok((
                *quotients.iter().min().expect("four quotients"),
                *quotients.iter().max().expect("four quotients"),
            ))
        }
        RuntimeScalarExpr::Rem(left, right) => {
            // Euclidean remainder by a positive divisor `d` lies in
            // `[0, d - 1]` for any numerator, and never exceeds a
            // nonnegative numerator. The divisor is clamped as for `Div`.
            let ((left_min, left_max), (_, right_max)) = binary(left, right)?;
            if right_max <= 0 {
                return Err(format!(
                    "the modulus of `{}` is never positive",
                    describe(expr)
                ));
            }
            let upper = if left_min >= 0 {
                left_max.min(right_max - 1)
            } else {
                right_max - 1
            };
            Ok((0, upper))
        }
    }
}

/// A readable rendering of one retained runtime scalar, for diagnostics.
fn describe(expr: &RuntimeScalarExpr) -> String {
    match expr {
        RuntimeScalarExpr::Const(value) => value.to_string(),
        RuntimeScalarExpr::Value(id) => format!("value#{}", id.0),
        RuntimeScalarExpr::Extent(id) => format!("runtime#{}", id.0),
        RuntimeScalarExpr::ShapeField(id) => format!("shape#{}", id.0),
        RuntimeScalarExpr::Add(a, b) => format!("({} + {})", describe(a), describe(b)),
        RuntimeScalarExpr::Sub(a, b) => format!("({} - {})", describe(a), describe(b)),
        RuntimeScalarExpr::Mul(a, b) => format!("({} * {})", describe(a), describe(b)),
        RuntimeScalarExpr::Div(a, b) => format!("({} / {})", describe(a), describe(b)),
        RuntimeScalarExpr::Rem(a, b) => format!("({} % {})", describe(a), describe(b)),
    }
}

impl<'w> Work<'w> {
    fn add(&mut self, spec: PrimitiveSpec) -> Result<Option<GraphValueId>, String> {
        let outcome = self.builder.add_primitive(spec)?;
        Ok(outcome.outputs.first().copied())
    }

    fn binding(&self, local: LocalId) -> Result<GraphValueId, String> {
        self.locals
            .get(local)
            .and_then(|binding| *binding)
            .ok_or_else(|| format!("local `{}` is not bound", self.local_tys[local].name))
    }

    /// The view a local's current value reads through: `Some` exactly for a
    /// view-backed tensor local (of storage or of a computed value).
    fn view_of_local(&self, local: LocalId) -> Result<Option<LogicalViewId>, String> {
        let value = self.binding(local)?;
        Ok(match self.builder.value(value)?.tensor_source() {
            Some(TensorSource::View(view)) => Some(view),
            Some(TensorSource::Computed) | None => None,
        })
    }

    /// The storage a local's current value reads through its view: `Some`
    /// exactly when the view's base is logical storage. A computed tensor
    /// and a view of one read no storage.
    fn storage_of_local(&self, local: LocalId) -> Result<Option<LogicalStorageId>, String> {
        Ok(match self.view_of_local(local)? {
            Some(view) => match self.builder.view(view).base {
                ViewBase::Storage(storage) => Some(storage),
                ViewBase::Value(_) => None,
            },
            None => None,
        })
    }

    /// Whether a tensor value has no logical storage behind it: a computed
    /// tensor, directly or through views of one.
    fn is_unstored_tensor(&self, value: GraphValueId) -> Result<bool, String> {
        Ok(match self.builder.value(value)?.tensor_source() {
            Some(TensorSource::Computed) => true,
            Some(TensorSource::View(view)) => {
                matches!(self.builder.view(view).base, ViewBase::Value(_))
            }
            None => false,
        })
    }

    // -- symbolic extent resolution ----------------------------------------

    /// Convert a checked type to the logical level: `Sym` extents become
    /// `Static` or `Runtime` extents, element parameters resolve.
    fn convert_type(&mut self, f: &mut Factory, ty: &ValueType) -> Result<ValueType, String> {
        match ty {
            ValueType::Scalar(d) => Ok(ValueType::Scalar(*d)),
            ValueType::Index { bound } => Ok(ValueType::Index {
                bound: self.convert_extent(f, bound)?,
            }),
            ValueType::Range { bound } => Ok(ValueType::Range {
                bound: self.convert_extent(f, bound)?,
            }),
            ValueType::Tensor(s) => Ok(ValueType::Tensor(s.specialize_elem(
                s.axes
                    .iter()
                    .map(|axis| self.convert_extent(f, axis))
                    .collect::<Result<_, _>>()?,
                self.convert_elem(&s.elem)?,
            )?)),
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
            ExtentExpr::Sym(sym) => Ok(self.resolve_sym(f, sym)?.extent()),
        }
    }

    fn convert_elem(&self, elem: &Elem) -> Result<Elem, String> {
        convert_elem(elem, &self.elems, "this occurrence")
    }

    /// Resolve one symbolic extent. A pure shape expression is rewritten into
    /// the entry's symbol space and resolved program-wide; a value-dependent
    /// expression resolves against the current graph values (allocating a
    /// runtime extent memoized per graph).
    fn resolve_sym(&mut self, f: &mut Factory, sym: &Sym) -> Result<ShapeExtent, String> {
        if let Some(value) = sym.as_constant() {
            return static_extent(value);
        }
        let entry_form = family::substitute(sym, &|name| self.sym_env.get(name).cloned());
        if f.is_entry_shape_sym(&entry_form) {
            return f.shape_extent(&entry_form);
        }
        let memo_key = self.sym_key(sym)?;
        if let Some(id) = self.extent_memo.get(&memo_key) {
            return Ok(ShapeExtent::Runtime(*id));
        }
        let expr = self.scalar_expr(f, sym)?;
        let resolved = match expr {
            RuntimeScalarExpr::Const(value) => static_extent(value)?,
            RuntimeScalarExpr::Extent(id) => ShapeExtent::Runtime(id),
            expr => {
                // The retained expression itself is the bound authority: its
                // extent leaves carry their capacities and its value leaves
                // carry their representation ranges (i32/u32 scalars, bounded
                // indices), so the bound is derived, never substituted.
                let builder = &self.builder;
                let value_kind = |id: GraphValueId| -> Result<GraphValueKind, String> {
                    Ok(builder.value(id)?.kind().clone())
                };
                let capacity = runtime_scalar_bound(
                    &expr,
                    &f.runtime_extents,
                    &f.shape_fields,
                    &value_kind,
                )
                .map_err(|reason| {
                    format!("extent `{sym}` has no finite bound because {reason}")
                })?;
                ShapeExtent::Runtime(f.runtime_extent(expr, capacity, None))
            }
        };
        if let ShapeExtent::Runtime(id) = resolved {
            self.extent_memo.insert(memo_key, id);
        }
        Ok(resolved)
    }

    /// A memo key for one symbolic extent: its display plus the current value
    /// ids of every value atom it mentions (SSA rebinding changes the key).
    fn sym_key(&self, sym: &Sym) -> Result<String, String> {
        let mut key = sym.to_string();
        for name in sym.params() {
            if let Some(local) = atom_var(&name) {
                let value = self.binding(local)?;
                key.push_str(&format!("#{}", value.0));
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
        self.shapes.get(name).copied().map(ResolvedAtom::Shape)
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
                Some(ResolvedAtom::Shape(ShapeExtent::Static(n))) => {
                    Ok(RuntimeScalarExpr::Const(n as i64))
                }
                Some(ResolvedAtom::Shape(ShapeExtent::Runtime(id))) => {
                    Ok(RuntimeScalarExpr::Extent(id))
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
                if values.is_empty() {
                    // Either a void return, or the checker's marker after a
                    // terminal `if` whose arms both returned (the joined
                    // values are already installed).
                    if self.returned.is_none() {
                        self.returned = Some(Vec::new());
                    }
                } else {
                    let mut returned = Vec::new();
                    for value in values {
                        returned.push(
                            self.expr(f, value)?
                                .ok_or("a returned expression is void")?,
                        );
                    }
                    self.returned = Some(returned);
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
                mutable: _,
            } => {
                let value = self.expr(f, value)?.ok_or("`let` binds a void expression")?;
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
                self.if_core(f, condition, then_body, else_body, None)?;
                Ok(Flow::Next)
            }
            CheckedStmt::Evaluate(expr) => {
                if let Some(value) = self.expr(f, expr)? {
                    return Err(format!(
                        "an expression statement must be void; this one produces {}",
                        self.builder.value_type(value)?
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
                    let component = self.tuple_get(f, value, index, Span::default())?;
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
        span: Span,
    ) -> Result<GraphValueId, String> {
        let _ = f;
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
            outputs: vec![Output::Computed(component)],
            safety: Vec::new(),
            span,
        };
        Ok(self.add(spec)?.expect("tuple.get has a value"))
    }

    /// Decompose one value along an ordinal path with `tuple.get` nodes.
    fn decompose_value(
        &mut self,
        value: GraphValueId,
        path: &[u32],
        span: Span,
    ) -> Result<GraphValueId, String> {
        let mut current = value;
        for index in path {
            let ty = self.builder.value_type(current)?;
            let ValueType::Tuple(components) = &ty else {
                return Err("a boundary path descends into a non-tuple".into());
            };
            let component = components
                .as_slice()
                .get(*index as usize)
                .cloned()
                .ok_or("a boundary path is out of bounds")?;
            let spec = PrimitiveSpec {
                op: PrimitiveOp::Primitive(PrimitiveId::TupleGet(*index as usize)),
                inputs: vec![current],
                reads: Vec::new(),
                write: None,
                outputs: vec![Output::Computed(component)],
                safety: Vec::new(),
                span,
            };
            current = self.add(spec)?.expect("tuple.get has a value");
        }
        Ok(current)
    }

    /// Rebuild a value of type `ty` from its canonical leaves: tuples pack
    /// their components, a leaf is looked up at its path.
    fn repack(
        &mut self,
        ty: &ValueType,
        path: &ValuePath,
        leaf: &dyn Fn(&ValuePath) -> Option<GraphValueId>,
        span: Span,
    ) -> Result<GraphValueId, String> {
        match ty {
            ValueType::Tuple(items) => {
                let mut components = Vec::new();
                for (index, item) in items.iter().enumerate() {
                    components.push(self.repack(item, &path.extend(index as u32), leaf, span)?);
                }
                let spec = PrimitiveSpec {
                    op: PrimitiveOp::Primitive(PrimitiveId::TuplePack),
                    inputs: components,
                    reads: Vec::new(),
                    write: None,
                    outputs: vec![Output::Computed(ty.clone())],
                    safety: Vec::new(),
                    span,
                };
                Ok(self.add(spec)?.expect("tuple.pack has a value"))
            }
            ValueType::Void => Err("a void value cannot be repacked".into()),
            ValueType::Scalar(_)
            | ValueType::Index { .. }
            | ValueType::Range { .. }
            | ValueType::Tensor(_)
            | ValueType::CapabilityValue(_) => {
                leaf(path).ok_or_else(|| format!("no boundary leaf value at {path}"))
            }
        }
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
                        Some(ShapeExtent::Static(n)) => {
                            PrimitiveOp::Constant(Literal::Int(*n as i64))
                        }
                        Some(ShapeExtent::Runtime(id)) => PrimitiveOp::RuntimeExtent(*id),
                        None => {
                            return Err(format!(
                                "shape parameter `{name}` is not bound at this occurrence"
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
                    outputs: vec![Output::Computed(ty)],
                    safety: Vec::new(),
                    span: e.span,
                };
                let out = self.add(spec)?;
                if let (Some(id), Literal::Int(value)) = (out, literal) {
                    self.constants.insert(id, *value);
                }
                Ok(out)
            }
            CheckedExprKind::Local(id) => Ok(Some(self.binding(*id)?)),
            CheckedExprKind::Primitive { id, operands } => self.primitive(f, e, id, operands),
            CheckedExprKind::Capability { id, args } => {
                let mut inputs = Vec::new();
                for arg in args {
                    inputs.push(
                        self.expr(f, arg)?
                            .ok_or("a capability argument is void")?,
                    );
                }
                let reads = self.builder.reads_of(&inputs)?;
                let ty = self.convert_type(f, &e.ty)?;
                let outputs = if ty.is_void() {
                    Vec::new()
                } else {
                    vec![Output::Computed(ty)]
                };
                let spec = PrimitiveSpec {
                    op: PrimitiveOp::Capability(id.clone()),
                    inputs,
                    reads,
                    write: None,
                    outputs,
                    safety: Vec::new(),
                    span: e.span,
                };
                self.add(spec)
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
            out.push(self.expr(f, operand)?.ok_or("a primitive operand is void")?);
        }
        Ok(out)
    }

    /// Fresh local storage with an identity view, for allocation-family
    /// primitives and materialization.
    fn fresh_local(&mut self, shape: TensorType) -> Result<(LogicalStorageId, LogicalViewId), String> {
        let storage = self.builder.declare_storage(
            shape.clone(),
            LogicalStorageOwner::Local,
            Initialization::Uninitialized,
        );
        let view = self
            .builder
            .declare_view(ViewBase::Storage(storage), shape, Access::Exclusive, ViewTransform::Identity)?;
        Ok((storage, view))
    }

    /// The base a view-producing or element-reading operand works over: one
    /// logical storage read through a view, or a computed tensor value read
    /// as a value (directly or through a view of one). A computed operand
    /// never declares storage; a derived view over it names the value as its
    /// base.
    fn operand_base(&self, value: GraphValueId) -> Result<OperandBase, String> {
        let (base, access, shape) = match self.builder.value(value)?.kind() {
            GraphValueKind::Tensor { ty, source } => match source {
                TensorSource::View(view) => {
                    let declared = self.builder.view(*view);
                    match declared.base {
                        ViewBase::Storage(storage) => (
                            ViewBase::Storage(storage),
                            declared.access,
                            declared.shape.clone(),
                        ),
                        // A view of a view flattens to the operand's base.
                        ViewBase::Value(base) => {
                            (ViewBase::Value(base), Access::Shared, declared.shape.clone())
                        }
                    }
                }
                TensorSource::Computed => {
                    (ViewBase::Value(value), Access::Shared, ty.clone())
                }
            },
            kind => {
                return Err(format!(
                    "operand {} is not a tensor ({:?})",
                    value.0, kind
                ))
            }
        };
        Ok(OperandBase {
            base,
            access,
            shape,
        })
    }

    /// The storage a mutating operand (an element-write place, a copy
    /// destination or an atomic base) writes through. An in-place write
    /// names mutable state: a tensor value without storage behind it is
    /// realized into its own local storage here, and when the operand
    /// expression is a bare local the local rebinds to it so later in-place
    /// writes and reads through it see one storage. This is the one
    /// materialization construction performs; reads and call arguments never
    /// reach it.
    fn ensure_storage_operand(
        &mut self,
        value: GraphValueId,
        operand: &CheckedExpr,
        span: Span,
    ) -> Result<(GraphValueId, LogicalViewId, LogicalStorageId), String> {
        let unstored = self.is_unstored_tensor(value)?;
        if !unstored {
            let view = match self.builder.tensor_source(value)? {
                TensorSource::View(view) => view,
                TensorSource::Computed => {
                    return Err("the write target is a computed tensor".into())
                }
            };
            let storage = self.builder.storage_base(view)?;
            return Ok((value, view, storage));
        }
        let (realized, view) = self.materialize_value(value, span)?;
        let storage = self.builder.storage_base(view)?;
        if let CheckedExprKind::Local(id) = &operand.kind {
            self.locals[*id] = Some(realized);
        }
        Ok((realized, view, storage))
    }

    /// The storage a local's in-place write targets, realizing and rebinding
    /// a tensor value without storage behind it (a computed tensor or a view
    /// of one).
    fn ensure_view_local(
        &mut self,
        local: LocalId,
        span: Span,
    ) -> Result<(GraphValueId, LogicalViewId, LogicalStorageId), String> {
        let value = self.binding(local)?;
        if self.is_unstored_tensor(value)? {
            let (realized, view) = self.materialize_value(value, span)?;
            let storage = self.builder.storage_base(view)?;
            self.locals[local] = Some(realized);
            Ok((realized, view, storage))
        } else {
            let view = match self.builder.tensor_source(value)? {
                TensorSource::View(view) => view,
                TensorSource::Computed => {
                    return Err("the write target is a computed tensor".into())
                }
            };
            let storage = self.builder.storage_base(view)?;
            Ok((value, view, storage))
        }
    }

    /// Realize one tensor value without storage behind it into fresh local
    /// storage, through one `tensor.materialize` application. The only caller
    /// is in-place-write realization; an authored `to_owned` reaches the same
    /// primitive from the source.
    fn materialize_value(
        &mut self,
        value: GraphValueId,
        span: Span,
    ) -> Result<(GraphValueId, LogicalViewId), String> {
        let ValueType::Tensor(shape) = self.builder.value_type(value)? else {
            return Err("materialize needs a tensor".into());
        };
        let (storage, view) = self.fresh_local(shape.clone())?;
        let reads = self.builder.reads_of(&[value])?;
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
        let materialized = self.add(spec)?.expect("materialize produces a value");
        Ok((materialized, view))
    }

    /// Realize every local about to be written in place whose current value
    /// has no storage behind it, so the storage exists outside the construct
    /// that writes it.
    fn materialize_written_locals(
        &mut self,
        locals: impl IntoIterator<Item = LocalId>,
        span: Span,
    ) -> Result<(), String> {
        for local in locals {
            let value = self.binding(local)?;
            if self.is_unstored_tensor(value)? {
                self.ensure_view_local(local, span)?;
            }
        }
        Ok(())
    }
}

/// The base of one tensor operand of a view-producing or element-reading
/// primitive, with the access a derived view carries and the shape the
/// operand selects.
struct OperandBase {
    base: ViewBase,
    access: Access,
    shape: TensorType,
}

impl<'w> Work<'w> {
    /// One primitive application. The tensor-kind decision is made here per
    /// `PrimitiveId`: a primitive produces a view only where its semantics
    /// name storage (allocation-family results, materialization, duplication)
    /// or transform an operand (transpose, reshape, slice) — over a view of
    /// storage, sharing that storage, or over a computed tensor, naming the
    /// value as the new view's base; every other tensor result is a computed
    /// value without storage.
    #[allow(clippy::too_many_lines)]
    fn primitive(
        &mut self,
        f: &mut Factory,
        e: &CheckedExpr,
        id: &PrimitiveId,
        operands: &[CheckedExpr],
    ) -> Result<Option<GraphValueId>, String> {
        let mut values = self.operand_values(f, operands)?;
        // A slice view's realized axes are built by its own arm; converting the
        // checked type up front would resolve the runtime-length atom before
        // the view that defines it exists.
        let ty = match id {
            PrimitiveId::SliceView { .. } => ValueType::Void,
            PrimitiveId::TuplePack
            | PrimitiveId::TupleGet(_)
            | PrimitiveId::RangeMake
            | PrimitiveId::RangeStart
            | PrimitiveId::RangeEnd
            | PrimitiveId::Unary(_)
            | PrimitiveId::Binary(_)
            | PrimitiveId::Cast(_)
            | PrimitiveId::Math(_)
            | PrimitiveId::Select
            | PrimitiveId::TensorAlloc { .. }
            | PrimitiveId::Fill { .. }
            | PrimitiveId::Materialize
            | PrimitiveId::Clone
            | PrimitiveId::Load
            | PrimitiveId::Decode
            | PrimitiveId::PackedRead(_)
            | PrimitiveId::Transpose
            | PrimitiveId::Reshape
            | PrimitiveId::ElementRead { .. }
            | PrimitiveId::ElementWrite { .. }
            | PrimitiveId::CopyInto
            | PrimitiveId::Extent { .. }
            | PrimitiveId::ValidExtent { .. }
            | PrimitiveId::Atomic { .. }
            | PrimitiveId::Reduce { .. } => self.convert_type(f, &e.ty)?,
        };
        let span = e.span;
        match id {
            PrimitiveId::TuplePack => {
                let spec = PrimitiveSpec {
                    op: PrimitiveOp::Primitive(id.clone()),
                    inputs: values,
                    reads: Vec::new(),
                    write: None,
                    outputs: vec![Output::Computed(ty)],
                    safety: Vec::new(),
                    span,
                };
                self.add(spec)
            }
            PrimitiveId::TupleGet(_) => {
                let spec = PrimitiveSpec {
                    op: PrimitiveOp::Primitive(id.clone()),
                    inputs: values,
                    reads: Vec::new(),
                    write: None,
                    outputs: vec![Output::Computed(ty)],
                    safety: Vec::new(),
                    span,
                };
                self.add(spec)
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
                    outputs: vec![Output::Computed(ty)],
                    safety,
                    span,
                };
                self.add(spec)
            }
            PrimitiveId::RangeStart | PrimitiveId::RangeEnd => {
                let spec = PrimitiveSpec {
                    op: PrimitiveOp::Primitive(id.clone()),
                    inputs: values,
                    reads: Vec::new(),
                    write: None,
                    outputs: vec![Output::Computed(ty)],
                    safety: Vec::new(),
                    span,
                };
                self.add(spec)
            }
            PrimitiveId::Unary(_) | PrimitiveId::Cast(_) | PrimitiveId::Math(_) => {
                // A tensor-level cast of a packed operand decodes first: the
                // decoded f32 elements then cast elementwise.
                if let (PrimitiveId::Cast(_), Some(first)) = (id, values.first().copied()) {
                    if let ValueType::Tensor(source) = self.builder.value_type(first)? {
                        if matches!(source.elem, Elem::Repr(_)) {
                            let decoded_ty = ValueType::Tensor(TensorType {
                                elem: Elem::Dtype(DType::F32),
                                axes: source.axes,
                                packed_axis: None,
                            });
                            let spec = PrimitiveSpec {
                                op: PrimitiveOp::Primitive(PrimitiveId::Decode),
                                inputs: vec![first],
                                reads: self.builder.reads_of(&[first])?,
                                write: None,
                                outputs: vec![Output::Computed(decoded_ty)],
                                safety: Vec::new(),
                                span,
                            };
                            values[0] = self
                                .add(spec)?
                                .ok_or("packed tensor decoding did not produce a logical value")?;
                        }
                    }
                }
                let reads = self.builder.reads_of(&values)?;
                let spec = PrimitiveSpec {
                    op: PrimitiveOp::Primitive(id.clone()),
                    inputs: values,
                    reads,
                    write: None,
                    outputs: vec![Output::Computed(ty)],
                    safety: Vec::new(),
                    span,
                };
                self.add(spec)
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
                let reads = self.builder.reads_of(&values)?;
                let spec = PrimitiveSpec {
                    op: PrimitiveOp::Primitive(id.clone()),
                    inputs: values,
                    reads,
                    write: None,
                    outputs: vec![Output::Computed(ty)],
                    safety,
                    span,
                };
                self.add(spec)
            }
            PrimitiveId::Select => {
                let reads = self.builder.reads_of(&values)?;
                let spec = PrimitiveSpec {
                    op: PrimitiveOp::Primitive(id.clone()),
                    inputs: values,
                    reads,
                    write: None,
                    outputs: vec![Output::Computed(ty)],
                    safety: Vec::new(),
                    span,
                };
                self.add(spec)
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
                        ExtentExpr::Sym(_) | ExtentExpr::Runtime(_) => None,
                    });
                    if !matches!(product, Some(p) if p <= u64::MAX as u128) {
                        safety.push(SafetyObligation::ShapeProductFits {
                            factors: shape.axes.clone(),
                            bits: 64,
                        });
                    }
                }
                let (storage, view) = self.fresh_local(shape.clone())?;
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
                self.add(spec)
            }
            PrimitiveId::Fill { .. } | PrimitiveId::Materialize | PrimitiveId::Clone => {
                // Allocation family: the result names fresh local storage,
                // fully written by the primitive.
                let ValueType::Tensor(shape) = &ty else {
                    return Err("an allocating primitive produces a tensor".into());
                };
                let reads = self.builder.reads_of(&values)?;
                let (storage, view) = self.fresh_local(shape.clone())?;
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
                self.add(spec)
            }
            PrimitiveId::Load | PrimitiveId::Decode | PrimitiveId::PackedRead(_) => {
                // Snapshot, decode and plane reads produce values.
                if !matches!(ty, ValueType::Tensor(_)) {
                    return Err("a tensor-reading primitive produces a tensor".into());
                }
                let reads = self.builder.reads_of(&values)?;
                let spec = PrimitiveSpec {
                    op: PrimitiveOp::Primitive(id.clone()),
                    inputs: values,
                    reads,
                    write: None,
                    outputs: vec![Output::Computed(ty)],
                    safety: Vec::new(),
                    span,
                };
                self.add(spec)
            }
            PrimitiveId::Transpose => {
                let base = self.operand_base(values[0])?;
                let rank = base.shape.rank();
                let permutation: Vec<u32> = (0..rank as u32).rev().collect();
                let ValueType::Tensor(shape) = ty else {
                    return Err("transpose produces a tensor".into());
                };
                let new_view = self.builder.declare_view(
                    base.base,
                    shape,
                    base.access,
                    ViewTransform::Transpose { permutation },
                )?;
                let spec = PrimitiveSpec {
                    op: PrimitiveOp::Primitive(id.clone()),
                    inputs: values,
                    reads: Vec::new(),
                    write: None,
                    outputs: vec![Output::View(new_view)],
                    safety: Vec::new(),
                    span,
                };
                self.add(spec)
            }
            PrimitiveId::Reshape => {
                let base = self.operand_base(values[0])?;
                let source_shape = base.shape.axes.clone();
                let ValueType::Tensor(shape) = ty else {
                    return Err("reshape produces a tensor".into());
                };
                let new_view = self.builder.declare_view(
                    base.base,
                    shape,
                    base.access,
                    ViewTransform::Reshape { source_shape },
                )?;
                let spec = PrimitiveSpec {
                    op: PrimitiveOp::Primitive(id.clone()),
                    inputs: values,
                    reads: Vec::new(),
                    write: None,
                    outputs: vec![Output::View(new_view)],
                    safety: Vec::new(),
                    span,
                };
                self.add(spec)
            }
            PrimitiveId::SliceView { indices } => {
                let base = self.operand_base(values[0])?;
                let base_axes = base.shape.axes.clone();
                if indices.len() > base_axes.len() {
                    return Err(format!(
                        "slice indexes {} axes of a rank-{} source",
                        indices.len(),
                        base_axes.len()
                    ));
                }
                let mut axes = Vec::new();
                let mut safety = Vec::new();
                let mut cursor = 1; // operands after the base
                for slot in indices {
                    match slot {
                        IndexSlot::Point => {
                            axes.push(SliceAxis::Point(values[cursor]));
                            safety.push(SafetyObligation::IndexInBounds {
                                index: values[cursor],
                                extent: base_axes[axes.len() - 1].clone(),
                            });
                            cursor += 1;
                        }
                        IndexSlot::Range { start, end } => {
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
                        IndexSlot::Point => {
                            axis_cursor += 1;
                        }
                        IndexSlot::Range { .. } => {
                            let (start, end) = range_values
                                .next()
                                .expect("every range slot has a range axis");
                            let base_axis = base_axes[axis_cursor].clone();
                            let extent = match (start, end) {
                                (None, None) => base_axis,
                                (Some(start), Some(end)) => {
                                    let expr = RuntimeScalarExpr::Sub(
                                        Box::new(RuntimeScalarExpr::Value(end)),
                                        Box::new(RuntimeScalarExpr::Value(start)),
                                    );
                                    // A sliced range is bounded by the axis it
                                    // selects from.
                                    let capacity = match &base_axis {
                                        ExtentExpr::Static(value) => *value,
                                        ExtentExpr::Runtime(id) => f.runtime_extents[id.index()].capacity,
                                        ExtentExpr::Sym(sym) => {
                                            return Err(format!(
                                                "slice source axis `{sym}` is still symbolic at the logical level"
                                            ))
                                        }
                                    };
                                    let id = f.runtime_extent(expr, capacity, None);
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
                                (Some(_), None) | (None, Some(_)) => {
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
                while axes.len() < base_axes.len() {
                    axes.push(SliceAxis::Full);
                }
                let packed_axis = slice_packed_axis(&base.shape, &axes);
                let new_view = self.builder.declare_view(
                    base.base,
                    TensorType {
                        axes: shape_axes,
                        elem: base.shape.elem.clone(),
                        packed_axis,
                    },
                    base.access,
                    ViewTransform::Slice { axes },
                )?;
                let spec = PrimitiveSpec {
                    op: PrimitiveOp::Primitive(id.clone()),
                    inputs: values,
                    reads: Vec::new(),
                    write: None,
                    outputs: vec![Output::View(new_view)],
                    safety,
                    span,
                };
                self.add(spec)
            }
            PrimitiveId::ElementRead { arity } => {
                // A computed tensor (directly or through a view of one) is
                // read as a value; only a view of storage reads storage.
                let base = self.operand_base(values[0])?;
                if *arity > base.shape.axes.len() || values.len() < 1 + arity {
                    return Err(format!(
                        "element read indexes {arity} axes of a rank-{} source",
                        base.shape.axes.len()
                    ));
                }
                let mut safety = Vec::new();
                for (axis, index) in values[1..1 + arity].iter().enumerate() {
                    safety.push(SafetyObligation::IndexInBounds {
                        index: *index,
                        extent: base.shape.axes[axis].clone(),
                    });
                }
                let reads = self.builder.reads_of(&values)?;
                let spec = PrimitiveSpec {
                    op: PrimitiveOp::Primitive(id.clone()),
                    inputs: values,
                    reads,
                    write: None,
                    outputs: vec![Output::Computed(ty)],
                    safety,
                    span,
                };
                self.add(spec)
            }
            PrimitiveId::ElementWrite { arity } => {
                // Reached from assignments (the normal path) or a checked
                // expression position.
                let (base, view, _storage) =
                    self.ensure_storage_operand(values[0], &operands[0], span)?;
                values[0] = base;
                if values.len() <= 1 + arity {
                    return Err("element write operands do not match its arity".into());
                }
                let indices = values[1..1 + arity].to_vec();
                self.element_write(base, view, &indices, values[values.len() - 1], span)?;
                Ok(None)
            }
            PrimitiveId::CopyInto => {
                let (base, view, _storage) =
                    self.ensure_storage_operand(values[0], &operands[0], span)?;
                self.copy_into(base, view, values[1], span)?;
                Ok(None)
            }
            PrimitiveId::Extent { .. } | PrimitiveId::ValidExtent { .. } => {
                let reads = self.builder.reads_of(&values)?;
                let spec = PrimitiveSpec {
                    op: PrimitiveOp::Primitive(id.clone()),
                    inputs: values,
                    reads,
                    write: None,
                    outputs: vec![Output::Computed(ty)],
                    safety: Vec::new(),
                    span,
                };
                self.add(spec)
            }
            PrimitiveId::Atomic { op, arity } => {
                let (base, view, storage) =
                    self.ensure_storage_operand(values[0], &operands[0], span)?;
                values[0] = base;
                let source = self.builder.view(view);
                let indices: Vec<GraphValueId> = values[1..1 + arity].to_vec();
                let mut safety = Vec::new();
                for (axis, index) in indices.iter().enumerate() {
                    safety.push(SafetyObligation::IndexInBounds {
                        index: *index,
                        extent: source.shape.axes[axis].clone(),
                    });
                }
                let coverage = self.write_coverage(&source.shape.axes, &indices);
                let dtype = source
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
                self.add(spec)?;
                Ok(None)
            }
            PrimitiveId::Reduce {
                op,
                axis,
                unordered,
            } => {
                // A computed operand is reduced directly; no storage is
                // invented for it.
                let operand_elem = match &operands[0].ty {
                    ValueType::Tensor(s) => s.elem.clone(),
                    ValueType::Scalar(_)
                    | ValueType::Index { .. }
                    | ValueType::Range { .. }
                    | ValueType::Tuple(_)
                    | ValueType::CapabilityValue(_)
                    | ValueType::Void => return Err("reduce needs a tensor operand".into()),
                };
                let input = match self.convert_elem(&operand_elem)? {
                    Elem::Dtype(d) => d,
                    Elem::Repr(_) | Elem::Param(_) => {
                        return Err("reduce needs a dense element".into())
                    }
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
                    ty,
                    span,
                )?;
                Ok(Some(value))
            }
        }
    }
}

/// Project a representation's packing axis through an exact slice: pointing
/// a preceding axis shifts it; retaining the packed axis records its new
/// output position; pointing the packed axis removes the axis descriptor
/// while `Elem::Repr` and the multi-plane backing view are preserved so a
/// later packed element read can decode through the slice transform.
fn slice_packed_axis(source: &TensorType, axes: &[SliceAxis]) -> Option<usize> {
    let packed = source.packed_axis?;
    let mut output_axis = 0usize;
    for (source_axis, selection) in axes.iter().enumerate() {
        if source_axis == packed {
            return match selection {
                SliceAxis::Point(_) => None,
                SliceAxis::Full | SliceAxis::Range { .. } => Some(output_axis),
            };
        }
        if !matches!(selection, SliceAxis::Point(_)) {
            output_axis += 1;
        }
    }
    None
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
            if self.write_coverage_axis(&axes[axis], *index) {
                covered[axis] = true;
            }
        }
        Coverage { axes: covered }
    }

    /// One axis is covered by an index when the index is an enclosing loop's
    /// binder, the write is unconditional within that loop, and the loop's
    /// range covers the axis. Zero-sized axes are trivially covered.
    fn write_coverage_axis(&self, extent: &ExtentExpr, index: GraphValueId) -> bool {
        if *extent == ExtentExpr::Static(0) {
            return true;
        }
        // The index addresses the loop binder either by its original value
        // or through the local's current value (a nested region rebound it
        // to a fresh parameter). A local currently holding the binder value
        // is the binder's value, whatever id carries it.
        self.loops.iter().rev().any(|info| {
            let addresses_binder = info.binder == index
                || self
                    .locals
                    .get(info.binder_local)
                    .is_some_and(|current| *current == Some(index));
            addresses_binder
                && self.if_depth == info.if_depth
                && info.start == ExtentExpr::Static(0)
                && info.end == *extent
        })
    }

    /// Structural coverage of writing through one view: identity/reshape/
    /// transpose views are whole-storage writes; slice views cover the axes
    /// their static bounds provably span.
    fn transform_coverage(&self, view: LogicalViewId) -> Result<Coverage, String> {
        let logical = self.builder.view(view);
        let storage_axes = &self.builder.storage(self.builder.storage_base(view)?).shape.axes;
        let rank = storage_axes.len();
        match &logical.transform {
            ViewTransform::Identity
            | ViewTransform::Reshape { .. }
            | ViewTransform::Transpose { .. } => Ok(Coverage::full(rank)),
            ViewTransform::Slice { axes } => {
                let mut covered = Vec::with_capacity(rank);
                let mut axis_cursor = 0usize;
                for slot in axes {
                    let extent = &storage_axes[axis_cursor];
                    let covers = match slot {
                        SliceAxis::Full => true,
                        SliceAxis::Point(index) => {
                            // A point axis is covered when the index is an
                            // enclosing loop's binder whose range covers the
                            // axis (the same proof `write_coverage` applies
                            // to element writes; the checker's disjoint-write
                            // proof covers the loop's independence).
                            self.write_coverage_axis(extent, *index)
                        }
                        SliceAxis::Range { start, end } => {
                            let start_ok = start
                                .map(|v| self.constants.get(&v) == Some(&0))
                                .unwrap_or(true);
                            let end_ok = end
                                .map(|v| match extent {
                                    ExtentExpr::Static(n) => {
                                        self.constants.get(&v) == Some(&(*n as i64))
                                    }
                                    ExtentExpr::Sym(_) | ExtentExpr::Runtime(_) => false,
                                })
                                .unwrap_or(true);
                            start_ok && end_ok
                        }
                    };
                    covered.push(covers);
                    axis_cursor += 1;
                }
                while covered.len() < rank {
                    covered.push(true);
                }
                Ok(Coverage { axes: covered })
            }
        }
    }

    /// Range-safety obligations of writing through a slice view.
    fn transform_safety(&self, view: LogicalViewId) -> Result<Vec<SafetyObligation>, String> {
        let logical = self.builder.view(view);
        let storage = self.builder.storage(self.builder.storage_base(view)?);
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
                Ok(safety)
            }
            ViewTransform::Identity
            | ViewTransform::Reshape { .. }
            | ViewTransform::Transpose { .. } => Ok(Vec::new()),
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
        base_value: GraphValueId,
        view: LogicalViewId,
        indices: &[GraphValueId],
        value: GraphValueId,
        span: Span,
    ) -> Result<(), String> {
        let base = self.builder.view(view);
        let storage = self.builder.storage_base(view)?;
        if indices.len() > base.shape.axes.len() {
            return Err(format!(
                "element write indexes {} axes of a rank-{} destination",
                indices.len(),
                base.shape.axes.len()
            ));
        }
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
        self.add(spec)?;
        Ok(())
    }

    fn copy_into(
        &mut self,
        dst_value: GraphValueId,
        view: LogicalViewId,
        src_value: GraphValueId,
        span: Span,
    ) -> Result<(), String> {
        let storage = self.builder.storage_base(view)?;
        let coverage = self.transform_coverage(view)?;
        let safety = self.transform_safety(view)?;
        let reads = self.builder.reads_of(&[src_value])?;
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
        self.add(spec)?;
        Ok(())
    }

    fn element_read(
        &mut self,
        base_value: GraphValueId,
        view: LogicalViewId,
        indices: &[GraphValueId],
        span: Span,
    ) -> Result<GraphValueId, String> {
        let base = self.builder.view(view);
        let storage = self.builder.storage_base(view)?;
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
            outputs: vec![Output::Computed(ValueType::Scalar(dtype))],
            safety,
            span,
        };
        Ok(self.add(spec)?.expect("an element read produces a scalar"))
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
            .ok_or("an assignment value is void")?;
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
                let current = self.binding(*root)?;
                let current_ty = self.builder.value_type(current)?;
                let value_ty = self.builder.value_type(value)?;
                match self.builder.value(current)?.tensor_source() {
                    Some(TensorSource::View(view)) if current_ty == value_ty => {
                        // Whole-tensor assignment to a view-backed local
                        // writes through the view (a compound op computes
                        // elementwise first). A view of a computed value is
                        // a snapshot: the write names the local's own state,
                        // so the local realizes its storage first and
                        // rebinds to it.
                        let (current, view) = match self.builder.view(view).base {
                            ViewBase::Storage(_) => (current, view),
                            ViewBase::Value(_) => {
                                let realized = self.materialize_value(current, span)?;
                                self.locals[*root] = Some(realized.0);
                                realized
                            }
                        };
                        if op == AssignOp::Assign {
                            self.copy_into(current, view, value, span)?;
                        } else {
                            let computed =
                                self.binary_assign_op(op, current, value, &current_ty, span)?;
                            self.copy_into(current, view, computed, span)?;
                        }
                    }
                    Some(TensorSource::View(_)) => {
                        if op != AssignOp::Assign {
                            return Err(
                                "a compound assignment cannot change the target's type".into(),
                            );
                        }
                        self.locals[*root] = Some(value);
                    }
                    Some(TensorSource::Computed) | None => {
                        // A computed tensor or a plain value rebinds (SSA).
                        if op == AssignOp::Assign {
                            self.locals[*root] = Some(value);
                        } else {
                            if current_ty != value_ty {
                                return Err(
                                    "a compound assignment cannot change the target's type".into(),
                                );
                            }
                            let computed =
                                self.binary_assign_op(op, current, value, &current_ty, span)?;
                            self.locals[*root] = Some(computed);
                        }
                    }
                }
                Ok(())
            }
            CheckedPlace::Element { root, indices } => {
                // An in-place write names the local's mutable state: a value
                // without storage behind it is realized into the local's own
                // storage first, and the local rebinds to it so later reads
                // see these writes.
                let (current, view, _storage) = self.ensure_view_local(*root, span)?;
                // Convert indices: values for points, optional bounds for
                // ranges.
                let mut point_indices = Vec::new();
                let mut slots = Vec::new();
                let mut slot_args = Vec::new();
                for index in indices {
                    match index {
                        CheckedIndex::Point(p) => {
                            let value = self.expr(f, p)?.ok_or("an index is void")?;
                            point_indices.push(value);
                            slots.push(IndexSlot::Point);
                            slot_args.push(SlotArg::Point(value));
                        }
                        CheckedIndex::Range { start, end } => {
                            let start_value = match start {
                                Some(e) => Some(self.expr(f, e)?.ok_or("a bound is void")?),
                                None => None,
                            };
                            let end_value = match end {
                                Some(e) => Some(self.expr(f, e)?.ok_or("a bound is void")?),
                                None => None,
                            };
                            slots.push(IndexSlot::Range {
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
                        self.element_write(current, view, &point_indices, value, span)?;
                    } else {
                        let read = self.element_read(current, view, &point_indices, span)?;
                        let read_ty = self.builder.value_type(read)?;
                        let computed = self.binary_assign_op(op, read, value, &read_ty, span)?;
                        self.element_write(current, view, &point_indices, computed, span)?;
                    }
                } else {
                    if op != AssignOp::Assign {
                        return Err("a compound assignment cannot target a slice".into());
                    }
                    // Build the destination view, then copy the source into it.
                    let (sliced, sliced_view) =
                        self.make_slice_view(f, current, view, slots, slot_args, span)?;
                    self.copy_into(sliced, sliced_view, value, span)?;
                }
                Ok(())
            }
            CheckedPlace::Tuple(places) => {
                for (index, place) in places.iter().enumerate() {
                    let component = self.tuple_get(f, value, index, span)?;
                    self.assign_place(f, place, op, component, span)?;
                }
                Ok(())
            }
        }
    }

    fn binary_assign_op(
        &mut self,
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
        let spec = PrimitiveSpec {
            op: PrimitiveOp::Primitive(PrimitiveId::Binary(binary)),
            inputs: vec![lhs, rhs],
            reads: self.builder.reads_of(&[lhs, rhs])?,
            write: None,
            outputs: vec![Output::Computed(ty.clone())],
            safety: Vec::new(),
            span,
        };
        Ok(self
            .add(spec)?
            .expect("a compound assignment computes a value"))
    }

    /// Construct the destination view of a slice write.
    fn make_slice_view(
        &mut self,
        f: &mut Factory,
        base_value: GraphValueId,
        base_view: LogicalViewId,
        slots: Vec<IndexSlot>,
        args: Vec<SlotArg>,
        span: Span,
    ) -> Result<(GraphValueId, LogicalViewId), String> {
        let base = self.builder.view(base_view).clone();
        let access = base.access;
        let base_axes = base.shape.axes.clone();
        if slots.len() != args.len() {
            return Err("a slice slot list and its arguments disagree".into());
        }
        if slots.len() > base_axes.len() {
            return Err(format!(
                "slice indexes {} axes of a rank-{} destination",
                slots.len(),
                base_axes.len()
            ));
        }
        let mut inputs = vec![base_value];
        let mut axes = Vec::new();
        let mut safety = Vec::new();
        for (axis, (slot, arg)) in slots.iter().zip(&args).enumerate() {
            match (slot, arg) {
                (IndexSlot::Point, SlotArg::Point(value)) => {
                    inputs.push(*value);
                    axes.push(SliceAxis::Point(*value));
                    safety.push(SafetyObligation::IndexInBounds {
                        index: *value,
                        extent: base_axes[axis].clone(),
                    });
                }
                (IndexSlot::Range { start, end }, SlotArg::Range { start: s, end: e }) => {
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
                (IndexSlot::Point, SlotArg::Range { .. })
                | (IndexSlot::Range { .. }, SlotArg::Point(_)) => {
                    return Err("a slice slot's arguments do not match its shape".into())
                }
            }
        }
        // The destination keeps the base element; its axes shrink
        // structurally: points drop the axis, ranges carry the sliced
        // extent (end - start, bounded by the base axis), trailing axes
        // are kept.
        let mut shape_axes = Vec::new();
        let mut axis_cursor = 0;
        let mut range_values = axes.iter().filter_map(|axis| match axis {
            SliceAxis::Range { start, end } => Some((*start, *end)),
            SliceAxis::Point(_) | SliceAxis::Full => None,
        });
        for slot in &slots {
            match slot {
                IndexSlot::Point => {
                    axis_cursor += 1;
                }
                IndexSlot::Range { .. } => {
                    let (start, end) = range_values
                        .next()
                        .expect("every range slot has a range axis");
                    let base_axis = base_axes[axis_cursor].clone();
                    let extent = match (start, end) {
                        (None, None) => base_axis,
                        (Some(start), Some(end)) => {
                            let expr = RuntimeScalarExpr::Sub(
                                Box::new(RuntimeScalarExpr::Value(end)),
                                Box::new(RuntimeScalarExpr::Value(start)),
                            );
                            // A sliced range is bounded by the axis it
                            // selects from.
                            let capacity = match &base_axis {
                                ExtentExpr::Static(value) => *value,
                                ExtentExpr::Runtime(id) => {
                                    f.runtime_extents[id.index()].capacity
                                }
                                ExtentExpr::Sym(sym) => {
                                    return Err(format!(
                                        "slice destination axis `{sym}` is still symbolic at the logical level"
                                    ))
                                }
                            };
                            ExtentExpr::Runtime(f.runtime_extent(expr, capacity, None))
                        }
                        (Some(_), None) | (None, Some(_)) => {
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
        while axes.len() < base_axes.len() {
            axes.push(SliceAxis::Full);
        }
        let packed_axis = slice_packed_axis(&base.shape, &axes);
        let shape = TensorType {
            axes: shape_axes,
            elem: base.shape.elem.clone(),
            packed_axis,
        };
        let view = self
            .builder
            .declare_view(base.base, shape, access, ViewTransform::Slice { axes })?;
        let spec = PrimitiveSpec {
            op: PrimitiveOp::Primitive(PrimitiveId::SliceView { indices: slots }),
            inputs,
            reads: Vec::new(),
            write: None,
            outputs: vec![Output::View(view)],
            safety,
            span,
        };
        let value = self.add(spec)?.expect("a slice view produces a value");
        Ok((value, view))
    }
}

enum SlotArg {
    Point(GraphValueId),
    Range {
        start: Option<GraphValueId>,
        end: Option<GraphValueId>,
    },
}

/// One state parameter of a loop body: a carried tensor local of an ordered
/// loop, or a joined storage of an independent loop.
enum StateCarry {
    Local(LocalId),
    Joined,
}

impl<'w> Work<'w> {
    // -- loops --------------------------------------------------------------

    fn loop_stmt(
        &mut self,
        f: &mut Factory,
        kind: LoopKind,
        binder: LocalId,
        range: &CheckedRange,
        body: &CheckedBlock,
        mutation: &LoopMutationSummary,
    ) -> Result<(), String> {
        let span = range.start.span;
        let start = self.expr(f, &range.start)?.ok_or("a range bound is void")?;
        let end = self.expr(f, &range.end)?.ok_or("a range bound is void")?;
        let binder_ty = self.convert_type(f, &self.local_tys[binder].ty.clone())?;
        let ValueType::Index { bound } = &binder_ty else {
            return Err("a loop binder is an index".into());
        };
        let range_logical = LogicalRange {
            start,
            end,
            bound: bound.clone(),
        };
        let binder_id = self.builder.fresh_value(GraphValueKind::Index {
            bound: bound.clone(),
        })?;

        // Symbolic range endpoints, for coverage proofs.
        let ctx_start = match &range.start.sym {
            Some(sym) => self.resolve_sym(f, sym)?.extent(),
            None => ExtentExpr::Static(0),
        };
        let ctx_end = match &range.end.sym {
            Some(sym) => self.resolve_sym(f, sym)?.extent(),
            None => bound.clone(),
        };

        let mut free = free_locals(body);
        free.remove(&binder);
        let storage_written = storage_written_roots(body);
        let mutated: Vec<LocalId> = mutation
            .carried
            .iter()
            .chain(&mutation.atomics)
            .copied()
            .filter(|local| free.contains(local))
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();

        // A local the body writes in place whose value has no storage behind
        // it (a computed tensor or a view of one) needs its storage outside
        // the loop: realize it (and rebind) before the body.
        self.materialize_written_locals(
            mutated
                .iter()
                .copied()
                .filter(|local| storage_written.contains(local)),
            span,
        )?;

        // Storages threaded through the body region: ordered loops carry the
        // view-backed captured locals the checker marked; independent loops
        // thread every captured storage the body writes (their visits join).
        let mut captured_storages = BTreeSet::new();
        for local in &mutated {
            if let Some(storage) = self.storage_of_local(*local)? {
                captured_storages.insert(storage);
            }
        }

        // Carried slots (ordered loops only): every changed captured local,
        // as a state (storage-backed) or a value (a tensor without storage
        // behind it, or a non-tensor).
        let mut carried_value_locals = Vec::new();
        let mut carried_state_locals = Vec::new();
        match kind {
            LoopKind::Ordered => {
                for local in &mutated {
                    if self.storage_of_local(*local)?.is_some() {
                        carried_state_locals.push(*local);
                    } else {
                        carried_value_locals.push(*local);
                    }
                }
            }
            LoopKind::Independent => {
                // Independent loops admit no value carry: every written
                // capture is storage-backed after materialization.
                for local in &mutated {
                    if self.storage_of_local(*local)?.is_none() {
                        return Err(format!(
                            "independent loop rebinds captured local `{}`; visits may only write storage",
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
            let binding = self.binding(*local)?;
            let kind = self.builder.value(binding)?.kind().clone();
            let param = self.builder.fresh_value(kind)?;
            param_ids.insert(*local, param);
            params.push(RegionParameter::Value {
                id: param,
                ty: self.builder.value_type(binding)?,
            });
            if !carried_value_locals.contains(local) {
                invariant_values.push(binding);
            }
        }
        let mut state_params: Vec<(StateCarry, LogicalStorageId)> = Vec::new();
        for local in carried_state_locals.iter().copied() {
            let storage = self
                .storage_of_local(local)?
                .ok_or("a carried tensor local has no storage")?;
            let token = self.builder.fresh_state(storage)?;
            params.push(RegionParameter::State { id: token, storage });
            state_params.push((StateCarry::Local(local), storage));
        }
        if kind == LoopKind::Independent {
            for storage in &captured_storages {
                let token = self.builder.fresh_state(*storage)?;
                params.push(RegionParameter::State {
                    id: token,
                    storage: *storage,
                });
                state_params.push((StateCarry::Joined, *storage));
            }
        }

        self.loops.push(LoopInfo {
            binder: binder_id,
            binder_local: binder,
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
        let mut body_exit: BTreeMap<LocalId, GraphValueId> = BTreeMap::new();
        for local in &free {
            body_exit.insert(*local, self.binding(*local)?);
        }
        // A state-carried local must still name its storage at body exit.
        for local in &carried_state_locals {
            let storage = self.storage_of_local(*local)?;
            let expected = state_params
                .iter()
                .find_map(|(carry, storage)| match carry {
                    StateCarry::Local(candidate) if candidate == local => Some(*storage),
                    StateCarry::Local(_) | StateCarry::Joined => None,
                })
                .expect("every state-carried local has a state parameter");
            if storage != Some(expected) {
                return Err(format!(
                    "captured tensor local `{}` is rebound to a different value inside an ordered loop; this is not representable",
                    self.local_tys[*local].name
                ));
            }
        }
        self.locals = saved_locals;
        if !matches!(flow, Flow::Next) {
            return Err("`return` inside a loop is rejected while checking".into());
        }
        // Restore the pre-loop current states: the loop node consumes those
        // tokens and installs its own exit tokens.
        for (storage, token) in &pre_states {
            self.builder.set_current_state(*storage, *token)?;
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
                initial: RegionInput::Value(self.binding(*local)?),
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
        for (ordinal, (carry, storage)) in state_params.iter().enumerate() {
            match carry {
                StateCarry::Joined => {
                    // Independent-loop state parameter: its result is
                    // assembled with the joins below.
                }
                StateCarry::Local(_) => {
                    let token = self.builder.current_state(*storage)?;
                    body_results.push(RegionResult::State {
                        id: token,
                        storage: *storage,
                        join: None,
                    });
                    carried.push(CarriedSlot {
                        initial: RegionInput::State(pre_states[storage]),
                        body_parameter: RegionParameterId((1 + free.len() + ordinal) as u32),
                        body_result: RegionResultId(body_results.len() as u32 - 1),
                        loop_result: RegionResultId(value_ordinal),
                    });
                    value_ordinal += 1;
                }
            }
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

struct ArmBuilt {
    region: GraphRegion,
    bindings: BTreeMap<LocalId, GraphValueId>,
    /// The arm's returned components when the construct returns.
    returned: Option<Vec<GraphValueId>>,
}

impl<'w> Work<'w> {
    // -- conditionals -------------------------------------------------------

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
        self.if_core(f, condition, then_body, else_body, Some((rest, terminator)))?;
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
        let cond = self.expr(f, condition)?.ok_or("a condition is void")?;
        let then_free = free_locals(then_body);
        let else_free = free_locals(else_body);
        let union_free: BTreeSet<LocalId> = then_free.union(&else_free).copied().collect();
        // A local an arm writes in place whose value has no storage behind
        // it needs its storage outside the `if` so both arms join its state.
        let storage_written: BTreeSet<LocalId> = storage_written_roots(then_body)
            .union(&storage_written_roots(else_body))
            .copied()
            .filter(|local| union_free.contains(local))
            .collect();
        self.materialize_written_locals(storage_written.iter().copied(), span)?;
        let mut union_writes: BTreeSet<LocalId> = BTreeSet::new();
        for local in written_roots(then_body).union(&written_roots(else_body)) {
            // Only storage-backed locals join as states; a written value
            // local rebinds and joins as a value.
            if union_free.contains(local) && self.storage_of_local(*local)?.is_some() {
                union_writes.insert(*local);
            }
        }
        let union_rebinds: BTreeSet<LocalId> = rebound_roots(then_body)
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
            let binding = self.binding(*local)?;
            let kind = self.builder.value(binding)?.kind().clone();
            let param = self.builder.fresh_value(kind)?;
            param_ids.insert(*local, param);
            captured.push(IfCapture {
                parameter: param,
                outer: binding,
            });
            params.push(RegionParameter::Value {
                id: param,
                ty: self.builder.value_type(binding)?,
            });
        }
        let mut state_params = BTreeMap::new();
        for local in &union_writes {
            let storage = self
                .storage_of_local(*local)?
                .ok_or("a written local has no storage")?;
            let token = self.builder.fresh_state(storage)?;
            params.push(RegionParameter::State { id: token, storage });
            state_params.insert(*local, storage);
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
            &union_rebinds,
            &union_writes,
            returning,
            span,
        )?;

        // Explicit joins. Ordinals follow the arm result layout: rebound
        // values, written states, then (when returning) result values.
        let mut joins = Vec::new();
        let mut joined_bindings = BTreeMap::new();
        let mut ordinal = 0u32;
        for local in &union_rebinds {
            let joined = self.join_value(then_arm.bindings[local], else_arm.bindings[local])?;
            joins.push(JoinSlot::Value {
                then_result: RegionResultId(ordinal),
                else_result: RegionResultId(ordinal),
                joined,
                ty: self.builder.value_type(joined)?,
            });
            joined_bindings.insert(*local, joined);
            ordinal += 1;
        }
        for local in &union_writes {
            let storage = state_params[local];
            let joined = self.builder.fresh_state(storage)?;
            joins.push(JoinSlot::State {
                then_result: RegionResultId(ordinal),
                else_result: RegionResultId(ordinal),
                joined,
                storage,
            });
            ordinal += 1;
        }
        if returning {
            let then_returned = then_arm
                .returned
                .as_ref()
                .ok_or("a returning arm binds its result values")?;
            let else_returned = else_arm
                .returned
                .as_ref()
                .ok_or("a returning arm binds its result values")?;
            if then_returned.len() != else_returned.len() {
                return Err("the arms of a returning `if` return different arities".into());
            }
            let mut returned = Vec::new();
            for (then_value, else_value) in then_returned.iter().zip(else_returned) {
                let joined = self.join_value(*then_value, *else_value)?;
                joins.push(JoinSlot::Value {
                    then_result: RegionResultId(ordinal),
                    else_result: RegionResultId(ordinal),
                    joined,
                    ty: self.builder.value_type(joined)?,
                });
                returned.push(joined);
                ordinal += 1;
            }
            self.returned = Some(returned);
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

    /// The joined value of two arm values: the same view when both arms
    /// leave one view value, otherwise a produced value of their type.
    fn join_value(
        &mut self,
        then_value: GraphValueId,
        else_value: GraphValueId,
    ) -> Result<GraphValueId, String> {
        let then_kind = self.builder.value(then_value)?.kind().clone();
        let else_kind = self.builder.value(else_value)?.kind().clone();
        let ty = self.builder.value_type(then_value)?;
        if ty != self.builder.value_type(else_value)? {
            return Err(format!(
                "the arms of an `if` join values of different types: {ty} and {}",
                self.builder.value_type(else_value)?
            ));
        }
        let kind = if then_kind == else_kind {
            then_kind
        } else {
            computed_kind(ty)?
        };
        self.builder.fresh_value(kind)
    }

    #[allow(clippy::too_many_arguments)]
    fn build_arm(
        &mut self,
        f: &mut Factory,
        params: &[RegionParameter],
        param_ids: &BTreeMap<LocalId, GraphValueId>,
        body: &CheckedBlock,
        continuation: Option<(&[CheckedStmt], &BlockTerminator)>,
        union_rebinds: &BTreeSet<LocalId>,
        union_writes: &BTreeSet<LocalId>,
        returning: bool,
        span: Span,
    ) -> Result<ArmBuilt, String> {
        self.builder.begin_region(params.to_vec())?;
        let saved_locals = self.locals.clone();
        let saved_returned = self.returned.clone();
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
            let value = self.binding(*local)?;
            let ty = self.builder.value_type(value)?;
            arm_results.push(RegionResult::Value { id: value, ty });
            bindings.insert(*local, value);
        }
        for local in union_writes {
            let storage = self
                .storage_of_local(*local)?
                .ok_or("a written local has no storage")?;
            let token = self.builder.current_state(storage)?;
            arm_results.push(RegionResult::State {
                id: token,
                storage,
                join: None,
            });
        }
        let arm_returned = if returning {
            let values = self
                .returned
                .clone()
                .ok_or("a returning arm binds its result values")?;
            for value in &values {
                let ty = self.builder.value_type(*value)?;
                arm_results.push(RegionResult::Value { id: *value, ty });
            }
            Some(values)
        } else {
            None
        };
        let region = self.builder.end_region(arm_results)?;
        self.locals = saved_locals;
        self.returned = saved_returned;
        Ok(ArmBuilt {
            region,
            bindings,
            returned: arm_returned,
        })
    }
}

impl<'w> Work<'w> {
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
            arg_values.push(self.expr(f, arg)?.ok_or("a call argument is void")?);
        }

        let family_name = f.program.families[call.family].name.clone();
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
        let elems = self.elems.clone();
        let caller_elem = |name: &str| elems.get(name).cloned();
        let sym_env = self.sym_env.clone();
        let domain = f.domain;
        let judge = |predicate: &Predicate| {
            judge_predicate(predicate, &|name| sym_env.get(name).cloned(), domain)
        };
        let app = family::applicable(
            f.program,
            &f.target.backend,
            f.supports,
            &candidates,
            &judge,
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
            .definition(f.program.families[call.family].contract);
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
            let mut child_sym_env = SymEnv::new();
            for (param, sym) in &candidate.shape_args {
                child_shapes.insert(param.clone(), self.resolve_sym(f, sym)?);
                child_sym_env.insert(
                    param.clone(),
                    family::substitute(sym, &|name| self.sym_env.get(name).cloned()),
                );
            }
            let mut child_elems = ElemEnv::new();
            for (param, elem) in &candidate.elem_args {
                child_elems.insert(param.clone(), self.convert_elem(elem)?);
            }
            f.ids = self.builder.take_ids();
            let built = f.build_graph(
                definition,
                child_shapes,
                child_sym_env,
                child_elems,
                choice,
                ordinal as u32,
            );
            self.builder.restore_ids(std::mem::take(&mut f.ids));
            let graph = built.map_err(|error| match error {
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
                interface: interface.clone(),
                alternatives: NonEmpty::new(alternatives)
                    .ok_or("the occurrence has alternatives")?,
            },
        );

        // The callee contract instantiated with this caller's identities:
        // one input per interface leaf at its canonical path. A tensor leaf
        // backed by caller storage consumes that storage's current state (an
        // exclusive borrow gets its final state); a computed tensor leaf —
        // the value itself, directly or through a view of one — is shared or
        // moved as a value, with no caller storage fabricated for it; one
        // produced value per result leaf.
        let mut inputs = BTreeMap::new();
        let mut final_states = BTreeMap::new();
        let mut exclusively_borrowed = Vec::new();
        for (ordinal, param) in interface.params.iter().enumerate() {
            let arg = arg_values[ordinal];
            for (path, leaf) in boundary_leaves(&param.ty) {
                let key = BoundaryLeaf::Input {
                    param: ordinal as u32,
                    leaf: path.clone(),
                };
                let leaf_value = self.decompose_value(arg, &path.0, span)?;
                match leaf {
                    LeafKind::Tensor(_) => {
                        if param.ownership == ParamOwnership::Value {
                            return Err(format!(
                                "tensor leaf {path} of parameter `{}` is passed by value; tensor parameters are owned or borrowed",
                                param.name
                            ));
                        }
                        let storage = match self.builder.tensor_source(leaf_value)? {
                            TensorSource::View(view) => match self.builder.view(view).base {
                                ViewBase::Storage(storage) => Some(storage),
                                ViewBase::Value(_) => None,
                            },
                            TensorSource::Computed => None,
                        };
                        match (param.ownership, storage) {
                            (ownership, Some(storage)) => {
                                let state = self.builder.current_state(storage)?;
                                if ownership == ParamOwnership::Exclusive {
                                    exclusively_borrowed.push(storage);
                                    final_states
                                        .insert(key.clone(), self.builder.fresh_state(storage)?);
                                }
                                inputs.insert(
                                    key,
                                    CallInput::Tensor {
                                        value: leaf_value,
                                        state,
                                        ownership,
                                    },
                                );
                            }
                            (ParamOwnership::Shared | ParamOwnership::Owned, None) => {
                                inputs.insert(
                                    key,
                                    CallInput::Computed {
                                        value: leaf_value,
                                        ownership: param.ownership,
                                    },
                                );
                            }
                            (ParamOwnership::Exclusive, None) => {
                                return Err(format!(
                                    "exclusively borrowed tensor leaf {path} of parameter `{}` is a computed value; an exclusive borrow requires a tensor place with storage",
                                    param.name
                                ));
                            }
                            (ParamOwnership::Value, None) => {
                                return Err(format!(
                                    "tensor leaf {path} of parameter `{}` is passed by value; tensor parameters are owned or borrowed",
                                    param.name
                                ));
                            }
                        }
                    }
                    LeafKind::Scalar(_)
                    | LeafKind::Index(_)
                    | LeafKind::Range(_)
                    | LeafKind::Capability(_) => {
                        inputs.insert(key, CallInput::Value(leaf_value));
                    }
                }
            }
        }
        let mut results = BTreeMap::new();
        for (path, leaf) in boundary_leaves(&interface.result) {
            let value = self.builder.fresh_value(leaf.produced_kind())?;
            results.insert(BoundaryLeaf::Result { leaf: path }, value);
        }
        self.builder.add_call(
            choice,
            CallBoundary {
                inputs,
                results: results.clone(),
                final_states,
            },
            span,
        )?;
        // The callee returns every exclusively borrowed storage fully
        // initialized: the checker admits an unassigned argument only on
        // that condition, and an assigned one stays initialized.
        for storage in exclusively_borrowed {
            self.builder.mark_fully_initialized(storage);
        }

        // The call's value: the result leaves repacked along the result
        // type; void calls retain completion only.
        if interface.result.is_void() {
            return Ok(None);
        }
        let value = self.repack(
            &interface.result,
            &ValuePath::default(),
            &|path| results.get(&BoundaryLeaf::Result { leaf: path.clone() }).copied(),
            span,
        )?;
        Ok(Some(value))
    }
}

// ---------------------------------------------------------------------------
// Entry construction
// ---------------------------------------------------------------------------

pub(super) fn construct(
    program: &Program,
    target: &EffectiveTargetIdentity,
    supports: &dyn Fn(&IntrinsicUse) -> Result<(), String>,
    domain: &SpecializationDomain,
) -> Result<LogicalProgram, BuildError> {
    let entry = domain.entry();
    let family_index = program.family_index(entry).map_err(BuildError::Invalid)?;
    let contract = program.definition(program.families[family_index].contract);
    let mut factory = Factory::new(program, target, supports, domain);

    // Entry shape parameters: exact bindings are static extents; bounded
    // bindings are the domain's retained invocation shape fields (one per
    // bounded parameter, in declared order), each carried by one runtime
    // extent with the field's capacity and expected value.
    let fields = domain.shape_fields();
    let mut field_extents: BTreeMap<String, RuntimeExtentId> = BTreeMap::new();
    for field in &fields {
        let extent = factory.runtime_extent(
            RuntimeScalarExpr::ShapeField(field.id),
            field.domain.max(),
            Some(field.expected),
        );
        field_extents.insert(field.name.clone(), extent);
    }
    factory.shape_fields = fields;
    for name in &contract.shape_params {
        let extent = match domain.binding(name) {
            ShapeBinding::Exact(value) => ShapeExtent::Static(value),
            ShapeBinding::Bounded { .. } => ShapeExtent::Runtime(field_extents[name]),
        };
        factory.entry_shapes.insert(name.clone(), extent);
    }
    let elems: ElemEnv = domain.elems().clone();

    let (candidates, unreachable) = family::entry_candidates(program, family_index, &target.backend);
    let caller_elem = |name: &str| elems.get(name).cloned();
    // Entry candidates bind their shape parameters to the entry's own
    // symbols; predicates are decided directly over the domain.
    let judge = |predicate: &Predicate| judge_predicate(predicate, &|_| None, domain);
    let app = family::applicable(
        program,
        &target.backend,
        supports,
        &candidates,
        &judge,
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
    let context = format!("the contract of `{}`", contract.name);
    let mut interface_params = Vec::new();
    for param in &contract.params {
        interface_params.push(InterfaceParam {
            name: param.name.clone(),
            mode: param.mode,
            ownership: param.ownership,
            ty: factory
                .convert_interface_type(&param.ty, &elems, &context)
                .map_err(BuildError::Invalid)?,
        });
    }
    let interface = FunctionInterface {
        name: contract.name.clone(),
        params: interface_params,
        result: factory
            .convert_interface_type(&contract.result, &elems, &context)
            .map_err(BuildError::Invalid)?,
    };

    let mut alternatives = Vec::new();
    for (ordinal, resolved) in app.alternatives.iter().enumerate() {
        let candidate = &resolved.candidate;
        let definition = program.definition(candidate.definition);
        let mut child_shapes = ShapeEnv::new();
        let mut child_sym_env = SymEnv::new();
        for (param, sym) in &candidate.shape_args {
            if !factory.is_entry_shape_sym(sym) {
                return Err(BuildError::Invalid(format!(
                    "shape parameter `{param}` of `{}` is not bound by the entry domain",
                    definition.name
                )));
            }
            child_shapes.insert(param.clone(), factory.shape_extent(sym).map_err(BuildError::Invalid)?);
            child_sym_env.insert(param.clone(), sym.clone());
        }
        let mut child_elems = ElemEnv::new();
        for (param, elem) in &candidate.elem_args {
            child_elems.insert(
                param.clone(),
                convert_elem(elem, &elems, &format!("`{}`", definition.name))
                    .map_err(BuildError::Invalid)?,
            );
        }
        let graph = factory.build_graph(
            definition,
            child_shapes,
            child_sym_env,
            child_elems,
            entry_choice,
            ordinal as u32,
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
    let identity = logical_identity(program, target, domain);
    Ok(LogicalProgram {
        identity,
        target: target.clone(),
        domain: domain.clone(),
        shape_fields: IdVec::new(factory.shape_fields),
        entry_choice,
        choices: IdVec::new(choices),
        graphs: IdVec::new(graphs),
        runtime_extents: IdVec::new(factory.runtime_extents),
    })
}

/// The logical identity: source identity, registry revision, target and the
/// specialization domain (entry, every shape binding, every element
/// binding). Changing any of them changes every downstream plan, cache and
/// evidence identity.
fn logical_identity(
    program: &Program,
    target: &EffectiveTargetIdentity,
    domain: &SpecializationDomain,
) -> LogicalIdentity {
    use sha2::{Digest, Sha256};
    let mut hash = Sha256::new();
    hash.update(program.identity());
    hash.update(crate::intrinsics::REGISTRY_REVISION.as_bytes());
    hash.update(target.backend.as_bytes());
    hash.update(target.capability_fingerprint.as_bytes());
    hash.update(domain.identity_bytes());
    LogicalIdentity(hash.finalize().into())
}
