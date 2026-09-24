//! The checker: resolution, contract families and target capabilities, typing
//! of every value kind through the intrinsic registry, ownership/moves,
//! borrow rules, loop carry and disjointness, initialization coverage, and
//! `where` predicates. Emits the checked representation directly.
//! Entry point is [`crate::checked::check_source`].
//!
//! Every symbolic integer is an `IntExpr` in the definition's private arena.
//! Definitions are checked once each, callees before their callers (L20).

mod call;
pub(crate) mod dimensions;
mod elements;
mod entry_build;
mod expr;
mod initialization;
#[cfg(test)]
mod initialization_tests;
#[cfg(test)]
mod stable_identity_tests;
pub(crate) mod ir;
mod ownership;
#[cfg(test)]
mod ownership_tests;
pub(crate) mod prove;
pub(crate) mod resolve;
mod stmt;
pub(crate) mod xfer;

pub(crate) use entry_build::build_entry;

use self::ir::{
    Block as CheckedBlock, Body as CheckedBody, Expr as CheckedExpr, Local as CheckedLocal,
    LocalId, Ownership as ParamOwnership, Placement, Predicate,
};
use crate::checked::{DiagnosticRule, EntryInfo};
use crate::expr::{ExprArena, IntExpr, SymbolId, SymbolSort};
use crate::ids::{CapabilityId, ModuleHash, ModuleId, ProgramId, StableFunctionId};
use crate::span::{Diagnostic, Span};
use crate::syntax::ast;
use crate::types::{DType, Elem, TensorType, ValueType};
use prove::Facts;
use resolve::{BodySig, Declared, Located, Resolved};
use std::collections::{BTreeSet, HashMap, HashSet};

/// What a local binds, for the checker's ownership and storage analysis. Not
/// part of the checked representation: the canonical type records what a value
/// is, this records how its storage is reached.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LocalKind {
    Param(usize),
    /// `let mut` state.
    State,
    /// An immutable `let` binding.
    Value,
    /// A loop binder.
    Binder,
}

/// How the storage of a checked expression is reached.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ValueClass {
    Scalar,
    /// A computed dense value (elementwise results, snapshots, decodes, packed reads).
    Computed,
    /// Owned tensor storage (allocations, materializations, clones, call results).
    Owned,
    /// A borrowed view of storage (tensor/view parameters, selections of storage).
    Borrowed,
}

pub(crate) struct Env<'a> {
    pub resolved: &'a Resolved<'a>,
    /// The text of every source file, for diagnostics that quote source.
    pub texts: &'a [&'a str],
    /// The outcome of every definition checked so far (every callee of the
    /// definition being checked).
    checked: &'a [Option<CheckedOutcome>],
}

pub(crate) struct Checker<'a> {
    pub env: &'a Env<'a>,
    pub def: usize,
    pub sig: BodySig,
    /// The target whose forms this body may name: a lowering target.
    pub target: Option<crate::registry::BackendName>,
    pub requires: Vec<(CapabilityId, Span)>,
    pub used_capabilities: BTreeSet<CapabilityId>,
    pub locals: Vec<CheckedLocal>,
    pub kinds: Vec<LocalKind>,
    pub scopes: Vec<HashMap<String, LocalId>>,
    pub facts: Facts,
    pub symbols: HashMap<LocalId, SymbolId>,
    pub scalar_symbols: HashMap<LocalId, IntExpr>,
    /// Symbols that stand for runtime data values (element reads, words,
    /// runtime slice lengths, binders over data bounds). A bound over one of
    /// them that is not proved is checked at run time instead of rejected.
    pub data_symbols: HashSet<SymbolId>,
    /// Runtime-bounded range views: (start, end, parent extent, realized-length atom).
    pub dyn_views: Vec<(Option<CheckedExpr>, Option<CheckedExpr>, IntExpr, SymbolId)>,
    /// Active independent (`parallel for`) loops: (captured floor, binder).
    pub logical_parallel: Vec<(usize, LocalId)>,
    /// L14: per innermost `parallel for` (the body's own call site at the
    /// bottom), the number of enclosing participant-divergent conditions.
    pub divergence: Vec<usize>,
    /// Binders of ordered loops whose bounds are participant-uniform.
    pub uniform_binders: HashSet<SymbolId>,
    /// Nesting depth of the block being checked; the function body is 0.
    pub block_depth: usize,
    pub diagnostics: Vec<Diagnostic>,
    /// Names whose binding was rejected; uses of them are not reported again.
    pub poisoned: HashSet<String>,
    pub arena: ExprArena,
    pub elements: elements::ElementUseSet,
    pub placement: Placement,
}

impl<'a> Checker<'a> {
    fn new(env: &'a Env<'a>, def: usize) -> Checker<'a> {
        let declared: &'a Declared<'a> = &env.resolved.declared[def];
        let target = declared.kind.target();
        let (sig, arena) = declared.sig.for_body();
        let mut c = Checker {
            env,
            def,
            sig,
            target,
            requires: declared.requires.clone(),
            used_capabilities: BTreeSet::new(),
            locals: Vec::new(),
            kinds: Vec::new(),
            scopes: vec![HashMap::new()],
            facts: Facts::new(),
            symbols: HashMap::new(),
            scalar_symbols: HashMap::new(),
            data_symbols: HashSet::new(),
            dyn_views: Vec::new(),
            logical_parallel: Vec::new(),
            divergence: vec![0],
            uniform_binders: HashSet::new(),
            block_depth: 0,
            diagnostics: Vec::new(),
            poisoned: HashSet::new(),
            arena,
            elements: elements::ElementUseSet::default(),
            placement: Placement::default(),
        };
        // Shape parameters are positive extents unless a `where` admits zero.
        for dimension in c.sig.dimensions.clone() {
            let lower = c.arena.int(if dimension.admits_zero { 0 } else { 1 });
            c.facts.set_range_lower(dimension.symbol, lower);
        }
        for conjunct in c.sig.predicates.clone() {
            match conjunct.predicate {
                Predicate::NonNegative(e) => c.assume_nonneg(e),
                Predicate::Zero(e) => c.assume_zero(e),
                Predicate::NonZero(e) => c.facts.assume_nonzero(&c.arena, e),
            }
        }
        for (i, p) in c.sig.params.clone().into_iter().enumerate() {
            let id = c.declare(
                &p.name,
                p.ty.clone(),
                p.span,
                LocalKind::Param(i),
                p.ownership == ParamOwnership::Exclusive,
            );
            match &p.ty {
                ValueType::Index { .. } => {
                    let symbol = c.arena.proof_variable(SymbolSort::Int);
                    c.facts.assume_type(&mut c.arena, symbol, &p.ty);
                    c.symbols.insert(id, symbol);
                    c.locals[id.index()].symbol = Some(symbol);
                }
                ValueType::Scalar(DType::I32 | DType::U32) => {
                    let symbol = c.fresh_data_symbol();
                    c.facts.assume_type(&mut c.arena, symbol, &p.ty);
                    c.symbols.insert(id, symbol);
                    c.locals[id.index()].symbol = Some(symbol);
                }
                _ => {}
            }
            if p.ownership == ParamOwnership::Exclusive {
                for element in tensor_elements(&p.ty) {
                    c.elements.stored(element);
                }
            }
        }
        for element in tensor_elements(&c.sig.result.clone()) {
            c.elements.stored(element);
        }
        let contract = env.resolved.families[declared.family.index()].contract.index() == def;
        if contract {
            c.signature_totality();
        }
        c
    }

    pub fn error(&mut self, rule: DiagnosticRule, span: Span, message: impl Into<String>) {
        self.diagnostics
            .push(Diagnostic::with_rule(rule, span, message));
    }

    /// The source text of a span of this definition's file.
    pub fn text(&self, span: Span) -> &str {
        let file = self.env.resolved.declared[self.def].file;
        &self.env.texts[file][span.start as usize..span.end as usize]
    }

    /// The name a symbol renders with in diagnostics: a dimension's name, or
    /// the name of the local it stands for.
    pub fn symbol_name(&self, symbol: SymbolId) -> String {
        if let Some(ordinal) = self.sig.dimension_of(symbol) {
            return self.sig.dimensions[ordinal].name.clone();
        }
        self.symbols
            .iter()
            .find(|(_, candidate)| **candidate == symbol)
            .or_else(|| {
                self.locals
                    .iter()
                    .enumerate()
                    .find(|(_, local)| local.symbol == Some(symbol))
                    .map(|(ordinal, local)| (&local.name, ordinal))
                    .and(None)
            })
            .map(|(local, _)| self.locals[local.index()].name.clone())
            .unwrap_or_else(|| "?".to_owned())
    }

    /// An integer expression in source spelling.
    pub fn render(&self, e: IntExpr) -> String {
        prove::display(&self.arena, e, &|symbol| self.symbol_name(symbol))
    }

    /// A type in source spelling, with its shape expressions.
    pub fn shown(&self, ty: &ValueType) -> String {
        ty.with_shapes(&self.arena, &|symbol| self.symbol_name(symbol))
            .to_string()
    }

    pub fn use_capability(&mut self, capability: &CapabilityId, span: Span, use_site: &str) {
        self.used_capabilities.insert(*capability);
        if !self
            .requires
            .iter()
            .any(|(declared, _)| declared == capability)
        {
            self.error(
                DiagnosticRule::Capability,
                span,
                format!(
                    "{use_site} requires capability `{}.{}`; add `requires {}.{}` to this declaration",
                    crate::registry::capability_info(*capability).backend.as_str(),
                    crate::registry::capability_info(*capability).name,
                    crate::registry::capability_info(*capability).backend.as_str(),
                    crate::registry::capability_info(*capability).name
                ),
            );
        }
    }

    pub fn lookup(&self, name: &str) -> Option<LocalId> {
        self.scopes.iter().rev().find_map(|s| s.get(name).copied())
    }

    /// Whether a local is visible at this point: declared in a live scope.
    pub fn in_scope(&self, id: LocalId) -> bool {
        self.scopes
            .iter()
            .any(|scope| scope.values().any(|local| *local == id))
    }

    pub fn declare(
        &mut self,
        name: &str,
        ty: ValueType,
        span: Span,
        kind: LocalKind,
        mutable: bool,
    ) -> LocalId {
        let id = LocalId::new(
            u32::try_from(self.locals.len()).expect("definition has more than u32::MAX locals"),
        );
        let ownership = self.default_ownership(id, &ty, kind);
        self.locals.push(CheckedLocal {
            ownership,
            name: name.to_string(),
            ty,
            mutable,
            span,
            symbol: None,
        });
        self.kinds.push(kind);
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name.to_string(), id);
        }
        id
    }

    /// A fresh proof variable.
    pub fn fresh_symbol(&mut self) -> SymbolId {
        self.arena.proof_variable(SymbolSort::Int)
    }

    /// A fresh proof variable standing for a runtime data value.
    pub fn fresh_data_symbol(&mut self) -> SymbolId {
        let symbol = self.fresh_symbol();
        self.data_symbols.insert(symbol);
        symbol
    }

    /// Whether an expression mentions a runtime data value.
    pub fn data_dependent(&self, e: IntExpr) -> bool {
        prove::symbols(&self.arena, e)
            .iter()
            .any(|symbol| self.data_symbols.contains(symbol))
    }

    /// Assume `e >= 0`.
    pub fn assume_nonneg(&mut self, e: IntExpr) {
        self.facts.assume_nonnegative(&mut self.arena, e);
    }

    /// Assume `e == 0`: a zero fact, and `e >= 0` and `-e >= 0`.
    pub fn assume_zero(&mut self, e: IntExpr) {
        self.facts.assume_zero(&self.arena, e);
        self.assume_nonneg(e);
        let zero = self.arena.int(0);
        let neg = self.arena.int_sub(zero, e);
        self.assume_nonneg(neg);
    }

    /// Prove `e >= 0`, or report `what` with the unproved goal.
    pub fn require_nonneg(&mut self, rule: DiagnosticRule, e: IntExpr, span: Span, what: &str) -> bool {
        if prove::nonneg(&self.arena, &self.facts, e) {
            return true;
        }
        let rendered = self.render(e);
        self.error(rule, span, format!("{what}: cannot prove `{rendered} >= 0`"));
        false
    }

    /// L27: every parameter and result type of a family contract is total
    /// under its positivity and `where` facts. Each axis and bound is
    /// nonnegative and each divisor is nonzero.
    fn signature_totality(&mut self) {
        let mut types = self
            .sig
            .params
            .iter()
            .map(|parameter| (parameter.ty.clone(), parameter.span))
            .collect::<Vec<_>>();
        types.push((self.sig.result.clone(), self.env.resolved.declared[self.def].name_span));
        for (ty, span) in types {
            for extent in type_extents(&ty) {
                for divisor in divisors(&self.arena, extent) {
                    if !self.facts.nonzero(&mut self.arena, divisor) {
                        let rendered = self.render(divisor);
                        let shown = self.shown(&ty);
                        self.error(
                            DiagnosticRule::Dimension,
                            span,
                            format!("type `{shown}` requires `{rendered} != 0`; add `where {rendered} >= 1`"),
                        );
                    }
                }
                if !prove::nonneg(&self.arena, &self.facts, extent) {
                    let rendered = self.render(extent);
                    let shown = self.shown(&ty);
                    self.error(
                        DiagnosticRule::Dimension,
                        span,
                        format!("type `{shown}` requires `{rendered} >= 0`; add `where {rendered} >= 0`"),
                    );
                }
            }
        }
    }

    // ---- types ----

    pub fn same_extent(&self, a: IntExpr, b: IntExpr) -> bool {
        prove::same(&self.arena, a, b)
    }

    pub fn same_axes(&self, a: &TensorType, b: &TensorType) -> bool {
        a.rank() == b.rank()
            && a.axes
                .iter()
                .zip(&b.axes)
                .all(|(x, y)| self.same_extent(*x, *y))
    }

    pub fn same_ty(&self, a: &ValueType, b: &ValueType) -> bool {
        match (a, b) {
            (ValueType::Tensor(x), ValueType::Tensor(y)) => {
                x.elem == y.elem && self.same_axes(x, y)
            }
            (ValueType::Index { bound: x }, ValueType::Index { bound: y })
            | (ValueType::Range { bound: x }, ValueType::Range { bound: y }) => {
                self.same_extent(*x, *y)
            }
            (ValueType::Tuple(x), ValueType::Tuple(y)) => {
                x.len() == y.len() && x.iter().zip(y.iter()).all(|(p, q)| self.same_ty(p, q))
            }
            _ => a == b,
        }
    }

    // ---- target-dependent forms ----

    /// A target-dependent form. Legal only in a body with a target context.
    pub fn target_form(&mut self, span: Span, what: &str, namespace: Option<&str>) -> bool {
        let Some(target) = self.target else {
            self.error(
                DiagnosticRule::Capability,
                span,
                format!("{what} is target-dependent and cannot appear in a portable body; write it in a `lower … for <target>` body"),
            );
            return false;
        };
        if namespace
            .and_then(crate::registry::BackendName::parse)
            .is_some_and(|ns| ns != target)
        {
            self.error(
                DiagnosticRule::Capability,
                span,
                format!(
                    "{what} belongs to target `{}` but this body is for `{}`",
                    namespace.unwrap_or_default(),
                    target.as_str()
                ),
            );
            return false;
        }
        true
    }

    // ---- storage class ----

    /// How the storage of an expression is reached.
    pub fn class_of(&self, e: &CheckedExpr) -> ValueClass {
        self.ownership_class(e)
    }

    /// Whether writes may target the storage rooted at `id`.
    pub fn writable_root(&self, id: LocalId) -> bool {
        self.writable_place(&ownership::LocalPlace::root(id))
    }

    /// Carry the captured participant and actual place into the checked
    /// semantic write. The source-order access walk decides independence for
    /// the complete body before any checked definition is published.
    pub(crate) fn exclusive_write_authority(
        &self,
        root: LocalId,
        region: &ir::Place,
    ) -> Vec<ir::ExclusiveWriteCapability> {
        let participants = self
            .logical_parallel
            .iter()
            .filter(|(floor, _)| root.index() < *floor)
            .map(|(_, binder)| *binder)
            .collect::<Vec<_>>();
        if participants.is_empty() {
            Vec::new()
        } else {
            vec![ir::ExclusiveWriteCapability::checked(
                region.clone(),
                participants,
            )]
        }
    }

    // ---- effects ----

    /// Check that the selected storage can be written. The source-order
    /// region walk decides independent-loop access legality for the body.
    pub fn write(&mut self, root: LocalId, binding: LocalId, span: Span) -> Option<LocalId> {
        let binding_name = self.locals[binding.index()].name.clone();
        match &self.kinds[binding.index()] {
            LocalKind::Param(i) if self.sig.params[*i].ownership == ParamOwnership::Owned => {
                self.error(
                    DiagnosticRule::Ownership,
                    span,
                    format!("`{binding_name}` is a read-only moved-in parameter; writing requires `&mut tensor`"),
                );
                return None;
            }
            LocalKind::Param(_) | LocalKind::State => {}
            LocalKind::Value if self.is_borrowed_local(binding) => {}
            _ => {
                self.error(
                    DiagnosticRule::Ownership,
                    span,
                    format!("`{binding_name}` is not mutable state; only `let mut` bindings and `&mut tensor` parameters are written"),
                );
                return None;
            }
        }
        if !self.writable_root(binding) || !self.writable_root(root) {
            self.error(
                DiagnosticRule::Ownership,
                span,
                format!(
                    "`{}` is not writable storage; writing requires `let mut` state or a `&mut tensor` parameter",
                    self.locals[root.index()].name
                ),
            );
            return None;
        }
        if let ValueType::Tensor(tensor) = &self.locals[root.index()].ty {
            if let Elem::Repr(representation) = &tensor.elem {
                if crate::registry::representation_info(*representation).access
                    != crate::registry::RepresentationAccess::ReadWrite
                {
                    self.error(
                        DiagnosticRule::Type,
                        span,
                        format!(
                            "representation `{}` is decode-only and has no canonical write contract",
                            crate::registry::representation_info(*representation).name
                        ),
                    );
                    return None;
                }
            }
            let element = tensor.elem.clone();
            self.elements.stored(&element);
        }
        if self
            .live_borrows()
            .iter()
            .any(|(borrow, borrowed, _)| borrowed.local == root && borrow.local != binding)
        {
            self.error(
                DiagnosticRule::Ownership,
                span,
                format!(
                    "cannot mutate `{}` while a tensor borrow is live",
                    self.locals[root.index()].name
                ),
            );
            return None;
        }
        self.scalar_symbols.remove(&root);
        let tensor_effect = matches!(self.locals[root.index()].ty, ValueType::Tensor(_));
        self.dyn_views.retain(|(start, end, _, _)| {
            let mentions =
                |e: &Option<CheckedExpr>| e.as_ref().is_some_and(|b| expr::mentions_local(b, root));
            !tensor_effect && !(mentions(start) || mentions(end))
        });
        Some(root)
    }

    // ---- result ----

    fn finish(mut self, root: CheckedBlock, result: Vec<CheckedExpr>) -> CheckedParts {
        for (capability, declared_at) in self.requires.clone() {
            if !self.used_capabilities.contains(&capability) {
                self.error(
                    DiagnosticRule::Capability,
                    declared_at,
                    format!(
                        "capability `{}.{}` is required but not used",
                        crate::registry::capability_info(capability).backend.as_str(),
                        crate::registry::capability_info(capability).name
                    ),
                );
            }
        }
        let element_domain = self.elements.domain(&self.sig.elem_params);
        CheckedParts {
            body: CheckedBody {
                locals: self.locals,
                root,
                result,
            },
            signature: self.sig,
            diagnostics: self.diagnostics,
            arena: self.arena,
            element_domain,
            placement: self.placement,
        }
    }
}

struct CheckedParts {
    body: CheckedBody,
    signature: BodySig,
    diagnostics: Vec<Diagnostic>,
    arena: ExprArena,
    element_domain: crate::checked::ElementDomain,
    placement: Placement,
}

/// The elements of every tensor leaf of a type.
fn tensor_elements(ty: &ValueType) -> Vec<&Elem> {
    match ty {
        ValueType::Tensor(tensor) => vec![&tensor.elem],
        ValueType::Tuple(items) => items.iter().flat_map(tensor_elements).collect(),
        _ => Vec::new(),
    }
}

/// Every axis and bound expression of a type.
fn type_extents(ty: &ValueType) -> Vec<IntExpr> {
    match ty {
        ValueType::Tensor(tensor) => tensor.axes.clone(),
        ValueType::Index { bound } | ValueType::Range { bound } => vec![*bound],
        ValueType::Tuple(items) => items.iter().flat_map(type_extents).collect(),
        _ => Vec::new(),
    }
}

/// Every divisor of a quotient or remainder beneath `e`.
fn divisors(arena: &ExprArena, e: IntExpr) -> Vec<IntExpr> {
    fn walk(arena: &ExprArena, node: crate::expr::AnyExpr, out: &mut Vec<IntExpr>) {
        match arena.view(node) {
            crate::expr::NodeView::Binary { op, lhs, rhs } => {
                if matches!(op, crate::expr::BinaryOp::Div | crate::expr::BinaryOp::Rem) {
                    if let crate::expr::AnyExpr::Int(divisor) = rhs {
                        out.push(divisor);
                    }
                }
                walk(arena, lhs, out);
                walk(arena, rhs, out);
            }
            crate::expr::NodeView::Unary { operand, .. } => walk(arena, operand, out),
            crate::expr::NodeView::Nary { operands, .. } => {
                for operand in operands {
                    walk(arena, *operand, out);
                }
            }
            _ => {}
        }
    }
    let mut out = Vec::new();
    walk(arena, crate::expr::AnyExpr::Int(e), &mut out);
    out
}

/// Whether publishing/assigning elements of `value` into storage of `target`
/// is a defined rounding.
pub(crate) fn elem_rounds(value: &Elem, target: &Elem) -> bool {
    match (value, target) {
        (Elem::Repr(a), Elem::Repr(b)) => a == b,
        (Elem::Repr(_), _) | (_, Elem::Repr(_)) => false,
        (Elem::Dtype(a), Elem::Dtype(b)) => a == b || (a.is_float() && b.is_float()),
        (Elem::Dtype(a), Elem::Param(_)) | (Elem::Param(_), Elem::Dtype(a)) => a.is_float(),
        (Elem::Param(_), Elem::Param(_)) => true,
    }
}

struct CheckedOutcome {
    body: CheckedBody,
    diagnostics: Vec<Diagnostic>,
    arena: ExprArena,
    signature: BodySig,
    initialization: initialization::Contract,
    element_domain: crate::checked::ElementDomain,
    placement: Placement,
    /// The family dimension plan, for a family contract whose dimensions
    /// input tensor axes determine.
    plan: Option<dimensions::DimensionPlan>,
}

fn check_definition(env: &Env, def: usize) -> CheckedOutcome {
    let declared = &env.resolved.declared[def];
    let mut c = Checker::new(env, def);
    let initialization_facts = c.facts.clone();
    let (mut root, mut result) = c.function_body(declared.body, declared.name_span);
    let initialization = initialization::check(&mut c, &mut root, &mut result, initialization_facts);
    let contract = env.resolved.families[declared.family.index()].contract.index() == def;
    let plan = if contract {
        match dimensions::plan_dimensions(&c.arena, &c.sig.dimensions, &c.sig.params) {
            Ok(plan) => Some(plan),
            Err(error) => {
                c.error(
                    DiagnosticRule::Dimension,
                    declared.name_span,
                    format!(
                        "dimensions {} are not determined by the input tensor axes; every dimension of `{}` must be derivable from its tensor parameters' extents",
                        error
                            .underdetermined
                            .iter()
                            .map(|name| format!("`{name}`"))
                            .collect::<Vec<_>>()
                            .join(", "),
                        c.sig.name
                    ),
                );
                None
            }
        }
    } else {
        None
    };
    let parts = c.finish(root, result);
    CheckedOutcome {
        body: parts.body,
        diagnostics: parts.diagnostics,
        arena: parts.arena,
        signature: parts.signature,
        initialization,
        element_domain: parts.element_domain,
        placement: parts.placement,
        plan,
    }
}

/// Check every declared body of the closed program once, callees first.
/// Returns the definitions and families when no diagnostic was reported.
pub(crate) fn check_program(
    files: &[(usize, ast::File)],
    texts: &[&str],
    program: ProgramId,
    semantic_hash: ModuleHash,
    diagnostics: &mut Vec<Located>,
) -> Option<(Vec<ir::Definition>, Vec<ir::Family>)> {
    let resolved = resolve::resolve(files, program, diagnostics);
    let count = resolved.declared.len();
    // L20: every recursion is one diagnostic, and every definition that
    // reaches no recursion is still checked.
    let (order, cycles) = resolved.call_graph.bottom_up_order();
    diagnostics.extend(cycles.iter().map(|cycle| cycle.diagnostic(&resolved.declared)));
    let mut outcomes: Vec<Option<CheckedOutcome>> = (0..count).map(|_| None).collect();
    for def in order {
        let env = Env {
            resolved: &resolved,
            texts,
            checked: &outcomes,
        };
        let checked = check_definition(&env, def);
        outcomes[def] = Some(checked);
    }
    let mut clean = diagnostics.is_empty();
    for (outcome, declared) in outcomes.iter_mut().zip(&resolved.declared) {
        if let Some(outcome) = outcome {
            clean &= outcome.diagnostics.is_empty();
            diagnostics.extend(outcome.diagnostics.drain(..).map(|diagnostic| Located {
                file: declared.file,
                diagnostic,
            }));
        }
    }
    if !clean {
        return None;
    }
    // Without a recursion, the order holds every definition.
    let mut outcomes = outcomes.into_iter().flatten().collect::<Vec<_>>();
    assert_eq!(outcomes.len(), count, "definition checking order omitted a body");
    let families = resolved
        .families
        .iter()
        .map(|family| ir::Family {
            name: family.name.clone(),
            contract: family.contract,
            bodies: family.bodies.clone(),
            lowerings: family.lowerings.clone(),
            dimension_plan: outcomes[family.contract.index()]
                .plan
                .take()
                .expect("a checked family contract has its dimension plan"),
        })
        .collect();
    let mut definitions = Vec::with_capacity(count);
    for (def, (checked, declared)) in outcomes.into_iter().zip(&resolved.declared).enumerate() {
        let params = checked
            .signature
            .params
            .iter()
            .enumerate()
            .map(|(i, p)| ir::Param {
                name: p.name.clone(),
                ownership: p.ownership.clone(),
                ty: p.ty.clone(),
                local: LocalId::new(
                    u32::try_from(i).expect("definition has more than u32::MAX parameters"),
                ),
                span: p.span,
            })
            .collect();
        let mut stable_hasher = sha2::Sha256::new();
        use sha2::Digest as _;
        stable_hasher.update(b"seismic-stable-function-v3");
        stable_hasher.update(semantic_hash.digest());
        stable_hasher.update((def as u64).to_le_bytes());
        let stable = StableFunctionId::new(stable_hasher.finalize().into());
        definitions.push(ir::Definition {
            stable,
            name: declared.sig.name.clone(),
            kind: declared.kind,
            requires: declared
                .requires
                .iter()
                .map(|(capability, _)| *capability)
                .collect(),
            dimensions: checked.signature.dimensions,
            elem_params: declared.sig.elem_params.clone(),
            elem_bindings: declared.elem_bindings.clone(),
            params,
            result: checked.signature.result,
            predicates: checked.signature.predicates,
            initialization: checked.initialization,
            element_domain: checked.element_domain,
            placement: checked.placement,
            body: checked.body,
            arena: checked.arena,
            file: declared.file,
            span: declared.span,
        });
    }
    Some((definitions, families))
}

pub(crate) fn check_closed(
    sources: crate::checked::SourceSet,
    module_id: ModuleId,
    program: ProgramId,
) -> Result<crate::checked::internals::Module, crate::checked::SourceError> {
    use crate::checked::{Diagnostics, SourceDiagnostic, SourceError};
    let mut parsed = Vec::new();
    let mut parse_diagnostics = Vec::new();
    for (file, source) in sources.files().iter().enumerate() {
        match crate::syntax::parse(&source.text) {
            Ok(ast) => parsed.push((file, ast)),
            Err(diagnostic) => parse_diagnostics.push(SourceDiagnostic::located(source, diagnostic)),
        }
    }
    if let Some(diagnostics) = Diagnostics::new(parse_diagnostics) {
        return Err(SourceError::new(diagnostics));
    }

    use sha2::Digest as _;
    let mut module_hasher = sha2::Sha256::new();
    module_hasher.update(crate::bundle::COMPILER_SEMANTIC_VERSION.as_bytes());
    module_hasher.update(crate::registry::REGISTRY_REVISION.as_bytes());
    for source in sources.files() {
        module_hasher.update((source.path.len() as u64).to_le_bytes());
        module_hasher.update(source.path.as_bytes());
        module_hasher.update((source.text.len() as u64).to_le_bytes());
        module_hasher.update(source.text.as_bytes());
    }
    let semantic_hash = ModuleHash::new(module_hasher.finalize().into());

    let texts = sources
        .files()
        .iter()
        .map(|file| file.text.as_str())
        .collect::<Vec<_>>();
    let mut located = Vec::new();
    let checked = check_program(&parsed, &texts, program, semantic_hash, &mut located);
    let diagnostics = located
        .into_iter()
        .map(|item| SourceDiagnostic::located(&sources.files()[item.file], item.diagnostic))
        .collect();
    if let Some(diagnostics) = Diagnostics::new(diagnostics) {
        return Err(SourceError::new(diagnostics));
    }
    let (definitions, families) = checked.expect("a program without diagnostics is checked");

    let mut entries = Vec::new();
    let mut entry_families = Vec::new();
    for (family_ordinal, family) in families.iter().enumerate() {
        let Some(contract) = definitions.get(family.contract.index()) else {
            panic!("checked family contract is outside the checked definition arena");
        };
        let ordinal = u32::try_from(entries.len()).expect("module has more than u32::MAX entries");
        let id = crate::ids::EntryId::new(module_id, ordinal);
        let mut stable_hasher = sha2::Sha256::new();
        stable_hasher.update(semantic_hash.digest());
        stable_hasher.update((family_ordinal as u64).to_le_bytes());
        let stable = crate::ids::StableEntryId::new(stable_hasher.finalize().into());
        entries.push(entry_info(id, stable, contract));
        entry_families.push(family_ordinal);
    }
    let mut native_implementations = Vec::new();
    let mut native_diagnostics = Vec::new();
    for (file, parsed_file) in &parsed {
        for declaration in &parsed_file.decls {
            let crate::syntax::ast::Decl::Native(native) = declaration else {
                continue;
            };
            let Some(backend) = crate::registry::BackendName::parse(&native.target.name) else {
                native_diagnostics.push(SourceDiagnostic::new(
                    &sources.files()[*file],
                    native.target.span,
                    DiagnosticRule::NativeDeclaration,
                    format!("unknown native backend `{}`", native.target.name),
                ));
                continue;
            };
            let matching = entries
                .iter()
                .filter(|entry| entry.name == native.function.name)
                .collect::<Vec<_>>();
            let entry = match matching.as_slice() {
                [entry] => *entry,
                [] => {
                    native_diagnostics.push(SourceDiagnostic::new(
                        &sources.files()[*file],
                        native.function.span,
                        DiagnosticRule::NativeDeclaration,
                        format!(
                            "native implementation refers to unknown portable function `{}`",
                            native.function.name
                        ),
                    ));
                    continue;
                }
                _ => {
                    native_diagnostics.push(SourceDiagnostic::new(
                        &sources.files()[*file],
                        native.function.span,
                        DiagnosticRule::NativeDeclaration,
                        format!(
                            "native implementation of overloaded function `{}` is ambiguous",
                            native.function.name
                        ),
                    ));
                    continue;
                }
            };
            if native_implementations.iter().any(
                |implementation: &crate::checked::NativeImplementation| {
                    implementation.entry == entry.id && implementation.backend == backend
                },
            ) {
                native_diagnostics.push(SourceDiagnostic::new(
                    &sources.files()[*file],
                    native.span,
                    DiagnosticRule::NativeDeclaration,
                    format!(
                        "function `{}` already has a native implementation for `{}`",
                        native.function.name,
                        backend.as_str()
                    ),
                ));
                continue;
            }
            match check_native(native, entry, backend, &sources.files()[*file].path) {
                Ok(implementation) => native_implementations.push(implementation),
                Err(errors) => {
                    for (span, message) in errors {
                        native_diagnostics.push(SourceDiagnostic::new(
                            &sources.files()[*file],
                            span,
                            DiagnosticRule::NativeDeclaration,
                            message,
                        ));
                    }
                }
            }
        }
    }
    if let Some(diagnostics) = Diagnostics::new(native_diagnostics) {
        return Err(SourceError::new(diagnostics));
    }
    Ok(crate::checked::internals::Module {
        id: module_id,
        semantic_hash,
        sources,
        entries,
        native_implementations,
        entry_families,
        definitions,
        families,
    })
}

/// Check one native declaration against its entry. Every problem is reported.
fn check_native(
    native: &crate::syntax::ast::NativeDecl,
    entry: &crate::checked::EntryInfo,
    backend: crate::registry::BackendName,
    declared_in: &str,
) -> Result<crate::checked::NativeImplementation, Vec<(Span, String)>> {
    use crate::checked::{
        NativeComparison, NativeConstraint, NativeLaunch, NativeNatExpr, NativeParameter,
        NativeScratch,
    };
    use crate::syntax::ast::{BinaryOp, ExprKind};
    let mut errors = Vec::new();

    let mut statics: Vec<String> = Vec::new();
    for name in &native.statics {
        if !entry.dimensions.contains(&name.name) {
            errors.push((
                name.span,
                format!(
                    "`{}` is not a shape dimension of `{}`",
                    name.name, entry.name
                ),
            ));
        } else if statics.contains(&name.name) {
            errors.push((name.span, format!("static dimension `{}` is listed twice", name.name)));
        } else {
            statics.push(name.name.clone());
        }
    }

    let mut params: Vec<NativeParameter> = Vec::new();
    for param in &native.params {
        let name = &param.name.name;
        if entry.dimensions.contains(name) {
            errors.push((
                param.name.span,
                format!("native parameter `{name}` shadows a dimension of `{}`", entry.name),
            ));
            continue;
        }
        if params.iter().any(|existing| &existing.name == name) {
            errors.push((param.name.span, format!("native parameter `{name}` is declared twice")));
            continue;
        }
        let mut values: Vec<u64> = Vec::new();
        for value in &param.values {
            if values.contains(value) {
                errors.push((
                    param.span,
                    format!("native parameter `{name}` lists {value} twice"),
                ));
            } else {
                values.push(*value);
            }
        }
        params.push(NativeParameter {
            name: name.clone(),
            arithmetic: param.arithmetic,
            values,
        });
    }
    let parameter_names = params
        .iter()
        .map(|parameter| parameter.name.clone())
        .collect::<Vec<_>>();

    let expression = |expr: &crate::syntax::ast::Expr, errors: &mut Vec<(Span, String)>| {
        native_nat_expr(expr, &entry.dimensions, &parameter_names)
            .map_err(|message| errors.push((expr.span, message)))
            .ok()
    };

    let mut constraints = Vec::new();
    for conjunct in &native.constraints {
        let ExprKind::Binary { op, lhs, rhs } = &conjunct.kind else {
            errors.push((
                conjunct.span,
                "a native `where` conjunct is a comparison".to_owned(),
            ));
            continue;
        };
        let comparison = match op {
            BinaryOp::Lt => NativeComparison::Lt,
            BinaryOp::Le => NativeComparison::Le,
            BinaryOp::Gt => NativeComparison::Gt,
            BinaryOp::Ge => NativeComparison::Ge,
            BinaryOp::Eq => NativeComparison::Eq,
            BinaryOp::Ne => NativeComparison::Ne,
            _ => {
                errors.push((
                    conjunct.span,
                    "a native `where` conjunct is a comparison joined by `and`".to_owned(),
                ));
                continue;
            }
        };
        let (Some(left), Some(right)) = (expression(lhs, &mut errors), expression(rhs, &mut errors))
        else {
            continue;
        };
        let mut read = Vec::new();
        left.dimensions(&mut read);
        right.dimensions(&mut read);
        if let Some(dynamic) = read.iter().find(|name| !statics.contains(name)) {
            errors.push((
                conjunct.span,
                format!(
                    "native `where` reads dimension `{dynamic}`, which is not static; its value is unknown at preparation"
                ),
            ));
            continue;
        }
        constraints.push(NativeConstraint {
            comparison,
            left,
            right,
        });
    }

    let mut scratch: Vec<NativeScratch> = Vec::new();
    for buffer in &native.scratch {
        if scratch.iter().any(|existing| existing.name == buffer.name.name) {
            errors.push((
                buffer.name.span,
                format!("scratch buffer `{}` is declared twice", buffer.name.name),
            ));
            continue;
        }
        if let Some(bytes) = expression(&buffer.bytes, &mut errors) {
            scratch.push(NativeScratch {
                name: buffer.name.name.clone(),
                bytes,
            });
        }
    }

    let mut launches = Vec::new();
    for launch in &native.launches {
        let groups = launch
            .threadgroups
            .each_ref()
            .map(|expr| expression(expr, &mut errors));
        let group_extent = launch
            .threads_per_threadgroup
            .each_ref()
            .map(|expr| expression(expr, &mut errors));
        let shared_bytes = match &launch.shared_bytes {
            Some(expr) => expression(expr, &mut errors),
            None => Some(NativeNatExpr::Constant(0)),
        };
        let ([Some(x), Some(y), Some(z)], [Some(ex), Some(ey), Some(ez)], Some(shared_bytes)) =
            (groups, group_extent, shared_bytes)
        else {
            continue;
        };
        launches.push(NativeLaunch {
            kernel: launch.kernel.name.clone(),
            groups: [x, y, z],
            group_extent: [ex, ey, ez],
            shared_bytes,
        });
    }

    if !errors.is_empty() {
        return Err(errors);
    }
    Ok(crate::checked::NativeImplementation {
        entry: entry.id,
        backend,
        declared_in: declared_in.to_owned(),
        source_path: native.source.clone(),
        statics,
        params,
        constraints,
        scratch,
        launches,
    })
}

fn native_nat_expr(
    expression: &crate::syntax::ast::Expr,
    dimensions: &[String],
    parameters: &[String],
) -> Result<crate::checked::NativeNatExpr, String> {
    use crate::checked::NativeNatExpr as N;
    use crate::syntax::ast::{BinaryOp, ExprKind};
    let binary = |left: &crate::syntax::ast::Expr,
                  right: &crate::syntax::ast::Expr,
                  make: fn(Box<N>, Box<N>) -> N| {
        Ok(make(
            Box::new(native_nat_expr(left, dimensions, parameters)?),
            Box::new(native_nat_expr(right, dimensions, parameters)?),
        ))
    };
    match &expression.kind {
        ExprKind::Int(value) => Ok(N::Constant(*value)),
        ExprKind::Name(name) if dimensions.contains(&name.name) => {
            Ok(N::Dimension(name.name.clone()))
        }
        ExprKind::Name(name) if parameters.contains(&name.name) => {
            Ok(N::Parameter(name.name.clone()))
        }
        ExprKind::Name(name) => Err(format!(
            "native expression references `{}`, which is neither a dimension nor a native parameter",
            name.name
        )),
        ExprKind::Binary { op, lhs, rhs } => match op {
            BinaryOp::Add => binary(lhs, rhs, N::Add),
            BinaryOp::Sub => binary(lhs, rhs, N::Sub),
            BinaryOp::Mul => binary(lhs, rhs, N::Mul),
            BinaryOp::Div => binary(lhs, rhs, N::Div),
            BinaryOp::Rem => binary(lhs, rhs, N::Rem),
            _ => Err("native expressions use only `+`, `-`, `*`, `/`, and `%`".to_owned()),
        },
        ExprKind::Call {
            callee,
            bindings,
            args,
        } if bindings.is_empty()
            && args.len() == 2
            && args.iter().all(|arg| arg.name.is_none()) =>
        {
            let make: fn(Box<N>, Box<N>) -> N = match &callee.kind {
                ExprKind::Name(name) if name.name == "ceil_div" => N::CeilDiv,
                ExprKind::Name(name) if name.name == "min" => N::Min,
                ExprKind::Name(name) if name.name == "max" => N::Max,
                _ => {
                    return Err(
                        "native expressions call only `ceil_div`, `min`, and `max`".to_owned()
                    );
                }
            };
            binary(&args[0].value, &args[1].value, make)
        }
        _ => Err("unsupported native expression".to_owned()),
    }
}

fn element_summary(element: &Elem) -> crate::checked::ElementSummary {
    match element {
        Elem::Dtype(dtype) => crate::checked::ElementSummary::Fixed(dtype.name().to_owned()),
        Elem::Repr(representation) => crate::checked::ElementSummary::Fixed(
            crate::registry::representation_info(*representation)
                .name
                .to_owned(),
        ),
        Elem::Param(name) => crate::checked::ElementSummary::Parameter(name.clone()),
    }
}

/// Source-declared result leaves retain their tuple paths after semantic
/// lowering flattens the corresponding producer values into one ordered list.
pub(super) fn result_leaves(ty: &ValueType) -> Vec<(Vec<u32>, &ValueType)> {
    fn walk<'a>(
        ty: &'a ValueType,
        path: &mut Vec<u32>,
        leaves: &mut Vec<(Vec<u32>, &'a ValueType)>,
    ) {
        match ty {
            ValueType::Tuple(items) => {
                for (ordinal, item) in items.iter().enumerate() {
                    path.push(u32::try_from(ordinal).expect("tuple has more than u32::MAX elements"));
                    walk(item, path, leaves);
                    path.pop();
                }
            }
            ValueType::Void => {}
            ValueType::Integer => unreachable!("mathematical integer has no source result spelling"),
            ValueType::Opaque { .. } => panic!("backend-opaque result escaped an exported portable entry"),
            _ => leaves.push((path.clone(), ty)),
        }
    }
    let mut leaves = Vec::new();
    walk(ty, &mut Vec::new(), &mut leaves);
    leaves
}

fn entry_info(
    id: crate::ids::EntryId,
    stable: crate::ids::StableEntryId,
    definition: &ir::Definition,
) -> crate::checked::EntryInfo {
    use crate::checked::{
        ParameterSummary, ParameterSummaryKind, ResultSummary, ResultSummaryKind, TensorAccess,
    };
    fn flatten_parameter(
        source: u32,
        name: &str,
        ownership: &ParamOwnership,
        ty: &ValueType,
        path: &mut Vec<u32>,
        output: &mut Vec<ParameterSummary>,
    ) {
        if let ValueType::Tuple(items) = ty {
            for (ordinal, item) in items.iter().enumerate() {
                path.push(u32::try_from(ordinal).expect("tuple has more than u32::MAX elements"));
                let ParamOwnership::Tuple(parts) = ownership else {
                    panic!("checked tuple parameter lost ownership product")
                };
                flatten_parameter(source, name, &parts[ordinal], item, path, output);
                path.pop();
            }
            return;
        }
        let kind = match ty {
            ValueType::Tensor(tensor) => ParameterSummaryKind::Tensor {
                access: match ownership {
                    ParamOwnership::Owned | ParamOwnership::Value => TensorAccess::Owned,
                    ParamOwnership::Shared => TensorAccess::Shared,
                    ParamOwnership::Exclusive => TensorAccess::Mutable,
                    ParamOwnership::Tuple(_) => {
                        unreachable!("tensor leaf ownership is not a tuple")
                    }
                },
                rank: u32::try_from(tensor.rank()).expect("tensor rank exceeds u32::MAX"),
                element: element_summary(&tensor.elem),
            },
            ValueType::Scalar(dtype) => ParameterSummaryKind::Scalar(*dtype),
            ValueType::Integer => unreachable!("mathematical integer has no source parameter spelling"),
            ValueType::Index { .. } => ParameterSummaryKind::Index,
            ValueType::Range { .. } => ParameterSummaryKind::Range,
            ValueType::Void => return,
            ValueType::Opaque { .. } => {
                panic!("backend-opaque value escaped a portable entry signature")
            }
            ValueType::Tuple(_) => unreachable!(),
        };
        output.push(ParameterSummary {
            source,
            path: path.clone(),
            name: name.to_owned(),
            kind,
        });
    }
    let mut parameters = Vec::new();
    for (ordinal, parameter) in definition.params.iter().enumerate() {
        flatten_parameter(
            u32::try_from(ordinal).expect("parameter count exceeds u32::MAX"),
            &parameter.name,
            &parameter.ownership,
            &parameter.ty,
            &mut Vec::new(),
            &mut parameters,
        );
    }
    fn signature(ty: &ValueType, ownership: &ParamOwnership) -> crate::checked::SignatureType {
        use crate::checked::SignatureType as S;
        match ty {
            ValueType::Void => S::Unit,
            ValueType::Tuple(items) => match ownership {
                ParamOwnership::Tuple(parts) => S::Tuple(
                    items
                        .iter()
                        .zip(parts)
                        .map(|(t, o)| signature(t, o))
                        .collect(),
                ),
                ParamOwnership::Owned => S::Tuple(
                    items
                        .iter()
                        .map(|t| signature(t, &ParamOwnership::Owned))
                        .collect(),
                ),
                _ => panic!("signature tuple lost ownership product"),
            },
            ValueType::Tensor(t) => S::Tensor {
                access: match ownership {
                    ParamOwnership::Shared => TensorAccess::Shared,
                    ParamOwnership::Exclusive => TensorAccess::Mutable,
                    _ => TensorAccess::Owned,
                },
                rank: t.rank() as u32,
                element: element_summary(&t.elem),
            },
            ValueType::Scalar(d) => S::Scalar(*d),
            ValueType::Integer => unreachable!("mathematical integer has no source signature spelling"),
            ValueType::Index { .. } => S::Index,
            ValueType::Range { .. } => S::Range,
            ValueType::Opaque { .. } => unreachable!("opaque portable signature"),
        }
    }
    let results = result_leaves(&definition.result)
        .into_iter()
        .map(|(path, ty)| ResultSummary {
            path,
            kind: match ty {
                ValueType::Tensor(tensor) => ResultSummaryKind::Tensor {
                    rank: u32::try_from(tensor.rank()).expect("tensor rank exceeds u32::MAX"),
                    element: element_summary(&tensor.elem),
                },
                ValueType::Scalar(dtype) => ResultSummaryKind::Scalar(*dtype),
                ValueType::Index { .. } => ResultSummaryKind::Index,
                ValueType::Range { .. } => ResultSummaryKind::Range,
                _ => unreachable!("result leaves contain only exported values"),
            },
        })
        .collect();
    EntryInfo {
        id,
        stable,
        name: definition.name.clone(),
        dimensions: definition
            .dimensions
            .iter()
            .map(|dimension| dimension.name.clone())
            .collect(),
        element_parameters: definition.elem_params.clone(),
        parameter_types: definition
            .params
            .iter()
            .map(|p| (p.name.clone(), signature(&p.ty, &p.ownership)))
            .collect(),
        result_type: signature(&definition.result, &ParamOwnership::Owned),
        parameters,
        results,
        element_domain: definition.element_domain.clone(),
    }
}
