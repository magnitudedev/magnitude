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
pub(crate) mod ir;
mod ownership;
#[cfg(test)]
mod ownership_tests;
pub(crate) mod prove;
pub(crate) mod resolve;
mod scalar_quantity;
#[cfg(test)]
mod stable_identity_tests;
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

    /// Symbols that stand for runtime data values (element reads, words,
    /// runtime slice lengths, binders over data bounds). A bound over one of
    /// them that is not proved is checked at run time instead of rejected.
    pub data_symbols: HashSet<SymbolId>,
    /// Reached source versions; Local.value remains the declaration identity.
    pub source_values: HashMap<LocalId, ir::SourceValue>,
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
    /// The host ports (`&host tensor`): parameters the body never names,
    /// so none is in scope.
    pub host_ports: HashSet<String>,
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

            data_symbols: HashSet::new(),
            source_values: HashMap::new(),
            dyn_views: Vec::new(),
            logical_parallel: Vec::new(),
            divergence: vec![0],
            uniform_binders: HashSet::new(),
            block_depth: 0,
            diagnostics: Vec::new(),
            poisoned: HashSet::new(),
            host_ports: HashSet::new(),
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
                Predicate::Boolean(e) => c.facts.assume_boolean(e),
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
            let value = c.source_value(&p.ty, None, true);
            c.locals[id.index()].value = value.clone();
            c.source_values.insert(id, value);
            // A host port keeps its local, so parameters and locals stay
            // aligned, but leaves the scope: naming it is an error.
            if matches!(p.ownership, ParamOwnership::Host { .. }) {
                if let Some(scope) = c.scopes.last_mut() {
                    scope.remove(&p.name);
                }
                c.host_ports.insert(p.name.clone());
            }
            if p.ownership == ParamOwnership::Exclusive {
                for element in tensor_elements(&p.ty) {
                    c.elements.record_stored(element);
                }
            }
        }
        for element in tensor_elements(&c.sig.result.clone()) {
            c.elements.stored(element);
        }
        let contract = env.resolved.families[declared.family.index()]
            .contract
            .index()
            == def;
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
        self.locals.iter().find(|local| local.value.leaf_bindings().iter().any(|(_, candidate)| *candidate == symbol))
            .map(|local| local.name.clone()).unwrap_or_else(|| "?".to_owned())
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

    /// Why `name`, which [`Self::lookup`] does not find, names nothing the
    /// body may use, when it is a host port; `None` when it is undeclared.
    pub fn host_port(&self, name: &str) -> Option<String> {
        self.host_ports.contains(name).then(|| {
            format!(
                "`{name}` is a host port: the host reads or rewrites it once it sees the launch, and the kernel never accesses it"
            )
        })
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
            value: ir::SourceValue::Untracked,
        });
        self.kinds.push(kind);
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name.to_string(), id);
        }
        id
    }

    /// A fresh proof variable.
    pub fn fresh_symbol(&mut self) -> SymbolId {
        self.arena.checker_variable(SymbolSort::Int)
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
    pub fn require_nonneg(
        &mut self,
        rule: DiagnosticRule,
        e: IntExpr,
        span: Span,
        what: &str,
    ) -> bool {
        if prove::derive_nonnegative(&mut self.arena, &self.facts, e).is_established() {
            return true;
        }
        let rendered = self.render(e);
        self.error(
            rule,
            span,
            format!("{what}: cannot prove `{rendered} >= 0`"),
        );
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
        types.push((
            self.sig.result.clone(),
            self.env.resolved.declared[self.def].name_span,
        ));
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
                if !prove::derive_nonnegative(&mut self.arena, &self.facts, extent).is_established() {
                    let rendered = self.render(extent);
                    let shown = self.shown(&ty);
                    self.error(
                        DiagnosticRule::Dimension,
                        span,
                        format!(
                            "type `{shown}` requires `{rendered} >= 0`; add `where {rendered} >= 0`"
                        ),
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
    ///
    /// `record` is a store of whole trailing matrices, the one write packed
    /// storage takes (`registry::store_rule`).
    pub fn write(
        &mut self,
        root: LocalId,
        binding: LocalId,
        record: bool,
        span: Span,
    ) -> Option<LocalId> {
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
                if let Err(refusal) = crate::registry::store_rule(*representation).admits(record) {
                    self.error(DiagnosticRule::Type, span, refusal);
                    return None;
                }
            }
            let element = tensor.elem.clone();
            if record {
                self.elements.record_stored(&element);
            } else {
                self.elements.stored(&element);
            }
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
        self.source_values.remove(&root);
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
                        crate::registry::capability_info(capability)
                            .backend
                            .as_str(),
                        crate::registry::capability_info(capability).name
                    ),
                );
            }
        }
        let element_domain = self
            .elements
            .domain(&self.sig.elem_params, &self.sig.element_predicates);
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
/// is a defined rounding. An element parameter binds a float-decoded
/// representation, so a pairing with one that some binding admits is decided
/// where lowering binds it (`install_value`).
pub(crate) fn elem_rounds(value: &Elem, target: &Elem) -> bool {
    match (value, target) {
        (Elem::Repr(a), Elem::Repr(b)) => a == b,
        (Elem::Repr(_), Elem::Dtype(_)) | (Elem::Dtype(_), Elem::Repr(_)) => false,
        (Elem::Dtype(a), Elem::Dtype(b)) => a == b || (a.is_float() && b.is_float()),
        (Elem::Dtype(a), Elem::Param(_)) | (Elem::Param(_), Elem::Dtype(a)) => a.is_float(),
        (Elem::Repr(_) | Elem::Param(_), Elem::Param(_)) | (Elem::Param(_), Elem::Repr(_)) => true,
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
}

fn check_definition(env: &Env, def: usize) -> CheckedOutcome {
    let declared = &env.resolved.declared[def];
    let mut c = Checker::new(env, def);
    let initialization_facts = c.facts.clone();
    let (mut root, mut result) = c.function_body(declared.body, declared.name_span);
    let initialization =
        initialization::check(&mut c, &mut root, &mut result, initialization_facts);
    let parts = c.finish(root, result);
    CheckedOutcome {
        body: parts.body,
        diagnostics: parts.diagnostics,
        arena: parts.arena,
        signature: parts.signature,
        initialization,
        element_domain: parts.element_domain,
        placement: parts.placement,
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
) -> Option<(Vec<ir::Definition>, Vec<ir::Family>, Vec<[u8; 32]>)> {
    let resolved = resolve::resolve(files, program, diagnostics);
    let count = resolved.declared.len();
    // L20: every recursion is one diagnostic, and every definition that
    // reaches no recursion is still checked.
    let (order, cycles) = resolved.call_graph.bottom_up_order();
    diagnostics.extend(
        cycles
            .iter()
            .map(|cycle| cycle.diagnostic(&resolved.declared)),
    );
    let mut outcomes: Vec<Option<CheckedOutcome>> = (0..count).map(|_| None).collect();
    // A definition reads only its callees' outcomes, so the definitions of
    // one call depth are independent: check each depth's definitions on
    // separate threads, after every deeper callee is complete.
    let mut depth = vec![0usize; count];
    let mut levels: Vec<Vec<usize>> = Vec::new();
    for &def in &order {
        depth[def] = resolved
            .call_graph
            .callees(def)
            .iter()
            .map(|callee| depth[*callee] + 1)
            .max()
            .unwrap_or(0);
        if levels.len() <= depth[def] {
            levels.resize_with(depth[def] + 1, Vec::new);
        }
        levels[depth[def]].push(def);
    }
    let workers = std::thread::available_parallelism().map_or(1, usize::from);
    for level in levels {
        let env = Env {
            resolved: &resolved,
            texts,
            checked: &outcomes,
        };
        let next = std::sync::atomic::AtomicUsize::new(0);
        let checked = std::thread::scope(|scope| {
            let threads = (0..workers.min(level.len()))
                .map(|_| {
                    scope.spawn(|| {
                        let mut checked = Vec::new();
                        loop {
                            let index = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            let Some(&def) = level.get(index) else {
                                return checked;
                            };
                            checked.push((def, check_definition(&env, def)));
                        }
                    })
                })
                .collect::<Vec<_>>();
            threads
                .into_iter()
                .flat_map(|thread| thread.join().expect("definition check thread panicked"))
                .collect::<Vec<_>>()
        });
        for (def, outcome) in checked {
            outcomes[def] = Some(outcome);
        }
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
    let outcomes = outcomes.into_iter().flatten().collect::<Vec<_>>();
    assert_eq!(
        outcomes.len(),
        count,
        "definition checking order omitted a body"
    );
    let contents = family_contents(&resolved, &order);
    let families = resolved
        .families
        .iter()
        .map(|family| ir::Family {
            name: family.name.clone(),
            contract: family.contract,
            bodies: family.bodies.clone(),
            lowerings: family.lowerings.clone(),
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
                binding: p.binding,
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
            body: checked.body,
            arena: checked.arena,
            file: declared.file,
            span: declared.span,
        });
    }
    Some((definitions, families, contents))
}

/// The content of each family, by family index: everything a call to it,
/// or the entry it is, depends on. A definition's content is its
/// declaration's structure (module constants are integers in it; source
/// positions are not) with the content of every family it may call (the
/// name call graph, the single owner of what a body may reach); a family's
/// is its contract's content, then its other bodies' and its lowerings',
/// each set in content order. So a change anywhere in a family's closure
/// changes its content, and nothing outside the closure (another
/// definition, a source path, where text sits) is in it. `order` is
/// bottom-up and holds every definition: a module that checked has no
/// recursion.
fn family_contents(resolved: &Resolved<'_>, order: &[usize]) -> Vec<[u8; 32]> {
    use sha2::Digest as _;
    fn family(family: &resolve::DeclaredFamily, definitions: &[Option<[u8; 32]>]) -> [u8; 32] {
        let content = |member: &crate::ids::FunctionId| {
            definitions[member.index()].expect("a family's members precede its callers bottom-up")
        };
        let sorted = |members: &mut dyn Iterator<Item = &crate::ids::FunctionId>| {
            let mut contents = members.map(content).collect::<Vec<_>>();
            contents.sort_unstable();
            contents
        };
        let bodies = sorted(&mut family.bodies.iter().filter(|body| **body != family.contract));
        let lowerings = sorted(&mut family.lowerings.iter());
        let mut hasher = sha2::Sha256::new();
        hasher.update(b"seismic-family-content-v1");
        hasher.update(content(&family.contract));
        for set in [bodies, lowerings] {
            hasher.update((set.len() as u64).to_le_bytes());
            set.iter().for_each(|member| hasher.update(member));
        }
        hasher.finalize().into()
    }
    let mut definitions = vec![None; resolved.declared.len()];
    for &definition in order {
        let declared = &resolved.declared[definition];
        let called = resolved
            .call_graph
            .callees(definition)
            .iter()
            .map(|callee| resolved.declared[*callee].family.index())
            .collect::<BTreeSet<_>>();
        let mut callees = called
            .into_iter()
            .map(|called| family(&resolved.families[called], &definitions))
            .collect::<Vec<_>>();
        callees.sort_unstable();
        let structure = ast::structure(declared.declaration);
        let mut hasher = sha2::Sha256::new();
        hasher.update(b"seismic-definition-content-v1");
        hasher.update((structure.len() as u64).to_le_bytes());
        hasher.update(structure.as_bytes());
        hasher.update((callees.len() as u64).to_le_bytes());
        callees.iter().for_each(|callee| hasher.update(callee));
        definitions[definition] = Some(hasher.finalize().into());
    }
    resolved
        .families
        .iter()
        .map(|declared| family(declared, &definitions))
        .collect()
}

/// Check `sources`, reading each native implementation's source with
/// `read` (its entry's name and its declaration): the module, and the
/// sources read, by entry and backend. Which of an implementation's
/// parameters change generated code is derived here, from the source read
/// ([`crate::native_source::derive_codes`]), so no checked implementation
/// holds a code parameter its source does not state.
pub(crate) fn check_closed<E>(
    sources: crate::checked::SourceSet,
    module_id: ModuleId,
    program: ProgramId,
    read: &mut crate::checked::NativeReader<'_, E>,
) -> Result<
    (crate::checked::internals::Module, crate::checked::NativeSources),
    crate::checked::CheckFailure<E>,
> {
    use crate::checked::{CheckFailure, Diagnostics, SourceDiagnostic, SourceError};
    // Two passes: the first finds every file's module constants, which the
    // second reads as the integers they name.
    let mut constants = std::collections::BTreeMap::new();
    let mut declared = Vec::new();
    let mut parse_diagnostics = Vec::new();
    let empty = std::collections::BTreeMap::new();
    for source in sources.files() {
        match crate::syntax::parse(&source.text, &empty) {
            Ok(ast) => {
                for constant in &ast.constants {
                    if constants.contains_key(&constant.name.name) {
                        parse_diagnostics.push(SourceDiagnostic::located(
                            source,
                            crate::span::Diagnostic::with_rule(
                                crate::checked::DiagnosticRule::Resolution,
                                constant.name.span,
                                format!("`{}` is declared twice", constant.name.name),
                            ),
                        ));
                    }
                    constants.insert(constant.name.name.clone(), constant.value);
                    declared.push((constant.name.name.clone(), constant.value));
                }
            }
            Err(diagnostic) => {
                parse_diagnostics.push(SourceDiagnostic::located(source, diagnostic))
            }
        }
    }
    if let Some(diagnostics) = Diagnostics::new(std::mem::take(&mut parse_diagnostics)) {
        return Err(CheckFailure::Source(SourceError::new(diagnostics)));
    }
    let mut parsed = Vec::new();
    for (file, source) in sources.files().iter().enumerate() {
        match crate::syntax::parse(&source.text, &constants) {
            Ok(ast) => {
                for decl in &ast.decls {
                    if constants.contains_key(&decl.name().name) {
                        parse_diagnostics.push(SourceDiagnostic::located(
                            source,
                            crate::span::Diagnostic::with_rule(
                                crate::checked::DiagnosticRule::Resolution,
                                decl.name().span,
                                format!("`{}` is a module constant's name", decl.name().name),
                            ),
                        ));
                    }
                }
                parsed.push((file, ast));
            }
            Err(diagnostic) => {
                parse_diagnostics.push(SourceDiagnostic::located(source, diagnostic))
            }
        }
    }
    for (file, diagnostic) in crate::projection::expand(&mut parsed) {
        parse_diagnostics.push(SourceDiagnostic::located(&sources.files()[file], diagnostic));
    }
    if let Some(diagnostics) = Diagnostics::new(parse_diagnostics) {
        return Err(CheckFailure::Source(SourceError::new(diagnostics)));
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
        return Err(CheckFailure::Source(SourceError::new(diagnostics)));
    }
    let (definitions, families, contents) =
        checked.expect("a program without diagnostics is checked");

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
        let mut content_hasher = sha2::Sha256::new();
        content_hasher.update(b"seismic-entry-content-v1");
        content_hasher.update(crate::bundle::COMPILER_SEMANTIC_VERSION.as_bytes());
        content_hasher.update(crate::registry::REGISTRY_REVISION.as_bytes());
        content_hasher.update(contents[family_ordinal]);
        let content = crate::ids::EntryContent::new(content_hasher.finalize().into());
        entries.push(entry_info(id, stable, content, contract));
        entry_families.push(family_ordinal);
    }
    let mut native_implementations = Vec::new();
    let mut native_sources = crate::checked::NativeSources::new();
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
            let ordinal = entries
                .iter()
                .position(|candidate| candidate.id == entry.id)
                .expect("a matched entry is one of the module's");
            let contract = &definitions[families[entry_families[ordinal]].contract.index()];
            match check_native(
                native,
                entry,
                contract,
                backend,
                &sources.files()[*file].path,
            ) {
                Ok(mut implementation) => {
                    let source = read(&entry.name, &implementation).map_err(CheckFailure::Read)?;
                    match crate::native_source::derive_codes(&mut implementation, source.as_deref()) {
                        Ok(()) => {
                            if let Some(source) = source {
                                native_sources.insert((entry.id, backend), source);
                            }
                            native_implementations.push(implementation);
                        }
                        Err(error) => native_diagnostics.push(SourceDiagnostic::new(
                            &sources.files()[*file],
                            native.span,
                            DiagnosticRule::NativeDeclaration,
                            format!("native source `{}`: {error}", implementation.source_path),
                        )),
                    }
                }
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
        return Err(CheckFailure::Source(SourceError::new(diagnostics)));
    }
    Ok((
        crate::checked::internals::Module {
            id: module_id,
            program,
            semantic_hash,
            sources: std::sync::Arc::new(sources),
            entries,
            native_implementations,
            entry_families,
            definitions,
            families,
            constants: declared,
        },
        native_sources,
    ))
}

/// The record ports `declared` lists (`records`), each a device tensor
/// parameter or tensor result with a leading record axis, listed once, on a
/// backend that binds record tables; what is wrong with any is in `errors`.
fn native_records(
    declared: &[crate::syntax::ast::NativeRecordDecl],
    contract: &ir::Definition,
    backend: crate::registry::BackendName,
    errors: &mut Vec<(Span, String)>,
) -> Vec<crate::checked::RecordPort> {
    let mut records = Vec::new();
    let results = result_leaves(&contract.result);
    for record in declared {
        let recorded = |ty: &ValueType| matches!(ty, ValueType::Tensor(tensor) if !tensor.axes.is_empty());
        let (span, port) = match record {
            ast::NativeRecordDecl::Parameter(name) => (
                name.span,
                match contract.params.iter().find(|parameter| parameter.name == name.name) {
                    Some(ir::Param {
                        ownership:
                            ir::Ownership::Owned | ir::Ownership::Shared | ir::Ownership::Exclusive,
                        ty,
                        ..
                    }) if recorded(ty) => Ok(crate::checked::RecordPort::Parameter(name.name.clone())),
                    Some(_) => Err(format!(
                        "record port `{}` must be a device tensor with a leading record axis",
                        name.name
                    )),
                    None => Err(format!("record port `{}` is not an entry parameter", name.name)),
                },
            ),
            ast::NativeRecordDecl::Result { ordinal, span } => (
                *span,
                match usize::try_from(*ordinal).ok().and_then(|ordinal| results.get(ordinal)) {
                    Some((_, ty)) if recorded(ty) => {
                        Ok(crate::checked::RecordPort::Result(*ordinal as u32))
                    }
                    Some(_) => Err(format!(
                        "record port `result {ordinal}` must be a tensor with a leading record axis"
                    )),
                    None => Err(format!("record port `result {ordinal}` is not an entry result")),
                },
            ),
        };
        match port {
            _ if !backend.binds_record_tables() => errors.push((
                span,
                format!("`records`: {} binds no record tables", backend.as_str()),
            )),
            Ok(port) if records.contains(&port) => errors.push((
                span,
                match record {
                    ast::NativeRecordDecl::Parameter(name) => {
                        format!("record port `{}` is listed twice", name.name)
                    }
                    ast::NativeRecordDecl::Result { ordinal, .. } => {
                        format!("record port `result {ordinal}` is listed twice")
                    }
                },
            )),
            Ok(port) => records.push(port),
            Err(message) => errors.push((span, message)),
        }
    }
    records
}

/// Check one native declaration against its entry. Every problem is reported.
fn check_native(
    native: &crate::syntax::ast::NativeDecl,
    entry: &crate::checked::EntryInfo,
    contract: &ir::Definition,
    backend: crate::registry::BackendName,
    declared_in: &str,
) -> Result<crate::checked::NativeImplementation, Vec<(Span, String)>> {
    use crate::checked::{NativeLaunch, NativeNatExpr, NativeParameter, NativeScratch};
    let mut errors = Vec::new();
    let destinations: Vec<String> = entry
        .element_domain
        .conversions()
        .iter()
        .filter_map(|conversion| match &conversion.target {
            crate::checked::ElementTarget::Parameter(parameter) => Some(parameter.clone()),
            crate::checked::ElementTarget::Concrete(_) => None,
        })
        .collect();
    let element_reads = NativeElements {
        parameters: &entry.element_parameters,
        destinations: &destinations,
        backend,
    };

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
            errors.push((
                name.span,
                format!("static dimension `{}` is listed twice", name.name),
            ));
        } else {
            statics.push(name.name.clone());
        }
    }

    // Each parameter with its values; its default is lowered once the terms
    // are known.
    let mut declared: Vec<(String, Vec<u64>)> = Vec::new();
    for param in &native.params {
        let name = &param.name.name;
        if entry.dimensions.contains(name) {
            errors.push((
                param.name.span,
                format!(
                    "native parameter `{name}` shadows a dimension of `{}`",
                    entry.name
                ),
            ));
            continue;
        }
        if declared.iter().any(|(existing, _)| existing == name) {
            errors.push((
                param.name.span,
                format!("native parameter `{name}` is declared twice"),
            ));
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
        declared.push((name.clone(), values));
    }
    let parameter_names = declared
        .iter()
        .map(|(name, _)| name.clone())
        .collect::<Vec<_>>();

    // Named terms are substitution: every later expression of the
    // declaration is checked with each term's name replaced by its
    // expression, so the checked implementation holds no trace of them.
    let mut terms: Vec<(String, crate::syntax::ast::Expr)> = Vec::new();
    for term in &native.terms {
        let name = &term.name.name;
        let taken = entry.dimensions.contains(name)
            || parameter_names.contains(name)
            || terms.iter().any(|(known, _)| known == name)
            || native
                .launches
                .iter()
                .any(|launch| launch.params.iter().any(|param| &param.name.name == name));
        if taken {
            errors.push((
                term.name.span,
                format!("term `{name}` is already a dimension, a parameter or a term"),
            ));
            continue;
        }
        let value = native_term_expansion(&term.value, &terms);
        match native_nat_expr(&value, &entry.dimensions, &parameter_names, element_reads) {
            Ok(_) => terms.push((name.clone(), value)),
            Err(message) => errors.push((term.value.span, message)),
        }
    }
    let expanded;
    let native = if terms.is_empty() {
        native
    } else {
        let expand = |expr: &mut crate::syntax::ast::Expr| {
            *expr = native_term_expansion(expr, &terms);
        };
        let mut declaration = native.clone();
        declaration.constraint.iter_mut().for_each(expand);
        for param in declaration
            .params
            .iter_mut()
            .chain(declaration.launches.iter_mut().flat_map(|launch| &mut launch.params))
        {
            match &mut param.default {
                ast::NativeDefaultDecl::Value(expr)
                | ast::NativeDefaultDecl::Least(expr)
                | ast::NativeDefaultDecl::Greatest(expr) => expand(expr),
            }
        }
        for scratch in &mut declaration.scratch {
            expand(&mut scratch.bytes);
            scratch.when.iter_mut().for_each(expand);
        }
        for launch in &mut declaration.launches {
            launch.when.iter_mut().for_each(expand);
            launch.constraint.iter_mut().for_each(expand);
            launch
                .threadgroups
                .iter_mut()
                .chain(&mut launch.threads_per_threadgroup)
                .chain(&mut launch.shared_bytes)
                .for_each(expand);
        }
        if let Some(repeat) = &mut declaration.repeat {
            expand(&mut repeat.count);
        }
        expanded = declaration;
        &expanded
    };

    // A launch's `where` restricts the configurations of the parameters it
    // reads, as one more conjunct of the declaration's: the checks below
    // hold it to what they hold every conjunct to.
    let folded;
    let native = if native.launches.iter().all(|launch| launch.constraint.is_none()) {
        native
    } else {
        let mut declaration = native.clone();
        for launch in &native.launches {
            if let Some(condition) = &launch.constraint {
                declaration.constraint = Some(match declaration.constraint.take() {
                    None => condition.clone(),
                    Some(constraint) => crate::syntax::ast::Expr {
                        span: constraint.span.to(condition.span),
                        kind: crate::syntax::ast::ExprKind::Binary {
                            op: crate::syntax::ast::BinaryOp::And,
                            lhs: Box::new(constraint),
                            rhs: Box::new(condition.clone()),
                        },
                    },
                });
            }
        }
        folded = declaration;
        &folded
    };

    // Every default reads only the device, the static dimensions and the
    // bound elements, and `least`/`greatest` the parameter itself; a default
    // that reads no dimension and no element is checked here on every
    // device of the backend.
    let lowered_default = |param: &ast::NativeParamDecl,
                           values: &[u64],
                           errors: &mut Vec<(Span, String)>|
     -> Option<crate::checked::NativeDefault> {
        let name = &param.name.name;
        let (expr, lowered) = match &param.default {
            ast::NativeDefaultDecl::Value(expr) => (
                expr,
                native_nat_expr(expr, &entry.dimensions, &[], element_reads)
                    .map(crate::checked::NativeDefault::Value)
                    .map_err(|message| (expr.span, message)),
            ),
            ast::NativeDefaultDecl::Least(expr) | ast::NativeDefaultDecl::Greatest(expr) => {
                let least = matches!(param.default, ast::NativeDefaultDecl::Least(_));
                (
                    expr,
                    native_condition(
                        "default",
                        expr,
                        &entry.dimensions,
                        std::slice::from_ref(name),
                        element_reads,
                    )
                    .map(|condition| match least {
                        true => crate::checked::NativeDefault::Least(condition),
                        false => crate::checked::NativeDefault::Greatest(condition),
                    }),
                )
            }
        };
        let default = lowered.map_err(|error| errors.push(error)).ok()?;
        let mut dimensions = Vec::new();
        let element = match &default {
            crate::checked::NativeDefault::Value(value) => {
                value.dimensions(&mut dimensions);
                value.element_read().is_some()
            }
            crate::checked::NativeDefault::Least(condition)
            | crate::checked::NativeDefault::Greatest(condition) => {
                condition.dimensions(&mut dimensions);
                condition.element_read().is_some()
            }
        };
        if let Some(dynamic) = dimensions.iter().find(|dimension| !statics.contains(dimension)) {
            errors.push((expr.span, format!("the default of `{name}` reads dimension `{dynamic}`, which is not static; a default reads only the device, the static dimensions and the bound elements")));
            return None;
        }
        if dimensions.is_empty() && !element {
            if let Err(error) = default.on_every_device(name, values, backend) {
                errors.push((expr.span, error.to_string()));
                return None;
            }
        }
        Some(default)
    };
    let params = declared
        .into_iter()
        .filter_map(|(name, values)| {
            let param = native
                .params
                .iter()
                .find(|param| param.name.name == name)
                .expect("every checked parameter is declared");
            let default = lowered_default(param, &values, &mut errors)?;
            Some(NativeParameter {
                name,
                // Derived from the source in `check_closed`, before the
                // implementation is kept.
                code: false,
                arithmetic: param.arithmetic,
                form: param.form,
                values,
                default,
                role: crate::checked::NativeParameterRole::Declared,
            })
        })
        .collect::<Vec<_>>();

    let records = native_records(&native.records, contract, backend, &mut errors);

    let mut elements: Vec<crate::checked::NativeElementCoverage> = Vec::new();
    for coverage in &native.elements {
        let name = &coverage.name.name;
        if backend != crate::registry::BackendName::Cpu {
            errors.push((
                coverage.span,
                "`elements` declares the element types of a CPU form compiled with the program; other backends compile each binding when it is prepared".to_owned(),
            ));
            continue;
        }
        let stored = entry
            .element_domain
            .parameters()
            .iter()
            .find(|parameter| &parameter.name == name);
        let converted = entry.element_domain.conversions().iter().any(|conversion| {
            conversion.target == crate::checked::ElementTarget::Parameter(name.clone())
        });
        match stored {
            None => {
                errors.push((
                    coverage.name.span,
                    format!("`{name}` is not an element parameter of `{}`", entry.name),
                ));
                continue;
            }
            Some(parameter)
                if !(parameter.uses.stored || parameter.uses.partial_copy) || converted =>
            {
                errors.push((
                    coverage.name.span,
                    format!(
                        "`{name}` is not stored by `{}`; a CPU form reads it as a weight or converts into it, for every representation",
                        entry.name
                    ),
                ));
                continue;
            }
            Some(_) => {}
        }
        if elements.iter().any(|existing| &existing.parameter == name) {
            errors.push((
                coverage.name.span,
                format!("the elements of `{name}` are declared twice"),
            ));
            continue;
        }
        let mut dtypes = Vec::new();
        for dtype in &coverage.dtypes {
            match DType::from_name(&dtype.name) {
                Some(value) if value != DType::Bool => {
                    if dtypes.contains(&value) {
                        errors.push((dtype.span, format!("`{}` is listed twice", dtype.name)));
                    } else {
                        dtypes.push(value);
                    }
                }
                _ => errors.push((
                    dtype.span,
                    format!(
                        "`{}` is not a CPU element type; they are f32, bf16, f16, i32 and u32",
                        dtype.name
                    ),
                )),
            }
        }
        elements.push(crate::checked::NativeElementCoverage {
            parameter: name.clone(),
            dtypes,
        });
    }

    // A CPU form binds every element parameter by how the entry uses it.
    if backend == crate::registry::BackendName::Cpu {
        for parameter in entry.element_domain.parameters() {
            if entry.element_domain.cpu_binding(&parameter.name).is_none() {
                errors.push((
                    native.function.span,
                    format!(
                        "`{}` has no CPU binding in `{}`: the entry neither reads, stores nor converts it",
                        parameter.name, entry.name
                    ),
                ));
            }
        }
    }

    let expression = |expr: &crate::syntax::ast::Expr, errors: &mut Vec<(Span, String)>| {
        native_nat_expr(expr, &entry.dimensions, &parameter_names, element_reads)
            .map_err(|message| errors.push((expr.span, message)))
            .ok()
    };

    // `None` inside the result means the clause is absent; an outer `None`
    // means it failed to check (its errors are recorded).
    let condition = |clause: &str,
                     expr: &Option<crate::syntax::ast::Expr>,
                     errors: &mut Vec<(Span, String)>| match expr {
        None => Some(None),
        Some(expr) => {
            native_condition(clause, expr, &entry.dimensions, &parameter_names, element_reads)
                .map_err(|error| errors.push(error))
                .ok()
                .map(Some)
        }
    };

    let launch_names = native
        .launches
        .iter()
        .flat_map(|launch| launch.params.iter().map(|param| param.name.name.clone()))
        .collect::<Vec<_>>();
    let where_names = parameter_names
        .iter()
        .cloned()
        .chain(launch_names.iter().cloned())
        .collect::<Vec<_>>();
    // `where` may read the elements: admissible configurations are those of
    // the implementation at its bindings (`NativeImplementation::at_elements`).
    let constraint = native.constraint.as_ref().and_then(|expr| {
        native_condition("where", expr, &entry.dimensions, &where_names, element_reads)
            .map_err(|error| errors.push(error))
            .ok()
    });
    if let (Some(constraint), Some(expr)) = (&constraint, &native.constraint) {
        let mut read = Vec::new();
        constraint.dimensions(&mut read);
        if let Some(dynamic) = read.iter().find(|name| !statics.contains(name)) {
            errors.push((
                expr.span,
                format!(
                    "native `where` reads dimension `{dynamic}`, which is not static; its value is unknown at preparation"
                ),
            ));
        }
        fn conjuncts<'a>(
            condition: &'a crate::checked::NativeCondition,
            out: &mut Vec<&'a crate::checked::NativeCondition>,
        ) {
            match condition {
                crate::checked::NativeCondition::And(left, right) => {
                    conjuncts(left, out);
                    conjuncts(right, out);
                }
                other => out.push(other),
            }
        }
        let mut parts = Vec::new();
        conjuncts(constraint, &mut parts);
        for part in parts {
            let mut names = Vec::new();
            part.parameters(&mut names);
            let mut owner = None;
            for name in names {
                let owners = native
                    .launches
                    .iter()
                    .enumerate()
                    .filter_map(|(index, launch)| {
                        launch
                            .params
                            .iter()
                            .any(|parameter| parameter.name.name == name)
                            .then_some(index)
                    })
                    .collect::<Vec<_>>();
                if owners.len() > 1 {
                    errors.push((
                        expr.span,
                        format!("native `where` parameter `{name}` is ambiguous across launches"),
                    ));
                } else if let Some(launch) = owners.first().copied() {
                    if owner.is_some_and(|previous| previous != launch) {
                        errors.push((
                            expr.span,
                            "one native `where` conjunct may read parameters of only one launch"
                                .to_owned(),
                        ));
                    }
                    owner = Some(launch);
                }
            }
        }
    }

    let mut scratch: Vec<NativeScratch> = Vec::new();
    for buffer in &native.scratch {
        if scratch
            .iter()
            .any(|existing| existing.name == buffer.name.name)
        {
            errors.push((
                buffer.name.span,
                format!("scratch buffer `{}` is declared twice", buffer.name.name),
            ));
            continue;
        }
        let bytes = expression(&buffer.bytes, &mut errors);
        let when = condition("when", &buffer.when, &mut errors);
        if let (Some(bytes), Some(when)) = (bytes, when) {
            scratch.push(NativeScratch {
                name: buffer.name.name.clone(),
                bytes,
                sync: buffer.sync,
                when,
            });
        }
    }

    let mut launches = Vec::new();
    let mut launch_records: Vec<(usize, Vec<crate::checked::RecordPort>)> = Vec::new();
    for launch in &native.launches {
        let mut launch_params: Vec<NativeParameter> = Vec::new();
        for param in &launch.params {
            let name = &param.name.name;
            if entry.dimensions.contains(name) || parameter_names.contains(name) {
                errors.push((
                    param.name.span,
                    format!("launch parameter `{name}` shadows an entry dimension or parameter"),
                ));
                continue;
            }
            if launch_params.iter().any(|existing| &existing.name == name) {
                errors.push((
                    param.name.span,
                    format!("launch parameter `{name}` is declared twice"),
                ));
                continue;
            }
            let mut values = Vec::new();
            for value in &param.values {
                if values.contains(value) {
                    errors.push((
                        param.span,
                        format!("launch parameter `{name}` lists {value} twice"),
                    ));
                } else {
                    values.push(*value);
                }
            }
            let Some(default) = lowered_default(param, &values, &mut errors) else {
                continue;
            };
            launch_params.push(NativeParameter {
                name: name.clone(),
                // Derived from the source in `check_closed`, before the
                // implementation is kept.
                code: false,
                arithmetic: param.arithmetic,
                form: param.form,
                values,
                default,
                role: crate::checked::NativeParameterRole::Declared,
            });
        }
        let mut reads = Vec::new();
        for read in &launch.reads {
            if !parameter_names.contains(&read.name) {
                errors.push((
                    read.span,
                    format!("kernel read `{}` is not an entry parameter", read.name),
                ));
            } else if reads.contains(&read.name) {
                errors.push((
                    read.span,
                    format!("kernel read `{}` is declared twice", read.name),
                ));
            } else {
                reads.push(read.name.clone());
            }
        }
        // A grid is three `i32` group counts an earlier launch wrote: a
        // shared rank-1 tensor of the entry, or a 12-byte scratch buffer
        // of the implementation, read when the launch runs. Every backend
        // dispatches one: Metal and Vulkan indirectly, the CPU reading the
        // words when the launch runs, CUDA as a grid-stride loop over them
        // (the runtime's `abi::written_grid_loop`).
        let grid = launch.grid.as_ref().and_then(|grid| {
            let declared = contract.params.iter().find(|parameter| parameter.name == grid.name);
            let held = scratch.iter().find(|buffer| buffer.name == grid.name);
            let refusal = match (declared, held) {
                (Some(_), Some(_)) => Some(format!(
                    "launch grid `{}` names both an entry parameter and a scratch buffer",
                    grid.name
                )),
                (None, None) => Some(format!(
                    "launch grid `{}` is not an entry parameter or scratch buffer",
                    grid.name
                )),
                (
                    Some(ir::Param {
                        ownership: ir::Ownership::Shared,
                        ty: ValueType::Tensor(tensor),
                        ..
                    }),
                    None,
                ) if tensor.elem == crate::types::Elem::Dtype(DType::I32)
                    && matches!(tensor.axes.as_slice(), [extent] if prove::constant(&contract.arena, *extent) == Some(3)) =>
                {
                    None
                }
                (Some(_), None) => Some(format!(
                    "launch grid `{}` must be a `&tensor[3] i32` the launch reads",
                    grid.name
                )),
                (None, Some(NativeScratch { bytes: NativeNatExpr::Constant(12), sync: false, when: None, .. })) => None,
                (None, Some(_)) => Some(format!(
                    "launch grid `{}` must be a scratch buffer of `bytes (12)`, neither `sync` nor `when`",
                    grid.name
                )),
            };
            match (refusal, held) {
                (Some(message), _) => {
                    errors.push((grid.span, message));
                    None
                }
                (None, None) => Some(crate::checked::LaunchGrid::Parameter(grid.name.clone())),
                (None, Some(_)) => Some(crate::checked::LaunchGrid::Scratch(grid.name.clone())),
            }
        });
        // A launch condition's word is one `i32` the host writes between
        // submissions: word `index` (0 unless written) of a shared rank-1
        // `i32` tensor of the entry, of a constant extent past it, read when
        // the run is encoded (`BackendName::encodes_runs`). A written grid's
        // word is the device's, so it gates no launch
        // (`checked::NativeLaunch::grid`). The launch's conditions are a
        // conjunction, each of its own word: a word named twice is
        // redundant or, under both values, leaves the launch never encoded.
        let mut named: Vec<(&str, u64)> = Vec::new();
        let conditions = launch
            .conditions
            .iter()
            .filter_map(|condition| {
                let word = &condition.word;
                // The index a constant (module constants read as their
                // integers), else unrepresentable: refused here.
                let index = match &condition.index {
                    None => Some(0),
                    Some(index) => match index.kind {
                        crate::syntax::ast::ExprKind::Int(index) => Some(index),
                        _ => None,
                    },
                };
                let Some(index) = index else {
                    errors.push((word.span, format!("launch condition `{}`'s index must be a constant", word.name)));
                    return None;
                };
                let written = match condition.index {
                    Some(_) => format!("{}[{index}]", word.name),
                    None => word.name.clone(),
                };
                let repeated = named.contains(&(word.name.as_str(), index));
                named.push((&word.name, index));
                let declared = contract.params.iter().find(|parameter| parameter.name == word.name);
                let written_grid = native
                    .launches
                    .iter()
                    .any(|other| other.grid.as_ref().is_some_and(|grid| grid.name == word.name));
                let refusal = match declared {
                    _ if written_grid => Some(format!(
                        "launch condition `{written}`: a written grid's word gates no launch"
                    )),
                    _ if !backend.encodes_runs() => Some(format!(
                        "launch condition `{written}`: {} does not encode each run's launches",
                        backend.as_str()
                    )),
                    _ if repeated => Some(format!("launch condition `{written}` is named twice")),
                    None => Some(format!("launch condition `{}` is not an entry parameter", word.name)),
                    Some(ir::Param {
                        ownership: ir::Ownership::Shared,
                        ty: ValueType::Tensor(tensor),
                        ..
                    }) if tensor.elem == crate::types::Elem::Dtype(DType::I32) => match tensor.axes.as_slice() {
                        [extent] => match prove::constant(&contract.arena, *extent) {
                            Some(extent) if i64::try_from(index).is_ok_and(|index| index < extent) => None,
                            Some(extent) => Some(format!(
                                "launch condition `{written}` indexes past its `&tensor[{extent}] i32`"
                            )),
                            None => Some(format!(
                                "launch condition `{written}` must be a word of a `&tensor[N] i32` of a constant N"
                            )),
                        },
                        _ => Some(format!(
                            "launch condition `{written}` must be a word of a `&tensor[N] i32` the host writes"
                        )),
                    },
                    Some(_) => Some(format!(
                        "launch condition `{written}` must be a word of a `&tensor[N] i32` the host writes"
                    )),
                };
                match refusal {
                    Some(message) => {
                        errors.push((word.span, message));
                        None
                    }
                    None => Some(crate::checked::LaunchCondition {
                        word: word.name.clone(),
                        index,
                        encoded: condition.encoded,
                    }),
                }
            })
            .collect::<Vec<_>>();
        let launch_names = parameter_names
            .iter()
            .cloned()
            .chain(launch_params.iter().map(|param| param.name.clone()))
            .collect::<Vec<_>>();
        let launch_expression = |expr: &crate::syntax::ast::Expr,
                                 errors: &mut Vec<(Span, String)>| {
            native_nat_expr(expr, &entry.dimensions, &launch_names, element_reads)
                .map_err(|message| errors.push((expr.span, message)))
                .ok()
        };
        let when = match &launch.when {
            None => Some(None),
            Some(expr) => {
                native_condition("when", expr, &entry.dimensions, &launch_names, element_reads)
                    .map_err(|error| errors.push(error))
                    .ok()
                    .map(Some)
            }
        };
        let groups = launch
            .threadgroups
            .each_ref()
            .map(|expr| launch_expression(expr, &mut errors));
        let group_extent = launch
            .threads_per_threadgroup
            .each_ref()
            .map(|expr| launch_expression(expr, &mut errors));
        let shared_bytes = match &launch.shared_bytes {
            Some(expr) => launch_expression(expr, &mut errors),
            None => Some(NativeNatExpr::Constant(0)),
        };
        let (
            Some(when),
            [Some(x), Some(y), Some(z)],
            [Some(ex), Some(ey), Some(ez)],
            Some(shared_bytes),
        ) = (when, groups, group_extent, shared_bytes)
        else {
            continue;
        };
        // A Vulkan pipeline is created with its workgroup size and shared
        // memory, so both must be known when the implementation is prepared.
        if backend == crate::registry::BackendName::Vulkan {
            let fixed = [
                (&ex, &launch.threads_per_threadgroup[0]),
                (&ey, &launch.threads_per_threadgroup[1]),
                (&ez, &launch.threads_per_threadgroup[2]),
            ]
            .into_iter()
            .chain(
                launch
                    .shared_bytes
                    .as_ref()
                    .map(|expr| (&shared_bytes, expr)),
            );
            for (checked, expr) in fixed {
                let mut read = Vec::new();
                checked.dimensions(&mut read);
                if let Some(dynamic) = read.iter().find(|name| !statics.contains(name)) {
                    errors.push((
                        expr.span,
                        format!(
                            "Vulkan fixes the group size and shared memory when the kernel is prepared; `{dynamic}` is only known per call (declare it `static` or use a static bound)"
                        ),
                    ));
                }
            }
        }
        if let Some(own) = &launch.records {
            launch_records.push((launches.len(), native_records(own, contract, backend, &mut errors)));
        }
        // A resident launch's grid is the backend's: as many threadgroups as
        // it holds co-resident for the formed pipeline, at most the first
        // axis of `threadgroups`, so its other axes are 1 and no written
        // grid sizes it. Its barrier is a `sync` buffer, zero whenever a
        // launch starts, of at least the barrier's words.
        let resident = launch.resident.as_ref().and_then(|barrier| {
            let refusal = if !backend.resident_launches() {
                Some(format!(
                    "`resident` needs a backend that runs a launch's threadgroups co-resident; {} does not",
                    native.target.name
                ))
            } else if launch.grid.is_some() {
                Some("a resident launch's grid is the backend's `RESIDENT`; it reads no written grid".to_owned())
            } else if !matches!((&y, &z), (NativeNatExpr::Constant(1), NativeNatExpr::Constant(1))) {
                Some("a resident launch's grid has one axis: `threadgroups (WORK, 1, 1)`".to_owned())
            } else {
                match scratch.iter().find(|buffer| buffer.name == barrier.name) {
                    Some(NativeScratch { bytes: NativeNatExpr::Constant(bytes), sync: true, when: None, .. })
                        if *bytes >= crate::registry::RESIDENT_BARRIER_BYTES =>
                    {
                        None
                    }
                    _ => Some(format!(
                        "resident barrier `{}` must be a `sync` scratch buffer of at least `bytes ({})`, without `when`",
                        barrier.name,
                        crate::registry::RESIDENT_BARRIER_BYTES
                    )),
                }
            };
            match refusal {
                Some(message) => {
                    errors.push((barrier.span, message));
                    None
                }
                None => Some(crate::checked::ResidentLaunch { barrier: barrier.name.clone() }),
            }
        });
        launches.push(NativeLaunch {
            kernel: launch.kernel.name.clone(),
            params: launch_params,
            reads,
            when,
            groups: [x, y, z],
            grid,
            conditions,
            group_extent: [ex, ey, ez],
            shared_bytes,
            resident,
        });
    }

    // Launches of one kernel share its formed source, which reads its record
    // ports as tables and every other port as its view.
    let records_of = |ordinal: usize| {
        launch_records.iter().find(|(own, _)| *own == ordinal).map_or(&records, |(_, own)| own)
    };
    for (ordinal, launch) in launches.iter().enumerate() {
        let first = launches.iter().position(|other: &NativeLaunch| other.kernel == launch.kernel);
        if first.is_some_and(|first| records_of(first) != records_of(ordinal)) {
            errors.push((
                native.launches[ordinal].span,
                format!("launches of kernel `{}` list different record ports", launch.kernel),
            ));
        }
        // A kernel's source forms once for its launches, resident or not
        // (`SEISMIC_RESIDENT`, under which its grid barriers form).
        if first.is_some_and(|first| launches[first].resident.is_some() != launch.resident.is_some()) {
            errors.push((
                native.launches[ordinal].span,
                format!("launches of kernel `{}` are not all resident or all dispatched", launch.kernel),
            ));
        }
    }

    let repeat = match &native.repeat {
        None => None,
        Some(repeat) if !backend.repeats_launches() => {
            errors.push((
                repeat.span,
                format!("`repeat` is not implemented for {}", native.target.name),
            ));
            None
        }
        Some(repeat) => {
            expression(&repeat.count, &mut errors).map(|count| crate::checked::NativeRepeat {
                first: repeat.first,
                launches: repeat.launches,
                count,
            })
        }
    };

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
        elements,
        records,
        launch_records,
        constraint,

        scratch,
        launches,
        repeat,
    })
}

/// `expression` with every name of `terms` replaced by that term's
/// expression (the parts a native expression or condition is made of; any
/// other form is left for the lowering to reject).
fn native_term_expansion(
    expression: &crate::syntax::ast::Expr,
    terms: &[(String, crate::syntax::ast::Expr)],
) -> crate::syntax::ast::Expr {
    use crate::syntax::ast::{Arg, Expr, ExprKind};
    let expand = |expression: &Expr| native_term_expansion(expression, terms);
    let kind = match &expression.kind {
        ExprKind::Name(name) => {
            return terms
                .iter()
                .find(|(term, _)| term == &name.name)
                .map_or_else(|| expression.clone(), |(_, value)| value.clone());
        }
        ExprKind::Binary { op, lhs, rhs } => ExprKind::Binary {
            op: *op,
            lhs: Box::new(expand(lhs)),
            rhs: Box::new(expand(rhs)),
        },
        ExprKind::Call {
            callee,
            bindings,
            args,
        } => ExprKind::Call {
            callee: callee.clone(),
            bindings: bindings.clone(),
            args: args
                .iter()
                .map(|arg| Arg {
                    name: arg.name.clone(),
                    value: expand(&arg.value),
                })
                .collect(),
        },
        _ => return expression.clone(),
    };
    Expr {
        kind,
        ..expression.clone()
    }
}

/// Lower a native `where` or `when` condition: comparisons of native
/// natural-number expressions joined by `and` and `or` (parentheses group).
fn native_condition(
    clause: &str,
    expression: &crate::syntax::ast::Expr,
    dimensions: &[String],
    parameters: &[String],
    elements: NativeElements<'_>,
) -> Result<crate::checked::NativeCondition, (Span, String)> {
    use crate::checked::{NativeComparison, NativeCondition};
    use crate::syntax::ast::{BinaryOp, ExprKind};
    let not_a_condition = || {
        (
            expression.span,
            format!(
                "a native `{clause}` condition is a comparison (`<`, `<=`, `>`, `>=`, `==`, `!=`) of natural-number expressions, or conditions joined by `and` and `or`"
            ),
        )
    };
    // A capability the backend provides stands as a condition: it holds.
    if let ExprKind::Attr { base, name } = &expression.kind {
        if matches!(&base.kind, ExprKind::Name(device) if device.name == "device")
            && crate::device::provided(elements.backend, &name.name)
                == Some(crate::device::PropertyKind::Support)
        {
            return Ok(NativeCondition::Compare {
                comparison: NativeComparison::Ne,
                left: crate::checked::NativeNatExpr::Device(name.name.clone()),
                right: crate::checked::NativeNatExpr::Constant(0),
            });
        }
    }
    let ExprKind::Binary { op, lhs, rhs } = &expression.kind else {
        return Err(not_a_condition());
    };
    let comparison = match op {
        BinaryOp::And | BinaryOp::Or => {
            let left = Box::new(native_condition(clause, lhs, dimensions, parameters, elements)?);
            let right = Box::new(native_condition(clause, rhs, dimensions, parameters, elements)?);
            return Ok(match op {
                BinaryOp::And => NativeCondition::And(left, right),
                _ => NativeCondition::Or(left, right),
            });
        }
        BinaryOp::Lt => NativeComparison::Lt,
        BinaryOp::Le => NativeComparison::Le,
        BinaryOp::Gt => NativeComparison::Gt,
        BinaryOp::Ge => NativeComparison::Ge,
        BinaryOp::Eq => NativeComparison::Eq,
        BinaryOp::Ne => NativeComparison::Ne,
        _ => return Err(not_a_condition()),
    };
    let operand = |expr: &crate::syntax::ast::Expr| {
        native_nat_expr(expr, dimensions, parameters, elements)
            .map_err(|message| (expr.span, message))
    };
    Ok(NativeCondition::Compare {
        comparison,
        left: operand(lhs)?,
        right: operand(rhs)?,
    })
}

/// The element parameters a native expression may read, those the entry
/// converts into (`repack[U = W]`: bound only to a registered conversion's
/// packed destination), and the backend whose forms `form(W, NAME)` names.
#[derive(Clone, Copy)]
struct NativeElements<'a> {
    parameters: &'a [String],
    destinations: &'a [String],
    backend: crate::registry::BackendName,
}

impl NativeElements<'_> {
    /// Element parameter `argument` of a read, as `call` names it.
    fn parameter(self, call: &str, argument: &crate::syntax::ast::Arg) -> Result<String, String> {
        match &argument.value.kind {
            crate::syntax::ast::ExprKind::Name(element)
                if argument.name.is_none() && self.parameters.contains(&element.name) =>
            {
                Ok(element.name.clone())
            }
            _ => Err(format!(
                "`{call}` reads one of the entry's element parameters ({})",
                self.parameters.join(", ")
            )),
        }
    }
}

fn native_nat_expr(
    expression: &crate::syntax::ast::Expr,
    dimensions: &[String],
    parameters: &[String],
    elements: NativeElements<'_>,
) -> Result<crate::checked::NativeNatExpr, String> {
    use crate::checked::{ElementQuery, NativeNatExpr as N};
    use crate::syntax::ast::{BinaryOp, ExprKind};
    let binary = |left: &crate::syntax::ast::Expr,
                  right: &crate::syntax::ast::Expr,
                  make: fn(Box<N>, Box<N>) -> N| {
        Ok(make(
            Box::new(native_nat_expr(left, dimensions, parameters, elements)?),
            Box::new(native_nat_expr(right, dimensions, parameters, elements)?),
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
        ExprKind::Name(name) => crate::forms::constants(elements.backend)
            .iter()
            .find(|(constant, _)| *constant == name.name)
            .map(|(_, value)| N::Constant(*value))
            .ok_or_else(|| {
                format!(
                    "native expression references `{}`, which is neither a dimension, a native parameter nor a constant of `{}`",
                    name.name,
                    elements.backend.as_str()
                )
            }),
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
        } if matches!(&callee.kind, ExprKind::Name(name) if name.name == "operands") => {
            match (bindings.is_empty(), args.as_slice()) {
                (true, [arg]) => Ok(N::Element(
                    elements.parameter("operands", arg)?,
                    ElementQuery::Operands,
                )),
                _ => Err("`operands` takes one element parameter".to_owned()),
            }
        }
        ExprKind::Call {
            callee,
            bindings,
            args,
        } if matches!(&callee.kind, ExprKind::Name(name) if name.name == "packed") => {
            match (bindings.is_empty(), args.as_slice()) {
                (true, [arg]) => Ok(N::Element(elements.parameter("packed", arg)?, ElementQuery::Packed)),
                _ => Err("`packed` takes one element parameter".to_owned()),
            }
        }
        ExprKind::Call {
            callee,
            bindings,
            args,
        } if matches!(&callee.kind, ExprKind::Name(name) if name.name == "unit_columns") => {
            match (bindings.is_empty(), args.as_slice()) {
                (true, [arg]) => {
                    let element = elements.parameter("unit_columns", arg)?;
                    if !elements.destinations.contains(&element) {
                        return Err(format!(
                            "`unit_columns` reads a conversion's destination; `{element}` is not one (`repack[U = {element}]`)"
                        ));
                    }
                    Ok(N::Element(element, ElementQuery::UnitColumns))
                }
                _ => Err("`unit_columns` takes one element parameter".to_owned()),
            }
        }
        ExprKind::Call {
            callee,
            bindings,
            args,
        } if matches!(&callee.kind, ExprKind::Name(name) if name.name == "form") => {
            let names = crate::forms::names(elements.backend);
            let form = |arg: &crate::syntax::ast::Arg| match &arg.value.kind {
                ExprKind::Name(name) if arg.name.is_none() => {
                    crate::forms::Form::named(elements.backend, &name.name).ok_or_else(|| {
                        format!(
                            "`{}` is not a form of `{}`; its forms are {}",
                            name.name,
                            elements.backend.as_str(),
                            if names.is_empty() { "none".to_owned() } else { names.join(", ") }
                        )
                    })
                }
                _ => Err("`form` names a form of the backend".to_owned()),
            };
            match (bindings.is_empty(), args.as_slice()) {
                (true, [element, name]) => Ok(N::Element(
                    elements.parameter("form", element)?,
                    ElementQuery::Form(form(name)?),
                )),
                _ => Err("`form` takes an element parameter and a form name".to_owned()),
            }
        }
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
                        "native expressions call only `ceil_div`, `min`, `max`, `operands`, `form`, `packed` and `unit_columns`"
                            .to_owned()
                    );
                }
            };
            binary(&args[0].value, &args[1].value, make)
        }
        ExprKind::Attr { base, name }
            if matches!(&base.kind, ExprKind::Name(device) if device.name == "device") =>
        {
            match crate::device::provided(elements.backend, &name.name) {
                Some(crate::device::PropertyKind::Support | crate::device::PropertyKind::Count) => {
                    Ok(N::Device(name.name.clone()))
                }
                // A measured property has no value until its probe has run,
                // and a declaration states no alternative to read instead.
                Some(crate::device::PropertyKind::Measured) => Err(format!(
                    "`device.{}` is measured, and a native declaration cannot read it: it has no value until its probe has run",
                    name.name
                )),
                None => Err(format!(
                    "`device.{}` is not a property `{}` provides",
                    name.name,
                    elements.backend.as_str()
                )),
            }
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
                    path.push(
                        u32::try_from(ordinal).expect("tuple has more than u32::MAX elements"),
                    );
                    walk(item, path, leaves);
                    path.pop();
                }
            }
            ValueType::Void => {}
            ValueType::Integer => {
                unreachable!("mathematical integer has no source result spelling")
            }
            ValueType::Opaque { .. } => {
                panic!("backend-opaque result escaped an exported portable entry")
            }
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
    content: crate::ids::EntryContent,
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
                    ParamOwnership::Host { write } => TensorAccess::Host { write: *write },
                    ParamOwnership::Tuple(_) => {
                        unreachable!("tensor leaf ownership is not a tuple")
                    }
                },
                rank: u32::try_from(tensor.rank()).expect("tensor rank exceeds u32::MAX"),
                element: element_summary(&tensor.elem),
            },
            ValueType::Scalar(dtype) => ParameterSummaryKind::Scalar(*dtype),
            ValueType::Integer => {
                unreachable!("mathematical integer has no source parameter spelling")
            }
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
                    ParamOwnership::Host { write } => TensorAccess::Host { write: *write },
                    _ => TensorAccess::Owned,
                },
                rank: t.rank() as u32,
                element: element_summary(&t.elem),
            },
            ValueType::Scalar(d) => S::Scalar(*d),
            ValueType::Integer => {
                unreachable!("mathematical integer has no source signature spelling")
            }
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
        content,
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

