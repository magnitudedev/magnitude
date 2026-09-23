//! The checker: resolution, contract families and target capabilities, typing
//! of every value kind through the intrinsic registry, ownership/moves,
//! borrow rules, loop carry and disjointness, initialization coverage, and
//! `where` predicates. Emits the checked representation directly.
//! Entry point is [`crate::checked::check_source`].
//!
//! Every symbolic integer is an `IntExpr` in the definition's private arena.

mod call;
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
    Block as CheckedBlock, Body as CheckedBody, DefKind, Expr as CheckedExpr,
    ExprKind as CheckedExprKind, Local as CheckedLocal, LocalId, Ownership as ParamOwnership,
    Predicate,
};
use crate::checked::EntryInfo;
use crate::expr::{ExprArena, IntExpr, SymbolId};
use crate::ids::{CapabilityId, ModuleHash, ModuleId, ProgramId, StableFunctionId};
use crate::intrinsics::PrimitiveId;
use crate::span::{Diagnostic, Span};
use crate::syntax::ast;
use crate::types::{Elem, TensorType, ValueType};
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

/// Per-definition facts callers need: how each shape parameter is used.
#[derive(Clone, Debug, Default)]
pub(crate) struct Summary {
    /// Shape parameters used as numbers (arithmetic, `extent`, range bounds).
    pub numeric: BTreeSet<String>,
    /// Shape parameters whose axis the body reduces over.
    pub reduces: BTreeSet<String>,
    /// `(callee definition, callee parameter, own parameter)`: passed along unchanged.
    pub passes: Vec<(usize, String, String)>,
}

pub(crate) struct Env<'a> {
    pub resolved: &'a Resolved<'a>,
    pub summaries: &'a [Summary],
    /// Second pass: summaries are complete.
    pub enforce: bool,
    checked: &'a [Option<CheckedOutcome>],
}

pub(crate) struct Checker<'a> {
    pub env: &'a Env<'a>,
    pub def: usize,
    pub sig: BodySig,
    pub kind: DefKind,
    /// The target whose forms this body may name: a backend-specific function
    /// or lowering target.
    pub target: Option<crate::registry::BackendName>,
    pub requires: Vec<(CapabilityId, Span)>,
    pub used_capabilities: BTreeSet<CapabilityId>,
    pub locals: Vec<CheckedLocal>,
    pub kinds: Vec<LocalKind>,
    pub scopes: Vec<HashMap<String, LocalId>>,
    pub facts: Facts,
    pub symbols: HashMap<LocalId, SymbolId>,
    pub scalar_symbols: HashMap<LocalId, IntExpr>,
    /// Runtime-bounded range views: (start, end, parent extent, realized-length atom).
    pub dyn_views: Vec<(Option<CheckedExpr>, Option<CheckedExpr>, IntExpr, SymbolId)>,
    pub mutated: Vec<LocalId>,
    /// Storage roots read so far.
    pub reads: Vec<LocalId>,
    /// Active independent (`parallel for`) loops: (captured floor, binder).
    pub logical_parallel: Vec<(usize, LocalId)>,
    /// Active loop contexts (innermost last).
    /// Depth of enclosing loops; `return` is invalid inside.
    pub loop_depth: usize,
    pub summary: Summary,
    pub diagnostics: Vec<Diagnostic>,
    pub counter: usize,
    /// Names whose binding was rejected; uses of them are not reported again.
    pub poisoned: HashSet<String>,
    pub arena: ExprArena,
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
            kind: declared.kind.clone(),
            target,
            requires: declared.requires.clone(),
            used_capabilities: BTreeSet::new(),
            locals: Vec::new(),
            kinds: Vec::new(),
            scopes: vec![HashMap::new()],
            facts: Facts::new(),
            symbols: HashMap::new(),
            scalar_symbols: HashMap::new(),
            dyn_views: Vec::new(),
            mutated: Vec::new(),
            reads: Vec::new(),
            logical_parallel: Vec::new(),
            loop_depth: 0,
            summary: Summary::default(),
            diagnostics: Vec::new(),
            counter: 0,
            poisoned: HashSet::new(),
            arena,
        };
        // Shape parameters are positive extents unless a `where` admits zero.
        for (ordinal, _p) in c.sig.shape_params.iter().enumerate() {
            let symbol = c.sig.shape_symbols[ordinal];
            let value = c.arena.int_symbol(symbol);
            let admits_zero = c.sig.predicates.iter().any(
                |q| matches!(q, Predicate::NonNegative(e) if prove::same(&c.arena, *e, value)),
            );
            let lower = c.arena.int(if admits_zero { 0 } else { 1 });
            c.facts.set_range_lower(symbol, lower);
        }
        for predicate in c.sig.predicates.clone() {
            match predicate {
                Predicate::NonNegative(e) => c.assume_nonneg(e),
                Predicate::Zero(e) => c.assume_zero(e),
                Predicate::NonZero(_) => {}
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
            if let ValueType::Index { bound } = &p.ty {
                let (_, symbol, _) = c.arena.loop_binder();
                let zero = c.arena.int(0);
                let one = c.arena.int(1);
                let upper = c.arena.int_sub(*bound, one);
                c.facts.set_range(symbol, zero, upper);
                c.symbols.insert(id, symbol);
                c.locals[id.index()].symbol = Some(symbol);
            }
            if let ValueType::Tensor(t) = &p.ty {
                if p.ownership == ParamOwnership::Owned {
                    // An owned tensor parameter starts initialized.
                    let _ = t;
                }
            }
        }
        c
    }

    pub fn error(&mut self, span: Span, message: impl Into<String>) {
        self.diagnostics.push(Diagnostic::new(span, message));
    }

    pub fn use_capability(&mut self, capability: &CapabilityId, span: Span, use_site: &str) {
        self.used_capabilities.insert(*capability);
        if !self
            .requires
            .iter()
            .any(|(declared, _)| declared == capability)
        {
            self.error(
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

    pub fn fresh_symbol(&mut self, _base: &str) -> SymbolId {
        self.counter += 1;
        self.arena.loop_binder().1
    }

    /// Assume `e >= 0` as bounds on every atom with a unit coefficient.
    pub fn assume_nonneg(&mut self, e: IntExpr) {
        for symbol in prove::symbols(&self.arena, e) {
            match prove::linear_in(&mut self.arena, e, symbol) {
                Some((1, rest)) => {
                    let zero = self.arena.int(0);
                    let neg = self.arena.int_sub(zero, rest);
                    self.facts.add_lower(symbol, neg)
                }
                Some((-1, rest)) => self.facts.add_upper(symbol, rest),
                _ => {}
            }
        }
    }

    /// Assume `e == 0`: a zero fact, and both bounds on every atom with a unit coefficient.
    pub fn assume_zero(&mut self, e: IntExpr) {
        self.facts.assume_zero(&self.arena, e);
        self.assume_nonneg(e);
        let zero = self.arena.int(0);
        let neg = self.arena.int_sub(zero, e);
        self.assume_nonneg(neg);
    }

    fn is_shape_param(&self, name: &str) -> bool {
        self.sig.shape_params.iter().any(|p| p == name)
    }

    /// Record that the value of these shape parameters is observed as a number.
    pub fn numeric_use(&mut self, value: IntExpr) {
        for symbol in prove::symbols(&self.arena, value) {
            if let Some(ordinal) = self
                .sig
                .shape_symbols
                .iter()
                .position(|candidate| *candidate == symbol)
            {
                self.summary
                    .numeric
                    .insert(self.sig.shape_params[ordinal].clone());
            }
        }
    }

    /// Prove `e >= 0`. Shape-arithmetic needs become a diagnostic asking for a `where`.
    pub fn require_nonneg(&mut self, e: IntExpr, span: Span, what: &str) {
        if prove::nonneg(&self.arena, &self.facts, e) {
            return;
        }
        let rendered = prove::display(&self.arena, e, &|symbol| {
            self.sig
                .shape_symbols
                .iter()
                .position(|candidate| *candidate == symbol)
                .map(|ordinal| self.sig.shape_params[ordinal].clone())
                .unwrap_or_else(|| format!("{symbol:?}"))
        });
        self.error(span, format!("{what}: cannot prove `{rendered} >= 0`"));
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

    /// Whether a value of type `value` may be installed into state of type
    /// `target` (floats round to the target's element type).
    pub fn assignable(&self, target: &ValueType, value: &ValueType) -> bool {
        match (target, value) {
            (ValueType::Scalar(a), _) => value
                .scalar_dtype()
                .is_some_and(|b| *a == b || (a.is_float() && b.is_float())),
            (ValueType::Tensor(a), ValueType::Tensor(b)) => {
                self.same_axes(a, b) && elem_rounds(&b.elem, &a.elem)
            }
            (ValueType::Tuple(a), ValueType::Tuple(b)) => {
                a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| self.assignable(x, y))
            }
            _ => self.same_ty(target, value),
        }
    }

    // ---- target-dependent forms ----

    /// A target-dependent form. Legal only in a body with a target context.
    pub fn target_form(&mut self, span: Span, what: &str, namespace: Option<&str>) -> bool {
        let Some(target) = self.target.clone() else {
            self.error(span, format!("{what} is target-dependent and cannot appear in a portable body; write it in a `fn … for <target>` or `lower … for <target>` body"));
            return false;
        };
        if namespace
            .and_then(crate::registry::BackendName::parse)
            .is_some_and(|ns| ns != target)
        {
            self.error(
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

    /// The storage root a place or value expression designates, if any.
    pub fn root_var(&self, e: &CheckedExpr) -> Option<LocalId> {
        match &e.kind {
            CheckedExprKind::Local(v) => Some(self.local_storage_root(*v)),
            CheckedExprKind::Primitive { id, operands } => match id {
                PrimitiveId::SliceView { .. } | PrimitiveId::Transpose | PrimitiveId::Reshape => {
                    operands.first().and_then(|b| self.root_var(b))
                }
                _ => None,
            },
            CheckedExprKind::PlaneView { base, .. } => self.root_var(base),
            CheckedExprKind::Atomic { .. } => None,
            _ => None,
        }
    }

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
                    span,
                    format!("`{binding_name}` is a read-only moved-in parameter; writing requires `&mut tensor`"),
                );
                return None;
            }
            LocalKind::Param(_) | LocalKind::State => {}
            LocalKind::Value if self.is_borrowed_local(binding) => {}
            _ => {
                self.error(
                    span,
                    format!("`{binding_name}` is not mutable state; only `let mut` bindings and `&mut tensor` parameters are written"),
                );
                return None;
            }
        }
        if !self.writable_root(binding) || !self.writable_root(root) {
            self.error(
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
                        span,
                        format!(
                            "representation `{}` is decode-only and has no canonical write contract",
                            crate::registry::representation_info(*representation).name
                        ),
                    );
                    return None;
                }
            }
        }
        if self
            .live_borrows()
            .iter()
            .any(|(borrow, borrowed, _)| borrowed.local == root && borrow.local != binding)
        {
            self.error(
                span,
                format!(
                    "cannot mutate `{}` while a tensor borrow is live",
                    self.locals[root.index()].name
                ),
            );
            return None;
        }
        // The source-order region walk checks every ordinary read/write and
        // write/write pair across distinct visits, including effects of calls
        // and views. A single-site index test would both miss collisions and
        // reject safe helper calls whose actual footprint is disjoint.
        self.mutated.push(root);
        self.scalar_symbols.remove(&root);
        let tensor_effect = matches!(self.locals[root.index()].ty, ValueType::Tensor(_));
        let root_exprs = self.dyn_views.clone();
        let _ = root_exprs;
        self.dyn_views.retain(|(start, end, _, _)| {
            let mentions =
                |e: &Option<CheckedExpr>| e.as_ref().is_some_and(|b| expr::mentions_local(b, root));
            !tensor_effect && !(mentions(start) || mentions(end))
        });
        Some(root)
    }

    // ---- result ----

    fn finish(
        mut self,
        root: CheckedBlock,
        span: Span,
    ) -> (CheckedBody, BodySig, Summary, Vec<Diagnostic>, ExprArena) {
        let well_formed = self.diagnostics.is_empty();
        if well_formed
            && self.sig.result != ValueType::Void
            && !matches!(root.terminator, ir::Terminator::Return(_))
        {
            self.error(
                span,
                format!(
                    "`{}` returns {} but not every path ends in `return`",
                    self.sig.name, self.sig.result
                ),
            );
        }
        for (capability, declared_at) in self.requires.clone() {
            if !self.used_capabilities.contains(&capability) {
                self.error(
                    declared_at,
                    format!(
                        "capability `{}.{}` is required but not used directly or through a backend-specific helper",
                        crate::registry::capability_info(capability).backend.as_str(),
                        crate::registry::capability_info(capability).name
                    ),
                );
            }
        }
        let body = CheckedBody {
            locals: self.locals,
            root,
        };
        (body, self.sig, self.summary, self.diagnostics, self.arena)
    }
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
    summary: Summary,
    diagnostics: Vec<Diagnostic>,
    arena: ExprArena,
    signature: BodySig,
    initialization: initialization::Contract,
}

fn check_definition(env: &Env, def: usize) -> CheckedOutcome {
    let declared = &env.resolved.declared[def];
    let mut c = Checker::new(env, def);
    let initialization_facts = c.facts.clone();
    let mut root = c.block(declared.body);
    let initialization = if env.enforce {
        initialization::check(&mut c, &mut root, initialization_facts)
    } else {
        initialization::Contract::empty()
    };
    let (body, signature, summary, diagnostics, arena) = c.finish(root, declared.name_span);
    CheckedOutcome {
        body,
        summary,
        diagnostics,
        arena,
        signature,
        initialization,
    }
}

/// Close numeric and reduction uses over parameters passed along unchanged to callees.
fn close_summaries(summaries: &mut [Summary]) {
    loop {
        let mut changed = false;
        for i in 0..summaries.len() {
            for (callee, callee_param, own) in summaries[i].passes.clone() {
                if summaries[callee].numeric.contains(&callee_param)
                    && summaries[i].numeric.insert(own.clone())
                {
                    changed = true;
                }
                if summaries[callee].reduces.contains(&callee_param)
                    && summaries[i].reduces.insert(own)
                {
                    changed = true;
                }
            }
        }
        if !changed {
            return;
        }
    }
}

/// The checked static call graph must be acyclic; recursion is rejected before
/// specialization because execution families are finite.
fn reject_cycles(definitions: &[ir::Definition], diagnostics: &mut Vec<Located>) {
    let count = definitions.len();
    // 0 = unvisited, 1 = on stack, 2 = done.
    let mut state = vec![0u8; count];
    let mut stack: Vec<usize> = Vec::new();
    fn visit(
        d: usize,
        definitions: &[ir::Definition],
        state: &mut [u8],
        stack: &mut Vec<usize>,
        cycle: &mut Option<usize>,
    ) {
        match state[d] {
            2 => return,
            1 => {
                if cycle.is_none() {
                    *cycle = stack
                        .get(stack.iter().position(|&s| s == d).unwrap_or(0))
                        .copied();
                }
                return;
            }
            _ => {}
        }
        state[d] = 1;
        stack.push(d);
        for callee in definitions[d].body.callees() {
            let c = callee.index();
            if c < definitions.len() {
                visit(c, definitions, state, stack, cycle);
            }
        }
        stack.pop();
        state[d] = 2;
    }
    for d in 0..count {
        let mut cycle = None;
        visit(d, definitions, &mut state, &mut stack, &mut cycle);
        if let Some(cycle_root) = cycle {
            let definition = &definitions[cycle_root];
            diagnostics.push(Located {
                file: definition.file,
                diagnostic: crate::span::Diagnostic::new(
                    definition.span,
                    format!(
                        "`{}` participates in a recursive call chain: the checked call graph must be acyclic and recursion is rejected before specialization",
                        definition.name
                    ),
                ),
            });
        }
    }
}

/// Check every declared body of the closed program. Returns the definitions and families.
pub(crate) fn check_program(
    files: &[(usize, ast::File)],
    program: ProgramId,
    semantic_hash: ModuleHash,
    diagnostics: &mut Vec<Located>,
) -> (Vec<ir::Definition>, Vec<ir::Family>) {
    let resolved = resolve::resolve(files, program, diagnostics);
    let count = resolved.declared.len();

    // Discover type-level calls and numeric/reduction usage. Initialization is
    // checked once, bottom-up, before any definition is published.
    let empty = vec![Summary::default(); count];
    let discovery = Env {
        resolved: &resolved,
        summaries: &empty,
        enforce: false,
        checked: &[],
    };
    let prototypes: Vec<_> = (0..count)
        .map(|def| check_definition(&discovery, def))
        .collect();
    let mut summaries: Vec<_> = prototypes.iter().map(|body| body.summary.clone()).collect();
    close_summaries(&mut summaries);
    fn visit(def: usize, prototypes: &[CheckedOutcome], state: &mut [u8], order: &mut Vec<usize>) {
        if state[def] != 0 {
            return;
        }
        state[def] = 1;
        for callee in prototypes[def].body.callees() {
            visit(callee.index(), prototypes, state, order);
        }
        state[def] = 2;
        order.push(def);
    }
    let mut order = Vec::new();
    let mut state = vec![0; count];
    for def in 0..count {
        visit(def, &prototypes, &mut state, &mut order);
    }
    let mut outcomes: Vec<Option<CheckedOutcome>> = (0..count).map(|_| None).collect();
    for def in order {
        let env = Env {
            resolved: &resolved,
            summaries: &summaries,
            enforce: true,
            checked: &outcomes,
        };
        let checked = check_definition(&env, def);
        outcomes[def] = Some(checked);
    }
    let mut definitions = Vec::with_capacity(count);
    for (def, declared) in resolved.declared.iter().enumerate() {
        let mut checked = outcomes[def]
            .take()
            .expect("definition checking order omitted a body");
        diagnostics.extend(checked.diagnostics.into_iter().map(|diagnostic| Located {
            file: declared.file,
            diagnostic,
        }));
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
        let dimensions = checked.signature.shape_params.iter().enumerate().map(|(ordinal, name)| {
            let symbol = checked.signature.shape_symbols[ordinal];
            let value = checked.arena.int_symbol(symbol);
            let admits_zero = checked.signature.predicates.iter().any(|predicate| {
                matches!(predicate, Predicate::NonNegative(expression) if prove::same(&checked.arena, *expression, value))
            });
            ir::Dimension { name: name.clone(), symbol, admits_zero }
        }).collect();
        definitions.push(ir::Definition {
            stable,
            name: declared.sig.name.clone(),
            kind: declared.kind.clone(),
            requires: declared
                .requires
                .iter()
                .map(|(capability, _)| *capability)
                .collect(),
            family: declared.family,
            dimensions,
            elem_params: declared.sig.elem_params.clone(),
            elem_bindings: declared.elem_bindings.clone(),
            params,
            aliases: checked.signature.aliases,
            result: checked.signature.result,
            predicates: checked.signature.predicates,
            initialization: checked.initialization,
            body: checked.body,
            arena: checked.arena,
            file: declared.file,
            span: declared.span,
        });
    }
    if diagnostics.is_empty() {
        reject_cycles(&definitions, diagnostics);
    }
    (definitions, resolved.families)
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
            Err(diagnostic) => parse_diagnostics.push(SourceDiagnostic {
                path: source.path.clone(),
                span: diagnostic.span,
                message: diagnostic.message,
            }),
        }
    }
    if let Some(diagnostics) = Diagnostics::new(parse_diagnostics) {
        return Err(SourceError::Parse(diagnostics));
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

    let mut located = Vec::new();
    let (definitions, families) = check_program(&parsed, program, semantic_hash, &mut located);
    if !located.is_empty() {
        located.sort_by_key(|item| (item.file, item.diagnostic.span.start));
        located
            .dedup_by(|left, right| left.file == right.file && left.diagnostic == right.diagnostic);
        let diagnostics = located
            .into_iter()
            .map(|item| SourceDiagnostic {
                path: sources.files()[item.file].path.clone(),
                span: item.diagnostic.span,
                message: item.diagnostic.message,
            })
            .collect();
        return Err(SourceError::Type(
            Diagnostics::new(diagnostics).expect("nonempty checker diagnostics disappeared"),
        ));
    }

    let mut entries = Vec::new();
    let mut entry_families = Vec::new();
    let mut entry_diagnostics = Vec::new();
    for (family_ordinal, family) in families.iter().enumerate() {
        let Some(contract) = definitions.get(family.contract.index()) else {
            panic!("checked family contract is outside the checked definition arena");
        };
        if !contract.kind.is_portable_body() {
            continue;
        }
        if let Err(message) = entry_build::validate_external_dimension_inference(contract) {
            entry_diagnostics.push(SourceDiagnostic {
                path: sources.files()[contract.file].path.clone(),
                span: contract.span,
                message,
            });
            continue;
        }
        let ordinal = u32::try_from(entries.len()).expect("module has more than u32::MAX entries");
        let id = crate::ids::EntryId::new(module_id, ordinal);
        let mut stable_hasher = sha2::Sha256::new();
        stable_hasher.update(semantic_hash.digest());
        stable_hasher.update((family_ordinal as u64).to_le_bytes());
        let stable = crate::ids::StableEntryId::new(stable_hasher.finalize().into());
        entries.push(entry_info(id, stable, contract));
        entry_families.push(family_ordinal);
    }
    if let Some(diagnostics) = Diagnostics::new(entry_diagnostics) {
        return Err(SourceError::Type(diagnostics));
    }
    let mut native_implementations = Vec::new();
    let mut native_diagnostics = Vec::new();
    for (file, parsed_file) in &parsed {
        for declaration in &parsed_file.decls {
            let crate::syntax::ast::Decl::Native(native) = declaration else {
                continue;
            };
            let Some(backend) = crate::registry::BackendName::parse(&native.target.name) else {
                native_diagnostics.push(SourceDiagnostic {
                    path: sources.files()[*file].path.clone(),
                    span: native.target.span,
                    message: format!("unknown native backend `{}`", native.target.name),
                });
                continue;
            };
            if backend != crate::registry::BackendName::Metal {
                native_diagnostics.push(SourceDiagnostic {
                    path: sources.files()[*file].path.clone(),
                    span: native.target.span,
                    message: "top-level native implementations currently support only `metal`"
                        .to_owned(),
                });
                continue;
            }
            let matching = entries
                .iter()
                .filter(|entry| entry.name == native.function.name)
                .collect::<Vec<_>>();
            let entry = match matching.as_slice() {
                [entry] => *entry,
                [] => {
                    native_diagnostics.push(SourceDiagnostic {
                        path: sources.files()[*file].path.clone(),
                        span: native.function.span,
                        message: format!(
                            "native implementation refers to unknown portable function `{}`",
                            native.function.name
                        ),
                    });
                    continue;
                }
                _ => {
                    native_diagnostics.push(SourceDiagnostic {
                        path: sources.files()[*file].path.clone(),
                        span: native.function.span,
                        message: format!(
                            "native implementation of overloaded function `{}` is ambiguous",
                            native.function.name
                        ),
                    });
                    continue;
                }
            };
            if native_implementations.iter().any(
                |implementation: &crate::checked::NativeImplementation| {
                    implementation.entry == entry.id && implementation.backend == backend
                },
            ) {
                native_diagnostics.push(SourceDiagnostic {
                    path: sources.files()[*file].path.clone(),
                    span: native.span,
                    message: format!(
                        "function `{}` already has a native implementation for `{}`",
                        native.function.name,
                        backend.as_str()
                    ),
                });
                continue;
            }
            let mut convert = |expression: &crate::syntax::ast::Expr| {
                native_nat_expr(expression, &entry.dimensions).map_err(|message| {
                    native_diagnostics.push(SourceDiagnostic {
                        path: sources.files()[*file].path.clone(),
                        span: expression.span,
                        message,
                    });
                })
            };
            let threadgroups = native.threadgroups.each_ref().map(&mut convert);
            let threads = native.threads_per_threadgroup.each_ref().map(&mut convert);
            let [Ok(x), Ok(y), Ok(z)] = threadgroups else {
                continue;
            };
            let [Ok(tx), Ok(ty), Ok(tz)] = threads else {
                continue;
            };
            native_implementations.push(crate::checked::NativeImplementation {
                entry: entry.id,
                backend,
                declared_in: sources.files()[*file].path.clone(),
                source: native.source.clone(),
                threadgroups: [x, y, z],
                threads_per_threadgroup: [tx, ty, tz],
            });
        }
    }
    if let Some(diagnostics) = Diagnostics::new(native_diagnostics) {
        return Err(SourceError::Type(diagnostics));
    }
    Ok(crate::checked::internals::Module {
        id: module_id,
        template_program: program,
        semantic_hash,
        sources,
        entries,
        native_implementations,
        entry_families,
        definitions,
        families,
    })
}

fn native_nat_expr(
    expression: &crate::syntax::ast::Expr,
    dimensions: &[String],
) -> Result<crate::checked::NativeNatExpr, String> {
    use crate::checked::NativeNatExpr as N;
    use crate::syntax::ast::{BinaryOp, ExprKind};
    let binary = |left: &crate::syntax::ast::Expr,
                  right: &crate::syntax::ast::Expr,
                  make: fn(Box<N>, Box<N>) -> N| {
        Ok(make(
            Box::new(native_nat_expr(left, dimensions)?),
            Box::new(native_nat_expr(right, dimensions)?),
        ))
    };
    match &expression.kind {
        ExprKind::Int(value) => Ok(N::Constant(*value)),
        ExprKind::Name(name) if dimensions.contains(&name.name) => {
            Ok(N::Dimension(name.name.clone()))
        }
        ExprKind::Name(name) => Err(format!(
            "native launch expression references unknown dimension `{}`",
            name.name
        )),
        ExprKind::Binary { op, lhs, rhs } => match op {
            BinaryOp::Add => binary(lhs, rhs, N::Add),
            BinaryOp::Sub => binary(lhs, rhs, N::Sub),
            BinaryOp::Mul => binary(lhs, rhs, N::Mul),
            BinaryOp::Div => binary(lhs, rhs, N::Div),
            BinaryOp::Rem => binary(lhs, rhs, N::Rem),
            _ => Err("native launch expressions use only `+`, `-`, `*`, `/`, and `%`".to_owned()),
        },
        ExprKind::Call {
            callee,
            bindings,
            args,
        } if bindings.is_empty()
            && matches!(&callee.kind, ExprKind::Name(name) if name.name == "ceil_div")
            && args.len() == 2
            && args.iter().all(|arg| arg.name.is_none()) =>
        {
            binary(&args[0].value, &args[1].value, N::CeilDiv)
        }
        _ => Err("unsupported native launch expression".to_owned()),
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
    }
}
