//! The checked semantic program.
//!
//! The checked representation mirrors only current source: `let`/`let mut`,
//! assignment, ordered/independent loops, `if`, `return`, expressions built
//! from registry primitives and static function-family calls.
//!
//! One `Definition` is a template. A call names a contract (family + argument
//! binding); which definition implements an occurrence is a selection decision
//! made over `family`, never here. The checked static call graph is acyclic;
//! recursion is rejected before specialization.

use crate::intrinsics::{IntrinsicId, PrimitiveId};
use crate::span::Span;
use crate::sym::Sym;
use crate::syntax::ast::AssignOp;
use crate::types::{Elem, ExtentExpr, ValueType};
use std::collections::HashSet;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DefId(pub u32);

/// Index into `CheckedBody::locals`.
pub type LocalId = usize;

/// Index of one call occurrence inside a definition body, stable for
/// diagnostics; call data itself lives in the `CheckedExprKind::Call` node.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CallId(pub u32);

/// Compiler-only parameter passing mode (signatures only).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Mode {
    In,
    Inout,
}

/// Logical call ownership of a parameter, kept beside the mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParamOwnership {
    /// A plain value (scalar, index, range, tuple, capability value).
    Value,
    /// An owned tensor that moves into the callee.
    Owned,
    /// A shared borrow (`&tensor`).
    Shared,
    /// An exclusive mutable borrow (`&mut tensor`).
    Exclusive,
}

// ---------------------------------------------------------------------------
// Program and definitions
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct Program {
    pub definitions: Vec<Definition>,
    pub families: Vec<ContractFamily>,
    /// Source files, for diagnostics: `(path, text)`.
    pub files: Vec<(String, String)>,
}

impl Program {
    /// Stable identity of the checked source set for numerical qualification records.
    pub fn identity(&self) -> [u8; 32] {
        use sha2::{Digest, Sha256};
        let mut hash = Sha256::new();
        for (path, text) in &self.files {
            hash.update((path.len() as u64).to_le_bytes());
            hash.update(path.as_bytes());
            hash.update((text.len() as u64).to_le_bytes());
            hash.update(text.as_bytes());
        }
        hash.finalize().into()
    }

    pub fn definition(&self, id: DefId) -> &Definition {
        &self.definitions[id.0 as usize]
    }

    pub fn family_of(&self, id: DefId) -> &ContractFamily {
        &self.families[self.definition(id).family]
    }

    /// Resolve a name-only root request. Calls already carry an exact family
    /// index, but an embedding request has no argument-type selector and
    /// therefore must fail closed when several disjoint overload families
    /// share the name.
    pub fn family_index(&self, name: &str) -> Result<usize, String> {
        let mut matches = self
            .families
            .iter()
            .enumerate()
            .filter(|(_, family)| family.name == name)
            .map(|(index, _)| index);
        let Some(first) = matches.next() else {
            return Err(format!("no linked function `{name}`"));
        };
        if matches.next().is_some() {
            return Err(format!(
                "ambiguous linked function `{name}`: a name-only root request matches multiple disjoint signatures"
            ));
        }
        Ok(first)
    }

    pub fn resolve_family(&self, name: &str) -> Result<&ContractFamily, String> {
        Ok(&self.families[self.family_index(name)?])
    }

    pub fn family(&self, name: &str) -> Option<&ContractFamily> {
        self.resolve_family(name).ok()
    }
}

/// A connected component of same-name implementations with overlapping
/// applicability and a compatible contract.
#[derive(Clone, Debug)]
pub struct ContractFamily {
    pub name: String,
    /// Canonical source contract for stable parameter names, ordering and
    /// generic bindings.
    pub contract: DefId,
    /// Portable and backend-specific `fn` bodies.
    pub bodies: Vec<DefId>,
    /// Backend `lower` bodies.
    pub lowerings: Vec<DefId>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DefKind {
    /// A `fn` body. `None` is portable; `Some` restricts it to one backend.
    Body { target: Option<String> },
    /// `lower … for target:` with a body.
    Lower { target: String },
}

impl DefKind {
    pub fn target(&self) -> Option<&str> {
        match self {
            DefKind::Body { target } => target.as_deref(),
            DefKind::Lower { target } => Some(target),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Definition {
    pub id: DefId,
    pub name: String,
    pub kind: DefKind,
    /// Capability namespaces explicitly declared by this source body.
    pub requires: Vec<crate::intrinsics::CapabilityId>,
    /// Exact typed capability signatures used directly by this body.
    pub intrinsic_uses: Vec<IntrinsicUse>,
    /// Index into `Program::families`.
    pub family: usize,
    pub shape_params: Vec<String>,
    pub elem_params: Vec<String>,
    /// Concrete element types this definition fixes where the contract has a parameter.
    pub elem_bindings: Vec<(String, Elem)>,
    pub params: Vec<Param>,
    /// Parameter ordinal pairs permitted to alias.
    pub aliases: Vec<(usize, usize)>,
    pub result: ValueType,
    /// Applicability: every predicate must hold.
    pub predicates: Vec<Predicate>,
    pub body: CheckedBody,
    /// Index into `Program::files`.
    pub file: usize,
    pub span: Span,
}

/// One exact typed capability use recorded for fingerprinting and planning.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IntrinsicUse {
    pub id: IntrinsicId,
    pub arguments: Vec<ValueType>,
    pub result: ValueType,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Param {
    pub name: String,
    pub mode: Mode,
    pub ownership: ParamOwnership,
    pub ty: ValueType,
    pub local: LocalId,
}

/// A decidable applicability predicate over shape parameters.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Predicate {
    /// `expr >= 0`
    NonNegative(Sym),
    /// `expr == 0` (equalities and divisibility `N % c == 0`)
    Zero(Sym),
    /// `expr != 0`
    NonZero(Sym),
}

// ---------------------------------------------------------------------------
// Checked bodies
// ---------------------------------------------------------------------------

/// A typed local of a checked body. Parameters occupy `0..params.len()` in
/// parameter order.
#[derive(Clone, Debug, PartialEq)]
pub struct CheckedLocal {
    pub name: String,
    pub ty: ValueType,
    pub mutable: bool,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CheckedBody {
    pub locals: Vec<CheckedLocal>,
    pub root: CheckedBlock,
}

impl CheckedBody {
    /// Every call occurrence in this body, in evaluation order.
    pub fn calls(&self) -> Vec<&CheckedCall> {
        let mut out = Vec::new();
        walk_block(&self.root, &mut |expr: &CheckedExpr| {
            if let CheckedExprKind::Call { call, .. } = &expr.kind {
                out.push(call.as_ref());
            }
        });
        out
    }

    /// Every static callee definition reachable from this body.
    pub fn callees(&self) -> Vec<crate::sir::DefId> {
        let mut seen = HashSet::new();
        let mut out = Vec::new();
        for call in self.calls() {
            for binding in &call.bindings {
                if seen.insert(binding.definition) {
                    out.push(binding.definition);
                }
            }
        }
        out
    }
}

fn walk_block<'a>(block: &'a CheckedBlock, visit: &mut dyn FnMut(&'a CheckedExpr)) {
    for statement in &block.statements {
        walk_stmt(statement, visit);
    }
    if let BlockTerminator::Return(values) = &block.terminator {
        values.iter().for_each(|v| walk_expr(v, visit));
    }
}

fn walk_stmt<'a>(statement: &'a CheckedStmt, visit: &mut dyn FnMut(&'a CheckedExpr)) {
    match statement {
        CheckedStmt::Let { value, .. } => walk_expr(value, visit),
        CheckedStmt::Assign { value, .. } => walk_expr(value, visit),
        CheckedStmt::Loop { range, body, .. } => {
            walk_expr(&range.start, visit);
            walk_expr(&range.end, visit);
            walk_block(body, visit);
        }
        CheckedStmt::If {
            condition,
            then_body,
            else_body,
        } => {
            walk_expr(condition, visit);
            walk_block(then_body, visit);
            walk_block(else_body, visit);
        }
        CheckedStmt::Evaluate(expr) => walk_expr(expr, visit),
    }
}

fn walk_expr<'a>(expr: &'a CheckedExpr, visit: &mut dyn FnMut(&'a CheckedExpr)) {
    visit(expr);
    match &expr.kind {
        CheckedExprKind::Primitive { operands, .. } => {
            operands.iter().for_each(|o| walk_expr(o, visit))
        }
        CheckedExprKind::Capability { args, .. } => args.iter().for_each(|a| walk_expr(a, visit)),
        CheckedExprKind::Call { args, .. } => args.iter().for_each(|a| walk_expr(a, visit)),
        CheckedExprKind::Literal(_) | CheckedExprKind::Local(_) => {}
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CheckedBlock {
    pub statements: Vec<CheckedStmt>,
    pub terminator: BlockTerminator,
}

#[derive(Clone, Debug, PartialEq)]
pub enum BlockTerminator {
    /// Control continues with the enclosing construct.
    Continue,
    /// The function boundary's result values.
    Return(Vec<CheckedExpr>),
}

#[derive(Clone, Debug, PartialEq)]
pub enum CheckedStmt {
    Let {
        pattern: Pattern,
        mutable: bool,
        value: CheckedExpr,
    },
    Assign {
        place: CheckedPlace,
        op: AssignOp,
        value: CheckedExpr,
    },
    Loop {
        kind: LoopKind,
        binder: LocalId,
        range: CheckedRange,
        body: CheckedBlock,
        mutation: LoopMutationSummary,
    },
    If {
        condition: CheckedExpr,
        then_body: CheckedBlock,
        else_body: CheckedBlock,
    },
    Evaluate(CheckedExpr),
}

/// Source-level loop semantics, independent of any physical execution width.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LoopKind {
    /// Ascending `for`; captured mutable values are carried across visits.
    Ordered,
    /// Independent `parallel for`; no scalar/owned carry, only proved-disjoint
    /// or atomic writes.
    Independent,
}

/// Half-open iteration range of a loop, in ascending coordinate order.
#[derive(Clone, Debug, PartialEq)]
pub struct CheckedRange {
    pub start: CheckedExpr,
    pub end: CheckedExpr,
}

/// What a loop body mutates, summarized once by the checker for logical
/// construction: ordered loops carry every changed captured value; independent
/// loops admit only proved-disjoint element writes and atomics.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LoopMutationSummary {
    /// Mutable locals captured from enclosing scopes and written by the body
    /// (ordered-loop carry inputs).
    pub carried: Vec<LocalId>,
    /// Every captured write went through an index that depends on the loop
    /// binder (proved disjoint across independent visits).
    pub disjoint_writes: bool,
    /// Storage roots updated atomically inside the loop.
    pub atomics: Vec<LocalId>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Pattern {
    Local(LocalId),
    Tuple(Vec<Pattern>),
}

/// A mutable place: a local's storage, an element/selection of it, or a tuple
/// of places (tuple assignment).
#[derive(Clone, Debug, PartialEq)]
pub enum CheckedPlace {
    Local {
        root: LocalId,
    },
    Element {
        root: LocalId,
        indices: Vec<CheckedIndex>,
    },
    Tuple(Vec<CheckedPlace>),
}

#[derive(Clone, Debug, PartialEq)]
pub enum CheckedIndex {
    Point(CheckedExpr),
    /// `lo:hi`; `None` bounds are the axis ends.
    Range {
        start: Option<CheckedExpr>,
        end: Option<CheckedExpr>,
    },
}

// ---------------------------------------------------------------------------
// Checked expressions
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub struct CheckedExpr {
    pub kind: CheckedExprKind,
    pub ty: ValueType,
    /// Symbolic value of integer expressions over shape parameters and
    /// indices, when the checker proved one.
    pub sym: Option<Sym>,
    pub span: Span,
}

impl CheckedExpr {
    pub fn new(kind: CheckedExprKind, ty: ValueType, sym: Option<Sym>, span: Span) -> CheckedExpr {
        CheckedExpr {
            kind,
            ty,
            sym,
            span,
        }
    }
}

/// A checked expression contains only registry primitives and static
/// function-family calls — never physical tiles, participants, launches,
/// structural slices, native fragments, or memory spaces.
#[derive(Clone, Debug, PartialEq)]
pub enum CheckedExprKind {
    Literal(Literal),
    Local(LocalId),
    Primitive {
        id: PrimitiveId,
        operands: Vec<CheckedExpr>,
    },
    Capability {
        id: IntrinsicId,
        args: Vec<CheckedExpr>,
    },
    Call {
        call: Box<CheckedCall>,
        args: Vec<CheckedExpr>,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub enum Literal {
    Int(i64),
    Float(f64),
    Bool(bool),
    /// A shape parameter used as a value.
    ShapeParam(String),
}

/// One static call occurrence: the family it names and, per candidate
/// definition that unifies with the arguments, how its parameters bind.
#[derive(Clone, Debug, PartialEq)]
pub struct CheckedCall {
    /// Index into `Program::families`.
    pub family: usize,
    /// Candidates whose unification succeeded, in definition order.
    /// Predicates are *not* evaluated here.
    pub bindings: Vec<CandidateBinding>,
    pub span: Span,
}

/// How one candidate definition's parameters bind at a call occurrence.
#[derive(Clone, Debug, PartialEq)]
pub struct CandidateBinding {
    pub definition: DefId,
    /// Callee shape parameter name -> bound symbolic extent.
    pub shape_args: Vec<(String, Sym)>,
    pub elem_args: Vec<(String, Elem)>,
    /// Argument expression ordinal for each callee parameter (named arguments resolved).
    pub arg_order: Vec<usize>,
    /// Element parameters of the CALLER that must equal these concrete element
    /// types for this candidate to apply.
    pub requires_elems: Vec<(String, Elem)>,
}

pub fn sym_extent(sym: Sym) -> ExtentExpr {
    match sym.as_constant() {
        Some(c) if c >= 0 => ExtentExpr::Static(c as u64),
        _ => ExtentExpr::Sym(sym),
    }
}
