//! Instantiation state: the flat execution variable table, the frame of one selected
//! candidate body, slice geometry, region-result storage and execution-IR builders.
use super::walk;
use crate::exec::ir::{self, Expr, ExprKind, Stmt, StmtKind};
use crate::exec::types::{Shaped, Ty};
use crate::family::{
    Candidate, CandidateRef, Family, SiteId, SiteKind, Template, UnitKind, Witness,
};
use crate::sir;
use crate::span::Span;
use crate::sym::{Atom, Sym};
use crate::syntax::ast::{AssignOp, BinaryOp};
use crate::types as st;
use crate::types::{DType, Elem};
use std::collections::{BTreeMap, BTreeSet};

/// Current geometry of one bound slice: `[lo, hi)` of its parent domain for this visit.
#[derive(Clone, Debug)]
pub(super) struct Slice {
    pub lo: Sym,
    pub hi: Sym,
    /// Selected width; equals `hi - lo` because only dividing widths are instantiated.
    pub capacity: i64,
    /// Piece ordinal of this visit within its traversal.
    pub ordinal: Sym,
}

/// Partition of one producer binder, kept by a region result for rebinding.
#[derive(Clone, Debug)]
pub(super) struct Binder {
    pub lo: Sym,
    pub width: i64,
    pub count: i64,
    pub site: Option<SiteId>,
}

/// Backing of one yielded member schema. Stores carry one leading piece axis per binder
/// of every enclosing producer.
#[derive(Clone, Debug)]
pub(super) enum Member {
    Scalar(Expr),
    Tile(Expr),
    Tuple(Vec<Member>),
    Result(Box<ResultValue>),
}

#[derive(Clone, Debug)]
pub(super) struct ResultValue {
    pub binders: Vec<Binder>,
    /// Piece index atoms of the producing traversal; nested geometry may mention them.
    pub atoms: Vec<Atom>,
    pub member: Member,
}

#[derive(Clone, Debug)]
pub(super) enum Value<'a> {
    Scalar(Expr),
    /// Tensor, view, tile or native fragment reference.
    Shaped(Expr),
    /// A folded single-consumer tile binding, computed by its consumer.
    Deferred(&'a sir::Expr),
    Slice(Slice),
    Tuple(Vec<Value<'a>>),
    Result(ResultValue),
    Void,
}

/// Values of a `yield`/`return` boundary that become locals of the enclosing owner.
#[derive(Debug, Default)]
pub(super) struct Slots<'a> {
    pub conditional: u32,
    pub loops: u32,
    pub direct: Option<Vec<Value<'a>>>,
    pub locals: Option<Vec<Value<'a>>>,
    pub prologue: Vec<Stmt>,
}

#[derive(Debug)]
pub(super) enum YieldSink<'a> {
    Slots(Slots<'a>),
    Result {
        member: Member,
        pieces: Vec<Expr>,
        loops: u32,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum FusedKind {
    Elementwise,
    Region,
}

/// A selected multi-unit interval, as statement ordinals of its authored block.
#[derive(Clone, Debug)]
pub(super) struct Fused {
    pub first: usize,
    pub last: usize,
    pub kind: FusedKind,
}

pub(super) struct Frame<'a> {
    pub candidate: CandidateRef,
    pub definition: &'a sir::Definition,
    pub body: &'a sir::Body,
    pub shapes: BTreeMap<String, i64>,
    pub elems: BTreeMap<String, Elem>,
    pub structural: BTreeMap<String, Slice>,
    /// Shape parameters bound to a runtime-valued extent of the caller.
    pub dynamic: BTreeMap<String, Sym>,
    pub vars: Vec<Option<Value<'a>>>,
    pub slices: BTreeMap<st::SliceId, Slice>,
    pub uses: Vec<usize>,
    pub folded: BTreeSet<sir::VarId>,
    pub fused: BTreeMap<usize, Vec<Fused>>,
    /// Per-coordinate scalar values of bindings inside the fused loop being emitted.
    pub overrides: BTreeMap<sir::VarId, Expr>,
    pub yields: Vec<YieldSink<'a>>,
    pub returns: Slots<'a>,
    pub conditional: u32,
    pub loops: u32,
    /// The statement being instantiated is a root statement of the entry (a launch scope).
    pub root: bool,
}

impl<'a> Frame<'a> {
    pub fn var(&self, id: sir::VarId) -> Result<&Value<'a>, String> {
        self.vars.get(id).and_then(Option::as_ref).ok_or_else(|| {
            format!(
                "`{}` in `{}` is used before instantiation bound it",
                self.name(id),
                self.definition.name
            )
        })
    }

    pub fn name(&self, id: sir::VarId) -> &str {
        self.body.vars.get(id).map_or("?", |v| v.name.as_str())
    }

    pub fn declared(&self, id: sir::VarId) -> Result<&'a sir::Var, String> {
        self.body.vars.get(id).ok_or_else(|| {
            format!(
                "variable {id} is outside the body of `{}`",
                self.definition.name
            )
        })
    }

    pub fn slice(&self, id: st::SliceId) -> Result<&Slice, String> {
        self.slices.get(&id).ok_or_else(|| {
            format!(
                "slice#{} of `{}` is not bound in this visit",
                id.0, self.definition.name
            )
        })
    }

    pub fn fused_at(&self, block: usize, index: usize) -> Option<&Fused> {
        self.fused.get(&block)?.iter().find(|f| f.first == index)
    }
}

pub(super) struct Instantiation<'a> {
    pub program: &'a sir::Program,
    pub family: &'a Family,
    pub witness: &'a Witness,
    pub vars: Vec<ir::Var>,
    pub counter: usize,
    /// Execution variables whose storage a later statement may mutate.
    pub mutable: BTreeSet<ir::VarId>,
    /// Fresh tiles not yet owned by a binding.
    pub temporaries: BTreeSet<ir::VarId>,
    pub live_sites: BTreeMap<SiteId, Slice>,
    /// Runtime-bounded windows by (start, end, parent extent): equal windows share the atom
    /// naming their extent.
    pub windows: Vec<(Option<Expr>, Option<Expr>, Sym, Atom)>,
    pub depth: usize,
    /// The root `parallel` launch whose owner body is being instantiated, while no element
    /// loop, branch or inner owner region encloses the current statement.
    pub launch_owner: Option<LaunchOwner>,
}

/// The owner scope of a root `parallel` launch: where an inner owner region may appear.
#[derive(Clone, Debug)]
pub(super) struct LaunchOwner {
    pub candidate: CandidateRef,
    pub binders: Vec<st::SliceId>,
}

pub(super) fn block_key(block: &sir::Block) -> usize {
    block as *const sir::Block as usize
}

impl<'a> Instantiation<'a> {
    pub fn new(program: &'a sir::Program, family: &'a Family, witness: &'a Witness) -> Self {
        Instantiation {
            program,
            family,
            witness,
            vars: Vec::new(),
            counter: 0,
            mutable: BTreeSet::new(),
            temporaries: BTreeSet::new(),
            live_sites: BTreeMap::new(),
            windows: Vec::new(),
            depth: 0,
            launch_owner: None,
        }
    }

    pub fn candidate(&self, r: CandidateRef) -> Result<&'a Candidate, String> {
        self.family
            .occurrences
            .get(r.occurrence.0 as usize)
            .and_then(|o| o.candidates.get(r.candidate as usize))
            .ok_or_else(|| format!("witness selects candidate {} of occurrence {}, which the family does not contain", r.candidate, r.occurrence.0))
    }

    pub fn template(&self, candidate: &Candidate) -> Result<&'a Template, String> {
        self.family
            .templates
            .get(candidate.template.0 as usize)
            .ok_or_else(|| format!("template {} is outside the family", candidate.template.0))
    }

    pub fn definition(&self, id: sir::DefId) -> Result<&'a sir::Definition, String> {
        self.program
            .definitions
            .get(id.0 as usize)
            .ok_or_else(|| format!("definition {} is outside the program", id.0))
    }

    /// The frame of a selected candidate: template bindings, use counts, folded bindings
    /// and the selected multi-unit intervals of its blocks.
    pub fn frame(&self, r: CandidateRef) -> Result<Frame<'a>, String> {
        let candidate = self.candidate(r)?;
        let template = self.template(candidate)?;
        let definition = self.definition(template.definition)?;
        let body = &definition.body;
        let mut structural = BTreeMap::new();
        for (name, site) in &candidate.structural {
            let slice = self.live_sites.get(&site.0).ok_or_else(|| {
                format!("structural shape `{name}` of `{}` is bound to site {}, which has no live slice", definition.name, site.0 .0)
            })?;
            structural.insert(name.clone(), slice.clone());
        }
        let uses = walk::uses(body);
        let mut blocks = Vec::new();
        let mut ranges = Vec::new();
        for id in &candidate.sequences {
            let sequence = self
                .family
                .sequences
                .get(id.0 as usize)
                .ok_or_else(|| format!("sequence {} is outside the family", id.0))?;
            let block = walk::scope(body, &sequence.scope)
                .map_err(|e| format!("sequence {} of `{}`: {e}", id.0, definition.name))?;
            let mut next = 0;
            for unit in &sequence.units {
                let stage = matches!(unit.kind, UnitKind::Stage(_)) && unit.statements.end == next;
                if !stage && (unit.statements.start != next || unit.statements.end <= next) {
                    return Err(format!(
                        "sequence {} of `{}` does not tile its block contiguously",
                        id.0, definition.name
                    ));
                }
                next = unit.statements.end;
            }
            if next != block.len() {
                return Err(format!(
                    "sequence {} of `{}` covers {next} statements but the block at {:?} has {}",
                    id.0,
                    definition.name,
                    sequence.scope,
                    block.len()
                ));
            }
            blocks.push(block);
            ranges.push(
                sequence
                    .units
                    .iter()
                    .map(|u| u.statements.clone())
                    .collect::<Vec<_>>(),
            );
        }
        let pairs: Vec<(&sir::Block, &[std::ops::Range<usize>])> = blocks
            .iter()
            .copied()
            .zip(ranges.iter().map(Vec::as_slice))
            .collect();
        let folded =
            walk::folded(body, &uses, &pairs).map_err(|e| format!("`{}`: {e}", definition.name))?;
        let mut frame = Frame {
            candidate: r,
            definition,
            body,
            shapes: template.shapes.clone(),
            elems: template.elems.clone(),
            structural,
            dynamic: BTreeMap::new(),
            vars: vec![None; body.vars.len()],
            slices: BTreeMap::new(),
            uses,
            folded,
            fused: BTreeMap::new(),
            overrides: BTreeMap::new(),
            yields: Vec::new(),
            returns: Slots::default(),
            conditional: 0,
            loops: 0,
            root: false,
        };
        self.fused_intervals(candidate, &mut frame)?;
        Ok(frame)
    }

    fn fused_intervals(&self, candidate: &Candidate, frame: &mut Frame<'a>) -> Result<(), String> {
        for id in &candidate.sequences {
            let sequence = self
                .family
                .sequences
                .get(id.0 as usize)
                .ok_or_else(|| format!("sequence {} is outside the family", id.0))?;
            let cover = self.witness.covers.get(id).ok_or_else(|| {
                format!(
                    "witness has no cover for active sequence {} of `{}`",
                    id.0, frame.definition.name
                )
            })?;
            if cover
                .iter()
                .all(|(start, end)| end.saturating_sub(*start) <= 1)
            {
                continue;
            }
            let block = walk::scope(frame.body, &sequence.scope)
                .map_err(|e| format!("sequence {} of `{}`: {e}", id.0, frame.definition.name))?;
            for &(start, end) in cover {
                if end <= start + 1 {
                    continue;
                }
                let units = sequence
                    .units
                    .get(start as usize..end as usize)
                    .ok_or_else(|| format!("cover of sequence {} leaves its units", id.0))?;
                let kind = if units.iter().all(|u| u.kind == UnitKind::Elementwise) {
                    FusedKind::Elementwise
                } else if units.iter().all(|u| matches!(u.kind, UnitKind::Region(_))) {
                    FusedKind::Region
                } else {
                    return Err(format!(
                        "sequence {} of `{}`: interval [{start}, {end}) groups units {:?}, for which instantiation has no fused realization",
                        id.0,
                        frame.definition.name,
                        units.iter().map(|u| &u.kind).collect::<Vec<_>>()
                    ));
                };
                let first = units[0].statements.start;
                let last = units[units.len() - 1].statements.end - 1;
                frame
                    .fused
                    .entry(block_key(block))
                    .or_default()
                    .push(Fused { first, last, kind });
            }
        }
        Ok(())
    }

    /// The numerical site of a binder owned by this frame's candidate, with its value.
    pub fn site(
        &self,
        f: &Frame<'a>,
        region: st::RegionId,
        slice: st::SliceId,
    ) -> Result<(SiteId, bool, i64), String> {
        let candidate = self.candidate(f.candidate)?;
        for id in &candidate.sites {
            let site = self
                .family
                .sites
                .get(id.0 as usize)
                .ok_or_else(|| format!("site {} is outside the family", id.0))?;
            let parts = match &site.kind {
                SiteKind::Width {
                    region: r,
                    slice: s,
                } if *r == region && *s == slice => false,
                SiteKind::Parts {
                    region: r,
                    slice: s,
                } if *r == region && *s == slice => true,
                _ => continue,
            };
            let value = *self.witness.sites.get(id).ok_or_else(|| {
                format!(
                    "witness has no value for active site {} of `{}`",
                    id.0, f.definition.name
                )
            })?;
            if value <= 0 {
                return Err(format!(
                    "site {} of `{}` has the non-positive value {value}",
                    id.0, f.definition.name
                ));
            }
            return Ok((*id, parts, value));
        }
        Err(format!(
            "region#{} binder slice#{} of `{}` has no numerical site in the family",
            region.0, slice.0, f.definition.name
        ))
    }

    // ---- execution variables ----

    pub fn fresh(&mut self) -> usize {
        self.counter += 1;
        self.counter
    }

    pub fn local(&mut self, name: &str, ty: Ty, span: Span) -> Expr {
        let id = self.vars.len();
        let n = self.fresh();
        self.vars.push(ir::Var {
            name: format!("{name}_{n}"),
            ty: ty.clone(),
            span,
            kind: ir::VarKind::Local,
        });
        Expr {
            kind: ExprKind::Var(id),
            ty,
            sym: None,
            span,
        }
    }

    /// A loop or work-item index with its own atom.
    pub fn index(&mut self, name: &str, span: Span) -> (ir::VarId, Atom, Expr) {
        let id = self.vars.len();
        let n = self.fresh();
        let atom = Atom::Param(format!("{name}#{n}"));
        let ty = Ty::Scalar(DType::I32);
        self.vars.push(ir::Var {
            name: format!("{name}_{n}"),
            ty: ty.clone(),
            span,
            kind: ir::VarKind::Index(atom.clone()),
        });
        (
            id,
            atom.clone(),
            Expr {
                kind: ExprKind::Var(id),
                ty,
                sym: Some(Sym::atom(atom)),
                span,
            },
        )
    }

    /// Symbolic value of an integer expression; a data-dependent one is bound to an index
    /// local so that ranges and slice bounds can name it.
    pub fn symbolic(&mut self, e: Expr, out: &mut Vec<Stmt>) -> Result<Sym, String> {
        if let Some(sym) = &e.sym {
            return Ok(sym.clone());
        }
        if !matches!(e.ty, Ty::Scalar(d) if d.is_int()) {
            return Err(format!("a {} value is used as an integer coordinate", e.ty));
        }
        let (_, atom, target) = self.index("bound", e.span);
        out.push(assign(target, coerce(e, DType::I32)));
        Ok(Sym::atom(atom))
    }

    // ---- types ----

    pub fn elem(&self, f: &Frame<'a>, elem: &Elem) -> Result<Elem, String> {
        match elem {
            Elem::Param(p) => f.elems.get(p).cloned().ok_or_else(|| {
                format!(
                    "element parameter `{p}` of `{}` is unbound in its template",
                    f.definition.name
                )
            }),
            other => Ok(other.clone()),
        }
    }

    pub fn dtype(&self, f: &Frame<'a>, ty: &st::Ty) -> Result<DType, String> {
        match ty {
            st::Ty::Scalar(d) => Ok(*d),
            st::Ty::Index(_) | st::Ty::Coord(_) => Ok(DType::I32),
            other => Err(format!(
                "`{other}` in `{}` is not a scalar type",
                f.definition.name
            )),
        }
    }

    pub fn extent(&self, f: &Frame<'a>, extent: &st::Extent) -> Result<Sym, String> {
        match extent {
            st::Extent::Semantic(sym) => self.resolve(f, sym),
            st::Extent::Structural(slice) => match f.slices.get(slice) {
                Some(bound) => Ok(Sym::constant(bound.capacity)),
                None => Ok(Sym::constant(self.static_geometry(f, *slice)?.0)),
            },
        }
    }

    pub fn shaped(&self, f: &Frame<'a>, shaped: &st::Shaped) -> Result<Shaped, String> {
        let shape = shaped
            .axes
            .iter()
            .map(|a| self.extent(f, a))
            .collect::<Result<Vec<_>, _>>()?;
        let elem = self.elem(f, &shaped.elem)?;
        let packed_axis = match (&elem, shaped.packed_axis) {
            (Elem::Repr(_), None) => Some(shape.len().saturating_sub(1)),
            (Elem::Repr(_), axis) => axis,
            _ => None,
        };
        Ok(Shaped {
            shape,
            elem,
            packed_axis,
        })
    }

    /// Selected width and piece count of a binder that is not bound in the current visit
    /// (an inner producer whose result storage an outer producer allocates).
    pub fn static_geometry(&self, f: &Frame<'a>, slice: st::SliceId) -> Result<(i64, i64), String> {
        let decl = f.body.slices.get(slice.0 as usize).ok_or_else(|| {
            format!(
                "slice#{} is outside the body of `{}`",
                slice.0, f.definition.name
            )
        })?;
        let extent = match &decl.parent {
            sir::SliceParent::Domain { lo, hi } => self.resolve(f, &hi.sub(lo))?.as_constant().ok_or_else(|| {
                format!("the extent of slice#{} in `{}` is not static where its storage is allocated", slice.0, f.definition.name)
            })?,
            sir::SliceParent::Refine(parent) => match f.slices.get(parent) {
                Some(bound) => bound.capacity,
                None => self.static_geometry(f, *parent)?.0,
            },
            sir::SliceParent::Rebind(origin) => return self.static_geometry(f, *origin),
        };
        let (_, parts, value) = self.site(f, decl.region, slice)?;
        let width = width_of(extent, parts, value)
            .map_err(|e| format!("slice#{} of `{}`: {e}", slice.0, f.definition.name))?;
        Ok((width, if width == 0 { 0 } else { extent / width }))
    }

    // ---- symbols ----

    /// Substitute shape parameters by the template's numbers, structural shape parameters
    /// by their capacity, and index variables by their execution atoms.
    pub fn resolve(&self, f: &Frame<'a>, sym: &Sym) -> Result<Sym, String> {
        substitute(sym, &|atom| match atom {
            Atom::Param(name) => self.parameter(f, name).map(Some),
            _ => Ok(None),
        })
    }

    fn parameter(&self, f: &Frame<'a>, name: &str) -> Result<Sym, String> {
        if let Some(value) = f.shapes.get(name) {
            return Ok(Sym::constant(*value));
        }
        if let Some(slice) = f.structural.get(name) {
            return Ok(Sym::constant(slice.capacity));
        }
        if let Some(extent) = f.dynamic.get(name) {
            return Ok(extent.clone());
        }
        let (base, ordinal) = match name.split_once('#') {
            Some((base, ordinal)) => (base, ordinal.parse::<usize>().ok()),
            None => (name, None),
        };
        let bound = |id: usize| match f.vars.get(id) {
            Some(Some(Value::Scalar(e))) if f.body.vars[id].name == base => e.sym.clone(),
            _ => None,
        };
        if let Some(sym) = ordinal.and_then(bound) {
            return Ok(sym);
        }
        (0..f.vars.len()).rev().find_map(bound).ok_or_else(|| {
            format!("symbolic extent in `{}` mentions `{name}`, which is neither a shape parameter nor a bound index", f.definition.name)
        })
    }
}

/// Selected piece width over a static extent. A `Parts` site partitions into near-equal
/// contiguous parts; only the dividing case has uniform tiles.
pub(super) fn width_of(extent: i64, parts: bool, value: i64) -> Result<i64, String> {
    if extent == 0 {
        return Ok(if parts { 0 } else { value });
    }
    if extent % value != 0 {
        return Err(format!(
            "tail pieces of extent {extent} under the selected {} {value} are not supported by instantiation yet",
            if parts { "partition count" } else { "width" }
        ));
    }
    Ok(if parts { extent / value } else { value })
}

/// Simultaneous substitution through quotient and remainder atoms.
pub(super) fn substitute(
    sym: &Sym,
    value: &dyn Fn(&Atom) -> Result<Option<Sym>, String>,
) -> Result<Sym, String> {
    let mut out = Sym::constant(0);
    for (monomial, coefficient) in sym.monomials() {
        let mut term = Sym::constant(coefficient);
        for (atom, power) in monomial {
            let factor = match value(atom)? {
                Some(v) => v,
                None => match atom {
                    Atom::Param(_) => Sym::atom(atom.clone()),
                    Atom::Quot(n, d) => substitute(n, value)?.quot(&substitute(d, value)?),
                    Atom::Rem(n, d) => substitute(n, value)?.rem(&substitute(d, value)?),
                },
            };
            for _ in 0..*power {
                term = term.mul(&factor);
            }
        }
        out = out.add(&term);
    }
    Ok(out)
}

// ---- execution-IR builders ----

pub(super) fn stmt(kind: StmtKind, span: Span) -> Stmt {
    Stmt {
        id: None,
        kind,
        span,
    }
}

pub(super) fn assign(target: Expr, value: Expr) -> Stmt {
    assign_op(target, AssignOp::Assign, value)
}

pub(super) fn assign_op(target: Expr, op: AssignOp, value: Expr) -> Stmt {
    let span = value.span;
    stmt(StmtKind::Assign { target, op, value }, span)
}

pub(super) fn int(n: i64, span: Span) -> Expr {
    Expr {
        kind: ExprKind::Int(n),
        ty: Ty::Scalar(DType::I32),
        sym: Some(Sym::constant(n)),
        span,
    }
}

/// An `i32` expression denoting exactly `sym`.
pub(super) fn symbol(sym: Sym, span: Span) -> Expr {
    match sym.as_constant() {
        Some(n) => int(n, span),
        None => Expr {
            kind: ExprKind::ShapeParam(sym.to_string()),
            ty: Ty::Scalar(DType::I32),
            sym: Some(sym),
            span,
        },
    }
}

pub(super) fn literal(dtype: DType, value: f64, span: Span) -> Expr {
    let kind = match dtype {
        DType::Bool => ExprKind::Bool(value != 0.0),
        d if d.is_int() => ExprKind::Int(value as i64),
        _ => ExprKind::Float(value),
    };
    let sym = match kind {
        ExprKind::Int(n) => Some(Sym::constant(n)),
        _ => None,
    };
    Expr {
        kind,
        ty: Ty::Scalar(dtype),
        sym,
        span,
    }
}

/// Conversion to `dtype` where assignment would not already convert.
pub(super) fn coerce(e: Expr, dtype: DType) -> Expr {
    match e.ty {
        Ty::Scalar(d) if d == dtype => e,
        _ => {
            let span = e.span;
            let sym = if dtype.is_int() { e.sym.clone() } else { None };
            Expr {
                kind: ExprKind::Cast {
                    dtype,
                    expr: Box::new(e),
                },
                ty: Ty::Scalar(dtype),
                sym,
                span,
            }
        }
    }
}

/// Assignment converts between float dtypes; every other scalar mismatch needs a cast.
pub(super) fn conform(e: Expr, dtype: DType) -> Expr {
    match e.ty {
        Ty::Scalar(d) if d == dtype || (d.is_float() && dtype.is_float()) => e,
        _ => coerce(e, dtype),
    }
}

pub(super) fn binary(op: BinaryOp, lhs: Expr, rhs: Expr, ty: Ty) -> Expr {
    let integer = matches!(ty, Ty::Scalar(d) if d.is_int());
    let sym = match (&lhs.sym, &rhs.sym) {
        (Some(a), Some(b)) if integer => match op {
            BinaryOp::Add => Some(a.add(b)),
            BinaryOp::Sub => Some(a.sub(b)),
            BinaryOp::Mul => Some(a.mul(b)),
            BinaryOp::Div if b.as_constant().is_some_and(|n| n > 0) => Some(a.quot(b)),
            BinaryOp::Rem if b.as_constant().is_some_and(|n| n > 0) => Some(a.rem(b)),
            _ => None,
        },
        _ => None,
    };
    let span = lhs.span;
    Expr {
        kind: ExprKind::Binary {
            op,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        },
        ty,
        sym,
        span,
    }
}

/// Indexing with the execution IR's typing: points drop their axis, slices keep it with
/// extent `end - start`, missing trailing indices keep whole axes.
pub(super) fn index(base: Expr, indices: Vec<ir::Index>, span: Span) -> Result<Expr, String> {
    index_with(base, indices, &[], span)
}

/// `extents[axis]` names the extent of a slice whose bounds are data-dependent.
pub(super) fn index_with(
    base: Expr,
    indices: Vec<ir::Index>,
    extents: &[Option<Sym>],
    span: Span,
) -> Result<Expr, String> {
    if indices.is_empty() {
        return Ok(base);
    }
    // A point prefix of a point prefix addresses the same storage directly.
    if let ExprKind::Index {
        base: inner,
        indices: prefix,
    } = &base.kind
    {
        if prefix.iter().all(|i| matches!(i, ir::Index::Point(_))) {
            let mut all = prefix.clone();
            let mut named = vec![None; prefix.len()];
            named.extend(extents.iter().cloned());
            all.extend(indices);
            return index_with((**inner).clone(), all, &named, span);
        }
    }
    let (source, tensor) = match &base.ty {
        Ty::Tensor(s) => (s.clone(), true),
        Ty::Tile(s) => (s.clone(), false),
        other => return Err(format!("cannot index a {other}")),
    };
    if indices.len() > source.shape.len() {
        return Err(format!(
            "{} indices for rank {}",
            indices.len(),
            source.shape.len()
        ));
    }
    let mut shape = Vec::new();
    let point = |axis: usize| matches!(indices.get(axis), Some(ir::Index::Point(_)));
    let packed_axis = source
        .packed_axis
        .filter(|p| !point(*p))
        .map(|p| p - (0..p).filter(|a| point(*a)).count());
    for (axis, idx) in indices.iter().enumerate() {
        match idx {
            ir::Index::Point(_) => {}
            ir::Index::Slice { start, end } => {
                if let Some(Some(extent)) = extents.get(axis) {
                    shape.push(extent.clone());
                    continue;
                }
                let lo = match start {
                    Some(e) => e.sym.clone().ok_or("slice start has no symbolic value")?,
                    None => Sym::constant(0),
                };
                let hi = match end {
                    Some(e) => e.sym.clone().ok_or("slice end has no symbolic value")?,
                    None => source.shape[axis].clone(),
                };
                shape.push(hi.sub(&lo));
            }
        }
    }
    shape.extend(source.shape[indices.len()..].iter().cloned());
    let ty = if shape.is_empty() {
        Ty::Scalar(
            source
                .elem
                .read_dtype()
                .ok_or("element read of an unbound element type")?,
        )
    } else if tensor {
        Ty::Tensor(Shaped {
            shape,
            elem: source.elem.clone(),
            packed_axis,
        })
    } else {
        Ty::Tile(Shaped {
            shape,
            elem: source.elem.clone(),
            packed_axis,
        })
    };
    Ok(Expr {
        kind: ExprKind::Index {
            base: Box::new(base),
            indices,
        },
        ty,
        sym: None,
        span,
    })
}

pub(super) fn points(base: Expr, coordinates: &[Expr]) -> Result<Expr, String> {
    let span = base.span;
    index(
        base,
        coordinates.iter().cloned().map(ir::Index::Point).collect(),
        span,
    )
}

pub(super) fn tile_shape(e: &Expr) -> Result<&Shaped, String> {
    e.ty.shaped()
        .ok_or_else(|| format!("a {} value is used as a tile", e.ty))
}

pub(super) fn root(e: &Expr) -> Option<ir::VarId> {
    match &e.kind {
        ExprKind::Var(v) => Some(*v),
        ExprKind::Index { base, .. }
        | ExprKind::Transpose(base)
        | ExprKind::Accessor { base, .. } => root(base),
        ExprKind::Builtin {
            name: ir::Builtin::Reshape,
            args,
        } => args.first().and_then(root),
        _ => None,
    }
}
